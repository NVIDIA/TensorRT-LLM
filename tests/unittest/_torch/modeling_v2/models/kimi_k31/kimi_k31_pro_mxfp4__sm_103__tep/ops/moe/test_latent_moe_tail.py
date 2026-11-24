# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 latent_moe_tail entries on one GB300 tray (TP 4).

Each cell runs on four MPI ranks: every rank builds the process-wide fused tail of the group,
regenerates every rank's routed and shared partials from the cell's seed, calls the entry on its
own partials and gates the result against the entry's reference over all of them. The ranks report
their deviation and a bit hash of their output, so the parent also checks that every rank got the
same tensor; a second call (the workspace's next Lamport buffer) and a CUDA-graph replay must give
the same bits. Folded cells hand over the routed experts' output before the top-k sum; pushed cells
store the shared rows ahead (latent_moe_tail_push_shared on a side stream) and must give the bits of
a call that pushes them itself. The mailbox cells run latent_moe_tail_to_mailbox and consume the
mailbox with latent_moe_tail_copy_out: the bits of the three-kernel call, the rows the mailbox
received, re-armed afterwards.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import pickle
import sys
import types
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.latent_moe_tail import (  # noqa: E501
    EPS,
    HIDDEN,
    LATENT,
    KimiK3LatentMoETail,
    UnfinalizedRoutedOutput,
    k31_latent_moe_tail,
    k31_latent_moe_tail_copy_out,
    k31_latent_moe_tail_push_shared,
    k31_latent_moe_tail_to_mailbox,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__extra_import_path__ = ["../../../../.."]
from _validating import ULP, validating  # noqa: E402 — needs the path declared above

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not IS_CUTLASS_DSL_AVAILABLE
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3),)
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs",
    ),
]

# Gate in bf16 ulps against the row's largest magnitude (element) and the relative RMS. Against
# the reference the kernels differ in the latent norm's rsqrt / reduction order (one BF16 step in
# ~0.7% of the latent elements) and in the split-K up projection's FP32 accumulation order; each of
# the two output roundings (up projection, sum with the shared shard) can then land one step away:
# measured on GB300 up to 2.12 ulp element / 0.4 ulp RMS, ~10% of the elements differing.
ELEMENT_ULP, RMS_ULP = 3.0, 0.5


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, world = spec["rows"], spec["world"]

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    routed = [r(rows, LATENT) for _ in range(world)]
    shared = [r(rows, HIDDEN, scale=0.25) for _ in range(world)]
    rms_weight = (1 + 0.1 * torch.randn(LATENT, generator=gen, device="cuda")).to(torch.bfloat16)
    up_weight = r(HIDDEN, LATENT, scale=LATENT**-0.5)
    return routed, shared, rms_weight, up_weight


def _unfinalized(rows: int, gen: torch.Generator) -> UnfinalizedRoutedOutput:
    """A rank's routed experts before the top-k sum: 1..8 of each token's 16 slots on this rank, their FC2 rows in a
    random permuted order (with spare rows), BF16 routing weights."""
    local = torch.randint(1, 9, (rows,), generator=gen, device="cuda")
    perm_rows = int(local.sum())
    y = torch.randn(perm_rows + 16, LATENT, generator=gen, device="cuda").to(torch.bfloat16)
    order = torch.randperm(perm_rows, generator=gen, device="cuda").to(torch.int32)
    slots = torch.full((rows, 16), -1, dtype=torch.int32, device="cuda")
    start = 0
    for t, n in enumerate(local.tolist()):
        owned = torch.randperm(16, generator=gen, device="cuda")[:n]
        slots[t, owned] = order[start : start + n]
        start += n
    weights = (torch.rand(rows, 16, generator=gen, device="cuda") * 0.25).to(torch.bfloat16)
    return UnfinalizedRoutedOutput(y, weights, slots)


def _partials(spec: dict, seed: int):
    """Every rank's routed and shared partials and the weights, the same on every rank; with ``fold`` the routed
    partials are the routed experts' unfinalized outputs."""
    if not spec.get("fold"):
        return _inputs(spec, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    routed = [_unfinalized(spec["rows"], gen) for _ in range(spec["world"])]
    _, shared, rms_weight, up_weight = _inputs(spec, seed + 1)
    return routed, shared, rms_weight, up_weight


def _tail(spec: dict, rank: int):
    """The group's process-wide tail of the cell (built with ``fold_finalize`` for folded cells), kept across cells."""
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=spec["world"], tp_size=spec["world"], rank=rank)
    tails = vars(sys.modules.setdefault("_k31_tails", types.ModuleType("_k31_tails"))).setdefault(
        "by_world", {}
    )
    fold = bool(spec.get("fold"))
    key = (spec["world"], "fold") if fold else spec["world"]
    if key not in tails:
        tails[key] = KimiK3LatentMoETail.get(
            mapping,
            hidden_size=HIDDEN,
            latent_size=LATENT,
            rms_eps=spec["eps"],
            fold_finalize=fold,
        )
    return tails[key]


def _push_ahead(shared: torch.Tensor, tail, side: torch.cuda.Stream) -> None:
    """``k31_latent_moe_tail_push_shared`` on a side stream forked from and joined back into the caller's, as the
    target's shared-expert stream does."""
    fork = torch.cuda.Event()
    fork.record()
    side.wait_event(fork)
    with torch.cuda.stream(side):
        k31_latent_moe_tail_push_shared(shared, tail)
    torch.cuda.current_stream().wait_stream(side)


def _deviation(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    o, r = out.float(), ref.float()
    row = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    element = ((o - r).abs() / row).max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item()
    return element, rms / ULP[torch.bfloat16]


def _digest(t: torch.Tensor) -> int:
    bits = t.contiguous().view(torch.int16).to(torch.int64).flatten()
    weights = torch.arange(1, bits.numel() + 1, device=t.device, dtype=torch.int64) % 65521
    return int((bits * weights).sum().item())


def _rank_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    tail = _tail(spec, rank)
    routed, shared, rms_weight, up_weight = _partials(spec, seed)
    args = (routed[rank], shared[rank], rms_weight, up_weight, tail)
    pushed = bool(spec.get("shared_pushed"))
    side = torch.cuda.Stream()

    def call():
        if pushed:
            _push_ahead(shared[rank], tail, side)
        return k31_latent_moe_tail(*args, shared_pushed=pushed)

    with torch.inference_mode():
        k31_latent_moe_tail.is_valid(*args, shared_pushed=pushed)
        if pushed:
            k31_latent_moe_tail_push_shared.is_valid(shared[rank], tail)
        out = call().clone()
        again = call().clone()
        # CUDA-graph replay of the same call (the Lamport buffers keep rotating inside the graph).
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graphed = call()
        graph.replay()
        graph.replay()
        replayed = graphed.clone()
        # A pushed call has the bits of the call that pushes its shared rows itself.
        unpushed = k31_latent_moe_tail(*args).clone() if pushed else out
        ref = k31_latent_moe_tail.reference(routed, shared, rms_weight, up_weight, spec["eps"])
        torch.cuda.synchronize()
        element, rms = _deviation(out, ref)
        result = dict(
            rank=rank,
            shape=tuple(out.shape),
            element_ulp=element,
            rms_ulp=rms,
            hash=_digest(out),
            repeat_equal=bool(torch.equal(out, again)),
            graph_equal=bool(torch.equal(out, replayed)),
            pushed_equal=bool(torch.equal(out.view(torch.int16), unpushed.view(torch.int16))),
            differ_frac=float((out.view(torch.int16) != ref.view(torch.int16)).float().mean()),
        )
    return result


def _one_per_rank(results: list) -> list:
    """The pool's results sorted by rank, one per rank. The pool may hand one worker two tasks; a rank that ran
    none would leave the others' collectives waiting, so a case must see every rank exactly once."""
    ranks = sorted(r["rank"] for r in results)
    assert ranks == list(range(WORLD)), f"the pool ran ranks {ranks}, not one task per rank"
    return sorted(results, key=lambda r: r["rank"])


def _run(executor, spec: dict, seed: int) -> list:
    return _one_per_rank(list(executor.map(_rank_cell, *zip(*[(spec, seed)] * WORLD))))


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", k31_latent_moe_tail.CELLS, ids=[c.why[:48] for c in k31_latent_moe_tail.CELLS]
)
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    assert spec["world"] == WORLD
    results = _run(mpi_pool_executor, spec, zlib.crc32(cell.why.encode()) % 2**31)
    worst = max(r["element_ulp"] for r in results), max(r["rms_ulp"] for r in results)
    print(
        f"{cell.why}: element {worst[0]:.3f} ulp, rms {worst[1]:.3f} ulp, "
        f"{results[0]['differ_frac']:.4%} of the elements differ from the reference"
    )
    for r in results:
        assert r["shape"] == (spec["rows"], HIDDEN), r
        assert r["repeat_equal"] and r["graph_equal"], f"rank {r['rank']}: calls disagree"
        assert r["pushed_equal"], f"rank {r['rank']}: the pushed shared rows changed the output"
        assert r["element_ulp"] <= ELEMENT_ULP and r["rms_ulp"] <= RMS_ULP, r
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"


def _rank_mailbox_cell(spec: dict, seed: int) -> dict:
    import tensorrt_llm
    from tensorrt_llm._utils import mpi_barrier

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    tail = _tail(spec, rank)
    routed, shared, rms_weight, up_weight = _partials(spec, seed)
    args = (routed[rank], shared[rank], rms_weight, up_weight, tail)
    rows = spec["rows"]
    pushed = bool(spec.get("shared_pushed"))
    side = torch.cuda.Stream()

    def to_mailbox():
        if pushed:
            _push_ahead(shared[rank], tail, side)
        return k31_latent_moe_tail_to_mailbox(*args, shared_pushed=pushed)

    with torch.inference_mode():
        k31_latent_moe_tail_to_mailbox.is_valid(*args, shared_pushed=pushed)
        whole = k31_latent_moe_tail(*args).clone()
        mailbox = to_mailbox()
        k31_latent_moe_tail_copy_out.is_valid(mailbox, rows, tail)
        # Every rank's shard is in this rank's mailbox once every rank's launches completed.
        torch.cuda.synchronize()
        mpi_barrier()
        received = k31_latent_moe_tail_copy_out.reference(mailbox, rows, tail)
        out = k31_latent_moe_tail_copy_out(mailbox, rows, tail).clone()
        torch.cuda.synchronize()
        rearmed = bool((mailbox[0, :rows].view(torch.int16) == -0x8000).all())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graphed = k31_latent_moe_tail_copy_out(to_mailbox(), rows, tail)
        graph.replay()
        graph.replay()
        replayed = graphed.clone()
        ref = k31_latent_moe_tail_to_mailbox.reference(
            routed, shared, rms_weight, up_weight, spec["eps"]
        )
        torch.cuda.synchronize()
        element, rms = _deviation(out, ref)
        result = dict(
            rank=rank,
            shape=tuple(out.shape),
            element_ulp=element,
            rms_ulp=rms,
            hash=_digest(out),
            whole_equal=bool(torch.equal(out.view(torch.int16), whole.view(torch.int16))),
            received_equal=bool(torch.equal(out.view(torch.int16), received.view(torch.int16))),
            rearmed=rearmed,
            graph_equal=bool(torch.equal(out, replayed)),
        )
    return result


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell",
    k31_latent_moe_tail_to_mailbox.CELLS,
    ids=[c.why[:48] for c in k31_latent_moe_tail_to_mailbox.CELLS],
)
def test_mailbox_cells(cell, mpi_pool_executor) -> None:
    """The mailbox entry, its rows consumed by the copy-out entry (whose cells are these)."""
    spec = cell.spec
    assert spec["world"] == WORLD
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = _one_per_rank(
        list(mpi_pool_executor.map(_rank_mailbox_cell, *zip(*[(spec, seed)] * WORLD)))
    )
    for r in results:
        assert r["shape"] == (spec["rows"], HIDDEN), r
        assert r["whole_equal"], f"rank {r['rank']}: mailbox rows differ from the three-kernel call"
        assert r["received_equal"], f"rank {r['rank']}: copy-out differs from the rows received"
        assert r["rearmed"], f"rank {r['rank']}: the copied rows were not re-armed"
        assert r["graph_equal"], f"rank {r['rank']}: graph replay differs"
        assert r["element_ulp"] <= ELEMENT_ULP and r["rms_ulp"] <= RMS_ULP, r
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"


def test_copy_out_and_push_cells_are_the_calls_driven_above() -> None:
    """The copy-out entry consumes every mailbox the mailbox cells fill and the push entry runs in every pushed cell,
    so their cells are those cells (driven by test_mailbox_cells and test_certified_cells)."""
    assert k31_latent_moe_tail_copy_out.CELLS == k31_latent_moe_tail_to_mailbox.CELLS
    pushed = [
        c
        for c in (*k31_latent_moe_tail.CELLS, *k31_latent_moe_tail_to_mailbox.CELLS)
        if c.spec.get("shared_pushed")
    ]
    assert pushed and list(k31_latent_moe_tail_push_shared.CELLS) == pushed


def test_reference_folds_like_the_finalize() -> None:
    """An unfinalized partial counts as its top-k weighted sum over the owned slots."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    routed = _unfinalized(5, gen)
    y, weights, slots = routed
    finalized = torch.zeros(5, LATENT, device="cuda")
    for t in range(5):
        for s in range(16):
            if slots[t, s] >= 0:
                finalized[t] += weights[t, s].float() * y[slots[t, s]].float()
    _, shared, rms_weight, up_weight = _inputs(dict(world=1, rows=5), seed=4)
    folded = k31_latent_moe_tail.reference([routed], shared, rms_weight, up_weight, EPS)
    plain = k31_latent_moe_tail.reference(
        [finalized.to(torch.bfloat16)], shared, rms_weight, up_weight, EPS
    )
    assert torch.equal(folded, plain)


def _rank_fold(rows: int, seed: int) -> dict:
    """One rank of a tail built with ``fold_finalize``: the routed experts' unfinalized rows (random expert rows,
    a random subset of each token's 16 slots on this rank) folded by the collective vs the same rows finalized
    by the CuTe DSL finalize kernel (the routed experts' own finalize) and passed in as the routed partial."""
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe.moe_finalize import (  # noqa: E501
        run_moe_finalize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe.moe_tail_latent import (  # noqa: E501
        KimiK3LatentMoETail,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe.moe_tail_routed_output import (  # noqa: E501
        UnfinalizedRoutedOutput,
    )
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank)
    tail = KimiK3LatentMoETail.get(
        mapping, hidden_size=HIDDEN, latent_size=LATENT, rms_eps=1e-5, fold_finalize=True
    )
    gen = torch.Generator(device="cuda").manual_seed(seed * 31 + rank)
    local = torch.randint(1, 9, (rows,), generator=gen, device="cuda")
    perm_rows = int(local.sum())
    y = torch.randn(perm_rows + 16, LATENT, generator=gen, device="cuda").to(torch.bfloat16)
    order = torch.randperm(perm_rows, generator=gen, device="cuda").to(torch.int32)
    e2p = torch.full((rows, 16), -1, dtype=torch.int32, device="cuda")
    start = 0
    for t, n in enumerate(local.tolist()):
        slots = torch.randperm(16, generator=gen, device="cuda")[:n]
        e2p[t, slots] = order[start : start + n]
        start += n
    weights = (torch.rand(rows, 16, generator=gen, device="cuda") * 0.25).to(torch.bfloat16)
    _, shared, rms_weight, up_weight = _inputs(dict(rows=rows, world=WORLD), seed)
    shared = shared[rank]
    with torch.inference_mode():
        finalized = torch.empty(rows, LATENT, dtype=torch.bfloat16, device="cuda")
        run_moe_finalize(y, e2p.view(-1), weights, finalized)
        unfinalized = UnfinalizedRoutedOutput(y, weights, e2p)
        ref = tail(finalized, shared, rms_weight, up_weight).clone()
        out = tail(unfinalized, shared, rms_weight, up_weight).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graphed = tail(unfinalized, shared, rms_weight, up_weight)
        graph.replay()
        replayed = graphed.clone()
        torch.cuda.synchronize()
    return dict(
        rank=rank,
        equal=bool(torch.equal(out.view(torch.int16), ref.view(torch.int16))),
        graph_equal=bool(torch.equal(out, replayed)),
        hash=_digest(out),
    )


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("rows", [6, 24, 72, 120, 192])
def test_fold_finalize_is_bitwise_to_finalized_rows(rows, mpi_pool_executor) -> None:
    """The tail's top-k fold (one-shot at <= 32 rows on 4 ranks, two-shot above) gives the bits of the CuTe DSL
    finalize kernel's rows: the attention-TP perf target folds the sum at every row count the tail serves."""
    results = _one_per_rank(list(mpi_pool_executor.map(_rank_fold, *zip(*[(rows, rows)] * WORLD))))
    for r in results:
        assert r["equal"], f"rank {r['rank']}: folded sum differs from the finalized rows"
        assert r["graph_equal"], f"rank {r['rank']}: graph replay differs"
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"


def test_gate_refuses_a_dropped_rank() -> None:
    spec = dict(world=WORLD, rows=8, eps=1e-5)
    routed, shared, rms_weight, up_weight = _inputs(spec, seed=1)
    right = k31_latent_moe_tail.reference(routed, shared, rms_weight, up_weight, 1e-5)
    wrong = k31_latent_moe_tail.reference(routed, shared[:-1], rms_weight, up_weight, 1e-5)
    element, rms = _deviation(wrong, right)
    assert element > ELEMENT_ULP and rms > RMS_ULP


def test_guard_refuses_a_transposed_up_weight() -> None:
    spec = dict(world=WORLD, rows=8, eps=1e-5)
    routed, shared, rms_weight, up_weight = _inputs(spec, seed=2)
    tail = types.SimpleNamespace(can_run=lambda m: True, max_num_tokens=128)
    with pytest.raises(AssertionError, match="up_weight"):
        with validating(k31_latent_moe_tail):
            k31_latent_moe_tail(routed[0], shared[0], rms_weight, up_weight.t(), tail)


def _rank_shared_pushed(spec: dict, seed: int) -> dict:
    """The call whose shared rows were pushed ahead (``push_shared`` on a side stream, as the target's shared-expert
    stream does) against the call that pushes them itself: the same bits, eager and in a graph."""
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.moe.moe_tail_latent import (  # noqa: E501
        KimiK3LatentMoETail,
    )
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=spec["world"], tp_size=spec["world"], rank=rank)
    tails = vars(sys.modules.setdefault("_k31_tails", types.ModuleType("_k31_tails"))).setdefault(
        "by_world", {}
    )
    if spec["world"] not in tails:
        tails[spec["world"]] = KimiK3LatentMoETail.get(
            mapping, hidden_size=HIDDEN, latent_size=LATENT, rms_eps=spec["eps"]
        )
    tail = tails[spec["world"]]
    routed, shared, rms_weight, up_weight = _inputs(spec, seed)
    mine = (routed[rank], shared[rank], rms_weight, up_weight)
    early = tail.pushes_shared_early(spec["rows"])
    side = torch.cuda.Stream()

    def pushed():
        fork = torch.cuda.Event()
        fork.record()
        side.wait_event(fork)
        with torch.cuda.stream(side):
            tail.push_shared(shared[rank])
        torch.cuda.current_stream().wait_stream(side)
        return tail(*mine, shared_pushed=True)

    with torch.inference_mode():
        out = tail(*mine).clone()
        if not early:
            # The one-shot collective needs every rank: wait for it, so this worker takes no other rank's task
            # while the collective still runs.
            torch.cuda.synchronize()
            return dict(rank=rank, early=False)
        first = pushed().clone()
        # Interleaved with calls that push for themselves (the rotation is shared).
        again = tail(*mine).clone()
        second = pushed().clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graphed = pushed()
            plain = tail(*mine)
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        same = [
            torch.equal(t.view(torch.int16), out.view(torch.int16))
            for t in (first, again, second, graphed, plain)
        ]
    return dict(rank=rank, early=True, same=same, hash=_digest(first))


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("rows", [24, 72, 128])
def test_shared_pushed_early_is_bitwise(rows, mpi_pool_executor) -> None:
    """Two-shot collective calls (TP 4: above 32 rows); one-shot calls push their shared rows themselves."""
    spec = dict(world=WORLD, rows=rows, eps=EPS)
    results = _one_per_rank(
        list(mpi_pool_executor.map(_rank_shared_pushed, *zip(*[(spec, 11 + rows)] * WORLD)))
    )
    assert all(r["early"] == (rows > 32) for r in results), results
    if rows > 32:
        assert all(all(r["same"]) for r in results), results
        assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"
