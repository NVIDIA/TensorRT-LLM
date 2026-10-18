# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU test for the musespark_open post_attn_hop entry.

The entry claims to be the built-in pair it replaces, bit for bit: the MNNVL
one-shot all-reduce of the o_proj partial (``AllReduce`` with strategy MNNVL)
followed by the attention-side cluster boundary (``cluster_boundary`` with no
gain). Every check below runs both on the same inputs and compares the two
outputs (FP32 residual, BF16 prenorm) as bits, on every transport variant
(multicast and unicast publication, and multicast with the early dependent-launch
trigger), each also with the stale-only polling schedule (``stale_poll``,
which the model's workspace applies from ``STALE_POLL_MIN_ROWS``
rows; ``mc_stale_all`` applies it to every row count). The cells are additionally
gated against the entry's reference within a band the tests state.

It is a collective test: every rank of a 4-rank (one node) or 8-rank (two
nodes, one NVLink domain) group runs this same file and every test is a
collective in the same order on every rank. Two ways to start it:

* under MPI, one process per rank, e.g. for TP8 on two GB300 trays

      srun -N2 --segment=2 --ntasks-per-node=4 --mpi=pmix \\
          python3 -m pytest -q -p no:cacheprovider --noconftest \\
          -o addopts="-p test_common.magic_import_hooks" <this file>

* from one pytest process on 4 GPUs: ``test_modeling_v2_post_attn_hop_rank_job.py``
  beside this file starts 4 ranks over ``torch.distributed`` (``TLLM_DISABLE_MPI=1``)
  with ``MODELING_V2_POST_ATTN_HOP_RANK_JOB=1``, each running this file under a deadline.

Outside either it is skipped. ``MODELING_V2_TIMING=1`` also runs the
microbenchmark (printed by rank 0, never gated).
"""

import os
import statistics
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce.post_attn_hop import (  # noqa: E501
    HIDDEN,
    MAX_ROWS,
    STALE_POLL_MIN_ROWS,
    PostAttnHopWorkspace,
    cluster_for_rows,
    hop_layout_ok,
    hop_supported,
    hop_takes,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce.post_attn_hop import (  # noqa: E501
    post_attn_hop as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

POST_EPS = 1e-8  # the checkpoint's post_norm_eps
PRE_EPS = 1e-5  # the checkpoint's text rms_norm_eps
#: Set by the 4-rank launcher beside this file: the ranks rendezvous over torch.distributed.
RANK_JOB_ENV = "MODELING_V2_POST_ATTN_HOP_RANK_JOB"
TRANSPORTS = ("mc", "uc", "mc_early", "mc_stale", "uc_stale", "mc_early_stale", "mc_stale_all")


class _TorchGroup:
    """The four MPI-communicator calls this file makes, over torch.distributed's CPU backend."""

    def __init__(self):
        import torch.distributed as dist

        self._dist = dist

    def Get_size(self):
        return self._dist.get_world_size()

    def Get_rank(self):
        return self._dist.get_rank()

    def Barrier(self):
        self._dist.barrier()

    def allreduce(self, value, op="sum"):
        t = torch.tensor([value], dtype=torch.float64)
        reduce = self._dist.ReduceOp.MAX if op == "max" else self._dist.ReduceOp.SUM
        self._dist.all_reduce(t, op=reduce)
        return type(value)(t.item())


def _group():
    if os.environ.get(RANK_JOB_ENV) == "1":
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        if not dist.is_initialized():
            dist.init_process_group(
                "cuda:nccl,cpu:gloo",
                init_method="env://",
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )
        comm = _TorchGroup()
        return comm, comm.Get_size(), comm.Get_rank()
    try:
        from tensorrt_llm._utils import mpi_comm

        comm = mpi_comm()
        return comm, comm.Get_size(), comm.Get_rank()
    except Exception:  # noqa: BLE001 - no MPI launch means no group
        return None, 1, 0


COMM, WORLD, RANK = _group()

pytestmark = pytest.mark.skipif(
    WORLD not in (4, 8)
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="collective test: run under a 4- or 8-rank launch on sm_103 or sm_107",
)


def _max_over_ranks(value: float) -> float:
    if isinstance(COMM, _TorchGroup):
        return COMM.allreduce(value, op="max")
    from mpi4py import MPI

    return COMM.allreduce(value, op=MPI.MAX)


def _coefficients(gate: torch.Tensor):
    """``MuseSparkOpenGatedResidual._compute_coefficients`` on an FP32 gate."""
    scaled = gate.to(torch.float32) / 0.3
    beta = torch.sigmoid(scaled)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-scaled) * (1.0 + beta), min=1e-3))
    return alpha, beta


def _cancelling(rows: int, g: torch.Generator):
    """Partials whose FP32 sum depends on the order the ranks are added in.

    Per element: +B and -B on two random ranks, ``B * 2**-26 * (1 + u)`` on a
    third (absorbed by B if added while B is in the accumulator), noise of
    order ``B * 2**-20`` on the rest (rounded against B).
    """
    big = torch.exp2(torch.randint(0, 8, (rows, HIDDEN), generator=g).float())
    vals = torch.randn(WORLD, rows, HIDDEN, generator=g) * big * 2.0**-20
    perm = torch.argsort(torch.rand(WORLD, rows, HIDDEN, generator=g), dim=0)
    vals.scatter_(0, perm[0:1], big.unsqueeze(0))
    vals.scatter_(0, perm[1:2], -big.unsqueeze(0))
    small = big * 2.0**-26 * (1.0 + torch.rand(rows, HIDDEN, generator=g))
    vals.scatter_(0, perm[2:3], small.unsqueeze(0))
    return [v.to(torch.bfloat16) for v in vals]


def _sum_bf16(partials, order) -> torch.Tensor:
    """The FP32 sum from +0 over ``order``, rounded once to BF16 (on the CPU)."""
    acc = torch.zeros(partials[0].shape, dtype=torch.float32)
    for r in order:
        acc = acc + partials[r].cpu().float()
    return acc.to(torch.bfloat16)


def _inputs(rows: int, seed: int, zeros: bool = False, cancel: bool = False):
    """Every rank's partial (rank order) and the shared residual/coefficients/weight.

    Each rank regenerates all of them from the seed, so the reference needs no
    second collective. Partials carry per-channel outliers (a shared
    log-normal channel scale), the residual is FP32 off the BF16 grid.
    """
    g = torch.Generator().manual_seed(seed)
    channel = torch.exp(0.75 * torch.randn(HIDDEN, generator=g))
    partials = [
        (0.5 * torch.randn(rows, HIDDEN, generator=g) * channel).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    if zeros:
        for r, part in enumerate(partials):
            pick = torch.rand(rows, HIDDEN, generator=g)
            part[pick < 0.08] = 0.0
            part[(pick >= 0.08) & (pick < 0.16)] = -0.0
        # Columns where every rank sends -0 and columns that cancel exactly.
        for part in partials:
            part[:, 0:64] = -0.0
        for r in range(0, WORLD, 2):
            partials[r + 1][:, 64:128] = -partials[r][:, 64:128]
    if cancel:
        partials = _cancelling(rows, g)
    residual = (
        4.0
        * torch.randn(rows, HIDDEN, generator=g)
        * torch.exp(0.5 * torch.randn(HIDDEN, generator=g))
    )
    alpha, beta = _coefficients(1.5 * torch.randn(HIDDEN, generator=g))
    weight = (0.2 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
    dev = torch.device("cuda", torch.cuda.current_device())
    return (
        [p.to(dev) for p in partials],
        residual.to(dev),
        alpha.to(dev),
        beta.to(dev),
        weight.to(dev),
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_bits(a) != _bits(b)).sum())


def _agree(ok: bool) -> bool:
    """True only if every rank says so (keeps ranks on the same control path)."""
    return COMM.allreduce(int(ok)) == WORLD


def _counter(ws: PostAttnHopWorkspace) -> int:
    """The workspace's arrival counter as an unsigned 32-bit value."""
    torch.cuda.synchronize()
    return int(ws.control[1].item()) & 0xFFFFFFFF


class _BuiltinMnnvlAllReduce:
    """``AllReduce(strategy=MNNVL)``'s plain all-reduce, for the ``torch.distributed`` launch.

    The built-in module also allocates the custom all-reduce's IPC workspace for
    its fallback strategies, and outside MPI that exchange goes through the
    Ray-backed communicator, which the image does not carry. A plain BF16
    all-reduce never reaches a fallback: ``AllReduce.forward`` makes the input
    contiguous and returns ``MNNVLAllReduce``'s one-shot result. This is that
    branch, with the same module and workspace, and nothing else; under MPI the
    tests use ``AllReduce`` itself.
    """

    def __init__(self, mapping):
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

        self.mnnvl_allreduce = MNNVLAllReduce(mapping, torch.bfloat16)

    def uses_nccl_symmetric_memory_window(self) -> bool:
        return False

    def __call__(self, tensor, *, all_reduce_params=None):
        from tensorrt_llm.functional import AllReduceParams

        if all_reduce_params is not None and not all_reduce_params.enable_allreduce:
            return tensor
        out = self.mnnvl_allreduce(tensor.contiguous(), all_reduce_params=AllReduceParams())
        assert out is not None, "MNNVLAllReduce declined a plain all-reduce"
        return out


def _builtin_all_reduce(mapping):
    """The built-in o_proj all-reduce: the module under MPI, its MNNVL branch otherwise."""
    if isinstance(COMM, _TorchGroup):
        return _BuiltinMnnvlAllReduce(mapping)
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm.functional import AllReduceStrategy

    all_reduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
    assert all_reduce.mnnvl_allreduce is not None, "the built-in MNNVL all-reduce is not active"
    return all_reduce


@pytest.fixture(scope="module")
def env():
    from tensorrt_llm._torch.models.musespark_open_boundary import cluster_boundary
    from tensorrt_llm.mapping import Mapping

    local = int(os.environ.get("SLURM_LOCALID", RANK % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    mapping = Mapping(world_size=WORLD, rank=RANK, gpus_per_node=4, tp_size=WORLD)
    variants = [(POST_EPS, PRE_EPS, False), (POST_EPS, PRE_EPS, True)]
    workspaces = {
        "mc": PostAttnHopWorkspace(mapping, multicast=True, variants=variants),
        "uc": PostAttnHopWorkspace(mapping, multicast=False, variants=variants),
        "mc_early": PostAttnHopWorkspace(
            mapping, multicast=True, variants=variants, early_trigger=True
        ),
        "mc_stale": PostAttnHopWorkspace(
            mapping, multicast=True, variants=variants, stale_poll=True
        ),
        "uc_stale": PostAttnHopWorkspace(
            mapping, multicast=False, variants=variants, stale_poll=True
        ),
        "mc_early_stale": PostAttnHopWorkspace(
            mapping, multicast=True, variants=variants, early_trigger=True, stale_poll=True
        ),
        "mc_stale_all": PostAttnHopWorkspace(
            mapping, multicast=True, variants=variants, stale_poll=True, stale_poll_min_rows=1
        ),
    }
    legacy_ar = _builtin_all_reduce(mapping)

    def legacy(partial, residual, alpha, beta, weight, norm_fp32):
        reduced = legacy_ar(partial)
        return cluster_boundary(
            residual,
            reduced,
            None,
            alpha,
            beta,
            weight,
            POST_EPS,
            PRE_EPS,
            norm_fp32_residual=norm_fp32,
        )

    COMM.Barrier()
    return dict(mapping=mapping, workspaces=workspaces, legacy=legacy, legacy_ar=legacy_ar)


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("cell", op.CELLS, ids=lambda c: c.why[:40].replace(" ", "_"))
def test_cell_is_the_builtin_pair_bit_for_bit(env, certifying, cell, transport):  # noqa: F811 — fixture by name
    rows = cell.spec["rows"]
    norm_fp32 = cell.spec.get("norm_fp32", False)
    cancel = cell.spec.get("cancel", False)
    seed = zlib.crc32(f"{cell.why}|{transport}".encode())
    partials, residual, alpha, beta, weight = _inputs(
        rows, seed, cell.spec.get("zeros", False), cancel
    )
    partial = partials[RANK]
    assert hop_supported(partial, residual, alpha, beta, weight)

    certifying(op)
    ws = env["workspaces"][transport]
    updated, ffn_input = op(
        partial, residual, alpha, beta, weight, POST_EPS, PRE_EPS, norm_fp32, ws
    )
    ref_updated, ref_ffn = env["legacy"](partial, residual, alpha, beta, weight, norm_fp32)
    torch.cuda.synchronize()
    mismatch = (_differing(updated, ref_updated), _differing(ffn_input, ref_ffn))

    exact_updated, exact_ffn = op.reference(
        partials, residual, alpha, beta, weight, POST_EPS, PRE_EPS, norm_fp32, ws
    )
    signed_zeros_ok = True
    if cell.spec.get("zeros"):
        # Columns 0..63: every rank sent -0, so both sides sum to +0 there.
        signed_zeros_ok = bool((_bits(ffn_input[:, :64]) == _bits(ref_ffn[:, :64])).all())
    order_observable = True
    if cancel:
        # The cell is only worth its place if the order shows: the rank-order sum
        # must differ from the reverse-order sum in many elements.
        forward = _sum_bf16(partials, range(WORLD))
        backward = _sum_bf16(partials, reversed(range(WORLD)))
        order_observable = _differing(forward, backward) > 1000
    # Every collective first, in the same order on every rank; local gates after.
    identical = _agree(mismatch == (0, 0))
    zeros_agree = _agree(signed_zeros_ok)
    order_agree = _agree(order_observable)
    assert identical, (
        f"rank {RANK}: {mismatch[0]} residual and {mismatch[1]} prenorm values differ "
        "from the built-in all-reduce + cluster boundary"
    )
    assert zeros_agree
    assert order_agree, "the cancelling inputs do not make the summation order observable"
    # Band vs the reference: both sides share the all-reduce's BF16 value (the
    # reference sums in the op's documented order), so what remains is the
    # BF16 rounding of the post norm, which can move `updated` by beta * 1
    # ulp(bf16) of the normalized branch, and the prenorm by one more rounding;
    # 2 BF16 ulps of the row scale element-wise covers both, RMS stays far below.
    assert_within_ulp(ffn_input, exact_ffn, element_ulp=2.0, rms_ulp=0.05)
    assert_within_ulp(updated, exact_updated, element_ulp=2.0, rms_ulp=0.05, ulp=2.0**-8)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_a_long_burst_of_mixed_shapes_stays_identical(env, transport):
    """Buffer alternation, slot retirement and the counter under 200 back-to-back calls.

    Row counts change from call to call (the slot layout is shared), and the
    built-in pair runs between the fused calls on its own workspace.
    """
    ws = env["workspaces"][transport]
    g = torch.Generator().manual_seed(1234)
    bad = 0
    for call in range(200):
        rows = int(torch.randint(1, MAX_ROWS + 1, (1,), generator=g))
        norm_fp32 = bool(torch.randint(0, 2, (1,), generator=g))
        cancel = bool(torch.randint(0, 4, (1,), generator=g) == 0)
        partials, residual, alpha, beta, weight = _inputs(rows, 10_000 + call, cancel=cancel)
        got = op(partials[RANK], residual, alpha, beta, weight, POST_EPS, PRE_EPS, norm_fp32, ws)
        want = env["legacy"](partials[RANK], residual, alpha, beta, weight, norm_fp32)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ over the burst"


def test_back_to_back_hops_on_all_transports_with_no_kernel_between(env):
    """Hops on one stream with nothing between them, so each one's dependent is the next hop.

    With programmatic dependent launch the next hop's prologue may start before
    this one completes (early trigger: before its boundary math). Its
    dependency wait must still order every transport and counter access.
    """
    g = torch.Generator().manual_seed(4321)
    calls = []
    for call in range(120):
        transport = TRANSPORTS[call % len(TRANSPORTS)]
        rows = int(torch.randint(1, MAX_ROWS + 1, (1,), generator=g))
        calls.append((transport, rows, _inputs(rows, 80_000 + call)))
    torch.cuda.synchronize()
    COMM.Barrier()
    outs = [
        op(p[RANK], r, a, b, w, POST_EPS, PRE_EPS, False, env["workspaces"][t])
        for t, _, (p, r, a, b, w) in calls
    ]
    bad = 0
    for (t, _, (p, r, a, b, w)), got in zip(calls, outs):
        want = env["legacy"](p[RANK], r, a, b, w, False)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ back to back"


@pytest.mark.parametrize("transport", ["mc", "uc", "mc_stale", "uc_stale", "mc_stale_all"])
def test_a_delayed_rank_is_waited_for(env, transport):
    """One rank arrives late (a GPU spin before its call); every rank still agrees."""
    ws = env["workspaces"][transport]
    bad = 0
    for late in range(WORLD):
        partials, residual, alpha, beta, weight = _inputs(8, 20_000 + late)
        want = env["legacy"](partials[RANK], residual, alpha, beta, weight, False)
        if RANK == late:
            torch.cuda._sleep(2_000_000)  # ~1 ms
        got = op(partials[RANK], residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, ws)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ with a delayed rank"


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_graph_replay_with_changing_inputs(env, transport):
    """A captured sequence of hops of several shapes replays identically with new inputs."""
    ws = env["workspaces"][transport]
    shapes = (8, 8, 1, 4, 8, 6)
    static = [_inputs(rows, 30_000 + i) for i, rows in enumerate(shapes)]
    static_inputs = [(s[0][RANK].clone(), *[t.clone() for t in s[1:]]) for s in static]

    def run(fn):
        return [
            fn(partial, residual, alpha, beta, weight)
            for partial, residual, alpha, beta, weight in static_inputs
        ]

    def fused(partial, residual, alpha, beta, weight):
        return op(partial, residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, ws)

    run(fused)  # warm
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run(fused)
    bad = 0
    for replay in range(30):
        for i, rows in enumerate(shapes):
            partials, residual, alpha, beta, weight = _inputs(rows, 40_000 + 100 * replay + i)
            for dst, src in zip(static_inputs[i], (partials[RANK], residual, alpha, beta, weight)):
                dst.copy_(src)
        graph.replay()
        # Eager calls on the same workspace between replays.
        partials, residual, alpha, beta, weight = _inputs(8, 50_000 + replay)
        eager = fused(partials[RANK], residual, alpha, beta, weight)
        eager_want = env["legacy"](partials[RANK], residual, alpha, beta, weight, False)
        bad += _differing(eager[0], eager_want[0]) + _differing(eager[1], eager_want[1])
        wants = run(lambda *a: env["legacy"](*a, False))
        for (got_u, got_n), (want_u, want_n) in zip(outs, wants):
            bad += _differing(got_u, want_u) + _differing(got_n, want_n)
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ across graph replays"


def test_each_workspace_built_the_schedule_each_row_count_runs(env):
    """Per split, the default schedule below the stale-poll threshold, stale-only from it.

    The model's ``stale_poll`` workspace runs the stale-only kernel only on calls of at
    least ``STALE_POLL_MIN_ROWS`` rows; ``mc_stale_all`` lowers the threshold to one row.
    """

    def expected(stale_poll, min_rows):
        return {
            (cluster_for_rows(rows), stale_poll and rows >= min_rows)
            for rows in range(1, MAX_ROWS + 1)
        }

    want = {
        "mc": expected(False, STALE_POLL_MIN_ROWS),
        "uc": expected(False, STALE_POLL_MIN_ROWS),
        "mc_early": expected(False, STALE_POLL_MIN_ROWS),
        "mc_stale": expected(True, STALE_POLL_MIN_ROWS),
        "uc_stale": expected(True, STALE_POLL_MIN_ROWS),
        "mc_early_stale": expected(True, STALE_POLL_MIN_ROWS),
        "mc_stale_all": expected(True, 1),
    }
    assert want["mc_stale"] == {(16, False), (8, False), (8, True)}
    assert want["mc_stale_all"] == {(16, True), (8, True)}
    built = {
        transport: {(key[0], key[4]) for key in env["workspaces"][transport]._kernels}
        for transport in TRANSPORTS
    }
    assert _agree(built == want), f"rank {RANK}: built {built}, expected {want}"


def test_each_call_adds_64_to_the_counter_across_its_wrap(env):
    """The buffer index is the 64s bit of the arrival counter: every call must add exactly 64.

    Whatever its grid (16 CTAs at one row, 8 per row otherwise). Started 5 calls
    short of the u32 wrap, at rest on every rank (every slot of both buffers
    then holds the sentinel, so moving the counter is safe); 12 calls of every
    row count later each counter must read the start plus 12 * 64 mod 2**32,
    and every call must still be the built-in pair bit for bit.
    """
    rows_seq = (1, 8, 3, 1, 5, 8, 2, 7, 4, 6, 1, 8)
    start = (1 << 32) - 64 * 5
    expected = (start + 64 * len(rows_seq)) & 0xFFFFFFFF
    report = {}
    bad = 0
    for transport in TRANSPORTS:
        ws = env["workspaces"][transport]
        torch.cuda.synchronize()
        COMM.Barrier()
        ws.control[1] = start - (1 << 32)
        torch.cuda.synchronize()
        COMM.Barrier()
        for i, rows in enumerate(rows_seq):
            partials, residual, alpha, beta, weight = _inputs(rows, 90_000 + 31 * i)
            norm_fp32 = bool(i % 2)
            got = op(
                partials[RANK], residual, alpha, beta, weight, POST_EPS, PRE_EPS, norm_fp32, ws
            )
            want = env["legacy"](partials[RANK], residual, alpha, beta, weight, norm_fp32)
            bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
        report[transport] = _counter(ws)
    counters_ok = all(value == expected for value in report.values())
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ across the counter wrap"
    assert _agree(counters_ok), f"rank {RANK}: counters {report}, expected {expected} each"


def test_unsupported_inputs_are_refused_before_any_launch(env):
    ws = env["workspaces"]["mc"]
    partials, residual, alpha, beta, weight = _inputs(8, 7)
    partial = partials[RANK]
    before = _counter(ws)
    with pytest.raises(ValueError):
        op(partial[:, :4096], residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, ws)
    with pytest.raises(ValueError):
        op(partial.float(), residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, ws)
    nine = torch.zeros(MAX_ROWS + 1, HIDDEN, dtype=torch.bfloat16, device=partial.device)
    with pytest.raises(ValueError):
        op(nine, nine.float(), alpha, beta, weight, POST_EPS, PRE_EPS, False, ws)
    # A layout the kernel cannot read in place: the op refuses it loudly.
    strided = torch.zeros(8, 2 * HIDDEN, dtype=torch.bfloat16, device=partial.device)[:, ::2]
    assert hop_takes(strided, residual, alpha, beta, weight) and not hop_layout_ok(strided)
    with pytest.raises(ValueError):
        op(strided, residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, ws)
    assert _counter(ws) == before
    # Nothing was launched, so the transports are still in step: one more real call agrees.
    got = op(partial, residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, ws)
    want = env["legacy"](partial, residual, alpha, beta, weight, False)
    torch.cuda.synchronize()
    assert _agree(_differing(got[0], want[0]) + _differing(got[1], want[1]) == 0)


def _builtin_join_and_norm(rows: int, seed: int):
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator().manual_seed(seed)
    gate = 1.5 * torch.randn(HIDDEN, generator=g)
    norm_weight = (0.2 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
    with torch.device(dev):
        join = MuseSparkOpenGatedResidual(HIDDEN, torch.bfloat16)
        pre_ffn_norm = RMSNorm(
            hidden_size=HIDDEN, eps=PRE_EPS, dtype=torch.bfloat16, use_gemma=True
        )
    with torch.no_grad():
        join.gate.copy_(gate.to(torch.bfloat16))
        pre_ffn_norm.weight.copy_(norm_weight)
    join.cache_derived_state()
    return join, pre_ffn_norm, g


@pytest.mark.parametrize("rows", [1, 8, 9])
def test_the_target_join_behind_a_real_o_proj_is_the_builtin_hop(env, rows):
    """The target's armed join behind a real row-parallel o_proj equals the built-in hop.

    Built-in: o_proj -- a row-parallel ``Linear`` whose all-reduce is the
    built-in MNNVL one-shot, as ``MuseSparkOpenAttention`` pins it -- then the
    built-in ``MuseSparkOpenGatedResidual._add_after_rmsnorm_and_norm`` with the
    post-attention eps and a Gemma pre-FFN ``RMSNorm``. Target: the same o_proj
    called with the all-reduce disabled (what ``DecodeTarget`` passes), then the
    same join re-classed by the target and armed with the fused hop. Nine rows
    exercise the armed join's fallback, which the hop does not take.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        layers as target_layers,
    )
    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
    from tensorrt_llm.functional import AllReduceParams

    mapping = env["mapping"]
    dev = torch.device("cuda", torch.cuda.current_device())
    join, pre_ffn_norm, g = _builtin_join_and_norm(rows, 70_000 + rows)
    full_weight = (0.01 * torch.randn(HIDDEN, HIDDEN, generator=g)).to(torch.bfloat16)
    residual = (4.0 * torch.randn(rows, HIDDEN, generator=g)).to(dev)
    shards = [
        (torch.randn(rows, HIDDEN // WORLD, generator=g) * 2.0).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    x = shards[RANK].to(dev)
    with torch.device(dev):
        o_proj = Linear(
            HIDDEN,
            HIDDEN,
            bias=False,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            # Built without its own all-reduce (whose AUTO strategy would set up
            # the IPC workspace), then given the built-in MNNVL one below.
            reduce_output=False,
        )
    o_proj.load_weights([{"weight": full_weight}])
    o_proj.reduce_output = True
    o_proj.all_reduce = _builtin_all_reduce(mapping)

    ws = env["workspaces"]["mc"]
    with torch.inference_mode():
        want = join._add_after_rmsnorm_and_norm(residual, o_proj(x), POST_EPS, pre_ffn_norm)
        target_layers.MuseSparkOpenTepPostAttnJoin.adopt(join, o_proj.all_reduce)
        unarmed = join._add_after_rmsnorm_and_norm(residual, o_proj(x), POST_EPS, pre_ffn_norm)
        before = _counter(ws)
        join._armed = target_layers.PostAttnHopCall(ws, (join,))
        partial = o_proj(x, all_reduce_params=AllReduceParams(enable_allreduce=False))
        got = join._add_after_rmsnorm_and_norm(residual, partial, POST_EPS, pre_ffn_norm)
        join._armed = None
        fused_calls = (_counter(ws) - before) & 0xFFFFFFFF
    torch.cuda.synchronize()
    bad = sum(_differing(a, b) for a, b in zip(got, want))
    bad += sum(_differing(a, b) for a, b in zip(unarmed, want))
    took_fused = fused_calls == (64 if rows <= MAX_ROWS else 0)
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ from the built-in layer hop"
    assert _agree(took_fused), f"rank {RANK}: counter moved by {fused_calls}"


def test_a_rank_local_layout_keeps_every_rank_on_the_fused_hop(env):
    """The armed join chooses the collective on rank-uniform facts only.

    Rank 0 hands it a non-contiguous partial, rank 1 a residual 4 bytes off a
    16-byte boundary, the other ranks fresh tensors. Every rank must still run
    the fused hop (its counter moves by exactly one call) -- choosing the
    built-in all-reduce on one rank would leave every rank waiting -- and the
    result must be the built-in hop bit for bit.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        layers as target_layers,
    )

    rows = 8
    join, pre_ffn_norm, _ = _builtin_join_and_norm(rows, 71_000)
    partials, residual, _, _, _ = _inputs(rows, 71_001)
    partial = partials[RANK]
    with torch.inference_mode():
        want = join._add_after_rmsnorm_and_norm(
            residual, env["legacy_ar"](partial), POST_EPS, pre_ffn_norm
        )
    if RANK == 0:
        wide = torch.zeros(rows, 2 * HIDDEN, dtype=torch.bfloat16, device=partial.device)
        wide[:, ::2] = partial
        partial = wide[:, ::2]
    if RANK == 1 % WORLD:
        flat = torch.zeros(rows * HIDDEN + 1, dtype=torch.float32, device=residual.device)
        flat[1:] = residual.reshape(-1)
        residual = flat[1:].view(rows, HIDDEN)
    layout_ok = hop_layout_ok(partial, residual)
    ws = env["workspaces"]["mc"]
    with torch.inference_mode():
        target_layers.MuseSparkOpenTepPostAttnJoin.adopt(join, env["legacy_ar"])
        before = _counter(ws)
        join._armed = target_layers.PostAttnHopCall(ws, (join,))
        got = join._add_after_rmsnorm_and_norm(residual, partial, POST_EPS, pre_ffn_norm)
        join._armed = None
        moved = (_counter(ws) - before) & 0xFFFFFFFF
    torch.cuda.synchronize()
    bad = sum(_differing(a, b) for a, b in zip(got, want))
    expected_layout = RANK not in (0, 1 % WORLD)
    assert _agree(layout_ok == expected_layout), "the test did not build the layouts it meant to"
    assert _agree(moved == 64), f"rank {RANK}: counter moved by {moved}, not one fused call"
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ from the built-in hop"


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="set MODELING_V2_TIMING=1")
def test_microbenchmark(env):
    """62 layers of o_proj GEMM + hop in one graph; built-in pair vs fused (every transport).

    o_proj is a BF16 ``[M, 1024] x [1024, 8192]`` per layer with a distinct
    weight (the per-rank shape at tep8), and each layer's GEMM reads the previous
    layer's prenorm, so the chain is serial like the decode step. Printed per
    layer = replay time / 62; per arm the median of 2 x 5 x 100 replays, the arms
    timed once in order and once in reverse order; max over ranks. Also the hop
    alone (no GEMM) as a graph of 62 back-to-back calls, where each hop's
    dependent is the next hop.
    """
    layers = 62
    dev = torch.device("cuda", torch.cuda.current_device())
    weights = [
        (0.02 * torch.randn(HIDDEN, 1024, device=dev)).to(torch.bfloat16) for _ in range(layers)
    ]
    names = (
        "builtin",
        "fused_mc",
        "fused_mc_stale",
        "fused_mc_early",
        "fused_mc_early_stale",
        "fused_mc_stale_all",
    )
    results = {}
    for rows in (1, 2, 4, 6, 8):
        partials, residual, alpha, beta, weight = _inputs(rows, 60_000 + rows)
        x0 = partials[RANK][:, :1024].contiguous()

        def chain(hop, with_gemm):
            x = x0
            h = residual
            for i in range(layers):
                part = torch.nn.functional.linear(x, weights[i]) if with_gemm else partials[RANK]
                h, ffn = hop(part, h, alpha, beta, weight)
                x = ffn[:, :1024]
            return h

        def fused(transport):
            ws = env["workspaces"][transport]
            return lambda p, h, a, b, w: op(p, h, a, b, w, POST_EPS, PRE_EPS, False, ws)

        arms = {
            "builtin": lambda p, h, a, b, w: env["legacy"](p, h, a, b, w, False),
            "fused_mc": fused("mc"),
            "fused_mc_stale": fused("mc_stale"),
            "fused_mc_early": fused("mc_early"),
            "fused_mc_early_stale": fused("mc_early_stale"),
            "fused_mc_stale_all": fused("mc_stale_all"),
        }
        for with_gemm in (False, True):
            graphs = {}
            for name in names:
                chain(arms[name], with_gemm)
                torch.cuda.synchronize()
                COMM.Barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    chain(arms[name], with_gemm)
                for _ in range(20):
                    graph.replay()
                graphs[name] = graph
            torch.cuda.synchronize()
            samples = {name: [] for name in names}
            for order in (names, tuple(reversed(names))):
                for name in order:
                    for _ in range(5):
                        COMM.Barrier()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(100):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        samples[name].append(start.elapsed_time(end) * 1000.0 / 100 / layers)
            for name in names:
                results[(rows, with_gemm, name)] = _max_over_ranks(statistics.median(samples[name]))
            del graphs
    if RANK == 0:
        print(f"\npost_attn_hop microbenchmark, world {WORLD}, us per layer (max over ranks)")
        print(f"rows | hop only: {' '.join(names)} | o_proj+hop: {' '.join(names)}")
        for rows in (1, 2, 4, 6, 8):
            a = " ".join(f"{results[(rows, False, n)]:7.2f}" for n in names)
            b = " ".join(f"{results[(rows, True, n)]:7.2f}" for n in names)
            print(f"{rows:4d} | {a} | {b}")
