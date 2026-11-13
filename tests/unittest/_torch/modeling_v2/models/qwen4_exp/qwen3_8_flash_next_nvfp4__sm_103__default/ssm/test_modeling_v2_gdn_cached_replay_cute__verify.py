# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the CuTe DSL cached-replay entry (`gdn_cached_replay_cute__verify`) at a
tp4/tep4 rank's 4 key / 12 value heads.

Each cell drives consecutive verify steps the way the engine does, through the Triton
cached-replay entry's test harness (`_Bench`: the stock manager's bf16 state page at the tep4 engine's slot
count, work items, the all-layer commit, `_advance_replay_state` with random acceptance), and
holds every call to this entry's `reference` on that call's own inputs (output and all four
state buffers, the Triton entry's bands) and every output to the exact twin (the fp32 token-by-token
recurrence over every accepted token). With `strided_gates` the raw a / b reach the kernel as
column views of an input-projection-shaped buffer, as the target hands them over.

Beyond the cells: CUDA-graph replay bitwise eager, the launch mappings bitwise each other, the
history and its bookkeeping at replay slots of their own (a manager's KVCM pages) bitwise the
coupled run -- through this entry and through the target's verify dispatch -- and `precompile`
covering every verify step.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.ssm.gdn_cached_replay_cute__verify as cr  # noqa: E501
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.modules.fla.cached_replay import CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE

_MV2_TESTS = Path(__file__).resolve().parents[4]
if str(_MV2_TESTS) not in sys.path:
    sys.path.insert(0, str(_MV2_TESTS))
from _validating import certifying, validating  # noqa: E402,F401 — certifying: fixture by name

_spec = importlib.util.spec_from_file_location(
    "cached_replay_harness",
    # The Triton entry's test, in the family tree (tests/unittest/_torch/modeling_v2/models/qwen4_exp/ops/ssm).
    _MV2_TESTS
    / "models/qwen4_exp/ops/ssm"
    / "test_modeling_v2_fused_recurrent_gated_delta_rule_cached_replay_update.py",
)
trep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trep)

assert torch.cuda.is_available(), "gdn_cached_replay_cute requires a CUDA device"

op = cr.gdn_cached_replay_cute
DEV = "cuda"
K, V = cr.K, cr.V


def _gates(bench, act, strided):
    """The raw a / b `[N, T, HV]`: the harness's contiguous tensors, or the same values as
    column views of a `[N*T, conv + z + 2 HV]` projection (row stride = its width)."""
    if not strided:
        return act["a"], act["b"]
    n, t, h, hv = bench.n, bench.t, bench.h, bench.hv
    # The input projection's column split at the rank's heads: conv (q|k|v), z, a, b.
    conv, z = 2 * h * K + hv * V, hv * V
    proj = torch.randn(n * t, conv + z + 2 * hv, device=DEV, dtype=torch.bfloat16)
    a = proj[:, conv + z : conv + z + hv]
    b = proj[:, conv + z + hv :]
    a.copy_(act["a"].reshape(n * t, hv))
    b.copy_(act["b"].reshape(n * t, hv))
    a, b = a.reshape(n, t, hv), b.reshape(n, t, hv)
    assert a.stride() == (t * proj.shape[1], proj.shape[1], 1), a.stride()
    return a, b


def _reference(bench, kw):
    """This entry's `reference` on a call's arguments, every per-slot buffer through the
    harness's `compact` (the touched slots' rows of an engine page)."""
    if bench.idx is None:
        return op.reference(**kw)
    per_slot = (
        "ssm_states",
        "old_u",
        "old_k",
        "old_G",
        "old_beta",
        "cache_buf_idx",
        "prev_num_accepted_tokens",
    )
    compact = {name: bench.compact(kw[name]) for name in per_slot}
    return op.reference(
        **dict(kw, **compact, state_indices=bench.compact_indices(kw["state_indices"]))
    )


def _drive(cell, bench) -> None:
    strided = cell.spec.get("strided_gates", False)
    wraps = appends = 0
    for _ in range(bench.steps):
        mask, wrapped, _ = bench.writes()
        wraps += wrapped.numel()
        appends += bench.live.numel() - wrapped.numel()
        bench.prepare()
        after_commit = []
        for layer in range(bench.layers):
            act = bench.inputs()
            kw = bench.kwargs(layer, act)
            kw["g"], kw["beta"] = _gates(bench, act, strided)
            _, u, k, G = bench.state(layer)
            before = dict(u=u.clone(), k=k.clone(), G=G.clone(), dA=bench.old_dA[layer].clone())
            ref = _reference(bench, kw)
            exact = op.reference(**bench.exact_kwargs(layer, kw))
            committed = None
            if bench.path == "all_layer":
                committed = _reference(
                    bench,
                    dict(kw, replay_work_items=None, n_writes=None, use_all_layer_commit=False),
                )
            got = op(**kw)
            trep._compare((got, *bench.observed(layer)), ref)
            _, u, k, G = bench.state(layer)
            assert torch.equal(bench.old_dA[layer], before["dA"]), "old_beta is never written"
            assert torch.equal(u[~mask], before["u"][~mask]), "old_u written outside the append"
            assert torch.equal(k[~mask], before["k"][~mask]), "old_k written outside the append"
            Gt, Gt0 = G.transpose(-1, -2), before["G"].transpose(-1, -2)
            assert torch.equal(Gt[~mask], Gt0[~mask]), "old_G written outside the append"
            bench.adopt_exact(layer, exact)
            after_commit.append((got, exact, committed))
        bench.commit()
        for layer, (got, exact, committed) in enumerate(after_commit):
            if committed is not None:
                # The partitioned call plus the manager's commit is the call that commits
                # in-kernel.
                trep._compare((got, *bench.observed(layer)), committed)
            trep._compare((got, None, None, None, None), (exact[0], None, None, None, None))
        if bench.page is not None:
            bench.page.assert_rest_unchanged()
        bench.accept(bench.draw_acceptance())
    assert wraps > 0 and appends > 0, f"the cell never wrapped or appended ({wraps}, {appends})"


def _drive_padding(cell) -> None:
    """Trailing rows sharing one slot, as CUDA-graph padding repeats the manager's dummy
    slot: the rows on unique slots, and every slot but the shared one, are held to the
    reference run on the unique rows alone (a dense pool: the shared slot's rows are
    indexed out)."""
    spec = {k_: v_ for k_, v_ in cell.spec.items() if k_ not in ("page_layers",)}
    spec["slots"] = 24
    bench = trep._Bench(spec, seed=trep._seed(cell.why), exact=False)
    dup = spec["dup"]
    unique = bench.n - len(dup)
    shared = int(bench.state_indices[unique])
    others = torch.ones(bench.slots, dtype=torch.bool, device=DEV)
    others[shared] = False
    for _ in range(bench.steps):
        for layer in range(bench.layers):
            act = bench.inputs()
            kw = bench.kwargs(layer, act)
            kw["g"], kw["beta"] = _gates(bench, act, spec.get("strided_gates", False))
            alone = dict(
                kw,
                q=kw["q"][:unique],
                k=kw["k"][:unique],
                v=kw["v"][:unique],
                g=kw["g"][:unique],
                beta=kw["beta"][:unique],
                state_indices=kw["state_indices"][:unique],
                packed_qkv=act["packed"][: unique * bench.t],
            )
            ref = op.reference(**alone)
            got = op(**kw)
            pool, u, k, G = bench.state(layer)
            trep._compare(
                (got[:unique], pool[others], u[others], k[others], G[others]),
                (ref[0], *(x[others] for x in ref[1:])),
            )
        bench.accept(bench.draw_acceptance())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    trep._MEASURED.clear()
    if "dup" in cell.spec:
        _drive_padding(cell)
    else:
        bench = trep._Bench(cell.spec, seed=trep._seed(cell.why))
        try:
            _drive(cell, bench)
        finally:
            bench.close()
    print(
        f"\n[{cell.why}] "
        + ", ".join(f"{k_} {v_:.3f}" for k_, v_ in sorted(trep._MEASURED.items()))
    )


def _rank_spec(heads=(cr.RANK_K_HEADS, cr.RANK_V_HEADS), **spec):
    return dict(state_dtype=torch.bfloat16, H=heads[0], HV=heads[1], **spec)


@pytest.mark.parametrize("heads", cr.GEOMETRIES, ids=[f"{h}x{hv}" for h, hv in cr.GEOMETRIES])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("n,path", [(1, "small"), (4, "small"), (16, "all_layer")])
def test_cuda_graph_replay_on_new_data_is_bitwise_eager(n, path, pdl, heads) -> None:
    """Two layers captured once (strided a / b, the all-layer commit after them when
    partitioned), replayed three times on new activations with acceptance in between,
    bitwise against the same steps run eagerly on a twin state; with `pdl` the replay
    launches with programmatic dependent launch, as the target's verify step does."""
    t = 4
    spec = _rank_spec(heads, path=path, n=n, t=t, layers=2, slots=40)
    seed = 9100 + n
    eager, graph, warm = (trep._Bench(spec, seed, exact=False) for _ in range(3))
    acts = [warm.inputs() for _ in range(2)]
    gates = [_gates(warm, act, True) for act in acts]

    def step(b):
        b.prepare()
        outs = []
        for layer in range(2):
            kw = b.kwargs(layer, acts[layer])
            kw["g"], kw["beta"] = gates[layer]
            kw["launch_with_pdl"] = pdl
            outs.append(op(**kw))
        b.commit()
        return outs

    step(warm)
    torch.cuda.synchronize()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        outs_graph = step(graph)
    refill = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(3):
        for act, (a, b) in zip(acts, gates):
            for x in (act["packed"], a, b):
                x.copy_(torch.randn(x.shape, generator=refill, device=DEV, dtype=x.dtype))
        outs_eager = step(eager)
        captured.replay()
        torch.cuda.synchronize()
        for e, g in zip(outs_eager, outs_graph):
            assert torch.equal(e, g), "replayed output differs from eager"
        for name in ("backing", "old_x", "old_B", "old_dt", "pnat", "buf"):
            assert torch.equal(getattr(eager, name), getattr(graph, name)), f"{name} differs"
        accepted = eager.draw_acceptance()
        eager.accept(accepted)
        graph.accept(accepted)


#: Launch mappings the 4 / 12 geometry can take, each checked against the tp1 default
#: small-batch mapping: every one computes the same instructions per state row.
_MAPPINGS = (
    dict(tpc=1, warps=1, pf=0, minb=4, early_acc=True),
    dict(tpc=2, warps=2, pf=0, minb=4, early_acc=True),
    dict(tpc=4, warps=4, pf=0, minb=4, early_acc=True),
    dict(tpc=8, warps=4, pf=1, minb=4),
    dict(tpc=24, warps=8, pf=1, minb=2),
)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("n,t", [(1, 4), (4, 4), (2, 2), (1, 8)])
def test_launch_mappings_agree(n, t, pdl) -> None:
    """One step bitwise equal across launch mappings (output, checkpoint with in-kernel
    commits, u / k / G history), with wrapping and appending rows and, from 4 rows, a
    negative slot."""
    spec = _rank_spec(path="small", n=n, t=t, steps=1, layers=1, slots=24)
    bench = trep._Bench(spec, 4242 + n * 10 + t, exact=False)
    sidx = bench.state_indices.clone()
    if n > 2:
        sidx[n // 2] = -1
    act = bench.inputs()
    a, b = _gates(bench, act, True)
    snap = [x.clone() for x in bench.state(0)]
    results = []
    for cfg in _MAPPINGS:
        for x, y in zip(bench.state(0), snap):
            x.copy_(y)
        kw = bench.kwargs(0, act)
        sentinel = torch.full((n, t, bench.hv, V), 3.0, dtype=torch.bfloat16, device=DEV)
        kw.update(g=a, beta=b, state_indices=sidx, output=sentinel, config=cfg, launch_with_pdl=pdl)
        out = op(**kw)
        torch.cuda.synchronize()
        results.append([out.clone()] + [x.clone() for x in bench.state(0)])
    for cfg, got in zip(_MAPPINGS[1:], results[1:]):
        for name, x, y in zip(("out", "state", "u", "k", "G"), got, results[0]):
            assert torch.equal(x, y), f"{name}: {cfg} vs {_MAPPINGS[0]}"


# ── replay slots of their own ────────────────────────────────────────────


def _replay_pages(slots, layers, h, hv):
    """The history and its bookkeeping in pages of their own, as a manager keeps them in its
    KVCM pool: one page per replay slot holding every layer's `old_u`, then every layer's
    `old_k`, then every layer's `old_G`, then the pending count and the half (int32), and 8
    bytes nobody writes; every byte 0xFF (NaN history, counts of -1) until written. The
    `[layers, slots, ...]` history views have the page as their slot stride, the bookkeeping
    columns the page as their element stride; `data` is the `[slots, page]` bytes, `tail`
    the offset of the unwritten bytes."""
    regions = (
        (torch.bfloat16, (2, trep.HIST, hv, V)),
        (torch.bfloat16, (2, trep.HIST, h, K)),
        (torch.float32, (2, hv, trep.HIST)),
    )
    sizes = [math.prod(shape) * dtype.itemsize for dtype, shape in regions]
    page = layers * sum(sizes) + 16
    data = torch.full((slots, page), 0xFF, dtype=torch.uint8, device=DEV)
    flat = data.view(-1)
    views, offset = [], 0
    for (dtype, shape), size in zip(regions, sizes):
        es = dtype.itemsize
        rows = torch.empty(shape, device="meta").stride()
        views.append(
            torch.as_strided(
                flat.view(dtype),
                (layers, slots, *shape),
                (size // es, page // es, *rows),
                offset // es,
            )
        )
        offset += layers * size
    pnat, buf = (
        torch.as_strided(flat.view(torch.int32), (slots,), (page // 4,), offset // 4 + field)
        for field in (0, 1)
    )
    return SimpleNamespace(
        data=data, u=views[0], k=views[1], G=views[2], pnat=pnat, buf=buf, tail=offset + 8
    )


class _ReplaySlotBench(trep._Bench):
    """The harness's bench (the same draws for the same seed) with the history and its
    bookkeeping moved to replay slots of their own (`_replay_pages`): row `i`'s at
    `replay_indices[i]`, a permutation of slots past the checkpoint pool's, so no row's replay
    slot is a checkpoint slot. The work items carry the replay slot as their fifth field; the
    kernel is handed their first `width` fields (4: it reads the replay slot from
    `replay_indices`), the manager's all-layer commit the full items. `writes` is the coupled
    bench's alone (it reads the bookkeeping at the checkpoint slots)."""

    def __init__(self, spec, seed, width):
        super().__init__(spec, seed, exact=False)
        n, slots = self.n, self.state_indices.long()
        cpu = torch.Generator().manual_seed(seed + 1)
        replay = torch.randperm(2 * n, generator=cpu)[:n] + self.slots
        self.replay_indices = replay.to(DEV, torch.int32)
        rows = self.replay_indices.long()
        self.pages = _replay_pages(self.slots + 2 * n, self.layers, self.h, self.hv)
        pages = self.pages
        for paged, coupled in zip(
            (pages.u, pages.k, pages.G), (self.old_x, self.old_B, self.old_dt)
        ):
            paged[:, rows] = coupled[:, slots]
        pages.pnat[rows] = self.pnat[slots]
        pages.buf[rows] = self.buf[slots]
        self.old_x, self.old_B, self.old_dt = pages.u, pages.k, pages.G
        self.pnat, self.buf = pages.pnat, pages.buf
        self.meta = trep.ReplayStateUpdateMetadata(self.pnat, self.buf, self.t, trep.HIST)
        self.item_width = width
        self.kernel_items = None

    def prepare(self):
        if self.path == "all_layer":
            self.work_items.zero_()
            trep._build_replay_work_items_triton(
                self.live,
                self.pnat,
                self.buf,
                self.work_items,
                self.n_writes,
                self.t,
                trep.HIST,
                self.replay_indices,
            )
            full = self.work_items.shape[1]
            self.kernel_items = (
                self.work_items
                if self.item_width == full
                else self.work_items[:, : self.item_width].contiguous()
            )

    def kwargs(self, layer, act):
        kw = super().kwargs(layer, act)
        kw["replay_indices"] = self.replay_indices
        if self.kernel_items is not None:
            kw["replay_work_items"] = self.kernel_items
        return kw

    def accept(self, accepted):
        trep._advance_replay_state(self.meta, self.replay_indices, accepted)


_SLOT_STATE = ("checkpoint", "old_u", "old_k", "old_G", "pending count", "half")


def _slot_state(bench):
    """A bench's checkpoints, and its rows' history and bookkeeping at their history slots
    (the replay slots, when it has them)."""
    rows = getattr(bench, "replay_indices", bench.state_indices).long()
    return (
        bench.backing,
        bench.old_x[:, rows],
        bench.old_B[:, rows],
        bench.old_dt[:, rows],
        bench.pnat[rows],
        bench.buf[rows],
    )


#: Replay-slot runs: requests, the work-item width the kernel is handed (None: no work items,
#: the small path), and whether the kernel commits wrapping rows itself.
_REPLAY_SLOT_RUNS = {
    "small_n1": (1, None, True),
    "small_n4": (4, None, True),
    "w4_in_kernel": (CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE, 4, True),
    "w4_all_layer": (CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE, 4, False),
    "w5_all_layer": (32, 5, False),
}


@pytest.mark.parametrize("heads", cr.GEOMETRIES, ids=[f"{h}x{hv}" for h, hv in cr.GEOMETRIES])
@pytest.mark.parametrize(
    "n,width,in_kernel", list(_REPLAY_SLOT_RUNS.values()), ids=list(_REPLAY_SLOT_RUNS)
)
def test_separate_replay_slots_match_coupled(
    n,
    width,
    in_kernel,
    heads,
    certifying,  # noqa: F811 — fixture by name
) -> None:
    """Replay slots that differ from the checkpoint slots give, bit for bit, the coupled run.

    The coupled side is the harness's bench: history and bookkeeping at the checkpoint slots.
    The other (`_ReplaySlotBench`) keeps the same history and bookkeeping at a permuted set of
    replay slots past the checkpoint pool's, in pages holding every layer's rows (a slot stride
    of the whole page, the bookkeeping as strided int32 fields of it), 0xFF elsewhere. Both run
    verify steps through this entry -- strided a / b and PDL, as the target hands them over --
    with checkpoint wraps committed in-kernel (without work items, or with them and
    `in_kernel_commit`) or by the manager's all-layer commit, the kernel handed the work items
    at `width` fields (4: the replay slot from `replay_indices`; 5: from the items' fifth
    field), then advance their bookkeeping with the same acceptance. Each replay-slot call is
    also held to this entry's `reference` given the same `replay_indices`, and no byte of the
    pages outside the rows' replay slots is written.
    """
    certifying(op)
    t = 4
    path = "small" if width is None else "all_layer"
    spec = _rank_spec(heads, path=path, n=n, t=t, layers=2, steps=trep.HIST // t + 3)
    seed = 31 + n
    coupled = trep._Bench(spec, seed, exact=False)
    paged = _ReplaySlotBench(spec, seed, width or trep.REPLAY_WORK_ITEM_WIDTH)
    sides = (coupled, paged)
    rows = paged.replay_indices.long()
    assert not torch.isin(rows, coupled.state_indices.long()).any()
    wraps = appends = 0
    for step in range(coupled.steps):
        wrapped = coupled.writes()[1].numel()
        wraps += wrapped
        appends += n - wrapped
        for side in sides:
            side.prepare()
        if width is not None:
            assert torch.equal(coupled.n_writes, paged.n_writes)
        for layer in range(coupled.layers):
            act = coupled.inputs()
            a, b = _gates(coupled, act, True)
            outs = []
            ref = None
            for side in sides:
                kw = dict(side.kwargs(layer, act), g=a, beta=b, launch_with_pdl=True)
                if in_kernel:
                    kw.update(in_kernel_commit=True, use_all_layer_commit=False)
                if side is paged:
                    ref = op.reference(**kw)
                outs.append(op(**kw))
            pool, u, k, G = paged.state(layer)
            trep._compare(
                (outs[1], pool, u[rows], k[rows], G[rows]),
                (ref[0], ref[1], ref[2][rows], ref[3][rows], ref[4][rows]),
            )
            assert torch.equal(outs[0], outs[1]), f"step {step} layer {layer}: outputs differ"
        if not in_kernel:
            for side in sides:
                side.commit()
        accepted = coupled.draw_acceptance()
        for side in sides:
            side.accept(accepted)
        for name, x, y in zip(_SLOT_STATE, _slot_state(coupled), _slot_state(paged)):
            assert torch.equal(x, y), f"step {step}: {name} differs"
    assert wraps > 0 and appends > 0, f"the run never wrapped or appended ({wraps}, {appends})"
    pages = paged.pages
    elsewhere = torch.ones(pages.data.shape[0], dtype=torch.bool, device=DEV)
    elsewhere[rows] = False
    assert (pages.data[elsewhere] == 0xFF).all(), "a page outside the rows' replay slots moved"
    assert (pages.data[:, pages.tail :] == 0xFF).all(), "a page's bytes past its rows moved"


def _dispatch_core(bench, separate):
    """What `_verify_state` and `_gdn_verify` read, over `bench`'s pool, history and
    bookkeeping: the GDN core's geometry and bind (the pools, the replay buffers, a manager
    with the all-layer commit on and, with `separate`, replay slots of its own), and the mamba
    metadata the step's prepare fills (`bench.prepare`)."""
    replay = SimpleNamespace(
        ring=None,
        old={
            layer: (
                bench.old_x[layer],
                bench.old_B[layer],
                bench.old_dt[layer],
                bench.old_dA[layer],
            )
            for layer in range(bench.layers)
        },
        cache_buf_idx=bench.buf,
        prev_num_accepted=bench.pnat,
        history=trep.HIST,
    )
    cm = SimpleNamespace(
        use_gdn_cached_replay_all_layer_commit=True,
        replay_layer_group_id=1 if separate else None,
    )
    core = SimpleNamespace(
        conv_dim=bench.width,
        key_dim=bench.h * K,
        value_dim=bench.hv * V,
        num_k_heads=bench.h,
        head_k_dim=K,
        num_v_heads=bench.hv,
        head_v_dim=V,
        _bound=SimpleNamespace(
            pools=SimpleNamespace(ssm=dict(enumerate(bench.pool))),
            replay=replay,
            cm=cm,
            int_idx=torch.arange(bench.n, dtype=torch.int32, device=DEV),
        ),
    )
    mm = SimpleNamespace(
        replay_work_items=bench.work_items,
        replay_n_writes=bench.n_writes,
        replay_state_indices=bench.replay_indices if separate else bench.state_indices,
    )
    return core, mm


@pytest.mark.parametrize("heads", cr.GEOMETRIES, ids=[f"{h}x{hv}" for h, hv in cr.GEOMETRIES])
@pytest.mark.parametrize("separate", [False, True], ids=["coupled", "replay_slots"])
@pytest.mark.parametrize("n,t", [(1, 4), (4, 4), (32, 4), (5, 3)])
def test_target_dispatch_matches_the_plain_call(n, t, separate, heads) -> None:
    """The target's verify dispatch over a bf16 pool (`_verify_state`, then `_gdn_verify` per
    layer, as the decode target calls them: `a` / `b` column views of the input projection,
    the conv output as `packed_qkv`, PDL after the conv) against the call it replaces
    (contiguous `a` / `b`, no PDL), each on its own copy of the pool, the history and the
    bookkeeping, over consecutive verify steps with random partial acceptance in between:
    outputs, checkpoints, history and bookkeeping bitwise equal at every step, with the
    entry's guard on. With `separate` the dispatched side's manager keeps the history and its
    bookkeeping at replay slots of its own (`_ReplaySlotBench`), which `_verify_state` takes
    from the metadata's `replay_state_indices` and `_gdn_verify` hands this entry as
    `replay_indices`. From `CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE` requests both sides take
    the work items and the manager's all-layer commit; below it the kernel commits wraps."""
    path = "all_layer" if n >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE else "small"
    spec = _rank_spec(heads, path=path, n=n, t=t, layers=2, steps=trep.HIST // t + 3)
    seed = 77 + n
    plain = trep._Bench(spec, seed, exact=False)
    disp = (
        _ReplaySlotBench(spec, seed, trep.REPLAY_WORK_ITEM_WIDTH)
        if separate
        else trep._Bench(spec, seed, exact=False)
    )
    core, mm = _dispatch_core(disp, separate)
    wraps = 0
    with validating(op):
        for step in range(plain.steps):
            wraps += plain.writes()[1].numel()
            plain.prepare()
            disp.prepare()
            verify = M._verify_state(core, mm, n)
            assert verify[3] == (path == "all_layer"), (
                "the all-layer commit exactly when partitioned"
            )
            if separate:
                assert torch.equal(verify[4], disp.replay_indices), "the rows' replay slots"
            else:
                assert verify[4] is None, "no replay slots of its own: the checkpoint slots"
            for layer in range(plain.layers):
                act = plain.inputs()
                a, b = _gates(plain, act, True)
                want = op(**plain.kwargs(layer, act))
                gating = SimpleNamespace(a_log=disp.a_log[layer], dt_bias=disp.dt_bias[layer])
                got = M._gdn_verify(
                    core,
                    None,
                    a.reshape(n * t, -1),
                    b.reshape(n * t, -1),
                    layer,
                    gating,
                    disp.state_indices,
                    n,
                    t,
                    verify,
                    mq_out=act["packed"],
                )
                assert torch.equal(got.view_as(want), want), f"step {step} layer {layer}"
            plain.commit()
            disp.commit()
            accepted = plain.draw_acceptance()
            plain.accept(accepted)
            disp.accept(accepted)
            for name, x, y in zip(_SLOT_STATE, _slot_state(disp), _slot_state(plain)):
                assert torch.equal(x, y), f"step {step}: {name} differs"
    assert wraps > 0, "no history wrapped: no checkpoint commit was exercised"


@pytest.mark.parametrize("heads", cr.GEOMETRIES, ids=[f"{h}x{hv}" for h, hv in cr.GEOMETRIES])
@pytest.mark.parametrize("separate", [False, True], ids=["coupled", "replay_slots"])
def test_precompile_covers_every_step(heads, separate) -> None:
    """After `precompile`, as the target runs it at bind, a verify step of any request count
    compiles nothing: steps at each launch mapping's edges, 9 to 15 requests (an eager mixed
    step's, which no padded decode graph captures) among them, the partitioned ones with work
    items (the metadata's width) and the all-layer commit as the target hands them over. With
    `separate` the manager keeps the history at replay slots of its own: precompiled with
    `separate_replay_slots`, every step passes `replay_indices` and work items carrying the
    replay slot (`_ReplaySlotBench`)."""
    t, partition_from = 4, CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    width = trep.REPLAY_WORK_ITEM_WIDTH
    call_h, call_hv = heads
    kernels = cr.precompile(
        t,
        call_h,
        call_hv,
        40,
        pdl=True,
        partition_from=partition_from,
        work_item_width=width,
        separate_replay_slots=separate,
    )
    assert kernels > 0, kernels
    compiled = len(cr._COMPILED)
    for n in (1, 2, 3, 8, 9, 12, 15, 16, 17, 40):
        path = "all_layer" if n >= partition_from else "small"
        spec = _rank_spec(heads, path=path, n=n, t=t, steps=1, layers=1, slots=64)
        bench = (
            _ReplaySlotBench(spec, 5150 + n, width)
            if separate
            else trep._Bench(spec, 5150 + n, exact=False)
        )
        bench.prepare()
        act = bench.inputs()
        kw = bench.kwargs(0, act)
        kw["g"], kw["beta"] = _gates(bench, act, True)
        kw["launch_with_pdl"] = True
        op(**kw)
        torch.cuda.synchronize()
        assert len(cr._COMPILED) == compiled, f"a step of {n} requests compiled a kernel"
