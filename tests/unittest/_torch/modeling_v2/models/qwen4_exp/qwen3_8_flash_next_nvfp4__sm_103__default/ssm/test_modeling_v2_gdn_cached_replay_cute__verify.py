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
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.ssm.gdn_cached_replay_cute__verify as cr  # noqa: E501
from tensorrt_llm._torch.modules.fla.cached_replay import CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE

_MV2_TESTS = Path(__file__).resolve().parents[4]
if str(_MV2_TESTS) not in sys.path:
    sys.path.insert(0, str(_MV2_TESTS))
from _validating import certifying  # noqa: E402,F401 — fixture by name

_spec = importlib.util.spec_from_file_location(
    "cached_replay_harness",
    # The Triton entry's test, in the family tree (tests/unittest/_torch/modeling_v2/models/qwen4_exp/ssm).
    _MV2_TESTS
    / "models/qwen4_exp/ssm"
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


@pytest.mark.parametrize("heads", cr.GEOMETRIES, ids=[f"{h}x{hv}" for h, hv in cr.GEOMETRIES])
def test_precompile_covers_every_step(heads) -> None:
    """After `precompile`, as the target runs it at bind, a verify step of any request count
    compiles nothing: steps at each launch mapping's edges, 9 to 15 requests (an eager mixed
    step's, which no padded decode graph captures) among them, the partitioned ones with work
    items and the all-layer commit as the target hands them over."""
    t, partition_from = 4, CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE
    kernels = cr.precompile(t, *heads, 40, pdl=True, partition_from=partition_from)
    assert kernels > 0, kernels
    compiled = len(cr._COMPILED)
    for n in (1, 2, 3, 8, 9, 12, 15, 16, 17, 40):
        path = "all_layer" if n >= partition_from else "small"
        spec = _rank_spec(heads, path=path, n=n, t=t, steps=1, layers=1, slots=64)
        bench = trep._Bench(spec, 5150 + n, exact=False)
        bench.prepare()
        act = bench.inputs()
        kw = bench.kwargs(0, act)
        kw["g"], kw["beta"] = _gates(bench, act, True)
        kw["launch_with_pdl"] = True
        op(**kw)
        torch.cuda.synchronize()
        assert len(cr._COMPILED) == compiled, f"a step of {n} requests compiled a kernel"
