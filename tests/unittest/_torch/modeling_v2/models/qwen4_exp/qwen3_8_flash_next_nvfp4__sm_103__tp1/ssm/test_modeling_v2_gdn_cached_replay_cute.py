# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the CuTe DSL cached-replay kernel (`gdn_cached_replay_cute`) over consecutive
verify steps against the Triton entry's fp32 `reference` and the Triton kernel itself, driven by
that entry's test harness (`_Bench`: work items, all-layer commit, `_advance_replay_state` with
random partial acceptance), in both commit modes: (a) no in-kernel commit + the manager's
all-layer commit when partitioned, (b) every wrapping row committed in-kernel. Plus rows with a
negative slot and CUDA-graph replay (PDL on and off) bitwise against eager."""

import importlib.util
import os

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.gdn_cached_replay_cute as cr  # noqa: E501

_spec = importlib.util.spec_from_file_location(
    "trep",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_modeling_v2_fused_recurrent_gated_delta_rule_cached_replay_update.py",
    ),
)
trep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trep)

assert torch.cuda.is_available(), "gdn_cached_replay_cute requires a CUDA device"

BF = torch.bfloat16
ULP = trep.BF16_ULP


def blk_err(got, want):
    """max |got - want| / absmax of its block (last two dims), in bf16 ulps; relative RMS."""
    block = got.shape[-2] * got.shape[-1]
    g, w = got.float().reshape(-1, block), want.float().reshape(-1, block)
    keep = ~torch.isnan(w).any(1)
    g, w = g[keep], w[keep]
    if g.numel() == 0:
        return 0.0, 0.0
    scale = w.abs().amax(1, keepdim=True).clamp_min(1e-30)
    worst = ((g - w).abs() / scale).nan_to_num(nan=float("inf")).max().item() / ULP
    rms = ((g - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt().clamp_min(1e-30)).item() / ULP
    return worst, rms


class Stat:
    def __init__(self):
        self.d = {}

    def add(self, name, val):
        a, b = val if isinstance(val, tuple) else (val, 0.0)
        o = self.d.get(name, (0.0, 0.0))
        self.d[name] = (max(o[0], a), max(o[1], b))

    def fmt(self):
        return "  ".join(f"{k}={v[0]:.2f}/{v[1]:.2f}" for k, v in self.d.items())


def run_cell(name, spec, mode, config=None, seed=1234):
    """mode 'a': mine without commit (+ the manager's all-layer commit when partitioned);
    mode 'b': mine commits wrapping rows in-kernel (no all-layer commit for mine)."""
    spec = dict(spec, state_dtype=BF)
    bt = trep._Bench(spec, seed, exact=False)
    bm = trep._Bench(spec, seed, exact=False)
    dup = tuple(spec.get("dup", ()))
    n = bt.n
    unique = n - len(dup)
    others = torch.ones(bm.slots, dtype=torch.bool, device="cuda")
    if dup:
        others[int(bm.state_indices[unique])] = False
    vs_ref, vs_tri, tri_ref = Stat(), Stat(), Stat()
    fails = []
    wraps = 0
    for step in range(bt.steps):
        _, wrapped, _ = bm.writes()
        wraps += wrapped.numel()
        bt.prepare()
        bm.prepare()
        wi_all = nw_all = None
        if bm.path == "all_layer":
            # the engine's work items cover every decode row (padding included)
            wi_all = torch.zeros(bm.n, 4, dtype=torch.int32, device="cuda")
            nw_all = torch.zeros(1, dtype=torch.int32, device="cuda")
            trep._build_replay_work_items_triton(
                bm.state_indices, bm.pnat, bm.buf, wi_all, nw_all, bm.t, trep.HIST
            )
        for layer in range(bt.layers):
            at, am = bt.inputs(), bm.inputs()
            kt, km = bt.kwargs(layer, at), bm.kwargs(layer, am)
            rkw = dict(km)
            if mode == "b":
                rkw.update(replay_work_items=None, n_writes=None, use_all_layer_commit=False)
            if dup:
                rkw.update(
                    q=km["q"][:unique],
                    k=km["k"][:unique],
                    v=km["v"][:unique],
                    g=km["g"][:unique],
                    beta=km["beta"][:unique],
                    state_indices=km["state_indices"][:unique],
                    packed_qkv=am["packed"][: unique * bm.t],
                )
            ref = trep.op.reference(**rkw)
            tkw = dict(kt)
            if mode == "b":
                tkw.update(replay_work_items=None, n_writes=None, use_all_layer_commit=False)
            rt = None if dup else trep.op.reference(**tkw)
            got_t = trep.op(**tkw)
            if rt is not None:
                tri_ref.add("out", blk_err(got_t, rt[0]))
                tri_ref.add("u", blk_err(bt.state(layer)[1], rt[2])[0])
                tri_ref.add("state", blk_err(bt.state(layer)[0], rt[1])[0])
            kmm = dict(km)
            if wi_all is not None:
                kmm.update(replay_work_items=wi_all, n_writes=nw_all)
            got_m = cr.cached_replay_update(**kmm, in_kernel_commit=(mode == "b"), config=config)
            pool, u, k, G = bm.state(layer)
            gm = (got_m[:unique], pool[others], u[others], k[others], G[others])
            rf = (ref[0], *(x[others] for x in ref[1:]))
            try:
                trep._compare(gm, rf)
            except AssertionError as e:  # record, keep going
                fails.append(f"step {step} layer {layer}: {str(e)[:200]}")
            vs_ref.add("out", blk_err(gm[0], rf[0]))
            vs_ref.add("state", blk_err(gm[1], rf[1])[0])
            vs_ref.add("u", blk_err(gm[2], rf[2])[0])
            vs_ref.add("k", blk_err(gm[3], rf[3])[0])
            vs_ref.add("G", (gm[4] - rf[4]).abs().max().item())
            tp, tu, tk, tG = bt.state(layer)
            vs_tri.add("out", blk_err(got_m[:unique], got_t[:unique]))
            vs_tri.add("u", blk_err(u[others], tu[others])[0])
            vs_tri.add("k", blk_err(k[others], tk[others])[0])
            vs_tri.add("G", (G[others] - tG[others]).abs().max().item())
        if mode == "a":
            bt.commit()
            bm.commit()
        for layer in range(bt.layers):
            vs_tri.add("state", blk_err(bm.state(layer)[0][others], bt.state(layer)[0][others])[0])
        acc = bt.draw_acceptance()
        bm.draw_acceptance()
        bt.accept(acc)
        bm.accept(acc)
    torch.cuda.synchronize()
    ok = not fails
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name} mode={mode} cfg={config} wraps={wraps}\n"
        f"     vs reference (max ulp of block absmax / rel-RMS ulp; G abs): {vs_ref.fmt()}\n"
        f"     vs Triton: {vs_tri.fmt()}\n"
        f"     (Triton vs reference: {tri_ref.fmt()})",
        flush=True,
    )
    for f in fails[:4]:
        print("       ", f)
    return ok


def run_negative(config=None, n=8, t=4, seed=99):
    """Rows with a negative slot: nothing read or written for them, output rows untouched."""
    spec = dict(path="small", n=n, t=t, steps=1, layers=1, state_dtype=BF)
    bm = trep._Bench(spec, seed, exact=False)
    sidx = bm.state_indices.clone()
    dead = [1, n - 2]
    sidx[dead] = -1
    ok = True
    for mode in ("b", "a"):
        act = bm.inputs()
        kw = bm.kwargs(0, act)
        sentinel = torch.full((n, t, trep.HV, trep.V), 3.0, dtype=BF, device="cuda")
        kw.update(state_indices=sidx, output=sentinel.clone())
        if mode == "a":
            # every row listed (row, slot, pending count, half), dead rows with their negative slot
            sl = sidx.long()
            safe = sl.clamp(min=0)
            wi = (
                torch.stack(
                    [torch.arange(n, device="cuda"), sl, bm.pnat[safe].long(), bm.buf[safe].long()],
                    1,
                )
                .to(torch.int32)
                .flip(0)
                .contiguous()
            )
            nw = torch.zeros(1, dtype=torch.int32, device="cuda")
            kw.update(replay_work_items=wi, n_writes=nw, use_all_layer_commit=True)
        ref = trep.op.reference(**kw)
        got = cr.cached_replay_update(**kw, config=config)
        try:
            trep._compare((got, *bm.state(0)), ref)
            assert torch.equal(got[dead], sentinel[dead]), "dead rows written"
        except AssertionError as e:
            ok = False
            print("   negative-slot FAIL", mode, str(e)[:200])
    print(f"[{'PASS' if ok else 'FAIL'}] negative slots cfg={config}", flush=True)
    return ok


def run_graph(config=None, n=32, t=4, mode="b", pdl=False):
    """Captured once, replayed on new activations with acceptance in between: bitwise the eager
    calls on a twin state (the op test's CUDA-graph check, bf16 state, both commit modes)."""
    spec = dict(path="all_layer", n=n, t=t, layers=2, slots=40, state_dtype=BF)
    seed = 7000 + 10 * n + t
    eager, graph, warm = (trep._Bench(spec, seed, exact=False) for _ in range(3))
    acts = [warm.inputs() for _ in range(2)]

    def step(b):
        b.prepare()
        outs = []
        for layer in range(2):
            kw = b.kwargs(layer, acts[layer])
            kw.update(launch_with_pdl=pdl)
            outs.append(
                cr.cached_replay_update(**kw, in_kernel_commit=(mode == "b"), config=config)
            )
        if mode == "a":
            b.commit()
        return outs

    step(warm)
    torch.cuda.synchronize()
    captured = torch.cuda.CUDAGraph()
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        step(warm)
    torch.cuda.current_stream().wait_stream(s_)
    torch.cuda.synchronize()
    with torch.cuda.graph(captured):
        outs_graph = step(graph)
    refill = torch.Generator(device="cuda").manual_seed(seed + 1)
    ok = True
    for _ in range(3):
        for act in acts:
            for key in ("packed", "a", "b"):
                act[key].copy_(
                    torch.randn(
                        act[key].shape, generator=refill, device="cuda", dtype=act[key].dtype
                    )
                )
        outs_eager = step(eager)
        captured.replay()
        torch.cuda.synchronize()
        for e, g in zip(outs_eager, outs_graph):
            ok &= torch.equal(e, g)
        for name in ("backing", "old_x", "old_B", "old_dt", "pnat", "buf"):
            ok &= torch.equal(getattr(eager, name), getattr(graph, name))
        accepted = eager.draw_acceptance()
        eager.accept(accepted)
        graph.accept(accepted)
    print(
        f"[{'PASS' if ok else 'FAIL'}] CUDA graph replay bitwise eager mode={mode} pdl={pdl} cfg={config}",
        flush=True,
    )
    return ok


CELLS = {
    "small_n4_t8_full": dict(path="small", n=4, t=8, start="fresh", accept="full", steps=8),
    "small_n8_t3_dup": dict(path="small", n=8, t=3, dup=(5, 6, 7), steps=6),
    "all_n32_t8_full": dict(path="all_layer", n=32, t=8, start="fresh", accept="full", steps=8),
    "all_n16_t3_L8_S461": dict(
        path="all_layer", n=16, t=3, layers=8, slots=461, edge_slots=True, steps=3
    ),
    "t4_n1": dict(path="small", n=1, t=4, steps=6),
    "t4_n4": dict(path="small", n=4, t=4, steps=6),
    "t4_n16": dict(path="all_layer", n=16, t=4, steps=5),
    "t4_n64": dict(path="all_layer", n=64, t=4, steps=4, slots=96),
    "t4_n256": dict(path="all_layer", n=256, t=4, steps=3, slots=300, layers=1),
    "t2_n8": dict(path="small", n=8, t=2, steps=6),
    "t5_n8": dict(path="small", n=8, t=5, steps=6),
    "t4_n32_dup": dict(path="all_layer", n=32, t=4, dup=(28, 29, 30, 31), steps=4),
}


_RUNS = [
    (name, mode)
    for name, spec in CELLS.items()
    for mode in ("a", "b")
    if not (mode == "a" and spec["path"] == "small")
]


@pytest.mark.parametrize("name,mode", _RUNS, ids=[f"{n}-{m}" for n, m in _RUNS])
def test_steps_match_the_reference(name, mode) -> None:
    assert run_cell(name, CELLS[name], mode)


@pytest.mark.parametrize(
    "cell",
    cr.gdn_cached_replay_cute.CELLS,
    ids=[c.why[:40] for c in cr.gdn_cached_replay_cute.CELLS],
)
def test_certified_cells(cell) -> None:
    assert run_cell(cell.why, cell.spec, "b")


def test_negative_slots() -> None:
    assert run_negative()


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("mode", ["a", "b"])
def test_graph_replay_matches_eager(mode, pdl) -> None:
    assert run_graph(mode=mode, pdl=pdl)
