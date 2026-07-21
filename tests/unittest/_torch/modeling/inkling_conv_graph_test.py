#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B2 probe: is ``causal_conv1d_update`` correct under MULTI-STEP CUDA-graph replay?

Motivation (iter72 attribution)
-------------------------------
The strict generation_parity gate splits cleanly on the runtime axes:
CGONLY (cuda_graph=1, overlap=0) ~= ENABLED and OVONLY (cuda_graph=0, overlap=1)
~= BASELINE, so the enabled-row decode corruption is CUDA-GRAPH CAPTURE/REPLAY
ALONE (the overlap scheduler is benign). The served enabled smoke reproduces it
at batch=1 ('!!!!'), so it is NOT a batched-only bug.

crit4 (attention) and crit5 (MoE) already validate their decode blocks under a
CUDA graph -- but each captures the graph and replays it EXACTLY ONCE. A stateful
short-conv decode advances its per-slot conv window on EVERY step; a bug in how
the captured graph reads-then-writes that window in place would only surface on
the SECOND and later replays, once state has to carry across replays. That path
is unvalidated. This probe closes the gap with the cheapest possible signal (one
op, no model, no checkpoint): drive N decode steps through ``causal_conv1d_update``
eager, then drive the SAME N steps by capturing the op ONCE and REPLAYING it N
times against a static input buffer (the exact LLM-API cuda-graph decode pattern),
and require per-step output + final-state parity.

PART A (B2): batch=1, eager multi-step vs graph multi-step replay.
PART B (B1): batch=2 per-slot independence -- one batched update of two slots must
             equal two independent single-slot updates (the nc>1 per-slot bug).

Run (single GPU, needs the TRTLLM CUDA extensions):
    python tests/unittest/_torch/modeling/inkling_conv_graph_test.py
"""
import os
import sys


def _cos_max(a, b):
    import torch
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    cos = float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())
    mx = float((a - b).abs().max().item())
    return cos, mx


def main() -> int:
    import torch

    from tensorrt_llm._torch.modules.mamba.causal_conv1d import \
        causal_conv1d_update

    assert torch.cuda.is_available(), "conv-graph probe needs a CUDA GPU"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    C = int(os.environ.get("INKLING_CG_CHANNELS", "128"))   # channels
    K = int(os.environ.get("INKLING_CG_KERNEL", "4"))       # sconv_kernel_size
    NSTEP = int(os.environ.get("INKLING_CG_STEPS", "16"))
    kwin = K - 1
    maxb = 4                                                # pool rows (+pad)
    dt = torch.bfloat16
    # Depthwise conv weight [channels, kernel] (model passes w.squeeze(1).to(dt)).
    w = torch.randn(C, K, device=device, dtype=dt) * 0.3
    # A fixed decode stream of NSTEP one-token inputs [1, C] for a single request.
    g = torch.Generator(device="cpu").manual_seed(7)
    xs = [torch.randn(1, C, generator=g).to(device).to(dt) for _ in range(NSTEP)]

    # ---- PART A: batch=1 eager vs multi-step graph replay -------------------
    slot = 0
    idx = torch.tensor([slot], dtype=torch.int32, device=device)

    # EAGER reference: advance the per-slot conv window over NSTEP steps.
    st_e = torch.zeros(maxb, C, kwin, device=device, dtype=dt)
    eager_ys = []
    for x in xs:
        y = causal_conv1d_update(x.clone(), st_e, w, None, activation=None,
                                 conv_state_indices=idx)
        eager_ys.append(y.detach().float().cpu().clone())
    eager_state = st_e[slot].detach().float().cpu().clone()

    # GRAPH: capture the op ONCE against a static input buffer, replay it NSTEP
    # times (the LLM-API cuda-graph decode pattern -- the model clones x inside
    # forward, so the captured region also clones the static buffer).
    st_g = torch.zeros(maxb, C, kwin, device=device, dtype=dt)
    x_static = torch.zeros(1, C, device=device, dtype=dt)

    def run():
        return causal_conv1d_update(x_static.clone(), st_g, w, None,
                                    activation=None, conv_state_indices=idx)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()                      # warmup advances st_g; reset below
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_g = run()                    # capture advances st_g once; reset below

    st_g.zero_()                       # clean state so replay starts like eager
    graph_ys = []
    for x in xs:
        x_static.copy_(x)
        graph.replay()
        graph_ys.append(y_g.detach().float().cpu().clone())
    graph_state = st_g[slot].detach().float().cpu().clone()

    a_ok = True
    worst = (1.0, 0.0, -1)
    for i, (ye, yg) in enumerate(zip(eager_ys, graph_ys)):
        cos, mx = _cos_max(ye, yg)
        if cos < worst[0]:
            worst = (cos, mx, i)
        step_ok = cos > 0.9999 and mx < 5e-2
        a_ok &= step_ok
        if not step_ok or i < 2 or i == NSTEP - 1:
            print(f"  [A step {i:2d}] out cos={cos:.6f} max={mx:.4f} "
                  f"{'ok' if step_ok else '<== DIVERGE'}", flush=True)
    sc, sm = _cos_max(eager_state, graph_state)
    a_ok &= sc > 0.9999
    print(f"  [A final-state] cos={sc:.6f} max={sm:.4f} "
          f"worst_step={worst[2]}(cos={worst[0]:.6f})", flush=True)
    print(f"[conv-graph] PART_A batch1 eager-vs-graph "
          f"{'PASS' if a_ok else 'FAIL'}", flush=True)

    # ---- PART B: batch=2 per-slot independence (B1) -------------------------
    # One batched update of two slots must equal two independent single-slot
    # updates. If the per-slot in-place update leaks across slots, this fails --
    # the nc>1 batched '!!!!' collapse signature.
    idx2 = torch.tensor([1, 2], dtype=torch.int32, device=device)
    g2 = torch.Generator(device="cpu").manual_seed(11)
    xa = [torch.randn(1, C, generator=g2).to(device).to(dt) for _ in range(NSTEP)]
    xb = [torch.randn(1, C, generator=g2).to(device).to(dt) for _ in range(NSTEP)]

    # Independent single-slot references.
    def solo(stream, slotid):
        st = torch.zeros(maxb, C, kwin, device=device, dtype=dt)
        ii = torch.tensor([slotid], dtype=torch.int32, device=device)
        ys = []
        for x in stream:
            ys.append(causal_conv1d_update(
                x.clone(), st, w, None, activation=None,
                conv_state_indices=ii).detach().float().cpu().clone())
        return ys

    ref_a = solo(xa, 1)
    ref_b = solo(xb, 2)

    # Batched: both slots updated in one call per step (packed [2, C]).
    st2 = torch.zeros(maxb, C, kwin, device=device, dtype=dt)
    b_ok = True
    worstb = (1.0, 0.0, -1)
    for i in range(NSTEP):
        xin = torch.cat([xa[i], xb[i]], dim=0)          # [2, C]
        yb = causal_conv1d_update(xin.clone(), st2, w, None, activation=None,
                                  conv_state_indices=idx2)
        ya = yb[0:1].detach().float().cpu().clone()
        yb2 = yb[1:2].detach().float().cpu().clone()
        ca, ma = _cos_max(ref_a[i], ya)
        cb, mb = _cos_max(ref_b[i], yb2)
        step_ok = ca > 0.9999 and cb > 0.9999
        b_ok &= step_ok
        if min(ca, cb) < worstb[0]:
            worstb = (min(ca, cb), max(ma, mb), i)
        if not step_ok or i < 2 or i == NSTEP - 1:
            print(f"  [B step {i:2d}] slotA cos={ca:.6f} slotB cos={cb:.6f} "
                  f"{'ok' if step_ok else '<== LEAK'}", flush=True)
    print(f"  [B] worst_step={worstb[2]}(cos={worstb[0]:.6f})", flush=True)
    print(f"[conv-graph] PART_B batch2 per-slot-independence "
          f"{'PASS' if b_ok else 'FAIL'}", flush=True)

    # ---- PART C: DUPLICATE conv_state_indices under cuda-graph (the padding case)
    # Under cuda_graph=on the runtime pads a decode batch by repeating the SAME
    # dummy request (cuda_graph_runner._get_padded_batch:
    # generation_requests.extend([dummy]*padding_size)), so slots_for maps every
    # padding row to ONE shared dummy slot -- the batched causal_conv1d_update then
    # gets conv_state_indices with DUPLICATES ([real, dummy, dummy, ...]). This is
    # cuda-graph-specific (padding only happens under graph), matching B2. Probe:
    # does the UNIQUE real row (index 0) stay correct under MULTI-STEP graph replay
    # while later rows share a slot and race each other's in-place state writes?
    B = 4
    real_slot, dummy_slot = 0, 3
    idxdup = torch.tensor([real_slot] + [dummy_slot] * (B - 1),
                          dtype=torch.int32, device=device)
    # eager batch=1 reference for the real stream on its own unique slot.
    ref_real = solo(xs, real_slot)
    g3 = torch.Generator(device="cpu").manual_seed(23)
    dummy_xs = [torch.randn(B - 1, C, generator=g3).to(device).to(dt)
                for _ in range(NSTEP)]

    stC = torch.zeros(maxb, C, kwin, device=device, dtype=dt)
    xC = torch.zeros(B, C, device=device, dtype=dt)

    def runC():
        return causal_conv1d_update(xC.clone(), stC, w, None, activation=None,
                                    conv_state_indices=idxdup)

    sideC = torch.cuda.Stream()
    sideC.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sideC):
        for _ in range(3):
            runC()
    torch.cuda.current_stream().wait_stream(sideC)
    graphC = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graphC):
        yC = runC()
    stC.zero_()
    c_ok = True
    worstc = (1.0, 0.0, -1)
    for i in range(NSTEP):
        xC[0].copy_(xs[i][0])
        xC[1:].copy_(dummy_xs[i])
        graphC.replay()
        y0 = yC[0:1].detach().float().cpu().clone()
        cc, cm = _cos_max(ref_real[i], y0)
        step_ok = cc > 0.9999 and cm < 5e-2
        c_ok &= step_ok
        if cc < worstc[0]:
            worstc = (cc, cm, i)
        if not step_ok or i < 2 or i == NSTEP - 1:
            print(f"  [C step {i:2d}] real-row(idx0) cos={cc:.6f} max={cm:.4f} "
                  f"{'ok' if step_ok else '<== REAL ROW CORRUPTED'}", flush=True)
    print(f"  [C] worst_step={worstc[2]}(cos={worstc[0]:.6f})", flush=True)
    print(f"[conv-graph] PART_C dup-slot-graph real-row-integrity "
          f"{'PASS' if c_ok else 'FAIL'}", flush=True)

    ok = a_ok and b_ok and c_ok
    print(f"INKLING_CONV_GRAPH_{'OK' if ok else 'FAIL'} "
          f"partA_batch1_graph={'ok' if a_ok else 'FAIL'} "
          f"partB_batch2_slots={'ok' if b_ok else 'FAIL'} "
          f"partC_dupslot_graph={'ok' if c_ok else 'FAIL'} "
          f"channels={C} kernel={K} steps={NSTEP}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
