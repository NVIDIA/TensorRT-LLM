#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LONG-HORIZON decode-vs-stateless self-consistency localizer.

Why this test exists
--------------------
The fair served GSM8K gap (TRT 0.92 vs SGLang 0.98) is driven entirely by
RUNAWAY generations on hard prompts: the model reaches the right answer, then
spirals in self-doubt to the token cap and never emits the stop/transition
token, while SGLang commits at ~200-320 tokens. crit6 short-prompt logit parity
and crit8 decode-carry both PASS -- but every one of those checks runs at
N < ~24 tokens, i.e. FAR inside Inkling's 512-token sliding window. The window
never masks anything at that length, so the SWA-slide + KV-paging + conv-carry
machinery has never been exercised in the regime where the runaway actually
lives (>512-token decode crossing many windows and pages).

This localizer closes that gap. It reuses the crit7/crit8 harness (reduced real
6-layer NVFP4 model, TP=1, real per-layer geometry: local window=512, hybrid
KV heads) and drives a LONG decode so the sliding window is fully active at
every decode step:

  * STATELESS reference: one whole-model prefill of N tokens (the crit4-validated
    path). Per layer, forward hooks capture the post-attention residual h_attn,
    the MLP/MoE output moe_out, and the final layer_out at every position.
  * DECODE: pool-prefill P (> 512) tokens, then step-decode P..N-1 through the
    fused conv pool + paged KVCacheManagerV2, capturing the same three per step.

Because P > 512, EVERY decode step's local layers must window to [pos-512, pos]
and evict older KV. If the decode path windows/pages/carries differently from
the stateless prefill, cosine drops -- and the step tells us WHICH window/page
boundary and the layer/sub-block tells us WHICH module (attention vs MLP/MoE).
TP=1 removes the TP-collective confound (that is the separate B2 cuda-graph
issue); this isolates the pure SWA/paging/conv machinery.

Interpretation
--------------
  * DISAGREE (cos drops): a real TRT-internal prefill-vs-decode inconsistency in
    the long-decode machinery -> localized, fixable bug.
  * AGREE (cos ~1.0 across all windows/pages): the window/paging/conv machinery
    is self-consistent, so the runaway is NOT a TRT cache bug but an fp4
    kernel-family difference vs SGLang's flashinfer kernels (needs the
    independent windowed-attention gold as the next step, not a cache fix).

DIAGNOSTIC localization signal, not an acceptance gate.

Run (single GPU; needs the TRTLLM CUDA extensions + the checkpoint):
    INKLING_LD_N=1440 INKLING_LD_P=1024 \
    python tests/unittest/_torch/modeling/inkling_longdecode_localize_test.py
Override the checkpoint with INKLING_CHECKPOINT=/path/to/Inkling-NVFP4-full.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")

N_LAYERS = int(os.environ.get("INKLING_LOC_LAYERS", "6"))  # 0/1 dense, 2-5 MoE, 5 global
# Long horizon: P must exceed the 512 window so the slide is active at step 0.
N_TOKENS = int(os.environ.get("INKLING_LD_N", "1440"))
P_PREFILL = int(os.environ.get("INKLING_LD_P", "1024"))
SWA_WINDOW = int(os.environ.get("INKLING_LD_WINDOW", "512"))
TOKENS_PER_BLOCK = 64  # matches rs._make_ml_manager; page boundaries are multiples
COS_TOL = float(os.environ.get("INKLING_LD_TOL", "0.999"))


def _cos_max(a, b):
    import torch
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    cos = float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())
    mx = float((a - b).abs().max().item())
    return cos, mx


def main() -> int:
    import inkling_moe_replay_test as moe
    import inkling_runtime_state_test as rs
    import torch

    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingConvRuntime, InklingConvStateCache)

    assert torch.cuda.is_available(), "long-decode localizer needs a CUDA GPU"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    moe.N_LAYERS = N_LAYERS
    model, config = moe.build_reduced_model(CKPT, device)
    tc = config.pretrained_config.text_config
    inner = model.model
    dense_mlp_idx = tc.dense_mlp_idx
    local_ids = set(tc.local_layer_ids) if hasattr(tc, "local_layer_ids") else set()
    kinds = [
        f"L{i}:{'dense' if i < dense_mlp_idx else 'moe'}/"
        f"{'local' if i in local_ids else 'global'}" for i in range(N_LAYERS)
    ]
    is_local = [i in local_ids for i in range(N_LAYERS)]
    head_dim = tc.head_dim
    kv_list = tc.num_kv_heads_per_layer()[:N_LAYERS]
    N, P = N_TOKENS, P_PREFILL
    assert P > SWA_WINDOW, (
        f"P ({P}) must exceed the SWA window ({SWA_WINDOW}) so the slide is "
        "active from the first decode step")

    # Realistic-scale random embeds (self-consistency is input-agnostic; the
    # rel-bias/window structure comes from position_ids, which ARE real).
    g = torch.Generator(device="cpu").manual_seed(3)
    x_embeds = torch.randn(N, tc.hidden_size, generator=g).to(device).bfloat16()
    pos_all = torch.arange(N, device=device, dtype=torch.int32)
    print(f"[longdec] N={N} P={P} decode_steps={N - P} window={SWA_WINDOW} "
          f"tok/block={TOKENS_PER_BLOCK} layers={kinds} kv_heads={kv_list} "
          f"head_dim={head_dim} cos_tol={COS_TOL}", flush=True)

    store = {}

    def mk_hooks(tag):
        store[tag] = {"h_attn": {}, "moe_out": {}, "layer_out": {}}
        handles = []
        for i, layer in enumerate(inner.layers):
            def pre(_m, args, _i=i):
                store[tag]["h_attn"].setdefault(_i, []).append(
                    args[0].detach().float().cpu())
                return None
            def mlp_hook(_m, _in, out, _i=i):
                o = out[0] if isinstance(out, tuple) else out
                store[tag]["moe_out"].setdefault(_i, []).append(
                    o.detach().float().cpu())
            def layer_hook(_m, _in, out, _i=i):
                o = out[0] if isinstance(out, tuple) else out
                store[tag]["layer_out"].setdefault(_i, []).append(
                    o.detach().float().cpu())
            handles.append(layer.mlp_norm.register_forward_pre_hook(pre))
            handles.append(layer.mlp.register_forward_hook(mlp_hook))
            handles.append(layer.register_forward_hook(layer_hook))
        return handles

    # --- STATELESS reference prefill (validated path, the windowed gold). ---
    h_ref = mk_hooks("ref")
    mgr = rs._make_ml_manager(kv_list, head_dim, [N], device)
    rs._set_layer_offsets(inner)
    try:
        with torch.no_grad():
            md = rs._md(mgr, num_contexts=1, seq_lens=[N], num_cached=[0],
                        request_ids=[0], N=N)
            inner.forward(md, inputs_embeds=x_embeds, position_ids=pos_all,
                          conv_cache=None, conv_rt=None)
    finally:
        mgr.shutdown()
        for h in h_ref:
            h.remove()

    # --- DECODE: pool prefill P, then step-decode P..N-1. ---
    h_dec = mk_hooks("dec")
    dec_cache = InklingConvStateCache(config, max_batch_size=2, device=device)
    dec_mgr = rs._make_ml_manager(kv_list, head_dim, [N], device)
    rs._set_layer_offsets(inner)
    try:
        with torch.no_grad():
            md_p = rs._md(dec_mgr, num_contexts=1, seq_lens=[P], num_cached=[0],
                          request_ids=[0], N=N)
            rt_p = InklingConvRuntime.build(md_p, dec_cache)
            for sub in store["dec"].values():
                for lst in sub.values():
                    lst.clear()
            inner.forward(md_p, inputs_embeds=x_embeds[:P],
                          position_ids=pos_all[:P], conv_cache=dec_cache,
                          conv_rt=rt_p)
            for sub in store["dec"].values():
                for lst in sub.values():
                    lst.clear()  # discard the prefill-seed capture
            for p in range(P, N):
                md_d = rs._md(dec_mgr, num_contexts=0, seq_lens=[1],
                              num_cached=[p], request_ids=[0], N=N)
                rt_d = InklingConvRuntime.build(md_d, dec_cache)
                inner.forward(md_d, inputs_embeds=x_embeds[p:p + 1],
                              position_ids=pos_all[p:p + 1],
                              conv_cache=dec_cache, conv_rt=rt_d)
    finally:
        dec_mgr.shutdown()
        for h in h_dec:
            h.remove()

    # --- Compare per layer/sub-block at each decode step; summarize the trend. ---
    n_steps = N - P
    first_bad = None
    # Per-layer running worst so an intermittent boundary dip is not lost in noise.
    worst_layer_cos = [1.0] * N_LAYERS
    worst_layer_step = [None] * N_LAYERS
    # Coarse trend table: min layer_out cos over each 32-step window.
    print(f"\n[longdec] decode-vs-stateless trend (min layer_out cos per "
          f"32-step window; page boundaries at multiples of {TOKENS_PER_BLOCK}):",
          flush=True)
    bucket = {}
    for j in range(n_steps):
        pos = P + j
        for i in range(N_LAYERS):
            ref_h = store["ref"]["h_attn"][i][0][pos]
            dec_h = store["dec"]["h_attn"][i][j][0]
            ref_m = store["ref"]["moe_out"][i][0][pos]
            dec_m = store["dec"]["moe_out"][i][j][0]
            ref_o = store["ref"]["layer_out"][i][0][pos]
            dec_o = store["dec"]["layer_out"][i][j][0]
            hc, _ = _cos_max(ref_h, dec_h)
            mc, _ = _cos_max(ref_m, dec_m)
            oc, om = _cos_max(ref_o, dec_o)
            if oc < worst_layer_cos[i]:
                worst_layer_cos[i] = oc
                worst_layer_step[i] = (j, pos, om)
            if oc < COS_TOL and first_bad is None:
                first_bad = (j, pos, i, kinds[i], "attn" if hc < COS_TOL else "mlp")
            b = j // 32
            cur = bucket.get(b)
            if cur is None or oc < cur[0]:
                bucket[b] = (oc, i, pos)
    for b in sorted(bucket):
        oc, i, pos = bucket[b]
        crosses_page = (pos % TOKENS_PER_BLOCK) < 32
        print(f"  steps[{b*32:4d}..{b*32+31:4d}] min_layer_out_cos={oc:.6f} "
              f"@ {kinds[i]:18s} pos={pos} "
              f"{'(near page bnd)' if crosses_page else ''}", flush=True)

    print(f"\n[longdec] per-layer WORST decode-vs-stateless over the long decode:",
          flush=True)
    for i in range(N_LAYERS):
        wc = worst_layer_cos[i]
        ws = worst_layer_step[i]
        tag = "  <== LOCAL/SWA" if is_local[i] else ""
        print(f"  {kinds[i]:18s} worst_layer_out_cos={wc:.6f} "
              f"at step/pos/max_abs={ws}{tag}", flush=True)

    local_worst = min((worst_layer_cos[i] for i in range(N_LAYERS)
                       if is_local[i]), default=1.0)
    global_worst = min((worst_layer_cos[i] for i in range(N_LAYERS)
                        if not is_local[i]), default=1.0)
    overall_worst = min(worst_layer_cos)
    ok = overall_worst >= COS_TOL
    print(f"\n[longdec] FIRST_DIVERGENCE={first_bad} (step,pos,layer,kind,subblock)",
          flush=True)
    print(f"INKLING_LONGDEC N={N} P={P} window={SWA_WINDOW} "
          f"local_worst_cos={local_worst:.6f} global_worst_cos={global_worst:.6f} "
          f"overall_worst_cos={overall_worst:.6f} first_div={first_bad} "
          f"{'OK' if ok else 'DIVERGENCE'}", flush=True)
    if ok:
        print("INKLING_LONGDEC_OK  # decode==stateless across all windows/pages; "
              "long-decode machinery is self-consistent (gap is not a cache bug)",
              flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
