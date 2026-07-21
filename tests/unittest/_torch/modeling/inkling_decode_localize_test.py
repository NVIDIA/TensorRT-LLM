#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit7 localizer: WHERE does the runtime DECODE path diverge from the validated
stateless PREFILL path, per layer and per sub-block.

Motivation
----------
crit8 proved pool-prefill == stateless-prefill (cos 1.0) and the DENSE-only decode
== stateless (cos 0.9998), but the full stacked-MoE decode drops to cos ~0.80 and
crit6's POS1 decode diagnostic shows a few prompts fork. This test isolates the
first divergent (layer, sub-block) so the fix targets the right module rather than
guessing.

Method (reuses the crit8 runtime-state harness)
-----------------------------------------------
Build the real reduced model (6 layers: 0/1 dense-local, 2-4 MoE-local, 5
MoE-global) at TP=1, load real NVFP4 weights. For a fixed input:
  * STATELESS reference: one whole-model prefill of N tokens (the crit4/5-validated
    path). Per-layer forward hooks capture, at every position, the raw
    post-attention residual ``h_attn`` (pre-``mlp_norm`` input), the MLP/MoE output
    ``moe_out`` (``mlp`` module output, pre-sconv), and the final layer output.
  * DECODE: pool-prefill P tokens, then step-decode P..N-1 through the fused
    runtime conv pool + paged KVCacheManagerV2, capturing the same three per layer
    at each decode step.

The FIRST decode step (position P) is the clean isolator: its input (the token
embedding at P) is identical to the stateless run and NOTHING has accumulated yet,
so any per-layer divergence there is a PURE decode-kernel-vs-prefill-kernel
difference (KV write/read or attention/MoE decode math), not compounded drift.
Per layer we then attribute the divergence to the attention block (``h_attn``
diverged) or the MLP/MoE block (``h_attn`` matched but ``moe_out`` diverged).

Run (single GPU, needs the TRTLLM CUDA extensions + the checkpoint):
    python tests/unittest/_torch/modeling/inkling_decode_localize_test.py
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
N_TOKENS = int(os.environ.get("INKLING_LOC_N", "12"))
P_PREFILL = int(os.environ.get("INKLING_LOC_P", "8"))


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

    assert torch.cuda.is_available(), "decode localizer needs a CUDA GPU"
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
    head_dim = tc.head_dim
    kv_list = tc.num_kv_heads_per_layer()[:N_LAYERS]
    N, P = N_TOKENS, P_PREFILL

    g = torch.Generator(device="cpu").manual_seed(3)
    x_embeds = torch.randn(N, tc.hidden_size, generator=g).to(device).bfloat16()
    pos_all = torch.arange(N, device=device, dtype=torch.int32)
    print(f"[loc] N={N} P={P} layers={kinds} kv_heads={kv_list} "
          f"head_dim={head_dim}", flush=True)

    # --- Per-layer sub-block capture via forward hooks (no model edits). ---
    store = {}

    def mk_hooks(tag):
        store[tag] = {"h_attn": {}, "moe_out": {}, "layer_out": {}}
        handles = []
        for i, layer in enumerate(inner.layers):
            def pre(_m, args, _i=i):
                # input to mlp_norm == raw post-attention residual h_attn
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

    # --- STATELESS reference prefill (validated path). ---
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
            # Prefill hooks fire but we only compare the decode steps below; drop
            # the prefill-phase captures so index 0 of each dec list is step P.
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

    # --- Compare per layer, per sub-block, at each decode step (step 0 = the
    #     clean, accumulation-free isolator at position P). ---
    n_steps = N - P
    print(f"\n[loc] per-layer decode-vs-stateless (step j -> position {P}+j):",
          flush=True)
    first_bad = None
    for j in range(n_steps):
        pos = P + j
        print(f"  --- decode step {j} (position {pos}) ---", flush=True)
        for i in range(N_LAYERS):
            ref_h = store["ref"]["h_attn"][i][0][pos]
            dec_h = store["dec"]["h_attn"][i][j][0]
            ref_m = store["ref"]["moe_out"][i][0][pos]
            dec_m = store["dec"]["moe_out"][i][j][0]
            ref_o = store["ref"]["layer_out"][i][0][pos]
            dec_o = store["dec"]["layer_out"][i][j][0]
            hc, hm = _cos_max(ref_h, dec_h)
            mc, mm = _cos_max(ref_m, dec_m)
            oc, om = _cos_max(ref_o, dec_o)
            flag = ""
            if oc < 0.9995 and first_bad is None:
                first_bad = (j, i, "attn" if hc < 0.9995 else "mlp")
                flag = "  <== FIRST DIVERGENCE"
            print(f"    {kinds[i]:18s} h_attn(cos={hc:.6f} max={hm:.4f}) "
                  f"moe_out(cos={mc:.6f} max={mm:.4f}) "
                  f"layer_out(cos={oc:.6f} max={om:.4f}){flag}", flush=True)

    print(f"\n[loc] FIRST_DIVERGENCE={first_bad} "
          f"(step, layer, sub-block); None means decode==prefill everywhere",
          flush=True)
    # Emit a single machine-greppable summary line.
    j0 = 0
    step0 = [(_cos_max(store["ref"]["layer_out"][i][0][P],
                       store["dec"]["layer_out"][i][j0][0])[0])
             for i in range(N_LAYERS)]
    print("INKLING_DECODE_LOC step0_layer_out_cos=["
          + ",".join(f"{c:.5f}" for c in step0) + f"] first_div={first_bad}",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
