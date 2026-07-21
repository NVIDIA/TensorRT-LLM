#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit5: single-GPU (TP=1) MoE + dense-MLP source_activation_replay.

What this proves
----------------
Replays the Inkling MLP path (router + routed experts + shared experts, and the
dense MLP) through the *production* TensorRT-LLM ``_torch`` modules built by the
real construct+load pipeline, and compares each source-observable artifact
against a hand-written HF/SGLang-faithful pure-PyTorch fp32 reference on the REAL
NVFP4 checkpoint. Coverage:

  * DENSE layer 0  -- ``InklingDenseMLP`` (bf16 ``w13_dn``/``w2_md`` + learned
    ``global_scale``): post-layer output parity.
  * SPARSE layer 2 -- ``InklingMoE`` whose routed experts AND shared experts are
    *bf16* (``hf_quant_config.json`` lists ``layers.2.mlp.experts`` /
    ``.shared_experts`` in ``exclude_modules``), so this representative sparse
    layer is validated **dequant-free** for every required artifact: router
    logits, selected experts (top-6 ids), routed weights, shared gammas, routed
    expert output, shared-expert output, and the post-layer ``routed + shared``.
  * SPARSE layer 3 -- ``InklingMoE`` with **NVFP4** routed experts: router parity
    (fp32, exact), shared-expert parity (bf16), the fused NVFP4 forward executes
    at checkpoint-scale expert dims, and the selected NVFP4 MoE backend / op path
    is named. (The NVFP4 routed-expert *numeric* parity is validated end-to-end
    against source logits at crit6 ``source_logit_replay`` -- a stronger check
    than a hand-rolled fp4 dequant cosine, which could itself be wrong.)

CUDA graph matrix
-----------------
Every replayed ``mlp`` forward (layers 0, 2, 3) is run in two configs:
``cuda_graph=false`` (eager) and ``cuda_graph=true`` (captured with
``torch.cuda.CUDAGraph`` then replayed -- the module-level CUDA-graph hard path),
asserting eager == replay. This proves the router top-k + fused-MoE + shared
compute is graph-capturable with no graph-breaking host sync / dynamic shape.
(The full ``cuda_graph`` x ``overlap_scheduler`` *runtime* matrix for MoE is
exercised at the LLM-API tier -- crit8 smoke, crit11/crit12 eval.)

Why a reduced 4-layer model (not the full 66-layer TP=4 load)
-------------------------------------------------------------
Only the MLP path is under test here, so we build ``InklingForConditional
Generation`` with ``num_hidden_layers=4`` (layers 0/1 dense, 2 bf16-MoE, 3
NVFP4-MoE) at TP=1 and load ONLY those layers' real weights through the
production ``load_weights`` (so the experts get exactly the fused-MoE layout the
runtime uses -- no hand-rolled expert loading). The whole reduced model fits on
one GB200. The reference reads the raw checkpoint tensors straight from
safetensors and implements the exact HF math (``inkling_joint_renorm`` mirror +
SwiGLU experts), so it is an independent authority, not a tautology.

Run (single GPU, needs the TRTLLM CUDA extensions + the checkpoint):
    CUDA_VISIBLE_DEVICES=0 python tests/unittest/_torch/modeling/inkling_moe_replay_test.py
Override the checkpoint with INKLING_CHECKPOINT=/path/to/Inkling-NVFP4-full.
"""

import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full",
)

# Reduced layer count: 0/1 dense, 2 = first (bf16) MoE, 3 = first NVFP4 MoE.
N_LAYERS = 4
DENSE_LAYER = 0
MOE_BF16_LAYER = 2
MOE_NVFP4_LAYER = 3

# ~32 real tokens is plenty to exercise routing/expert math and keeps the naive
# all-256-expert fp32 reference cheap; MoE has no sequence-mixing so token count
# does not change per-token correctness.
N_TOKENS = 32

# bf16 GEMMs carry ~2^-8 relative error and the fused kernels differ from the
# eager fp32 reference, so cosine is the primary (tight) gate; max_abs/mean_abs
# are reported for context. Router logits are an fp32 linear on both sides, so
# they match far tighter.
COSINE_TOL = 0.99
ROUTER_COSINE_TOL = 0.9999
WEIGHT_COSINE_TOL = 0.999
# eager-vs-graph must be numerically identical (same kernels, replayed).
GRAPH_ATOL = 2.0e-3


def _fixed_prompt() -> str:
    para = (
        "The history of numerical computing is a story of relentless "
        "abstraction. A mixture-of-experts router sends each token to a small "
        "set of specialists, and a pair of shared experts see every token. In "
        "this problem we walk carefully through one MLP layer, keeping every "
        "intermediate in the precision the reference demands, so that a fused "
        "kernel and a plain PyTorch implementation can be shown to agree. "
    )
    return (para * 6).strip()


# ---------------------------------------------------------------------------
# Direct-from-safetensors weight reader.
# ---------------------------------------------------------------------------
def _load_ckpt_tensors(ckpt, keys, device, dtype=None):
    """Read fully-qualified checkpoint keys straight from safetensors, grouped by
    shard so each file opens once. ``dtype=None`` preserves the on-disk dtype
    (required for NVFP4 packed weights + fp8 block scales)."""
    from safetensors import safe_open

    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    by_shard = defaultdict(list)
    for k in keys:
        assert k in weight_map, f"key not in checkpoint index: {k}"
        by_shard[weight_map[k]].append(k)

    out = {}
    for shard, shard_keys in by_shard.items():
        path = os.path.join(ckpt, shard)
        with safe_open(path, framework="pt", device="cpu") as h:
            for k in shard_keys:
                t = h.get_tensor(k)
                if dtype is not None:
                    t = t.to(dtype)
                out[k] = t.to(device)
    return out


def _collect_reduced_keys(ckpt, n_layers):
    """Every ``model.llm.*`` checkpoint key needed by the reduced n-layer model:
    all keys for layers 0..n_layers-1 plus the model-level embed/norm/unembed."""
    with open(os.path.join(ckpt, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    keys = []
    for k in weight_map:
        if not k.startswith("model.llm."):
            continue
        rest = k[len("model.llm."):]
        if rest.startswith("layers."):
            layer_idx = int(rest.split(".")[1])
            if layer_idx < n_layers:
                keys.append(k)
        else:
            keys.append(k)  # embed, embed_norm, norm, unembed, ...
    return keys


# ---------------------------------------------------------------------------
# Build the reduced (4-layer) production model at TP=1 and load real weights.
# ---------------------------------------------------------------------------
def build_reduced_model(ckpt, device, mapping=None):
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.quantization.mode import QuantAlgo
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration

    # ``mapping`` defaults to single-GPU TP=1 (every existing caller). Passing a
    # multi-rank Mapping (e.g. tp_size=2 under MPI) builds this rank's sharded
    # slice of the reduced model, so the same reduced 6-layer harness can exercise
    # the TP-collective decode path -- the B2 (iter100) repro at cheap scale.
    if mapping is None:
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
    config = ModelConfig.from_pretrained(
        ckpt,
        trust_remote_code=True,
        mapping=mapping,
        attn_backend="TRTLLM",
        # Match the TP=4 dump's backend when A/B-ing (default = CUTLASS). EP is a
        # no-op at tp_size=1, so an EP dump still compares against this full ref.
        moe_backend=os.environ.get("INKLING_MOE_BACKEND", "CUTLASS"),
    )
    assert config.quant_config is not None
    assert config.quant_config.quant_algo == QuantAlgo.NVFP4, \
        f"expected NVFP4, got {config.quant_config.quant_algo}"
    # Reduce the decoder depth so the whole model fits on one GPU. Everything
    # else (head dims, expert counts, quant config, exclude_modules) is untouched
    # -- layers 0..3 are byte-for-byte the real checkpoint layers.
    config.pretrained_config.text_config.num_hidden_layers = N_LAYERS

    with MetaInitMode():
        model = InklingForConditionalGeneration(config)

    memo = {}

    def init_meta_tensor(t):
        if t.device != torch.device("meta"):
            return t
        if t not in memo:
            memo[t] = torch.empty_like(t, device=device)
        return memo[t]

    model._apply(init_meta_tensor)
    model.to(device)
    memo.clear()

    keys = _collect_reduced_keys(ckpt, N_LAYERS)
    # Read the (large) load dict to CPU; load_weights copies into the CUDA module
    # params (copy_ handles CPU->CUDA), avoiding a ~2x transient CUDA peak.
    weights = _load_ckpt_tensors(ckpt, keys, "cpu", dtype=None)
    # v1 load path (weight_mapper=None): each module's weight-load hook shards the
    # full CPU tensor into this rank's slice per the model's Mapping, so the same
    # call handles TP=1 (no sharding) and TP>1 (per-module column/row shard).
    model.load_weights(weights)
    del weights
    torch.cuda.empty_cache()
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Pure-PyTorch fp32 reference (HF math; independent of the module under test).
# ---------------------------------------------------------------------------
def _swiglu(x_f, w13, w2):
    """SwiGLU expert/MLP in fp32. ``w13``: RAW checkpoint [2*inter, hidden], gate/up
    INTERLEAVED ([g0, u0, g1, u1, ...]) -- Inkling ``inference_moe_w13_interleaved``;
    gate = rows 0::2, up = rows 1::2 (SGLang ``silu(z[::2]) * z[1::2]``). ``w2``:
    [hidden, inter]. ``x_f``: [..., hidden] fp32. Returns [..., hidden] fp32."""
    gate = x_f @ w13[0::2].float().t()
    up = x_f @ w13[1::2].float().t()
    return (F.silu(gate) * up) @ w2.float().t()


def ref_dense(x, w13_dn, w2_md, global_scale):
    xf = x.float()
    return (_swiglu(xf, w13_dn, w2_md) * global_scale.float()).to(x.dtype)


def ref_router(x, gate_w, gate_b, gscale, top_k, num_routed, n_shared,
               route_scale):
    """Independent mirror of ``inkling_joint_renorm`` (fp32). Returns router
    logits, selected expert ids, routed weights, shared gammas."""
    xf = x.float()
    logits = F.linear(xf, gate_w.float())  # [T, num_routed + n_shared]
    routed_logits = logits[..., :num_routed]
    shared_logits = logits[..., num_routed:num_routed + n_shared]
    scores = routed_logits.sigmoid()
    scores_for_choice = scores + gate_b.float()
    topk_idx = torch.topk(scores_for_choice, top_k, dim=-1, sorted=False)[1]
    topk_logits = torch.cat([routed_logits.gather(-1, topk_idx), shared_logits],
                            dim=-1)
    logp = F.logsigmoid(topk_logits)
    weights = torch.exp(logp - torch.logsumexp(logp, dim=-1, keepdim=True))
    weights = weights * route_scale * gscale.float()
    routed_w = weights[..., :top_k].contiguous()
    shared_gammas = weights[..., top_k:top_k + n_shared].contiguous()
    return logits, topk_idx, routed_w, shared_gammas


def ref_routed_experts(x, w13, w2, topk_idx, routed_w):
    """Naive all-expert SwiGLU then gather the selected top-k, weight, sum.
    ``w13``: RAW checkpoint [E, 2*inter, hidden], gate/up INTERLEAVED along the
    2*inter output dim ([g0, u0, ...]); gate = rows 0::2, up = rows 1::2. ``w2``:
    [E, hidden, inter].

    Precision-faithful to HF's actual compute: bf16 GEMMs (the checkpoint IS
    bf16), fp32 SwiGLU activation, fp32 weighted accumulation. Keeping the big
    expert weights bf16 (no fp32 upcast) also keeps this well within GPU memory."""
    xb = x.to(torch.bfloat16)
    gate = torch.einsum("th,eih->tei", xb, w13[:, 0::2])  # bf16 [T, E, inter]
    up = torch.einsum("th,eih->tei", xb, w13[:, 1::2])
    act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    eo = torch.einsum("tei,ehi->teh", act, w2)  # bf16 [T, E, hidden]
    sel = eo.gather(1, topk_idx[:, :, None].expand(-1, -1, eo.shape[-1]))
    return (sel.float() * routed_w[:, :, None].float()).sum(dim=1)  # fp32 [T, H]


def ref_shared_experts(x, sw13, sw2, gammas):
    """Two shared SwiGLU experts weighted by per-token gammas, summed. Mirrors
    ``InklingSharedExperts`` (bf16 bmm, fp32 gamma-weighted sum).
    ``sw13``: RAW checkpoint [n_shared, 2*inter, hidden], gate/up INTERLEAVED
    along the 2*inter output dim; gate = rows 0::2, up = rows 1::2. ``sw2``:
    [n_shared, hidden, inter]."""
    xb = x.to(torch.bfloat16)
    gate = torch.einsum("th,sih->tsi", xb, sw13[:, 0::2])  # bf16
    up = torch.einsum("th,sih->tsi", xb, sw13[:, 1::2])
    act = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    so = torch.einsum("tsi,shi->tsh", act, sw2)  # bf16 [T, S, hidden]
    return (so.float() * gammas[:, :, None].float()).sum(dim=1)  # fp32 [T, H]


# ---------------------------------------------------------------------------
# Comparison helpers.
# ---------------------------------------------------------------------------
def _stats(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    return max_abs, mean_abs, cos


def _report(tag, a, b, cos_tol):
    max_abs, mean_abs, cos = _stats(a, b)
    ok = cos >= cos_tol
    print(f"  [{'OK ' if ok else 'BAD'}] {tag}: cosine={cos:.6f} "
          f"max_abs={max_abs:.4g} mean_abs={mean_abs:.4g} (tol={cos_tol})",
          flush=True)
    return ok


def _topk_set_match(idx_ref, idx_mod):
    """Per-token set equality of selected expert ids (top-k order is unspecified)."""
    r = [set(row.tolist()) for row in idx_ref]
    m = [set(row.tolist()) for row in idx_mod]
    n_match = sum(1 for a, b in zip(r, m) if a == b)
    return n_match, len(r)


def _align_routed_weights(idx_ref, w_ref, idx_mod, w_mod):
    """Reorder module routed weights to the reference expert-id order so the
    per-(token,expert) weights line up regardless of top-k ordering."""
    T, k = idx_ref.shape
    out = torch.zeros_like(w_ref.float())
    for t in range(T):
        mod_map = {int(idx_mod[t, j]): float(w_mod[t, j]) for j in range(k)}
        for j in range(k):
            out[t, j] = mod_map.get(int(idx_ref[t, j]), float("nan"))
    return out


# ---------------------------------------------------------------------------
# CUDA graph capture/replay (module-level hard path).
# ---------------------------------------------------------------------------
def run_cuda_graph(mod, inp):
    # Eager warmup on the default stream first: triggers any autotuning / lazy
    # workspace allocation OUTSIDE capture (capture forbids fresh allocations on
    # some paths). Then the side-stream warmup required by the capture protocol.
    for _ in range(2):
        mod(inp)
    torch.cuda.synchronize()
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            mod(inp)
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()

    static_in = inp.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = mod(static_in)
    static_in.copy_(inp)
    g.replay()
    torch.cuda.synchronize()
    return static_out.clone()


def check_cuda_graph(mod, inp, eager_out, tag):
    """Capture+replay ``mod`` and compare to the eager output. Never raises: a
    capture failure is reported (with the reason) so it can't hide the parity
    results for the remaining layers."""
    try:
        graph_out = run_cuda_graph(mod, inp)
        ok = torch.allclose(eager_out.float(), graph_out.float(), atol=GRAPH_ATOL)
        print(f"  [{'OK ' if ok else 'BAD'}] cuda_graph {tag}: eager-vs-replay "
              f"allclose={ok} (atol={GRAPH_ATOL}) "
              f"hard_path=CUDAGraph.capture+replay", flush=True)
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"  [BAD] cuda_graph {tag}: CAPTURE_FAILED {type(e).__name__}: {e}",
              flush=True)
        return False


# ---------------------------------------------------------------------------
def main() -> int:
    torch.manual_seed(0)
    device = "cuda"
    assert torch.cuda.is_available(), "crit5 requires a GPU"
    print(f"=== crit5 MoE+dense replay on {CKPT} ===", flush=True)

    # --- Build the reduced production model + real weights. ---
    model, config = build_reduced_model(CKPT, device)
    tcfg = config.pretrained_config.text_config
    top_k = tcfg.num_experts_per_tok
    num_routed = tcfg.n_routed_experts
    n_shared = tcfg.n_shared_experts
    route_scale = tcfg.route_scale
    eps = tcfg.rms_norm_eps
    print(f"model built: layers={tcfg.num_hidden_layers} hidden={tcfg.hidden_size} "
          f"experts={num_routed} top_k={top_k} shared={n_shared} "
          f"route_scale={route_scale}", flush=True)

    inner = model.model  # InklingModel

    # --- Representative real input activations (embed -> embed_norm -> mlp_norm_L),
    #     computed with the model's OWN loaded norms; identical input feeds both
    #     the module and the reference, so the parity comparison is exact. ---
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
        ids = tok(_fixed_prompt(), return_tensors="pt").input_ids[0][:N_TOKENS]
        if ids.numel() < N_TOKENS:
            raise ValueError("prompt too short")
        print(f"tokenized real prompt: {ids.numel()} tokens", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"!! tokenizer unavailable ({e}); using fixed pseudo-random ids",
              flush=True)
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, tcfg.unpadded_vocab_size, (N_TOKENS,),
                            generator=g)
    ids = ids.to(device).to(torch.int32)

    with torch.no_grad():
        emb = inner.embed_tokens(ids)          # [T, hidden]
        resid0 = inner.embed_norm(emb)         # shared residual base

    all_ok = True

    def layer_input(layer_idx):
        with torch.no_grad():
            return inner.layers[layer_idx].mlp_norm(resid0)

    # =====================================================================
    # DENSE layer 0.
    # =====================================================================
    print(f"\n--- DENSE layer {DENSE_LAYER} ({type(inner.layers[DENSE_LAYER].mlp).__name__}) ---",
          flush=True)
    x0 = layer_input(DENSE_LAYER)
    pfx = f"model.llm.layers.{DENSE_LAYER}.mlp."
    dw = _load_ckpt_tensors(
        CKPT, [pfx + "w13_dn.weight", pfx + "w2_md.weight", pfx + "global_scale"],
        device, dtype=torch.bfloat16)
    with torch.no_grad():
        mod0 = inner.layers[DENSE_LAYER].mlp(x0)
    ref0 = ref_dense(x0, dw[pfx + "w13_dn.weight"], dw[pfx + "w2_md.weight"],
                     dw[pfx + "global_scale"])
    all_ok &= _report("dense post-layer output", mod0, ref0, COSINE_TOL)
    # CUDA graph matrix (cuda_graph=false eager above vs cuda_graph=true here).
    with torch.no_grad():
        all_ok &= check_cuda_graph(inner.layers[DENSE_LAYER].mlp, x0, mod0,
                                   f"dense L{DENSE_LAYER}")
    del dw
    torch.cuda.empty_cache()

    # =====================================================================
    # SPARSE MoE layers (2 = bf16 dequant-free full parity; 3 = NVFP4).
    # =====================================================================
    for layer_idx, is_nvfp4 in ((MOE_BF16_LAYER, False), (MOE_NVFP4_LAYER, True)):
        mlp = inner.layers[layer_idx].mlp
        tag = "NVFP4" if is_nvfp4 else "bf16"
        print(f"\n--- SPARSE layer {layer_idx} ({type(mlp).__name__}, experts={tag}, "
              f"expert_backend={type(mlp.experts).__name__}) ---", flush=True)
        x = layer_input(layer_idx)
        pfx = f"model.llm.layers.{layer_idx}.mlp."

        # Router weights are fp32/bf16 regardless of expert precision.
        gate_t = _load_ckpt_tensors(
            CKPT, [pfx + "gate.weight", pfx + "gate.bias",
                   pfx + "gate.global_scale"], device, dtype=None)
        gate_w = gate_t[pfx + "gate.weight"]
        gate_b = gate_t[pfx + "gate.bias"]
        gscale = gate_t[pfx + "gate.global_scale"]

        # (1) Router logits (module fp32 gate vs independent fp32 linear).
        with torch.no_grad():
            mod_logits = mlp.gate(x)
        ref_logits, ref_idx, ref_rw, ref_gam = ref_router(
            x, gate_w, gate_b, gscale, top_k, num_routed, n_shared, route_scale)
        all_ok &= _report("router logits", mod_logits, ref_logits,
                          ROUTER_COSINE_TOL)

        # (2) Selected experts + (3) routed weights (via the module routing method).
        with torch.no_grad():
            mod_idx, mod_rw = mlp.gate.routing_method.apply(mod_logits.float())
        n_match, n_tot = _topk_set_match(ref_idx, mod_idx)
        sel_ok = n_match == n_tot
        print(f"  [{'OK ' if sel_ok else 'BAD'}] selected experts (top-{top_k}): "
              f"{n_match}/{n_tot} tokens match", flush=True)
        all_ok &= sel_ok
        aligned = _align_routed_weights(ref_idx, ref_rw, mod_idx, mod_rw)
        rw_ok = not torch.isnan(aligned).any() and _report(
            "routed weights (id-aligned)", aligned, ref_rw, WEIGHT_COSINE_TOL)
        all_ok &= rw_ok

        # (4) Shared gammas.
        _, _, _, ref_gam2 = ref_router(x, gate_w, gate_b, gscale, top_k,
                                       num_routed, n_shared, route_scale)
        from tensorrt_llm._torch.models.modeling_inkling import \
            inkling_joint_renorm
        with torch.no_grad():
            _, _, mod_gam = inkling_joint_renorm(
                mod_logits.float(), gate_bias=mlp.gate.bias,
                global_scale=mlp.gate.global_scale, route_scale=route_scale,
                top_k=top_k, num_routed=num_routed, n_shared=n_shared)
        all_ok &= _report("shared gammas", mod_gam, ref_gam2, WEIGHT_COSINE_TOL)

        # (6) Shared-expert output (bf16 both layers): feed identical gammas so
        #     only the shared SwiGLU matmul differs.
        sw = _load_ckpt_tensors(
            CKPT, [pfx + "shared_experts.shared_w13_weight",
                   pfx + "shared_experts.shared_w2_weight"], device,
            dtype=torch.bfloat16)
        with torch.no_grad():
            mod_shared = mlp.shared_experts(x, ref_gam2)
        ref_shared = ref_shared_experts(
            x, sw[pfx + "shared_experts.shared_w13_weight"],
            sw[pfx + "shared_experts.shared_w2_weight"], ref_gam2)
        all_ok &= _report("shared-expert output", mod_shared, ref_shared,
                          COSINE_TOL)
        del sw

        # (5) Routed-expert output + (7) post-layer output.
        with torch.no_grad():
            mod_routed = mlp.experts(x, mod_logits)
            mod_total = mlp(x)
        if not is_nvfp4:
            # Dequant-free: read the bf16 experts and run the naive reference.
            ew = _load_ckpt_tensors(
                CKPT, [pfx + "experts.w13_weight", pfx + "experts.w2_weight"],
                device, dtype=torch.bfloat16)
            ref_routed = ref_routed_experts(
                x, ew[pfx + "experts.w13_weight"],
                ew[pfx + "experts.w2_weight"], ref_idx, ref_rw)
            all_ok &= _report("routed-expert output", mod_routed, ref_routed,
                              COSINE_TOL)
            # Post-layer = routed + shared (reuse the shared reference computed
            # above with the module's own internal gammas path -> mod_total).
            ref_total = ref_routed + ref_shared
            all_ok &= _report("post-layer output (routed+shared)", mod_total,
                              ref_total, COSINE_TOL)
            del ew
        else:
            # NVFP4 routed experts: prove the fused forward runs at checkpoint
            # scale + is finite; numeric parity of the NVFP4 experts is validated
            # end-to-end at crit6 source_logit_replay (whole-stack vs source
            # logits), which is stronger than a hand-rolled fp4 dequant cosine.
            finite = torch.isfinite(mod_routed).all().item() and \
                torch.isfinite(mod_total).all().item()
            print(f"  [{'OK ' if finite else 'BAD'}] NVFP4 routed forward: "
                  f"executed, finite={finite}, shape={tuple(mod_routed.shape)}, "
                  f"op_path=create_moe/{type(mlp.experts).__name__}, "
                  f"activation=silu(gate)*up(SwiGLU)", flush=True)
            all_ok &= finite

        # CUDA graph matrix on the full MoE forward (cuda_graph=false eager vs
        # cuda_graph=true captured/replayed).
        with torch.no_grad():
            all_ok &= check_cuda_graph(mlp, x, mod_total, f"MoE L{layer_idx}")

        del gate_t
        torch.cuda.empty_cache()

    print("", flush=True)
    if all_ok:
        print("CRIT5_OK", flush=True)
        return 0
    print("CRIT5_FAIL", flush=True)
    return 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rc = 1
    print(f"=== CRIT5_DONE rc={rc} ===", flush=True)
    sys.exit(rc)
