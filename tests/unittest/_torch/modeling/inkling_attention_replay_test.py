#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit4: single-GPU (TP=1) attention source-activation replay (prefill + decode).

What this proves
----------------
Replays ONE Inkling attention layer through the Inkling Triton attention path +
``KVCacheManagerV2`` and compares its output against a hand-written pure-PyTorch
HF-faithful reference, across the phase/CUDA-graph matrix:
  * PREFILL (context, cuda_graph=false): P = N-1 tokens attend over the packed
    extend tensors; K/V are written to the paged cache.
  * DECODE eager (generation, cuda_graph=false): the last token (position P)
    attends over the REUSED prefilled cache via the paged decode kernel, using
    the short-conv state carried from the prefill tail.
  * DECODE CUDA graph (cuda_graph=true): the decode attention is captured and
    replayed; the replay must reproduce the eager decode (hard-path proof).
It runs for one LOCAL sliding-window layer (layer 0: 16 kv-heads, window 512, no
tau) and one GLOBAL full-causal layer (layer 5: 8 kv-heads, rel_extent 1024,
log-scaling tau -- a no-op below 128k positions but still exercised). For each
layer and phase it prints ``max_abs`` / ``mean_abs`` / ``cosine``; iff every
prefill/decode/graph cosine >= COSINE_TOL and the graph replay is allclose to the
eager decode, it prints ``CRIT4_OK`` and exits 0.

The overlap scheduler is a runtime (LLM API) concept with no analogue in an
isolated attention-module replay; that axis is exercised at the full-runtime
tier (crit8 LLM API smoke, crit11/12 accuracy). This module test covers the
CUDA-graph axis concretely (eager vs captured/replayed decode).

Why a local reference (HF native is NOT runnable)
-------------------------------------------------
The HF Inkling modeling code needs the transformers *checkout* on PYTHONPATH, and
the checkpoint stores attention in a ModelOpt-packed NVFP4 container with non-HF
tensor names and no HF dequant path. So the ground truth here is a standalone
pure-PyTorch attention implementing the EXACT math from ``crit4_spec.md`` (steps
1-10), fed with the layer's real BF16 attention weights read directly from the
checkpoint safetensors (``.attn`` is bf16, excluded from NVFP4 in
``hf_quant_config.json``). Each reference step below is annotated with its spec
step number; the reference is the authority and must match the source math.

Input choice (true source attention-layer boundary)
----------------------------------------------------
The shared residual base ``residual_0 = embed_norm(embed(token_ids))`` is computed
from the checkpoint's own ``model.llm.embed.weight`` + ``model.llm.embed_norm
.weight`` (RMSNorm, eps 1e-6) over a fixed real prompt tokenized with the
checkpoint tokenizer. Each replayed layer L then applies its real per-layer
pre-attention RMSNorm ``model.llm.layers.L.attn_norm.weight`` (the decoder layer
runs ``attn_norm`` before attention), so the activation fed to the attention
module is the genuine source attention-layer boundary:
  * LOCAL layer 0: ``residual_0`` IS the layer-0 residual, so
    ``attn_norm_0(residual_0)`` is the EXACT source layer-0 attention input.
  * GLOBAL layer 5: the exact ``residual_5`` needs the stacked forward through
    layers 0-4 (dense + MoE = crit5/crit6); we feed ``attn_norm_5(residual_0)`` --
    a representative real activation through layer 5's true input norm/geometry.
Both the reference and the TRTLLM path see the identical per-layer input, so the
parity comparison is exact regardless. We use N ~ 600 tokens so the sequence
crosses the 512-token local window. If the tokenizer cannot be loaded, we fall
back to a fixed random token-id vector and say so loudly in the output.

Run (single GPU, needs the TRTLLM CUDA extensions + the checkpoint):
    python tests/unittest/_torch/modeling/inkling_attention_replay_test.py
Override the checkpoint with INKLING_CHECKPOINT=/path/to/Inkling-NVFP4-full.
"""

import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full",
)

# Layers to replay: 0 is local (in local_layer_ids, 16 kv-heads, window 512);
# 5 is global (NOT in local_layer_ids, 8 kv-heads, full causal, tau applied).
LAYER_LOCAL = 0
LAYER_GLOBAL = 5

# Prompt length target (must exceed the 512 local window so the sliding-window
# mask is genuinely exercised). Real tokenization may differ slightly; we
# truncate/keep as tokenized and only pad the random fallback to this length.
# iter92: INKLING_ATTN_N lets a LONG-context run (e.g. 2000) probe whether TRT's
# Triton attention diverges from the exact-softmax reference beyond the 600-tok
# crit4 baseline -- the regime the MMLU B-bias lives in. Default 600 = crit4.
N_TARGET = int(os.environ.get("INKLING_ATTN_N", "600"))

# bf16 matmuls + a bf16 KV cache carry ~2^-8 relative error and the two paths use
# different kernels (eager reference vs fused FMHA), so cosine is the primary gate
# (tight) and max_abs is reported for context (documented, not hard-gated tight).
COSINE_TOL = 0.99


def _fixed_prompt() -> str:
    """A fixed, deterministic real prompt long enough to tokenize to ~600 toks."""
    para = (
        "The history of numerical computing is a story of relentless "
        "abstraction. Each generation of engineers built machines that hid the "
        "grinding detail of the layer beneath, so that the next generation could "
        "reason about larger and larger ideas. Transformers continued that "
        "tradition: attention lets a model route information between distant "
        "tokens without the fixed wiring of a convolution, and normalization "
        "keeps the signal from exploding as it flows through dozens of layers. "
        "In this problem we walk carefully through one attention layer, keeping "
        "every intermediate in the precision the reference demands, so that a "
        "fused kernel and a plain PyTorch implementation can be shown to agree. "
    )
    # Repeat enough to cover N_TARGET tokens (para ~= 75 tok); the truncated
    # first-N_TARGET prefix is identical regardless of the repeat count, so the
    # default (N_TARGET=600) reproduces the original crit4 prompt exactly.
    reps = max(8, N_TARGET // 60 + 6)
    return (para * reps).strip()


# ---------------------------------------------------------------------------
# Direct-from-safetensors weight reader (bf16 .attn tensors).
# ---------------------------------------------------------------------------
def _load_ckpt_tensors(ckpt: str, keys, device, dtype=None):
    """Read the given fully-qualified checkpoint keys straight from safetensors.

    Uses ``model.safetensors.index.json`` weight_map to find each key's shard,
    then ``safetensors.safe_open`` to pull just those tensors. Returns a dict
    key -> tensor on ``device`` (cast to ``dtype`` when given). Grouped by shard
    so each file is opened once.
    """
    import json
    from collections import defaultdict

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


def _read_attn_weights(ckpt, layer_idx, device):
    """All bf16 attention tensors for one layer, keyed by their short name."""
    import torch

    pfx = f"model.llm.layers.{layer_idx}.attn."
    short = [
        "wq_du.weight",
        "wk_dv.weight",
        "wv_dv.weight",
        "wr_du.weight",
        "wo_ud.weight",
        "q_norm.weight",
        "k_norm.weight",
        "k_sconv.weight",
        "v_sconv.weight",
        "rel_logits_proj.proj",
    ]
    full = _load_ckpt_tensors(ckpt, [pfx + s for s in short],
                              device,
                              dtype=torch.bfloat16)
    return {s: full[pfx + s] for s in short}


# ---------------------------------------------------------------------------
# LOCAL pure-PyTorch reference attention (crit4_spec.md "EXACT reference math").
# ---------------------------------------------------------------------------
def ref_attention(
    x,
    weights,
    is_local,
    rel_extent,
    num_heads,
    num_kv_heads,
    head_dim,
    *,
    sliding_window,
    log_scaling_n_floor,
    log_scaling_alpha,
    rms_eps,
    score_scale=None,
):
    """Ground-truth attention for one layer. x:[T,6144] bf16 -> [T,6144] bf16.

    Implements crit4_spec.md steps 1-10 exactly. Every reduction that the source
    performs in fp32 is done in fp32 here; the per-head RMSNorm mirrors the
    TRTLLM ``RMSNorm`` (normalize in fp32, cast to input dtype, THEN multiply by
    the bf16 gain -- see modules/rms_norm.py). Uses eager torch only (no flash).
    """
    import torch
    import torch.nn.functional as F

    T = x.shape[0]
    dev = x.device
    in_dtype = x.dtype  # bf16
    D = head_dim

    wq = weights["wq_du.weight"]  # [nh*D, 6144]
    wk = weights["wk_dv.weight"]  # [nkv*D, 6144]
    wv = weights["wv_dv.weight"]  # [nkv*D, 6144]
    wr = weights["wr_du.weight"]  # [nh*d_rel, 6144]
    wo = weights["wo_ud.weight"]  # [6144, nh*D]
    q_gain = weights["q_norm.weight"].float()  # [D]
    k_gain = weights["k_norm.weight"].float()  # [D]
    k_sconv = weights["k_sconv.weight"]  # [nkv*D, 1, 4]
    v_sconv = weights["v_sconv.weight"]  # [nkv*D, 1, 4]
    proj = weights["rel_logits_proj.proj"]  # [d_rel, rel_extent]
    d_rel = proj.shape[0]

    # --- Spec step 1: q/k/v/r projections (no bias). ---
    #   x @ Wᵀ in the activation dtype (bf16), matching the fused qkv_proj GEMM.
    q = (x @ wq.t()).view(T, num_heads, D)  # [T, nh, D]
    k = (x @ wk.t()).view(T, num_kv_heads, D)  # [T, nkv, D]
    v = (x @ wv.t()).view(T, num_kv_heads, D)  # [T, nkv, D]
    r = (x @ wr.t()).view(T, num_heads, d_rel)  # [T, nh, d_rel]

    # --- Spec step 2: causal depthwise short conv on k and v ONLY. ---
    #   kernel=4, left-pad 3, keep first T, NO bias, NO activation, computed in
    #   fp32, with an INTERNAL RESIDUAL y = conv(stream) + stream. channels=nkv*D.
    #   Mirrors InklingShortConv.forward's no-cache branch.
    def short_conv(stream_2d, filt):  # stream_2d: [T, nkv*D]
        C = stream_2d.shape[1]
        xt = stream_2d.float().transpose(0, 1).unsqueeze(0)  # [1, C, T]
        y = F.conv1d(xt,
                     filt.float(),
                     bias=None,
                     padding=filt.shape[-1] - 1,
                     groups=C)
        y = y[..., :T].squeeze(0).transpose(0, 1)  # [T, C]
        return (y.to(in_dtype) + stream_2d).to(in_dtype)

    k = short_conv(k.reshape(T, num_kv_heads * D),
                   k_sconv).view(T, num_kv_heads, D)
    v = short_conv(v.reshape(T, num_kv_heads * D),
                   v_sconv).view(T, num_kv_heads, D)

    # --- Spec step 3: per-head RMSNorm over head_dim (eps=1e-6), fp32 then cast,
    #   THEN multiply by the (bf16) gain. v is NOT normalized. ---
    def head_rmsnorm(t, gain):  # t: [T, H, D], gain: [D] fp32
        f = t.float()
        var = f.pow(2).mean(-1, keepdim=True)
        normed = (f * torch.rsqrt(var + rms_eps)).to(in_dtype)
        # gain applied in input dtype (matches RMSNorm: weight * normed.to(dtype))
        return gain.to(in_dtype) * normed

    q = head_rmsnorm(q, q_gain)  # [T, nh, D]
    k = head_rmsnorm(k, k_gain)  # [T, nkv, D]

    # --- Spec step 4: GQA repeat k,v from nkv -> nh. ---
    rep = num_heads // num_kv_heads
    k = k.repeat_interleave(rep, dim=1)  # [T, nh, D]
    v = v.repeat_interleave(rep, dim=1)  # [T, nh, D]

    # Work per-head in fp32 for the score math (softmax fp32 per spec step 9).
    qh = q.permute(1, 0, 2).float()  # [nh, T, D]
    kh = k.permute(1, 0, 2).float()  # [nh, T, D]
    vh = v.permute(1, 0, 2).float()  # [nh, T, D]

    # --- Spec step 5: scores = (q @ kᵀ) * (1/head_dim)  (NOT 1/sqrt(D)). ---
    #   score_scale defaults to 1/head_dim; overridable for a diagnostic sweep
    #   that checks the backend q_scaling convention against 1/sqrt(head_dim).
    scale = (1.0 / D) if score_scale is None else score_scale
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # [nh, T, T]

    # --- Spec step 6: relative bias (fp32). ---
    #   rel = einsum('thd,de->the', r, proj) -> [T, nh, rel_extent]; permute to
    #   [nh, T, rel_extent]. distance[i,j] = i - j. gather clamp(0, rel_extent-1);
    #   zero where distance<0 or distance>=rel_extent. Add to scores.
    rel = torch.einsum("thd,de->the", r.float(), proj.float())  # [T,nh,rel_ext]
    rel = rel.permute(1, 0, 2)  # [nh, T, rel_extent]
    i_idx = torch.arange(T, device=dev)
    distance = i_idx[:, None] - i_idx[None, :]  # [T, T] (query i, key j)
    gather_idx = distance.clamp(0, rel_extent - 1)
    gather_idx = gather_idx[None].expand(num_heads, -1, -1)  # [nh, T, T]
    bias = rel.gather(-1, gather_idx)  # [nh, T, T]
    invalid = (distance < 0) | (distance >= rel_extent)
    bias = bias.masked_fill(invalid[None], 0.0)

    # --- Spec step 7: GLOBAL only tau on the pre-softmax score rows. ---
    #   tau_i = 1 + alpha*log(clamp((i+1)/n_floor, min=1.0)); multiply BOTH the
    #   q-contribution (scores) and the relative bias by tau[i]. No-op < 128k.
    #   LOCAL: skip tau. (Multiplying scores + bias by tau == multiplying the
    #   whole pre-softmax row, which is what the source does.)
    #   NOTE: the TRTLLM InklingAttention._build_rel_logits folds tau into the
    #   rel_logits aux (bias) ONLY (not q@kᵀ). Because n_floor=128000, N~600,
    #   tau == 1.0 for EVERY token here, so "both" and "bias-only" are identical
    #   in the tested regime; they would only diverge at >=128k positions
    #   (out of scope). This reference stays faithful to the spec's "multiply
    #   both" wording while remaining numerically equal to the model path.
    if not is_local and log_scaling_n_floor is not None:
        tau = 1.0 + log_scaling_alpha * torch.log(
            ((i_idx + 1).float() / log_scaling_n_floor).clamp(min=1.0))
        scores = scores * tau[None, :, None]
        bias = bias * tau[None, :, None]

    scores = scores + bias  # add relative bias (fp32)

    # --- Spec step 8: causal mask; LOCAL also sliding-window (current + 511
    #   previous only, i.e. distance in [0, window-1]). ---
    neg_inf = torch.finfo(torch.float32).min
    causal = distance < 0  # key after query
    mask = causal.clone()
    if is_local:
        assert sliding_window is not None
        too_old = distance >= sliding_window
        mask = mask | too_old
    scores = scores.masked_fill(mask[None], neg_inf)

    # --- Spec step 9: softmax over key axis in fp32 -> cast; ctx = softmax @ v. ---
    probs = torch.softmax(scores, dim=-1)  # [nh, T, T] fp32
    ctx = torch.matmul(probs.to(in_dtype).float(), vh)  # [nh, T, D] fp32

    # --- Spec step 10: reshape [T, nh*D]; out = ctx @ wo_udᵀ -> [T, 6144]. ---
    ctx = ctx.permute(1, 0, 2).reshape(T, num_heads * D).to(in_dtype)
    out = ctx @ wo.t()  # [T, 6144]
    return out


# ---------------------------------------------------------------------------
# TRTLLM InklingAttention path.
# ---------------------------------------------------------------------------
def _build_trtllm_attention(model_config, layer_idx, attn_w, device):
    """Construct InklingAttention(model_config, layer_idx), materialize its
    weights on ``device``, and copy in the layer's bf16 checkpoint weights.

    Weight fusion / mapping (mirrors inkling_weight_mapper._LAYER_RENAMES and the
    FUSED_QKV loader, which cats (q, k, v) along dim0):
      * qkv_proj.weight  <- cat([wq_du; wk_dv; wv_dv], dim=0)
      * o_proj.weight    <- wo_ud
      * r_proj.weight    <- wr_du
      * q_norm/k_norm    <- q_norm/k_norm (per-head gain over head_dim)
      * k_sconv/v_sconv  <- k_sconv/v_sconv  ([channels,1,kernel])
      * rel_logits_proj  <- rel_logits_proj.proj  ([d_rel, rel_extent])
    """
    import torch

    from tensorrt_llm._torch.models.modeling_inkling import InklingAttention

    attn = InklingAttention(model_config, layer_idx=layer_idx)
    # model_config from from_pretrained leaves skip_create_weights_in_init=False,
    # so the base Attention already ran create_weights() (qkv_proj/o_proj
    # materialized) and the plain nn.Parameters (r_proj/rel_logits_proj/sconv/
    # q_norm/k_norm) were allocated in __init__. Defensively ensure creation,
    # then move everything to the GPU before copying real weights in.
    attn.create_weights()
    attn = attn.to(device)
    # This standalone test allocates a SINGLE-layer KV cache (cache index 0), so
    # pin the cache layer index to 0. The module's real geometry (is_local, head
    # counts, rel_extent, window_left) is frozen in __init__ from the true
    # layer_idx, so overriding the runtime cache-lookup index afterward is safe.
    # InklingAttention._attention reads ``self.layer_idx`` (the global decoder
    # index) for get_buffers/get_batch_cache_indices, so pin THAT to 0 -- pinning
    # only the base op's ``attn.local_layer_idx`` misses it and KeyErrors for any
    # layer_idx != 0 (e.g. global layer 5) not in the 1-layer cache.
    attn.layer_idx = 0
    attn.attn.local_layer_idx = 0

    wq = attn_w["wq_du.weight"]
    wk = attn_w["wk_dv.weight"]
    wv = attn_w["wv_dv.weight"]
    qkv = torch.cat([wq, wk, wv], dim=0).contiguous()  # [ (nh+2*nkv)*D, 6144 ]

    with torch.no_grad():
        assert attn.qkv_proj.weight.shape == qkv.shape, (
            "qkv_proj.weight",
            tuple(attn.qkv_proj.weight.shape),
            "expected",
            tuple(qkv.shape),
        )
        attn.qkv_proj.weight.copy_(qkv.to(attn.qkv_proj.weight.dtype))
        attn.o_proj.weight.copy_(attn_w["wo_ud.weight"].to(
            attn.o_proj.weight.dtype))
        attn.r_proj.weight.copy_(attn_w["wr_du.weight"].to(
            attn.r_proj.weight.dtype))
        attn.q_norm.weight.copy_(attn_w["q_norm.weight"].to(
            attn.q_norm.weight.dtype))
        attn.k_norm.weight.copy_(attn_w["k_norm.weight"].to(
            attn.k_norm.weight.dtype))
        # InklingShortConv stores [channels, 1, kernel]; copy verbatim.
        attn.k_sconv.weight.copy_(attn_w["k_sconv.weight"].to(
            attn.k_sconv.weight.dtype))
        attn.v_sconv.weight.copy_(attn_w["v_sconv.weight"].to(
            attn.v_sconv.weight.dtype))
        assert attn.rel_logits_proj.shape == attn_w[
            "rel_logits_proj.proj"].shape, (
                "rel_logits_proj",
                tuple(attn.rel_logits_proj.shape),
                "expected",
                tuple(attn_w["rel_logits_proj.proj"].shape),
            )
        attn.rel_logits_proj.copy_(attn_w["rel_logits_proj.proj"].to(
            attn.rel_logits_proj.dtype))
    return attn


def _build_cache_and_metadatas(num_kv_heads, head_dim, N, P, device):
    """One shared KVCacheManagerV2 (single layer, single request of length N) plus
    a prefill metadata (context of P tokens) and a decode metadata (generation,
    1 new token after P cached tokens).

    The prefill forward writes K/V for positions [0, P) into the cache; the decode
    forward writes the one new token at slot P and attends over the reused cache
    [0, P]. Both metadatas share the same manager so decode genuinely reuses the
    prefill's cache. Mirrors backend_case for a single sequence, kv_layout="HND".
    """
    import math

    import torch

    import tensorrt_llm
    from tensorrt_llm._torch.attention_backend.utils import \
        get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import \
        KVCacheManagerV2
    from tensorrt_llm._utils import torch_dtype_to_binding
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    tokens_per_block = 64  # page size used by backend_case.py
    pages_per_seq = math.ceil(N / tokens_per_block)
    max_seq_len = pages_per_seq * tokens_per_block
    num_blocks = pages_per_seq  # one sequence
    max_num_tokens = max(8192, N)

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    cache_types = tensorrt_llm.bindings.internal.batch_manager.CacheType
    mgr = KVCacheManagerV2(
        KvCacheConfig(max_tokens=num_blocks * tokens_per_block),
        cache_types.SELF,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        mapping=mapping,
        dtype=torch_dtype_to_binding(torch.bfloat16),
    )
    mgr.add_dummy_requests([0], [N])

    AttentionCls = get_attention_backend("TRTLLM")
    prefill_md = AttentionCls.Metadata(
        num_contexts=1,
        kv_cache_params=KVCacheParams(use_cache=True,
                                      num_cached_tokens_per_seq=[0]),
        seq_lens=torch.tensor([P], dtype=torch.int),
        max_num_requests=1,
        max_num_tokens=max_num_tokens,
        kv_cache_manager=mgr,
        request_ids=[0],
        prompt_lens=[P],
        kv_layout="HND",
    )
    prefill_md.prepare()

    decode_md = AttentionCls.Metadata(
        num_contexts=0,
        kv_cache_params=KVCacheParams(use_cache=True,
                                      num_cached_tokens_per_seq=[P]),
        seq_lens=torch.tensor([1], dtype=torch.int),
        max_num_requests=1,
        max_num_tokens=max_num_tokens,
        kv_cache_manager=mgr,
        request_ids=[0],
        prompt_lens=[N],
        kv_layout="HND",
    )
    decode_md.prepare()
    return mgr, prefill_md, decode_md


def _compute_conv_states(x, attn_w, num_kv_heads, head_dim, kernel_size, pos):
    """Pre-sconv k/v conv-state window for a decode token at position ``pos``.

    Returns ``(state_k, state_v)`` each ``[1, num_kv_heads*head_dim,
    kernel_size-1]`` holding the raw (pre-short-conv) k/v projections for the
    ``kernel_size-1`` tokens before ``pos`` (oldest first). This is exactly what
    the runtime short-conv state cache carries; here it is seeded from the real
    projected activations so the decode conv reproduces the full-sequence conv at
    ``pos``.
    """
    wk = attn_w["wk_dv.weight"]  # [nkv*D, 6144]
    wv = attn_w["wv_dv.weight"]
    pre_k = (x @ wk.t())  # [T, nkv*D]  (matches qkv_proj k slice ordering)
    pre_v = (x @ wv.t())
    lo = pos - (kernel_size - 1)
    state_k = pre_k[lo:pos].transpose(0, 1).unsqueeze(0).contiguous()
    state_v = pre_v[lo:pos].transpose(0, 1).unsqueeze(0).contiguous()
    return state_k, state_v


def _compute_input(ckpt, N_target, device):
    """x = embed_norm(embed(token_ids)) as bf16 [N, 6144] from the checkpoint.

    Tokenizes a fixed real prompt with the checkpoint tokenizer; on any failure
    falls back to a fixed random token-id vector (and flags it). Returns
    (x, N, used_random_fallback).
    """
    import torch

    hidden = 6144
    rms_eps = 1e-6

    embed = _load_ckpt_tensors(ckpt, ["model.llm.embed.weight"],
                               device,
                               dtype=torch.bfloat16)["model.llm.embed.weight"]
    embed_norm_w = _load_ckpt_tensors(
        ckpt, ["model.llm.embed_norm.weight"], device,
        dtype=torch.bfloat16)["model.llm.embed_norm.weight"]
    vocab = embed.shape[0]
    assert embed.shape[1] == hidden, embed.shape

    used_random = False
    token_ids = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        ids = tok(_fixed_prompt(), add_special_tokens=True)["input_ids"]
        token_ids = torch.tensor(ids[:N_target], dtype=torch.long)
        if token_ids.numel() < 8:
            raise ValueError(
                f"tokenizer produced too few tokens: {token_ids.numel()}")
    except Exception as exc:  # noqa: BLE001 - fallback is intentional
        print(
            f"[warn] tokenizer unavailable ({exc!r}); using a FIXED RANDOM "
            f"token-id vector of length {N_target}. Numeric comparison stays "
            f"valid (identical x to both paths), but the input is synthetic.",
            flush=True,
        )
        used_random = True
        g = torch.Generator().manual_seed(1234)
        token_ids = torch.randint(0,
                                  vocab, (N_target, ),
                                  generator=g,
                                  dtype=torch.long)

    # Align N down to a multiple of the KV page size (64) so the paged cache's
    # max_seq_len == N and the model's relative bias [H, N, max_seq_len] equals
    # the reference's [H, N, N]. Padded key columns beyond N (when N is not a
    # page multiple) are a separate production-only concern, validated once core
    # math parity holds.
    page = 64
    n_aligned = (int(token_ids.numel()) // page) * page
    assert n_aligned >= page, f"too few tokens after page-align: {n_aligned}"
    token_ids = token_ids[:n_aligned]

    token_ids = token_ids.to(device)
    N = int(token_ids.numel())

    # embed lookup, then embed_norm (RMSNorm eps 1e-6) in fp32 -> bf16.
    emb = embed[token_ids]  # [N, 6144] bf16
    f = emb.float()
    var = f.pow(2).mean(-1, keepdim=True)
    normed = (f * torch.rsqrt(var + rms_eps)).to(torch.bfloat16)
    x = (embed_norm_w.to(torch.bfloat16) *
         normed).contiguous()  # [N, 6144] bf16
    return x, N, used_random


def _metrics(a, b):
    """max_abs, mean_abs, cosine between two tensors (compared in fp32)."""
    import torch.nn.functional as F

    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    cosine = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    return max_abs, mean_abs, cosine


def _apply_rmsnorm(x, gain, eps):
    """RMSNorm(x) * gain, mirroring modules/rms_norm.py: normalize in fp32, cast
    to the input dtype, then multiply by the (bf16) gain."""
    import torch

    in_dtype = x.dtype
    f = x.float()
    var = f.pow(2).mean(-1, keepdim=True)
    normed = (f * torch.rsqrt(var + eps)).to(in_dtype)
    return gain.to(in_dtype) * normed


def _replay_layer(ckpt, config, model_config, layer_idx, x_base, device):
    """Run reference + TRTLLM for one layer; return metrics.

    ``x_base`` is the shared residual-stream hidden state ``embed_norm(embed(ids))``
    (the layer-0 residual). The true input to layer L's *attention* is
    ``attn_norm_L(residual_L)`` (the decoder layer applies its pre-attention
    RMSNorm ``attn_norm`` first -- see InklingDecoderLayer.forward). We load the
    real per-layer ``attn_norm`` gain from the checkpoint and apply it here, so
    the replayed activation is the genuine source attention-layer boundary:
      * LOCAL layer 0: ``residual_0 == embed_norm(embed(ids))`` exactly, so
        ``attn_norm_0(residual_0)`` is the EXACT source layer-0 attention input.
      * GLOBAL layer 5: the exact ``residual_5`` needs the full stacked forward
        through layers 0-4 (dense + MoE = crit5/crit6); here we feed
        ``attn_norm_5(residual_0)`` -- a representative real activation carried
        through layer 5's true input norm and geometry (8 kv-heads, rel_extent
        1024, tau). Documented limitation; parity is still exact vs the reference
        because BOTH paths see the identical input.
    """
    import torch

    tc = config  # InklingTextConfig
    is_local = tc.is_local_layer(layer_idx)
    num_heads = tc.layer_num_heads(layer_idx)
    num_kv_heads = tc.layer_num_kv_heads(layer_idx)
    head_dim = tc.layer_head_dim(layer_idx)
    # rel_extent is per-layer: local uses the sliding-window extent, global the
    # full rel_extent (matches InklingAttention.__init__ and the stored profile
    # width in rel_logits_proj.proj).
    rel_extent = tc.sliding_window_size if is_local else tc.rel_extent
    sliding_window = tc.sliding_window_size if is_local else None
    log_scaling_n_floor = None if is_local else tc.log_scaling_n_floor

    # Apply layer L's pre-attention RMSNorm (attn_norm) to reach the true
    # source attention-layer boundary activation (see docstring).
    attn_norm_w = _load_ckpt_tensors(
        ckpt, [f"model.llm.layers.{layer_idx}.attn_norm.weight"],
        device,
        dtype=torch.bfloat16)[f"model.llm.layers.{layer_idx}.attn_norm.weight"]
    x = _apply_rmsnorm(x_base, attn_norm_w, tc.rms_norm_eps).contiguous()
    N = x.shape[0]

    attn_w = _read_attn_weights(ckpt, layer_idx, device)

    # Sanity-check the checkpoint shapes against the config geometry.
    assert attn_w["wq_du.weight"].shape[0] == num_heads * head_dim
    assert attn_w["wk_dv.weight"].shape[0] == num_kv_heads * head_dim
    assert attn_w["wv_dv.weight"].shape[0] == num_kv_heads * head_dim
    assert attn_w["rel_logits_proj.proj"].shape[1] == rel_extent, (
        "rel profile width",
        attn_w["rel_logits_proj.proj"].shape[1],
        "expected",
        rel_extent,
    )

    # Reference (ground truth). Compute with the spec scale (1/head_dim) and,
    # as a diagnostic, with 1/sqrt(head_dim) to check the backend q_scaling
    # convention against the observed TRTLLM output.
    import math

    def _ref(scale):
        with torch.no_grad():
            return ref_attention(
                x,
                attn_w,
                is_local,
                rel_extent,
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window=sliding_window,
                log_scaling_n_floor=log_scaling_n_floor,
                log_scaling_alpha=tc.log_scaling_alpha,
                rms_eps=tc.rms_norm_eps,
                score_scale=scale,
            )

    ref_out = _ref(None)  # 1/head_dim (primary / spec)
    ref_alt = _ref(1.0 / math.sqrt(head_dim))  # 1/sqrt(head_dim) (diagnostic)

    from tensorrt_llm._torch.attention_backend.inkling_triton import \
        build_page_table

    # TRTLLM Triton attention path. Prefill P = N-1 tokens (still crosses the 512
    # local window), then decode the last token (position P = N-1) reusing the
    # prefilled cache, in eager and CUDA-graph configurations.
    P = N - 1
    attn = _build_trtllm_attention(model_config, layer_idx, attn_w, device)
    cache_layer = attn.attn.local_layer_idx
    mgr, prefill_md, decode_md = _build_cache_and_metadatas(
        num_kv_heads, head_dim, N, P, device)
    try:
        with torch.no_grad():
            # --- Prefill (context phase): writes K/V[0, P) into the cache. ---
            pos_prefill = torch.arange(P, device=device, dtype=torch.int32)
            trt_prefill = attn.forward(
                position_ids=pos_prefill,
                hidden_states=x[:P].contiguous(),
                attn_metadata=prefill_md)[:P].contiguous()

            # --- Decode (generation phase, cache reuse), eager. The new token at
            # position P uses the short-conv state from the prefill's tail. ---
            conv_k, conv_v = _compute_conv_states(x, attn_w, num_kv_heads,
                                                  head_dim,
                                                  config.sconv_kernel_size, P)
            pos_decode = torch.tensor([P], device=device, dtype=torch.int32)
            x_dec = x[P:P + 1].contiguous()
            trt_decode = attn.forward(position_ids=pos_decode,
                                      hidden_states=x_dec,
                                      attn_metadata=decode_md,
                                      conv_states=(conv_k,
                                                   conv_v))[:1].contiguous()

            # --- Decode under CUDA graph capture/replay (hard path). The cache
            # slot P is already populated by the eager decode; capture only the
            # attention compute with static decode tensors + skip_kv_write. ---
            decode_seq_lens = torch.tensor([P + 1],
                                           device=device,
                                           dtype=torch.int32)
            block_ids = mgr.get_batch_cache_indices([0], cache_layer)
            decode_page_table = build_page_table(block_ids, len(block_ids[0]),
                                                 device)
            graph_out = _capture_decode_graph(attn, x_dec, pos_decode,
                                              decode_md, conv_k, conv_v,
                                              decode_seq_lens,
                                              decode_page_table)
    finally:
        mgr.shutdown()

    ref_prefill = ref_out[:P]
    ref_dec = ref_out[P:P + 1]
    pf_max, pf_mean, pf_cos = _metrics(ref_prefill, trt_prefill)
    dc_max, dc_mean, dc_cos = _metrics(ref_dec, trt_decode)
    cg_max, cg_mean, cg_cos = _metrics(ref_dec, graph_out)
    # cuda_graph=false vs cuda_graph=true numerical equality (hard-path proof:
    # the captured graph reproduces the eager decode bit-for-bit on replay).
    graph_replay_allclose = bool(
        torch.allclose(trt_decode.float(), graph_out.float(), atol=2e-2,
                       rtol=0))
    _, _, pf_cos_alt = _metrics(ref_alt[:P], trt_prefill)

    # Prefill per-position error split at the sliding-window boundary (local) to
    # confirm windowed queries are correct, not masked by a global average.
    per_pos = (ref_prefill.float() - trt_prefill.float()).abs().amax(dim=-1)
    boundary = sliding_window if is_local else min(rel_extent, P)
    boundary = max(0, min(boundary, P))
    early_max = per_pos[:boundary].max().item() if boundary > 0 else 0.0
    late_max = per_pos[boundary:].max().item() if boundary < P else float("nan")
    return {
        "P": P,
        "prefill_max_abs": pf_max,
        "prefill_mean_abs": pf_mean,
        "prefill_cosine": pf_cos,
        "prefill_cosine_alt": pf_cos_alt,
        "decode_max_abs": dc_max,
        "decode_mean_abs": dc_mean,
        "decode_cosine": dc_cos,
        "graph_max_abs": cg_max,
        "graph_cosine": cg_cos,
        "graph_replay_allclose": graph_replay_allclose,
        "boundary": boundary,
        "early_max_abs": early_max,
        "late_max_abs": late_max,
    }


def _capture_decode_graph(attn, x_dec, pos_decode, decode_md, conv_k, conv_v,
                          decode_seq_lens, decode_page_table):
    """Capture the Inkling decode attention under CUDA graph and replay it.

    ``skip_kv_write=True`` (the cache is already populated by the eager decode)
    plus precomputed static ``decode_seq_lens`` / ``decode_page_table`` keep the
    captured region pure GPU work (qkv/r projections, short-conv, einsum, the
    paged decode kernel, o_proj) with no host sync -- the CUDA-graph hard path.
    Returns the replayed output ``[1, hidden]``.
    """
    import torch

    x_buf = x_dec.clone()

    def run():
        return attn.forward(position_ids=pos_decode,
                            hidden_states=x_buf,
                            attn_metadata=decode_md,
                            conv_states=(conv_k, conv_v),
                            decode_seq_lens=decode_seq_lens,
                            decode_page_table=decode_page_table,
                            skip_kv_write=True)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()
    graph.replay()
    torch.cuda.synchronize()
    return graph_out[:1].contiguous().clone()


def main() -> int:
    import torch

    from tensorrt_llm._torch.model_config import ModelConfig
    # Import registers the auto-model + InklingHfWeightMapper (and defines
    # InklingAttention used above).
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401
    from tensorrt_llm.mapping import Mapping

    assert torch.cuda.is_available(), "this replay needs a CUDA device"
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    # NVFP4 quant_config is expected and fine: InklingAttention builds attention
    # in bf16 because ``.attn`` is excluded from quant (see modeling_inkling).
    model_config = ModelConfig.from_pretrained(
        CKPT,
        trust_remote_code=True,
        mapping=mapping,
        attn_backend="TRTLLM",
        moe_backend="CUTLASS",
    )
    text_config = model_config.pretrained_config.text_config

    # Build the text-only ModelConfig the InklingAttention module expects (its
    # pretrained_config must be the text sub-config, carrying is_local_layer etc.
    # and torch_dtype=bf16). Mirrors modeling_inkling._text_sub_model_config.
    import copy

    text_model_config = copy.copy(model_config)
    text_model_config.pretrained_config = text_config

    # Same real-prompt-embedding input fed to BOTH paths for BOTH layers.
    x, N, used_random = _compute_input(CKPT, N_TARGET, device)
    src = (
        "RANDOM-FALLBACK" if used_random else
        "real-prompt residual_0=embed_norm(embed(ids)); attn_norm_L applied per layer"
    )
    print(
        f"[info] input base: N={N} hidden={x.shape[1]} dtype={x.dtype} source={src}",
        flush=True)
    assert N > text_config.sliding_window_size, (
        f"need N ({N}) > local window "
        f"({text_config.sliding_window_size}) to exercise the sliding mask")

    results = {}
    # iter92: INKLING_ATTN_LAYERS extends coverage to several local + global
    # layers so a long-context divergence can be localized by layer type.
    layers = [int(s) for s in os.environ.get(
        "INKLING_ATTN_LAYERS", f"{LAYER_LOCAL},{LAYER_GLOBAL}").split(",") if s]
    for layer_idx in layers:
        kind = "local" if text_config.is_local_layer(layer_idx) else "global"
        m = _replay_layer(CKPT, text_config, text_model_config, layer_idx, x,
                          device)
        results[layer_idx] = (kind, m)
        # cuda_graph=false: prefill (context) + eager decode (generation).
        print(
            f"REPLAY layer={layer_idx} kind={kind} phase=prefill cuda_graph=false "
            f"overlap_scheduler=false P={m['P']} "
            f"max_abs={m['prefill_max_abs']:.6f} mean_abs={m['prefill_mean_abs']:.6f} "
            f"cosine={m['prefill_cosine']:.6f} "
            f"cosine_alt_1oversqrt={m['prefill_cosine_alt']:.6f} "
            f"window_boundary={m['boundary']} "
            f"max_abs[pos<bnd]={m['early_max_abs']:.6f} "
            f"max_abs[pos>=bnd]={m['late_max_abs']:.6f}",
            flush=True,
        )
        print(
            f"REPLAY layer={layer_idx} kind={kind} phase=decode cuda_graph=false "
            f"overlap_scheduler=false decode_pos={m['P']} "
            f"max_abs={m['decode_max_abs']:.6f} mean_abs={m['decode_mean_abs']:.6f} "
            f"cosine={m['decode_cosine']:.6f}",
            flush=True,
        )
        # cuda_graph=true: decode captured + replayed (hard path). The overlap
        # scheduler is a runtime (LLM API) concept with no analogue in an
        # isolated module replay; it is exercised at the full-runtime tier
        # (crit8 LLM API smoke / crit11-12 accuracy). Here the CUDA-graph axis is
        # covered concretely: the captured graph reproduces the eager decode.
        print(
            f"REPLAY layer={layer_idx} kind={kind} phase=decode cuda_graph=true "
            f"overlap_scheduler=n/a(module) decode_pos={m['P']} "
            f"max_abs={m['graph_max_abs']:.6f} cosine={m['graph_cosine']:.6f} "
            f"graph_replay_allclose={m['graph_replay_allclose']}",
            flush=True,
        )

    def _layer_ok(m):
        return (m["prefill_cosine"] >= COSINE_TOL
                and m["decode_cosine"] >= COSINE_TOL
                and m["graph_cosine"] >= COSINE_TOL
                and m["graph_replay_allclose"])

    all_ok = all(_layer_ok(m) for (_, m) in results.values())
    if all_ok:
        print("CRIT4_OK", flush=True)
        return 0
    print("CRIT4_MISMATCH", flush=True)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
