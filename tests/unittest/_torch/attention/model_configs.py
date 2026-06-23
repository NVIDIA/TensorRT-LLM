# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-derived attention configurations for the backend test sweep.

Instead of an arbitrary ``itertools.product`` of head counts and dims, the sweep
is driven by the *distinct* attention configurations actually used by the
supported models under ``tensorrt_llm/_torch/models/`` and
``tensorrt_llm/_torch/visual_gen/models/``. Each entry records the
``(num_heads, num_kv_heads, head_dim, rope, mask, is_mla, is_cross)`` tuple and
names the model family it comes from, so a failure points at real workloads.

Many models collapse onto the same tuple (e.g. Llama/Mistral/Qwen-1.5 all use
32 Q / 8 KV / 128); only the distinct tuples are listed. Numeric values are the
canonical-checkpoint values for that family. Where a family's distinguishing
feature is a RoPE *table* variant (yarn, longrope, mrope) the attention math is
identical given pre-rotated Q/K, so it is represented by the base RoPE style
(neox/gptj); the variant is noted in ``models``.

``rope`` is one of ``None`` / ``"neox"`` / ``"gptj"``. ``mask`` is
``"causal"`` / ``"full"`` / ``"sliding"``. ``no_cache=True`` marks the
bidirectional, KV-cache-free DiT / encoder workloads.

Compute/KV dtype is intentionally NOT a field here: a model can be served in
many precisions (fp16/bf16/fp8), so dtype is an orthogonal test dimension, not
part of the attention architecture.

Deliberate exclusions (not distinct at the dense-backend level, or covered
elsewhere):
- Per-layer-varying models (Gemma4 global vs sliding layers, Laguna per-layer
  rope) decompose into the per-layer tuples already listed (sliding 128, MHA
  128, head_dim 256).
- Vision/text encoders (SigLip/Radio/CLIP/Parakeet, MiniMax-VL tower) collapse
  onto the bidirectional MHA tuples already listed (e.g. 16x64 / 12x64 full).
- Sparse/DSA indexer attention (GLM-DSA, NSA, RocketKV) is a separate paradigm
  validated under sparse/; there is no dense Vanilla golden for it.
- Multimodal cross variants (Llama4-vision, Gemma4-MM) reuse the cross tuples.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ModelAttnConfig:
    id: str  # short, stable test id
    models: str  # canonical model family/families this tuple comes from
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope: Optional[str] = "neox"  # None | "neox" | "gptj"
    mask: str = "causal"  # "causal" | "full" | "sliding"
    sliding_window: Optional[int] = None
    no_cache: bool = False  # bidirectional DiT/encoder (no KV cache)
    is_cross: bool = False  # encoder-decoder cross attention
    # MLA (DeepSeek-style latent attention). For absorbed generation num_kv_heads
    # is 1 (single latent head); for the up-projected context pass (mla_context)
    # it is MHA (num_kv_heads == num_heads) with asymmetric K/V (K head_dim =
    # qk_nope + qk_rope, V head_dim = v_head_dim).
    is_mla: bool = False
    mla_context: bool = False  # up-projected MHA context pass (vs absorbed gen)
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None


# ---------------------------------------------------------------------------
# Standard self-attention (decoder LLMs). The dominant configurations.
# ---------------------------------------------------------------------------
_STANDARD = [
    # GQA 4:1, neox, causal -- by far the most common LLM shape.
    ModelAttnConfig(
        "llama3_gqa",
        "Llama-2/3/4, Qwen-1.5, Phi-3, Nemotron, Mixtral",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
    ),
    # MHA, neox, causal.
    ModelAttnConfig(
        "qwen3_mha",
        "Qwen3, Qwen3-MoE, ExaOne-MoE, Hunyuan-MoE",
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
    ),
    # GQA + sliding window (4096), neox.
    ModelAttnConfig(
        "mistral_swa",
        "Mistral-7B, Mixtral, StarCoder2, ExaOne-4",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask="sliding",
        sliding_window=4096,
    ),
    # GQA + sliding window, gptj-style RoPE (Cohere2 per-layer PE).
    ModelAttnConfig(
        "cohere2_gptj_swa",
        "Cohere2 / Command-R (gptj RoPE, per-layer PE)",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        rope="gptj",
        mask="sliding",
        sliding_window=4096,
    ),
    # GQA + sliding window, yarn-scaled RoPE (represented as neox).
    ModelAttnConfig(
        "mistral_nemo_yarn_swa",
        "Mistral-Nemo, GPT-OSS (yarn RoPE)",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask="sliding",
        sliding_window=4096,
    ),
    # MHA + sliding window, mrope (represented as neox); Qwen2-VL.
    ModelAttnConfig(
        "qwen2vl_mrope_swa",
        "Qwen2-VL, Step-3.7-VL (mrope)",
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        mask="sliding",
        sliding_window=4096,
    ),
    # MHA, mrope causal (represented as neox); Qwen3-VL / Kimi-K2.5 / SeedOSS.
    ModelAttnConfig(
        "qwen3vl_mrope",
        "Qwen3-VL, Kimi-K2.5, SeedOSS, ExaOne-4.5 (mrope)",
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
    ),
    # Large head_dim edge case (256), GQA, sliding window; Gemma3.
    ModelAttnConfig(
        "gemma3_hd256_swa",
        "Gemma3 (head_dim=256)",
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        mask="sliding",
        sliding_window=4096,
    ),
    # MQA (single KV head): SYNTHETIC -- no current supported model uses it, but
    # it is the GQA extreme (n_rep == num_heads) and worth guarding.
    ModelAttnConfig(
        "mqa_synthetic",
        "(synthetic MQA -- no model; GQA extreme)",
        num_heads=16,
        num_kv_heads=1,
        head_dim=128,
    ),
]

# ---------------------------------------------------------------------------
# MLA (DeepSeek-style absorbed latent attention). num_kv_heads == 1 latent head.
# ---------------------------------------------------------------------------
_MLA = [
    # DeepSeek-V3: 128 Q heads, qk_nope=128, qk_rope=64, kv_lora=512, v=128.
    ModelAttnConfig(
        "deepseekv3_mla",
        "DeepSeek-V3 / R1",
        num_heads=128,
        num_kv_heads=1,
        head_dim=192,  # qk_nope+qk_rope
        is_mla=True,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    # Hunyuan-Dense MLA variant: 32 Q heads, qk_nope=128.
    ModelAttnConfig(
        "hunyuan_mla",
        "Hunyuan-Dense (MLA mode)",
        num_heads=32,
        num_kv_heads=1,
        head_dim=192,
        is_mla=True,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    # DeepSeek-V3 MLA *context* pass: the module up-projects compressed_kv to
    # full per-head K/V, so attention is MHA (num_kv_heads == num_heads) with
    # asymmetric K/V -- K head_dim = qk_nope + qk_rope = 192, V head_dim = 128.
    # (Heads scaled down from 128 for test speed; the tuple shape is the point.)
    ModelAttnConfig(
        "deepseekv3_mla_ctx",
        "DeepSeek-V3 (MLA up-proj context, asymmetric K/V)",
        num_heads=16,
        num_kv_heads=16,
        head_dim=192,
        mla_context=True,
        is_mla=True,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
]

# ---------------------------------------------------------------------------
# Cross-attention (encoder-decoder). Decoder Q attends to encoder K/V (full).
# ---------------------------------------------------------------------------
_CROSS = [
    ModelAttnConfig(
        "t5_cross",
        "T5 / BART (NoPE, bidirectional encoder)",
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        rope=None,
        mask="full",
        is_cross=True,
    ),
    ModelAttnConfig(
        "mllama_cross",
        "MLLama (GQA decoder, llama3 RoPE)",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        rope="neox",
        mask="full",
        is_cross=True,
    ),
]

# ---------------------------------------------------------------------------
# Bidirectional, KV-cache-free workloads: DiT image/video + ViT/text encoders.
# Tested through the normal TRTLLM backend with a full (non-causal) mask.
# ---------------------------------------------------------------------------
_NO_CACHE = [
    ModelAttnConfig(
        "bert_encoder",
        "BERT / CLIP / SigLip (NoPE bidirectional encoder)",
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "flux_dit",
        "FLUX-1 (DiT, dual-stream joint attention)",
        num_heads=24,
        num_kv_heads=24,
        head_dim=128,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "ltx2_dit",
        "LTX-2 (DiT, video)",
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "cosmos3_dit",
        "Cosmos3 (DiT, 3D mrope -> neox)",
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    # Vision towers with non-standard head_dims (the production kernels pad
    # these to a supported cubin size; here they exercise the head_dim directly).
    ModelAttnConfig(
        "qwen2vl_vision",
        "Qwen2-VL vision tower (head_dim=80)",
        num_heads=16,
        num_kv_heads=16,
        head_dim=80,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "gemma_vision",
        "Gemma3/Qwen3-VL vision tower (head_dim=72)",
        num_heads=16,
        num_kv_heads=16,
        head_dim=72,
        rope=None,
        mask="full",
        no_cache=True,
    ),
]

MODEL_CONFIGS: List[ModelAttnConfig] = _STANDARD + _MLA + _CROSS + _NO_CACHE
