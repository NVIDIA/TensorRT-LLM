# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-derived attention configurations for the backend test sweep.

Instead of an arbitrary ``itertools.product`` of head counts and dims, the sweep
is driven by the *distinct* attention configurations actually used by the
supported models under ``tensorrt_llm/_torch/models/`` and
``tensorrt_llm/_torch/visual_gen/models/``. Each entry records the
``(num_heads, num_kv_heads, head_dim, rope, mask, sliding_window, is_mla,
is_cross)`` tuple and names the model family it comes from, so a failure points
at real workloads.

Many models collapse onto the same tuple (e.g. Llama/Mistral/Qwen-1.5 all use
32 Q / 8 KV / 128); only the distinct tuples are listed. Numeric values are the
canonical-checkpoint values for that family. Where a family's distinguishing
feature is a RoPE *table* variant (yarn, longrope, mrope) the attention math is
identical given pre-rotated Q/K, so it is represented by the base RoPE style
(neox/gptj); the variant is noted in ``models``.

``rope`` is one of ``None`` / ``"neox"`` / ``"gptj"``. ``mask`` is
``"causal"`` / ``"full"`` / ``"sliding"``. ``no_cache=True`` marks the
bidirectional, KV-cache-free DiT / encoder workloads.

ID naming rule:
- Use lowercase snake_case:
  ``<family>[_<variant_or_size>]_<attn_kind>[_hdN][_rope][_swaN|_self|_cross|_encoder|_vision|_mla|_mla_ctx]``.
- ``attn_kind`` is ``mha`` / ``gqa`` / ``mqa`` for dense attention. MLA uses
  ``_mla`` or ``_mla_ctx`` instead of a separate dense-attention kind.
- Omit ``hd128`` for the common head dimension; include ``hdN`` for other
  dimensions. Omit default GPT-NeoX RoPE; include ``gptj`` or ``nope`` when
  they distinguish the tuple.
- Sliding-window IDs must include the real window size as ``swaN`` because the
  window is part of the backend config tuple.
- When multiple model families share one tuple, the ID names a representative
  family or compact group; the ``models`` field lists the known families.

Compute/KV dtype is intentionally NOT a field here: a model can be served in
many precisions (fp16/bf16/fp8), so dtype is an orthogonal test dimension, not
part of the attention architecture.

Deliberate exclusions (not distinct at the dense-backend level, or covered
elsewhere):
- Per-layer-varying models decompose into the per-layer tuples listed here. A
  config value such as ``sliding_window`` is ignored when the PyTorch model path
  does not pass it to the dense attention backend (for example Qwen2 /
  Qwen2.5-VL with ``use_sliding_window=False``).
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
    # Dense full-attention decoder LLMs.
    ModelAttnConfig(
        "llama3_8b_gqa",
        "Llama-3 8B, Qwen3-8B, Mixtral-8x7B, Qwen3-VL-8B, Nemotron-H-8B",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
    ),
    ModelAttnConfig(
        "llama3_70b_qwen3_32b_gqa",
        "Llama-3 70B, Qwen2/3-32B/72B, MiniMax-M1, Step-3.5/3.7 full layers",
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
    ),
    ModelAttnConfig(
        "llama3_405b_gqa",
        "Llama-3.1 405B",
        num_heads=128,
        num_kv_heads=8,
        head_dim=128,
    ),
    ModelAttnConfig(
        "llama4_gqa",
        "Llama-4, Qwen3-14B, Qwen2.5-14B/32B, Nemotron Nano 4B/9B/12B",
        num_heads=40,
        num_kv_heads=8,
        head_dim=128,
    ),
    ModelAttnConfig(
        "llama3_2_1b_gqa_hd64",
        "Llama-3.2 1B, EXAONE-4.0-1.2B",
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
    ),
    ModelAttnConfig(
        "llama3_2_3b_gqa",
        "Llama-3.2 3B, Phi-4-mini, HCXVision-3B, Minitron/Nemotron-4B",
        num_heads=24,
        num_kv_heads=8,
        head_dim=128,
    ),
    ModelAttnConfig(
        "mixtral_8x22b_minimax_laguna_gqa",
        "Mixtral-8x22B, MiniMax-M2, Laguna full layers",
        num_heads=48,
        num_kv_heads=8,
        head_dim=128,
    ),
    ModelAttnConfig(
        "seed_oss_gqa",
        "Seed-OSS-36B",
        num_heads=80,
        num_kv_heads=8,
        head_dim=128,
    ),
    ModelAttnConfig(
        "qwen2_7b_gqa",
        "Qwen2-7B, Qwen2.5-VL-7B, Qwen2-MoE-57B, LLaVA-OneVision-Qwen2",
        num_heads=28,
        num_kv_heads=4,
        head_dim=128,
    ),
    ModelAttnConfig(
        "qwen2_0_5b_gqa_hd64",
        "Qwen2/Qwen2.5-0.5B",
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
    ),
    ModelAttnConfig(
        "qwen2_1_5b_gqa",
        "Qwen2/Qwen2.5-1.5B",
        num_heads=12,
        num_kv_heads=2,
        head_dim=128,
    ),
    ModelAttnConfig(
        "qwen2_3b_gqa",
        "Qwen2.5-3B, Qwen2.5-VL-3B",
        num_heads=16,
        num_kv_heads=2,
        head_dim=128,
    ),
    ModelAttnConfig(
        "qwen3_small_gqa",
        "Qwen3-0.6B/1.7B, Qwen3-VL-2B, HyperCLOVAX-SEED-Text-0.5B",
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    ),
    ModelAttnConfig(
        "qwen3_moe_30b_gqa",
        "Qwen3-MoE-30B-A3B, Qwen3-VL-MoE-30B-A3B",
        num_heads=32,
        num_kv_heads=4,
        head_dim=128,
    ),
    ModelAttnConfig(
        "qwen3_moe_235b_gqa",
        "Qwen3-MoE-235B-A22B, Qwen3-VL-MoE-235B-A22B",
        num_heads=64,
        num_kv_heads=4,
        head_dim=128,
    ),
    ModelAttnConfig(
        "tinyllama_gqa_hd64",
        "TinyLlama / synthetic Llama GQA-4, head_dim=64",
        num_heads=32,
        num_kv_heads=4,
        head_dim=64,
    ),
    ModelAttnConfig(
        "nemotron_h_32h_gqa",
        "Nemotron-H 30B/120B",
        num_heads=32,
        num_kv_heads=2,
        head_dim=128,
    ),
    ModelAttnConfig(
        "nemotron_h_64h_gqa",
        "Nemotron-H 47B/56B/Ultra sample",
        num_heads=64,
        num_kv_heads=2,
        head_dim=128,
    ),
    ModelAttnConfig(
        "glm4_moe_gqa",
        "GLM-4.5/4.6/4.7 MoE",
        num_heads=96,
        num_kv_heads=8,
        head_dim=128,
    ),
    ModelAttnConfig(
        "qwen3_next_gqa_hd256",
        "Qwen3-Next full-attention layers, Qwen3.5 35B-A3B full layers",
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
    ),
    ModelAttnConfig(
        "qwen3_5_27b_gqa_hd256",
        "Qwen3.5-27B full-attention layers",
        num_heads=24,
        num_kv_heads=4,
        head_dim=256,
    ),
    ModelAttnConfig(
        "qwen3_5_397b_gqa_hd256",
        "Qwen3.5-397B-A17B full-attention layers",
        num_heads=32,
        num_kv_heads=2,
        head_dim=256,
    ),
    ModelAttnConfig(
        "llama2_7b_qwen1_5_7b_mha",
        "Llama-2-7B, Qwen1.5-7B, DeciLM-7B",
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
    ),
    ModelAttnConfig(
        "vila1_5_3b_mha",
        "VILA1.5-3B LLM",
        num_heads=20,
        num_kv_heads=20,
        head_dim=128,
    ),
    ModelAttnConfig(
        "llama2_13b_qwen1_5_14b_mha",
        "Llama-2-13B, Qwen1.5-14B",
        num_heads=40,
        num_kv_heads=40,
        head_dim=128,
    ),
    ModelAttnConfig(
        "llama_30b_mha",
        "Llama-30B",
        num_heads=52,
        num_kv_heads=52,
        head_dim=128,
    ),
    ModelAttnConfig(
        "llama_65b_qwen1_5_72b_mha",
        "Llama-65B, Qwen1.5-72B",
        num_heads=64,
        num_kv_heads=64,
        head_dim=128,
    ),
    ModelAttnConfig(
        "nemotron_nas_ultra_mha",
        "Nemotron-NAS Ultra 253B",
        num_heads=128,
        num_kv_heads=128,
        head_dim=128,
    ),
    ModelAttnConfig(
        "qwen1_5_0_5b_mha_hd64",
        "Qwen1.5-0.5B",
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
    ),
    ModelAttnConfig(
        "qwen1_5_moe_mha",
        "Qwen1.5-MoE-A2.7B",
        num_heads=16,
        num_kv_heads=16,
        head_dim=128,
    ),
    ModelAttnConfig(
        "phi3_mha_hd96",
        "Phi-3-mini / Phi-3.5-mini full-attention PyTorch path",
        num_heads=32,
        num_kv_heads=32,
        head_dim=96,
    ),
    ModelAttnConfig(
        "phi3_medium_gqa",
        "Phi-3-medium full-attention PyTorch path",
        num_heads=40,
        num_kv_heads=10,
        head_dim=128,
    ),
    ModelAttnConfig(
        "nemotron_nas_mha_hd64",
        "Nemotron-NAS / DeciLM mini",
        num_heads=32,
        num_kv_heads=32,
        head_dim=64,
    ),
    ModelAttnConfig(
        "gpt_oss_gqa_hd64_gptj",
        "GPT-OSS full-attention layers (yarn, gptj-style rotation)",
        num_heads=64,
        num_kv_heads=8,
        head_dim=64,
        rope="gptj",
    ),
    # NoPE decoder self-attention (learned/relative positional embeddings are
    # handled outside the dense backend).
    ModelAttnConfig(
        "flan_t5_small_mha_hd64_nope_self",
        "FLAN-T5-small / ByT5-small decoder self-attention",
        num_heads=6,
        num_kv_heads=6,
        head_dim=64,
        rope=None,
    ),
    ModelAttnConfig(
        "t5_small_mha_hd64_nope_self",
        "T5-small decoder self-attention",
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        rope=None,
    ),
    ModelAttnConfig(
        "t5_base_mha_hd64_nope_self",
        "T5-base decoder self-attention",
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        rope=None,
    ),
    ModelAttnConfig(
        "bart_mha_hd64_nope_self",
        "BART / mBART decoder self-attention",
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        rope=None,
    ),
    ModelAttnConfig(
        "flan_t5_xl_mha_hd64_nope_self",
        "FLAN-T5-XL decoder self-attention",
        num_heads=32,
        num_kv_heads=32,
        head_dim=64,
        rope=None,
    ),
    ModelAttnConfig(
        "exaone4_gqa_nope",
        "EXAONE-4.0/4.5 full-attention layers (NoPE)",
        num_heads=40,
        num_kv_heads=8,
        head_dim=128,
        rope=None,
    ),
    ModelAttnConfig(
        "exaone_moe_gqa_nope",
        "K-EXAONE full-attention layers (NoPE)",
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        rope=None,
    ),
    ModelAttnConfig(
        "gemma3_1b_mqa_hd256",
        "Gemma3-1B full-attention layers",
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
    ),
    ModelAttnConfig(
        "gemma3_12b_gqa_hd256",
        "Gemma3-12B full-attention layers",
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
    ),
    ModelAttnConfig(
        "gemma3_27b_gqa",
        "Gemma3-27B full-attention layers",
        num_heads=32,
        num_kv_heads=16,
        head_dim=128,
    ),
    ModelAttnConfig(
        "gemma4_e2b_mqa_hd512",
        "Gemma4-E2B full-attention layers",
        num_heads=8,
        num_kv_heads=1,
        head_dim=512,
    ),
    ModelAttnConfig(
        "gemma4_26b_gqa_hd512",
        "Gemma4-26B-A4B full-attention layers",
        num_heads=16,
        num_kv_heads=2,
        head_dim=512,
    ),
    ModelAttnConfig(
        "gemma4_31b_gqa_hd512",
        "Gemma4-31B full-attention layers",
        num_heads=32,
        num_kv_heads=4,
        head_dim=512,
    ),
    # Dense sliding-window decoder LLMs. The window size is part of the tuple:
    # two otherwise-identical head layouts with different windows stay separate.
    ModelAttnConfig(
        "mistral_gqa_swa4096",
        "Mistral-7B-v0.1",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask="sliding",
        sliding_window=4096,
    ),
    ModelAttnConfig(
        "ministral_gqa_swa32768",
        "Ministral-style Mistral layer_types",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        mask="sliding",
        sliding_window=32768,
    ),
    ModelAttnConfig(
        "cohere2_gqa_gptj_swa4096",
        "Cohere2 / Command-R (gptj RoPE, per-layer PE)",
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        rope="gptj",
        mask="sliding",
        sliding_window=4096,
    ),
    ModelAttnConfig(
        "gpt_oss_gqa_hd64_gptj_swa128",
        "GPT-OSS sliding layers (yarn, gptj-style rotation)",
        num_heads=64,
        num_kv_heads=8,
        head_dim=64,
        rope="gptj",
        mask="sliding",
        sliding_window=128,
    ),
    ModelAttnConfig(
        "exaone4_gqa_swa4096",
        "EXAONE-4.0/4.5 sliding layers",
        num_heads=40,
        num_kv_heads=8,
        head_dim=128,
        mask="sliding",
        sliding_window=4096,
    ),
    ModelAttnConfig(
        "exaone_moe_gqa_swa128",
        "K-EXAONE sliding layers",
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        mask="sliding",
        sliding_window=128,
    ),
    ModelAttnConfig(
        "laguna_step_gqa_swa512",
        "Laguna sliding layers, Step-3.5/3.7 sliding layers",
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        mask="sliding",
        sliding_window=512,
    ),
    ModelAttnConfig(
        "starcoder2_3b_gqa_swa4096",
        "StarCoder2-3B",
        num_heads=24,
        num_kv_heads=2,
        head_dim=128,
        mask="sliding",
        sliding_window=4096,
    ),
    ModelAttnConfig(
        "starcoder2_7b_gqa_swa4096",
        "StarCoder2-7B",
        num_heads=36,
        num_kv_heads=4,
        head_dim=128,
        mask="sliding",
        sliding_window=4096,
    ),
    ModelAttnConfig(
        "starcoder2_15b_gqa_swa4096",
        "StarCoder2-15B",
        num_heads=48,
        num_kv_heads=4,
        head_dim=128,
        mask="sliding",
        sliding_window=4096,
    ),
    ModelAttnConfig(
        "gemma3_1b_mqa_hd256_swa512",
        "Gemma3-1B local layers",
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        mask="sliding",
        sliding_window=512,
    ),
    ModelAttnConfig(
        "gemma3_12b_gqa_hd256_swa1024",
        "Gemma3-12B local layers, Gemma4-26B-A4B sliding layers",
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        mask="sliding",
        sliding_window=1024,
    ),
    ModelAttnConfig(
        "gemma3_27b_gqa_swa1024",
        "Gemma3-27B local layers",
        num_heads=32,
        num_kv_heads=16,
        head_dim=128,
        mask="sliding",
        sliding_window=1024,
    ),
    ModelAttnConfig(
        "gemma4_e2b_mqa_hd256_swa512",
        "Gemma4-E2B sliding layers",
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        mask="sliding",
        sliding_window=512,
    ),
    ModelAttnConfig(
        "gemma4_31b_gqa_hd256_swa1024",
        "Gemma4-31B sliding layers",
        num_heads=32,
        num_kv_heads=16,
        head_dim=256,
        mask="sliding",
        sliding_window=1024,
    ),
]

# ---------------------------------------------------------------------------
# MLA (DeepSeek-style absorbed latent attention). num_kv_heads == 1 latent head.
# ---------------------------------------------------------------------------
_MLA = [
    # DeepSeek-V3: 128 Q heads, qk_nope=128, qk_rope=64, kv_lora=512, v=128.
    ModelAttnConfig(
        "deepseekv3_mla",
        "DeepSeek-V3 / R1 / V3.2, MistralLarge3",
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
        "Hunyuan-Dense (MLA mode), DeepSeek-V3-Lite",
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
    # Kimi K2.5 uses a DeepSeek-style MLA text backbone with 64 query heads.
    ModelAttnConfig(
        "kimi_k25_mla",
        "Kimi-K2.5",
        num_heads=64,
        num_kv_heads=1,
        head_dim=192,
        is_mla=True,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    # MLA *context* pass: the module up-projects compressed_kv to full
    # per-head K/V, so attention is MHA (num_kv_heads == num_heads) with
    # asymmetric K/V -- K head_dim = qk_nope + qk_rope = 192, V head_dim = 128.
    ModelAttnConfig(
        "deepseekv3_mla_ctx",
        "DeepSeek-V3 / R1 / V3.2 MLA context",
        num_heads=128,
        num_kv_heads=128,
        head_dim=192,
        mla_context=True,
        is_mla=True,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    ModelAttnConfig(
        "hunyuan_mla_ctx",
        "Hunyuan-Dense / DeepSeek-V3-Lite MLA context",
        num_heads=32,
        num_kv_heads=32,
        head_dim=192,
        mla_context=True,
        is_mla=True,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    ),
    ModelAttnConfig(
        "kimi_k25_mla_ctx",
        "Kimi-K2/K2.5 MLA context",
        num_heads=64,
        num_kv_heads=64,
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
        "flan_t5_small_mha_hd64_nope_cross",
        "FLAN-T5-small / ByT5-small cross-attention",
        num_heads=6,
        num_kv_heads=6,
        head_dim=64,
        rope=None,
        mask="full",
        is_cross=True,
    ),
    ModelAttnConfig(
        "t5_small_mha_hd64_nope_cross",
        "T5-small cross-attention",
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        rope=None,
        mask="full",
        is_cross=True,
    ),
    ModelAttnConfig(
        "t5_base_mha_hd64_nope_cross",
        "T5-base cross-attention",
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        rope=None,
        mask="full",
        is_cross=True,
    ),
    ModelAttnConfig(
        "bart_mha_hd64_nope_cross",
        "BART / mBART cross-attention",
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        rope=None,
        mask="full",
        is_cross=True,
    ),
    ModelAttnConfig(
        "flan_t5_xl_mha_hd64_nope_cross",
        "FLAN-T5-XL cross-attention",
        num_heads=32,
        num_kv_heads=32,
        head_dim=64,
        rope=None,
        mask="full",
        is_cross=True,
    ),
    ModelAttnConfig(
        "mllama_gqa_cross",
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
        "t5_small_mha_hd64_nope_encoder",
        "FLAN-T5-small / ByT5-small encoder",
        num_heads=6,
        num_kv_heads=6,
        head_dim=64,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "bert_mha_hd64_nope_encoder",
        "BERT / CLIP / T5-base / SigLip (NoPE bidirectional encoder)",
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "flux_mha_dit",
        "FLUX-1 (DiT, dual-stream joint attention)",
        num_heads=24,
        num_kv_heads=24,
        head_dim=128,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "ltx2_mha_hd64_dit",
        "T5-small encoder, LTX-2 (DiT, video)",
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "cosmos3_mha_hd64_dit",
        "BART / mBART encoder, Cosmos3 (DiT, 3D mrope -> neox)",
        num_heads=16,
        num_kv_heads=16,
        head_dim=64,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "flan_t5_xl_mha_hd64_nope_encoder",
        "FLAN-T5-XL encoder",
        num_heads=32,
        num_kv_heads=32,
        head_dim=64,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    # Vision towers with non-standard head_dims (the production kernels pad
    # these to a supported cubin size; here they exercise the head_dim directly).
    ModelAttnConfig(
        "qwen2vl_mha_hd80_nope_vision",
        "Qwen2-VL vision tower (head_dim=80)",
        num_heads=16,
        num_kv_heads=16,
        head_dim=80,
        rope=None,
        mask="full",
        no_cache=True,
    ),
    ModelAttnConfig(
        "gemma_qwen3vl_mha_hd72_nope_vision",
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
