# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""TensorRT-LLM PyTorch backend bring-up for the Step3p7 Flash checkpoints.

Source semantics live in ``/home/scratch.kevxie_sw_1/workspace/quiet_harbor/
step-3-7-flash-fp8_vv1/modeling_step3p7.py``.  Highlights:

- 45 text decoder layers; layer 0,4,8,...,44 are full-attention (Q=64, KV=8)
  and the rest are sliding-attention (Q=96, KV=8, window=512).
- Per-layer RoPE: full layers use partial rotary 0.5 with llama3 scaling
  (theta 5e6); sliding layers use full rotary 1.0 with theta 1e4 and no
  scaling.
- Gemma-style Q/K RMSNorm (``weight + 1``) and a head-wise output gate
  (``sigmoid(g_proj(hidden))`` multiplied per attention head before
  ``o_proj``).
- Layers 0..2 use dense SwiGLU MLPs; layers 3..44 mix routed quantized MoE
  (288 experts, top-k 8, sigmoid+bias top-k, renormalized weights times
  routed_scaling_factor=3.0, fp32 gate) with a bf16 shared expert.
- Late layers 43/44 carry non-zero SwiGLU clamp limits for the routed and
  shared experts.

Three checkpoint variants are supported by this single module:

- **BF16 reference** (``step-3-7-flash-bf16_vv1``): unquantized text decoder,
  no FP8 KV cache.
- **FP8 block-scale flash** (``step-3-7-flash-fp8_vv1``): routed MoE experts
  in FP8 with 128x128 block scales; attention / dense MLP / shared expert
  stay BF16.
- **NVFP4 multimodal NIM/VNIM** (``step-3.7-flash-nim_vnim-nvfp4-260528``):
  routed MoE experts in NVFP4 (e2m1 + per-16 FP8 block scale + per-tensor
  FP32 global scale) with FP8 KV cache.  The on-disk checkpoint nests the
  text decoder under ``model.language_model.*`` and the vision tower under
  ``model.vision_model.*``; ``rewrite_language_model_keys`` flattens the
  namespace at load time so the TRT-LLM module tree (still rooted at
  ``model.layers.*``) is reused unchanged.

The on-disk checkpoints also contain 3 MTP layers (45..47) and a vision
tower.  The vision tower is ignored for text-only generation; MTP layers
are loaded when ``MTPDecodingConfig`` is enabled.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..distributed import AllReduce, AllReduceParams, allgather
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding, LMHead
from ..modules.fused_moe import create_moe
from ..modules.fused_moe.interface import MoEWeightLoadingMode
from ..modules.fused_moe.routing import MiniMaxM2MoeRoutingMethod
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, create_lm_head_tp_mapping
from .modeling_speculative import SpecDecOneEngineForCausalLM, _slice_spec_position_ids
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model

# ---------------------------------------------------------------------------
# FP8 / NVFP4 dequant helpers (used by the layer-43/44 SwiGLU-clamp path)
# ---------------------------------------------------------------------------


def _fp8_block_dequant_3d(
    weight_fp8: torch.Tensor, scale_inv: torch.Tensor, block: int = 128
) -> torch.Tensor:
    """Dequantise a 3D per-expert FP8 e4m3 block-scale tensor to bf16.

    Args:
        weight_fp8: ``(E, M, K)`` FP8 weight tensor.
        scale_inv: ``(E, ceil(M/block), ceil(K/block))`` FP32 scale tensor.
        block: per-axis block size (default 128 -- DeepSeek/Step3p7 convention).
    Returns:
        ``(E, M, K)`` bf16 tensor on the same device as ``weight_fp8``.
    """
    _, M, K = weight_fp8.shape
    scale = scale_inv.to(torch.float32).repeat_interleave(block, dim=-2)
    scale = scale[..., :M, :].repeat_interleave(block, dim=-1)[..., :K]
    return (weight_fp8.to(torch.float32) * scale).to(torch.bfloat16)


# NVFP4 / e2m1 nibble lookup. Index 0-15 are the 16 representable values for
# the e2m1 format used by NVFP4 (1 sign bit + 2 exponent + 1 mantissa).
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _nvfp4_dequant_batched(
    weight_uint8: torch.Tensor,
    block_scale_fp8: torch.Tensor,
    global_scale_fp32: torch.Tensor,
    target_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Dequantise an N-D NVFP4 weight tensor to bf16 in one batched op.

    Matches modelopt's ``_dequantize_nvfp4`` reference: per-16 block scale
    (fp8_e4m3) and per-tensor (or per-expert) global scale (fp32) multiply
    directly into the e2m1 nibble values, then cast to bf16.

    Args:
        weight_uint8: ``(*B, M, K/2)`` packed weight (two e2m1 nibbles/byte).
        block_scale_fp8: ``(*B, M, K/16)`` per-16 block scale (fp8_e4m3).
        global_scale_fp32: scalar fp32 or ``(B,)`` per-expert fp32.
        target_device: device to run the dequant on. Defaults to
            ``weight_uint8.device``. For checkpoint tensors that live on
            CPU it is *much* faster to move the inputs to GPU first because
            the per-element ``lut[idx]`` gather costs ~50ms per expert
            tensor on CPU but <1ms on B200.
    Returns:
        ``(*B, M, K)`` bf16 tensor on ``target_device``.
    """
    device = target_device or weight_uint8.device
    w = weight_uint8.to(device=device, non_blocking=True)
    s1 = block_scale_fp8.to(device=device, non_blocking=True)
    s2 = global_scale_fp32.to(device=device, non_blocking=True)

    K_half = w.shape[-1]
    K = K_half * 2
    lut = _E2M1_VALUES.to(device=device)

    high = (w >> 4) & 0x0F
    low = w & 0x0F
    # gather → (*shape_of_w, ) float32; then interleave low/high along K.
    vals = torch.empty(*w.shape[:-1], K, dtype=torch.float32, device=device)
    vals[..., 0::2] = lut[low.long()]
    vals[..., 1::2] = lut[high.long()]

    # Broadcast (*B,) global scale to (*B, 1, 1) so it multiplies (*B, M, K/16).
    s2_b = s2.to(torch.float32)
    for _ in range(s1.dim() - s2_b.dim()):
        s2_b = s2_b.unsqueeze(-1)
    scale = (s1.to(torch.float32) * s2_b).unsqueeze(-1)  # (*B, M, K/16, 1)

    vals = vals.view(*w.shape[:-1], K // 16, 16) * scale
    return vals.view(*w.shape[:-1], K).to(torch.bfloat16)


def _nvfp4_stack_dequant(
    weights,
    base: str,
    w_name: str,
    expert_ids: list,
    target_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Dequant NVFP4 expert tensors for ``expert_ids`` and stack along dim 0.

    Reads ``{base}.<e>.<w_name>.{weight,weight_scale,weight_scale_2}`` for
    each ``e``, stacks once on CPU (with ``torch.stack`` over views), then
    runs a single batched GPU dequant. Returns ``(len(expert_ids), M, K)``
    bf16 on ``target_device``. Used by the Python SwiGLU-clamp fallback on
    layers with non-zero ``swiglu_limits``.
    """
    w_stack = torch.stack([weights[f"{base}.{e}.{w_name}.weight"] for e in expert_ids])
    s1_stack = torch.stack([weights[f"{base}.{e}.{w_name}.weight_scale"] for e in expert_ids])
    s2_stack = torch.stack(
        [weights[f"{base}.{e}.{w_name}.weight_scale_2"].reshape([]) for e in expert_ids]
    )
    return _nvfp4_dequant_batched(w_stack, s1_stack, s2_stack, target_device=target_device)


# ---------------------------------------------------------------------------
# Config normalization
# ---------------------------------------------------------------------------


def _normalize_torch_dtype(cfg) -> None:
    """Normalize ``cfg.torch_dtype`` from a string (HF JSON form) to ``torch.dtype``."""
    dt = getattr(cfg, "torch_dtype", None)
    if isinstance(dt, str):
        try:
            mapped = getattr(torch, dt)
            if isinstance(mapped, torch.dtype):
                cfg.torch_dtype = mapped
                return
        except AttributeError:
            pass
        # Fall back: try Transformers' canonical mapping
        try:
            from transformers.utils import STR_TO_DTYPE  # type: ignore

            cfg.torch_dtype = STR_TO_DTYPE.get(dt, torch.bfloat16)
            return
        except Exception:
            pass
        cfg.torch_dtype = torch.bfloat16
    elif dt is None:
        cfg.torch_dtype = torch.bfloat16


def _mirror_step3p7_text_aliases(pretrained_config: PretrainedConfig) -> None:
    """Promote ``text_config`` aliases the runtime expects on the top-level config.

    The HF checkpoint stores text decoder fields (``hidden_size``,
    ``num_hidden_layers``, ``vocab_size``, ``rope_theta`` per-layer, etc.) under
    ``text_config``.  TensorRT-LLM ``model_config.from_pretrained`` already
    runs ``_mirror_text_subconfig_attrs`` for unknown attributes; this helper
    plugs in Step3p7-specific aliases that need explicit defaults.
    """
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is None:
        return

    # Normalize torch_dtype string -> torch.dtype on both top-level and text.
    # HF transformers 5.x keeps torch_dtype as the raw JSON string when read
    # via ``trust_remote_code``; TRT-LLM module constructors (Embedding,
    # Linear, RMSNorm) require an actual ``torch.dtype``.
    _normalize_torch_dtype(pretrained_config)
    _normalize_torch_dtype(text_config)

    # `num_key_value_heads` is what the rest of TRT-LLM expects.
    if not hasattr(pretrained_config, "num_key_value_heads") and hasattr(
        text_config, "num_attention_groups"
    ):
        pretrained_config.num_key_value_heads = text_config.num_attention_groups
    if not hasattr(text_config, "num_key_value_heads") and hasattr(
        text_config, "num_attention_groups"
    ):
        text_config.num_key_value_heads = text_config.num_attention_groups

    # Some downstream code reads ``max_position_embeddings`` from the top.
    if not hasattr(pretrained_config, "max_position_embeddings") and hasattr(
        text_config, "max_position_embeddings"
    ):
        pretrained_config.max_position_embeddings = text_config.max_position_embeddings

    if not hasattr(pretrained_config, "num_nextn_predict_layers") and hasattr(
        text_config, "num_nextn_predict_layers"
    ):
        pretrained_config.num_nextn_predict_layers = text_config.num_nextn_predict_layers


def _get_text_config(model_config: ModelConfig) -> PretrainedConfig:
    """Return the text sub-config, falling back to the top-level config."""
    cfg = model_config.pretrained_config
    text_cfg = getattr(cfg, "text_config", None)
    return text_cfg if text_cfg is not None else cfg


def _layer_attention_type(text_config: PretrainedConfig, layer_idx: int) -> str:
    layer_types = getattr(text_config, "layer_types", None) or []
    if layer_idx < len(layer_types):
        return layer_types[layer_idx]
    # Default: layer 0,4,8,... are full attention.
    return "full_attention" if (layer_idx % 4 == 0) else "sliding_attention"


def _layer_query_heads(text_config: PretrainedConfig, layer_idx: int) -> int:
    if _layer_attention_type(text_config, layer_idx) == "sliding_attention":
        other = getattr(text_config, "attention_other_setting", None) or {}
        if "num_attention_heads" in other:
            return int(other["num_attention_heads"])
    return int(text_config.num_attention_heads)


def _layer_kv_heads(text_config: PretrainedConfig, layer_idx: int) -> int:
    if _layer_attention_type(text_config, layer_idx) == "sliding_attention":
        other = getattr(text_config, "attention_other_setting", None) or {}
        if "num_attention_groups" in other:
            return int(other["num_attention_groups"])
    return int(
        getattr(text_config, "num_key_value_heads", getattr(text_config, "num_attention_groups"))
    )


def _layer_rope_theta(text_config: PretrainedConfig, layer_idx: int) -> float:
    theta = getattr(text_config, "rope_theta", 10000.0)
    if isinstance(theta, (list, tuple)) and layer_idx < len(theta):
        return float(theta[layer_idx])
    if isinstance(theta, (list, tuple)):
        layer_types = getattr(text_config, "layer_types", None)
        cur_layer_type = _layer_attention_type(text_config, layer_idx)
        if layer_types:
            for prev_idx in range(min(len(theta), len(layer_types)) - 1, -1, -1):
                if layer_types[prev_idx] == cur_layer_type:
                    return float(theta[prev_idx])
        return float(theta[-1])
    return float(theta)


def _layer_partial_rotary(text_config: PretrainedConfig, layer_idx: int) -> float:
    factors = getattr(text_config, "partial_rotary_factors", None)
    if factors is None:
        return 1.0
    if layer_idx < len(factors):
        return float(factors[layer_idx])
    layer_types = getattr(text_config, "layer_types", None)
    cur_layer_type = _layer_attention_type(text_config, layer_idx)
    if layer_types:
        for prev_idx in range(min(len(factors), len(layer_types)) - 1, -1, -1):
            if layer_types[prev_idx] == cur_layer_type:
                return float(factors[prev_idx])
    return float(factors[-1])


def _layer_uses_rope_scaling(text_config: PretrainedConfig, layer_idx: int) -> bool:
    """llama3 scaling only applies to layer types listed in yarn_only_types."""
    yarn_only = getattr(text_config, "yarn_only_types", None)
    if not yarn_only:
        return True
    return _layer_attention_type(text_config, layer_idx) in yarn_only


def _layer_swiglu_limit(
    text_config: PretrainedConfig, layer_idx: int, shared: bool = False
) -> Optional[float]:
    name = "swiglu_limits_shared" if shared else "swiglu_limits"
    limits = getattr(text_config, name, None)
    if limits is None or layer_idx >= len(limits):
        return None
    val = limits[layer_idx]
    if val is None or float(val) == 0.0:
        return None
    return float(val)


def _is_moe_layer(text_config: PretrainedConfig, layer_idx: int) -> bool:
    enum = getattr(text_config, "moe_layers_enum", None)
    if enum is None:
        return False
    if isinstance(enum, str):
        moe_layers = {int(x) for x in enum.split(",") if x.strip()}
    else:
        moe_layers = {int(x) for x in enum}
    return layer_idx in moe_layers


# ---------------------------------------------------------------------------
# MoE routing
# ---------------------------------------------------------------------------


class Step3p7RouterBiasHolder(nn.Module):
    """Standalone parameter holder for ``router_bias``.

    Mirrors MiniMaxM2's ``_EScoreCorrectionBiasHolder`` so the generic weight
    loader has a narrow prefix to mark consumed without dropping the rest of
    the MoE block (see the comment in MiniMaxM2 about issue #11119).

    The holder module is named ``router_bias`` on the parent ``Step3p7MoE``
    (NOT ``router_bias_holder``), so the resulting parameter path is
    ``...moe.router_bias.router_bias``.  The HF source key is
    ``...moe.router_bias`` (a 1-D tensor).  When the generic loader visits
    this module with prefix ``...moe.router_bias`` and calls ``filter_weights``,
    that prefix matches the source key exactly, so the filter yields
    ``{"": tensor}`` (empty subkey).  The ``""`` branch below handles that case
    and the explicit ``router_bias`` branch handles the per-rank shard format.
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.router_bias = nn.Parameter(
            torch.empty((num_experts,), dtype=torch.float32),
            requires_grad=False,
        )

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        w = weights[0]
        if "" in w:
            src = w[""]
        elif "router_bias" in w:
            src = w["router_bias"]
        else:
            (src,) = w.values()
        self.router_bias.copy_(src[:].to(self.router_bias.dtype))


class Step3p7MoeRoutingMethod(MiniMaxM2MoeRoutingMethod):
    """Step3p7 routing: ``sigmoid -> add bias -> top-k -> renormalize -> scale``.

    Mirrors the source ``router_bias_func`` semantics in modeling_step3p7.py.
    The unbiased sigmoid probabilities are gathered (not the biased ones);
    renormalization uses ``sum + 1e-20`` and the final weights are multiplied
    by ``routed_scaling_factor`` (3.0 for Step3p7).

    Inherits from ``MiniMaxM2MoeRoutingMethod`` so the TRTLLMGen
    ``_extract_routing_params`` helper recognises us via ``isinstance(...)``
    and feeds the bias pointer to the kernel.  The ``MiniMax2`` C++ routing
    path hard-codes ``routeScale = 1.0f`` (see ``runner.cu``), so the
    constant ``routed_scaling_factor`` is applied to the MoE output in
    ``Step3p7MoE.forward`` rather than in ``apply`` here — the Python
    ``apply`` is only invoked in the post-quant-comm path, while the no-comm
    path executes routing inside the kernel.
    """

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        callable_router_bias,
        routed_scaling_factor: float = 1.0,
        output_dtype: torch.dtype = torch.float32,
    ):
        # ``MiniMaxM2MoeRoutingMethod`` exposes the bias via the
        # ``e_score_correction_bias`` attribute name (see its docstring); we
        # name our callable to match so ``isinstance``-based lookups work.
        super().__init__(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=callable_router_bias,
            output_dtype=output_dtype,
        )
        self.routed_scaling_factor = float(routed_scaling_factor)

    # Apply path (used in the post-quant-comm branch only) reuses
    # MiniMaxM2's apply for the bias-renormalize math, then multiplies the
    # weights by ``routed_scaling_factor`` to match source semantics.
    def apply(
        self,
        router_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(router_logits.to(torch.float32))
        scores_with_bias = scores + self.router_bias.unsqueeze(0)
        _, topk_idx = torch.topk(scores_with_bias, k=self.top_k, dim=1)
        topk_weights = torch.gather(scores, 1, topk_idx)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights.to(torch.float32) * self.routed_scaling_factor
            topk_weights = topk_weights.to(self.output_dtype)
        return topk_idx.to(torch.int32), topk_weights

    @property
    def router_bias(self) -> torch.Tensor:
        # Alias for callers that read the bias under the Step3p7-native name.
        return self.callable_e_score_correction_bias()


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class _PerLayerRopeShim:
    """Materialise per-layer RoPE/q-head fields onto a copy of the config.

    TensorRT-LLM's ``RopeParams.from_config`` and ``Attention.__init__`` read
    ``num_attention_heads``, ``num_key_value_heads``, ``rope_theta``, etc. as
    top-level scalars.  Step3p7 carries them as per-layer lists; this helper
    builds a shallow copy with the layer-specific scalars applied so existing
    TRT-LLM machinery composes unchanged.
    """

    @staticmethod
    def build(text_config: PretrainedConfig, layer_idx: int) -> PretrainedConfig:
        import copy

        cfg = copy.copy(text_config)
        cfg.num_attention_heads = _layer_query_heads(text_config, layer_idx)
        cfg.num_key_value_heads = _layer_kv_heads(text_config, layer_idx)
        theta_scalar = _layer_rope_theta(text_config, layer_idx)
        cfg.rope_theta = theta_scalar
        cfg.partial_rotary_factor = _layer_partial_rotary(text_config, layer_idx)
        # Transformers 5.x stores rope_parameters as a nested dict that includes
        # the per-layer ``rope_theta`` list.  ``RopeParams.from_config`` does
        # ``config.update(rope_parameters)`` and would overwrite our scalar
        # back to the list.  Materialize a per-layer rope_parameters dict here.
        src_rope_params = getattr(text_config, "rope_parameters", None)
        if isinstance(src_rope_params, dict):
            new_params = {k: v for k, v in src_rope_params.items()}
            new_params["rope_theta"] = theta_scalar
            cfg.rope_parameters = new_params
        if not _layer_uses_rope_scaling(text_config, layer_idx):
            cfg.rope_scaling = None
            if isinstance(getattr(cfg, "rope_parameters", None), dict):
                # Strip llama3 scaling so sliding layers fall back to plain RoPE.
                stripped = {
                    k: v
                    for k, v in cfg.rope_parameters.items()
                    if k
                    not in {
                        "rope_type",
                        "factor",
                        "original_max_position_embeddings",
                        "low_freq_factor",
                        "high_freq_factor",
                    }
                }
                cfg.rope_parameters = stripped
        return cfg


class Step3p7Attention(Attention):
    """Per-layer Step3p7 attention.

    Owns its own Q/K/V/O shapes derived from the per-layer query-head count.
    Q/K Gemma-style RMSNorm and head-wise output gate are applied module-side
    so the production attention backend (FlashInfer + KVCacheManagerV2) only
    sees standard QKV.
    """

    def __init__(self, model_config: ModelConfig, layer_idx: int):
        text_config = _get_text_config(model_config)
        per_layer_cfg = _PerLayerRopeShim.build(text_config, layer_idx)
        self.text_config = text_config
        self.layer_attention_type = _layer_attention_type(text_config, layer_idx)
        self.sliding_window = (
            text_config.sliding_window if self.layer_attention_type == "sliding_attention" else None
        )
        self.head_dim = int(
            getattr(
                text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads
            )
        )
        self.use_head_wise_gate = bool(getattr(text_config, "use_head_wise_attn_gate", False))

        # RoPE: theta is per-layer; rope_scaling is per-layer-type.
        rope_params = RopeParams.from_config(per_layer_cfg)
        rope_params.theta = per_layer_cfg.rope_theta
        if not _layer_uses_rope_scaling(text_config, layer_idx):
            rope_params.scale_type = RotaryScalingType.none
            rope_params.scale = 1.0
        else:
            rope_params.scale_type = RotaryScalingType.llama3
        partial = _layer_partial_rotary(text_config, layer_idx)
        rope_params.dim = int(self.head_dim * partial)

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )

        # Build a wrapped ModelConfig view so super().__init__ sees the
        # per-layer scalars when it consults pretrained_config.
        # ``ModelConfig`` is frozen; ``pretrained_config`` is exempt from the
        # freeze (see ``ModelConfig.__setattr__``), so we can swap it in place
        # for the duration of the base Attention constructor and restore it
        # immediately after.
        original_pretrained_config = model_config.pretrained_config
        model_config.pretrained_config = per_layer_cfg
        try:
            super().__init__(
                hidden_size=text_config.hidden_size,
                num_attention_heads=per_layer_cfg.num_attention_heads,
                num_key_value_heads=per_layer_cfg.num_key_value_heads,
                max_position_embeddings=per_layer_cfg.max_position_embeddings,
                bias=False,
                pos_embd_params=pos_embd_params,
                rope_fusion=True,
                layer_idx=layer_idx,
                dtype=text_config.torch_dtype,
                dense_bias=False,
                config=model_config,
                head_dim=self.head_dim,
            )
        finally:
            model_config.pretrained_config = original_pretrained_config

        # Gemma-style Q/K norm (`weight + 1`).
        self.q_norm = RMSNorm(
            hidden_size=self.head_dim,
            eps=text_config.rms_norm_eps,
            dtype=text_config.torch_dtype,
            use_gemma=True,
        )
        self.k_norm = RMSNorm(
            hidden_size=self.head_dim,
            eps=text_config.rms_norm_eps,
            dtype=text_config.torch_dtype,
            use_gemma=True,
        )

        # Head-wise output gate projection.  Hidden_size -> num_heads.
        if self.use_head_wise_gate:
            self.g_proj = Linear(
                in_features=text_config.hidden_size,
                out_features=per_layer_cfg.num_attention_heads,
                bias=False,
                dtype=text_config.torch_dtype,
                mapping=self.qkv_proj.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=False,
                quant_config=None,
            )

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_view = q.reshape(-1, self.head_dim)
        k_view = k.reshape(-1, self.head_dim)
        q = self.q_norm(q_view).reshape(q.shape)
        k = self.k_norm(k_view).reshape(k.shape)
        return q, k

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        # Split (q,k,v) first because module-side QK norm requires the
        # heads-separated layout.  Apply Gemma RMSNorm, then defer to the
        # base Attention's unfused RoPE path so backend selection still
        # owns the actual rotation.
        q, k, v = self.split_qkv(q, k, v)
        q, k = self.apply_qk_norm(q, k)
        q, k, v = super().apply_rope(q, k, v, position_ids)
        return q, k, v

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Step3p7 attention with module-side QK-norm + RoPE + head gate.

        We bypass the base ``Attention.forward`` so we can apply the source's
        per-head output gate between the attention backend and ``o_proj``.
        The gate is computed from the *pre-attention* hidden states via
        ``g_proj`` (output = num_heads), broadcast over head_dim, and
        multiplied into the attention output before the o_proj.

        Helix CP, LoRA, and attention sinks from the base class are
        intentionally *not* plumbed here.  Future iterations can re-enable
        those by upstreaming a per-head gate hook into ``Attention.forward``.
        """
        # Use the per-layer sliding window unless explicitly overridden.
        effective_window = (
            attention_window_size if attention_window_size is not None else self.sliding_window
        )

        if (
            not self.rope_fusion
            and getattr(attn_metadata, "is_spec_dec_dynamic_tree", False)
            and getattr(attn_metadata, "use_spec_decoding", False)
            and getattr(attn_metadata, "spec_decoding_position_offsets", None) is not None
            and attn_metadata.spec_decoding_position_offsets.dim() == 1
            and position_ids is not None
        ):
            position_ids = self._adjust_position_ids_for_spec_dec(
                position_ids.clone(), attn_metadata
            ).clamp_min_(0)

        num_text_layers = int(getattr(self.text_config, "num_hidden_layers", 0))
        if position_ids is not None and self.layer_idx >= num_text_layers:
            position_ids = position_ids.clamp_min(0)

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv, None, None
        q, k, v = self.apply_rope(q, k, v, position_ids)
        q, k, v = self.convert_qkv(q, k, v)

        attn_output = self.forward_impl(
            q,
            k,
            v,
            attn_metadata,
            attention_mask,
            effective_window,
            kwargs.get("attention_mask_data"),
            mrope_config=kwargs.get("mrope_config"),
            attention_sinks=None,
            has_lora=False,
        )

        if self.use_head_wise_gate:
            gate = self.g_proj(hidden_states)
            head_dim = self.head_dim
            # ``self.num_heads`` is already the per-rank head count after
            # Attention.__init__ shards by tp_size; do NOT divide again.
            num_heads_tp = self.num_heads
            # attn_output is [tokens, num_heads_tp * head_dim].  Reshape so
            # we can broadcast the per-head gate scalar over head_dim.
            orig_shape = attn_output.shape
            attn_output = attn_output.view(*orig_shape[:-1], num_heads_tp, head_dim)
            # Source semantics: ``gate_states.unsqueeze(-1).sigmoid()``
            # broadcasts the per-head scalar over ``head_dim``.  gate has
            # shape ``(tokens, num_heads_tp)``, unsqueeze to
            # ``(tokens, num_heads_tp, 1)``, then sigmoid.
            attn_output = attn_output * gate.unsqueeze(-1).sigmoid()
            attn_output = attn_output.view(*orig_shape)

        attn_output = self.o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# MLP / MoE / Decoder layer
# ---------------------------------------------------------------------------


class _ClampedGatedMLP(GatedMLP):
    """Dense SwiGLU MLP with optional per-layer clamp on gate/up activations.

    Mirrors source ``Step3p7MLP``: when ``swiglu_limit`` is set, ``gate`` is
    clamped to ``[-, limit]`` and ``up`` is clamped to ``[-limit, limit]``
    before the elementwise multiply.  Otherwise it behaves exactly like the
    standard ``GatedMLP``.

    The class **inherits** from ``GatedMLP`` rather than wrapping it so that
    the weight loader sees the standard module path
    ``...mlp.gate_up_proj`` / ``...mlp.down_proj`` matching the HF source
    convention with ``params_map['gate_up_proj'] = ['gate_proj', 'up_proj']``.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        intermediate_size: int,
        swiglu_limit: Optional[float],
        is_shared_expert: bool = False,
    ):
        text_config = _get_text_config(model_config)
        super().__init__(
            hidden_size=text_config.hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            dtype=text_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            # Shared experts must not all-reduce; the routed MoE call does it.
            reduce_output=not is_shared_expert,
            overridden_tp_size=None,
            is_shared_expert=is_shared_expert,
        )
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.swiglu_limit is None:
            return super().forward(hidden_states, **kwargs)
        # Compute gate/up separately to apply the clamp like the source does.
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = torch.nn.functional.silu(gate)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        hidden = gate * up
        return self.down_proj(hidden)


class Step3p7MoE(nn.Module):
    """Routed FP8 block-scale MoE block (router gate + routed experts).

    Routing is sigmoid + bias top-k with renormalization and scaling.
    Routed experts are FP8 block-scale.

    The shared expert is intentionally **not** owned by this module — it lives
    on the decoder layer because the HF source stores it at
    ``model.layers.<i>.share_expert.{gate,up,down}_proj.weight`` (no ``moe.``
    prefix) and TensorRT-LLM's weight loader works module path by module path.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        text_config = _get_text_config(model_config)
        self.layer_idx = layer_idx
        self.hidden_size = text_config.hidden_size
        self.num_experts = int(text_config.moe_num_experts)
        self.top_k = int(text_config.moe_top_k)
        self.moe_intermediate_size = int(text_config.moe_intermediate_size)
        self.routed_scaling_factor = float(getattr(text_config, "moe_router_scaling_factor", 1.0))
        self.need_fp32_gate = bool(getattr(text_config, "need_fp32_gate", False))
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        # Router (gate) projection.  Source computes router logits in fp32.
        gate_dtype = torch.float32 if self.need_fp32_gate else text_config.torch_dtype
        self.gate = Linear(
            in_features=self.hidden_size,
            out_features=self.num_experts,
            bias=False,
            dtype=gate_dtype,
            quant_config=None,
        )

        # Router bias holder: name the holder ``router_bias`` so the resulting
        # parameter path matches the HF source key prefix exactly.  See the
        # docstring of ``Step3p7RouterBiasHolder`` for the loader contract.
        self.router_bias = Step3p7RouterBiasHolder(self.num_experts)
        routing_method = Step3p7MoeRoutingMethod(
            top_k=self.top_k,
            num_experts=self.num_experts,
            callable_router_bias=lambda: self.router_bias.router_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        # Per-layer SwiGLU clamp limit for routed experts.  Source ``Step3p7MoEMLP``
        # applies ``gate = gate.clamp(max=limit)`` and ``up = up.clamp(-limit, limit)``
        # when ``swiglu_limits[layer_idx]`` is non-zero -- needed for layers 43/44
        # of Step3p7-Flash-FP8 (limit=7).  Neither TRTLLMGen FP8-block-scale
        # (the production B200 path) nor Cutlass FP8-block-scale (DeepGEMM is
        # Hopper-only) accepts ``swiglu_limit`` for fp8_block_scales today.  As
        # the design-review's MoE Direction C explicitly allows, those two
        # clamp-active layers run a local Python expert path on bf16-dequant
        # weights so the source-defined clamp can be expressed.
        #
        # The optional ``STEP3P7_PYTHON_FALLBACK_THRESHOLD`` knob extends the
        # same Python fp32 expert path to later layers when HF-matching expert
        # math is needed.  It defaults to ``999`` (disabled).  The FP8 backend
        # tensors stay allocated for clamp and fallback layers because backend
        # metadata and CUDA graph setup expect those tensors to remain present.
        scalar_limit = _layer_swiglu_limit(text_config, layer_idx, shared=False)
        self._routed_swiglu_limit: Optional[float] = scalar_limit
        _clamp_active = scalar_limit is not None and scalar_limit > 0
        try:
            _python_threshold = int(os.environ.get("STEP3P7_PYTHON_FALLBACK_THRESHOLD", "999"))
        except ValueError:
            _python_threshold = 999
        _late_layer = layer_idx >= _python_threshold
        # BF16 checkpoint kernel workaround.  The flashinfer
        # ``trtllm_bf16_moe`` kernel (selected by ``BF16TRTLLMGenFusedMoEMethod``
        # for the unquantized routed-expert path) produces functionally wrong
        # tokens for Step3p7's geometry / routing combination
        # (288 experts, top-k 8, MiniMax2 sigmoid+bias routing, no
        # routed_scaling_factor inside the kernel, hidden=4096,
        # moe_intermediate=1280, BlockMajorK shuffle).  The FP8 path goes
        # through a different kernel (``run_fp8_block_scale_moe``).  Until the
        # BF16-kernel root cause is fixed upstream, the BF16 checkpoint forces
        # the Python fp32 expert path for every MoE layer.  The Python path is
        # CUDA-graph safe (fixed iteration count, no ``.nonzero()`` / ``.any()``
        # / variable indexing), so it composes with ``CudaGraphConfig()``.
        quant_config_obj = getattr(model_config, "quant_config", None)
        quant_algo_val = (
            getattr(quant_config_obj, "quant_algo", None) if quant_config_obj is not None else None
        )
        is_bf16_checkpoint = quant_algo_val is None
        self._use_python_clamp: bool = bool(_clamp_active or _late_layer or is_bf16_checkpoint)
        if _clamp_active:
            self._python_path_reason: str = "clamp"
        elif _late_layer:
            self._python_path_reason = "late_layer"
        elif is_bf16_checkpoint:
            self._python_path_reason = "bf16_kernel_workaround"
        else:
            self._python_path_reason = ""

        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=routing_method,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            aux_stream_dict=aux_stream_dict,
            dtype=text_config.torch_dtype,
            reduce_results=False,
            model_config=model_config,
            layer_idx=layer_idx,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        )

        # When the layer has a non-zero SwiGLU clamp limit, allocate a parallel
        # bf16 expert weight set.  These are filled at weight-load time by
        # dequantising the on-disk FP8 block-scale tensors via
        # ``_fp8_block_dequant``.  The forward path for these layers replaces
        # the FP8 MoE call with a Python expert loop that applies the source
        # clamp.  Total bf16 storage per layer per rank:
        # ``num_local_experts * (2*moe_moe_interediate_size + hidden_size)
        #  * moe_moe_interediate_size * 2 bytes``  -- about 1.1 GiB at
        # ``num_local_experts=36``.
        if self._use_python_clamp:
            num_local_experts = max(1, self.num_experts // max(1, self.mapping.moe_ep_size))
            # Match HF's expert-local weight orientation: gate_proj (w1)
            # is (moe_interediate, hidden); up_proj (w3) is (moe_interediate,
            # hidden); down_proj (w2) is (hidden, moe_interediate).
            # Use non-persistent buffers (NOT nn.Parameters) so the generic
            # checkpoint loader skips them (it only walks ``_parameters``,
            # not ``_buffers``).  We fill them ourselves in
            # ``load_clamp_weights_from_fp8_experts`` after the FP8 backend
            # has materialised its quantised tensors.
            self._clamp_num_local_experts = num_local_experts
            self.register_buffer(
                "_clamp_gate_proj",
                torch.empty(
                    num_local_experts,
                    self.moe_intermediate_size,
                    self.hidden_size,
                    dtype=text_config.torch_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_clamp_up_proj",
                torch.empty(
                    num_local_experts,
                    self.moe_intermediate_size,
                    self.hidden_size,
                    dtype=text_config.torch_dtype,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_clamp_down_proj",
                torch.empty(
                    num_local_experts,
                    self.hidden_size,
                    self.moe_intermediate_size,
                    dtype=text_config.torch_dtype,
                ),
                persistent=False,
            )
            self._clamp_local_expert_ids: Optional[torch.Tensor] = None
            self._clamp_weights_loaded: bool = False

        self.allreduce = None
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=self.mapping, strategy=model_config.allreduce_strategy
            )

    def load_clamp_weights_from_fp8_experts(self) -> None:
        """Populate ``_clamp_{gate,up,down}_proj`` by dequantising FP8 experts.

        Called once by the model loader after ``super().load_weights`` finishes
        materialising the FP8 routed-expert weights in ``self.experts``.  For
        every layer with a non-zero SwiGLU clamp limit, we dequantise the
        rank-local subset of FP8 expert weights to bf16 and store them in the
        pre-allocated ``_clamp_*_proj`` parameters.  The forward path for those
        layers replaces the FP8 MoE op (which the production B200 backends
        cannot pair with ``swiglu_limit``) with a Python expert loop that
        applies ``gate.clamp(max=limit)`` and ``up.clamp(-limit,limit)``.

        For the bf16 reference checkpoint, the TRTLLMGen-Bf16 backend's
        ``post_load_weights`` reshuffles ``w3_w1_weight`` into a 4D
        BlockMajorK layout that cannot be sliced/transposed back to 2D in
        place.  In that case ``Step3p7ForCausalLM.load_weights`` calls
        ``_capture_bf16_clamp_weights`` earlier (before the backend layout
        transform runs) to populate the buffers directly from the HF
        weights dict, sets ``_clamp_weights_loaded=True``, and we short-
        circuit here.
        """
        if not self._use_python_clamp:
            return
        if self._clamp_weights_loaded:
            return
        # ``self.experts`` is a ``ConfigurableMoE`` wrapper around the actual
        # backend (TRTLLMGenFusedMoE / CutlassFusedMoE).  The expert weight
        # tensors live on the backend.
        e = getattr(self.experts, "backend", None) or self.experts
        if not hasattr(e, "w3_w1_weight") or not hasattr(e, "w2_weight"):
            # The backend has not exposed the expert weight tensors yet.  Skip;
            # the forward path will fall back to ``self.experts`` (which will
            # at least produce output, even if without the clamp).
            return
        w3_w1 = e.w3_w1_weight.data
        w2 = e.w2_weight.data
        moe_inter = self.moe_intermediate_size
        # Slice w3 (up) and w1 (gate) halves along the moe_interediate dim.
        # ``w3_w1`` layout is ``[w3 (up), w1 (gate)]`` -- see
        # ``DeepSeekFP8BlockScalesFusedMoEMethod.load_expert_all_*`` in
        # ``fused_moe/quantization.py`` (the ``chunk(2, dim=0)`` yields
        # ``(w3, w1)``).  Same layout is used by the bf16 backend.
        w3 = w3_w1[:, :moe_inter, :]
        w1 = w3_w1[:, moe_inter:, :]
        scale_w3_w1 = getattr(e, "w3_w1_weight_scaling_factor", None)
        scale_w2 = getattr(e, "w2_weight_scaling_factor", None)
        is_fp8 = (
            scale_w3_w1 is not None and scale_w2 is not None and w3_w1.dtype == torch.float8_e4m3fn
        )
        if is_fp8:
            # FP8 block-scale path: dequantize per-block to bf16.
            BLOCK = 128
            Iblk = (moe_inter + BLOCK - 1) // BLOCK
            w3_scale = scale_w3_w1.data[:, :Iblk, :]
            w1_scale = scale_w3_w1.data[:, Iblk:, :]
            up_bf16 = _fp8_block_dequant_3d(w3, w3_scale, BLOCK)
            gate_bf16 = _fp8_block_dequant_3d(w1, w1_scale, BLOCK)
            down_bf16 = _fp8_block_dequant_3d(w2, scale_w2.data, BLOCK)
        else:
            # Unquantized bf16/fp16 backend: weights are already in compute
            # dtype, just slice and copy.  Used by the bf16 reference
            # checkpoint (``step-3-7-flash-bf16``) where no FP8
            # weight_scale_inv tensors exist.
            up_bf16 = w3.to(torch.bfloat16).contiguous()
            gate_bf16 = w1.to(torch.bfloat16).contiguous()
            down_bf16 = w2.to(torch.bfloat16).contiguous()
        # Resize self._clamp_* buffers if the FP8 backend's actual local-
        # expert count differs from what __init__ guessed (some EP modes
        # round to a multiple of TP).  We trust the backend's tensor shapes.
        if self._clamp_gate_proj.shape != gate_bf16.shape:
            self.register_buffer("_clamp_gate_proj", torch.empty_like(gate_bf16), persistent=False)
            self.register_buffer("_clamp_up_proj", torch.empty_like(up_bf16), persistent=False)
            self.register_buffer("_clamp_down_proj", torch.empty_like(down_bf16), persistent=False)
            self._clamp_num_local_experts = int(gate_bf16.shape[0])
        self._clamp_gate_proj.copy_(gate_bf16)
        self._clamp_up_proj.copy_(up_bf16)
        self._clamp_down_proj.copy_(down_bf16)
        self._clamp_weights_loaded = True
        del up_bf16, gate_bf16, down_bf16
        # Keep the FP8 backend tensors allocated for clamp layers.  Autotuning,
        # CUDA-graph setup, and backend metadata can still depend on their
        # shapes after weight loading.

    def _python_clamped_moe_forward(
        self, h: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Python expert path with explicit SwiGLU clamp.

        Mirrors HF source ``Step3p7MoEMLP.get_expert_output``:

            gate = silu(gate_proj(x)).clamp(max=limit)
            up   = up_proj(x).clamp(-limit, limit)
            out  = down_proj(gate * up)

        Uses rank-local bf16 expert weights and produces a per-rank partial
        sum compatible with the FP8 backend's pre-allreduce semantics: the
        decoder layer's all-reduce on ``hidden_states = routed + shared``
        will combine partials across EP ranks.

        **CUDA-graph capture friendly:** the per-expert loop has a fixed
        Python iteration count (``num_local`` is constant per layer), and
        every op inside the loop produces a fixed-shape tensor.  In
        particular we avoid ``mask.any()`` / ``mask.nonzero()`` /
        ``index_add_`` with variable token indices -- those trigger host
        sync and are illegal during ``torch.cuda.graph`` capture.  Instead
        we precompute a dense ``weight_per_expert`` matrix of shape
        ``(N, num_experts)`` via ``scatter_add_`` (fixed shape), then for
        every local expert compute its output for all N tokens and
        multiply by the corresponding column (zero for tokens not routed
        to that expert).  Wasted compute scales by
        ``num_local / top_k = 36 / 8 = 4.5x`` over the masked version,
        which is negligible at Step3p7 layer dimensions (~37 GFLOPs/layer/
        rank/forward at N=68 prefill tokens).
        """
        routing = self.experts.routing_method
        topk_idx, topk_weights = routing.apply(router_logits)
        # topk_idx: (N, top_k) int32 global expert IDs;
        # topk_weights: (N, top_k) float32 already-scaled weights
        N, _ = topk_idx.shape
        ep_rank = self.mapping.moe_ep_rank
        num_local = self._clamp_num_local_experts
        local_start = ep_rank * num_local
        limit = float(self._routed_swiglu_limit or 0.0)

        # Build the dense (N, num_experts) routing matrix: entry [i, e] is
        # the routing weight for token i if expert e is in the top-k of
        # token i, else 0.  ``scatter_add_`` keeps the operation
        # fixed-shape (independent of how many tokens route to expert e).
        weight_per_expert = torch.zeros((N, self.num_experts), dtype=torch.float32, device=h.device)
        weight_per_expert.scatter_add_(1, topk_idx.long(), topk_weights.to(torch.float32))

        # HF source computes experts in fp32 (``MoELinear.forward`` casts
        # both ``x`` and ``self.weight[expert_id]`` to float before
        # ``F.linear``).  Cast once for all experts; reused across the
        # num_local matmuls below.
        x_f32 = h.to(torch.float32)

        # Source ``Step3p7MoEMLP`` accumulates into a tensor with the same
        # dtype as ``hidden_states``: each fp32 expert result is cast back to
        # bf16 before ``index_add_``.  Keep that rounding point here instead
        # of accumulating all local experts in fp32 and casting only once.
        output = torch.zeros((N, self.hidden_size), dtype=h.dtype, device=h.device)
        for local_e in range(num_local):
            global_e = local_start + local_e
            gate_w = self._clamp_gate_proj[local_e].to(torch.float32)
            up_w = self._clamp_up_proj[local_e].to(torch.float32)
            down_w = self._clamp_down_proj[local_e].to(torch.float32)
            gate = torch.nn.functional.silu(x_f32 @ gate_w.t())
            up = x_f32 @ up_w.t()
            if limit > 0.0:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            expert_out = (gate * up) @ down_w.t()
            # weight_per_expert[:, global_e] is zero for tokens not routed
            # to this expert, so the multiply masks out their contribution
            # without any dynamic-shape indexing.
            expert_out = expert_out * weight_per_expert[:, global_e : global_e + 1]
            output = output + expert_out.to(h.dtype)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_size
        orig_shape = hidden_states.shape
        h = hidden_states.view(-1, self.hidden_size)

        if self.need_fp32_gate:
            router_logits = self.gate(h.to(torch.float32))
        else:
            router_logits = self.gate(h)

        # The Python SwiGLU-clamp forward is CUDA-graph capture safe: it
        # uses only fixed-shape ops (``scatter_add_`` for the dense
        # weight-per-expert matrix, a Python-level loop over the constant
        # ``num_local`` count, and per-iteration matmul + clamp).  No
        # ``.nonzero()`` / ``.any()`` / variable ``index_add_`` calls --
        # all of which would trigger host sync during capture.
        if self._use_python_clamp and self._clamp_weights_loaded:
            routed = self._python_clamped_moe_forward(h, router_logits)
            # ``Step3p7MoeRoutingMethod.apply`` (used by the Python forward)
            # already multiplies the per-expert weights by
            # ``routed_scaling_factor``, so the outer scaling below must be
            # skipped for this path (otherwise the scaling is doubled).
            _routed_scaling_already_applied = True
        else:
            routed = self.experts(
                h,
                router_logits,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                use_dp_padding=False,
            )
            _routed_scaling_already_applied = False
        # The TRTLLMGen MiniMax2 routing path hard-codes ``routeScale = 1.0`` in
        # ``runner.cu`` — see the routing impl comment in
        # ``Step3p7MoeRoutingMethod``.  Apply the source ``routed_scaling_factor``
        # to the MoE output here so the no-comm path matches semantics.  This is
        # mathematically equivalent to scaling every topk_weight (sum of
        # scaled_weight × expert_out == scale × sum of unscaled_weight × expert_out).
        if self.routed_scaling_factor != 1.0 and not _routed_scaling_already_applied:
            routed = routed * self.routed_scaling_factor
        return routed.view(orig_shape)


def _model_config_without_quant(model_config: ModelConfig) -> ModelConfig:
    """Return a shallow copy of ``model_config`` with ``quant_config=None``.

    Used to keep the bf16 paths (attention, shared expert, dense MLP, router
    gate) out of the FP8 quantization the routed experts use.

    ``ModelConfig`` is a frozen dataclass except for ``quant_config`` /
    ``pretrained_config`` / ``extra_attrs`` / ``_frozen`` (see
    ``ModelConfig.__setattr__``).  We work around the freeze by temporarily
    flipping ``_frozen`` on the clone to clear ``quant_config_dict`` too.
    """
    import copy as _copy

    from tensorrt_llm.models.modeling_utils import QuantConfig

    cloned = _copy.copy(model_config)
    # quant_config is exempt from the freeze; force it to a no-op QuantConfig.
    cloned.quant_config = QuantConfig()
    object.__setattr__(cloned, "_frozen", False)
    try:
        cloned.quant_config_dict = None
    finally:
        object.__setattr__(cloned, "_frozen", True)
    return cloned


class Step3p7DecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        text_config = _get_text_config(model_config)
        self.layer_idx = layer_idx
        self.hidden_size = text_config.hidden_size

        # Attention, dense MLPs, and shared experts are NOT routed-expert FP8
        # / NVFP4.  The attention block still wants ``kv_cache_quant_algo`` so
        # FP8 KV cache (set by modelopt) flows through the attention backend.
        bf16_model_config = _model_config_without_quant(model_config)
        attn_model_config = _model_config_keep_kv_quant(model_config)
        self.self_attn = Step3p7Attention(attn_model_config, layer_idx)

        self.is_moe_layer = _is_moe_layer(text_config, layer_idx)
        if self.is_moe_layer:
            # Routed experts may be FP8; router gate + bias holder live inside
            # ``moe`` and the shared expert lives on the decoder layer (the
            # HF source stores its weights at ``share_expert.*``, not
            # ``moe.share_expert.*``).
            self.moe = Step3p7MoE(
                model_config, layer_idx=layer_idx, aux_stream_dict=aux_stream_dict
            )
            shared_intermediate_size = int(
                getattr(text_config, "share_expert_dim", text_config.moe_intermediate_size)
            )
            shared_swiglu_limit = _layer_swiglu_limit(text_config, layer_idx, shared=True)
            self.share_expert = _ClampedGatedMLP(
                bf16_model_config,
                layer_idx=layer_idx,
                intermediate_size=shared_intermediate_size,
                swiglu_limit=shared_swiglu_limit,
                is_shared_expert=True,
            )
            self.mlp = None
        else:
            # Dense MLP layers (0..2) use the model's main ``intermediate_size``,
            # not the per-expert ``moe_intermediate_size``.
            self.mlp = _ClampedGatedMLP(
                bf16_model_config,
                layer_idx=layer_idx,
                intermediate_size=int(text_config.intermediate_size),
                swiglu_limit=_layer_swiglu_limit(text_config, layer_idx, shared=False),
                is_shared_expert=False,
            )
            self.moe = None
            self.share_expert = None

        # All Step3p7 RMSNorms use Gemma-style ``(weight + 1)`` scaling.
        # The source ``Step3p7RMSNorm`` (in modeling_step3p7.py at the
        # checkpoint root) defines exactly this formula, and the layer
        # norms (input_layernorm, post_attention_layernorm, final norm) all
        # use it -- not just the Q/K norms.  Regular RMSNorm multiplies by
        # ``weight`` alone instead of ``(weight + 1)``.
        self.input_layernorm = RMSNorm(
            hidden_size=text_config.hidden_size,
            eps=text_config.rms_norm_eps,
            dtype=text_config.torch_dtype,
            use_gemma=True,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=text_config.hidden_size,
            eps=text_config.rms_norm_eps,
            dtype=text_config.torch_dtype,
            use_gemma=True,
        )

        # Optional final all-reduce when running TP without attention DP.
        self.allreduce = None
        if (
            self.is_moe_layer
            and not model_config.mapping.enable_attention_dp
            and model_config.mapping.tp_size > 1
        ):
            self.allreduce = AllReduce(
                mapping=model_config.mapping, strategy=model_config.allreduce_strategy
            )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.moe is not None:
            routed = self.moe(hidden_states, attn_metadata)
            shared = self.share_expert(hidden_states)
            hidden_states = routed + shared
            if self.allreduce is not None:
                hidden_states = self.allreduce(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ---------------------------------------------------------------------------
# Top-level model & registration
# ---------------------------------------------------------------------------


class Step3p7TextModel(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        text_config = _get_text_config(model_config)
        self.vocab_size = int(text_config.vocab_size)
        self.num_hidden_layers = int(text_config.num_hidden_layers)
        self.aux_stream_dict = {
            AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
            AuxStreamType.MoeBalancer: torch.cuda.Stream(),
            AuxStreamType.MoeOutputMemset: torch.cuda.Stream(),
        }

        self.embed_tokens = Embedding(
            self.vocab_size,
            int(text_config.hidden_size),
            dtype=text_config.torch_dtype,
        )
        self.layers = nn.ModuleList(
            [
                Step3p7DecoderLayer(model_config, idx, self.aux_stream_dict)
                for idx in range(self.num_hidden_layers)
            ]
        )
        # Final RMSNorm is also Gemma-style ``(weight + 1)``.
        self.norm = RMSNorm(
            hidden_size=int(text_config.hidden_size),
            eps=text_config.rms_norm_eps,
            dtype=text_config.torch_dtype,
            use_gemma=True,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers[: self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                **kwargs,
            )
        if residual is not None:
            hidden_states = (hidden_states + residual).to(hidden_states.dtype)
        return hidden_states


class Step3p7MTPHead(nn.Module):
    """Step3p7 MTP shared head.

    The checkpoint stores a per-MTP-layer head under
    ``model.layers.<idx>.transformer.shared_head``.  The MTP block returns the
    residual-added hidden states, and this module applies ``norm`` before the
    per-layer output projection to match the Step3p5/Step3p7 reference layout.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        text_config = _get_text_config(model_config)
        self.model_config = model_config
        self.norm = RMSNorm(
            hidden_size=int(text_config.hidden_size),
            eps=text_config.rms_norm_eps,
            dtype=text_config.torch_dtype,
            use_gemma=True,
        )
        mapping = model_config.mapping
        if mapping.enable_attention_dp and not mapping.enable_lm_head_tp_in_adp:
            self.output = LMHead(
                int(text_config.vocab_size),
                int(text_config.hidden_size),
                dtype=text_config.torch_dtype,
            )
        else:
            self.output = LMHead(
                int(text_config.vocab_size),
                int(text_config.hidden_size),
                dtype=text_config.torch_dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                reduce_output=False,
            )
        self.mapping_lm_head_tp = None

    def _get_last_token_states(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        last_tokens = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
        return hidden_states[last_tokens]

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: LMHead,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        del lm_head
        if not return_context_logits:
            if attn_metadata is not None:
                hidden_states = self._get_last_token_states(hidden_states, attn_metadata)
            else:
                hidden_states = hidden_states[-1].unsqueeze(0)

        mapping = self.model_config.mapping
        enable_attention_dp = mapping.enable_attention_dp
        enable_lm_head_tp_in_adp = enable_attention_dp and mapping.enable_lm_head_tp_in_adp

        if enable_lm_head_tp_in_adp:
            self.mapping_lm_head_tp = create_lm_head_tp_mapping(mapping, hidden_states.shape[0])
            hidden_states = allgather(hidden_states, self.mapping_lm_head_tp, dim=0)

        hidden_states = self.norm(hidden_states)

        if not enable_attention_dp or enable_lm_head_tp_in_adp:
            self.output.gather_output = False
        logits = self.output(
            hidden_states,
            mapping_lm_head_tp=self.mapping_lm_head_tp,
            is_spec_decoding_head=True,
        )
        if not enable_attention_dp or enable_lm_head_tp_in_adp:
            self.output.gather_output = True
        return logits


class Step3p7MTP(nn.Module):
    """Step3p7 native MTP predictor layer.

    This mirrors vLLM's Step3p5-MTP layout for this checkpoint:
    ``enorm(embed)``, ``hnorm(target_hidden)``, ``eh_proj(concat(...))``, a
    regular Step3p7 decoder block under ``mtp_block``, then
    a residual add.  ``shared_head.norm`` is applied by ``Step3p7MTPHead`` when
    draft logits are requested.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        text_config = _get_text_config(model_config)
        self.model_config = model_config
        self.enorm = RMSNorm(
            hidden_size=int(text_config.hidden_size),
            eps=text_config.rms_norm_eps,
            dtype=text_config.torch_dtype,
            use_gemma=True,
        )
        self.hnorm = RMSNorm(
            hidden_size=int(text_config.hidden_size),
            eps=text_config.rms_norm_eps,
            dtype=text_config.torch_dtype,
            use_gemma=True,
        )
        if model_config.mapping.enable_attention_dp:
            self.eh_proj = Linear(
                in_features=int(text_config.hidden_size) * 2,
                out_features=int(text_config.hidden_size),
                bias=False,
                dtype=text_config.torch_dtype,
                quant_config=None,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
        else:
            self.eh_proj = Linear(
                in_features=int(text_config.hidden_size) * 2,
                out_features=int(text_config.hidden_size),
                bias=False,
                dtype=text_config.torch_dtype,
                tensor_parallel_mode=TensorParallelMode.ROW,
                mapping=model_config.mapping,
                reduce_output=True,
                quant_config=None,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
        self.mtp_block = Step3p7DecoderLayer(model_config, layer_idx, aux_stream_dict)
        self.shared_head = Step3p7MTPHead(model_config)

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        del all_rank_num_tokens
        inputs_embeds = self.enorm(embed_tokens(input_ids))
        hidden_states = self.hnorm(hidden_states)
        hidden_states = torch.concat([inputs_embeds, hidden_states], dim=-1)

        mapping = self.model_config.mapping
        if mapping.tp_size > 1 and not mapping.enable_attention_dp:
            hidden_states = torch.chunk(hidden_states, mapping.tp_size, dim=-1)[mapping.tp_rank]
        hidden_states = self.eh_proj(hidden_states)

        hidden_states, residual = self.mtp_block(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=None,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if residual is not None:
            hidden_states = (hidden_states + residual).to(hidden_states.dtype)
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(0, hidden_states, None)
        return hidden_states


_STACKED_MOE_PROJ_TO_W = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


# Suffixes carried by stacked routed-expert tensors that need per-expert split.
# Covers FP8 block-scale (``weight_scale_inv``) and NVFP4
# (``weight_scale`` + ``weight_scale_2`` + ``input_scale``) checkpoints.
_STACKED_MOE_WEIGHT_SUFFIXES = (
    "weight",
    "weight_scale_inv",
    "weight_scale",
    "weight_scale_2",
    "input_scale",
)


def split_stacked_moe_weights(weights, text_config: PretrainedConfig) -> int:
    """Expand stacked routed-expert tensors into per-expert keys, in place.

    The HF source stores each routed-expert projection as a single stacked
    tensor of shape ``(num_experts, ...)``:

    - ``model.layers.<i>.moe.gate_proj.weight`` shape ``(N, moe_interediate, hidden[/2])``
    - FP8 block scale checkpoints also carry
      ``model.layers.<i>.moe.gate_proj.weight_scale_inv`` shape ``(N, I/128, H/128)``.
    - NVFP4 checkpoints carry ``weight_scale`` (FP8 per-16 block scale),
      ``weight_scale_2`` (FP32 per-tensor global scale, shape ``(N,)``), and
      ``input_scale`` (FP32 per-tensor activation scale, shape ``(N,)``).

    The TensorRT-LLM ``VANILLA`` MoE loader expects per-expert keys
    ``experts.<e>.w{1,2,3}.<suffix>``.  This helper slices the stacked tensors
    along dim 0 and writes one view per expert under the per-expert key, then
    deletes the stacked keys so the generic loader's ``mark_consumed`` path
    stays consistent.

    Returns the number of MoE layers split (for diagnostics / tests).
    """
    num_layers = int(text_config.num_hidden_layers)
    num_experts = int(text_config.moe_num_experts)

    moe_layer_indices = [i for i in range(num_layers) if _is_moe_layer(text_config, i)]

    layers_split = 0
    for layer_idx in moe_layer_indices:
        prefix = f"model.layers.{layer_idx}.moe."
        # First pass: collect available stacked keys so we don't half-split.
        proj_present = []
        for proj in _STACKED_MOE_PROJ_TO_W:
            stacked_w = f"{prefix}{proj}.weight"
            if stacked_w in weights:
                proj_present.append(proj)
        if not proj_present:
            continue
        layers_split += 1

        for proj in proj_present:
            dst_w = _STACKED_MOE_PROJ_TO_W[proj]
            for suffix in _STACKED_MOE_WEIGHT_SUFFIXES:
                stacked_key = f"{prefix}{proj}.{suffix}"
                if stacked_key not in weights:
                    continue
                stacked = weights[stacked_key]
                # Tensor shape (N, ...); torch tensors support direct view
                # indexing along dim 0 cheaply (no data copy).
                if stacked.shape[0] != num_experts:
                    raise RuntimeError(
                        f"Step3p7 stacked MoE tensor {stacked_key} has "
                        f"leading dim {stacked.shape[0]} but "
                        f"num_experts={num_experts}."
                    )
                for expert_id in range(num_experts):
                    new_key = f"{prefix}experts.{expert_id}.{dst_w}.{suffix}"
                    weights[new_key] = stacked[expert_id]
                del weights[stacked_key]
    return layers_split


def _prepare_step3p7_mtp_spec_config(model_config: ModelConfig) -> None:
    """Populate MTP config fields before ``SpecDecOneEngineForCausalLM`` builds draft layers."""
    spec_config = getattr(model_config, "spec_config", None)
    if spec_config is None or getattr(spec_config, "decoding_type", None) != "MTP":
        return
    model_layers = int(getattr(model_config.pretrained_config, "num_nextn_predict_layers", 0) or 0)
    if model_layers <= 0:
        return

    spec_config.num_nextn_predict_layers = model_layers
    if not spec_config.spec_dec_mode.is_mtp_vanilla():
        return

    user_set = "max_draft_len" in spec_config.model_fields_set
    if not user_set or spec_config.max_draft_len >= model_layers:
        spec_config.max_draft_len = model_layers
    spec_config.max_total_draft_tokens = spec_config.max_draft_len


_MTP_DIRECT_WEIGHT_PREFIXES = (
    "enorm.",
    "hnorm.",
    "eh_proj.",
    "shared_head.",
)


# Some Step3p7 checkpoints (multimodal / NIM-VNIM packaging, e.g. the NVFP4
# release ``step-3.7-flash-nim_vnim-nvfp4-260528``) nest the text decoder
# under ``model.language_model.*`` instead of ``model.*``, and the vision
# tower / projector under ``model.vision_model.*`` and
# ``model.vit_large_projector.*``.  The TRT-LLM module tree mirrors the
# text-only Flash-FP8 layout (``model.layers.*``, ``model.norm`` etc.) so we
# normalize keys in place once at load time.
_LANGUAGE_MODEL_RENAMES = (
    ("model.language_model.", "model."),  # text decoder weights
    ("model.vision_model.", "vision_model."),  # ignored vision tower
    ("model.vit_large_projector", "vit_large_projector"),  # ignored projector
)


def rewrite_language_model_keys(weights) -> int:
    """Flatten the multimodal ``model.language_model.*`` namespace in place.

    Returns the number of keys rewritten. Idempotent and safe on text-only
    checkpoints that already use ``model.*`` keys.
    """
    if weights is None or not hasattr(weights, "keys"):
        return 0
    rewritten = 0
    for key in list(weights.keys()):
        for src_prefix, dst_prefix in _LANGUAGE_MODEL_RENAMES:
            if key.startswith(src_prefix):
                new_key = dst_prefix + key[len(src_prefix) :]
                if new_key == key:
                    break
                value = weights[key]
                del weights[key]
                weights[new_key] = value
                rewritten += 1
                break
    return rewritten


def strip_language_model_prefix_from_exclude_modules(exclude_modules):
    """Drop the ``language_model.`` segment from quant-config exclude patterns.

    ModelOpt's ``hf_quant_config.json`` for the multimodal Step3p7 NVFP4
    checkpoint expresses ``exclude_modules`` using the on-disk
    ``model.language_model.layers.<i>.*`` namespace.  TRT-LLM's
    ``is_module_excluded_from_quantization`` matches against the runtime
    module path (``model.layers.<i>.*``) so we drop the ``language_model.``
    segment to keep the patterns aligned.

    Returns a new list (or ``None`` if input was ``None``).
    """
    if exclude_modules is None:
        return None
    rewritten = []
    for entry in exclude_modules:
        # ``re:`` prefixed entries opt out of glob processing; leave them alone.
        if isinstance(entry, str) and not entry.startswith("re:"):
            entry = entry.replace("model.language_model.", "model.")
        rewritten.append(entry)
    return rewritten


def _model_config_keep_kv_quant(model_config: ModelConfig) -> ModelConfig:
    """Like ``_model_config_without_quant`` but preserves ``kv_cache_quant_algo``.

    Used by Step3p7's attention block so the production attention backend
    still picks up FP8 (or future NVFP4) KV cache when the checkpoint quant
    config sets ``kv_cache_quant_algo`` even though the QKV weights
    themselves remain BF16.
    """
    import copy as _copy

    from tensorrt_llm.models.modeling_utils import QuantConfig

    cloned = _copy.copy(model_config)
    src_kv = getattr(model_config.quant_config, "kv_cache_quant_algo", None)
    cloned.quant_config = QuantConfig(kv_cache_quant_algo=src_kv)
    object.__setattr__(cloned, "_frozen", False)
    try:
        cloned.quant_config_dict = None
    finally:
        object.__setattr__(cloned, "_frozen", True)
    return cloned


def rewrite_mtp_weights_for_step3p7(weights, text_config: PretrainedConfig) -> int:
    """Rewrite Step3p7 MTP checkpoint keys to the TRT-LLM module layout.

    Source keys store the transformer block directly under
    ``model.layers.<idx>`` and the draft head under
    ``transformer.shared_head``.  TRT-LLM follows the Step3p5-MTP structure
    used by vLLM: non-projection block weights live under ``mtp_block`` and
    the head lives under ``shared_head``.
    """
    if weights is None or not hasattr(weights, "keys"):
        return 0

    num_layers = int(text_config.num_hidden_layers)
    num_mtp_layers = int(getattr(text_config, "num_nextn_predict_layers", 0) or 0)
    if num_mtp_layers <= 0:
        return 0

    rewritten = 0
    keys_snapshot = list(weights.keys())
    for layer_idx in range(num_layers, num_layers + num_mtp_layers):
        prefix = f"model.layers.{layer_idx}."
        for key in keys_snapshot:
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :]
            if suffix.startswith("transformer.shared_head."):
                new_key = prefix + suffix.replace("transformer.", "", 1)
            elif suffix.startswith(_MTP_DIRECT_WEIGHT_PREFIXES):
                continue
            else:
                new_key = prefix + "mtp_block." + suffix
            if new_key == key:
                continue
            value = weights[key]
            del weights[key]
            weights[new_key] = value
            rewritten += 1
    return rewritten


@register_auto_model("Step3p7ForConditionalGeneration")
class Step3p7ForCausalLM(SpecDecOneEngineForCausalLM[Step3p7TextModel, PretrainedConfig]):
    """Top-level Step3p7 text-generation entry point.

    GSM8K is text-only.  The vision tower carried in the checkpoint is
    intentionally ignored.  MTP layers (45..47) are ignored by the plain text
    path and loaded when ``MTPDecodingConfig`` enables one-engine MTP.
    """

    # Step3p7Attention overrides the base ``Attention.forward`` to plumb the
    # source's per-head output gate between the attention backend and
    # ``o_proj``.  Helix CP, LoRA, and attention sinks from the base class are
    # intentionally not exercised by this bring-up; they can be re-enabled by
    # upstreaming a per-head gate hook into ``Attention.forward``.

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]

        normed_hidden_states = self.model.norm(hidden_states)

        if self.spec_worker is not None:
            logits = self.logits_processor.forward(
                normed_hidden_states[spec_metadata.gather_ids],
                self.lm_head,
                attn_metadata,
                True,
            )

            spec_input_ids = input_ids
            spec_position_ids = position_ids
            if attn_metadata.padded_num_tokens is not None:
                if input_ids is not None:
                    spec_input_ids = input_ids[: attn_metadata.num_tokens]
                if position_ids is not None:
                    spec_position_ids = _slice_spec_position_ids(
                        position_ids, attn_metadata.num_tokens
                    )

            return self.spec_worker(
                input_ids=spec_input_ids,
                position_ids=spec_position_ids,
                hidden_states=hidden_states,
                logits=logits,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=self.draft_model,
                resource_manager=resource_manager,
            )

        return self.logits_processor.forward(
            normed_hidden_states,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def __init__(self, model_config: ModelConfig):
        # Promote text-config aliases the rest of TRT-LLM expects.
        _mirror_step3p7_text_aliases(model_config.pretrained_config)
        # The multimodal NVFP4 checkpoint expresses ``exclude_modules`` against
        # the on-disk ``model.language_model.*`` namespace; normalize to the
        # TRT-LLM runtime namespace so module-name matching works correctly.
        qc = getattr(model_config, "quant_config", None)
        if qc is not None and qc.exclude_modules is not None:
            qc.exclude_modules = strip_language_model_prefix_from_exclude_modules(
                qc.exclude_modules
            )
        # ``model_loader.py`` reads ``pretrained_config.torch_dtype`` *before*
        # the model is constructed and stashes the value in
        # ``extra_attrs['allreduce_dtype']`` so the NCCL window prealloc path
        # can size buffers.  HF 5.x keeps ``torch_dtype`` as a string when the
        # config is loaded via ``trust_remote_code``, so when we run our own
        # normalisation here we must also refresh the extra_attrs entry that
        # was captured under the unnormalised string.
        if hasattr(model_config, "extra_attrs") and isinstance(model_config.extra_attrs, dict):
            cur = model_config.extra_attrs.get("allreduce_dtype")
            if isinstance(cur, str):
                try:
                    mapped = getattr(torch, cur)
                    if isinstance(mapped, torch.dtype):
                        model_config.extra_attrs["allreduce_dtype"] = mapped
                except AttributeError:
                    model_config.extra_attrs["allreduce_dtype"] = torch.bfloat16
        _prepare_step3p7_mtp_spec_config(model_config)
        text_config = _get_text_config(model_config)
        # Stash text_config so loaders/tests can reach it through the model.
        super().__init__(Step3p7TextModel(model_config), model_config)
        self.text_config = text_config
        self.num_hidden_layers = int(text_config.num_hidden_layers)
        active_mtp_layers = 0
        if (
            model_config.spec_config is not None
            and model_config.spec_config.spec_dec_mode.is_mtp_one_model()
        ):
            self.model.layers.extend(self.draft_model.mtp_layers)
            self.epilogue.extend(self.draft_model.mtp_layers)
            self.epilogue.append(self.spec_worker)
            active_mtp_layers = len(self.draft_model.mtp_layers)

        # Recognized ignored key prefixes (for the weight accounting test).
        mtp_prefixes = tuple(
            f"model.layers.{idx}."
            for idx in range(
                int(text_config.num_hidden_layers),
                int(text_config.num_hidden_layers)
                + int(getattr(text_config, "num_nextn_predict_layers", 0)),
            )
        )
        if active_mtp_layers > 0:
            inactive_mtp_prefixes = mtp_prefixes[active_mtp_layers:]
            self.ignored_key_prefixes = (
                "vision_model.",
                "vit_large_projector.",
                *inactive_mtp_prefixes,
            )
        else:
            self.ignored_key_prefixes = ("vision_model.", "vit_large_projector.", *mtp_prefixes)

    # ------------------------------------------------------------------
    # Weight loading helpers
    # ------------------------------------------------------------------

    def get_ignored_key_prefixes(self) -> Tuple[str, ...]:
        return self.ignored_key_prefixes

    def _capture_bf16_clamp_weights(self, weights) -> List[int]:
        """Pre-load 2D bf16 expert weights for Python-clamp layers.

        Walks each MoE layer that uses the Python clamp expert path
        (``_use_python_clamp=True``), probes the dtype of one expert
        weight in ``weights``, and if it is **not** FP8 then captures the
        local-EP-rank subset of original 2D bf16 expert weights from the
        HF weights dict and copies them into the pre-allocated
        ``_clamp_{gate,up,down}_proj`` buffers.  Sets
        ``_clamp_weights_loaded=True`` so ``load_clamp_weights_from_fp8_experts``
        is a no-op.

        Why this path exists: the bf16 reference checkpoint uses
        ``BF16TRTLLMGenFusedMoEMethod``, whose ``post_load_weights`` (run
        inside ``super().load_weights``) reshuffles ``w3_w1_weight`` into
        a 4D BlockMajorK layout for the TRTLLM-Gen-Bf16 cubins.  The
        Python clamp path needs 2D ``(I, H)`` per expert -- attempting to
        ``[expert_idx]`` and ``.t()`` the 4D tensor crashes with
        "t() expects a tensor with <= 2 dimensions, but self is 3D".  By
        capturing **before** ``super().load_weights`` is called, we see
        the original 2D HF tensors (after ``split_stacked_moe_weights``
        but before any backend layout transform).

        FP8 checkpoint: the FP8 backend keeps a 3D
        ``(num_local_experts, 2*I, H)`` layout that
        ``load_clamp_weights_from_fp8_experts`` can dequantize directly,
        so this method skips FP8 layers and the post-load dequant runs as
        usual.

        Returns the list of layer indices for which bf16 weights were
        captured (for diagnostic logging).
        """
        if weights is None or not hasattr(weights, "__getitem__"):
            return []
        captured: List[int] = []
        for layer in self.model.layers:
            moe = getattr(layer, "moe", None)
            if moe is None:
                continue
            if not getattr(moe, "_use_python_clamp", False):
                continue
            if getattr(moe, "_clamp_weights_loaded", False):
                continue
            probe_key = f"model.layers.{moe.layer_idx}.moe.experts.0.w1.weight"
            if probe_key not in weights:
                continue
            probe = weights[probe_key]
            # FP8 block-scale routed experts (dtype ``float8_e4m3fn``) are
            # handled post-load by ``load_clamp_weights_from_fp8_experts``
            # because the FP8 backend keeps a 3D layout we can slice.  The
            # BF16 reference and the NVFP4 (modelopt) checkpoint both need
            # the pre-load capture: NVFP4 weights are ``uint8`` (each byte
            # = 2 fp4 nibbles) and the NVFP4 backend's post-load layout
            # transform (block-scale interleave, padding) can't be reversed
            # for the Python clamp loop.
            if probe.dtype == torch.float8_e4m3fn:
                continue
            try:
                ep_size = max(1, moe.mapping.moe_ep_size)
                ep_rank = int(moe.mapping.moe_ep_rank)
                num_local = max(1, moe.num_experts // ep_size)
                local_start = ep_rank * num_local
                local_ids = list(range(local_start, local_start + num_local))
                base = f"model.layers.{moe.layer_idx}.moe.experts"
                if probe.dtype == torch.uint8:
                    # NVFP4: dequant the rank-local subset on the same device
                    # the clamp buffers live on (GPU when running on a node;
                    # ``meta`` during dry-init).  Per-element ``lut[idx]``
                    # gather is orders of magnitude faster on B200 than on
                    # CPU, so we always batch the per-expert ops into one
                    # tensor before moving to device.
                    target_dev = moe._clamp_gate_proj.device
                    if target_dev.type == "meta":
                        target_dev = (
                            torch.device("cuda")
                            if torch.cuda.is_available()
                            else torch.device("cpu")
                        )
                    gate_stack = _nvfp4_stack_dequant(
                        weights, base, "w1", local_ids, target_device=target_dev
                    )
                    up_stack = _nvfp4_stack_dequant(
                        weights, base, "w3", local_ids, target_device=target_dev
                    )
                    down_stack = _nvfp4_stack_dequant(
                        weights, base, "w2", local_ids, target_device=target_dev
                    )
                else:
                    gate_stack = torch.stack([weights[f"{base}.{e}.w1.weight"] for e in local_ids])
                    up_stack = torch.stack([weights[f"{base}.{e}.w3.weight"] for e in local_ids])
                    down_stack = torch.stack([weights[f"{base}.{e}.w2.weight"] for e in local_ids])
            except KeyError:
                continue
            target_dev = moe._clamp_gate_proj.device
            target_dtype = moe._clamp_gate_proj.dtype
            # Re-register buffers if the on-disk per-expert shape disagrees
            # with the __init__ guess (e.g. unusual EP rounding).  This
            # mirrors the resize logic in ``load_clamp_weights_from_fp8_experts``.
            target_shape = gate_stack.shape
            if moe._clamp_gate_proj.shape != target_shape:
                moe.register_buffer(
                    "_clamp_gate_proj",
                    torch.empty(target_shape, dtype=target_dtype, device=target_dev),
                    persistent=False,
                )
                moe.register_buffer(
                    "_clamp_up_proj",
                    torch.empty(up_stack.shape, dtype=target_dtype, device=target_dev),
                    persistent=False,
                )
                moe.register_buffer(
                    "_clamp_down_proj",
                    torch.empty(down_stack.shape, dtype=target_dtype, device=target_dev),
                    persistent=False,
                )
                moe._clamp_num_local_experts = int(target_shape[0])
            moe._clamp_gate_proj.copy_(gate_stack.to(device=target_dev, dtype=target_dtype))
            moe._clamp_up_proj.copy_(up_stack.to(device=target_dev, dtype=target_dtype))
            moe._clamp_down_proj.copy_(down_stack.to(device=target_dev, dtype=target_dtype))
            moe._clamp_weights_loaded = True
            captured.append(moe.layer_idx)
        return captured

    def load_weights(
        self,
        weights,
        weight_mapper=None,
        skip_modules=None,
        params_map=None,
        allow_partial_loading: bool = False,
    ):
        """Step3p7 weight loader.

        Performs two transformations before delegating to the generic loader:

        1. **Drop ignored prefixes.** The on-disk checkpoint includes vision
           tower weights, and the plain text path also ignores MTP layer
           weights (layers 45..47).  We remove inactive keys so
           ``mark_consumed`` accounting stays focused on the active path.
        2. **Rewrite active MTP weights.** Step3p7 stores the draft block
           directly under ``model.layers.<idx>`` and the draft head under
           ``transformer.shared_head``; TRT-LLM modules use ``mtp_block`` and
           ``shared_head``.
        3. **Split stacked routed-expert tensors.** Source stores
           ``...moe.{gate,up,down}_proj.{weight,weight_scale_inv}`` as a
           single tensor with leading dim ``num_experts``.  The generic
           ``VANILLA`` MoE backend expects per-expert keys
           ``...moe.experts.<e>.w{1,3,2}.{weight,weight_scale_inv}``.  We
           materialize per-expert views before delegating so the FP8
           block-scale kernel sees both the weights and their scales.
        """
        from tensorrt_llm.logger import logger as _logger

        skip_modules = list(skip_modules) if skip_modules else []

        # Flatten any multimodal ``model.language_model.*`` namespace before
        # the ignored-prefix sweep so the existing ignore list (``vision_model.``,
        # ``vit_large_projector.``) keeps matching.
        rewrite_language_model_keys(weights)

        if hasattr(weights, "keys"):
            keys_snapshot = list(weights.keys())
            for k in keys_snapshot:
                if any(k.startswith(p) for p in self.ignored_key_prefixes):
                    try:
                        del weights[k]
                    except KeyError:
                        pass
                    continue
                # FP8 KV cache calibration scales (``...k_proj.k_scale`` /
                # ``...v_proj.v_scale``) live on the QKV linears in the
                # checkpoint but our attention runs as BF16 (its kv_scales
                # parameter is only allocated for NVFP4 KV cache today), so
                # nothing in the loader consumes them.  Drop them up front so
                # the generic loader's ``mark_consumed`` accounting stays
                # clean.  KV cache stays FP8 with the default 1.0 scale; a
                # future change can plumb these into the trtllm attention
                # backend's ``kv_cache_scaling_factor``.
                if k.endswith(".k_scale") or k.endswith(".v_scale"):
                    try:
                        del weights[k]
                    except KeyError:
                        pass

        rewrite_mtp_weights_for_step3p7(weights, self.text_config)
        split_stacked_moe_weights(weights, self.text_config)

        # Capture original 2D bf16 expert weights for Python-clamp layers
        # before the backend transforms them into a layout that the Python
        # expert path cannot read directly (see TRTLLMGen-Bf16 BlockMajorK
        # 4D layout in `BF16TRTLLMGenFusedMoEMethod.process_weights_after_loading`).
        # No-op for the FP8 checkpoint -- the FP8 backend keeps a 3D layout
        # that the post-load `load_clamp_weights_from_fp8_experts` path can
        # dequantize directly.
        self._capture_bf16_clamp_weights(weights)

        rc = DecoderModelForCausalLM.load_weights(
            self,
            weights,
            weight_mapper=weight_mapper,
            skip_modules=[*skip_modules, "draft_model"],
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )
        # Populate bf16 expert weights for the Python expert path:
        #  * Layers with ``swiglu_limits`` set (e.g. 43/44) where the source
        #    clamp must be expressed and no FP8-block-scale backend on B200
        #    supports ``swiglu_limit``.
        #  * Layers ``layer_idx >= STEP3P7_PYTHON_FALLBACK_THRESHOLD`` when the
        #    optional fallback knob is enabled.
        # Must run AFTER the FP8 backend has materialised its w3_w1/w2 tensors.
        for layer in self.model.layers:
            moe = getattr(layer, "moe", None)
            if moe is None:
                continue
            if not getattr(moe, "_use_python_clamp", False):
                continue
            reason = getattr(moe, "_python_path_reason", "")
            try:
                moe.load_clamp_weights_from_fp8_experts()
            except Exception as e:
                _logger.warning(
                    "[Step3p7] failed to populate bf16 expert weights "
                    "for layer %d (%s): %s.  Forward will fall back to "
                    "the FP8 backend without the Python path.",
                    moe.layer_idx,
                    reason,
                    str(e)[:256],
                )
        return rc


def scan_fp8_scale_inv_tensors(checkpoint_dir: str):
    """Enumerate FP8 ``weight_scale_inv`` tensors missing from the safetensors index.

    The Step3p7 Flash FP8 checkpoint stores routed-expert ``weight_scale_inv``
    tensors **inside** the safetensors shards but does **not** list them in
    ``model.safetensors.index.json``.  The standard HF weight loader iterates
    only over the index, so the FP8 scales never reach the model and the
    TRTLLMGen block-scale batched-GEMM aborts at warm-up with
    ``CUDA_ERROR_INVALID_HANDLE``.

    This helper scans every ``model-*.safetensors`` shard and returns a
    mapping ``{key: (filename, dtype, shape)}`` for every scale_inv key it
    finds that is **not** already in the index.  The next iteration's weight
    loader can then materialize those tensors and inject them into the
    weights stream alongside the index-listed weights.

    Returns
    -------
    dict[str, dict]
        ``{full_key: {"shard": str, "dtype": str, "shape": tuple}}`` for each
        scale tensor missing from the index.
    """
    import glob
    import json as _json
    from pathlib import Path as _Path

    import safetensors

    ckpt = _Path(checkpoint_dir)
    idx_path = ckpt / "model.safetensors.index.json"
    indexed = set()
    if idx_path.exists():
        with open(idx_path) as f:
            indexed = set(_json.load(f).get("weight_map", {}).keys())

    missing = {}
    for shard in sorted(glob.glob(str(ckpt / "model-*.safetensors"))):
        with safetensors.safe_open(shard, framework="pt") as h:
            for key in h.keys():
                if key in indexed:
                    continue
                if "scale_inv" in key or "weight_scale" in key:
                    t = h.get_tensor(key)
                    missing[key] = {
                        "shard": _Path(shard).name,
                        "dtype": str(t.dtype),
                        "shape": tuple(t.shape),
                    }
    return missing
