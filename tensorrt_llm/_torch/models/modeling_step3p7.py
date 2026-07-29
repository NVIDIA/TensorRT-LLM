# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""TensorRT-LLM PyTorch backend for the Step3p7 Flash text decoder.

This module owns the text-only causal LM plus its MTP draft layers; the
sibling file ``modeling_step3p7vl`` (PerceptionEncoder vision tower +
multimodal input processor + ForConditionalGeneration wrapper) builds on
top of it for the VLM checkpoint variants.

Step3p7 has three notable per-layer behaviors not shared with other models:

- Per-layer RoPE: full-attention layers (0, 4, 8, ..., 44) use partial rotary
  with llama3 scaling and theta 5e6; sliding-attention layers use full rotary
  with theta 1e4 and no scaling.
- Gemma-style RMSNorm (``weight + 1``) on Q/K and all layer norms, and a
  head-wise output gate (``sigmoid(g_proj(hidden))`` multiplied per attention
  head before ``o_proj``).
- Layers 43/44 carry non-zero SwiGLU clamp limits for routed and shared
  experts; routed experts use sigmoid+bias top-k routing with renormalization
  and ``routed_scaling_factor=3.0``.

Supported checkpoint variants: BF16 reference, FP8 block-scale flash, and
NVFP4 multimodal (with text decoder under ``model.language_model.*``,
flattened at load time). The on-disk checkpoints also carry 3 MTP layers
(loaded when ``MTPDecodingConfig`` is enabled) and a vision tower (ignored
here; handled by the VLM wrapper).
"""

from __future__ import annotations

import copy
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

_DEFAULT_FULL_ATTENTION_PERIOD = 4
_DEFAULT_ROPE_THETA = 10000.0
_FP8_BLOCK_SIZE = 128
_PYTHON_FALLBACK_DISABLED_LAYER = 999
_PYTHON_FALLBACK_THRESHOLD_ENV = "STEP3P7_PYTHON_FALLBACK_THRESHOLD"
_KV_SCALE_SUFFIXES = (".k_scale", ".v_scale")


def _fp8_block_dequant_3d(
    weight_fp8: torch.Tensor, scale_inv: torch.Tensor, block: int = _FP8_BLOCK_SIZE
) -> torch.Tensor:
    """Dequantise ``(E, M, K)`` FP8 e4m3 block-scale tensor to bf16."""
    _, M, K = weight_fp8.shape
    scale = scale_inv.to(torch.float32).repeat_interleave(block, dim=-2)
    scale = scale[..., :M, :].repeat_interleave(block, dim=-1)[..., :K]
    return (weight_fp8.to(torch.float32) * scale).to(torch.bfloat16)


# e2m1 nibble lookup: the 16 representable values for the NVFP4 4-bit float
# (1 sign + 2 exponent + 1 mantissa).
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _nvfp4_dequant_batched(
    weight_uint8: torch.Tensor,
    block_scale_fp8: torch.Tensor,
    global_scale_fp32: torch.Tensor,
    target_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Dequantise an N-D NVFP4 weight tensor to bf16 in one batched op.

    ``weight_uint8`` is ``(*B, M, K/2)`` packed (two e2m1 nibbles per byte),
    ``block_scale_fp8`` is ``(*B, M, K/16)`` (fp8_e4m3), and
    ``global_scale_fp32`` is scalar or ``(B,)``. For checkpoint tensors that
    live on CPU, pass ``target_device='cuda'`` -- the per-element ``lut[idx]``
    gather costs ~50ms per expert on CPU but <1ms on B200.
    """
    device = target_device or weight_uint8.device
    w = weight_uint8.to(device=device, non_blocking=True)
    s1 = block_scale_fp8.to(device=device, non_blocking=True)
    s2 = global_scale_fp32.to(device=device, non_blocking=True)

    K = w.shape[-1] * 2
    lut = _E2M1_VALUES.to(device=device)

    high = (w >> 4) & 0x0F
    low = w & 0x0F
    vals = torch.empty(*w.shape[:-1], K, dtype=torch.float32, device=device)
    vals[..., 0::2] = lut[low.long()]
    vals[..., 1::2] = lut[high.long()]

    s2_b = s2.to(torch.float32)
    for _ in range(s1.dim() - s2_b.dim()):
        s2_b = s2_b.unsqueeze(-1)
    scale = (s1.to(torch.float32) * s2_b).unsqueeze(-1)

    vals = vals.view(*w.shape[:-1], K // 16, 16) * scale
    return vals.view(*w.shape[:-1], K).to(torch.bfloat16)


def _nvfp4_stack_dequant(
    weights,
    base: str,
    w_name: str,
    expert_ids: list,
    target_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Stack and dequant NVFP4 experts ``expert_ids`` for ``{base}.<e>.<w_name>``."""
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
    if dt is None:
        cfg.torch_dtype = torch.bfloat16
        return
    if isinstance(dt, str):
        mapped = getattr(torch, dt, None)
        if isinstance(mapped, torch.dtype):
            cfg.torch_dtype = mapped
            return
        try:
            from transformers.utils import STR_TO_DTYPE  # type: ignore

            cfg.torch_dtype = STR_TO_DTYPE.get(dt, torch.bfloat16)
        except ImportError:
            cfg.torch_dtype = torch.bfloat16


def _mirror_step3p7_text_aliases(pretrained_config: PretrainedConfig) -> None:
    """Promote Step3p7 ``text_config`` aliases the runtime expects on the top level.

    Complements ``ModelConfig.from_pretrained``'s generic ``_mirror_text_subconfig_attrs``
    by plugging in Step3p7-specific aliases (``num_key_value_heads`` from
    ``num_attention_groups``, ``num_nextn_predict_layers``, etc.) and normalising
    ``torch_dtype`` from the raw HF JSON string to ``torch.dtype``.
    """
    text_config = getattr(pretrained_config, "text_config", None)
    if text_config is None:
        return

    _normalize_torch_dtype(pretrained_config)
    _normalize_torch_dtype(text_config)

    if not hasattr(pretrained_config, "num_key_value_heads") and hasattr(
        text_config, "num_attention_groups"
    ):
        pretrained_config.num_key_value_heads = text_config.num_attention_groups
    if not hasattr(text_config, "num_key_value_heads") and hasattr(
        text_config, "num_attention_groups"
    ):
        text_config.num_key_value_heads = text_config.num_attention_groups

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
    if layer_idx % _DEFAULT_FULL_ATTENTION_PERIOD == 0:
        return "full_attention"
    return "sliding_attention"


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


def _per_layer_lookup(text_config: PretrainedConfig, values, layer_idx: int) -> float:
    """Pick ``values[layer_idx]`` from a per-layer list, falling back to the most
    recent entry that matches the current layer's attention type."""
    if layer_idx < len(values):
        return float(values[layer_idx])
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        cur_type = _layer_attention_type(text_config, layer_idx)
        for prev_idx in range(min(len(values), len(layer_types)) - 1, -1, -1):
            if layer_types[prev_idx] == cur_type:
                return float(values[prev_idx])
    return float(values[-1])


def _layer_rope_theta(text_config: PretrainedConfig, layer_idx: int) -> float:
    theta = getattr(text_config, "rope_theta", _DEFAULT_ROPE_THETA)
    if isinstance(theta, (list, tuple)):
        return _per_layer_lookup(text_config, theta, layer_idx)
    return float(theta)


def _layer_partial_rotary(text_config: PretrainedConfig, layer_idx: int) -> float:
    factors = getattr(text_config, "partial_rotary_factors", None)
    if factors is None:
        return 1.0
    return _per_layer_lookup(text_config, factors, layer_idx)


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


def _parse_python_fallback_threshold() -> int:
    try:
        return int(
            os.environ.get(_PYTHON_FALLBACK_THRESHOLD_ENV, str(_PYTHON_FALLBACK_DISABLED_LAYER))
        )
    except ValueError:
        return _PYTHON_FALLBACK_DISABLED_LAYER


def _select_python_expert_path(
    model_config: ModelConfig,
    text_config: PretrainedConfig,
    layer_idx: int,
) -> tuple[float | None, bool, str]:
    """Return ``(swiglu_limit, use_python_path, reason)`` for a routed MoE layer.

    The Python expert loop is selected for clamp-active layers (no FP8 backend
    on B200 supports ``swiglu_limit``), for BF16 checkpoints (the FP8 kernel
    can't run against bf16 weights), and via an env-var diagnostic override.
    """
    swiglu_limit = _layer_swiglu_limit(text_config, layer_idx, shared=False)
    qc = getattr(model_config, "quant_config", None)
    is_bf16_checkpoint = qc is None or getattr(qc, "quant_algo", None) is None

    if swiglu_limit is not None and swiglu_limit > 0:
        return swiglu_limit, True, "clamp"
    if layer_idx >= _parse_python_fallback_threshold():
        return swiglu_limit, True, "late_layer"
    if is_bf16_checkpoint:
        return swiglu_limit, True, "bf16_kernel_workaround"
    return swiglu_limit, False, ""


# ---------------------------------------------------------------------------
# MoE routing
# ---------------------------------------------------------------------------


class Step3p7RouterBiasHolder(nn.Module):
    """Standalone parameter holder for ``router_bias`` (mirrors MiniMaxM2).

    The holder is attached as ``moe.router_bias`` so the resulting parameter
    path ``moe.router_bias.router_bias`` matches the HF source key
    ``moe.router_bias`` exactly when the generic loader filters weights.
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

    Inherits from ``MiniMaxM2MoeRoutingMethod`` so the TRTLLMGen
    ``_extract_routing_params`` helper recognises us via ``isinstance`` and
    feeds the bias pointer to the kernel. The MiniMax2 C++ routing path
    hard-codes ``routeScale = 1.0f`` (see ``runner.cu``), so
    ``routed_scaling_factor`` is applied to the MoE output in
    ``Step3p7MoE.forward`` instead of inside the kernel.
    """

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        callable_router_bias,
        routed_scaling_factor: float = 1.0,
        output_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=callable_router_bias,
            output_dtype=output_dtype,
        )
        self.routed_scaling_factor = float(routed_scaling_factor)

    def apply(
        self,
        router_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
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
        return self.callable_e_score_correction_bias()


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


_LLAMA3_ROPE_PARAM_KEYS = frozenset(
    {
        "rope_type",
        "factor",
        "original_max_position_embeddings",
        "low_freq_factor",
        "high_freq_factor",
    }
)


def _build_per_layer_config(text_config: PretrainedConfig, layer_idx: int) -> PretrainedConfig:
    """Shallow-copy ``text_config`` with per-layer RoPE / head scalars applied.

    Step3p7 stores ``num_attention_heads``, ``rope_theta`` etc. as per-layer
    lists, while TRT-LLM's ``RopeParams.from_config`` and ``Attention.__init__``
    read them as top-level scalars. Transformers 5.x also keeps the per-layer
    list inside the ``rope_parameters`` dict; ``RopeParams.from_config`` does
    ``config.update(rope_parameters)``, so we must materialise a per-layer
    ``rope_parameters`` too — otherwise the scalar gets overwritten by the list.
    """
    cfg = copy.copy(text_config)
    cfg.num_attention_heads = _layer_query_heads(text_config, layer_idx)
    cfg.num_key_value_heads = _layer_kv_heads(text_config, layer_idx)
    theta_scalar = _layer_rope_theta(text_config, layer_idx)
    cfg.rope_theta = theta_scalar
    cfg.partial_rotary_factor = _layer_partial_rotary(text_config, layer_idx)

    src_rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(src_rope_params, dict):
        cfg.rope_parameters = {**src_rope_params, "rope_theta": theta_scalar}

    if not _layer_uses_rope_scaling(text_config, layer_idx):
        cfg.rope_scaling = None
        if isinstance(getattr(cfg, "rope_parameters", None), dict):
            cfg.rope_parameters = {
                k: v for k, v in cfg.rope_parameters.items() if k not in _LLAMA3_ROPE_PARAM_KEYS
            }
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
        per_layer_cfg = _build_per_layer_config(text_config, layer_idx)
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

        rope_params = RopeParams.from_config(per_layer_cfg)
        rope_params.theta = per_layer_cfg.rope_theta
        if _layer_uses_rope_scaling(text_config, layer_idx):
            rope_params.scale_type = RotaryScalingType.llama3
        else:
            rope_params.scale_type = RotaryScalingType.none
            rope_params.scale = 1.0
        rope_params.dim = int(self.head_dim * _layer_partial_rotary(text_config, layer_idx))

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )

        # Swap in the per-layer pretrained_config for the duration of the base
        # constructor so it sees scalar (not per-layer-list) head/RoPE fields.
        # ``ModelConfig`` is frozen but ``pretrained_config`` is exempt.
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
        # Split before QK-norm: Gemma RMSNorm needs the heads-separated layout.
        q, k, v = self.split_qkv(q, k, v)
        q, k = self.apply_qk_norm(q, k)
        return super().apply_rope(q, k, v, position_ids)

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

        Bypasses the base ``Attention.forward`` to apply the per-head output
        gate (``sigmoid(g_proj(hidden))``) between the attention backend and
        ``o_proj``. Helix CP, LoRA, and attention sinks are not plumbed here.
        """
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
        q, k, v = self.apply_rope(qkv, None, None, position_ids)
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
            # self.num_heads is already per-rank after TP sharding.
            gate = self.g_proj(hidden_states)
            orig_shape = attn_output.shape
            attn_output = attn_output.view(*orig_shape[:-1], self.num_heads, self.head_dim)
            attn_output = attn_output * gate.unsqueeze(-1).sigmoid()
            attn_output = attn_output.view(*orig_shape)

        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# MLP / MoE / Decoder layer
# ---------------------------------------------------------------------------


class ClampedGatedMLP(GatedMLP):
    """Dense SwiGLU MLP with optional per-layer clamp on gate/up activations.

    When ``swiglu_limit`` is set, ``gate`` is clamped to ``[-inf, limit]`` and
    ``up`` is clamped to ``[-limit, limit]`` before the elementwise multiply
    (matching source ``Step3p7MLP``). Otherwise this is a standard ``GatedMLP``.
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
            # Shared expert defers all-reduce to the routed MoE call.
            reduce_output=not is_shared_expert,
            overridden_tp_size=None,
            is_shared_expert=is_shared_expert,
        )
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.swiglu_limit is None:
            return super().forward(hidden_states, **kwargs)
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        gate = torch.nn.functional.silu(gate).clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(gate * up)


class Step3p7MoE(nn.Module):
    """Routed MoE block: router gate + bias-holder + routed experts.

    The shared expert is owned by the decoder layer (HF source stores it
    under ``share_expert.*``, not ``moe.share_expert.*``).
    """

    _CLAMP_BUFFER_NAMES = ("_clamp_gate_proj", "_clamp_up_proj", "_clamp_down_proj")

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
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

        gate_dtype = torch.float32 if self.need_fp32_gate else text_config.torch_dtype
        self.gate = Linear(
            in_features=self.hidden_size,
            out_features=self.num_experts,
            bias=False,
            dtype=gate_dtype,
            quant_config=None,
        )

        self.router_bias = Step3p7RouterBiasHolder(self.num_experts)
        routing_method = Step3p7MoeRoutingMethod(
            top_k=self.top_k,
            num_experts=self.num_experts,
            callable_router_bias=lambda: self.router_bias.router_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        (
            self._routed_swiglu_limit,
            self._use_python_clamp,
            self._python_path_reason,
        ) = _select_python_expert_path(model_config, text_config, layer_idx)

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

        if self._use_python_clamp:
            self._allocate_clamp_buffers(text_config.torch_dtype)

        self.allreduce = None
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=self.mapping, strategy=model_config.allreduce_strategy
            )

    def _allocate_clamp_buffers(
        self,
        dtype: torch.dtype,
        shapes: Optional[Tuple[tuple, tuple, tuple]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Allocate non-persistent expert buffers for the Python clamp path."""
        if shapes is None:
            num_local = max(1, self.num_experts // max(1, self.mapping.moe_ep_size))
            gw_shape = (num_local, self.moe_intermediate_size, self.hidden_size)
            up_shape = (num_local, self.moe_intermediate_size, self.hidden_size)
            dn_shape = (num_local, self.hidden_size, self.moe_intermediate_size)
        else:
            gw_shape, up_shape, dn_shape = shapes
        for name, shape in zip(self._CLAMP_BUFFER_NAMES, (gw_shape, up_shape, dn_shape)):
            self.register_buffer(
                name, torch.empty(shape, dtype=dtype, device=device), persistent=False
            )
        self._clamp_num_local_experts = int(gw_shape[0])
        self._clamp_weights_loaded = False

    def _copy_clamp_weights(self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> None:
        if self._clamp_gate_proj.shape != gate.shape:
            self._allocate_clamp_buffers(
                self._clamp_gate_proj.dtype,
                shapes=(tuple(gate.shape), tuple(up.shape), tuple(down.shape)),
                device=self._clamp_gate_proj.device,
            )
        dev, dt = self._clamp_gate_proj.device, self._clamp_gate_proj.dtype
        self._clamp_gate_proj.copy_(gate.to(device=dev, dtype=dt))
        self._clamp_up_proj.copy_(up.to(device=dev, dtype=dt))
        self._clamp_down_proj.copy_(down.to(device=dev, dtype=dt))
        self._clamp_weights_loaded = True

    def _clamp_weight_device(self) -> torch.device:
        dev = self._clamp_gate_proj.device
        if dev.type == "meta":
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return dev

    def _local_expert_ids(self) -> list[int]:
        ep_size = max(1, self.mapping.moe_ep_size)
        num_local = max(1, self.num_experts // ep_size)
        local_start = int(self.mapping.moe_ep_rank) * num_local
        return list(range(local_start, local_start + num_local))

    def load_clamp_weights_from_fp8_experts(self) -> None:
        """Populate clamp buffers by dequantising the backend's expert weights.

        Called once by the model loader after ``super().load_weights`` has
        materialised the backend tensors. For the bf16 reference checkpoint
        the TRTLLMGen-Bf16 backend reshuffles ``w3_w1_weight`` into a 4D
        BlockMajorK layout that can't be sliced back; in that case
        ``Step3p7ForCausalLM._capture_bf16_clamp_weights`` runs earlier (from
        the raw HF tensors) and this method short-circuits.
        """
        if not self._use_python_clamp or self._clamp_weights_loaded:
            return
        # ``self.experts`` is a ConfigurableMoE wrapper; weights live on the backend.
        e = getattr(self.experts, "backend", None) or self.experts
        if not hasattr(e, "w3_w1_weight") or not hasattr(e, "w2_weight"):
            return
        w3_w1 = e.w3_w1_weight.data
        w2 = e.w2_weight.data
        moe_inter = self.moe_intermediate_size
        # w3_w1 layout is [w3 (up), w1 (gate)] along dim 1, same for FP8 and bf16 backends.
        w3 = w3_w1[:, :moe_inter, :]
        w1 = w3_w1[:, moe_inter:, :]
        scale_w3_w1 = getattr(e, "w3_w1_weight_scaling_factor", None)
        scale_w2 = getattr(e, "w2_weight_scaling_factor", None)
        is_fp8 = (
            scale_w3_w1 is not None and scale_w2 is not None and w3_w1.dtype == torch.float8_e4m3fn
        )
        if is_fp8:
            intermediate_blocks = _ceil_div(moe_inter, _FP8_BLOCK_SIZE)
            w3_scale = scale_w3_w1.data[:, :intermediate_blocks, :]
            w1_scale = scale_w3_w1.data[:, intermediate_blocks:, :]
            up_bf16 = _fp8_block_dequant_3d(w3, w3_scale)
            gate_bf16 = _fp8_block_dequant_3d(w1, w1_scale)
            down_bf16 = _fp8_block_dequant_3d(w2, scale_w2.data)
        else:
            up_bf16 = w3.to(torch.bfloat16).contiguous()
            gate_bf16 = w1.to(torch.bfloat16).contiguous()
            down_bf16 = w2.to(torch.bfloat16).contiguous()
        self._copy_clamp_weights(gate_bf16, up_bf16, down_bf16)

    def _python_clamped_moe_forward(
        self, h: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Python expert path with explicit SwiGLU clamp.

        Mirrors HF source ``Step3p7MoEMLP.get_expert_output`` and produces a
        per-rank partial sum (the decoder layer's all-reduce combines partials
        across EP ranks).

        CUDA-graph capture safe: the per-expert loop has a fixed Python count,
        and routing is materialised as a dense ``(N, num_experts)`` matrix via
        ``scatter_add_`` so each iteration multiplies by a fixed-shape column.
        Wasted compute is ``num_local / top_k`` over the masked version,
        negligible at Step3p7 layer dimensions.
        """
        topk_idx, topk_weights = self.experts.routing_method.apply(router_logits)
        N = topk_idx.shape[0]
        num_local = self._clamp_num_local_experts
        local_start = self.mapping.moe_ep_rank * num_local
        limit = float(self._routed_swiglu_limit or 0.0)

        weight_per_expert = torch.zeros((N, self.num_experts), dtype=torch.float32, device=h.device)
        weight_per_expert.scatter_add_(1, topk_idx.long(), topk_weights.to(torch.float32))

        # HF source computes experts in fp32 and casts each result back to
        # bf16 before accumulation -- keep that rounding point.
        x_f32 = h.to(torch.float32)
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

        gate_input = h.to(torch.float32) if self.need_fp32_gate else h
        router_logits = self.gate(gate_input)

        if self._use_python_clamp and self._clamp_weights_loaded:
            # Python path's routing.apply already scales topk_weights.
            return self._python_clamped_moe_forward(h, router_logits).view(orig_shape)

        routed = self.experts(
            h,
            router_logits,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            use_dp_padding=False,
        )
        # TRTLLMGen MiniMax2 kernel hard-codes routeScale=1.0, so apply
        # ``routed_scaling_factor`` to the MoE output here (mathematically
        # equivalent to scaling each topk weight).
        if self.routed_scaling_factor != 1.0:
            routed = routed * self.routed_scaling_factor
        return routed.view(orig_shape)


def _copy_model_config_with_quant(model_config: ModelConfig, quant_config: object) -> ModelConfig:
    """Shallow-copy ``model_config`` with a replacement ``quant_config``.

    ``ModelConfig`` is frozen but ``quant_config`` is exempt; we also need to
    clear ``quant_config_dict``, which requires bypassing the freeze.
    """
    cloned = copy.copy(model_config)
    cloned.quant_config = quant_config
    object.__setattr__(cloned, "_frozen", False)
    try:
        cloned.quant_config_dict = None
    finally:
        object.__setattr__(cloned, "_frozen", True)
    return cloned


def _model_config_without_quant(model_config: ModelConfig) -> ModelConfig:
    """Shallow copy with a no-op quant config (for bf16 paths like attention)."""
    from tensorrt_llm.models.modeling_utils import QuantConfig

    return _copy_model_config_with_quant(model_config, QuantConfig())


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

        # Attention/dense MLP/shared expert stay bf16 (only routed experts
        # are FP8/NVFP4). Attention still needs ``kv_cache_quant_algo`` so
        # FP8 KV cache (set by modelopt) reaches the attention backend.
        bf16_model_config = _model_config_without_quant(model_config)
        attn_model_config = _model_config_keep_kv_quant(model_config)
        self.self_attn = Step3p7Attention(attn_model_config, layer_idx)

        self.is_moe_layer = _is_moe_layer(text_config, layer_idx)
        if self.is_moe_layer:
            self.moe = Step3p7MoE(
                model_config, layer_idx=layer_idx, aux_stream_dict=aux_stream_dict
            )
            self.share_expert = ClampedGatedMLP(
                bf16_model_config,
                layer_idx=layer_idx,
                intermediate_size=int(
                    getattr(text_config, "share_expert_dim", text_config.moe_intermediate_size)
                ),
                swiglu_limit=_layer_swiglu_limit(text_config, layer_idx, shared=True),
                is_shared_expert=True,
            )
            self.mlp = None
        else:
            self.mlp = ClampedGatedMLP(
                bf16_model_config,
                layer_idx=layer_idx,
                intermediate_size=int(text_config.intermediate_size),
                swiglu_limit=_layer_swiglu_limit(text_config, layer_idx, shared=False),
                is_shared_expert=False,
            )
            self.moe = None
            self.share_expert = None

        # All Step3p7 RMSNorms (Q/K norms and every layer norm) use Gemma-style
        # ``(weight + 1)`` scaling, not the standard ``weight`` form.
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

# Per-expert suffixes that may accompany the stacked weights — covers FP8
# block-scale (``weight_scale_inv``) and NVFP4 (``weight_scale``,
# ``weight_scale_2``, ``input_scale``) checkpoints.
_STACKED_MOE_WEIGHT_SUFFIXES = (
    "weight",
    "weight_scale_inv",
    "weight_scale",
    "weight_scale_2",
    "input_scale",
)


def split_stacked_moe_weights(weights, text_config: PretrainedConfig) -> int:
    """Expand stacked routed-expert tensors into per-expert keys, in place.

    HF source stores each projection as a single ``(num_experts, ...)`` tensor;
    the VANILLA MoE loader expects per-expert keys
    ``experts.<e>.w{1,2,3}.<suffix>``. Slices along dim 0 (zero-copy view) and
    deletes the stacked keys. Returns the number of MoE layers split.
    """
    num_layers = int(text_config.num_hidden_layers)
    num_experts = int(text_config.moe_num_experts)

    layers_split = 0
    for layer_idx in range(num_layers):
        if not _is_moe_layer(text_config, layer_idx):
            continue
        prefix = f"model.layers.{layer_idx}.moe."
        proj_present = [p for p in _STACKED_MOE_PROJ_TO_W if f"{prefix}{p}.weight" in weights]
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
                if stacked.shape[0] != num_experts:
                    raise RuntimeError(
                        f"Step3p7 stacked MoE tensor {stacked_key} has "
                        f"leading dim {stacked.shape[0]} but num_experts={num_experts}."
                    )
                for expert_id in range(num_experts):
                    weights[f"{prefix}experts.{expert_id}.{dst_w}.{suffix}"] = stacked[expert_id]
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

    # max_draft_len is None when the user didn't set it (use the model default);
    # otherwise cap it at the model's MTP layer count.
    if spec_config.max_draft_len is None or spec_config.max_draft_len >= model_layers:
        spec_config.max_draft_len = model_layers
    spec_config.max_total_draft_tokens = spec_config.max_draft_len


_MTP_DIRECT_WEIGHT_PREFIXES = (
    "enorm.",
    "hnorm.",
    "eh_proj.",
    "shared_head.",
)


# Multimodal Step3p7 checkpoints (e.g. NVFP4 NIM/VNIM) nest the text decoder
# under ``model.language_model.*`` instead of ``model.*``; flatten at load time.
_LANGUAGE_MODEL_RENAMES = (
    ("model.language_model.", "model."),
    ("model.vision_model.", "vision_model."),
    ("model.vit_large_projector", "vit_large_projector"),
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
    """Drop ``language_model.`` from quant-config exclude patterns.

    ModelOpt's ``hf_quant_config.json`` for the multimodal NVFP4 checkpoint
    expresses ``exclude_modules`` against the on-disk namespace; TRT-LLM matches
    against the runtime module path, so we strip the prefix. ``re:`` entries are
    passed through untouched (they opt out of glob processing).
    """
    if exclude_modules is None:
        return None
    return [
        e.replace("model.language_model.", "model.")
        if isinstance(e, str) and not e.startswith("re:")
        else e
        for e in exclude_modules
    ]


def _model_config_keep_kv_quant(model_config: ModelConfig) -> ModelConfig:
    """Strip weight quant but preserve ``kv_cache_quant_algo`` for attention."""
    from tensorrt_llm.models.modeling_utils import QuantConfig

    src_kv = getattr(getattr(model_config, "quant_config", None), "kv_cache_quant_algo", None)
    return _copy_model_config_with_quant(model_config, QuantConfig(kv_cache_quant_algo=src_kv))


def rewrite_mtp_weights_for_step3p7(weights, text_config: PretrainedConfig) -> int:
    """Rewrite Step3p7 MTP checkpoint keys to the TRT-LLM module layout.

    Source stores the transformer block directly under ``model.layers.<idx>``
    and the head under ``transformer.shared_head``; TRT-LLM uses ``mtp_block``
    and ``shared_head`` (matching the Step3p5-MTP layout used by vLLM).
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


@register_auto_model("Step3p5ForCausalLM")
class Step3p7ForCausalLM(SpecDecOneEngineForCausalLM[Step3p7TextModel, PretrainedConfig]):
    """Step3p7 text-only causal LM core.

    Registered under ``Step3p5ForCausalLM`` (the text-config architecture);
    the VLM entry point ``Step3p7ForConditionalGeneration`` wraps this class
    in ``Step3p7VLForConditionalGeneration`` (see ``modeling_step3p7vl``).
    When the wrapper sees a request without multimodal data, it delegates
    straight here so plain text generation (GSM8K-style) keeps the original
    behaviour.  MTP layers (45..47) are still loaded only when
    ``MTPDecodingConfig`` enables one-engine MTP; the vision tower lives on
    the wrapper.
    """

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        spec_input_ids: Optional[torch.LongTensor] = None,
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

            # The MTP/spec worker always needs the real token ids. On the
            # multimodal path the main model consumes fused ``inputs_embeds``
            # and ``input_ids`` is None, so the VLM wrapper forwards the
            # pre-fusion token ids via ``spec_input_ids``.
            spec_token_ids = spec_input_ids if spec_input_ids is not None else input_ids
            spec_position_ids = position_ids
            if attn_metadata.padded_num_tokens is not None:
                if spec_token_ids is not None:
                    spec_token_ids = spec_token_ids[: attn_metadata.num_tokens]
                if position_ids is not None:
                    spec_position_ids = _slice_spec_position_ids(
                        position_ids, attn_metadata.num_tokens
                    )

            return self.spec_worker(
                input_ids=spec_token_ids,
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
        _mirror_step3p7_text_aliases(model_config.pretrained_config)

        qc = getattr(model_config, "quant_config", None)
        if qc is not None and qc.exclude_modules is not None:
            qc.exclude_modules = strip_language_model_prefix_from_exclude_modules(
                qc.exclude_modules
            )

        # ``model_loader.py`` snapshots ``pretrained_config.torch_dtype`` into
        # ``extra_attrs['allreduce_dtype']`` before construction; when HF 5.x
        # keeps it as a string (trust_remote_code path), refresh the snapshot
        # so the NCCL window prealloc path sees a real ``torch.dtype``.
        if hasattr(model_config, "extra_attrs") and isinstance(model_config.extra_attrs, dict):
            cur = model_config.extra_attrs.get("allreduce_dtype")
            if isinstance(cur, str):
                mapped = getattr(torch, cur, None)
                model_config.extra_attrs["allreduce_dtype"] = (
                    mapped if isinstance(mapped, torch.dtype) else torch.bfloat16
                )

        _prepare_step3p7_mtp_spec_config(model_config)
        text_config = _get_text_config(model_config)
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

        num_mtp = int(getattr(text_config, "num_nextn_predict_layers", 0))
        mtp_prefixes = tuple(
            f"model.layers.{idx}."
            for idx in range(self.num_hidden_layers, self.num_hidden_layers + num_mtp)
        )
        self.ignored_key_prefixes = (
            "vision_model.",
            "vit_large_projector.",
            *mtp_prefixes[active_mtp_layers:],
        )

    def get_ignored_key_prefixes(self) -> Tuple[str, ...]:
        return self.ignored_key_prefixes

    def _capture_bf16_clamp_weights(self, weights) -> List[int]:
        """Pre-load 2D expert weights for Python-clamp layers, before backend layout transforms.

        The bf16 reference checkpoint's TRTLLMGen-Bf16 backend reshuffles
        ``w3_w1_weight`` into a 4D BlockMajorK layout during
        ``post_load_weights``; the NVFP4 backend likewise transforms its
        weights. The Python clamp path needs the original 2D ``(I, H)``
        per-expert tensors, so we capture them here before
        ``super().load_weights`` runs. FP8 layers are skipped because the FP8
        backend keeps a 3D layout that ``load_clamp_weights_from_fp8_experts``
        can dequantize directly post-load.

        Returns the list of layer indices captured (for diagnostic logging).
        """
        if weights is None or not hasattr(weights, "__getitem__"):
            return []
        captured: List[int] = []
        for layer in self.model.layers:
            moe = getattr(layer, "moe", None)
            if moe is None or not getattr(moe, "_use_python_clamp", False):
                continue
            if getattr(moe, "_clamp_weights_loaded", False):
                continue
            probe_key = f"model.layers.{moe.layer_idx}.moe.experts.0.w1.weight"
            if probe_key not in weights:
                continue
            probe = weights[probe_key]
            if probe.dtype == torch.float8_e4m3fn:
                continue
            try:
                local_ids = moe._local_expert_ids()
                base = f"model.layers.{moe.layer_idx}.moe.experts"
                if probe.dtype == torch.uint8:
                    # NVFP4: batch the per-element ``lut[idx]`` gather on GPU
                    # (orders of magnitude faster than CPU).
                    target_dev = moe._clamp_weight_device()
                    gate_stack = _nvfp4_stack_dequant(weights, base, "w1", local_ids, target_dev)
                    up_stack = _nvfp4_stack_dequant(weights, base, "w3", local_ids, target_dev)
                    down_stack = _nvfp4_stack_dequant(weights, base, "w2", local_ids, target_dev)
                else:
                    gate_stack = torch.stack([weights[f"{base}.{e}.w1.weight"] for e in local_ids])
                    up_stack = torch.stack([weights[f"{base}.{e}.w3.weight"] for e in local_ids])
                    down_stack = torch.stack([weights[f"{base}.{e}.w2.weight"] for e in local_ids])
            except KeyError:
                continue
            moe._copy_clamp_weights(gate_stack, up_stack, down_stack)
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

        Pre-processing before delegating to the generic loader:
        1. Flatten multimodal ``model.language_model.*`` keys to ``model.*``.
        2. Drop ignored prefixes (vision tower, inactive MTP) and FP8 KV scales.
        3. Rewrite active MTP keys to the TRT-LLM ``mtp_block`` / ``shared_head`` layout.
        4. Split stacked routed-expert tensors into per-expert keys.
        5. Capture 2D bf16/NVFP4 expert weights for Python-clamp layers.
        """
        from tensorrt_llm.logger import logger as _logger

        skip_modules = list(skip_modules) if skip_modules else []
        rewrite_language_model_keys(weights)

        if hasattr(weights, "keys"):
            for k in list(weights.keys()):
                drop = any(k.startswith(p) for p in self.ignored_key_prefixes) or k.endswith(
                    _KV_SCALE_SUFFIXES
                )
                if drop:
                    try:
                        del weights[k]
                    except KeyError:
                        pass

        rewrite_mtp_weights_for_step3p7(weights, self.text_config)
        split_stacked_moe_weights(weights, self.text_config)
        self._capture_bf16_clamp_weights(weights)

        rc = DecoderModelForCausalLM.load_weights(
            self,
            weights,
            weight_mapper=weight_mapper,
            skip_modules=[*skip_modules, "draft_model"],
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )
        # Must run after the backend materialises w3_w1/w2.
        for layer in self.model.layers:
            moe = getattr(layer, "moe", None)
            if moe is None or not getattr(moe, "_use_python_clamp", False):
                continue
            try:
                moe.load_clamp_weights_from_fp8_experts()
            except (AttributeError, KeyError, RuntimeError, ValueError) as e:
                _logger.warning(
                    "[Step3p7] failed to populate bf16 expert weights for layer %d (%s): %s. "
                    "Forward will fall back to the FP8 backend without the Python path.",
                    moe.layer_idx,
                    getattr(moe, "_python_path_reason", ""),
                    str(e)[:256],
                )
        return rc
