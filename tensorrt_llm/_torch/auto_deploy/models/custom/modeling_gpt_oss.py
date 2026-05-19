# Copyright 2018 The HuggingFace Team
# Licensed under the Apache License, Version 2.0.
# Original source: https://github.com/huggingface/transformers
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS model with explicit sharding hint ops (sharding-IR default).

Default GPT-OSS modeling for AutoDeploy: every attention Linear is
expressed via ``torch.ops.auto_deploy.torch_linear_simple`` with sharding hint
kwargs (``tp_mode``, ``tp_min_local_shape``, ``layer_type``), and the
post-attention all-reduce is expressed via the ``torch.ops.auto_deploy.all_reduce``
placeholder.  This makes the exported graph a complete, self-contained
specification of how the attention block should be tensor-parallel sharded; the
``apply_sharding_hints`` transform then reads those hints together with a
runtime ``DistConfig`` to produce deterministic, node-local sharding.

Scope of this IR variant (matches the ``qwen3_ir`` / ``qwen3_5_moe_ir``
convention):

  * Attention q/k/v/o use ``torch_linear_simple`` with hints (q/k/v colwise
    + ``tp_min_local_shape=head_dim`` for GQA, o rowwise) plus a trailing
    ``auto_deploy.all_reduce`` for the rowwise output.
  * View ops on q/k/v/attn_out use ``torch.ops.auto_deploy.view`` with
    ``tp_scaled_dim=2`` so the head-count dimension scales with TP.
  * MoE router (``torch_moe_router``) and experts (``torch_moe_dense_mlp``)
    are unchanged from ``modeling_gpt_oss.py`` -- expert weights stay
    replicated under sharding-IR; EP/TP-MoE for the trtllm-gen path
    happens via a separate ``ShardableNode`` (Step 5 of the V4 plan).
  * ``lm_head`` is left as a plain ``nn.Linear`` -- there is no canonical
    sharding-IR pattern for col-parallel-linear-then-all-gather in this
    codebase, and the absolute gain (~80 us / token at TP=4 for
    gpt-oss-120b) is marginal compared to attention TP.  ``qwen3_ir`` and
    ``qwen3_5_moe_ir`` make the same choice.

Historical note: the legacy non-IR ``modeling_gpt_oss.py`` was removed in
favor of this sharding-IR path so TP > 1 attention sharding works out of
the box without an opt-in env var.

Shardable custom ops used:
  - torch.ops.auto_deploy.torch_linear_simple  (tp_mode, tp_min_local_shape, layer_type)
  - torch.ops.auto_deploy.view                 (tp_scaled_dim, layer_type)
  - torch.ops.auto_deploy.all_reduce           (placeholder, layer_type)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._utils import get_hf_rope_theta

from ... import custom_ops  # noqa: F401 -- ensure all custom ops are registered
from ..hf import AutoModelForCausalLMFactory

# GPT-OSS hard-codes these in the HF reference (see modeling_gpt_oss.GptOssExperts).
_GPTOSS_GLU_ALPHA = 1.702
_GPTOSS_GLU_LIMIT_FALLBACK = 7.0


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GptOssModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class GptOssCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# YaRN helpers (faithful copy of transformers._compute_yarn_parameters)
# ---------------------------------------------------------------------------


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rot: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float, max_pos: int, truncate: bool
) -> Tuple[float, float]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_pos)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_pos)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_factor(min_v: float, max_v: float, dim: int) -> torch.Tensor:
    if min_v == max_v:
        max_v = max_v + 0.001
    factor = (torch.arange(dim, dtype=torch.float32) - min_v) / (max_v - min_v)
    return torch.clamp(factor, 0.0, 1.0)


# ---------------------------------------------------------------------------
# RMSNorm (using AD canonical op)
# ---------------------------------------------------------------------------


class GptOssRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


# ---------------------------------------------------------------------------
# Rotary Embedding (YaRN, pre-cached, sliced once per forward)
# ---------------------------------------------------------------------------


class GptOssRotaryEmbedding(nn.Module):
    """YaRN-scaled rotary embedding for GPT-OSS.

    Identical to ``modeling_gpt_oss.GptOssRotaryEmbedding``; no sharding
    hints are needed for the rotary table itself.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()

        attention_scaling = 1.0
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        else:
            rope_type = "default"

        if rope_type == "yarn":
            factor = float(rope_scaling["factor"])
            beta_fast = float(rope_scaling.get("beta_fast", 32.0))
            beta_slow = float(rope_scaling.get("beta_slow", 1.0))
            mscale = rope_scaling.get("mscale", None)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", None)
            attention_factor = rope_scaling.get("attention_factor", None)
            original_max = int(
                rope_scaling.get("original_max_position_embeddings") or max_position_embeddings
            )
            truncate = bool(rope_scaling.get("truncate", True))

            if attention_factor is None:
                if mscale and mscale_all_dim:
                    attention_scaling = float(
                        _yarn_get_mscale(factor, float(mscale))
                        / _yarn_get_mscale(factor, float(mscale_all_dim))
                    )
                else:
                    attention_scaling = _yarn_get_mscale(factor)
            else:
                attention_scaling = float(attention_factor)

            pos_freqs = rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            inv_freq_extra = 1.0 / pos_freqs
            inv_freq_inter = 1.0 / (factor * pos_freqs)

            low, high = _yarn_find_correction_range(
                beta_fast, beta_slow, head_dim, rope_theta, original_max, truncate
            )
            extra_factor = 1.0 - _yarn_linear_ramp_factor(low, high, head_dim // 2)
            inv_freq = inv_freq_inter * (1.0 - extra_factor) + inv_freq_extra * extra_factor
        else:
            inv_freq = 1.0 / (
                rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )

        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos() * attention_scaling, persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin() * attention_scaling, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached[position_ids].to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached[position_ids].to(dtype=x.dtype, device=x.device)
        return cos, sin


# ---------------------------------------------------------------------------
# Router (replaces HF GptOssTopKRouter; eliminates the gptoss_topk_router patch)
# ---------------------------------------------------------------------------


class GptOssTopKRouter(nn.Module):
    """Top-K router: linear projection + topk + softmax + scatter.

    The router lives on every TP rank (replicated) under sharding-IR --
    expert routing decisions must agree across ranks.  No sharding hints
    are needed.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = int(config.num_local_experts)
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_moe_router(
            hidden_states, self.weight, self.bias, self.top_k
        )


# ---------------------------------------------------------------------------
# Experts (stacked weights with biases; uses torch_moe_dense_mlp)
# ---------------------------------------------------------------------------


class GptOssExperts(nn.Module):
    """GPT-OSS dense experts module — bf16 placeholder layout.

    Always allocates the four bf16 placeholder params (``gate_up_proj`` /
    ``gate_up_proj_bias`` / ``down_proj`` / ``down_proj_bias``) and emits
    ``torch_moe_dense_mlp`` in :meth:`forward`. Quantization (MXFP4 →
    Triton / TRT-LLM-Gen) is handled by the ``quantize_mxfp4_moe`` transform,
    which rewrites the FX graph + swaps parameters at PATTERN_MATCHER time
    (see :mod:`tensorrt_llm._torch.auto_deploy.transform.library.mxfp4_moe`).

    Dtype protection (kept here as a generic mechanism): when a transform
    registers MXFP4-specific params (uint8 weights / ue8m0 scales / fp32
    biases / fp32 SwiGLU constants) on this module, it should also set
    ``self._dtype_protected_params`` to a tuple of those param names. The
    overridden :meth:`_apply` then preserves their dtype across
    ``model.to(dtype)`` walks (which would otherwise corrupt the
    kernel-required dtypes). Modules without that attribute behave like a
    plain ``nn.Module``.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = int(config.num_local_experts)
        self.hidden_size = int(config.hidden_size)
        self.expert_dim = int(config.intermediate_size)
        self.alpha = _GPTOSS_GLU_ALPHA
        self.limit = float(getattr(config, "swiglu_limit", _GPTOSS_GLU_LIMIT_FALLBACK))

        # Bf16 placeholder params. On MXFP4 checkpoints the
        # ``quantize_mxfp4_moe`` transform deletes these and registers the
        # backend-specific MXFP4 params before WEIGHT_LOAD fires (so the
        # placeholders never get materialised from meta device).
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

    def _apply(self, fn, recurse=True):
        """Preserve dtype on params listed in ``self._dtype_protected_params``.

        The ``quantize_mxfp4_moe`` transform sets ``_dtype_protected_params``
        to a tuple of names whose kernel-required dtype (uint8 for MXFP4
        weights and ue8m0 scales, float32 for biases and SwiGLU constants)
        must survive ``model.to(dtype)``. Without this protection
        ``model.to(bf16)`` would downcast those params and produce garbage
        MoE output.

        If the attribute is absent or empty, this override is a no-op and
        behaves identically to ``nn.Module._apply``.
        """
        protected_names = tuple(getattr(self, "_dtype_protected_params", ()) or ())
        if not protected_names:
            return super()._apply(fn, recurse=recurse)

        protected = {}
        for name in protected_names:
            p = self._parameters.get(name)
            if p is not None:
                protected[name] = (p, p.dtype)
                # Drop temporarily so super()._apply doesn't include it in its walk.
                del self._parameters[name]

        super()._apply(fn, recurse=recurse)

        # Re-attach with dtype preserved. Apply ``fn`` to pick up the device /
        # layout part of the transform, then cast back to the original dtype.
        for name, (orig_param, orig_dtype) in protected.items():
            new_data = fn(orig_param.data)
            if new_data.dtype != orig_dtype:
                new_data = new_data.to(orig_dtype)
            orig_param.data = new_data
            self._parameters[name] = orig_param

        return self

    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        # Legacy bf16 dense forward; MXFP4 trtllm-gen path bypasses this via
        # the ``GptOssMLP.forward`` dispatch.
        return torch.ops.auto_deploy.torch_moe_dense_mlp(
            hidden_states,
            routing_weights,
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
            self.alpha,
            self.limit,
        )


class GptOssMLP(nn.Module):
    """Router + experts. Drop-in replacement for HF ``GptOssMLP`` in prefill."""

    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)
        self.top_k = int(getattr(config, "num_experts_per_tok", 4))
        # ``RoutingMethodType.Renormalize`` == 1 (matches PT's gpt-oss path).
        self._routing_method_type = 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Bf16 dense MoE forward. Quantization is applied by the
        ``quantize_mxfp4_moe`` transform, which rewrites the underlying
        ``torch_moe_dense_mlp`` node into a backend-specific fused op
        (Triton or TRT-LLM-Gen) and, for the TRT-LLM-Gen path, inserts
        the MoE-TP all-reduce after the downstream ``view`` so the
        ``view -> AR -> add -> norm`` ordering matches
        ``fuse_allreduce_residual_rmsnorm``.
        """
        bsz, seq_len, hidden_dim = hidden_states.shape
        routing_weights = self.router(hidden_states)  # [B*S, E]
        out = self.experts(hidden_states, routing_weights)
        return out.view(bsz, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Attention (GQA + sinks + per-layer sliding window) -- sharding-IR variant
# ---------------------------------------------------------------------------


class GptOssAttention(nn.Module):
    """GPT-OSS attention with sharding hints.

    Sharding strategy (matches ``qwen3_ir.Qwen3Attention``):
      q_proj -> colwise  (+ tp_min_local_shape=head_dim for GQA)
      k_proj -> colwise  (+ tp_min_local_shape=head_dim for GQA)
      v_proj -> colwise  (+ tp_min_local_shape=head_dim for GQA)
      view   -> tp_scaled_dim=2 (head-count dim shrinks with TP)
      o_proj -> rowwise + auto_deploy.all_reduce
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_bias = bool(getattr(config, "attention_bias", True))

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=self.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=self.attention_bias
        )

        self.sinks = nn.Parameter(torch.empty(self.num_heads))

        # Per-layer sliding window: only enabled on layers tagged "sliding_attention".
        layer_types = getattr(config, "layer_types", None)
        is_sliding = layer_types is not None and layer_types[layer_idx] == "sliding_attention"
        sliding_window = getattr(config, "sliding_window", None)
        self.sliding_window = int(sliding_window) if (is_sliding and sliding_window) else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.q_proj.weight,
            self.q_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        k = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.k_proj.weight,
            self.k_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        v = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.v_proj.weight,
            self.v_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )

        q = torch.ops.auto_deploy.view(
            q,
            [bsz, q_len, self.num_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        k = torch.ops.auto_deploy.view(
            k,
            [bsz, q_len, self.num_kv_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        v = torch.ops.auto_deploy.view(
            v,
            [bsz, q_len, self.num_kv_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )

        cos, sin = position_embeddings
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scaling,
            sinks=self.sinks,
            sliding_window=self.sliding_window,
            layout="bsnd",
        )

        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [bsz, q_len, self.num_heads * self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )

        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output,
            self.o_proj.weight,
            self.o_proj.bias,
            tp_mode="rowwise",
            layer_type="mha",
        )
        attn_output = torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mha")
        return attn_output


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = GptOssAttention(config, layer_idx)
        self.mlp = GptOssMLP(config)
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Model + CausalLM
# ---------------------------------------------------------------------------


class GptOssPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    _no_split_modules = ["GptOssDecoderLayer"]
    supports_gradient_checkpointing = False


class GptOssModel(GptOssPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None)
        )
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        head_dim = int(
            getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        )
        self.rotary_emb = GptOssRotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            # FIX: transformers 5.x moved rope_theta to config.rope_scaling['rope_theta'].
            # Use get_hf_rope_theta() helper (same as PT modeling).
            rope_theta=get_hf_rope_theta(config, 10000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> GptOssModelOutput:
        assert position_ids is not None, "position_ids is required"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings=position_embeddings)
        hidden_states = self.norm(hidden_states)
        return GptOssModelOutput(last_hidden_state=hidden_states)


class GptOssForCausalLM(GptOssPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GptOssModel(config)
        # lm_head stays as plain nn.Linear -- matches qwen3_ir convention; no
        # canonical sharding-IR pattern for col-parallel-then-all-gather exists
        # in this codebase, and the absolute gain from sharding lm_head on
        # gpt-oss-120b is marginal (<1% of total ITL).
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MXFP4 + trtllm-gen weight prep (the raw-HF → prepared-layout CPU
        # conversion done by a ``load_state_dict`` pre-hook) is now registered
        # by the ``quantize_mxfp4_moe`` transform when it picks the ``trtllm``
        # backend, not here. Keeping it transform-side avoids the modeling
        # code having to know about MXFP4-specific param layouts and matches
        # the dispatcher pattern used by other quantizations in AutoDeploy.

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> GptOssCausalLMOutput:
        assert position_ids is not None, "position_ids is required"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return GptOssCausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# Registers AFTER ``modeling_gpt_oss``; last-registration-wins semantics in the
# factory means this IR variant takes precedence when ``AD_USE_IR_MODELS`` is
# set (see ``models/custom/__init__.py``).
AutoModelForCausalLMFactory.register_custom_model_cls("GptOssConfig", GptOssForCausalLM)
