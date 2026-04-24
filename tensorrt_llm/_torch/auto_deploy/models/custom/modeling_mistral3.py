# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill-only Mistral3 wrapper with a custom Mistral4 text backbone for AutoDeploy.

Source checkpoints:
  - mistralai/Mistral-Small-3.1-24B-Instruct-2503
  - mistralai/Mistral-Small-3.2-24B-Instruct-2506
  - mistralai/Mistral-Small-4-119B-2603

This implementation follows the repo's current multimodal AD flow:
  * The top-level Mistral3 wrapper is present so AutoModelForImageTextToTextFactory can
    identify and export the inner text model.
  * Vision modules are intentionally omitted from the AD model. Only the text path is exported.
    For multimodal use, callers may provide pre-merged ``inputs_embeds``.
  * The new Mistral4 text backbone is translated into a lean prefill-only implementation using
    AutoDeploy canonical ops for RMSNorm, RoPE, MLA, and MoE.
"""

import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from tokenizers import Tokenizer
from torch import nn
from transformers import (
    AutoConfig,
    Mistral3Config,
    PixtralImageProcessorFast,
    PixtralProcessor,
    PretrainedConfig,
    PreTrainedTokenizerFast,
)
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM
from transformers.utils import ModelOutput, cached_file

from ..._compat import ActivationType
from ..factory import ModelFactoryRegistry
from ..hf import AutoModelForCausalLMFactory, AutoModelForImageTextToTextFactory
from . import mla_rope_utils


class Mistral4TextConfig(PretrainedConfig):
    """Minimal text config for Mistral Small 4."""

    model_type = "mistral4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-6,
        # MLA
        kv_lora_rank: int = 256,
        q_lora_rank: int = 1024,
        qk_head_dim: int = 128,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        # MoE
        n_routed_experts: int = 128,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 4,
        n_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        # RoPE
        rope_theta: float = 10000.0,
        rope_parameters: Optional[dict] = None,
        rope_interleave: bool = True,
        # Other
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        pad_token_id: int = 11,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_head_dim = qk_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.rope_interleave = rope_interleave
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range

        rope_parameters = rope_parameters or {
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 128.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": rope_theta,
            "llama_4_scaling_beta": 0.1,
        }
        self.rope_parameters = rope_parameters
        self.rope_scaling = rope_parameters
        self.llama_4_scaling = {
            "original_max_position_embeddings": rope_parameters.get(
                "original_max_position_embeddings", 8192
            ),
            "beta": rope_parameters.get("llama_4_scaling_beta", 0.1),
        }

        for key in [
            "architectures",
            "model_type",
            "transformers_version",
            "dtype",
            "use_cache",
            "quantization_config",
            "head_dim",
            "sliding_window",
            "pretraining_tp",
            "topk_method",
            "scoring_func",
        ]:
            kwargs.pop(key, None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def _register_config(model_type: str, config_cls) -> None:
    try:
        AutoConfig.register(model_type, config_cls, exist_ok=True)
    except TypeError:
        try:
            AutoConfig.register(model_type, config_cls)
        except ValueError:
            pass


_register_config("mistral4", Mistral4TextConfig)


class Mistral4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight.to(hidden_states.dtype), self.variance_epsilon
        )


class Mistral4RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class Mistral4YarnRotaryEmbedding(Mistral4RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float,
        original_max_position_embeddings: int,
        beta_fast: float,
        beta_slow: float,
        mscale: float,
        mscale_all_dim: float,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base)

    @staticmethod
    def _yarn_find_correction_dim(
        num_rotations: float, dim: int, base: float, max_position_embeddings: int
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    @classmethod
    def _yarn_find_correction_range(
        cls, low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int
    ) -> Tuple[int, int]:
        low = math.floor(cls._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(
            cls._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _yarn_get_mscale(scale: float, mscale: float) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    def _build_cache(self, seq_len: int) -> None:
        dim = self.dim
        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        mscale = float(
            self._yarn_get_mscale(self.scaling_factor, self.mscale)
            / self._yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        self.register_buffer("_ad_cos_cached", emb.cos() * mscale, persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin() * mscale, persistent=False)


class Mistral4MLP(nn.Module):
    def __init__(self, config: Mistral4TextConfig, intermediate_size: Optional[int] = None) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Mistral4MoEGate(nn.Module):
    def __init__(self, config: Mistral4TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(torch.empty((config.n_routed_experts, config.hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(config.n_routed_experts), persistent=False
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.ops.auto_deploy.torch_moe_router(
            hidden_states,
            self.weight.to(hidden_states.dtype),
            self.e_score_correction_bias.to(hidden_states.dtype),
            self.top_k,
        )
        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        return topk_idx, topk_weight


class Mistral4MoE(nn.Module):
    def __init__(self, config: Mistral4TextConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.experts = nn.ModuleList(
            [
                Mistral4MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )
        self.shared_experts = (
            Mistral4MLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
            )
            if config.n_shared_experts
            else None
        )
        self.gate = Mistral4MoEGate(config)
        self._register_load_state_dict_pre_hook(self._load_experts_from_fused_checkpoint)

    def _owned_expert_ids(self) -> list[int]:
        """Derive expert ids from the params/buffers that currently exist on this module.

        This stays shard-agnostic in modeling code. On an unsharded module the state dict
        contains all experts; after AD sharding it contains only the surviving local experts.
        """
        expert_ids = set()
        for key in self.state_dict().keys():
            if not key.startswith("experts."):
                continue
            parts = key.split(".", 3)
            if len(parts) < 4:
                continue
            try:
                expert_ids.add(int(parts[1]))
            except ValueError:
                continue
        return sorted(expert_ids)

    def _load_experts_from_fused_checkpoint(self, state_dict, prefix, *args):
        gate_up_key = prefix + "experts.gate_up_proj"
        down_key = prefix + "experts.down_proj"
        gate_up_scale_key = prefix + "experts.gate_up_proj_scale_inv"
        down_scale_key = prefix + "experts.down_proj_scale_inv"
        gate_up_act_scale_key = prefix + "experts.gate_up_proj_activation_scale"
        down_act_scale_key = prefix + "experts.down_proj_activation_scale"
        local_expert_ids = self._owned_expert_ids()

        if gate_up_key in state_dict:
            fused = state_dict.pop(gate_up_key)
            intermediate_dim = fused.shape[1] // 2
            gate_weights = fused[:, :intermediate_dim, :]
            up_weights = fused[:, intermediate_dim:, :]
            for idx in local_expert_ids:
                state_dict[f"{prefix}experts.{idx}.gate_proj.weight"] = gate_weights[idx]
                state_dict[f"{prefix}experts.{idx}.up_proj.weight"] = up_weights[idx]

        if gate_up_scale_key in state_dict:
            fused_scale = state_dict.pop(gate_up_scale_key)
            for idx in local_expert_ids:
                # The checkpoint stores one static FP8 weight scale per fused gate/up expert.
                # Both split projections reuse that same expert-local scale.
                expert_scale = fused_scale[idx].reshape(())
                state_dict[f"{prefix}experts.{idx}.gate_proj.weight_scale"] = expert_scale
                state_dict[f"{prefix}experts.{idx}.up_proj.weight_scale"] = expert_scale

        if gate_up_act_scale_key in state_dict:
            fused_act_scale = state_dict.pop(gate_up_act_scale_key)
            for idx in local_expert_ids:
                # The fused gate/up checkpoint also uses one input scale per expert.
                state_dict[f"{prefix}experts.{idx}.gate_proj.input_scale"] = fused_act_scale[idx]
                state_dict[f"{prefix}experts.{idx}.up_proj.input_scale"] = fused_act_scale[idx]

        if down_key in state_dict:
            fused = state_dict.pop(down_key)
            for idx in local_expert_ids:
                state_dict[f"{prefix}experts.{idx}.down_proj.weight"] = fused[idx]

        if down_scale_key in state_dict:
            fused_scale = state_dict.pop(down_scale_key)
            for idx in local_expert_ids:
                state_dict[f"{prefix}experts.{idx}.down_proj.weight_scale"] = fused_scale[
                    idx
                ].reshape(())

        if down_act_scale_key in state_dict:
            fused_act_scale = state_dict.pop(down_act_scale_key)
            for idx in local_expert_ids:
                state_dict[f"{prefix}experts.{idx}.down_proj.input_scale"] = fused_act_scale[idx]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        selected_experts, routing_weights = self.gate(hidden_states_flat)
        output = torch.ops.auto_deploy.torch_moe(
            hidden_states_flat,
            selected_experts,
            routing_weights,
            [expert.gate_proj.weight for expert in self.experts],
            [expert.down_proj.weight for expert in self.experts],
            [expert.up_proj.weight for expert in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
        )
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states_flat)
        return output.view(*original_shape).to(hidden_states.dtype)


class Mistral4Attention(nn.Module):
    def __init__(self, config: Mistral4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_parameters.get("rope_theta", 10000.0)

        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = Mistral4RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_a_layernorm = Mistral4RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        # kv_b_proj is absorbed into torch_mla, so it bypasses the generic FP8 linear transform.
        # Dequantize it during checkpoint load using the model-card reference conversion.
        self._register_load_state_dict_pre_hook(self._load_absorbed_kv_b_proj_from_fp8_checkpoint)
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if config.rope_scaling is not None:
            scale = config.rope_scaling.get("factor", 1.0)
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0.0)
            if mscale_all_dim:
                yarn_scale = Mistral4YarnRotaryEmbedding._yarn_get_mscale(scale, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * yarn_scale * yarn_scale
        self._init_rope()

    def _init_rope(self) -> None:
        rope_scaling = self.config.rope_scaling
        if (
            rope_scaling is None
            or rope_scaling.get("type", rope_scaling.get("rope_type")) != "yarn"
        ):
            self.rotary_emb = Mistral4RotaryEmbedding(
                self.qk_rope_head_dim, self.max_position_embeddings, self.rope_theta
            )
            return
        self.rotary_emb = Mistral4YarnRotaryEmbedding(
            self.qk_rope_head_dim,
            self.max_position_embeddings,
            self.rope_theta,
            rope_scaling["factor"],
            rope_scaling.get("original_max_position_embeddings", 8192),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("mscale", 1.0),
            rope_scaling.get("mscale_all_dim", 1.0),
        )

    def _load_absorbed_kv_b_proj_from_fp8_checkpoint(self, state_dict, prefix, *args):
        weight_key = prefix + "kv_b_proj.weight"
        scale_key = prefix + "kv_b_proj.weight_scale_inv"
        activation_scale_key = prefix + "kv_b_proj.activation_scale"
        if weight_key not in state_dict or scale_key not in state_dict:
            return

        weight = state_dict[weight_key]
        if weight.dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}:
            return

        target_dtype = self.kv_b_proj.weight.dtype
        scale = state_dict.pop(scale_key).to(torch.float32)
        state_dict[weight_key] = weight.to(torch.float32).mul(scale).to(target_dtype)
        state_dict.pop(activation_scale_key, None)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(batch_size, seq_len, 1, self.qk_rope_head_dim)

        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
        cos = cos[position_ids]
        sin = sin[position_ids]
        q_pe_rotated, k_pe_rotated = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q_pe,
            k_pe,
            cos,
            sin,
            2,
        )

        attn_output = torch.ops.auto_deploy.torch_mla(
            q_nope,
            q_pe_rotated,
            compressed_kv,
            k_pe_rotated,
            self.kv_b_proj.weight,
            True,
            self.softmax_scale,
            "bsnd",
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)


class Mistral4DecoderLayer(nn.Module):
    def __init__(self, config: Mistral4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Mistral4Attention(config, layer_idx)
        self.input_layernorm = Mistral4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Mistral4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Mistral4MoE(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@dataclass
class Mistral4ModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


class Mistral4PreTrainedModel(PreTrainedModel):
    config_class = Mistral4TextConfig
    base_model_prefix = "model"


class Mistral4Model(Mistral4PreTrainedModel):
    def __init__(self, config: Mistral4TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Mistral4DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = Mistral4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Mistral4ModelOutput:
        del kwargs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds.")
        if position_ids is None:
            raise ValueError("position_ids must be provided for AutoDeploy export.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids)
        hidden_states = self.norm(hidden_states)
        return Mistral4ModelOutput(last_hidden_state=hidden_states)


class Mistral4ForCausalLM(Mistral4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Mistral4TextConfig, **kwargs):
        super().__init__(config)
        del kwargs
        self.model = Mistral4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._register_load_state_dict_pre_hook(
            partial(
                mla_rope_utils._rope_deinterleave_load_hook,
                qk_rope_head_dim=config.qk_rope_head_dim,
                qk_nope_head_dim=config.qk_nope_head_dim,
                num_heads=config.num_attention_heads,
                kv_lora_rank=config.kv_lora_rank,
                num_layers=config.num_hidden_layers,
            )
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        model_kwargs = dict(kwargs)
        model_kwargs["input_ids"] = input_ids
        model_kwargs["position_ids"] = position_ids
        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds
        outputs = self.model(**model_kwargs)
        logits = self.lm_head(outputs.last_hidden_state).float()
        return CausalLMOutputWithPast(logits=logits)


@dataclass
class Mistral3ADOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class Mistral3ForConditionalGenerationAD(PreTrainedModel, GenerationMixin):
    """Top-level Mistral3 wrapper used by the multimodal AD factory.

    Only the text submodule is exported. The wrapper is still needed so:
      1. HF config loading succeeds for ``text_config.model_type == "mistral4"``.
      2. AutoModelForImageTextToTextFactory can locate the inner ``language_model`` submodule.
    """

    config_class = Mistral3Config
    _checkpoint_conversion_mapping = {
        "^model.language_model": "language_model",
        "^model.vision_tower": "vision_tower",
        "^model.multi_modal_projector": "multi_modal_projector",
        "^lm_head": "language_model.lm_head",
    }

    def __init__(self, config: Mistral3Config, **kwargs):
        super().__init__(config)
        del kwargs
        text_config = config.text_config
        if getattr(text_config, "model_type", None) == "mistral4":
            self.language_model = Mistral4ForCausalLM(text_config)
        else:
            self.language_model = AutoModelForCausalLM.from_config(
                text_config, trust_remote_code=True
            )
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Mistral3ADOutput:
        model_kwargs = dict(kwargs)
        model_kwargs["input_ids"] = input_ids
        model_kwargs["position_ids"] = position_ids
        if inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = inputs_embeds
        outputs = self.language_model(**model_kwargs)
        return Mistral3ADOutput(logits=outputs.logits)


AutoModelForCausalLMFactory.register_custom_model_cls("Mistral4TextConfig", Mistral4ForCausalLM)
AutoModelForCausalLMFactory.register_custom_model_cls(
    "Mistral3Config", Mistral3ForConditionalGenerationAD
)
AutoModelForImageTextToTextFactory.register_custom_model_cls(
    "Mistral3Config", Mistral3ForConditionalGenerationAD
)

# ---------------------------------------------------------------------------
# Wrapper tokenizer / processor for Mistral Small 4
#
# The upstream HF checkpoint references a tokenizer class that requires
# transformers v5+.  These thin wrappers delegate to the upstream tokenizer
# assets while staying compatible with the current transformers version.
# ---------------------------------------------------------------------------

_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_TOKENIZER_FILE = "tokenizer.json"


class ADMistralSmall4Tokenizer(PreTrainedTokenizerFast):
    """Wrapper that loads the upstream Mistral Small 4 tokenizer on current transformers."""

    vocab_files_names = {"tokenizer_file": _TOKENIZER_FILE}
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *inputs,
        **kwargs,
    ) -> "ADMistralSmall4Tokenizer":
        del inputs
        for k in ("_from_auto", "_commit_hash", "trust_remote_code"):
            kwargs.pop(k, None)

        config_path = cached_file(pretrained_model_name_or_path, _TOKENIZER_CONFIG_FILE, **kwargs)
        assert config_path is not None
        config = json.loads(Path(config_path).read_text())

        tokenizer_file = cached_file(pretrained_model_name_or_path, _TOKENIZER_FILE, **kwargs)
        assert tokenizer_file is not None

        tokenizer = cls(
            tokenizer_object=Tokenizer.from_file(tokenizer_file),
            name_or_path=str(pretrained_model_name_or_path),
            bos_token=config.get("bos_token"),
            eos_token=config.get("eos_token"),
            unk_token=config.get("unk_token"),
            pad_token=config.get("pad_token"),
            additional_special_tokens=config.get("extra_special_tokens", []),
            clean_up_tokenization_spaces=config.get("clean_up_tokenization_spaces", False),
            model_max_length=config.get("model_max_length"),
            padding_side=config.get("padding_side", "left"),
            truncation_side=config.get("truncation_side", "left"),
        )

        template_path = cached_file(
            pretrained_model_name_or_path,
            _CHAT_TEMPLATE_FILE,
            _raise_exceptions_for_missing_entries=False,
            **kwargs,
        )
        if template_path is not None:
            tokenizer.chat_template = Path(template_path).read_text()

        return tokenizer


class ADMistralSmall4Processor(PixtralProcessor):
    """Wrapper Pixtral processor wired to ``ADMistralSmall4Tokenizer``."""

    @classmethod
    def register_for_auto_class(cls, auto_class: str = "AutoProcessor") -> None:
        del auto_class

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs,
    ) -> "ADMistralSmall4Processor":
        for k in ("_from_auto", "_commit_hash", "trust_remote_code"):
            kwargs.pop(k, None)

        config_path = cached_file(pretrained_model_name_or_path, "processor_config.json", **kwargs)
        assert config_path is not None
        processor_config = json.loads(Path(config_path).read_text())

        image_processor = PixtralImageProcessorFast.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True, **kwargs
        )
        tokenizer = ADMistralSmall4Tokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=processor_config.get("patch_size", 16),
            spatial_merge_size=processor_config.get("spatial_merge_size", 1),
            chat_template=getattr(tokenizer, "chat_template", None),
            image_token=processor_config.get("image_token", "[IMG]"),
            image_break_token=processor_config.get("image_break_token", "[IMG_BREAK]"),
            image_end_token=processor_config.get("image_end_token", "[IMG_END]"),
        )


@ModelFactoryRegistry.register("Mistral3ForConditionalGeneration")
class Mistral3ForConditionalGenerationFactory(AutoModelForImageTextToTextFactory):
    """Factory that wires the wrapper tokenizer/processor for Mistral Small 4."""

    def init_tokenizer(self) -> Optional[Any]:
        processor = self.init_processor()
        if processor is None:
            return None
        return processor.tokenizer

    def init_processor(self) -> Optional[Any]:
        if self.tokenizer is None:
            return None
        return ADMistralSmall4Processor.from_pretrained(self.tokenizer)


# __init_subclass__ resets _custom_model_mapping for each subclass, so re-register here.
Mistral3ForConditionalGenerationFactory.register_custom_model_cls(
    "Mistral3Config", Mistral3ForConditionalGenerationAD
)
