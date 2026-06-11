# Copyright 2018 The HuggingFace Team
# Licensed under the Apache License, Version 2.0.
# Original source: https://github.com/huggingface/transformers
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeekV3 model with explicit sharding-hint custom ops for AutoDeploy IR sharding.

Sharding-aware variant of ``modeling_deepseek.py``. The non-IR file is the
authoritative source of model logic; this file applies a strictly mechanical
structural transform that encodes TP sharding intent in the exported FX graph
via ``auto_deploy`` custom ops with sharding-hint kwargs. The
``apply_sharding_hints`` transform reads these hints together with a runtime
``DistConfig`` and applies deterministic, node-local sharding.

Mechanical deltas vs ``modeling_deepseek.py``:
* ``nn.Linear`` / ``F.linear`` projections become
  ``torch.ops.auto_deploy.torch_linear_simple`` calls carrying ``tp_mode`` and
  ``layer_type`` hints.
* TP-scaled head reshapes use ``torch.ops.auto_deploy.view`` with
  ``tp_scaled_dim``.
* Rowwise projections / MoE merge points are followed by
  ``torch.ops.auto_deploy.all_reduce(..., layer_type=...)``.
* ``torch.ops.auto_deploy.torch_mla`` carries ``enable_sharding=True`` and
  ``layer_type="mla"`` so ``_apply_hint_mla`` shards ``kv_b_proj.weight``
  column-wise per head without decomposing the MLA op.
* ``torch.ops.auto_deploy.torch_moe`` carries ``layer_type="moe"``.

Sharding strategy:
* **MLA** (``DeepSeekV3Attention``): ``q_a_proj`` and ``kv_a_proj_with_mqa``
  stay replicated (``tp_mode="none"``); ``q_b_proj`` is colwise (sharded by
  ``num_heads``); ``o_proj`` is rowwise + ``all_reduce(layer_type="mla")``;
  ``torch_mla`` carries ``enable_sharding=True, layer_type="mla"``.
* **MoE** (``DeepSeekV3MoE``): the ``noaux_tc`` router gate is TP-replicated
  and keeps its ``torch.ops.trtllm.{dsv3_router_gemm_op, noaux_tc_op}`` calls
  verbatim from the non-IR base (no sharding hints; AD has no fusion that
  recovers these kernels from a vanilla rewrite). ``torch_moe`` carries
  ``layer_type="moe"``. The shared expert MLP is constructed with
  ``add_all_reduce=False, layer_type="shared_expert"`` so it stays REPLICATED
  (excluded from ``shard_layers``; NVFP4 TP-sharding corrupts its deswizzled
  weight scales). The routed (EP) output is ``all_reduce(layer_type="moe")``-d
  first, then the replicated shared output is added.
* **MLP** (``DeepSeekV3MLP``): SwiGLU gate/up are colwise, down is rowwise +
  ``all_reduce(layer_type="mlp")``.

Both ``load_state_dict`` pre-hooks from the non-IR file are preserved verbatim:
* ``mla_rope_utils._rope_deinterleave_load_hook`` -- permutes RoPE columns of
  ``q_b_proj`` and ``kv_a_proj_with_mqa`` from interleaved to NeoX layout so
  the forward can use ``torch_rope_with_explicit_cos_sin``
  (-> ``flashinfer_rope``).
* ``mla_rope_utils._kv_b_proj_dequant_load_hook`` -- dequantizes FP8
  ``kv_b_proj.weight`` using its block-wise ``weight_scale_inv``;
  ``kv_b_proj`` is consumed directly by ``torch_mla`` (not via a quantized
  linear op), so the FineGrainedFP8 transform skips it.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ... import custom_ops  # noqa: F401 -- register all ops
from ..._compat import ActivationType
from ..hf import AutoModelForCausalLMFactory
from . import mla_rope_utils
from .rotary_utils import RotaryEmbeddingBase, build_rope_cos_sin_cache


class DeepSeekV3RMSNorm(nn.Module):
    """RMS Normalization for DeepSeekV3."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.triton_rms_norm(
            hidden_states, self.weight, self.variance_epsilon
        ).to(hidden_states.dtype)


class DeepSeekV3RotaryEmbedding(RotaryEmbeddingBase):
    """Rotary Position Embedding for DeepSeekV3.

    Keeps only the small inv_freq buffer before graph-cache transforms. The full
    cos/sin table is graph-computed and materialized by later RoPE transforms.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return build_rope_cos_sin_cache(
            self.inv_freq, self.max_position_embeddings, x, self.attention_scaling
        )


class DeepSeekV3YarnRotaryEmbedding(DeepSeekV3RotaryEmbedding):
    """YaRN-extended rotary embedding for DeepSeekV3."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
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

        _mscale = float(
            self._yarn_get_mscale(self.scaling_factor, self.mscale)
            / self._yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        self.attention_scaling = _mscale

    @staticmethod
    def _yarn_find_correction_dim(
        num_rotations: float, dim: int, base: float = 10000, max_position_embeddings: int = 2048
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _yarn_find_correction_range(
        self, low_rot: int, high_rot: int, dim: int, base: float, max_position_embeddings: int
    ) -> Tuple[int, int]:
        low = math.floor(
            self._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            self._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)


class DeepSeekV3MLP(nn.Module):
    """MLP layer for DeepSeekV3 (SwiGLU activation).

    When used as a shared expert inside MoE, ``add_all_reduce=False`` and
    ``layer_type="moe"`` so the closing all_reduce is deferred to the merge
    point and combined with the routed expert output.
    """

    def __init__(
        self,
        config,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        add_all_reduce: bool = True,
        layer_type: str = "mlp",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.add_all_reduce = add_all_reduce
        self.layer_type = layer_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.ops.auto_deploy.torch_linear_simple(
            x,
            self.gate_proj.weight,
            self.gate_proj.bias,
            tp_mode="colwise",
            layer_type=self.layer_type,
        )
        up = torch.ops.auto_deploy.torch_linear_simple(
            x,
            self.up_proj.weight,
            self.up_proj.bias,
            tp_mode="colwise",
            layer_type=self.layer_type,
        )
        down = torch.ops.auto_deploy.torch_linear_simple(
            self.act_fn(gate) * up,
            self.down_proj.weight,
            self.down_proj.bias,
            tp_mode="rowwise",
            layer_type=self.layer_type,
        )
        if self.add_all_reduce:
            down = torch.ops.auto_deploy.all_reduce(down, layer_type=self.layer_type)
        return down


class DeepSeekV3MoEGate(nn.Module):
    """MoE Gating for DeepSeekV3 with noaux_tc top-k selection.

    The router gate is TP-replicated; weight and outputs are identical on every
    rank, so no sharding hints are applied here. The fused
    ``torch.ops.trtllm.dsv3_router_gemm_op`` and
    ``torch.ops.trtllm.noaux_tc_op`` calls are kept verbatim from the non-IR
    base -- AD has no transform that recovers these kernels from a vanilla
    PyTorch rewrite.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize gate weights using kaiming uniform (matches original DeepSeek implementation)."""
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (selected_experts, routing_weights)."""
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute router logits
        if self.weight.dtype == torch.float32:
            router_logits = F.linear(hidden_states_flat.float(), self.weight)
        else:
            router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_states_flat, self.weight.t(), bias=None, out_dtype=torch.float32
            )

        # Use fused noaux_tc_op kernel for top-k selection
        topk_weights, topk_indices = torch.ops.trtllm.noaux_tc_op(
            router_logits,
            self.e_score_correction_bias,
            self.n_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
        )

        return topk_indices, topk_weights


class DeepSeekV3MoE(nn.Module):
    """Mixture of Experts layer for DeepSeekV3.

    Routed experts are dispatched via ``torch_moe`` (sharded by
    ``apply_sharding_hints`` using ``layer_type="moe"``). The shared expert is a
    TP-sharded MLP whose closing all_reduce is deferred so the routed and
    shared partial sums can be combined with a single
    ``all_reduce(layer_type="moe")`` at the merge point.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        # Routed experts
        self.experts = nn.ModuleList(
            [
                DeepSeekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )

        # Gate
        self.gate = DeepSeekV3MoEGate(config)

        # Shared experts (if configured)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepSeekV3MLP(
                config,
                intermediate_size=intermediate_size,
                add_all_reduce=False,
                layer_type="shared_expert",
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape

        selected_experts, routing_weights = self.gate(hidden_states)

        # Compute shared expert BEFORE routed experts so that in the FX graph
        # the shared-expert nodes precede the MoE node.  This lets the
        # multi_stream_moe transform overlap them on separate CUDA streams.
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts(identity)

        # Use torch_moe custom op for routed experts
        final_hidden_states = torch.ops.auto_deploy.torch_moe(
            hidden_states.view(-1, hidden_states.shape[-1]),
            selected_experts,
            routing_weights,
            w1_weight=[expert.gate_proj.weight for expert in self.experts],
            w2_weight=[expert.down_proj.weight for expert in self.experts],
            w3_weight=[expert.up_proj.weight for expert in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
            layer_type="moe",
        )

        final_hidden_states = final_hidden_states.view(*orig_shape)

        # All-reduce the (EP-sharded) routed-expert output first, then add the
        # replicated shared-expert output. The shared expert is excluded from TP
        # sharding (NVFP4 TP-sharding corrupts its deswizzled weight scales), so
        # adding it before the all-reduce would scale it by the TP world size.
        final_hidden_states = torch.ops.auto_deploy.all_reduce(
            final_hidden_states, layer_type="moe"
        )

        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_expert_output

        return final_hidden_states.to(hidden_states.dtype)


class DeepSeekV3Attention(nn.Module):
    """Multi-head Latent Attention (MLA) for DeepSeekV3.

    Uses compressed KV representation with latent projections.

    Sharding strategy:
      ``q_a_proj`` / ``kv_a_proj_with_mqa`` -> ``tp_mode="none"`` (replicated
      latent projections).
      ``q_b_proj`` -> ``tp_mode="colwise"`` (sharded by ``num_heads``).
      Q reshape -> ``auto_deploy.view`` with ``tp_scaled_dim=2``.
      ``torch_mla`` -> ``enable_sharding=True, layer_type="mla"``. Do NOT
      decompose ``torch_mla`` into separate linears + ``torch_attention`` --
      ``_apply_hint_mla`` shards ``kv_b_proj.weight`` column-wise per head.
      Post-attention reshape -> ``auto_deploy.view`` with ``tp_scaled_dim=2``.
      ``o_proj`` -> ``tp_mode="rowwise"`` + ``all_reduce(layer_type="mla")``.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
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
        self.rope_theta = config.rope_theta

        # Q projection (with optional LoRA)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, config.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = DeepSeekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(
                config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
            )

        # KV projection with MQA
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepSeekV3RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Initialize rotary embedding
        self._init_rope()

        # Softmax scale
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            # transformers 5.x populates rope_scaling to {"rope_type": "default"} when the
            # checkpoint has no scaling (e.g. DeepSeek-V3-Lite), so "factor" may be absent.
            # Only apply the YaRN mscale correction when an explicit factor is present.
            scaling_factor = config.rope_scaling.get("factor")
            if scaling_factor is not None and mscale_all_dim:
                mscale = DeepSeekV3YarnRotaryEmbedding._yarn_get_mscale(
                    scaling_factor, mscale_all_dim
                )
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        rope_scaling = self.config.rope_scaling
        # In transformers 5.x rope_scaling is never None; treat "default"
        # rope_type the same as no scaling.
        scaling_type = None
        if rope_scaling is not None:
            scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        if scaling_type is None or scaling_type == "default":
            self.rotary_emb = DeepSeekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_factor = self.config.rope_scaling["factor"]

            if scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepSeekV3YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                # Default to base rotary embedding for unsupported types
                self.rotary_emb = DeepSeekV3RotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # KV projection FIRST (before Q) so the KV-cone precedes the Q-cone in
        # graph order. This is the invariant multi_stream_mla_attn pattern 2
        # relies on to overlap the (light) KV-cone on the aux stream with the
        # (heavy) q_b_proj on the main stream. Q and KV are independent (both
        # from hidden_states / the fused a-proj), so the reorder is numerically
        # identical. Keep compressed form; latent compression is replicated.
        kv_a_output = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.kv_a_proj_with_mqa.weight,
            self.kv_a_proj_with_mqa.bias,
            tp_mode="none",
            layer_type="mla",
        )
        compressed_kv, k_pe = torch.split(
            kv_a_output, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # Apply layernorm to compressed_kv
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        # k_pe: [B, S, 1, qk_rope_head_dim] (BSND layout, shared across heads)
        # dim 2 is fixed at 1 and never scales with TP, so plain `.view` is correct.
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)

        # Q projection: latent projections replicated, q_b_proj colwise.
        if self.q_lora_rank is None:
            q = torch.ops.auto_deploy.torch_linear_simple(
                hidden_states,
                self.q_proj.weight,
                self.q_proj.bias,
                tp_mode="colwise",
                layer_type="mla",
            )
        else:
            q = torch.ops.auto_deploy.torch_linear_simple(
                hidden_states,
                self.q_a_proj.weight,
                self.q_a_proj.bias,
                tp_mode="none",
                layer_type="mla",
            )
            q = self.q_a_layernorm(q)
            q = torch.ops.auto_deploy.torch_linear_simple(
                q,
                self.q_b_proj.weight,
                self.q_b_proj.bias,
                tp_mode="colwise",
                layer_type="mla",
            )

        # Shape: [B, S, N, q_head_dim] (BSND layout); num_heads (dim 2) scales with TP.
        q = torch.ops.auto_deploy.view(
            q,
            [bsz, q_len, self.num_heads, self.q_head_dim],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_seq_len = q_len

        cos, sin = self.rotary_emb(hidden_states, seq_len=kv_seq_len)
        cos = cos[position_ids]  # [B, S, head_dim]
        sin = sin[position_ids]  # [B, S, head_dim]

        # Use torch_rope_with_explicit_cos_sin: this is the op fuse_rope_into_trtllm_mla
        # matches and for which it builds the rotary_cos_sin table that pairs correctly
        # with the fused kernel's is_neox=True rotation.  (The eager rotation output is
        # discarded once RoPE is fused into trtllm_mla; only the op identity + cos/sin
        # args matter.  The interleaving-aware op mis-pairs with is_neox=True and costs
        # ~0.7 GSM8K, so do NOT use it here.)
        q_pe_rotated, kpe = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q_pe,
            k_pe,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND layout
        )

        # Call MLA with compressed KV. enable_sharding=True lets _apply_hint_mla
        # shard kv_b_proj_weight column-wise along the head dimension. Do NOT
        # decompose torch_mla into separate linears + torch_attention -- that
        # introduces concrete-shape view/expand that break under TP.
        attn_output = torch.ops.auto_deploy.torch_mla(
            q_nope,  # [B, S, N, qk_nope_head_dim]
            q_pe_rotated,  # [B, S, N, qk_rope_head_dim]
            compressed_kv,  # [B, S, kv_lora_rank]
            kpe,  # [B, S, 1, qk_rope_head_dim]
            self.kv_b_proj.weight,  # [N*(qk_nope+v), kv_lora_rank]
            True,  # is_causal
            self.softmax_scale,
            "bsnd",  # layout
            enable_sharding=True,
            layer_type="mla",
        )

        # Output: [B, S, N, v_head_dim] -> [B, S, N * v_head_dim].
        # Collapsed dim scales with TP via num_heads.
        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [bsz, q_len, self.num_heads * self.v_head_dim],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output,
            self.o_proj.weight,
            self.o_proj.bias,
            tp_mode="rowwise",
            layer_type="mla",
        )
        attn_output = torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mla")

        return attn_output


class DeepSeekV3DecoderLayer(nn.Module):
    """Transformer decoder layer for DeepSeekV3."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = DeepSeekV3Attention(config, layer_idx=layer_idx)

        # MLP or MoE
        # MoE layers are used after first_k_dense_replace and at moe_layer_freq intervals
        use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        if use_moe:
            self.mlp = DeepSeekV3MoE(config)
        else:
            self.mlp = DeepSeekV3MLP(config)

        # Layer norms
        self.input_layernorm = DeepSeekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepSeekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        # MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class DeepSeekV3Output(ModelOutput):
    """Output for DeepSeekV3Model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class DeepSeekV3CausalLMOutput(ModelOutput):
    """Output for DeepSeekV3ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class DeepSeekV3PreTrainedModel(PreTrainedModel):
    """Base class for DeepSeekV3 models."""

    base_model_prefix = "model"
    _no_split_modules = ["DeepSeekV3DecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class DeepSeekV3Model(DeepSeekV3PreTrainedModel):
    """DeepSeekV3 transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                DeepSeekV3DecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepSeekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> DeepSeekV3Output:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids)

        hidden_states = self.norm(hidden_states)

        return DeepSeekV3Output(last_hidden_state=hidden_states)


class DeepSeekV3ForCausalLM(DeepSeekV3PreTrainedModel, GenerationMixin):
    """DeepSeekV3 model with language modeling head."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepSeekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # No _rope_deinterleave_load_hook: weights stay in native GPTJ layout.
        # The fused trtllm_mla path applies the correct rotation directly, so the
        # load-time de-interleave + fusion-time undo round-trip is omitted.
        # fuse_rope_into_trtllm_mla skips the undo for this model_type
        # (mla_rope_utils.is_gptj_layout / _GPTJ_LAYOUT_MODEL_TYPES).

        # Dequantize kv_b_proj FP8 weights at load time.
        # kv_b_proj.weight is passed directly to torch_mla (not via a quantized linear op)
        # so it is NOT processed by the FineGrainedFP8 quantization transform.  Without
        # this hook the FP8 weight is stored into the BF16 model parameter via a raw dtype
        # cast that ignores weight_scale_inv, making the attention scores ~1000x too large
        # and producing NaN/Inf logits.
        self._register_load_state_dict_pre_hook(
            partial(
                mla_rope_utils._kv_b_proj_dequant_load_hook,
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

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> DeepSeekV3CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return DeepSeekV3CausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("DeepseekV3Config", DeepSeekV3ForCausalLM)
