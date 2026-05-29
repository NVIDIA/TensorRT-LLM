# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Slimmed-down PyTorch StepFun Step-3.7-Flash text model for AutoDeploy export (prefill only).

Source:
https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8 (and bf16 sibling Step-3.7-Flash)

Step-3.7-Flash is a vision-language model. This file ports ONLY the text decoder
(``Step3p5``-style ``model_type="step3p5"``); the vision tower is intentionally not
exported (AutoDeploy onboards the text generation path).

Key text-architecture features:
* Per-layer attention type: ``full_attention`` and ``sliding_attention`` alternate
  (1 full + 3 sliding per group). Full-attention layers use 64 Q heads; sliding-attention
  layers use 96 Q heads. Both use 8 KV heads (GQA) and head_dim=128.
* Head-wise attention gate (``g_proj``): the attention output of each head is multiplied by
  ``sigmoid(g_proj(hidden_states))`` before the output projection.
* Per-head QK RMSNorm over head_dim (Qwen3-style).
* Per-layer-type partial RoPE: full-attention layers rotate the first half of head_dim
  (partial_rotary_factor=0.5, rope_theta=5e6, llama3 rope-scaling); sliding-attention layers
  rotate the full head_dim (partial_rotary_factor=1.0, rope_theta=1e4, no scaling).
* Dense SwiGLU MLP on the first ``len(layers) - len(moe_layers)`` layers (layers 0-2);
  the remaining layers are MoE (288 routed experts, top-8, sigmoid routing with a per-expert
  bias used for *selection only*, fp32 gate, scaling 3.0) plus a dense shared expert.
* Gemma-style ``(1 + weight)`` RMSNorm convention for ALL norms (absorbed into the weight at
  load time via a pre-hook so the graph uses plain ``torch_rmsnorm``).

Differences from the HF reference (modeling_step3p7.py):
* Vision tower, multimodal merging, KV cache, training paths, dropout, and the MTP/spec
  ``mtp_block`` layers (45-47) are all removed — prefill text decode only.
* Uses AD canonical ops: torch_rmsnorm, torch_attention, torch_rope_with_explicit_cos_sin,
  torch_moe. No repeat_kv (torch_attention handles GQA natively).
* Stacked checkpoint MoE expert weights are split into per-expert Linear modules via a
  load-state-dict pre-hook for torch_moe dispatch.
* The SwiGLU activation clamp (``swiglu_limits``) present on routed experts of the last two
  MoE layers is NOT applied (the clamp limits are large numerical guards; see note in the
  MoE block). It is still applied on the dense shared-expert path where it is a plain MLP.

Tensor-parallel sharding (sharding-IR hints):
* Every shardable projection uses ``torch.ops.auto_deploy.torch_linear_simple`` with explicit
  ``tp_mode`` / ``layer_type`` hints, head reshapes use ``torch.ops.auto_deploy.view`` with
  ``tp_scaled_dim``, and rowwise outputs are followed by ``torch.ops.auto_deploy.all_reduce``.
  The exported graph fully specifies sharding; ``apply_sharding_hints`` applies it.
* MHA: q/k/v/g colwise (k/v use ``tp_min_local_shape=head_dim`` for GQA; the head-wise gate
  ``g_proj`` is a per-head column shard, ``tp_min_local_shape=1``), o_proj rowwise + all_reduce.
* MoE: routed experts via ``torch_moe(layer_type="moe")`` (EP/TP handled by the sharder); the
  shared expert is a colwise/rowwise MLP with no internal all_reduce — a single all_reduce at
  the ``routed + shared`` merge point covers both. Dense MLP layers reduce internally.
* The fp32 router gate is TP-replicated (kept as plain ``F.linear``).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ... import custom_ops  # noqa: F401 -- register all sharding-aware ops
from ..._compat import ActivationType
from ..hf import AutoModelForCausalLMFactory
from .rotary_utils import RotaryEmbeddingBase, build_rope_cos_sin_cache

# ---------------------------------------------------------------------------
# Bundled config
# ---------------------------------------------------------------------------


class Step3p7Config(PretrainedConfig):
    """Minimal flat text config for Step-3.7-Flash.

    Real deployments load the model's ``trust_remote_code`` config (the VLM wrapper
    ``Step3p7Config`` with a nested ``text_config``); AutoDeploy passes that object straight to
    ``_from_config`` and the model reads ``config.text_config`` via ``_get_text_config``. This
    bundled class is the resolvable config the model registers under, used for standalone
    construction and the offline sharding-IR equivalence harness (which builds a tiny instance and
    overrides the universal dims). Its defaults are intentionally small and tensor-parallel
    friendly so a 4-layer / 4-head tiny model shards cleanly; production values come from the
    checkpoint config.
    """

    model_type = "step3p5"

    def __init__(
        self,
        vocab_size: int = 128896,
        hidden_size: int = 64,
        head_dim: int = 16,
        num_attention_heads: int = 4,
        num_attention_groups: int = 4,
        attention_other_setting: Optional[dict] = None,
        intermediate_size: int = 64,
        num_hidden_layers: int = 4,
        layer_types: Optional[list] = None,
        moe_layers_enum: tuple = (2, 3),
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_intermediate_size: int = 16,
        share_expert_dim: int = 16,
        moe_router_scaling_factor: float = 3.0,
        rms_norm_eps: float = 1e-5,
        sliding_window: int = 4,
        max_position_embeddings: int = 256,
        rope_theta=(5e6, 1e4, 5e6, 1e4),
        partial_rotary_factors=(0.5, 1.0, 0.5, 1.0),
        rope_scaling: Optional[dict] = None,
        yarn_only_types=("full_attention",),
        swiglu_limits: Optional[list] = None,
        swiglu_limits_shared: Optional[list] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.attention_other_setting = attention_other_setting or {
            "num_attention_heads": num_attention_heads,
            "num_attention_groups": num_attention_groups,
            "head_dim": head_dim,
        }
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = layer_types or [
            "full_attention" if i % 2 == 0 else "sliding_attention"
            for i in range(num_hidden_layers)
        ]
        self.moe_layers_enum = moe_layers_enum
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = list(rope_theta)
        self.partial_rotary_factors = list(partial_rotary_factors)
        self.rope_scaling = rope_scaling or {
            "rope_type": "llama3",
            "factor": 2.0,
            "original_max_position_embeddings": max_position_embeddings,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
        }
        self.yarn_only_types = list(yarn_only_types)
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Step3p7ModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Step3p7CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Config access helper
# ---------------------------------------------------------------------------


def _get_text_config(config):
    """Return the text sub-config (Step-3.7 wraps the LLM in a VLM ``Step3p7Config``)."""
    return getattr(config, "text_config", config)


# ---------------------------------------------------------------------------
# Load-state-dict pre-hooks (run on the full ForCausalLM)
# ---------------------------------------------------------------------------


def _step3p7_norm_weight_load_hook(state_dict, prefix, *args, **kwargs):
    """Absorb Step's ``(1 + weight)`` RMSNorm convention into the weight at load time.

    HF Step stores all norm weights as a bias around zero and applies ``(1 + weight)``.
    Adding 1.0 here lets the forward use the standard ``torch_rmsnorm(x, weight, eps)``
    without an extra add node in the exported graph (matches the Gemma onboarding pattern).
    """
    for key in list(state_dict.keys()):
        if key.endswith("layernorm.weight") or key.endswith("norm.weight"):
            state_dict[key] = state_dict[key] + 1.0


def _step3p7_moe_split_load_hook(state_dict, prefix, *args, **kwargs):
    """Split stacked routed-expert weights into per-expert Linear weights.

    The checkpoint stores routed experts as stacked tensors per projection:
      * ``...moe.gate_proj.weight``  [E, moe_intermediate, hidden]
      * ``...moe.up_proj.weight``    [E, moe_intermediate, hidden]
      * ``...moe.down_proj.weight``  [E, hidden, moe_intermediate]
    The custom model keeps per-expert ``nn.Linear`` modules for ``torch_moe`` dispatch, i.e.
      * ``...moe.experts.{e}.{gate,up,down}_proj.weight``
    """
    for key in list(state_dict.keys()):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            suffix = f".moe.{proj}.weight"
            if key.endswith(suffix) and state_dict[key].dim() == 3:
                stacked = state_dict.pop(key)
                base = key[: -len(suffix)]
                for e in range(stacked.shape[0]):
                    state_dict[f"{base}.moe.experts.{e}.{proj}.weight"] = stacked[e]
                break


# ---------------------------------------------------------------------------
# RMSNorm (using AD canonical op; (1 + weight) absorbed at load time)
# ---------------------------------------------------------------------------


class Step3p7RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.variance_epsilon)


# ---------------------------------------------------------------------------
# Rotary Embedding (per-layer-type: partial rotation + optional llama3 scaling)
# ---------------------------------------------------------------------------


def _compute_step3p7_inv_freq(
    head_dim: int,
    partial_rotary_factor: float,
    base: float,
    rope_scaling: Optional[dict],
) -> torch.Tensor:
    """Inverse frequencies for Step RoPE (default or llama3-scaled)."""
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if not rope_scaling:
        return inv_freq

    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    assert rope_type == "llama3", f"Step-3.7 only supports llama3 rope-scaling, got {rope_type!r}"

    # Faithful copy of transformers _compute_llama3_parameters scaling math.
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


class Step3p7RotaryEmbedding(RotaryEmbeddingBase):
    """RoPE table builder for one attention type (partial rotation, optional llama3 scaling).

    Keeps only the small ``inv_freq`` buffer; the cos/sin tables are graph-computed in forward
    (so AD's ``optimize_rope`` can materialize a fused cache). For llama3 the attention scaling
    factor is 1.0, so cos/sin are not rescaled.
    """

    def __init__(
        self,
        head_dim: int,
        partial_rotary_factor: float,
        base: float,
        max_position_embeddings: int,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        inv_freq = _compute_step3p7_inv_freq(head_dim, partial_rotary_factor, base, rope_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = build_rope_cos_sin_cache(self.inv_freq, self.max_position_embeddings, x)
        return cos[position_ids], sin[position_ids]


def _apply_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to the first ``rotary_dim`` dims (bsnd layout), passing the rest through.

    ``rotary_dim = cos.shape[-1]`` may be < head_dim (full-attention layers) or == head_dim
    (sliding-attention layers, in which case there is nothing to pass through).
    """
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot, k_rot = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
        q_rot,
        k_rot,
        cos,
        sin,
        2,  # unsqueeze_dim=2 for bsnd
    )
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


# ---------------------------------------------------------------------------
# Dense SwiGLU MLP (dense layers + shared expert)
# ---------------------------------------------------------------------------


class Step3p7MLP(nn.Module):
    """SwiGLU MLP with an optional post-activation clamp (Step ``swiglu_limit``).

    Sharding: gate/up colwise, down rowwise. ``apply_all_reduce`` controls whether the rowwise
    output is reduced here (True for a standalone dense MLP) or left partial for a downstream
    merge-point all_reduce (False when used as a MoE shared expert).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: Optional[float] = None,
        layer_type: str = "mlp",
        apply_all_reduce: bool = True,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit
        self.layer_type = layer_type
        self.apply_all_reduce = apply_all_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.ops.auto_deploy.torch_linear_simple(
            x, self.gate_proj.weight, None, tp_mode="colwise", layer_type=self.layer_type
        )
        up = torch.ops.auto_deploy.torch_linear_simple(
            x, self.up_proj.weight, None, tp_mode="colwise", layer_type=self.layer_type
        )
        gate = self.act_fn(gate)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        down = torch.ops.auto_deploy.torch_linear_simple(
            gate * up, self.down_proj.weight, None, tp_mode="rowwise", layer_type=self.layer_type
        )
        if self.apply_all_reduce:
            down = torch.ops.auto_deploy.all_reduce(down, layer_type=self.layer_type)
        return down


# ---------------------------------------------------------------------------
# Sparse MoE block (routed experts + shared expert)
# ---------------------------------------------------------------------------


class Step3p7MoE(nn.Module):
    """Routed MoE with sigmoid routing + per-expert bias (selection only).

    The dense shared expert is a sibling of this module on the decoder layer (matching the HF
    hierarchy and the checkpoint layout ``model.layers.N.share_expert.*``), so it is NOT part of
    this module.

    Routing (HF ``router_bias_func``):
      1. ``probs = sigmoid(fp32 router logits)``
      2. select top-k experts by ``probs + router_bias``
      3. gather the *un-biased* ``probs`` for the selected experts
      4. renormalize the gathered weights and scale by ``moe_router_scaling_factor``

    NOTE on ``swiglu_limit``: the routed experts of the last two MoE layers carry a SwiGLU
    activation clamp in the HF reference. ``torch_moe`` has no clamp parameter, so the routed
    clamp is not applied here (the limits are large guards that rarely activate). The clamp IS
    applied on the dense shared-expert path (a plain MLP, see ``Step3p7DecoderLayer``).
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = getattr(config, "moe_router_scaling_factor", 1.0)

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.register_buffer(
            "router_bias", torch.zeros(self.num_experts, dtype=torch.float32), persistent=True
        )

        self.experts = nn.ModuleList(
            [
                Step3p7MLP(self.hidden_size, config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # fp32 router GEMM (config.need_fp32_gate)
        router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())
        probs = torch.sigmoid(router_logits)

        scores = probs + self.router_bias.unsqueeze(0)
        _, selected_experts = torch.topk(scores, self.top_k, dim=-1)
        routing_weights = torch.gather(probs, 1, selected_experts)
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.routed_scaling_factor
        routing_weights = routing_weights.to(hidden_flat.dtype)

        routed = torch.ops.auto_deploy.torch_moe(
            hidden_flat,
            selected_experts,
            routing_weights,
            w1_weight=[e.gate_proj.weight for e in self.experts],
            w2_weight=[e.down_proj.weight for e in self.experts],
            w3_weight=[e.up_proj.weight for e in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
            layer_type="moe",
        )
        return routed.view(bsz, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Attention (GQA + per-head QK norm + head-wise gate + partial RoPE)
# ---------------------------------------------------------------------------


class Step3p7Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.attention_type = config.layer_types[layer_idx]
        is_sliding = self.attention_type == "sliding_attention"

        if is_sliding:
            other = config.attention_other_setting
            self.num_heads = other["num_attention_heads"]
            self.num_kv_heads = other["num_attention_groups"]
            self.sliding_window = config.sliding_window
        else:
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_attention_groups
            self.sliding_window = None

        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        # Head-wise attention gate: one sigmoid scalar per head, applied to the attention output
        # before o_proj. Sharded as a per-head column shard (tp_min_local_shape=1) so it follows
        # the same head partition as q/k/v under tensor parallelism.
        self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)

        # Per-head QK RMSNorm over head_dim.
        self.q_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.q_proj.weight,
            None,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        k = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.k_proj.weight,
            None,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        v = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.v_proj.weight,
            None,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        q = torch.ops.auto_deploy.view(
            q, [bsz, q_len, self.num_heads, self.head_dim], tp_scaled_dim=2, layer_type="mha"
        )
        k = torch.ops.auto_deploy.view(
            k, [bsz, q_len, self.num_kv_heads, self.head_dim], tp_scaled_dim=2, layer_type="mha"
        )
        v = torch.ops.auto_deploy.view(
            v, [bsz, q_len, self.num_kv_heads, self.head_dim], tp_scaled_dim=2, layer_type="mha"
        )

        # Per-head QK norm over head_dim.
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = _apply_partial_rope(q, k, cos, sin)

        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,  # scale
            None,  # sinks
            self.sliding_window,  # sliding_window
            None,  # logit_cap
            "bsnd",  # layout
        )  # [B, S, N, head_dim]

        # Head-wise gate: scale each head's output by sigmoid(per-head gate). g_proj is a per-head
        # column shard (tp_min_local_shape=1), so its [B, S, N] output is sharded over the same
        # head partition as the attention output.
        gate = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.g_proj.weight,
            None,
            tp_mode="colwise",
            tp_min_local_shape=1,
            layer_type="mha",
        ).sigmoid()  # [B, S, N]
        attn_output = attn_output * gate.unsqueeze(-1)

        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [bsz, q_len, self.num_heads * self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output, self.o_proj.weight, None, tp_mode="rowwise", layer_type="mha"
        )
        attn_output = torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mha")
        return attn_output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Step3p7DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, is_moe_layer: bool):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Step3p7Attention(config, layer_idx)

        _, shared_swiglu_limit = _layer_swiglu_limits(config, layer_idx)
        self.is_moe_layer = is_moe_layer
        if is_moe_layer:
            self.moe = Step3p7MoE(config)
            # Shared expert is a sibling of ``moe`` (checkpoint key model.layers.N.share_expert.*).
            # No internal all_reduce: the single merge-point all_reduce (routed + shared) reduces it.
            self.share_expert = Step3p7MLP(
                config.hidden_size,
                config.share_expert_dim,
                swiglu_limit=shared_swiglu_limit,
                layer_type="moe",
                apply_all_reduce=False,
            )
        else:
            self.mlp = Step3p7MLP(
                config.hidden_size, config.intermediate_size, swiglu_limit=shared_swiglu_limit
            )

        self.input_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        sliding_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        position_embeddings = (
            sliding_position_embeddings
            if self.attention_type == "sliding_attention"
            else full_position_embeddings
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe_layer:
            # Single all_reduce at the routed + shared merge point (both are left partial above).
            hidden_states = self.moe(hidden_states) + self.share_expert(hidden_states)
            hidden_states = torch.ops.auto_deploy.all_reduce(hidden_states, layer_type="moe")
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def _moe_layer_indices(config) -> List[int]:
    moe_layers_enum = getattr(config, "moe_layers_enum", None)
    if moe_layers_enum is None:
        return list(range(1, config.num_hidden_layers))
    if isinstance(moe_layers_enum, str):
        return [int(i) for i in moe_layers_enum.split(",") if i.strip()]
    return [int(i) for i in moe_layers_enum]


def _layer_swiglu_limits(config, layer_idx: int) -> Tuple[Optional[float], Optional[float]]:
    """Return (routed-expert limit, shared/dense limit) for a layer, or None when disabled."""

    def _val(values):
        if not values or layer_idx >= len(values):
            return None
        v = values[layer_idx]
        return float(v) if v else None

    return _val(getattr(config, "swiglu_limits", None)), _val(
        getattr(config, "swiglu_limits_shared", None)
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Step3p7PreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    _no_split_modules = ["Step3p7DecoderLayer"]
    supports_gradient_checkpointing = False


class Step3p7TextModel(Step3p7PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        text_config = _get_text_config(config)
        self.config = config

        moe_layers = set(_moe_layer_indices(text_config))
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Step3p7DecoderLayer(text_config, idx, is_moe_layer=idx in moe_layers)
                for idx in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = Step3p7RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

        self.full_rotary_emb, self.sliding_rotary_emb = _build_rotary_embeddings(text_config)

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
    ) -> Step3p7ModelOutput:
        assert position_ids is not None, "position_ids is required for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        full_pe = self.full_rotary_emb(inputs_embeds, position_ids)
        sliding_pe = self.sliding_rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, full_pe, sliding_pe)

        hidden_states = self.norm(hidden_states)
        return Step3p7ModelOutput(last_hidden_state=hidden_states)


def _build_rotary_embeddings(text_config):
    """Build the two RoPE tables (full-attention and sliding-attention) from per-layer config.

    ``rope_theta`` and ``partial_rotary_factors`` are per-layer lists in the checkpoint config,
    but they are constant within each attention type, so we read the value from a representative
    layer of each type. llama3 rope-scaling applies only to ``yarn_only_types`` (full attention).
    """
    layer_types = text_config.layer_types
    rope_theta = text_config.rope_theta
    partial_rotary_factors = getattr(text_config, "partial_rotary_factors", None)
    rope_scaling = getattr(text_config, "rope_scaling", None)
    yarn_only_types = getattr(text_config, "yarn_only_types", None)
    head_dim = text_config.head_dim
    max_pos = text_config.max_position_embeddings

    def _theta(idx):
        return rope_theta[idx] if isinstance(rope_theta, (list, tuple)) else rope_theta

    def _partial(idx):
        if partial_rotary_factors is not None:
            return partial_rotary_factors[idx]
        return getattr(text_config, "partial_rotary_factor", 1.0)

    def _scaling(layer_type):
        if rope_scaling is None:
            return None
        if yarn_only_types is not None and layer_type not in yarn_only_types:
            return None
        return rope_scaling

    def _rep_index(layer_type):
        return next(i for i, t in enumerate(layer_types) if t == layer_type)

    embeds = {}
    for layer_type in ("full_attention", "sliding_attention"):
        idx = _rep_index(layer_type)
        embeds[layer_type] = Step3p7RotaryEmbedding(
            head_dim=head_dim,
            partial_rotary_factor=_partial(idx),
            base=_theta(idx),
            max_position_embeddings=max_pos,
            rope_scaling=_scaling(layer_type),
        )
    return embeds["full_attention"], embeds["sliding_attention"]


class Step3p7ForCausalLM(Step3p7PreTrainedModel, GenerationMixin):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        text_config = _get_text_config(config)
        self.model = Step3p7TextModel(config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        # Load-time checkpoint adapters: absorb (1 + weight) RMSNorm convention and split stacked
        # MoE expert weights into per-expert Linear modules.
        self._register_load_state_dict_pre_hook(_step3p7_norm_weight_load_hook)
        self._register_load_state_dict_pre_hook(_step3p7_moe_split_load_hook)

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
    ) -> Step3p7CausalLMOutput:
        assert position_ids is not None, "position_ids is required for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return Step3p7CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AutoModelForCausalLMFactory.register_custom_model_cls("Step3p7Config", Step3p7ForCausalLM)
