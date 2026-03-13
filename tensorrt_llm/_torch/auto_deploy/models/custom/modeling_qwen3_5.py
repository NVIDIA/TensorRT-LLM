# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Qwen3.5 dense model for auto_deploy (text + vision).

Reference HF modeling file (not yet in a released transformers version):
  transformers/models/qwen3_next/modeling_qwen3_next.py
  (Qwen3.5 dense is architecturally a Qwen3Next without MoE)

This implementation differs from the HuggingFace original in the following ways:
  * External kernel dependencies (flash-linear-attention, causal_conv1d) are replaced with
    autodeploy custom ops.
  * Cache-related code paths have been removed (prefill-only).
  * Training-related code paths have been removed.
  * Unnecessary output fields have been removed.
  * The GatedDeltaNet forward uses autodeploy custom ops:
    torch_causal_conv1d, torch_l2norm, torch_gated_delta_rule.
  * mRoPE cos/sin can be computed outside the export boundary and
    passed in as ``position_embeddings`` for multimodal inputs.

This allows us to have a "pytorch" native reference implementation decoupled from bugs and
dependency issues in the source, while remaining weight-compatible with HF checkpoints.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

# =============================================================================
# Configuration
# =============================================================================


class Qwen3_5TextConfig(PretrainedConfig):
    """Minimal config class for Qwen3.5 dense text model.

    Bundled here because ``qwen3_5`` / ``qwen3_5_text`` are not yet registered
    in the installed transformers version (requires transformers >= 4.58).
    Mirrors the attributes of the upstream Qwen3_5TextConfig.  Only attributes
    needed by the slimmed-down prefill model are included.
    """

    model_type = "qwen3_5_text"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_act="silu",
        intermediate_size=3584,
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        attn_output_gate=True,
        # linear attention (GatedDeltaNet) params
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        # layer types
        layer_types=None,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.attn_output_gate = attn_output_gate

        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            }
        self.rope_parameters = rope_parameters

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        self.layer_types = layer_types
        if self.layer_types is None:
            # Default pattern: every 4th layer is full_attention, rest are linear_attention
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# =============================================================================
# Normalization
# =============================================================================


class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm using the ``torch_rmsnorm`` autodeploy custom op.

    The HF checkpoint stores weights in ``(1 + w)`` parameterisation (zeros
    init). A load-time pre-hook adds 1.0 so that the forward can use a plain
    ``weight * x`` multiply, which matches the autodeploy RMSNorm pattern and
    gets fused into a single ``flashinfer_rms_norm`` kernel.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self._register_load_state_dict_pre_hook(self._offset_weight)

    @staticmethod
    def _offset_weight(state_dict, prefix, *args):
        key = prefix + "weight"
        if key in state_dict:
            state_dict[key] = state_dict[key] + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


class Qwen3_5RMSNormGated(nn.Module):
    """Gated RMSNorm using the ``torch_rmsnorm_gated`` autodeploy custom op.

    Computes: norm(x) * weight * silu(gate).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm_gated(
            hidden_states,
            self.weight,
            gate,
            self.variance_epsilon,
            group_size=self.hidden_size,
            norm_before_gate=True,
        )


# =============================================================================
# Rotary Position Embedding (mRoPE)
# =============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply partial RoPE to query and key tensors.

    Supports partial rotary where only the first `rotary_dim` dimensions are rotated.
    Default unsqueeze_dim=2 is for bsnd layout (B, S, N, D).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class Qwen3_5TextRotaryEmbedding(nn.Module):
    """Simplified mRoPE for text-only prefill. Supports only the "default" rope type.

    Note: No AD rope op is used here because the existing ``torch_rope_*`` ops do not
    support interleaved mRoPE with 3D position_ids (shape ``(3, B, S)``).  Plain PyTorch
    is used instead; AD fusion transforms handle replacement at deployment time.
    """

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        rope_params = config.rope_parameters
        base = rope_params["rope_theta"]
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self.mrope_interleaved = rope_params.get("mrope_interleaved", True)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings.

        Args:
            x: Hidden states tensor, used only for dtype/device.
            position_ids: Shape (3, B, S) for mRoPE or (B, S) for plain.
        """
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # inv_freq: (dim/2,) -> (3, B, dim/2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        # position_ids: (3, B, S) -> (3, B, 1, S)
        position_ids_expanded = position_ids[:, :, None, :].float()

        # freqs: (3, B, S, dim/2)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)

        if self.mrope_interleaved:
            freqs = self._apply_interleaved_mrope(freqs)
        else:
            freqs = self._apply_concat_mrope(freqs)

        # Double for cos/sin: (B, S, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Apply interleaved mRoPE. Merges T/H/W frequency channels into one tensor."""
        freqs_t = freqs[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim_idx, ..., idx]
        return freqs_t

    def _apply_concat_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """Apply concatenated mRoPE (non-interleaved). Splits by section sizes."""
        sections = self.mrope_section
        result = torch.cat(
            [freqs[i, ..., : sections[i]] for i in range(3)],
            dim=-1,
        )
        return result


# =============================================================================
# GatedDeltaNet (Linear Attention)
# =============================================================================


class Qwen3_5GatedDeltaNet(nn.Module):
    """Prefill-only GatedDeltaNet using autodeploy custom ops."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        # QKV convolution
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # dt_bias and A_log for gated delta rule
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Gated RMSNorm (per head_v_dim)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        # Projections — split layout matches the Qwen3.5 checkpoint directly
        # (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a).
        # Note: HF Qwen3Next uses fused projections (in_proj_qkvz, in_proj_ba) which
        # differ from the Qwen3.5 checkpoint format.  No load hook is needed here since
        # the weight names already match the checkpoint.
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Projections (separate, matching checkpoint)
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, S, conv_dim]
        z = self.in_proj_z(hidden_states)  # [B, S, value_dim]
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)  # [B, S, num_v_heads, head_v_dim]
        b = self.in_proj_b(hidden_states)  # [B, S, num_v_heads]
        a = self.in_proj_a(hidden_states)  # [B, S, num_v_heads]

        # 2. Causal Conv1d via autodeploy op
        mixed_qkv = torch.ops.auto_deploy.torch_causal_conv1d(
            mixed_qkv,
            self.conv1d.weight,
            self.conv1d.bias,
            self.conv1d.stride[0],
            self.conv1d.padding[0],
            self.conv1d.dilation[0],
            self.conv1d.groups,
            self.conv1d.padding_mode,
        )
        mixed_qkv = F.silu(mixed_qkv)

        # Split back into Q, K, V
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        # Reshape to per-head: [B, S, num_heads, head_dim]
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # 3. L2 normalize Q and K via autodeploy op
        query = torch.ops.auto_deploy.torch_l2norm(query)
        key = torch.ops.auto_deploy.torch_l2norm(key)

        # 4. Compute beta and gating
        beta = b.sigmoid()  # [B, S, num_v_heads]
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # [B, S, num_v_heads]

        # GQA head expansion: expand Q/K to match V's head count before GDN op.
        # NOTE: The GDN uncached op handles GQA internally, but we also expand here
        # so that the delta sharding transform sees uniform head counts and applies
        # the correct sharding. Without this, TP sharding produces garbled output.
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # 5. Gated Delta Rule via autodeploy custom op
        # Op expects [B, S, H, D] layout (bsnd convention)
        core_attn_out = torch.ops.auto_deploy.torch_gated_delta_rule(query, key, value, g, beta)

        # 6. Gated RMSNorm
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # 7. Output projection
        output = self.out_proj(core_attn_out)
        return output


# =============================================================================
# Attention
# =============================================================================


class Qwen3_5Attention(nn.Module):
    """Multi-headed attention with gating, Q/K norms, and partial RoPE.

    Key differences from standard attention:
      - q_proj outputs 2x (query + gate), gating applied to attention output.
      - q_norm / k_norm applied per-head before RoPE.
      - Partial RoPE: only first rotary_dim dimensions are rotated.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # q_proj outputs 2x for query + gate
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        # Per-head Q/K norms
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Q projection with gate: output shape (B, S, N, 2*D)
        qg = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(qg, 2, dim=-1)  # each (B, S, N, D)
        gate = gate.reshape(bsz, q_len, -1)  # (B, S, N*D)

        # K, V projections in bsnd layout
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        # Per-head Q/K norms (norm operates on last dim = head_dim)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Partial RoPE in bsnd layout (unsqueeze_dim=2 for the N dimension)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        # Attention via autodeploy op (bsnd layout)
        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            layout="bsnd",
        )
        attn_output = attn_output.view(bsz, q_len, -1)  # (B, S, N*D)

        # Gated output
        attn_output = attn_output * torch.sigmoid(gate)

        # Output projection
        attn_output = self.o_proj(attn_output)
        return attn_output


# =============================================================================
# MLP
# =============================================================================


class Qwen3_5MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# =============================================================================
# Decoder Layer
# =============================================================================


class Qwen3_5DecoderLayer(nn.Module):
    """Single decoder layer: token mixer (linear_attention or full_attention) + MLP."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Token mixer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)

        hidden_states = residual + hidden_states

        # Channel mixer (dense MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Model
# =============================================================================


class Qwen3_5PreTrainedModel(PreTrainedModel):
    """Base class for Qwen3.5 pretrained models."""

    config_class = Qwen3_5TextConfig
    base_model_prefix = "model"
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Qwen3_5RMSNormGated):
            module.weight.data.fill_(1.0)
        elif isinstance(module, Qwen3_5GatedDeltaNet):
            module.dt_bias.data.fill_(1.0)
            module.A_log.data.copy_(torch.empty_like(module.A_log).uniform_(0, 16).log_())


@dataclass
class Qwen3_5Output(ModelOutput):
    """Output of the Qwen3.5 text model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Qwen3_5CausalLMOutput(ModelOutput):
    """Output of the Qwen3.5 causal language model."""

    logits: Optional[torch.FloatTensor] = None


class Qwen3_5TextModel(Qwen3_5PreTrainedModel):
    """Qwen3.5 text model (embed + decoder layers + final norm)."""

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        pad_token_id = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5Output]:
        """Forward pass.

        There are three ways to provide position information (checked in order):

        1. ``rope_cos`` + ``rope_sin``: separate tensors, each ``(B, S, rotary_dim)``.
           Export-friendly -- each is a proper graph input with its own dynamic shape.
        2. ``position_embeddings``: pre-computed ``(cos, sin)`` tuple. Convenient for
           the multimodal wrapper calling at the plain-PyTorch level.
        3. ``position_ids`` (or ``None``): standard 2D ``(B, S)`` or 3D ``(3, B, S)``
           position IDs.  The internal ``rotary_emb`` computes cos/sin.  This is the
           default text-only path.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Resolve position embeddings from one of the three input modes.
        if rope_cos is not None and rope_sin is not None:
            position_embeddings = (rope_cos, rope_sin)
        elif position_embeddings is None:
            if position_ids is None:
                raise ValueError(
                    "position_ids must be provided when position_embeddings and "
                    "rope_cos/rope_sin are not given."
                )
            if position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings=position_embeddings)

        hidden_states = self.norm(hidden_states)
        return Qwen3_5Output(last_hidden_state=hidden_states)


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    """Qwen3.5 causal language model (text model + lm_head)."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3_5TextConfig, **kwargs):
        super().__init__(config)
        self.model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.model.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3_5CausalLMOutput]:
        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return Qwen3_5CausalLMOutput(logits=logits)


# =============================================================================
# Vision Configuration
# =============================================================================


class Qwen3_5VisionConfig(PretrainedConfig):
    """Config class for the Qwen3.5 vision tower.

    Bundled here because ``qwen3_5`` is not yet registered in the installed
    transformers version (requires transformers >= 4.58).
    """

    model_type = "qwen3_5_vision"

    def __init__(
        self,
        depth: int = 12,
        hidden_size: int = 768,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 3072,
        num_heads: int = 12,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 1024,
        num_position_embeddings: int = 2304,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen3_5Config(PretrainedConfig):
    """Composite config containing both text and vision configs."""

    model_type = "qwen3_5"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = Qwen3_5VisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen3_5VisionConfig()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = Qwen3_5TextConfig(**text_config)
        elif text_config is None:
            self.text_config = Qwen3_5TextConfig()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# =============================================================================
# Vision Tower Components (plain PyTorch -- NOT exported)
# =============================================================================


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    """Simple rotary embedding for the vision tower (not mRoPE)."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to vision Q/K tensors. Layout: (seq, heads, dim)."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3_5VisionPatchEmbed(nn.Module):
    """3D convolution patch embedding for images/videos."""

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3_5VisionMLP(nn.Module):
    """Feed-forward network for vision blocks."""

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3_5VisionAttention(nn.Module):
    """Bidirectional attention for vision tokens with cu_seqlens support.

    Always non-causal (is_causal=False).
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Eager path: process each variable-length sequence separately
        query_states = query_states.transpose(0, 1).unsqueeze(0)  # (1, H, S, D)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_splits = torch.split(query_states, lengths, dim=2)
        k_splits = torch.split(key_states, lengths, dim=2)
        v_splits = torch.split(value_states, lengths, dim=2)

        attn_outputs = []
        for q, k, v in zip(q_splits, k_splits, v_splits):
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_outputs.append(torch.matmul(attn_weights, v))

        attn_output = torch.cat(attn_outputs, dim=2)  # (1, H, total_S, D)
        attn_output = attn_output.squeeze(0).transpose(0, 1)  # (S, H, D)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen3_5VisionBlock(nn.Module):
    """Vision transformer block."""

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VisionAttention(config=config)
        self.mlp = Qwen3_5VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionPatchMerger(nn.Module):
    """Merges spatial_merge_size^2 patches into one token and projects to LLM hidden size."""

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3_5VisionModel(nn.Module):
    """Complete vision tower: PatchEmbed + PositionEmbed + VisionBlocks + PatchMerger.

    This module is NOT exported -- it runs in plain PyTorch.
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.dtype = None  # set after loading weights

        self.patch_embed = Qwen3_5VisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3_5VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3_5VisionPatchMerger(config=config)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute rotary position embeddings for vision tokens."""
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of learned positional embeddings for variable image sizes."""
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list: List[List[int]] = [[] for _ in range(4)]
        weight_list: List[List[float]] = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(h.item()))
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, int(w.item()))

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws)]
        )

        merge_size = self.config.spatial_merge_size
        patch_pos_embeds_permute = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t.item()), int(h.item()), int(w.item())
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        return torch.cat(patch_pos_embeds_permute)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> BaseModelOutputWithPooling:
        """Run the vision tower."""
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        merged_hidden_states = self.merger(hidden_states)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
        )


# =============================================================================
# Multimodal Wrapper (plain PyTorch -- NOT exported)
# =============================================================================


@dataclass
class Qwen3_5ConditionalOutput(ModelOutput):
    """Output of the Qwen3.5 conditional generation model."""

    logits: Optional[torch.FloatTensor] = None


class Qwen3_5Model(nn.Module):
    """Multimodal wrapper: vision tower + embedding merge + mRoPE + language model.

    This module is NOT exported. It orchestrates the vision pipeline in plain
    PyTorch and calls the (potentially exported) language model with pre-computed
    ``(cos, sin)`` position embeddings.
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)

        # mRoPE embedding -- computed in this wrapper, passed to language model
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config.text_config)

        # Cache rope_deltas for decode steps
        self.rope_deltas: Optional[torch.Tensor] = None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 3D mRoPE position IDs for multimodal sequences."""
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)

            for i, ids in enumerate(total_input_ids):
                ids = ids[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(ids == vision_start_token_id).squeeze(1)
                vision_tokens = ids[vision_start_indices + 1]
                image_nums = int((vision_tokens == image_token_id).sum().item())
                video_nums = int((vision_tokens == video_token_id).sum().item())
                input_tokens = ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums

                for _ in range(image_nums + video_nums):
                    ed_image = (
                        input_tokens.index(image_token_id, st)
                        if image_token_id in input_tokens and remain_images > 0
                        else len(input_tokens) + 1
                    )
                    ed_video = (
                        input_tokens.index(video_token_id, st)
                        if video_token_id in input_tokens and remain_videos > 0
                        else len(input_tokens) + 1
                    )

                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index].tolist()
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index].tolist()
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video

                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t,
                        h // spatial_merge_size,
                        w // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1)
                        + st_idx
                    )

                    t_index = (
                        torch.arange(llm_grid_t, device=input_ids.device)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h, device=input_ids.device)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w, device=input_ids.device)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )

                    llm_pos_ids_list.append(
                        torch.stack([st_idx + t_index, st_idx + h_index, st_idx + w_index])
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1)
                        + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[:, i, attention_mask[i] == 1] = llm_positions.to(position_ids.dtype)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if input_ids is not None:
                batch_size, seq_length = input_ids.shape[:2]
                device = input_ids.device
            else:
                raise ValueError("input_ids must be provided")

            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - seq_length
            else:
                position_ids = (
                    torch.arange(seq_length, device=device).view(1, 1, -1).expand(3, batch_size, -1)
                )
                mrope_position_deltas = torch.zeros(batch_size, 1, device=device, dtype=torch.long)

            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen3_5Output:
        """Forward pass with optional vision inputs.

        For text-only inputs the runtime-provided ``position_ids`` are used
        directly (expanded to 3D for mRoPE).  For multimodal inputs,
        ``get_rope_index`` computes spatial (T, H, W) positions.
        """
        # Get input embeddings
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        # Process images if provided
        if pixel_values is not None:
            pixel_values = pixel_values.to(inputs_embeds.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
            image_mask = (
                (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            )
            image_embeds = image_embeds.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Process videos if provided
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(inputs_embeds.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output
            video_mask = (
                (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            )
            video_embeds = video_embeds.to(inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Determine position IDs — position_ids is REQUIRED from the runtime.
        # For text-only: runtime provides 2D position_ids (B, S) which we expand to 3D.
        # For multimodal: position_ids must incorporate the mRoPE offset from image tokens.
        #
        # TODO: Implement a self-contained mRoPE position delta cache transform that:
        #   1. Stores the per-sequence position delta (from image token offsets) in a state
        #      cache keyed by slot_idx during prefill
        #   2. Retrieves and applies the delta during decode so the runtime's sequential
        #      position_ids are correctly offset for mRoPE
        # See PR discussion: https://github.com/nv-auto-deploy/TensorRT-LLM/pull/189
        has_vision = pixel_values is not None or pixel_values_videos is not None
        if has_vision:
            # Multimodal: compute mRoPE positions with spatial (T, H, W) layout.
            # NOTE: This path needs the mRoPE position delta cache transform to work
            # correctly with the AD runtime during decode steps.
            assert False, (
                "Vision path not yet supported in AD runtime. Requires mRoPE position "
                "delta cache transform. See: "
                "https://github.com/nv-auto-deploy/TensorRT-LLM/pull/189#discussion_r2915084063"
            )
            position_ids, _ = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        elif position_ids is not None:
            # Text-only with runtime-provided position_ids: expand 2D -> 3D for mRoPE
            if position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        else:
            raise ValueError(
                "position_ids must be provided by the runtime. The Qwen3.5 multimodal "
                "wrapper does not support computing position_ids from scratch — use the "
                "text model (Qwen3_5ForCausalLM) directly for standalone inference, or "
                "ensure the AD runtime provides position_ids."
            )

        # Compute position embeddings outside the language model
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # Forward through language model with pre-computed position embeddings
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_embeddings=position_embeddings,
        )

        return outputs


class Qwen3_5ForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin):
    """Multimodal conditional generation model wrapping text + vision.

    The ``model`` attribute is a ``Qwen3_5Model`` (vision + language)
    at the top level -- matching the HF checkpoint weight layout.
    """

    config_class = Qwen3_5Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3_5Config, **kwargs):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen3_5ConditionalOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return Qwen3_5ConditionalOutput(logits=logits)


# =============================================================================
# Registration
# =============================================================================

AutoConfig.register("qwen3_5", Qwen3_5Config, exist_ok=True)
AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig, exist_ok=True)

AutoModelForCausalLMFactory.register_custom_model_cls("Qwen3_5TextConfig", Qwen3_5ForCausalLM)
AutoModelForCausalLMFactory.register_custom_model_cls(
    "Qwen3_5Config", Qwen3_5ForConditionalGeneration
)
