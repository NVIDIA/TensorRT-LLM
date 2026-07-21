# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import TimestepEmbedding

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """
        Qwen3VLTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.dtype = dtype

    def post_load_weights(self):
        self.weight.data = self.weight.data.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states.to(input_dtype)
        return output


@dataclass
class TransformerOutput:
    """Velocity predictions from Cosmos3VFMTransformer.forward()."""

    video: torch.Tensor
    """[B, C, T, H, W] video (or image when T=1) velocity prediction."""

    image: torch.Tensor
    """[B, C, 1, H, W] alias of video for image generation (same tensor)."""

    audio: Optional[torch.Tensor] = None
    """[B, audio_dim, T_audio] audio velocity prediction, or None."""

    action: Optional[torch.Tensor] = None
    """[B, T_action, action_dim] action velocity prediction, or None."""


def compute_mrope_position_ids_text(
    num_tokens: int,
    temporal_offset: int,
) -> tuple[torch.Tensor, int]:
    """Generate 3D mRoPE position IDs for text tokens.

    Text tokens: all three axes (T, H, W) share the same monotonically
    increasing position IDs: (0,0,0), (1,1,1), (2,2,2), ...

    Returns:
        (position_ids [3, num_tokens], next_temporal_offset)
    """
    ids = torch.arange(num_tokens, dtype=torch.long) + temporal_offset
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def compute_mrope_position_ids_vision(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    enable_fps_modulation: bool = False,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for vision tokens.

    Creates a (T, H, W) position grid. Spatial indices reset to 0
    per vision segment (Qwen3VL-style, reset_spatial_indices=True).
    Flattened in T-major order.

    When ``enable_fps_modulation`` is ``True``, temporal positions are scaled
    to reflect real time so that videos at different frame rates get comparable
    temporal embeddings.

    Returns:
        (position_ids [3, grid_t * grid_h * grid_w], next_temporal_offset)
    """
    if enable_fps_modulation and fps is not None:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / temporal_compression_factor
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        t_index = (
            (frame_indices / tps * base_tps + temporal_offset)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
        )
    else:
        t_index = torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(
            -1, grid_h * grid_w
        ).flatten() + int(temporal_offset)

    h_index = (
        torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    )
    w_index = (
        torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()
    )

    if enable_fps_modulation:
        mrope_ids = torch.stack(
            [t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0
        )
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_offset = math.ceil(mrope_ids.max().item()) + 1
    return mrope_ids, next_offset


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        frequency_embedding_size=256,
        max_period=10000,
        target_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.mlp = TimestepEmbedding(
            in_channels=frequency_embedding_size, time_embed_dim=hidden_size, act_fn="silu"
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size

        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=target_dtype) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def _init_weights(self):
        std = 1.0 / math.sqrt(self.frequency_embedding_size)
        torch.nn.init.trunc_normal_(self.mlp.linear_1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.mlp.linear_2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, t):
        # use .float() here if acc loss
        args = t[:, None] * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


def qwen3_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen3/Llama-style rotate_half: split first/second half of head_dim."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qwen3_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-style RoPE: (x * cos) + (rotate_half(x) * sin).

    Args:
        q: [B, S, H, D]
        k: [B, S, H_kv, D]
        cos: [1, S, 1, D] or broadcastable
        sin: [1, S, 1, D] or broadcastable
    """
    q_embed = (q * cos) + (qwen3_rotate_half(q) * sin)
    k_embed = (k * cos) + (qwen3_rotate_half(k) * sin)
    return q_embed, k_embed


class Cosmos3CausalAttention(Attention):
    """Understanding pathway: causal self-attention on text tokens.

    Inherits from Attention for projections (SEPARATE_QKV), per-head QK norms,
    and backend. Overrides forward to:
    - Reshape to 4D before QK norm (per-head)
    - Apply Qwen3-style RoPE (rotate_half, not interleaved)
    - Pass causal mask to backend
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        model_config: DiffusionModelConfig,
        layer_idx: int = 0,
        module_name: Optional[str] = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            qkv_mode=QKVMode.SEPARATE_QKV,
            qk_norm=False,
            qk_norm_mode="per_head",
            bias=False,
            config=model_config,
            layer_idx=layer_idx,
            module_name=module_name,
            enable_sequence_parallel=False,
        )
        self.norm_q = Qwen3VLTextRMSNorm(hidden_size=head_dim, dtype=torch.bfloat16)
        self.norm_k = Qwen3VLTextRMSNorm(hidden_size=head_dim, dtype=torch.bfloat16)

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head RMSNorm on 4D tensors [B, S, H, D]."""
        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)
        return q, k

    def forward_with_kv(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        timestep=None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]

        q, k, v = self.get_qkv(hidden_states)

        q = q.view(batch_size, seq_len, self.local_num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.local_num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.local_num_key_value_heads, self.head_dim)

        q, k = self.apply_qk_norm(q, k)
        q, k = qwen3_apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        out = self._attn_impl(
            q,
            k,
            v,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            timestep=timestep,
        )

        return self.to_out[0](out), k, v

    def forward(self):
        raise NotImplementedError(
            "forward method not implemented for Cosmos3CausalAttention. Use forward_with_kv instead."
        )


class Cosmos3CrossAttention(Attention):
    """Generation pathway: full attention where visual Q attends to all K/V.

    Inherits from Attention for gen-pathway projections, per-head QK norms,
    and backend. Overrides forward to:
    - Accept pre-computed und K/V for concatenation
    - Reshape to 4D before QK norm (per-head)
    - Apply Qwen3-style RoPE
    - Full (non-causal) attention with Q_gen attending to [K_und, K_gen]
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        model_config: DiffusionModelConfig,
        layer_idx: int = 0,
        module_name: Optional[str] = None,
    ):
        original_backend = model_config.attention.backend
        if model_config.attention.backend == "TRTLLM":
            # TRTLLM backend is not supported for Cosmos3CrossAttention
            model_config.attention.backend = "VANILLA"

        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            qkv_mode=QKVMode.FUSE_QKV,
            qk_norm=False,
            qk_norm_mode="per_head",
            bias=False,
            config=model_config,
            layer_idx=layer_idx,
            module_name=module_name,
            enable_sequence_parallel=True,
        )
        model_config.attention.backend = original_backend

        self.norm_q = Qwen3VLTextRMSNorm(hidden_size=head_dim, dtype=torch.bfloat16)
        self.norm_k = Qwen3VLTextRMSNorm(hidden_size=head_dim, dtype=torch.bfloat16)

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-head RMSNorm on 4D tensors [B, S, H, D]."""
        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)
        return q, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        timestep=None,
        real_text_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S_gen, hidden_size] visual tokens
            k_und: [B, S_und, H_kv, D] pre-computed und keys (post-norm, post-RoPE)
            v_und: [B, S_und, H_kv, D] pre-computed und values
            freqs_cos: [B, S_gen, 1, D] cosine part of RoPE
            freqs_sin: [B, S_gen, 1, D] sine part of RoPE

        Returns:
            [B, S_gen, hidden_size] cross-attention output
        """
        batch_size, seq_len_gen = hidden_states.shape[:2]

        q, k, v = self.get_qkv(hidden_states)

        q = q.view(batch_size, seq_len_gen, self.local_num_attention_heads, self.head_dim)
        k = k.view(batch_size, seq_len_gen, self.local_num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len_gen, self.local_num_key_value_heads, self.head_dim)

        q, k = self.apply_qk_norm(q, k)
        q, k = qwen3_apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        if real_text_lens is not None and batch_size > 1:
            outs = []
            for b in range(batch_size):
                Lb = int(real_text_lens[b])
                k_all_b = torch.cat([k_und[b : b + 1, :Lb], k[b : b + 1]], dim=1)
                v_all_b = torch.cat([v_und[b : b + 1, :Lb], v[b : b + 1]], dim=1)
                outs.append(
                    self._attn_impl(
                        q[b : b + 1],
                        k_all_b,
                        v_all_b,
                        attention_mask=PredefinedAttentionMask.FULL,
                        timestep=timestep,
                    )
                )
            out = torch.cat(outs, dim=0)
        else:
            k_all = torch.cat([k_und, k], dim=1).contiguous()
            v_all = torch.cat([v_und, v], dim=1).contiguous()

            out = self._attn_impl(
                q,
                k_all,
                v_all,
                attention_mask=PredefinedAttentionMask.FULL,
                timestep=timestep,
            )

        return self.to_out[0](out)


class Cosmos3UndDecoderLayer(nn.Module):
    """Understanding pathway decoder layer: causal self-attention + MLP."""

    def __init__(self, model_config: DiffusionModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = model_config.pretrained_config.hidden_size
        intermediate_size = model_config.pretrained_config.intermediate_size

        self.self_attn = Cosmos3CausalAttention(
            hidden_size=hidden_size,
            num_attention_heads=model_config.pretrained_config.num_attention_heads,
            num_key_value_heads=model_config.pretrained_config.num_key_value_heads,
            head_dim=model_config.pretrained_config.head_dim,
            model_config=model_config,
            layer_idx=layer_idx,
            module_name=f"layers.{layer_idx}.self_attn",
        )
        self.input_layernorm = Qwen3VLTextRMSNorm(
            hidden_size=hidden_size,
            eps=model_config.pretrained_config.rms_norm_eps,
            dtype=torch.bfloat16,
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            hidden_size=hidden_size,
            eps=model_config.pretrained_config.rms_norm_eps,
            dtype=torch.bfloat16,
        )
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            dtype=torch.bfloat16,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=model_config.mapping.tp_size > 1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor],
        timestep=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (hidden_states, K, V) where K/V are post-QKnorm, post-RoPE
            for consumption by the GEN cross-attention.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        cos, sin = freqs
        attn_out, k, v = self.self_attn.forward_with_kv(
            hidden_states,
            cos,
            sin,
            timestep=timestep,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        B, S, D = hidden_states.shape
        hidden_states = self.mlp(hidden_states.view(-1, D)).view(B, S, D)
        hidden_states = residual + hidden_states

        return hidden_states, k, v


class Cosmos3GenDecoderLayer(nn.Module):
    """Generation pathway decoder layer: cross-attention (to UND K/V) + MLP."""

    def __init__(self, model_config: DiffusionModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = model_config.pretrained_config.hidden_size
        intermediate_size = model_config.pretrained_config.intermediate_size

        self.cross_attention = Cosmos3CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=model_config.pretrained_config.num_attention_heads,
            num_key_value_heads=model_config.pretrained_config.num_key_value_heads,
            head_dim=model_config.pretrained_config.head_dim,
            model_config=model_config,
            layer_idx=layer_idx,
            module_name=f"layers.{layer_idx}.cross_attention",
        )
        self.input_layernorm = Qwen3VLTextRMSNorm(
            hidden_size=hidden_size,
            eps=model_config.pretrained_config.rms_norm_eps,
            dtype=torch.bfloat16,
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            hidden_size=hidden_size,
            eps=model_config.pretrained_config.rms_norm_eps,
            dtype=torch.bfloat16,
        )
        self.mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            dtype=torch.bfloat16,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=model_config.mapping.tp_size > 1,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor],
        timestep=None,
        real_text_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        cos, sin = freqs
        hidden_states = self.cross_attention(
            hidden_states,
            k_und=k_und,
            v_und=v_und,
            freqs_cos=cos,
            freqs_sin=sin,
            timestep=timestep,
            real_text_lens=real_text_lens,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        B, S, D = hidden_states.shape
        hidden_states = self.mlp(hidden_states.view(-1, D)).view(B, S, D)
        hidden_states = residual + hidden_states

        return hidden_states


def _compute_default_rope_parameters(
    model_config: DiffusionModelConfig,
) -> tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration. This function assumes that the config will provide at least the following
            properties:

            *   rope_theta (`float`): The base wavelength from which the inverse frequencies will be derived.
            *   hidden_size (`int`): The numerator when deriving a head_dim, if not provided directly.
            *   num_attention_heads (`int`): The denominator when deriving a head_dim, if not provided directly.

            Additionally, this function will make use of the following properties if they are found in the config:

            *   head_dim (`int`, *optional*): The size of the key-value heads in the model. If None, this value will be
                derived as hidden_size // num_attention_heads.
            *   partial_rotary_factor (`float`, *optional*): If less than 1.0, inverse frequencies will be returned for
                the first fraction of the head_dim. Defaults to 1.0.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.

    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = model_config.pretrained_config.rope_theta
    partial_rotary_factor = 1
    head_dim = model_config.pretrained_config.head_dim
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
    )
    return inv_freq, attention_factor


class Qwen3VLTextRotaryEmbedding(nn.Module):
    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__()
        self.rope_type = model_config.pretrained_config.rope_scaling["rope_type"]
        self.max_seq_len_cached = model_config.pretrained_config.max_position_embeddings
        self.original_max_seq_len = model_config.pretrained_config.max_position_embeddings

        self.mrope_section = model_config.pretrained_config.rope_scaling["mrope_section"]

        inv_freq, self.attention_scaling = _compute_default_rope_parameters(model_config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x, position_ids):
        assert self.inv_freq.dtype == torch.float32, (
            f"inv_freq must be float32, but got {self.inv_freq.dtype}"
        )

        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Cosmos3LanguageModel(nn.Module):
    """Understanding pathway: a standard causal LM that processes text tokens.

    Returns per-layer K/V tensors for the generation pathway's cross-attention.
    The UND pathway is independent of the denoising step, so its K/V can be
    computed once and reused across all sampling steps.
    """

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__()
        hidden_size = model_config.pretrained_config.hidden_size
        num_hidden_layers = model_config.pretrained_config.num_hidden_layers

        self.embed_tokens = Embedding(
            model_config.pretrained_config.vocab_size,
            hidden_size,
            dtype=torch.bfloat16,
            gather_output=True,
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(model_config)
        self.layers = nn.ModuleList(
            [Cosmos3UndDecoderLayer(model_config, layer_idx=i) for i in range(num_hidden_layers)]
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        freqs: Tuple[torch.Tensor, torch.Tensor],
        timestep=None,
    ) -> list[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            text_ids: [B, S] token IDs
            text_mask: [B, S] float mask (1=real, 0=pad)
            freqs: (cos, sin) each [B, S, 1, D] — precomputed UND RoPE

        Returns:
            List of (K, V) per layer. K/V are [B, S, H_kv, D], post-QKnorm
            and post-RoPE, ready for GEN cross-attention.
        """
        hidden = self.embed_tokens(text_ids)
        mask_3d = text_mask.unsqueeze(-1)  # [B, S, 1]

        cached_kv: list[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden = hidden * mask_3d
            hidden, k, v = layer(hidden, freqs, timestep=timestep)
            cached_kv.append((k, v))

        return cached_kv


class Cosmos3VFMTransformer(BaseDiffusionModel):
    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__(model_config)
        pretrained_config = model_config.pretrained_config
        self.audio_gen = getattr(pretrained_config, "sound_gen", False)
        self.action_gen = getattr(pretrained_config, "action_gen", False)

        self.hidden_size = pretrained_config.hidden_size
        self.num_hidden_layers = pretrained_config.num_hidden_layers
        self.latent_patch_size = pretrained_config.latent_patch_size
        self.latent_channel_size = pretrained_config.latent_channel
        self.patch_latent_dim = (self.latent_patch_size**2) * self.latent_channel_size
        self.timestep_scale = pretrained_config.timestep_scale
        self.base_fps = pretrained_config.base_fps

        # Comes from VAE. Updated after VAE is loaded.
        self.temporal_compression_factor = 4

        self.unified_3d_mrope_temporal_modality_margin = (
            pretrained_config.unified_3d_mrope_temporal_modality_margin
        )
        self.num_attention_heads = pretrained_config.num_attention_heads
        self.num_kv_heads = pretrained_config.num_key_value_heads
        self.enable_fps_modulation = pretrained_config.enable_fps_modulation

        if self.audio_gen:
            self.audio_dim = pretrained_config.sound_dim
            self.audio_latent_fps = pretrained_config.sound_latent_fps
            self.temporal_compression_factor_audio = (
                pretrained_config.temporal_compression_factor_sound
            )

        if pretrained_config.position_embedding_type != "unified_3d_mrope":
            raise ValueError(
                f"Position embedding type {pretrained_config.position_embedding_type} not supported"
            )

        vgm = model_config.visual_gen_mapping

        self.sharder = SequenceSharder.from_vgm(
            vgm,
            num_attention_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
        )
        tp_size = vgm.tp_size if vgm else 1
        ulysses_size = vgm.ulysses_size if vgm else 1
        ring_size = vgm.ring_size if vgm else 1
        head_divisibility_factor = tp_size * ulysses_size

        if (ulysses_size > 1 or tp_size > 1) and (
            self.num_attention_heads % head_divisibility_factor != 0
            or self.num_kv_heads % head_divisibility_factor != 0
        ):
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) and "
                f"num_kv_heads ({self.num_kv_heads}) must be divisible by "
                f"TP * Ulysses size ({tp_size} * {ulysses_size})"
            )

        if ring_size > 1:
            # Ring parallelism is not compatible with Cosmos3 cross-attention.
            raise NotImplementedError(
                "Ring parallelism is not supported for Cosmos3 cross-attention."
            )

        self.language_model = Cosmos3LanguageModel(model_config)

        self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
        self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)

        if self.audio_gen:
            # Projections for audio modality (mirrors cosmos3-internal Cosmos3VFMNetwork)
            self.audio2llm = nn.Linear(self.audio_dim, self.hidden_size)
            self.llm2audio = nn.Linear(self.hidden_size, self.audio_dim)
            self.audio_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        # try timestep embedder in float32 if acc loss
        self.time_embedder = TimestepEmbedder(self.hidden_size, target_dtype=torch.bfloat16)

        self.gen_layers = nn.ModuleList(
            [
                Cosmos3GenDecoderLayer(model_config, layer_idx=i)
                for i in range(self.num_hidden_layers)
            ]
        )

        self.norm_moe_gen = Qwen3VLTextRMSNorm(
            hidden_size=self.hidden_size,
            eps=pretrained_config.rms_norm_eps,
        )

        self.cached_kv = None
        self.cached_freqs_gen = None

        self.__post_init__()

    @property
    def device(self):
        return next(self.parameters()).device

    def __post_init__(self):
        # TODO: move this to pipeline loader under meta init so transformers' dont need to know about it here
        self.apply_quant_config_exclude_modules()

        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def apply_quant_config_exclude_modules(self):
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return

        kv_cache_quant_algo = quant_config.kv_cache_quant_algo if quant_config else None
        no_quant_config = QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                is_excluded = quant_config.is_module_excluded_from_quantization(name)
                if is_excluded and getattr(module, "quant_config", None) is not None:
                    module.quant_config = no_quant_config

    def _pad_to_patch_size(self, H: int, W: int) -> Tuple[int, int, int, int]:
        """Compute padded spatial dims aligned to patch_size.

        Returns (Hp, Wp, H_padded, W_padded) where Hp/Wp are the patch grid
        dimensions and H_padded/W_padded are the padded latent dimensions.
        """
        p = self.latent_patch_size
        H_padded = ((H + p - 1) // p) * p
        W_padded = ((W + p - 1) // p) * p
        return H_padded // p, W_padded // p, H_padded, W_padded

    def patchify(self, latents: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """[B, C, T, H, W] -> [B, T*Hp*Wp, p*p*C], padding H/W if needed."""
        B = latents.shape[0]
        p = self.latent_patch_size
        C = self.latent_channel_size
        Hp, Wp, H_padded, W_padded = self._pad_to_patch_size(H, W)

        if H_padded != H or W_padded != W:
            latents = F.pad(latents, (0, W_padded - W, 0, H_padded - H))

        x = latents.reshape(B, C, T, Hp, p, Wp, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)  # [B, T, Hp, Wp, p, p, C]
        return x.reshape(B, T * Hp * Wp, p * p * C)

    def unpatchify(self, tokens: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """[B, T*Hp*Wp, p*p*C] -> [B, C, T, H, W], cropping padding if needed."""
        B = tokens.shape[0]
        p = self.latent_patch_size
        C = self.latent_channel_size
        Hp, Wp, H_padded, W_padded = self._pad_to_patch_size(H, W)

        x = tokens.reshape(B, T, Hp, Wp, p, p, C)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)  # [B, C, T, Hp, p, Wp, p]
        x = x.reshape(B, C, T, H_padded, W_padded)

        if H_padded != H or W_padded != W:
            x = x[:, :, :, :H, :W]
        return x

    def _compute_rope_freqs(
        self,
        text_mask: torch.Tensor,
        T: int,
        Hp: int,
        Wp: int,
        fps: float | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Compute mRoPE cos/sin for UND (text) and GEN (visual) pathways."""
        B = text_mask.shape[0]
        S_text = text_mask.shape[1]
        text_lengths = text_mask.sum(dim=1).long()
        effective_fps = fps if fps is not None and T > 1 else None

        text_pos_list = []
        vis_pos_list = []
        for b in range(B):
            real_len = int(text_lengths[b].item())
            t_pos, t_offset = compute_mrope_position_ids_text(real_len, temporal_offset=0)
            v_pos, _ = compute_mrope_position_ids_vision(
                T,
                Hp,
                Wp,
                temporal_offset=t_offset + self.unified_3d_mrope_temporal_modality_margin,
                fps=effective_fps,
                base_fps=self.base_fps,
                temporal_compression_factor=self.temporal_compression_factor,
                enable_fps_modulation=self.enable_fps_modulation,
            )
            if real_len < S_text:
                t_pos = torch.cat(
                    [t_pos, torch.zeros(3, S_text - real_len, dtype=t_pos.dtype)], dim=1
                )
            text_pos_list.append(t_pos)
            vis_pos_list.append(v_pos)

        text_pos_ids = torch.stack(text_pos_list, dim=1).to(device)  # [3, B, S_text]
        vis_pos_ids = torch.stack(vis_pos_list, dim=1).to(device)  # [3, B, S_vis]

        rotary_emb = self.language_model.rotary_emb
        _dummy = torch.tensor([], dtype=dtype, device=device)
        cos_und, sin_und = rotary_emb(_dummy, position_ids=text_pos_ids)
        cos_gen, sin_gen = rotary_emb(_dummy, position_ids=vis_pos_ids)

        freqs_und = (cos_und.unsqueeze(2), sin_und.unsqueeze(2))  # (B, S, 1, 128)
        freqs_gen = (cos_gen.unsqueeze(2), sin_gen.unsqueeze(2))
        return freqs_und, freqs_gen

    # -------------------------------------------------------------------------
    # Audio helpers
    # -------------------------------------------------------------------------

    def _compute_audio_rope_freqs(
        self,
        T_audio: int,
        text_mask: torch.Tensor,
        fps_audio: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mRoPE cos/sin for audio tokens.

        Audio tokens use a 1×1 spatial grid (H=W=1) aligned with the vision
        temporal axis at the audio latent rate.  This mirrors the cosmos3-internal
        ``sequence_packing.py`` treatment where audio mRoPE uses
        ``get_3d_mrope_ids_vae_tokens(grid_h=1, grid_w=1, tcf=1)``.
        """
        B = text_mask.shape[0]
        text_lengths = text_mask.sum(dim=1).long()

        audio_pos_list = []
        for b in range(B):
            real_len = int(text_lengths[b].item())
            _, t_offset = compute_mrope_position_ids_text(real_len, temporal_offset=0)
            # Audio tokens share the vision temporal space; use modality margin offset.
            s_pos, _ = compute_mrope_position_ids_vision(
                T_audio,
                1,  # grid_h
                1,  # grid_w
                temporal_offset=t_offset + self.unified_3d_mrope_temporal_modality_margin,
                fps=fps_audio,
                base_fps=self.base_fps,
                temporal_compression_factor=1,  # audio latent is already at audio_latent_fps
                enable_fps_modulation=self.enable_fps_modulation,
            )
            audio_pos_list.append(s_pos)

        audio_pos_ids = torch.stack(audio_pos_list, dim=1).to(device)  # [3, B, T_audio]
        rotary_emb = self.language_model.rotary_emb
        _dummy = torch.tensor([], dtype=dtype, device=device)
        cos_a, sin_a = rotary_emb(_dummy, position_ids=audio_pos_ids)
        return cos_a.unsqueeze(2), sin_a.unsqueeze(2)  # [B, T_audio, 1, head_dim]

    def pack_audio_latents(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """[B, audio_dim, T_audio] → [B, T_audio, audio_dim]."""
        return audio_latents.permute(0, 2, 1)

    def unpack_audio_latents(self, hidden_audio: torch.Tensor) -> torch.Tensor:
        """[B, T_audio, audio_dim] → [B, audio_dim, T_audio]."""
        return hidden_audio.permute(0, 2, 1)

    def reset_cache(self):
        self.cached_kv = None
        self.cached_freqs_gen = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        raw_timestep: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        video_shape: Optional[Tuple[int, int, int]] = None,
        fps: float | None = None,
        noisy_frame_mask: torch.Tensor | None = None,
        audio_latents: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "TransformerOutput":
        """
        Forward pass for parallel denoising.

        Args:
            hidden_states: [B, C, T, H, W] noisy latents
            timestep: Normalized diffusion timestep in [0, 1], shape [B].
            raw_timestep: Raw scheduler diffusion timestep, shape [B], used by
                the Cosmos3 time embedding path.
            text_ids: [B, S_text] tokenized text input
            text_mask: [B, S_text] attention mask for text (1=real, 0=pad)
            video_shape: (T, H, W) in latent space
            fps: video frame rate; when provided, temporal mRoPE positions are
                 scaled to reflect real time (FPS modulation).
            noisy_frame_mask: Optional [B, 1, T, 1, 1] mask where 1=noisy (add
                timestep embedding, predict velocity) and 0=conditioned (clean
                context, skip timestep embedding).  None means all frames noisy
                (T2V mode).
            audio_latents: Optional [B, audio_dim, T_audio] noisy audio latents.
                When provided, audio tokens are appended to the generation
                sequence and an audio velocity is returned alongside the video
                velocity.  Requires ``audio_gen=True`` in the pretrained config.

        Returns:
            TransformerOutput with video (and image alias) always set.
            audio is set to the predicted audio velocity when audio_latents is
            provided; otherwise None.  action is always None for now.
        """
        del kwargs  # Kept for diffusers API compatibility.
        if timestep is None:
            raise ValueError("Cosmos3VFMTransformer.forward requires normalized timestep.")
        if raw_timestep is None:
            raise ValueError("Cosmos3VFMTransformer.forward requires raw_timestep.")
        T, H, W = video_shape
        Hp, Wp, _, _ = self._pad_to_patch_size(H, W)
        max_real_len = text_mask.sum(dim=1).max().item()
        real_text_lens = text_mask.sum(dim=1).tolist()

        hidden_gen = self.vae2llm(self.patchify(hidden_states, T, H, W))

        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            time_embed = self.time_embedder((raw_timestep * self.timestep_scale))
        time_embed = time_embed.to(hidden_states.dtype)

        if noisy_frame_mask is not None:
            # Build per-token mask from per-frame mask.
            # noisy_frame_mask: [B, 1, T, 1, 1] → token mask: [B, T*Hp*Wp, 1]
            noisy_frame_mask = noisy_frame_mask.expand(hidden_gen.shape[0], -1, -1, -1, -1)
            token_noisy_mask = (
                noisy_frame_mask[:, 0, :, 0, 0]  # [B, T]
                .unsqueeze(-1)  # [B, T, 1]
                .expand(-1, -1, Hp * Wp)  # [B, T, Hp*Wp]
                .reshape(hidden_gen.shape[0], -1, 1)  # [B, T*Hp*Wp, 1]
            )
            hidden_gen = hidden_gen + time_embed.unsqueeze(1) * token_noisy_mask
        else:
            hidden_gen = hidden_gen + time_embed.unsqueeze(1)

        if self.cached_kv is None:
            freqs_und, freqs_gen = self._compute_rope_freqs(
                text_mask,
                T,
                Hp,
                Wp,
                fps,
                hidden_states.device,
                hidden_states.dtype,
            )
            cached_kv_full = self.language_model(
                text_ids,
                text_mask,
                freqs_und,
                timestep=timestep,
            )
            self.cached_freqs_gen = freqs_gen

            if self.sharder.is_active:
                # Round max_real_len up to next multiple of sharder.size.
                # At most size-1 extra positions, negligible softmax dilution.
                val = (self.sharder.size - max_real_len % self.sharder.size) % self.sharder.size
                S_text_shard_total = int(max_real_len) + val

                self.cached_kv = []
                for k, v in cached_kv_full:
                    k = k[:, :S_text_shard_total].clone()
                    v = v[:, :S_text_shard_total].clone()
                    if val > 0:
                        k[:, int(max_real_len) :] = 0
                        v[:, int(max_real_len) :] = 0
                    self.cached_kv.append(
                        (self.sharder.shard(k, dim=1), self.sharder.shard(v, dim=1))
                    )
            else:
                self.cached_kv = cached_kv_full

        # --- Audio token injection -------------------------------------------------
        T_vid_tokens = hidden_gen.shape[1]  # T * Hp * Wp
        T_audio = 0
        if audio_latents is not None and self.audio_gen:
            T_audio = audio_latents.shape[2]
            hidden_audio = self.pack_audio_latents(audio_latents).to(hidden_gen.dtype)
            hidden_audio = self.audio2llm(hidden_audio) + self.audio_modality_embed
            hidden_audio = hidden_audio + time_embed.unsqueeze(1)
            cos_a, sin_a = self._compute_audio_rope_freqs(
                T_audio,
                text_mask,
                float(self.audio_latent_fps),
                hidden_states.device,
                hidden_gen.dtype,
            )
            # [B, T_vid+T_audio, hidden_size]
            hidden_gen = torch.cat([hidden_gen, hidden_audio], dim=1)
            cos_v, sin_v = self.cached_freqs_gen
            freqs_gen_combined = (
                torch.cat([cos_v, cos_a], dim=1),
                torch.cat([sin_v, sin_a], dim=1),
            )
        else:
            freqs_gen_combined = self.cached_freqs_gen
        # --------------------------------------------------------------------------

        S_gen = hidden_gen.shape[1]
        hidden_gen = self.sharder.shard(hidden_gen, dim=1, pad_to_multiple=True)
        cos, sin = freqs_gen_combined
        cos = self.sharder.shard(cos, dim=1, pad_to_multiple=True)
        sin = self.sharder.shard(sin, dim=1, pad_to_multiple=True)
        freqs_gen = (cos, sin)

        for i, layer in enumerate(self.gen_layers):
            k_und, v_und = self.cached_kv[i]
            if not self.sharder.is_active:
                k_und = k_und[:, :max_real_len]
                v_und = v_und[:, :max_real_len]
                hidden_gen = layer(
                    hidden_gen,
                    k_und,
                    v_und,
                    freqs_gen,
                    timestep=timestep,
                    real_text_lens=real_text_lens,
                )
            else:
                hidden_gen = layer(
                    hidden_gen,
                    k_und,
                    v_und,
                    freqs_gen,
                    timestep=timestep,
                )

        hidden_gen = self.sharder.gather(hidden_gen, dim=1, unpad_to=S_gen)

        hidden_gen = self.norm_moe_gen(hidden_gen)

        # --- Decode video velocity ------------------------------------------------
        video_vel = self.unpatchify(self.llm2vae(hidden_gen[:, :T_vid_tokens]), T, H, W)

        # --- Decode audio velocity (if requested) ---------------------------------
        audio_vel = None
        if T_audio > 0 and audio_latents is not None and self.audio_gen:
            # hidden_gen[:, T_vid_tokens:] → [B, T_audio, hidden_size]
            # → llm2audio → [B, T_audio, audio_dim] → unpack → [B, audio_dim, T_audio]
            audio_vel = self.unpack_audio_latents(
                self.llm2audio(hidden_gen[:, T_vid_tokens : T_vid_tokens + T_audio])
            )

        return TransformerOutput(video=video_vel, image=video_vel, audio=audio_vel)

    def load_weights(self, weights: dict) -> None:
        """Load weights with key remapping from Cosmos3-Nano / Diffusers checkpoints.

        Expects tensor names as in ``diffusion_pytorch_model.safetensors.index.json``
        (e.g. ``layers.{i}.self_attn.to_q.weight``, ``proj_in.weight``).
        Maps UND vs GEN blocks into this module's layout (causal self-attn vs cross-attn + MLPs).
        """
        remapped = {}
        skip_prefixes = (
            "lm_head.",
            "action_modality_embed",
            "action_proj_",
        )

        for key, value in weights.items():
            k = key

            if k.startswith(skip_prefixes):
                continue

            # Normalize a leading "model." prefix up front so every remap below
            # matches whether or not the checkpoint namespaces top-level tensors
            # (e.g. "model.audio_proj_in.weight") under "model.".
            if k.startswith("model."):
                k = k[len("model.") :]

            if k.startswith("proj_in."):
                remapped[k.replace("proj_in.", "vae2llm.", 1)] = value
                continue

            if k.startswith("proj_out."):
                remapped[k.replace("proj_out.", "llm2vae.", 1)] = value
                continue

            if k.startswith("audio_proj_in."):
                remapped[k.replace("audio_proj_in.", "audio2llm.", 1)] = value
                continue

            if k.startswith("audio_proj_out."):
                remapped[k.replace("audio_proj_out.", "llm2audio.", 1)] = value
                continue

            if k.startswith("audio_modality_embed"):
                remapped[k] = value
                continue

            if k.startswith("time_embedder.linear"):
                k = k.replace("time_embedder.linear_1.", "time_embedder.mlp.linear_1.")
                k = k.replace("time_embedder.linear_2.", "time_embedder.mlp.linear_2.")
                remapped[k] = value
                continue

            # embed_tokens and norm → language_model.*
            if k.startswith("embed_tokens.") or k.startswith("norm."):
                remapped[f"language_model.{k}"] = value
                continue

            # norm_moe_gen stays at top level
            if k.startswith("norm_moe_gen."):
                remapped[k] = value
                continue

            if not k.startswith("layers."):
                logger.warning(f"Skipping unknown checkpoint key: {key}")
                continue

            parts = k.split(".", 2)  # ['layers', '{i}', '{rest}']
            layer_idx = parts[1]
            rest = parts[2]

            # UND (language_model) prefix
            und_lp = f"language_model.layers.{layer_idx}"
            # GEN prefix
            gen_lp = f"gen_layers.{layer_idx}"

            # --- UND attention → language_model.layers.{i}.self_attn.* ---
            attn_und_map = {
                "self_attn.to_q.": f"{und_lp}.self_attn.to_q.",
                "self_attn.to_k.": f"{und_lp}.self_attn.to_k.",
                "self_attn.to_v.": f"{und_lp}.self_attn.to_v.",
                "self_attn.to_out.": f"{und_lp}.self_attn.to_out.0.",
                "self_attn.norm_q.": f"{und_lp}.self_attn.norm_q.",
                "self_attn.norm_k.": f"{und_lp}.self_attn.norm_k.",
            }

            # --- GEN attention → gen_layers.{i}.cross_attention.* ---
            attn_gen_map = {
                "self_attn.add_q_proj.": f"{gen_lp}.cross_attention.to_q.",
                "self_attn.add_k_proj.": f"{gen_lp}.cross_attention.to_k.",
                "self_attn.add_v_proj.": f"{gen_lp}.cross_attention.to_v.",
                "self_attn.to_add_out.": f"{gen_lp}.cross_attention.to_out.0.",
                "self_attn.norm_added_q.": f"{gen_lp}.cross_attention.norm_q.",
                "self_attn.norm_added_k.": f"{gen_lp}.cross_attention.norm_k.",
            }

            # --- Norms ---
            norm_map = {
                "input_layernorm.": f"{und_lp}.input_layernorm.",
                "post_attention_layernorm.": f"{und_lp}.post_attention_layernorm.",
                "input_layernorm_moe_gen.": f"{gen_lp}.input_layernorm.",
                "post_attention_layernorm_moe_gen.": f"{gen_lp}.post_attention_layernorm.",
            }

            # --- MLPs ---
            mlp_map = {
                "mlp.gate_proj.": f"{und_lp}.mlp.gate_proj.",
                "mlp.up_proj.": f"{und_lp}.mlp.up_proj.",
                "mlp.down_proj.": f"{und_lp}.mlp.down_proj.",
                "mlp_moe_gen.gate_proj.": f"{gen_lp}.mlp.gate_proj.",
                "mlp_moe_gen.up_proj.": f"{gen_lp}.mlp.up_proj.",
                "mlp_moe_gen.down_proj.": f"{gen_lp}.mlp.down_proj.",
            }

            matched = False
            for mapping in [attn_gen_map, attn_und_map, norm_map, mlp_map]:
                for pattern, replacement in mapping.items():
                    if rest.startswith(pattern):
                        suffix = rest[len(pattern) :]
                        remapped[replacement + suffix] = value
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                logger.warning(f"Unmatched layer key: {key}")

        # --- Load using DynamicLinearWeightLoader (handles QKV/gate_up fusion + quantization) ---
        params_map = {
            "qkv_proj": ["to_q", "to_k", "to_v"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        loader = DynamicLinearWeightLoader(self.model_config, params_map=params_map)

        for param_name, param in self._parameters.items():
            if param is not None and param_name in remapped:
                param.data.copy_(remapped[param_name].to(param.dtype))

        loaded_linear = 0
        loaded_other = 0
        skipped_modules = []
        for name, module in self.named_modules():
            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, remapped)
                if weight_dicts:
                    loader.load_linear_weights(module, name, weight_dicts)
                    loaded_linear += 1
                else:
                    skipped_modules.append(f"{name}(Linear)")
            else:
                module_weights = loader.filter_weights(name, remapped)
                if module_weights:
                    loaded_other += 1
                else:
                    has_params = any(p is not None for p in module._parameters.values())
                    if has_params and name:
                        skipped_modules.append(f"{name}({type(module).__name__})")
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(module_weights[param_name].to(param.dtype))

    def post_load_weights(self) -> None:
        """Post-load processing: dtype conversion and Linear finalization."""
        target_dtype = self.model_config.torch_dtype

        self.time_embedder.to(torch.float32)
        self.language_model.embed_tokens.to(target_dtype)
        self.vae2llm.to(target_dtype)
        self.llm2vae.to(target_dtype)

        if self.audio_gen:
            self.audio2llm.to(target_dtype)
            self.llm2audio.to(target_dtype)
            self.audio_modality_embed.data = self.audio_modality_embed.data.to(target_dtype)

        for _, module in self.named_modules():
            if isinstance(module, Linear) or isinstance(module, Qwen3VLTextRMSNorm):
                module.post_load_weights()
