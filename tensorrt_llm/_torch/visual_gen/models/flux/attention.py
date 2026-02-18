# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX attention modules: joint attention, position embeddings, and parallel self-attention.

Key Components:
- FluxJointAttention: Joint attention for dual-stream blocks (FLUX.1 and FLUX.2)
- Flux2ParallelSelfAttention: Fused QKV+MLP for FLUX.2 single-stream blocks
- Flux2PosEmbed: 4-axis rotary position embeddings (FLUX.2)
- Flux2SwiGLU: SwiGLU activation for FLUX.2 FFN
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn

from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode, _per_head_norm

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


# =============================================================================
# Joint Attention (shared by FLUX.1 and FLUX.2 dual-stream blocks)
# =============================================================================


class FluxJointAttention(Attention):
    """Joint attention module for FLUX transformer models (FLUX.1 and FLUX.2).

    Extends base Attention with:
    - Text-stream QKV projection (add_qkv_proj) for dual-stream blocks
    - FLUX-style RoPE on concatenated text+image tokens
    - pre_only mode for single-stream blocks (no output projection)

    For dual-stream blocks: Returns (img_attn_output, txt_attn_output)
    For single-stream blocks: pre_only=True, returns attention output only
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int = 128,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: bool = True,
        out_bias: bool = True,
        eps: float = 1e-6,
        pre_only: bool = False,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            qkv_mode=QKVMode.FUSE_QKV,
            qk_norm=True,
            qk_norm_mode="per_head",
            eps=eps,
            bias=bias,
            out_bias=out_bias,
            config=config,
            layer_idx=layer_idx,
        )

        self.pre_only = pre_only
        self.added_kv_proj_dim = added_kv_proj_dim

        # Delete output projection for single-stream blocks
        if self.pre_only:
            del self.to_out

        # Text-stream projections for joint attention (dual-stream blocks only)
        if added_kv_proj_dim is not None:
            self.add_qkv_proj = Linear(
                added_kv_proj_dim,
                3 * self.q_dim,
                bias=added_proj_bias,
                dtype=self.dtype,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                disable_deep_gemm=True,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR
                ),
                fused_weight_shard_indices_mapping={
                    "q": (0, self.q_dim),
                    "k": (self.q_dim, self.q_dim),
                    "v": (2 * self.q_dim, self.q_dim),
                },
            )

            self.norm_added_q = RMSNorm(
                hidden_size=head_dim, eps=eps, dtype=self.dtype, has_weights=True
            )
            self.norm_added_k = RMSNorm(
                hidden_size=head_dim, eps=eps, dtype=self.dtype, has_weights=True
            )

            self.to_add_out = Linear(
                self.q_dim,
                added_kv_proj_dim,
                bias=out_bias,
                dtype=self.dtype,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                disable_deep_gemm=True,
            )

    @staticmethod
    def _apply_rope(
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FLUX-style rotary embeddings to a 4D tensor.

        Args:
            x: Input [B, S, H, D]
            freqs_cos: Cosine frequencies [S, D]
            freqs_sin: Sine frequencies [S, D]
        """
        x_fp32 = x.float()
        cos = freqs_cos.float().unsqueeze(0).unsqueeze(2)  # [1, S, 1, D]
        sin = freqs_sin.float().unsqueeze(0).unsqueeze(2)

        # Rotate pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        x_rotated = torch.stack([-x_fp32[..., 1::2], x_fp32[..., 0::2]], dim=-1).flatten(-2)
        return (x_fp32 * cos + x_rotated * sin).to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of joint attention.

        Args:
            hidden_states: Image tokens [batch, img_seq, dim]
            encoder_hidden_states: Text tokens [batch, txt_seq, dim] (for dual-stream)
            attention_mask: Optional attention mask (unused, for API compat)
            image_rotary_emb: Tuple of (cos, sin) for RoPE

        Returns:
            For dual-stream: Tuple of (img_attn_output, txt_attn_output)
            For single-stream: Attention output tensor
        """
        batch_size = hidden_states.shape[0]

        # Image QKV via base (returns 3D), then reshape to 4D for per-head ops
        query, key, value = self.get_qkv(hidden_states)
        query = query.unflatten(-1, (self.num_attention_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_attention_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_attention_heads, self.head_dim))

        # Per-head QK normalization via base (per_head mode operates on 4D)
        query, key = self.apply_qk_norm(query, key)

        # Text QKV for joint attention (dual-stream blocks)
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            txt_seq_len = encoder_hidden_states.shape[1]

            encoder_qkv = self.add_qkv_proj(encoder_hidden_states)
            enc_q, enc_k, enc_v = encoder_qkv.chunk(3, dim=-1)
            enc_q = enc_q.unflatten(-1, (self.num_attention_heads, self.head_dim))
            enc_k = enc_k.unflatten(-1, (self.num_attention_heads, self.head_dim))
            enc_v = enc_v.unflatten(-1, (self.num_attention_heads, self.head_dim))

            enc_q = _per_head_norm(enc_q, self.norm_added_q)
            enc_k = _per_head_norm(enc_k, self.norm_added_k)

            # Concatenate text + image for joint attention
            query = torch.cat([enc_q, query], dim=1)
            key = torch.cat([enc_k, key], dim=1)
            value = torch.cat([enc_v, value], dim=1)

        # Apply RoPE
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            query = self._apply_rope(query, freqs_cos, freqs_sin)
            key = self._apply_rope(key, freqs_cos, freqs_sin)

        # Flatten 4D->3D for base _attn_impl
        seq_len = query.shape[1]
        query = query.flatten(2)
        key = key.flatten(2)
        value = value.flatten(2)

        hidden_states = self._attn_impl(query, key, value, batch_size, seq_len)
        hidden_states = hidden_states.to(query.dtype)

        # Split and project outputs
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_hidden_states_out, hidden_states = hidden_states.split(
                [txt_seq_len, hidden_states.shape[1] - txt_seq_len], dim=1
            )

            if not self.pre_only:
                hidden_states = self.to_out[0](hidden_states)
            encoder_hidden_states_out = self.to_add_out(encoder_hidden_states_out)

            return hidden_states, encoder_hidden_states_out
        else:
            if not self.pre_only:
                hidden_states = self.to_out[0](hidden_states)
            return hidden_states


# =============================================================================
# Activation Functions
# =============================================================================


class Flux2SwiGLU(nn.Module):
    """SwiGLU activation function used in FLUX.2 FFN.

    FLUX.2 uses gate_fn(x1) * x2 (different from standard SwiGLU which is x * gate_fn(gate)).
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


# =============================================================================
# Position Embedding
# =============================================================================


class Flux2PosEmbed(nn.Module):
    """4-axis RoPE position embedding for FLUX.2.

    FLUX.2 uses 4 axes: [32, 32, 32, 32] = 128 head_dim
    FLUX.1 uses 3 axes: [16, 56, 56] = 128 head_dim
    """

    def __init__(self, theta: float = 2000.0, axes_dim: Tuple[int, ...] = (32, 32, 32, 32)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate RoPE embeddings from position IDs.

        Args:
            ids: Position IDs of shape [seq_len, num_axes]

        Returns:
            Tuple of (freqs_cos, freqs_sin), each of shape [seq_len, head_dim]
        """
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()

        freqs_dtype = torch.float32 if ids.device.type in ("mps", "npu") else torch.float64

        for i in range(n_axes):
            cos, sin = self._get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i], self.theta, freqs_dtype
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)

        return freqs_cos, freqs_sin

    def _get_1d_rotary_pos_embed(
        self, dim: int, pos: torch.Tensor, theta: float, freqs_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate 1D rotary position embeddings."""
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
        )
        freqs = torch.outer(pos.to(freqs_dtype), freqs)
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(pos.dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(pos.dtype)
        return cos, sin


# =============================================================================
# Parallel Self-Attention (for FLUX.2 single-stream blocks)
# =============================================================================


class Flux2ParallelSelfAttention(FluxJointAttention):
    """FLUX.2 parallel self-attention for single-stream blocks.

    Reuses from FluxJointAttention/Attention:
    - apply_qk_norm(): Per-head QK normalization
    - _apply_rope(): FLUX-style rotary embeddings
    - _attn_impl(): Backend dispatch + layout handling (+ Ulysses)
    - Attention backend creation (VANILLA/TRTLLM)

    Uses fused QKV+MLP projection: to_qkv_mlp_proj
    Output: concatenate attention + MLP outputs, then project with to_out

    This is a key architectural difference from FLUX.1:
    - FLUX.1: Separate attention and FFN
    - FLUX.2: Fused QKV+MLP projection for efficiency

    Dimensions (default FLUX.2):
    - input: 6144
    - QKV: 3 x 6144 = 18432
    - MLP: 2 x 18432 = 36864 (for SwiGLU)
    - Fused projection: 6144 -> 55296 (18432 + 36864)
    - Output: concat(6144, 18432) -> 6144
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int = 128,
        mlp_ratio: float = 3.0,
        bias: bool = False,
        eps: float = 1e-6,
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: int = 0,
    ):
        # Set MLP dims BEFORE super().__init__() — _init_qkv_proj() needs them
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2  # SwiGLU doubles input

        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            bias=bias,
            added_kv_proj_dim=None,  # No text stream
            out_bias=bias,
            eps=eps,
            pre_only=True,  # Deletes base to_out
            config=config,
            layer_idx=layer_idx,
        )

        # MLP activation (SwiGLU: 2*mlp_hidden_dim -> mlp_hidden_dim)
        self.mlp_act_fn = Flux2SwiGLU()

        # Combined output: [q_dim + mlp_hidden_dim] -> [hidden_size]
        self.to_out = Linear(
            self.q_dim + self.mlp_hidden_dim,
            hidden_size,
            bias=bias,
            dtype=self.dtype,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
            disable_deep_gemm=True,
        )

    def _init_qkv_proj(self):
        """Override: fused QKV+MLP projection instead of standard QKV."""
        qkv_dim = 3 * self.q_dim
        mlp_in_dim = self.mlp_hidden_dim * self.mlp_mult_factor
        self.to_qkv_mlp_proj = Linear(
            self.hidden_size,
            qkv_dim + mlp_in_dim,
            bias=self.bias,
            dtype=self.dtype,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
            disable_deep_gemm=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, dim]
            attention_mask: Optional attention mask
            image_rotary_emb: Tuple of (freqs_cos, freqs_sin)

        Returns:
            hidden_states [batch, seq, dim]
        """
        batch_size = hidden_states.shape[0]

        # Fused QKV + MLP projection
        proj_out = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden = torch.split(
            proj_out, [3 * self.q_dim, self.mlp_hidden_dim * self.mlp_mult_factor], dim=-1
        )

        # Split QKV -> 4D
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.unflatten(-1, (self.num_attention_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_attention_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_attention_heads, self.head_dim))

        # Per-head QK norm (inherited)
        q, k = self.apply_qk_norm(q, k)

        # RoPE (inherited static method)
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            q = self._apply_rope(q, freqs_cos, freqs_sin)
            k = self._apply_rope(k, freqs_cos, freqs_sin)

        # Flatten 4D->3D for _attn_impl
        seq_len = q.shape[1]
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        # Backend dispatch (inherited — handles layout, Ulysses, etc.)
        attn_out = self._attn_impl(q, k, v, batch_size, seq_len)
        attn_out = attn_out.to(q.dtype)

        # Parallel MLP path
        mlp_out = self.mlp_act_fn(mlp_hidden)

        # Concatenate + project
        return self.to_out(torch.cat([attn_out, mlp_out], dim=-1))
