# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX attention modules: joint attention and parallel self-attention.

Key Components:
- FluxJointAttention: Joint attention for dual-stream blocks (FLUX.1 and FLUX.2)
- Flux2ParallelSelfAttention: Fused QKV+MLP for FLUX.2 single-stream blocks
"""

from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode, apply_rotary_emb

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
                bias=self.bias,
                dtype=self.dtype,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
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
                bias=self.bias,
                dtype=self.dtype,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
            )

    # -----------------------------------------------------------------
    # Fused kernel helper (used by fused subclasses)
    # -----------------------------------------------------------------

    def _invoke_fused_kernel(
        self,
        qkv: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        num_txt_tokens: int,
    ) -> None:
        """Apply fused DiT QK Norm + RoPE kernel in-place on packed QKV.

        Args:
            qkv: Packed QKV tensor [batch, seq, 3 * q_dim]
            freqs_cos: RoPE cos embeddings [1, seq, 1, head_dim]
            freqs_sin: RoPE sin embeddings [1, seq, 1, head_dim]
            num_txt_tokens: Text token boundary (tokens [0, num_txt) use add_weights).
                           Set to -1 for no dual-stream.
        """
        # [1, S, 1, D] -> [S, D], float32
        cos_2d = freqs_cos.squeeze(0).squeeze(1).float().contiguous()
        sin_2d = freqs_sin.squeeze(0).squeeze(1).float().contiguous()

        for b in range(qkv.shape[0]):
            torch.ops.trtllm.fused_dit_qk_norm_rope(
                qkv[b],  # [seq, 3*q_dim] — view into original, modified in-place
                self.num_attention_heads,
                self.num_attention_heads,
                self.num_attention_heads,
                self.head_dim,
                self.eps,
                self.norm_q.weight,
                self.norm_k.weight,
                self._q_add_weight,
                self._k_add_weight,
                cos_2d,
                sin_2d,
                num_txt_tokens,
            )

    # -----------------------------------------------------------------
    # QKV preparation: unfused (default) and fused (override)
    # -----------------------------------------------------------------

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override: use F.rms_norm for per-head norm
        - ~1.6x speedup on Flux.2 inputs
        - Better fusion with torch.compile
        """
        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)
        return q, k

    def apply_qk_added_norm(
        self, enc_q: torch.Tensor, enc_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override: Apply F.rms_norm to text-stream QK,
        - ~1.6x speedup on Flux.2 inputs
        - Better fusion with torch.compile
        """
        enc_q = F.rms_norm(
            enc_q, (enc_q.shape[-1],), self.norm_added_q.weight, self.norm_added_q.variance_epsilon
        )
        enc_k = F.rms_norm(
            enc_k, (enc_k.shape[-1],), self.norm_added_k.weight, self.norm_added_k.variance_epsilon
        )
        return enc_q, enc_k

    def _prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Prepare Q, K, V with separate norm + RoPE (unfused default).

        Returns:
            (query, key, value, seq_len) — all 3D [B, S, q_dim].
        """
        batch_size = hidden_states.shape[0]
        is_dual_stream = encoder_hidden_states is not None and self.added_kv_proj_dim is not None

        query, key, value = self.get_qkv(hidden_states)
        query = query.view(batch_size, -1, self.num_attention_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_attention_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_attention_heads, self.head_dim)

        query, key = self.apply_qk_norm(query, key)

        if is_dual_stream:
            encoder_qkv = self.add_qkv_proj(encoder_hidden_states)
            enc_q, enc_k, enc_v = encoder_qkv.chunk(3, dim=-1)
            enc_q = enc_q.view(batch_size, -1, self.num_attention_heads, self.head_dim)
            enc_k = enc_k.view(batch_size, -1, self.num_attention_heads, self.head_dim)
            enc_v = enc_v.view(batch_size, -1, self.num_attention_heads, self.head_dim)

            enc_q, enc_k = self.apply_qk_added_norm(enc_q, enc_k)

            query = torch.cat([enc_q, query], dim=1)
            key = torch.cat([enc_k, key], dim=1)
            value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            query = apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb(key, freqs_cos, freqs_sin)

        seq_len = query.shape[1]
        return query.flatten(2), key.flatten(2), value.flatten(2), seq_len

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = hidden_states.shape[0]
        is_dual_stream = encoder_hidden_states is not None and self.added_kv_proj_dim is not None
        txt_seq_len = encoder_hidden_states.shape[1] if is_dual_stream else 0

        query, key, value, seq_len = self._prepare_qkv(
            hidden_states, encoder_hidden_states, image_rotary_emb
        )

        hidden_states = self._attn_impl(query, key, value, batch_size, seq_len)
        hidden_states = hidden_states.to(query.dtype)

        if is_dual_stream:
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
# Parallel Self-Attention (for FLUX.2 single-stream blocks)
# =============================================================================


class Flux2ParallelSelfAttention(FluxJointAttention):
    """FLUX.2 parallel self-attention for single-stream blocks.

    Uses fused QKV+MLP projection: to_qkv_mlp_proj
    Output: concatenate attention + MLP outputs, then project with to_out

    This is a key architectural difference from FLUX.1:
    - FLUX.1: Separate attention and FFN
    - FLUX.2: Fused QKV+MLP projection for efficiency
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
            eps=eps,
            pre_only=True,  # Deletes base to_out
            config=config,
            layer_idx=layer_idx,
        )

        # Combined output: [q_dim + mlp_hidden_dim] -> [hidden_size]
        self.to_out = Linear(
            self.q_dim + self.mlp_hidden_dim,
            hidden_size,
            bias=bias,
            dtype=self.dtype,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
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
        )

    def _apply_norm_rope(
        self,
        qkv: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Apply separate norm + RoPE to packed QKV (unfused default).

        Returns:
            (q, k, v, seq_len) — all 3D [B, S, q_dim].
        """
        batch_size = qkv.shape[0]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_attention_heads, self.head_dim)

        q, k = self.apply_qk_norm(q, k)

        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        seq_len = q.shape[1]
        return q.flatten(2), k.flatten(2), v.flatten(2), seq_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # Fused QKV + MLP projection
        proj_out = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden = torch.split(
            proj_out, [3 * self.q_dim, self.mlp_hidden_dim * self.mlp_mult_factor], dim=-1
        )

        q, k, v, seq_len = self._apply_norm_rope(qkv, image_rotary_emb)

        attn_out = self._attn_impl(q, k, v, batch_size, seq_len)
        attn_out = attn_out.to(q.dtype)

        # Parallel MLP path (reshape to 2D for Triton kernel, then back)
        shape = mlp_hidden.shape
        mlp_out = swiglu(mlp_hidden.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)

        # Concatenate + project
        return self.to_out(torch.cat([attn_out, mlp_out], dim=-1))


# =============================================================================
# Fused subclasses (fused DiT QK Norm + RoPE CUDA kernel)
# =============================================================================


class _FusedFluxJointAttention(FluxJointAttention):
    """FluxJointAttention with fused QK Norm + RoPE via custom CUDA kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._q_add_weight = self.norm_added_q.weight if hasattr(self, "norm_added_q") else None
        self._k_add_weight = self.norm_added_k.weight if hasattr(self, "norm_added_k") else None

    def _prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        is_dual_stream = encoder_hidden_states is not None and self.added_kv_proj_dim is not None
        if image_rotary_emb is None:
            raise ValueError("Fused QK Norm + RoPE kernel requires image_rotary_emb")
        freqs_cos, freqs_sin = image_rotary_emb

        img_qkv = self.qkv_proj(hidden_states)

        if is_dual_stream:
            txt_qkv = self.add_qkv_proj(encoder_hidden_states)
            qkv = torch.cat([txt_qkv, img_qkv], dim=1)
            num_txt = encoder_hidden_states.shape[1]
        else:
            qkv = img_qkv
            num_txt = -1

        self._invoke_fused_kernel(qkv, freqs_cos, freqs_sin, num_txt)

        query, key, value = qkv.split([self.q_dim, self.q_dim, self.q_dim], dim=-1)
        return query, key, value, query.shape[1]


class _FusedFlux2ParallelSelfAttention(Flux2ParallelSelfAttention):
    """Flux2ParallelSelfAttention with fused QK Norm + RoPE via custom CUDA kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._q_add_weight = None
        self._k_add_weight = None

    def _apply_norm_rope(
        self,
        qkv: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if image_rotary_emb is None:
            raise ValueError("Fused QK Norm + RoPE kernel requires image_rotary_emb")
        freqs_cos, freqs_sin = image_rotary_emb

        # torch.split produces non-contiguous views; fused kernel requires contiguous
        qkv = qkv.contiguous()

        self._invoke_fused_kernel(qkv, freqs_cos, freqs_sin, num_txt_tokens=-1)

        q, k, v = qkv.split([self.q_dim, self.q_dim, self.q_dim], dim=-1)
        return q, k, v, q.shape[1]


# =============================================================================
# Factory functions
# =============================================================================


def create_joint_attention(**kwargs) -> FluxJointAttention:
    """Create a FluxJointAttention, using the fused kernel variant when enabled."""
    config = kwargs.get("config")
    if config is not None and config.attention.fuse_qk_norm_rope:
        return _FusedFluxJointAttention(**kwargs)
    return FluxJointAttention(**kwargs)


def create_parallel_self_attention(**kwargs) -> Flux2ParallelSelfAttention:
    """Create a Flux2ParallelSelfAttention, using the fused kernel variant when enabled."""
    config = kwargs.get("config")
    if config is not None and config.attention.fuse_qk_norm_rope:
        return _FusedFlux2ParallelSelfAttention(**kwargs)
    return Flux2ParallelSelfAttention(**kwargs)
