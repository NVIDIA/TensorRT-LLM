# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX attention modules: joint attention and parallel self-attention.

Key Components:
- FluxJointAttention: Joint attention for dual-stream blocks (FLUX.1 and FLUX.2)
- Flux2ParallelSelfAttention: Fused QKV+MLP for FLUX.2 single-stream blocks
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    WeightMode,
    WeightsLoadingConfig,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.utils import Fp4QuantizedTensor
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.flux.joint_proj import (
    FluxJointAttnMLPProj,
    FluxJointQKVMLPProj,
)
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode, apply_rotary_emb
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.quantization.mode import QuantAlgo

# =============================================================================
# Joint Attention (shared by FLUX.1 and FLUX.2 dual-stream blocks)
# =============================================================================


class FluxJointAttention(Attention):
    """Joint attention module for FLUX transformer models (FLUX.1 and FLUX.2).

    Extends base Attention with:
    - Text-stream QKV projection (add_qkv_proj) for dual-stream blocks
    - FLUX-style RoPE on concatenated text+image tokens
    - pre_only mode for single-stream blocks (no output projection)

    FLUX enables fuse_qk_norm_rope by default when TP=1: the fused CUDA
    kernel handles QK norm + RoPE in a single pass. It falls back to separate
    F.rms_norm + apply_rotary_emb calls when disabled, and is disabled under
    tensor parallelism because the fused op currently requires TP=1
    (apply_packed_qk_norm_rope asserts tp_size == 1).
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
        config: Optional[DiffusionModelConfig] = None,
        layer_idx: int = 0,
        module_name: Optional[str] = None,
    ):
        # Opt in to the fused DiT QK-norm + RoPE kernel (per-head template), but
        # only when TP=1: the fused op asserts tp_size == 1
        # (apply_packed_qk_norm_rope), so under TP>1 we fall back to the unfused
        # F.rms_norm + apply_rotary_emb path. Mirrors WAN's gating.
        tp_size = config.mapping.tp_size if config and config.mapping else 1
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            qkv_mode=QKVMode.FUSE_QKV,
            qk_norm=True,
            qk_norm_mode="per_head",
            eps=eps,
            bias=bias,
            interleave=True,
            fuse_qk_norm_rope=(tp_size == 1),
            config=config,
            layer_idx=layer_idx,
            module_name=module_name,
        )

        self.pre_only = pre_only
        self.added_kv_proj_dim = added_kv_proj_dim

        if self.pre_only:
            del self.to_out

        # Text-stream projections for joint attention (dual-stream blocks only)
        if added_kv_proj_dim is not None:
            self.add_qkv_proj = Linear(
                added_kv_proj_dim,
                self.q_dim + 2 * self.kv_dim,
                bias=self.bias,
                dtype=self.dtype,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR
                ),
                fused_weight_shard_indices_mapping={
                    "q": (0, self.local_q_dim),
                    "k": (self.local_q_dim, self.local_kv_dim),
                    "v": (self.local_q_dim + self.local_kv_dim, self.local_kv_dim),
                },
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                reduce_output=False,
                override_tp_sharding={
                    "q": (self.local_q_dim_start, self.local_q_dim_end),
                    "k": (self.local_kv_dim_start, self.local_kv_dim_end),
                    "v": (self.local_kv_dim_start, self.local_kv_dim_end),
                },
            )

            # Need not pass any mapping info since this is intra-head normalization
            # Hence it is unaffected by TP which only changes cross-head work
            self.norm_added_q = RMSNorm(
                hidden_size=head_dim,
                eps=eps,
                dtype=self.dtype,
                has_weights=True,
            )
            self.norm_added_k = RMSNorm(
                hidden_size=head_dim,
                eps=eps,
                dtype=self.dtype,
                has_weights=True,
            )

            self.to_add_out = Linear(
                self.kv_dim,
                added_kv_proj_dim,
                bias=self.bias,
                dtype=self.dtype,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                mapping=config.mapping,
                allreduce_strategy=config.allreduce_strategy,
                tensor_parallel_mode=TensorParallelMode.ROW,
                reduce_output=True,
                override_tp_sharding=(self.local_kv_dim_start, self.local_kv_dim_end),
            )

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override: use F.rms_norm for per-head norm
        - ~1.6× speedup on Flux.2 inputs
        - Better fusion with torch.compile
        """
        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)
        return q, k

    def apply_qk_added_norm(
        self, enc_q: torch.Tensor, enc_k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Override: Apply F.rms_norm to text-stream QK,
        - ~1.6× speedup on Flux.2 inputs
        - Better fusion with torch.compile
        """
        enc_q = F.rms_norm(
            enc_q, (enc_q.shape[-1],), self.norm_added_q.weight, self.norm_added_q.variance_epsilon
        )
        enc_k = F.rms_norm(
            enc_k, (enc_k.shape[-1],), self.norm_added_k.weight, self.norm_added_k.variance_epsilon
        )
        return enc_q, enc_k

    def _prepare_qkv_unfused(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare Q, K, V with separate norm + RoPE (unfused path)."""
        batch_size = hidden_states.shape[0]
        is_dual_stream = encoder_hidden_states is not None and self.added_kv_proj_dim is not None

        query, key, value = self.get_qkv(hidden_states)
        query = query.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)
        key = key.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)
        value = value.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)

        query, key = self.apply_qk_norm(query, key)

        if is_dual_stream:
            encoder_qkv = self.add_qkv_proj(encoder_hidden_states)
            enc_q, enc_k, enc_v = encoder_qkv.chunk(3, dim=-1)
            enc_q = enc_q.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)
            enc_k = enc_k.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)
            enc_v = enc_v.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)

            enc_q, enc_k = self.apply_qk_added_norm(enc_q, enc_k)

            query = torch.cat([enc_q, query], dim=1)
            key = torch.cat([enc_k, key], dim=1)
            value = torch.cat([enc_v, value], dim=1)

        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            query = apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb(key, freqs_cos, freqs_sin)

        return query.flatten(2), key.flatten(2), value.flatten(2)

    def _prepare_qkv_fused(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare Q, K, V with fused QK Norm + RoPE CUDA kernel."""
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

        q_add = self.norm_added_q.weight if hasattr(self, "norm_added_q") else None
        k_add = self.norm_added_k.weight if hasattr(self, "norm_added_k") else None
        self.apply_packed_qk_norm_rope(
            qkv,
            freqs_cos,
            freqs_sin,
            num_txt_tokens=num_txt,
            q_add_weight=q_add,
            k_add_weight=k_add,
        )

        query, key, value = qkv.split(
            [self.local_q_dim, self.local_kv_dim, self.local_kv_dim], dim=-1
        )
        return query, key, value

    def _prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to fused or unfused QKV preparation."""
        if self.fuse_qk_norm_rope and image_rotary_emb is not None:
            return self._prepare_qkv_fused(hidden_states, encoder_hidden_states, image_rotary_emb)
        return self._prepare_qkv_unfused(hidden_states, encoder_hidden_states, image_rotary_emb)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep: Optional[torch.Tensor] = None,
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
        is_dual_stream = encoder_hidden_states is not None and self.added_kv_proj_dim is not None
        txt_seq_len = encoder_hidden_states.shape[1] if is_dual_stream else 0

        query, key, value = self._prepare_qkv(
            hidden_states, encoder_hidden_states, image_rotary_emb
        )

        hidden_states = self._attn_impl(query, key, value, timestep=timestep)
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

    _SWIGLU_FP4_TILE_SIZE = 128
    _SWIGLU_WEIGHT_INTERLEAVE_SIZE = 128

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int = 128,
        mlp_ratio: float = 3.0,
        bias: bool = False,
        eps: float = 1e-6,
        config: Optional[DiffusionModelConfig] = None,
        layer_idx: int = 0,
        module_name: Optional[str] = None,
    ):
        # self.tp_size is set in super().__init__
        tp_size = config.mapping.tp_size if config and config.mapping else 1

        # Set MLP dims BEFORE super().__init__() — _init_qkv_proj() needs them
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.local_mlp_hidden_dim = self.mlp_hidden_dim // tp_size
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
            module_name=module_name,
        )

        # Output projection needs FULL dims (ROW parallel divides internally)
        self.to_out = FluxJointAttnMLPProj(
            attn_dim=self.q_dim,
            mlp_dim=self.mlp_hidden_dim,
            out_dim=hidden_size,
            bias=bias,
            dtype=self.dtype,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
            config=config,
            attn_shard=(self.local_q_dim_start, self.local_q_dim_end),
        )

    def _init_qkv_proj(self):
        """Override: fused QKV+MLP projection instead of standard QKV."""
        mlp_in_dim = self.mlp_hidden_dim * self.mlp_mult_factor
        local_mlp_hidden_start = Linear._calc_shard(
            self.mlp_hidden_dim, self.mapping.tp_size, self.mapping.tp_rank
        )
        local_mlp_hidden_end = Linear._calc_shard(
            self.mlp_hidden_dim, self.mapping.tp_size, self.mapping.tp_rank + 1
        )
        self.local_mlp_hidden_dim = local_mlp_hidden_end - local_mlp_hidden_start
        use_cute_dsl_swiglu = (
            torch.cuda.is_available()
            and is_sm_100f()
            and getattr(self.quant_config, "quant_algo", None) == QuantAlgo.NVFP4
            and not self.bias
            and self._is_cute_dsl_swiglu_layout_compatible(
                self.mapping.tp_size,
                self.local_mlp_hidden_dim * self.mlp_mult_factor,
                self.local_mlp_hidden_dim,
            )
        )
        self.to_qkv_mlp_proj = FluxJointQKVMLPProj(
            in_dim=self.hidden_size,
            q_dim=self.q_dim,
            kv_dim=self.kv_dim,
            mlp_dim=mlp_in_dim,
            bias=self.bias,
            dtype=self.dtype,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=use_cute_dsl_swiglu,
            mapping=self.mapping,
            override_qkv_sharding={
                "q": (self.local_q_dim_start, self.local_q_dim_end),
                "k": (self.local_kv_dim_start, self.local_kv_dim_end),
                "v": (self.local_kv_dim_start, self.local_kv_dim_end),
            },
        )

    @staticmethod
    def _is_cute_dsl_swiglu_layout_compatible(
        tp_size: int,
        gate_up_out_features: int,
        down_in_features: int,
    ) -> bool:
        return (
            tp_size > 1
            and gate_up_out_features % Flux2ParallelSelfAttention._SWIGLU_WEIGHT_INTERLEAVE_SIZE
            == 0
            and down_in_features % 2 == 0
            and gate_up_out_features // 4 == down_in_features // 2
        )

    def _apply_norm_rope_unfused(
        self,
        qkv: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply separate norm + RoPE to packed QKV (unfused path)."""
        batch_size = qkv.shape[0]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)
        k = k.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)
        v = v.view(batch_size, -1, self.local_num_attention_heads, self.head_dim)

        q, k = self.apply_qk_norm(q, k)

        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        return q.flatten(2), k.flatten(2), v.flatten(2)

    def _apply_norm_rope_fused(
        self,
        qkv: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply fused QK Norm + RoPE kernel to packed QKV."""
        if image_rotary_emb is None:
            raise ValueError("Fused QK Norm + RoPE kernel requires image_rotary_emb")
        freqs_cos, freqs_sin = image_rotary_emb

        # torch.split produces non-contiguous views; fused kernel requires contiguous
        qkv = qkv.contiguous()

        self.apply_packed_qk_norm_rope(qkv, freqs_cos, freqs_sin, num_txt_tokens=-1)

        q, k, v = qkv.split([self.local_q_dim, self.local_kv_dim, self.local_kv_dim], dim=-1)
        return q, k, v

    def _apply_norm_rope(
        self,
        qkv: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to fused or unfused norm + RoPE."""
        if self.fuse_qk_norm_rope and image_rotary_emb is not None:
            return self._apply_norm_rope_fused(qkv, image_rotary_emb)
        return self._apply_norm_rope_unfused(qkv, image_rotary_emb)

    def _can_project_mlp_out_from_fp4(self) -> bool:
        if self.to_qkv_mlp_proj.tp_size <= 1 or not hasattr(self.to_out, "mlp_proj"):
            return False
        if not torch.cuda.is_available() or not is_sm_100f():
            return False

        mlp_proj = self.to_out.mlp_proj
        if not getattr(mlp_proj, "_weights_created", False):
            return False

        return (
            mlp_proj.has_nvfp4
            and mlp_proj.input_scale is not None
            and mlp_proj.pre_quant_scale is None
            and not mlp_proj.force_dynamic_quantization
        )

    def _can_project_hidden_mlp_with_cute_dsl(self) -> bool:
        if (
            self.to_qkv_mlp_proj.tp_size <= 1
            or not hasattr(self.to_qkv_mlp_proj, "qkv_proj")
            or not hasattr(self.to_qkv_mlp_proj, "mlp_proj")
            or not hasattr(self.to_out, "mlp_proj")
        ):
            return False
        if not torch.cuda.is_available() or not is_sm_100f():
            return False

        gate_up_proj = self.to_qkv_mlp_proj.mlp_proj
        down_proj = self.to_out.mlp_proj
        if not getattr(gate_up_proj, "_weights_created", False) or not getattr(
            down_proj, "_weights_created", False
        ):
            return False

        return (
            gate_up_proj.use_cute_dsl_blockscaling_mm
            and gate_up_proj.has_nvfp4
            and not gate_up_proj.has_bias
            and self._is_cute_dsl_swiglu_layout_compatible(
                self.to_qkv_mlp_proj.tp_size,
                gate_up_proj.out_features,
                down_proj.in_features,
            )
        )

    def _can_project_hidden_mlp_with_fp4out(self, hidden_states: torch.Tensor) -> bool:
        if not self._can_project_hidden_mlp_with_cute_dsl():
            return False

        gate_up_proj = self.to_qkv_mlp_proj.mlp_proj
        down_proj = self.to_out.mlp_proj

        if not torch.compiler.is_compiling():
            num_tokens = hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0]
            if num_tokens < Flux2ParallelSelfAttention._SWIGLU_FP4_TILE_SIZE:
                return False

        return (
            gate_up_proj.has_nvfp4
            and not gate_up_proj.has_bias
            and gate_up_proj.input_scale is not None
            and gate_up_proj.pre_quant_scale is None
            and not gate_up_proj.force_dynamic_quantization
            and down_proj.has_nvfp4
            and down_proj.input_scale is not None
            and down_proj.pre_quant_scale is None
            and not down_proj.force_dynamic_quantization
        )

    def _project_hidden_mlp_with_cute_dsl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up_proj = self.to_qkv_mlp_proj.mlp_proj
        down_proj = self.to_out.mlp_proj

        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        act_fp4, act_sf, alpha = gate_up_proj.quant_method._input_prepare(
            gate_up_proj, hidden_states
        )
        if self._can_project_hidden_mlp_with_fp4out(hidden_states):
            mlp_fp4, mlp_sf = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_fp4out_blackwell(
                act_fp4,
                gate_up_proj.weight,
                act_sf,
                gate_up_proj.weight_scale,
                alpha,
                down_proj.input_scale,
            )
            mlp_fp4 = mlp_fp4.reshape(*original_shape[:-1], mlp_fp4.shape[-1])
            return down_proj(Fp4QuantizedTensor(mlp_fp4, mlp_sf))

        mlp_hidden = torch.ops.trtllm.cute_dsl_nvfp4_dense_gemm_swiglu_blackwell(
            act_fp4,
            gate_up_proj.weight,
            act_sf,
            gate_up_proj.weight_scale,
            alpha,
            gate_up_proj.dtype,
        )
        expected_out = gate_up_proj.out_features // 2
        if mlp_hidden.shape[-1] > expected_out:
            mlp_hidden = mlp_hidden[..., :expected_out].contiguous()
        mlp_hidden = mlp_hidden.reshape(*original_shape[:-1], mlp_hidden.shape[-1])
        return down_proj(mlp_hidden)

    def _combine_split_projection(
        self,
        attn_out: torch.Tensor,
        mlp_projected: torch.Tensor,
    ) -> torch.Tensor:
        out = self.to_out.allreduce(self.to_out.attn_proj(attn_out) + mlp_projected)
        if self.to_out.has_bias:
            out = out + self.to_out.bias
        return out

    def _project_split_output_with_fp4_mlp(
        self,
        attn_out: torch.Tensor,
        mlp_hidden: torch.Tensor,
    ) -> torch.Tensor:
        shape = mlp_hidden.shape
        gate, up = mlp_hidden.chunk(2, dim=-1)
        mlp_hidden = torch.cat((up, gate), dim=-1)
        mlp_hidden = mlp_hidden.reshape(-1, shape[-1])
        tile_size = Flux2ParallelSelfAttention._SWIGLU_FP4_TILE_SIZE
        num_tokens = mlp_hidden.shape[0]
        num_tiles = (num_tokens + tile_size - 1) // tile_size
        padded_tokens = num_tiles * tile_size
        if padded_tokens != num_tokens:
            mlp_hidden = F.pad(mlp_hidden, (0, 0, 0, padded_tokens - num_tokens))

        tile_idx_to_mn_limit = (
            torch.arange(1, num_tiles + 1, dtype=torch.int32, device=mlp_hidden.device) * tile_size
        )
        num_non_exiting_tiles = torch.tensor(
            [num_tiles], dtype=torch.int32, device=mlp_hidden.device
        )
        mlp_fp4, mlp_sf = torch.ops.trtllm.moe_swiglu_nvfp4_quantize(
            mlp_hidden,
            self.to_out.mlp_proj.input_scale,
            tile_idx_to_mn_limit,
            num_non_exiting_tiles,
            tile_size,
        )
        mlp_out = self.to_out.mlp_proj(Fp4QuantizedTensor(mlp_fp4, mlp_sf))
        if padded_tokens != num_tokens:
            mlp_out = mlp_out[:num_tokens]
        mlp_out = mlp_out.reshape(*shape[:-1], mlp_out.shape[-1])
        return self._combine_split_projection(attn_out, mlp_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, dim]
            attention_mask: Optional attention mask
            image_rotary_emb: Tuple of (freqs_cos, freqs_sin)

        Returns:
            hidden_states [batch, seq, dim]
        """
        if self._can_project_hidden_mlp_with_cute_dsl():
            qkv = self.to_qkv_mlp_proj.qkv_proj(hidden_states)
            q, k, v = self._apply_norm_rope(qkv, image_rotary_emb)
            attn_out = self._attn_impl(q, k, v, timestep=timestep)
            attn_out = attn_out.to(q.dtype)
            mlp_out = self._project_hidden_mlp_with_cute_dsl(hidden_states)
            return self._combine_split_projection(attn_out, mlp_out)

        # Fused QKV + MLP projection
        qkv, mlp_hidden = self.to_qkv_mlp_proj(hidden_states)

        q, k, v = self._apply_norm_rope(qkv, image_rotary_emb)

        attn_out = self._attn_impl(q, k, v, timestep=timestep)
        attn_out = attn_out.to(q.dtype)

        # Parallel MLP path (reshape to 2D for Triton kernel, then back)
        shape = mlp_hidden.shape
        if self._can_project_mlp_out_from_fp4():
            return self._project_split_output_with_fp4_mlp(attn_out, mlp_hidden)

        mlp_out = swiglu(mlp_hidden.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)

        # Concatenate + project
        return self.to_out(attn_out, mlp_out)
