# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-local static-FP8 producer experiment; product routing stays guarded."""

import fp8_producers as kernels
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
    modeling as scenario,
)

QKVMode = scenario.QKVMode
fusion_mode = scenario.fusion_mode
set_reduce_group = scenario.set_reduce_group


def _use_fused_fp8_qk(
    self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]
) -> bool:
    """Select the fused QK-norm + RoPE + FP8 cast for the per-tensor FP8 recipe.

    Operates on a packed BF16
    projection with equal Q / K head counts.
    """
    if fusion_mode() == "off" or self.qkv_mode != QKVMode.FUSE_QKV or not self.qk_norm:
        return False
    backend = getattr(self.attn, "inner_backend", self.attn)
    if getattr(backend, "quant_dtype", None) != "fp8":
        return False
    # Ulysses: the ranks hold shards of one sequence; the dynamic amax must be reduced over
    # them so the codes that travel through the all-to-all share one scale per tensor.
    if getattr(self.attn, "world_size", 1) > 1:
        set_reduce_group(self.attn.process_group)
    if self.local_num_attention_heads != self.local_num_key_value_heads:
        return False
    for norm in (self.norm_q, self.norm_k):
        if (
            getattr(norm, "allreduce", None) is not None
            or getattr(norm, "use_gemma", False)
            or norm.weight.dtype != torch.bfloat16
            or norm.weight.numel() != self.head_dim
        ):
            return False
    cos, sin = rotary_emb
    rot = cos.shape[-1]
    # The NVFP4 producer hands an Fp4QuantizedTensor (codes + scales): read device, rank and
    # sequence length from the codes; the projection output is BF16 either way.
    codes = getattr(hidden_states, "fp4_tensor", None)
    x = codes if codes is not None else hidden_states
    return (
        x.is_cuda
        and (codes is not None or hidden_states.dtype in (torch.bfloat16, torch.float8_e4m3fn))
        and x.ndim == 3
        and cos.shape == (x.shape[1], rot)
        and sin.shape == cos.shape
        and rot % 2 == 0
        and rot <= self.head_dim
        and (self.head_dim & (self.head_dim - 1)) == 0
        and not torch.is_grad_enabled()
    )


def forward(
    self, hidden_states, temb, adaln_indices, rotary_emb, key_padding_mask, timestep, valid_seq_len
):
    mod = self.adaln_proj.table(temb)
    quantized = kernels.norm(
        hidden_states,
        self.norm1.weight,
        mod,
        adaln_indices,
        self.attn.qkv_proj.input_scale,
        self.norm1.variance_epsilon,
    )
    attn_output = self.attn(quantized, rotary_emb, key_padding_mask, timestep, valid_seq_len)
    hidden_states, quantized = kernels.gate_norm(
        hidden_states,
        attn_output,
        self.norm2.weight,
        mod,
        adaln_indices,
        self.ff.gate_up_proj.input_scale,
        self.norm2.variance_epsilon,
    )
    ff_output = self.ff(quantized.reshape(-1, quantized.shape[-1])).reshape_as(hidden_states)
    width = hidden_states.shape[-1]
    return hidden_states + mod[:, 5 * width : 6 * width].index_select(0, adaln_indices) * ff_output


def install():
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP

    original_activation = GatedMLP._apply_activation

    def activation(self, x, *, has_lora=False):
        if (
            self.down_proj.has_fp8_qdq
            and not has_lora
            and self._is_plain_swiglu()
            and self.swiglu_limit is None
            and self.down_proj.input_scale is not None
        ):
            return kernels.swiglu(x, self.down_proj.input_scale)
        return original_activation(self, x, has_lora=has_lora)

    GatedMLP._apply_activation = activation
    scenario.ScenarioAttention._use_fused_fp8_qk = _use_fused_fp8_qk
    scenario.ScenarioBlock._forward_fused_adaln = forward
