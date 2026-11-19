# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-load down GEMM/residual fusion, restricted to the qualified full request."""

import breuse_down_kernel as kernel
import breuse_gateup
import fp8_producers as producers
import residual_kernel
import torch

ACTIVE = None


def baseline(a, weight, input_scale, weight_scale, x, mod, indices):
    value = torch.ops.trtllm.cublas_scaled_mm(
        a,
        weight.t(),
        scale_a=input_scale,
        scale_b=weight_scale,
        bias=None,
        out_dtype=torch.bfloat16,
    )
    return residual_kernel.compiled_expression(x, value.reshape_as(x), mod, indices)


@torch.library.custom_op("h3_fp8_long3::down_residual", mutates_args=())
def down_residual(
    a: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    x: torch.Tensor,
    mod: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    runtime = ACTIVE
    if runtime is None:
        raise RuntimeError("Install the post-load down runtime first")
    if not runtime.supports(a, weight, x, mod, indices):
        runtime.fallback_calls += 1
        return baseline(a, weight, input_scale, weight_scale, x, mod, indices)
    runtime.calls += 1
    return kernel.run(a, weight, input_scale, weight_scale, x, mod, indices).reshape_as(x)


@down_residual.register_fake
def _(a, weight, input_scale, weight_scale, x, mod, indices):
    return torch.empty_like(x)


def forward(
    self, hidden_states, temb, adaln_indices, rotary_emb, key_padding_mask, timestep, valid_seq_len
):
    mod = self.adaln_proj.table(temb)
    quantized = producers.norm(
        hidden_states,
        self.norm1.weight,
        mod,
        adaln_indices,
        self.attn.qkv_proj.input_scale,
        self.norm1.variance_epsilon,
    )
    attn_output = self.attn(quantized, rotary_emb, key_padding_mask, timestep, valid_seq_len)
    hidden_states, quantized = producers.gate_norm(
        hidden_states,
        attn_output,
        self.norm2.weight,
        mod,
        adaln_indices,
        self.ff.gate_up_proj.input_scale,
        self.norm2.variance_epsilon,
    )
    up, down = self.ff.gate_up_proj, self.ff.down_proj
    codes = breuse_gateup.gateup_swiglu(
        quantized.reshape(-1, quantized.shape[-1]),
        up.weight,
        up.input_scale,
        up.weight_scale,
        down.input_scale,
    )
    return down_residual(
        codes,
        down.weight,
        down.input_scale,
        down.weight_scale,
        hidden_states,
        mod,
        adaln_indices,
    )


class DownRuntime:
    """Preserve the native small-M GEMM path; no prepacked weight allocation."""

    def __init__(self, blocks):
        global ACTIVE
        from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
            modeling,
        )

        self.scenario = modeling
        self.original = modeling.ScenarioBlock._forward_fused_adaln
        self.calls = 0
        self.fallback_calls = 0
        for block in blocks:
            down = block.ff.down_proj
            if not (
                block.ff.mapping.tp_size == 1
                and down.bias is None
                and down.weight.shape == (5376, 14336)
                and down.weight.dtype == torch.float8_e4m3fn
                and down.weight.is_contiguous()
            ):
                raise ValueError("Expected the original bias-free static-FP8 H3 down projection")
        ACTIVE = self

    @staticmethod
    def supports(a, weight, x, mod, indices):
        # Native cuBLAS chooses a different reduction at tiny M. Tail sanitizer
        # checks prove memory safety under the full-M contract, not numerical
        # equivalence to that separate small-M GEMM. Keep its native fallback.
        return (
            a.shape == (37820, 14336)
            and weight.shape == (5376, 14336)
            and a.dtype == weight.dtype == torch.float8_e4m3fn
            and x.dtype == torch.bfloat16
            and x.numel() == 37820 * 5376
            and a.is_contiguous()
            and weight.is_contiguous()
            and x.is_contiguous()
            and mod.ndim == 2
            and mod.shape[1] == 6 * 5376
            and mod.dtype in (torch.bfloat16, torch.float32)
            and mod.is_contiguous()
            and mod.data_ptr() % 32 == 0
            and mod.stride(0) * mod.element_size() % 32 == 0
            and indices.numel() == 37820
            and indices.is_contiguous()
            and indices.dtype in (torch.int32, torch.int64)
        )

    def configure(self, enabled):
        self.scenario.ScenarioBlock._forward_fused_adaln = forward if enabled else self.original
