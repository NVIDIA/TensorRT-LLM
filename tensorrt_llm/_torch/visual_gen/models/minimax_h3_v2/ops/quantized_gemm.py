# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.custom_ops.torch_custom_ops import BufferKind
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import Fp4Quantize


class TunableFp4Quantize(Fp4Quantize):
    CELLS = (
        Cell(
            why="Unaligned token count and swizzled activation scales", spec=dict(shape=(17, 256))
        ),
    )

    def __call__(
        self,
        input: torch.Tensor,
        global_scale: torch.Tensor,
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, scales = torch.ops.trtllm.tunable_fp4_quantize(
            input, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
        )
        return q, scales


class Fp8ScaledGemm(OpWrapper):
    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (
        Cell(why="Packed activation rows with a per-tensor scale", spec=dict(m=17, n=128, k=128)),
    )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale_x: torch.Tensor,
        scale_w: torch.Tensor,
        dtype: torch.dtype,
        cuda_core: bool = False,
        rubin: bool = False,
        output_buffer_kind: int = int(BufferKind.DEFAULT),
        group: list[int] | None = None,
    ) -> torch.Tensor:
        if rubin:
            return torch.ops.trtllm.cute_dsl_fp8_per_tensor_gemm_rubin(
                x, weight, input_scale=scale_x, weight_scale=scale_w, output_dtype=dtype
            )
        mm = torch.ops.trtllm.cuda_scaled_mm if cuda_core else torch.ops.trtllm.cublas_scaled_mm
        return mm(
            x,
            weight.t(),
            scale_a=scale_x,
            scale_b=scale_w,
            bias=None,
            out_dtype=dtype,
            output_buffer_kind=output_buffer_kind,
            group=group,
        )

    def reference(
        self,
        x,
        weight,
        scale_x,
        scale_w,
        dtype,
        cuda_core=False,
        rubin=False,
        output_buffer_kind=int(BufferKind.DEFAULT),
        group=None,
    ):
        with true_fp32_matmul():
            return ((x.float() @ weight.float().t()) * scale_x * scale_w).to(dtype)

    def is_valid(
        self,
        x,
        weight,
        scale_x,
        scale_w,
        dtype,
        cuda_core=False,
        rubin=False,
        output_buffer_kind=int(BufferKind.DEFAULT),
        group=None,
    ):
        assert x.shape[-1] == weight.shape[-1]
        assert scale_x.numel() == scale_w.numel() == 1


tunable_fp4_quantize = TunableFp4Quantize()
fp8_scaled_gemm = Fp8ScaledGemm()


class Fp8BlockGemm(OpWrapper):
    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (Cell(why="Two K scale blocks and an odd token count", spec=dict(m=17, k=256, n=128)),)

    def __call__(self, x, weight, weight_scale, packed_sm100, packed_sm120, low_m_limit):
        from tensorrt_llm.quantization.utils.fp8_utils import per_token_quant_and_transform

        if isinstance(x, tuple):
            x, scale = x
            return torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(
                x, scale, weight, weight_scale, disable_ue8m0_cast=True
            )
        if packed_sm100:
            if 1 <= x.shape[0] <= low_m_limit and x.is_contiguous():
                return torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, weight, weight_scale)
            return torch.ops.trtllm.fp8_swap_ab_gemm(
                x, weight, weight_scale, output_dtype=torch.bfloat16, disable_ue8m0_cast=True
            )
        if packed_sm120:
            act, scale = per_token_quant_and_transform(x)
        else:
            act, scale = torch.ops.trtllm.fp8_quantize_1x128(x)
        return torch.ops.trtllm.fp8_block_scaling_gemm(act, weight, scale, weight_scale)

    def reference(self, x, weight, weight_scale, packed_sm100, packed_sm120, low_m_limit):
        from tensorrt_llm.quantization.utils.fp8_utils import (
            unpack_col_major_tma_aligned_packed_tensor,
        )

        n, k = weight.shape
        k_blocks = (k + 127) // 128
        if packed_sm100 or packed_sm120:
            scale = unpack_col_major_tma_aligned_packed_tensor(weight_scale, n, k_blocks)
        else:
            scale = weight_scale.repeat_interleave(128, dim=0)[:n]
        w = weight.float() * scale.repeat_interleave(128, dim=1)[:, :k]
        if isinstance(x, tuple):
            xq, xs = x
            xs = unpack_col_major_tma_aligned_packed_tensor(xs, xq.shape[0], k_blocks)
            x = xq.float() * xs.repeat_interleave(128, dim=1)[:, :k]
        with true_fp32_matmul():
            return (x.float() @ w.t()).to(torch.bfloat16)

    def is_valid(self, x, weight, weight_scale, packed_sm100, packed_sm120, low_m_limit):
        xvals = x[0] if isinstance(x, tuple) else x
        assert xvals.shape[-1] == weight.shape[-1]
        if isinstance(x, tuple):
            assert packed_sm100 and x[1].dtype == torch.int32


fp8_block_gemm = Fp8BlockGemm()
