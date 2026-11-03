# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed swapped-operand adapter for the low-latency Blackwell TGV GEMM."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.utils import create_cute_tensor_for_fp8

from . import tgv_gemm_dsl as _TGV_GEMM

CompiledBmm = Callable[[cute.Tensor, cute.Tensor, cute.Tensor, None, cuda.CUstream], None]
TgvAbDtype = Literal["bf16", "tf32"]
TgvOutputDtype = Literal["bf16", "fp32"]


@dataclass(frozen=True)
class TgvElementSpec:
    """Storage and MMA element types for one compiled TGV instance."""

    ab_dtype: TgvAbDtype
    output_dtype: TgvOutputDtype


BF16_TGV_ELEMENTS = TgvElementSpec("bf16", "bf16")
BF16_FP32_TGV_ELEMENTS = TgvElementSpec("bf16", "fp32")


@dataclass(frozen=True)
class TgvSchedule:
    """Compile-time tile and pipeline schedule for one TGV instance."""

    cta_n: int
    cta_k: int
    num_ab_stage: int
    use_pdl: bool = True
    split_k: int = 1


@dataclass(frozen=True)
class CompiledSwappedTgvGemm:
    """Launch a compiled swapped-operand TGV projection."""

    input_shape: torch.Size
    weight_shape: torch.Size
    output_shape: torch.Size
    schedule: TgvSchedule
    element_spec: TgvElementSpec
    compiled_bmm: CompiledBmm

    def __call__(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Launch on the current PyTorch CUDA stream.

        :raises ValueError: If a runtime tensor shape differs from the shape
            used to compile this callable.
        """
        expected_shapes = (self.input_shape, self.weight_shape, self.output_shape)
        runtime_shapes = (input_tensor.shape, weight.shape, output.shape)
        if runtime_shapes != expected_shapes:
            raise ValueError(
                "compiled projection shape mismatch: "
                f"expected input/weight/output={expected_shapes}, "
                f"got {runtime_shapes}"
            )
        a, b, c = _make_runtime_arguments(input_tensor, weight, output, self.element_spec)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled_bmm(a, b, c, None, stream)


def _make_runtime_arguments(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    element_spec: TgvElementSpec,
) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
    """Create explicitly typed dynamic-layout views for ``weight @ input.T``."""
    expected_ab_dtype = torch.bfloat16 if element_spec.ab_dtype == "bf16" else torch.float32
    expected_output_dtype = torch.bfloat16 if element_spec.output_dtype == "bf16" else torch.float32
    if input_tensor.dtype != expected_ab_dtype or weight.dtype != expected_ab_dtype:
        raise ValueError(
            f"TGV {element_spec.ab_dtype} operands require {expected_ab_dtype}, got "
            f"input={input_tensor.dtype} weight={weight.dtype}"
        )
    if output.dtype != expected_output_dtype:
        raise ValueError(
            f"TGV {element_spec.output_dtype} output requires "
            f"{expected_output_dtype}, got {output.dtype}"
        )

    ab_dtype = cutlass.BFloat16 if element_spec.ab_dtype == "bf16" else cutlass.TFloat32
    output_dtype = cutlass.BFloat16 if element_spec.output_dtype == "bf16" else cutlass.Float32
    # TRT-LLM owns this tensor as an autograd-enabled Parameter. DLPack
    # requires a non-autograd view; detach is zero-copy and preserves storage.
    weight = weight.detach()
    a = create_cute_tensor_for_fp8(weight.unsqueeze(0), ab_dtype, 2)
    b = create_cute_tensor_for_fp8(input_tensor.unsqueeze(0).transpose(1, 2), ab_dtype, 1)
    c = create_cute_tensor_for_fp8(output.unsqueeze(0).transpose(1, 2), output_dtype, 1)
    return a, b, c


def _validate_schedule(
    input_features: int,
    schedule: TgvSchedule,
    element_spec: TgvElementSpec,
) -> None:
    """Reject configurations that the shared example checker does not cover."""
    if schedule.cta_n != 8 and schedule.cta_n % 16:
        raise ValueError("TGV cta_n must be 8 or a multiple of 16")
    if not 8 <= schedule.cta_n <= 256:
        raise ValueError("TGV cta_n must be in [8, 256]")
    if schedule.cta_k not in (32, 64, 128) or input_features % schedule.cta_k:
        raise ValueError(
            f"TGV cta_k={schedule.cta_k} must be 32/64/128 and divide K={input_features}"
        )
    if schedule.num_ab_stage <= 0 or not 1 <= schedule.split_k <= 16:
        raise ValueError("TGV stages must be positive and split_k must be in [1, 16]")

    values_per_thread = 64 * schedule.cta_n // 128
    if values_per_thread % schedule.split_k:
        raise ValueError(
            f"TGV split_k={schedule.split_k} must divide "
            f"{values_per_thread} output values per thread"
        )

    ab_bytes = 2 if element_spec.ab_dtype == "bf16" else 4
    staged_bytes = (64 + schedule.cta_n) * schedule.cta_k * ab_bytes * schedule.num_ab_stage
    shard_values = values_per_thread // schedule.split_k
    mailbox_bytes = (schedule.split_k - 1) * 128 * shard_values * cutlass.Float32.width // 8
    # The exact struct padding is layout-dependent.  Reserve 512 bytes for
    # barriers and alignment so an invalid schedule fails before CUDA launch.
    if staged_bytes + mailbox_bytes + 512 > 227 * 1024:
        raise ValueError(
            "TGV schedule exceeds the conservative 227 KiB SMEM budget: "
            f"staged={staged_bytes} mailbox={mailbox_bytes}"
        )


def compile_swapped_tgv_gemm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    schedule: TgvSchedule,
    element_spec: TgvElementSpec = BF16_TGV_ELEMENTS,
) -> CompiledSwappedTgvGemm:
    """Compile one exact-N TGV tile from real runtime-layout tensors."""
    active_m, input_features = input_tensor.shape
    output_features, weight_k = weight.shape
    if weight_k != input_features or output.shape != (active_m, output_features):
        raise ValueError(
            "incompatible projection shapes: "
            f"input={input_tensor.shape}, weight={weight.shape}, output={output.shape}"
        )
    _validate_schedule(input_features, schedule, element_spec)
    a, b, c = _make_runtime_arguments(input_tensor, weight, output, element_spec)
    compiled_bmm = _TGV_GEMM.compile_tgv_gemm(
        (output_features, active_m, input_features, 1),
        a,
        b,
        c,
        cutlass.Float32,
        cta_m=64,
        cta_n=schedule.cta_n,
        cta_k=schedule.cta_k,
        num_ab_stage=schedule.num_ab_stage,
        use_pdl=schedule.use_pdl,
        split_k=schedule.split_k,
        bias=None,
    )
    return CompiledSwappedTgvGemm(
        input_tensor.shape,
        weight.shape,
        output.shape,
        schedule,
        element_spec,
        compiled_bmm,
    )
