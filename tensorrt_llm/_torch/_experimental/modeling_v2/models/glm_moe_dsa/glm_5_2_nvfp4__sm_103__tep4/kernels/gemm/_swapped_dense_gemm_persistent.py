# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared layout adapter for GLM BF16 persistent projections.

The high-performance device implementation lives in
``dense_gemm_persistent.py``.  This helper
only specializes that implementation.  A schedule can use the conventional
row-major projection layout or swap operands so the column-major epilogue
writes public row-major ``[M, N]`` storage without a transpose kernel.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import testing
from cutlass.utils import create_cute_tensor_for_fp8

from . import dense_gemm_persistent as _DENSE_GEMM

CompiledBmm = Callable[[cute.Tensor, cute.Tensor, cute.Tensor, cuda.CUstream], None]


@dataclass(frozen=True)
class GemmSchedule:
    """Compile-time schedule for one persistent GEMM instance."""

    use_2cta_instrs: bool
    mma_tiler_mn: tuple[int, int]
    cluster_shape_mn: tuple[int, int]
    orientation: Literal["direct", "swapped"] = "swapped"
    swizzle_size: int = 1
    raster_along: Literal["m", "n"] = "m"
    use_tma_store: bool = True


@dataclass(frozen=True)
class CompiledPersistentGemm:
    """Launch one compiled projection schedule."""

    input_shape: torch.Size
    weight_shape: torch.Size
    output_shape: torch.Size
    orientation: Literal["direct", "swapped"]
    compiled_bmm: CompiledBmm

    def __call__(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Launch the projection on the current PyTorch CUDA stream.

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
        a, b, c = _make_runtime_arguments(
            input_tensor,
            weight,
            output,
            self.orientation,
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.compiled_bmm(a, b, c, stream)


def _check_glm_mma_tiler_and_cluster_shape(self) -> None:
    """Admit the measured SM103 BF16 MMA-N multiples of eight."""
    valid_m = (not self.use_2cta_instrs and self.mma_tiler_mn[0] in (64, 128)) or (
        self.use_2cta_instrs and self.mma_tiler_mn[0] in (128, 256)
    )
    if not valid_m:
        raise testing.CantImplementError(f"invalid MMA-M: {self.mma_tiler_mn}")
    if self.mma_tiler_mn[1] not in range(8, 257, 8):
        raise testing.CantImplementError(f"invalid MMA-N: {self.mma_tiler_mn}")
    cluster_m, cluster_n = self.cluster_shape_mn
    if cluster_m % (2 if self.use_2cta_instrs else 1) != 0:
        raise testing.CantImplementError(f"invalid cluster M: {self.cluster_shape_mn}")
    if (
        cluster_m * cluster_n > 16
        or cluster_m <= 0
        or cluster_n <= 0
        or cluster_m & (cluster_m - 1)
        or cluster_n & (cluster_n - 1)
    ):
        raise testing.CantImplementError(f"invalid cluster shape: {self.cluster_shape_mn}")


class _GlmPersistentDenseGemmKernel(_DENSE_GEMM.PersistentDenseGemmKernel):
    """Extend the example's conservative MMA-N gate for measured GLM tiles."""

    check_mma_tiler_and_cluster_shape = _check_glm_mma_tiler_and_cluster_shape


def _make_runtime_arguments(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    orientation: Literal["direct", "swapped"],
) -> tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
    """Create dynamic-layout views for one projection orientation."""
    # TRT-LLM constructs weights with requires_grad=True.  CuTe's DLPack
    # bridge requires a non-autograd view; detach is zero-copy and leaves the
    # original Parameter ownership and storage unchanged.
    weight = weight.detach()
    if orientation == "swapped":
        a = create_cute_tensor_for_fp8(weight.unsqueeze(0), cutlass.BFloat16, 2)
        b = create_cute_tensor_for_fp8(
            input_tensor.unsqueeze(0).transpose(1, 2), cutlass.BFloat16, 1
        )
        c = create_cute_tensor_for_fp8(output.unsqueeze(0).transpose(1, 2), cutlass.BFloat16, 1)
    elif orientation == "direct":
        a = create_cute_tensor_for_fp8(input_tensor.unsqueeze(0), cutlass.BFloat16, 2)
        b = create_cute_tensor_for_fp8(weight.t().unsqueeze(0), cutlass.BFloat16, 1)
        c = create_cute_tensor_for_fp8(output.unsqueeze(0), cutlass.BFloat16, 2)
    else:
        raise ValueError(f"unsupported GEMM orientation: {orientation}")
    return a, b, c


def _problem_shape_and_majors(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    orientation: Literal["direct", "swapped"],
) -> tuple[tuple[int, int, int, int], tuple[str, str, str]]:
    """Return compile-time problem shape and operand major modes."""
    active_m, input_features = input_tensor.shape
    output_features = weight.shape[0]
    if orientation == "swapped":
        return (output_features, active_m, input_features, 1), ("k", "k", "m")
    if orientation == "direct":
        return (active_m, output_features, input_features, 1), ("k", "k", "n")
    raise ValueError(f"unsupported GEMM orientation: {orientation}")


def compile_persistent_gemm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    schedule: GemmSchedule,
    *,
    kernel_name_prefix: str | None = None,
) -> CompiledPersistentGemm:
    """Compile from real tensor layouts during adapter construction.

    Compilation uses caller-owned tensors only for their types, shapes, and
    dynamic strides.  Runtime pointers remain arguments to the returned
    callable, so the compiled instance is safe for CUDA Graph replay.
    """
    active_m, input_features = input_tensor.shape
    output_features, weight_k = weight.shape
    if weight_k != input_features or output.shape != (active_m, output_features):
        raise ValueError(
            "incompatible projection shapes: "
            f"input={input_tensor.shape}, weight={weight.shape}, output={output.shape}"
        )
    mnkl, majors = _problem_shape_and_majors(
        input_tensor,
        weight,
        schedule.orientation,
    )
    a, b, c = _make_runtime_arguments(
        input_tensor,
        weight,
        output,
        schedule.orientation,
    )
    cluster_size = schedule.cluster_shape_mn[0] * schedule.cluster_shape_mn[1]
    max_active_clusters = _DENSE_GEMM.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    gemm = _GlmPersistentDenseGemmKernel(
        cutlass.Float32,
        schedule.use_2cta_instrs,
        schedule.mma_tiler_mn,
        schedule.cluster_shape_mn,
        schedule.use_tma_store,
        schedule.swizzle_size,
        schedule.raster_along,
    )
    if not gemm.can_implement(
        mnkl,
        a.element_type,
        b.element_type,
        c.element_type,
        *majors,
    ):
        raise testing.CantImplementError(
            "persistent GLM projection cannot implement "
            f"mnkl={mnkl}, majors={majors}, schedule={schedule}"
        )
    if kernel_name_prefix is not None:
        gemm.kernel.set_name_prefix(
            kernel_name_prefix,
            remove_cutlass_symbol=True,
        )
    from cutlass.cute.runtime import make_fake_stream

    compiled_bmm = cute.compile(
        _DENSE_GEMM.bmm,
        gemm,
        a,
        b,
        c,
        max_active_clusters,
        make_fake_stream(),
        lambda x: x,
    )
    return CompiledPersistentGemm(
        input_tensor.shape,
        weight.shape,
        output.shape,
        schedule.orientation,
        compiled_bmm,
    )
