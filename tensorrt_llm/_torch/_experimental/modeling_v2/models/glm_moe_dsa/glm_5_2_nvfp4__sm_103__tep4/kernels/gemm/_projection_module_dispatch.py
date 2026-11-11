# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared build/cache glue for per-module GLM projection adapters.

Each public module file owns its exact TRT-LLM tensor ABI and measured
``SCHEDULE_BY_M`` catalog.  This helper only validates that catalog, compiles
the selected tcgen05/TGV/persistent backend, and returns the registry ABI::

    build() -> callable(input_tensor, weight, output)

The returned callable is pinned to the first M it sees.  The caller's registry
already caches one callable per ``(device, op, M, N, K, path, compile_fn)``, so
backend selection and compilation stay outside CUDA Graph replay.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

import cutlass
import torch

from . import _projection_tcgen05_template as tcgen05
from . import _swapped_dense_gemm_persistent as _swappeddensegemmpersiste
from . import _swapped_tgv_gemm as _swappedtgvgemm

CompiledPersistentGemm = _swappeddensegemmpersiste.CompiledPersistentGemm
GemmSchedule = _swappeddensegemmpersiste.GemmSchedule
compile_persistent_gemm = _swappeddensegemmpersiste.compile_persistent_gemm

BF16_FP32_TGV_ELEMENTS = _swappedtgvgemm.BF16_FP32_TGV_ELEMENTS
BF16_TGV_ELEMENTS = _swappedtgvgemm.BF16_TGV_ELEMENTS
CompiledSwappedTgvGemm = _swappedtgvgemm.CompiledSwappedTgvGemm
TgvElementSpec = _swappedtgvgemm.TgvElementSpec
TgvSchedule = _swappedtgvgemm.TgvSchedule
compile_swapped_tgv_gemm = _swappedtgvgemm.compile_swapped_tgv_gemm


INPUT_ALIGNMENT_BYTES = 128
WEIGHT_ALIGNMENT_BYTES = 128
OUTPUT_ALIGNMENT_BYTES = 32

Backend = Literal["tcgen05", "tgv", "persistent"]
CompiledKernel: TypeAlias = tcgen05.CompiledKernel | CompiledSwappedTgvGemm | CompiledPersistentGemm


class PersistentCompiler(Protocol):
    """Compile one owner-specific persistent GEMM implementation."""

    def __call__(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        schedule: GemmSchedule,
        *,
        kernel_name_prefix: str | None = None,
    ) -> CompiledPersistentGemm: ...


@dataclass(frozen=True)
class ProjectionAbi:
    """One fixed TRT-LLM local projection tensor contract."""

    module_name: str
    output_features: int
    input_features: int
    output_dtype: torch.dtype = torch.bfloat16
    tgv_elements: TgvElementSpec = BF16_TGV_ELEMENTS
    kernel_name_prefix: str | None = None
    pdl_persistent_compiler: PersistentCompiler | None = None
    pdl_enabled_m: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if self.output_dtype == torch.bfloat16:
            expected_tgv_elements = BF16_TGV_ELEMENTS
        elif self.output_dtype == torch.float32:
            expected_tgv_elements = BF16_FP32_TGV_ELEMENTS
        else:
            raise ValueError(
                f"{self.module_name}: output_dtype must be torch.bfloat16 or "
                f"torch.float32, got {self.output_dtype}"
            )
        if self.tgv_elements != expected_tgv_elements:
            raise ValueError(
                f"{self.module_name}: tgv_elements={self.tgv_elements} does not "
                f"match output_dtype={self.output_dtype}"
            )
        if self.kernel_name_prefix is not None and not self.kernel_name_prefix.startswith("glm52_"):
            raise ValueError(f"{self.module_name}: kernel_name_prefix must start with 'glm52_'")
        if bool(self.pdl_persistent_compiler) != bool(self.pdl_enabled_m):
            raise ValueError(
                f"{self.module_name}: pdl_persistent_compiler and "
                "pdl_enabled_m must be configured together"
            )


@dataclass(frozen=True)
class Tcgen05Schedule:
    """Compile-time schedule for the lightweight physical-M64 kernel."""

    mma_n: int
    stages: int
    split_k: int
    prewait_tma_operand: str = tcgen05.PREWAIT_TMA_W
    # Weight L2 prefetch depth in K steps; see ProjectionConfig. Off unless the
    # module opts in with a measured depth.
    prefetch_w_k_steps: int = 0


Schedule: TypeAlias = Tcgen05Schedule | TgvSchedule | GemmSchedule


@dataclass(frozen=True)
class ProjectionInstance:
    """Backend and compile-time schedule selected for one exact M."""

    backend: Backend
    schedule: Schedule


def tcgen05_instance(
    mma_n: int,
    stages: int,
    split_k: int,
    prewait_tma_operand: str = tcgen05.PREWAIT_TMA_W,
    *,
    prefetch_w_k_steps: int = 0,
) -> ProjectionInstance:
    """Construct one lightweight tcgen05 projection instance."""
    return ProjectionInstance(
        "tcgen05",
        Tcgen05Schedule(
            mma_n,
            stages,
            split_k,
            prewait_tma_operand,
            prefetch_w_k_steps=prefetch_w_k_steps,
        ),
    )


def tgv_instance(schedule: TgvSchedule) -> ProjectionInstance:
    """Construct one low-latency TGV projection instance."""
    return ProjectionInstance("tgv", schedule)


def persistent_instance(schedule: GemmSchedule) -> ProjectionInstance:
    """Construct one persistent projection instance."""
    return ProjectionInstance("persistent", schedule)


def context_instances(
    m1099: ProjectionInstance,
    m3282: ProjectionInstance,
    tail_75k: ProjectionInstance,
    m8192: ProjectionInstance,
) -> dict[int, ProjectionInstance]:
    """Expand the four context schedule regions to the context M buckets."""
    return {
        1_099: m1099,
        3_282: m3282,
        7_491: tail_75k,
        7_527: tail_75k,
        7_554: tail_75k,
        7_578: tail_75k,
        8_192: m8192,
    }


def _validate_catalog(
    abi: ProjectionAbi,
    schedule_by_m: Mapping[int, ProjectionInstance],
) -> None:
    if not schedule_by_m:
        raise ValueError(f"{abi.module_name}: SCHEDULE_BY_M must not be empty")
    for active_m, instance in schedule_by_m.items():
        if active_m <= 0:
            raise ValueError(f"{abi.module_name}: M must be positive, got {active_m}")
        expected_type: type[Tcgen05Schedule] | type[TgvSchedule] | type[GemmSchedule]
        if instance.backend == "tcgen05":
            expected_type = Tcgen05Schedule
            if active_m > tcgen05.MMA_M:
                raise ValueError(
                    f"{abi.module_name}: tcgen05 physical-M64 backend cannot serve M={active_m}"
                )
        elif instance.backend == "tgv":
            expected_type = TgvSchedule
        elif instance.backend == "persistent":
            expected_type = GemmSchedule
        else:
            raise ValueError(f"{abi.module_name}: unsupported backend {instance.backend!r}")
        if not isinstance(instance.schedule, expected_type):
            raise TypeError(
                f"{abi.module_name}: backend {instance.backend!r} requires "
                f"{expected_type.__name__}, got {type(instance.schedule).__name__}"
            )
    missing_pdl_m = abi.pdl_enabled_m - set(schedule_by_m)
    if missing_pdl_m:
        raise ValueError(
            f"{abi.module_name}: PDL M values are absent from SCHEDULE_BY_M: "
            f"{sorted(missing_pdl_m)}"
        )
    invalid_pdl_m = {
        active_m
        for active_m in abi.pdl_enabled_m
        if schedule_by_m[active_m].backend != "persistent"
    }
    if invalid_pdl_m:
        raise ValueError(
            f"{abi.module_name}: PDL requires the persistent backend at M values "
            f"{sorted(invalid_pdl_m)}"
        )


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def _validate_tensor(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, int],
    dtype: torch.dtype,
    alignment_bytes: int,
) -> None:
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or not tensor.is_cuda
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous CUDA {dtype} {shape}, got "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} contiguous={tensor.is_contiguous()}"
        )
    if tensor.data_ptr() % alignment_bytes:
        raise ValueError(f"{name} must be {alignment_bytes}-byte aligned")


def _validate_contract(
    abi: ProjectionAbi,
    supported_m: tuple[int, ...],
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> int:
    if input_tensor.ndim != 2:
        raise ValueError(f"{abi.module_name} input must be rank-2, got rank {input_tensor.ndim}")
    active_m = int(input_tensor.shape[0])
    if active_m not in supported_m:
        raise ValueError(f"{abi.module_name}: unsupported M={active_m}; expected {supported_m}")
    _validate_tensor(
        input_tensor,
        "input_tensor",
        (active_m, abi.input_features),
        torch.bfloat16,
        INPUT_ALIGNMENT_BYTES,
    )
    _validate_tensor(
        weight,
        "weight",
        (abi.output_features, abi.input_features),
        torch.bfloat16,
        WEIGHT_ALIGNMENT_BYTES,
    )
    _validate_tensor(
        output,
        "output",
        (active_m, abi.output_features),
        abi.output_dtype,
        OUTPUT_ALIGNMENT_BYTES,
    )
    if input_tensor.device != weight.device or output.device != weight.device:
        raise ValueError("input_tensor, weight, and output must share one CUDA device")
    return active_m


def _compile_instance(
    abi: ProjectionAbi,
    active_m: int,
    instance: ProjectionInstance,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> CompiledKernel:
    schedule = instance.schedule
    if instance.backend == "tcgen05":
        assert isinstance(schedule, Tcgen05Schedule)
        config = tcgen05.ProjectionConfig(
            module_name=abi.module_name,
            m=tcgen05.MMA_M,
            n=abi.output_features,
            k=abi.input_features,
            out_fp32=abi.output_dtype == torch.float32,
            mma_n=schedule.mma_n,
            stages=schedule.stages,
            split_k=schedule.split_k,
            standalone_perf="selected by per-module measured M dispatch",
            prewait_tma_operand=schedule.prewait_tma_operand,
            prefetch_w_k_steps=schedule.prefetch_w_k_steps,
            kernel_name_prefix_override=abi.kernel_name_prefix,
        )
        return tcgen05.compile_projection(
            config,
            active_m=active_m,
            output_alignment_bytes=OUTPUT_ALIGNMENT_BYTES,
        )
    if instance.backend == "tgv":
        assert isinstance(schedule, TgvSchedule)
        return compile_swapped_tgv_gemm(
            input_tensor,
            weight.detach(),
            output,
            schedule,
            abi.tgv_elements,
        )
    assert instance.backend == "persistent"
    assert isinstance(schedule, GemmSchedule)
    persistent_compiler = compile_persistent_gemm
    if active_m in abi.pdl_enabled_m:
        if abi.pdl_persistent_compiler is None:
            raise RuntimeError(f"{abi.module_name}: PDL compiler is missing for M={active_m}")
        persistent_compiler = abi.pdl_persistent_compiler
    return persistent_compiler(
        input_tensor,
        weight.detach(),
        output,
        schedule,
        kernel_name_prefix=abi.kernel_name_prefix,
    )


def make_build(
    abi: ProjectionAbi,
    schedule_by_m: Mapping[int, ProjectionInstance],
) -> Callable[[], CompiledKernel]:
    """Create the exact no-argument factory consumed by TRT-LLM's registry."""
    catalog = dict(schedule_by_m)
    _validate_catalog(abi, catalog)
    supported_m = tuple(catalog)
    compiled_by_key: dict[tuple[int, int, ProjectionInstance], CompiledKernel] = {}

    def build() -> CompiledKernel:
        compiled_fn: CompiledKernel | None = None
        pinned_m = -1

        def run_linear(
            input_tensor: torch.Tensor,
            weight: torch.Tensor,
            output: torch.Tensor,
        ) -> object:
            nonlocal compiled_fn, pinned_m
            if compiled_fn is None:
                active_m = _validate_contract(
                    abi,
                    supported_m,
                    input_tensor,
                    weight,
                    output,
                )
                instance = catalog[active_m]
                device_index = _device_index(input_tensor)
                key = (device_index, active_m, instance)
                compiled = compiled_by_key.get(key)
                if compiled is None:
                    cutlass.cuda.initialize_cuda_context(device_index)
                    compiled = _compile_instance(
                        abi,
                        active_m,
                        instance,
                        input_tensor,
                        weight,
                        output,
                    )
                    compiled_by_key[key] = compiled
                compiled_fn = compiled
                pinned_m = active_m
            elif input_tensor.shape[0] != pinned_m:
                raise ValueError(
                    f"{abi.module_name}: this build() instance is pinned to "
                    f"M={pinned_m}, got M={int(input_tensor.shape[0])}"
                )
            return compiled_fn(input_tensor, weight, output)

        return run_linear

    return build


FP32_OUTPUT_ABI = BF16_FP32_TGV_ELEMENTS
