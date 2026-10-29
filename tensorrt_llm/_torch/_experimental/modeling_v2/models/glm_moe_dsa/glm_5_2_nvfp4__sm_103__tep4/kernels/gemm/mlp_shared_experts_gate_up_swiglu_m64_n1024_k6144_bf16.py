# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 shared-expert gate/up projection fused with SwiGLU.

The kernel consumes BF16 ``input[M,6144]`` and an inference-time prepared BF16
``weight[1024,6144]`` and writes BF16 ``output[M,512]`` for every ``M`` in
``[1,64]``.  It performs the gate/up GEMM, the internal DSMEM split-K reduction,
and SwiGLU in one launch.  The old ``[M,1024]`` intermediate and the standalone
activation launch therefore disappear.

Checkpoint rows arrive as ``[gate_0..gate_511, up_0..up_511]``.  Call
``prepare_weight`` once while loading the checkpoint to store them as
``[gate_0, up_0, gate_1, up_1, ...]``.  Each activation pair then occupies
adjacent accumulator registers; the epilogue needs no cross-CTA exchange.

The per-M schedules are bounded by the production NVJET gate/up grid and shared
memory measured on B300.  They intentionally do not maximize standalone GPU
occupancy: the remaining SMs are available to the routed-expert stream.

Every schedule prefetches its CTA's whole per-rank weight slice into L2 before the
barriers are initialised (``prefetch_w_k_steps`` = the per-rank K steps, 24 or 32).
It costs no SMEM, no registers and no CTAs.  On B200 with cold weights it is worth
4-10% at every ``M`` in ``[1,64]`` and flattens the per-M spread (6.02-6.04 us for
every 96-CTA schedule): the barrier init and TMEM allocation that precede the
first load become the work that hides the DRAM round trip, and the pipeline then
streams from L2.  One pipeline's worth (``stages``) recovers only about half of it.
"""

from dataclasses import dataclass
from typing import Callable

import cutlass
import torch

from ._projection_tcgen05_template import (
    EPILOGUE_SWIGLU_INTERLEAVED,
    MMA_K,
    PREWAIT_TMA_NONE,
    CompiledKernel,
    ProjectionConfig,
    compile_projection,
)

MODULE_NAME = "mlp.shared_experts.gate_up_swiglu"
INPUT_FEATURES = 6_144
GATE_UP_FEATURES = 1_024
OUTPUT_FEATURES = 512
MAX_M = 64


@dataclass(frozen=True)
class Schedule:
    """Resource-matched B300 schedule for one contiguous M range."""

    mma_n: int
    split_k: int
    stages: int


_LOW_96 = Schedule(mma_n=32, split_k=3, stages=15)
_LOW_64 = Schedule(mma_n=64, split_k=4, stages=8)
_MID_96 = Schedule(mma_n=32, split_k=3, stages=11)
_MID_128 = Schedule(mma_n=32, split_k=4, stages=11)
_HIGH_96 = Schedule(mma_n=32, split_k=3, stages=14)
_HIGH_128 = Schedule(mma_n=32, split_k=4, stages=11)
_TAIL_96 = Schedule(mma_n=32, split_k=3, stages=13)


def schedule_for_m(active_m: int) -> Schedule:
    """Select the measured resource-matched schedule for ``M`` in ``[1,64]``."""
    if not 1 <= active_m <= MAX_M:
        raise ValueError(f"M must be in [1,{MAX_M}], got {active_m}")
    if active_m <= 8 or active_m in (16, 32):
        return _LOW_96
    if active_m <= 15:
        return _LOW_64
    if active_m <= 24:
        return _MID_96
    if active_m <= 31:
        return _MID_128
    if active_m <= 40:
        return _HIGH_96
    if active_m <= 63:
        return _HIGH_128
    return _TAIL_96


def _config(schedule: Schedule) -> ProjectionConfig:
    return ProjectionConfig(
        module_name=MODULE_NAME,
        m=64,
        n=GATE_UP_FEATURES,
        k=INPUT_FEATURES,
        out_fp32=False,
        mma_n=schedule.mma_n,
        stages=schedule.stages,
        split_k=schedule.split_k,
        standalone_perf="B300 T=1..64",
        prewait_tma_operand=PREWAIT_TMA_NONE,
        epilogue=EPILOGUE_SWIGLU_INTERLEAVED,
        prefetch_w_k_steps=INPUT_FEATURES // MMA_K // schedule.split_k,
    )


_CONFIG_BY_SCHEDULE = {
    schedule: _config(schedule)
    for schedule in {
        _LOW_96,
        _LOW_64,
        _MID_96,
        _MID_128,
        _HIGH_96,
        _HIGH_128,
        _TAIL_96,
    }
}
_COMPILED_BY_KEY: dict[tuple[int, int], CompiledKernel] = {}


def prepare_weight(weight: torch.Tensor) -> torch.Tensor:
    """Interleave checkpoint gate/up rows once before inference begins."""
    expected = (GATE_UP_FEATURES, INPUT_FEATURES)
    if tuple(weight.shape) != expected:
        raise ValueError(f"weight must have shape {expected}, got {tuple(weight.shape)}")
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"weight must be BF16, got {weight.dtype}")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous before row interleaving")
    return (
        weight.view(2, OUTPUT_FEATURES, INPUT_FEATURES)
        .transpose(0, 1)
        .reshape(GATE_UP_FEATURES, INPUT_FEATURES)
        .contiguous()
    )


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def _validate_contract(
    input_tensor: torch.Tensor,
    prepared_weight: torch.Tensor,
    output: torch.Tensor,
) -> int:
    if input_tensor.ndim != 2:
        raise ValueError(f"input must be rank-2, got rank {input_tensor.ndim}")
    active_m = int(input_tensor.shape[0])
    schedule_for_m(active_m)
    expected = (
        (active_m, INPUT_FEATURES),
        (GATE_UP_FEATURES, INPUT_FEATURES),
        (active_m, OUTPUT_FEATURES),
    )
    for name, tensor, shape in (
        ("input", input_tensor, expected[0]),
        ("prepared_weight", prepared_weight, expected[1]),
        ("output", output, expected[2]),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != torch.bfloat16
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA BF16 {shape}, got "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"device={tensor.device} contiguous={tensor.is_contiguous()}"
            )
        if tensor.data_ptr() % 128:
            raise ValueError(f"{name} must be 128-byte aligned")
    if input_tensor.device != prepared_weight.device or output.device != input_tensor.device:
        raise ValueError("input, prepared_weight, and output must share one CUDA device")
    return active_m


def compile(active_m: int) -> CompiledKernel:  # noqa: A001
    """Compile the schedule selected for one active token count."""
    schedule = schedule_for_m(active_m)
    return compile_projection(_CONFIG_BY_SCHEDULE[schedule], active_m=active_m)


def build() -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]:
    """Return a TRT-LLM-style launcher over prepared weights and fused output."""
    compiled_fn: CompiledKernel | None = None
    pinned_m = -1

    def run_fused(
        input_tensor: torch.Tensor,
        prepared_weight: torch.Tensor,
        output: torch.Tensor,
    ) -> object:
        nonlocal compiled_fn, pinned_m
        if compiled_fn is None:
            active_m = _validate_contract(input_tensor, prepared_weight, output)
            device_index = _device_index(input_tensor)
            key = (device_index, active_m)
            compiled = _COMPILED_BY_KEY.get(key)
            if compiled is None:
                cutlass.cuda.initialize_cuda_context(device_index)
                compiled = compile(active_m)
                _COMPILED_BY_KEY[key] = compiled
            compiled_fn = compiled
            pinned_m = active_m
        elif int(input_tensor.shape[0]) != pinned_m:
            raise ValueError(
                f"this build() instance is pinned to M={pinned_m}, "
                f"got M={int(input_tensor.shape[0])}"
            )
        return compiled_fn(input_tensor, prepared_weight, output)

    return run_fused
