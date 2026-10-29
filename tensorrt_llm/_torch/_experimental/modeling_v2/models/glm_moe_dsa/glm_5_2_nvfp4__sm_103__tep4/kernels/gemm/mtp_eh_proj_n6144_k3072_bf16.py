# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 MTP ``eh_proj`` module adapter for B300.

TRT-LLM contract
----------------
The MTP layer RMS-normalizes embedding and target hidden states independently,
concatenates the two 6144-wide results, and takes one TP4 chunk before its
row-parallel projection.  Each rank therefore executes this local GEMM::

    input:   BF16 [M, 3072], column stride 1
    weight:  contiguous BF16 [6144, 3072]
    output:  contiguous BF16 [M, 6144]

The staged input is a view into contiguous ``[M,12288]`` concat storage, so its
row stride is 12288 rather than 3072.  The adapter also accepts row stride 3072
for the fused norm/shard producer (``mtp_norm_shard_*``).  The projection result is passed to
the original TP4 all-reduce; that collective and its ``all_reduce_params`` stay
part of the public module ABI but are not included in local-GEMM timings.

With ``max_draft_len=1``, generation M is the number of MTP draft rows, equal to
the graph batch bucket: 1, 2, 4, 8, 16, 24, or 32.  It is not the target-model
``2 * batch`` token count.  The measured context/prefill buckets 48, 64, 128,
256, 512, 1024, 2048, 4096, and 8192 are also specialized; other M values
retain the TRT-LLM fallback.

Implementation and module ABI
-----------------------------
This file reuses existing Blackwell GEMMs rather than copying device code.  The
low-latency TGV kernel handles M1--M32 with CTA-N 8, 16, or 32.  The persistent
dense kernel handles M48--M8192 with bucket-specific tile, cluster, raster, and
direct/swapped orientation.  Both paths write the public row-major output
directly.  The input row stride remains a runtime TMA descriptor field.

``GLMMtpEhProjAdapter.forward`` matches ``Linear.forward``: it owns the original
weight parameter, preserves the original all-reduce callable and keyword
arguments, selects an exact-M precompiled GEMM using tensor metadata only, and
returns the collective result.  Passing ``all_reduce=None`` exposes the local
projection for isolated validation; the TRT-LLM factory requires the original
collective module.

Performance and PDL
-------------------
On B300 the TGV cells (M1--M32) match or beat the cuBLAS ``F.linear`` kernel,
and the persistent cells (M48--M8192) are within about 5% of it.  The TGV
K-pipeline uses K128 with eight stages.

PDL defaults on.  The TGV weight DMA does not depend on the producer, while
activation DMA waits before reading the norm output.  PDL is kept for correct
composition and downstream scheduling; it gives no isolated speedup.
"""

from collections.abc import Callable

import torch

from ._swapped_dense_gemm_persistent import (
    CompiledPersistentGemm,
    GemmSchedule,
    compile_persistent_gemm,
)
from ._swapped_tgv_gemm import CompiledSwappedTgvGemm, TgvSchedule, compile_swapped_tgv_gemm

HIDDEN_SIZE = 6_144
FULL_CONCAT_SIZE = HIDDEN_SIZE * 2
LOCAL_INPUT_SIZE = FULL_CONCAT_SIZE // 4

AllReduce = Callable[..., torch.Tensor]

TGV_SCHEDULE_BY_M = {
    1: TgvSchedule(8, 128, 8),
    2: TgvSchedule(8, 128, 8),
    4: TgvSchedule(8, 128, 8),
    8: TgvSchedule(8, 128, 8),
    16: TgvSchedule(16, 128, 8),
    24: TgvSchedule(32, 128, 8),
    32: TgvSchedule(32, 128, 8),
}
PERSISTENT_SCHEDULE_BY_M = {
    48: GemmSchedule(True, (128, 64), (4, 1), raster_along="n"),
    64: GemmSchedule(True, (128, 64), (4, 1), raster_along="n"),
    128: GemmSchedule(True, (128, 128), (2, 1)),
    256: GemmSchedule(
        True,
        (256, 96),
        (2, 1),
        orientation="direct",
        raster_along="n",
    ),
    512: GemmSchedule(True, (256, 176), (2, 1), raster_along="n"),
    1_024: GemmSchedule(True, (256, 176), (2, 1), raster_along="n"),
    2_048: GemmSchedule(True, (256, 256), (4, 1), raster_along="n"),
    4_096: GemmSchedule(True, (256, 160), (2, 1), raster_along="n"),
    8_192: GemmSchedule(True, (256, 224), (2, 1)),
}
SCHEDULE_BY_M = TGV_SCHEDULE_BY_M | PERSISTENT_SCHEDULE_BY_M
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)


def _compile_projection(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    schedule: TgvSchedule | GemmSchedule,
) -> CompiledSwappedTgvGemm | CompiledPersistentGemm:
    """Compile the device implementation selected for one M bucket."""
    if isinstance(schedule, TgvSchedule):
        return compile_swapped_tgv_gemm(input_tensor, weight, output, schedule)
    return compile_persistent_gemm(input_tensor, weight, output, schedule)


def _validate_weight(weight: torch.nn.Parameter) -> None:
    """Validate the exact local row-parallel shard from staged TRT-LLM."""
    expected_shape = (HIDDEN_SIZE, LOCAL_INPUT_SIZE)
    if not isinstance(weight, torch.nn.Parameter):
        raise TypeError("weight must be the original torch.nn.Parameter")
    if (
        weight.shape != expected_shape
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise ValueError(
            f"MTP eh_proj weight must be contiguous CUDA BF16 {expected_shape}, "
            f"got shape={weight.shape} dtype={weight.dtype} device={weight.device} "
            f"contiguous={weight.is_contiguous()}"
        )


def _validate_input(input_tensor: torch.Tensor) -> int:
    """Validate staged or fused-producer local input and return runtime M."""
    if input_tensor.ndim != 2 or input_tensor.shape[1] != LOCAL_INPUT_SIZE:
        raise ValueError(
            f"MTP eh_proj input must have shape [M,{LOCAL_INPUT_SIZE}], got {input_tensor.shape}"
        )
    active_m = input_tensor.shape[0]
    if active_m not in SUPPORTED_M_VALUES:
        raise ValueError(
            f"unsupported MTP eh_proj M={active_m}; expected {SUPPORTED_M_VALUES}. "
            "Use the TRT-LLM fallback outside decode graph buckets."
        )
    if input_tensor.dtype != torch.bfloat16 or not input_tensor.is_cuda:
        raise ValueError("MTP eh_proj input must be a CUDA BF16 tensor")
    if input_tensor.stride(1) != 1 or input_tensor.stride(0) not in (
        LOCAL_INPUT_SIZE,
        FULL_CONCAT_SIZE,
    ):
        raise ValueError(
            "MTP eh_proj input must be a contiguous local shard or staged concat "
            f"view, got stride={input_tensor.stride()}"
        )
    return active_m


class GLMMtpEhProjAdapter(torch.nn.Module):
    """Match staged row-parallel ``Linear.forward`` including its all-reduce."""

    def __init__(
        self,
        weight: torch.nn.Parameter,
        all_reduce: AllReduce | torch.nn.Module | None,
    ) -> None:
        super().__init__()
        _validate_weight(weight)
        self.weight = weight
        self.all_reduce = all_reduce
        self._kernels: dict[int, CompiledSwappedTgvGemm | CompiledPersistentGemm] = {}
        for active_m, schedule in SCHEDULE_BY_M.items():
            concat_storage = weight.new_empty((active_m, FULL_CONCAT_SIZE))
            compile_input = concat_storage[:, LOCAL_INPUT_SIZE : 2 * LOCAL_INPUT_SIZE]
            compile_output = weight.new_empty((active_m, HIDDEN_SIZE))
            self._kernels[active_m] = _compile_projection(
                compile_input, weight, compile_output, schedule
            )

    def forward(
        self,
        input: torch.Tensor,
        *,
        all_reduce_params: object | None = None,
        lora_params: dict | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        """Project one TP shard, then invoke the original all-reduce module."""
        del layer_idx
        if lora_params:
            raise ValueError("MTP eh_proj specialization does not support LoRA")
        active_m = _validate_input(input)
        if input.device != self.weight.device:
            raise ValueError("MTP eh_proj input and weight must share a CUDA device")
        local_output = input.new_empty((active_m, HIDDEN_SIZE))
        self._kernels[active_m](input, self.weight, local_output)
        if self.all_reduce is None:
            return local_output
        return self.all_reduce(local_output, all_reduce_params=all_reduce_params)


def make_trtllm_adapter(
    weight: torch.nn.Parameter,
    all_reduce: AllReduce | torch.nn.Module,
) -> GLMMtpEhProjAdapter:
    """Construct the callsite-compatible TP4 MTP ``eh_proj`` module."""
    return GLMMtpEhProjAdapter(weight, all_reduce)
