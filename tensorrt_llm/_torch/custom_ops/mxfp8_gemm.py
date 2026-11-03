# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 GEMM implementations with explicit, externally selected tactics.

Every candidate accumulates each output over K in one sequential FP32 pass and rounds once, so all
tactics return the same bits for the same inputs and an output row never depends on the other rows.
"""

from collections import Counter
from functools import lru_cache
from typing import ClassVar

import torch

from tensorrt_llm.logger import logger

from ..autotuner import AutoTuner, OptimizationProfile, TunableRunner, TuningConfig
from ..modules.mxfp8_swapab import MXFP8SwapABGemmRunner

MXFP8_DEFAULT_BACKENDS = (
    "flashinfer_cutlass",
    "flashinfer_cute_dsl",
    "cutedsl_swap_ab",
    "trtllm_cutlass",
)
# Decode-graph GEMMs read their weights from HBM on every use: profile with a cold L2.
_COLD_L2_FLUSH_BYTES = 256 << 20  # 256 MiB


def validate_mxfp8_backends(backends: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and canonicalize the ordered MXFP8 candidate list."""
    if not backends or any(backend not in MXFP8_DEFAULT_BACKENDS for backend in backends):
        raise ValueError(
            f"Invalid MXFP8 allowed backends {backends!r}; "
            f"expected a nonempty subset of {MXFP8_DEFAULT_BACKENDS!r}"
        )
    return tuple(dict.fromkeys(backends))


@lru_cache
def _flashinfer_runner(backend: str, major: int, minor: int, output_dtype: torch.dtype):
    # These runner interfaces belong to the pinned flashinfer-python dependency.
    from flashinfer.gemm.gemm_base import _cute_dsl_gemm_mxfp8_runner, get_cutlass_mxfp8_gemm_module

    if backend == "flashinfer_cutlass":
        return get_cutlass_mxfp8_gemm_module(major).cutlass_mxfp8_gemm_runner()
    return _cute_dsl_gemm_mxfp8_runner(major, minor, True, output_dtype)


class FlashInferMXFP8GemmRunner(TunableRunner):
    """Adapt a FlashInfer backend without invoking its autotuner or tactic cache."""

    def __init__(self, output_dtype: torch.dtype, backend: str = "flashinfer_cutlass"):
        if backend not in MXFP8_DEFAULT_BACKENDS[:2]:
            raise ValueError(f"Invalid FlashInfer MXFP8 backend: {backend}")
        self.output_dtype = output_dtype
        self.backend = backend

    def unique_id(self) -> tuple:
        return (self.output_dtype, self.backend)

    def _supports(self, inputs: list[torch.Tensor]) -> bool:
        from flashinfer.gemm.gemm_base import (
            _check_mm_mxfp8_problem_size,
            _cute_dsl_gemm_mxfp8_requirement,
            _cutlass_gemm_mxfp8_requirement,
        )

        act, act_scale, weight, weight_scale = inputs
        if (
            act.device.type != "cuda"
            or any(t.device != act.device or not t.is_contiguous() for t in inputs)
            or act.ndim != 2
            or weight.ndim != 2
            or min(*act.shape, *weight.shape) <= 0
        ):
            return False
        requirement = (
            _cutlass_gemm_mxfp8_requirement
            if self.backend == "flashinfer_cutlass"
            else _cute_dsl_gemm_mxfp8_requirement
        )
        args = (act, weight.t(), act_scale, weight_scale)
        kwargs = dict(out_dtype=self.output_dtype, use_8x4_sf_layout=False)
        try:
            _check_mm_mxfp8_problem_size(*args, **kwargs)
            return bool(requirement(*args, **kwargs))
        except (ValueError, RuntimeError):
            # An unavailable architecture/layout can still be served by a peer.
            return False

    def _prepare(self, inputs: list[torch.Tensor]):
        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE, _get_cache_buf

        act, act_scale, weight, weight_scale = inputs
        major, minor = torch.cuda.get_device_capability(act.device)
        runner = _flashinfer_runner(self.backend, major, minor, self.output_dtype)
        out = act.new_empty((act.shape[0], weight.shape[0]), dtype=self.output_dtype)
        workspace = _get_cache_buf("mm_mxfp8_workspace", DEFAULT_WORKSPACE_SIZE, act.device)
        return runner, [act, weight.t(), act_scale, weight_scale, self.output_dtype, out, workspace]

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile | None, **kwargs
    ) -> list:
        if not self._supports(inputs):
            return []
        runner, prepared = self._prepare(inputs)
        tactics = runner.get_valid_tactics(prepared, None)
        if self.backend == "flashinfer_cute_dsl":
            # Split-K tactics (last field > 1) reduce K in another order; keep the bits fixed.
            tactics = [t for t in tactics if not isinstance(t, tuple) or t[-1] == 1]
        return tactics

    def forward(self, inputs: list[torch.Tensor], tactic=-1) -> torch.Tensor:
        runner, prepared = self._prepare(inputs)
        runner.forward(prepared, tactic=tactic)
        return prepared[-2]


class TrtllmMXFP8GemmRunner(TunableRunner):
    """TensorRT-LLM's CUTLASS MXFP8 GEMM with an explicit config as the tactic."""

    _ALPHA: ClassVar[dict] = {}

    def __init__(self, output_dtype: torch.dtype):
        self.output_dtype = output_dtype

    def unique_id(self) -> tuple:
        return (self.output_dtype,)

    def _native(self):
        from .torch_custom_ops import MXFP8GemmRunner

        return MXFP8GemmRunner(self.output_dtype)

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile | None, **kwargs
    ) -> list:
        act, act_scale, weight, weight_scale = inputs
        if (
            act.ndim != 2
            or weight.ndim != 2
            or act.dtype != torch.float8_e4m3fn
            or weight.dtype != torch.float8_e4m3fn
            or act.shape[1] != weight.shape[1]
            or any(t.device != act.device or not t.is_contiguous() for t in inputs)
        ):
            return []
        return list(range(self._native().mxfp8_gemm_runner.get_num_configs()))

    def forward(self, inputs: list[torch.Tensor], tactic=-1) -> torch.Tensor:
        act = inputs[0]
        alpha = self._ALPHA.get(act.device.index)
        if alpha is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Run the native MXFP8 GEMM once before CUDA graph capture")
            alpha = self._ALPHA[act.device.index] = torch.ones(
                1, dtype=torch.float32, device=act.device
            )
        return self._native().forward([*inputs, alpha], tactic=tactic)


class MXFP8GemmUnifiedRunner(TunableRunner):
    """Compare concrete FlashInfer, TensorRT-LLM CUTLASS and Swap-AB configurations in one tuner."""

    # Graph warmup visits each capture shape; cache and profiling use that exact M.
    tuning_config = TuningConfig(cold_l2_flush_bytes=_COLD_L2_FLUSH_BYTES)
    # Expose JIT participation so loaded winners are primed before graph capture.
    kernel_cache: ClassVar[dict] = MXFP8SwapABGemmRunner.kernel_cache

    def __init__(
        self, output_dtype: torch.dtype, allowed_backends: tuple[str, ...] = MXFP8_DEFAULT_BACKENDS
    ):
        self.output_dtype = output_dtype
        self.allowed_backends = validate_mxfp8_backends(tuple(allowed_backends))
        self.runners = {
            backend: (
                MXFP8SwapABGemmRunner(output_dtype)
                if backend == "cutedsl_swap_ab"
                else TrtllmMXFP8GemmRunner(output_dtype)
                if backend == "trtllm_cutlass"
                else FlashInferMXFP8GemmRunner(output_dtype, backend)
            )
            for backend in self.allowed_backends
        }

    def unique_id(self) -> tuple:
        return (self.output_dtype, self.allowed_backends)

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile | None, **kwargs
    ) -> list[tuple]:
        return [
            (backend, tactic)
            for backend, runner in self.runners.items()
            for tactic in runner.get_valid_tactics(inputs, profile)
        ]

    def forward(self, inputs: list[torch.Tensor], tactic=-1) -> torch.Tensor:
        if tactic == -1:
            for backend, runner in self.runners.items():
                supported = (
                    bool(runner.get_valid_tactics(inputs, None))
                    if backend in ("cutedsl_swap_ab", "trtllm_cutlass")
                    else runner._supports(inputs)
                )
                if supported:
                    return runner(inputs, tactic=-1)
            raise ValueError("No allowed MXFP8 GEMM backend supports these inputs")
        backend, sub_tactic = tactic
        return self.runners[backend](inputs, tactic=sub_tactic)


def agree_on_tactics(per_rank: list[dict]) -> dict:
    """Per cache key, the entry whose tactic most ranks chose (ties: the lowest rank's), from the
    ranks' cache entries in rank order. Ranks of a data-parallel step wait for the slowest one, so
    a rank that kept a slower kernel by profiling noise would cost them all."""
    agreed = {}
    for key in {key for entries in per_rank for key in entries}:
        held = [entries[key] for entries in per_rank if key in entries]
        votes = Counter(repr(value[1]) for value in held)
        top = max(votes.values())
        agreed[key] = next(value for value in held if votes[repr(value[1])] == top)
    return agreed


def agree_mxfp8_tactics(custom_op: str) -> None:
    """Make every rank of the tuner's tensor-parallel group hold the same ``custom_op`` tactics
    (``agree_on_tactics``). A collective: every rank calls it at the same point of the warmup."""
    tuner = AutoTuner.get()
    AutoTuner.L2_FLUSH_BUFFERS.clear()
    if not tuner._is_distributed():
        return
    cache = tuner.profiling_cache
    mine = cache.get_specific_custom_op(custom_op)
    agreed = agree_on_tactics(tuner._dist.tp_cp_allgather(obj=mine))
    changed = sum(1 for key, value in agreed.items() if key not in mine or mine[key][1] != value[1])
    cache.merge_cache_data(agreed)
    if changed:
        logger.info(f"[MXFP8] {changed} of {len(agreed)} GEMM tactics set to the ranks' majority")
