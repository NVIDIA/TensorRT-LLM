# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's tunable NVFP4 W4A4 GEMM: candidates, tuning buckets, tactic lookup.

One `TunableRunner` over four SM120 implementations of ``alpha * (act @ weight.T)``
on block-scaled NVFP4 operands (packed E2M1, 128x4-swizzled E4M3 scales, fp32 alpha):

* ``cutlass`` -- TRT-LLM's CUTLASS FP4 GEMM (``torch.classes.trtllm.FP4GemmRunner``),
  one tactic per config; tactic ``-1`` (the fallback) is its default config;
* ``cublaslt`` -- cuBLASLt's block-scaled FP4 GEMM (``CublasLtFP4GemmRunner``); a
  tactic is the algorithm's position in the heuristic list;
* ``cuda_core`` -- the target's cuda-core GEMM (``native_ops.cuda_core_nvfp4_gemm``,
  built from its ``csrc/``) at ``M <= 8`` with BF16 output, reading the activation's
  swizzled scales in place;
* ``fi_cutlass`` -- FlashInfer's SM120 CUTLASS FP4 GEMM (JIT module), data-parallel
  schedulers only (no workspace, fixed reduction order). Offered *alone* at
  ``64 <= M <= 255`` when TRT-LLM's 128x128 grid would fill at most half the SMs
  (``2 * ceil(N/128) * ceil(M/128) <= #SMs``) and N >= 1024, N % 128 == 0, K % 256 == 0,
  BF16/FP16 output: its 128x32 / 128x64 tiles and swap-AB fill the grid in one PDL
  kernel where the producer-less tuning graph would rank cuBLASLt split-K (no PDL, a
  separate reduce) ahead of it.

Tuning (`TUNING_CONFIG`): cold L2 (rotated input copies -- decode weights stream from
DRAM in serving, so L2-warm timings mis-rank small-M tactics); power-of-two row
buckets plus "mixed-step" buckets for prefill chunks with a few decode rows riding
along: rows in ``(p, p+32]`` for ``p >= 2048`` a power of two map to a bucket profiled
at ``p + 32``, rows in ``[p-32, p)`` for ``p >= 4096`` to one profiled at ``p - 1``.

`run(..., tune_rows)` picks the tactic of ``tune_rows`` rows for a call of fewer rows
(the final decoder layer run on only the rows the logits read): outside tuning it
looks the tactic up in the AutoTuner's profiling cache with the call's shapes and
``tune_rows`` in place of the row count -- the cache key ``choose_one`` would have
built for the unsliced call -- and runs the runner with it. Inside tuning the call
tunes as itself, as every other call does.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm._torch.cublaslt_utils import IS_CUBLASLT_AVAILABLE
from tensorrt_llm._torch.utils import (
    fp4_scale_infer_shape,
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
    next_positive_power_of_2,
)
from tensorrt_llm.logger import logger

from ... import native_ops

#: The AutoTuner cache namespace of this GEMM (its own; the built-in nvfp4_gemm's
#: cache is not shared).
CUSTOM_OP = "qwen3_5::nvfp4_gemm_sm120"

CUDA_CORE_MAX_M = 8

FI_MIN_M = 64
FI_MAX_M = 255
#: FlashInfer tactic id = tile * 4 + {0: swap-AB data-parallel, 1: data-parallel};
#: tiles 0-5 are 128x{32,64,128}x{128,256}. The stream-K schedulers (2, 3) are not offered.
FI_TACTICS = tuple(t * 4 + s for t in range(6) for s in (0, 1))

_MIXED_M_MIN_BASE = 2048
_MIXED_M_BAND = 32


# ── tuning buckets ──────────────────────────────────────────────────────────────────


def _nvfp4_mixed_m_bucket(m: int) -> Optional[int]:
    """The mixed-step bucket of ``m`` rows, or None for a power-of-two-bucket row count."""
    if m <= _MIXED_M_MIN_BASE:
        return None
    hi = next_positive_power_of_2(m)
    if hi != m and hi >= 2 * _MIXED_M_MIN_BASE and hi - m <= _MIXED_M_BAND:
        return hi - 1
    lo = last_positive_power_of_2(m - 1)
    if m - lo <= _MIXED_M_BAND:
        return lo + _MIXED_M_BAND
    return None


def _nvfp4_gen_m_buckets(max_num_tokens: int) -> Tuple[int, ...]:
    """Power-of-two buckets up to ``max_num_tokens`` plus the mixed-step buckets
    (``{2080, 4095, 4128, 8191}`` at 8192)."""
    buckets = set(get_last_power_of_2_num_tokens_buckets(max_num_tokens))
    p = _MIXED_M_MIN_BASE
    while p - _MIXED_M_BAND <= max_num_tokens:
        if p >= 2 * _MIXED_M_MIN_BASE:
            buckets.add(p - 1)
        if p < max_num_tokens:
            buckets.add(p + _MIXED_M_BAND)
        p *= 2
    return tuple(sorted(buckets))


def _nvfp4_map_m(m: int) -> int:
    mixed = _nvfp4_mixed_m_bucket(m)
    return mixed if mixed is not None else last_positive_power_of_2(m)


TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(DynamicTensorSpec(0, 0, _nvfp4_gen_m_buckets, _nvfp4_map_m),),
    constraint_specs=(ConstraintSpec(2, 0, fp4_scale_infer_shape),),
    distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
    use_cold_l2_cache=True,
)


# ── the candidates ──────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _cutlass(output_dtype: torch.dtype):
    return torch.classes.trtllm.FP4GemmRunner(
        output_dtype, int(fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4)
    )


@functools.lru_cache(maxsize=None)
def _cublaslt(output_dtype: torch.dtype):
    return torch.classes.trtllm.CublasLtFP4GemmRunner(output_dtype)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


class _FlashInferSm120:
    """FlashInfer's SM120 CUTLASS FP4 GEMM module, built once per process."""

    module = None
    failed = False
    workspace: dict = {}

    @classmethod
    def get(cls):
        if cls.module is None and not cls.failed:
            try:
                from flashinfer.jit.gemm.core import gen_gemm_sm120_module_cutlass_fp4

                cls.module = gen_gemm_sm120_module_cutlass_fp4().build_and_load()
            except Exception as e:  # noqa: BLE001 - the other candidates still serve
                cls.failed = True
                logger.warning(f"FlashInfer SM120 CUTLASS NVFP4 GEMM unavailable: {e}")
        return cls.module

    @classmethod
    def get_workspace(cls, device: torch.device) -> torch.Tensor:
        # The data-parallel schedulers need no workspace; the FFI signature takes one.
        ws = cls.workspace.get(device)
        if ws is None:
            ws = torch.empty(1 << 20, dtype=torch.uint8, device=device)
            cls.workspace[device] = ws
        return ws


def fi_cutlass_offered(
    act_fp4: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype
) -> bool:
    """Whether the FlashInfer GEMM is the (only) candidate set for these operands."""
    if output_dtype not in (torch.bfloat16, torch.float16):
        return False
    if act_fp4.dim() != 2 or weight.dim() != 2:
        return False
    m, n, k = act_fp4.shape[0], weight.shape[0], act_fp4.shape[1] * 2
    if not FI_MIN_M <= m <= FI_MAX_M:
        return False
    if n < 1024 or n % 128 != 0 or k % 256 != 0:
        return False
    tiles = ((n + 127) // 128) * ((m + 127) // 128)
    if 2 * tiles > _sm_count(act_fp4.device.index):
        return False
    return _FlashInferSm120.get() is not None


class Nvfp4GemmSm120Runner(TunableRunner):
    """The candidates above as one runner; a tactic is ``(backend, sub_tactic)``."""

    def __init__(self, output_dtype: torch.dtype):
        super().__init__()
        self.output_dtype = output_dtype

    def unique_id(self):
        return (self.output_dtype,)

    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs
    ) -> List[Tuple[str, int]]:
        act_fp4, weight, act_sf, weight_scale, alpha = inputs
        if fi_cutlass_offered(act_fp4, weight, self.output_dtype):
            return [("fi_cutlass", t) for t in FI_TACTICS]
        tactics: List[Tuple[str, int]] = []
        if act_fp4.shape[0] <= CUDA_CORE_MAX_M and self.output_dtype == torch.bfloat16:
            tactics.append(("cuda_core", 0))
        tactics.extend(("cutlass", t) for t in range(_cutlass(self.output_dtype).get_num_configs()))
        if IS_CUBLASLT_AVAILABLE:
            runner = _cublaslt(self.output_dtype)
            num_algos = runner.get_num_heuristic_algos(act_fp4, weight, act_sf, weight_scale)
            tactics.extend(("cublaslt", t) for t in range(num_algos))
        return tactics

    def forward(self, inputs: List[torch.Tensor], tactic=-1, **kwargs) -> torch.Tensor:
        act_fp4, weight, act_sf, weight_scale, alpha = inputs
        if tactic == -1:
            tactic = ("cutlass", -1)
        backend, sub_tactic = tactic
        if backend == "cutlass":
            return _cutlass(self.output_dtype).run_gemm(
                act_fp4, weight, act_sf, weight_scale, alpha, 0, sub_tactic, None, None
            )
        if backend == "cublaslt":
            return _cublaslt(self.output_dtype).run_gemm(
                act_fp4, weight, act_sf, weight_scale, alpha, 0, sub_tactic, None, None
            )
        if backend == "cuda_core":
            return native_ops.cuda_core_nvfp4_gemm(act_fp4, weight, act_sf, weight_scale, alpha)
        if backend == "fi_cutlass":
            if sub_tactic == -1:
                sub_tactic = FI_TACTICS[0]
            out = torch.empty(
                (act_fp4.shape[0], weight.shape[0]), dtype=self.output_dtype, device=act_fp4.device
            )
            _FlashInferSm120.get().fp4_gemm(
                act_fp4.view(torch.uint8),
                weight.view(torch.uint8),
                act_sf.view(torch.uint8),
                weight_scale.view(torch.uint8),
                alpha,
                out,
                _FlashInferSm120.get_workspace(act_fp4.device),
                int(sub_tactic),
            )
            return out
        raise ValueError(f"invalid NVFP4 GEMM tactic {tactic!r}")


@functools.lru_cache(maxsize=None)
def runner_for(output_dtype: torch.dtype) -> Nvfp4GemmSm120Runner:
    return Nvfp4GemmSm120Runner(output_dtype)


def lookup_tactic(runner: Nvfp4GemmSm120Runner, inputs: List[torch.Tensor], tune_rows: int):
    """The cached tactic of the call ``inputs`` would be with ``tune_rows`` rows, or the
    fallback ``-1`` on a miss. Reads the cache only; never profiles."""
    shapes = [t.size() if isinstance(t, torch.Tensor) else torch.Size((0,)) for t in inputs]
    shapes[0] = torch.Size((tune_rows,) + tuple(shapes[0][1:]))
    hit, _, tactic, _ = AutoTuner.get().profiling_cache.search_cache(
        CUSTOM_OP, [runner], tuple(shapes), TUNING_CONFIG
    )
    if not hit:
        logger.warning_once(
            f"[AutoTuner] {CUSTOM_OP} using the fallback tactic, due to cache miss on "
            f"input shapes={tuple(shapes)}",
            key=(CUSTOM_OP, "warning_autotuning_cache_miss_fallback"),
        )
    return tactic


def run(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    tune_rows: int,
) -> torch.Tensor:
    runner = runner_for(output_dtype)
    inputs = [act_fp4, weight, act_sf, weight_scale, alpha]
    tuner = AutoTuner.get()
    if tune_rows == act_fp4.shape[0] or tuner.is_tuning_mode:
        _, tactic = tuner.choose_one(CUSTOM_OP, [runner], TUNING_CONFIG, inputs)
    else:
        tactic = lookup_tactic(runner, inputs, tune_rows)
    return runner(inputs, tactic=tactic)
