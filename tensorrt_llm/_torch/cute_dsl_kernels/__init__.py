# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL kernels.

``TRTLLM_CUTEDSL_SM_RESERVE=<n>`` (default 0) keeps ``n`` SMs out of every
CuTeDSL persistent grid, so CTAs that would land on SMs another kernel holds
resident do not run their static tile share as a second wave.  The hook wraps
``HardwareInfo.get_max_active_clusters`` on the class, lives here so it runs
before any kernel module imports, and reads the variable at each grid sizing;
callers that memoize the occupancy must key on ``sm_reserve()`` (the dense
GEMM custom op does) or a reserve set after their first query is lost.
"""
import math
import os

from tensorrt_llm.logger import logger

_SM_RESERVE_ENV = "TRTLLM_CUTEDSL_SM_RESERVE"
_FC1_SITU_FAST_ENV = "TRTLLM_CUTEDSL_FC1_SITU_FAST"
_MXFP8_STATIONARY_ENV = "TRTLLM_CUTEDSL_MXFP8_STATIONARY"
_MOE_WEIGHT_EVICT_FIRST_ENV = "TRTLLM_CUTEDSL_MOE_WEIGHT_EVICT_FIRST"
_EVICT_FIRST_MAX_TOKENS = 512
_EVICT_FIRST_MAX_LOCAL_EXPERTS = 512
_announced = set()


def sm_reserve() -> int:
    """SMs to keep out of persistent grids, from ``TRTLLM_CUTEDSL_SM_RESERVE`` (0 = none)."""
    try:
        return max(0, int(os.environ.get(_SM_RESERVE_ENV, "0")))
    except ValueError:
        return 0


def _switch(name: str, default: str = "0") -> bool:
    """A 0 / 1 environment switch, read at use time; any other value is a configuration error."""
    value = os.environ.get(name, default).strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def fc1_situ_fast() -> bool:
    """Use the single-MUFU SiTU epilogue in the Rubin routed-expert FC1 kernel.

    ``TRTLLM_CUTEDSL_FC1_SITU_FAST=1`` (default 0 = off) evaluates the three SiTU sigmoid-type
    terms with the hardware ``tanh.approx.f32`` instead of ``ex2`` + ``rcp`` per sigmoid. The epilogue is the exposed part of the FC1 critical path at
    prefill sizes, and its MUFU time is the largest component. The result differs from the exact
    path by the MUFU tanh error before the NVFP4 quantization (about 2**-11: relative for the two
    tanh factors, absolute for the gate factor ``1 + tanh(g / 2)``, see
    ``_apply_situ_epilogue_fast``), so it is not bitwise equal and needs an accuracy gate. Read
    when a kernel is built and part of its cache key.
    """
    return _switch(_FC1_SITU_FAST_ENV)


def mxfp8_stationary() -> bool:
    """Offer B-stationary tactics to the Rubin MXFP8 GEMM runners that support them (the MLA kv_b GEMM).

    ``TRTLLM_CUTEDSL_MXFP8_STATIONARY=1`` (default 0 = off) adds the tactics of
    ``rubin/kvb_stationary_gemm.py`` (whole B tile resident in shared memory, A streamed) to the
    autotuner's candidates for short-K GEMMs with large M and N; the autotuner still picks the fastest
    tactic. The output is bitwise equal to the dense kernel's. Read when tactics are listed and part of
    the runner's autotuner cache key.
    """
    return _switch(_MXFP8_STATIONARY_ENV)


def moe_weight_evict_first(num_tokens: int,
                           top_k: int,
                           num_experts: int,
                           num_local_experts: int,
                           tile_size: int,
                           die_local: bool = False) -> bool:
    """Whether a Rubin NVFP4 routed-expert GEMM marks its weight loads L2 evict-first.

    With the hint on, the FC1 (gather GEMM) and FC2 (GEMM + finalize) kernels issue the TMA loads of the
    weight tile and of its block scales with the CUTLASS ``CacheHintSm90::EVICT_FIRST`` policy; the
    activation loads keep the default policy. The hint only changes which lines the L2 keeps, never a
    value. It pays only when both kernels carry it (hinting one of them is about neutral) and only for some
    call shapes, so it is gated by shape. Per layer on Rubin, hint on vs off at the same tactic:

    * default allocation, decode (MoE TP4xEP2: 448 local experts x inter 768, 6-768 tokens): faster;
      with 112 or 56 local experts x inter 3072 and up to 512 tokens: faster.
    * die-local layout, decode (DEP16: 56 local experts x inter 3072, 512-7168 dispatched slots):
      faster while the mean routed rows per expert fit one routing tile; above that the result is
      mixed and noisy.
    * default allocation, larger calls: slower, increasingly so with the number of input rows (the
      loss follows the number of input rows, not the rows per expert); about 1K tokens is neutral.
      The die-local layout does not show this loss.
    * 896 local experts (DWDP16 prefill, every expert on one GPU): slower at 256-512 tokens
      and mixed above, noisy.

    Why the sign depends on the call shape is not understood, so the thresholds below are empirical.

    Args:
        num_tokens: Rows of the MoE input (tokens, or dispatched slots under expert parallelism).
        top_k: Experts selected per token.
        num_experts: Experts of the whole layer.
        num_local_experts: Experts whose weights this GPU holds.
        tile_size: Routing tile of the call (rows per expert tile).
        die_local: Whether the call runs on the die-local expert layout.

    Returns:
        True when the weight loads are marked evict-first: with at most ``_EVICT_FIRST_MAX_LOCAL_EXPERTS``
        local experts, on the die-local layout while ``num_tokens * top_k <= tile_size * num_experts``,
        otherwise up to ``_EVICT_FIRST_MAX_TOKENS`` tokens. ``TRTLLM_CUTEDSL_MOE_WEIGHT_EVICT_FIRST=0``
        (default 1) turns the hint off everywhere. The result is part of the kernel cache key, so it is
        read when a kernel is built.
    """
    if not _switch(_MOE_WEIGHT_EVICT_FIRST_ENV, default="1"):
        return False
    if num_local_experts > _EVICT_FIRST_MAX_LOCAL_EXPERTS:
        return False
    if die_local:
        return num_tokens * top_k <= tile_size * num_experts
    return num_tokens <= _EVICT_FIRST_MAX_TOKENS


def log_moe_weight_evict_first(kernel: str, enabled: bool,
                               num_tokens: int) -> None:
    """Announce once per process, kernel and decision how a routed-expert kernel was built.

    Called when a kernel is compiled, never per call, so the log shows whether the gate was on for the first
    shape that reached each kernel kind (GEN workers: on; prefill workers: off).
    """
    logger.info_once(
        f"{_MOE_WEIGHT_EVICT_FIRST_ENV}: Rubin NVFP4 {kernel} weight loads L2 evict-first "
        f"{'on' if enabled else 'off'} (first build at {num_tokens} tokens)",
        key=f"moe_weight_evict_first_{kernel}_{enabled}")


def _install_sm_reserve() -> None:
    try:
        from cutlass.utils import HardwareInfo
    except ImportError:  # CuTeDSL not installed; nothing to reserve from.
        return
    if getattr(HardwareInfo.get_max_active_clusters, "_trtllm_sm_reserve_hook",
               False):
        return
    original = HardwareInfo.get_max_active_clusters

    def get_max_active_clusters(self, cluster_size: int, *args,
                                **kwargs) -> int:
        full = original(self, cluster_size, *args, **kwargs)
        reserve = sm_reserve()
        if reserve <= 0:
            return full
        if reserve not in _announced:
            _announced.add(reserve)
            logger.info(
                f"[cute_dsl_kernels] {_SM_RESERVE_ENV}={reserve}: persistent grids "
                f"launch with {reserve} SMs reserved")
        # One cluster occupies cluster_size SMs; drop enough clusters to free
        # ``reserve`` SMs, never below one cluster.
        return max(1, full - math.ceil(reserve / max(1, cluster_size)))

    get_max_active_clusters._trtllm_sm_reserve_hook = True
    HardwareInfo.get_max_active_clusters = get_max_active_clusters


_install_sm_reserve()
