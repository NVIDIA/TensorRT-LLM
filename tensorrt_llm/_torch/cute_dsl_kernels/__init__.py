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
_announced = set()


def sm_reserve() -> int:
    """SMs to keep out of persistent grids, from ``TRTLLM_CUTEDSL_SM_RESERVE`` (0 = none)."""
    try:
        return max(0, int(os.environ.get(_SM_RESERVE_ENV, "0")))
    except ValueError:
        return 0


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
