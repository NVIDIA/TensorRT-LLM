# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 MLA Q-B projection with per-M dispatch.

The public registry ABI is ``build() -> launcher(input, weight, output)`` for
contiguous BF16 ``[M,2048] x [4096,2048] -> [M,4096]`` tensors.

Runs on SM100+ (Blackwell) GPUs.  Supported shapes: M in
``SUPPORTED_M_VALUES``, N=4096, K=2048; BF16 input, weight and output.
M<=16 uses persistent cells and M>=32 the tcgen05 (64,6,2) cell.

``build(prefetch_w_k_steps=N)`` is a schedule option of the tcgen05 cells
(M=32/48/64).  It runs the same kernel and issues
``cp.async.bulk.prefetch.tensor`` for the first N weight K-steps before
barrier init.  A prefetch has no data effect, so the output is bit-exact
with N=0; only the cold-weight latency changes.  The default is
``PREFETCH_W_K_STEPS`` (3), the shipped depth; ``prefetch_w_k_steps=0`` is
the unprefetched schedule.
"""

from functools import lru_cache

from . import _projection_module_dispatch as _projectionmoduledispatch
from . import _projection_tcgen05_template as _projectiontcgen05template
from . import _swapped_dense_gemm_persistent as _swappeddensegemmpersistent

ProjectionAbi = _projectionmoduledispatch.ProjectionAbi
context_instances = _projectionmoduledispatch.context_instances
make_build = _projectionmoduledispatch.make_build
persistent_instance = _projectionmoduledispatch.persistent_instance
tcgen05_instance = _projectionmoduledispatch.tcgen05_instance
GemmSchedule = _swappeddensegemmpersistent.GemmSchedule

PREWAIT_TMA_W = _projectiontcgen05template.PREWAIT_TMA_W


ABI = ProjectionAbi(
    "self_attn.q_b_proj",
    4_096,
    2_048,
    kernel_name_prefix="glm52_self_attn_q_b_proj",
)

SCHEDULE_BY_M = {
    1: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    2: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    4: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    8: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    16: persistent_instance(GemmSchedule(False, (64, 16), (4, 1))),
    # The preceding kernel produces A. Only the immutable weight TMA may
    # cross the PDL wait. M=32/48 use the M=64 tcgen05 schedule; it measured
    # faster than a (32, 8, 1) cell.
    32: tcgen05_instance(64, 6, 2, PREWAIT_TMA_W),
    48: tcgen05_instance(64, 6, 2, PREWAIT_TMA_W),
    64: tcgen05_instance(64, 6, 2, PREWAIT_TMA_W),
} | context_instances(
    persistent_instance(GemmSchedule(True, (256, 128), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1), raster_along="n")),
    persistent_instance(GemmSchedule(True, (256, 224), (2, 1), raster_along="n")),
)
# The (64,6,2) schedule is slower than the persistent cells at M<=16, so
# those cells are kept. prefetch_w_k_steps applies to every tcgen05 cell
# (32/48/64); it is bit-exact and only changes the cold-weight latency.
# The decode cells that carry the prefetch option (tcgen05 (64,6,2,PREWAIT_TMA_W)).
TCGEN05_M_VALUES = (32, 48, 64)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)
PREFETCH_W_K_STEPS = 3  # default prefetch depth of the tcgen05 cells


def schedule_by_m(prefetch_w_k_steps: int = 0) -> dict:
    """Return the catalog; the tcgen05 cells (M=32/48/64) carry the prefetch option."""
    if prefetch_w_k_steps == 0:
        return SCHEDULE_BY_M
    return SCHEDULE_BY_M | {
        active_m: tcgen05_instance(64, 6, 2, PREWAIT_TMA_W, prefetch_w_k_steps=prefetch_w_k_steps)
        for active_m in TCGEN05_M_VALUES
    }


@lru_cache(maxsize=None)
def _build_factory(prefetch_w_k_steps: int):
    return make_build(ABI, schedule_by_m(prefetch_w_k_steps))


def build(*, prefetch_w_k_steps: int = PREFETCH_W_K_STEPS):
    """Registry factory; ``prefetch_w_k_steps`` selects the prefetching M=64 cell."""
    return _build_factory(int(prefetch_w_k_steps))()
