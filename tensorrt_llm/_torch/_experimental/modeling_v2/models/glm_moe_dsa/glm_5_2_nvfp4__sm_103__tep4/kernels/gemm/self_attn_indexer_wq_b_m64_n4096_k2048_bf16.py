# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 DSA indexer WQ-B projection with per-M dispatch.

At TP4 this module has the same contiguous BF16
``[M,2048] x [4096,2048] -> [M,4096]`` ABI as MLA Q-B.  It intentionally owns
an identical schedule table because the shape coincidence disappears at other
TP sizes.
"""

from . import _projection_module_dispatch as _projectionmoduledispatch
from . import _swapped_dense_gemm_persistent as _swappeddensegemmpersistent

ProjectionAbi = _projectionmoduledispatch.ProjectionAbi
context_instances = _projectionmoduledispatch.context_instances
make_build = _projectionmoduledispatch.make_build
persistent_instance = _projectionmoduledispatch.persistent_instance
tcgen05_instance = _projectionmoduledispatch.tcgen05_instance

GemmSchedule = _swappeddensegemmpersistent.GemmSchedule


ABI = ProjectionAbi("self_attn.indexer.wq_b", 4_096, 2_048)

SCHEDULE_BY_M = {
    1: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    2: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    4: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    8: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    16: persistent_instance(GemmSchedule(False, (64, 16), (4, 1))),
    # tcgen05 rows: MMA_N=64, 12 stages, split-K 2 (cluster of 2, DSMEM reduce),
    # 64 N tiles x 2 = 128 CTAs, and the first 4 K steps of the weight tile
    # prefetched into L2 ahead of the prologue. Split-K 2 changes the
    # accumulation order, so the output is tolerance-equal to the single-rank
    # schedule (1 bf16 ulp), not bitwise. The persistent rows above have no
    # prefetch option.
    32: tcgen05_instance(64, 12, 2, prefetch_w_k_steps=4),
    48: tcgen05_instance(64, 12, 2, prefetch_w_k_steps=4),
    64: tcgen05_instance(64, 12, 2, prefetch_w_k_steps=4),
} | context_instances(
    persistent_instance(GemmSchedule(True, (256, 128), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1), raster_along="n")),
    persistent_instance(GemmSchedule(True, (256, 224), (2, 1), raster_along="n")),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
