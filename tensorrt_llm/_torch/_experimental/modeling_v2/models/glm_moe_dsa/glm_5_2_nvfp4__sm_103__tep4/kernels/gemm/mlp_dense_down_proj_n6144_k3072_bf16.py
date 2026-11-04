# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 dense-MLP down projection with per-M dispatch.

The local TRT-LLM linear boundary supplies contiguous BF16
``input[M,3072]``, ``weight[6144,3072]``, and ``output[M,6144]`` tensors.
"""

from ._projection_module_dispatch import (
    ProjectionAbi,
    context_instances,
    make_build,
    persistent_instance,
)
from ._swapped_dense_gemm_persistent import GemmSchedule

ABI = ProjectionAbi("mlp.dense.down_proj", 6_144, 3_072)

SCHEDULE_BY_M = {
    1: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    2: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    4: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    8: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    16: persistent_instance(GemmSchedule(False, (64, 16), (4, 1))),
    24: persistent_instance(GemmSchedule(False, (64, 32), (4, 1))),
    32: persistent_instance(GemmSchedule(False, (64, 32), (4, 1))),
    48: persistent_instance(GemmSchedule(True, (128, 64), (4, 1), raster_along="n")),
    64: persistent_instance(GemmSchedule(True, (128, 64), (4, 1), raster_along="n")),
} | context_instances(
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1), raster_along="n")),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
