# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 shared-expert gate/up projection with per-M dispatch.

The public registry ABI is ``build() -> launcher(input, weight, output)`` for
contiguous BF16 ``[M,6144] x [1024,6144] -> [M,1024]`` tensors.

Runs on SM100+ (Blackwell) GPUs.  Supported shapes: M in
``SUPPORTED_M_VALUES``, N=1024, K=6144; BF16 input, weight and output.
"""

from ._projection_module_dispatch import (
    ProjectionAbi,
    context_instances,
    make_build,
    persistent_instance,
    tcgen05_instance,
    tgv_instance,
)
from ._swapped_dense_gemm_persistent import GemmSchedule
from ._swapped_tgv_gemm import TgvSchedule

ABI = ProjectionAbi("mlp.shared_experts.gate_up_proj", 1_024, 6_144)

SCHEDULE_BY_M = {
    1: tgv_instance(TgvSchedule(8, 128, 8, split_k=4)),
    2: tgv_instance(TgvSchedule(8, 128, 8, split_k=4)),
    4: tgv_instance(TgvSchedule(8, 128, 8, split_k=4)),
    8: tgv_instance(TgvSchedule(8, 128, 8, split_k=4)),
    16: tgv_instance(TgvSchedule(16, 128, 8, split_k=4)),
    32: tgv_instance(TgvSchedule(32, 128, 8, split_k=4)),
    48: tcgen05_instance(32, 8, 3),
    64: tcgen05_instance(32, 8, 3),
} | context_instances(
    persistent_instance(GemmSchedule(True, (128, 128), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 208), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 256), (2, 2))),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
