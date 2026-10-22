# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 shared-expert down projection with per-M dispatch.

The public registry ABI is ``build() -> launcher(input, weight, output)`` for
contiguous BF16 ``[M,512] x [6144,512] -> [M,6144]`` tensors.

Runs on SM100+ (Blackwell) GPUs.  Supported shapes: M in
``SUPPORTED_M_VALUES``, N=6144, K=512; BF16 input, weight and output.
"""

from ._projection_module_dispatch import (
    ProjectionAbi,
    context_instances,
    make_build,
    persistent_instance,
    tgv_instance,
)
from ._swapped_dense_gemm_persistent import GemmSchedule
from ._swapped_tgv_gemm import TgvSchedule

ABI = ProjectionAbi("mlp.shared_experts.down_proj", 6_144, 512)


def schedule_for_m(active_m: int) -> TgvSchedule:
    """Select the measured 96-CTA cold-weight schedule for ``M`` in ``[1,64]``."""
    if not 1 <= active_m <= 64:
        raise ValueError(f"M must be in [1,64], got {active_m}")
    if active_m <= 8:
        cta_n = 8
    elif active_m <= 16:
        cta_n = 16
    elif active_m <= 32:
        cta_n = 32
    elif active_m <= 48:
        cta_n = 48
    else:
        cta_n = 64
    return TgvSchedule(cta_n, 128, 4)


SCHEDULE_BY_M = {
    active_m: tgv_instance(schedule_for_m(active_m)) for active_m in range(1, 65)
} | context_instances(
    persistent_instance(GemmSchedule(True, (128, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 256), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 256), (2, 1))),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
