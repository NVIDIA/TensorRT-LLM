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


SCHEDULE_BY_M = (
    {active_m: tgv_instance(schedule_for_m(active_m)) for active_m in range(1, 65)}
    | {
        # MTP3 / MTP5 decode batches (32 sequences x 4 / 6 rows), 96 CTAs each. B300, cold
        # weights: 4.25 / 5.41 us, against 6.98 / 7.00 us on the shared expert's prefill down
        # kernel and 6.04 us on the prefill GEMM those rows took.
        128: tgv_instance(TgvSchedule(128, 128, 4)),
        192: persistent_instance(GemmSchedule(False, (64, 192), (4, 1))),
    }
    | context_instances(
        persistent_instance(GemmSchedule(True, (128, 192), (2, 1))),
        persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
        persistent_instance(GemmSchedule(True, (256, 256), (2, 1))),
        persistent_instance(GemmSchedule(True, (256, 256), (2, 1))),
    )
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)
# The decode row counts above 64, built only for an engine drafting more than one token.
MTP_DECODE_M_VALUES = (128, 192)

build = make_build(ABI, SCHEDULE_BY_M)
