# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 MoE router projection with per-M dispatch.

TRT-LLM calls the router hook with contiguous BF16 ``input[M,6144]`` and
``weight[256,6144]`` and supplies an FP32 ``output[M,256]``.  ``build`` keeps
the registry's no-argument factory ABI and returns the exact three-pointer
launcher.  M dispatch and compilation happen on the first call, before CUDA
Graph replay.

Runs on SM100+ (Blackwell) GPUs.  Supported shapes: M in
``SUPPORTED_M_VALUES``, N=256, K=6144; BF16 input and weight, FP32 output.
"""

import torch

from ._projection_module_dispatch import (
    FP32_OUTPUT_ABI,
    ProjectionAbi,
    context_instances,
    make_build,
    persistent_instance,
    tcgen05_instance,
    tgv_instance,
)
from ._swapped_dense_gemm_persistent import GemmSchedule
from ._swapped_tgv_gemm import TgvSchedule

ABI = ProjectionAbi(
    "mlp.gate",
    output_features=256,
    input_features=6_144,
    output_dtype=torch.float32,
    tgv_elements=FP32_OUTPUT_ABI,
)

SCHEDULE_BY_M = {
    1: tgv_instance(TgvSchedule(16, 128, 4, split_k=8)),
    2: tgv_instance(TgvSchedule(16, 64, 8, split_k=8)),
    4: tgv_instance(TgvSchedule(16, 128, 4, split_k=8)),
    8: tgv_instance(TgvSchedule(16, 64, 8, split_k=8)),
    16: tgv_instance(TgvSchedule(16, 64, 6, split_k=8)),
    32: tgv_instance(TgvSchedule(16, 64, 8, split_k=8)),
    48: tcgen05_instance(16, 8, 6),
    64: tcgen05_instance(16, 8, 6),
} | context_instances(
    tgv_instance(TgvSchedule(64, 64, 12, split_k=2)),
    tgv_instance(TgvSchedule(96, 128, 5)),
    persistent_instance(GemmSchedule(True, (256, 128), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 128), (2, 1), raster_along="n")),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
