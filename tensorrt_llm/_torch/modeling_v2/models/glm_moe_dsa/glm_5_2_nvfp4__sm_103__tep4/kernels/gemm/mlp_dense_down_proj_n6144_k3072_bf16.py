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
from ._swapped_dense_gemm_pdl_chain import compile_pdl_wait_signal_persistent_gemm
from ._swapped_dense_gemm_persistent import GemmSchedule

# The decode row counts above 64 (32 requests x MTP3 / MTP5 widths 4 / 6), built only
# for an engine drafting more than one token.
MTP_DECODE_M_VALUES = (128, 192)

ABI = ProjectionAbi(
    "mlp.dense.down_proj",
    6_144,
    3_072,
    # The MTP cells take the wait/signal PDL role with the weight preloaded ahead of the
    # wait (1.0-1.1 us faster than without at both row counts).
    pdl_persistent_compiler=compile_pdl_wait_signal_persistent_gemm,
    pdl_enabled_m=frozenset(MTP_DECODE_M_VALUES),
)

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
    # MTP3 / MTP5: 2-CTA tiles over every row, 96 CTAs, PDL role. B300, cold weights, us:
    # 128 rows 9.27 ((128, 128) c4x1 10.53-11.03, (64, 128) 12.22, (64, 64) 12.73); 192 rows
    # 10.21 ((128, 192) c4x1 11.48-11.86, (64, 192) 14.68). The context MXF8 quant + GEMM
    # those decode rows took: 10.40-11.07 / 10.46-11.23; the bf16 prefill GEMM 14.46 / 14.60.
    128: persistent_instance(GemmSchedule(True, (128, 128), (2, 1), raster_along="n")),
    192: persistent_instance(GemmSchedule(True, (128, 192), (2, 1), raster_along="n")),
} | context_instances(
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1), raster_along="n")),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
