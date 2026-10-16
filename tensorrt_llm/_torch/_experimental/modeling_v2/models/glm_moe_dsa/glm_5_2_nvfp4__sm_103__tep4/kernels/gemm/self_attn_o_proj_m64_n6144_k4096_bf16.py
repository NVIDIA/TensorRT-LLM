# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 MLA output projection with per-M dispatch.

The public registry ABI is ``build() -> launcher(input, weight, output)`` for
contiguous BF16 ``[M,4096] x [6144,4096] -> [M,6144]`` tensors.  The measured
M1--M64 crossover keeps the existing persistent backend for this shape.

Runs on SM100+ (Blackwell) GPUs.  Supported shapes: M in
``SUPPORTED_M_VALUES``, N=6144, K=4096; BF16 input, weight and output.
"""

from . import _projection_module_dispatch as _projectionmoduledispatch
from . import _swapped_dense_gemm_pdl_chain as _swappeddensegemmpdlchain
from . import _swapped_dense_gemm_persistent as _swappeddensegemmpersistent
from . import _swapped_tgv_gemm as _swappedtgvgemm

ProjectionAbi = _projectionmoduledispatch.ProjectionAbi
context_instances = _projectionmoduledispatch.context_instances
make_build = _projectionmoduledispatch.make_build
persistent_instance = _projectionmoduledispatch.persistent_instance
tgv_instance = _projectionmoduledispatch.tgv_instance
GemmSchedule = _swappeddensegemmpersistent.GemmSchedule
TgvSchedule = _swappedtgvgemm.TgvSchedule

compile_pdl_wait_signal_persistent_gemm = (
    _swappeddensegemmpdlchain.compile_pdl_wait_signal_persistent_gemm
)

# Decode row counts of the serving CUDA-graph buckets (1/2/4/8/16/24/32 requests
# x 2 MTP rows) plus the M=1 cell.
DECODE_M_VALUES = (1, 2, 4, 8, 16, 32, 48, 64)

ABI = ProjectionAbi(
    "self_attn.o_proj",
    6_144,
    4_096,
    kernel_name_prefix="glm52_self_attn_o_proj",
    pdl_persistent_compiler=compile_pdl_wait_signal_persistent_gemm,
    # The wait/signal PDL role applies at every persistent decode cell (no
    # measurable standalone cost; the gain is overlap with the neighbor kernels).  M = 8 is a
    # TGV cell and carries
    # the TGV template's own PDL instead; the catalog validator rejects the
    # persistent role on TGV rows.
    pdl_enabled_m=frozenset(DECODE_M_VALUES) - {8},
)

# Schedules chosen per M from B300 measurements. The M=64 (64,64) schedule is
# much slower below M=64, so each decode M has its own entry.
SCHEDULE_BY_M = {
    1: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    2: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    4: persistent_instance(GemmSchedule(False, (64, 8), (4, 1))),
    8: tgv_instance(TgvSchedule(8, 128, 4)),
    16: persistent_instance(GemmSchedule(False, (64, 16), (4, 1))),
    32: persistent_instance(GemmSchedule(False, (64, 32), (4, 1))),
    48: persistent_instance(GemmSchedule(True, (128, 64), (4, 1), raster_along="n")),
    64: persistent_instance(GemmSchedule(False, (64, 64), (4, 1), raster_along="n")),
} | context_instances(
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1), raster_along="n")),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
