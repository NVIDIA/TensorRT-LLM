# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 DSA indexer WQ-B projection with per-M dispatch.

At TP4 this module has the same contiguous BF16
``[M,2048] x [4096,2048] -> [M,4096]`` ABI as MLA Q-B.  It intentionally owns
an identical schedule table because the shape coincidence disappears at other
TP sizes.
"""

from . import _projection_module_dispatch as _projectionmoduledispatch
from . import _swapped_dense_gemm_pdl_chain as _swappeddensegemmpdlchain
from . import _swapped_dense_gemm_persistent as _swappeddensegemmpersistent

ProjectionAbi = _projectionmoduledispatch.ProjectionAbi
context_instances = _projectionmoduledispatch.context_instances
make_build = _projectionmoduledispatch.make_build
persistent_instance = _projectionmoduledispatch.persistent_instance
tcgen05_instance = _projectionmoduledispatch.tcgen05_instance

GemmSchedule = _swappeddensegemmpersistent.GemmSchedule

# The decode row counts above 64 (32 requests x MTP3 / MTP5 widths 4 / 6), built only
# for an engine drafting more than one token.
MTP_DECODE_M_VALUES = (128, 192)

ABI = ProjectionAbi(
    "self_attn.indexer.wq_b",
    4_096,
    2_048,
    # The MTP cells take the wait/signal PDL role with the weight preloaded ahead of the
    # wait (the tcgen05 cells prefetch their first weight steps instead).
    pdl_persistent_compiler=_swappeddensegemmpdlchain.compile_pdl_wait_signal_persistent_gemm,
    pdl_enabled_m=frozenset(MTP_DECODE_M_VALUES),
)

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
    # MTP3 / MTP5 (the tcgen05 cells stop at 64): q_b's MTP cells, measured on this ABI.
    # B300, cold weights, us: 5.59 / 5.83 at 128 / 192 rows, against cuBLAS (the path those
    # rows took; bitwise the same output) at 5.56 / 5.74; 6.34 / 6.71 with no PDL role.
    128: persistent_instance(GemmSchedule(True, (128, 64), (2, 1), raster_along="n")),
    192: persistent_instance(GemmSchedule(True, (128, 96), (2, 1), raster_along="n")),
} | context_instances(
    persistent_instance(GemmSchedule(True, (256, 128), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1), raster_along="n")),
    persistent_instance(GemmSchedule(True, (256, 224), (2, 1), raster_along="n")),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
