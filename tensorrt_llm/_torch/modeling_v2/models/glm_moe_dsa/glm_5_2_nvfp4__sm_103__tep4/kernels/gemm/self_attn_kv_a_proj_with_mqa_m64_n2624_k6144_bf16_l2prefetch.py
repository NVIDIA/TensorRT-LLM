# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 MLA KV-A projection with L2 weight prefetch for the large decode M.

Used in decode chains where the kv_a GEMM runs right behind the fused
allreduce + residual-add + RMSNorm kernel, which triggers its PDL dependents
at entry. For M=48/64 the tcgen05 schedule pulls its
whole per-CTA weight slice into L2 with ``cp.async.bulk.prefetch.tensor`` before
``griddepcontrol.wait``, so the 32 MB weight read happens while the allreduce is
still waiting on NVLink. M<=32 keeps the shipped low-latency TGV schedules,
which are ~2.5 us faster than any tcgen05 variant there when both are queued
behind the allreduce in a CUDA graph. Context M values keep the shipped persistent
schedules.

The public registry ABI is ``build() -> launcher(input, weight, output)`` for
contiguous BF16 ``[M,6144] x [2624,6144] -> [M,2624]`` tensors.
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

ABI = ProjectionAbi("self_attn.kv_a_proj_with_mqa", 2_624, 6_144)

# mma_n=64, stages=12, split_k=3 -> 41 x 3 = 123 CTAs, 32 local K steps of 64;
# prefetch all 32 steps (256 KB per CTA, 32 MB total) into L2 ahead of the wait.
# With the allreduce and kv_a captured in one CUDA graph and weights streamed from
# HBM, (64,12,3) is the fastest schedule at M=64 (about 8 us, against about 11 us for
# cuBLAS); split_k >= 4 and mma_n 128 are slower. At M=2 the TGV schedule beats every
# tcgen05 variant and cuBLAS.
_DECODE_LARGE = tcgen05_instance(64, 12, 3, prefetch_w_k_steps=32)
_TGV_SMALL = tgv_instance(TgvSchedule(8, 128, 8, split_k=2))

SCHEDULE_BY_M = {
    1: _TGV_SMALL,
    2: _TGV_SMALL,
    4: _TGV_SMALL,
    8: _TGV_SMALL,
    16: tgv_instance(TgvSchedule(16, 128, 8, split_k=2)),
    32: tgv_instance(TgvSchedule(32, 128, 8, split_k=2)),
    48: _DECODE_LARGE,
    64: _DECODE_LARGE,
    # MTP3 / MTP5 (the tcgen05 cells stop at 64): persistent 2-CTA (128, 64), 21 x 2 / 3
    # clusters. B300, cold weights, us: 128 rows 11.07-11.14 ((64, 64) c1x1 11.41, c4x1
    # 12.06-12.42, TGV split-K 2 15.69); 192 rows 11.32-11.36 ((128, 96) 11.62, (64, 64)
    # 12.12, (64, 96) 14.58-14.93, TGV split-K 3 16.29); the PDL wait/signal role measured
    # 0.2-0.3 us slower here. The context MXF8 quant + GEMM those decode rows took: 13.24-14.03
    # / 13.76-14.61; the bf16 prefill GEMM 22.93 / 22.92.
    128: persistent_instance(GemmSchedule(True, (128, 64), (2, 1), raster_along="n")),
    192: persistent_instance(GemmSchedule(True, (128, 64), (2, 1), raster_along="n")),
} | context_instances(
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 256), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 256), (2, 1))),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)
# The decode row counts above 64, built only for an engine drafting more than one token.
MTP_DECODE_M_VALUES = (128, 192)

build = make_build(ABI, SCHEDULE_BY_M)
