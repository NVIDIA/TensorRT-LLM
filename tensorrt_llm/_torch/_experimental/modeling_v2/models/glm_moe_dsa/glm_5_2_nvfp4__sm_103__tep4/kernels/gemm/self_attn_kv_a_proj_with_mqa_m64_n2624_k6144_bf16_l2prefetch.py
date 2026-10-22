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
} | context_instances(
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 256), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 192), (2, 1))),
    persistent_instance(GemmSchedule(True, (256, 256), (2, 1))),
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
