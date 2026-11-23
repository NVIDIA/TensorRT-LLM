# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launch the family split-KV planner with the ULL batch-sized grid."""

import torch
import triton

from ..k3_mla.split_kv_plan import (
    _SPLIT_PLAN_KV_TILE,
    _SPLIT_PLAN_MAX_REQUESTS,
    _SPLIT_PLAN_MAX_SPLIT,
    _SPLIT_PLAN_MIN_TILES,
    _SPLIT_PLAN_PIECE_COST_TILES,
    _mla_split_kv_plan_kernel,
    plan_mla_split_kv as plan_mla_split_kv,
)

# The plan kernel is one program over the batch, padded to a power of two of at
# least one warp: its candidate cuts are each a reduction over the batch, so a
# block sized for the maximum request count would make every launch pay for
# _SPLIT_PLAN_MAX_REQUESTS lanes.
_SPLIT_PLAN_MIN_BLOCK = 32
_SPLIT_PLAN_LANES_PER_WARP = 256


def launch_mla_split_kv_plan(
    cache_seqs: torch.Tensor,
    plan: torch.Tensor,
    *,
    num_requests: int,
    rows_per_request: int,
    wave_clusters: int,
    max_split: int = _SPLIT_PLAN_MAX_SPLIT,
    piece_cost: int = _SPLIT_PLAN_PIECE_COST_TILES,
) -> None:
    """Write the per-request split counts for ``cache_seqs`` into ``plan`` (one launch).

    The same plan as ``plan_mla_split_kv`` with the same arguments.
    """
    if not 0 < num_requests <= _SPLIT_PLAN_MAX_REQUESTS:
        raise ValueError(f"split plan supports 1..{_SPLIT_PLAN_MAX_REQUESTS} requests")
    if piece_cost < 0:
        raise ValueError(f"split plan piece cost must be >= 0, got {piece_cost}")
    for name, tensor in (("cache_seqs", cache_seqs), ("plan", plan)):
        if (
            tensor.dtype != torch.int32
            or tensor.ndim != 1
            or tensor.stride(0) != 1
            or tensor.numel() < num_requests
            or not tensor.is_cuda
        ):
            raise ValueError(f"{name} must be a contiguous CUDA int32 vector of >= {num_requests}")
    block = max(_SPLIT_PLAN_MIN_BLOCK, triton.next_power_of_2(num_requests))
    _mla_split_kv_plan_kernel[(1,)](
        cache_seqs,
        plan,
        num_requests,
        max(rows_per_request, 1),
        max(wave_clusters, 1),
        max(max_split, 1),
        KV_TILE=_SPLIT_PLAN_KV_TILE,
        MIN_TILES=_SPLIT_PLAN_MIN_TILES,
        PIECE_COST=int(piece_cost),
        MAX_PIECES=_SPLIT_PLAN_MAX_SPLIT,
        BLOCK=block,
        num_warps=max(1, min(4, block // _SPLIT_PLAN_LANES_PER_WARP)),
    )
