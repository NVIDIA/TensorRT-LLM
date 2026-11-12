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

"""The GPU-validated B1/K7 integer preparation kernel, unchanged."""

from typing import NamedTuple

import torch
import triton
import triton.language as tl


class Inputs(NamedTuple):
    accepted: torch.Tensor
    count: torch.Tensor
    slots: torch.Tensor
    ctx_len: torch.Tensor
    block_counts: torch.Tensor


class Outputs(NamedTuple):
    bonus: torch.Tensor
    lookup_ids: torch.Tensor
    qpos: torch.Tensor
    cpos: torch.Tensor
    n64: torch.Tensor
    columns: torch.Tensor
    write_mask: torch.Tensor
    gen_rows: torch.Tensor
    offsets: torch.Tensor


@triton.jit
def _prepare_kernel(
    accepted_ptr,
    count_ptr,
    slots_ptr,
    ctx_len_ptr,
    blocks_ptr,
    bonus_ptr,
    lookup_ptr,
    query_ptr,
    context_ptr,
    n64_ptr,
    columns_ptr,
    mask_ptr,
    rows_ptr,
    offsets_ptr,
    MAX_CTX: tl.constexpr,
    MASK_TOKEN: tl.constexpr,
):
    lane = tl.arange(0, 32)
    i64 = lane.to(tl.int64)
    count = tl.load(count_ptr).to(tl.int64)
    slot = tl.load(slots_ptr).to(tl.int64)
    old = tl.load(ctx_len_ptr + slot).to(tl.int64)
    capacity = tl.load(blocks_ptr).to(tl.int64) * 64
    bonus = tl.load(accepted_ptr + tl.maximum(count - 1, 0)).to(tl.int64)
    now = tl.minimum(old + count, tl.full((), MAX_CTX, tl.int64))
    positions = old + i64
    columns = tl.maximum(tl.minimum(positions, capacity - 1), 0)

    tl.store(bonus_ptr, bonus)
    tl.store(n64_ptr, count)
    tl.store(rows_ptr, tl.full((), 0, tl.int64))
    lookup = tl.where(lane == 0, bonus, tl.full((), MASK_TOKEN, tl.int64))
    tl.store(lookup_ptr + lane, lookup, mask=lane < 2)
    tl.store(query_ptr + lane, now + i64, mask=lane < 7)
    tl.store(context_ptr + lane, positions, mask=lane < 8)
    tl.store(columns_ptr + lane, columns, mask=lane < 8)
    tl.store(mask_ptr + lane, i64 < count, mask=lane < 8)
    tl.store(offsets_ptr + lane, i64, mask=lane < 8)
