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
"""One-launch paged attention inputs for the DSpark rolling window."""

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit(do_not_specialize=["num_rows"])
def _ring_metadata_kernel(
    slots_ptr,
    start_pos_ptr,
    valid_len_ptr,
    block_tables_ptr,
    positions_ptr,
    valid_lengths_ptr,
    capacities_ptr,
    num_rows,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < num_rows
    slot = tl.load(slots_ptr + rows, mask=mask).to(tl.int32)
    start = tl.load(start_pos_ptr + rows, mask=mask).to(tl.int64)
    tl.store(block_tables_ptr + 2 * rows, slot, mask=mask)
    tl.store(block_tables_ptr + 2 * rows + 1, slot, mask=mask)
    tl.store(positions_ptr + rows, start % WINDOW + WINDOW + 1, mask=mask)
    tl.store(
        valid_lengths_ptr + rows, tl.load(valid_len_ptr + rows, mask=mask).to(tl.int64), mask=mask
    )
    tl.store(capacities_ptr + rows, tl.full([BLOCK], 2 * WINDOW, tl.int64), mask=mask)


def dspark_ring_metadata(
    slots: torch.Tensor, start_pos: torch.Tensor, valid_len: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block tables, positions, valid lengths and capacities that alias each slot's window twice."""
    num_rows = slots.shape[0]
    device = slots.device
    block_tables = torch.empty((num_rows, 2), dtype=torch.int32, device=device)
    positions = torch.empty((num_rows,), dtype=torch.int64, device=device)
    valid_lengths = torch.empty((num_rows,), dtype=torch.int64, device=device)
    capacities = torch.empty((num_rows,), dtype=torch.int64, device=device)
    if num_rows == 0:
        return block_tables, positions, valid_lengths, capacities
    _ring_metadata_kernel[(triton.cdiv(num_rows, _BLOCK),)](
        slots.contiguous(),
        start_pos.contiguous(),
        valid_len.contiguous(),
        block_tables,
        positions,
        valid_lengths,
        capacities,
        num_rows,
        WINDOW=window_size,
        BLOCK=_BLOCK,
    )
    return block_tables, positions, valid_lengths, capacities
