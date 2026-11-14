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
# =============================================================================
# DSpark / DFlash drafter block-table refresh: one launch per step
# =============================================================================
#
# It replaces, bit for bit, what DFlashWorker._refresh_ctx_block_tables does with the draft pool's encoded block
# offsets of this step (field 0 of the K/V pair, pool ``pool_idx``, the first ``num_seqs`` rows):
#   counts[r] = number of entries >= 0 in row r (placeholders are negative)
#   tables[r, j] = max(offset[r, j], 0) // divisor   (int32)
# Integer arithmetic only, so the result is exact; the stock path spends one launch per torch op on it.
#
# The launch is the C++ op family_ops().minimax_m31_dspark_refresh_block_tables
# (the family's csrc/m31/dsparkBlockTables.cu; no Triton in served paths). It replaced the former Triton kernels
# (_refresh_kernel and the grid form's _refresh_kernel_2d) bit for bit on every input class.
# Without the op (a build without it), refresh_block_tables returns False and the caller keeps the stock path.

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

from .dspark_ctx_advance import _GLUE_CPP_OFF, _glue_log

_OP = "minimax_m31_dspark_refresh_block_tables"
_MAX_WIDTH = 65534 * 2048  # the op's launch grid (dsparkBlockTables.h kDsparkRefreshMaxWidth)


def refresh_block_tables(
    offsets: torch.Tensor,
    pool_idx: int,
    num_seqs: int,
    divisor: int,
    tables: torch.Tensor,
    counts: torch.Tensor,
) -> bool:
    """Decode this step's draft block offsets ``offsets[pool_idx, :num_seqs, 0]`` into ``tables[:num_seqs]`` and their
    non-placeholder counts into ``counts[:num_seqs]``, in one launch; returns False (nothing written) when the layout
    is not one the op reproduces or the op is not built, so the caller keeps the stock path."""
    if _GLUE_CPP_OFF:
        _glue_log(_OP, "killed (TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1)")
        return False
    op = getattr(family_ops(), _OP, None)
    if (
        op is None
        or not supported(offsets, tables, counts, divisor)
        or not offsets.is_cuda
        or not counts.is_contiguous()
        or offsets.shape[-1] > _MAX_WIDTH
    ):
        return False
    if num_seqs <= 0:
        return True
    if offsets.shape[1] < num_seqs or tables.shape[0] < num_seqs or counts.shape[0] < num_seqs:
        return False
    op(offsets[pool_idx, :num_seqs, 0], int(divisor), tables, counts)
    return True


_parallel = [False]  # the block-table refresh grid form (see enable_parallel_refresh)


def enable_parallel_refresh(enabled: bool = True) -> bool:
    """The grid form's flag (DRAFTER_BLOCK_TABLE_REFRESH_GRID), kept for the targets' engagement bookkeeping. It chose
    the former Triton (row, chunk) int32 kernel; the C++ op picks its grid from the width (both former modes wrote the
    same bytes), so the flag no longer changes the launch."""
    _parallel[0] = bool(enabled)
    return _parallel[0]


def supported(
    offsets: torch.Tensor, tables: torch.Tensor, counts: torch.Tensor, divisor: Optional[int]
) -> bool:
    """[pools, rows, 2, width] int32 / int64 offsets on the tables' device; int32 row-major tables of the same width;
    int64 counts; a positive integer divisor."""
    return (
        offsets.dim() == 4
        and offsets.dtype in (torch.int32, torch.int64)
        and tables.dtype == torch.int32
        and tables.dim() == 2
        and tables.stride(1) == 1
        and tables.shape[1] == offsets.shape[-1]
        and counts.dtype == torch.int64
        and offsets.device == tables.device == counts.device
        and isinstance(divisor, int)
        and divisor >= 1
    )
