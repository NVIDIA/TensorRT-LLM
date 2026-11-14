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
# DSpark / DFlash drafter step inputs: one launch per step
# =============================================================================
#
# It replaces, bit for bit, the start of DFlashWorker.prepare_1st_drafter_inputs' generation branch, for a drafter whose
# token embedding is a plain row gather of a replicated table:
#   bonus[r]          = accepted[r, max(num_accepted[r] - 1, 0)]                       (int64)
#   query_pos[r, j]   = min(ctx_len[slot[r]] + num_accepted[r], max_ctx) + j,  j < block (int64)
#   ctx_pos[r, j]     = ctx_len[slot[r]] + j,                                 j < K + 1 (int64)
#   noise[r, 0, :]    = table[bonus[r]], or +0 for an id outside [0, vocab) (m3_embedding's masked_fill)
#   noise[r, j>0, :]  = mask_row   (the drafter's trained mask embedding, else table[mask_token_id])
# Integer arithmetic and row copies only, so the result is exact; the stock path spends a launch per torch op on it.
#
# The launch is the C++ op family_ops().minimax_m31_dspark_drafter_inputs(_refresh)
# (the family's csrc/m31/dsparkDrafterInputsSm107.cu; no Triton in served paths). It replaced the former Triton
# kernels bit for bit on every input class. Without the op (a build without it), the wrappers return None and the caller
# keeps the stock path.

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

from .dspark_ctx_advance import _GLUE_CPP_OFF, _glue_log

_OP = "minimax_m31_dspark_drafter_inputs"
_OP_REFRESH = "minimax_m31_dspark_drafter_inputs_refresh"


def _op(name: str):
    if _GLUE_CPP_OFF:
        return _glue_log(name, "killed (TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1)")
    return getattr(family_ops(), name, None)


def _inputs_supported(
    accepted, num_accepted, slots, ctx_len, table, mask_row, block_size, tokens_per_request
):
    n = num_accepted.shape[0]
    return (
        accepted.dim() == 2
        and accepted.shape[0] == n
        and accepted.stride(1) == 1
        and accepted.dtype in (torch.int32, torch.int64)
        and num_accepted.dtype in (torch.int32, torch.int64)
        and slots.shape == (n,)
        and ctx_len.dtype == torch.int64
        and ctx_len.dim() == 1
        and ctx_len.is_contiguous()
        and table.dim() == 2
        and table.is_contiguous()
        and mask_row.shape == (table.shape[1],)
        and mask_row.dtype == table.dtype
        and mask_row.is_contiguous()
        and block_size >= 1
        and tokens_per_request >= 1
    )


def _outputs(n, block_size, tokens_per_request, table):
    dev = table.device
    return (
        torch.empty(n, dtype=torch.int64, device=dev),
        torch.empty(n, block_size, dtype=torch.int64, device=dev),
        torch.empty(n, tokens_per_request, dtype=torch.int64, device=dev),
        torch.empty(n, block_size, table.shape[1], dtype=table.dtype, device=dev),
    )


def drafter_inputs_and_refresh(
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    max_ctx: int,
    table: torch.Tensor,
    mask_row: torch.Tensor,
    block_size: int,
    tokens_per_request: int,
    offsets: torch.Tensor,
    pool_idx: int,
    num_seqs: int,
    divisor: int,
    tables: torch.Tensor,
    counts: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """drafter_inputs' four outputs, with this step's block-table refresh (what dspark_block_tables.refresh_block_tables
    writes into ``tables[:num_seqs]`` / ``counts[:num_seqs]``) decoded in the same launch. None, with nothing written,
    when either half is not a layout it reproduces, the refresh rows are not the input rows (``num_seqs`` must equal
    the request count), or the op is not built: the caller then runs both entries on their own."""
    from .dspark_block_tables import supported as refresh_supported

    n = num_accepted.shape[0]
    op = _op(_OP_REFRESH)
    if not (
        op is not None
        and n >= 1
        and num_seqs == n
        and _inputs_supported(
            accepted, num_accepted, slots, ctx_len, table, mask_row, block_size, tokens_per_request
        )
        and refresh_supported(offsets, tables, counts, divisor)
        and tables.shape[0] >= n
        and counts.shape[0] >= n
        and counts.is_contiguous()
    ):
        return None
    bonus, qpos, cpos, noise = _outputs(n, block_size, tokens_per_request, table)
    op(
        accepted,
        num_accepted.contiguous(),
        slots.to(torch.int64).contiguous(),
        ctx_len,
        int(max_ctx),
        table,
        mask_row,
        bonus,
        qpos,
        cpos,
        noise,
        offsets[pool_idx, :n, 0],
        int(divisor),
        tables,
        counts,
    )
    return bonus, qpos, cpos, noise


def drafter_inputs(
    accepted: torch.Tensor,
    num_accepted: torch.Tensor,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    max_ctx: int,
    table: torch.Tensor,
    mask_row: torch.Tensor,
    block_size: int,
    tokens_per_request: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(bonus [n] int64, query positions [n, block_size] int64, ctx positions [n, tokens_per_request] int64,
    noise block [n, block_size, hidden]) in one launch, or None when the inputs are not a layout the kernel reproduces
    or the op is not built (the caller keeps the stock path). ``mask_row`` [hidden] is the row every slot after the
    first takes."""
    op = _op(_OP)
    if op is None or not _inputs_supported(
        accepted, num_accepted, slots, ctx_len, table, mask_row, block_size, tokens_per_request
    ):
        return None
    n = num_accepted.shape[0]
    bonus, qpos, cpos, noise = _outputs(n, block_size, tokens_per_request, table)
    if n == 0:
        return bonus, qpos, cpos, noise
    op(
        accepted,
        num_accepted.contiguous(),
        slots.to(torch.int64).contiguous(),
        ctx_len,
        int(max_ctx),
        table,
        mask_row,
        bonus,
        qpos,
        cpos,
        noise,
    )
    return bonus, qpos, cpos, noise
