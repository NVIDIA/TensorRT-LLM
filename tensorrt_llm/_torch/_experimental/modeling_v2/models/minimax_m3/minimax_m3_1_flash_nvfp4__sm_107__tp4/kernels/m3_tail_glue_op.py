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
"""TAILGLUE_A's launch: the C++ op ``minimax_m31_tail_glue_a`` (csrc/tailGlueSm107.cu).

One launch of the DRAFTER_CTX_KV_STORE_FUSED context store (columns from the positions on a bound pool), the
DRAFTER_CTX_ADVANCE_FUSED context advance and read view, and the DRAFTER_FORWARD_INDICES_FUSED forward indices of the
drafter's paged block decode, bit for bit those four launches and the Triton kernel of ``m3_tail_glue`` (kept as the
test oracle; no Triton kernel serves on this target).
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
    native_ops,
)


def available() -> bool:
    """Whether this build carries the C++ op."""
    return hasattr(native_ops, "minimax_m31_tail_glue_a")


def launch(
    store,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    max_ctx: int,
    dummy_slot: int,
    base_pages: torch.Tensor,
    tables_read: torch.Tensor,
    page_size: int,
    block_size: int,
    query_positions: torch.Tensor,
    iblock: int,
    ipage: int,
):
    """``m3_tail_glue.launch``'s contract on the C++ op: ``store`` is the DRAFTER_CTX_KV_STORE_FUSED store's
    ``StoreOperands`` (COL_FROM_POS and BOUND). Returns (advertised context lengths [n] int64, (cache_seqlens,
    cache_batch_idx, seq_lens_after, kv_indptr, kv_last_page_len, append_positions, query_positions_flat) int32). The
    caller checks every operand (``TailGlue.advance``); the op raises a RuntimeError, with nothing enqueued, on an
    operand it does not take."""
    meta = store.meta
    rows = store.request_rows
    n = rows.shape[0]
    dev = ctx_len.device
    out = torch.empty(n, dtype=torch.int64, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    seqlens = torch.empty(n, **i32)
    batch_out = torch.empty(n, **i32)
    after = torch.empty(n, **i32)
    indptr = torch.empty(n + 1, **i32)
    last = torch.empty(n, **i32)
    append = torch.empty(n * iblock, **i32)
    qflat = torch.empty(n * iblock, **i32)
    native_ops.minimax_m31_tail_glue_a(
        store.kv_flat,
        store.k,
        store.w,
        store.cos_sin,
        store.positions,
        store.num_accepted,
        rows,
        store.counts,
        store.block_table,
        store.cache,
        store.cache_stride_layer,
        meta["KP1"],
        meta["L"],
        meta["NKV"],
        meta["HD"],
        meta["PS"],
        meta["PTR_LIST"],
        meta["FOLD_W"],
        slots,
        ctx_len,
        base_pages,
        tables_read,
        out,
        int(max_ctx),
        int(dummy_slot),
        int(page_size),
        int(block_size),
        query_positions,
        seqlens,
        batch_out,
        after,
        indptr,
        last,
        append,
        qflat,
        int(iblock),
        int(ipage),
    )
    return out, (seqlens, batch_out, after, indptr, last, append, qflat)
