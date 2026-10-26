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
# DSpark / DFlash drafter context advance + read view: one launch per step
# =============================================================================
#
# It replaces, bit for bit, what DFlashWorker's generation branch does after the context store on a bound draft pool
# (_advance_ctx_len, then the advertised context length and the rebased read view of dflash_rebased_ctx_view):
#   ctx_len[slot] = min(ctx_len[slot] + num_accepted, max_ctx); ctx_len[dummy_slot] = 0
#   num_ctx       = min(ctx_len[slot], max(block_counts[row] * page - block_size, 0))
#   rebase        = min(base_pages[slot], num_ctx // page)
#   num_ctx_out   = max(num_ctx - base_pages[slot] * page, 0)                                   (int64)
#   read[row, j]  = tables[row, min(j + rebase, width - 1)]                                     (int32)
# One program per generation request, so each request's advance is read back by the same program; rows on the dummy
# slot (padding) skip the add and read 0, which the stock path's zeroing after the add makes them read too.
#
# ADV_PAR (``parallel=True``, chosen by the caller): at batch 1 the one program copies the whole read-view row
# (width = the pool's max blocks per sequence) alone. With it, the per-request program does the advance only and
# stores its rebase, and a second launch copies the row with one program per _COPY_BLOCK columns. The same values are
# stored (a pure index copy), so bit for bit the one-launch path; one launch cannot split the copy, since its programs
# would race the context-length store.

import functools
import os
from typing import Optional

import torch
import triton
import triton.language as tl

_adv_par_logged = False
_COPY_BLOCK = 256

# DSPARK_GLUE_CPP: the C++ op (torch.ops.trtllm.minimax_m31_dspark_advance, kernels/minimaxM31/dsparkGlueSm107.cu,
# bit for bit the one-launch Triton kernel) on SM107 when it is built, for the one-launch layout (parallel=False); the
# Triton kernels otherwise (other GPUs, parallel=True) or with TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1 (read once, at
# import).
_GLUE_CPP_OFF = os.environ.get("TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP", "").strip().lower() in (
    "1",
    "true",
    "on",
    "yes",
)


@functools.lru_cache(maxsize=None)
def _is_sm107(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 7)


def _glue_cpp_op(name: str, device: torch.device):
    """The C++ op, or None with the reason logged once (the Triton kernel runs)."""
    if _GLUE_CPP_OFF:
        return _glue_log(name, "killed (TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1)")
    if not _is_sm107(device.index if device.index is not None else torch.cuda.current_device()):
        return _glue_log(name, "not an sm_107 device")
    op = getattr(torch.ops.trtllm, name, None)
    return op if op is not None else _glue_log(name, "the op is not registered")


_glue_logged = set()


def _glue_log(name: str, why: Optional[str] = None):
    """Engagement line, once per (op, outcome) per process: "[M3 dispatch] DSPARK_GLUE_CPP taken: <op>" when the C++ op
    runs, "... not taken: <op>: <why>; Triton kernel" when it falls back. Returns None."""
    if (name, why) not in _glue_logged:
        _glue_logged.add((name, why))
        from tensorrt_llm.logger import logger

        if why is None:
            logger.info(f"[M3 dispatch] DSPARK_GLUE_CPP taken: {name}")
        else:
            logger.info(f"[M3 dispatch] DSPARK_GLUE_CPP not taken: {name}: {why}; Triton kernel")
    return None


@triton.jit
def _advance_kernel(
    slots_ptr,  # [n] int64
    rows_ptr,  # [n] int64: block-table rows of the requests
    nacc_ptr,  # [n] int
    ctx_len_ptr,  # [slots] int64, in / out
    counts_ptr,  # [rows] int64
    base_ptr,  # [slots] int64
    tables_ptr,  # int32 [rows, width]
    read_ptr,  # int32 [rows, width]
    out_ptr,  # [n] int64 advertised context lengths
    width,
    tables_stride,
    read_stride,
    max_ctx,
    dummy_slot,
    page,
    block_size,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + r)
    row = tl.load(rows_ptr + r)
    is_dummy = slot == dummy_slot
    old = tl.load(ctx_len_ptr + slot)
    new = tl.minimum(old + tl.load(nacc_ptr + r).to(tl.int64), max_ctx)
    if not is_dummy:
        tl.store(ctx_len_ptr + slot, new)
    if r == 0:
        tl.store(ctx_len_ptr + dummy_slot, tl.zeros((), dtype=tl.int64))
    new = tl.where(is_dummy, 0, new)
    allocated = tl.maximum(tl.load(counts_ptr + row) * page - block_size, 0)
    num_ctx = tl.minimum(new, allocated)
    base = tl.load(base_ptr + slot)
    rebase = tl.minimum(base, num_ctx // page)
    tl.store(out_ptr + r, tl.maximum(num_ctx - base * page, 0))
    for start in range(0, width, BLOCK):
        j = start + tl.arange(0, BLOCK)
        m = j < width
        src = tl.minimum(j + rebase, width - 1)
        v = tl.load(tables_ptr + row * tables_stride + src, mask=m)
        tl.store(read_ptr + row * read_stride + j, v, mask=m)


@triton.jit
def _advance_only_kernel(
    slots_ptr,  # [n] int64
    rows_ptr,  # [n] int64
    nacc_ptr,  # [n] int
    ctx_len_ptr,  # [slots] int64, in / out
    counts_ptr,  # [rows] int64
    base_ptr,  # [slots] int64
    out_ptr,  # [n] int64 advertised context lengths
    rebase_ptr,  # [n] int64: each request's read-view rebase, for _read_view_kernel
    max_ctx,
    dummy_slot,
    page,
    block_size,
):
    r = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + r)
    row = tl.load(rows_ptr + r)
    is_dummy = slot == dummy_slot
    old = tl.load(ctx_len_ptr + slot)
    new = tl.minimum(old + tl.load(nacc_ptr + r).to(tl.int64), max_ctx)
    if not is_dummy:
        tl.store(ctx_len_ptr + slot, new)
    if r == 0:
        tl.store(ctx_len_ptr + dummy_slot, tl.zeros((), dtype=tl.int64))
    new = tl.where(is_dummy, 0, new)
    allocated = tl.maximum(tl.load(counts_ptr + row) * page - block_size, 0)
    num_ctx = tl.minimum(new, allocated)
    base = tl.load(base_ptr + slot)
    tl.store(rebase_ptr + r, tl.minimum(base, num_ctx // page))
    tl.store(out_ptr + r, tl.maximum(num_ctx - base * page, 0))


@triton.jit
def _read_view_kernel(
    rows_ptr,  # [n] int64
    rebase_ptr,  # [n] int64
    tables_ptr,  # int32 [rows, width]
    read_ptr,  # int32 [rows, width]
    width,
    tables_stride,
    read_stride,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    row = tl.load(rows_ptr + r)
    rebase = tl.load(rebase_ptr + r)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = j < width
    src = tl.minimum(j + rebase, width - 1)
    v = tl.load(tables_ptr + row * tables_stride + src, mask=m)
    tl.store(read_ptr + row * read_stride + j, v, mask=m)


def advance_ctx_and_read_view(
    slots: torch.Tensor,
    rows: torch.Tensor,
    num_accepted: torch.Tensor,
    ctx_len: torch.Tensor,
    max_ctx: int,
    dummy_slot: int,
    block_counts: torch.Tensor,
    base_pages: torch.Tensor,
    tables: torch.Tensor,
    tables_read: torch.Tensor,
    page_size: int,
    block_size: int,
    parallel: bool = False,
):
    """Advance the generation requests' context lengths and build their read view in one launch: returns the
    advertised context lengths [n] (int64) and writes ``tables_read[rows]``; None (nothing written) for a layout the
    kernel does not reproduce, so the caller keeps the stock path."""
    n = slots.shape[0]
    if not (
        slots.dim() == 1
        and rows.shape == (n,)
        and num_accepted.shape == (n,)
        and num_accepted.dtype in (torch.int32, torch.int64)
        and ctx_len.dtype == torch.int64
        and block_counts.dtype == torch.int64
        and base_pages.dtype == torch.int64
        and tables.dtype == torch.int32
        and tables_read.dtype == torch.int32
        and tables.shape == tables_read.shape
        and tables.stride(1) == 1
        and tables_read.stride(1) == 1
        and page_size >= 1
    ):
        return None
    out = torch.empty(n, dtype=torch.int64, device=ctx_len.device)
    if n == 0:
        ctx_len[dummy_slot] = 0
        return out
    op = None if parallel else _glue_cpp_op("minimax_m31_dspark_advance", ctx_len.device)
    if op is not None and not all(t.is_contiguous() for t in (ctx_len, block_counts, base_pages)):
        op = _glue_log(
            "minimax_m31_dspark_advance", "non-contiguous ctx_len / block_counts / base_pages"
        )
    if op is not None:
        _glue_log("minimax_m31_dspark_advance")
        op(
            slots.to(torch.int64).contiguous(),
            rows.to(torch.int64).contiguous(),
            num_accepted.contiguous(),
            ctx_len,
            block_counts,
            base_pages,
            tables,
            tables_read,
            out,
            int(max_ctx),
            int(dummy_slot),
            int(page_size),
            int(block_size),
        )
        return out
    width = tables.shape[1]
    slots = slots.to(torch.int64).contiguous()
    rows = rows.to(torch.int64).contiguous()
    if parallel:
        global _adv_par_logged
        if not _adv_par_logged:
            _adv_par_logged = True
            from tensorrt_llm.logger import logger

            logger.info(
                f"[M3 dispatch] ADV_PAR taken: ctx advance + parallel read-view copy (width {width}, "
                f"{triton.cdiv(width, _COPY_BLOCK)} programs per request)"
            )
        rebase = torch.empty(n, dtype=torch.int64, device=ctx_len.device)
        _advance_only_kernel[(n,)](
            slots,
            rows,
            num_accepted.contiguous(),
            ctx_len,
            block_counts,
            base_pages,
            out,
            rebase,
            int(max_ctx),
            int(dummy_slot),
            int(page_size),
            int(block_size),
            num_warps=1,
        )
        _read_view_kernel[(n, triton.cdiv(width, _COPY_BLOCK))](
            rows,
            rebase,
            tables,
            tables_read,
            width,
            tables.stride(0),
            tables_read.stride(0),
            BLOCK=_COPY_BLOCK,
            num_warps=4,
        )
        return out
    _advance_kernel[(n,)](
        slots,
        rows,
        num_accepted.contiguous(),
        ctx_len,
        block_counts,
        base_pages,
        tables,
        tables_read,
        out,
        width,
        tables.stride(0),
        tables_read.stride(0),
        int(max_ctx),
        int(dummy_slot),
        int(page_size),
        int(block_size),
        BLOCK=min(1024, triton.next_power_of_2(width)),
        num_warps=4,
    )
    return out
