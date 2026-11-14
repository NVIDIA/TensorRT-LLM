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
# ADV_PAR (``parallel=True``, chosen by the caller): at batch 1 the one CTA copies the whole read-view row (width = the
# pool's max blocks per sequence) alone. With it, an advance-only launch (one thread per request) stores each rebase,
# and a second launch copies the row with one CTA per 512 columns. The same values are stored (a pure index copy), so
# bit for bit the one-launch path; one launch cannot split the copy, since its CTAs would race the context-length store.
#
# Both layouts are CUDA C++ ops (kernels/minimaxM31/dsparkGlueSm107.cu, no Triton): minimax_m31_dspark_advance
# (one launch) and minimax_m31_dspark_advance_par (ADV_PAR, PDL launches). Where an op is not taken (not an sm_107
# device, the op not built, TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1, non-contiguous ctx_len / block_counts / base_pages),
# the function returns None and the caller keeps the stock torch path.

import functools
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_switches import (
    m31_disabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

_adv_par_logged = False
_PAR_COLS = 512  # minimax_m31_dspark_advance_par: columns per copy CTA (128 threads x 4)

# DSPARK_GLUE_CPP: the C++ ops (family_ops().minimax_m31_dspark_advance / _advance_par) on SM107 when they are
# built; None (the caller's stock path) otherwise or with TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1 (read once, at import).
_GLUE_CPP_OFF = m31_disabled("DSPARK_GLUE_CPP")


@functools.lru_cache(maxsize=None)
def _is_sm107(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 7)


def _glue_cpp_op(name: str, device: torch.device):
    """The C++ op, or None with the reason logged once (the caller's stock path runs)."""
    if _GLUE_CPP_OFF:
        return _glue_log(name, "killed (TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1)")
    if not _is_sm107(device.index if device.index is not None else torch.cuda.current_device()):
        return _glue_log(name, "not an sm_107 device")
    op = getattr(family_ops(), name, None)
    return op if op is not None else _glue_log(name, "the op is not registered")


_glue_logged = set()


def _glue_log(name: str, why: Optional[str] = None):
    """Engagement line, once per (op, outcome) per process: "[M3 dispatch] DSPARK_GLUE_CPP taken: <op>" when the C++ op
    runs, "... not taken: <op>: <why>; stock path" when it does not. Returns None."""
    if (name, why) not in _glue_logged:
        _glue_logged.add((name, why))
        from tensorrt_llm.logger import logger

        if why is None:
            logger.info(f"[M3 dispatch] DSPARK_GLUE_CPP taken: {name}")
        else:
            logger.info(f"[M3 dispatch] DSPARK_GLUE_CPP not taken: {name}: {why}; stock path")
    return None


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
    """Advance the generation requests' context lengths and build their read view (one launch, or two with
    ``parallel``): returns the advertised context lengths [n] (int64) and writes ``tables_read[rows]``; None (nothing
    written) for a layout the kernels do not reproduce or where the C++ op is not taken, so the caller keeps the stock
    path."""
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
    name = "minimax_m31_dspark_advance_par" if parallel else "minimax_m31_dspark_advance"
    op = _glue_cpp_op(name, ctx_len.device)
    if op is not None and not all(t.is_contiguous() for t in (ctx_len, block_counts, base_pages)):
        op = _glue_log(name, "non-contiguous ctx_len / block_counts / base_pages")
    if op is None:
        return None
    out = torch.empty(n, dtype=torch.int64, device=ctx_len.device)
    if n == 0:
        ctx_len[dummy_slot] = 0
        return out
    _glue_log(name)
    args = (
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
    if not parallel:
        op(*args)
        return out
    global _adv_par_logged
    if not _adv_par_logged:
        _adv_par_logged = True
        from tensorrt_llm.logger import logger

        width = tables.shape[1]
        logger.info(
            f"[M3 dispatch] ADV_PAR taken: ctx advance + parallel read-view copy (width {width}, "
            f"{-(-width // _PAR_COLS)} CTAs per request)"
        )
    op(*args, True)
    return out
