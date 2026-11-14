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
# DSpark / DFlash drafter forward: the paged block decode's index prelude in one launch
# =============================================================================
#
# It replaces, bit for bit, the integer lines at the top of DFlashForCausalLM.dflash_forward on a paged backend:
#   cache_seqlens     = num_ctx[:B].to(int32)
#   cache_batch_idx   = ctx_cache_batch_idx.to(int32)
#   seq_lens_after    = cache_seqlens + block_size
#   kv_indptr         = arange(0, (B + 1) * pages_per_slot, pages_per_slot)          (int32)
#   kv_last_page_len  = ((seq_lens_after - 1) % page_size) + 1                       (floored modulo, as torch)
#   append_positions  = (cache_seqlens.view(-1, 1) + arange(block_size)).reshape(-1) (int32)
#   query_pos_i32     = query_positions.reshape(-1).to(int32)
# The int64 -> int32 casts keep the low 32 bits, as torch's do. One CTA writes every output.

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_switches import (
    m31_disabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

# DSPARK_GLUE_CPP (no Triton in served paths): the C++ op family_ops().minimax_m31_dspark_indices
# (kernels/minimaxM31/dsparkGlueSm107.cu; an arch-generic single-CTA kernel, built for every target arch) computes the
# prelude, bit for bit the Triton _indices_kernel it replaced. The Triton fallback is deleted: non-contiguous
# num_ctx / batch_idx go through .contiguous() (query_positions may have any strides, the op reads them). Without the
# op, or with TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1 (read once, at import), forward_indices returns None and the caller
# keeps the stock lines.
# Note (latent bug, now gone): the deleted Triton fallback took num_ctx / batch_idx as contiguous and ignored their
# strides, so a strided num_ctx / batch_idx that reached it (the C++ op's contiguity guard sent those there) read the
# wrong elements. The .contiguous() route reads them by value.
_GLUE_CPP_OFF = m31_disabled("DSPARK_GLUE_CPP")
_OP = "minimax_m31_dspark_indices"
_glue_logged = set()


def _glue_log(name: str, why: Optional[str] = None):
    """Engagement line, once per (op, outcome) per process: "[M3 dispatch] DSPARK_GLUE_CPP taken: <op>" when the C++ op
    runs, "... not taken: <op>: <why>; stock lines" when the caller keeps its stock lines. Returns None."""
    if (name, why) not in _glue_logged:
        _glue_logged.add((name, why))
        from tensorrt_llm.logger import logger

        if why is None:
            logger.info(f"[M3 dispatch] DSPARK_GLUE_CPP taken: {name}")
        else:
            logger.info(f"[M3 dispatch] DSPARK_GLUE_CPP not taken: {name}: {why}; stock lines")
    return None


def _indices_op():
    """The C++ op, or None with the reason logged once (the caller keeps the stock lines)."""
    if _GLUE_CPP_OFF:
        return _glue_log(_OP, "killed (TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1)")
    op = getattr(family_ops(), _OP, None)
    return op if op is not None else _glue_log(_OP, "the op is not registered")


def forward_indices(
    num_ctx: torch.Tensor,
    batch_idx: torch.Tensor,
    query_positions: torch.Tensor,
    block_size: int,
    pages_per_slot: int,
    page_size: int,
):
    """The paged dflash_forward's index prelude in one launch: (cache_seqlens, cache_batch_idx, seq_lens_after,
    kv_indptr, kv_last_page_len, append_positions, query_positions_flat), all int32 and bitwise the stock lines; or
    None (nothing computed) for a layout the kernel does not reproduce, so the caller keeps the stock lines."""
    ok_dtypes = (torch.int32, torch.int64)
    B = num_ctx.shape[0] if num_ctx.dim() == 1 else -1
    if not (
        B >= 1
        and num_ctx.dtype in ok_dtypes
        and batch_idx.dtype in ok_dtypes
        and query_positions.dtype in ok_dtypes
        and batch_idx.shape == (B,)
        and query_positions.dim() == 2
        and tuple(query_positions.shape) == (B, block_size)
        and block_size >= 1
        and pages_per_slot >= 1
        and page_size >= 1
        and num_ctx.is_cuda
    ):
        return None
    if B * block_size > 2**31 - 1:
        return None
    op = _indices_op()
    if op is None:
        return None
    dev = num_ctx.device
    i32 = dict(dtype=torch.int32, device=dev)
    seqlens = torch.empty(B, **i32)
    batch_out = torch.empty(B, **i32)
    after = torch.empty(B, **i32)
    indptr = torch.empty(B + 1, **i32)
    last = torch.empty(B, **i32)
    append = torch.empty(B * block_size, **i32)
    qflat = torch.empty(B * block_size, **i32)
    _glue_log(_OP)
    op(num_ctx.contiguous(), batch_idx.contiguous(), query_positions, seqlens, batch_out, after, indptr, last, append,
       qflat, block_size, pages_per_slot, page_size)  # fmt: skip
    return seqlens, batch_out, after, indptr, last, append, qflat
