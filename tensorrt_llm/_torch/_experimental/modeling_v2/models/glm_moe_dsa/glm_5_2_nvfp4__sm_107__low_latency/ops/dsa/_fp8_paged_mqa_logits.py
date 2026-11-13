# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile cache and launch of the indexer's FP8 paged MQA logits kernel (FP32 epilogue).

The kernel (``_fp8_paged_mqa_logits_kernel.FP8MQALogitsKernel``) scores every query row of a
request against the request's paged indexer keys: ``logits[b * next_n + t, j] = scale[j] *
sum_h relu(q[b, t, h] . k[j]) * weights[b * next_n + t, h]`` for the keys ``j`` of the request's
context. Two policies decide the compiled variant:

- full query: a request's ``next_n`` query rows stay one atom of ``next_n * num_heads`` MMA
  columns (``full_query_weight_cache_limit``); the weight register cache is capped so that the
  shared-memory tail handles the remaining heads without changing the FMA order;
- early KV (``early_kv``, shapes of ``early_kv_supported``): the N-split variant launched with
  PDL, which loads the older KV blocks before ``griddepcontrol.wait``.

All variants compute the same per-element arithmetic in the same order.
"""

from __future__ import annotations

import functools
from typing import Dict, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch

from ._fp8_paged_mqa_logits_kernel import FP8MQALogitsKernel

#: Keys per MMA tile; a schedule split is one tile per math warp group.
COMPUTE_BLOCK_KV = 128
#: Keys per schedule split (DeepGEMM's SPLIT_KV of the SM100 paged metadata).
SPLIT_KV = 2 * COMPUTE_BLOCK_KV

_KERNELS: Dict[Tuple, object] = {}


@functools.lru_cache(maxsize=None)
def num_sms(device_index: int) -> int:
    """SMs of the device, the number of persistent CTAs the kernel launches."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def full_query_weight_cache_limit(
    next_n: int, num_heads: int, head_dim: int, phys_block_kv: int, q_dtype: torch.dtype
) -> Optional[int]:
    """The weight register cache of the full-query kernel, or None for the default kernel.

    The two MMA groups share 512 TMEM columns. Each needs ``next_n * num_heads`` columns, with an
    eight-column MMA granularity. D = 128 and three Q/KV stages fit the shared-memory budget
    throughout this range. Physical pages must divide the 128-token compute tile into at most four
    pieces. At most 96 FP32 weights per thread (16 for each query row) stay in registers; the
    shared-memory tail handles the remaining heads without changing the FMA order. Up to four
    query rows keep the default kernel.
    """
    columns = next_n * num_heads
    if (
        next_n <= 4
        or head_dim != 128
        or not 8 <= columns <= 256
        or columns % 8
        or phys_block_kv not in (32, 64, 128)
        or num_heads % 4
        or q_dtype != torch.float8_e4m3fn
    ):
        return None
    limit = min(16, num_heads, (96 // next_n // 4) * 4)
    return limit if limit >= 4 else None


def early_kv_supported(next_n: int, num_heads: int) -> bool:
    """Shapes of the early-KV kernel (PDL launch, N split): each group's accumulator splits into
    two N halves of next_n / 2 query rows, and a half must be a whole number of 16-column UMMA
    atoms."""
    return next_n % 2 == 0 and (next_n * num_heads // 2) % 16 == 0


def _compile(
    phys_block_kv: int,
    num_heads: int,
    head_dim: int,
    next_n: int,
    sms: int,
    weight_cache_limit: Optional[int],
    early_kv: bool,
):
    """Compile one variant against fake tensors with symbolic runtime extents (TVM FFI)."""
    n = next_n * num_heads
    block_bytes = phys_block_kv * (head_dim + 4)
    sym_num_phys_blocks = cute.sym_int()
    sym_b = cute.sym_int()
    max_ctx = cute.sym_int()
    max_blocks_per_seq = cute.sym_int()
    num_ctas = cute.sym_int()

    # The indexer K-cache pool view can be strided in dim 0 (pages of other layers in between):
    # the per-page stride is read at runtime; the bytes of one page are contiguous.
    kv_fake = cute.runtime.make_fake_tensor(
        cutlass.Uint8, (sym_num_phys_blocks, block_bytes), stride=(cute.sym_int64(), 1)
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (n, head_dim, sym_b), stride_order=(1, 0, 2)
    )
    w_fake = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (n, sym_b), stride_order=(0, 1))
    logits_fake = cute.runtime.make_fake_tensor(
        cutlass.Float32, (cute.sym_int(), max_ctx), stride=(cute.sym_int64(), 1)
    )
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_b, max_blocks_per_seq), stride_order=(1, 0)
    )
    cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_b,), stride_order=(0,))
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_ctas, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = FP8MQALogitsKernel(
        block_kv=COMPUTE_BLOCK_KV,
        phys_block_kv=phys_block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        next_n=next_n,
        num_sms=sms,
        num_epi_subtiles=1,
        epi_dtype=cutlass.Float32,
        acc_dtype=cutlass.Float32,
        output_dtype=cutlass.Float32,
        weight_cache_limit=weight_cache_limit,
        split_n=early_kv,
        early_kv=early_kv,
    )
    return cute.compile(
        kernel,
        kv_fake,
        q_fake,
        w_fake,
        logits_fake,
        bt_fake,
        cl_fake,
        sm_fake,
        cutlass.Int32(1),
        cutlass.Int32(1),
        fake_stream,
        options="--enable-tvm-ffi",
    )


def run_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    logits: torch.Tensor,
    max_context_len: int,
    early_kv: bool,
) -> torch.Tensor:
    """Launch the kernel; returns ``logits[:, :max_context_len]``.

    Args:
        q: [B, next_n, H, D] float8_e4m3fn.
        kv_cache: [num_pages, page_size, 1, D + 4] uint8 indexer K-cache view.
        weights: [B * next_n, H] float32.
        context_lens: [B] int32 keys per request.
        block_table: [B, max_pages] int32, contiguous.
        schedule_meta: [num_sms + 1, 2] int32 (the SM100 paged MQA-logits schedule).
        logits: [B * next_n, >= round_up(max_context_len, SPLIT_KV)] float32 output, unit
            column stride; the kernel writes whole splits.
        max_context_len: columns of the returned view.
        early_kv: run the early-KV variant where its shape allows (``early_kv_supported``).
    """
    b, next_n, h, d = q.shape
    n = next_n * h
    phys_block_kv = kv_cache.shape[1]
    num_phys_blocks = kv_cache.shape[0]
    sms = num_sms(q.device.index if q.device.index is not None else torch.cuda.current_device())
    weight_cache_limit = full_query_weight_cache_limit(next_n, h, d, phys_block_kv, q.dtype)
    early_kv = early_kv and early_kv_supported(next_n, h)
    key = (phys_block_kv, h, d, next_n, sms, weight_cache_limit, early_kv)
    compiled = _KERNELS.get(key)
    if compiled is None:
        compiled = _KERNELS[key] = _compile(
            phys_block_kv, h, d, next_n, sms, weight_cache_limit, early_kv
        )
    # Q [B, next_n, H, D] -> [N, D, B]; weights [B * next_n, H] -> [N, B]; FP8 Q as uint8 bytes.
    q_3d = q.reshape(b, n, d).permute(1, 2, 0).view(torch.uint8)
    w_2d = weights.reshape(b, n).t()
    kv_flat = kv_cache.reshape(num_phys_blocks, -1)
    out = logits[:, :max_context_len]
    compiled(kv_flat, q_3d, w_2d, out, block_table, context_lens, schedule_meta, num_phys_blocks, b)
    return out
