# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax MSA-v1 Q8KV4 decode indexer over an NVFP4 index-K mirror (opt-in).

With ``TRTLLM_M3_MSA_V1_INDEXER=1`` (and ``indexer_kv_dtype: fp8``), every sparse layer's
INDEX_KEY page holds two copies of index-K for its 128 tokens:

* ``[0, 16384)``: the E4M3 index-K ``[128, 128]`` that the fused QKV/index producer, the
  cache writers and the prefill proxy scorer use unchanged;
* ``[16384, 24576)``: packed E2M1 index-K ``[128, 64]`` (two elements per byte, low nibble
  first) and ``[24576, 25600)``: its E4M3 block scales ``[128, 8]``, one per 16 elements,
  token-major, with a unit per-tensor scale. This is the layout
  ``inference.msa_v1.indexer.decode.q8kv4`` reads.

``quantize_index_k_nvfp4`` refreshes the mirror for a step's cache slots from their E4M3
rows; ``try_select`` then scores and selects a pure decode/verify step with the MSA kernel
and returns the table ``MsaIndexer.select_blocks`` would: ``[total_q, num_kv_heads, 16]``
int32 on a head-major backing, ascending with ``-1`` padding (so the local page is the last
valid entry). Calls outside its contract return None and the CuTe DSL scorer runs instead.

The MSA plan is built once per (batch, query length, metadata buffers) outside capture and
cached in the attention metadata; its length-dependent schedule is refreshed by one
``update()`` per forward, issued by the first sparse layer that used the plan, so CUDA graphs
replay it.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

ENV_VAR = "TRTLLM_M3_MSA_V1_INDEXER"
PLAN_CACHE_KEY = "minimax_m3_msa_v1_indexer"

PAGE = 128
HEAD_DIM = 128
TOPK = 16
SF_VEC = 16
FP8_PAGE_BYTES = PAGE * HEAD_DIM
NVFP4_DATA_PAGE_BYTES = PAGE * HEAD_DIM // 2
NVFP4_SCALE_PAGE_BYTES = PAGE * HEAD_DIM // SF_VEC
# Bytes per token of an INDEX_KEY page carrying both the E4M3 index-K and its NVFP4 mirror.
MIRRORED_BYTES_PER_TOKEN = (FP8_PAGE_BYTES + NVFP4_DATA_PAGE_BYTES + NVFP4_SCALE_PAGE_BYTES) // PAGE
_INDEX_HEADS = (1, 2, 4)
_MAX_QUERY_LEN = 16
_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))

_logged: set = set()


def _log_once(key: str, message: str) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.info(f"MiniMax-M3 MSA-v1 indexer: {message}")


def env_enabled() -> bool:
    return os.environ.get(ENV_VAR, "0").strip().lower() in ("1", "true", "on", "yes")


@functools.lru_cache(maxsize=1)
def _msa_classes():
    """(wrapper class, shared plan class), or None when the path is off or MSA is missing."""
    if not env_enabled():
        return None
    try:
        from inference.msa_v1.indexer.decode import BatchDecodeIndexerPlan
        from inference.msa_v1.indexer.decode.q8kv4 import (
            BatchDecodeIndexerWithPagedKVCacheWrapper,
        )
    except ImportError as err:
        logger.warning(f"{ENV_VAR}=1 ignored: MiniMax MSA is not importable ({err}).")
        return None
    return BatchDecodeIndexerWithPagedKVCacheWrapper, BatchDecodeIndexerPlan


def is_enabled() -> bool:
    """Whether the opt-in is set and the MSA q8kv4 decode indexer is importable."""
    return _msa_classes() is not None


def split_mirrored_index_k(raw: torch.Tensor):
    """Split a mirrored INDEX_KEY view into its three regions.

    Args:
        raw: [num_pages, 1, 128, MIRRORED_BYTES_PER_TOKEN] uint8 view of the INDEX_KEY pool
            (any page stride, each page contiguous).

    Returns:
        (fp8 [num_pages, 1, 128, 128] float8_e4m3fn, data [num_pages, 128, 64] uint8,
        scale [num_pages, 128, 8] float8_e4m3fn), all sharing the pool's page stride.
    """
    if raw.dim() != 4 or tuple(raw.shape[1:]) != (1, PAGE, MIRRORED_BYTES_PER_TOKEN):
        raise ValueError(f"unexpected mirrored index-K view shape {tuple(raw.shape)}")
    if raw.dtype != torch.uint8 or raw.stride(3) != 1 or raw.stride(2) != MIRRORED_BYTES_PER_TOKEN:
        raise ValueError("mirrored index-K view must be uint8 with contiguous pages")
    num_pages, page_stride, base = int(raw.shape[0]), raw.stride(0), raw.storage_offset()
    fp8 = torch.as_strided(
        raw, (num_pages, 1, PAGE, HEAD_DIM), (page_stride, FP8_PAGE_BYTES, HEAD_DIM, 1), base
    ).view(torch.float8_e4m3fn)
    data = torch.as_strided(
        raw,
        (num_pages, PAGE, HEAD_DIM // 2),
        (page_stride, HEAD_DIM // 2, 1),
        base + FP8_PAGE_BYTES,
    )
    scale = torch.as_strided(
        raw,
        (num_pages, PAGE, HEAD_DIM // SF_VEC),
        (page_stride, HEAD_DIM // SF_VEC, 1),
        base + FP8_PAGE_BYTES + NVFP4_DATA_PAGE_BYTES,
    ).view(torch.float8_e4m3fn)
    return fp8, data, scale


@triton.jit
def _e2m1_code(a):
    """|value| (already divided by its block scale) -> 3-bit E2M1 magnitude, round-to-nearest-even."""
    code = (a > 0.25).to(tl.int32)
    code += (a >= 0.75).to(tl.int32)
    code += (a > 1.25).to(tl.int32)
    code += (a >= 1.75).to(tl.int32)
    code += (a > 2.5).to(tl.int32)
    code += (a >= 3.5).to(tl.int32)
    code += (a > 5.0).to(tl.int32)
    return code


@triton.jit
def _quantize_index_k_nvfp4_kernel(
    fp8_ptr,
    data_ptr,
    scale_ptr,
    loc_ptr,
    fp8_page_stride,
    data_page_stride,
    scale_page_stride,
    PAGE_SIZE: tl.constexpr,
    D: tl.constexpr,
    VEC: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(loc_ptr + t).to(tl.int64)
    if slot < 0:
        return
    page = slot // PAGE_SIZE
    within = slot % PAGE_SIZE
    groups: tl.constexpr = D // VEC
    half: tl.constexpr = VEC // 2
    g = tl.arange(0, groups)[:, None]
    p = tl.arange(0, half)[None, :]
    src = fp8_ptr + page * fp8_page_stride + within * D + g * VEC + 2 * p
    lo = tl.load(src).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    hi = tl.load(src + 1).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(lo), axis=1), tl.max(tl.abs(hi), axis=1))
    sf = (amax / 6.0).to(tl.float8e4nv)
    sf_f = sf.to(tl.float32)
    inv = tl.where(sf_f > 0, 1.0 / sf_f, 0.0)[:, None]
    lo_s = lo * inv
    hi_s = hi * inv
    lo_c = _e2m1_code(tl.abs(lo_s)) | tl.where(lo_s < 0, 8, 0)
    hi_c = _e2m1_code(tl.abs(hi_s)) | tl.where(hi_s < 0, 8, 0)
    packed = (lo_c | (hi_c << 4)).to(tl.uint8)
    tl.store(data_ptr + page * data_page_stride + within * (D // 2) + g * half + p, packed)
    s = tl.arange(0, groups)
    tl.store(
        scale_ptr + page * scale_page_stride + within * groups + s,
        sf.to(tl.uint8, bitcast=True),
    )


def quantize_index_k_nvfp4(
    fp8_cache: torch.Tensor,
    data_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
) -> None:
    """Refresh the NVFP4 mirror of the given cache slots from their E4M3 index-K rows.

    Args:
        fp8_cache: [num_pages, 1, 128, 128] float8_e4m3fn (page-strided).
        data_cache: [num_pages, 128, 64] uint8 packed E2M1.
        scale_cache: [num_pages, 128, 8] float8_e4m3fn block scales.
        out_cache_loc: [num_tokens] int32 slot ids (page * 128 + token); negative entries skip.
    """
    num_tokens = int(out_cache_loc.shape[0])
    if num_tokens == 0:
        return
    _quantize_index_k_nvfp4_kernel[(num_tokens,)](
        fp8_cache.view(torch.uint8),
        data_cache,
        scale_cache.view(torch.uint8),
        out_cache_loc,
        fp8_cache.stride(0),
        data_cache.stride(0),
        scale_cache.stride(0),
        PAGE_SIZE=PAGE,
        D=HEAD_DIM,
        VEC=SF_VEC,
        num_warps=1,
    )


@triton.jit
def _sort_topk_rows_kernel(table_ptr, num_rows, K: tl.constexpr, ROWS: tl.constexpr):
    """Sort each K-wide row ascending in place with -1 entries moved to the end."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    k = tl.arange(0, K)[None, :]
    mask = r < num_rows
    x = tl.load(table_ptr + r * K + k, mask=mask, other=-1)
    big = 2147483647
    key = tl.where(x < 0, big, x)
    key = tl.sort(key, dim=1)
    tl.store(table_ptr + r * K + k, tl.where(key == big, -1, key), mask=mask)


@dataclass
class _Plan:
    wrapper: object
    shared_plan: object
    out: torch.Tensor  # [num_index_heads, total_q, 16] int32
    first_layer: int


def _get_or_plan(cache, *, block_table, seq_lens, query_len, num_heads, layer_idx):
    key = (
        "plan",
        int(block_table.shape[0]),
        int(block_table.shape[1]),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        int(query_len),
        int(num_heads),
    )
    plan = cache.get(key)
    if plan is not None:
        return plan
    if torch.cuda.is_current_stream_capturing():
        _log_once(
            f"capture{key[1]}x{query_len}",
            f"no plan for batch {key[1]} x {query_len} queries before capture; this graph "
            "keeps the CuTe DSL scorer.",
        )
        return None
    wrapper_cls, plan_cls = _msa_classes()
    shared_plan = plan_cls(
        seq_lens,
        max_pages=int(block_table.shape[1]),
        num_index_heads=int(num_heads),
        query_length=int(query_len),
    )
    wrapper = wrapper_cls(
        use_cuda_graph=True, page_table_buffer=block_table, seq_lens_buffer=seq_lens
    )
    wrapper.plan(
        block_table,
        seq_lens,
        num_index_heads=int(num_heads),
        query_length=int(query_len),
        shared_plan=shared_plan,
    )
    out = torch.empty(
        (int(num_heads), int(block_table.shape[0]) * int(query_len), TOPK),
        dtype=torch.int32,
        device=block_table.device,
    )
    plan = _Plan(wrapper=wrapper, shared_plan=shared_plan, out=out, first_layer=int(layer_idx))
    cache[key] = plan
    _log_once("planned", f"Q8KV4 decode indexer on ({num_heads} index head(s)/rank).")
    return plan


def try_select(
    idx_q: torch.Tensor,
    data_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    num_kv_heads: int,
    init_blocks: int,
    local_blocks: int,
    layer_idx: int,
    plan_cache: dict,
) -> Optional[torch.Tensor]:
    """Select a pure decode/verify step's blocks with the MSA q8kv4 indexer, or return None.

    Args:
        idx_q: [total_q, num_index_heads, 128] float8_e4m3fn index queries, request-major.
        data_cache / scale_cache: the NVFP4 mirror from split_mirrored_index_k.
        block_table: [batch, max_pages] int32; seq_lens: [batch] int32 including this step.
        num_kv_heads: local KV heads; each must own exactly one index head.

    Returns:
        [total_q, num_kv_heads, 16] int32 on a head-major backing, ascending with -1 padding,
        or None when the call is outside the kernel's contract.
    """
    if _msa_classes() is None:
        return None
    total_q, num_heads, head_dim = idx_q.shape
    batch = int(block_table.shape[0])
    if (
        init_blocks != 0
        or local_blocks != 1
        or num_heads != num_kv_heads
        or num_heads not in _INDEX_HEADS
        or head_dim != HEAD_DIM
        or idx_q.dtype != torch.float8_e4m3fn
        or not 1 <= decode_query_len <= _MAX_QUERY_LEN
        or total_q != batch * decode_query_len
        or not (block_table.is_contiguous() and seq_lens.is_contiguous())
        or block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or torch.cuda.get_device_capability(idx_q.device) not in _SUPPORTED_CAPABILITIES
    ):
        return None
    plan = _get_or_plan(
        plan_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        query_len=decode_query_len,
        num_heads=num_heads,
        layer_idx=layer_idx,
    )
    if plan is None:
        return None
    if layer_idx == plan.first_layer:
        plan.shared_plan.update()
    q = idx_q if idx_q.is_contiguous() else idx_q.contiguous()
    plan.wrapper.run(
        q.view(batch, decode_query_len, num_heads, HEAD_DIM),
        data_cache,
        k_scale=scale_cache,
        out=plan.out,
    )
    rows = num_heads * total_q
    block_rows = 16
    _sort_topk_rows_kernel[(triton.cdiv(rows, block_rows),)](
        plan.out, rows, K=TOPK, ROWS=block_rows, num_warps=1
    )
    return plan.out.permute(1, 0, 2)


__all__ = [
    "ENV_VAR",
    "MIRRORED_BYTES_PER_TOKEN",
    "PLAN_CACHE_KEY",
    "env_enabled",
    "is_enabled",
    "quantize_index_k_nvfp4",
    "split_mirrored_index_k",
    "try_select",
]
