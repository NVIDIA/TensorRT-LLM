# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapters from TRT-LLM's paged MSA cache to MiniMax MSA's native prefill wrappers.

The native MiniMax MSA ``inference.msa_v1`` package is an optional external
dependency that the user provisions separately (it is not the fmha_sm100
package bundled with TensorRT-LLM). Both wrappers take a per-request page
table, cumulative query and KV lengths, and the per-query TopK selection, and
are planned once per layer call so each plan owns the selection it was built
from.
"""

from __future__ import annotations

import importlib
import math
import types

import torch

_FP8_PREFILL_MODULE = "inference.msa_v1.attention.prefill.q8kv8"
_NVFP4_PREFILL_MODULE = "inference.msa_v1.attention.prefill.q8kv4"


def _import_native_prefill(module_name: str) -> types.ModuleType:
    """Import a native MSA prefill module, or explain how to provision it.

    Imported on first use only: the package is optional, is installed outside
    TensorRT-LLM, and compiles its kernels on first use, so a module-scope
    import would make every attention import depend on it.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Native MiniMax MSA prefill requires {module_name} from the MiniMax "
            "MSA repository on PYTHONPATH; see examples/minimax_m3/rubin/setup_msa.sh."
        ) from exc


def virtual_contiguous_pages(cache: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Expose an interleaved HND cache plane as contiguous virtual pages.

    TRT-LLM pools interleave each page's K and V planes (and possibly other
    layers), so one plane's page stride is a whole multiple of its payload. The
    returned view steps one payload per page, which makes physical page ``p``
    virtual page ``p * multiplier``; the gaps become unreferenced virtual pages.
    No data moves, and logical TopK indices and lengths are unchanged.

    Args:
        cache: A ``[pages, heads, tokens, head_dim]`` plane whose pages are
            each contiguous.

    Returns:
        The zero-copy contiguous view and the physical-to-virtual page multiplier.
    """
    if cache.ndim != 4 or cache.shape[0] <= 0:
        raise ValueError("MSA cache must be [pages, heads, tokens, head_dim] with pages")
    payload = math.prod(cache.shape[1:])
    if payload <= 0 or not cache[0].is_contiguous():
        raise ValueError("MSA virtual pages require contiguous inner HND pages")
    page_stride = cache.stride(0)
    if page_stride < payload or page_stride % payload:
        raise ValueError("MSA page stride must be a positive whole number of page payloads")
    multiplier = page_stride // payload
    virtual_count = (cache.shape[0] - 1) * multiplier + 1
    # as_strided keeps this plane's storage offset (K and V start at different
    # planes). The last virtual page is the last physical page, so the view
    # ends where the original plane does.
    view = cache.as_strided((virtual_count, *cache.shape[1:]), (payload, *cache.stride()[1:]))
    return view, multiplier


def map_cache_pages(
    page_table: torch.Tensor, *caches: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Map one physical page table onto the virtual pages of matching cache planes.

    Args:
        page_table: ``[batch, max_pages]`` int32 physical page ids; negative
            ids are left as they are.
        caches: Cache planes that the page table addresses together.

    Returns:
        The page table in virtual page ids and each plane's virtual-page view.
    """
    if not caches:
        raise ValueError("At least one cache plane is required")
    if any(cache.shape[0] != caches[0].shape[0] for cache in caches):
        raise ValueError("MSA cache planes must share their page capacity")
    views_and_factors = tuple(virtual_contiguous_pages(cache) for cache in caches)
    factor = views_and_factors[0][1]
    if any(value != factor for _, value in views_and_factors):
        raise ValueError("MSA cache planes must share the virtual-page multiplier")
    mapped = page_table
    if factor != 1:
        mapped = torch.where(page_table >= 0, page_table * factor, page_table)
    return mapped, tuple(view for view, _ in views_and_factors)


def run_fp8_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    *,
    total_k: int,
    total_rows: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    out: torch.Tensor,
) -> None:
    """Run native Q8KV8 sparse prefill over an FP8 cache into ``out``.

    Args:
        q: ``[tokens, q_heads, 128]`` E4M3 queries.
        k_cache: ``[pages, kv_heads, 128, 128]`` E4M3 K plane.
        v_cache: The matching V plane.
        topk_indices: ``[kv_heads, tokens, topk]`` logical page selection.
        cu_seqlens_q: ``[batch + 1]`` cumulative new-token counts.
        cu_seqlens_k: ``[batch + 1]`` cumulative attended KV lengths.
        page_table: ``[batch, max_pages]`` int32 physical page ids.
        total_k: Upper bound on the summed KV length.
        total_rows: Upper bound on the summed KV page count.
        max_seqlen_q: Upper bound on one request's new-token count.
        max_seqlen_k: Upper bound on one request's KV length.
        sm_scale: Softmax scale.
        out: ``[tokens, q_heads, 128]`` BF16 destination.
    """
    module = _import_native_prefill(_FP8_PREFILL_MODULE)
    mapped_pages, (k_pages, v_pages) = map_cache_pages(page_table, k_cache, v_cache)
    wrapper = module.BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        topk_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        mapped_pages,
        num_q_heads=int(q.shape[1]),
        num_kv_heads=int(k_cache.shape[1]),
        total_k=total_k,
        total_rows=total_rows,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sm_scale=sm_scale,
    )
    wrapper.run(q, (k_pages, v_pages), out=out)


def run_nvfp4_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_block_scale: torch.Tensor,
    v_block_scale: torch.Tensor,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    topk_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    *,
    total_k: int,
    total_rows: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
    out: torch.Tensor,
) -> None:
    """Run native Q8KV4 sparse prefill over TRT-LLM's NVFP4 cache into ``out``.

    TRT-LLM stores the V block scales in four-token groups and dequantizes
    with per-layer global scales, so this passes ``swizzle4x4`` and the live
    global scales. The packed data and scale pools can have different page
    pitches, so each plane keeps its own zero-copy view and page stride under
    the one physical page table.

    Args:
        q: ``[tokens, q_heads, 128]`` queries; converted to E4M3.
        k_cache: ``[pages, kv_heads, 128, 64]`` packed E2M1 K plane.
        v_cache: The matching V plane.
        k_block_scale: ``[pages, kv_heads, 128, 8]`` E4M3 K block scales.
        v_block_scale: The matching V block scales, in four-token groups.
        k_global_scale: One 16-byte-aligned FP32 K dequantization scale.
        v_global_scale: One 16-byte-aligned FP32 V dequantization scale.
        topk_indices: ``[kv_heads, tokens, topk]`` logical page selection.
        cu_seqlens_q: ``[batch + 1]`` cumulative new-token counts.
        cu_seqlens_k: ``[batch + 1]`` cumulative attended KV lengths.
        page_table: ``[batch, max_pages]`` int32 physical page ids.
        total_k: Upper bound on the summed KV length.
        total_rows: Upper bound on the summed KV page count.
        max_seqlen_q: Upper bound on one request's new-token count.
        max_seqlen_k: Upper bound on one request's KV length.
        sm_scale: Softmax scale.
        out: ``[tokens, q_heads, 128]`` BF16 destination.
    """
    module = _import_native_prefill(_NVFP4_PREFILL_MODULE)
    wrapper = module.BatchPrefillWithPagedKVCacheWrapper()
    wrapper.plan(
        topk_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        page_table,
        total_k=total_k,
        total_rows=total_rows,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        sm_scale=sm_scale,
    )
    wrapper.run(
        q.to(torch.float8_e4m3fn),
        (k_cache.view(torch.uint8), v_cache.view(torch.uint8)),
        kv_cache_sf=(
            k_block_scale.view(torch.float8_e4m3fn),
            v_block_scale.view(torch.float8_e4m3fn),
        ),
        kv_cache_global_sf=(k_global_scale, v_global_scale),
        v_scale_layout="swizzle4x4",
        allow_strided_cache=True,
        out=out,
    )


__all__ = [
    "map_cache_pages",
    "run_fp8_prefill",
    "run_nvfp4_prefill",
    "virtual_contiguous_pages",
]
