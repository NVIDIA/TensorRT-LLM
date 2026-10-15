# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental TRTLLM-Gen consumer for MiniMax-M3 sparse NVFP4 decode.

TRTLLM-Gen does not accept a different sparse page list for every KV head.
Represent each ``(query token, KV head)`` pair as an independent one-KV-head
request instead.  A physical P128 page is exposed as four P32 pages by viewing
the existing packed cache; neither KV data nor block scales are copied.

Only the compact block table and sequence lengths are materialized.  Their
conversion is fused into one small Triton kernel so the performance experiment
measures the attention architecture rather than a chain of framework gathers.
"""

from __future__ import annotations

from typing import Protocol

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._utils import get_sm_version

from .msa_utils import check_decode_span_shape
from .trtllm_gen_dense_decode import _counter_buffer, _workspace

_M3_PAGE_SIZE = 128
_TRTLLM_GEN_PAGE_SIZE = 32
_PAGES_PER_SELECTED_BLOCK = _M3_PAGE_SIZE // _TRTLLM_GEN_PAGE_SIZE
_LOG2_E = 1.4426950408889634


class _MiniMaxM3SparseKVCacheManager(Protocol):
    def get_kv_subpage_pool(self, layer_idx: int, kv_layout: str) -> tuple[torch.Tensor, int]: ...

    def get_kv_scale_subpage_pool(
        self, layer_idx: int, kv_layout: str
    ) -> tuple[torch.Tensor, int]: ...


@triton.autotune(
    configs=[triton.Config({}, num_warps=1, num_stages=1)],
    key=["num_pseudo_requests"],
)
@triton.jit
def _build_sparse_p32_table_kernel(
    topk_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_table_ptr,
    out_seq_lens_ptr,
    num_pseudo_requests,
    num_kv_heads: tl.constexpr,
    decode_query_len,
    subpages_per_slot,
    stride_t_token,
    stride_t_head,
    stride_t_topk,
    stride_bt_req,
    stride_bt_block,
    stride_out_req,
    stride_out_role,
    stride_out_block,
    max_topk: tl.constexpr,
    output_pages: tl.constexpr,
):
    """Build one compact P32 K/V page-table row per pseudo request."""
    pseudo_req = tl.program_id(0)
    active = pseudo_req < num_pseudo_requests
    token = pseudo_req // num_kv_heads
    kv_head = pseudo_req - token * num_kv_heads
    req = token // decode_query_len
    query_in_req = token - req * decode_query_len

    final_seq_len = tl.load(seq_lens_ptr + req, mask=active, other=0)
    kv_len = tl.maximum(final_seq_len - decode_query_len + query_in_req + 1, 0)
    num_logical_blocks = (kv_len + 127) // 128
    real_topk = tl.minimum(max_topk, num_logical_blocks)

    page_offset = tl.arange(0, output_pages)
    topk_offset = page_offset // 4
    subpage = page_offset - topk_offset * 4
    valid = active & (topk_offset < real_topk)
    logical_block = tl.load(
        topk_ptr + token * stride_t_token + kv_head * stride_t_head + topk_offset * stride_t_topk,
        mask=valid,
        other=0,
    )
    slot = tl.load(
        block_table_ptr + req * stride_bt_req + logical_block * stride_bt_block,
        mask=valid,
        other=0,
    )

    # The old flat pool is [role-page, H, P128, packed-D].  Viewing its
    # contiguous storage as [physical-page, 1, P32, packed-D] makes the new
    # page id ((old_page * H + head) * 4 + subpage).
    old_k_page = slot * subpages_per_slot
    k_page = (old_k_page * num_kv_heads + kv_head) * 4 + subpage
    v_page = ((old_k_page + 1) * num_kv_heads + kv_head) * 4 + subpage
    k_page = tl.where(valid, k_page, 0)
    v_page = tl.where(valid, v_page, 0)
    out_base = out_table_ptr + pseudo_req * stride_out_req + page_offset * stride_out_block
    tl.store(out_base, k_page, mask=active)
    tl.store(out_base + stride_out_role, v_page, mask=active)

    # The selected logical block ids are ascending.  Only the final selected
    # block can therefore be the sequence's partial last P128 block.
    last_offset = tl.maximum(real_topk - 1, 0)
    last_selected = tl.load(
        topk_ptr + token * stride_t_token + kv_head * stride_t_head + last_offset * stride_t_topk,
        mask=active & (real_topk > 0),
        other=0,
    )
    unused_tail = num_logical_blocks * 128 - kv_len
    compact_len = real_topk * 128
    compact_len -= tl.where(last_selected == num_logical_blocks - 1, unused_tail, 0)
    compact_len = tl.where(real_topk > 0, compact_len, 0)
    tl.store(out_seq_lens_ptr + pseudo_req, compact_len, mask=active)


def write_sparse_p32_table(
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    subpages_per_slot: int,
    out_table: torch.Tensor,
    out_seq_lens: torch.Tensor,
) -> None:
    """Write TRTLLM-Gen pseudo-request metadata into caller-owned buffers."""
    total_q, num_kv_heads, max_topk = topk_idx.shape
    num_pseudo_requests = total_q * num_kv_heads
    expected_table = (
        num_pseudo_requests,
        2,
        max_topk * _PAGES_PER_SELECTED_BLOCK,
    )
    if tuple(out_table.shape) != expected_table:
        raise ValueError(
            f"expected sparse P32 table {expected_table}, got {tuple(out_table.shape)}"
        )
    if tuple(out_seq_lens.shape) != (num_pseudo_requests,):
        raise ValueError(
            f"expected {num_pseudo_requests} sparse sequence lengths, "
            f"got {tuple(out_seq_lens.shape)}"
        )
    if topk_idx.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise ValueError("MiniMax-M3 sparse page tables must be int32")
    if seq_lens.dtype != torch.int32 or out_table.dtype != torch.int32:
        raise ValueError("MiniMax-M3 sparse sequence metadata must be int32")
    if out_seq_lens.dtype != torch.int32:
        raise ValueError("MiniMax-M3 sparse output sequence lengths must be int32")
    if max_topk <= 0 or max_topk & (max_topk - 1):
        raise ValueError(f"topk must be a positive power of two; got {max_topk}")

    _build_sparse_p32_table_kernel[(num_pseudo_requests,)](
        topk_idx,
        block_table,
        seq_lens,
        out_table,
        out_seq_lens,
        num_pseudo_requests,
        num_kv_heads,
        decode_query_len,
        subpages_per_slot,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        out_table.stride(0),
        out_table.stride(1),
        out_table.stride(2),
        max_topk,
        max_topk * _PAGES_PER_SELECTED_BLOCK,
    )


def build_sparse_p32_table(
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    subpages_per_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate and build pseudo-request metadata (test/benchmark wrapper)."""
    total_q, num_kv_heads, max_topk = topk_idx.shape
    num_pseudo_requests = total_q * num_kv_heads
    out_table = torch.empty(
        (num_pseudo_requests, 2, max_topk * _PAGES_PER_SELECTED_BLOCK),
        dtype=torch.int32,
        device=topk_idx.device,
    )
    out_seq_lens = torch.empty(
        num_pseudo_requests,
        dtype=torch.int32,
        device=topk_idx.device,
    )
    write_sparse_p32_table(
        topk_idx,
        block_table,
        seq_lens,
        decode_query_len,
        subpages_per_slot,
        out_table,
        out_seq_lens,
    )
    return out_table, out_seq_lens


def _get_sparse_buffers(topk_idx: torch.Tensor, reserve: bool) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_kv_heads, max_topk = topk_idx.shape
    pseudo_batch = total_q * num_kv_heads
    buffers = get_memory_buffers()
    table = buffers.get_buffer(
        [pseudo_batch, 2, max_topk * _PAGES_PER_SELECTED_BLOCK],
        torch.int32,
        buffer_name="m3_trtllm_gen_sparse_block_table",
        reserve_buffer=reserve,
    )
    lengths = buffers.get_buffer(
        [pseudo_batch],
        torch.int32,
        buffer_name="m3_trtllm_gen_sparse_seq_lens",
        reserve_buffer=reserve,
    )
    return table, lengths


def _get_native_output(output: torch.Tensor, reserve: bool) -> torch.Tensor:
    """Return the E4M3 output buffer required by shipped NVFP4 cubins.

    The SM100/SM103 TRTLLM-Gen manifest only contains Q=E4M3, KV=E2M1,
    O=E4M3 generation kernels for this shape. Quantized ``o_proj`` consumers
    use the caller's E4M3 buffer directly; BF16 callers use a graph-stable
    temporary and widen after FMHA.
    """
    if not output.is_contiguous():
        raise ValueError("TRTLLM-Gen sparse decode requires a contiguous output buffer")
    if output.dtype == torch.float8_e4m3fn:
        return output
    return get_memory_buffers().get_buffer(
        list(output.shape),
        torch.float8_e4m3fn,
        buffer_name="m3_trtllm_gen_sparse_native_output",
        reserve_buffer=reserve,
    )


def _get_bmm_scales(
    owner: object,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    sm_scale: float,
    *,
    refresh: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return graph-stable TRTLLM-Gen raw/log2e BMM scale buffers.

    ``bmm2_scale`` converts the NVFP4 V operand back to the logical attention
    dtype.  Native E4M3 output is paired with unit MXFP8 block scales, so it
    must remain in the same logical scale as the BF16 output path.
    """
    source = (
        int(k_global_scale.data_ptr()),
        int(v_global_scale.data_ptr()),
        float(sm_scale),
    )
    cache = getattr(owner, "_msa_trtllm_gen_sparse_scales", None)
    if cache is None:
        if k_global_scale.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TRTLLM-Gen sparse BMM scales must be initialized during warmup")
        cache = torch.empty((2, 4), dtype=torch.float32, device=k_global_scale.device)
        owner._msa_trtllm_gen_sparse_scales = cache
        refresh = True
    if cache.device != k_global_scale.device or cache.device != v_global_scale.device:
        raise ValueError("TRTLLM-Gen sparse scale reload must preserve the scale buffer device")
    if refresh or getattr(owner, "_msa_trtllm_gen_sparse_scale_source", None) != source:
        cache[0, :1].copy_(k_global_scale).mul_(float(sm_scale))
        cache[0, 1:2].copy_(cache[0, :1]).mul_(_LOG2_E)
        cache[1, :1].copy_(v_global_scale)
        owner._msa_trtllm_gen_sparse_scale_source = source
    return cache[0, :2], cache[1, :1]


@torch.no_grad()
def minimax_m3_trtllm_gen_sparse_decode(
    owner: object,
    q: torch.Tensor,
    kv_cache_manager: _MiniMaxM3SparseKVCacheManager,
    layer_idx: int,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    max_num_requests: int,
    k_global_scale: torch.Tensor,
    v_global_scale: torch.Tensor,
    enable_pdl: bool = True,
) -> None:
    """Run selected-page NVFP4 decode through TRTLLM-Gen in place."""
    if get_sm_version() not in (100, 103):
        raise RuntimeError("Experimental M3 TRTLLM-Gen NVFP4 sparse decode requires SM100/SM103")
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = int(topk_idx.shape[1])
    check_decode_span_shape(
        "MiniMax-M3 TRTLLM-Gen sparse decode",
        total_q,
        int(seq_lens.shape[0]),
        decode_query_len,
    )
    if head_dim != 128 or num_heads % num_kv_heads:
        raise ValueError(
            f"TRTLLM-Gen sparse decode requires H128 GQA; got heads={num_heads}, "
            f"kv_heads={num_kv_heads}, head_dim={head_dim}"
        )

    data_pool, subpages_per_slot = kv_cache_manager.get_kv_subpage_pool(layer_idx, "HND")
    scale_pool, scale_subpages_per_slot = kv_cache_manager.get_kv_scale_subpage_pool(
        layer_idx, "HND"
    )
    if int(scale_subpages_per_slot) != int(subpages_per_slot):
        raise RuntimeError("MiniMax-M3 NVFP4 data and scale page geometry differs")
    expected_data_tail = (num_kv_heads, _M3_PAGE_SIZE, head_dim // 2)
    expected_scale_tail = (num_kv_heads, _M3_PAGE_SIZE, head_dim // 16)
    if tuple(data_pool.shape[1:]) != expected_data_tail:
        raise ValueError(
            f"expected packed NVFP4 pool tail {expected_data_tail}, got {tuple(data_pool.shape[1:])}"
        )
    if tuple(scale_pool.shape[1:]) != expected_scale_tail:
        raise ValueError(
            f"expected NVFP4 scale pool tail {expected_scale_tail}, got {tuple(scale_pool.shape[1:])}"
        )
    if not data_pool.is_contiguous() or not scale_pool.is_contiguous():
        raise ValueError("TRTLLM-Gen sparse P32 views require contiguous flat pools")

    physical_pages = int(data_pool.shape[0]) * num_kv_heads * _PAGES_PER_SELECTED_BLOCK
    kv_pool = data_pool.view(torch.uint8).view(
        physical_pages,
        1,
        _TRTLLM_GEN_PAGE_SIZE,
        head_dim // 2,
    )
    kv_scale_pool = scale_pool.view(torch.float8_e4m3fn).view(
        physical_pages,
        1,
        _TRTLLM_GEN_PAGE_SIZE,
        head_dim // 16,
    )

    gqa_group = num_heads // num_kv_heads
    q_pseudo = q.to(torch.float8_e4m3fn).reshape(total_q, num_kv_heads, gqa_group, head_dim)
    q_pseudo = q_pseudo.flatten(0, 1)
    max_pseudo_requests = max_num_requests * decode_query_len * num_kv_heads
    if int(q_pseudo.shape[0]) > max_pseudo_requests:
        raise ValueError("Sparse decode batch exceeds max_num_requests")
    reserve = torch.cuda.is_current_stream_capturing()
    native_output = _get_native_output(output, reserve)
    native_output_pseudo = native_output.reshape(
        total_q, num_kv_heads, gqa_group, head_dim
    ).flatten(0, 1)
    sparse_table, sparse_seq_lens = _get_sparse_buffers(topk_idx, reserve)
    write_sparse_p32_table(
        topk_idx,
        block_table,
        seq_lens,
        decode_query_len,
        int(subpages_per_slot),
        sparse_table,
        sparse_seq_lens,
    )
    bmm1_scale, bmm2_scale = _get_bmm_scales(
        getattr(owner, "attn", owner),
        k_global_scale,
        v_global_scale,
        sm_scale,
    )

    import flashinfer

    workspace = get_memory_buffers().get_buffer(
        [_workspace(q_pseudo.dtype, gqa_group, head_dim, 1)],
        torch.uint8,
        buffer_name="m3_trtllm_gen_sparse_workspace",
        reserve_buffer=reserve,
    )
    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
        query=q_pseudo,
        kv_cache=(kv_pool, kv_pool),
        workspace_buffer=workspace,
        block_tables=sparse_table,
        seq_lens=sparse_seq_lens,
        max_seq_len=int(topk_idx.shape[-1]) * _M3_PAGE_SIZE,
        bmm1_scale=bmm1_scale[:1],
        bmm1_scale_log2=bmm1_scale[1:2],
        bmm2_scale=bmm2_scale,
        window_left=-1,
        out=native_output_pseudo,
        sinks=None,
        kv_layout="HND",
        enable_pdl=enable_pdl,
        backend="trtllm-gen",
        q_len_per_req=1,
        kv_cache_sf=(kv_scale_pool, kv_scale_pool),
        uses_shared_paged_kv_idx=False,
        multi_ctas_kv_counter_buffer=_counter_buffer(
            q.device, gqa_group, max_pseudo_requests, reserve
        ),
    )
    if native_output.data_ptr() != output.data_ptr():
        output.copy_(native_output)
