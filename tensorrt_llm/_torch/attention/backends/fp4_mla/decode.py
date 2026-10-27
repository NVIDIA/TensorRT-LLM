# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 MLA decode dispatch and Triton/CuTeDSL launch preparation."""

import os
from typing import Any

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version

from .cache_update import _prepare_fp4_mla_q_buffers
from .config import (
    _FP4_MLA_CUTEDSL_BACKEND,
    FP4_BLOCK_SIZE,
    FP4_MLA_ATTENTION_BACKEND_ENV,
    FP4_MLA_K_RESIDUAL_DIM,
    FP4_MLA_P_GLOBAL_SCALE,
    FP4_MLA_Q_LOGICAL_DIM,
    FP4_MLA_Q_PACKED_DIM,
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    _ceil_div,
    _cutedsl_backend_available,
    _env_int,
    _fp4_mla_attention_backend,
    _fp4_mla_cutedsl_fused_v_transpose_enabled,
    _fp4_mla_cutedsl_kernel_module,
    _fp4_mla_packed_rows_per_seq,
)
from .layout import (
    _ensure_workspace_tensor,
    _get_fp4_mla_global_scale,
    _get_fp4_mla_kv_cache_tensors,
    _get_fp4_mla_q_global_scale,
    _get_fp4_mla_swizzled_scale_size,
    _validate_fp4_mla_attention_q_shape,
    _validate_fp4_mla_cache_shape,
    _validate_fp4_mla_kv_storage_shape,
    get_fp4_mla_v_scale_pool_view,
)
from .metadata import (
    _fp4_mla_generation_page_ids,
    _fp4_mla_uniform_generation_lengths,
    _get_linear_mtp_query_len_per_seq,
    _infer_assume_full_pages,
    _materialize_fp4_mla_device_page_table_for_forward,
    _max_generation_pages,
)
from .packed_plan import (
    _fp4_mla_packed_plan_geometry,
    _get_sm_count,
    _launch_fp4_mla_packed_plan,
    _PackedPlanGeometry,
)
from .v_cache import (
    _get_cutedsl_persistent_v_packed_cache,
    _get_fp4_mla_v_packed_pool_base,
    _get_fp4_mla_v_scale_pool_base,
    _get_triton_v_packed_cache,
    _select_triton_block_v,
    _triton_can_prepack_v,
    _triton_prepack_v_enabled,
    _update_triton_v_packed_cache,
)


@triton.jit
def _cutedsl_swizzled_sf_offset(row_idx, col_idx, sf_cols: tl.constexpr):
    padded_cols = ((sf_cols + 3) // 4) * 4
    return (
        col_idx % 4
        + (col_idx // 4) * (4 * 128)
        + (row_idx % 32) * 16
        + ((row_idx % 128) // 32) * 4
        + (row_idx // 128) * (128 * padded_cols)
    )


@triton.jit
def _cutedsl_pad_q_and_sf_kernel(
    q_padded_ptr,
    q_ptr,
    q_sf_padded_ptr,
    q_sf_ptr,
    num_heads,
    output_heads: tl.constexpr,
    packed_dim: tl.constexpr,
    block_bytes: tl.constexpr,
    sf_cols: tl.constexpr,
    sf_cols_per_byte_block: tl.constexpr,
):
    query_idx = tl.program_id(0)
    byte_block = tl.program_id(1)
    head_offsets = tl.arange(0, output_heads)
    byte_offsets = byte_block * block_bytes + tl.arange(0, block_bytes)
    head_mask = head_offsets < num_heads
    byte_mask = byte_offsets < packed_dim
    source_rows = query_idx * num_heads + head_offsets
    destination_rows = query_idx * output_heads + head_offsets
    values = tl.load(
        q_ptr + source_rows[:, None] * packed_dim + byte_offsets[None, :],
        mask=head_mask[:, None] & byte_mask[None, :],
        other=0,
    )
    tl.store(
        q_padded_ptr + destination_rows[:, None] * packed_dim + byte_offsets[None, :],
        values,
        mask=byte_mask[None, :],
    )
    sf_col_offsets = byte_block * sf_cols_per_byte_block + tl.arange(0, sf_cols_per_byte_block)
    sf_col_mask = sf_col_offsets < sf_cols
    source_offsets = _cutedsl_swizzled_sf_offset(
        source_rows[:, None], sf_col_offsets[None, :], sf_cols
    )
    destination_offsets = _cutedsl_swizzled_sf_offset(
        destination_rows[:, None], sf_col_offsets[None, :], sf_cols
    )
    sf_values = tl.load(
        q_sf_ptr + source_offsets,
        mask=head_mask[:, None] & sf_col_mask[None, :],
        other=1.0,
    )
    tl.store(
        q_sf_padded_ptr + destination_offsets,
        sf_values,
        mask=sf_col_mask[None, :],
    )


def _run_triton_attention_decode(
    *,
    metadata: Any,
    layer_idx: int,
    local_layer: int,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_sf: torch.Tensor,
    global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    kv_lens: torch.Tensor,
    p_fp4: torch.Tensor,
    p_sf: torch.Tensor,
    max_scores: torch.Tensor,
    denom: torch.Tensor,
    output: torch.Tensor,
    num_queries: int,
    num_heads: int,
    head_dim: int,
    kv_lora_rank: int,
    q_residual_dim: int,
    query_len_per_seq: int,
    max_pages: int,
    sm_scale: float,
    q_global_scale: torch.Tensor,
) -> None:
    """Dispatch the ``triton`` FP4 MLA decode pipeline.

    Mirrors the four-stage layout used by ``fp4_mla_cutile.py``
    (page-stats with packed P -> reduce-stats -> prob-scale -> PV) but
    routes through the self-contained kernels in
    ``fp4_mla_triton.py``. Threads through the constexpr assume flags,
    TMA descriptors, occupancy/num-warps launch meta, and pipelined PV loop.
    """
    from .fp4_mla_triton import (
        _fp4_mla_attention_group_reduce_stats_kernel as _attn_group_reduce_stats_kernel,
    )
    from .fp4_mla_triton import _fp4_mla_attention_page_stats_kernel as _attn_page_stats_kernel
    from .fp4_mla_triton import _fp4_mla_attention_prob_scale_kernel as _attn_prob_scale_kernel
    from .fp4_mla_triton import _fp4_mla_attention_pv_kernel as _attn_pv_kernel
    from .fp4_mla_triton import (
        _fp4_mla_attention_pv_prepacked_v_kernel as _attn_pv_prepacked_v_kernel,
    )
    from .fp4_mla_triton import _fp4_mla_attention_pv_reduce_kernel as _attn_pv_reduce_kernel
    from .fp4_mla_triton import _fp4_mla_attention_reduce_stats_kernel as _attn_reduce_stats_kernel

    block_h = 128
    block_t = metadata.page_size
    # Adaptive BLOCK_V: the fallback PV path uses a finer V split at small batch
    # on B200 (~148 SMs). PV grid = num_queries * num_head_blocks(1) *
    # (kv_lora_rank / BLOCK_V). We want >= ~2*num_SMs programs so that >1 CTA
    # lands per SM and hides the L1TEX scoreboard stalls. Empirically (sweep):
    #   bs<=32 -> BLOCK_V=32; bs>=64 -> BLOCK_V=128.
    # (BLOCK_V=16 is rejected by the V TMA descriptor min-stride requirement.)
    # With prepacked V, BLOCK_V=128 avoids reloading the same P tile four times
    # and matches the cutile prepacked-V tile shape.
    block_v = _select_triton_block_v(num_queries, prefer_prepacked_v=_triton_prepack_v_enabled())
    q_storage_head_dim = head_dim + q_residual_dim
    # The virtual GEMM tail evaluates QK + Q_r K + Q K_r in one reduction.
    # Q and Q_r still occupy the 640-channel interleaved physical Q buffer;
    # the final Q term reuses Q's main tail groups while K_r comes from the
    # contiguous 64-channel tail of the primary paged KV cache.
    q_head_dim = head_dim + q_residual_dim + FP4_MLA_K_RESIDUAL_DIM
    # BLOCK_K = 512 aligns the K-window with the 512-channel non-residual prefix.
    block_k = 512
    full_block_end = (q_head_dim // block_k) * block_k
    tail_k = q_head_dim - full_block_end
    tail_block_k = 1 << (tail_k - 1).bit_length() if tail_k > 0 else block_k
    q_sf_per_token = q_storage_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = (head_dim + FP4_MLA_K_RESIDUAL_DIM) // FP4_BLOCK_SIZE
    sf_per_page = metadata.page_size // FP4_BLOCK_SIZE
    num_head_blocks = triton.cdiv(num_heads, block_h)

    assume_full_heads = num_heads % block_h == 0
    assume_full_v = kv_lora_rank % block_v == 0
    # Match the cutile path: only mark pages "full" when we can prove every
    # generation sequence has the same number of cached tokens AND
    # query_len_per_seq == 1 (so the kv_len adjustment is a no-op).
    assume_full_pages = (
        _infer_assume_full_pages(metadata, max_pages, metadata.page_size) and query_len_per_seq == 1
    )
    # Leave validity checks on. Matches cutile's default and is correctness-
    # safe. The perfect-shape PV fast path (tl.ext.make_view + load_view_tko)
    # remains gated off — when measured on the TileIR backend (ENABLE_TILE=1)
    # it was net-slower on the bench, so the cost of enabling it isn't worth
    # the win on the FP4 MLA shapes we care about.
    assume_valid_pages = False
    num_gen_seqs = num_queries // query_len_per_seq
    if (
        not assume_valid_pages
        and assume_full_pages
        and src_page_ids.numel() == num_gen_seqs * max_pages
    ):
        assume_valid_pages = True
    # cutile checks only `make_tensor_descriptor`; on the nvt backend the
    # presence of TMA descriptors implies `tl.ext.make_view` is available too.
    use_tma_data_load = hasattr(triton.language, "make_tensor_descriptor")

    # Install the device-side scratch allocator on every call. Triton stores
    # the allocator in a ContextVar (triton.runtime._allocation), so a single
    # process-wide install is not visible from worker threads / asyncio tasks
    # that run with a different Context — the kernel launch would then hit the
    # default NullAllocator and raise. Matches the cutile path.
    if use_tma_data_load:

        def _tma_alloc(size: int, alignment: int, stream):
            return torch.empty(size, device=q_fp4.device, dtype=torch.int8)

        triton.set_allocator(_tma_alloc)

    # cutile-equivalent launch meta. occupancy=2 lets two CTAs land per SM
    # which improves wave-tail efficiency at the bs=32 hot point.
    # NOTE: num_stages=2 (instead of the Triton 3.6 default of 3) sidesteps
    # the TritonGPUAutomaticWarpSpecialization + NVWSInsertTmemAref pass that
    # ICEs on the page_stats kernel under Triton 3.6.0 / sm_100.
    launch_meta = {"occupancy": 2}
    # The matmul kernels (page-stats QK and PV) are register-limited: at the
    # Triton default of num_warps=4 the [BLOCK_H, BLOCK_T] epilogue spills the
    # register file down to ~2 CTAs/SM (12.5% occupancy), so there are too few
    # warps to hide the QK/PV load latency (ncu: ~0.3 eligible warps/scheduler).
    # Spreading the tile epilogue over num_warps=8 halves the per-thread
    # register need and roughly doubles resident warps. Matches the cutile
    # ("nvt") backend, which launches page-stats at num_warps=8. Both are
    # overridable for tuning.
    sm_count = _get_sm_count(q_fp4.device)
    # page-stats num_warps: the full-pages fast path (uniform q_len==1 decode)
    # benefits from num_warps=8 (more warps hide the QK load latency); the
    # masked path (q_len>1 / ragged lengths) carries extra per-thread state and
    # measured markedly faster at num_warps=4 (e.g. bs256 q_len4: 131->95ms).
    page_stats_num_warps = _env_int("TRTLLM_FP4_MLA_PAGE_STATS_NUM_WARPS")
    if page_stats_num_warps is None:
        page_stats_num_warps = 8 if assume_full_pages else 4
    page_stats_launch_meta = {"occupancy": 2, "num_warps": page_stats_num_warps}
    # PV benefits from num_warps=8 across shapes measured.
    pv_num_warps = _env_int("TRTLLM_FP4_MLA_PV_NUM_WARPS") or 8
    pv_launch_meta = {"occupancy": 2, "num_warps": pv_num_warps}
    # PV loop pipelining. With TMA loads, num_stages>=2 lets the next page's
    # loads overlap with the current MMA via mbarrier. The PV report shows
    # long_scoreboard=4.5 cycles avg on V loads at PV_LOOP_STAGES=2; bumping the
    # depth pays off when the grid is small enough that occupancy can absorb
    # the extra in-flight tile state — i.e. medium batch / large max_pages.
    # Larger pipelines hurt at small batch (more live state, fewer dim blocks).
    if num_queries <= 16 or max_pages <= 4:
        pv_loop_stages = 2
    else:
        pv_loop_stages = 3

    # Page-stats kernel: per (query, head_block, page) program, does QK,
    # softmax stats, and packs probs into FP4 with the per-page local-max
    # scaling trick. The page-max correction is applied later by
    # prob_scale_kernel via p_sf in-place rescaling.
    page_stats_shape = (num_queries, max_pages, num_heads)
    page_max = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_page_max_buf",
        page_stats_shape,
        dtype=torch.float32,
        device=q_fp4.device,
    )
    page_sum = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_page_sum_buf",
        page_stats_shape,
        dtype=torch.float32,
        device=q_fp4.device,
    )

    pack_prob_in_page_stats = True
    _attn_page_stats_kernel[(num_queries, num_head_blocks, max_pages)](
        page_max,
        page_sum,
        p_fp4,
        p_sf,
        q_fp4,
        q_sf,
        kv_cache,
        sf_cache,
        global_scale,
        q_global_scale,
        src_page_ids,
        metadata.fp4_mla_state.paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        kv_cache.shape[0],
        q_fp4.stride(0),
        q_fp4.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(2),
        kv_cache.stride(4),
        sf_cache.stride(0),
        page_max.stride(0),
        page_max.stride(1),
        p_fp4.stride(0),
        p_fp4.stride(1),
        p_fp4.shape[0],
        q_fp4.shape[0],
        sm_scale,
        NUM_HEADS=num_heads,
        Q_HEAD_D=q_head_dim,
        Q_STORAGE_HEAD_D=q_storage_head_dim,
        K_HEAD_D=head_dim,
        Q_RESIDUAL_D=q_residual_dim,
        K_RESIDUAL_D=FP4_MLA_K_RESIDUAL_DIM,
        PAGE_SIZE=metadata.page_size,
        FP4_BLOCK=FP4_BLOCK_SIZE,
        Q_SF_PER_TOKEN=q_sf_per_token,
        K_SF_PER_TOKEN=k_sf_per_token,
        SF_PER_PAGE=sf_per_page,
        P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
        QUERY_LEN_PER_SEQ=query_len_per_seq,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        BLOCK_T=block_t,
        BLOCK_K=block_k,
        FULL_BLOCK_END=full_block_end,
        TAIL_BLOCK_K=tail_block_k,
        USE_TMA_DATA_LOAD=use_tma_data_load,
        PACK_PROBS=pack_prob_in_page_stats,
        ASSUME_FULL_HEADS=assume_full_heads,
        ASSUME_FULL_PAGES=assume_full_pages,
        ASSUME_VALID_PAGES=assume_valid_pages,
        **page_stats_launch_meta,
    )
    # Two-level softmax-stats reduction. The single-level reduce launched only
    # (num_queries * num_head_blocks) CTAs, each serially walking all max_pages
    # twice -- at small batch that handful of CTAs left the GPU almost idle and
    # the reduce cost more than the QK matmul. Level 1 parallelizes the page
    # reduction across a page-group axis (online-softmax partials, pipelined);
    # level 2 reuses the existing reduce kernel to fold the few groups into the
    # global (max, denom). When the (query, head) grid already fills the GPU the
    # group count collapses to 1 and this degenerates to the original reduce.
    seqhead_ctas = num_queries * num_head_blocks
    # Aim for ~3 waves of level-1 CTAs so page loads have enough memory-level
    # parallelism to hide latency, while keeping the group count small enough
    # that the level-2 combine loop stays short.
    target_l1_ctas = 3 * sm_count
    num_reduce_groups = _ceil_div(target_l1_ctas, max(seqhead_ctas, 1))
    num_reduce_groups = max(1, min(num_reduce_groups, max_pages, 64))
    # The grouped (two-level) reduce needs an auxiliary workspace, and
    # _ensure_workspace_tensor can only (re)allocate it outside CUDA graph
    # capture. If a warmup forward did not already size that workspace (e.g. the
    # warmup batch took the single-level path), fall back to the single-level
    # reduce during capture so we never allocate mid-capture. The single-level
    # reduce is numerically identical (it just launches fewer CTAs).
    if num_reduce_groups > 1 and torch.cuda.is_current_stream_capturing():
        gmax = metadata.fp4_mla_state.workspaces.get("_fp4_mla_attention_group_max_buf")
        gsum = metadata.fp4_mla_state.workspaces.get("_fp4_mla_attention_group_sum_buf")
        groups_ready = (
            gmax is not None
            and gsum is not None
            and gmax.shape[0] >= num_queries
            and gmax.shape[1] >= num_reduce_groups
            and gmax.shape[2] >= num_heads
            and gsum.shape[0] >= num_queries
            and gsum.shape[1] >= num_reduce_groups
            and gsum.shape[2] >= num_heads
        )
        if not groups_ready:
            num_reduce_groups = 1
    if num_reduce_groups <= 1:
        _attn_reduce_stats_kernel[(num_queries, num_head_blocks)](
            max_scores,
            denom,
            page_max,
            page_sum,
            max_pages,
            max_scores.stride(0),
            page_max.stride(0),
            page_max.stride(1),
            NUM_HEADS=num_heads,
            MAX_PAGES=max_pages,
            BLOCK_H=block_h,
            **launch_meta,
        )
    else:
        group_pages = _ceil_div(max_pages, num_reduce_groups)
        num_reduce_groups = _ceil_div(max_pages, group_pages)
        group_max = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_group_max_buf",
            (num_queries, num_reduce_groups, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        group_sum = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_group_sum_buf",
            (num_queries, num_reduce_groups, num_heads),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        _attn_group_reduce_stats_kernel[(num_queries, num_head_blocks, num_reduce_groups)](
            group_max,
            group_sum,
            page_max,
            page_sum,
            max_pages,
            group_max.stride(0),
            group_max.stride(1),
            page_max.stride(0),
            page_max.stride(1),
            NUM_HEADS=num_heads,
            GROUP_PAGES=group_pages,
            BLOCK_H=block_h,
            PIPELINE_STAGES=min(group_pages, 4),
            **launch_meta,
        )
        _attn_reduce_stats_kernel[(num_queries, num_head_blocks)](
            max_scores,
            denom,
            group_max,
            group_sum,
            num_reduce_groups,
            max_scores.stride(0),
            group_max.stride(0),
            group_max.stride(1),
            NUM_HEADS=num_heads,
            MAX_PAGES=num_reduce_groups,
            BLOCK_H=block_h,
            **launch_meta,
        )
    _attn_prob_scale_kernel[(num_queries, num_head_blocks, max_pages)](
        p_sf,
        max_scores,
        denom,
        page_max,
        metadata.fp4_mla_state.paged_kv_indptr_decode,
        kv_lens,
        src_page_ids.shape[0],
        max_scores.stride(0),
        page_max.stride(0),
        page_max.stride(1),
        NUM_HEADS=num_heads,
        PAGE_SIZE=metadata.page_size,
        SF_PER_PAGE=sf_per_page,
        QUERY_LEN_PER_SEQ=query_len_per_seq,
        MAX_PAGES=max_pages,
        BLOCK_H=block_h,
        ASSUME_FULL_HEADS=assume_full_heads,
        ASSUME_FULL_PAGES=assume_full_pages,
        ASSUME_VALID_PAGES=assume_valid_pages,
        **launch_meta,
    )
    num_dim_blocks = triton.cdiv(kv_lora_rank, block_v)
    v_packed = _get_triton_v_packed_cache(
        metadata,
        layer_idx,
        kv_cache,
        v_head_dim=kv_lora_rank,
        page_size=metadata.page_size,
        block_v=block_v,
        local_layer=local_layer,
        v_sf=v_sf,
        page_ids=src_page_ids,
    )
    if (
        v_packed is None
        and _triton_can_prepack_v(kv_lora_rank, metadata.page_size, block_v)
        and not torch.cuda.is_current_stream_capturing()
    ):
        v_packed = _update_triton_v_packed_cache(
            metadata,
            layer_idx,
            kv_cache,
            src_page_ids,
            v_head_dim=kv_lora_rank,
            page_size=metadata.page_size,
            block_v=block_v,
            local_layer=local_layer,
            v_sf=v_sf,
        )
    use_triton_v_packed_cache = v_packed is not None

    # PV page split: partition the page range across additional programs and
    # reduce in a follow-up kernel. ncu showed PV at waves/SM=0.49 for bs=32 —
    # PV is L1-bandwidth bound, so raising in-flight CTAs is the lever.
    # BLOCK_V is bounded below by the 16-byte TMA descriptor min-stride.
    # PV page split: ncu shows that with the current shape (bs=32, max_pages=256)
    # the PV kernel is L1-cache-throughput bound (long_scoreboard=4.5 cycles
    # avg, L1 global LD hit-rate <40%). Increasing the program count via page
    # splitting reduced waves/SM idle time but did NOT improve wall-time at
    # current shapes — the per-CTA L1 thrash is the limit. Gate the split off
    # by default; re-enable only for very small grids where occupancy is the
    # bottleneck rather than per-CTA L1 pressure.
    page_split = 1
    base_grid = num_queries * num_head_blocks * num_dim_blocks
    if max_pages >= 16 and base_grid < 148:
        for p in (8, 4, 2):
            if max_pages % p == 0 and max_pages // p >= 16 and base_grid * p <= 148 * 4:
                page_split = p
                break
    # The page-split PV path needs a partial-output workspace, which
    # _ensure_workspace_tensor can only (re)allocate outside CUDA graph capture.
    # Fall back to the unsplit PV (numerically identical) during capture unless a
    # warmup forward already sized that workspace, so capture never allocates.
    if page_split > 1 and torch.cuda.is_current_stream_capturing():
        pbuf = metadata.fp4_mla_state.workspaces.get("_fp4_mla_attention_pv_partial_buf")
        partial_ready = (
            pbuf is not None
            and pbuf.shape[0] >= num_queries
            and pbuf.shape[1] >= page_split
            and pbuf.shape[2] >= num_heads
            and pbuf.shape[3] >= kv_lora_rank
        )
        if not partial_ready:
            page_split = 1
    if page_split > 1:
        pages_per_split = max_pages // page_split
        partial_out = _ensure_workspace_tensor(
            metadata,
            "_fp4_mla_attention_pv_partial_buf",
            (num_queries, page_split, num_heads, kv_lora_rank),
            dtype=torch.float32,
            device=q_fp4.device,
        )
        if use_triton_v_packed_cache:
            _attn_pv_prepacked_v_kernel[
                (num_queries, num_head_blocks, num_dim_blocks * page_split)
            ](
                output,
                p_fp4,
                p_sf,
                v_packed,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.fp4_mla_state.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
                USE_TMA_OUT_STORE=use_tma_data_load and assume_full_heads and assume_full_v,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                PAGE_SPLIT=page_split,
                PAGES_PER_SPLIT=pages_per_split,
                PARTIAL_OUT=True,
                partial_out_ptr=partial_out,
                partial_s0=partial_out.stride(0),
                partial_s1=partial_out.stride(1),
                partial_s2=partial_out.stride(2),
                partial_s3=partial_out.stride(3),
                **pv_launch_meta,
            )
        else:
            _attn_pv_kernel[(num_queries, num_head_blocks, num_dim_blocks * page_split)](
                output,
                p_fp4,
                p_sf,
                kv_cache,
                kv_cache,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.fp4_mla_state.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                kv_cache.stride(0),
                kv_cache.stride(2),
                kv_cache.stride(4),
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
                USE_TMA_V_LOAD=use_tma_data_load and kv_lora_rank % block_v == 0,
                USE_PREPACKED_V=False,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                PAGE_SPLIT=page_split,
                PAGES_PER_SPLIT=pages_per_split,
                PARTIAL_OUT=True,
                partial_out_ptr=partial_out,
                partial_s0=partial_out.stride(0),
                partial_s1=partial_out.stride(1),
                partial_s2=partial_out.stride(2),
                partial_s3=partial_out.stride(3),
                **pv_launch_meta,
            )
        _attn_pv_reduce_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
            output,
            partial_out,
            global_scale,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            NUM_HEADS=num_heads,
            V_HEAD_D=kv_lora_rank,
            PAGE_SPLIT=page_split,
            P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
            BLOCK_H=block_h,
            BLOCK_V=block_v,
            ASSUME_FULL_HEADS=assume_full_heads,
            ASSUME_FULL_V=assume_full_v,
            **launch_meta,
        )
    else:
        if use_triton_v_packed_cache:
            _attn_pv_prepacked_v_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
                output,
                p_fp4,
                p_sf,
                v_packed,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.fp4_mla_state.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
                USE_TMA_OUT_STORE=use_tma_data_load and assume_full_heads and assume_full_v,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **pv_launch_meta,
            )
        else:
            _attn_pv_kernel[(num_queries, num_head_blocks, num_dim_blocks)](
                output,
                p_fp4,
                p_sf,
                kv_cache,
                kv_cache,
                v_sf,
                global_scale,
                src_page_ids,
                metadata.fp4_mla_state.paged_kv_indptr_decode,
                kv_lens,
                src_page_ids.shape[0],
                kv_cache.shape[0],
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.shape[0] * output.shape[1],
                p_fp4.stride(0),
                p_fp4.stride(1),
                p_fp4.shape[0],
                kv_cache.stride(0),
                kv_cache.stride(2),
                kv_cache.stride(4),
                v_sf.stride(0),
                NUM_HEADS=num_heads,
                V_HEAD_D=kv_lora_rank,
                PAGE_SIZE=metadata.page_size,
                FP4_BLOCK=FP4_BLOCK_SIZE,
                SF_PER_PAGE=sf_per_page,
                QUERY_LEN_PER_SEQ=query_len_per_seq,
                MAX_PAGES=max_pages,
                P_GLOBAL_SCALE=FP4_MLA_P_GLOBAL_SCALE,
                BLOCK_H=block_h,
                BLOCK_V=block_v,
                USE_TMA_P_LOAD=use_tma_data_load and assume_full_heads and assume_valid_pages,
                USE_TMA_V_LOAD=use_tma_data_load and kv_lora_rank % block_v == 0,
                USE_PREPACKED_V=False,
                PV_LOOP_STAGES=pv_loop_stages,
                ASSUME_FULL_HEADS=assume_full_heads,
                ASSUME_FULL_PAGES=assume_full_pages,
                ASSUME_FULL_V=assume_full_v,
                ASSUME_VALID_PAGES=assume_valid_pages,
                **pv_launch_meta,
            )


def _run_cutedsl_packed_decode(
    module: Any,
    metadata: Any,
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv_cache: torch.Tensor,
    sf_cache: torch.Tensor,
    v_packed: torch.Tensor,
    v_sf: torch.Tensor,
    kv_global_scale: torch.Tensor,
    q_global_scale: torch.Tensor,
    src_page_ids: torch.Tensor,
    page_indptr: torch.Tensor,
    kv_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    num_heads: int,
    query_len: int,
    num_requests: int,
    rows_per_seq: int,
    sm_scale: float,
    v_page_offset: int,
    helix_kv_bounds: torch.Tensor | None,
    softmax_stats_tensor: torch.Tensor | None,
    plan_geometry: _PackedPlanGeometry | None = None,
) -> None:
    """Packed CuTe DSL decode: one kernel over (split, request) tasks.

    A single split writes the output (and Helix stats) directly; several
    splits write FP32 partials that one reduction kernel folds into the
    output and stats. Nothing pads, replicates, or copies Q or the output.
    """
    valid_m, physical_m = module.packed_query_rows(num_heads, query_len)
    if physical_m != rows_per_seq:
        raise RuntimeError(
            f"FP4 MLA packed Q rows {rows_per_seq} do not match the kernel geometry {physical_m}."
        )
    if v_packed is None:
        raise RuntimeError("FP4 MLA packed decode requires the persistent packed-V cache.")
    if not output.is_contiguous() or output.numel() != num_requests * valid_m * 512:
        raise RuntimeError(
            "FP4 MLA packed decode requires a contiguous [tokens, heads, 512] output."
        )
    device = output.device
    if plan_geometry is None:
        plan_geometry = _fp4_mla_packed_plan_geometry(
            metadata,
            num_heads=num_heads,
            query_len=query_len,
            num_requests=num_requests,
            device=device,
            helix_kv_bounds=helix_kv_bounds,
        )
        if plan_geometry is None:
            raise RuntimeError("FP4 MLA packed decode requires generation cache pages.")
    splits = plan_geometry.splits
    output_rows = output.view(num_requests * valid_m, 512)
    stats_rows = None
    if softmax_stats_tensor is not None:
        stats_rows = softmax_stats_tensor.view(num_requests * valid_m, 2)
    common = dict(
        max_kv_len=plan_geometry.split_kv_len,
        sm_scale=sm_scale,
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        num_splits=splits,
        q_global_scale=q_global_scale,
        v_page_offset=v_page_offset,
        helix_kv_bounds=helix_kv_bounds,
    )
    if splits == 1:
        module.run_trtllm_fp4_mla_decode_packed(
            q_fp4,
            q_sf,
            kv_cache,
            sf_cache,
            v_packed,
            v_sf,
            kv_global_scale,
            src_page_ids,
            page_indptr,
            kv_lens,
            output_rows,
            softmax_stats=stats_rows,
            **common,
        )
        return
    from .fp4_mla_packed_reduce import reduce_packed_splits

    tasks = splits * num_requests
    task_capacity = plan_geometry.task_capacity
    plan = plan_geometry.plan
    # The first FP4 MLA kernel of the step wrote the plan fused with the
    # generation lengths; a stale key means this call plans on other bounds.
    if metadata.fp4_mla_state.packed_plan_key != plan_geometry.key:
        _launch_fp4_mla_packed_plan(plan_geometry, kv_lens, metadata)
    partial = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_packed_split_partial",
        (task_capacity, physical_m, 512),
        dtype=torch.float32,
        device=device,
    )[:tasks]
    maxima = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_packed_split_maxima",
        (task_capacity, physical_m),
        dtype=torch.float32,
        device=device,
    )[:tasks]
    sums = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_packed_split_sums",
        (task_capacity, physical_m),
        dtype=torch.float32,
        device=device,
    )[:tasks]
    module.run_trtllm_fp4_mla_decode_packed(
        q_fp4,
        q_sf,
        kv_cache,
        sf_cache,
        v_packed,
        v_sf,
        kv_global_scale,
        src_page_ids,
        page_indptr,
        kv_lens,
        output_rows,
        split_kv_buffers=(partial, maxima, sums),
        plan=plan,
        **common,
    )
    reduce_packed_splits(
        partial,
        maxima,
        sums,
        output_rows,
        q_global_scale,
        kv_global_scale,
        num_requests=num_requests,
        num_splits=splits,
        valid_m=valid_m,
        physical_m=physical_m,
        sm_scale=sm_scale,
        plan=plan,
        output_scale=1.0 / module.FP4_MLA_P_GLOBAL_SCALE,
        stats=stats_rows,
        sm_count=_get_sm_count(device),
    )


def run_fp4_mla_attention_decode(
    metadata: Any,
    layer_idx: int,
    local_layer: int,
    q: torch.Tensor,
    output: torch.Tensor,
    *,
    sm_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    prequantized_q: torch.Tensor,
    prequantized_q_sf: torch.Tensor,
    q_batch_capacity: int,
    softmax_stats_tensor: torch.Tensor | None = None,
) -> None:
    """Run MLA decode with FP4 QK and FP4 PV tensor-core matmuls.

    Q is supplied in its assembled ``[latent, RoPE]`` layout and quantized to
    FP4 directly. QK reads ``[KV-nope, K-RoPE, K-RoPE-residual]`` contiguously
    from the primary cache with swizzled block scales. Softmax probabilities
    are quantized to FP4 per page, and PV repacks V nibbles from the shared KV
    cache while reading the auxiliary V-view scale pool. No BF16 dequantized
    KV workspace is materialized on this path. Callers must supply the packed Q
    and scales produced by the fused generation cache update.
    """
    head_dim = kv_lora_rank + qk_rope_head_dim
    if qk_rope_head_dim != FP4_MLA_K_RESIDUAL_DIM:
        raise ValueError(
            "FP4 MLA K residual attention requires "
            f"qk_rope_head_dim={FP4_MLA_K_RESIDUAL_DIM}, got {qk_rope_head_dim}."
        )
    _validate_fp4_mla_cache_shape(metadata.page_size, head_dim)
    if metadata.page_size != FP4_MLA_TOKENS_PER_BLOCK:
        raise ValueError(
            f"FP4 MLA attention decode requires page_size={FP4_MLA_TOKENS_PER_BLOCK}, "
            f"got {metadata.page_size}."
        )

    if q.ndim != 3 or q.shape[-1] != head_dim:
        raise ValueError(
            "FP4 MLA attention Q must have shape "
            f"[tokens, heads, {head_dim}], got {tuple(q.shape)}."
        )
    if not q.is_contiguous():
        raise ValueError("FP4 MLA attention Q must be contiguous.")

    num_queries = q.shape[0]
    if num_queries == 0:
        raise ValueError("FP4 MLA attention decode requires at least one query token.")
    num_gen_seqs = metadata.num_seqs - metadata.num_contexts
    query_len_per_seq = _get_linear_mtp_query_len_per_seq(
        metadata,
        num_queries=num_queries,
        num_gen_seqs=num_gen_seqs,
    )

    num_heads = q.shape[1]
    if output.shape[:2] != (num_queries, num_heads):
        raise ValueError("FP4 MLA attention output batch dimensions do not match.")

    backend = _fp4_mla_attention_backend()
    helix_spec_tokens_valid = bool(getattr(metadata, "_helix_spec_tokens_valid", False))
    helix_kv_bounds = None
    if softmax_stats_tensor is not None:
        if backend != _FP4_MLA_CUTEDSL_BACKEND:
            raise NotImplementedError(
                "FP4 MLA Helix softmax stats require the cutedsl attention backend."
            )
        if query_len_per_seq != 1 and not helix_spec_tokens_valid:
            raise NotImplementedError(
                "FP4 MLA multi-token Helix requires speculative per-token metadata."
            )
        expected_stats_shape = (num_queries, num_heads, 2)
        if (
            softmax_stats_tensor.shape != expected_stats_shape
            or softmax_stats_tensor.dtype != torch.float32
            or softmax_stats_tensor.device != q.device
            or not softmax_stats_tensor.is_contiguous()
        ):
            raise ValueError(
                "FP4 MLA Helix requires contiguous same-device float32 softmax "
                f"stats with shape {expected_stats_shape}."
            )
        if helix_spec_tokens_valid:
            helix_kv_bounds = getattr(metadata, "helix_kv_bounds", None)
            if (
                not isinstance(helix_kv_bounds, torch.Tensor)
                or helix_kv_bounds.dtype != torch.int32
                or helix_kv_bounds.device != q.device
                or helix_kv_bounds.ndim != 1
                or helix_kv_bounds.numel() < num_queries
                or not helix_kv_bounds.is_contiguous()
            ):
                raise ValueError(
                    "FP4 MLA speculative Helix KV bounds must be a contiguous "
                    "same-device int32 tensor covering every query token."
                )
            helix_kv_bounds = helix_kv_bounds[:num_queries]
    if getattr(metadata.fp4_mla_state, "v_scale_pool", None) is None:
        raise RuntimeError(
            "FP4 MLA attention decode requires the auxiliary V scale pool to be allocated."
        )

    global_scale = _get_fp4_mla_global_scale(metadata, q.device)
    q_residual_dim = FP4_MLA_Q_RESIDUAL_DIM
    _validate_fp4_mla_attention_q_shape(head_dim, q_residual_dim)

    if prequantized_q is None or prequantized_q_sf is None or q_batch_capacity is None:
        raise RuntimeError(
            "FP4 MLA decode requires Q prequantized by the fused generation cache update."
        )

    capacity = int(q_batch_capacity)
    # The fused generation update records the Q layout it produced; metadata
    # that never went through it (contract tests) carries the legacy layout.
    recorded_rows_per_seq = getattr(
        getattr(metadata, "fp4_mla_state", None), "prequantized_q_rows_per_seq", None
    )
    packed_rows_per_seq = int(recorded_rows_per_seq or 0)
    if recorded_rows_per_seq is not None and packed_rows_per_seq != _fp4_mla_packed_rows_per_seq(
        num_heads, query_len_per_seq
    ):
        raise RuntimeError(
            "FP4 MLA prequantized Q layout does not match the packed decode "
            f"configuration (rows per request {packed_rows_per_seq})."
        )
    if packed_rows_per_seq:
        active_q_rows = num_gen_seqs * packed_rows_per_seq
    else:
        active_q_rows = num_queries * num_heads
    active_q_sf_bytes = _get_fp4_mla_swizzled_scale_size(
        active_q_rows,
        FP4_MLA_Q_LOGICAL_DIM,
    )
    if packed_rows_per_seq:
        q_contract_ok = (
            prequantized_q.ndim == 2
            and prequantized_q.shape[0] >= active_q_rows
            and prequantized_q.shape[1] == FP4_MLA_Q_PACKED_DIM
            and prequantized_q_sf.numel() >= active_q_sf_bytes
        )
    else:
        expected_q_shape = (capacity * num_heads, FP4_MLA_Q_PACKED_DIM)
        expected_q_sf_shape = (
            _get_fp4_mla_swizzled_scale_size(
                capacity * num_heads,
                FP4_MLA_Q_LOGICAL_DIM,
            ),
        )
        q_contract_ok = (
            tuple(prequantized_q.shape) == expected_q_shape
            and tuple(prequantized_q_sf.shape) == expected_q_sf_shape
        )
    if (
        q.dtype != torch.bfloat16
        or capacity <= 0
        or num_queries > capacity
        or not q_contract_ok
        or prequantized_q.dtype != torch.uint8
        or not prequantized_q.is_contiguous()
        or prequantized_q_sf.dtype != torch.float8_e4m3fn
        or not prequantized_q_sf.is_contiguous()
    ):
        raise ValueError("FP4 MLA prequantized Q does not satisfy the fused-Q contract.")
    q_fp4 = prequantized_q[:active_q_rows]
    q_sf = prequantized_q_sf[:active_q_sf_bytes]
    q_global_scale = _get_fp4_mla_q_global_scale(metadata, q.device)

    kv_cache, sf_cache = _get_fp4_mla_kv_cache_tensors(metadata, layer_idx)
    _validate_fp4_mla_kv_storage_shape(
        kv_cache,
        sf_cache,
        head_dim=head_dim,
        backend=backend,
    )
    sf_cache = sf_cache.view(torch.float8_e4m3fn)

    v_sf_pool = metadata.fp4_mla_state.v_scale_pool
    v_sf = get_fp4_mla_v_scale_pool_view(metadata, v_head_dim=kv_lora_rank)[local_layer].view(
        torch.float8_e4m3fn
    )
    # The kv_lens runtime alias can lag at the decode anchor (seq_lens == 1) under
    # CUDA graph / one-engine MTP; recover the true total per sequence so the
    # per-query causal masking sees the full 1 + draft_len window (no-op when the
    # alias already matches).
    plan_geometry = _fp4_mla_packed_plan_geometry(
        metadata,
        num_heads=num_heads,
        query_len=query_len_per_seq,
        num_requests=num_gen_seqs,
        device=q.device,
        helix_kv_bounds=helix_kv_bounds,
    )
    kv_lens, _ = _fp4_mla_uniform_generation_lengths(
        metadata, num_queries, num_gen_seqs, plan_geometry=plan_geometry
    )
    _materialize_fp4_mla_device_page_table_for_forward(metadata, kv_lens)
    src_page_ids = _fp4_mla_generation_page_ids(metadata, num_gen_seqs)
    max_pages = _max_generation_pages(metadata)
    if max_pages == 0:
        raise RuntimeError("FP4 MLA attention decode requires generation cache pages.")
    if backend == _FP4_MLA_CUTEDSL_BACKEND:
        if get_sm_version() != 107:
            raise RuntimeError(
                "FP4 MLA cutedsl attention backend requires Rubin SM107; "
                f"current architecture is SM{get_sm_version()}."
            )
        if not _cutedsl_backend_available():
            raise RuntimeError(
                "FP4 MLA cutedsl attention backend requires a CuTe DSL build "
                "that provides the tcgen05 descriptor K-segment offset, and "
                "cuda-python."
            )
        if not 0 < num_heads <= 128 or kv_lora_rank != 512:
            raise ValueError(
                "FP4 MLA cutedsl attention requires 1-128 local heads and "
                f"kv_lora_rank=512, got num_heads={num_heads}, "
                f"kv_lora_rank={kv_lora_rank}."
            )

        cutedsl_kernel = _fp4_mla_cutedsl_kernel_module()
        QK_LOGICAL_DIM = cutedsl_kernel.QK_LOGICAL_DIM
        QK_SF_GROUPS = cutedsl_kernel.QK_SF_GROUPS
        SMEM_P4_V_N_PER_CTA = cutedsl_kernel.SMEM_P4_V_N_PER_CTA
        run_trtllm_fp4_mla_decode_page_native_from_raw = (
            cutedsl_kernel.run_trtllm_fp4_mla_decode_page_native_from_raw
        )

        # Keep general cases on the original path. This specialization uses
        # the prepacked-V mufu16 core and linear six-query MTP only.
        fold_queries = (
            not packed_rows_per_seq
            and num_heads == 6
            and query_len_per_seq == 6
            and num_gen_seqs == 1
            and helix_kv_bounds is None
            and softmax_stats_tensor is None
            and getattr(cutedsl_kernel, "SUPPORTS_FOLDED_QUERIES", False)
            and not _fp4_mla_cutedsl_fused_v_transpose_enabled()
            and getattr(metadata, "_fp4_mla_fold_q", True)
            and os.environ.get("TRTLLM_FP4_MLA_FOLD_Q", "1") != "0"
        )
        if fold_queries:
            from .fp4_mla_splitkv import select_split_count

            fold_queries = select_split_count(max_pages * metadata.page_size) > 1
        physical_heads = 128
        kernel_q = q_fp4
        kernel_q_sf = q_sf
        if not packed_rows_per_seq and num_heads < physical_heads and not fold_queries:
            kernel_q_storage, kernel_q_sf_storage, q_batch_capacity = _prepare_fp4_mla_q_buffers(
                metadata,
                num_queries,
                physical_heads,
                q.device,
            )
            active_q_rows = num_queries * physical_heads
            active_q_sf_bytes = _get_fp4_mla_swizzled_scale_size(
                active_q_rows,
                QK_LOGICAL_DIM,
            )
            kernel_q = kernel_q_storage[:active_q_rows]
            kernel_q_sf = kernel_q_sf_storage[:active_q_sf_bytes]
            _cutedsl_pad_q_and_sf_kernel[(num_queries, _ceil_div(QK_LOGICAL_DIM // 2, 64))](
                kernel_q,
                q_fp4,
                kernel_q_sf,
                q_sf,
                num_heads,
                output_heads=physical_heads,
                packed_dim=QK_LOGICAL_DIM // 2,
                block_bytes=64,
                sf_cols=QK_SF_GROUPS,
                sf_cols_per_byte_block=8,
            )

        fused_v_transpose = _fp4_mla_cutedsl_fused_v_transpose_enabled()
        if fused_v_transpose:
            # The fusion kernel reads V from the canonical KV cache and uses
            # the current layer V scales directly. Keep a None placeholder so
            # the mufu16 and fused-V launchers share one Python call site.
            core_v_packed = None
            core_v_sf = v_sf
            v_page_offset = 0
        else:
            v_packed = _get_cutedsl_persistent_v_packed_cache(
                metadata,
                local_layer,
                kv_cache,
                v_head_dim=kv_lora_rank,
                page_size=metadata.page_size,
                block_v=SMEM_P4_V_N_PER_CTA,
            )
            core_v_packed = _get_fp4_mla_v_packed_pool_base(metadata)
            if core_v_packed is None:
                raise RuntimeError("Persistent FP4 MLA V packing requires a stable full-pool base.")
            get_v_page_offset = getattr(
                metadata.kv_cache_manager, "get_mla_v_packed_page_offset", None
            )
            v_page_offset = (
                int(get_v_page_offset(local_layer))
                if callable(get_v_page_offset)
                else local_layer * kv_cache.shape[0]
            )
            page_bytes = kv_lora_rank * (metadata.page_size // 2)
            expected_layer_ptr = core_v_packed.data_ptr() + v_page_offset * page_bytes
            if expected_layer_ptr != v_packed.data_ptr():
                raise RuntimeError(
                    "Persistent FP4 MLA V-packed layer view does not match its "
                    "full-pool base and page offset."
                )

            v_sf_pool_base = _get_fp4_mla_v_scale_pool_base(metadata)
            if v_sf_pool_base is None:
                v_sf_pool_base = v_sf_pool.flatten(0, 1).view(torch.uint8)
                if v_sf_pool_base.data_ptr() != v_sf_pool.data_ptr():
                    raise RuntimeError(
                        "Persistent FP4 MLA V-scale pool must flatten without a copy."
                    )
            if (
                not isinstance(v_sf_pool_base, torch.Tensor)
                or v_sf_pool_base.dtype != torch.uint8
                or v_sf_pool_base.device != v_sf.device
                or v_sf_pool_base.ndim != 2
                or v_sf_pool_base.shape[1] != v_sf_pool.shape[-1]
                or not v_sf_pool_base.is_contiguous()
            ):
                raise RuntimeError(
                    "Persistent FP4 MLA V-scale pool base must be a contiguous "
                    "two-dimensional uint8 tensor with the configured page stride."
                )
            core_v_sf = v_sf_pool_base.view(torch.float8_e4m3fn)
            get_v_sf_page_offset = getattr(
                metadata.kv_cache_manager, "get_mla_v_scale_page_offset", None
            )
            v_sf_page_offset = (
                int(get_v_sf_page_offset(local_layer))
                if callable(get_v_sf_page_offset)
                else local_layer * kv_cache.shape[0]
            )
            if v_sf_page_offset != v_page_offset:
                raise RuntimeError(
                    "Persistent FP4 MLA V-packed and V-scale pools require "
                    "matching encoded layer offsets."
                )
            expected_v_sf_ptr = (
                core_v_sf.data_ptr()
                + v_sf_page_offset * v_sf_pool.stride(1) * v_sf_pool.element_size()
            )
            if expected_v_sf_ptr != v_sf.data_ptr():
                raise RuntimeError(
                    "Persistent FP4 MLA V-scale layer view does not match its "
                    "full-pool base and page offset."
                )

        if packed_rows_per_seq:
            _run_cutedsl_packed_decode(
                cutedsl_kernel,
                metadata,
                q_fp4,
                q_sf,
                kv_cache,
                sf_cache,
                core_v_packed,
                core_v_sf,
                global_scale,
                q_global_scale,
                src_page_ids,
                metadata.fp4_mla_state.paged_kv_indptr_decode[: num_gen_seqs + 1],
                kv_lens,
                output,
                num_heads=num_heads,
                query_len=query_len_per_seq,
                num_requests=num_gen_seqs,
                rows_per_seq=packed_rows_per_seq,
                sm_scale=float(sm_scale),
                v_page_offset=v_page_offset,
                helix_kv_bounds=helix_kv_bounds,
                softmax_stats_tensor=softmax_stats_tensor,
                plan_geometry=plan_geometry,
            )
            return

        from .fp4_mla_splitkv import run_fp4_mla_splitkv, select_split_count

        if fold_queries:
            from .fp4_mla_foldq import run_fp4_mla_foldq

            run_fp4_mla_foldq(
                cutedsl_kernel,
                metadata,
                q_fp4,
                q_sf,
                kv_cache,
                sf_cache,
                core_v_packed,
                core_v_sf,
                global_scale,
                src_page_ids,
                metadata.fp4_mla_state.paged_kv_indptr_decode[: num_gen_seqs + 1],
                kv_lens,
                output,
                splits=select_split_count(max_pages * metadata.page_size),
                max_kv_len=max_pages * metadata.page_size,
                sm_scale=float(sm_scale),
                q_global_scale=q_global_scale,
                query_len_per_seq=query_len_per_seq,
                v_page_offset=v_page_offset,
            )
            return

        # Helix also needs per-query bounds and softmax stats from the native path.
        splits = (
            select_split_count(max_pages * metadata.page_size)
            if helix_kv_bounds is None and softmax_stats_tensor is None
            else 1
        )
        direct_split_output = splits > 1 and getattr(
            metadata, "_fp4_mla_splitkv_direct_output", True
        )
        kernel_output = output
        kernel_softmax_stats = None
        if softmax_stats_tensor is not None:
            kernel_softmax_stats = _ensure_workspace_tensor(
                metadata,
                "_fp4_mla_cutedsl_softmax_stats_buf",
                (2, num_queries, physical_heads),
                dtype=torch.float32,
                device=output.device,
            )
        if num_heads < physical_heads and not direct_split_output:
            kernel_output = _ensure_workspace_tensor(
                metadata,
                "_fp4_mla_cutedsl_output_buf",
                (q_batch_capacity, physical_heads, kv_lora_rank),
                dtype=output.dtype,
                device=output.device,
            )[:num_queries]

        if splits > 1:
            run_fp4_mla_splitkv(
                _fp4_mla_cutedsl_kernel_module(),
                metadata,
                kernel_q,
                kernel_q_sf,
                kv_cache,
                sf_cache,
                core_v_packed,
                core_v_sf,
                global_scale,
                src_page_ids,
                metadata.fp4_mla_state.paged_kv_indptr_decode[: num_gen_seqs + 1],
                kv_lens,
                kernel_output,
                splits=splits,
                max_kv_len=max_pages * metadata.page_size,
                sm_scale=float(sm_scale),
                q_global_scale=q_global_scale,
                query_len_per_seq=query_len_per_seq,
                v_page_offset=v_page_offset,
                q_batch_capacity=q_batch_capacity,
            )
        else:
            run_trtllm_fp4_mla_decode_page_native_from_raw(
                kernel_q,
                kernel_q_sf,
                kv_cache,
                sf_cache,
                core_v_packed,
                core_v_sf,
                global_scale,
                src_page_ids,
                metadata.fp4_mla_state.paged_kv_indptr_decode[: num_gen_seqs + 1],
                kv_lens,
                kernel_output,
                max_kv_len=max_pages * metadata.page_size,
                sm_scale=float(sm_scale),
                num_heads=physical_heads,
                q_global_scale=q_global_scale,
                page_size=metadata.page_size,
                query_len_per_seq=query_len_per_seq,
                v_page_offset=v_page_offset,
                q_batch_capacity=q_batch_capacity,
                partition_runtime_valid_k=bool(getattr(metadata, "is_cuda_graph", False)),
                softmax_row_max=(None if kernel_softmax_stats is None else kernel_softmax_stats[0]),
                softmax_row_sum=(None if kernel_softmax_stats is None else kernel_softmax_stats[1]),
                helix_kv_bounds=helix_kv_bounds,
            )
        if kernel_output is not output:
            output.copy_(kernel_output[:, :num_heads])
        if kernel_softmax_stats is not None:
            softmax_stats_tensor[..., 0].copy_(kernel_softmax_stats[0, :, :num_heads])
            softmax_stats_tensor[..., 1].copy_(kernel_softmax_stats[1, :, :num_heads])
        return

    total_p_rows = num_queries * max_pages * num_heads
    p_fp4 = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_p_buf",
        (max(total_p_rows, 1), metadata.page_size // 2),
        dtype=torch.uint8,
        device=q.device,
    )[:total_p_rows]
    p_sf = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_p_sf_buf",
        (max(_get_fp4_mla_swizzled_scale_size(total_p_rows, metadata.page_size), 1),),
        dtype=torch.float8_e4m3fn,
        device=q.device,
    )
    stats_shape = (num_queries, num_heads)
    max_scores = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_max_buf",
        stats_shape,
        dtype=torch.float32,
        device=q.device,
    )
    denom = _ensure_workspace_tensor(
        metadata,
        "_fp4_mla_attention_denom_buf",
        stats_shape,
        dtype=torch.float32,
        device=q.device,
    )

    if backend != "triton":
        raise ValueError(
            f"Unsupported FP4 MLA attention backend '{backend}'. "
            f"Set {FP4_MLA_ATTENTION_BACKEND_ENV} to 'triton' or "
            f"'{_FP4_MLA_CUTEDSL_BACKEND}'."
        )

    # Self-contained public-Triton path: TMA-loaded QK + fused page-stats pack,
    # reduce-stats, prob-scale, and PV with an optional prepacked V cache.
    _run_triton_attention_decode(
        metadata=metadata,
        layer_idx=layer_idx,
        local_layer=local_layer,
        q_fp4=q_fp4,
        q_sf=q_sf.contiguous().view(-1),
        kv_cache=kv_cache,
        sf_cache=sf_cache,
        v_sf=v_sf,
        global_scale=global_scale,
        src_page_ids=src_page_ids,
        kv_lens=kv_lens,
        p_fp4=p_fp4,
        p_sf=p_sf,
        max_scores=max_scores,
        denom=denom,
        output=output,
        num_queries=num_queries,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_lora_rank=kv_lora_rank,
        q_residual_dim=q_residual_dim,
        query_len_per_seq=query_len_per_seq,
        max_pages=max_pages,
        sm_scale=float(sm_scale),
        q_global_scale=q_global_scale,
    )
