# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA attention calls over the engine's KV cache: context FMHA, CuTe DSL decode and the op fallback.

Every call takes the role's :class:`~.k3_mla_state.K3MlaAttnState` and the step's attention metadata, from
which it reads the pools, block offsets and lengths (the engine owns the KV cache manager), and launches:

* :func:`mla_context_attention`: the context FMHA of the attention op (catalog ``thop_attention``, MLA
  context phase) over explicit K / V, writing ``output``. With ``latent_cache`` the op also rotates (identity
  RoPE) and appends the chunk's latent rows to the paged cache; with ``full_mask`` it attends a cached-prefix
  partition (padding mask) and writes its softmax statistics to ``softmax_stats``. ``quant_q_buffer`` /
  ``quant_scale_qkv`` hand the op an already-quantized FP8 Q; the FP8 context FMHA reads FP8 K / V as given.
* :func:`mla_decode`: the absorbed decode over an FP8 or BF16 / FP16 latent cache. Where the CuTe DSL MLA
  decode serves the batch (:func:`cute_dsl_decode_applies`) it runs ``trtllm::cute_dsl_mla_decode_fp8_blackwell``
  on the quantized query the generation cache update wrote (FP8 cache), or
  ``trtllm::cute_dsl_mla_decode_fp16_blackwell`` on the query as is (a cache of the query's dtype), with the
  per-request split-KV plan of the step; otherwise the attention op's MLA generation phase.

These are the launches the attention op's FMHA libraries make for these calls on SM107 (the fallback library
for context and for a decode the CuTe DSL kernel does not take, the CuTe DSL MLA library for the decode),
with the same arguments; their tests compare them with that path.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    MLA_CONTEXT,
    MLA_GENERATION,
    thop_attention,
)

# The FP4 MLA KV-cache layout contract shared with the KV cache manager.
from tensorrt_llm._torch.attention.backends.fp4_mla.config import FP4_MLA_TOKENS_PER_BLOCK
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger

from .k3_mla_state import K3MlaAttnState

# ---------------------------------------------------------------------------------------------
# The attention op (catalog thop_attention).
# ---------------------------------------------------------------------------------------------


def _attention_window_size(state: K3MlaAttnState, md) -> int:
    """The manager's attention window of this layer, or the step's max sequence length."""
    manager = md.kv_cache_manager
    if manager is not None:
        window = manager.max_attention_window_vec[state.get_local_layer_idx(md)]
        if window is not None:
            return window
    return md.max_seq_len


def _op_args(state: K3MlaAttnState, md) -> dict:
    """Arguments of one attention-op call besides the phase inputs: layer configuration and batch state."""
    # MLA context attention takes separate Q / K / V, never the paged-context FMHA.
    md.use_paged_context_fmha = False
    state._ensure_rope_table_size(md.max_seq_len)
    local_layer_idx = state.get_local_layer_idx(md)
    return dict(
        workspace=md.effective_workspace,
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        host_total_kv_lens=md.host_total_kv_lens,
        context_lengths=md.prompt_lens_cuda_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        host_request_types=md.host_request_types_runtime,
        max_context_q_len_override=md.max_context_q_len_override,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        cache_indirection=md.cache_indirection,
        kv_scale_orig_quant=state.kv_scale_orig_quant,
        kv_scale_quant_orig=state.kv_scale_quant_orig,
        out_scale=None,
        rotary_inv_freq=state.rotary_inv_freq,
        rotary_cos_sin=state.rotary_cos_sin,
        block_ids_per_seq=md.block_ids_per_seq,
        attention_sinks=None,
        is_fused_qkv=False,
        update_kv_cache=True,
        predicted_tokens_per_seq=state.predicted_tokens_per_seq,
        local_layer_idx=local_layer_idx,
        num_heads=state.num_heads,
        num_kv_heads=state.num_kv_heads,
        head_size=state.head_dim,
        tokens_per_block=md.tokens_per_block,
        max_num_requests=md.max_num_requests,
        max_context_length=md.max_context_length,
        max_seq_len=md.max_seq_len,
        attention_window_size=_attention_window_size(state, md),
        beam_width=md.beam_width,
        quant_mode=int(state.quant_mode),
        q_scaling=state.q_scaling,
        position_embedding_type=int(state.position_embedding_type),
        rope_dim=state.rope_dim,
        rope_base=state.rope_base,
        rope_scale_type=0,
        rope_scale=1.0,
        rope_short_m_scale=1.0,
        rope_long_m_scale=1.0,
        rope_max_positions=state.rope_max_positions,
        rope_original_max_positions=state.rope_original_max_positions,
        use_paged_context_fmha=False,
        is_mla_enable=True,
        q_lora_rank=state.q_lora_rank,
        kv_lora_rank=state.kv_lora_rank,
        qk_nope_head_dim=state.qk_nope_head_dim,
        qk_rope_head_dim=state.qk_rope_head_dim,
        v_head_dim=state.v_head_dim,
        rope_append=state.rope_append,
        mrope_rotary_cos_sin=None,
        mrope_position_deltas=None,
        helix_position_offsets=md.helix_position_offsets,
        helix_is_inactive_rank=md.helix_is_inactive_rank,
        attention_chunk_size=None,
        is_spec_decoding_enabled=md.is_spec_decoding_enabled,
        is_spec_dec_tree=md.is_spec_dec_tree,
        sparse_kv_indices=None,
        sparse_kv_offsets=None,
        sparse_attn_indices=None,
        sparse_attn_offsets=None,
        sparse_attn_indices_block_size=0,
        num_sparse_topk=md.num_sparse_topk,
        skip_softmax_stat=state.skip_softmax_stat,
        flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
        flash_mla_num_splits=md.flash_mla_num_splits,
        num_contexts=md.num_contexts,
        num_ctx_tokens=md.num_ctx_tokens,
        trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
        spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
        max_num_sequences=md.max_num_sequences,
        force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
        skip_correction_threshold=state.skip_correction_threshold,
    )


_NO_SPEC_DECODING = dict(
    use_spec_decoding=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
)


def mla_context_attention(
    state: K3MlaAttnState,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    md,
    output: torch.Tensor,
    *,
    latent_cache: Optional[torch.Tensor] = None,
    full_mask: bool = False,
    softmax_stats: Optional[torch.Tensor] = None,
    chunked_prefill_buffer_batch_size: int = 1,
    quant_q_buffer: Optional[torch.Tensor] = None,
    quant_scale_qkv: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Context attention of the context rows of ``q`` (``[T, H * 192]``) over ``k`` (``[T_kv, H * 192]``)
    and ``v`` (``[T_kv, H * 128]``, any row stride) into ``output`` (``[T, H * 128]``); see the module
    docstring. Returns ``output``."""
    if quant_scale_qkv is not None and quant_q_buffer is None:
        raise ValueError("quant_scale_qkv requires quant_q_buffer")
    if md.is_cuda_graph:
        md.workspace_reclaimable = False
    mask_type = AttentionMaskType.padding if full_mask else AttentionMaskType.causal
    return thop_attention(
        q=q,
        k=k,
        v=v,
        output=output,
        output_sf=None,
        latent_cache=latent_cache,
        q_pe=None,
        mask_type=int(mask_type),
        attention_input_type=MLA_CONTEXT,
        chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size,
        softmax_stats_tensor=softmax_stats,
        cu_q_seqlens=md.cu_q_seqlens,
        cu_kv_seqlens=md.cu_kv_seqlens,
        quant_q_buffer=quant_q_buffer,
        quant_scale_qkv=quant_scale_qkv,
        **_NO_SPEC_DECODING,
        **_op_args(state, md),
    )


def _generation_spec_args(md) -> dict:
    """The speculative-decoding arguments of a generation phase."""
    if not (md.is_spec_decoding_enabled and md.use_spec_decoding):
        return dict(_NO_SPEC_DECODING)
    offsets = md.spec_decoding_position_offsets
    if offsets is not None and offsets.dim() == 1:
        if not md.is_sm_version_trtllm_gen_kernel(sm=get_sm_version()):
            query_len = md.spec_decoding_query_len
            if query_len <= 0:
                raise ValueError(
                    "1-D speculative position offsets require a positive query length."
                )
            offsets = offsets[: md.max_num_requests * query_len].view(
                md.max_num_requests, query_len
            )
        else:
            offsets = offsets.view(md.max_num_requests, -1)
    return dict(
        use_spec_decoding=True,
        spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
        spec_decoding_position_offsets=offsets,
        spec_decoding_packed_mask=md.spec_decoding_packed_mask,
        spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
        spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
        spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
    )


def mla_generation_attention(
    state: K3MlaAttnState,
    fused_q: torch.Tensor,
    md,
    output: torch.Tensor,
    *,
    latent_cache: torch.Tensor,
    q_pe: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    quant_q_buffer: Optional[torch.Tensor],
    softmax_stats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The attention op's MLA generation phase over the absorbed query ``fused_q`` (``[T, H * 576]``)
    into ``output`` (``[T, H * 512]``), after :func:`~.k3_mla_cache.mla_rope_generation`."""
    if md.is_cuda_graph:
        md.workspace_reclaimable = False
    return thop_attention(
        q=fused_q,
        k=None,
        v=None,
        output=output,
        output_sf=None,
        latent_cache=latent_cache,
        q_pe=q_pe,
        mask_type=int(AttentionMaskType.causal),
        attention_input_type=MLA_GENERATION,
        chunked_prefill_buffer_batch_size=1,
        softmax_stats_tensor=softmax_stats,
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        quant_q_buffer=quant_q_buffer,
        **_generation_spec_args(md),
        **_op_args(state, md),
    )


# ---------------------------------------------------------------------------------------------
# The CuTe DSL MLA decode.
# ---------------------------------------------------------------------------------------------

# Per-request split-KV plan for the CuTe DSL MLA decode kernels. The kernel grid
# carries the tactic's split_kv slots per request; the plan decides how many a
# request uses this step from the batch's KV lengths (the kernel clamps it to
# the grid and the slots beyond exit before their mainloop).
_SPLIT_PLAN_MAX_REQUESTS = 1024
# CuteDSLNVMlaDecodeBlackwellRunner._MAX_SPLIT_KV: the largest grid any tactic uses.
_SPLIT_PLAN_MAX_SPLIT = 32
# mma_qk_tiler N of every tactic: the KV tile a split is made of.
_SPLIT_PLAN_KV_TILE = 128
# Shortest piece worth its partial-output write and reduction read.
_SPLIT_PLAN_MIN_TILES = 4
# Fixed cost of one piece (cluster launch, Q load, pipeline fill, partial store
# and its reduction) in KV tiles, fitted on SM107 forced-split sweeps (H96,
# 14K-256K KV, B1-B16): with it the wave-quantized makespan picks the measured
# optimum within 5%.
_SPLIT_PLAN_PIECE_COST_TILES = 60
# ``assumed_align`` the decode runner compiles the page table and the KV
# lengths with; their runtime pointers must keep it.
_KERNEL_INPUT_ALIGN_BYTES = 16
# Page size of the fallback thop op's precompiled TRTLLM-Gen MLA generation
# kernels.
_FALLBACK_MLA_GENERATION_TOKENS_PER_BLOCK = 32


@triton.jit(
    do_not_specialize=["num_requests", "rows_per_request", "wave_clusters", "max_split"],
    do_not_specialize_on_alignment=[
        "num_requests",
        "rows_per_request",
        "wave_clusters",
        "max_split",
    ],
)
def _mla_split_kv_plan_kernel(
    cache_seqs_ptr,
    plan_ptr,
    num_requests,
    rows_per_request,
    wave_clusters,
    max_split,
    KV_TILE: tl.constexpr,
    MIN_TILES: tl.constexpr,
    PIECE_COST: tl.constexpr,
    MAX_PIECES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < num_requests
    kv_len = tl.load(cache_seqs_ptr + offsets, mask=mask, other=0)
    tiles = tl.maximum((kv_len + KV_TILE - 1) // KV_TILE, 1)
    tiles = tl.where(mask, tiles, 0)
    longest = tl.max(tiles, axis=0)
    # Cut the longest request into k equal pieces (every request then takes
    # ceil(tiles / piece) pieces) and keep the k whose wave-quantized makespan,
    # waves x (piece + fixed piece cost), is smallest; ties go to fewer pieces.
    best_piece = longest
    best_cost = tl.full([], 2**31 - 1, tl.int32)
    for pieces in tl.static_range(1, MAX_PIECES + 1):
        piece = tl.maximum((longest + pieces - 1) // pieces, MIN_TILES)
        splits = tl.minimum(tl.maximum((tiles + piece - 1) // piece, 1), max_split)
        clusters = tl.sum(tl.where(mask, splits, 0), axis=0) * rows_per_request
        waves = (clusters + wave_clusters - 1) // wave_clusters
        cost = waves * (piece + PIECE_COST)
        better = cost < best_cost
        best_cost = tl.where(better, cost, best_cost)
        best_piece = tl.where(better, piece, best_piece)
    plan = tl.minimum(tl.maximum((tiles + best_piece - 1) // best_piece, 1), max_split)
    tl.store(plan_ptr + offsets, plan, mask=mask)


def plan_mla_split_kv(
    kv_lens: list[int],
    *,
    rows_per_request: int,
    wave_clusters: int,
    max_split: int = _SPLIT_PLAN_MAX_SPLIT,
    kv_tile: int = _SPLIT_PLAN_KV_TILE,
    min_tiles: int = _SPLIT_PLAN_MIN_TILES,
    piece_cost: int = _SPLIT_PLAN_PIECE_COST_TILES,
) -> list[int]:
    """Host mirror of ``_mla_split_kv_plan_kernel``: per-request split counts.

    ``rows_per_request`` is the kernel's packed 128-row query tiles per
    request, each a 2-CTA cluster per split; ``wave_clusters`` the clusters one
    wave holds.
    The longest request is cut into ``k`` equal pieces (every request takes
    ``ceil(tiles / piece)`` of them) for the ``k`` with the smallest
    wave-quantized makespan ``waves * (piece + piece_cost)``.
    """
    num_requests = len(kv_lens)
    if num_requests <= 0 or rows_per_request <= 0 or wave_clusters <= 0 or max_split <= 0:
        raise ValueError("split plan needs requests, rows, wave clusters and a split cap")
    tiles = [max(math.ceil(max(kv, 0) / kv_tile), 1) for kv in kv_lens]
    longest = max(tiles)
    best_piece, best_cost = longest, None
    for pieces in range(1, max_split + 1):
        piece = max(math.ceil(longest / pieces), min_tiles)
        splits = [min(max(math.ceil(t / piece), 1), max_split) for t in tiles]
        waves = math.ceil(sum(splits) * rows_per_request / wave_clusters)
        cost = waves * (piece + piece_cost)
        if best_cost is None or cost < best_cost:
            best_piece, best_cost = piece, cost
    return [min(max(math.ceil(t / best_piece), 1), max_split) for t in tiles]


def launch_mla_split_kv_plan(
    cache_seqs: torch.Tensor,
    plan: torch.Tensor,
    *,
    num_requests: int,
    rows_per_request: int,
    wave_clusters: int,
    max_split: int = _SPLIT_PLAN_MAX_SPLIT,
) -> None:
    """Write the per-request split counts for ``cache_seqs`` into ``plan`` (one launch)."""
    if not 0 < num_requests <= _SPLIT_PLAN_MAX_REQUESTS:
        raise ValueError(f"split plan supports 1..{_SPLIT_PLAN_MAX_REQUESTS} requests")
    for name, tensor in (("cache_seqs", cache_seqs), ("plan", plan)):
        if (
            tensor.dtype != torch.int32
            or tensor.ndim != 1
            or tensor.stride(0) != 1
            or tensor.numel() < num_requests
            or not tensor.is_cuda
        ):
            raise ValueError(f"{name} must be a contiguous CUDA int32 vector of >= {num_requests}")
    _mla_split_kv_plan_kernel[(1,)](
        cache_seqs,
        plan,
        num_requests,
        max(rows_per_request, 1),
        max(wave_clusters, 1),
        max(max_split, 1),
        KV_TILE=_SPLIT_PLAN_KV_TILE,
        MIN_TILES=_SPLIT_PLAN_MIN_TILES,
        PIECE_COST=_SPLIT_PLAN_PIECE_COST_TILES,
        MAX_PIECES=_SPLIT_PLAN_MAX_SPLIT,
        BLOCK=_SPLIT_PLAN_MAX_REQUESTS,
        num_warps=4,
    )


def _cute_dsl_mla_available(state: K3MlaAttnState) -> bool:
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    return bool(
        IS_CUTLASS_DSL_AVAILABLE
        and get_sm_version() in (100, 103, 107)
        and state.predicted_tokens_per_seq >= 1
        and state.qk_nope_head_dim > 0
        and state.kv_lora_rank == 512
        and state.qk_rope_head_dim == 64
        and state.num_heads <= 128
    )


def _page_size_exclusive_to_cute_dsl(md) -> bool:
    """Whether 128-token FP4 MLA pages are read by no other MLA generation kernel of this build."""
    tokens_per_block = md.tokens_per_block
    if tokens_per_block != FP4_MLA_TOKENS_PER_BLOCK:
        return False
    manager = md.kv_cache_manager
    if tokens_per_block in getattr(manager, "trtllm_gen_extra_tokens_per_block", ()):
        return False
    from tensorrt_llm.bindings import DataType

    # The opt-in PrimTS library reads 128-token pages of FP8 / BF16 MLA caches on SM100 / SM103.
    return not (
        get_sm_version() in (100, 103)
        and getattr(manager, "dtype", None) in (DataType.FP8, DataType.BF16)
    )


# The CuTe DSL decode on a BF16 / FP16 cache wins over the op's decode at these head counts.
_PERF_FAVORABLE_HEADS_FP16 = (6, 12, 16, 96)
# On an FP8 cache it wins from these per-rank decode batch sizes, keyed by (num_heads, seq_len_q); shapes
# outside the table take the op unless the next rule admits them.
_PERF_MIN_BATCH_FP8 = {
    (6, 1): 1,
    (12, 1): 1,
    (16, 2): 64,
    (16, 4): 32,
    (16, 8): 16,
    (96, 1): 8,
    (128, 1): 64,
    (128, 2): 32,
}


def _kernel_dtype(state: K3MlaAttnState, fused_q: torch.Tensor) -> Optional[torch.dtype]:
    """The CuTe DSL decode's input dtype: FP8 on an FP8 cache, else the query's (BF16 / FP16)."""
    if state.has_fp8_kv_cache:
        return torch.float8_e4m3fn
    if fused_q.dtype in (torch.float16, torch.bfloat16):
        return fused_q.dtype
    return None


def cute_dsl_decode_applies(
    state: K3MlaAttnState,
    fused_q: torch.Tensor,
    md,
    output: torch.Tensor,
    softmax_stats: Optional[torch.Tensor],
) -> bool:
    """Whether the CuTe DSL MLA decode serves this generation batch (see the module docstring): a
    linear-chain, beam-1 decode of up to 128 heads, with Helix softmax statistics under Helix, admitted by
    its measured win rule or by having no other kernel, over an FP8 cache or a cache of the query's dtype,
    and accepted by the kernel's ``can_implement``."""
    if state.has_fp4_kv_cache or not _cute_dsl_mla_available(state):
        return False
    kernel_dtype = _kernel_dtype(state, fused_q)
    if kernel_dtype is None:
        return False
    if md.num_generations <= 0:
        return False
    seq_len_q = fused_q.shape[0] // md.num_generations
    batch_size = md.num_generations
    helix = md.helix_position_offsets is not None
    helix_verify_group = seq_len_q > 1 and helix
    helix_decode_without_fallback_kernel = (
        helix
        and 32 < state.num_heads < 64
        and md.tokens_per_block != _FALLBACK_MLA_GENERATION_TOKENS_PER_BLOCK
    )
    exclusive_page_size = _page_size_exclusive_to_cute_dsl(md)
    if md.num_contexts != 0 and not (
        exclusive_page_size or helix_verify_group or helix_decode_without_fallback_kernel
    ):
        return False
    if md.beam_width != 1:
        return False
    if getattr(md, "is_spec_dec_tree", False) or getattr(md, "is_spec_dec_dynamic_tree", False):
        return False
    if helix:
        if seq_len_q != 1 and not md._helix_spec_tokens_valid:
            return False
        if (
            softmax_stats is None
            or softmax_stats.shape != (fused_q.shape[0], state.num_heads, 2)
            or softmax_stats.dtype != torch.float32
            or not softmax_stats.is_contiguous()
        ):
            return False
    fp8_wide_head_verify_group = (
        64 < state.num_heads < 128
        and seq_len_q > 1
        and not helix
        and kernel_dtype == torch.float8_e4m3fn
    )
    if not (
        helix_verify_group
        or fp8_wide_head_verify_group
        or helix_decode_without_fallback_kernel
        or exclusive_page_size
    ):
        from tensorrt_llm._torch.autotuner import AutoTuner

        if kernel_dtype != torch.float8_e4m3fn:
            if state.num_heads not in _PERF_FAVORABLE_HEADS_FP16:
                return False
        else:
            min_batch = _PERF_MIN_BATCH_FP8.get((state.num_heads, seq_len_q))
            if min_batch is None:
                return False
            if not AutoTuner.get().is_tuning_mode and batch_size < min_batch:
                return False
    if md.kv_cache_manager is None or output is None:
        return False
    if (
        kernel_dtype != torch.float8_e4m3fn
        and md.kv_cache_manager.get_buffers(state.layer_idx).dtype != kernel_dtype
    ):
        return False
    return _kernel_can_implement(
        kernel_dtype,
        output.dtype,
        batch_size,
        seq_len_q,
        md.tokens_per_block,
        state.num_heads,
        state.kv_lora_rank,
        state.qk_rope_head_dim,
    )


def _kernel_can_implement(
    in_dtype, out_dtype, batch_size, seq_len_q, page_size, num_heads, kv_lora_rank, qk_rope_head_dim
) -> bool:
    """The MLA decode kernel's (FP8 or FP16 / BF16 input) ``can_implement`` under the tiler the op
    launches with."""
    import cutlass

    # The kernel classes the CuTe DSL MLA decode ops compile, as the op module exposes them.
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        BlackwellMultiHeadLatentAttentionForwardFP8,
        BlackwellMultiHeadLatentAttentionForwardFP16,
    )

    to_cute = {
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
    }
    cute_in_dtype, cute_out_dtype = to_cute.get(in_dtype), to_cute.get(out_dtype)
    if cute_in_dtype is None or cute_out_dtype is None:
        return False
    kernel_class = (
        BlackwellMultiHeadLatentAttentionForwardFP8
        if cute_in_dtype == cutlass.Float8E4M3FN
        else BlackwellMultiHeadLatentAttentionForwardFP16
    )
    return bool(
        kernel_class.can_implement(
            batch_size,
            seq_len_q,
            2,  # A fake K to bypass the K check.
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            cute_in_dtype,
            cute_out_dtype,
            cutlass.Float32,
            cutlass.Float32,
            (128, 128),
            (128, 256),
            1,
            False,
            True,
            False,
            page_size,
        )
    )


def _aligned_kernel_input(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor``, or a contiguous copy where its pointer breaks the kernel's 16-byte alignment."""
    if tensor.data_ptr() % _KERNEL_INPUT_ALIGN_BYTES == 0:
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _decode_kv_lengths(md, sequence_lengths: torch.Tensor) -> torch.Tensor:
    """This step's int32 KV lengths, 16-byte aligned (staged on the metadata, shared by every layer)."""
    lengths = sequence_lengths.to(torch.int32)
    if lengths.data_ptr() % _KERNEL_INPUT_ALIGN_BYTES == 0:
        return lengths
    num_lengths = lengths.numel()
    staging = getattr(md, "_cute_dsl_mla_kv_lens_staging", None)
    if staging is None or staging.numel() < num_lengths or staging.device != lengths.device:
        if lengths.is_cuda and torch.cuda.is_current_stream_capturing():
            return _aligned_kernel_input(lengths)
        capacity = max(int(getattr(md, "max_num_requests", 0) or 0), num_lengths)
        staging = torch.empty(capacity, dtype=torch.int32, device=lengths.device)
        md._cute_dsl_mla_kv_lens_staging = staging
    staged = staging[:num_lengths].view(lengths.shape)
    staged.copy_(lengths)
    return staged


def _split_kv_plan(
    md, cache_seqs: torch.Tensor, batch_size: int, seq_len_q: int, num_heads: int
) -> Optional[torch.Tensor]:
    """This step's per-request split-KV counts, planned by the first layer of the step (the metadata
    resets the key when the lengths change); None (uniform split) where it cannot be planned."""
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    if not 0 < batch_size <= _SPLIT_PLAN_MAX_REQUESTS:
        return None
    plan = getattr(md, "_cute_dsl_mla_split_plan", None)
    capacity = max(int(getattr(md, "max_num_requests", 0) or 0), batch_size)
    if plan is None or plan.numel() < batch_size or plan.device != cache_seqs.device:
        if torch.cuda.is_current_stream_capturing():
            return None
        plan = torch.empty(capacity, dtype=torch.int32, device=cache_seqs.device)
        md._cute_dsl_mla_split_plan = plan
    runner = CuteDSLNVMlaDecodeBlackwellRunner
    rows_per_request = runner.get_num_q_tiles(num_heads, seq_len_q)
    wave_clusters = max(runner._get_max_active_blocks() // runner._CLUSTER_SHAPE_MNK[0], 1)
    key = (batch_size, rows_per_request, wave_clusters, cache_seqs.data_ptr(), plan.data_ptr())
    if getattr(md, "_cute_dsl_mla_split_plan_key", None) != key:
        launch_mla_split_kv_plan(
            cache_seqs,
            plan,
            num_requests=batch_size,
            rows_per_request=rows_per_request,
            wave_clusters=wave_clusters,
        )
        md._cute_dsl_mla_split_plan_key = key
    return plan[:batch_size]


def _required_workspace_bytes(
    num_heads: int, seq_len_q: int, kv_lora_rank: int, max_num_requests: int
) -> int:
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    return CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
        num_heads, seq_len_q, kv_lora_rank, max_num_requests, cutlass.Float32
    )


def cute_dsl_decode(
    state: K3MlaAttnState,
    fused_q: torch.Tensor,
    md,
    output: torch.Tensor,
    *,
    quant_q_buffer: Optional[torch.Tensor],
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    softmax_stats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The CuTe DSL MLA decode over this layer's latent cache into ``output`` (``[T, H * 512]``): on an FP8
    cache ``trtllm::cute_dsl_mla_decode_fp8_blackwell`` on the quantized query ``quant_q_buffer``
    (``fused_q`` provides shapes) with the device BMM scales; on a cache of the query's dtype
    ``trtllm::cute_dsl_mla_decode_fp16_blackwell`` on ``fused_q`` with the softmax scale of the head
    dims."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
    from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

    kernel_dtype = _kernel_dtype(state, fused_q)
    fp8 = kernel_dtype == torch.float8_e4m3fn
    if fp8:
        for name, scale in (("mla_bmm1_scale", mla_bmm1_scale), ("mla_bmm2_scale", mla_bmm2_scale)):
            if scale is None or scale.dtype != torch.float32 or scale.numel() < 1:
                raise RuntimeError(f"{name} must contain at least one float32 value.")
        softmax_scale, output_scale = mla_bmm1_scale, mla_bmm2_scale
        op = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell
    else:
        softmax_scale = float(
            1.0 / (math.sqrt(state.qk_nope_head_dim + state.qk_rope_head_dim) * state.q_scaling)
        )
        output_scale = 1.0
        op = torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell
    state._ensure_rope_table_size(md.max_seq_len)
    local_layer_idx = state.get_local_layer_idx(md)
    # The decode keeps state across the workspace; it may not be reclaimed.
    md.workspace_reclaimable = False
    num_tokens = fused_q.shape[0]
    batch_size = md.num_generations // md.beam_width
    seq_len_q = num_tokens // batch_size
    d_latent, d_rope, num_heads = state.kv_lora_rank, state.qk_rope_head_dim, state.num_heads
    page_size = md.tokens_per_block if md.tokens_per_block is not None else 64

    workspace = md.effective_workspace
    required = _required_workspace_bytes(num_heads, seq_len_q, d_latent, md.max_num_requests)
    if workspace.numel() * workspace.element_size() < required:
        if md.is_cuda_graph and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Attention CUDA graph workspace is smaller than the required size for Cute DSL MLA "
                "decode."
            )
        workspace.resize_((math.ceil(required / workspace.element_size()),))

    logger.info_once(
        f"Kimi K3 MLA decode: layer {state.layer_idx} runs the CuTe DSL MLA decode "
        f"({'cute_dsl_mla_decode_fp8_blackwell' if fp8 else 'cute_dsl_mla_decode_fp16_blackwell'}).",
        key=("k3_mla_cute_dsl_decode", state.layer_idx),
    )
    if fp8 and quant_q_buffer is not None:
        q_kernel = quant_q_buffer.view(torch.float8_e4m3fn).view_as(fused_q)
    else:
        q_kernel = fused_q if fused_q.dtype == kernel_dtype else fused_q.to(kernel_dtype)
    q_view = q_kernel.view(batch_size, seq_len_q, num_heads, d_latent + d_rope)
    kv_pool = md.kv_cache_manager.get_buffers(state.layer_idx)
    packed_block = 1
    for size in kv_pool.shape[1:]:
        packed_block *= size
    block_stride = kv_pool.stride(0)
    layers_in_pool = block_stride // packed_block if packed_block else 1
    layer_in_pool = 0
    if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
        layer_in_pool = kv_pool.storage_offset() // packed_block
        kv_pool = kv_pool.as_strided(
            (kv_pool.shape[0] * layers_in_pool, *kv_pool.shape[1:]),
            (packed_block, *kv_pool.stride()[1:]),
            0,
        )
    else:
        layers_in_pool = 1
    kv_pool = kv_pool.view(kernel_dtype)
    if kv_pool.dim() != 5 or kv_pool.shape[1] != 1 or kv_pool.shape[3] != 1:
        raise RuntimeError(
            "CuTe DSL MLA decode expects KV cache layout "
            f"[num_pages, 1, page_size, 1, head_dim], got {tuple(kv_pool.shape)}."
        )
    if kv_pool.shape[2] != page_size or kv_pool.shape[-1] < d_latent + d_rope:
        raise RuntimeError(
            f"CuTe DSL MLA decode got incompatible KV cache shape {tuple(kv_pool.shape)}."
        )

    pool_idx = int(md.host_kv_cache_pool_mapping[local_layer_idx, 0])
    page_table_layer = md.kv_cache_block_offsets[pool_idx, :, 0, :]
    cache_seqs = _decode_kv_lengths(md, md.kv_lens_cuda_runtime[md.num_contexts :])
    rows = page_table_layer[md.num_contexts : md.num_contexts + batch_size].to(torch.int32)
    if layer_in_pool:
        rows = rows + layer_in_pool
    page_table = _aligned_kernel_input(rows).transpose(0, 1)
    manager = md.kv_cache_manager
    if (
        layers_in_pool == 1
        and isinstance(manager, KVCacheManagerV2)
        and manager._use_per_layer_page_tables
    ):
        layer_offset = int(
            manager.impl.get_page_index_converter(local_layer_idx, Role.KEY).layer_offset
        )
        if layer_offset:
            page_table = torch.where(
                page_table == BAD_PAGE_INDEX, page_table, page_table - layer_offset
            )

    kv_pages = kv_pool[:, 0, :, 0, : d_latent + d_rope]
    c_pool_latent = kv_pages[..., :d_latent].permute(1, 2, 0)
    c_pool_rope = kv_pages[..., d_latent:].permute(1, 2, 0)
    workspace = workspace[: math.ceil(required / workspace.element_size())]
    block_split_kvs = _split_kv_plan(md, cache_seqs, batch_size, seq_len_q, num_heads)
    output_view = output.view(batch_size, seq_len_q, num_heads, d_latent)
    op(
        q_view[..., :d_latent].permute(2, 3, 1, 0),
        q_view[..., d_latent:].permute(2, 3, 1, 0),
        c_pool_latent,
        c_pool_rope,
        page_table,
        cache_seqs,
        output_view.permute(2, 3, 1, 0),
        workspace,
        num_heads,
        seq_len_q,
        page_size,
        softmax_scale,
        output_scale,
        int(md.max_num_requests),
        softmax_stats,
        (
            md.helix_kv_bounds[:num_tokens]
            if (md.helix_position_offsets is not None and md._helix_spec_tokens_valid)
            else None
        ),
        block_split_kvs,
    )
    return output


def mla_decode(
    state: K3MlaAttnState,
    fused_q: torch.Tensor,
    md,
    output: torch.Tensor,
    *,
    latent_cache: torch.Tensor,
    q_pe: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: Optional[torch.Tensor],
    mla_bmm2_scale: Optional[torch.Tensor],
    quant_q_buffer: Optional[torch.Tensor],
    softmax_stats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The absorbed decode of an FP8- or BF16 / FP16-cache layer (see the module docstring)."""
    if cute_dsl_decode_applies(state, fused_q, md, output, softmax_stats):
        return cute_dsl_decode(
            state,
            fused_q,
            md,
            output,
            quant_q_buffer=quant_q_buffer,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            softmax_stats=softmax_stats,
        )
    logger.info_once(
        f"Kimi K3 MLA decode: layer {state.layer_idx} batch outside the CuTe DSL MLA decode runs "
        "the attention op's MLA generation.",
        key=("k3_mla_decode_op", state.layer_idx),
    )
    return mla_generation_attention(
        state,
        fused_q,
        md,
        output,
        latent_cache=latent_cache,
        q_pe=q_pe,
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        quant_q_buffer=quant_q_buffer,
        softmax_stats=softmax_stats,
    )


__all__ = [
    "cute_dsl_decode",
    "cute_dsl_decode_applies",
    "launch_mla_split_kv_plan",
    "mla_context_attention",
    "mla_decode",
    "mla_generation_attention",
]
