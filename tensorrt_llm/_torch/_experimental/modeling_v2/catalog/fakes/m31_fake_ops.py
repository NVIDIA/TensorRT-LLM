# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake (meta) impls of the MiniMax-M3.1 C++ torch ops (cpp/tensorrt_llm/thop/minimaxM31/).

The ops exist only on builds that carry them, either in the main library or in an overlay library the M3.1 targets load
on first use, so each fake is registered only once its op exists: ``register()`` runs at import and again after a
target loads an overlay library. It is idempotent.
"""

import torch

_REGISTERED = set()


def _ctx_residual_rmsnorm_mxfp8_fake(
    inter, residual, weight, eps, write_norm, fp4_global_scale=None, enable_pdl=False
):
    """minimaxM31Sm103CtxNormMxfp8Op.cpp: (norm_out, MXFP8 rows, swizzled scale bytes, NVFP4 rows, linear NVFP4 scale
    bytes); inter is updated in place. Outputs that are off are empty 1-D tensors of inter's dtype."""
    hidden = inter.shape[-1]
    rows = inter.numel() // hidden
    empty = inter.new_empty((0,))
    norm_out = torch.empty_like(inter) if write_norm else empty
    mxfp8 = torch.empty_like(inter, dtype=torch.float8_e4m3fn)
    sf = inter.new_empty(
        ((rows + 127) // 128 * 128 * ((hidden // 32 + 3) // 4 * 4),), dtype=torch.uint8
    )
    if fp4_global_scale is None:
        return (norm_out, mxfp8, sf, empty, empty)
    fp4 = inter.new_empty((*inter.shape[:-1], hidden // 2), dtype=torch.uint8)
    fp4_sf = inter.new_empty((rows * (hidden // 16),), dtype=torch.uint8)
    return (norm_out, mxfp8, sf, fp4, fp4_sf)


def _ctx_residual_rmsnorm_mxfp8_f32_fake(
    inter, residual, weight, eps, fp4_global_scale=None, enable_pdl=False
):
    """The _f32 variant (TP2CTX_NORMQ_F32 / TP2AGG_ROUTER_F32): the base op's outputs with write_norm, plus the norm
    widened to float32."""
    outs = _ctx_residual_rmsnorm_mxfp8_fake(
        inter, residual, weight, eps, True, fp4_global_scale, enable_pdl
    )
    return (*outs, torch.empty_like(inter, dtype=torch.float32))


def _select_prefill_tile_fake(scores, n_valid_blocks, out, init_blocks, local_blocks):
    """minimaxM31Sm103SelectPrefillTileOp.cpp: writes the block table into out; returns nothing."""
    return None


def _sparse_decode_lc_fake(
    q,
    k,
    v,
    k_block_scale,
    v_block_scale,
    k_global_scale,
    v_global_scale,
    topk,
    block_table,
    seq_lens,
    out,
    q8,
    sf,
    num_sf_k_tiles,
    decode_query_len,
    sm_scale,
    linear_softmax,
    use_pdl,
    early_trigger,
    btsmem,
    sync3,
    one_div=False,
):
    """minimaxM31Tp4SparseDecodeLcOp.cpp (LC4; LongAttn a7154ae3fa): writes out (and q8 / sf when given) in place;
    returns nothing."""
    return None


def _sparse_decode_lc8_fake(
    q,
    k,
    v,
    k_block_scale,
    v_block_scale,
    k_global_scale,
    v_global_scale,
    topk,
    block_table,
    seq_lens,
    out,
    q8,
    sf,
    num_sf_k_tiles,
    decode_query_len,
    sm_scale,
    linear_softmax,
    use_pdl,
    prewait,
    one_div=False,
):
    """minimaxM31Tp8SparseDecodeLc8Op.cpp (LC8; LongAttn a7154ae3fa): writes out (and q8 / sf when given) in place;
    returns nothing."""
    return None


def _swiglu_oai_fake(x, out, pdl):
    """m3_dense_swiglu: writes the [rows, d] bf16 activation into out in place; returns nothing."""
    return None


def _swiglu_oai_vec_fake(x, pdl=True, nv=1):
    """minimaxM31SwigluOaiVecOp.cpp: the SwiGLU-OAI activation of bf16 x [rows, 2d] -> a new [rows, d]."""
    return x.new_empty((x.shape[0], x.shape[1] // 2))


def _moe_finalize_fake(fc2, weights, idx, shared=None, cfg=-1):
    """minimaxM31MoeFinalizeOp.cpp: the routed experts' rows fc2 [P, H] reduced per token by weights / idx
    [tokens, top_k] (+ shared) -> a new [tokens, H] of fc2's dtype."""
    return fc2.new_empty((weights.shape[0], fc2.shape[1]))


def _topp_renorm_det_v3_fake(
    probs, top_p, workspace, out, ctas_per_row, pdl, candidate_list, list_path=True, wide_hist=True
):
    """m31_topp_det v3: as minimax_m31_topp_renorm_det (in place into out, workspace updated); returns nothing.
    minimax_m31_topp_renorm_det_v3_ws_bytes takes no tensor and needs no fake."""
    return None


def _topp_renorm_det_fake(probs, top_p, workspace, out, ctas_per_row):
    """m31_topp_det (ST-01): writes the renormed [rows, d] fp32 probs into out in place (and updates the persistent
    workspace); returns nothing. minimax_m31_topp_renorm_det_ws_bytes takes no tensor and needs no fake."""
    return None


def _softmax_pathb_fake(logits, temperature, temperature_val, out):
    """Writes fp32 probabilities into out; no workspace and no return value."""
    return None


def _softmax_mc_fake(logits, temperature, temperature_val, partials, out, write_ctas):
    """m31_softmax_mc (ST-01-SM): writes the [rows, d] fp32 probs into out in place (and overwrites the partials
    workspace); returns nothing."""
    return None


def _markov_epilogue_fake(base, k, bias, out, prev, w1, emb_out, partials, counters, splits):
    """m31_markov_epilogue (S4FE): writes position k of out, prev and emb_out in place; returns nothing."""
    return None


def _zero_padding_rows_fake(logits, slot_ids, dummy_slot_row, rows_per_request):
    """m31_zero_padding_rows (TAIL16_G2): zeroes padding rows of logits in place; returns nothing."""
    return None


def _s1b_fold_fake(partials, bias, logits_out, ids, weights, route_scale, pdl):
    """m31_s1b_fold (S1B_FOLD): writes ids / weights (and logits_out) in place; returns nothing."""
    return None


def _scatter_rows_fake(dst, slot_ids, src):
    """Copies source rows into destination slots in place; returns nothing."""
    return None


def _dspark_advance_fake(
    slots,
    rows,
    num_accepted,
    ctx_len,
    counts,
    base_pages,
    tables,
    read_view,
    out,
    max_ctx,
    dummy_slot,
    page,
    block_size,
):
    """DSPARK_GLUE_CPP: updates ctx_len / read_view / out in place; returns nothing."""
    return None


def _dspark_indices_fake(
    num_ctx,
    batch_idx,
    query_positions,
    seqlens,
    batch_out,
    after,
    indptr,
    last,
    append,
    qflat,
    block,
    pages_per_slot,
    page_size,
):
    """DSPARK_GLUE_CPP: writes the seven int32 index outputs in place; returns nothing."""
    return None


def _markov_prev_embed_fake(prev, w1):
    """m31_markov_prev_embed (TAIL_BK3 B): bf16 [n, rank]."""
    return w1.new_empty((prev.shape[0], w1.shape[1]))


def _expand_block_sampling_fake(
    temps, top_ks, top_ps, min_ps, seeds, offsets, K, slot_offset, span
):
    """m31_expand_block_sampling (TAIL_BK3 C): six [n * K] tensors in the inputs' dtypes."""
    n = temps.shape[0] * K
    return [t.new_empty((n,)) for t in (temps, top_ks, top_ps, min_ps, seeds, offsets)]


def _chain_spec_sampling_mc_fake(
    draft_probs, draft_token_ids, target_probs, seed, offset, workspace, ctas_per_row
):
    """m31_chain_spec_mc (ST-01-REJ): (output_token_ids [batch, numDraft + 1], num_tokens [batch]), int32 as
    trtllm::chain_speculative_sampling; updates the persistent workspace in place.
    minimax_m31_chain_speculative_sampling_mc_ws_bytes takes no tensor and needs no fake."""
    batch_size, num_draft = draft_token_ids.shape
    return (
        draft_token_ids.new_empty((batch_size, num_draft + 1)),
        draft_token_ids.new_empty((batch_size,)),
    )


def _chain_spec_sampling_fake(draft_probs, draft_token_ids, target_probs, seed, offset):
    """trtllm::chain_speculative_sampling: (output_token_ids [batch, numDraft + 1], num_tokens [batch]), int32."""
    batch_size, num_draft = draft_token_ids.shape
    return (
        draft_token_ids.new_empty((batch_size, num_draft + 1)),
        draft_token_ids.new_empty((batch_size,)),
    )


def _chain_spec_sampling_mc_slots_fake(
    draft_buf,
    slot_ids,
    num_draft,
    draft_token_ids,
    target_probs,
    seed,
    offset,
    workspace,
    ctas_per_row,
):
    """m31_chain_spec_mc REJ_SLOTS: as minimax_m31_chain_speculative_sampling_mc, draft rows read by slot."""
    batch_size = draft_token_ids.shape[0]
    return (
        draft_token_ids.new_empty((batch_size, num_draft + 1)),
        draft_token_ids.new_empty((batch_size,)),
    )


def _fc_shard_gemm_fake(x, w, cfg):
    """m31_fc_shard: bf16 x [M, K] times w [N, K]^T -> a new [M, N]."""
    return x.new_empty((x.shape[0], w.shape[0]))


def _s2_moe_tail_fake(
    fc2,
    scales,
    perm,
    shared,
    residual,
    norm_weight,
    workspace,
    rank,
    nranks,
    eps,
    trigger_completion_at_end=True,
):
    """m31_s2_moe_tail: (norm_out, residual_out) like shared / residual, the MXFP8 activation [tokens, hidden] and
    its swizzled scale bytes (rows padded to 128, scale columns to 4)."""
    tokens, hidden = shared.shape
    sf_size = (tokens + 127) // 128 * 128 * ((hidden // 32 + 3) // 4 * 4)
    return (
        torch.empty_like(shared),
        torch.empty_like(residual),
        shared.new_empty((tokens, hidden), dtype=torch.float8_e4m3fn),
        shared.new_empty((sf_size,), dtype=torch.uint8),
    )


def _sparse_decode_lc_prepare_fake(anchor, kv_heads):
    """m3_sparse_decode_lc_prepare reserves the split workspace once, before graph capture; traced, it reserves
    nothing."""
    return 0


def _qkv_indexer_producer_rows_fake(
    packed,
    kv_data_cache,
    kv_scale_cache,
    index_k_cache,
    out_cache_loc,
    kv_quant_scale,
    num_heads_q,
    num_heads_kv,
    num_heads_index,
    head_dim,
    rotary_dim,
    eps,
    q_weight,
    k_weight,
    index_q_weight,
    index_k_weight,
    rotary_cos_sin,
    position_ids,
):
    """minimax_m31_nvfp4_qkv_indexer_producer_rows (TP2CTX_QKV_ROWS): FP8 q [T, Hq, 128] and index-q [T, Hidx, 128];
    the paged caches are written in place."""
    num_tokens = packed.shape[0]
    return (
        packed.new_empty((num_tokens, num_heads_q, head_dim), dtype=torch.float8_e4m3fn),
        packed.new_empty((num_tokens, num_heads_index, head_dim), dtype=torch.float8_e4m3fn),
    )


def _swiglu_oai_mxfp8_fake(x, q, sf, h):
    """minimax_m31_swiglu_oai_mxfp8 (TP2CTX_SWIGLU_MXFP8): writes the E4M3 activation, its swizzled UE8M0 scales and
    (optionally) the bf16 activation into the caller's buffers; returns nothing."""
    return None


def _mxfp8_cublaslt_gemm_fake(act, act_scale, weight, weight_scale, algo):
    """minimax_m31_mxfp8_cublaslt_gemm (TP2CTX_MXFP8_CUBLASLT / AGG_PREFILL_CUBLASLT): E4M3 act [M, K] times the E4M3
    weight [N, K]^T with UE8M0 block scales -> a new bf16 [M, N]."""
    return act.new_empty((act.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def _tp2ctx_peer_push_fake(src, dst):
    """minimax_m31_tp2ctx_peer_push (TP2CTX_SP): copies src into the peer's IPC slot at dst; returns nothing."""
    return None


def _bf16_to_fp32_fake(x):
    """minimax_m31_bf16_to_fp32 (TP2CTX_GATE_CAST): x.to(torch.float32), same shape."""
    return x.new_empty(x.shape, dtype=torch.float32)


# op name -> fake impl.
FAKES = {
    "minimax_m31_fc_shard_gemm": _fc_shard_gemm_fake,
    "minimax_m31_ctx_residual_rmsnorm_mxfp8": _ctx_residual_rmsnorm_mxfp8_fake,
    "minimax_m31_ctx_residual_rmsnorm_mxfp8_f32": _ctx_residual_rmsnorm_mxfp8_f32_fake,
    "minimax_m31_select_prefill_tile": _select_prefill_tile_fake,
    "minimax_m31_tp4_sparse_decode_lc": _sparse_decode_lc_fake,
    "minimax_m31_tp8_sparse_decode_lc8": _sparse_decode_lc8_fake,
    "minimax_m31_swiglu_oai": _swiglu_oai_fake,
    "minimax_m31_swiglu_oai_vec": _swiglu_oai_vec_fake,
    "minimax_m31_moe_finalize": _moe_finalize_fake,
    "minimax_m31_swiglu_oai_mxfp8": _swiglu_oai_mxfp8_fake,
    "minimax_m31_bf16_to_fp32": _bf16_to_fp32_fake,
    "minimax_m31_mxfp8_cublaslt_gemm": _mxfp8_cublaslt_gemm_fake,
    "minimax_m31_tp2ctx_peer_push": _tp2ctx_peer_push_fake,
    "minimax_m31_nvfp4_qkv_indexer_producer_rows": _qkv_indexer_producer_rows_fake,
    "minimax_m31_topp_renorm_det": _topp_renorm_det_fake,
    "minimax_m31_topp_renorm_det_v3": _topp_renorm_det_v3_fake,
    "minimax_m31_softmax_mc": _softmax_mc_fake,
    "minimax_m31_softmax_pathb": _softmax_pathb_fake,
    "minimax_m31_markov_epilogue": _markov_epilogue_fake,
    "minimax_m31_zero_padding_rows": _zero_padding_rows_fake,
    "minimax_m31_scatter_rows": _scatter_rows_fake,
    "minimax_m31_dspark_advance": _dspark_advance_fake,
    "minimax_m31_dspark_indices": _dspark_indices_fake,
    "minimax_m31_s1b_fold": _s1b_fold_fake,
    "minimax_m31_markov_prev_embed": _markov_prev_embed_fake,
    "minimax_m31_expand_block_sampling": _expand_block_sampling_fake,
    "chain_speculative_sampling": _chain_spec_sampling_fake,
    "minimax_m31_chain_speculative_sampling_mc": _chain_spec_sampling_mc_fake,
    "minimax_m31_chain_speculative_sampling_mc_slots": _chain_spec_sampling_mc_slots_fake,
    "minimax_m31_s2_moe_tail": _s2_moe_tail_fake,
    "minimax_m3_sm107_tp2_sparse_decode_lc_prepare": _sparse_decode_lc_prepare_fake,
}


def _has_fake(qualname):
    """True when the op already has a fake: a C++ Meta kernel or a torch.library fake (register_fake raises on both).
    simple_registry is private torch API (checked on the image's torch 2.14); torch.library has no public query."""
    if torch._C._dispatch_has_kernel_for_dispatch_key(qualname, "Meta"):
        return True
    from torch._library.simple_registry import singleton

    return singleton.find(qualname).fake_impl.kernel is not None


def register():
    """Register the fake of every M3.1 op this process has loaded and that has none yet (another module may have
    registered one first; that one is kept)."""
    for name, fake in FAKES.items():
        if name in _REGISTERED or not hasattr(torch.ops.trtllm, name):
            continue
        if not _has_fake(f"trtllm::{name}"):
            torch.library.register_fake(f"trtllm::{name}")(fake)
        _REGISTERED.add(name)


register()
