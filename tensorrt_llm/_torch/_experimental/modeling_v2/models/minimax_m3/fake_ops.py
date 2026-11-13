# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register MiniMax-M3 fake ops.

The trtllm:: decode ops register through their shared kernel module. The family's own native ops (csrc/, namespace
mv2_minimax_m3) register below; the loader imports this module when the family library opens, so a fake for an op the
library does not define fails the load.
"""

import torch

from ..._native import namespace_of
from .kernels import fake_ops  # noqa: F401

_NS = namespace_of(__name__)


def _returns_none(*args, **kwargs):
    return None


def _in_place(name: str) -> None:
    """An op that only writes its tensor arguments in place returns nothing."""
    torch.library.register_fake(f"{_NS}::{name}")(_returns_none)


def _host_query(name: str) -> None:
    """Ops that answer a host-side question at setup (workspace sizes, launch shapes); tracing never runs them."""

    def fake(*args, **kwargs):
        raise NotImplementedError(f"{_NS}::{name} is a setup-time query; call it outside tracing")

    torch.library.register_fake(f"{_NS}::{name}")(fake)


def _ctx_residual_rmsnorm_mxfp8_fake(
    inter, residual, weight, eps, write_norm, fp4_global_scale=None, enable_pdl=False, lb2=True
):
    """sm103CtxNormMxfp8Op.cpp: (norm_out, MXFP8 rows, swizzled scale bytes, NVFP4 rows, linear NVFP4 scale
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
    inter, residual, weight, eps, fp4_global_scale=None, enable_pdl=False, lb2=True
):
    """The _f32 variant (TP2CTX_NORMQ_F32 / TP2AGG_ROUTER_F32): the base op's outputs with write_norm, plus the norm
    widened to float32."""
    outs = _ctx_residual_rmsnorm_mxfp8_fake(
        inter, residual, weight, eps, True, fp4_global_scale, enable_pdl
    )
    return (*outs, torch.empty_like(inter, dtype=torch.float32))


def _select_prefill_tile_fake(scores, n_valid_blocks, out, init_blocks, local_blocks, narrow=True):
    """sm103SelectPrefillTileOp.cpp: writes the block table into out; returns nothing."""
    return None


def _select_prefill_tile_rows_fake(
    scores, n_valid_blocks, out, init_blocks, local_blocks, narrow=True
):
    """sm103SelectPrefillTileOp.cpp (row-major scores, DEP2CTX_TSEL_ROWS): writes the block table into out;
    returns nothing."""
    return None


def _swiglu_oai_vec_fake(x, pdl=True, nv=1):
    """swigluOaiVecOp.cpp: the SwiGLU-OAI activation of bf16 x [rows, 2d] -> a new [rows, d]."""
    return x.new_empty((x.shape[0], x.shape[1] // 2))


def _moe_finalize_fake(fc2, weights, idx, shared=None, cfg=-1):
    """moeFinalizeOp.cpp: the routed experts' rows fc2 [P, H] reduced per token by weights / idx
    [tokens, top_k] (+ shared) -> a new [tokens, H] of fc2's dtype."""
    return fc2.new_empty((weights.shape[0], fc2.shape[1]))


def _topp_renorm_det_v3_fake(
    probs,
    top_p,
    workspace,
    out,
    ctas_per_row,
    pdl,
    candidate_list,
    list_path=True,
    wide_hist=True,
    keep_all_ge1=True,
    chain_prefetch=True,
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
    """m31_markov_epilogue (the Markov chain epilogue fold): writes position k of out, prev and emb_out in place;
    returns nothing."""
    return None


def _zero_padding_rows_fake(logits, slot_ids, dummy_slot_row, rows_per_request):
    """m31_zero_padding_rows (TAIL16_G2): zeroes padding rows of logits in place; returns nothing."""
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


def _dspark_advance_par_fake(
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
    enable_pdl=True,
):
    """DSPARK_GLUE_CPP (ADV_PAR): updates ctx_len / read_view / out in place; returns nothing."""
    return None


def _dspark_qk_norm_rope_fake(
    qkv,
    positions,
    cos_sin_cache,
    q_weight,
    k_weight,
    q_out,
    k_out,
    eps_q,
    eps_k,
    num_q_heads,
    num_kv_heads,
    cache=None,
    batch_indices=None,
    append_positions=None,
    kv_indices=None,
    kv_indptr=None,
    enable_pdl=True,
):
    """DSPARK_GLUE_CPP: writes q_out / k_out (and the cache pages when given) in place; returns nothing."""
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


def _dspark_drafter_inputs_fake(
    accepted,
    num_accepted,
    slots,
    ctx_len,
    max_ctx,
    table,
    mask_row,
    bonus,
    query_positions,
    ctx_positions,
    noise,
):
    """minimax_m31_dspark_drafter_inputs: writes bonus / query_positions / ctx_positions / noise in place,
    returns nothing."""
    return None


def _dspark_drafter_inputs_refresh_fake(
    accepted,
    num_accepted,
    slots,
    ctx_len,
    max_ctx,
    table,
    mask_row,
    bonus,
    query_positions,
    ctx_positions,
    noise,
    offsets,
    divisor,
    tables,
    counts,
):
    """minimax_m31_dspark_drafter_inputs_refresh: the drafter_inputs outputs plus tables / counts, all written
    in place; returns nothing."""
    return None


def _dspark_refresh_block_tables_fake(offsets, divisor, tables, counts):
    """minimax_m31_dspark_refresh_block_tables: writes tables / counts in place, returns nothing."""
    return None


def _dspark_prev_embeddings_fake(first_prev, tokens, w1, out):
    """minimax_m31_dspark_prev_embeddings (DSPARK_MARKOV_BIAS_CPP): writes out in place, returns nothing."""
    return None


def _dspark_widen_add_fake(logits, logits_row_stride, bias, out):
    """minimax_m31_dspark_widen_add (DSPARK_MARKOV_BIAS_CPP): writes out in place, returns nothing."""
    return None


def _dspark_ctx_store_fake(*args):
    """minimax_m31_dspark_ctx_store (DSPARK_CTX_STORE_CPP): writes the drafter's paged cache in place, returns
    nothing."""
    return None


def _expand_block_sampling_fake(
    temps, top_ks, top_ps, min_ps, seeds, offsets, K, slot_offset, span
):
    """m31_expand_block_sampling (TAIL_BK3 C): six [n * K] tensors in the inputs' dtypes."""
    n = temps.shape[0] * K
    return [t.new_empty((n,)) for t in (temps, top_ks, top_ps, min_ps, seeds, offsets)]


def _chain_spec_sampling_mc_fake(
    draft_probs, draft_token_ids, target_probs, seed, offset, workspace, ctas_per_row, v3=True
):
    """m31_chain_spec_mc (ST-01-REJ): (output_token_ids [batch, numDraft + 1], num_tokens [batch]), int32 as
    trtllm::chain_speculative_sampling; updates the persistent workspace in place.
    minimax_m31_chain_speculative_sampling_mc_ws_bytes takes no tensor and needs no fake."""
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
    v3=True,
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


def _swiglu_oai_mxfp8_fake(x, q, sf, h, rows_form=True, k_rows=4):
    """minimax_m31_swiglu_oai_mxfp8 (TP2CTX_SWIGLU_MXFP8): writes the E4M3 activation, its swizzled UE8M0 scales and
    (optionally) the bf16 activation into the caller's buffers; returns nothing."""
    return None


def _swiglu_oai_mxfp8_decode_fake(x, q, sf, finite_code0=True):
    """minimax_m31_swiglu_oai_mxfp8_decode (SWQ_CPP: TP2, TP8 SWQ16): writes the E4M3 activation and its
    swizzled UE8M0 scales into the caller's buffers; returns nothing."""
    return None


def _mxfp8_cublaslt_gemm_fake(act, act_scale, weight, weight_scale, algo):
    """minimax_m31_mxfp8_cublaslt_gemm (TP2CTX_MXFP8_CUBLASLT / AGG_PREFILL_CUBLASLT): E4M3 act [M, K] times the E4M3
    weight [N, K]^T with UE8M0 block scales -> a new bf16 [M, N]."""
    return act.new_empty((act.shape[0], weight.shape[0]), dtype=torch.bfloat16)


def _tp2ctx_peer_push_fake(src, dst):
    """minimax_m31_tp2ctx_peer_push (TP2CTX_SP): copies src into the peer's IPC slot at dst; returns nothing."""
    return None


def _draft_prob1_fake(
    base,
    w1,
    w2s,
    voff,
    k,
    prev_in,
    prev0,
    prev1,
    tok_out,
    ws,
    peers,
    caches,
    rank,
    world,
    grid,
    pdl,
    slot_ids,
    dummy_slot_row,
    offsets,
    rng_base,
    span,
    idx_map,
    sample_pos,
    temperature,
    seeds,
    cache,
    fault,
    fault_count,
    seq,
    fault_peers,
    cap_ns,
    stats,
    draft_probs,
    tokens32,
    vocab,
):
    """minimax_m3_sm107tp4_draft_prob1 / minimax_m3_sm107tp8_draft_prob1 (TP4 / TP8 probabilistic
    Markov draft + draft_probs adapter): writes tok_out / cache / draft_probs / tokens32 (and its workspace) in place;
    returns nothing."""
    return None


def _fsel_topk_fake(
    scores,
    n_valid_blocks,
    page_table,
    out,
    init_blocks,
    local_blocks,
    use_pdl=False,
    trigger_phase=0,
    variant=6,
):
    """minimax_m3_sm107tp4_fsel_topk (TP4 FSEL top-16 block select): fills out in place; returns nothing."""
    return None


def _bf16_to_fp32_fake(x):
    """minimax_m31_bf16_to_fp32 (TP2CTX_GATE_CAST): x.to(torch.float32), same shape."""
    return x.new_empty(x.shape, dtype=torch.float32)


def _widen_rows_fake(x, pdl=True):
    """minimax_m31_widen_rows (DEP_LOGITS_WIDEN): x.float() as a contiguous float32 [rows, cols]."""
    return x.new_empty(x.shape, dtype=torch.float32)


def _fill_padding_rows_fake(rows, is_padding, value, col0=None, pdl=True):
    """minimax_m31_fill_padding_rows (DEP_PAD_FILL): fills the padding rows in place; returns nothing."""
    return None


def _widen_add_rows_fake(base, bias, pdl=True):
    """minimax_m31_widen_add_rows (DEP_MARKOV_GLUE): base + bias.float() as a contiguous float32 [rows, cols]."""
    return base.new_empty(base.shape, dtype=torch.float32)


def _markov_inplace_fake(*args, **kwargs):
    """minimax_m31_markov_pick / _draw / _proposal (NATIVE_MARKOV_SAMPLING): write their outputs in place; return
    nothing."""
    return None


def _add_rmsnorm_fake(
    input,
    residual,
    weight,
    eps,
    q=None,
    sf=None,
    replicas=None,
    pdl=True,
    prefetch0=None,
    prefetch0_bytes=0,
    prefetch1=None,
    prefetch1_bytes=0,
):
    """minimax_m31_add_rmsnorm (NATIVE_ADD_RMSNORM): writes input / residual (and q / sf / replicas) in place; returns
    nothing."""
    return None


def _dflash_merge_later_keys_fake(
    out, lse_log2, query, block_key, block_value, scale, window_right, pdl=True
):
    """minimax_m31_dflash_merge_later_keys (the DFlash block merge): updates out in place; returns nothing."""
    return None


def _mc_allreduce_norm_fake(
    x,
    x2,
    residual,
    norm_w,
    eps,
    ws_uc,
    ws_mc_ptr,
    flags,
    rank,
    world,
    cluster,
    pdl,
    perm,
    scales,
    maxt,
    quant,
    prepushed,
    opts,
    ar2_order_fix=True,
):
    """[normed, updated residual] (+ [E4M3 rows, swizzled scale bytes] with quant)."""
    outputs = [torch.empty_like(residual), torch.empty_like(residual)]
    if quant:
        m, h = residual.shape
        outputs += [
            residual.new_empty((m, h), dtype=torch.float8_e4m3fn),
            residual.new_empty(((m + 127) // 128 * 128 * (h // 32),), dtype=torch.uint8),
        ]
    return outputs


def _mxfp8_low_m_gemm_fake(x, w, wsf, warps, cluster_k, row_tiles, pdl):
    """bf16 [M, N] of x [M, K] times w [N, K]^T."""
    return x.new_empty((x.shape[0], w.shape[0]))


def _sm107_mc_allreduce_norm_fake(
    x,
    x2,
    residual,
    norm_w,
    eps,
    ws_uc,
    ws_mc_ptr,
    flags,
    rank,
    world,
    cluster,
    pdl,
    perm,
    scales,
    maxt,
    quant,
    prepushed,
    opts,
    cap_out=None,
    cap_w=None,
    cap_eps=0.0,
    ar2_order_fix=True,
):
    """As minimax_m3_mc_allreduce_norm; cap_out is written in place."""
    return _mc_allreduce_norm_fake(
        x,
        x2,
        residual,
        norm_w,
        eps,
        ws_uc,
        ws_mc_ptr,
        flags,
        rank,
        world,
        cluster,
        pdl,
        perm,
        scales,
        maxt,
        quant,
        prepushed,
        opts,
    )


def _chain_spec_sampling_mc_win_fake(
    draft_probs,
    draft_token_ids,
    target_probs,
    seed,
    offset,
    workspace,
    ctas_per_row,
    base,
    add,
    window,
    base_row,
    v3=True,
):
    """As minimax_m31_chain_speculative_sampling_mc; the window is written in place."""
    return _chain_spec_sampling_mc_fake(
        draft_probs, draft_token_ids, target_probs, seed, offset, workspace, ctas_per_row
    )


def _chain_spec_sampling_mc_slots_win_fake(
    draft_buf,
    slot_ids,
    num_draft,
    draft_token_ids,
    target_probs,
    seed,
    offset,
    workspace,
    ctas_per_row,
    base,
    add,
    window,
    base_row,
    v3=True,
):
    """As minimax_m31_chain_speculative_sampling_mc_slots; the window is written in place."""
    return _chain_spec_sampling_mc_slots_fake(
        draft_buf,
        slot_ids,
        num_draft,
        draft_token_ids,
        target_probs,
        seed,
        offset,
        workspace,
        ctas_per_row,
    )


def _router_half_pin_fake(x, weight):
    """float32 [rows, experts] of x [rows, hidden] times weight [experts, hidden]^T."""
    return x.new_empty((x.shape[0], weight.shape[0]))


torch.library.register_fake(f"{_NS}::minimax_m3_mc_allreduce_norm")(_mc_allreduce_norm_fake)
_in_place("minimax_m3_msa_kv_update")
_in_place("minimax_m3_msa_kv_update_tail")
torch.library.register_fake(f"{_NS}::minimax_m3_mxfp8_low_m_gemm")(_mxfp8_low_m_gemm_fake)
torch.library.register_fake(f"{_NS}::minimax_m3_sm107_tp2_mc_allreduce_norm")(
    _sm107_mc_allreduce_norm_fake
)
_in_place("minimax_m3_sm107_tp2_sparse_decode_lc")
torch.library.register_fake(f"{_NS}::minimax_m3_sm107_tp2_sparse_decode_lc_prepare")(
    _sparse_decode_lc_prepare_fake
)
torch.library.register_fake(f"{_NS}::minimax_m3_sm107_tp4_mc_allreduce_norm")(
    _sm107_mc_allreduce_norm_fake
)
torch.library.register_fake(f"{_NS}::minimax_m3_sm107_tp8_mc_allreduce_norm")(
    _sm107_mc_allreduce_norm_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_add_rmsnorm")(_add_rmsnorm_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_ctx_residual_rmsnorm_mxfp8_f32")(
    _ctx_residual_rmsnorm_mxfp8_f32_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_bf16_to_fp32")(_bf16_to_fp32_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_chain_speculative_sampling_mc")(
    _chain_spec_sampling_mc_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_chain_speculative_sampling_mc_slots")(
    _chain_spec_sampling_mc_slots_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_chain_speculative_sampling_mc_win")(
    _chain_spec_sampling_mc_win_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_chain_speculative_sampling_mc_slots_win")(
    _chain_spec_sampling_mc_slots_win_fake
)
_host_query("minimax_m31_chain_speculative_sampling_mc_ws_bytes")
_host_query("minimax_m31_chain_speculative_sampling_mc_version")
torch.library.register_fake(f"{_NS}::minimax_m31_chain_speculative_sampling_mc_fi")(
    _chain_spec_sampling_mc_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_chain_speculative_sampling_mc_fi_slots")(
    _chain_spec_sampling_mc_slots_fake
)
_host_query("minimax_m31_chain_speculative_sampling_mc_fi_ws_bytes")
_in_place("minimax_m3_sm107_dep_vllm_topk")
torch.library.register_fake(f"{_NS}::minimax_m31_dflash_merge_later_keys")(
    _dflash_merge_later_keys_fake
)
_host_query("minimax_m3_sm107tp4_draft_prob1_max_grid")
torch.library.register_fake(f"{_NS}::minimax_m3_sm107tp4_draft_prob1")(_draft_prob1_fake)
_host_query("minimax_m3_sm107tp8_draft_prob1_max_grid")
torch.library.register_fake(f"{_NS}::minimax_m3_sm107tp8_draft_prob1")(_draft_prob1_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_refresh_block_tables")(
    _dspark_refresh_block_tables_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_ctx_store")(_dspark_ctx_store_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_drafter_inputs")(
    _dspark_drafter_inputs_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_drafter_inputs_refresh")(
    _dspark_drafter_inputs_refresh_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_advance")(_dspark_advance_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_indices")(_dspark_indices_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_advance_par")(_dspark_advance_par_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_qk_norm_rope")(_dspark_qk_norm_rope_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_prev_embeddings")(
    _dspark_prev_embeddings_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_dspark_widen_add")(_dspark_widen_add_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_fc_shard_gemm")(_fc_shard_gemm_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_fill_padding_rows")(_fill_padding_rows_fake)
torch.library.register_fake(f"{_NS}::minimax_m3_sm107tp4_fsel_topk")(_fsel_topk_fake)
_in_place("m31idxk_nvfp4_idxk_write")
_in_place("m31idxk_nvfp4_idxk_write_from_sink")
_in_place("m31idxk_nvfp4_idxk_gather_fp8")
torch.library.register_fake(f"{_NS}::minimax_m31_markov_epilogue")(_markov_epilogue_fake)
_in_place("minimax_m31_markov_epilogue_bf16")
torch.library.register_fake(f"{_NS}::minimax_m31_zero_padding_rows")(_zero_padding_rows_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_scatter_rows")(_scatter_rows_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_markov_prev_embed")(_markov_prev_embed_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_expand_block_sampling")(
    _expand_block_sampling_fake
)
_host_query("minimax_m31_markov_epilogue_splits")
torch.library.register_fake(f"{_NS}::minimax_m31_markov_pick")(_markov_inplace_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_markov_draw")(_markov_inplace_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_markov_proposal")(_markov_inplace_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_moe_finalize")(_moe_finalize_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_mxfp8_cublaslt_gemm")(_mxfp8_cublaslt_gemm_fake)
_in_place("minimax_m31_mxfp8_cublaslt_gemm_out")
_in_place("noaux_tc_fp4_quantize_stage_out")
_in_place("noaux_tc_route_stage_out")
torch.library.register_fake(f"{_NS}::minimax_m31_nvfp4_qkv_indexer_producer_rows")(
    _qkv_indexer_producer_rows_fake
)
_host_query("m31_router_half_pin_prepare")
torch.library.register_fake(f"{_NS}::m31_router_half_pin")(_router_half_pin_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_ctx_residual_rmsnorm_mxfp8")(
    _ctx_residual_rmsnorm_mxfp8_fake
)
_in_place("minimax_m31_ctx_residual_rmsnorm_mxfp8_out")
torch.library.register_fake(f"{_NS}::minimax_m31_select_prefill_tile")(_select_prefill_tile_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_select_prefill_tile_rows")(
    _select_prefill_tile_rows_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_softmax_mc")(_softmax_mc_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_softmax_pathb")(_softmax_pathb_fake)
_in_place("minimax_m31_tp2ctx_peer_add_normq_out")
torch.library.register_fake(f"{_NS}::minimax_m31_swiglu_oai_mxfp8")(_swiglu_oai_mxfp8_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_swiglu_oai_mxfp8_decode")(
    _swiglu_oai_mxfp8_decode_fake
)
torch.library.register_fake(f"{_NS}::minimax_m31_swiglu_oai_vec")(_swiglu_oai_vec_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_topp_renorm_det")(_topp_renorm_det_fake)
_host_query("minimax_m31_topp_renorm_det_ws_bytes")
torch.library.register_fake(f"{_NS}::minimax_m31_topp_renorm_det_v3")(_topp_renorm_det_v3_fake)
_in_place("minimax_m31_topp_renorm_det_v3_rows")
_host_query("minimax_m31_topp_renorm_det_v3_ws_bytes")
_in_place("minimax_m31_tp2ctx_peer_add")
torch.library.register_fake(f"{_NS}::minimax_m31_tp2ctx_peer_push")(_tp2ctx_peer_push_fake)
_in_place("minimax_m31_tp2ctx_ipc_barrier")
torch.library.register_fake(f"{_NS}::minimax_m31_widen_add_rows")(_widen_add_rows_fake)
torch.library.register_fake(f"{_NS}::minimax_m31_widen_rows")(_widen_rows_fake)
