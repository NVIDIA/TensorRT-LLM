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


def _topp_renorm_det_v3_fake(probs, top_p, workspace, out, ctas_per_row, pdl, candidate_list):
    """m31_topp_det v3: as minimax_m31_topp_renorm_det (in place into out, workspace updated); returns nothing.
    minimax_m31_topp_renorm_det_v3_ws_bytes takes no tensor and needs no fake."""
    return None


def _topp_renorm_det_fake(probs, top_p, workspace, out, ctas_per_row):
    """m31_topp_det (ST-01): writes the renormed [rows, d] fp32 probs into out in place (and updates the persistent
    workspace); returns nothing. minimax_m31_topp_renorm_det_ws_bytes takes no tensor and needs no fake."""
    return None


def _softmax_mc_fake(logits, temperature, temperature_val, partials, out, write_ctas):
    """m31_softmax_mc (ST-01-SM): writes the [rows, d] fp32 probs into out in place (and overwrites the partials
    workspace); returns nothing."""
    return None


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


# op name -> fake impl (schemas as of img9, ONE_DIV included).
FAKES = {
    "minimax_m31_ctx_residual_rmsnorm_mxfp8": _ctx_residual_rmsnorm_mxfp8_fake,
    "minimax_m31_select_prefill_tile": _select_prefill_tile_fake,
    "minimax_m31_tp4_sparse_decode_lc": _sparse_decode_lc_fake,
    "minimax_m31_tp8_sparse_decode_lc8": _sparse_decode_lc8_fake,
    "minimax_m31_topp_renorm_det": _topp_renorm_det_fake,
    "minimax_m31_topp_renorm_det_v3": _topp_renorm_det_v3_fake,
    "minimax_m31_softmax_mc": _softmax_mc_fake,
    "minimax_m31_chain_speculative_sampling_mc": _chain_spec_sampling_mc_fake,
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
