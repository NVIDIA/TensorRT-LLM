# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-launch promotion of the accepted convolution state with GDN replay bookkeeping."""

import torch
import triton
import triton.language as tl


@triton.jit
def _promote_conv_and_advance_replay_kernel(
    src_ptr,
    dst_ptr,
    src_idx_ptr,
    accepted_position_source_ptr,
    num_accepted_tokens_ptr,
    blk_ptr,
    replay_idx_ptr,
    replay_pnat_ptr,
    replay_cache_buf_idx_ptr,
    dummy_request_mask_ptr,
    num_gens,
    count,
    replay_pnat_stride,
    replay_cache_buf_idx_stride,
    src_s_layer,
    src_s_row,
    src_s_step,
    dst_s_layer,
    dst_s_block,
    POSITION_SOURCE_IS_TOKEN_COUNT: tl.constexpr,
    REPLAY_STEP_WIDTH: tl.constexpr,
    REPLAY_HISTORY_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program copies a BLOCK-sized tile of the contiguous inner state for a
    # single (layer, gen) pair. grid = (num_layers * num_gens, ceil(count/BLOCK)).
    pair = tl.program_id(0)
    tile = tl.program_id(1)
    layer = (pair // num_gens).to(tl.int64)
    g = pair % num_gens
    row = tl.load(src_idx_ptr + g).to(tl.int64)
    accepted_position = tl.load(accepted_position_source_ptr + g)
    if POSITION_SOURCE_IS_TOKEN_COUNT:
        accepted_position -= 1
    acc = accepted_position.to(tl.int64)
    blk = tl.load(blk_ptr + g).to(tl.int64)

    # The replay metadata update is batch-scoped, so layer 0's first copy tile
    # owns it: PNAT and active-buffer maintenance ride on the promotion launch.
    if (layer == 0) & (tile == 0):
        num_accepted_tokens = tl.load(num_accepted_tokens_ptr + g)
        is_dummy_request = tl.load(dummy_request_mask_ptr + g)
        replay_slot = tl.load(replay_idx_ptr + g).to(tl.int64)
        pnat_ptr = replay_pnat_ptr + replay_slot * replay_pnat_stride
        cache_buf_idx_ptr = replay_cache_buf_idx_ptr + replay_slot * replay_cache_buf_idx_stride
        pnat = tl.load(pnat_ptr)
        cache_buf_idx = tl.load(cache_buf_idx_ptr)
        wrote_checkpoint = pnat + REPLAY_STEP_WIDTH > REPLAY_HISTORY_SIZE
        next_pnat = tl.where(wrote_checkpoint, num_accepted_tokens, pnat + num_accepted_tokens)
        next_cache_buf_idx = tl.where(wrote_checkpoint, 1 - cache_buf_idx, cache_buf_idx)
        tl.store(pnat_ptr, next_pnat, mask=~is_dummy_request)
        tl.store(cache_buf_idx_ptr, next_cache_buf_idx, mask=~is_dummy_request)

    # int64 throughout: per-layer strides are O(1e8), so layer*stride overflows
    # int32 and would corrupt addresses / fault.
    src_base = layer * src_s_layer + row * src_s_row + acc * src_s_step
    dst_base = layer * dst_s_layer + blk * dst_s_block
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    v = tl.load(src_ptr + src_base + offs, mask=mask)
    tl.store(dst_ptr + dst_base + offs, v, mask=mask)


def promote_conv_and_advance_replay_triton(
    dst: torch.Tensor,
    intermediate: torch.Tensor,
    src_state_indices: torch.Tensor,
    accepted_position_source: torch.Tensor,
    dst_state_indices: torch.Tensor,
    *,
    num_accepted_tokens: torch.Tensor,
    replay_pnat: torch.Tensor,
    replay_cache_buf_idx: torch.Tensor,
    dummy_request_mask: torch.Tensor,
    replay_step_width: int,
    replay_history_size: int,
    replay_indices: torch.Tensor | None = None,
    position_source_is_token_count: bool = False,
    BLOCK: int = 2048,
) -> None:
    """Promote the accepted state and advance the replay bookkeeping in one launch.

    The copy is ``_promote_mamba_state_triton``'s:
    ``dst[:, dst_state_indices[g]] = intermediate[:, src_state_indices[g], accepted_position]``.
    The same launch advances PNAT and the active history half of each row's
    replay slot, as ``_advance_replay_state`` does.

    Args:
        dst: ``[num_layers, num_blocks, *state_shape]`` (strided) view of the pool.
        intermediate: ``[num_layers, max_batch_size, T, *state_shape]`` dense.
        src_state_indices, accepted_position_source, dst_state_indices:
            ``[num_gens]`` int.
        num_accepted_tokens: ``[num_gens]`` accepted-token counts.
        replay_pnat, replay_cache_buf_idx: Replay bookkeeping, one int32 per
            replay slot at any element stride.
        dummy_request_mask: ``[num_gens]`` bool; dummy rows keep their bookkeeping.
        replay_step_width, replay_history_size: The replay geometry.
        replay_indices: ``[num_gens]`` replay slots of the rows; ``None`` means
            ``dst_state_indices``.
        position_source_is_token_count: Whether ``accepted_position_source``
            holds accepted-token counts instead of zero-based positions.
    """
    num_layers = dst.shape[0]
    num_gens = src_state_indices.shape[0]
    if num_gens == 0:
        return
    assert replay_step_width > 0
    assert replay_history_size >= replay_step_width
    assert replay_pnat.dtype == torch.int32
    assert replay_cache_buf_idx.dtype == torch.int32
    assert dummy_request_mask.dtype == torch.bool
    assert dummy_request_mask.numel() >= num_gens
    if replay_indices is None:
        replay_indices = dst_state_indices
    assert replay_indices.numel() >= num_gens
    count = 1
    for s in dst.shape[2:]:
        count *= s
    # (layer, gen) pairs in grid dim 0 (limit 2^31-1), tiles in dim 1 (limit 65535).
    grid = (num_layers * num_gens, triton.cdiv(count, BLOCK))
    _promote_conv_and_advance_replay_kernel[grid](
        intermediate,
        dst,
        src_state_indices,
        accepted_position_source,
        num_accepted_tokens,
        dst_state_indices,
        replay_indices,
        replay_pnat,
        replay_cache_buf_idx,
        dummy_request_mask,
        num_gens,
        count,
        replay_pnat.stride(0),
        replay_cache_buf_idx.stride(0),
        intermediate.stride(0),
        intermediate.stride(1),
        intermediate.stride(2),
        dst.stride(0),
        dst.stride(1),
        POSITION_SOURCE_IS_TOKEN_COUNT=position_source_is_token_count,
        REPLAY_STEP_WIDTH=replay_step_width,
        REPLAY_HISTORY_SIZE=replay_history_size,
        BLOCK=BLOCK,
    )


__all__ = ["promote_conv_and_advance_replay_triton"]
