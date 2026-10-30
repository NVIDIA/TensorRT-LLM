# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-torch mirrors of what both draft-step ops write besides their main output.

The greedy draft token over the vocabulary-sharded logits, and the next draft step's state: position ids,
sequence and KV lengths, and the DSA decode metadata that the fused DSA decode-metadata kernel writes for a
generation-only batch. Shared by the references of ``glm_draft_sample_glue`` and ``glm_draft_ehproj_table``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def global_argmax(per_rank_logits: Sequence[torch.Tensor]) -> torch.Tensor:
    """The argmax of the rank-ordered concatenation of the logits shards, in torch.argmax's order.

    A NaN beats every number and the first NaN wins; otherwise the largest value, the lowest index on ties.
    Returns int32 ``[rows]``.
    """
    logits = torch.cat([shard.float() for shard in per_rank_logits], dim=-1)
    return torch.argmax(logits, dim=-1).to(torch.int32)


def _rms_normalize(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``bf16((x * rsqrt(mean(x^2) + eps)) * weight)`` in FP32, one row per leading index."""
    rstd = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((x.float() * rstd) * weight.float()).to(torch.bfloat16)


def eh_proj_input_window(
    tokens: torch.Tensor,
    embed_table: torch.Tensor,
    hidden: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
    col_lo: int,
    col_hi: int,
) -> torch.Tensor:
    """Columns ``[col_lo, col_hi)`` of ``cat([enorm(embed(tokens)), hnorm(hidden)], -1)``.

    An id outside the table gives a zero embedding row.
    """
    vocab = embed_table.shape[0]
    ids = tokens.long()
    valid = (ids >= 0) & (ids < vocab)
    embeds = embed_table[ids.clamp(0, vocab - 1)].float() * valid.unsqueeze(1)
    concat = torch.cat(
        [_rms_normalize(embeds, enorm_weight, eps), _rms_normalize(hidden, hnorm_weight, eps)],
        dim=-1,
    )
    return concat[:, col_lo:col_hi].contiguous()


def next_step_state(
    rows: int,
    positions_in: Optional[torch.Tensor],
    positions_out: Optional[torch.Tensor],
    gather_ids: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    fill_seq_lens: bool,
    kv_lens: Optional[torch.Tensor],
    kv_update: int,
    rewind_accepted: Optional[torch.Tensor],
    draft_len: int,
    block_offsets: Optional[torch.Tensor],
    dsa_layout: List[int],
    req_idx_per_token: Optional[torch.Tensor],
    slot_mapping_fp8: Optional[torch.Tensor],
    slot_mapping_scale: Optional[torch.Tensor],
    gen_kv_indptr: Optional[torch.Tensor],
    gen_cached_token_indptr: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """What a draft-step op leaves in each state buffer it is handed, read as the buffers were before the call.

    Position ids: ``positions_out[r] = positions_in[gather_ids[r] if gather_ids else r] + 1``. Sequence lengths:
    filled with 1 when ``fill_seq_lens``. KV lengths: ``+ 1`` (kv_update 1) or ``- (draft_len -
    rewind_accepted[r])`` (kv_update 2). DSA decode metadata, from those lengths: the generation KV and cached
    indptrs and, for each request's tokens (at most ``dsa_layout[1]`` per request, below ``dsa_layout[0]`` in
    all), the request index and the indexer-K cache slot offsets, the block index truncated and clamped as the
    fused DSA decode-metadata kernel computes it. Entries the op does not write keep their values. Returns the
    expected buffers by argument name.
    """
    out: Dict[str, torch.Tensor] = {}
    if positions_out is not None:
        expected = positions_out.clone()
        source = gather_ids[:rows].long() if gather_ids is not None else torch.arange(rows)
        expected[:rows] = positions_in.reshape(-1)[source.to(positions_in.device)] + 1
        out["positions_out"] = expected
    seq = None
    if seq_lens is not None:
        expected = seq_lens.clone()
        if fill_seq_lens:
            expected[:rows] = 1
        out["seq_lens"] = expected
        seq = [int(v) for v in expected[:rows].tolist()]
    kv = None
    if kv_lens is not None:
        expected = kv_lens.clone()
        if kv_update == 1:
            expected[:rows] += 1
        elif kv_update == 2:
            expected[:rows] -= draft_len - rewind_accepted[:rows].to(expected.dtype)
        out["kv_lens"] = expected
        kv = [int(v) for v in expected[:rows].tolist()]
    if req_idx_per_token is None:
        return out

    num_tokens, lanes, tokens_per_block, data_bytes, scale_size, block_stride, scale_base = (
        dsa_layout
    )
    req_idx = req_idx_per_token.clone()
    fp8 = slot_mapping_fp8.clone()
    scale = slot_mapping_scale.clone()
    kv_indptr = gen_kv_indptr.clone()
    cached_indptr = gen_cached_token_indptr.clone()
    kv_indptr[0] = 0
    cached_indptr[0] = 0
    blocks = block_offsets.cpu().tolist()
    max_blocks = block_offsets.shape[1]
    flat_start = 0
    kv_sum = 0
    cached_sum = 0
    for r in range(rows):
        kv_sum += kv[r]
        cached_sum += kv[r] - seq[r]
        kv_indptr[1 + r] = kv_sum
        cached_indptr[1 + r] = cached_sum
        start = kv[r] - seq[r]
        for j in range(min(seq[r], lanes)):
            g = flat_start + j
            if g >= num_tokens:
                break
            position = start + j
            # Truncating division, a non-negative remainder, then the clamp.
            block = abs(position) // tokens_per_block * (1 if position >= 0 else -1)
            offset = position - block * tokens_per_block
            if offset < 0:
                offset += tokens_per_block
            block = min(max(block, 0), max_blocks - 1)
            block_id = blocks[r][block]
            req_idx[g] = r
            fp8[g] = block_id * block_stride + offset * data_bytes
            scale[g] = block_id * block_stride + scale_base + offset * scale_size
        flat_start += seq[r]
    out.update(
        req_idx_per_token=req_idx,
        slot_mapping_fp8=fp8,
        slot_mapping_scale=scale,
        gen_kv_indptr=kv_indptr,
        gen_cached_token_indptr=cached_indptr,
    )
    return out


def assert_state_args_valid(
    rows: int,
    draft_tokens: torch.Tensor,
    positions_in: Optional[torch.Tensor],
    positions_out: Optional[torch.Tensor],
    gather_ids: Optional[torch.Tensor],
    kv_update: int,
    draft_len: int,
    dsa_layout: List[int],
    req_idx_per_token: Optional[torch.Tensor],
) -> None:
    """The state arguments both draft-step ops accept and answer wrongly."""
    if positions_out is not None and positions_in is not None:
        # Lanes read positions_in[gather_ids[r]] while others write positions_out[r]: overlapping buffers race.
        lo_in, hi_in = positions_in.data_ptr(), positions_in.data_ptr() + positions_in.nbytes
        lo_out, hi_out = positions_out.data_ptr(), positions_out.data_ptr() + positions_out.nbytes
        assert hi_in <= lo_out or hi_out <= lo_in, "positions_out must not overlap positions_in"
        if gather_ids is not None:
            source = gather_ids[:rows]
            assert bool(((source >= 0) & (source < positions_in.numel())).all()), (
                "gather_ids must index positions_in"
            )
    # The rewind subtracts draft_len - accepted from each KV length: another draft length rewinds wrongly.
    assert kv_update != 2 or draft_len == draft_tokens.shape[1], (
        "the KV rewind needs draft_len equal to the step's draft length"
    )
    if req_idx_per_token is not None:
        # No token lanes, or no tokens per block, writes no metadata or divides by zero.
        assert len(dsa_layout) == 7 and dsa_layout[1] >= 1 and dsa_layout[2] >= 1, (
            "dsa_layout needs at least one token lane and one token per block"
        )
