# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merge the ranks' draft top-k keys into ``torch.topk``'s full-vocab result,
and build the DFlash 2 selector's two einsum operands from it.

Input is the all-gather (along dim 0) of every rank's ``draft_shard_topk``
keys. Per block position this entry reproduces what the built-in path gets
from ``torch.topk(gathered_fp32_logits, top_k, dim=-1)`` -- values, vocab ids
*and their order* -- and then does the selector's gathers in the same program:

* selection: the ``top_k`` largest under (torch radix key desc, vocab id asc).
  Every element above the k-th value, plus the lowest-id elements equal to it;
  the union of per-chunk top-k lists holds them all.
* arrangement: torch's ``gatherTopK`` writes the selected elements strictly
  above the k-th value first, then those equal to it, each group in ascending
  id. Rebuilt here from the selected set.
* order: torch then sorts the k pairs in place with
  ``bitonicSortKVInPlace`` (``SortUtils.cuh``): 32 slots, slot ``i`` = input
  ``i``, the rest invalid; ``bitonicSort`` with ``GTOp<float, true>``. The
  network is not stable, so tied values come out in an order that depends on
  the arrangement; it is run here step for step.
* operands: ``successor_codebook[ids].float()`` and
  ``predecessor_codebook[prev].float() * gate.float()``, ``prev`` being the
  anchor token at position 0 and the previous position's candidates after.
  Both are exact: a bf16 widens to fp32 exactly and a bf16 * bf16 product
  fits fp32's significand, so these equal the built-in tensors bit for bit,
  and the built-in einsum over them is unchanged.

With a drafter logit transform ``t`` the keys are canonical
(``draft_shard_topk``'s ``canon``) and ``values`` is the transform's table
(``kernels/draft_head/logit_transform.py``): ``values[key16]`` is ``t`` of the
bf16 value of ``key16``, computed by the drafter's own method. The selection,
the arrangement and the network above run on canonical keys and their decoded
bf16 values unchanged, which compare as the transformed values do, so they give
``torch.topk(t(logits))``'s ids and order; the unary values are then
``values[key16]``, bitwise ``t`` of every candidate, since all members of a tie
group transform to the same bits.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .draft_shard_topk import CANON_ENTRIES, SHARD, radix_key16

#: ``torch.topk`` sorts k <= 32 results with this many bitonic slots.
SORT_SLOTS = 32
#: Codebook columns per program, and warps per program: eight programs per row
#: share the codebook gathers, each recomputing the (small) merge.
RANK_BLOCK = 32
NUM_WARPS = 4


@triton.jit
def _decode_value(key16):
    """fp32 value of a 16-bit radix key (the inverse of torch's bf16 key)."""
    bits = tl.where((key16 & 0x8000) != 0, key16 ^ 0x8000, key16 ^ 0xFFFF)
    return (bits << 16).to(tl.int32).to(tl.float32, bitcast=True)


@triton.jit
def _bitonic_step(
    slot, STRIDE: tl.constexpr, SIZE: tl.constexpr, FINAL: tl.constexpr, SLOTS: tl.constexpr
):
    """One ``bitonicSwap`` round of torch's ``bitonicSort`` over all slots.

    ``slot`` holds ``(vocab_id << 16) | key16`` for a valid entry and -1 for an
    invalid one. The pairs are (``pos``, ``pos + STRIDE``) with ``pos`` the
    slots whose ``STRIDE`` bit is clear -- a ``[SLOTS / 2·STRIDE, 2, STRIDE]``
    view puts each pair on the middle axis; torch's thread-index direction flag
    ``tid & (size / 2)`` is ``pos & size`` for that pair.
    """
    BLOCKS: tl.constexpr = SLOTS // (2 * STRIDE)
    pairs = tl.reshape(slot, [BLOCKS, 2, STRIDE])
    side = tl.arange(0, 2)[None, :, None]
    a = tl.sum(tl.where(side == 0, pairs, 0), axis=1)
    b = tl.sum(tl.where(side == 1, pairs, 0), axis=1)
    pos = tl.arange(0, BLOCKS)[:, None] * (2 * STRIDE) + tl.arange(0, STRIDE)[None, :]
    if FINAL:
        flag = pos < 0
    else:
        flag = (pos & SIZE) != 0
    fa = _decode_value(a & 0xFFFF)
    fb = _decode_value(b & 0xFFFF)
    greater = ((fa != fa) & (fb == fb)) | (fa > fb)
    swap = (greater & (a >= 0)) | (b < 0)
    exchange = swap == flag
    low = tl.where(exchange, b, a)
    high = tl.where(exchange, a, b)
    return tl.reshape(tl.where(side == 0, low[:, None, :], high[:, None, :]), [SLOTS])


@triton.jit
def _draft_topk_merge_kernel(
    keys_ptr,
    anchor_ptr,
    gate_ptr,
    pred_cb_ptr,
    succ_cb_ptr,
    ids_ptr,
    unary_ptr,
    pred_out_ptr,
    succ_out_ptr,
    values_ptr,
    rows_per_rank,
    num_ranks,
    num_keys,
    shard,
    num_positions,
    rank_dim,
    keys_row_stride,
    gate_stride_b,
    gate_stride_k,
    pred_cb_stride,
    succ_cb_stride,
    RANKS_PAD: tl.constexpr,
    KEYS_PAD: tl.constexpr,
    TOP_K: tl.constexpr,
    SLOTS: tl.constexpr,
    RANK_BLOCK: tl.constexpr,
    HALVES: tl.constexpr,
    TRANSFORM: tl.constexpr,
):
    # Every program of a row runs the (small) merge; each gathers one RANK_BLOCK
    # slice of the codebook rows, so the gathers spread over more SMs.
    b = tl.program_id(0)
    t = tl.program_id(1)
    part = tl.program_id(2)
    row = b * num_positions + t

    # Loads that do not depend on the candidates, issued first.
    rc = part * RANK_BLOCK + tl.arange(0, RANK_BLOCK)[None, :]
    in_rank = rc < rank_dim
    gate_row = gate_ptr + b.to(tl.int64) * gate_stride_b
    has_next = t + 1 < num_positions
    gate_next = tl.load(
        gate_row + (t + 1) * gate_stride_k + rc, mask=in_rank & has_next, other=0.0
    ).to(tl.float32)
    gate0 = tl.load(gate_row + rc, mask=in_rank & (t == 0), other=0.0).to(tl.float32)
    anchor = tl.load(anchor_ptr + b).to(tl.int64)
    pred0 = tl.load(
        pred_cb_ptr + anchor * pred_cb_stride + rc, mask=in_rank & (t == 0), other=0.0
    ).to(tl.float32)

    # Candidates: every rank's keys for this row, as 64-bit (key16, -vocab id).
    rr = tl.arange(0, RANKS_PAD)[:, None]
    cc = tl.arange(0, KEYS_PAD)[None, :]
    present = (rr < num_ranks) & (cc < num_keys)
    if HALVES:
        # [M, ranks * keys * 2] fp32: each key as its two 16-bit halves, exact integers.
        half = keys_ptr + row.to(tl.int64) * keys_row_stride + (rr * num_keys + cc) * 2
        high = tl.load(half, mask=present, other=32768.0).to(tl.int32)
        low = tl.load(half + 1, mask=present, other=0.0).to(tl.int32)
        key = ((high << 16) | low).to(tl.int64)
    else:
        # [ranks * M, keys] int32, rank-major.
        key = tl.load(
            keys_ptr + (rr * rows_per_rank + row).to(tl.int64) * keys_row_stride + cc,
            mask=present,
            other=-2147483648,
        ).to(tl.int64)
    key16 = ((key >> 16) & 0xFFFF) ^ 0x8000
    vocab = rr.to(tl.int64) * shard + (0xFFFF - (key & 0xFFFF))
    merged = tl.where(key != -2147483648, (key16 << 32) | (0xFFFFFFFF - vocab), -1)
    merged = tl.reshape(merged, (RANKS_PAD * KEYS_PAD,))

    # Selection: the TOP_K largest merged keys, largest first.
    kk = tl.arange(0, TOP_K)
    chosen = tl.full([TOP_K], -1, tl.int64)
    for j in tl.static_range(TOP_K):
        best = tl.max(merged, axis=0)
        chosen = tl.where(kk == j, best, chosen)
        merged = tl.where(merged == best, -1, merged)
    chosen16 = chosen >> 32
    chosen_id = 0xFFFFFFFF - (chosen & 0xFFFFFFFF)
    kth16 = tl.sum(tl.where(kk == TOP_K - 1, chosen16, 0), axis=0)

    # Arrangement: above the k-th value by ascending id, then equal to it by ascending id.
    order = ((chosen16 == kth16).to(tl.int64) << 40) | chosen_id
    position = tl.sum((order[None, :] < order[:, None]).to(tl.int32), axis=1)
    payload = (chosen_id << 16) | chosen16
    ss = tl.arange(0, SLOTS)
    slot = tl.sum(tl.where(ss[:, None] == position[None, :], payload[None, :], 0), axis=1)
    slot = tl.where(ss < TOP_K, slot, -1)

    # torch's bitonicSort<32>: sizes 2..16 with the thread-index direction, then the final merge.
    slot = _bitonic_step(slot, 1, 2, False, SLOTS)
    slot = _bitonic_step(slot, 2, 4, False, SLOTS)
    slot = _bitonic_step(slot, 1, 4, False, SLOTS)
    slot = _bitonic_step(slot, 4, 8, False, SLOTS)
    slot = _bitonic_step(slot, 2, 8, False, SLOTS)
    slot = _bitonic_step(slot, 1, 8, False, SLOTS)
    slot = _bitonic_step(slot, 8, 16, False, SLOTS)
    slot = _bitonic_step(slot, 4, 16, False, SLOTS)
    slot = _bitonic_step(slot, 2, 16, False, SLOTS)
    slot = _bitonic_step(slot, 1, 16, False, SLOTS)
    slot = _bitonic_step(slot, 16, 0, True, SLOTS)
    slot = _bitonic_step(slot, 8, 0, True, SLOTS)
    slot = _bitonic_step(slot, 4, 0, True, SLOTS)
    slot = _bitonic_step(slot, 2, 0, True, SLOTS)
    slot = _bitonic_step(slot, 1, 0, True, SLOTS)

    result = tl.sum(tl.where(ss[None, :] == kk[:, None], slot[None, :], 0), axis=1)
    ids = result >> 16
    if part == 0:
        tl.store(ids_ptr + row * TOP_K + kk, ids)
        if TRANSFORM:
            # The drafter transform of each candidate: one table entry per tie group.
            tl.store(unary_ptr + row * TOP_K + kk, tl.load(values_ptr + (result & 0xFFFF)))
        else:
            tl.store(unary_ptr + row * TOP_K + kk, _decode_value(result & 0xFFFF))

    # Selector operands for this row (successors) and the next one (predecessors).
    out_rows = (row * TOP_K + kk[:, None]).to(tl.int64) * rank_dim
    id_col = ids[:, None]
    succ = tl.load(succ_cb_ptr + id_col * succ_cb_stride + rc, mask=in_rank, other=0.0)
    tl.store(succ_out_ptr + out_rows + rc, succ.to(tl.float32), mask=in_rank)
    if has_next:
        pred = tl.load(pred_cb_ptr + id_col * pred_cb_stride + rc, mask=in_rank, other=0.0)
        pred = pred.to(tl.float32) * gate_next
        tl.store(pred_out_ptr + out_rows + TOP_K * rank_dim + rc, pred, mask=in_rank)
    if t == 0:
        tl.store(
            pred_out_ptr + out_rows + rc,
            tl.broadcast_to(pred0 * gate0, [TOP_K, RANK_BLOCK]),
            mask=in_rank,
        )


def unpack_halves(packed: torch.Tensor, num_ranks: int) -> torch.Tensor:
    """``[M, ranks * n * 2]`` fp32 key halves (the all-reduce layout) -> ``[ranks * M, n]`` int32."""
    rows = packed.shape[0]
    halves = packed.view(rows, num_ranks, -1, 2).to(torch.int64)
    key = (halves[..., 0] << 16) | halves[..., 1]
    key = torch.where(key >= 2**31, key - 2**32, key)
    return key.permute(1, 0, 2).reshape(num_ranks * rows, -1).to(torch.int32)


def _decode_keys(keys: torch.Tensor, ranks: int, shard: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(vocab ids int64, fp32 values) of gathered keys ``[ranks * M, n]``; id -1 where empty."""
    key = keys.to(torch.int64)
    rows = keys.shape[0] // ranks
    key16 = ((key >> 16) & 0xFFFF) ^ 0x8000
    rank = torch.arange(ranks, device=keys.device).repeat_interleave(rows)[:, None]
    vocab = rank * shard + (0xFFFF - (key & 0xFFFF))
    bits = torch.where((key16 & 0x8000) != 0, key16 ^ 0x8000, key16 ^ 0xFFFF)
    bits = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    values = bits.view(torch.bfloat16).float()
    empty = keys == -(2**31)
    return torch.where(empty, torch.full_like(vocab, -1), vocab), values


class DraftTopKMerge(OpWrapper):
    """Full-vocab top-k of the gathered draft keys, and the selector's einsum operands.

        gathered_keys         every rank's ``draft_shard_topk`` keys, either
                              [num_ranks * M, n] int32 (all-gathered along dim 0,
                              rank-major) or [M, num_ranks * n * 2] fp32 (each key
                              as two 16-bit halves at its rank's slot: the
                              all-reduce layout ``draft_shard_topk(rank=...)`` writes)
        shard                 vocab rows per rank (global id = rank * shard + column)
        num_requests (B), num_positions (K): M = B * K rows, request-major
        anchor_ids            [B] int: each request's last accepted token
        gate                  [B, K, rank] bf16: ``hidden_projection`` of the block
        predecessor_codebook  [vocab, rank] bf16, stride(-1) == 1
        successor_codebook    [vocab, rank] bf16, stride(-1) == 1
        top_k                 <= 32, a power of two
        num_ranks             ranks in the gather (the tensor-parallel size)
        rank_block, num_warps launch geometry
        values                None, or a drafter logit transform's table [65536]
                              fp32 (key16 -> transformed value); the keys are then
                              ``draft_shard_topk``'s canonical keys of that transform

    Returns ``(candidate_ids [B, K, top_k] int64, unary [B, K, top_k] fp32,
    predecessors [B, K, top_k, rank] fp32, successors [B, K, top_k, rank] fp32)``:
    ``candidate_ids`` / ``unary`` are ``torch.topk(full_logits.float(), top_k)``'s
    indices / values -- with ``values``, ``torch.topk(t(full_logits))``'s, ``t``
    the transform -- and the operands are ``dflash2_score_edges``'s
    ``predecessors * gate.float()[:, :, None]`` and ``successors``. Exact.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            "one request at TP8, int32 all-gather keys",
            {
                "requests": 1,
                "positions": 7,
                "shard": SHARD,
                "ranks": 8,
                "layout": "int32",
                "transform": None,
            },
        ),
        Cell(
            "one request at TP8, fp32 all-reduce halves",
            {
                "requests": 1,
                "positions": 7,
                "shard": SHARD,
                "ranks": 8,
                "layout": "halves",
                "transform": None,
            },
        ),
        Cell(
            "one request at TP8, fp32 halves, the Muse Spark Open drafter transform",
            {
                "requests": 1,
                "positions": 7,
                "shard": SHARD,
                "ranks": 8,
                "layout": "halves",
                "transform": "musespark",
            },
        ),
        Cell(
            "two requests at TP4, int32 keys, the drafter transform",
            {
                "requests": 2,
                "positions": 7,
                "shard": 2 * SHARD,
                "ranks": 4,
                "layout": "int32",
                "transform": "musespark",
            },
        ),
        Cell(
            "24 requests at TP8, rank top-k halves, the drafter transform",
            {
                "requests": 24,
                "positions": 7,
                "shard": SHARD,
                "ranks": 8,
                "layout": "rank",
                "transform": "musespark",
            },
        ),
    )
    note = (
        "Tie order is torch 2.14's (SortUtils.cuh bitonicSort, k <= 32 -> 32 "
        "slots), which torch does not document. draft_head.install checks it "
        "against the running torch.topk (tie_order_reason) and refuses the "
        "feature on a mismatch; the test compares with torch.topk directly."
    )

    def __call__(
        self,
        gathered_keys: torch.Tensor,
        shard: int,
        num_requests: int,
        num_positions: int,
        anchor_ids: torch.Tensor,
        gate: torch.Tensor,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        top_k: int,
        num_ranks: int,
        rank_block: int = RANK_BLOCK,
        num_warps: int = NUM_WARPS,
        values: torch.Tensor | None = None,
    ):
        rows = num_requests * num_positions
        halves = gathered_keys.dtype == torch.float32
        num_keys = gathered_keys.shape[1] // (2 * num_ranks) if halves else gathered_keys.shape[1]
        rank_dim = predecessor_codebook.shape[1]
        device = gathered_keys.device
        ids = torch.empty((num_requests, num_positions, top_k), dtype=torch.int64, device=device)
        unary = torch.empty(
            (num_requests, num_positions, top_k), dtype=torch.float32, device=device
        )
        preds = torch.empty(
            (num_requests, num_positions, top_k, rank_dim), dtype=torch.float32, device=device
        )
        succs = torch.empty_like(preds)
        grid = (num_requests, num_positions, triton.cdiv(rank_dim, rank_block))
        _draft_topk_merge_kernel[grid](
            gathered_keys,
            anchor_ids,
            gate,
            predecessor_codebook,
            successor_codebook,
            ids,
            unary,
            preds,
            succs,
            unary if values is None else values,
            rows,
            num_ranks,
            num_keys,
            shard,
            num_positions,
            rank_dim,
            gathered_keys.stride(0),
            gate.stride(0),
            gate.stride(1),
            predecessor_codebook.stride(0),
            successor_codebook.stride(0),
            RANKS_PAD=triton.next_power_of_2(num_ranks),
            KEYS_PAD=triton.next_power_of_2(num_keys),
            TOP_K=top_k,
            SLOTS=SORT_SLOTS,
            RANK_BLOCK=rank_block,
            HALVES=halves,
            TRANSFORM=values is not None,
            num_warps=num_warps,
        )
        return ids, unary, preds, succs

    def reference(
        self,
        gathered_keys: torch.Tensor,
        shard: int,
        num_requests: int,
        num_positions: int,
        anchor_ids: torch.Tensor,
        gate: torch.Tensor,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        top_k: int,
        num_ranks: int,
        rank_block: int = RANK_BLOCK,
        num_warps: int = NUM_WARPS,
        values: torch.Tensor | None = None,
    ):
        """``torch.topk`` over the candidates scattered into a -inf vocabulary row.

        ``rank_block`` and ``num_warps`` (launch geometry) are ignored.

        Every element outside the candidate lists is below the k-th value or
        tied with it at a higher id, so replacing it with -inf changes neither
        the selection nor the arrangement torch sorts. With ``values`` the
        candidates carry the decoded canonical values, which order and tie as
        the transformed values do; each selected value is then replaced by
        ``values`` at its key16.
        """
        rows = num_requests * num_positions
        ranks = num_ranks
        if gathered_keys.dtype == torch.float32:
            gathered_keys = unpack_halves(gathered_keys, ranks)
        vocab_size = ranks * shard
        vocab, decoded = _decode_keys(gathered_keys, ranks, shard)
        dense = torch.full((rows, vocab_size), float("-inf"), device=gathered_keys.device)
        row_index = torch.arange(rows, device=gathered_keys.device)[:, None].expand_as(vocab[:rows])
        for r in range(ranks):
            ids_r, vals_r = vocab[r * rows : (r + 1) * rows], decoded[r * rows : (r + 1) * rows]
            keep = ids_r >= 0
            dense[row_index[keep], ids_r[keep]] = vals_r[keep]
        unary, ids = torch.topk(dense, top_k, dim=-1)
        if values is not None:
            # Decoded canonical values are bf16-exact: their key16 indexes the table.
            key16 = radix_key16(unary.to(torch.bfloat16).view(torch.int16))
            unary = values.to(unary.device)[key16.long()]
        ids = ids.view(num_requests, num_positions, top_k)
        unary = unary.view(num_requests, num_positions, top_k)
        successors = successor_codebook[ids].float()
        predecessor_ids = torch.cat(
            (anchor_ids.long()[:, None, None].expand(-1, 1, top_k), ids[:, :-1]), dim=1
        )
        predecessors = predecessor_codebook[predecessor_ids].float() * gate.float()[:, :, None]
        return ids, unary, predecessors, successors

    def is_valid(
        self,
        gathered_keys: torch.Tensor,
        shard: int,
        num_requests: int,
        num_positions: int,
        anchor_ids: torch.Tensor,
        gate: torch.Tensor,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        top_k: int,
        num_ranks: int,
        rank_block: int = RANK_BLOCK,
        num_warps: int = NUM_WARPS,
        values: torch.Tensor | None = None,
        **unused,
    ) -> None:
        if values is not None and (
            values.dtype != torch.float32
            or tuple(values.shape) != (CANON_ENTRIES,)
            or not values.is_contiguous()
            or values.device != gathered_keys.device
        ):
            raise ValueError(
                f"values must be a contiguous [{CANON_ENTRIES}] fp32 table on the keys' device"
            )
        rows = num_requests * num_positions
        ranks = num_ranks
        if gathered_keys.dtype == torch.int32:
            if gathered_keys.shape[0] != ranks * rows:
                raise ValueError("int32 gathered_keys must be [num_ranks * B * K, n]")
            keys_per_rank = gathered_keys.shape[1]
        elif gathered_keys.dtype == torch.float32:
            if gathered_keys.shape[0] != rows or gathered_keys.shape[1] % (2 * ranks):
                raise ValueError("fp32 gathered_keys must be [B * K, num_ranks * n * 2]")
            keys_per_rank = gathered_keys.shape[1] // (2 * ranks)
        else:
            raise ValueError(f"gathered_keys must be int32 or fp32, got {gathered_keys.dtype}")
        if gathered_keys.stride(-1) != 1:
            raise ValueError("gathered_keys must have unit column stride")
        if predecessor_codebook.shape[0] != ranks * shard:
            raise ValueError(
                f"codebook vocab {predecessor_codebook.shape[0]} != ranks {ranks} * shard {shard}"
            )
        if top_k > SORT_SLOTS or top_k & (top_k - 1):
            raise ValueError(f"top_k {top_k} must be a power of two <= {SORT_SLOTS}")
        if keys_per_rank < top_k:
            raise ValueError("each rank must contribute at least top_k keys per row")
        for name, cb in (("predecessor", predecessor_codebook), ("successor", successor_codebook)):
            if cb.dtype != torch.bfloat16 or cb.stride(-1) != 1:
                raise ValueError(f"{name}_codebook must be bf16 with unit column stride")
        if gate.dtype != torch.bfloat16 or gate.stride(-1) != 1:
            raise ValueError("gate must be bf16 with unit column stride")
        if tuple(gate.shape) != (num_requests, num_positions, predecessor_codebook.shape[1]):
            raise ValueError(f"gate shape {tuple(gate.shape)} does not match [B, K, rank]")
