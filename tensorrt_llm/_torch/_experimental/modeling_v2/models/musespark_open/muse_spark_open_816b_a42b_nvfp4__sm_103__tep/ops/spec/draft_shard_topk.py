# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-rank top-k of the drafter's vocab-sharded logits, as 32-bit sort keys.

The DFlash 2 selector needs each block position's top ``selector_top_k``
candidates over the full vocabulary, in exactly the order ``torch.topk`` gives
them. The draft LM head is column-parallel, so each rank holds a contiguous
slice of the vocabulary. This entry reduces a rank's ``[M, shard]`` bf16 slice
to the top ``top_k`` of every ``chunk`` columns, encoded so that the merge
after the all-gather (``draft_topk_merge``) needs nothing else.

The order is ``torch.topk``'s: fp32 values are ranked by torch's radix key
(``TopKTypeConfig<float>::convert``: sign-flipped bits, NaN canonicalized to
the top, so -0.0 ranks below +0.0) and ties go to the lower column. A bf16
value widens to fp32 exactly, and its fp32 radix key is its 16-bit radix key
followed by 16 constant bits, so the 16-bit key orders the slice exactly as
torch orders the ``.float()`` logits. Each element becomes

    key = ((key16 ^ 0x8000) << 16) | (0xFFFF - local_column)        (int32)

whose signed order is (value desc, column asc) and which is unique per element.
``INT32_MIN`` marks an empty slot; no element encodes to it.

With ``canon`` (the canonical-key table of a drafter logit transform ``t``,
``kernels/draft_head/logit_transform.py``) the keys order the slice as
``torch.topk`` orders the transformed fp32 logits ``t(x)``. Every key16 outside
the band ``canon_band = (lo, hi)`` is replaced by ``canon[key16]``, the key16 of
its tie group's representative (a tie group is the bf16 values whose
transformed values have the same bits). ``t`` never decreases, so a tie group
is a run of consecutive key16 values and ``canon`` is non-decreasing: two
canonical keys compare as the radix keys of the transformed values do, and
equal ones tie on the column as torch's do. Inside the band every group is a
single value and ``canon`` is the identity, so those elements load nothing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The draft head this target runs: 202048 / 8 vocabulary rows per rank, K + 1
#: = 8 block positions with K <= 7 drafted, top 16 candidates.
SHARD = 202048 // 8
TOP_K = 16
#: Columns per program and its warps. 4096 bf16 columns are 4 per thread at 32
#: warps; seven programs cover a shard.
CHUNK = 4096
NUM_WARPS = 32

#: The empty-slot key.
EMPTY_KEY = -(2**31)

#: Entries of a canonical-key table: one per 16-bit radix key.
CANON_ENTRIES = 1 << 16

#: The band of a call without a table: every key16 lies inside it.
FULL_BAND = (0, CANON_ENTRIES - 1)

#: The non-NaN 16-bit radix keys, -inf (0x007F) to +inf (0xFF80).
FINITE_KEYS = (0x007F, 0xFF80)


@triton.jit
def _draft_shard_topk_kernel(
    bits_ptr,
    keys_ptr,
    canon_ptr,
    shard,
    bits_row_stride,
    keys_row_stride,
    rank,
    num_ranks,
    keys_per_rank,
    canon_lo,
    canon_hi,
    CHUNK: tl.constexpr,
    TOP_K: tl.constexpr,
    HALVES: tl.constexpr,
    CANON: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    local = chunk * CHUNK + tl.arange(0, CHUNK)
    in_shard = local < shard
    bits = tl.load(bits_ptr + row.to(tl.int64) * bits_row_stride + local, mask=in_shard, other=0)
    bits = bits.to(tl.int32) & 0xFFFF
    key16 = tl.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits ^ 0x8000)
    is_nan = ((bits & 0x7F80) == 0x7F80) & ((bits & 0x7F) != 0)
    key16 = tl.where(is_nan, 0xFFFF, key16)
    if CANON:
        # Only the keys outside the identity band read the table.
        far = ((key16 < canon_lo) | (key16 > canon_hi)) & in_shard
        key16 = tl.where(far, tl.load(canon_ptr + key16, mask=far, other=0), key16)
    key = ((key16 ^ 0x8000) << 16) | (0xFFFF - local)
    key = tl.where(in_shard, key, -2147483648)

    slots = tl.arange(0, TOP_K)
    out = tl.full([TOP_K], -2147483648, tl.int32)
    for j in tl.static_range(TOP_K):
        best = tl.max(key, axis=0)
        out = tl.where(slots == j, best, out)
        key = tl.where(key == best, -2147483648, key)
    if HALVES:
        # The all-reduce layout: this rank's slot holds each key as two exact
        # 16-bit halves in fp32; every other rank's slot is +0.0.
        base = keys_ptr + row.to(tl.int64) * keys_row_stride
        col = (rank * keys_per_rank + chunk * TOP_K + slots) * 2
        tl.store(base + col, ((out >> 16) & 0xFFFF).to(tl.float32))
        tl.store(base + col + 1, (out & 0xFFFF).to(tl.float32))
        zeros = tl.zeros([2 * TOP_K], tl.float32)
        span = tl.arange(0, 2 * TOP_K)
        for other in range(num_ranks):
            if other != rank:
                tl.store(base + (other * keys_per_rank + chunk * TOP_K) * 2 + span, zeros)
    else:
        tl.store(keys_ptr + row.to(tl.int64) * keys_row_stride + chunk * TOP_K + slots, out)


def radix_key16(bits: torch.Tensor) -> torch.Tensor:
    """torch's radix key of bf16 bit patterns, as int32 in [0, 0xFFFF]; every NaN is 0xFFFF."""
    b = bits.to(torch.int32) & 0xFFFF
    key = torch.where((b & 0x8000) != 0, b ^ 0xFFFF, b ^ 0x8000)
    is_nan = ((b & 0x7F80) == 0x7F80) & ((b & 0x7F) != 0)
    return torch.where(is_nan, torch.full_like(key, 0xFFFF), key)


def canon_reason(canon: torch.Tensor, canon_band: tuple[int, int], device) -> str | None:
    """Why ``canon`` is not a canonical-key table the kernels read correctly, or None.

    A table is ``[65536]`` int32, contiguous, on ``device``; holds 16-bit keys;
    is non-decreasing over the non-NaN keys (``FINITE_KEYS``), so it keeps
    their order; is the identity on ``canon_band`` (the kernels skip the load
    there); and maps 0xFFFF (every NaN) to itself.
    """
    if canon.dtype != torch.int32 or tuple(canon.shape) != (CANON_ENTRIES,):
        return f"canon must be [{CANON_ENTRIES}] int32, got {canon.dtype} {tuple(canon.shape)}"
    if not canon.is_contiguous() or canon.device != torch.device(device):
        return "canon must be contiguous and on the logits' device"
    lo, hi = canon_band
    if not 0 <= lo <= hi < CANON_ENTRIES:
        return f"canon_band {canon_band} is not an interval of 16-bit keys"
    table = canon.cpu()
    if int(table.min()) < 0 or int(table.max()) >= CANON_ENTRIES:
        return "canon holds a value outside the 16-bit keys"
    ordered = table[FINITE_KEYS[0] : FINITE_KEYS[1] + 1]
    if bool((ordered[1:] < ordered[:-1]).any()):
        return "canon is not non-decreasing over the non-NaN keys"
    if not torch.equal(table[lo : hi + 1], torch.arange(lo, hi + 1, dtype=torch.int32)):
        return f"canon is not the identity on its band {tuple(canon_band)}"
    if int(table[CANON_ENTRIES - 1]) != CANON_ENTRIES - 1:
        return "canon must map 0xFFFF (NaN) to itself"
    return None


class DraftShardTopK(OpWrapper):
    """Top ``top_k`` sort keys of every ``chunk`` columns of a bf16 logits slice.

        logits      [M, shard] bf16, stride(-1) == 1, row stride free; shard <= 65536
        top_k       candidates kept per chunk (a power of two)
        chunk       columns per program (a power of two >= top_k)
        num_warps
        rank, num_ranks
                    with ``rank`` set, the all-reduce layout (see below)
        canon       None, or a canonical-key table ``[65536]`` int32 (``canon_reason``)
        canon_band  ``(lo, hi)``: the key16 interval on which ``canon`` is the identity

    Returns ``[M, ceil(shard / chunk) * top_k]`` int32: chunk ``c``'s keys at
    columns ``[c * top_k, (c + 1) * top_k)``, largest first, ``EMPTY_KEY`` where
    the chunk has fewer than ``top_k`` columns. With ``rank`` set, ``[M,
    num_ranks * ceil(shard / chunk) * top_k * 2]`` fp32: this rank's keys as
    (high, low) 16-bit halves at its slot, +0.0 in every other rank's. Integer
    selection: the output is exact, not approximate.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            "one request at TP8: 7 rows of a 25256-column shard, int32 keys for the all-gather",
            {"rows": 7, "shard": SHARD, "halves": False, "transform": None},
        ),
        Cell(
            "one request at TP8, fp32 halves for the MNNVL all-reduce",
            {"rows": 7, "shard": SHARD, "halves": True, "transform": None},
        ),
        Cell(
            "one request at TP8, fp32 halves, the Muse Spark Open drafter transform's canon",
            {"rows": 7, "shard": SHARD, "halves": True, "transform": "musespark"},
        ),
        Cell(
            "eight requests at TP4: a 50512-column shard, int32 keys, the drafter transform",
            {"rows": 56, "shard": 2 * SHARD, "halves": False, "transform": "musespark"},
        ),
        Cell(
            "24 requests at TP8, fp32 halves, the drafter transform",
            {"rows": 168, "shard": SHARD, "halves": True, "transform": "musespark"},
        ),
    )
    note = (
        "Used with draft_topk_merge, which decodes the keys; the pair is exact "
        "against torch.topk over the gathered fp32 logits, and with canon (and the "
        "merge's values table) against torch.topk over the drafter's transformed "
        "logits. NaN payloads are not carried (a NaN logit ranks first, as in "
        "torch, and decodes to the canonical NaN)."
    )

    def __call__(
        self,
        logits: torch.Tensor,
        top_k: int = TOP_K,
        chunk: int = CHUNK,
        num_warps: int = NUM_WARPS,
        rank: int | None = None,
        num_ranks: int = 1,
        canon: torch.Tensor | None = None,
        canon_band: tuple[int, int] = FULL_BAND,
    ) -> torch.Tensor:
        rows, shard = logits.shape
        chunks = triton.cdiv(shard, chunk)
        keys_per_rank = chunks * top_k
        halves = rank is not None
        if halves:
            keys = torch.empty(
                (rows, num_ranks * keys_per_rank * 2), dtype=torch.float32, device=logits.device
            )
        else:
            keys = torch.empty((rows, keys_per_rank), dtype=torch.int32, device=logits.device)
        bits = logits.view(torch.int16)
        lo, hi = canon_band
        _draft_shard_topk_kernel[(rows, chunks)](
            bits,
            keys,
            bits if canon is None else canon,
            shard,
            bits.stride(0),
            keys.stride(0),
            rank if halves else 0,
            num_ranks,
            keys_per_rank,
            lo,
            hi,
            CHUNK=chunk,
            TOP_K=top_k,
            HALVES=halves,
            CANON=canon is not None,
            num_warps=num_warps,
        )
        return keys

    def reference(
        self,
        logits: torch.Tensor,
        top_k: int = TOP_K,
        chunk: int = CHUNK,
        num_warps: int = NUM_WARPS,
        rank: int | None = None,
        num_ranks: int = 1,
        canon: torch.Tensor | None = None,
        canon_band: tuple[int, int] = FULL_BAND,
    ) -> torch.Tensor:
        """The same keys from torch ops; ``num_warps`` is ignored."""
        rows, shard = logits.shape
        local = torch.arange(shard, device=logits.device, dtype=torch.int64)
        key16 = radix_key16(logits.contiguous().view(torch.int16))
        if canon is not None:
            lo, hi = canon_band
            far = (key16 < lo) | (key16 > hi)
            key16 = torch.where(far, canon.to(key16.device)[key16.long()], key16)
        key16 = key16.to(torch.int64)
        # The unsigned 32-bit key: (value desc, column asc) as plain integer order.
        key32 = (key16 << 16) | (0xFFFF - local)
        out = []
        for start in range(0, shard, chunk):
            part = key32[:, start : start + chunk]
            k = min(top_k, part.shape[1])
            best = torch.topk(part, k, dim=-1).values
            # The kernel's encoding: top bit flipped, read as a signed int32.
            best = best ^ 0x80000000
            best = torch.where(best >= 2**31, best - 2**32, best)
            if k < top_k:
                pad = torch.full(
                    (rows, top_k - k), EMPTY_KEY, dtype=torch.int64, device=logits.device
                )
                best = torch.cat([best, pad], dim=-1)
            out.append(best)
        keys = torch.cat(out, dim=-1)
        if rank is None:
            return keys.to(torch.int32)
        unsigned = torch.where(keys < 0, keys + 2**32, keys)
        halves = torch.stack(((unsigned >> 16) & 0xFFFF, unsigned & 0xFFFF), dim=-1).float()
        packed = torch.zeros((rows, num_ranks, keys.shape[1], 2), device=logits.device)
        packed[:, rank] = halves
        return packed.view(rows, -1)

    def is_valid(
        self,
        logits: torch.Tensor,
        top_k: int = TOP_K,
        chunk: int = CHUNK,
        num_warps: int = NUM_WARPS,
        rank: int | None = None,
        num_ranks: int = 1,
        canon: torch.Tensor | None = None,
        canon_band: tuple[int, int] = FULL_BAND,
        **unused,
    ) -> None:
        if logits.dtype != torch.bfloat16 or logits.dim() != 2:
            raise ValueError(f"logits must be 2-D bf16, got {logits.dtype} {tuple(logits.shape)}")
        if logits.stride(-1) != 1:
            raise ValueError("logits must have unit column stride")
        if logits.shape[1] > 65536:
            raise ValueError(f"shard {logits.shape[1]} exceeds the 16-bit column field")
        if top_k & (top_k - 1) or chunk & (chunk - 1) or chunk < top_k:
            raise ValueError(
                f"top_k {top_k} and chunk {chunk} must be powers of two, chunk >= top_k"
            )
        if rank is not None and not 0 <= rank < num_ranks:
            raise ValueError(f"rank {rank} outside [0, {num_ranks})")
        if canon is not None:
            reason = canon_reason(canon, canon_band, logits.device)
            if reason is not None:
                raise ValueError(reason)
