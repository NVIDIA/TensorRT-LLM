# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram embedding for decode-shaped batches: id history, n-gram hash,
table gather and per-tensor fp8 dequantization in one Triton launch, with the
per-slot n-gram context published back in the same pass.

Tier 5 (Triton written for this target). It replaces, on the decode path,
the composition ``ngram_token_history`` (~20 launches) + ``ngram_hash`` (~30
launches) + ``nn.functional.embedding`` + ``dequantize_e4m3_per_tensor`` --
nothing registered under ``torch.ops.trtllm`` or in the installed FlashInfer hashes
n-grams or gathers-and-dequantizes, and each of those entries is launch-bound
at decode (0.3-0.4 ms eager each).

One program per **sequence**, which loops over that sequence's rows: the carry
read and the publish of one slot happen in one program, so no other program can
observe a half-published row.

Context-shaped batches (more than ``PER_SEQ_MAX_ROWS`` rows per sequence) take
a second form: one program per 32 packed rows, each row binary-searching
``query_start_loc`` for its sequence, then a one-program-per-sequence publish
launched after it -- two launches, bitwise equal to the first form.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..embedding.ngram_hash import ngram_hash
from .ngram_token_history import ngram_token_history

#: This checkpoint's PLE geometry: 3-gram, 16 heads of 160 fp8 columns.
CKPT_NGRAM_SIZE = 3
CKPT_HEAD_DIM = 160
CKPT_TABLE_ROWS = 320001536
CKPT_EOS_TOKEN_ID = 248044
CKPT_SCALE = 0.00019931793212890625
#: Largest `ple_ngram` pool seen for this checkpoint (see `ngram_token_history`).
ENGINE_SLOTS = 461
#: Above this many rows per sequence (MTP verify carries at most 4) the call
#: takes the context-shaped path: packed row tiles plus a separate publish.
PER_SEQ_MAX_ROWS = 4
_ROWS_BLOCK_T = 32
#: The row-to-sequence search covers at least 2^8 sequences, so one compiled
#: variant serves every batch up to that size.
_MIN_LOG_S = 8


@triton.jit
def _ple_ngram_embed_kernel(
    ids_ptr,
    pool_ptr,
    slots_ptr,
    carry_ptr,
    qsl_ptr,
    table_ptr,
    scale_ptr,
    mult_ptr,
    vocab_ptr,
    offs_ptr,
    out_ptr,
    stride_pool,
    stride_table,
    stride_out,
    eos,
    NGRAM: tl.constexpr,
    HEADS_PER: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_CARRY: tl.constexpr,
):
    CTX: tl.constexpr = NGRAM - 1
    s = tl.program_id(0)
    begin = tl.load(qsl_ptr + s).to(tl.int64)
    length = tl.load(qsl_ptr + s + 1).to(tl.int64) - begin
    slot = tl.load(slots_ptr + s).to(tl.int64)
    pool_row = pool_ptr + slot * stride_pool
    eos64 = tl.full([BLOCK_P], eos, tl.int64)
    if HAS_CARRY:
        fresh = tl.load(carry_ptr + s) == 0

    t = tl.arange(0, BLOCK_P)
    tmask = t < length
    heads = tl.arange(0, BLOCK_H)
    hmask = heads < HEADS
    order = heads // HEADS_PER + 2
    vocab = tl.load(vocab_ptr + heads, mask=hmask, other=1)
    offs = tl.load(offs_ptr + heads, mask=hmask, other=0)

    # Shift 0 is the token itself; shift k reads `k` back in the sequence's own
    # stream, i.e. the packed ids or, before the slice starts, the carry row.
    # Once a shift lands on an EOS, every deeper shift reads EOS: the hash's
    # segment starts one past the most recent EOS strictly before the token.
    mix = tl.load(ids_ptr + begin + t, mask=tmask, other=0).to(tl.int64) * tl.load(mult_ptr)
    mixed = tl.zeros([BLOCK_P, BLOCK_H], dtype=tl.int64)
    hit = t < 0
    for k in tl.static_range(1, NGRAM):
        pos = t - k
        from_ids = tl.load(ids_ptr + begin + pos, mask=tmask & (pos >= 0), other=0).to(tl.int64)
        from_pool = tl.load(pool_row + (CTX + pos), mask=tmask & (pos < 0), other=0)
        if HAS_CARRY:
            from_pool = tl.where(fresh, eos64, from_pool)
        ck = tl.where(pos >= 0, from_ids, from_pool)
        sk = tl.where(hit, eos64, ck)
        hit = hit | (ck == eos64)
        mix = mix ^ (sk * tl.load(mult_ptr + k))
        mixed = tl.where((order == k + 1)[None, :], mix[:, None], mixed)

    rem = mixed % vocab[None, :]
    rows = tl.where(rem < 0, rem + vocab[None, :], rem) + offs[None, :]

    d = tl.arange(0, BLOCK_D)
    m3 = tmask[:, None, None] & hmask[None, :, None] & (d < HEAD_DIM)[None, None, :]
    q = tl.load(table_ptr + rows[:, :, None] * stride_table + d[None, None, :], mask=m3, other=0.0)
    val = q.to(tl.float32) * tl.load(scale_ptr).to(tl.float32)
    cols = heads[None, :, None] * HEAD_DIM + d[None, None, :]
    tl.store(
        out_ptr + (begin + t)[:, None, None] * stride_out + cols,
        val.to(out_ptr.dtype.element_ty),
        mask=m3,
    )

    # Publish the last CTX ids of `carry + slice` -- after every thread's carry
    # read, since the row being overwritten is the one just read.
    c = tl.arange(0, BLOCK_C)
    cmask = c < CTX
    src = length + c
    from_slice = tl.load(ids_ptr + begin + src - CTX, mask=cmask & (src >= CTX), other=0).to(
        tl.int64
    )
    old = tl.load(pool_row + src, mask=cmask & (src < CTX), other=0)
    if HAS_CARRY:
        old = tl.where(fresh, tl.full([BLOCK_C], eos, tl.int64), old)
    published = tl.where(src >= CTX, from_slice, old)
    tl.debug_barrier()
    tl.store(pool_row + c, published, mask=cmask)


@triton.jit
def _seq_of_rows(qsl_ptr, t, num_seqs, LOG_S: tl.constexpr):
    """Per packed row `t`: the largest `s < num_seqs` with `qsl[s] <= t`, i.e.
    the sequence that owns the row (zero-row sequences are skipped over)."""
    s = tl.zeros_like(t)
    for i in tl.static_range(LOG_S):
        cand = s + (1 << (LOG_S - 1 - i))
        ok = cand < num_seqs
        start = tl.load(qsl_ptr + cand, mask=ok, other=0)
        s = tl.where(ok & (start <= t), cand, s)
    return s


@triton.jit
def _ple_ngram_embed_rows_kernel(
    ids_ptr,
    pool_ptr,
    slots_ptr,
    carry_ptr,
    qsl_ptr,
    table_ptr,
    scale_ptr,
    mult_ptr,
    vocab_ptr,
    offs_ptr,
    out_ptr,
    num_rows,
    num_seqs,
    stride_pool,
    stride_table,
    stride_out,
    eos,
    NGRAM: tl.constexpr,
    HEADS_PER: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    LOG_S: tl.constexpr,
    HAS_CARRY: tl.constexpr,
):
    """Context-shaped batches: one program per BLOCK_T packed rows, each row
    finding its sequence in `query_start_loc`. Window, hash, gather and
    dequantize only; the publish is `_ple_ngram_publish_kernel`, launched
    after, because the rows that read a slot's carry may span two programs."""
    CTX: tl.constexpr = NGRAM - 1
    t = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < num_rows
    s = _seq_of_rows(qsl_ptr, t, num_seqs, LOG_S)
    begin = tl.load(qsl_ptr + s, mask=tmask, other=0).to(tl.int64)
    tpos = t - begin
    slot = tl.load(slots_ptr + s, mask=tmask, other=0).to(tl.int64)
    eos64 = tl.full([BLOCK_T], eos, tl.int64)
    if HAS_CARRY:
        fresh = tl.load(carry_ptr + s, mask=tmask, other=1) == 0

    heads = tl.arange(0, BLOCK_H)
    hmask = heads < HEADS
    order = heads // HEADS_PER + 2
    vocab = tl.load(vocab_ptr + heads, mask=hmask, other=1)
    offs = tl.load(offs_ptr + heads, mask=hmask, other=0)

    mix = tl.load(ids_ptr + t, mask=tmask, other=0).to(tl.int64) * tl.load(mult_ptr)
    mixed = tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.int64)
    hit = tmask & (t < 0)
    for k in tl.static_range(1, NGRAM):
        pos = tpos - k
        from_ids = tl.load(ids_ptr + t - k, mask=tmask & (pos >= 0), other=0).to(tl.int64)
        from_pool = tl.load(
            pool_ptr + slot * stride_pool + (CTX + pos), mask=tmask & (pos < 0), other=0
        )
        if HAS_CARRY:
            from_pool = tl.where(fresh, eos64, from_pool)
        ck = tl.where(pos >= 0, from_ids, from_pool)
        sk = tl.where(hit, eos64, ck)
        hit = hit | (ck == eos64)
        mix = mix ^ (sk * tl.load(mult_ptr + k))
        mixed = tl.where((order == k + 1)[None, :], mix[:, None], mixed)

    rem = mixed % vocab[None, :]
    rows = tl.where(rem < 0, rem + vocab[None, :], rem) + offs[None, :]

    d = tl.arange(0, BLOCK_D)
    dmask = d < HEAD_DIM
    m2 = tmask[:, None] & dmask[None, :]
    scale = tl.load(scale_ptr).to(tl.float32)
    for h in tl.static_range(HEADS):
        row_h = tl.sum(tl.where((heads == h)[None, :], rows, 0), axis=1)
        q = tl.load(table_ptr + row_h[:, None] * stride_table + d[None, :], mask=m2, other=0.0)
        val = q.to(tl.float32) * scale
        tl.store(
            out_ptr + t[:, None] * stride_out + (h * HEAD_DIM + d)[None, :],
            val.to(out_ptr.dtype.element_ty),
            mask=m2,
        )


@triton.jit
def _ple_ngram_publish_kernel(
    ids_ptr,
    pool_ptr,
    slots_ptr,
    carry_ptr,
    qsl_ptr,
    stride_pool,
    eos,
    NGRAM: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_CARRY: tl.constexpr,
):
    """One program per sequence: the last `NGRAM - 1` ids of `carry + slice`
    published to its slot, after every row's carry read (stream order)."""
    CTX: tl.constexpr = NGRAM - 1
    s = tl.program_id(0)
    begin = tl.load(qsl_ptr + s).to(tl.int64)
    length = tl.load(qsl_ptr + s + 1).to(tl.int64) - begin
    slot = tl.load(slots_ptr + s).to(tl.int64)
    pool_row = pool_ptr + slot * stride_pool
    c = tl.arange(0, BLOCK_C)
    cmask = c < CTX
    src = length + c
    from_slice = tl.load(ids_ptr + begin + src - CTX, mask=cmask & (src >= CTX), other=0).to(
        tl.int64
    )
    old = tl.load(pool_row + src, mask=cmask & (src < CTX), other=0)
    if HAS_CARRY:
        fresh = tl.load(carry_ptr + s) == 0
        old = tl.where(fresh, tl.full([BLOCK_C], eos, tl.int64), old)
    published = tl.where(src >= CTX, from_slice, old)
    tl.debug_barrier()
    tl.store(pool_row + c, published, mask=cmask)


def _cell(why: str, lengths, **spec) -> Cell:
    return Cell(why=why, spec=dict(lengths=tuple(lengths), **spec))


class PleNgramEmbed(OpWrapper):
    """For sequence `s` over packed rows `[b, e)` with `slot = state_indices[s]`,
    its id stream is `carry + input_ids[b:e]`, where `carry` is
    `context_pool[slot]` (oldest first) when `has_initial_states[s]` -- or
    always, when `has_initial_states` is None -- and `ngram_size - 1` copies of
    `eos_token_id` when not. Row `t` of that slice gets the window
    `stream[t : t + ngram_size]` (newest last), exactly `ngram_token_history`'s;
    each window hashes to one global row per head exactly as `ngram_hash` with
    `num_query_tokens=1`; and::

        out[t, h * D : (h + 1) * D] = to(scale.dtype)(float(table[row[t, h]]) * float(scale))

    with `D = table.shape[1]` -- `dequantize_e4m3_per_tensor`'s one rounding.
    Then, **in place**, `context_pool[slot]` becomes the last `ngram_size - 1`
    ids of the stream (a zero-row sequence republishes its carry). Unnamed slots
    are untouched.

    Returns a fresh contiguous `[num_tokens, heads * D]` in `scale`'s dtype.
    `context_pool` is the only argument modified.

    Fusion boundary: carry read and seed, window, hash, gather, dequantize,
    publish. The key/value projections that consume the result are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: lengths (rows per sequence), pool_slots (default num_seqs + 2),
    # carries (None -> the argument is None), max_rows (default max(lengths)),
    # eos_density (0.0), table "small" | "host" | "ckpt" (the full
    # 320001536-row device table, rows filled where touched). The checkpoint's
    # 3-gram of 8 heads per order, int32 ids.
    CELLS: tuple[Cell, ...] = (
        _cell(
            "decode 128 x 1 row, host-resident table, 461 slots",
            (1,) * 128,
            pool_slots=ENGINE_SLOTS,
            table="host",
        ),
        _cell(
            "decode 128 x 1 row, the full checkpoint-height table",
            (1,) * 128,
            pool_slots=ENGINE_SLOTS,
            table="ckpt",
            eos_density=0.05,
        ),
        # Context-shaped (prefill / mixed) steps: the packed-row form.
        _cell(
            "mixed step: 2 contexts + 64 decode rows",
            (3001, 5000) + (1,) * 64,
            carries=(False, True) + (True,) * 64,
            pool_slots=ENGINE_SLOTS,
            eos_density=0.05,
        ),
    )

    note = """
    Metadata consumed: none -- batch layout, pool, slot map are explicit.
    Call site: the decode target's PLE block, once per forward:
    `input_ids` int32 `[T]`, the whole `[slots, 2]` int64 n-gram pool,
    int32 `state_indices[:batch]`, `has_initial_states=None` (every decode
    sequence carries), int64 `query_start_loc_long` (or the
    `cu_seqlens_from_seq_lens` rebuild under MTP verify), the
    `[320001536, 160]` fp8 table (host-resident by default), the `[1, 1, 1]`
    bf16 scale, `max_rows = gen_p`.

    Preconditions (data; not guarded -- each would need a sync):
    - every sequence has at most `max_rows` rows; rows past it are neither
      written nor published;
    - slots real (`0 <= slot < slots`) and distinct; -1 is not supported;
    - `query_start_loc[-1] == num_tokens`;
    - token id * max multiplier < 2^63 (int64 wraps, as in `ngram_hash`).

    Certified at a 128 x 1-row decode on a host-resident table and on the
    full checkpoint-height device table, and a mixed step of 2 contexts (one
    fresh, one carrying) + 64 decode rows on the packed-row form. Not
    certified: more than one row per sequence on the per-sequence path
    (`1 < max_rows <= PER_SEQ_MAX_ROWS`) or zero-row sequences -- a verify
    step takes the unfused ops, so the target's per-sequence path runs one
    row.

    Bitwise equal to `ngram_token_history` + `ngram_hash` + embedding +
    `dequantize_e4m3_per_tensor`, window, rows, values and pool alike.
    CUDA-graph capturable: no host sync, one launch, output allocated per call.
    Reads the table through its device address, so a host-mapped table needs
    no copy.
    """

    def __call__(
        self,
        input_ids: torch.Tensor,
        context_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: Optional[torch.Tensor],
        query_start_loc: torch.Tensor,
        table: torch.Tensor,
        scale: torch.Tensor,
        layer_multipliers: torch.Tensor,
        head_vocab_sizes: torch.Tensor,
        head_offsets: torch.Tensor,
        eos_token_id: int,
        max_rows: int,
    ) -> torch.Tensor:
        num_tokens = input_ids.shape[0]
        num_seqs = state_indices.shape[0]
        ngram_size = layer_multipliers.numel()
        heads = head_vocab_sizes.numel()
        head_dim = table.shape[1]
        out = torch.empty(
            (num_tokens, heads * head_dim), dtype=scale.dtype, device=input_ids.device
        )
        if num_seqs > 0 and max_rows > PER_SEQ_MAX_ROWS:
            carry = has_initial_states if has_initial_states is not None else state_indices
            if num_tokens > 0:
                _ple_ngram_embed_rows_kernel[(triton.cdiv(num_tokens, _ROWS_BLOCK_T),)](
                    input_ids,
                    context_pool,
                    state_indices,
                    carry,
                    query_start_loc,
                    table,
                    scale,
                    layer_multipliers,
                    head_vocab_sizes,
                    head_offsets,
                    out,
                    num_tokens,
                    num_seqs,
                    context_pool.stride(0),
                    table.stride(0),
                    out.stride(0),
                    eos_token_id,
                    NGRAM=ngram_size,
                    HEADS_PER=heads // (ngram_size - 1),
                    HEADS=heads,
                    BLOCK_H=triton.next_power_of_2(heads),
                    HEAD_DIM=head_dim,
                    BLOCK_D=triton.next_power_of_2(head_dim),
                    BLOCK_T=_ROWS_BLOCK_T,
                    LOG_S=max(_MIN_LOG_S, (num_seqs - 1).bit_length()),
                    HAS_CARRY=has_initial_states is not None,
                    num_warps=4,
                )
            _ple_ngram_publish_kernel[(num_seqs,)](
                input_ids,
                context_pool,
                state_indices,
                carry,
                query_start_loc,
                context_pool.stride(0),
                eos_token_id,
                NGRAM=ngram_size,
                BLOCK_C=triton.next_power_of_2(ngram_size - 1),
                HAS_CARRY=has_initial_states is not None,
                num_warps=1,
            )
        elif num_seqs > 0:
            _ple_ngram_embed_kernel[(num_seqs,)](
                input_ids,
                context_pool,
                state_indices,
                has_initial_states if has_initial_states is not None else state_indices,
                query_start_loc,
                table,
                scale,
                layer_multipliers,
                head_vocab_sizes,
                head_offsets,
                out,
                context_pool.stride(0),
                table.stride(0),
                out.stride(0),
                eos_token_id,
                NGRAM=ngram_size,
                HEADS_PER=heads // (ngram_size - 1),
                HEADS=heads,
                BLOCK_H=triton.next_power_of_2(heads),
                HEAD_DIM=head_dim,
                BLOCK_D=triton.next_power_of_2(head_dim),
                BLOCK_P=triton.next_power_of_2(max(max_rows, 1)),
                BLOCK_C=triton.next_power_of_2(ngram_size - 1),
                HAS_CARRY=has_initial_states is not None,
                num_warps=4,
            )
        return out

    def reference(
        self,
        input_ids: torch.Tensor,
        context_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: Optional[torch.Tensor],
        query_start_loc: torch.Tensor,
        table: torch.Tensor,
        scale: torch.Tensor,
        layer_multipliers: torch.Tensor,
        head_vocab_sizes: torch.Tensor,
        head_offsets: torch.Tensor,
        eos_token_id: int,
        max_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, context_pool)` as the call leaves them, from the scalar
        references of the two entries this replaces -- `ngram_token_history`
        (window and published pool) and `ngram_hash` (rows) -- and a float64
        gather-and-scale rounded once. Mutates nothing."""
        carries = (
            torch.ones_like(state_indices, dtype=torch.bool)
            if has_initial_states is None
            else has_initial_states
        )
        ngram_size = layer_multipliers.numel()
        window, pool = ngram_token_history.reference(
            input_ids,
            context_pool,
            state_indices,
            carries,
            query_start_loc,
            eos_token_id,
            ngram_size,
        )
        rows = ngram_hash.reference(
            window, layer_multipliers, head_vocab_sizes, head_offsets, eos_token_id, 1
        ).reshape(window.shape[0], -1)
        gathered = table[rows.reshape(-1)].double().reshape(rows.shape[0], -1)
        out = (gathered * scale.double().reshape(())).to(scale.dtype)
        return out, pool

    def is_valid(
        self,
        input_ids: torch.Tensor,
        context_pool: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: Optional[torch.Tensor],
        query_start_loc: torch.Tensor,
        table: torch.Tensor,
        scale: torch.Tensor,
        layer_multipliers: torch.Tensor,
        head_vocab_sizes: torch.Tensor,
        head_offsets: torch.Tensor,
        eos_token_id: int,
        max_rows: int,
    ) -> None:
        num_seqs = state_indices.shape[0]
        ngram_size = layer_multipliers.numel()
        assert input_ids.dim() == 1 and input_ids.is_contiguous(), "input_ids [T], contiguous"
        assert ngram_size >= 2 and layer_multipliers.dtype == torch.int64, (
            "layer_multipliers int64, at least two orders"
        )
        assert context_pool.dtype == torch.int64 and context_pool.stride(1) == 1, (
            "context_pool int64 with unit column stride"
        )
        assert context_pool.shape[1] == ngram_size - 1, (
            f"context_pool must be ngram_size - 1 = {ngram_size - 1} wide"
        )
        assert query_start_loc.shape[0] == num_seqs + 1, (
            "query_start_loc must be [num_seqs + 1]; a short one misattributes rows"
        )
        assert has_initial_states is None or has_initial_states.shape[0] == num_seqs, (
            "has_initial_states must be one flag per sequence"
        )
        assert head_vocab_sizes.numel() % (ngram_size - 1) == 0, (
            "heads must split evenly over the n-gram orders"
        )
        assert head_offsets.numel() == head_vocab_sizes.numel(), "one offset per head"
        assert head_vocab_sizes.dtype == head_offsets.dtype == torch.int64, "int64 vocab/offsets"
        assert table.dim() == 2 and table.stride(1) == 1, "table [rows, D], unit column stride"
        assert scale.numel() == 1, "one per-tensor scale"
        assert max_rows >= 1, "max_rows >= 1"


ple_ngram_embed = PleNgramEmbed()
