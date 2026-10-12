# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-token-per-request Engram history and hash computation."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_tokens", "history_columns"])
def _decode_hash_kernel(
    input_ids,
    positions,
    rows,
    token_mask,
    lookup,
    history,
    multipliers,
    output,
    num_tokens,
    history_columns,
    VOCAB_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr,
    DEAD: tl.constexpr,
    MODULI: tl.constexpr,
    N_HEAD: tl.constexpr,
    MAX_NGRAM: tl.constexpr,
    WRITE_HISTORY: tl.constexpr,
    READ_CURRENT_HISTORY: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    token = tl.program_id(0)
    real = token < num_tokens
    if not READ_CURRENT_HISTORY:
        raw = tl.load(input_ids + token, real, other=0).to(tl.int64)
        compressed = tl.load(lookup + tl.minimum(tl.maximum(raw, 0), VOCAB_SIZE - 1))
        if HAS_MASK:
            text = tl.load(token_mask + token, real, other=False)
            compressed = tl.where(text, compressed, DEAD)
        # The general path scatters to int32 history before gathering into int64.
        compressed = compressed.to(tl.int32).to(tl.int64)
    position = tl.load(positions + token, real, other=0).to(tl.int64)
    row = tl.load(rows + token, real, other=0).to(tl.int64)
    if READ_CURRENT_HISTORY:
        # Aliases were scattered by the original Torch operation. All tokens
        # sharing a cell must hash its resolved value, not their own input ID.
        compressed = tl.load(
            history + row * history_columns + tl.maximum(position, 0),
            real & (position < history_columns),
            other=DEAD,
        ).to(tl.int64)
    valid_position = (position >= -history_columns) & (position < history_columns)
    if WRITE_HISTORY:
        # Preserve PyTorch's negative advanced-index semantics, and report bad
        # positions as indexing errors rather than accessing unrelated memory.
        tl.device_assert((~real) | valid_position, "Engram history position out of bounds")
        write_position = tl.where(position < 0, position + history_columns, position)
        tl.store(
            history + row * history_columns + write_position,
            compressed,
            real & valid_position,
        )
    blocked = (position < 0) | (compressed == DEAD)
    source = tl.where(blocked, PAD_ID, compressed)
    mix = source * tl.load(multipliers).to(tl.int64)
    heads = tl.arange(0, BLOCK_HEAD)
    for offset in tl.static_range(1, MAX_NGRAM):
        previous = position - offset
        source = tl.load(
            history + row * history_columns + tl.maximum(previous, 0),
            real & (previous >= 0) & (previous < history_columns),
            other=DEAD,
        ).to(tl.int64)
        blocked = blocked | (previous < 0) | (source == DEAD)
        source = tl.where(blocked, PAD_ID, source)
        mix = mix ^ (source * tl.load(multipliers + offset).to(tl.int64))
        modulus = tl.full((BLOCK_HEAD,), 1, tl.int64)
        for head in tl.static_range(N_HEAD):
            modulus = tl.where(heads == head, MODULI[offset - 1][head], modulus)
        # Triton uses signed C remainder; PyTorch uses nonnegative modulo for
        # positive divisors, including after signed int64 multiplication/XOR.
        remainder = mix % modulus
        hashed = tl.where(remainder < 0, remainder + modulus, remainder)
        output_offset = token * ((MAX_NGRAM - 1) * N_HEAD) + (offset - 1) * N_HEAD + heads
        tl.store(output + output_offset, tl.where(real, hashed, 0), heads < N_HEAD)
