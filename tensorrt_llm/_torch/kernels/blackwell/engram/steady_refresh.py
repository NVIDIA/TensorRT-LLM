# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engram device code used by the fused steady decode stage."""

import triton
import triton.language as tl


@triton.jit
def scatter_and_hash_decode_rows(
    cells,
    history_columns,
    position,
    compressed,
    window,
    multipliers,
    moduli,
    hashes,
    hash_rows,
    token,
    valid,
    pad_id,
    dead,
    N_HEAD: tl.constexpr,
    MAX_NGRAM: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_NGRAM: tl.constexpr,
    BLOCK_LAYERS: tl.constexpr,
    BLOCK_ORDERS: tl.constexpr,
):
    """Scatter this step's compressed IDs and hash each token's prepared n-gram window."""
    ks = tl.arange(0, BLOCK_NGRAM)
    heads = tl.arange(0, BLOCK_HEAD)
    in_heads = heads < N_HEAD
    layer_ids = tl.arange(0, BLOCK_LAYERS)
    mults = tl.load(
        multipliers + layer_ids[:, None] * MAX_NGRAM + ks[None, :],
        mask=(layer_ids[:, None] < NUM_LAYERS) & (ks[None, :] < MAX_NGRAM),
        other=0,
    )
    orders = tl.arange(0, BLOCK_ORDERS)
    mods = tl.load(
        moduli + orders[:, None] * N_HEAD + heads[None, :],
        mask=(orders[:, None] < NUM_LAYERS * (MAX_NGRAM - 1)) & in_heads[None, :],
        other=1,
    )
    target = tl.where(position < 0, position + history_columns, position)
    tl.store(
        cells + target,
        compressed.to(tl.int32),
        mask=valid & (target >= 0) & (target < history_columns),
    )
    width = (MAX_NGRAM - 1) * N_HEAD
    for layer in tl.static_range(NUM_LAYERS):
        current = tl.sum(tl.where(ks[None, :] == 0, window, 0), axis=1)
        blocked = (position < 0) | (current == dead)
        first = tl.sum(tl.where((layer_ids[:, None] == layer) & (ks[None, :] == 0), mults, 0))
        mix = tl.where(blocked, pad_id, current) * first
        for offset in tl.static_range(1, MAX_NGRAM):
            previous = position - offset
            source = tl.sum(tl.where(ks[None, :] == offset, window, 0), axis=1)
            blocked = blocked | (previous < 0) | (source == dead)
            multiplier = tl.sum(
                tl.where((layer_ids[:, None] == layer) & (ks[None, :] == offset), mults, 0)
            )
            mix = mix ^ (tl.where(blocked, pad_id, source) * multiplier)
            modulus = tl.sum(
                tl.where(orders[:, None] == layer * (MAX_NGRAM - 1) + offset - 1, mods, 0),
                axis=0,
            )
            remainder = mix[:, None] % modulus[None, :]
            remainder = tl.where(remainder < 0, remainder + modulus[None, :], remainder)
            out = (
                (layer * hash_rows + token[:, None]) * width
                + (offset - 1) * N_HEAD
                + heads[None, :]
            )
            tl.store(hashes + out, remainder, mask=valid[:, None] & in_heads[None, :])
