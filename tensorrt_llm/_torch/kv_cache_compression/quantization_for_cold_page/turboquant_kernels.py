# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Triton kernels that encode hot KV rows into TurboQuant cold pages and decode them back.

One program handles ``BLOCK_M`` rows (K or V vectors) of one buffer of one Page, or with fractional
bits one KV head of the Page. It reads the rows from the GPU slot, rotates them on tensor cores,
quantizes every rotated coordinate with the Lloyd-Max codebook, packs the codes into bit planes,
and writes them straight into the cold page, which usually lives in pinned host memory. Decoding
runs the same steps backwards. Every address is computed in 64 bits from the KVCM base pointers;
Page index pairs are read on the device.

Cold layout of one quantized buffer (offsets relative to the buffer's section, each region padded
to 16 bytes; ``R`` rows, ``G`` coordinates per group, chunks of ``C`` coordinates):

* code planes: for each group and each bit plane of width ``w`` (``bits`` split into 4, 2 or 1
  bit planes), ``[G / C chunks][R rows][C * w / 8 bytes]``;
* per-row bf16 scalars ``[scalar][R]``: the least-squares scale of every instance (rows wider
  than 512 hold several); fractional bits store one scale per group;
* fractional bits: one ``N``-bit mask of the high-bit channels per KV head (channel ``c`` in bit
  ``c % 8`` of byte ``c / 8``);
* the elements of each row outside the quantized range, copied unchanged.

With fractional bits, each of the two groups is a TurboQuant instance of ``G = N / 2``
coordinates: its channels in channel order. The kernels load and store a group's channels by
index, so both groups use fixed rotations, the diagonal blocks of one block-diagonal matrix.
"""

import triton
import triton.language as tl

OUTLIER_RATIO: tl.constexpr = tl.constexpr(4.0)
"""Energy ratio to the median channel above which a channel is an outlier (fractional bits)."""

LUT_STRIDE: tl.constexpr = tl.constexpr(16)
"""fp32 entries per codebook in the centroid table (codebook 0, then codebook 1)."""


@triton.jit
def _pack(codes, W: tl.constexpr, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """Pack ``[BLOCK_M, C]`` codes of ``W`` bits into ``[BLOCK_M, C * W / 8]`` bytes."""
    K: tl.constexpr = 8 // W
    grouped = tl.reshape(codes, (BLOCK_M, C // K, K))
    shifts = tl.arange(0, K) * W
    return tl.sum(grouped << shifts[None, None, :], axis=2).to(tl.uint8)


@triton.jit
def _unpack(packed, W: tl.constexpr, BLOCK_M: tl.constexpr, C: tl.constexpr):
    K: tl.constexpr = 8 // W
    shifts = tl.arange(0, K) * W
    fields = (packed.to(tl.int32)[:, :, None] >> shifts[None, None, :]) & ((1 << W) - 1)
    return tl.reshape(fields, (BLOCK_M, C))


@triton.jit
def _quantize(v, BITS: tl.constexpr, CB, lut):
    """Return (code, centroid) of every coordinate of ``v`` (sqrt(n)-scaled units).

    The codebook is symmetric: a binary search over the positive thresholds picks the cell ``m``
    of the magnitude, the sign the half. The centroid is selected with the search's own
    comparisons, so it needs no compare of its own. Codes ``H + m`` (positive) and
    ``H - 1 - m`` (negative) order like the centroids (``lut`` holds them for decoding).
    """
    H: tl.constexpr = 1 << (BITS - 1)
    T1, T2, T3, T4, T5, T6, T7 = CB[0], CB[1], CB[2], CB[3], CB[4], CB[5], CB[6]
    C0, C1, C2, C3, C4, C5, C6, C7 = CB[7], CB[8], CB[9], CB[10], CB[11], CB[12], CB[13], CB[14]
    a = tl.abs(v)
    if BITS == 1:
        m = tl.zeros(a.shape, tl.int32)
        magnitude = tl.zeros(a.shape, tl.float32) + C0
    elif BITS == 2:
        b0 = a > T1
        m = b0.to(tl.int32)
        magnitude = tl.where(b0, C1, C0)
    elif BITS == 3:
        b1 = a > T2
        b0 = a > tl.where(b1, T3, T1)
        m = 2 * b1.to(tl.int32) + b0.to(tl.int32)
        magnitude = tl.where(b1, tl.where(b0, C3, C2), tl.where(b0, C1, C0))
    else:
        b2 = a > T4
        b1 = a > tl.where(b2, T6, T2)
        b0 = a > tl.where(b2, tl.where(b1, T7, T5), tl.where(b1, T3, T1))
        m = 4 * b2.to(tl.int32) + 2 * b1.to(tl.int32) + b0.to(tl.int32)
        magnitude = tl.where(
            b2,
            tl.where(b1, tl.where(b0, C7, C6), tl.where(b0, C5, C4)),
            tl.where(b1, tl.where(b0, C3, C2), tl.where(b0, C1, C0)),
        )
    positive = v > 0
    return tl.where(positive, H + m, H - 1 - m), tl.where(positive, magnitude, -magnitude)


@triton.jit
def _dequantize(code, lut):
    return tl.load(lut + code)


@triton.jit
def _store_codes(
    cold,
    code,
    j,
    rows,
    row_mask,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
):
    """Store chunk ``j`` of the codes into the (up to two) bit planes of a group."""
    CB0: tl.constexpr = C * W0 // 8
    plane = code & ((1 << W0) - 1)
    offsets = j * (ROWS * CB0) + rows[:, None] * CB0 + tl.arange(0, CB0)[None, :]
    tl.store(cold + OFF0 + offsets, _pack(plane, W0, BLOCK_M, C), mask=row_mask[:, None])
    if W1 > 0:
        CB1: tl.constexpr = C * W1 // 8
        offsets = j * (ROWS * CB1) + rows[:, None] * CB1 + tl.arange(0, CB1)[None, :]
        tl.store(cold + OFF1 + offsets, _pack(code >> W0, W1, BLOCK_M, C), mask=row_mask[:, None])


@triton.jit
def _load_codes(
    cold,
    j,
    rows,
    row_mask,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
):
    CB0: tl.constexpr = C * W0 // 8
    offsets = j * (ROWS * CB0) + rows[:, None] * CB0 + tl.arange(0, CB0)[None, :]
    code = _unpack(tl.load(cold + OFF0 + offsets, mask=row_mask[:, None], other=0), W0, BLOCK_M, C)
    if W1 > 0:
        CB1: tl.constexpr = C * W1 // 8
        offsets = j * (ROWS * CB1) + rows[:, None] * CB1 + tl.arange(0, CB1)[None, :]
        high = tl.load(cold + OFF1 + offsets, mask=row_mask[:, None], other=0)
        code |= _unpack(high, W1, BLOCK_M, C) << W0
    return code


@triton.jit
def _load_code_planes(
    cold,
    j,
    rows,
    row_mask,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    ROWS: tl.constexpr,
    C: tl.constexpr,
):
    """The packed bytes of chunk ``j`` of a group's bit planes (``high`` repeats ``low`` without a
    second plane); ``_unpack_planes`` turns them into codes."""
    CB0: tl.constexpr = C * W0 // 8
    offsets = j * (ROWS * CB0) + rows[:, None] * CB0 + tl.arange(0, CB0)[None, :]
    low = tl.load(cold + OFF0 + offsets, mask=row_mask[:, None], other=0)
    high = low
    if W1 > 0:
        CB1: tl.constexpr = C * W1 // 8
        offsets = j * (ROWS * CB1) + rows[:, None] * CB1 + tl.arange(0, CB1)[None, :]
        high = tl.load(cold + OFF1 + offsets, mask=row_mask[:, None], other=0)
    return low, high


@triton.jit
def _unpack_planes(
    low, high, W0: tl.constexpr, W1: tl.constexpr, BLOCK_M: tl.constexpr, C: tl.constexpr
):
    code = _unpack(low, W0, BLOCK_M, C)
    if W1 > 0:
        code |= _unpack(high, W1, BLOCK_M, C) << W0
    return code


@triton.jit
def _rotate_chunk(
    w_ptr,
    cols,
    x0,
    x1,
    x2,
    x3,
    ROW0: tl.constexpr,
    N: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    PIECES: tl.constexpr,
):
    """Rotated coordinates ``cols`` of the rows held as power-of-two pieces ``x0..x3``.

    ``w_ptr`` is the row-major ``N x N`` matrix whose row ``ROW0 + i`` maps input coordinate ``i``
    (the pieces in order) to the rotated coordinates. Bit ``i`` of ``PIECES`` selects piece ``i``:
    a block-diagonal matrix skips the pieces of other blocks.
    """
    if PIECES & 1:
        r0 = ROW0 + tl.arange(0, P0)
        acc = tl.dot(x0, tl.load(w_ptr + r0[:, None] * N + cols[None, :]))
    else:
        acc = tl.zeros([x0.shape[0], cols.shape[0]], tl.float32)
    if P1 > 0 and PIECES & 2:
        r1 = ROW0 + P0 + tl.arange(0, P1)
        acc += tl.dot(x1, tl.load(w_ptr + r1[:, None] * N + cols[None, :]))
    if P2 > 0 and PIECES & 4:
        r2 = ROW0 + P0 + P1 + tl.arange(0, P2)
        acc += tl.dot(x2, tl.load(w_ptr + r2[:, None] * N + cols[None, :]))
    if P3 > 0 and PIECES & 8:
        r3 = ROW0 + P0 + P1 + P2 + tl.arange(0, P3)
        acc += tl.dot(x3, tl.load(w_ptr + r3[:, None] * N + cols[None, :]))
    return acc


@triton.jit
def _load_piece(row_ptr, row_mask, START: tl.constexpr, P: tl.constexpr):
    columns = START + tl.arange(0, P)
    return tl.load(row_ptr[:, None] + columns[None, :], mask=row_mask[:, None], other=0.0).to(
        tl.float32
    )


@triton.jit
def _abs_max(x):
    return tl.max(tl.abs(x), axis=1)


@triton.jit
def _float_bits(x):
    return x.to(tl.int32, bitcast=True)


@triton.jit
def _energy_bits(energy):
    """Channel energies as uint32 bit patterns, which order like the non-negative floats."""
    # Non-finite rows do not vote, so the energies are finite; a NaN would rank lowest, so exactly
    # G channels win whatever the input.
    return tl.where(energy == energy, energy, 0.0).to(tl.uint32, bitcast=True)


@triton.jit
def _count_at_least(bits, candidates):
    """How many of ``bits`` are at least each of ``candidates``."""
    return tl.sum((bits[:, None] >= candidates[None, :]).to(tl.int32), axis=0)


@triton.jit
def _fill(outlier, taken, need):
    """Add the first regular channels (in channel order) until ``need`` more are outliers."""
    free = (outlier == 0).to(tl.int32)
    chosen = outlier | ((free == 1) & (taken + tl.cumsum(free, axis=0) - free < need))
    return chosen, taken + tl.sum(free, axis=0)


@triton.jit
def _select_outliers(
    e0,
    e1,
    e2,
    e3,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    G: tl.constexpr,
):
    """Choose the ``G`` channels of the high-bit group from the channel energies.

    Channels whose energy exceeds ``OUTLIER_RATIO`` times the median channel energy are
    outliers; the remaining places go to the lowest-numbered other channels. A small
    perturbation (such as the error of a previous round trip) rarely moves a channel across that
    margin, so re-encoding a decoded Page usually selects the same channels again.
    ``e*`` hold the non-negative float energies as uint32 bit patterns, which order like the
    floats; the median (the G-th largest energy) is found four bits at a time, without sorting.
    """
    digits = tl.arange(0, 16).to(tl.uint32)
    median = tl.sum(tl.zeros([16], tl.uint32), axis=0)
    for step in tl.static_range(8):
        # Digit `step` (bits 28 - 4 * step and up) of the median. The counts fall as the digit
        # grows, and digit 0 (the median so far) always has G.
        candidates = median | (digits << (28 - 4 * step))
        count = _count_at_least(e0, candidates)
        if P1 > 0:
            count += _count_at_least(e1, candidates)
        if P2 > 0:
            count += _count_at_least(e2, candidates)
        if P3 > 0:
            count += _count_at_least(e3, candidates)
        median = median | ((tl.sum((count >= G).to(tl.uint32), axis=0) - 1) << (28 - 4 * step))
    limit = OUTLIER_RATIO * median.to(tl.float32, bitcast=True)
    o0 = e0.to(tl.float32, bitcast=True) > limit
    need = G - tl.sum(o0.to(tl.int32), axis=0)
    o1 = o0
    o2 = o0
    o3 = o0
    if P1 > 0:
        o1 = e1.to(tl.float32, bitcast=True) > limit
        need -= tl.sum(o1.to(tl.int32), axis=0)
    if P2 > 0:
        o2 = e2.to(tl.float32, bitcast=True) > limit
        need -= tl.sum(o2.to(tl.int32), axis=0)
    if P3 > 0:
        o3 = e3.to(tl.float32, bitcast=True) > limit
        need -= tl.sum(o3.to(tl.int32), axis=0)
    o0, taken = _fill(o0, 0, need)
    if P1 > 0:
        o1, taken = _fill(o1, taken, need)
    if P2 > 0:
        o2, taken = _fill(o2, taken, need)
    if P3 > 0:
        o3, taken = _fill(o3, taken, need)
    return o0, o1, o2, o3


@triton.jit
def _store_mask(mask_ptr, high, START: tl.constexpr, P: tl.constexpr):
    grouped = tl.reshape(high.to(tl.int32), (P // 8, 8))
    shifts = tl.arange(0, 8)
    packed = tl.sum(grouped << shifts[None, :], axis=1).to(tl.uint8)
    tl.store(mask_ptr + START // 8 + tl.arange(0, P // 8), packed)


@triton.jit
def _mask_words(high, P: tl.constexpr):
    """``P`` channel flags as ``P / 16`` 16-bit words (uint32): channel ``16 w + i`` in bit ``i``."""
    bits = tl.reshape(high.to(tl.uint32), (P // 16, 16))
    return tl.sum(bits << tl.arange(0, 16).to(tl.uint32)[None, :], axis=1)


@triton.jit
def _load_mask_words(mask_ptr, START: tl.constexpr, P: tl.constexpr):
    """The stored mask of channels ``START..START+P-1`` as ``_mask_words`` gives it."""
    packed = tl.load(mask_ptr + START // 8 + tl.arange(0, P // 8)).to(tl.uint32)
    pairs = tl.reshape(packed, (P // 16, 2))
    return tl.sum(pairs << (8 * tl.arange(0, 2)).to(tl.uint32)[None, :], axis=1)


@triton.jit
def _popcount(x):
    """Set bits of every uint32 of ``x``, as int32."""
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _locate(k, words, before, WORD0: tl.constexpr, WP: tl.constexpr, word, rank, base):
    """Add, for the positions ``k`` whose set bit lies in these ``WP`` words, the word, the bit's
    rank among the word's set bits, and the word's first channel; ``before`` set bits precede the
    words. Also returns the set bits up to the end of the words."""
    count = _popcount(words)
    first = before + tl.cumsum(count, axis=0) - count
    hit = (first[None, :] <= k[:, None]) & (k[:, None] < (first + count)[None, :])
    word += tl.sum(tl.where(hit, words[None, :], 0), axis=1)
    rank += tl.sum(tl.where(hit, k[:, None] - first[None, :], 0), axis=1)
    base += tl.sum(tl.where(hit, 16 * (WORD0 + tl.arange(0, WP))[None, :], 0), axis=1)
    return word, rank, base, before + tl.sum(count, axis=0)


@triton.jit
def _select_bit(word, rank):
    """Position in every 16-bit ``word`` of the set bit with ``rank`` set bits below it."""
    position = tl.zeros_like(rank)
    for step in tl.static_range(4):
        # Halves of 8, 4, 2, and 1 bits.
        low = _popcount(word & ((1 << (8 >> step)) - 1))
        up = rank >= low
        rank = tl.where(up, rank - low, rank)
        word = tl.where(up, word >> (8 >> step), word)
        position += tl.where(up, 8 >> step, 0)
    return position


@triton.jit
def _group_channels(
    w0,
    w1,
    w2,
    w3,
    HIGH: tl.constexpr,
    START: tl.constexpr,
    Q: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
):
    """Channels at positions ``START..START+Q-1`` of the high-bit group (``HIGH``) or the other.

    ``w0..w3`` are the mask words of the channel pieces. A group lists its channels in channel
    order, so position ``k`` holds the channel of the ``k``-th set bit of the group's words.
    """
    if not HIGH:
        w0 = w0 ^ 0xFFFF
        w1 = w1 ^ 0xFFFF
        w2 = w2 ^ 0xFFFF
        w3 = w3 ^ 0xFFFF
    k = START + tl.arange(0, Q)
    word = tl.zeros([Q], tl.uint32)
    rank = tl.zeros([Q], tl.int32)
    base = tl.zeros([Q], tl.int32)
    word, rank, base, before = _locate(k, w0, 0, 0, P0 // 16, word, rank, base)
    if P1 > 0:
        word, rank, base, before = _locate(k, w1, before, P0 // 16, P1 // 16, word, rank, base)
    if P2 > 0:
        word, rank, base, before = _locate(
            k, w2, before, (P0 + P1) // 16, P2 // 16, word, rank, base
        )
    if P3 > 0:
        word, rank, base, before = _locate(
            k, w3, before, (P0 + P1 + P2) // 16, P3 // 16, word, rank, base
        )
    return base + _select_bit(word, rank)


@triton.jit
def _group_channel_pieces(
    w0,
    w1,
    w2,
    w3,
    HIGH: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
):
    """The channels of a group as power-of-two pieces ``Q0..Q3`` of its positions."""
    c0 = _group_channels(w0, w1, w2, w3, HIGH, 0, Q0, P0, P1, P2, P3)
    c1 = c0
    c2 = c0
    c3 = c0
    if Q1 > 0:
        c1 = _group_channels(w0, w1, w2, w3, HIGH, Q0, Q1, P0, P1, P2, P3)
    if Q2 > 0:
        c2 = _group_channels(w0, w1, w2, w3, HIGH, Q0 + Q1, Q2, P0, P1, P2, P3)
    if Q3 > 0:
        c3 = _group_channels(w0, w1, w2, w3, HIGH, Q0 + Q1 + Q2, Q3, P0, P1, P2, P3)
    return c0, c1, c2, c3


@triton.jit
def _load_channels(row_ptr, row_mask, channels):
    return tl.load(row_ptr[:, None] + channels[None, :], mask=row_mask[:, None], other=0.0).to(
        tl.float32
    )


@triton.jit
def _quantize_group(
    cold,
    w_ptr,
    x0,
    x1,
    x2,
    x3,
    inv_norm,
    rows,
    row_mask,
    COORD0: tl.constexpr,
    G: tl.constexpr,
    BITS: tl.constexpr,
    CB,
    lut,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    N: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    ROW0: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    PIECES: tl.constexpr,
):
    """Quantize rotated coordinates [COORD0, COORD0 + G); return the least-squares scale.

    The input pieces ``x0..x3`` are matrix rows ``ROW0`` on (see ``_rotate_chunk``).
    """
    dot_yc = tl.zeros([BLOCK_M], tl.float32)
    dot_cc = tl.zeros([BLOCK_M], tl.float32)
    for j in range(G // C):
        cols = COORD0 + j * C + tl.arange(0, C)
        y = _rotate_chunk(w_ptr, cols, x0, x1, x2, x3, ROW0, N, P0, P1, P2, P3, PIECES)
        code, value = _quantize(y * inv_norm[:, None], BITS, CB, lut)
        dot_yc += tl.sum(y * value, axis=1)
        dot_cc += tl.sum(value * value, axis=1)
        _store_codes(cold[:, None], code, j, rows, row_mask, OFF0, W0, OFF1, W1, ROWS, BLOCK_M, C)
    return tl.where(dot_cc > 0, dot_yc / dot_cc, 0.0)


@triton.jit
def _encode_block(
    cold,
    w_ptr,
    h0,
    h1,
    h2,
    h3,
    norm,
    prescale,
    rows,
    row_mask,
    COORD0: tl.constexpr,
    G: tl.constexpr,
    PIECES: tl.constexpr,
    BITS: tl.constexpr,
    CB,
    lut,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    ALPHA_AT: tl.constexpr,
    N: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
):
    """Encode one TurboQuant instance: rotated coordinates [COORD0, COORD0 + G) of pieces PIECES.

    Stores the codes and the least-squares scale at scalar ``ALPHA_AT``.
    """
    inv_norm = tl.where(norm > 0, tl.sqrt(G * 1.0) * tl.math.rsqrt(norm), 0.0)
    alpha = _quantize_group(
        cold,
        w_ptr,
        h0,
        h1,
        h2,
        h3,
        inv_norm,
        rows,
        row_mask,
        COORD0,
        G,
        BITS,
        CB,
        lut,
        OFF0,
        W0,
        OFF1,
        W1,
        N,
        P0,
        P1,
        P2,
        P3,
        0,
        ROWS,
        BLOCK_M,
        C,
        PIECES,
    )
    scale_ptr = (cold + OFF_SCALE).to(tl.pointer_type(tl.bfloat16))
    tl.store(scale_ptr + ALPHA_AT + rows, (alpha / prescale).to(tl.bfloat16), mask=row_mask)


@triton.jit
def _prescale(amax):
    """Power of two that puts a row's largest magnitude in [1, 2): exact, and safe for the fp16
    tensor-core operands and the fp32 sums whatever the input range."""
    exponent = (_float_bits(amax) >> 23) & 0xFF
    return (tl.maximum(254 - exponent, 1) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _encode_instance(
    cold,
    w_ptr,
    row_ptr,
    prescale,
    rows,
    row_mask,
    START: tl.constexpr,
    P: tl.constexpr,
    PIECE: tl.constexpr,
    BITS: tl.constexpr,
    CB,
    lut,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    ALPHA_AT: tl.constexpr,
    N: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
):
    """Encode the mse instance held by piece ``PIECE`` (bit) of a blocked row, loading only it."""
    x = _load_piece(row_ptr, row_mask, START, P) * prescale[:, None]
    h = x.to(tl.float16)
    _encode_block(
        cold,
        w_ptr,
        h,
        h,
        h,
        h,
        tl.sum(x * x, axis=1),
        prescale,
        rows,
        row_mask,
        START,
        P,
        PIECE,
        BITS,
        CB,
        lut,
        OFF0,
        W0,
        OFF1,
        W1,
        OFF_SCALE,
        ALPHA_AT,
        N,
        P0,
        P1,
        P2,
        P3,
        ROWS,
        BLOCK_M,
        C,
    )


@triton.jit
def _piece_stats(row_ptr, row_mask, START: tl.constexpr, P: tl.constexpr):
    """Largest magnitude and count of non-finite values of every row in one piece, read in chunks
    of at most 128 values to keep few of them in registers."""
    CHUNK: tl.constexpr = min(P, 128)
    amax = tl.zeros([row_ptr.shape[0]], tl.float32)
    bad = tl.zeros([row_ptr.shape[0]], tl.int32)
    for i in tl.static_range(P // CHUNK):
        x = _load_piece(row_ptr, row_mask, START + i * CHUNK, CHUNK)
        amax = tl.maximum(amax, _abs_max(x))
        bad += tl.sum(((x != x) | (tl.abs(x) == float("inf"))).to(tl.int32), axis=1)
    return amax, bad


@triton.jit
def _row_prescale(
    row_ptr,
    row_mask,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
):
    """Prescale of every row (see ``_prescale``) and whether all of the row's values are finite."""
    amax, bad = _piece_stats(row_ptr, row_mask, 0, P0)
    if P1 > 0:
        piece_max, piece_bad = _piece_stats(row_ptr, row_mask, P0, P1)
        amax = tl.maximum(amax, piece_max)
        bad += piece_bad
    if P2 > 0:
        piece_max, piece_bad = _piece_stats(row_ptr, row_mask, P0 + P1, P2)
        amax = tl.maximum(amax, piece_max)
        bad += piece_bad
    if P3 > 0:
        piece_max, piece_bad = _piece_stats(row_ptr, row_mask, P0 + P1 + P2, P3)
        amax = tl.maximum(amax, piece_max)
        bad += piece_bad
    return _prescale(amax), bad == 0


@triton.jit
def _channel_energy(row_ptr, voters, prescale, START: tl.constexpr, P: tl.constexpr):
    """Energy of every channel of one piece over the voting rows (prescaled)."""
    x = _load_piece(row_ptr, voters, START, P) * prescale[:, None]
    return tl.sum(x * x, axis=0)


@triton.jit
def _encode_group(
    cold,
    w_ptr,
    row_ptr,
    row_mask,
    rows,
    prescale,
    c0,
    c1,
    c2,
    c3,
    COORD0: tl.constexpr,
    G: tl.constexpr,
    BITS: tl.constexpr,
    CB,
    lut,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    N: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
):
    """Encode one group of a fractional-bit row: its channels ``c0..c3`` (its positions, as pieces
    ``Q0..Q3``), rotated by the diagonal block at ``COORD0``. Returns the least-squares scale."""
    x0 = _load_channels(row_ptr, row_mask, c0) * prescale[:, None]
    norm = tl.sum(x0 * x0, axis=1)
    x1 = x0
    x2 = x0
    x3 = x0
    if Q1 > 0:
        x1 = _load_channels(row_ptr, row_mask, c1) * prescale[:, None]
        norm += tl.sum(x1 * x1, axis=1)
    if Q2 > 0:
        x2 = _load_channels(row_ptr, row_mask, c2) * prescale[:, None]
        norm += tl.sum(x2 * x2, axis=1)
    if Q3 > 0:
        x3 = _load_channels(row_ptr, row_mask, c3) * prescale[:, None]
        norm += tl.sum(x3 * x3, axis=1)
    inv_norm = tl.where(norm > 0, tl.sqrt(G * 1.0) * tl.math.rsqrt(norm), 0.0)
    return _quantize_group(
        cold,
        w_ptr,
        x0.to(tl.float16),
        x1.to(tl.float16),
        x2.to(tl.float16),
        x3.to(tl.float16),
        inv_norm,
        rows,
        row_mask,
        COORD0,
        G,
        BITS,
        CB,
        lut,
        OFF0,
        W0,
        OFF1,
        W1,
        N,
        Q0,
        Q1,
        Q2,
        Q3,
        COORD0,
        ROWS,
        BLOCK_M,
        C,
        15,
    )


@triton.jit
def _encode_fractional(
    page_ptr,
    cold_ptr,
    cold_page_bytes,
    hot_ptr,
    table_ptr,
    w_ptr,
    lut_ptr,
    ELEM: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    QS: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    C: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    SUB_ROWS: tl.constexpr,
    BITS0: tl.constexpr,
    CB0: tl.constexpr,
    BITS1: tl.constexpr,
    CB1: tl.constexpr,
    OFF_A0: tl.constexpr,
    W_A0: tl.constexpr,
    OFF_A1: tl.constexpr,
    W_A1: tl.constexpr,
    OFF_B0: tl.constexpr,
    W_B0: tl.constexpr,
    OFF_B1: tl.constexpr,
    W_B1: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    OFF_MASK: tl.constexpr,
    OFF_RAW: tl.constexpr,
    RAW: tl.constexpr,
    RAW_BLOCK: tl.constexpr,
):
    """Encode one KV head (``TILE_ROWS`` rows) of one buffer of one Page with fractional bits.

    The head's rows vote on the split of the channels into the high-bit group and the other (see
    ``_select_outliers``); then each group of every row is encoded as one TurboQuant instance,
    ``SUB_ROWS`` rows at a time.
    """
    G: tl.constexpr = N // 2
    PARTS: tl.constexpr = TILE_ROWS // SUB_ROWS
    page = tl.program_id(0)
    buffer = tl.program_id(1)
    head = tl.program_id(2)
    lane = tl.arange(0, BLOCK_M)
    row_mask = lane < SUB_ROWS
    # Page-index pairs (dst, src) live in device memory (ColdPageIndexLocation.DEVICE).
    dst = tl.load(page_ptr + 2 * page).to(tl.int64)
    src = tl.load(page_ptr + 2 * page + 1).to(tl.int64)
    # Buffer table: hot buffer offset from hot_ptr and Slot stride (elements), cold offset (bytes).
    hot_offset = tl.load(table_ptr + 3 * buffer)
    slot_elements = tl.load(table_ptr + 3 * buffer + 1)
    cold_offset = tl.load(table_ptr + 3 * buffer + 2)
    ALIGN: tl.constexpr = 16 if ELEM == tl.float8e4nv else 8
    hot = hot_ptr + tl.multiple_of(hot_offset + src * slot_elements, ALIGN)
    cold = cold_ptr + tl.multiple_of(dst * cold_page_bytes + cold_offset, 16)
    first_row = head * TILE_ROWS

    # The vote: the energy of every channel over the head's rows, VOTE rows at a time to keep few
    # values in registers. Rows holding any NaN or Inf (garbage tokens of a partial Page) do not
    # vote.
    VOTE: tl.constexpr = 8
    e0 = tl.zeros([P0], tl.float32)
    e1 = tl.zeros([P1 if P1 > 0 else 16], tl.float32)
    e2 = tl.zeros([P2 if P2 > 0 else 16], tl.float32)
    e3 = tl.zeros([P3 if P3 > 0 else 16], tl.float32)
    vote_lane = tl.arange(0, VOTE)
    for part in range((TILE_ROWS + VOTE - 1) // VOTE):
        in_head = part * VOTE + vote_lane < TILE_ROWS
        rows = first_row + part * VOTE + vote_lane
        row_ptr = hot + rows.to(tl.int64) * STRIDE + QS
        prescale, finite = _row_prescale(row_ptr, in_head, P0, P1, P2, P3)
        voters = in_head & finite
        e0 += _channel_energy(row_ptr, voters, prescale, 0, P0)
        if P1 > 0:
            e1 += _channel_energy(row_ptr, voters, prescale, P0, P1)
        if P2 > 0:
            e2 += _channel_energy(row_ptr, voters, prescale, P0 + P1, P2)
        if P3 > 0:
            e3 += _channel_energy(row_ptr, voters, prescale, P0 + P1 + P2, P3)
    o0, o1, o2, o3 = _select_outliers(
        _energy_bits(e0), _energy_bits(e1), _energy_bits(e2), _energy_bits(e3), P0, P1, P2, P3, G
    )
    mask_ptr = cold + OFF_MASK + head * (N // 8)
    _store_mask(mask_ptr, o0, 0, P0)
    w0 = _mask_words(o0, P0)
    w1 = w0
    w2 = w0
    w3 = w0
    if P1 > 0:
        _store_mask(mask_ptr, o1, P0, P1)
        w1 = _mask_words(o1, P1)
    if P2 > 0:
        _store_mask(mask_ptr, o2, P0 + P1, P2)
        w2 = _mask_words(o2, P2)
    if P3 > 0:
        _store_mask(mask_ptr, o3, P0 + P1 + P2, P3)
        w3 = _mask_words(o3, P3)
    h0, h1, h2, h3 = _group_channel_pieces(w0, w1, w2, w3, True, P0, P1, P2, P3, Q0, Q1, Q2, Q3)
    l0, l1, l2, l3 = _group_channel_pieces(w0, w1, w2, w3, False, P0, P1, P2, P3, Q0, Q1, Q2, Q3)

    cold_rows = cold + tl.zeros([BLOCK_M], tl.int64)
    scale_ptr = (cold + OFF_SCALE).to(tl.pointer_type(tl.bfloat16))
    for part in range(PARTS):
        rows = first_row + part * SUB_ROWS + lane
        row_ptr = hot + rows.to(tl.int64) * STRIDE + QS
        prescale, finite = _row_prescale(row_ptr, row_mask, P0, P1, P2, P3)
        alpha_hi = _encode_group(
            cold_rows,
            w_ptr,
            row_ptr,
            row_mask,
            rows,
            prescale,
            h0,
            h1,
            h2,
            h3,
            0,
            G,
            BITS0,
            CB0,
            lut_ptr,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            N,
            Q0,
            Q1,
            Q2,
            Q3,
            ROWS,
            BLOCK_M,
            C,
        )
        tl.store(scale_ptr + rows, (alpha_hi / prescale).to(tl.bfloat16), mask=row_mask)
        alpha_lo = _encode_group(
            cold_rows,
            w_ptr,
            row_ptr,
            row_mask,
            rows,
            prescale,
            l0,
            l1,
            l2,
            l3,
            G,
            G,
            BITS1,
            CB1,
            lut_ptr + LUT_STRIDE,
            OFF_B0,
            W_B0,
            OFF_B1,
            W_B1,
            N,
            Q0,
            Q1,
            Q2,
            Q3,
            ROWS,
            BLOCK_M,
            C,
        )
        tl.store(scale_ptr + ROWS + rows, (alpha_lo / prescale).to(tl.bfloat16), mask=row_mask)
        if RAW > 0:
            # Elements outside the quantized range (e.g. RoPE) are copied unchanged.
            raw_cols = tl.arange(0, RAW_BLOCK)
            raw_src = tl.where(raw_cols < QS, raw_cols, raw_cols + N)
            raw_mask = row_mask[:, None] & (raw_cols < RAW)[None, :]
            values = tl.load(
                hot + rows.to(tl.int64)[:, None] * STRIDE + raw_src[None, :], mask=raw_mask
            )
            raw_ptr = (cold + OFF_RAW).to(tl.pointer_type(ELEM))
            tl.store(raw_ptr + rows[:, None] * RAW + raw_cols[None, :], values, mask=raw_mask)


@triton.jit
def _encode_tiles(
    page_ptr,
    cold_ptr,
    cold_page_bytes,
    num_pages,
    hot_ptr,
    table_ptr,
    w_ptr,
    lut_ptr,
    ELEM: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    QS: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    C: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    MODE: tl.constexpr,
    BLOCKED: tl.constexpr,
    PASS: tl.constexpr,
    BITS0: tl.constexpr,
    CB0: tl.constexpr,
    BITS1: tl.constexpr,
    CB1: tl.constexpr,
    OFF_A0: tl.constexpr,
    W_A0: tl.constexpr,
    OFF_A1: tl.constexpr,
    W_A1: tl.constexpr,
    OFF_B0: tl.constexpr,
    W_B0: tl.constexpr,
    OFF_B1: tl.constexpr,
    W_B1: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    OFF_MASK: tl.constexpr,
    OFF_RAW: tl.constexpr,
    RAW: tl.constexpr,
    RAW_BLOCK: tl.constexpr,
):
    tile_page = tl.program_id(0)
    buffer = tl.program_id(1)
    tile = tl.program_id(2)
    lane = tl.arange(0, BLOCK_M)
    lut0 = lut_ptr
    lut1 = lut_ptr + LUT_STRIDE
    if PAGES_PER_TILE > 1:
        # Buffers with few rows per Page: one tile takes the rows of several Pages.
        page = tile_page * PAGES_PER_TILE + lane // ROWS_PAD
        rows = lane % ROWS_PAD
        row_mask = (rows < ROWS) & (page < num_pages)
    else:
        # One tile of rows of one Page.
        page = tile_page + lane * 0
        rows = tile * TILE_ROWS + lane
        row_mask = (lane < TILE_ROWS) & (rows < ROWS)
    # Page-index pairs (dst, src) live in device memory (ColdPageIndexLocation.DEVICE).
    pair_ptr = page_ptr + 2 * page.to(tl.int64)
    dst = tl.load(pair_ptr, mask=row_mask, other=0).to(tl.int64)
    src = tl.load(pair_ptr + 1, mask=row_mask, other=0).to(tl.int64)
    # Buffer table: hot buffer offset from hot_ptr and Slot stride (elements), cold offset (bytes).
    hot_offset = tl.load(table_ptr + 3 * buffer)
    slot_elements = tl.load(table_ptr + 3 * buffer + 1)
    cold_offset = tl.load(table_ptr + 3 * buffer + 2)
    # Slots, cold pages, and buffer sections are 16-byte aligned (checked at configure time):
    # offsets from the aligned pointer arguments let the loads and stores use 16-byte vectors.
    ALIGN: tl.constexpr = 16 if ELEM == tl.float8e4nv else 8
    hot = hot_ptr + tl.multiple_of(hot_offset + src * slot_elements, ALIGN)
    cold_rows = tl.multiple_of(dst * cold_page_bytes + cold_offset, 16)
    cold = cold_ptr + cold_rows
    row_ptr = hot + rows.to(tl.int64) * STRIDE + QS

    if BLOCKED:
        # Rows wider than 512: every piece is an independent instance, its codes in the
        # piece's chunks of the shared bit planes, its scale at scalar row i. The pieces are
        # encoded one at a time, so only one is live: a first pass over the row finds the prescale
        # (the same power of two as for any other row), then each piece is read again (from L2).
        amax = _abs_max(_load_piece(row_ptr, row_mask, 0, P0))
        if P1 > 0:
            amax = tl.maximum(amax, _abs_max(_load_piece(row_ptr, row_mask, P0, P1)))
        if P2 > 0:
            amax = tl.maximum(amax, _abs_max(_load_piece(row_ptr, row_mask, P0 + P1, P2)))
        if P3 > 0:
            amax = tl.maximum(amax, _abs_max(_load_piece(row_ptr, row_mask, P0 + P1 + P2, P3)))
        prescale = _prescale(amax)
        _encode_instance(
            cold,
            w_ptr,
            row_ptr,
            prescale,
            rows,
            row_mask,
            0,
            P0,
            1,
            BITS0,
            CB0,
            lut0,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            OFF_SCALE,
            0,
            N,
            P0,
            P1,
            P2,
            P3,
            ROWS,
            BLOCK_M,
            C,
        )
        if P1 > 0:
            _encode_instance(
                cold,
                w_ptr,
                row_ptr,
                prescale,
                rows,
                row_mask,
                P0,
                P1,
                2,
                BITS0 if P1 == P0 else BITS1,
                CB0 if P1 == P0 else CB1,
                lut0 if P1 == P0 else lut1,
                OFF_A0 + P0 // C * ROWS * (C * W_A0 // 8),
                W_A0,
                OFF_A1 + P0 // C * ROWS * (C * W_A1 // 8),
                W_A1,
                OFF_SCALE,
                ROWS,
                N,
                P0,
                P1,
                P2,
                P3,
                ROWS,
                BLOCK_M,
                C,
            )
        if P2 > 0:
            _encode_instance(
                cold,
                w_ptr,
                row_ptr,
                prescale,
                rows,
                row_mask,
                P0 + P1,
                P2,
                4,
                BITS0 if P2 == P0 else BITS1,
                CB0 if P2 == P0 else CB1,
                lut0 if P2 == P0 else lut1,
                OFF_A0 + (P0 + P1) // C * ROWS * (C * W_A0 // 8),
                W_A0,
                OFF_A1 + (P0 + P1) // C * ROWS * (C * W_A1 // 8),
                W_A1,
                OFF_SCALE,
                2 * ROWS,
                N,
                P0,
                P1,
                P2,
                P3,
                ROWS,
                BLOCK_M,
                C,
            )
        if P3 > 0:
            _encode_instance(
                cold,
                w_ptr,
                row_ptr,
                prescale,
                rows,
                row_mask,
                P0 + P1 + P2,
                P3,
                8,
                BITS0 if P3 == P0 else BITS1,
                CB0 if P3 == P0 else CB1,
                lut0 if P3 == P0 else lut1,
                OFF_A0 + (P0 + P1 + P2) // C * ROWS * (C * W_A0 // 8),
                W_A0,
                OFF_A1 + (P0 + P1 + P2) // C * ROWS * (C * W_A1 // 8),
                W_A1,
                OFF_SCALE,
                3 * ROWS,
                N,
                P0,
                P1,
                P2,
                P3,
                ROWS,
                BLOCK_M,
                C,
            )
    else:
        x0 = _load_piece(row_ptr, row_mask, 0, P0)
        amax = _abs_max(x0)
        x1 = x0
        x2 = x0
        x3 = x0
        if P1 > 0:
            x1 = _load_piece(row_ptr, row_mask, P0, P1)
            amax = tl.maximum(amax, _abs_max(x1))
        if P2 > 0:
            x2 = _load_piece(row_ptr, row_mask, P0 + P1, P2)
            amax = tl.maximum(amax, _abs_max(x2))
        if P3 > 0:
            x3 = _load_piece(row_ptr, row_mask, P0 + P1 + P2, P3)
            amax = tl.maximum(amax, _abs_max(x3))
        prescale = _prescale(amax)
        x0 = x0 * prescale[:, None]
        x1 = x1 * prescale[:, None]
        x2 = x2 * prescale[:, None]
        x3 = x3 * prescale[:, None]

        norm_rest = tl.zeros([BLOCK_M], tl.float32)
        if P1 > 0:
            norm_rest += tl.sum(x1 * x1, axis=1)
        if P2 > 0:
            norm_rest += tl.sum(x2 * x2, axis=1)
        if P3 > 0:
            norm_rest += tl.sum(x3 * x3, axis=1)
        _encode_block(
            cold,
            w_ptr,
            x0.to(tl.float16),
            x1.to(tl.float16),
            x2.to(tl.float16),
            x3.to(tl.float16),
            tl.sum(x0 * x0, axis=1) + norm_rest,
            prescale,
            rows,
            row_mask,
            0,
            N,
            15,
            BITS0,
            CB0,
            lut0,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            OFF_SCALE,
            0,
            N,
            P0,
            P1,
            P2,
            P3,
            ROWS,
            BLOCK_M,
            C,
        )

    if RAW > 0:
        # Elements outside the quantized range (e.g. RoPE) are copied unchanged.
        raw_cols = tl.arange(0, RAW_BLOCK)
        raw_src = tl.where(raw_cols < QS, raw_cols, raw_cols + N)
        raw_mask = row_mask[:, None] & (raw_cols < RAW)[None, :]
        values = tl.load(
            hot[:, None] + rows.to(tl.int64)[:, None] * STRIDE + raw_src[None, :], mask=raw_mask
        )
        raw_ptr = (cold + OFF_RAW).to(tl.pointer_type(ELEM))
        tl.store(raw_ptr[:, None] + rows[:, None] * RAW + raw_cols[None, :], values, mask=raw_mask)


@triton.jit(do_not_specialize=["page_ptr", "cold_page_bytes", "num_pages"])
def _tq_encode_kernel(
    page_ptr,
    cold_ptr,
    cold_page_bytes,
    num_pages,
    hot_ptr,
    table_ptr,
    w_ptr,
    lut_ptr,
    ELEM: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    QS: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    C: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    MODE: tl.constexpr,
    BLOCKED: tl.constexpr,
    PASS: tl.constexpr,
    BITS0: tl.constexpr,
    CB0: tl.constexpr,
    BITS1: tl.constexpr,
    CB1: tl.constexpr,
    OFF_A0: tl.constexpr,
    W_A0: tl.constexpr,
    OFF_A1: tl.constexpr,
    W_A1: tl.constexpr,
    OFF_B0: tl.constexpr,
    W_B0: tl.constexpr,
    OFF_B1: tl.constexpr,
    W_B1: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    OFF_MASK: tl.constexpr,
    OFF_RAW: tl.constexpr,
    RAW: tl.constexpr,
    RAW_BLOCK: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    SUB_ROWS: tl.constexpr,
):
    # Fractional bits (MODE 1): one program per KV head of a Page (see _encode_fractional).
    if MODE == 1:
        _encode_fractional(
            page_ptr,
            cold_ptr,
            cold_page_bytes,
            hot_ptr,
            table_ptr,
            w_ptr,
            lut_ptr,
            ELEM,
            N,
            STRIDE,
            QS,
            P0,
            P1,
            P2,
            P3,
            Q0,
            Q1,
            Q2,
            Q3,
            C,
            ROWS,
            BLOCK_M,
            TILE_ROWS,
            SUB_ROWS,
            BITS0,
            CB0,
            BITS1,
            CB1,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            OFF_B0,
            W_B0,
            OFF_B1,
            W_B1,
            OFF_SCALE,
            OFF_MASK,
            OFF_RAW,
            RAW,
            RAW_BLOCK,
        )
    else:
        _encode_tiles(
            page_ptr,
            cold_ptr,
            cold_page_bytes,
            num_pages,
            hot_ptr,
            table_ptr,
            w_ptr,
            lut_ptr,
            ELEM,
            N,
            STRIDE,
            QS,
            P0,
            P1,
            P2,
            P3,
            C,
            ROWS,
            BLOCK_M,
            TILE_ROWS,
            PAGES_PER_TILE,
            ROWS_PAD,
            MODE,
            BLOCKED,
            PASS,
            BITS0,
            CB0,
            BITS1,
            CB1,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            OFF_B0,
            W_B0,
            OFF_B1,
            W_B1,
            OFF_SCALE,
            OFF_MASK,
            OFF_RAW,
            RAW,
            RAW_BLOCK,
        )


@triton.jit
def _decode_matrix(w_ptr, krows, START, P: tl.constexpr, N: tl.constexpr):
    """Tile [krows, START..START+P-1] of the decode matrix: row k maps rotated coordinate k."""
    cols = START + tl.arange(0, P)
    return tl.load(w_ptr + krows[:, None] * N + cols[None, :])


@triton.jit
def _accumulate_group(
    cold,
    w_ptr,
    alpha,
    a0,
    a1,
    a2,
    a3,
    rows,
    row_mask,
    COORD0: tl.constexpr,
    G: tl.constexpr,
    BITS: tl.constexpr,
    lut,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    N: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
):
    """Add the decoded group to the output pieces FIRST..LAST-1."""
    for j in range(G // C):
        code = _load_codes(cold[:, None], j, rows, row_mask, OFF0, W0, OFF1, W1, ROWS, BLOCK_M, C)
        z = (alpha[:, None] * _dequantize(code, lut)).to(tl.bfloat16)
        krows = COORD0 + j * C + tl.arange(0, C)
        if FIRST <= 0:
            a0 += tl.dot(z, _decode_matrix(w_ptr, krows, 0, P0, N))
        if P1 > 0 and FIRST <= 1 and 1 < LAST:
            a1 += tl.dot(z, _decode_matrix(w_ptr, krows, P0, P1, N))
        if P2 > 0 and FIRST <= 2 and 2 < LAST:
            a2 += tl.dot(z, _decode_matrix(w_ptr, krows, P0 + P1, P2, N))
        if P3 > 0 and 3 < LAST:
            a3 += tl.dot(z, _decode_matrix(w_ptr, krows, P0 + P1 + P2, P3, N))
    return a0, a1, a2, a3


@triton.jit
def _decode_rows(
    cold,
    w_ptr,
    alpha,
    rows,
    row_mask,
    N: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    BITS0: tl.constexpr,
    lut0,
    BITS1: tl.constexpr,
    lut1,
    OFF_A0: tl.constexpr,
    W_A0: tl.constexpr,
    OFF_A1: tl.constexpr,
    W_A1: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    BLOCKED: tl.constexpr,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
):
    """Output pieces FIRST..LAST-1 of the decoded rows (integer bits), in fp32."""
    a0 = tl.zeros([BLOCK_M, P0], tl.float32)
    a1 = tl.zeros([BLOCK_M, P1 if P1 > 0 else 16], tl.float32)
    a2 = tl.zeros([BLOCK_M, P2 if P2 > 0 else 16], tl.float32)
    a3 = tl.zeros([BLOCK_M, P3 if P3 > 0 else 16], tl.float32)
    if BLOCKED:
        # Pass FIRST decodes the instance held by piece FIRST.
        START: tl.constexpr = (
            (P0 if FIRST >= 1 else 0) + (P1 if FIRST >= 2 else 0) + (P2 if FIRST >= 3 else 0)
        )
        WIDTH: tl.constexpr = P0 if FIRST == 0 else P1 if FIRST == 1 else P2 if FIRST == 2 else P3
        a0, a1, a2, a3 = _accumulate_group(
            cold,
            w_ptr,
            alpha,
            a0,
            a1,
            a2,
            a3,
            rows,
            row_mask,
            START,
            WIDTH,
            BITS0 if WIDTH == P0 else BITS1,
            lut0 if WIDTH == P0 else lut1,
            OFF_A0 + START // C * ROWS * (C * W_A0 // 8),
            W_A0,
            OFF_A1 + START // C * ROWS * (C * W_A1 // 8),
            W_A1,
            N,
            P0,
            P1,
            P2,
            P3,
            ROWS,
            BLOCK_M,
            C,
            FIRST,
            LAST,
        )
    else:
        a0, a1, a2, a3 = _accumulate_group(
            cold,
            w_ptr,
            alpha,
            a0,
            a1,
            a2,
            a3,
            rows,
            row_mask,
            0,
            N,
            BITS0,
            lut0,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            N,
            P0,
            P1,
            P2,
            P3,
            ROWS,
            BLOCK_M,
            C,
            FIRST,
            LAST,
        )
    return a0, a1, a2, a3


@triton.jit
def _load_code_words(
    cold,
    j: tl.constexpr,
    rows,
    row_mask,
    OFF: tl.constexpr,
    W: tl.constexpr,
    ROWS: tl.constexpr,
    C: tl.constexpr,
):
    """Chunk ``j`` of one bit plane as ``[rows, C * W / 32]`` 32-bit words (codes LSB first)."""
    WORDS: tl.constexpr = C * W // 32
    plane = (cold + OFF + j * (ROWS * (C * W // 8))).to(tl.pointer_type(tl.uint32))
    offsets = rows[:, None] * WORDS + tl.arange(0, WORDS)[None, :]
    return tl.load(plane[:, None] + offsets, mask=row_mask[:, None], other=0)


@triton.jit
def _unpack_words(words, W: tl.constexpr, BLOCK_M: tl.constexpr, C: tl.constexpr):
    """``[BLOCK_M, C * W / 32]`` words of ``W``-bit codes into ``[BLOCK_M, C]`` codes."""
    K: tl.constexpr = 32 // W
    shifts = tl.arange(0, K) * W
    fields = (words[:, :, None] >> shifts[None, None, :]) & ((1 << W) - 1)
    return tl.reshape(fields, (BLOCK_M, C)).to(tl.int32)


@triton.jit
def _decode_half(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    b2,
    b3,
    w_ptr,
    alpha,
    lut,
    col0,
    START: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    N: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
):
    """Output columns ``col0..col0+HALF-1`` of the instance from its four chunks of code words."""
    acc = tl.zeros([BLOCK_M, HALF], tl.float32)
    cols = col0 + tl.arange(0, HALF)
    for j in tl.static_range(4):
        if j == 0:
            low, high = a0, b0
        elif j == 1:
            low, high = a1, b1
        elif j == 2:
            low, high = a2, b2
        else:
            low, high = a3, b3
        code = _unpack_words(low, W0, BLOCK_M, C)
        if W1 > 0:
            code |= _unpack_words(high, W1, BLOCK_M, C) << W0
        z = (alpha[:, None] * _dequantize(code, lut)).to(tl.bfloat16)
        krows = START + j * C + tl.arange(0, C)
        acc += tl.dot(z, tl.load(w_ptr + krows[:, None] * N + cols[None, :]))
    return acc


@triton.jit
def _decode_halves(
    cold,
    w_ptr,
    alpha,
    rows,
    row_mask,
    row_ptr,
    START: tl.constexpr,
    G: tl.constexpr,
    lut,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    ELEM: tl.constexpr,
):
    """Decode one mse instance of G coordinates into output columns START..START+G, in two halves.

    One accumulator of G columns and its staged operand do not fit tensor memory, and neither do
    two of G / 2 columns, so each half is accumulated and stored before the next. The codes are
    loaded once, as 32-bit words that stay in registers, and unpacked again for the second half.
    """
    tl.static_assert(G // C == 4, "a single instance of 512 values is four chunks of 128")
    a0 = _load_code_words(cold, 0, rows, row_mask, OFF0, W0, ROWS, C)
    a1 = _load_code_words(cold, 1, rows, row_mask, OFF0, W0, ROWS, C)
    a2 = _load_code_words(cold, 2, rows, row_mask, OFF0, W0, ROWS, C)
    a3 = _load_code_words(cold, 3, rows, row_mask, OFF0, W0, ROWS, C)
    if W1 > 0:
        b0 = _load_code_words(cold, 0, rows, row_mask, OFF1, W1, ROWS, C)
        b1 = _load_code_words(cold, 1, rows, row_mask, OFF1, W1, ROWS, C)
        b2 = _load_code_words(cold, 2, rows, row_mask, OFF1, W1, ROWS, C)
        b3 = _load_code_words(cold, 3, rows, row_mask, OFF1, W1, ROWS, C)
    else:  # one bit plane: placeholders, never unpacked
        b0, b1, b2, b3 = a0, a1, a2, a3
    HALF: tl.constexpr = G // 2
    lo = _decode_half(
        a0,
        a1,
        a2,
        a3,
        b0,
        b1,
        b2,
        b3,
        w_ptr,
        alpha,
        lut,
        START,
        START,
        W0,
        W1,
        N,
        HALF,
        BLOCK_M,
        C,
    )
    _store_piece(row_ptr, row_mask, lo, START, HALF, ELEM)
    hi = _decode_half(
        a0,
        a1,
        a2,
        a3,
        b0,
        b1,
        b2,
        b3,
        w_ptr,
        alpha,
        lut,
        START + HALF,
        START,
        W0,
        W1,
        N,
        HALF,
        BLOCK_M,
        C,
    )
    _store_piece(row_ptr, row_mask, hi, START + HALF, HALF, ELEM)


@triton.jit
def _store_piece(row_ptr, row_mask, acc, START, P: tl.constexpr, ELEM: tl.constexpr):
    # A reconstruction can exceed the largest finite value next to it: saturate, never store inf.
    if ELEM == tl.float8e4nv:
        acc = tl.minimum(tl.maximum(acc, -448.0), 448.0)
    elif ELEM == tl.float16:
        acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
    columns = START + tl.arange(0, P)
    tl.store(row_ptr[:, None] + columns[None, :], acc.to(ELEM), mask=row_mask[:, None])


@triton.jit
def _store_channels(row_ptr, row_mask, acc, channels, ELEM: tl.constexpr):
    """Store decoded values into the given channels of the rows (see ``_store_piece``)."""
    if ELEM == tl.float8e4nv:
        acc = tl.minimum(tl.maximum(acc, -448.0), 448.0)
    elif ELEM == tl.float16:
        acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
    tl.store(row_ptr[:, None] + channels[None, :], acc.to(ELEM), mask=row_mask[:, None])


@triton.jit
def _decode_group(
    cold,
    w_ptr,
    alpha,
    rows,
    row_mask,
    row_ptr,
    w0,
    w1,
    w2,
    w3,
    HIGH: tl.constexpr,
    COORD0: tl.constexpr,
    G: tl.constexpr,
    lut,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    N: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    ELEM: tl.constexpr,
):
    """Decode one group of fractional-bit rows into its channels: the high-bit group (``HIGH``)
    or the other, of the mask words ``w0..w3``. The channels are found after the products, so
    that their indices do not hold registers during them."""
    a0 = tl.zeros([BLOCK_M, Q0], tl.float32)
    a1 = tl.zeros([BLOCK_M, Q1 if Q1 > 0 else 16], tl.float32)
    a2 = tl.zeros([BLOCK_M, Q2 if Q2 > 0 else 16], tl.float32)
    a3 = tl.zeros([BLOCK_M, Q3 if Q3 > 0 else 16], tl.float32)
    low, high = _load_code_planes(cold[:, None], 0, rows, row_mask, OFF0, W0, OFF1, W1, ROWS, C)
    for j in range(G // C):
        # The next chunk's codes are read (usually from host memory) while this one is decoded.
        next_low, next_high = _load_code_planes(
            cold[:, None], j + 1, rows, row_mask & (j + 1 < G // C), OFF0, W0, OFF1, W1, ROWS, C
        )
        code = _unpack_planes(low, high, W0, W1, BLOCK_M, C)
        z = (alpha[:, None] * _dequantize(code, lut)).to(tl.bfloat16)
        low = next_low
        high = next_high
        krows = COORD0 + j * C + tl.arange(0, C)
        a0 += tl.dot(z, _decode_matrix(w_ptr, krows, COORD0, Q0, N))
        if Q1 > 0:
            a1 += tl.dot(z, _decode_matrix(w_ptr, krows, COORD0 + Q0, Q1, N))
        if Q2 > 0:
            a2 += tl.dot(z, _decode_matrix(w_ptr, krows, COORD0 + Q0 + Q1, Q2, N))
        if Q3 > 0:
            a3 += tl.dot(z, _decode_matrix(w_ptr, krows, COORD0 + Q0 + Q1 + Q2, Q3, N))
    c0, c1, c2, c3 = _group_channel_pieces(w0, w1, w2, w3, HIGH, P0, P1, P2, P3, Q0, Q1, Q2, Q3)
    _store_channels(row_ptr, row_mask, a0, c0, ELEM)
    if Q1 > 0:
        _store_channels(row_ptr, row_mask, a1, c1, ELEM)
    if Q2 > 0:
        _store_channels(row_ptr, row_mask, a2, c2, ELEM)
    if Q3 > 0:
        _store_channels(row_ptr, row_mask, a3, c3, ELEM)


@triton.jit
def _decode_fractional(
    page_ptr,
    cold_ptr,
    cold_page_bytes,
    hot_ptr,
    table_ptr,
    w_ptr,
    lut_ptr,
    ELEM: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    QS: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    C: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    SUB_ROWS: tl.constexpr,
    OFF_A0: tl.constexpr,
    W_A0: tl.constexpr,
    OFF_A1: tl.constexpr,
    W_A1: tl.constexpr,
    OFF_B0: tl.constexpr,
    W_B0: tl.constexpr,
    OFF_B1: tl.constexpr,
    W_B1: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    OFF_MASK: tl.constexpr,
    OFF_RAW: tl.constexpr,
    RAW: tl.constexpr,
    RAW_BLOCK: tl.constexpr,
):
    """Decode one KV head (``TILE_ROWS`` rows) of one buffer of one Page with fractional bits,
    ``SUB_ROWS`` rows at a time: each group into its channels, as the head's mask records them."""
    G: tl.constexpr = N // 2
    page = tl.program_id(0)
    buffer = tl.program_id(1)
    head = tl.program_id(2)
    lane = tl.arange(0, BLOCK_M)
    row_mask = lane < SUB_ROWS
    # Page-index pairs (dst, src) live in device memory (ColdPageIndexLocation.DEVICE).
    dst = tl.load(page_ptr + 2 * page).to(tl.int64)
    src = tl.load(page_ptr + 2 * page + 1).to(tl.int64)
    # Buffer table: hot buffer offset from hot_ptr and Slot stride (elements), cold offset (bytes).
    hot_offset = tl.load(table_ptr + 3 * buffer)
    slot_elements = tl.load(table_ptr + 3 * buffer + 1)
    cold_offset = tl.load(table_ptr + 3 * buffer + 2)
    ALIGN: tl.constexpr = 16 if ELEM == tl.float8e4nv else 8
    hot = hot_ptr + tl.multiple_of(hot_offset + dst * slot_elements, ALIGN)
    cold = cold_ptr + tl.multiple_of(src * cold_page_bytes + cold_offset, 16)
    mask_ptr = cold + OFF_MASK + head * (N // 8)
    w0 = _load_mask_words(mask_ptr, 0, P0)
    w1 = w0
    w2 = w0
    w3 = w0
    if P1 > 0:
        w1 = _load_mask_words(mask_ptr, P0, P1)
    if P2 > 0:
        w2 = _load_mask_words(mask_ptr, P0 + P1, P2)
    if P3 > 0:
        w3 = _load_mask_words(mask_ptr, P0 + P1 + P2, P3)
    cold_rows = cold + tl.zeros([BLOCK_M], tl.int64)
    scale_ptr = (cold + OFF_SCALE).to(tl.pointer_type(tl.bfloat16))
    for part in range(TILE_ROWS // SUB_ROWS):
        rows = head * TILE_ROWS + part * SUB_ROWS + lane
        row_ptr = hot + rows.to(tl.int64) * STRIDE + QS
        alpha_hi = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        alpha_lo = tl.load(scale_ptr + ROWS + rows, mask=row_mask, other=0.0).to(tl.float32)
        _decode_group(
            cold_rows,
            w_ptr,
            alpha_hi,
            rows,
            row_mask,
            row_ptr,
            w0,
            w1,
            w2,
            w3,
            True,
            0,
            G,
            lut_ptr,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            N,
            P0,
            P1,
            P2,
            P3,
            Q0,
            Q1,
            Q2,
            Q3,
            ROWS,
            BLOCK_M,
            C,
            ELEM,
        )
        _decode_group(
            cold_rows,
            w_ptr,
            alpha_lo,
            rows,
            row_mask,
            row_ptr,
            w0,
            w1,
            w2,
            w3,
            False,
            G,
            G,
            lut_ptr + LUT_STRIDE,
            OFF_B0,
            W_B0,
            OFF_B1,
            W_B1,
            N,
            P0,
            P1,
            P2,
            P3,
            Q0,
            Q1,
            Q2,
            Q3,
            ROWS,
            BLOCK_M,
            C,
            ELEM,
        )
        if RAW > 0:
            raw_cols = tl.arange(0, RAW_BLOCK)
            raw_dst = tl.where(raw_cols < QS, raw_cols, raw_cols + N)
            raw_mask = row_mask[:, None] & (raw_cols < RAW)[None, :]
            raw_ptr = (cold + OFF_RAW).to(tl.pointer_type(ELEM))
            values = tl.load(raw_ptr + rows[:, None] * RAW + raw_cols[None, :], mask=raw_mask)
            tl.store(
                hot + rows.to(tl.int64)[:, None] * STRIDE + raw_dst[None, :],
                values,
                mask=raw_mask,
            )


@triton.jit
def _decode_tiles(
    page_ptr,
    cold_ptr,
    cold_page_bytes,
    num_pages,
    hot_ptr,
    table_ptr,
    w_ptr,
    lut_ptr,
    ELEM: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    QS: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    C: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    MODE: tl.constexpr,
    BLOCKED: tl.constexpr,
    PASS: tl.constexpr,
    BITS0: tl.constexpr,
    CB0: tl.constexpr,
    BITS1: tl.constexpr,
    CB1: tl.constexpr,
    OFF_A0: tl.constexpr,
    W_A0: tl.constexpr,
    OFF_A1: tl.constexpr,
    W_A1: tl.constexpr,
    OFF_B0: tl.constexpr,
    W_B0: tl.constexpr,
    OFF_B1: tl.constexpr,
    W_B1: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    OFF_MASK: tl.constexpr,
    OFF_RAW: tl.constexpr,
    RAW: tl.constexpr,
    RAW_BLOCK: tl.constexpr,
    HALVES: tl.constexpr,
):
    # HALVES (an mse pass that decodes a single instance of 512 values): the instance's output
    # columns are accumulated in two halves, so that both accumulators fit tensor memory; each
    # chunk of codes is still read once.
    tile_page = tl.program_id(0)
    buffer = tl.program_id(1)
    tile = tl.program_id(2)
    lane = tl.arange(0, BLOCK_M)
    if PAGES_PER_TILE > 1:
        # Buffers with few rows per Page: one tile takes the rows of several Pages.
        page = tile_page * PAGES_PER_TILE + lane // ROWS_PAD
        rows = lane % ROWS_PAD
        row_mask = (rows < ROWS) & (page < num_pages)
    else:
        # One tile of rows of one Page.
        page = tile_page + lane * 0
        rows = tile * TILE_ROWS + lane
        row_mask = (lane < TILE_ROWS) & (rows < ROWS)
    # Page-index pairs (dst, src) live in device memory (ColdPageIndexLocation.DEVICE).
    pair_ptr = page_ptr + 2 * page.to(tl.int64)
    dst = tl.load(pair_ptr, mask=row_mask, other=0).to(tl.int64)
    src = tl.load(pair_ptr + 1, mask=row_mask, other=0).to(tl.int64)
    # Buffer table: hot buffer offset from hot_ptr and Slot stride (elements), cold offset (bytes).
    hot_offset = tl.load(table_ptr + 3 * buffer)
    slot_elements = tl.load(table_ptr + 3 * buffer + 1)
    cold_offset = tl.load(table_ptr + 3 * buffer + 2)
    # Slots, cold pages, and buffer sections are 16-byte aligned (checked at configure time):
    # offsets from the aligned pointer arguments let the loads and stores use 16-byte vectors.
    ALIGN: tl.constexpr = 16 if ELEM == tl.float8e4nv else 8
    hot = hot_ptr + tl.multiple_of(hot_offset + dst * slot_elements, ALIGN)
    cold_rows = tl.multiple_of(src * cold_page_bytes + cold_offset, 16)
    cold = cold_ptr + cold_rows
    row_ptr = hot + rows.to(tl.int64) * STRIDE + QS
    scale_ptr = (cold + OFF_SCALE).to(tl.pointer_type(tl.bfloat16))
    if BLOCKED:
        alpha = tl.load(scale_ptr + PASS * ROWS + rows, mask=row_mask, other=0.0).to(tl.float32)
    else:
        alpha = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    # Rows with several instances decode one instance per launch (PASS): each reads only its
    # own codes and accumulates only its own output columns.
    if BLOCKED:
        FIRST: tl.constexpr = PASS
        LAST: tl.constexpr = PASS + 1
    else:
        FIRST: tl.constexpr = 0
        LAST: tl.constexpr = 4
    if HALVES:
        # A single-piece mse row of 512 values, decoded by one pass.
        START: tl.constexpr = (
            (P0 if FIRST >= 1 else 0) + (P1 if FIRST >= 2 else 0) + (P2 if FIRST >= 3 else 0)
        )
        WIDTH: tl.constexpr = P0 if FIRST == 0 else P1 if FIRST == 1 else P2 if FIRST == 2 else P3
        _decode_halves(
            cold,
            w_ptr,
            alpha,
            rows,
            row_mask,
            row_ptr,
            START,
            WIDTH,
            lut_ptr if WIDTH == P0 else lut_ptr + LUT_STRIDE,
            OFF_A0 + START // C * ROWS * (C * W_A0 // 8),
            W_A0,
            OFF_A1 + START // C * ROWS * (C * W_A1 // 8),
            W_A1,
            N,
            ROWS,
            BLOCK_M,
            C,
            ELEM,
        )
    else:
        a0, a1, a2, a3 = _decode_rows(
            cold,
            w_ptr,
            alpha,
            rows,
            row_mask,
            N,
            P0,
            P1,
            P2,
            P3,
            BITS0,
            lut_ptr,
            BITS1,
            lut_ptr + LUT_STRIDE,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            ROWS,
            BLOCK_M,
            C,
            BLOCKED,
            FIRST,
            LAST,
        )
        if FIRST == 0:
            _store_piece(row_ptr, row_mask, a0, 0, P0, ELEM)
        if P1 > 0 and FIRST <= 1 and LAST > 1:
            _store_piece(row_ptr, row_mask, a1, P0, P1, ELEM)
        if P2 > 0 and FIRST <= 2 and LAST > 2:
            _store_piece(row_ptr, row_mask, a2, P0 + P1, P2, ELEM)
        if P3 > 0 and LAST > 3:
            _store_piece(row_ptr, row_mask, a3, P0 + P1 + P2, P3, ELEM)

    if RAW > 0 and FIRST == 0:
        raw_cols = tl.arange(0, RAW_BLOCK)
        raw_dst = tl.where(raw_cols < QS, raw_cols, raw_cols + N)
        raw_mask = row_mask[:, None] & (raw_cols < RAW)[None, :]
        raw_ptr = (cold + OFF_RAW).to(tl.pointer_type(ELEM))
        values = tl.load(raw_ptr[:, None] + rows[:, None] * RAW + raw_cols[None, :], mask=raw_mask)
        tl.store(
            hot[:, None] + rows.to(tl.int64)[:, None] * STRIDE + raw_dst[None, :],
            values,
            mask=raw_mask,
        )


@triton.jit(do_not_specialize=["page_ptr", "cold_page_bytes", "num_pages"])
def _tq_decode_kernel(
    page_ptr,
    cold_ptr,
    cold_page_bytes,
    num_pages,
    hot_ptr,
    table_ptr,
    w_ptr,
    lut_ptr,
    ELEM: tl.constexpr,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    QS: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    C: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    PAGES_PER_TILE: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    MODE: tl.constexpr,
    BLOCKED: tl.constexpr,
    PASS: tl.constexpr,
    BITS0: tl.constexpr,
    CB0: tl.constexpr,
    BITS1: tl.constexpr,
    CB1: tl.constexpr,
    OFF_A0: tl.constexpr,
    W_A0: tl.constexpr,
    OFF_A1: tl.constexpr,
    W_A1: tl.constexpr,
    OFF_B0: tl.constexpr,
    W_B0: tl.constexpr,
    OFF_B1: tl.constexpr,
    W_B1: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    OFF_MASK: tl.constexpr,
    OFF_RAW: tl.constexpr,
    RAW: tl.constexpr,
    RAW_BLOCK: tl.constexpr,
    HALVES: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    Q2: tl.constexpr,
    Q3: tl.constexpr,
    SUB_ROWS: tl.constexpr,
):
    # Fractional bits (MODE 1): one program per KV head of a Page (see _decode_fractional).
    if MODE == 1:
        _decode_fractional(
            page_ptr,
            cold_ptr,
            cold_page_bytes,
            hot_ptr,
            table_ptr,
            w_ptr,
            lut_ptr,
            ELEM,
            N,
            STRIDE,
            QS,
            P0,
            P1,
            P2,
            P3,
            Q0,
            Q1,
            Q2,
            Q3,
            C,
            ROWS,
            BLOCK_M,
            TILE_ROWS,
            SUB_ROWS,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            OFF_B0,
            W_B0,
            OFF_B1,
            W_B1,
            OFF_SCALE,
            OFF_MASK,
            OFF_RAW,
            RAW,
            RAW_BLOCK,
        )
    else:
        _decode_tiles(
            page_ptr,
            cold_ptr,
            cold_page_bytes,
            num_pages,
            hot_ptr,
            table_ptr,
            w_ptr,
            lut_ptr,
            ELEM,
            N,
            STRIDE,
            QS,
            P0,
            P1,
            P2,
            P3,
            C,
            ROWS,
            BLOCK_M,
            TILE_ROWS,
            PAGES_PER_TILE,
            ROWS_PAD,
            MODE,
            BLOCKED,
            PASS,
            BITS0,
            CB0,
            BITS1,
            CB1,
            OFF_A0,
            W_A0,
            OFF_A1,
            W_A1,
            OFF_B0,
            W_B0,
            OFF_B1,
            W_B1,
            OFF_SCALE,
            OFF_MASK,
            OFF_RAW,
            RAW,
            RAW_BLOCK,
            HALVES,
        )


@triton.jit(do_not_specialize=["page_addr", "cold_base", "cold_page_bytes"])
def _copy_kernel(
    page_addr,
    cold_base,
    cold_page_bytes,
    table_ptr,
    ENCODE: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy whole buffers between hot slots and cold pages in ``UNIT``-typed words."""
    page = tl.program_id(0).to(tl.int64)
    buffer = tl.program_id(1)
    block = tl.program_id(2).to(tl.int64)
    # Page-index pairs (dst, src) live in device memory (ColdPageIndexLocation.DEVICE).
    page_ptr = page_addr.to(tl.pointer_type(tl.int32))
    dst = tl.load(page_ptr + 2 * page).to(tl.int64)
    src = tl.load(page_ptr + 2 * page + 1).to(tl.int64)
    hot_base = tl.load(table_ptr + 4 * buffer)
    slot_bytes = tl.load(table_ptr + 4 * buffer + 1)
    words = tl.load(table_ptr + 4 * buffer + 2)
    cold_offset = tl.load(table_ptr + 4 * buffer + 3)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < words
    if ENCODE:
        source = (hot_base + src * slot_bytes).to(tl.pointer_type(UNIT))
        target = (cold_base + dst * cold_page_bytes + cold_offset).to(tl.pointer_type(UNIT))
    else:
        source = (cold_base + src * cold_page_bytes + cold_offset).to(tl.pointer_type(UNIT))
        target = (hot_base + dst * slot_bytes).to(tl.pointer_type(UNIT))
    tl.store(target + offsets, tl.load(source + offsets, mask=mask), mask=mask)
