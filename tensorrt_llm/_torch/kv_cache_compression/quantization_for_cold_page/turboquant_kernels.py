# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Triton kernels that encode hot KV rows into TurboQuant cold pages and decode them back.

One program handles ``BLOCK_M`` rows (K or V vectors) of one buffer of one Page. It reads the
rows from the GPU slot, rotates them on tensor cores, quantizes every rotated coordinate with the
Lloyd-Max codebook, packs the codes into bit planes, and writes them straight into the cold page,
which usually lives in pinned host memory. Decoding runs the same steps backwards. Every address is
computed in 64 bits from the KVCM base pointers; Page index pairs are read on the device.

Cold layout of one quantized buffer (offsets relative to the buffer's section, each region padded
to 16 bytes; ``R`` rows, ``G`` coordinates per group, chunks of ``C`` coordinates):

* code planes: for each group and each bit plane of width ``w`` (``bits`` split into 4, 2 or 1
  bit planes), ``[G / C chunks][R rows][C * w / 8 bytes]``;
* QJL signs (prod): ``[N / C chunks][R rows][C / 8 bytes]``;
* per-row bf16 scalars ``[scalar][R]``: the mse scale; prod adds the residual norm;
  fractional bits store one scale per group;
* fractional bits: one ``N``-bit outlier-channel mask per KV head;
* the elements of each row outside the quantized range, copied unchanged.
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
def _rotate_chunk(
    w_ptr,
    cols,
    x0,
    x1,
    x2,
    x3,
    pos0,
    pos1,
    pos2,
    pos3,
    N: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    GATHER: tl.constexpr,
    PIECES: tl.constexpr,
):
    """Rotated coordinates ``cols`` of the rows held as power-of-two pieces ``x0..x3``.

    ``w_ptr`` is the row-major ``N x N`` matrix whose row ``c`` maps input channel ``c`` to the
    rotated coordinates; with ``GATHER`` input channel ``c`` uses matrix row ``pos[c]``. Bit ``i``
    of ``PIECES`` selects piece ``i``: a block-diagonal matrix skips the pieces of other blocks.
    """
    if PIECES & 1:
        if GATHER:
            r0 = pos0
        else:
            r0 = tl.arange(0, P0)
        acc = tl.dot(x0, tl.load(w_ptr + r0[:, None] * N + cols[None, :]))
    else:
        acc = tl.zeros([x0.shape[0], cols.shape[0]], tl.float32)
    if P1 > 0 and PIECES & 2:
        if GATHER:
            r1 = pos1
        else:
            r1 = P0 + tl.arange(0, P1)
        acc += tl.dot(x1, tl.load(w_ptr + r1[:, None] * N + cols[None, :]))
    if P2 > 0 and PIECES & 4:
        if GATHER:
            r2 = pos2
        else:
            r2 = P0 + P1 + tl.arange(0, P2)
        acc += tl.dot(x2, tl.load(w_ptr + r2[:, None] * N + cols[None, :]))
    if P3 > 0 and PIECES & 8:
        if GATHER:
            r3 = pos3
        else:
            r3 = P0 + P1 + P2 + tl.arange(0, P3)
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
def _piece_energy_bits(x, row_mask):
    """Energy of every channel over the rows; ``row_mask`` excludes padding and non-finite rows."""
    energy = tl.sum(tl.where(row_mask[:, None], x * x, 0.0), axis=0)
    # NaN energies (garbage rows of a partial Page) rank lowest, so exactly G channels win.
    return tl.where(energy == energy, energy, 0.0).to(tl.int32, bitcast=True)


@triton.jit
def _count_at_least(bits, threshold):
    return tl.sum((bits >= threshold).to(tl.int32), axis=0)


@triton.jit
def _outlier_positions(outlier, before, START: tl.constexpr, P: tl.constexpr, G: tl.constexpr):
    """Rotation-matrix row of every channel: outliers first, then regular channels, in order."""
    flag = outlier.to(tl.int32)
    outliers_before = before + tl.cumsum(flag, axis=0) - flag
    channel = START + tl.arange(0, P)
    position = tl.where(outlier, outliers_before, G + channel - outliers_before)
    # A mask without exactly G outliers cannot come from the encoder; stay inside the matrix.
    return tl.minimum(tl.maximum(position, 0), 2 * G - 1), before + tl.sum(flag, axis=0)


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
    ``e*`` hold the non-negative float energies as int32 bit patterns, which order like the
    floats; a binary search over the bits finds the median without sorting.
    """
    median = 0
    for b in tl.static_range(31):
        candidate = median | (1 << (30 - b))
        count = _count_at_least(e0, candidate)
        if P1 > 0:
            count += _count_at_least(e1, candidate)
        if P2 > 0:
            count += _count_at_least(e2, candidate)
        if P3 > 0:
            count += _count_at_least(e3, candidate)
        median = tl.where(count >= G, candidate, median)
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
def _store_mask(mask_ptr, outlier, START: tl.constexpr, P: tl.constexpr):
    grouped = tl.reshape(outlier.to(tl.int32), (P // 8, 8))
    shifts = tl.arange(0, 8)
    packed = tl.sum(grouped << shifts[None, :], axis=1).to(tl.uint8)
    tl.store(mask_ptr + START // 8 + tl.arange(0, P // 8), packed)


@triton.jit
def _load_mask(mask_ptr, START: tl.constexpr, P: tl.constexpr):
    packed = tl.load(mask_ptr + START // 8 + tl.arange(0, P // 8)).to(tl.int32)
    bits = (packed[:, None] >> tl.arange(0, 8)[None, :]) & 1
    return tl.reshape(bits, (P,)) != 0


@triton.jit
def _quantize_group(
    cold,
    w_ptr,
    x0,
    x1,
    x2,
    x3,
    pos0,
    pos1,
    pos2,
    pos3,
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
    GATHER: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    PIECES: tl.constexpr,
):
    """Quantize rotated coordinates [COORD0, COORD0 + G); return the least-squares scale."""
    dot_yc = tl.zeros([BLOCK_M], tl.float32)
    dot_cc = tl.zeros([BLOCK_M], tl.float32)
    for j in range(G // C):
        cols = COORD0 + j * C + tl.arange(0, C)
        y = _rotate_chunk(
            w_ptr, cols, x0, x1, x2, x3, pos0, pos1, pos2, pos3, N, P0, P1, P2, P3, GATHER, PIECES
        )
        code, value = _quantize(y * inv_norm[:, None], BITS, CB, lut)
        dot_yc += tl.sum(y * value, axis=1)
        dot_cc += tl.sum(value * value, axis=1)
        _store_codes(cold[:, None], code, j, rows, row_mask, OFF0, W0, OFF1, W1, ROWS, BLOCK_M, C)
    return tl.where(dot_cc > 0, dot_yc / dot_cc, 0.0)


@triton.jit
def _qjl_piece(acc, residual, s_ptr, krows, START: tl.constexpr, P: tl.constexpr, N: tl.constexpr):
    columns = START + tl.arange(0, P)
    return acc + tl.dot(residual, tl.load(s_ptr + krows[:, None] * N + columns[None, :]))


@triton.jit
def _store_signs(
    cold,
    projection,
    rows,
    row_mask,
    OFF: tl.constexpr,
    START: tl.constexpr,
    P: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
):
    """Store the signs of projections [START, START + P) into the chunk-major QJL plane."""
    CB: tl.constexpr = C // 8
    packed = _pack((projection >= 0).to(tl.int32), 1, BLOCK_M, P)
    byte = tl.arange(0, P // 8)
    chunk = START // C + byte // CB
    offsets = chunk[None, :] * (ROWS * CB) + rows[:, None] * CB + (byte % CB)[None, :]
    tl.store(cold + OFF + offsets, packed, mask=row_mask[:, None])


@triton.jit
def _encode_block(
    cold,
    w_ptr,
    s_ptr,
    h0,
    h1,
    h2,
    h3,
    norm,
    prescale,
    rows,
    row_mask,
    MODE: tl.constexpr,
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
    OFF_QJL: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    ALPHA_AT: tl.constexpr,
    RHO_AT: tl.constexpr,
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

    Stores the codes, the least-squares scale at scalar ``ALPHA_AT`` and, for prod, the QJL signs
    of the residual (taken in the rotated domain, where S is i.i.d. Gaussian as well) and the
    residual norm at scalar ``RHO_AT``.
    """
    inv_norm = tl.where(norm > 0, tl.sqrt(G * 1.0) * tl.math.rsqrt(norm), 0.0)
    alpha = _quantize_group(
        cold,
        w_ptr,
        h0,
        h1,
        h2,
        h3,
        rows,
        rows,
        rows,
        rows,
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
        False,
        ROWS,
        BLOCK_M,
        C,
        PIECES,
    )
    scale_ptr = (cold + OFF_SCALE).to(tl.pointer_type(tl.bfloat16))
    tl.store(scale_ptr + ALPHA_AT + rows, (alpha / prescale).to(tl.bfloat16), mask=row_mask)
    if MODE == 1:
        q0 = tl.zeros([BLOCK_M, P0], tl.float32)
        q1 = tl.zeros([BLOCK_M, P1 if P1 > 0 else 16], tl.float32)
        q2 = tl.zeros([BLOCK_M, P2 if P2 > 0 else 16], tl.float32)
        q3 = tl.zeros([BLOCK_M, P3 if P3 > 0 else 16], tl.float32)
        residual_norm = tl.zeros([BLOCK_M], tl.float32)
        for j in range(G // C):
            cols = COORD0 + j * C + tl.arange(0, C)
            y = _rotate_chunk(
                w_ptr,
                cols,
                h0,
                h1,
                h2,
                h3,
                rows,
                rows,
                rows,
                rows,
                N,
                P0,
                P1,
                P2,
                P3,
                False,
                PIECES,
            )
            code, value = _quantize(y * inv_norm[:, None], BITS, CB, lut)
            residual = y - alpha[:, None] * value
            residual_norm += tl.sum(residual * residual, axis=1)
            r16 = residual.to(tl.float16)
            if PIECES & 1:
                q0 = _qjl_piece(q0, r16, s_ptr, cols, 0, P0, N)
            if P1 > 0 and PIECES & 2:
                q1 = _qjl_piece(q1, r16, s_ptr, cols, P0, P1, N)
            if P2 > 0 and PIECES & 4:
                q2 = _qjl_piece(q2, r16, s_ptr, cols, P0 + P1, P2, N)
            if P3 > 0 and PIECES & 8:
                q3 = _qjl_piece(q3, r16, s_ptr, cols, P0 + P1 + P2, P3, N)
        if PIECES & 1:
            _store_signs(cold[:, None], q0, rows, row_mask, OFF_QJL, 0, P0, ROWS, BLOCK_M, C)
        if P1 > 0 and PIECES & 2:
            _store_signs(cold[:, None], q1, rows, row_mask, OFF_QJL, P0, P1, ROWS, BLOCK_M, C)
        if P2 > 0 and PIECES & 4:
            _store_signs(cold[:, None], q2, rows, row_mask, OFF_QJL, P0 + P1, P2, ROWS, BLOCK_M, C)
        if P3 > 0 and PIECES & 8:
            _store_signs(
                cold[:, None], q3, rows, row_mask, OFF_QJL, P0 + P1 + P2, P3, ROWS, BLOCK_M, C
            )
        tl.store(
            scale_ptr + RHO_AT + rows,
            (tl.sqrt(residual_norm) / prescale).to(tl.bfloat16),
            mask=row_mask,
        )


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
    s_ptr,
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
    OFF_QJL: tl.constexpr,
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
        s_ptr,
        h,
        h,
        h,
        h,
        tl.sum(x * x, axis=1),
        prescale,
        rows,
        row_mask,
        0,
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
        OFF_QJL,
        OFF_SCALE,
        ALPHA_AT,
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


@triton.jit(do_not_specialize=["page_ptr", "cold_page_bytes", "num_pages"])
def _tq_encode_kernel(
    page_ptr,
    cold_ptr,
    cold_page_bytes,
    num_pages,
    hot_ptr,
    table_ptr,
    w_ptr,
    s_ptr,
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
    OFF_QJL: tl.constexpr,
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
        # Fractional bits: a tile is one KV head of the Page, whose tokens share the channel split.
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

    if BLOCKED == 2:
        # Rows wider than 512 (mse): every piece is an independent instance, its codes in the
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
            s_ptr,
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
            OFF_QJL,
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
                s_ptr,
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
                OFF_QJL,
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
                s_ptr,
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
                OFF_QJL,
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
                s_ptr,
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
                OFF_QJL,
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

        if MODE == 2:
            G: tl.constexpr = N // 2
            # Rows holding NaN or Inf (garbage tokens of a partial Page) do not vote on the split.
            voters = row_mask & (amax < float("inf"))
            e0 = _piece_energy_bits(x0, voters)
            e1 = e0
            e2 = e0
            e3 = e0
            if P1 > 0:
                e1 = _piece_energy_bits(x1, voters)
            if P2 > 0:
                e2 = _piece_energy_bits(x2, voters)
            if P3 > 0:
                e3 = _piece_energy_bits(x3, voters)
            o0, o1, o2, o3 = _select_outliers(e0, e1, e2, e3, P0, P1, P2, P3, G)
            mask_ptr = cold_ptr + tl.max(cold_rows, axis=0) + OFF_MASK + tile * (N // 8)
            _store_mask(mask_ptr, o0, 0, P0)
            pos0, before = _outlier_positions(o0, 0, 0, P0, G)
            norm_hi = tl.sum(tl.where(o0[None, :], x0 * x0, 0.0), axis=1)
            norm_all = tl.sum(x0 * x0, axis=1)
            pos1 = pos0
            pos2 = pos0
            pos3 = pos0
            if P1 > 0:
                _store_mask(mask_ptr, o1, P0, P1)
                pos1, before = _outlier_positions(o1, before, P0, P1, G)
                norm_hi += tl.sum(tl.where(o1[None, :], x1 * x1, 0.0), axis=1)
                norm_all += tl.sum(x1 * x1, axis=1)
            if P2 > 0:
                _store_mask(mask_ptr, o2, P0 + P1, P2)
                pos2, before = _outlier_positions(o2, before, P0 + P1, P2, G)
                norm_hi += tl.sum(tl.where(o2[None, :], x2 * x2, 0.0), axis=1)
                norm_all += tl.sum(x2 * x2, axis=1)
            if P3 > 0:
                _store_mask(mask_ptr, o3, P0 + P1 + P2, P3)
                pos3, before = _outlier_positions(o3, before, P0 + P1 + P2, P3, G)
                norm_hi += tl.sum(tl.where(o3[None, :], x3 * x3, 0.0), axis=1)
                norm_all += tl.sum(x3 * x3, axis=1)
            norm_lo = tl.maximum(norm_all - norm_hi, 0.0)
            root_g = tl.sqrt(G * 1.0)
            inv_hi = tl.where(norm_hi > 0, root_g * tl.math.rsqrt(norm_hi), 0.0)
            inv_lo = tl.where(norm_lo > 0, root_g * tl.math.rsqrt(norm_lo), 0.0)
            alpha_hi = _quantize_group(
                cold,
                w_ptr,
                x0.to(tl.float16),
                x1.to(tl.float16),
                x2.to(tl.float16),
                x3.to(tl.float16),
                pos0,
                pos1,
                pos2,
                pos3,
                inv_hi,
                rows,
                row_mask,
                0,
                G,
                BITS0,
                CB0,
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
                True,
                ROWS,
                BLOCK_M,
                C,
                15,
            )
            alpha_lo = _quantize_group(
                cold,
                w_ptr,
                x0.to(tl.float16),
                x1.to(tl.float16),
                x2.to(tl.float16),
                x3.to(tl.float16),
                pos0,
                pos1,
                pos2,
                pos3,
                inv_lo,
                rows,
                row_mask,
                G,
                G,
                BITS1,
                CB1,
                lut1,
                OFF_B0,
                W_B0,
                OFF_B1,
                W_B1,
                N,
                P0,
                P1,
                P2,
                P3,
                True,
                ROWS,
                BLOCK_M,
                C,
                15,
            )
            scale_ptr = (cold + OFF_SCALE).to(tl.pointer_type(tl.bfloat16))
            tl.store(scale_ptr + rows, (alpha_hi / prescale).to(tl.bfloat16), mask=row_mask)
            tl.store(scale_ptr + ROWS + rows, (alpha_lo / prescale).to(tl.bfloat16), mask=row_mask)
        else:
            h0 = x0.to(tl.float16)
            h1 = x1.to(tl.float16)
            h2 = x2.to(tl.float16)
            h3 = x3.to(tl.float16)
            norm_first = tl.sum(x0 * x0, axis=1)
            norm_rest = tl.zeros([BLOCK_M], tl.float32)
            if P1 > 0:
                norm_rest += tl.sum(x1 * x1, axis=1)
            if P2 > 0:
                norm_rest += tl.sum(x2 * x2, axis=1)
            if P3 > 0:
                norm_rest += tl.sum(x3 * x3, axis=1)
            if BLOCKED == 1:
                # Rows wider than 512 (prod): the first piece and the rest are independent instances.
                _encode_block(
                    cold,
                    w_ptr,
                    s_ptr,
                    h0,
                    h1,
                    h2,
                    h3,
                    norm_first,
                    prescale,
                    rows,
                    row_mask,
                    MODE,
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
                    OFF_QJL,
                    OFF_SCALE,
                    0,
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
                _encode_block(
                    cold,
                    w_ptr,
                    s_ptr,
                    h0,
                    h1,
                    h2,
                    h3,
                    norm_rest,
                    prescale,
                    rows,
                    row_mask,
                    MODE,
                    P0,
                    N - P0,
                    14,
                    BITS1,
                    CB1,
                    lut1,
                    OFF_B0,
                    W_B0,
                    OFF_B1,
                    W_B1,
                    OFF_QJL,
                    OFF_SCALE,
                    ROWS,
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
                _encode_block(
                    cold,
                    w_ptr,
                    s_ptr,
                    h0,
                    h1,
                    h2,
                    h3,
                    norm_first + norm_rest,
                    prescale,
                    rows,
                    row_mask,
                    MODE,
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
                    OFF_QJL,
                    OFF_SCALE,
                    0,
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


@triton.jit
def _decode_matrix(w_ptr, krows, cols, pos, N: tl.constexpr, GATHER: tl.constexpr):
    """Tile [krows, cols] of the decode matrix: row k of it maps rotated coordinate k to channels."""
    if GATHER:
        # Fractional bits: channel c reads row pos[c] of the encode matrix, transposed.
        return tl.load(w_ptr + pos[None, :] * N + krows[:, None])
    else:
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
    pos0,
    pos1,
    pos2,
    pos3,
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
    GATHER: tl.constexpr,
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
            a0 += tl.dot(z, _decode_matrix(w_ptr, krows, tl.arange(0, P0), pos0, N, GATHER))
        if P1 > 0 and FIRST <= 1 and 1 < LAST:
            a1 += tl.dot(z, _decode_matrix(w_ptr, krows, P0 + tl.arange(0, P1), pos1, N, GATHER))
        if P2 > 0 and FIRST <= 2 and 2 < LAST:
            a2 += tl.dot(
                z, _decode_matrix(w_ptr, krows, P0 + P1 + tl.arange(0, P2), pos2, N, GATHER)
            )
        if P3 > 0 and 3 < LAST:
            a3 += tl.dot(
                z, _decode_matrix(w_ptr, krows, P0 + P1 + P2 + tl.arange(0, P3), pos3, N, GATHER)
            )
    return a0, a1, a2, a3


@triton.jit
def _accumulate_signs(
    cold,
    s_ptr,
    weight,
    a0,
    a1,
    a2,
    a3,
    rows,
    row_mask,
    OFF: tl.constexpr,
    N: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    P3: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    J0: tl.constexpr,
    J1: tl.constexpr,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
):
    """Add ``weight * (S Pi)^T sign`` of sign chunks J0..J1-1 to output pieces FIRST..LAST-1."""
    CB: tl.constexpr = C // 8
    for j in range(J0, J1):
        offsets = j * (ROWS * CB) + rows[:, None] * CB + tl.arange(0, CB)[None, :]
        bits = _unpack(
            tl.load(cold[:, None] + OFF + offsets, mask=row_mask[:, None], other=0), 1, BLOCK_M, C
        )
        z = tl.where(bits != 0, weight[:, None], -weight[:, None]).to(tl.bfloat16)
        krows = j * C + tl.arange(0, C)
        if FIRST <= 0:
            a0 += tl.dot(z, _decode_matrix(s_ptr, krows, tl.arange(0, P0), krows, N, False))
        if P1 > 0 and FIRST <= 1 and 1 < LAST:
            a1 += tl.dot(z, _decode_matrix(s_ptr, krows, P0 + tl.arange(0, P1), krows, N, False))
        if P2 > 0 and FIRST <= 2 and 2 < LAST:
            a2 += tl.dot(
                z, _decode_matrix(s_ptr, krows, P0 + P1 + tl.arange(0, P2), krows, N, False)
            )
        if P3 > 0 and 3 < LAST:
            a3 += tl.dot(
                z, _decode_matrix(s_ptr, krows, P0 + P1 + P2 + tl.arange(0, P3), krows, N, False)
            )
    return a0, a1, a2, a3


@triton.jit
def _decode_rows(
    cold,
    w_ptr,
    s_ptr,
    alpha,
    alpha_lo,
    weight,
    weight_lo,
    pos0,
    pos1,
    pos2,
    pos3,
    rows,
    row_mask,
    MODE: tl.constexpr,
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
    OFF_B0: tl.constexpr,
    W_B0: tl.constexpr,
    OFF_B1: tl.constexpr,
    W_B1: tl.constexpr,
    OFF_QJL: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
    BLOCKED: tl.constexpr,
    FIRST: tl.constexpr,
    LAST: tl.constexpr,
):
    """Output pieces FIRST..LAST-1 of the decoded rows, in fp32."""
    a0 = tl.zeros([BLOCK_M, P0], tl.float32)
    a1 = tl.zeros([BLOCK_M, P1 if P1 > 0 else 16], tl.float32)
    a2 = tl.zeros([BLOCK_M, P2 if P2 > 0 else 16], tl.float32)
    a3 = tl.zeros([BLOCK_M, P3 if P3 > 0 else 16], tl.float32)
    if MODE == 2:
        G: tl.constexpr = N // 2
        a0, a1, a2, a3 = _accumulate_group(
            cold,
            w_ptr,
            alpha,
            a0,
            a1,
            a2,
            a3,
            pos0,
            pos1,
            pos2,
            pos3,
            rows,
            row_mask,
            0,
            G,
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
            True,
            ROWS,
            BLOCK_M,
            C,
            FIRST,
            LAST,
        )
        a0, a1, a2, a3 = _accumulate_group(
            cold,
            w_ptr,
            alpha_lo,
            a0,
            a1,
            a2,
            a3,
            pos0,
            pos1,
            pos2,
            pos3,
            rows,
            row_mask,
            G,
            G,
            BITS1,
            lut1,
            OFF_B0,
            W_B0,
            OFF_B1,
            W_B1,
            N,
            P0,
            P1,
            P2,
            P3,
            True,
            ROWS,
            BLOCK_M,
            C,
            FIRST,
            LAST,
        )
    elif BLOCKED == 2:
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
            pos0,
            pos1,
            pos2,
            pos3,
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
            False,
            ROWS,
            BLOCK_M,
            C,
            FIRST,
            LAST,
        )
    elif BLOCKED == 1 and FIRST == 0:
        a0, a1, a2, a3 = _accumulate_group(
            cold,
            w_ptr,
            alpha,
            a0,
            a1,
            a2,
            a3,
            pos0,
            pos1,
            pos2,
            pos3,
            rows,
            row_mask,
            0,
            P0,
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
            False,
            ROWS,
            BLOCK_M,
            C,
            0,
            1,
        )
        if MODE == 1:
            a0, a1, a2, a3 = _accumulate_signs(
                cold,
                s_ptr,
                weight,
                a0,
                a1,
                a2,
                a3,
                rows,
                row_mask,
                OFF_QJL,
                N,
                P0,
                P1,
                P2,
                P3,
                ROWS,
                BLOCK_M,
                C,
                0,
                P0 // C,
                0,
                1,
            )
    elif BLOCKED == 1:
        a0, a1, a2, a3 = _accumulate_group(
            cold,
            w_ptr,
            alpha_lo,
            a0,
            a1,
            a2,
            a3,
            pos0,
            pos1,
            pos2,
            pos3,
            rows,
            row_mask,
            P0,
            N - P0,
            BITS1,
            lut1,
            OFF_B0,
            W_B0,
            OFF_B1,
            W_B1,
            N,
            P0,
            P1,
            P2,
            P3,
            False,
            ROWS,
            BLOCK_M,
            C,
            1,
            4,
        )
        if MODE == 1:
            a0, a1, a2, a3 = _accumulate_signs(
                cold,
                s_ptr,
                weight_lo,
                a0,
                a1,
                a2,
                a3,
                rows,
                row_mask,
                OFF_QJL,
                N,
                P0,
                P1,
                P2,
                P3,
                ROWS,
                BLOCK_M,
                C,
                P0 // C,
                N // C,
                1,
                4,
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
            pos0,
            pos1,
            pos2,
            pos3,
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
            False,
            ROWS,
            BLOCK_M,
            C,
            FIRST,
            LAST,
        )
        if MODE == 1:
            a0, a1, a2, a3 = _accumulate_signs(
                cold,
                s_ptr,
                weight,
                a0,
                a1,
                a2,
                a3,
                rows,
                row_mask,
                OFF_QJL,
                N,
                P0,
                P1,
                P2,
                P3,
                ROWS,
                BLOCK_M,
                C,
                0,
                N // C,
                FIRST,
                LAST,
            )
    return a0, a1, a2, a3


@triton.jit
def _decode_columns(
    cold,
    w_ptr,
    alpha,
    rows,
    row_mask,
    col0,
    COORD0: tl.constexpr,
    G: tl.constexpr,
    lut,
    OFF0: tl.constexpr,
    W0: tl.constexpr,
    OFF1: tl.constexpr,
    W1: tl.constexpr,
    N: tl.constexpr,
    COLS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C: tl.constexpr,
):
    """Output columns ``col0..col0+COLS-1`` of one mse instance (coordinates COORD0..+G), fp32."""
    acc = tl.zeros([BLOCK_M, COLS], tl.float32)
    cols = col0 + tl.arange(0, COLS)
    for j in range(G // C):
        code = _load_codes(cold[:, None], j, rows, row_mask, OFF0, W0, OFF1, W1, ROWS, BLOCK_M, C)
        z = (alpha[:, None] * _dequantize(code, lut)).to(tl.bfloat16)
        krows = COORD0 + j * C + tl.arange(0, C)
        acc += tl.dot(z, tl.load(w_ptr + krows[:, None] * N + cols[None, :]))
    return acc


@triton.jit
def _store_piece(row_ptr, row_mask, acc, START, P: tl.constexpr, ELEM: tl.constexpr):
    # A reconstruction can exceed the largest finite value next to it: saturate, never store inf.
    if ELEM == tl.float8e4nv:
        acc = tl.minimum(tl.maximum(acc, -448.0), 448.0)
    elif ELEM == tl.float16:
        acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
    columns = START + tl.arange(0, P)
    tl.store(row_ptr[:, None] + columns[None, :], acc.to(ELEM), mask=row_mask[:, None])


@triton.jit(do_not_specialize=["page_ptr", "cold_page_bytes", "num_pages"])
def _tq_decode_kernel(
    page_ptr,
    cold_ptr,
    cold_page_bytes,
    num_pages,
    hot_ptr,
    table_ptr,
    w_ptr,
    s_ptr,
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
    OFF_QJL: tl.constexpr,
    OFF_SCALE: tl.constexpr,
    OFF_MASK: tl.constexpr,
    OFF_RAW: tl.constexpr,
    RAW: tl.constexpr,
    RAW_BLOCK: tl.constexpr,
    COL_BLOCKS: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # COL_BLOCKS > 1 (mse passes of one instance): the instance's output columns are accumulated
    # in COL_BLOCKS blocks, so that each accumulator fits tensor memory. SPLIT neighbouring
    # programs share the rows and take COL_BLOCKS / SPLIT blocks each (one program per row tile
    # reads the codes once; more fit an SM).
    if SPLIT > 1:
        tile_page = tl.program_id(0) // SPLIT
        col_block = tl.program_id(0) % SPLIT
    else:
        tile_page = tl.program_id(0)
        col_block = 0
    buffer = tl.program_id(1)
    tile = tl.program_id(2)
    lane = tl.arange(0, BLOCK_M)
    if PAGES_PER_TILE > 1:
        # Buffers with few rows per Page: one tile takes the rows of several Pages.
        page = tile_page * PAGES_PER_TILE + lane // ROWS_PAD
        rows = lane % ROWS_PAD
        row_mask = (rows < ROWS) & (page < num_pages)
    else:
        # Fractional bits: a tile is one KV head of the Page, whose tokens share the channel split.
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
    alpha = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    pos0 = rows
    pos1 = rows
    pos2 = rows
    pos3 = rows
    alpha_lo = alpha
    weight = alpha
    weight_lo = alpha
    if MODE == 2:
        G: tl.constexpr = N // 2
        mask_ptr = cold_ptr + tl.max(cold_rows, axis=0) + OFF_MASK + tile * (N // 8)
        pos0, before = _outlier_positions(_load_mask(mask_ptr, 0, P0), 0, 0, P0, G)
        if P1 > 0:
            pos1, before = _outlier_positions(_load_mask(mask_ptr, P0, P1), before, P0, P1, G)
        if P2 > 0:
            pos2, before = _outlier_positions(
                _load_mask(mask_ptr, P0 + P1, P2), before, P0 + P1, P2, G
            )
        if P3 > 0:
            pos3, before = _outlier_positions(
                _load_mask(mask_ptr, P0 + P1 + P2, P3), before, P0 + P1 + P2, P3, G
            )
        alpha_lo = tl.load(scale_ptr + ROWS + rows, mask=row_mask, other=0.0).to(tl.float32)
    SQRT_HALF_PI: tl.constexpr = 1.2533141373155003
    if BLOCKED == 2:
        alpha = tl.load(scale_ptr + PASS * ROWS + rows, mask=row_mask, other=0.0).to(tl.float32)
    elif BLOCKED == 1:
        alpha_lo = tl.load(scale_ptr + ROWS + rows, mask=row_mask, other=0.0).to(tl.float32)
        if MODE == 1:
            rho = tl.load(scale_ptr + 2 * ROWS + rows, mask=row_mask, other=0.0).to(tl.float32)
            rho_lo = tl.load(scale_ptr + 3 * ROWS + rows, mask=row_mask, other=0.0).to(tl.float32)
            weight = rho * (SQRT_HALF_PI / P0)
            weight_lo = rho_lo * (SQRT_HALF_PI / (N - P0))
    elif MODE == 1:
        rho = tl.load(scale_ptr + ROWS + rows, mask=row_mask, other=0.0).to(tl.float32)
        weight = rho * (SQRT_HALF_PI / N)

    # Rows with several instances decode one instance per launch (PASS): each reads only its
    # own codes and accumulates only its own output columns.
    if BLOCKED == 2:
        FIRST: tl.constexpr = PASS
        LAST: tl.constexpr = PASS + 1
    elif BLOCKED == 1:
        FIRST: tl.constexpr = 0 if PASS == 0 else 1
        LAST: tl.constexpr = 1 if PASS == 0 else 4
    else:
        FIRST: tl.constexpr = 0
        LAST: tl.constexpr = 4
    if COL_BLOCKS > 1:
        # One instance per pass (blocked mse, or a single-piece row): its codes against its
        # output columns, one block at a time.
        START: tl.constexpr = (
            (P0 if FIRST >= 1 else 0) + (P1 if FIRST >= 2 else 0) + (P2 if FIRST >= 3 else 0)
        )
        WIDTH: tl.constexpr = P0 if FIRST == 0 else P1 if FIRST == 1 else P2 if FIRST == 2 else P3
        COLS: tl.constexpr = WIDTH // COL_BLOCKS
        BLOCKS: tl.constexpr = COL_BLOCKS // SPLIT
        for block in tl.static_range(BLOCKS):
            col0 = START + (col_block * BLOCKS + block) * COLS
            acc = _decode_columns(
                cold,
                w_ptr,
                alpha,
                rows,
                row_mask,
                col0,
                START,
                WIDTH,
                lut_ptr if WIDTH == P0 else lut_ptr + LUT_STRIDE,
                OFF_A0 + START // C * ROWS * (C * W_A0 // 8),
                W_A0,
                OFF_A1 + START // C * ROWS * (C * W_A1 // 8),
                W_A1,
                N,
                COLS,
                ROWS,
                BLOCK_M,
                C,
            )
            _store_piece(row_ptr, row_mask, acc, col0, COLS, ELEM)
    else:
        a0, a1, a2, a3 = _decode_rows(
            cold,
            w_ptr,
            s_ptr,
            alpha,
            alpha_lo,
            weight,
            weight_lo,
            pos0,
            pos1,
            pos2,
            pos3,
            rows,
            row_mask,
            MODE,
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
            OFF_B0,
            W_B0,
            OFF_B1,
            W_B1,
            OFF_QJL,
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
        if SPLIT > 1:  # the first column block of the pass copies them
            raw_mask = raw_mask & (col_block == 0)
        raw_ptr = (cold + OFF_RAW).to(tl.pointer_type(ELEM))
        values = tl.load(raw_ptr[:, None] + rows[:, None] * RAW + raw_cols[None, :], mask=raw_mask)
        tl.store(
            hot[:, None] + rows.to(tl.int64)[:, None] * STRIDE + raw_dst[None, :],
            values,
            mask=raw_mask,
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
