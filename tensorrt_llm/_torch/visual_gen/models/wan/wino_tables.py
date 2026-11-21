# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd coefficient matrices shared by the portable and the sm100 paths.

``SPATIAL_WINO[tile]`` holds ``B^T`` (input transform), ``G`` (filter transform) and ``A^T``
(output transform) of ``F(tile, 3)`` as plain Python lists; the callers build their fp64 or fp32
tensors from them. Tile 2 is Lavin & Gray's F(2,3) on the points (0, +-1, inf). Tile 4 is F(4,3)
on the points (0, +-3/2, +-5/8, inf), chosen for low fp16 rounding sensitivity; its ``B^T`` and
``A^T`` entries are dyadic, so their products with bf16 / fp16 inputs are exact in fp32.
:func:`bt_row` and :func:`at_row` apply single rows of ``B^T`` / ``A^T`` in factored form; the
transform kernels (portable and sm100) are written with them.
"""

SPATIAL_WINO = {
    2: dict(
        BT=[[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]],
        G=[[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]],
        AT=[[1, 1, 1, 0], [0, 1, -1, -1]],
    ),
    4: dict(
        BT=[
            [225 / 256, 0, -169 / 64, 0, 1, 0],
            [0, -75 / 128, -25 / 64, 3 / 2, 1, 0],
            [0, 75 / 128, -25 / 64, -3 / 2, 1, 0],
            [0, -45 / 32, -9 / 4, 5 / 8, 1, 0],
            [0, 45 / 32, -9 / 4, -5 / 8, 1, 0],
            [0, 225 / 256, 0, -169 / 64, 0, 1],
        ],
        G=[
            [256 / 225, 0, 0],
            [128 / 1071, 64 / 357, 32 / 119],
            [128 / 1071, -64 / 357, 32 / 119],
            [-2048 / 2975, -256 / 595, -32 / 119],
            [-2048 / 2975, 256 / 595, -32 / 119],
            [0, 0, 1],
        ],
        AT=[
            [1, 1, 1, 1, 1, 0],
            [0, 3 / 2, -3 / 2, 5 / 8, -5 / 8, 0],
            [0, 9 / 4, 9 / 4, 25 / 64, 25 / 64, 0],
            [0, 27 / 8, -27 / 8, 125 / 512, -125 / 512, 1],
        ],
    ),
}


# F(4x4, 3x3) interpolation points, besides 0 and infinity: +-P and +-Q. Chosen for low fp16
# rounding error; both are dyadic (3/2, 5/8), so every product of them used below is exact in
# fp32 and folds into a single constant at trace time.
#
# Where the coefficients come from (Toom-Cook on the points 0, -P, +P, -Q, +Q, inf):
#
#   Each row of B^T holds the coefficients (of x^0 .. x^5, applied to d0 .. d5) of the
#   polynomial that vanishes on all finite points but one:
#     row 0 drops 0:   (x^2 - P^2)(x^2 - Q^2) = P^2 Q^2 - (P^2 + Q^2) x^2 + x^4
#                      -> [P^2 Q^2, 0, -(P^2 + Q^2), 0, 1, 0]
#     rows 1, 2 drop one of +-P:  x (x^2 - Q^2)(x +- P) -> [0, -+P Q^2, -Q^2, +-P, 1, 0]
#     rows 3, 4 drop one of +-Q:  the same with P and Q swapped
#     row 5 (infinity): row 0 times x, i.e. shifted by one.
#   Rows 1 and 2 share an even part (d4 - Q^2 d2) and an odd part (P d3 - P Q^2 d1) and take
#   their sum and difference; rows 3 and 4 likewise.
#
#   A^T row r evaluates the m outputs at the points: [0^r, P^r, (-P)^r, Q^r, (-Q)^r, inf],
#   where 0^0 = 1 and the infinity column is 1 only in the last row r = 3:
#     row 0: t0 + (t1 + t2) + (t3 + t4)
#     row r: P^r (t1 +- t2) + Q^r (t3 +- t4)  (+ for even r, - for odd r), plus t5 for r = 3.
#
# The filter transform G (scaled so that A^T [(G g) * (B^T d)] is the convolution) and these
# same B^T / A^T are tabulated in SPATIAL_WINO above; the kernel tests check the two agree.
P = 3 / 2
Q = 5 / 8


def bt_row(m: int, x: int, d: list):
    """Row ``x`` of the F(m, 3) B^T (m = 2 or 4) applied to the ``m + 2`` values ``d``: scalars
    or vectors of any type with ``*`` (by a Python float), ``+`` and ``-``."""
    if m == 2:
        return (d[0] - d[2], d[1] + d[2], d[2] - d[1], d[1] - d[3])[x]
    if x == 0:
        return (P * Q) ** 2 * d[0] - (P * P + Q * Q) * d[2] + d[4]
    if x == 5:
        return (P * Q) ** 2 * d[1] - (P * P + Q * Q) * d[3] + d[5]
    a, b = (P, Q) if x in (1, 2) else (Q, P)
    even = d[4] - b * b * d[2]
    odd = a * d[3] - a * b * b * d[1]
    return even + odd if x in (1, 3) else even - odd


def at_row(m: int, r: int, t: list):
    """Row ``r`` of the F(m, 3) A^T (m = 2 or 4) applied to the ``m + 2`` values ``t``, as in
    :func:`bt_row`."""
    if m == 2:
        return t[0] + t[1] + t[2] if r == 0 else t[1] - t[2] - t[3]
    if r == 0:
        # Pairwise: the pair sums are row 2's too, so row 0 costs two adds instead of four.
        return t[0] + (t[1] + t[2]) + (t[3] + t[4])
    if r % 2 == 0:
        y = P**r * (t[1] + t[2]) + Q**r * (t[3] + t[4])
    else:
        y = P**r * (t[1] - t[2]) + Q**r * (t[3] - t[4])
    return y + t[5] if r == 3 else y
