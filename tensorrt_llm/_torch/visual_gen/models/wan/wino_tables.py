# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial Winograd coefficient matrices shared by the portable and the sm100 paths.

``SPATIAL_WINO[tile]`` holds ``B^T`` (input transform), ``G`` (filter transform) and ``A^T``
(output transform) of ``F(tile, 3)`` as plain Python lists; the callers build their fp64 or fp32
tensors from them. Tile 2 is Lavin & Gray's F(2,3) on the points (0, +-1, inf). Tile 4 is F(4,3)
on the points (0, +-3/2, +-5/8, inf), chosen for low fp16 rounding sensitivity; its ``B^T`` and
``A^T`` entries are dyadic, so their products with bf16 / fp16 inputs are exact in fp32. The
portable CUDA transform kernels (``vaeWinoSpatialKernel.cu``) carry the same ``B^T`` / ``A^T``
and are checked against this module's ``G`` by the cross-backend kernel tests.
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
