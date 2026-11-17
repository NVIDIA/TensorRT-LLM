# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 routed-expert FC1 weight: the k-block layout both FC1 grouped GEMMs read.

One expert's FC1 weight is `[N = 4096 rows, K/2 = 3072 B]` of packed NVFP4 (two values a
byte, up and gate alternating every 64 rows). Both FC1 kernels -- the decode GG1
(`moe_gather_grouped_gemm_act_fusion`, N128 tiles) and the prefill pair's GG1
(`moe_prefill_nvfp4_gg1`, N128 / N256 tiles) -- load B per K tile as one TMA box of
`tile_n` rows x 128 B (256 NVFP4 values). In the row-major layout that box is 128 B
pieces of `tile_n` rows at a 3072 B pitch, and the weight stream is DRAM-pattern bound:
on a B300 a read-only kernel with that access pattern reached 6.19 TB/s, against 6.90
TB/s for the k-block pattern and 6.89 TB/s for one contiguous range.

The k-block layout stores the same bytes as

    [N/128 row blocks][K/256 k-blocks][128 rows][128 B]

so every 128-row x 128 B box is one contiguous 16 KB read (an N256 box: two of them,
32 KB). Values, scales and math are unchanged -- the shared-memory image a box lands in
is identical -- so the kernels' outputs are bitwise those of the row-major layout. The
decode GG1 on real routing (cold L2, mean of 20 layer/rank cases) went 78.1 -> 74.9 us
(6.27 -> 6.54 TB/s) at 32 rows (32 requests, MTP off) and 104.1 -> 99.5 us (6.34 ->
6.63 TB/s) at 64 rows; -4.0 .. -4.6 % from 32 to 192 rows.

What does not change: the tensor's shape and dtype (`[E, N, K/2]` uint8, so the weight
table and the loader's shape checks are layout-blind); each expert stays one contiguous
`N * K/2` byte range, and each 128-row block (hence each N-tile of either GG1) stays one
contiguous range, so per-expert copies (replica slots) and the decode L2 prefetch, which
addresses (expert, 256-row N-tile) byte ranges, read the same bytes as before; the FC1
block scales keep their 512-byte atom layout.

`TRTLLM_GLM52_MOE_FC1_KBLK` (default 1) selects it. The model reads it once at build, and
the loader, the MTP layer's quantized experts and the compiled FC1 kernels all follow that
one value; 0 restores the row-major layout and today's kernels.

FC2 (`TRTLLM_GLM52_MOE_FC2_KBLK`, default 1, `fc2_enabled`): the same relayout of each
expert's FC2 weight `[N = 6144 rows, K/2 = 1024 B]`, read by the three FC2 grouped GEMMs
(the decode dynamic FC2, its static fallback and the prefill pair's GG2) through the same
5-D descriptor. Their 128-row x 128 B boxes were 128 B pieces at a 1 KB pitch; k-block
makes each one contiguous 16 KB read, again with the same shared-memory image and a
bitwise-identical output. On a B300 (cold L2, real routing): prefill-pair GG2 -2.8 / -2.9 %
at 128 / 192 rows (MTP3 / MTP5 decode) and -2.1 % at a 7513-row context step; the decode
dynamic FC2 -1.5 / -1.2 % at 32 / 64 rows. The switch is separate from FC1's and read at
the same place; 0 restores the row-major FC2 layout and kernels. `to_kblock` /
`from_kblock` are shape-generic (`[..., N, K/2]` bytes) and serve both weights.
"""

from __future__ import annotations

import os

import torch

ENV = "TRTLLM_GLM52_MOE_FC1_KBLK"
FC2_ENV = "TRTLLM_GLM52_MOE_FC2_KBLK"
#: Rows per block: the decode GG1's MMA N tile (the prefill pair's N256 reads two blocks).
BLOCK_ROWS = 128
#: Bytes of K per block: one 256-value NVFP4 K tile, the 128 B swizzle span.
BLOCK_BYTES = 128


def _switch(env: str) -> bool:
    v = os.environ.get(env, "1")
    if v not in ("0", "1"):
        raise ValueError(f"{env} must be 0 or 1, got {v!r}")
    return v == "1"


def enabled() -> bool:
    """`TRTLLM_GLM52_MOE_FC1_KBLK`, validated: "1" (default, k-block) or "0" (row-major)."""
    return _switch(ENV)


def fc2_enabled() -> bool:
    """`TRTLLM_GLM52_MOE_FC2_KBLK`, validated: "1" (default, k-block) or "0" (row-major)."""
    return _switch(FC2_ENV)


def _check(w: torch.Tensor) -> tuple[int, int]:
    n, kk = int(w.shape[-2]), int(w.shape[-1])
    if w.element_size() != 1 or n % BLOCK_ROWS or kk % BLOCK_BYTES:
        raise ValueError(
            f"FC1 / FC2 weight must be bytes [..., N, K/2] with N % {BLOCK_ROWS} == 0 and "
            f"K/2 % {BLOCK_BYTES} == 0, got {w.dtype} {tuple(w.shape)}"
        )
    return n, kk


def to_kblock(w: torch.Tensor) -> torch.Tensor:
    """Row-major `[..., N, K/2]` bytes -> the k-block layout, same shape (a new tensor)."""
    n, kk = _check(w)
    lead = tuple(w.shape[:-2])
    blocks = w.reshape(*lead, n // BLOCK_ROWS, BLOCK_ROWS, kk // BLOCK_BYTES, BLOCK_BYTES)
    return blocks.transpose(-3, -2).contiguous().reshape(w.shape)


def from_kblock(w: torch.Tensor) -> torch.Tensor:
    """The inverse of `to_kblock`: k-block `[..., N, K/2]` -> row-major (a new tensor)."""
    n, kk = _check(w)
    lead = tuple(w.shape[:-2])
    blocks = w.reshape(*lead, n // BLOCK_ROWS, kk // BLOCK_BYTES, BLOCK_ROWS, BLOCK_BYTES)
    return blocks.transpose(-3, -2).contiguous().reshape(w.shape)
