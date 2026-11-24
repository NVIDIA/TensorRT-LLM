# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep16 MXFP8 epilogue (``rubin_mxfp8_epilogue``) on given BF16 rows, and the rows that
sweep it: the GPU bit test's kernel and inputs (imported by the sm_107 tests only).

``mx_rows`` runs the SiTU entry's epilogue lines (``mx_amax``, ``mx_scale``, ``mx_codes``,
``r128c4_offset``) on BF16 rows instead of the SiTU values. ``sweep_rows`` holds every finite BF16
1x32 group maximum with either sign (the other 31 values below it by bits, random signs; half the
groups mix in BF16 subnormals) and the named edge groups: zeros and signed zeros, subnormals, the
``amax * rcp(448)`` subnormal band (2^-127, 2^-126], subnormal values under small scale bytes,
``448 * 2^m`` maxima and one BF16 step either side, infinities and NaNs (a group of NaN, NaN in
half the group, NaN beside an infinity) and the quantizer's NaN rule by parity (every odd element
a NaN: a NaN maximum; every even one: the odd maximum), beside zeros, large values and infinities.
"""

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm import (
    rubin_mxfp8_epilogue,
)

mx_amax = rubin_mxfp8_epilogue.mx_amax
mx_codes = rubin_mxfp8_epilogue.mx_codes
mx_scale = rubin_mxfp8_epilogue.mx_scale
r128c4_offset = rubin_mxfp8_epilogue.r128c4_offset

#: Columns per program and per sweep row.
BLOCK = 1024


@triton.jit
def _mx_rows_kernel(q_ptr, s_ptr, x_ptr, k, kb_tiles, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)
    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    yb = tl.load(x_ptr + i * k + offsets)
    groups: tl.constexpr = BLOCK_SIZE // 32
    amax = mx_amax(yb, groups)
    byte, mult = mx_scale(amax)
    mult = tl.reshape(tl.broadcast_to(mult[:, None], (groups, 32)), (BLOCK_SIZE,))
    q = mx_codes(yb, mult)
    tl.store(q_ptr + i * k + offsets, q)
    kb = j * groups + tl.arange(0, groups)
    tl.store(s_ptr + r128c4_offset(i, kb, kb_tiles), byte.to(tl.uint8))


def mx_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The epilogue's E4M3 codes and flat R128c4 scales of BF16 ``x`` [rows, k], k % 1024 == 0;
    the padding rows' scale bytes zero, as ``mxfp8_quantize`` writes them."""
    rows, k = x.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and k % BLOCK == 0
    codes = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=x.device)
    nbytes = triton.cdiv(rows, 128) * 128 * (k // 32)
    scales = torch.zeros(nbytes, dtype=torch.uint8, device=x.device)
    _mx_rows_kernel[(rows, k // BLOCK)](codes, scales, x, k, k // 32 // 4, BLOCK_SIZE=BLOCK)
    return codes, scales


def _named_groups(gen: torch.Generator) -> list[list[int]]:
    def below(b: int, n: int) -> list[int]:
        return torch.randint(0, max(1, b & 0x7FFF), (n,), generator=gen).tolist()

    groups = [[0] * 32, [0x8000] * 32, [0x8000 if i % 3 else 0 for i in range(32)]]
    groups += [[k] + [0] * 31 for k in range(1, 0x80, 7)]
    groups += [[0x8000 | k] + [0x8000] * 31 for k in range(1, 0x80, 7)]
    groups += [[0x7F] * 32, torch.randint(1, 0x80, (32,), generator=gen).tolist()]
    groups += [[0x7F7F] + [0x3F80] * 31, [0xFF7F] + [0xBF80] * 31]
    for m in range(-133, 121):
        v = torch.tensor(448.0 * 2.0**m, dtype=torch.float32).to(torch.bfloat16)
        b = int(v.view(torch.int16)) & 0xFFFF
        for d in (-1, 0, 1):
            if 0 < ((b + d) & 0x7FFF) < 0x7F80:
                groups.append([b + d] + below(b + d, 31))
    for b in range(0x0450, 0x0510):  # amax * rcp(448) around (2^-127, 2^-126]
        groups.append([b] + [0] * 31)
        groups.append([b] + torch.randint(0, 0x80, (31,), generator=gen).tolist())
    for b in range(0x0480, 0x0A00, 0x10):  # subnormal values under small scale bytes
        rest = torch.randint(0, 0x80, (27,), generator=gen).tolist()
        groups.append([b, 0x7F, 0x40, 0x01, 0x807F] + rest)
    nan, inf = 0x7FC0, 0x7F80
    groups += [
        [inf] + [0x3F80] * 31,
        [0x8000 | inf] + [0x4000] * 31,
        [inf, 0x7F7F, 0x7E00, 0x7A00] + [0x3F80] * 28,
        [inf] * 32,
        [nan] + [0x3F80] * 31,
        [0x8000 | nan] + [0x3F80] * 31,
        [nan] * 32,
        [0xFFC1] * 32,
        [nan if i % 2 == 0 else 0x3F80 for i in range(32)],
        [nan if i % 2 else 0x3F80 for i in range(32)],
        [nan, inf] + [0x3F80] * 30,
    ]
    # The NaN rule by parity: vecMax = even > odd ? even : odd over the two NaN-skipping maxima.
    for other in (0, 0x8000, 0x0001, 0x3F80, 0xBF80, 0x7A00, 0x7F7F, inf, 0x8000 | inf):
        groups.append([nan if i % 2 else other for i in range(32)])  # every odd a NaN
        groups.append([other if i % 2 else nan for i in range(32)])  # every even a NaN
    groups += [
        [0x3F80 if i == 31 else nan for i in range(32)],  # one odd number among NaN
        [0x3F80 if i == 0 else nan for i in range(32)],  # one even number among NaN
        [0x7F7F if i == 6 else (nan if i % 2 else 0x3F80) for i in range(32)],
        [inf if i == 1 else (nan if i % 2 else 0x4000) for i in range(32)],  # odd: inf and NaN
        [0xFFC1 if i % 2 else 0x0450 for i in range(32)],  # band evens under odd NaN
    ]
    return groups


def sweep_rows(seed: int = 20261112) -> torch.Tensor:
    """BF16 [rows, 1024] (CPU): the sweep's 1x32 groups, 32 per row, zero groups at the end."""
    gen = torch.Generator().manual_seed(seed)
    mags = torch.arange(0, 0x7F80, dtype=torch.int64)
    blocks = []
    for sign in (0, 0x8000):
        for sub_mix in (False, True):
            n = mags.numel()
            unit = torch.rand((n, 31), generator=gen, dtype=torch.float64)
            others = (unit * (mags[:, None] + 1)).long()
            if sub_mix:
                subs = torch.minimum(torch.randint(0, 0x80, (n, 31), generator=gen), mags[:, None])
                others = torch.where(torch.rand((n, 31), generator=gen) < 0.5, subs, others)
            signs = torch.randint(0, 2, (n, 31), generator=gen) * 0x8000
            blk = torch.cat([(mags | sign)[:, None], others | signs], dim=1)
            perm = torch.argsort(torch.rand((n, 32), generator=gen), dim=1)
            blocks.append(torch.gather(blk, 1, perm))
    blocks.append(torch.tensor(_named_groups(gen), dtype=torch.int64))
    flat = torch.cat(blocks)
    pad = (-flat.shape[0]) % (BLOCK // 32)
    flat = torch.cat([flat, torch.zeros((pad, 32), dtype=torch.int64)])
    u16 = flat.to(torch.int32).to(torch.int16)  # two's complement keeps the 16 bits
    return u16.view(torch.bfloat16).reshape(-1, BLOCK)
