# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""1x128 FP8 activation quantization for the Rubin MXFP8 GEMM, with a movable PDL trigger.

Bitwise twin of ``torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0`` (the C++ kernel in
``fp8_blockscale_quant_packed.cu``): per row and 128-column block, ``amax`` of the BF16 values
(at least 1e-10), the UE8M0 scale ``e`` = ``amax * rcp.approx.ftz(448)`` rounded up to a power
of two, E4M3 codes of ``x * 2^(127 - e)`` (round to nearest even, saturating), and the scale
byte replicated four times (one per 32-column group) in the R128c4 layout of the CuTe GEMM,
zero for the padding rows up to a multiple of 128.

The C++ kernel waits for its predecessor at entry and lets its dependent launch only after its
last store. That keeps a PDL GEMM behind it from starting before the quantization ends. Here
``trigger`` places ``griddepcontrol.launch_dependents``: 0 after the stores (as the C++ kernel),
1 right after ``griddepcontrol.wait``, 2 at entry, before the wait. The dependent waits for
this grid before it reads the codes either way; modes 1 and 2 only let it launch (and stage its
weights) earlier. Mode 2 suits a dependent that reads nothing but weights before its wait (the
K3 FP8 weight-read GEMM).

An early trigger only helps if the dependent's CTAs can become resident while this grid runs:
a GEMM CTA needs nearly all of an SM's shared memory. The C++ kernel (448 CTAs at K 7168 for
any M <= 128) and a one-row-per-program grid occupy every SM, so each program here quantizes
16 rows x 1024 columns (7 CTAs at M <= 16, K 7168) and the last row tile also zeroes the scale
words of the padding rows.
"""

import os

import torch
import triton
import triton.language as tl

#: Columns per scale (the GEMM's K block); rows and 128-column blocks per program.
_BLOCK_K = 128
_ROWS_PER_PROGRAM = 16
_BLOCKS_PER_PROGRAM = 8
#: Rows of a scale tile (R128c4 layout).
_SF_TILE_ROWS = 128


@triton.jit
def _gdc_wait():
    # The memory clobber keeps the compiler from hoisting the loads above the wait.
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _gdc_launch_dependents():
    tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _r128c4_word_index(rows, kb, K: tl.constexpr):
    # 512 bytes per (128-row tile, 128-column block); row r at byte (r % 32) * 16 +
    # ((r % 128) // 32) * 4 of its tile; one 32-bit word per (row, block).
    byte = (rows // 128) * (K // 128) * 512 + kb * 512 + (rows % 32) * 16 + ((rows % 128) // 32) * 4
    return byte // 4


# m and padded_m are not specialized: no recompile when serving meets a new M class.
@triton.jit(do_not_specialize=["m", "padded_m"])
def _fp8_quant_1x128_r128c4_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    m,
    padded_m,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCKS: tl.constexpr,
    PAD_ROWS: tl.constexpr,
    USE_PDL: tl.constexpr,
    TRIGGER: tl.constexpr,
):
    if USE_PDL and TRIGGER == 2:
        _gdc_launch_dependents()
    if USE_PDL:
        _gdc_wait()
        if TRIGGER == 1:
            _gdc_launch_dependents()
    pid_r = tl.program_id(0)
    rows = pid_r * ROWS + tl.arange(0, ROWS)  # [ROWS]
    kb = tl.program_id(1) * BLOCKS + tl.arange(0, BLOCKS)  # [BLOCKS] 128-column blocks
    kb_ok = kb < K // 128
    row_ok = rows < m
    cols = kb[:, None] * 128 + tl.arange(0, 128)[None, :]  # [BLOCKS, 128]
    offs = rows.to(tl.int64)[:, None, None] * K + cols[None, :, :]  # [ROWS, BLOCKS, 128]
    mask = row_ok[:, None, None] & kb_ok[None, :, None]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    amax = tl.max(tl.abs(x.to(tl.float32)), axis=2)  # [ROWS, BLOCKS], exact: BF16 -> FP32
    amax = tl.maximum(amax, 1e-10)
    raw = amax * _rcp_approx_ftz(tl.full([ROWS, BLOCKS], 448.0, tl.float32))
    bits = raw.to(tl.uint32, bitcast=True)
    # Round up to a power of two (UE8M0, saturating at 2^127); raw is a positive normal.
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
    e = tl.minimum(e, 254)
    qscale = ((254 - e) << 23).to(tl.float32, bitcast=True)  # 2^(127 - e)
    q = (x.to(tl.float32) * qscale[:, :, None]).to(tl.float8e4nv)
    tl.store(q_ptr + offs, q, mask=mask)
    # The scale byte four times (one per 32-column group); rows >= m get zero words.
    word = tl.where(row_ok[:, None], (e * 0x01010101).to(tl.int32, bitcast=True), 0)
    tl.store(s_ptr + _r128c4_word_index(rows[:, None], kb[None, :], K), word, mask=kb_ok[None, :])
    if pid_r == tl.num_programs(0) - 1:
        # Padding rows past the last row tile, up to the next multiple of 128 (< 128 of them).
        pad = (pid_r + 1) * ROWS + tl.arange(0, PAD_ROWS)
        pad_ok = (pad < padded_m)[:, None] & kb_ok[None, :]
        zeros = tl.zeros([PAD_ROWS, BLOCKS], dtype=tl.int32)
        tl.store(s_ptr + _r128c4_word_index(pad[:, None], kb[None, :], K), zeros, mask=pad_ok)
    if USE_PDL and TRIGGER == 0:
        _gdc_launch_dependents()


def fp8_quantize_1x128_r128c4(
    x: torch.Tensor, trigger: int = 0, use_pdl: bool | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(codes [M, K] float8_e4m3fn, scales uint8 [pad128(M) * K / 32])`` of BF16 ``x [M, K]``.

    The same bytes as ``torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0``. ``trigger``: see
    the module docstring. ``use_pdl`` defaults to ``TRTLLM_ENABLE_PDL`` (on unless "0").
    """
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
        raise ValueError("expected a contiguous bfloat16 [M, K] tensor")
    m, k = x.shape
    if k % _BLOCK_K != 0:
        raise ValueError(f"K must be a multiple of {_BLOCK_K}, got {k}")
    if trigger not in (0, 1, 2):
        raise ValueError(f"trigger must be 0, 1 or 2, got {trigger}")
    if use_pdl is None:
        use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    padded_m = (m + _SF_TILE_ROWS - 1) // _SF_TILE_ROWS * _SF_TILE_ROWS
    codes = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((padded_m * (k // 32),), dtype=torch.uint8, device=x.device)
    if padded_m == 0:
        return codes, scales
    grid = (
        triton.cdiv(m, _ROWS_PER_PROGRAM),
        triton.cdiv(k // _BLOCK_K, _BLOCKS_PER_PROGRAM),
    )
    _fp8_quant_1x128_r128c4_kernel[grid](
        x,
        codes,
        scales.view(torch.int32),
        m,
        padded_m,
        K=k,
        ROWS=_ROWS_PER_PROGRAM,
        BLOCKS=_BLOCKS_PER_PROGRAM,
        PAD_ROWS=_SF_TILE_ROWS,
        USE_PDL=use_pdl,
        TRIGGER=trigger,
        num_warps=8,
        launch_pdl=use_pdl,
    )
    return codes, scales
