# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU-gated multiply with an MXFP8 epilogue (Triton).

The family's ``situ_and_mul_kernel`` (``situ_and_mul.py`` beside this file, the engine kernel's
copy that the ``situ_and_mul`` entry launches) stores
``beta * tanh(g / beta) * sigmoid(g) * linear_beta * tanh(u / linear_beta)`` in BF16, and the
shared experts' down projection then quantizes it (``mxfp8_quantize``). This kernel computes each
value exactly as that kernel does (the same FP32 expression, ``beta`` and ``linear_beta`` passed
at run time, the same rounding to BF16) and quantizes the BF16 values in the same pass
(``rubin_mxfp8_epilogue``): it stores the E4M3 codes and the R128c4 scales, and no BF16 rows.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._utils import get_sm_version

from ..gemm.rubin_mxfp8_epilogue import mx_amax, mx_codes, mx_scale, r128c4_offset

#: Values per UE8M0 scale.
SF_VEC = 32
#: Columns per program: the SiTU kernel's block.
BLOCK = 1024


@triton.jit
def _situ_and_mul_mx_kernel(
    q_ptr,  # [rows, d] E4M3 codes
    s_ptr,  # flat R128c4 scales of the [rows, d] operand
    x_ptr,  # [rows, 2 * d] packed [gate | up], unit last stride
    x_stride,
    d,
    beta,
    linear_beta,
    kb_tiles,  # pad4(d / 32) / 4
    BLOCK_SIZE: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
) -> None:
    """``situ_and_mul_kernel``'s value of each element, stored as MXFP8."""
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    x_row_ptr = x_ptr + x_stride * i

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    situ_a = beta * tldevice.tanh(gate / beta) * tl.sigmoid(gate)
    up = linear_beta * tldevice.tanh(up / linear_beta)
    result = situ_a * up

    # The SiTU kernel stores result in BF16 here; the quantizer reads those values.
    yb = result.to(tl.bfloat16)
    groups: tl.constexpr = BLOCK_SIZE // 32
    amax = mx_amax(yb, groups)
    byte, mult = mx_scale(amax)
    mult = tl.reshape(tl.broadcast_to(mult[:, None], (groups, 32)), (BLOCK_SIZE,))
    q = mx_codes(yb, mult)
    tl.store(q_ptr + i * d + offsets, q, mask=mask)
    kb = j * groups + tl.arange(0, groups)
    tl.store(s_ptr + r128c4_offset(i, kb, kb_tiles), byte.to(tl.uint8), mask=kb * 32 < d)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def situ_and_mul_mxfp8(
    x: torch.Tensor, beta: float, linear_beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``mxfp8_quantize(situ_and_mul(x, beta, linear_beta), True, 32)``, one kernel.

    x: ``[rows, 2 * d]`` BF16 ``[gate | up]`` with a unit last stride (rows may be strided).
    Returns the E4M3 codes ``[rows, d]`` and the flat R128c4 UE8M0 scales (``pad128(rows) *
    pad4(d / 32)`` bytes; the padding rows' bytes zero, as the quantizer writes them).
    """
    rows, n = x.shape
    assert n % 2 == 0, "a packed [gate | up] row"
    if x.stride(1) != 1:
        raise ValueError("situ_and_mul_mxfp8 requires a contiguous last dimension")
    assert x.dtype == torch.bfloat16, "BF16 rows (the quantizer's input dtype)"
    d = n // 2
    assert d % (SF_VEC * 4) == 0, "d a multiple of 128 (whole scale groups, no padded columns)"
    codes = torch.empty((rows, d), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty(
        triton.cdiv(rows, 128) * 128 * (d // SF_VEC), dtype=torch.uint8, device=x.device
    )
    if rows % 128:
        # The last 128-row tile's padding rows: zero, as the quantizer writes them.
        scales[rows // 128 * 128 * (d // SF_VEC) :].zero_()
    if rows == 0:
        return codes, scales
    launch_with_pdl = get_env_enable_pdl() and get_sm_version() >= 90
    _situ_and_mul_mx_kernel[(rows, triton.cdiv(d, BLOCK))](
        q_ptr=codes,
        s_ptr=scales,
        x_ptr=x,
        x_stride=x.stride(0),
        d=d,
        beta=float(beta),
        linear_beta=float(linear_beta),
        kb_tiles=d // SF_VEC // 4,
        BLOCK_SIZE=BLOCK,
        LAUNCH_WITH_PDL=launch_with_pdl,
        launch_pdl=launch_with_pdl,
    )
    return codes, scales
