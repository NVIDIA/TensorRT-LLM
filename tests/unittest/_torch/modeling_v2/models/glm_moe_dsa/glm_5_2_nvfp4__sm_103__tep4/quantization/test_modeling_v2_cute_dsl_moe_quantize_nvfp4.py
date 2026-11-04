# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_moe_quantize_nvfp4 entry."""

import math
import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("cute_dsl_moe_quantize_nvfp4 runs sm_103 CuTe DSL kernels", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (  # noqa: E402, E501
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.quantization.cute_dsl_moe_quantize_nvfp4 import (  # noqa: E402, E501
    DECODE_MAX_ROWS,
    HIDDEN,
    SF_VEC,
    CompiledMoeQuantize,
    build_decode,
    build_prefill,
    quantize_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.quantization.cute_dsl_moe_quantize_nvfp4 import (  # noqa: E402, E501
    cute_dsl_moe_quantize_nvfp4 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

#: Compiled quantizers, one per decode row count plus one prefill quantizer, built once a session.
_HANDLES: dict[object, CompiledMoeQuantize] = {}


def _compiled(rows: int) -> CompiledMoeQuantize:
    key = "prefill" if rows > DECODE_MAX_ROWS else rows
    if key not in _HANDLES:
        device = torch.cuda.current_device()
        _HANDLES[key] = build_prefill(device) if key == "prefill" else build_decode(rows, device)
    return _HANDLES[key]


def _inputs(cell) -> tuple[torch.Tensor, torch.Tensor]:
    # Drawn on the CPU and moved over, so a seed names the same input on every
    # device and a cell's tie count below can be checked without a GPU.
    g = torch.Generator().manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    x = torch.randn(cell.spec["rows"], HIDDEN, generator=g).to(torch.bfloat16).cuda()
    # The canonical NVFP4 activation global scale, 448*6/amax.
    gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    return x, gs


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """[M, K/2] packed bytes -> [M, K] codes (element 2i in the low nibble)."""
    m, half = packed.shape
    codes = torch.empty(m, half * 2, dtype=torch.uint8, device=packed.device)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    return codes


#: The stock op forms the block scale as `gs * (amax * rcp(6))` where this op forms
#: `(gs * amax) * fp32(1/6)`. Each fp32 value is within 2**-22 of the exact product:
#: two round-to-nearest multiplies (2**-24 each), plus at most one ulp (2**-23) in the
#: approximate reciprocal of 6 on the stock side. So the two round to different e4m3 codes
#: only where the exact product lies within 2**-22 (relative) of an e4m3 rounding
#: boundary; the window below is four times that.
SCALE_BOUNDARY = 2.0**-20
#: How many distinct amax mantissas the blocks whose scales differ may share. All blocks
#: share one gs and amax is a bf16 value (128 mantissas per binade), so one unlucky
#: mantissa moves every block whose amax has it, in every binade (the 64-row cell: 175
#: blocks, all with amax 1.734375 or 3.46875, one mantissa). A defect in the scale
#: arithmetic -- a wrong constant, rounding mode or truncation -- spreads over most of
#: the ~100 mantissas a cell draws.
MAX_DIFFERING_MANTISSAS = 4


def _assert_codes(
    packed: torch.Tensor,
    ref_packed: torch.Tensor,
    scaled: torch.Tensor,
    gs: torch.Tensor,
    skip_blocks: torch.Tensor | None = None,
) -> None:
    """Codes bit-exact away from the rounding ties (and outside `skip_blocks`).

    A code is the e2m1 rounding of `x * gs / scale`, which the kernel forms through
    approximate reciprocals: inside `fp4_quantize.TIE_WINDOW` of an e2m1 midpoint either
    neighbour is a correct answer, and `near_tie` marks those elements. Outside it the
    code has to match bit for bit; a wrong code there is a whole e2m1 step, and an
    unwritten byte (0xFF) a wrong code.

    The tie count is bounded so the window cannot become a licence for wrong codes. It
    is set by the input's global amax, not by the kernel: the fp4_quantize test measured
    0.13% to 0.24% for ordinary draws, and 1.22% at an amax of 5.25, where the global
    scale is 2688 / 5.25 = 512, a power of two, and many products land exactly on a
    midpoint. The 1% bound applies to every draw except a power-of-two global scale,
    which gets 1.5%. On the CPU, these cells' seeds give 0.07% to 0.20%, except the
    2049-row cell, whose draw has amax 5.25 and gives 1.21%.
    """
    tie = fp4_quantize.near_tie(scaled)
    keep = ~tie
    if skip_blocks is not None:
        keep &= ~skip_blocks.repeat_interleave(SF_VEC, dim=1)
    codes, ref_codes = _unpack(packed), _unpack(ref_packed)
    assert torch.equal(codes[keep], ref_codes[keep]), "codes differ away from a rounding tie"
    bound = 0.015 if math.frexp(gs.item())[0] == 0.5 else 0.01
    assert tie.sum().item() <= bound * tie.numel(), (
        f"{tie.sum().item()} of {tie.numel()} elements sit in the tie window; "
        "a window this wide is no longer a statement about rounding"
    )


def _near_scale_boundary(x: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    """Blocks whose exact scale `gs * amax / 6` lies within `SCALE_BOUNDARY` (relative)
    of an e4m3 rounding boundary: the two ends of the window round to different codes."""
    rows = int(x.shape[0])
    amax = x.double().reshape(rows, HIDDEN // SF_VEC, SF_VEC).abs().amax(-1)
    exact = gs.double().reshape(()) * amax / 6.0

    def code(v: torch.Tensor) -> torch.Tensor:
        return v.float().clamp(max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)

    return code(exact * (1.0 - SCALE_BOUNDARY)) != code(exact * (1.0 + SCALE_BOUNDARY))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    rows = cell.spec["rows"]
    compiled = _compiled(rows)
    x, gs = _inputs(cell)
    # 0xFF bytes: an e4m3 NaN scale and two -6 codes, so an unwritten byte fails.
    fp4_out = torch.full((rows, HIDDEN // 2), 0xFF, dtype=torch.uint8, device="cuda")
    scale_out = torch.full((rows, HIDDEN // SF_VEC), 0xFF, dtype=torch.uint8, device="cuda")
    with validating(op):
        packed, scales = op(x, gs, fp4_out, scale_out, compiled)
    assert packed is fp4_out and scales is scale_out, "the op writes the buffers it is given"
    ref_packed, ref_scales = op.reference(x, gs, fp4_out, scale_out, compiled)
    # The reference forms the block scale in the kernels' order, `(gs * amax) * fp32(1/6)`,
    # each multiply rounded to nearest, then one rounding to e4m3: the same roundings in
    # the same places, so any scale that differs is a defect.
    assert torch.equal(scales, ref_scales), "block scales are exact; a mismatch is a defect"
    _assert_codes(packed, ref_packed, quantize_reference(x, gs)[2], gs)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_the_stock_op_writes_the_same_layout(cell) -> None:
    """The call site falls back to `fp4_quantize` with linear scales when this
    op did not build, and the grouped GEMM reads either result the same way.

    So the two must agree: scales exact (the stock op's 1-D buffer viewed as
    `[rows, 384]`) except blocks next to an e4m3 rounding boundary, where the two
    association orders may pick neighbouring codes (`SCALE_BOUNDARY`); those blocks may
    differ by one code, must trace to at most `MAX_DIFFERING_MANTISSAS` amax values, and
    are left out of the code comparison (their codes are scaled differently). Codes
    exact elsewhere, outside the tie window where both kernels may pick either neighbour.
    """
    rows = cell.spec["rows"]
    compiled = _compiled(rows)
    x, gs = _inputs(cell)
    fp4_out = torch.full((rows, HIDDEN // 2), 0xFF, dtype=torch.uint8, device="cuda")
    scale_out = torch.full((rows, HIDDEN // SF_VEC), 0xFF, dtype=torch.uint8, device="cuda")
    with validating(op, fp4_quantize):
        op(x, gs, fp4_out, scale_out, compiled)
        stock_packed, stock_scales = fp4_quantize(x, gs, SF_VEC, False, False)
    assert stock_packed.shape == fp4_out.shape
    assert stock_scales.numel() == scale_out.numel(), "the stock linear scales are not padded"
    stock_scales = stock_scales.view(rows, HIDDEN // SF_VEC)
    differ = scale_out != stock_scales
    near = _near_scale_boundary(x, gs)
    assert not bool((differ & ~near).any()), "scales differ away from an e4m3 rounding boundary"
    step = (scale_out.int() - stock_scales.int()).abs()
    assert bool((step[differ] <= 1).all()), "a scale next to a boundary moved more than one code"
    amax = x.float().reshape(rows, HIDDEN // SF_VEC, SF_VEC).abs().amax(-1)
    mantissas = (amax[differ].to(torch.bfloat16).view(torch.int16) & 0x7F).unique()
    assert mantissas.numel() <= MAX_DIFFERING_MANTISSAS, (
        f"differing scales span {mantissas.numel()} amax mantissas; a boundary effect "
        f"touches at most {MAX_DIFFERING_MANTISSAS}"
    )
    _assert_codes(fp4_out, stock_packed, quantize_reference(x, gs)[2], gs, skip_blocks=differ)


def _unreachable(*args) -> None:
    raise AssertionError("the guard should have refused before the kernel ran")


def _ok(rows: int):
    x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    fp4_out = torch.empty(rows, HIDDEN // 2, dtype=torch.uint8, device="cuda")
    scale_out = torch.empty(rows, HIDDEN // SF_VEC, dtype=torch.uint8, device="cuda")
    return x, fp4_out, scale_out


def test_guard_refuses_a_per_row_global_scale() -> None:
    """The kernel loads element 0 and applies it to every row."""
    x, fp4_out, scale_out = _ok(8)
    gs = torch.ones(8, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="global_scale must hold one fp32 element"):
        with validating(op):
            op(x, gs, fp4_out, scale_out, CompiledMoeQuantize("decode", 8, _unreachable))


def test_guard_refuses_a_row_count_the_decode_handle_was_not_compiled_for() -> None:
    """The decode quantizer's row count is compiled in: a 9th row is never quantized."""
    x, fp4_out, scale_out = _ok(9)
    gs = torch.ones(1, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="compiled for 8 rows, got 9"):
        with validating(op):
            op(x, gs, fp4_out, scale_out, CompiledMoeQuantize("decode", 8, _unreachable))


def test_guard_refuses_a_strided_input() -> None:
    """A column slice of a wider buffer, read as dense rows through its pointer."""
    wide = torch.zeros(8, HIDDEN + 64, dtype=torch.bfloat16, device="cuda")
    _, fp4_out, scale_out = _ok(8)
    gs = torch.ones(1, dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="x must be contiguous"):
        with validating(op):
            op(
                wide[:, :HIDDEN],
                gs,
                fp4_out,
                scale_out,
                CompiledMoeQuantize("decode", 8, _unreachable),
            )
