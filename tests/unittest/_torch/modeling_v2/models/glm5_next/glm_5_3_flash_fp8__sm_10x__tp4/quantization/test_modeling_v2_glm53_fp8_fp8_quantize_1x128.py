# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_quantize_1x128 operation of the glm5_next FP8 target.

The entry owns the contract. This file turns a cell into an input, holds every code
disagreement of a cell to a rounding tie, and adds what a cell cannot carry: the loud
rejection of a strided input.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    BLOCK,
    block_scales,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.quantization.fp8_quantize_1x128 import (  # noqa: E501
    fp8_quantize_1x128 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="fp8_quantize_1x128 is certified on sm_100"
)

DEV = "cuda"

#: The kernel's two fp32 divisions are approximate, each within CUDA's documented 2 ulp of the
#: exact quotient, and the returned scale is two divisions deep.
SCALE_ULP_GATE = 4

#: A code can only disagree with the exactly rounded one when the exact product sits within
#: the first division's 2 ulp of an e4m3 rounding midpoint.
TIE_REL_GATE = 2.0 * 2.0**-23


def _ordinal(codes: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude ordinal of an e4m3 byte: adjacent codes differ by 1."""
    b = codes.view(torch.uint8).int()
    return torch.where(b >= 128, -(b & 0x7F), b & 0x7F)


def _compare(
    out: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Scales within 4 fp32 ulp; codes at most one e4m3 step apart, >= 99.5% bit-exact.

    The kernel's two fp32 divisions are approximate, so neither output is reproducible to the
    last bit: a scale can sit up to 2 ulp from `amax / 448` per division, and a code can land
    on the other side of a rounding midpoint when the exact product sits within that band of
    it -- one e4m3 step, never more. `_assert_misses_are_ties` holds every such miss to the
    band. bf16 inputs make exact midpoints common (`x * 448 / amax` of two bf16 values often
    has few significant bits), which is why the bit-exact floor is 99.5% and not higher.

    A wrong recipe is far outside this: a power-of-two scale, a per-row or per-tensor amax,
    blocks down the row axis or offset by 64 columns, or 240 as the e4m3 maximum each change
    most codes and move the scales by O(1) relative.
    """
    data, scales = out
    ref_data, ref_scales = ref
    assert data.dtype == ref_data.dtype == torch.float8_e4m3fn, (data.dtype, ref_data.dtype)
    assert scales.dtype == ref_scales.dtype == torch.float32, (scales.dtype, ref_scales.dtype)
    assert data.shape == ref_data.shape, (data.shape, ref_data.shape)
    assert scales.shape == ref_scales.shape, (scales.shape, ref_scales.shape)
    ulp = (
        scales.contiguous().view(torch.int32).long()
        - ref_scales.contiguous().view(torch.int32).long()
    )
    worst = int(ulp.abs().max())
    assert worst <= SCALE_ULP_GATE, f"scale {worst} fp32 ulp off > {SCALE_ULP_GATE}"
    step = (_ordinal(data) - _ordinal(ref_data)).abs()
    assert int(step.max()) <= 1, f"a code is {int(step.max())} e4m3 steps off"
    exact = 1.0 - (step != 0).float().mean().item()
    assert exact >= 0.995, f"only {100 * exact:.4f}% of codes bit-exact"


def _input(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(rows, cols, device=DEV, generator=g) * 3.0).to(torch.bfloat16)


def _assert_misses_are_ties(x: torch.Tensor, data: torch.Tensor, ref_data: torch.Tensor) -> float:
    """Every code disagreement is a rounding tie: the exact product sits within the scale's
    2-ulp band of the midpoint between the two codes."""
    qs, _ = block_scales(x, False)
    prod = x.float() * qs.repeat_interleave(BLOCK, dim=1)[:, : x.shape[1]]
    diff = data.float() != ref_data.float()  # -0 and +0 are the same code here
    if not bool(diff.any()):
        return 0.0
    mid = ((data.float() + ref_data.float()) / 2).double()
    rel = ((prod.double()[diff] - mid[diff]).abs() / mid[diff].abs()).max().item()
    assert rel <= TIE_REL_GATE, f"a miss {rel:.3e} from its midpoint is not a tie"
    return rel


# The shapes the target calls this operation with, on top of the operation's own `CELLS`.
#
# GLM-5.3-Flash at TP4: every FFN input is the hidden row (4096); the down projections read
# the SwiGLU output at the per-rank intermediate (dense 12288 / 4 = 3072, shared expert
# 2048 / 4 = 512); the sequence-parallel MLA prefill projects q_resid (q_lora 1536) through q_b.
# Generation steps are MTP-3 verifies of 4 rows per request (up to 64 requests); a 13,312-row
# prefill runs sequence parallel, each rank quantizing its own quarter (3,328 rows) of the
# FFN input and the down-projection inputs over every gathered row.
CELLS: tuple[Cell, ...] = op.CELLS + (
    Cell(
        why="MTP-3 verify of one request: 4 rows at hidden 4096",
        spec=dict(rows=4, cols=4096, use_ue8m0=False),
    ),
    Cell(
        why="MTP-3 verify of 64 requests: 256 rows at hidden 4096",
        spec=dict(rows=256, cols=4096, use_ue8m0=False),
    ),
    Cell(
        why="sequence-parallel prefill: one rank's 3,328 rows of 13,312 at hidden 4096",
        spec=dict(rows=3328, cols=4096, use_ue8m0=False),
    ),
    Cell(
        why="dense MLP down_proj input at TP4 (3072), 256 rows",
        spec=dict(rows=256, cols=3072, use_ue8m0=False),
    ),
    Cell(
        why="dense MLP down_proj input over a 13,312-row prefill step",
        spec=dict(rows=13312, cols=3072, use_ue8m0=False),
    ),
    Cell(
        why="shared expert down_proj input at TP4 (512), 256 rows",
        spec=dict(rows=256, cols=512, use_ue8m0=False),
    ),
    Cell(
        why="shared expert down_proj input over a 13,312-row prefill step",
        spec=dict(rows=13312, cols=512, use_ue8m0=False),
    ),
    Cell(
        why="MLA q_b input (q_lora 1536) over one rank's 3,328 sequence-parallel rows",
        spec=dict(rows=3328, cols=1536, use_ue8m0=False),
    ),
)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    x = _input(spec["rows"], spec["cols"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    data, scales = op(x, spec["use_ue8m0"])
    nb = (spec["cols"] + BLOCK - 1) // BLOCK
    assert data.shape == x.shape and data.is_contiguous() and data.device == x.device
    assert scales.shape == (nb, spec["rows"]) and scales.is_contiguous()
    ref = op.reference(x, spec["use_ue8m0"])
    _compare((data, scales), ref)
    rel = _assert_misses_are_ties(x, data, ref[0])
    exact = (data.view(torch.uint8) == ref[0].view(torch.uint8)).float().mean().item()
    print(f"  {cell.why[:44]}: {100 * exact:.4f}% bit-exact, worst tie {rel:.3e}")


def test_rejects_a_strided_input():
    """The op's own precondition, loud, which is why there is no guard for it."""
    x = torch.randn(8, 8192, dtype=torch.bfloat16, device=DEV)[:, :4096]
    with pytest.raises((RuntimeError, NotImplementedError)):
        op(x)
