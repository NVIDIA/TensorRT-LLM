# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the dsv3_router_gemm operation of the glm5_next FP8 target.

The entry owns the contract. This file turns a cell into the router's operands, gates the op
against the entry's float64 reference in fp32 ulp, and drives the bias guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.gemm.dsv3_router_gemm import (  # noqa: E501
    GLM_ROUTER,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.gemm.dsv3_router_gemm import (  # noqa: E501
    dsv3_router_gemm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="dsv3_router_gemm is certified on sm_100"
)

DEV = "cuda"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """256 fp32 ulp of the row max element-wise, 64 fp32 ulp of relative RMS.

    The reference sums in float64, so the band measures only the op's own accumulation of
    4096 exact bf16 products. The small-M kernel adds in IEEE fp32: an fp32 simulation of its
    order (32 sequential products per thread, butterfly, 4-warp sum) against a float64 sum
    over 256 x 288 outputs of N(0, 1) bf16 operands reaches 1.2 ulp worst and 0.7 RMS. The
    cuBLASLt path accumulates on the tensor cores, whose fp32 accumulation is coarser than
    IEEE round-to-nearest: on B200 it reached 74 ulp worst (13,312 rows) and 17.7 ulp RMS
    (256 rows) -- more than a fully sequential IEEE sum, which the same simulation puts at
    17.6 / 5.2. The band leaves 3.5x over that and still rejects a bf16-rounded output (one
    bf16 ulp is 65536 fp32 ulp) or a TF32 product (thousands of ulp RMS) by orders of
    magnitude.
    """
    assert out.dtype == ref.dtype == torch.float32, (out.dtype, ref.dtype)
    assert_within_ulp(out, ref, element_ulp=256.0, rms_ulp=64.0, ulp=ULP[torch.float32])


def _build(spec: dict, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 hidden rows and the router weight's `.t()`: the weight is built `[N, K]` and
    transposed, which is what makes `mat_b` column-major without a copy."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    hidden = torch.randn(spec["rows"], spec["hidden"], generator=g, device=DEV)
    weight = torch.randn(spec["experts"], spec["hidden"], generator=g, device=DEV)
    return hidden.to(torch.bfloat16), weight.to(torch.bfloat16).t()


# The shapes the target calls this operation with, on top of the operation's own `CELLS`.
#
# GLM-5.3-Flash at TP4: the router is replicated, so every rank routes its rows over all 288
# experts from the full hidden width. Generation steps are MTP-3 verifies of 4 rows per
# request (up to 64 requests) and the drafter's single rows; a 13,312-row prefill runs
# sequence parallel, each rank routing its own quarter (3,328 rows).
CELLS: tuple[Cell, ...] = op.CELLS + (
    Cell(why="one row: the MTP drafter's decode step", spec=dict(rows=1, **GLM_ROUTER)),
    Cell(
        why="MTP-3 verify of one request: 4 rows on the small-M kernel",
        spec=dict(rows=4, **GLM_ROUTER),
    ),
    Cell(
        why="MTP-3 verify of two requests: 8 rows on the small-M kernel",
        spec=dict(rows=8, **GLM_ROUTER),
    ),
    Cell(
        why="MTP-3 verify of 8 requests: 32 rows on cuBLASLt",
        spec=dict(rows=32, **GLM_ROUTER),
    ),
    Cell(
        why="MTP-3 verify of 64 requests: 256 rows on cuBLASLt",
        spec=dict(rows=256, **GLM_ROUTER),
    ),
    Cell(
        why="sequence-parallel prefill: one rank's 3,328 rows of a 13,312-row step",
        spec=dict(rows=3328, **GLM_ROUTER),
    ),
    Cell(
        why="a 13,312-row prefill step routed in one call",
        spec=dict(rows=13312, **GLM_ROUTER),
    ),
)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    mat_a, mat_b = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(mat_a, mat_b, None, torch.float32)
    assert out.shape == (cell.spec["rows"], cell.spec["experts"]) and out.is_contiguous()
    _compare(out, op.reference(mat_a, mat_b, None, torch.float32))


def test_padded_operands_are_repacked() -> None:
    """A hidden row read out of a wider buffer, and a router weight with padded rows, are
    copied into the dense layout before either kernel reads them -- no guard is needed."""
    g = torch.Generator(device=DEV).manual_seed(3)
    wide = torch.randn(8, 4096 + 64, generator=g, device=DEV).to(torch.bfloat16)
    padded = torch.randn(288, 4096 + 64, generator=g, device=DEV).to(torch.bfloat16)
    mat_a, mat_b = wide[:, :4096], padded[:, :4096].t()
    assert mat_a.stride(0) != 4096 and mat_b.stride(1) != 4096
    with validating(op):
        out = op(mat_a, mat_b, None, torch.float32)
    _compare(out, op.reference(mat_a, mat_b, None, torch.float32))


def _ok() -> tuple[torch.Tensor, torch.Tensor]:
    return _build(dict(rows=4, hidden=256, experts=32), seed=5)


def test_guard_refuses_a_bias_in_the_wrong_dtype() -> None:
    mat_a, mat_b = _ok()
    bias = torch.randn(32, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(AssertionError, match="bias dtype"):
        with validating(op):
            op(mat_a, mat_b, bias, torch.float32)


def test_guard_refuses_a_bias_of_the_wrong_shape() -> None:
    mat_a, mat_b = _ok()
    bias = torch.randn(64, dtype=torch.float32, device=DEV)[::2]
    with pytest.raises(AssertionError, match="contiguous \\[N\\] tensor"):
        with validating(op):
            op(mat_a, mat_b, bias, torch.float32)
