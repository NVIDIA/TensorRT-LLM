# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_mxfp8_linear entry: ModelOpt MXFP8 W8A8 projections on sm_107.

Every cell (the KDA projections and the shared experts at each layout's shapes) runs the entry
against its reference: the input quantized the kernel's way, both operands dequantized exactly,
an FP32 product, one BF16 rounding.
"""

import pytest
import torch
from _rubin_gemm_util import activation, generator, mxfp8_weight, requires_sm107

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_mxfp8_linear import (  # noqa: E501
    rubin_mxfp8_linear as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = requires_sm107

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _args(spec, gen):
    x = activation(spec["m"], spec["k"], gen)
    weight, weight_scale = mxfp8_weight(spec["n"], spec["k"], gen)
    return x, weight, weight_scale


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _args(cell.spec, generator(cell.why))
    out = op(*args)
    # The reference quantizes the input to the kernel's bytes (test_rubin_mxfp8_gemm checks that
    # bit for bit), so what is left is the GEMM's: exact FP32 terms summed in another order and
    # rounded once to BF16 (2 units of 2^-8 of the row's largest magnitude, half a unit in RMS).
    assert_within_ulp(out, op.reference(*args), element_ulp=2.0, rms_ulp=0.5)


def test_a_replayed_graph_matches_eager() -> None:
    args = _args(dict(m=8, n=6144, k=7168), generator("graph"))
    eager = op(*args)  # compiles and makes the FP32 one outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_guard_refuses_twelve_output_rows() -> None:
    """The KDA b_proj at TP8 has 12 rows: 24-byte BF16 output rows. The runner offers no tactic
    for them and falls back to its default tile, which is not built for them."""
    gen = generator("guard rows")
    x = activation(8, 7168, gen)
    weight, weight_scale = mxfp8_weight(12, 7168, gen)
    with pytest.raises(AssertionError, match="multiple of 8"):
        with validating(op):
            op(x, weight, weight_scale)


def test_guard_refuses_a_row_major_weight_scale() -> None:
    gen = generator("guard scale")
    x = activation(8, 128, gen)
    weight, flat = mxfp8_weight(128, 128, gen)
    with pytest.raises(AssertionError, match="row-major"):
        with validating(op):
            op(x, weight, flat.view(128, 4))
