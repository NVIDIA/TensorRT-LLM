# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the k3_kda_fab_proj and k3_kda_fab_fa entries: every cell of
``k3_kda_fab_proj.CELLS`` and ``k3_kda_fab_fa.CELLS`` against the fp64 GEMM references, and f_a /
beta of the two entries bitwise equal."""

import zlib

import pytest
import torch

pytest.importorskip("cutlass")

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_kda_fab_fa import (  # noqa: E501, E402
    k3_kda_fab_fa,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_kda_fab_proj import (  # noqa: E501, E402
    FA_DIM,
    HIDDEN,
    bfa_rows,
    k3_kda_fab_proj,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating


def _driven_here() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in {arch.value for arch in k3_kda_fab_proj.ARCHS}


pytestmark = pytest.mark.skipif(not _driven_here(), reason="k3_kda_fab_proj.ARCHS")


def _operands(heads: int, rows: int, seed: int):
    """``x`` with a few outlier channels (as the KDA input carries), ``[f_a; b]`` with its zero
    padding rows, and ``f_b``."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=gen)
    x[:, torch.randperm(HIDDEN, generator=gen)[:16]] *= 30.0
    w_bfa = torch.randn(bfa_rows(heads), HIDDEN, generator=gen) * 0.02
    w_bfa[FA_DIM + heads :] = 0
    w_fb = torch.randn(heads * FA_DIM, FA_DIM, generator=gen) * 0.05
    return (t.bfloat16().cuda() for t in (x, w_bfa, w_fb))


def _seed(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


# The kernel sums fp32 partials of a 7168-long dot product in a fixed split order; this side
# sums in fp64. Both round once to bf16, so an element lands at most one ulp of its row's scale
# away, and the forget gate (a GEMM of those bf16 values) inherits that ulp: two ulps bound it.
ELEMENT_ULP = 2.0
RMS_ULP = 0.25


@pytest.mark.parametrize(
    "cell", k3_kda_fab_proj.CELLS, ids=[c.why[:60] for c in k3_kda_fab_proj.CELLS]
)
@torch.no_grad()
def test_fab_proj_cells(cell) -> None:
    heads, rows = cell.spec["heads"], cell.spec["rows"]
    x, w_bfa, w_fb = _operands(heads, rows, _seed(cell.why))
    with validating(k3_kda_fab_proj):
        beta, gate = k3_kda_fab_proj(x, w_bfa, w_fb, heads)
    ref_beta, ref_gate = k3_kda_fab_proj.reference(x, w_bfa, w_fb, heads)
    assert beta.shape == (rows, heads) and gate.shape == (rows, heads * FA_DIM), cell.why
    assert_within_ulp(beta, ref_beta, ELEMENT_ULP, RMS_ULP)
    assert_within_ulp(gate, ref_gate, ELEMENT_ULP, RMS_ULP)


@pytest.mark.parametrize("cell", k3_kda_fab_fa.CELLS, ids=[c.why[:60] for c in k3_kda_fab_fa.CELLS])
@torch.no_grad()
def test_fab_fa_cells(cell) -> None:
    heads, rows = cell.spec["heads"], cell.spec["rows"]
    x, w_bfa, w_fb = _operands(heads, rows, _seed(cell.why))
    with validating(k3_kda_fab_fa):
        bfa = k3_kda_fab_fa(x, w_bfa, heads, w_fb)
    assert bfa.shape == (rows, bfa_rows(heads)), cell.why
    assert_within_ulp(bfa, k3_kda_fab_fa.reference(x, w_bfa, heads, w_fb), ELEMENT_ULP, RMS_ULP)
    # f_a mode is the full kernel's GEMM 1 and reduction: bitwise its beta (and f_a).
    beta, _ = k3_kda_fab_proj(x, w_bfa, w_fb, heads)
    assert torch.equal(bfa[:, FA_DIM : FA_DIM + heads], beta), cell.why
