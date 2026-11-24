# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The low_m_linear entry on SM 10.7 (VR200) with the helix8 drafter's plans (``helix8_draft``, ``helix8_draft_d``).

* every cell of the entry and of the helix8 target computes the product within the band (the entry's ARCHS lists
  SM_107);
* every row of both helix8 plans is servable at both ends of its M range and computes the product within the band;
* at every helix8 row count (requests x 6) ``helix8_draft`` names a kernel for, its output equals F.linear's
  (cuBLAS) bit for bit over the bit battery: the bit-identical plan's claim.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm.low_m_gemm import (
    validate_choice,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
    k31_low_m_linear as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
    low_m_choice,
    plan_choice,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import (
    K31_LOW_M_PLANS,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", "../../kernels/gemm"]
from _bit_battery import bit_battery  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the helix8 drafter's plans were measured on VR200 (SM 10.7)",
)


def _cell(why: str, m: int, n: int, k: int, layout: str, kind: str) -> Cell:
    return Cell(why=why, spec=dict(m=m, n=n, k=k, layout=layout, kind=kind))


def _helix8_cells() -> tuple:
    """The helix8 drafter's projections at b4 and b9 (24 / 54 rows), with the branch each plan takes."""
    shapes = (
        ("block-convolution kernel projection", 1792, 7168),
        ("q_a", 1536, 7168),
        ("q_b", 18432, 1536),
        ("kv_a", 576, 7168),
        ("o_proj (12 heads)", 7168, 1536),
        ("gate_up (I / 8)", 3584, 7168),
        ("down (I / 8)", 7168, 1792),
        ("history kv_a (5 layers)", 2880, 7168),
    )
    cells = []
    for layout in ("helix8_draft", "helix8_draft_d"):
        for name, n, k in shapes:
            for m in (24, 54):
                choice = plan_choice(K31_LOW_M_PLANS[layout], m, n, k)
                kind = "cublas" if choice is None else choice[0]
                cells.append(
                    _cell(f"helix8 drafter {name}, {layout}, {m} rows", m, n, k, layout, kind)
                )
    return tuple(cells)


TARGET_CELLS = {"kimi_k31/kimi_k31_pro_nvfp4__sm_107__helix8": _helix8_cells()}
CELLS = certified_cells(op, TARGET_CELLS)
#: Inputs per bitwise check (the battery: random, then four each of equal rows, mixed magnitudes, cancellation and
#: spikes).
BATTERY = 20


def _build(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    return x, w


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:60] for c in CELLS])
def test_certified_cells(cell) -> None:
    s = cell.spec
    plan = K31_LOW_M_PLANS[s["layout"]]
    x, w = _build(s["m"], s["n"], s["k"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    choice = low_m_choice(plan, x, w)
    assert ("cublas" if choice is None else choice[0]) == s["kind"], (
        "the plan's branch for this cell moved"
    )
    with validating(op):
        out = op(x, w, plan, True)
    assert_within_ulp(out, op.reference(x, w, plan, True), element_ulp=2.0, rms_ulp=0.5)


def _plan_points(layouts):
    for layout in layouts:
        for (n, k), rows in K31_LOW_M_PLANS[layout].items():
            lo = 1
            for bound, choice in rows:
                if choice[0] != "default":
                    for m in sorted({lo, bound}):
                        yield layout, n, k, m, choice
                lo = bound + 1


PLAN_POINTS = list(_plan_points(("helix8_draft", "helix8_draft_d")))


@pytest.mark.parametrize(
    "layout,n,k,m,choice", PLAN_POINTS, ids=[f"{p[0]}-{p[1]}x{p[2]}-m{p[3]}" for p in PLAN_POINTS]
)
def test_every_helix8_plan_row(layout, n, k, m, choice) -> None:
    validate_choice(choice, m, n, k)
    plan = K31_LOW_M_PLANS[layout]
    x, w = _build(m, n, k, seed=m * 7919 + n)
    assert low_m_choice(plan, x, w) == choice
    assert_within_ulp(
        op(x, w, plan, True), op.reference(x, w, plan, True), element_ulp=2.0, rms_ulp=0.5
    )


#: helix8's drafter row counts: requests x 6 (a k5 verify block per request), the rows its plans were measured at.
HELIX8_ROWS = range(6, 129, 6)
B_ROWS = [
    (n, k, m)
    for (n, k) in K31_LOW_M_PLANS["helix8_draft"]
    for m in HELIX8_ROWS
    if plan_choice(K31_LOW_M_PLANS["helix8_draft"], m, n, k) is not None
]


@pytest.mark.parametrize("n,k,m", B_ROWS, ids=[f"{n}x{k}-m{m}" for n, k, m in B_ROWS])
def test_bit_identical_plan_rows(n, k, m) -> None:
    """At every helix8 row count the bit-identical plan names a kernel for, the output is F.linear's word for word
    (cuBLAS picks its kernel per row count, so the claim is held at the row counts the target runs)."""
    plan = K31_LOW_M_PLANS["helix8_draft"]
    for i, (x, w) in enumerate(bit_battery(m, n, k, seed=m * 131 + n, count=BATTERY)):
        out = op(x, w, plan, True)
        assert torch.equal(out.view(torch.int16), F.linear(x, w).view(torch.int16)), (
            f"{low_m_choice(plan, x, w)} at {m} rows, battery input {i}"
        )
