# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 low_m_linear entry (decode / verify BF16 projections by a per-layout plan)."""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.gemm.low_m_gemm import (
    encode_choice,
    launch,
    validate_choice,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
    k31_low_m_linear as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_linear import (
    low_m_choice,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import (
    K31_LOW_M_PLANS,
    k31_low_m_draft_plan,
    k31_low_m_plan,
)

__extra_import_path__ = ["../../../..", "../../kernels/gemm"]
from _bit_battery import bit_battery  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "low_m_linear requires a CUDA device"

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _build(m, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    return x, w


def _kind(choice):
    return "cublas" if choice is None else choice[0]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    s = cell.spec
    plan = K31_LOW_M_PLANS[s["layout"]]
    x, w = _build(s["m"], s["n"], s["k"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    assert _kind(low_m_choice(plan, x, w)) == s["kind"], "the plan's branch for this cell moved"
    with validating(op):
        out = op(x, w, plan, True)
    ref = op.reference(x, w, plan, True)
    # One BF16 rounding of an FP32 sum taken in another order (see test_fused_a_gemm).
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


def _plan_points():
    """Every plan row at its lowest and highest M: (layout, n, k, m, choice)."""
    for layout, plan in K31_LOW_M_PLANS.items():
        for (n, k), rows in plan.items():
            lo = 1
            for bound, choice in rows:
                if choice[0] != "default":
                    for m in sorted({lo, bound}):
                        yield layout, n, k, m, choice
                lo = bound + 1


PLAN_POINTS = list(_plan_points())


@pytest.mark.parametrize(
    "layout,n,k,m,choice", PLAN_POINTS, ids=[f"{p[0]}-{p[1]}x{p[2]}-m{p[3]}" for p in PLAN_POINTS]
)
def test_every_plan_row(layout, n, k, m, choice) -> None:
    """Every row of every layout's plan is servable at both ends of its M range and computes the product."""
    validate_choice(choice, m, n, k)
    plan = K31_LOW_M_PLANS[layout]
    x, w = _build(m, n, k, seed=m * 7919 + n)
    assert low_m_choice(plan, x, w) == choice
    out = op(x, w, plan, True)
    assert_within_ulp(out, op.reference(x, w, plan, True), element_ulp=2.0, rms_ulp=0.5)


def test_unservable_operands_take_cublas() -> None:
    """A strided activation (a column slice) or no plan runs F.linear, bitwise."""
    plan = K31_LOW_M_PLANS["tep32"]
    wide, w = _build(8, 2 * 7168, 1536, seed=5)
    x = wide[:, :7168]
    assert low_m_choice(plan, x, w) is None
    assert torch.equal(op(x, w, plan, True), F.linear(x, w))
    x2, w2 = _build(8, 1536, 7168, seed=6)
    assert torch.equal(op(x2, w2, None, True), F.linear(x2, w2))


def test_guard_refuses_mismatched_k() -> None:
    x = torch.randn(4, 7168, device="cuda").to(torch.bfloat16)
    w = torch.randn(1536, 1536, device="cuda").to(torch.bfloat16)
    with pytest.raises(AssertionError, match="weight"):
        with validating(op):
            op(x, w, K31_LOW_M_PLANS["tep32"], True)


# Inputs per bitwise check: the bit battery (``_bit_battery``; random inputs, then four each of equal rows, mixed
# magnitudes, exact cancellation and spikes). One random input can match a kernel that sums K in another order.
BATTERY = 20

# Plan rows that are bitwise the GEMM they replaced: (layout, n, k, rows, the previous branch). A TGV tactic gives the
# bits of every other TGV tactic, and of cuBLAS (F.linear) where cuBLAS's kernel has no split-K.
_T16 = ("tgv", 64, 16, 12, True)
_T32 = ("tgv", 64, 32, 11, True)
BITWISE_ROWS = (
    tuple(
        ("tep16", 3072, 7168, m, prev)
        for ms, prev in (
            ((33, 40, 48), _T16),
            ((49, 64, 65, 80, 96), _T32),
            ((97, 105, 112, 113, 120, 128, 129, 144, 145, 160, 176, 192), "cublas"),
            # 2-CTA 80 / 96-token tiles (outside FlashInfer's table) against its 128-token tile.
            ((130, 141, 150, 161, 170, 184), ("tgv", 64, 128, 7, True)),
        )
        for m in ms
    )
    + tuple(
        # The TEP16 shared expert's gate_up on the chunked TGV with cuBLAS's split: cuBLAS's split-K + reduce bits.
        ("tep16_shared", 768, 7168, m, "cublas")
        for m in (33, 40, 48, 49, 56, 64)
    )
    + tuple(
        # TEP32 MLA g_proj / shared-expert gate_up above 32 rows on the chunked split-K: cuBLAS's split-K + reduce bits.
        ("tep32", 384, 7168, m, "cublas")
        for m in (33, 35, 40, 48, 49, 56, 64)
    )
)


@pytest.mark.parametrize(
    "layout,n,k,m,prev", BITWISE_ROWS, ids=[f"{r[0]}-{r[1]}x{r[2]}-m{r[3]}" for r in BITWISE_ROWS]
)
def test_bitwise_rows(layout, n, k, m, prev) -> None:
    """The plan's output at these rows equals, bit for bit, the branch the rows took before (a TGV tactic or
    cuBLAS's F.linear)."""
    plan = K31_LOW_M_PLANS[layout]
    for i, (x, w) in enumerate(bit_battery(m, n, k, seed=m * 131 + n, count=BATTERY)):
        out = op(x, w, plan, True)
        if prev == "cublas":
            before = F.linear(x, w)
        else:
            before = torch.empty_like(out)
            launch(*encode_choice(prev), x, w, before, True)
        assert torch.equal(out.view(torch.int16), before.view(torch.int16)), (
            f"{low_m_choice(plan, x, w)} at {m} rows, battery input {i}"
        )


# TEP16 projections the target runs at 1-32 rows, with the GEMM the general target runs for each: F.linear (the
# engine Linear), cublas_mm for the indexer wq_b, k31_fused_a_gemm for the MoE latent down / up.
TEP16_GENERAL = (
    ("KDA [q|k|v|g]", 3072, 7168, "linear"),
    ("MLA g_proj, shared-expert gate_up", 768, 7168, "linear"),
    ("MLA q_b", 1152, 1536, "linear"),
    ("indexer wq_b", 4096, 1536, "cublas_mm"),
    ("MoE latent down", 3584, 7168, "fused_a"),
    ("MoE latent up", 7168, 3584, "fused_a"),
    ("dense layer-0 gate / up", 4224, 7168, "linear"),
)


@pytest.mark.parametrize("name,n,k,general", TEP16_GENERAL, ids=[g[0] for g in TEP16_GENERAL])
def test_tep16_rows_bitwise_to_general_target(name, n, k, general) -> None:
    """At 1-32 rows the TEP16 plan's output equals, bit for bit, the GEMM the general target runs for the projection
    (where the plan names no kernel the target runs that GEMM itself)."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import (
        k31_fused_a_gemm,
    )

    plan = K31_LOW_M_PLANS["tep16"]
    for m in range(1, 33):
        for i, (x, w) in enumerate(bit_battery(m, n, k, seed=m * 977 + n, count=BATTERY)):
            if general == "fused_a":
                reference = k31_fused_a_gemm(x, w.t())
            elif general == "cublas_mm":
                reference = cublas_mm(x, w.t(), None, None, 0, None)
            else:
                reference = F.linear(x, w)
            if low_m_choice(plan, x, w) is None and general == "fused_a":
                out = k31_fused_a_gemm(x, w.t())
            else:
                out = op(x, w, plan, True)
            assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), (
                f"{name}: {low_m_choice(plan, x, w)} at {m} rows, battery input {i}"
            )


# The TEP16 KDA gate pair [f_a | b] (128 f_a rows + 6 heads, padded to 136): cuBLAS splits K in 4 even chunks at 65-128
# rows, which the plan's wide chunked split-K reproduces where it names it.
TEP16_GATE_PAIR = (136, 7168, 65, 128)


def test_tep16_gate_pair_bitwise_to_general_target() -> None:
    """At 65-128 rows the TEP16 plan's [f_a | b] output equals, bit for bit, the general target's F.linear."""
    n, k, lo, hi = TEP16_GATE_PAIR
    plan = K31_LOW_M_PLANS["tep16"]
    planned = 0
    for m in range(lo, hi + 1):
        for i, (x, w) in enumerate(bit_battery(m, n, k, seed=m * 977 + n, count=BATTERY)):
            planned += low_m_choice(plan, x, w) is not None
            out = op(x, w, plan, True)
            assert torch.equal(out.view(torch.int16), F.linear(x, w).view(torch.int16)), (
                f"[f_a | b]: {low_m_choice(plan, x, w)} at {m} rows, battery input {i}"
            )
    assert planned, "the TEP16 plan names no kernel for the gate pair"


# DEP16 projections the target runs at 1-80 rows per rank (the plan's rows, held to the general target's bits), with
# the GEMM the general target runs for each at attention DP: F.linear (the engine Linear; the KDA [q|k|v|g] on its
# fused weight), cublas_mm for the indexer wq_b, k31_fused_a_gemm for the MoE latent down / up. Not listed: the packed
# DSA hidden projections (the general target runs them as separate GEMMs) and the drafter's projections (drafter-only
# numerics).
DEP16_GENERAL = (
    ("KDA [q|k|v|g]", 49152, 7168, "linear"),
    ("KDA / MLA o_proj", 7168, 12288, "linear"),
    ("MLA g_proj, shared-expert gate_up", 12288, 7168, "linear"),
    ("shared-expert down", 7168, 6144, "linear"),
    ("MLA q_b", 18432, 1536, "linear"),
    ("indexer wq_b", 4096, 1536, "cublas_mm"),
    ("MoE latent down", 3584, 7168, "fused_a"),
    ("MoE latent up", 7168, 3584, "fused_a"),
    ("dense layer-0 gate / up", 67584, 7168, "linear"),
)
DEP16_ROWS = 80


@pytest.mark.parametrize("name,n,k,general", DEP16_GENERAL, ids=[g[0] for g in DEP16_GENERAL])
def test_dep16_rows_bitwise_to_general_target(name, n, k, general) -> None:
    """At 1-80 rows DEP16's plan gives, bit for bit, the GEMM the general target runs for the projection
    (where the plan names no kernel the target runs that GEMM itself)."""
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import (
        k31_fused_a_gemm,
    )

    plan = K31_LOW_M_PLANS["dep16"]
    for m in range(1, DEP16_ROWS + 1):
        for i, (x, w) in enumerate(bit_battery(m, n, k, seed=m * 977 + n, count=BATTERY)):
            if general == "fused_a":
                reference = k31_fused_a_gemm(x, w.t())
            elif general == "cublas_mm":
                reference = cublas_mm(x, w.t(), None, None, 0, None)
            else:
                reference = F.linear(x, w)
            if low_m_choice(plan, x, w) is None and general == "fused_a":
                out = k31_fused_a_gemm(x, w.t())
            else:
                out = op(x, w, plan, True)
            assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), (
                f"{name}: {low_m_choice(plan, x, w)} at {m} rows, battery input {i}"
            )


# TEP8 projections with a bitwise plan entry, with the GEMM the TEP8 target runs for each without a plan (F.linear: the
# engine Linear, the packed DSA hidden projection as one GEMM; k31_fused_a_gemm for the MoE latent down) and the rows
# the plan names kernels for.
TEP8_PLANLESS = (
    ("MLA g_proj, shared-expert gate_up", 1536, 7168, "linear", 1, 32),
    ("packed DSA hidden projection", 3808, 7168, "linear", 1, 32),
    ("MoE latent down", 3584, 7168, "fused_a", 1, 32),
    ("MLA q_b", 2304, 1536, "linear", 1, 8),
    ("DSA indexer wq_b", 4096, 1536, "linear", 1, 8),
    # Above 32 rows the [q|k|v|g] GEMM takes the chunked TGV with cuBLAS's split (below 32 its TGV is not bitwise).
    ("KDA [q|k|v|g] at 33-64 rows", 6144, 7168, "linear", 33, 64),
)


@pytest.mark.parametrize(
    "name,n,k,planless,lo,hi", TEP8_PLANLESS, ids=[g[0] for g in TEP8_PLANLESS]
)
def test_tep8_rows_bitwise_to_planless_path(name, n, k, planless, lo, hi) -> None:
    """At every row the TEP8 plan names in ``lo``-``hi`` its output equals, bit for bit, the GEMM the TEP8 target runs
    without a plan."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import (
        k31_fused_a_gemm,
    )

    plan = K31_LOW_M_PLANS["tep8"]
    for m in range(lo, hi + 1):
        for i, (x, w) in enumerate(bit_battery(m, n, k, seed=m * 977 + n, count=BATTERY)):
            reference = k31_fused_a_gemm(x, w.t()) if planless == "fused_a" else F.linear(x, w)
            out = op(x, w, plan, True)
            assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), (
                f"{name}: {low_m_choice(plan, x, w)} at {m} rows, battery input {i}"
            )


# The TEP8 target plan's shapes: KDA [q|k|v|g], MLA g_proj / shared-expert gate_up, packed DSA hidden projection, MoE
# latent down, MLA q_b, indexer wq_b.
TEP8_TARGET_SHAPES = {
    (6144, 7168),
    (1536, 7168),
    (3808, 7168),
    (3584, 7168),
    (2304, 1536),
    (4096, 1536),
}


def test_tep8_target_plan_has_no_drafter_rows() -> None:
    """The TEP8 target's lookup reads its own table only: the drafter's rows, including those for the shapes both run
    (the MoE latent down / drafter gate-up, the MLA q_b), live in the drafter's table, which differs from it there."""
    target, draft = K31_LOW_M_PLANS["tep8"], K31_LOW_M_PLANS["tep8_draft"]
    assert set(target) == TEP8_TARGET_SHAPES
    assert k31_low_m_plan(8, False) is target
    assert k31_low_m_draft_plan(8, False) is draft
    assert draft is not target
    for tp, dp in ((16, False), (32, False), (16, True)):
        assert k31_low_m_draft_plan(tp, dp) is k31_low_m_plan(tp, dp)
