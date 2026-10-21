# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m3_draft_lowm entry: the Eagle3 drafter's BF16 linears on FlashInfer's fixed-tactic low-M
kernels (direct SIMT and split-K tensor-core), at the TP4 drafter shards, at the rows its table certifies.

Every cell is held to the fp64 reference within the derived band. The drafter's projections (``M3Eagle3Draft._linear``
over ``ops/m3_draft_lowm``) are held on the drafter's weight table: a covered row count is the entry's call bit for
bit, an uncovered one the plain GEMM's.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
    m3_draft_lowm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.gemm.m3_draft_lowm import (  # noqa: E501
    ROWS,
    TACTICS,
    tactic_for,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.gemm.m3_draft_lowm import (  # noqa: E501
    m3_draft_lowm as op,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_draft_lowm requires a CUDA device"

# Element band, in BF16 ulps of the row scale, and the relative RMS (``assert_within_ulp``).
#
# Kernel and reference multiply the same BF16 operands (every product exact in fp32); the kernel sums in fp32 (the
# direct kernel one dot product in K order, the split-K kernel tensor-core blocks and then its split partials), the
# reference in fp64, and both round once to BF16. With w ~ N(0, 1 / K) and x ~ N(0, 1) the outputs are O(1) and the
# fp32 order difference is ~sqrt(K) * 2^-24 of the summed magnitudes, a few 1e-6 at K = 18432: far below half a BF16
# ulp, so the two roundings disagree by at most one ulp of the element -- up to two units of the row scale for an
# element in its row maximum's binade -- and only where the sum sits that close to a rounding boundary, a few in ten
# thousand elements, hence 0.1 ulp RMS. The same derivation as the decode GEMV's band (test_modeling_v2_m3_mx_gemv),
# with no slack under it: a second ulp would mean a wrong sum.
ELEMENT_ULP = 2.0
RMS_ULP = 0.1

# (N, K) -> the drafter's weight key (M3Eagle3Draft.w).
DRAFTER_LINEARS = {
    (2304, 12288): "qkv",
    (6144, 2048): "o",
    (9216, 6144): "gu",
    (6144, 4608): "dn",
    (6144, 18432): "fc",
}


@pytest.fixture(autouse=True)
def _policy():
    dispatch.set_require(False)
    dispatch.reset_logged()
    yield
    dispatch.set_require(False)
    dispatch.reset_logged()


def build(n: int, k: int, m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / k**0.5).to(torch.bfloat16)
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    return x, w


def test_the_table_is_the_tp4_drafter() -> None:
    assert set(TACTICS) == set(DRAFTER_LINEARS)
    assert {c.spec["nk"] for c in op.CELLS} == set(TACTICS)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    (n, k), m = cell.spec["nk"], cell.spec["m"]
    x, w = build(n, k, m, seed=zlib.crc32(cell.why.encode()) % 2**31)
    kind, tactic = tactic_for(n, k, m)
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    args = (
        x,
        w,
        out,
        kind,
        tactic,
        True,
    )  # the target launches with PDL when the environment enables it
    certifying(op)
    assert op(*args) is out
    ref = op.reference(*args)
    assert out.shape == (m, n) and out.dtype is torch.bfloat16
    assert_within_ulp(out, ref, element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)
    # PDL only changes the launch attribute.
    plain = torch.empty_like(out)
    op(x, w, plain, kind, tactic, False)
    assert torch.equal(plain.view(torch.int16), out.view(torch.int16))


def _draft(linears=DRAFTER_LINEARS):
    """A drafter holding only its projection weights (the module is not built)."""
    draft = object.__new__(dispatch.M3Eagle3Draft)
    draft.w = {key: build(*nk, 1, seed=nk[0] + nk[1])[1] for nk, key in linears.items()}
    draft._lowm_keys = frozenset()
    return draft


@pytest.fixture(scope="module")
def prebuilt():
    draft = _draft()
    draft.prebuild_linears()  # 0 when an earlier test in this process compiled them
    assert draft._lowm_keys == frozenset(DRAFTER_LINEARS.values())
    assert draft.prebuild_linears() == 0, "a second prebuild finds nothing left to compile"
    return draft


@pytest.mark.parametrize("nk", sorted(DRAFTER_LINEARS), ids=lambda nk: DRAFTER_LINEARS[nk])
def test_the_projection_is_the_entry_and_falls_back_past_its_rows(prebuilt, nk) -> None:
    n, k = nk
    key = DRAFTER_LINEARS[nk]
    w = prebuilt.w[key]
    use_pdl = get_env_enable_pdl()
    dispatch.set_require(True)
    for m in ROWS[nk]:
        assert dispatch.DRAFT_LOWM.covers((nk, m))
        x, _ = build(n, k, m, seed=m)
        got = prebuilt._linear(x, key)
        kind, tactic = tactic_for(n, k, m)
        want = op(x, w, torch.empty(m, n, dtype=x.dtype, device="cuda"), kind, tactic, use_pdl)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), m
    # Rows the table does not certify (past its rows): the plain GEMM, silently, even under require.
    for m in sorted(set(range(1, dispatch.DECODE_MAX_TOKENS + 2)) - set(ROWS[nk])):
        assert not dispatch.DRAFT_LOWM.covers((nk, m))
        x, _ = build(n, k, m, seed=m)
        got = prebuilt._linear(x, key)
        assert torch.equal(got.view(torch.int16), F.linear(x, w).view(torch.int16)), m


def test_a_projection_outside_the_table_is_the_plain_gemm() -> None:
    assert not m3_draft_lowm.installed(torch.empty(256, 512, dtype=torch.bfloat16))
    draft = _draft({(256, 512): "o"})
    draft._lowm_keys = frozenset(k for k in ("o",) if m3_draft_lowm.installed(draft.w[k]))
    x, _ = build(256, 512, 1, seed=3)
    got = draft._linear(x, "o")
    assert torch.equal(got.view(torch.int16), F.linear(x, draft.w["o"]).view(torch.int16))


def test_a_covered_call_not_prebuilt_raises_under_require(monkeypatch) -> None:
    monkeypatch.setattr(m3_draft_lowm, "_BUILT", set())
    monkeypatch.setattr(m3_draft_lowm, "_KEYS", frozenset(dispatch.DRAFT_LOWM.keys))
    w = build(6144, 2048, 1, seed=4)[1]
    x, _ = build(6144, 2048, 1, seed=5)
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="not compiled before warmup"):
        m3_draft_lowm.draft_linear(w, x, None)
    # Outside require the drafter falls back to the plain GEMM.
    dispatch.set_require(False)
    assert m3_draft_lowm.draft_linear(w, x, None) is None


def test_guard_refuses_a_strided_activation() -> None:
    x, w = build(6144, 2048, 2, seed=6)
    wide = torch.zeros(2, 4096, dtype=torch.bfloat16, device="cuda")
    wide[:, :2048] = x
    kind, tactic = tactic_for(6144, 2048, 2)
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(
                wide[:, :2048],
                w,
                torch.empty(2, 6144, dtype=torch.bfloat16, device="cuda"),
                kind,
                tactic,
                False,
            )
