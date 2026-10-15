# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m3_draft_lowm entry: the Eagle3 drafter's BF16 linears on FlashInfer's fixed-tactic low-M
kernels (direct SIMT and split-K tensor-core), at the TP8 drafter shards.

Every cell is held to the fp64 reference within the derived band. The target's linear method
(``kernels/m3_draft_lowm.DraftLowMLinearMethod``) is held on a real ``Linear`` (BF16, no bias): a covered row count is
the entry's call bit for bit, an uncovered one the in-tree method's.
"""

import zlib

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8 import (  # noqa: E501
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.kernels import (  # noqa: E501
    m3_draft_lowm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm.m3_draft_lowm import (  # noqa: E501
    TACTICS,
    tactic_for,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm.m3_draft_lowm import (  # noqa: E501
    m3_draft_lowm as op,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod

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

DRAFTER_LINEARS = {
    (1280, 12288): "qkv",
    (6144, 1024): "o_proj",
    (4608, 6144): "gate_up",
    (6144, 2304): "down",
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


def test_the_table_is_the_tp8_drafter() -> None:
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


def _linear(nk, seed: int, bias: bool = False) -> Linear:
    n, k = nk
    linear = Linear(k, n, bias=bias, dtype=torch.bfloat16).cuda()
    _, w = build(n, k, 1, seed)
    linear.weight.data.copy_(w)
    if bias:
        linear.bias.data.zero_()
    return linear


class _Drafter(nn.Module):
    """A module tree holding drafter linears, as ``install`` / ``prebuild`` walk it."""

    def __init__(self, *linears: Linear):
        super().__init__()
        self.linears = nn.ModuleList(linears)


@pytest.mark.parametrize("nk", sorted(DRAFTER_LINEARS), ids=lambda nk: DRAFTER_LINEARS[nk])
def test_the_linear_method_is_the_entry_and_falls_back_past_its_rows(nk) -> None:
    n, k = nk
    linear = _linear(nk, seed=n + k)
    drafter = _Drafter(linear)
    assert m3_draft_lowm.install(drafter) == 1
    assert isinstance(linear.quant_method, m3_draft_lowm.DraftLowMLinearMethod)
    m3_draft_lowm.prebuild(drafter)
    use_pdl = get_env_enable_pdl()
    last = TACTICS[nk][-1][0]
    dispatch.set_require(True)
    for m in range(1, last + 1):
        assert dispatch.DRAFT_LOWM.covers((nk, m))
        x, _ = build(n, k, m, seed=m)
        got = linear.quant_method.apply(linear, x, None)
        kind, tactic = tactic_for(n, k, m)
        want = op(
            x, linear.weight, torch.empty(m, n, dtype=x.dtype, device="cuda"), kind, tactic, use_pdl
        )
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), m
        assert torch.equal(linear(x).view(torch.int16), want.view(torch.int16)), m
    # Past the table: the in-tree method, silently, even under require.
    for m in (last + 1, 2 * last + 3):
        assert not dispatch.DRAFT_LOWM.covers((nk, m))
        x, _ = build(n, k, m, seed=m)
        got = linear.quant_method.apply(linear, x, None)
        stock = UnquantizedLinearMethod().apply(linear, x, None)
        assert torch.equal(got.view(torch.int16), stock.view(torch.int16)), m


def test_install_takes_only_covered_bias_free_bf16_linears() -> None:
    covered = _linear((6144, 1024), seed=1)
    biased = _linear((6144, 1024), seed=2, bias=True)
    other = _linear((256, 512), seed=3)
    drafter = _Drafter(covered, biased, other)
    assert m3_draft_lowm.install(drafter) == 1
    assert isinstance(covered.quant_method, m3_draft_lowm.DraftLowMLinearMethod)
    assert type(biased.quant_method) is UnquantizedLinearMethod
    assert type(other.quant_method) is UnquantizedLinearMethod
    # A second install finds nothing left to take.
    assert m3_draft_lowm.install(drafter) == 0


def test_a_covered_call_not_prebuilt_raises_under_require(monkeypatch) -> None:
    monkeypatch.setattr(m3_draft_lowm, "_BUILT", set())
    linear = _linear((6144, 1024), seed=4)
    m3_draft_lowm.install(_Drafter(linear))
    x, _ = build(6144, 1024, 1, seed=5)
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="not compiled before warmup"):
        linear.quant_method.apply(linear, x, None)
    # Outside require it falls back to the in-tree method.
    dispatch.set_require(False)
    got = linear.quant_method.apply(linear, x, None)
    stock = UnquantizedLinearMethod().apply(linear, x, None)
    assert torch.equal(got.view(torch.int16), stock.view(torch.int16))


def test_guard_refuses_a_strided_activation() -> None:
    x, w = build(6144, 1024, 2, seed=6)
    wide = torch.zeros(2, 2048, dtype=torch.bfloat16, device="cuda")
    wide[:, :1024] = x
    kind, tactic = tactic_for(6144, 1024, 2)
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(
                wide[:, :1024],
                w,
                torch.empty(2, 6144, dtype=torch.bfloat16, device="cuda"),
                kind,
                tactic,
                False,
            )
