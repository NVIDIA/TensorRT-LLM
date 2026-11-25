# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's `csa2_bf16_lowm_proj` (bf16 projections of decode-sized steps on the low-M kernel, as the engine's
CSA2 `bf16_projection`), which the dep and tep8 targets use for the indexer's head weights and the compressor.

Band: both paths accumulate the bf16 operands in fp32 and round once to bf16, so against the float64 product rounded
to bf16 they differ by at most one bf16 step of an element (element_ulp 2 in units of the row max's top-of-binade
spacing) and well under one ulp RMS; the scaled form multiplies that bf16 value in fp32 exactly as the reference.
"""

import contextlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # _v41_util
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # _validating
from _v41_util import DEV, gen_for  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_bf16_lowm_proj import (  # noqa: E402,E501
    csa2_bf16_lowm_proj,
)
from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)


@pytest.mark.parametrize(
    "cell", csa2_bf16_lowm_proj.CELLS, ids=[c.why[:48] for c in csa2_bf16_lowm_proj.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_bf16_lowm_proj)
    s, gen = cell.spec, gen_for(cell)
    x = torch.randn(s["m"], s["k"], generator=gen, device=DEV).bfloat16()
    w = (torch.randn(s["n"], s["k"], generator=gen, device=DEV) * 0.02).bfloat16()
    scale = 128**-0.5 * 32**-0.5 if s["scale"] else None
    kw = {"out_dtype": s["out_dtype"]} if "out_dtype" in s else {}
    with (
        flashinfer_mxfp8_decode_graph_capture() if s.get("graph") else contextlib.nullcontext()
    ):  # a decode-graph step
        got = csa2_bf16_lowm_proj(x, w, scale, **kw)
        fp32_form = csa2_bf16_lowm_proj(x, w, scale) if kw else None
    ref = csa2_bf16_lowm_proj.reference(x, w, scale, **kw)
    assert got.shape == ref.shape and got.dtype == (
        kw.get("out_dtype", torch.float32) if scale else torch.bfloat16
    )
    assert_within_ulp(
        got.float(), ref.float(), element_ulp=2.0, rms_ulp=0.5, ulp=2**-8
    )  # bf16 steps, also for the fp32 form
    if (
        kw
    ):  # tep8's candidate consumers: one launch into bf16, the fp32 form cast to bf16, bit for bit
        assert torch.equal(got, fp32_form.to(torch.bfloat16))


def test_scaled_form_is_the_scaled_projection():
    """The fp32 form is `(y * scale)` of the bf16 form, bit for bit (the targets' former `cublas_mm` + `torch.mul`)."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    for m in (24, 160):
        x = torch.randn(m, 5120, generator=gen, device=DEV).bfloat16()
        w = (torch.randn(32, 5120, generator=gen, device=DEV) * 0.02).bfloat16()
        scale = 128**-0.5 * 32**-0.5
        assert torch.equal(
            csa2_bf16_lowm_proj(x, w, scale), csa2_bf16_lowm_proj(x, w).float() * scale
        )


def test_low_m_only_in_a_decode_graph(monkeypatch):
    """24 rows: the low-M kernel under the decode-graph flag, `F.linear` outside it (the engine's `bf16_projection` gate
    and its `linear(x)`), bit for bit; a spy on the low-M kernel checks which path ran."""
    import tensorrt_llm._torch.modules.low_m_gemm as low_m_gemm

    calls = []
    lowm_linear = low_m_gemm.bf16_lowm_linear

    def spy(*args, **kwargs):
        calls.append(args[0].shape[0])
        return lowm_linear(*args, **kwargs)

    monkeypatch.setattr(low_m_gemm, "bf16_lowm_linear", spy)
    gen = torch.Generator(device=DEV).manual_seed(7)
    x = torch.randn(24, 5120, generator=gen, device=DEV).bfloat16()
    w = (torch.randn(1024, 5120, generator=gen, device=DEV) * 0.02).bfloat16()
    assert torch.equal(csa2_bf16_lowm_proj(x, w), torch.nn.functional.linear(x, w))
    assert calls == [], "outside a decode graph the low-M kernel must not run"
    with flashinfer_mxfp8_decode_graph_capture():
        lowm = csa2_bf16_lowm_proj(x, w)
    assert calls == [24], "in a decode-graph step the low-M kernel runs once"
    assert_within_ulp(
        lowm.float(),
        csa2_bf16_lowm_proj.reference(x, w).float(),
        element_ulp=2.0,
        rms_ulp=0.5,
        ulp=2**-8,
    )


@pytest.mark.parametrize("graph", [False, True])
def test_cuda_graph(graph):
    with flashinfer_mxfp8_decode_graph_capture() if graph else contextlib.nullcontext():
        _graph()


def _graph():
    gen = torch.Generator(device=DEV).manual_seed(3)
    x = torch.randn(24, 5120, generator=gen, device=DEV).bfloat16()
    w = (torch.randn(1024, 5120, generator=gen, device=DEV) * 0.02).bfloat16()
    csa2_bf16_lowm_proj(x, w)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = csa2_bf16_lowm_proj(x, w)
    x.copy_(torch.randn(24, 5120, generator=gen, device=DEV))
    g.replay()
    assert torch.equal(out, csa2_bf16_lowm_proj(x, w))


def test_rejections():
    x = torch.zeros(6, 64, dtype=torch.bfloat16, device=DEV)
    csa2_bf16_lowm_proj.is_valid(x, torch.zeros(8, 64, dtype=torch.bfloat16, device=DEV))
    with pytest.raises(AssertionError):
        csa2_bf16_lowm_proj.is_valid(x, torch.zeros(8, 64, device=DEV))
    with pytest.raises(AssertionError):
        csa2_bf16_lowm_proj.is_valid(x, torch.zeros(64, 8, dtype=torch.bfloat16, device=DEV).t())


@pytest.mark.parametrize("m", [65, 2246, 8996])
def test_large_steps_are_the_engines_linear(m):
    """Past the low-M rows (a context chunk; 8996 rows: the 9k-token prompt whose index weights differed from stock in
    tiles of rows): the engine's `linear(x)` (`F.linear`) bit for bit."""
    gen = torch.Generator(device=DEV).manual_seed(m)
    x = torch.randn(m, 5120, generator=gen, device=DEV).bfloat16()
    for n in (32, 512, 1024):
        w = (torch.randn(n, 5120, generator=gen, device=DEV) * 0.02).bfloat16()
        assert torch.equal(csa2_bf16_lowm_proj(x, w), torch.nn.functional.linear(x, w)), (m, n)
