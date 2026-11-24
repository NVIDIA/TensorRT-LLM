# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the family's ``rubin_*`` GEMM entries: references, guards, cells, tuning key.

No GPU and no kernel: the entries import the kernels on first call, so the references and the
guards run here on small shapes.
"""

import ast
import inspect
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm import (
    rubin_l2_prefetch,
    rubin_mxfp8_gemm,
    rubin_mxfp8_linear,
)

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7),
        reason="K3.1 Rubin (sm_107) entries; the CPU stage runs this file",
    ),
]

G = rubin_mxfp8_gemm
_ENTRIES = (
    G.rubin_mxfp8_gemm,
    rubin_mxfp8_linear.rubin_mxfp8_linear,
    rubin_l2_prefetch.rubin_l2_prefetch,
)


def test_quantization_reference_scales_and_bound() -> None:
    gen = torch.Generator().manual_seed(1)
    x = (torch.randn(5, 256, generator=gen) * torch.logspace(-3, 3, 256)).to(torch.bfloat16)
    x[2, 32:64] = 0.0  # an all-zero block takes scale byte 0 and zero values
    values, scales = G.quantize_mxfp8_reference(x)
    assert values.dtype == torch.float8_e4m3fn and scales.shape == (5, 8)
    amax = x.float().view(5, 8, 32).abs().amax(-1)
    scale = torch.exp2(scales.float() - 127.0)
    nonzero = amax > 0
    # The smallest power of two at or above amax / 448.
    assert torch.all(scale[nonzero] >= amax[nonzero] / 448.0)
    assert torch.all(scale[nonzero] / 2 < amax[nonzero] / 448.0)
    assert scales[2, 1] == 0 and torch.all(values.float()[2, 32:64] == 0)
    # E4M3 keeps 3 mantissa bits: at most 2^-4 relative error on each value.
    deq = G.dequantize_mxfp8(values, G.mxfp8_scales_to_r128c4(scales))
    err = (deq - x.float()).abs().view(5, 8, 32)
    assert torch.all(err <= x.float().abs().view(5, 8, 32) * 2.0**-4 + scale.unsqueeze(-1) * 2**-9)


@pytest.mark.parametrize("m, n, k", [(1, 8, 32), (5, 96, 64), (130, 24, 256)])
def test_gemm_reference_is_the_dequantized_product(m, n, k) -> None:
    gen = torch.Generator().manual_seed(m + n + k)
    a, a_s = G.quantize_mxfp8_reference(torch.randn(m, k, generator=gen).to(torch.bfloat16))
    b, b_s = G.quantize_mxfp8_reference(torch.randn(n, k, generator=gen).to(torch.bfloat16))
    out = G.rubin_mxfp8_gemm.reference(
        a, b, G.mxfp8_scales_to_r128c4(a_s), G.mxfp8_scales_to_r128c4(b_s)
    )
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    # Independently: float64 values times 2^(byte - 127), row-major scales, no R128c4.
    da = a.double().view(m, k // 32, 32) * torch.exp2(a_s.double() - 127.0).unsqueeze(-1)
    db = b.double().view(n, k // 32, 32) * torch.exp2(b_s.double() - 127.0).unsqueeze(-1)
    exact = da.view(m, k) @ db.view(n, k).t()
    # One BF16 rounding (at most 2^-9 of the value) on an FP32 sum: within 2^-8 of the row's
    # largest magnitude.
    row_max = exact.abs().amax(dim=1, keepdim=True)
    assert torch.all((out.double() - exact).abs() <= row_max * 2.0**-8)


def test_linear_reference_is_quantize_then_gemm() -> None:
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(9, 128, generator=gen).to(torch.bfloat16)
    w, w_s = G.quantize_mxfp8_reference(torch.randn(16, 128, generator=gen).to(torch.bfloat16))
    w_r = G.mxfp8_scales_to_r128c4(w_s)
    out = rubin_mxfp8_linear.rubin_mxfp8_linear.reference(x, w, w_r)
    a, a_s = G.quantize_mxfp8_reference(x)
    gemm = G.rubin_mxfp8_gemm.reference(a, w, G.mxfp8_scales_to_r128c4(a_s), w_r)
    assert torch.equal(out, gemm)
    # And it tracks the unquantized product within MXFP8's rounding.
    dense = x.float() @ G.dequantize_mxfp8(w, w_r).t()
    assert (out.float() - dense).norm() / dense.norm() < 2.0**-4


def test_guards_refuse_on_cpu_tensors() -> None:
    gen = torch.Generator().manual_seed(3)
    a, a_s = G.quantize_mxfp8_reference(torch.randn(4, 128, generator=gen).to(torch.bfloat16))
    b, b_s = G.quantize_mxfp8_reference(torch.randn(16, 128, generator=gen).to(torch.bfloat16))
    a_r, b_r = G.mxfp8_scales_to_r128c4(a_s), G.mxfp8_scales_to_r128c4(b_s)
    gemm = G.rubin_mxfp8_gemm
    with pytest.raises(AssertionError, match="E4M3"):
        gemm.is_valid(a.float().to(torch.bfloat16), b, a_r, b_r)
    with pytest.raises(AssertionError, match="row-major"):
        gemm.is_valid(a, b, a_r, b_s)
    with pytest.raises(AssertionError, match="multiple of 8"):
        gemm.is_valid(a, b[:12].contiguous(), a_r, b_r)
    with pytest.raises(AssertionError, match="32-value scale blocks"):
        gemm.is_valid(a[:, :112].contiguous(), b[:, :112].contiguous(), a_r, b_r)
    with pytest.raises(AssertionError, match="contiguous"):
        gemm.is_valid(a, b.t().contiguous().t(), a_r, b_r)
    linear = rubin_mxfp8_linear.rubin_mxfp8_linear
    x = torch.randn(4, 128, generator=gen).to(torch.bfloat16)
    with pytest.raises(AssertionError, match="BF16"):
        linear.is_valid(x.float(), b, b_r)
    with pytest.raises(AssertionError, match="row-major"):
        linear.is_valid(x, b, b_s)
    with pytest.raises(AssertionError, match="CUDA device"):
        linear.is_valid(x, b, b_r)  # well formed, but on the CPU


def test_l2_prefetch_reference_changes_nothing() -> None:
    op = rubin_l2_prefetch.rubin_l2_prefetch
    assert op.reference(((0x1000, 4096),), max_ctas=8) is None


def test_alpha_is_made_outside_graph_capture(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before CUDA graph capture"):
        G._alpha(torch.device("cuda", 31415))


@pytest.mark.parametrize("entry", _ENTRIES, ids=lambda e: type(e).__name__)
def test_entries_have_cells_with_reasons(entry) -> None:
    assert entry.CELLS and all(cell.why.strip() for cell in entry.CELLS)
    specs = [repr(sorted(cell.spec.items())) for cell in entry.CELLS]
    assert len(set(specs)) == len(specs), "a repeated cell certifies nothing new"
    assert inspect.signature(type(entry).reference).parameters.keys() == (
        inspect.signature(type(entry).__call__).parameters.keys()
    )


def test_gemm_cells_meet_the_guard() -> None:
    for entry in (G.rubin_mxfp8_gemm, rubin_mxfp8_linear.rubin_mxfp8_linear):
        for cell in entry.CELLS:
            assert cell.spec["k"] % 128 == 0 and cell.spec["n"] % 8 == 0, cell.why


def test_the_tuning_key_and_runner_are_the_familys() -> None:
    """The family tunes under its own key with its own runner class, so no autotuner entry
    is shared with the engine's MXFP8 GEMM (key ``trtllm::cute_dsl_mxfp8_gemm_rubin``)."""
    assert G.TUNING_KEY.startswith("kimi_k31::rubin_")
    source = Path(G.__file__).read_text()
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "choose_one"
    )
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == "TUNING_KEY"
    assert "RubinMxfp8GemmRunner(" in source
    engine = Path(G.__file__).resolve().parents[6] / "custom_ops" / "cute_dsl_custom_ops.py"
    text = engine.read_text()
    assert G.TUNING_KEY not in text and "RubinMxfp8GemmRunner" not in text
