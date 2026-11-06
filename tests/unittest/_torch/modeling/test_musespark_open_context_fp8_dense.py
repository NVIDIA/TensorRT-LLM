# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``context_fp8_dense``: the MXFP8 quant against the built-in op, the copy's GEMM against BF16,
and which calls take it (sm107)."""

import importlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod

_PACKAGE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_fp8_dense"
)

_ON_RUBIN = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (10, 7)
    and IS_CUTLASS_DSL_RUBIN_AVAILABLE
)
needs_rubin = pytest.mark.skipif(not _ON_RUBIN, reason="needs sm107 with the Rubin CuTe DSL")


def _install():
    return importlib.import_module(f"{_PACKAGE}.install")


def _cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def test_lever_parameters_are_the_installers():
    """The lever table's ``context_fp8_dense`` parameters are the ones ``install`` takes."""
    install = _install()
    levers = importlib.import_module(_PACKAGE.rsplit(".", 2)[0] + ".levers")
    rule = levers.RULES["context_fp8_dense"]
    assert rule.param("PROJECTIONS").choices == install.PROJECTIONS
    assert rule.param("PROJECTIONS").default == install.PROJECTIONS
    assert rule.param("MIN_ROWS").default == install.DEFAULT_MIN_ROWS


def test_tactic_table_is_ascending_and_unswapped():
    install = _install()
    for shape, entries in install.TACTICS.items():
        rows = [fewest for fewest, _ in entries]
        assert rows == sorted(rows) and rows[0] == 0, shape
        for _, tactic in entries:
            swap = tactic[4] if tactic[0] == "base" else tactic[5]
            assert swap is False, (shape, tactic)


@needs_rubin
@pytest.mark.parametrize("m", [1, 100, 128, 1536, 4100, 8192])
@pytest.mark.parametrize("k", [2048, 4096, 8192])
def test_quant_matches_builtin_op(m, k):
    quant = importlib.import_module(f"{_PACKAGE}.quant")
    torch.manual_seed(m + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    x *= 1 + 7 * torch.rand(m, 1, device="cuda", dtype=torch.bfloat16)
    x[0, :64] = 0  # an all-zero block
    values, scale = quant.mxfp8_quantize(x)
    ref_values, ref_scale = torch.ops.trtllm.mxfp8_quantize(x, True)
    assert torch.equal(values.view(torch.uint8), ref_values.view(torch.uint8))
    rows_padded = -(-m // 128) * 128
    assert scale.numel() == ref_scale.numel() == rows_padded * (k // 32)
    assert torch.equal(scale, ref_scale)


@needs_rubin
@pytest.mark.parametrize(
    "n, k, m",
    [
        (4608, 8192, 8192),
        (4608, 8192, 6500),
        (8192, 2048, 4100),
        (4096, 8192, 2048),
        (8192, 4096, 1536),
        (8192, 4096, 1025),
    ],
)
def test_projection_matches_bf16_at_every_listed_tactic(n, k, m):
    install = _install()
    torch.manual_seed(0)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    copy = install.Fp8Projection(weight, min_rows=1024)
    ref = F.linear(x, weight)
    outputs = [copy(x, tactic) for _, tactic in copy.tactics]
    for out in outputs:
        assert out.shape == ref.shape and out.dtype == torch.bfloat16
        assert _cos_diff(out, ref) < 2e-3
    chosen = copy(x)
    assert _cos_diff(chosen, ref) < 2e-3


@needs_rubin
def test_method_dispatches_by_rows_and_captures():
    install = _install()
    torch.manual_seed(0)
    linear = Linear(2048, 8192, bias=False, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight) / 2048**0.5)
    assert type(linear.quant_method) is UnquantizedLinearMethod
    copy = install.Fp8Projection(linear.weight.data, min_rows=1024)
    method = install.ContextFp8LinearMethod.__new__(install.ContextFp8LinearMethod)
    method.__dict__.update(linear.quant_method.__dict__)
    linear.quant_method = method
    linear._context_fp8 = copy
    assert isinstance(linear.quant_method, UnquantizedLinearMethod)

    small = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    # Below min_rows: the BF16 GEMM, bit for bit.
    assert torch.equal(linear(small), F.linear(small, linear.weight))

    large = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
    eager = linear(large)
    assert torch.equal(eager, copy(large))
    assert not torch.equal(eager, F.linear(large, linear.weight))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = linear(large)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


@needs_rubin
@pytest.mark.parametrize("rows", [1024, 1300, 4096])
def test_gather_quantized_before_matches_quantized_after(rows):
    """Quantizing the rows before the gather (row-major scales, swizzled after) feeds the GEMM
    the same bytes as quantizing the gathered rows; below min_rows the BF16 rows pass through."""
    from tensorrt_llm.mapping import Mapping

    install = _install()
    quant = importlib.import_module(f"{_PACKAGE}.quant")
    torch.manual_seed(rows)
    k = 8192
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    gather = install.Fp8AttentionInputGather(
        Mapping(world_size=1, tp_size=1, rank=0), min_rows=1024
    )
    gathered = gather(x, rows)
    assert isinstance(gathered, install.GatheredMxfp8) and tuple(gathered.shape) == (rows, k)
    values, scale = quant.mxfp8_quantize(x)
    assert torch.equal(gathered.values.view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(gathered.scale, scale)

    weight = torch.randn(4608, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    linear = Linear(k, 4608, bias=False, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        linear.weight.copy_(weight)
    method = install.ContextFp8LinearMethod.__new__(install.ContextFp8LinearMethod)
    method.__dict__.update(linear.quant_method.__dict__)
    linear.quant_method = method
    linear._context_fp8 = install.Fp8Projection(linear.weight.data, min_rows=1024)
    assert torch.equal(linear(gathered), linear(x))

    small = x[:512]
    passed = gather(small, 512)
    assert isinstance(passed, torch.Tensor) and torch.equal(passed, small)
