# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused K128 quantization/GEMM numerical and graph replay coverage."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cute_dsl_fused_fp8_block_scale_linear import (
    cute_dsl_fused_fp8_block_scale_linear as op,
)

__extra_import_path__ = [".."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires VR200/SM107",
)


def _input(m, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    storage = torch.randn(m, k + 16, dtype=torch.bfloat16, device="cuda", generator=generator)
    view = storage[:, 8 : k + 8]
    exponents = torch.randint(-20, 21, (m, k // 128, 1), device="cuda", generator=generator)
    view.copy_(torch.ldexp(view.reshape(m, k // 128, 128).float(), exponents).reshape(m, k))
    view[0, :128] = 0
    return view


def _weights(n, k, seed):
    # The fused kernel reads weights before its PDL wait, so tests create them
    # before the activations instead of directly ahead of the GEMM.
    weight = _input(n, k, seed).contiguous()
    return torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(weight, True)


@pytest.mark.parametrize("m", (1, 3, 4, 8, 12, 16))
@pytest.mark.parametrize("k", (768, 1024, 1536, 2048, 3072, 6144))
@pytest.mark.parametrize("seed", (11, 23, 47))
def test_fused_matches_k128_reference(m, k, seed):
    weight, scale = _weights(128, k, seed + 1)
    x = _input(m, k, seed)
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.fused_fp8_quant_gemv import fused_fp8_quant_gemv

    actual = fused_fp8_quant_gemv(x, weight, scale)
    expected = op.reference(x, weight, scale)
    assert_within_ulp(actual, expected, element_ulp=4, rms_ulp=2)


@pytest.mark.parametrize("m,k", ((4, 768), (16, 1536), (4, 6144), (16, 6144)))
def test_fused_changing_input_graph(m, k):
    weight, scale = _weights(128, k, 23)
    x = _input(m, k, 11)
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.fused_fp8_quant_gemv import fused_fp8_quant_gemv

    for _ in range(3):
        fused_fp8_quant_gemv(x, weight, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fused_fp8_quant_gemv(x, weight, scale)
    for seed in range(101):
        x.copy_(_input(m, k, seed))
        graph.replay()
        expected = op.reference(x, weight, scale)
        assert_within_ulp(output, expected, element_ulp=4, rms_ulp=2)


@pytest.mark.parametrize("m", (1, 3, 4, 8, 12, 16))
@pytest.mark.parametrize("k", (768, 1024, 1536, 2048, 3072, 6144))
@pytest.mark.parametrize("seed", (11, 23, 47))
def test_fused_quantization_byte_contract(m, k, seed):
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.fused_fp8_quant_gemv import debug_quantization

    x = _input(m, k, seed)
    codes, scale = debug_quantization(x)
    expected_codes, physical = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(
        x.contiguous(), True
    )
    logical = (
        physical.reshape(-1, k // 128, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(-1, k // 32)[:m, ::4]
    )
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    assert torch.equal(scale, logical)


def test_fused_quantization_finite_extremes():
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.fused_fp8_quant_gemv import debug_quantization

    values = torch.tensor(
        [
            0.0,
            1.0e-40,
            -1.0e-40,
            torch.finfo(torch.bfloat16).max,
            -torch.finfo(torch.bfloat16).max,
            7.0,
            7.03125,
            6.96875,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = values.repeat(4 * 768 // values.numel()).reshape(4, 768)
    actual, scales = debug_quantization(x)
    expected, physical = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x, True)
    logical = physical.reshape(-1, 6, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(-1, 24)[:4, ::4]
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(scales, logical)


def test_fused_opposite_scale_exponents():
    # The dequantized operands and their output are finite even though applying
    # the activation scale alone to an unscaled K128 subtotal would overflow.
    weight = torch.full((128, 768), 448.0, device="cuda", dtype=torch.float8_e4m3fn)
    logical = torch.full((128, 24), 17, device="cuda", dtype=torch.uint8)
    scale = torch.ops.trtllm.block_scale_interleave(logical)
    x = torch.full((4, 768), 2.0**118, device="cuda", dtype=torch.bfloat16)
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.fused_fp8_quant_gemv import fused_fp8_quant_gemv

    actual = fused_fp8_quant_gemv(x, weight, scale)
    expected = op.reference(x, weight, scale)
    assert bool(torch.isfinite(actual).all())
    assert_within_ulp(actual, expected, element_ulp=4, rms_ulp=2)


@pytest.mark.parametrize("cell", op.CELLS, ids=[cell.why for cell in op.CELLS])
@pytest.mark.parametrize("draw", range(3))
def test_certified_cells(cell, draw, certifying) -> None:  # noqa: F811 — pytest fixture
    certifying(op)
    m, n, k = (cell.spec[name] for name in ("m", "n", "k"))
    seed = cell.seed_base() + 2 * draw
    weight, scale = _weights(n, k, seed + 1)
    x = _input(m, k, seed)
    actual = op(x, weight, scale)
    expected = op.reference(x, weight, scale)
    assert_within_ulp(actual, expected, element_ulp=4, rms_ulp=2)


@pytest.mark.parametrize("m", (64, 128, 129, 256, 2048, 16384))
def test_fused_catalog_protection(m):
    x = _input(m, 768, 11).contiguous()
    weight, scale = _weights(128, 768, 23)
    actual = op(x, weight, scale)
    expected = op.reference(x, weight, scale)
    assert_within_ulp(actual, expected, element_ulp=4, rms_ulp=2)


@pytest.mark.parametrize("n,k", ((1024, 6144), (1536, 6144), (6144, 768), (6144, 1024)))
def test_fused_catalog_changing_input_graph(n, k):
    weight, scale = _weights(n, k, 23)
    x = _input(4, k, 11).contiguous()
    for _ in range(3):
        op(x, weight, scale)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(x, weight, scale)
    for seed in range(101):
        x.copy_(_input(4, k, seed))
        graph.replay()
        expected = op.reference(x, weight, scale)
        assert_within_ulp(output, expected, element_ulp=4, rms_ulp=2)


@pytest.mark.parametrize("tp_size", (4, 8))
def test_decode_target_dispatch(tp_size):
    from importlib import import_module
    from types import SimpleNamespace

    from torch.utils._python_dispatch import TorchDispatchMode

    modeling = import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4."
        "mistral_large_4_nvfp4__sm_107__tp4.modeling"
    )

    class Operations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.seen = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.seen.append(str(func))
            return func(*args, **(kwargs or {}))

    target = object.__new__(modeling.DecodeTarget)
    target.core = SimpleNamespace(tp_size=tp_size)
    weight, scale = _weights(1024, 6144, 23)
    x = _input(4, 6144, 11).contiguous()
    with Operations() as operations:
        actual = target._linear_fp8(x, weight, scale)
    fused = any("trtllm.cute_dsl_fused_fp8_block_scale_linear" in name for name in operations.seen)
    assert fused == (tp_size == 8), operations.seen
    if tp_size == 4:
        assert any("trtllm.cute_dsl_mxfp8_gemm_rubin" in name for name in operations.seen)
    expected = op.reference(x, weight, scale)
    assert_within_ulp(actual, expected, element_ulp=4, rms_ulp=2)
