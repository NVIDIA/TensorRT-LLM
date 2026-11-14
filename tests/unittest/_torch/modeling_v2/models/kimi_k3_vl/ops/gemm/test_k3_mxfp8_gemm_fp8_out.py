# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's SM107 MXFP8 GEMM with FP8 output (``ops/gemm/k3_mxfp8_gemm_fp8_out``): the epilogue's
BF16-to-E4M3 rounding equals main's BF16 GEMM followed by a unit-scale E4M3 quantization, bit for bit
(the Kimi K3 context kv_b expansion).

* host side (any GPU with the Rubin CuTe DSL): the kernel's ``can_implement`` for an E4M3 C, the untuned
  tactic for the kv_b shape and the runner's autotuner identity;
* SM107: fixed tactics (both launch interfaces, alphas including saturation and non-finite values), the
  untuned 2-CTA 256x256 tile with 128x128 epilogue subtiles, and the entry's ``CELLS`` through the
  AutoTuner.
"""

import importlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="needs a GPU and the CuTe DSL with Rubin support",
)

sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin MXFP8 GEMM runs on SM107",
)

#: The kv_b expansion: M x 512 -> 96 heads x (128 + 128).
KV_B_N, KV_B_K = 24576, 512
UNTUNED = ("base", (256, 256, 128), (256, 256, 64), (2, 1), False, False, "static", "m", 1, 128)


def _modules():
    import cutlass

    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops

    impl = importlib.import_module(f"{K3}.ops.gemm._mxfp8_gemm_fp8_out")
    entry = importlib.import_module(f"{K3}.ops.gemm.k3_mxfp8_gemm_fp8_out")
    return cutlass, cute_dsl_custom_ops, impl, entry


def _can(runner, c_dtype, n=KV_B_N, split_k=1, tile=(128, 128, 128), inst=(128, 128, 64)):
    return runner.kernel_class.can_implement(
        (8192, n, KV_B_K, 1),
        runner.a_dtype,
        runner.b_dtype,
        runner.sf_dtype,
        c_dtype,
        "k",
        "k",
        "n",
        runner.sf_vec_size,
        tile,
        inst,
        (1, 1),
        split_k,
    )


def test_host_side(monkeypatch):
    cutlass, ops, impl, _ = _modules()
    monkeypatch.delenv("TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER", raising=False)
    runner_cls = impl.K3MXFP8Fp8OutRubinRunner
    runner = runner_cls()
    assert runner.output_dtype == torch.float8_e4m3fn and runner.c_dtype == cutlass.Float8E4M3FN
    # The autotuner identity of main's FP8-output runner: E4M3 output, no userbuffers, TVM-FFI.
    assert runner.unique_id() == (torch.float8_e4m3fn, False, True)
    assert runner.unique_id() != ops.CuteDSLMXFP8RubinLinear(torch.bfloat16).unique_id()
    assert runner_cls.kernel_cache is not ops.CuteDSLMXFP8RubinLinear.kernel_cache
    assert runner_cls.split_k_candidates == (1,)
    # E4M3 C: MXFP8 operands, no split-K, 16-element rows; BF16 C keeps main's rules.
    assert _can(runner_cls, cutlass.Float8E4M3FN)
    assert _can(runner_cls, cutlass.BFloat16)
    assert not _can(runner_cls, cutlass.Float8E4M3FN, split_k=2)
    assert _can(runner_cls, cutlass.BFloat16, split_k=2)
    assert not _can(runner_cls, cutlass.Float8E4M3FN, n=KV_B_N + 8)
    assert _can(runner_cls, cutlass.BFloat16, n=KV_B_N + 8)
    assert not runner_cls.mixed_kernel_class.can_implement(
        (8192, KV_B_N + 8, KV_B_K, 1),
        runner_cls.a_dtype,
        runner_cls.b_dtype,
        runner_cls.sf_dtype,
        cutlass.Float8E4M3FN,
        "k",
        "k",
        "n",
        32,
        (256, 256, 128),
        (256, 256, 64),
        (4, 1),
        (2, 1),
    )
    # The untuned tactic of the kv_b shape: the 2-CTA 256x256 tile from 1024 rows.
    assert runner.untuned_default_tactic(8192, KV_B_N, KV_B_K) == UNTUNED
    assert runner.untuned_default_tactic(1024, KV_B_N, KV_B_K) == UNTUNED
    assert runner.untuned_default_tactic(1023, KV_B_N, KV_B_K) is None
    assert runner.untuned_default_tactic(8192, KV_B_N + 8, KV_B_K) is None


def _operands(m, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x)
    b, b_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(w)
    return a, b, a_sf, b_sf


def _bf16_then_quant(ops, inputs, tactic, use_tvm_ffi):
    """Main's BF16 GEMM, then the unit-scale E4M3 quantization."""
    bf16 = ops.CuteDSLMXFP8RubinLinear(torch.bfloat16, use_tvm_ffi=use_tvm_ffi)(
        inputs, tactic=tactic
    )
    out, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        bf16, torch.ones_like(inputs[4])
    )
    return out


@sm107
@pytest.mark.parametrize("m", [1, 124, 257])
@pytest.mark.parametrize("use_tvm_ffi", [False, True])
def test_fixed_tactic_matches_bf16_then_quant(m, use_tvm_ffi):
    _, ops, impl, _ = _modules()
    a, b, a_sf, b_sf = _operands(m, 256, 512, 107)
    alpha = torch.ones((), dtype=torch.float32, device="cuda")
    inputs = [a, b, a_sf, b_sf, alpha]
    tactic = ("base", (128, 128, 128), (128, 128, 64), (1, 1), False, False, "static", "m")
    candidate = impl.K3MXFP8Fp8OutRubinRunner(use_tvm_ffi=use_tvm_ffi)
    # Values above the E4M3 range exercise finite saturation, in addition to the ordinary
    # distribution and a non-unit GEMM alpha.
    scales = [1.0, 0.5, 1.3, 32.0]
    if m == 124 and use_tvm_ffi:
        scales += [float("inf"), float("-inf"), float("nan")]
    for scale in scales:
        alpha.fill_(scale)
        expected = _bf16_then_quant(ops, inputs, tactic, use_tvm_ffi)
        actual = candidate(inputs, tactic=tactic)
        assert actual.dtype == torch.float8_e4m3fn
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), scale


@sm107
def test_untuned_tile_matches_bf16_then_quant(monkeypatch):
    _, ops, impl, _ = _modules()
    monkeypatch.delenv("TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER", raising=False)
    a, b, a_sf, b_sf = _operands(2048, KV_B_N, KV_B_K, 108)
    alpha = torch.ones((), dtype=torch.float32, device="cuda")
    inputs = [a, b, a_sf, b_sf, alpha]
    runner = impl.K3MXFP8Fp8OutRubinRunner()
    assert runner.untuned_default_tactic(2048, KV_B_N, KV_B_K) == UNTUNED
    expected = _bf16_then_quant(ops, inputs, -1, True)
    for tactic in (-1, UNTUNED):
        actual = runner(inputs, tactic=tactic)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), tactic


@sm107
@pytest.mark.parametrize(
    "cell",
    importlib.import_module(f"{K3}.ops.gemm.k3_mxfp8_gemm_fp8_out").k3_mxfp8_gemm_fp8_out.CELLS
    if IS_CUTLASS_DSL_RUBIN_AVAILABLE
    else [],
    ids=lambda c: c.why[:48],
)
def test_k3_mxfp8_gemm_fp8_out_cells(cell):
    """The entry through the AutoTuner, against main's BF16 op and the unit-scale quantization."""
    _, _, _, entry = _modules()
    s = cell.spec
    a, b, a_sf, b_sf = _operands(s["m"], s["n"], s["k"], 109)
    actual = entry.k3_mxfp8_gemm_fp8_out(a, b, a_sf, b_sf)
    bf16 = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(a, b, a_sf, b_sf)
    expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        bf16, torch.ones((), dtype=torch.float32, device="cuda")
    )
    assert actual.shape == (s["m"], s["n"]) and actual.dtype == torch.float8_e4m3fn
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
