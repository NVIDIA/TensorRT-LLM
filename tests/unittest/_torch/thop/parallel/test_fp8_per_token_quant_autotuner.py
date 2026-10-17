# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import patch

import pytest
import torch


class TestVectorizedPerTokenFP8Quant(unittest.TestCase):
    """Test the vectorized FP8 per-token quantization kernel."""

    def setUp(self):
        import tensorrt_llm  # noqa: F401 — registers C++ ops

        torch.manual_seed(42)

    def test_output_shape_and_dtype_bf16(self):
        x = torch.randn(1024, 3072, dtype=torch.bfloat16, device="cuda")
        qx, scale = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x)

        self.assertEqual(qx.shape, x.shape)
        self.assertEqual(qx.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (1024, 1))
        self.assertEqual(scale.dtype, torch.float32)

    def test_output_shape_and_dtype_fp16(self):
        x = torch.randn(512, 4096, dtype=torch.float16, device="cuda")
        qx, scale = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x)

        self.assertEqual(qx.shape, x.shape)
        self.assertEqual(qx.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (512, 1))
        self.assertEqual(scale.dtype, torch.float32)

    def test_output_shape_and_dtype_fp32(self):
        x = torch.randn(256, 2048, dtype=torch.float32, device="cuda")
        qx, scale = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x)

        self.assertEqual(qx.shape, x.shape)
        self.assertEqual(qx.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (256, 1))
        self.assertEqual(scale.dtype, torch.float32)

    def test_scales_positive(self):
        x = torch.randn(512, 3072, dtype=torch.bfloat16, device="cuda")
        _, scale = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x)
        self.assertTrue((scale > 0).all())

    def test_scale_matches_trtllm_kernel(self):
        """Vectorized and TRT-LLM kernels must produce numerically equivalent scales."""
        x = torch.randn(1024, 3072, dtype=torch.bfloat16, device="cuda")

        _, scale_strided = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x)
        _, scale_trtllm = torch.ops.tensorrt_llm.quantize_e4m3_activation(x)

        # TRT-LLM returns bf16 scales; vectorized kernel returns f32.
        self.assertTrue(
            torch.allclose(
                scale_strided.squeeze(-1),
                scale_trtllm.squeeze(-1).float(),
                rtol=1e-2,
                atol=1e-5,
            )
        )

    def test_3d_input(self):
        """Kernel must handle inputs with batch dim, e.g. [batch, seq, hidden]."""
        x = torch.randn(4, 256, 3072, dtype=torch.bfloat16, device="cuda")
        qx, scale = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x)

        self.assertEqual(qx.shape, x.shape)
        self.assertEqual(scale.shape, (4, 256, 1))


class TestTunableFP8PerTokenQuant(unittest.TestCase):
    """Test selection between original, vectorized, and warp quantization."""

    def setUp(self):
        import tensorrt_llm  # noqa: F401

        torch.manual_seed(42)

    def test_output_shape_and_dtype(self):
        x = torch.randn(1024, 3072, dtype=torch.bfloat16, device="cuda")
        qx, scale = torch.ops.trtllm.tunable_fp8_per_token_quant(x)

        self.assertEqual(qx.shape, x.shape)
        self.assertEqual(qx.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.shape, (1024, 1))
        self.assertEqual(scale.dtype, torch.float32)

    def test_both_tactics_numerically_equivalent(self):
        """Both runner tactics must produce numerically equivalent quantizations.

        The TRTLLM tactic rounds its per-token scale to bf16 before quantizing;
        the VECTORIZED tactic keeps the scale in float32.  Raw fp8 values near
        a rounding boundary will therefore differ by one bucket.  Equivalence is
        verified on the dequantized (reconstructed) values instead.
        """
        from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
            Fp8PerTokenQuantRunner,
            Fp8PerTokenQuantTactic,
        )

        x = torch.randn(1024, 3072, dtype=torch.bfloat16, device="cuda")
        runner = Fp8PerTokenQuantRunner()

        qx_trt, scale_trt = runner(inputs=[x], tactic=Fp8PerTokenQuantTactic.TRTLLM)
        qx_vec, scale_vec = runner(inputs=[x], tactic=Fp8PerTokenQuantTactic.VECTORIZED)

        self.assertTrue(
            torch.allclose(
                scale_trt.float().squeeze(-1),
                scale_vec.squeeze(-1),
                rtol=1e-2,
                atol=1e-5,
            ),
            "Scales must match between TRTLLM and VECTORIZED tactics",
        )

        # Compare dequantized values; tactics may differ by one fp8 step at bucket boundaries.
        recon_trt = qx_trt.float() * scale_trt.float()
        recon_vec = qx_vec.float() * scale_vec
        self.assertTrue(
            torch.allclose(recon_trt, recon_vec, rtol=0.0, atol=0.5),
            "Dequantized outputs must match between TRTLLM and VECTORIZED tactics",
        )

    def test_scale_dtype_is_always_float32(self):
        """tunable_fp8_per_token_quant must normalize scales to float32.

        The raw TRTLLM kernel op returns input-dtype scales (e.g. bf16); the
        vectorized kernel op already returns float32. Fp8PerTokenQuantRunner.forward()
        normalizes both to float32 (so autotuner profiling times the same cast
        inference pays), and the public op inherits that.
        """
        from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
            Fp8PerTokenQuantRunner,
            Fp8PerTokenQuantTactic,
        )

        x = torch.randn(512, 3072, dtype=torch.bfloat16, device="cuda")

        # The raw kernel op is unnormalized: TRTLLM returns bf16 scales.
        _, scale_raw = torch.ops.tensorrt_llm.quantize_e4m3_activation(x)
        self.assertEqual(scale_raw.dtype, torch.bfloat16, "Raw TRTLLM kernel scale should be bf16")

        # The runner normalizes regardless of tactic.
        runner = Fp8PerTokenQuantRunner()
        _, scale_trt = runner(inputs=[x], tactic=Fp8PerTokenQuantTactic.TRTLLM)
        self.assertEqual(
            scale_trt.dtype, torch.float32, "Runner must normalize TRTLLM scale to float32"
        )

        # The public op must normalize regardless.
        _, scale_public = torch.ops.trtllm.tunable_fp8_per_token_quant(x)
        self.assertEqual(
            scale_public.dtype, torch.float32, "Public op must always return float32 scales"
        )

    def _assert_forced_tactic_bypasses_autotuner(self, env_value, expected_op):
        """Shared body: the public op must dispatch directly to the forced
        tactic without ever consulting AutoTuner.choose_one() (and therefore
        without being able to hit a stale cached tactic)."""
        from tensorrt_llm._torch.autotuner import AutoTuner

        x = torch.randn(128, 3072, dtype=torch.bfloat16, device="cuda")

        old = os.environ.pop("TRTLLM_FP8_QUANT_TACTIC", None)
        try:
            os.environ["TRTLLM_FP8_QUANT_TACTIC"] = env_value

            called = []
            orig_choose_one = AutoTuner.choose_one

            def spy_choose_one(self, *args, **kwargs):
                called.append(True)
                return orig_choose_one(self, *args, **kwargs)

            AutoTuner.choose_one = spy_choose_one
            try:
                qx, scale = torch.ops.trtllm.tunable_fp8_per_token_quant(x)
            finally:
                AutoTuner.choose_one = orig_choose_one

            self.assertFalse(
                called,
                "AutoTuner.choose_one must not run when TRTLLM_FP8_QUANT_TACTIC is set",
            )

            qx_ref, scale_ref = expected_op(x)
            self.assertTrue(torch.equal(qx, qx_ref))
            self.assertTrue(torch.allclose(scale, scale_ref.float()))
        finally:
            if old is None:
                os.environ.pop("TRTLLM_FP8_QUANT_TACTIC", None)
            else:
                os.environ["TRTLLM_FP8_QUANT_TACTIC"] = old

    def test_forced_trtllm_tactic(self):
        """TRTLLM_FP8_QUANT_TACTIC=trtllm must dispatch to the TRTLLM kernel."""
        self._assert_forced_tactic_bypasses_autotuner(
            "trtllm", torch.ops.tensorrt_llm.quantize_e4m3_activation
        )

    def test_forced_vectorized_tactic(self):
        """TRTLLM_FP8_QUANT_TACTIC=vectorized must dispatch to the vectorized kernel."""
        self._assert_forced_tactic_bypasses_autotuner(
            "vectorized", torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant
        )

    def test_forced_warp_tactic(self):
        self._assert_forced_tactic_bypasses_autotuner(
            "warp_pdl", torch.ops.tensorrt_llm.warp_per_token_fp8_quant
        )

    def test_default_tactics_include_all_three(self):
        from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
            Fp8PerTokenQuantRunner,
            Fp8PerTokenQuantTactic,
        )

        runner = Fp8PerTokenQuantRunner()
        x = torch.randn(128, 3072, dtype=torch.bfloat16, device="cuda")

        old = os.environ.pop("TRTLLM_FP8_QUANT_TACTIC", None)
        try:
            tactics = runner.get_valid_tactics([x], profile=None)
            self.assertIn(Fp8PerTokenQuantTactic.TRTLLM, tactics)
            self.assertIn(Fp8PerTokenQuantTactic.VECTORIZED, tactics)
            self.assertIn(Fp8PerTokenQuantTactic.WARP_PDL, tactics)
        finally:
            if old is None:
                os.environ.pop("TRTLLM_FP8_QUANT_TACTIC", None)
            else:
                os.environ["TRTLLM_FP8_QUANT_TACTIC"] = old


if __name__ == "__main__":
    unittest.main()


def test_default_quant_profiles_all_tactics_and_reuses_cache(monkeypatch):
    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import Fp8PerTokenQuantTactic

    monkeypatch.delenv("TRTLLM_FP8_QUANT_TACTIC", raising=False)
    rows = torch.cuda.get_device_properties(0).multi_processor_count * 16 + 1
    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    tuner = AutoTuner()
    tuner.is_tuning_mode = True
    with (
        patch.object(AutoTuner, "get", return_value=tuner),
        patch.object(
            tuner, "_profile_single_kernel", wraps=tuner._profile_single_kernel
        ) as profile,
    ):
        expected_q, expected_scale = torch.ops.trtllm.tunable_fp8_per_token_quant(x)
        assert {call.kwargs["tactic"] for call in profile.call_args_list} == set(
            Fp8PerTokenQuantTactic
        )
        assert profile.call_count == 3
        profile.reset_mock()
        tuner.is_tuning_mode = False
        q, scale = torch.ops.trtllm.tunable_fp8_per_token_quant(x)
        profile.assert_not_called()
        torch.testing.assert_close(q.float(), expected_q.float(), rtol=0, atol=0)
        torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)


def _assert_quantized_parity(actual, expected):
    actual, expected = actual.float(), expected.float()
    error = actual - expected
    norm = expected.norm()
    if norm == 0:
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        return
    metrics = {
        "relative_l2": (error.norm() / norm).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        ).item(),
        "max_abs": error.abs().max().item(),
        "p99_abs": error.abs().flatten().quantile(0.99).item(),
    }
    print(metrics)
    assert metrics["relative_l2"] <= 0.01 and metrics["cosine"] >= 0.9999, metrics


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_warp_fp8_quant_reference(dtype, enable_pdl):
    """L1 T1: warp quantization versus FP32 rowwise math and the existing kernel."""
    import tensorrt_llm  # noqa: F401

    torch.manual_seed(53)
    rows = torch.cuda.get_device_properties(0).multi_processor_count * 16 + 1
    x = torch.randn(rows, 3072, device="cuda", dtype=dtype)
    x[0].zero_()
    x[1].fill_(1e-8)
    x[2].fill_(32)
    expected_scale = (x.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1 / (448 * 512))
    expected_q = (x.float() / expected_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    q, scale = torch.ops.tensorrt_llm.warp_per_token_fp8_quant(x, enable_pdl)
    reference_q, reference_scale = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x)
    torch.testing.assert_close(scale, expected_scale, rtol=1e-6, atol=0)
    _assert_quantized_parity(q.float() * scale, expected_q.float() * expected_scale)
    _assert_quantized_parity(q.float() * scale, reference_q.float() * reference_scale)


@pytest.mark.parametrize("shape,offset", [((3, 3072), 0), ((8193, 513), 0), ((8193, 512), 1)])
def test_warp_fp8_quant_fallback(shape, offset):
    """L1 T1: small, odd-width, and unaligned inputs preserve the existing path."""
    import tensorrt_llm  # noqa: F401

    torch.manual_seed(54)
    storage = torch.randn(shape[0] * shape[1] + offset, device="cuda", dtype=torch.bfloat16)
    x = storage[offset:].view(shape)
    q, scale = torch.ops.tensorrt_llm.warp_per_token_fp8_quant(x)
    reference_q, reference_scale = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x)
    _assert_quantized_parity(q.float() * scale, reference_q.float() * reference_scale)


@pytest.mark.parametrize("compile_model", [False, True])
def test_warp_fp8_quant_graph_replay(monkeypatch, compile_model):
    """The warp tactic supports eager/compiled graph replay."""
    import tensorrt_llm  # noqa: F401

    monkeypatch.setenv("TRTLLM_FP8_QUANT_TACTIC", "warp_pdl")
    monkeypatch.delenv("TRTLLM_ENABLE_PDL", raising=False)
    torch.manual_seed(55)
    rows = torch.cuda.get_device_properties(0).multi_processor_count * 16 + 1
    x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)

    def run():
        produced = x * 0.5 + 0.125
        q, scale = torch.ops.trtllm.tunable_fp8_per_token_quant(produced)
        return q.float() * scale

    if compile_model:
        run = torch.compile(run, fullgraph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for factor in (0.25, -2.0):
        x.mul_(factor)
        graph.replay()
        q, scale = torch.ops.tensorrt_llm.vectorized_per_token_fp8_quant(x * 0.5 + 0.125)
        _assert_quantized_parity(result, q.float() * scale)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        run()
    assert any("warpPerTokenFp8QuantKernel" in event.name for event in profile.events())


@pytest.mark.parametrize(
    "shape,offset", [((3, 3072), 0), ((8193, 3072), 0), ((8193, 513), 0), ((8193, 512), 1)]
)
@pytest.mark.parametrize("compiled", [False, True])
def test_warp_fp8_quant_bfl_recipe(shape, offset, compiled):
    """L1 T1: explicit 1e-12 scale floor matches BFL on warp and fallback paths."""
    import tensorrt_llm  # noqa: F401

    torch.manual_seed(56)
    storage = torch.randn(shape[0] * shape[1] + offset, device="cuda", dtype=torch.bfloat16)
    x = storage[offset:].view(shape)
    # Include zero rows and values for which the shared quantizer's floor is lossy.
    x[0].zero_()
    x[1].mul_(1e-8)
    x[2].mul_(1e-4)
    rows = x.float()
    scale = (rows.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-12)
    expected = (rows / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale

    def run(value):
        return torch.ops.trtllm.warp_fp8_per_token_quant(value, 1e-12)

    if compiled:
        run = torch.compile(run, fullgraph=True)
    q, actual_scale = run(x)
    torch.testing.assert_close(actual_scale, scale, rtol=1e-6, atol=0)
    # Check the tiny rows independently so normal rows cannot conceal their errors.
    for start, end in ((0, 1), (1, 2), (2, 3), (3, shape[0])):
        if end > start:
            _assert_quantized_parity((q.float() * actual_scale)[start:end], expected[start:end])


@pytest.mark.parametrize("min_scale", [0.0, -1.0, float("nan"), float("inf"), 1e-100])
def test_warp_fp8_quant_invalid_scale(min_scale):
    import tensorrt_llm  # noqa: F401

    x = torch.ones(3, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="min_scale"):
        torch.ops.tensorrt_llm.warp_per_token_fp8_quant(x, True, min_scale)
