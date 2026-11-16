# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

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
    """Test the autotuner op that selects between TRT-LLM and vectorized kernels."""

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

    def test_default_tactics_include_both(self):
        """Without env override, both tactics must be available."""
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
        finally:
            if old is None:
                os.environ.pop("TRTLLM_FP8_QUANT_TACTIC", None)
            else:
                os.environ["TRTLLM_FP8_QUANT_TACTIC"] = old


if __name__ == "__main__":
    unittest.main()
