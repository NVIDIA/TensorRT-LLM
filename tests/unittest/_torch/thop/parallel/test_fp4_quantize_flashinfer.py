# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests that TRTLLM and FlashInfer FP4 quantization kernels produce identical output."""

import unittest

import pytest
import torch
from parameterized import parameterized
from utils.util import unittest_name_func

import tensorrt_llm

try:
    from flashinfer.fp4_quantization import nvfp4_quantize as flashinfer_nvfp4_quantize

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
class TestFp4QuantizeFlashinfer(unittest.TestCase):
    """Compare TRTLLM and FlashInfer FP4 quantization kernels."""

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand(
        [
            # [M, K, scaling_vector_size, dtype]
            [1, 1024, 16, torch.bfloat16],
            [4, 512, 16, torch.bfloat16],
            [8, 1024, 16, torch.bfloat16],
            [32, 2048, 16, torch.bfloat16],
            [64, 4096, 16, torch.bfloat16],
            [128, 1024, 16, torch.bfloat16],
            [256, 2048, 16, torch.bfloat16],
            [512, 4096, 16, torch.bfloat16],
            [1024, 1024, 16, torch.bfloat16],
            [1, 1024, 16, torch.float16],
            [32, 2048, 16, torch.float16],
            [256, 4096, 16, torch.float16],
        ],
        name_func=unittest_name_func,
    )
    def test_fp4_quantize_output_match(self, m, k, sf_vec_size, dtype):
        """Verify TRTLLM and FlashInfer FP4 quantize produce identical outputs."""
        input_tensor = torch.randn([m, k], dtype=dtype, device="cuda")

        # Compute global scale factor
        FP8_MAX, E2M1_MAX = 448.0, 6.0
        amax = input_tensor.float().abs().max()
        input_scale = (FP8_MAX * E2M1_MAX / amax).cuda()

        # --- TRTLLM kernel ---
        trtllm_fp4, trtllm_sf = torch.ops.trtllm.fp4_quantize(
            input_tensor, input_scale, sf_vec_size, False
        )

        # --- FlashInfer kernel ---
        fi_fp4, fi_sf = flashinfer_nvfp4_quantize(
            input_tensor,
            input_scale,
            do_shuffle=False,
            sf_vec_size=sf_vec_size,
            enable_pdl=False,
        )

        # Both should produce identical packed FP4 values
        torch.testing.assert_close(
            trtllm_fp4,
            fi_fp4,
            atol=0,
            rtol=0,
            msg=f"FP4 quantized values differ for shape [{m}, {k}]",
        )

        # Scale factors should also match exactly
        # Handle potential shape differences by flattening
        trtllm_sf_flat = trtllm_sf.reshape(-1)
        fi_sf_flat = fi_sf.reshape(-1)
        min_len = min(trtllm_sf_flat.shape[0], fi_sf_flat.shape[0])
        torch.testing.assert_close(
            trtllm_sf_flat[:min_len],
            fi_sf_flat[:min_len],
            atol=0,
            rtol=0,
            msg=f"Scale factors differ for shape [{m}, {k}]",
        )

    @parameterized.expand(
        [
            [32, 2048, 16, torch.bfloat16],
            [128, 4096, 16, torch.bfloat16],
        ],
        name_func=unittest_name_func,
    )
    def test_tunable_fp4_quantize_op(self, m, k, sf_vec_size, dtype):
        """Verify the tunable custom op returns valid output."""
        input_tensor = torch.randn([m, k], dtype=dtype, device="cuda")

        FP8_MAX, E2M1_MAX = 448.0, 6.0
        amax = input_tensor.float().abs().max()
        input_scale = (FP8_MAX * E2M1_MAX / amax).cuda()

        # Call the tunable op (uses TRTLLM by default without autotune context)
        result = torch.ops.trtllm.tunable_fp4_quantize(
            input_tensor, input_scale, sf_vec_size, False
        )
        act_fp4, act_sf = result

        # Verify output shapes
        self.assertEqual(act_fp4.shape[0], m)
        self.assertEqual(act_fp4.shape[1], k // 2)
        self.assertEqual(act_fp4.dtype, torch.uint8)

        # Compare with direct TRTLLM call
        ref_fp4, ref_sf = torch.ops.trtllm.fp4_quantize(
            input_tensor, input_scale, sf_vec_size, False
        )

        torch.testing.assert_close(act_fp4, ref_fp4, atol=0, rtol=0)

    @parameterized.expand(
        [
            [32, 2048, 16, torch.bfloat16],
            [128, 4096, 16, torch.bfloat16],
        ],
        name_func=unittest_name_func,
    )
    def test_tunable_fp4_quantize_with_autotune(self, m, k, sf_vec_size, dtype):
        """Verify the tunable op works within autotune context."""
        from tensorrt_llm._torch.autotuner import autotune

        input_tensor = torch.randn([m, k], dtype=dtype, device="cuda")

        FP8_MAX, E2M1_MAX = 448.0, 6.0
        amax = input_tensor.float().abs().max()
        input_scale = (FP8_MAX * E2M1_MAX / amax).cuda()

        with autotune():
            result = torch.ops.trtllm.tunable_fp4_quantize(
                input_tensor, input_scale, sf_vec_size, False
            )
            act_fp4, act_sf = result

        # Verify output is valid
        self.assertEqual(act_fp4.shape[0], m)
        self.assertEqual(act_fp4.shape[1], k // 2)

        # The autotuned result should match one of the two backends
        ref_fp4, ref_sf = torch.ops.trtllm.fp4_quantize(
            input_tensor, input_scale, sf_vec_size, False
        )
        # Allow match with either backend (both should be numerically equivalent)
        if HAS_FLASHINFER:
            fi_fp4, fi_sf = flashinfer_nvfp4_quantize(
                input_tensor,
                input_scale,
                do_shuffle=False,
                sf_vec_size=sf_vec_size,
                enable_pdl=False,
            )
            matches_trtllm = torch.equal(act_fp4, ref_fp4)
            matches_flashinfer = torch.equal(act_fp4, fi_fp4)
            self.assertTrue(
                matches_trtllm or matches_flashinfer,
                "Autotuned result doesn't match either TRTLLM or FlashInfer",
            )
        else:
            torch.testing.assert_close(act_fp4, ref_fp4, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
