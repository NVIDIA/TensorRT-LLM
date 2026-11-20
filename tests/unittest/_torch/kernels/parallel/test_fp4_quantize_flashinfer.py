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
"""Tests for tunable FP4 quantization (TRTLLM vs FlashInfer backends)."""

import unittest
from unittest.mock import patch

import pytest
import torch
from parameterized import parameterized
from torch._subclasses.fake_tensor import FakeTensorMode
from utils.util import skip_pre_blackwell_unittest, unittest_name_func

import tensorrt_llm

try:
    from flashinfer.fp4_quantization import nvfp4_quantize as flashinfer_nvfp4_quantize

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


@skip_pre_blackwell_unittest
class TestFp4QuantizeFlashinfer(unittest.TestCase):
    """Test tunable FP4 quantization across TRTLLM and FlashInfer backends."""

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand(
        [
            (num_rows, num_columns, input_dtype, is_sf_swizzled_layout)
            for num_rows, num_columns, is_sf_swizzled_layout in (
                (1, 512, False),
                (128, 512, False),
                (128, 512, True),
                (256, 3840, False),
                (256, 3840, True),
                (1344, 3840, False),
            )
            for input_dtype in (torch.bfloat16, torch.float16)
        ],
        name_func=unittest_name_func,
    )
    @pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
    def test_forced_flashinfer_dispatch_preserves_layout(
        self,
        num_rows: int,
        num_columns: int,
        input_dtype: torch.dtype,
        is_sf_swizzled_layout: bool,
    ) -> None:
        """Preserve activation rows and the requested scale-factor layout."""
        from tensorrt_llm._torch.custom_ops import torch_custom_ops

        input_tensor = torch.randint(-4, 5, (num_rows, num_columns), device="cuda").to(input_dtype)
        input_tensor[:, ::16] = 6
        block_scales = torch.exp2(
            torch.randint(-2, 3, (num_rows, num_columns // 16), device="cuda").float()
        )
        input_tensor.mul_(block_scales.repeat_interleave(16, dim=1))
        input_scale = torch.ones((), dtype=torch.float32, device="cuda")
        expected_values, expected_scales = torch.ops.trtllm.fp4_quantize(
            input_tensor, input_scale, 16, False, is_sf_swizzled_layout
        )

        with patch.object(
            torch_custom_ops,
            "_flashinfer_nvfp4_quantize",
            wraps=torch_custom_ops._flashinfer_nvfp4_quantize,
        ) as flashinfer_quantize:
            actual_values, actual_scales = torch_custom_ops._fp4_quantize_dispatch(
                input_tensor,
                input_scale,
                16,
                False,
                is_sf_swizzled_layout,
                torch_custom_ops.Fp4QuantTactic.FLASHINFER,
            )

        flashinfer_quantize.assert_called_once()
        torch.testing.assert_close(actual_values, expected_values, atol=0, rtol=0)
        torch.testing.assert_close(actual_scales, expected_scales, atol=0, rtol=0)

    @parameterized.expand([(False, 32), (True, 4096)], name_func=unittest_name_func)
    def test_fake_quantizer_scale_layout(
        self, is_sf_swizzled_layout: bool, expected_scale_size: int
    ) -> None:
        with FakeTensorMode():
            input_tensor = torch.empty((1, 512), dtype=torch.bfloat16, device="cuda")
            input_scale = torch.empty((), dtype=torch.float32, device="cuda")
            quantized, scales = torch.ops.trtllm.tunable_fp4_quantize(
                input_tensor, input_scale, 16, False, is_sf_swizzled_layout
            )
        self.assertEqual(quantized.shape, (1, 256))
        self.assertEqual(scales.shape, (expected_scale_size,))

    @pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not available")
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

        # Scale factors should match exactly (shape and values)
        trtllm_sf_flat = trtllm_sf.reshape(-1)
        fi_sf_flat = fi_sf.reshape(-1)
        self.assertEqual(
            trtllm_sf_flat.numel(),
            fi_sf_flat.numel(),
            f"Scale factor sizes differ for shape [{m}, {k}]: "
            f"trtllm={trtllm_sf_flat.numel()}, flashinfer={fi_sf_flat.numel()}",
        )
        torch.testing.assert_close(
            trtllm_sf_flat,
            fi_sf_flat,
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
        """Verify the tunable custom op returns valid output matching TRTLLM."""
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

        # Compare with direct TRTLLM call (both fp4 and scale factors)
        ref_fp4, ref_sf = torch.ops.trtllm.fp4_quantize(
            input_tensor, input_scale, sf_vec_size, False
        )

        torch.testing.assert_close(act_fp4, ref_fp4, atol=0, rtol=0)
        self.assertEqual(
            act_sf.shape,
            ref_sf.shape,
            f"Scale factor shapes differ: tunable={act_sf.shape}, ref={ref_sf.shape}",
        )
        torch.testing.assert_close(
            act_sf,
            ref_sf,
            atol=0,
            rtol=0,
            msg="Scale factors from tunable op differ from TRTLLM reference",
        )

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

        if HAS_FLASHINFER:
            fi_fp4, fi_sf = flashinfer_nvfp4_quantize(
                input_tensor,
                input_scale,
                do_shuffle=False,
                sf_vec_size=sf_vec_size,
                enable_pdl=False,
            )
            # act_fp4 must match either backend
            matches_trtllm = torch.equal(act_fp4, ref_fp4)
            matches_flashinfer = torch.equal(act_fp4, fi_fp4)
            self.assertTrue(
                matches_trtllm or matches_flashinfer,
                "Autotuned FP4 values don't match either TRTLLM or FlashInfer",
            )
            # act_sf must also match the corresponding backend
            sf_matches_trtllm = torch.equal(act_sf.reshape(-1), ref_sf.reshape(-1))
            sf_matches_flashinfer = torch.equal(act_sf.reshape(-1), fi_sf.reshape(-1))
            self.assertTrue(
                sf_matches_trtllm or sf_matches_flashinfer,
                "Autotuned scale factors don't match either TRTLLM or FlashInfer",
            )
        else:
            # Without FlashInfer, must match TRTLLM exactly
            torch.testing.assert_close(act_fp4, ref_fp4, atol=0, rtol=0)
            torch.testing.assert_close(
                act_sf,
                ref_sf,
                atol=0,
                rtol=0,
                msg="Scale factors from tunable op differ from TRTLLM reference",
            )


if __name__ == "__main__":
    unittest.main()
