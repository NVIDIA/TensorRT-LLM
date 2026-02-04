# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import torch
from parameterized import parameterized
from utils.util import skip_pre_blackwell_unittest, unittest_name_func

import tensorrt_llm


class TestARCQuantFP4(unittest.TestCase):
    """Test cases for ARCQuant FP4 quantization with reorder and residual."""

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand(
        list(
            [
                [128, 512, 64],
                [256, 1024, 128],
                [64, 256, 32],
                [1024, 4096, 512],
                [16, 128, 16],
            ]
        ),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_with_reorder_residual_shape(self, m, kq, ke):
        """Test fp4_quantize_with_reorder_residual output shapes.

        This function quantizes an input tensor with reordering and residual handling.
        Input X has shape [M, KQ], and the output will have K = KQ + KE dimensions
        where the last KE dimensions are interleaved with KQ dimensions.

        Args:
            m: Batch size / sequence length
            kq: Query dimension (input hidden dimension)
            ke: Residual dimension (expanded dimension)
        """
        # Create input tensor
        X = torch.randn([m, kq], dtype=torch.bfloat16).cuda()

        # Create reorder index tensor
        # The reorder index defines how to shuffle the KQ + KE dimensions
        k = kq + ke
        reorder_index = torch.arange(k, dtype=torch.int16).cuda()

        # Call the quantization function
        qx, sfx = torch.ops.trtllm.fp4_quantize_with_reorder_residual(X, reorder_index, ke)

        # Verify output shapes
        # QX should be [M, K/2] since FP4 packs 2 values per byte
        self.assertEqual(qx.shape[0], m)
        self.assertEqual(qx.shape[1], k // 2)
        self.assertEqual(qx.dtype, torch.uint8)

        # Verify scale factor shape
        # Scale factors use a swizzled block-scaled layout
        self.assertGreater(sfx.numel(), 0)
        self.assertEqual(sfx.dtype, torch.float8_e4m3fn)

    @parameterized.expand(
        list(
            [
                [128, 512, 64],
                [256, 1024, 128],
                [16, 128, 16],
            ]
        ),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_with_reorder_residual_validity(self, m, kq, ke):
        """Test that fp4_quantize_with_reorder_residual produces valid outputs."""
        # Create input tensor with known range
        X = torch.randn([m, kq], dtype=torch.bfloat16).cuda()

        # Create identity reorder (no reordering)
        k = kq + ke
        reorder_index = torch.arange(k, dtype=torch.int16).cuda()

        # Call the quantization function
        qx, sfx = torch.ops.trtllm.fp4_quantize_with_reorder_residual(X, reorder_index, ke)

        # Verify that the function runs without errors and produces valid outputs
        self.assertFalse(torch.isnan(qx.float()).any(), "QX contains NaN values")
        self.assertFalse(torch.isnan(sfx.float()).any(), "SFX contains NaN values")
        self.assertFalse(torch.isinf(qx.float()).any(), "QX contains Inf values")
        self.assertFalse(torch.isinf(sfx.float()).any(), "SFX contains Inf values")

    @parameterized.expand(
        list(
            [
                [64, 256, 32],
                [128, 512, 64],
            ]
        ),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_with_reorder_residual_permutation(self, m, kq, ke):
        """Test fp4_quantize_with_reorder_residual with different reorder patterns."""
        X = torch.randn([m, kq], dtype=torch.bfloat16).cuda()
        k = kq + ke

        # Test 1: Identity permutation (no reordering)
        reorder_identity = torch.arange(k, dtype=torch.int16).cuda()
        qx1, sfx1 = torch.ops.trtllm.fp4_quantize_with_reorder_residual(X, reorder_identity, ke)

        # Test 2: Random permutation
        perm = torch.randperm(k)
        reorder_random = torch.arange(k, dtype=torch.int16)[perm].cuda()
        qx2, sfx2 = torch.ops.trtllm.fp4_quantize_with_reorder_residual(X, reorder_random, ke)

        # Both should produce valid outputs with the same shape
        self.assertEqual(qx1.shape, qx2.shape)

        # The outputs should be different due to different reordering
        # (unless the random permutation happens to be identity)
        if not torch.equal(reorder_identity, reorder_random):
            self.assertFalse(torch.equal(qx1, qx2))

    @parameterized.expand(
        list(
            [
                [16, 64, 16],
                [32, 128, 32],
            ]
        ),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_with_reorder_residual_deterministic(self, m, kq, ke):
        """Test that fp4_quantize_with_reorder_residual is deterministic."""
        X = torch.randn([m, kq], dtype=torch.bfloat16).cuda()
        k = kq + ke
        reorder_index = torch.arange(k, dtype=torch.int16).cuda()

        # Run twice with the same inputs
        qx1, sfx1 = torch.ops.trtllm.fp4_quantize_with_reorder_residual(X, reorder_index, ke)
        qx2, sfx2 = torch.ops.trtllm.fp4_quantize_with_reorder_residual(X, reorder_index, ke)

        # Results should be identical
        self.assertTrue(torch.equal(qx1, qx2), "QX outputs are not deterministic")
        self.assertTrue(torch.equal(sfx1, sfx2), "SFX outputs are not deterministic")

    @parameterized.expand(
        list(
            [
                [128, 512, 64, torch.bfloat16],
                [256, 1024, 128, torch.bfloat16],
            ]
        ),
        name_func=unittest_name_func,
    )
    @skip_pre_blackwell_unittest
    def test_fp4_quantize_with_reorder_residual_dtype(self, m, kq, ke, dtype):
        """Test fp4_quantize_with_reorder_residual with different input dtypes."""
        # Currently only bfloat16 is supported based on the implementation
        X = torch.randn([m, kq], dtype=dtype).cuda()
        k = kq + ke
        reorder_index = torch.arange(k, dtype=torch.int16).cuda()

        # Call the quantization function
        qx, sfx = torch.ops.trtllm.fp4_quantize_with_reorder_residual(X, reorder_index, ke)

        # Verify outputs
        self.assertEqual(qx.shape[0], m)
        self.assertEqual(qx.shape[1], k // 2)
        self.assertFalse(torch.isnan(qx.float()).any())
        self.assertFalse(torch.isnan(sfx.float()).any())


if __name__ == "__main__":
    unittest.main()
