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


def reference_calculate_global_scale(input_tensor):
    max_abs_values = input_tensor.abs().max(dim=-1, keepdim=True).values.to(
        torch.float32)
    global_scales = (448 * 6) / max_abs_values
    return global_scales


class TestFP4CalculateGlobalScale(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @parameterized.expand([
        [1, 64, 7168, torch.bfloat16, False],
        [1, 64, 7168, torch.float16, False],
        [1, 64, 7168, torch.bfloat16, True],
        [1, 64, 4096, torch.bfloat16, True],
        [8, 8 * 64, 7168, torch.bfloat16, False],
        [8, 8 * 64, 7168, torch.bfloat16, True],
        [16, 16 * 64, 7168, torch.bfloat16, True],
        [32, 32 * 64, 7168, torch.bfloat16, True],
    ],
                          name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_calculate_nvfp4_global_scale_accuracy(self, batch_size,
                                                   max_token_num, hidden_size,
                                                   dtype, use_tokens_per_batch):
        if batch_size == 1:
            input_shape = (max_token_num, hidden_size)
        else:
            input_shape = (batch_size, max_token_num, hidden_size)
        input_tensor = torch.randn(input_shape, dtype=dtype, device='cuda')

        assert hidden_size % 16 == 0, f"Hidden size {hidden_size} must be divisible by 16"

        tokens_per_batch = None
        if use_tokens_per_batch:
            # Create tokensPerBatch tensor with shape (batch_size)
            # Each value represents the actual number of meaningful tokens in that batch
            tokens_per_batch = torch.randint(0,
                                             max_token_num + 1, (batch_size, ),
                                             device='cuda',
                                             dtype=torch.int32)

        reference_result = reference_calculate_global_scale(input_tensor)
        custom_result = torch.ops.trtllm.calculate_nvfp4_global_scale(
            input_tensor, tokens_per_batch)
        torch.cuda.synchronize()

        self.assertEqual(custom_result.shape, reference_result.shape)

        if use_tokens_per_batch:
            # Create mask for meaningful tokens based on tokens_per_batch
            # Only compare results for tokens that are within the meaningful range
            meaningful_mask = torch.zeros(custom_result.shape,
                                          dtype=torch.bool,
                                          device='cuda')
            if batch_size == 1:
                meaningful_mask[:tokens_per_batch[0]] = True
            else:
                for i in range(batch_size):
                    meaningful_mask[i, :tokens_per_batch[i]] = True

            custom_result = custom_result * meaningful_mask
            reference_result = reference_result * meaningful_mask

        torch.testing.assert_close(
            custom_result,
            reference_result,
            atol=1e-3,
            rtol=1e-3,
            msg=
            f"Shape: {input_shape}, dtype: {dtype}, custom_result: {custom_result}, reference_result: {reference_result}"
        )

    @parameterized.expand(
        [
            # [local_experts_num, ranks_num * max_token_num_per_rank, max_token_num_per_rank, hidden_size]
            [32, 8 * 64, 64, 7168, torch.bfloat16, False],
            [32, 8 * 64, 64, 7168, torch.bfloat16, True],
            [16, 16 * 64, 64, 7168, torch.bfloat16, False],
            [16, 16 * 64, 64, 7168, torch.bfloat16, True],
            [8, 32 * 64, 64, 7168, torch.bfloat16, False],
            [8, 32 * 64, 64, 7168, torch.bfloat16, True],
        ],
        name_func=unittest_name_func)
    @skip_pre_blackwell_unittest
    def test_calculate_nvfp4_global_scale_performance(self, batch_size,
                                                      max_token_num,
                                                      real_token_num,
                                                      hidden_size, dtype,
                                                      use_tokens_per_batch):
        if batch_size == 1:
            input_shape = (max_token_num, hidden_size)
        else:
            input_shape = (batch_size, max_token_num, hidden_size)
        input_tensor = torch.randn(input_shape, dtype=dtype, device='cuda')

        tokens_per_batch = None
        if use_tokens_per_batch:
            tokens_per_batch = torch.zeros((batch_size, ),
                                           device='cuda',
                                           dtype=torch.int32)
            tokens_per_batch[:] = real_token_num

        for _ in range(10):
            _ = torch.ops.trtllm.calculate_nvfp4_global_scale(
                input_tensor, tokens_per_batch)
            _ = reference_calculate_global_scale(input_tensor)

        torch.cuda.synchronize()

        num_iterations = 100

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iterations):
            _ = torch.ops.trtllm.calculate_nvfp4_global_scale(
                input_tensor, tokens_per_batch)
        end_event.record()
        torch.cuda.synchronize()
        custom_time = start_event.elapsed_time(end_event)

        start_event.record()
        for _ in range(num_iterations):
            _ = reference_calculate_global_scale(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        reference_time = start_event.elapsed_time(end_event)

        custom_avg_time = custom_time / num_iterations
        reference_avg_time = reference_time / num_iterations
        speedup = reference_avg_time / custom_avg_time

        tokens_info = "with tokensPerBatch" if use_tokens_per_batch else "without tokensPerBatch"
        print(
            f"\nPerformance Test Results for {input_shape}, {real_token_num}, {dtype}, {tokens_info}:"
        )
        print(f"Custom op average time: {custom_avg_time*1000:.3f} us")
        print(f"Reference average time: {reference_avg_time*1000:.3f} us")
        print(f"Speedup: {speedup:.2f}x")

    @skip_pre_blackwell_unittest
    def test_calculate_nvfp4_global_scale_invalid_inputs(self):
        # Test with 1D tensor (should fail)
        input_tensor = torch.randn(4096, dtype=torch.float16, device='cuda')
        with self.assertRaises(Exception):
            torch.ops.trtllm.calculate_nvfp4_global_scale(input_tensor)

        # Test with hidden_size not divisible by 16 (should fail)
        input_tensor = torch.randn((4, 32, 4095),
                                   dtype=torch.float16,
                                   device='cuda')
        with self.assertRaises(Exception):
            torch.ops.trtllm.calculate_nvfp4_global_scale(input_tensor)

        # Test with mismatched tokensPerBatch shape (wrong batch size)
        input_tensor = torch.randn((4, 32, 4096),
                                   dtype=torch.float16,
                                   device='cuda')
        tokens_per_batch = torch.randint(1,
                                         33, (5, ),
                                         device='cuda',
                                         dtype=torch.int32)  # Wrong batch size
        with self.assertRaises(Exception):
            torch.ops.trtllm.calculate_nvfp4_global_scale(
                input_tensor, tokens_per_batch)

        # Test with tokensPerBatch having wrong number of dimensions (should be 1D)
        input_tensor = torch.randn((4, 32, 4096),
                                   dtype=torch.float16,
                                   device='cuda')
        tokens_per_batch = torch.randint(1,
                                         33, (4, 32),
                                         device='cuda',
                                         dtype=torch.int32)  # 2D instead of 1D
        with self.assertRaises(Exception):
            torch.ops.trtllm.calculate_nvfp4_global_scale(
                input_tensor, tokens_per_batch)

        # Test with tokensPerBatch having wrong first dimension size
        input_tensor = torch.randn((4, 32, 4096),
                                   dtype=torch.float16,
                                   device='cuda')
        tokens_per_batch = torch.randint(1,
                                         33, (3, ),
                                         device='cuda',
                                         dtype=torch.int32)  # Wrong batch size
        with self.assertRaises(Exception):
            torch.ops.trtllm.calculate_nvfp4_global_scale(
                input_tensor, tokens_per_batch)


if __name__ == '__main__':
    unittest.main()
