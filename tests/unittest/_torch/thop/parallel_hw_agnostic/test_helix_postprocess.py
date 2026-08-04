# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch
from parameterized import parameterized

import tensorrt_llm


def baseline(gathered_o, gathered_stats, kv_lora_rank, scale, native_v1=False, native_v2=False):
    """Reference implementation (libtorch)

    Args:
        gathered_o: Input tensor
            - native_v1=False, native_v2=False: [cp_size, num_tokens, num_heads * kv_lora_rank]
            - native_v1=True (cp_dim=2): [num_tokens, num_heads, cp_size, kv_lora_rank]
            - native_v2=True (cp_dim=1): [num_tokens, cp_size, num_heads, kv_lora_rank]
        gathered_stats: Stats tensor
            - native_v1=False, native_v2=False: [cp_size, num_tokens, num_heads, 2]
            - native_v1=True (cp_dim=2): [num_tokens, num_heads, cp_size, 2]
            - native_v2=True (cp_dim=1): [num_tokens, cp_size, num_heads, 2]
        kv_lora_rank: KV LoRA rank
        scale: Scale factor
        native_v1: Whether to use native V1 layout (cp_dim=2).
        native_v2: Whether to use native V2 layout (cp_dim=1).
    """
    if native_v2:
        # Native V2 layout: cp_dim=1.
        # [num_tokens, cp_size, num_heads, 2] -> reduce over cp_size (dim=1).
        global_max = gathered_stats[..., 0].max(dim=1, keepdim=True)[0]
        corrected_max = gathered_stats[..., 0] - global_max
        corrected_max_exp = torch.exp(corrected_max)
        corrected_sum = gathered_stats[..., 1] * corrected_max_exp
        global_sum = corrected_sum.sum(dim=1, keepdim=True)
        correction = (gathered_stats[..., 1] * corrected_max_exp / global_sum).unsqueeze(-1)
        gathered_o_fp32 = gathered_o.to(torch.float32)
        corrected_o = gathered_o_fp32 * correction
        # Sum over cp_size dimension (dim=1), result: [num_tokens, num_heads, kv_lora_rank]
        corrected_o = corrected_o.sum(dim=1)
        # Reshape to [num_tokens, num_heads * kv_lora_rank]
        corrected_o = corrected_o.view(corrected_o.shape[0], -1)
    elif native_v1:
        # Native V1 layout: cp_dim=2.
        # [num_tokens, num_heads, cp_size]
        global_max = gathered_stats[..., 0].max(dim=-1, keepdim=True)[0]
        corrected_max = gathered_stats[..., 0] - global_max
        corrected_max_exp = torch.exp(corrected_max)
        corrected_sum = gathered_stats[..., 1] * corrected_max_exp
        global_sum = corrected_sum.sum(dim=-1, keepdim=True)
        correction = (gathered_stats[..., 1] * corrected_max_exp / global_sum).unsqueeze(-1)
        gathered_o_fp32 = gathered_o.to(torch.float32)
        corrected_o = gathered_o_fp32 * correction
        # Sum over cp_size dimension (dim=2), result: [num_tokens, num_heads, kv_lora_rank]
        corrected_o = corrected_o.sum(dim=2)
        # Reshape to [num_tokens, num_heads * kv_lora_rank]
        corrected_o = corrected_o.view(corrected_o.shape[0], -1)
    else:
        # Original layout: cp_dim=0
        # [cp_size, num_tokens, num_heads]
        global_max = gathered_stats[..., 0].max(dim=0, keepdim=True)[0]
        corrected_max = gathered_stats[..., 0] - global_max
        corrected_max_exp = torch.exp(corrected_max)
        corrected_sum = gathered_stats[..., 1] * corrected_max_exp
        global_sum = corrected_sum.sum(dim=0, keepdim=True)
        correction = (gathered_stats[..., 1] * corrected_max_exp / global_sum).unsqueeze(-1)
        gathered_o_fp32 = gathered_o.to(torch.float32).view(*correction.shape[:-1], kv_lora_rank)
        corrected_o = gathered_o_fp32 * correction
        # [num_tokens, num_heads * kv_lora_rank]
        corrected_o = corrected_o.view(*gathered_o.shape[:-1], -1).sum(dim=0)

    return corrected_o.to(gathered_o.dtype) * scale


class TestHelixPostProcess(unittest.TestCase):
    def setUp(self):
        tensorrt_llm.logger.set_level("warning")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def _test_helix_postprocess(
        self,
        cp_size,
        num_tokens,
        num_heads,
        kv_lora_rank,
        scale,
        dtype,
        native_v1=False,
        native_v2=False,
    ):
        """Test helix postprocessing with given parameters.

        Args:
            cp_size: Context parallelism size.
            num_tokens: Number of tokens.
            num_heads: Number of attention heads.
            kv_lora_rank: KV LoRA rank.
            scale: Scale factor.
            dtype: Data type (float16 or bfloat16).
            native_v1: Whether to use native V1 layout (cp_dim=2).
            native_v2: Whether to use native V2 layout (cp_dim=1).
        """
        device = torch.device("cuda")

        if native_v2:
            # Native V2 layout: [num_tokens, cp_size, num_heads, kv_lora_rank].
            gathered_o = torch.empty(
                num_tokens, cp_size, num_heads, kv_lora_rank, dtype=dtype, device=device
            ).uniform_(-1, 1)
            # gathered_stats: [num_tokens, cp_size, num_heads, 2].
            gathered_stats = torch.empty(
                num_tokens, cp_size, num_heads, 2, dtype=torch.float32, device=device
            )
            gathered_o_max = torch.max(gathered_o, dim=-1, keepdim=True)[0]
            gathered_stats[..., 0] = gathered_o_max[..., 0]
            gathered_o_sum = torch.sum(torch.exp(gathered_o - gathered_o_max), dim=-1)
            gathered_stats[..., 1] = gathered_o_sum

            # Call the custom operator with cp_dim=1 (V2).
            output = torch.ops.trtllm.helix_post_process_native(
                gathered_o, gathered_stats, scale, 1
            )
        elif native_v1:
            # Native V1 layout: [num_tokens, num_heads, cp_size, kv_lora_rank].
            gathered_o = torch.empty(
                num_tokens, num_heads, cp_size, kv_lora_rank, dtype=dtype, device=device
            ).uniform_(-1, 1)
            # gathered_stats: [num_tokens, num_heads, cp_size, 2].
            gathered_stats = torch.empty(
                num_tokens, num_heads, cp_size, 2, dtype=torch.float32, device=device
            )
            gathered_o_max = torch.max(gathered_o, dim=-1, keepdim=True)[0]
            gathered_stats[..., 0] = gathered_o_max[..., 0]
            gathered_o_sum = torch.sum(torch.exp(gathered_o - gathered_o_max), dim=-1)
            gathered_stats[..., 1] = gathered_o_sum

            # Call the custom operator with cp_dim=2 (V1).
            output = torch.ops.trtllm.helix_post_process_native(
                gathered_o, gathered_stats, scale, 2
            )
        else:
            # Original layout: [cp_size, num_tokens, num_heads, kv_lora_rank].
            gathered_o_init = torch.empty(
                cp_size, num_tokens, num_heads, kv_lora_rank, dtype=dtype, device=device
            ).uniform_(-1, 1)
            # gathered_stats: [cp_size, num_tokens, num_heads, 2].
            gathered_stats = torch.empty(
                cp_size, num_tokens, num_heads, 2, dtype=torch.float32, device=device
            )
            gathered_o_max = torch.max(gathered_o_init, dim=-1, keepdim=True)[0]
            gathered_stats[..., 0] = gathered_o_max[..., 0]
            gathered_o_sum = torch.sum(torch.exp(gathered_o_init - gathered_o_max), dim=-1)
            gathered_stats[..., 1] = gathered_o_sum

            gathered_o = gathered_o_init.view(cp_size, num_tokens, num_heads * kv_lora_rank)

            # Call the custom operator.
            output = torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, scale)

        # Compute baseline.
        expected_output = baseline(
            gathered_o,
            gathered_stats,
            kv_lora_rank,
            scale,
            native_v1=native_v1,
            native_v2=native_v2,
        )

        # Compare results.
        torch.testing.assert_close(output, expected_output, atol=1e-3, rtol=1e-2)

    @parameterized.expand(
        [
            # (cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype, native_v1, native_v2)
            # Original layout.
            (4, 8, 2, 64, 1.0, torch.float16, False, False),
            (8, 16, 4, 128, 0.5, torch.float16, False, False),
            (16, 32, 8, 256, 2.0, torch.float16, False, False),
            (4, 8, 2, 64, 1.0, torch.bfloat16, False, False),
            (8, 16, 4, 128, 0.5, torch.bfloat16, False, False),
            (16, 32, 8, 256, 2.0, torch.bfloat16, False, False),
            # Native V1 layout (cp_dim=2).
            (4, 8, 2, 64, 1.0, torch.float16, True, False),
            (8, 16, 4, 128, 0.5, torch.float16, True, False),
            (16, 32, 8, 256, 2.0, torch.float16, True, False),
            (4, 8, 2, 64, 1.0, torch.bfloat16, True, False),
            (8, 16, 4, 128, 0.5, torch.bfloat16, True, False),
            (16, 32, 8, 256, 2.0, torch.bfloat16, True, False),
            # Native V2 layout (cp_dim=1).
            (4, 8, 2, 64, 1.0, torch.float16, False, True),
            (8, 16, 4, 128, 0.5, torch.float16, False, True),
            (16, 32, 8, 256, 2.0, torch.float16, False, True),
            (4, 8, 2, 64, 1.0, torch.bfloat16, False, True),
            (8, 16, 4, 128, 0.5, torch.bfloat16, False, True),
            (16, 32, 8, 256, 2.0, torch.bfloat16, False, True),
        ]
    )
    def test_helix_postprocess_basic(
        self, cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype, native_v1, native_v2
    ):
        """Test basic helix postprocessing functionality."""
        self._test_helix_postprocess(
            cp_size,
            num_tokens,
            num_heads,
            kv_lora_rank,
            scale,
            dtype,
            native_v1=native_v1,
            native_v2=native_v2,
        )

    @parameterized.expand(
        [
            # (cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype, native_v1, native_v2)
            # Edge cases for original layout.
            (1, 1, 1, 16, 1.0, torch.float16, False, False),  # Minimal sizes.
            (256, 1, 1, 16, 1.0, torch.float16, False, False),  # Max cp_size.
            (128, 1, 1, 16, 1.0, torch.float16, False, False),  # Single token.
            (4, 8, 1, 16, 1.0, torch.float16, False, False),  # Single head.
            (4, 8, 2, 2048, 1.0, torch.float16, False, False),  # Large kv_lora_rank.
            # Edge cases for native V1 layout.
            (1, 1, 1, 16, 1.0, torch.float16, True, False),  # Minimal sizes.
            (256, 1, 1, 16, 1.0, torch.float16, True, False),  # Max cp_size.
            (128, 1, 1, 16, 1.0, torch.float16, True, False),  # Single token.
            (4, 8, 1, 16, 1.0, torch.float16, True, False),  # Single head.
            # Note: Large kv_lora_rank (2048) exceeds MAX_KV_LORA_BYTES for native kernel.
            # Edge cases for native V2 layout.
            (1, 1, 1, 16, 1.0, torch.float16, False, True),  # Minimal sizes.
            (256, 1, 1, 16, 1.0, torch.float16, False, True),  # Max cp_size.
            (128, 1, 1, 16, 1.0, torch.float16, False, True),  # Single token.
            (4, 8, 1, 16, 1.0, torch.float16, False, True),  # Single head.
        ]
    )
    def test_helix_postprocess_edge_cases(
        self, cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype, native_v1, native_v2
    ):
        """Test edge cases with minimal dimensions."""
        self._test_helix_postprocess(
            cp_size,
            num_tokens,
            num_heads,
            kv_lora_rank,
            scale,
            dtype,
            native_v1=native_v1,
            native_v2=native_v2,
        )

    @parameterized.expand(
        [
            # (cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype, native_v1, native_v2)
            (16, 16, 64, 512, 1.0, torch.float16, False, False),
            (16, 16, 64, 512, 1.0, torch.bfloat16, False, False),
            (16, 16, 64, 512, 1.0, torch.float16, True, False),
            (16, 16, 64, 512, 1.0, torch.bfloat16, True, False),
            (16, 16, 64, 512, 1.0, torch.float16, False, True),
            (16, 16, 64, 512, 1.0, torch.bfloat16, False, True),
        ]
    )
    def test_helix_postprocess_large_inputs(
        self, cp_size, num_tokens, num_heads, kv_lora_rank, scale, dtype, native_v1, native_v2
    ):
        """Test with larger inputs to ensure performance and correctness."""
        self._test_helix_postprocess(
            cp_size,
            num_tokens,
            num_heads,
            kv_lora_rank,
            scale,
            dtype,
            native_v1=native_v1,
            native_v2=native_v2,
        )

    def test_helix_postprocess_invalid_inputs(self):
        """Test error handling for invalid inputs (non-native)"""
        device = torch.device("cuda")

        # Test with wrong tensor dimensions
        gathered_o = torch.randn(4, 8, 128, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(4, 8, 3, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)
        gathered_stats = torch.randn(4, 8, 2, 1, dtype=torch.float32, device=device)
        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)

        # Test with wrong data types
        gathered_o = torch.randn(4, 8, 128, dtype=torch.float32, device=device)
        gathered_stats = torch.randn(4, 8, 2, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)

        # Test with non-contiguous tensors
        gathered_o = torch.randn(4, 8, 128, dtype=torch.float16, device=device).transpose(0, 1)
        gathered_stats = torch.randn(4, 8, 2, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)

    def test_helix_postprocess_native_invalid_inputs(self):
        """Test error handling for invalid inputs (native layout)"""
        device = torch.device("cuda")

        # Test with wrong cp_dim (only cp_dim=1 and cp_dim=2 are supported).
        gathered_o = torch.randn(8, 2, 4, 64, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 0)
        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 3)

        # Test with wrong tensor dimensions (3D instead of 4D)
        gathered_o = torch.randn(8, 2, 256, dtype=torch.float16, device=device)
        gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)

        # Test with wrong data types
        gathered_o = torch.randn(8, 2, 4, 64, dtype=torch.float32, device=device)
        gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)

        # Test with non-contiguous tensors
        gathered_o = torch.randn(8, 2, 4, 64, dtype=torch.float16, device=device).transpose(0, 1)
        gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError):
            torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)

    @parameterized.expand(
        [
            # (layout,) — "nccl", "fifo_v1", "fifo_v2".
            ("nccl",),
            ("fifo_v1",),
            ("fifo_v2",),
        ]
    )
    def test_helix_postprocess_alignment_requirements(self, layout):
        """Test alignment requirements for all layouts."""
        device = torch.device("cuda")

        # For float16 (2 bytes), kv_lora_rank must be multiple of 8 for 16-byte alignment.
        if layout == "fifo_v1":
            # V1 layout: [num_tokens, num_heads, cp_size, kv_lora_rank].
            # This should work (kv_lora_rank = 64 is multiple of 8).
            gathered_o = torch.randn(8, 2, 4, 64, dtype=torch.float16, device=device)
            gathered_stats = torch.randn(8, 2, 4, 2, dtype=torch.float32, device=device)

            try:
                torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)
            except RuntimeError as e:
                pytest.fail(f"Should not raise error for valid alignment: {e}")

            # Test with kv_lora_rank that doesn't satisfy alignment requirements.
            gathered_o = torch.randn(8, 1, 4, 4, dtype=torch.float16, device=device)
            gathered_stats = torch.randn(8, 1, 4, 2, dtype=torch.float32, device=device)
            with pytest.raises(RuntimeError):
                torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 2)
        elif layout == "fifo_v2":
            # V2 layout: [num_tokens, cp_size, num_heads, kv_lora_rank].
            # This should work (kv_lora_rank = 64 is multiple of 8).
            gathered_o = torch.randn(8, 4, 2, 64, dtype=torch.float16, device=device)
            gathered_stats = torch.randn(8, 4, 2, 2, dtype=torch.float32, device=device)

            try:
                torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 1)
            except RuntimeError as e:
                pytest.fail(f"Should not raise error for valid alignment: {e}")

            # Test with kv_lora_rank that doesn't satisfy alignment requirements.
            gathered_o = torch.randn(8, 4, 1, 4, dtype=torch.float16, device=device)
            gathered_stats = torch.randn(8, 4, 1, 2, dtype=torch.float32, device=device)
            with pytest.raises(RuntimeError):
                torch.ops.trtllm.helix_post_process_native(gathered_o, gathered_stats, 1.0, 1)
        else:
            # NCCL layout.
            # This should work (kv_lora_rank = 64 is multiple of 8).
            gathered_o = torch.randn(4, 8, 2 * 64, dtype=torch.float16, device=device)
            gathered_stats = torch.randn(4, 8, 2, 2, dtype=torch.float32, device=device)

            try:
                torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)
            except RuntimeError as e:
                pytest.fail(f"Should not raise error for valid alignment: {e}")

            # Test with kv_lora_rank that doesn't satisfy alignment requirements.
            gathered_o = torch.randn(4, 8, 4, dtype=torch.float16, device=device)
            gathered_stats = torch.randn(4, 8, 1, 2, dtype=torch.float32, device=device)
            with pytest.raises(RuntimeError):
                torch.ops.trtllm.helix_post_process(gathered_o, gathered_stats, 1.0)


if __name__ == "__main__":
    unittest.main()
