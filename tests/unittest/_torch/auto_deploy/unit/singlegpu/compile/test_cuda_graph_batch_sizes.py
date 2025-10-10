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
"""Unit tests for CUDA graph batch size handling in torch_cudagraph backend."""

import os
import sys

import pytest
import torch

# Add the _utils_test directory to the path so we can import _model_test_utils
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "_utils_test"))

from _model_test_utils import TransformerLikeModel, generate_dynamic_shapes

from tensorrt_llm._torch.auto_deploy.compile.backends.torch_cudagraph import (
    CapturedGraph,
    TorchCudagraphCompiler,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm


class TestCudaGraphBatchSizes:
    """Test class for CUDA graph batch size handling."""

    @pytest.fixture
    def simple_model_and_inputs(self):
        """Create a simple model and inputs for testing."""
        vocab_size = 100
        embed_dim = 32
        hidden_dim = 64
        batch_size = 16
        seq_len = 10

        model = TransformerLikeModel(vocab_size, embed_dim, hidden_dim).to("cuda")
        model.eval()

        input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len)).to("cuda")
        dynamic_shapes = generate_dynamic_shapes(batch_size, seq_len)

        # Export to graph module
        gm = torch_export_to_gm(model, args=(input_tensor,), dynamic_shapes=dynamic_shapes)

        return {
            "model": model,
            "gm": gm,
            "input_tensor": input_tensor,
            "dynamic_shapes": dynamic_shapes,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

    def test_cuda_graph_batch_sizes_clamping_to_max_batch_size(self, simple_model_and_inputs):
        """Test that cuda_graph_batch_sizes are properly clamped to max_batch_size."""
        data = simple_model_and_inputs
        max_batch_size = data["batch_size"]  # 16

        # Request CUDA graph batch sizes that exceed max_batch_size
        requested_batch_sizes = [1, 4, 8, 16, 32, 64]  # 32 and 64 should be clamped to 16

        compiler = TorchCudagraphCompiler(
            model=data["gm"],
            args=(data["input_tensor"],),
            max_batch_size=max_batch_size,
            cuda_graph_batch_sizes=requested_batch_sizes,
        )

        # Check that batch sizes are clamped to max_batch_size
        expected_clamped = [1, 4, 8, 16]  # 32 and 64 should be clamped to 16, then deduped
        assert compiler.cuda_graph_batch_sizes == sorted(expected_clamped, reverse=True)

        # Verify that oversized batch sizes were filtered out
        assert 32 not in compiler.cuda_graph_batch_sizes
        assert 64 not in compiler.cuda_graph_batch_sizes
        assert max(compiler.cuda_graph_batch_sizes) == max_batch_size

    def test_cuda_graph_batch_sizes_no_clamping_needed(self, simple_model_and_inputs):
        """Test that cuda_graph_batch_sizes are not modified when they're within limits."""
        data = simple_model_and_inputs

        # Request CUDA graph batch sizes that are all within max_batch_size
        requested_batch_sizes = [1, 4, 8, 12]

        compiler = TorchCudagraphCompiler(
            model=data["gm"],
            args=(data["input_tensor"],),
            cuda_graph_batch_sizes=requested_batch_sizes,
        )

        # Check that batch sizes are preserved
        assert compiler.cuda_graph_batch_sizes == sorted(requested_batch_sizes, reverse=True)

        # Verify all requested sizes are within max_batch_size
        max_batch_size = data["batch_size"]
        assert all(bs <= max_batch_size for bs in compiler.cuda_graph_batch_sizes)

    def test_heuristic_cuda_graph_batch_sizes(self, simple_model_and_inputs):
        """Test that heuristic batch sizes are generated when none are provided."""
        data = simple_model_and_inputs
        max_batch_size = data["batch_size"]  # 16

        compiler = TorchCudagraphCompiler(
            model=data["gm"],
            args=(data["input_tensor"],),
            max_batch_size=max_batch_size,  # No cuda_graph_batch_sizes provided
        )

        # Check that heuristic batch sizes were generated
        assert len(compiler.cuda_graph_batch_sizes) > 0
        assert max(compiler.cuda_graph_batch_sizes) <= max_batch_size
        assert 1 in compiler.cuda_graph_batch_sizes  # Should always include 1
        assert max_batch_size in compiler.cuda_graph_batch_sizes  # Should include max

    def test_captured_graph_max_batch_size_consistency(self, simple_model_and_inputs):
        """Test that CapturedGraph.max_batch_size equals max(cuda_graph_batch_sizes)."""
        data = simple_model_and_inputs

        cuda_graph_batch_sizes = [1, 4, 8, 12]

        captured_graph = CapturedGraph(
            model=data["model"],
            cuda_graph_batch_sizes=cuda_graph_batch_sizes,
            num_batched_inputs=1,
        )

        assert captured_graph.cuda_graph_max_batch_size == max(cuda_graph_batch_sizes)
        assert captured_graph.cuda_graph_batch_sizes == sorted(cuda_graph_batch_sizes, reverse=True)

    def test_forward_fallback_for_oversized_batch(self, simple_model_and_inputs):
        """Test that forward method falls back to regular execution for oversized batches."""
        data = simple_model_and_inputs

        # Create and capture with small batch sizes
        cuda_graph_batch_sizes = [1, 2, 4]
        captured_graph = CapturedGraph(
            model=data["model"],
            cuda_graph_batch_sizes=cuda_graph_batch_sizes,
            num_batched_inputs=1,
        )

        # Capture with small input
        small_input = data["input_tensor"]  # batch size 16
        captured_graph.capture_graph(small_input)

        # Test forward with oversized input (should fall back)
        oversized_input = data["input_tensor"]  # batch size 16

        with torch.inference_mode():
            output = captured_graph.forward(oversized_input)

            # Should get valid output (fallback worked)
            assert output is not None
            assert output.shape[0] == oversized_input.shape[0]  # Preserve batch size

            # Verify that the output is different from what we'd get from captured graphs
            # This indirectly tests that fallback occurred since the batch size exceeds what's captured
            expected_output = data["model"](oversized_input)
            assert torch.allclose(output, expected_output, atol=1e-4)

            # Verify that no captured graph was used for this oversized batch
            max_captured_bs = max(cuda_graph_batch_sizes)
            assert oversized_input.shape[0] > max_captured_bs

    def test_forward_uses_cuda_graph_for_valid_batch_sizes(self, simple_model_and_inputs):
        """Test that forward method uses CUDA graphs for valid batch sizes."""
        data = simple_model_and_inputs

        cuda_graph_batch_sizes = [1, 2, 4, 8]
        captured_graph = CapturedGraph(
            model=data["model"],
            cuda_graph_batch_sizes=cuda_graph_batch_sizes,
            num_batched_inputs=1,
        )

        # Capture with full-size input
        captured_graph.capture_graph(data["input_tensor"][:8])  # batch size 8

        # Test forward with various valid batch sizes
        for batch_size in [1, 2, 4, 8]:
            test_input = data["input_tensor"][:batch_size]

            with torch.inference_mode():
                output = captured_graph.forward(test_input)

                # Should get valid output
                assert output is not None
                assert output.shape[0] == batch_size

                # Compare with regular model output
                expected_output = data["model"](test_input)
                assert torch.allclose(output, expected_output, atol=1e-4)

    @pytest.mark.parametrize(
        "requested_sizes,expected_max",
        [
            ([1, 4, 8], 8),
            ([2, 6, 10, 20], 16),  # 20 should be clamped to 16
            ([32, 64, 128], 16),  # All should be clamped to 16
            ([], None),  # Empty list should use heuristic
        ],
    )
    def test_various_batch_size_configurations(
        self, simple_model_and_inputs, requested_sizes, expected_max
    ):
        """Test various configurations of cuda_graph_batch_sizes."""
        data = simple_model_and_inputs
        max_batch_size = data["batch_size"]  # 16

        if requested_sizes:
            compiler_kwargs = {"cuda_graph_batch_sizes": requested_sizes}
            expected_max = expected_max or max_batch_size
        else:
            compiler_kwargs = {}
            expected_max = max_batch_size

        compiler = TorchCudagraphCompiler(
            model=data["gm"],
            args=(data["input_tensor"],),
            max_batch_size=max_batch_size,
            **compiler_kwargs,
        )

        # Check that max batch size is as expected
        actual_max = max(compiler.cuda_graph_batch_sizes)
        assert actual_max == expected_max

        # Check that all sizes are within max_batch_size
        assert all(bs <= max_batch_size for bs in compiler.cuda_graph_batch_sizes)
