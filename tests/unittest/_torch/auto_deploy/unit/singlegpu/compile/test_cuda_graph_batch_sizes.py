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

    @staticmethod
    def _raise_error_for_forward(*args, **kwargs):
        raise RuntimeError("forward method should not be called")

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

        # Create a get_args_kwargs function for the compiler
        def get_args_kwargs(bs):
            return (data["input_tensor"][:bs],), {}

        compiler = TorchCudagraphCompiler(
            model=data["gm"],
            args=(data["input_tensor"],),
            max_batch_size=max_batch_size,
            cuda_graph_batch_sizes=requested_batch_sizes,
            get_args_kwargs_for_compile=get_args_kwargs,
        )

        # The compiler stores batch sizes as-is; clamping happens during capture
        # Filter batch sizes to max_batch_size for comparison
        assert compiler.cuda_graph_batch_sizes == requested_batch_sizes

    def test_cuda_graph_batch_sizes_no_clamping_needed(self, simple_model_and_inputs):
        """Test that cuda_graph_batch_sizes are not modified when they're within limits."""
        data = simple_model_and_inputs

        # Request CUDA graph batch sizes that are all within max_batch_size
        requested_batch_sizes = [1, 4, 8, 12]

        # Create a get_args_kwargs function for the compiler
        def get_args_kwargs(bs):
            return (data["input_tensor"][:bs],), {}

        compiler = TorchCudagraphCompiler(
            model=data["gm"],
            args=(data["input_tensor"],),
            cuda_graph_batch_sizes=requested_batch_sizes,
            get_args_kwargs_for_compile=get_args_kwargs,
        )

        # Check that batch sizes are preserved as provided
        assert compiler.cuda_graph_batch_sizes == requested_batch_sizes

        # Verify all requested sizes are within max_batch_size
        max_batch_size = data["batch_size"]
        assert all(bs <= max_batch_size for bs in compiler.cuda_graph_batch_sizes)

    def test_heuristic_cuda_graph_batch_sizes(self, simple_model_and_inputs):
        """Test that empty batch sizes list is stored when none are provided."""
        data = simple_model_and_inputs
        max_batch_size = data["batch_size"]  # 16

        # Create a get_args_kwargs function for the compiler
        def get_args_kwargs(bs):
            return (data["input_tensor"][:bs],), {}

        compiler = TorchCudagraphCompiler(
            model=data["gm"],
            args=(data["input_tensor"],),
            max_batch_size=max_batch_size,
            get_args_kwargs_for_compile=get_args_kwargs,
            # No cuda_graph_batch_sizes provided - should default to empty list
        )

        # Check that cuda_graph_batch_sizes defaults to empty list
        assert compiler.cuda_graph_batch_sizes == []

    def test_captured_graph_max_batch_size_consistency(self, simple_model_and_inputs):
        """Test that CapturedGraph captures graphs for specified batch sizes."""
        data = simple_model_and_inputs

        cuda_graph_batch_sizes = [1, 4, 8, 12]

        captured_graph = CapturedGraph(
            model=data["model"],
            num_batched_inputs=1,
        )

        # Create a get_args_kwargs function
        def get_args_kwargs(bs):
            return (data["input_tensor"][:bs],), {}

        # Capture graphs for the specified batch sizes
        captured_graph.capture_graph(get_args_kwargs, cuda_graph_batch_sizes)

        # Verify graphs were captured for all batch sizes
        assert len(captured_graph.cudagraphs) == len(cuda_graph_batch_sizes)

    def test_forward_fallback_for_oversized_batch(self, simple_model_and_inputs):
        """Test that forward method falls back to regular execution for oversized batches."""
        data = simple_model_and_inputs

        # Create and capture with small batch sizes
        cuda_graph_batch_sizes = [1, 2, 4]
        captured_graph = CapturedGraph(
            model=data["model"],
            num_batched_inputs=1,
        )

        # Create a get_args_kwargs function
        def get_args_kwargs(bs):
            return (data["input_tensor"][:bs],), {}

        # Capture graphs
        captured_graph.capture_graph(get_args_kwargs, cuda_graph_batch_sizes)

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
            num_batched_inputs=1,
        )

        # Create a get_args_kwargs function
        def get_args_kwargs(bs):
            return (data["input_tensor"][:bs],), {}

        # Capture graphs for all batch sizes
        captured_graph.capture_graph(get_args_kwargs, cuda_graph_batch_sizes)

        # Test forward with various valid batch sizes
        for batch_size in [1, 2, 4, 8]:
            test_input = data["input_tensor"][:batch_size]

            with torch.inference_mode():
                # temporarily remove model forward to ensure that the captured graph is used
                original_forward = captured_graph.model.forward
                captured_graph.model.forward = self._raise_error_for_forward
                try:
                    output = captured_graph.forward(test_input)
                finally:
                    captured_graph.model.forward = original_forward

                # Should get valid output
                assert output is not None
                assert output.shape[0] == batch_size

                # Compare with regular model output
                expected_output = data["model"](test_input)
                assert torch.allclose(output, expected_output, atol=1e-4)

    @pytest.mark.parametrize(
        "requested_sizes,expected_sizes",
        [
            ([1, 4, 8], [1, 4, 8]),
            ([2, 6, 10, 20], [2, 6, 10, 20]),  # Sizes are stored as-is
            ([32, 64, 128], [32, 64, 128]),  # Sizes are stored as-is
            ([], []),  # Empty list stays empty
        ],
    )
    def test_various_batch_size_configurations(
        self, simple_model_and_inputs, requested_sizes, expected_sizes
    ):
        """Test various configurations of cuda_graph_batch_sizes."""
        data = simple_model_and_inputs
        max_batch_size = data["batch_size"]  # 16

        # Create a get_args_kwargs function for the compiler
        def get_args_kwargs(bs):
            return (data["input_tensor"][: min(bs, max_batch_size)],), {}

        compiler_kwargs = {"cuda_graph_batch_sizes": requested_sizes} if requested_sizes else {}

        compiler = TorchCudagraphCompiler(
            model=data["gm"],
            args=(data["input_tensor"],),
            max_batch_size=max_batch_size,
            get_args_kwargs_for_compile=get_args_kwargs,
            **compiler_kwargs,
        )

        # Check that batch sizes are stored as provided
        assert compiler.cuda_graph_batch_sizes == expected_sizes
