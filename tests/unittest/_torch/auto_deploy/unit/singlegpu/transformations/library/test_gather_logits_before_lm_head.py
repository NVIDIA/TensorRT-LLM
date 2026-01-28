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

"""Unit tests for gather_logits_before_lm_head transform."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import Dim
from torch.fx import GraphModule

# Import to register custom op
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class SimpleLMHeadModel(torch.nn.Module):
    """Simple model with LM head for testing."""

    def __init__(self, hidden_size: int = 128, vocab_size: int = 1000):
        super().__init__()
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size, device="cuda", dtype=torch.float16)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, device="cuda", dtype=torch.float16)

    def forward(self, hidden_states, logit_gather_ids=None, seq_len=None):
        # Simulate transformer output
        hidden_states = self.linear1(hidden_states)
        # LM head
        logits = self.lm_head(hidden_states)
        return logits


class TestGatherLogitsBeforeLmHeadOp:
    """Test the custom op directly."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_generate_format(self, batch_size):
        """Test gather op with generate format input [batch, 1, hidden]."""
        hidden_size = 128
        hidden_states = torch.randn(batch_size, 1, hidden_size, device="cuda", dtype=torch.float16)

        # Create gather info: num_tokens_to_gather=batch_size, gather_required=0 (False)
        logits_gather_indices = torch.arange(batch_size, dtype=torch.long, device="cuda")
        logits_gather_info_host = torch.tensor([batch_size, 0], dtype=torch.int32, device="cpu")

        output = torch.ops.auto_deploy.gather_logits_before_lm_head.default(
            hidden_states, logits_gather_indices, logits_gather_info_host
        )

        # Should return [batch, 1, hidden] for generate format (3D shape preserved)
        assert output.shape == (batch_size, 1, hidden_size)
        assert output.dtype == hidden_states.dtype
        assert output.device == hidden_states.device
        torch.testing.assert_close(output, hidden_states)

    @pytest.mark.parametrize("total_tokens", [10, 50, 100])
    def test_packed_format(self, total_tokens):
        """Test gather op with packed format input [1, total_tokens, hidden]."""
        hidden_size = 128
        max_batch_size = 8
        hidden_states = torch.randn(
            1, total_tokens, hidden_size, device="cuda", dtype=torch.float16
        )

        # Create gather indices: gather tokens at indices [0, 5, 10, ...] up to max_batch_size
        num_gather = min(total_tokens, max_batch_size)
        gather_indices = torch.arange(0, num_gather, dtype=torch.long, device="cuda")

        # Create gather info: num_tokens_to_gather=num_gather, gather_required=1 (True)
        logits_gather_info_host = torch.tensor([num_gather, 1], dtype=torch.int32, device="cpu")

        output = torch.ops.auto_deploy.gather_logits_before_lm_head.default(
            hidden_states, gather_indices, logits_gather_info_host
        )

        # Should return [1, num_gather, hidden] for packed format (3D shape preserved)
        assert output.shape == (1, num_gather, hidden_size)
        assert output.dtype == hidden_states.dtype
        assert output.device == hidden_states.device

        # Verify gathered values match expected indices
        expected = hidden_states[:, gather_indices, :]
        torch.testing.assert_close(output, expected)

    def test_fake_implementation_generate_format(self):
        """Test fake implementation for generate format."""
        batch_size = 4
        hidden_size = 128
        hidden_states = torch.randn(batch_size, 1, hidden_size, device="cuda", dtype=torch.float16)

        # Create gather info
        logits_gather_indices = torch.arange(batch_size, dtype=torch.long, device="cuda")
        logits_gather_info_host = torch.tensor([batch_size, 0], dtype=torch.int32, device="cpu")

        # Use fake implementation directly
        with FakeTensorMode() as mode:
            hidden_states_fake = mode.from_tensor(hidden_states)
            logits_gather_indices_fake = mode.from_tensor(logits_gather_indices)
            logits_gather_info_host_fake = mode.from_tensor(logits_gather_info_host)
            output = torch.ops.auto_deploy.gather_logits_before_lm_head.default(
                hidden_states_fake, logits_gather_indices_fake, logits_gather_info_host_fake
            )

        # Should return [batch, 1, hidden_size] (fake returns empty_like which preserves 3D shape)
        assert output.shape == (batch_size, 1, hidden_size)
        assert output.dtype == hidden_states.dtype
        assert output.device == hidden_states.device

    def test_fake_implementation_packed_format(self):
        """Test fake implementation for packed format."""
        total_tokens = 50
        hidden_size = 128
        num_gather = 8
        hidden_states = torch.randn(
            1, total_tokens, hidden_size, device="cuda", dtype=torch.float16
        )

        # Create gather info
        logits_gather_indices = torch.arange(num_gather, dtype=torch.long, device="cuda")
        logits_gather_info_host = torch.tensor([num_gather, 1], dtype=torch.int32, device="cpu")

        # Use fake implementation directly
        with FakeTensorMode() as mode:
            hidden_states_fake = mode.from_tensor(hidden_states)
            logits_gather_indices_fake = mode.from_tensor(logits_gather_indices)
            logits_gather_info_host_fake = mode.from_tensor(logits_gather_info_host)
            output = torch.ops.auto_deploy.gather_logits_before_lm_head.default(
                hidden_states_fake, logits_gather_indices_fake, logits_gather_info_host_fake
            )

        # The fake implementation returns empty_like which preserves input shape [1, total_tokens, hidden]
        assert output.shape == (1, total_tokens, hidden_size)
        assert output.dtype == hidden_states.dtype
        assert output.device == hidden_states.device


class TestGatherLogitsBeforeLmHeadTransform:
    """Test the transform application."""

    def _create_cached_sequence_interface(self, max_batch_size: int = 8, device: str = "cuda"):
        """Create a mock CachedSequenceInterface for testing."""
        return CachedSequenceInterface(
            max_seq_len=64,
            max_batch_size=max_batch_size,
            device=device,
            max_num_tokens=1024,
        )

    def _check_gather_op_in_graph(self, gm: GraphModule) -> bool:
        """Check if gather_logits_before_lm_head op is in the graph."""
        return any(
            is_op(n, torch.ops.auto_deploy.gather_logits_before_lm_head) for n in gm.graph.nodes
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_transform_generate_format(self, batch_size):
        """Test transform with generate format input."""
        hidden_size = 128
        vocab_size = 1000
        model = SimpleLMHeadModel(hidden_size, vocab_size).cuda()

        # Create input in generate format [batch, 1, hidden]
        hidden_states = torch.randn(batch_size, 1, hidden_size, device="cuda", dtype=torch.float16)
        max_batch_size = 8
        logit_gather_ids = torch.zeros(max_batch_size, dtype=torch.long, device="cuda")
        seq_len = torch.ones(batch_size, dtype=torch.long, device="cuda")

        # Export model
        # When batch_size=1, torch.export specializes it to a constant, so we skip dynamic shapes
        # For batch_size > 1, we use dynamic shapes to test the transform with varying batch sizes
        if batch_size == 1:
            dynamic_shapes = None
        else:
            # dynamic_shapes should be a tuple matching the number of positional args
            dynamic_shapes = (
                {0: Dim.DYNAMIC},  # hidden_states
                None,  # logit_gather_ids (static)
                None,  # seq_len (static)
            )
        gm = torch_export_to_gm(
            model,
            args=(hidden_states, logit_gather_ids, seq_len),
            dynamic_shapes=dynamic_shapes,
            clone=True,
        )

        # Apply transform
        cm = self._create_cached_sequence_interface(max_batch_size)
        transform_config = {
            "gather_logits_before_lm_head": {
                "stage": "post_load_fusion",
                "max_batch_size": max_batch_size,
            }
        }
        optimizer = InferenceOptimizer(None, transform_config)
        gm_transformed = optimizer(cm, gm)

        # Check that gather op was inserted
        assert self._check_gather_op_in_graph(gm_transformed), "Gather op not found in graph"

        # Test forward pass
        # We must pass the new graph inputs manually since we are running the graph directly
        logits_gather_indices = torch.arange(batch_size, dtype=torch.long, device="cuda")
        logits_gather_info_host = torch.tensor([batch_size, 0], dtype=torch.int32, device="cpu")
        output = gm_transformed(
            hidden_states,
            logit_gather_ids,
            seq_len,
            logits_gather_indices=logits_gather_indices,
            logits_gather_info_host=logits_gather_info_host,
        )
        # Output should be [batch_size, 1, vocab_size] since gather now returns 3D
        assert output.shape == (batch_size, 1, vocab_size)

    @pytest.mark.parametrize("total_tokens", [10, 50])
    def test_transform_packed_format(self, total_tokens):
        """Test transform with packed format input."""
        hidden_size = 128
        vocab_size = 1000
        max_batch_size = 8
        model = SimpleLMHeadModel(hidden_size, vocab_size).cuda()

        # Create input in packed format [1, total_tokens, hidden]
        hidden_states = torch.randn(
            1, total_tokens, hidden_size, device="cuda", dtype=torch.float16
        )
        logit_gather_ids = torch.arange(
            0, min(total_tokens, max_batch_size), dtype=torch.long, device="cuda"
        )
        # Pad to max_batch_size
        logit_gather_ids_padded = torch.zeros(max_batch_size, dtype=torch.long, device="cuda")
        logit_gather_ids_padded[: len(logit_gather_ids)] = logit_gather_ids

        seq_len = torch.ones(max_batch_size, dtype=torch.long, device="cuda")
        seq_len[: len(logit_gather_ids)] = torch.ones(
            len(logit_gather_ids), dtype=torch.long, device="cuda"
        )

        # Export model
        gm = torch_export_to_gm(
            model,
            args=(hidden_states, logit_gather_ids_padded, seq_len),
            dynamic_shapes=None,
            clone=True,
        )

        # Apply transform
        cm = self._create_cached_sequence_interface(max_batch_size)
        transform_config = {
            "gather_logits_before_lm_head": {
                "stage": "post_load_fusion",
                "max_batch_size": max_batch_size,
            }
        }
        optimizer = InferenceOptimizer(None, transform_config)
        gm_transformed = optimizer(cm, gm)

        # Check that gather op was inserted
        assert self._check_gather_op_in_graph(gm_transformed), "Gather op not found in graph"

        # Test forward pass
        # We must pass the new graph inputs manually since we are running the graph directly
        num_gather = len(logit_gather_ids)
        logits_gather_indices = logit_gather_ids
        logits_gather_info_host = torch.tensor([num_gather, 1], dtype=torch.int32, device="cpu")
        output = gm_transformed(
            hidden_states,
            logit_gather_ids_padded,
            seq_len,
            logits_gather_indices=logits_gather_indices,
            logits_gather_info_host=logits_gather_info_host,
        )
        # Output should be [1, num_gather, vocab_size] since gather now returns 3D
        assert output.shape == (1, num_gather, vocab_size)

    def test_transform_skips_when_disabled(self):
        """Test that transform skips when disabled."""
        hidden_size = 128
        vocab_size = 1000
        model = SimpleLMHeadModel(hidden_size, vocab_size).cuda()

        hidden_states = torch.randn(4, 1, hidden_size, device="cuda", dtype=torch.float16)
        max_batch_size = 8
        logit_gather_ids = torch.zeros(max_batch_size, dtype=torch.long, device="cuda")
        seq_len = torch.ones(4, dtype=torch.long, device="cuda")

        # Export model
        gm = torch_export_to_gm(
            model,
            args=(hidden_states, logit_gather_ids, seq_len),
            dynamic_shapes=None,
            clone=True,
        )

        # Apply transform with disabled config
        cm = self._create_cached_sequence_interface(max_batch_size)
        transform_config = {
            "gather_logits_before_lm_head": {
                "stage": "post_load_fusion",
                "enabled": False,
                "max_batch_size": max_batch_size,
            }
        }
        optimizer = InferenceOptimizer(None, transform_config)
        gm_transformed = optimizer(cm, gm)

        # Check that gather op was NOT inserted
        assert not self._check_gather_op_in_graph(gm_transformed), (
            "Gather op should not be in graph"
        )
