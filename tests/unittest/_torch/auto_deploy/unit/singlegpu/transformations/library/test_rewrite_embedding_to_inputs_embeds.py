# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for rewrite_embedding_to_inputs_embeds transformation.

This transform rewrites the graph to accept inputs_embeds instead of input_ids,
which is necessary for EdgeLLM to support multimodal models where embedding
lookup is performed at runtime.
"""

import inspect
import tempfile
from pathlib import Path

import pytest
import safetensors.torch
import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer

torch.manual_seed(0)


class EmbeddingModel(torch.nn.Module):
    """
    Simple model with embedding layer followed by a linear projection.

    This model represents the minimal pattern that rewrite_embedding_to_inputs_embeds
    expects to transform:
        input_ids → embedding(weight, input_ids) → subsequent operations
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        """Initialize EmbeddingModel with embedding and linear projection layers."""
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] int tensor

        Returns:
            [batch_size, seq_len, hidden_size] float tensor
        """
        x = self.embed_tokens(input_ids)  # [batch, seq, hidden]
        return self.proj(x)

    def get_dynamic_shapes(self):
        """Return dynamic shape specifications for torch.export."""
        return [
            {0: Dim("batch_size", max=8), 1: Dim("seq_len", max=128)},
        ]


def _run_test(
    vocab_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
):
    """Helper function to run the transformation test."""
    # Create model in fp16 (pre-condition for the transform)
    model = EmbeddingModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    ).to("cuda", torch.float16)

    # Create input tensors
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.int32
    )
    dynamic_shapes = model.get_dynamic_shapes()

    # Export to graph module
    args = (input_ids,)
    gm = torch_export_to_gm(model, args=args, dynamic_shapes=dynamic_shapes, clone=True)

    # Verify pre-conditions: input_ids placeholder exists, embedding node exists
    placeholder_nodes_before = gm.graph.find_nodes(op="placeholder")
    placeholder_names_before = {node.target for node in placeholder_nodes_before}
    assert "input_ids" in placeholder_names_before, "Pre-condition: input_ids should exist"

    embedding_nodes_before = gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.embedding.default
    )
    assert len(embedding_nodes_before) == 1, "Pre-condition: embedding node should exist"

    # Create temp directory for safetensors output
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)

        # Apply rewrite_embedding_to_inputs_embeds transformation
        # Note: factory=None because rewrite_embedding_to_inputs_embeds does not use it
        optimizer = InferenceOptimizer(
            None,
            {
                "rewrite_embedding_to_inputs_embeds": {
                    "stage": "pattern_matcher",
                    "output_dir": str(output_dir),
                },
            },
        )
        gm_transformed = optimizer(None, gm)

        # =========================================================================
        # Assertions
        # =========================================================================

        # 1. Verify input_ids placeholder is removed
        placeholder_nodes_after = gm_transformed.graph.find_nodes(op="placeholder")
        placeholder_names_after = {node.target for node in placeholder_nodes_after}
        assert "input_ids" not in placeholder_names_after, (
            f"input_ids should be removed, but found placeholders: {placeholder_names_after}"
        )

        # 2. Verify inputs_embeds placeholder is added
        assert "inputs_embeds" in placeholder_names_after, (
            f"inputs_embeds should be added, but found placeholders: {placeholder_names_after}"
        )

        # 3. Verify inputs_embeds has correct dtype and shape
        inputs_embeds_node = None
        for node in placeholder_nodes_after:
            if node.target == "inputs_embeds":
                inputs_embeds_node = node
                break

        assert inputs_embeds_node is not None, "inputs_embeds node not found"
        inputs_embeds_meta = inputs_embeds_node.meta.get("val")
        assert inputs_embeds_meta is not None, "inputs_embeds should have meta['val']"
        assert inputs_embeds_meta.dtype == torch.float16, (
            f"inputs_embeds dtype should be float16, got {inputs_embeds_meta.dtype}"
        )
        # Shape should be [batch_size, seq_len, hidden_size]
        assert len(inputs_embeds_meta.shape) == 3, (
            f"inputs_embeds should have 3 dimensions, got {len(inputs_embeds_meta.shape)}"
        )

        # 4. Verify embedding node is removed
        embedding_nodes_after = gm_transformed.graph.find_nodes(
            op="call_function", target=torch.ops.aten.embedding.default
        )
        assert len(embedding_nodes_after) == 0, (
            f"embedding node should be removed, but found {len(embedding_nodes_after)}"
        )

        # 5. Verify safetensors file is created with correct content
        safetensors_path = output_dir / "embedding.safetensors"
        assert safetensors_path.exists(), f"safetensors file should exist at {safetensors_path}"

        # Load and verify the safetensors content
        loaded_weights = safetensors.torch.load_file(str(safetensors_path))
        assert "weight" in loaded_weights, "safetensors should contain 'weight' key"

        weight_tensor = loaded_weights["weight"]
        assert weight_tensor.dtype == torch.float16, (
            f"weight dtype should be float16, got {weight_tensor.dtype}"
        )
        assert weight_tensor.shape == (vocab_size, hidden_size), (
            f"weight shape should be ({vocab_size}, {hidden_size}), got {weight_tensor.shape}"
        )

        # 6. Verify the graph signature is updated (recompile was called)

        sig = inspect.signature(gm_transformed.forward)
        param_names = list(sig.parameters.keys())
        assert "input_ids" not in param_names, "input_ids should not be in signature"
        assert "inputs_embeds" in param_names, "inputs_embeds should be in signature"


@pytest.mark.parametrize(
    "vocab_size,hidden_size,batch_size,seq_len",
    [
        pytest.param(1000, 64, 2, 16, id="small_model"),
        pytest.param(32000, 128, 4, 32, id="medium_model"),
    ],
)
@torch.inference_mode()
def test_rewrite_embedding_to_inputs_embeds(
    vocab_size: int,
    hidden_size: int,
    batch_size: int,
    seq_len: int,
):
    """Test rewrite_embedding_to_inputs_embeds transformation with various configurations."""
    _run_test(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        batch_size=batch_size,
        seq_len=seq_len,
    )


@torch.inference_mode()
def test_rewrite_embedding_skips_when_no_input_ids():
    """Test that the transform is skipped when input_ids placeholder doesn't exist."""

    # Create a model that doesn't use input_ids naming
    class NoInputIdsModel(torch.nn.Module):
        """Simple model without input_ids placeholder to test skip behavior."""

        def __init__(self, hidden_size: int):
            """Initialize with a single linear projection layer."""
            super().__init__()
            self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the linear projection."""
            return self.proj(x)

    hidden_size = 64
    batch_size = 2
    seq_len = 16

    model = NoInputIdsModel(hidden_size).to("cuda", torch.float16)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)

    # hidden_size is static, only batch and seq are dynamic
    dynamic_shapes = [{0: Dim("batch_size", max=8), 1: Dim("seq_len", max=128)}]
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=dynamic_shapes, clone=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)

        optimizer = InferenceOptimizer(
            None,
            {
                "rewrite_embedding_to_inputs_embeds": {
                    "stage": "pattern_matcher",
                    "output_dir": str(output_dir),
                },
            },
        )
        gm_transformed = optimizer(None, gm)

        # Should be skipped - no safetensors file created
        safetensors_path = output_dir / "embedding.safetensors"
        assert not safetensors_path.exists(), (
            "safetensors should not be created when transform is skipped"
        )

        # Graph should remain unchanged
        placeholder_nodes = gm_transformed.graph.find_nodes(op="placeholder")
        placeholder_names = {node.target for node in placeholder_nodes}
        assert "input_ids" not in placeholder_names
        assert "inputs_embeds" not in placeholder_names
