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
"""
Module-level tests for GPT-OSS Embedding and LMHead modules.

Tests verify that:
1. TRT-LLM Embedding produces the same output as HF nn.Embedding given the same weights.
2. TRT-LLM LMHead produces the same output as HF nn.Linear given the same weights.
3. Weight loading from checkpoint is correct.
"""

import json
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_PATH = "/scratch.trt_llm_data/llm-models/gpt_oss/gpt-oss-20b/"

# Model config values from config.json
VOCAB_SIZE = 201088
HIDDEN_SIZE = 2880
DTYPE = torch.bfloat16


def load_weight_from_checkpoint(weight_name: str) -> torch.Tensor:
    """Load a single weight tensor from the sharded safetensors checkpoint."""
    from safetensors.torch import load_file

    index_path = os.path.join(CHECKPOINT_PATH, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    shard_file = index["weight_map"][weight_name]
    shard_path = os.path.join(CHECKPOINT_PATH, shard_file)
    weights = load_file(shard_path)
    return weights[weight_name]


@pytest.fixture(scope="module")
def embed_weight():
    """Load embedding weight from checkpoint."""
    return load_weight_from_checkpoint("model.embed_tokens.weight")


@pytest.fixture(scope="module")
def lm_head_weight():
    """Load lm_head weight from checkpoint."""
    return load_weight_from_checkpoint("lm_head.weight")


class TestEmbedding:
    """Tests for the Embedding module."""

    def test_embedding_weight_shape(self, embed_weight):
        """Verify embedding weight has expected shape."""
        assert embed_weight.shape == (VOCAB_SIZE, HIDDEN_SIZE), (
            f"Expected shape ({VOCAB_SIZE}, {HIDDEN_SIZE}), "
            f"got {embed_weight.shape}")

    def test_embedding_weight_dtype(self, embed_weight):
        """Verify embedding weight is BF16."""
        assert embed_weight.dtype == DTYPE, (
            f"Expected dtype {DTYPE}, got {embed_weight.dtype}")

    def test_embedding_output_matches_hf(self, embed_weight):
        """Compare TRT-LLM Embedding output against HF nn.Embedding."""
        from tensorrt_llm._torch.modules.embedding import Embedding as TrtEmbedding

        # Create HF embedding
        hf_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE).to(dtype=DTYPE,
                                                             device="cuda")
        hf_embed.weight.data.copy_(embed_weight)

        # Create TRT-LLM embedding (no TP, no mapping)
        trt_embed = TrtEmbedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=HIDDEN_SIZE,
            dtype=DTYPE,
        ).cuda()
        # Load weight directly
        trt_embed.weight.data.copy_(embed_weight)

        # Test with various input token IDs
        test_inputs = [
            torch.tensor([0, 1, 100, 1000, 200000], device="cuda"),
            torch.tensor([42, 12345, 99999], device="cuda"),
            torch.randint(0, VOCAB_SIZE, (8,), device="cuda"),
        ]

        for input_ids in test_inputs:
            hf_out = hf_embed(input_ids)
            trt_out = trt_embed(input_ids)

            assert torch.allclose(hf_out, trt_out, atol=0, rtol=0), (
                f"Embedding output mismatch for input {input_ids}.\n"
                f"HF output shape: {hf_out.shape}, TRT output shape: {trt_out.shape}\n"
                f"Max diff: {(hf_out - trt_out).abs().max().item()}")

    def test_embedding_batch_input(self, embed_weight):
        """Test embedding with batched 2D input."""
        from tensorrt_llm._torch.modules.embedding import Embedding as TrtEmbedding

        hf_embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE).to(dtype=DTYPE,
                                                             device="cuda")
        hf_embed.weight.data.copy_(embed_weight)

        trt_embed = TrtEmbedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=HIDDEN_SIZE,
            dtype=DTYPE,
        ).cuda()
        trt_embed.weight.data.copy_(embed_weight)

        # Batch of sequences
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 16), device="cuda")
        hf_out = hf_embed(input_ids)
        trt_out = trt_embed(input_ids)

        assert torch.allclose(hf_out, trt_out, atol=0, rtol=0), (
            f"Batched embedding output mismatch.\n"
            f"Max diff: {(hf_out - trt_out).abs().max().item()}")


class TestLMHead:
    """Tests for the LMHead module."""

    def test_lm_head_weight_shape(self, lm_head_weight):
        """Verify lm_head weight has expected shape."""
        assert lm_head_weight.shape == (VOCAB_SIZE, HIDDEN_SIZE), (
            f"Expected shape ({VOCAB_SIZE}, {HIDDEN_SIZE}), "
            f"got {lm_head_weight.shape}")

    def test_lm_head_weight_dtype(self, lm_head_weight):
        """Verify lm_head weight is BF16."""
        assert lm_head_weight.dtype == DTYPE, (
            f"Expected dtype {DTYPE}, got {lm_head_weight.dtype}")

    def test_lm_head_output_matches_hf(self, lm_head_weight):
        """Compare TRT-LLM LMHead output against HF nn.Linear (no bias)."""
        from tensorrt_llm._torch.modules.embedding import LMHead as TrtLMHead

        # Create HF linear (no bias, matching MixtralForCausalLM.lm_head)
        hf_lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE,
                                bias=False).to(dtype=DTYPE, device="cuda")
        hf_lm_head.weight.data.copy_(lm_head_weight)

        # Create TRT-LLM LMHead (no TP, no mapping)
        trt_lm_head = TrtLMHead(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=HIDDEN_SIZE,
            dtype=DTYPE,
        ).cuda()
        trt_lm_head.weight.data.copy_(lm_head_weight)

        # Test with random hidden states
        hidden_states = torch.randn(4, HIDDEN_SIZE, dtype=DTYPE, device="cuda")

        hf_out = hf_lm_head(hidden_states)
        trt_out = trt_lm_head(hidden_states)

        # BF16 may have small numerical differences due to matmul implementations
        assert torch.allclose(hf_out, trt_out, atol=1e-2, rtol=1e-2), (
            f"LMHead output mismatch.\n"
            f"HF output shape: {hf_out.shape}, TRT output shape: {trt_out.shape}\n"
            f"Max diff: {(hf_out - trt_out).abs().max().item()}\n"
            f"Mean diff: {(hf_out - trt_out).abs().mean().item()}")

    def test_lm_head_no_bias(self, lm_head_weight):
        """Verify TRT-LLM LMHead has no bias parameter."""
        from tensorrt_llm._torch.modules.embedding import LMHead as TrtLMHead

        trt_lm_head = TrtLMHead(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=HIDDEN_SIZE,
            dtype=DTYPE,
        )
        assert trt_lm_head.bias is None, "LMHead should have no bias"

    def test_lm_head_batch_output(self, lm_head_weight):
        """Test LMHead with batched 3D input (batch, seq_len, hidden)."""
        from tensorrt_llm._torch.modules.embedding import LMHead as TrtLMHead

        hf_lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE,
                                bias=False).to(dtype=DTYPE, device="cuda")
        hf_lm_head.weight.data.copy_(lm_head_weight)

        trt_lm_head = TrtLMHead(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=HIDDEN_SIZE,
            dtype=DTYPE,
        ).cuda()
        trt_lm_head.weight.data.copy_(lm_head_weight)

        hidden_states = torch.randn(2, 8, HIDDEN_SIZE,
                                     dtype=DTYPE,
                                     device="cuda")

        hf_out = hf_lm_head(hidden_states)
        trt_out = trt_lm_head(hidden_states)

        assert torch.allclose(hf_out, trt_out, atol=1e-2, rtol=1e-2), (
            f"Batched LMHead output mismatch.\n"
            f"Max diff: {(hf_out - trt_out).abs().max().item()}")


class TestTiedEmbeddings:
    """Tests verifying tie_word_embeddings=False behavior."""

    def test_weights_are_different(self, embed_weight, lm_head_weight):
        """Verify embedding and lm_head weights are different (not tied)."""
        assert not torch.equal(embed_weight, lm_head_weight), (
            "Embedding and LMHead weights should be different when "
            "tie_word_embeddings=False")

    def test_config_tie_word_embeddings_false(self):
        """Verify config.json has tie_word_embeddings=False."""
        config_path = os.path.join(CHECKPOINT_PATH, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        assert config["tie_word_embeddings"] is False, (
            "Expected tie_word_embeddings=False in config.json")
