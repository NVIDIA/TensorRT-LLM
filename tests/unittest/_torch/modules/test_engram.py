# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the Engram module."""

import pytest
import torch

from tensorrt_llm._torch.modules.engram import Engram, EngramConfig, EngramHashProvider
from tensorrt_llm._torch.modules.engram.engram import (
    CompressedTokenizer,
    MultiHeadEmbedding,
    NgramHashMapping,
    ShortConv,
)


class TestEngramConfig:
    """Test suite for EngramConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EngramConfig()
        assert config.max_ngram_size == 3
        assert config.n_embed_per_ngram == 512
        assert config.n_head_per_ngram == 8
        assert config.layer_ids == [1, 15]
        assert config.pad_id == 2
        assert config.seed == 0
        assert config.kernel_size == 4
        assert config.hidden_size == 1024
        assert config.hc_mult == 4
        assert config.norm_eps == 1e-5

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EngramConfig(
            tokenizer_name_or_path="gpt2",
            hidden_size=256,
            hc_mult=2,
            layer_ids=[0, 5],
        )
        assert config.tokenizer_name_or_path == "gpt2"
        assert config.hidden_size == 256
        assert config.hc_mult == 2
        assert config.layer_ids == [0, 5]


class TestMultiHeadEmbedding:
    """Test suite for MultiHeadEmbedding."""

    def test_forward_shape(self):
        """Test forward pass output shape."""
        list_of_N = [100, 200, 150]
        D = 64
        module = MultiHeadEmbedding(list_of_N=list_of_N, D=D)

        seq_len = 32
        num_heads = len(list_of_N)

        # Create input indices within valid range for each head
        input_ids = torch.stack(
            [torch.randint(0, list_of_N[i], (seq_len,)) for i in range(num_heads)],
            dim=-1,
        )

        output = module(input_ids)

        assert output.shape == (seq_len, num_heads, D)

    def test_offset_handling(self):
        """Test that offsets are correctly applied."""
        list_of_N = [10, 20, 30]
        D = 8
        module = MultiHeadEmbedding(list_of_N=list_of_N, D=D)

        expected_offsets = torch.tensor([0, 10, 30])
        torch.testing.assert_close(module.offsets, expected_offsets)

        # Total embedding size should be sum of all N
        assert module.embedding.num_embeddings == sum(list_of_N)


class TestShortConv:
    """Test suite for ShortConv."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_forward_shape(self, dtype):
        """Test forward pass output shape."""
        hidden_size = 64
        hc_mult = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
        module = ShortConv(hidden_size=hidden_size, hc_mult=hc_mult, dtype=dtype).to(device)

        seq_len = 32
        x = torch.randn(seq_len, hc_mult, hidden_size, dtype=dtype, device=device)

        output = module(x)

        assert output.shape == x.shape
        assert output.dtype == dtype

    def test_causal_masking(self):
        """Test that convolution is causal (output at t depends only on inputs <= t)."""
        hidden_size = 32
        hc_mult = 2
        kernel_size = 4
        dtype = torch.bfloat16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        module = ShortConv(
            hidden_size=hidden_size, kernel_size=kernel_size, hc_mult=hc_mult, dtype=dtype
        ).to(device)
        module.eval()

        seq_len = 16
        x = torch.randn(seq_len, hc_mult, hidden_size, dtype=dtype, device=device)

        with torch.no_grad():
            output_full = module(x)

            # Run on truncated input and compare
            for t in range(1, seq_len):
                x_truncated = x[:t, :, :]
                output_truncated = module(x_truncated)
                # Output at position t-1 should match
                torch.testing.assert_close(
                    output_truncated[-1, :, :], output_full[t - 1, :, :], rtol=1e-3, atol=1e-3
                )

    @pytest.mark.parametrize("activation", [True, False])
    def test_activation_flag(self, activation):
        """Test activation flag."""
        module = ShortConv(hidden_size=32, hc_mult=2, activation=activation)
        assert hasattr(module, "act_fn") == activation


class TestNgramHashMapping:
    """Test suite for NgramHashMapping."""

    @pytest.fixture
    def hash_mapping(self):
        """Create a hash mapping instance for testing."""
        return NgramHashMapping(
            engram_vocab_size=[1000, 1000],
            max_ngram_size=3,
            n_embed_per_ngram=64,
            n_head_per_ngram=4,
            layer_ids=[0, 1],
            tokenizer_name_or_path="gpt2",
            pad_id=0,
            seed=42,
        )

    def test_hash_output_shape(self, hash_mapping):
        """Test hash output shape."""
        seq_len = 32
        # GPT-2 vocab size is 50257
        input_ids = torch.randint(0, 50257, (seq_len,))

        result = hash_mapping.hash(input_ids)

        assert 0 in result
        assert 1 in result

        # num_heads = (max_ngram_size - 1) * n_head_per_ngram = 2 * 4 = 8
        expected_num_heads = (hash_mapping.max_ngram_size - 1) * hash_mapping.n_head_per_ngram
        for layer_id in [0, 1]:
            assert result[layer_id].shape == (seq_len, expected_num_heads)

    def test_deterministic_hashing(self, hash_mapping):
        """Test that hashing is deterministic."""
        input_ids = torch.randint(0, 50257, (32,))

        result1 = hash_mapping.hash(input_ids)
        result2 = hash_mapping.hash(input_ids)

        for layer_id in hash_mapping.layer_ids:
            assert (result1[layer_id] == result2[layer_id]).all()

    def test_different_seeds_produce_different_hashes(self):
        """Test that different seeds produce different hash mappings."""
        common_args = dict(
            engram_vocab_size=[1000, 1000],
            max_ngram_size=3,
            n_embed_per_ngram=64,
            n_head_per_ngram=4,
            layer_ids=[0],
            tokenizer_name_or_path="gpt2",
            pad_id=0,
        )

        mapping1 = NgramHashMapping(**common_args, seed=42)
        mapping2 = NgramHashMapping(**common_args, seed=123)

        input_ids = torch.randint(0, 50257, (32,))

        result1 = mapping1.hash(input_ids)[0]
        result2 = mapping2.hash(input_ids)[0]

        # Results should differ for different seeds
        assert not (result1 == result2).all()


def _make_engram_config(**overrides):
    """Helper to create a test EngramConfig with small sizes."""
    defaults = dict(
        tokenizer_name_or_path="gpt2",
        engram_vocab_size=[1000, 1000],
        max_ngram_size=3,
        n_embed_per_ngram=64,
        n_head_per_ngram=4,
        layer_ids=[0, 1],
        pad_id=0,
        seed=42,
        kernel_size=4,
        hidden_size=128,
        hc_mult=4,
        norm_eps=1e-5,
        dtype=torch.bfloat16,
    )
    defaults.update(overrides)
    return EngramConfig(**defaults)


def _precompute_embeddings(module, config, input_ids, dtype, device="cuda"):
    """Helper: compute hash indices and precompute embeddings for an Engram module."""
    hash_provider = EngramHashProvider(config)
    hash_cache = hash_provider.compute_hashes(input_ids)
    hash_indices = hash_cache[module.layer_id].to(device)
    return module.precompute(hash_indices, dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEngram:
    """Test suite for the Engram module."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return _make_engram_config()

    def test_forward_shape(self, config):
        """Test forward pass output shape."""
        device = "cuda"
        dtype = config.dtype
        module = Engram(layer_id=0, config=config).to(device)

        seq_len = 32
        hidden_states = torch.randn(
            seq_len, config.hc_mult, config.hidden_size, device=device, dtype=dtype
        )
        input_ids = torch.randint(0, 50257, (seq_len,))
        embeddings = _precompute_embeddings(module, config, input_ids, dtype, device)

        output = module(hidden_states, embeddings)

        assert output.shape == hidden_states.shape

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_dtype_support(self, config, dtype):
        """Test support for multiple dtypes."""
        device = "cuda"
        config_with_dtype = _make_engram_config(dtype=dtype)
        module = Engram(layer_id=0, config=config_with_dtype).to(device)

        seq_len = 16
        hidden_states = torch.randn(
            seq_len, config.hc_mult, config.hidden_size, device=device, dtype=dtype
        )
        input_ids = torch.randint(0, 50257, (seq_len,))
        embeddings = _precompute_embeddings(module, config_with_dtype, input_ids, dtype, device)

        output = module(hidden_states, embeddings)

        assert output.dtype == dtype

    def test_deterministic_forward(self, config):
        """Test that forward pass is deterministic."""
        device = "cuda"
        dtype = config.dtype
        module = Engram(layer_id=0, config=config).to(device)
        module.eval()

        seq_len = 16
        hidden_states = torch.randn(
            seq_len, config.hc_mult, config.hidden_size, device=device, dtype=dtype
        )
        input_ids = torch.randint(0, 50257, (seq_len,))
        embeddings = _precompute_embeddings(module, config, input_ids, dtype, device)

        with torch.no_grad():
            output1 = module(hidden_states.clone(), embeddings.clone())
            output2 = module(hidden_states.clone(), embeddings.clone())

        torch.testing.assert_close(output1, output2)

    def test_residual_pattern(self, config):
        """Test that output can be used as residual addition."""
        device = "cuda"
        dtype = config.dtype
        module = Engram(layer_id=0, config=config).to(device)

        seq_len = 16
        hidden_states = torch.randn(
            seq_len, config.hc_mult, config.hidden_size, device=device, dtype=dtype
        )
        input_ids = torch.randint(0, 50257, (seq_len,))
        embeddings = _precompute_embeddings(module, config, input_ids, dtype, device)

        output = module(hidden_states, embeddings)

        # Should be able to add as residual
        result = hidden_states + output

        assert result.shape == hidden_states.shape
        # Result should be different from input (non-zero output)
        assert not torch.allclose(result, hidden_states, atol=1e-6)

    def test_different_layer_ids(self, config):
        """Test that different layer IDs produce different hash mappings."""
        device = "cuda"
        module0 = Engram(layer_id=0, config=config).to(device)
        module1 = Engram(layer_id=1, config=config).to(device)

        # Note: We cannot copy weights between modules because each layer has
        # different embedding sizes (due to unique prime moduli per layer/head).
        # Instead, we verify that the hash mappings themselves differ.

        seq_len = 16
        input_ids = torch.randint(0, 50257, (seq_len,))

        # Verify that hash mappings produce different results for different layers
        hash0 = module0.hash_mapping.hash(input_ids)[0]
        hash1 = module1.hash_mapping.hash(input_ids)[1]

        # Hash indices should differ between layers
        assert not (hash0 == hash1).all()

    def test_different_seq_lengths(self, config):
        """Test that the module handles different sequence lengths correctly."""
        device = "cuda"
        dtype = config.dtype
        module = Engram(layer_id=0, config=config).to(device)
        module.eval()

        for seq_len in [1, 8, 32]:
            hidden_states = torch.randn(
                seq_len, config.hc_mult, config.hidden_size, device=device, dtype=dtype
            )
            input_ids = torch.randint(0, 50257, (seq_len,))
            embeddings = _precompute_embeddings(module, config, input_ids, dtype, device)

            with torch.no_grad():
                output = module(hidden_states, embeddings)

            assert output.shape == hidden_states.shape


class TestCompressedTokenizer:
    """Test suite for CompressedTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a compressed tokenizer for testing."""
        return CompressedTokenizer("gpt2")

    def test_compression_reduces_vocab(self, tokenizer):
        """Test that compression reduces vocabulary size."""
        # GPT-2 has 50257 tokens
        assert len(tokenizer) < 50257

    def test_lookup_table_shape(self, tokenizer):
        """Test lookup table shape matches original vocab."""
        assert tokenizer.lookup_table.shape[0] == 50257

    def test_call_returns_compressed_ids(self, tokenizer):
        """Test that calling tokenizer returns compressed IDs."""
        input_ids = [[100, 200, 300], [400, 500, 600]]
        compressed = tokenizer(input_ids)

        assert compressed.shape == (2, 3)
        # All compressed IDs should be less than num_new_token
        assert (compressed < len(tokenizer)).all()

    def test_negative_ids_preserved(self, tokenizer):
        """Test that negative IDs (padding) are preserved."""
        input_ids = [[100, -1, 200], [-1, -1, 300]]
        compressed = tokenizer(input_ids)

        assert compressed[0, 1] == -1
        assert compressed[1, 0] == -1
        assert compressed[1, 1] == -1


class TestEngramHashProvider:
    """Test suite for EngramHashProvider."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return _make_engram_config(dtype=None)

    @pytest.fixture
    def hash_provider(self, config):
        """Create a hash provider for testing."""
        return EngramHashProvider(config)

    def test_compute_hashes_returns_dict(self, hash_provider):
        """Test that compute_hashes returns dict with correct keys."""
        input_ids = torch.randint(0, 50257, (32,))
        hash_cache = hash_provider.compute_hashes(input_ids)

        assert isinstance(hash_cache, dict)
        assert 0 in hash_cache
        assert 1 in hash_cache

    def test_compute_hashes_shape(self, hash_provider, config):
        """Test that hash tensors have correct shape."""
        seq_len = 32
        input_ids = torch.randint(0, 50257, (seq_len,))
        hash_cache = hash_provider.compute_hashes(input_ids)

        expected_num_heads = (config.max_ngram_size - 1) * config.n_head_per_ngram
        for layer_id in [0, 1]:
            assert hash_cache[layer_id].shape == (seq_len, expected_num_heads)
            assert hash_cache[layer_id].dtype == torch.int64

    def test_compute_hashes_on_cpu(self, hash_provider):
        """Test that hash tensors are on CPU when input is CPU."""
        input_ids = torch.randint(0, 50257, (32,))
        hash_cache = hash_provider.compute_hashes(input_ids)

        for _, hashes in hash_cache.items():
            assert hashes.device == torch.device("cpu")

    def test_compute_hashes_deterministic(self, hash_provider):
        """Test that hash computation is deterministic."""
        input_ids = torch.randint(0, 50257, (32,))

        hash_cache_1 = hash_provider.compute_hashes(input_ids)
        hash_cache_2 = hash_provider.compute_hashes(input_ids)

        for layer_id in hash_provider.layer_ids:
            torch.testing.assert_close(hash_cache_1[layer_id], hash_cache_2[layer_id])

    def test_layer_ids_property(self, hash_provider, config):
        """Test layer_ids property."""
        assert hash_provider.layer_ids == config.layer_ids

    def test_vocab_size_across_layers(self, hash_provider, config):
        """Test vocab_size_across_layers property."""
        vocab_sizes = hash_provider.vocab_size_across_layers
        assert isinstance(vocab_sizes, dict)
        for layer_id in config.layer_ids:
            assert layer_id in vocab_sizes
            # Should have (max_ngram_size - 1) n-gram levels
            assert len(vocab_sizes[layer_id]) == config.max_ngram_size - 1
            for ngram_sizes in vocab_sizes[layer_id]:
                assert len(ngram_sizes) == config.n_head_per_ngram


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEngramWithHashProvider:
    """Test suite for Engram with EngramHashProvider integration."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return _make_engram_config()

    def test_forward_with_precomputed_embeddings(self, config):
        """Test forward pass using pre-computed embeddings from hash provider."""
        device = "cuda"
        dtype = config.dtype
        hash_provider = EngramHashProvider(config)

        # Create Engram with vocab sizes from hash provider
        vocab_sizes = [x for y in hash_provider.vocab_size_across_layers[0] for x in y]
        module = Engram(
            layer_id=0,
            config=config,
            vocab_sizes_flat=vocab_sizes,
        ).to(device)

        seq_len = 32
        hidden_states = torch.randn(
            seq_len, config.hc_mult, config.hidden_size, device=device, dtype=dtype
        )
        input_ids = torch.randint(0, 50257, (seq_len,))

        # Compute hashes and precompute embeddings
        hash_cache = hash_provider.compute_hashes(input_ids)
        embeddings = module.precompute(hash_cache[0].to(device), dtype=dtype)

        # Forward with pre-computed embeddings
        output = module(hidden_states, embeddings)

        assert output.shape == hidden_states.shape

    def test_precompute_output_shape(self, config):
        """Test that precompute returns correct shape."""
        device = "cuda"
        dtype = config.dtype
        hash_provider = EngramHashProvider(config)

        vocab_sizes = [x for y in hash_provider.vocab_size_across_layers[0] for x in y]
        module = Engram(
            layer_id=0,
            config=config,
            vocab_sizes_flat=vocab_sizes,
        ).to(device)

        seq_len = 32
        input_ids = torch.randint(0, 50257, (seq_len,))
        hash_cache = hash_provider.compute_hashes(input_ids)
        embeddings = module.precompute(hash_cache[0].to(device), dtype=dtype)

        # Embeddings should be [T, (max_ngram_size - 1) * n_embed_per_ngram]
        expected_dim = (config.max_ngram_size - 1) * config.n_embed_per_ngram
        assert embeddings.shape == (seq_len, expected_dim)
        assert embeddings.dtype == dtype

    def test_standalone_vs_provider_equivalence(self, config):
        """Test that standalone mode and provider mode produce same results."""
        device = "cuda"
        dtype = config.dtype

        # Create standalone module (with internal hash_mapping)
        module_standalone = Engram(layer_id=0, config=config).to(device)
        module_standalone.eval()

        # Create provider-based module with same weights
        hash_provider = EngramHashProvider(config)
        vocab_sizes = [x for y in hash_provider.vocab_size_across_layers[0] for x in y]
        module_provider = Engram(
            layer_id=0,
            config=config,
            vocab_sizes_flat=vocab_sizes,
        ).to(device)
        module_provider.eval()

        # Copy weights from standalone to provider-based
        module_provider.load_state_dict(module_standalone.state_dict())

        seq_len = 32
        hidden_states = torch.randn(
            seq_len, config.hc_mult, config.hidden_size, device=device, dtype=dtype
        )
        input_ids = torch.randint(0, 50257, (seq_len,))

        # Standalone: use internal hash_mapping + precompute
        hash_standalone = module_standalone.hash_mapping.hash(input_ids)
        hash_indices_standalone = hash_standalone[0].to(device)
        embeddings_standalone = module_standalone.precompute(hash_indices_standalone, dtype=dtype)

        # Provider: use hash_provider + precompute (1-D input)
        hash_cache = hash_provider.compute_hashes(input_ids)
        embeddings_provider = module_provider.precompute(hash_cache[0].to(device), dtype=dtype)

        with torch.no_grad():
            output_standalone = module_standalone(hidden_states.clone(), embeddings_standalone)
            output_provider = module_provider(hidden_states.clone(), embeddings_provider)

        torch.testing.assert_close(output_standalone, output_provider)

    def test_no_hash_mapping_when_vocab_sizes_provided(self, config):
        """Test that hash_mapping is None when vocab_sizes_flat is provided."""
        hash_provider = EngramHashProvider(config)
        vocab_sizes = [x for y in hash_provider.vocab_size_across_layers[0] for x in y]

        module = Engram(
            layer_id=0,
            config=config,
            vocab_sizes_flat=vocab_sizes,
        )

        assert module.hash_mapping is None
