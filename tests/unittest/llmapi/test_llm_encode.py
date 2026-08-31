# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import EncodeCudaGraphConfig
from tensorrt_llm.llmapi.llm import EncoderOutput

# isort: off
from .test_llm import get_model_path

# isort: on

BERT_MODEL_PATH = "bert/bert-base-uncased-yelp-polarity"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@pytest.fixture(scope="module")
def bert_encode_llm():
    """Create an LLM with encode_only=True for BERT, shared across tests."""
    model_dir = get_model_path(BERT_MODEL_PATH)
    llm = LLM(model=model_dir, encode_only=True)
    yield llm
    llm.shutdown()


@pytest.fixture(scope="module")
def bert_encode_llm_cuda_graph():
    """BERT encode_only LLM with a tight encoder CUDA graph bucket grid."""
    model_dir = get_model_path(BERT_MODEL_PATH)
    cgc = EncodeCudaGraphConfig(
        batch_sizes=[1, 4],
        num_tokens=[16, 64],
        seq_lens=[8, 32],
        enable_padding=True,
    )
    llm = LLM(model=model_dir, encode_only=True, cuda_graph_config=cgc)
    yield llm
    llm.shutdown()


# --------------------------------------------------------------------------- #
# Basic encode() functionality
# --------------------------------------------------------------------------- #


def test_encode_single_string(bert_encode_llm):
    """encode() with a single string returns a single EncoderOutput."""
    result = bert_encode_llm.encode("Hello, my name is")

    assert isinstance(result, EncoderOutput)
    assert isinstance(result.logits, torch.Tensor)
    assert result.logits.dim() == 1  # [num_classes] for classification
    assert result.logits.shape[0] == 2  # yelp-polarity has 2 classes
    assert result.prompt == "Hello, my name is"
    assert isinstance(result.prompt_token_ids, list)
    assert len(result.prompt_token_ids) > 0


def test_encode_batch(bert_encode_llm):
    """encode() with a list of strings returns a list of EncoderOutput."""
    results = bert_encode_llm.encode(PROMPTS)

    assert isinstance(results, list)
    assert len(results) == len(PROMPTS)
    for i, result in enumerate(results):
        assert isinstance(result, EncoderOutput)
        assert result.logits.shape == (2,)  # 2 classes
        assert result.prompt == PROMPTS[i]


def test_encode_token_ids(bert_encode_llm):
    """encode() accepts pre-tokenized token ID lists."""
    token_ids = [101, 7592, 1010, 2026, 2171, 2003, 102]  # "[CLS] hello, my name is [SEP]"
    result = bert_encode_llm.encode(token_ids)

    assert isinstance(result, EncoderOutput)
    assert result.logits.shape == (2,)
    assert result.prompt is None  # no text prompt when passing token IDs
    assert result.prompt_token_ids == token_ids


def test_encode_mixed_batch(bert_encode_llm):
    """encode() handles mixed input types in a batch."""
    from tensorrt_llm.inputs import TextPrompt, TokensPrompt

    inputs = [
        "Hello world",
        TextPrompt(prompt="Test sentence"),
        TokensPrompt(prompt_token_ids=[101, 7592, 2088, 102]),
    ]
    results = bert_encode_llm.encode(inputs)

    assert len(results) == 3
    assert results[0].prompt == "Hello world"
    assert results[1].prompt == "Test sentence"
    assert results[2].prompt is None


# --------------------------------------------------------------------------- #
# Cross-API guards
# --------------------------------------------------------------------------- #


def test_generate_raises_on_encoder_only(bert_encode_llm):
    """generate() raises RuntimeError when encode_only=True."""
    with pytest.raises(RuntimeError, match="encode_only=True"):
        bert_encode_llm.generate(PROMPTS)


def test_generate_async_raises_on_encoder_only(bert_encode_llm):
    """generate_async() raises RuntimeError when encode_only=True."""
    with pytest.raises(RuntimeError, match="encode_only=True"):
        bert_encode_llm.generate_async("Hello")


def test_encode_raises_without_encoder_only():
    """encode() raises RuntimeError on a decoder model (encode_only=False)."""
    model_dir = get_model_path(BERT_MODEL_PATH)
    with LLM(model=model_dir, encode_only=False, disable_overlap_scheduler=True) as llm:
        with pytest.raises(RuntimeError, match="encode_only=True"):
            llm.encode("Hello")


def test_get_stats_raises_on_encoder_only(bert_encode_llm):
    """get_stats() raises RuntimeError when encode_only=True."""
    with pytest.raises(RuntimeError, match="encode_only=True"):
        bert_encode_llm.get_stats()


def test_get_kv_cache_capacity_raises_on_encoder_only(bert_encode_llm):
    """get_kv_cache_capacity() raises RuntimeError when encode_only=True."""
    with pytest.raises(RuntimeError, match="encode_only=True"):
        bert_encode_llm.get_kv_cache_capacity()


# --------------------------------------------------------------------------- #
# Batch tokenization (Triton pattern)
# --------------------------------------------------------------------------- #


def test_encode_batch_token_ids(bert_encode_llm):
    """encode() with a batch of pre-tokenized token IDs (Triton serving pattern).

    This validates the batch tokenization pattern used in the Triton backend
    example, where the tokenizer is called once for the entire batch and
    the resulting token IDs are passed to encode().
    """
    from transformers import AutoTokenizer

    model_dir = get_model_path(BERT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Batch tokenize — one tokenizer call for all prompts
    encoded = tokenizer(PROMPTS, padding=False, truncation=True, max_length=512)
    token_ids_list = encoded["input_ids"]

    # Pass pre-tokenized IDs to encode()
    results_from_ids = bert_encode_llm.encode(token_ids_list)

    # Compare with string-based tokenization
    results_from_strings = bert_encode_llm.encode(PROMPTS)

    assert len(results_from_ids) == len(results_from_strings)
    for r_ids, r_str in zip(results_from_ids, results_from_strings):
        # Logits should be identical — same model, same tokens
        torch.testing.assert_close(r_ids.logits, r_str.logits)
        # Token IDs passed directly don't get re-tokenized
        assert r_ids.prompt is None
        assert r_str.prompt is not None


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_encode_empty_string(bert_encode_llm):
    """encode("") should either raise or produce a valid (empty-ish) result.

    Tokenizing "" with add_special_tokens=True produces [CLS][SEP] (2 tokens),
    so this is actually a valid input for BERT.
    """
    result = bert_encode_llm.encode("")
    assert isinstance(result, EncoderOutput)
    assert result.logits.shape == (2,)


def test_encode_oversized_batch(bert_encode_llm):
    """encode() raises ValueError when batch exceeds max_batch_size."""
    engine = bert_encode_llm._encoder_executor.model_engine
    max_batch = engine.batch_size

    # Create a batch that exceeds max_batch_size
    oversized = ["Hello"] * (max_batch + 1)
    with pytest.raises(ValueError, match="max_batch_size"):
        bert_encode_llm.encode(oversized)


def test_encode_add_special_tokens_false(bert_encode_llm):
    """add_special_tokens=False skips [CLS]/[SEP] tokens."""
    result_with = bert_encode_llm.encode("Hello world", add_special_tokens=True)
    result_without = bert_encode_llm.encode("Hello world", add_special_tokens=False)

    # With special tokens: [CLS] hello world [SEP] = more tokens
    # Without: hello world = fewer tokens
    assert len(result_with.prompt_token_ids) > len(result_without.prompt_token_ids)
    # Both should still produce valid classification output
    assert result_with.logits.shape == (2,)
    assert result_without.logits.shape == (2,)


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #


def test_check_health_encoder_only(bert_encode_llm):
    """_check_health() returns True for a live encoder-only LLM."""
    assert bert_encode_llm._check_health() is True


# --------------------------------------------------------------------------- #
# CUDA graph + return_raw_logits (contract only — accuracy lives in the
# integration tests under tests/integration/defs/accuracy/)
# --------------------------------------------------------------------------- #


def test_encode_cuda_graph_and_return_raw_logits(bert_encode_llm_cuda_graph):
    """Exercises both the encoder CUDA graph path and the return_raw_logits flag.

    Contract checks only:
    - Default wrapping (batched input) returns a list of EncoderOutput, each
      with .logits of shape [num_classes].
    - return_raw_logits=True returns a single torch.Tensor of shape
      [batch_size, num_classes] regardless of whether the input was batched.
      Note the asymmetry: the default path unwraps a single-prompt input back
      to a single EncoderOutput, but the raw path always returns the full
      tensor (still 2D for BERT, [1, num_classes] for a single prompt).
    """
    # Batched input — default wrapping.
    outs = bert_encode_llm_cuda_graph.encode(PROMPTS)
    assert isinstance(outs, list)
    assert len(outs) == len(PROMPTS)
    for o in outs:
        assert isinstance(o, EncoderOutput)
        assert o.logits.shape == (2,)  # yelp-polarity: 2 classes

    # Batched input — return_raw_logits=True returns a single 2D tensor.
    raw = bert_encode_llm_cuda_graph.encode(PROMPTS, return_raw_logits=True)
    assert isinstance(raw, torch.Tensor)
    assert raw.shape == (len(PROMPTS), 2)

    # Single-prompt input — raw path stays 2D [1, num_classes], unlike the
    # default path which unwraps to a single EncoderOutput with 1D .logits.
    raw_single = bert_encode_llm_cuda_graph.encode(PROMPTS[0], return_raw_logits=True)
    assert isinstance(raw_single, torch.Tensor)
    assert raw_single.shape == (1, 2)
