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

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import EncodeCudaGraphConfig, EncodeExtraInputSpec
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


# --------------------------------------------------------------------------- #
# EncodeExtraInputSpec — Pydantic-level validation (no GPU)
# --------------------------------------------------------------------------- #


class TestEncodeExtraInputSpec:
    """Validation of the user-facing spec class. Pure Pydantic; no model load."""

    def test_basic_spec(self):
        s = EncodeExtraInputSpec(name="token_type_ids", shape=("num_tokens",), dtype="int32")
        assert s.name == "token_type_ids"
        assert s.resolve_shape(num_tokens=64, batch_size=8) == (64,)
        assert s.symbolic_dim() == ("num_tokens", 0)
        assert s.torch_dtype() == torch.int32

    def test_multidim_spec(self):
        s = EncodeExtraInputSpec(name="inputs_embeds", shape=("num_tokens", 768), dtype="bfloat16")
        assert s.resolve_shape(num_tokens=32, batch_size=8) == (32, 768)
        assert s.symbolic_dim() == ("num_tokens", 0)
        assert s.torch_dtype() == torch.bfloat16

    def test_batch_size_spec(self):
        """A spec keyed on 'batch_size' instead of 'num_tokens' (e.g. per-request features)."""
        s = EncodeExtraInputSpec(
            name="per_request_feature", shape=("batch_size", 40), dtype="int32"
        )
        assert s.resolve_shape(num_tokens=64, batch_size=4) == (4, 40)
        assert s.symbolic_dim() == ("batch_size", 0)

    def test_batch_size_symbolic_axis_nonzero(self):
        s = EncodeExtraInputSpec(name="x", shape=(32, "batch_size"), dtype="float32")
        assert s.resolve_shape(num_tokens=64, batch_size=8) == (32, 8)
        assert s.symbolic_dim() == ("batch_size", 1)

    def test_mixed_symbolic_dims_rejected(self):
        """A single tensor cannot use both 'num_tokens' and 'batch_size'."""
        with pytest.raises(Exception, match="symbolic dim"):
            EncodeExtraInputSpec(name="x", shape=("num_tokens", "batch_size"), dtype="int32")

    def test_reserved_name_rejected(self):
        for reserved in (
            "input_ids",
            "position_ids",
            "seq_lens",
            "multi_item_part_lens",
            "attn_metadata",
            "return_context_logits",
        ):
            with pytest.raises(Exception, match="reserved"):
                EncodeExtraInputSpec(name=reserved, shape=("num_tokens",), dtype="int32")

    def test_non_identifier_name_rejected(self):
        with pytest.raises(Exception, match="identifier"):
            EncodeExtraInputSpec(name="123abc", shape=("num_tokens",), dtype="int32")

    def test_missing_symbolic_dim_rejected(self):
        with pytest.raises(Exception, match="symbolic dim"):
            EncodeExtraInputSpec(name="x", shape=(64,), dtype="int32")

    def test_duplicate_symbolic_dim_rejected(self):
        with pytest.raises(Exception, match="symbolic dim"):
            EncodeExtraInputSpec(name="x", shape=("num_tokens", "num_tokens"), dtype="int32")

    def test_unknown_dtype_rejected(self):
        with pytest.raises(Exception, match="dtype"):
            EncodeExtraInputSpec(name="x", shape=("num_tokens",), dtype="not_a_dtype")

    def test_duplicate_names_in_extra_model_inputs_rejected(self):
        s = EncodeExtraInputSpec(name="token_type_ids", shape=("num_tokens",), dtype="int32")
        with pytest.raises(Exception, match="duplicate name"):
            EncodeCudaGraphConfig(num_tokens=[32], seq_lens=[16], extra_model_inputs=[s, s])


# --------------------------------------------------------------------------- #
# LLM.encode() validation under encoder CUDA graphs + extra_model_inputs
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def bert_encode_llm_cuda_graph_with_token_type_ids():
    """BERT encode_only LLM with token_type_ids declared as an extra model input."""
    model_dir = get_model_path(BERT_MODEL_PATH)
    cgc = EncodeCudaGraphConfig(
        batch_sizes=[1, 4],
        num_tokens=[16, 64],
        seq_lens=[8, 32],
        enable_padding=True,
        extra_model_inputs=[
            EncodeExtraInputSpec(name="token_type_ids", shape=("num_tokens",), dtype="int32"),
        ],
    )
    llm = LLM(model=model_dir, encode_only=True, cuda_graph_config=cgc)
    yield llm
    llm.shutdown()


def _build_token_type_ids(prompts, device="cuda"):
    """Build observable packed token_type_ids sized to the BERT tokenization."""
    from transformers import AutoTokenizer

    model_dir = get_model_path(BERT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoded = tokenizer(prompts, padding=False, truncation=True, max_length=512)
    total = sum(len(t) for t in encoded["input_ids"])
    return torch.ones(total, dtype=torch.int32, device=device)


def test_encode_with_declared_token_type_ids(bert_encode_llm_cuda_graph_with_token_type_ids):
    """Encoder CUDA graphs + declared token_type_ids — encode() succeeds."""
    llm = bert_encode_llm_cuda_graph_with_token_type_ids
    token_type_ids = _build_token_type_ids(PROMPTS)
    outs = llm.encode(PROMPTS, token_type_ids=token_type_ids)
    assert len(outs) == len(PROMPTS)
    for o in outs:
        assert o.logits.shape == (2,)


def test_encode_rejects_undeclared_kwarg(bert_encode_llm_cuda_graph_with_token_type_ids):
    """encode() with a kwarg that wasn't declared in extra_model_inputs raises."""
    llm = bert_encode_llm_cuda_graph_with_token_type_ids
    token_type_ids = _build_token_type_ids(PROMPTS)
    bogus = torch.zeros(token_type_ids.shape[0], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="not declared"):
        llm.encode(PROMPTS, token_type_ids=token_type_ids, bogus_kwarg=bogus)


def test_encode_rejects_missing_declared_kwarg(bert_encode_llm_cuda_graph_with_token_type_ids):
    """encode() that omits a declared kwarg raises."""
    llm = bert_encode_llm_cuda_graph_with_token_type_ids
    with pytest.raises(ValueError, match="missing model_kwargs"):
        llm.encode(PROMPTS)


def test_encode_rejects_wrong_dtype(bert_encode_llm_cuda_graph_with_token_type_ids):
    """encode() with a declared kwarg of the wrong dtype raises."""
    llm = bert_encode_llm_cuda_graph_with_token_type_ids
    token_type_ids = _build_token_type_ids(PROMPTS)
    wrong_dtype = token_type_ids.to(torch.int64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        llm.encode(PROMPTS, token_type_ids=wrong_dtype)


def test_encode_rejects_wrong_rank(bert_encode_llm_cuda_graph_with_token_type_ids):
    """encode() with a declared kwarg of the wrong rank raises."""
    llm = bert_encode_llm_cuda_graph_with_token_type_ids
    token_type_ids = _build_token_type_ids(PROMPTS)
    wrong_rank = token_type_ids.unsqueeze(0)  # ('num_tokens',) -> (1, n)
    with pytest.raises(ValueError, match="rank mismatch"):
        llm.encode(PROMPTS, token_type_ids=wrong_rank)


def test_encode_rejects_non_tensor_kwarg(bert_encode_llm_cuda_graph_with_token_type_ids):
    """encode() with a non-tensor value for a declared kwarg raises."""
    llm = bert_encode_llm_cuda_graph_with_token_type_ids
    with pytest.raises(ValueError, match="must be a torch.Tensor"):
        llm.encode(PROMPTS, token_type_ids=[0] * 32)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encode_declared_kwarg_accepts_either_device(
    bert_encode_llm, bert_encode_llm_cuda_graph_with_token_type_ids, device
):
    """A declared kwarg may be passed on the host or on the device.

    Graph replay copies it into the static buffer and the eager path moves it
    to the device, so the caller must not have to know which one runs. Non-zero
    segment ids make dropping the caller-provided tensor observable.
    """
    llm = bert_encode_llm_cuda_graph_with_token_type_ids
    token_type_ids = _build_token_type_ids(PROMPTS, device=device)
    outs = llm.encode(PROMPTS, token_type_ids=token_type_ids)

    eager_token_type_ids = _build_token_type_ids(PROMPTS)
    eager_outs = bert_encode_llm.encode(PROMPTS, token_type_ids=eager_token_type_ids)
    got = torch.stack([o.logits.cpu() for o in outs])
    eager = torch.stack([o.logits.cpu() for o in eager_outs])
    torch.testing.assert_close(got, eager, rtol=1e-3, atol=1e-3)


def test_encode_host_declared_kwarg_on_eager_fallback(
    bert_encode_llm, bert_encode_llm_cuda_graph_with_token_type_ids
):
    """A host declared kwarg still works when the call falls back to eager.

    The undeclared non-tensor kwarg forces the eager path, which hands the
    kwargs straight to forward(); the engine has to move `token_type_ids` to
    the device rather than letting the embedding lookup fail on a CPU tensor.
    """
    llm = bert_encode_llm_cuda_graph_with_token_type_ids
    token_type_ids = _build_token_type_ids(PROMPTS, device="cpu")
    outs = llm.encode(PROMPTS, token_type_ids=token_type_ids, some_flag=True)

    eager_token_type_ids = _build_token_type_ids(PROMPTS)
    eager_outs = bert_encode_llm.encode(PROMPTS, token_type_ids=eager_token_type_ids)
    got = torch.stack([o.logits.cpu() for o in outs])
    eager = torch.stack([o.logits.cpu() for o in eager_outs])
    torch.testing.assert_close(got, eager, rtol=1e-3, atol=1e-3)


def test_encode_undeclared_non_tensor_kwarg_falls_back_to_eager(
    bert_encode_llm, bert_encode_llm_cuda_graph_with_token_type_ids
):
    """Verify an undeclared non-tensor model kwarg falls back to eager execution.

    Such a kwarg is allowed under encoder CUDA graphs but forces the call onto
    the eager path because a captured graph cannot represent a non-tensor
    value. The declared tensor kwarg is still provided; BERT ignores the extra
    non-tensor kwarg via **kwargs, so the output must match plain eager
    execution.
    """
    llm = bert_encode_llm_cuda_graph_with_token_type_ids
    token_type_ids = _build_token_type_ids(PROMPTS)
    # `some_flag` is undeclared and non-tensor: previously this raised
    # ("not declared"); now it is allowed and triggers eager fallback.
    graph_runner = llm._encoder_executor.model_engine._runner._encoder_cuda_graph_runner
    with patch.object(graph_runner, "replay", wraps=graph_runner.replay) as replay:
        llm.encode(PROMPTS, token_type_ids=token_type_ids)
        assert replay.call_count == 1
        replay.reset_mock()

        graph_outs = llm.encode(PROMPTS, token_type_ids=token_type_ids, some_flag=True)
        replay.assert_not_called()
    assert len(graph_outs) == len(PROMPTS)

    eager_outs = bert_encode_llm.encode(PROMPTS, token_type_ids=token_type_ids)
    graph = torch.stack([o.logits.cpu() for o in graph_outs])
    eager = torch.stack([o.logits.cpu() for o in eager_outs])
    torch.testing.assert_close(graph, eager, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# batch_size-shaped extra inputs (per-request features)
# BERT's forward ignores unknown kwargs via **kwargs, so an extra
# batch_size-shaped tensor exercises the static-buffer / capture / replay
# machinery without changing the model's output. That lets us pin numerical
# parity between graph-on (with the kwarg declared and passed) and eager
# (no kwarg) — a regression here would mean the static buffer machinery is
# corrupting the model's compute.
# --------------------------------------------------------------------------- #

_BATCH_SIZE_EXTRA = EncodeExtraInputSpec(
    name="per_request_feature", shape=("batch_size", 40), dtype="int32"
)


@pytest.fixture(scope="module")
def bert_encode_llm_cuda_graph_with_batch_size_extra():
    """BERT encode_only LLM with a per_request_feature shape=(batch_size, 40) extra."""
    model_dir = get_model_path(BERT_MODEL_PATH)
    cgc = EncodeCudaGraphConfig(
        batch_sizes=[1, 4],
        num_tokens=[16, 64],
        seq_lens=[8, 32],
        enable_padding=True,
        extra_model_inputs=[_BATCH_SIZE_EXTRA],
    )
    llm = LLM(model=model_dir, encode_only=True, cuda_graph_config=cgc)
    yield llm
    llm.shutdown()


def _build_batch_size_feature(batch: int):
    return torch.arange(batch * 40, dtype=torch.int32, device="cuda").reshape(batch, 40)


def test_encode_accepts_batch_size_extra(bert_encode_llm_cuda_graph_with_batch_size_extra):
    """encode() with a declared batch_size-shaped kwarg succeeds."""
    llm = bert_encode_llm_cuda_graph_with_batch_size_extra
    feat = _build_batch_size_feature(len(PROMPTS))
    outs = llm.encode(PROMPTS, per_request_feature=feat)
    assert len(outs) == len(PROMPTS)
    for o in outs:
        assert o.logits.shape == (2,)


def test_encode_batch_size_extra_wrong_dim_rejected(
    bert_encode_llm_cuda_graph_with_batch_size_extra,
):
    """Wrong batch dim along the symbolic axis raises a clear error."""
    llm = bert_encode_llm_cuda_graph_with_batch_size_extra
    # One extra row vs. number of prompts.
    feat = _build_batch_size_feature(len(PROMPTS) + 1)
    with pytest.raises(ValueError, match="'batch_size'"):
        llm.encode(PROMPTS, per_request_feature=feat)


def test_encode_batch_size_extra_wrong_literal_dim_rejected(
    bert_encode_llm_cuda_graph_with_batch_size_extra,
):
    """Wrong literal dim raises a shape mismatch."""
    llm = bert_encode_llm_cuda_graph_with_batch_size_extra
    # 40 → 39 along the literal axis.
    feat = torch.zeros(len(PROMPTS), 39, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="shape mismatch"):
        llm.encode(PROMPTS, per_request_feature=feat)


def test_encode_batch_size_extra_wrong_dtype_rejected(
    bert_encode_llm_cuda_graph_with_batch_size_extra,
):
    llm = bert_encode_llm_cuda_graph_with_batch_size_extra
    feat = _build_batch_size_feature(len(PROMPTS)).to(torch.int64)
    with pytest.raises(ValueError, match="dtype mismatch"):
        llm.encode(PROMPTS, per_request_feature=feat)


def test_encode_batch_size_extra_replay_matches_eager(
    bert_encode_llm, bert_encode_llm_cuda_graph_with_batch_size_extra
):
    """Verify batch-size static-buffer replay reproduces eager output.

    BERT ignores unknown kwargs (**kwargs in forward), so the kwarg has no
    effect on compute. The comparison directly checks that the per-spec static
    buffer plumbing doesn't perturb the graph.
    """
    eager_outs = bert_encode_llm.encode(PROMPTS)
    graph_llm = bert_encode_llm_cuda_graph_with_batch_size_extra
    feat = _build_batch_size_feature(len(PROMPTS))
    graph_outs = graph_llm.encode(PROMPTS, per_request_feature=feat)

    eager = torch.stack([o.logits.cpu() for o in eager_outs])
    graph = torch.stack([o.logits.cpu() for o in graph_outs])
    torch.testing.assert_close(graph, eager, rtol=1e-3, atol=1e-3)
