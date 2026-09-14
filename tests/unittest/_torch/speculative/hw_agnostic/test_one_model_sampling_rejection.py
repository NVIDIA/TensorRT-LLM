# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for admission-time rejection of sampling features the
one-model speculative path cannot honor.

``SpecSampler.validate_request`` has always rejected min_length, bad_words,
no_repeat_ngram_size, embedding_bias, top_p_decay and min_p, but it runs on the
executor's admission path. For a streaming OpenAI request that is too late: the
frontend has already answered 200 and opened the response stream, so the client
receives a truncated stream (``ClientPayloadError`` / ``TransferEncodingError``)
that is indistinguishable from a network fault rather than a status code.

``LLM._check_arguments`` now runs the same rules while ``generate_async`` is
still synchronous, so the request gets a structured 4xx instead. These tests pin
three things: the frontend predicate fires on each feature, it stays silent for
modes that keep TorchSampler (which implements them), and both sides raise the
identical message so they cannot drift apart.
"""

import types

import pytest
import torch

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.speculative.spec_sampler_base import (
    UNSUPPORTED_BAD_WORDS_MSG,
    UNSUPPORTED_EMBEDDING_BIAS_MSG,
    UNSUPPORTED_MIN_LENGTH_MSG,
    UNSUPPORTED_MIN_P_MSG,
    UNSUPPORTED_NO_REPEAT_NGRAM_MSG,
    UNSUPPORTED_TOP_P_DECAY_MSG,
    SpecSampler,
    one_model_sampling_rejection_reason,
)
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi import LLM
from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig, SADecodingConfig

# (id, SamplingParams kwargs, expected message). Every entry is a value the
# OpenAI frontend can send today.
UNSUPPORTED_CASES = [
    ("min_tokens", dict(min_tokens=16), UNSUPPORTED_MIN_LENGTH_MSG),
    ("bad_token_ids", dict(bad_token_ids=[7]), UNSUPPORTED_BAD_WORDS_MSG),
    ("bad", dict(bad="nope"), UNSUPPORTED_BAD_WORDS_MSG),
    ("no_repeat_ngram_size", dict(no_repeat_ngram_size=3), UNSUPPORTED_NO_REPEAT_NGRAM_MSG),
    ("top_p_decay", dict(top_p_decay=0.5), UNSUPPORTED_TOP_P_DECAY_MSG),
    ("min_p", dict(min_p=0.1), UNSUPPORTED_MIN_P_MSG),
]

# Values a frontend forwards explicitly at their neutral setting. These must not
# be rejected, or every default OpenAI request would 400 under SA.
NEUTRAL_CASES = [
    ("min_tokens_zero", dict(min_tokens=0)),
    ("no_repeat_ngram_zero", dict(no_repeat_ngram_size=0)),
    ("min_p_zero", dict(min_p=0.0)),
    # top_p_decay == 1.0 means "no decay"; top_p_min / top_p_reset_ids alone do
    # not activate dynamic behavior either.
    ("top_p_decay_one", dict(top_p_decay=1.0)),
    ("top_p_min_only", dict(top_p_min=0.1)),
    ("plain_greedy", dict(temperature=0.0, top_k=1, max_tokens=8)),
]


class _ArgsOnlyLLM:
    """Drives LLM's real argument checks without a checkpoint or a GPU.

    ``_check_arguments`` reads only ``self.args`` on this path, and the logic
    under test is pure config inspection -- building a real LLM would need
    weights and GPUs to assert something neither influences.
    """

    _check_arguments = LLM._check_arguments
    _check_one_model_speculative_sampling = LLM._check_one_model_speculative_sampling

    def __init__(self, spec_config):
        self.args = types.SimpleNamespace(
            backend="pytorch",
            enable_chunked_prefill=True,
            max_num_tokens=None,
            speculative_config=spec_config,
        )


def _check_arguments(spec_config, sampling_params):
    _ArgsOnlyLLM(spec_config)._check_arguments(
        prompt_len=8, sampling_params=sampling_params, is_gen_only=False
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "name,kwargs,expected", UNSUPPORTED_CASES, ids=[case[0] for case in UNSUPPORTED_CASES]
)
def test_rejection_reason_names_the_feature(name, kwargs, expected):
    assert one_model_sampling_rejection_reason(SamplingParams(**kwargs)) == expected


@pytest.mark.cpu_only
def test_embedding_bias_is_rejected():
    # Kept out of the parametrized table: a tensor field cannot be compared by
    # value in an id-generating parametrize list.
    params = SamplingParams(embedding_bias=torch.zeros(16))
    assert one_model_sampling_rejection_reason(params) == UNSUPPORTED_EMBEDDING_BIAS_MSG


@pytest.mark.cpu_only
@pytest.mark.parametrize("name,kwargs", NEUTRAL_CASES, ids=[case[0] for case in NEUTRAL_CASES])
def test_neutral_values_are_not_rejected(name, kwargs):
    assert one_model_sampling_rejection_reason(SamplingParams(**kwargs)) is None


@pytest.mark.cpu_only
def test_no_sampling_params_is_not_rejected():
    assert one_model_sampling_rejection_reason(None) is None


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "name,kwargs,expected", UNSUPPORTED_CASES, ids=[case[0] for case in UNSUPPORTED_CASES]
)
def test_check_arguments_raises_before_submission_under_sa(name, kwargs, expected):
    """SA is one-engine, so the request must fail while generate_async is
    still synchronous -- that is what turns the aborted stream into a 4xx."""
    spec_config = SADecodingConfig(max_draft_len=4, max_matching_ngram_size=-1)
    with pytest.raises(RequestError) as excinfo:
        _check_arguments(spec_config, SamplingParams(**kwargs))
    assert str(excinfo.value) == expected


@pytest.mark.cpu_only
@pytest.mark.parametrize("name,kwargs", NEUTRAL_CASES, ids=[case[0] for case in NEUTRAL_CASES])
def test_check_arguments_admits_neutral_requests_under_sa(name, kwargs):
    spec_config = SADecodingConfig(max_draft_len=4, max_matching_ngram_size=-1)
    _check_arguments(spec_config, SamplingParams(**kwargs))


@pytest.mark.cpu_only
def test_check_arguments_admits_when_speculation_is_off():
    _check_arguments(None, SamplingParams(min_tokens=16))


@pytest.mark.cpu_only
def test_check_arguments_admits_drafter_modes():
    """NGram drafts with a separate pass and keeps TorchSampler, which
    implements every feature above. Rejecting there would be a regression, not
    a fix -- the guard has to be scoped to the modes get_spec_decoder routes to
    SpecSampler."""
    spec_config = NGramDecodingConfig(max_draft_len=4, max_matching_ngram_size=2)
    assert not spec_config.spec_dec_mode.use_one_engine()
    _check_arguments(spec_config, SamplingParams(min_tokens=16))


# The executor-side helper decides these four from py_ fields alone, so a stub
# request is enough. min_p and top_p_decay are decided from the C++
# SamplingConfig instead and are covered by the frontend cases above plus the
# shared constants.
EXECUTOR_SIDE_CASES = [
    ("min_length", dict(py_min_length=16), UNSUPPORTED_MIN_LENGTH_MSG),
    ("bad_words", dict(py_bad_words=[[7]]), UNSUPPORTED_BAD_WORDS_MSG),
    ("no_repeat_ngram_size", dict(py_no_repeat_ngram_size=3), UNSUPPORTED_NO_REPEAT_NGRAM_MSG),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "name,fields,expected", EXECUTOR_SIDE_CASES, ids=[case[0] for case in EXECUTOR_SIDE_CASES]
)
def test_executor_side_check_still_rejects(name, fields, expected):
    """The executor-side check is the backstop for submission paths that skip
    the LLM API, so moving the rejection earlier must not remove it. It raises
    the same shared constant, so a client cannot get two different explanations
    for one cause."""
    neutral = dict(
        py_min_length=None, py_bad_words=None, py_no_repeat_ngram_size=None, py_embedding_bias=None
    )
    request = types.SimpleNamespace(**{**neutral, **fields})
    with pytest.raises(ValueError) as excinfo:
        SpecSampler._validate_unsupported_logits_processors(request)
    assert str(excinfo.value) == expected


@pytest.mark.cpu_only
def test_executor_side_check_rejects_embedding_bias():
    request = types.SimpleNamespace(
        py_min_length=None,
        py_bad_words=None,
        py_no_repeat_ngram_size=None,
        py_embedding_bias=torch.zeros(16),
    )
    with pytest.raises(ValueError) as excinfo:
        SpecSampler._validate_unsupported_logits_processors(request)
    assert str(excinfo.value) == UNSUPPORTED_EMBEDDING_BIAS_MSG
