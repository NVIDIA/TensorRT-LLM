# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Mock pb_utils
sys.modules["triton_python_backend_utils"] = MagicMock()

# Use PYTHONPATH=../inflight_batcher_llm/tensorrt_llm/1/
from model import *

import tensorrt_llm.bindings.executor as trtllm


@dataclass
class MockTritonTensor:
    _name: str
    _tensor: Union[np.ndarray, torch.Tensor]

    def name(self) -> str:
        return self._name

    def as_numpy(self) -> np.ndarray:
        if self.is_cpu():
            return self._tensor
        else:
            return self._tensor.as_numpy()

    def is_cpu(self) -> bool:
        if isinstance(self._tensor, np.ndarray):
            return True
        else:
            return False

    def to_dlpack(self):
        if self.is_cpu():
            return self._tensor.__dlpack__()
        else:
            return self._tensor.to_dlpack()


@dataclass
class MockTritonError:
    message: str


@dataclass
class MockTritonResponse:
    tensors: Dict[str, MockTritonTensor]
    error: MockTritonError

    def __init__(self,
                 output_tensors: List[MockTritonTensor],
                 error: MockTritonError = None):
        self.tensors = {}
        for tensor in output_tensors:
            self.tensors[tensor.name()] = tensor
        self.error = error

    def output_tensors(self):
        return self.tensors.values()

    def has_error(self):
        return self.error is not None


@dataclass
class MockTritonRequest:
    tensors: Dict[str, MockTritonTensor]

    def get_input_tensor_by_name(self, name: str) -> MockTritonTensor:
        return self.tensors[name] if name in self.tensors else None

    def get_response_sender(self):
        return None


def mock_pb_utils_get_input_tensor_by_name_side_effect(
        request: MockTritonRequest, name: str) -> MockTritonTensor:
    return request.get_input_tensor_by_name(name)


def make_mock_triton_request(
        tensors: Dict[str, np.ndarray]) -> MockTritonRequest:
    return MockTritonRequest({
        k: MockTritonTensor(k, np.array(v))
        for k, v in tensors.items()
    })


@pytest.fixture(autouse=True)
def apply_patches():
    patch("model.pb_utils.Tensor", new=MockTritonTensor).start()
    patch("model.pb_utils.InferenceResponse", new=MockTritonResponse).start()
    patch("model.pb_utils.TritonError", new=MockTritonError).start()
    patch("model.pb_utils.InferenceRequest", new=MockTritonRequest).start()
    patch("model.pb_utils.get_input_tensor_by_name",
          new=mock_pb_utils_get_input_tensor_by_name_side_effect).start()
    patch("model.pb_utils.TritonModelException", new=Exception).start()


@pytest.fixture
def triton_request() -> MockTritonRequest:
    inputs = {
        "input_ids": [[28524, 287, 5093, 12]],
        "request_output_len": [16],
        "streaming": [True],
        "end_id": [50256],
        "pad_id": [50256],
        "stop_words_list": [[[14480, 326, 262, 1171], [1, 4, -1, -1]]],
        "bad_words_list": [[[24044, 76, 1230], [2, 3, -1]]],
        "embedding_bias":
        np.array([[0., 0., 0.]], dtype=np.float32),
        "beam_width": [2],
        "runtime_top_k": [1],
        "runtime_top_p": [0.],
        "seed": [4],
        "temperature": [1.],
        "min_tokens": [3],
        "repetition_penalty": [1.0],
        "presence_penalty": [2.0],
        "frequency_penalty": [4.0],
        "len_penalty": [8.0],
        "runtime_top_p_min": [1.0],
        "runtime_top_p_reset_ids": [1],
        "runtime_top_p_decay": [1.0],
        "beam_search_diversity_rate": [1.0],
        "early_stopping": [True],
        "return_log_probs":
        True,
        "return_context_logits":
        True,
        "return_generation_logits":
        True,
        "draft_input_ids": [[0, 1]],
        "draft_logits":
        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        "draft_acceptance_threshold":
        1.0,
        "prompt_embedding_table":
        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float16),
        "lora_task_id": [1],
        "lora_weights":
        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float16),
        "lora_config":
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.int32),
        "retention_token_range_starts":
        np.array([[0, 100]], dtype=np.int32),
        "retention_token_range_ends":
        np.array([[100, 200]], dtype=np.int32),
        "retention_token_range_priorities":
        np.array([[100, 50]], dtype=np.int32),
        "prompt_vocab_size": [2],
    }
    return make_mock_triton_request(inputs)


@pytest.fixture
def batched_triton_request() -> MockTritonRequest:
    inputs = {
        "input_ids": [[28524, 287, 5093, 12], [1, 2, 3, 4]],
        "input_lengths": [4, 2],
        "request_output_len": [16, 3],
        "streaming": [True, False],
        "end_id": [50256, 50257],
        "pad_id": [50256, 50257],
        "stop_words_list": [[[14480, 326, 262, 1171], [1, 4, -1, -1]],
                            [[66, 77, -1, -1], [1, 2, -1, -1]]],
        "bad_words_list": [[[24044, 76, 1230], [2, 3, -1]],
                           [[88, 99, 111], [1, 3, -1]]],
        "embedding_bias":
        np.array([[0., 0., 0.], [1., 1., 1.]], dtype=np.float32),
        "beam_width": [2, 3],
        "runtime_top_k": [1, 2],
        "runtime_top_p": [0., 1.],
        "seed": [4, 7],
        "temperature": [1., 0.5],
        "min_tokens": [3, 10],
        "repetition_penalty": [1.0, 1.1],
        "presence_penalty": [2.0, 2.1],
        "frequency_penalty": [4.0, 4.1],
        "len_penalty": [8.0, 8.1],
        "runtime_top_p_min": [1.0, 0.5],
        "runtime_top_p_reset_ids": [1, 3],
        "runtime_top_p_decay": [1.0, 0.1],
        "beam_search_diversity_rate": [1.0, 0.7],
        "early_stopping": [True, False],
        "return_log_probs": [True, False],
        "return_context_logits": [True, False],
        "return_generation_logits": [True, False],
        "draft_input_ids": [[0, 1], [2, 3]],
        "draft_logits":
        np.array([[[1.0, 2.0], [3.0, 4.0]], [[1.1, 2.1], [3.1, 4.1]]],
                 dtype=np.float32),
        "draft_acceptance_threshold": [1.0, 0.5],
        "prompt_embedding_table":
        np.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 3.0], [4.0, 5.0]]],
                 dtype=np.float16),
        "lora_task_id": [1, 2],
        "lora_weights":
        np.array([[[1.0, 2.0], [3.0, 4.0]], [[3.0, 4.0], [5.0, 6.0]]],
                 dtype=np.float16),
        "lora_config":
        np.array([[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]],
                 dtype=np.int32),
        "retention_token_range_starts":
        np.array([[0, 100], [0, 200]], dtype=np.int32),
        "retention_token_range_ends":
        np.array([[100, 200], [200, 300]], dtype=np.int32),
        "retention_token_range_priorities":
        np.array([[100, 50], [0, 0]], dtype=np.int32),
        "prompt_vocab_size": [2],
    }
    return make_mock_triton_request(inputs)


@pytest.fixture
def triton_request_minimal() -> MockTritonRequest:
    inputs = {
        "input_ids": [[28524, 287, 5093, 12]],
        "request_output_len": [[16]],
    }
    return MockTritonRequest({
        k: MockTritonTensor(k, np.array(v))
        for k, v in inputs.items()
    })


@pytest.fixture
def trtllm_response() -> trtllm.Response:
    result = trtllm.Result()
    result.is_final = True
    result.output_token_ids = [[1, 2, 3]]
    result.cum_log_probs = [1]
    result.log_probs = [[1, 3]]
    result.context_logits = torch.ones(3, 10)
    result.generation_logits = torch.ones(1, 5, 10)
    return trtllm.Response(0, result)


@pytest.fixture
def trtllm_response_minimal() -> trtllm.Response:
    result = trtllm.Result()
    result.is_final = False
    result.output_token_ids = [[1, 2, 3]]
    return trtllm.Response(0, result)


@pytest.fixture
def trtllm_response_error() -> trtllm.Response:
    return trtllm.Response(0, "internal error")


def test_get_input_tensor_by_name(triton_request: MockTritonRequest):
    assert (get_input_tensor_by_name(triton_request, "input_ids") == np.array(
        [[28524, 287, 5093, 12]])).all()
    assert get_input_tensor_by_name(triton_request, "no_value") is None


def test_get_input_scalar_by_name(triton_request: MockTritonRequest):
    assert get_input_scalar_by_name(triton_request, "request_output_len") == 16
    assert get_input_scalar_by_name(triton_request, "streaming") == True
    assert get_input_scalar_by_name(triton_request, "end_id") == 50256
    assert get_input_scalar_by_name(triton_request, "pad_id") == 50256
    assert get_input_scalar_by_name(triton_request, "beam_width") == 2
    assert get_input_scalar_by_name(triton_request, "runtime_top_k") == 1
    assert get_input_scalar_by_name(triton_request, "runtime_top_p") == 0.
    assert get_input_scalar_by_name(triton_request, "temperature") == 1.


def test_read_parameter_as_type():
    assert read_parameter_as_type("", "name") is None
    assert read_parameter_as_type("", "name", int) is None
    assert read_parameter_as_type("", "name", float) is None
    assert read_parameter_as_type("", "name", bool) is None
    assert read_parameter_as_type("${unfilled_parameter}", "name") is None
    assert read_parameter_as_type("foo", "name", int) is None
    assert read_parameter_as_type("string_value", "name") == "string_value"
    assert read_parameter_as_type("4", "name", int) == 4
    assert read_parameter_as_type("0.5", "name", float) == 0.5
    assert read_parameter_as_type("1", "name", bool) == True
    assert read_parameter_as_type("true", "name", bool) == True
    assert read_parameter_as_type("True", "name", bool) == True
    assert read_parameter_as_type("0", "name", bool) == False
    assert read_parameter_as_type("false", "name", bool) == False
    assert read_parameter_as_type("False", "name", bool) == False


def test_get_parameter():
    model_config = {"parameters": {"max_beam_width": {"string_value": "1"}}}
    assert get_parameter(model_config, "max_beam_width", int) == 1
    assert get_parameter(model_config, "gpt_model_type", str) is None


def test_convert_word_list():
    assert convert_word_list(None) is None
    assert convert_word_list(np.array([[[], []]])) == []
    assert convert_word_list(
        np.array([[[14480, 326, 262, 1171], [1, 4, -1,
                                             -1]]])) == [[14480],
                                                         [326, 262, 1171]]
    assert convert_word_list(np.array([[[24044, 76, 1230],
                                        [2, 3, -1]]])) == [[24044, 76], [1230]]
    assert convert_word_list(np.array([[[326, 262, 1230],
                                        [3, -1, -1]]])) == [[326, 262, 1230]]
    for bad_format in [
            np.array([]),
            np.array([[]]),
            np.array([[[]]]),
            np.array([[[1], [2], [3]]]),
            np.array([[[262], [5]]]),
    ]:
        with pytest.raises(Exception, match="Invalid format for word list"):
            convert_word_list(bad_format)


def test_parse_medusa_choices():
    assert parse_medusa_choices("{0, 0, 0}, {0, 1}") == [[0, 0, 0], [0, 1]]
    for bad_format in [
            "{{}",
            "{",
            "{{}",
            "}",
            "{0, 1, 2",
            "0, 1, 2",
            "{0, 1, 2}, {\"foo\"}",
    ]:
        with pytest.raises(Exception,
                           match="Invalid format for medusa_choices"):
            parse_medusa_choices(bad_format)


def check_converted_request(converted):
    assert isinstance(converted, trtllm.Request)
    assert converted.input_token_ids == [28524, 287, 5093, 12]
    assert converted.max_tokens == 16
    assert converted.streaming == True
    assert converted.end_id == 50256
    assert converted.pad_id == 50256
    assert converted.stop_words == [[14480], [326, 262, 1171]]
    assert converted.bad_words == [[24044, 76], [1230]]
    assert (converted.embedding_bias == torch.tensor([0., 0., 0.])).all()
    assert converted.logits_post_processor_name is None

    assert isinstance(converted.external_draft_tokens_config,
                      trtllm.ExternalDraftTokensConfig)
    assert converted.external_draft_tokens_config.tokens == [0, 1]
    assert (converted.external_draft_tokens_config.logits == torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]])).all()
    assert converted.external_draft_tokens_config.acceptance_threshold == 1.0

    assert isinstance(converted.prompt_tuning_config, trtllm.PromptTuningConfig)
    assert (converted.prompt_tuning_config.embedding_table == torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]])).all()

    assert isinstance(converted.lora_config, trtllm.LoraConfig)
    assert converted.lora_config.task_id == 1
    assert (converted.lora_config.weights == torch.tensor([[1.0, 2.0],
                                                           [3.0, 4.0]])).all()
    assert (converted.lora_config.config == torch.tensor([[1, 2, 3],
                                                          [4, 5, 6]])).all()

    assert isinstance(converted.kv_cache_retention_config,
                      trtllm.KvCacheRetentionConfig)
    assert len(
        converted.kv_cache_retention_config.token_range_retention_configs)
    assert converted.kv_cache_retention_config.token_range_retention_configs[
        0].token_start == 0
    assert converted.kv_cache_retention_config.token_range_retention_configs[
        0].token_end == 100
    assert converted.kv_cache_retention_config.token_range_retention_configs[
        0].priority == 100

    assert converted.sampling_config.beam_width == 2
    assert converted.sampling_config.top_k == 1
    assert converted.sampling_config.top_p is None
    assert converted.sampling_config.top_p_min == 1.0
    assert converted.sampling_config.top_p_reset_ids == 1
    assert converted.sampling_config.top_p_decay == 1.0
    assert converted.sampling_config.seed == 4
    assert converted.sampling_config.temperature == 1.0
    assert converted.sampling_config.min_tokens == 3
    assert converted.sampling_config.beam_search_diversity_rate == 1.0
    assert converted.sampling_config.repetition_penalty == 1.0
    assert converted.sampling_config.presence_penalty == 2.0
    assert converted.sampling_config.frequency_penalty == 4.0
    assert converted.sampling_config.length_penalty == 8.0
    assert converted.sampling_config.early_stopping == True

    assert converted.output_config.return_log_probs == True
    assert converted.output_config.return_context_logits == True
    assert converted.output_config.return_generation_logits == True
    assert converted.output_config.exclude_input_from_output == True


def test_convert_batched_request(batched_triton_request: MockTritonRequest):
    converted_reqs = convert_request(batched_triton_request,
                                     exclude_input_from_output=True,
                                     decoupled=True)
    assert len(converted_reqs) == 2
    converted0 = converted_reqs[0]
    check_converted_request(converted0)

    converted = converted_reqs[1]

    assert isinstance(converted, trtllm.Request)
    assert converted.input_token_ids == [1, 2]
    assert converted.max_tokens == 3
    assert converted.streaming == False
    assert converted.end_id == 50257
    assert converted.pad_id == 50257
    assert converted.stop_words == [[66], [77]]
    assert converted.bad_words == [[88], [99, 111]]
    assert (converted.embedding_bias == torch.tensor([1., 1., 1.])).all()
    assert converted.logits_post_processor_name is None

    assert isinstance(converted.external_draft_tokens_config,
                      trtllm.ExternalDraftTokensConfig)
    assert converted.external_draft_tokens_config.tokens == [2, 3]
    assert (converted.external_draft_tokens_config.logits == torch.tensor(
        [[1.1, 2.1], [3.1, 4.1]])).all()
    assert converted.external_draft_tokens_config.acceptance_threshold == 0.5

    assert isinstance(converted.prompt_tuning_config, trtllm.PromptTuningConfig)
    print(converted.prompt_tuning_config.embedding_table)
    assert (converted.prompt_tuning_config.embedding_table == torch.tensor(
        [[2.0, 3.0], [4.0, 5.0]])).all()

    assert isinstance(converted.lora_config, trtllm.LoraConfig)
    assert converted.lora_config.task_id == 2
    assert (converted.lora_config.weights == torch.tensor([[3.0, 4.0],
                                                           [5.0, 6.0]])).all()
    assert (converted.lora_config.config == torch.tensor([[11, 12, 13],
                                                          [14, 15, 16]])).all()

    assert converted.sampling_config.beam_width == 3
    assert converted.sampling_config.top_k == 2
    assert converted.sampling_config.top_p == 1.
    assert converted.sampling_config.top_p_min == 0.5
    assert converted.sampling_config.top_p_reset_ids == 3
    assert converted.sampling_config.top_p_decay == pytest.approx(0.1)
    assert converted.sampling_config.seed == 7
    assert converted.sampling_config.temperature == 0.5
    assert converted.sampling_config.min_tokens == 10
    assert converted.sampling_config.beam_search_diversity_rate == pytest.approx(
        0.7)
    assert converted.sampling_config.repetition_penalty == pytest.approx(1.1)
    assert converted.sampling_config.presence_penalty == pytest.approx(2.1)
    assert converted.sampling_config.frequency_penalty == pytest.approx(4.1)
    assert converted.sampling_config.length_penalty == pytest.approx(8.1)
    assert converted.sampling_config.early_stopping == False

    assert converted.output_config.return_log_probs == False
    assert converted.output_config.return_context_logits == False
    assert converted.output_config.return_generation_logits == False
    assert converted.output_config.exclude_input_from_output == True


def test_convert_request(triton_request: MockTritonRequest):
    converted_reqs = convert_request(triton_request,
                                     exclude_input_from_output=True,
                                     decoupled=True)
    assert len(converted_reqs) == 1
    converted = converted_reqs[0]
    check_converted_request(converted)


def test_convert_request_minimal(triton_request_minimal: MockTritonRequest):
    converted_reqs = convert_request(triton_request_minimal,
                                     exclude_input_from_output=False,
                                     decoupled=False)
    assert len(converted_reqs) == 1
    converted = converted_reqs[0]
    assert converted.input_token_ids == [28524, 287, 5093, 12]
    assert converted.max_tokens == 16
    assert converted.streaming == False
    assert converted.end_id is None
    assert converted.pad_id is None
    assert converted.stop_words is None
    assert converted.bad_words is None
    assert converted.embedding_bias is None
    assert converted.logits_post_processor_name is None
    assert converted.external_draft_tokens_config is None
    assert converted.prompt_tuning_config is None
    assert converted.lora_config is None
    assert converted.kv_cache_retention_config is None

    assert converted.sampling_config.beam_width == 1
    assert converted.sampling_config.top_k is None
    assert converted.sampling_config.top_p is None
    assert converted.sampling_config.top_p_min is None
    assert converted.sampling_config.top_p_reset_ids is None
    assert converted.sampling_config.top_p_decay is None
    assert converted.sampling_config.seed is None
    assert converted.sampling_config.temperature is None
    assert converted.sampling_config.min_tokens is None
    assert converted.sampling_config.beam_search_diversity_rate is None
    assert converted.sampling_config.repetition_penalty is None
    assert converted.sampling_config.presence_penalty is None
    assert converted.sampling_config.frequency_penalty is None
    assert converted.sampling_config.length_penalty is None
    assert converted.sampling_config.early_stopping is None

    assert converted.output_config.return_log_probs == False
    assert converted.output_config.return_context_logits == False
    assert converted.output_config.return_generation_logits == False
    assert converted.output_config.exclude_input_from_output == False


def test_kv_cache_retention_config_invalid():

    def check_retention_config(d: Dict[str, np.ndarray], is_valid: bool):
        req = make_mock_triton_request(d)
        if is_valid:
            get_kv_cache_retention_config_from_request(req, 1, 0)
        else:
            with pytest.raises(RuntimeError):
                get_kv_cache_retention_config_from_request(req, 1, 0)

    check_retention_config(
        {
            "retention_token_range_starts": np.array([[0, 100]]),
            "retention_token_range_ends": np.array([[100, 200]])
        }, False)

    check_retention_config(
        {
            "retention_token_range_starts": np.array([[0, 100]]),
            "retention_token_range_ends": np.array([[100, 200]]),
            "retention_token_range_priorities": np.array([[100]])
        }, False)

    check_retention_config(
        {
            "retention_token_range_starts": np.array([[0, 100]]),
            "retention_token_range_ends": np.array([[100, 200]]),
            "retention_token_range_priorities": np.array([[50, 50]])
        }, True)

    check_retention_config(
        {
            "retention_token_range_starts": np.array([[0, 100]]),
            "retention_token_range_ends": np.array([[100, 200]]),
            "retention_token_range_priorities": np.array([[50, 50]]),
            "retention_token_range_durations_ms": np.array([[100]])
        }, False)

    check_retention_config(
        {
            "retention_token_range_starts": np.array([[0, 100]]),
            "retention_token_range_ends": np.array([[100, 200]]),
            "retention_token_range_priorities": np.array([[50, 50]]),
            "retention_token_range_durations_ms": np.array([[100, 50]])
        }, True)

    check_retention_config(
        {
            "retention_token_range_starts": np.array([[0]]),
            "retention_token_range_ends": np.array([[-1]]),
            "retention_token_range_priorities": np.array([[50]]),
            "retention_token_range_durations_ms": np.array([[1000]])
        }, True)


# Need to test with Executor lookahead config.
def test_request_lookahead_config():

    def check_request_lookahead_config(request_config: Dict[str, str],
                                       executor_config, is_valid: bool):
        req = make_mock_triton_request(request_config)

        if is_valid:
            get_lookahead_decoding_config_from_request(req,
                                                       executor_config,
                                                       batch_size=1,
                                                       batch_index=0)
        else:
            with pytest.raises(Exception):
                get_lookahead_decoding_config_from_request(req,
                                                           executor_config,
                                                           batch_size=1,
                                                           batch_index=0)

    # When request and executor lookahead_config are set correctly
    check_request_lookahead_config(
        {
            "lookahead_window_size": np.array([[3]], dtype=np.int32),
            "lookahead_ngram_size": np.array([[3]], dtype=np.int32),
            "lookahead_verification_set_size": np.array([[3]], dtype=np.int32),
        }, trtllm.LookaheadDecodingConfig(3, 3, 3), True)

    # When request lookahead_config is not specified
    check_request_lookahead_config({}, trtllm.LookaheadDecodingConfig(3, 3, 3),
                                   True)

    # When request lookahead_config is incomplete
    check_request_lookahead_config(
        {
            "lookahead_window_size": np.array([[3]], dtype=np.int32),
        }, trtllm.LookaheadDecodingConfig(3, 3, 3), False)

    # When request lookahead_config is incomplete
    check_request_lookahead_config(
        {
            "lookahead_window_size": np.array([[3]], dtype=np.int32),
            "lookahead_ngram_size": np.array([[3]], dtype=np.int32),
        }, trtllm.LookaheadDecodingConfig(3, 3, 3), False)

    # When request lookahead_config is set while executor_lookahead_config is None
    check_request_lookahead_config(
        {
            "lookahead_window_size": np.array([[3]], dtype=np.int32),
            "lookahead_ngram_size": np.array([[3]], dtype=np.int32),
            "lookahead_verification_set_size": np.array([[3]], dtype=np.int32),
        }, None, False)


def test_convert_request_invalid():
    with pytest.raises(Exception, match="A value is required for input_ids"):
        no_input_ids = MockTritonRequest({
            "request_output_len":
            MockTritonTensor("request_output_len", np.array([[128]]))
        })
        convert_request(no_input_ids, False, False)
    with pytest.raises(Exception, match="Invalid format for input_ids"):
        bad_input_ids = MockTritonRequest(
            {"input_ids": MockTritonTensor("input_ids", np.array([]))})
        convert_request(bad_input_ids, False, False)
    with pytest.raises(Exception,
                       match="A value is required for request_output_len"):
        no_output_len = MockTritonRequest(
            {"input_ids": MockTritonTensor("input_ids", np.array([[1, 2, 3]]))})
        convert_request(no_output_len, False, False)
    with pytest.raises(Exception,
                       match="Streaming is only supported in decoupled mode."):
        streaming_non_decoupled = MockTritonRequest({
            "input_ids":
            MockTritonTensor("input_ids", np.array([[1, 2, 3]])),
            "request_output_len":
            MockTritonTensor("request_output_len", np.array([[128]])),
            "streaming":
            MockTritonTensor("streaming", np.array([[True]])),
        })
        convert_request(streaming_non_decoupled, False, False)


def test_convert_response(trtllm_response: trtllm.Response):
    batch_index = 2
    batch_size = 3
    num_return_sequences = 1
    response, is_final, output_length = convert_response(
        trtllm_response, batch_index, batch_size, num_return_sequences)
    assert is_final == True
    assert (response.tensors["output_ids"].as_numpy() == np.array([[1, 2,
                                                                    3]])).all()
    assert (response.tensors["sequence_length"].as_numpy() == np.array(
        [[3]])).all()
    assert (response.tensors["cum_log_probs"].as_numpy() == np.array([1])).all()
    assert (response.tensors["output_log_probs"].as_numpy() == np.array(
        [[1, 3]])).all()
    assert (response.tensors["context_logits"].as_numpy() == np.ones(
        (3, 10), dtype=np.float32)).all()
    assert (response.tensors["generation_logits"].as_numpy() == np.ones(
        (1, 5, 10), dtype=np.float32)).all()
    assert (response.tensors["batch_index"].as_numpy() == np.array(
        [[batch_index]])).all()


def test_convert_response_minimal(trtllm_response_minimal: trtllm.Response):
    batch_index = 2
    batch_size = 3
    num_return_sequences = 1
    response, is_final, output_length = convert_response(
        trtllm_response_minimal, batch_index, batch_size, num_return_sequences)
    assert is_final == False
    assert (response.tensors["output_ids"].as_numpy() == np.array([[1, 2,
                                                                    3]])).all()
    assert (response.tensors["sequence_length"].as_numpy() == np.array(
        [[3]])).all()
    assert "cum_log_probs" not in response.tensors
    assert "output_log_probs" not in response.tensors
    assert "output_log_probs" not in response.tensors
    assert "context_logits" not in response.tensors
    assert "generation_logits" not in response.tensors
    assert (response.tensors["batch_index"].as_numpy() == np.array(
        [[batch_index]])).all()


def test_convert_response_error(trtllm_response_error: trtllm.Response):
    batch_index = 2
    batch_size = 3
    num_return_sequences = 1
    response, is_final, output_length = convert_response(
        trtllm_response_error, batch_index, batch_size, num_return_sequences)
    assert is_final == True
    assert response.has_error() and response.error.message == "internal error"


def test_convert_scheduler_policy():
    assert convert_scheduler_policy(
        "max_utilization") == trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION
    assert convert_scheduler_policy(
        "guaranteed_no_evict"
    ) == trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    with pytest.raises(
            Exception,
            match="batch_scheduler_policy value of 'other' is not supported"):
        convert_scheduler_policy("other")


def test_convert_batching_type():
    assert convert_batching_type(
        "inflight_fused_batching") == trtllm.BatchingType.INFLIGHT
    assert convert_batching_type(
        "inflight_batching") == trtllm.BatchingType.INFLIGHT
    assert convert_batching_type("v1") == trtllm.BatchingType.STATIC
    with pytest.raises(
            Exception,
            match="gpt_model_type value of 'other' is not supported"):
        convert_batching_type("other")


def test_convert_decoding_mode():
    assert convert_decoding_mode(None) is None
    assert convert_decoding_mode("auto").isAuto()
    assert convert_decoding_mode("top_k").isTopK()
    assert convert_decoding_mode("top_p").isTopP()
    assert convert_decoding_mode("top_k_top_p").isTopKandTopP()
    assert convert_decoding_mode("beam_search").isBeamSearch()
    assert convert_decoding_mode("medusa").isMedusa()
    assert convert_decoding_mode("redrafter").isExplicitDraftTokens()
    assert convert_decoding_mode("lookahead").isLookahead()
    assert convert_decoding_mode("eagle").isEagle()
    with pytest.raises(Exception,
                       match="decoding_mode value of 'other' is not supported"):
        convert_decoding_mode("other")


@pytest.fixture
def model_config() -> Dict:
    config = {
        "max_beam_width": "2",
        "enable_chunked_context": "true",
        "normalize_log_probs": "false",
        "gpt_model_type": "inflight_batching",
        "medusa_choices": "{1, 2, 3, 4}, {5, 6, 7}",
        "decoding_mode": "medusa",
        "batch_scheduler_policy": "max_utilization",
        "enable_kv_cache_reuse": "false",
        "max_tokens_in_paged_kv_cache": "1",
        "max_attention_window_size": "2",
        "sink_token_length": "3",
        "kv_cache_free_gpu_mem_fraction": "0.5",
        "cross_kv_cache_fraction": "0.5",
        "kv_cache_host_memory_bytes": "4",
        "kv_cache_onboard_blocks": "false",
        "gpu_device_ids": "0,1,2,3",
        "executor_worker_path": str(os.path.abspath(__file__)),
        "lora_cache_optimal_adapter_size": "1",
        "lora_cache_max_adapter_size": "2",
        "lora_cache_gpu_memory_fraction": "0.5",
        "lora_cache_host_memory_bytes": "4",
        "lora_prefetch_dir": "",
        "enable_context_fmha_fp32_acc": "true"
    }
    return {"parameters": {k: {"string_value": v} for k, v in config.items()}}


def test_get_executor_config(model_config: Dict):
    os.environ["TRTLLM_ORCHESTRATOR"] = "0"
    config = TritonPythonModel().get_executor_config(model_config)
    assert config.max_beam_width == 2
    assert config.enable_chunked_context == True
    assert config.normalize_log_probs == False
    assert config.batching_type == trtllm.BatchingType.INFLIGHT
    assert config.decoding_config.medusa_choices == [[1, 2, 3, 4], [5, 6, 7]]
    assert config.decoding_config.decoding_mode.isMedusa()
    assert config.scheduler_config.capacity_scheduler_policy == trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION
    assert config.kv_cache_config.enable_block_reuse == False
    assert config.kv_cache_config.max_tokens == 1
    assert config.kv_cache_config.max_attention_window == [2]
    assert config.kv_cache_config.sink_token_length == 3
    assert config.kv_cache_config.free_gpu_memory_fraction == 0.5
    assert config.kv_cache_config.cross_kv_cache_fraction == 0.5
    assert config.kv_cache_config.host_cache_size == 4
    assert config.kv_cache_config.onboard_blocks == False
    assert config.parallel_config.device_ids == [0, 1, 2, 3]
    assert config.parallel_config.orchestrator_config is None
    assert config.peft_cache_config.optimal_adapter_size == 1
    assert config.peft_cache_config.max_adapter_size == 2
    assert config.peft_cache_config.device_cache_percent == 0.5
    assert config.peft_cache_config.host_cache_size == 4
    assert config.iter_stats_max_iterations == 1000
    assert config.request_stats_max_iterations == 0
    assert config.logits_post_processor_config is None
    assert config.extended_runtime_perf_knob_config.enable_context_fmha_fp32_acc == True
    assert config.extended_runtime_perf_knob_config.multi_block_mode == True
    del os.environ["TRTLLM_ORCHESTRATOR"]


def test_get_executor_config_orchestrator_mode(model_config: Dict):
    os.environ["TRTLLM_ORCHESTRATOR"] = "1"
    config = TritonPythonModel().get_executor_config(model_config)
    assert config.parallel_config.device_ids == [0, 1, 2, 3]
    assert config.parallel_config.orchestrator_config.is_orchestrator == True
    assert config.parallel_config.orchestrator_config.worker_executable_path == str(
        os.path.abspath(__file__))
    del os.environ["TRTLLM_ORCHESTRATOR"]


def test_get_executor_config_minimal():
    if "TRTLLM_ORCHESTRATOR" in os.environ:
        del os.environ["TRTLLM_ORCHESTRATOR"]
    config = TritonPythonModel().get_executor_config({"parameters": {}})
    assert config.max_beam_width == 1
    assert config.enable_chunked_context == False
    assert config.normalize_log_probs == True
    assert config.batching_type == trtllm.BatchingType.INFLIGHT
    assert config.decoding_config.decoding_mode is None
    assert config.decoding_config.medusa_choices is None
    assert config.decoding_config.eagle_config is None
    assert config.decoding_config.lookahead_decoding_config is None
    assert config.scheduler_config.capacity_scheduler_policy == trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    assert config.kv_cache_config.enable_block_reuse == True
    assert config.kv_cache_config.max_tokens is None
    assert config.kv_cache_config.max_attention_window is None
    assert config.kv_cache_config.sink_token_length is None
    assert config.kv_cache_config.free_gpu_memory_fraction is None
    assert config.kv_cache_config.cross_kv_cache_fraction is None
    assert config.kv_cache_config.host_cache_size is None
    assert config.kv_cache_config.onboard_blocks == True
    assert config.parallel_config is None
    assert config.peft_cache_config.optimal_adapter_size == 8
    assert config.peft_cache_config.max_adapter_size == 64
    assert config.peft_cache_config.device_cache_percent is None
    assert config.peft_cache_config.host_cache_size is None
    assert config.iter_stats_max_iterations == 1000
    assert config.request_stats_max_iterations == 0
    assert config.logits_post_processor_config is None
    assert config.extended_runtime_perf_knob_config.enable_context_fmha_fp32_acc == False
    assert config.extended_runtime_perf_knob_config.multi_block_mode == True


def test_convert_timestamp_to_seconds():
    assert convert_timestamp_to_seconds("01-01-1970 00:00:00.000000") == 0
    assert convert_timestamp_to_seconds(
        "05-17-2024 23:28:39.000000") == 1715988519
