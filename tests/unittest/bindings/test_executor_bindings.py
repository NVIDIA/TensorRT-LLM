import datetime
import json
import os as _os
import pickle
import random
import sys as _sys
import time
import typing as tp
from pathlib import Path

import numpy as np
import pytest
import torch
from binding_test_utils import *
from pydantic import BaseModel

import tensorrt_llm.bindings.executor as trtllm
import tensorrt_llm.version as trtllm_version
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.models.modeling_utils import PretrainedConfig

_sys.path.append(_os.path.join(_os.path.dirname(__file__), '..'))
import inspect

from utils.cpp_paths import *
from utils.llm_data import llm_models_root
from utils.util import skip_pre_hopper


@pytest.fixture
def model_files(llm_root: Path, resource_path: Path, results_data_path: Path):
    # Model engines and expected outputs need to be generated.
    if not results_data_path.exists():
        model_cache = llm_models_root()
        model_cache_arg = ["--model_cache", str(model_cache)
                           ] if model_cache is not None else []
        prepare_model_tests(llm_root, resource_path, "gpt", model_cache_arg)


@pytest.fixture
def lora_config_paths(llm_root: Path, resource_path: Path,
                      lora_config_path: Path):
    if not lora_config_path.exists():
        prepare_lora_configs(llm_root, resource_path, lora_config_path)
    return (lora_config_path / "source.npy", lora_config_path / "config.npy")


def get_expected_num_tokens(prompt_len, max_tokens, streaming,
                            exclude_input_from_output):
    if not streaming and not exclude_input_from_output:
        return prompt_len + max_tokens
    return max_tokens


def test_executor_valid_ctor(model_files, model_path):
    executor_config = trtllm.ExecutorConfig(
        1, kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)


def test_executor_from_memory(model_files, model_path):
    executor_config = trtllm.ExecutorConfig(
        1, kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    engine_buffer = open(model_path / "rank0.engine", mode="rb").read()
    json_config_str = open(model_path / "config.json", 'r').read()
    executor = trtllm.Executor(engine_buffer, json_config_str,
                               trtllm.ModelType.DECODER_ONLY, executor_config)


def test_executor_with_managed_weights(model_files, model_path):
    """Test executor constructor with standard dtypes in managed weights."""

    executor_config = trtllm.ExecutorConfig(
        1, kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    engine_buffer = open(model_path / "rank0.engine", mode="rb").read()
    json_config_str = open(model_path / "config.json", 'r').read()

    managed_weights = {
        "weight_float32":
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "weight_int32":
        np.array([[1, 2], [3, 4]], dtype=np.int32),
        "weight_int64":
        np.array([[1, 2], [3, 4]], dtype=np.int64),
        "weight_int8":
        np.array([[1, 2], [3, 4]], dtype=np.int8),
        "weight_fp16":
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16),
        "weight_bf16":
        torch_to_numpy(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)),
        "weight_fp8":
        torch_to_numpy(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float8_e4m3fn)),
    }

    executor = trtllm.Executor(engine_buffer, json_config_str,
                               trtllm.ModelType.DECODER_ONLY, executor_config,
                               managed_weights)

    assert executor.can_enqueue_requests() == True


def test_executor_invalid_ctor():
    executor_config = trtllm.ExecutorConfig(
        1, kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    invalid_path = "Bla"
    try:
        executor = trtllm.Executor(invalid_path, trtllm.ModelType.DECODER_ONLY,
                                   executor_config)
        assert False, "Expected an error"
    except Exception as e:
        assert "File does not exist" in str(e)


def test_shutdown(model_files, model_path):

    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             streaming=False,
                             sampling_config=trtllm.SamplingConfig())

    # Enqueue the request
    assert executor.can_enqueue_requests() == True
    req_id = executor.enqueue_request(request)

    executor.shutdown()
    assert executor.can_enqueue_requests() == False

    with pytest.raises(Exception):
        executor.enqueue_request(request)
    with pytest.raises(Exception):
        executor.await_responses()
    with pytest.raises(Exception):
        executor.get_latest_iteration_stats()
    with pytest.raises(Exception):
        executor.get_latest_request_stats()
    with pytest.raises(Exception):
        executor.get_latest_debug_tensors()
    with pytest.raises(Exception):
        executor.cancel_request(req_id)
    with pytest.raises(Exception):
        executor.get_num_responses_ready(req_id)


def test_embedding_bias(model_files, model_path):
    streaming = False
    exclude_input_from_output = False
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_tokens = 5
    input_tokens = [1, 2, 3, 4]
    # Set embedding bias so "biased_output" is always picked
    biased_output = 10
    vocab_size_padded = 50257
    embedding_bias = torch.zeros(vocab_size_padded)
    embedding_bias[biased_output] = torch.finfo(torch.float32).max
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             streaming=streaming,
                             sampling_config=trtllm.SamplingConfig(),
                             output_config=output_config,
                             embedding_bias=embedding_bias)

    # Enqueue the request
    request_id = executor.enqueue_request(request)

    # Get the new tokens
    tokens = []
    done = False
    i = 0
    max_wait_ms = 10000
    while not done and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(request_id, wait_time)
        for response in responses:
            assert not response.has_error(
            ), f"Request id {request_id} failed with err {response.error_msg}"
            result = response.result
            done = result.is_final
            new_tokens = result.output_token_ids[beam_width - 1]
            tokens.extend(new_tokens)
        i += 1
    assert i < max_wait_ms
    assert len(tokens) == get_expected_num_tokens(
        len(input_tokens), max_tokens, streaming,
        exclude_input_from_output), f"{request_id}"
    # All generated tokens should equal biased_output
    assert tokens[-max_tokens:] == [biased_output] * max_tokens


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exclude_input_from_output", [False])
def test_single_request(streaming: bool, exclude_input_from_output: bool,
                        model_files, model_path):
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             streaming=streaming,
                             sampling_config=trtllm.SamplingConfig(),
                             output_config=output_config)

    # Enqueue the request
    request_id = executor.enqueue_request(request)

    # Get the new tokens
    tokens = []
    done = False
    i = 0
    max_wait_ms = 10000
    while not done and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(request_id, wait_time)
        for response in responses:
            assert not response.has_error(
            ), f"Request id {request_id} failed with err {response.error_msg}"
            result = response.result
            done = result.is_final
            new_tokens = result.output_token_ids[beam_width - 1]
            tokens.extend(new_tokens)
        i += 1
    assert i < max_wait_ms
    assert len(tokens) == get_expected_num_tokens(
        len(input_tokens), max_tokens, streaming,
        exclude_input_from_output), f"{request_id}"

    executor.get_latest_iteration_stats()
    executor.get_latest_request_stats()
    executor.get_latest_debug_tensors()


def test_single_request_lora(model_files, model_path_lora, lora_config_paths):
    streaming = False
    exclude_input_from_output = False
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    beam_width = 1

    peft_cache_config = trtllm.PeftCacheConfig(num_put_workers=4,
                                               num_ensure_workers=4)
    executor_config = trtllm.ExecutorConfig(
        1,
        peft_cache_config=peft_cache_config,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path_lora, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_tokens = 5
    input_tokens = [1, 2, 3, 4]
    lora_weights = torch.tensor(np.load(lora_config_paths[0])).half()
    lora_config = torch.tensor(np.load(lora_config_paths[1]))
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             streaming=streaming,
                             sampling_config=trtllm.SamplingConfig(),
                             output_config=output_config,
                             lora_config=trtllm.LoraConfig(
                                 0, lora_weights, lora_config))

    # Enqueue the request
    request_id = executor.enqueue_request(request)

    # Get the new tokens
    tokens = []
    done = False
    i = 0
    max_wait_ms = 10000
    while not done and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(request_id, wait_time)
        for response in responses:
            assert not response.has_error(
            ), f"Request id {request_id} failed with err {response.error_msg}"
            result = response.result
            done = result.is_final
            new_tokens = result.output_token_ids[beam_width - 1]
            tokens.extend(new_tokens)
        i += 1
    assert i < max_wait_ms
    assert len(tokens) == get_expected_num_tokens(
        len(input_tokens), max_tokens, streaming,
        exclude_input_from_output), f"{request_id}"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exclude_input_from_output", [False])
def test_multi_request(streaming: bool, exclude_input_from_output: bool,
                       model_files, model_path):
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    num_requests = 20
    max_prompt_len = 20
    max_max_tokens = 20
    end_id = -1

    # Enqueue the requests
    tokens = {}
    expected_num_tokens = {}
    for i in range(num_requests):
        prompt_len = random.randint(1, max_prompt_len)
        max_tokens = random.randint(1, max_max_tokens)
        input_tokens = [1] * prompt_len

        # Some requests has num_return_sequences > 1.
        num_return_sequences = 2 if i % 5 == 1 else 1

        request = trtllm.Request(input_tokens,
                                 max_tokens=max_tokens,
                                 streaming=streaming,
                                 sampling_config=trtllm.SamplingConfig(
                                     num_return_sequences=num_return_sequences),
                                 output_config=output_config,
                                 end_id=end_id)
        request_id = executor.enqueue_request(request)
        tokens[request_id] = [
            [] for _ in range(request.sampling_config.num_return_sequences)
        ]
        expected_num_tokens[request_id] = get_expected_num_tokens(
            prompt_len, max_tokens, streaming, exclude_input_from_output)

    # Get the new tokens for each request
    num_finished = 0
    i = 0
    num_responses = 0
    max_wait_ms = 10000
    while num_finished < num_requests and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(wait_time)
        for response in responses:
            num_responses += 1
            assert not response.has_error(
            ), f"Request id {response.request_id} failed with err {response.error_msg}"
            result = response.result
            num_finished += result.is_final
            new_tokens = result.output_token_ids[beam_width - 1]
            tokens[response.request_id][result.sequence_index].extend(
                new_tokens)
        i += 1
    assert i < max_wait_ms

    for request_id in expected_num_tokens:
        for actual_tokens in tokens[request_id]:
            assert len(actual_tokens) == expected_num_tokens[request_id]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exclude_input_from_output", [False])
def test_multi_request_with_ids(streaming: bool,
                                exclude_input_from_output: bool, model_files,
                                model_path):
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    num_requests = 20
    max_prompt_len = 20
    max_max_tokens = 20
    end_id = -1

    # Enqueue the requests
    tokens = {}
    expected_num_tokens = {}
    for i in range(num_requests):
        prompt_len = random.randint(1, max_prompt_len)
        max_tokens = random.randint(1, max_max_tokens)
        input_tokens = [1] * prompt_len
        num_return_sequences = 2 if i % 5 == 1 else 1

        request = trtllm.Request(input_tokens,
                                 max_tokens=max_tokens,
                                 streaming=streaming,
                                 sampling_config=trtllm.SamplingConfig(
                                     num_return_sequences=num_return_sequences),
                                 output_config=output_config,
                                 end_id=end_id)
        request_id = executor.enqueue_request(request)
        tokens[request_id] = [
            [] for _ in range(request.sampling_config.num_return_sequences)
        ]
        expected_num_tokens[request_id] = get_expected_num_tokens(
            prompt_len, max_tokens, streaming, exclude_input_from_output)

    # Get the new tokens for each request
    num_finished = 0
    i = 0
    num_responses = 0
    max_wait_ms = 10000
    while num_finished < num_requests and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        id_responses = executor.await_responses(list(tokens.keys()), wait_time)
        for responses in id_responses:
            for response in responses:
                num_responses += 1
                # Allow response with error only if await_response processed a terminated request id
                if response.has_error():
                    terminated_request_error = "ReqId " + str(
                        response.request_id
                    ) + " has already been processed and was terminated."
                    assert response.error_msg == terminated_request_error, (
                        f"Request id {response.request_id} failed with err "
                        f"{response.error_msg}")
                else:
                    result = response.result
                    num_finished += result.is_final
                    new_tokens = result.output_token_ids[beam_width - 1]
                    tokens[response.request_id][result.sequence_index].extend(
                        new_tokens)
            i += 1
    assert i < max_wait_ms

    for request_id in expected_num_tokens:
        for seq_idx, actual_tokens in enumerate(tokens[request_id]):
            assert len(actual_tokens) == expected_num_tokens[request_id]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exclude_input_from_output", [False])
def test_get_num_responses_ready(streaming: bool,
                                 exclude_input_from_output: bool, model_files,
                                 model_path):
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    executor_config = trtllm.ExecutorConfig(
        1, kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    max_prompt_len = 20
    max_max_tokens = 20

    # Enqueue the requests
    num_requests = random.randint(1, 50)
    num_expected_responses = 0
    req_num_expected_responses = {}
    for i in range(num_requests):
        prompt_len = random.randint(1, max_prompt_len)
        max_tokens = random.randint(1, max_max_tokens)
        num_return_sequences = 2 if i % 5 == 1 else 1

        request = trtllm.Request([1] * prompt_len,
                                 max_tokens=max_tokens,
                                 streaming=streaming,
                                 sampling_config=trtllm.SamplingConfig(
                                     num_return_sequences=num_return_sequences),
                                 output_config=output_config)
        request_id = executor.enqueue_request(request)
        req_num_expected_responses[request_id] = (
            (max_tokens if streaming else 1) * num_return_sequences)
        num_expected_responses += req_num_expected_responses[request_id]

    i = 0
    num_ready = 0
    max_wait_ms = 10000
    while num_ready < num_expected_responses and i < max_wait_ms:
        num_ready = 0
        for request_id in req_num_expected_responses:
            num_ready += executor.get_num_responses_ready(request_id)
        time.sleep(0.001)
        i += 1
    assert i < max_wait_ms

    for request_id in req_num_expected_responses:
        num_ready = executor.get_num_responses_ready(request_id)
        assert num_ready == req_num_expected_responses[request_id]
    assert executor.get_num_responses_ready() == num_expected_responses


@pytest.mark.parametrize("batching_type", [trtllm.BatchingType.INFLIGHT])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("beam_width", [1])
@pytest.mark.parametrize("compute_log_probs", [False, True])
@pytest.mark.parametrize("exclude_input_from_output", [False])
@pytest.mark.parametrize("return_context_logits", [False, True])
@pytest.mark.parametrize("return_generation_logits", [False, True])
def test_token_comparison(batching_type: trtllm.BatchingType, streaming: bool,
                          beam_width: int, compute_log_probs: bool,
                          exclude_input_from_output: bool,
                          return_context_logits: bool,
                          return_generation_logits: bool, model_files,
                          model_path, model_path_return_logits, input_data_path,
                          results_data_path, results_data_path_beam_width_2):
    if streaming and beam_width > 1:
        pytest.skip("Test does not support streaming with beam search")

    vocab_size_padded = 50257
    pad_id = 50256
    remove_input = not exclude_input_from_output and not streaming

    def load_test_data(input_path, results_path):
        # Inputs
        assert input_path.is_file()
        given_input = np.load(input_path).astype("int32")
        input_shape = given_input.shape
        assert len(input_shape) == 2
        max_input_length = input_shape[1]
        given_input_lengths = sequence_lengths(given_input, pad_id)
        assert np.all(given_input_lengths <= max_input_length)
        # Expected results
        assert results_path.is_file()
        expected_outputs = np.load(results_path).astype("int32")
        output_shape = expected_outputs.shape
        assert len(output_shape) == 2
        assert input_shape[0] * beam_width == output_shape[0]
        max_seq_length = output_shape[1]
        max_tokens = max_seq_length - max_input_length

        end_ids = [pad_id for _ in range(len(given_input_lengths))]
        expected_lengths = []
        for i in range(len(given_input_lengths)):
            expected_lengths.append([
                given_input_lengths[i] + max_tokens for _ in range(beam_width)
            ])

        test_data = {
            "expected_output_ids": expected_outputs,
            "expected_output_lengths": expected_lengths,
            "max_seq_length": max_seq_length,
            "end_ids": end_ids
        }
        return given_input, given_input_lengths, max_input_length, test_data

    def validate_results_shapes(result, input_length, max_output_len,
                                beam_tokens):
        if compute_log_probs:
            assert result.cum_log_probs is not None
            assert result.log_probs is not None
            assert len(result.cum_log_probs) == beam_width
            assert len(result.log_probs) == beam_width
            for beam in range(beam_width):
                expected_len = len(
                    beam_tokens[beam]) - (input_length if remove_input else 0)
                assert len(result.log_probs[beam]) == expected_len
        else:
            assert result.cum_log_probs is None
            assert result.log_probs is None
        if return_context_logits:
            assert result.context_logits is not None
            assert len(result.context_logits.shape) == 2
            assert list(result.context_logits.shape) == [
                input_length, vocab_size_padded
            ]
        else:
            assert result.context_logits is None
        if return_generation_logits:
            assert len(result.generation_logits.shape) == 3
            if streaming:
                assert list(result.generation_logits.shape) == [
                    max_output_len, beam_width, vocab_size_padded
                ] or list(result.generation_logits.shape) == [
                    1, beam_width, vocab_size_padded
                ]
            else:
                assert list(result.generation_logits.shape) == [
                    beam_width, max_output_len, vocab_size_padded
                ]

    def verify_output(beam_tokens, test_data, given_input_lengths):

        for batch_id, seq_tokens in beam_tokens.items():
            input_length = given_input_lengths[batch_id]
            end_id = test_data["end_ids"][batch_id]
            for tokens in seq_tokens:
                for beam in range(beam_width):

                    predicted_tokens = tokens[beam]
                    if remove_input:
                        predicted_tokens = predicted_tokens[input_length:]
                    expected_length = test_data["expected_output_lengths"][
                        batch_id][beam] - input_length
                    assert len(predicted_tokens) == expected_length

                    expected_tokens = test_data["expected_output_ids"][
                        batch_id * beam_width + beam][input_length:]

                    # From experiments find out when set return_context_logits
                    # or return_generation_logits, the predicted_tokens cannot match with expected_tokens
                    # Fixed by comparing partial output tokens like in c++ test
                    compare_length = 2 if (
                        return_context_logits
                        or return_generation_logits) else len(predicted_tokens)

                    for i in range(compare_length):
                        if expected_tokens[i] == end_id:
                            break
                        # Predicted: [21221, 290, 373, 257, 2888, 286, 262, 4141]
                        # Expected: [21221, 290, 257, 4255, 379, 262, 1957, 7072]
                        # generation logits are almost same at token ids 257 and 373,
                        # which causes unstable generation results.
                        assert predicted_tokens[i] == expected_tokens[i], \
                            f"Predicted: {predicted_tokens} vs Expected: {expected_tokens}"

    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output
    output_config.return_log_probs = compute_log_probs
    output_config.return_generation_logits = return_generation_logits
    output_config.return_context_logits = return_context_logits
    # Change free_gpu_memory_fraction to solve OOM error
    kv_cache_config = trtllm.KvCacheConfig(False, free_gpu_memory_fraction=0.3)
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor_config.batching_type = batching_type
    executor_config.kv_cache_config = kv_cache_config
    if return_generation_logits:
        executor_config.gather_generation_logits = True

    if return_context_logits or return_generation_logits:
        model_path = model_path_return_logits
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Load test data
    results_path = results_data_path if beam_width == 1 else results_data_path_beam_width_2
    given_input, given_input_lengths, max_input_length, test_data = load_test_data(
        input_data_path, results_path)

    # Create requests from input data
    num_requests = len(given_input_lengths)
    requests = []
    req_max_tokens = []

    for i in range(num_requests):
        input_len = given_input_lengths[i]
        max_tokens = test_data["max_seq_length"] - max_input_length
        req_max_tokens.append(max_tokens)
        req_tokens = given_input[i][:input_len]
        num_return_sequences = 2 if i % 5 == 1 else 1
        request = trtllm.Request(req_tokens,
                                 max_tokens=max_tokens,
                                 streaming=streaming,
                                 sampling_config=trtllm.SamplingConfig(
                                     beam_width,
                                     num_return_sequences=num_return_sequences),
                                 output_config=output_config,
                                 end_id=-1)
        requests.append(request)

    req_ids = executor.enqueue_requests(requests)

    req_to_batch_id = {req_ids[i]: i for i in range(len(requests))}
    tokens = {
        i: [[[] for _ in range(beam_width)]
            for _ in range(req.sampling_config.num_return_sequences)]
        for i, req in enumerate(requests)
    }

    num_finished = 0
    i = 0
    num_responses = 0
    max_wait_ms = 10000
    while num_finished < num_requests and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(wait_time)
        for response in responses:
            num_responses += 1
            assert not response.has_error(
            ), f"Request id {response.request_id} failed with err {response.error_msg}"
            result = response.result
            num_finished += result.is_final

            batch_id = req_to_batch_id[response.request_id]
            for beam in range(beam_width):
                new_tokens = result.output_token_ids[beam]
                tokens[batch_id][result.sequence_index][beam] += new_tokens

            validate_results_shapes(result, given_input_lengths[batch_id],
                                    req_max_tokens[batch_id],
                                    tokens[batch_id][result.sequence_index])
        i += 1
    assert i < max_wait_ms
    verify_output(tokens, test_data, given_input_lengths)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("beam_width", [1])
def test_finish_reason(streaming: bool, beam_width: int, model_files,
                       model_path):
    if streaming and beam_width > 1:
        pytest.skip("Test does not support streaming with beam search")
    executor = trtllm.Executor(
        model_path, trtllm.ModelType.DECODER_ONLY,
        trtllm.ExecutorConfig(
            beam_width,
            kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5)))
    requests = [
        # Finish due to length.
        trtllm.Request([1, 2, 3, 4],
                       max_tokens=5,
                       streaming=streaming,
                       sampling_config=trtllm.SamplingConfig(beam_width)),
        # Finish due to end id.
        trtllm.Request([1, 2, 3, 4],
                       max_tokens=5,
                       streaming=streaming,
                       sampling_config=trtllm.SamplingConfig(beam_width),
                       end_id=4),
        # Finish due to stop word.
        trtllm.Request([1, 2, 3, 4],
                       max_tokens=5,
                       streaming=streaming,
                       sampling_config=trtllm.SamplingConfig(beam_width),
                       stop_words=[[4, 2]]),
    ]
    req_ids = executor.enqueue_requests(requests)
    req_to_batch_id = {req_ids[i]: i for i in range(len(requests))}

    num_finished = 0
    i = 0
    num_responses = 0
    max_wait_ms = 10000
    while num_finished < len(requests) and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(wait_time)
        for response in responses:
            num_responses += 1
            assert not response.has_error(
            ), f"Request id {response.request_id} failed with err {response.error_msg}"
            result = response.result
            num_finished += result.is_final
            batch_id = req_to_batch_id[response.request_id]

            # Non final results should have "NOT_FINISHED". Revise this when streaming + beam_width > 1 is enabled.
            if not result.is_final:
                assert all([
                    r == trtllm.FinishReason.NOT_FINISHED
                    for r in result.finish_reasons
                ])
            # Check if finish reason is correct.
            elif batch_id == 0:
                assert all([
                    r == trtllm.FinishReason.LENGTH
                    for r in result.finish_reasons
                ])
            elif batch_id == 1:
                assert all([
                    r == trtllm.FinishReason.END_ID
                    for r in result.finish_reasons
                ])
            elif batch_id == 2:
                assert all([
                    r == trtllm.FinishReason.STOP_WORDS
                    for r in result.finish_reasons
                ])
        i += 1
    assert i < max_wait_ms


def test_gpt_executor_timed_out(model_files, model_path):
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # No requests enqueued, expect no responses
    num_responses_ready = executor.get_num_responses_ready()
    assert num_responses_ready == 0

    wait_time = datetime.timedelta(milliseconds=10)
    responses = executor.await_responses(wait_time)
    assert len(responses) == 0


def test_single_request_invalid_inputs(model_files, model_path):
    streaming = True
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    max_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             streaming=streaming)
    # Invalid embedding bias shape
    embedding_bias = torch.ones(1)
    request.embedding_bias = embedding_bias
    expected_error_msg = "embedding bias shape is not as expected"

    request_id = executor.enqueue_request(request)

    done = False
    i = 0
    max_wait_ms = 10000
    while not done and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(request_id, wait_time)
        for response in responses:
            assert response.has_error(), "Expected an error"
            assert expected_error_msg in response.error_msg
            done = True
        i += 1
    assert done


def test_sampling_config():
    beam_width = 1
    kwargs = {
        "top_k": 2,
        "top_p": 1.0,
        "top_p_min": 1.0,
        "top_p_reset_ids": 3,
        "top_p_decay": 1.0,
        "seed": 7,
        "temperature": 1.0,
        "min_tokens": 4,
        "beam_search_diversity_rate": 1.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.0,
        "frequency_penalty": 1.0,
        "length_penalty": 1.0,
        "early_stopping": 5,
        "num_return_sequences": 2,
    }
    config = trtllm.SamplingConfig(beam_width, **kwargs)
    for k, v in kwargs.items():
        assert getattr(config, k) == v
    del config

    config = trtllm.SamplingConfig(beam_width)
    assert config.beam_width == beam_width
    for k in kwargs:
        assert getattr(config, k) is None


def test_output_config():
    config = trtllm.OutputConfig()
    assert config.return_log_probs == False
    assert config.return_context_logits == False
    assert config.return_generation_logits == False
    assert config.exclude_input_from_output == False
    assert config.return_encoder_output == False
    assert config.return_perf_metrics == False
    assert config.additional_model_outputs is None

    config = trtllm.OutputConfig(
        True, False, True, False, True, False,
        list([trtllm.AdditionalModelOutput("topKLogits", True)]))
    assert config.return_log_probs == True
    assert config.return_context_logits == False
    assert config.return_generation_logits == True
    assert config.exclude_input_from_output == False
    assert config.return_encoder_output == True
    assert config.return_perf_metrics == False
    assert len(config.additional_model_outputs) == 1
    additional_model_output = config.additional_model_outputs[0]
    assert additional_model_output.name == "topKLogits"
    assert additional_model_output.gather_context == True


def test_output_config_pickle():
    config = trtllm.OutputConfig(
        True, False, True, False, True, False,
        list([trtllm.AdditionalModelOutput("topKLogits", True)]))
    config_copy = pickle.loads(pickle.dumps(config))
    assert config_copy.return_log_probs == True
    assert config_copy.return_context_logits == False
    assert config_copy.return_generation_logits == True
    assert config_copy.exclude_input_from_output == False
    assert config_copy.return_encoder_output == True
    assert config_copy.return_perf_metrics == False
    assert len(config_copy.additional_model_outputs) == 1
    additional_model_output = config_copy.additional_model_outputs[0]
    assert additional_model_output.name == "topKLogits"
    assert additional_model_output.gather_context == True


def test_external_draft_tokens_config():
    tokens = [1, 2, 3]
    config = trtllm.ExternalDraftTokensConfig(tokens)
    assert config.tokens == tokens
    assert config.logits is None
    assert config.acceptance_threshold is None
    del config

    logits = torch.ones(3, 1)
    acceptance_threshold = 1.0
    fast_logits = False
    config = trtllm.ExternalDraftTokensConfig(tokens, logits,
                                              acceptance_threshold, fast_logits)
    assert config.tokens == tokens
    assert (config.logits == logits).all()
    assert config.acceptance_threshold == acceptance_threshold
    assert config.fast_logits == fast_logits


def test_prompt_tuning_config():
    embedding_table = torch.ones(100, 64)
    config = trtllm.PromptTuningConfig(embedding_table)
    assert (config.embedding_table == embedding_table).all()


def test_multimodal_embedding():

    def get_base_kwargs():
        return {
            "input_token_ids": [1, 2, 3],
            "max_tokens":
            1,
            "streaming":
            False,
            "sampling_config":
            trtllm.SamplingConfig(),
            "output_config":
            trtllm.OutputConfig(),
            "end_id":
            -1,
            "pad_id":
            -2,
            "bad_words": [[4, 5, 6]],
            "stop_words": [[7, 8, 9]],
            "embedding_bias":
            torch.ones(1),
            "external_draft_tokens_config":
            trtllm.ExternalDraftTokensConfig([1, 2, 3]),
            "prompt_tuning_config":
            trtllm.PromptTuningConfig(torch.ones(100, 64)),
            "lora_config":
            trtllm.LoraConfig(1),
            "logits_post_processor_name":
            "my_logits_pp",
            "client_id":
            1234,
        }

    # Test with ones
    embedding = torch.ones(576, 1024)
    kwargs = get_base_kwargs()
    kwargs["multimodal_embedding"] = embedding
    request = trtllm.Request(**kwargs)
    assert torch.equal(request.multimodal_embedding,
                       embedding), "Multimodal embedding with ones failed"

    # Test with random values
    random_embedding = torch.randn(576, 1024)
    kwargs["multimodal_embedding"] = random_embedding
    request = trtllm.Request(**kwargs)
    assert torch.equal(
        request.multimodal_embedding,
        random_embedding), "Multimodal embedding with random values failed"

    # Test with different shapes
    small_embedding = torch.ones(10, 20)
    kwargs["multimodal_embedding"] = small_embedding
    request = trtllm.Request(**kwargs)
    assert torch.equal(
        request.multimodal_embedding,
        small_embedding), "Multimodal embedding with different shape failed"


def test_multimodal_input():
    multimodal_hashes = [[1, 2, 3], [4, 5, 6]]
    multimodal_positions = [1, 2, 3]
    multimodal_lengths = [4, 5, 6]
    config = trtllm.MultimodalInput(multimodal_hashes, multimodal_positions,
                                    multimodal_lengths)
    assert config.multimodal_hashes == multimodal_hashes
    assert config.multimodal_positions == multimodal_positions
    assert config.multimodal_lengths == multimodal_lengths


def test_mrope_config():
    mrope_rotary_cos_sin = torch.ones(1, 4194304)
    mrope_position_deltas = torch.tensor([-50])
    config = trtllm.MropeConfig(mrope_rotary_cos_sin, mrope_position_deltas)
    assert (config.mrope_rotary_cos_sin == mrope_rotary_cos_sin).all()
    assert (config.mrope_position_deltas == mrope_position_deltas).all()


def test_lora_config():
    task_id = 1
    lora_config = trtllm.LoraConfig(task_id)
    assert lora_config.task_id == task_id
    assert lora_config.weights is None
    assert lora_config.config is None

    task_id = 2
    weights = torch.ones(1, 2)
    config = torch.ones(1, 2, dtype=torch.int32)
    lora_config = trtllm.LoraConfig(task_id, weights, config)
    assert lora_config.task_id == task_id
    assert (lora_config.weights == weights).all()
    assert (lora_config.config == config).all()


def test_wakeup(model_files, model_path):
    import threading

    def resp_thread(stop_signal: threading.Event, executor: trtllm.Executor):
        while not stop_signal.is_set():
            timeout = None
            responses = executor.await_responses(timeout=timeout)
            if stop_signal.is_set():
                return
            for response in responses:
                response.result.output_token_ids

    executor = trtllm.Executor(
        model_path, trtllm.ModelType.DECODER_ONLY,
        trtllm.ExecutorConfig(kv_cache_config=trtllm.KvCacheConfig(
            free_gpu_memory_fraction=0.5)))
    stop_signal = threading.Event()
    thread = threading.Thread(target=resp_thread, args=(stop_signal, executor))
    thread.start()
    request = trtllm.Request(input_token_ids=[1, 2, 3, 4],
                             max_tokens=5,
                             streaming=True)
    executor.enqueue_request(request)
    time.sleep(2)
    stop_signal.set()
    executor.shutdown()
    thread.join()
    assert not thread.is_alive()


def test_guided_decoding_params():
    guided_decoding_params = trtllm.GuidedDecodingParams(
        trtllm.GuidedDecodingParams.GuideType.JSON)
    assert guided_decoding_params.guide_type == trtllm.GuidedDecodingParams.GuideType.JSON

    class Answer(BaseModel):
        answer: int

    json_schema = json.dumps(Answer.model_json_schema())
    guided_decoding_params = trtllm.GuidedDecodingParams(
        trtllm.GuidedDecodingParams.GuideType.JSON_SCHEMA, guide=json_schema)
    assert guided_decoding_params.guide_type == trtllm.GuidedDecodingParams.GuideType.JSON_SCHEMA
    assert guided_decoding_params.guide == json_schema

    with pytest.raises(Exception):
        trtllm.GuidedDecodingParams(
            trtllm.GuidedDecodingParams.GuideType.JSON_SCHEMA)

    regex = r"\d+"
    guided_decoding_params = trtllm.GuidedDecodingParams(
        trtllm.GuidedDecodingParams.GuideType.REGEX, guide=regex)
    assert guided_decoding_params.guide_type == trtllm.GuidedDecodingParams.GuideType.REGEX
    assert guided_decoding_params.guide == regex

    ebnf_grammar = "root ::= [0-9]+"
    guided_decoding_params = trtllm.GuidedDecodingParams(
        trtllm.GuidedDecodingParams.GuideType.EBNF_GRAMMAR, guide=ebnf_grammar)
    assert guided_decoding_params.guide_type == trtllm.GuidedDecodingParams.GuideType.EBNF_GRAMMAR
    assert guided_decoding_params.guide == ebnf_grammar


def test_request():
    kwargs = {
        "input_token_ids": [1, 2, 3],
        "max_tokens": 1,
        "streaming": False,
        "sampling_config": trtllm.SamplingConfig(),
        "output_config": trtllm.OutputConfig(),
        "end_id": -1,
        "pad_id": -2,
        "bad_words": [[4, 5, 6]],
        "stop_words": [[7, 8, 9]],
        "embedding_bias": torch.ones(1),
        "external_draft_tokens_config":
        trtllm.ExternalDraftTokensConfig([1, 2, 3]),
        "prompt_tuning_config": trtllm.PromptTuningConfig(torch.ones(100, 64)),
        "multimodal_embedding": torch.ones(100, 64),
        "lora_config": trtllm.LoraConfig(1),
        "logits_post_processor_name": "my_logits_pp",
        "client_id": 1234,
    }
    request = trtllm.Request(**kwargs)
    for k, v in kwargs.items():
        if "config" not in k:
            attr_value = getattr(request, k)
            if isinstance(attr_value, torch.Tensor):
                assert (attr_value == v).all()
            else:
                assert attr_value == v
    assert isinstance(request.sampling_config, trtllm.SamplingConfig)
    assert isinstance(request.output_config, trtllm.OutputConfig)
    assert isinstance(request.external_draft_tokens_config,
                      trtllm.ExternalDraftTokensConfig)
    assert request.external_draft_tokens_config.tokens == [1, 2, 3]
    assert isinstance(request.prompt_tuning_config, trtllm.PromptTuningConfig)
    assert (request.prompt_tuning_config.embedding_table == torch.ones(
        100, 64)).all()
    assert isinstance(request.lora_config, trtllm.LoraConfig)


def test_spec_dec_fast_logits_info():
    fast_logits_info = trtllm.SpeculativeDecodingFastLogitsInfo()
    fast_logits_info.draft_request_id = 3
    fast_logits_info.draft_participant_id = 5
    assert fast_logits_info.draft_request_id == 3
    assert fast_logits_info.draft_participant_id == 5


def test_result():
    result = trtllm.Result()
    result.is_final = True
    result.output_token_ids = [[1, 2, 3]]
    result.cum_log_probs = [1.0, 2.0, 3.0]
    result.log_probs = [[1.0, 2.0, 3.0]]
    result.context_logits = torch.ones(3, 100)
    result.generation_logits = torch.ones(1, 3, 100)
    result.encoder_output = torch.ones(1, 1)
    result.finish_reasons = [trtllm.FinishReason.LENGTH]
    result.sequence_index = 1
    result.is_sequence_final = True
    result.decoding_iter = 1
    result.request_perf_metrics = trtllm.RequestPerfMetrics()
    result.request_perf_metrics.last_iter = 33
    result.additional_outputs = [
        trtllm.AdditionalOutput("topKLogits", torch.ones(1, 4, 100))
    ]
    assert result.is_final is True
    assert result.output_token_ids == [[1, 2, 3]]
    assert result.cum_log_probs == [1.0, 2.0, 3.0]
    assert result.log_probs == [[1.0, 2.0, 3.0]]
    assert (result.context_logits == torch.ones(3, 100)).all()
    assert (result.generation_logits == torch.ones(1, 3, 100)).all()
    assert (result.encoder_output == torch.ones(1, 1)).all()
    assert result.finish_reasons == [trtllm.FinishReason.LENGTH]
    assert result.sequence_index == 1
    assert result.is_sequence_final is True
    assert result.decoding_iter == 1
    assert result.request_perf_metrics is not None
    assert result.request_perf_metrics.last_iter == 33
    assert len(result.additional_outputs) == 1
    additional_output = result.additional_outputs[0]
    assert additional_output.name == "topKLogits"
    assert (additional_output.output == torch.ones(1, 4, 100)).all()


def test_result_pickle():
    result = trtllm.Result()
    result.is_final = True
    result.output_token_ids = [[1, 2, 3]]
    result.cum_log_probs = [1.0, 2.0, 3.0]
    result.log_probs = [[1.0, 2.0, 3.0]]
    result.context_logits = torch.ones(3, 100)
    result.generation_logits = torch.ones(1, 3, 100)
    result.encoder_output = torch.ones(1, 1)
    result.finish_reasons = [trtllm.FinishReason.LENGTH]
    result.sequence_index = 1
    result.is_sequence_final = True
    result.decoding_iter = 1
    result.context_phase_params = trtllm.ContextPhaseParams([1, 2], 123,
                                                            bytes([0, 1]),
                                                            [10, 20, 30])
    result.request_perf_metrics = trtllm.RequestPerfMetrics()
    result.request_perf_metrics.last_iter = 33
    result_str = pickle.dumps(result)
    result_copy = pickle.loads(result_str)
    assert result.is_final == result_copy.is_final
    assert result.output_token_ids == result_copy.output_token_ids
    assert result.cum_log_probs == result_copy.cum_log_probs
    assert result.log_probs == result_copy.log_probs
    assert (result.context_logits == result_copy.context_logits).all()
    assert (result.generation_logits == result_copy.generation_logits).all()
    assert (result.encoder_output == result_copy.encoder_output).all()
    assert result.finish_reasons == result_copy.finish_reasons
    assert result.sequence_index == result_copy.sequence_index
    assert result.is_sequence_final == result_copy.is_sequence_final
    assert result.decoding_iter == result_copy.decoding_iter
    assert result.context_phase_params.req_id == result_copy.context_phase_params.req_id
    assert result.context_phase_params.first_gen_tokens == result_copy.context_phase_params.first_gen_tokens
    assert result.context_phase_params.draft_tokens == result_copy.context_phase_params.draft_tokens
    assert result.context_phase_params.opaque_state == result_copy.context_phase_params.opaque_state
    assert result.request_perf_metrics.last_iter == result_copy.request_perf_metrics.last_iter


def test_response():
    request_id = 0
    error_msg = "error"
    response = trtllm.Response(request_id, error_msg)
    assert response.request_id == request_id
    assert response.has_error()
    assert response.error_msg == error_msg

    result = trtllm.Result()
    result.is_final = True
    result.output_token_ids = [[1, 2, 3]]
    request_id = 1
    response = trtllm.Response(request_id, result)
    assert response.request_id == request_id
    assert not response.has_error()
    assert response.result.is_final
    assert response.result.output_token_ids == [[1, 2, 3]]


def test_dynamic_batch_config():
    config = trtllm.DynamicBatchConfig(enable_batch_size_tuning=True,
                                       enable_max_num_tokens_tuning=True,
                                       dynamic_batch_moving_average_window=128)
    assert config.enable_batch_size_tuning == True
    assert config.enable_max_num_tokens_tuning == True
    assert config.dynamic_batch_moving_average_window == 128


def test_dynamic_batch_config_pickle():
    config = trtllm.DynamicBatchConfig(enable_batch_size_tuning=True,
                                       enable_max_num_tokens_tuning=True,
                                       dynamic_batch_moving_average_window=128)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config_copy.enable_batch_size_tuning == True
    assert config_copy.enable_max_num_tokens_tuning == True
    assert config_copy.dynamic_batch_moving_average_window == 128


def test_scheduler_config():
    capacity_scheduler_policy = trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    config = trtllm.SchedulerConfig()
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy is None

    capacity_scheduler_policy = trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION
    config = trtllm.SchedulerConfig(capacity_scheduler_policy)
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy is None

    capacity_scheduler_policy = trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    config = trtllm.SchedulerConfig(capacity_scheduler_policy)
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy is None

    capacity_scheduler_policy = trtllm.CapacitySchedulerPolicy.STATIC_BATCH
    config = trtllm.SchedulerConfig(capacity_scheduler_policy)
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy is None

    context_chunking_policy = trtllm.ContextChunkingPolicy.FIRST_COME_FIRST_SERVED
    config = trtllm.SchedulerConfig(capacity_scheduler_policy,
                                    context_chunking_policy)
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy == context_chunking_policy

    dynamic_batch_config = trtllm.DynamicBatchConfig(True, True, 128)
    config = trtllm.SchedulerConfig(capacity_scheduler_policy,
                                    context_chunking_policy,
                                    dynamic_batch_config)
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy == context_chunking_policy
    assert config.dynamic_batch_config.enable_batch_size_tuning == True
    assert config.dynamic_batch_config.enable_max_num_tokens_tuning == True
    assert config.dynamic_batch_config.dynamic_batch_moving_average_window == 128


def test_kv_cache_config():
    config = trtllm.KvCacheConfig()
    assert config.enable_block_reuse == True
    assert config.max_tokens is None
    assert config.max_attention_window is None
    assert config.sink_token_length is None
    assert config.free_gpu_memory_fraction is None
    assert config.cross_kv_cache_fraction is None
    assert config.host_cache_size is None
    assert config.onboard_blocks == True
    assert config.secondary_offload_min_priority is None
    assert config.event_buffer_max_size == 0
    assert config.enable_partial_reuse == True
    assert config.copy_on_partial_reuse == True
    assert config.use_uvm == False
    assert config.attention_dp_events_gather_period_ms == 5

    config.enable_block_reuse = False
    config.max_tokens = 1
    config.max_attention_window = [2]
    config.sink_token_length = 3
    config.free_gpu_memory_fraction = 0.5
    config.cross_kv_cache_fraction = 0.5
    config.host_cache_size = 4
    config.onboard_blocks = False
    config.secondary_offload_min_priority = 50
    config.event_buffer_max_size = 1024
    config.enable_partial_reuse = False
    config.copy_on_partial_reuse = False
    config.use_uvm = True
    config.attention_dp_events_gather_period_ms = 10
    assert config.enable_block_reuse == False
    assert config.max_tokens == 1
    assert config.max_attention_window == [2]
    assert config.sink_token_length == 3
    assert config.free_gpu_memory_fraction == 0.5
    assert config.cross_kv_cache_fraction == 0.5
    assert config.host_cache_size == 4
    assert config.onboard_blocks == False
    assert config.secondary_offload_min_priority == 50
    assert config.event_buffer_max_size == 1024
    assert config.enable_partial_reuse == False
    assert config.copy_on_partial_reuse == False
    assert config.use_uvm == True
    assert config.attention_dp_events_gather_period_ms == 10

    kwargs = {
        "enable_block_reuse": True,
        "max_tokens": 3,
        "max_attention_window": [10],
        "sink_token_length": 2,
        "free_gpu_memory_fraction": 0.5,
        "cross_kv_cache_fraction": 0.5,
        "host_cache_size": 1024,
        "onboard_blocks": False,
        "event_buffer_max_size": 2048,
        "enable_partial_reuse": True,
        "copy_on_partial_reuse": False,
        "use_uvm": True,
        "attention_dp_events_gather_period_ms": 10
    }
    config = trtllm.KvCacheConfig(**kwargs)
    for k, v in kwargs.items():
        assert getattr(config, k) == v

    config = trtllm.KvCacheConfig(**kwargs)
    max_attention_window, sink_token_length = config.max_attention_window, config.sink_token_length
    runtime_defaults = trtllm.RuntimeDefaults(
        max_attention_window=max_attention_window + [1],
        sink_token_length=sink_token_length + 1)

    config.fill_empty_fields_from_runtime_defaults(runtime_defaults)
    assert config.max_attention_window == max_attention_window, "runtime defaults shouldn't override existing values"
    assert config.sink_token_length == sink_token_length, "runtime defaults shouldn't override existing values"

    config = trtllm.KvCacheConfig(**{
        **kwargs, "max_attention_window": None,
        "sink_token_length": None
    })
    config.fill_empty_fields_from_runtime_defaults(runtime_defaults)
    assert config.max_attention_window == runtime_defaults.max_attention_window, "runtime defaults should apply to non existent values"
    assert config.sink_token_length == runtime_defaults.sink_token_length, "runtime defaults should apply to non existent values"

    config = trtllm.KvCacheConfig(**kwargs, runtime_defaults=runtime_defaults)
    setter_config = trtllm.KvCacheConfig(**kwargs)
    setter_config.fill_empty_fields_from_runtime_defaults(runtime_defaults)
    for k in kwargs.keys():
        assert getattr(config, k) == getattr(
            setter_config, k
        ), "passing runtime_defaults to the constructor or settings it manually should be equivalent"


def test_kv_cache_retention_config():

    TokenRangeRetentionConfig = trtllm.KvCacheRetentionConfig.TokenRangeRetentionConfig

    test_dir = "test_dir"
    config = trtllm.KvCacheRetentionConfig(
        [TokenRangeRetentionConfig(0, 2, 30, datetime.timedelta(seconds=30))],
        80, None, trtllm.KvCacheTransferMode.GDS, "test_dir")
    assert len(config.token_range_retention_configs) == 1
    assert config.token_range_retention_configs[0].token_start == 0
    assert config.token_range_retention_configs[0].token_end == 2
    assert config.token_range_retention_configs[0].priority == 30
    assert config.token_range_retention_configs[
        0].duration_ms == datetime.timedelta(seconds=30)
    assert config.decode_retention_priority == 80
    assert config.decode_duration_ms is None
    assert config.transfer_mode == trtllm.KvCacheTransferMode.GDS
    assert config.directory == test_dir

    config = trtllm.KvCacheRetentionConfig(
        [
            TokenRangeRetentionConfig(0, 64, 80),
            TokenRangeRetentionConfig(64, 100, 10)
        ], 10, datetime.timedelta(milliseconds=30000),
        trtllm.KvCacheTransferMode.POSIX_DEBUG_FALLBACK, test_dir)

    assert len(config.token_range_retention_configs) == 2
    assert config.token_range_retention_configs[0].token_start == 0
    assert config.token_range_retention_configs[0].token_end == 64
    assert config.token_range_retention_configs[0].priority == 80
    assert config.token_range_retention_configs[0].duration_ms is None

    assert config.token_range_retention_configs[1].token_start == 64
    assert config.token_range_retention_configs[1].token_end == 100
    assert config.token_range_retention_configs[1].priority == 10
    assert config.token_range_retention_configs[1].duration_ms is None

    assert config.decode_retention_priority == 10
    assert config.decode_duration_ms == datetime.timedelta(seconds=30)
    assert config.transfer_mode == trtllm.KvCacheTransferMode.POSIX_DEBUG_FALLBACK
    assert config.directory == test_dir

    with pytest.raises(Exception):
        # Invalid token ranges
        trtllm.KvCacheRetentionConfig([
            TokenRangeRetentionConfig(0, 64, 10),
            TokenRangeRetentionConfig(32, 128, 50)
        ], 50)


def test_speculative_decoding_config():
    config = trtllm.SpeculativeDecodingConfig(True)
    assert config.fast_logits == True

    kwargs = {
        "fast_logits": True,
    }

    config = trtllm.SpeculativeDecodingConfig(**kwargs)
    for k, v in kwargs.items():
        assert getattr(config, k) == v


def test_speculative_decoding_config_pickle():
    config = trtllm.SpeculativeDecodingConfig(True)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config_copy.fast_logits == True


def test_lookahead_decoding_config():
    config = trtllm.LookaheadDecodingConfig(3, 5, 7)
    assert config.max_window_size == 3
    assert config.max_ngram_size == 5
    assert config.max_verification_set_size == 7

    config = trtllm.LookaheadDecodingConfig(5, 10, 3)
    assert config.max_window_size == 5
    assert config.max_ngram_size == 10
    assert config.max_verification_set_size == 3

    kwargs = {
        "max_window_size": 5,
        "max_ngram_size": 3,
        "max_verification_set_size": 7,
    }

    config = trtllm.LookaheadDecodingConfig(**kwargs)
    for k, v in kwargs.items():
        assert getattr(config, k) == v


def test_lookahead_decoding_config_pickle():
    config = trtllm.LookaheadDecodingConfig(3, 5, 7)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config_copy.max_window_size == 3
    assert config_copy.max_ngram_size == 5
    assert config_copy.max_verification_set_size == 7


def test_eagle_config():
    config = trtllm.EagleConfig([[0, 0], [0, 1]], False, 0.5)
    assert config.eagle_choices == [[0, 0], [0, 1]]
    assert config.greedy_sampling == False
    assert config.posterior_threshold == 0.5
    assert config.use_dynamic_tree == False
    assert config.dynamic_tree_max_topK is None

    config = trtllm.EagleConfig([[0, 0], [0, 1, 0]], True)
    assert config.eagle_choices == [[0, 0], [0, 1, 0]]
    assert config.greedy_sampling == True
    assert config.posterior_threshold is None
    assert config.use_dynamic_tree == False
    assert config.dynamic_tree_max_topK is None

    config = trtllm.EagleConfig(None, True, 0.5)
    assert config.eagle_choices is None
    assert config.greedy_sampling == True
    assert config.posterior_threshold == 0.5
    assert config.use_dynamic_tree == False
    assert config.dynamic_tree_max_topK is None

    config = trtllm.EagleConfig(None, False, 0.5, True, 3)
    assert config.eagle_choices is None
    assert config.greedy_sampling == False
    assert config.posterior_threshold == 0.5
    assert config.use_dynamic_tree == True
    assert config.dynamic_tree_max_topK == 3

    kwargs1 = {
        "eagle_choices": [[0, 0], [0, 1], [0, 2]],
        "greedy_sampling": True,
        "posterior_threshold": 0.5
    }

    config = trtllm.EagleConfig(**kwargs1)
    for k, v in kwargs1.items():
        assert getattr(config, k) == v

    kwargs2 = {
        "eagle_choices": None,
        "greedy_sampling": True,
        "posterior_threshold": 0.5,
        "use_dynamic_tree": True,
        "dynamic_tree_max_topK": 3,
    }

    config = trtllm.EagleConfig(**kwargs2)
    for k, v in kwargs2.items():
        assert getattr(config, k) == v


def test_eagle_config_pickle():
    config = trtllm.EagleConfig([[0, 0], [0, 1]], False, 0.5)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config.dynamic_tree_max_topK == config_copy.dynamic_tree_max_topK
    assert config.eagle_choices == config_copy.eagle_choices
    assert config.posterior_threshold == config_copy.posterior_threshold
    assert config.use_dynamic_tree == config_copy.use_dynamic_tree
    assert config.greedy_sampling == config_copy.greedy_sampling

    config = trtllm.EagleConfig(None, False, 0.5, True, 3)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config.eagle_choices == config_copy.eagle_choices
    assert config.greedy_sampling == config_copy.greedy_sampling
    assert config.posterior_threshold == config_copy.posterior_threshold
    assert config.use_dynamic_tree == config_copy.use_dynamic_tree
    assert config.dynamic_tree_max_topK == config_copy.dynamic_tree_max_topK


def test_decoding_mode():
    mode = trtllm.DecodingMode.Auto()
    assert mode.isAuto()

    mode = trtllm.DecodingMode.TopK()
    assert mode.isTopK()

    mode = trtllm.DecodingMode.TopP()
    assert mode.isTopP()

    mode = trtllm.DecodingMode.TopKTopP()
    assert mode.isTopKandTopP()

    mode = trtllm.DecodingMode.BeamSearch()
    assert mode.isBeamSearch()

    mode = trtllm.DecodingMode.Medusa()
    assert mode.isMedusa()

    mode = trtllm.DecodingMode.Lookahead()
    assert mode.isLookahead()

    mode = trtllm.DecodingMode.ExplicitDraftTokens()
    assert mode.isExplicitDraftTokens()

    mode = trtllm.DecodingMode.Eagle()
    assert mode.isEagle()


def test_speculative_decoding_config():
    config = trtllm.DecodingConfig()
    assert config.decoding_mode is None
    assert config.lookahead_decoding_config is None
    assert config.medusa_choices is None
    assert config.eagle_config is None

    config = trtllm.DecodingConfig()
    config.decoding_mode = trtllm.DecodingMode.TopKTopP()
    assert config.decoding_mode.isTopKandTopP()
    assert config.lookahead_decoding_config is None
    assert config.medusa_choices is None
    assert config.eagle_config is None

    config = trtllm.DecodingConfig()
    la_decoding_config = trtllm.LookaheadDecodingConfig(3, 5, 7)
    config.lookahead_decoding_config = la_decoding_config

    assert config.decoding_mode.isLookahead()
    assert config.lookahead_decoding_config.max_ngram_size == la_decoding_config.max_ngram_size
    assert config.lookahead_decoding_config.max_window_size == la_decoding_config.max_window_size
    assert config.lookahead_decoding_config.max_verification_set_size == la_decoding_config.max_verification_set_size
    assert config.medusa_choices is None
    assert config.eagle_config is None

    config = trtllm.DecodingConfig()
    config.medusa_choices = [[0, 0], [0, 1]]

    assert config.decoding_mode.isMedusa()
    assert config.lookahead_decoding_config is None
    assert config.medusa_choices == [[0, 0], [0, 1]]
    assert config.eagle_config is None

    config = trtllm.DecodingConfig()
    config.eagle_config = trtllm.EagleConfig([[0, 0], [0, 1]])

    assert config.decoding_mode.isEagle()
    assert config.lookahead_decoding_config is None
    assert config.medusa_choices is None
    assert config.eagle_config is not None
    assert config.eagle_config.eagle_choices == [[0, 0], [0, 1]]


def test_logits_post_processor_config():
    config = trtllm.LogitsPostProcessorConfig()
    assert config.processor_map is None
    assert config.processor_batched is None
    assert config.replicate == True

    kwargs = {
        "processor_map": {
            "test_pp": None
        },
        "processor_batched": None,
        "replicate": False
    }
    config = trtllm.LogitsPostProcessorConfig(**kwargs)
    for k, v in kwargs.items():
        assert getattr(config, k) == v


def test_guided_decoding_config():
    encoded_vocab = ["eos", "a", "b", "c", "d"]
    tokenizer_str = None
    stop_token_ids = [0]
    guided_decoding_config = trtllm.GuidedDecodingConfig(
        backend=trtllm.GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
        encoded_vocab=encoded_vocab,
        tokenizer_str=tokenizer_str,
        stop_token_ids=stop_token_ids)
    assert guided_decoding_config.backend == trtllm.GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR
    assert guided_decoding_config.encoded_vocab == encoded_vocab
    assert guided_decoding_config.tokenizer_str == tokenizer_str
    assert guided_decoding_config.stop_token_ids == stop_token_ids


def test_executor_config():
    config = trtllm.ExecutorConfig()
    assert config.max_beam_width == 1
    assert config.max_batch_size is None
    assert config.max_num_tokens is None
    assert isinstance(config.scheduler_config, trtllm.SchedulerConfig)
    assert isinstance(config.kv_cache_config, trtllm.KvCacheConfig)
    assert config.enable_chunked_context == False
    assert config.normalize_log_probs == True
    assert config.iter_stats_max_iterations == 1000
    assert config.batching_type == trtllm.BatchingType.INFLIGHT
    assert config.parallel_config is None
    assert isinstance(config.peft_cache_config, trtllm.PeftCacheConfig)
    assert config.logits_post_processor_config is None
    assert config.decoding_config is None
    assert config.debug_config is None
    assert config.recv_poll_period_ms == 0
    assert config.max_seq_idle_microseconds == 180000000
    assert config.spec_dec_config is None
    assert config.guided_decoding_config is None
    assert config.additional_model_outputs is None
    assert config.gather_generation_logits is False
    assert config.use_gpu_direct_storage is False
    assert config.mm_embedding_offloading is False
    assert config.enable_trt_overlap is False

    kwargs = {
        "max_beam_width":
        2,
        "max_batch_size":
        8,
        "max_num_tokens":
        128,
        "scheduler_config":
        trtllm.SchedulerConfig(trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION),
        "kv_cache_config":
        trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5),
        "enable_chunked_context":
        True,
        "normalize_log_probs":
        False,
        "iter_stats_max_iterations":
        100,
        "batching_type":
        trtllm.BatchingType.STATIC,
        "parallel_config":
        trtllm.ParallelConfig(),
        "peft_cache_config":
        trtllm.PeftCacheConfig(10),
        "logits_post_processor_config":
        trtllm.LogitsPostProcessorConfig(),
        "decoding_config":
        trtllm.DecodingConfig(trtllm.DecodingMode.TopKTopP()),
        "extended_runtime_perf_knob_config":
        trtllm.ExtendedRuntimePerfKnobConfig(multi_block_mode=True),
        "debug_config":
        trtllm.DebugConfig(debug_input_tensors=True,
                           debug_output_tensors=True,
                           debug_tensor_names=["test"]),
        "recv_poll_period_ms":
        50,
        "max_seq_idle_microseconds":
        240 * 1000 * 1000,
        "spec_dec_config":
        trtllm.SpeculativeDecodingConfig(fast_logits=True),
        "guided_decoding_config":
        trtllm.GuidedDecodingConfig(
            trtllm.GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
            encoded_vocab=["eos", "a", "b", "c", "d"]),
        "additional_model_outputs":
        [trtllm.AdditionalModelOutput("topKLogits")],
        "gather_generation_logits":
        True,
        "use_gpu_direct_storage":
        True,
        "mm_embedding_offloading":
        True,
        "enable_trt_overlap":
        True,
    }
    config = trtllm.ExecutorConfig(**kwargs)
    for k, v in kwargs.items():
        if "config" not in k and k != "additional_model_outputs":
            assert getattr(config, k) == v
    assert isinstance(config.scheduler_config, trtllm.SchedulerConfig)
    assert config.scheduler_config.capacity_scheduler_policy == trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION
    assert isinstance(config.kv_cache_config, trtllm.KvCacheConfig)
    assert isinstance(config.parallel_config, trtllm.ParallelConfig)
    assert isinstance(config.peft_cache_config, trtllm.PeftCacheConfig)
    assert config.extended_runtime_perf_knob_config.multi_block_mode is True
    assert isinstance(config.debug_config, trtllm.DebugConfig)
    assert isinstance(config.logits_post_processor_config,
                      trtllm.LogitsPostProcessorConfig)
    assert isinstance(config.spec_dec_config, trtllm.SpeculativeDecodingConfig)
    assert isinstance(config.guided_decoding_config,
                      trtllm.GuidedDecodingConfig)
    assert isinstance(config.additional_model_outputs, list)
    assert len(config.additional_model_outputs) == 1
    assert config.additional_model_outputs[0].name == "topKLogits"
    assert config.additional_model_outputs[0].gather_context is False
    assert config.gather_generation_logits is True
    assert config.use_gpu_direct_storage is True
    assert config.mm_embedding_offloading is True
    assert config.enable_trt_overlap is True


def test_parallel_config():
    comm_type = trtllm.CommunicationType.MPI
    comm_mode = trtllm.CommunicationMode.LEADER
    device_ids = [0, 1, 2, 3]
    participant_ids = [4, 5, 6, 7]
    num_nodes = 2
    parallel_config = trtllm.ParallelConfig(comm_type,
                                            comm_mode,
                                            device_ids,
                                            participant_ids,
                                            num_nodes=num_nodes)
    assert parallel_config.communication_type == comm_type
    assert parallel_config.communication_mode == comm_mode
    assert parallel_config.device_ids == device_ids
    assert parallel_config.participant_ids == participant_ids
    assert parallel_config.num_nodes == num_nodes

    comm_mode = trtllm.CommunicationMode.ORCHESTRATOR
    #Dummy path to worker executable
    worker_path = _os.path.abspath(__file__)
    orchestrator_config = trtllm.OrchestratorConfig(True, str(worker_path),
                                                    None, True)
    parallel_config = trtllm.ParallelConfig(comm_type, comm_mode, device_ids,
                                            participant_ids,
                                            orchestrator_config)
    assert parallel_config.communication_mode == comm_mode
    assert parallel_config.orchestrator_config.is_orchestrator == True
    assert parallel_config.orchestrator_config.worker_executable_path == str(
        worker_path)
    assert parallel_config.orchestrator_config.spawn_processes == True


def test_parallel_config_pickle():
    comm_type = trtllm.CommunicationType.MPI
    comm_mode = trtllm.CommunicationMode.LEADER
    device_ids = [0, 1, 2, 3]
    participant_ids = [4, 5, 6, 7]
    num_nodes = 2
    parallel_config = trtllm.ParallelConfig(comm_type,
                                            comm_mode,
                                            device_ids,
                                            participant_ids,
                                            num_nodes=num_nodes)
    parallel_config_copy = pickle.loads(pickle.dumps(parallel_config))
    assert parallel_config_copy.communication_type == comm_type
    assert parallel_config_copy.communication_mode == comm_mode
    assert parallel_config_copy.device_ids == device_ids
    assert parallel_config_copy.participant_ids == participant_ids
    assert parallel_config_copy.num_nodes == num_nodes


def test_peft_cache_config():
    num_host_module_layer = 1
    num_device_module_layer = 2
    optimal_adapter_size = 3
    max_adapter_size = 4
    num_put_workers = 5
    num_ensure_workers = 6
    num_copy_streams = 7
    max_pages_per_block_host = 8
    max_pages_per_block_device = 9
    device_cache_percent = 0.9
    host_cache_size = 1024
    lora_prefetch_dir = "/tmp/lora_prefetch"
    peft_cache_config = trtllm.PeftCacheConfig(
        num_host_module_layer, num_device_module_layer, optimal_adapter_size,
        max_adapter_size, num_put_workers, num_ensure_workers, num_copy_streams,
        max_pages_per_block_host, max_pages_per_block_device,
        device_cache_percent, host_cache_size, lora_prefetch_dir)

    assert peft_cache_config.num_host_module_layer == num_host_module_layer
    assert peft_cache_config.num_device_module_layer == num_device_module_layer
    assert peft_cache_config.optimal_adapter_size == optimal_adapter_size
    assert peft_cache_config.max_adapter_size == max_adapter_size
    assert peft_cache_config.num_put_workers == num_put_workers
    assert peft_cache_config.num_ensure_workers == num_ensure_workers
    assert peft_cache_config.num_copy_streams == num_copy_streams
    assert peft_cache_config.max_pages_per_block_host == max_pages_per_block_host
    assert peft_cache_config.max_pages_per_block_device == max_pages_per_block_device
    assert np.isclose(peft_cache_config.device_cache_percent,
                      device_cache_percent)
    assert peft_cache_config.host_cache_size == host_cache_size
    assert peft_cache_config.lora_prefetch_dir == lora_prefetch_dir


def test_logits_post_processor(model_files, model_path):

    # Define the logits post-processor callback
    def logits_post_processor(req_id: int, logits: torch.Tensor,
                              ids: tp.List[tp.List[int]], stream_ptr: int,
                              client_id: tp.Optional[int]):
        assert client_id == 123
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits[:] = float("-inf")
            logits[..., 42] = 0

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor_config.logits_post_processor_config = trtllm.LogitsPostProcessorConfig(
        {"my_logits_pp": logits_post_processor})
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             streaming=False,
                             client_id=123)
    request.logits_post_processor_name = "my_logits_pp"

    # Enqueue the request
    request_id = executor.enqueue_request(request)

    # Get the new tokens
    tokens = []
    done = False
    i = 0
    max_wait_ms = 10000
    while not done and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(request_id, wait_time)
        for response in responses:
            assert not response.has_error(
            ), f"Request id {request_id} failed with err {response.error_msg}"
            result = response.result
            done = result.is_final
            new_tokens = result.output_token_ids[beam_width - 1]
            tokens.extend(new_tokens)
        i += 1
    assert i < max_wait_ms
    assert len(tokens) == get_expected_num_tokens(len(input_tokens), max_tokens,
                                                  False, False), f"{request_id}"

    # check that all output tokens are 42
    print(tokens)
    assert tokens[-max_tokens:] == [42] * max_tokens


def test_logits_post_processor_batched(model_files, model_path):

    # Define the logits post-processor callback
    def logits_post_processor_batched(
            req_id_batch: tp.List[int], logits_batch: tp.List[torch.Tensor],
            ids_batch: tp.List[tp.List[tp.List[int]]], stream_ptr: int,
            client_id_batch: tp.List[tp.Optional[int]]):
        for client_id in client_id_batch:
            assert client_id == 123
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            for logits in logits_batch:
                logits[:] = float("-inf")
                logits[..., 42] = 0

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor_config.logits_post_processor_config = trtllm.LogitsPostProcessorConfig(
        None, logits_post_processor_batched)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             streaming=False,
                             client_id=123)
    request.logits_post_processor_name = request.BATCHED_POST_PROCESSOR_NAME

    batch_size = 4
    # Enqueue the requests
    request_ids = []
    for _ in range(batch_size):
        request_id = executor.enqueue_request(request)
        request_ids.append(request_id)

    # Get the new tokens
    tokens = {req_id: [] for req_id in request_ids}
    num_finished = 0
    i = 0
    max_wait_ms = 10000
    while num_finished < len(request_ids) and i < max_wait_ms:
        responses = executor.await_responses(datetime.timedelta(milliseconds=1))
        for response in responses:
            req_id = response.request_id
            assert not response.has_error(
            ), f"Request id {req_id} failed with err {response.error_msg}"
            result = response.result
            num_finished += 1 if result.is_final else 0
            new_tokens = result.output_token_ids[beam_width - 1]
            tokens[req_id].extend(new_tokens)
    assert i < max_wait_ms

    expected_num_tokens = get_expected_num_tokens(len(input_tokens), max_tokens,
                                                  False, False)
    for req_id in request_ids:
        assert len(tokens[req_id]) == expected_num_tokens, f"{req_id}"


@pytest.mark.skip("https://nvbugs/5082576")
def test_kv_event_stream(model_path):

    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(True,
                                             4 * 64,
                                             event_buffer_max_size=1024,
                                             host_cache_size=3000000,
                                             free_gpu_memory_fraction=0.5))

    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    cache_manager = executor.get_kv_cache_event_manager()

    events = cache_manager.get_latest_events()

    assert len(events) == 1
    assert isinstance(events[0], trtllm.kv_cache.KVCacheEvent)
    assert events[0].event_id == 0
    assert isinstance(events[0].data, trtllm.kv_cache.KVCacheCreatedData)

    for req in range(2):
        input_tokens = list(range(req, req + 127))
        request = trtllm.Request(input_tokens,
                                 max_tokens=5,
                                 streaming=False,
                                 sampling_config=trtllm.SamplingConfig())

        id = executor.enqueue_request(request)

        responses = executor.await_responses(id)

        for response in responses:
            assert not response.has_error()
            if response.result.is_final:
                time.sleep(0.1)
                events = cache_manager.get_latest_events(
                    datetime.timedelta(milliseconds=100))

                if req == 0:
                    assert events[0].event_id == 1
                    assert isinstance(events[0].data,
                                      trtllm.kv_cache.KVCacheStoredData)
                    assert events[0].data.parent_hash is None
                    assert len(events[0].data.blocks) == 1

                    assert events[1].data.parent_hash == events[0].data.blocks[
                        0].block_hash
                    assert len(events[1].data.blocks) == 2
                else:
                    # Swap a block to secondary
                    assert isinstance(events[0].data,
                                      trtllm.kv_cache.KVCacheUpdatedData)
                    assert events[0].data.cache_level.old_value == 0
                    assert events[0].data.cache_level.new_value == 1
                    # Store the filled context block
                    assert isinstance(events[1].data,
                                      trtllm.kv_cache.KVCacheStoredData)
                    assert len(events[1].data.blocks) == 1
                    assert events[1].data.parent_hash is None
                    # Swap another block to secondary
                    assert isinstance(events[2].data,
                                      trtllm.kv_cache.KVCacheUpdatedData)
                    assert events[2].data.cache_level.old_value == 0
                    assert events[2].data.cache_level.new_value == 1
                    assert isinstance(events[2].data.cache_level,
                                      trtllm.kv_cache.KVCacheEventDiffInt)
                    # Remove the first block in secondary
                    assert isinstance(events[3].data,
                                      trtllm.kv_cache.KVCacheRemovedData)
                    assert len(events[3].data.block_hashes) == 1
                    assert events[3].data.block_hashes[0] == events[
                        0].data.block_hash
                    # Store the second context block and the decode block
                    assert isinstance(events[4].data,
                                      trtllm.kv_cache.KVCacheStoredData)
                    assert len(events[4].data.blocks) == 2
                    assert events[4].data.parent_hash == events[1].data.blocks[
                        0].block_hash


@pytest.mark.parametrize("streaming", [False, True])
def test_request_perf_metrics(streaming: bool, model_path):

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create request
    max_tokens = 5
    input_tokens = [1, 2, 3, 4]
    output_config = trtllm.OutputConfig(return_perf_metrics=True)
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             streaming=streaming,
                             output_config=output_config)

    # Enqueue the request
    request_id = executor.enqueue_request(request)

    def check_perf_metrics(perf_metrics, done, response_id):
        assert perf_metrics is not None

        timing_metrics = perf_metrics.timing_metrics
        assert timing_metrics.arrival_time < timing_metrics.first_scheduled_time
        assert timing_metrics.first_scheduled_time < timing_metrics.first_token_time
        if done:
            assert timing_metrics.first_token_time < timing_metrics.last_token_time
        else:
            assert timing_metrics.last_token_time == datetime.timedelta(0)

        kv_cache_metrics = perf_metrics.kv_cache_metrics
        assert kv_cache_metrics.num_total_allocated_blocks == 1
        assert kv_cache_metrics.num_new_allocated_blocks == 1
        assert kv_cache_metrics.num_reused_blocks == 0
        assert kv_cache_metrics.num_missed_blocks == 1
        assert kv_cache_metrics.kv_cache_hit_rate == 0

        assert perf_metrics.first_iter == 0
        if done:
            assert perf_metrics.iter == (max_tokens - 1)
            assert perf_metrics.last_iter == max_tokens - 1
        else:
            assert perf_metrics.iter == response_id
            assert perf_metrics.last_iter is None

    # Get the new tokens
    tokens = []
    done = False
    i = 0
    max_wait_ms = 10000
    response_id = 0
    while not done and i < max_wait_ms:
        wait_time = datetime.timedelta(milliseconds=1)
        responses = executor.await_responses(request_id, wait_time)
        for response in responses:
            assert not response.has_error(
            ), f"Request id {request_id} failed with err {response.error_msg}"
            result = response.result
            done = result.is_final
            check_perf_metrics(result.request_perf_metrics, done, response_id)
            new_tokens = result.output_token_ids[beam_width - 1]
            tokens.extend(new_tokens)
            response_id += 1
        i += 1
    assert i < max_wait_ms
    assert len(tokens) == get_expected_num_tokens(
        len(input_tokens),
        max_tokens,
        streaming=streaming,
        exclude_input_from_output=False), f"{request_id}"


def test_request_perf_metrics_kv_cache(model_path):

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create request: model uses 32 tokens per block, so it will fill a full block
    max_tokens = 32
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens, max_tokens=max_tokens)

    # Enqueue the request
    request_id = executor.enqueue_request(request)
    # Store two blocks with a total of 35 reusable tokens (4 input + 32 output, but last token is not stored)

    # Get the response
    responses = executor.await_responses(request_id)
    assert not responses[0].has_error()
    result = responses[0].result
    assert result.is_final

    # Prepare second request using the first output
    input_tokens = result.output_token_ids[beam_width - 1] + [1, 2, 3, 4]
    output_config = trtllm.OutputConfig(return_perf_metrics=True)
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             output_config=output_config)

    # Enqueue the request
    # New query has 36 tokens that match input and output of first request plus 4 additional tokens.
    # First block reused completely (32 tokens), from second block we'll partially reuse 3 tokens.
    request_id = executor.enqueue_request(request)

    # Get the response
    responses = executor.await_responses(request_id)
    assert not responses[0].has_error()
    result = responses[0].result
    assert result.is_final

    # Check KV cache metric: Two blocks will be reused, although second block is partially reused.
    # Cache hit rate is 100% since granularity is blocks, not tokens.
    kv_cache_metrics = result.request_perf_metrics.kv_cache_metrics
    assert kv_cache_metrics.num_total_allocated_blocks == 0
    assert kv_cache_metrics.num_new_allocated_blocks == 0
    assert kv_cache_metrics.num_reused_blocks == 2
    assert kv_cache_metrics.num_missed_blocks == 0
    assert kv_cache_metrics.kv_cache_hit_rate == 1.0


# Skip test for pre-Hopper: https://nvbugs/5404000
@skip_pre_hopper
@pytest.mark.parametrize("exclude_input_from_output", [False, True])
def test_request_perf_metrics_draft(model_path_draft_tokens_external,
                                    exclude_input_from_output: bool):

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5))
    executor = trtllm.Executor(model_path_draft_tokens_external,
                               trtllm.ModelType.DECODER_ONLY, executor_config)

    # Create request
    max_tokens = 5
    input_tokens = [1, 2, 3, 4]

    # Only first two tokens will be accepted -> 50% acceptance rate
    draft_config = trtllm.ExternalDraftTokensConfig([2, 4, 9, 10])
    output_config = trtllm.OutputConfig(
        exclude_input_from_output=exclude_input_from_output,
        return_perf_metrics=True)
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             output_config=output_config,
                             external_draft_tokens_config=draft_config)

    # Enqueue the request
    request_id = executor.enqueue_request(request)

    # Get the response
    responses = executor.await_responses(request_id)
    assert not responses[0].has_error()
    result = responses[0].result
    assert result.is_final

    # check the new tokens
    new_tokens = result.output_token_ids[beam_width - 1]
    if exclude_input_from_output:
        assert new_tokens == [2, 4, 2]
    else:
        assert new_tokens == [1, 2, 3, 4, 2, 4, 2]

    # Check the perf metrics
    perf_metrics = result.request_perf_metrics
    assert perf_metrics is not None

    timing_metrics = perf_metrics.timing_metrics
    assert timing_metrics.arrival_time < timing_metrics.first_scheduled_time
    assert timing_metrics.first_scheduled_time < timing_metrics.first_token_time
    assert timing_metrics.first_token_time <= timing_metrics.last_token_time

    assert perf_metrics.first_iter == 0
    assert perf_metrics.iter == 0
    assert perf_metrics.last_iter == 0

    spec_dec_metrics = perf_metrics.speculative_decoding
    assert spec_dec_metrics.acceptance_rate == 0.5
    assert spec_dec_metrics.total_accepted_draft_tokens == 2
    assert spec_dec_metrics.total_draft_tokens == 4


def test_kv_event_stream_timeout(model_path):

    beam_width = 1
    executor_config = trtllm.ExecutorConfig(
        beam_width,
        kv_cache_config=trtllm.KvCacheConfig(True,
                                             4 * 64,
                                             event_buffer_max_size=1024,
                                             free_gpu_memory_fraction=0.5))

    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    cache_manager = executor.get_kv_cache_event_manager()

    events = cache_manager.get_latest_events()
    assert len(events) == 1

    start = datetime.datetime.now()
    events = cache_manager.get_latest_events(1000)
    end = datetime.datetime.now()
    # Make sure that it actually waited
    assert abs(end - start) > datetime.timedelta(milliseconds=900)
    assert len(events) == 0


def test_request_perf_metrics_pickle():
    metrics = trtllm.RequestPerfMetrics()
    random_delta = datetime.timedelta(seconds=42, milliseconds=123)

    metrics.timing_metrics.arrival_time = random_delta
    metrics.timing_metrics.first_scheduled_time = 2 * random_delta
    metrics.timing_metrics.first_token_time = 3 * random_delta
    metrics.timing_metrics.last_token_time = 4 * random_delta
    metrics.timing_metrics.kv_cache_transfer_start = 5 * random_delta
    metrics.timing_metrics.kv_cache_transfer_end = 6 * random_delta
    metrics.timing_metrics.kv_cache_size = 1024
    metrics.kv_cache_metrics.num_total_allocated_blocks = 1
    metrics.kv_cache_metrics.num_new_allocated_blocks = 1
    metrics.kv_cache_metrics.num_reused_blocks = 0
    metrics.kv_cache_metrics.num_missed_blocks = 1
    metrics.kv_cache_metrics.kv_cache_hit_rate = 0
    metrics.speculative_decoding.acceptance_rate = 0.5
    metrics.speculative_decoding.total_accepted_draft_tokens = 2
    metrics.speculative_decoding.total_draft_tokens = 4
    metrics.first_iter = 0
    metrics.iter = 40
    metrics.last_iter = 50
    metrics_str = pickle.dumps(metrics)
    metrics_copy = pickle.loads(metrics_str)
    assert metrics.timing_metrics.arrival_time == metrics_copy.timing_metrics.arrival_time
    assert metrics.timing_metrics.first_scheduled_time == metrics_copy.timing_metrics.first_scheduled_time
    assert metrics.timing_metrics.first_token_time == metrics_copy.timing_metrics.first_token_time
    assert metrics.timing_metrics.last_token_time == metrics_copy.timing_metrics.last_token_time
    assert metrics.timing_metrics.kv_cache_transfer_start == metrics_copy.timing_metrics.kv_cache_transfer_start
    assert metrics.timing_metrics.kv_cache_transfer_end == metrics_copy.timing_metrics.kv_cache_transfer_end
    assert metrics.timing_metrics.kv_cache_size == metrics_copy.timing_metrics.kv_cache_size
    assert metrics.kv_cache_metrics.num_total_allocated_blocks == metrics_copy.kv_cache_metrics.num_total_allocated_blocks
    assert metrics.kv_cache_metrics.num_new_allocated_blocks == metrics_copy.kv_cache_metrics.num_new_allocated_blocks
    assert metrics.kv_cache_metrics.num_reused_blocks == metrics_copy.kv_cache_metrics.num_reused_blocks
    assert metrics.kv_cache_metrics.num_missed_blocks == metrics_copy.kv_cache_metrics.num_missed_blocks
    assert metrics.kv_cache_metrics.kv_cache_hit_rate == metrics_copy.kv_cache_metrics.kv_cache_hit_rate
    assert metrics.speculative_decoding.acceptance_rate == metrics_copy.speculative_decoding.acceptance_rate
    assert metrics.speculative_decoding.total_accepted_draft_tokens == metrics_copy.speculative_decoding.total_accepted_draft_tokens
    assert metrics.speculative_decoding.total_draft_tokens == metrics_copy.speculative_decoding.total_draft_tokens
    assert metrics.first_iter == metrics_copy.first_iter
    assert metrics.iter == metrics_copy.iter
    assert metrics.last_iter == metrics_copy.last_iter


def test_iteration_stats():
    stats = trtllm.IterationStats()
    stats.timestamp = "01:23:56"
    stats.iter = 1
    stats.iter_latency_ms = 100
    stats.num_active_requests = 2
    stats.num_queued_requests = 10
    stats.max_num_active_requests = 3
    stats.gpu_mem_usage = 1024
    stats.cpu_mem_usage = 2048
    stats.pinned_mem_usage = 4096
    stats_json = json.loads(stats.to_json_str())
    assert stats_json["timestamp"] == stats.timestamp
    assert stats_json["iter"] == stats.iter
    assert stats_json["iterLatencyMS"] == stats.iter_latency_ms
    assert stats_json[
        "newActiveRequestsQueueLatencyMS"] == stats.new_active_requests_queue_latency_ms
    assert stats_json["numActiveRequests"] == stats.num_active_requests
    assert stats_json["numQueuedRequests"] == stats.num_queued_requests
    assert stats_json["numCompletedRequests"] == stats.num_completed_requests
    assert stats_json["maxNumActiveRequests"] == stats.max_num_active_requests
    assert stats_json["gpuMemUsage"] == stats.gpu_mem_usage
    assert stats_json["cpuMemUsage"] == stats.cpu_mem_usage
    assert stats_json["pinnedMemUsage"] == stats.pinned_mem_usage
    assert stats_json["kvCacheStats"] is None
    assert stats_json["staticBatchingStats"] is None
    assert stats_json["inflightBatchingStats"] is None


def test_request_stats():
    stats = trtllm.RequestStats()
    stats.id = 1
    stats.stage = trtllm.RequestStage.CONTEXT_IN_PROGRESS
    stats.context_prefill_position = 2
    stats.num_generated_tokens = 3
    stats.avg_num_decoded_tokens_per_iter = 2.5
    stats.scheduled = True
    stats.paused = False
    stats_json = json.loads(stats.to_json_str())
    assert stats_json["id"] == stats.id
    assert stats_json["stage"] == "CONTEXT_IN_PROGRESS"
    assert stats_json[
        "contextPrefillPosition"] == stats.context_prefill_position
    assert stats_json["numGeneratedTokens"] == stats.num_generated_tokens
    assert stats_json[
        "avgNumDecodedTokensPerIter"] == stats.avg_num_decoded_tokens_per_iter
    assert stats_json["scheduled"] == stats.scheduled
    assert stats_json["paused"] == stats.paused
    assert stats_json["disServingStats"] is None


def test_request_stats_per_iteration():
    stats = trtllm.RequestStatsPerIteration()
    stats.iter = 1
    req_stat = trtllm.RequestStats()
    req_stat.id = 1
    stats.request_stats = [req_stat]
    stats_json = json.loads(stats.to_json_str())
    assert stats_json["iter"] == 1
    assert stats_json["requestStats"][0]["id"] == 1


def test_scheduler_config_pickle():
    policy = trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION
    config = trtllm.SchedulerConfig(policy)
    config_str = pickle.dumps(config)
    config_copy = pickle.loads(config_str)
    assert config.capacity_scheduler_policy == config_copy.capacity_scheduler_policy


def test_kv_cache_config_pickle():
    config = trtllm.KvCacheConfig(free_gpu_memory_fraction=0.5)
    config.enable_block_reuse = True
    config.max_tokens = 1
    config.max_attention_window = [2]
    config.sink_token_length = 3
    config.free_gpu_memory_fraction = 0.3
    config.cross_kv_cache_fraction = 0.5
    config.host_cache_size = 4
    config.onboard_blocks = False
    config.secondary_offload_min_priority = 50
    config.event_buffer_max_size = 1024
    config.enable_partial_reuse = False
    config.copy_on_partial_reuse = False
    config.use_uvm = True
    config_copy = pickle.loads(pickle.dumps(config))
    assert config.enable_block_reuse == config_copy.enable_block_reuse
    assert config.max_tokens == config_copy.max_tokens
    assert config.max_attention_window == config_copy.max_attention_window
    assert config.sink_token_length == config_copy.sink_token_length
    assert config.free_gpu_memory_fraction == config_copy.free_gpu_memory_fraction
    assert config.cross_kv_cache_fraction == config_copy.cross_kv_cache_fraction
    assert config.host_cache_size == config_copy.host_cache_size
    assert config.onboard_blocks == config_copy.onboard_blocks
    assert config.secondary_offload_min_priority == config_copy.secondary_offload_min_priority
    assert config.event_buffer_max_size == config_copy.event_buffer_max_size
    assert config.enable_partial_reuse == config_copy.enable_partial_reuse
    assert config.copy_on_partial_reuse == config_copy.copy_on_partial_reuse
    assert config.use_uvm == config_copy.use_uvm


def test_kv_cache_retention_config_pickle():
    config = trtllm.KvCacheRetentionConfig([
        trtllm.KvCacheRetentionConfig.TokenRangeRetentionConfig(
            0, 2, 30, datetime.timedelta(seconds=30))
    ], 80, None, trtllm.KvCacheTransferMode.GDS, "test_dir")
    config_copy = pickle.loads(pickle.dumps(config))
    assert config == config_copy


def test_peft_cache_config_pickle():
    config = trtllm.PeftCacheConfig(1, 2, 3, 4, 5, 6, 7, 8, 9, 0.9, 1024)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config.num_host_module_layer == config_copy.num_host_module_layer
    assert config.num_device_module_layer == config_copy.num_device_module_layer
    assert config.optimal_adapter_size == config_copy.optimal_adapter_size
    assert config.max_adapter_size == config_copy.max_adapter_size
    assert config.num_put_workers == config_copy.num_put_workers
    assert config.num_ensure_workers == config_copy.num_ensure_workers
    assert config.num_copy_streams == config_copy.num_copy_streams
    assert config.max_pages_per_block_host == config_copy.max_pages_per_block_host
    assert config.max_pages_per_block_device == config_copy.max_pages_per_block_device
    assert config.device_cache_percent == config_copy.device_cache_percent
    assert config.host_cache_size == config_copy.host_cache_size


def test_decoding_config_pickle():
    config = trtllm.DecodingConfig(
        decoding_mode=trtllm.DecodingMode.BeamSearch())
    config_copy = pickle.loads(pickle.dumps(config))
    assert config_copy.decoding_mode.isBeamSearch
    assert config.lookahead_decoding_config == config_copy.lookahead_decoding_config
    assert config.medusa_choices == config_copy.medusa_choices


def test_debug_config_pickle():
    config = trtllm.DebugConfig(debug_input_tensors=True,
                                debug_output_tensors=True,
                                debug_tensor_names=["test"],
                                debug_tensors_max_iterations=5)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config.debug_input_tensors == config_copy.debug_input_tensors
    assert config.debug_output_tensors == config_copy.debug_output_tensors
    assert config.debug_tensor_names == config_copy.debug_tensor_names
    assert config.debug_tensors_max_iterations == config_copy.debug_tensors_max_iterations


def test_logits_post_processor_config_pickle():
    kwargs = {
        "processor_map": {
            "test_pp": None
        },
        "processor_batched": None,
        "replicate": False
    }
    config = trtllm.LogitsPostProcessorConfig(**kwargs)
    config_copy = pickle.loads(pickle.dumps(config))
    for k in kwargs:
        assert getattr(config, k) == getattr(config_copy, k)


def test_guided_decoding_params_pickle():

    class Answer(BaseModel):
        answer: int

    json_schema = json.dumps(Answer.model_json_schema())
    params = trtllm.GuidedDecodingParams(
        trtllm.GuidedDecodingParams.GuideType.JSON_SCHEMA, guide=json_schema)
    params_copy = pickle.loads(pickle.dumps(params))
    assert params_copy.guide_type == params.guide_type
    assert params_copy.guide == params.guide


def test_guided_decoding_config_pickle():
    encoded_vocab = ["eos", "a", "b", "c", "d"]
    tokenizer_str = None
    stop_token_ids = [0]
    config = trtllm.GuidedDecodingConfig(
        backend=trtllm.GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
        encoded_vocab=encoded_vocab,
        tokenizer_str=tokenizer_str,
        stop_token_ids=stop_token_ids)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config_copy.backend == config.backend
    assert config_copy.encoded_vocab == config.encoded_vocab
    assert config_copy.tokenizer_str == config.tokenizer_str
    assert config_copy.stop_token_ids == config.stop_token_ids


def test_cache_transceiver_config_pickle():
    config = trtllm.CacheTransceiverConfig(
        backend=trtllm.CacheTransceiverBackendType.UCX,
        max_tokens_in_buffer=1024)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config_copy.backend == config.backend
    assert config_copy.max_tokens_in_buffer == config.max_tokens_in_buffer


def test_executor_config_pickle():
    beam_width = 2
    config = trtllm.ExecutorConfig(beam_width)

    kwargs = {
        "max_beam_width":
        2,
        "max_batch_size":
        8,
        "max_num_tokens":
        128,
        "scheduler_config":
        trtllm.SchedulerConfig(trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION),
        "kv_cache_config":
        trtllm.KvCacheConfig(enable_block_reuse=True,
                             free_gpu_memory_fraction=0.5),
        "enable_chunked_context":
        True,
        "normalize_log_probs":
        False,
        "iter_stats_max_iterations":
        100,
        "batching_type":
        trtllm.BatchingType.STATIC,
        "parallel_config":
        trtllm.ParallelConfig(),
        "peft_cache_config":
        trtllm.PeftCacheConfig(10),
        "logits_post_processor_config":
        trtllm.LogitsPostProcessorConfig(),
        "decoding_config":
        trtllm.DecodingConfig(trtllm.DecodingMode.TopKTopP()),
        "extended_runtime_perf_knob_config":
        trtllm.ExtendedRuntimePerfKnobConfig(multi_block_mode=True),
        "debug_config":
        trtllm.DebugConfig(debug_input_tensors=True,
                           debug_output_tensors=True,
                           debug_tensor_names=["test"]),
        "recv_poll_period_ms":
        50,
        "max_seq_idle_microseconds":
        240 * 1000 * 1000,
        "spec_dec_config":
        trtllm.SpeculativeDecodingConfig(fast_logits=True),
        "guided_decoding_config":
        trtllm.GuidedDecodingConfig(
            trtllm.GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
            encoded_vocab=["eos", "a", "b", "c", "d"]),
        "additional_model_outputs":
        [trtllm.AdditionalModelOutput("topKLogits")],
        "gather_generation_logits":
        True,
        "mm_embedding_offloading":
        True,
        "enable_trt_overlap":
        True,
    }
    config = trtllm.ExecutorConfig(**kwargs)
    for k, v in kwargs.items():
        if "config" not in k and k != "additional_model_outputs":
            assert getattr(config, k) == v

    config.backend = 'pytorch'

    pickle.dumps(config)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config.max_beam_width == config_copy.max_beam_width
    assert config.max_batch_size == config_copy.max_batch_size
    assert config.max_num_tokens == config_copy.max_num_tokens
    assert config.scheduler_config.capacity_scheduler_policy == config_copy.scheduler_config.capacity_scheduler_policy
    assert config.kv_cache_config.enable_block_reuse == config_copy.kv_cache_config.enable_block_reuse
    assert config.enable_chunked_context == config_copy.enable_chunked_context
    assert config.normalize_log_probs == config_copy.normalize_log_probs
    assert config.normalize_log_probs == config_copy.normalize_log_probs
    assert config.iter_stats_max_iterations == config_copy.iter_stats_max_iterations
    assert config.batching_type == config_copy.batching_type
    assert config.parallel_config.communication_type == config_copy.parallel_config.communication_type
    assert config.peft_cache_config.num_host_module_layer == config_copy.peft_cache_config.num_host_module_layer
    assert config_copy.decoding_config.decoding_mode.isTopKandTopP
    assert config.extended_runtime_perf_knob_config.multi_block_mode == config_copy.extended_runtime_perf_knob_config.multi_block_mode
    assert config.debug_config.debug_input_tensors == config_copy.debug_config.debug_input_tensors
    assert config.max_seq_idle_microseconds == config_copy.max_seq_idle_microseconds
    assert config.backend == config_copy.backend
    assert config.spec_dec_config.fast_logits == config_copy.spec_dec_config.fast_logits
    assert config.use_gpu_direct_storage == config_copy.use_gpu_direct_storage

    assert config_copy.guided_decoding_config.backend == config.guided_decoding_config.backend
    assert config_copy.guided_decoding_config.encoded_vocab == config.guided_decoding_config.encoded_vocab
    assert config_copy.guided_decoding_config.tokenizer_str == config.guided_decoding_config.tokenizer_str
    assert config_copy.guided_decoding_config.stop_token_ids == config.guided_decoding_config.stop_token_ids

    assert config.additional_model_outputs[
        0].name == config_copy.additional_model_outputs[0].name
    assert config.additional_model_outputs[
        0].gather_context == config_copy.additional_model_outputs[
            0].gather_context
    assert config.gather_generation_logits == config_copy.gather_generation_logits
    assert config.mm_embedding_offloading == config_copy.mm_embedding_offloading
    assert config.enable_trt_overlap == config_copy.enable_trt_overlap


def test_return_full_tokens():
    max_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens,
                             max_tokens=max_tokens,
                             streaming=False,
                             sampling_config=trtllm.SamplingConfig())
    request.return_all_generated_tokens = True
    assert request.return_all_generated_tokens == True
    request.return_all_generated_tokens = False
    assert request.return_all_generated_tokens == False


def test_getters_return_references():
    config = trtllm.ExecutorConfig()
    # Make sure kv_cache_config is a reference. Returning a value
    # will lead to the very confusing behavior of this set statement
    # not working.
    config.kv_cache_config.max_tokens = 42
    assert config.kv_cache_config.max_tokens == 42


def test_allotted_time_ms():
    allotted_time = datetime.timedelta(milliseconds=2)
    input_tokens = [1, 2, 3, 4]

    max_new_tokens = 5
    request = trtllm.Request(input_tokens, max_tokens=max_new_tokens)

    request.allotted_time_ms = allotted_time

    assert request.allotted_time_ms == datetime.timedelta(milliseconds=2)


def test_executor_version():
    assert trtllm.__version__ == trtllm_version.__version__


def get_all_field_names_of_class(cls: type) -> list[str]:
    return [
        name for name, obj in inspect.getmembers(cls)
        if isinstance(obj, property) or (
            not callable(obj) and not name.startswith('__'))
    ]


def test_runtime_defaults():
    full_runtime_defaults: dict[str, tp.Any] = json.loads("""{
        "max_attention_window": [1, 2],
        "sink_token_length": 4
    }""")
    all_field_names = set(full_runtime_defaults)

    assert set(
        get_all_field_names_of_class(trtllm.RuntimeDefaults)
    ) == all_field_names, "Expected fields of runtime_defaults to match actual data"

    msg = """\
    Rather than create a `from_dict` on top of the bound class, \
    we rely on being able to directly provide the dict created from raw json as kwargs to `RuntimeDefaults.` \
    See: `PretrainedConfig.__init__()`"""

    assert PretrainedConfig.create_runtime_defaults(
        full_runtime_defaults) is not None, msg

    default_runtime_defaults = trtllm.RuntimeDefaults()
    for key in all_field_names:
        assert getattr(default_runtime_defaults, key) is None


def test_request_pickle():

    samplingConfig = trtllm.SamplingConfig(beam_width=1,
                                           top_k=10,
                                           top_p=0.1,
                                           num_return_sequences=3)

    request = trtllm.Request(input_token_ids=[1, 2, 3, 4],
                             max_tokens=5,
                             streaming=False,
                             client_id=123,
                             sampling_config=samplingConfig)
    request_copy = pickle.loads(pickle.dumps(request))

    assert request.sampling_config.beam_width == request_copy.sampling_config.beam_width
    assert request.sampling_config.top_k == request_copy.sampling_config.top_k
    assert request.sampling_config.num_return_sequences == request_copy.sampling_config.num_return_sequences
    assert request.request_type == request_copy.request_type
    assert request.input_token_ids == request_copy.input_token_ids
    assert request.max_tokens == request_copy.max_tokens
    assert request.output_config.return_log_probs == request_copy.output_config.return_log_probs
    assert request.guided_decoding_params == request_copy.guided_decoding_params
    assert request.kv_cache_retention_config == request_copy.kv_cache_retention_config
