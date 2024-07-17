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

import tensorrt_llm.bindings.executor as trtllm

_sys.path.append(_os.path.join(_os.path.dirname(__file__), '..'))
from utils.cpp_paths import *
from utils.llm_data import llm_models_root
from utils.util import skip_pre_ampere


@pytest.fixture
def model_files(llm_root: Path, resource_path: Path, results_data_path):
    # Model engines and expected outputs need to be generated.
    if not results_data_path.exists():
        model_cache = llm_models_root()
        model_cache_arg = ["--model_cache", str(model_cache)
                           ] if model_cache is not None else []
        prepare_model_tests(llm_root, resource_path, "gpt", model_cache_arg)


def get_expected_num_tokens(prompt_len, max_new_tokens, streaming,
                            exclude_input_from_output):
    if not streaming and not exclude_input_from_output:
        return prompt_len + max_new_tokens
    return max_new_tokens


@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_executor_valid_ctor(model_files, model_path):
    executor_config = trtllm.ExecutorConfig(1)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)


@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_executor_from_memory(model_files, model_path):
    executor_config = trtllm.ExecutorConfig(1)
    engine_buffer = open(model_path / "rank0.engine", mode="rb").read()
    json_config_str = open(model_path / "config.json", 'r').read()
    executor = trtllm.Executor(engine_buffer, json_config_str,
                               trtllm.ModelType.DECODER_ONLY, executor_config)


def test_executor_invalid_ctor():
    executor_config = trtllm.ExecutorConfig(1)
    invalid_path = "Bla"
    try:
        executor = trtllm.Executor(invalid_path, trtllm.ModelType.DECODER_ONLY,
                                   executor_config)
        assert False, "Expected an error"
    except Exception as e:
        assert "File does not exist" in str(e)


@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_shutdown(model_files, model_path):

    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_new_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens, max_new_tokens, False,
                             trtllm.SamplingConfig())

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
        executor.get_latest_iteration_stats()()
    with pytest.raises(Exception):
        executor.get_latest_request_stats()()
    with pytest.raises(Exception):
        executor.cancel_request(req_id)()
    with pytest.raises(Exception):
        executor.get_num_responses_ready(req_id)()


@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_embedding_bias(model_files, model_path):
    streaming = False
    exclude_input_from_output = False
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_new_tokens = 5
    input_tokens = [1, 2, 3, 4]
    # Set embedding bias so "biased_output" is always picked
    biased_output = 10
    vocab_size_padded = 50257
    embedding_bias = torch.zeros(vocab_size_padded)
    embedding_bias[biased_output] = torch.finfo(torch.float32).max
    request = trtllm.Request(input_tokens,
                             max_new_tokens,
                             streaming,
                             trtllm.SamplingConfig(),
                             output_config,
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
        len(input_tokens), max_new_tokens, streaming,
        exclude_input_from_output), f"{request_id}"
    # All generated tokens should equal biased_output
    assert tokens[-max_new_tokens:] == [biased_output] * max_new_tokens


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exclude_input_from_output", [False])
@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_single_request(streaming: bool, exclude_input_from_output: bool,
                        model_files, model_path):
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_new_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens, max_new_tokens, streaming,
                             trtllm.SamplingConfig(), output_config)

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
        len(input_tokens), max_new_tokens, streaming,
        exclude_input_from_output), f"{request_id}"

    executor.get_latest_iteration_stats()
    executor.get_latest_request_stats()


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exclude_input_from_output", [False])
@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_multi_request(streaming: bool, exclude_input_from_output: bool,
                       model_files, model_path):
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    num_requests = 20
    max_prompt_len = 20
    max_max_new_tokens = 20
    end_id = -1

    # Enqueue the requests
    tokens = {}
    expected_num_tokens = {}
    for i in range(num_requests):
        prompt_len = random.randint(1, max_prompt_len)
        max_new_tokens = random.randint(1, max_max_new_tokens)
        input_tokens = [1] * prompt_len
        request = trtllm.Request(input_tokens, max_new_tokens, streaming,
                                 trtllm.SamplingConfig(), output_config, end_id)
        request_id = executor.enqueue_request(request)
        tokens[request_id] = []
        expected_num_tokens[request_id] = get_expected_num_tokens(
            prompt_len, max_new_tokens, streaming, exclude_input_from_output)

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
            tokens[response.request_id].extend(new_tokens)
        i += 1
    assert i < max_wait_ms

    for request_id in expected_num_tokens:
        assert len(tokens[request_id]) == expected_num_tokens[request_id]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exclude_input_from_output", [False])
@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_multi_request_with_ids(streaming: bool,
                                exclude_input_from_output: bool, model_files,
                                model_path):
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    num_requests = 20
    max_prompt_len = 20
    max_max_new_tokens = 20
    end_id = -1

    # Enqueue the requests
    tokens = {}
    expected_num_tokens = {}
    for i in range(num_requests):
        prompt_len = random.randint(1, max_prompt_len)
        max_new_tokens = random.randint(1, max_max_new_tokens)
        input_tokens = [1] * prompt_len
        request = trtllm.Request(input_tokens, max_new_tokens, streaming,
                                 trtllm.SamplingConfig(), output_config, end_id)
        request_id = executor.enqueue_request(request)
        tokens[request_id] = []
        expected_num_tokens[request_id] = get_expected_num_tokens(
            prompt_len, max_new_tokens, streaming, exclude_input_from_output)

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
                    assert response.error_msg == terminated_request_error, f"Request id {response.request_id} failed with err {response.error_msg}"
                else:
                    result = response.result
                    num_finished += result.is_final
                    new_tokens = result.output_token_ids[beam_width - 1]
                    tokens[response.request_id].extend(new_tokens)
            i += 1
    assert i < max_wait_ms

    for request_id in expected_num_tokens:
        assert len(tokens[request_id]) == expected_num_tokens[request_id]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("exclude_input_from_output", [False])
@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_get_num_responses_ready(streaming: bool,
                                 exclude_input_from_output: bool, model_files,
                                 model_path):
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output

    # Create executor
    executor_config = trtllm.ExecutorConfig(1)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    max_prompt_len = 20
    max_max_new_tokens = 20

    # Enqueue the requests
    num_requests = random.randint(1, 50)
    num_expected_responses = 0
    req_num_expected_responses = {}
    for i in range(num_requests):
        prompt_len = random.randint(1, max_prompt_len)
        max_new_tokens = random.randint(1, max_max_new_tokens)

        request = trtllm.Request([1] * prompt_len, max_new_tokens, streaming,
                                 trtllm.SamplingConfig(), output_config)
        request_id = executor.enqueue_request(request)
        req_num_expected_responses[
            request_id] = max_new_tokens if streaming else 1
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
@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
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
        max_new_tokens = max_seq_length - max_input_length

        end_ids = [pad_id for _ in range(len(given_input_lengths))]
        expected_lengths = []
        for i in range(len(given_input_lengths)):
            expected_lengths.append([
                given_input_lengths[i] + max_new_tokens
                for _ in range(beam_width)
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
            assert list(result.generation_logits.shape) == [
                beam_width, max_output_len, vocab_size_padded
            ]

    def verify_output(beam_tokens, test_data, given_input_lengths):
        for batch_id, tokens in beam_tokens.items():
            input_length = given_input_lengths[batch_id]
            end_id = test_data["end_ids"][batch_id]
            for beam in range(beam_width):
                predicted_tokens = tokens[beam]
                if remove_input:
                    predicted_tokens = predicted_tokens[input_length:]
                expected_length = test_data["expected_output_lengths"][
                    batch_id][beam] - input_length
                assert len(predicted_tokens) == expected_length
                expected_tokens = test_data["expected_output_ids"][
                    batch_id * beam_width + beam][input_length:]
                for i in range(len(predicted_tokens)):
                    if expected_tokens[i] == end_id:
                        break
                    assert predicted_tokens[i] == expected_tokens[i], \
                        f"Predicted: {predicted_tokens} vs Expected: {expected_tokens}"

    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = exclude_input_from_output
    output_config.return_log_probs = compute_log_probs
    output_config.return_generation_logits = return_generation_logits
    output_config.return_context_logits = return_context_logits

    kv_cache_config = trtllm.KvCacheConfig(False, free_gpu_memory_fraction=0.5)
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor_config.batching_type = batching_type
    executor_config.kv_cache_config = kv_cache_config

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
    req_max_new_tokens = []

    for i in range(num_requests):
        input_len = given_input_lengths[i]
        max_new_tokens = test_data["max_seq_length"] - max_input_length
        req_max_new_tokens.append(max_new_tokens)
        req_tokens = given_input[i][:input_len]
        requests.append(
            trtllm.Request(req_tokens,
                           max_new_tokens,
                           streaming,
                           trtllm.SamplingConfig(beam_width),
                           output_config,
                           end_id=-1))

    req_ids = executor.enqueue_requests(requests)

    req_to_batch_id = {req_ids[i]: i for i in range(len(requests))}
    tokens = {i: [[] for _ in range(beam_width)] for i in range(len(requests))}

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
                tokens[batch_id][beam] += new_tokens

            validate_results_shapes(result, given_input_lengths[batch_id],
                                    req_max_new_tokens[batch_id],
                                    tokens[batch_id])
        i += 1
    assert i < max_wait_ms
    verify_output(tokens, test_data, given_input_lengths)


@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_gpt_executor_timed_out(model_files, model_path):
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # No requests enqueued, expect no responses
    num_responses_ready = executor.get_num_responses_ready()
    assert num_responses_ready == 0

    wait_time = datetime.timedelta(milliseconds=10)
    responses = executor.await_responses(wait_time)
    assert len(responses) == 0


@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_single_request_invalid_inputs(model_files, model_path):
    streaming = True
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    max_new_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens, max_new_tokens, streaming)
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
        "random_seed": 7,
        "temperature": 1.0,
        "min_length": 4,
        "beam_search_diversity_rate": 1.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 1.0,
        "frequency_penalty": 1.0,
        "length_penalty": 1.0,
        "early_stopping": 5
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

    config = trtllm.OutputConfig(True, False, True, False)
    assert config.return_log_probs == True
    assert config.return_context_logits == False
    assert config.return_generation_logits == True
    assert config.exclude_input_from_output == False


def test_external_draft_tokens_config():
    tokens = [1, 2, 3]
    config = trtllm.ExternalDraftTokensConfig(tokens)
    assert config.tokens == tokens
    assert config.logits is None
    assert config.acceptance_threshold is None
    del config

    logits = torch.ones(3, 1)
    acceptance_threshold = 1.0
    config = trtllm.ExternalDraftTokensConfig(tokens, logits,
                                              acceptance_threshold)
    assert config.tokens == tokens
    assert (config.logits == logits).all()
    assert config.acceptance_threshold == acceptance_threshold


def test_prompt_tuning_config():
    embedding_table = torch.ones(100, 64)
    config = trtllm.PromptTuningConfig(embedding_table)
    assert (config.embedding_table == embedding_table).all()


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


@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
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

    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               trtllm.ExecutorConfig())
    stop_signal = threading.Event()
    thread = threading.Thread(target=resp_thread, args=(stop_signal, executor))
    thread.start()
    request = trtllm.Request(input_token_ids=[1, 2, 3, 4],
                             max_new_tokens=5,
                             streaming=True)
    executor.enqueue_request(request)
    time.sleep(2)
    stop_signal.set()
    executor.shutdown()
    thread.join()
    assert not thread.is_alive()


def test_request():
    kwargs = {
        "input_token_ids": [1, 2, 3],
        "max_new_tokens": 1,
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
        "lora_config": trtllm.LoraConfig(1)
    }
    request = trtllm.Request(**kwargs)
    for k, v in kwargs.items():
        if "config" not in k:
            assert getattr(request, k) == v
    assert isinstance(request.sampling_config, trtllm.SamplingConfig)
    assert isinstance(request.output_config, trtllm.OutputConfig)
    assert isinstance(request.external_draft_tokens_config,
                      trtllm.ExternalDraftTokensConfig)
    assert request.external_draft_tokens_config.tokens == [1, 2, 3]
    assert isinstance(request.prompt_tuning_config, trtllm.PromptTuningConfig)
    assert (request.prompt_tuning_config.embedding_table == torch.ones(
        100, 64)).all()
    assert isinstance(request.lora_config, trtllm.LoraConfig)


def test_result():
    result = trtllm.Result()
    result.is_final = True
    result.output_token_ids = [[1, 2, 3]]
    result.cum_log_probs = [1.0, 2.0, 3.0]
    result.log_probs = [[1.0, 2.0, 3.0]]
    result.context_logits = torch.ones(3, 100)
    result.generation_logits = torch.ones(1, 3, 100)
    assert result.is_final == True
    assert result.output_token_ids == [[1, 2, 3]]
    assert result.cum_log_probs == [1.0, 2.0, 3.0]
    assert result.log_probs == [[1.0, 2.0, 3.0]]
    assert (result.context_logits == torch.ones(3, 100)).all()
    assert (result.generation_logits == torch.ones(1, 3, 100)).all()


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


def test_scheduler_config():
    capacity_scheduler_policy = trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    config = trtllm.SchedulerConfig()
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy == None

    capacity_scheduler_policy = trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION
    config = trtllm.SchedulerConfig(capacity_scheduler_policy)
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy == None

    capacity_scheduler_policy = trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    config = trtllm.SchedulerConfig(capacity_scheduler_policy)
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy == None

    context_chunking_policy = trtllm.ContextChunkingPolicy.FIRST_COME_FIRST_SERVED
    config = trtllm.SchedulerConfig(capacity_scheduler_policy,
                                    context_chunking_policy)
    assert config.capacity_scheduler_policy == capacity_scheduler_policy
    assert config.context_chunking_policy == context_chunking_policy


def test_kv_cache_config():
    config = trtllm.KvCacheConfig()
    assert config.enable_block_reuse == False
    assert config.max_tokens is None
    assert config.max_attention_window is None
    assert config.sink_token_length is None
    assert config.free_gpu_memory_fraction is None
    assert config.host_cache_size is None
    assert config.onboard_blocks == True

    config.enable_block_reuse = True
    config.max_tokens = 1
    config.max_attention_window = 2
    config.sink_token_length = 3
    config.free_gpu_memory_fraction = 0.5
    config.host_cache_size = 4
    config.onboard_blocks = False
    assert config.enable_block_reuse == True
    assert config.max_tokens == 1
    assert config.max_attention_window == 2
    assert config.sink_token_length == 3
    assert config.free_gpu_memory_fraction == 0.5
    assert config.host_cache_size == 4
    assert config.onboard_blocks == False

    kwargs = {
        "enable_block_reuse": True,
        "max_tokens": 3,
        "max_attention_window": 10,
        "sink_token_length": 2,
        "free_gpu_memory_fraction": 0.5,
        "host_cache_size": 1024,
        "onboard_blocks": False,
    }
    config = trtllm.KvCacheConfig(**kwargs)
    for k, v in kwargs.items():
        assert getattr(config, k) == v


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


def test_speculative_decoding_config():
    config = trtllm.DecodingConfig()
    assert config.decoding_mode is None
    assert config.lookahead_decoding_config is None
    assert config.medusa_choices is None

    config = trtllm.DecodingConfig()
    config.decoding_mode = trtllm.DecodingMode.TopKTopP()
    assert config.decoding_mode.isTopKandTopP()
    assert config.lookahead_decoding_config == None
    assert config.medusa_choices == None

    config = trtllm.DecodingConfig()
    la_decoding_config = trtllm.LookaheadDecodingConfig(3, 5, 7)
    config.lookahead_decoding_config = la_decoding_config

    assert config.decoding_mode.isLookahead()
    assert config.lookahead_decoding_config.max_ngram_size == la_decoding_config.max_ngram_size
    assert config.lookahead_decoding_config.max_window_size == la_decoding_config.max_window_size
    assert config.lookahead_decoding_config.max_verification_set_size == la_decoding_config.max_verification_set_size
    assert config.medusa_choices == None

    config = trtllm.DecodingConfig()
    config.medusa_choices = [[0, 0], [0, 1]]

    assert config.decoding_mode.isMedusa()
    assert config.lookahead_decoding_config == None
    assert config.medusa_choices == [[0, 0], [0, 1]]


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
    assert config.logits_post_processor_map is None
    assert config.logits_post_processor_batched is None
    assert config.decoding_config is None

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
        trtllm.KvCacheConfig(),
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
        "logits_post_processor_map": {},
        "decoding_config":
        trtllm.DecodingConfig(trtllm.DecodingMode.TopKTopP()),
    }
    config = trtllm.ExecutorConfig(**kwargs)
    for k, v in kwargs.items():
        if "config" not in k:
            assert getattr(config, k) == v
    assert isinstance(config.scheduler_config, trtllm.SchedulerConfig)
    assert config.scheduler_config.capacity_scheduler_policy == trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION
    assert isinstance(config.kv_cache_config, trtllm.KvCacheConfig)
    assert isinstance(config.parallel_config, trtllm.ParallelConfig)
    assert isinstance(config.peft_cache_config, trtllm.PeftCacheConfig)


def test_parallel_config():
    comm_type = trtllm.CommunicationType.MPI
    comm_mode = trtllm.CommunicationMode.LEADER
    device_ids = [0, 1, 2, 3]
    participant_ids = [4, 5, 6, 7]
    parallel_config = trtllm.ParallelConfig(comm_type, comm_mode, device_ids,
                                            participant_ids)
    assert parallel_config.communication_type == comm_type
    assert parallel_config.communication_mode == comm_mode
    assert parallel_config.device_ids == device_ids
    assert parallel_config.participant_ids == participant_ids

    comm_mode = trtllm.CommunicationMode.ORCHESTRATOR
    #Dummy path to worker executable
    worker_path = _os.path.abspath(__file__)
    orchestrator_config = trtllm.OrchestratorConfig(True, str(worker_path))
    parallel_config = trtllm.ParallelConfig(comm_type, comm_mode, device_ids,
                                            participant_ids,
                                            orchestrator_config)
    assert parallel_config.communication_mode == comm_mode
    assert parallel_config.orchestrator_config.is_orchestrator == True
    assert parallel_config.orchestrator_config.worker_executable_path == str(
        worker_path)


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
    peft_cache_config = trtllm.PeftCacheConfig(
        num_host_module_layer, num_device_module_layer, optimal_adapter_size,
        max_adapter_size, num_put_workers, num_ensure_workers, num_copy_streams,
        max_pages_per_block_host, max_pages_per_block_device,
        device_cache_percent, host_cache_size)

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


@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_logits_post_processor(model_files, model_path):

    # Define the logits post-processor callback
    def logits_post_processor(req_id: int, logits: torch.Tensor,
                              ids: tp.List[tp.List[int]], stream_ptr: int):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits[:] = float("-inf")
            logits[..., 42] = 0

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor_config.logits_post_processor_map = {
        "my_logits_pp": logits_post_processor
    }
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_new_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens, max_new_tokens, False)
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
    assert len(tokens) == get_expected_num_tokens(len(input_tokens),
                                                  max_new_tokens, False,
                                                  False), f"{request_id}"

    # check that all output tokens are 42
    print(tokens)
    assert tokens[-max_new_tokens:] == [42] * max_new_tokens


@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_logits_post_processor_batched(model_files, model_path):

    # Define the logits post-processor callback
    def logits_post_processor_batched(req_id_batch: tp.List[int],
                                      logits_batch: tp.List[torch.Tensor],
                                      ids_batch: tp.List[tp.List[tp.List[int]]],
                                      stream_ptr: int):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            for logits in logits_batch:
                logits[:] = float("-inf")
                logits[..., 42] = 0

    # Create executor
    beam_width = 1
    executor_config = trtllm.ExecutorConfig(beam_width)
    executor_config.logits_post_processor_batched = logits_post_processor_batched
    executor = trtllm.Executor(model_path, trtllm.ModelType.DECODER_ONLY,
                               executor_config)

    # Create the request
    max_new_tokens = 5
    input_tokens = [1, 2, 3, 4]
    request = trtllm.Request(input_tokens, max_new_tokens, False)
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

    expected_num_tokens = get_expected_num_tokens(len(input_tokens),
                                                  max_new_tokens, False, False)
    for req_id in request_ids:
        assert len(tokens[req_id]) == expected_num_tokens, f"{req_id}"


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
    assert stats_json["numActiveRequests"] == stats.num_active_requests
    assert stats_json["numQueuedRequests"] == stats.num_queued_requests
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
    config = trtllm.KvCacheConfig()
    config.enable_block_reuse = True
    config.free_gpu_memory_fraction = 0.3
    config_copy = pickle.loads(pickle.dumps(config))
    assert config.enable_block_reuse == config_copy.enable_block_reuse
    assert config.max_tokens == config_copy.max_tokens
    assert config.max_attention_window == config_copy.max_attention_window
    assert config.sink_token_length == config_copy.sink_token_length
    assert config.free_gpu_memory_fraction == config_copy.free_gpu_memory_fraction
    assert config.host_cache_size == config_copy.host_cache_size
    assert config.onboard_blocks == config_copy.onboard_blocks


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


def test_executor_config_pickle():
    beam_width = 2
    config = trtllm.ExecutorConfig(beam_width)
    config.scheduler_config = trtllm.SchedulerConfig()
    config.kv_cache_config = trtllm.KvCacheConfig()
    config.parallel_config = trtllm.ParallelConfig()
    config.peft_cache_config = trtllm.PeftCacheConfig(1)
    pickle.dumps(config)
    config_copy = pickle.loads(pickle.dumps(config))
    assert config.max_beam_width == config_copy.max_beam_width
    assert config.scheduler_config.capacity_scheduler_policy == config_copy.scheduler_config.capacity_scheduler_policy
    assert config.kv_cache_config.enable_block_reuse == config_copy.kv_cache_config.enable_block_reuse
