import json as _json
import os as _os
import pathlib as _pl
import sys as _sys
import time as _time
import typing as _tp

import numpy as _np
import pytest
import torch as _tor
from binding_test_utils import *
from transformers import AutoTokenizer

import tensorrt_llm.bindings as _tb

_sys.path.append(_os.path.join(_os.path.dirname(__file__), '..'))
from utils.cpp_paths import *
from utils.llm_data import llm_models_root
from utils.util import skip_pre_ampere

_sys.path.append(_os.path.join(_os.path.dirname(__file__), '..', '..'))
from cpp.tests.resources.scripts.build_engines_utils import \
    init_model_spec_module

init_model_spec_module()

import model_spec


def get_model_spec() -> model_spec.ModelSpec:
    if not hasattr(get_model_spec, 'model_spec_obj'):
        model_spec_obj = model_spec.ModelSpec(
            'input_tokens.npy',
            _tb.DataType.HALF).use_gpt_plugin().set_kv_cache_type(
                _tb.KVCacheType.PAGED).use_packed_input()
        get_model_spec.model_spec_obj = model_spec_obj

    return get_model_spec.model_spec_obj


@pytest.mark.parametrize("variant, results_file", [
    (get_model_spec().get_model_path(), get_model_spec().get_results_file()),
])
@skip_pre_ampere  # ContextFMHAType with fp32 acc is not supported in pre-ampere architecture
def test_gpt_manager(variant, results_file, llm_root: _pl.Path,
                     resource_path: _pl.Path, engine_path: _pl.Path,
                     data_path: _pl.Path):

    model_dir = "gpt2"
    tp_size = 1
    pp_size = 1
    beam_width = 1
    max_batch_size = 8
    end_id = 50256
    pad_id = 50256

    # load input data
    input_path = data_path / "input_tokens.npy"
    assert input_path.is_file()
    given_input = _np.load(input_path).astype("int32")
    input_shape = given_input.shape
    assert len(input_shape) == 2
    num_given_inputs = input_shape[0]
    assert max_batch_size <= num_given_inputs
    max_input_length = input_shape[1]
    given_input_lengths = sequence_lengths(given_input, pad_id)
    assert _np.all(given_input_lengths <= max_input_length)

    # load expected output data
    results_path = data_path / model_dir / (
        "sampling"
        if beam_width == 1 else f"beam_search_{beam_width}") / results_file

    if not results_path.exists():
        model_cache = llm_models_root()
        model_cache_arg = ["--model_cache", str(model_cache)
                           ] if model_cache is not None else []
        prepare_model_tests(llm_root, resource_path, "gpt", model_cache_arg)

    assert results_path.is_file()
    expected_output = _np.load(results_path)
    output_shape = expected_output.shape
    assert len(output_shape) == 2
    assert num_given_inputs * beam_width == output_shape[0]
    max_seq_length = output_shape[1]
    assert max_input_length <= max_seq_length
    expected_output_lengths = sequence_lengths(expected_output, end_id)
    assert _np.all(expected_output_lengths <= max_seq_length)

    gpu_size_path = f"tp{tp_size}-pp{pp_size}-gpu"
    model_path = engine_path / model_dir / variant / gpu_size_path
    assert model_path.is_dir()
    config_path = model_path / "config.json"
    config_json = _tb.GptJsonConfig.parse_file(config_path)
    assert config_json.tensor_parallelism == tp_size
    assert config_json.pipeline_parallelism == pp_size
    world_config = _tb.WorldConfig.mpi(tensor_parallelism=tp_size,
                                       pipeline_parallelism=pp_size)
    engine_filename = config_json.engine_filename(world_config)
    assert (model_path / engine_filename).is_file()

    config_json.model_config

    max_new_tokens = max_seq_length - max_input_length

    inference_request_list = []
    for i, (req, length) in enumerate(zip(given_input, given_input_lengths)):
        ir = _tb.InferenceRequest(i)
        ir.input_ids = _tor.tensor(req[:length].tolist(), dtype=_tor.int32)
        ir.max_new_tokens = _tor.tensor([[length + max_new_tokens]],
                                        dtype=_tor.int32)
        ir.end_id = _tor.tensor([end_id], dtype=_tor.int32)
        ir.pad_id = _tor.tensor([pad_id], dtype=_tor.int32)
        ir.beam_width = _tor.tensor([beam_width], dtype=_tor.int32)
        ir.temperature = _tor.tensor([1.0], dtype=_tor.float32)
        ir.min_length = _tor.tensor([1], dtype=_tor.int32)
        ir.random_seed = _tor.tensor([42], dtype=_tor.int64)
        ir.runtime_top_k = _tor.tensor([0], dtype=_tor.int32)
        ir.runtime_top_p = _tor.tensor([0.0], dtype=_tor.float32)
        inference_request_list.append(ir)

    def logits_post_processor(req_id: int, logits: _tor.Tensor,
                              ids: _tp.List[_tp.List[int]], stream: _tor.Stream,
                              client_id: _tp.Optional[int]):
        del req_id, ids

        cuda_stream = _tor.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=1,  # == kCUDA
        )

        with _tor.cuda.stream(cuda_stream):
            logits[:] = float("-inf")
            logits[..., 42] = 0

    ir = _tb.InferenceRequest(42, logits_post_processor)
    ir.input_ids = _tor.tensor(given_input[0].tolist(), dtype=_tor.int32)
    ir.max_new_tokens = _tor.tensor([[8]], dtype=_tor.int32)
    ir.end_id = _tor.tensor([end_id], dtype=_tor.int32)
    ir.pad_id = _tor.tensor([pad_id], dtype=_tor.int32)
    ir.beam_width = _tor.tensor([beam_width], dtype=_tor.int32)
    ir.temperature = _tor.tensor([1.0], dtype=_tor.float32)
    ir.min_length = _tor.tensor([1], dtype=_tor.int32)
    ir.random_seed = _tor.tensor([42], dtype=_tor.int64)
    ir.runtime_top_k = _tor.tensor([0], dtype=_tor.int32)
    ir.runtime_top_p = _tor.tensor([0.0], dtype=_tor.float32)
    inference_request_list.append(ir)

    def fetch_requests(max_num_sequences: int):
        nonlocal inference_request_list
        fetched = []
        for _ in range(max_num_sequences):
            try:
                fetched.append(inference_request_list.pop())
            except IndexError:
                break
        return fetched

    def response_cb(req_id: int, tensors: _tp.List[_tb.NamedTensor],
                    is_ok: bool, err_msg: str):
        nonlocal remaining_requests
        remaining_requests -= 1

        assert is_ok
        assert not err_msg
        tensor_dict = {item.name: item.tensor for item in tensors}

        batch_idx = req_id
        observed_output = tensor_dict[_tb.tensor_names.OUTPUT_IDS]
        assert observed_output is not None

        if req_id == 42:
            outputs = observed_output[..., len(given_input[0]):]
            assert _tor.allclose(outputs, _tor.tensor([42], dtype=_tor.int32))
            return

        expected_length = expected_output_lengths[batch_idx]
        observed_length = tensor_dict[_tb.tensor_names.SEQUENCE_LENGTH].item(
        ) - given_input_lengths[batch_idx]
        assert expected_length == observed_length, (batch_idx, expected_length,
                                                    observed_length)
        expected = expected_output[batch_idx, :expected_length]
        observed = observed_output[0, 0, :expected_length].numpy()
        unmatched = expected != observed
        if _np.any(unmatched):
            assert False, (batch_idx, _np.where(unmatched),
                           _np.column_stack((expected, observed))[unmatched])

    def should_stop():
        return set()

    def stats_cb(stats_json: str):
        assert _json.loads(stats_json)

    opt_params = _tb.TrtGptModelOptionalParams()
    opt_params.max_beam_width = 1
    opt_params.scheduler_config = _tb.executor.SchedulerConfig(
        _tb.executor.CapacitySchedulerPolicy.MAX_UTILIZATION)

    memory_counters = _tb.MemoryCounters.instance()
    init_gpu_mem = memory_counters.gpu

    for _ in range(3):
        remaining_requests = len(inference_request_list)
        with _tb.GptManager(model_path, _tb.TrtGptModelType.InflightBatching,
                            fetch_requests, response_cb, should_stop, stats_cb,
                            opt_params, 10000) as manager:
            while remaining_requests > 0:
                _time.sleep(0.1)
            assert manager is not None
            assert memory_counters.gpu > init_gpu_mem

        assert memory_counters.gpu == init_gpu_mem


@pytest.mark.parametrize("variant, results_file", [
    (get_model_spec().get_model_path(), get_model_spec().get_results_file()),
])
def test_gpt_manager_constrained_generation(variant, results_file,
                                            llm_root: _pl.Path,
                                            resource_path: _pl.Path,
                                            engine_path: _pl.Path,
                                            data_path: _pl.Path):
    try:
        from lmformatenforcer import (JsonSchemaParser, TokenEnforcer,
                                      TokenEnforcerTokenizerData)
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("Cannot import lmformatenforcer, skipping test")

    def _build_regular_tokens_list(
            tokenizer) -> _tp.List[_tp.Tuple[int, str, bool]]:
        token_0 = [tokenizer.encode("0")[-1]]
        regular_tokens = []
        vocab_size = tokenizer.vocab_size
        for token_idx in range(vocab_size):
            if token_idx in tokenizer.all_special_ids:
                continue
            # We prepend token 0 and skip the first letter of the result to get a space if the token is a start word.
            tensor_after_0 = _tor.tensor(token_0 + [token_idx], dtype=_tor.long)
            decoded_after_0 = tokenizer.decode(tensor_after_0)[1:]
            decoded_regular = tokenizer.decode(token_0)
            is_word_start_token = len(decoded_after_0) > len(decoded_regular)
            regular_tokens.append(
                (token_idx, decoded_after_0, is_word_start_token))
        return regular_tokens

    def build_token_enforcer(tokenizer, character_level_parser):
        """
        Build logits processor for feeding it into generate function (use_py_session should be True)
        """
        regular_tokens = _build_regular_tokens_list(tokenizer)

        def _decode(tokens: _tp.List[int]) -> str:
            tensor = _tor.tensor(tokens, dtype=_tor.long)
            return tokenizer.decode(tensor)

        tokenizer_data = TokenEnforcerTokenizerData(regular_tokens, _decode,
                                                    tokenizer.eos_token_id)
        return TokenEnforcer(tokenizer_data, character_level_parser)

    tp_size = 1
    pp_size = 1
    model_dir = "gpt2"
    gpu_size_path = f"tp{tp_size}-pp{pp_size}-gpu"
    model_path = engine_path / model_dir / variant / gpu_size_path

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input = "Please give me information about Michael Jordan. You MUST answer using the following json schema: "
    prompt = tokenizer.encode(input)

    class AnswerFormat(BaseModel):
        last_name: str
        year_of_birth: int

    parser = JsonSchemaParser(AnswerFormat.model_json_schema())
    token_enforcer = build_token_enforcer(tokenizer, parser)

    def fetch_requests(max_num_sequences: int):
        nonlocal inference_request_list
        fetched = []
        for _ in range(max_num_sequences):
            try:
                fetched.append(inference_request_list.pop())
            except IndexError:
                break
        return fetched

    def logits_post_processor(req_id: int, logits: _tor.Tensor,
                              ids: _tp.List[_tp.List[int]], stream: _tor.Stream,
                              client_id: _tp.Optional[int]):
        del req_id

        cuda_stream = _tor.cuda.Stream(
            stream_id=stream.stream_id,
            device_index=stream.device_index,
            device_type=1,  # == kCUDA
        )

        def _trim(ids):
            return [x for x in ids if x != tokenizer.eos_token_id]

        allowed = token_enforcer.get_allowed_tokens(_trim(ids[0]))
        mask = _tor.full_like(logits, fill_value=float("-inf"), device="cpu")
        mask[:, :, allowed] = 0
        mask = mask.to(logits.device)

        with _tor.cuda.stream(cuda_stream):
            logits += mask

    result = ""

    def response_cb(req_id: int, tensors: _tp.List[_tb.NamedTensor],
                    is_finished: bool, err_msg: str):
        nonlocal remaining_requests, result
        assert not err_msg
        assert is_finished
        tensors_dict = {t.name: t.tensor for t in tensors}

        result = tokenizer.decode(tensors_dict["output_ids"].squeeze().tolist())
        remaining_requests -= 1

    def should_stop():
        return set()

    def stats_cb(stats_json: str):
        assert _json.loads(stats_json)

    inference_request_list = []
    ir = _tb.InferenceRequest(42, logits_post_processor)
    ir.input_ids = _tor.tensor(prompt, dtype=_tor.int32)
    ir.max_new_tokens = _tor.tensor([[64]], dtype=_tor.int32)
    ir.end_id = _tor.tensor([tokenizer.eos_token_id], dtype=_tor.int32)
    inference_request_list.append(ir)

    remaining_requests = len(inference_request_list)
    opt_params = _tb.TrtGptModelOptionalParams()
    opt_params.max_beam_width = 1
    with _tb.GptManager(model_path, _tb.TrtGptModelType.InflightBatching,
                        fetch_requests, response_cb, should_stop, stats_cb,
                        opt_params, 10000):
        while remaining_requests > 0:
            _time.sleep(0.1)

    assert result == input + '            { "last_name": "Michael Jordan", "year_of_birth": 18 }            '
