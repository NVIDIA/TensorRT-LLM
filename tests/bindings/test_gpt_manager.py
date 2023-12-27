import json as _json
import pathlib as _pl
import time as _time
import typing as _tp

import numpy as _np
import torch as _tor
from binding_test_utils import *

import tensorrt_llm.bindings as _tb


@pytest.mark.parametrize("variant, results_file", [
    ("fp16-plugin-packed-paged",
     "output_tokens_fp16_plugin_packed_paged_tp1_pp1.npy"),
])
def test_gpt_manager(variant, results_file, llm_root: _pl.Path,
                     resource_path: _pl.Path, engine_path: _pl.Path,
                     data_path: _pl.Path, llm_model_root):
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
        model_cache_arg = ["--model_cache",
                           str(llm_model_root)
                           ] if llm_model_root is not None else []
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
    sampling_config = _tb.SamplingConfig(beam_width)
    sampling_config.temperature = [1.0]
    sampling_config.min_length = [1]
    sampling_config.random_seed = [42]
    sampling_config.top_k = [0]
    sampling_config.top_p = [0.0]

    inference_request_list = []
    remaining_requests = len(given_input)
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

        assert is_ok
        assert not err_msg
        tensor_dict = {item.name: item.tensor for item in tensors}

        batch_idx = req_id
        observed_output = tensor_dict[_tb.tensor_names.OUTPUT_IDS]

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

        remaining_requests -= 1

    def should_stop():
        return set()

    def stats_cb(stats_json: str):
        assert _json.loads(stats_json)

    opt_params = _tb.TrtGptModelOptionalParams()

    with _tb.GptManager(model_path, _tb.TrtGptModelType.InflightBatching, 1,
                        _tb.SchedulerPolicy.MAX_UTILIZATION, fetch_requests,
                        response_cb, should_stop, stats_cb, opt_params, 10000):
        while remaining_requests > 0:
            _time.sleep(0.1)
