import os
import pathlib as _pl
import sys

import numpy as _np
import torch as _tor
from binding_test_utils import *

import tensorrt_llm.bindings as _tb

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


@pytest.mark.parametrize(
    "variant, results_file, load_bytearray",
    [
        ("fp32-default", "output_tokens_fp32_tp1_pp1.npy", True),
        ("fp32-plugin", "output_tokens_fp32_plugin_tp1_pp1.npy", False),
        ("fp16-default", "output_tokens_fp16_tp1_pp1.npy", True),
        ("fp16-plugin", "output_tokens_fp16_plugin_tp1_pp1.npy", False),
        # ("fp16-plugin-packed", "output_tokens_fp16_plugin_packed_tp1_pp1.npy"),
        # ("fp16-plugin-packed-paged", "output_tokens_fp16_plugin_packed_paged_tp1_pp1.npy"),
    ])
def test_gpt_session(variant, results_file, load_bytearray, llm_root: _pl.Path,
                     resource_path: _pl.Path, engine_path: _pl.Path,
                     data_path: _pl.Path, llm_model_root):
    if getSMVersion() < 80:
        pytest.skip(
            "ContextFMHAType with fp32 acc is not supported in pre-ampere architecture"
        )
    model_dir = "gpt2"
    tp_size = 1
    pp_size = 1
    beam_width = 1
    max_batch_size = 8
    end_id = 50256
    pad_id = 50256
    repetitions = 2

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
    session_config = _tb.GptSessionConfig(max_batch_size, beam_width,
                                          max_seq_length)

    model_config = config_json.model_config
    full_engine_path = str(model_path / engine_filename)
    if load_bytearray:
        with open(full_engine_path, "rb") as f:
            engine_data = bytearray(f.read())
        session = _tb.GptSession(session_config, model_config, world_config,
                                 engine_data)
    else:
        session = _tb.GptSession(session_config, model_config, world_config,
                                 full_engine_path)

    assert isinstance(session, _tb.GptSession)
    assert isinstance(session.model_config, _tb.GptModelConfig)
    assert isinstance(session.world_config, _tb.WorldConfig)
    assert session.device == world_config.device
    cuda_device = _tor.device("cuda", world_config.device)

    max_new_tokens = max_seq_length - max_input_length
    sampling_config = _tb.SamplingConfig(beam_width)
    sampling_config.temperature = [1.0]
    sampling_config.min_length = [1]
    sampling_config.random_seed = [42]
    sampling_config.top_k = [0]
    sampling_config.top_p = [0.0]

    packed_input = model_config.use_packed_input
    assert not packed_input
    input_ids = _tor.from_numpy(
        given_input[:max_batch_size, :max_input_length]).to(cuda_device)
    assert input_ids.dtype == _tor.int32
    input_lengths = _tor.from_numpy(
        given_input_lengths[:max_batch_size]).to(cuda_device)
    assert input_lengths.dtype == _tor.int32
    generation_input = _tb.GenerationInput(end_id, pad_id, input_ids,
                                           input_lengths, packed_input)
    generation_input.max_new_tokens = max_new_tokens

    for r in range(repetitions):
        output_ids = _tor.empty(0, dtype=_tor.int32, device=cuda_device)
        output_lengths = _tor.empty(0, dtype=_tor.int32, device=cuda_device)
        generation_output = _tb.GenerationOutput(output_ids, output_lengths)
        num_steps = 0

        def on_token_generated(ids, step, finished):
            assert ids.shape == (max_batch_size, 1, max_seq_length)
            nonlocal num_steps
            assert step == num_steps
            num_steps += 1
            # check that we only finish after producing `maxNewTokens` tokens
            assert not finished or num_steps == max_new_tokens
            # check that `finished` is set to true after producing `maxNewTokens` tokens
            assert num_steps != max_new_tokens or finished

        generation_output.on_token_generated = on_token_generated

        session.generate(generation_output, generation_input, sampling_config)
        observed_output = output_ids.squeeze().cpu().numpy()
        assert observed_output.shape == (max_batch_size, max_seq_length)
        observed_output_lengths = output_lengths.squeeze().cpu().numpy()
        assert _np.all(observed_output_lengths <= max_seq_length)

        for batch_idx in range(max_batch_size):
            expected_length = expected_output_lengths[batch_idx]
            observed_length = observed_output_lengths[batch_idx]
            assert expected_length == observed_length, (batch_idx,
                                                        expected_length,
                                                        observed_length)
            expected = expected_output[batch_idx, :expected_length]
            observed = observed_output[batch_idx, :expected_length]
            unmatched = expected != observed
            if _np.any(unmatched):
                assert False, (batch_idx, _np.where(unmatched),
                               _np.column_stack(
                                   (expected, observed))[unmatched])
