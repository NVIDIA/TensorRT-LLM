import json
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from utils.runtime_defaults import assert_runtime_defaults_are_parsed_correctly

import tensorrt_llm.bindings as _tb
from tensorrt_llm.mapping import Mapping


def test_quant_mode():
    assert _tb.QuantMode.none().value == 0
    assert _tb.QuantMode.int4_weights().has_int4_weights
    assert _tb.QuantMode.int8_weights().has_int8_weights
    assert _tb.QuantMode.activations().has_activations
    assert _tb.QuantMode.per_channel_scaling().has_per_channel_scaling
    assert _tb.QuantMode.per_token_scaling().has_per_token_scaling
    assert _tb.QuantMode.per_group_scaling().has_per_group_scaling
    assert _tb.QuantMode.int8_kv_cache().has_int8_kv_cache
    assert _tb.QuantMode.fp8_kv_cache().has_fp8_kv_cache
    assert _tb.QuantMode.fp8_qdq().has_fp8_qdq

    quant_mode = _tb.QuantMode.from_description(True, True, True, True, True,
                                                True, True, True, False, False,
                                                False, False, False, False,
                                                False, False)
    assert quant_mode.has_int4_weights
    quant_mode -= _tb.QuantMode.int4_weights()
    assert not quant_mode.has_int4_weights
    quant_mode += _tb.QuantMode.int4_weights()
    assert quant_mode.has_int4_weights

    assert _tb.QuantMode.none() == _tb.QuantMode.none()


def test_model_config():
    vocab_size = 10000
    num_attention_layers = 12
    num_rnn_layers = 2
    num_heads = 16
    hidden_size = 768
    data_type = _tb.DataType.FLOAT
    model_config = _tb.ModelConfig(vocab_size,
                                   num_attention_layers + num_rnn_layers,
                                   num_attention_layers, num_rnn_layers,
                                   num_heads, hidden_size, data_type)
    assert model_config.vocab_size == vocab_size
    assert model_config.num_attention_layers() == num_attention_layers
    assert model_config.num_rnn_layers() == num_rnn_layers
    assert model_config.num_heads == num_heads
    assert model_config.hidden_size == hidden_size
    assert model_config.data_type == data_type

    assert model_config.vocab_size_padded(1) is not None
    assert model_config.size_per_head == hidden_size // num_heads

    num_kv_heads_per_layer = model_config.num_kv_heads_per_layer
    for layer_idx in range(num_attention_layers):
        assert model_config.num_kv_heads(layer_idx) == num_heads
        assert num_kv_heads_per_layer[layer_idx] == num_heads

    num_kv_heads = 1
    model_config.set_num_kv_heads(num_kv_heads)
    num_kv_heads_per_layer = model_config.num_kv_heads_per_layer
    for layer_idx in range(num_attention_layers):
        assert model_config.num_kv_heads(layer_idx) == num_kv_heads
        assert num_kv_heads_per_layer[layer_idx] == num_kv_heads

    num_kv_heads_per_layer[-1] = 2
    model_config.num_kv_heads_per_layer = num_kv_heads_per_layer
    for nheads, ref in zip(model_config.num_kv_heads_per_layer,
                           num_kv_heads_per_layer):
        assert nheads == ref

    assert not model_config.use_gpt_attention_plugin
    model_config.use_gpt_attention_plugin = True
    assert model_config.use_gpt_attention_plugin

    assert not model_config.use_packed_input
    model_config.use_packed_input = True
    assert model_config.use_packed_input

    assert model_config.kv_cache_type is not None
    for enum_val in [
            _tb.KVCacheType.CONTINUOUS, _tb.KVCacheType.PAGED,
            _tb.KVCacheType.DISABLED
    ]:
        model_config.kv_cache_type = enum_val
        assert model_config.kv_cache_type == enum_val

    assert model_config.tokens_per_block == 64
    tokens_per_block = 1024
    model_config.tokens_per_block = tokens_per_block
    assert model_config.tokens_per_block == tokens_per_block

    assert model_config.quant_mode == _tb.QuantMode.none()
    model_config.quant_mode = _tb.QuantMode.int4_weights()
    assert model_config.quant_mode.has_int4_weights

    assert model_config.supports_inflight_batching

    assert model_config.max_batch_size == 0
    max_batch_size = 1000
    model_config.max_batch_size = max_batch_size
    assert model_config.max_batch_size == max_batch_size

    assert model_config.max_input_len == 0
    max_input_len = 2048
    model_config.max_input_len = max_input_len
    assert model_config.max_input_len == max_input_len

    assert model_config.max_num_tokens is None
    max_num_tokens = 10000
    model_config.max_num_tokens = max_num_tokens
    assert model_config.max_num_tokens == max_num_tokens

    assert not model_config.compute_context_logits
    model_config.compute_context_logits = True
    assert model_config.compute_context_logits

    assert not model_config.compute_generation_logits
    model_config.compute_generation_logits = True
    assert model_config.compute_generation_logits

    assert model_config.model_variant == _tb.GptModelVariant.GPT
    model_variant = _tb.GptModelVariant.GLM
    model_config.model_variant = model_variant
    assert model_config.model_variant == model_variant


def test_world_config():
    tensor_parallelism = 2
    pipeline_parallelism = 4
    context_parallelism = 1
    rank = 3
    gpus_per_node = 10
    world_config = _tb.WorldConfig(tensor_parallelism, pipeline_parallelism,
                                   context_parallelism, rank, gpus_per_node)
    assert world_config.tensor_parallelism == tensor_parallelism
    assert world_config.pipeline_parallelism == pipeline_parallelism
    assert world_config.context_parallelism == context_parallelism
    assert world_config.rank == rank
    assert world_config.gpus_per_node == gpus_per_node
    assert world_config.gpus_per_group == gpus_per_node
    assert world_config.size == tensor_parallelism * pipeline_parallelism * context_parallelism
    assert world_config.is_pipeline_parallel
    assert world_config.is_tensor_parallel
    assert world_config.device == rank % gpus_per_node
    assert world_config.pipeline_parallel_rank == rank // tensor_parallelism
    assert world_config.tensor_parallel_rank == rank % tensor_parallelism

    world_config = _tb.WorldConfig.mpi(gpus_per_node)
    assert world_config.tensor_parallelism == 1
    assert world_config.pipeline_parallelism == 1
    assert world_config.context_parallelism == 1
    assert world_config.gpus_per_node == gpus_per_node
    assert world_config.rank == 0

    gpus_per_group = gpus_per_node // 2
    device_ids = list(gpus_per_group + x for x in range(gpus_per_group))
    assert max(device_ids) < gpus_per_node
    world_config = _tb.WorldConfig(rank=rank,
                                   gpus_per_node=gpus_per_node,
                                   device_ids=device_ids)
    assert world_config.gpus_per_node == gpus_per_node
    assert world_config.gpus_per_group == gpus_per_group
    assert world_config.rank == rank
    assert world_config.device == rank + gpus_per_group


def test_sampling_config():
    beam_width = 12
    sampling_config = _tb.SamplingConfig(beam_width)
    assert sampling_config.beam_width == 12

    def check_empty_then_set(member, value):
        assert getattr(sampling_config, member) is None
        setattr(sampling_config, member, value)
        assert getattr(sampling_config, member) == value

    float_array = [1., 2., 3.]
    size_t_array = [1, 2, 3]
    check_empty_then_set("temperature", float_array)
    check_empty_then_set("min_length", size_t_array)
    check_empty_then_set("repetition_penalty", float_array)
    check_empty_then_set("presence_penalty", float_array)
    check_empty_then_set("frequency_penalty", float_array)
    check_empty_then_set("top_k", size_t_array)
    check_empty_then_set("top_p", float_array)
    check_empty_then_set("random_seed", size_t_array)
    check_empty_then_set("top_p_decay", float_array)
    check_empty_then_set("top_p_min", float_array)
    check_empty_then_set("top_p_reset_ids", size_t_array)
    check_empty_then_set("beam_search_diversity_rate", float_array)
    check_empty_then_set("length_penalty", float_array)
    check_empty_then_set("early_stopping", size_t_array)


def test_gpt_json_config():
    model_config = {
        "vocab_size": 1000,
        "num_layers": 18,  # >= attn + rnn
        "num_attention_layers": 12,
        "num_rnn_layers": 2,
        "num_heads": 4,
        "hidden_size": 512,
        "data_type": _tb.DataType.FLOAT,
    }
    trt_model_config = _tb.ModelConfig(**model_config)
    json_config = {
        "name": "gpt",
        "version": "none",
        "precision": "float32",
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "context_parallelism": 1,
        "gpus_per_node": 8,
        "model_config": trt_model_config
    }

    gpt_json_config = _tb.GptJsonConfig(**json_config)

    def check_properties(the_object, properties, model_config):
        for property, value in properties.items():
            if isinstance(value, _tb.ModelConfig):
                object_config = getattr(the_object, property)
                for subproperty, subvalue in model_config.items():
                    member = getattr(object_config, subproperty)
                    if callable(member):
                        member = member()
                    assert member == subvalue
            else:
                assert getattr(the_object, property) == value

    check_properties(gpt_json_config, json_config, model_config)

    assert gpt_json_config.runtime_defaults is None

    json_dict = {
        "builder_config": {
            "name": json_config["name"],
            "vocab_size": model_config["vocab_size"],
            "num_layers": model_config["num_attention_layers"],
            "num_heads": model_config["num_heads"],
            "hidden_size": model_config["hidden_size"],
            "precision": json_config["precision"],
            "tensor_parallel": json_config["tensor_parallelism"],
            "pipeline_parallel": json_config["pipeline_parallelism"],
            "context_parallel": json_config["context_parallelism"],
        },
        "plugin_config": {
            "paged_kv_cache": False,
            "tokens_per_block": 0,
            "gpt_attention_plugin": False,
            "remove_input_padding": False,
            "context_fmha": False,
            "use_paged_context_fmha": False,
            "lora_plugin": False,
        }
    }

    gpt_json_config = _tb.GptJsonConfig.parse(json.dumps(json_dict))

    with tempfile.NamedTemporaryFile("w", delete=False) as fp:
        json.dump(json_dict, fp)
        fp.close()

        gpt_json_config = _tb.GptJsonConfig.parse_file(Path(fp.name))
        Path(fp.name).unlink()

    rank = 3
    gpus_per_node = 10
    world_config = _tb.WorldConfig(json_config["tensor_parallelism"],
                                   json_config["pipeline_parallelism"],
                                   json_config["context_parallelism"], rank,
                                   gpus_per_node)

    assert gpt_json_config.engine_filename(
        world_config) == json_config["name"] + "_float32_tp1_rank3.engine"
    assert gpt_json_config.engine_filename(
        world_config, "llama") == "llama_float32_tp1_rank3.engine"

    def parse_runtime_defaults(defaults_dict: dict | None = None):
        config = _tb.GptJsonConfig.parse(
            json.dumps({
                "version": "some.version",
                "build_config": {
                    "plugin_config": json_dict["plugin_config"],
                    "lora_config": {},
                },
                "pretrained_config": {
                    **json_dict["builder_config"],
                    "architecture": "LlamaForCausalLM",
                    "mapping": Mapping().to_dict(),
                    "dtype": "bfloat16",
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "quantization": {},
                    "runtime_defaults": defaults_dict,
                },
            }))
        return config.runtime_defaults

    strict_keys = False  # GptJsonConfig is written in cpp, and there is currently no nice way to throw on extra keys
    assert_runtime_defaults_are_parsed_correctly(parse_runtime_defaults,
                                                 strict_keys=strict_keys)


def test_llm_request():
    beam_width = 2
    sampling_config = _tb.SamplingConfig(beam_width)
    kwargs = {
        "request_id": 0,
        "max_new_tokens": 5,
        "sampling_config": sampling_config,
        "input_tokens": [0, 1, 2],
        "position_ids": [0, 1, 2],
        "is_streaming": True,
        "pad_id": 99,
        "end_id": 100,
        "prompt_embedding_table": torch.tensor((10, 10)),
        "prompt_vocab_size": 2,
        "embedding_bias": torch.tensor((10, 10)),
        "stop_words_list": torch.tensor((10, 10)),
        "bad_words_list": torch.tensor((10, 10)),
        "return_log_probs": True,
        "return_context_logits": False,
        "return_generation_logits": False
    }
    llm_request = _tb.internal.batch_manager.LlmRequest(**kwargs)

    assert llm_request.request_id == 0
    assert llm_request.prompt_len == 3
    assert llm_request.sampling_config.beam_width == sampling_config.beam_width
    assert llm_request.streaming
    assert llm_request.pad_id == 99
    assert llm_request.end_id == 100
    assert llm_request.seq_slot is None
    assert torch.equal(llm_request.prompt_embedding_table,
                       kwargs["prompt_embedding_table"])
    assert llm_request.prompt_vocab_size == 2
    assert torch.equal(llm_request.embedding_bias, kwargs["embedding_bias"])
    assert torch.equal(llm_request.stop_words_list, kwargs["stop_words_list"])
    assert torch.equal(llm_request.bad_words_list, kwargs["bad_words_list"])

    assert llm_request.get_num_tokens(0) == 3
    assert llm_request.max_beam_num_tokens == 3
    assert llm_request.get_token(1, 2) == 2
    assert llm_request.get_tokens(1) == [0, 1, 2]
    assert llm_request.max_num_generated_tokens == 0
    assert llm_request.position_ids == [0, 1, 2]

    llm_request.add_new_token(42, 0)
    assert llm_request.get_token(0, 3) == 42

    llm_request.add_new_tokens([43, 44])
    assert llm_request.get_token(0, 4) == 43
    assert llm_request.get_token(1, 3) == 44

    llm_request.set_generated_tokens([[10, 11], [12, 13]])
    assert llm_request.get_tokens(0) == [0, 1, 2, 10, 11]
    assert llm_request.max_num_generated_tokens == 2

    llm_request.pause(0)
    assert llm_request.state == _tb.LlmRequestState.CONTEXT_INIT

    llm_request.max_sent_token_len = 1
    assert llm_request.max_sent_token_len == 1

    assert llm_request.return_log_probs
    llm_request.set_log_probs([0.1], 0)
    llm_request.set_log_probs([0.2], 1)
    assert np.allclose(llm_request.get_log_probs(0), np.array([0.1]))
    assert np.allclose(llm_request.log_probs, np.array([[0.1], [0.2]]))

    llm_request.set_return_encoder_output(True)
    assert llm_request.get_return_encoder_output()
    llm_request.set_return_encoder_output(False)
    assert not llm_request.get_return_encoder_output()

    assert np.allclose(llm_request.priority(), 0.5)
    llm_request.set_priority(1.0)
    assert np.allclose(llm_request.priority(), 1.0)
    llm_request.set_priority(0.0)
    assert np.allclose(llm_request.priority(), 0.0)

    llm_request.set_cum_log_prob(0.1, 0)
    llm_request.set_cum_log_prob(0.2, 1)
    assert np.allclose(llm_request.cum_log_probs, np.array([0.1, 0.2]))

    assert llm_request.orig_prompt_len == 3

    assert not llm_request.draft_tokens
    llm_request.draft_tokens = [1, 2, 3]
    assert llm_request.draft_tokens == [1, 2, 3]

    logits = torch.tensor([-5, -6 - 7], dtype=torch.float)
    llm_request.draft_logits = logits
    assert torch.equal(llm_request.draft_logits, logits)


def test_Mpicomm():
    size1 = _tb.MpiComm.size()
    rank1 = _tb.MpiComm.rank()

    session_size = (size1 + 1) // 2
    session_color = rank1 // session_size
    session_rank = rank1 % session_size
    _tb.MpiComm.split(session_color, session_rank)

    rank2 = _tb.MpiComm.rank()
    size2 = _tb.MpiComm.size()

    assert rank2 == session_rank
    assert size2 == session_size


def test_SamplingConfig_pickle():
    config = _tb.SamplingConfig()
    config.beam_width = 5
    config.num_return_sequences = 1
    config.top_k = [1, 2]
    config.top_p = [0.1, 0.2]
    config.top_p_min = [1.0, 2.0]
    config.top_p_reset_ids = [1, 2]
    config.top_p_decay = [1.0, 2.0]
    config.random_seed = [1, 2]
    config.temperature = [1.0, 2.0]
    config.min_length = [32, 64]
    config.beam_search_diversity_rate = [1.0, 2.0]
    config.repetition_penalty = [1.0, 2.0]
    config.presence_penalty = [1.0, 2.0]
    config.frequency_penalty = [1.0, 2.0]
    config.length_penalty = [1.0, 2.0]
    config.early_stopping = [1, 2]
    config.no_repeat_ngram_size = [4, 6]
    config.num_return_sequences = 1
    config.min_p = [0.5, 0.5]
    config.beam_width_array = [[2, 3, 4, 5]]

    config1 = pickle.loads(pickle.dumps(config))
    assert config1 == config


def test_KvCache_events_binding():
    stream = torch.cuda.Stream()
    max_sequence_length = 10
    kwargs = {
        'num_kv_heads_per_layer': [1, 1],
        'size_per_head':
        128,
        'tokens_per_block':
        64,
        'blocks_per_window': {
            max_sequence_length: (1000, 10000)
        },
        'max_num_sequences':
        1,
        'max_beam_width':
        1,
        'max_attention_window_vec': [max_sequence_length],
        'temp_attention_window_inputs':
        None,
        'dtype':
        _tb.DataType.FLOAT,
        'sink_token_length':
        0,
        'stream':
        stream.cuda_stream,
        'max_sequence_length':
        max_sequence_length,
        'enable_block_reuse':
        True,
        'onboard_blocks':
        False,
        'cache_type':
        _tb.internal.batch_manager.CacheType.SELF,
        'event_manager':
        _tb.internal.batch_manager.KVCacheEventManager(
            max_kv_event_entries=1024)
    }

    kv_cache_manager = _tb.internal.batch_manager.KVCacheManager(**kwargs)
    kv_cache_manager.flush_iteration_events()
    time.sleep(0.0001)
    events = kv_cache_manager.get_latest_events()
    assert len(events) == 1
    assert isinstance(events[0], _tb.executor.kv_cache.KVCacheEvent)
    assert events[0].event_id == 0 and isinstance(
        events[0].data, _tb.executor.kv_cache.KVCacheCreatedData)

    del stream
    torch.cuda.empty_cache()


def test_ReqIdsSet_pickle():
    ids = _tb.internal.batch_manager.ReqIdsSet()
    ids1 = pickle.loads(pickle.dumps(ids))
    assert ids1 == ids

    ids.insert(0)
    ids.insert(1)
    ids.insert(2)
    ids1 = pickle.loads(pickle.dumps(ids))
    assert ids1 == ids

    ids1.erase(0)
    ids2 = _tb.internal.batch_manager.ReqIdsSet()
    ids2.insert(1)
    ids2.insert(2)
    assert ids1 == ids2
