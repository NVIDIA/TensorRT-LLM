import json
import pickle
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

import tensorrt_llm.bindings as _tb


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
                                                True, True, True)
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
    model_config = _tb.ModelConfig(vocab_size, num_attention_layers,
                                   num_rnn_layers, num_heads, hidden_size,
                                   data_type)
    assert model_config.vocab_size == vocab_size
    assert model_config.num_attention_layers() == num_attention_layers
    assert model_config.num_rnn_layers() == num_rnn_layers
    assert model_config.num_heads == num_heads
    assert model_config.hidden_size == hidden_size
    assert model_config.data_type == data_type

    assert model_config.vocab_size_padded(1) is not None
    assert model_config.size_per_head == hidden_size // num_heads

    assert model_config.num_kv_heads == num_heads
    num_kv_heads = 1
    model_config.num_kv_heads = num_kv_heads
    assert model_config.num_kv_heads == num_kv_heads

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
    rank = 3
    gpus_per_node = 10
    world_config = _tb.WorldConfig(tensor_parallelism, pipeline_parallelism,
                                   rank, gpus_per_node)
    assert world_config.tensor_parallelism == tensor_parallelism
    assert world_config.pipeline_parallelism == pipeline_parallelism
    assert world_config.rank == rank
    assert world_config.gpus_per_node == gpus_per_node
    assert world_config.gpus_per_group == gpus_per_node
    assert world_config.size == tensor_parallelism * pipeline_parallelism
    assert world_config.is_pipeline_parallel
    assert world_config.is_tensor_parallel
    assert world_config.device == rank % gpus_per_node
    assert world_config.pipeline_parallel_rank == rank // tensor_parallelism
    assert world_config.tensor_parallel_rank == rank % tensor_parallelism

    world_config = _tb.WorldConfig.mpi(gpus_per_node)
    assert world_config.tensor_parallelism == 1
    assert world_config.pipeline_parallelism == 1
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
                                   json_config["pipeline_parallelism"], rank,
                                   gpus_per_node)

    assert gpt_json_config.engine_filename(
        world_config) == json_config["name"] + "_float32_tp1_rank3.engine"
    assert gpt_json_config.engine_filename(
        world_config, "llama") == "llama_float32_tp1_rank3.engine"


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
    llm_request = _tb.LlmRequest(**kwargs)

    assert llm_request.request_id == 0
    assert llm_request.prompt_len == 3
    assert llm_request.sampling_config.beam_width == sampling_config.beam_width
    assert llm_request.is_streaming
    assert llm_request.pad_id == 99
    assert llm_request.end_id == 100
    assert llm_request.seq_slot == None
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


def test_inference_request():
    input_ids = torch.tensor((10, 10))

    def logits_post_processor(req_id: int, logits: torch.Tensor,
                              ids: List[List[int]], client_id: Optional[int]):
        del req_id, ids

    ir = _tb.InferenceRequest(42, logits_post_processor)
    setattr(ir, _tb.tensor_names.INPUT_IDS, input_ids)

    ir.position_ids = torch.tensor((0, 1), dtype=torch.int32)
    torch.equal(ir.position_ids, torch.tensor((0, 1), dtype=torch.int32))

    assert ir.request_id == 42
    assert ir.input_ids is not None
    assert torch.equal(ir.input_ids, input_ids)

    assert not ir.is_streaming
    ir.is_streaming = True
    assert ir.is_streaming

    data_tensor = torch.tensor((5, 5))

    assert ir.draft_input_ids is None
    ir.draft_input_ids = data_tensor
    assert torch.equal(ir.draft_input_ids, data_tensor)

    assert ir.draft_logits is None
    ir.draft_logits = data_tensor
    assert torch.equal(ir.draft_logits, data_tensor)

    assert ir.bad_words_list is None
    ir.bad_words_list = data_tensor
    assert torch.equal(ir.bad_words_list, data_tensor)

    assert ir.beam_width is None
    ir.beam_width = data_tensor
    assert torch.equal(ir.beam_width, data_tensor)

    assert ir.embedding_bias is None
    ir.embedding_bias = data_tensor
    assert torch.equal(ir.embedding_bias, data_tensor)

    assert ir.end_id is None
    ir.end_id = data_tensor
    assert torch.equal(ir.end_id, data_tensor)

    assert ir.length_penalty is None
    ir.length_penalty = data_tensor
    assert torch.equal(ir.length_penalty, data_tensor)

    assert ir.early_stopping is None
    ir.early_stopping = data_tensor
    assert torch.equal(ir.early_stopping, data_tensor)

    assert ir.max_new_tokens is None
    ir.max_new_tokens = data_tensor
    assert torch.equal(ir.max_new_tokens, data_tensor)

    assert ir.min_length is None
    ir.min_length = data_tensor
    assert torch.equal(ir.min_length, data_tensor)

    assert ir.pad_id is None
    ir.pad_id = data_tensor
    assert torch.equal(ir.pad_id, data_tensor)

    assert ir.presence_penalty is None
    ir.presence_penalty = data_tensor
    assert torch.equal(ir.presence_penalty, data_tensor)

    assert ir.frequency_penalty is None
    ir.frequency_penalty = data_tensor
    assert torch.equal(ir.frequency_penalty, data_tensor)

    assert ir.prompt_embedding_table is None
    ir.prompt_embedding_table = data_tensor
    assert torch.equal(ir.prompt_embedding_table, data_tensor)

    assert ir.prompt_vocab_size is None
    ir.prompt_vocab_size = data_tensor
    assert torch.equal(ir.prompt_vocab_size, data_tensor)

    assert ir.lora_weights is None
    ir.lora_weights = data_tensor
    assert torch.equal(ir.lora_weights, data_tensor)

    assert ir.lora_config is None
    ir.lora_config = data_tensor
    assert torch.equal(ir.lora_config, data_tensor)

    assert ir.random_seed is None
    ir.random_seed = data_tensor
    assert torch.equal(ir.random_seed, data_tensor)

    assert ir.repetition_penalty is None
    ir.repetition_penalty = data_tensor
    assert torch.equal(ir.repetition_penalty, data_tensor)

    assert ir.return_log_probs is None
    ir.return_log_probs = data_tensor
    assert torch.equal(ir.return_log_probs, data_tensor)

    assert ir.runtime_top_k is None
    ir.runtime_top_k = data_tensor
    assert torch.equal(ir.runtime_top_k, data_tensor)

    assert ir.runtime_top_p is None
    ir.runtime_top_p = data_tensor
    assert torch.equal(ir.runtime_top_p, data_tensor)

    assert ir.stop_words_list is None
    ir.stop_words_list = data_tensor
    assert torch.equal(ir.stop_words_list, data_tensor)

    assert ir.temperature is None
    ir.temperature = data_tensor
    assert torch.equal(ir.temperature, data_tensor)

    ir.logits_post_processor = None
    serialized = pickle.dumps(ir)
    deserialized = pickle.loads(serialized)

    assert isinstance(deserialized, _tb.InferenceRequest)
    assert deserialized.request_id == ir.request_id
    assert deserialized.is_streaming == ir.is_streaming
    assert torch.equal(deserialized.input_ids, ir.input_ids)


def test_trt_gpt_model_optional_params():
    opt_params = _tb.TrtGptModelOptionalParams()

    kv_cache_config = _tb.KvCacheConfig(10, [10], 0, 0.5, False)
    opt_params.kv_cache_config = kv_cache_config
    assert opt_params.kv_cache_config.free_gpu_memory_fraction == kv_cache_config.free_gpu_memory_fraction

    assert not opt_params.enable_trt_overlap
    opt_params.enable_trt_overlap = True
    assert opt_params.enable_trt_overlap

    assert opt_params.device_ids is None
    opt_params.device_ids = [0, 1]
    assert opt_params.device_ids == [0, 1]

    assert not opt_params.enable_chunked_context
    opt_params.enable_chunked_context = True
    assert opt_params.enable_chunked_context

    assert opt_params.normalize_log_probs
    opt_params.normalize_log_probs = False
    assert not opt_params.normalize_log_probs

    assert not opt_params.decoding_config.decoding_mode
    opt_params.decoding_config.decoding_mode = _tb.executor.DecodingMode.TopKTopP(
    )
    assert opt_params.decoding_config.decoding_mode.isTopKandTopP()

    assert not opt_params.max_beam_width
    opt_params.max_beam_width = 4
    assert opt_params.max_beam_width == 4

    assert opt_params.scheduler_config.capacity_scheduler_policy == _tb.executor.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    assert opt_params.scheduler_config.context_chunking_policy == None
    opt_params.scheduler_config = _tb.executor.SchedulerConfig(
        _tb.executor.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        _tb.executor.ContextChunkingPolicy.FIRST_COME_FIRST_SERVED)
    assert opt_params.scheduler_config.capacity_scheduler_policy == _tb.executor.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    assert opt_params.scheduler_config.context_chunking_policy == _tb.executor.ContextChunkingPolicy.FIRST_COME_FIRST_SERVED


def test_trt_gpt_model_optional_params_ctor():
    kv_cache_config = _tb.KvCacheConfig(10, [10], 0, 0.5, False)
    enable_trt_overlap = True
    device_ids = [0, 1]
    normalize_log_probs = False
    enable_chunked_context = True
    peft_cache_manager_config = _tb.PeftCacheManagerConfig()

    opt_params = _tb.TrtGptModelOptionalParams(kv_cache_config,
                                               enable_trt_overlap, device_ids,
                                               normalize_log_probs,
                                               enable_chunked_context,
                                               peft_cache_manager_config)
    assert opt_params.kv_cache_config.free_gpu_memory_fraction == kv_cache_config.free_gpu_memory_fraction
    assert opt_params.enable_trt_overlap
    assert opt_params.device_ids == device_ids
    assert opt_params.normalize_log_probs == normalize_log_probs
    assert opt_params.enable_chunked_context == enable_chunked_context
    assert opt_params.gpu_weights_percent == 1


def test_KvCacheConfig_pickle():
    cache = _tb.KvCacheConfig(free_gpu_memory_fraction=0.4)
    cache1 = pickle.dumps(cache)
    cache2 = pickle.loads(cache1)

    assert cache2 == cache


def test_TrtGptModelOptionalParams_pickle():
    cache = _tb.KvCacheConfig(free_gpu_memory_fraction=0.4)
    params1 = _tb.TrtGptModelOptionalParams(
        kv_cache_config=cache,
        enable_trt_overlap=True,
    )
    params1.enable_chunked_context = True
    params2 = pickle.loads(pickle.dumps(params1))

    assert params2 == params1

    params1 = _tb.TrtGptModelOptionalParams()
    params2 = pickle.loads(pickle.dumps(params1))

    assert params2 == params1


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
    config.beam_width = 2
    config.temperature = [1.0, 2.0]
    config.top_k = [1, 2]
    config.top_p = [0.1, 0.2]
    config.random_seed = [1, 2]
    config.repetition_penalty = [1.0, 2.0]
    config.presence_penalty = [1.0, 2.0]
    config.frequency_penalty = [1.0, 2.0]
    config.length_penalty = [1.0, 2.0]
    config.early_stopping = [1, 2]
    config.top_p_decay = [1.0, 2.0]
    config.top_p_min = [1.0, 2.0]
    config.top_p_reset_ids = [1, 2]
    config.beam_search_diversity_rate = [1.0, 2.0]

    config1 = pickle.loads(pickle.dumps(config))

    assert config1 == config
