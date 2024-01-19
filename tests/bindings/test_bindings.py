import inspect
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

import tensorrt_llm.bindings as _tb


def test_generation_output():
    ids = torch.ones(1)
    lengths = torch.ones(2)
    gen_output = _tb.GenerationOutput(ids, lengths)
    assert torch.equal(gen_output.ids, ids)
    assert torch.equal(gen_output.lengths, lengths)

    assert gen_output.log_probs is None
    log_probs = torch.ones(1)
    gen_output.log_probs = log_probs
    assert gen_output.log_probs == log_probs

    assert gen_output.context_logits is None
    torch.ones(1)
    gen_output.context_logits = log_probs
    assert gen_output.context_logits == log_probs


def test_generation_input():
    end_id = 42
    pad_id = 13
    ids = torch.ones(1)
    lengths = torch.ones(2)
    packed = True
    gen_input = _tb.GenerationInput(end_id, pad_id, ids, lengths, packed)
    assert gen_input.end_id == end_id
    assert gen_input.pad_id == pad_id
    assert torch.equal(gen_input.ids, ids)
    assert torch.equal(gen_input.lengths, lengths)
    assert gen_input.packed == packed

    assert gen_input.max_new_tokens is None
    max_new_tokens = 100
    gen_input.max_new_tokens = max_new_tokens
    assert gen_input.max_new_tokens == max_new_tokens

    assert gen_input.embedding_bias is None
    embedding_bias = torch.ones(3)
    gen_input.embedding_bias = embedding_bias
    assert torch.equal(gen_input.embedding_bias, embedding_bias)

    assert gen_input.prompt_tuning_params.embedding_table is None
    assert gen_input.prompt_tuning_params.tasks is None
    assert gen_input.prompt_tuning_params.vocab_size is None

    embedding_table = torch.ones(3)
    tasks = torch.ones(2)
    vocab_size = torch.ones(1)
    prompt_tuning_params = _tb.PromptTuningParams(
        embedding_table=embedding_table, tasks=tasks, vocab_size=vocab_size)
    assert len(prompt_tuning_params.prompt_tuning_enabled) == 0
    prompt_tuning_enabled = [True, False]
    prompt_tuning_params.prompt_tuning_enabled = prompt_tuning_enabled
    assert len(prompt_tuning_params.prompt_tuning_enabled) == 2
    assert prompt_tuning_params.prompt_tuning_enabled == prompt_tuning_enabled
    gen_input.prompt_tuning_params = prompt_tuning_params
    assert gen_input.prompt_tuning_params is not None
    assert torch.equal(gen_input.prompt_tuning_params.embedding_table,
                       embedding_table)
    assert torch.equal(gen_input.prompt_tuning_params.tasks, tasks)
    assert torch.equal(gen_input.prompt_tuning_params.vocab_size, vocab_size)
    assert gen_input.prompt_tuning_params.prompt_tuning_enabled == prompt_tuning_enabled


def test_gpt_session_config():
    kv_cache_config = _tb.KvCacheConfig()
    assert kv_cache_config.max_tokens is None
    max_tokens = 13
    kv_cache_config.max_tokens = max_tokens
    assert kv_cache_config.max_tokens == max_tokens
    assert kv_cache_config.free_gpu_memory_fraction is None
    free_gpu_memory_fraction = 0.5
    kv_cache_config.free_gpu_memory_fraction = free_gpu_memory_fraction
    assert kv_cache_config.free_gpu_memory_fraction == free_gpu_memory_fraction

    max_batch_size = 1000
    max_beam_width = 64
    max_sequence_length = 1 << 20
    gpt_session_config = _tb.GptSessionConfig(max_batch_size, max_beam_width,
                                              max_sequence_length)
    assert gpt_session_config.max_batch_size == max_batch_size
    assert gpt_session_config.max_beam_width == max_beam_width
    assert gpt_session_config.max_sequence_length == max_sequence_length

    assert gpt_session_config.kv_cache_config is not None
    assert gpt_session_config.kv_cache_config.max_tokens is None
    assert gpt_session_config.kv_cache_config.free_gpu_memory_fraction is None
    gpt_session_config.kv_cache_config = kv_cache_config
    assert gpt_session_config.kv_cache_config.max_tokens == max_tokens
    assert gpt_session_config.kv_cache_config.free_gpu_memory_fraction == free_gpu_memory_fraction
    gpt_session_config.kv_cache_config.max_tokens = None
    assert gpt_session_config.kv_cache_config.max_tokens is None
    gpt_session_config.kv_cache_config.free_gpu_memory_fraction = None
    assert gpt_session_config.kv_cache_config.free_gpu_memory_fraction is None

    assert not gpt_session_config.decoder_per_request
    gpt_session_config.decoder_per_request = True
    assert gpt_session_config.decoder_per_request

    assert not gpt_session_config.cuda_graph_mode
    gpt_session_config.cuda_graph_mode = True
    assert gpt_session_config.cuda_graph_mode

    assert gpt_session_config.ctx_micro_batch_size is None
    ctx_micro_batch_size = 10
    gpt_session_config.ctx_micro_batch_size = ctx_micro_batch_size
    assert gpt_session_config.ctx_micro_batch_size == ctx_micro_batch_size

    assert gpt_session_config.gen_micro_batch_size is None
    gen_micro_batch_size = 20
    gpt_session_config.gen_micro_batch_size = gen_micro_batch_size
    assert gpt_session_config.gen_micro_batch_size == gen_micro_batch_size


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


def test_gpt_model_config():
    vocab_size = 10000
    num_layers = 12
    num_heads = 16
    hidden_size = 768
    data_type = _tb.DataType.FLOAT
    gpt_model_config = _tb.GptModelConfig(vocab_size, num_layers, num_heads,
                                          hidden_size, data_type)
    assert gpt_model_config.vocab_size == vocab_size
    assert gpt_model_config.num_layers() == num_layers
    assert gpt_model_config.num_heads == num_heads
    assert gpt_model_config.hidden_size == hidden_size
    assert gpt_model_config.data_type == data_type

    assert gpt_model_config.vocab_size_padded(1) is not None
    assert gpt_model_config.size_per_head == hidden_size // num_heads

    assert gpt_model_config.num_kv_heads == num_heads
    num_kv_heads = 1
    gpt_model_config.num_kv_heads = num_kv_heads
    assert gpt_model_config.num_kv_heads == num_kv_heads

    assert not gpt_model_config.use_gpt_attention_plugin
    gpt_model_config.use_gpt_attention_plugin = True
    assert gpt_model_config.use_gpt_attention_plugin

    assert not gpt_model_config.use_packed_input
    gpt_model_config.use_packed_input = True
    assert gpt_model_config.use_packed_input

    assert not gpt_model_config.use_paged_kv_cache
    gpt_model_config.use_paged_kv_cache = True
    assert gpt_model_config.use_paged_kv_cache

    assert gpt_model_config.tokens_per_block == 64
    tokens_per_block = 1024
    gpt_model_config.tokens_per_block = tokens_per_block
    assert gpt_model_config.tokens_per_block == tokens_per_block

    assert gpt_model_config.quant_mode == _tb.QuantMode.none()
    gpt_model_config.quant_mode = _tb.QuantMode.int4_weights()
    assert gpt_model_config.quant_mode.has_int4_weights

    assert gpt_model_config.supports_inflight_batching

    assert gpt_model_config.max_batch_size == 0
    max_batch_size = 1000
    gpt_model_config.max_batch_size = max_batch_size
    assert gpt_model_config.max_batch_size == max_batch_size

    assert gpt_model_config.max_input_len == 0
    max_input_len = 2048
    gpt_model_config.max_input_len = max_input_len
    assert gpt_model_config.max_input_len == max_input_len

    assert gpt_model_config.max_num_tokens is None
    max_num_tokens = 10000
    gpt_model_config.max_num_tokens = max_num_tokens
    assert gpt_model_config.max_num_tokens == max_num_tokens

    assert not gpt_model_config.compute_context_logits
    gpt_model_config.compute_context_logits = True
    assert gpt_model_config.compute_context_logits

    assert not gpt_model_config.compute_generation_logits
    gpt_model_config.compute_generation_logits = True
    assert gpt_model_config.compute_generation_logits

    assert gpt_model_config.model_variant == _tb.GptModelVariant.GPT
    model_variant = _tb.GptModelVariant.GLM
    gpt_model_config.model_variant = model_variant
    assert gpt_model_config.model_variant == model_variant

    assert not gpt_model_config.use_custom_all_reduce
    gpt_model_config.use_custom_all_reduce = True
    assert gpt_model_config.use_custom_all_reduce


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


def test_gpt_json_config():
    model_config = {
        "vocab_size": 1000,
        "num_layers": 12,
        "num_heads": 4,
        "hidden_size": 512,
        "data_type": _tb.DataType.FLOAT,
    }
    gpt_model_config = _tb.GptModelConfig(**model_config)
    json_config = {
        "name": "gpt",
        "version": "none",
        "precision": "float32",
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "model_config": gpt_model_config
    }

    gpt_json_config = _tb.GptJsonConfig(**json_config)

    def check_properties(the_object, properties, model_config):
        for property, value in properties.items():
            if isinstance(value, _tb.GptModelConfig):
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
            "num_layers": model_config["num_layers"],
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
            "use_custom_all_reduce": False,
            "use_context_fmha_for_generation": False,
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


def test_gpt_session():
    members = {name: tpe for (name, tpe) in inspect.getmembers(_tb.GptSession)}
    assert isinstance(members["model_config"], property)
    assert isinstance(members["world_config"], property)
    assert isinstance(members["device"], property)
    assert "generate" in members


def test_llm_request():
    beam_width = 2
    sampling_config = _tb.SamplingConfig(beam_width)
    kwargs = {
        "request_id": 0,
        "max_new_tokens": 5,
        "sampling_config": sampling_config,
        "input_tokens": [0, 1, 2],
        "is_streaming": True,
        "pad_id": 99,
        "end_id": 100,
        "prompt_embedding_table": torch.tensor((10, 10)),
        "prompt_vocab_size": 2,
        "embedding_bias": torch.tensor((10, 10)),
        "stop_words_list": torch.tensor((10, 10)),
        "bad_words_list": torch.tensor((10, 10)),
        "return_log_probs": True
    }
    llm_request = _tb.LlmRequest(**kwargs)

    assert llm_request.request_id == 0
    assert llm_request.prompt_len == 3
    assert llm_request.sampling_config.beam_width == sampling_config.beam_width
    assert llm_request.is_streaming
    assert llm_request.pad_id == 99
    assert llm_request.end_id == 100
    assert llm_request.seq_slot == -1  # seq_slot is still uninitialized
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

    llm_request.add_new_token(42, 0)
    assert llm_request.get_token(0, 3) == 42

    llm_request.add_new_tokens([43, 44])
    assert llm_request.get_token(0, 4) == 43
    assert llm_request.get_token(1, 3) == 44

    llm_request.set_generated_tokens([[10, 11], [12, 13]])
    assert llm_request.get_tokens(0) == [0, 1, 2, 10, 11]
    assert llm_request.max_num_generated_tokens == 2

    llm_request.pause(0)
    assert llm_request.state == _tb.LlmRequestState.REQUEST_STATE_CONTEXT_INIT

    llm_request.max_sent_token_pos = 1
    assert llm_request.max_sent_token_pos == 1

    assert llm_request.return_log_probs
    llm_request.set_log_probs([0.1], 0)
    llm_request.set_log_probs([0.2], 1)
    assert np.allclose(llm_request.get_log_probs(0), np.array([0.1]))
    assert np.allclose(llm_request.log_probs, np.array([[0.1], [0.2]]))

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
    vm = {_tb.tensor_names.INPUT_IDS: input_ids}
    ir = _tb.InferenceRequest(42, vm)
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

    serialized = pickle.dumps(ir)
    deserialized = pickle.loads(serialized)

    assert isinstance(deserialized, _tb.InferenceRequest)
    assert deserialized.request_id == ir.request_id
    assert deserialized.is_streaming == ir.is_streaming
    assert torch.equal(deserialized.input_ids, ir.input_ids)


def test_trt_gpt_model_optional_params():
    opt_params = _tb.TrtGptModelOptionalParams()

    kv_cache_config = _tb.KvCacheConfig(10, 10, 0, 0.5, False)
    opt_params.kv_cache_config = kv_cache_config
    assert opt_params.kv_cache_config.free_gpu_memory_fraction == kv_cache_config.free_gpu_memory_fraction

    opt_params.max_num_sequences = 10
    assert opt_params.max_num_sequences == 10

    opt_params.enable_trt_overlap = True
    assert opt_params.enable_trt_overlap

    assert opt_params.device_ids is None
    opt_params.device_ids = [0, 1]
    assert opt_params.device_ids == [0, 1]
