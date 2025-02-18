import random
from collections.abc import Iterable

import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm.bindings.executor import ExecutorConfig
from tensorrt_llm.mapping import Mapping

from .llm_request import LlmRequest
from .resource_manager import KVCacheManager, ResourceManager
from .scheduler import ScheduledRequests


def is_mla(config):
    if hasattr(config, "kv_lora_rank"):
        assert hasattr(
            config, "qk_rope_head_dim"
        ), "both of kv_lora_rank and qk_rope_head_dim are required."
        return True
    return False


def cal_max_tokens(peak_memory, total_gpu_memory, fraction, model_config,
                   mapping: Mapping):
    mem_per_token = 2
    quant_config = model_config.quant_config
    if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache():
        mem_per_token = 1

    config = model_config.pretrained_config

    num_key_value_heads = getattr(config, 'num_key_value_heads',
                                  config.num_attention_heads)
    if isinstance(num_key_value_heads, Iterable):
        num_key_value_heads = sum(num_key_value_heads) / len(
            num_key_value_heads)

    mla = is_mla(config)
    tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size

    kv_factor = 2
    if mla:
        # MLA has kv_lora_rank and qk_rope_head_dim
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim
        kv_factor = 1
    else:
        head_dim = (config.hidden_size * num_key_value_heads /
                    config.num_attention_heads / tp_size)

    mem_per_token *= config.num_hidden_layers * head_dim
    # K and V
    mem_per_token *= kv_factor

    if fraction is None:
        fraction = 0.9

    available_kv_mem = (total_gpu_memory - peak_memory) * fraction
    max_tokens = int((available_kv_mem) // mem_per_token)
    max_tokens = max(max_tokens, 0)
    return max_tokens


def _create_dummy_context(req_id: int, input_len: int, vocab_size: int):
    # To avoid recursive dependency during init tensorrt_llm.executor
    from tensorrt_llm import SamplingParams
    sampling_params = SamplingParams()

    result = LlmRequest(
        request_id=req_id,
        max_new_tokens=1,
        input_tokens=[
            random.randint(0, vocab_size - 1) for _ in range(input_len)
        ],
        position_ids=list(range(input_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
    )
    result.paged_kv_block_ids = []
    return result


def create_dummy_context_request(
        req_id: int, input_len: int, vocab_size: int,
        kv_cache_manager: KVCacheManager) -> LlmRequest:

    requests = [_create_dummy_context(req_id, input_len, vocab_size)]
    result = ScheduledRequests()
    result.generation_requests = []
    result.context_requests = requests

    if kv_cache_manager and kv_cache_manager.resource_managers[
            'kv_cache_manager']:
        for req in requests:
            kv_cache_manager.resource_managers[
                'kv_cache_manager'].add_padding_request(req)

    return result


def estimate_max_kv_cache_tokens(model_engine: PyTorchModelEngine,
                                 executor_config: ExecutorConfig,
                                 mapping: Mapping):
    vocab_size = model_engine.model.model_config.pretrained_config.vocab_size
    max_num_tokens = executor_config.max_num_tokens
    fraction = executor_config.kv_cache_config.free_gpu_memory_fraction
    kv_cache_max_tokens = executor_config.kv_cache_config.max_tokens
    # todo: to support max token evaluation for cp mode
    if 'cp_type' not in mapping.cp_config:
        resource_manager = ResourceManager({'kv_cache_manager': None})
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        req = create_dummy_context_request(0, max_num_tokens, vocab_size,
                                           resource_manager)
        model_engine.forward(req, resource_manager)
        torch.cuda.synchronize()
        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        end, total_gpu_memory = torch.cuda.mem_get_info()

        torch.cuda.empty_cache()
        torch_used_bytes = torch.cuda.memory_stats(
        )["allocated_bytes.all.current"]
        total_used_bytes = total_gpu_memory - end
        extra_cost = max(total_used_bytes - torch_used_bytes, 0)
        peak_memory += extra_cost
        kv_cache_max_tokens = cal_max_tokens(peak_memory, total_gpu_memory,
                                             fraction,
                                             model_engine.model.model_config,
                                             mapping)

    if executor_config.kv_cache_config.max_tokens is not None and kv_cache_max_tokens is not None:
        kv_cache_max_tokens = min(kv_cache_max_tokens,
                                  executor_config.kv_cache_config.max_tokens)

    return kv_cache_max_tokens
