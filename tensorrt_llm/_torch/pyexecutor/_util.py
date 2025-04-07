import random
from collections.abc import Iterable

import torch

import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import (BindCapacityScheduler,
                                                      BindMicroBatchScheduler,
                                                      SimpleScheduler)
from tensorrt_llm._torch.speculative import get_num_spec_layers
from tensorrt_llm._utils import (mpi_allgather, str_dtype_to_binding,
                                 torch_dtype_to_str)
from tensorrt_llm.bindings.executor import ExecutorConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .resource_manager import KVCacheManager

GB = 1 << 30


def is_mla(config):
    if hasattr(config, "kv_lora_rank"):
        assert hasattr(
            config, "qk_rope_head_dim"
        ), "both of kv_lora_rank and qk_rope_head_dim are required."
        return True
    return False


def cal_max_tokens(peak_memory, total_gpu_memory, fraction, model_config,
                   mapping: Mapping, kv_tokens: int, alloc_kv_tokens: int):
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

    num_hidden_layers = len(mapping.pp_layers_torch(config.num_hidden_layers))
    mem_per_token *= num_hidden_layers * head_dim
    # K and V
    mem_per_token *= kv_factor

    if fraction is None:
        fraction = 0.9

    available_kv_mem = (total_gpu_memory - peak_memory + alloc_kv_tokens *
                        mem_per_token - kv_tokens * mem_per_token) * fraction
    logger.info(
        f"Peak memory during memory usage profiling (torch + non-torch): {peak_memory / (GB):.2f} GiB, "
        f"available KV cache memory when calculating max tokens: {available_kv_mem / (GB):.2f} GiB"
    )
    max_tokens = int((available_kv_mem) // mem_per_token)
    max_tokens = max(max_tokens, 0)
    return max_tokens


def create_dummy_context_requests(max_seq_len: int, input_len: int,
                                  vocab_size: int):
    requests = []
    input_len = min(max_seq_len, input_len)
    import math
    batch_size = math.ceil(max_seq_len / input_len)
    for idx in range(batch_size):
        input_tokens = [
            random.randint(0, vocab_size - 1) for _ in range(input_len)
        ]
        output_config = trtllm.OutputConfig()
        output_config.exclude_input_from_output = True
        output_config.return_log_probs = False
        output_config.return_generation_logits = False
        output_config.return_context_logits = False
        request = trtllm.Request(input_tokens,
                                 max_tokens=1,
                                 streaming=False,
                                 sampling_config=trtllm.SamplingConfig(
                                     1, num_return_sequences=1),
                                 output_config=output_config,
                                 end_id=-1)
        requests.append(request)
    return requests


def get_token_num_for_estimation(executor_config):
    mapping = executor_config.mapping

    if 'cp_type' not in mapping.cp_config:
        return max(executor_config.max_batch_size,
                   executor_config.max_num_tokens, executor_config.max_seq_len)


def estimate_max_kv_cache_tokens_maybe_update_executor(
        py_executor: PyExecutor, model_engine: PyTorchModelEngine,
        executor_config: ExecutorConfig, mapping: Mapping, origin_seq_len: int,
        resources: dict, ctx_chunk_config):
    # TODO: support CP by generating dummy requests for it.
    if 'cp_type' in mapping.cp_config:
        return

    vocab_size = model_engine.model.model_config.pretrained_config.vocab_size
    max_num_tokens = executor_config.max_num_tokens
    fraction = executor_config.kv_cache_config.free_gpu_memory_fraction
    kv_cache_max_tokens_in = executor_config.kv_cache_config.max_tokens

    end, total_gpu_memory = torch.cuda.mem_get_info()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    py_executor.set_dist_response(True)
    origin_iter_stats = py_executor.enable_iter_perf_stats
    py_executor.enable_iter_perf_stats = False
    req_ids = []
    if py_executor.dist.mapping.rank == 0:
        req = create_dummy_context_requests(max_num_tokens, origin_seq_len,
                                            vocab_size)
        req_ids = py_executor.enqueue_requests(req)
    all_ids = py_executor.dist.allgather(req_ids)
    req_ids = [req_id for ids in all_ids for req_id in ids]

    py_executor.await_responses(req_ids)
    torch.cuda.synchronize()

    torch_peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

    # Clear the caching allocator before measuring the current memory usage
    torch.cuda.empty_cache()
    end, total_gpu_memory = torch.cuda.mem_get_info()
    torch_used_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    total_used_bytes = total_gpu_memory - end
    extra_cost = max(total_used_bytes - torch_used_bytes, 0)
    peak_memory = torch_peak_memory + extra_cost
    logger.info(
        f"Memory used outside torch in memory usage profiling: {extra_cost / (1<<30):.2f} GiB"
    )
    kv_stats = py_executor.resource_manager.resource_managers.get(
        "kv_cache_manager").get_kv_cache_stats()

    kv_occupied_blocks = kv_stats.max_num_blocks

    kv_cache_max_tokens = cal_max_tokens(
        peak_memory, total_gpu_memory, fraction,
        model_engine.model.model_config, mapping,
        kv_stats.used_num_blocks * kv_stats.tokens_per_block,
        kv_occupied_blocks * kv_stats.tokens_per_block)

    if kv_cache_max_tokens_in is not None and kv_cache_max_tokens is not None:
        kv_cache_max_tokens = min(kv_cache_max_tokens, kv_cache_max_tokens_in)

    logger.info(f"Estimated max tokens in KV cache : {kv_cache_max_tokens}")
    py_executor.set_dist_response(False)
    py_executor.enable_iter_perf_stats = origin_iter_stats

    py_executor.resource_manager.resource_managers.get(
        "kv_cache_manager").shutdown()
    executor_config.kv_cache_config.max_tokens = kv_cache_max_tokens
    kv_cache_manager = create_kv_cache_manager(executor_config, mapping,
                                               model_engine)

    if model_engine.attn_metadata is not None and kv_cache_manager is not None:
        model_engine.attn_metadata.kv_cache_manager = kv_cache_manager

    # KVCacheManager modifies these fields, update them to executor_config
    if kv_cache_manager is not None:
        executor_config.max_seq_len = kv_cache_manager.max_seq_len
        resources[
            "kv_cache_manager"] = kv_cache_manager if kv_cache_manager is not None else None

        resource_manager = ResourceManager(resources)
        py_executor.resource_manager = resource_manager

        capacity_scheduler = BindCapacityScheduler(
            executor_config.max_batch_size,
            kv_cache_manager.impl if kv_cache_manager is not None else None,
            executor_config.scheduler_config.capacity_scheduler_policy,
            num_micro_batches=mapping.pp_size)
        mb_scheduler = BindMicroBatchScheduler(executor_config.max_batch_size,
                                               executor_config.max_num_tokens,
                                               ctx_chunk_config)
        scheduler = SimpleScheduler(capacity_scheduler, mb_scheduler)
        if py_executor.kv_cache_transceiver is not None:
            py_executor.kv_cache_transceiver.reset_kv_cache_manager(
                kv_cache_manager)
        py_executor.scheduler = scheduler

        # mpi_barrier() always hang here.
        # sync all ranks after setting new kv_cache_manager
        mpi_allgather(0)


def create_kv_cache_manager(executor_config: ExecutorConfig, mapping,
                            model_engine):
    if executor_config.pytorch_backend_config.use_kv_cache:
        config = model_engine.model.model_config.pretrained_config
        spec_config = executor_config.speculative_config
        hidden_size = model_engine.model.config.hidden_size
        num_attention_heads = model_engine.model.config.num_attention_heads
        num_key_value_heads = getattr(model_engine.model.config,
                                      'num_key_value_heads',
                                      num_attention_heads)
        head_dim = hidden_size // num_attention_heads
        quant_config = model_engine.model.model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
        else:
            kv_cache_dtype = str_dtype_to_binding(
                torch_dtype_to_str(model_engine.dtype))

        num_hidden_layers = len(
            mapping.pp_layers_torch(
                model_engine.model.config.num_hidden_layers))
        # has kv cache
        if is_mla(config):
            if check_flash_mla_config(config):
                executor_config.tokens_per_block = 64
                logger.info(
                    f"Change tokens_per_block to: {executor_config.tokens_per_block} for using FlashMLA"
                )
            executor_config.kv_cache_config.enable_block_reuse = False
            executor_config.enable_chunked_context = False
            if spec_config is not None:
                num_hidden_layers += get_num_spec_layers(spec_config)
            kv_cache_manager = KVCacheManager(
                executor_config.kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.
                SELFKONLY,
                num_layers=num_hidden_layers,
                num_kv_heads=1,
                head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
                tokens_per_block=executor_config.tokens_per_block,
                max_seq_len=executor_config.max_seq_len,
                max_batch_size=executor_config.max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                num_extra_kv_tokens=0
                if spec_config is None else spec_config.num_extra_kv_tokens,
            )
        else:
            # the number of layers using attention in Nemotron5 is lower from the number of hidden layers
            if model_engine.model.config.architectures[
                    0] == "Nemotron5ForCausalLM":
                # attention layers are derived from configuration (hybrid_override_pattern)
                num_hidden_layers = model_engine.model.config.hybrid_override_pattern.count(
                    "*")
            kv_cache_manager = KVCacheManager(
                executor_config.kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
                num_layers=num_hidden_layers,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                tokens_per_block=executor_config.tokens_per_block,
                max_seq_len=executor_config.max_seq_len,
                max_batch_size=executor_config.max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                num_extra_kv_tokens=0
                if spec_config is None else spec_config.num_extra_kv_tokens,
            )
    else:
        # no kv cache
        kv_cache_manager = None

    return kv_cache_manager
