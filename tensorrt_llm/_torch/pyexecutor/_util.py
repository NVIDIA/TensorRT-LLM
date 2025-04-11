import random
from collections.abc import Iterable

import torch

import tensorrt_llm
import tensorrt_llm.bindings as tllm
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._utils import (mpi_allgather, mpi_broadcast,
                                 str_dtype_to_binding, torch_dtype_to_str)
from tensorrt_llm.bindings.executor import ExecutorConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..speculative import (get_num_spec_layers, get_spec_decoder,
                           get_spec_resource_manager)
from .decoder import (EarlyStopDecoder, TorchDecoder, TorchStarAttentionDecoder,
                      TRTLLMDecoder)
from .guided_decoder import GuidedDecoderResourceManager
from .kv_cache_transceiver import AttentionTypeCpp, create_kv_cache_transceiver
from .model_engine import (DRAFT_KV_CACHE_MANAGER_KEY, KV_CACHE_MANAGER_KEY,
                           PyTorchModelEngine)
from .py_executor import PyExecutor
from .resource_manager import KVCacheManager, ResourceManager
from .scheduler import (BindCapacityScheduler, BindMicroBatchScheduler,
                        SimpleScheduler)


def is_mla(config):
    if hasattr(config, "kv_lora_rank"):
        assert hasattr(
            config, "qk_rope_head_dim"
        ), "both of kv_lora_rank and qk_rope_head_dim are required."
        return True
    return False


GB = 1 << 30


def get_cache_size_per_token(model_config, mapping):
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
    return mem_per_token


def get_fraction_from_executor_config(executor_config):
    fraction = executor_config.kv_cache_config.free_gpu_memory_fraction
    if fraction is None:
        fraction = 0.9
    return fraction


def cal_max_tokens(peak_memory, total_gpu_memory, fraction, model_config,
                   mapping: Mapping, alloc_kv_tokens: int):
    kv_size_per_token = get_cache_size_per_token(model_config, mapping)

    available_kv_mem = (total_gpu_memory - peak_memory +
                        alloc_kv_tokens * kv_size_per_token) * fraction
    logger.info(
        f"Peak memory during memory usage profiling (torch + non-torch): {peak_memory / (GB):.2f} GiB, "
        f"available KV cache memory when calculating max tokens: {available_kv_mem / (GB):.2f} GiB"
    )
    max_tokens = int((available_kv_mem) // kv_size_per_token)
    max_tokens = max(max_tokens, 0)
    return max_tokens


def create_dummy_context_requests(max_num_tokens: int, max_seq_len: int,
                                  vocab_size: int):
    requests = []
    max_seq_len = min(max_num_tokens, max_seq_len)
    remaining_tokens = max_num_tokens
    while remaining_tokens > 0:
        input_len = min(max_seq_len, remaining_tokens)
        input_tokens = [
            random.randint(0, vocab_size - 1) for _ in range(input_len)
        ]
        request = trtllm.Request(input_tokens,
                                 max_tokens=1,
                                 streaming=False,
                                 sampling_config=trtllm.SamplingConfig(),
                                 output_config=trtllm.OutputConfig(),
                                 end_id=-1)
        requests.append(request)
        remaining_tokens -= input_len
    return requests


def get_token_num_for_estimation(executor_config, model_config):
    mapping = executor_config.mapping
    if 'cp_type' not in mapping.cp_config:
        end, _ = torch.cuda.mem_get_info()
        fraction = get_fraction_from_executor_config(executor_config)
        kv_size_per_token = get_cache_size_per_token(model_config, mapping)
        max_tokens_limit = int(end * fraction // kv_size_per_token)
        return min(
            max(executor_config.max_batch_size, executor_config.max_num_tokens,
                executor_config.max_seq_len), max_tokens_limit)
    else:
        return None


def estimate_max_kv_cache_tokens(py_executor: PyExecutor,
                                 model_engine: PyTorchModelEngine,
                                 executor_config: ExecutorConfig,
                                 mapping: Mapping, origin_seq_len: int,
                                 ctx_chunk_config,
                                 draft_model_engine: PyTorchModelEngine):
    # TODO: support CP by generating dummy requests for it.
    if 'cp_type' in mapping.cp_config:
        return executor_config.max_num_tokens

    vocab_size = model_engine.model.model_config.pretrained_config.vocab_size
    max_num_tokens = executor_config.max_num_tokens
    fraction = get_fraction_from_executor_config(executor_config)
    kv_cache_max_tokens_in = executor_config.kv_cache_config.max_tokens

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    py_executor.set_gather_responses(True)
    origin_iter_stats = py_executor.enable_iter_perf_stats
    py_executor.enable_iter_perf_stats = False
    req_ids = []
    if py_executor.dist.mapping.rank == 0:
        req = create_dummy_context_requests(max_num_tokens, origin_seq_len,
                                            vocab_size)
        req_ids = py_executor.enqueue_requests(req)
    req_ids = mpi_broadcast(req_ids, root=0)
    py_executor.start_worker()
    py_executor.await_responses(req_ids)
    # sync all ranks after processing dummy requests
    mpi_allgather(0)

    torch_peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

    # Clear the caching allocator before measuring the current memory usage
    torch.cuda.empty_cache()
    end, total_gpu_memory = torch.cuda.mem_get_info()
    torch_used_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    total_used_bytes = total_gpu_memory - end
    extra_cost = max(total_used_bytes - torch_used_bytes, 0)
    peak_memory = torch_peak_memory + extra_cost
    logger.info(
        f"Memory used outside torch in memory usage profiling: {extra_cost / (GB):.2f} GiB"
    )
    kv_stats = py_executor.resource_manager.resource_managers.get(
        "kv_cache_manager").get_kv_cache_stats()

    kv_cache_max_tokens = cal_max_tokens(
        peak_memory, total_gpu_memory, fraction,
        model_engine.model.model_config, mapping,
        kv_stats.max_num_blocks * kv_stats.tokens_per_block)

    if kv_cache_max_tokens_in is not None and kv_cache_max_tokens is not None:
        kv_cache_max_tokens = min(kv_cache_max_tokens, kv_cache_max_tokens_in)

    logger.info(f"Estimated max tokens in KV cache : {kv_cache_max_tokens}")
    py_executor.set_gather_responses(False)
    py_executor.enable_iter_perf_stats = origin_iter_stats

    py_executor.resource_manager.resource_managers.get(
        "kv_cache_manager").shutdown()

    py_executor.shutdown()
    # sync all ranks after creating new pyExecutor
    mpi_allgather(0)

    return kv_cache_max_tokens


def create_kv_cache_manager(model_engine: PyTorchModelEngine, mapping: Mapping,
                            executor_config: ExecutorConfig) -> KVCacheManager:
    if executor_config.pytorch_backend_config.use_kv_cache:
        config = model_engine.model.model_config.pretrained_config
        quant_config = model_engine.model.model_config.quant_config
        spec_config = executor_config.speculative_config

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads',
                                      num_attention_heads)
        head_dim = hidden_size // num_attention_heads

        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
        else:
            kv_cache_dtype = str_dtype_to_binding(
                torch_dtype_to_str(model_engine.dtype))

        num_hidden_layers = len(
            mapping.pp_layers_torch(config.num_hidden_layers))
        # the number of layers using attention in Nemotron5 is lower than the number of hidden layers
        if config.architectures[0] == "Nemotron5ForCausalLM":
            # attention layers are derived from configuration (hybrid_override_pattern)
            num_hidden_layers = config.hybrid_override_pattern.count("*")

        if is_mla(config):
            if spec_config is not None:
                num_hidden_layers += get_num_spec_layers(spec_config)

            return KVCacheManager(
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
            if spec_config is not None:
                num_hidden_layers += get_num_spec_layers(spec_config)
            return KVCacheManager(
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
        return None


def create_py_executor_instance(dist, kv_cache_manager, draft_kv_cache_manager,
                                mapping, pytorch_backend_config,
                                executor_config, ctx_chunk_config, model_engine,
                                draft_model_engine, start_worker):
    spec_config = model_engine.spec_config
    resources = {
        KV_CACHE_MANAGER_KEY: kv_cache_manager
    } if kv_cache_manager is not None else {}

    if draft_kv_cache_manager is not None:
        resources[DRAFT_KV_CACHE_MANAGER_KEY] = draft_kv_cache_manager

    if spec_config is not None:
        spec_resource_manager = get_spec_resource_manager(
            spec_config, model_engine.model.config, model_engine.batch_size * 2)
        spec_decoder = get_spec_decoder(max_seq_len=model_engine.max_seq_len,
                                        spec_config=spec_config)
        if spec_resource_manager is not None:
            resources["spec_resource_manager"] = spec_resource_manager
    else:
        spec_decoder = None

    if mapping.is_last_pp_rank(
    ) and executor_config.guided_decoding_config is not None:
        if spec_config is not None:
            raise ValueError(
                "Guided decoding does not support with speculative decoding.")
        resources[
            "guided_decoder_resource_manager"] = GuidedDecoderResourceManager(
                executor_config.max_batch_size)

    logger.info(
        f"max_seq_len={executor_config.max_seq_len}, max_num_requests={executor_config.max_batch_size}, max_num_tokens={executor_config.max_num_tokens}"
    )

    for key, value in pytorch_backend_config.extra_resource_managers.items():
        if key in resources:
            raise ValueError(
                f"Cannot overwrite existing resource manager {key}.")
        resources[key] = value

    resource_manager = ResourceManager(resources)

    # Make sure the kv cache manager is always invoked last as it could
    # depend on the results of other resource managers.
    if kv_cache_manager is not None:
        resource_manager.resource_managers.move_to_end("kv_cache_manager",
                                                       last=True)

    num_micro_batches = 1
    if mapping.has_pp:
        num_micro_batches = mapping.pp_size + pytorch_backend_config.enable_overlap_scheduler

    capacity_scheduler = BindCapacityScheduler(
        executor_config.max_batch_size,
        kv_cache_manager.impl if kv_cache_manager is not None else None,
        executor_config.scheduler_config.capacity_scheduler_policy,
        num_micro_batches=num_micro_batches)
    mb_scheduler = BindMicroBatchScheduler(executor_config.max_batch_size,
                                           executor_config.max_num_tokens,
                                           ctx_chunk_config)
    scheduler = SimpleScheduler(capacity_scheduler, mb_scheduler)

    config = model_engine.model.model_config.pretrained_config
    attention_type = AttentionTypeCpp.MLA if is_mla(
        config) else AttentionTypeCpp.DEFAULT
    kv_cache_transceiver = create_kv_cache_transceiver(mapping,
                                                       kv_cache_manager,
                                                       attention_type)

    if mapping.cp_config.get('cp_type') == 'star_attention':
        assert pytorch_backend_config.attn_backend == "FLASHINFER_STAR_ATTENTION", "attention backend of star attention should be 'FLASHINFER_STAR_ATTENTION'"
        decoder = TorchStarAttentionDecoder(
            max_seq_len=model_engine.max_seq_len)
    elif spec_decoder is not None:
        decoder = spec_decoder
    elif pytorch_backend_config.enable_trtllm_decoder:
        decoder = TRTLLMDecoder(executor_config, model_engine.model,
                                model_engine.dtype, mapping,
                                tllm.executor.DecodingMode.TopKTopP())
    else:
        # NOTE: choose decoder based on model type
        if not model_engine.model.model_config.is_generation:
            decoder = EarlyStopDecoder()
        else:
            decoder = TorchDecoder(
                max_seq_len=model_engine.max_seq_len,
                mixed_decoder=pytorch_backend_config.mixed_decoder)

    return PyExecutor(resource_manager,
                      scheduler,
                      model_engine=model_engine,
                      decoder=decoder,
                      dist=dist,
                      enable_overlap_scheduler=pytorch_backend_config.
                      enable_overlap_scheduler,
                      max_batch_size=executor_config.max_batch_size,
                      max_draft_tokens=spec_config.max_draft_tokens
                      if spec_config is not None else 0,
                      kv_cache_transceiver=kv_cache_transceiver,
                      draft_model_engine=draft_model_engine,
                      start_worker=start_worker)
