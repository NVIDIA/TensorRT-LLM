import math
import random
from collections.abc import Iterable
from typing import Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import DecodingMode, ExecutorConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import (LoraConfig,
                                       get_default_trtllm_modules_to_hf_modules,
                                       load_torch_hf_lora)
from tensorrt_llm.mapping import Mapping

from ..speculative import get_spec_decoder
from .config_utils import is_mla, is_nemotron_hybrid
from .kv_cache_transceiver import AttentionTypeCpp, create_kv_cache_transceiver
from .model_engine import KV_CACHE_MANAGER_KEY, PyTorchModelEngine
from .py_executor import PyExecutor
from .resource_manager import (KVCacheManager, MambaHybridCacheManager,
                               PeftCacheManager, ResourceManager)
from .sampler import (EarlyStopSampler, TorchSampler, TorchStarAttentionSampler,
                      TRTLLMSampler)
from .scheduler import (BindCapacityScheduler, BindMicroBatchScheduler,
                        SimpleScheduler)
from .seq_slot_manager import SeqSlotManager

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
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        ) * num_key_value_heads // tp_size

    # provide at least 1 layer to prevent division by zero cache size
    num_hidden_layers = max(len(mapping.pp_layers(config.num_hidden_layers)), 1)
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
                   draft_model_config, mapping: Mapping,
                   alloc_kv_tokens: int) -> int:
    model_kv_size_per_token = get_cache_size_per_token(model_config, mapping)
    draft_kv_size_per_token = get_cache_size_per_token(
        draft_model_config, mapping) if draft_model_config is not None else 0
    kv_size_per_token = model_kv_size_per_token + draft_kv_size_per_token

    available_kv_mem = (total_gpu_memory - peak_memory +
                        alloc_kv_tokens * kv_size_per_token) * fraction
    logger.info(
        f"Peak memory during memory usage profiling (torch + non-torch): {peak_memory / (GB):.2f} GiB, "
        f"available KV cache memory when calculating max tokens: {available_kv_mem / (GB):.2f} GiB, "
        f"fraction is set {fraction}, kv size is {kv_size_per_token}")
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
        # When reusing KV cache blocks, we need to add extra tokens to account for partially filled blocks
        # that cannot be reused. For each sequence of max_num_tokens length, we may need up to one extra
        # block (tokens_per_block tokens) if the sequence length is not perfectly divisible by tokens_per_block.
        # So we add math.ceil(max_num_tokens/max_seq_len) * tokens_per_block extra tokens.
        return min(
            max(
                executor_config.max_batch_size, executor_config.max_num_tokens +
                math.ceil(executor_config.max_num_tokens /
                          executor_config.max_seq_len) *
                executor_config.tokens_per_block, executor_config.max_seq_len),
            max_tokens_limit)
    else:
        return None


def estimate_max_kv_cache_tokens(
        py_executor: PyExecutor, model_engine: PyTorchModelEngine,
        executor_config: ExecutorConfig, mapping: Mapping, origin_seq_len: int,
        ctx_chunk_config,
        draft_model_engine: PyTorchModelEngine) -> Optional[int]:
    # TODO: support CP by generating dummy requests for it.
    if 'cp_type' in mapping.cp_config:
        return executor_config.max_num_tokens

    vocab_size = model_engine.model.model_config.pretrained_config.vocab_size
    max_num_tokens = executor_config.max_num_tokens
    fraction = get_fraction_from_executor_config(executor_config)
    kv_cache_max_tokens_in = executor_config.kv_cache_config.max_tokens

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    logger.info(
        f"Memory used after loading model weights (inside torch) in memory usage profiling: {model_bytes / (GB):.2f} GiB"
    )

    py_executor.set_gather_responses(True)
    origin_iter_stats = py_executor.enable_iter_perf_stats
    py_executor.enable_iter_perf_stats = False
    req_ids = []
    if py_executor.dist.mapping.rank == 0:
        # NOTE: TRTLLMSampler requires origin_seq_len - 1 for requests.
        #       Spec decoders with overlap require origin_seq_len.
        seq_len = origin_seq_len - 1 if type(
            py_executor.sampler) == TRTLLMSampler else origin_seq_len
        req = create_dummy_context_requests(max_num_tokens, seq_len, vocab_size)
        req_ids = py_executor.enqueue_requests(req)
    req_ids = py_executor.dist.broadcast(req_ids, root=0)
    py_executor.is_warmup = True
    py_executor.start_worker()
    py_executor.await_responses(req_ids)

    torch_peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

    # Clear the caching allocator before measuring the current memory usage
    torch.cuda.empty_cache()
    end, total_gpu_memory = torch.cuda.mem_get_info()
    torch_used_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    total_used_bytes = total_gpu_memory - end
    activation_bytes = torch_peak_memory - model_bytes
    extra_cost = max(total_used_bytes - torch_used_bytes, 0)
    peak_memory = torch_peak_memory + extra_cost
    logger.info(
        f"Memory dynamically allocated during inference (inside torch) in memory usage profiling: {activation_bytes / (GB):.2f} GiB"
    )
    logger.info(
        f"Memory used outside torch (e.g., NCCL and CUDA graphs) in memory usage profiling: {extra_cost / (GB):.2f} GiB"
    )
    kv_stats = py_executor.resource_manager.resource_managers.get(
        "kv_cache_manager").get_kv_cache_stats()

    draft_model_config = draft_model_engine.model.model_config if draft_model_engine is not None else None
    kv_cache_max_tokens = cal_max_tokens(
        peak_memory, total_gpu_memory, fraction,
        model_engine.model.model_config, draft_model_config, mapping,
        kv_stats.max_num_blocks * kv_stats.tokens_per_block)

    if kv_cache_max_tokens_in is not None:
        kv_cache_max_tokens = min(kv_cache_max_tokens, kv_cache_max_tokens_in)

    logger.info(f"Estimated max tokens in KV cache : {kv_cache_max_tokens}")
    py_executor.set_gather_responses(False)
    py_executor.enable_iter_perf_stats = origin_iter_stats

    py_executor.resource_manager.resource_managers.get(
        "kv_cache_manager").shutdown()

    py_executor.is_warmup = False
    py_executor.shutdown()

    return kv_cache_max_tokens


def create_kv_cache_manager(model_engine: PyTorchModelEngine, mapping: Mapping,
                            executor_config: ExecutorConfig) -> KVCacheManager:
    kv_cache_manager = None
    if executor_config.pytorch_backend_config.use_kv_cache:
        config = model_engine.model.model_config.pretrained_config
        quant_config = model_engine.model.model_config.quant_config
        spec_config = executor_config.speculative_config

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads',
                                      num_attention_heads)
        head_dim = getattr(config, "head_dim",
                           hidden_size // num_attention_heads)

        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
        else:
            kv_cache_dtype = str_dtype_to_binding(
                torch_dtype_to_str(model_engine.dtype))

        num_hidden_layers = config.num_hidden_layers

        if is_mla(config):
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
                spec_config=spec_config,
            )
        elif is_nemotron_hybrid(config):
            config = model_engine.model.model_config.pretrained_config
            num_layers = config.hybrid_override_pattern.count("*")
            layer_mask = [
                char == "*" for char in config.hybrid_override_pattern
            ]
            mamba_num_layers = config.hybrid_override_pattern.count("M")
            mamba_layer_mask = [
                char == "M" for char in config.hybrid_override_pattern
            ]
            kv_cache_manager = MambaHybridCacheManager(
                # mamba cache parameters
                config.hidden_size,
                config.ssm_state_size,
                config.conv_kernel,
                config.expand,
                config.n_groups,
                config.mamba_head_dim,
                mamba_num_layers,
                mamba_layer_mask,
                config.torch_dtype,
                # kv cache parameters
                executor_config.kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
                num_layers=num_layers,
                layer_mask=layer_mask,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                tokens_per_block=executor_config.tokens_per_block,
                max_seq_len=executor_config.max_seq_len,
                max_batch_size=executor_config.max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                spec_config=spec_config,
            )
        else:
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
                spec_config=spec_config,
            )
        # KVCacheManager (Non-draft) modifies the max_seq_len field, update it to executor_config
        if model_engine.kv_cache_manager_key == KV_CACHE_MANAGER_KEY:
            executor_config.max_seq_len = kv_cache_manager.max_seq_len

    assert kv_cache_manager is not None
    return kv_cache_manager


def create_py_executor_instance(
        dist,
        resources,
        mapping,
        pytorch_backend_config,
        executor_config,
        ctx_chunk_config,
        model_engine,
        draft_model_engine,
        start_worker,
        sampler,
        lora_config: Optional[LoraConfig] = None) -> PyExecutor:
    kv_cache_manager = resources.get(KV_CACHE_MANAGER_KEY, None)

    spec_config = model_engine.spec_config
    if mapping.is_last_pp_rank(
    ) and executor_config.guided_decoding_config is not None:
        if spec_config is not None:
            raise ValueError(
                "Guided decoding is not supported with speculative decoding.")
        if not pytorch_backend_config.disable_overlap_scheduler:
            raise ValueError(
                "Guided decoding is not supported with overlap scheduler.")

    logger.info(
        f"max_seq_len={executor_config.max_seq_len}, max_num_requests={executor_config.max_batch_size}, max_num_tokens={executor_config.max_num_tokens}"
    )

    for key, value in pytorch_backend_config.extra_resource_managers.items():
        if key in resources:
            raise ValueError(
                f"Cannot overwrite existing resource manager {key}.")
        resources[key] = value

    if lora_config is not None:
        from tensorrt_llm.bindings import LoraModule

        if len(lora_config.lora_dir) == 1:
            load_torch_hf_lora(lora_config)
        else:
            assert len(lora_config.lora_target_modules
                       ) >= 1, "Expecting at least one lora target module"
            if not bool(lora_config.trtllm_modules_to_hf_modules):
                lora_config.trtllm_modules_to_hf_modules = get_default_trtllm_modules_to_hf_modules(
                )

        model_binding_config = model_engine.model.model_config.get_bindings_model_config(
        )

        num_experts = _try_infer_num_experts(model_engine.model.model_config)

        lora_modules = LoraModule.create_lora_modules(
            lora_module_names=lora_config.lora_target_modules,
            hidden_size=model_binding_config.hidden_size,
            mlp_hidden_size=model_binding_config.mlp_hidden_size,
            num_attention_heads=model_binding_config.num_heads,
            num_kv_attention_heads=model_binding_config.num_heads,
            attention_head_size=model_binding_config.head_size,
            tp_size=mapping.tp_size,
            num_experts=num_experts)
        model_binding_config.use_lora_plugin = True
        model_binding_config.lora_modules = lora_modules
        model_binding_config.max_lora_rank = lora_config.max_lora_rank

        max_lora_rank = lora_config.max_lora_rank
        num_lora_modules = model_engine.model.model_config.pretrained_config.num_hidden_layers * \
            len(lora_config.lora_target_modules + lora_config.missing_qkv_modules)

        # TODO smor- need to figure out how to set these values
        executor_config.peft_cache_config = trtllm.PeftCacheConfig(
            num_device_module_layer=max_lora_rank * num_lora_modules *
            lora_config.max_loras,
            num_host_module_layer=max_lora_rank * num_lora_modules *
            lora_config.max_cpu_loras,
        )

        from tensorrt_llm.bindings import WorldConfig
        world_config = WorldConfig(
            tensor_parallelism=mapping.tp_size,
            pipeline_parallelism=mapping.pp_size,
            context_parallelism=mapping.cp_size,
            rank=dist.mapping.rank,
            gpus_per_node=dist.mapping.gpus_per_node,
        )
        peft_cache_manager = PeftCacheManager(
            peft_cache_config=executor_config.peft_cache_config,
            model_config=model_binding_config,
            world_config=world_config,
        )
        resources["peft_cache_manager"] = peft_cache_manager
        model_engine.set_lora_model_config(
            lora_config.lora_target_modules,
            lora_config.trtllm_modules_to_hf_modules)

    if mapping.has_pp():
        num_micro_batches = mapping.pp_size
    else:
        num_micro_batches = 1 if pytorch_backend_config.disable_overlap_scheduler else 2

    resources["seq_slot_manager"] = SeqSlotManager(
        executor_config.max_batch_size * num_micro_batches)

    resource_manager = ResourceManager(resources)

    # Make sure the kv cache manager is always invoked last as it could
    # depend on the results of other resource managers.
    if kv_cache_manager is not None:
        resource_manager.resource_managers.move_to_end("kv_cache_manager",
                                                       last=True)

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
    cache_transceiver_config = executor_config.cache_transceiver_config
    kv_cache_transceiver = create_kv_cache_transceiver(
        mapping, kv_cache_manager, attention_type, cache_transceiver_config)

    return PyExecutor(resource_manager,
                      scheduler,
                      model_engine=model_engine,
                      sampler=sampler,
                      dist=dist,
                      disable_overlap_scheduler=pytorch_backend_config.
                      disable_overlap_scheduler,
                      max_batch_size=executor_config.max_batch_size,
                      max_draft_tokens=spec_config.max_draft_tokens
                      if spec_config is not None else 0,
                      kv_cache_transceiver=kv_cache_transceiver,
                      draft_model_engine=draft_model_engine,
                      start_worker=start_worker)


def instantiate_sampler(model_engine: PyTorchModelEngine,
                        executor_config: ExecutorConfig,
                        pytorch_backend_config: PyTorchConfig,
                        mapping: Mapping):
    if mapping.cp_config.get('cp_type') == 'star_attention':
        assert pytorch_backend_config.attn_backend == "FLASHINFER_STAR_ATTENTION", "attention backend of star attention should be 'FLASHINFER_STAR_ATTENTION'"
        sampler = TorchStarAttentionSampler(
            max_seq_len=model_engine.max_seq_len)
    elif model_engine.spec_config is not None:
        sampler = get_spec_decoder(max_seq_len=model_engine.max_seq_len,
                                   spec_config=model_engine.spec_config)
    elif pytorch_backend_config.enable_trtllm_sampler:
        decoding_mode = get_decoding_mode(executor_config)
        sampler = TRTLLMSampler(
            executor_config, model_engine.model, model_engine.dtype, mapping,
            decoding_mode, pytorch_backend_config.disable_overlap_scheduler)
    elif not model_engine.model.model_config.is_generation:
        # NOTE: choose sampler based on model type
        sampler = EarlyStopSampler()
    else:
        sampler = TorchSampler(
            max_seq_len=model_engine.max_seq_len,
            mixed_sampler=pytorch_backend_config.mixed_sampler)
    return sampler


def get_decoding_mode(executor_config: ExecutorConfig) -> DecodingMode:
    '''This implementation is based off trtGptModelInflightBatching.cpp getDecodingMode().'''

    if executor_config.decoding_config and executor_config.decoding_config.decoding_mode and not executor_config.decoding_config.decoding_mode.isAuto(
    ):
        decoding_mode = executor_config.decoding_config.decoding_mode
    elif executor_config.max_beam_width == 1:
        decoding_mode = DecodingMode.TopKTopP()
    else:
        decoding_mode = DecodingMode.BeamSearch()

    # Override decoding mode when beam width is one
    if executor_config.max_beam_width == 1 and decoding_mode.isBeamSearch():
        logger.warning(
            "Beam width is set to 1, but decoding mode is BeamSearch. Overwriting decoding mode to TopKTopP."
        )
        decoding_mode = DecodingMode.TopKTopP()

    # Override decoding mode when Medusa is used
    if executor_config.speculative_config and executor_config.speculative_config.is_medusa and not decoding_mode.isMedusa(
    ):
        logger.warning(
            "Model is Medusa, but decoding mode is not Medusa. Overwriting decoding mode to Medusa."
        )
        decoding_mode = DecodingMode.Medusa()

    # Override decoding mode when Medusa is not used
    if (not executor_config.speculative_config
            or not executor_config.speculative_config.is_medusa
        ) and decoding_mode.isMedusa():
        logger.warning(
            "Model is not Medusa, but decoding mode is Medusa. Overwriting decoding mode."
        )
        if executor_config.max_beam_width == 1:
            decoding_mode = DecodingMode.TopKTopP()
        else:
            decoding_mode = DecodingMode.BeamSearch()

    # Override decoding mode when lookahead decoding is used
    if executor_config.speculative_config and executor_config.speculative_config.is_lookahead and not decoding_mode.isLookahead(
    ):
        logger.warning(
            "Model is Lookahead, but decoding mode is not Lookahead. Overwriting decoding mode to Lookahead."
        )
        decoding_mode = DecodingMode.Lookahead()

    # Override decoding mode when lookahead decoding is not used
    if (not executor_config.speculative_config
            or not executor_config.speculative_config.is_lookahead
        ) and decoding_mode.isLookahead():
        logger.warning(
            "Model is not built with Lookahead decoding, but decoding mode is Lookahead. Overwriting decoding mode."
        )
        if executor_config.max_beam_width == 1:
            decoding_mode = DecodingMode.TopKTopP()
        else:
            decoding_mode = DecodingMode.BeamSearch()

    # Override decoding mode when 'explicit draft tokens' is used
    if executor_config.speculative_config and executor_config.speculative_config.is_explicit_draft_tokens and not decoding_mode.isExplicitDraftTokens(
    ):
        logger.warning(
            "Model is built with 'explicit draft tokens' decoding, but decoding mode is something else. Overwriting decoding mode."
        )
        decoding_mode = DecodingMode.ExplicitDraftTokens()

    # Override decoding mode when 'explicit draft tokens' is not used
    if (not executor_config.speculative_config
            or not executor_config.speculative_config.is_explicit_draft_tokens
        ) and decoding_mode.isExplicitDraftTokens():
        logger.warning(
            "Model is not built with 'explicit draft tokens' decoding, but decoding mode is set to it. Overwriting decoding mode to default."
        )
        if executor_config.max_beam_width == 1:
            decoding_mode = DecodingMode.TopKTopP()
        else:
            decoding_mode = DecodingMode.BeamSearch()

    # Override decoding mode when EAGLE is used
    if executor_config.speculative_config and executor_config.speculative_config.is_eagle and not decoding_mode.isEagle(
    ):
        logger.warning(
            "Model is Eagle, but decoding mode is not Eagle. Overwriting decoding mode to Eagle."
        )
        decoding_mode = DecodingMode.Eagle()

    # Override decoding mode when Eagle is not used
    if (not executor_config.speculative_config
            or not executor_config.speculative_config.is_eagle
        ) and decoding_mode.isEagle():
        logger.warning(
            "Model is not Eagle, but decoding mode is Eagle. Overwriting decoding mode."
        )
        if executor_config.max_beam_width == 1:
            decoding_mode = DecodingMode.TopKTopP()
        else:
            decoding_mode = DecodingMode.BeamSearch()

    # Override decoding mode when draft tokens are external
    if executor_config.speculative_config and executor_config.speculative_config.is_draft_tokens_external:
        logger.warning("Overwriting decoding mode to external draft token")
        decoding_mode = DecodingMode.ExternalDraftTokens()

    logger.debug(f"DecodingMode: {decoding_mode.name}")
    return decoding_mode


def _try_infer_num_experts(model_config: ModelConfig) -> int:
    """
    Attempt to infer the number of experts from the model configuration.

    Different MoE models use different attribute names for storing the number of experts,
    so this function checks for various possible names and returns a default of 1 if none are found.
    However, this function is not exhaustive and may miss some cases, so it should be revised.
    """
    config = getattr(model_config, 'pretrained_config', model_config)

    expert_attr_names = [
        'num_experts', 'num_local_experts', 'moe_num_experts',
        'experts_per_router'
    ]
    num_experts = None
    for attr_name in expert_attr_names:
        if hasattr(config, attr_name):
            num_experts = getattr(config, attr_name)
            break

    # Default to 1 for non-MoE models or if no experts attribute is found
    if num_experts is None:
        return 1

    return num_experts
