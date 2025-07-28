import os
import random
from collections.abc import Iterable
from typing import Dict, List, Optional

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
                                       load_torch_lora)
from tensorrt_llm.mapping import Mapping

from ..model_config import ModelConfig
from ..speculative import get_num_extra_kv_tokens, get_spec_decoder
from .config import PyTorchConfig
from .config_utils import is_mla, is_nemotron_hybrid
from .guided_decoder import GuidedDecoder
from .kv_cache_transceiver import AttentionTypeCpp, create_kv_cache_transceiver
from .llm_request import ExecutorResponse
from .model_engine import PyTorchModelEngine
from .py_executor import PyExecutor
from .resource_manager import (KVCacheManager, MambaHybridCacheManager,
                               PeftCacheManager, ResourceManager,
                               ResourceManagerType)
from .sampler import EarlyStopSampler, TorchSampler, TRTLLMSampler
from .scheduler import (BindCapacityScheduler, BindMicroBatchScheduler,
                        SimpleScheduler)
from .seq_slot_manager import SeqSlotManager

GB = 1 << 30


class KvCacheCreator:
    """Groups together logic related to KV cache construction."""

    def __init__(self, *, executor_config: ExecutorConfig,
                 model_engine: PyTorchModelEngine,
                 draft_model_engine: Optional[PyTorchModelEngine],
                 mapping: Mapping, net_max_seq_len: int):
        self._executor_config = executor_config
        self._model_engine = model_engine
        self._draft_model_engine = draft_model_engine
        self._mapping = mapping
        self._max_kv_tokens_in = self._executor_config.kv_cache_config.max_tokens
        self._dummy_reqs = self._create_dummy_context_requests(net_max_seq_len -
                                                               1)

    @staticmethod
    def _get_cache_size_per_token(model_config: ModelConfig,
                                  mapping: Mapping) -> int:
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
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
            _head_dim = getattr(config, 'head_dim', None)
            if not isinstance(_head_dim, int):
                _head_dim = config.hidden_size // config.num_attention_heads
            head_dim = _head_dim * num_key_value_heads // tp_size

        # provide at least 1 layer to prevent division by zero cache size
        num_attention_layers = max(
            len(mapping.pp_layers(model_config.get_num_attention_layers())), 1)
        mem_per_token *= num_attention_layers * head_dim
        # K and V
        mem_per_token *= kv_factor
        return mem_per_token

    def _get_free_gpu_memory_fraction(self) -> float:
        fraction = self._executor_config.kv_cache_config.free_gpu_memory_fraction
        if fraction is None:
            fraction = 0.9
        return fraction

    def _cal_max_tokens(self, peak_memory, total_gpu_memory, fraction,
                        alloc_kv_tokens: int) -> int:
        model_config = self._model_engine.model.model_config
        mapping = self._mapping
        kv_size_per_token = self._get_cache_size_per_token(
            model_config, mapping)
        if self._draft_model_engine is not None:
            draft_model_config = self._draft_model_engine.model.model_config
            kv_size_per_token += self._get_cache_size_per_token(
                draft_model_config, mapping)

        available_kv_mem = (total_gpu_memory - peak_memory +
                            alloc_kv_tokens * kv_size_per_token) * fraction
        logger.info(
            f"Peak memory during memory usage profiling (torch + non-torch): {peak_memory / (GB):.2f} GiB, "
            f"available KV cache memory when calculating max tokens: {available_kv_mem / (GB):.2f} GiB, "
            f"fraction is set {fraction}, kv size is {kv_size_per_token}. device total memory {total_gpu_memory / (GB):.2f} GiB, "
            f", tmp kv_mem { (alloc_kv_tokens * kv_size_per_token) / (GB):.2f} GiB"
        )
        max_tokens = int((available_kv_mem) // kv_size_per_token)
        max_tokens = max(max_tokens, 0)
        return max_tokens

    def _create_dummy_context_requests(
            self, input_seq_len: int) -> List[trtllm.Request]:
        vocab_size = self._model_engine.model.model_config.pretrained_config.vocab_size
        max_num_tokens = self._executor_config.max_num_tokens
        max_beam_width = self._executor_config.max_beam_width

        requests = []
        input_seq_len = min(max_num_tokens, input_seq_len)
        remaining_tokens = max_num_tokens
        while remaining_tokens > 0:
            input_seq_len = min(input_seq_len, remaining_tokens)
            input_tokens = [
                random.randint(0, vocab_size - 1) for _ in range(input_seq_len)
            ]
            request = trtllm.Request(input_tokens,
                                     max_tokens=1,
                                     streaming=False,
                                     sampling_config=trtllm.SamplingConfig(
                                         beam_width=max_beam_width, ),
                                     output_config=trtllm.OutputConfig(),
                                     end_id=-1)
            requests.append(request)
            remaining_tokens -= input_seq_len
        if self._mapping.enable_attention_dp:
            requests = requests * self._mapping.tp_size
        return requests

    def _get_token_num_for_estimation(self) -> int:
        """Compute KV cache capacity required for estimate_max_kv_cache_tokens to succeed."""
        executor_config = self._executor_config
        if 'cp_type' in self._mapping.cp_config:
            raise ValueError(
                "KV cache size estimation not supported with context parallelism."
            )
        # estimate_max_kv_cache_tokens submits self._dummy_reqs
        num_cache_blocks = 0
        num_extra_tokens_per_seq = 1  # account for generated tokens
        pytorch_backend_config = executor_config.pytorch_backend_config
        spec_cfg = executor_config.speculative_config
        if not pytorch_backend_config.disable_overlap_scheduler:
            num_extra_tokens_per_seq = num_extra_tokens_per_seq + 1
            if spec_cfg is not None:
                num_extra_tokens_per_seq += spec_cfg.max_draft_len

        if spec_cfg is not None:
            num_extra_tokens_per_seq += spec_cfg.max_draft_len
            num_extra_tokens_per_seq += get_num_extra_kv_tokens(spec_cfg)
        for req in self._dummy_reqs:
            num_req_tokens = len(req.input_token_ids) + num_extra_tokens_per_seq
            # Requests cannot share KV cache blocks. Round up to nearest integer multiple of block size.
            num_cache_blocks += (num_req_tokens +
                                 executor_config.tokens_per_block -
                                 1) // executor_config.tokens_per_block
        # Multiply by beam width, to prevent rescaling of the max_seq_len caused by the influence of beam width during the preparation for kv_cache_estimation
        return num_cache_blocks * executor_config.tokens_per_block * self._dummy_reqs[
            0].sampling_config.beam_width

    def try_prepare_estimation(self) -> bool:
        """Prepare for possible KV cache capacity estimation.

        This updates `kv_cache_config` and returns a boolean indicating whether KV cache
        estimation is to be performend.
        """
        estimating_kv_cache = False
        if 'cp_type' not in self._mapping.cp_config:
            estimating_kv_cache = True
            self._executor_config.kv_cache_config.max_tokens = self._get_token_num_for_estimation(
            )
        return estimating_kv_cache

    def estimate_max_tokens(self, py_executor: PyExecutor) -> None:
        """Perform KV cache capacity estimation.

        This updates `kv_cache_config`.
        """
        executor_config = self._executor_config
        mapping = self._mapping

        # TODO: support CP by generating dummy requests for it.
        assert 'cp_type' not in mapping.cp_config

        fraction = self._get_free_gpu_memory_fraction()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        end, total_gpu_memory = torch.cuda.mem_get_info()
        total_used_bytes = total_gpu_memory - end
        model_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        logger.info(
            f"Memory used after loading model weights (inside torch) in memory usage profiling: {model_bytes / (GB):.2f} GiB"
        )
        logger.info(
            f"Memory used after loading model weights (outside torch) in memory usage profiling: {((total_used_bytes - model_bytes) if total_used_bytes > model_bytes else 0) / (GB):.2f} GiB"
        )

        py_executor.set_gather_responses(True)
        origin_iter_stats = py_executor.enable_iter_perf_stats
        py_executor.enable_iter_perf_stats = False
        req_ids = []
        if py_executor.dist.mapping.rank == 0:
            req_ids = py_executor.enqueue_requests(self._dummy_reqs)
        req_ids = py_executor.dist.broadcast(req_ids, root=0)
        py_executor.is_warmup = True
        py_executor.start_worker()
        try:
            responses = py_executor.await_responses(req_ids)
            for response_or_list in responses:
                response_list = [response_or_list] if isinstance(
                    response_or_list, ExecutorResponse) else response_or_list
                for response in response_list:
                    if response.has_error():
                        raise RuntimeError(response.error_msg)

            torch_peak_memory = torch.cuda.memory_stats(
            )["allocated_bytes.all.peak"]

            # Clear the caching allocator before measuring the current memory usage
            torch.cuda.empty_cache()
            end, total_gpu_memory = torch.cuda.mem_get_info()
            torch_used_bytes = torch.cuda.memory_stats(
            )["allocated_bytes.all.current"]
        finally:
            py_executor.shutdown()
            py_executor.is_warmup = False
            py_executor.enable_iter_perf_stats = origin_iter_stats
            py_executor.set_gather_responses(False)

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
            ResourceManagerType.KV_CACHE_MANAGER).get_kv_cache_stats()

        kv_cache_max_tokens = self._cal_max_tokens(
            peak_memory, total_gpu_memory, fraction,
            kv_stats.max_num_blocks * kv_stats.tokens_per_block)

        if self._max_kv_tokens_in is not None:
            kv_cache_max_tokens = min(kv_cache_max_tokens,
                                      self._max_kv_tokens_in)

        logger.info(f"Estimated max tokens in KV cache : {kv_cache_max_tokens}")
        executor_config.kv_cache_config.max_tokens = kv_cache_max_tokens

    def _create_kv_cache_manager(
            self, model_engine: PyTorchModelEngine) -> KVCacheManager:
        executor_config = self._executor_config
        mapping = self._mapping
        assert model_engine.model.model_config.is_generation, "Only construct KV cache for generation models."

        config = model_engine.model.model_config.pretrained_config
        quant_config = model_engine.model.model_config.quant_config
        spec_config = executor_config.speculative_config

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads',
                                      num_attention_heads)
        head_dim = getattr(config, "head_dim", None)
        if not isinstance(head_dim, int):
            head_dim = hidden_size // num_attention_heads

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
                max_beam_width=executor_config.max_beam_width,
            )
        elif is_nemotron_hybrid(config):
            if executor_config.max_beam_width > 1:
                raise ValueError(
                    "MambaHybridCacheManager + beam search is not supported yet."
                )
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
            # NOTE: this is a workaround for VSWA to switch to calculate_max_num_blocks_from_cpp in KVCahceManager
            is_vswa = executor_config.kv_cache_config.max_attention_window is not None and len(
                set(executor_config.kv_cache_config.max_attention_window)) > 1
            binding_model_config = model_engine.model.model_config.get_bindings_model_config(
                tokens_per_block=executor_config.tokens_per_block
            ) if is_vswa else None

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
                max_num_tokens=executor_config.max_num_tokens,
                model_config=binding_model_config,
                max_beam_width=executor_config.max_beam_width,
            )
        # KVCacheManager (Non-draft) modifies the max_seq_len field, update it to executor_config
        if model_engine.kv_cache_manager_key == ResourceManagerType.KV_CACHE_MANAGER:
            executor_config.max_seq_len = kv_cache_manager.max_seq_len

        return kv_cache_manager

    def build_managers(self, resources: Dict) -> None:
        """Construct KV caches for model and draft model (if applicable)."""
        kv_cache_manager = self._create_kv_cache_manager(self._model_engine)
        draft_kv_cache_manager = self._create_kv_cache_manager(
            self._draft_model_engine
        ) if self._draft_model_engine is not None else None
        resources[ResourceManagerType.KV_CACHE_MANAGER] = kv_cache_manager
        resources[
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER] = draft_kv_cache_manager

    def teardown_managers(self, resources: Dict) -> None:
        """Clean up KV caches for model and draft model (if applicable)."""
        resources[ResourceManagerType.KV_CACHE_MANAGER].shutdown()
        del resources[ResourceManagerType.KV_CACHE_MANAGER]
        draft_kv_cache_manager = resources[
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER]
        if draft_kv_cache_manager:
            draft_kv_cache_manager.shutdown()
        del resources[ResourceManagerType.DRAFT_KV_CACHE_MANAGER]


def create_py_executor_instance(
        *,
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
        drafter,
        guided_decoder: Optional[GuidedDecoder] = None,
        lora_config: Optional[LoraConfig] = None,
        garbage_collection_gen0_threshold: Optional[int] = None) -> PyExecutor:
    kv_cache_manager = resources.get(ResourceManagerType.KV_CACHE_MANAGER, None)

    spec_config = model_engine.spec_config

    logger.info(
        f"max_seq_len={executor_config.max_seq_len}, max_num_requests={executor_config.max_batch_size}, max_num_tokens={executor_config.max_num_tokens}, max_batch_size={executor_config.max_batch_size}"
    )

    for key, value in pytorch_backend_config.extra_resource_managers.items():
        if key in resources:
            raise ValueError(
                f"Cannot overwrite existing resource manager {key}.")
        resources[key] = value

    peft_cache_manager = None
    if lora_config is not None:
        from tensorrt_llm.bindings import LoraModule

        if len(lora_config.lora_dir) == 1:
            # Route to appropriate loader based on checkpoint source
            load_torch_lora(lora_config)
        else:
            assert len(lora_config.lora_target_modules
                       ) >= 1, "Expecting at least one lora target module"
            if not bool(lora_config.trtllm_modules_to_hf_modules):
                lora_config.trtllm_modules_to_hf_modules = get_default_trtllm_modules_to_hf_modules(
                )

        model_binding_config = model_engine.model.model_config.get_bindings_model_config(
        )

        num_experts = _try_infer_num_experts(model_engine.model.model_config)

        num_attn_layers = model_binding_config.num_attention_layers()
        per_layer_kv_heads = [
            model_binding_config.num_kv_heads(i) for i in range(num_attn_layers)
        ]
        num_kv_attention_heads = max(per_layer_kv_heads)
        if len(set(per_layer_kv_heads)) > 1:
            # NOTE: This code-path is currently untested and not validated. Can fail!
            # This support is tracked in TRTLLM-6561
            logger.warning(
                f"Non-uniform KV heads per layer detected, using max ({num_kv_attention_heads}) for LoRA. "
                "This code-path is currently untested and not validated. May fail!"
            )

        lora_modules = LoraModule.create_lora_modules(
            lora_module_names=lora_config.lora_target_modules,
            hidden_size=model_binding_config.hidden_size,
            mlp_hidden_size=model_binding_config.mlp_hidden_size,
            num_attention_heads=model_binding_config.num_heads,
            num_kv_attention_heads=num_kv_attention_heads,
            attention_head_size=model_binding_config.head_size,
            tp_size=mapping.tp_size,
            num_experts=num_experts)
        model_binding_config.use_lora_plugin = True
        model_binding_config.lora_modules = lora_modules
        model_binding_config.max_lora_rank = lora_config.max_lora_rank

        max_lora_rank = lora_config.max_lora_rank
        num_lora_modules = model_engine.model.model_config.pretrained_config.num_hidden_layers * \
            len(lora_config.lora_target_modules + lora_config.missing_qkv_modules)

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
        resources[ResourceManagerType.PEFT_CACHE_MANAGER] = peft_cache_manager
        model_engine.set_lora_model_config(
            lora_config.lora_target_modules,
            lora_config.trtllm_modules_to_hf_modules)

    max_num_sequences = executor_config.max_batch_size * mapping.pp_size

    resources[ResourceManagerType.SEQ_SLOT_MANAGER] = SeqSlotManager(
        max_num_sequences)

    resource_manager = ResourceManager(resources)

    # Make sure the kv cache manager is always invoked last as it could
    # depend on the results of other resource managers.
    if kv_cache_manager is not None:
        resource_manager.resource_managers.move_to_end(
            ResourceManagerType.KV_CACHE_MANAGER, last=True)

    capacity_scheduler = BindCapacityScheduler(
        max_num_sequences,
        kv_cache_manager.impl if kv_cache_manager is not None else None,
        peft_cache_manager.impl if peft_cache_manager is not None else None,
        executor_config.scheduler_config.capacity_scheduler_policy,
        two_step_lookahead=mapping.has_pp())
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
    return PyExecutor(
        resource_manager,
        scheduler,
        model_engine=model_engine,
        sampler=sampler,
        drafter=drafter,
        dist=dist,
        max_num_sequences=max_num_sequences,
        disable_overlap_scheduler=pytorch_backend_config.
        disable_overlap_scheduler,
        max_batch_size=executor_config.max_batch_size,
        max_beam_width=executor_config.max_beam_width,
        max_draft_len=spec_config.max_draft_len
        if spec_config is not None else 0,
        kv_cache_transceiver=kv_cache_transceiver,
        draft_model_engine=draft_model_engine,
        guided_decoder=guided_decoder,
        start_worker=start_worker,
        garbage_collection_gen0_threshold=garbage_collection_gen0_threshold)


def create_torch_sampler_args(executor_config: ExecutorConfig, mapping: Mapping,
                              *, max_seq_len: int, enable_mixed_sampler: bool):
    max_num_sequences = executor_config.max_batch_size * mapping.pp_size
    max_draft_len = (0 if executor_config.speculative_config is None else
                     executor_config.speculative_config.max_draft_len)
    return TorchSampler.Args(
        max_seq_len=max_seq_len,
        max_draft_len=max_draft_len,
        max_num_sequences=max_num_sequences,
        max_beam_width=executor_config.max_beam_width,
        enable_mixed_sampler=enable_mixed_sampler,
    )


def instantiate_sampler(engine: PyTorchModelEngine,
                        executor_config: ExecutorConfig,
                        pytorch_backend_config: PyTorchConfig,
                        mapping: Mapping):
    sampler_args = create_torch_sampler_args(
        executor_config,
        mapping,
        max_seq_len=engine.max_seq_len,
        enable_mixed_sampler=pytorch_backend_config.enable_mixed_sampler)
    if mapping.cp_config.get('cp_type') == 'star_attention':
        assert pytorch_backend_config.attn_backend == "FLASHINFER_STAR_ATTENTION", "attention backend of star attention should be 'FLASHINFER_STAR_ATTENTION'"
        return TorchSampler(sampler_args)
    if engine.spec_config is not None and engine.spec_config.spec_dec_mode.has_spec_decoder(
    ):
        return get_spec_decoder(sampler_args, engine.spec_config)
    if pytorch_backend_config.enable_trtllm_sampler:
        decoding_mode = get_decoding_mode(executor_config)
        return TRTLLMSampler(executor_config, engine.model, engine.dtype,
                             mapping, decoding_mode,
                             pytorch_backend_config.disable_overlap_scheduler)
    if not engine.model.model_config.is_generation:
        # NOTE: choose sampler based on model type
        return EarlyStopSampler()
    return TorchSampler(sampler_args)


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


def _adjust_torch_mem_fraction(pytorch_backend_config: PyTorchConfig):
    # FIXME: PyTorch only uses the garbage_collection_threshold setting
    #        if a memory fraction is set, cf.
    #   https://github.com/pytorch/pytorch/blob/cd995bfb2aac8891465809be3ce29543bd524287/c10/cuda/CUDACachingAllocator.cpp#L1357
    logger.debug("Setting PyTorch memory fraction to 1.0")
    torch.cuda.set_per_process_memory_fraction(1.0)

    # FIXME: As soon as
    #     torch.cuda._set_allocator_settings (added in PyTorch 2.8.0-rc1)
    #   or a similar API is available, the warning below should be removed
    #   and the allocator GC threshold be set via the new API instead.
    torch_allocator_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    torch_mem_threshold_advised = (
        torch.cuda.get_allocator_backend() == "native"
        and "expandable_segments:True" not in torch_allocator_config)
    torch_mem_threshold_set = "garbage_collection_threshold:" in torch_allocator_config
    if torch_mem_threshold_advised and not torch_mem_threshold_set:
        logger.warning(
            "It is recommended to incl. 'garbage_collection_threshold:0.???' or 'backend:cudaMallocAsync'"
            " or 'expandable_segments:True' in PYTORCH_CUDA_ALLOC_CONF.")

    # NOTE: Even if a memory threshold was not set (cf. warning above), setting a memory
    #       fraction < 1.0 is beneficial, because
    #         https://github.com/pytorch/pytorch/blob/5228986c395dc79f90d2a2b991deea1eef188260/c10/cuda/CUDACachingAllocator.cpp#L2719
    #       and
    #         https://github.com/pytorch/pytorch/blob/5228986c395dc79f90d2a2b991deea1eef188260/c10/cuda/CUDACachingAllocator.cpp#L1240
    #       lead PyTorch to release all unused memory before hitting the set fraction. This
    #       still mitigates OOM, although at a higher performance impact, because it
    #       effectively resets the allocator cache.
    if not pytorch_backend_config._limit_torch_cuda_mem_fraction:
        return
    mem_reserved = torch.cuda.memory_reserved()
    mem_free, mem_total = torch.cuda.mem_get_info()
    safety_margin = 32 * 1024**2
    mem_torch_max = mem_free + mem_reserved - safety_margin
    mem_torch_fraction = mem_torch_max / mem_total
    logger.info(
        f"Setting PyTorch memory fraction to {mem_torch_fraction} ({mem_torch_max / 1024**3} GiB)"
    )
    torch.cuda.set_per_process_memory_fraction(mem_torch_fraction)
