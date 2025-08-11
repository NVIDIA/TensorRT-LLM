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
                                       load_torch_hf_lora)
from tensorrt_llm.mapping import Mapping

from ..model_config import ModelConfig
from ..speculative import get_spec_decoder
from .config_utils import is_mla, is_nemotron_hybrid
from .kv_cache_transceiver import AttentionTypeCpp, create_kv_cache_transceiver
from .llm_request import ExecutorResponse
from .model_engine import (DRAFT_KV_CACHE_MANAGER_KEY, KV_CACHE_MANAGER_KEY,
                           PyTorchModelEngine)
from .py_executor import PyExecutor
from .resource_manager import (BlockKVCacheManager, KVCacheManager, MambaHybridCacheManager,
                               PeftCacheManager, ResourceManager)
from .sampler import (EarlyStopSampler, TorchSampler, TorchStarAttentionSampler,
                      TRTLLMSampler)
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
            head_dim = getattr(
                config,
                "head_dim",
                config.hidden_size // config.num_attention_heads,
            ) * num_key_value_heads // tp_size

        # provide at least 1 layer to prevent division by zero cache size
        num_hidden_layers = max(
            len(mapping.pp_layers(config.num_hidden_layers)), 1)
        mem_per_token *= num_hidden_layers * head_dim
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
                                     sampling_config=trtllm.SamplingConfig(),
                                     output_config=trtllm.OutputConfig(),
                                     end_id=-1)
            requests.append(request)
            remaining_tokens -= input_seq_len
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
        spec_cfg = executor_config.speculative_config
        if spec_cfg is not None:
            num_extra_tokens_per_seq += spec_cfg.max_draft_tokens
            num_extra_tokens_per_seq += spec_cfg.num_extra_kv_tokens
        for req in self._dummy_reqs:
            num_req_tokens = len(req.input_token_ids) + num_extra_tokens_per_seq
            # Requests cannot share KV cache blocks. Round up to nearest integer multiple of block size.
            num_cache_blocks += (num_req_tokens +
                                 executor_config.tokens_per_block -
                                 1) // executor_config.tokens_per_block
        return num_cache_blocks * executor_config.tokens_per_block

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
        """
        Perform KV cache capacity estimation.
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
            "kv_cache_manager").get_kv_cache_stats()

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
            if model_engine.pytorch_backend_config.enable_block_prediction:
                kv_cache_manager = BlockKVCacheManager(
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

        return kv_cache_manager

    def build_managers(self, resources: Dict) -> None:
        """Construct KV caches for model and draft model (if applicable)."""
        kv_cache_manager = self._create_kv_cache_manager(self._model_engine)
        draft_kv_cache_manager = self._create_kv_cache_manager(
            self._draft_model_engine
        ) if self._draft_model_engine is not None else None
        resources[KV_CACHE_MANAGER_KEY] = kv_cache_manager
        resources[DRAFT_KV_CACHE_MANAGER_KEY] = draft_kv_cache_manager

    def teardown_managers(self, resources: Dict) -> None:
        """Clean up KV caches for model and draft model (if applicable)."""
        resources[KV_CACHE_MANAGER_KEY].shutdown()
        del resources[KV_CACHE_MANAGER_KEY]
        draft_kv_cache_manager = resources[DRAFT_KV_CACHE_MANAGER_KEY]
        if draft_kv_cache_manager:
            draft_kv_cache_manager.shutdown()
        del resources[DRAFT_KV_CACHE_MANAGER_KEY]


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
        lora_config: Optional[LoraConfig] = None,
        garbage_collection_gen0_threshold: Optional[int] = None) -> PyExecutor:
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
        f"max_seq_len={executor_config.max_seq_len}, max_num_requests={executor_config.max_batch_size}, max_num_tokens={executor_config.max_num_tokens}, max_batch_size={executor_config.max_batch_size}"
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

    max_num_sequences = executor_config.max_batch_size * mapping.pp_size

    resources["seq_slot_manager"] = SeqSlotManager(max_num_sequences)

    resource_manager = ResourceManager(resources)

    # Make sure the kv cache manager is always invoked last as it could
    # depend on the results of other resource managers.
    if kv_cache_manager is not None:
        resource_manager.resource_managers.move_to_end("kv_cache_manager",
                                                       last=True)

    capacity_scheduler = BindCapacityScheduler(
        max_num_sequences,
        kv_cache_manager.impl if kv_cache_manager is not None else None,
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
        dist=dist,
        disable_overlap_scheduler=pytorch_backend_config.
        disable_overlap_scheduler,
        max_batch_size=executor_config.max_batch_size,
        max_draft_tokens=spec_config.max_draft_tokens
        if spec_config is not None else 0,
        kv_cache_transceiver=kv_cache_transceiver,
        draft_model_engine=draft_model_engine,
        start_worker=start_worker,
        garbage_collection_gen0_threshold=garbage_collection_gen0_threshold)


def instantiate_sampler(model_engine: PyTorchModelEngine,
                        executor_config: ExecutorConfig,
                        pytorch_backend_config: PyTorchConfig,
                        mapping: Mapping):
    if mapping.cp_config.get('cp_type') == 'star_attention':
        assert pytorch_backend_config.attn_backend == "FLASHINFER_STAR_ATTENTION", "attention backend of star attention should be 'FLASHINFER_STAR_ATTENTION'"
        sampler = TorchStarAttentionSampler(
            max_seq_len=model_engine.max_seq_len)
    elif model_engine.spec_config is not None and model_engine.spec_config.spec_dec_mode.has_spec_decoder(
    ):
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
