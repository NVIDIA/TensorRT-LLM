import os
import random
from typing import Dict, List, Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import \
    MODEL_CLASS_VISION_ENCODER_MAPPING
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import DecodingMode
from tensorrt_llm.llmapi.llm_args import (CacheTransceiverConfig,
                                          EagleDecodingConfig, KvCacheConfig,
                                          MTPDecodingConfig, PeftCacheConfig,
                                          SamplerType, SchedulerConfig,
                                          SparseAttentionConfig,
                                          SpeculativeConfig, TorchLlmArgs)
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_helper import (LoraConfig,
                                      get_default_trtllm_modules_to_hf_modules)
from tensorrt_llm.lora_manager import load_torch_lora
from tensorrt_llm.mapping import CpType, Mapping

from ..attention_backend import get_sparse_attn_kv_cache_manager
from ..model_config import ModelConfig
from ..speculative import get_num_extra_kv_tokens, get_spec_decoder
from .config import PyTorchConfig
from .config_utils import is_mla, is_nemotron_hybrid, is_qwen3_next
from .guided_decoder import GuidedDecoder
from .kv_cache_connector import KvCacheConnectorManager
from .kv_cache_transceiver import AttentionTypeCpp, create_kv_cache_transceiver
from .llm_request import ExecutorResponse
from .mamba_cache_manager import MambaHybridCacheManager
from .model_engine import PyTorchModelEngine
from .py_executor import PyExecutor
from .resource_manager import (KVCacheManager, PeftCacheManager,
                               ResourceManager, ResourceManagerType)
from .sampler import (EarlyStopSampler, EarlyStopWithMMResult, TorchSampler,
                      TRTLLMSampler)
from .scheduler import (BindCapacityScheduler, BindMicroBatchScheduler,
                        SimpleScheduler)
from .seq_slot_manager import SeqSlotManager

GB = 1 << 30


def get_kv_cache_manager_cls(model_config: ModelConfig):
    config = model_config.pretrained_config
    sparse_attn_config = model_config.sparse_attention_config
    if sparse_attn_config is not None:
        return get_sparse_attn_kv_cache_manager(sparse_attn_config)
    elif is_nemotron_hybrid(config) or is_qwen3_next(config):
        return MambaHybridCacheManager
    else:
        return KVCacheManager


class KvCacheCreator:
    """Groups together logic related to KV cache construction."""

    def __init__(
        self,
        *,
        model_engine: PyTorchModelEngine,
        draft_model_engine: Optional[PyTorchModelEngine],
        mapping: Mapping,
        net_max_seq_len: int,
        kv_connector_manager: Optional[KvCacheConnectorManager],
        max_num_tokens: int,
        max_beam_width: int,
        tokens_per_block: int,
        max_seq_len: int,
        max_batch_size: int,
        kv_cache_config: KvCacheConfig,
        pytorch_backend_config: PyTorchConfig,
        speculative_config: SpeculativeConfig,
        sparse_attention_config: SparseAttentionConfig,
        profiling_stage_data: Optional[dict],
    ):
        self._model_engine = model_engine
        self._draft_model_engine = draft_model_engine
        self._mapping = mapping
        self._kv_cache_config = kv_cache_config
        self._max_kv_tokens_in = self._kv_cache_config.max_tokens
        self._max_num_tokens = max_num_tokens
        self._max_beam_width = max_beam_width
        self._kv_connector_manager = kv_connector_manager
        self._pytorch_backend_config = pytorch_backend_config
        self._speculative_config = speculative_config
        self._sparse_attention_config = sparse_attention_config
        self._tokens_per_block = tokens_per_block
        self._max_seq_len = max_seq_len
        self._max_batch_size = max_batch_size
        self._net_max_seq_len = net_max_seq_len
        self._dummy_reqs = None
        self._profiling_stage_data = profiling_stage_data
        self._kv_cache_manager_cls = get_kv_cache_manager_cls(
            model_engine.model.model_config)

    def _get_kv_size_per_token(self):
        model_config = self._model_engine.model.model_config
        mapping = self._mapping
        kv_size_per_token = self._kv_cache_manager_cls.get_cache_size_per_token(
            model_config, mapping, tokens_per_block=self._tokens_per_block)
        if self._draft_model_engine is not None:
            draft_model_config = self._draft_model_engine.model.model_config
            kv_size_per_token += self._kv_cache_manager_cls.get_cache_size_per_token(
                draft_model_config,
                mapping,
                tokens_per_block=self._tokens_per_block)
        return kv_size_per_token

    def _cal_max_memory(self, peak_memory, total_gpu_memory, fraction,
                        allocated_bytes: int) -> int:
        """
        Calculate the max KV cache capacity.

        NOTE: `allocated_bytes` is the total KV-cache memory that must be pre-allocated during the estimation phase (for both the main and draft models) so the estimation run can complete successfully. When computing `available_kv_mem`, add this amount back in.
        """
        kv_size_per_token = self._get_kv_size_per_token()

        available_kv_mem = (total_gpu_memory - peak_memory +
                            allocated_bytes) * fraction
        logger.info(
            f"Peak memory during memory usage profiling (torch + non-torch): {peak_memory / (GB):.2f} GiB, "
            f"available KV cache memory when calculating max tokens: {available_kv_mem / (GB):.2f} GiB, "
            f"fraction is set {fraction}, kv size is {kv_size_per_token}. device total memory {total_gpu_memory / (GB):.2f} GiB, "
            f", tmp kv_mem { (allocated_bytes) / (GB):.2f} GiB")
        return int(available_kv_mem)

    def _create_dummy_mm_context_request(
            self, input_seq_len: int) -> List[trtllm.Request]:
        requests = []
        if isinstance(
                self._profiling_stage_data,
                dict) and not self._profiling_stage_data.get("enable_mm_reqs"):
            return requests

        input_processor = self._model_engine.input_processor
        if not (hasattr(input_processor, "get_dummy_prompt")):
            logger.warning("The input processor of the model does not have the method [get_dummy_prompt] implemented." \
            "Profiling with the default input dummy context request. This may not take into account the memory consumption of " \
            "the image encoder")
            return requests
        prompt = input_processor.get_dummy_prompt(input_seq_len)

        prompt_token_ids, extra_processed_inputs = self._model_engine.input_processor_with_hash(
            prompt, None)

        multimodal_input = extra_processed_inputs.get('multimodal_input')
        multimodal_data = extra_processed_inputs.get('multimodal_data')

        max_num_tokens = len(prompt_token_ids)
        assert max_num_tokens > 0, "the length of the prompt of the dummy mm req is less than or equal to 0"
        remaining_tokens = min(max_num_tokens, input_seq_len)
        if remaining_tokens > input_seq_len:
            logger.warning(f"Profiling with multimedia prompt which contains more tokens than the allowed input_seq_len. " \
                           f"Multimodal prompt has {remaining_tokens} while the input_seq_len is: {input_seq_len}")
        while remaining_tokens > 0:
            req_mm_input = trtllm.MultimodalInput(
                multimodal_hashes=multimodal_input.multimodal_hashes,
                multimodal_positions=multimodal_input.multimodal_positions,
                multimodal_lengths=multimodal_input.multimodal_lengths
            ) if multimodal_input else None
            request = trtllm.Request(prompt_token_ids,
                                     max_tokens=1,
                                     streaming=False,
                                     sampling_config=trtllm.SamplingConfig(
                                         beam_width=self._max_beam_width, ),
                                     output_config=trtllm.OutputConfig(),
                                     end_id=-1,
                                     multimodal_input=req_mm_input)
            # TODO:
            # create_input_processor_with_hash shouldn’t be required during profiling,
            # but is temporarily needed due to the multimodal input dependency for chunked prefill
            request.py_multimodal_data = multimodal_data
            remaining_tokens -= max_num_tokens
            requests.append(request)

        if self._mapping.enable_attention_dp:
            requests = requests * self._mapping.tp_size

        return requests

    def _create_dummy_context_requests(
            self, input_seq_len: int) -> List[trtllm.Request]:
        requests = []
        if hasattr(self._model_engine.model,
                   "original_arch") and MODEL_CLASS_VISION_ENCODER_MAPPING.get(
                       self._model_engine.model.original_arch, None):
            input_seq_len = min(self._max_num_tokens, input_seq_len)
            requests = self._create_dummy_mm_context_request(input_seq_len)
        # if succeed profiling with multimodal requests then return, otherwise profile
        # with default case
        if requests:
            return requests
        vocab_size = self._model_engine.model.model_config.pretrained_config.vocab_size
        max_num_tokens = self._max_num_tokens
        max_beam_width = self._max_beam_width

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
            if self._model_engine.use_mrope:
                request.py_multimodal_data = {
                    "mrope_config": {
                        "mrope_position_ids":
                        torch.zeros(3, 1, input_seq_len, dtype=torch.int32),
                        "mrope_position_deltas":
                        torch.zeros(1, 1, dtype=torch.int32)
                    }
                }
            requests.append(request)
            remaining_tokens -= input_seq_len
        if self._mapping.enable_attention_dp:
            requests = requests * self._mapping.tp_size
        return requests

    def _get_token_num_for_estimation(self) -> int:
        """Compute KV cache capacity required for estimate_max_kv_cache_tokens to succeed."""
        if 'cp_type' in self._mapping.cp_config:
            raise ValueError(
                "KV cache size estimation not supported with context parallelism."
            )
        # estimate_max_kv_cache_tokens submits self._dummy_reqs
        num_cache_blocks = 0
        num_extra_tokens_per_seq = 1  # account for generated tokens
        pytorch_backend_config = self._pytorch_backend_config
        spec_cfg = self._speculative_config
        if not pytorch_backend_config.disable_overlap_scheduler:
            num_extra_tokens_per_seq = num_extra_tokens_per_seq + 1
            if spec_cfg is not None:
                num_extra_tokens_per_seq += spec_cfg.max_total_draft_tokens

        if spec_cfg is not None:
            num_extra_tokens_per_seq += spec_cfg.max_total_draft_tokens
            num_extra_tokens_per_seq += get_num_extra_kv_tokens(spec_cfg)

        if self._dummy_reqs is None:
            self._dummy_reqs = self._create_dummy_context_requests(
                max(1, self._net_max_seq_len - 1))
        for req in self._dummy_reqs:
            num_req_tokens = len(req.input_token_ids) + num_extra_tokens_per_seq
            # Requests cannot share KV cache blocks. Round up to nearest integer multiple of block size.
            num_cache_blocks += (num_req_tokens + self._tokens_per_block -
                                 1) // self._tokens_per_block
        # Multiply by beam width, to prevent rescaling of the max_seq_len caused by the influence of beam width during the preparation for kv_cache_estimation
        return num_cache_blocks * self._tokens_per_block * self._dummy_reqs[
            0].sampling_config.beam_width

    def try_prepare_estimation(self) -> bool:
        """Prepare for possible KV cache capacity estimation.

        This updates `kv_cache_config` and returns a boolean indicating whether KV cache
        estimation is to be performend.
        """
        estimating_kv_cache = False
        if 'cp_type' not in self._mapping.cp_config:
            estimating_kv_cache = True
            self._kv_cache_config.max_tokens = self._get_token_num_for_estimation(
            )
        model_config = self._model_engine.model.model_config
        if model_config.attn_backend == "VANILLA":
            logger.info(
                "KV cache size estimation is not supported for Vanilla attention backend, disable it."
            )
            estimating_kv_cache = False
        return estimating_kv_cache

    def configure_kv_cache_capacity(self, py_executor: PyExecutor) -> None:
        """Perform KV cache capacity estimation.
        NOTE: for VSWA case, we calculate and set kv cache memory instead of using max_tokens in kv_cache_config.

        This updates `kv_cache_config`.
        """
        mapping = self._mapping

        # TODO: support CP by generating dummy requests for it.
        assert 'cp_type' not in mapping.cp_config

        fraction = self._kv_cache_config.free_gpu_memory_fraction

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
            py_executor.is_warmup = False
            py_executor.shutdown()
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

        # get kv cache stats for both model and draft model
        kv_stats = py_executor.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER).get_kv_cache_stats()
        kv_stats_draft = py_executor.resource_manager.resource_managers.get(
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER).get_kv_cache_stats(
            ) if self._draft_model_engine is not None else None

        # get total allocated bytes
        allocated_bytes = kv_stats.allocated_bytes + (
            kv_stats_draft.allocated_bytes if kv_stats_draft is not None else 0)

        # calculate max memory from peak memory and free gpu memory fraction
        kv_cache_max_memory = self._cal_max_memory(peak_memory,
                                                   total_gpu_memory, fraction,
                                                   allocated_bytes)

        max_attention_window = self._kv_cache_config.max_attention_window
        is_vswa = max_attention_window and len(set(max_attention_window)) > 1

        # NOTE:
        # KvCacheCreator currently controls KV-cache capacity using two parameters in KVCacheConfig:
        #   • max_tokens
        #   • max_gpu_total_bytes
        # Ideally, the internal logic would rely solely on max_gpu_total_bytes,
        # leaving max_tokens as a user-defined constraint.

        # ---------------------------handle max_tokens---------------------------------
        # if user provided max_tokens, calculate max memory from max_tokens
        if self._max_kv_tokens_in is not None:
            # raise error if it is VSWA case
            if is_vswa:
                logger.warning(
                    "max_tokens should not be set for VSWA case as it is ambiguous concept for VSWA."
                )
            # calculate max memory from max_tokens
            kv_cache_max_memory_from_max_tokens = self._max_kv_tokens_in * self._get_kv_size_per_token(
            )
            kv_cache_max_memory = min(kv_cache_max_memory,
                                      kv_cache_max_memory_from_max_tokens)
            logger.info(
                f"max_tokens={self._max_kv_tokens_in} is provided, max_memory is set to {kv_cache_max_memory / (GB):.2f} GiB"
            )
        # For KvCacheManager, its logic still relies on max_tokens, need to improve in the future.
        self._kv_cache_config.max_tokens = int(kv_cache_max_memory //
                                               self._get_kv_size_per_token())
        # ---------------------------handle max_tokens---------------------------------

        # ---------------------------handle max_gpu_total_bytes---------------------------------
        # if user provided max_gpu_total_bytes, set max memory from max_gpu_total_bytes
        if self._kv_cache_config.max_gpu_total_bytes > 0:
            kv_cache_max_memory = min(kv_cache_max_memory,
                                      self._kv_cache_config.max_gpu_total_bytes)
            logger.info(
                f"max_gpu_total_bytes={self._kv_cache_config.max_gpu_total_bytes / (GB):.2f} GiB is provided, max_memory is set to {kv_cache_max_memory / (GB):.2f} GiB"
            )

        logger.info(
            f"Estimated max memory in KV cache : {kv_cache_max_memory / (GB):.2f} GiB"
        )
        # set max_gpu_total_bytes
        self._kv_cache_config.max_gpu_total_bytes = kv_cache_max_memory
        if isinstance(self._profiling_stage_data, dict):
            self._profiling_stage_data["activation_bytes"] = activation_bytes
        # ---------------------------handle max_gpu_total_bytes---------------------------------

    def _create_kv_cache_manager(
            self,
            model_engine: PyTorchModelEngine,
            estimating_kv_cache: bool = False) -> KVCacheManager:
        mapping = self._mapping
        assert model_engine.model.model_config.is_generation, "Only construct KV cache for generation models."

        config = model_engine.model.model_config.pretrained_config
        quant_config = model_engine.model.model_config.quant_config
        spec_config = self._speculative_config
        sparse_attn_config = self._sparse_attention_config

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
        elif quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache(
        ):
            kv_cache_dtype = tensorrt_llm.bindings.DataType.NVFP4
        else:
            kv_cache_dtype = str_dtype_to_binding(
                torch_dtype_to_str(model_engine.dtype))

        num_hidden_layers = config.num_hidden_layers

        if is_mla(config):
            kv_cache_manager = self._kv_cache_manager_cls(
                self._kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.
                SELFKONLY,
                num_layers=num_hidden_layers,
                num_kv_heads=1,
                head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
                tokens_per_block=self._tokens_per_block,
                max_seq_len=self._max_seq_len,
                max_batch_size=self._max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                spec_config=spec_config,
                max_beam_width=self._max_beam_width,
                is_draft=model_engine.is_draft_model,
                kv_connector_manager=self._kv_connector_manager
                if not estimating_kv_cache else None,
                sparse_attn_config=sparse_attn_config,
            )
        elif is_nemotron_hybrid(config):
            if self._max_beam_width > 1:
                raise ValueError(
                    "MambaHybridCacheManager + beam search is not supported yet."
                )

            if not estimating_kv_cache and self._kv_connector_manager is not None:
                raise NotImplementedError(
                    "Connector manager is not supported for MambaHybridCacheManager."
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
            kv_cache_manager = self._kv_cache_manager_cls(
                # mamba cache parameters
                config.ssm_state_size,
                config.conv_kernel,
                config.mamba_num_heads,
                config.n_groups,
                config.mamba_head_dim,
                mamba_num_layers,
                mamba_layer_mask,
                config.torch_dtype,
                model_engine.model.model_config.quant_config.
                mamba_ssm_cache_dtype,
                # kv cache parameters
                self._kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
                num_layers=num_layers,
                layer_mask=layer_mask,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                tokens_per_block=self._tokens_per_block,
                max_seq_len=self._max_seq_len,
                max_batch_size=self._max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                spec_config=spec_config,
            )
        elif is_qwen3_next(config):
            if self._max_beam_width > 1:
                raise ValueError(
                    "MambaHybridCacheManager + beam search is not supported yet."
                )

            if not estimating_kv_cache and self._kv_connector_manager is not None:
                raise NotImplementedError(
                    "Connector manager is not supported for MambaHybridCacheManager."
                )
            config = model_engine.model.model_config.pretrained_config
            mamba_layer_mask = [
                True if i % config.full_attention_interval
                != config.full_attention_interval - 1 else False
                for i in range(num_hidden_layers)
            ]
            layer_mask = [
                False if i % config.full_attention_interval
                != config.full_attention_interval - 1 else True
                for i in range(num_hidden_layers)
            ]
            num_mamba_layers = num_hidden_layers // config.full_attention_interval * (
                config.full_attention_interval - 1)
            num_layers = num_hidden_layers - num_mamba_layers
            kv_cache_manager = self._kv_cache_manager_cls(
                # mamba cache parameters
                config.linear_key_head_dim,
                config.linear_conv_kernel_dim,
                config.linear_num_value_heads,
                config.linear_num_key_heads,
                config.linear_value_head_dim,
                num_mamba_layers,
                mamba_layer_mask,
                config.torch_dtype,
                model_engine.model.model_config.quant_config.
                mamba_ssm_cache_dtype,
                # kv cache parameters
                self._kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
                num_layers=num_layers,
                layer_mask=layer_mask,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                tokens_per_block=self._tokens_per_block,
                max_seq_len=self._max_seq_len,
                max_batch_size=self._max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                spec_config=spec_config,
            )
        else:
            # NOTE: this is a workaround for VSWA to switch to calculate_max_num_blocks_from_cpp in KVCahceManager
            is_vswa = self._kv_cache_config.max_attention_window is not None and len(
                set(self._kv_cache_config.max_attention_window)) > 1
            binding_model_config = model_engine.model.model_config.get_bindings_model_config(
                tokens_per_block=self._tokens_per_block) if is_vswa else None

            kv_cache_manager = self._kv_cache_manager_cls(
                self._kv_cache_config,
                tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
                num_layers=num_hidden_layers,
                num_kv_heads=num_key_value_heads,
                head_dim=head_dim,
                tokens_per_block=self._tokens_per_block,
                max_seq_len=self._max_seq_len,
                max_batch_size=self._max_batch_size,
                mapping=mapping,
                dtype=kv_cache_dtype,
                spec_config=spec_config,
                max_num_tokens=self._max_num_tokens,
                model_config=binding_model_config,
                max_beam_width=self._max_beam_width,
                is_draft=model_engine.is_draft_model,
                kv_connector_manager=self._kv_connector_manager
                if not estimating_kv_cache else None,
                sparse_attn_config=sparse_attn_config,
            )
        # KVCacheManager (Non-draft) modifies the max_seq_len field, update it to self
        if model_engine.kv_cache_manager_key == ResourceManagerType.KV_CACHE_MANAGER:
            # When SWA is enabled, max_seq_len is updated inside kv_cache_manager.
            if kv_cache_manager is not None:
                if kv_cache_manager.max_seq_len < self._max_seq_len:
                    self._dummy_reqs = self._create_dummy_context_requests(
                        max(
                            1, self._net_max_seq_len - 1 -
                            (self._max_seq_len - kv_cache_manager.max_seq_len)))
                self._max_seq_len = kv_cache_manager.max_seq_len

        # When SWA is enabled, max_seq_len is updated inside kv_cache_manager.
        if kv_cache_manager is not None:
            if kv_cache_manager.max_seq_len < self._max_seq_len:
                self._dummy_reqs = self._create_dummy_context_requests(
                    max(1, kv_cache_manager.max_seq_len - 1))
            self._max_seq_len = kv_cache_manager.max_seq_len

        return kv_cache_manager

    def build_managers(self,
                       resources: Dict,
                       estimating_kv_cache: bool = False) -> None:
        """Construct KV caches for model and draft model (if applicable)."""
        kv_cache_manager = self._create_kv_cache_manager(
            self._model_engine, estimating_kv_cache)

        if not estimating_kv_cache and self._kv_connector_manager is not None and self._draft_model_engine is not None:
            raise NotImplementedError(
                "Connector manager is not supported for draft model.")

        draft_kv_cache_manager = self._create_kv_cache_manager(
            self._draft_model_engine, estimating_kv_cache
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
    ctx_chunk_config,
    model_engine,
    start_worker,
    sampler,
    drafter,
    guided_decoder: Optional[GuidedDecoder] = None,
    lora_config: Optional[LoraConfig] = None,
    garbage_collection_gen0_threshold: Optional[int] = None,
    kv_connector_manager: Optional[KvCacheConnectorManager] = None,
    max_seq_len: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    max_beam_width: Optional[int] = None,
    max_num_tokens: Optional[int] = None,
    peft_cache_config: Optional[PeftCacheConfig] = None,
    scheduler_config: Optional[SchedulerConfig] = None,
    cache_transceiver_config: Optional[CacheTransceiverConfig] = None,
) -> PyExecutor:
    kv_cache_manager = resources.get(ResourceManagerType.KV_CACHE_MANAGER, None)

    spec_config = model_engine.spec_config

    logger.info(
        f"max_seq_len={max_seq_len}, max_num_requests={max_batch_size}, max_num_tokens={max_num_tokens}, max_batch_size={max_batch_size}"
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

        num_kv_attention_heads_per_layer = model_binding_config.num_kv_heads_per_layer
        if max(num_kv_attention_heads_per_layer) != min(
                num_kv_attention_heads_per_layer):
            logger.warning(
                "Defining LORA with per-layer KV heads is not supported for LORA, using the max number of KV heads per layer"
            )
            num_kv_attention_heads = max(num_kv_attention_heads_per_layer)
        else:
            # all layers have the same number of KV heads
            num_kv_attention_heads = num_kv_attention_heads_per_layer[0]

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

        peft_cache_config_model = PeftCacheConfig(
        ) if peft_cache_config is None else peft_cache_config
        if lora_config.max_loras is not None:
            peft_cache_config_model.num_device_module_layer = \
                max_lora_rank * num_lora_modules * lora_config.max_loras
        if lora_config.max_cpu_loras is not None:
            peft_cache_config_model.num_host_module_layer = \
                max_lora_rank * num_lora_modules * lora_config.max_cpu_loras

        from tensorrt_llm.bindings import WorldConfig
        world_config = WorldConfig(
            tensor_parallelism=mapping.tp_size,
            pipeline_parallelism=mapping.pp_size,
            context_parallelism=mapping.cp_size,
            rank=dist.mapping.rank,
            gpus_per_node=dist.mapping.gpus_per_node,
        )
        peft_cache_manager = PeftCacheManager(
            peft_cache_config=peft_cache_config_model,
            lora_config=lora_config,
            model_config=model_binding_config,
            world_config=world_config,
        )
        resources[ResourceManagerType.PEFT_CACHE_MANAGER] = peft_cache_manager
        model_engine.set_lora_model_config(
            lora_config.lora_target_modules,
            lora_config.trtllm_modules_to_hf_modules,
            lora_config.swap_gate_up_proj_lora_b_weight)

    max_num_sequences = max_batch_size * mapping.pp_size

    resources[ResourceManagerType.SEQ_SLOT_MANAGER] = SeqSlotManager(
        max_num_sequences)

    resource_manager = ResourceManager(resources)

    # Make sure the kv cache manager is always invoked last as it could
    # depend on the results of other resource managers.
    if kv_cache_manager is not None:
        resource_manager.resource_managers.move_to_end(
            ResourceManagerType.KV_CACHE_MANAGER, last=True)

    # When scheduler_capacity == 1, attention dp dummy request will prevent the scheduling of DISAGG_GENERATION_INIT.
    # Enlarge scheduler capacity to avoid DISAGG_GENERATION_INIT stuck in the scheduler.
    scheduler_capacity = max_num_sequences
    if scheduler_capacity == 1 and mapping.enable_attention_dp and kv_cache_manager:
        scheduler_capacity += 1

    capacity_scheduler = BindCapacityScheduler(
        scheduler_capacity,
        kv_cache_manager.impl if kv_cache_manager is not None else None,
        peft_cache_manager.impl if peft_cache_manager is not None else None,
        scheduler_config.capacity_scheduler_policy,
        two_step_lookahead=mapping.has_pp())
    mb_scheduler = BindMicroBatchScheduler(max_batch_size, max_num_tokens,
                                           ctx_chunk_config)
    scheduler = SimpleScheduler(capacity_scheduler, mb_scheduler)

    config = model_engine.model.model_config.pretrained_config
    attention_type = AttentionTypeCpp.MLA if is_mla(
        config) else AttentionTypeCpp.DEFAULT
    kv_cache_transceiver = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager, attention_type,
        cache_transceiver_config)
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
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        max_draft_len=spec_config.max_draft_len
        if spec_config is not None else 0,
        max_total_draft_tokens=spec_config.max_total_draft_tokens
        if spec_config is not None else 0,
        kv_cache_transceiver=kv_cache_transceiver,
        guided_decoder=guided_decoder,
        start_worker=start_worker,
        garbage_collection_gen0_threshold=garbage_collection_gen0_threshold,
        kv_connector_manager=kv_connector_manager,
        max_seq_len=max_seq_len,
        peft_cache_config=peft_cache_config)


def create_torch_sampler_args(mapping: Mapping, *, max_seq_len: int,
                              max_batch_size: int,
                              speculative_config: SpeculativeConfig,
                              max_beam_width: int):
    max_num_sequences = max_batch_size * mapping.pp_size
    max_draft_len = (0 if speculative_config is None else
                     speculative_config.max_draft_len)
    max_total_draft_tokens = (0 if speculative_config is None else
                              speculative_config.max_total_draft_tokens)

    return TorchSampler.Args(
        max_seq_len=max_seq_len,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        max_num_sequences=max_num_sequences,
        max_beam_width=max_beam_width,
    )


def instantiate_sampler(engine: PyTorchModelEngine,
                        pytorch_backend_config: PyTorchConfig, mapping: Mapping,
                        max_batch_size: int, max_beam_width: int,
                        max_seq_len: int, mm_encoder_only: bool,
                        speculative_config: SpeculativeConfig,
                        decoding_config: trtllm.DecodingConfig,
                        kv_cache_config: KvCacheConfig):
    sampler_args = create_torch_sampler_args(
        mapping,
        max_seq_len=engine.max_seq_len,
        max_batch_size=max_batch_size,
        speculative_config=speculative_config,
        max_beam_width=max_beam_width)
    decoding_mode = get_decoding_mode(decoding_config=decoding_config,
                                      max_beam_width=max_beam_width)
    if mapping.cp_config.get('cp_type') == CpType.STAR:
        assert pytorch_backend_config.attn_backend == "FLASHINFER_STAR_ATTENTION", "attention backend of star attention should be 'FLASHINFER_STAR_ATTENTION'"
        return TorchSampler(sampler_args)
    if engine.spec_config is not None and engine.spec_config.spec_dec_mode.has_spec_decoder(
    ):
        return get_spec_decoder(sampler_args, engine.spec_config)

    if mm_encoder_only:
        # NOTE: handle model outputs specially for mm encoder executor/engine
        return EarlyStopWithMMResult()
    if pytorch_backend_config.sampler_type == SamplerType.TRTLLMSampler or (
            pytorch_backend_config.sampler_type == SamplerType.auto
            and decoding_mode.isBeamSearch()):
        logger.debug(f"DecodingMode: {decoding_mode.name}")
        return TRTLLMSampler(engine.model,
                             engine.dtype,
                             mapping,
                             decoding_mode,
                             pytorch_backend_config.disable_overlap_scheduler,
                             max_seq_len=max_seq_len,
                             max_batch_size=max_batch_size,
                             max_beam_width=max_beam_width,
                             decoding_config=decoding_config,
                             kv_cache_config=kv_cache_config)
    if not engine.model.model_config.is_generation:
        # NOTE: choose sampler based on model type
        return EarlyStopSampler()
    return TorchSampler(sampler_args)


def get_decoding_mode(
    decoding_config: trtllm.DecodingConfig,
    max_beam_width: int,
) -> DecodingMode:
    '''This implementation is based off trtGptModelInflightBatching.cpp getDecodingMode().'''
    if decoding_config and decoding_config.decoding_mode and not decoding_config.decoding_mode.isAuto(
    ):
        decoding_mode = decoding_config.decoding_mode
    elif max_beam_width == 1:
        decoding_mode = DecodingMode.TopKTopP()
    else:
        decoding_mode = DecodingMode.BeamSearch()

    # Override decoding mode when beam width is one
    if max_beam_width == 1 and decoding_mode.isBeamSearch():
        logger.warning(
            "Beam width is set to 1, but decoding mode is BeamSearch. Overwriting decoding mode to TopKTopP."
        )
        decoding_mode = DecodingMode.TopKTopP()

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


def validate_feature_combination(llm_args, model_engine, sampler_type):
    # Validate the flags for features' combination
    def init_feature_status(llm_args) -> Dict[str, bool]:
        assert isinstance(
            llm_args, TorchLlmArgs
        ), "Expect TorchLlmArgs used for feature status validation."
        feature_list = [
            "overlap_scheduler",
            "cuda_graph",
            "attention_dp",
            "disaggregated_serving",
            "chunked_prefill",
            "mtp",
            "eagle3_one_model",
            "eagle3_two_model",
            "torch_sampler",
            "trtllm_sampler",
            "kv_cache_reuse",
            "slide_window_attention",
            "guided_decoding",
        ]
        feature_status: Dict[str, bool] = dict.fromkeys(feature_list)
        feature_status[
            "overlap_scheduler"] = not llm_args.disable_overlap_scheduler
        feature_status["cuda_graph"] = llm_args.cuda_graph_config is not None
        feature_status["attention_dp"] = llm_args.enable_attention_dp
        feature_status[
            "disaggregated_serving"] = llm_args.cache_transceiver_config is not None
        feature_status["chunked_prefill"] = llm_args.enable_chunked_prefill
        feature_status["mtp"] = (
            isinstance(llm_args.speculative_config, MTPDecodingConfig)
            and llm_args.speculative_config.num_nextn_predict_layers > 0)
        feature_status["eagle3_one_model"] = (
            isinstance(llm_args.speculative_config, EagleDecodingConfig)
            and llm_args.speculative_config.eagle3_one_model)
        feature_status["eagle3_two_model"] = (
            isinstance(llm_args.speculative_config, EagleDecodingConfig)
            and not llm_args.speculative_config.eagle3_one_model)
        feature_status[
            "torch_sampler"] = sampler_type == SamplerType.TorchSampler
        feature_status[
            "trtllm_sampler"] = sampler_type == SamplerType.TRTLLMSampler

        feature_status[
            "kv_cache_reuse"] = llm_args.kv_cache_config is not None and llm_args.kv_cache_config.enable_block_reuse
        feature_status["slide_window_attention"] = (
            hasattr(model_engine.model.model_config.pretrained_config,
                    "layer_types") and "sliding_attention"
            in model_engine.model.model_config.pretrained_config.layer_types)
        feature_status[
            "guided_decoding"] = llm_args.guided_decoding_backend is not None
        assert all(v is not None for v in feature_status.values()
                   ), "feature status has not been fully initialized."
        assert all(
            k in feature_list for k in
            feature_status.keys()), "unexpected feature type in feature_status."
        return feature_status

    feature_status: Dict[str, bool] = init_feature_status(llm_args)

    ERR_MSG_TMPL = "{feature1} and {feature2} enabled together is not supported yet."

    CONFLICT_RULES = [
        {
            "features": ["mtp", "slide_window_attention"],
            "message":
            ERR_MSG_TMPL.format(feature1="mtp",
                                feature2="slide_window_attention")
        },
        {
            "features": ["trtllm_sampler", "mtp"],
            "message":
            ERR_MSG_TMPL.format(feature1="trtllm_sampler", feature2="mtp") +
            " Please use sampler type auto instead."
        },
        {
            "features": ["trtllm_sampler", "eagle3_one_model"],
            "message":
            ERR_MSG_TMPL.format(feature1="trtllm_sampler",
                                feature2="eagle3_one_model") +
            " Please use sampler type auto instead."
        },
        {
            "features": ["trtllm_sampler", "eagle3_two_model"],
            "message":
            ERR_MSG_TMPL.format(feature1="trtllm_sampler",
                                feature2="eagle3_two_model") +
            " Please use sampler type auto instead."
        },
        # Add new conflict rules here in the future
    ]
    for rule in CONFLICT_RULES:
        if all(feature_status[feature] for feature in rule["features"]):
            raise ValueError(rule["message"])
