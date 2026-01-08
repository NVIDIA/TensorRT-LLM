import bisect
import contextlib
import functools
import gc
import inspect
import math
import os
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch._dynamo.config

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._utils import (is_trace_enabled, nvtx_range, release_gc,
                                 torch_dtype_to_str, trace_func)
from tensorrt_llm.inputs.multimodal import (MultimodalParams,
                                            MultimodalRuntimeData)
from tensorrt_llm.inputs.registry import (create_input_processor,
                                          create_input_processor_with_hash)
from tensorrt_llm.llmapi.llm_args import (CudaGraphConfig, TorchCompileConfig,
                                          TorchLlmArgs)
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.lora_manager import LoraModelConfig
from tensorrt_llm.mapping import CpType, Mapping

from ..attention_backend.interface import (AttentionMetadata,
                                           AttentionRuntimeFeatures)
from ..attention_backend.trtllm import TrtllmAttentionMetadata
from ..attention_backend.utils import get_attention_backend
from ..attention_backend.vanilla import VanillaAttentionMetadata
from ..autotuner import AutoTuner, autotune
from ..compilation.backend import Backend
from ..compilation.utils import capture_piecewise_cuda_graph
from ..distributed import MPIDist
from ..distributed.communicator import init_pp_comm
from ..expert_statistic import ExpertStatistic
from ..memory_buffer_utils import with_shared_pool
from ..metadata import KVCacheParams
from ..models.modeling_multimodal_utils import filter_mm_token_from_input_ids
from ..models.modeling_utils import DecoderModelForCausalLM
from ..modules.fused_moe.moe_load_balancer import (MoeLoadBalancer,
                                                   MoeLoadBalancerIterContext)
from ..speculative import (SpecMetadata, get_num_extra_kv_tokens,
                           get_spec_metadata,
                           update_spec_config_from_model_config)
from ..speculative.drafting_loops import BaseDraftingLoopWrapper
from ..speculative.eagle3 import (Eagle3OneModelSpecMetadata,
                                  Eagle3ResourceManager, Eagle3SpecMetadata)
from ..speculative.mtp import SampleStateTensorsMTP
from ..speculative.utils import SpecDecodingTensor
from ..utils import (get_model_extra_attrs,
                     set_per_request_piecewise_cuda_graph_flag,
                     set_torch_compiling, with_model_extra_attrs)
from .config_utils import is_mla
from .cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from .guided_decoder import CapturableGuidedDecoder
from .layerwise_nvtx_marker import LayerwiseNvtxMarker
from .llm_request import LlmRequest, get_draft_token_length
from .model_loader import ModelLoader, _construct_checkpoint_loader
from .resource_manager import (BaseResourceManager, KVCacheManager,
                               ResourceManager, ResourceManagerType)
from .sampler import SampleStateTensors
from .scheduler import ScheduledRequests


class ModelEngine(ABC):

    @abstractmethod
    def get_max_num_sequences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tensors_device: Optional[SampleStateTensors],
                gather_context_logits: bool = False,
                cache_indirection_buffer: Optional[torch.Tensor] = None,
                num_accepted_tokens_device: Optional[torch.Tensor] = None):
        raise NotImplementedError

    def warmup(self, resource_manager: ResourceManager) -> None:
        """
        This method is called after the KV cache manager is initialized
        inside the given resource manager. Override to perform any
        warmup actions: instantiating CUDA graphs, running torch.compile, etc.
        """
        return


def _filter_cuda_graph_batch_sizes(cuda_graph_batch_sizes: list[int],
                                   max_batch_size: int, max_num_tokens: int,
                                   max_total_draft_tokens: int,
                                   enable_padding: bool) -> list[int]:
    # This is the largest possible batch size for a pure decoding batch.
    max_cuda_graph_bs = min(max_batch_size,
                            int(max_num_tokens / (1 + max_total_draft_tokens)))

    result = []
    # This function assumes cuda_graph_batch_sizes is sorted
    for i, bs in enumerate(cuda_graph_batch_sizes):
        if bs <= max_cuda_graph_bs:
            result.append(bs)
        else:
            # One extra special case for padding. The user gave us at least
            # one batch size to pad to which is larger than the executor's max
            # batch size. In this case, padding to max_cuda_graph_bs is acceptable. The logic
            # is that if the user is OK padding to a batch size B, they should also
            # be OK with padding to some size B' < B since the performance will generally
            # just be better in the smaller case.
            if enable_padding and (i == 0
                                   or result[i - 1] != max_cuda_graph_bs):
                logger.warning(
                    "CUDA graph padding is enabled, but one of the given CUDA graph "
                    f"batch sizes ({bs}) is larger than the executor's max batch size "
                    f"({max_cuda_graph_bs}). We will pad batches to {max_cuda_graph_bs}."
                )
                result.append(max_cuda_graph_bs)
            break

    return result


class PyTorchModelEngine(ModelEngine):

    def __init__(
        self,
        *,
        model_path: str,
        llm_args: TorchLlmArgs,
        mapping: Optional[Mapping] = None,
        attn_runtime_features: Optional[AttentionRuntimeFeatures] = None,
        dist: Optional[MPIDist] = None,
        spec_config: Optional["DecodingBaseConfig"] = None,
        is_draft_model: bool = False,
        drafting_loop_wrapper: Optional[Callable[[torch.nn.Module],
                                                 torch.nn.Module]] = None,
        model: Optional[torch.nn.Module] = None,
    ):
        self.forward_pass_callable = None
        self.ub_buffers = None
        (
            max_beam_width,
            max_num_tokens,
            max_seq_len,
            max_batch_size,
        ) = llm_args.get_runtime_sizes()

        self.batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_seq_len = max_seq_len
        self.max_beam_width = max_beam_width

        checkpoint_loader = _construct_checkpoint_loader(
            llm_args.backend, llm_args.checkpoint_loader,
            llm_args.checkpoint_format)

        self.mapping = mapping
        if mapping.has_pp():
            init_pp_comm(mapping)
        self.dist = dist
        if dist is not None:
            ExpertStatistic.create(self.dist.rank)
        self.llm_args = llm_args
        self.original_max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self.original_max_total_draft_tokens = spec_config.max_total_draft_tokens if spec_config is not None else 0

        # The draft model won't have any draft tokens attached to
        # generation requests when we invoke it autoregressively
        if spec_config is not None and is_draft_model:
            spec_config.max_draft_len = 0
            spec_config.max_total_draft_tokens = 0
        self.spec_config = spec_config
        self.is_spec_decode = spec_config is not None
        self.sparse_attention_config = None if is_draft_model else llm_args.sparse_attention_config
        self.enable_spec_decode = self.is_spec_decode
        self.is_draft_model = is_draft_model

        self.attn_runtime_features = attn_runtime_features or AttentionRuntimeFeatures(
        )

        self.input_processor = create_input_processor(
            model_path,
            tokenizer=None,
            checkpoint_format=llm_args.checkpoint_format)
        self.input_processor_with_hash = create_input_processor_with_hash(
            self.input_processor)
        if model is None:
            lora_config: Optional[
                LoraConfig] = None if is_draft_model else llm_args.lora_config
            # Keep the model_loader to support reloading the model weights later
            self.model_loader = ModelLoader(
                llm_args=llm_args,
                mapping=self.mapping,
                spec_config=self.spec_config,
                sparse_attention_config=self.sparse_attention_config,
                max_num_tokens=self.max_num_tokens,
                max_seq_len=self.max_seq_len,
                lora_config=lora_config,
            )
            self.model, moe_load_balancer = self.model_loader.load(
                checkpoint_dir=model_path, checkpoint_loader=checkpoint_loader)
            if isinstance(moe_load_balancer, MoeLoadBalancer):
                setattr(self, "moe_load_balancer", moe_load_balancer)
        else:
            self.model = model
        if drafting_loop_wrapper is not None:
            self.model = drafting_loop_wrapper(self.model)
            self.model_is_wrapped = True
        else:
            self.model_is_wrapped = False
        self.sparse_attention_config = self.model.model_config.sparse_attention_config
        # In case that some tests use stub models and override `_load_model`.
        if not hasattr(self.model, 'extra_attrs'):
            self.model.extra_attrs = {}
        if self.llm_args.enable_layerwise_nvtx_marker:
            layerwise_nvtx_marker = LayerwiseNvtxMarker()
            module_prefix = 'Model'
            if self.model.model_config and self.model.model_config.pretrained_config and self.model.model_config.pretrained_config.architectures:
                module_prefix = '|'.join(
                    self.model.model_config.pretrained_config.architectures)
            layerwise_nvtx_marker.register_hooks(self.model, module_prefix)

        self.enable_attention_dp = self.model.model_config.mapping.enable_attention_dp
        self._disable_overlap_scheduler = self.llm_args.disable_overlap_scheduler
        self._torch_compile_backend = None
        self.dtype = self.model.config.torch_dtype
        self._init_model_capacity()

        self.cuda_graph_config = self.llm_args.cuda_graph_config
        cuda_graph_batch_sizes = self.cuda_graph_config.batch_sizes if self.cuda_graph_config else CudaGraphConfig.model_fields[
            'batch_sizes'].default
        cuda_graph_padding_enabled = self.cuda_graph_config.enable_padding if self.cuda_graph_config else CudaGraphConfig.model_fields[
            'enable_padding'].default

        self.torch_compile_config = self.llm_args.torch_compile_config
        torch_compile_enabled = bool(self.torch_compile_config is not None)
        torch_compile_fullgraph = self.torch_compile_config.enable_fullgraph if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_fullgraph'].default
        torch_compile_inductor_enabled = self.torch_compile_config.enable_inductor if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_inductor'].default
        torch_compile_piecewise_cuda_graph = self.torch_compile_config.enable_piecewise_cuda_graph if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_piecewise_cuda_graph'].default
        torch_compile_piecewise_cuda_graph_num_tokens = self.torch_compile_config.capture_num_tokens if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'capture_num_tokens'].default
        torch_compile_enable_userbuffers = self.torch_compile_config.enable_userbuffers if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_userbuffers'].default
        torch_compile_max_num_streams = self.torch_compile_config.max_num_streams if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'max_num_streams'].default

        # Eagle3 draft model now does not support torch.compile
        self._torch_compile_enabled = torch_compile_enabled
        self._torch_compile_piecewise_cuda_graph = torch_compile_piecewise_cuda_graph

        piecewise_cuda_graph_num_tokens = (
            torch_compile_piecewise_cuda_graph_num_tokens
            or cuda_graph_batch_sizes or [])

        self._piecewise_cuda_graph_num_tokens = [
            i for i in piecewise_cuda_graph_num_tokens
            if i <= self.max_num_tokens
        ]

        try:
            use_ub_for_nccl = (
                self.llm_args.allreduce_strategy == "NCCL_SYMMETRIC"
                and self._init_userbuffers(self.model.config.hidden_size))
            if self._torch_compile_enabled:
                set_torch_compiling(True)
                use_ub = not use_ub_for_nccl and (
                    torch_compile_enable_userbuffers
                    and self._init_userbuffers(self.model.config.hidden_size))
                self.backend_num_streams = Backend.Streams([
                    torch.cuda.Stream()
                    for _ in range(torch_compile_max_num_streams - 1)
                ])
                self._torch_compile_backend = Backend(
                    torch_compile_inductor_enabled,
                    enable_userbuffers=use_ub,
                    enable_piecewise_cuda_graph=self.
                    _torch_compile_piecewise_cuda_graph,
                    capture_num_tokens=self._piecewise_cuda_graph_num_tokens,
                    max_num_streams=torch_compile_max_num_streams,
                    mapping=self.mapping)
                if isinstance(self.model, DecoderModelForCausalLM):
                    self.model.model = torch.compile(
                        self.model.model,
                        backend=self._torch_compile_backend,
                        fullgraph=torch_compile_fullgraph)
                else:
                    self.model = torch.compile(
                        self.model,
                        backend=self._torch_compile_backend,
                        fullgraph=torch_compile_fullgraph)
                torch._dynamo.config.cache_size_limit = 16
            else:
                set_torch_compiling(False)
        except Exception as e:
            import traceback
            traceback.print_exception(Exception, e, e.__traceback__)
            raise e

        self.is_warmup = False
        self.previous_request_ids = []
        self.has_previous_device_draft = False
        self.previous_accepted_tokens_cuda = torch.empty((self.batch_size, ),
                                                         dtype=torch.int,
                                                         device='cuda')

        self.attn_backend = get_attention_backend(
            self.llm_args.attn_backend,
            sparse_attn_config=self.sparse_attention_config)

        if self.is_spec_decode:
            self.spec_metadata = None
            update_spec_config_from_model_config(self.spec_config,
                                                 self.model.config)
            max_num_draft_tokens = self.original_max_total_draft_tokens * self.batch_size
            self.draft_tokens_cuda = torch.empty((max_num_draft_tokens, ),
                                                 dtype=torch.int,
                                                 device='cuda')
            self.gather_ids_cuda = torch.empty((self.max_num_tokens, ),
                                               dtype=torch.int,
                                               device='cuda')
            self.num_accepted_draft_tokens_cuda = torch.empty(
                (self.batch_size, ), dtype=torch.int, device='cuda')
            self.previous_pos_indices_cuda = torch.empty(
                (self.max_num_tokens, ), dtype=torch.int, device='cuda')
            self.previous_pos_id_offsets_cuda = torch.zeros(
                (self.max_num_tokens, ), dtype=torch.int, device='cuda')
            self.previous_kv_lens_offsets_cuda = torch.zeros(
                (self.batch_size, ), dtype=torch.int, device='cuda')
            self.without_logits = self.spec_config.spec_dec_mode.without_logits(
            ) or self.model_is_wrapped
            self.max_draft_len = spec_config.max_draft_len
            self.max_total_draft_tokens = spec_config.max_total_draft_tokens
        else:
            self.without_logits = False
            self.max_draft_len = 0
            self.max_total_draft_tokens = 0

        self.guided_decoder: Optional[CapturableGuidedDecoder] = None

        # This field is initialized lazily on the first forward pass.
        # This is convenient because:
        # 1) The attention metadata depends on the KV cache manager.
        # 2) The KV cache manager depends on the model configuration.
        # 3) The model configuration is not loaded until the model engine
        # is initialized.
        #
        # NOTE: This can simplified by decoupling the model config loading and
        # the model engine.
        self.attn_metadata = None
        self.iter_states = {}
        self._cuda_graph_mem_pool = self._torch_compile_backend._graph_pool_handle if self._torch_compile_enabled else None

        self._cuda_graph_padding_enabled = cuda_graph_padding_enabled

        self._cuda_graph_batch_sizes = _filter_cuda_graph_batch_sizes(
            cuda_graph_batch_sizes, self.batch_size, self.max_num_tokens,
            self.original_max_total_draft_tokens,
            self._cuda_graph_padding_enabled) if cuda_graph_batch_sizes else []

        self._max_cuda_graph_batch_size = (self._cuda_graph_batch_sizes[-1] if
                                           self._cuda_graph_batch_sizes else 0)

        self.previous_batch_indices_cuda = torch.empty((self.max_num_tokens, ),
                                                       dtype=torch.int,
                                                       device='cuda')
        self.input_ids_cuda = torch.empty((self.max_num_tokens, ),
                                          dtype=torch.int,
                                          device='cuda')
        self.position_ids_cuda = torch.empty((self.max_num_tokens, ),
                                             dtype=torch.int,
                                             device='cuda')
        if self.use_mrope:
            self.mrope_position_ids_cuda = torch.empty(
                (3, 1, self.max_num_tokens), dtype=torch.int, device='cuda')

        # Pre-allocated buffers for draft model to avoid implicit synchronization
        # These are used to build index tensors without creating tensors from Python lists
        max_first_draft_tokens = self.batch_size * (
            self.original_max_draft_len + 1) if spec_config else self.batch_size
        tokens_per_draft = self.original_max_draft_len + 1
        self.idx_accepted_tokens_cache = None
        self.draft_token_positions_cache = None
        if spec_config:
            # Cache for idx_accepted_tokens (pattern: 0,0,0...1,1,1...2,2,2...)
            self.idx_accepted_tokens_cache = torch.arange(
                max_first_draft_tokens, dtype=torch.long,
                device='cuda') // tokens_per_draft

        if self.is_draft_model:
            self.draft_ctx_token_indices_cuda = torch.empty((self.batch_size, ),
                                                            dtype=torch.long,
                                                            device='cuda')
            self.draft_ctx_seq_slots_cuda = torch.empty((self.batch_size, ),
                                                        dtype=torch.long,
                                                        device='cuda')
            # Buffers for first_draft requests (max_draft_len+1 tokens per request)
            self.draft_first_draft_indices_cuda = torch.empty(
                (max_first_draft_tokens, ), dtype=torch.long, device='cuda')
            self.draft_first_draft_seq_slots_cuda = torch.empty(
                (max_first_draft_tokens, ), dtype=torch.long, device='cuda')
            # Buffers for seq_slots and request indices
            self.draft_seq_slots_buffer_cuda = torch.empty((self.batch_size, ),
                                                           dtype=torch.int,
                                                           device='cuda')
            self.draft_request_indices_buffer_cuda = torch.empty(
                (self.batch_size, ), dtype=torch.int, device='cuda')

            # Pre-computed constant tensors for incremental update optimization
            # Cache for token_positions (pattern: 0,1,2...N repeated)
            self.draft_token_positions_cache = torch.arange(tokens_per_draft,
                                                            dtype=torch.long,
                                                            device='cuda')

        # We look up this key in resource_manager during forward to find the
        # kv cache manager. Can be changed to support multiple model engines
        # with different KV cache managers.
        self.kv_cache_manager_key = ResourceManagerType.DRAFT_KV_CACHE_MANAGER if is_draft_model else ResourceManagerType.KV_CACHE_MANAGER
        self.lora_model_config: Optional[LoraModelConfig] = None

        # Create config and runner
        cuda_graph_runner_config = CUDAGraphRunnerConfig(
            use_cuda_graph=self.cuda_graph_config is not None,
            cuda_graph_padding_enabled=self._cuda_graph_padding_enabled,
            cuda_graph_batch_sizes=self._cuda_graph_batch_sizes,
            max_cuda_graph_batch_size=self._max_cuda_graph_batch_size,
            max_beam_width=self.max_beam_width,
            spec_config=self.spec_config,
            cuda_graph_mem_pool=self._cuda_graph_mem_pool,
            max_num_tokens=self.max_num_tokens,
            use_mrope=self.use_mrope,
            original_max_draft_len=self.original_max_draft_len,
            original_max_total_draft_tokens=self.
            original_max_total_draft_tokens,
            is_draft_model=self.is_draft_model,
            enable_attention_dp=self.enable_attention_dp,
            batch_size=self.batch_size,
            mapping=self.mapping,
            dist=self.dist,
            kv_cache_manager_key=self.kv_cache_manager_key,
            sparse_attention_config=self.sparse_attention_config,
        )
        self.cuda_graph_runner = CUDAGraphRunner(cuda_graph_runner_config)

        # Setup the local cache indirection buffer only once and reuse it.
        # This way it can also be used for CUDA graphs.
        if self.use_beam_search:
            self.cache_indirection_attention = torch.zeros(
                (self.batch_size, self.max_beam_width, self.max_seq_len +
                 (0 if self._disable_overlap_scheduler else 1)),
                device="cuda",
                dtype=torch.int32)
        else:
            self.cache_indirection_attention = None

        self.kv_cache_dtype_byte_size = self.get_kv_cache_dtype_byte_size()

    def register_forward_pass_callable(self, callable: Callable):
        self.forward_pass_callable = callable

    def get_kv_cache_dtype_byte_size(self) -> float:
        """
        Returns the size (in bytes) occupied by kv cache type.
        """
        layer_quant_mode = self.model.model_config.quant_config.layer_quant_mode
        if layer_quant_mode.has_fp4_kv_cache():
            return 1 / 2
        elif layer_quant_mode.has_fp8_kv_cache(
        ) or layer_quant_mode.has_int8_kv_cache():
            return 1
        else:
            return 2

    @property
    def runtime_draft_len(self):
        return self.max_total_draft_tokens if self.enable_spec_decode else 0

    def set_lora_model_config(self,
                              lora_target_modules: list[str],
                              trtllm_modules_to_hf_modules: dict[str, str],
                              swap_gate_up_proj_lora_b_weight: bool = True):
        self.lora_model_config = LoraModelConfig(
            lora_target_modules=lora_target_modules,
            trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
            hidden_size=self.model.config.hidden_size,
            dtype=torch_dtype_to_str(self.model.config.torch_dtype),
            swap_gate_up_proj_lora_b_weight=swap_gate_up_proj_lora_b_weight)

    def set_guided_decoder(self,
                           guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.model, "set_guided_decoder"):
            success = self.model.set_guided_decoder(guided_decoder)
            if success:
                self.guided_decoder = guided_decoder
            return success
        return False

    @property
    def use_mrope(self):
        use_mrope = False
        try:
            use_mrope = self.model.model_config.pretrained_config.rope_scaling[
                'type'] == 'mrope'
        except Exception:
            pass
        logger.debug(f"Detected use_mrope: {use_mrope}")
        return use_mrope

    @property
    def is_warmup(self):
        return getattr(self, "_is_warmup", False)

    @is_warmup.setter
    def is_warmup(self, value: bool):
        self._is_warmup = value

        self.moe_load_balancer_iter_info = (not value, not value)

    @property
    def moe_load_balancer_iter_info(self):
        moe_load_balancer: MoeLoadBalancer = getattr(self, 'moe_load_balancer',
                                                     None)
        if moe_load_balancer is not None:
            return moe_load_balancer.enable_statistic, moe_load_balancer.enable_update_weights
        return False, False

    @moe_load_balancer_iter_info.setter
    def moe_load_balancer_iter_info(self, value: Tuple[bool, bool]):
        moe_load_balancer: MoeLoadBalancer = getattr(self, 'moe_load_balancer',
                                                     None)
        if moe_load_balancer is not None:
            moe_load_balancer.set_iter_info(enable_statistic=value[0],
                                            enable_update_weights=value[1])

    @property
    def use_beam_search(self):
        return self.max_beam_width > 1

    @contextmanager
    def set_warmup_flag(self):
        prev_is_warmup = self.is_warmup
        self.is_warmup = True
        try:
            yield
        finally:
            self.is_warmup = prev_is_warmup

    @staticmethod
    def with_warmup_flag(method):

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            with self.set_warmup_flag():
                return method(self, *args, **kwargs)

        return wrapper

    @contextlib.contextmanager
    def no_cuda_graph(self):
        _run_cuda_graphs = self.cuda_graph_runner.enabled
        self.cuda_graph_runner.enabled = False
        try:
            yield
        finally:
            self.cuda_graph_runner.enabled = _run_cuda_graphs

    @with_warmup_flag
    def warmup(self, resource_manager: ResourceManager) -> None:
        """
        Orchestrates the warmup process by calling specialized warmup methods for
        torch.compile, the autotuner, and CUDA graphs.
        """
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)

        if kv_cache_manager is None:
            logger.info("Skipping warm up as no KV Cache manager allocated.")
            return

        # The lifetime of model engine and kv cache manager can be different.
        # Reset the global cuda graph dummy request to None in warmup.
        self.cuda_graph_runner.padding_dummy_request = None

        if self.mapping.cp_size > 1:
            cp_type = self.mapping.cp_config.get("cp_type", None)
            logger.info(
                f"[ModelEngine::warmup] Skipping warmup for cp_type: {None if cp_type is None else cp_type.name}."
            )
            return

        self._run_torch_compile_warmup(resource_manager)
        self._run_autotuner_warmup(resource_manager)
        self._run_cuda_graph_warmup(resource_manager)

        # Set the value back to the original value after all warmups are complete
        self.enable_spec_decode = self.is_spec_decode

    def _general_warmup(self,
                        resource_manager: ResourceManager,
                        reverse: bool = False):
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        curr_max_num_tokens = min(
            kv_cache_manager.get_num_available_tokens(
                self.original_max_draft_len), self.max_num_tokens,
            self.batch_size * (self.max_seq_len - 1))
        max_batch_size = min(
            self.batch_size,
            curr_max_num_tokens // (1 + self.runtime_draft_len))

        warmup_requests_configs = {
            (1, 1),  # Specialize for 1 token.
            (max_batch_size, max_batch_size),  # max_batch_size, pure generation
            (2, 0),  # Non-one, pure context
            (curr_max_num_tokens, 0),  # max_num_tokens, pure context
        }

        warmup_requests_configs = sorted(list(warmup_requests_configs),
                                         reverse=reverse)

        for num_tokens, num_gen_tokens in warmup_requests_configs:
            with self._release_batch_context(
                    self._create_warmup_request(resource_manager, num_tokens,
                                                num_gen_tokens),
                    resource_manager) as batch:
                if batch is None:
                    continue  # Not enough KV cache space
                logger.info(
                    f"Run warmup with {num_tokens} tokens, include {num_gen_tokens} generation tokens"
                )
                self.forward(batch,
                             new_tensors_device=None,
                             resource_manager=resource_manager)
                torch.cuda.synchronize()

    def _run_torch_compile_warmup(self, resource_manager: ResourceManager):
        """Runs warmup iterations to specialize torch.compile kernels."""
        if not self._torch_compile_enabled:
            return

        logger.info("Running torch.compile warmup...")

        # Disable cuda graph capture here so that we can properly capture it later
        with self.no_cuda_graph():
            self._general_warmup(resource_manager)

    def _run_autotuner_warmup(self, resource_manager: ResourceManager):
        """Runs a forward pass to populate the autotuner cache."""
        if not self.llm_args.enable_autotuner:
            return
        AutoTuner.get().setup_distributed_state(self.mapping, self.dist)
        logger.info("Running autotuner warmup...")
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        curr_max_num_tokens = min(
            kv_cache_manager.get_num_available_tokens(
                self.original_max_draft_len), self.max_num_tokens,
            self.batch_size * (self.max_seq_len - 1))

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        with self.no_cuda_graph(), autotune(cache_path=cache_path):
            warmup_request = self._create_warmup_request(
                resource_manager, curr_max_num_tokens, 0)
            with self._release_batch_context(warmup_request,
                                             resource_manager) as batch:
                if batch is not None:
                    # Reset the flag is_first_draft for the draft model.
                    # This is necessary for overlap scheduler.
                    spec_resource_manager = resource_manager.get_resource_manager(
                        ResourceManagerType.SPEC_RESOURCE_MANAGER)
                    if self.is_draft_model and isinstance(
                            spec_resource_manager, Eagle3ResourceManager):
                        spec_resource_manager.is_first_draft = True

                    self.forward(batch,
                                 new_tensors_device=None,
                                 resource_manager=resource_manager)

                    # pp_recv in AutoTuner choose_one will never be called if there is no tuning op during the forward pass.
                    # So we need to make an extra call to consume the previous rank's pp_send to guarantee that the previous rank's pp_send is released.
                    AutoTuner.get().cache_pp_recv()
                    # Send the cache after the tuning process to the next PP rank
                    AutoTuner.get().cache_pp_send()
                    # Clean the pp flag to avoid deadlock with synchronous send/recv
                    AutoTuner.get().clean_pp_flag()

                    torch.cuda.synchronize()

        logger.info(
            f"[Autotuner] Cache size after warmup is {len(AutoTuner.get().profiling_cache)}"
        )
        AutoTuner.get().print_profiling_cache()

    def _run_cuda_graph_warmup(self, resource_manager: ResourceManager):
        """Captures CUDA graphs for various batch sizes and draft lengths."""
        if not (self.cuda_graph_runner.enabled
                or self._torch_compile_piecewise_cuda_graph):
            return

        self._capture_generation_cuda_graphs(resource_manager)
        self._capture_piecewise_cuda_graphs(resource_manager)

    def _capture_generation_cuda_graphs(self,
                                        resource_manager: ResourceManager):
        """Captures CUDA graphs for pure generation steps."""
        if not self.cuda_graph_runner.enabled:
            return

        logger.info(
            f"Creating CUDA graph instances for {len(self._cuda_graph_batch_sizes)} batch sizes."
        )
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)

        # Reverse order so smaller graphs can reuse memory from larger ones
        cuda_graph_batch_sizes = sorted(self._cuda_graph_batch_sizes,
                                        reverse=True)
        # Create CUDA graphs for different draft lengths
        draft_lengths = []
        if self.is_draft_model:
            if self.model_is_wrapped and self.is_spec_decode and spec_resource_manager is not None and isinstance(
                    spec_resource_manager, Eagle3ResourceManager):
                # The CDL path uses draft_len > 0 for the number of iterations in the drafting loop.
                draft_lengths.append(self.original_max_total_draft_tokens)
            else:
                draft_lengths.append(self.max_total_draft_tokens)
        else:
            draft_lengths.append(self.max_total_draft_tokens)
            # For non-draft model, we also capture the CUDA graph instance for draft length 0,
            # so that when we disable spec decode at runtime, we can still run the captured graph.
            # Note that for one engine mode, we are not able to turn off spec decode at runtime.
            if (self.max_total_draft_tokens > 0
                    and not self.spec_config.spec_dec_mode.use_one_engine()
                    # Assume that speculation is always on if the user didn't give us a max_concurrency
                    # value. This will save on memory.
                    and self.spec_config.max_concurrency is not None):
                draft_lengths.append(0)
        # Reverse order so smaller graphs can reuse memory from larger ones
        draft_lengths = sorted(set(draft_lengths), reverse=True)

        # Create CUDA graphs for short and long sequences separately for sparse attention.
        sparse_config = self.sparse_attention_config
        if sparse_config is not None and sparse_config.needs_separate_short_long_cuda_graphs(
        ):
            # For short sequences, use the (seq_len_threshold - max_draft_len - 1) as the maximum sequence length
            # to make sure all of the past and current input tokens are within the sequence length threshold.
            # For long sequences, use the default maximum sequence length (self.max_seq_len).
            max_seq_len = sparse_config.seq_len_threshold - (
                self.max_draft_len + 1)
            if max_seq_len < self.max_seq_len:
                max_seq_len_list = [self.max_seq_len, max_seq_len]
            else:
                max_seq_len_list = [self.max_seq_len]
        else:
            max_seq_len_list = [self.max_seq_len]

        for bs in cuda_graph_batch_sizes:
            if bs > self.batch_size:
                continue

            for draft_len in draft_lengths:
                for max_seq_len in max_seq_len_list:
                    warmup_request = self._create_cuda_graph_warmup_request(
                        resource_manager, bs, draft_len, max_seq_len)
                    with self._release_batch_context(warmup_request,
                                                     resource_manager) as batch:
                        if batch is None:
                            # No KV cache space, cannot continue capturing graphs
                            continue

                        logger.info(
                            f"Run generation-only CUDA graph warmup for batch size={bs}, draft_len={draft_len}, max_seq_len={max_seq_len}"
                        )

                        self.enable_spec_decode = draft_len > 0 or self.is_draft_model
                        self._update_draft_inference_state_for_warmup(
                            batch, draft_len > 0, resource_manager)

                        self.forward(batch,
                                     new_tensors_device=None,
                                     resource_manager=resource_manager)
                        torch.cuda.synchronize()

    def _capture_piecewise_cuda_graphs(self, resource_manager: ResourceManager):
        """Captures piecewise CUDA graphs for context/prefill steps via torch.compile."""
        if not (self._torch_compile_piecewise_cuda_graph
                and self._torch_compile_enabled):
            return

        logger.info("Running piecewise CUDA graph warmup...")
        piecewise_cuda_graph_num_tokens = sorted(
            self._piecewise_cuda_graph_num_tokens, reverse=True)

        with capture_piecewise_cuda_graph(True), self.no_cuda_graph():
            for num_tokens in piecewise_cuda_graph_num_tokens:
                warmup_request = self._create_warmup_request(
                    resource_manager, num_tokens, 0)
                with self._release_batch_context(warmup_request,
                                                 resource_manager) as batch:
                    if batch is None:
                        continue

                    logger.info(
                        f"Run piecewise CUDA graph warmup for num tokens={num_tokens}"
                    )
                    # Run a few times to ensure capture
                    for _ in range(3):
                        self.forward(batch,
                                     new_tensors_device=None,
                                     resource_manager=resource_manager)

                    self.forward(batch,
                                 new_tensors_device=None,
                                 resource_manager=resource_manager)
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

        # When using piecewise cuda graph, the logits may suffer severe memory faction problem.
        # When the num of requests is growing, the block allocated by torch cannot be reused.
        # So after piecewise cuda graph capture, a request with most requests is triggered to make
        # sure that large enough blocks are allocated and can be correctly reused.
        for num_tokens in piecewise_cuda_graph_num_tokens:
            warmup_request = self._create_warmup_request(resource_manager,
                                                         num_tokens,
                                                         0,
                                                         least_requests=False)
            with self._release_batch_context(warmup_request,
                                             resource_manager) as batch:
                if batch is None:
                    continue
                logger.info(
                    f"Run piecewise CUDA graph warmup for num tokens={num_tokens} with most requests"
                )
                self.forward(batch,
                             new_tensors_device=None,
                             resource_manager=resource_manager)
                torch.cuda.synchronize()

    ### Helper methods promoted from the original warmup method ###

    @contextlib.contextmanager
    def _release_batch_context(self, batch: Optional[ScheduledRequests],
                               resource_manager: ResourceManager):
        """A context manager to automatically free resources of a dummy batch."""
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        try:
            yield batch
        finally:
            if batch is not None and kv_cache_manager is not None:
                for req in batch.all_requests():
                    kv_cache_manager.free_resources(req)
                    if spec_resource_manager is not None:
                        spec_resource_manager.free_resources(req)

    def _get_num_extra_decoding_steps(self) -> int:
        """Determines extra decoding steps needed for fused drafting loops."""
        if isinstance(self.model, BaseDraftingLoopWrapper):
            return self.model.max_total_draft_tokens
        else:
            assert not self.model_is_wrapped, (
                f"Please add logic to determine num_extra_decoding_steps for drafting loop {type(self.model)}"
            )
            return 0

    def _create_warmup_request(
            self,
            resource_manager: ResourceManager,
            num_tokens: int,
            num_gen_requests: int,
            least_requests: bool = True) -> Optional[ScheduledRequests]:
        """Creates a generic dummy ScheduledRequests object for warmup."""
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)

        available_tokens = kv_cache_manager.get_num_available_tokens(
            self.runtime_draft_len)
        available_blocks = kv_cache_manager.get_num_free_blocks()
        if num_tokens > self.max_num_tokens or num_tokens > available_tokens:
            return None

        num_extra_decoding_steps = self._get_num_extra_decoding_steps()

        if num_gen_requests > self.batch_size:
            return None
        num_gen_tokens = num_gen_requests * (1 + self.runtime_draft_len)
        if num_gen_tokens > self.max_num_tokens:
            return None

        num_ctx_tokens = num_tokens - num_gen_tokens
        num_ctx_requests = 0
        ctx_requests = []
        gen_requests = []

        # For drafting loops, reduce max_seq_len to leave room for extra decoding steps
        max_seq_len = self.max_seq_len - 1 - num_extra_decoding_steps
        if max_seq_len < 1:
            return None  # Not enough sequence length for drafting loop
        num_full_seqs = 0
        num_left_over_tokens = 0

        max_context_requests = self.batch_size - num_gen_requests
        if max_context_requests * max_seq_len < num_ctx_tokens:
            return None

        if num_ctx_tokens > 0:
            if least_requests:
                num_full_seqs = num_ctx_tokens // max_seq_len
                num_left_over_tokens = num_ctx_tokens - num_full_seqs * max_seq_len

            else:
                max_bs = min(num_ctx_tokens, max_context_requests)
                if num_ctx_tokens % max_bs == 0:
                    num_full_seqs = max_bs
                else:
                    num_full_seqs = max_bs - 1
                max_seq_len = num_ctx_tokens // num_full_seqs
                num_left_over_tokens = num_ctx_tokens - max_seq_len * num_full_seqs
            num_ctx_requests = num_full_seqs + (1 if num_left_over_tokens > 0
                                                else 0)

        if num_ctx_requests + num_gen_requests > self.batch_size:
            return None  # Not enough batch size to fill the request

        blocks_to_use = num_full_seqs * math.ceil(
            max_seq_len / kv_cache_manager.tokens_per_block) + math.ceil(
                num_left_over_tokens /
                kv_cache_manager.tokens_per_block) + num_gen_requests

        if blocks_to_use > available_blocks:
            return None

        if num_ctx_tokens > 0:
            ctx_token_nums = [max_seq_len] * num_full_seqs
            if num_left_over_tokens > 0:
                ctx_token_nums.append(num_left_over_tokens)

            ctx_requests = kv_cache_manager.add_dummy_requests(
                list(range(num_ctx_requests)),
                token_nums=ctx_token_nums,
                is_gen=False,
                max_num_draft_tokens=self.runtime_draft_len,
                use_mrope=self.use_mrope,
                num_extra_decoding_steps=num_extra_decoding_steps)

            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(
                    request_ids=list(range(num_ctx_requests)))

        if num_gen_requests > 0:
            gen_requests = kv_cache_manager.add_dummy_requests(
                list(
                    range(num_ctx_requests,
                          num_ctx_requests + num_gen_requests)),
                token_nums=[1] * num_gen_requests,
                is_gen=True,
                max_num_draft_tokens=self.max_total_draft_tokens,
                use_mrope=self.use_mrope,
                max_beam_width=self.max_beam_width,
                num_extra_decoding_steps=num_extra_decoding_steps)
            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(request_ids=list(
                    range(num_ctx_requests, num_ctx_requests +
                          num_gen_requests)))

        result = ScheduledRequests()
        result.context_requests = ctx_requests
        result.generation_requests = gen_requests
        return result

    def _create_cuda_graph_warmup_request(
            self,
            resource_manager: ResourceManager,
            batch_size: int,
            draft_len: int,
            max_seq_len: int = None) -> Optional[ScheduledRequests]:
        """Creates a dummy ScheduledRequests tailored for CUDA graph capture."""
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)

        available_blocks = kv_cache_manager.get_num_free_blocks(
        ) // self.max_beam_width
        if available_blocks < batch_size:
            return None

        result = ScheduledRequests()
        result.context_requests = []
        num_extra_decoding_steps = self._get_num_extra_decoding_steps()

        # Add (batch_size - 1) dummy requests with seq_len=1.
        requests = kv_cache_manager.add_dummy_requests(
            list(range(batch_size - 1)),
            is_gen=True,
            max_num_draft_tokens=draft_len,
            use_mrope=self.use_mrope,
            max_beam_width=self.max_beam_width,
            num_extra_decoding_steps=num_extra_decoding_steps)

        available_tokens = kv_cache_manager.get_num_available_tokens(draft_len)

        # Add one dummy request with the maximum possible sequence length.
        max_seq_len = self.max_seq_len if max_seq_len is None else max_seq_len
        token_num = max(1, min(available_tokens, max_seq_len - 1))
        model_config = self.model.model_config.pretrained_config
        max_position_embeddings = getattr(model_config,
                                          'max_position_embeddings', None)
        if max_position_embeddings is not None:
            token_num = min(token_num, max_position_embeddings - draft_len)

        assert token_num > num_extra_decoding_steps, (
            "Cannot fuse drafting loop. Not enough KV cache space for all draft tokens."
        )
        token_num -= num_extra_decoding_steps

        max_seq_len_request = kv_cache_manager.add_dummy_requests(
            request_ids=[batch_size - 1],
            token_nums=[token_num],
            is_gen=True,
            max_num_draft_tokens=draft_len,
            use_mrope=self.use_mrope,
            max_beam_width=self.max_beam_width,
            num_extra_decoding_steps=num_extra_decoding_steps)[0]

        # Insert the longest request first to simulate padding for the CUDA graph.
        requests.insert(0, max_seq_len_request)
        result.generation_requests = requests
        if spec_resource_manager is not None:
            spec_resource_manager.add_dummy_requests(
                request_ids=list(range(batch_size)))
        return result

    def _update_draft_inference_state_for_warmup(
            self, batch: ScheduledRequests, is_first_draft: bool,
            resource_manager: ResourceManager):
        """Updates request states for specific draft model warmups like Eagle3."""
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        if self.is_draft_model and isinstance(spec_resource_manager,
                                              Eagle3ResourceManager):
            spec_resource_manager.is_first_draft = is_first_draft
            if is_first_draft:
                for req in batch.generation_requests:
                    req.py_is_first_draft = True
                    req.py_draft_tokens = []

    def _set_up_attn_metadata(self, kv_cache_manager: KVCacheManager):
        enable_context_mla_with_cached_kv = is_mla(
            self.model.model_config.pretrained_config) and (
                self.attn_runtime_features.cache_reuse
                or self.attn_runtime_features.chunked_prefill)
        cache_indirection = self.cache_indirection_attention if self.attn_backend.Metadata is TrtllmAttentionMetadata else None
        num_attention_heads = getattr(self.model.model_config.pretrained_config,
                                      'num_attention_heads', None)
        config = self.model.model_config.pretrained_config

        num_attention_heads = getattr(config, 'num_attention_heads', None)
        num_key_value_heads = getattr(config, 'num_key_value_heads', None)

        # Calculate the number of attention heads per KV head (GQA ratio)
        if isinstance(num_key_value_heads, (list, tuple)):
            # Filter out invalid KV heads, default to 0 if no valid KV heads are found
            num_key_value_heads = min(
                (kv for kv in num_key_value_heads if kv and kv > 0), default=0)
        if num_attention_heads and num_key_value_heads:
            num_heads_per_kv = num_attention_heads // num_key_value_heads
        else:
            num_heads_per_kv = 1

        if kv_cache_manager is None:
            return self.attn_backend.Metadata(
                max_num_requests=self.batch_size,
                max_num_tokens=self.max_num_tokens,
                max_num_sequences=self.batch_size * self.max_beam_width,
                kv_cache_manager=None,
                mapping=self.mapping,
                runtime_features=self.attn_runtime_features,
                enable_flash_mla=self.model.model_config.enable_flash_mla,
                enable_context_mla_with_cached_kv=
                enable_context_mla_with_cached_kv,
                cache_indirection=cache_indirection,
                sparse_attention_config=self.sparse_attention_config,
                num_heads_per_kv=num_heads_per_kv)

        if self.attn_metadata is not None:
            # This assertion can be relaxed if needed: just create a new metadata
            # object if it changes.
            assert self.attn_metadata.kv_cache_manager is kv_cache_manager
            return self.attn_metadata

        self.attn_metadata = self.attn_backend.Metadata(
            max_num_requests=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_num_sequences=self.batch_size * self.max_beam_width,
            kv_cache_manager=kv_cache_manager,
            mapping=self.mapping,
            runtime_features=self.attn_runtime_features,
            enable_flash_mla=self.model.model_config.enable_flash_mla,
            enable_context_mla_with_cached_kv=enable_context_mla_with_cached_kv,
            cache_indirection=cache_indirection,
            sparse_attention_config=self.sparse_attention_config,
            num_heads_per_kv=num_heads_per_kv,
        )

        return self.attn_metadata

    def _set_up_spec_metadata(
            self,
            spec_resource_manager: Optional[BaseResourceManager],
            no_cache=False):
        spec_config = self.spec_config if self.enable_spec_decode else None
        if no_cache:
            return get_spec_metadata(
                spec_config,
                self.model.config,
                self.batch_size,
                max_num_tokens=self.max_num_tokens,
                spec_resource_manager=spec_resource_manager,
                is_draft_model=self.is_draft_model)

        if self.spec_metadata is not None:
            return self.spec_metadata
        self.spec_metadata = get_spec_metadata(
            spec_config,
            self.model.config,
            self.batch_size,
            max_num_tokens=self.max_num_tokens,
            spec_resource_manager=spec_resource_manager,
            is_draft_model=self.is_draft_model)
        return self.spec_metadata

    def __del__(self) -> None:
        self.model = None
        self.model_loader = None
        self._release_cuda_graphs()
        self.input_processor = None
        self.input_processor_with_hash = None
        if getattr(self, 'ub_buffers', None):
            for u in self.ub_buffers:
                ub.ub_deallocate(u.addr)
        # Release model weights.
        release_gc()

    def _init_max_seq_len(self):
        # For mm_encoder_only mode, infer_max_seq_len() is for LLM decoder models
        if hasattr(self.model, 'infer_max_seq_len'):
            inferred_max_seq_len = self.model.infer_max_seq_len()
        else:
            inferred_max_seq_len = self._infer_max_seq_len_from_config()

        if self.max_seq_len is None:
            logger.info(
                f"max_seq_len is not specified, using inferred value {inferred_max_seq_len}"
            )
            self.max_seq_len = inferred_max_seq_len

        elif inferred_max_seq_len < self.max_seq_len:
            # NOTE: py_executor_creator makes sure that the executor uses this
            # smaller value as its max_seq_len too.
            logger.warning(
                f"Specified {self.max_seq_len=} is larger than what the model can support "
                f"({inferred_max_seq_len}). Setting max_seq_len to {inferred_max_seq_len}. "
            )
            self.max_seq_len = inferred_max_seq_len

    def _infer_max_seq_len_from_config(self) -> int:

        if hasattr(self.model, 'model_config') and self.model.model_config:
            model_config = self.model.model_config.pretrained_config
            rope_scaling = getattr(model_config, 'rope_scaling', None)
            rope_factor = 1
            if rope_scaling is not None:
                rope_type = rope_scaling.get('type',
                                             rope_scaling.get('rope_type'))
                if rope_type not in ("su", "longrope", "llama3", "yarn"):
                    rope_factor = rope_scaling.get('factor', 1.0)

            # Step 1: Find the upper bound of max_seq_len
            inferred_max_seq_len = 2048
            max_position_embeddings = getattr(model_config,
                                              'max_position_embeddings', None)
            if max_position_embeddings is None and hasattr(
                    model_config, 'text_config'):
                max_position_embeddings = getattr(model_config.text_config,
                                                  'max_position_embeddings',
                                                  None)
            if max_position_embeddings is not None:
                inferred_max_seq_len = max_position_embeddings

            # Step 2: Scale max_seq_len with rotary scaling
            if rope_factor != 1:
                inferred_max_seq_len = int(
                    math.ceil(inferred_max_seq_len * rope_factor))
                logger.warning(
                    f'max_seq_len is scaled to {inferred_max_seq_len} by rope scaling {rope_factor}'
                )

            return inferred_max_seq_len

        default_max_seq_len = 8192
        logger.warning(
            f"Could not infer max_seq_len from model config, using default value: {default_max_seq_len}"
        )
        return default_max_seq_len

    def _init_max_num_tokens(self):
        # Modified from tensorrt_llm/_common.py check_max_num_tokens
        if self.max_num_tokens is None:
            self.max_num_tokens = self.max_seq_len * self.batch_size
        if self.max_num_tokens > self.max_seq_len * self.batch_size:
            logger.warning(
                f"max_num_tokens ({self.max_num_tokens}) shouldn't be greater than "
                f"max_seq_len * max_batch_size ({self.max_seq_len * self.batch_size}), "
                f"specifying to max_seq_len * max_batch_size ({self.max_seq_len * self.batch_size})."
            )
            self.max_num_tokens = self.max_seq_len * self.batch_size

    def _init_model_capacity(self):
        self._init_max_seq_len()
        self._init_max_num_tokens()

    def _release_cuda_graphs(self):
        self.cuda_graph_runner.clear()

    def get_max_num_sequences(self) -> int:
        """
        Return the maximum number of sequences that the model supports. PyExecutor need this to compute max_num_active_requests
        """
        num_batches = self.mapping.pp_size
        return num_batches * self.batch_size

    def _preprocess_inputs(self, inputs: Dict[str, Any]):
        """
        Make some changes to the device inputs and avoid block the async data transfer
        """
        if self.enable_spec_decode and not self._disable_overlap_scheduler:
            # When enabling overlap scheduler, the kv cache for draft tokens will
            # be prepared in advance by using the max_total_draft_tokens. But we need to use
            # new_tokens_lens_device to get the real past kv lengths and the
            # correct position ids. And to avoid blocking the async data transfer,
            # we need to preprocess the inputs in forward to update the position_ids and
            # kv cache length.
            if inputs['attn_metadata'].kv_cache_manager is not None:
                num_seqs = inputs['attn_metadata'].num_seqs
                num_ctx_requests = inputs['attn_metadata'].num_contexts
                num_gen_requests = inputs['attn_metadata'].num_generations
                num_ctx_tokens = inputs['attn_metadata'].num_ctx_tokens
                num_chunked_ctx_requests = inputs[
                    'attn_metadata'].num_chunked_ctx_requests
                previous_batch_tokens = inputs['input_ids'].shape[
                    0] - num_ctx_tokens
                inputs['position_ids'][0, num_ctx_tokens:] += (
                    self.previous_pos_id_offsets_cuda[:previous_batch_tokens])
                if hasattr(inputs['attn_metadata'], 'kv_lens_cuda'):
                    if num_ctx_requests >= num_chunked_ctx_requests and num_chunked_ctx_requests > 0:
                        # The generation requests with draft_tokens are treated as chunked context requests when extend_ctx returns True.
                        inputs['attn_metadata'].kv_lens_cuda[
                            num_ctx_requests -
                            num_chunked_ctx_requests:num_ctx_requests] += (
                                self.
                                previous_kv_lens_offsets_cuda[:
                                                              num_chunked_ctx_requests]
                            )
                    else:
                        inputs['attn_metadata'].kv_lens_cuda[
                            num_ctx_requests:num_seqs] += (
                                self.
                                previous_kv_lens_offsets_cuda[:num_gen_requests]
                            )
                    inputs['attn_metadata'].on_update_kv_lens()

        if self.guided_decoder is not None:
            self.guided_decoder.token_event.record()

        return inputs

    def _postprocess_inputs(self, inputs: Dict[str, Any]):
        """
        Postprocess to make sure model forward doesn't change the inputs.
        It is only used in cuda graph capture, because other cases will prepare
        new inputs before the model forward.
        """
        if self.enable_spec_decode and not self._disable_overlap_scheduler:
            if inputs['attn_metadata'].kv_cache_manager is not None:
                num_seqs = inputs['attn_metadata'].num_seqs
                num_ctx_requests = inputs['attn_metadata'].num_contexts
                num_gen_requests = inputs['attn_metadata'].num_generations
                num_ctx_tokens = inputs['attn_metadata'].num_ctx_tokens
                num_chunked_ctx_requests = inputs[
                    'attn_metadata'].num_chunked_ctx_requests
                previous_batch_tokens = inputs['input_ids'].shape[
                    0] - num_ctx_tokens
                inputs['position_ids'][0, num_ctx_tokens:] -= (
                    self.previous_pos_id_offsets_cuda[:previous_batch_tokens])
                # Only TrtllmAttentionMetadata has kv_lens_cuda.
                if isinstance(inputs['attn_metadata'], TrtllmAttentionMetadata):
                    if num_ctx_requests >= num_chunked_ctx_requests and num_chunked_ctx_requests > 0:
                        inputs['attn_metadata'].kv_lens_cuda[
                            num_ctx_requests -
                            num_chunked_ctx_requests:num_ctx_requests] -= (
                                self.
                                previous_kv_lens_offsets_cuda[:
                                                              num_chunked_ctx_requests]
                            )
                    else:
                        inputs['attn_metadata'].kv_lens_cuda[
                            num_ctx_requests:num_seqs] -= (
                                self.
                                previous_kv_lens_offsets_cuda[:num_gen_requests]
                            )

    def _get_all_rank_num_tokens(self, attn_metadata: AttentionMetadata):
        if self.enable_attention_dp:
            return list(self.dist.tp_cp_allgather(attn_metadata.num_tokens))
        return None

    def _get_all_rank_ctx_requests(self, num_ctx_requests: int):
        if self.enable_attention_dp:
            return list(self.dist.tp_allgather(num_ctx_requests))
        return None

    def _get_padding_params(
        self, total_num_tokens: int, num_ctx_requests: int,
        attn_all_rank_num_tokens: Optional[List[int]]
    ) -> Tuple[int, bool, Optional[List[int]]]:
        """
        Get the padding parameters for tensor padding.
        Return:
            padded_num_tokens: the padded number of tokens
            can_run_piecewise_cuda_graph: whether the piecewise cuda graph can be run
            attn_all_rank_num_tokens: the number of tokens for each rank
        """
        padded_num_tokens = total_num_tokens

        all_rank_ctx_requests = self._get_all_rank_ctx_requests(
            num_ctx_requests)

        def get_padded_piecewise_tokens(tokens):
            captured_num_tokens = self._torch_compile_backend.capture_num_tokens
            return captured_num_tokens[bisect.bisect_left(
                captured_num_tokens, tokens)]

        if (self._torch_compile_backend is not None
                and self._torch_compile_piecewise_cuda_graph
                and self._torch_compile_backend.capture_num_tokens):
            max_captured_num_tokens = self._torch_compile_backend.capture_num_tokens[
                -1]
            # Torch piecewise cuda graph is enabled.
            if attn_all_rank_num_tokens is not None:
                # Any rank has context requests, we enable piecewise cuda graph.
                has_ctx_requests = num_ctx_requests != 0 or (
                    all_rank_ctx_requests is not None
                    and any(ctx_requests != 0
                            for ctx_requests in all_rank_ctx_requests))
                can_run_piecewise_cuda_graph = (has_ctx_requests and
                                                max(attn_all_rank_num_tokens)
                                                <= max_captured_num_tokens)
                all_ranks_can_run_piecewise_cuda_graph = list(
                    self.dist.tp_allgather(can_run_piecewise_cuda_graph))
                if all(all_ranks_can_run_piecewise_cuda_graph):
                    padded_num_tokens = get_padded_piecewise_tokens(
                        max(attn_all_rank_num_tokens))
                    logger.debug(
                        f"Pad tensor with {total_num_tokens} tokens to {padded_num_tokens} tokens"
                    )
                    return padded_num_tokens, True, [
                        padded_num_tokens
                    ] * len(attn_all_rank_num_tokens)
                else:
                    logger.debug(
                        f"Not all ranks can run piecewise cuda graph, disable piecewise cuda graph"
                    )
                    return total_num_tokens, False, attn_all_rank_num_tokens
            elif num_ctx_requests != 0 and total_num_tokens <= max_captured_num_tokens:
                padded_num_tokens = get_padded_piecewise_tokens(
                    total_num_tokens)
                logger.debug(
                    f"Pad tensor with {total_num_tokens} tokens to {padded_num_tokens} tokens"
                )
                return padded_num_tokens, True, None
            else:
                logger.debug(
                    f"Picewise cudagraph cannot be used with {total_num_tokens} tokens, {num_ctx_requests} context requests"
                )
                return total_num_tokens, False, None

        return total_num_tokens, False, attn_all_rank_num_tokens

    def _prepare_multimodal_indices(self, input_ids: list[int]):
        input_ids = torch.tensor(input_ids, dtype=torch.int, device="cpu")
        vocab_size = self.model.config.vocab_size
        # TODO: unify naming of mm_token_ids across models
        mm_token_ids = getattr(self.model, "mm_token_ids", None)

        text_token_indices, mm_token_indices = filter_mm_token_from_input_ids(
            input_ids, vocab_size=vocab_size, mm_token_ids=mm_token_ids)
        return text_token_indices, mm_token_indices

    def _can_use_incremental_update(
            self, scheduled_requests: ScheduledRequests,
            new_tokens_device: Optional[torch.Tensor],
            next_draft_tokens_device: Optional[torch.Tensor]) -> bool:
        """
        Check if we can use incremental update for the given scheduled requests and new tensors device.
        """
        # Not use this approach for non-speculative decoding
        if self.spec_config is None:
            return False

        # Not allowed for one-model speculative decoding
        if not self.spec_config.spec_dec_mode.has_draft_model():
            return False

        if not self.cuda_graph_runner.enabled:
            return False

        if self.use_mrope:
            return False

        # Not allowed for non-overlap scheduler
        if new_tokens_device is None:
            return False

        # The changes between context and generation requests are not straightforward.
        if len(scheduled_requests.context_requests) > 0:
            return False

        # Check if the request_ids changes
        request_ids = [
            request.py_request_id
            for request in scheduled_requests.generation_requests
        ]
        if self.previous_request_ids != request_ids:
            return False

        has_current_device_draft = next_draft_tokens_device is not None
        return (self.is_draft_model and self.model_is_wrapped) or (
            has_current_device_draft and self.has_previous_device_draft)

    @nvtx_range("_apply_incremental_update")
    def _apply_incremental_update(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            new_tensors_device: Optional[SampleStateTensors] = None,
            cache_indirection_buffer: Optional[torch.Tensor] = None,
            num_accepted_tokens_device: Optional[torch.Tensor] = None,
            req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None,
            resource_manager: Optional[ResourceManager] = None):
        """
        Apply incremental update for the given scheduled requests and new tensors device.
        """

        if self.is_draft_model:
            return self._apply_incremental_update_draft(
                scheduled_requests, kv_cache_manager, attn_metadata,
                spec_metadata, new_tensors_device, num_accepted_tokens_device)
        else:
            return self._apply_incremental_update_target(
                scheduled_requests, kv_cache_manager, attn_metadata,
                spec_metadata, new_tensors_device, num_accepted_tokens_device)

    @nvtx_range("_prepare_incremental_update_metadata")
    def _prepare_incremental_update_metadata(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata],
            prompt_lengths: List[int],
            num_cached_tokens_per_seq: List[int],
            total_num_tokens: int,
            num_generation_tokens: int,
            request_accepted_path: Optional[Dict[int, Any]] = None,
            num_extend_ctx_requests: int = 0):
        """
        Common metadata preparation logic for incremental updates.
        """

        enable_spec_decode = self.enable_spec_decode
        enable_attention_dp = self.enable_attention_dp
        spec_config = self.spec_config if enable_spec_decode else None

        # Set up attention metadata - batch simple assignments
        attn_metadata.beam_width = 1
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = num_extend_ctx_requests if (
            enable_spec_decode and spec_config.spec_dec_mode.extend_ctx(
                self.attn_backend) and spec_config.is_linear_tree) else 0
        attn_metadata.num_chunked_ctx_requests = attn_metadata.num_contexts

        # Create KV cache params and prepare metadata
        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            num_extra_kv_tokens=get_num_extra_kv_tokens(spec_config))
        attn_metadata.kv_cache_manager = kv_cache_manager
        attn_metadata.prepare()

        # Get LoRA parameters
        lora_params = self._get_lora_params_from_requests(
            scheduled_requests, attn_metadata)

        # Handle padding for piecewise CUDA graphs
        attn_metadata.padded_num_tokens = None

        # Handle attention DP
        if enable_attention_dp:
            attn_metadata.all_rank_num_tokens = self._get_all_rank_num_tokens(
                attn_metadata)

        # Prepare speculative metadata
        if spec_metadata is not None:
            # Set request_accepted_path if Eagle3
            if isinstance(spec_metadata, Eagle3SpecMetadata):
                spec_metadata.request_accepted_path = request_accepted_path

            spec_metadata.num_tokens = total_num_tokens
            spec_metadata.prepare()

            # Handle distributed spec metadata
            if enable_attention_dp:
                sequence_lengths = spec_metadata.seq_lens
                all_rank_num_tokens = self.dist.tp_allgather(
                    [spec_metadata.num_tokens,
                     len(sequence_lengths)])
                spec_metadata.all_rank_num_tokens = [
                    item[0] for item in all_rank_num_tokens
                ]
                spec_metadata.all_rank_num_seqs = [
                    item[1] for item in all_rank_num_tokens
                ]

        # Set iteration states - batch dictionary updates
        self.iter_states.update({
            'num_ctx_requests': 0,
            'num_ctx_tokens': 0,
            'num_generation_tokens': num_generation_tokens
        })

        return lora_params

    def _update_draft_input_tensors(self,
                                    num_accepted_tokens_device: torch.Tensor,
                                    new_tokens_device: torch.Tensor,
                                    total_num_tokens: int,
                                    num_first_draft_requests: int):
        """
        This function performs in-place updates on position_ids, num_accepted_draft_tokens,
        gather_ids, and input_ids tensors for speculative decoding draft operations.
        """
        # Prepare position_ids
        idx_accepted_tokens = self.idx_accepted_tokens_cache[:total_num_tokens]
        self.position_ids_cuda[:total_num_tokens].add_(
            self.num_accepted_draft_tokens_cuda[idx_accepted_tokens] + 1)

        # Prepare gather_ids
        old_accepted_tokens = self.num_accepted_draft_tokens_cuda[:
                                                                  num_first_draft_requests].clone(
                                                                  )
        self.num_accepted_draft_tokens_cuda[:num_first_draft_requests].copy_(
            num_accepted_tokens_device[
                self.draft_seq_slots_buffer_cuda[:num_first_draft_requests]],
            non_blocking=True)
        self.gather_ids_cuda[:num_first_draft_requests].add_(
            self.num_accepted_draft_tokens_cuda[:num_first_draft_requests] -
            old_accepted_tokens)

        # Prepare token_positions for input_ids update
        tokens_per_first_draft = self.original_max_draft_len + 1
        token_positions = self.draft_token_positions_cache[:tokens_per_first_draft].repeat(
            num_first_draft_requests)

        # Prepare input_ids
        self.input_ids_cuda[
            self.
            draft_first_draft_indices_cuda[:total_num_tokens]] = new_tokens_device[
                token_positions,
                self.draft_first_draft_seq_slots_cuda[:total_num_tokens], 0]

    def _apply_incremental_update_draft(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            new_tensors_device: Optional[SampleStateTensors] = None,
            num_accepted_tokens_device: Optional[torch.Tensor] = None):
        new_tokens_device = new_tensors_device.new_tokens

        num_generation_tokens = len(scheduled_requests.generation_requests)
        num_gen_requests = 0

        tokens_per_first_draft = self.original_max_draft_len + 1
        prompt_lengths = []  # per sequence
        num_cached_tokens_per_seq = []  # per sequence

        for request in scheduled_requests.generation_requests:
            if request.is_dummy:
                num_gen_requests += 1
                past_seen_token_num = request.max_beam_num_tokens - 1
                request.cached_tokens = past_seen_token_num
            else:
                assert request.py_is_first_draft
                past_seen_token_num = request.max_beam_num_tokens - tokens_per_first_draft

            num_cached_tokens_per_seq.append(past_seen_token_num)
            prompt_lengths.append(request.py_prompt_len)
            request.py_batch_idx = request.py_seq_slot

        num_first_draft_requests = num_generation_tokens - num_gen_requests
        total_num_tokens = num_first_draft_requests * tokens_per_first_draft

        self._update_draft_input_tensors(
            num_accepted_tokens_device=num_accepted_tokens_device,
            new_tokens_device=new_tokens_device,
            total_num_tokens=total_num_tokens,
            num_first_draft_requests=num_first_draft_requests)

        # Prepare spec_metadata
        if spec_metadata is not None:
            spec_metadata.draft_tokens = []
            spec_metadata.gather_ids = self.gather_ids_cuda[:
                                                            num_generation_tokens]
            spec_metadata.num_accepted_draft_tokens = self.num_accepted_draft_tokens_cuda[:
                                                                                          num_generation_tokens]

        # Use common metadata preparation logic
        virtual_num_tokens = total_num_tokens + num_gen_requests
        lora_params = self._prepare_incremental_update_metadata(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            prompt_lengths=prompt_lengths,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            total_num_tokens=virtual_num_tokens,
            num_generation_tokens=num_generation_tokens,
            num_extend_ctx_requests=0)

        # No padding because there are only generation requests.
        attn_metadata.padded_num_tokens = None
        if self.enable_attention_dp:
            attn_metadata.all_rank_num_tokens = self._get_all_rank_num_tokens(
                attn_metadata)

        final_position_ids = self.position_ids_cuda[:
                                                    virtual_num_tokens].unsqueeze(
                                                        0)

        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:virtual_num_tokens],
            'position_ids': final_position_ids,
            'inputs_embeds': None,
            "multimodal_params": [],
        }

        if bool(lora_params):
            inputs['lora_params'] = lora_params

        if spec_metadata is not None:
            inputs['spec_metadata'] = spec_metadata

        return inputs, self.gather_ids_cuda[:num_generation_tokens]

    def _update_target_input_tensors(
            self, num_accepted_tokens_device: torch.Tensor,
            new_tokens_device: torch.Tensor,
            next_draft_tokens_device: torch.Tensor,
            new_tokens_lens_device: torch.Tensor, previous_slots: torch.Tensor,
            total_num_tokens: int, num_extend_reqeust_wo_dummy: int,
            num_tokens_per_extend_request: int,
            previous_batch_draft_tokens: int):
        """
        This function performs in-place updates on position_ids, num_accepted_draft_tokens,
        input_ids, draft_tokens, and offset tensors for speculative decoding extend context operations.
        """

        # Prepare position_ids
        idx_accepted_tokens = self.idx_accepted_tokens_cache[:total_num_tokens]
        self.position_ids_cuda[:total_num_tokens].add_(
            self.num_accepted_draft_tokens_cuda[idx_accepted_tokens] + 1)

        self.num_accepted_draft_tokens_cuda[:num_extend_reqeust_wo_dummy].copy_(
            num_accepted_tokens_device[:num_extend_reqeust_wo_dummy],
            non_blocking=True)

        # Initialize offset tensors to zeros
        self.previous_pos_id_offsets_cuda.mul_(0)
        self.previous_kv_lens_offsets_cuda.mul_(0)

        # Prepare input_ids
        new_tokens = new_tokens_device.transpose(
            0, 1)[previous_slots, :].flatten()
        self.input_ids_cuda[:total_num_tokens].copy_(new_tokens,
                                                     non_blocking=True)

        # Prepare draft tokens
        self.draft_tokens_cuda[:previous_batch_draft_tokens].copy_(
            next_draft_tokens_device[previous_slots, :].flatten(),
            non_blocking=True)

        # Compute kv_len_offsets and update offset tensors
        previous_pos_indices = previous_slots.repeat_interleave(
            num_tokens_per_extend_request)
        self.previous_pos_indices_cuda[:total_num_tokens].copy_(
            previous_pos_indices, non_blocking=True)
        kv_len_offsets_device = new_tokens_lens_device - num_tokens_per_extend_request
        self.previous_pos_id_offsets_cuda[:num_extend_reqeust_wo_dummy *
                                          num_tokens_per_extend_request].copy_(
                                              new_tokens_lens_device[
                                                  self.
                                                  previous_pos_indices_cuda[:
                                                                            total_num_tokens]],
                                              non_blocking=True)
        self.previous_kv_lens_offsets_cuda[:num_extend_reqeust_wo_dummy].copy_(
            kv_len_offsets_device[previous_slots], non_blocking=True)

    def _apply_incremental_update_target(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            new_tensors_device: Optional[SampleStateTensors] = None,
            num_accepted_tokens_device: Optional[torch.Tensor] = None):
        # Extract tensors from new_tensors_device
        new_tokens_device = new_tensors_device.new_tokens  # [batch, 1 + draft_len]
        new_tokens_lens_device = new_tensors_device.new_tokens_lens  # [batch]
        next_draft_tokens_device = new_tensors_device.next_draft_tokens  # [batch, draft_len]

        # Pre-compute constants
        extend_requests = scheduled_requests.generation_requests
        num_extend_requests = len(extend_requests)
        num_tokens_per_extend_request = self.original_max_draft_len + 1
        spec_config = self.spec_config

        prompt_lengths = torch.empty(num_extend_requests,
                                     dtype=torch.int,
                                     device='cpu',
                                     pin_memory=True)
        num_cached_tokens_per_seq = torch.empty(num_extend_requests,
                                                dtype=torch.int,
                                                device='cpu',
                                                pin_memory=True)
        previous_batch_indices = torch.empty(num_extend_requests,
                                             dtype=torch.int,
                                             device='cpu',
                                             pin_memory=True)

        request_accepted_path = {}
        num_extend_dummy_requests = 0
        num_previous_batch = 0

        use_extend_ctx = (self.enable_spec_decode
                          and spec_config.spec_dec_mode.extend_ctx(
                              self.attn_backend) and spec_config.is_linear_tree)

        for idx, request in enumerate(extend_requests):
            request_accepted_path[request.py_request_id] = \
                request.py_num_accepted_draft_tokens_indices

            base_past_seen = request.max_beam_num_tokens - 1

            if use_extend_ctx:
                # We're treating the prompt lengths as context requests here, so
                # the prompt lens should not include the cached tokens.
                prompt_lengths[idx] = num_tokens_per_extend_request
            else:
                prompt_lengths[idx] = request.py_prompt_len

            if request.is_dummy:
                num_cached_tokens_per_seq[idx] = base_past_seen
                request.cached_tokens = base_past_seen
                num_extend_dummy_requests += 1
            else:
                # Request has previous tensor
                previous_batch_indices[
                    num_previous_batch] = request.py_batch_idx
                num_previous_batch += 1

                num_cached_tokens_per_seq[
                    idx] = base_past_seen + num_tokens_per_extend_request
                request.cached_tokens = num_cached_tokens_per_seq[idx].item()

            request.py_batch_idx = request.py_seq_slot

        num_extend_reqeust_wo_dummy = num_extend_requests - num_extend_dummy_requests
        total_num_tokens = num_extend_reqeust_wo_dummy * num_tokens_per_extend_request

        previous_slots = self.previous_batch_indices_cuda[:num_previous_batch]
        previous_slots.copy_(previous_batch_indices[:num_previous_batch],
                             non_blocking=True)

        prompt_lengths = prompt_lengths.tolist()
        num_cached_tokens_per_seq = num_cached_tokens_per_seq.tolist()

        previous_batch_draft_tokens = num_extend_reqeust_wo_dummy * self.runtime_draft_len

        self._update_target_input_tensors(
            num_accepted_tokens_device=num_accepted_tokens_device,
            new_tokens_device=new_tokens_device,
            next_draft_tokens_device=next_draft_tokens_device,
            new_tokens_lens_device=new_tokens_lens_device,
            previous_slots=previous_slots,
            total_num_tokens=total_num_tokens,
            num_extend_reqeust_wo_dummy=num_extend_reqeust_wo_dummy,
            num_tokens_per_extend_request=num_tokens_per_extend_request,
            previous_batch_draft_tokens=previous_batch_draft_tokens)

        # Prepare spec_metadata
        num_generation_tokens = num_extend_requests * num_tokens_per_extend_request
        if spec_metadata is not None:
            total_draft_lens = self.max_total_draft_tokens * num_extend_requests
            spec_metadata.draft_tokens = self.draft_tokens_cuda[:
                                                                total_draft_lens]
            spec_metadata.gather_ids = self.gather_ids_cuda[:total_num_tokens]
            spec_metadata.num_accepted_draft_tokens = self.num_accepted_draft_tokens_cuda[:
                                                                                          num_extend_requests]

        # Determine if we're using extend_ctx mode for linear tree decoding
        num_extend_ctx_requests = 0
        if self.enable_spec_decode and spec_config.spec_dec_mode.extend_ctx(
                self.attn_backend) and spec_config.is_linear_tree:
            num_extend_ctx_requests = num_extend_requests

        virtual_num_tokens = num_generation_tokens
        lora_params = self._prepare_incremental_update_metadata(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            prompt_lengths=prompt_lengths,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            total_num_tokens=virtual_num_tokens,
            num_generation_tokens=num_generation_tokens,
            request_accepted_path=request_accepted_path,
            num_extend_ctx_requests=num_extend_ctx_requests)

        # No padding because there are only generation requests.
        attn_metadata.padded_num_tokens = None
        if self.enable_attention_dp:
            attn_metadata.all_rank_num_tokens = self._get_all_rank_num_tokens(
                attn_metadata)

        final_position_ids = self.position_ids_cuda[:
                                                    virtual_num_tokens].unsqueeze(
                                                        0)

        # Prepare inputs
        # Note: multimodal_params is always empty for incremental updates because:
        # - This function only processes generation requests (no context requests)
        # - Multimodal data (images/videos) is only needed during context/prefill phase
        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:virtual_num_tokens],
            'position_ids': final_position_ids,
            'inputs_embeds': None,
            "multimodal_params": [],
        }

        if bool(lora_params):
            inputs['lora_params'] = lora_params

        if spec_metadata is not None:
            inputs['spec_metadata'] = spec_metadata

        return inputs, self.gather_ids_cuda[:num_generation_tokens]

    def _prepare_tp_inputs(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            new_tensors_device: Optional[SampleStateTensors] = None,
            cache_indirection_buffer: Optional[torch.Tensor] = None,
            num_accepted_tokens_device: Optional[torch.Tensor] = None,
            req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None,
            resource_manager: Optional[ResourceManager] = None):
        """
        Prepare inputs for Pytorch Model.
        """

        new_tokens_device, new_tokens_lens_device, next_draft_tokens_device = None, None, None
        if new_tensors_device is not None:
            # speculative decoding cases: [batch, 1 + draft_len], others: [batch]
            new_tokens_device = new_tensors_device.new_tokens
            # When using overlap scheduler with speculative decoding, the target model's inputs would be SampleStateTensorsMTP.
            if isinstance(new_tensors_device, SampleStateTensorsMTP):
                assert self.enable_spec_decode and not self.is_draft_model
                new_tokens_lens_device = new_tensors_device.new_tokens_lens  # [batch]
                next_draft_tokens_device = new_tensors_device.next_draft_tokens  # [batch, draft_len]

        # Must be before the update of py_batch_idx
        if self.guided_decoder is not None:
            self.guided_decoder.add_batch(scheduled_requests,
                                          new_tokens=new_tokens_device)

        if self._can_use_incremental_update(scheduled_requests,
                                            new_tokens_device,
                                            next_draft_tokens_device):
            return self._apply_incremental_update(
                scheduled_requests, kv_cache_manager, attn_metadata,
                spec_metadata, new_tensors_device, cache_indirection_buffer,
                num_accepted_tokens_device, req_id_to_old_request,
                resource_manager)

        # if new_tensors_device exist, input_ids will only contain new context tokens
        input_ids = []  # per sequence
        sequence_lengths = []  # per sequence
        prompt_lengths = []  # per sequence
        request_ids = []  # per request
        gather_ids = []
        position_ids = []  # per sequence
        num_cached_tokens_per_seq = []  # per sequence
        draft_tokens = []
        draft_lens = []
        gen_request_seq_slots = []  # per generation request
        multimodal_params_list = []
        mrope_position_ids = []
        num_accepted_draft_tokens = []  # per request
        # if using tree decoding, we need to store the request type and accepted path for each request,
        # which will be used to update the hidden_states_read_indices.
        request_accepted_path = {}  # per request

        # Variables for updating the inputs of draft model
        # Base values for gather_ids computation
        first_draft_base_gather_ids = []
        # seq_slots to index into num_accepted_tokens_device
        first_draft_seq_slots = []
        # Indices in the num_accepted_draft_tokens list
        first_draft_request_indices = []

        # (start_idx, end_idx, seq_slot) for context requests
        context_input_ids_positions = []
        # (start_idx, end_idx, seq_slot) for first_draft requests
        first_draft_input_ids_positions = []

        for request in scheduled_requests.context_requests:
            request_ids.append(request.py_request_id)
            all_prompt_tokens = request.get_tokens(0)
            draft_lens.append(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            prompt_tokens = all_prompt_tokens[begin_compute:end_compute]
            position_ids.extend(
                range(begin_compute, begin_compute + len(prompt_tokens)))

            # Track position for updating the inputs of draft model
            if self.is_draft_model and num_accepted_tokens_device is not None:
                start_idx = len(input_ids)
                input_ids.extend(prompt_tokens)
                end_idx = len(input_ids)
                slot_idx = req_id_to_old_request[
                    request.py_request_id].py_seq_slot
                context_input_ids_positions.append(
                    (start_idx, end_idx - 1,
                     slot_idx))  # end_idx-1 is the last token position
            else:
                input_ids.extend(prompt_tokens)

            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(prompt_tokens))
            num_accepted_draft_tokens.append(len(prompt_tokens) - 1)
            request_accepted_path[
                request.
                py_request_id] = request.py_num_accepted_draft_tokens_indices
            prompt_lengths.append(len(prompt_tokens))
            past_seen_token_num = begin_compute
            num_cached_tokens_per_seq.append(past_seen_token_num)
            request.cached_tokens = num_cached_tokens_per_seq[-1]

            # Multimodal
            py_multimodal_runtime = MultimodalRuntimeData(
                mm_token_lengths=request.multimodal_lengths,
                mm_token_positions=request.multimodal_positions,
                past_seen_token_num=past_seen_token_num,
                chunk_end_pos=end_compute,
                special_token_offsets=request.py_multimodal_data.get(
                    'special_token_offsets', []),
            ) if request.multimodal_hashes is not None else None

            multimodal_params = MultimodalParams(
                multimodal_data=request.py_multimodal_data,
                multimodal_runtime=py_multimodal_runtime)
            if multimodal_params.has_content():
                if self.use_mrope:
                    ctx_mrope_position_ids = multimodal_params.multimodal_data[
                        'mrope_config'][
                            'mrope_position_ids'][:, :,
                                                  begin_compute:begin_compute +
                                                  len(prompt_tokens)]
                    mrope_position_ids.append(ctx_mrope_position_ids)

                # TODO: Visit later to decide the appropriate position of sending multimodal data & selectively sending multimodal data
                multimodal_params.to_device("multimodal_data",
                                            "cuda",
                                            pin_memory=True,
                                            target_keywords=getattr(
                                                self.model,
                                                "multimodal_data_device_paths",
                                                None))

                #re-assign the multimodal_data to the request after to_device for generation requests
                request.py_multimodal_data = multimodal_params.multimodal_data
                multimodal_params_list.append(multimodal_params)

            request.py_batch_idx = request.py_seq_slot

        if len(multimodal_params_list) > 0:
            # discard the text token indices as it only includes context tokens at this moment
            _, mm_token_indices = self._prepare_multimodal_indices(input_ids)
        else:
            mm_token_indices = None
        num_ctx_requests = len(scheduled_requests.context_requests)
        num_ctx_tokens = len(input_ids)

        # Requests with draft tokens are treated like extend requests. Dummy extend requests should be
        # at the end of extend_requests.
        extend_requests = []
        extend_dummy_requests = []
        generation_requests = []
        first_draft_requests = []
        for request in scheduled_requests.generation_requests:
            if get_draft_token_length(
                    request) > 0 or next_draft_tokens_device is not None:
                if request.is_dummy:
                    extend_dummy_requests.append(request)
                else:
                    extend_requests.append(request)
            elif request.py_is_first_draft:
                first_draft_requests.append(request)
            else:
                generation_requests.append(request)
        extend_requests += extend_dummy_requests

        spec_config = self.spec_config if self.enable_spec_decode else None
        if not self._disable_overlap_scheduler and spec_config is not None:
            assert spec_config.spec_dec_mode.support_overlap_scheduler(
            ), f"{spec_config.decoding_type} does not support overlap scheduler"

        spec_resource_manager, spec_tree_manager = None, None
        if spec_config is not None:
            spec_resource_manager = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if spec_resource_manager is not None and hasattr(
                    spec_resource_manager, 'spec_tree_manager'):
                spec_tree_manager = spec_resource_manager.spec_tree_manager

        # will contain previous batch indices of generation requests
        previous_batch_indices = []
        previous_pos_indices = []
        for request in extend_requests:
            request_ids.append(request.py_request_id)
            request_accepted_path[
                request.
                py_request_id] = request.py_num_accepted_draft_tokens_indices
            # the request has no previous tensor:
            # (1) next_draft_tokens_device is None, which means overlap scheduler is disabled; or
            # (2) a dummy request; or
            # (3) the first step in the generation server of disaggregated serving
            if next_draft_tokens_device is None or request.is_dummy or request.py_batch_idx is None:
                # get token ids, including input token ids and draft token ids. For these dummy requests,
                # no need to copy the token ids.
                if not (request.is_attention_dp_dummy
                        or request.is_cuda_graph_dummy):
                    input_ids.append(request.get_last_tokens(0))
                    input_ids.extend(request.py_draft_tokens)
                    draft_tokens.extend(request.py_draft_tokens)
                # get other ids and lengths
                num_draft_tokens = get_draft_token_length(request)
                past_seen_token_num = request.max_beam_num_tokens - 1
                draft_lens.append(num_draft_tokens)
                if self.enable_spec_decode and spec_config.spec_dec_mode.extend_ctx(
                        self.attn_backend) and spec_config.is_linear_tree:
                    # We're treating the prompt lengths as context requests here, so
                    # the the prompt lens should not include the cached tokens.
                    prompt_lengths.append(1 + num_draft_tokens)
                else:
                    prompt_lengths.append(request.py_prompt_len)

                sequence_lengths.append(1 + num_draft_tokens)
                num_accepted_draft_tokens.append(num_draft_tokens)
                gather_ids.extend(
                    list(
                        range(len(position_ids),
                              len(position_ids) + 1 + self.runtime_draft_len)))
                # For the target model + tree decoding
                if not self.is_draft_model and not spec_config.is_linear_tree:
                    assert spec_tree_manager is not None
                    assert num_draft_tokens == spec_tree_manager.max_total_draft_tokens
                    position_ids.extend(
                        past_seen_token_num +
                        spec_tree_manager.spec_dec_position_offsets[
                            0]  # [max_total_draft_tokens + 1]
                    )
                else:
                    position_ids.extend(
                        list(
                            range(past_seen_token_num,
                                  past_seen_token_num + 1 + num_draft_tokens)))
                num_cached_tokens_per_seq.append(past_seen_token_num)
                request.cached_tokens = num_cached_tokens_per_seq[-1]
                # update batch index
                request.py_batch_idx = request.py_seq_slot
            else:
                # update batch index
                previous_batch_idx = request.py_batch_idx
                request.py_batch_idx = request.py_seq_slot
                # inputs
                # overlap scheduler can only support the speculative decoding
                # methods with a fixed number of draft tokens
                sequence_lengths.append(1 + self.runtime_draft_len)
                num_accepted_draft_tokens.append(
                    request.py_num_accepted_draft_tokens)
                past_seen_token_num = request.max_beam_num_tokens - 1
                draft_lens.append(self.runtime_draft_len)
                gather_ids.extend(
                    list(
                        range(len(position_ids),
                              len(position_ids) + 1 + self.runtime_draft_len)))
                # For the target model + tree decoding
                if not self.is_draft_model and not spec_config.is_linear_tree:
                    assert spec_tree_manager is not None
                    position_ids.extend(
                        past_seen_token_num +
                        spec_tree_manager.spec_dec_position_offsets[
                            0]  # [max_total_draft_tokens + 1]
                    )
                else:
                    position_ids.extend(
                        list(
                            range(
                                past_seen_token_num, past_seen_token_num + 1 +
                                self.runtime_draft_len)))
                # previous tensor
                previous_batch_indices.append(previous_batch_idx)
                previous_pos_indices.extend([previous_batch_idx] *
                                            (1 + self.runtime_draft_len))

                num_cached_tokens_per_seq.append(past_seen_token_num +
                                                 self.runtime_draft_len + 1)
                request.cached_tokens = num_cached_tokens_per_seq[-1]
                if self.enable_spec_decode and spec_config.spec_dec_mode.extend_ctx(
                        self.attn_backend) and spec_config.is_linear_tree:
                    prompt_lengths.append(1 + self.runtime_draft_len)
                else:
                    prompt_lengths.append(request.py_prompt_len)

        for request in first_draft_requests:
            request_ids.append(request.py_request_id)
            all_prompt_tokens = request.get_tokens(0)
            draft_lens.append(0)
            begin_compute = len(
                all_prompt_tokens) - self.original_max_draft_len - 1
            end_compute = begin_compute + self.original_max_draft_len + 1
            prompt_tokens = all_prompt_tokens[begin_compute:end_compute]
            position_ids.extend(
                range(begin_compute, begin_compute + len(prompt_tokens)))

            # Track position for updating the inputs of draft model
            if self.is_draft_model and num_accepted_tokens_device is not None:
                start_idx = len(input_ids)
                input_ids.extend(prompt_tokens)
                end_idx = len(input_ids)
                # For first_draft, we need to replace the last original_max_draft_len+1 tokens
                slot_idx = req_id_to_old_request[
                    request.py_request_id].py_seq_slot
                first_draft_input_ids_positions.append(
                    (start_idx, end_idx, slot_idx))

                # Store info for GPU computation of gather_ids and num_accepted_draft_tokens
                base_gather_id = len(
                    input_ids) - 1 - self.original_max_draft_len
                # Placeholder, will be corrected on GPU
                gather_ids.append(base_gather_id)
                first_draft_base_gather_ids.append(base_gather_id)
                first_draft_seq_slots.append(slot_idx)
                first_draft_request_indices.append(
                    len(num_accepted_draft_tokens))

                # Placeholder, will be corrected on GPU
                num_accepted_draft_tokens.append(0)
            else:
                input_ids.extend(prompt_tokens)
                gather_ids.append(
                    len(input_ids) - 1 - (self.original_max_draft_len -
                                          request.py_num_accepted_draft_tokens))
                num_accepted_draft_tokens.append(
                    request.py_num_accepted_draft_tokens)

            sequence_lengths.append(1 + self.original_max_draft_len)
            request_accepted_path[
                request.
                py_request_id] = request.py_num_accepted_draft_tokens_indices
            prompt_lengths.append(request.py_prompt_len)
            past_seen_token_num = begin_compute
            num_cached_tokens_per_seq.append(past_seen_token_num)

            # update batch index
            request.py_batch_idx = request.py_seq_slot

        helix_is_inactive_rank, helix_position_offsets = [], []
        for request in generation_requests:
            request_ids.append(request.py_request_id)
            beam_width = request.sampling_config.beam_width
            for beam in range(beam_width):
                # the request has no previous tensor:
                # (1) new_tokens_device is None, which means overlap scheduler is disabled; or
                # (2) a dummy request; or
                # (3) the first step in the generation server of disaggregated serving
                if new_tokens_device is None or request.is_dummy or request.py_batch_idx is None:
                    # skip adding input_ids of CUDA graph dummy requests so that new_tokens_device
                    # can be aligned to the correct positions.
                    if not request.is_cuda_graph_dummy:
                        # Track position for GPU update (draft model only)
                        if self.is_draft_model and num_accepted_tokens_device is not None:
                            start_idx = len(input_ids)
                            input_ids.append(request.get_last_tokens(beam))
                            end_idx = len(input_ids)
                            slot_idx = req_id_to_old_request[
                                request.py_request_id].py_seq_slot
                            first_draft_input_ids_positions.append(
                                (start_idx, end_idx, slot_idx))
                        else:
                            input_ids.append(request.get_last_tokens(beam))
                    past_seen_token_num = request.max_beam_num_tokens - 1
                else:
                    # the request has previous tensor
                    # previous_batch_indices is used per request, not per beam
                    # Only append it once for the first beam of each request
                    first_beam = 0
                    if beam == first_beam:
                        previous_batch_indices.append(request.py_batch_idx)
                    past_seen_token_num = request.max_beam_num_tokens

                position_id = past_seen_token_num
                if self.mapping.has_cp_helix():
                    assert not self.is_warmup, "Warmup is not called for helix parallelism."
                    # We compute a global position_id because each helix rank has only a subset of
                    # tokens for a sequence.
                    position_id = request.total_input_len_cp + request.py_decoding_iter - 1
                    if request.py_helix_is_inactive_rank:
                        past_seen_token_num = request.seqlen_this_rank_cp
                    else:
                        # Discount the token added to active rank in resource manager as it hasn't
                        # been previously seen.
                        past_seen_token_num = request.seqlen_this_rank_cp - 1

                    # Update helix-specific parameters.
                    helix_is_inactive_rank.append(
                        request.py_helix_is_inactive_rank)
                    helix_position_offsets.append(position_id)

                position_ids.append(position_id)
                num_cached_tokens_per_seq.append(past_seen_token_num)
                request.cached_tokens = num_cached_tokens_per_seq[-1]
                prompt_lengths.append(request.py_prompt_len)
                draft_lens.append(0)
                sequence_lengths.append(1)
                num_accepted_draft_tokens.append(0)
                gather_ids.append(len(position_ids) - 1)

                # Multimodal
                multimodal_params = MultimodalParams(
                    multimodal_data=request.py_multimodal_data)
                multimodal_params.strip_for_generation()
                if multimodal_params.has_content():
                    if self.use_mrope:
                        mrope_position_deltas = multimodal_params.multimodal_data[
                            'mrope_config']['mrope_position_deltas']
                        # NOTE: Expanding position_ids to 3D tensor who is using mrope
                        gen_mrope_position_ids = (past_seen_token_num +
                                                  mrope_position_deltas).expand(
                                                      3, 1, 1)
                        mrope_position_ids.append(gen_mrope_position_ids)
                        if mrope_position_deltas.device.type == "cpu":
                            multimodal_params.to_device(
                                "multimodal_data",
                                "cuda",
                                pin_memory=True,
                                target_keywords=[
                                    "mrope_config.mrope_position_deltas"
                                ])
                        multimodal_params_list.append(multimodal_params)

            request.py_batch_idx = request.py_seq_slot
            # Do not add a gen_request_seq_slot for CUDA graph dummy requests
            # to prevent access errors due to None values
            if not request.is_cuda_graph_dummy:
                gen_request_seq_slots.append(request.py_seq_slot)

        previous_batch_len = len(previous_batch_indices)

        def previous_seq_slots_device():
            previous_batch_indices_host = torch.tensor(previous_batch_indices,
                                                       dtype=torch.int,
                                                       pin_memory=True)
            previous_slots = self.previous_batch_indices_cuda[:
                                                              previous_batch_len]
            previous_slots.copy_(previous_batch_indices_host, non_blocking=True)
            return previous_slots

        num_tokens = len(input_ids)
        num_draft_tokens = len(draft_tokens)
        total_num_tokens = len(position_ids)
        assert total_num_tokens <= self.max_num_tokens, (
            f"total_num_tokens ({total_num_tokens}) should be less than or equal to max_num_tokens ({self.max_num_tokens})"
        )
        # if exist requests that do not have previous batch, copy input_ids and draft_tokens
        if num_tokens > 0:
            input_ids = torch.tensor(input_ids,
                                     dtype=torch.int,
                                     pin_memory=True)
            self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

            # Update input_ids_cuda with new tokens from new_tensors_device (draft model only)
            if self.is_draft_model and num_accepted_tokens_device is not None:
                # For context requests: replace the last token with new_tensors_device[0, seq_slot, 0]
                if len(context_input_ids_positions) > 0:
                    # Build tensors on CPU first, then copy to GPU to avoid implicit sync
                    num_ctx_positions = len(context_input_ids_positions)
                    ctx_token_indices_cpu = torch.tensor([
                        last_token_idx
                        for _, last_token_idx, _ in context_input_ids_positions
                    ],
                                                         dtype=torch.long,
                                                         pin_memory=True)
                    ctx_seq_slots_cpu = torch.tensor([
                        seq_slot
                        for _, _, seq_slot in context_input_ids_positions
                    ],
                                                     dtype=torch.long,
                                                     pin_memory=True)
                    # Copy to pre-allocated GPU buffers
                    self.draft_ctx_token_indices_cuda[:num_ctx_positions].copy_(
                        ctx_token_indices_cpu, non_blocking=True)
                    self.draft_ctx_seq_slots_cuda[:num_ctx_positions].copy_(
                        ctx_seq_slots_cpu, non_blocking=True)
                    self.input_ids_cuda[
                        self.
                        draft_ctx_token_indices_cuda[:num_ctx_positions]] = new_tensors_device.new_tokens[
                            0,
                            self.draft_ctx_seq_slots_cuda[:num_ctx_positions],
                            0]

                # For first_draft requests: replace the last (original_max_draft_len+1) tokens
                # with new_tensors_device[:, seq_slot, 0]
                if len(first_draft_input_ids_positions) > 0:
                    # All first_draft requests have same token length (original_max_draft_len + 1)
                    # Build index tensors on CPU first, then copy to GPU to avoid implicit sync
                    num_requests = len(first_draft_input_ids_positions)
                    tokens_per_request = first_draft_input_ids_positions[0][
                        1] - first_draft_input_ids_positions[0][0]

                    # Create flat index array for all tokens to update on CPU
                    all_indices = []
                    all_seq_slots = []
                    for start_idx, end_idx, seq_slot in first_draft_input_ids_positions:
                        all_indices.extend(range(start_idx, end_idx))
                        all_seq_slots.extend([seq_slot] * (end_idx - start_idx))

                    # Create CPU tensors with pinned memory
                    total_tokens = len(all_indices)
                    idx_tensor_cpu = torch.tensor(all_indices,
                                                  dtype=torch.long,
                                                  pin_memory=True)
                    seq_slots_tensor_cpu = torch.tensor(all_seq_slots,
                                                        dtype=torch.long,
                                                        pin_memory=True)

                    # Copy to pre-allocated GPU buffers
                    self.draft_first_draft_indices_cuda[:total_tokens].copy_(
                        idx_tensor_cpu, non_blocking=True)
                    self.draft_first_draft_seq_slots_cuda[:total_tokens].copy_(
                        seq_slots_tensor_cpu, non_blocking=True)

                    # Create token position indices (repeating 0..tokens_per_request for each request)
                    token_positions = torch.arange(
                        tokens_per_request, dtype=torch.long,
                        device='cuda').repeat(num_requests)

                    self.input_ids_cuda[
                        self.
                        draft_first_draft_indices_cuda[:total_tokens]] = new_tensors_device.new_tokens[
                            token_positions, self.
                            draft_first_draft_seq_slots_cuda[:total_tokens], 0]

        if num_draft_tokens > 0:
            draft_tokens = torch.tensor(draft_tokens,
                                        dtype=torch.int,
                                        pin_memory=True)
            self.draft_tokens_cuda[:len(draft_tokens)].copy_(draft_tokens,
                                                             non_blocking=True)
        if self.is_spec_decode and len(num_accepted_draft_tokens) > 0:
            num_accepted_draft_tokens = torch.tensor(num_accepted_draft_tokens,
                                                     dtype=torch.int,
                                                     pin_memory=True)
            self.num_accepted_draft_tokens_cuda[:len(
                num_accepted_draft_tokens)].copy_(num_accepted_draft_tokens,
                                                  non_blocking=True)

            # Update num_accepted_draft_tokens_cuda for first_draft_requests directly from num_accepted_tokens_device (draft model only)
            if self.is_draft_model and len(first_draft_seq_slots) > 0:
                # Build tensors on CPU first, then copy to GPU to avoid implicit sync
                num_first_draft = len(first_draft_seq_slots)
                first_draft_seq_slots_cpu = torch.tensor(first_draft_seq_slots,
                                                         dtype=torch.int,
                                                         pin_memory=True)
                first_draft_indices_cpu = torch.tensor(
                    first_draft_request_indices,
                    dtype=torch.int,
                    pin_memory=True)

                # Copy to pre-allocated GPU buffers
                self.draft_seq_slots_buffer_cuda[:num_first_draft].copy_(
                    first_draft_seq_slots_cpu, non_blocking=True)
                self.draft_request_indices_buffer_cuda[:num_first_draft].copy_(
                    first_draft_indices_cpu, non_blocking=True)

                # Extract accepted tokens for first_draft requests from device tensor
                accepted_tokens = num_accepted_tokens_device[
                    self.draft_seq_slots_buffer_cuda[:num_first_draft]]
                # Update the correct positions in num_accepted_draft_tokens_cuda
                self.num_accepted_draft_tokens_cuda[
                    self.
                    draft_request_indices_buffer_cuda[:
                                                      num_first_draft]] = accepted_tokens
        if next_draft_tokens_device is not None:
            # Initialize these two values to zeros
            self.previous_pos_id_offsets_cuda *= 0
            self.previous_kv_lens_offsets_cuda *= 0

            if previous_batch_len > 0:
                previous_slots = previous_seq_slots_device()
                # previous input ids
                previous_batch_tokens = previous_batch_len * (
                    1 + self.runtime_draft_len)
                new_tokens = new_tokens_device.transpose(
                    0, 1)[previous_slots, :].flatten()
                self.input_ids_cuda[num_tokens:num_tokens +
                                    previous_batch_tokens].copy_(
                                        new_tokens, non_blocking=True)
                # previous draft tokens
                previous_batch_draft_tokens = previous_batch_len * self.runtime_draft_len
                self.draft_tokens_cuda[num_draft_tokens:num_draft_tokens +
                                       previous_batch_draft_tokens].copy_(
                                           next_draft_tokens_device[
                                               previous_slots, :].flatten(),
                                           non_blocking=True)
                # prepare data for the preprocess inputs
                kv_len_offsets_device = new_tokens_lens_device - self.runtime_draft_len - 1
                previous_pos_indices_host = torch.tensor(previous_pos_indices,
                                                         dtype=torch.int,
                                                         pin_memory=True)
                self.previous_pos_indices_cuda[0:previous_batch_tokens].copy_(
                    previous_pos_indices_host, non_blocking=True)

                # The order of requests in a batch: [context requests, generation requests]
                # generation requests: ['requests that do not have previous batch', 'requests that already have previous batch', 'dummy requests']
                #   1) 'requests that do not have previous batch': disable overlap scheduler or the first step in the generation server of disaggregated serving.
                #   2) 'requests that already have previous batch': previous iteration's requests.
                #   3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                # Therefore, both of self.previous_pos_id_offsets_cuda and self.previous_kv_lens_offsets_cuda are also 3 segments.
                #   For 1) 'requests that do not have previous batch': disable overlap scheduler or the first step in the generation server of disaggregated serving.
                #       Set these requests' previous_pos_id_offsets and previous_kv_lens_offsets to '0' to skip the value changes in _preprocess_inputs.
                #       Already set to '0' during initialization.
                #   For 2) 'requests that already have previous batch': enable overlap scheduler.
                #       Set their previous_pos_id_offsets and previous_kv_lens_offsets according to new_tokens_lens_device and kv_len_offsets_device.
                #   For 3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                #       Already set to '0' during initialization.

                num_extend_reqeust_wo_dummy = len(extend_requests) - len(
                    extend_dummy_requests)
                self.previous_pos_id_offsets_cuda[
                    (num_extend_reqeust_wo_dummy - previous_batch_len) *
                    (1 + self.runtime_draft_len):num_extend_reqeust_wo_dummy *
                    (1 + self.runtime_draft_len)].copy_(
                        new_tokens_lens_device[self.previous_pos_indices_cuda[
                            0:previous_batch_tokens]],
                        non_blocking=True)

                self.previous_kv_lens_offsets_cuda[
                    num_extend_reqeust_wo_dummy -
                    previous_batch_len:num_extend_reqeust_wo_dummy].copy_(
                        kv_len_offsets_device[previous_slots],
                        non_blocking=True)

        elif new_tokens_device is not None:
            seq_slots_device = previous_seq_slots_device()
            max_draft_len = max(draft_lens)
            new_tokens = new_tokens_device[:max_draft_len + 1,
                                           seq_slots_device, :self.
                                           max_beam_width]
            self.input_ids_cuda[num_tokens:num_tokens +
                                previous_batch_len * self.max_beam_width].copy_(
                                    new_tokens.flatten(), non_blocking=True)

        if (not self._disable_overlap_scheduler
                and next_draft_tokens_device is None
                and len(extend_requests) > 0):
            # During warmup, for those generation requests, we don't have previous tensors,
            # so we need to set the previous_pos_id_offsets and previous_kv_lens_offsets to zeros
            # to skip the value changes in _preprocess_inputs. Otherwise, there will be illegal memory access
            # when writing key/values to the KV cache.
            self.previous_pos_id_offsets_cuda *= 0
            self.previous_kv_lens_offsets_cuda *= 0

        if self.use_mrope and mrope_position_ids:
            # NOTE: self.use_mrope is enough for differentiating whether to use mrope_position_ids but
            # `_create_dummy_context_requests` from `kv_cache_creater` makes an exception that I can not add multimodal_data to the dummy_request
            # so that we only replace position_ids with mrope_position_ids when it is not a dummy request and for models who is using mrope.
            mrope_position_ids = torch.cat(mrope_position_ids, dim=-1)
            if mrope_position_ids.device.type == "cpu":
                mrope_position_ids = mrope_position_ids.pin_memory()
            self.mrope_position_ids_cuda[:, :, :total_num_tokens].copy_(
                mrope_position_ids[:, :, :total_num_tokens], non_blocking=True)
            final_position_ids = self.mrope_position_ids_cuda[:, :, :
                                                              total_num_tokens]
        else:
            position_ids = torch.tensor(position_ids,
                                        dtype=torch.int,
                                        pin_memory=True)
            self.position_ids_cuda[:total_num_tokens].copy_(position_ids,
                                                            non_blocking=True)
            final_position_ids = self.position_ids_cuda[:
                                                        total_num_tokens].unsqueeze(
                                                            0)

        if self.enable_spec_decode:
            self.gather_ids_cuda[:len(gather_ids)].copy_(torch.tensor(
                gather_ids, dtype=torch.int, pin_memory=True),
                                                         non_blocking=True)

            # Update gather_ids for first_draft_requests on GPU (draft model only)
            if self.is_draft_model and len(first_draft_seq_slots) > 0:
                # Build tensors on CPU first, then copy to GPU to avoid implicit sync
                num_first_draft = len(first_draft_seq_slots)
                first_draft_seq_slots_cpu = torch.tensor(first_draft_seq_slots,
                                                         dtype=torch.int,
                                                         pin_memory=True)
                first_draft_indices_cpu = torch.tensor(
                    first_draft_request_indices,
                    dtype=torch.int,
                    pin_memory=True)

                # Copy to pre-allocated GPU buffers
                self.draft_seq_slots_buffer_cuda[:num_first_draft].copy_(
                    first_draft_seq_slots_cpu, non_blocking=True)
                self.draft_request_indices_buffer_cuda[:num_first_draft].copy_(
                    first_draft_indices_cpu, non_blocking=True)

                # Extract accepted tokens for first_draft requests from device tensor
                accepted_tokens = num_accepted_tokens_device[
                    self.draft_seq_slots_buffer_cuda[:num_first_draft]]
                # Update gather_ids: gather_id = base_gather_id + num_accepted_tokens
                # (since gather_id = len(input_ids) - 1 - (max_draft_len - num_accepted))
                self.gather_ids_cuda[
                    self.
                    draft_request_indices_buffer_cuda[:
                                                      num_first_draft]] += accepted_tokens

        if self.mapping.has_cp_helix():
            attn_metadata.update_helix_param(
                helix_position_offsets=helix_position_offsets,
                helix_is_inactive_rank=helix_is_inactive_rank,
            )

        if not attn_metadata.is_cuda_graph:
            # Assumes seq lens do not change between CUDA graph invocations. This applies
            # to draft sequences too. This means that all draft sequences must be padded.
            attn_metadata.seq_lens = torch.tensor(
                sequence_lengths,
                dtype=torch.int,
                pin_memory=True,
            )

        num_generation_requests = len(gen_request_seq_slots)
        # Cache indirection is only used for beam search on generation requests
        if self.use_beam_search and num_generation_requests > 0:
            # CUDA Graph needs to set beam width during warmup (where the graph is captured), to ensure that cache indirection buffer is correctly picked up by the CUDA graph
            is_cuda_graph_during_warmup = self.is_warmup and attn_metadata.is_cuda_graph
            if cache_indirection_buffer is not None:
                #Copy cache indirection to local buffer with offsets changing:  seq_slots[i] -> i
                # Convert to GPU tensor to avoid implicit sync
                gen_request_seq_slots_tensor = torch.tensor(
                    gen_request_seq_slots, dtype=torch.long, device='cuda')
                self.cache_indirection_attention[:num_generation_requests].copy_(
                    cache_indirection_buffer[gen_request_seq_slots_tensor])
            if cache_indirection_buffer is not None or is_cuda_graph_during_warmup:
                attn_metadata.beam_width = self.max_beam_width
        else:
            attn_metadata.beam_width = 1

        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = len(scheduled_requests.context_requests)
        # Use num_chunked_ctx_requests to record the number of extend context requests,
        # so that we can update the kv_lens_cuda correctly in _preprocess_inputs.
        attn_metadata.num_chunked_ctx_requests = 0
        if self.enable_spec_decode and spec_config.spec_dec_mode.extend_ctx(
                self.attn_backend) and spec_config.is_linear_tree:
            # For the tree decoding, we want to use XQA to process the draft tokens for the target model.
            # Therefore, we do not treat them as the chunked context requests.
            attn_metadata.num_contexts += len(extend_requests)
            attn_metadata.num_chunked_ctx_requests = len(extend_requests)

        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            num_extra_kv_tokens=get_num_extra_kv_tokens(spec_config))
        attn_metadata.kv_cache_manager = kv_cache_manager

        attn_metadata.prepare()

        lora_params = self._get_lora_params_from_requests(
            scheduled_requests, attn_metadata)

        attn_all_rank_num_tokens = self._get_all_rank_num_tokens(attn_metadata)
        padded_num_tokens, can_run_piecewise_cuda_graph, attn_all_rank_num_tokens = self._get_padding_params(
            total_num_tokens, num_ctx_requests, attn_all_rank_num_tokens)
        set_per_request_piecewise_cuda_graph_flag(can_run_piecewise_cuda_graph)
        attn_metadata.padded_num_tokens = padded_num_tokens if padded_num_tokens != total_num_tokens else None

        virtual_num_tokens = total_num_tokens
        if attn_metadata.padded_num_tokens is not None:
            self.input_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens
            if self.use_mrope and mrope_position_ids:
                self.mrope_position_ids_cuda[
                    total_num_tokens:padded_num_tokens].fill_(0)
                final_position_ids = self.mrope_position_ids_cuda[:, :, :
                                                                  virtual_num_tokens]
            else:
                self.position_ids_cuda[
                    total_num_tokens:padded_num_tokens].fill_(0)
                final_position_ids = self.position_ids_cuda[:
                                                            virtual_num_tokens].unsqueeze(
                                                                0)

        if self.enable_attention_dp:
            attn_metadata.all_rank_num_tokens = attn_all_rank_num_tokens

        # Prepare inputs
        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:virtual_num_tokens],
            'position_ids': final_position_ids,
            'inputs_embeds': None,
            "multimodal_params": multimodal_params_list,
        }

        if bool(lora_params):
            inputs['lora_params'] = lora_params

        if spec_metadata is not None:
            total_draft_lens = sum(draft_lens)
            spec_metadata.draft_tokens = self.draft_tokens_cuda[:
                                                                total_draft_lens]
            spec_metadata.request_ids = request_ids
            spec_metadata.gather_ids = self.gather_ids_cuda[:len(gather_ids)]
            spec_metadata.num_generations = len(
                scheduled_requests.generation_requests)
            spec_metadata.num_tokens = total_num_tokens
            spec_metadata.seq_lens = sequence_lengths
            spec_metadata.num_accepted_draft_tokens = self.num_accepted_draft_tokens_cuda[:len(
                num_accepted_draft_tokens)]
            if isinstance(spec_metadata, Eagle3SpecMetadata):
                spec_metadata.request_accepted_path = request_accepted_path
            if isinstance(spec_metadata, Eagle3OneModelSpecMetadata):
                spec_metadata.populate_sampling_params_for_one_model(
                    scheduled_requests.all_requests())
            spec_metadata.prepare()
            inputs['spec_metadata'] = spec_metadata

            if self.enable_attention_dp:
                all_rank_num_tokens = self.dist.tp_allgather(
                    [spec_metadata.num_tokens,
                     len(sequence_lengths)])

                spec_all_rank_num_tokens = [
                    item[0] for item in all_rank_num_tokens
                ]
                all_rank_num_seqs = [item[1] for item in all_rank_num_tokens]
                spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
                spec_metadata.all_rank_num_seqs = all_rank_num_seqs

        if mm_token_indices is not None:
            mask = torch.ones(total_num_tokens, dtype=torch.bool)
            mask[mm_token_indices] = False
            inputs['mm_token_indices'] = mm_token_indices.pin_memory().to(
                "cuda", non_blocking=True)
            inputs['text_token_indices'] = torch.where(mask)[0].pin_memory().to(
                "cuda", non_blocking=True)

        num_generation_tokens = len(generation_requests) + len(
            extend_requests) + sum(draft_lens) + len(first_draft_requests)
        self.iter_states['num_ctx_requests'] = num_ctx_requests
        self.iter_states['num_ctx_tokens'] = num_ctx_tokens
        self.iter_states['num_generation_tokens'] = num_generation_tokens

        if not self.is_warmup:
            self.previous_request_ids = [
                request.py_request_id
                for request in scheduled_requests.generation_requests
            ]
            self.has_previous_device_draft = next_draft_tokens_device is not None

        return inputs, self.gather_ids_cuda[:len(
            gather_ids)] if self.enable_spec_decode else None

    def _prepare_tp_inputs_no_cache(
            self,
            scheduled_requests: ScheduledRequests,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None):
        """
        Prepare inputs for Pytorch Model.
        """
        sequence_lengths = []
        input_ids = []
        gather_ids = []
        position_ids = []
        multi_modal_data = []
        draft_lens = []
        request_ids = []
        multimodal_params_list = []

        for request in scheduled_requests.context_requests:
            prompt_tokens = request.get_tokens(0)
            input_ids.extend(prompt_tokens)
            request_ids.append(request.py_request_id)
            if request.position_ids is None:
                position_ids.extend(range(len(prompt_tokens)))
            else:
                position_ids.extend(request.position_ids)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(prompt_tokens))
            draft_lens.append(0)
            multimodal_embedding = request.multimodal_embedding
            if multimodal_embedding is not None:
                multi_modal_data.append(multimodal_embedding)

            # Multimodal
            if request.py_multimodal_data is not None:
                multimodal_params = MultimodalParams(
                    multimodal_data=request.py_multimodal_data, )
                multimodal_params.to_device("multimodal_data",
                                            "cuda",
                                            pin_memory=True)
                multimodal_params_list.append(multimodal_params)

            request.py_batch_idx = request.py_seq_slot

        num_tokens = len(input_ids)
        assert num_tokens <= self.max_num_tokens, (
            "num_tokens should be less than or equal to max_num_tokens")
        input_ids = torch.tensor(input_ids, dtype=torch.int, pin_memory=True)
        self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int,
                                    pin_memory=True)
        self.position_ids_cuda[:num_tokens].copy_(position_ids,
                                                  non_blocking=True)
        if self.enable_spec_decode:
            self.gather_ids_cuda[:len(gather_ids)].copy_(torch.tensor(
                gather_ids, dtype=torch.int, pin_memory=True),
                                                         non_blocking=True)

        if not attn_metadata.is_cuda_graph:
            # No need to overwrite seq lens when using CUDA graphs -
            # CUDA graphs are only used for pure decoding batches
            # and have static batch size, so the seqlens never change.
            # Note that it's important to not free the seq_lens_cuda
            # buffer once the graph has been captured also - this will invalidate
            # the graph and force an expensive recapture.
            attn_metadata.seq_lens = torch.tensor(
                sequence_lengths,
                dtype=torch.int,
                pin_memory=True,
            )

        attn_metadata.num_contexts = len(scheduled_requests.context_requests)

        attn_all_rank_num_tokens = self._get_all_rank_num_tokens(attn_metadata)
        padded_num_tokens, can_run_piecewise_cuda_graph, attn_all_rank_num_tokens = self._get_padding_params(
            num_tokens, attn_metadata.num_contexts, attn_all_rank_num_tokens)
        set_per_request_piecewise_cuda_graph_flag(can_run_piecewise_cuda_graph)
        attn_metadata.padded_num_tokens = padded_num_tokens if padded_num_tokens != num_tokens else None

        if self.enable_attention_dp:
            attn_metadata.all_rank_num_tokens = attn_all_rank_num_tokens

        virtual_num_tokens = num_tokens
        if attn_metadata.padded_num_tokens is not None:
            self.input_ids_cuda[num_tokens:padded_num_tokens].fill_(0)
            self.position_ids_cuda[num_tokens:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens

        # this is for no cache attention, not for dummy attention
        if attn_metadata.kv_cache_manager is None:
            assert isinstance(
                attn_metadata,
                (VanillaAttentionMetadata, TrtllmAttentionMetadata)
            ), "Only vanilla and trtllm attention metadata are supported for no cache attention for now"
            attn_metadata.max_seq_len = self.max_seq_len
            attn_metadata.request_ids = request_ids
            attn_metadata.prepare()

        lora_params = self._get_lora_params_from_requests(
            scheduled_requests, attn_metadata)

        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:virtual_num_tokens],
            'position_ids':
            self.position_ids_cuda[:virtual_num_tokens].unsqueeze(0),
            'inputs_embeds': None,
            "multimodal_params": multimodal_params_list
        }

        if bool(lora_params):
            inputs['lora_params'] = lora_params

        if spec_metadata is not None:
            total_draft_lens = sum(draft_lens)
            spec_metadata.draft_tokens = self.draft_tokens_cuda[:
                                                                total_draft_lens]
            spec_metadata.request_ids = request_ids
            spec_metadata.gather_ids = self.gather_ids_cuda[:len(gather_ids)]
            spec_metadata.num_generations = len(
                scheduled_requests.generation_requests)
            spec_metadata.num_tokens = num_tokens
            spec_metadata.seq_lens = sequence_lengths
            spec_metadata.prepare()
            inputs['spec_metadata'] = spec_metadata

        # support attention dp
        if self.enable_attention_dp:
            if spec_metadata is not None:
                all_rank_num_tokens = self.dist.tp_allgather([
                    attn_metadata.num_tokens, spec_metadata.num_tokens,
                    len(sequence_lengths)
                ])
                attn_all_rank_num_tokens = [
                    item[0] for item in all_rank_num_tokens
                ]
                spec_all_rank_num_tokens = [
                    item[1] for item in all_rank_num_tokens
                ]
                all_rank_num_seqs = [item[2] for item in all_rank_num_tokens]
                attn_metadata.all_rank_num_tokens = attn_all_rank_num_tokens
                spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
                spec_metadata.all_rank_num_seqs = all_rank_num_seqs
            else:
                all_rank_num_tokens = self.dist.tp_allgather(
                    attn_metadata.num_tokens)
                attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        return inputs, None

    def _prepare_star_attention_inputs(self,
                                       scheduled_requests: ScheduledRequests,
                                       kv_cache_manager,
                                       attn_metadata: AttentionMetadata):
        """
        Prepare inputs for Pytorch Model.
        """
        sequence_lengths = []
        input_ids = []
        prompt_lengths = []
        request_ids = []
        gather_ids = []
        position_ids = []
        # for star attention, we need customized block ids
        block_ids_per_seq = []
        num_cached_tokens_per_seq = []
        for request in scheduled_requests.context_requests:
            request_ids.append(request.py_request_id)
            prompt_lengths.append(request.py_prompt_len)

            ctx_iter = request.ctx_iters
            ctx_blocks = request.ctx_blocks
            ctx_position_blocks = request.ctx_position_blocks
            all_cache_indices = kv_cache_manager.get_cache_indices(request)
            ### for the first iteration, we need to construct input as C[0]  + C[1]
            if ctx_iter == 0:
                input_id = ctx_blocks[0] + ctx_blocks[1]
                num_kv_blocks = kv_cache_manager.get_num_kv_blocks(
                    len(input_id))
                position_id = ctx_position_blocks[0] + ctx_position_blocks[1]
                past_seen_token_num = 0
                all_cache_indices = all_cache_indices[:num_kv_blocks]
            else:
                input_id = ctx_blocks[ctx_iter + 1]
                position_id = ctx_position_blocks[ctx_iter + 1]
                ## compute C[0] and ctx_blocks
                if ctx_iter < len(ctx_blocks) - 2:
                    if self.mapping.cp_rank == 0:
                        anchor_block = ctx_blocks[
                            0][:self.mapping.cp_config['cp_anchor_size']]
                    else:
                        anchor_block = ctx_blocks[0]

                    num_anchor_cache_blocks = kv_cache_manager.get_num_kv_blocks(
                        len(anchor_block))
                    ### we need to construct input as C[0] + C[x+i]
                    #C0 has been computed, can be shared across all blocks
                    anchor_indices = all_cache_indices[:num_anchor_cache_blocks]

                    # C1~C[ctx_iter] should be skipped in the computation
                    token_start_idx = sum(
                        len(block) for block in ctx_blocks[:(ctx_iter + 1)])
                    token_end_idx = sum(
                        len(block) for block in ctx_blocks[:(ctx_iter + 2)])
                    block_start_idx = kv_cache_manager.get_num_kv_blocks(
                        token_start_idx)
                    block_end_idx = kv_cache_manager.get_num_kv_blocks(
                        token_end_idx)
                    block_indices = all_cache_indices[
                        block_start_idx:block_end_idx]

                    all_cache_indices = anchor_indices + block_indices
                    past_seen_token_num = len(
                        anchor_block)  ### C[0] can be reused
                else:
                    continue
            input_ids.extend(input_id)
            position_ids.extend(position_id)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(input_id))
            block_ids_per_seq.extend([all_cache_indices])
            num_cached_tokens_per_seq.append(past_seen_token_num)
            request.cached_tokens = num_cached_tokens_per_seq[-1]
        num_contexts = len(sequence_lengths)
        for request in scheduled_requests.context_requests:
            ctx_iter = request.ctx_iters
            ctx_blocks = request.ctx_blocks
            ctx_position_blocks = request.ctx_position_blocks
            num_kvblocks_per_ctx_block = kv_cache_manager.get_num_kv_blocks(
                len(ctx_blocks[0]))
            all_cache_indices = kv_cache_manager.get_cache_indices(request)
            ### for query phase
            ## compute C[0~blocks] with query for the first rank
            ## compute C[1~blocks] with query for the other rank
            if ctx_iter == len(ctx_blocks) - 2:
                input_id = ctx_blocks[ctx_iter + 1]
                position_id = ctx_position_blocks[ctx_iter + 1]
                if self.mapping.cp_rank == 0:
                    past_seen_token_num = sum(
                        len(block) for block in ctx_blocks[:ctx_iter + 1])
                else:
                    # drop C0, free KV cache
                    all_cache_indices = all_cache_indices[
                        num_kvblocks_per_ctx_block:]
                    past_seen_token_num = sum(
                        len(block) for block in ctx_blocks[1:ctx_iter + 1])
                if self.mapping.cp_rank == self.mapping.cp_size - 1:
                    num_kv_tokens = past_seen_token_num + len(input_id)
                else:
                    num_kv_tokens = past_seen_token_num  # don't need to append/compute query's kv cache
                num_kv_blocks = kv_cache_manager.get_num_kv_blocks(
                    num_kv_tokens)
                all_cache_indices = all_cache_indices[:num_kv_blocks]
            else:
                continue

            input_ids.extend(input_id)
            position_ids.extend(position_id)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(input_id))
            block_ids_per_seq.extend([all_cache_indices])
            num_cached_tokens_per_seq.append(past_seen_token_num)
            request.cached_tokens = num_cached_tokens_per_seq[-1]
        num_queries = len(sequence_lengths) - num_contexts

        # Requests with draft tokens are treated like extend requests.
        extend_requests = [
            request for request in scheduled_requests.generation_requests
            if request.py_draft_tokens
        ]
        generation_requests = [
            request for request in scheduled_requests.generation_requests
            if not request.py_draft_tokens
        ]
        is_spec_decode = len(extend_requests) > 0
        assert not is_spec_decode, 'star attention does not support draft tokens now.'

        for request in generation_requests:
            request_ids.append(request.py_request_id)
            prompt_lengths.append(request.py_prompt_len)

            input_token_id = request.get_token(0, request.get_num_tokens(0) - 1)
            input_ids.append(input_token_id)
            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(1)
            past_seen_token_num = request.max_beam_num_tokens - 1

            # for sp, we only increase the generated KV cache for the last rank
            ctx_blocks = request.ctx_blocks
            total_anchor_ctx_query_len = sum(
                [len(block) for block in ctx_blocks])
            query_len = len(ctx_blocks[-1])
            anchor_len = len(ctx_blocks[0])

            if self.mapping.cp_size == 1:
                past_seen_token_num = total_anchor_ctx_query_len + request.gen_iters
                num_kv_tokens = past_seen_token_num + 1
            else:
                if self.mapping.cp_rank == self.mapping.cp_size - 1:
                    past_seen_token_num = total_anchor_ctx_query_len + request.gen_iters - anchor_len
                    num_kv_tokens = past_seen_token_num + 1
                else:
                    if self.mapping.cp_rank != 0:
                        past_seen_token_num = total_anchor_ctx_query_len - anchor_len - query_len
                    else:
                        past_seen_token_num = total_anchor_ctx_query_len - query_len
                    num_kv_tokens = past_seen_token_num  # don't need to append kv cache

            num_kv_blocks = kv_cache_manager.get_num_kv_blocks(num_kv_tokens)
            all_cache_indices = kv_cache_manager.get_cache_indices(request)
            if self.mapping.cp_rank != 0:
                num_kvblocks_per_ctx_block = kv_cache_manager.get_num_kv_blocks(
                    anchor_len)
                all_cache_indices = all_cache_indices[
                    num_kvblocks_per_ctx_block:]
            cache_indices = all_cache_indices[:num_kv_blocks]
            last_query_pos_id = request.ctx_position_blocks[-1][-1]
            position_ids.append(last_query_pos_id + request.gen_iters + 1)
            block_ids_per_seq.extend([all_cache_indices])
            num_cached_tokens_per_seq.append(past_seen_token_num)
            request.cached_tokens = num_cached_tokens_per_seq[-1]

        num_tokens = len(input_ids)
        assert num_tokens <= self.max_num_tokens, (
            "num_tokens should be less than or equal to max_num_tokens")
        input_ids = torch.tensor(input_ids, dtype=torch.int, pin_memory=True)
        self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int,
                                    pin_memory=True)
        self.position_ids_cuda[:num_tokens].copy_(position_ids,
                                                  non_blocking=True)

        if not attn_metadata.is_cuda_graph:
            # No need to overwrite seq lens when using CUDA graphs -
            # CUDA graphs are only used for pure decoding batches
            # and have static batch size, so the seqlens never change.
            # Note that it's important to not free the seq_lens_cuda
            # buffer once the graph has been captured also - this will invalidate
            # the graph and force an expensive recapture.
            attn_metadata.seq_lens = torch.tensor(
                sequence_lengths,
                dtype=torch.int,
                pin_memory=True,
            )

        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = num_contexts
        attn_metadata.num_queries = num_queries

        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            block_ids_per_seq=block_ids_per_seq,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq)

        attn_metadata.kv_cache_manager = kv_cache_manager

        attn_metadata.prepare()
        if self.enable_attention_dp:
            all_rank_num_tokens = self.dist.tp_allgather(
                attn_metadata.num_tokens)
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        return {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:num_tokens],
            'position_ids': self.position_ids_cuda[:num_tokens].unsqueeze(0),
            'inputs_embeds': None
        }, gather_ids if is_spec_decode else None

    def _get_lora_params_from_requests(self,
                                       scheduled_requests: ScheduledRequests,
                                       attn_metadata: AttentionMetadata):
        '''
        lora_params: dict
        {
            layer_id: dict
            {
                module_id: dict
                {
                    adapter_size: torch tensor: int
                    weight_pointers: torch tensor: int64
                }
            }
        }
        '''
        lora_params = {}
        tmp_lora_params = {}

        request_list = scheduled_requests.context_requests + scheduled_requests.generation_requests

        # trace all requests to get the union set of the lora params
        for request in request_list:
            if request.py_lora_task_layer_module_configs is None:
                continue

            for module in request.py_lora_task_layer_module_configs:
                module_id = module.module_id
                layer_id = module.layer_id

                if layer_id not in lora_params:
                    lora_params[layer_id] = {}
                if module_id not in lora_params[layer_id]:
                    lora_params[layer_id][module_id] = {
                        'adapter_size': [],
                        'weight_pointers': [],
                    }

                scaling_vec_pointer = module.scaling_vec_pointer
                if scaling_vec_pointer is None:
                    scaling_vec_pointer = 0
                tmp_lora_params[(request.py_request_id, layer_id,
                                 module_id)] = {
                                     'adapter_size': [module.adapter_size],
                                     'weight_pointers': [
                                         module.weights_in_pointer,
                                         module.weights_out_pointer,
                                         scaling_vec_pointer
                                     ],
                                 }

        for request in request_list:
            # Need to set default values for this case
            if request.py_lora_task_layer_module_configs is None:
                for layer_id in lora_params:
                    for module_id in lora_params[layer_id]:
                        current_lora_params = lora_params[layer_id][module_id]
                        current_lora_params['adapter_size'].append(0)
                        current_lora_params['weight_pointers'] += [0, 0, 0]

            else:
                for layer_id in lora_params:
                    for module_id in lora_params[layer_id]:
                        current_tmp_lora_params = tmp_lora_params.get(
                            (request.py_request_id, layer_id, module_id), None)
                        current_lora_params = lora_params[layer_id][module_id]
                        if current_tmp_lora_params is None:
                            current_lora_params['adapter_size'].append(0)
                            current_lora_params['weight_pointers'] += [0, 0, 0]
                        else:
                            current_lora_params[
                                'adapter_size'] += current_tmp_lora_params[
                                    'adapter_size']
                            current_lora_params[
                                'weight_pointers'] += current_tmp_lora_params[
                                    'weight_pointers']

        for layer_id in lora_params:
            for module_id in lora_params[layer_id]:
                current_lora_params = lora_params[layer_id][module_id]
                current_lora_params['adapter_size'] = torch.IntTensor(
                    current_lora_params['adapter_size'])
                current_lora_params['weight_pointers'] = torch.LongTensor(
                    current_lora_params['weight_pointers'])

        if lora_params:
            lora_params['host_request_types'] = attn_metadata.host_request_types
            lora_params['prompt_lens_cpu'] = attn_metadata.prompt_lens_cpu
            lora_params['num_seqs'] = attn_metadata.num_seqs

        return lora_params

    @nvtx_range("_prepare_inputs")
    def _prepare_inputs(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            new_tensors_device: Optional[SampleStateTensors] = None,
            cache_indirection_buffer: Optional[torch.Tensor] = None,
            num_accepted_tokens_device: Optional[torch.Tensor] = None,
            req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None,
            resource_manager: Optional[ResourceManager] = None):
        if self.mapping is not None and 'cp_type' in self.mapping.cp_config:
            cp_type = self.mapping.cp_config['cp_type']
            if CpType.STAR == cp_type:
                return self._prepare_star_attention_inputs(
                    scheduled_requests, kv_cache_manager, attn_metadata)
            elif CpType.HELIX == cp_type:
                # Take the usual route of _prepare_tp_inputs.
                pass
            else:
                raise NotImplementedError(
                    f"Unsupported cp_type {getattr(cp_type, 'name', cp_type)}.")

        return self._prepare_tp_inputs(scheduled_requests, kv_cache_manager,
                                       attn_metadata, spec_metadata,
                                       new_tensors_device,
                                       cache_indirection_buffer,
                                       num_accepted_tokens_device,
                                       req_id_to_old_request, resource_manager)

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self.model.extra_attrs)
    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tensors_device: Optional[SampleStateTensors] = None,
                gather_context_logits: bool = False,
                cache_indirection_buffer: Optional[torch.Tensor] = None,
                spec_decoding_tensor: Optional[SpecDecodingTensor] = None,
                num_accepted_tokens_device: Optional[torch.Tensor] = None,
                req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None):
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)

        attn_metadata = self._set_up_attn_metadata(kv_cache_manager)
        if self.enable_spec_decode:
            spec_resource_manager = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            spec_tree_manager = None
            if isinstance(spec_resource_manager, Eagle3ResourceManager):
                spec_tree_manager = spec_resource_manager.spec_tree_manager
            spec_metadata = self._set_up_spec_metadata(spec_resource_manager,
                                                       no_cache=kv_cache_manager
                                                       is None)
            # attn_metadata now depends on spec_metadata since it determines the shape/content of spec_dec parameter Tensors
            is_spec_dec_mode = spec_metadata.spec_dec_mode.attention_need_spec_dec_mode(
                spec_resource_manager, self.is_draft_model, self.attn_backend,
                self.model_is_wrapped)
            attn_metadata.update_spec_dec_param(
                batch_size=scheduled_requests.batch_size,
                is_spec_decoding_enabled=is_spec_dec_mode,
                is_spec_dec_tree=spec_metadata.is_spec_dec_tree,
                is_spec_dec_dynamic_tree=spec_metadata.is_spec_dec_dynamic_tree,
                max_draft_len=self.original_max_draft_len,
                max_total_draft_tokens=self.original_max_total_draft_tokens,
                model_is_wrapped=self.model_is_wrapped,
                spec_metadata=spec_metadata,
                spec_tree_manager=spec_tree_manager,
                spec_decoding_tensor=spec_decoding_tensor)
        else:
            spec_resource_manager = None
            spec_metadata = None

        moe_load_balancer: MoeLoadBalancer = getattr(self, 'moe_load_balancer',
                                                     None)

        if kv_cache_manager is None:
            inputs, gather_ids = self._prepare_tp_inputs_no_cache(
                scheduled_requests, attn_metadata, spec_metadata)

            with MoeLoadBalancerIterContext(moe_load_balancer):
                # Special handling for multimodal encoder only mode
                if self.llm_args.mm_encoder_only:
                    return self._forward_step_mm_encoder_only(
                        inputs, scheduled_requests)
                else:
                    return self._forward_step(inputs, gather_ids,
                                              gather_context_logits)
        with self.cuda_graph_runner.pad_batch(
                scheduled_requests, resource_manager,
                self.runtime_draft_len) as padded_requests:

            maybe_attn_metadata, maybe_spec_metadata, key = self.cuda_graph_runner.maybe_get_cuda_graph(
                padded_requests,
                enable_spec_decode=self.enable_spec_decode,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_tokens_cuda=self.draft_tokens_cuda
                if self.is_spec_decode else None,
                new_tensors_device=new_tensors_device,
                spec_resource_manager=spec_resource_manager,
            )
            can_run_graph = key is not None
            if can_run_graph:
                attn_metadata = maybe_attn_metadata
                spec_metadata = maybe_spec_metadata
            else:
                attn_metadata = self.attn_metadata
                if self.enable_spec_decode:
                    spec_metadata = self.spec_metadata
                else:
                    spec_metadata = None

            inputs, gather_ids = self._prepare_inputs(
                padded_requests, kv_cache_manager, attn_metadata, spec_metadata,
                new_tensors_device, cache_indirection_buffer,
                num_accepted_tokens_device, req_id_to_old_request,
                resource_manager)

            with with_shared_pool(self.cuda_graph_runner.get_graph_pool()):
                if not can_run_graph:
                    # Fallback to eager execution if graph was not used
                    with MoeLoadBalancerIterContext(moe_load_balancer):
                        outputs = self._forward_step(inputs, gather_ids,
                                                     gather_context_logits)
                else:
                    if self.cuda_graph_runner.needs_capture(key):

                        def capture_forward_fn(inputs: Dict[str, Any]):
                            with MoeLoadBalancerIterContext(moe_load_balancer):
                                return self._forward_step(
                                    inputs,
                                    gather_ids=gather_ids,
                                    gather_context_logits=gather_context_logits)

                        def capture_postprocess_fn(inputs: Dict[str, Any]):
                            self._postprocess_inputs(inputs)

                        self.cuda_graph_runner.capture(
                            key,
                            capture_forward_fn,
                            inputs,
                            enable_spec_decode=self.enable_spec_decode,
                            postprocess_fn=capture_postprocess_fn)

                        # here we don't need to use context since cuda graph capture didn't run kernel.
                        # maybe we need a cleaner way to do this.
                        outputs = self.cuda_graph_runner.replay(key, inputs)
                    else:
                        with MoeLoadBalancerIterContext(moe_load_balancer):
                            outputs = self.cuda_graph_runner.replay(key, inputs)

            if self.forward_pass_callable is not None:
                self.forward_pass_callable()

            self._execute_logit_post_processors(scheduled_requests, outputs)

            return outputs

    def model_forward(self, **kwargs):
        attrs = get_model_extra_attrs()
        assert attrs is not None, "Model extra attrs is not set"
        attrs["attention_metadata"] = weakref.ref(kwargs['attn_metadata'])
        attrs.update(self.model.model_config.extra_attrs)

        if self._torch_compile_backend is not None:
            # Register aux streams and events to model extra attrs.
            # The streams and events are list which could be updated during compilation.
            attrs["aux_streams"] = weakref.ref(self.backend_num_streams)
            attrs["events"] = weakref.ref(self._torch_compile_backend.events)
            attrs["global_stream"] = torch.cuda.current_stream()

        if is_trace_enabled("TLLM_TRACE_MODEL_FORWARD"):
            return trace_func(self.model.forward)(**kwargs)
        else:
            return self.model.forward(**kwargs)

    @nvtx_range("_forward_step")
    def _forward_step(self,
                      inputs: Dict[str, Any],
                      gather_ids: Optional[torch.Tensor],
                      gather_context_logits: bool = False) -> Dict[str, Any]:
        inputs = self._preprocess_inputs(inputs)
        if inputs.get('spec_metadata', None):
            gather_ids = inputs['spec_metadata'].gather_ids

        # For simplicity, just return all the the logits if we have special gather_ids
        # from speculative decoding.
        outputs = self.model_forward(
            **inputs,
            return_context_logits=gather_ids is not None
            or gather_context_logits,
        )

        if self.without_logits:
            return outputs

        if isinstance(outputs, dict):
            # If the model returns a dict, get the logits from it. All other keys are kept.
            logits = outputs.get('logits', None)
            # If the logits are not found, no further processing is needed.
            if logits is None:
                return outputs
        else:
            # If the model returns a single tensor, assume it is the logits and wrap it in a dict.
            logits = outputs
            outputs = {'logits': logits}

        # If we have special gather_ids, gather the logits
        if gather_ids is not None:
            outputs['logits'] = logits[gather_ids]

        return outputs

    @nvtx_range("_forward_step_mm_encoder_only")
    def _forward_step_mm_encoder_only(
            self, inputs: Dict[str, Any],
            scheduled_requests: ScheduledRequests) -> Dict[str, Any]:
        """Forward step for multimodal encoder only mode - returns mm_embeddings instead of logits."""
        # Get multimodal parameters from inputs
        multimodal_params = inputs.get("multimodal_params", [])
        if not multimodal_params or len(multimodal_params) == 0:
            # Return empty embeddings if no multimodal data
            return {'mm_embeddings': []}
        if getattr(scheduled_requests.context_requests[0], 'multimodal_lengths',
                   None) is None:
            multimodal_chunks = None
        else:
            multimodal_chunks = [
                sum(request.multimodal_lengths)
                for request in scheduled_requests.context_requests
                if request.multimodal_lengths is not None
            ]
        # For mm_encoder_only mode, we only run the vision encoder part
        # The model should be a vision encoder (e.g., Qwen2VisionModelBase)
        mm_embeddings = self.model.forward(multimodal_params)
        assert len(
            mm_embeddings
        ) == 1, "mm_embeddings should be a 1-element list, mix modality (video+image) is not supported"

        if multimodal_chunks is None or len(multimodal_chunks) != len(
                multimodal_params):
            mm_embeddings = list(
                torch.chunk(mm_embeddings[0],
                            len(scheduled_requests.context_requests),
                            dim=0))
        else:
            mm_embeddings = list(
                torch.split(mm_embeddings[0], multimodal_chunks, dim=0))

        # Extract mrope position data from multimodal_params if available
        mrope_position_ids_list = []
        mrope_position_deltas_list = []
        for multimodal_param in multimodal_params:
            mrope_config = multimodal_param.multimodal_data.get(
                'mrope_config', {})
            mrope_position_ids = mrope_config.get('mrope_position_ids')
            mrope_position_deltas = mrope_config.get('mrope_position_deltas')
            if mrope_position_ids is not None:
                mrope_position_ids_list.append(mrope_position_ids)
            if mrope_position_deltas is not None:
                mrope_position_deltas_list.append(mrope_position_deltas)

        result = {'mm_embeddings': mm_embeddings, 'logits': None}
        if mrope_position_ids_list:
            result['mrope_position_ids'] = mrope_position_ids_list
        if mrope_position_deltas_list:
            result['mrope_position_deltas'] = mrope_position_deltas_list

        return result

    def _init_userbuffers(self, hidden_size):
        if self.mapping.tp_size <= 1 or self.mapping.pp_size > 1:
            return False

        # Disable UB for unsupported platforms
        if not ub.ub_supported():
            return False
        # NCCL_SYMMETRIC strategy no longer requires UserBuffer allocator initialization.
        # It uses NCCLWindowAllocator from ncclUtils directly.
        if self.llm_args.allreduce_strategy == "NCCL_SYMMETRIC":
            # Skip UB initialization for NCCL_SYMMETRIC - it uses NCCLWindowAllocator directly
            return False
        ub.initialize_userbuffers_manager(self.mapping.tp_size,
                                          self.mapping.pp_size,
                                          self.mapping.cp_size,
                                          self.mapping.rank,
                                          self.mapping.gpus_per_node,
                                          hidden_size * self.max_num_tokens * 2)

        return True

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        """
        When doing spec decode, sometimes draft models need to share certain weights
        with their target models. Here, we set up such weights by invoking
        self.model.load_weights_from_target_model if such a method exists.
        """
        loader = getattr(self.model, "load_weights_from_target_model", None)
        if callable(loader):
            loader(target_model)

    def _execute_logit_post_processors(self,
                                       scheduled_requests: ScheduledRequests,
                                       outputs: dict):
        """Apply logit post processors (in-place modify outputs Tensors) if any."""

        if not (self.mapping.is_last_pp_rank()):
            return

        if not isinstance(outputs, dict) or "logits" not in outputs:
            # TODO: support models that don't return outputs as dict
            return

        num_ctx_req = len(scheduled_requests.context_requests)
        logits_tensor = outputs["logits"]

        for idx, request in enumerate(scheduled_requests.all_requests()):
            logits_processors = getattr(request, "py_logits_post_processors",
                                        None)
            if not logits_processors:
                continue

            token_ids = request.get_tokens(0)
            if idx < num_ctx_req and request.py_orig_prompt_len < len(
                    token_ids):
                # Skip as we only need to apply logit processor on the last context request
                continue

            logits_row = logits_tensor[idx]
            # Reshape to align w/ the shape used in the TRT backend,
            # so the same logit processors can be used across both backends.
            logits_row = logits_row.view(1, 1, -1)
            token_ids = [token_ids]
            for lp in logits_processors:
                lp_params = inspect.signature(lp).parameters

                assert 4 <= len(lp_params) <= 5, (
                    "Logit post processor signature must match the `LogitsProcessor` interface "
                    "defined in `tensorrtllm.sampling_params`.")
                lp(request.py_request_id, logits_row, token_ids, None, None)

            logits_tensor[idx] = logits_row.view(-1)
