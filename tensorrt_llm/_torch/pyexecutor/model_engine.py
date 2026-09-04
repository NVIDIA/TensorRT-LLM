# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Type,
                    Union)

import torch
import torch._dynamo.config

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._torch.utils import torch_multi_arange
from tensorrt_llm._utils import (global_mpi_rank, is_trace_enabled,
                                 maybe_pin_memory, nvtx_range, prefer_pinned,
                                 release_gc, torch_dtype_to_str, trace_func)
from tensorrt_llm.bindings.internal import \
    batch_manager as batch_manager_bindings
from tensorrt_llm.bindings.internal.runtime import TaskLayerModuleConfig
from tensorrt_llm.inputs.multimodal import (MultimodalParams,
                                            MultimodalRuntimeData,
                                            _has_mm_payload_keys,
                                            check_mm_embed_cumsum_if_needed,
                                            strip_mm_data_for_generation)
from tensorrt_llm.inputs.registry import (BaseMultimodalInputProcessor,
                                          create_input_processor)
from tensorrt_llm.llmapi.llm_args import (CudaGraphConfig, DecodingBaseConfig,
                                          EncodeCudaGraphConfig,
                                          PrefillCudaGraphBackend,
                                          SeqLenAwareSparseAttentionConfig,
                                          TorchCompileConfig, TorchLlmArgs)

# isort: split
from tensorrt_llm.llmapi.llm_args import validate_token_encoder_bucket_config
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.sampling_params import SamplingParams

from ..attention_backend.interface import (AttentionMetadata,
                                           AttentionRuntimeFeatures)
from ..attention_backend.trtllm import TrtllmAttentionMetadata
from ..attention_backend.utils import get_attention_backend
from ..attention_backend.vanilla import VanillaAttentionMetadata
from ..autotuner import AutoTuner, autotune
from ..compilation.backend import Backend
from ..compilation.utils import capture_piecewise_cuda_graph
from ..distributed import Distributed
from ..distributed.communicator import init_pp_comm
from ..memory_buffer_utils import clear_memory_buffers, with_shared_pool
from ..metadata import KVCacheParams
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.modeling_multimodal_mixin import (MultimodalModelMixin,
                                                _build_request_multimodal_input)
from ..models.modeling_multimodal_utils import filter_mm_token_from_input_ids
from ..models.modeling_utils import DecoderModelForCausalLM
from ..modules.mamba.mamba2_metadata import Mamba2Metadata
from ..moe.expert_statistic import ExpertStatistic
from ..moe.fused_moe.moe_load_balancer import (MoeLoadBalancer,
                                               MoeLoadBalancerIterContext)
from ..peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from ..speculative import (SpecMetadata, get_draft_kv_cache_manager,
                           get_num_extra_kv_tokens, get_spec_metadata,
                           prepare_attn_metadata_for_draft_replay,
                           restore_attn_metadata_after_draft_replay,
                           update_spec_config_from_loaded_model)
from ..speculative.dspark_ragged import ragged_gather_index_lists
from ..speculative.eagle3 import Eagle3ResourceManager, Eagle3SpecMetadata
from ..speculative.interface import INVALID_PROMPT_LOOKAHEAD_TOKEN
from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from ..utils import (get_model_extra_attrs,
                     get_per_request_prefill_cuda_graph_flag,
                     set_per_request_prefill_cuda_graph_flag,
                     set_torch_compiling, with_model_extra_attrs)
from .breakable_cuda_graph_runner import BreakableCUDAGraphRunner
from .config_utils import is_mla
from .cuda_graph_runner import (ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM,
                                CUDAGraphRunner, CUDAGraphRunnerConfig,
                                EncoderCUDAGraphRunner,
                                EncoderCUDAGraphRunnerConfig)
from .engine.multimodal import (MultimodalItemScheduler, is_multimodal,
                                mm_encoder_cache_enabled,
                                setup_mm_encoder_attn_metadata)
from .guided_decoder import CapturableGuidedDecoder
from .kv_cache_manager_v2 import KVCacheManagerV2
from .layerwise_nvtx_marker import LayerwiseNvtxMarker
from .llm_request import (LlmRequest, LlmRequestState, get_draft_token_length,
                          get_multimodal_embedding_lengths,
                          get_request_tokens_per_gen_step)
from .mamba_cache_manager import BaseMambaCacheManager, MambaHybridCacheManager
from .model_loader import ModelLoader, _construct_checkpoint_loader
from .resource_manager import (BaseResourceManager, KVCacheManager,
                               PeftCacheManager, ResourceManager,
                               ResourceManagerType)
from .sampler import SampleStateTensors
from .sampler.ops.flashinfer import warmup_sampling_module
from .scheduler import ScheduledRequests
from .trace_log_utils import log_mem_snapshot


def _get_context_prompt_lookahead_token(request: LlmRequest,
                                        chunk_end: int) -> int:
    """Read the prompt token immediately following a context chunk."""
    if chunk_end >= request.py_prompt_len:
        return INVALID_PROMPT_LOOKAHEAD_TOKEN
    return request.get_token(0, chunk_end)


def resolve_mamba_metadata_cls(model: torch.nn.Module) -> Type[Mamba2Metadata]:
    """Resolve the model-specific Mamba metadata class with a default."""
    return getattr(model, 'mamba_metadata_cls', None) or Mamba2Metadata


def _make_single_token_context_graph_batch(
    scheduled_requests: ScheduledRequests,
    is_multimodal_decode_compatible: Optional[Callable[[LlmRequest],
                                                       bool]] = None,
) -> tuple[ScheduledRequests, frozenset[int]]:
    """Build a decode-shaped graph candidate for final one-token contexts.

    Multimodal rows remain fail-closed unless the engine proves that their one
    remaining prompt token is representable by the existing decode provider.
    """
    if scheduled_requests.num_context_requests == 0:
        return scheduled_requests, frozenset()

    context_requests = scheduled_requests.context_requests_last_chunk
    if (scheduled_requests.encoder_requests
            or scheduled_requests.context_requests_chunking):
        return scheduled_requests, frozenset()

    for request in context_requests:
        if (request.context_chunk_size != 1
                or request.context_remaining_length != 1
                or request.context_current_position + 1 != request.py_prompt_len
                or request.py_beam_width != 1
                or get_draft_token_length(request) > 0
                or request.py_is_first_draft or request.is_context_only_request
                or request.is_generation_only_request()
                or request.py_disaggregated_params is not None
                or request.py_mm_encoder_event is not None
                or (request.py_multimodal_data is not None and
                    (is_multimodal_decode_compatible is None
                     or not is_multimodal_decode_compatible(request)))):
            return scheduled_requests, frozenset()

    for request in scheduled_requests.generation_requests:
        if (request.py_beam_width != 1 or get_draft_token_length(request) > 0
                or request.py_is_first_draft
                or request.py_disaggregated_params is not None):
            return scheduled_requests, frozenset()

    graph_batch = ScheduledRequests()
    graph_batch.generation_requests = list(context_requests) + list(
        scheduled_requests.generation_requests)
    graph_batch.paused_requests = list(scheduled_requests.paused_requests)
    promoted_context_request_ids = frozenset(request.py_request_id
                                             for request in context_requests)
    return graph_batch, promoted_context_request_ids


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


def _filter_piecewise_capture_num_tokens(
    candidate_num_tokens: list[int],
    max_num_tokens: int,
    max_batch_size: int,
    max_seq_len: int,
) -> Tuple[list[int], list[int]]:
    """Cap piecewise CUDA graph capture candidates at the engine's reachable
    `num_tokens` ceiling `max_batch_size * (max_seq_len - 1)`
    clamping user-requested sizes above it down to the ceiling.

    Each in-flight request must leave room for at least one decode token,
    so the ceiling is the largest forward-pass `num_tokens` the warmup
    builder can construct. Candidates above the ceiling cannot be
    recorded; clamping them down to the ceiling preserves the user's
    intent (a requested 128 becomes 127 when only 127 is recordable)
    without inventing capture sizes the user never asked
    for. Appending sizes beyond the user's list is harmful: runtime
    padding rounds iterations up to the nearest captured size, so a far
    appended ceiling (e.g. 65536 over a list topping at 13914) would
    make every iteration in the gap execute the full ceiling shape.

    Returns `(kept, unrecordable)` where `kept` is sorted ascending and
    deduped, with above-ceiling candidates clamped to the ceiling.
    `unrecordable` is the sorted unique set of input entries above the
    ceiling but within `max_num_tokens` (the clamped ones, reported so
    the caller's warning fires).
    """
    max_capturable_num_tokens = max(0, max_batch_size * (max_seq_len - 1))
    piecewise_capacity_limit = min(max_num_tokens, max_capturable_num_tokens)
    if piecewise_capacity_limit > 0:
        kept = sorted({
            min(i, piecewise_capacity_limit)
            for i in candidate_num_tokens if 0 < i <= max_num_tokens
        })
    else:
        kept = []
    unrecordable = sorted({
        i
        for i in candidate_num_tokens
        if max_capturable_num_tokens < i <= max_num_tokens
    })
    return kept, unrecordable


# BCG uses the same capture-bucket filtering semantics as PCG.
_filter_prefill_capture_num_tokens = _filter_piecewise_capture_num_tokens


def _filter_cuda_graph_batch_sizes(cuda_graph_batch_sizes: list[int],
                                   max_batch_size: int, max_num_tokens: int,
                                   tokens_per_request: int,
                                   enable_padding: bool) -> list[int]:
    """Drop the batch sizes that exceed the request or token budget.

    `tokens_per_request` is what a single request costs against
    `max_num_tokens`: `1 + max_total_draft_tokens` for a pure decoding batch,
    or the fixed encoder output length for an encoder whose input is a
    fixed-shape per-request feature tensor.
    """
    max_cuda_graph_bs = min(max_batch_size,
                            max_num_tokens // tokens_per_request)
    if max_cuda_graph_bs < 1:
        # Not even a single request fits the token budget, so there is no
        # capturable batch size and padding has nothing to pad to.
        return []

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


def _filter_cuda_graph_num_tokens(cuda_graph_num_tokens: list[int],
                                  max_num_tokens: int,
                                  enable_padding: bool) -> list[int]:
    """Filter encoder CUDA graph total-token counts to the system-wide limit."""
    result = []
    for i, nt in enumerate(cuda_graph_num_tokens):
        if nt <= max_num_tokens:
            result.append(nt)
        else:
            if enable_padding and (i == 0 or result[i - 1] != max_num_tokens):
                logger.warning(
                    "CUDA graph padding is enabled, but one of the given encoder "
                    f"CUDA graph num_tokens ({nt}) is larger than the system "
                    f"max_num_tokens ({max_num_tokens}). We will pad to "
                    f"{max_num_tokens}.")
                result.append(max_num_tokens)
            break
    return result


def _filter_cuda_graph_seq_lens(cuda_graph_seq_lens: list[int],
                                max_seq_len: int,
                                enable_padding: bool) -> list[int]:
    """Filter encoder CUDA graph max sequence lengths to the system-wide limit."""
    result = []
    for i, sl in enumerate(cuda_graph_seq_lens):
        if sl <= max_seq_len:
            result.append(sl)
        else:
            if enable_padding and (i == 0 or result[i - 1] != max_seq_len):
                logger.warning(
                    "CUDA graph padding is enabled, but one of the given encoder "
                    f"CUDA graph seq_lens ({sl}) is larger than the system "
                    f"max_seq_len ({max_seq_len}). We will pad to "
                    f"{max_seq_len}.")
                result.append(max_seq_len)
            break
    return result


_DEEP_GEMM_PDL_CONFIGURED = False

# Arbitrary non-greedy params used to force the advanced-sampling CUDA graph
# warmup capture path.
NON_GREEDY_CAPTURE_SAMPLING_PARAMS = SamplingParams(temperature=0.7,
                                                    top_k=50,
                                                    top_p=0.9)


def _configure_deep_gemm_pdl() -> None:
    global _DEEP_GEMM_PDL_CONFIGURED
    if _DEEP_GEMM_PDL_CONFIGURED:
        return

    from tensorrt_llm import deep_gemm

    deep_gemm.set_pdl(os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1")
    _DEEP_GEMM_PDL_CONFIGURED = True


@contextlib.contextmanager
def _moe_a2a_steady_state_budget_for_capture():
    """Force the steady-state MoE all-to-all budget across CUDA-graph capture.

    The budget is a kernel launch argument, so it is frozen into each captured
    graph. Capture happens inside the warmup window, so without this a replay
    would keep warmup's relaxed deadline for the life of the process.
    """
    _set_moe_a2a_warmup(False)
    try:
        yield
    finally:
        _set_moe_a2a_warmup(True)


def _set_moe_a2a_warmup(in_warmup: bool) -> None:
    """Select the MoE all-to-all completion-flag budget for the current phase.

    No-op when the op is unavailable (older bindings).
    """
    try:
        torch.ops.trtllm.moe_a2a_set_warmup(in_warmup)
        logger.info(f"moe_a2a completion-flag budget: in_warmup={in_warmup}")
    except (AttributeError, RuntimeError) as e:
        logger.warning(
            f"moe_a2a_set_warmup unavailable, the all-to-all timeout "
            f"budget was not switched: {type(e).__name__}: {e}")


class PyTorchModelEngine(ModelEngine):

    def __init__(
        self,
        *,
        model_path: str,
        llm_args: TorchLlmArgs,
        mapping: Optional[Mapping] = None,
        attn_runtime_features: Optional[AttentionRuntimeFeatures] = None,
        dist: Optional[Distributed] = None,
        spec_config: Optional[DecodingBaseConfig] = None,
        is_draft_model: bool = False,
        model: Optional[torch.nn.Module] = None,
        checkpoint_loader: Optional[BaseCheckpointLoader] = None,
        model_weights_memory_tag: Optional[str] = None,
        model_weights_restore_mode=None,
    ):
        _configure_deep_gemm_pdl()

        self.forward_pass_callable = None
        self._cleanup_done = False
        if llm_args.encode_only and llm_args.mm_encoder_only:
            raise ValueError(
                "encode_only and mm_encoder_only are mutually exclusive.")
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
        self.encoder_batch_size = (llm_args.encoder_max_batch_size
                                   if llm_args.encoder_max_batch_size
                                   is not None else self.batch_size)
        # The multimodal encoder token budget falls back to the LLM-side value
        # when unset. It may be raised after model load because atomic MM items
        # cannot be split.
        self.encoder_max_num_tokens = (llm_args.encoder_max_num_tokens
                                       if llm_args.encoder_max_num_tokens
                                       is not None else self.max_num_tokens)

        if checkpoint_loader is None:
            checkpoint_loader = _construct_checkpoint_loader(
                llm_args.backend,
                llm_args.checkpoint_loader,
                llm_args.checkpoint_format,
                mx_config=llm_args.mx_config,
                mx_model_name=llm_args.model,
                checkpoint_io_policy=llm_args.checkpoint_io_policy,
                load_format=llm_args.load_format,
                partial_model_loading=llm_args.is_partial_model_loading,
            )

        self.mapping = mapping
        if mapping.has_pp():
            init_pp_comm(mapping)
        # Disaggregated attention-DP can backfill a batch before the overlap
        # scheduler releases the previous batch's terminal sequence slots.
        from ._util import (compute_max_num_sequences,
                            should_enable_adp_dummy_fixes,
                            should_enable_disagg_adp_overlap_headroom,
                            should_enable_non_overlap_adp_forward_intent,
                            should_enable_scheduler_aware_adp_dummy)
        self._enable_disagg_adp_overlap_headroom = (
            should_enable_disagg_adp_overlap_headroom(
                mapping, llm_args.cache_transceiver_config,
                llm_args.disable_overlap_scheduler))
        self._enable_adp_dummy_fixes = should_enable_adp_dummy_fixes(mapping)
        self.max_num_seq_slots = compute_max_num_sequences(
            mapping,
            self.batch_size,
            llm_args.disable_overlap_scheduler,
            enable_overlap_headroom=self._enable_disagg_adp_overlap_headroom,
        )
        self.dist = dist
        if dist is not None:
            ExpertStatistic.create(self.dist.rank)
        self.llm_args = llm_args
        self.original_max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self.original_max_total_draft_tokens = (
            spec_config.tokens_per_gen_step -
            1) if spec_config is not None else 0
        # Saved before zeroing for draft models; used by update_spec_dec_param.
        self._spec_dec_max_total_draft_tokens = (
            spec_config.max_total_draft_tokens
            if spec_config is not None else 0)

        # Dynamic tree draft loop produces up to K * max_draft_len tokens,
        # which may exceed max_total_draft_tokens. Use the larger value for
        # KV cache reservation only; verify/tree output stays at max_total_draft_tokens.
        if (spec_config is not None
                and getattr(spec_config, 'use_dynamic_tree', False)
                and getattr(spec_config, 'dynamic_tree_max_topK', 0) > 0):
            self.max_draft_loop_tokens = max(
                self.original_max_total_draft_tokens,
                spec_config.dynamic_tree_max_topK * spec_config.max_draft_len)
        else:
            self.max_draft_loop_tokens = self.original_max_total_draft_tokens

        # The draft model won't have any draft tokens attached to
        # generation requests when we invoke it autoregressively
        if spec_config is not None and is_draft_model:
            spec_config.max_draft_len = 0
            spec_config.max_total_draft_tokens = 0
        self.spec_config = spec_config
        self.is_spec_decode = spec_config is not None
        self._dspark_confidence_enabled = bool(
            spec_config is not None
            and getattr(spec_config, "enable_confidence_scheduling", False))
        self._dspark_trims_submitted_tokens = bool(
            self._dspark_confidence_enabled)
        self._dspark_device_windows = bool(
            self._dspark_trims_submitted_tokens and getattr(
                spec_config, "enable_fused_confidence_scheduler", False))
        self.sparse_attention_config = None if is_draft_model else llm_args.sparse_attention_config
        self.enable_spec_decode = self.is_spec_decode
        self.is_draft_model = is_draft_model

        self.attn_runtime_features = attn_runtime_features or AttentionRuntimeFeatures(
        )

        input_processor_kwargs = {}
        video_pruning_rate = llm_args.multimodal_config.video_pruning_rate
        if video_pruning_rate is not None:
            input_processor_kwargs['video_pruning_rate'] = video_pruning_rate
        self.input_processor = create_input_processor(
            model_path,
            tokenizer=None,
            checkpoint_format=llm_args.checkpoint_format,
            trust_remote_code=llm_args.trust_remote_code,
            **input_processor_kwargs)

        self.moe_load_balancer: Optional[MoeLoadBalancer] = None
        self.model_loader: Optional[ModelLoader] = None
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
                model_weights_memory_tag=model_weights_memory_tag,
                model_weights_restore_mode=model_weights_restore_mode,
            )
            # Open checkpoint and load the LLM module object.
            self.model, moe_load_balancer = self.model_loader.load(
                checkpoint_dir=model_path, checkpoint_loader=checkpoint_loader)
            if isinstance(moe_load_balancer, MoeLoadBalancer):
                self.moe_load_balancer = moe_load_balancer
        else:
            self.model = model
        self._validate_breakable_cuda_graph_compatibility()
        pretrained_config = self.model.model_config.pretrained_config
        model_type = getattr(pretrained_config, "model_type", None)
        self._enable_scheduler_aware_adp_dummy = (
            should_enable_scheduler_aware_adp_dummy(
                model_type, mapping, llm_args.disable_overlap_scheduler))
        self._enable_non_overlap_adp_forward_intent = (
            should_enable_non_overlap_adp_forward_intent(
                mapping, llm_args.disable_overlap_scheduler))
        self.sparse_attention_config = self.model.model_config.sparse_attention_config
        # In case that some tests use stub models and override `_load_model`.
        if not hasattr(self.model, 'extra_attrs'):
            self.model.extra_attrs = {}
        # Every MM item-scheduling decision -- policy, capability, feature
        # validation, budget resolution -- lives in engine/multimodal.py; the
        # engine only copies back the three budgets that are external contract.
        mm_item_scheduler = MultimodalItemScheduler.maybe_create(
            llm_args=self.llm_args,
            model=self.model,
            input_processor=self.input_processor,
            encoder_max_num_tokens=self.encoder_max_num_tokens)
        self._mm_item_scheduler = mm_item_scheduler
        # `getattr`-read by py_executor.py and _util.py.
        self.mm_encoder_item_scheduling_enabled = mm_item_scheduler is not None
        self.mm_encoder_output_budget_bytes: Optional[int] = None
        if mm_item_scheduler is not None:
            # The raised encoder token budget is read back off the engine by
            # `_util.py`, and sizes the encoder metadata set up below.
            self.encoder_max_num_tokens = mm_item_scheduler.encoder_max_num_tokens
            self.mm_encoder_output_budget_bytes = mm_item_scheduler.output_budget_bytes
            # Absent, not None, when item scheduling is off: external readers
            # rely on the `getattr` default.
            self.bytes_per_mm_encoder_embedding = mm_item_scheduler.bytes_per_embedding
        setup_mm_encoder_attn_metadata(
            self.model, self.input_processor, self.encoder_max_num_tokens,
            mm_item_scheduler.attention_metadata_capacity
            if mm_item_scheduler is not None else None)
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
        self._is_encode_only = (self.llm_args.encode_only
                                and not self.llm_args.mm_encoder_only)
        if (self._is_encode_only
                and isinstance(self.cuda_graph_config, EncodeCudaGraphConfig)):
            self.encoder_cuda_graph_config = self.cuda_graph_config
        else:
            self.encoder_cuda_graph_config = (
                self.llm_args.encoder_cuda_graph_config)

        if (isinstance(self.cuda_graph_config, EncodeCudaGraphConfig)
                and self._is_encoder_decoder_model()):
            logger.warning(
                "EncodeCudaGraphConfig is not supported for encoder-decoder "
                "models through cuda_graph_config. Use DecodeCudaGraphConfig "
                "for cuda_graph_config and configure encoder graphs through "
                "encoder_cuda_graph_config. Decoder CUDA graphs will be "
                "disabled.")
            self.cuda_graph_config = None

        cuda_graph_batch_sizes = self.cuda_graph_config.batch_sizes if self.cuda_graph_config else CudaGraphConfig.model_fields[
            'batch_sizes'].default
        cuda_graph_padding_enabled = self.cuda_graph_config.enable_padding if self.cuda_graph_config else CudaGraphConfig.model_fields[
            'enable_padding'].default

        # CUDA graph detection for encoder-decoder models and encoder-only models.
        # Decode configs do not define these encoder-specific bucket fields.
        encoder_cuda_graph_batch_sizes = (
            self.encoder_cuda_graph_config.batch_sizes
            if self.encoder_cuda_graph_config is not None else [])
        encoder_cuda_graph_num_tokens = (
            self.encoder_cuda_graph_config.num_tokens
            if self.encoder_cuda_graph_config is not None else [])
        encoder_cuda_graph_seq_lens = (self.encoder_cuda_graph_config.seq_lens
                                       if self.encoder_cuda_graph_config
                                       is not None else [])
        encoder_cuda_graph_padding_enabled = (
            self.encoder_cuda_graph_config.enable_padding
            if self.encoder_cuda_graph_config is not None else False)

        self._check_encoder_graph_bucket_config(encoder_cuda_graph_num_tokens,
                                                encoder_cuda_graph_seq_lens)

        self._cuda_graph_padding_enabled = cuda_graph_padding_enabled

        decode_tokens_per_request = 1 + self.original_max_total_draft_tokens
        self._cuda_graph_batch_sizes = _filter_cuda_graph_batch_sizes(
            cuda_graph_batch_sizes, self.batch_size, self.max_num_tokens,
            decode_tokens_per_request,
            self._cuda_graph_padding_enabled) if cuda_graph_batch_sizes else []

        self._max_cuda_graph_batch_size = (self._cuda_graph_batch_sizes[-1] if
                                           self._cuda_graph_batch_sizes else 0)

        # Load the deployment's SPS economics before graph capture. Exact
        # schema-v2 tables define the positive V keys that must be captured for
        # each G, and are accepted only after matching a separate live-runtime
        # fingerprint. The worker receives this already-validated object when
        # confidence scheduling starts.
        self._dspark_sps_cost_table = None
        self._dspark_exact_candidate_cells = ()
        self._dspark_exact_identity_words = (0, ) * 8
        if (self.spec_config is not None and getattr(
                self.spec_config, "enable_confidence_scheduling", False) and
                getattr(self.spec_config, "confidence_sps_table_path", None)):
            from ..speculative.dspark_planner import load_runtime_sps_cost_table
            top_verify_len = int(self.spec_config.max_draft_len)
            self._dspark_sps_cost_table, _ = load_runtime_sps_cost_table(
                self.spec_config.confidence_sps_table_path,
                graph_batch_sizes=self._cuda_graph_batch_sizes,
                max_draft_len=top_verify_len,
                live_engine_fingerprint_path=getattr(
                    self.spec_config, "confidence_sps_live_fingerprint_path",
                    None),
            )
            self._dspark_exact_candidate_cells = (
                self._dspark_sps_cost_table.candidate_cells())
            self._dspark_exact_identity_words = (
                self._dspark_sps_cost_table.collective_identity_words)

        self._encoder_cuda_graph_padding_enabled = (
            encoder_cuda_graph_padding_enabled)

        # A feature-driven encoder (Whisper) declares a fixed-shape per-request
        # contract instead of packed tokens: every request costs exactly
        # `fixed_seq_len` of the encoder token budget, and the num_tokens /
        # seq_lens buckets are derived from the model rather than configured.
        # The model selects the mode; `encoder_cuda_graph_config` only opts in.
        (self._encoder_feature_shape, self._encoder_feature_dtype,
         self._encoder_fixed_seq_len) = self._encoder_graph_spec()

        self._encoder_cuda_graph_batch_sizes = (_filter_cuda_graph_batch_sizes(
            encoder_cuda_graph_batch_sizes, self.encoder_batch_size,
            self.encoder_max_num_tokens, self._encoder_fixed_seq_len or 1,
            self._encoder_cuda_graph_padding_enabled) if
                                                encoder_cuda_graph_batch_sizes
                                                else [])

        # Encoder CUDA graph bucket lists
        self._cuda_graph_num_tokens = (_filter_cuda_graph_num_tokens(
            encoder_cuda_graph_num_tokens, self.encoder_max_num_tokens,
            self._encoder_cuda_graph_padding_enabled)
                                       if encoder_cuda_graph_num_tokens else [])

        self._max_cuda_graph_num_tokens = (self._cuda_graph_num_tokens[-1] if
                                           self._cuda_graph_num_tokens else 0)
        self._cuda_graph_seq_lens = (_filter_cuda_graph_seq_lens(
            encoder_cuda_graph_seq_lens, self.max_seq_len,
            self._encoder_cuda_graph_padding_enabled)
                                     if encoder_cuda_graph_seq_lens else [])

        # Resolve which capture mode has usable shapes. In feature mode the
        # batch sizes *are* the whole key space, so an empty list after budget
        # filtering leaves nothing to capture. A model that declares a feature
        # contract cannot consume the packed token inputs the token-shaped
        # capture path synthesizes, so when feature mode is unavailable for it
        # the encoder stays eager instead of falling through to token capture.
        if self._encoder_feature_shape is not None:
            encoder_graph_shapes_available = bool(
                self._encoder_cuda_graph_batch_sizes)
            if not encoder_graph_shapes_available:
                logger.warning(
                    "Feature-mode encoder CUDA graphs: no configured batch "
                    "size fits within encoder max_num_tokens "
                    f"({self.encoder_max_num_tokens}) // encoder output "
                    f"length ({self._encoder_fixed_seq_len}); the encoder "
                    "step stays eager.")
                self._encoder_feature_shape = None
                self._encoder_feature_dtype = None
                self._encoder_fixed_seq_len = None
        elif self._model_encoder_graph_spec() is not None:
            encoder_graph_shapes_available = False
            if self.encoder_cuda_graph_config is not None:
                logger.warning(
                    "This model's encoder consumes fixed-shape features and "
                    "feature-mode encoder CUDA graphs are unavailable; the "
                    "encoder step stays eager.")
        else:
            encoder_graph_shapes_available = (bool(self._cuda_graph_num_tokens)
                                              and bool(
                                                  self._cuda_graph_seq_lens))

        use_encoder_cuda_graph = ((self._is_encoder_decoder_model()
                                   or self._is_encode_only)
                                  and self.encoder_cuda_graph_config is not None
                                  and encoder_graph_shapes_available)

        self.torch_compile_config = self.llm_args.torch_compile_config
        self.prefill_cuda_graph_backend = self.llm_args.prefill_cuda_graph_backend
        torch_compile_enabled = bool(self.torch_compile_config is not None)
        torch_compile_fullgraph = self.torch_compile_config.enable_fullgraph if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_fullgraph'].default
        torch_compile_inductor_enabled = self.torch_compile_config.enable_inductor if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_inductor'].default
        torch_compile_piecewise_cuda_graph = (self.prefill_cuda_graph_backend ==
                                              PrefillCudaGraphBackend.PIECEWISE)
        torch_compile_enable_userbuffers = self.torch_compile_config.enable_userbuffers if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_userbuffers'].default
        torch_compile_max_num_streams = self.torch_compile_config.max_num_streams if self.torch_compile_config is not None else TorchCompileConfig.model_fields[
            'max_num_streams'].default

        self._torch_compile_enabled = torch_compile_enabled
        self._torch_compile_piecewise_cuda_graph = torch_compile_piecewise_cuda_graph

        prefill_cuda_graph_num_tokens = self.llm_args.prefill_capture_num_tokens
        if prefill_cuda_graph_num_tokens is None:
            prefill_cuda_graph_num_tokens = cuda_graph_batch_sizes or []

        self._prefill_cuda_graph_num_tokens, unrecordable = (
            _filter_prefill_capture_num_tokens(
                prefill_cuda_graph_num_tokens,
                max_num_tokens=self.max_num_tokens,
                max_batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
            ))
        if unrecordable:
            logger.warning(
                f"Skipping prefill CUDA graph capture for num_tokens="
                f"{unrecordable}: exceeds reachable ceiling "
                f"max_batch_size*(max_seq_len-1)="
                f"{max(0, self.batch_size * (self.max_seq_len - 1))}. "
                f"Clamping them to the ceiling; raise max_seq_len for larger graphs."
            )

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
                    capture_num_tokens=self._prefill_cuda_graph_num_tokens,
                    max_num_streams=torch_compile_max_num_streams,
                    mapping=self.mapping)
                apply_llm_torch_compile = getattr(self.model,
                                                  "apply_llm_torch_compile",
                                                  None)
                if isinstance(self.model, DecoderModelForCausalLM):
                    self.model.model = torch.compile(
                        self.model.model,
                        backend=self._torch_compile_backend,
                        fullgraph=torch_compile_fullgraph)
                elif callable(apply_llm_torch_compile):
                    # TODO: Move this contract to MultimodalModelMixin once
                    # multimodal models consistently expose their LLM compile
                    # scope through the mixin.
                    apply_llm_torch_compile(backend=self._torch_compile_backend,
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
        # Per-request verify windows seen by the last full _prepare_tp_inputs
        # pass; all None without ragged verification.
        self.previous_verify_lens = []
        self.has_previous_device_draft = False

        self._encoder_decoder_host_buffer_pool: List[Dict[str, Any]] = []
        self._encoder_decoder_input_fast_path_static_eligible: Optional[
            bool] = None
        self._encoder_decoder_position_id_offset: Optional[int] = None

        sparse_params = (self.sparse_attention_config.to_sparse_params(
            pretrained_config=self.model.model_config.pretrained_config)
                         if self.sparse_attention_config is not None else None)
        self.attn_backend = get_attention_backend(self.llm_args.attn_backend,
                                                  sparse_params=sparse_params)

        self.get_runtime_tokens_per_gen_step = spec_config.get_runtime_tokens_per_gen_step if spec_config is not None else lambda runtime_draft_len: 1

        self.spec_metadata = None
        if self.is_spec_decode:
            if not self.is_draft_model:
                update_spec_config_from_loaded_model(self.spec_config,
                                                     self.model)
            max_num_draft_tokens = self.max_draft_loop_tokens * self.batch_size
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
            )
            self.max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
            self.max_draft_len = spec_config.max_draft_len
            # Mutable per-iteration draft length (updated each iteration when
            # dynamic draft length is enabled; otherwise stays fixed).  Tree
            # modes verify all tree nodes per step, which can be wider than the
            # tree depth used by the drafter loop.
            self.runtime_draft_len = (self.max_total_draft_tokens
                                      if not spec_config.is_linear_tree else
                                      self.max_draft_len)

        else:
            self.without_logits = False
            self.max_draft_len = 0
            self.runtime_draft_len = 0
            self.max_total_draft_tokens = 0

        self.guided_decoder: Optional[CapturableGuidedDecoder] = None

        # This field is initialized lazily on the first forward pass.
        # This is convenient because:
        # 1) The attention metadata depends on the KV cache manager.
        # 2) The KV cache manager depends on the model configuration.
        # 3) The model configuration is not loaded until the model engine
        # is initialized.
        #
        # NOTE: This can be simplified by decoupling the model config loading and
        # the model engine.
        self.attn_metadata = None
        self.encoder_attn_metadata = None
        self.spec_metadata = None
        self.iter_states = {}
        # Let the first CUDA graph capture create its private pool. Piecewise
        # CUDA graphs use a separate pool owned by their runners, so sharing a
        # pre-created pool handle with the outer graph runner is unnecessary.
        self._cuda_graph_mem_pool = None

        self._dynamic_draft_len_mapping = self._compute_dynamic_draft_len_mapping(
        )

        self.previous_batch_indices_cuda = torch.empty((self.max_num_tokens, ),
                                                       dtype=torch.int,
                                                       device='cuda')
        self._encoder_decoder_staged_request_ids: Optional[List[int]] = None
        # Ragged verification windows, read *inside* the captured graph: they
        # must live at a stable address and be written in place.
        self.ragged_verify_lens_cuda = torch.empty((self.batch_size + 1, ),
                                                   dtype=torch.int,
                                                   device='cuda')
        self.ragged_qo_indptr_cuda = torch.empty((self.batch_size + 2, ),
                                                 dtype=torch.int,
                                                 device='cuda')
        self.input_ids_cuda = torch.empty((self.max_num_tokens, ),
                                          dtype=torch.int,
                                          device='cuda')
        self.position_ids_cuda = torch.empty((self.max_num_tokens, ),
                                             dtype=torch.int,
                                             device='cuda')
        # Steady-state generation-only prepare cache (non-speculative overlap
        # decode). Holds the per-request lists that are invariant while the
        # scheduled generation batch keeps the same composition, plus a pinned
        # cached-token counter advanced by one per step (host-side bookkeeping
        # only; the device position buffer is advanced in place and this
        # buffer is never the source of an async H2D). Invalidated (set to
        # None) by every full _prepare_tp_inputs pass.
        self._steady_gen_cache: Optional[Dict[str, Any]] = None
        self._steady_gen_positions_pinned = torch.empty(
            (self.max_num_tokens, ),
            dtype=torch.int,
            pin_memory=prefer_pinned())
        if self.use_mrope:
            self.mrope_position_ids_cuda = torch.empty(
                (3, 1, self.max_num_tokens), dtype=torch.int, device='cuda')

        # Pre-allocated buffers for draft model to avoid implicit synchronization
        # These are used to build index tensors without creating tensors from Python lists
        max_first_draft_tokens = self.batch_size * (
            self.original_max_total_draft_tokens +
            1) if spec_config else self.batch_size
        tokens_per_draft = self.original_max_total_draft_tokens + 1
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
        self._trtllm_gen_jit_warmup = False

        # Create the encoder runner first. For encoder-decoder models it derives
        # every reachable startup capture key through get_graph_key().
        encoder_graph_batch_sizes = self._encoder_cuda_graph_batch_sizes
        encoder_graph_max_batch_size = (encoder_graph_batch_sizes[-1]
                                        if encoder_graph_batch_sizes else 0)
        # Feature mode's graph shapes follow from the batch sizes alone, so its
        # token budget is one fixed-length encoder output per request; the
        # token path uses the configured num_tokens buckets.
        feature_shape = self._encoder_feature_shape
        feature_dtype = self._encoder_feature_dtype
        fixed_seq_len = self._encoder_fixed_seq_len
        encoder_graph_max_num_tokens = (encoder_graph_max_batch_size *
                                        fixed_seq_len
                                        if feature_shape is not None else
                                        self._max_cuda_graph_num_tokens)

        encoder_cuda_graph_runner_config = EncoderCUDAGraphRunnerConfig(
            use_cuda_graph=use_encoder_cuda_graph,
            cuda_graph_padding_enabled=(
                self._encoder_cuda_graph_padding_enabled),
            cuda_graph_batch_sizes=encoder_graph_batch_sizes,
            cuda_graph_num_tokens=self._cuda_graph_num_tokens,
            cuda_graph_seq_lens=self._cuda_graph_seq_lens,
            max_cuda_graph_batch_size=encoder_graph_max_batch_size,
            max_cuda_graph_num_tokens=encoder_graph_max_num_tokens,
            max_num_tokens=self.encoder_max_num_tokens,
            max_seq_len=self.max_seq_len,
            # The encoder runner must never be handed the decoder's pool:
            # encoder replay runs on `encoder_stream`, device-concurrent with
            # decoder replay, and torch's pool-sharing contract assumes
            # replays from a shared pool are not concurrent.
            #
            # Literal None rather than `self._cuda_graph_mem_pool` states that
            # as a requirement instead of leaving it to a coincidence. The
            # engine attribute is None for the engine's whole life (nothing
            # assigns it after its declaration), so each runner already
            # created its own pool at its first capture and this allocates
            # nothing new — but reading it here would silently start sharing
            # the day someone gives that attribute a value.
            cuda_graph_mem_pool=None,
            is_encoder_decoder=self._is_encoder_decoder_model(),
            use_fixed_sequence_slots=(self._is_encoder_decoder_model()
                                      and hasattr(
                                          pretrained_config,
                                          "relative_attention_num_buckets")),
            feature_shape=feature_shape,
            feature_dtype=feature_dtype,
            fixed_seq_len=fixed_seq_len,
        )
        self.encoder_cuda_graph_runner = EncoderCUDAGraphRunner(
            encoder_cuda_graph_runner_config)
        if feature_shape is not None:
            logger.info(
                f"Feature-mode encoder CUDA graphs enabled for batch sizes "
                f"{encoder_graph_batch_sizes} (fixed_seq_len={fixed_seq_len}, "
                f"feature_shape={tuple(feature_shape)}).")

        # Once encoder CUDA graphs are usable, enable mixed decoder graphs by
        # default unless the user explicitly opts out.
        encoder_decoder_cuda_graph_enabled = (
            self.encoder_cuda_graph_runner.enabled
            and self.encoder_cuda_graph_runner.is_encoder_decoder
            and bool(self.encoder_cuda_graph_runner.capture_keys))
        enable_encoder_decoder_mixed_cuda_graph = (
            encoder_decoder_cuda_graph_enabled
            and self.cuda_graph_config is not None
            and self.llm_args.enable_encoder_decoder_mixed_cuda_graph)

        # Create decoder CUDA graph config and runner.
        cuda_graph_runner_config = CUDAGraphRunnerConfig(
            use_cuda_graph=(not self._is_encode_only
                            and self.cuda_graph_config is not None),
            cuda_graph_padding_enabled=self._cuda_graph_padding_enabled,
            cuda_graph_batch_sizes=self._cuda_graph_batch_sizes,
            max_cuda_graph_batch_size=self._max_cuda_graph_batch_size,
            max_beam_width=self.max_beam_width,
            spec_config=self.spec_config,
            cuda_graph_mem_pool=self._cuda_graph_mem_pool,
            dynamic_draft_len_mapping=self._dynamic_draft_len_mapping,
            max_num_tokens=self.max_num_tokens,
            use_mrope=self.use_mrope,
            original_max_draft_len=self.original_max_draft_len,
            original_max_total_draft_tokens=self.
            original_max_total_draft_tokens,
            is_draft_model=self.is_draft_model,
            enable_attention_dp=self.enable_attention_dp,
            is_encoder_decoder=self._is_encoder_decoder_model(),
            batch_size=self.batch_size,
            mapping=self.mapping,
            dist=self.dist,
            kv_cache_manager_key=self.kv_cache_manager_key,
            sparse_attention_config=self.sparse_attention_config,
            enable_encoder_decoder_mixed_cuda_graph=(
                enable_encoder_decoder_mixed_cuda_graph),
        )
        self.cuda_graph_runner = CUDAGraphRunner(cuda_graph_runner_config)
        self.breakable_cuda_graph_runner = None
        if self.prefill_cuda_graph_backend == PrefillCudaGraphBackend.BREAKABLE:
            decoder_model = (self.model if isinstance(
                self.model, DecoderModelForCausalLM) else getattr(
                    self.model, "llm", None))
            if not isinstance(decoder_model, DecoderModelForCausalLM):
                raise ValueError(
                    "breakable prefill CUDA graph requires a decoder model body"
                )
            self.breakable_cuda_graph_runner = BreakableCUDAGraphRunner(
                decoder_model.model)

        # Pinned staging buffers for async H2D copies on the ragged path;
        # see _pinned_host for the two-slot WAR-guard rationale.
        self._pinned_host_cache = {}
        self._pinned_host_events = {}
        self._pinned_host_active = {}

        # Initialize CUDA Graph LoRA manager if LoRA is enabled
        self.cuda_graph_lora_manager: Optional[CudaGraphLoraManager] = None
        self._force_lora_graph_for_capture: Optional[bool] = None

        # Setup the local cache indirection buffer only once and reuse it.
        # This way it can also be used for CUDA graphs.
        if self.use_beam_search:
            self.cache_indirection_attention = torch.zeros(
                (self.batch_size, self.max_beam_width, self.max_seq_len),
                device="cuda",
                dtype=torch.int32)
        else:
            self.cache_indirection_attention = None

        self.kv_cache_dtype_byte_size = self.get_kv_cache_dtype_byte_size()

        self._prepare_inputs_event: Optional[torch.cuda.Event] = None

        # Cache for enc-dec cross-attention stable generation steps.
        # Populated on the first CUDA-graph generation step; cleared whenever
        # the batch composition changes (new encoder request arrives).
        self._cross_attn_stable_cached_tokens: Optional[List[int]] = None
        self._cross_attn_stable_request_ids: Optional[List[int]] = None

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

    def _init_cuda_graph_lora_manager(self, lora_config: LoraConfig):
        """Initialize CUDA Graph LoRA manager with model configuration."""
        # Get model configuration
        if self.cuda_graph_runner.enabled:
            max_lora_size = lora_config.max_loras or 8  # Default fallback
            max_batch_size = self.batch_size  # Use engine's max batch size

            # For spec decode, each generation request contributes
            # max_draft_len + 1 tokens per forward pass.
            max_tokens_per_seq = (self.original_max_draft_len +
                                  1) if self.is_spec_decode else 1
            self.cuda_graph_lora_manager = CudaGraphLoraManager(
                max_lora_size=max_lora_size,
                max_batch_size=max_batch_size,
                max_lora_rank=lora_config.max_lora_rank,
                model=self.model,
                lora_model_config=self.lora_model_config,
                overlap_lora_and_base=lora_config.overlap_lora_and_base,
                device='cuda',
                max_tokens_per_seq=max_tokens_per_seq)

            logger.info(
                f"Initialized CUDA Graph LoRA manager, "
                f"max {max_lora_size} adapters, max rank {lora_config.max_lora_rank}"
            )

    def _use_lora_cuda_graph(self,
                             scheduled_requests: ScheduledRequests) -> bool:
        """
        Determines whether a non-LoRA or LoRA CUDA graph should be used, if
        both are available (cuda_graph_specialize_lora==True).
        """
        if self.cuda_graph_lora_manager is None:
            return False
        # Needed during graph capture to enforce a given mode
        if self._force_lora_graph_for_capture is not None:
            return self._force_lora_graph_for_capture
        if not self.llm_args.lora_config.cuda_graph_specialize_lora:
            return True
        return any(request.lora_task_id is not None
                   for request in scheduled_requests.generation_requests)

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

    @functools.cached_property
    def _mm_encoder_cache_enabled(self) -> bool:
        """Whether the multimodal encoder cache is active for this model."""
        return mm_encoder_cache_enabled(self.model)

    @property
    def is_warmup(self):
        return getattr(self, "_is_warmup", False)

    @is_warmup.setter
    def is_warmup(self, value: bool):
        self._is_warmup = value

        # This setter is the one choke point every warmup transition passes
        # through, including PyExecutor's, so select the MoE all-to-all budget
        # here rather than in set_warmup_flag().
        _set_moe_a2a_warmup(value)

        self.moe_load_balancer_iter_info = (not value, not value)

    @property
    def moe_load_balancer_iter_info(self):
        moe_load_balancer = self.moe_load_balancer
        if moe_load_balancer is not None:
            return moe_load_balancer.enable_statistic, moe_load_balancer.enable_update_weights
        return False, False

    @moe_load_balancer_iter_info.setter
    def moe_load_balancer_iter_info(self, value: Tuple[bool, bool]):
        moe_load_balancer = self.moe_load_balancer
        if moe_load_balancer is not None:
            moe_load_balancer.set_iter_info(enable_statistic=value[0],
                                            enable_update_weights=value[1])

    @property
    def use_beam_search(self):
        return self.max_beam_width > 1

    def _get_draft_kv_cache_manager(
        self, resource_manager: ResourceManager
    ) -> Optional[Union[KVCacheManager, KVCacheManagerV2]]:
        """
        Returns the draft KV cache manager only in one-model speculative decoding
        mode where the target model manages a separate draft KV cache.
        """
        return get_draft_kv_cache_manager(self.spec_config, resource_manager)

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

    def _pad_batch_seed_mrope_delta_cache(
            self, padded_requests: ScheduledRequests) -> None:
        if not self.use_mrope or padded_requests.num_generation_requests == 0:
            return

        mrope_position_deltas_cache = getattr(self.model,
                                              "mrope_position_deltas_cache",
                                              None)
        if mrope_position_deltas_cache is None:
            mrope_position_deltas_cache = getattr(
                getattr(self.model, "draft_model", None),
                "mrope_position_deltas_cache", None)
        if mrope_position_deltas_cache is None:
            return

        mrope_seed_seq_slots = []
        mrope_seed_deltas = []
        mrope_seed_requests = []
        for request in padded_requests.generation_requests:
            if (request.py_seq_slot is None or request.is_dummy
                    or getattr(request, "py_mrope_delta_cache_slot",
                               None) == request.py_seq_slot):
                continue
            mrope_position_delta = getattr(request, "py_mrope_position_delta",
                                           None)
            if mrope_position_delta is None and request.py_multimodal_data:
                mrope_config = request.py_multimodal_data.get('mrope_config')
                if mrope_config is not None:
                    mrope_position_delta = mrope_config.get(
                        'mrope_position_deltas')
            if mrope_position_delta is None:
                continue
            if mrope_position_delta.device.type == "cpu":
                mrope_position_delta = maybe_pin_memory(
                    mrope_position_delta).to(device='cuda',
                                             dtype=torch.int32,
                                             non_blocking=True)
            elif mrope_position_delta.dtype != torch.int32:
                mrope_position_delta = mrope_position_delta.to(
                    dtype=torch.int32)
            request.py_mrope_position_delta = mrope_position_delta
            mrope_seed_seq_slots.append(request.py_seq_slot)
            mrope_seed_deltas.append(mrope_position_delta.reshape(1))
            mrope_seed_requests.append(request)

        if not mrope_seed_seq_slots:
            return

        mrope_seed_seq_slots_tensor = torch.tensor(
            mrope_seed_seq_slots, dtype=torch.long,
            pin_memory=prefer_pinned()).to(device='cuda', non_blocking=True)
        mrope_seed_deltas_tensor = torch.cat(mrope_seed_deltas, dim=0)
        mrope_position_deltas_cache.index_copy_(
            0, mrope_seed_seq_slots_tensor,
            mrope_seed_deltas_tensor.to(
                dtype=mrope_position_deltas_cache.dtype))
        for request in mrope_seed_requests:
            request.py_mrope_delta_cache_slot = request.py_seq_slot

    @staticmethod
    def warmup_with_kv_cache_cleanup(method):
        """
        Decorator for warmup methods that cleans up NaNs/Infs in KV Cache after warmup execution.

        Why this is needed:
        - Our attention kernel uses multiplication by zero to mask out invalid tokens within
          the same page. Since NaN/Inf * 0 = NaN, any NaNs/Infs in these invalid KV areas
          will persist after masking.
        - These NaNs/Infs propagate to outputs and subsequent KV Cache entries, corrupting
          future computations with higher probability.
        - During warmup, we execute with placeholder data rather than actual valid inputs,
          which can introduce NaNs/Infs into KV Cache pages and cause random, hard-to-debug
          accuracy issues.
        """

        @functools.wraps(method)
        def wrapper(self, resource_manager: ResourceManager, *args, **kwargs):
            result = method(self, resource_manager, *args, **kwargs)
            kv_cache_manager = resource_manager.get_resource_manager(
                self.kv_cache_manager_key)
            if kv_cache_manager is not None:
                has_invalid_values = kv_cache_manager.check_invalid_values_in_kv_cache(
                    fill_with_zero=True)
                if has_invalid_values:
                    logger.warning(
                        "NaNs/Infs have been introduced to KVCache during warmup, KVCache was filled with zeros to avoid potential issues"
                    )
            return result

        return wrapper

    def _get_max_shape_warmup_requests(
            self, resource_manager: ResourceManager) -> List[Tuple[int, int]]:
        """
        Returns warmup configs covering the maximum context and generation shapes.
        """

        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        token_num_upper_bound = min(self.max_num_tokens,
                                    self.batch_size * (self.max_seq_len - 1))
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self.original_max_draft_len)
        max_batch_size = min(
            self.batch_size, curr_max_num_tokens //
            (1 + self.max_draft_loop_tokens) // self.max_beam_width)

        warmup_requests_configs = [
            (curr_max_num_tokens, 0),  # max_num_tokens, pure context
            (max_batch_size, max_batch_size),  # max_batch_size, pure generation
        ]

        return warmup_requests_configs

    def _get_full_general_warmup_requests(
            self, resource_manager: ResourceManager) -> List[Tuple[int, int]]:
        """
        Returns the ordered warmup configs for torch.compile specialization.

        Covers 1-token (0-1 graph specialization), max-shape (best triton autotuning),
        and small-context (2-token path) cases.
        """
        max_configs = self._get_max_shape_warmup_requests(resource_manager)
        # Specialize for 1 token pure ctx and pure gen
        one_token_configs = [(1, 0), (1, 1)]
        # Small ctx specialization
        small_ctx_configs = [(2, 0)]

        # Ordering matters for torch.compile graph specialization:
        # 1-token first to capture the 0→1 transition graph; max-shape next to seed
        # triton autotuning with the largest inputs; 2-token last for the small-ctx path.
        warmup_configs = one_token_configs + max_configs + small_ctx_configs
        # Deduplicate the warmup_configs while keeping the order.
        return list(dict.fromkeys(warmup_configs))

    @contextmanager
    def maybe_autotune_lora(self):
        """Enable autotuning while warming up CUDA-graph LoRA kernels."""
        if not (self.llm_args.enable_autotuner
                and self.cuda_graph_lora_manager is not None):
            yield
            return

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        with autotune(cache_path=cache_path):
            try:
                yield
            finally:
                # Complete the PP cache hand-off even on ranks without a
                # CUDA-graph-only tunable op.
                autotuner = AutoTuner.get()
                autotuner.cache_pp_recv()
                autotuner.cache_pp_send()
                autotuner.clean_pp_flag()

    @with_warmup_flag
    @warmup_with_kv_cache_cleanup
    def warmup(self, resource_manager: ResourceManager) -> None:
        """
        Orchestrates the warmup process by calling specialized warmup methods for
        torch.compile, the autotuner, and CUDA graphs.
        """
        # Ahead of the early returns below, since it holds regardless of why
        # warmup is skipped: only the advanced-sampling CUDA graph capture pass
        # exercises the non-greedy sampler, so with cuda_graph_config=None
        # flashinfer's sampling kernels would be JIT-built mid-serving.
        warmup_sampling_module()

        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)

        if kv_cache_manager is None:
            logger.info("Skipping warm up as no KV Cache manager allocated.")
            return

        # The lifetime of model engine and kv cache manager can be different.
        # Reset the global cuda graph dummy requests in warmup.
        self.cuda_graph_runner.padding_dummy_requests = {}
        self.cuda_graph_runner.secondary_padding_dummy_requests = {}
        self.cuda_graph_runner.ragged_zero_real_high_rows = 0

        is_enc_dec = self._is_encoder_decoder_model()
        if self.mapping.cp_size > 1:
            cp_type = self.mapping.cp_config.get("cp_type", None)
            if cp_type != CpType.HELIX:
                logger.info(
                    f"[ModelEngine::warmup] Skipping warmup for cp_type: {None if cp_type is None else cp_type.name}."
                )
                return

        # Create AutoTuner singleton in eager context before any compiled forward.
        # Otherwise the first get() can happen inside torch.compile tracing and
        # trigger non-traceable code (time.time(), torch.cuda.*) in the cache.
        AutoTuner.get()

        # ``guided_decoder`` is installed only on the last pipeline rank, so
        # this predicate is not rank-uniform on its own. Agree it before it
        # gates either the attention or the general phase.
        can_run_general_warmup = self._agree_warmup_flag(
            not is_enc_dec and not self.is_draft_model
            and not self.mapping.has_cp_helix() and self.guided_decoder is None
            and not isinstance(kv_cache_manager, MambaHybridCacheManager))

        log_mem_snapshot("warmup/before_warmup")
        # Compile the DSv4 indexer-Q CuTe DSL kernels before the first
        # collective-bearing forward, so their JIT cost is not charged against the
        # MoE all-to-all completion-flag deadline.
        self._prewarm_cute_dsl_indexer_q()
        log_mem_snapshot("warmup/after_cute_dsl_indexer_q")
        if not is_enc_dec:
            self._run_attention_warmup(resource_manager, can_run_general_warmup)

        if can_run_general_warmup:
            # Specialize torch.compile graphs across the key input shapes before CUDA graph capture.
            warmup_requests_configs = self._agree_warmup_shapes(
                self._get_full_general_warmup_requests(resource_manager))
            # Currently graph has not been captured, disable cuda graph for this warmup.
            with self.no_cuda_graph():
                self._general_warmup(resource_manager, warmup_requests_configs)
                # Release C++ MoE workspace buffers so the autotuner can
                # reclaim the memory.  They will be re-allocated on next use.
                from ..custom_ops.torch_custom_ops import MoERunner
                MoERunner.clear_all_workspaces()
                # Clear Cache now as autotuner may use additional memory.
                # Memory pool will be warmed up later.
                gc.collect()
                torch.cuda.empty_cache()

        # Helix CP is decode-only and runs into issues with the
        # autotuner warmup's context requests.
        if not is_enc_dec and not self.mapping.has_cp_helix():
            self._run_autotuner_warmup(resource_manager)
            log_mem_snapshot("warmup/after_autotuner")
            # Pre-JIT Mamba SSD multi-seq + HAS_INITSTATES=True Triton kernels
            # for Mamba hybrid models. Runs regardless of enable_autotuner,
            # since MambaHybridCacheManager skips _general_warmup and the
            # default autotuner shape is single-seq / no-initstates. Safe
            # no-op for non-Mamba models.
            self._run_mamba_hybrid_warmup(resource_manager)
            log_mem_snapshot("warmup/after_mamba_hybrid")
            # Release the autotuner's exploration-mode intermediates. The
            # exploration leftovers are pure waste that hide tens of GiB from
            # non-torch allocators (cuBLAS handle workspace, UCX/NIXL,
            # NVSHMEM).
            gc.collect()
            torch.cuda.empty_cache()
        # Warm up every graph shape before capturing any graph. Attention
        # kernels can switch implementations at smaller batch sizes and require
        # a larger workspace, so the first pass grows the workspace to its
        # maximum size. The second pass runs the final per-shape warmup and
        # captures without resizing the workspace.
        # Capture with the steady-state MoE all-to-all budget: the timeout is a
        # launch argument and is baked into every later replay.
        with _moe_a2a_steady_state_budget_for_capture():
            with self.cuda_graph_runner.allow_capture():
                self.cuda_graph_runner.is_warmup_only = True
                try:
                    with self.maybe_autotune_lora():
                        self._run_cuda_graph_warmup(resource_manager)
                finally:
                    self.cuda_graph_runner.is_warmup_only = False
                self.cuda_graph_runner.padding_dummy_requests = {}
                self.cuda_graph_runner.secondary_padding_dummy_requests = {}
                self.cuda_graph_runner.ragged_zero_real_high_rows = 0
                self._run_cuda_graph_warmup(resource_manager)
        log_mem_snapshot("warmup/after_cuda_graph_capture")
        self._warmup_dspark_ragged_compressor_metadata()
        log_mem_snapshot("warmup/after_dspark_ragged_compressor_metadata")
        # Pre-compile DeepGEMM paged_mqa_logits_metadata for every 32-aligned
        # batch bucket the runtime can produce (max_batch_size scaled by the
        # MTP / DSL expansion factor when applicable). CUDA-graph warmup only
        # exercises the batch sizes in cuda_graph_batch_sizes, which round
        # up to a subset of buckets; any inference iter whose
        # context_lens.size(0) lands on an uncovered bucket triggers an
        # nvcc-driven JIT compile (~3s stall inside _prepare_inputs) on
        # first touch. Pre-touching every bucket funnels that cost into
        # warmup. No-op on non-DSA models.
        self._warmup_dg_paged_mqa_logits_metadata()
        log_mem_snapshot("warmup/after_dg_paged_mqa_logits_metadata")
        self._warmup_cute_dsl_radix_topk()
        log_mem_snapshot("warmup/after_cute_dsl_radix_topk")
        if can_run_general_warmup:
            # Pre-populate the memory pool with max-shape allocations to reduce
            # fragmentation at runtime.
            warmup_requests_configs = self._get_max_shape_warmup_requests(
                resource_manager)
            self._general_warmup(resource_manager, warmup_requests_configs)
            log_mem_snapshot("warmup/after_memory_pool_prepop")

        # Allocate the CUDA graph padding dummies now, while the KV cache is
        # empty. Waiting for the first padded step can race KV saturation:
        # once the cache is full, the lazy allocation in _get_padded_batch
        # fails every step and padded batches silently run eager.
        self.cuda_graph_runner.preallocate_padding_dummies(resource_manager)
        log_mem_snapshot("warmup/after_preallocate_padding_dummies")

        # If this is a BOLT-instrumented build (the profile-gen job sets
        # TLLM_BOLT_CLEAR_COUNTERS=1), reset the instrumentation counters now
        # that all startup JIT/autotune/graph-capture is done, so the emitted
        # .fdata reflects steady-state serving only. No-op on normal builds.
        from ..bolt_profiling import maybe_bolt_clear_counters
        maybe_bolt_clear_counters()

    @torch.inference_mode()
    def _warmup_dspark_ragged_compressor_metadata(self) -> None:
        """Compile DSv4 ragged metadata for a non-CUDA-graph batch size.

        Whole-model CUDA-graph warmup exercises only configured graph batch
        sizes. The first mixed attention-DP iteration can contain an
        intermediate number of generation requests, which otherwise triggers
        a long ``torch.compile`` specialization immediately before a TP
        collective. Ranks without generation work reach the collective first
        and can trip the executor hang detector while their peers compile.

        One non-bucket probe forces the dynamic specialization during engine
        warmup, before the hang detector starts. It uses the metadata's real
        persistent buffers so the tensor shape and stride guards match serving.
        Those buffers are created under inference mode, so all probe and
        cleanup mutations must run under the same mode.
        """
        if (not self._dspark_confidence_enabled
                or not self._dspark_trims_submitted_tokens):
            return

        metadata = getattr(self, "attn_metadata", None)
        required_attributes = (
            "_compute_compressed_mask",
            "_compute_gen_compressed_position_ids",
            "_compress_ratios_sorted",
            "compressed_mask_cuda",
            "compressed_position_ids_cuda",
            "cu_new_comp_kv_cuda",
            "max_draft_tokens",
            "new_comp_kv_lens_cuda",
            "past_kv_lens_cuda",
        )
        missing_attributes = [
            name for name in required_attributes
            if metadata is None or not hasattr(metadata, name)
        ]
        if missing_attributes:
            raise RuntimeError(
                "DSpark ragged metadata warmup requires the DSv4 compressor "
                "contract; missing " + ", ".join(missing_attributes))

        graph_batch_sizes = {
            int(batch_size)
            for batch_size in self._cuda_graph_batch_sizes
        }
        num_generations = next(
            (candidate for candidate in range(int(self.batch_size) - 1, 0, -1)
             if candidate not in graph_batch_sizes), None)
        if num_generations is None:
            return

        compress_ratios = [
            int(ratio) for ratio in metadata._compress_ratios_sorted
        ]
        tokens_per_generation = 1 + int(metadata.max_draft_tokens)
        total_compressed_tokens = {}
        gen_output_offsets = {ratio: 0 for ratio in compress_ratios}
        prepared_ratios = []
        try:
            for ratio in compress_ratios:
                compressed_tokens_per_generation = (tokens_per_generation +
                                                    ratio - 1) // ratio
                total_tokens = (num_generations *
                                compressed_tokens_per_generation)
                past_kv_lens = metadata.past_kv_lens_cuda[ratio]
                cu_new_comp = metadata.cu_new_comp_kv_cuda[ratio]
                new_comp = metadata.new_comp_kv_lens_cuda[ratio]
                compressed_positions = (
                    metadata.compressed_position_ids_cuda[ratio])
                compressed_mask = metadata.compressed_mask_cuda[ratio]
                if (past_kv_lens.numel() < num_generations
                        or cu_new_comp.numel() < num_generations + 1
                        or new_comp.numel() < num_generations
                        or compressed_positions.numel() < total_tokens
                        or compressed_mask.numel() < total_tokens):
                    raise RuntimeError(
                        "DSpark ragged compressor metadata buffers cannot "
                        f"hold the compile probe with {num_generations} "
                        "generation requests and compression ratio "
                        f"{ratio}")

                total_compressed_tokens[ratio] = total_tokens
                prepared_ratios.append(ratio)
                past_kv_lens[:num_generations].zero_()
                new_comp[:num_generations].fill_(
                    compressed_tokens_per_generation)
                cu_new_comp[:num_generations + 1].copy_(
                    torch.arange(num_generations + 1,
                                 dtype=cu_new_comp.dtype,
                                 device=cu_new_comp.device))
                cu_new_comp[:num_generations +
                            1].mul_(compressed_tokens_per_generation)

            logger.info(
                "DSpark ragged metadata warmup: compiling one non-graph "
                f"shape with {num_generations} generation requests")
            metadata._compute_gen_compressed_position_ids(
                metadata.past_kv_lens_cuda,
                metadata.cu_new_comp_kv_cuda,
                metadata.compressed_position_ids_cuda,
                0,
                num_generations,
                tokens_per_generation,
                compress_ratios,
                gen_output_offsets,
            )
            metadata._compute_compressed_mask(
                metadata.new_comp_kv_lens_cuda,
                metadata.cu_new_comp_kv_cuda,
                metadata.compressed_mask_cuda,
                num_generations,
                total_compressed_tokens,
                compress_ratios,
            )
            torch.cuda.synchronize()
        finally:
            for ratio in prepared_ratios:
                total_tokens = total_compressed_tokens[ratio]
                metadata.past_kv_lens_cuda[ratio][:num_generations].zero_()
                metadata.new_comp_kv_lens_cuda[ratio][:num_generations].zero_()
                metadata.cu_new_comp_kv_cuda[ratio][:num_generations +
                                                    1].zero_()
                metadata.compressed_position_ids_cuda[
                    ratio][:total_tokens].zero_()
                metadata.compressed_mask_cuda[ratio][:total_tokens].zero_()
            torch.cuda.synchronize()
        logger.info("DSpark ragged metadata warmup complete")

    def _warmup_dg_paged_mqa_logits_metadata(self) -> None:
        """Pre-compile DeepGEMM's `get_paged_mqa_logits_metadata` helper for
        every 32-aligned batch bucket the runtime can produce.

        DSA's `Indexer.prepare_scheduler_metadata` calls
        `deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv,
        num_sms)` inside `_prepare_inputs` every iteration. The underlying
        kernel is templated on `<kAlignedBatchSize, split_kv, num_sms>`
        where `kAlignedBatchSize = align(context_lens.size(0), 32)` and
        `split_kv` / `num_sms` are fixed for a given (block_kv, device).
        deep_gemm's Python-side JIT compiles a fresh cubin (spawning
        nvcc/cicc/ptxas, ~3s on GB300) the first time each `aligned_bs`
        is requested. CUDA-graph warmup exercises only the batch sizes in
        `cuda_graph_batch_sizes`, which round up to a subset of the 32-
        aligned buckets; every uncovered bucket that the inference
        workload later touches produces a 3s stall on that iteration.
        Pre-touching every bucket here funnels those compiles into the
        deterministic warmup phase.

        `context_lens.size(0)` is not always `num_generations`. For MTP
        with `use_expanded_buffers_for_mtp=True` the expanded call passes
        `num_generations * (1 + max_draft_tokens)`. For DSL expansion the
        call passes `num_generations * dsl_expand_factor`, where
        `dsl_expand_factor = next_n // eff` (`eff in kernel_atoms`, see
        `_pick_dsl_expand` in `dsa.py`); its worst case is
        `next_n = 1 + max_draft_tokens` when `eff == 1`. Reading the
        current `dsl_expand_factor` off the metadata would under-estimate
        the eventual max (it defaults to 1 before any prepare() has run,
        and per-iter picks can differ across iters when CUDA graph is
        off), so we use the static upper bound `1 + max_draft_tokens`
        for both expansion paths. Bucket range is also scaled by
        `max_beam_width` as a defense-in-depth ceiling for future beam
        support (no-op today — DSA does not use beam). No-op on non-DSA
        models.

        Best-effort: per-bucket JIT failures are logged and skipped so a
        single broken bucket does not abort PyExecutor startup.
        """
        attn_meta = getattr(self, "attn_metadata", None)
        if attn_meta is None:
            return
        try:
            from tensorrt_llm._torch.attention_backend.sparse.dsa import (
                _DG_SCHEDULE_BLOCK_KV, DSAtrtllmAttentionMetadata)
        except ImportError:
            return
        if not isinstance(attn_meta, DSAtrtllmAttentionMetadata):
            return
        try:
            from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata
        except ImportError:
            logger.info(
                "[DG warmup] deep_gemm.get_paged_mqa_logits_metadata not "
                "available; skipping paged_mqa_logits_metadata prewarm.")
            return

        num_sms = attn_meta.num_sms
        max_bs = max(1, int(self.batch_size))
        beam_width = max(1, int(getattr(self, "max_beam_width", 1) or 1))
        # Static upper bound on the row-count multiplier applied to
        # `context_lens`. Both MTP-expanded and DSL-expanded call sites
        # are bounded above by `(1 + max_draft_tokens)`; see the
        # docstring for why we don't read the runtime `dsl_expand_factor`
        # here.
        max_draft_tokens = int(getattr(attn_meta, "max_draft_tokens", 0) or 0)
        expands_batch = (getattr(attn_meta, "use_expanded_buffers_for_mtp",
                                 False)
                         or getattr(attn_meta, "expand_for_dsl", False))
        expand_factor = 1 + max_draft_tokens if expands_batch else 1
        max_aligned = ((max_bs * beam_width * expand_factor + 31) // 32) * 32
        buckets = list(range(32, max_aligned + 32, 32))
        logger.info(f"[DG warmup] Pre-compiling paged_mqa_logits_metadata for "
                    f"{len(buckets)} aligned batch buckets up to {max_aligned} "
                    f"(block_kv={_DG_SCHEDULE_BLOCK_KV}, num_sms={num_sms}, "
                    f"max_bs={max_bs}, beam_width={beam_width}, "
                    f"expand_factor={expand_factor})")
        for aligned_bs in buckets:
            # Kernel scans `context_lens` and prefix-sums schedules; a
            # zero-filled 2D tensor of shape (aligned_bs, 1) is enough to
            # trigger dispatch and compile — the metadata output is
            # discarded.
            dummy = torch.zeros(aligned_bs, 1, dtype=torch.int32, device="cuda")
            try:
                _ = get_paged_mqa_logits_metadata(dummy, _DG_SCHEDULE_BLOCK_KV,
                                                  num_sms)
            except RuntimeError as e:
                # Narrow to RuntimeError so signature drifts in
                # get_paged_mqa_logits_metadata (TypeError / ValueError)
                # surface loudly instead of silently degrading perf.
                logger.warning(
                    f"[DG warmup] paged_mqa_logits_metadata prewarm failed "
                    f"for aligned_bs={aligned_bs} "
                    f"(block_kv={_DG_SCHEDULE_BLOCK_KV}, num_sms={num_sms}); "
                    f"skipping bucket. {type(e).__name__}: {e}")
        torch.cuda.synchronize()

    def _prewarm_cute_dsl_indexer_q(self) -> None:
        """Pre-compile the DSv4 indexer-Q CuTe DSL kernels, then barrier.

        Runs before any collective-bearing forward so this op's first-touch
        ``cute.compile`` is not charged against the MoE all-to-all
        completion-flag deadline. It is a partial mitigation only: other
        first-touch compiles remain inside collective-bearing forwards, and some
        sit on the all-to-all path itself and cannot be pre-compiled this way.
        The runtime budget (``moeA2AGetTimeoutCycles``) covers the general case.

        Only the fallback tactics are compiled -- what an eager, cache-miss
        forward selects. The runner's kernel cache key excludes m/n/k, so one
        compile per tactic covers every shape. Uses the real module and weights,
        so it cannot drift from what the model runs.

        No-op on non-DSA models. See nvbugs/6482566.
        """
        try:
            from ..attention_backend.sparse.deepseek_v4.indexer import \
                DeepseekV4Indexer
        except ImportError:
            return

        indexer = next(
            (m
             for m in self.model.modules() if isinstance(m, DeepseekV4Indexer)
             and getattr(m, "wq_b", None) is not None), None)
        if indexer is None:
            return

        weight = indexer.wq_b.weight
        # _fallback_tactic() branches on m at 4 and 8, so these three token
        # counts cover every fallback tactic it can return.
        with torch.inference_mode():
            for num_tokens in (4, 8, 16):
                try:
                    qr = torch.zeros((num_tokens, weight.shape[1]),
                                     dtype=torch.bfloat16,
                                     device=weight.device)
                    position_ids = torch.zeros((num_tokens, ),
                                               dtype=torch.int32,
                                               device=weight.device)
                    indexer._project_and_quantize_q(qr, position_ids)
                except Exception as e:
                    # Never fail startup for a prewarm miss; the kernel would
                    # simply be compiled later, as it is today.
                    logger.warning(
                        f"indexer-Q CuTe DSL prewarm skipped for {num_tokens} "
                        f"tokens. {type(e).__name__}: {e}")
        torch.cuda.synchronize()

        # Hold every rank here until the slowest has finished compiling, so the
        # first MoE all-to-all dispatch is entered without JIT skew.
        if self.mapping.tp_size > 1 and self.dist is not None:
            self.dist.tp_allgather(1)
        logger.info("indexer-Q CuTe DSL prewarm complete")

    def _warmup_cute_dsl_radix_topk(self) -> None:
        """Pre-compile the DSA radix-filter CuTe DSL decode top-k for every
        cluster_size band during warmup, before serving.

        Captured geometries are already compiled by the warmup-step forwards;
        this fills in the bands the eager (non-captured) decode path can still
        hit (mixed prefill+decode batch, or cuda_graph disabled) so they do
        not pay a first-touch JIT stall on a live request. DSA-specific params
        live on the metadata, so delegate to it. No-op on non-DSA models.
        """
        attn_meta = getattr(self, "attn_metadata", None)
        if attn_meta is None:
            return
        try:
            from ..attention_backend.sparse.dsa import \
                DSAtrtllmAttentionMetadata
        except ImportError:
            return
        if isinstance(attn_meta, DSAtrtllmAttentionMetadata):
            next_n = 1 + self.original_max_draft_len
            attn_meta.warmup_cute_dsl_radix_topk(next_n)
            if hasattr(attn_meta, "warmup_selfsampling_topk"):
                attn_meta.warmup_selfsampling_topk(
                    next_n, batch_sizes=self._cuda_graph_batch_sizes)

    def _general_warmup(self, resource_manager: ResourceManager,
                        warmup_requests_configs: List[Tuple[int, int]]):
        """
        Runs forward passes for each config in warmup_requests_configs.

        Serves both torch.compile graph specialization and memory pool pre-population.
        """
        # Disable CUDA graph replay during general warmup to avoid replaying
        # graphs with stale KV cache block offsets from capture time.
        with self.no_cuda_graph():
            self._general_warmup_impl(resource_manager, warmup_requests_configs)

    def _is_distributed_forward(self) -> bool:
        """Return whether model forward can communicate with peer workers.

        ``dist`` is optional. An engine built without a communicator cannot
        enter a collective at all, so it has no peers to strand and every
        warmup failure stays rank-local.
        """
        if self.dist is None:
            return False
        return self.dist.world_size > 1 or self.mapping.dwdp_enabled

    def _warmup_agreement_allgather(
            self) -> Optional[Callable[[int], List[int]]]:
        """Return an allgather over the ranks this warmup forward synchronizes with.

        ``None`` means agreement cannot be established here, so a missing batch
        stays fatal rather than being skipped unilaterally.

        DWDP is the case that cannot be answered: its peers are reached through
        a ``COMM_WORLD``-derived subgroup built in ``dwdp.py``, not through
        ``self.dist``, so a ``self.dist`` allgather would report a unanimity it
        never observed.
        """
        if self.dist is None or self.mapping.dwdp_enabled:
            return None
        if self.dist.world_size <= 1:
            return None
        return self.dist.allgather

    def _agree_warmup_flag(self, flag: bool) -> bool:
        """Reduce a phase-entry decision to one the whole forward group shares.

        Several predicates that gate a warmup phase are rank-local. The
        capturable guided decoder is installed only on the last pipeline rank,
        so ``guided_decoder is None`` -- and through it
        ``can_run_general_warmup`` -- differs across pipeline stages. Mamba's
        entry test reads this rank's free KV capacity. Letting one rank enter a
        phase its peers skip strands whoever enters the collective, and it also
        unbalances the per-shape agreement below.

        Any rank opting out takes the whole group out with it.
        """
        allgather = self._warmup_agreement_allgather()
        if allgather is None:
            return flag
        return all(allgather(int(flag)))

    def _agree_warmup_shapes(
            self, configs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Reduce warmup shapes to the ones every rank in the group proposed.

        ``_get_max_shape_warmup_requests`` derives shapes from this rank's free
        KV capacity, so under attention-DP the values -- and, once
        ``dict.fromkeys`` drops a collision, the length -- differ per rank.
        Ranks would then walk different loops and meet in different forwards.

        The intersection keeps rank 0's ordering, which matters because the
        general warmup list is ordered for torch.compile specialization.
        """
        allgather = self._warmup_agreement_allgather()
        if allgather is None:
            return configs
        per_rank = allgather([list(config) for config in configs])
        shared = set.intersection(*({tuple(config)
                                     for config in rank_configs}
                                    for rank_configs in per_rank))
        agreed = [config for config in configs if config in shared]
        dropped = [config for config in configs if config not in shared]
        if dropped:
            logger.warning(
                f"Dropping warmup shapes {dropped} that not every rank could "
                f"propose; per-rank KV capacity differs. Remaining: {agreed}.")
        return agreed

    def _should_run_warmup_batch(self, batch: Optional[ScheduledRequests],
                                 num_tokens: int, shape: str) -> bool:
        """Decide whether this warmup shape runs, is skipped, or fails the rank.

        A rank that skips a shape its peers run leaves them blocked in that
        forward's collectives for the rest of the job. Skipping is therefore
        safe exactly when every rank in the forward group skips too, and an
        allgather establishes that before any rank enters the forward.

        The plan agreed by ``_agree_warmup_plan`` is what makes that allgather
        safe: every rank walks the same shape list, so this runs the same
        number of times everywhere.
        """
        if not self._is_distributed_forward():
            if batch is not None:
                return True
            # Safe to skip, but never silently: a skip during KV cache
            # estimation makes the profiling peak unrepresentative of
            # this shape.
            logger.warning(f"Skipping warmup shape ({shape}): not enough KV "
                           f"cache space.")
            return False

        allgather = self._warmup_agreement_allgather()
        if allgather is None:
            # No reachable group to agree with. Keep the TP-only check, which
            # still catches the attention-DP asymmetry it was written for.
            self._assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
            if batch is None:
                raise RuntimeError(
                    f"Warmup batch creation failed for shape ({shape}) on "
                    f"global_rank={global_mpi_rank()}, "
                    f"model_rank={self.dist.rank}, and this topology offers no "
                    f"way to confirm that peers are skipping it too. They may "
                    f"already be inside the matching forward, so this rank "
                    f"cannot skip the shape without stranding them.")
            return True

        flags = list(allgather(int(batch is not None)))
        if all(flags):
            return True
        if not any(flags):
            # Every rank in the forward group is skipping, so none of them is
            # left inside a collective. This is the ordinary outcome for a
            # shape that does not fit the configuration at all, such as a
            # mixed context+generation shape under ``max_batch_size=1``.
            logger.warning(f"Skipping warmup shape ({shape}) on all "
                           f"{len(flags)} ranks: not enough KV cache space.")
            return False

        all_tokens = list(allgather(num_tokens))
        failed_ranks = [i for i, flag in enumerate(flags) if not flag]
        raise RuntimeError(
            f"Warmup batch creation failed for shape ({shape}) on rank(s) "
            f"{failed_ranks} but succeeded on others, so entering this forward "
            f"would deadlock the ranks that still hold a batch. Per-rank "
            f"curr_max_num_tokens: {all_tokens}. This indicates asymmetric KV "
            f"cache capacity across ranks. Consider increasing "
            f"--kv_cache_free_gpu_mem_fraction.")

    def _assert_all_tp_ranks_have_warmup_batch(self, batch,
                                               num_tokens: int) -> None:
        """Assert every TP rank has a valid warmup batch, or raise with diagnostics.

        Under attention-DP, each rank's KV cache available capacity can differ at
        runtime, causing _create_warmup_request to return None on some ranks while
        others proceed into forward() with tp_comm collectives — deadlocking the
        job. This check prevents the deadlock by failing early with diagnostic info.

        ``tp_size`` alone does not establish that peers are reachable: ``dist``
        is optional, and without a communicator there is no tp_comm collective
        to deadlock in.
        """
        if self.mapping.tp_size <= 1 or self.dist is None:
            return
        has_batch = int(batch is not None)
        all_flags = list(self.dist.tp_allgather(has_batch))
        if any(all_flags) and not all(all_flags):
            # Gather token counts for diagnostics
            all_tokens = list(self.dist.tp_allgather(num_tokens))
            failed_ranks = [i for i, f in enumerate(all_flags) if not f]
            raise RuntimeError(
                f"Warmup batch creation failed on TP rank(s) {failed_ranks} "
                f"but succeeded on others. This would cause a collective "
                f"deadlock. Per-rank curr_max_num_tokens: {all_tokens}. "
                f"This indicates asymmetric KV cache capacity across TP ranks. "
                f"Consider increasing --kv_cache_free_gpu_mem_fraction.")

    def _general_warmup_impl(
            self, resource_manager: ResourceManager,
            warmup_requests_configs: List[Tuple[int, int]]) -> None:

        for num_tokens, num_gen_tokens in warmup_requests_configs:
            # Helix CP does not support warmup with context requests.
            if self.mapping.has_cp_helix() and num_tokens != num_gen_tokens:
                continue
            try:
                with self._release_batch_context(
                        self._create_warmup_request(resource_manager,
                                                    num_tokens, num_gen_tokens),
                        resource_manager) as batch:
                    if not self._should_run_warmup_batch(
                            batch, num_tokens,
                            f"general, num_tokens={num_tokens}, "
                            f"num_gen_tokens={num_gen_tokens}"):
                        continue
                    logger.info(
                        f"Run warmup with {num_tokens} tokens, include {num_gen_tokens} generation tokens"
                    )
                    self.forward(batch,
                                 new_tensors_device=None,
                                 resource_manager=resource_manager)
                    torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                if self._is_distributed_forward():
                    # Peers are inside the same forward's collectives and
                    # cannot follow a rank-local skip.
                    raise
                logger.warning(
                    f"OOM during general warmup with {num_tokens} tokens, "
                    f"{num_gen_tokens} generation tokens. Skipping.")
                # If the OOM aborted the forward between dispatch() and
                # combine(), the MoE A2A state machines are stuck in
                # ``dispatched`` and the next warmup will hit
                # ``dispatch called twice``. Reset them before retrying a
                # smaller shape.
                self._reset_moe_alltoall_state()
                torch.cuda.empty_cache()

    def _reset_moe_alltoall_state(self) -> None:
        """Reset all MoE all-to-all state machines reachable from ``self.model``.

        Each MoE backend keeps a small dispatch/combine phase state per layer
        (``MoeAlltoAll`` or ``NVLinkOneSided``). A forward that calls
        ``dispatch`` but raises before reaching ``combine`` (e.g., a warmup
        OOM mid-MoE) leaves that state in ``dispatched``, which fails the
        invariant on the next ``dispatch`` call. This helper walks the model
        and resets any A2A state found, so subsequent forwards start clean.
        """
        for module in self.model.modules():
            for attr_name in ("moe_a2a", "comm"):
                obj = getattr(module, attr_name, None)
                reset = getattr(obj, "reset_state", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"Failed to reset MoE A2A state on {type(module).__name__}.{attr_name}: {e}"
                        )

    def _run_attention_warmup(self,
                              resource_manager: ResourceManager,
                              can_run_general_warmup: bool = True) -> None:
        if not issubclass(self.attn_backend.Metadata, TrtllmAttentionMetadata):
            return

        @contextlib.contextmanager
        def trtllm_gen_fmha_jit_warmup():
            previous = self._trtllm_gen_jit_warmup
            self._trtllm_gen_jit_warmup = True
            try:
                yield
            finally:
                self._trtllm_gen_jit_warmup = previous

        logger.info("Running TRTLLM-Gen FMHA JIT warmup")

        warmup_requests_configs = []
        if not self.is_draft_model and self.guided_decoder is None:
            # doesn't support 2-model speculative draft and guided decoding
            warmup_requests_configs.append(
                (1 + self.max_total_draft_tokens, 1))  # one generation request
        else:
            logger.debug("Skipped TRTLLM-Gen FMHA JIT warmup for Gen kernels")

        if can_run_general_warmup:
            warmup_requests_configs.append((1, 0))  # one context token
        else:
            logger.debug("Skipped TRTLLM-Gen FMHA JIT warmup for Ctx kernels")

        model_type = getattr(self.model.model_config.pretrained_config,
                             "model_type", None)
        if can_run_general_warmup and model_type in ("kimi_k3", "kimi_linear"):
            # Kimi's one-token context takes the NT < 4 FLA fallback and does
            # not compile the optimized single-sequence K123 variant. A
            # non-aligned five-chunk context enters the pure K123 path.
            _KIMI_KDA_PREFILL_WARMUP_TOKENS = 257
            logger.info("Adding Kimi KDA pure-prefill warmup with "
                        f"{_KIMI_KDA_PREFILL_WARMUP_TOKENS} context tokens")
            warmup_requests_configs.append((_KIMI_KDA_PREFILL_WARMUP_TOKENS, 0))

        if (not self.is_draft_model and self.guided_decoder is None
                and can_run_general_warmup):
            # The cute_dsl_mla FMHA lib now only support the generation-only batch, we need to warmup the TRTLLM-Gen FMHA lib for the mixed context+generation batch.
            # One MIXED context+generation batch (1 ctx token + 1 gen request).
            warmup_requests_configs.append(
                (1 + self.max_total_draft_tokens + 1, 1))
        else:
            logger.debug(
                "Skipped TRTLLM-Gen flashinfer_trtllm_gen FMHA lib JIT warmup When enable cute_dsl_mla FMHA lib"
            )

        for num_tokens, num_gen_requests in warmup_requests_configs:
            warmup_request = self._create_warmup_request(
                resource_manager,
                num_tokens=num_tokens,
                num_gen_requests=num_gen_requests)

            with self.no_cuda_graph(), self._release_batch_context(
                    warmup_request, resource_manager) as batch:
                if not self._should_run_warmup_batch(
                        batch, num_tokens,
                        f"attention, num_tokens={num_tokens}, "
                        f"num_gen_requests={num_gen_requests}"):
                    continue
                with trtllm_gen_fmha_jit_warmup():
                    self.forward(batch,
                                 new_tensors_device=None,
                                 resource_manager=resource_manager)
                torch.cuda.synchronize()

    @staticmethod
    def _release_megamoe_profiling_scratch():
        # MegaMoE tuning resources are shared across layers, so only the engine
        # can release them after its full autotune warmup and before graph
        # capture. Later eviction could invalidate a captured workspace pointer.
        from ..moe.custom_ops import cute_dsl_megamoe_custom_op as _megamoe_op
        release_megamoe_scratch = getattr(_megamoe_op,
                                          "release_megamoe_profiling_scratch",
                                          None)
        if release_megamoe_scratch is not None:
            release_megamoe_scratch()

    def _run_autotuner_warmup(self, resource_manager: ResourceManager) -> None:
        """Runs forward passes to populate the autotuner cache."""
        from ..custom_ops.torch_custom_ops import MXFP8GemmRunner
        from ..modules.linear import (MXFP8LinearMethod,
                                      flashinfer_mxfp8_autotune)

        enable_trtllm_autotuner = self.llm_args.enable_autotuner
        if not enable_trtllm_autotuner:
            return

        mxfp8_methods = []
        for module in self.model.modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, MXFP8LinearMethod):
                mxfp8_methods.append(quant_method)

        # This engine owns startup warmup, so it explicitly opts its MXFP8
        # methods into native tuning. Standalone modules and engine paths that
        # skip this warmup remain on the direct native op.
        for method in mxfp8_methods:
            method.enable_native_autotune()

        # Native and FlashInfer tuning are independent. Capture native
        # eligibility before enabling graph-only FlashInfer dispatch.
        native_mxfp8_methods = [
            method for method in mxfp8_methods if method.needs_native_autotune
        ]
        use_mxfp8_flashinfer_graph_default = (
            self.cuda_graph_runner.enabled
            and "TRTLLM_MXFP8_GEMM_BACKEND" not in os.environ and any(
                getattr(module, "_use_flashinfer_mxfp8_decode_graph_default",
                        False) for module in self.model.modules()))
        if use_mxfp8_flashinfer_graph_default:
            for quant_method in mxfp8_methods:
                quant_method.enable_flashinfer_auto()
        flashinfer_mxfp8_methods = [
            method for method in mxfp8_methods
            if method.needs_flashinfer_autotune
        ]

        # Every TP and PP rank must make the same backend decision before any
        # rank returns or enters a tuning forward with model collectives.
        if self.mapping.tp_size > 1 or self.mapping.has_pp():
            local_flashinfer_enabled = int(bool(flashinfer_mxfp8_methods))
            all_flashinfer_enabled = [local_flashinfer_enabled]
            if self.mapping.tp_size > 1:
                all_flashinfer_enabled = list(
                    self.dist.tp_allgather(local_flashinfer_enabled))
            if self.mapping.has_pp():
                all_flashinfer_enabled = [
                    enabled for stage_flags in self.dist.pp_allgather(
                        all_flashinfer_enabled) for enabled in stage_flags
                ]
            if any(all_flashinfer_enabled) and not all(all_flashinfer_enabled):
                forced_flashinfer = any(method.backend == "flashinfer"
                                        for method in mxfp8_methods)
                for method in mxfp8_methods:
                    method.disable_flashinfer_auto()
                flashinfer_mxfp8_methods = []
                if forced_flashinfer:
                    raise RuntimeError(
                        "FlashInfer MXFP8 was explicitly requested but is not "
                        "available on every TP/PP rank")
                logger.warning(
                    "FlashInfer MXFP8 availability differs across TP/PP ranks; "
                    "using the native TensorRT-LLM GEMM backend on every rank.")

        enable_flashinfer_mxfp8_autotuner = bool(flashinfer_mxfp8_methods)
        enable_native_mxfp8_autotuner = bool(native_mxfp8_methods)

        AutoTuner.get().setup_distributed_state(self.mapping, self.dist)
        logger.info(
            f"Running autotuner warmup (TRT-LLM={enable_trtllm_autotuner}, "
            f"native MXFP8={enable_native_mxfp8_autotuner}, "
            f"FlashInfer MXFP8={enable_flashinfer_mxfp8_autotuner})...")
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        token_num_upper_bound = min(self.max_num_tokens,
                                    self.batch_size * (self.max_seq_len - 1))
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self.original_max_draft_len)

        warmup_configs = [(curr_max_num_tokens, 0)]
        if (not self.is_draft_model and self.guided_decoder is None
                and not self.mapping.has_pp()):
            # Add generation request to warmup the autotuner cache.
            warmup_configs.append((1 + self.max_total_draft_tokens, 1))

        def run_autotuner_pass(autotune_context: Any,
                               synchronize_trtllm_cache: bool) -> bool:
            """Run one isolated tuning pass with fresh synthetic batches."""
            ran_forward = False
            with self.no_cuda_graph(), autotune_context:
                for num_tokens, num_gen_requests in warmup_configs:
                    warmup_request = self._create_warmup_request(
                        resource_manager, num_tokens, num_gen_requests)
                    with self._release_batch_context(warmup_request,
                                                     resource_manager) as batch:
                        if not self._should_run_warmup_batch(
                                batch, num_tokens,
                                f"autotuner, num_tokens={num_tokens}, "
                                f"num_gen_requests={num_gen_requests}"):
                            continue
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
                        ran_forward = True
                        torch.cuda.synchronize()

                if ran_forward and synchronize_trtllm_cache:
                    # pp_recv in AutoTuner choose_one will never be called if there is no tuning op during the forward pass.
                    # So we need to make an extra call to consume the previous rank's pp_send to guarantee that the previous rank's pp_send is released.
                    AutoTuner.get().cache_pp_recv()
                    # Send the cache after the tuning process to the next PP rank
                    AutoTuner.get().cache_pp_send()
                    # Clean the pp flag to avoid deadlock with synchronous send/recv
                    AutoTuner.get().clean_pp_flag()
            return ran_forward

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        ran_native_forward = run_autotuner_pass(autotune(cache_path=cache_path),
                                                synchronize_trtllm_cache=True)
        ran_flashinfer_forward = False
        if enable_flashinfer_mxfp8_autotuner:
            ran_flashinfer_forward = run_autotuner_pass(
                flashinfer_mxfp8_autotune(), synchronize_trtllm_cache=False)

        if enable_flashinfer_mxfp8_autotuner:
            if ran_flashinfer_forward:
                for method in flashinfer_mxfp8_methods:
                    method.mark_flashinfer_autotuned()
            else:
                forced_flashinfer = any(method.backend == "flashinfer"
                                        for method in flashinfer_mxfp8_methods)
                for method in flashinfer_mxfp8_methods:
                    method.disable_flashinfer_auto()
                if forced_flashinfer:
                    raise RuntimeError(
                        "FlashInfer MXFP8 was explicitly requested but its autotuner "
                        "warmup forward could not run")
                logger.warning(
                    "FlashInfer MXFP8 autotuning could not run; using the native "
                    "TensorRT-LLM GEMM backend.")

        if enable_native_mxfp8_autotuner:
            if ran_native_forward:
                MXFP8GemmRunner.sync_all_tactic_caches(AutoTuner.get())
                for method in native_mxfp8_methods:
                    method.mark_native_autotuned()
            else:
                for method in native_mxfp8_methods:
                    method.disable_native_autotune()
                logger.warning(
                    "Native MXFP8 autotuning had no runnable warmup batch; "
                    "using the default native GEMM tactic.")

        logger.info(
            f"[Autotuner] Cache size after warmup is {len(AutoTuner.get().profiling_cache)}"
        )
        AutoTuner.get().print_profiling_cache()

        self._release_megamoe_profiling_scratch()

        # Clear workspace buffers allocated during the autotuner forward pass.
        # The autotuner runs a context-only forward with max_num_tokens, which
        # causes the global Buffers pool to cache large MoE/GEMM workspaces.
        # If not cleared, these inflate the memory baseline seen by the KV cache
        # profiler, reducing memory available for activations during inference.
        clear_memory_buffers()
        torch.cuda.empty_cache()

    def _run_mamba_hybrid_warmup(self,
                                 resource_manager: ResourceManager) -> None:
        """Pre-JIT the Mamba SSD multi-seq + HAS_INITSTATES=True Triton kernels.

        Mamba hybrid models (e.g. Nemotron 3 Super 120B, Nemotron-Nano-12B-v2)
        skip ``_general_warmup`` because ``can_run_general_warmup`` is False
        when the KV cache manager is a ``MambaHybridCacheManager``. The default
        ``_run_autotuner_warmup`` then issues a single ``least_requests=True``
        prefill = 1 sequence with ``num_cached_tokens_per_seq = 0``, which only
        compiles the ``num_seqs == 1`` / ``HAS_INITSTATES=False`` variants of
        the SSD kernels. The first real serve iteration with chunked prefill
        and multiple context requests then triggers autotune of the missing
        variants mid-inference, producing a ~30 s stall / large P99 spike.

        This method runs two extra forward passes to compile those variants
        during warmup:

        1. ``least_requests=False`` — splits ``curr_max_num_tokens`` into many
           short sequences, forcing the multi-seq path of
           ``cu_seqlens_to_chunk_indices_offsets_triton`` and its
           ``_cu_seqlens_triton_kernel``.
        2. ``least_requests=False`` inside
           ``Mamba2Metadata.force_initial_states_for_warmup()`` — same as (1)
           plus the ``HAS_INITSTATES=True`` variants of
           ``_state_passing_fwd_kernel``, ``_chunk_scan_fwd_kernel``, and
           ``_chunk_state_varlen_kernel``.

        Runs regardless of ``enable_autotuner``. Wraps in ``autotune()`` when
        the autotuner is enabled so op-level (M,N,K) caches also get primed
        for these shapes. Set ``TLLM_MAMBA_MULTISEQ_WARMUP=0`` to disable.
        """
        if os.environ.get("TLLM_MAMBA_MULTISEQ_WARMUP", "1") != "1":
            return
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        if kv_cache_manager is None or not isinstance(kv_cache_manager,
                                                      MambaHybridCacheManager):
            return

        token_num_upper_bound = min(self.max_num_tokens,
                                    self.batch_size * (self.max_seq_len - 1))
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self.original_max_draft_len)
        # Rank-local capacity, so peers can disagree. Leaving the phase alone
        # would unbalance the per-shape agreement inside it.
        if not self._agree_warmup_flag(curr_max_num_tokens >= 4):
            return

        # Cap the multi-seq warmup token count so we don't fill the KV cache
        # to the brim. The autotuner warmup that ran just before this uses
        # ``least_requests=True`` (few long sequences) which fits comfortably
        # even when ``curr_max_num_tokens`` is close to the block ceiling.
        # ``least_requests=False`` instead spreads the token budget across
        # ``batch_size`` short sequences; when each sequence's length lands
        # exactly on a block boundary AND the KV cache has
        # ``num_extra_kv_tokens`` > 0 (e.g. spec decoding cases),
        # ``add_token`` needs to allocate one extra block per sequence, which
        # ``_create_warmup_request``'s ``blocks_to_use`` estimate doesn't
        # account for. On a small KV pool (e.g. Qwen3.5 hybrid with DFlash spec
        # decoding on a single H100: 259 blocks total, ``max_num_tokens=8192``
        # nearly saturates it), that extra per-sequence block overflows the
        # pool and crashes with "Can't allocate new blocks for window size N".
        # The point of this warmup is only to trigger ``num_seqs > 1`` +
        # ``HAS_INITSTATES=True`` kernel variants — a modest token budget
        # achieves that with plenty of block headroom.
        WARMUP_TOKEN_CAP = 4096
        capped_num_tokens = min(curr_max_num_tokens, WARMUP_TOKEN_CAP)

        logger.info(
            "Running Mamba hybrid warmup (multi-seq + HAS_INITSTATES=True)...")

        # (num_tokens, num_gen_requests, least_requests, force_initstates)
        mamba_warmup_shapes = [
            (capped_num_tokens, 0, False, False),
            (capped_num_tokens, 0, False, True),
        ]

        autotuner_enabled = self.llm_args.enable_autotuner
        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        autotune_ctx = (autotune(cache_path=cache_path)
                        if autotuner_enabled else contextlib.nullcontext())

        with self.no_cuda_graph(), autotune_ctx:
            for (num_tokens_i, num_gen_requests_i, least_req_i,
                 force_init_i) in mamba_warmup_shapes:
                init_ctx = (Mamba2Metadata.force_initial_states_for_warmup()
                            if force_init_i else contextlib.nullcontext())
                shape = (f"Mamba hybrid, num_tokens={num_tokens_i}, "
                         f"num_gen_requests={num_gen_requests_i}, "
                         f"force_initstates={force_init_i}")
                with init_ctx:
                    try:
                        warmup_request = self._create_warmup_request(
                            resource_manager,
                            num_tokens_i,
                            num_gen_requests_i,
                            least_requests=least_req_i)
                    except torch.OutOfMemoryError as e:
                        if self._is_distributed_forward():
                            raise
                        logger.warning(f"Warmup skipped for shape ({shape}): "
                                       f"{type(e).__name__}: {e}")
                        torch.cuda.empty_cache()
                        continue
                    except RuntimeError as e:
                        # The known KV allocation failure happens before
                        # forward and is recoverable only when no peer worker
                        # can advance independently. Any other RuntimeError is
                        # a defect, not a capacity limit, and is fatal.
                        if self._is_distributed_forward():
                            raise
                        if "Can't allocate new blocks for window size" not in str(
                                e):
                            raise
                        logger.warning(f"Warmup skipped for shape ({shape}): "
                                       f"{type(e).__name__}: {e}")
                        torch.cuda.empty_cache()
                        continue

                    try:
                        with self._release_batch_context(
                                warmup_request, resource_manager) as batch:
                            if not self._should_run_warmup_batch(
                                    batch, num_tokens_i, shape):
                                continue
                            spec_resource_manager = resource_manager.get_resource_manager(
                                ResourceManagerType.SPEC_RESOURCE_MANAGER)
                            if self.is_draft_model and isinstance(
                                    spec_resource_manager,
                                    Eagle3ResourceManager):
                                spec_resource_manager.is_first_draft = True

                            self.forward(batch,
                                         new_tensors_device=None,
                                         resource_manager=resource_manager)

                            if autotuner_enabled:
                                AutoTuner.get().cache_pp_recv()
                                AutoTuner.get().cache_pp_send()
                                AutoTuner.get().clean_pp_flag()

                            torch.cuda.synchronize()
                    # Once peers can enter warmup synchronization or forward,
                    # any rank-local exception can strand them in a collective.
                    except Exception as e:  # noqa: BLE001
                        if self._is_distributed_forward():
                            raise
                        # ``torch.OutOfMemoryError`` is a ``RuntimeError``
                        # subclass; anything outside that hierarchy is a defect
                        # rather than a capacity limit.
                        if not isinstance(e, RuntimeError):
                            raise
                        # A single-rank warmup is a pure perf optimization. If
                        # a forward shape does not fit, it can be compiled
                        # lazily on the first real request.
                        logger.warning(f"Warmup skipped for shape ({shape}): "
                                       f"{type(e).__name__}: {e}")
                        # An OOM between dispatch() and combine() leaves the
                        # local MoE A2A state in ``dispatched``.
                        self._reset_moe_alltoall_state()
                        torch.cuda.empty_cache()

        clear_memory_buffers()
        torch.cuda.empty_cache()

    def _compute_dynamic_draft_len_mapping(self) -> Optional[Dict[int, int]]:
        """Compute graph_bs → draft_len mapping for dynamic draft length feature.

        Example: draft_len_schedule = {4:4, 8:2, 32:1}, cuda_graph_batch_sizes = [1,2,3,4,5,6,7,8,16,24,32,64]
        - Batch sizes 1-4:   use draft_len=4 (up to key 4)
        - Batch sizes 5-8:   use draft_len=2 (up to key 8)
        - Batch sizes 9-32:  use draft_len=1 (up to key 32)
        - Batch sizes 33+:   use draft_len=0 (implicit, speculation disabled)

        Returns: {1:4, 2:4, 3:4, 4:4, 5:2, 6:2, 7:2, 8:2, 16:1, 24:1, 32:1, 64:0}
        """
        # Dynamic draft length for CUDA graphs is only supported for one-model path
        if (not self.spec_config or not self.spec_config.draft_len_schedule or
                not self.spec_config.spec_dec_mode.support_dynamic_draft_len()):
            return None

        schedule = self.spec_config.draft_len_schedule
        schedule_keys = list(schedule.keys())

        mapping = {}
        key_idx = 0
        for graph_bs in self._cuda_graph_batch_sizes:
            while key_idx < len(
                    schedule_keys) and schedule_keys[key_idx] < graph_bs:
                key_idx += 1
            if key_idx < len(schedule_keys):
                draft_len = schedule[schedule_keys[key_idx]]
            else:
                draft_len = 0
            mapping[graph_bs] = draft_len
        return mapping

    def _get_graphs_to_capture(
            self, cuda_graph_batch_sizes: list[int]) -> list[tuple[int, int]]:
        """Determine which (batch_size, draft_len) graphs to capture.

        Returns:
            List of (batch_size, draft_len) tuples for CUDA graph capture.
        """
        # Case 1: Draft model (two-model speculative decoding)
        # Two-model path is deprecated and will be removed in the near future
        if self.is_draft_model:
            draft_len = self.max_total_draft_tokens
            return [(bs, draft_len) for bs in cuda_graph_batch_sizes]

        # Case 2: One-model with dynamic draft length
        if self.spec_config is not None and self.spec_config.draft_len_schedule is not None and self.spec_config.spec_dec_mode.support_dynamic_draft_len(
        ):
            graphs = [(graph_bs, draft_len) for graph_bs, draft_len in
                      self._dynamic_draft_len_mapping.items()]
            # Workaround for dynamic draft length:
            # capture the maximum speculative graph shape up front. Dynamic draft length
            # breaks the previous assumption that attention workspace demand can be safely
            # ordered by batch size alone; a later graph shape may require a larger shared
            # graph workspace, and resizing that workspace can change its data_ptr and
            # invalidate pointers captured by earlier graphs, causing illegal memory access
            # on replay.
            #
            # This adds the overhead of one extra captured graph, and that graph is not
            # expected to be used by the normal schedule-driven dynamic draft-length path.
            #
            # Follow-up first-principles fix:
            # query or precompute the exact attention workspace requirement for all
            # reachable graph shapes, pre-size the shared graph workspace once without
            # capturing an extra graph, and avoid resizing it in graph mode afterward.
            max_spec_graph = (max(cuda_graph_batch_sizes),
                              self.original_max_draft_len)
            if max_spec_graph not in graphs:
                graphs.append(max_spec_graph)
            logger.info(f"Dynamic draft length enabled for one-model path. "
                        f"Capturing {len(graphs)} graphs: {graphs}")
            return graphs

        # Case 2b: DSpark confidence-scheduled verification. The drafted block
        # is always full, so draft_len stays at max_draft_len. Ragged
        # verification adds only the exact measured V values to the token
        # axis, while the static fallback keeps one graph per batch size.
        # Captured graphs cost KV cache, so capture no cell the runtime cannot
        # select.
        if (self.spec_config is not None and getattr(
                self.spec_config, "enable_confidence_scheduling", False)):
            max_draft_len = int(self.spec_config.max_draft_len)
            if self._dspark_trims_submitted_tokens:
                # Keep the ordinary uniform graph for every G. A confidence
                # decision that resolves to full K deliberately clears the
                # ragged bucket so it can reuse the production static path;
                # without this native key those fallback steps silently run
                # eager even though the equivalent full-token ragged key was
                # captured. Ragged keys remain separate because their packed
                # attention metadata cannot safely alias the uniform graph.
                native_graphs = [(bs, max_draft_len)
                                 for bs in cuda_graph_batch_sizes]
                from ..speculative.dspark_planner import ExactSpsCostTable
                exact_table = getattr(self, "_dspark_sps_cost_table", None)
                if not isinstance(exact_table, ExactSpsCostTable):
                    raise RuntimeError(
                        "DSpark confidence graph capture requires an authenticated "
                        "exact SPS table")
                ragged_graphs = [(bs, max_draft_len, verifier_budget)
                                 for bs in cuda_graph_batch_sizes
                                 for verifier_budget in
                                 exact_table.production_candidate_budgets(bs)]
                graphs = native_graphs + ragged_graphs
                logger.info(
                    f"DSpark ragged verification: capturing {len(graphs)} graphs "
                    f"({len(cuda_graph_batch_sizes)} native static-K{max_draft_len} "
                    f"graphs + {len(ragged_graphs)} exact measured (G,V) cells; "
                    f"draft_len pinned to {max_draft_len}).")
                return graphs
            # Nothing varies draft_len at runtime, so shorter draft graphs
            # would never be replayed.
            graphs = [(bs, max_draft_len) for bs in cuda_graph_batch_sizes]
            logger.info(
                f"DSpark confidence scheduling: capturing {len(graphs)} graphs "
                f"({len(cuda_graph_batch_sizes)} batch sizes, draft_len pinned "
                f"to {max_draft_len}). The drafted block is always full; only "
                f"verification is trimmed.")
            return graphs

        # Case 3: Target model (two-model) or one-model without dynamic draft
        # Match the runtime_draft_len semantics enforced in _prepare_tp_inputs:
        # logical K for linear-tree modes, total tree tokens for tree decoding.
        # spec_config is None for non-spec models — fall back to max_draft_len (= 0).
        draft_lengths = [
            self.max_draft_len if
            (self.spec_config is None or self.spec_config.is_linear_tree) else
            self.max_total_draft_tokens
        ]
        should_capture_no_spec = (
            self.max_total_draft_tokens > 0
            and not self.spec_config.spec_dec_mode.use_one_engine()
            # Assume speculation is always on if no max_concurrency set (saves memory)
            and self.spec_config.max_concurrency is not None)
        if should_capture_no_spec:
            draft_lengths.append(0)
        return [(bs, draft_len) for bs in cuda_graph_batch_sizes
                for draft_len in draft_lengths]

    def _run_cuda_graph_warmup(self, resource_manager: ResourceManager):
        """Warm up or capture CUDA graphs for the configured graph shapes."""
        if not (self.cuda_graph_runner.enabled
                or self.prefill_cuda_graph_backend
                != PrefillCudaGraphBackend.DISABLED):
            return

        from ..modules.linear import (MXFP8LinearMethod,
                                      flashinfer_mxfp8_autotune,
                                      flashinfer_mxfp8_decode_graph_capture)

        # The automatic MiniMax-M3 MXFP8 selection is decode-graph-only.
        # Tune every generation graph shape during the warmup-only pass. Keep
        # piecewise context/prefill graph capture on the native backend.
        flashinfer_methods = [
            quant_method for module in self.model.modules()
            if isinstance((quant_method := getattr(module, "quant_method", None)
                           ), MXFP8LinearMethod)
            and quant_method.needs_flashinfer_autotune
        ]
        flashinfer_autotune_context = (
            flashinfer_mxfp8_autotune() if self.cuda_graph_runner.is_warmup_only
            and flashinfer_methods else contextlib.nullcontext())
        with flashinfer_autotune_context, flashinfer_mxfp8_decode_graph_capture(
        ):
            self._capture_generation_cuda_graphs(resource_manager)
        self._capture_mixed_encoder_decoder_cuda_graphs(resource_manager)
        # Piecewise graphs have separate capture machinery and do not use the
        # whole-model attention workspace. Capture them only on the second pass.
        if not self.cuda_graph_runner.is_warmup_only:
            self._capture_prefill_cuda_graphs(resource_manager)

    @torch.inference_mode()
    @with_warmup_flag
    def _warmup_encoder_cuda_graphs_enc_dec(
            self, resource_manager: ResourceManager) -> None:
        """Capture encoder-decoder encoder graphs on their runtime host thread."""
        runner = self.encoder_cuda_graph_runner
        if not runner.is_encoder_decoder:
            return

        capture = functools.partial(
            self._capture_encoder_cuda_graphs_enc_dec,
            resource_manager,
        )
        self._warmup_and_capture_encoder_cuda_graphs(capture)

    def _warmup_and_capture_encoder_cuda_graphs(
            self, capture: Callable[[], None]) -> None:
        """Warm up every encoder graph shape, then capture those shapes."""
        runner = self.encoder_cuda_graph_runner
        if not runner.enabled:
            return

        with runner.allow_capture():
            runner.is_warmup_only = True
            try:
                capture()
            finally:
                runner.is_warmup_only = False
            capture()

    def _capture_encoder_cuda_graphs_enc_dec(
            self, resource_manager: ResourceManager) -> None:
        """Warm up or capture encoder graphs used by encoder-decoder models."""
        runner = self.encoder_cuda_graph_runner
        if not runner.enabled or not runner.is_encoder_decoder:
            return

        operation = "warmup" if runner.is_warmup_only else "capture"
        num_processed = 0
        logger.info(
            f"Running encoder-decoder encoder CUDA graph {operation} ...")
        for key in sorted(runner.capture_keys, reverse=True):
            sequence_lengths = runner.get_capture_warmup_sequence_lengths(key)
            if sequence_lengths is None:
                continue

            logger.info("Encoder-decoder encoder CUDA graph "
                        f"{operation}: key={key}")
            if runner.feature_mode:
                # A zero waveform is a valid fixed-shape feature, and the
                # encoder step writes no KV cache, so no LlmRequests and no
                # KV/cross-pool resources are involved.
                self._feature_encoder_graph_forward(
                    features=[
                        torch.zeros((1, *runner.config.feature_shape),
                                    dtype=runner.config.feature_dtype)
                        for _ in sequence_lengths
                    ],
                    seq_lens=list(sequence_lengths),
                    request_ids=list(range(len(sequence_lengths))),
                )
            else:
                encoder_input_ids = [0] * sum(sequence_lengths)
                encoder_position_ids = []
                for sequence_length in sequence_lengths:
                    encoder_position_ids.extend(
                        self._apply_position_id_offset(
                            list(range(sequence_length))))
                inputs = self._prepare_encoder_decoder_encoder_inputs(
                    encoder_input_ids=encoder_input_ids,
                    encoder_position_ids=encoder_position_ids,
                    sequence_lengths=sequence_lengths,
                    request_ids=list(range(len(sequence_lengths))),
                    resource_manager=resource_manager,
                )
                self._encoder_forward_enc_dec(inputs)
            torch.cuda.synchronize()
            num_processed += 1

        logger.info("Completed encoder-decoder encoder CUDA graph "
                    f"{operation} for {num_processed} graph shape(s).")

    def _capture_generation_cuda_graphs(self,
                                        resource_manager: ResourceManager):
        """Warm up or capture pure-generation CUDA graph shapes."""
        if not self.cuda_graph_runner.enabled:
            return

        operation = ("warmup"
                     if self.cuda_graph_runner.is_warmup_only else "capture")
        logger.info(f"Running CUDA graph {operation} for "
                    f"{len(self._cuda_graph_batch_sizes)} batch sizes.")

        # Reverse order so smaller graphs can reuse memory from larger ones
        cuda_graph_batch_sizes = sorted(self._cuda_graph_batch_sizes,
                                        reverse=True)

        # Determine which graph shapes to process.
        graphs_to_capture = self._get_graphs_to_capture(cuda_graph_batch_sizes)
        graphs_to_capture = sorted(graphs_to_capture, reverse=True)
        # Create CUDA graphs for short and long sequences separately for sparse attention.
        # self.max_seq_len is the global max sequence length. For Helix CP each
        # rank only holds max_seq_len / cp_size tokens, so scale accordingly to
        # avoid creating warmup requests whose position_ids exceed the RoPE
        # table (max_position_embeddings).
        effective_max_seq_len = self.max_seq_len
        if self.mapping is not None and self.mapping.has_cp_helix():
            effective_max_seq_len = self.max_seq_len // self.mapping.cp_size

        sparse_config = self.sparse_attention_config
        if (isinstance(sparse_config, SeqLenAwareSparseAttentionConfig)
                and sparse_config.needs_separate_short_long_cuda_graphs()):
            # For short sequences, subtract the maximum runtime tokens consumed
            # by a generation step so all current-step tokens stay within the
            # sequence length threshold. PARD uses 2K tokens here, not K+1.
            max_runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(
                self.max_draft_len)
            # For long sequences, use the default maximum sequence length.
            max_seq_len = (sparse_config.seq_len_threshold -
                           max_runtime_tokens_per_gen_step)
            if max_seq_len < effective_max_seq_len:
                max_seq_len_list = [effective_max_seq_len, max_seq_len]
            else:
                max_seq_len_list = [effective_max_seq_len]
        else:
            max_seq_len_list = [effective_max_seq_len]

        def prepare_cross_batch(batch: ScheduledRequests,
                                resource_manager: ResourceManager) -> None:
            """Populate dummy gen requests' cross-KV cache before capture.

            Dummy generation requests used for graph capture never ran a
            context step, so their cross-KV cache blocks are uninitialized
            and captured kernels would read garbage. Temporarily switch each
            request to a one-token context chunk with a fake encoder output
            to run just the cross-KV projection (via _populate_cross_kv_cache),
            then restore generation state for the actual capture.
            """
            if not batch.generation_requests:
                return

            max_encoder_output_len = self._get_max_encoder_output_len(
                resource_manager)
            hidden_size = self._get_enc_dec_hidden_size()
            saved_request_state = []
            for request in batch.generation_requests:
                saved_request_state.append(
                    (request, request.py_encoder_output,
                     request.py_skip_cross_kv_projection, request.state,
                     request.py_batch_idx, request._cached_tokens,
                     request._cached_tokens_set))
                request.py_encoder_output = torch.ones(
                    (max_encoder_output_len, hidden_size),
                    device="cuda",
                    dtype=self.dtype)
                request.py_skip_cross_kv_projection = False
                request.state = LlmRequestState.CONTEXT_INIT
                request.context_current_position = 0
                request.context_chunk_size = 1

            projection_batch = ScheduledRequests()
            projection_batch.reset_context_requests(batch.generation_requests)
            kv_cache_manager = resource_manager.get_resource_manager(
                self.kv_cache_manager_key)
            draft_kv_cache_manager = self._get_draft_kv_cache_manager(
                resource_manager)
            attn_metadata = self._set_up_attn_metadata(kv_cache_manager,
                                                       draft_kv_cache_manager)
            with self.no_cuda_graph():
                projection_inputs, _ = self._prepare_inputs(
                    projection_batch,
                    kv_cache_manager,
                    attn_metadata,
                    spec_metadata=None,
                    new_tensors_device=None,
                    resource_manager=resource_manager,
                    maybe_graph=False)
                self._populate_cross_kv_cache(projection_inputs)
            torch.cuda.synchronize()

            for (request, encoder_output, skip_cross_kv_projection, state,
                 batch_idx, cached_tokens,
                 cached_tokens_set) in saved_request_state:
                request.py_encoder_output = encoder_output
                request.py_skip_cross_kv_projection = skip_cross_kv_projection
                request.state = state
                if state == LlmRequestState.GENERATION_IN_PROGRESS:
                    request.context_current_position = request.prompt_len
                request.py_batch_idx = batch_idx
                request._cached_tokens = cached_tokens
                request._cached_tokens_set = cached_tokens_set

        def _run_capture_pass(force_non_greedy: bool,
                              label: str,
                              force_lora_graph: bool,
                              entries=None) -> None:
            spec_metadata = self.spec_metadata
            assert self._force_lora_graph_for_capture is None
            self._force_lora_graph_for_capture = force_lora_graph
            if force_non_greedy and spec_metadata is not None:
                spec_metadata._force_non_greedy_for_capture = True
                # maybe_get_cuda_graph reads spec_metadata.is_all_greedy_sample
                # to build the graph cache key BEFORE populate runs inside
                # _prepare_inputs. Pre-flip it here so the very first capture
                # in this pass uses the non-greedy key; populate's override
                # below will keep it False on every subsequent iteration.
                spec_metadata.is_all_greedy_sample = False
            elif spec_metadata is not None:
                # Symmetric pre-flip for interleaved capture: a greedy capture
                # right after an advanced one would otherwise key on the stale
                # False before populate corrects it.
                spec_metadata.is_all_greedy_sample = True
            try:
                for entry in (graphs_to_capture
                              if entries is None else entries):
                    # Ragged verification adds a token-count axis, so entries
                    # may be (bs, draft_len) or (bs, draft_len, token_bucket).
                    bs, draft_len = entry[0], entry[1]
                    verify_bucket = entry[2] if len(entry) > 2 else None
                    if bs > self.batch_size:
                        continue

                    for max_seq_len in max_seq_len_list:
                        warmup_request = self._create_cuda_graph_warmup_request(
                            resource_manager,
                            bs,
                            draft_len,
                            max_seq_len,
                            force_non_greedy=force_non_greedy)
                        with self._release_batch_context(
                                warmup_request, resource_manager) as batch:
                            if batch is not None and verify_bucket is not None:
                                self._set_warmup_ragged_windows(
                                    batch, verify_bucket, draft_len)
                            if batch is None:
                                # No KV cache space for this batch size. During KV
                                # cache estimation this makes the profiling peak
                                # unrepresentative (the final executor still
                                # captures this graph), so don't skip silently.
                                logger.warning(
                                    f"Skipping CUDA graph warmup ({label}) for "
                                    f"batch size={bs}, draft_len={draft_len}: "
                                    f"not enough KV cache space.")
                                continue
                            logger.info(
                                f"Run generation-only CUDA graph {operation} ({label}) "
                                f"for batch size={bs}, draft_len={draft_len}, "
                                f"max_seq_len={max_seq_len}")
                            self.enable_spec_decode = draft_len > 0 or self.is_draft_model or (
                                self.spec_config is not None and
                                self.spec_config.spec_dec_mode.use_one_engine())
                            self._update_draft_inference_state_for_warmup(
                                batch, draft_len > 0, resource_manager)
                            self.runtime_draft_len = draft_len
                            if self._is_encoder_decoder_model():
                                prepare_cross_batch(batch, resource_manager)
                            self.forward(batch,
                                         new_tensors_device=None,
                                         resource_manager=resource_manager)
                            torch.cuda.synchronize()
            finally:
                self._force_lora_graph_for_capture = None

                # Warmup published a bucket per captured shape; the first real
                # step must not inherit the last one (a step that never reaches
                # the fit would otherwise key on a capture-time bucket).
                self.cuda_graph_runner.agreed_ragged_bucket = None

        if self.cuda_graph_lora_manager is None:
            lora_graph_cases = [False]
        elif self.llm_args.lora_config.cuda_graph_specialize_lora:
            # Capture the larger LoRA graph first so the base-only graph can
            # reuse its CUDA graph memory-pool allocations.
            lora_graph_cases = [True, False]
        else:
            lora_graph_cases = [True]

        # The capture loop stamps `self.runtime_draft_len` per captured shape
        # and nothing puts it back, so the engine would exit capture claiming
        # the LAST shape's draft length; restore it after the passes, before
        # the generic token warmup builds requests at the full draft length.
        saved_runtime_draft_len = self.runtime_draft_len
        # Pass 1: greedy fast-path (dummy requests carry no sampling params,
        # so is_all_greedy_sample is naturally True).
        for use_lora_graph in lora_graph_cases:
            label = "greedy"
            if self.cuda_graph_lora_manager is not None:
                label += ", LoRA" if use_lora_graph else ", base-only"
            _run_capture_pass(force_non_greedy=False,
                              label=label,
                              force_lora_graph=use_lora_graph)
        # Pass 2: advanced sampling variant. Required because on-the-fly capture
        # is disabled outside warmup, so any inference batch that contains a
        # non-greedy request would otherwise fall back to eager. Only meaningful
        # for one-engine spec dec (where is_all_greedy_sample participates in
        # the graph key); other paths default to True and would never key into
        # this variant.
        needs_non_greedy_capture = (
            self.spec_config is not None
            and self.spec_config.spec_dec_mode.use_one_engine())
        if needs_non_greedy_capture:
            for use_lora_graph in lora_graph_cases:
                label = "advanced sampling"
                if self.cuda_graph_lora_manager is not None:
                    label += ", LoRA" if use_lora_graph else ", base-only"
                _run_capture_pass(force_non_greedy=True,
                                  label=label,
                                  force_lora_graph=use_lora_graph)
        # Set the value back to the original value after cuda graph warmups are complete
        self.enable_spec_decode = self.is_spec_decode
        self.runtime_draft_len = saved_runtime_draft_len
        # The advanced-sampling capture pass above leaves is_all_greedy_sample
        # set to False on spec_metadata. Reset it to the default so the first
        # real iteration's graph-key selection is not seeded with this
        # capture-only value. (update_is_all_greedy_sample refreshes it every
        # iteration; this is a defensive guard.)
        if self.spec_metadata is not None:
            self.spec_metadata.is_all_greedy_sample = True

    def _capture_mixed_encoder_decoder_cuda_graphs(
            self, resource_manager: ResourceManager) -> None:
        """Warm and capture reachable mixed encoder-decoder graph shapes.

        The first global CUDA-graph pass warms every shape so shared attention
        workspace reaches its final size. The second pass captures the same
        shapes. Runtime capture is deliberately disabled because graph capture
        executes KV-cache writes and must never run against live requests.
        """
        runner = self.cuda_graph_runner
        if not runner.enable_encoder_decoder_mixed_cuda_graph:
            return

        max_encoder_output_len = self._get_max_encoder_output_len(
            resource_manager)
        context_shapes = {(batch_size, total_tokens)
                          for batch_size, total_tokens, _ in
                          self.encoder_cuda_graph_runner.capture_keys}
        if not context_shapes:
            logger.warning("Skipping mixed encoder-decoder CUDA graph capture: "
                           "no encoder CUDA graph shapes were captured.")
            return

        max_encoder_batch_size = max(batch_size
                                     for batch_size, _ in context_shapes)
        max_batch_token_counts = {
            total_tokens
            for batch_size, total_tokens in context_shapes
            if batch_size == max_encoder_batch_size
        }
        paired_context_count = 2 * max_encoder_batch_size
        if runner.max_supported_batch_size > paired_context_count:
            paired_token_counts = {
                first + second
                for first in max_batch_token_counts
                for second in max_batch_token_counts
            }
            context_shapes.update((paired_context_count, token_count)
                                  for token_count in paired_token_counts)

        operation = ("warmup" if runner.is_warmup_only else "capture")
        hidden_size = self._get_enc_dec_hidden_size()
        max_num_encoder_tokens = max(
            (total_encoder_tokens
             for num_contexts, total_encoder_tokens in context_shapes
             if total_encoder_tokens <= num_contexts * max_encoder_output_len
             and any(batch_size > num_contexts
                     for batch_size in runner.supported_batch_sizes)),
            default=0)
        if max_num_encoder_tokens == 0:
            return
        model_config = self.model.model_config.pretrained_config
        # The capture query length must equal the runtime decoder prefix or
        # every mixed batch misses its graph, silently and with no counter to
        # show it. Prefer the input processor's actual prefix (Whisper forces
        # [decoder_start, lang, task, no_timestamps] = 4); fall back to the
        # token-model heuristic: BART/mBART prepend a forced BOS token after
        # decoder_start, T5 uses decoder_start alone.
        prefix_fn = getattr(self.input_processor, "get_decoder_prefix_len",
                            None)
        mixed_context_query_len = prefix_fn() if prefix_fn is not None else None
        if not mixed_context_query_len:
            mixed_context_query_len = (2 if getattr(
                model_config, "model_type", None) in ("bart", "mbart") else 1)
        logger.info("Mixed encoder/decoder graph capture using decoder prefix "
                    f"length {mixed_context_query_len}.")
        for num_contexts, total_encoder_tokens in sorted(
                context_shapes, key=lambda shape: shape[1], reverse=True):
            if total_encoder_tokens > num_contexts * max_encoder_output_len:
                continue
            base_encoder_len, remainder = divmod(total_encoder_tokens,
                                                 num_contexts)
            encoder_output_lens = ([base_encoder_len + 1] * remainder +
                                   [base_encoder_len] *
                                   (num_contexts - remainder))
            if not encoder_output_lens or encoder_output_lens[-1] <= 0:
                continue

            for batch_size in runner.supported_batch_sizes:
                if batch_size <= num_contexts:
                    continue
                warmup_request = self._create_cuda_graph_warmup_request(
                    resource_manager,
                    batch_size,
                    draft_len=0,
                    mixed_context_encoder_output_lens=encoder_output_lens,
                    mixed_context_query_len=mixed_context_query_len)
                with self._release_batch_context(warmup_request,
                                                 resource_manager) as batch:
                    if batch is None:
                        logger.warning(
                            "Skipping mixed encoder-decoder CUDA graph "
                            f"{operation}: not enough KV cache space for "
                            f"batch size={batch_size}.")
                        continue

                    context_requests = batch.context_requests
                    for request, encoder_output_len in zip(
                            context_requests, encoder_output_lens):
                        request.state = LlmRequestState.CONTEXT_INIT
                        request.context_current_position = 0
                        request.context_chunk_size = mixed_context_query_len
                        request.cached_tokens = 0
                        request.py_batch_idx = None
                        request.py_encoder_output = torch.ones(
                            (encoder_output_len, hidden_size),
                            device="cuda",
                            dtype=self.dtype,
                        )
                        request.py_skip_cross_kv_projection = False

                    runner._get_static_encoder_hidden_states(
                        context_requests[0].py_encoder_output,
                        max_num_encoder_tokens,
                        allow_allocate=True,
                    )
                    logger.info("Run mixed encoder-decoder CUDA graph "
                                f"{operation} for batch size={batch_size}, "
                                f"context requests={num_contexts}, "
                                f"packed encoder tokens={total_encoder_tokens}")
                    saved_enable_spec_decode = self.enable_spec_decode
                    saved_runtime_draft_len = self.runtime_draft_len
                    try:
                        self.enable_spec_decode = False
                        self.runtime_draft_len = 0
                        self.forward(batch,
                                     new_tensors_device=None,
                                     resource_manager=resource_manager)
                        torch.cuda.synchronize()
                    finally:
                        self.enable_spec_decode = saved_enable_spec_decode
                        self.runtime_draft_len = saved_runtime_draft_len

    def _capture_prefill_cuda_graphs(self, resource_manager: ResourceManager):
        """Capture configured CUDA graphs for context/prefill steps."""
        if (self.prefill_cuda_graph_backend == PrefillCudaGraphBackend.DISABLED
                or (self.prefill_cuda_graph_backend
                    == PrefillCudaGraphBackend.PIECEWISE
                    and not self._torch_compile_enabled)):
            return

        logger.info("Running prefill CUDA graph warmup...")
        prefill_cuda_graph_num_tokens = sorted(
            self._prefill_cuda_graph_num_tokens, reverse=True)

        capture_context = (capture_piecewise_cuda_graph(True)
                           if self._torch_compile_piecewise_cuda_graph else
                           contextlib.nullcontext())
        with capture_context, self.no_cuda_graph():
            for num_tokens in prefill_cuda_graph_num_tokens:
                warmup_request = self._create_warmup_request(
                    resource_manager, num_tokens, 0)
                with self._release_batch_context(warmup_request,
                                                 resource_manager) as batch:
                    self._assert_all_tp_ranks_have_warmup_batch(
                        batch, num_tokens)
                    if batch is None:
                        continue

                    logger.info(
                        f"Run prefill CUDA graph capture for num tokens={num_tokens}"
                    )
                    if self.breakable_cuda_graph_runner is not None:
                        self.breakable_cuda_graph_runner.capture(
                            num_tokens, lambda: self.forward(
                                batch,
                                new_tensors_device=None,
                                resource_manager=resource_manager))
                    else:
                        # Run a few times to ensure torch.compile capture.
                        for _ in range(4):
                            self.forward(batch,
                                         new_tensors_device=None,
                                         resource_manager=resource_manager)

        # The logits allocations grow with the number of requests and are not
        # part of the captured model body. Warm up the largest request count so
        # those allocations can be reused during stable inference.
        for num_tokens in prefill_cuda_graph_num_tokens:
            warmup_request = self._create_warmup_request(resource_manager,
                                                         num_tokens,
                                                         0,
                                                         least_requests=False)
            with self._release_batch_context(warmup_request,
                                             resource_manager) as batch:
                self._assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
                if batch is None:
                    continue
                logger.info(
                    f"Run prefill CUDA graph warmup for num tokens={num_tokens} with most requests"
                )
                if self.breakable_cuda_graph_runner is not None:
                    with self.no_cuda_graph():
                        self.breakable_cuda_graph_runner.warmup(
                            lambda: self.forward(batch,
                                                 new_tensors_device=None,
                                                 resource_manager=
                                                 resource_manager),
                            steps=1)
                else:
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
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(
            resource_manager)
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        try:
            yield batch
        finally:
            if batch is not None and kv_cache_manager is not None:
                for req in batch.all_requests():
                    kv_cache_manager.free_resources(req)
                    if draft_kv_cache_manager is not None:
                        draft_kv_cache_manager.free_resources(req)
                    if cross_kv_cache_manager is not None:
                        cross_kv_cache_manager.free_resources(req)
                    if spec_resource_manager is not None:
                        spec_resource_manager.free_resources(req)

    def _create_warmup_request(
            self,
            resource_manager: ResourceManager,
            num_tokens: int,
            num_gen_requests: int,
            least_requests: bool = True) -> Optional[ScheduledRequests]:
        """Creates a generic dummy ScheduledRequests object for warmup."""
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(
            resource_manager)

        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)

        available_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=num_tokens,
            max_num_draft_tokens=self.max_total_draft_tokens)
        available_blocks = kv_cache_manager.get_num_free_blocks()
        if num_tokens > self.max_num_tokens or num_tokens > available_tokens:
            return None

        if num_gen_requests > self.batch_size:
            return None
        num_gen_tokens = num_gen_requests * (1 + self.max_total_draft_tokens)
        if num_gen_tokens > self.max_num_tokens:
            return None

        num_ctx_tokens = num_tokens - num_gen_tokens
        num_ctx_requests = 0
        ctx_requests = []
        gen_requests = []

        # Leave room for at least one decode token per request.
        max_seq_len = self.max_seq_len - 1
        if max_seq_len < 1:
            return None
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

        # Mirror add_dummy_requests' actual allocation: on top of the raw
        # token count, every sequence gets num_extra_kv_tokens add_token
        # calls, and generation dummies additionally reserve
        # max_draft_loop_tokens for the draft loop.
        # In one-engine spec modes that is (max_draft_len - 1) extra KV
        # tokens plus max_draft_len draft-loop tokens per gen dummy, i.e.
        # 2 * max_draft_len - 1 on top of the single prompt token.
        # Under-counting these let warmup start an allocation that fails
        # midway and, before the partial-allocation cleanup existed,
        # permanently leaked most of the estimation-sized KV pool
        # (TRTLLM-14903).
        def blocks_for_seq(num_tokens: int) -> int:
            return math.ceil(num_tokens / kv_cache_manager.tokens_per_block)

        extra_ctx_tokens = (getattr(kv_cache_manager, "num_extra_kv_tokens", 0)
                            or 0)
        extra_gen_tokens = extra_ctx_tokens + self.max_draft_loop_tokens
        blocks_to_use = num_full_seqs * blocks_for_seq(max_seq_len +
                                                       extra_ctx_tokens)
        if num_left_over_tokens > 0:
            blocks_to_use += blocks_for_seq(num_left_over_tokens +
                                            extra_ctx_tokens)
        blocks_to_use += (num_gen_requests * self.max_beam_width *
                          blocks_for_seq(1 + extra_gen_tokens))

        if blocks_to_use > available_blocks and isinstance(
                kv_cache_manager, KVCacheManager):
            return None

        if num_ctx_tokens > 0:
            ctx_token_nums = [max_seq_len] * num_full_seqs
            if num_left_over_tokens > 0:
                ctx_token_nums.append(num_left_over_tokens)

            ctx_requests = kv_cache_manager.add_dummy_requests(
                list(range(num_ctx_requests)),
                token_nums=ctx_token_nums,
                is_gen=False,
                max_num_draft_tokens=self.max_total_draft_tokens,
                kv_reserve_draft_tokens=self.max_draft_loop_tokens,
                use_mrope=self.use_mrope,
                draft_kv_cache_manager=draft_kv_cache_manager)

            if ctx_requests is None:
                return None

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
                kv_reserve_draft_tokens=self.max_draft_loop_tokens,
                use_mrope=self.use_mrope,
                max_beam_width=self.max_beam_width,
                draft_kv_cache_manager=draft_kv_cache_manager)

            if gen_requests is None:
                for r in ctx_requests:
                    kv_cache_manager.free_resources(r)
                    if draft_kv_cache_manager is not None:
                        draft_kv_cache_manager.free_resources(r)
                return None

            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(request_ids=list(
                    range(num_ctx_requests, num_ctx_requests +
                          num_gen_requests)))

        result = ScheduledRequests()
        result.reset_context_requests(ctx_requests)
        result.generation_requests = gen_requests
        return result

    def _create_cuda_graph_warmup_request(
        self,
        resource_manager: ResourceManager,
        batch_size: int,
        draft_len: int,
        max_seq_len: int = None,
        mixed_context_encoder_output_lens: Optional[Sequence[int]] = None,
        mixed_context_query_len: int = ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM,
        force_non_greedy: bool = False,
    ) -> Optional[ScheduledRequests]:
        """Creates a dummy ScheduledRequests tailored for CUDA graph capture."""
        capture_sampling_params = (NON_GREEDY_CAPTURE_SAMPLING_PARAMS
                                   if force_non_greedy else None)
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(
            resource_manager)

        available_blocks = kv_cache_manager.get_num_free_blocks(
        ) // self.max_beam_width
        if available_blocks < batch_size:
            return None

        result = ScheduledRequests()
        runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(
            draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
        is_enc_dec = self._is_encoder_decoder_model()
        max_encoder_output_len = (
            self._get_max_encoder_output_len(resource_manager)
            if is_enc_dec else None)
        num_mixed_contexts = len(mixed_context_encoder_output_lens or
                                 ()) if is_enc_dec else 0
        if num_mixed_contexts >= batch_size:
            return None

        # Add (batch_size - 1) dummy requests with the minimal sequence
        # length. Mixed capture must create its context rows as real context
        # requests; converting generation dummies afterward leaves their
        # native prompt/context bookkeeping at one token.
        if mixed_context_encoder_output_lens:
            context_request_ids = list(range(num_mixed_contexts))
            context_requests = kv_cache_manager.add_dummy_requests(
                context_request_ids,
                token_nums=[mixed_context_query_len] * num_mixed_contexts,
                is_gen=False,
                max_num_draft_tokens=runtime_draft_token_buffer_width,
                kv_reserve_draft_tokens=self.max_draft_loop_tokens,
                use_mrope=self.use_mrope,
                max_beam_width=self.max_beam_width,
                encoder_output_lens=list(mixed_context_encoder_output_lens),
                draft_kv_cache_manager=draft_kv_cache_manager,
                capture_sampling_params=capture_sampling_params)
            if context_requests is None:
                return None

            generation_request_ids = list(
                range(num_mixed_contexts, batch_size - 1))
            generation_requests = []
            if generation_request_ids:
                generation_requests = kv_cache_manager.add_dummy_requests(
                    generation_request_ids,
                    token_nums=[ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM] *
                    len(generation_request_ids),
                    is_gen=True,
                    max_num_draft_tokens=runtime_draft_token_buffer_width,
                    kv_reserve_draft_tokens=self.max_draft_loop_tokens,
                    use_mrope=self.use_mrope,
                    max_beam_width=self.max_beam_width,
                    encoder_output_lens=[max_encoder_output_len] *
                    len(generation_request_ids),
                    draft_kv_cache_manager=draft_kv_cache_manager,
                    capture_sampling_params=capture_sampling_params)
                if generation_requests is None:
                    for request in context_requests:
                        kv_cache_manager.free_resources(request)
                        if draft_kv_cache_manager is not None:
                            draft_kv_cache_manager.free_resources(request)
                    return None
            requests = context_requests + generation_requests
        else:
            token_nums = ([ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM] *
                          (batch_size - 1)) if is_enc_dec else None
            encoder_output_lens = ([max_encoder_output_len] *
                                   (batch_size - 1)) if is_enc_dec else None
            requests = kv_cache_manager.add_dummy_requests(
                list(range(batch_size - 1)),
                token_nums=token_nums,
                is_gen=True,
                max_num_draft_tokens=runtime_draft_token_buffer_width,
                kv_reserve_draft_tokens=self.max_draft_loop_tokens,
                use_mrope=self.use_mrope,
                max_beam_width=self.max_beam_width,
                encoder_output_lens=encoder_output_lens,
                draft_kv_cache_manager=draft_kv_cache_manager,
                capture_sampling_params=capture_sampling_params)
            if requests is None:
                return None

        def free_warmup_requests() -> None:
            for r in requests:
                kv_cache_manager.free_resources(r)
                if draft_kv_cache_manager is not None:
                    draft_kv_cache_manager.free_resources(r)

        # Add one dummy request with the maximum possible sequence length.
        max_seq_len = min(
            self.max_seq_len if max_seq_len is None else max_seq_len,
            kv_cache_manager.max_seq_len)

        # Use max_draft_loop_tokens for capacity estimation to account
        # for the actual KV reservation per request.
        _kv_draft = self.max_draft_loop_tokens
        available_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=max_seq_len,
            batch_size=batch_size,
            max_num_draft_tokens=_kv_draft)

        # Also consider draft KV cache capacity when it exists
        if draft_kv_cache_manager is not None:
            draft_available_tokens = draft_kv_cache_manager.get_num_available_tokens(
                batch_size=batch_size,
                token_num_upper_bound=max_seq_len,
                max_num_draft_tokens=_kv_draft)
            available_tokens = min(available_tokens, draft_available_tokens)

        token_num = max(
            ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM if is_enc_dec else 1,
            min(
                available_tokens, max_seq_len - 1 -
                get_num_extra_kv_tokens(self.spec_config) - _kv_draft))
        model_config = self.model.model_config.pretrained_config
        max_position_embeddings = getattr(model_config,
                                          'max_position_embeddings', None)
        if is_enc_dec:
            # For enc-dec models the engine max_seq_len covers the encoder
            # sequence, which may exceed the decoder's position table (e.g.
            # Whisper: 1500 encoder positions vs max_target_positions=448).
            decoder_position_limit = getattr(model_config,
                                             'max_target_positions', None)
            if decoder_position_limit is not None:
                max_position_embeddings = (
                    decoder_position_limit if max_position_embeddings is None
                    else min(max_position_embeddings, decoder_position_limit))
        if max_position_embeddings is not None:
            token_num = min(token_num, max_position_embeddings - _kv_draft)

        token_num = int(
            token_num)  # Ensure int for range() in add_dummy_requests

        max_seq_len_request = kv_cache_manager.add_dummy_requests(
            request_ids=[batch_size - 1],
            token_nums=[token_num],
            is_gen=True,
            max_num_draft_tokens=runtime_draft_token_buffer_width,
            kv_reserve_draft_tokens=self.max_draft_loop_tokens,
            use_mrope=self.use_mrope,
            max_beam_width=self.max_beam_width,
            encoder_output_lens=[max_encoder_output_len]
            if is_enc_dec else None,
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params)

        if max_seq_len_request is None:
            free_warmup_requests()
            return None
        else:
            max_seq_len_request = max_seq_len_request[0]

        if mixed_context_encoder_output_lens:
            requests.append(max_seq_len_request)
            for request in requests[:num_mixed_contexts]:
                request.state = LlmRequestState.CONTEXT_INIT
                request.context_current_position = 0
                request.context_chunk_size = mixed_context_query_len
                request.cached_tokens = 0
                request.py_batch_idx = None
            result.context_requests_last_chunk = requests[:num_mixed_contexts]
            result.generation_requests = requests[num_mixed_contexts:]
        else:
            # Insert the longest request first to simulate padding for the CUDA
            # graph.
            requests.insert(0, max_seq_len_request)
            result.generation_requests = requests
        if spec_resource_manager is not None:
            spec_resource_manager.add_dummy_requests(
                request_ids=list(range(batch_size)))
        if self._is_encoder_decoder_model():
            if not self._add_cross_dummy_requests(result.all_requests(),
                                                  resource_manager):
                return None
        return result

    def _get_max_encoder_output_len(self,
                                    resource_manager: ResourceManager) -> int:
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER)
        max_encoder_output_len = int(self.max_seq_len)
        if cross_kv_cache_manager is not None:
            max_encoder_output_len = min(
                max_encoder_output_len,
                int(
                    getattr(cross_kv_cache_manager, "max_seq_len",
                            max_encoder_output_len)))
        return max(1, max_encoder_output_len)

    def _add_cross_dummy_requests(self, requests: List[LlmRequest],
                                  resource_manager: ResourceManager) -> bool:
        if not requests:
            return True
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER)
        if cross_kv_cache_manager is None:
            raise RuntimeError("Encoder-decoder CUDA graph warmup requires "
                               "ResourceManagerType.CROSS_KV_CACHE_MANAGER.")

        max_encoder_output_len = self._get_max_encoder_output_len(
            resource_manager)
        for request in requests:
            request.py_encoder_output = None
            request.py_skip_cross_kv_projection = True

        encoder_output_lens = [max_encoder_output_len] * len(requests)
        cross_dummy_requests = cross_kv_cache_manager.add_dummy_requests(
            request_ids=[request.py_request_id for request in requests],
            token_nums=encoder_output_lens,
            is_gen=True,
            max_beam_width=1,
            encoder_output_lens=encoder_output_lens)
        if cross_dummy_requests is not None:
            return True

        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(
            resource_manager)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        for request in requests:
            kv_cache_manager.free_resources(request)
            if draft_kv_cache_manager is not None:
                draft_kv_cache_manager.free_resources(request)
            if spec_resource_manager is not None:
                spec_resource_manager.free_resources(request)
        return False

    def _populate_cross_kv_cache(self, inputs: Dict[str, Any]) -> None:
        encoder_hidden_states = inputs.get("encoder_hidden_states")
        cross_attn_metadata = inputs.get("cross_attn_metadata")
        if encoder_hidden_states is None or cross_attn_metadata is None:
            return

        decoder = getattr(self._get_top_level_model(), "decoder", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise RuntimeError("Encoder-decoder CUDA graph warmup requires a "
                               "decoder with cross-attention layers.")

        attn_metadata = inputs["attn_metadata"]
        hidden_states = torch.ones(
            (attn_metadata.num_tokens, self._get_enc_dec_hidden_size()),
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype)
        for layer in layers:
            cross_attn = getattr(layer, "cross_attn", None)
            if cross_attn is None:
                raise RuntimeError(
                    "Encoder-decoder CUDA graph warmup requires every decoder "
                    "layer to expose a cross_attn module.")
            cross_attn(hidden_states=hidden_states,
                       encoder_hidden_states=encoder_hidden_states,
                       attn_metadata=attn_metadata,
                       cross_attn_metadata=cross_attn_metadata,
                       skip_cross_kv_projection=False)

    def _get_enc_dec_hidden_size(self) -> int:
        config = self.model.model_config.pretrained_config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(config, "d_model", None)
        if hidden_size is None:
            raise RuntimeError(
                "Encoder-decoder CUDA graph warmup could not infer encoder "
                "hidden size from the model config.")
        return int(hidden_size)

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

    def _set_up_attn_metadata(
        self,
        kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
        draft_kv_cache_manager: Optional[Union[KVCacheManager,
                                               KVCacheManagerV2]] = None):
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

        metadata_cls = self.attn_backend.Metadata
        sparse_metadata_params = (
            self.sparse_attention_config.to_sparse_metadata_params(
                pretrained_config=config)
            if self.sparse_attention_config is not None else None)

        if kv_cache_manager is None:
            # Cache the no-cache metadata.
            if self.encoder_attn_metadata is not None:
                return self.encoder_attn_metadata
            self.encoder_attn_metadata = metadata_cls(
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
                num_heads_per_kv=num_heads_per_kv,
                sparse_metadata_params=sparse_metadata_params)
            self.encoder_attn_metadata.block_ids_per_seq = None
            self.encoder_attn_metadata.kv_block_ids_per_seq = None
            return self.encoder_attn_metadata

        if self.attn_metadata is not None:
            # This assertion can be relaxed if needed: just create a new metadata
            # object if it changes.
            assert self.attn_metadata.kv_cache_manager is kv_cache_manager
            return self.attn_metadata

        self.attn_metadata = metadata_cls(
            max_num_requests=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_num_sequences=self.batch_size * self.max_beam_width,
            kv_cache_manager=kv_cache_manager,
            draft_kv_cache_manager=draft_kv_cache_manager,
            mapping=self.mapping,
            runtime_features=self.attn_runtime_features,
            enable_flash_mla=self.model.model_config.enable_flash_mla,
            enable_context_mla_with_cached_kv=enable_context_mla_with_cached_kv,
            cache_indirection=cache_indirection,
            num_heads_per_kv=num_heads_per_kv,
            sparse_metadata_params=sparse_metadata_params,
        )
        if isinstance(kv_cache_manager, BaseMambaCacheManager):
            self.attn_metadata.mamba_chunk_size = getattr(
                config, 'chunk_size', self.attn_metadata.mamba_chunk_size)
        self.attn_metadata.mamba_metadata_cls = resolve_mamba_metadata_cls(
            self.model)

        return self.attn_metadata

    @property
    def is_multimodal(self) -> bool:
        """True iff this engine drives a multimodal model."""
        return is_multimodal(self.model, self.input_processor)

    def _validate_breakable_cuda_graph_compatibility(self) -> None:
        if self.llm_args.prefill_cuda_graph_backend != PrefillCudaGraphBackend.BREAKABLE:
            return

        if isinstance(self.model, DecoderModelForCausalLM):
            return
        decoder_model = getattr(self.model, "llm", None)
        if (self.llm_args.disable_mm_encoder
                and isinstance(decoder_model, DecoderModelForCausalLM)
                and getattr(self.model, "mm_encoder", None) is None):
            return
        if (isinstance(self.model, MultimodalModelMixin) or isinstance(
                self.input_processor, BaseMultimodalInputProcessor)):
            raise ValueError(
                "breakable prefill CUDA graph does not support multimodal models"
            )

    def forward_multimodal_encoder_items(
        self,
        requests: List[LlmRequest],
        scheduled_items: Dict[int, List[int]],
    ) -> None:
        """Forward selected MM encoder items and commit request-local outputs."""
        if not scheduled_items:
            return
        if self._mm_item_scheduler is None:
            raise TypeError(
                "Item-level MM scheduling requires MultimodalModelMixin")
        self._mm_item_scheduler.forward_items(requests, scheduled_items)

    def _set_up_spec_metadata(
            self,
            spec_resource_manager: Optional[BaseResourceManager],
            no_cache=False):
        spec_config = self.spec_config if self.enable_spec_decode else None
        # The disaggregated attention-DP overlap path opts into larger metadata
        # buffers. Passing None preserves the established max_num_requests
        # fallback for other configurations, including PP.
        num_seq_slots = (self.max_num_seq_slots
                         if self._enable_disagg_adp_overlap_headroom else None)
        if no_cache:
            return get_spec_metadata(
                spec_config,
                self.model.config,
                self.batch_size,
                max_num_tokens=self.max_num_tokens,
                spec_resource_manager=spec_resource_manager,
                is_draft_model=self.is_draft_model,
                max_seq_len=self.max_seq_len,
                num_seq_slots=num_seq_slots)

        if self.spec_metadata is not None:
            return self.spec_metadata
        self.spec_metadata = get_spec_metadata(
            spec_config,
            self.model.config,
            self.batch_size,
            max_num_tokens=self.max_num_tokens,
            spec_resource_manager=spec_resource_manager,
            is_draft_model=self.is_draft_model,
            max_seq_len=self.max_seq_len,
            num_seq_slots=num_seq_slots)
        return self.spec_metadata

    def cleanup(self) -> None:
        """Release resources owned by this model engine.

        Tears down, in order:

        1. The optional ``ModelLoader`` (which in turn releases any
           GMS client; see :meth:`ModelLoader.cleanup`).
        2. The model module reference, and the MM item scheduler, which
           holds one of its own.
        3. CUDA Graph captures (via :meth:`_release_cuda_graphs`).
        4. Input processors.

        Idempotency:
            Subsequent calls are no-ops (guarded by ``_cleanup_done``).
            The flag is set only at the end, so a partial cleanup that
            raises mid-way will be retried on the next call.

        Called from:
            :meth:`__del__`, and only from there. ``PyExecutor.shutdown``
            deliberately does *not* call this: it is also invoked mid-init by
            ``configure_kv_cache_capacity``, which reads ``model`` right
            afterwards, so clearing ``model`` here would break it. That path
            calls :meth:`_release_cuda_graphs` and then drops its reference
            instead.
        """
        if self._cleanup_done:
            return

        # Cleanup is not truly atomic: released CUDA/GMS resources cannot be
        # rolled back.  Keep each handle live until its own release succeeds,
        # so a failed cleanup can be retried without double-freeing resources
        # that were already released.
        model_loader = self.model_loader
        if model_loader is not None:
            model_loader.cleanup()
            self.model_loader = None

        # The scheduler keeps its own references to the model and the input
        # processor, so clearing the engine's attributes alone would leave the
        # weights reachable past `release_gc()` below.
        self._mm_item_scheduler = None
        self.model = None

        self._release_cuda_graphs()
        self.input_processor = None

        # Release model weights.
        release_gc()
        self._cleanup_done = True

    def __del__(self) -> None:
        """Best-effort cleanup during garbage collection.

        Delegates to :meth:`cleanup`. Catches ``RuntimeError`` (which a
        release step such as :meth:`_release_cuda_graphs` or
        ``ModelLoader.cleanup`` may raise) and ``AttributeError`` (typical
        on partially-initialized engines torn down during interpreter
        shutdown when module references have already been cleared); both
        are logged and swallowed because destructors cannot reliably
        surface exceptions.

        This is the only production caller of :meth:`cleanup` -- see the
        note there on why ``PyExecutor.shutdown`` must not call it.
        """
        try:
            self.cleanup()
        except (RuntimeError, AttributeError) as e:
            logger.warning(
                "PyTorchModelEngine cleanup failed during destruction: %s", e)

    def _init_max_seq_len(self):
        # Allow user to override the inferred max_seq_len with a warning.
        allow_long_max_model_len = os.getenv(
            "TLLM_ALLOW_LONG_MAX_MODEL_LEN",
            "0").lower() in ["1", "true", "yes", "y"]

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
            if allow_long_max_model_len:
                logger.warning(
                    f"User specified max_seq_len is larger than the config in the model config file "
                    f"({inferred_max_seq_len}). Setting max_seq_len to user's specified value {self.max_seq_len}. "
                )
            else:
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
        # Modified from tensorrt_llm/_bootstrap.py check_max_num_tokens
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
        if self._torch_compile_backend is not None:
            self._torch_compile_backend.clear_piecewise_cuda_graphs()
        if hasattr(self,
                   'cuda_graph_runner') and self.cuda_graph_runner is not None:
            self.cuda_graph_runner.clear()
        if (hasattr(self, 'breakable_cuda_graph_runner')
                and self.breakable_cuda_graph_runner is not None):
            self.breakable_cuda_graph_runner.clear()
        if hasattr(self, 'encoder_cuda_graph_runner'
                   ) and self.encoder_cuda_graph_runner is not None:
            self.encoder_cuda_graph_runner.clear()

    def get_max_num_sequences(self) -> int:
        """
        Return the maximum number of sequences that the model supports. PyExecutor needs this to compute max_num_active_requests
        """
        num_batches = self.mapping.pp_size
        return num_batches * self.batch_size

    def _should_use_full_generation_page_table(
            self, spec_config: Optional[DecodingBaseConfig],
            attn_metadata: AttentionMetadata) -> bool:
        """Return whether overlap decode needs every reserved generation page."""
        # FlashInfer metadata owns the optional device-side KV-length correction used with this
        # wider page table.
        return (self.enable_spec_decode and not self._disable_overlap_scheduler
                and getattr(spec_config, '_use_shared_kv_cache', False)
                and hasattr(attn_metadata, 'apply_spec_decode_kv_lens_offsets'))

    def _preprocess_inputs(self, inputs: Dict[str, Any]):
        """
        Make some changes to the device inputs and avoid blocking the async data transfer
        """
        attn_meta = inputs.get('attn_metadata')
        # Invalidate per-forward-pass caches so they are recomputed (and captured) on every _forward_step.
        if attn_meta is not None:
            attn_meta.on_update_kv_lens()

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
                if inputs['position_ids'].ndim == 3:  # mrope: [3, 1, N]
                    inputs['position_ids'][:, :, num_ctx_tokens:] += (
                        self.
                        previous_pos_id_offsets_cuda[:previous_batch_tokens])
                else:
                    inputs['position_ids'][0, num_ctx_tokens:] += (
                        self.
                        previous_pos_id_offsets_cuda[:previous_batch_tokens])

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
                # TRTLLM uses `kv_lens_cuda` above; FlashInfer exposes this backend-specific
                # correction without coupling the engine to its metadata type.
                elif hasattr(inputs['attn_metadata'],
                             'apply_spec_decode_kv_lens_offsets'):
                    inputs['attn_metadata'].apply_spec_decode_kv_lens_offsets(
                        self.previous_kv_lens_offsets_cuda,
                        num_gen_requests,
                        self.get_runtime_tokens_per_gen_step(
                            self.runtime_draft_len),
                        num_chunked_contexts=num_chunked_ctx_requests,
                    )

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
                if inputs['position_ids'].ndim == 3:  # mrope: [3, 1, N]
                    inputs['position_ids'][:, :, num_ctx_tokens:] -= (
                        self.
                        previous_pos_id_offsets_cuda[:previous_batch_tokens])
                else:
                    inputs['position_ids'][0, num_ctx_tokens:] -= (
                        self.
                        previous_pos_id_offsets_cuda[:previous_batch_tokens])

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
                # Restore the FlashInfer-specific logical KV lengths through the same optional hook
                # used by `_preprocess_inputs`.
                elif hasattr(inputs['attn_metadata'],
                             'apply_spec_decode_kv_lens_offsets'):
                    inputs['attn_metadata'].apply_spec_decode_kv_lens_offsets(
                        self.previous_kv_lens_offsets_cuda,
                        num_gen_requests,
                        self.get_runtime_tokens_per_gen_step(
                            self.runtime_draft_len),
                        num_chunked_contexts=num_chunked_ctx_requests,
                        restore=True,
                    )

    def _get_all_rank_num_tokens(self, attn_metadata: AttentionMetadata):
        if self.enable_attention_dp:
            num_tokens = attn_metadata.num_tokens
            if self.mapping.has_cp_helix():
                # With CP, attention uses reduce-scatter to divide tokens
                # among CP ranks. Report the post-RS token count.
                # Use tp_cp_allgather so MoE (which sees the repurposed
                # mapping where tp_size = original tp * cp) can index
                # with its tp_rank.
                num_tokens = math.ceil(num_tokens / self.mapping.cp_size)
                return self.dist.tp_cp_allgather_int64([num_tokens
                                                        ])[:, 0].tolist()
            return self.dist.tp_allgather_int64([num_tokens])[:, 0].tolist()
        return None

    def _get_all_rank_ctx_requests(self, num_ctx_requests: int):
        if self.enable_attention_dp:
            return self.dist.tp_allgather_int64([num_ctx_requests])[:,
                                                                    0].tolist()
        return None

    def _get_all_rank_num_tokens_and_spec_counts(
        self, attn_metadata: AttentionMetadata, spec_counts: Tuple[int, ...]
    ) -> Tuple[Optional[List[int]], Optional[List[List[int]]]]:
        """Exchange the attention and speculative per-rank counts in a single
        collective instead of one collective each."""
        if not self.enable_attention_dp:
            return None, None
        if self.mapping.cp_size > 1 and not self.mapping.has_cp_helix():
            # attn counts span TP only while spec counts span TP*CP; keep the
            # two exchanges separate.
            gathered = self.dist.tp_cp_allgather_int64(list(spec_counts))
            return (self._get_all_rank_num_tokens(attn_metadata),
                    gathered.T.tolist())
        num_tokens = attn_metadata.num_tokens
        if self.mapping.has_cp_helix():
            num_tokens = math.ceil(num_tokens / self.mapping.cp_size)
        gathered = self.dist.tp_cp_allgather_int64([num_tokens, *spec_counts])
        cols = gathered.T.tolist()
        return cols[0], cols[1:]

    def _sync_group_all_greedy_sample(self, spec_metadata) -> None:
        """All-gather the per-rank greedy flags and store the group AND.

        Why the sampling-path choice must be group-uniform under
        ADP + LM-head TP is documented on the anchor,
        ``SpecMetadata.group_all_greedy_sample``. Local contract: called once
        per iteration, right after ``update_is_all_greedy_sample`` and BEFORE
        the CUDA graph key is built. The gate is pure config (identical on
        every rank), so ranks also agree on whether the exchange happens; the
        gather spans the whole TP group, a superset of any LM-head-TP
        subgroup. A dedicated host all-gather rather than a piggyback on the
        ``all_rank_num_tokens`` exchange, which runs in ``_prepare_inputs`` --
        after the graph key, too late for the key to see the synced value.
        """
        # enable_lm_head_tp_in_adp implies enable_attention_dp (asserted in
        # Mapping.__init__), so ADP needs no separate check here.
        if not (self.mapping.enable_lm_head_tp_in_adp
                and spec_metadata.use_rejection_sampling):
            return
        local_flag = bool(spec_metadata.is_all_greedy_sample)
        all_flags = self.dist.tp_allgather_int64([local_flag])[:, 0]
        spec_metadata.group_all_greedy_sample = bool(all_flags.all())
        # Also overwrite the live flag directly: this iteration's scan already
        # ran (update_is_all_greedy_sample just returned) and the CUDA graph
        # key reads the flag next -- the stored override only takes effect on
        # the NEXT rescan (populate), which is after key selection.
        spec_metadata.is_all_greedy_sample = spec_metadata.group_all_greedy_sample

    def _set_spec_metadata_all_rank_num_tokens(
            self,
            spec_metadata: SpecMetadata,
            spec_all_rank_num_tokens: List[int],
            all_rank_num_seqs: List[int],
            all_rank_num_gens: Optional[List[int]] = None) -> None:
        # Eagle3 / MTP-eagle one-model use subseq_all_rank_num_tokens for
        # draft loop iterations i>0 (per-sequence counts, since each
        # sequence contributes one token per iteration).
        spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
        spec_metadata.all_rank_num_seqs = all_rank_num_seqs
        # DSpark drafts only for generation requests (it needs the bonus
        # token's target hidden states), so on mixed steps num_seqs
        # over-counts the draft MoE workload; gen-only per-rank counts keep
        # the FUSED_COMM chunk loop identical across EP ranks.
        if all_rank_num_gens is not None:
            spec_metadata.all_rank_num_gens = all_rank_num_gens
        if (spec_metadata.spec_dec_mode.is_mtp_eagle_one_model()
                or spec_metadata.spec_dec_mode.is_eagle3_one_model()):
            spec_metadata.subseq_all_rank_num_tokens = all_rank_num_seqs

    def _get_padding_params(
        self,
        total_num_tokens: int,
        num_ctx_requests: int,
        attn_all_rank_num_tokens: Optional[List[int]],
    ) -> Tuple[int, bool, Optional[List[int]]]:
        """
        Get the padding parameters for tensor padding.
        Return:
            padded_num_tokens: the padded number of tokens
            can_run_prefill_cuda_graph: whether a prefill CUDA graph can run
            attn_all_rank_num_tokens: the number of tokens for each rank
        """

        def get_padded_prefill_tokens(tokens: int) -> int:
            return self._prefill_cuda_graph_num_tokens[bisect.bisect_left(
                self._prefill_cuda_graph_num_tokens, tokens)]

        if (self.prefill_cuda_graph_backend != PrefillCudaGraphBackend.DISABLED
                and self._prefill_cuda_graph_num_tokens):
            all_rank_ctx_requests = self._get_all_rank_ctx_requests(
                num_ctx_requests)
            max_captured_num_tokens = self._prefill_cuda_graph_num_tokens[-1]
            if attn_all_rank_num_tokens is not None:
                has_ctx_requests = num_ctx_requests != 0 or (
                    all_rank_ctx_requests is not None
                    and any(ctx_requests != 0
                            for ctx_requests in all_rank_ctx_requests))
                can_run_prefill_cuda_graph = (has_ctx_requests
                                              and max(attn_all_rank_num_tokens)
                                              <= max_captured_num_tokens)
                # Inputs are rank-uniform, so the flag already agrees on
                # every rank.
                if can_run_prefill_cuda_graph:
                    padded_num_tokens = get_padded_prefill_tokens(
                        max(attn_all_rank_num_tokens))
                    logger.debug(
                        f"Pad tensor with {total_num_tokens} tokens to {padded_num_tokens} tokens"
                    )
                    return padded_num_tokens, True, [
                        padded_num_tokens
                    ] * len(attn_all_rank_num_tokens)
                else:
                    logger.debug(
                        "Not all ranks can run prefill CUDA graph, disable prefill CUDA graph"
                    )
                    return total_num_tokens, False, attn_all_rank_num_tokens
            elif num_ctx_requests != 0 and total_num_tokens <= max_captured_num_tokens:
                padded_num_tokens = get_padded_prefill_tokens(total_num_tokens)
                logger.debug(
                    f"Pad tensor with {total_num_tokens} tokens to {padded_num_tokens} tokens"
                )
                return padded_num_tokens, True, None
            else:
                logger.debug(
                    f"Prefill CUDA graph cannot be used with {total_num_tokens} tokens, {num_ctx_requests} context requests"
                )
                return total_num_tokens, False, None

        return total_num_tokens, False, attn_all_rank_num_tokens

    def _prepare_multimodal_indices(self, input_ids: list[int]):
        input_ids = torch.tensor(input_ids, dtype=torch.int, device="cpu")
        vocab_size = self.model.config.vocab_size
        # `multimodal_token_ids` is the common wrapper-model contract. Keep the legacy name as a
        # fallback for models not yet migrated to `MultimodalModelMixin`.
        mm_token_ids = getattr(self.model, "multimodal_token_ids", None)
        if mm_token_ids is None:
            mm_token_ids = getattr(self.model, "mm_token_ids", None)

        text_token_indices, mm_token_indices = filter_mm_token_from_input_ids(
            input_ids, vocab_size=vocab_size, mm_token_ids=mm_token_ids)
        return text_token_indices, mm_token_indices

    def _is_final_multimodal_context_decode_compatible(
            self, request: LlmRequest) -> bool:
        """Return whether the final prompt token uses the decode input path.

        KV reuse has already materialized every preceding prompt token. A
        multimodal final-context row therefore needs its prepared embedding
        only when the one remaining token is itself an MM placeholder. Text
        tokens can use the existing decode provider; MRoPE deltas are seeded
        into the per-sequence cache before graph lookup. An MRoPE request with
        real MM payload remains eager until its delta is available.
        """
        final_prompt_token = request.get_tokens(0)[
            request.context_current_position]
        _, mm_token_indices = self._prepare_multimodal_indices(
            [final_prompt_token])
        if mm_token_indices.numel() != 0:
            return False

        multimodal_data = request.py_multimodal_data
        if not self.use_mrope or not _has_mm_payload_keys(multimodal_data):
            return True
        return CUDAGraphRunner._get_mrope_position_delta(request) is not None

    def _is_encoder_decoder_model(self) -> bool:
        return bool(
            getattr(getattr(self.model, "model_config", None),
                    "is_encoder_decoder", False))

    def _model_encoder_graph_spec(
            self) -> Optional[Tuple[Tuple[int, ...], torch.dtype, int]]:
        """The model's fixed-shape encoder contract, or None. Queried once."""
        if not hasattr(self, "_cached_model_encoder_graph_spec"):
            # torch.compile wraps the model; the spec is on the original.
            model = getattr(self.model, "_orig_mod", self.model)
            spec_fn = getattr(model, "encoder_graph_spec", None)
            self._cached_model_encoder_graph_spec = (spec_fn() if spec_fn
                                                     is not None else None)
        return self._cached_model_encoder_graph_spec

    def _check_encoder_graph_bucket_config(
            self, encoder_cuda_graph_num_tokens: List[int],
            encoder_cuda_graph_seq_lens: List[int]) -> None:
        """Reject an encoder graph config the model cannot complete.

        A feature encoder derives both bucket lists from the model, so only a
        token encoder needs them supplied — and there they are the whole key
        space, so a config missing them can only run eager. The request to
        capture was explicit, so raise rather than degrade silently.

        Encode-only warns instead: its buckets arrive through
        `cuda_graph_config` (see `__init__`), a slot that has always accepted a
        batch-sizes-only config and run eager, so raising would break
        deployments predating feature mode.
        """
        if (self.encoder_cuda_graph_config is None
                or self._model_encoder_graph_spec() is not None):
            return
        bucket_config_error = validate_token_encoder_bucket_config(
            encoder_cuda_graph_num_tokens,
            encoder_cuda_graph_seq_lens,
            stays_eager=self._is_encode_only)
        if bucket_config_error is None:
            return
        if self._is_encode_only:
            logger.warning(bucket_config_error)
            return
        raise ValueError(bucket_config_error)

    def _encoder_graph_spec(
        self
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[torch.dtype], Optional[int]]:
        """Fixed-shape encoder contract, or (None, None, None) if unavailable.

        Returns ``(feature_shape, feature_dtype, fixed_seq_len)`` when the model
        declares ``encoder_graph_spec()`` and feature-mode encoder CUDA graphs
        are viable. The model selects the mode, not the config: an encoder
        either takes fixed-shape features or it does not. Gated to TP=1
        (allreduce inside encoder capture is unverified) and to non-draft
        models.
        """
        none = (None, None, None)
        if (self.encoder_cuda_graph_config is None or self.is_draft_model
                or not self._is_encoder_decoder_model()):
            return none

        spec = self._model_encoder_graph_spec()
        if spec is None:
            return none

        if self.mapping.tp_size > 1:
            logger.warning(
                "Feature-mode encoder CUDA graphs are gated to TP=1 in this "
                "phase; the encoder step stays eager.")
            return none

        return spec

    def _get_top_level_model(self) -> Any:
        model = getattr(self.model, "_orig_mod", self.model)
        top_level_model = getattr(model, "model", model)
        return getattr(top_level_model, "_orig_mod", top_level_model)

    @functools.cached_property
    def _model_uses_ple_recurrent_state(self) -> bool:
        """Detect PLE on text-only and multimodal model wrappers.

        The answer is fixed once the model is loaded, and the CUDA-graph gate
        below consults it on every forward that has context requests.
        """
        top_level_model = self._get_top_level_model()
        if getattr(top_level_model, "has_ple", False):
            return True
        llm = getattr(top_level_model, "llm", None)
        text_model = getattr(llm, "model", llm)
        return bool(getattr(text_model, "has_ple", False))

    def _get_position_id_offset(self) -> int:
        offset = getattr(self._get_top_level_model(), "position_id_offset", 0)
        return 0 if offset is None else int(offset)

    def _apply_position_id_offset(self, position_ids: List[int]) -> List[int]:
        offset = self._get_position_id_offset()
        if offset == 0:
            return position_ids
        return [position_id + offset for position_id in position_ids]

    def _prepare_enc_dec_cross_attn_inputs(
        self,
        encoder_hidden_states: List[torch.Tensor],
        encoder_seq_lens: List[int],
        encoder_num_cached_tokens_per_seq: List[int],
        attn_metadata: AttentionMetadata,
        resource_manager: Optional[ResourceManager],
        encoder_kv_lens: Optional[torch.Tensor] = None,
        context_encoder_kv_tokens: int = 0,
        generation_encoder_kv_tokens: int = 0,
        max_encoder_kv_len: int = 0,
    ) -> Dict[str, Any]:
        if not encoder_seq_lens:
            return {}

        if len(encoder_seq_lens) != attn_metadata.num_seqs:
            raise RuntimeError(
                "Cross-attention encoder lengths must align with decoder "
                f"sequences: got {len(encoder_seq_lens)} encoder lengths for "
                f"{attn_metadata.num_seqs} decoder sequences.")

        if resource_manager is None:
            raise RuntimeError(
                "Encoder-decoder decoder forward requires a resource manager "
                "with a cross-KV cache manager.")
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER)
        if cross_kv_cache_manager is None:
            raise RuntimeError("Encoder-decoder decoder forward requires "
                               "ResourceManagerType.CROSS_KV_CACHE_MANAGER.")

        new_encoder_tokens = sum(encoder_seq_lens)
        if encoder_hidden_states:
            packed_encoder_hidden_states = (
                encoder_hidden_states[0] if len(encoder_hidden_states) == 1 else
                torch.cat(encoder_hidden_states, dim=0))
            if packed_encoder_hidden_states.shape[0] != new_encoder_tokens:
                raise RuntimeError(
                    "Packed encoder hidden states do not match cross-attention "
                    "metadata: got "
                    f"{packed_encoder_hidden_states.shape[0]} rows for "
                    f"{new_encoder_tokens} new encoder KV tokens.")
            skip_cross_kv_projection = False
        else:
            if new_encoder_tokens != 0:
                raise RuntimeError(
                    "Cross-attention metadata asks to project encoder K/V, "
                    "but no encoder hidden states were supplied.")
            packed_encoder_hidden_states = None
            skip_cross_kv_projection = True

        def prepare_cross_metadata(
                cross_attn_metadata: AttentionMetadata) -> None:
            if encoder_kv_lens is None:
                cross_attn_metadata.prepare()
                return
            assert isinstance(cross_attn_metadata, TrtllmAttentionMetadata)
            cross_attn_metadata.prepare_encoder_decoder_from_precomputed_lengths(
                prompt_lens=attn_metadata.prompt_lens,
                kv_lens=encoder_kv_lens,
                context_kv_tokens=context_encoder_kv_tokens,
                generation_kv_tokens=generation_encoder_kv_tokens,
                max_kv_len=max_encoder_kv_len)

        if attn_metadata.is_cuda_graph and attn_metadata.has_cross_sub_metadata:
            # Fast path for stable CUDA-graph generation steps: the encoder
            # KV lengths (kv_lens_cuda) and the frozen prompt lengths
            # (prompt_lens_cuda) are identical across all generation steps
            # for a fixed batch. Skip the expensive torch.tensor() allocations
            # and H2D copies inside prepare() when nothing has changed.
            is_stable_gen_step = (
                new_encoder_tokens == 0  # pure generation, no new cross-KV
                and self._cross_attn_stable_cached_tokens
                == encoder_num_cached_tokens_per_seq
                and self._cross_attn_stable_request_ids
                == attn_metadata.request_ids  # same batch and row order
            )
            if is_stable_gen_step:
                cross_attn_metadata = attn_metadata.cross
                # Only refresh the decoder-side Python references that the
                # kernel reads; these are pointer-level updates with no alloc.
                cross_attn_metadata._seq_lens = attn_metadata.seq_lens
                cross_attn_metadata._seq_lens_cuda = attn_metadata.seq_lens_cuda
                cross_attn_metadata.prompt_lens = attn_metadata.prompt_lens
                cross_attn_metadata.request_ids = attn_metadata.request_ids
                cross_attn_metadata.num_contexts = attn_metadata.num_contexts
            else:
                cross_attn_metadata = attn_metadata.update_cross_metadata(
                    encoder_seq_lens=encoder_seq_lens,
                    cross_kv_cache_manager=cross_kv_cache_manager,
                    encoder_num_cached_tokens_per_seq=
                    encoder_num_cached_tokens_per_seq,
                )
                prepare_cross_metadata(cross_attn_metadata)
                if new_encoder_tokens == 0:
                    # Record this stable state for future fast-path use.
                    self._cross_attn_stable_cached_tokens = list(
                        encoder_num_cached_tokens_per_seq)
                    self._cross_attn_stable_request_ids = list(
                        attn_metadata.request_ids)
                else:
                    # Batch changed (new encoder request); reset cache.
                    self._cross_attn_stable_cached_tokens = None
                    self._cross_attn_stable_request_ids = None
        else:
            cross_attn_metadata = attn_metadata.create_cross_metadata(
                cross_kv_cache_manager=cross_kv_cache_manager,
                encoder_seq_lens=encoder_seq_lens,
                encoder_num_cached_tokens_per_seq=
                encoder_num_cached_tokens_per_seq,
            )
            if attn_metadata.is_cuda_graph:
                attn_metadata.cross = cross_attn_metadata
                if new_encoder_tokens == 0:
                    self._cross_attn_stable_cached_tokens = list(
                        encoder_num_cached_tokens_per_seq)
                    self._cross_attn_stable_request_ids = list(
                        attn_metadata.request_ids)
                else:
                    self._cross_attn_stable_cached_tokens = None
                    self._cross_attn_stable_request_ids = None
            else:
                self._cross_attn_stable_cached_tokens = None
                self._cross_attn_stable_request_ids = None
            prepare_cross_metadata(cross_attn_metadata)

        return {
            "encoder_hidden_states": packed_encoder_hidden_states,
            "cross_attn_metadata": cross_attn_metadata,
            "skip_cross_kv_projection": skip_cross_kv_projection,
        }

    def _ship_multimodal_indices(
        self,
        inputs: dict,
        *,
        mm_token_indices_cpu: torch.Tensor,
        text_token_indices_cpu: torch.Tensor,
        num_ctx_tokens: int,
        total_num_tokens: int,
    ) -> None:
        """Pin and async-copy executor-precomputed MM/text token indices into
        ``inputs`` so ``fuse_input_embeds`` can skip its ``torch.where`` host
        sync. If ``total_num_tokens > num_ctx_tokens`` (KV-cache path with
        extend/draft tokens appended after the indices were computed), the
        post-context positions are appended as text. Current speculative decode
        paths do not append multimodal placeholders after the context tokens."""
        mm_token_indices_cpu = maybe_pin_memory(mm_token_indices_cpu)
        inputs['mm_token_indices'] = mm_token_indices_cpu.to("cuda",
                                                             non_blocking=True)
        if total_num_tokens > num_ctx_tokens:
            extra_text = torch.arange(num_ctx_tokens,
                                      total_num_tokens,
                                      dtype=text_token_indices_cpu.dtype)
            text_token_indices_cpu = torch.cat(
                [text_token_indices_cpu, extra_text])
        text_token_indices_cpu = maybe_pin_memory(text_token_indices_cpu)
        inputs['text_token_indices'] = text_token_indices_cpu.to(
            "cuda", non_blocking=True)

    def _can_use_encoder_decoder_input_fast_path(
            self, scheduled_requests: ScheduledRequests,
            new_tokens_device: Optional[torch.Tensor],
            next_draft_tokens_device: Optional[torch.Tensor]) -> bool:
        """Return whether the TRT-like persistent input path is sufficient."""
        static_eligible = self._encoder_decoder_input_fast_path_static_eligible
        if static_eligible is None:
            static_eligible = (
                hasattr(batch_manager_bindings,
                        "prepare_encoder_decoder_inputs")
                and self._is_encoder_decoder_model() and not self.is_draft_model
                and self.max_beam_width == 1
                and self.sparse_attention_config is None and not self.use_mrope
                and not self.enable_attention_dp
                and not self.mapping.has_cp_helix() and not self.is_multimodal
                and not self.attn_runtime_features.chunked_prefill
                and not self.attn_runtime_features.cache_reuse
                and not self.attn_runtime_features.has_speculative_draft_tokens)
            self._encoder_decoder_input_fast_path_static_eligible = \
                static_eligible
        if (not static_eligible or self.enable_spec_decode
                or self.lora_model_config is not None
                or new_tokens_device is None
                or next_draft_tokens_device is not None
                or self.guided_decoder is not None):
            return False

        if scheduled_requests.batch_size == 0:
            return False
        for request in scheduled_requests.generation_requests:
            if request.py_batch_idx is None and not request.is_dummy:
                return False
        return True

    def _acquire_encoder_decoder_host_buffers(self) -> Dict[str, Any]:
        """Acquire pinned staging whose preceding asynchronous copies finished."""
        pool = self._encoder_decoder_host_buffer_pool
        for buffers in pool:
            event = buffers['event']
            if event is None or event.query():
                return buffers

        buffers = {
            'input_ids':
            torch.empty(self.max_num_tokens,
                        dtype=torch.int,
                        pin_memory=prefer_pinned()),
            'position_ids':
            torch.empty(self.max_num_tokens,
                        dtype=torch.int,
                        pin_memory=prefer_pinned()),
            'sequence_lengths':
            torch.empty(self.batch_size,
                        dtype=torch.int,
                        pin_memory=prefer_pinned()),
            'prompt_lengths':
            torch.empty(self.batch_size,
                        dtype=torch.int,
                        pin_memory=prefer_pinned()),
            'cached_token_lengths':
            torch.empty(self.batch_size,
                        dtype=torch.int,
                        pin_memory=prefer_pinned()),
            'kv_lengths':
            torch.empty(self.batch_size,
                        dtype=torch.int,
                        pin_memory=prefer_pinned()),
            'encoder_kv_lengths':
            torch.empty(self.batch_size,
                        dtype=torch.int,
                        pin_memory=prefer_pinned()),
            'previous_batch_indices':
            torch.empty(self.batch_size,
                        dtype=torch.int,
                        pin_memory=prefer_pinned()),
            'event':
            None,
        }
        pool.append(buffers)
        return buffers

    @nvtx_range("_prepare_encoder_decoder_inputs_fast")
    def _prepare_encoder_decoder_inputs_fast(
            self, scheduled_requests: ScheduledRequests,
            kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
            attn_metadata: AttentionMetadata, new_tokens_device: torch.Tensor,
            resource_manager: Optional[ResourceManager]):
        """Prepare a simple BART batch with native collation and reused buffers."""
        buffers = self._acquire_encoder_decoder_host_buffers()
        position_id_offset = self._encoder_decoder_position_id_offset
        if position_id_offset is None:
            position_id_offset = self._get_position_id_offset()
            self._encoder_decoder_position_id_offset = position_id_offset
        (request_ids, encoder_seq_lens, encoder_cached_token_lengths,
         total_num_tokens, num_context_tokens, num_previous_batch_requests,
         cached_kv_tokens, context_kv_tokens, generation_kv_tokens, max_kv_len,
         context_encoder_kv_tokens, generation_encoder_kv_tokens,
         max_encoder_kv_len
         ) = batch_manager_bindings.prepare_encoder_decoder_inputs(
             scheduled_requests.context_requests,
             scheduled_requests.generation_requests,
             buffers['input_ids'],
             buffers['position_ids'],
             buffers['sequence_lengths'],
             buffers['prompt_lengths'],
             buffers['cached_token_lengths'],
             buffers['kv_lengths'],
             buffers['encoder_kv_lengths'],
             buffers['previous_batch_indices'],
             position_id_offset,
         )

        num_sequences = scheduled_requests.batch_size
        num_context_requests = scheduled_requests.num_context_requests
        num_generation_requests = scheduled_requests.num_generation_requests
        generation_request_ids = request_ids[num_context_requests:]
        if num_context_tokens:
            self.input_ids_cuda[:num_context_tokens].copy_(
                buffers['input_ids'][:num_context_tokens], non_blocking=True)
        if num_previous_batch_requests:
            previous_slots = self.previous_batch_indices_cuda[:
                                                              num_previous_batch_requests]
            staged_request_ids = generation_request_ids[:
                                                        num_previous_batch_requests]
            # Sequence slots are stable for a request's lifetime, so the
            # device indices remain valid while this ordered batch does.
            if self._encoder_decoder_staged_request_ids != staged_request_ids:
                previous_slots.copy_(buffers['previous_batch_indices']
                                     [:num_previous_batch_requests],
                                     non_blocking=True)
                self._encoder_decoder_staged_request_ids = staged_request_ids
            generation_begin = num_context_tokens
            generation_end = generation_begin + num_previous_batch_requests
            torch.index_select(
                new_tokens_device[0, :, 0],
                dim=0,
                index=previous_slots,
                out=self.input_ids_cuda[generation_begin:generation_end])
        else:
            self._encoder_decoder_staged_request_ids = None
        dummy_begin = num_context_tokens + num_previous_batch_requests
        if dummy_begin < total_num_tokens:
            self.input_ids_cuda[dummy_begin:total_num_tokens].fill_(0)

        self.position_ids_cuda[:total_num_tokens].copy_(
            buffers['position_ids'][:total_num_tokens], non_blocking=True)
        final_position_ids = self.position_ids_cuda[:
                                                    total_num_tokens].unsqueeze(
                                                        0)

        sequence_lengths = buffers['sequence_lengths'][:num_sequences]
        attn_metadata._seq_lens = sequence_lengths
        if (attn_metadata.is_cuda_graph
                and attn_metadata._seq_lens_cuda is not None):
            attn_metadata._seq_lens_cuda.copy_(sequence_lengths,
                                               non_blocking=True)
        else:
            attn_metadata._seq_lens_cuda = sequence_lengths.cuda(
                non_blocking=True)

        attn_metadata._num_contexts = scheduled_requests.num_context_requests
        attn_metadata._num_ctx_tokens = num_context_tokens
        attn_metadata._num_generations = num_generation_requests
        attn_metadata._num_tokens = total_num_tokens
        attn_metadata.beam_width = 1
        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = buffers['prompt_lengths'][:num_sequences]
        attn_metadata.num_chunked_ctx_requests = 0
        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=buffers['cached_token_lengths']
            [:num_sequences],
            num_extra_kv_tokens=0)
        attn_metadata.kv_cache_manager = kv_cache_manager
        assert isinstance(attn_metadata, TrtllmAttentionMetadata)
        attn_metadata.prepare_encoder_decoder_from_precomputed_lengths(
            prompt_lens=buffers['prompt_lengths'][:num_sequences],
            kv_lens=buffers['kv_lengths'][:num_sequences],
            context_kv_tokens=context_kv_tokens,
            generation_kv_tokens=generation_kv_tokens,
            max_kv_len=max_kv_len)

        encoder_hidden_states = []
        for request in scheduled_requests.context_requests:
            encoder_output = request.py_encoder_output
            if encoder_output is None:
                raise RuntimeError(
                    f"Decoder context request {request.py_request_id} has no "
                    "encoder output.")
            encoder_hidden_states.append(encoder_output)
            request.py_batch_idx = request.py_seq_slot

        cross_attention_inputs = self._prepare_enc_dec_cross_attn_inputs(
            encoder_hidden_states,
            encoder_seq_lens,
            encoder_cached_token_lengths,
            attn_metadata,
            resource_manager,
            encoder_kv_lens=buffers['encoder_kv_lengths'][:num_sequences],
            context_encoder_kv_tokens=context_encoder_kv_tokens,
            generation_encoder_kv_tokens=generation_encoder_kv_tokens,
            max_encoder_kv_len=max_encoder_kv_len,
        )

        attn_all_rank_num_tokens = self._get_all_rank_num_tokens(attn_metadata)
        (padded_num_tokens, can_run_piecewise_cuda_graph,
         attn_all_rank_num_tokens) = self._get_padding_params(
             total_num_tokens, scheduled_requests.num_context_requests,
             attn_all_rank_num_tokens)
        set_per_request_prefill_cuda_graph_flag(can_run_piecewise_cuda_graph)
        attn_metadata.padded_num_tokens = (padded_num_tokens
                                           if padded_num_tokens
                                           != total_num_tokens else None)

        virtual_num_tokens = total_num_tokens
        if attn_metadata.padded_num_tokens is not None:
            self.input_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
            self.position_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens
            final_position_ids = self.position_ids_cuda[:
                                                        virtual_num_tokens].unsqueeze(
                                                            0)

        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:virtual_num_tokens],
            'position_ids': final_position_ids,
            'inputs_embeds': None,
            'multimodal_params': [],
            'resource_manager': resource_manager,
        }
        inputs.update(cross_attention_inputs)

        self.iter_states[
            'num_ctx_requests'] = scheduled_requests.num_context_requests
        self.iter_states['num_ctx_tokens'] = num_context_tokens
        self.iter_states['num_generation_tokens'] = num_generation_requests
        self.iter_states['cached_kv_tokens'] = cached_kv_tokens
        if not self.is_warmup:
            self.previous_request_ids = generation_request_ids
            self.has_previous_device_draft = False

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        buffers['event'] = event
        return inputs, None

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
        if scheduled_requests.num_context_requests > 0:
            return False

        # Check if the request_ids changes
        request_ids = [
            request.py_request_id
            for request in scheduled_requests.generation_requests
        ]
        if self.previous_request_ids != request_ids:
            return False

        # The incremental path is only valid while every request's sequence
        # length is unchanged; ragged verification re-picks windows every
        # step, so fall back to a full prepare whenever they moved.
        verify_lens = [
            getattr(request, "py_verify_len", None)
            for request in scheduled_requests.generation_requests
        ]
        if self.previous_verify_lens != verify_lens:
            return False

        has_current_device_draft = next_draft_tokens_device is not None
        return has_current_device_draft and self.has_previous_device_draft

    @nvtx_range("_apply_incremental_update")
    def _apply_incremental_update(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
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
                spec_metadata, new_tensors_device, num_accepted_tokens_device,
                resource_manager)

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
            num_extra_kv_tokens=get_num_extra_kv_tokens(spec_config),
            use_full_generation_page_table=(
                self._should_use_full_generation_page_table(
                    spec_config, attn_metadata)))
        attn_metadata.kv_cache_manager = kv_cache_manager
        attn_metadata.prepare()

        # Get LoRA parameters
        lora_params = self._get_lora_params_from_requests(
            scheduled_requests, attn_metadata)

        # Handle padding for piecewise CUDA graphs
        attn_metadata.padded_num_tokens = None

        # Handle attention DP
        spec_all_rank_counts = None
        if enable_attention_dp:
            if spec_metadata is not None:
                (attn_metadata.all_rank_num_tokens, spec_all_rank_counts
                 ) = self._get_all_rank_num_tokens_and_spec_counts(
                     attn_metadata,
                     (total_num_tokens, len(spec_metadata.seq_lens),
                      attn_metadata.num_generations))
            else:
                attn_metadata.all_rank_num_tokens = \
                    self._get_all_rank_num_tokens(attn_metadata)

        # Prepare speculative metadata
        if spec_metadata is not None:
            # Set request_accepted_path if Eagle3
            if isinstance(spec_metadata, Eagle3SpecMetadata):
                spec_metadata.request_accepted_path = request_accepted_path

            spec_metadata.num_tokens = total_num_tokens
            spec_metadata.prepare()

            # Handle distributed spec metadata
            if enable_attention_dp:
                self._set_spec_metadata_all_rank_num_tokens(
                    spec_metadata, *spec_all_rank_counts)

        # Set iteration states - batch dictionary updates
        self.iter_states.update({
            'num_ctx_requests':
            0,
            'num_ctx_tokens':
            0,
            'num_generation_tokens':
            num_generation_tokens,
            'cached_kv_tokens':
            sum(num_cached_tokens_per_seq),
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

        num_generation_tokens = scheduled_requests.num_generation_requests
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

    def _set_warmup_ragged_windows(self, batch, verify_bucket: int,
                                   draft_len: int) -> None:
        """Give a capture batch per-request windows summing to ``verify_bucket``.

        The warmup batch must hit the bucket total exactly, or the graph is
        keyed under a total no runtime batch produces and every ragged step
        silently falls back to eager. Any split with the right sum captures
        the right shape (the raggedness lives in the contents of qo_indptr).
        """
        requests = batch.generation_requests
        if not requests:
            return
        from ..speculative.dspark_ragged import RaggedVerifyLayout

        lens = torch.ones(len(requests), dtype=torch.int32)
        layout = RaggedVerifyLayout.from_verify_lens(
            lens,
            graph_num_tokens=int(verify_bucket),
            total_verify_tokens=len(requests))
        filled = layout.fill_bucket(max_verify_len=1 + int(draft_len))
        for request, tokens in zip(requests, filled.verify_lens.tolist()):
            request.py_verify_len = int(tokens) - 1
        # Publish to the runner too: the graph key carries this value, and the
        # runtime fit never runs during capture.
        self.cuda_graph_runner.agreed_ragged_bucket = int(verify_bucket)

    def ragged_verify_token_buckets(self, padded_bs: int) -> List[int]:
        """Measured compact ``V`` cells captured for one exact ``G``."""
        if self.spec_config is None:
            return []
        from ..speculative.dspark_planner import ExactSpsCostTable
        exact_table = getattr(self, "_dspark_sps_cost_table", None)
        if not isinstance(exact_table, ExactSpsCostTable):
            raise RuntimeError(
                "DSpark ragged verification requires an authenticated exact SPS table"
            )
        return list(exact_table.production_candidate_budgets(int(padded_bs)))

    def _get_spec_worker(self):
        """Return the one-model speculative worker when it is initialized."""
        return getattr(self.model, "spec_worker", None)

    def fit_ragged_verify_lens(self,
                               generation_requests,
                               verify_lens: List[int],
                               exact_shape: Tuple[int, int, int],
                               peer_stats: Optional[List[List[int]]] = None,
                               exact_zero_real: bool = False) -> Optional[int]:
        """Validate and publish a globally selected exact ``(G,V)`` shape.

        ``peer_stats`` is every attention-DP rank's ``[num_requests,
        total_tokens, all_can_graph]``, including this rank's: the bucket must
        be sized from the cross-rank maximum or ranks pick different shapes,
        and ``all_can_graph`` is the group's answer to whether that maximum
        will be taken at all.

        ``exact_shape`` is the agreed ``(G,V,pad_tokens)`` from the exact SPS
        selector. Changing V here would execute a different measured cell from
        the one the policy priced, so this method only validates the split.

        ``exact_zero_real`` means this rank contributes no logical requests
        but carries the one scheduler-owned attention-DP dummy. The dummy is
        mapped onto the high row when ``V % G != 0``; CUDA padding supplies
        the remaining low/high rows without publishing output or KV ownership
        for a real request.
        """
        runner = self.cuda_graph_runner
        # Cleared on entry so every early return leaves no stale value: a
        # bucket carried over from a previous step would key into the wrong
        # graph. The padded row count follows the same rule.
        runner.agreed_ragged_bucket = None
        runner.ragged_zero_real_high_rows = 0
        self._dspark_last_padded_bs = None
        if not runner.enabled or not verify_lens:
            return None
        if len(verify_lens) != len(generation_requests):
            # A partially windowed batch would go ragged in the token layout
            # while the spec metadata stayed uniform; refuse instead.
            logger.debug(
                f"DSpark ragged: got {len(verify_lens)} verify lengths for "
                f"{len(generation_requests)} generation requests; falling back "
                f"to uniform scheduling")
            return None
        max_verify_len = 1 + int(self.spec_config.max_draft_len)
        token_lens = [1 + int(v) for v in verify_lens]

        from ..speculative.dspark_ragged import resolve_ragged_pad_split

        # Resolve rows first, then tokens: the bucket grid depends on the
        # widest rank's row count. `all_can_graph` (third peer-stat element;
        # absent means single-rank) says whether _get_padded_batch will take
        # the cross-rank row maximum at all -- only when EVERY rank can graph.
        all_can_graph = (bool(peer_stats[0][2])
                         if peer_stats and len(peer_stats[0]) > 2 else True)
        if not all_can_graph:
            # No rank will replay a graph this step, and _get_padded_batch
            # pads differently than this fit assumes on such steps, so the
            # budget would go to rows that never appear. Decline (conservative:
            # an eager ragged step would still save compute).
            logger.debug(
                "DSpark ragged: some rank cannot run a CUDA graph this step, "
                "so no captured bucket applies; falling back to uniform")
            return None
        widest_rows = max(
            [int(s[0]) for s in peer_stats],
            default=len(token_lens)) if peer_stats else len(token_lens)
        padded_bs = int(exact_shape[0])
        expected_padded_bs = runner._round_up_batch_size(widest_rows)
        if padded_bs != expected_padded_bs:
            logger.warning(
                "DSpark exact ragged shape disagrees with graph ladder: "
                f"selected G={padded_bs}, expected G={expected_padded_bs}")
            return None
        if padded_bs == 0:
            return None
        # The fit assumes the batch will actually be padded to `padded_bs`;
        # padding may decline, and then the fitted total lands in no captured
        # bucket. Decline to go ragged rather than fit against rows that will
        # not exist (`will_pad_to` mirrors only the SIZE guards; the group's
        # graph answer is `all_can_graph` above).
        if not runner.will_pad_to(padded_bs, len(token_lens)):
            logger.debug(
                f"DSpark ragged: padding to {padded_bs} rows is not available "
                f"for {len(token_lens)} requests, so the bucket grid derived "
                f"from it would not be realised; falling back to uniform")
            return None
        buckets = self.ragged_verify_token_buckets(padded_bs)
        if not buckets:
            return None
        bucket = int(exact_shape[1])
        if bucket not in buckets:
            logger.warning(
                f"DSpark exact V={bucket} was not captured for G={padded_bs}")
            return None

        full_bucket = int(padded_bs) * int(max_verify_len)
        if bucket == full_bucket:
            # A full verifier budget has no ragged work reduction. Preserve
            # the native static-K path and its ordinary multi-G tail handling.
            return None

        if exact_zero_real:
            if (len(generation_requests) != 1
                    or not generation_requests[0].is_attention_dp_dummy
                    or len(verify_lens) != 1):
                raise RuntimeError(
                    "DSpark zero-real exact fit lost its single scheduled "
                    "attention-DP dummy invariant")
            quotient, remainder = divmod(int(bucket), int(padded_bs))
            if (quotient < 1 or quotient + int(remainder > 0) > max_verify_len):
                raise RuntimeError(
                    "DSpark zero-real exact bucket has no bounded low/high "
                    f"row split: G={padded_bs}, V={bucket}, "
                    f"max_tokens={max_verify_len}")
            expected_scheduled_window = quotient - 1 + int(remainder > 0)
            if int(verify_lens[0]) != expected_scheduled_window:
                raise RuntimeError(
                    "DSpark zero-real scheduled dummy window differs from "
                    f"the exact quotient/remainder split: got={verify_lens[0]}, "
                    f"expected={expected_scheduled_window}")
            if remainder > 1:
                draft_len = int(self.spec_config.max_draft_len)
                if draft_len not in runner.secondary_padding_dummy_requests:
                    raise RuntimeError(
                        "DSpark secondary padding dummy disappeared after "
                        "the all-rank exact-cell agreement")
            self._dspark_last_padded_bs = int(padded_bs)
            self._dspark_last_num_real = 0
            runner.agreed_ragged_bucket = int(bucket)
            runner.ragged_pad_verify_len = int(quotient) - 1
            runner.ragged_zero_real_high_rows = int(remainder)
            generation_requests[0].py_verify_len = expected_scheduled_window
            return int(bucket)

        # Pad rows are appended later as a *single shared dummy object*, so
        # they all carry the same window and their contribution must be
        # decided here, not left to the fill. They take the minimum (one
        # token) so real requests get the slack, growing only when the real
        # rows cannot absorb the bucket.
        n_real = len(token_lens)
        n_pad = padded_bs - n_real
        floor_tokens = sum(token_lens)
        fixed_pad_len = int(exact_shape[2])
        split = resolve_ragged_pad_split(
            bucket=int(bucket),
            num_real_requests=n_real,
            total_real_tokens=floor_tokens,
            padded_bs=int(padded_bs),
            max_verify_len=max_verify_len,
            fixed_pad_len=fixed_pad_len,
        )
        if split is None:
            logger.warning(
                f"DSpark ragged: bucket {bucket} admits no pad-row window "
                f"for {n_real} real requests and {n_pad} pad rows; falling "
                f"back to uniform scheduling")
            return None
        pad_len = split.pad_len
        real_target = split.real_target
        if real_target != sum(token_lens):
            logger.warning(
                "DSpark exact layout changed between policy and fit: "
                f"real tokens={sum(token_lens)}, target={real_target}, "
                f"G={padded_bs}, V={bucket}, pad={pad_len}")
            return None
        # The exact selector already returned a bounded split whose real rows
        # sum to the measured cell; repacking it cannot change a value.
        published = token_lens

        # Published only now, past the last way this fit can fail: a stale
        # published bucket on a fallback step is the state behind the ragged
        # IMA. The row count and pad window feed the fresh-confidence device
        # prologue; `ragged_pad_verify_len` is also stamped on the shared dummy
        # by _get_padded_batch.
        self._dspark_last_padded_bs = int(padded_bs)
        self._dspark_last_num_real = int(n_real)
        runner.agreed_ragged_bucket = int(bucket)
        runner.ragged_pad_verify_len = pad_len - 1

        for request, tokens in zip(generation_requests, published):
            request.py_verify_len = int(tokens) - 1
        return int(bucket)

    def _apply_device_window_prologue(self, inputs, new_tensors_device) -> bool:
        """Re-rank this step's verify windows on device, with fresh confidence.

        Runs after ``_prepare_inputs`` and before the graph replay, all device
        ops on the current stream -- stream order makes every write visible to
        the replayed graph, and nothing here reads device data back to the
        host. The host has already agreed ``(padded_bs, bucket)`` and staged a
        SHAPE SPLIT of the (lagged) budget through the normal fit; this
        prologue re-distributes the same real/pad token totals by the verified
        block's OWN confidence (ranking lag zero) and overwrites the layout's
        content: verify lens, qo_indptr, the per-token row maps, the packed
        input/position/draft tokens, and the per-request kv-length delta from
        the shape split to the true windows.

        Returns True when applied; False when a precondition fails (the step
        then runs the shape split as-is, which is a valid window assignment).
        """
        from ..speculative.dspark_device_select import (
            gather_packed_draft_tokens, select_windows_device)

        runner = self.cuda_graph_runner
        budget = self._dspark_device_budget
        self._dspark_device_budget = None
        if budget is None or runner.agreed_ragged_bucket is None:
            return False
        if not getattr(self, "_dspark_prev_covers_batch", False):
            # The input gathers below address new_tokens_device by
            # previous_batch_indices in batch order; a request without a
            # previous device tensor breaks that addressing.
            return False
        worker = self._get_spec_worker()
        planner = worker.verify_planner
        if planner is None or worker.staged_confidence_buffer() is None:
            return False
        if worker.batch_slot_view(1) is None:
            return False
        spec_metadata = inputs.get('spec_metadata')
        attn_metadata = inputs.get('attn_metadata')
        if spec_metadata is None or attn_metadata is None:
            return False
        if self.use_mrope:
            # Host-selected ragged windows build the complete three-axis
            # MRoPE positions correctly.  The device prologue currently owns
            # only the scalar position buffer, so decline fresh reranking
            # rather than leave the 3-D positions on the stale shape split.
            return False
        apply_device_layout = getattr(attn_metadata,
                                      "apply_device_ragged_layout", None)
        if not callable(apply_device_layout):
            # Device-selected row ownership is an attention-backend
            # capability.  Decline before mutating any shared buffers when a
            # backend only supports the ordinary host-selected ragged layout.
            return False
        if int(attn_metadata.num_contexts) != 0:
            return False

        bucket = int(runner.agreed_ragged_bucket)
        padded_bs = int(self._dspark_last_padded_bs)
        n_real = int(self._dspark_last_num_real)
        pad_len_tok = int(runner.ragged_pad_verify_len) + 1
        real_tokens = bucket - (padded_bs - n_real) * pad_len_tok
        cfg = planner.cfg
        # The scheduler cannot grant more than the real rows can absorb under
        # the published split; the fill tops up any shortfall.
        budget = max(
            0, min(int(budget),
                   real_tokens - n_real * (cfg.min_verify_len + 1)))

        # Snapshot the shape split BEFORE overwriting: past_seen per row is
        # the staged position at each row's first token, and the kv delta
        # needs the split the host baked into kv_lens_cuda.
        lens_buf = self.ragged_verify_lens_cuda
        qo_buf = self.ragged_qo_indptr_cuda
        split_lens = lens_buf[:padded_bs].clone()
        split_qo = qo_buf[:padded_bs + 1].to(torch.long)
        past_seen = self.position_ids_cuda[split_qo[:-1]].clone()

        expected_stamp = worker.verified_draft_seq_cuda()
        stamps = worker.confidence_stamp_buffer(
        ) if expected_stamp is not None else None
        result = select_windows_device(
            confidence_logits=worker.staged_confidence_buffer(),
            slot_idx=worker.batch_slot_view(padded_bs),
            num_real=n_real,
            budget=budget,
            graph_num_tokens=bucket,
            cfg=cfg,
            apply_calibration=planner.apply_calibration,
            stamp=stamps,
            expected_stamp=expected_stamp,
            pad_len=pad_len_tok,
        )

        lens_buf[:padded_bs].copy_(result.verify_lens)
        qo_buf[:padded_bs + 1].copy_(result.qo_indptr)
        # The host stages kv_lens as num_cached + seq_lens_kv, and BOTH terms
        # bake the per-request token window (num_cached = past + tokens;
        # seq_lens_kv = tokens), so kv_lens = past + 2*S -- the window counts
        # TWICE. Moving to the true windows therefore needs 2*(w - S), not
        # (w - S): the single-delta variant left every re-ranked request's
        # kv_len off by (w - S), which shifted the indexer K-cache slot
        # mapping (slot_mapping_fp8) and silently wrote K entries into the
        # wrong cache slots. Established empirically by an A/B tensor diff
        # against a full host restage with the same windows.
        window_delta = result.verify_lens - split_lens
        attn_metadata.kv_lens_cuda[:padded_bs] += 2 * window_delta
        # The graph adds previous_kv_lens_offsets (staged as new_tokens_lens -
        # shape_lens, per request) to kv_lens during replay; host-with-w
        # stages new_tokens_lens - w there, so the offsets move by -(w - S).
        # Combined: (past + 2S) + 2(w-S) + (new - S) - (w-S) = past + w + new,
        # exactly the host-with-w in-graph sum.
        self.previous_kv_lens_offsets_cuda[:padded_bs] -= window_delta.to(
            self.previous_kv_lens_offsets_cuda.dtype)

        req_idx = result.req_idx
        spec_metadata.remap_expanded_sampling_params(req_idx, bucket)
        device = req_idx.device
        flat = torch.arange(bucket, device=device)
        offset = flat - result.qo_indptr.to(torch.long)[req_idx]
        prev_slots = self.previous_batch_indices_cuda[:n_real].to(torch.long)
        slots_tok = prev_slots[req_idx.clamp(max=n_real - 1)]

        new_tokens_device = new_tensors_device.new_tokens
        new_tokens_lens_device = new_tensors_device.new_tokens_lens
        next_draft_tokens_device = new_tensors_device.next_draft_tokens

        self.input_ids_cuda[:bucket] = new_tokens_device.transpose(
            0, 1)[slots_tok, offset].flatten().to(self.input_ids_cuda.dtype)
        self.position_ids_cuda[:bucket] = past_seen[req_idx] + offset.to(
            past_seen.dtype)
        # Overlap corrections gather by each token's OWNER; rebuild both the
        # index and the per-token offset it feeds (mirrors the host staging at
        # the previous_pos_indices block).
        self.previous_pos_indices_cuda[:
                                       real_tokens] = slots_tok[:real_tokens].to(
                                           self.previous_pos_indices_cuda.dtype)
        self.previous_pos_id_offsets_cuda[:real_tokens].copy_(
            new_tokens_lens_device[slots_tok[:real_tokens]])
        # Draft tokens pack compactly, omitting each request's bonus/anchor.
        # Build the draft-only row owners at the statically known real-draft
        # size.  The previous scheme parked anchors at ``real_draft``; at a
        # full K / full batch that index is exactly one past the allocation.
        real_draft = real_tokens - n_real
        if real_draft > 0:
            self.draft_tokens_cuda[:real_draft].copy_(
                gather_packed_draft_tokens(
                    next_draft_tokens=next_draft_tokens_device,
                    batch_slots=prev_slots,
                    verify_lens=result.verify_lens,
                    qo_indptr=result.qo_indptr,
                    num_real=n_real,
                    total_draft_tokens=real_draft,
                ).to(self.draft_tokens_cuda.dtype))

        apply_device_layout(result.verify_lens, req_idx, result.kv_correction)
        return True

    @staticmethod
    def _ragged_token_lens(generation_requests) -> Optional[List[int]]:
        """Each generation request's token window, or None if the batch is uniform.

        ``py_verify_len`` counts drafted positions; the token window adds the
        bonus position. None unless *every* request carries a window --
        partially windowed batches are silent token misattribution.
        """
        verify_lens = [
            getattr(request, "py_verify_len", None)
            for request in generation_requests
        ]
        if not verify_lens or any(v is None for v in verify_lens):
            return None
        return [1 + int(v) for v in verify_lens]

    def _publish_gen_token_layout(self, attn_metadata,
                                  generation_requests) -> None:
        """Hand the attention metadata this step's gen-token layout, before it
        prepares.

        Split out of :meth:`_attach_ragged_verify_layout` for ordering:
        ``attn_metadata.prepare()`` is the only consumer of
        ``ragged_verify_lens`` and runs well before the spec metadata is
        assembled. The uniform stride is published here too, because
        ``max_draft_tokens`` is a static buffer-sizing ceiling that does not
        move when a shorter tier is chosen. It cannot live in
        ``update_spec_dec_param`` either: that runs against the base metadata,
        and ``prepare()`` runs against the per-key CUDA-graph copy.
        """
        if attn_metadata is None:
            return
        if hasattr(attn_metadata, "runtime_tokens_per_gen_step"):
            attn_metadata.runtime_tokens_per_gen_step = (
                self.get_runtime_tokens_per_gen_step(self.runtime_draft_len))
        if hasattr(attn_metadata, "ragged_verify_lens"):
            attn_metadata.ragged_verify_lens = (
                self._ragged_token_lens(generation_requests)
                if self._dspark_trims_submitted_tokens else None)
        if hasattr(attn_metadata, "device_windows_mode"):
            # Tells the attention prepare that the host window VALUES are a
            # shape split (bounds only); the true windows land on device
            # through apply_device_ragged_layout after prepare.
            attn_metadata.device_windows_mode = (
                self._dspark_trims_submitted_tokens
                and self._dspark_device_windows)

    def _pinned_host(self, key: str, values, dtype) -> torch.Tensor:
        """A persistent pinned staging buffer holding ``values``.

        Async H2D sources must outlive the queued copy -- PyTorch does not
        extend the lifetime of an async source, so a temporary tensor can be
        reclaimed while the DMA still reads it. Buffers are keyed by name and
        grown monotonically.
        """
        values = list(values)
        # WAR guard: a slot must not be rewritten while its previous
        # non_blocking H2D is still queued. Callers record the active slot's
        # event via _pinned_host_record after enqueuing; two slots alternate,
        # so this wait targets the copy from two steps ago and is free in
        # steady state (a single slot would stall prepare behind the previous
        # step's stream position).
        slot = 1 - self._pinned_host_active.get(key, 1)
        evt = self._pinned_host_events.get((key, slot))
        if evt is not None:
            evt.synchronize()
        bufs = self._pinned_host_cache.setdefault(key, [None, None])
        buf = bufs[slot]
        if buf is None or buf.numel() < len(values) or buf.dtype != dtype:
            buf = torch.empty(max(len(values), 1),
                              dtype=dtype,
                              pin_memory=prefer_pinned())
            bufs[slot] = buf
        self._pinned_host_active[key] = slot
        view = buf[:len(values)]
        if values:
            view.copy_(torch.tensor(values, dtype=dtype))
        return view

    def _pinned_host_record(self, *keys: str) -> None:
        """Mark the H2D copies from these staging buffers as enqueued.

        Call after the non_blocking copy that reads a `_pinned_host` view.
        Skipped during graph capture (the staging copies are eager on every
        step, including graph steps; only the one-time capture pass is not).
        """
        if torch.cuda.is_current_stream_capturing():
            return
        for key in keys:
            slot = self._pinned_host_active.get(key)
            if slot is None:
                continue
            evt = self._pinned_host_events.get((key, slot))
            if evt is None:
                evt = self._pinned_host_events[(key, slot)] = torch.cuda.Event()
            evt.record()

    def _attach_ragged_verify_layout(self, spec_metadata, attn_metadata,
                                     generation_requests) -> None:
        """Publish this step's per-request verify windows to both metadatas.

        Only the DSpark confidence scheduler sets ``py_verify_len``; every
        other path leaves the fields None. Acceptance needs the windows to
        slice correctly; the DSA indexer needs them to expand kv_lens/block
        tables per query token.
        """
        if not self._dspark_trims_submitted_tokens:
            spec_metadata.verify_lens = None
            spec_metadata.qo_indptr = None
            spec_metadata.total_verify_tokens = None
            if attn_metadata is not None and hasattr(attn_metadata,
                                                     "ragged_verify_lens"):
                attn_metadata.ragged_verify_lens = None
            return

        token_lens = self._ragged_token_lens(generation_requests)
        if token_lens is None:
            spec_metadata.verify_lens = None
            spec_metadata.qo_indptr = None
            spec_metadata.total_verify_tokens = None
            if attn_metadata is not None and hasattr(attn_metadata,
                                                     "ragged_verify_lens"):
                attn_metadata.ragged_verify_lens = None
            return

        from ..speculative.dspark_ragged import build_qo_indptr

        n = len(token_lens)
        # Persistent buffers: a captured graph baked in the address it saw at
        # capture time, so a fresh tensor would be invisible to every replay.
        lens_view = self.ragged_verify_lens_cuda[:n]
        lens_view.copy_(self._pinned_host("ragged_verify_lens", token_lens,
                                          torch.int32),
                        non_blocking=True)
        self._pinned_host_record("ragged_verify_lens")
        indptr_view = self.ragged_qo_indptr_cuda[:n + 1]
        indptr_view.copy_(build_qo_indptr(lens_view), non_blocking=True)
        spec_metadata.verify_lens = lens_view
        spec_metadata.qo_indptr = indptr_view
        spec_metadata.total_verify_tokens = sum(token_lens)
        if attn_metadata is not None and hasattr(attn_metadata,
                                                 "ragged_verify_lens"):
            attn_metadata.ragged_verify_lens = token_lens

    def _ragged_gather_indices(
            self, slots: List[int],
            counts: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Row/column index pair for gathering a ragged block out of a
        [num_slots, max_width] device tensor.

        ``slots[i]`` contributes ``counts[i]`` entries taken from columns
        ``0..counts[i]-1``, concatenated in batch order. This replaces the
        ``tensor[slots, :width]`` strided gather used when every request
        verifies the same number of positions.
        """
        rows, cols = ragged_gather_index_lists(slots, counts)
        rows_dev = self._pinned_host("gather_rows", rows,
                                     torch.long).to('cuda', non_blocking=True)
        self._pinned_host_record("gather_rows")
        cols_dev = self._pinned_host("gather_cols", cols,
                                     torch.long).to('cuda', non_blocking=True)
        self._pinned_host_record("gather_cols")
        return (rows_dev, cols_dev)

    def _update_target_input_tensors(
            self,
            num_accepted_tokens_device: torch.Tensor,
            new_tokens_device: torch.Tensor,
            next_draft_tokens_device: torch.Tensor,
            new_tokens_lens_device: torch.Tensor,
            previous_slots: torch.Tensor,
            total_num_tokens: int,
            num_extend_reqeust_wo_dummy: int,
            num_tokens_per_extend_request: int,
            previous_batch_draft_tokens: int,
            tokens_per_extend_request: Optional[List[int]] = None,
            previous_batch_slots: Optional[List[int]] = None):
        """
        This function performs in-place updates on position_ids, num_accepted_draft_tokens,
        input_ids, draft_tokens, and offset tensors for speculative decoding extend context operations.

        ``tokens_per_extend_request`` carries the per-request token counts under
        ragged verification; it is None (and every gather below stays strided
        on ``num_tokens_per_extend_request``) whenever the batch is uniform.
        """
        is_ragged = tokens_per_extend_request is not None

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
        # CRITICAL: Only extract the needed tokens based on num_tokens_per_extend_request
        # new_tokens_device shape: [batch, 1 + max_draft_len]
        # We need: [previous_batch, num_tokens_per_extend_request]
        if is_ragged:
            token_rows, token_cols = self._ragged_gather_indices(
                previous_batch_slots, tokens_per_extend_request)
            new_tokens = new_tokens_device.transpose(0,
                                                     1)[token_rows,
                                                        token_cols].flatten()
        else:
            new_tokens = new_tokens_device.transpose(
                0, 1)[previous_slots, :num_tokens_per_extend_request].flatten()
        self.input_ids_cuda[:total_num_tokens].copy_(new_tokens,
                                                     non_blocking=True)

        # Prepare draft tokens
        if is_ragged:
            if previous_batch_draft_tokens > 0:
                draft_rows, draft_cols = self._ragged_gather_indices(
                    previous_batch_slots,
                    [t - 1 for t in tokens_per_extend_request])
                self.draft_tokens_cuda[:previous_batch_draft_tokens].copy_(
                    next_draft_tokens_device[draft_rows, draft_cols].flatten(),
                    non_blocking=True)
        else:
            num_draft_tokens_per_extend_request = num_tokens_per_extend_request - 1
            self.draft_tokens_cuda[:previous_batch_draft_tokens].copy_(
                next_draft_tokens_device[
                    previous_slots, :num_draft_tokens_per_extend_request].
                flatten(),
                non_blocking=True)

        # Compute kv_len_offsets and update offset tensors.
        # kv_len_offsets pairs with num_cached_tokens_per_seq (past_seen +
        # tokens_this_step) computed by the caller: the two cancel to
        # past_seen + accepted, so both must use the same per-request count.
        if is_ragged:
            tokens_per_request_device = torch.tensor(
                tokens_per_extend_request,
                dtype=torch.long,
                pin_memory=prefer_pinned()).to(new_tokens_lens_device.device,
                                               non_blocking=True)
            # `output_size` is not optional: without it repeat_interleave with
            # tensor `repeats` reads the cumulative sum back to the host -- a
            # device->host sync on every ragged step.
            previous_pos_indices = torch.repeat_interleave(
                previous_slots,
                tokens_per_request_device,
                output_size=total_num_tokens)
            previous_kv_len_offsets = (new_tokens_lens_device[previous_slots] -
                                       tokens_per_request_device)
        else:
            previous_pos_indices = previous_slots.repeat_interleave(
                num_tokens_per_extend_request)
            kv_len_offsets_device = new_tokens_lens_device - num_tokens_per_extend_request
            previous_kv_len_offsets = kv_len_offsets_device[previous_slots]
        self.previous_pos_indices_cuda[:total_num_tokens].copy_(
            previous_pos_indices, non_blocking=True)
        self.previous_pos_id_offsets_cuda[:total_num_tokens].copy_(
            new_tokens_lens_device[
                self.previous_pos_indices_cuda[:total_num_tokens]],
            non_blocking=True)
        self.previous_kv_lens_offsets_cuda[:num_extend_reqeust_wo_dummy].copy_(
            previous_kv_len_offsets, non_blocking=True)

    def _apply_incremental_update_target(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager: KVCacheManager,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            new_tensors_device: Optional[SampleStateTensors] = None,
            num_accepted_tokens_device: Optional[torch.Tensor] = None,
            resource_manager: Optional[ResourceManager] = None):
        # Extract tensors from new_tensors_device
        new_tokens_device = new_tensors_device.new_tokens  # [batch, 1 + draft_len]
        new_tokens_lens_device = new_tensors_device.new_tokens_lens  # [batch]
        next_draft_tokens_device = new_tensors_device.next_draft_tokens  # [batch, draft_len]

        # Pre-compute constants
        extend_requests = scheduled_requests.generation_requests
        num_extend_requests = len(extend_requests)
        spec_config = self.spec_config
        num_tokens_per_extend_request = self.get_runtime_tokens_per_gen_step(
            self.runtime_draft_len)

        prompt_lengths = torch.empty(num_extend_requests,
                                     dtype=torch.int,
                                     device='cpu',
                                     pin_memory=prefer_pinned())
        num_cached_tokens_per_seq = torch.empty(num_extend_requests,
                                                dtype=torch.int,
                                                device='cpu',
                                                pin_memory=prefer_pinned())
        previous_batch_indices = torch.empty(num_extend_requests,
                                             dtype=torch.int,
                                             device='cpu',
                                             pin_memory=prefer_pinned())

        request_accepted_path = {}
        num_extend_dummy_requests = 0
        num_previous_batch = 0
        # Per-request token counts, in previous-batch order. Stays all-equal to
        # num_tokens_per_extend_request unless a ragged scheduler assigned each
        # request its own verify window.
        tokens_per_extend_request = []
        previous_batch_slots = []

        use_extend_ctx = (self.enable_spec_decode
                          and spec_config.spec_dec_mode.extend_ctx(
                              self.attn_backend) and spec_config.is_linear_tree)

        for idx, request in enumerate(extend_requests):
            request_accepted_path[request.py_request_id] = \
                request.py_num_accepted_draft_tokens_indices

            base_past_seen = request.max_beam_num_tokens - 1
            # Under ragged verification every length derived below has to come
            # from this one per-request value; mixing it with the batch-wide
            # count desynchronizes the flat token layout from the KV-length
            # correction applied in _preprocess_inputs.
            req_tokens_per_gen_step = (get_request_tokens_per_gen_step(
                request, num_tokens_per_extend_request)
                                       if self._dspark_trims_submitted_tokens
                                       else num_tokens_per_extend_request)

            if use_extend_ctx:
                # We're treating the prompt lengths as context requests here, so
                # the prompt lens should not include the cached tokens.
                prompt_lengths[idx] = req_tokens_per_gen_step
            else:
                prompt_lengths[idx] = request.py_prompt_len

            # Physical KV length for the kernels: subtract the tokens a
            # KV-cache compression manager evicted (tracked on the request,
            # 0 without compression). Position ids and the cached_tokens stat
            # keep the logical count.
            if request.is_dummy:
                num_cached_tokens_per_seq[idx] = base_past_seen
                request.cached_tokens = base_past_seen
                num_extend_dummy_requests += 1
            else:
                # Request has previous tensor
                previous_batch_indices[
                    num_previous_batch] = request.py_batch_idx
                num_previous_batch += 1
                previous_batch_slots.append(request.py_batch_idx)
                tokens_per_extend_request.append(req_tokens_per_gen_step)

                request.cached_tokens = (base_past_seen +
                                         req_tokens_per_gen_step)
                num_cached_tokens_per_seq[idx] = (
                    base_past_seen + req_tokens_per_gen_step -
                    request.py_num_compressed_tokens)

            request.py_batch_idx = request.py_seq_slot

        num_extend_reqeust_wo_dummy = num_extend_requests - num_extend_dummy_requests
        is_ragged_gen = self._dspark_trims_submitted_tokens and any(
            tokens != num_tokens_per_extend_request
            for tokens in tokens_per_extend_request)
        total_num_tokens = sum(tokens_per_extend_request)

        previous_slots = self.previous_batch_indices_cuda[:num_previous_batch]
        previous_slots.copy_(previous_batch_indices[:num_previous_batch],
                             non_blocking=True)

        prompt_lengths = prompt_lengths.tolist()
        num_cached_tokens_per_seq = num_cached_tokens_per_seq.tolist()

        previous_batch_draft_tokens = (total_num_tokens -
                                       num_extend_reqeust_wo_dummy)

        self._update_target_input_tensors(
            num_accepted_tokens_device=num_accepted_tokens_device,
            new_tokens_device=new_tokens_device,
            next_draft_tokens_device=next_draft_tokens_device,
            new_tokens_lens_device=new_tokens_lens_device,
            previous_slots=previous_slots,
            total_num_tokens=total_num_tokens,
            num_extend_reqeust_wo_dummy=num_extend_reqeust_wo_dummy,
            num_tokens_per_extend_request=num_tokens_per_extend_request,
            previous_batch_draft_tokens=previous_batch_draft_tokens,
            tokens_per_extend_request=(tokens_per_extend_request
                                       if is_ragged_gen else None),
            previous_batch_slots=previous_batch_slots)

        # Prepare spec_metadata. Dummy requests are padded to the batch-wide
        # window, so only the real requests contribute ragged token counts.
        num_generation_tokens = (
            total_num_tokens +
            num_extend_dummy_requests * num_tokens_per_extend_request)
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
            'resource_manager': resource_manager,
        }

        if bool(lora_params):
            inputs['lora_params'] = lora_params

        if spec_metadata is not None:
            inputs['spec_metadata'] = spec_metadata

        return inputs, self.gather_ids_cuda[:num_generation_tokens]

    def _can_use_steady_gen_fast_prepare(
            self, scheduled_requests: ScheduledRequests,
            new_tokens_device: Optional[torch.Tensor],
            next_draft_tokens_device: Optional[torch.Tensor],
            spec_metadata: Optional[SpecMetadata]) -> bool:
        """Check whether the cached steady-state generation prepare applies.

        The cache is only recorded by a full _prepare_tp_inputs pass whose
        batch consisted purely of non-dummy generation requests that all had
        a previous overlap-scheduler tensor (see the recording site), so the
        per-step check only needs to confirm the dynamic conditions: still a
        generation-only batch with the exact same requests in the same order.
        """
        cache = self._steady_gen_cache
        if cache is None or self.is_warmup:
            return False
        if new_tokens_device is None or next_draft_tokens_device is not None \
                or spec_metadata is not None:
            return False
        if scheduled_requests.num_context_requests > 0:
            return False
        generation_requests = scheduled_requests.generation_requests
        if len(generation_requests) != cache['num_requests']:
            return False
        return cache['request_ids'] == [
            request.py_request_id for request in generation_requests
        ]

    @nvtx_range("_apply_steady_gen_fast_prepare")
    def _apply_steady_gen_fast_prepare(
            self, kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
            attn_metadata: AttentionMetadata,
            new_tensors_device: SampleStateTensors,
            resource_manager: Optional[ResourceManager]):
        """Prepare inputs for an unchanged generation-only batch.

        Every request advanced by exactly one committed token since the last
        prepare, so instead of re-walking the batch in Python this advances
        the cached positions in place (device position buffer plus a pinned
        host counter), reuses the seq-slot buffer already on device, and
        refreshes only the per-step metadata. For mrope models (recorded only
        for batches with no actual mrope work) the (3,1,N) broadcast buffer
        the model reads is the one advanced.
        """
        cache = self._steady_gen_cache
        num_requests = cache['num_requests']

        # Positions and cached-token counts are the same values in this
        # regime; advance both by one. The device-side position buffer is
        # advanced in place: it still holds the previous step's positions
        # because only _prepare_tp_inputs writes it and the cache validity
        # invariant guarantees the previous pass wrote these same rows. This
        # avoids reusing a mutated pinned buffer as the source of an async
        # H2D whose previous-step copy may still be pending under the overlap
        # scheduler (the nvbug 6293536 hazard class; see
        # KVCacheManager._stage_block_offsets_for_copy). The pinned buffer is
        # host-side bookkeeping only.
        use_mrope = cache['use_mrope']
        positions = self._steady_gen_positions_pinned[:num_requests]
        positions.add_(1)
        if use_mrope:
            # Text-only batch on an mrope model: the recording pass broadcast
            # the scalar positions onto all three axes of the (3,1,N) buffer,
            # which is what the model (and any captured CUDA graph) reads, so
            # advance it in place. position_ids_cuda is reseeded by the next
            # full pass.
            self.mrope_position_ids_cuda[:, :, :num_requests].add_(1)
        else:
            self.position_ids_cuda[:num_requests].add_(1)
        num_cached_tokens_per_seq = positions.tolist()

        # Gather this step's input tokens from the previous iteration's device
        # sample buffer; the seq-slot indices in previous_batch_indices_cuda
        # are unchanged since the last full pass.
        previous_slots = self.previous_batch_indices_cuda[:num_requests]
        new_tokens = new_tensors_device.new_tokens[:1, previous_slots, :self.
                                                   max_beam_width]
        self.input_ids_cuda[:num_requests * self.max_beam_width].copy_(
            new_tokens.flatten(), non_blocking=True)

        if not attn_metadata.is_cuda_graph:
            attn_metadata.seq_lens = cache['seq_lens_ones']
        attn_metadata.beam_width = 1
        attn_metadata.request_ids = cache['request_ids']
        attn_metadata.prompt_lens = cache['prompt_lens']
        attn_metadata.num_contexts = 0
        attn_metadata.num_chunked_ctx_requests = 0
        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            num_extra_kv_tokens=get_num_extra_kv_tokens(None))
        attn_metadata.kv_cache_manager = kv_cache_manager
        if hasattr(self.model.model_config.pretrained_config, 'chunk_size'):
            attn_metadata.mamba_chunk_size = \
                self.model.model_config.pretrained_config.chunk_size
        with nvtx_range("steady_gen_metadata_prepare"):
            attn_metadata.prepare()

        attn_all_rank_num_tokens = self._get_all_rank_num_tokens(attn_metadata)
        padded_num_tokens, can_run_piecewise_cuda_graph, attn_all_rank_num_tokens = \
            self._get_padding_params(num_requests, 0, attn_all_rank_num_tokens)
        set_per_request_prefill_cuda_graph_flag(can_run_piecewise_cuda_graph)
        attn_metadata.padded_num_tokens = (
            padded_num_tokens if padded_num_tokens != num_requests else None)
        virtual_num_tokens = num_requests
        if attn_metadata.padded_num_tokens is not None:
            self.input_ids_cuda[num_requests:padded_num_tokens].fill_(0)
            # Zero-fill the padding tail of whichever position layout the
            # model consumes, matching the full pass.
            if use_mrope:
                self.mrope_position_ids_cuda[:, :, num_requests:
                                             padded_num_tokens].fill_(0)
            else:
                self.position_ids_cuda[num_requests:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens

        self.iter_states['num_ctx_requests'] = 0
        self.iter_states['num_ctx_tokens'] = 0
        self.iter_states['num_generation_tokens'] = num_requests
        self.iter_states['cached_kv_tokens'] = sum(num_cached_tokens_per_seq)

        if use_mrope:
            final_position_ids = \
                self.mrope_position_ids_cuda[:, :, :virtual_num_tokens]
        else:
            final_position_ids = \
                self.position_ids_cuda[:virtual_num_tokens].unsqueeze(0)
        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:virtual_num_tokens],
            'position_ids': final_position_ids,
            'inputs_embeds': None,
            'multimodal_params': [],
            'resource_manager': resource_manager,
        }
        return inputs, None

    def _prepare_tp_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        new_tensors_device: Optional[SampleStateTensors] = None,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
        num_accepted_tokens_device: Optional[torch.Tensor] = None,
        req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None,
        resource_manager: Optional[ResourceManager] = None,
        maybe_graph: bool = False,
        promoted_context_request_ids: frozenset[int] = frozenset(),
        use_lora_graph: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        """
        Prepare inputs for Pytorch Model.
        """

        new_tokens_device, new_tokens_lens_device, next_draft_tokens_device = None, None, None
        if new_tensors_device is not None:
            # speculative decoding cases: [batch, 1 + draft_len], others: [batch]
            new_tokens_device = new_tensors_device.new_tokens
            # When using overlap scheduler with speculative decoding, the target model's inputs would be SampleStateTensorsSpec.
            if isinstance(new_tensors_device, SampleStateTensorsSpec):
                assert self.enable_spec_decode and not self.is_draft_model
                new_tokens_lens_device = new_tensors_device.new_tokens_lens  # [batch]
                next_draft_tokens_device = new_tensors_device.next_draft_tokens  # [batch, draft_len]

        # Must be before the update of py_batch_idx
        if self.guided_decoder is not None:
            self.guided_decoder.add_batch(
                scheduled_requests,
                new_tokens=new_tokens_device,
                runtime_draft_len=self.runtime_draft_len)

        if (not promoted_context_request_ids
                and self._can_use_incremental_update(scheduled_requests,
                                                     new_tokens_device,
                                                     next_draft_tokens_device)):
            # Spec engines never record the steady-gen cache, but invalidate
            # defensively so the two fast paths can never interleave if the
            # gates ever evolve.
            self._steady_gen_cache = None
            self._encoder_decoder_staged_request_ids = None
            return self._apply_incremental_update(
                scheduled_requests, kv_cache_manager, attn_metadata,
                spec_metadata, new_tensors_device, cache_indirection_buffer,
                num_accepted_tokens_device, req_id_to_old_request,
                resource_manager)

        if (not promoted_context_request_ids
                and type(attn_metadata) is TrtllmAttentionMetadata
                and self._can_use_encoder_decoder_input_fast_path(
                    scheduled_requests, new_tokens_device,
                    next_draft_tokens_device)):
            return self._prepare_encoder_decoder_inputs_fast(
                scheduled_requests, kv_cache_manager, attn_metadata,
                new_tokens_device, resource_manager)

        self._encoder_decoder_staged_request_ids = None
        if (not promoted_context_request_ids
                and self._can_use_steady_gen_fast_prepare(
                    scheduled_requests, new_tokens_device,
                    next_draft_tokens_device, spec_metadata)):
            return self._apply_steady_gen_fast_prepare(kv_cache_manager,
                                                       attn_metadata,
                                                       new_tensors_device,
                                                       resource_manager)
        # Any full pass invalidates the steady-state cache; it is re-recorded
        # at the end of this pass when the batch qualifies.
        self._steady_gen_cache = None

        # Hoist self.use_mrope to a function-scope local so the per-request /
        # per-context-request mrope branches use LOAD_FAST instead of LOAD_ATTR.
        _use_mrope = self.use_mrope

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
        # One-model rejection: slots of gen requests that produced 0 real draft
        # tokens this step (marked in _handle_dynamic_draft_len); their stale
        # draft_probs rows are one-hot'd after spec_metadata.prepare().
        padding_gen_slots = []
        multimodal_params_list = []
        mrope_position_ids = [
        ]  # (start_idx, end_idx, (3,1,L) mrope_pos_ids) per multimodal request
        mrope_delta_write_seq_slots = []
        mrope_delta_read_seq_slots = []
        # Whether any generation request in this batch carries real MRoPE
        # metadata; see the post-loop cleanup below.
        has_gen_mrope_delta = False
        # Extra model-side cache slot reserved for CUDA graph / warmup dummy
        # requests, whose outputs are discarded, and for generation requests
        # that carry no MRoPE metadata at all. The cache is zero-initialized and
        # the write path only ever targets real ``py_seq_slot``s, so this slot
        # permanently reads back a zero delta.
        mrope_dummy_seq_slot = self.max_num_tokens * self.mapping.pp_size
        num_accepted_draft_tokens = []  # per request
        is_enc_dec = self._is_encoder_decoder_model()
        cross_encoder_hidden_states: List[torch.Tensor] = []
        cross_encoder_seq_lens: List[int] = [
        ]  # new encoder K/V tokens per decoder sequence
        cross_encoder_cached_tokens_per_seq: List[int] = []
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

        context_prompt_lookahead = None
        if (spec_metadata is not None
                and spec_metadata.context_prompt_lookahead_tokens is not None):
            context_prompt_lookahead = []

        def append_cross_attention_state(request: LlmRequest,
                                         project_encoder_output: bool,
                                         repeat: int = 1) -> None:
            if not is_enc_dec:
                return

            encoder_output_len = int(request.encoder_output_len)
            if project_encoder_output:
                encoder_output = getattr(request, "py_encoder_output", None)
                if encoder_output is None:
                    raise RuntimeError(
                        "Decoder context request "
                        f"{request.py_request_id} has no encoder output. "
                        "The encoder iteration must populate "
                        "req.py_encoder_output before the first decoder "
                        "context step.")
                if encoder_output.shape[0] != encoder_output_len:
                    raise RuntimeError(
                        "Decoder context request "
                        f"{request.py_request_id} encoder output length "
                        f"({encoder_output.shape[0]}) does not match "
                        f"encoder_output_len ({encoder_output_len}).")
                cross_encoder_hidden_states.append(encoder_output)
                cross_encoder_seq_lens.append(encoder_output_len)
                cross_encoder_cached_tokens_per_seq.append(0)
                return

            for _ in range(repeat):
                cross_encoder_seq_lens.append(0)
                cross_encoder_cached_tokens_per_seq.append(encoder_output_len)

        for request in scheduled_requests.context_requests:
            request_ids.append(request.py_request_id)
            draft_lens.append(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            if context_prompt_lookahead is not None:
                context_prompt_lookahead.append(
                    _get_context_prompt_lookahead_token(request, end_compute))
            # Fetch only the current chunk. get_tokens(0) marshals the whole
            # O(seq_len) VecTokens into a Python list of boxed ints; chunked
            # prefill re-enters this loop for every chunk of the same prompt, so
            # that is O(L) per chunk = O(L^2/chunk) over the prefill.
            # get_tokens_range copies only [begin, end) -> O(chunk).
            prompt_tokens = request.get_tokens_range(0, begin_compute,
                                                     end_compute)
            position_ids.extend(
                range(begin_compute, begin_compute + len(prompt_tokens)))

            # Start offset of this request's (current-chunk) tokens within the
            # flattened input_ids. Recorded on multimodal_params below so models
            # that rewrite token IDs in place write into the request's own span
            # rather than assuming a contiguous multimodal prefix.
            context_start_idx = len(input_ids)
            # Track position for updating the inputs of draft model
            if self.is_draft_model and num_accepted_tokens_device is not None:
                input_ids.extend(prompt_tokens)
                end_idx = len(input_ids)
                slot_idx = req_id_to_old_request[
                    request.py_request_id].py_seq_slot
                context_input_ids_positions.append(
                    (context_start_idx, end_idx - 1,
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
            num_cached_tokens_per_seq.append(past_seen_token_num -
                                             request.py_num_compressed_tokens)
            request.cached_tokens = past_seen_token_num
            append_cross_attention_state(
                request,
                project_encoder_output=not request.py_skip_cross_kv_projection
                and
                (not getattr(request, "is_dummy", False)
                 or getattr(request, "py_encoder_output", None) is not None))

            # Embed mask is required only for partial iterations (chunked
            # prefill or KV-cache reuse); full-prefill degrades gracefully.
            check_mm_embed_cumsum_if_needed(
                request.py_multimodal_data,
                begin_compute=past_seen_token_num,
                end_compute=end_compute,
                prompt_len=request.get_num_tokens(0),
            )
            mm_data = request.py_multimodal_data or {}
            cumsum = mm_data.get('multimodal_embed_mask_cumsum')
            py_multimodal_runtime = None
            if cumsum is not None:
                py_multimodal_runtime = MultimodalRuntimeData(
                    embed_mask_cumsum=cumsum,
                    past_seen_token_num=past_seen_token_num,
                    chunk_end_pos=end_compute,
                )

            multimodal_params = MultimodalParams(
                multimodal_input=_build_request_multimodal_input(
                    request, self._mm_encoder_cache_enabled),
                multimodal_data=request.py_multimodal_data,
                multimodal_runtime=py_multimodal_runtime,
                mm_item_order=getattr(request, "py_mm_item_order", None),
                input_ids_start_offset=context_start_idx)
            # Transfer any cross-iter MM encoder prefetch event stamped on the request onto the
            # freshly-built MultimodalParams. The downstream consume site reads it from the wrapper,
            # not from the request.
            # NOTE: the prefetch producer always writes the cached embedding into
            # `py_multimodal_data` before stamping the event, so whenever the event is present,
            # `has_content()` below is `True` and the wrapper reaches the consume site that waits on
            # it.
            mm_encoder_event = request.py_mm_encoder_event
            if mm_encoder_event is not None:
                multimodal_params.encoder_event = mm_encoder_event
                request.py_mm_encoder_event = None
            if multimodal_params.has_content():
                # TODO(TRTLLM-14726): Check the persistent MM encoder cache before H2D and avoid
                # transferring raw encoder inputs for full hits in both regular and
                # side-stream-prefetched paths.
                multimodal_params.to_device("multimodal_data",
                                            "cuda",
                                            pin_memory=prefer_pinned(),
                                            target_keywords=getattr(
                                                self.model,
                                                "multimodal_data_device_paths",
                                                None))
                if _use_mrope:
                    # A request may carry multimodal content but no MRoPE
                    # metadata (a text-only prompt whose input processor skips
                    # ``mrope_config``, or a model that does not consume it).
                    # Its per-axis positions are just the scalar positions,
                    # which the (3,1,N) seeding further below already
                    # broadcasts, so leave that span alone.
                    mrope_config = multimodal_params.multimodal_data.get(
                        'mrope_config') or {}
                    mrope_pos_ids = mrope_config.get('mrope_position_ids')
                    if mrope_pos_ids is not None:
                        ctx_mrope_position_ids = mrope_pos_ids[:, :,
                                                               begin_compute:
                                                               begin_compute +
                                                               len(prompt_tokens
                                                                   )]
                        # Record as (start_idx, end_idx, (3,1,L) mrope_pos_ids)
                        mrope_position_ids.append(
                            (len(position_ids) - len(prompt_tokens),
                             len(position_ids), ctx_mrope_position_ids))
                    mrope_position_delta = mrope_config.get(
                        'mrope_position_deltas')
                    if mrope_position_delta is not None:
                        request.py_mrope_position_delta = mrope_position_delta
                    if (mrope_position_delta is not None
                            and request.py_seq_slot is not None):
                        mrope_delta_write_seq_slots.append(request.py_seq_slot)
                        request.py_mrope_delta_cache_slot = request.py_seq_slot

                #re-assign the multimodal_data to the request after to_device for generation requests
                request.py_multimodal_data = multimodal_params.multimodal_data
                multimodal_params_list.append(multimodal_params)

                # Re-register mrope tensors for context-only requests (EPD disaggregated serving).
                # This creates new IPC handles owned by the prefill worker, so the decode worker
                # can access them even after the encode worker's GC deallocates the original memory.
                # Without this, the decode worker would receive handles pointing to freed memory.
                if (request.is_context_only_request and _use_mrope and
                        "mrope_config" in multimodal_params.multimodal_data):
                    mrope_config = multimodal_params.multimodal_data[
                        "mrope_config"]
                    _mrope_position_ids = mrope_config.get("mrope_position_ids")
                    _mrope_position_deltas = mrope_config.get(
                        "mrope_position_deltas")
                    if _mrope_position_ids is not None and _mrope_position_deltas is not None:
                        # Clone to allocate new memory owned by this (prefill) worker.
                        request.py_result.set_mrope_position(
                            _mrope_position_ids.clone(),
                            _mrope_position_deltas.clone())

            request.py_batch_idx = request.py_seq_slot

        num_ctx_requests = scheduled_requests.num_context_requests
        num_ctx_tokens = len(input_ids)
        if len(multimodal_params_list) > 0:
            # input_ids holds only context tokens here; extend/draft tokens are
            # appended below and are by construction text, so we reuse the
            # CPU-side text_token_indices and just extend it with the
            # post-context arange instead of recomputing via a bool mask +
            # torch.where over the full range.
            text_token_indices_ctx, mm_token_indices = \
                self._prepare_multimodal_indices(input_ids)
        else:
            text_token_indices_ctx = None
            mm_token_indices = None

        # Requests with draft tokens are treated like extend requests. Dummy extend requests should be
        # at the end of extend_requests.
        extend_requests = []
        extend_dummy_requests = []
        generation_requests = []
        first_draft_requests = []
        # Collect generation request IDs during categorization to avoid
        # a separate iteration over scheduled_requests.generation_requests later.
        all_gen_request_ids = []
        for request in scheduled_requests.generation_requests:
            is_promoted_context = (request.py_request_id
                                   in promoted_context_request_ids)
            if not is_promoted_context:
                all_gen_request_ids.append(request.py_request_id)
            # In speculative iterations, keep promoted rows ahead of existing
            # generation rows in the extend-request packing order. Although
            # their q_len is one, this category provides the "no previous
            # speculative tensor" branch needed to source their prompt token
            # without disturbing the overlap offsets of ordinary generation
            # siblings. Non-speculative promoted rows retain the established
            # ordinary generation path below.
            if is_promoted_context and self.enable_spec_decode:
                extend_requests.append(request)
            elif is_promoted_context:
                generation_requests.append(request)
            elif (get_draft_token_length(request) > 0
                  or next_draft_tokens_device is not None):
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

        # For tree decoding, runtime_draft_len should match total tree
        # tokens (not tree depth).  py_executor resets it every iteration.
        if spec_config is not None and not spec_config.is_linear_tree:
            self.runtime_draft_len = self.max_total_draft_tokens

        # will contain previous batch indices of generation requests
        previous_batch_indices = []
        previous_pos_indices = []
        # Token count each previous-batch request contributes, in batch order.
        # Uniform speculation makes every entry runtime_tokens_per_gen_step;
        # ragged verification does not, and the device-side gathers below
        # switch to an index-list layout when they disagree.
        previous_batch_tokens_per_request = (
            [] if self._dspark_trims_submitted_tokens else None)
        # Flat generation-token offset of the first previous-batch request,
        # captured while walking the batch so the ragged layout does not have
        # to assume a fixed stride.
        previous_batch_token_start = None
        extend_tokens_emitted = 0
        runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(
            self.runtime_draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
        for request in extend_requests:
            is_promoted_context = (request.py_request_id
                                   in promoted_context_request_ids)
            if getattr(request, "py_needs_onehot_draft_probs", False):
                if request.py_seq_slot is not None:
                    padding_gen_slots.append(request.py_seq_slot)
                request.py_needs_onehot_draft_probs = False  # consume once
            request_ids.append(request.py_request_id)
            request_accepted_path[
                request.
                py_request_id] = request.py_num_accepted_draft_tokens_indices
            # the request has no previous tensor:
            # (1) next_draft_tokens_device is None, which means overlap scheduler is disabled; or
            # (2) a dummy request; or
            # (3) the first step in the generation server of disaggregated serving
            if (is_promoted_context or next_draft_tokens_device is None
                    or request.is_dummy or request.py_batch_idx is None):
                # get token ids, including input token ids and draft token ids. For these dummy requests,
                # no need to copy the token ids.
                # Only the request's own window is submitted to the target;
                # taking the full drafted length here would desynchronize the
                # flat token layout from the spec metadata and the graph key.
                num_draft_tokens = get_request_tokens_per_gen_step(
                    request, 1 + get_draft_token_length(request)) - 1
                if not (request.is_attention_dp_dummy
                        or request.is_cuda_graph_dummy):
                    if is_promoted_context:
                        input_ids.append(
                            request.get_tokens(0)[
                                request.context_current_position])
                    else:
                        input_ids.append(request.get_last_tokens(0))
                    input_ids.extend(request.py_draft_tokens[:num_draft_tokens])
                    draft_tokens.extend(
                        request.py_draft_tokens[:num_draft_tokens])
                # get other ids and lengths
                past_seen_token_num = (request.context_current_position
                                       if is_promoted_context else
                                       request.max_beam_num_tokens - 1)
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
                              len(position_ids) + 1 + num_draft_tokens)))
                position_ids.extend(
                    list(
                        range(past_seen_token_num,
                              past_seen_token_num + 1 + num_draft_tokens)))
                num_cached_tokens_per_seq.append(
                    past_seen_token_num - request.py_num_compressed_tokens)
                request.cached_tokens = past_seen_token_num
                extend_tokens_emitted += 1 + num_draft_tokens
                # update batch index
                request.py_batch_idx = request.py_seq_slot
            else:
                # update batch index
                previous_batch_idx = request.py_batch_idx
                request.py_batch_idx = request.py_seq_slot

                if previous_batch_token_start is None:
                    previous_batch_token_start = extend_tokens_emitted

                # Under ragged verification each request gets its own window;
                # every length below has to come from the same per-request
                # value, otherwise the flat token layout and the KV-length
                # correction in _preprocess_inputs disagree.
                req_tokens_per_gen_step = (get_request_tokens_per_gen_step(
                    request, runtime_tokens_per_gen_step) if
                                           self._dspark_trims_submitted_tokens
                                           else runtime_tokens_per_gen_step)

                sequence_lengths.append(req_tokens_per_gen_step)
                num_accepted_draft_tokens.append(
                    request.py_num_accepted_draft_tokens)
                past_seen_token_num = request.max_beam_num_tokens - 1

                draft_lens.append(req_tokens_per_gen_step - 1)
                gather_ids.extend(
                    list(
                        range(len(position_ids),
                              len(position_ids) + req_tokens_per_gen_step)))
                position_ids.extend(
                    list(
                        range(past_seen_token_num,
                              past_seen_token_num + req_tokens_per_gen_step)))
                # previous tensor
                previous_batch_indices.append(previous_batch_idx)
                previous_pos_indices.extend([previous_batch_idx] *
                                            req_tokens_per_gen_step)
                if previous_batch_tokens_per_request is not None:
                    previous_batch_tokens_per_request.append(
                        req_tokens_per_gen_step)
                extend_tokens_emitted += req_tokens_per_gen_step

                num_cached_tokens_per_seq.append(
                    past_seen_token_num + req_tokens_per_gen_step -
                    request.py_num_compressed_tokens)
                request.cached_tokens = (past_seen_token_num +
                                         req_tokens_per_gen_step)
                if self.enable_spec_decode and spec_config.spec_dec_mode.extend_ctx(
                        self.attn_backend) and spec_config.is_linear_tree:
                    prompt_lengths.append(req_tokens_per_gen_step)
                else:
                    prompt_lengths.append(request.py_prompt_len)

            append_cross_attention_state(request, project_encoder_output=False)

        for request in first_draft_requests:
            request_ids.append(request.py_request_id)
            draft_lens.append(0)
            # Only the length and the last (original_max_draft_len+1) tokens are
            # needed here; get_num_tokens is O(1) and get_tokens_range copies only
            # the requested window, whereas get_tokens(0) marshals the whole
            # O(seq_len) VecTokens into a Python list.
            _num_tokens = request.get_num_tokens(0)
            begin_compute = _num_tokens - self.original_max_draft_len - 1
            end_compute = begin_compute + self.original_max_draft_len + 1
            prompt_tokens = request.get_tokens_range(0, begin_compute,
                                                     end_compute)
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
            num_cached_tokens_per_seq.append(past_seen_token_num -
                                             request.py_num_compressed_tokens)
            append_cross_attention_state(request, project_encoder_output=False)

            # update batch index
            request.py_batch_idx = request.py_seq_slot

        helix_is_inactive_rank, helix_position_offsets = [], []
        # Cache invariant method result to avoid repeated calls per-request
        _has_cp_helix = self.mapping.has_cp_helix()
        _n_gen = len(generation_requests)
        # One-shot batch-level flag — True iff any generation request actually
        # carries multimodal payload. Lets the strip_mm_data branch below
        # short-circuit on a LOAD_FAST rather than a per-request LOAD_ATTR
        # of py_multimodal_data for non-multimodal models (the gpt-oss-120b
        # GEN case).
        _has_any_multimodal_request = any(r.py_multimodal_data is not None
                                          for r in generation_requests)
        if _n_gen > 0:
            # The whole batch is laid out with request 0's beam width: every
            # generation request contributes exactly this many rows to
            # input_ids / position_ids / sequence_lengths and to the logits the
            # model returns. The sampler, in turn, locates a request's logits by
            # accumulating the *per-request* beam widths
            # (TorchSampler._select_generated_logits ->
            # calculate_request_offsets). Both agree only while every request in
            # the batch has the same beam width.
            #
            # Mixing widths would desynchronize the two: the sampler would read
            # a request's rows at the wrong offset, and `logits.view(batch,
            # beam_width_in, vocab)` succeeds for any shape whose element count
            # divides, so the result is silently wrong rather than an error.
            # Supporting mixed widths needs the forward path to emit a fixed
            # max_beam_width stride and the sampler offsets to match; until
            # then, fail loudly.
            beam_width = generation_requests[0].py_beam_width
            # Admission pins every request to max_beam_width, but a
            # variable-beam-width request narrows or widens per iteration, so
            # the widths can still diverge mid-batch. Compare the
            # *per-iteration* width: py_beam_width is fixed at admission and
            # would be identical across those requests. Dummy requests are
            # excluded -- they carry no user request and are built at their own
            # width (CUDA-graph padding at the engine width, attention-DP and
            # warmup dummies at width one), so they would otherwise trip this
            # on an ordinary padded batch.
            real_requests = [
                req for req in generation_requests if not req.is_dummy
            ]
            iter_widths = {
                req.get_beam_width_by_iter()
                for req in real_requests
            }
            if len(iter_widths) > 1:
                # NB: this aborts the whole batch, not just the offending
                # requests -- ModelEngine has no per-request failure channel,
                # and by this point the batch is already scheduled. Scoping the
                # failure needs the scheduler to group by beam width in the
                # first place, so that no such batch is formed; TRTLLM-14792.
                raise ValueError(
                    "Generation requests in one batch must all have the same "
                    f"beam width; got {sorted(iter_widths)}. Mixed beam widths "
                    "within a batch are not supported yet (TRTLLM-14792).")

            # Pre-extend constant-value lists to avoid per-request append
            # overhead (saves ~3 append calls per request).
            draft_lens.extend([0] * (_n_gen * beam_width))
            sequence_lengths.extend([1] * (_n_gen * beam_width))
            num_accepted_draft_tokens.extend([0] * (_n_gen * beam_width))

            for request in generation_requests:
                request_ids.append(request.py_request_id)
                is_promoted_context = (request.py_request_id
                                       in promoted_context_request_ids)
                if is_promoted_context:
                    input_ids.append(
                        request.get_tokens(0)[request.context_current_position])
                    past_seen_token_num = request.context_current_position
                    request_has_previous_tensor = False
                # The request has no previous tensor:
                # (1) new_tokens_device is None, which means overlap scheduler is disabled; or
                # (2) a dummy request; or
                # (3) the first step in the generation server of disaggregated serving.
                elif new_tokens_device is None or request.is_dummy or request.py_batch_idx is None:
                    # skip adding input_ids of CUDA graph dummy requests so that new_tokens_device
                    # can be aligned to the correct positions.
                    if not request.is_cuda_graph_dummy:
                        for beam in range(beam_width):
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
                    request_has_previous_tensor = False
                else:
                    # the request has previous tensor
                    # previous_batch_indices is per-request, not per-beam
                    previous_batch_indices.append(request.py_batch_idx)
                    past_seen_token_num = request.max_beam_num_tokens
                    request_has_previous_tensor = True

                position_id = past_seen_token_num
                if _has_cp_helix:
                    # We compute a global position_id because each helix rank has only a subset of
                    # tokens for a sequence.
                    position_id = request.total_input_len_cp + request.py_decoding_iter - 1
                    if request_has_previous_tensor:
                        # With the overlap scheduler this batch is prepared
                        # before the previous iteration's _update_requests has
                        # advanced py_decoding_iter, so the counter is one
                        # behind. Compensate exactly like the non-helix path
                        # above, which uses max_beam_num_tokens *without* the
                        # -1 in this case. Without this, the position repeats
                        # once (L, L, L+1, ...) and the new token's K is roped
                        # at the wrong position before being written to the KV
                        # cache, corrupting every later step.
                        # TODO: revisit for helix x speculative decoding -
                        # the base formula and this +1 both assume exactly
                        # one new token per step (draft-token modes are
                        # currently rejected under helix).
                        position_id += 1
                    if request.py_helix_is_inactive_rank:
                        past_seen_token_num = request.seqlen_this_rank_cp
                    else:
                        # Discount the token added to active rank in resource manager as it hasn't
                        # been previously seen.
                        past_seen_token_num = request.seqlen_this_rank_cp - 1

                    for beam in range(beam_width):
                        # Update helix-specific parameters.
                        helix_is_inactive_rank.append(
                            request.py_helix_is_inactive_rank)
                        helix_position_offsets.append(position_id)

                request.cached_tokens = past_seen_token_num
                for beam in range(beam_width):
                    position_ids.append(position_id)
                    num_cached_tokens_per_seq.append(
                        past_seen_token_num - request.py_num_compressed_tokens)
                    prompt_lengths.append(request.py_prompt_len)
                    gather_ids.append(len(position_ids) - 1)

                if _use_mrope:
                    mrope_position_delta = getattr(request,
                                                   "py_mrope_position_delta",
                                                   None)
                    if mrope_position_delta is None and request.py_multimodal_data:
                        mrope_config = request.py_multimodal_data.get(
                            'mrope_config') or {}
                        mrope_position_delta = mrope_config.get(
                            'mrope_position_deltas')
                        if mrope_position_delta is not None:
                            if mrope_position_delta.device.type == "cpu":
                                mrope_position_delta = maybe_pin_memory(
                                    mrope_position_delta).to(device='cuda',
                                                             dtype=torch.int32,
                                                             non_blocking=True)
                                mrope_config[
                                    'mrope_position_deltas'] = mrope_position_delta
                            request.py_mrope_position_delta = mrope_position_delta
                    if mrope_position_delta is not None:
                        has_gen_mrope_delta = True
                        # NOTE: Expanding position_ids to 3D tensor who is using mrope
                        gen_mrope_position_ids = (past_seen_token_num +
                                                  mrope_position_delta).expand(
                                                      3, 1, 1)
                        update_mrope_delta = (
                            request.py_seq_slot is not None
                            and not request.is_dummy
                            and getattr(request, "py_mrope_delta_cache_slot",
                                        None) != request.py_seq_slot)
                        delta_read_seq_slot = (mrope_dummy_seq_slot
                                               if request.is_dummy
                                               or request.py_seq_slot is None
                                               else request.py_seq_slot)
                        if update_mrope_delta:
                            multimodal_params = MultimodalParams(
                                multimodal_data={
                                    'mrope_config': {
                                        'mrope_position_deltas':
                                        mrope_position_delta
                                    }
                                })
                            mrope_delta_write_seq_slots.append(
                                request.py_seq_slot)
                            multimodal_params_list.append(multimodal_params)
                            request.py_mrope_delta_cache_slot = request.py_seq_slot
                        for beam in range(beam_width):
                            # Locate this beam's single token in the flat array.
                            token_start = len(position_ids) - beam_width + beam
                            mrope_position_ids.append(
                                (token_start, token_start + 1,
                                 gen_mrope_position_ids))
                            mrope_delta_read_seq_slots.append(
                                delta_read_seq_slot)
                    else:
                        # No MRoPE metadata for this request (text-only prompt
                        # on an MRoPE model): its delta is zero by construction,
                        # so read the reserved zero slot instead of skipping the
                        # append. The kernel indexes ``mrope_position_deltas``
                        # by *generation batch index*
                        # (decoderMaskedMultiheadAttentionTemplate.h), so a list
                        # that is sparse w.r.t. the generation batch would
                        # silently shift every later request onto another
                        # request's delta. No ``mrope_position_ids`` span is
                        # recorded: the broadcast scalar position is already
                        # this request's answer on all three axes.
                        for _ in range(beam_width):
                            mrope_delta_read_seq_slots.append(
                                mrope_dummy_seq_slot)
                # Equivalent to the original `is_generation_admission and
                # request.py_multimodal_data`. The batch-level flag is checked
                # first so non-multimodal models pay one LOAD_FAST per request
                # instead of LOAD_ATTR(py_multimodal_data) + LOAD_ATTR(py_batch_idx).
                if (_has_any_multimodal_request and request.py_multimodal_data
                        and request.py_batch_idx is None):
                    strip_mm_data_for_generation(request.py_multimodal_data)

                request.py_batch_idx = request.py_seq_slot
                append_cross_attention_state(request,
                                             project_encoder_output=False,
                                             repeat=beam_width)
                # Do not add a gen_request_seq_slot for CUDA graph dummy requests
                # to prevent access errors due to None values
                if not request.is_cuda_graph_dummy:
                    gen_request_seq_slots.append(request.py_seq_slot)

        if _use_mrope and not has_gen_mrope_delta:
            # Every generation request in this batch resolved to the zero slot,
            # so the gathered deltas would be an all-zero vector -- identical to
            # passing no deltas at all. Dropping the list keeps the steady-state
            # generation fast path (which requires the mrope lists to be empty)
            # reachable for text-only batches on MRoPE models.
            mrope_delta_read_seq_slots.clear()

        previous_batch_len = len(previous_batch_indices)
        # Device-window prologue precondition: it gathers this step's inputs
        # by (slot, offset) through previous_batch_indices_cuda, which only
        # covers requests that carried a previous device tensor and only in
        # batch order when EVERY real generation request did. Recorded here,
        # consumed by _apply_device_window_prologue.
        self._dspark_prev_covers_batch = (
            previous_batch_len > 0 and previous_batch_len
            == len(extend_requests) - len(extend_dummy_requests))

        def previous_seq_slots_device():
            previous_batch_indices_host = torch.tensor(
                previous_batch_indices,
                dtype=torch.int,
                pin_memory=prefer_pinned())
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
                                     pin_memory=prefer_pinned())
            self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

            # Update input_ids_cuda with new tokens from new_tensors_device (draft model only)
            if self.is_draft_model and num_accepted_tokens_device is not None:
                # For context requests: replace the last token with new_tensors_device[0, seq_slot, 0]
                if len(context_input_ids_positions) > 0:
                    # Build tensors on CPU first, then copy to GPU to avoid implicit sync
                    num_ctx_positions = len(context_input_ids_positions)
                    ctx_token_indices_cpu = torch.tensor(
                        [
                            last_token_idx for _, last_token_idx, _ in
                            context_input_ids_positions
                        ],
                        dtype=torch.long,
                        pin_memory=prefer_pinned())
                    ctx_seq_slots_cpu = torch.tensor([
                        seq_slot
                        for _, _, seq_slot in context_input_ids_positions
                    ],
                                                     dtype=torch.long,
                                                     pin_memory=prefer_pinned())
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
                                                  pin_memory=prefer_pinned())
                    seq_slots_tensor_cpu = torch.tensor(
                        all_seq_slots,
                        dtype=torch.long,
                        pin_memory=prefer_pinned())

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
                                        pin_memory=prefer_pinned())
            self.draft_tokens_cuda[:len(draft_tokens)].copy_(draft_tokens,
                                                             non_blocking=True)
        if self.is_spec_decode and len(num_accepted_draft_tokens) > 0:
            num_accepted_draft_tokens = torch.tensor(num_accepted_draft_tokens,
                                                     dtype=torch.int,
                                                     pin_memory=prefer_pinned())
            self.num_accepted_draft_tokens_cuda[:len(
                num_accepted_draft_tokens)].copy_(num_accepted_draft_tokens,
                                                  non_blocking=True)

            # Update num_accepted_draft_tokens_cuda for first_draft_requests directly from num_accepted_tokens_device (draft model only)
            if self.is_draft_model and len(first_draft_seq_slots) > 0:
                # Build tensors on CPU first, then copy to GPU to avoid implicit sync
                num_first_draft = len(first_draft_seq_slots)
                first_draft_seq_slots_cpu = torch.tensor(
                    first_draft_seq_slots,
                    dtype=torch.int,
                    pin_memory=prefer_pinned())
                first_draft_indices_cpu = torch.tensor(
                    first_draft_request_indices,
                    dtype=torch.int,
                    pin_memory=prefer_pinned())

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
            runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(
                self.runtime_draft_len)
            runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1

            if previous_batch_len > 0:
                previous_slots = previous_seq_slots_device()
                # Ragged windows break the fixed strided gathers below; use
                # explicit (row, col) index lists, keeping the cheaper strided
                # path for uniform batches.
                is_ragged_gen = (
                    previous_batch_tokens_per_request is not None
                    and any(tokens != runtime_tokens_per_gen_step
                            for tokens in previous_batch_tokens_per_request))
                # previous input ids
                previous_batch_tokens = len(previous_pos_indices)
                if is_ragged_gen:
                    token_rows, token_cols = self._ragged_gather_indices(
                        previous_batch_indices,
                        previous_batch_tokens_per_request)
                    new_tokens = new_tokens_device.transpose(
                        0, 1)[token_rows, token_cols].flatten()
                else:
                    new_tokens = new_tokens_device.transpose(0, 1)[
                        previous_slots, :runtime_tokens_per_gen_step].flatten()
                self.input_ids_cuda[num_tokens:num_tokens +
                                    previous_batch_tokens].copy_(
                                        new_tokens, non_blocking=True)

                # previous draft tokens
                if is_ragged_gen:
                    previous_batch_draft_tokens = (previous_batch_tokens -
                                                   previous_batch_len)
                    if previous_batch_draft_tokens > 0:
                        draft_rows, draft_cols = self._ragged_gather_indices(
                            previous_batch_indices,
                            [t - 1 for t in previous_batch_tokens_per_request])
                        self.draft_tokens_cuda[
                            num_draft_tokens:num_draft_tokens +
                            previous_batch_draft_tokens].copy_(
                                next_draft_tokens_device[draft_rows,
                                                         draft_cols].flatten(),
                                non_blocking=True)
                else:
                    previous_batch_draft_tokens = (
                        previous_batch_len * runtime_draft_token_buffer_width)
                    if runtime_draft_token_buffer_width > 0:
                        self.draft_tokens_cuda[
                            num_draft_tokens:num_draft_tokens +
                            previous_batch_draft_tokens].copy_(
                                next_draft_tokens_device[
                                    previous_slots, :
                                    runtime_draft_token_buffer_width].flatten(),
                                non_blocking=True)
                # prepare data for the preprocess inputs.
                # kv_len_offsets pairs with the num_cached_tokens_per_seq the
                # host wrote above (past_seen + tokens_this_step): the two
                # cancel to past_seen + accepted, so both sides must use the
                # SAME per-request token count or the KV length is off by
                # exactly their difference.
                if is_ragged_gen:
                    tokens_per_previous_request = torch.tensor(
                        previous_batch_tokens_per_request,
                        dtype=torch.long,
                        pin_memory=prefer_pinned()).to(
                            new_tokens_lens_device.device, non_blocking=True)
                    previous_kv_len_offsets = (
                        new_tokens_lens_device[previous_slots] -
                        tokens_per_previous_request)
                    # A slot the sampler has not yet written yields a stale
                    # count, and the captured kv_lens correction then walks the
                    # KV append out of bounds (the ragged IMA); clamp to the
                    # physically possible range.
                    previous_kv_len_offsets = previous_kv_len_offsets.clamp_(
                        min=-int(self.runtime_draft_len + 1),
                        max=int(self.runtime_draft_len + 1))
                else:
                    kv_len_offsets_device = (new_tokens_lens_device -
                                             runtime_tokens_per_gen_step)
                    previous_kv_len_offsets = kv_len_offsets_device[
                        previous_slots]
                previous_pos_indices_host = torch.tensor(
                    previous_pos_indices,
                    dtype=torch.int,
                    pin_memory=prefer_pinned())
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
                if is_ragged_gen:
                    # previous_pos_id_offsets_cuda is indexed by flat
                    # generation-token position, which is no longer
                    # request_index * runtime_tokens_per_gen_step. Use the
                    # offset accumulated while walking the batch.
                    pos_offsets_start = previous_batch_token_start
                else:
                    pos_offsets_start = (
                        num_extend_reqeust_wo_dummy -
                        previous_batch_len) * runtime_tokens_per_gen_step
                self.previous_pos_id_offsets_cuda[
                    pos_offsets_start:pos_offsets_start +
                    previous_batch_tokens].copy_(
                        new_tokens_lens_device[self.previous_pos_indices_cuda[
                            0:previous_batch_tokens]],
                        non_blocking=True)

                self.previous_kv_lens_offsets_cuda[
                    num_extend_reqeust_wo_dummy -
                    previous_batch_len:num_extend_reqeust_wo_dummy].copy_(
                        previous_kv_len_offsets, non_blocking=True)

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

        position_ids = self._apply_position_id_offset(position_ids)
        # Use the (3,1,N) MRoPE layout whenever the model declares MRoPE, even
        # for text-only batches: keeping position_ids rank-consistent between
        # warmup and serving keeps torch.compile guards stable, so piecewise
        # CUDA graphs captured at warmup remain usable at runtime.
        if self.use_mrope:
            # Mixed batches may have only some requests with multimodal MRoPE
            # data. Seed the full (3,1,N) buffer from scalar position_ids
            # (text-only tokens get the same value on all 3 axes), then
            # overwrite only the multimodal spans with their real MRoPE coords.
            position_ids_tensor = torch.tensor(position_ids,
                                               dtype=torch.int,
                                               pin_memory=prefer_pinned())
            self.position_ids_cuda[:total_num_tokens].copy_(position_ids_tensor,
                                                            non_blocking=True)
            # Broadcast [N] to [3,1,N]: default for text-only tokens.
            self.mrope_position_ids_cuda[:, :, :total_num_tokens].copy_(
                self.position_ids_cuda[:total_num_tokens].view(1, 1, -1).expand(
                    3, 1, -1),
                non_blocking=True)
            # Overwrite multimodal spans with per-axis MRoPE positions.
            for start_idx, end_idx, segment in mrope_position_ids:
                if segment.ndim != 3:
                    raise RuntimeError(
                        f"Expected 3D mrope_position_ids, got shape {tuple(segment.shape)}"
                    )
                if segment.shape[0] != 3 and segment.shape[-1] == 3:
                    logger.warning(
                        "Transposing unexpected mrope_position_ids shape from "
                        f"{tuple(segment.shape)}")
                    segment = segment.transpose(0, 2).contiguous()
                if segment.shape[:2] != (3, 1):
                    raise RuntimeError(
                        f"Unexpected mrope_position_ids shape {tuple(segment.shape)} for span {start_idx}:{end_idx}"
                    )
                segment = segment.contiguous()
                if segment.device.type == "cpu":
                    segment = maybe_pin_memory(segment)
                self.mrope_position_ids_cuda[:, :, start_idx:end_idx].copy_(
                    segment[:, :, :end_idx - start_idx], non_blocking=True)
            final_position_ids = self.mrope_position_ids_cuda[:, :, :
                                                              total_num_tokens]
        else:
            position_ids = torch.tensor(position_ids,
                                        dtype=torch.int,
                                        pin_memory=prefer_pinned())
            self.position_ids_cuda[:total_num_tokens].copy_(position_ids,
                                                            non_blocking=True)
            final_position_ids = self.position_ids_cuda[:
                                                        total_num_tokens].unsqueeze(
                                                            0)

        if self.enable_spec_decode:
            self.gather_ids_cuda[:len(gather_ids)].copy_(torch.tensor(
                gather_ids, dtype=torch.int, pin_memory=prefer_pinned()),
                                                         non_blocking=True)

            # Update gather_ids for first_draft_requests on GPU (draft model only)
            if self.is_draft_model and len(first_draft_seq_slots) > 0:
                # Build tensors on CPU first, then copy to GPU to avoid implicit sync
                num_first_draft = len(first_draft_seq_slots)
                first_draft_seq_slots_cpu = torch.tensor(
                    first_draft_seq_slots,
                    dtype=torch.int,
                    pin_memory=prefer_pinned())
                first_draft_indices_cpu = torch.tensor(
                    first_draft_request_indices,
                    dtype=torch.int,
                    pin_memory=prefer_pinned())

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

        # Under ragged verification the per-request lengths move between
        # replays of the same graph (the key pins the row count and token
        # total, not the split), so seq_lens must be refreshed or
        # attn_metadata.num_tokens disagrees with the input_ids width. Publish
        # BEFORE the refresh decision below: it asks the metadata whether this
        # step is ragged, and asking before this step's windows are on it
        # reads the previous step's answer.
        self._publish_gen_token_layout(attn_metadata,
                                       scheduled_requests.generation_requests)

        refresh_seq_lens = not attn_metadata.is_cuda_graph
        if (not refresh_seq_lens
                and getattr(attn_metadata, "is_ragged_verify", False)
                and attn_metadata.seq_lens is not None
                and len(sequence_lengths) == attn_metadata.seq_lens.shape[0]):
            refresh_seq_lens = True
        if refresh_seq_lens:
            # Assumes seq lens do not change between CUDA graph invocations. This applies
            # to draft sequences too. This means that all draft sequences must be padded.
            attn_metadata.seq_lens = torch.tensor(
                sequence_lengths,
                dtype=torch.int,
                pin_memory=prefer_pinned(),
            )

        num_generation_requests = len(gen_request_seq_slots)
        # Cache indirection is only used for beam search on generation requests
        if self.use_beam_search and num_generation_requests > 0:
            if cache_indirection_buffer is not None:
                #Copy cache indirection to local buffer with offsets changing:  seq_slots[i] -> i
                # Convert to GPU tensor to avoid implicit sync
                gen_request_seq_slots_tensor = torch.tensor(
                    gen_request_seq_slots,
                    dtype=torch.long,
                    pin_memory=prefer_pinned()).to(device='cuda',
                                                   non_blocking=True)
                self.cache_indirection_attention[:num_generation_requests].copy_(
                    cache_indirection_buffer[gen_request_seq_slots_tensor])
            if cache_indirection_buffer is not None or self.is_warmup:
                attn_metadata.beam_width = self.max_beam_width
        else:
            attn_metadata.beam_width = 1

        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = scheduled_requests.num_context_requests
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
            num_extra_kv_tokens=get_num_extra_kv_tokens(spec_config),
            use_full_generation_page_table=(
                self._should_use_full_generation_page_table(
                    spec_config, attn_metadata)))
        attn_metadata.kv_cache_manager = kv_cache_manager

        if hasattr(self.model.model_config.pretrained_config, 'chunk_size'):
            attn_metadata.mamba_chunk_size = self.model.model_config.pretrained_config.chunk_size
        # Some sparse backends (RocketKV) clamp
        # kv_cache_params.num_cached_tokens_per_seq in place during prepare(),
        # and KVCacheParams holds the list by reference. Snapshot the true
        # pre-prepare counts so the steady-gen recording below stores values
        # that the per-step prepare() can re-clamp from scratch.
        num_cached_tokens_snapshot = list(num_cached_tokens_per_seq)
        # The gen-token layout was published above, before the seq_lens refresh
        # that depends on it. prepare() is its other consumer: it decides the DSA
        # expanded-buffer layout, strides the expansions by this step's
        # per-request token count, builds the per-row causal extents, and derives
        # DeepSeek-V4's per-request compressor token counts. The rest of the
        # layout is attached later, once the spec metadata exists.
        attn_metadata.prepare()
        cross_attention_inputs = (self._prepare_enc_dec_cross_attn_inputs(
            cross_encoder_hidden_states,
            cross_encoder_seq_lens,
            cross_encoder_cached_tokens_per_seq,
            attn_metadata,
            resource_manager,
        ) if is_enc_dec else {})

        peft_cache_manager = resource_manager and resource_manager.get_resource_manager(
            ResourceManagerType.PEFT_CACHE_MANAGER)
        lora_params = self._get_lora_params_from_requests(
            scheduled_requests,
            attn_metadata,
            peft_cache_manager,
            maybe_graph,
            use_lora_graph=use_lora_graph)

        spec_all_rank_counts = None
        if spec_metadata is not None and self.enable_attention_dp:
            (attn_all_rank_num_tokens, spec_all_rank_counts
             ) = self._get_all_rank_num_tokens_and_spec_counts(
                 attn_metadata, (total_num_tokens, len(sequence_lengths),
                                 len(scheduled_requests.generation_requests)))
        else:
            attn_all_rank_num_tokens = self._get_all_rank_num_tokens(
                attn_metadata)
        (padded_num_tokens, can_run_prefill_cuda_graph,
         attn_all_rank_num_tokens) = self._get_padding_params(
             total_num_tokens, num_ctx_requests, attn_all_rank_num_tokens)
        set_per_request_prefill_cuda_graph_flag(can_run_prefill_cuda_graph)
        attn_metadata.padded_num_tokens = padded_num_tokens if padded_num_tokens != total_num_tokens else None

        virtual_num_tokens = total_num_tokens
        if attn_metadata.padded_num_tokens is not None:
            self.input_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens
            # Match the rank of the unpadded branch: MRoPE models always use
            # the (3,1,N) layout (see the seeding block above), so the padded
            # view must stay 3D as well to keep torch.compile guards stable.
            if self.use_mrope:
                # Zero-fill padding on dim 2 (token dim) of (3,1,N) buffer.
                self.mrope_position_ids_cuda[:, :, total_num_tokens:
                                             padded_num_tokens].fill_(0)
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
            'resource_manager': resource_manager,
        }
        inputs.update(cross_attention_inputs)

        if self.use_mrope:
            if mrope_delta_write_seq_slots:
                delta_write_seq_slots = torch.tensor(
                    mrope_delta_write_seq_slots,
                    dtype=torch.long,
                    pin_memory=prefer_pinned())
                inputs[
                    'mrope_delta_write_seq_slots'] = delta_write_seq_slots.to(
                        device='cuda', non_blocking=True)

            if mrope_delta_read_seq_slots:
                delta_read_seq_slots = torch.tensor(mrope_delta_read_seq_slots,
                                                    dtype=torch.long,
                                                    pin_memory=prefer_pinned())
                inputs['mrope_delta_read_seq_slots'] = delta_read_seq_slots.to(
                    device='cuda', non_blocking=True)

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
            if context_prompt_lookahead is not None:
                spec_metadata.populate_context_prompt_lookahead(
                    context_prompt_lookahead)
            self._attach_ragged_verify_layout(
                spec_metadata, attn_metadata,
                scheduled_requests.generation_requests)
            if isinstance(spec_metadata, Eagle3SpecMetadata):
                spec_metadata.request_accepted_path = request_accepted_path
            # No-op for non 1-model
            spec_metadata.populate_sampling_params_for_one_model(
                scheduled_requests.all_requests())
            spec_metadata.prepare()
            # One-model rejection: one-hot the stale draft_probs rows of gen
            # requests that produced no draft tokens this step, so the (possibly
            # captured) rejection kernel reads a legal placeholder distribution.
            spec_metadata.write_padding_onehot_draft_probs(
                padding_gen_slots, self.runtime_draft_len)
            inputs['spec_metadata'] = spec_metadata

            if self.enable_attention_dp:
                self._set_spec_metadata_all_rank_num_tokens(
                    spec_metadata, *spec_all_rank_counts)

        if mm_token_indices is not None:
            self._ship_multimodal_indices(
                inputs,
                mm_token_indices_cpu=mm_token_indices,
                text_token_indices_cpu=text_token_indices_ctx,
                num_ctx_tokens=num_ctx_tokens,
                total_num_tokens=total_num_tokens,
            )

        num_generation_tokens = len(generation_requests) + len(
            extend_requests) + sum(draft_lens) + len(first_draft_requests)
        self.iter_states['num_ctx_requests'] = num_ctx_requests
        self.iter_states['num_ctx_tokens'] = num_ctx_tokens
        self.iter_states['num_generation_tokens'] = num_generation_tokens
        # Count the already-cached prefix for the sequences scheduled this iteration.
        self.iter_states['cached_kv_tokens'] = sum(num_cached_tokens_per_seq)

        if not self.is_warmup:
            self.previous_request_ids = all_gen_request_ids
            self.previous_verify_lens = [
                getattr(request, "py_verify_len", None)
                for request in scheduled_requests.generation_requests
            ]
            self.has_previous_device_draft = next_draft_tokens_device is not None

            # Record the steady-state generation cache when this pass handled
            # purely non-dummy generation requests that all carried a previous
            # overlap-scheduler tensor (previous_batch_len == _n_gen implies
            # every request took that branch and none appended input_ids).
            # While the batch composition holds, the next passes only need to
            # advance positions by one and refresh per-step metadata.
            # MRoPE models are supported only for batches with no actual mrope
            # work (text-only requests, empty mrope lists below): the full
            # pass routes use_mrope models through the (3,1,N)
            # mrope_position_ids_cuda layout even then (to keep torch.compile
            # guards stable), with all three axes equal to the scalar
            # positions, so the fast path advances that buffer in place and
            # returns the same layout (see _apply_steady_gen_fast_prepare).
            if (self.spec_config is None and not self.is_draft_model
                    and spec_metadata is None and new_tokens_device is not None
                    and self.guided_decoder is None
                    and not self.enable_attention_dp and not mrope_position_ids
                    and not mrope_delta_write_seq_slots
                    and not mrope_delta_read_seq_slots
                    and not self.use_beam_search and self.max_beam_width == 1
                    and not is_enc_dec and not _has_cp_helix
                    and num_ctx_requests == 0 and not extend_requests
                    and not first_draft_requests and _n_gen > 0
                    and previous_batch_len == _n_gen and num_tokens == 0
                    and not _has_any_multimodal_request
                    and not multimodal_params_list and not lora_params
                    and attn_metadata.padded_num_tokens is None
                    and self._get_position_id_offset() == 0
                    and not getattr(kv_cache_manager,
                                    "kv_compression_manages_history", False)):
                self._steady_gen_positions_pinned[:_n_gen].copy_(
                    torch.as_tensor(num_cached_tokens_snapshot,
                                    dtype=torch.int))
                self._steady_gen_cache = {
                    'num_requests':
                    _n_gen,
                    'request_ids':
                    all_gen_request_ids,
                    'prompt_lens':
                    prompt_lengths,
                    'seq_lens_ones':
                    maybe_pin_memory(torch.ones(_n_gen, dtype=torch.int)),
                    'use_mrope':
                    _use_mrope,
                }

        return inputs, self.gather_ids_cuda[:len(
            gather_ids)] if self.enable_spec_decode else None

    def _prepare_tp_inputs_no_cache(
            self,
            scheduled_requests: ScheduledRequests,
            attn_metadata: AttentionMetadata,
            spec_metadata: Optional[SpecMetadata] = None,
            resource_manager: Optional[ResourceManager] = None):
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
            # Start offset of this request's tokens within the flattened
            # input_ids (see _prepare_tp_inputs for rationale).
            context_start_idx = len(input_ids)
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
                    multimodal_input=_build_request_multimodal_input(
                        request, self._mm_encoder_cache_enabled),
                    multimodal_data=request.py_multimodal_data,
                    mm_item_order=getattr(request, "py_mm_item_order", None),
                    input_ids_start_offset=context_start_idx)
                multimodal_params.to_device("multimodal_data",
                                            "cuda",
                                            pin_memory=prefer_pinned())
                multimodal_params_list.append(multimodal_params)

            request.py_batch_idx = request.py_seq_slot

        num_tokens = len(input_ids)
        assert num_tokens <= self.max_num_tokens, (
            "num_tokens should be less than or equal to max_num_tokens")
        # Compute MM/text token indices on CPU input_ids so that
        # fuse_input_embeds can skip its torch.where host sync. Must run before
        # the input_ids list is rebound to a tensor below. Skipped when
        # ``self.model`` is a vision encoder (no ``config.vocab_size`` to filter
        # against, and its forward doesn't consume the indices anyway); this
        # is a structural check on the model rather than a flag lookup, so it
        # naturally extends to any future "LLM-less" engine setup.
        _model_config = getattr(self.model, "config", None)
        if (len(multimodal_params_list) > 0
                and getattr(_model_config, "vocab_size", None) is not None):
            text_token_indices_cpu, mm_token_indices_cpu = \
                self._prepare_multimodal_indices(input_ids)
        else:
            text_token_indices_cpu = None
            mm_token_indices_cpu = None
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.int,
                                 pin_memory=prefer_pinned())
        self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        position_ids = self._apply_position_id_offset(position_ids)
        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int,
                                    pin_memory=prefer_pinned())
        self.position_ids_cuda[:num_tokens].copy_(position_ids,
                                                  non_blocking=True)
        if self.enable_spec_decode:
            self.gather_ids_cuda[:len(gather_ids)].copy_(torch.tensor(
                gather_ids, dtype=torch.int, pin_memory=prefer_pinned()),
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
                pin_memory=prefer_pinned(),
            )

        attn_metadata.num_contexts = scheduled_requests.num_context_requests

        attn_all_rank_num_tokens = self._get_all_rank_num_tokens(attn_metadata)
        padded_num_tokens, can_run_prefill_cuda_graph, attn_all_rank_num_tokens = self._get_padding_params(
            num_tokens, attn_metadata.num_contexts, attn_all_rank_num_tokens)
        set_per_request_prefill_cuda_graph_flag(can_run_prefill_cuda_graph)
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
            "multimodal_params": multimodal_params_list,
            'resource_manager': resource_manager,
        }

        if mm_token_indices_cpu is not None:
            # No extend/draft tokens in the no-cache path, so num_tokens covers
            # the full range and the helper's arange/cat branch is skipped.
            self._ship_multimodal_indices(
                inputs,
                mm_token_indices_cpu=mm_token_indices_cpu,
                text_token_indices_cpu=text_token_indices_cpu,
                num_ctx_tokens=num_tokens,
                total_num_tokens=num_tokens,
            )

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
                all_rank_num_tokens = self.dist.tp_cp_allgather_int64([
                    attn_metadata.num_tokens, spec_metadata.num_tokens,
                    len(sequence_lengths), spec_metadata.num_generations
                ]).tolist()
                attn_metadata.all_rank_num_tokens = [
                    item[0] for item in all_rank_num_tokens
                ]
                self._set_spec_metadata_all_rank_num_tokens(
                    spec_metadata, [item[1] for item in all_rank_num_tokens],
                    [item[2] for item in all_rank_num_tokens],
                    [item[3] for item in all_rank_num_tokens])
            else:
                all_rank_num_tokens = self.dist.tp_cp_allgather_int64(
                    [attn_metadata.num_tokens])[:, 0].tolist()
                attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        return inputs, None

    def _get_lora_params_from_requests(
            self,
            scheduled_requests: ScheduledRequests,
            attn_metadata: AttentionMetadata,
            peft_cache_manager: Optional[PeftCacheManager] = None,
            maybe_graph: bool = False,
            use_lora_graph: bool = False):
        '''
        Get LoRA parameters from scheduled requests.

        Uses CUDA Graph compatible mode in decode only batch, otherwise falls back to eager mode.

        Returns:
            Dictionary containing LoRA parameters, or None if no LoRA requests
        '''
        use_cuda_graph_mode = self.cuda_graph_lora_manager is not None and maybe_graph

        if use_cuda_graph_mode:
            if not use_lora_graph:
                self.cuda_graph_lora_manager.prepare_base_only_batch(
                    peft_cache_manager)
                return None
            # For spec decode verification (non-extend_ctx), each sequence has
            # runtime_draft_len + 1 tokens in the forward pass.
            tokens_per_seq = 1
            if (self.enable_spec_decode and self.runtime_draft_len > 0
                    and self.spec_config.is_linear_tree
                    and not self.spec_config.spec_dec_mode.extend_ctx(
                        self.attn_backend)):
                tokens_per_seq = self.runtime_draft_len + 1
            return self.cuda_graph_lora_manager.prepare_cuda_graph_lora_params(
                scheduled_requests, attn_metadata, peft_cache_manager,
                tokens_per_seq)
        else:
            if self.cuda_graph_lora_manager is not None:
                self.cuda_graph_lora_manager.adapter_slot_manager.remove_evicted_slots_in_cpp(
                    peft_cache_manager)
            peft_table = peft_cache_manager.get_and_reset_batch_peft_table(
            ) if peft_cache_manager is not None else None
            lora_params = peft_table and self._get_eager_lora_params_from_requests(
                scheduled_requests, attn_metadata, peft_table)
            if lora_params:
                lora_params["data_type"] = peft_cache_manager.data_type
            return lora_params

    def _get_eager_lora_params_from_requests(
            self, scheduled_requests: ScheduledRequests,
            attn_metadata: AttentionMetadata,
            peft_table: Dict[int, list[TaskLayerModuleConfig]]):
        '''
        Eager mode LoRA parameter preparation logic.

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

        request_list = scheduled_requests.all_requests()

        # trace all requests to get the union set of the lora params
        for request in request_list:
            if request.lora_task_id is None:
                continue

            layer_module_configs = peft_table[request.lora_task_id]

            for module in layer_module_configs:
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
            if request.lora_task_id is None:
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
            host_request_types = attn_metadata.host_request_types
            prompt_lens_cpu = attn_metadata.prompt_lens_cpu
            num_seqs = attn_metadata.num_seqs
            num_contexts = attn_metadata.num_contexts
            num_generations = attn_metadata.num_generations

            # During spec decode verification (non-extend_ctx mode), each
            # generation request processes (runtime_draft_len + 1) tokens at
            # once. The LoRA op's C++ kernel only advances 1 token per
            # kGENERATION request, so we re-label generation requests as
            # kCONTEXT and set prompt_lens_cpu to the actual per-request token
            # count so the kernel correctly expands LoRA weights for all tokens.
            if (self.enable_spec_decode and self.runtime_draft_len > 0
                    and self.spec_config.is_linear_tree
                    and not self.spec_config.spec_dec_mode.extend_ctx(
                        self.attn_backend) and num_generations > 0):
                tokens_per_req = self.runtime_draft_len + 1
                host_request_types = host_request_types.clone()
                host_request_types[num_contexts:num_seqs].fill_(0)  # kCONTEXT
                prompt_lens_cpu = prompt_lens_cpu.clone()
                prompt_lens_cpu[num_contexts:num_seqs].fill_(tokens_per_req)

            lora_params['host_request_types'] = host_request_types
            lora_params['prompt_lens_cpu'] = prompt_lens_cpu
            lora_params['num_seqs'] = num_seqs

        return lora_params

    @nvtx_range("_prepare_inputs")
    def _prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2],
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        new_tensors_device: Optional[SampleStateTensors] = None,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
        num_accepted_tokens_device: Optional[torch.Tensor] = None,
        req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None,
        resource_manager: Optional[ResourceManager] = None,
        maybe_graph: bool = False,
        promoted_context_request_ids: frozenset[int] = frozenset(),
        use_lora_graph: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        set_per_request_prefill_cuda_graph_flag(False)
        if self.mapping is not None and 'cp_type' in self.mapping.cp_config:
            cp_type = self.mapping.cp_config['cp_type']
            if cp_type in (CpType.HELIX, CpType.ULYSSES):
                # Take the usual route of _prepare_tp_inputs.
                pass
            else:
                raise NotImplementedError(
                    f"Unsupported cp_type {getattr(cp_type, 'name', cp_type)}.")

        # Initialize SA state for new requests (MTP+SA, EAGLE3+SA, PARD+SA, etc.)
        has_sa_enhancer = (self.spec_config is not None and getattr(
            self.spec_config, 'sa_config', None) is not None)
        if has_sa_enhancer and resource_manager is not None and self.mapping.is_last_pp_rank(
        ):
            from tensorrt_llm._torch.speculative.suffix_automaton import \
                SuffixAutomatonManager
            spec_rm = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            sa_manager = None
            if spec_rm is not None:
                if isinstance(spec_rm, SuffixAutomatonManager):
                    sa_manager = spec_rm
                else:
                    sa_manager = getattr(spec_rm, 'sa_manager', None)
            if sa_manager is not None:
                for request in scheduled_requests.all_requests():
                    if request.py_request_id not in sa_manager._initialized_requests:
                        sa_manager.add_request(request.py_request_id,
                                               request.get_tokens(0))
                        sa_manager._initialized_requests.add(
                            request.py_request_id)

        return self._prepare_tp_inputs(scheduled_requests,
                                       kv_cache_manager,
                                       attn_metadata,
                                       spec_metadata,
                                       new_tensors_device,
                                       cache_indirection_buffer,
                                       num_accepted_tokens_device,
                                       req_id_to_old_request,
                                       resource_manager,
                                       maybe_graph,
                                       promoted_context_request_ids,
                                       use_lora_graph=use_lora_graph)

    def _prepare_encoder_inputs(
        self,
        inputs: Dict[str, Any],
        attn_metadata: Optional[Any] = None,
        padded_num_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Prepare model-ready inputs dict for encode-only path.

        - Eager / graph-miss (`attn_metadata is None`): tensorize input_ids
          and position_ids here, copy them into the model engine's CUDA
          buffers, and run the full attention metadata setter chain.

        - CUDA graph hit (`attn_metadata` passed in): minimal CPU work.
          input_ids / position_ids stay as their raw input forms (Python
          list / None / tensor) and are written directly into the runner's
          pinned static CPU buffers. The attention
          metadata updates the runner-bound seq_lens buffer; the H2D copy to
          `_seq_lens_cuda` and inputs are captured inside the graph itself.
        """
        input_ids = inputs['input_ids']
        seq_lens = inputs['seq_lens']  # Only seq_lens includes padding
        position_ids = inputs.get('position_ids')
        multi_item_part_lens = inputs.get('multi_item_part_lens')
        actual_num_tokens = len(input_ids)
        batch_size = len(seq_lens)

        # Eager / encoder graph-miss path. Tensorize inputs and run the full
        # setter chain.
        if attn_metadata is None:
            input_ids_t = torch.tensor(input_ids,
                                       dtype=torch.int,
                                       pin_memory=prefer_pinned())
            if position_ids is None:
                if multi_item_part_lens is not None:
                    if len(multi_item_part_lens) != len(seq_lens):
                        raise ValueError(
                            "\"multi_item_part_lens\" must either be provided for all prompts or for none"
                        )

                    # Scoring items have overlapping position IDs. Position IDs of delimiters
                    # are irrelevant.
                    starts_cuda = torch.tensor(
                        [
                            start
                            for req_multi_item_part_lens in multi_item_part_lens
                            for start in [0] + [req_multi_item_part_lens[0]] *
                            (len(req_multi_item_part_lens) - 1)
                        ],
                        pin_memory=prefer_pinned(),
                        dtype=torch.int32,
                    ).to(device=self.position_ids_cuda.device,
                         non_blocking=True)
                    ends_cuda = torch.tensor(
                        [
                            end + 1
                            for req_multi_item_part_lens in multi_item_part_lens
                            for end in [req_multi_item_part_lens[0]] + [
                                req_multi_item_part_lens[0] + item_len
                                for item_len in req_multi_item_part_lens[1:]
                            ]
                        ],
                        pin_memory=prefer_pinned(),
                        dtype=torch.int32,
                    ).to(device=self.position_ids_cuda.device,
                         non_blocking=True)
                    position_ids_t = torch_multi_arange(
                        starts=starts_cuda,
                        ends=ends_cuda,
                        output_length=input_ids_t.numel(),
                    )
                else:
                    # Auto-generate packed position IDs: [0..n1-1, 0..n2-1, ...]
                    position_ids_t = torch.cat([
                        torch.arange(s, dtype=torch.int) for s in seq_lens
                    ])[:actual_num_tokens]
                    position_ids_t = maybe_pin_memory(position_ids_t)
            elif not isinstance(position_ids, torch.Tensor):
                position_ids_t = torch.tensor(position_ids,
                                              dtype=torch.int,
                                              pin_memory=prefer_pinned())
            else:
                position_ids_t = position_ids

            attn_metadata = self._set_up_attn_metadata(kv_cache_manager=None)
            attn_metadata.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
            attn_metadata.num_contexts = batch_size
            attn_metadata.max_seq_len = self.max_seq_len
            attn_metadata.request_ids = list(range(batch_size))
            if multi_item_part_lens is not None and not self.attn_backend.support_multi_item_scoring(
            ):
                raise ValueError(
                    "The selected attention backend does not support multi-item scoring."
                )
            attn_metadata.multi_item_part_lens = multi_item_part_lens
            if hasattr(attn_metadata, 'prepare_encoder_only'):
                attn_metadata.prepare_encoder_only()
            else:
                attn_metadata.prepare()

            self.input_ids_cuda[:actual_num_tokens].copy_(input_ids_t,
                                                          non_blocking=True)
            self.position_ids_cuda[:actual_num_tokens].copy_(position_ids_t,
                                                             non_blocking=True)
            return {
                **inputs,
                'attn_metadata':
                attn_metadata,
                'input_ids':
                self.input_ids_cuda[:actual_num_tokens],
                'position_ids':
                self.position_ids_cuda[:actual_num_tokens].unsqueeze(0),
            }

        # CUDA graph hit path.
        assert self.encoder_cuda_graph_runner.enabled, "Encoder CUDA graph runner is not enabled"

        # NB: The multi-item scoring arguments lack '_buf' counterparts (cf., e.g.,
        #     https://github.com/flashinfer-ai/flashinfer/blob/2aa1d49cf140d73ccdd3761051c5f2944406cb83/flashinfer/prefill.py#L1622 ),
        #     which are typically used to support CUDA graphs in FlashInfer.
        assert multi_item_part_lens is None, "multi-item scoring with CUDA graph not implemented"

        attn_metadata.prepare_encoder_cuda_graph_replay(seq_lens,
                                                        padded_num_tokens)

        return {
            **inputs,
            'attn_metadata': attn_metadata,
            'input_ids': input_ids,
            'position_ids': position_ids,
        }

    def _create_encoder_warmup_inputs(
            self, batch_size: int, num_tokens: int,
            max_seq_len: int) -> Optional[Dict[str, Any]]:
        """Synthesize an inputs dict that will bucket exactly at
        (batch_size, num_tokens, max_seq_len).

        Returns None for infeasible combinations (e.g., batch_size <= 0).
        """
        lengths = (
            self.encoder_cuda_graph_runner.build_capture_sequence_lengths(
                batch_size, num_tokens, max_seq_len))
        if lengths is None:
            return None

        inputs: Dict[str, Any] = {
            'input_ids': [0] * sum(lengths),
            'seq_lens': lengths,
        }
        return inputs

    @contextlib.contextmanager
    def no_encoder_cuda_graph(self):
        """Temporarily disable the encoder CUDA graph runner."""
        prev = self.encoder_cuda_graph_runner.enabled
        self.encoder_cuda_graph_runner.enabled = False
        try:
            yield
        finally:
            self.encoder_cuda_graph_runner.enabled = prev

    @with_warmup_flag
    def warmup_encoder(self) -> None:
        """
        Orchestrates the encoder warmup process by calling specialized
        warmup methods for torch.compile, the autotuner, and CUDA graphs.
        """
        # Create AutoTuner singleton in eager context before any compiled
        # forward.  Otherwise the first get() can happen inside torch.compile
        # tracing and trigger non-traceable code (time.time(), torch.cuda.*).
        AutoTuner.get()

        # General warmup configs come from engine capacity, NOT CUDA graph
        # config — torch.compile specialization must work even when CUDA
        # graphs are disabled.  max_num_tokens is already capped to
        # batch_size * max_seq_len by _init_max_num_tokens().
        max_shape = (self.batch_size, self.max_num_tokens, self.max_seq_len)
        warmup_configs: List[Tuple[int, int, int]] = list(
            dict.fromkeys([
                (1, 1, 1),
                max_shape,
                (1, 2, 2),
            ]))
        # Currently graph has not been captured, disable cuda graph for this warmup.
        with self.no_encoder_cuda_graph():
            self._general_warmup_encoder(warmup_configs)
            gc.collect()
            torch.cuda.empty_cache()

        self._run_autotuner_warmup_encoder()
        # Warm up every encoder graph shape before capturing any graph. Some
        # attention kernels switch implementations at smaller shapes and need
        # a larger workspace, so the first pass grows the workspace to its
        # maximum size. The second pass runs the final per-shape warmup and
        # captures without resizing the workspace.
        self._warmup_and_capture_encoder_cuda_graphs(
            self._capture_encoder_cuda_graphs)

        # Pre-populate the memory pool with max-shape allocations to reduce
        # fragmentation at runtime.
        self._general_warmup_encoder([max_shape])

    def _general_warmup_encoder(self, configs: List[Tuple[int, int,
                                                          int]]) -> None:
        """Run encoder forward passes for each (bs, nt, sl) config.

        Serves both torch.compile graph specialization and memory pool
        pre-population.
        """
        with self.no_encoder_cuda_graph():
            for bs, nt, sl in configs:
                inputs = self._create_encoder_warmup_inputs(bs, nt, sl)
                if inputs is None:
                    continue
                try:
                    logger.info(
                        f"Encoder general warmup: bs={bs}, nt={nt}, sl={sl}")
                    self.encoder_forward(inputs)
                    torch.cuda.synchronize()
                except torch.OutOfMemoryError:
                    if self._is_distributed_forward():
                        # Peers are inside the same forward's collectives and
                        # cannot follow a rank-local skip.
                        raise
                    logger.warning(f"OOM during encoder general warmup with "
                                   f"bs={bs}, nt={nt}, sl={sl}. Skipping.")
                    torch.cuda.empty_cache()

    def _run_autotuner_warmup_encoder(self) -> None:
        """Run a forward pass to populate the autotuner cache for the encoder."""
        if not self.llm_args.enable_autotuner:
            return
        AutoTuner.get().setup_distributed_state(self.mapping, self.dist)
        logger.info("Running encoder autotuner warmup...")

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        with self.no_encoder_cuda_graph(), autotune(cache_path=cache_path):
            inputs = self._create_encoder_warmup_inputs(self.batch_size,
                                                        self.max_num_tokens,
                                                        self.max_seq_len)
            if inputs is not None:
                self.encoder_forward(inputs)
                torch.cuda.synchronize()

        logger.info(f"[Encoder Autotuner] Cache size after warmup is "
                    f"{len(AutoTuner.get().profiling_cache)}")
        AutoTuner.get().print_profiling_cache()

    def _capture_encoder_cuda_graphs(self) -> None:
        """Warm up or capture encoder CUDA graphs for all feasible keys.

        Feasibility filter (also used in source):
          nt >= prev_sl + bs   (enough tokens for this sl bucket)
          prev_nt < bs * sl    (not enough tokens for a smaller nt bucket)
          nt <= bs * sl        (num tokens should not exceed total possible in batch)
          sl <= nt             (seq len should not exceed num tokens)
        """
        runner = self.encoder_cuda_graph_runner
        if not runner.enabled:
            return

        batch_sizes = sorted(self._encoder_cuda_graph_batch_sizes, reverse=True)
        num_tokens_list = sorted(self._cuda_graph_num_tokens)
        seq_lens_list = sorted(self._cuda_graph_seq_lens)

        operation = "warmup" if runner.is_warmup_only else "capture"
        num_processed = 0
        logger.info(f"Running encoder CUDA graph {operation} ...")
        for bs in batch_sizes:
            if bs > self.encoder_batch_size:
                continue
            for sl_idx, sl in reversed(list(enumerate(seq_lens_list))):
                prev_sl = seq_lens_list[sl_idx - 1] if sl_idx > 0 else 0
                for nt_idx, nt in reversed(list(enumerate(num_tokens_list))):
                    prev_nt = num_tokens_list[nt_idx - 1] if nt_idx > 0 else 0

                    if nt < prev_sl + bs or prev_nt >= bs * sl:
                        continue

                    if nt > bs * sl or sl > nt:
                        continue

                    inputs = self._create_encoder_warmup_inputs(bs, nt, sl)
                    if inputs is None:
                        continue

                    logger.info(f"Encoder CUDA graph {operation}: "
                                f"bs={bs}, nt={nt}, sl={sl}")
                    self.encoder_forward(inputs)
                    torch.cuda.synchronize()
                    num_processed += 1

        logger.info(f"Completed encoder CUDA graph {operation} for "
                    f"{num_processed} graph shape(s).")

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self.model.extra_attrs)
    @nvtx_range("encoder_forward")
    def encoder_forward(self, inputs: Dict[str, Any],
                        **kwargs) -> Dict[str, Any]:
        """Direct tensor-level forward for encode-only path.

        Bypasses ScheduledRequests/LlmRequest entirely. Takes a raw inputs
        dict, attempts encoder CUDA graph capture/replay if enabled, otherwise falls
        back to eager execution.

        Args:
            inputs: Dict with 'input_ids' and 'seq_lens' (required), plus
                any model-specific kwargs (token_type_ids, inputs_embeds, etc.).

        Returns:
            Dict with 'logits' tensor and any other model outputs.
        """
        moe_load_balancer = self.moe_load_balancer

        batch_size = len(inputs['seq_lens'])
        with self.encoder_cuda_graph_runner.pad_batch(
                inputs, batch_size) as padded_inputs:
            attn_metadata = self._set_up_attn_metadata(
                kv_cache_manager=None
            ) if self.encoder_attn_metadata is None else self.encoder_attn_metadata
            graph_attn_metadata, key = self.encoder_cuda_graph_runner.maybe_get_cuda_graph(
                padded_inputs, attn_metadata)
            # Unpad seq_lens when fallback to eager path.
            if key is None:
                padded_inputs['seq_lens'] = padded_inputs[
                    'seq_lens'][:batch_size]
            model_inputs = self._prepare_encoder_inputs(
                padded_inputs,
                attn_metadata=graph_attn_metadata,
                padded_num_tokens=key[1] if key is not None else None)
            forward_kwargs = {
                "gather_ids": None,
                "gather_context_logits": False,
                **kwargs,
            }

            with with_shared_pool(
                    self.encoder_cuda_graph_runner.get_graph_pool()):
                if key is None:
                    with MoeLoadBalancerIterContext(moe_load_balancer):
                        # Eager path — no graph for this bucket.
                        return self._forward_step(model_inputs,
                                                  **forward_kwargs)

                needs_capture = self.encoder_cuda_graph_runner.needs_capture(
                    key)
                if needs_capture:

                    def forward_fn(
                            capture_inputs: Dict[str, Any]) -> Dict[str, Any]:
                        capture_inputs = capture_inputs.copy()
                        forward_kwargs = capture_inputs.pop("_forward_kwargs")
                        with MoeLoadBalancerIterContext(moe_load_balancer):
                            return self._forward_step(capture_inputs,
                                                      **forward_kwargs)

                    capture_outputs = self.encoder_cuda_graph_runner.capture(
                        key, forward_fn, {
                            **model_inputs, "_forward_kwargs": forward_kwargs
                        })

                if self.encoder_cuda_graph_runner.is_warmup_only:
                    graph_outputs = capture_outputs
                else:
                    with MoeLoadBalancerIterContext(moe_load_balancer):
                        graph_outputs = self.encoder_cuda_graph_runner.replay(
                            key, {
                                **model_inputs, "_forward_kwargs":
                                forward_kwargs
                            })

            # Return a clone to avoid sharing data_ptr with the static buffers.
            outputs = {}
            for name, value in graph_outputs.items():
                if isinstance(value, torch.Tensor):
                    if name == "logits":
                        value = value[:batch_size]
                    outputs[name] = value.clone()
                else:
                    outputs[name] = value

            return outputs

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self.model.extra_attrs)
    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: ResourceManager,
                new_tensors_device: Optional[SampleStateTensors] = None,
                gather_context_logits: bool = False,
                cache_indirection_buffer: Optional[torch.Tensor] = None,
                num_accepted_tokens_device: Optional[torch.Tensor] = None,
                req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None):
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(
            resource_manager)

        attn_metadata = self._set_up_attn_metadata(kv_cache_manager,
                                                   draft_kv_cache_manager)
        if isinstance(attn_metadata, TrtllmAttentionMetadata):
            attn_metadata.trtllm_gen_jit_warmup = self._trtllm_gen_jit_warmup
        if self.enable_spec_decode:
            spec_resource_manager = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            spec_tree_manager = None
            if spec_resource_manager is not None and hasattr(
                    spec_resource_manager, 'spec_tree_manager'):
                spec_tree_manager = spec_resource_manager.spec_tree_manager
            spec_metadata = self._set_up_spec_metadata(spec_resource_manager,
                                                       no_cache=kv_cache_manager
                                                       is None)
            # attn_metadata now depends on spec_metadata since it determines the shape/content of spec_dec parameter Tensors
            is_spec_dec_mode = spec_metadata.spec_dec_mode.attention_need_spec_dec_mode(
                self.is_draft_model, self.attn_backend)
            # Propagate runtime_draft_len (already set on self by py_executor)
            # to spec_metadata so downstream code (eagle3, interface, trtllm) can read it.
            spec_metadata.runtime_draft_len = self.runtime_draft_len
            spec_metadata.runtime_tokens_per_gen_step = (
                self.get_runtime_tokens_per_gen_step(self.runtime_draft_len))

            # Parallel-draft modes advertise a per-gen-step width via
            # tokens_per_gen_step (PARD: 2K, DFlash: K+1).  Pass
            # (tokens_per_gen_step - 1) so generation_lengths = tokens_per_gen_step
            # and the XQA kernel computes the correct past_kv_len.
            if spec_metadata.spec_dec_mode.is_parallel_draft():
                sd_max_draft_len = self.original_max_total_draft_tokens
                sd_max_total = self.original_max_total_draft_tokens
            else:
                sd_max_draft_len = self.original_max_draft_len
                sd_max_total = self._spec_dec_max_total_draft_tokens

            # Fill slot-ID buffer for update_spec_dec_param
            if (spec_tree_manager is not None
                    and spec_tree_manager.use_dynamic_tree
                    and not self.is_draft_model):
                spec_tree_manager.slot_storage.fill_all_slot_ids(
                    scheduled_requests.context_requests,
                    scheduled_requests.generation_requests,
                )

            attn_metadata.update_spec_dec_param(
                batch_size=scheduled_requests.batch_size,
                is_spec_decoding_enabled=is_spec_dec_mode,
                is_spec_dec_tree=spec_metadata.is_spec_dec_tree,
                is_spec_dec_dynamic_tree=spec_metadata.is_spec_dec_dynamic_tree,
                max_draft_len=sd_max_draft_len,
                max_total_draft_tokens=sd_max_total,
                spec_metadata=spec_metadata,
                spec_tree_manager=spec_tree_manager,
                num_contexts=scheduled_requests.num_context_requests)
        else:
            spec_resource_manager = None
            spec_metadata = None

        moe_load_balancer = self.moe_load_balancer
        if kv_cache_manager is None:
            inputs, gather_ids = self._prepare_tp_inputs_no_cache(
                scheduled_requests, attn_metadata, spec_metadata,
                resource_manager)

            with MoeLoadBalancerIterContext(moe_load_balancer):
                # Special handling for multimodal encoder only mode
                if self.llm_args.mm_encoder_only:
                    return self._forward_step_mm_encoder_only(
                        inputs, scheduled_requests)
                else:
                    return self._forward_step(
                        inputs,
                        gather_ids=gather_ids,
                        gather_context_logits=gather_context_logits)

        graph_requests = scheduled_requests
        promoted_context_request_ids: frozenset[int] = frozenset()
        # Non-linear tree input preparation expands runtime_draft_len to the
        # total tree width after graph selection. Only linear-tree zero-draft
        # iterations can therefore safely reuse a zero-draft graph.
        can_promote_spec_decode = (not self.enable_spec_decode
                                   or (not self.is_draft_model
                                       and self.runtime_draft_len == 0
                                       and self.spec_config is not None
                                       and self.spec_config.is_linear_tree))
        # TODO: Generalize these conservative gates as actual-draft, beam, and
        # context-parallel providers for decoder-only LLMs gain support for
        # promoted final-context rows. Each relaxation must preserve whole-batch
        # fallback on graph miss and prove parity with the provider's native
        # q_len=1 path. Encoder-decoder and non-LLM engines remain out of scope.
        if (scheduled_requests.num_context_requests > 0
                and self.cuda_graph_runner.enabled and can_promote_spec_decode
                and not self.use_beam_search
                and not self._is_encoder_decoder_model()
                and not self._is_encode_only
                and not self.llm_args.mm_encoder_only
                # PLE owns recurrent n-gram and convolution state. Promoting a
                # fresh final-context row would skip its cache-slot reset.
                and not self._model_uses_ple_recurrent_state and
                self.mapping.cp_size == 1):
            graph_requests, promoted_context_request_ids = \
                _make_single_token_context_graph_batch(
                    scheduled_requests,
                    self._is_final_multimodal_context_decode_compatible)

        with self.cuda_graph_runner.pad_batch(
                graph_requests, resource_manager,
                self.runtime_draft_len) as padded_graph_requests:
            # Callee already no-ops when use_mrope=False, but the Python call /
            # frame setup itself is non-trivial under high concurrency. Gating
            # at the caller avoids that overhead for non-mrope models.
            if self.use_mrope:
                self._pad_batch_seed_mrope_delta_cache(padded_graph_requests)

            # Refresh is_all_greedy_sample for the *current* batch BEFORE the
            # CUDA graph key is built below. The key includes this flag to pick
            # the argmax vs advanced-sampling graph variant; populate (inside
            # _prepare_inputs) runs later and fills the matching GPU buffers.
            # Without this pre-scan the key would use the previous iteration's
            # stale value and could replay the advanced graph against
            # unpopulated (greedy) buffers, hanging the run (e.g. MTP nextn>=2).
            if spec_metadata is not None:
                spec_metadata.update_is_all_greedy_sample(
                    padded_graph_requests.all_requests())
                self._sync_group_all_greedy_sample(spec_metadata)

            peft_cache_data_type = None
            if getattr(self, "cuda_graph_lora_manager", None) is not None:
                peft_cache_manager = resource_manager.get_resource_manager(
                    ResourceManagerType.PEFT_CACHE_MANAGER)
                peft_cache_data_type = peft_cache_manager.data_type

            use_lora_graph = self._use_lora_cuda_graph(padded_graph_requests)
            maybe_attn_metadata, maybe_spec_metadata, key = self.cuda_graph_runner.maybe_get_cuda_graph(
                padded_graph_requests,
                enable_spec_decode=self.enable_spec_decode,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_tokens_cuda=self.draft_tokens_cuda
                if self.is_spec_decode else None,
                new_tensors_device=new_tensors_device,
                spec_resource_manager=spec_resource_manager,
                promoted_context_request_ids=promoted_context_request_ids,
                peft_cache_data_type=peft_cache_data_type,
                use_lora_graph=use_lora_graph,
            )

            can_run_graph = key is not None
            if can_run_graph:
                attn_metadata = maybe_attn_metadata
                spec_metadata = maybe_spec_metadata
                execution_requests = padded_graph_requests
                execution_promoted_context_ids = promoted_context_request_ids
            else:
                attn_metadata = self.attn_metadata
                if self.enable_spec_decode:
                    spec_metadata = self.spec_metadata
                else:
                    spec_metadata = None
                execution_requests = scheduled_requests
                execution_promoted_context_ids = frozenset()

            # Fill slot-ID buffer for scatter inside draft loop
            if (self.enable_spec_decode and spec_tree_manager is not None
                    and spec_tree_manager.use_dynamic_tree
                    and not self.is_draft_model):
                spec_tree_manager.slot_storage.fill_all_slot_ids(
                    execution_requests.context_requests,
                    execution_requests.generation_requests,
                )

            inputs, gather_ids = self._prepare_inputs(
                execution_requests,
                kv_cache_manager,
                attn_metadata,
                spec_metadata,
                new_tensors_device,
                cache_indirection_buffer,
                num_accepted_tokens_device,
                req_id_to_old_request,
                resource_manager,
                can_run_graph,
                execution_promoted_context_ids,
                use_lora_graph=use_lora_graph)
            if execution_promoted_context_ids:
                self.iter_states[
                    'num_ctx_requests'] = scheduled_requests.num_context_requests
                self.iter_states['num_ctx_tokens'] = sum(
                    request.context_chunk_size
                    for request in scheduled_requests.context_requests)
                self.iter_states[
                    'num_generation_tokens'] = scheduled_requests.num_generation_requests
            self._prepare_inputs_event = torch.cuda.Event()
            self._prepare_inputs_event.record()

            breakable_runner = self.breakable_cuda_graph_runner
            # Device-window selection: re-rank the staged ragged layout by the
            # verified block's own confidence, between input staging and the
            # replay (stream-ordered, no host sync). Graph steps only: eager
            # and capture steps run the host shape split, which is a valid
            # window assignment.
            if (can_run_graph and not self.is_warmup and getattr(
                    self, "_dspark_device_budget", None) is not None):
                self._apply_device_window_prologue(inputs, new_tensors_device)

            with with_shared_pool(self.cuda_graph_runner.get_graph_pool()):

                def forward_step():
                    with MoeLoadBalancerIterContext(moe_load_balancer):
                        return self._forward_step(
                            inputs,
                            gather_ids=gather_ids,
                            gather_context_logits=gather_context_logits)

                if not can_run_graph:
                    if (breakable_runner is not None
                            and breakable_runner.is_capturing):
                        return breakable_runner.capture_model_body(forward_step)

                    num_tokens = inputs['input_ids'].shape[0]
                    can_run_breakable_graph = (
                        breakable_runner is not None
                        and get_per_request_prefill_cuda_graph_flag()
                        and not gather_context_logits
                        and breakable_runner.has_graph(num_tokens))
                    if can_run_breakable_graph and not breakable_runner.is_warming_up:
                        outputs = breakable_runner.execute(
                            num_tokens, forward_step)
                    else:
                        # real eager or BCG warmup or PCG
                        outputs = forward_step()
                else:
                    needs_capture = self.cuda_graph_runner.needs_capture(key)
                    if needs_capture:

                        def capture_forward_fn(inputs: Dict[str, Any]):
                            with MoeLoadBalancerIterContext(moe_load_balancer):
                                return self._forward_step(
                                    inputs,
                                    gather_ids=gather_ids,
                                    gather_context_logits=gather_context_logits)

                        def capture_postprocess_fn(inputs: Dict[str, Any]):
                            self._postprocess_inputs(inputs)

                        capture_outputs = self.cuda_graph_runner.capture(
                            key,
                            capture_forward_fn,
                            inputs,
                            enable_spec_decode=self.enable_spec_decode,
                            postprocess_fn=capture_postprocess_fn)

                    if self.cuda_graph_runner.is_warmup_only:
                        outputs = capture_outputs
                    elif needs_capture:
                        # Refresh attention metadata for the current batch's
                        # draft cache before replaying the captured graph.
                        saved_draft = prepare_attn_metadata_for_draft_replay(
                            attn_metadata, draft_kv_cache_manager)
                        try:
                            outputs = self.cuda_graph_runner.replay(key, inputs)
                        finally:
                            restore_attn_metadata_after_draft_replay(
                                attn_metadata, saved_draft)
                    else:
                        saved_draft = prepare_attn_metadata_for_draft_replay(
                            attn_metadata, draft_kv_cache_manager)
                        try:
                            with MoeLoadBalancerIterContext(moe_load_balancer):
                                outputs = self.cuda_graph_runner.replay(
                                    key, inputs)
                        finally:
                            restore_attn_metadata_after_draft_replay(
                                attn_metadata, saved_draft)

            if self.forward_pass_callable is not None:
                self.forward_pass_callable()

            self._execute_logit_post_processors(scheduled_requests, outputs)

            return outputs

    def model_forward(self, **kwargs):
        attrs = get_model_extra_attrs()
        assert attrs is not None, "Model extra attrs is not set"
        attrs["attention_metadata"] = weakref.ref(kwargs['attn_metadata'])
        attrs.update(self.model.model_config.extra_attrs)
        attrs["spec_metadata"] = kwargs.get('spec_metadata', None)

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
                      *,
                      gather_ids: Optional[torch.Tensor] = None,
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
            return {
                'mm_embeddings': [],
                'mm_embedding_request_indices': [],
                'mm_embedding_lengths': [],
            }
        # Some ctx requests carry only mrope metadata (no actual vision
        # content). Skip them so the encoder only runs on real image payloads.
        mm_context_requests = [(request_idx, request) for request_idx, request
                               in enumerate(scheduled_requests.context_requests)
                               if request.py_multimodal_data is not None]
        if len(mm_context_requests) != len(multimodal_params):
            raise ValueError(
                "mm_encoder_only expects one multimodal payload per context "
                "request carrying py_multimodal_data")
        mm_request_indices_with_payload = []
        mm_params_with_payload = []
        mm_embedding_lengths = []
        for (request_idx,
             request), multimodal_param in zip(mm_context_requests,
                                               multimodal_params):
            if not _has_mm_payload_keys(request.py_multimodal_data):
                # mrope-only warmup request (no actual vision content) -> skip.
                continue
            multimodal_embedding_lengths = get_multimodal_embedding_lengths(
                request)
            if multimodal_embedding_lengths is None:
                # Vision payload keys present but no pre-computed embedding
                # lengths — skip to avoid a downstream sum(None) TypeError.
                continue
            mm_request_indices_with_payload.append(request_idx)
            mm_params_with_payload.append(multimodal_param)
            mm_embedding_lengths.append(multimodal_embedding_lengths)
        if not mm_params_with_payload:
            return {
                'mm_embeddings': [],
                'mm_embedding_request_indices': [],
                'mm_embedding_lengths': [],
            }
        # For mm_encoder_only mode, we only run the vision encoder part
        # The model should be a vision encoder (e.g., Qwen2VisionModelBase)
        mm_embeddings = self.model.forward(mm_params_with_payload)
        assert len(
            mm_embeddings
        ) == 1, "mm_embeddings should be a 1-element list, mix modality (video+image) is not supported"

        split_lengths = [sum(lengths) for lengths in mm_embedding_lengths]
        mm_embeddings = list(torch.split(mm_embeddings[0], split_lengths,
                                         dim=0))
        if len(mm_embeddings) != len(mm_embedding_lengths):
            raise ValueError(
                "mm_encoder_only produced an embedding batch that does not "
                "match mm_embedding_lengths")

        # Extract mrope position data from multimodal_params if available
        mrope_position_ids_list = []
        mrope_position_deltas_list = []
        for multimodal_param in mm_params_with_payload:
            mrope_config = multimodal_param.multimodal_data.get(
                'mrope_config', {})
            mrope_position_ids = mrope_config.get('mrope_position_ids')
            mrope_position_deltas = mrope_config.get('mrope_position_deltas')
            if mrope_position_ids is not None:
                mrope_position_ids_list.append(mrope_position_ids)
            if mrope_position_deltas is not None:
                mrope_position_deltas_list.append(mrope_position_deltas)

        # mrope lists must align 1:1 with multimodal_params (or be empty);
        # the sampler indexes them by per-MM-result position into mm_embeddings.
        assert (len(mrope_position_ids_list) == len(mrope_position_deltas_list)
                and len(mrope_position_ids_list)
                in (0, len(mm_params_with_payload))), (
                    f"mrope alignment: got {len(mrope_position_ids_list)} ids, "
                    f"{len(mrope_position_deltas_list)} deltas, "
                    f"{len(mm_params_with_payload)} mm params")

        result = {
            'mm_embeddings': mm_embeddings,
            'logits': None,
            'mm_embedding_request_indices': mm_request_indices_with_payload,
            'mm_embedding_lengths': mm_embedding_lengths,
        }
        if mrope_position_ids_list:
            result['mrope_position_ids'] = mrope_position_ids_list
        if mrope_position_deltas_list:
            result['mrope_position_deltas'] = mrope_position_deltas_list

        return result

    def _make_encoder_attn_metadata(
        self,
        sequence_lengths: List[int],
        request_ids: List[int],
    ):
        """Build fresh, no-cache attention metadata for one packed encoder
        batch. ``self.attn_metadata`` is not reused because that object is
        bound to the decoder's KV-cache manager."""
        if len(sequence_lengths) != len(request_ids):
            raise ValueError("Encoder sequence lengths and request IDs must "
                             "have the same length.")
        sparse_metadata_params = (
            self.sparse_attention_config.to_sparse_metadata_params(
                pretrained_config=self.model.model_config.pretrained_config)
            if self.sparse_attention_config is not None else None)
        encoder_attn_metadata = self.attn_backend.Metadata(
            max_num_requests=self.encoder_batch_size,
            max_num_tokens=self.encoder_max_num_tokens,
            max_num_sequences=self.encoder_batch_size * self.max_beam_width,
            kv_cache_manager=None,
            mapping=self.mapping,
            runtime_features=self.attn_runtime_features,
            enable_flash_mla=self.model.model_config.enable_flash_mla,
            enable_context_mla_with_cached_kv=False,
            cache_indirection=None,
            sparse_metadata_params=sparse_metadata_params,
            num_heads_per_kv=1,
        )
        assert isinstance(
            encoder_attn_metadata,
            (VanillaAttentionMetadata, TrtllmAttentionMetadata)
        ), "Only vanilla and trtllm attention metadata are supported for the encoder pass"

        encoder_attn_metadata.seq_lens = torch.tensor(
            sequence_lengths,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        encoder_attn_metadata.num_contexts = len(sequence_lengths)
        encoder_attn_metadata.max_seq_len = self.max_seq_len
        encoder_attn_metadata.request_ids = request_ids
        encoder_attn_metadata.prepare_encoder_only()
        return encoder_attn_metadata

    def _prepare_encoder_decoder_encoder_inputs(
        self,
        encoder_input_ids: List[int],
        encoder_position_ids: List[int],
        sequence_lengths: List[int],
        request_ids: List[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> Dict[str, Any]:
        num_tokens = len(encoder_input_ids)
        if num_tokens != len(encoder_position_ids):
            raise ValueError("Encoder input IDs and position IDs must have "
                             "the same length.")
        assert num_tokens <= self.encoder_max_num_tokens, (
            f"encoder packed length ({num_tokens}) exceeds "
            f"encoder_max_num_tokens ({self.encoder_max_num_tokens})")

        encoder_attn_metadata = self._make_encoder_attn_metadata(
            sequence_lengths, request_ids)
        encoder_input_ids_t = torch.tensor(encoder_input_ids,
                                           dtype=torch.int,
                                           pin_memory=prefer_pinned())
        encoder_position_ids_t = torch.tensor(encoder_position_ids,
                                              dtype=torch.int,
                                              pin_memory=prefer_pinned())
        encoder_graph_runner = self.encoder_cuda_graph_runner
        encoder_batch_size = len(sequence_lengths)
        use_graph_staging = (
            encoder_graph_runner.enabled and
            (encoder_batch_size in encoder_graph_runner.supported_batch_sizes or
             (encoder_graph_runner.padding_enabled and encoder_batch_size
              <= encoder_graph_runner.max_supported_batch_size)))

        return {
            'encoder_input_ids':
            (encoder_input_ids_t if use_graph_staging else
             encoder_input_ids_t.to('cuda', non_blocking=True)),
            'encoder_position_ids':
            ((encoder_position_ids_t
              if use_graph_staging else encoder_position_ids_t.to(
                  'cuda', non_blocking=True)).unsqueeze(0)),
            'encoder_attn_metadata':
            encoder_attn_metadata,
            'encoder_seq_lens':
            sequence_lengths,
            'encoder_input_ids_host':
            encoder_input_ids_t,
            'encoder_position_ids_host':
            encoder_position_ids_t,
            'resource_manager':
            resource_manager,
        }

    @nvtx_range("_prepare_tp_inputs_encoder_features")
    def _prepare_tp_inputs_encoder_features(
        self,
        encoder_requests: List[LlmRequest],
        resource_manager: Optional[ResourceManager] = None,
    ):
        """Pack encoder inputs for feature-driven audio encoders (Whisper).

        The encoder input is a per-request feature tensor (an opaque audio
        tensor, e.g. Whisper's 30 s-padded waveform) rather than token ids, and
        the packed sequence lengths are the post-encoder position counts
        (``encoder_output_len``), not the raw feature length.
        """
        features: List[torch.Tensor] = []
        sequence_lengths: List[int] = []
        request_ids: List[int] = []

        for request in encoder_requests:
            request_features = request.py_encoder_input_features
            if request_features is None:
                raise ValueError(
                    f"Encoder request {request.py_request_id} has no "
                    "encoder_input_features; feature- and token-driven "
                    "encoder requests cannot share one batch.")
            features.append(request_features)
            sequence_lengths.append(int(request.encoder_output_len))
            request_ids.append(request.py_request_id)

        num_tokens = sum(sequence_lengths)
        assert num_tokens <= self.encoder_max_num_tokens, (
            f"encoder packed length ({num_tokens}) exceeds "
            f"encoder_max_num_tokens ({self.encoder_max_num_tokens})")

        encoder_attn_metadata = self._make_encoder_attn_metadata(
            sequence_lengths, request_ids)

        inputs = {
            'input_features': self._pack_encoder_features(features),
            'encoder_attn_metadata': encoder_attn_metadata,
            'encoder_seq_lens': sequence_lengths,
            'resource_manager': resource_manager,
        }
        return inputs

    def _pack_encoder_features(self,
                               features: List[torch.Tensor]) -> torch.Tensor:
        """Pack per-request feature tensors into one device tensor.

        Copies through a lazily-grown pinned staging buffer so the H2D
        transfer is a single async DMA. ``torch.cat(...).to('cuda')`` from
        pageable request tensors forces a synchronous driver-staged copy per
        batch, which dominates encoder host time at large batch sizes
        (measured 51.7 ms/call at bs32 on a Xeon 8570 host).
        """
        first = features[0]
        uniform = first.device.type == 'cpu' and all(
            f.shape[1:] == first.shape[1:] and f.dtype == first.dtype
            and f.device.type == 'cpu' for f in features)
        if not uniform:
            return torch.cat(features, dim=0).to('cuda', non_blocking=True)

        rows = sum(f.shape[0] for f in features)
        staging = getattr(self, '_encoder_feature_staging', None)
        if (staging is None or staging.dtype != first.dtype
                or staging.shape[1:] != first.shape[1:]
                or staging.shape[0] < rows):
            # Retire the previous batch's H2D before dropping the last
            # reference to the buffer it reads from.
            if staging is not None:
                self._encoder_feature_staging_event.synchronize()
            staging = torch.empty((rows, *first.shape[1:]),
                                  dtype=first.dtype,
                                  pin_memory=prefer_pinned())
            self._encoder_feature_staging = staging
            self._encoder_feature_staging_event = torch.cuda.Event()
            # Dedicated copy stream: enqueued on the encoder stream the H2D
            # would queue behind the previous encoder forward, and the next
            # batch's staging reuse would host-block on that forward. One
            # stream for the runner's lifetime, so a reallocation cannot
            # strand work on a stream nothing waits on again.
            if getattr(self, '_encoder_feature_copy_stream', None) is None:
                self._encoder_feature_copy_stream = torch.cuda.Stream()
        else:
            # The previous batch's H2D from this buffer must be complete
            # before its rows are overwritten. It ran on the copy stream,
            # concurrent with the previous forward, so this is ~always done.
            self._encoder_feature_staging_event.synchronize()

        offset = 0
        for f in features:
            staging[offset:offset + f.shape[0]].copy_(f)
            offset += f.shape[0]
        consumer_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._encoder_feature_copy_stream):
            packed = staging[:rows].to('cuda', non_blocking=True)
            self._encoder_feature_staging_event.record()
        consumer_stream.wait_event(self._encoder_feature_staging_event)
        # The device tensor was allocated on the copy stream; mark it used by
        # the consumer stream so the allocator does not recycle it early.
        packed.record_stream(consumer_stream)
        return packed

    @nvtx_range("_prepare_tp_inputs_encoder")
    def _prepare_tp_inputs_encoder(
        self,
        encoder_requests: List[LlmRequest],
        resource_manager: Optional[ResourceManager] = None,
    ):
        """Pack encoder-side inputs for an encoder-decoder forward pass.

        Mirrors the no-cache path used by ``mm_encoder_only`` and the
        legacy ``EncoderBuffers`` shape contract: ``encoder_input_ids``
        and ``encoder_position_ids`` are concatenated across requests
        into a single ``[sum(encoder_output_len)]`` tensor, with one
        non-causal :class:`AttentionMetadata` describing the packed
        encoder batch.

        The encoder pass does not touch any KV-cache pool. The cross pool is
        only written by the decoder's cross-attention on the first context
        step. Self-pool blocks for the decoder are reserved on the next
        scheduler iteration when the request transitions to ``CONTEXT_INIT``.
        """
        if not encoder_requests:
            raise ValueError(
                "_prepare_tp_inputs_encoder called with no encoder requests")

        # Feature-driven audio encoders (Whisper) carry a tensor instead of
        # encoder token ids; they take a dedicated prep path (which rejects
        # mixed feature/token batches).
        if any(
                getattr(request, "py_encoder_input_features", None) is not None
                for request in encoder_requests):
            return self._prepare_tp_inputs_encoder_features(
                encoder_requests, resource_manager=resource_manager)

        encoder_input_ids: List[int] = []
        encoder_position_ids: List[int] = []
        sequence_lengths: List[int] = []
        request_ids: List[int] = []

        for request in encoder_requests:
            tokens = request.encoder_tokens
            if tokens is None:
                raise ValueError(
                    f"Encoder request {request.py_request_id} has no "
                    "encoder_tokens; encoder_input_token_ids must be wired "
                    "through executor_request_to_llm_request.")
            seq_len = len(tokens)
            encoder_input_ids.extend(tokens)
            encoder_position_ids.extend(
                self._apply_position_id_offset(list(range(seq_len))))
            sequence_lengths.append(seq_len)
            request_ids.append(request.py_request_id)

        return self._prepare_encoder_decoder_encoder_inputs(
            encoder_input_ids=encoder_input_ids,
            encoder_position_ids=encoder_position_ids,
            sequence_lengths=sequence_lengths,
            request_ids=request_ids,
            resource_manager=resource_manager,
        )

    @nvtx_range("_forward_step_encoder")
    def _forward_step_encoder(
        self,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        """Run the encoder stack and return packed encoder hidden states.

        Returns ``[sum(encoder_output_len), hidden_size]`` (matches the
        ``EncoderBuffers`` shape contract from the legacy TRT path).
        Slicing back into per-request hidden states is the executor's
        responsibility — see :meth:`PyExecutor._scatter_encoder_output`.
        """
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            inner = getattr(self.model, "model", None)
            encoder = getattr(inner, "encoder",
                              None) if inner is not None else None
        if encoder is None:
            raise AttributeError(
                "Model does not expose an `encoder` submodule; encoder-decoder "
                "models must define a top-level `encoder` (or `model.encoder`) "
                "stack to participate in the encoder iteration.")

        # Feature-driven encoders (Whisper): the feature tensor is opaque to
        # the engine — no token embedding, no position ids, no dtype cast.
        # The model's forward casts internally (Whisper's raw waveforms must
        # reach the log-mel STFT in fp32).
        input_features = inputs.get('input_features')
        if input_features is not None:
            return encoder(
                input_features=input_features,
                attn_metadata=inputs['encoder_attn_metadata'],
            )

        # Encoder operates on packed token IDs.  Models like T5 own the
        # shared embedding on ``self.model`` rather than inside the
        # encoder stack, so we go through the top-level model when
        # available so the embedding is applied consistently with the
        # decoder pass.
        top_level_model = self._get_top_level_model()
        embed = getattr(top_level_model, "shared_embedding", None) or getattr(
            top_level_model, "embed_tokens", None)
        encoder_input_ids = inputs['encoder_input_ids']
        if embed is not None:
            hidden_states = embed(encoder_input_ids)
            embed_scale = getattr(top_level_model, "embed_scale", None)
            if embed_scale is not None:
                hidden_states = hidden_states * embed_scale
        else:
            # Fall back to letting the encoder accept token ids directly.
            hidden_states = encoder_input_ids

        encoder_attn_metadata = inputs['encoder_attn_metadata']
        position_ids = inputs.get('encoder_position_ids')
        if position_ids is not None and position_ids.dim() == 2:
            position_ids = position_ids.squeeze(0)

        encoder_hidden_states = encoder(
            hidden_states=hidden_states,
            attn_metadata=encoder_attn_metadata,
            position_ids=position_ids,
        )
        return encoder_hidden_states

    def _forward_step_encoder_cuda_graph(
        self,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        return self._forward_step_encoder({
            'encoder_input_ids':
            inputs['input_ids'],
            'encoder_position_ids':
            inputs.get('position_ids'),
            'encoder_attn_metadata':
            inputs['attn_metadata'],
            'resource_manager':
            inputs.get('resource_manager'),
        })

    def _encoder_forward_enc_dec(
        self,
        inputs: Dict[str, Any],
    ) -> torch.Tensor:
        """Run the encoder-decoder encoder, using a CUDA graph when eligible."""
        input_ids = inputs.get('encoder_input_ids_host')
        position_ids = inputs.get('encoder_position_ids_host')
        seq_lens = inputs['encoder_seq_lens']
        runner = self.encoder_cuda_graph_runner

        if input_ids is None or position_ids is None:
            return self._forward_step_encoder(inputs)

        runner_inputs = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'seq_lens': seq_lens,
            'resource_manager': inputs.get('resource_manager'),
        }
        with runner.pad_batch(runner_inputs,
                              len(seq_lens)) as padded_runner_inputs:
            graph_attn_metadata, key = runner.maybe_get_cuda_graph(
                padded_runner_inputs, inputs['encoder_attn_metadata'])
            if key is None:
                if inputs['encoder_input_ids'].device.type == 'cpu':
                    inputs = dict(inputs)
                    inputs['encoder_input_ids'] = inputs[
                        'encoder_input_ids'].to('cuda', non_blocking=True)
                    inputs['encoder_position_ids'] = inputs[
                        'encoder_position_ids'].to('cuda', non_blocking=True)
                return self._forward_step_encoder(inputs)

            # Every graph key aliases the same pinned staging allocation. Retire
            # the previous captured H2D before updating seq_lens or any other
            # shared host input for this replay.
            runner.retire_staging()
            model_inputs = runner.prepare_encoder_decoder_inputs(
                padded_runner_inputs, key, seq_lens)
            graph_attn_metadata.prepare_encoder_cuda_graph_replay(
                model_inputs['seq_lens'], key[1])
            model_inputs['attn_metadata'] = graph_attn_metadata

            moe_load_balancer = self.moe_load_balancer
            with with_shared_pool(runner.get_graph_pool()):
                capture_outputs = None
                if runner.needs_capture(key):

                    def capture_forward_fn(
                            capture_inputs: Dict[str, Any]) -> torch.Tensor:
                        with MoeLoadBalancerIterContext(moe_load_balancer):
                            return self._forward_step_encoder_cuda_graph(
                                capture_inputs)

                    capture_outputs = runner.capture(key, capture_forward_fn,
                                                     model_inputs)

                if runner.is_warmup_only:
                    graph_outputs = capture_outputs
                else:
                    with MoeLoadBalancerIterContext(moe_load_balancer):
                        graph_outputs = runner.replay(key, model_inputs)

        if not isinstance(graph_outputs, torch.Tensor):
            raise TypeError("Encoder-decoder CUDA graph replay must return "
                            "a tensor of encoder hidden states.")
        return runner.restore_encoder_decoder_output(key, graph_outputs,
                                                     model_inputs)

    @nvtx_range("forward_encoder")
    def forward_encoder(
        self,
        encoder_requests: List[LlmRequest],
        resource_manager: Optional[ResourceManager] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Run the encoder stack for ``encoder_requests``.

        Returns a tuple ``(encoder_hidden_states, encoder_seq_lens)``
        where the hidden states tensor is shaped
        ``[sum(encoder_seq_lens), hidden_size]`` (one packed batch).
        The accompanying ``encoder_seq_lens`` list is in the same
        ordering as ``encoder_requests``, so callers can split the
        packed output 1:1.

        This entry point is the encoder-step analog of the legacy
        ``TrtEncoderModel::forwardAsync`` (see §2.6/§2.7).  The decoder
        IFB step is unchanged and continues to flow through
        :meth:`forward`.
        """
        if not encoder_requests:
            raise ValueError("forward_encoder called with no encoder requests")

        with torch.inference_mode():
            graph_result = self._maybe_forward_encoder_graph(encoder_requests)
            if graph_result is not None:
                return graph_result

            inputs = self._prepare_tp_inputs_encoder(
                encoder_requests, resource_manager=resource_manager)
            encoder_hidden_states = self._encoder_forward_enc_dec(inputs)

        return encoder_hidden_states, inputs['encoder_seq_lens']

    def _maybe_forward_encoder_graph(
        self,
        encoder_requests: List[LlmRequest],
    ) -> Optional[Tuple[torch.Tensor, List[int]]]:
        """Try to serve the encoder batch from a captured CUDA graph.

        Returns ``(encoder_hidden_states, encoder_seq_lens)`` on a graph hit
        (the hidden states are CLONED from the graph's static output buffer —
        the executor stores views of the result across scheduler iterations,
        and a later replay of the same bucket would clobber them), or None to
        fall back to the eager path.
        """
        runner = self.encoder_cuda_graph_runner
        if not runner.enabled or not runner.feature_mode:
            return None

        fixed = runner.config.fixed_seq_len
        features: List[torch.Tensor] = []
        for request in encoder_requests:
            f = request.py_encoder_input_features
            # Exactly one row per request: `_replay_features` copies
            # `f.shape[0]` rows per request into a mirror slice the bucket
            # sizes at one row per request, so a multi-row feature would
            # overrun it.
            if (f is None or int(request.encoder_output_len) != fixed
                    or tuple(f.shape) != (1, *runner.config.feature_shape)
                    or f.dtype != runner.config.feature_dtype):
                # A shape the model's `encoder_graph_spec()` did not predict
                # misses on every request, not just this one: capture already
                # spent its time and memory and nothing will ever replay. Say
                # so once — silence here reads as "graphs are working".
                logger.warning_once(
                    "Encoder CUDA graph: request features do not match the "
                    "captured contract (expected shape "
                    f"{(1, *runner.config.feature_shape)} dtype "
                    f"{runner.config.feature_dtype} encoder_output_len "
                    f"{fixed}, got shape "
                    f"{None if f is None else tuple(f.shape)} dtype "
                    f"{None if f is None else f.dtype} encoder_output_len "
                    f"{int(request.encoder_output_len)}); the encoder step "
                    "stays eager.",
                    key="encoder_cuda_graph_feature_contract_warning")
                return None
            features.append(f)

        seq_lens = [fixed] * len(encoder_requests)
        output = self._feature_encoder_graph_forward(
            features=features,
            seq_lens=seq_lens,
            request_ids=[r.py_request_id for r in encoder_requests],
        )
        if output is None:
            return None

        real_tokens = fixed * len(encoder_requests)
        return output[:real_tokens].clone(), seq_lens

    def _feature_encoder_graph_forward(
        self,
        features: List[torch.Tensor],
        seq_lens: List[int],
        request_ids: List[int],
    ) -> Optional[torch.Tensor]:
        """Run one feature encoder batch through its CUDA graph.

        Shared by the runtime path and by warmup/capture. Returns the packed
        hidden states for the *padded* batch (the caller slices back to the
        real rows), or None when no captured graph fits and the caller must
        fall back to eager.
        """
        runner = self.encoder_cuda_graph_runner
        fixed = runner.config.fixed_seq_len
        graph_inputs = {'seq_lens': seq_lens, 'input_features': features}

        with runner.pad_batch(graph_inputs, len(seq_lens)) as padded_inputs:
            # `pad_batch` extends seq_lens to the captured bucket, and the
            # metadata takes one request id per sequence. Pad slots carry no
            # request; the encoder pass runs without a KV cache, so their ids
            # are never looked up and only have to exist and stay distinct.
            padded_seq_lens = padded_inputs['seq_lens']
            padded_request_ids = list(request_ids) + [
                -(i + 1)
                for i in range(len(padded_seq_lens) - len(request_ids))
            ]
            # A captured bucket is served from graph-resident metadata, so
            # only a miss pays for a fresh `TrtllmAttentionMetadata` +
            # `prepare_encoder_only()`.
            graph_attn_metadata, key = runner.captured_graph_metadata(
                padded_inputs)
            if key is None:
                eager_attn_metadata = self._make_encoder_attn_metadata(
                    padded_seq_lens, padded_request_ids)
                graph_attn_metadata, key = runner.maybe_get_cuda_graph(
                    padded_inputs, eager_attn_metadata)
            if key is None:
                return None
            padded_inputs['attn_metadata'] = graph_attn_metadata

            capture_output = None
            if runner.needs_capture(key):
                padded_batch_size, padded_num_tokens, _ = key
                # Feature-mode seq_lens are constant per bucket: initialize
                # the graph-resident metadata once at capture.
                graph_attn_metadata.prepare_encoder_cuda_graph_replay(
                    [fixed] * padded_batch_size, padded_num_tokens)
                capture_output = runner.capture(
                    key, self._enc_dec_encoder_graph_forward_fn, padded_inputs)

            if runner.is_warmup_only:
                return capture_output
            return runner.replay(key, padded_inputs)

    def _enc_dec_encoder_graph_forward_fn(
            self, capture_inputs: Dict[str, Any]) -> torch.Tensor:
        """Run the encoder step over a graph runner's capture inputs.

        Adapts the runner's flat capture dict to `_forward_step_encoder`'s
        keyword names. Passed to `EncoderCUDAGraphRunner.capture` so the body
        traced into the graph is the same code path a replay stands in for.

        Args:
            capture_inputs: Padded encoder inputs owned by the graph runner.

        Returns:
            Encoder hidden states, `[padded_batch, fixed_seq_len, hidden]`.
        """
        return self._forward_step_encoder({
            'input_features':
            capture_inputs['input_features'],
            'encoder_attn_metadata':
            capture_inputs['attn_metadata'],
            'encoder_seq_lens':
            capture_inputs['seq_lens'],
        })

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

    @staticmethod
    def _apply_logits_processors(request, logits_processors, logits_tensor,
                                 beam_width, token_ids, logits_row_offset):
        logits_rows = logits_tensor[logits_row_offset:logits_row_offset +
                                    beam_width]
        # Reshape to align w/ the shape used in the TRT backend,
        # so the same logit processors can be used across both backends.
        logits_rows = logits_rows.view(beam_width, 1, -1)
        for lp in logits_processors:
            lp_params = inspect.signature(lp).parameters

            assert 4 <= len(lp_params) <= 5, (
                "Logit post processor signature must match the `LogitsProcessor` interface "
                "defined in `tensorrtllm.sampling_params`.")
            lp(request.py_request_id, logits_rows, token_ids, None, None)

        # logits_rows is a view into logits_tensor (narrow + view never
        # copy), so the processors already mutated it in place. Writing it
        # back would be a self-assignment, which torch rejects for the
        # non-contiguous slices a TP-padded vocab produces.

    def _execute_logit_post_processors(self,
                                       scheduled_requests: ScheduledRequests,
                                       outputs: dict):
        """Apply logit post processors (in-place modify outputs Tensors) if any."""

        if not (self.mapping.is_last_pp_rank()):
            return

        if not isinstance(outputs, dict) or "logits" not in outputs:
            # TODO: support models that don't return outputs as dict
            return

        logits_tensor = outputs["logits"]

        logits_row_offset = 0
        request_groups = (
            (scheduled_requests.context_requests, True),
            (scheduled_requests.generation_requests, False),
        )

        for requests, is_context_request in request_groups:
            for request in requests:
                if is_context_request:
                    beam_width = 1
                    row_stride = 1
                else:
                    # Generation rows are laid out at the static admission
                    # width, so that is the stride between requests, while
                    # only the leading beam_width rows hold live beams under
                    # a variable beam width array. Advancing the offset by the
                    # narrower width would make every request after the first
                    # rewrite another request's logits rows in place.
                    beam_width = request.get_beam_width_by_iter(
                        for_next_iteration=False)
                    row_stride = request.py_beam_width

                logits_processors = getattr(request,
                                            "py_logits_post_processors", None)
                if logits_processors:
                    if (not is_context_request and getattr(
                            request, "py_verify_len", None) is not None):
                        raise RuntimeError(
                            "per-request logits post-processors are not "
                            "supported with DSpark ragged verification: the "
                            "processor interface assumes one uniform row "
                            "stride and would modify another request's packed "
                            "logits after a short verify window")
                    token_ids = ([request.get_tokens(0)]
                                 if is_context_request else [
                                     request.get_tokens(beam_idx)
                                     for beam_idx in range(beam_width)
                                 ])
                    if (is_context_request
                            and request.py_orig_prompt_len < len(token_ids[0])):
                        # Skip as we only need to apply logit processor on the last context request
                        logits_row_offset += row_stride
                        continue

                    self._apply_logits_processors(request, logits_processors,
                                                  logits_tensor, beam_width,
                                                  token_ids, logits_row_offset)
                logits_row_offset += row_stride

    def wait_for_input_copy(self):
        """
        Wait for input preparation and H2D copy of previous iteration before modifying host input,
        otherwise the input of previous iteration will be overwritten.
        """
        if self._prepare_inputs_event is not None:
            self._prepare_inputs_event.synchronize()
