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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch._dynamo.config

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.utils import torch_multi_arange
from tensorrt_llm._utils import (is_trace_enabled, maybe_pin_memory, nvtx_range,
                                 prefer_pinned, release_gc, torch_dtype_to_str,
                                 trace_func)
from tensorrt_llm.bindings.internal.runtime import TaskLayerModuleConfig
from tensorrt_llm.inputs.multimodal import (MultimodalInput, MultimodalParams,
                                            MultimodalRuntimeData,
                                            _has_mm_payload_keys,
                                            check_mm_embed_cumsum_if_needed,
                                            strip_mm_data_for_generation)
from tensorrt_llm.inputs.registry import (BaseMultimodalInputProcessor,
                                          create_input_processor,
                                          create_input_processor_with_hash)
from tensorrt_llm.llmapi.llm_args import (CudaGraphConfig, DecodingBaseConfig,
                                          EncodeCudaGraphConfig,
                                          SeqLenAwareSparseAttentionConfig,
                                          TorchCompileConfig, TorchLlmArgs)
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
from ..distributed import Distributed
from ..distributed.communicator import init_pp_comm
from ..expert_statistic import ExpertStatistic
from ..memory_buffer_utils import clear_memory_buffers, with_shared_pool
from ..metadata import KVCacheParams
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.modeling_multimodal_encoder import MultimodalEncoderMixin
from ..models.modeling_multimodal_mixin import MultimodalModelMixin
from ..models.modeling_multimodal_utils import filter_mm_token_from_input_ids
from ..models.modeling_utils import DecoderModelForCausalLM
from ..modules.fused_moe.moe_load_balancer import (MoeLoadBalancer,
                                                   MoeLoadBalancerIterContext)
from ..peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from ..speculative import (SpecMetadata, get_draft_kv_cache_manager,
                           get_num_extra_kv_tokens, get_spec_metadata,
                           prepare_attn_metadata_for_draft_replay,
                           restore_attn_metadata_after_draft_replay,
                           update_spec_config_from_loaded_model)
from ..speculative.drafting_loops import BaseDraftingLoopWrapper
from ..speculative.eagle3 import Eagle3ResourceManager, Eagle3SpecMetadata
from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from ..utils import (get_model_extra_attrs,
                     set_per_request_piecewise_cuda_graph_flag,
                     set_torch_compiling, with_model_extra_attrs)
from .config_utils import is_mla
from .cuda_graph_runner import (ENC_DEC_CUDA_GRAPH_DUMMY_TOKEN_NUM,
                                CUDAGraphRunner, CUDAGraphRunnerConfig,
                                EncoderCUDAGraphRunner,
                                EncoderCUDAGraphRunnerConfig)
from .guided_decoder import CapturableGuidedDecoder
from .kv_cache_manager_v2 import KVCacheManagerV2
from .layerwise_nvtx_marker import LayerwiseNvtxMarker
from .llm_request import (LlmRequest, LlmRequestState, get_draft_token_length,
                          get_multimodal_embedding_lengths)
from .mamba_cache_manager import MambaHybridCacheManager
from .model_loader import ModelLoader, _construct_checkpoint_loader
from .resource_manager import (BaseResourceManager, KVCacheManager,
                               PeftCacheManager, ResourceManager,
                               ResourceManagerType)
from .sampler import SampleStateTensors
from .scheduler import ScheduledRequests
from .trace_log_utils import log_mem_snapshot


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
    num_extra_decoding_steps: int = 0,
) -> Tuple[list[int], list[int]]:
    """Cap piecewise CUDA graph capture candidates at the engine's reachable
    `num_tokens` ceiling `max_batch_size * (max_seq_len - 1 - num_extra_decoding_steps)`
    and ensure the ceiling itself is captured.

    Each in-flight request must leave room for at least one decode token,
    so the ceiling is the largest forward-pass `num_tokens` the warmup
    builder can construct. Including it in the capture set closes the
    runtime padding gap between the next-largest candidate and the ceiling
    (otherwise ISLs in that gap have no graph >= them and fall back to
    eager).

    Returns `(kept, unrecordable)` where `kept` is sorted ascending,
    deduped, and contains the ceiling whenever it is positive.
    `unrecordable` is the sorted unique set of input entries above the
    ceiling but within `max_num_tokens`.
    """
    max_capturable_num_tokens = max(
        0, max_batch_size * (max_seq_len - 1 - num_extra_decoding_steps))
    piecewise_capacity_limit = min(max_num_tokens, max_capturable_num_tokens)
    kept = sorted(
        {i
         for i in candidate_num_tokens if 0 < i <= piecewise_capacity_limit})
    if piecewise_capacity_limit > 0 and (not kept or kept[-1]
                                         < piecewise_capacity_limit):
        kept.append(piecewise_capacity_limit)
    unrecordable = sorted({
        i
        for i in candidate_num_tokens
        if max_capturable_num_tokens < i <= max_num_tokens
    })
    return kept, unrecordable


def _build_request_multimodal_input(
        request: LlmRequest, cache_enabled: bool) -> Optional[MultimodalInput]:
    # Skip building this input (and its `from_components` validation) when the cache is disabled.
    if not cache_enabled or request.multimodal_hashes is None:
        return None
    # `multimodal_input` is consumed only by the encoder-cache key path
    # (`MultimodalModelMixin._encoder_cache_keys`), which uses UUID-aware multimodal hashes
    # internally. Although the `multimodal_uuids` are not exposed as an attribute, they remain in
    # the backing C++ request for KV-cache block keys and cache events.
    return MultimodalInput.from_components(
        request.multimodal_hashes,
        request.multimodal_positions,
        request.multimodal_lengths,
        mm_item_run_cu_offsets=request.multimodal_item_run_cu_offsets,
        mm_run_positions=request.multimodal_run_positions,
        mm_run_lengths=request.multimodal_run_lengths,
    )


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


def _configure_deep_gemm_pdl() -> None:
    global _DEEP_GEMM_PDL_CONFIGURED
    if _DEEP_GEMM_PDL_CONFIGURED:
        return

    from tensorrt_llm import deep_gemm

    deep_gemm.set_pdl(os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1")
    _DEEP_GEMM_PDL_CONFIGURED = True


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
        drafting_loop_wrapper: Optional[Callable[[torch.nn.Module],
                                                 torch.nn.Module]] = None,
        model: Optional[torch.nn.Module] = None,
        checkpoint_loader: Optional[BaseCheckpointLoader] = None,
        model_weights_memory_tag: Optional[str] = None,
        model_weights_restore_mode=None,
    ):
        _configure_deep_gemm_pdl()

        self.forward_pass_callable = None
        self.ub_buffers = None
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
        # Multimodal encoder runtime sizes; fall back to LLM-side values when
        # the encoder-specific knobs are unset.
        (
            self.encoder_batch_size,
            self.encoder_max_num_tokens,
        ) = llm_args.get_encoder_runtime_sizes()

        if checkpoint_loader is None:
            checkpoint_loader = _construct_checkpoint_loader(
                llm_args.backend,
                llm_args.checkpoint_loader,
                llm_args.checkpoint_format,
                mx_config=llm_args.mx_config,
                mx_model_name=llm_args.model,
            )

        self.mapping = mapping
        if mapping.has_pp():
            init_pp_comm(mapping)
        # Start with the established pool size. Once the model is loaded we
        # selectively enable headroom for the non-PP DeepSeek-V4 overlap path.
        from ._util import (compute_max_num_sequences,
                            should_enable_dsv4_adp_dummy_fixes,
                            should_enable_dsv4_overlap_headroom)
        self.max_num_seq_slots = compute_max_num_sequences(
            mapping, self.batch_size, llm_args.disable_overlap_scheduler)
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

        preserve_wrapped_eagle3_widths = (spec_config is not None
                                          and is_draft_model
                                          and drafting_loop_wrapper is not None
                                          and
                                          spec_config.spec_dec_mode.is_eagle3())
        # The draft model won't have any draft tokens attached to
        # generation requests when we invoke it autoregressively
        if spec_config is not None and is_draft_model and not preserve_wrapped_eagle3_widths:
            spec_config.max_draft_len = 0
            spec_config.max_total_draft_tokens = 0
        self.spec_config = spec_config
        self.is_spec_decode = spec_config is not None
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
        self.input_processor_with_hash = create_input_processor_with_hash(
            self.input_processor,
            encoder_cache_enabled=(
                llm_args.multimodal_config is not None
                and llm_args.multimodal_config.encoder_cache_max_bytes > 0),
        )
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
            self.model, moe_load_balancer = self.model_loader.load(
                checkpoint_dir=model_path, checkpoint_loader=checkpoint_loader)
            if isinstance(moe_load_balancer, MoeLoadBalancer):
                setattr(self, "moe_load_balancer", moe_load_balancer)
        else:
            self.model = model
        pretrained_config = self.model.model_config.pretrained_config
        model_type = getattr(pretrained_config, "model_type", None)
        # Keep the scheduler/dummy fix model-scoped, while the larger slot pool
        # is restricted to the validated MTP overlap configuration. PP remains
        # on its established path for follow-up changes.
        self._enable_dsv4_adp_dummy_fixes = (should_enable_dsv4_adp_dummy_fixes(
            model_type, mapping))
        self._enable_dsv4_overlap_headroom = (
            should_enable_dsv4_overlap_headroom(
                model_type, spec_config, mapping,
                llm_args.disable_overlap_scheduler))
        self.max_num_seq_slots = compute_max_num_sequences(
            mapping,
            self.batch_size,
            llm_args.disable_overlap_scheduler,
            enable_overlap_headroom=self._enable_dsv4_overlap_headroom,
        )
        if drafting_loop_wrapper is not None:
            self.model = drafting_loop_wrapper(self.model)
            self.model_is_wrapped = True
        else:
            self.model_is_wrapped = False
        self.sparse_attention_config = self.model.model_config.sparse_attention_config
        # In case that some tests use stub models and override `_load_model`.
        if not hasattr(self.model, 'extra_attrs'):
            self.model.extra_attrs = {}
        self._set_up_multimodal_encoder_attn_metadata()
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

        if (isinstance(self.cuda_graph_config, EncodeCudaGraphConfig)
                and self._is_encoder_decoder_model()):
            logger.warning(
                "EncodeCudaGraphConfig is not supported for encoder-decoder "
                "models. Use DecodeCudaGraphConfig or CudaGraphConfig for "
                "decoder CUDA graphs. CUDA graphs will be disabled.")
            self.cuda_graph_config = None

        cuda_graph_batch_sizes = self.cuda_graph_config.batch_sizes if self.cuda_graph_config else CudaGraphConfig.model_fields[
            'batch_sizes'].default
        cuda_graph_padding_enabled = self.cuda_graph_config.enable_padding if self.cuda_graph_config else CudaGraphConfig.model_fields[
            'enable_padding'].default

        # Encode-only CUDA graph detection. Decode configs do not define these
        # encoder-specific bucket fields.
        cuda_graph_num_tokens = []
        cuda_graph_seq_lens = []
        if isinstance(self.cuda_graph_config, EncodeCudaGraphConfig):
            cuda_graph_num_tokens = self.cuda_graph_config.num_tokens or []
            cuda_graph_seq_lens = self.cuda_graph_config.seq_lens or []

        if (self._is_encode_only and self.cuda_graph_config is not None
                and (not cuda_graph_num_tokens or not cuda_graph_seq_lens)):
            missing = []
            if not cuda_graph_num_tokens:
                missing.append("num_tokens/max_num_token")
            if not cuda_graph_seq_lens:
                missing.append("seq_lens/max_seq_len")
            logger.warning(
                f"encode_only=True with a CudaGraphConfig, but "
                f"{' and '.join(missing)} not set. Encoder CUDA graphs "
                f"require both. Encoder CUDA graphs will be disabled. "
                f"To enable them, specify e.g. "
                f"EncodeCudaGraphConfig(max_batch_size=64, num_tokens=[128, 256, "
                f"512], max_seq_len=128, enable_padding=True).")

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

        self._torch_compile_enabled = torch_compile_enabled
        self._torch_compile_piecewise_cuda_graph = torch_compile_piecewise_cuda_graph

        piecewise_cuda_graph_num_tokens = (
            torch_compile_piecewise_cuda_graph_num_tokens
            or cuda_graph_batch_sizes or [])

        num_extra_decoding_steps = self._get_num_extra_decoding_steps()
        self._piecewise_cuda_graph_num_tokens, unrecordable = (
            _filter_piecewise_capture_num_tokens(
                piecewise_cuda_graph_num_tokens,
                max_num_tokens=self.max_num_tokens,
                max_batch_size=self.batch_size,
                max_seq_len=self.max_seq_len,
                num_extra_decoding_steps=num_extra_decoding_steps,
            ))
        if unrecordable:
            logger.warning(
                f"Skipping piecewise CUDA graph capture for num_tokens="
                f"{unrecordable}: exceeds reachable ceiling "
                f"max_batch_size*(max_seq_len-1-num_extra_decoding_steps)="
                f"{max(0, self.batch_size * (self.max_seq_len - 1 - num_extra_decoding_steps))}. "
                f"Capturing the ceiling itself; raise max_seq_len for larger graphs."
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
                    capture_num_tokens=self._piecewise_cuda_graph_num_tokens,
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
        self.has_previous_device_draft = False
        self.previous_accepted_tokens_cuda = torch.empty((self.batch_size, ),
                                                         dtype=torch.int,
                                                         device='cuda')

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
            ) or self.model_is_wrapped
            self.max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
            self.max_draft_len = spec_config.max_draft_len
            self.runtime_draft_len = spec_config.max_draft_len

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
        self._cuda_graph_mem_pool = self._torch_compile_backend._graph_pool_handle if self._torch_compile_enabled else None

        self._cuda_graph_padding_enabled = cuda_graph_padding_enabled

        self._cuda_graph_batch_sizes = _filter_cuda_graph_batch_sizes(
            cuda_graph_batch_sizes, self.batch_size, self.max_num_tokens,
            self.original_max_total_draft_tokens,
            self._cuda_graph_padding_enabled) if cuda_graph_batch_sizes else []

        self._max_cuda_graph_batch_size = (self._cuda_graph_batch_sizes[-1] if
                                           self._cuda_graph_batch_sizes else 0)

        # Encoder CUDA graph bucket lists
        self._cuda_graph_num_tokens = _filter_cuda_graph_num_tokens(
            cuda_graph_num_tokens, self.max_num_tokens,
            self._cuda_graph_padding_enabled) if cuda_graph_num_tokens else []

        self._max_cuda_graph_num_tokens = (self._cuda_graph_num_tokens[-1] if
                                           self._cuda_graph_num_tokens else 0)
        self._cuda_graph_seq_lens = _filter_cuda_graph_seq_lens(
            cuda_graph_seq_lens, self.max_seq_len,
            self._cuda_graph_padding_enabled) if cuda_graph_seq_lens else []

        self._max_cuda_graph_seq_len = (self._cuda_graph_seq_lens[-1]
                                        if self._cuda_graph_seq_lens else 0)

        self._dynamic_draft_len_mapping = self._compute_dynamic_draft_len_mapping(
        )

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

        # Create config and runner
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
        )
        self.cuda_graph_runner = CUDAGraphRunner(cuda_graph_runner_config)

        # Create Encoder CUDA graph config and runner.
        encoder_cuda_graph_runner_config = EncoderCUDAGraphRunnerConfig(
            use_cuda_graph=(self._is_encode_only
                            and self.cuda_graph_config is not None
                            and bool(self._cuda_graph_num_tokens)
                            and bool(self._cuda_graph_seq_lens)),
            cuda_graph_padding_enabled=self._cuda_graph_padding_enabled,
            cuda_graph_batch_sizes=self._cuda_graph_batch_sizes,
            cuda_graph_num_tokens=self._cuda_graph_num_tokens,
            cuda_graph_seq_lens=self._cuda_graph_seq_lens,
            max_cuda_graph_batch_size=self._max_cuda_graph_batch_size,
            max_cuda_graph_num_tokens=self._max_cuda_graph_num_tokens,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            cuda_graph_mem_pool=self._cuda_graph_mem_pool,
        )
        self.encoder_cuda_graph_runner = EncoderCUDAGraphRunner(
            encoder_cuda_graph_runner_config)

        # Initialize CUDA Graph LoRA manager if LoRA is enabled
        self.cuda_graph_lora_manager: Optional[CudaGraphLoraManager] = None

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
                device='cuda',
                max_tokens_per_seq=max_tokens_per_seq)

            logger.info(
                f"Initialized CUDA Graph LoRA manager, "
                f"max {max_lora_size} adapters, max rank {lora_config.max_lora_rank}"
            )

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
        multimodal_config = self.model.model_config.multimodal_config
        return (multimodal_config is not None
                and multimodal_config.encoder_cache_max_bytes > 0)

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

    @with_warmup_flag
    @warmup_with_kv_cache_cleanup
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
        # Reset the global cuda graph dummy requests in warmup.
        self.cuda_graph_runner.padding_dummy_requests = {}

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

        can_run_general_warmup = (
            not is_enc_dec and not self.is_draft_model
            and not self.mapping.has_cp_helix() and self.guided_decoder is None
            and not isinstance(kv_cache_manager, MambaHybridCacheManager))

        log_mem_snapshot("warmup/before_warmup")
        if not is_enc_dec:
            self._run_attention_warmup(resource_manager, can_run_general_warmup)

        if can_run_general_warmup:
            # Specialize torch.compile graphs across the key input shapes before CUDA graph capture.
            warmup_requests_configs = self._get_full_general_warmup_requests(
                resource_manager)
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

        # Autotuner warmup uses context-only requests. Helix CP
        # is decode-only and runs into issues with autotuner warmup.
        if not is_enc_dec and not self.mapping.has_cp_helix():
            self._run_autotuner_warmup(resource_manager)
            log_mem_snapshot("warmup/after_autotuner")
            # Release the autotuner's exploration-mode intermediates. The
            # exploration leftovers are pure waste that hide tens of GiB from
            # non-torch allocators (cuBLAS handle workspace, UCX/NIXL,
            # NVSHMEM).
            gc.collect()
            torch.cuda.empty_cache()
        with self.cuda_graph_runner.allow_capture():
            self._run_cuda_graph_warmup(resource_manager)
        log_mem_snapshot("warmup/after_cuda_graph_capture")
        if can_run_general_warmup:
            # Pre-populate the memory pool with max-shape allocations to reduce
            # fragmentation at runtime.
            warmup_requests_configs = self._get_max_shape_warmup_requests(
                resource_manager)
            self._general_warmup(resource_manager, warmup_requests_configs)
            log_mem_snapshot("warmup/after_memory_pool_prepop")

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

    def _assert_all_tp_ranks_have_warmup_batch(self, batch,
                                               num_tokens: int) -> None:
        """Assert every TP rank has a valid warmup batch, or raise with diagnostics.

        Under attention-DP, each rank's KV cache available capacity can differ at
        runtime, causing _create_warmup_request to return None on some ranks while
        others proceed into forward() with tp_comm collectives — deadlocking the
        job. This check prevents the deadlock by failing early with diagnostic info.
        """
        if self.mapping.tp_size <= 1:
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
                    if batch is None and self.mapping.tp_size <= 1:
                        # Safe to skip, but never silently: a skip during KV
                        # cache estimation makes the profiling peak
                        # unrepresentative of this shape.
                        logger.warning(
                            f"Skipping general warmup with {num_tokens} tokens "
                            f"({num_gen_tokens} generation): not enough KV "
                            f"cache space.")
                        continue
                    self._assert_all_tp_ranks_have_warmup_batch(
                        batch, num_tokens)
                    if batch is None:
                        logger.warning(
                            f"Skipping general warmup with {num_tokens} tokens "
                            f"({num_gen_tokens} generation): not enough KV "
                            f"cache space on any TP rank.")
                        continue
                    logger.info(
                        f"Run warmup with {num_tokens} tokens, include {num_gen_tokens} generation tokens"
                    )
                    self.forward(batch,
                                 new_tensors_device=None,
                                 resource_manager=resource_manager)
                    torch.cuda.synchronize()
            except torch.OutOfMemoryError:
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

        for num_tokens, num_gen_requests in warmup_requests_configs:
            warmup_request = self._create_warmup_request(
                resource_manager,
                num_tokens=num_tokens,
                num_gen_requests=num_gen_requests)

            with self.no_cuda_graph(), self._release_batch_context(
                    warmup_request, resource_manager) as batch:
                if batch is None and self.mapping.tp_size <= 1:
                    continue  # Not enough KV cache space (single rank, safe to skip)
                self._assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
                if batch is None:
                    continue  # All ranks agree: not enough space
                with trtllm_gen_fmha_jit_warmup():
                    self.forward(batch,
                                 new_tensors_device=None,
                                 resource_manager=resource_manager)
                torch.cuda.synchronize()

    def _run_autotuner_warmup(self, resource_manager: ResourceManager):
        """Runs a forward pass to populate the autotuner cache."""
        if not self.llm_args.enable_autotuner:
            return
        AutoTuner.get().setup_distributed_state(self.mapping, self.dist)
        logger.info("Running autotuner warmup...")
        kv_cache_manager = resource_manager.get_resource_manager(
            self.kv_cache_manager_key)
        token_num_upper_bound = min(self.max_num_tokens,
                                    self.batch_size * (self.max_seq_len - 1))
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self.original_max_draft_len)

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        with self.no_cuda_graph(), autotune(cache_path=cache_path):
            warmup_request = self._create_warmup_request(
                resource_manager, curr_max_num_tokens, 0)
            with self._release_batch_context(warmup_request,
                                             resource_manager) as batch:
                if batch is None and self.mapping.tp_size <= 1:
                    pass  # Single rank, safe to skip
                else:
                    self._assert_all_tp_ranks_have_warmup_batch(
                        batch, curr_max_num_tokens)
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

        # Clear workspace buffers allocated during the autotuner forward pass.
        # The autotuner runs a context-only forward with max_num_tokens, which
        # causes the global Buffers pool to cache large MoE/GEMM workspaces.
        # If not cleared, these inflate the memory baseline seen by the KV cache
        # profiler, reducing memory available for activations during inference.
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
        self, cuda_graph_batch_sizes: list[int],
        spec_resource_manager: Optional[BaseResourceManager]
    ) -> list[tuple[int, int]]:
        """Determine which (batch_size, draft_len) graphs to capture.

        Returns:
            List of (batch_size, draft_len) tuples for CUDA graph capture.
        """
        # Case 1: Draft model (two-model speculative decoding)
        # Two-model path is deprecated and will be removed in the near future
        if self.is_draft_model:
            if self.model_is_wrapped and self.is_spec_decode and spec_resource_manager is not None and isinstance(
                    spec_resource_manager, Eagle3ResourceManager):
                # The CDL path uses draft_len > 0 for the number of iterations in the drafting loop.
                draft_len = self.original_max_total_draft_tokens
            else:
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

        # Determine which graphs to capture
        graphs_to_capture = self._get_graphs_to_capture(cuda_graph_batch_sizes,
                                                        spec_resource_manager)
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

        def _run_capture_pass(force_non_greedy: bool, label: str) -> None:
            spec_metadata = self.spec_metadata
            if force_non_greedy and spec_metadata is not None:
                spec_metadata._force_non_greedy_for_capture = True
                # maybe_get_cuda_graph reads spec_metadata.is_all_greedy_sample
                # to build the graph cache key BEFORE populate runs inside
                # _prepare_inputs. Pre-flip it here so the very first capture
                # in this pass uses the non-greedy key; populate's override
                # below will keep it False on every subsequent iteration.
                spec_metadata.is_all_greedy_sample = False
            try:
                for bs, draft_len in graphs_to_capture:
                    if bs > self.batch_size:
                        continue

                    for max_seq_len in max_seq_len_list:
                        warmup_request = self._create_cuda_graph_warmup_request(
                            resource_manager, bs, draft_len, max_seq_len)
                        with self._release_batch_context(
                                warmup_request, resource_manager) as batch:
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
                                f"Run generation-only CUDA graph warmup ({label}) "
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
                if force_non_greedy and spec_metadata is not None:
                    spec_metadata._force_non_greedy_for_capture = False

        # Pass 1: greedy fast-path (dummy requests carry no sampling params,
        # so is_all_greedy_sample is naturally True).
        _run_capture_pass(force_non_greedy=False, label="greedy")
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
            _run_capture_pass(force_non_greedy=True, label="advanced sampling")
        # Set the value back to the original value after cuda graph warmups are complete
        self.enable_spec_decode = self.is_spec_decode
        # The advanced-sampling capture pass above leaves is_all_greedy_sample
        # set to False on spec_metadata. Reset it to the default so the first
        # real iteration's graph-key selection is not seeded with this
        # capture-only value. (update_is_all_greedy_sample refreshes it every
        # iteration; this is a defensive guard.)
        if self.spec_metadata is not None:
            self.spec_metadata.is_all_greedy_sample = True

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

        # When using piecewise cuda graph, the logits may suffer severe memory fragmentation problem.
        # As the number of requests grows, the blocks allocated by torch cannot be reused.
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
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(
            resource_manager)
        cross_kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CROSS_KV_CACHE_MANAGER)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        # Request-lifetime managers that piggyback on the KV cache (e.g. Inkling's
        # short-conv state pool) must release their per-request rows for warmup /
        # estimation dummy batches too; otherwise a leaked slot is later reused
        # (with stale state) by a real request whose id collides with a dummy id.
        conv_state_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CONV_STATE_MANAGER)
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
                    if conv_state_manager is not None:
                        conv_state_manager.free_resources(req)

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

        num_extra_decoding_steps = self._get_num_extra_decoding_steps()

        if num_gen_requests > self.batch_size:
            return None
        num_gen_tokens = num_gen_requests * (1 + self.max_total_draft_tokens)
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
                num_left_over_tokens / kv_cache_manager.tokens_per_block
            ) + num_gen_requests * self.max_beam_width

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
                num_extra_decoding_steps=num_extra_decoding_steps,
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
                num_extra_decoding_steps=num_extra_decoding_steps,
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
            max_seq_len: int = None) -> Optional[ScheduledRequests]:
        """Creates a dummy ScheduledRequests tailored for CUDA graph capture."""
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
        num_extra_decoding_steps = self._get_num_extra_decoding_steps()
        runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(
            draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
        is_enc_dec = self._is_encoder_decoder_model()
        max_encoder_output_len = (
            self._get_max_encoder_output_len(resource_manager)
            if is_enc_dec else None)

        # Add (batch_size - 1) dummy requests with the minimal seq_len.
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
            num_extra_decoding_steps=num_extra_decoding_steps,
            draft_kv_cache_manager=draft_kv_cache_manager)

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
        if max_position_embeddings is not None:
            token_num = min(token_num, max_position_embeddings - _kv_draft)

        assert token_num > num_extra_decoding_steps, (
            "Cannot fuse drafting loop. Not enough KV cache space for all draft tokens."
        )
        token_num -= num_extra_decoding_steps
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
            num_extra_decoding_steps=num_extra_decoding_steps,
            draft_kv_cache_manager=draft_kv_cache_manager)

        if max_seq_len_request is None:
            free_warmup_requests()
            return None
        else:
            max_seq_len_request = max_seq_len_request[0]

        # Insert the longest request first to simulate padding for the CUDA graph.
        requests.insert(0, max_seq_len_request)
        result.generation_requests = requests
        if spec_resource_manager is not None:
            spec_resource_manager.add_dummy_requests(
                request_ids=list(range(batch_size)))
        if self._is_encoder_decoder_model():
            if not self._add_cross_dummy_requests(result.generation_requests,
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

        return self.attn_metadata

    @property
    def is_multimodal(self) -> bool:
        """True iff this engine drives a multimodal model.

        Primary signal: ``MultimodalModelMixin`` is the
        canonical marker — multimodal LM classes inherit from it. Until
        every model has migrated (Mistral done; Qwen-VL, Nemotron, Gemma,
        Phi-4-MM, etc. pending), fall back to whether the input processor
        subclasses ``BaseMultimodalInputProcessor``, which every
        multimodal model necessarily provides at the data boundary.

        TODO(TRTLLM-13542): Once all multimodal models inherit
        ``MultimodalModelMixin``, drop the input-processor fallback so the
        model class itself is the single source of truth.
        """
        if isinstance(self.model, MultimodalModelMixin):
            return True
        return isinstance(self.input_processor, BaseMultimodalInputProcessor)

    def _set_up_multimodal_encoder_attn_metadata(self) -> None:
        """Construct AttentionMetadata for any multimodal encoders inside the
        loaded model, using the engine's encoder runtime sizes
        (`encoder_max_batch_size` / `encoder_max_num_tokens`, falling back to
        the LLM-side `max_batch_size` / `max_num_tokens`).

        Mirrors `_set_up_attn_metadata` for the LLM backbone: encoders opt in
        by inheriting `MultimodalEncoderMixin`, and the engine drives the construction
        so the sizes match ``llm_args.get_encoder_runtime_sizes()`` rather
        than being hardcoded inside each encoder's ``__init__``.
        """
        for module in self.model.modules():
            if isinstance(module, MultimodalEncoderMixin):
                module.setup_attn_metadata(
                    max_num_requests=self.encoder_batch_size,
                    max_num_tokens=self.encoder_max_num_tokens)

    def _set_up_spec_metadata(
            self,
            spec_resource_manager: Optional[BaseResourceManager],
            no_cache=False):
        spec_config = self.spec_config if self.enable_spec_decode else None
        # Only the scoped DeepSeek-V4 overlap path opts into larger metadata
        # buffers. Passing None preserves the established max_num_requests
        # fallback for every other model, including MTP-Eagle with PP.
        num_seq_slots = (self.max_num_seq_slots
                         if self._enable_dsv4_overlap_headroom else None)
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
        2. The model module reference.
        3. CUDA Graph captures (via :meth:`_release_cuda_graphs`).
        4. Input processors.
        5. Userbuffers (``ub.ub_deallocate`` per buffer); on per-buffer
           failure the unfreed buffers are kept attached so a deterministic
           retry doesn't double-free already-released ones, and the
           collected errors are re-raised after the loop.

        Idempotency:
            Subsequent calls are no-ops (guarded by ``_cleanup_done``).
            The flag is set only at the end, so a partial cleanup that
            raises mid-way will be retried on the next call.

        Raises:
            RuntimeError: If one or more userbuffer deallocations fail
                (chained from the first error). All other steps are
                best-effort and either succeed or leak silently with
                their errors logged at warning level by callees.

        Called from:
            - :meth:`PyExecutor.shutdown` (deterministic teardown).
            - :meth:`__del__` (best-effort fallback during garbage
              collection / interpreter shutdown).
        """
        if getattr(self, "_cleanup_done", False):
            return

        # Cleanup is not truly atomic: released CUDA/GMS resources cannot be
        # rolled back.  Keep each handle live until its own release succeeds,
        # so a failed cleanup can be retried without double-freeing resources
        # that were already released.
        model_loader = getattr(self, "model_loader", None)
        if model_loader is not None:
            model_loader.cleanup()
            self.model_loader = None

        self.model = None

        self._release_cuda_graphs()
        self.input_processor = None
        self.input_processor_with_hash = None

        ub_buffers = getattr(self, 'ub_buffers', None)
        if ub_buffers:
            remaining_ub_buffers = []
            ub_errors = []
            for u in ub_buffers:
                try:
                    ub.ub_deallocate(u.addr)
                except RuntimeError as e:
                    # Keep failed buffers attached so a deterministic
                    # cleanup() call can retry without double-freeing buffers
                    # that were already deallocated successfully.
                    remaining_ub_buffers.append(u)
                    ub_errors.append(e)
            self.ub_buffers = remaining_ub_buffers or None
            if ub_errors:
                raise RuntimeError(
                    "Failed to deallocate one or more userbuffers during "
                    "PyTorchModelEngine cleanup") from ub_errors[0]

        # Release model weights.
        release_gc()
        self._cleanup_done = True

    def __del__(self) -> None:
        """Best-effort cleanup during garbage collection.

        Delegates to :meth:`cleanup`. Catches ``RuntimeError`` (raised
        when one or more userbuffer deallocations fail) and
        ``AttributeError`` (typical on partially-initialized engines
        torn down during interpreter shutdown when module references
        have already been cleared); both are logged and swallowed
        because destructors cannot reliably surface exceptions.

        Deterministic callers (``PyExecutor.shutdown``) should call
        :meth:`cleanup` directly so they see any failure.
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
        if self._torch_compile_backend is not None:
            self._torch_compile_backend.clear_piecewise_cuda_graphs()
        if hasattr(self,
                   'cuda_graph_runner') and self.cuda_graph_runner is not None:
            self.cuda_graph_runner.clear()
        if hasattr(self, 'encoder_cuda_graph_runner'
                   ) and self.encoder_cuda_graph_runner is not None:
            self.encoder_cuda_graph_runner.clear()

    def get_max_num_sequences(self) -> int:
        """
        Return the maximum number of sequences that the model supports. PyExecutor needs this to compute max_num_active_requests
        """
        num_batches = self.mapping.pp_size
        return num_batches * self.batch_size

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
                return list(self.dist.tp_cp_allgather(num_tokens))
            return list(self.dist.tp_allgather(num_tokens))
        return None

    def _get_all_rank_ctx_requests(self, num_ctx_requests: int):
        if self.enable_attention_dp:
            return list(self.dist.tp_allgather(num_ctx_requests))
        return None

    def _set_spec_metadata_all_rank_num_tokens(
            self, spec_metadata: SpecMetadata,
            spec_all_rank_num_tokens: List[int],
            all_rank_num_seqs: List[int]) -> None:
        # Eagle3 / MTP-eagle one-model use subseq_all_rank_num_tokens for
        # draft loop iterations i>0 (per-sequence counts, since each
        # sequence contributes one token per iteration).
        spec_metadata.all_rank_num_tokens = spec_all_rank_num_tokens
        spec_metadata.all_rank_num_seqs = all_rank_num_seqs
        if (spec_metadata.spec_dec_mode.is_mtp_eagle_one_model()
                or spec_metadata.spec_dec_mode.is_eagle3_one_model()):
            spec_metadata.subseq_all_rank_num_tokens = all_rank_num_seqs

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
                        "Not all ranks can run piecewise cuda graph, disable piecewise cuda graph"
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
                    f"Piecewise CUDA graph cannot be used with {total_num_tokens} tokens, {num_ctx_requests} context requests"
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

    def _is_encoder_decoder_model(self) -> bool:
        return bool(
            getattr(getattr(self.model, "model_config", None),
                    "is_encoder_decoder", False))

    def _get_top_level_model(self) -> Any:
        model = getattr(self.model, "_orig_mod", self.model)
        top_level_model = getattr(model, "model", model)
        return getattr(top_level_model, "_orig_mod", top_level_model)

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

        if attn_metadata.is_cuda_graph and attn_metadata.has_cross_sub_metadata:
            cross_attn_metadata = attn_metadata.update_cross_metadata(
                encoder_seq_lens=encoder_seq_lens,
                cross_kv_cache_manager=cross_kv_cache_manager,
                encoder_num_cached_tokens_per_seq=
                encoder_num_cached_tokens_per_seq,
            )
        else:
            cross_attn_metadata = attn_metadata.create_cross_metadata(
                cross_kv_cache_manager=cross_kv_cache_manager,
                encoder_seq_lens=encoder_seq_lens,
                encoder_num_cached_tokens_per_seq=
                encoder_num_cached_tokens_per_seq,
            )
            if attn_metadata.is_cuda_graph:
                attn_metadata.cross = cross_attn_metadata
        cross_attn_metadata.prepare()

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

        has_current_device_draft = next_draft_tokens_device is not None
        return (self.is_draft_model and self.model_is_wrapped) or (
            has_current_device_draft and self.has_previous_device_draft)

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
                all_rank_num_tokens = self.dist.tp_cp_allgather(
                    [spec_metadata.num_tokens,
                     len(sequence_lengths)])
                self._set_spec_metadata_all_rank_num_tokens(
                    spec_metadata, [item[0] for item in all_rank_num_tokens],
                    [item[1] for item in all_rank_num_tokens])

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
        # CRITICAL: Only extract the needed tokens based on num_tokens_per_extend_request
        # new_tokens_device shape: [batch, 1 + max_draft_len]
        # We need: [previous_batch, num_tokens_per_extend_request]
        new_tokens = new_tokens_device.transpose(
            0, 1)[previous_slots, :num_tokens_per_extend_request].flatten()
        self.input_ids_cuda[:total_num_tokens].copy_(new_tokens,
                                                     non_blocking=True)

        # Prepare draft tokens
        num_draft_tokens_per_extend_request = num_tokens_per_extend_request - 1
        self.draft_tokens_cuda[:previous_batch_draft_tokens].copy_(
            next_draft_tokens_device[
                previous_slots, :num_draft_tokens_per_extend_request].flatten(),
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
        runtime_draft_token_buffer_width = num_tokens_per_extend_request - 1

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

        previous_batch_draft_tokens = (num_extend_reqeust_wo_dummy *
                                       runtime_draft_token_buffer_width)

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
            'resource_manager': resource_manager,
        }

        if bool(lora_params):
            inputs['lora_params'] = lora_params

        if spec_metadata is not None:
            inputs['spec_metadata'] = spec_metadata

        return inputs, self.gather_ids_cuda[:num_generation_tokens]

    def _maybe_prepare_inkling_runtime(self, inputs, attn_metadata,
                                       resource_manager):
        """Eagerly publish Inkling's per-request runtime state into its STABLE
        GPU buffers BEFORE CUDA-graph capture/replay, so the captured decode
        forward performs no host->device copy:

        * short-conv pool slots -> ``conv_cache``/``conv_rt`` (the pool's
          ``state_indices`` buffer);
        * attention decode metadata (total-KV seq_lens + per-layer page table)
          -> each layer's ``InklingDecodeMeta`` buffers.

        This MUST run in both ``_prepare_tp_inputs`` (the KV-cache generation path
        used by graph capture AND every decode replay) and
        ``_prepare_tp_inputs_no_cache``. Publishing here (not inside the captured
        ``model.forward``) is what makes both graph-safe and refreshed every step;
        the model's in-forward fallbacks only cover eager, never-captured paths.
        No-op for non-Inkling models (gated on the registered conv-state manager).
        """
        if resource_manager is None:
            return
        conv_state_manager = resource_manager.get_resource_manager(
            ResourceManagerType.CONV_STATE_MANAGER)
        if conv_state_manager is None:
            return
        conv_cache, conv_rt = conv_state_manager.prepare_conv_runtime(
            attn_metadata)
        inputs['conv_cache'] = conv_cache
        inputs['conv_rt'] = conv_rt
        model = getattr(self.model, '_orig_mod', self.model)
        if hasattr(model, 'prepare_inkling_attn_decode'):
            model.prepare_inkling_attn_decode(attn_metadata)

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
            maybe_graph: bool = False):
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

        if self._can_use_incremental_update(scheduled_requests,
                                            new_tokens_device,
                                            next_draft_tokens_device):
            return self._apply_incremental_update(
                scheduled_requests, kv_cache_manager, attn_metadata,
                spec_metadata, new_tensors_device, cache_indirection_buffer,
                num_accepted_tokens_device, req_id_to_old_request,
                resource_manager)

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
        multimodal_params_list = []
        mrope_position_ids = [
        ]  # (start_idx, end_idx, (3,1,L) mrope_pos_ids) per multimodal request
        mrope_delta_write_seq_slots = []
        mrope_delta_read_seq_slots = []
        # Extra model-side cache slot reserved for CUDA graph / warmup dummy
        # requests, whose outputs are discarded.
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
            all_prompt_tokens = request.get_tokens(0)
            draft_lens.append(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            prompt_tokens = all_prompt_tokens[begin_compute:end_compute]
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
            num_cached_tokens_per_seq.append(past_seen_token_num)
            request.cached_tokens = num_cached_tokens_per_seq[-1]
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
                prompt_len=len(all_prompt_tokens),
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
                # TODO: Visit later to decide the appropriate position of sending multimodal data & selectively sending multimodal data
                multimodal_params.to_device("multimodal_data",
                                            "cuda",
                                            pin_memory=prefer_pinned(),
                                            target_keywords=getattr(
                                                self.model,
                                                "multimodal_data_device_paths",
                                                None))
                if _use_mrope:
                    mrope_config = multimodal_params.multimodal_data[
                        'mrope_config']
                    mrope_pos_ids = mrope_config['mrope_position_ids']
                    ctx_mrope_position_ids = mrope_pos_ids[:, :, begin_compute:
                                                           begin_compute +
                                                           len(prompt_tokens)]
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
            all_gen_request_ids.append(request.py_request_id)
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

        # For tree decoding, runtime_draft_len should match total tree
        # tokens (not tree depth).  py_executor resets it every iteration.
        if spec_config is not None and not spec_config.is_linear_tree:
            self.runtime_draft_len = self.max_total_draft_tokens

        # will contain previous batch indices of generation requests
        previous_batch_indices = []
        previous_pos_indices = []
        runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(
            self.runtime_draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
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
                              len(position_ids) + 1 + num_draft_tokens)))
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

                sequence_lengths.append(runtime_tokens_per_gen_step)
                num_accepted_draft_tokens.append(
                    request.py_num_accepted_draft_tokens)
                past_seen_token_num = request.max_beam_num_tokens - 1

                draft_lens.append(runtime_draft_token_buffer_width)
                gather_ids.extend(
                    list(
                        range(len(position_ids),
                              len(position_ids) + runtime_tokens_per_gen_step)))
                position_ids.extend(
                    list(
                        range(past_seen_token_num, past_seen_token_num +
                              runtime_tokens_per_gen_step)))
                # previous tensor
                previous_batch_indices.append(previous_batch_idx)
                previous_pos_indices.extend([previous_batch_idx] *
                                            runtime_tokens_per_gen_step)

                num_cached_tokens_per_seq.append(past_seen_token_num +
                                                 runtime_tokens_per_gen_step)
                request.cached_tokens = num_cached_tokens_per_seq[-1]
                if self.enable_spec_decode and spec_config.spec_dec_mode.extend_ctx(
                        self.attn_backend) and spec_config.is_linear_tree:
                    prompt_lengths.append(runtime_tokens_per_gen_step)
                else:
                    prompt_lengths.append(request.py_prompt_len)

            append_cross_attention_state(request, project_encoder_output=False)

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
            # All generation requests have the same beam width
            beam_width = generation_requests[0].py_beam_width

            # Pre-extend constant-value lists to avoid per-request append
            # overhead (saves ~3 append calls per request).
            draft_lens.extend([0] * (_n_gen * beam_width))
            sequence_lengths.extend([1] * (_n_gen * beam_width))
            num_accepted_draft_tokens.extend([0] * (_n_gen * beam_width))

            for request in generation_requests:
                request_ids.append(request.py_request_id)
                # the request has no previous tensor:
                # (1) new_tokens_device is None, which means overlap scheduler is disabled; or
                # (2) a dummy request; or
                # (3) the first step in the generation server of disaggregated serving
                if new_tokens_device is None or request.is_dummy or request.py_batch_idx is None:
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
                else:
                    # the request has previous tensor
                    # previous_batch_indices is per-request, not per-beam
                    previous_batch_indices.append(request.py_batch_idx)
                    past_seen_token_num = request.max_beam_num_tokens

                position_id = past_seen_token_num
                if _has_cp_helix:
                    # We compute a global position_id because each helix rank has only a subset of
                    # tokens for a sequence.
                    position_id = request.total_input_len_cp + request.py_decoding_iter - 1
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
                    num_cached_tokens_per_seq.append(past_seen_token_num)
                    prompt_lengths.append(request.py_prompt_len)
                    gather_ids.append(len(position_ids) - 1)

                if _use_mrope:
                    mrope_position_delta = getattr(request,
                                                   "py_mrope_position_delta",
                                                   None)
                    if mrope_position_delta is None and request.py_multimodal_data:
                        mrope_config = request.py_multimodal_data[
                            'mrope_config']
                        mrope_position_delta = mrope_config[
                            'mrope_position_deltas']
                        if mrope_position_delta.device.type == "cpu":
                            mrope_position_delta = maybe_pin_memory(
                                mrope_position_delta).to(device='cuda',
                                                         dtype=torch.int32,
                                                         non_blocking=True)
                            mrope_config[
                                'mrope_position_deltas'] = mrope_position_delta
                        request.py_mrope_position_delta = mrope_position_delta
                    if mrope_position_delta is not None:
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

        previous_batch_len = len(previous_batch_indices)

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
                # previous input ids
                previous_batch_tokens = (previous_batch_len *
                                         runtime_tokens_per_gen_step)
                new_tokens = new_tokens_device.transpose(
                    0,
                    1)[previous_slots, :runtime_tokens_per_gen_step].flatten()
                self.input_ids_cuda[num_tokens:num_tokens +
                                    previous_batch_tokens].copy_(
                                        new_tokens, non_blocking=True)

                # previous draft tokens
                previous_batch_draft_tokens = (previous_batch_len *
                                               runtime_draft_token_buffer_width)
                if runtime_draft_token_buffer_width > 0:
                    self.draft_tokens_cuda[
                        num_draft_tokens:num_draft_tokens +
                        previous_batch_draft_tokens].copy_(
                            next_draft_tokens_device[
                                previous_slots, :
                                runtime_draft_token_buffer_width].flatten(),
                            non_blocking=True)
                # prepare data for the preprocess inputs
                kv_len_offsets_device = (new_tokens_lens_device -
                                         runtime_tokens_per_gen_step)
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
                self.previous_pos_id_offsets_cuda[
                    (num_extend_reqeust_wo_dummy - previous_batch_len) *
                    runtime_tokens_per_gen_step:num_extend_reqeust_wo_dummy *
                    runtime_tokens_per_gen_step].copy_(
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

        if not attn_metadata.is_cuda_graph:
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
            num_extra_kv_tokens=get_num_extra_kv_tokens(spec_config))
        attn_metadata.kv_cache_manager = kv_cache_manager

        if hasattr(self.model.model_config.pretrained_config, 'chunk_size'):
            attn_metadata.mamba_chunk_size = self.model.model_config.pretrained_config.chunk_size
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
            scheduled_requests, attn_metadata, peft_cache_manager, maybe_graph)

        attn_all_rank_num_tokens = self._get_all_rank_num_tokens(attn_metadata)
        padded_num_tokens, can_run_piecewise_cuda_graph, attn_all_rank_num_tokens = self._get_padding_params(
            total_num_tokens, num_ctx_requests, attn_all_rank_num_tokens)
        set_per_request_piecewise_cuda_graph_flag(can_run_piecewise_cuda_graph)
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
            if isinstance(spec_metadata, Eagle3SpecMetadata):
                spec_metadata.request_accepted_path = request_accepted_path
            # No-op for non 1-model
            spec_metadata.populate_sampling_params_for_one_model(
                scheduled_requests.all_requests())
            spec_metadata.prepare()
            inputs['spec_metadata'] = spec_metadata

            if self.enable_attention_dp:
                all_rank_num_tokens = self.dist.tp_cp_allgather(
                    [spec_metadata.num_tokens,
                     len(sequence_lengths)])
                self._set_spec_metadata_all_rank_num_tokens(
                    spec_metadata, [item[0] for item in all_rank_num_tokens],
                    [item[1] for item in all_rank_num_tokens])

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
            self.has_previous_device_draft = next_draft_tokens_device is not None

        # Inkling runtime state (short-conv pool + attention decode metadata) must
        # be published here: this is the KV-cache path taken by CUDA-graph capture
        # AND every decode replay, so publishing eagerly per step keeps the
        # captured forward copy-free and the stable buffers fresh.
        self._maybe_prepare_inkling_runtime(inputs, attn_metadata,
                                            resource_manager)

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
                all_rank_num_tokens = self.dist.tp_cp_allgather([
                    attn_metadata.num_tokens, spec_metadata.num_tokens,
                    len(sequence_lengths)
                ])
                attn_metadata.all_rank_num_tokens = [
                    item[0] for item in all_rank_num_tokens
                ]
                self._set_spec_metadata_all_rank_num_tokens(
                    spec_metadata, [item[1] for item in all_rank_num_tokens],
                    [item[2] for item in all_rank_num_tokens])
            else:
                all_rank_num_tokens = self.dist.tp_cp_allgather(
                    attn_metadata.num_tokens)
                attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        return inputs, None

    def _prepare_star_attention_inputs(
            self,
            scheduled_requests: ScheduledRequests,
            kv_cache_manager,
            attn_metadata: AttentionMetadata,
            resource_manager: Optional[ResourceManager] = None):
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
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.int,
                                 pin_memory=prefer_pinned())
        self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        position_ids = torch.tensor(position_ids,
                                    dtype=torch.int,
                                    pin_memory=prefer_pinned())
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
                pin_memory=prefer_pinned(),
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

        inputs = {
            'attn_metadata': attn_metadata,
            'input_ids': self.input_ids_cuda[:num_tokens],
            'position_ids': self.position_ids_cuda[:num_tokens].unsqueeze(0),
            'inputs_embeds': None,
            'resource_manager': resource_manager,
        }
        # Models with a per-request short-conv state pool (e.g. Inkling) resolve
        # this batch's pool rows and publish them into their stable state_indices
        # CUDA buffer EAGERLY here -- before CUDA-graph capture/replay and before
        # model.forward -- so the captured forward performs no host->device slot
        # copy and each replay reads the current (padded) batch's rows. Only a
        # registered CONV_STATE_MANAGER triggers this; other models are
        # unaffected (no extra kwargs added).
        self._maybe_prepare_inkling_runtime(inputs, attn_metadata,
                                            resource_manager)
        return inputs, gather_ids if is_spec_decode else None

    def _get_lora_params_from_requests(
            self,
            scheduled_requests: ScheduledRequests,
            attn_metadata: AttentionMetadata,
            peft_cache_manager: Optional[PeftCacheManager] = None,
            maybe_graph: bool = False):
        '''
        Get LoRA parameters from scheduled requests.

        Uses CUDA Graph compatible mode in decode only batch, otherwise falls back to eager mode.

        Returns:
            Dictionary containing LoRA parameters, or None if no LoRA requests
        '''
        use_cuda_graph_mode = self.cuda_graph_lora_manager is not None and maybe_graph

        if use_cuda_graph_mode:
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
            return peft_table and self._get_eager_lora_params_from_requests(
                scheduled_requests, attn_metadata, peft_table)

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
            maybe_graph: bool = False):
        if self.mapping is not None and 'cp_type' in self.mapping.cp_config:
            cp_type = self.mapping.cp_config['cp_type']
            if CpType.STAR == cp_type:
                return self._prepare_star_attention_inputs(
                    scheduled_requests, kv_cache_manager, attn_metadata,
                    resource_manager)
            elif cp_type in (CpType.HELIX, CpType.ULYSSES):
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

        return self._prepare_tp_inputs(
            scheduled_requests, kv_cache_manager, attn_metadata, spec_metadata,
            new_tensors_device, cache_indirection_buffer,
            num_accepted_tokens_device, req_id_to_old_request, resource_manager,
            maybe_graph)

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

        Uses two distribution strategies:
        - Case A: `total >= max_seq_len + (batch_size - 1)` — one request at
          `max_seq_len` tokens, remaining tokens distributed evenly across
          the other `batch_size - 1` requests.
        - Case B: `total < max_seq_len + (batch_size - 1)` — one request of
          `total - (batch_size - 1)` tokens, the rest at 1 token each.

        Returns None for infeasible combinations (e.g., batch_size <= 0).
        """
        if batch_size <= 0 or num_tokens <= 0 or max_seq_len <= 0:
            return None

        total = min(num_tokens, batch_size * max_seq_len)

        if batch_size == 1:
            lengths = [total]
        elif total >= max_seq_len + batch_size - 1:
            # Case A
            remaining = total - max_seq_len
            base = remaining // (batch_size - 1)
            extra = remaining % (batch_size - 1)
            lengths = [max_seq_len]
            lengths += [base + 1] * extra + [base] * (batch_size - 1 - extra)
        else:
            # Case B
            first_len = total - (batch_size - 1)
            lengths = [first_len] + [1] * (batch_size - 1)

        # Sanity: every length must be >= 1.
        if any(length <= 0 for length in lengths):
            return None

        actual_num_tokens = sum(lengths)
        input_ids = [0] * actual_num_tokens

        inputs: Dict[str, Any] = {
            'input_ids': input_ids,
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
        with self.encoder_cuda_graph_runner.allow_capture():
            self._run_cuda_graph_warmup_encoder()

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

    def _run_cuda_graph_warmup_encoder(self) -> None:
        """Captures whole-model CUDA graphs for the encode-only path."""
        if not self.encoder_cuda_graph_runner.enabled:
            return

        self._capture_encoder_cuda_graphs()

    def _capture_encoder_cuda_graphs(self) -> None:
        """Capture whole-model encoder CUDA graphs for all feasible keys.

        Feasibility filter (also used in source):
          nt >= prev_sl + bs   (enough tokens for this sl bucket)
          prev_nt < bs * sl    (not enough tokens for a smaller nt bucket)
          nt <= bs * sl        (num tokens should not exceed total possible in batch)
          sl <= nt             (seq len should not exceed num tokens)
        """
        runner = self.encoder_cuda_graph_runner
        if not runner.enabled:
            return

        batch_sizes = sorted(self._cuda_graph_batch_sizes, reverse=True)
        num_tokens_list = sorted(self._cuda_graph_num_tokens)
        seq_lens_list = sorted(self._cuda_graph_seq_lens)

        num_captured = 0
        logger.info("Capturing encoder CUDA graphs ...")
        for bs in batch_sizes:
            if bs > self.batch_size:
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

                    logger.info(f"Encoder CUDA graph capture: "
                                f"bs={bs}, nt={nt}, sl={sl}")
                    self.encoder_forward(inputs)
                    torch.cuda.synchronize()
                    num_captured += 1

        logger.info(f"Captured {num_captured} encoder CUDA graph(s).")

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
        moe_load_balancer: MoeLoadBalancer = getattr(self, 'moe_load_balancer',
                                                     None)

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

                if self.encoder_cuda_graph_runner.needs_capture(key):

                    def forward_fn(
                            capture_inputs: Dict[str, Any]) -> Dict[str, Any]:
                        capture_inputs = capture_inputs.copy()
                        forward_kwargs = capture_inputs.pop("_forward_kwargs")
                        with MoeLoadBalancerIterContext(moe_load_balancer):
                            return self._forward_step(capture_inputs,
                                                      **forward_kwargs)

                    self.encoder_cuda_graph_runner.capture(
                        key, forward_fn, {
                            **model_inputs, "_forward_kwargs": forward_kwargs
                        })

                with MoeLoadBalancerIterContext(moe_load_balancer):
                    graph_outputs = self.encoder_cuda_graph_runner.replay(
                        key, {
                            **model_inputs, "_forward_kwargs": forward_kwargs
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
                spec_resource_manager, self.is_draft_model, self.attn_backend,
                self.model_is_wrapped)
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
                model_is_wrapped=self.model_is_wrapped,
                spec_metadata=spec_metadata,
                spec_tree_manager=spec_tree_manager,
                num_contexts=scheduled_requests.num_context_requests)
        else:
            spec_resource_manager = None
            spec_metadata = None

        moe_load_balancer: MoeLoadBalancer = getattr(self, 'moe_load_balancer',
                                                     None)

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
        with self.cuda_graph_runner.pad_batch(
                scheduled_requests, resource_manager,
                self.runtime_draft_len) as padded_requests:
            # Callee already no-ops when use_mrope=False, but the Python call /
            # frame setup itself is non-trivial under high concurrency. Gating
            # at the caller avoids that overhead for non-mrope models.
            if self.use_mrope:
                self._pad_batch_seed_mrope_delta_cache(padded_requests)

            # Refresh is_all_greedy_sample for the *current* batch BEFORE the
            # CUDA graph key is built below. The key includes this flag to pick
            # the argmax vs advanced-sampling graph variant; populate (inside
            # _prepare_inputs) runs later and fills the matching GPU buffers.
            # Without this pre-scan the key would use the previous iteration's
            # stale value and could replay the advanced graph against
            # unpopulated (greedy) buffers, hanging the run (e.g. MTP nextn>=2).
            if spec_metadata is not None:
                spec_metadata.update_is_all_greedy_sample(
                    padded_requests.all_requests())

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

            # Fill slot-ID buffer for scatter inside draft loop
            if (self.enable_spec_decode and spec_tree_manager is not None
                    and spec_tree_manager.use_dynamic_tree
                    and not self.is_draft_model):
                spec_tree_manager.slot_storage.fill_all_slot_ids(
                    padded_requests.context_requests,
                    padded_requests.generation_requests,
                )

            inputs, gather_ids = self._prepare_inputs(
                padded_requests, kv_cache_manager, attn_metadata, spec_metadata,
                new_tensors_device, cache_indirection_buffer,
                num_accepted_tokens_device, req_id_to_old_request,
                resource_manager, can_run_graph)
            self._prepare_inputs_event = torch.cuda.Event()
            self._prepare_inputs_event.record()

            with with_shared_pool(self.cuda_graph_runner.get_graph_pool()):
                if not can_run_graph:
                    # Fallback to eager execution if graph was not used
                    with MoeLoadBalancerIterContext(moe_load_balancer):
                        outputs = self._forward_step(
                            inputs,
                            gather_ids=gather_ids,
                            gather_context_logits=gather_context_logits)
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

                        # Pre-replay: set DSA slot mappings for current batch's draft cache (fixes 2nd warmup)
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

    def _get_spec_worker(self):
        """Access the spec_worker from DecoderModelForCausalLM (one-model spec dec)."""
        return getattr(self.model, 'spec_worker', None)

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

        num_tokens = len(encoder_input_ids)
        assert num_tokens <= self.max_num_tokens, (
            f"encoder packed length ({num_tokens}) exceeds max_num_tokens "
            f"({self.max_num_tokens})")

        # Build a fresh, no-cache attention metadata for the encoder
        # pass.  We do not reuse ``self.attn_metadata`` because that
        # object is bound to the decoder's KV-cache manager.
        sparse_metadata_params = (
            self.sparse_attention_config.to_sparse_metadata_params(
                pretrained_config=self.model.model_config.pretrained_config)
            if self.sparse_attention_config is not None else None)
        encoder_attn_metadata = self.attn_backend.Metadata(
            max_num_requests=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_num_sequences=self.batch_size * self.max_beam_width,
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
        encoder_attn_metadata.num_contexts = len(encoder_requests)
        encoder_attn_metadata.max_seq_len = self.max_seq_len
        encoder_attn_metadata.request_ids = request_ids
        encoder_attn_metadata.prepare()

        encoder_input_ids_t = torch.tensor(encoder_input_ids,
                                           dtype=torch.int,
                                           pin_memory=prefer_pinned())
        encoder_position_ids_t = torch.tensor(encoder_position_ids,
                                              dtype=torch.int,
                                              pin_memory=prefer_pinned())

        inputs = {
            'encoder_input_ids':
            encoder_input_ids_t.to('cuda', non_blocking=True),
            'encoder_position_ids':
            encoder_position_ids_t.to('cuda', non_blocking=True).unsqueeze(0),
            'encoder_attn_metadata':
            encoder_attn_metadata,
            'encoder_seq_lens':
            sequence_lengths,
            'resource_manager':
            resource_manager,
        }
        return inputs

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
            inputs = self._prepare_tp_inputs_encoder(
                encoder_requests, resource_manager=resource_manager)
            encoder_hidden_states = self._forward_step_encoder(inputs)

        return encoder_hidden_states, inputs['encoder_seq_lens']

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

        logits_tensor[logits_row_offset:logits_row_offset +
                      beam_width] = logits_rows.view(beam_width, -1)

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
                else:
                    beam_width = request.get_beam_width_by_iter(
                        for_next_iteration=False)

                logits_processors = getattr(request,
                                            "py_logits_post_processors", None)
                if logits_processors:
                    token_ids = ([request.get_tokens(0)]
                                 if is_context_request else [
                                     request.get_tokens(beam_idx)
                                     for beam_idx in range(beam_width)
                                 ])
                    if (is_context_request
                            and request.py_orig_prompt_len < len(token_ids[0])):
                        # Skip as we only need to apply logit processor on the last context request
                        logits_row_offset += beam_width
                        continue

                    self._apply_logits_processors(request, logits_processors,
                                                  logits_tensor, beam_width,
                                                  token_ids, logits_row_offset)
                logits_row_offset += beam_width

    def wait_for_input_copy(self):
        """
        Wait for input preparation and H2D copy of previous iteration before modifying host input,
        otherwise the input of previous iteration will be overwritten.
        """
        if self._prepare_inputs_event is not None:
            self._prepare_inputs_event.synchronize()
