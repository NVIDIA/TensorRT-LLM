# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
                    Union)

import torch
import torch._dynamo.config

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._utils import release_gc
from tensorrt_llm.inputs.registry import (BaseMultimodalInputProcessor,
                                          create_input_processor)
from tensorrt_llm.llmapi.llm_args import (CudaGraphConfig, DecodingBaseConfig,
                                          EncodeCudaGraphConfig,
                                          PrefillCudaGraphBackend,
                                          TorchCompileConfig, TorchLlmArgs)

# isort: split
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention.backends.interface import (AttentionMetadata,
                                            AttentionRuntimeFeatures)
from ..attention.backends.utils import get_attention_backend
from ..compilation.backend import Backend
from ..distributed import Distributed
from ..distributed.communicator import init_pp_comm
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.modeling_multimodal_mixin import MultimodalModelMixin
from ..models.modeling_utils import DecoderModelForCausalLM, timing_metric
from ..moe.expert_statistic import ExpertStatistic
from ..moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from ..route_capture import ROUTE_CAPTURE_ATTR, RouteCapture
from ..speculative import SpecMetadata, update_spec_config_from_loaded_model
from ..speculative.utils import get_static_draft_len
from ..utils import get_per_request_prefill_cuda_graph_flag, set_torch_compiling
from .config_utils import is_hybrid_linear
from .cuda_graph_runner import CUDAGraphRunner
from .engine.cuda_graph import filter_cuda_graph_batch_sizes
from .engine.lora import make_lora_model_config
from .engine.model_call import ModelCaller
from .engine.multimodal import (MultimodalItemScheduler, is_multimodal,
                                mm_encoder_cache_enabled,
                                setup_mm_encoder_attn_metadata)
from .engine.runners import resolve_runner_type
from .engine.runners.common import (make_scheduled_inputs,
                                    resolve_mrope_position_deltas_cache,
                                    set_moe_a2a_warmup)
from .engine.runners.decoder import DecoderRunner, DecoderRunnerConfig
from .engine.runners.encoder import EncoderRunner, EncoderRunnerConfig
from .engine.runners.encoder_decoder import (EncoderDecoderRunner,
                                             EncoderDecoderRunnerConfig,
                                             EncoderStageConfig)
from .engine.runners.interface import (ModelRunner, PackedInputs,
                                       PackedModelRunner, ScheduledInputs,
                                       ScheduledModelRunner)
from .engine.runners.no_kv_cache import NoKVCacheRunner, NoKVCacheRunnerConfig
from .engine.runners.pooling import PoolingRunner
from .guided_decoder import CapturableGuidedDecoder
from .layerwise_nvtx_marker import LayerwiseNvtxMarker
from .llm_request import LlmRequest
from .model_loader import ModelLoader, _construct_checkpoint_loader
from .resource_manager import ResourceManager, ResourceManagerType
from .sampler import SampleStateTensors
from .scheduler import ScheduledRequests


class _PrefillCompiledModel(torch.nn.Module):
    """Share weights between eligible prefill/mixed and original eager paths.

    The prefill flag includes the all-rank attention-DP decision and capture
    ceiling. A decode-only rank must still compile when another rank prefills.
    """

    def __init__(self, eager_model: torch.nn.Module,
                 compiled_model: torch.nn.Module) -> None:
        """Keep eager and compiled entry points sharing the same model weights."""
        super().__init__()
        self.eager_model = eager_model
        # The compiled callable references the same weights. Register only the
        # eager tree so state_dict(), children() and _apply() visit it once.
        object.__setattr__(self, "compiled_model", compiled_model)

    def named_modules(
        self,
        memo: Optional[set[torch.nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, torch.nn.Module]]:
        """Expose checkpoint-compatible module names for partial weight reloads."""
        # Weight reloads match checkpoint prefixes against this traversal.
        # Hide the eager_model prefix even with remove_duplicate=False.
        yield from self.eager_model.named_modules(memo, prefix,
                                                  remove_duplicate)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Use the compiled path only for globally eligible prefill batches."""
        model = (self.compiled_model
                 if get_per_request_prefill_cuda_graph_flag() else
                 self.eager_model)
        return model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate model-specific attributes to the original eager model."""
        # Epilogues can access transformer attributes after forward returns.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("eager_model"), name)


class ModelEngine(ABC):

    @abstractmethod
    def get_max_num_sequences(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self,
                scheduled_requests: ScheduledRequests,
                resource_manager: Optional[ResourceManager],
                new_tensors_device: Optional[SampleStateTensors],
                cache_indirection_buffer: Optional[torch.Tensor] = None):
        raise NotImplementedError

    def warmup(self, resource_manager: Optional[ResourceManager]) -> None:
        """
        This method is called after runtime resources are initialized. The
        resource manager is absent for drivers that allocate none. Override to
        perform warmup actions: instantiating CUDA graphs, torch.compile, etc.
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
        model: Optional[torch.nn.Module] = None,
        checkpoint_loader: Optional[BaseCheckpointLoader] = None,
        model_weights_memory_tag: Optional[str] = None,
        model_weights_restore_mode=None,
    ):
        _configure_deep_gemm_pdl()

        self._metrics: dict[str, float] = defaultdict(float)
        self._cleanup_done = False
        self._model_caller: Optional[ModelCaller] = None
        self._runner: Optional[ModelRunner] = None
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
                llm_args.checkpoint_loader,
                llm_args.checkpoint_format,
                mx_config=llm_args.mx_config,
                checkpoint_io_policy=llm_args.checkpoint_io_policy,
                load_format=llm_args.load_format,
                partial_model_loading=llm_args.is_partial_model_loading,
            )

        self.mapping = mapping
        if mapping.has_pp():
            init_pp_comm(mapping)
        from ._util import (compute_max_num_sequences,
                            resolved_kv_cache_manager_is_v2,
                            should_enable_adp_dummy_fixes,
                            should_enable_non_overlap_adp_forward_intent,
                            should_enable_overlap_headroom,
                            should_enable_scheduler_aware_adp_dummy)
        self._enable_adp_dummy_fixes = should_enable_adp_dummy_fixes(mapping)
        self.dist = dist
        self.llm_args = llm_args
        if dist is not None:
            ExpertStatistic.create(self.dist.rank)
        # Opt-in tiered sampling captured into the forward graph. Off by
        # default: it captures one extra graph per enabled tier, which costs
        # startup time and memory that deployments not bound by sampling
        # overhead should not pay.
        self.enable_in_graph_sampling = bool(
            getattr(llm_args, "enable_in_graph_sampling", False))
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

        self.spec_config = spec_config
        self.is_spec_decode = spec_config is not None
        self.sparse_attention_config = llm_args.sparse_attention_config
        self.enable_spec_decode = self.is_spec_decode

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
            lora_config: Optional[LoraConfig] = llm_args.lora_config
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
        # In-graph sampling needs the full vocabulary: top-k / top-p over a
        # tensor-parallel shard would rank against a slice of the logits and
        # emit the wrong token, silently. The LM head gathers by default, but
        # the draft models of some speculation modes turn that off, so refuse
        # the fast path rather than sample a sharded row.
        if self.enable_in_graph_sampling and not getattr(
                self.model.model_config, "lm_head_gather_output", True):
            logger.warning(
                "Disabling enable_in_graph_sampling: this model's LM head does not "
                "gather its output, so the logits are sharded across tensor "
                "parallel ranks and cannot be sampled in-graph.")
            self.enable_in_graph_sampling = False
        pretrained_config = self.model.model_config.pretrained_config
        model_type = getattr(pretrained_config, "model_type", None)
        self._enable_scheduler_aware_adp_dummy = (
            should_enable_scheduler_aware_adp_dummy(
                model_type, mapping, llm_args.disable_overlap_scheduler))
        self._enable_non_overlap_adp_forward_intent = (
            should_enable_non_overlap_adp_forward_intent(
                mapping, llm_args.disable_overlap_scheduler))
        self._enable_overlap_headroom = should_enable_overlap_headroom(
            mapping,
            llm_args.disable_overlap_scheduler,
            kv_cache_manager_is_v2=resolved_kv_cache_manager_is_v2(
                llm_args.kv_cache_config, self.max_beam_width),
            is_hybrid=is_hybrid_linear(pretrained_config),
            has_mrope_delta_cache=resolve_mrope_position_deltas_cache(
                self.model) is not None)
        self.max_num_seq_slots = compute_max_num_sequences(
            mapping,
            self.batch_size,
            llm_args.disable_overlap_scheduler,
            enable_overlap_headroom=self._enable_overlap_headroom,
        )
        self.sparse_attention_config = self.model.model_config.sparse_attention_config
        # In case that some tests use stub models and override `_load_model`.
        if not hasattr(self.model, 'extra_attrs'):
            self.model.extra_attrs = {}
        # Router Replay (R3) capturer owned by this engine (None when disabled or
        # for draft engines). Registered in the model extra attrs so the MoE
        # routing hook reaches this engine's capturer during its forward, the
        # same way ``moe_layers`` is looked up -- no process-wide state.
        self.route_capture = RouteCapture.create(
            rank=self.dist.rank if self.dist is not None else 0,
            model_engine=self,
            enabled=self.llm_args.enable_return_routed_experts,
            pp_size=self.mapping.pp_size,
            is_spec_decode=self.is_spec_decode)
        if self.route_capture is not None:
            self.model.extra_attrs[ROUTE_CAPTURE_ATTR] = self.route_capture
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
        self._is_encode_only = self.llm_args.encode_only

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

        self._cuda_graph_padding_enabled = cuda_graph_padding_enabled

        decode_tokens_per_request = 1 + self.original_max_total_draft_tokens
        self._cuda_graph_batch_sizes = filter_cuda_graph_batch_sizes(
            cuda_graph_batch_sizes, self.batch_size, self.max_num_tokens,
            decode_tokens_per_request,
            self._cuda_graph_padding_enabled) if cuda_graph_batch_sizes else []

        self._max_cuda_graph_batch_size = (self._cuda_graph_batch_sizes[-1] if
                                           self._cuda_graph_batch_sizes else 0)

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
        self._torch_compile_prefill_only = False

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
                    eager_model = self.model.model
                    compiled_model = torch.compile(
                        eager_model,
                        backend=self._torch_compile_backend,
                        fullgraph=torch_compile_fullgraph)
                    self._torch_compile_prefill_only = (
                        self._torch_compile_piecewise_cuda_graph
                        and not self.model.use_fx_for_pcg_fallback)
                    self.model.model = (
                        _PrefillCompiledModel(eager_model, compiled_model)
                        if self._torch_compile_prefill_only else compiled_model)
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

        sparse_params = (self.sparse_attention_config.to_sparse_params(
            pretrained_config=self.model.model_config.pretrained_config)
                         if self.sparse_attention_config is not None else None)
        self.attn_backend = get_attention_backend(self.llm_args.attn_backend,
                                                  sparse_params=sparse_params)

        self.get_runtime_tokens_per_gen_step = spec_config.get_runtime_tokens_per_gen_step if spec_config is not None else lambda runtime_draft_len: 1

        runner_cls = resolve_runner_type(self.model, self.llm_args)
        if self.is_spec_decode and issubclass(runner_cls, NoKVCacheRunner):
            raise ValueError(
                f"{runner_cls.__name__} does not support speculative decoding; "
                "pooling and multimodal encoder models must not set "
                "speculative_config.")

        if self.is_spec_decode:
            update_spec_config_from_loaded_model(self.spec_config, self.model)
            self.without_logits = self.spec_config.spec_dec_mode.without_logits(
            )
            self.max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
            self.max_draft_len = spec_config.max_draft_len
            # Mutable per-iteration draft length (updated each iteration when
            # dynamic draft length is enabled; otherwise stays fixed).  Tree
            # modes verify all tree nodes per step, which can be wider than the
            # tree depth used by the drafter loop.
            self.runtime_draft_len = get_static_draft_len(self.spec_config)

        else:
            self.without_logits = False
            self.max_draft_len = 0
            self.runtime_draft_len = 0
            self.max_total_draft_tokens = 0

        # Memos for engines without a decoder runner; see the properties.
        self._attn_metadata = None
        self._spec_metadata = None
        self.iter_states = {}

        # We look up this key in resource_manager during forward to find the
        # kv cache manager. Can be changed to support multiple model engines
        # with different KV cache managers.
        self.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
        self.lora_model_config: Optional[LoraModelConfig] = None
        self._warmup_timer = _WarmupTimer(self.mapping.rank)

        self._model_caller = ModelCaller(
            self.model,
            compile_backend=self._torch_compile_backend,
            aux_streams=(self.backend_num_streams
                         if self._torch_compile_backend is not None else None),
            prefill_compile_only=self._torch_compile_prefill_only,
        )
        self._runner = self._initialize_runner(runner_cls)

        self.kv_cache_dtype_byte_size = self.get_kv_cache_dtype_byte_size()

    def _initialize_runner(self, runner_cls: Type[ModelRunner]) -> ModelRunner:
        assert self._model_caller is not None
        if issubclass(runner_cls, EncoderRunner):
            return self._initialize_encoder_runner(runner_cls)
        if issubclass(runner_cls, EncoderDecoderRunner):
            return self._initialize_encoder_decoder_runner(runner_cls)
        if issubclass(runner_cls, NoKVCacheRunner):
            return self._initialize_no_kv_cache_runner(runner_cls)
        if issubclass(runner_cls, DecoderRunner):
            return self._initialize_decoder_runner(runner_cls)
        raise TypeError(f"No runner initializer registered for "
                        f"{runner_cls.__module__}.{runner_cls.__qualname__}")

    def _initialize_encoder_runner(
            self, runner_cls: Type[EncoderRunner]) -> EncoderRunner:
        runner_config = EncoderRunnerConfig.create(
            model=self.model,
            mapping=self.mapping,
            graph_config=(self.cuda_graph_config if isinstance(
                self.cuda_graph_config, EncodeCudaGraphConfig) else None),
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            max_beam_width=self.max_beam_width,
            without_logits=self.without_logits,
            attention_backend=self.attn_backend,
            attention_runtime_features=self.attn_runtime_features,
            enable_autotuner=self.llm_args.enable_autotuner,
        )
        return runner_cls(
            self.model,
            runner_config,
            mapping=self.mapping,
            dist=self.dist,
            moe_load_balancer=self.moe_load_balancer,
            model_caller=self._model_caller,
        )

    def _initialize_encoder_decoder_runner(
            self,
            runner_cls: Type[EncoderDecoderRunner]) -> EncoderDecoderRunner:
        encoder_config = EncoderStageConfig.create(
            model=self.model,
            mapping=self.mapping,
            graph_config=self.llm_args.encoder_cuda_graph_config,
            max_batch_size=self.encoder_batch_size,
            max_num_tokens=self.encoder_max_num_tokens,
            max_seq_len=self.max_seq_len,
            max_beam_width=self.max_beam_width,
            without_logits=self.without_logits,
            attention_backend=self.attn_backend,
            attention_runtime_features=self.attn_runtime_features,
            enable_autotuner=self.llm_args.enable_autotuner,
            is_encoder_decoder=True,
        )
        runner_config = self._decoder_runner_config(
            EncoderDecoderRunnerConfig,
            enable_encoder_decoder_mixed_cuda_graph=(
                self.llm_args.enable_encoder_decoder_mixed_cuda_graph),
        )
        return self._initialize_decoder_runner(runner_cls,
                                               runner_config,
                                               encoder_config=encoder_config)

    def _initialize_no_kv_cache_runner(
            self, runner_cls: Type[NoKVCacheRunner]) -> NoKVCacheRunner:
        runner_config = NoKVCacheRunnerConfig(
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            max_beam_width=self.max_beam_width,
            without_logits=self.without_logits,
            enable_attention_dp=self.enable_attention_dp,
            prefill_cuda_graph_backend=self.prefill_cuda_graph_backend,
            prefill_cuda_graph_num_tokens=self._prefill_cuda_graph_num_tokens,
            attention_backend=self.attn_backend,
            attention_runtime_features=self.attn_runtime_features,
            mm_encoder_cache_enabled=self._mm_encoder_cache_enabled,
        )
        if issubclass(runner_cls, PoolingRunner):
            return runner_cls(
                self.model,
                runner_config,
                mapping=self.mapping,
                dist=self.dist,
                moe_load_balancer=self.moe_load_balancer,
                model_caller=self._model_caller,
            )
        return runner_cls(
            self.model,
            runner_config,
            mapping=self.mapping,
            dist=self.dist,
            moe_load_balancer=self.moe_load_balancer,
        )

    def _decoder_runner_config(
            self,
            config_cls: Type[DecoderRunnerConfig] = DecoderRunnerConfig,
            **extra_fields: Any) -> DecoderRunnerConfig:
        return config_cls(
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            max_beam_width=self.max_beam_width,
            without_logits=self.without_logits,
            attention_backend=self.attn_backend,
            attention_runtime_features=self.attn_runtime_features,
            dtype=self.dtype,
            enable_attention_dp=self.enable_attention_dp,
            disable_overlap_scheduler=self._disable_overlap_scheduler,
            is_encode_only=self._is_encode_only,
            is_spec_decode=self.is_spec_decode,
            max_draft_len=self.max_draft_len,
            max_total_draft_tokens=self.max_total_draft_tokens,
            max_draft_loop_tokens=self.max_draft_loop_tokens,
            spec_config=self.spec_config,
            num_seq_slots=self.max_num_seq_slots,
            original_max_draft_len=self.original_max_draft_len,
            original_max_total_draft_tokens=(
                self.original_max_total_draft_tokens),
            spec_dec_max_total_draft_tokens=(
                self._spec_dec_max_total_draft_tokens),
            cuda_graph_config=self.cuda_graph_config,
            cuda_graph_batch_sizes=self._cuda_graph_batch_sizes,
            cuda_graph_padding_enabled=self._cuda_graph_padding_enabled,
            max_cuda_graph_batch_size=self._max_cuda_graph_batch_size,
            prefill_cuda_graph_backend=self.prefill_cuda_graph_backend,
            prefill_cuda_graph_num_tokens=self._prefill_cuda_graph_num_tokens,
            enable_in_graph_sampling=self.enable_in_graph_sampling,
            torch_compile_enabled=self._torch_compile_enabled,
            torch_compile_piecewise_cuda_graph=(
                self._torch_compile_piecewise_cuda_graph),
            torch_compile_prefill_only=self._torch_compile_prefill_only,
            use_mrope=self.use_mrope,
            is_multimodal=self.is_multimodal,
            mm_encoder_cache_enabled=self._mm_encoder_cache_enabled,
            enable_autotuner=self.llm_args.enable_autotuner,
            cuda_graph_specialize_lora=(
                self.llm_args.lora_config is not None
                and self.llm_args.lora_config.cuda_graph_specialize_lora),
            **extra_fields,
        )

    def _initialize_decoder_runner(
            self,
            runner_cls: Type[DecoderRunner],
            runner_config: Optional[DecoderRunnerConfig] = None,
            **runner_kwargs: Any) -> DecoderRunner:
        assert self._model_caller is not None
        if runner_config is None:
            runner_config = self._decoder_runner_config()
        return runner_cls(
            self.model,
            runner_config,
            input_processor=self.input_processor,
            model_caller=self._model_caller,
            mapping=self.mapping,
            dist=self.dist,
            moe_load_balancer=self.moe_load_balancer,
            sparse_attention_config=self.sparse_attention_config,
            torch_compile_backend=self._torch_compile_backend,
            get_runtime_tokens_per_gen_step=self.
            get_runtime_tokens_per_gen_step,
            warmup_timer=self._warmup_timer,
            iter_states=self.iter_states,
            metrics=self._metrics,
            kv_cache_manager_key=self.kv_cache_manager_key,
            **runner_kwargs,
        )

    @property
    def attn_metadata(self) -> Optional[AttentionMetadata]:
        if isinstance(self._runner, DecoderRunner):
            return self._runner.attn_metadata
        return self._attn_metadata

    @attn_metadata.setter
    def attn_metadata(self, value: Optional[AttentionMetadata]) -> None:
        if isinstance(self._runner, DecoderRunner):
            self._runner.attn_metadata = value
        else:
            self._attn_metadata = value

    @property
    def spec_metadata(self) -> Optional[SpecMetadata]:
        if isinstance(self._runner, DecoderRunner):
            return self._runner.spec_metadata
        return self._spec_metadata

    @spec_metadata.setter
    def spec_metadata(self, value: Optional[SpecMetadata]) -> None:
        if isinstance(self._runner, DecoderRunner):
            self._runner.spec_metadata = value
        else:
            self._spec_metadata = value

    @property
    def cuda_graph_runner(self) -> Optional[CUDAGraphRunner]:
        # PyExecutor suspends the padding dummies around a KV pool rebalance.
        if not isinstance(self._runner, DecoderRunner):
            return None
        return self._runner.cuda_graph_runner

    @property
    def _dspark_confidence_enabled(self) -> bool:
        return isinstance(
            self._runner,
            DecoderRunner) and self._runner._dspark_confidence_enabled

    @property
    def _dspark_trims_submitted_tokens(self) -> bool:
        return isinstance(
            self._runner,
            DecoderRunner) and self._runner._dspark_trims_submitted_tokens

    @property
    def _dspark_sps_cost_table(self):
        if not isinstance(self._runner, DecoderRunner):
            return None
        return self._runner._dspark_sps_cost_table

    @property
    def _dspark_exact_candidate_cells(self):
        if not isinstance(self._runner, DecoderRunner):
            return ()
        return self._runner._dspark_exact_candidate_cells

    @property
    def _dspark_exact_identity_words(self):
        if not isinstance(self._runner, DecoderRunner):
            return (0, ) * 8
        return self._runner._dspark_exact_identity_words

    @property
    def _dspark_device_budget(self):
        if not isinstance(self._runner, DecoderRunner):
            return None
        return getattr(self._runner, "_dspark_device_budget", None)

    @_dspark_device_budget.setter
    def _dspark_device_budget(self, value) -> None:
        if isinstance(self._runner, DecoderRunner):
            self._runner._dspark_device_budget = value

    def ragged_verify_token_buckets(self, batch_size: int):
        if not isinstance(self._runner, DecoderRunner):
            return ()
        return self._runner.ragged_verify_token_buckets(batch_size)

    def fit_ragged_verify_lens(self, *args, **kwargs):
        assert isinstance(self._runner, DecoderRunner)
        return self._runner.fit_ragged_verify_lens(*args, **kwargs)

    def _get_spec_worker(self):
        if isinstance(self._runner, DecoderRunner):
            return self._runner._get_spec_worker()
        return None

    @property
    def metrics(self) -> dict[str, float]:
        """Return model-engine warmup time metrics."""
        return self._metrics

    def register_forward_pass_callable(self, callable: Callable):
        if isinstance(self._runner, DecoderRunner):
            self._runner.forward_pass_callable = callable

    def register_sample_in_graph_callable(self, callable: Optional[Callable]):
        """Register the hook that samples at the tail of the forward graph."""
        if isinstance(self._runner, DecoderRunner):
            self._runner.sample_in_graph_callable = callable

    def register_sample_type_resolver(self,
                                      resolver: Optional[Callable],
                                      stage: Optional[Callable] = None):
        """Register how a batch maps to its sampling tier, for the graph key."""
        assert isinstance(self._runner, DecoderRunner)
        self._runner.register_sample_type_resolver(resolver, stage)

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
        # Called by `_util.py` after model loading, before graph-manager initialization.
        self.lora_model_config = make_lora_model_config(
            self.model, lora_target_modules, trtllm_modules_to_hf_modules,
            swap_gate_up_proj_lora_b_weight)
        if isinstance(self._runner, DecoderRunner):
            self._runner.lora_model_config = self.lora_model_config

    def _init_cuda_graph_lora_manager(self, lora_config: LoraConfig):
        """Build LoRA preparation state for the current executor resources."""
        if isinstance(self._runner, DecoderRunner):
            self._runner.init_cuda_graph_lora_manager(lora_config)

    def set_guided_decoder(self,
                           guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.model, "set_guided_decoder"):
            success = self.model.set_guided_decoder(guided_decoder)
            if success and isinstance(self._runner, DecoderRunner):
                self._runner.guided_decoder = guided_decoder
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
        set_moe_a2a_warmup(value)

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
        def wrapper(self,
                    resource_manager: Optional[ResourceManager] = None,
                    *args,
                    **kwargs):
            with timing_metric("total_warmup_seconds", self._metrics):
                result = method(self, resource_manager, *args, **kwargs)
                with timing_metric("kv_cache_cleanup_seconds", self._metrics):
                    kv_cache_manager = (resource_manager.get_resource_manager(
                        self.kv_cache_manager_key) if resource_manager
                                        is not None else None)
                    if kv_cache_manager is not None:
                        has_invalid_values = kv_cache_manager.check_invalid_values_in_kv_cache(
                            fill_with_zero=True)
                        if has_invalid_values:
                            logger.warning(
                                "NaNs/Infs have been introduced to KVCache during warmup, KVCache was filled with zeros to avoid potential issues"
                            )
            return result

        return wrapper

    @with_warmup_flag
    def _warmup_encoder_cuda_graphs_enc_dec(
        self,
        resource_manager: ResourceManager,
    ) -> None:
        if not self._is_encoder_decoder_model():
            return
        if not isinstance(self._runner, EncoderDecoderRunner):
            raise RuntimeError(
                "Encoder-decoder model did not initialize a model runner.")
        self._runner.warmup_encoder(resource_manager)

    def _get_encoder_cuda_graph_batch_sizes(
            self, max_batch_size: int) -> tuple[int, ...]:
        """Use startup settings while encoder scheduling remains in PyExecutor."""
        if not isinstance(self._runner, EncoderDecoderRunner):
            return ()
        return self._runner.encoder_graph_batch_sizes(max_batch_size)

    def forward_encoder(
        self,
        encoder_requests: List[LlmRequest],
        resource_manager: Optional[ResourceManager] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        if not isinstance(self._runner, EncoderDecoderRunner):
            raise RuntimeError(
                "Encoder phase requires an initialized encoder-decoder model runner."
            )
        assert resource_manager is not None, (
            "the encoder phase requires a resource manager")
        scheduled_requests = ScheduledRequests()
        scheduled_requests.encoder_requests = list(encoder_requests)
        outputs = self._runner.forward_encoder(
            ScheduledInputs(batch=scheduled_requests),
            resource_manager=resource_manager,
            is_dummy=self.is_warmup,
        )
        return (
            outputs["encoder_hidden_states"],
            outputs["encoder_seq_lens"],
        )

    @with_warmup_flag
    @warmup_with_kv_cache_cleanup
    def warmup(self,
               resource_manager: Optional[ResourceManager] = None) -> None:
        """Run model warmup and record its total wall-clock duration."""
        self._warmup_impl(resource_manager)

    def _warmup_impl(self,
                     resource_manager: Optional[ResourceManager] = None
                     ) -> None:
        """
        Orchestrates the warmup process by calling specialized warmup methods for
        torch.compile, the autotuner, and CUDA graphs.
        """
        if isinstance(self._runner, PackedModelRunner):
            self._runner.warmup()
            return
        assert resource_manager is not None, (
            "scheduled warmup requires a resource manager")
        assert isinstance(self._runner, ScheduledModelRunner)
        self._runner.warmup(resource_manager)

    ### Helper methods promoted from the original warmup method ###

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

    def cleanup(self) -> None:
        """Release resources owned by this model engine.

        Tears down, in order:

        1. The optional ``ModelLoader`` (which in turn releases any
           GMS client; see :meth:`ModelLoader.cleanup`).
        2. Runner resources and engine-owned CUDA Graph captures.
        3. The runner, model caller, MM item scheduler, and model references.
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

        # Release runner-owned graphs before dropping the runner. Keep the
        # handle available if graph release fails and cleanup is retried.
        self._release_cuda_graphs()

        # The runner, caller and scheduler retain the model, so
        # clearing the engine's attribute alone would leave the weights
        # reachable past `release_gc()` below.
        self._runner = None
        self._model_caller = None
        self._mm_item_scheduler = None
        self.model = None

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

        # Vision encoders may expose their own sequence-length inference.
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

    def _release_cuda_graphs(self) -> None:
        if self._runner is not None:
            self._runner.release_graphs()
        if self._torch_compile_backend is not None:
            self._torch_compile_backend.clear_piecewise_cuda_graphs()

    def get_max_num_sequences(self) -> int:
        """
        Return the maximum number of sequences that the model supports. PyExecutor needs this to compute max_num_active_requests
        """
        num_batches = self.mapping.pp_size
        return num_batches * self.batch_size

    def _is_encoder_decoder_model(self) -> bool:
        return bool(
            getattr(getattr(self.model, "model_config", None),
                    "is_encoder_decoder", False))

    def forward(self,
                batch: Union[ScheduledRequests, PackedInputs],
                resource_manager: Optional[ResourceManager] = None,
                new_tensors_device: Optional[SampleStateTensors] = None,
                cache_indirection_buffer: Optional[torch.Tensor] = None):
        if isinstance(batch, PackedInputs):
            assert isinstance(self._runner, PackedModelRunner), (
                "a packed batch requires a packed-batch runner")
            return self._runner.forward(batch)
        assert resource_manager is not None, (
            "scheduled execution requires a resource manager")
        inputs = make_scheduled_inputs(
            batch,
            new_tensors_device,
            cache_indirection_buffer,
            enable_spec_decode=self.enable_spec_decode,
            runtime_draft_len=self.runtime_draft_len)
        # Executor memory profiling establishes this flag. Padding requests in
        # a serving batch do not make the pass dummy.
        outputs = self._forward_scheduled(
            inputs,
            resource_manager=resource_manager,
            is_dummy=self.is_warmup,
        )
        if isinstance(outputs, dict):
            self.runtime_draft_len = outputs.pop("runtime_draft_len",
                                                 inputs.runtime_draft_len)
        return outputs

    def _forward_scheduled(
        self,
        inputs: ScheduledInputs,
        *,
        resource_manager: ResourceManager,
        is_dummy: bool = False,
    ) -> Any:
        assert isinstance(self._runner, ScheduledModelRunner), (
            "scheduled execution requires a scheduled runner")
        return self._runner.forward(
            inputs,
            resource_manager=resource_manager,
            is_dummy=is_dummy,
        )

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

    def wait_for_input_copy(self):
        """
        Wait for input preparation and H2D copy of previous iteration before modifying host input,
        otherwise the input of previous iteration will be overwritten.
        """
        if self._runner is not None:
            self._runner.wait_for_input_copy()
