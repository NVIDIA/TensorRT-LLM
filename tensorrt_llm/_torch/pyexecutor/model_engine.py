# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import (Any, Callable, Dict, Hashable, Iterator, List, Optional,
                    Tuple, Type, Union)

import torch
import torch._dynamo.config

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._torch.peft.lora.ownership import DecoderLoraOwner
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._utils import mpi_disabled, release_gc
from tensorrt_llm.inputs.registry import (BaseMultimodalInputProcessor,
                                          create_input_processor)
from tensorrt_llm.llmapi.llm_args import (DecodingBaseConfig,
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
from ..distributed.ops import MNNVLAllReduce
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.modeling_multimodal_encoder import MultimodalEncoderMixin
from ..models.modeling_multimodal_mixin import (MultimodalEncoderContractError,
                                                MultimodalModelMixin,
                                                _build_request_multimodal_input)
from ..models.modeling_multimodal_utils import filter_mm_token_from_input_ids
from ..models.modeling_speculative import SpecDecOneEngineForCausalLM
from ..models.modeling_utils import DecoderModelForCausalLM
from ..modules.fused_moe.moe_load_balancer import (MoeLoadBalancer,
                                                   MoeLoadBalancerIterContext)
from ..modules.mamba.mamba2_metadata import Mamba2Metadata
from ..peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from ..speculative import (SpecMetadata, get_draft_kv_cache_manager,
                           get_num_extra_kv_tokens, get_spec_metadata,
                           prepare_attn_metadata_for_draft_replay,
                           restore_attn_metadata_after_draft_replay,
                           update_spec_config_from_loaded_model)
from ..speculative.drafting_loops import BaseDraftingLoopWrapper
from ..speculative.eagle3 import Eagle3ResourceManager, Eagle3SpecMetadata
from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from ..tensor_lru_cache import TensorLRUCache
from ..utils import set_torch_compiling
from .config_utils import is_hybrid_linear
from .cuda_graph_runner import CUDAGraphRunner
from .engine.cuda_graph import plan_cuda_graphs
from .engine.lora import resolve_lora_owner
from .engine.model_call import ModelCaller
from .engine.multimodal import (MultimodalItemScheduler, is_multimodal,
                                mm_encoder_cache_enabled,
                                setup_mm_encoder_attn_metadata,
                                setup_mm_encoder_cuda_graphs, use_mrope)
from .engine.runners import resolve_runner_type
from .engine.runners.common import (make_scheduled_inputs,
                                    resolve_mrope_position_deltas_cache)
from .engine.runners.decoder import DecoderRunner, DecoderRunnerConfig
from .engine.runners.decoder.runner import PrefillCompiledModel
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

# Refit only needs the compile wrapper removed before weights are loaded: the
# wrapper inserts "_orig_mod" into every parameter path, and load_weights()
# matches checkpoint tensors by dotted path (modeling_utils.py:1141-1160). With
# allow_partial_loading=True the whole compiled subtree then silently loads
# nothing and keeps its pre-refit weights.
#
# Bisected on a TP8 Qwen3.5 397B refit against a real-weight reference; only the
# unwrap changes the outcome, so the refit lifecycle is unwrap -> load ->
# re-wrap and nothing else. The measurements, for anyone tempted to add the
# other steps back:
#   unwrap + compiler.reset() + recompile + recapture   clean 6.5 s
#   unwrap + reset + recompile, no recapture            clean 5.7 s
#   unwrap + recompile                                  clean 5.6 s
#   unwrap + re-wrap only                    (shipped)  clean 5.1 s
#   no unwrap                                           GARBAGE
# Clearing and recapturing the piecewise graphs is the part that was dropped:
# refit moves no tensor, so the captures stay valid and PWCG stays hot across a
# refit (976 graphs measured still live) instead of going cold.

_DEEP_GEMM_PDL_CONFIGURED = False


def _configure_deep_gemm_pdl() -> None:
    global _DEEP_GEMM_PDL_CONFIGURED
    if _DEEP_GEMM_PDL_CONFIGURED:
        return

    from tensorrt_llm import deep_gemm

    deep_gemm.set_pdl(os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1")
    _DEEP_GEMM_PDL_CONFIGURED = True


class ModelEngine:
    """Own model setup and coordinate runner execution and resource lifetimes."""

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
        # cleanup() also runs on an engine whose construction failed part-way.
        self._cleanup_done = False
        self.model_loader: Optional[ModelLoader] = None
        self._runner: Optional[ModelRunner] = None
        self._torch_compile_backend: Optional[Backend] = None
        configured_encoder_max_num_tokens = self._init_from_args(
            model_path=model_path,
            llm_args=llm_args,
            mapping=mapping,
            attn_runtime_features=attn_runtime_features,
            dist=dist,
            spec_config=spec_config,
        )
        self._load_model(
            model_path=model_path,
            model=model,
            checkpoint_loader=checkpoint_loader,
            model_weights_memory_tag=model_weights_memory_tag,
            model_weights_restore_mode=model_weights_restore_mode,
        )
        self._init_from_model()
        self._size_cuda_graphs()
        self._compile_model(configured_encoder_max_num_tokens)
        self._init_execution_services()

    def _init_from_args(
        self,
        *,
        model_path: str,
        llm_args: TorchLlmArgs,
        mapping: Optional[Mapping],
        attn_runtime_features: Optional[AttentionRuntimeFeatures],
        dist: Optional[Distributed],
        spec_config: Optional[DecodingBaseConfig],
    ) -> Optional[int]:
        """Resolve the requested sizes, the original speculation widths and the
        services derived from the arguments; capacity may still be unset.
        Returns the encoder token budget before any multimodal increase."""
        _configure_deep_gemm_pdl()

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
        configured_encoder_max_num_tokens = (
            llm_args.encoder_max_num_tokens if llm_args.encoder_max_num_tokens
            is not None else self.max_num_tokens)
        self.encoder_max_num_tokens = configured_encoder_max_num_tokens

        self.mapping = mapping
        if mapping.has_pp():
            init_pp_comm(mapping)
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
        return configured_encoder_max_num_tokens

    def _load_model(
        self,
        *,
        model_path: str,
        model: Optional[torch.nn.Module],
        checkpoint_loader: Optional[BaseCheckpointLoader],
        model_weights_memory_tag: Optional[str],
        model_weights_restore_mode,
    ) -> None:
        """Resolve the checkpoint loader, then load the model, or take the given
        one, at the requested capacity."""
        if checkpoint_loader is None:
            checkpoint_loader = _construct_checkpoint_loader(
                self.llm_args.checkpoint_loader,
                self.llm_args.checkpoint_format,
                mx_config=self.llm_args.mx_config,
                checkpoint_io_policy=self.llm_args.checkpoint_io_policy,
                load_format=self.llm_args.load_format,
                partial_model_loading=self.llm_args.is_partial_model_loading,
            )
        self.moe_load_balancer: Optional[MoeLoadBalancer] = None
        if model is None:
            lora_config: Optional[LoraConfig] = self.llm_args.lora_config
            # Keep the model_loader to support reloading the model weights later
            self.model_loader = ModelLoader(
                llm_args=self.llm_args,
                mapping=self.mapping,
                spec_config=self.spec_config,
                sparse_attention_config=self.llm_args.sparse_attention_config,
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

    def _init_from_model(self) -> None:
        """Copy the loaded model's facts onto the engine, rewriting the requested
        capacity, encoder token budget and in-graph sampling flag from them,
        and leave the model in its final pre-compile form."""
        from ._util import (compute_max_num_sequences,
                            resolved_kv_cache_manager_is_v2,
                            should_enable_overlap_headroom)

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
        self.enable_overlap_headroom = should_enable_overlap_headroom(
            self.mapping,
            self.llm_args.disable_overlap_scheduler,
            kv_cache_manager_is_v2=resolved_kv_cache_manager_is_v2(
                self.llm_args.kv_cache_config,
                self.max_beam_width,
                enable_sleep=self.llm_args.sleep_config is not None),
            is_hybrid=is_hybrid_linear(pretrained_config),
            has_mrope_delta_cache=resolve_mrope_position_deltas_cache(
                self.model) is not None)
        self.max_num_seq_slots = compute_max_num_sequences(
            self.mapping,
            self.batch_size,
            self.llm_args.disable_overlap_scheduler,
            enable_overlap_headroom=self.enable_overlap_headroom,
        )
        # AutoModelForCausalLM sets extra_attrs; a model passed in through
        # `model=`, such as a test stub, may not have it.
        if not hasattr(self.model, 'extra_attrs'):
            self.model.extra_attrs = {}
        # Router Replay (R3) capturer owned by this engine (None when disabled).
        # Registered in the model extra attrs so the MoE
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
        # Read by py_executor.py and _util.py.
        self.mm_encoder_item_scheduling_enabled = mm_item_scheduler is not None
        self.mm_encoder_output_budget_bytes: Optional[int] = None
        self.bytes_per_mm_encoder_embedding: Optional[int] = None
        if mm_item_scheduler is not None:
            # The raised encoder token budget is read back off the engine by
            # `_util.py`, and sizes the encoder metadata set up below.
            self.encoder_max_num_tokens = mm_item_scheduler.encoder_max_num_tokens
            self.mm_encoder_output_budget_bytes = mm_item_scheduler.output_budget_bytes
            self.bytes_per_mm_encoder_embedding = mm_item_scheduler.bytes_per_embedding
        if isinstance(self.model,
                      MultimodalModelMixin) and self.mapping.is_first_pp_rank():
            multimodal_config = self.model.model_config.multimodal_config
            reuse_capacity_bytes = (multimodal_config.encoder_cache_max_bytes
                                    if self.model.encoder_cache_active else 0)
            cache_capacity_bytes = max(
                self.mm_encoder_output_budget_bytes or 0,
                reuse_capacity_bytes,
            )
            self.model._initialize_multimodal_encoder_cache(
                cache_capacity_bytes)
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
        self._init_model_capacity()

    def _size_cuda_graphs(self) -> None:
        """Size the decode and prefill CUDA graphs from the clamped capacity."""
        self.cuda_graph_config = self.llm_args.cuda_graph_config

        if (isinstance(self.cuda_graph_config, EncodeCudaGraphConfig)
                and self.model.model_config.is_encoder_decoder):
            logger.warning(
                "EncodeCudaGraphConfig is not supported for encoder-decoder "
                "models through cuda_graph_config. Use DecodeCudaGraphConfig "
                "for cuda_graph_config and configure encoder graphs through "
                "encoder_cuda_graph_config. Decoder CUDA graphs will be "
                "disabled.")
            self.cuda_graph_config = None

        plan = plan_cuda_graphs(
            self.cuda_graph_config,
            self.llm_args.prefill_capture_num_tokens,
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            decode_tokens_per_request=1 + self.original_max_total_draft_tokens,
        )
        self._cuda_graph_padding_enabled = plan.padding_enabled
        self._cuda_graph_batch_sizes = plan.batch_sizes
        self.max_cuda_graph_batch_size = plan.max_batch_size
        self._prefill_cuda_graph_num_tokens = plan.prefill_num_tokens

    def _compile_model(
            self, configured_encoder_max_num_tokens: Optional[int]) -> None:
        """Compile the model, or install the eager NCCL window tensor scopes,
        capture the local multimodal encoders' CUDA graphs, and wrap the result
        in the model caller."""
        torch_compile_config = self.llm_args.torch_compile_config
        prefill_cuda_graph_backend = self.llm_args.prefill_cuda_graph_backend
        torch_compile_enabled = bool(torch_compile_config is not None)
        torch_compile_fullgraph = torch_compile_config.enable_fullgraph if torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_fullgraph'].default
        compile_only_piecewise_graphs = (
            torch_compile_config.compile_only_piecewise_graphs
            if torch_compile_config is not None else TorchCompileConfig.
            model_fields['compile_only_piecewise_graphs'].default)
        torch_compile_inductor_enabled = torch_compile_config.enable_inductor if torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_inductor'].default
        torch_compile_piecewise_cuda_graph = (
            prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE)
        torch_compile_enable_userbuffers = torch_compile_config.enable_userbuffers if torch_compile_config is not None else TorchCompileConfig.model_fields[
            'enable_userbuffers'].default
        torch_compile_max_num_streams = torch_compile_config.max_num_streams if torch_compile_config is not None else TorchCompileConfig.model_fields[
            'max_num_streams'].default

        self._torch_compile_prefill_only = False

        backend_num_streams = None
        try:
            use_ub_for_nccl = (
                self.llm_args.allreduce_strategy == "NCCL_SYMMETRIC"
                and self._init_userbuffers(self.model.config.hidden_size))
            self._nccl_window_tensor_scope_hooks = []
            if torch_compile_enabled:
                set_torch_compiling(True)
                if mpi_disabled() and self.mapping.world_size > 1:
                    # Resolve the device-mesh dims the compiled forward reads
                    # (mapping.tp_group & friends) while still eager: the
                    # DeviceMesh resolver is compiler-disabled and must never
                    # run under tracing; once memoized these reads trace as
                    # constants.
                    _ = self.mapping.tp_group
                    _ = self.mapping.pp_group
                    _ = self.mapping.cp_group
                    if self.mapping.moe_ep_size > 1:
                        _ = self.mapping.moe_tp_group
                        _ = self.mapping.moe_ep_group
                # torch.compile traces AllReduce.forward, whose lazy MNNVL
                # workspace rescale cannot allocate inside the traced region -
                # grow the workspaces to the engine maximum up front.
                num_mnnvl_allreduces = 0
                for module in self.model.modules():
                    if isinstance(module, MNNVLAllReduce):
                        module.prescale_workspace(self.max_num_tokens,
                                                  self.model.config.hidden_size)
                        num_mnnvl_allreduces += 1
                if num_mnnvl_allreduces:
                    logger.info(f"Pre-scaled the MNNVL allreduce workspace for "
                                f"{num_mnnvl_allreduces} modules to cover "
                                f"max_num_tokens={self.max_num_tokens} before "
                                "torch.compile.")
                use_ub = not use_ub_for_nccl and (
                    torch_compile_enable_userbuffers
                    and self._init_userbuffers(self.model.config.hidden_size))
                backend_num_streams = Backend.Streams([
                    torch.cuda.Stream()
                    for _ in range(torch_compile_max_num_streams - 1)
                ])
                self._torch_compile_backend = Backend(
                    torch_compile_inductor_enabled,
                    enable_userbuffers=use_ub,
                    enable_piecewise_cuda_graph=(
                        torch_compile_piecewise_cuda_graph),
                    capture_num_tokens=self._prefill_cuda_graph_num_tokens,
                    max_num_streams=torch_compile_max_num_streams,
                    mapping=self.mapping)
                if compile_only_piecewise_graphs and not isinstance(
                        self.model, DecoderModelForCausalLM):
                    raise ValueError(
                        "TorchCompileConfig."
                        "compile_only_piecewise_graphs=True is "
                        "only supported for DecoderModelForCausalLM models.")

                self._apply_torch_compile(self._torch_compile_backend,
                                          torch_compile_fullgraph)
                torch._dynamo.config.cache_size_limit = 16
            else:
                set_torch_compiling(False)
                self._nccl_window_tensor_scope_hooks = (
                    install_eager_nccl_window_tensor_scopes(self.model))

            # Capture local encoders after their metadata is sized and before
            # KV-cache estimation, so graph memory is included in the profile.
            mm_config = self.llm_args.multimodal_config
            setup_mm_encoder_cuda_graphs(
                self.model,
                mm_config.encoder_cuda_graph if mm_config is not None else None,
                max_num_tokens=configured_encoder_max_num_tokens)
        except Exception as e:
            import traceback
            traceback.print_exception(Exception, e, e.__traceback__)
            raise e
        self._model_caller = ModelCaller(
            self.model,
            compile_backend=self._torch_compile_backend,
            aux_streams=backend_num_streams,
            prefill_compile_only=self._torch_compile_prefill_only,
        )

    def _init_execution_services(self) -> None:
        """Build the attention backend, update the speculation config from the
        loaded model, and build the runner around the model caller."""
        # Start outside warmup: load-balancer statistics on, serving MoE
        # communication timeouts.
        self._warmup_depth = 0
        self._set_warmup_effects(False)

        sparse_attention_config = self.model.model_config.sparse_attention_config
        sparse_params = (sparse_attention_config.to_sparse_params(
            pretrained_config=self.model.model_config.pretrained_config)
                         if sparse_attention_config is not None else None)
        self.attn_backend = get_attention_backend(self.llm_args.attn_backend,
                                                  sparse_params=sparse_params)

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
            self.max_total_draft_tokens = self.spec_config.tokens_per_gen_step - 1
            self.max_draft_len = self.spec_config.max_draft_len
        else:
            self.without_logits = False
            self.max_draft_len = 0
            self.max_total_draft_tokens = 0

        self._metrics: dict[str, float] = defaultdict(float)
        self.iter_states: dict[str, Any] = {}
        self.lora_model_config: Optional[LoraModelConfig] = None
        self._lora_owner: Optional[DecoderLoraOwner] = None
        self._lora_model: Optional[torch.nn.Module] = None
        self._warmup_timer = _WarmupTimer(self.mapping.rank)

        self._runner = self._initialize_runner(runner_cls)

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

        # Cache for enc-dec cross-attention stable generation steps.
        # Populated on the first CUDA-graph generation step; cleared whenever
        # the batch composition changes (new encoder request arrives).
        self._cross_attn_stable_cached_tokens: Optional[List[int]] = None
        self._cross_attn_stable_request_ids: Optional[List[int]] = None

    def register_forward_pass_callable(self, callable: Callable):
        self.forward_pass_callable = callable

    def get_speculative_model(self) -> Optional[SpecDecOneEngineForCausalLM]:
        """Return the one-engine speculative model, including inside a VLM."""
        if not self.is_spec_decode:
            return None
        model = self.model
        if isinstance(model, MultimodalModelMixin):
            model = model.language_model
        if not isinstance(model, SpecDecOneEngineForCausalLM):
            # Kimi's legacy multimodal wrapper exposes its decoder as llm
            # without inheriting MultimodalModelMixin.
            model = getattr(model, "llm", None)
        return model if isinstance(model, SpecDecOneEngineForCausalLM) else None

    def get_kv_cache_dtype_byte_size(self) -> float:
        """
        Returns the size (in bytes) occupied by kv cache type.
        """
        layer_quant_mode = self.model.model_config.quant_config.layer_quant_mode
        if layer_quant_mode.has_fp4_kv_cache():
            self.kv_cache_dtype_byte_size = 1 / 2
        elif (layer_quant_mode.has_fp8_kv_cache()
              or layer_quant_mode.has_int8_kv_cache()):
            self.kv_cache_dtype_byte_size = 1
        else:
            self.kv_cache_dtype_byte_size = 2

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
            enable_autotuner=self.llm_args.enable_autotuner,
        )
        return runner_cls(
            self.model,
            runner_config,
            mapping=self.mapping,
            dist=self.dist,
            moe_load_balancer=self.moe_load_balancer,
            model_caller=self._model_caller,
            attention_runtime_features=self.attn_runtime_features,
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
            enable_autotuner=self.llm_args.enable_autotuner,
            is_encoder_decoder=True,
        )
        runner_config = self._decoder_runner_config(
            EncoderDecoderRunnerConfig,
            enable_encoder_decoder_mixed_cuda_graph=(
                self.llm_args.enable_encoder_decoder_mixed_cuda_graph),
        )
        return self._initialize_decoder_runner(
            runner_cls,
            runner_config,
            encoder_config=encoder_config,
            input_processor=self.input_processor)

    def _initialize_no_kv_cache_runner(
            self, runner_cls: Type[NoKVCacheRunner]) -> NoKVCacheRunner:
        runner_config = NoKVCacheRunnerConfig(
            max_batch_size=self.batch_size,
            max_num_tokens=self.max_num_tokens,
            max_seq_len=self.max_seq_len,
            max_beam_width=self.max_beam_width,
            without_logits=self.without_logits,
            enable_attention_dp=self.enable_attention_dp,
            prefill_cuda_graph_backend=self.llm_args.prefill_cuda_graph_backend,
            prefill_cuda_graph_num_tokens=self._prefill_cuda_graph_num_tokens,
            attention_backend=self.attn_backend,
            mm_encoder_cache_enabled=mm_encoder_cache_enabled(self.model),
        )
        if issubclass(runner_cls, PoolingRunner):
            return runner_cls(
                self.model,
                runner_config,
                mapping=self.mapping,
                dist=self.dist,
                moe_load_balancer=self.moe_load_balancer,
                model_caller=self._model_caller,
                attention_runtime_features=self.attn_runtime_features,
            )
        return runner_cls(
            self.model,
            runner_config,
            mapping=self.mapping,
            dist=self.dist,
            moe_load_balancer=self.moe_load_balancer,
            attention_runtime_features=self.attn_runtime_features,
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
            dtype=self.model.config.torch_dtype,
            enable_attention_dp=self.enable_attention_dp,
            disable_overlap_scheduler=self.llm_args.disable_overlap_scheduler,
            is_encode_only=self.llm_args.encode_only,
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
            max_cuda_graph_batch_size=self.max_cuda_graph_batch_size,
            prefill_cuda_graph_backend=self.llm_args.prefill_cuda_graph_backend,
            prefill_cuda_graph_num_tokens=self._prefill_cuda_graph_num_tokens,
            enable_in_graph_sampling=self.enable_in_graph_sampling,
            torch_compile_enabled=self._torch_compile_backend is not None,
            torch_compile_piecewise_cuda_graph=(
                self.llm_args.prefill_cuda_graph_backend ==
                PrefillCudaGraphBackend.PIECEWISE),
            torch_compile_prefill_only=self._torch_compile_prefill_only,
            compile_only_piecewise_graphs=(
                self.llm_args.torch_compile_config is not None and
                self.llm_args.torch_compile_config.compile_only_piecewise_graphs
            ),
            use_mrope=use_mrope(self.model),
            is_multimodal=is_multimodal(self.model, self.input_processor),
            mm_encoder_cache_enabled=mm_encoder_cache_enabled(self.model),
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
            model_caller=self._model_caller,
            mapping=self.mapping,
            dist=self.dist,
            moe_load_balancer=self.moe_load_balancer,
            sparse_attention_config=(
                self.model.model_config.sparse_attention_config),
            attention_runtime_features=self.attn_runtime_features,
            get_runtime_tokens_per_gen_step=(
                self.spec_config.get_runtime_tokens_per_gen_step if
                self.spec_config is not None else lambda runtime_draft_len: 1),
            warmup_timer=self._warmup_timer,
            iter_states=self.iter_states,
            metrics=self._metrics,
            mm_item_scheduler=self._mm_item_scheduler,
            **runner_kwargs,
        )

    @property
    def attn_metadata(self) -> Optional[AttentionMetadata]:
        if isinstance(self._runner, DecoderRunner):
            return self._runner.attn_metadata
        return None

    @property
    def spec_metadata(self) -> Optional[SpecMetadata]:
        if isinstance(self._runner, DecoderRunner):
            return self._runner.spec_metadata
        return None

    @property
    def cuda_graph_runner(self) -> Optional[CUDAGraphRunner]:
        # PyExecutor suspends the padding dummies around a KV pool rebalance.
        if not isinstance(self._runner, DecoderRunner):
            return None
        return self._runner.cuda_graph_runner

    @property
    def cuda_graph_batch_sizes(self) -> list[int]:
        """Return the resolved decode graph batch sizes."""
        return self._cuda_graph_batch_sizes

    @property
    def cuda_graph_padding_enabled(self) -> bool:
        """Return whether decode graph batches are padded to capture sizes."""
        return self._cuda_graph_padding_enabled

    @property
    def metrics(self) -> dict[str, float]:
        """Return model-engine warmup time metrics."""
        return self._metrics

    def register_forward_pass_callable(self, callable: Callable) -> None:
        """Install the KV connector hook on the decoder runner."""
        if isinstance(self._runner, DecoderRunner):
            self._runner.forward_pass_callable = callable

    def register_sample_in_graph_callable(self,
                                          callable: Optional[Callable]) -> None:
        """Register the hook that samples at the tail of the forward graph."""
        if isinstance(self._runner, DecoderRunner):
            self._runner.sample_in_graph_callable = callable

    def register_beam_histories_callable(self,
                                         callable: Optional[Callable]) -> None:
        """Register where logits post-processors get live beam histories."""
        if isinstance(self._runner, DecoderRunner):
            self._runner.start_beam_histories_callable = callable

    def register_sample_type_resolver(self,
                                      resolver: Optional[Callable],
                                      stage: Optional[Callable] = None) -> None:
        """Register how a batch maps to its sampling tier, for the graph key."""
        assert isinstance(self._runner, DecoderRunner)
        self._runner.register_sample_type_resolver(resolver, stage)

    def initialize_lora_owner(self) -> DecoderLoraOwner:
        """Bind the decoder owner once, across resource-capacity estimation passes."""
        if self._lora_owner is None:
            self._lora_owner, self._lora_model = resolve_lora_owner(
                self.model, self.input_processor)
        return self._lora_owner

    def set_lora_model_config(self, lora_model_config: LoraModelConfig) -> None:
        """Share the validated configuration with decoder execution."""
        self.lora_model_config = lora_model_config
        if isinstance(self._runner, DecoderRunner):
            self._runner.lora_model_config = self.lora_model_config

    def init_cuda_graph_lora_manager(self, lora_config: LoraConfig) -> None:
        """Build LoRA preparation state for the current executor resources."""
        if isinstance(self._runner, DecoderRunner):
            assert self._lora_model is not None
            self._runner.init_cuda_graph_lora_manager(lora_config,
                                                      self._lora_model)

    def set_guided_decoder(self,
                           guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.model, "set_guided_decoder"):
            success = self.model.set_guided_decoder(guided_decoder)
            if success and isinstance(self._runner, DecoderRunner):
                self._runner.guided_decoder = guided_decoder
            return success
        return False

    def _set_warmup_effects(self, in_warmup: bool) -> None:
        set_moe_comm_warmup(in_warmup)
        if self.moe_load_balancer is not None:
            self.moe_load_balancer.set_iter_info(
                enable_statistic=not in_warmup,
                enable_update_weights=not in_warmup)

    @contextmanager
    def warmup_scope(self) -> Iterator[None]:
        """Run the enclosed passes as warmup.

        Forwards inside the scope are dummy passes, the MoE communication
        kernels use their warmup timeouts, and the load balancer neither records
        statistics nor updates weights. Scopes nest; the outermost one switches
        the effects.
        """
        self._warmup_depth += 1
        if self._warmup_depth == 1:
            self._set_warmup_effects(True)
        try:
            yield
        finally:
            self._warmup_depth -= 1
            if self._warmup_depth == 0:
                self._set_warmup_effects(False)

    def set_warmup_purpose(self, purpose: str) -> None:
        """Label the following warmup passes in the warmup timing log."""
        self._warmup_timer.purpose = purpose

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
                        ResourceManagerType.KV_CACHE_MANAGER) if
                                        resource_manager is not None else None)
                    if kv_cache_manager is not None:
                        has_invalid_values = kv_cache_manager.check_invalid_values_in_kv_cache(
                            fill_with_zero=True)
                        if has_invalid_values:
                            logger.warning(
                                "NaNs/Infs have been introduced to KVCache during warmup, KVCache was filled with zeros to avoid potential issues"
                            )
            return result

        return wrapper

    def warmup_encoder(self, resource_manager: ResourceManager) -> None:
        """Capture the encoder CUDA graphs of an encoder-decoder runner."""
        if isinstance(self._runner, EncoderDecoderRunner):
            with self.warmup_scope():
                self._runner.warmup_encoder(resource_manager)

    def encoder_graph_batch_sizes(self, max_batch_size: int) -> tuple[int, ...]:
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
            is_dummy=self._warmup_depth > 0,
        )
        return (
            outputs["encoder_hidden_states"],
            outputs["encoder_seq_lens"],
        )

    @warmup_with_kv_cache_cleanup
    def warmup(self,
               resource_manager: Optional[ResourceManager] = None) -> None:
        """Run model warmup and record its total wall-clock duration."""
        with self.warmup_scope():
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
        """Forward selected MM encoder items into the unified output cache."""
        if not scheduled_items:
            return
        if self._mm_item_scheduler is None:
            raise TypeError(
                "Item-level MM scheduling requires MultimodalModelMixin")
        self._mm_item_scheduler.forward_items(requests, scheduled_items)

    @property
    def mm_encoder_cache(self) -> Optional[TensorLRUCache]:
        """Return the model-owned encoder-output cache used by this engine."""
        # Whole-model torch.compile wraps the model after its cache is created.
        model = getattr(self.model, "_orig_mod", self.model)
        if not isinstance(model, MultimodalModelMixin):
            return None
        if self._mm_item_scheduler is None and not model.encoder_cache_active:
            return None
        return model._multimodal_encoder_cache

    def get_mm_encoder_item_cache_keys(
            self, request: LlmRequest) -> Optional[List[Hashable]]:
        """Return stable cache keys for an item-scheduled request, when available."""
        if self._mm_item_scheduler is None:
            return None
        return self._mm_item_scheduler.item_cache_keys(request)

    def _resolve_bytes_per_mm_encoder_embedding(self) -> int:
        """Bytes occupied by one multimodal encoder output embedding.

        Prefers the mixin's explicit `embedding_dim`/`embedding_dtype`
        contract, then the text embedding layer's weight, then the
        pretrained config's hidden size with the loaded weights' dtype —
        both mixin properties are optional and most VLMs implement
        neither.
        """
        model = self.model
        try:
            return (model.embedding_dim * torch.empty(
                (), dtype=model.embedding_dtype).element_size())
        except NotImplementedError:
            pass
        try:
            weight = model.text_embedding_layer.weight
            return weight.shape[-1] * weight.element_size()
        except NotImplementedError:
            pass
        pretrained = model.model_config.pretrained_config
        hidden_size = getattr(pretrained, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(pretrained, "text_config", None),
                                  "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Cannot derive the MM embedding row size: the model "
                "implements neither embedding_dim/embedding_dtype nor "
                "text_embedding_layer, and its pretrained config exposes "
                "no (text_config.)hidden_size")
        element_size = next(model.parameters()).dtype.itemsize
        return hidden_size * element_size

    def _set_up_multimodal_encoder_attn_metadata(self) -> None:
        """Construct AttentionMetadata for any multimodal encoders inside the
        loaded model, using the engine's encoder token budget
        (`encoder_max_num_tokens`, falling back to the LLM-side
        `max_num_tokens`).

        Mirrors `_set_up_attn_metadata` for the LLM backbone: encoders opt in
        by inheriting `MultimodalEncoderMixin`, and the engine drives the construction
        so the sizes match the resolved encoder token budget rather than being
        hardcoded inside each encoder's `__init__`. The optional
        per-segment capacity combines the encoder token budget with the input
        processor's largest supported item.
        """
        max_seq_len = self.encoder_max_num_tokens
        if isinstance(self.input_processor, BaseMultimodalDummyInputsBuilder):
            max_tokens_per_item = (
                self.input_processor.get_mm_max_tokens_per_item())
            max_seq_len = max(max_seq_len,
                              max(max_tokens_per_item.values(), default=0))

        for module in self.model.modules():
            if isinstance(module, MultimodalEncoderMixin):
                setup_kwargs: Dict[str, Any] = dict(
                    max_num_tokens=self.encoder_max_num_tokens)
                if self.mm_encoder_attention_metadata_capacity is not None:
                    setup_kwargs["attention_metadata_capacity"] = (
                        self.mm_encoder_attention_metadata_capacity)
                module.setup_attn_metadata(**setup_kwargs)
                module.set_attn_max_seq_len(max_seq_len)

    def _set_up_spec_metadata(
            self,
            spec_resource_manager: Optional[BaseResourceManager],
            no_cache=False):
        spec_config = self.spec_config if self.enable_spec_decode else None
        # The disaggregated attention-DP overlap path opts into larger metadata
        # buffers. Passing None preserves the established max_num_requests
        # fallback for other configurations, including PP.
        # Size slot-indexed spec buffers (e.g. rejection-sampling draft_probs)
        # by the same pool that sizes the SeqSlotManager
        # (py_executor_creator passes model_engine.max_num_seq_slots), so every
        # py_seq_slot has a row. With max_num_requests only, DSpark + rejection
        # sampling on disaggregated attention-DP gen indexed past the buffer.
        num_seq_slots = self.max_num_seq_slots
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
        2. Runner resources and executor-owned CUDA Graph captures (via
           :meth:`_release_cuda_graphs`).
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
            uses an explicit temporary-executor shutdown and then drops its
            reference instead.
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
        # Destruction timing is not rank-symmetric; only explicit shutdown releases owners.
        self._release_cuda_graphs(release_nccl_window_owners=False)

        # The runner, caller and scheduler retain the model, so
        # clearing the engine's attribute alone would leave the weights
        # reachable past `release_gc()` below.
        self._runner = None
        self._model_caller = None
        self._mm_item_scheduler = None
        self._lora_model = None
        self._lora_owner = None
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
            logger.warning("ModelEngine cleanup failed during destruction: %s",
                           e)

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

    def _release_model_owned_cuda_graphs(
        self,
        *,
        release_nccl_window_owners: bool = True,
    ) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            for module in model.modules():
                clear_blocks_graph = getattr(module, "clear_blocks_cuda_graph",
                                             None)
                if clear_blocks_graph is not None:
                    clear_blocks_graph(
                        release_nccl_window_owners=release_nccl_window_owners)

    def _release_cuda_graphs(
        self,
        *,
        release_nccl_window_owners: bool = True,
    ) -> None:
        if self._runner is not None:
            self._runner.release_graphs(
                release_nccl_window_owners=release_nccl_window_owners)
        if self._torch_compile_backend is not None:
            self._torch_compile_backend.clear_piecewise_cuda_graphs(
                release_nccl_window_owners=release_nccl_window_owners)

    def release_executor_resources(self, *,
                                   release_model_owned_graphs: bool) -> None:
        """Release CUDA graphs and attention metadata bound to the executor's
        resource managers, before those managers shut down.

        ``release_model_owned_graphs`` also clears graphs captured inside the
        model's own modules; the KV cache estimation shutdown keeps them.
        """
        if release_model_owned_graphs:
            self._release_model_owned_cuda_graphs()
        self._release_cuda_graphs()
        if isinstance(self._runner, DecoderRunner):
            self._runner.attn_metadata = None

    def _eager_decoder_scope(self) -> torch.nn.Module:
        """Return the eager decoder scope under any compile or prefill wrapper."""
        scope = self.model.model
        if isinstance(scope, PrefillCompiledModel):
            scope = scope.eager_model
        return getattr(scope, "_orig_mod", scope)

    def _apply_torch_compile(self, backend: Backend, fullgraph: bool) -> None:
        """Compile the eager model scope."""
        apply_llm_torch_compile = getattr(self.model, "apply_llm_torch_compile",
                                          None)
        if isinstance(self.model, DecoderModelForCausalLM):
            eager_model = self._eager_decoder_scope()
            compiled_model = torch.compile(eager_model,
                                           backend=backend,
                                           fullgraph=fullgraph)
            self._torch_compile_prefill_only = (
                self.llm_args.prefill_cuda_graph_backend
                == PrefillCudaGraphBackend.PIECEWISE
                and (self.llm_args.torch_compile_config.
                     compile_only_piecewise_graphs
                     or not self.model.use_fx_for_pcg_fallback))
            self.model.model = (PrefillCompiledModel(
                eager_model, compiled_model) if self._torch_compile_prefill_only
                                else compiled_model)
        elif callable(apply_llm_torch_compile):
            # Avoid nesting a new wrapper around the previous compile.
            llm = getattr(self.model, "llm", None)
            compiled_model = getattr(llm, "model", None)
            if hasattr(compiled_model, "_orig_mod"):
                llm.model = compiled_model._orig_mod
            # TODO: Move this contract to MultimodalModelMixin once
            # multimodal models consistently expose their LLM compile scope.
            apply_llm_torch_compile(backend=backend, fullgraph=fullgraph)
        else:
            eager_model = getattr(self.model, "_orig_mod", self.model)
            self.model = torch.compile(eager_model,
                                       backend=backend,
                                       fullgraph=fullgraph)

    def _remove_torch_compile(self) -> None:
        """Restore the eager model scope."""
        if isinstance(self.model, DecoderModelForCausalLM):
            self.model.model = self._eager_decoder_scope()
            return
        apply_llm_torch_compile = getattr(self.model, "apply_llm_torch_compile",
                                          None)
        if callable(apply_llm_torch_compile):
            llm = getattr(self.model, "llm", None)
            compiled_model = getattr(llm, "model", None)
            if hasattr(compiled_model, "_orig_mod"):
                llm.model = compiled_model._orig_mod
            return
        self.model = getattr(self.model, "_orig_mod", self.model)

    def unwrap_compiled_model_for_refit(self) -> None:
        """Release compiled state before refit.

        Gated on torch.compile, NOT on piecewise CUDA graphs. PWCG is only one
        thing a compiled engine carries across a refit; the compiled wrapper
        itself has to be unwrapped either way. Gating this on
        ``_torch_compile_piecewise_cuda_graph`` left ``TorchCompileConfig()``
        (compile on, PWCG off -- the default shape of that config) with no refit
        lifecycle at all, and a TP8 Qwen3.5 397B refit in that configuration
        emits degenerate repetition while the same engine with PWCG enabled is
        clean. Nothing goes stale: the wrapper inserts ``_orig_mod`` into every
        parameter path, ``load_weights`` matches checkpoint tensors by dotted
        path, and ``allow_partial_loading=True`` then skips the whole compiled
        subtree in silence, so it simply keeps its pre-refit weights.

        The piecewise captures are deliberately NOT cleared. Refit moves no
        tensor (0 of 1405 parameters, buffers and tensor attributes change
        address), so the captures stay valid; clearing them only forces PWCG
        cold and replay-less until something recaptures.
        """
        if self._torch_compile_backend is None:
            return
        # Barrier before refit overwrites weights, so prior GPU work cannot still
        # be reading them. In the original code this sat *after* the PWCG-gated
        # early return, so a compile-without-PWCG engine skipped it entirely.
        torch.cuda.synchronize()
        self._remove_torch_compile()

    def restore_compiled_model_after_refit(
            self, resource_manager: ResourceManager) -> None:
        """Re-install the torch.compile wrapper after refit.

        Gated on torch.compile, not on PWCG -- see
        :meth:`unwrap_compiled_model_for_refit` for why.

        This ONLY re-wraps: the cached compiled artifact is reused, so there is
        no ``torch.compiler.reset()`` and no piecewise recapture. Both were
        measured unnecessary on a TP8 Qwen3.5 397B refit -- 976 piecewise graphs
        were still live afterwards, so the captures survive intact and PWCG
        stays hot.

        No ``gc.collect()`` either. The previous wrapper is released by
        reference count when it is replaced, and a full collection in a
        process holding the executor, the request bookkeeping and the
        trainer-side Ray state costs seconds; this hook sits inside the
        refit bubble, so that was pure generation-idle time.

        Deliberately NOT decorated with ``warmup_with_kv_cache_cleanup``: that
        decorator ends with ``check_invalid_values_in_kv_cache(fill_with_zero=
        True)``, which zero-fills every attention KV pool whether or not it
        found a NaN. It belongs after a capture warm-up that ran forwards on
        placeholder inputs; this hook never runs a forward, and under an
        in-flight weight update (``drain=False``) the zero-fill would wipe the
        KV of every request kept across the refit. The wrapper also ran when
        torch.compile is disabled, because the early return below happens
        inside the wrapped method.

        ``resource_manager`` is unused here; it stays in the signature because
        callers (including NeMo-RL) already pass it.
        """
        if self._torch_compile_backend is None:
            return
        with self.warmup_scope():
            self._apply_torch_compile(
                self._torch_compile_backend,
                self.llm_args.torch_compile_config.enable_fullgraph)

    # Back-compat aliases. Callers probe these names with getattr() and fall
    # back to doing nothing, so removing them would silently skip the unwrap
    # and let refit load into a wrapped model again.
    release_piecewise_cuda_graphs_for_refit = unwrap_compiled_model_for_refit
    recapture_piecewise_cuda_graphs_after_refit = restore_compiled_model_after_refit

    def get_max_num_sequences(self) -> int:
        """
        Return the maximum number of sequences that the model supports. PyExecutor needs this to compute max_num_active_requests
        """
        num_batches = self.mapping.pp_size
        return num_batches * self.batch_size

    def forward(self,
                batch: Union[ScheduledRequests, PackedInputs],
                resource_manager: Optional[ResourceManager] = None,
                new_tensors_device: Optional[SampleStateTensors] = None,
                cache_indirection_buffer: Optional[torch.Tensor] = None,
                num_accepted_tokens_device: Optional[torch.Tensor] = None,
                req_id_to_old_request: Optional[Dict[int, LlmRequest]] = None):
        if not self._disable_overlap_scheduler:
            # Do not refill reusable host staging while the previous
            # iteration's asynchronous H2D copies still consume it.  This
            # event precedes the model forward, so its synchronization retains
            # GPU forward overlap while protecting every forward entry,
            # including speculative/draft paths.
            self.wait_for_input_copy()

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
                and self.mapping.cp_size == 1):
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
                execution_requests, kv_cache_manager, attn_metadata,
                spec_metadata, new_tensors_device, cache_indirection_buffer,
                num_accepted_tokens_device, req_id_to_old_request,
                resource_manager, can_run_graph, execution_promoted_context_ids)
            speculative_model = self.get_speculative_model()
            spec_worker = (speculative_model.spec_worker
                           if speculative_model is not None else None)
            if spec_worker is not None:
                spec_worker.prepare_managed_draft_cache(
                    speculative_model.draft_model, spec_metadata, attn_metadata,
                    resource_manager)
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

            if (spec_worker is not None and not self.is_warmup
                    and not self.cuda_graph_runner.is_warmup_only):
                draft_history_update = spec_worker.snapshot_managed_draft_history(
                )
                if draft_history_update is not None:
                    # Graph output dictionaries persist across replays. Each
                    # overlapped iteration must retain its own host readback.
                    outputs = dict(outputs)
                    outputs['draft_history_update'] = draft_history_update

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
            'input_features':
            torch.cat(features, dim=0).to('cuda', non_blocking=True),
            'encoder_attn_metadata':
            encoder_attn_metadata,
            'encoder_seq_lens':
            sequence_lengths,
            'resource_manager':
            resource_manager,
        }
        return inputs

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
            is_dummy=self._warmup_depth > 0,
        )

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

        # The userbuffers bootstrap exchanges CUDA memory handles over raw
        # MPI_COMM_WORLD collectives (userbuffers-host.cpp). Under non-MPI
        # orchestrators (Ray sets TLLM_DISABLE_MPI) every worker holds a
        # 1-rank MPI world, so importer ranks would receive uninitialized
        # bytes instead of peer handles and crash in
        # cuMemImportFromShareableHandle with 'operation not supported'.
        if mpi_disabled():
            logger.info(
                "Disabling userbuffers: its bootstrap requires MPI, which is "
                "disabled under this orchestrator.")
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

    def wait_for_input_copy(self) -> None:
        """
        Wait for input preparation and H2D copy of previous iteration before modifying host input,
        otherwise the input of previous iteration will be overwritten.
        """
        if self._runner is not None:
            self._runner.wait_for_input_copy()
