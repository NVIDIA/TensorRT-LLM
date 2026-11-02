# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import math
import os
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch._dynamo.config

import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._torch.peft.lora.ownership import DecoderLoraOwner
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._utils import mpi_disabled, release_gc
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor, create_input_processor
from tensorrt_llm.llmapi.llm_args import (
    DecodingBaseConfig,
    EncodeCudaGraphConfig,
    PrefillCudaGraphBackend,
    TorchCompileConfig,
    TorchLlmArgs,
)

# isort: split
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention.backends.interface import AttentionMetadata, AttentionRuntimeFeatures
from ..attention.backends.utils import get_attention_backend
from ..compilation.backend import Backend
from ..distributed import Distributed
from ..distributed.communicator import init_pp_comm
from ..distributed.ops import MNNVLAllReduce
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.modeling_multimodal_mixin import MultimodalModelMixin
from ..models.modeling_utils import DecoderModelForCausalLM, timing_metric
from ..moe.expert_statistic import ExpertStatistic
from ..moe.fused_moe.moe_comm_timeout_guard import set_moe_comm_warmup
from ..moe.fused_moe.moe_load_balancer import BaseMoeLoadBalancer
from ..nccl_window_tensor_scope import install_eager_nccl_window_tensor_scopes
from ..route_capture import ROUTE_CAPTURE_ATTR, RouteCapture
from ..speculative import SpecMetadata, update_spec_config_from_loaded_model
from ..tensor_lru_cache import TensorLRUCache
from ..utils import set_torch_compiling
from .config_utils import is_hybrid_linear
from .cuda_graph_runner import CUDAGraphRunner
from .engine.cuda_graph import plan_cuda_graphs
from .engine.lora import resolve_lora_owner
from .engine.model_call import ModelCaller
from .engine.multimodal import (
    MultimodalItemScheduler,
    is_multimodal,
    mm_encoder_cache_enabled,
    setup_mm_encoder_attn_metadata,
    setup_mm_encoder_cuda_graphs,
    use_mrope,
)
from .engine.runners import resolve_runner_type
from .engine.runners.common import make_scheduled_inputs, resolve_mrope_position_deltas_cache
from .engine.runners.decoder import DecoderRunner, DecoderRunnerConfig
from .engine.runners.decoder.runner import PrefillCompiledModel
from .engine.runners.encoder import EncoderRunner, EncoderRunnerConfig
from .engine.runners.encoder_decoder import (
    EncoderDecoderRunner,
    EncoderDecoderRunnerConfig,
    EncoderStageConfig,
)
from .engine.runners.interface import (
    ModelRunner,
    PackedInputs,
    PackedModelRunner,
    ScheduledInputs,
    ScheduledModelRunner,
)
from .engine.runners.no_kv_cache import NoKVCacheRunner, NoKVCacheRunnerConfig
from .engine.runners.pooling import PoolingRunner
from .guided_decoder import CapturableGuidedDecoder
from .layerwise_nvtx_marker import LayerwiseNvtxMarker
from .llm_request import LlmRequest
from .model_loader import ModelLoader
from .resource_manager import ResourceManager, ResourceManagerType
from .sampler import SampleStateTensors
from .scheduler import ScheduledRequests


def _spec_decode_fast_paths(model: torch.nn.Module) -> bool:
    """``spec_decode_fast_paths`` of the one-engine speculative language model
    (the model itself, or a multimodal model's language model); False for any
    other model."""
    from ..models.modeling_speculative import SpecDecOneEngineForCausalLM

    if isinstance(model, MultimodalModelMixin):
        try:
            model = model.language_model
        except NotImplementedError:
            return False
    return isinstance(model, SpecDecOneEngineForCausalLM) and model.spec_decode_fast_paths


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
        mapping: Mapping,
        attn_runtime_features: AttentionRuntimeFeatures | None = None,
        dist: Distributed | None = None,
        spec_config: DecodingBaseConfig | None = None,
        model: torch.nn.Module | None = None,
        checkpoint_loader: BaseCheckpointLoader | None = None,
        model_weights_memory_tag: str | None = None,
        model_weights_restore_mode=None,
    ):
        # cleanup() also runs on an engine whose construction failed part-way.
        self._cleanup_done = False
        self.model_loader: ModelLoader | None = None
        self._runner: ModelRunner | None = None
        self._torch_compile_backend: Backend | None = None
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
        mapping: Mapping,
        attn_runtime_features: AttentionRuntimeFeatures | None,
        dist: Distributed | None,
        spec_config: DecodingBaseConfig | None,
    ) -> int | None:
        """Resolve the requested sizes, the original speculation widths and the
        services derived from the arguments; capacity may still be unset.
        Returns the encoder token budget before any multimodal increase."""
        _configure_deep_gemm_pdl()

        if llm_args.encode_only and llm_args.mm_encoder_only:
            raise ValueError("encode_only and mm_encoder_only are mutually exclusive.")
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
        self.encoder_batch_size = (
            llm_args.encoder_max_batch_size
            if llm_args.encoder_max_batch_size is not None
            else self.batch_size
        )
        # The multimodal encoder token budget falls back to the LLM-side value
        # when unset. It may be raised after model load because atomic MM items
        # cannot be split.
        configured_encoder_max_num_tokens = (
            llm_args.encoder_max_num_tokens
            if llm_args.encoder_max_num_tokens is not None
            else self.max_num_tokens
        )
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
        self.enable_in_graph_sampling = bool(getattr(llm_args, "enable_in_graph_sampling", False))
        self.original_max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self.original_max_total_draft_tokens = (
            (spec_config.tokens_per_gen_step - 1) if spec_config is not None else 0
        )
        # Saved before zeroing for draft models; used by update_spec_dec_param.
        self._spec_dec_max_total_draft_tokens = (
            spec_config.max_total_draft_tokens if spec_config is not None else 0
        )

        # Dynamic tree draft loop produces up to K * max_draft_len tokens,
        # which may exceed max_total_draft_tokens. Use the larger value for
        # KV cache reservation only; verify/tree output stays at max_total_draft_tokens.
        if (
            spec_config is not None
            and getattr(spec_config, "use_dynamic_tree", False)
            and getattr(spec_config, "dynamic_tree_max_topK", 0) > 0
        ):
            self.max_draft_loop_tokens = max(
                self.original_max_total_draft_tokens,
                spec_config.dynamic_tree_max_topK * spec_config.max_draft_len,
            )
        else:
            self.max_draft_loop_tokens = self.original_max_total_draft_tokens

        self.spec_config = spec_config
        self.is_spec_decode = spec_config is not None

        self.attn_runtime_features = attn_runtime_features or AttentionRuntimeFeatures()

        input_processor_kwargs = {}
        video_pruning_rate = llm_args.multimodal_config.video_pruning_rate
        if video_pruning_rate is not None:
            input_processor_kwargs["video_pruning_rate"] = video_pruning_rate
        self.input_processor = create_input_processor(
            model_path,
            tokenizer=None,
            checkpoint_format=llm_args.checkpoint_format,
            trust_remote_code=llm_args.trust_remote_code,
            **input_processor_kwargs,
        )
        return configured_encoder_max_num_tokens

    def _load_model(
        self,
        *,
        model_path: str,
        model: torch.nn.Module | None,
        checkpoint_loader: BaseCheckpointLoader | None,
        model_weights_memory_tag: str | None,
        model_weights_restore_mode,
    ) -> None:
        """Load the model, or take the given one, at the requested capacity."""
        self.moe_load_balancer: BaseMoeLoadBalancer | None = None
        if model is None:
            lora_config: LoraConfig | None = self.llm_args.lora_config
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
                checkpoint_dir=model_path, checkpoint_loader=checkpoint_loader
            )
            if isinstance(moe_load_balancer, BaseMoeLoadBalancer):
                self.moe_load_balancer = moe_load_balancer
        else:
            self.model = model
        self.spec_decode_fast_paths = _spec_decode_fast_paths(self.model)
        self._validate_breakable_cuda_graph_compatibility()

    def _init_from_model(self) -> None:
        """Copy the loaded model's facts onto the engine, rewriting the requested
        capacity, encoder token budget and in-graph sampling flag from them,
        and leave the model in its final pre-compile form."""
        from ._util import (
            compute_max_num_sequences,
            resolved_kv_cache_manager_is_v2,
            should_enable_overlap_headroom,
        )

        # In-graph sampling needs the full vocabulary: top-k / top-p over a
        # tensor-parallel shard would rank against a slice of the logits and
        # emit the wrong token, silently. The LM head gathers by default, but
        # the draft models of some speculation modes turn that off, so refuse
        # the fast path rather than sample a sharded row.
        if self.enable_in_graph_sampling and not getattr(
            self.model.model_config, "lm_head_gather_output", True
        ):
            logger.warning(
                "Disabling enable_in_graph_sampling: this model's LM head does not "
                "gather its output, so the logits are sharded across tensor "
                "parallel ranks and cannot be sampled in-graph."
            )
            self.enable_in_graph_sampling = False
        pretrained_config = self.model.model_config.pretrained_config
        self.enable_overlap_headroom = should_enable_overlap_headroom(
            self.mapping,
            self.llm_args.disable_overlap_scheduler,
            kv_cache_manager_is_v2=resolved_kv_cache_manager_is_v2(
                self.llm_args.kv_cache_config,
                self.max_beam_width,
                enable_sleep=self.llm_args.sleep_config is not None,
            ),
            is_hybrid=is_hybrid_linear(pretrained_config),
            has_mrope_delta_cache=resolve_mrope_position_deltas_cache(self.model) is not None,
        )
        self.max_num_seq_slots = compute_max_num_sequences(
            self.mapping,
            self.batch_size,
            self.llm_args.disable_overlap_scheduler,
            enable_overlap_headroom=self.enable_overlap_headroom,
        )
        # AutoModelForCausalLM sets extra_attrs; a model passed in through
        # `model=`, such as a test stub, may not have it.
        if not hasattr(self.model, "extra_attrs"):
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
            is_spec_decode=self.is_spec_decode,
        )
        if self.route_capture is not None:
            self.model.extra_attrs[ROUTE_CAPTURE_ATTR] = self.route_capture
        # Every MM item-scheduling decision -- policy, capability, feature
        # validation, budget resolution -- lives in engine/multimodal.py; the
        # engine only copies back the three budgets that are external contract.
        mm_item_scheduler = MultimodalItemScheduler.maybe_create(
            llm_args=self.llm_args,
            model=self.model,
            input_processor=self.input_processor,
            encoder_max_num_tokens=self.encoder_max_num_tokens,
        )
        self._mm_item_scheduler = mm_item_scheduler
        # Read by py_executor.py and _util.py.
        self.mm_encoder_item_scheduling_enabled = mm_item_scheduler is not None
        self.mm_encoder_output_budget_bytes: int | None = None
        self.bytes_per_mm_encoder_embedding: int | None = None
        if mm_item_scheduler is not None:
            # The raised encoder token budget is read back off the engine by
            # `_util.py`, and sizes the encoder metadata set up below.
            self.encoder_max_num_tokens = mm_item_scheduler.encoder_max_num_tokens
            self.mm_encoder_output_budget_bytes = mm_item_scheduler.output_budget_bytes
            self.bytes_per_mm_encoder_embedding = mm_item_scheduler.bytes_per_embedding
        if isinstance(self.model, MultimodalModelMixin) and self.mapping.is_first_pp_rank():
            multimodal_config = self.model.model_config.multimodal_config
            reuse_capacity_bytes = (
                multimodal_config.encoder_cache_max_bytes if self.model.encoder_cache_active else 0
            )
            cache_capacity_bytes = max(
                self.mm_encoder_output_budget_bytes or 0,
                reuse_capacity_bytes,
            )
            self.model._initialize_multimodal_encoder_cache(cache_capacity_bytes)
        setup_mm_encoder_attn_metadata(
            self.model,
            self.input_processor,
            self.encoder_max_num_tokens,
            mm_item_scheduler.attention_metadata_capacity
            if mm_item_scheduler is not None
            else None,
        )
        if self.llm_args.enable_layerwise_nvtx_marker:
            layerwise_nvtx_marker = LayerwiseNvtxMarker()
            module_prefix = "Model"
            if (
                self.model.model_config
                and self.model.model_config.pretrained_config
                and self.model.model_config.pretrained_config.architectures
            ):
                module_prefix = "|".join(self.model.model_config.pretrained_config.architectures)
            layerwise_nvtx_marker.register_hooks(self.model, module_prefix)

        self.enable_attention_dp = self.model.model_config.mapping.enable_attention_dp
        self._init_model_capacity()

    def _size_cuda_graphs(self) -> None:
        """Size the decode and prefill CUDA graphs from the clamped capacity."""
        self.cuda_graph_config = self.llm_args.cuda_graph_config

        if (
            isinstance(self.cuda_graph_config, EncodeCudaGraphConfig)
            and self.model.model_config.is_encoder_decoder
        ):
            logger.warning(
                "EncodeCudaGraphConfig is not supported for encoder-decoder "
                "models through cuda_graph_config. Use DecodeCudaGraphConfig "
                "for cuda_graph_config and configure encoder graphs through "
                "encoder_cuda_graph_config. Decoder CUDA graphs will be "
                "disabled."
            )
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

    def _compile_model(self, configured_encoder_max_num_tokens: int | None) -> None:
        """Compile the model, or install the eager NCCL window tensor scopes,
        capture the local multimodal encoders' CUDA graphs, and wrap the result
        in the model caller."""
        torch_compile_config = self.llm_args.torch_compile_config
        prefill_cuda_graph_backend = self.llm_args.prefill_cuda_graph_backend
        torch_compile_enabled = bool(torch_compile_config is not None)
        torch_compile_fullgraph = (
            torch_compile_config.enable_fullgraph
            if torch_compile_config is not None
            else TorchCompileConfig.model_fields["enable_fullgraph"].default
        )
        compile_only_piecewise_graphs = (
            torch_compile_config.compile_only_piecewise_graphs
            if torch_compile_config is not None
            else TorchCompileConfig.model_fields["compile_only_piecewise_graphs"].default
        )
        torch_compile_inductor_enabled = (
            torch_compile_config.enable_inductor
            if torch_compile_config is not None
            else TorchCompileConfig.model_fields["enable_inductor"].default
        )
        torch_compile_piecewise_cuda_graph = (
            prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
        )
        torch_compile_enable_userbuffers = (
            torch_compile_config.enable_userbuffers
            if torch_compile_config is not None
            else TorchCompileConfig.model_fields["enable_userbuffers"].default
        )
        torch_compile_max_num_streams = (
            torch_compile_config.max_num_streams
            if torch_compile_config is not None
            else TorchCompileConfig.model_fields["max_num_streams"].default
        )

        self._torch_compile_prefill_only = False

        backend_num_streams = None
        try:
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
                        module.prescale_workspace(
                            self.max_num_tokens, self.model.config.hidden_size
                        )
                        num_mnnvl_allreduces += 1
                if num_mnnvl_allreduces:
                    logger.info(
                        f"Pre-scaled the MNNVL allreduce workspace for "
                        f"{num_mnnvl_allreduces} modules to cover "
                        f"max_num_tokens={self.max_num_tokens} before "
                        "torch.compile."
                    )
                use_ub = torch_compile_enable_userbuffers and self._init_userbuffers(
                    self.model.config.hidden_size
                )
                backend_num_streams = Backend.Streams(
                    [torch.cuda.Stream() for _ in range(torch_compile_max_num_streams - 1)]
                )
                self._torch_compile_backend = Backend(
                    torch_compile_inductor_enabled,
                    enable_userbuffers=use_ub,
                    enable_piecewise_cuda_graph=(torch_compile_piecewise_cuda_graph),
                    capture_num_tokens=self._prefill_cuda_graph_num_tokens,
                    max_num_streams=torch_compile_max_num_streams,
                    mapping=self.mapping,
                )
                if compile_only_piecewise_graphs and not isinstance(
                    self.model, DecoderModelForCausalLM
                ):
                    raise ValueError(
                        "TorchCompileConfig."
                        "compile_only_piecewise_graphs=True is "
                        "only supported for DecoderModelForCausalLM models."
                    )

                self._apply_torch_compile(self._torch_compile_backend, torch_compile_fullgraph)
                torch._dynamo.config.cache_size_limit = 16
            else:
                set_torch_compiling(False)
                install_eager_nccl_window_tensor_scopes(self.model)

            # Capture local encoders after their metadata is sized and before
            # KV-cache estimation, so graph memory is included in the profile.
            mm_config = self.llm_args.multimodal_config
            setup_mm_encoder_cuda_graphs(
                self.model,
                mm_config.encoder_cuda_graph if mm_config is not None else None,
                max_num_tokens=configured_encoder_max_num_tokens,
            )
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
        sparse_params = (
            sparse_attention_config.to_sparse_params(
                pretrained_config=self.model.model_config.pretrained_config
            )
            if sparse_attention_config is not None
            else None
        )
        self.attn_backend = get_attention_backend(
            self.llm_args.attn_backend, sparse_params=sparse_params
        )

        runner_cls = resolve_runner_type(self.model, self.llm_args)
        if self.is_spec_decode and issubclass(runner_cls, NoKVCacheRunner):
            raise ValueError(
                f"{runner_cls.__name__} does not support speculative decoding; "
                "pooling and multimodal encoder models must not set "
                "speculative_config."
            )

        if self.is_spec_decode:
            update_spec_config_from_loaded_model(self.spec_config, self.model)
            self.without_logits = self.spec_config.spec_dec_mode.without_logits()
            self.max_total_draft_tokens = self.spec_config.tokens_per_gen_step - 1
            self.max_draft_len = self.spec_config.max_draft_len
        else:
            self.without_logits = False
            self.max_draft_len = 0
            self.max_total_draft_tokens = 0

        self._metrics: dict[str, float] = defaultdict(float)
        self.iter_states: dict[str, Any] = {}
        self.lora_model_config: LoraModelConfig | None = None
        self._lora_owner: DecoderLoraOwner | None = None
        self._lora_model: torch.nn.Module | None = None
        self._warmup_timer = _WarmupTimer(self.mapping.rank)

        self._runner = self._initialize_runner(runner_cls)

        layer_quant_mode = self.model.model_config.quant_config.layer_quant_mode
        if layer_quant_mode.has_fp4_kv_cache():
            self.kv_cache_dtype_byte_size = 1 / 2
        elif layer_quant_mode.has_fp8_kv_cache() or layer_quant_mode.has_int8_kv_cache():
            self.kv_cache_dtype_byte_size = 1
        else:
            self.kv_cache_dtype_byte_size = 2

    def _initialize_runner(self, runner_cls: type[ModelRunner]) -> ModelRunner:
        assert self._model_caller is not None
        if issubclass(runner_cls, EncoderRunner):
            return self._initialize_encoder_runner(runner_cls)
        if issubclass(runner_cls, EncoderDecoderRunner):
            return self._initialize_encoder_decoder_runner(runner_cls)
        if issubclass(runner_cls, NoKVCacheRunner):
            return self._initialize_no_kv_cache_runner(runner_cls)
        if issubclass(runner_cls, DecoderRunner):
            return self._initialize_decoder_runner(runner_cls)
        raise TypeError(
            f"No runner initializer registered for "
            f"{runner_cls.__module__}.{runner_cls.__qualname__}"
        )

    def _initialize_encoder_runner(self, runner_cls: type[EncoderRunner]) -> EncoderRunner:
        runner_config = EncoderRunnerConfig.create(
            model=self.model,
            mapping=self.mapping,
            graph_config=(
                self.cuda_graph_config
                if isinstance(self.cuda_graph_config, EncodeCudaGraphConfig)
                else None
            ),
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
        self, runner_cls: type[EncoderDecoderRunner]
    ) -> EncoderDecoderRunner:
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
                self.llm_args.enable_encoder_decoder_mixed_cuda_graph
            ),
        )
        return self._initialize_decoder_runner(
            runner_cls,
            runner_config,
            encoder_config=encoder_config,
            input_processor=self.input_processor,
        )

    def _initialize_no_kv_cache_runner(self, runner_cls: type[NoKVCacheRunner]) -> NoKVCacheRunner:
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
        self, config_cls: type[DecoderRunnerConfig] = DecoderRunnerConfig, **extra_fields: Any
    ) -> DecoderRunnerConfig:
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
            is_spec_decode=self.is_spec_decode,
            max_draft_len=self.max_draft_len,
            max_total_draft_tokens=self.max_total_draft_tokens,
            max_draft_loop_tokens=self.max_draft_loop_tokens,
            spec_config=self.spec_config,
            num_seq_slots=self.max_num_seq_slots,
            original_max_draft_len=self.original_max_draft_len,
            original_max_total_draft_tokens=(self.original_max_total_draft_tokens),
            spec_dec_max_total_draft_tokens=(self._spec_dec_max_total_draft_tokens),
            cuda_graph_config=self.cuda_graph_config,
            cuda_graph_batch_sizes=self._cuda_graph_batch_sizes,
            cuda_graph_padding_enabled=self._cuda_graph_padding_enabled,
            max_cuda_graph_batch_size=self.max_cuda_graph_batch_size,
            prefill_cuda_graph_backend=self.llm_args.prefill_cuda_graph_backend,
            prefill_cuda_graph_num_tokens=self._prefill_cuda_graph_num_tokens,
            enable_in_graph_sampling=self.enable_in_graph_sampling,
            torch_compile_enabled=self._torch_compile_backend is not None,
            torch_compile_piecewise_cuda_graph=(
                self.llm_args.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
            ),
            torch_compile_prefill_only=self._torch_compile_prefill_only,
            compile_only_piecewise_graphs=(
                self.llm_args.torch_compile_config is not None
                and self.llm_args.torch_compile_config.compile_only_piecewise_graphs
            ),
            use_mrope=use_mrope(self.model),
            is_multimodal=is_multimodal(self.model, self.input_processor),
            mm_encoder_cache_enabled=mm_encoder_cache_enabled(self.model),
            enable_autotuner=self.llm_args.enable_autotuner,
            cuda_graph_specialize_lora=(
                self.llm_args.lora_config is not None
                and self.llm_args.lora_config.cuda_graph_specialize_lora
            ),
            spec_decode_fast_paths=self.spec_decode_fast_paths,
            **extra_fields,
        )

    def _initialize_decoder_runner(
        self,
        runner_cls: type[DecoderRunner],
        runner_config: DecoderRunnerConfig | None = None,
        **runner_kwargs: Any,
    ) -> DecoderRunner:
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
            sparse_attention_config=(self.model.model_config.sparse_attention_config),
            attention_runtime_features=self.attn_runtime_features,
            get_runtime_tokens_per_gen_step=(
                self.spec_config.get_runtime_tokens_per_gen_step
                if self.spec_config is not None
                else lambda runtime_draft_len: 1
            ),
            warmup_timer=self._warmup_timer,
            iter_states=self.iter_states,
            metrics=self._metrics,
            mm_item_scheduler=self._mm_item_scheduler,
            **runner_kwargs,
        )

    @property
    def attn_metadata(self) -> AttentionMetadata | None:
        if isinstance(self._runner, DecoderRunner):
            return self._runner.attn_metadata
        return None

    @property
    def spec_metadata(self) -> SpecMetadata | None:
        if isinstance(self._runner, DecoderRunner):
            return self._runner.spec_metadata
        return None

    @property
    def cuda_graph_runner(self) -> CUDAGraphRunner | None:
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
    def _dspark_confidence_enabled(self) -> bool:
        return isinstance(self._runner, DecoderRunner) and self._runner._dspark_confidence_enabled

    @property
    def _dspark_trims_submitted_tokens(self) -> bool:
        return (
            isinstance(self._runner, DecoderRunner) and self._runner._dspark_trims_submitted_tokens
        )

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
            return (0,) * 8
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

    def register_forward_pass_callable(self, callable: Callable) -> None:
        """Install the KV connector hook on the decoder runner."""
        if isinstance(self._runner, DecoderRunner):
            self._runner.forward_pass_callable = callable

    def register_sample_in_graph_callable(self, callable: Callable | None) -> None:
        """Register the hook that samples at the tail of the forward graph."""
        if isinstance(self._runner, DecoderRunner):
            self._runner.sample_in_graph_callable = callable

    def register_beam_histories_callable(self, callable: Callable | None) -> None:
        """Register where logits post-processors get live beam histories."""
        if isinstance(self._runner, DecoderRunner):
            self._runner.start_beam_histories_callable = callable

    def register_sample_type_resolver(
        self, resolver: Callable | None, stage: Callable | None = None
    ) -> None:
        """Register how a batch maps to its sampling tier, for the graph key."""
        assert isinstance(self._runner, DecoderRunner)
        self._runner.register_sample_type_resolver(resolver, stage)

    def initialize_lora_owner(self) -> DecoderLoraOwner:
        """Bind the decoder owner once, across resource-capacity estimation passes."""
        if self._lora_owner is None:
            self._lora_owner, self._lora_model = resolve_lora_owner(
                self.model, self.input_processor
            )
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
            self._runner.init_cuda_graph_lora_manager(lora_config, self._lora_model)

    def drop_greedy_only_outputs(self) -> None:
        """Have every batch return logits, for an executor-side guided decoder that reads them."""
        if isinstance(self._runner, DecoderRunner):
            self._runner.keep_greedy_only_outputs = False

    def set_guided_decoder(self, guided_decoder: CapturableGuidedDecoder) -> bool:
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
                enable_statistic=not in_warmup, enable_update_weights=not in_warmup
            )

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
        def wrapper(self, resource_manager: ResourceManager | None = None, *args, **kwargs):
            with timing_metric("total_warmup_seconds", self._metrics):
                result = method(self, resource_manager, *args, **kwargs)
                with timing_metric("kv_cache_cleanup_seconds", self._metrics):
                    kv_cache_manager = (
                        resource_manager.get_resource_manager(ResourceManagerType.KV_CACHE_MANAGER)
                        if resource_manager is not None
                        else None
                    )
                    if kv_cache_manager is not None:
                        has_invalid_values = kv_cache_manager.check_invalid_values_in_kv_cache(
                            fill_with_zero=True
                        )
                        if has_invalid_values:
                            logger.warning(
                                "NaNs/Infs have been introduced to KVCache during warmup, "
                                "KVCache was filled with zeros to avoid potential issues"
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
        encoder_requests: list[LlmRequest],
        resource_manager: ResourceManager | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        if not isinstance(self._runner, EncoderDecoderRunner):
            raise RuntimeError(
                "Encoder phase requires an initialized encoder-decoder model runner."
            )
        assert resource_manager is not None, "the encoder phase requires a resource manager"
        scheduled_requests = ScheduledRequests()
        scheduled_requests.encoder_requests = list(encoder_requests)
        outputs = self._runner.forward_encoder(
            ScheduledInputs(batch=scheduled_requests),
            resource_manager=resource_manager,
        )
        return (
            outputs["encoder_hidden_states"],
            outputs["encoder_seq_lens"],
        )

    @warmup_with_kv_cache_cleanup
    def warmup(self, resource_manager: ResourceManager | None = None) -> None:
        """Run model warmup and record its total wall-clock duration."""
        with self.warmup_scope():
            self._warmup_impl(resource_manager)

    def _warmup_impl(self, resource_manager: ResourceManager | None = None) -> None:
        """
        Orchestrates the warmup process by calling specialized warmup methods for
        torch.compile, the autotuner, and CUDA graphs.
        """
        if isinstance(self._runner, PackedModelRunner):
            self._runner.warmup()
            return
        assert resource_manager is not None, "scheduled warmup requires a resource manager"
        assert isinstance(self._runner, ScheduledModelRunner)
        self._runner.warmup(resource_manager)

    ### Helper methods promoted from the original warmup method ###

    def _validate_breakable_cuda_graph_compatibility(self) -> None:
        if self.llm_args.prefill_cuda_graph_backend != PrefillCudaGraphBackend.BREAKABLE:
            return

        if isinstance(self.model, DecoderModelForCausalLM):
            return
        decoder_model = getattr(self.model, "llm", None)
        if (
            self.llm_args.disable_mm_encoder
            and isinstance(decoder_model, DecoderModelForCausalLM)
            and getattr(self.model, "mm_encoder", None) is None
        ):
            return
        if isinstance(self.model, MultimodalModelMixin) or isinstance(
            self.input_processor, BaseMultimodalInputProcessor
        ):
            raise ValueError("breakable prefill CUDA graph does not support multimodal models")

    def forward_multimodal_encoder_items(
        self,
        requests: list[LlmRequest],
        scheduled_items: dict[int, list[int]],
    ) -> None:
        """Forward selected MM encoder items into the unified output cache."""
        if not scheduled_items:
            return
        if self._mm_item_scheduler is None:
            raise TypeError("Item-level MM scheduling requires MultimodalModelMixin")
        self._mm_item_scheduler.forward_items(requests, scheduled_items)

    @property
    def mm_encoder_cache(self) -> TensorLRUCache | None:
        """Return the model-owned encoder-output cache used by this engine."""
        # Whole-model torch.compile wraps the model after its cache is created.
        model = getattr(self.model, "_orig_mod", self.model)
        if not isinstance(model, MultimodalModelMixin):
            return None
        if self._mm_item_scheduler is None and not model.encoder_cache_active:
            return None
        return model._multimodal_encoder_cache

    def get_mm_encoder_item_cache_keys(self, request: LlmRequest) -> list[Hashable] | None:
        """Return stable cache keys for an item-scheduled request, when available."""
        if self._mm_item_scheduler is None:
            return None
        return self._mm_item_scheduler.item_cache_keys(request)

    def invalidate_multimodal_encoder_cache(self) -> None:
        """Clear cached MM encoder outputs when no request is using them."""
        encoder_cache = self.mm_encoder_cache
        if encoder_cache is not None:
            encoder_cache.clear()

    def prepare_cleanup(self, *, release_nccl_window_owners: bool = True) -> None:
        """Release local graph captures before collective model teardown."""
        if self._cleanup_done:
            return
        self._release_cuda_graphs(release_nccl_window_owners=release_nccl_window_owners)

    def shutdown_moe_load_balancer(self, *, local_safe: bool) -> bool:
        """Run the manager's rank-aligned teardown preflight and collectives."""
        if self._cleanup_done:
            return True
        moe_load_balancer = getattr(self, "moe_load_balancer", None)
        if moe_load_balancer is None:
            return bool(local_safe)
        return moe_load_balancer.shutdown(local_safe=bool(local_safe)) is not False

    def finish_cleanup(self) -> None:
        """Release rank-local model resources after collective teardown."""
        if self._cleanup_done:
            return

        self._release_model_loader()
        self.moe_load_balancer = None
        # The runner, caller and scheduler retain the model, so clearing the
        # engine's attribute alone would leave the weights reachable.
        self._runner = None
        self._model_caller = None
        self._mm_item_scheduler = None
        self._lora_model = None
        self._lora_owner = None
        self.model = None
        self.input_processor = None
        release_gc()
        self._cleanup_done = True

    def _release_model_loader(self) -> None:
        # Keep the handle live until its release succeeds, so a failed cleanup
        # can be retried without releasing the loader twice.
        model_loader = self.model_loader
        if model_loader is not None:
            model_loader.cleanup()
            self.model_loader = None

    def cleanup(self, *, collective: bool = False) -> None:
        """Release model resources.

        By default this is the rank-local cleanup that Python finalizers run:
        the model loader, then graph captures, then the model references. With
        ``collective``, it first releases local graph captures on every rank,
        then passes that local result into the load-balancer's collective
        preflight, and only then drops rank-local model state; worker-owned
        terminal cleanup is responsible for those collective resources.
        """
        if self._cleanup_done:
            return

        if not collective:
            self._release_model_loader()
            # Destruction timing is not rank-symmetric; only explicit shutdown
            # releases NCCL window owners.
            self.prepare_cleanup(release_nccl_window_owners=False)
            moe_load_balancer = getattr(self, "moe_load_balancer", None)
            if moe_load_balancer is not None and not getattr(
                moe_load_balancer, "shutdown_on_finalize", False
            ):
                logger.warning(
                    "Skipping collective MoE load-balancer shutdown from Python finalization; "
                    "terminal executor shutdown must release it explicitly"
                )
            self.finish_cleanup()
            return

        prepare_error = None
        try:
            self.prepare_cleanup()
        except BaseException as error:  # Preserve rank alignment before surfacing it.
            prepare_error = error

        manager_safe = self.shutdown_moe_load_balancer(local_safe=prepare_error is None)
        if prepare_error is not None:
            raise prepare_error
        if not manager_safe:
            raise RuntimeError("MoE load-balancer collective shutdown was not safe on every rank")
        self.finish_cleanup()

    def __del__(self) -> None:
        """Best-effort rank-local cleanup during garbage collection."""
        try:
            self.cleanup(collective=False)
        except (RuntimeError, AttributeError) as e:
            logger.warning("ModelEngine cleanup failed during destruction: %s", e)

    def _init_max_seq_len(self):
        # Allow user to override the inferred max_seq_len with a warning.
        allow_long_max_model_len = os.getenv("TLLM_ALLOW_LONG_MAX_MODEL_LEN", "0").lower() in [
            "1",
            "true",
            "yes",
            "y",
        ]

        # Vision encoders may expose their own sequence-length inference.
        if hasattr(self.model, "infer_max_seq_len"):
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
        if hasattr(self.model, "model_config") and self.model.model_config:
            model_config = self.model.model_config.pretrained_config
            rope_scaling = getattr(model_config, "rope_scaling", None)
            rope_factor = 1
            if rope_scaling is not None:
                rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
                if rope_type not in ("su", "longrope", "llama3", "yarn"):
                    rope_factor = rope_scaling.get("factor", 1.0)

            # Step 1: Find the upper bound of max_seq_len
            inferred_max_seq_len = 2048
            max_position_embeddings = getattr(model_config, "max_position_embeddings", None)
            if max_position_embeddings is None and hasattr(model_config, "text_config"):
                max_position_embeddings = getattr(
                    model_config.text_config, "max_position_embeddings", None
                )
            if max_position_embeddings is not None:
                inferred_max_seq_len = max_position_embeddings

            # Step 2: Scale max_seq_len with rotary scaling
            if rope_factor != 1:
                inferred_max_seq_len = int(math.ceil(inferred_max_seq_len * rope_factor))
                logger.warning(
                    f"max_seq_len is scaled to {inferred_max_seq_len} by rope scaling {rope_factor}"
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
                clear_blocks_graph = getattr(module, "clear_blocks_cuda_graph", None)
                if clear_blocks_graph is not None:
                    clear_blocks_graph(release_nccl_window_owners=release_nccl_window_owners)

    def _release_cuda_graphs(
        self,
        *,
        release_nccl_window_owners: bool = True,
    ) -> None:
        if self._runner is not None:
            self._runner.release_graphs(release_nccl_window_owners=release_nccl_window_owners)
        if self._torch_compile_backend is not None:
            self._torch_compile_backend.clear_piecewise_cuda_graphs(
                release_nccl_window_owners=release_nccl_window_owners
            )

    def release_executor_resources(self, *, release_model_owned_graphs: bool) -> None:
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
        apply_llm_torch_compile = getattr(self.model, "apply_llm_torch_compile", None)
        if isinstance(self.model, DecoderModelForCausalLM):
            eager_model = self._eager_decoder_scope()
            compiled_model = torch.compile(eager_model, backend=backend, fullgraph=fullgraph)
            self._torch_compile_prefill_only = (
                self.llm_args.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
                and (
                    self.llm_args.torch_compile_config.compile_only_piecewise_graphs
                    or not self.model.use_fx_for_pcg_fallback
                )
            )
            self.model.model = (
                PrefillCompiledModel(eager_model, compiled_model)
                if self._torch_compile_prefill_only
                else compiled_model
            )
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
            self.model = torch.compile(eager_model, backend=backend, fullgraph=fullgraph)

    def _remove_torch_compile(self) -> None:
        """Restore the eager model scope."""
        if isinstance(self.model, DecoderModelForCausalLM):
            self.model.model = self._eager_decoder_scope()
            return
        apply_llm_torch_compile = getattr(self.model, "apply_llm_torch_compile", None)
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

    def restore_compiled_model_after_refit(self, resource_manager: ResourceManager) -> None:
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
                self._torch_compile_backend, self.llm_args.torch_compile_config.enable_fullgraph
            )

    # Back-compat aliases. Callers probe these names with getattr() and fall
    # back to doing nothing, so removing them would silently skip the unwrap
    # and let refit load into a wrapped model again.
    release_piecewise_cuda_graphs_for_refit = unwrap_compiled_model_for_refit
    recapture_piecewise_cuda_graphs_after_refit = restore_compiled_model_after_refit

    def get_max_num_sequences(self) -> int:
        """
        Return the maximum number of sequences that the model supports.
        PyExecutor needs this to compute max_num_active_requests
        """
        num_batches = self.mapping.pp_size
        return num_batches * self.batch_size

    def forward(
        self,
        batch: ScheduledRequests | PackedInputs,
        resource_manager: ResourceManager | None = None,
        new_tensors_device: SampleStateTensors | None = None,
        cache_indirection_buffer: torch.Tensor | None = None,
        *,
        enable_spec_decode: bool = False,
        runtime_draft_len: int = 0,
    ):
        if isinstance(batch, PackedInputs):
            assert isinstance(self._runner, PackedModelRunner), (
                "a packed batch requires a packed-batch runner"
            )
            return self._runner.forward(batch)
        assert resource_manager is not None, "scheduled execution requires a resource manager"
        inputs = make_scheduled_inputs(
            batch,
            new_tensors_device,
            cache_indirection_buffer,
            enable_spec_decode=enable_spec_decode,
            runtime_draft_len=runtime_draft_len,
        )
        # Only a warmup scope makes the pass dummy; padding requests in a
        # serving batch do not.
        return self._forward_scheduled(
            inputs,
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
            "scheduled execution requires a scheduled runner"
        )
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
                "disabled under this orchestrator."
            )
            return False

        # Disable UB for unsupported platforms
        if not ub.ub_supported():
            return False
        # NCCL_SYMMETRIC strategy no longer requires UserBuffer allocator initialization.
        # It uses NCCLWindowAllocator from ncclUtils directly.
        if self.llm_args.allreduce_strategy == "NCCL_SYMMETRIC":
            # Skip UB initialization for NCCL_SYMMETRIC - it uses NCCLWindowAllocator directly
            return False
        ub.initialize_userbuffers_manager(
            self.mapping.tp_size,
            self.mapping.pp_size,
            self.mapping.cp_size,
            self.mapping.rank,
            self.mapping.gpus_per_node,
            hidden_size * self.max_num_tokens * 2,
        )

        return True

    def wait_for_input_copy(self) -> None:
        """
        Wait for input preparation and H2D copy of previous iteration before modifying host input,
        otherwise the input of previous iteration will be overwritten.
        """
        if self._runner is not None:
            self._runner.wait_for_input_copy()
