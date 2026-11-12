# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._torch.peft.lora.ownership import DecoderLoraOwner
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._utils import release_gc
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor, create_input_processor
from tensorrt_llm.llmapi.llm_args import (
    DecodingBaseConfig,
    EncodeCudaGraphConfig,
    PrefillCudaGraphBackend,
    TorchLlmArgs,
)

# isort: split
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention.backends.interface import AttentionMetadata, AttentionRuntimeFeatures
from ..compilation.backend import Backend
from ..distributed import Distributed
from ..models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from ..models.modeling_multimodal_mixin import MultimodalModelMixin
from ..models.modeling_utils import DecoderModelForCausalLM, timing_metric
from ..moe.fused_moe.moe_comm_timeout_guard import set_moe_comm_warmup
from ..moe.fused_moe.moe_load_balancer import BaseMoeLoadBalancer
from ..route_capture import ROUTE_CAPTURE_ATTR, RouteCapture
from ..speculative import SpecMetadata, update_spec_config_from_loaded_model
from ..tensor_lru_cache import TensorLRUCache
from .config_utils import is_hybrid_linear
from .cuda_graph_runner import CUDAGraphRunner
from .engine.cuda_graph import plan_cuda_graphs
from .engine.lora import resolve_lora_owner
from .engine.model_setup import (
    apply_torch_compile,
    compile_model,
    create_model_loader,
    init_process_setup,
    prepare_loaded_model,
    remove_torch_compile,
    resolve_attention_backend,
    resolve_max_num_tokens,
    resolve_max_seq_len,
)
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
from .engine.runners.encoder import EncoderConfig, EncoderRunner, resolve_scheduled_encoder_args
from .engine.runners.encoder_decoder import EncoderDecoderRunner, EncoderDecoderRunnerConfig
from .engine.runners.interface import ModelRunner, ScheduledInputs, ScheduledModelRunner
from .engine.runners.mm_encoder import MultimodalEncoderRunner
from .guided_decoder import CapturableGuidedDecoder
from .llm_request import LlmRequest
from .model_loader import ModelLoader
from .resource_manager import ResourceManager, ResourceManagerType
from .sampler import SampleStateTensors
from .scheduler import ScheduledRequests


def _spec_decode_fast_paths(model: torch.nn.Module) -> bool:
    """``spec_decode_fast_paths`` of the language model (the model itself, or a
    multimodal model's language model): the one-engine speculative language
    model's, or that of a model that builds its own speculative branch, such as
    a Modeling V2 target; False for a model that does not set it."""
    if isinstance(model, MultimodalModelMixin):
        try:
            model = model.language_model
        except NotImplementedError:
            return False
    return bool(getattr(model, "spec_decode_fast_paths", False))


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
        defer_checkpoint: bool = False,
    ):
        """``defer_checkpoint``: build the engine on dummy weights for a KernelBuilder
        rehearsal, where the loader can (``ModelLoader.load``); ``has_deferred_checkpoint``
        then tells, and the caller runs ``rehearse`` and ``load_deferred_checkpoint``."""
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
            defer_checkpoint=defer_checkpoint,
        )
        if self.has_deferred_checkpoint and not self.model.model_config.is_generation:
            # No KV cache to rehearse with.
            self.model_loader.load_deferred_checkpoint(self.model)
        if not self.model.model_config.is_generation:
            self.llm_args = resolve_scheduled_encoder_args(self.llm_args)
        self._validate_breakable_cuda_graph_compatibility()
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
        if llm_args.encode_only:
            raise ValueError(
                "encode_only models run through EncoderExecutor, which does not use ModelEngine."
            )
        init_process_setup(mapping, dist)

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
        self.dist = dist
        self.llm_args = llm_args
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
        if llm_args.multimodal_config.vision_gpu_normalize:
            input_processor_kwargs["vision_gpu_normalize"] = True
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
        defer_checkpoint: bool = False,
    ) -> None:
        """Load the model, or take the given one, at the requested capacity."""
        self.moe_load_balancer: BaseMoeLoadBalancer | None = None
        if model is None:
            # Keep the model_loader to support reloading the model weights later
            self.model_loader = create_model_loader(
                self.llm_args,
                self.mapping,
                spec_config=self.spec_config,
                max_num_tokens=self.max_num_tokens,
                max_seq_len=self.max_seq_len,
                model_weights_memory_tag=model_weights_memory_tag,
                model_weights_restore_mode=model_weights_restore_mode,
            )
            # Open checkpoint and load the LLM module object.
            self.model, moe_load_balancer = self.model_loader.load(
                checkpoint_dir=model_path,
                checkpoint_loader=checkpoint_loader,
                defer_checkpoint=defer_checkpoint,
            )
            if isinstance(moe_load_balancer, BaseMoeLoadBalancer):
                self.moe_load_balancer = moe_load_balancer
        else:
            self.model = model
        self.spec_decode_fast_paths = _spec_decode_fast_paths(self.model)

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
        from .scheduler.gen_preemption import headroom_for_args

        self.gen_preemption_headroom = headroom_for_args(self.batch_size, self.llm_args)
        self.max_num_seq_slots = compute_max_num_sequences(
            self.mapping,
            self.batch_size,
            self.llm_args.disable_overlap_scheduler,
            enable_overlap_headroom=self.enable_overlap_headroom,
            gen_preemption_headroom=self.gen_preemption_headroom,
        )
        prepare_loaded_model(self.model, self.llm_args)
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
        """Compile the model, capture the local multimodal encoders' CUDA
        graphs, and keep the model caller."""
        compiled = compile_model(
            self.model,
            llm_args=self.llm_args,
            mapping=self.mapping,
            dist=self.dist,
            max_num_tokens=self.max_num_tokens,
            prefill_capture_num_tokens=self._prefill_cuda_graph_num_tokens,
        )
        self.model = compiled.model
        self._torch_compile_backend = compiled.backend
        self._torch_compile_prefill_only = compiled.prefill_only
        self._nccl_window_tensor_scope_hooks = compiled.nccl_window_tensor_scope_hooks
        # Capture local encoders after their metadata is sized and before
        # KV-cache estimation, so graph memory is included in the profile.
        mm_config = self.llm_args.multimodal_config
        setup_mm_encoder_cuda_graphs(
            self.model,
            mm_config.encoder_cuda_graph if mm_config is not None else None,
            max_num_tokens=configured_encoder_max_num_tokens,
        )
        self._model_caller = compiled.caller

    def _init_execution_services(self) -> None:
        """Build the attention backend, update the speculation config from the
        loaded model, and build the runner around the model caller."""
        # Start outside warmup: load-balancer statistics on, serving MoE
        # communication timeouts.
        self._warmup_depth = 0
        self._set_warmup_effects(False)

        self.attn_backend = resolve_attention_backend(self.model, self.llm_args)

        runner_cls = resolve_runner_type(self.model, self.llm_args)
        if self.is_spec_decode and issubclass(runner_cls, (MultimodalEncoderRunner, EncoderRunner)):
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
            return runner_cls.create(
                self.model,
                llm_args=self.llm_args,
                mapping=self.mapping,
                dist=self.dist,
                max_batch_size=self.batch_size,
                max_num_tokens=self.max_num_tokens,
                max_seq_len=self.max_seq_len,
                max_beam_width=self.max_beam_width,
                model_caller=self._model_caller,
                attention_backend=self.attn_backend,
                attention_runtime_features=self.attn_runtime_features,
                moe_load_balancer=self.moe_load_balancer,
            )
        if issubclass(runner_cls, EncoderDecoderRunner):
            return self._initialize_encoder_decoder_runner(runner_cls)
        if issubclass(runner_cls, MultimodalEncoderRunner):
            return runner_cls.create(self.model, moe_load_balancer=self.moe_load_balancer)
        if issubclass(runner_cls, DecoderRunner):
            return self._initialize_decoder_runner(runner_cls)
        raise TypeError(
            f"No runner initializer registered for "
            f"{runner_cls.__module__}.{runner_cls.__qualname__}"
        )

    def _initialize_encoder_decoder_runner(
        self, runner_cls: type[EncoderDecoderRunner]
    ) -> EncoderDecoderRunner:
        encoder_config = EncoderConfig.create(
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
            watermark_config=self.llm_args.watermark_config,
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

    def _configure_allreduce_tuning(self) -> None:
        """Configure profiling buckets from the resolved budget before runner warmup."""
        from ..custom_ops.torch_custom_ops import AllReduceRunner

        if not self.llm_args.enable_autotuner:
            return
        AllReduceRunner.configure_tuning_buckets(self.max_num_tokens)

    def _warmup_impl(self, resource_manager: ResourceManager | None = None) -> None:
        """
        Orchestrates the warmup process by calling specialized warmup methods for
        torch.compile, the autotuner, and CUDA graphs.
        """
        self._configure_allreduce_tuning()
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

    def _init_model_capacity(self):
        self.max_seq_len = resolve_max_seq_len(self.model, self.max_seq_len)
        self.max_num_tokens = resolve_max_num_tokens(
            self.max_num_tokens, max_seq_len=self.max_seq_len, max_batch_size=self.batch_size
        )

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

    @property
    def has_deferred_checkpoint(self) -> bool:
        """True while the model holds the dummy weights of a deferred checkpoint load."""
        return self.model_loader is not None and self.model_loader.has_deferred_checkpoint

    def rehearse(self, resource_manager: ResourceManager) -> None:
        """Warmup forwards on the dummy weights, inside the caller's KernelBuilder
        rehearsal, so the ops list their kernels; every rank calls it."""
        if not isinstance(self._runner, DecoderRunner):
            return
        self.set_warmup_purpose("rehearsal")
        with self.warmup_scope():
            self._runner.rehearse(resource_manager)

    def load_deferred_checkpoint(self) -> None:
        """Load the checkpoint the model loader deferred, as a weight update does."""
        self.unwrap_compiled_model_for_refit()
        self.model_loader.load_deferred_checkpoint(self.model)
        self.restore_compiled_model_after_refit(None)
        self.invalidate_multimodal_encoder_cache()

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
        self.model = remove_torch_compile(self.model)

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
            self.model, self._torch_compile_prefill_only = apply_torch_compile(
                self.model,
                self._torch_compile_backend,
                llm_args=self.llm_args,
                fullgraph=self.llm_args.torch_compile_config.enable_fullgraph,
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
        batch: ScheduledRequests,
        resource_manager: ResourceManager | None = None,
        new_tensors_device: SampleStateTensors | None = None,
        cache_indirection_buffer: torch.Tensor | None = None,
        *,
        enable_spec_decode: bool = False,
        runtime_draft_len: int = 0,
    ):
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

    def wait_for_input_copy(self) -> None:
        """
        Wait for input preparation and H2D copy of previous iteration before modifying host input,
        otherwise the input of previous iteration will be overwritten.
        """
        if self._runner is not None:
            self._runner.wait_for_input_copy()
