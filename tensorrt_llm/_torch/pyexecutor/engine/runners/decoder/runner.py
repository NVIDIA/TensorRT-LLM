# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder execution moved out of ``PyTorchModelEngine``."""

import contextlib
import functools
import gc
import inspect
import os
from collections.abc import Callable
from typing import Any

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.memory_buffer_utils import clear_memory_buffers, with_shared_pool
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM, timing_metric
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDAGraphRunner,
    CUDAGraphRunnerConfig,
    KeyType,
)
from tensorrt_llm._torch.pyexecutor.guided_decoder import CapturableGuidedDecoder
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import warmup_sampling_module
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RemoteTailPhase
from tensorrt_llm._torch.pyexecutor.trace_log_utils import log_mem_snapshot
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._torch.pyexecutor.workspace import EagerWorkspaceReclaimer
from tensorrt_llm._torch.speculative import (
    SpecMetadata,
    get_draft_kv_cache_manager,
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata
from tensorrt_llm._torch.speculative.utils import get_static_draft_len
from tensorrt_llm._torch.utils import (
    get_per_request_prefill_cuda_graph_flag,
    with_model_extra_attrs,
)
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.multimodal import _has_mm_payload_keys
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor, InputProcessor
from tensorrt_llm.llmapi.llm_args import BaseSparseAttentionConfig, PrefillCudaGraphBackend
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType, Mapping

from ...lora import LoraParamBuilder, make_cuda_graph_lora_manager
from ...model_call import ModelCaller
from ...multimodal import MultimodalItemScheduler
from ..common import (
    get_top_level_model,
    make_scheduled_inputs,
    moe_a2a_steady_state_budget_for_capture,
    prepare_multimodal_indices,
)
from ..interface import PreparedInputs, ScheduledInputs, ScheduledModelRunner
from .config import DecoderRunnerConfig
from .dummy_requests import DummyRequestFactory
from .graphs import DecoderCudaGraphs
from .inputs import DecoderInputBuffers, DecoderInputPreparer, ExtraInputsCollector
from .metadata import DecoderMetadata, resolve_mamba_metadata_cls
from .speculative import get_spec_managers, resolve_runtime_draft_len, update_spec_metadata
from .warmup import WarmupAgreement


def _make_single_token_context_graph_batch(
    scheduled_requests: ScheduledRequests,
    is_multimodal_decode_compatible: Callable[[LlmRequest], bool] | None = None,
) -> tuple[ScheduledRequests, frozenset[int]]:
    """Build a decode-shaped graph candidate for final one-token contexts.

    Multimodal rows remain fail-closed unless the engine proves that their one
    remaining prompt token is representable by the existing decode provider.
    """
    if scheduled_requests.num_context_requests == 0:
        return scheduled_requests, frozenset()

    context_requests = scheduled_requests.context_requests_last_chunk
    if scheduled_requests.encoder_requests or scheduled_requests.context_requests_chunking:
        return scheduled_requests, frozenset()

    for request in context_requests:
        if (
            request.context_chunk_size != 1
            or request.context_remaining_length != 1
            or request.context_current_position + 1 != request.py_prompt_len
            or request.py_beam_width != 1
            or get_draft_token_length(request) > 0
            or request.py_is_first_draft
            or request.is_context_only_request
            or request.is_generation_only_request
            or request.py_disaggregated_params is not None
            or request.py_mm_encoder_event is not None
            or (
                request.py_multimodal_data is not None
                and (
                    is_multimodal_decode_compatible is None
                    or not is_multimodal_decode_compatible(request)
                )
            )
        ):
            return scheduled_requests, frozenset()

    for request in scheduled_requests.generation_requests:
        if (
            request.py_beam_width != 1
            or get_draft_token_length(request) > 0
            or request.py_is_first_draft
            or request.py_disaggregated_params is not None
        ):
            return scheduled_requests, frozenset()

    graph_batch = ScheduledRequests()
    graph_batch.generation_requests = list(context_requests) + list(
        scheduled_requests.generation_requests
    )
    graph_batch.paused_requests = list(scheduled_requests.paused_requests)
    promoted_context_request_ids = frozenset(request.py_request_id for request in context_requests)
    return graph_batch, promoted_context_request_ids


class DecoderRunner(ScheduledModelRunner):
    """Run scheduled decoder generation with its own buffers, graphs and metadata."""

    # Warmups that synthesize context batches; a subclass can turn them off.
    _context_warmups_supported = True
    # Prompt tokens of the filler CUDA graph dummy requests, also the floor of
    # the longest one; None keeps the cache manager's default.
    _dummy_request_tokens: int | None = None
    # Collects per-sequence model inputs beyond the decoder's own; None when the
    # model takes none. Steady-generation reuse and context-row promotion do
    # not carry such inputs, so both stay off while it is set.
    _new_extra_inputs: Callable[[], ExtraInputsCollector] | None = None

    def __init__(
        self,
        model: torch.nn.Module,
        config: DecoderRunnerConfig,
        *,
        input_processor: InputProcessor | BaseMultimodalInputProcessor,
        model_caller: ModelCaller,
        mapping: Mapping,
        dist: Distributed | None,
        moe_load_balancer: MoeLoadBalancer | None,
        sparse_attention_config: BaseSparseAttentionConfig | None,
        torch_compile_backend: Backend | None,
        get_runtime_tokens_per_gen_step: Callable[[int], int],
        warmup_timer: _WarmupTimer,
        iter_states: dict[str, Any],
        metrics: dict[str, float],
        kv_cache_manager_key: ResourceManagerType,
        mm_item_scheduler: MultimodalItemScheduler | None = None,
    ) -> None:
        self.model = model
        self.input_processor = input_processor
        self._model_caller = model_caller
        self.mapping = mapping
        self.dist = dist
        self._agreement = WarmupAgreement(dist, mapping)
        self.moe_load_balancer = moe_load_balancer
        self.sparse_attention_config = sparse_attention_config
        self._torch_compile_backend = torch_compile_backend
        self.get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step
        self._warmup_timer = warmup_timer
        self.iter_states = iter_states
        self._metrics = metrics
        self.kv_cache_manager_key = kv_cache_manager_key

        self._config = config

        self.lora_model_config: LoraModelConfig | None = None
        self.forward_pass_callable = None
        # Optional in-graph sampling hook, registered by PyExecutor. Called at
        # the tail of _forward_step -- i.e. right after the LM head, and inside
        # capture_forward_fn -- so that for graph-capturable sampling tiers the
        # sampling lands at the end of the captured forward graph instead of
        # costing a separate launch. Left None when the sampler does not opt in.
        self.sample_in_graph_callable = None
        # Stages (or clears) in-graph sampling state once the batch is settled.
        self._stage_in_graph_sampling = None

        self._buffers = DecoderInputBuffers()
        self._buffers.allocate(self._config, self.use_beam_search)
        self._metadata = DecoderMetadata(self._config, mapping, model, self._buffers)
        self._dummy_requests = DummyRequestFactory(
            self._config,
            kv_cache_manager_key,
            get_runtime_tokens_per_gen_step,
            self._dummy_request_tokens,
            request_kwargs=self._dummy_request_kwargs,
            position_limit=self._dummy_position_limit,
            add_resources=self._add_dummy_request_resources,
            get_draft_kv_cache_manager=self._get_draft_kv_cache_manager,
        )
        self._init_decoder_state()
        self._inputs = DecoderInputPreparer(
            self._buffers,
            self._config,
            mapping,
            dist,
            model,
            get_runtime_tokens_per_gen_step,
            iter_states,
            self._lora,
            use_beam_search=self.use_beam_search,
            fast_path=self._prepare_inputs_fast_path,
            new_extra_inputs=self._new_extra_inputs,
            mm_item_scheduler=mm_item_scheduler,
        )
        cuda_graph_runner = self._initialize_cuda_graph_runner()
        self._graphs = DecoderCudaGraphs(
            cuda_graph_runner,
            self._initialize_breakable_cuda_graph_runner(cuda_graph_runner),
            self._config,
            mapping,
            model,
            sparse_attention_config,
            metrics,
            kv_cache_manager_key,
            self._dummy_requests,
            self._agreement,
            self._metadata,
            get_runtime_tokens_per_gen_step,
            self._dynamic_draft_len_mapping,
            self._lora,
            forward_warmup=self._forward_warmup,
            prepare_capture_batch=self._prepare_capture_batch,
            capture_additional_graphs=self._capture_additional_cuda_graphs,
            release_batch_context=self._release_batch_context,
        )

    @property
    def attn_metadata(self) -> AttentionMetadata | None:
        return self._metadata.attn_metadata

    @attn_metadata.setter
    def attn_metadata(self, value: AttentionMetadata | None) -> None:
        self._metadata.attn_metadata = value

    @property
    def spec_metadata(self) -> SpecMetadata | None:
        return self._metadata.spec_metadata

    @spec_metadata.setter
    def spec_metadata(self, value: SpecMetadata | None) -> None:
        self._metadata.spec_metadata = value

    @property
    def guided_decoder(self) -> CapturableGuidedDecoder | None:
        return self._inputs.guided_decoder

    @guided_decoder.setter
    def guided_decoder(self, value: CapturableGuidedDecoder | None) -> None:
        self._inputs.guided_decoder = value

    @property
    def cuda_graph_runner(self) -> CUDAGraphRunner | None:
        return self._graphs.cuda_graph_runner

    @property
    def breakable_cuda_graph_runner(self) -> BreakableCUDAGraphRunner | None:
        return self._graphs.breakable_cuda_graph_runner

    @property
    def max_beam_width(self) -> int:
        return self._config.max_beam_width

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self.model.extra_attrs)
    def forward(
        self,
        inputs: ScheduledInputs,
        *,
        resource_manager: ResourceManager,
        is_dummy: bool = False,
    ) -> Any:
        return self._forward_decoder(inputs, resource_manager, is_dummy=is_dummy)

    def warmup(self, resource_manager: ResourceManager) -> None:
        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        # Retain a partial summary when a phase raises.
        with self._warmup_timer:
            self._warmup_scheduled(resource_manager, kv_cache_manager)

    def release_graphs(self) -> None:
        self._graphs.release()

    def wait_for_input_copy(self) -> None:
        self._wait_for_decoder_input_copy()

    def register_sample_type_resolver(
        self, resolver: Callable | None, stage: Callable | None = None
    ) -> None:
        """Register how a batch maps to its sampling tier, for the graph key."""
        self.cuda_graph_runner.register_sample_type_resolver(resolver)
        self._stage_in_graph_sampling = stage

    def init_cuda_graph_lora_manager(self, lora_config: LoraConfig) -> None:
        """Build LoRA preparation state for the current executor resources."""
        cuda_graph_manager = None
        if self.cuda_graph_runner is not None and self.cuda_graph_runner.enabled:
            # For spec decode, each generation request contributes
            # max_draft_len + 1 tokens per forward pass.
            max_tokens_per_seq = (
                (self._config.original_max_draft_len + 1) if self._config.is_spec_decode else 1
            )
            cuda_graph_manager = make_cuda_graph_lora_manager(
                self.model,
                lora_config,
                self.lora_model_config,
                self._config.max_batch_size,
                max_tokens_per_seq,
                self._config.max_num_tokens,
            )
        # Resource creation runs again after capacity estimation, once the old
        # executor has released its graphs. Replace the complete preparation state.
        self._lora = LoraParamBuilder(
            spec_config=self._config.spec_config,
            attn_backend=self._config.attention_backend,
            cuda_graph_manager=cuda_graph_manager,
        )
        self._graphs.set_lora(self._lora)
        self._inputs.set_lora(self._lora)

    def _init_decoder_state(self) -> None:
        """Initialize the mutable state read only by engine decoder execution."""
        self._eager_workspace_reclaimer: EagerWorkspaceReclaimer | None = None
        self._trtllm_gen_jit_warmup = False
        self._lora = LoraParamBuilder(
            spec_config=self._config.spec_config,
            attn_backend=self._config.attention_backend,
            cuda_graph_manager=None,
        )
        self._prepare_inputs_event: torch.cuda.Event | None = None
        # Let the first CUDA graph capture create its private pool. Piecewise
        # CUDA graphs use a separate pool owned by their runners, so sharing a
        # pre-created pool handle with the outer graph runner is unnecessary.
        self._cuda_graph_mem_pool = None
        self._dynamic_draft_len_mapping = self._compute_dynamic_draft_len_mapping()

    def _initialize_cuda_graph_runner(self) -> CUDAGraphRunner | None:
        cuda_graph_runner = CUDAGraphRunner(self._cuda_graph_runner_config())
        register_hooks = getattr(self.model, "register_cuda_graph_pre_replay_hooks", None)
        if register_hooks is not None:
            register_hooks(cuda_graph_runner)
        return cuda_graph_runner

    def _cuda_graph_runner_config(self) -> CUDAGraphRunnerConfig:
        return CUDAGraphRunnerConfig(
            use_cuda_graph=(
                not self._config.is_encode_only and self._config.cuda_graph_config is not None
            ),
            cuda_graph_padding_enabled=self._config.cuda_graph_padding_enabled,
            cuda_graph_batch_sizes=self._config.cuda_graph_batch_sizes,
            max_cuda_graph_batch_size=self._config.max_cuda_graph_batch_size,
            max_beam_width=self._config.max_beam_width,
            spec_config=self._config.spec_config,
            cuda_graph_mem_pool=self._cuda_graph_mem_pool,
            dynamic_draft_len_mapping=self._dynamic_draft_len_mapping,
            max_num_tokens=self._config.max_num_tokens,
            use_mrope=self._config.use_mrope,
            original_max_draft_len=self._config.original_max_draft_len,
            original_max_total_draft_tokens=self._config.original_max_total_draft_tokens,
            enable_attention_dp=self._config.enable_attention_dp,
            is_encoder_decoder=False,
            batch_size=self._config.max_batch_size,
            mapping=self.mapping,
            dist=self.dist,
            kv_cache_manager_key=self.kv_cache_manager_key,
            sparse_attention_config=self.sparse_attention_config,
            enable_in_graph_sampling=self._config.enable_in_graph_sampling,
        )

    def _initialize_breakable_cuda_graph_runner(
        self, cuda_graph_runner: CUDAGraphRunner | None
    ) -> BreakableCUDAGraphRunner | None:
        if (
            cuda_graph_runner is None
            or self._config.prefill_cuda_graph_backend != PrefillCudaGraphBackend.BREAKABLE
        ):
            return None

        decoder_model = (
            self.model
            if isinstance(self.model, DecoderModelForCausalLM)
            else getattr(self.model, "llm", None)
        )
        if not isinstance(decoder_model, DecoderModelForCausalLM):
            raise ValueError("breakable prefill CUDA graph requires a decoder model body")
        return BreakableCUDAGraphRunner(decoder_model.model)

    def _use_lora_cuda_graph(self, scheduled_requests: ScheduledRequests) -> bool:
        """
        Determines whether a non-LoRA or LoRA CUDA graph should be used, if
        both are available (cuda_graph_specialize_lora==True).
        """
        if self._lora.cuda_graph_manager is None:
            return False
        # Needed during graph capture to enforce a given mode
        if self._graphs.force_lora_graph_for_capture is not None:
            return self._graphs.force_lora_graph_for_capture
        if not self._config.cuda_graph_specialize_lora:
            return True
        return any(
            request.lora_task_id is not None for request in scheduled_requests.generation_requests
        )

    @property
    def use_beam_search(self):
        return self._config.max_beam_width > 1

    def _get_draft_kv_cache_manager(
        self, resource_manager: ResourceManager
    ) -> KVCacheManager | KVCacheManagerV2 | None:
        """
        Returns the draft KV cache manager only in one-model speculative decoding
        mode where the target model manages a separate draft KV cache.
        """
        return get_draft_kv_cache_manager(self._config.spec_config, resource_manager)

    def _get_max_shape_warmup_requests(
        self, resource_manager: ResourceManager
    ) -> list[tuple[int, int]]:
        """
        Returns warmup configs covering the maximum context and generation shapes.
        """

        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        token_num_upper_bound = min(
            self._config.max_num_tokens,
            self._config.max_batch_size * (self._config.max_seq_len - 1),
        )
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self._config.original_max_draft_len,
        )
        max_batch_size = min(
            self._config.max_batch_size,
            curr_max_num_tokens
            // (1 + self._config.max_draft_loop_tokens)
            // self._config.max_beam_width,
        )

        warmup_requests_configs = [
            (curr_max_num_tokens, 0),  # max_num_tokens, pure context
            (max_batch_size, max_batch_size),  # max_batch_size, pure generation
        ]

        return warmup_requests_configs

    def _get_full_general_warmup_requests(
        self, resource_manager: ResourceManager
    ) -> list[tuple[int, int]]:
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

    def _warmup_scheduled(self, resource_manager: ResourceManager, kv_cache_manager) -> None:
        """Body of ``warmup`` for the scheduled (KV-cache-backed) path."""
        # Ahead of the legacy early returns below: only the advanced-sampling
        # CUDA graph capture pass exercises the non-greedy sampler, so with
        # cuda_graph_config=None flashinfer's sampling kernels would be
        # JIT-built mid-serving.
        self._eager_workspace_reclaimer = None
        with self._warmup_timer.phase(
            "sampling_module_prewarm", metrics=self._metrics, metric_name="sampling_warmup_seconds"
        ):
            warmup_sampling_module()
            self._prewarm_engram_kernels(kv_cache_manager)
            self._prewarm_csa2_sparse_indexer(kv_cache_manager)

        if kv_cache_manager is None:
            logger.info("Skipping warm up as no KV Cache manager allocated.")
            return

        # The lifetime of model engine and kv cache manager can be different.
        # Reset the global cuda graph dummy requests in warmup.
        self.cuda_graph_runner.padding_dummy_requests = {}

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
        can_run_general_warmup = self._agreement.agree_flag(
            self._context_warmups_supported
            and not self.mapping.has_cp_helix()
            and self.guided_decoder is None
            and not isinstance(kv_cache_manager, MambaHybridCacheManager)
        )

        log_mem_snapshot("warmup/before_warmup")
        # Compile the DSv4 indexer-Q CuTe DSL kernels before the first
        # collective-bearing forward, so their JIT cost is not charged against the
        # MoE all-to-all completion-flag deadline.
        with self._warmup_timer.phase("cute_dsl_indexer_q"):
            self._prewarm_cute_dsl_indexer_q()
        log_mem_snapshot("warmup/after_cute_dsl_indexer_q")
        if self._context_warmups_supported:
            with self._warmup_timer.phase(
                "attention_jit", metrics=self._metrics, metric_name="attention_warmup_seconds"
            ):
                self._run_attention_warmup(resource_manager, can_run_general_warmup)

        if can_run_general_warmup:
            # Specialize torch.compile graphs across the key input shapes before CUDA graph capture.
            with self._warmup_timer.phase(
                "general", metrics=self._metrics, metric_name="general_warmup_seconds"
            ):
                warmup_requests_configs = self._agreement.agree_shapes(
                    self._get_full_general_warmup_requests(resource_manager)
                )
                # Currently graph has not been captured, disable cuda graph for this warmup.
                with self._graphs.no_cuda_graph():
                    self._general_warmup(resource_manager, warmup_requests_configs)
                    # Release C++ MoE workspace buffers so the autotuner can
                    # reclaim the memory.  They will be re-allocated on next use.
                    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

                    MoERunner.clear_all_workspaces()
                    # Clear Cache now as autotuner may use additional memory.
                    # Memory pool will be warmed up later.
                    gc.collect()
                    torch.cuda.empty_cache()

        # Helix CP is decode-only and runs into issues with the
        # autotuner warmup's context requests.
        if self._context_warmups_supported and not self.mapping.has_cp_helix():
            with self._warmup_timer.phase(
                "autotuner", metrics=self._metrics, metric_name="autotuner_warmup_seconds"
            ):
                self._run_autotuner_warmup(resource_manager)
            log_mem_snapshot("warmup/after_autotuner")
            # Pre-JIT Mamba SSD multi-seq + HAS_INITSTATES=True Triton kernels
            # for Mamba hybrid models. Runs regardless of enable_autotuner,
            # since MambaHybridCacheManager skips _general_warmup and the
            # default autotuner shape is single-seq / no-initstates. Safe
            # no-op for non-Mamba models.
            with self._warmup_timer.phase(
                "mamba_hybrid", metrics=self._metrics, metric_name="mamba_hybrid_warmup_seconds"
            ):
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
        with (
            self._warmup_timer.phase("cuda_graph_capture"),
            moe_a2a_steady_state_budget_for_capture(),
        ):
            with self.cuda_graph_runner.allow_capture():
                self.cuda_graph_runner.is_warmup_only = True
                try:
                    self._graphs.run_cuda_graph_warmup(resource_manager)
                finally:
                    self.cuda_graph_runner.is_warmup_only = False
                self.cuda_graph_runner.padding_dummy_requests = {}
                self._graphs.run_cuda_graph_warmup(resource_manager)
        log_mem_snapshot("warmup/after_cuda_graph_capture")
        # Pre-compile DeepGEMM paged_mqa_logits_metadata for every 32-aligned
        # batch bucket the runtime can produce (max_batch_size scaled by the
        # MTP / DSL expansion factor when applicable). CUDA-graph warmup only
        # exercises the batch sizes in cuda_graph_batch_sizes, which round
        # up to a subset of buckets; any inference iter whose
        # context_lens.size(0) lands on an uncovered bucket triggers an
        # nvcc-driven JIT compile (~3s stall inside _prepare_inputs) on
        # first touch. Pre-touching every bucket funnels that cost into
        # warmup. No-op on non-DSA models.
        # Both DSA hooks read attn_metadata, which only a warmup forward
        # creates; build it when every forward above was skipped.
        with self._warmup_timer.phase("dsa_prewarm"):
            self._ensure_dsa_attn_metadata_for_warmup(resource_manager)
            with timing_metric("dg_paged_mqa_warmup_seconds", self._metrics):
                self._warmup_dg_paged_mqa_logits_metadata()
            log_mem_snapshot("warmup/after_dg_paged_mqa_logits_metadata")
            with timing_metric("cute_dsl_radix_topk_warmup_seconds", self._metrics):
                self._warmup_cute_dsl_radix_topk()
        log_mem_snapshot("warmup/after_cute_dsl_radix_topk")
        if can_run_general_warmup:
            # Pre-populate the memory pool with max-shape allocations to reduce
            # fragmentation at runtime.
            with self._warmup_timer.phase(
                "memory_pool_prepop",
                metrics=self._metrics,
                metric_name="memory_pool_prepopulation_seconds",
            ):
                warmup_requests_configs = self._get_max_shape_warmup_requests(resource_manager)
                self._general_warmup(resource_manager, warmup_requests_configs)
            log_mem_snapshot("warmup/after_memory_pool_prepop")

        # Allocate the CUDA graph padding dummies now, while the KV cache is
        # empty. Waiting for the first padded step can race KV saturation:
        # once the cache is full, the lazy allocation in _get_padded_batch
        # fails every step and padded batches silently run eager.
        if getattr(kv_cache_manager, "context_swa_layer_limit", None) is None:
            with self._warmup_timer.phase("preallocate_padding_dummies"):
                self.cuda_graph_runner.preallocate_padding_dummies(resource_manager)
        log_mem_snapshot("warmup/after_preallocate_padding_dummies")

        # If this is a BOLT-instrumented build (the profile-gen job sets
        # TLLM_BOLT_CLEAR_COUNTERS=1), reset the instrumentation counters now
        # that all startup JIT/autotune/graph-capture is done, so the emitted
        # .fdata reflects steady-state serving only. No-op on normal builds.
        from tensorrt_llm._torch.bolt_profiling import maybe_bolt_clear_counters

        maybe_bolt_clear_counters()

        self._freeze_eager_workspace_floor()

    def _freeze_eager_workspace_floor(self) -> None:
        if os.environ.get("TRTLLM_RECLAIM_WORKSPACE", "1") == "0":
            return
        metadata = self._metadata.attn_metadata
        if (
            self._config.is_spec_decode
            or self.mapping.cp_size != 1
            or self.sparse_attention_config is not None
            or self._torch_compile_backend is not None
            or self.breakable_cuda_graph_runner is not None
        ):
            logger.info(
                "Eager workspace reclamation is disabled for speculative, "
                "CP, sparse, compiled, or breakable-graph models"
            )
            return
        if (
            type(metadata) is not TrtllmAttentionMetadata
            or not metadata.workspace_reclaimable
            or metadata.workspace is None
            or metadata.workspace.untyped_storage().nbytes() == 0
        ):
            logger.info("Eager workspace reclamation requires warmup of pure fallback scratch")
            return
        self._eager_workspace_reclaimer = EagerWorkspaceReclaimer(metadata)

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
            from tensorrt_llm._torch.attention.backends.sparse.dsa import (
                _DG_SCHEDULE_BLOCK_KV,
                DSAtrtllmAttentionMetadata,
            )
        except ImportError:
            return
        if not isinstance(attn_meta, DSAtrtllmAttentionMetadata):
            return
        try:
            from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata
        except ImportError:
            logger.info(
                "[DG warmup] deep_gemm.get_paged_mqa_logits_metadata not "
                "available; skipping paged_mqa_logits_metadata prewarm."
            )
            return

        num_sms = attn_meta.num_sms
        max_bs = max(1, int(self._config.max_batch_size))
        beam_width = max(1, int(getattr(self, "max_beam_width", 1) or 1))
        # Static upper bound on the row-count multiplier applied to
        # `context_lens`. Both MTP-expanded and DSL-expanded call sites
        # are bounded above by `(1 + max_draft_tokens)`; see the
        # docstring for why we don't read the runtime `dsl_expand_factor`
        # here.
        max_draft_tokens = int(getattr(attn_meta, "max_draft_tokens", 0) or 0)
        expands_batch = getattr(attn_meta, "use_expanded_buffers_for_mtp", False) or getattr(
            attn_meta, "expand_for_dsl", False
        )
        expand_factor = 1 + max_draft_tokens if expands_batch else 1
        max_aligned = ((max_bs * beam_width * expand_factor + 31) // 32) * 32
        buckets = list(range(32, max_aligned + 32, 32))
        logger.info(
            f"[DG warmup] Pre-compiling paged_mqa_logits_metadata for "
            f"{len(buckets)} aligned batch buckets up to {max_aligned} "
            f"(block_kv={_DG_SCHEDULE_BLOCK_KV}, num_sms={num_sms}, "
            f"max_bs={max_bs}, beam_width={beam_width}, "
            f"expand_factor={expand_factor})"
        )
        for aligned_bs in buckets:
            # Kernel scans `context_lens` and prefix-sums schedules; a
            # zero-filled 2D tensor of shape (aligned_bs, 1) is enough to
            # trigger dispatch and compile — the metadata output is
            # discarded.
            dummy = torch.zeros(aligned_bs, 1, dtype=torch.int32, device="cuda")
            try:
                _ = get_paged_mqa_logits_metadata(dummy, _DG_SCHEDULE_BLOCK_KV, num_sms)
            except RuntimeError as e:
                # Narrow to RuntimeError so signature drifts in
                # get_paged_mqa_logits_metadata (TypeError / ValueError)
                # surface loudly instead of silently degrading perf.
                logger.warning(
                    f"[DG warmup] paged_mqa_logits_metadata prewarm failed "
                    f"for aligned_bs={aligned_bs} "
                    f"(block_kv={_DG_SCHEDULE_BLOCK_KV}, num_sms={num_sms}); "
                    f"skipping bucket. {type(e).__name__}: {e}"
                )
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
            from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.deepseek_v4 import (
                DeepseekV4Indexer,
            )
        except ImportError:
            return

        indexer = next(
            (
                m
                for m in self.model.modules()
                if isinstance(m, DeepseekV4Indexer) and getattr(m, "wq_b", None) is not None
            ),
            None,
        )
        if indexer is None:
            return

        weight = indexer.wq_b.weight
        # _fallback_tactic() branches on m at 4 and 8, so these three token
        # counts cover every fallback tactic it can return.
        with torch.inference_mode():
            for num_tokens in (4, 8, 16):
                try:
                    qr = torch.zeros(
                        (num_tokens, weight.shape[1]), dtype=torch.bfloat16, device=weight.device
                    )
                    position_ids = torch.zeros(
                        (num_tokens,), dtype=torch.int32, device=weight.device
                    )
                    indexer._project_and_quantize_q(qr, position_ids)
                except Exception as e:
                    # Never fail startup for a prewarm miss; the kernel would
                    # simply be compiled later, as it is today.
                    logger.warning(
                        f"indexer-Q CuTe DSL prewarm skipped for {num_tokens} "
                        f"tokens. {type(e).__name__}: {e}"
                    )
        torch.cuda.synchronize()

        # Hold every rank here until the slowest has finished compiling, so the
        # first MoE all-to-all dispatch is entered without JIT skew.
        if self.mapping.tp_size > 1 and self.dist is not None:
            self.dist.tp_allgather(1)
        logger.info("indexer-Q CuTe DSL prewarm complete")

    def _ensure_dsa_attn_metadata_for_warmup(self, resource_manager: ResourceManager) -> None:
        """Build the DSA attention metadata if no warmup forward created it, so
        the top-K pre-compile hooks still run (draft engine, guided decoder or
        context-only server without general warmup). No-op unless DSA."""
        if getattr(self, "attn_metadata", None) is not None:
            return
        try:
            from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
        except ImportError:
            return
        metadata_cls = getattr(self._config.attention_backend, "Metadata", None)
        if metadata_cls is None or not issubclass(metadata_cls, DSAtrtllmAttentionMetadata):
            return
        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        if kv_cache_manager is None:
            return
        self._metadata.set_up_attn(
            kv_cache_manager, self._get_draft_kv_cache_manager(resource_manager)
        )

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
            from tensorrt_llm._torch.attention.backends.sparse.dsa import DSAtrtllmAttentionMetadata
        except ImportError:
            return
        if isinstance(attn_meta, DSAtrtllmAttentionMetadata):
            next_n = 1 + self._config.original_max_draft_len
            attn_meta.warmup_cute_dsl_radix_topk(next_n)
            if hasattr(attn_meta, "warmup_selfsampling_topk"):
                attn_meta.warmup_selfsampling_topk(
                    next_n, batch_sizes=self._config.cuda_graph_batch_sizes
                )

    def _general_warmup(
        self, resource_manager: ResourceManager, warmup_requests_configs: list[tuple[int, int]]
    ):
        """
        Runs forward passes for each config in warmup_requests_configs.

        Serves both torch.compile graph specialization and memory pool pre-population.
        """
        # Disable CUDA graph replay during general warmup to avoid replaying
        # graphs with stale KV cache block offsets from capture time.
        with self._graphs.no_cuda_graph():
            self._general_warmup_impl(resource_manager, warmup_requests_configs)

    def _general_warmup_impl(
        self, resource_manager: ResourceManager, warmup_requests_configs: list[tuple[int, int]]
    ) -> None:
        for num_tokens, num_gen_tokens in warmup_requests_configs:
            # Helix CP does not support warmup with context requests.
            if self.mapping.has_cp_helix() and num_tokens != num_gen_tokens:
                continue
            try:
                with self._release_batch_context(
                    self._dummy_requests.create_warmup_request(
                        resource_manager, num_tokens, num_gen_tokens
                    ),
                    resource_manager,
                ) as batch:
                    if not self._agreement.should_run_warmup_batch(
                        batch,
                        num_tokens,
                        f"general, num_tokens={num_tokens}, num_gen_tokens={num_gen_tokens}",
                    ):
                        continue
                    logger.info(
                        f"Run warmup with {num_tokens} tokens, include {num_gen_tokens} generation tokens"
                    )
                    with self._warmup_timer.phase(
                        f"general shape num_tokens={num_tokens}, num_gen_tokens={num_gen_tokens}",
                        record=False,
                        log_start=False,
                    ):
                        self._forward_warmup(
                            batch,
                            resource_manager,
                            enable_spec_decode=self._config.is_spec_decode,
                            runtime_draft_len=get_static_draft_len(self._config.spec_config),
                        )
                        torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                if self._agreement.is_distributed_forward():
                    # Peers are inside the same forward's collectives and
                    # cannot follow a rank-local skip.
                    raise
                logger.warning(
                    f"OOM during general warmup with {num_tokens} tokens, "
                    f"{num_gen_tokens} generation tokens. Skipping."
                )
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

    def _run_attention_warmup(
        self, resource_manager: ResourceManager, can_run_general_warmup: bool = True
    ) -> None:
        if not issubclass(self._config.attention_backend.Metadata, TrtllmAttentionMetadata):
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
        if self.guided_decoder is None:
            # doesn't support guided decoding
            warmup_requests_configs.append(
                (1 + self._config.max_total_draft_tokens, 1)
            )  # one generation request
        else:
            logger.debug("Skipped TRTLLM-Gen FMHA JIT warmup for Gen kernels")

        if can_run_general_warmup:
            warmup_requests_configs.append((1, 0))  # one context token
        else:
            logger.debug("Skipped TRTLLM-Gen FMHA JIT warmup for Ctx kernels")

        # Kimi K3 / Kimi Linear are warmed by ``_run_mamba_hybrid_warmup``, not
        # here: they always run on a ``MambaHybridCacheManager``, so
        # ``can_run_general_warmup`` is False and the configs above never run.

        if self.guided_decoder is None and can_run_general_warmup:
            # The cute_dsl_mla FMHA lib now only support the generation-only batch, we need to
            # warmup the TRTLLM-Gen FMHA lib for the mixed context+generation batch.
            # One MIXED context+generation batch (1 ctx token + 1 gen request).
            warmup_requests_configs.append((1 + self._config.max_total_draft_tokens + 1, 1))
        else:
            logger.debug(
                "Skipped TRTLLM-Gen flashinfer_trtllm_gen FMHA lib JIT warmup When enable cute_dsl_mla FMHA lib"
            )

        for num_tokens, num_gen_requests in warmup_requests_configs:
            warmup_request = self._dummy_requests.create_warmup_request(
                resource_manager, num_tokens=num_tokens, num_gen_requests=num_gen_requests
            )

            with (
                self._graphs.no_cuda_graph(),
                self._release_batch_context(warmup_request, resource_manager) as batch,
            ):
                if not self._agreement.should_run_warmup_batch(
                    batch,
                    num_tokens,
                    f"attention, num_tokens={num_tokens}, num_gen_requests={num_gen_requests}",
                ):
                    continue
                # The first forward of the process lands here, so this shape
                # also absorbs every first-touch JIT (CuTe DSL GEMM, FMHA
                # NVRTC grid, ...). Time it per shape so a slow startup can be
                # attributed without a debugger.
                with self._warmup_timer.phase(
                    f"attention shape num_tokens={num_tokens}, num_gen_requests={num_gen_requests}",
                    record=False,
                    log_start=True,
                ):
                    with trtllm_gen_fmha_jit_warmup():
                        self._forward_warmup(
                            batch,
                            resource_manager,
                            enable_spec_decode=self._config.is_spec_decode,
                            runtime_draft_len=get_static_draft_len(self._config.spec_config),
                        )
                    torch.cuda.synchronize()

    @staticmethod
    def _release_megamoe_profiling_scratch():
        # MegaMoE tuning resources are shared across layers, so only the engine
        # can release them after its full autotune warmup and before graph
        # capture. Later eviction could invalidate a captured workspace pointer.
        from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as _megamoe_op

        release_megamoe_scratch = getattr(_megamoe_op, "release_megamoe_profiling_scratch", None)
        if release_megamoe_scratch is not None:
            release_megamoe_scratch()

    def _run_autotuner_warmup(self, resource_manager: ResourceManager) -> None:
        """Runs forward passes to populate the autotuner cache."""
        from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
            IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE,
            MXFP8GemmRunner,
        )
        from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod, flashinfer_mxfp8_autotune

        enable_trtllm_autotuner = self._config.enable_autotuner
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
        native_mxfp8_methods = [method for method in mxfp8_methods if method.needs_native_autotune]
        compile_all_batches = (
            self._config.torch_compile_enabled and not self._config.torch_compile_prefill_only
        )
        if compile_all_batches and "TRTLLM_MXFP8_GEMM_BACKEND" not in os.environ:
            # Compiled auto dispatch uses native; do not tune unused backends.
            # Prefill-only compile retains the eager generation-graph policy.
            for method in mxfp8_methods:
                method.disable_flashinfer_auto()
        use_mxfp8_flashinfer_graph_default = (
            self.cuda_graph_runner.enabled
            and not compile_all_batches
            and "TRTLLM_MXFP8_GEMM_BACKEND" not in os.environ
            and any(
                getattr(module, "_use_flashinfer_mxfp8_decode_graph_default", False)
                for module in self.model.modules()
            )
        )
        if use_mxfp8_flashinfer_graph_default:
            # CuTeDSL is the alternative backend; PP has no graph-pass handoff.
            tune_with_cute_dsl = (
                IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE and not self.mapping.has_pp()
            )
            for quant_method in mxfp8_methods:
                quant_method.enable_flashinfer_auto()
                quant_method.tune_decode_graph_backends = (
                    tune_with_cute_dsl and quant_method.uses_flashinfer
                )
        # An explicit auto setting stays intact, but compiled auto dispatch
        # uses native GEMM. Do not run or mark an unused FlashInfer tuning pass.
        flashinfer_mxfp8_methods = [
            method
            for method in mxfp8_methods
            if method.needs_flashinfer_autotune
            and (not compile_all_batches or method.backend == "flashinfer")
        ]

        # Every TP and PP rank must make the same backend decision before any
        # rank returns or enters a tuning forward with model collectives.
        if self.mapping.tp_size > 1 or self.mapping.has_pp():
            local_flashinfer_enabled = int(bool(flashinfer_mxfp8_methods))
            all_flashinfer_enabled = [local_flashinfer_enabled]
            if self.mapping.tp_size > 1:
                all_flashinfer_enabled = list(self.dist.tp_allgather(local_flashinfer_enabled))
            if self.mapping.has_pp():
                all_flashinfer_enabled = [
                    enabled
                    for stage_flags in self.dist.pp_allgather(all_flashinfer_enabled)
                    for enabled in stage_flags
                ]
            if any(all_flashinfer_enabled) and not all(all_flashinfer_enabled):
                forced_flashinfer = any(method.backend == "flashinfer" for method in mxfp8_methods)
                for method in mxfp8_methods:
                    method.disable_flashinfer_auto()
                flashinfer_mxfp8_methods = []
                if forced_flashinfer:
                    raise RuntimeError(
                        "FlashInfer MXFP8 was explicitly requested but is not "
                        "available on every TP/PP rank"
                    )
                logger.warning(
                    "FlashInfer MXFP8 availability differs across TP/PP ranks; "
                    "using the native TensorRT-LLM GEMM backend on every rank."
                )

        enable_flashinfer_mxfp8_autotuner = bool(flashinfer_mxfp8_methods)
        enable_native_mxfp8_autotuner = bool(native_mxfp8_methods)

        AutoTuner.get().setup_distributed_state(self.mapping, self.dist)
        logger.info(
            f"Running autotuner warmup (TRT-LLM={enable_trtllm_autotuner}, "
            f"native MXFP8={enable_native_mxfp8_autotuner}, "
            f"FlashInfer MXFP8={enable_flashinfer_mxfp8_autotuner})..."
        )
        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        token_num_upper_bound = min(
            self._config.max_num_tokens,
            self._config.max_batch_size * (self._config.max_seq_len - 1),
        )
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self._config.original_max_draft_len,
        )

        warmup_configs = [(curr_max_num_tokens, 0)]
        if self.guided_decoder is None and not self.mapping.has_pp():
            # Add generation request to warmup the autotuner cache.
            warmup_configs.append((1 + self._config.max_total_draft_tokens, 1))

        def run_autotuner_pass(autotune_context: Any, synchronize_trtllm_cache: bool) -> bool:
            """Run one isolated tuning pass with fresh synthetic batches."""
            ran_forward = False
            with self._graphs.no_cuda_graph(), autotune_context:
                for num_tokens, num_gen_requests in warmup_configs:
                    warmup_request = self._dummy_requests.create_warmup_request(
                        resource_manager, num_tokens, num_gen_requests
                    )
                    with self._release_batch_context(warmup_request, resource_manager) as batch:
                        if not self._agreement.should_run_warmup_batch(
                            batch,
                            num_tokens,
                            f"autotuner, num_tokens={num_tokens}, "
                            f"num_gen_requests={num_gen_requests}",
                        ):
                            continue
                        with self._warmup_timer.phase(
                            f"autotuner shape num_tokens={num_tokens}, "
                            f"num_gen_requests={num_gen_requests}",
                            record=False,
                            log_start=True,
                        ):
                            self._forward_warmup(
                                batch,
                                resource_manager,
                                enable_spec_decode=self._config.is_spec_decode,
                                runtime_draft_len=get_static_draft_len(self._config.spec_config),
                            )
                            ran_forward = True
                            torch.cuda.synchronize()

                if ran_forward and synchronize_trtllm_cache:
                    # pp_recv in AutoTuner choose_one will never be called if there is no tuning op
                    # during the forward pass.
                    # So we need to make an extra call to consume the previous rank's pp_send to
                    # guarantee that the previous rank's pp_send is released.
                    AutoTuner.get().cache_pp_recv()
                    # Send the cache after the tuning process to the next PP rank
                    AutoTuner.get().cache_pp_send()
                    # Clean the pp flag to avoid deadlock with synchronous send/recv
                    AutoTuner.get().clean_pp_flag()
            return ran_forward

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        ran_native_forward = run_autotuner_pass(
            autotune(cache_path=cache_path), synchronize_trtllm_cache=True
        )
        ran_flashinfer_forward = False
        if enable_flashinfer_mxfp8_autotuner:
            ran_flashinfer_forward = run_autotuner_pass(
                flashinfer_mxfp8_autotune(), synchronize_trtllm_cache=False
            )

        if enable_flashinfer_mxfp8_autotuner:
            if ran_flashinfer_forward:
                for method in flashinfer_mxfp8_methods:
                    method.mark_flashinfer_autotuned()
            else:
                forced_flashinfer = any(
                    method.backend == "flashinfer" for method in flashinfer_mxfp8_methods
                )
                for method in flashinfer_mxfp8_methods:
                    method.disable_flashinfer_auto()
                if forced_flashinfer:
                    raise RuntimeError(
                        "FlashInfer MXFP8 was explicitly requested but its autotuner "
                        "warmup forward could not run"
                    )
                logger.warning(
                    "FlashInfer MXFP8 autotuning could not run; using the native "
                    "TensorRT-LLM GEMM backend."
                )

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
                    "using the default native GEMM tactic."
                )

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

    def _run_mamba_hybrid_warmup(self, resource_manager: ResourceManager) -> None:
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

        1. ``least_requests=False``, chunk-ragged — splits
           ``curr_max_num_tokens`` into many short sequences, forcing the
           multi-seq path of
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
        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        if kv_cache_manager is None or not isinstance(kv_cache_manager, MambaHybridCacheManager):
            return

        token_num_upper_bound = min(
            self._config.max_num_tokens,
            self._config.max_batch_size * (self._config.max_seq_len - 1),
        )
        curr_max_num_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=token_num_upper_bound,
            max_num_draft_tokens=self._config.original_max_draft_len,
        )
        # Rank-local capacity, so peers can disagree. Leaving the phase alone
        # would unbalance the per-shape agreement inside it.
        if not self._agreement.agree_flag(curr_max_num_tokens >= 4):
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

        logger.info("Running Mamba hybrid warmup (multi-seq + HAS_INITSTATES=True)...")

        # A model whose prefill kernel specializes on chunk alignment declares
        # it on its Mamba metadata class; the two passes then cover both
        # variants, for free since alignment is independent of HAS_INITSTATES.
        # Resolved the same way the runtime does, so warmup can't prime the
        # alignment variant of a class the runtime never instantiates.
        metadata_cls = resolve_mamba_metadata_cls(self.model)
        chunk_alignment = metadata_cls.prefill_chunk_alignment

        # (num_tokens, num_gen_requests, least_requests, force_initstates,
        #  chunk_aligned)
        mamba_warmup_shapes = [
            (capped_num_tokens, 0, False, False, False),
            (capped_num_tokens, 0, False, True, True),
        ]

        autotuner_enabled = self._config.enable_autotuner
        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH", None)
        autotune_ctx = (
            autotune(cache_path=cache_path) if autotuner_enabled else contextlib.nullcontext()
        )

        with self._graphs.no_cuda_graph(), autotune_ctx:
            for (
                num_tokens_i,
                num_gen_requests_i,
                least_req_i,
                force_init_i,
                chunk_aligned_i,
            ) in mamba_warmup_shapes:
                init_ctx = (
                    Mamba2Metadata.force_initial_states_for_warmup()
                    if force_init_i
                    else contextlib.nullcontext()
                )
                shape = (
                    f"Mamba hybrid, num_tokens={num_tokens_i}, "
                    f"num_gen_requests={num_gen_requests_i}, "
                    f"force_initstates={force_init_i}"
                )
                with init_ctx:
                    try:
                        warmup_request = self._dummy_requests.create_warmup_request(
                            resource_manager,
                            num_tokens_i,
                            num_gen_requests_i,
                            least_requests=least_req_i,
                            chunk_alignment=chunk_alignment,
                            chunk_aligned=chunk_aligned_i,
                        )
                    except torch.OutOfMemoryError as e:
                        if self._agreement.is_distributed_forward():
                            raise
                        logger.warning(
                            f"Warmup skipped for shape ({shape}): {type(e).__name__}: {e}"
                        )
                        torch.cuda.empty_cache()
                        continue
                    except RuntimeError as e:
                        # The known KV allocation failure happens before
                        # forward and is recoverable only when no peer worker
                        # can advance independently. Any other RuntimeError is
                        # a defect, not a capacity limit, and is fatal.
                        if self._agreement.is_distributed_forward():
                            raise
                        if "Can't allocate new blocks for window size" not in str(e):
                            raise
                        logger.warning(
                            f"Warmup skipped for shape ({shape}): {type(e).__name__}: {e}"
                        )
                        torch.cuda.empty_cache()
                        continue

                    try:
                        with self._release_batch_context(warmup_request, resource_manager) as batch:
                            if not self._agreement.should_run_warmup_batch(
                                batch, num_tokens_i, shape
                            ):
                                continue
                            self._forward_warmup(
                                batch,
                                resource_manager,
                                enable_spec_decode=self._config.is_spec_decode,
                                runtime_draft_len=get_static_draft_len(self._config.spec_config),
                            )

                            if autotuner_enabled:
                                AutoTuner.get().cache_pp_recv()
                                AutoTuner.get().cache_pp_send()
                                AutoTuner.get().clean_pp_flag()

                            torch.cuda.synchronize()
                    # Once peers can enter warmup synchronization or forward,
                    # any rank-local exception can strand them in a collective.
                    except Exception as e:  # noqa: BLE001
                        if self._agreement.is_distributed_forward():
                            raise
                        # ``torch.OutOfMemoryError`` is a ``RuntimeError``
                        # subclass; anything outside that hierarchy is a defect
                        # rather than a capacity limit.
                        if not isinstance(e, RuntimeError):
                            raise
                        # A single-rank warmup is a pure perf optimization. If
                        # a forward shape does not fit, it can be compiled
                        # lazily on the first real request.
                        logger.warning(
                            f"Warmup skipped for shape ({shape}): {type(e).__name__}: {e}"
                        )
                        # An OOM between dispatch() and combine() leaves the
                        # local MoE A2A state in ``dispatched``.
                        self._reset_moe_alltoall_state()
                        torch.cuda.empty_cache()

        clear_memory_buffers()
        torch.cuda.empty_cache()

    def _compute_dynamic_draft_len_mapping(self) -> dict[int, int] | None:
        """Compute graph_bs → draft_len mapping for dynamic draft length feature.

        Example: draft_len_schedule = {4:4, 8:2, 32:1}, cuda_graph_batch_sizes = [1,2,3,4,5,6,7,8,16,24,32,64]
        - Batch sizes 1-4:   use draft_len=4 (up to key 4)
        - Batch sizes 5-8:   use draft_len=2 (up to key 8)
        - Batch sizes 9-32:  use draft_len=1 (up to key 32)
        - Batch sizes 33+:   use draft_len=0 (implicit, speculation disabled)

        Returns: {1:4, 2:4, 3:4, 4:4, 5:2, 6:2, 7:2, 8:2, 16:1, 24:1, 32:1, 64:0}
        """
        # Dynamic draft length for CUDA graphs is only supported for one-model path
        if (
            not self._config.spec_config
            or not self._config.spec_config.draft_len_schedule
            or not self._config.spec_config.spec_dec_mode.support_dynamic_draft_len()
        ):
            return None

        schedule = self._config.spec_config.draft_len_schedule
        schedule_keys = list(schedule.keys())

        mapping = {}
        key_idx = 0
        for graph_bs in self._config.cuda_graph_batch_sizes:
            while key_idx < len(schedule_keys) and schedule_keys[key_idx] < graph_bs:
                key_idx += 1
            if key_idx < len(schedule_keys):
                draft_len = schedule[schedule_keys[key_idx]]
            else:
                draft_len = 0
            mapping[graph_bs] = draft_len
        return mapping

    def _capture_additional_cuda_graphs(self, resource_manager: ResourceManager) -> None:
        """Capture graphs beyond the generation-only shapes; none by default."""

    def _prepare_capture_batch(
        self,
        batch: ScheduledRequests,
        resource_manager: ResourceManager,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
    ) -> None:
        """Prepare a dummy batch's request state before its graph is captured."""

    @contextlib.contextmanager
    def _release_batch_context(
        self, batch: ScheduledRequests | None, resource_manager: ResourceManager
    ):
        """A context manager to automatically free resources of a dummy batch."""
        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(resource_manager)
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        try:
            yield batch
        finally:
            if batch is not None and kv_cache_manager is not None:
                release_request_state = getattr(self.model, "release_request_state", None)
                for req in batch.all_requests():
                    kv_cache_manager.free_resources(req)
                    if draft_kv_cache_manager is not None:
                        draft_kv_cache_manager.free_resources(req)
                    if spec_resource_manager is not None:
                        spec_resource_manager.free_resources(req)
                    if release_request_state is not None:
                        release_request_state(req.py_request_id)

    def _dummy_request_kwargs(
        self, resource_manager: ResourceManager, num_requests: int
    ) -> dict[str, Any]:
        """Extra ``add_dummy_requests`` arguments for ``num_requests`` dummy requests."""
        return {}

    def _dummy_position_limit(self) -> int | None:
        """Return the largest position a dummy request may reach."""
        model_config = self.model.model_config.pretrained_config
        return getattr(model_config, "max_position_embeddings", None)

    def _add_dummy_request_resources(
        self, requests: list[LlmRequest], resource_manager: ResourceManager
    ) -> bool:
        """Allocate resources dummy requests need beyond the KV caches."""
        return True

    def _preprocess_inputs(
        self, inputs: dict[str, Any], *, enable_spec_decode: bool, runtime_draft_len: int
    ):
        """
        Make some changes to the device inputs and avoid blocking the async data transfer
        """
        attn_meta = inputs.get("attn_metadata")
        # Invalidate per-forward-pass caches so they are recomputed (and captured) on every _forward_step.
        if attn_meta is not None:
            attn_meta.on_update_kv_lens()

        if enable_spec_decode and not self._config.disable_overlap_scheduler:
            # When enabling overlap scheduler, the kv cache for draft tokens will
            # be prepared in advance by using the max_total_draft_tokens. But we need to use
            # new_tokens_lens_device to get the real past kv lengths and the
            # correct position ids. And to avoid blocking the async data transfer,
            # we need to preprocess the inputs in forward to update the position_ids and
            # kv cache length.
            if inputs["attn_metadata"].kv_cache_manager is not None:
                num_seqs = inputs["attn_metadata"].num_seqs
                num_ctx_requests = inputs["attn_metadata"].num_contexts
                num_gen_requests = inputs["attn_metadata"].num_generations
                num_ctx_tokens = inputs["attn_metadata"].num_ctx_tokens
                num_chunked_ctx_requests = inputs["attn_metadata"].num_chunked_ctx_requests
                previous_batch_tokens = inputs["input_ids"].shape[0] - num_ctx_tokens
                if inputs["position_ids"].ndim == 3:  # mrope: [3, 1, N]
                    inputs["position_ids"][:, :, num_ctx_tokens:] += (
                        self._buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )
                else:
                    inputs["position_ids"][0, num_ctx_tokens:] += (
                        self._buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )

                if hasattr(inputs["attn_metadata"], "kv_lens_cuda"):
                    if (
                        num_ctx_requests >= num_chunked_ctx_requests
                        and num_chunked_ctx_requests > 0
                    ):
                        # The generation requests with draft_tokens are treated as chunked context
                        # requests when extend_ctx returns True.
                        inputs["attn_metadata"].kv_lens_cuda[
                            num_ctx_requests - num_chunked_ctx_requests : num_ctx_requests
                        ] += self._buffers.previous_kv_lens_offsets_cuda[:num_chunked_ctx_requests]
                    else:
                        inputs["attn_metadata"].kv_lens_cuda[num_ctx_requests:num_seqs] += (
                            self._buffers.previous_kv_lens_offsets_cuda[:num_gen_requests]
                        )
                    inputs["attn_metadata"].on_update_kv_lens()
                # TRTLLM uses `kv_lens_cuda` above; FlashInfer exposes this backend-specific
                # correction without coupling the engine to its metadata type.
                elif hasattr(inputs["attn_metadata"], "apply_spec_decode_kv_lens_offsets"):
                    inputs["attn_metadata"].apply_spec_decode_kv_lens_offsets(
                        self._buffers.previous_kv_lens_offsets_cuda,
                        num_gen_requests,
                        self.get_runtime_tokens_per_gen_step(runtime_draft_len),
                        num_chunked_contexts=num_chunked_ctx_requests,
                    )

        if enable_spec_decode and self.mapping.has_cp_helix():
            # Helix verify groups: the per-token device buffers (write slots,
            # attention bounds, rank-local kv lens) must be derived on EVERY
            # spec step, overlap or not -- the append/mask kernels consume
            # them whenever _helix_spec_tokens_valid is armed. Under overlap
            # the host packed provisional positions from a stale base, so
            # first apply the same accepted-count correction position_ids
            # got above; without overlap the host values are already exact.
            md = inputs.get("attn_metadata")
            if (
                md is not None
                and md.kv_cache_manager is not None
                and getattr(md, "_helix_spec_tokens_valid", False)
            ):
                helix_gen_tokens = inputs["input_ids"].shape[0] - md.num_ctx_tokens
                if not self._config.disable_overlap_scheduler:
                    # The kv_lens override in the recompute supersedes the
                    # generic previous_kv_lens_offsets adjustment above,
                    # which is not ownership-aware.
                    md.helix_position_offsets[:helix_gen_tokens] += (
                        self._buffers.previous_pos_id_offsets_cuda[:helix_gen_tokens]
                    )
                md.recompute_helix_spec_buffers(
                    helix_gen_tokens, self.get_runtime_tokens_per_gen_step(runtime_draft_len)
                )
                md.on_update_kv_lens()

        if self.guided_decoder is not None:
            self.guided_decoder.token_event.record()

        return inputs

    def _postprocess_inputs(
        self, inputs: dict[str, Any], *, enable_spec_decode: bool, runtime_draft_len: int
    ):
        """
        Postprocess to make sure model forward doesn't change the inputs.
        It is only used in cuda graph capture, because other cases will prepare
        new inputs before the model forward.
        """
        if enable_spec_decode and not self._config.disable_overlap_scheduler:
            if inputs["attn_metadata"].kv_cache_manager is not None:
                num_seqs = inputs["attn_metadata"].num_seqs
                num_ctx_requests = inputs["attn_metadata"].num_contexts
                num_gen_requests = inputs["attn_metadata"].num_generations
                num_ctx_tokens = inputs["attn_metadata"].num_ctx_tokens
                num_chunked_ctx_requests = inputs["attn_metadata"].num_chunked_ctx_requests
                previous_batch_tokens = inputs["input_ids"].shape[0] - num_ctx_tokens
                if inputs["position_ids"].ndim == 3:  # mrope: [3, 1, N]
                    inputs["position_ids"][:, :, num_ctx_tokens:] -= (
                        self._buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )
                else:
                    inputs["position_ids"][0, num_ctx_tokens:] -= (
                        self._buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )

                # Only TrtllmAttentionMetadata has kv_lens_cuda.
                if isinstance(inputs["attn_metadata"], TrtllmAttentionMetadata):
                    if (
                        num_ctx_requests >= num_chunked_ctx_requests
                        and num_chunked_ctx_requests > 0
                    ):
                        inputs["attn_metadata"].kv_lens_cuda[
                            num_ctx_requests - num_chunked_ctx_requests : num_ctx_requests
                        ] -= self._buffers.previous_kv_lens_offsets_cuda[:num_chunked_ctx_requests]
                    else:
                        inputs["attn_metadata"].kv_lens_cuda[num_ctx_requests:num_seqs] -= (
                            self._buffers.previous_kv_lens_offsets_cuda[:num_gen_requests]
                        )
                # Restore the FlashInfer-specific logical KV lengths through the same optional hook
                # used by `_preprocess_inputs`.
                elif hasattr(inputs["attn_metadata"], "apply_spec_decode_kv_lens_offsets"):
                    inputs["attn_metadata"].apply_spec_decode_kv_lens_offsets(
                        self._buffers.previous_kv_lens_offsets_cuda,
                        num_gen_requests,
                        self.get_runtime_tokens_per_gen_step(runtime_draft_len),
                        num_chunked_contexts=num_chunked_ctx_requests,
                        restore=True,
                    )

                if self.mapping.has_cp_helix() and getattr(
                    inputs["attn_metadata"], "_helix_spec_tokens_valid", False
                ):
                    # Mirror of the helix position correction in
                    # _preprocess_inputs (capture symmetry, like position_ids
                    # above). The recompute's OVERWRITES (slots/bounds/
                    # kv_lens) need no reversal: every consumer buffer is
                    # rewritten from host state at the next step's prepare.
                    inputs["attn_metadata"].helix_position_offsets[:previous_batch_tokens] -= (
                        self._buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )

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
        if not (self.mapping.enable_lm_head_tp_in_adp and spec_metadata.use_rejection_sampling):
            return
        local_flag = bool(spec_metadata.is_all_greedy_sample)
        all_flags = self.dist.tp_allgather_int64([local_flag])[:, 0]
        spec_metadata.group_all_greedy_sample = bool(all_flags.all())
        # Also overwrite the live flag directly: this iteration's scan already
        # ran (update_is_all_greedy_sample just returned) and the CUDA graph
        # key reads the flag next -- the stored override only takes effect on
        # the NEXT rescan (populate), which is after key selection.
        spec_metadata.is_all_greedy_sample = spec_metadata.group_all_greedy_sample

    def _is_final_multimodal_context_decode_compatible(self, request: LlmRequest) -> bool:
        """Return whether the final prompt token uses the decode input path.

        KV reuse has already materialized every preceding prompt token. A
        multimodal final-context row therefore needs its prepared embedding
        only when the one remaining token is itself an MM placeholder. Text
        tokens can use the existing decode provider; MRoPE deltas are seeded
        into the per-sequence cache before graph lookup. An MRoPE request with
        real MM payload remains eager until its delta is available.
        """
        final_prompt_token = request.get_tokens(0)[request.context_current_position]
        _, mm_token_indices = prepare_multimodal_indices([final_prompt_token], model=self.model)
        if mm_token_indices.numel() != 0:
            return False

        multimodal_data = request.py_multimodal_data
        if not self._config.use_mrope or not _has_mm_payload_keys(multimodal_data):
            return True
        return CUDAGraphRunner._get_mrope_position_delta(request) is not None

    @functools.cached_property
    def _model_uses_ple_recurrent_state(self) -> bool:
        """Detect PLE on text-only and multimodal model wrappers.

        The answer is fixed once the model is loaded, and the CUDA-graph gate
        below consults it on every forward that has context requests.
        """
        top_level_model = get_top_level_model(self.model)
        if getattr(top_level_model, "has_ple", False):
            return True
        llm = getattr(top_level_model, "llm", None)
        text_model = getattr(llm, "model", llm)
        return bool(getattr(text_model, "has_ple", False))

    def _prepare_inputs_fast_path(
        self,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        attn_metadata: AttentionMetadata,
        new_tokens_device: torch.Tensor | None,
        next_draft_tokens_device: torch.Tensor | None,
        resource_manager: ResourceManager | None,
        *,
        promoted_context_request_ids: frozenset[int],
        enable_spec_decode: bool,
    ) -> PreparedInputs | None:
        """Return the prepared inputs when a specialized path prepares the batch."""
        return None

    def _forward_warmup(
        self,
        batch: ScheduledRequests,
        resource_manager: ResourceManager,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
    ):
        """Run a dummy scheduled pass with this call's speculation values."""
        inputs = make_scheduled_inputs(
            batch,
            None,
            None,
            enable_spec_decode=enable_spec_decode,
            runtime_draft_len=runtime_draft_len,
        )
        return self.forward(inputs, resource_manager=resource_manager, is_dummy=True)

    def _forward_decoder(
        self,
        forward_inputs: ScheduledInputs,
        resource_manager: ResourceManager,
        *,
        is_dummy: bool,
    ) -> Any:
        scheduled_requests = forward_inputs.batch
        new_tensors_device = forward_inputs.new_tensors_device
        cache_indirection_buffer = forward_inputs.cache_indirection_buffer
        gather_context_logits = forward_inputs.gather_context_logits
        enable_spec_decode = forward_inputs.enable_spec_decode
        runtime_draft_len = forward_inputs.runtime_draft_len
        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        assert kv_cache_manager is not None, "the legacy runner requires a KV cache manager"
        semantic_requests = scheduled_requests
        scheduled_requests = self._get_attention_dp_execution_batch(scheduled_requests)
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(resource_manager)

        attn_metadata = self._metadata.set_up_attn(kv_cache_manager, draft_kv_cache_manager)
        if isinstance(attn_metadata, TrtllmAttentionMetadata):
            attn_metadata.trtllm_gen_jit_warmup = self._trtllm_gen_jit_warmup
        if enable_spec_decode:
            spec_resource_manager, spec_tree_manager = get_spec_managers(resource_manager)
            spec_metadata = self._metadata.set_up_spec(spec_resource_manager)
            assert spec_metadata is not None
            update_spec_metadata(
                spec_metadata,
                self._config,
                scheduled_requests,
                attn_metadata,
                spec_tree_manager=spec_tree_manager,
                runtime_draft_len=runtime_draft_len,
            )
        else:
            spec_resource_manager = None
            spec_metadata = None

        moe_load_balancer = self.moe_load_balancer
        from tensorrt_llm._torch.pyexecutor.ced_replay import uses_private_decoder_cache

        ced_enabled = uses_private_decoder_cache(
            getattr(self.model, "model", None), kv_cache_manager
        )
        graph_requests = scheduled_requests
        promoted_context_request_ids: frozenset[int] = frozenset()
        # Non-linear tree decoding expands runtime_draft_len to the total
        # tree width after graph selection. Only linear-tree zero-draft
        # iterations can therefore safely reuse a zero-draft graph.
        can_promote_spec_decode = not enable_spec_decode or (
            runtime_draft_len == 0
            and self._config.spec_config is not None
            and self._config.spec_config.is_linear_tree
        )
        # TODO: Generalize these conservative gates as actual-draft, beam, and
        # context-parallel providers for decoder-only LLMs gain support for
        # promoted final-context rows. Each relaxation must preserve whole-batch
        # fallback on graph miss and prove parity with the provider's native
        # q_len=1 path. Encoder-decoder and non-LLM engines remain out of scope.
        if (
            scheduled_requests.num_context_requests > 0
            and scheduled_requests.attention_dp_phase is not RemoteTailPhase.CONTEXT
            # A physically pruned context cache cannot become a decode
            # batch, including single-token synthetic prefill warmup rows.
            and getattr(kv_cache_manager, "context_swa_layer_limit", None) is None
            and self.cuda_graph_runner.enabled
            and can_promote_spec_decode
            and not ced_enabled
            and not self.use_beam_search
            and self._new_extra_inputs is None
            # PLE owns recurrent n-gram and convolution state. Promoting a
            # fresh final-context row would skip its cache-slot reset.
            and not self._model_uses_ple_recurrent_state
            and self.mapping.cp_size == 1
        ):
            graph_requests, promoted_context_request_ids = _make_single_token_context_graph_batch(
                scheduled_requests, self._is_final_multimodal_context_decode_compatible
            )

        with self.cuda_graph_runner.pad_batch(
            graph_requests, resource_manager, runtime_draft_len
        ) as padded_graph_requests:
            # Callee already no-ops when use_mrope=False, but the Python call /
            # frame setup itself is non-trivial under high concurrency. Gating
            # at the caller avoids that overhead for non-mrope models.
            if self._config.use_mrope:
                self._inputs.pad_batch_seed_mrope_delta_cache(padded_graph_requests)

            # Refresh is_all_greedy_sample for the *current* batch BEFORE the
            # CUDA graph key is built below. The key includes this flag to pick
            # the argmax vs advanced-sampling graph variant; populate (inside
            # _prepare_inputs) runs later and fills the matching GPU buffers.
            # Without this pre-scan the key would use the previous iteration's
            # stale value and could replay the advanced graph against
            # unpopulated (greedy) buffers, hanging the run (e.g. MTP nextn>=2).
            if spec_metadata is not None:
                spec_metadata.update_is_all_greedy_sample(padded_graph_requests.all_requests())
                self._sync_group_all_greedy_sample(spec_metadata)

            peft_cache_data_type = None
            if self._lora.cuda_graph_manager is not None:
                peft_cache_manager = resource_manager.get_resource_manager(
                    ResourceManagerType.PEFT_CACHE_MANAGER
                )
                peft_cache_data_type = peft_cache_manager.data_type

            use_lora_graph = self._use_lora_cuda_graph(padded_graph_requests)
            maybe_attn_metadata, maybe_spec_metadata, key = (
                self.cuda_graph_runner.maybe_get_cuda_graph(
                    padded_graph_requests,
                    enable_spec_decode=enable_spec_decode,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    draft_tokens_cuda=self._buffers.draft_tokens_cuda
                    if self._config.is_spec_decode
                    else None,
                    new_tensors_device=new_tensors_device,
                    spec_resource_manager=spec_resource_manager,
                    promoted_context_request_ids=promoted_context_request_ids,
                    peft_cache_data_type=peft_cache_data_type,
                    use_lora_graph=use_lora_graph,
                )
            )

            can_run_graph = key is not None and not (
                ced_enabled and scheduled_requests.num_context_requests > 0
            )
            if can_run_graph:
                attn_metadata = maybe_attn_metadata
                spec_metadata = maybe_spec_metadata
                execution_requests = padded_graph_requests
                execution_promoted_context_ids = promoted_context_request_ids
            else:
                attn_metadata = self._metadata.attn_metadata
                if enable_spec_decode:
                    spec_metadata = self._metadata.spec_metadata
                else:
                    spec_metadata = None
                execution_requests = scheduled_requests
                execution_promoted_context_ids = frozenset()

            # Stage in-graph sampling now that the batch is settled: the staged
            # scatter width has to match the batch the forward actually runs on,
            # which differs between the graph (padded) and eager (unpadded)
            # branches above. Falling back to eager also means no graph replays,
            # so the tier is dropped and the sampler runs after the forward.
            #
            # Promoted one-token contexts join the graph batch's generation
            # requests, but sample_async is handed the original scheduled batch,
            # which excludes them. Staging a tier here would sample rows that
            # are then discarded and advance those requests' Philox streams an
            # extra time, so keep those steps on the eager path.
            if self._stage_in_graph_sampling is not None:
                staged_sample_type = (
                    key.sample_type
                    if can_run_graph and not execution_promoted_context_ids
                    else SampleType.FULL
                )
                self._stage_in_graph_sampling(execution_requests, staged_sample_type)

            # Fill slot-ID buffer for scatter inside draft loop
            if enable_spec_decode and spec_tree_manager is not None:
                spec_tree_manager.slot_storage.fill_all_slot_ids(
                    execution_requests.context_requests,
                    execution_requests.generation_requests,
                )

            with self._cuda_graph_metadata_prepare_scope(
                attn_metadata,
                key,
                execution_requests,
                execution_promoted_context_ids,
                enable_spec_decode=enable_spec_decode,
            ):
                prepared = self._inputs.prepare_inputs(
                    execution_requests,
                    kv_cache_manager,
                    attn_metadata,
                    spec_metadata,
                    new_tensors_device,
                    cache_indirection_buffer,
                    resource_manager,
                    can_run_graph,
                    execution_promoted_context_ids,
                    use_lora_graph=use_lora_graph,
                    enable_spec_decode=enable_spec_decode,
                    runtime_draft_len=runtime_draft_len,
                    is_dummy=is_dummy,
                    gather_context_logits=gather_context_logits,
                )
            inputs = prepared.kwargs
            runtime_draft_len = resolve_runtime_draft_len(
                self._config.spec_config if enable_spec_decode else None, runtime_draft_len
            )
            if execution_promoted_context_ids:
                self.iter_states["num_ctx_requests"] = scheduled_requests.num_context_requests
                self.iter_states["num_ctx_tokens"] = sum(
                    request.context_chunk_size for request in scheduled_requests.context_requests
                )
                self.iter_states["num_generation_tokens"] = (
                    scheduled_requests.num_generation_requests
                )
            self._prepare_inputs_event = torch.cuda.Event()
            self._prepare_inputs_event.record()

            breakable_runner = self.breakable_cuda_graph_runner

            with with_shared_pool(self.cuda_graph_runner.get_graph_pool()):

                def forward_step():
                    with MoeLoadBalancerIterContext(moe_load_balancer):
                        return self._forward_step(
                            inputs,
                            enable_spec_decode=enable_spec_decode,
                            runtime_draft_len=runtime_draft_len,
                            is_dummy=is_dummy,
                            gather_ids=prepared.gather_ids,
                            gather_context_logits=gather_context_logits,
                        )

                if not can_run_graph:
                    if breakable_runner is not None and breakable_runner.is_capturing:
                        return breakable_runner.capture_model_body(forward_step)

                    num_tokens = inputs["input_ids"].shape[0]
                    can_run_breakable_graph = (
                        breakable_runner is not None
                        and get_per_request_prefill_cuda_graph_flag()
                        and not gather_context_logits
                        and breakable_runner.has_graph(num_tokens)
                    )
                    if can_run_breakable_graph and not breakable_runner.is_warming_up:
                        outputs = breakable_runner.execute(num_tokens, forward_step)
                    else:
                        # real eager or BCG warmup or PCG
                        outputs = forward_step()
                else:
                    position_id_offsets = None
                    if (
                        enable_spec_decode
                        and not self._config.disable_overlap_scheduler
                        and attn_metadata.kv_cache_manager is not None
                    ):
                        position_id_offsets = self._buffers.previous_pos_id_offsets_cuda
                    needs_capture = self.cuda_graph_runner.needs_capture(key)
                    if needs_capture:

                        def capture_forward_fn(inputs: dict[str, Any]):
                            with MoeLoadBalancerIterContext(moe_load_balancer):
                                return self._forward_step(
                                    inputs,
                                    enable_spec_decode=enable_spec_decode,
                                    runtime_draft_len=runtime_draft_len,
                                    is_dummy=is_dummy,
                                    gather_ids=prepared.gather_ids,
                                    gather_context_logits=gather_context_logits,
                                )

                        def capture_postprocess_fn(inputs: dict[str, Any]):
                            self._postprocess_inputs(
                                inputs,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=runtime_draft_len,
                            )

                        capture_outputs = self.cuda_graph_runner.capture(
                            key,
                            capture_forward_fn,
                            inputs,
                            enable_spec_decode=enable_spec_decode,
                            postprocess_fn=capture_postprocess_fn,
                        )

                    if self.cuda_graph_runner.is_warmup_only:
                        outputs = capture_outputs
                    elif needs_capture:
                        # Refresh attention metadata for the current batch's
                        # draft cache before replaying the captured graph.
                        saved_draft = prepare_attn_metadata_for_draft_replay(
                            attn_metadata, draft_kv_cache_manager
                        )
                        try:
                            outputs = self.cuda_graph_runner.replay(
                                key, inputs, position_id_offsets=position_id_offsets
                            )
                        finally:
                            restore_attn_metadata_after_draft_replay(attn_metadata, saved_draft)
                    else:
                        saved_draft = prepare_attn_metadata_for_draft_replay(
                            attn_metadata, draft_kv_cache_manager
                        )
                        try:
                            with MoeLoadBalancerIterContext(moe_load_balancer):
                                outputs = self.cuda_graph_runner.replay(
                                    key, inputs, position_id_offsets=position_id_offsets
                                )
                        finally:
                            restore_attn_metadata_after_draft_replay(attn_metadata, saved_draft)

            if self.forward_pass_callable is not None:
                self.forward_pass_callable()

            self._execute_logit_post_processors(semantic_requests, outputs)

            if not isinstance(outputs, dict):
                return outputs
            outputs = self._restore_attention_dp_semantic_outputs(semantic_requests, outputs)
            return {**outputs, "runtime_draft_len": runtime_draft_len}

    def model_forward(self, *, is_dummy: bool, **kwargs):
        assert self._model_caller is not None
        reclaimer = self._eager_workspace_reclaimer
        metadata = kwargs["attn_metadata"]
        reclaim_scope = (
            reclaimer.forward(metadata)
            if reclaimer is not None
            and not is_dummy
            and isinstance(metadata, TrtllmAttentionMetadata)
            else contextlib.nullcontext()
        )
        with reclaim_scope:
            return self._model_caller(**kwargs)

    @nvtx_range("_forward_step")
    def _forward_step(
        self,
        inputs: dict[str, Any],
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
        is_dummy: bool,
        gather_ids: torch.Tensor | None = None,
        gather_context_logits: bool = False,
    ) -> dict[str, Any]:
        inputs = self._preprocess_inputs(
            inputs, enable_spec_decode=enable_spec_decode, runtime_draft_len=runtime_draft_len
        )
        if inputs.get("spec_metadata", None):
            gather_ids = inputs["spec_metadata"].gather_ids

        # For simplicity, just return all the the logits if we have special gather_ids
        # from speculative decoding.
        dspark_metadata = inputs.get("spec_metadata")
        if isinstance(dspark_metadata, DSparkSpecMetadata):
            # CED retains the final context row and all verification rows.
            # Distinguish that sparse gather from a real all-context request.
            dspark_metadata._requires_all_context_logits = bool(gather_context_logits)
        try:
            outputs = self.model_forward(
                **inputs,
                is_dummy=is_dummy,
                return_context_logits=gather_ids is not None or gather_context_logits,
            )
        finally:
            if isinstance(dspark_metadata, DSparkSpecMetadata):
                # Do not let a later direct caller inherit engine-only proof
                # that dropping non-final context rows is safe.
                dspark_metadata._requires_all_context_logits = None

        if self._config.without_logits:
            return outputs

        if isinstance(outputs, dict):
            # If the model returns a dict, get the logits from it. All other keys are kept.
            logits = outputs.get("logits", None)
            # If the logits are not found, no further processing is needed.
            if logits is None:
                return outputs
        else:
            # If the model returns a single tensor, assume it is the logits and wrap it in a dict.
            logits = outputs
            outputs = {"logits": logits}

        # If we have special gather_ids, gather the logits
        if gather_ids is not None:
            outputs["logits"] = logits[gather_ids]

        # Sample at the tail of the forward pass, so that under CUDA graph
        # capture the sampling kernels are recorded as part of this graph. The
        # hook is a no-op unless the sampler staged a graph-capturable tier for
        # this batch.
        #
        # Only the last PP rank runs the LM head. The others get a placeholder
        # that is the right shape but never filled, so a shape check waves it
        # through and they would sample garbage every step -- discarded, but
        # not free. _execute_logit_post_processors skips them for this reason.
        if self.sample_in_graph_callable is not None and self.mapping.is_last_pp_rank():
            self.sample_in_graph_callable(outputs)

        return outputs

    @staticmethod
    def _apply_logits_processors(
        request, logits_processors, logits_tensor, beam_width, token_ids, logits_row_offset
    ):
        logits_rows = logits_tensor[logits_row_offset : logits_row_offset + beam_width]
        # Reshape to align w/ the shape used in the TRT backend,
        # so the same logit processors can be used across both backends.
        logits_rows = logits_rows.view(beam_width, 1, -1)
        for lp in logits_processors:
            lp_params = inspect.signature(lp).parameters

            assert 4 <= len(lp_params) <= 5, (
                "Logit post processor signature must match the `LogitsProcessor` interface "
                "defined in `tensorrtllm.sampling_params`."
            )
            lp(request.py_request_id, logits_rows, token_ids, None, None)

        # logits_rows is a view into logits_tensor (narrow + view never
        # copy), so the processors already mutated it in place. Writing it
        # back would be a self-assignment, which torch rejects for the
        # non-contiguous slices a TP-padded vocab produces.

    def _execute_logit_post_processors(self, scheduled_requests: ScheduledRequests, outputs: dict):
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
                    beam_width = request.get_beam_width_by_iter(for_next_iteration=False)
                    row_stride = request.py_beam_width

                logits_processors = getattr(request, "py_logits_post_processors", None)
                if logits_processors:
                    token_ids = (
                        [request.get_tokens(0)]
                        if is_context_request
                        else [request.get_tokens(beam_idx) for beam_idx in range(beam_width)]
                    )
                    if is_context_request and request.py_orig_prompt_len < len(token_ids[0]):
                        # Skip as we only need to apply logit processor on the last context request
                        logits_row_offset += row_stride
                        continue

                    self._apply_logits_processors(
                        request,
                        logits_processors,
                        logits_tensor,
                        beam_width,
                        token_ids,
                        logits_row_offset,
                    )
                logits_row_offset += row_stride

    def _wait_for_decoder_input_copy(self) -> None:
        if self._prepare_inputs_event is not None:
            self._prepare_inputs_event.synchronize()

    def _prewarm_engram_kernels(
        self, kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None
    ) -> None:
        """Compile Engram kernels before conditional model warmup."""
        from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
            CSA2CacheManager,
        )

        if not isinstance(kv_cache_manager, CSA2CacheManager):
            return

        from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41Engram

        with torch.inference_mode():
            for module in self.model.modules():
                if isinstance(module, DeepseekV41Engram):
                    module.warmup_kernels()
        torch.cuda.synchronize()
        # Idle attention-DP ranks also finish JIT before collective forwards.
        if self.mapping.tp_size > 1 and self.dist is not None:
            self.dist.tp_allgather(1)
        logger.info("Engram kernel prewarm complete")

    def _prewarm_csa2_sparse_indexer(
        self, kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None
    ) -> None:
        """Compile the DeepGEMM sparse MQA-logits kernels of the CSA2 candidate
        consumers before the first collective-bearing forward.

        Like `_warmup_dg_paged_mqa_logits_metadata` for DSA, this moves
        deep_gemm's JIT (nvcc/ptxas, several seconds per kernel) out of the
        warm-up forwards, where it would count against the MoE all-to-all
        completion deadline. The kernels are specialized on the candidate
        geometry only, so one compile per kernel covers every layer and batch.
        Best-effort: a failure here only defers the compile to the first use.
        """
        from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
            CSA2CacheManager,
        )

        if not isinstance(kv_cache_manager, CSA2CacheManager):
            return

        from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import (
            prewarm_sparse_candidate_kernels,
        )

        config = self.model.model_config.pretrained_config
        with torch.inference_mode():
            try:
                compiled = prewarm_sparse_candidate_kernels(
                    kv_cache_manager.layout,
                    config.index_n_heads,
                    config.index_head_dim,
                    torch.device("cuda"),
                )
            except RuntimeError as e:
                logger.warning(f"CSA2 sparse indexer prewarm skipped. {type(e).__name__}: {e}")
                compiled = False
        torch.cuda.synchronize()
        if self.mapping.tp_size > 1 and self.dist is not None:
            self.dist.tp_allgather(1)
        if compiled:
            logger.info("CSA2 sparse indexer prewarm complete")

    def _cuda_graph_metadata_prepare_scope(
        self,
        attn_metadata: AttentionMetadata,
        key,
        scheduled_requests: ScheduledRequests,
        promoted_context_request_ids: frozenset[int],
        *,
        enable_spec_decode: bool,
    ):
        """Allow derived outputs only when this forward owns a decode graph.

        Both capture/warmup and replay execute _preprocess_inputs before model
        consumers. Non-speculative overlap has the same ordering; speculative
        acceptance corrections and draft-manager swaps keep full preparation.
        """
        defer = getattr(attn_metadata, "defer_cuda_graph_decode_outputs", None)
        if (
            defer is not None
            and isinstance(key, KeyType)
            and attn_metadata.is_cuda_graph is True
            and key.num_contexts == 0
            and key.draft_len == 0
            and scheduled_requests.num_context_requests == 0
            and scheduled_requests.num_generation_requests > 0
            and not promoted_context_request_ids
            and not enable_spec_decode
            and not self.use_beam_search
            and self._new_extra_inputs is None
            and self.mapping.cp_size == 1
        ):
            return defer()
        return contextlib.nullcontext()

    def _get_attention_dp_execution_batch(
        self, scheduled_requests: ScheduledRequests
    ) -> ScheduledRequests:
        """Give a phase-idle rank its retained graph row without admitting work."""
        if not scheduled_requests.is_attention_dp_phase_idle:
            return scheduled_requests
        if scheduled_requests.batch_size or scheduled_requests.encoder_requests:
            raise ValueError("An attention-DP phase-idle batch must be semantically empty")
        dummy = self.cuda_graph_runner.padding_dummy_requests.get(0)
        if dummy is None:
            raise RuntimeError("The retained attention-DP execution row is missing")
        execution_requests = ScheduledRequests()
        execution_requests.attention_dp_phase = scheduled_requests.attention_dp_phase
        execution_requests.is_attention_dp_phase_idle = True
        execution_requests.generation_requests = [dummy]
        return execution_requests

    @staticmethod
    def _restore_attention_dp_semantic_outputs(
        scheduled_requests: ScheduledRequests, outputs: dict[str, Any]
    ) -> dict[str, Any]:
        if not scheduled_requests.is_attention_dp_phase_idle:
            return outputs
        # The sampler sees the original empty batch, never the execution row.
        # Copy the map because CUDA graph outputs are retained across replays.
        return {**outputs, "logits": outputs["logits"][:0]}
