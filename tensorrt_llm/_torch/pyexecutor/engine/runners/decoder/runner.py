# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder execution moved out of ``PyTorchModelEngine``."""

import contextlib
import functools
import gc
import inspect
import math
import os
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import Any

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.compilation.piecewise_optimizer import PiecewiseRunner
from tensorrt_llm._torch.compilation.utils import capture_piecewise_cuda_graph
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.memory_buffer_utils import clear_memory_buffers, with_shared_pool
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.models.modeling_multimodal_mixin import _build_request_multimodal_input
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
    get_mrope_dummy_seq_slot,
)
from tensorrt_llm._torch.pyexecutor.guided_decoder import CapturableGuidedDecoder
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    BaseMambaCacheManager,
    MambaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager,
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTensors
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import warmup_sampling_module
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.trace_log_utils import log_mem_snapshot
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._torch.pyexecutor.workspace import EagerWorkspaceReclaimer
from tensorrt_llm._torch.speculative import (
    SpecMetadata,
    get_draft_kv_cache_manager,
    get_num_extra_kv_tokens,
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)
from tensorrt_llm._torch.speculative.interface import INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec
from tensorrt_llm._torch.speculative.utils import get_static_draft_len, resolve_draft_len
from tensorrt_llm._torch.utils import (
    get_per_request_prefill_cuda_graph_flag,
    helix_local_len,
    set_per_request_prefill_cuda_graph_flag,
    with_model_extra_attrs,
)
from tensorrt_llm._utils import global_mpi_rank, maybe_pin_memory, nvtx_range, prefer_pinned
from tensorrt_llm.inputs.multimodal import (
    MultimodalParams,
    MultimodalRuntimeData,
    _has_mm_payload_keys,
    check_mm_embed_cumsum_if_needed,
    strip_mm_data_for_generation,
)
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor, InputProcessor
from tensorrt_llm.llmapi.llm_args import (
    BaseSparseAttentionConfig,
    DecodingBaseConfig,
    PrefillCudaGraphBackend,
    SeqLenAwareSparseAttentionConfig,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType, Mapping
from tensorrt_llm.sampling_params import SamplingParams

from ...lora import LoraParamBuilder, make_cuda_graph_lora_manager
from ...metadata import build_attention_metadata
from ...model_call import ModelCaller
from ..common import (
    apply_position_id_offset,
    get_all_rank_num_tokens,
    get_padding_params,
    get_position_id_offset,
    get_top_level_model,
    make_scheduled_inputs,
    moe_a2a_steady_state_budget_for_capture,
    prepare_multimodal_indices,
    resolve_mrope_position_deltas_cache,
    ship_multimodal_indices,
)
from ..interface import ScheduledInputs, ScheduledModelRunner
from .config import DecoderRunnerConfig
from .speculative import (
    create_spec_metadata,
    get_spec_managers,
    set_spec_metadata_all_rank_num_tokens,
    update_spec_metadata,
)


def _get_context_prompt_lookahead_token(request: LlmRequest, chunk_end: int) -> int:
    """Prompt token immediately following a context chunk. Uses the live C++
    ``mPromptLen``; ``py_prompt_len`` goes stale after a non-recompute preemption.
    """
    if request.is_last_context_chunk:
        return INVALID_PROMPT_LOOKAHEAD_TOKEN
    return request.get_token(0, chunk_end)


def resolve_mamba_metadata_cls(model: torch.nn.Module) -> type[Mamba2Metadata]:
    """Resolve the model-specific Mamba metadata class with a default."""
    return getattr(model, "mamba_metadata_cls", None) or Mamba2Metadata


def uses_full_generation_page_table(
    disable_overlap_scheduler: bool,
    spec_config: DecodingBaseConfig | None,
    attn_metadata,
) -> bool:
    """Return whether overlap decode needs every reserved generation page.

    ``attn_metadata`` may be an ``AttentionMetadata`` instance or its class:
    only the presence of the correction hook is read, so the executor creator
    can evaluate this before any metadata exists.
    """
    # FlashInfer metadata owns the optional device-side KV-length correction used with this
    # wider page table.
    return (
        not disable_overlap_scheduler
        and getattr(spec_config, "_use_shared_kv_cache", False)
        and hasattr(attn_metadata, "apply_spec_decode_kv_lens_offsets")
    )


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


# Arbitrary non-greedy params used to force the advanced-sampling CUDA graph
# warmup capture path.
NON_GREEDY_CAPTURE_SAMPLING_PARAMS = SamplingParams(
    temperature=0.7, top_k=50, top_p=0.9, min_p=0.05
)


class ExtraInputsCollector:
    """Collect per-sequence model inputs beyond the decoder's own; none by default."""

    def add_context_request(self, request: LlmRequest) -> None:
        pass

    def add_generation_request(self, request: LlmRequest, repeat: int = 1) -> None:
        pass

    def build(
        self, attn_metadata: AttentionMetadata, resource_manager: ResourceManager | None
    ) -> dict[str, Any]:
        return {}


_NO_EXTRA_INPUTS = ExtraInputsCollector()


class DecoderRunner(ScheduledModelRunner):
    """Run scheduled decoder generation with its own buffers, graphs and metadata."""

    # Optional execution paths that a subclass can turn off.
    _context_warmups_supported = True
    _context_graph_promotion_supported = True
    _steady_gen_cache_supported = True
    _eager_workspace_reclaim_supported = True
    # Prompt tokens of the filler CUDA graph dummy requests, also the floor of
    # the longest one; None keeps the cache manager's default.
    _dummy_request_tokens: int | None = None

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
    ) -> None:
        self.model = model
        self.input_processor = input_processor
        self._model_caller = model_caller
        self.mapping = mapping
        self.dist = dist
        self.moe_load_balancer = moe_load_balancer
        self.sparse_attention_config = sparse_attention_config
        self._torch_compile_backend = torch_compile_backend
        self.get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step
        self._warmup_timer = warmup_timer
        self.iter_states = iter_states
        self._metrics = metrics
        self.kv_cache_manager_key = kv_cache_manager_key

        self._config = config

        self.guided_decoder: CapturableGuidedDecoder | None = None
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
        self.spec_metadata = None

        self.cache_indirection_attention = None
        self._allocate_decoder_buffers()
        self._init_decoder_state()
        self.cuda_graph_runner = self._initialize_cuda_graph_runner()
        self.breakable_cuda_graph_runner = self._initialize_breakable_cuda_graph_runner()

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
        self._release_decoder_graphs()

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

    def _allocate_decoder_buffers(self) -> None:
        """Allocate the persistent buffers read by engine decoder execution.

        CUDA graphs record these addresses, so they are allocated once here and
        only ever sliced or written in place.
        """
        if self._config.is_spec_decode:
            max_num_draft_tokens = self._config.max_draft_loop_tokens * self._config.max_batch_size
            self.draft_tokens_cuda = torch.empty(
                (max_num_draft_tokens,), dtype=torch.int, device="cuda"
            )
            self.gather_ids_cuda = torch.empty(
                (self._config.max_num_tokens,), dtype=torch.int, device="cuda"
            )
            self.num_accepted_draft_tokens_cuda = torch.empty(
                (self._config.max_batch_size,), dtype=torch.int, device="cuda"
            )
            self.previous_pos_indices_cuda = torch.empty(
                (self._config.max_num_tokens,), dtype=torch.int, device="cuda"
            )
            self.previous_pos_id_offsets_cuda = torch.zeros(
                (self._config.max_num_tokens,), dtype=torch.int, device="cuda"
            )
            self.previous_kv_lens_offsets_cuda = torch.zeros(
                (self._config.max_batch_size,), dtype=torch.int, device="cuda"
            )
        self.previous_batch_indices_cuda = torch.empty(
            (self._config.max_num_tokens,), dtype=torch.int, device="cuda"
        )
        self.input_ids_cuda = torch.empty(
            (self._config.max_num_tokens,), dtype=torch.int, device="cuda"
        )
        self.position_ids_cuda = torch.empty(
            (self._config.max_num_tokens,), dtype=torch.int, device="cuda"
        )
        # Host counter of the steady-state generation prepare cache.
        self._steady_gen_positions_pinned = torch.empty(
            (self._config.max_num_tokens,), dtype=torch.int, pin_memory=prefer_pinned()
        )
        if self._config.use_mrope:
            self.mrope_position_ids_cuda = torch.empty(
                (3, 1, self._config.max_num_tokens), dtype=torch.int, device="cuda"
            )
        if self.use_beam_search:
            self.cache_indirection_attention = torch.zeros(
                (
                    self._config.max_batch_size,
                    self._config.max_beam_width,
                    self._config.max_seq_len,
                ),
                device="cuda",
                dtype=torch.int32,
            )

    def _init_decoder_state(self) -> None:
        """Initialize the mutable state read only by engine decoder execution."""
        self._eager_workspace_reclaimer: EagerWorkspaceReclaimer | None = None
        # Steady-state generation-only prepare cache (non-speculative overlap
        # decode). Holds the per-request lists that are invariant while the
        # scheduled generation batch keeps the same composition, plus a pinned
        # cached-token counter advanced by one per step (host-side bookkeeping
        # only; the device position buffer is advanced in place and this
        # buffer is never the source of an async H2D). Invalidated (set to
        # None) by every full _prepare_tp_inputs pass.
        self._steady_gen_cache: dict[str, Any] | None = None
        # Log cached prefixes in model-input sequence order when enabled.
        self._log_cached_kv_tokens_per_req = (
            os.getenv("TLLM_LOG_CACHED_KV_TOKENS_PER_REQ", "0") == "1"
        )
        self._trtllm_gen_jit_warmup = False
        self._force_lora_graph_for_capture: bool | None = None
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
        return CUDAGraphRunner(self._cuda_graph_runner_config())

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

    def _initialize_breakable_cuda_graph_runner(self) -> BreakableCUDAGraphRunner | None:
        if (
            self.cuda_graph_runner is None
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
        if self._force_lora_graph_for_capture is not None:
            return self._force_lora_graph_for_capture
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

    @contextlib.contextmanager
    def no_cuda_graph(self):
        if self.cuda_graph_runner is None:
            yield
            return
        _run_cuda_graphs = self.cuda_graph_runner.enabled
        self.cuda_graph_runner.enabled = False
        try:
            yield
        finally:
            self.cuda_graph_runner.enabled = _run_cuda_graphs

    def _pad_batch_seed_mrope_delta_cache(self, padded_requests: ScheduledRequests) -> None:
        if not self._config.use_mrope or padded_requests.num_generation_requests == 0:
            return

        mrope_position_deltas_cache = resolve_mrope_position_deltas_cache(self.model)
        if mrope_position_deltas_cache is None:
            return

        mrope_seed_seq_slots = []
        mrope_seed_deltas = []
        mrope_seed_requests = []
        for request in padded_requests.generation_requests:
            if (
                request.py_seq_slot is None
                or request.is_dummy
                or getattr(request, "py_mrope_delta_cache_slot", None) == request.py_seq_slot
            ):
                continue
            mrope_position_delta = getattr(request, "py_mrope_position_delta", None)
            if mrope_position_delta is None and request.py_multimodal_data:
                mrope_config = request.py_multimodal_data.get("mrope_config")
                if mrope_config is not None:
                    mrope_position_delta = mrope_config.get("mrope_position_deltas")
            if mrope_position_delta is None:
                continue
            if mrope_position_delta.device.type == "cpu":
                mrope_position_delta = maybe_pin_memory(mrope_position_delta).to(
                    device="cuda", dtype=torch.int32, non_blocking=True
                )
            elif mrope_position_delta.dtype != torch.int32:
                mrope_position_delta = mrope_position_delta.to(dtype=torch.int32)
            request.py_mrope_position_delta = mrope_position_delta
            mrope_seed_seq_slots.append(request.py_seq_slot)
            mrope_seed_deltas.append(mrope_position_delta.reshape(1))
            mrope_seed_requests.append(request)

        if not mrope_seed_seq_slots:
            return

        mrope_seed_seq_slots_tensor = torch.tensor(
            mrope_seed_seq_slots, dtype=torch.long, pin_memory=prefer_pinned()
        ).to(device="cuda", non_blocking=True)
        mrope_seed_deltas_tensor = torch.cat(mrope_seed_deltas, dim=0)
        mrope_position_deltas_cache.index_copy_(
            0,
            mrope_seed_seq_slots_tensor,
            mrope_seed_deltas_tensor.to(dtype=mrope_position_deltas_cache.dtype),
        )
        for request in mrope_seed_requests:
            request.py_mrope_delta_cache_slot = request.py_seq_slot

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

    @contextmanager
    def maybe_autotune_lora(self):
        """Enable autotuning while warming up CUDA-graph LoRA kernels."""
        if not (self._config.enable_autotuner and self._lora.cuda_graph_manager is not None):
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
        can_run_general_warmup = self._agree_warmup_flag(
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
                warmup_requests_configs = self._agree_warmup_shapes(
                    self._get_full_general_warmup_requests(resource_manager)
                )
                # Currently graph has not been captured, disable cuda graph for this warmup.
                with self.no_cuda_graph():
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
                    self._run_cuda_graph_warmup(resource_manager)
                finally:
                    self.cuda_graph_runner.is_warmup_only = False
                self.cuda_graph_runner.padding_dummy_requests = {}
                self._run_cuda_graph_warmup(resource_manager)
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
        metadata = self.attn_metadata
        if (
            self._config.is_spec_decode
            or self.mapping.cp_size != 1
            or not self._eager_workspace_reclaim_supported
            or self.sparse_attention_config is not None
            or self._torch_compile_backend is not None
            or self.breakable_cuda_graph_runner is not None
        ):
            logger.info(
                "Eager workspace reclamation is disabled for speculative, "
                "CP, encoder-decoder, sparse, compiled, or breakable-graph models"
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
        self._set_up_attn_metadata(
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

    def _warmup_agreement_allgather(self) -> Callable[[int], list[int]] | None:
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

    def _agree_warmup_shapes(self, configs: list[tuple[int, int]]) -> list[tuple[int, int]]:
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
        shared = set.intersection(
            *({tuple(config) for config in rank_configs} for rank_configs in per_rank)
        )
        agreed = [config for config in configs if config in shared]
        dropped = [config for config in configs if config not in shared]
        if dropped:
            logger.warning(
                f"Dropping warmup shapes {dropped} that not every rank could "
                f"propose; per-rank KV capacity differs. Remaining: {agreed}."
            )
        return agreed

    def _should_run_warmup_batch(
        self, batch: ScheduledRequests | None, num_tokens: int, shape: str
    ) -> bool:
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
            logger.warning(f"Skipping warmup shape ({shape}): not enough KV cache space.")
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
                    f"cannot skip the shape without stranding them."
                )
            return True

        flags = list(allgather(int(batch is not None)))
        if all(flags):
            return True
        if not any(flags):
            # Every rank in the forward group is skipping, so none of them is
            # left inside a collective. This is the ordinary outcome for a
            # shape that does not fit the configuration at all, such as a
            # mixed context+generation shape under ``max_batch_size=1``.
            logger.info(
                f"Skipping warmup shape ({shape}) on all "
                f"{len(flags)} ranks: not enough KV cache space."
            )
            return False

        all_tokens = list(allgather(num_tokens))
        failed_ranks = [i for i, flag in enumerate(flags) if not flag]
        raise RuntimeError(
            f"Warmup batch creation failed for shape ({shape}) on rank(s) "
            f"{failed_ranks} but succeeded on others, so entering this forward "
            f"would deadlock the ranks that still hold a batch. Per-rank "
            f"curr_max_num_tokens: {all_tokens}. This indicates asymmetric KV "
            f"cache capacity across ranks. Consider increasing "
            f"--kv_cache_free_gpu_mem_fraction."
        )

    def _assert_all_tp_ranks_have_warmup_batch(self, batch, num_tokens: int) -> None:
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
                f"Consider increasing --kv_cache_free_gpu_mem_fraction."
            )

    def _general_warmup_impl(
        self, resource_manager: ResourceManager, warmup_requests_configs: list[tuple[int, int]]
    ) -> None:
        for num_tokens, num_gen_tokens in warmup_requests_configs:
            # Helix CP does not support warmup with context requests.
            if self.mapping.has_cp_helix() and num_tokens != num_gen_tokens:
                continue
            try:
                with self._release_batch_context(
                    self._create_warmup_request(resource_manager, num_tokens, num_gen_tokens),
                    resource_manager,
                ) as batch:
                    if not self._should_run_warmup_batch(
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
                if self._is_distributed_forward():
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
            warmup_request = self._create_warmup_request(
                resource_manager, num_tokens=num_tokens, num_gen_requests=num_gen_requests
            )

            with (
                self.no_cuda_graph(),
                self._release_batch_context(warmup_request, resource_manager) as batch,
            ):
                if not self._should_run_warmup_batch(
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
            with self.no_cuda_graph(), autotune_context:
                for num_tokens, num_gen_requests in warmup_configs:
                    warmup_request = self._create_warmup_request(
                        resource_manager, num_tokens, num_gen_requests
                    )
                    with self._release_batch_context(warmup_request, resource_manager) as batch:
                        if not self._should_run_warmup_batch(
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

        with self.no_cuda_graph(), autotune_ctx:
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
                        warmup_request = self._create_warmup_request(
                            resource_manager,
                            num_tokens_i,
                            num_gen_requests_i,
                            least_requests=least_req_i,
                            chunk_alignment=chunk_alignment,
                            chunk_aligned=chunk_aligned_i,
                        )
                    except torch.OutOfMemoryError as e:
                        if self._is_distributed_forward():
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
                        if self._is_distributed_forward():
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
                            if not self._should_run_warmup_batch(batch, num_tokens_i, shape):
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

    def _get_graphs_to_capture(self, cuda_graph_batch_sizes: list[int]) -> list[tuple[int, int]]:
        """Determine which (batch_size, draft_len) graphs to capture.

        Returns:
            List of (batch_size, draft_len) tuples for CUDA graph capture.
        """
        # Case 1: One-model with dynamic draft length
        if (
            self._config.spec_config is not None
            and self._config.spec_config.draft_len_schedule is not None
            and self._config.spec_config.spec_dec_mode.support_dynamic_draft_len()
        ):
            graphs = [
                (graph_bs, draft_len)
                for graph_bs, draft_len in self._dynamic_draft_len_mapping.items()
            ]
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
            max_spec_graph = (max(cuda_graph_batch_sizes), self._config.original_max_draft_len)
            if max_spec_graph not in graphs:
                graphs.append(max_spec_graph)
            logger.info(
                f"Dynamic draft length enabled for one-model path. "
                f"Capturing {len(graphs)} graphs: {graphs}"
            )
            return graphs

        # Case 2: Static draft length
        # Match the runtime_draft_len semantics enforced in _prepare_tp_inputs:
        # logical K for linear-tree modes, total tree tokens for tree decoding.
        # spec_config is None for non-spec models — fall back to max_draft_len (= 0).
        draft_lengths = [get_static_draft_len(self._config.spec_config)]
        should_capture_no_spec = (
            self._config.max_total_draft_tokens > 0
            and not self._config.spec_config.spec_dec_mode.use_one_engine()
            # Assume speculation is always on if no max_concurrency set (saves memory)
            and self._config.spec_config.max_concurrency is not None
        )
        if should_capture_no_spec:
            draft_lengths.append(0)
        return [(bs, draft_len) for bs in cuda_graph_batch_sizes for draft_len in draft_lengths]

    def _run_cuda_graph_warmup(self, resource_manager: ResourceManager):
        """Warm up or capture CUDA graphs for the configured graph shapes."""
        is_warmup_only = self.cuda_graph_runner.is_warmup_only
        metric_name = (
            "gen_cuda_graph_warmup_seconds" if is_warmup_only else "gen_cuda_graph_capture_seconds"
        )
        with timing_metric(metric_name, self._metrics):
            # Include LoRA autotuning and its PP cache hand-off/cleanup in
            # warmup timing, including when this rank has no graph shapes.
            lora_context = (
                self.maybe_autotune_lora() if is_warmup_only else contextlib.nullcontext()
            )
            with lora_context:
                if not (
                    self.cuda_graph_runner.enabled
                    or self._config.prefill_cuda_graph_backend != PrefillCudaGraphBackend.DISABLED
                ):
                    return

                from tensorrt_llm._torch.modules.linear import (
                    MXFP8LinearMethod,
                    flashinfer_mxfp8_autotune,
                    flashinfer_mxfp8_decode_graph_capture,
                )

                # The automatic MiniMax-M3 MXFP8 selection is decode-graph-only.
                # Tune every generation graph shape during the warmup-only pass.
                # Keep piecewise context/prefill capture on the native backend.
                flashinfer_methods = [
                    quant_method
                    for module in self.model.modules()
                    if isinstance(
                        (quant_method := getattr(module, "quant_method", None)), MXFP8LinearMethod
                    )
                    and quant_method.needs_flashinfer_autotune
                ]
                flashinfer_autotune_context = (
                    flashinfer_mxfp8_autotune()
                    if is_warmup_only and flashinfer_methods
                    else contextlib.nullcontext()
                )
                with flashinfer_autotune_context, flashinfer_mxfp8_decode_graph_capture():
                    self._capture_generation_cuda_graphs(resource_manager)
                self._capture_additional_cuda_graphs(resource_manager)
        # Piecewise graphs have separate capture machinery and do not use the
        # whole-model attention workspace. Capture them only on the second pass.
        if not is_warmup_only:
            self._capture_prefill_cuda_graphs(resource_manager)

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

    def _capture_generation_cuda_graphs(self, resource_manager: ResourceManager):
        """Warm up or capture pure-generation CUDA graph shapes."""
        if not self.cuda_graph_runner.enabled:
            return

        operation = "warmup" if self.cuda_graph_runner.is_warmup_only else "capture"
        logger.info(
            f"Running CUDA graph {operation} for {len(self._config.cuda_graph_batch_sizes)} batch sizes."
        )

        # Reverse order so smaller graphs can reuse memory from larger ones
        cuda_graph_batch_sizes = sorted(self._config.cuda_graph_batch_sizes, reverse=True)

        # Determine which graph shapes to process.
        graphs_to_capture = self._get_graphs_to_capture(cuda_graph_batch_sizes)
        graphs_to_capture = sorted(graphs_to_capture, reverse=True)
        # Create CUDA graphs for short and long sequences separately for sparse attention.
        # self.max_seq_len is the global max sequence length. For Helix CP each
        # rank only holds max_seq_len / cp_size tokens, so scale accordingly to
        # avoid creating warmup requests whose position_ids exceed the RoPE
        # table (max_position_embeddings).
        effective_max_seq_len = self._config.max_seq_len
        if self.mapping is not None and self.mapping.has_cp_helix():
            effective_max_seq_len = self._config.max_seq_len // self.mapping.cp_size

        sparse_config = self.sparse_attention_config
        if (
            isinstance(sparse_config, SeqLenAwareSparseAttentionConfig)
            and sparse_config.needs_separate_short_long_cuda_graphs()
        ):
            # For short sequences, subtract the maximum runtime tokens consumed
            # by a generation step so all current-step tokens stay within the
            # sequence length threshold. PARD uses 2K tokens here, not K+1.
            max_runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(
                self._config.max_draft_len
            )
            # For long sequences, use the default maximum sequence length.
            max_seq_len = sparse_config.seq_len_threshold - max_runtime_tokens_per_gen_step
            if max_seq_len < effective_max_seq_len:
                max_seq_len_list = [effective_max_seq_len, max_seq_len]
            else:
                max_seq_len_list = [effective_max_seq_len]
        else:
            max_seq_len_list = [effective_max_seq_len]

        def _run_capture_pass(
            force_non_greedy: bool,
            label: str,
            force_lora_graph: bool,
            sample_type: SampleType | None = None,
        ) -> None:
            assert self._force_lora_graph_for_capture is None
            self._force_lora_graph_for_capture = force_lora_graph
            # Pin the sampling tier for this pass. maybe_get_cuda_graph reads
            # it to build the graph key, so every graph captured below records
            # the kernels of this tier and nothing else. Passes that do not name
            # a tier capture FULL graphs -- ones carrying no sampling at all --
            # which is what a batch resolving to FULL replays.
            pinned_tier = sample_type or SampleType.FULL
            self.cuda_graph_runner.set_capture_sample_type(pinned_tier)
            try:
                for bs, draft_len in graphs_to_capture:
                    if bs > self._config.max_batch_size:
                        continue

                    for max_seq_len in max_seq_len_list:
                        warmup_request = self._create_cuda_graph_warmup_request(
                            resource_manager,
                            bs,
                            draft_len,
                            max_seq_len,
                            force_non_greedy=force_non_greedy,
                        )
                        with self._release_batch_context(warmup_request, resource_manager) as batch:
                            if batch is None:
                                # No KV cache space for this batch size. During KV
                                # cache estimation this makes the profiling peak
                                # unrepresentative (the final executor still
                                # captures this graph), so don't skip silently.
                                logger.warning(
                                    f"Skipping CUDA graph warmup ({label}) for "
                                    f"batch size={bs}, draft_len={draft_len}: "
                                    f"not enough KV cache space."
                                )
                                continue
                            logger.info(
                                f"Run generation-only CUDA graph {operation} ({label}) "
                                f"for batch size={bs}, draft_len={draft_len}, "
                                f"max_seq_len={max_seq_len}"
                            )
                            enable_spec_decode = draft_len > 0 or (
                                self._config.spec_config is not None
                                and self._config.spec_config.spec_dec_mode.use_one_engine()
                            )
                            self._prepare_capture_batch(
                                batch,
                                resource_manager,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=draft_len,
                            )
                            self._forward_warmup(
                                batch,
                                resource_manager,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=draft_len,
                            )
                            torch.cuda.synchronize()
            finally:
                self._force_lora_graph_for_capture = None
                self.cuda_graph_runner.set_capture_sample_type(None)

        if self._lora.cuda_graph_manager is None:
            lora_graph_cases = [False]
        elif self._config.cuda_graph_specialize_lora:
            # Capture the larger LoRA graph first so the base-only graph can
            # reuse its CUDA graph memory-pool allocations.
            lora_graph_cases = [True, False]
        else:
            lora_graph_cases = [True]

        # Which variants to capture depends on which sampler this engine will
        # actually run, and the two are mutually exclusive: a spec-decoding mode
        # in a one-engine mode samples inside its worker and gets a dedicated
        # sampler from get_spec_decoder, so TorchSampler -- the only sampler
        # implementing in-graph sampling -- never runs. Capturing the other
        # branch's variants would spend warmup on graphs whose keys no batch can
        # ever produce.
        #
        # The predicate is use_one_engine() rather than has_spec_decoder(): the
        # two-model eagle3 / mtp_eagle modes also "have a spec decoder", but
        # get_spec_decoder hands them a plain TorchSampler, so they belong on
        # the TorchSampler branch and do need the FAST graphs.
        uses_own_spec_decoder = (
            self._config.spec_config is not None
            and self._config.spec_config.spec_dec_mode.use_one_engine()
        )

        def _capture_variant(
            label: str, force_non_greedy: bool = False, sample_type: SampleType | None = None
        ) -> None:
            for use_lora_graph in lora_graph_cases:
                variant_label = label
                if self._lora.cuda_graph_manager is not None:
                    variant_label += ", LoRA" if use_lora_graph else ", base-only"
                _run_capture_pass(
                    force_non_greedy=force_non_greedy,
                    label=variant_label,
                    force_lora_graph=use_lora_graph,
                    sample_type=sample_type,
                )

        if uses_own_spec_decoder:
            # One-engine spec branch: the greedy argmax fast path plus the
            # advanced-sampling variant. The latter is needed because on-the-fly
            # capture is disabled outside warmup, so a batch containing a
            # non-greedy request would otherwise fall back to eager.
            #
            # Dummy warmup requests carry no sampling params, so the greedy pass
            # needs no override while the advanced one has to force the flag.
            _capture_variant("greedy")
            _capture_variant("advanced sampling", force_non_greedy=True)
        else:
            # TorchSampler branch: FULL carries no sampling in the graph and is
            # what every batch replays unless in-graph sampling is enabled, so it
            # is always captured. FAST adds the in-graph sampling kernels and is
            # captured only when opted in, since it costs extra warmup time and
            # memory.
            _capture_variant("full", sample_type=SampleType.FULL)
            if self._config.enable_in_graph_sampling:
                # Give the warmup requests real non-greedy sampling params:
                # dummies otherwise carry none, resolve to greedy, and the
                # capture would record argmax rather than the fast tier's
                # top-k/top-p kernels. Same substitution the advanced-sampling
                # pass relies on.
                _capture_variant("fast", force_non_greedy=True, sample_type=SampleType.FAST)

        # update_is_all_greedy_sample inside each forward call during the
        # non-greedy capture pass leaves is_all_greedy_sample=False on
        # spec_metadata. Reset it so the first real iteration starts clean;
        # update_is_all_greedy_sample will refresh it on every iteration anyway.
        # This is a defensive guard.
        if self.spec_metadata is not None:
            self.spec_metadata.is_all_greedy_sample = True

    def _capture_prefill_cuda_graphs(self, resource_manager: ResourceManager):
        """Warm up and capture prefill CUDA graphs, timing each phase separately.

        After capture, run prefill batches with many requests to warm up
        logits buffers outside the captured model body. This post-capture
        warmup runs for both piecewise and breakable prefill CUDA graphs.
        """
        if self._config.prefill_cuda_graph_backend == PrefillCudaGraphBackend.DISABLED or (
            self._config.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE
            and not self._config.torch_compile_enabled
        ):
            return

        logger.info("Running prefill CUDA graph warmup...")
        prefill_cuda_graph_num_tokens = sorted(
            self._config.prefill_cuda_graph_num_tokens, reverse=True
        )

        capture_context = (
            capture_piecewise_cuda_graph(True)
            if self._config.torch_compile_piecewise_cuda_graph
            else contextlib.nullcontext()
        )
        enable_spec_decode = self._config.is_spec_decode
        runtime_draft_len = get_static_draft_len(self._config.spec_config)
        with capture_context, self.no_cuda_graph():
            for num_tokens in prefill_cuda_graph_num_tokens:
                warmup_request = self._create_warmup_request(resource_manager, num_tokens, 0)
                with self._release_batch_context(warmup_request, resource_manager) as batch:
                    self._assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
                    if batch is None:
                        continue

                    logger.info(f"Run prefill CUDA graph capture for num tokens={num_tokens}")
                    if self.breakable_cuda_graph_runner is not None:
                        runner = self.breakable_cuda_graph_runner
                        try:
                            runner.capture(
                                num_tokens,
                                lambda: self._forward_warmup(
                                    batch,
                                    resource_manager,
                                    enable_spec_decode=enable_spec_decode,
                                    runtime_draft_len=runtime_draft_len,
                                ),
                            )
                        finally:
                            self._metrics["ctx_cuda_graph_warmup_seconds"] += runner.metrics.get(
                                BreakableCUDAGraphRunner.CUDA_GRAPH_WARMUP_METRIC, 0.0
                            )
                            self._metrics["ctx_cuda_graph_capture_seconds"] += runner.metrics.get(
                                BreakableCUDAGraphRunner.CUDA_GRAPH_CAPTURE_METRIC, 0.0
                            )
                    else:
                        with timing_metric("ctx_cuda_graph_warmup_seconds", self._metrics):
                            for _ in range(PiecewiseRunner.WARMUP_STEPS):
                                self._forward_warmup(
                                    batch,
                                    resource_manager,
                                    enable_spec_decode=enable_spec_decode,
                                    runtime_draft_len=runtime_draft_len,
                                )
                            torch.cuda.synchronize()
                        with timing_metric("ctx_cuda_graph_capture_seconds", self._metrics):
                            self._forward_warmup(
                                batch,
                                resource_manager,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=runtime_draft_len,
                            )
                            torch.cuda.synchronize()

        # The logits allocations grow with the number of requests and are not
        # part of the captured model body. Warm up the largest request count so
        # those allocations can be reused during stable inference.
        with timing_metric("post_ctx_cuda_graph_capture_warmup_seconds", self._metrics):
            for num_tokens in prefill_cuda_graph_num_tokens:
                warmup_request = self._create_warmup_request(
                    resource_manager, num_tokens, 0, least_requests=False
                )
                with self._release_batch_context(warmup_request, resource_manager) as batch:
                    self._assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
                    if batch is None:
                        continue
                    logger.info(
                        f"Run prefill CUDA graph warmup for num tokens={num_tokens} with most requests"
                    )
                    if self.breakable_cuda_graph_runner is not None:
                        with self.no_cuda_graph():
                            self.breakable_cuda_graph_runner.warmup(
                                lambda: self._forward_warmup(
                                    batch,
                                    resource_manager,
                                    enable_spec_decode=enable_spec_decode,
                                    runtime_draft_len=runtime_draft_len,
                                ),
                                steps=1,
                            )
                    else:
                        self._forward_warmup(
                            batch,
                            resource_manager,
                            enable_spec_decode=enable_spec_decode,
                            runtime_draft_len=runtime_draft_len,
                        )
                    torch.cuda.synchronize()

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
                for req in batch.all_requests():
                    kv_cache_manager.free_resources(req)
                    if draft_kv_cache_manager is not None:
                        draft_kv_cache_manager.free_resources(req)
                    if spec_resource_manager is not None:
                        spec_resource_manager.free_resources(req)

    @staticmethod
    def _apply_chunk_alignment(
        ctx_token_nums: list[int], alignment: int, aligned: bool
    ) -> list[int] | None:
        """Rewrite warmup context lengths onto one side of ``alignment``.

        Returns lengths that are all multiples of ``alignment`` (``aligned``) or
        that include at least one non-multiple, or None if that would leave the
        batch empty. Only shrinks or drops sequences.
        """
        if aligned:
            floored = [n - n % alignment for n in ctx_token_nums]
            kept = [n for n in floored if n > 0]
            return kept or None
        if any(n % alignment != 0 for n in ctx_token_nums):
            return list(ctx_token_nums)
        # Every length is a multiple; one token off the last breaks it.
        if ctx_token_nums[-1] <= 1:
            return None
        return ctx_token_nums[:-1] + [ctx_token_nums[-1] - 1]

    def _create_warmup_request(
        self,
        resource_manager: ResourceManager,
        num_tokens: int,
        num_gen_requests: int,
        least_requests: bool = True,
        chunk_alignment: int | None = None,
        chunk_aligned: bool = False,
    ) -> ScheduledRequests | None:
        """Creates a generic dummy ScheduledRequests object for warmup.

        ``chunk_alignment``, when set, rewrites the context lengths onto one
        side of it, so a kernel specializing on chunk alignment can be primed
        for both variants instead of whichever one the split happens to hit.
        Only shrinks or drops sequences, so the block estimate below stays a
        safe over-estimate.
        """
        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(resource_manager)

        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )

        available_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=num_tokens,
            max_num_draft_tokens=self._config.max_total_draft_tokens,
        )
        available_blocks = kv_cache_manager.get_num_free_blocks()
        if num_tokens > self._config.max_num_tokens or num_tokens > available_tokens:
            return None

        if num_gen_requests > self._config.max_batch_size:
            return None
        num_gen_tokens = num_gen_requests * (1 + self._config.max_total_draft_tokens)
        if num_gen_tokens > self._config.max_num_tokens:
            return None

        num_ctx_tokens = num_tokens - num_gen_tokens
        num_ctx_requests = 0
        ctx_requests = []
        gen_requests = []

        # Leave room for at least one decode token per request.
        max_seq_len = self._config.max_seq_len - 1
        if max_seq_len < 1:
            return None
        num_full_seqs = 0
        num_left_over_tokens = 0

        max_context_requests = self._config.max_batch_size - num_gen_requests
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
            num_ctx_requests = num_full_seqs + (1 if num_left_over_tokens > 0 else 0)

        if num_ctx_requests + num_gen_requests > self._config.max_batch_size:
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

        extra_ctx_tokens = getattr(kv_cache_manager, "num_extra_kv_tokens", 0) or 0
        extra_gen_tokens = extra_ctx_tokens + self._config.max_draft_loop_tokens
        blocks_to_use = num_full_seqs * blocks_for_seq(max_seq_len + extra_ctx_tokens)
        if num_left_over_tokens > 0:
            blocks_to_use += blocks_for_seq(num_left_over_tokens + extra_ctx_tokens)
        blocks_to_use += (
            num_gen_requests * self._config.max_beam_width * blocks_for_seq(1 + extra_gen_tokens)
        )

        if blocks_to_use > available_blocks and isinstance(kv_cache_manager, KVCacheManager):
            return None

        if num_ctx_tokens > 0:
            ctx_token_nums = [max_seq_len] * num_full_seqs
            if num_left_over_tokens > 0:
                ctx_token_nums.append(num_left_over_tokens)

            if chunk_alignment:
                adjusted = self._apply_chunk_alignment(
                    ctx_token_nums, chunk_alignment, chunk_aligned
                )
                if adjusted is None:
                    logger.debug(
                        f"Warmup batch of {ctx_token_nums} cannot be made "
                        f"{'aligned' if chunk_aligned else 'ragged'} modulo "
                        f"{chunk_alignment}; that variant will compile on the "
                        f"first request needing it."
                    )
                else:
                    ctx_token_nums = adjusted
                    num_ctx_requests = len(ctx_token_nums)

            ctx_requests = kv_cache_manager.add_dummy_requests(
                list(range(num_ctx_requests)),
                token_nums=ctx_token_nums,
                is_gen=False,
                max_num_draft_tokens=self._config.max_total_draft_tokens,
                kv_reserve_draft_tokens=self._config.max_draft_loop_tokens,
                use_mrope=self._config.use_mrope,
                draft_kv_cache_manager=draft_kv_cache_manager,
            )

            if ctx_requests is None:
                return None

            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(request_ids=list(range(num_ctx_requests)))

        if num_gen_requests > 0:
            gen_requests = kv_cache_manager.add_dummy_requests(
                list(range(num_ctx_requests, num_ctx_requests + num_gen_requests)),
                token_nums=[1] * num_gen_requests,
                is_gen=True,
                max_num_draft_tokens=self._config.max_total_draft_tokens,
                kv_reserve_draft_tokens=self._config.max_draft_loop_tokens,
                use_mrope=self._config.use_mrope,
                max_beam_width=self._config.max_beam_width,
                draft_kv_cache_manager=draft_kv_cache_manager,
            )

            if gen_requests is None:
                for r in ctx_requests:
                    kv_cache_manager.free_resources(r)
                    if draft_kv_cache_manager is not None:
                        draft_kv_cache_manager.free_resources(r)
                return None

            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests(
                    request_ids=list(range(num_ctx_requests, num_ctx_requests + num_gen_requests))
                )

        result = ScheduledRequests()
        result.reset_context_requests(ctx_requests)
        result.generation_requests = gen_requests
        static_draft_len = get_static_draft_len(self._config.spec_config)
        resolve_draft_len(
            self._config.spec_config,
            result,
            max_draft_len=self._config.max_draft_len,
            static_draft_len=static_draft_len,
            draft_len=static_draft_len,
        )
        return result

    def _create_cuda_graph_warmup_request(
        self,
        resource_manager: ResourceManager,
        batch_size: int,
        draft_len: int,
        max_seq_len: int = None,
        force_non_greedy: bool = False,
    ) -> ScheduledRequests | None:
        """Creates a dummy ScheduledRequests tailored for CUDA graph capture."""
        capture_sampling_params = NON_GREEDY_CAPTURE_SAMPLING_PARAMS if force_non_greedy else None
        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(resource_manager)

        available_blocks = kv_cache_manager.get_num_free_blocks() // self._config.max_beam_width
        if available_blocks < batch_size:
            return None

        result = ScheduledRequests()
        runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1

        # Add (batch_size - 1) dummy requests with the minimal sequence length.
        num_fillers = batch_size - 1
        requests = kv_cache_manager.add_dummy_requests(
            list(range(num_fillers)),
            token_nums=(
                None
                if self._dummy_request_tokens is None
                else [self._dummy_request_tokens] * num_fillers
            ),
            is_gen=True,
            max_num_draft_tokens=runtime_draft_token_buffer_width,
            kv_reserve_draft_tokens=self._config.max_draft_loop_tokens,
            use_mrope=self._config.use_mrope,
            max_beam_width=self._config.max_beam_width,
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
            **self._dummy_request_kwargs(resource_manager, num_fillers),
        )
        if requests is None:
            return None

        max_seq_len_request = self._add_longest_dummy_request(
            resource_manager,
            requests,
            batch_size,
            max_seq_len,
            runtime_draft_token_buffer_width,
            capture_sampling_params,
        )
        if max_seq_len_request is None:
            return None

        # Insert the longest request first to simulate padding for the CUDA
        # graph.
        requests.insert(0, max_seq_len_request)
        result.generation_requests = requests
        return self._finish_cuda_graph_warmup_request(
            result, resource_manager, batch_size, draft_len
        )

    def _add_longest_dummy_request(
        self,
        resource_manager: ResourceManager,
        requests: list[LlmRequest],
        batch_size: int,
        max_seq_len: int | None,
        runtime_draft_token_buffer_width: int,
        capture_sampling_params: SamplingParams | None,
    ) -> LlmRequest | None:
        """Add the dummy request with the longest sequence; free ``requests`` if it does not fit."""
        kv_cache_manager = resource_manager.get_resource_manager(self.kv_cache_manager_key)
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(resource_manager)

        def free_warmup_requests() -> None:
            for r in requests:
                kv_cache_manager.free_resources(r)
                if draft_kv_cache_manager is not None:
                    draft_kv_cache_manager.free_resources(r)

        # Add one dummy request with the maximum possible sequence length.
        max_seq_len = min(
            self._config.max_seq_len if max_seq_len is None else max_seq_len,
            kv_cache_manager.max_seq_len,
        )

        # Use max_draft_loop_tokens for capacity estimation to account
        # for the actual KV reservation per request.
        _kv_draft = self._config.max_draft_loop_tokens
        available_tokens = kv_cache_manager.get_num_available_tokens(
            token_num_upper_bound=max_seq_len, batch_size=batch_size, max_num_draft_tokens=_kv_draft
        )

        # Also consider draft KV cache capacity when it exists
        if draft_kv_cache_manager is not None:
            draft_available_tokens = draft_kv_cache_manager.get_num_available_tokens(
                batch_size=batch_size,
                token_num_upper_bound=max_seq_len,
                max_num_draft_tokens=_kv_draft,
            )
            available_tokens = min(available_tokens, draft_available_tokens)

        token_num = max(
            self._dummy_request_tokens or 1,
            min(
                available_tokens,
                max_seq_len - 1 - get_num_extra_kv_tokens(self._config.spec_config) - _kv_draft,
            ),
        )
        max_position_embeddings = self._dummy_position_limit()
        if max_position_embeddings is not None:
            token_num = min(token_num, max_position_embeddings - _kv_draft)

        token_num = int(token_num)  # Ensure int for range() in add_dummy_requests

        max_seq_len_request = kv_cache_manager.add_dummy_requests(
            request_ids=[batch_size - 1],
            token_nums=[token_num],
            is_gen=True,
            max_num_draft_tokens=runtime_draft_token_buffer_width,
            kv_reserve_draft_tokens=self._config.max_draft_loop_tokens,
            use_mrope=self._config.use_mrope,
            max_beam_width=self._config.max_beam_width,
            draft_kv_cache_manager=draft_kv_cache_manager,
            capture_sampling_params=capture_sampling_params,
            **self._dummy_request_kwargs(resource_manager, 1),
        )

        if max_seq_len_request is None:
            free_warmup_requests()
            return None
        return max_seq_len_request[0]

    def _finish_cuda_graph_warmup_request(
        self,
        result: ScheduledRequests,
        resource_manager: ResourceManager,
        batch_size: int,
        draft_len: int,
    ) -> ScheduledRequests | None:
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        if spec_resource_manager is not None:
            spec_resource_manager.add_dummy_requests(request_ids=list(range(batch_size)))
        if not self._add_dummy_request_resources(result.all_requests(), resource_manager):
            return None
        resolve_draft_len(
            self._config.spec_config,
            result,
            max_draft_len=self._config.max_draft_len,
            static_draft_len=get_static_draft_len(self._config.spec_config),
            draft_len=draft_len,
        )
        return result

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

    def _set_up_attn_metadata(
        self,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        draft_kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None = None,
    ):
        if self.attn_metadata is not None:
            # This assertion can be relaxed if needed: just create a new metadata
            # object if it changes.
            assert self.attn_metadata.kv_cache_manager is kv_cache_manager
            return self.attn_metadata

        config = self.model.model_config.pretrained_config
        self.attn_metadata = build_attention_metadata(
            self.model.model_config,
            max_batch_size=self._config.max_batch_size,
            max_num_tokens=self._config.max_num_tokens,
            max_beam_width=self._config.max_beam_width,
            attention_backend=self._config.attention_backend,
            attention_runtime_features=self._config.attention_runtime_features,
            mapping=self.mapping,
            cache_indirection=self.cache_indirection_attention
            if self._config.attention_backend.Metadata is TrtllmAttentionMetadata
            else None,
            kv_cache_manager=kv_cache_manager,
            draft_kv_cache_manager=draft_kv_cache_manager,
        )
        if isinstance(kv_cache_manager, BaseMambaCacheManager):
            self.attn_metadata.mamba_chunk_size = getattr(
                config, "chunk_size", self.attn_metadata.mamba_chunk_size
            )
        self.attn_metadata.mamba_metadata_cls = resolve_mamba_metadata_cls(self.model)

        return self.attn_metadata

    def _set_up_spec_metadata(self, spec_resource_manager: BaseResourceManager | None):
        if self.spec_metadata is not None:
            return self.spec_metadata
        self.spec_metadata = create_spec_metadata(
            self._config, self.model.config, spec_resource_manager
        )
        return self.spec_metadata

    def _release_decoder_graphs(self) -> None:
        if hasattr(self, "cuda_graph_runner") and self.cuda_graph_runner is not None:
            self.cuda_graph_runner.clear()
        if (
            hasattr(self, "breakable_cuda_graph_runner")
            and self.breakable_cuda_graph_runner is not None
        ):
            self.breakable_cuda_graph_runner.clear()

    def _should_use_full_generation_page_table(
        self, spec_config: DecodingBaseConfig | None, attn_metadata: AttentionMetadata
    ) -> bool:
        """Return whether overlap decode needs every reserved generation page."""
        return uses_full_generation_page_table(
            self._config.disable_overlap_scheduler, spec_config, attn_metadata
        )

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
                        self.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )
                else:
                    inputs["position_ids"][0, num_ctx_tokens:] += self.previous_pos_id_offsets_cuda[
                        :previous_batch_tokens
                    ]

                if hasattr(inputs["attn_metadata"], "kv_lens_cuda"):
                    if (
                        num_ctx_requests >= num_chunked_ctx_requests
                        and num_chunked_ctx_requests > 0
                    ):
                        # The generation requests with draft_tokens are treated as chunked context
                        # requests when extend_ctx returns True.
                        inputs["attn_metadata"].kv_lens_cuda[
                            num_ctx_requests - num_chunked_ctx_requests : num_ctx_requests
                        ] += self.previous_kv_lens_offsets_cuda[:num_chunked_ctx_requests]
                    else:
                        inputs["attn_metadata"].kv_lens_cuda[num_ctx_requests:num_seqs] += (
                            self.previous_kv_lens_offsets_cuda[:num_gen_requests]
                        )
                    inputs["attn_metadata"].on_update_kv_lens()
                # TRTLLM uses `kv_lens_cuda` above; FlashInfer exposes this backend-specific
                # correction without coupling the engine to its metadata type.
                elif hasattr(inputs["attn_metadata"], "apply_spec_decode_kv_lens_offsets"):
                    inputs["attn_metadata"].apply_spec_decode_kv_lens_offsets(
                        self.previous_kv_lens_offsets_cuda,
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
                        self.previous_pos_id_offsets_cuda[:helix_gen_tokens]
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
                        self.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )
                else:
                    inputs["position_ids"][0, num_ctx_tokens:] -= self.previous_pos_id_offsets_cuda[
                        :previous_batch_tokens
                    ]

                # Only TrtllmAttentionMetadata has kv_lens_cuda.
                if isinstance(inputs["attn_metadata"], TrtllmAttentionMetadata):
                    if (
                        num_ctx_requests >= num_chunked_ctx_requests
                        and num_chunked_ctx_requests > 0
                    ):
                        inputs["attn_metadata"].kv_lens_cuda[
                            num_ctx_requests - num_chunked_ctx_requests : num_ctx_requests
                        ] -= self.previous_kv_lens_offsets_cuda[:num_chunked_ctx_requests]
                    else:
                        inputs["attn_metadata"].kv_lens_cuda[num_ctx_requests:num_seqs] -= (
                            self.previous_kv_lens_offsets_cuda[:num_gen_requests]
                        )
                # Restore the FlashInfer-specific logical KV lengths through the same optional hook
                # used by `_preprocess_inputs`.
                elif hasattr(inputs["attn_metadata"], "apply_spec_decode_kv_lens_offsets"):
                    inputs["attn_metadata"].apply_spec_decode_kv_lens_offsets(
                        self.previous_kv_lens_offsets_cuda,
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
                        self.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )

    def _get_all_rank_num_tokens_and_spec_counts(
        self, attn_metadata: AttentionMetadata, spec_metadata: SpecMetadata
    ) -> tuple[list[int] | None, list[list[int]] | None]:
        """Exchange the attention and speculative per-rank counts in a single
        collective instead of one collective each.

        The spec token count is derived via ``SpecMetadata.dp_num_tokens``
        because this collective runs *before* ``spec_metadata.prepare()``
        rewrites ``num_tokens`` into the same shape (the attention count it
        shares the collective with is consumed earlier, by padding selection).
        Both read the same fields, so callers must have set
        ``spec_metadata.num_tokens``, ``num_generations`` and ``seq_lens`` for
        this batch before calling this. ``num_generations`` counts every
        generation request, CUDA-graph dummies included.
        """
        if not self._config.enable_attention_dp:
            return None, None
        spec_counts = (
            spec_metadata.dp_num_tokens(),
            len(spec_metadata.seq_lens),
            spec_metadata.num_generations,
        )
        if self.mapping.cp_size > 1 and not self.mapping.has_cp_helix():
            # attn counts span TP only while spec counts span TP*CP; keep the
            # two exchanges separate.
            gathered = self.dist.tp_cp_allgather_int64(list(spec_counts))
            return (
                get_all_rank_num_tokens(
                    attn_metadata,
                    enable_attention_dp=self._config.enable_attention_dp,
                    mapping=self.mapping,
                    dist=self.dist,
                ),
                gathered.T.tolist(),
            )
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

    def _record_cached_kv_tokens_per_req(
        self,
        num_cached_tokens_per_seq: Sequence[int],
        request_groups: Sequence[tuple[Sequence[LlmRequest], int]] = (),
    ) -> None:
        """Log post-prepare (backend-adjusted) counts in packed order, retaining
        ADP dummies to match num_scheduled_requests at beam width one and
        separating CUDA graph padding.
        """
        counts = []
        if request_groups:
            offset = 0
            for requests, rows_per_request in request_groups:
                for request in requests:
                    end = offset + rows_per_request
                    if not request.is_cuda_graph_dummy:
                        counts.extend(num_cached_tokens_per_seq[offset:end])
                    offset = end
            assert offset == len(num_cached_tokens_per_seq)
        else:
            counts = list(num_cached_tokens_per_seq)
        self.iter_states["cached_kv_tokens_per_req"] = counts
        self.iter_states["cached_kv_tokens_cuda_graph_padding"] = sum(
            num_cached_tokens_per_seq
        ) - sum(counts)

    def _can_use_steady_gen_fast_prepare(
        self,
        scheduled_requests: ScheduledRequests,
        new_tokens_device: torch.Tensor | None,
        next_draft_tokens_device: torch.Tensor | None,
        spec_metadata: SpecMetadata | None,
        is_dummy: bool,
    ) -> bool:
        """Check whether the cached steady-state generation prepare applies.

        The cache is only recorded by a full _prepare_tp_inputs pass whose
        batch consisted purely of non-dummy generation requests that all had
        a previous overlap-scheduler tensor (see the recording site), so the
        per-step check only needs to confirm the dynamic conditions: still a
        generation-only batch with the exact same requests in the same order.
        """
        cache = self._steady_gen_cache
        if cache is None or is_dummy:
            return False
        if (
            new_tokens_device is None
            or next_draft_tokens_device is not None
            or spec_metadata is not None
        ):
            return False
        if scheduled_requests.num_context_requests > 0:
            return False
        generation_requests = scheduled_requests.generation_requests
        if len(generation_requests) != cache["num_requests"]:
            return False
        return cache["request_ids"] == [request.py_request_id for request in generation_requests]

    @nvtx_range("_apply_steady_gen_fast_prepare")
    def _apply_steady_gen_fast_prepare(
        self,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        attn_metadata: AttentionMetadata,
        new_tensors_device: SampleStateTensors,
        resource_manager: ResourceManager | None,
    ):
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
        num_requests = cache["num_requests"]

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
        use_mrope = cache["use_mrope"]
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
        torch.index_select(
            new_tensors_device.new_tokens[0, :, : self._config.max_beam_width],
            0,
            previous_slots,
            out=self.input_ids_cuda[: num_requests * self._config.max_beam_width].view(
                num_requests, self._config.max_beam_width
            ),
        )

        if not attn_metadata.is_cuda_graph:
            attn_metadata.seq_lens = cache["seq_lens_ones"]
        attn_metadata.beam_width = 1
        attn_metadata.request_ids = cache["request_ids"]
        attn_metadata.prompt_lens = cache["prompt_lens"]
        attn_metadata.num_contexts = 0
        attn_metadata.num_chunked_ctx_requests = 0
        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            num_extra_kv_tokens=get_num_extra_kv_tokens(None),
        )
        attn_metadata.kv_cache_manager = kv_cache_manager
        if hasattr(self.model.model_config.pretrained_config, "chunk_size"):
            attn_metadata.mamba_chunk_size = self.model.model_config.pretrained_config.chunk_size
        with nvtx_range("steady_gen_metadata_prepare"):
            attn_metadata.prepare()

        attn_all_rank_num_tokens = get_all_rank_num_tokens(
            attn_metadata,
            enable_attention_dp=self._config.enable_attention_dp,
            mapping=self.mapping,
            dist=self.dist,
        )
        padded_num_tokens, can_run_piecewise_cuda_graph, attn_all_rank_num_tokens = (
            get_padding_params(
                num_requests,
                0,
                attn_all_rank_num_tokens,
                dist=self.dist,
                enable_attention_dp=self._config.enable_attention_dp,
                prefill_cuda_graph_backend=self._config.prefill_cuda_graph_backend,
                prefill_cuda_graph_num_tokens=self._config.prefill_cuda_graph_num_tokens,
            )
        )
        set_per_request_prefill_cuda_graph_flag(can_run_piecewise_cuda_graph)
        attn_metadata.padded_num_tokens = (
            padded_num_tokens if padded_num_tokens != num_requests else None
        )
        virtual_num_tokens = num_requests
        if attn_metadata.padded_num_tokens is not None:
            self.input_ids_cuda[num_requests:padded_num_tokens].fill_(0)
            # Zero-fill the padding tail of whichever position layout the
            # model consumes, matching the full pass.
            if use_mrope:
                self.mrope_position_ids_cuda[:, :, num_requests:padded_num_tokens].fill_(0)
            else:
                self.position_ids_cuda[num_requests:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens

        self.iter_states["num_ctx_requests"] = 0
        self.iter_states["num_ctx_tokens"] = 0
        self.iter_states["num_generation_tokens"] = num_requests
        self.iter_states["cached_kv_tokens"] = sum(num_cached_tokens_per_seq)
        if self._log_cached_kv_tokens_per_req:
            self._record_cached_kv_tokens_per_req(num_cached_tokens_per_seq)

        if use_mrope:
            final_position_ids = self.mrope_position_ids_cuda[:, :, :virtual_num_tokens]
        else:
            final_position_ids = self.position_ids_cuda[:virtual_num_tokens].unsqueeze(0)
        inputs = {
            "attn_metadata": attn_metadata,
            "input_ids": self.input_ids_cuda[:virtual_num_tokens],
            "position_ids": final_position_ids,
            "inputs_embeds": None,
            "multimodal_params": [],
            "resource_manager": resource_manager,
        }
        return inputs, None

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
    ) -> tuple[dict[str, Any], torch.Tensor | None] | None:
        """Return ``(inputs, gather_ids)`` when a specialized path prepares the batch."""
        return None

    def _new_extra_inputs(self) -> "ExtraInputsCollector":
        return _NO_EXTRA_INPUTS

    def _prepare_tp_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata | None = None,
        new_tensors_device: SampleStateTensors | None = None,
        cache_indirection_buffer: torch.Tensor | None = None,
        resource_manager: ResourceManager | None = None,
        maybe_graph: bool = False,
        promoted_context_request_ids: frozenset[int] = frozenset(),
        use_lora_graph: bool = False,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
        is_dummy: bool,
    ) -> tuple[dict[str, Any], torch.Tensor | None, int]:
        """
        Prepare inputs for Pytorch Model.
        """

        new_tokens_device, new_tokens_lens_device, next_draft_tokens_device = None, None, None
        if new_tensors_device is not None:
            # speculative decoding cases: [batch, 1 + draft_len], others: [batch]
            new_tokens_device = new_tensors_device.new_tokens
            # When using overlap scheduler with speculative decoding, the target model's inputs
            # would be SampleStateTensorsSpec.
            if isinstance(new_tensors_device, SampleStateTensorsSpec):
                assert enable_spec_decode
                new_tokens_lens_device = new_tensors_device.new_tokens_lens  # [batch]
                next_draft_tokens_device = (
                    new_tensors_device.next_draft_tokens
                )  # [batch, draft_len]

        # Must be before the update of py_batch_idx
        if self.guided_decoder is not None:
            self.guided_decoder.add_batch(
                scheduled_requests,
                new_tokens=new_tokens_device,
                runtime_draft_len=runtime_draft_len,
            )

        fast_inputs = self._prepare_inputs_fast_path(
            scheduled_requests,
            kv_cache_manager,
            attn_metadata,
            new_tokens_device,
            next_draft_tokens_device,
            resource_manager,
            promoted_context_request_ids=promoted_context_request_ids,
            enable_spec_decode=enable_spec_decode,
        )
        if fast_inputs is not None:
            inputs, gather_ids = fast_inputs
            return inputs, gather_ids, runtime_draft_len

        if not promoted_context_request_ids and self._can_use_steady_gen_fast_prepare(
            scheduled_requests, new_tokens_device, next_draft_tokens_device, spec_metadata, is_dummy
        ):
            inputs, gather_ids = self._apply_steady_gen_fast_prepare(
                kv_cache_manager, attn_metadata, new_tensors_device, resource_manager
            )
            return inputs, gather_ids, runtime_draft_len
        # Any full pass invalidates the steady-state cache; it is re-recorded
        # at the end of this pass when the batch qualifies.
        self._steady_gen_cache = None

        # Hoist self.use_mrope to a function-scope local so the per-request /
        # per-context-request mrope branches use LOAD_FAST instead of LOAD_ATTR.
        _use_mrope = self._config.use_mrope

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
        mrope_position_ids = []  # (start_idx, end_idx, (3,1,L) mrope_pos_ids) per multimodal request
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
        mrope_dummy_seq_slot = get_mrope_dummy_seq_slot(
            self._config.max_num_tokens, self.mapping.pp_size
        )
        num_accepted_draft_tokens = []  # per request
        extra_inputs = self._new_extra_inputs()

        context_prompt_lookahead = None
        if spec_metadata is not None and spec_metadata.context_prompt_lookahead_tokens is not None:
            context_prompt_lookahead = []

        for request in scheduled_requests.context_requests:
            request_ids.append(request.py_request_id)
            draft_lens.append(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            if context_prompt_lookahead is not None:
                context_prompt_lookahead.append(
                    _get_context_prompt_lookahead_token(request, end_compute)
                )
            # Fetch only the current chunk. get_tokens(0) marshals the whole
            # O(seq_len) VecTokens into a Python list of boxed ints; chunked
            # prefill re-enters this loop for every chunk of the same prompt, so
            # that is O(L) per chunk = O(L^2/chunk) over the prefill.
            # get_tokens_range copies only [begin, end) -> O(chunk).
            prompt_tokens = request.get_tokens_range(0, begin_compute, end_compute)
            position_ids.extend(range(begin_compute, begin_compute + len(prompt_tokens)))

            # Start offset of this request's (current-chunk) tokens within the
            # flattened input_ids. Recorded on multimodal_params below so models
            # that rewrite token IDs in place write into the request's own span
            # rather than assuming a contiguous multimodal prefix.
            context_start_idx = len(input_ids)
            input_ids.extend(prompt_tokens)

            gather_ids.append(len(input_ids) - 1)
            sequence_lengths.append(len(prompt_tokens))
            num_accepted_draft_tokens.append(len(prompt_tokens) - 1)
            prompt_lengths.append(len(prompt_tokens))
            past_seen_token_num = begin_compute
            num_cached_tokens_per_seq.append(past_seen_token_num - request.py_num_compressed_tokens)
            request.cached_tokens = past_seen_token_num
            extra_inputs.add_context_request(request)

            # Embed mask is required only for partial iterations (chunked
            # prefill or KV-cache reuse); full-prefill degrades gracefully.
            check_mm_embed_cumsum_if_needed(
                request.py_multimodal_data,
                begin_compute=past_seen_token_num,
                end_compute=end_compute,
                prompt_len=request.get_num_tokens(0),
            )
            mm_data = request.py_multimodal_data or {}
            cumsum = mm_data.get("multimodal_embed_mask_cumsum")
            py_multimodal_runtime = None
            if cumsum is not None:
                py_multimodal_runtime = MultimodalRuntimeData(
                    embed_mask_cumsum=cumsum,
                    past_seen_token_num=past_seen_token_num,
                    chunk_end_pos=end_compute,
                )

            multimodal_params = MultimodalParams(
                multimodal_input=_build_request_multimodal_input(
                    request, self._config.mm_encoder_cache_enabled
                ),
                multimodal_data=request.py_multimodal_data,
                multimodal_runtime=py_multimodal_runtime,
                mm_item_order=getattr(request, "py_mm_item_order", None),
                input_ids_start_offset=context_start_idx,
            )
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
                multimodal_params.to_device(
                    "multimodal_data",
                    "cuda",
                    pin_memory=prefer_pinned(),
                    target_keywords=getattr(self.model, "multimodal_data_device_paths", None),
                )
                if _use_mrope:
                    # A request may carry multimodal content but no MRoPE
                    # metadata (a text-only prompt whose input processor skips
                    # ``mrope_config``, or a model that does not consume it).
                    # Its per-axis positions are just the scalar positions,
                    # which the (3,1,N) seeding further below already
                    # broadcasts, so leave that span alone.
                    mrope_config = multimodal_params.multimodal_data.get("mrope_config") or {}
                    mrope_pos_ids = mrope_config.get("mrope_position_ids")
                    if mrope_pos_ids is not None:
                        ctx_mrope_position_ids = mrope_pos_ids[
                            :, :, begin_compute : begin_compute + len(prompt_tokens)
                        ]
                        # Record as (start_idx, end_idx, (3,1,L) mrope_pos_ids)
                        mrope_position_ids.append(
                            (
                                len(position_ids) - len(prompt_tokens),
                                len(position_ids),
                                ctx_mrope_position_ids,
                            )
                        )
                    mrope_position_delta = mrope_config.get("mrope_position_deltas")
                    if mrope_position_delta is not None:
                        request.py_mrope_position_delta = mrope_position_delta
                    if mrope_position_delta is not None and request.py_seq_slot is not None:
                        mrope_delta_write_seq_slots.append(request.py_seq_slot)
                        request.py_mrope_delta_cache_slot = request.py_seq_slot

                # re-assign the multimodal_data to the request after to_device for generation requests
                request.py_multimodal_data = multimodal_params.multimodal_data
                multimodal_params_list.append(multimodal_params)

                # Re-register mrope tensors for context-only requests (EPD disaggregated serving).
                # This creates new IPC handles owned by the prefill worker, so the decode worker
                # can access them even after the encode worker's GC deallocates the original memory.
                # Without this, the decode worker would receive handles pointing to freed memory.
                if (
                    request.is_context_only_request
                    and _use_mrope
                    and "mrope_config" in multimodal_params.multimodal_data
                ):
                    mrope_config = multimodal_params.multimodal_data["mrope_config"]
                    _mrope_position_ids = mrope_config.get("mrope_position_ids")
                    _mrope_position_deltas = mrope_config.get("mrope_position_deltas")
                    if _mrope_position_ids is not None and _mrope_position_deltas is not None:
                        # Clone to allocate new memory owned by this (prefill) worker.
                        request.py_result.set_mrope_position(
                            _mrope_position_ids.clone(), _mrope_position_deltas.clone()
                        )

            request.py_batch_idx = request.py_seq_slot

        num_ctx_requests = scheduled_requests.num_context_requests
        num_ctx_tokens = len(input_ids)
        if len(multimodal_params_list) > 0:
            # input_ids holds only context tokens here; extend/draft tokens are
            # appended below and are by construction text, so we reuse the
            # CPU-side text_token_indices and just extend it with the
            # post-context arange instead of recomputing via a bool mask +
            # torch.where over the full range.
            text_token_indices_ctx, mm_token_indices = prepare_multimodal_indices(
                input_ids, model=self.model
            )
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
            is_promoted_context = request.py_request_id in promoted_context_request_ids
            if is_promoted_context:
                # A promoted row is a one-token final context chunk riding
                # the decode path: this is its context-phase forward, so
                # latch the reused prefix as the context branch does.
                # Ordinary decode rows never write cached_tokens.
                request.cached_tokens = request.context_current_position
            else:
                all_gen_request_ids.append(request.py_request_id)
            # In speculative iterations, keep promoted rows ahead of existing
            # generation rows in the extend-request packing order. Although
            # their q_len is one, this category provides the "no previous
            # speculative tensor" branch needed to source their prompt token
            # without disturbing the overlap offsets of ordinary generation
            # siblings. Non-speculative promoted rows retain the established
            # ordinary generation path below.
            if is_promoted_context and enable_spec_decode:
                extend_requests.append(request)
            elif is_promoted_context:
                generation_requests.append(request)
            elif get_draft_token_length(request) > 0 or next_draft_tokens_device is not None:
                if request.is_dummy:
                    extend_dummy_requests.append(request)
                else:
                    extend_requests.append(request)
            elif request.py_is_first_draft:
                first_draft_requests.append(request)
            else:
                generation_requests.append(request)
        extend_requests += extend_dummy_requests

        # Helix bookkeeping is needed by BOTH the extend (speculative verify
        # group) and the plain generation packing loops below, so initialize
        # it ahead of them. Positions are global; KV ownership follows the
        # round-robin ledger (page b -> rank b % cp); the host-side
        # provisional packing values come from the one shared definition in
        # _torch.utils.helix_local_len.
        helix_is_inactive_rank, helix_position_offsets = [], []
        helix_owned_new_tokens = []
        _has_cp_helix = self.mapping.has_cp_helix()
        if _has_cp_helix and kv_cache_manager is not None:
            _helix_phys = kv_cache_manager.tokens_per_block
            _helix_cp_size = self.mapping.cp_size
            _helix_cp_rank = self.mapping.cp_rank

            def _helix_local_len_host(global_len: int) -> int:
                return helix_local_len(global_len, _helix_phys, _helix_cp_size, _helix_cp_rank)

            def _helix_pack_extend(request, group: int) -> int:
                # A helix gen worker's token list is the rank-LOCAL
                # round-robin subset, so max_beam_num_tokens is not a global
                # base; rebuild it from the global prompt length plus the
                # rank-invariant generated count. Also repacks position_ids,
                # which the caller filled from the local base.
                generated_len = request.max_beam_num_tokens - request.py_prompt_len
                base = request.total_input_len_cp + generated_len - 1
                helix_position_offsets.extend(range(base, base + group))
                position_ids[-group:] = range(base, base + group)
                helix_is_inactive_rank.append(False)
                return base

        spec_config = self._config.spec_config if enable_spec_decode else None
        if not self._config.disable_overlap_scheduler and spec_config is not None:
            assert spec_config.spec_dec_mode.support_overlap_scheduler(), (
                f"{spec_config.decoding_type} does not support overlap scheduler"
            )

        # For tree decoding, runtime_draft_len should match total tree
        # tokens (not tree depth).  py_executor resets it every iteration.
        if spec_config is not None and not spec_config.is_linear_tree:
            runtime_draft_len = get_static_draft_len(spec_config)

        # will contain previous batch indices of generation requests
        previous_batch_indices = []
        previous_pos_indices = []
        runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(runtime_draft_len)
        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
        for request in extend_requests:
            is_promoted_context = request.py_request_id in promoted_context_request_ids
            if getattr(request, "py_needs_onehot_draft_probs", False):
                if request.py_seq_slot is not None:
                    padding_gen_slots.append(request.py_seq_slot)
                request.py_needs_onehot_draft_probs = False  # consume once
            request_ids.append(request.py_request_id)
            # the request has no previous tensor:
            # (1) next_draft_tokens_device is None, which means overlap scheduler is disabled; or
            # (2) a dummy request; or
            # (3) the first step in the generation server of disaggregated serving
            if (
                is_promoted_context
                or next_draft_tokens_device is None
                or request.is_dummy
                or request.py_batch_idx is None
            ):
                # get token ids, including input token ids and draft token ids. For these dummy requests,
                # no need to copy the token ids.
                if not (request.is_attention_dp_dummy or request.is_cuda_graph_dummy):
                    if is_promoted_context:
                        input_ids.append(request.get_tokens(0)[request.context_current_position])
                    else:
                        input_ids.append(request.get_last_tokens(0))
                    input_ids.extend(request.py_draft_tokens)
                    draft_tokens.extend(request.py_draft_tokens)
                # get other ids and lengths
                num_draft_tokens = get_draft_token_length(request)
                past_seen_token_num = (
                    request.context_current_position
                    if is_promoted_context
                    else request.max_beam_num_tokens - 1
                )
                draft_lens.append(num_draft_tokens)
                if (
                    enable_spec_decode
                    and spec_config.spec_dec_mode.extend_ctx(self._config.attention_backend)
                    and spec_config.is_linear_tree
                ):
                    # We're treating the prompt lengths as context requests here, so
                    # the the prompt lens should not include the cached tokens.
                    prompt_lengths.append(1 + num_draft_tokens)
                else:
                    prompt_lengths.append(request.py_prompt_len)

                sequence_lengths.append(1 + num_draft_tokens)
                num_accepted_draft_tokens.append(num_draft_tokens)
                gather_ids.extend(
                    list(range(len(position_ids), len(position_ids) + 1 + num_draft_tokens))
                )
                position_ids.extend(
                    list(range(past_seen_token_num, past_seen_token_num + 1 + num_draft_tokens))
                )
                num_cached_tokens_per_seq.append(
                    past_seen_token_num - request.py_num_compressed_tokens
                )
                if _has_cp_helix:
                    # Verify group [base, base+group) in GLOBAL positions.
                    # On a helix gen worker the request's token list is the
                    # rank-LOCAL round-robin subset, so max_beam_num_tokens
                    # (= local_prompt + generated) must NOT be used as a
                    # global base; reconstruct it from the global prompt
                    # length plus the (rank-invariant) generated count. This
                    # branch has no in-flight predecessor, so every value is
                    # exact (no device correction needed).
                    group = 1 + num_draft_tokens
                    base = _helix_pack_extend(request, group)
                    local_cached = _helix_local_len_host(base)
                    helix_owned_new_tokens.append(
                        _helix_local_len_host(base + group) - local_cached
                    )
                    num_cached_tokens_per_seq[-1] = local_cached - request.py_num_compressed_tokens
                # update batch index
                request.py_batch_idx = request.py_seq_slot
            else:
                # update batch index
                previous_batch_idx = request.py_batch_idx
                request.py_batch_idx = request.py_seq_slot

                sequence_lengths.append(runtime_tokens_per_gen_step)
                num_accepted_draft_tokens.append(request.py_num_accepted_draft_tokens)
                past_seen_token_num = request.max_beam_num_tokens - 1

                draft_lens.append(runtime_draft_token_buffer_width)
                gather_ids.extend(
                    list(range(len(position_ids), len(position_ids) + runtime_tokens_per_gen_step))
                )
                position_ids.extend(
                    list(
                        range(
                            past_seen_token_num, past_seen_token_num + runtime_tokens_per_gen_step
                        )
                    )
                )
                # previous tensor
                previous_batch_indices.append(previous_batch_idx)
                previous_pos_indices.extend([previous_batch_idx] * runtime_tokens_per_gen_step)

                num_cached_tokens_per_seq.append(
                    past_seen_token_num
                    + runtime_tokens_per_gen_step
                    - request.py_num_compressed_tokens
                )
                if _has_cp_helix:
                    # In-flight predecessor: mirror the non-helix convention
                    # above -- positions are packed from the stale base (the
                    # overlap device correction adds the accepted count) and
                    # KV numbers assume full acceptance (the device recompute
                    # in recompute_helix_spec_buffers overrides them). The
                    # base is reconstructed GLOBALLY (see the no-previous
                    # branch: the token list is rank-local under helix).
                    group = runtime_tokens_per_gen_step
                    base = _helix_pack_extend(request, group)
                    local_full = _helix_local_len_host(base + group)
                    helix_owned_new_tokens.append(0)
                    num_cached_tokens_per_seq[-1] = local_full - request.py_num_compressed_tokens
                if (
                    enable_spec_decode
                    and spec_config.spec_dec_mode.extend_ctx(self._config.attention_backend)
                    and spec_config.is_linear_tree
                ):
                    prompt_lengths.append(runtime_tokens_per_gen_step)
                else:
                    prompt_lengths.append(request.py_prompt_len)

            extra_inputs.add_generation_request(request)

        for request in first_draft_requests:
            request_ids.append(request.py_request_id)
            draft_lens.append(0)
            # Only the length and the last (original_max_draft_len+1) tokens are
            # needed here; get_num_tokens is O(1) and get_tokens_range copies only
            # the requested window, whereas get_tokens(0) marshals the whole
            # O(seq_len) VecTokens into a Python list.
            _num_tokens = request.get_num_tokens(0)
            begin_compute = _num_tokens - self._config.original_max_draft_len - 1
            end_compute = begin_compute + self._config.original_max_draft_len + 1
            prompt_tokens = request.get_tokens_range(0, begin_compute, end_compute)
            position_ids.extend(range(begin_compute, begin_compute + len(prompt_tokens)))
            input_ids.extend(prompt_tokens)
            gather_ids.append(
                len(input_ids)
                - 1
                - (self._config.original_max_draft_len - request.py_num_accepted_draft_tokens)
            )
            num_accepted_draft_tokens.append(request.py_num_accepted_draft_tokens)

            sequence_lengths.append(1 + self._config.original_max_draft_len)
            prompt_lengths.append(request.py_prompt_len)
            past_seen_token_num = begin_compute
            num_cached_tokens_per_seq.append(past_seen_token_num - request.py_num_compressed_tokens)
            extra_inputs.add_generation_request(request)

            # update batch index
            request.py_batch_idx = request.py_seq_slot

        # Cache invariant method result to avoid repeated calls per-request
        _n_gen = len(generation_requests)
        # One-shot batch-level flag — True iff any generation request actually
        # carries multimodal payload. Lets the strip_mm_data branch below
        # short-circuit on a LOAD_FAST rather than a per-request LOAD_ATTR
        # of py_multimodal_data for non-multimodal models (the gpt-oss-120b
        # GEN case).
        _has_any_multimodal_request = any(
            r.py_multimodal_data is not None for r in generation_requests
        )
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
            real_requests = [req for req in generation_requests if not req.is_dummy]
            iter_widths = {req.get_beam_width_by_iter() for req in real_requests}
            if len(iter_widths) > 1:
                # NB: this aborts the whole batch, not just the offending
                # requests -- ModelEngine has no per-request failure channel,
                # and by this point the batch is already scheduled. Scoping the
                # failure needs the scheduler to group by beam width in the
                # first place, so that no such batch is formed; TRTLLM-14792.
                raise ValueError(
                    "Generation requests in one batch must all have the same "
                    f"beam width; got {sorted(iter_widths)}. Mixed beam widths "
                    "within a batch are not supported yet (TRTLLM-14792)."
                )

            # Pre-extend constant-value lists to avoid per-request append
            # overhead (saves ~3 append calls per request).
            draft_lens.extend([0] * (_n_gen * beam_width))
            sequence_lengths.extend([1] * (_n_gen * beam_width))
            num_accepted_draft_tokens.extend([0] * (_n_gen * beam_width))

            for request in generation_requests:
                request_ids.append(request.py_request_id)
                is_promoted_context = request.py_request_id in promoted_context_request_ids
                if is_promoted_context:
                    input_ids.append(request.get_tokens(0)[request.context_current_position])
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
                        helix_is_inactive_rank.append(request.py_helix_is_inactive_rank)
                        helix_position_offsets.append(position_id)
                        # Keep the per-seq owned-count list aligned when the
                        # spec path is active in the same batch. Whether the
                        # list arms the spec path at all is decided once, at
                        # the update_helix_param call below.
                        helix_owned_new_tokens.append(0 if request.py_helix_is_inactive_rank else 1)

                for beam in range(beam_width):
                    position_ids.append(position_id)
                    num_cached_tokens_per_seq.append(
                        past_seen_token_num - request.py_num_compressed_tokens
                    )
                    prompt_lengths.append(request.py_prompt_len)
                    gather_ids.append(len(position_ids) - 1)

                if _use_mrope:
                    mrope_position_delta = getattr(request, "py_mrope_position_delta", None)
                    if mrope_position_delta is None and request.py_multimodal_data:
                        mrope_config = request.py_multimodal_data.get("mrope_config") or {}
                        mrope_position_delta = mrope_config.get("mrope_position_deltas")
                        if mrope_position_delta is not None:
                            if mrope_position_delta.device.type == "cpu":
                                mrope_position_delta = maybe_pin_memory(mrope_position_delta).to(
                                    device="cuda", dtype=torch.int32, non_blocking=True
                                )
                                mrope_config["mrope_position_deltas"] = mrope_position_delta
                            request.py_mrope_position_delta = mrope_position_delta
                    if mrope_position_delta is not None:
                        has_gen_mrope_delta = True
                        # NOTE: Expanding position_ids to 3D tensor who is using mrope
                        gen_mrope_position_ids = (
                            past_seen_token_num + mrope_position_delta
                        ).expand(3, 1, 1)
                        update_mrope_delta = (
                            request.py_seq_slot is not None
                            and not request.is_dummy
                            and getattr(request, "py_mrope_delta_cache_slot", None)
                            != request.py_seq_slot
                        )
                        delta_read_seq_slot = (
                            mrope_dummy_seq_slot
                            if request.is_dummy or request.py_seq_slot is None
                            else request.py_seq_slot
                        )
                        if update_mrope_delta:
                            multimodal_params = MultimodalParams(
                                multimodal_data={
                                    "mrope_config": {"mrope_position_deltas": mrope_position_delta}
                                }
                            )
                            mrope_delta_write_seq_slots.append(request.py_seq_slot)
                            multimodal_params_list.append(multimodal_params)
                            request.py_mrope_delta_cache_slot = request.py_seq_slot
                        for beam in range(beam_width):
                            # Locate this beam's single token in the flat array.
                            token_start = len(position_ids) - beam_width + beam
                            mrope_position_ids.append(
                                (token_start, token_start + 1, gen_mrope_position_ids)
                            )
                            mrope_delta_read_seq_slots.append(delta_read_seq_slot)
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
                            mrope_delta_read_seq_slots.append(mrope_dummy_seq_slot)
                # Equivalent to the original `is_generation_admission and
                # request.py_multimodal_data`. The batch-level flag is checked
                # first so non-multimodal models pay one LOAD_FAST per request
                # instead of LOAD_ATTR(py_multimodal_data) + LOAD_ATTR(py_batch_idx).
                if (
                    _has_any_multimodal_request
                    and request.py_multimodal_data
                    and request.py_batch_idx is None
                ):
                    strip_mm_data_for_generation(request.py_multimodal_data)

                request.py_batch_idx = request.py_seq_slot
                extra_inputs.add_generation_request(request, repeat=beam_width)
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

        def previous_seq_slots_device():
            previous_batch_indices_host = torch.tensor(
                previous_batch_indices, dtype=torch.int, pin_memory=prefer_pinned()
            )
            previous_slots = self.previous_batch_indices_cuda[:previous_batch_len]
            previous_slots.copy_(previous_batch_indices_host, non_blocking=True)
            return previous_slots

        num_tokens = len(input_ids)
        num_draft_tokens = len(draft_tokens)
        total_num_tokens = len(position_ids)
        assert total_num_tokens <= self._config.max_num_tokens, (
            f"total_num_tokens ({total_num_tokens}) should be less than or equal to max_num_tokens ({self._config.max_num_tokens})"  # noqa: E501
        )
        # if exist requests that do not have previous batch, copy input_ids and draft_tokens
        if num_tokens > 0:
            input_ids = torch.tensor(input_ids, dtype=torch.int, pin_memory=prefer_pinned())
            self.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        if num_draft_tokens > 0:
            draft_tokens = torch.tensor(draft_tokens, dtype=torch.int, pin_memory=prefer_pinned())
            self.draft_tokens_cuda[: len(draft_tokens)].copy_(draft_tokens, non_blocking=True)
        if self._config.is_spec_decode and len(num_accepted_draft_tokens) > 0:
            num_accepted_draft_tokens = torch.tensor(
                num_accepted_draft_tokens, dtype=torch.int, pin_memory=prefer_pinned()
            )
            self.num_accepted_draft_tokens_cuda[: len(num_accepted_draft_tokens)].copy_(
                num_accepted_draft_tokens, non_blocking=True
            )
        if next_draft_tokens_device is not None:
            # Initialize these two values to zeros
            self.previous_pos_id_offsets_cuda *= 0
            self.previous_kv_lens_offsets_cuda *= 0
            runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(runtime_draft_len)
            runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1

            if previous_batch_len > 0:
                previous_slots = previous_seq_slots_device()
                # previous input ids
                previous_batch_tokens = previous_batch_len * runtime_tokens_per_gen_step
                new_tokens = new_tokens_device.transpose(0, 1)[
                    previous_slots, :runtime_tokens_per_gen_step
                ].flatten()
                self.input_ids_cuda[num_tokens : num_tokens + previous_batch_tokens].copy_(
                    new_tokens, non_blocking=True
                )

                # previous draft tokens
                previous_batch_draft_tokens = previous_batch_len * runtime_draft_token_buffer_width
                if runtime_draft_token_buffer_width > 0:
                    self.draft_tokens_cuda[
                        num_draft_tokens : num_draft_tokens + previous_batch_draft_tokens
                    ].copy_(
                        next_draft_tokens_device[
                            previous_slots, :runtime_draft_token_buffer_width
                        ].flatten(),
                        non_blocking=True,
                    )
                # prepare data for the preprocess inputs
                kv_len_offsets_device = new_tokens_lens_device - runtime_tokens_per_gen_step
                previous_pos_indices_host = torch.tensor(
                    previous_pos_indices, dtype=torch.int, pin_memory=prefer_pinned()
                )
                self.previous_pos_indices_cuda[0:previous_batch_tokens].copy_(
                    previous_pos_indices_host, non_blocking=True
                )

                # The order of requests in a batch: [context requests, generation requests]
                # generation requests: ['requests that do not have previous batch', 'requests that
                # already have previous batch', 'dummy requests']
                #   1) 'requests that do not have previous batch': disable overlap scheduler or the
                #      first step in the generation server of disaggregated serving.
                #   2) 'requests that already have previous batch': previous iteration's requests.
                #   3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                # Therefore, both of self.previous_pos_id_offsets_cuda and
                # self.previous_kv_lens_offsets_cuda are also 3 segments.
                #   For 1) 'requests that do not have previous batch': disable overlap scheduler or
                #          the first step in the generation server of disaggregated serving.
                #       Set these requests' previous_pos_id_offsets and previous_kv_lens_offsets to
                #       '0' to skip the value changes in _preprocess_inputs.
                #       Already set to '0' during initialization.
                #   For 2) 'requests that already have previous batch': enable overlap scheduler.
                #       Set their previous_pos_id_offsets and previous_kv_lens_offsets according to
                #       new_tokens_lens_device and kv_len_offsets_device.
                #   For 3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                #       Already set to '0' during initialization.

                num_extend_reqeust_wo_dummy = len(extend_requests) - len(extend_dummy_requests)
                self.previous_pos_id_offsets_cuda[
                    (num_extend_reqeust_wo_dummy - previous_batch_len)
                    * runtime_tokens_per_gen_step : num_extend_reqeust_wo_dummy
                    * runtime_tokens_per_gen_step
                ].copy_(
                    new_tokens_lens_device[self.previous_pos_indices_cuda[0:previous_batch_tokens]],
                    non_blocking=True,
                )

                self.previous_kv_lens_offsets_cuda[
                    num_extend_reqeust_wo_dummy - previous_batch_len : num_extend_reqeust_wo_dummy
                ].copy_(kv_len_offsets_device[previous_slots], non_blocking=True)

        elif new_tokens_device is not None:
            seq_slots_device = previous_seq_slots_device()
            max_draft_len = max(draft_lens)
            new_tokens = new_tokens_device[
                : max_draft_len + 1, seq_slots_device, : self._config.max_beam_width
            ]
            self.input_ids_cuda[
                num_tokens : num_tokens + previous_batch_len * self._config.max_beam_width
            ].copy_(new_tokens.flatten(), non_blocking=True)

        if (
            not self._config.disable_overlap_scheduler
            and next_draft_tokens_device is None
            and len(extend_requests) > 0
        ):
            # During warmup, for those generation requests, we don't have previous tensors,
            # so we need to set the previous_pos_id_offsets and previous_kv_lens_offsets to zeros
            # to skip the value changes in _preprocess_inputs. Otherwise, there will be illegal memory access
            # when writing key/values to the KV cache.
            self.previous_pos_id_offsets_cuda *= 0
            self.previous_kv_lens_offsets_cuda *= 0

        position_ids = apply_position_id_offset(position_ids, model=self.model)
        host_position_ids = torch.tensor(position_ids, dtype=torch.int, pin_memory=prefer_pinned())
        # Use the (3,1,N) MRoPE layout whenever the model declares MRoPE, even
        # for text-only batches: keeping position_ids rank-consistent between
        # warmup and serving keeps torch.compile guards stable, so piecewise
        # CUDA graphs captured at warmup remain usable at runtime.
        if self._config.use_mrope:
            # Mixed batches may have only some requests with multimodal MRoPE
            # data. Seed the full (3,1,N) buffer from scalar position_ids
            # (text-only tokens get the same value on all 3 axes), then
            # overwrite only the multimodal spans with their real MRoPE coords.
            self.position_ids_cuda[:total_num_tokens].copy_(host_position_ids, non_blocking=True)
            # Broadcast [N] to [3,1,N]: default for text-only tokens.
            self.mrope_position_ids_cuda[:, :, :total_num_tokens].copy_(
                self.position_ids_cuda[:total_num_tokens].view(1, 1, -1).expand(3, 1, -1),
                non_blocking=True,
            )
            # Overwrite multimodal spans with per-axis MRoPE positions.
            for start_idx, end_idx, segment in mrope_position_ids:
                if segment.ndim != 3:
                    raise RuntimeError(
                        f"Expected 3D mrope_position_ids, got shape {tuple(segment.shape)}"
                    )
                if segment.shape[0] != 3 and segment.shape[-1] == 3:
                    logger.warning(
                        "Transposing unexpected mrope_position_ids shape from "
                        f"{tuple(segment.shape)}"
                    )
                    segment = segment.transpose(0, 2).contiguous()
                if segment.shape[:2] != (3, 1):
                    raise RuntimeError(
                        f"Unexpected mrope_position_ids shape {tuple(segment.shape)} for span {start_idx}:{end_idx}"
                    )
                segment = segment.contiguous()
                if segment.device.type == "cpu":
                    segment = maybe_pin_memory(segment)
                self.mrope_position_ids_cuda[:, :, start_idx:end_idx].copy_(
                    segment[:, :, : end_idx - start_idx], non_blocking=True
                )
            final_position_ids = self.mrope_position_ids_cuda[:, :, :total_num_tokens]
        else:
            self.position_ids_cuda[:total_num_tokens].copy_(host_position_ids, non_blocking=True)
            final_position_ids = self.position_ids_cuda[:total_num_tokens].unsqueeze(0)

        if enable_spec_decode:
            self.gather_ids_cuda[: len(gather_ids)].copy_(
                torch.tensor(gather_ids, dtype=torch.int, pin_memory=prefer_pinned()),
                non_blocking=True,
            )

        if self.mapping.has_cp_helix():
            # A non-None owned-count list is what arms
            # _helix_spec_tokens_valid, and the per-token slots/bounds that
            # flag gates are only ever filled by recompute_helix_spec_buffers,
            # which _preprocess_inputs runs under enable_spec_decode. Gate the
            # hand-off here, at the single choke point, so no packing loop can
            # arm the spec path for ordinary helix generation and send its
            # consumers to uninitialized buffers.
            helix_spec_active = bool(enable_spec_decode and helix_owned_new_tokens)
            attn_metadata.update_helix_param(
                helix_position_offsets=helix_position_offsets,
                helix_is_inactive_rank=helix_is_inactive_rank,
                helix_owned_new_tokens=(helix_owned_new_tokens if helix_spec_active else None),
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
                # Copy cache indirection to local buffer with offsets changing:  seq_slots[i] -> i
                # Convert to GPU tensor to avoid implicit sync
                gen_request_seq_slots_tensor = torch.tensor(
                    gen_request_seq_slots, dtype=torch.long, pin_memory=prefer_pinned()
                ).to(device="cuda", non_blocking=True)
                self.cache_indirection_attention[:num_generation_requests].copy_(
                    cache_indirection_buffer[gen_request_seq_slots_tensor]
                )
            if cache_indirection_buffer is not None or is_dummy:
                attn_metadata.beam_width = self._config.max_beam_width
        else:
            attn_metadata.beam_width = 1

        attn_metadata.request_ids = request_ids
        attn_metadata.prompt_lens = prompt_lengths
        attn_metadata.num_contexts = scheduled_requests.num_context_requests
        # Use num_chunked_ctx_requests to record the number of extend context requests,
        # so that we can update the kv_lens_cuda correctly in _preprocess_inputs.
        attn_metadata.num_chunked_ctx_requests = 0
        if (
            enable_spec_decode
            and spec_config.spec_dec_mode.extend_ctx(self._config.attention_backend)
            and spec_config.is_linear_tree
        ):
            # For the tree decoding, we want to use XQA to process the draft tokens for the target model.
            # Therefore, we do not treat them as the chunked context requests.
            attn_metadata.num_contexts += len(extend_requests)
            attn_metadata.num_chunked_ctx_requests = len(extend_requests)

        attn_metadata.kv_cache_params = KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            num_extra_kv_tokens=get_num_extra_kv_tokens(spec_config),
            use_full_generation_page_table=(
                self._should_use_full_generation_page_table(spec_config, attn_metadata)
            ),
        )
        attn_metadata.kv_cache_manager = kv_cache_manager

        if hasattr(self.model.model_config.pretrained_config, "chunk_size"):
            attn_metadata.mamba_chunk_size = self.model.model_config.pretrained_config.chunk_size
        # Some sparse backends (RocketKV) clamp
        # kv_cache_params.num_cached_tokens_per_seq in place during prepare(),
        # and KVCacheParams holds the list by reference. Snapshot the true
        # pre-prepare counts so the steady-gen recording below stores values
        # that the per-step prepare() can re-clamp from scratch.
        num_cached_tokens_snapshot = list(num_cached_tokens_per_seq)
        attn_metadata.prepare()
        extra_model_inputs = extra_inputs.build(attn_metadata, resource_manager)

        peft_cache_manager = resource_manager and resource_manager.get_resource_manager(
            ResourceManagerType.PEFT_CACHE_MANAGER
        )
        lora_params = self._lora.build(
            scheduled_requests,
            attn_metadata,
            enable_spec_decode=enable_spec_decode,
            runtime_draft_len=runtime_draft_len,
            peft_cache_manager=peft_cache_manager,
            maybe_graph=maybe_graph,
            use_lora_graph=use_lora_graph,
        )

        if spec_metadata is not None:
            # Set the per-batch counts here, before the attention-DP allgather
            # below: the allgather and prepare() must derive the DP token count
            # from the same fields (see SpecMetadata.dp_num_tokens). Use
            # scheduled_requests.num_generation_requests -- the same-named
            # local above excludes CUDA-graph dummies and would not match the
            # count prepare() uses.
            spec_metadata.num_tokens = total_num_tokens
            spec_metadata.num_generations = scheduled_requests.num_generation_requests
            spec_metadata.seq_lens = sequence_lengths

        spec_all_rank_counts = None
        if spec_metadata is not None and self._config.enable_attention_dp:
            (attn_all_rank_num_tokens, spec_all_rank_counts) = (
                self._get_all_rank_num_tokens_and_spec_counts(attn_metadata, spec_metadata)
            )
        else:
            attn_all_rank_num_tokens = get_all_rank_num_tokens(
                attn_metadata,
                enable_attention_dp=self._config.enable_attention_dp,
                mapping=self.mapping,
                dist=self.dist,
            )
        (padded_num_tokens, can_run_prefill_cuda_graph, attn_all_rank_num_tokens) = (
            get_padding_params(
                total_num_tokens,
                num_ctx_requests,
                attn_all_rank_num_tokens,
                dist=self.dist,
                enable_attention_dp=self._config.enable_attention_dp,
                prefill_cuda_graph_backend=self._config.prefill_cuda_graph_backend,
                prefill_cuda_graph_num_tokens=self._config.prefill_cuda_graph_num_tokens,
            )
        )
        set_per_request_prefill_cuda_graph_flag(can_run_prefill_cuda_graph)
        attn_metadata.padded_num_tokens = (
            padded_num_tokens if padded_num_tokens != total_num_tokens else None
        )

        virtual_num_tokens = total_num_tokens
        if attn_metadata.padded_num_tokens is not None:
            self.input_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
            virtual_num_tokens = padded_num_tokens
            # Match the rank of the unpadded branch: MRoPE models always use
            # the (3,1,N) layout (see the seeding block above), so the padded
            # view must stay 3D as well to keep torch.compile guards stable.
            if self._config.use_mrope:
                # Zero-fill padding on dim 2 (token dim) of (3,1,N) buffer.
                self.mrope_position_ids_cuda[:, :, total_num_tokens:padded_num_tokens].fill_(0)
                final_position_ids = self.mrope_position_ids_cuda[:, :, :virtual_num_tokens]
            else:
                self.position_ids_cuda[total_num_tokens:padded_num_tokens].fill_(0)
                final_position_ids = self.position_ids_cuda[:virtual_num_tokens].unsqueeze(0)

        if self._config.enable_attention_dp:
            attn_metadata.all_rank_num_tokens = attn_all_rank_num_tokens

        # Prepare inputs
        inputs = {
            "attn_metadata": attn_metadata,
            "input_ids": self.input_ids_cuda[:virtual_num_tokens],
            "position_ids": final_position_ids,
            "inputs_embeds": None,
            "multimodal_params": multimodal_params_list,
            "resource_manager": resource_manager,
        }
        inputs.update(extra_model_inputs)

        if self._config.use_mrope:
            if mrope_delta_write_seq_slots:
                delta_write_seq_slots = torch.tensor(
                    mrope_delta_write_seq_slots, dtype=torch.long, pin_memory=prefer_pinned()
                )
                inputs["mrope_delta_write_seq_slots"] = delta_write_seq_slots.to(
                    device="cuda", non_blocking=True
                )

            if mrope_delta_read_seq_slots:
                delta_read_seq_slots = torch.tensor(
                    mrope_delta_read_seq_slots, dtype=torch.long, pin_memory=prefer_pinned()
                )
                inputs["mrope_delta_read_seq_slots"] = delta_read_seq_slots.to(
                    device="cuda", non_blocking=True
                )

        if bool(lora_params):
            inputs["lora_params"] = lora_params

        if spec_metadata is not None:
            total_draft_lens = sum(draft_lens)
            spec_metadata.draft_tokens = self.draft_tokens_cuda[:total_draft_lens]
            spec_metadata.request_ids = request_ids
            spec_metadata.gather_ids = self.gather_ids_cuda[: len(gather_ids)]
            # num_generations / num_tokens / seq_lens are set above, before the
            # attention-DP allgather that must agree with prepare().
            spec_metadata.host_position_ids = host_position_ids
            spec_metadata.num_accepted_draft_tokens = self.num_accepted_draft_tokens_cuda[
                : len(num_accepted_draft_tokens)
            ]
            if context_prompt_lookahead is not None:
                spec_metadata.populate_context_prompt_lookahead(context_prompt_lookahead)
            # No-op for non 1-model
            spec_metadata.populate_sampling_params_for_one_model(scheduled_requests.all_requests())
            spec_metadata.prepare()
            # One-model rejection: one-hot the stale draft_probs rows of gen
            # requests that produced no draft tokens this step, so the (possibly
            # captured) rejection kernel reads a legal placeholder distribution.
            spec_metadata.write_padding_onehot_draft_probs(padding_gen_slots, runtime_draft_len)
            inputs["spec_metadata"] = spec_metadata

            if self._config.enable_attention_dp:
                set_spec_metadata_all_rank_num_tokens(spec_metadata, *spec_all_rank_counts)

        if mm_token_indices is not None:
            ship_multimodal_indices(
                inputs,
                mm_token_indices_cpu=mm_token_indices,
                text_token_indices_cpu=text_token_indices_ctx,
                num_ctx_tokens=num_ctx_tokens,
                total_num_tokens=total_num_tokens,
            )

        num_generation_tokens = (
            len(generation_requests)
            + len(extend_requests)
            + sum(draft_lens)
            + len(first_draft_requests)
        )
        self.iter_states["num_ctx_requests"] = num_ctx_requests
        self.iter_states["num_ctx_tokens"] = num_ctx_tokens
        self.iter_states["num_generation_tokens"] = num_generation_tokens
        # Count the already-cached prefix for the sequences scheduled this iteration.
        self.iter_states["cached_kv_tokens"] = sum(num_cached_tokens_per_seq)
        if self._log_cached_kv_tokens_per_req:
            self._record_cached_kv_tokens_per_req(
                num_cached_tokens_per_seq,
                (
                    (scheduled_requests.context_requests, 1),
                    (extend_requests, 1),
                    (first_draft_requests, 1),
                    (generation_requests, beam_width if generation_requests else 1),
                ),
            )

        if not is_dummy:
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
            if (
                self._config.spec_config is None
                and spec_metadata is None
                and new_tokens_device is not None
                and self.guided_decoder is None
                and not self._config.enable_attention_dp
                and not mrope_position_ids
                and not mrope_delta_write_seq_slots
                and not mrope_delta_read_seq_slots
                and not self.use_beam_search
                and self._config.max_beam_width == 1
                and self._steady_gen_cache_supported
                and not _has_cp_helix
                and num_ctx_requests == 0
                and not extend_requests
                and not first_draft_requests
                and _n_gen > 0
                and previous_batch_len == _n_gen
                and num_tokens == 0
                and not _has_any_multimodal_request
                and not multimodal_params_list
                and not lora_params
                and attn_metadata.padded_num_tokens is None
                and get_position_id_offset(self.model) == 0
                and not getattr(kv_cache_manager, "kv_compression_manages_history", False)
            ):
                self._steady_gen_positions_pinned[:_n_gen].copy_(
                    torch.as_tensor(num_cached_tokens_snapshot, dtype=torch.int)
                )
                self._steady_gen_cache = {
                    "num_requests": _n_gen,
                    "request_ids": all_gen_request_ids,
                    "prompt_lens": prompt_lengths,
                    "seq_lens_ones": maybe_pin_memory(torch.ones(_n_gen, dtype=torch.int)),
                    "use_mrope": _use_mrope,
                }

        gather_ids_device = self.gather_ids_cuda[: len(gather_ids)] if enable_spec_decode else None
        return inputs, gather_ids_device, runtime_draft_len

    @nvtx_range("_prepare_inputs")
    def _prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata | None = None,
        new_tensors_device: SampleStateTensors | None = None,
        cache_indirection_buffer: torch.Tensor | None = None,
        resource_manager: ResourceManager | None = None,
        maybe_graph: bool = False,
        promoted_context_request_ids: frozenset[int] = frozenset(),
        use_lora_graph: bool = False,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
        is_dummy: bool,
    ) -> tuple[dict[str, Any], torch.Tensor | None, int]:
        set_per_request_prefill_cuda_graph_flag(False)
        if self.mapping is not None and "cp_type" in self.mapping.cp_config:
            cp_type = self.mapping.cp_config["cp_type"]
            if cp_type in (CpType.HELIX, CpType.ULYSSES):
                # Take the usual route of _prepare_tp_inputs.
                pass
            else:
                raise NotImplementedError(
                    f"Unsupported cp_type {getattr(cp_type, 'name', cp_type)}."
                )

        # Initialize SA state for new requests (MTP+SA, EAGLE3+SA, PARD+SA, etc.)
        has_sa_enhancer = (
            self._config.spec_config is not None
            and getattr(self._config.spec_config, "sa_config", None) is not None
        )
        if has_sa_enhancer and resource_manager is not None and self.mapping.is_last_pp_rank():
            from tensorrt_llm._torch.speculative.suffix_automaton import SuffixAutomatonManager

            spec_rm = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER
            )
            sa_manager = None
            if spec_rm is not None:
                if isinstance(spec_rm, SuffixAutomatonManager):
                    sa_manager = spec_rm
                else:
                    sa_manager = getattr(spec_rm, "sa_manager", None)
            if sa_manager is not None:
                for request in scheduled_requests.all_requests():
                    if request.py_request_id not in sa_manager._initialized_requests:
                        sa_manager.add_request(request.py_request_id, request.get_tokens(0))
                        sa_manager._initialized_requests.add(request.py_request_id)

        return self._prepare_tp_inputs(
            scheduled_requests,
            kv_cache_manager,
            attn_metadata,
            spec_metadata,
            new_tensors_device,
            cache_indirection_buffer,
            resource_manager,
            maybe_graph,
            promoted_context_request_ids,
            use_lora_graph=use_lora_graph,
            enable_spec_decode=enable_spec_decode,
            runtime_draft_len=runtime_draft_len,
            is_dummy=is_dummy,
        )

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
        draft_kv_cache_manager = self._get_draft_kv_cache_manager(resource_manager)

        attn_metadata = self._set_up_attn_metadata(kv_cache_manager, draft_kv_cache_manager)
        if isinstance(attn_metadata, TrtllmAttentionMetadata):
            attn_metadata.trtllm_gen_jit_warmup = self._trtllm_gen_jit_warmup
        if enable_spec_decode:
            spec_resource_manager, spec_tree_manager = get_spec_managers(resource_manager)
            spec_metadata = self._set_up_spec_metadata(spec_resource_manager)
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
        graph_requests = scheduled_requests
        promoted_context_request_ids: frozenset[int] = frozenset()
        # Non-linear tree input preparation expands runtime_draft_len to the
        # total tree width after graph selection. Only linear-tree zero-draft
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
            and self.cuda_graph_runner.enabled
            and can_promote_spec_decode
            and not self.use_beam_search
            and self._context_graph_promotion_supported
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
                self._pad_batch_seed_mrope_delta_cache(padded_graph_requests)

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
                    draft_tokens_cuda=self.draft_tokens_cuda
                    if self._config.is_spec_decode
                    else None,
                    new_tensors_device=new_tensors_device,
                    spec_resource_manager=spec_resource_manager,
                    promoted_context_request_ids=promoted_context_request_ids,
                    peft_cache_data_type=peft_cache_data_type,
                    use_lora_graph=use_lora_graph,
                )
            )

            can_run_graph = key is not None
            if can_run_graph:
                attn_metadata = maybe_attn_metadata
                spec_metadata = maybe_spec_metadata
                execution_requests = padded_graph_requests
                execution_promoted_context_ids = promoted_context_request_ids
            else:
                attn_metadata = self.attn_metadata
                if enable_spec_decode:
                    spec_metadata = self.spec_metadata
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

            inputs, gather_ids, runtime_draft_len = self._prepare_inputs(
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
                            gather_ids=gather_ids,
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
                    needs_capture = self.cuda_graph_runner.needs_capture(key)
                    if needs_capture:

                        def capture_forward_fn(inputs: dict[str, Any]):
                            with MoeLoadBalancerIterContext(moe_load_balancer):
                                return self._forward_step(
                                    inputs,
                                    enable_spec_decode=enable_spec_decode,
                                    runtime_draft_len=runtime_draft_len,
                                    is_dummy=is_dummy,
                                    gather_ids=gather_ids,
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
                            outputs = self.cuda_graph_runner.replay(key, inputs)
                        finally:
                            restore_attn_metadata_after_draft_replay(attn_metadata, saved_draft)
                    else:
                        saved_draft = prepare_attn_metadata_for_draft_replay(
                            attn_metadata, draft_kv_cache_manager
                        )
                        try:
                            with MoeLoadBalancerIterContext(moe_load_balancer):
                                outputs = self.cuda_graph_runner.replay(key, inputs)
                        finally:
                            restore_attn_metadata_after_draft_replay(attn_metadata, saved_draft)

            if self.forward_pass_callable is not None:
                self.forward_pass_callable()

            self._execute_logit_post_processors(scheduled_requests, outputs)

            if not isinstance(outputs, dict):
                return outputs
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
        outputs = self.model_forward(
            **inputs,
            is_dummy=is_dummy,
            return_context_logits=gather_ids is not None or gather_context_logits,
        )

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
