# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder execution moved out of ``PyTorchModelEngine``."""

import contextlib
import functools
import inspect
from collections.abc import Callable
from typing import Any

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.memory_buffer_utils import with_shared_pool
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
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
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RemoteTailPhase
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm._torch.speculative import (
    SpecMetadata,
    get_draft_kv_cache_manager,
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata
from tensorrt_llm._torch.utils import (
    get_per_request_prefill_cuda_graph_flag,
    with_model_extra_attrs,
)
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.multimodal import _has_mm_payload_keys
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor, InputProcessor
from tensorrt_llm.llmapi.llm_args import BaseSparseAttentionConfig, PrefillCudaGraphBackend
from tensorrt_llm.mapping import Mapping

from ...lora import LoraParamBuilder, make_cuda_graph_lora_manager
from ...model_call import ModelCaller
from ...multimodal import MultimodalItemScheduler
from ..common import get_top_level_model, make_scheduled_inputs, prepare_multimodal_indices
from ..interface import PreparedInputs, ScheduledInputs, ScheduledModelRunner
from .config import DecoderRunnerConfig
from .dummy_requests import DummyRequestFactory
from .graphs import DecoderCudaGraphs
from .inputs import (
    DecoderInputBuffers,
    DecoderInputPreparer,
    ExtraInputsCollector,
    resolve_runtime_draft_len,
)
from .metadata import (
    DecoderMetadata,
    get_spec_managers,
    sync_group_all_greedy_sample,
    update_spec_metadata,
)
from .warmup import DecoderWarmup, WarmupAgreement


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

    # Prompt tokens of the filler CUDA graph dummy requests, also the floor of
    # the longest one; None keeps the cache manager's default.
    _dummy_request_tokens: int | None = None
    # Collects per-sequence model inputs beyond the decoder's own; None when the
    # model takes none. Steady-generation reuse, context-row promotion, the
    # synthetic context warmups and eager workspace reclamation do not carry
    # such inputs, so all of them stay off while it is set.
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
            free_resources=self._free_dummy_request_resources,
            get_draft_kv_cache_manager=self._get_draft_kv_cache_manager,
            release_request_state=getattr(model, "release_request_state", None),
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
        )
        self._warmup = DecoderWarmup(
            self._config,
            mapping,
            dist,
            model,
            sparse_attention_config,
            torch_compile_backend,
            kv_cache_manager_key,
            warmup_timer,
            metrics,
            self._agreement,
            self._metadata,
            self._dummy_requests,
            self._graphs,
            self._inputs,
            forward_warmup=self._forward_warmup,
            get_draft_kv_cache_manager=self._get_draft_kv_cache_manager,
            context_warmups_supported=self._new_extra_inputs is None,
            eager_workspace_reclaimable=self._new_extra_inputs is None,
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
        self._warmup.run(resource_manager)

    def release_graphs(self) -> None:
        self._graphs.release()

    def wait_for_input_copy(self) -> None:
        self._inputs.wait_for_input_copy()

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
        self._lora = LoraParamBuilder(
            spec_config=self._config.spec_config,
            attn_backend=self._config.attention_backend,
            cuda_graph_manager=None,
        )
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

    def _free_dummy_request_resources(
        self, requests: list[LlmRequest], resource_manager: ResourceManager
    ) -> None:
        """Free what ``_add_dummy_request_resources`` allocated."""

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

        attn_metadata = self._metadata.setup_attention(kv_cache_manager, draft_kv_cache_manager)
        if isinstance(attn_metadata, TrtllmAttentionMetadata):
            attn_metadata.trtllm_gen_jit_warmup = self._warmup.trtllm_gen_jit_warmup
        if enable_spec_decode:
            spec_resource_manager, spec_tree_manager = get_spec_managers(resource_manager)
            spec_metadata = self._metadata.setup_speculative(spec_resource_manager)
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
                sync_group_all_greedy_sample(spec_metadata, self.mapping, self.dist)

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
                self._inputs.record_promoted_iter_states(scheduled_requests)
            self._inputs.record_input_copy()

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
                            self._inputs.postprocess_inputs(
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
        reclaimer = self._warmup.eager_workspace_reclaimer
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
        inputs = self._inputs.preprocess_inputs(
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
