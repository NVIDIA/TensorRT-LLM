# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decoder CUDA graph capture and replay state."""

import contextlib
import functools
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple, Protocol

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.compilation.piecewise_optimizer import PiecewiseRunner
from tensorrt_llm._torch.compilation.utils import capture_piecewise_cuda_graph
from tensorrt_llm._torch.memory_buffer_utils import with_shared_pool
from tensorrt_llm._torch.models.modeling_utils import timing_metric
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import moe_comm_serving_timeouts
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.nccl_window_tensor_scope import discard_nccl_window_tensor_outputs
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    BaseResourceManager,
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import runtime_hooks_of
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTensors
from tensorrt_llm._torch.pyexecutor.sampler.sampler_common import SampleType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RemoteTailPhase
from tensorrt_llm._torch.speculative import (
    SpecMetadata,
    prepare_attn_metadata_for_draft_replay,
    restore_attn_metadata_after_draft_replay,
)
from tensorrt_llm._torch.speculative.utils import get_static_draft_len
from tensorrt_llm._torch.utils import get_per_request_prefill_cuda_graph_flag
from tensorrt_llm.inputs.multimodal import _has_mm_payload_keys
from tensorrt_llm.llmapi.llm_args import BaseSparseAttentionConfig, PrefillCudaGraphBackend
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...lora import LoraParamBuilder
from ..common import get_top_level_model, prepare_multimodal_indices
from .config import DecoderRunnerConfig
from .dummy_requests import DummyRequestFactory
from .inputs import DecoderInputPreparer
from .metadata import DecoderMetadata, resolve_mamba_metadata_cls
from .warmup import WarmupAgreement

# Context requests per warmup batch whose breakable prefill graph breaks get
# CUDA graphs; the short prefill iterations these graphs speed up rarely
# schedule more, and their breaks run eagerly.
_BREAK_GRAPH_NUM_CONTEXTS = (1, 2)


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


def compute_dynamic_draft_len_mapping(config: DecoderRunnerConfig) -> dict[int, int] | None:
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
        not config.spec_config
        or not config.spec_config.draft_len_schedule
        or not config.spec_config.spec_dec_mode.support_dynamic_draft_len()
    ):
        return None

    schedule = config.spec_config.draft_len_schedule
    schedule_keys = list(schedule.keys())

    mapping = {}
    key_idx = 0
    for graph_bs in config.cuda_graph_batch_sizes:
        while key_idx < len(schedule_keys) and schedule_keys[key_idx] < graph_bs:
            key_idx += 1
        if key_idx < len(schedule_keys):
            draft_len = schedule[schedule_keys[key_idx]]
        else:
            draft_len = 0
        mapping[graph_bs] = draft_len
    return mapping


class GraphSelection(NamedTuple):
    """The graph a batch runs as, and the metadata that graph was captured with."""

    key: KeyType | None
    attn_metadata: AttentionMetadata | None
    spec_metadata: SpecMetadata | None
    use_lora_graph: bool


class ForwardWarmup(Protocol):
    """Run one dummy scheduled pass through the runner's forward.

    ``trtllm_gen_jit_warmup`` marks the pass as the TRTLLM-Gen FMHA JIT warmup.
    """

    def __call__(
        self,
        batch: ScheduledRequests,
        resource_manager: ResourceManager,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
        trtllm_gen_jit_warmup: bool = False,
    ) -> Any: ...


class DecoderCudaGraphs:
    """Owns the decoder's CUDA graph runners and captures their graphs.

    A model family that prepares capture batches differently, or captures
    shapes beyond the generation ones, overrides ``_prepare_capture_batch``
    and ``_capture_additional_cuda_graphs``.
    """

    def __init__(
        self,
        *,
        cuda_graph_runner: CUDAGraphRunner,
        breakable_cuda_graph_runner: BreakableCUDAGraphRunner | None,
        config: DecoderRunnerConfig,
        mapping: Mapping,
        model: torch.nn.Module,
        sparse_attention_config: BaseSparseAttentionConfig | None,
        metrics: dict[str, float],
        dummy_requests: DummyRequestFactory,
        agreement: WarmupAgreement,
        metadata: DecoderMetadata,
        get_runtime_tokens_per_gen_step: Callable[[int], int],
        dynamic_draft_len_mapping: dict[int, int] | None,
        lora: LoraParamBuilder,
        forward_warmup: ForwardWarmup,
        torch_compile_bypass: Callable[[], contextlib.AbstractContextManager[None]],
        context_promotion_supported: bool,
        warmup_inputs: torch.Tensor | None = None,
        inputs: DecoderInputPreparer | None = None,
    ) -> None:
        self.cuda_graph_runner = cuda_graph_runner
        self._inputs = inputs
        self.breakable_cuda_graph_runner = breakable_cuda_graph_runner
        self._config = config
        self.mapping = mapping
        self.model = model
        self.sparse_attention_config = sparse_attention_config
        self._metrics = metrics
        self._dummy_requests = dummy_requests
        self._agreement = agreement
        self._metadata = metadata
        self.get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step
        self._dynamic_draft_len_mapping = dynamic_draft_len_mapping
        self._lora = lora
        self._forward_warmup = forward_warmup
        self._warmup_inputs = warmup_inputs
        self._torch_compile_bypass = torch_compile_bypass
        # Promoted context rows carry no per-sequence extra model inputs.
        self._context_promotion_supported = context_promotion_supported
        self._force_lora_graph_for_capture: bool | None = None

    def forward_discarded_warmup(
        self,
        batch: ScheduledRequests,
        resource_manager: ResourceManager,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
    ) -> None:
        """Release eager warmup outputs after their last consumer."""
        # Dummy requests can be host-only; the persistent input buffer opens
        # the native scope before forward creates NCCL window tensors.
        with discard_nccl_window_tensor_outputs((self._warmup_inputs, batch)):
            self._forward_warmup(
                batch,
                resource_manager,
                enable_spec_decode=enable_spec_decode,
                runtime_draft_len=runtime_draft_len,
            )

    @property
    def force_lora_graph_for_capture(self) -> bool | None:
        return self._force_lora_graph_for_capture

    @property
    def enabled(self) -> bool:
        """Return whether generation batches may run as CUDA graphs."""
        return self.cuda_graph_runner.enabled

    def register_sample_type_resolver(self, resolver: Callable | None) -> None:
        """Register how a batch maps to its sampling tier, for the graph key."""
        self.cuda_graph_runner.register_sample_type_resolver(resolver)

    def pad_batch(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        runtime_draft_len: int,
    ) -> contextlib.AbstractContextManager[ScheduledRequests]:
        """Pad ``scheduled_requests`` with dummy rows up to a captured batch size."""
        return self.cuda_graph_runner.pad_batch(
            scheduled_requests, resource_manager, runtime_draft_len
        )

    def select(
        self,
        padded_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        *,
        enable_spec_decode: bool,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata | None,
        draft_tokens_cuda: torch.Tensor | None,
        new_tensors_device: SampleStateTensors | None,
        spec_resource_manager: BaseResourceManager | None,
        promoted_context_request_ids: frozenset[int],
    ) -> GraphSelection:
        """Pick the graph for ``padded_requests``; the key is None to run eagerly."""
        peft_cache_data_type = None
        if self._lora.cuda_graph_manager is not None:
            peft_cache_manager = resource_manager.get_resource_manager(
                ResourceManagerType.PEFT_CACHE_MANAGER
            )
            peft_cache_data_type = peft_cache_manager.data_type

        use_lora_graph = self.use_lora_cuda_graph(padded_requests)
        graph_attn_metadata, graph_spec_metadata, key = self.cuda_graph_runner.maybe_get_cuda_graph(
            padded_requests,
            enable_spec_decode=enable_spec_decode,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_tokens_cuda=draft_tokens_cuda,
            new_tensors_device=new_tensors_device,
            spec_resource_manager=spec_resource_manager,
            promoted_context_request_ids=promoted_context_request_ids,
            peft_cache_data_type=peft_cache_data_type,
            use_lora_graph=use_lora_graph,
        )
        return GraphSelection(key, graph_attn_metadata, graph_spec_metadata, use_lora_graph)

    def attention_dp_execution_batch(
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

    def captures_model_body(self, key: KeyType | None) -> bool:
        """Return whether an eager pass records the model body into a breakable graph."""
        breakable_runner = self.breakable_cuda_graph_runner
        return key is None and breakable_runner is not None and breakable_runner.is_capturing

    def reset_padding_dummies(self) -> None:
        """Drop padding dummies that belong to an earlier KV cache manager."""
        self.cuda_graph_runner.reset_padding_dummies()

    def preallocate_padding_dummies(self, resource_manager: ResourceManager) -> None:
        """Allocate the padding dummies while the KV cache is still empty."""
        self.cuda_graph_runner.preallocate_padding_dummies(resource_manager)

    def warmup_and_capture(self, resource_manager: ResourceManager) -> None:
        """Warm up every graph shape, then capture them all.

        Attention kernels can switch implementations at smaller batch sizes and
        require a larger workspace, so the first pass grows the workspace to
        its maximum size. The second pass runs the final per-shape warmup and
        captures without resizing the workspace.
        """
        with self.cuda_graph_runner.allow_capture():
            self.cuda_graph_runner.is_warmup_only = True
            try:
                self.run_cuda_graph_warmup(resource_manager)
            finally:
                self.cuda_graph_runner.is_warmup_only = False
            self.reset_padding_dummies()
            # Captured graphs keep the MoE communication timeouts that were
            # current at capture, and serving replays them, so capture with the
            # serving timeouts. The warmup-only pass above keeps the warmup ones.
            with moe_comm_serving_timeouts():
                self.run_cuda_graph_warmup(resource_manager)

    def _prepare_capture_batch(
        self,
        batch: ScheduledRequests,
        resource_manager: ResourceManager,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
    ) -> None:
        """Prepare a dummy batch's request state before its graph is captured."""

    def _capture_additional_cuda_graphs(self, resource_manager: ResourceManager) -> None:
        """Capture graphs beyond the generation-only shapes; none by default."""

    @contextlib.contextmanager
    def no_cuda_graph(self) -> Iterator[None]:
        _run_cuda_graphs = self.cuda_graph_runner.enabled
        self.cuda_graph_runner.enabled = False
        try:
            yield
        finally:
            self.cuda_graph_runner.enabled = _run_cuda_graphs

    @contextmanager
    def maybe_autotune_lora(self) -> Iterator[None]:
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

    def use_lora_cuda_graph(self, scheduled_requests: ScheduledRequests) -> bool:
        """
        Determines whether a non-LoRA or LoRA CUDA graph should be used, if
        both are available (cuda_graph_specialize_lora==True).
        """
        if self._lora.cuda_graph_manager is None:
            return False
        # Needed during graph capture to enforce a given mode
        if self.force_lora_graph_for_capture is not None:
            return self.force_lora_graph_for_capture
        if not self._config.cuda_graph_specialize_lora:
            return True
        return any(
            request.lora_task_id is not None for request in scheduled_requests.generation_requests
        )

    def promote_final_contexts(
        self,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
        ced_enabled: bool,
    ) -> tuple[ScheduledRequests, frozenset[int]]:
        """Return the graph batch, with final one-token contexts run as decode rows."""
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
        if not (
            scheduled_requests.num_context_requests > 0
            and scheduled_requests.attention_dp_phase is not RemoteTailPhase.CONTEXT
            # A physically pruned context cache cannot become a decode
            # batch, including single-token synthetic prefill warmup rows.
            and getattr(kv_cache_manager, "context_swa_layer_limit", None) is None
            and self.cuda_graph_runner.enabled
            and can_promote_spec_decode
            and not ced_enabled
            and self._config.max_beam_width == 1
            and self._context_promotion_supported
            # PLE owns recurrent n-gram and convolution state. Promoting a
            # fresh final-context row would skip its cache-slot reset.
            and not self._model_uses_ple_recurrent_state
            and self.mapping.cp_size == 1
        ):
            return scheduled_requests, frozenset()
        return _make_single_token_context_graph_batch(
            scheduled_requests, self._is_final_multimodal_context_decode_compatible
        )

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

    def run(
        self,
        key: KeyType | None,
        inputs: dict[str, Any],
        run_model: Callable[[dict[str, Any]], Any],
        postprocess_inputs: Callable[[dict[str, Any]], None],
        *,
        attn_metadata: AttentionMetadata,
        draft_kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None,
        moe_load_balancer: MoeLoadBalancer | None,
        enable_spec_decode: bool,
        gather_context_logits: bool,
        position_id_offsets: torch.Tensor | None,
        static_inputs_staged: int = 0,
    ) -> Any:
        """Run the model by replaying the graph for ``key``, capturing it first if needed.

        Without a key, record the model body while a breakable prefill graph
        captures, run through one when it covers the tokens, otherwise eagerly.
        """
        with with_shared_pool(self.cuda_graph_runner.get_graph_pool()):
            if self.captures_model_body(key):
                return self.breakable_cuda_graph_runner.capture_model_body(
                    lambda: run_model(inputs)
                )
            return self._run(
                key,
                inputs,
                run_model,
                postprocess_inputs,
                attn_metadata=attn_metadata,
                draft_kv_cache_manager=draft_kv_cache_manager,
                moe_load_balancer=moe_load_balancer,
                enable_spec_decode=enable_spec_decode,
                gather_context_logits=gather_context_logits,
                position_id_offsets=position_id_offsets,
                static_inputs_staged=static_inputs_staged,
            )

    def _run(
        self,
        key: KeyType | None,
        inputs: dict[str, Any],
        run_model: Callable[[dict[str, Any]], Any],
        postprocess_inputs: Callable[[dict[str, Any]], None],
        *,
        attn_metadata: AttentionMetadata,
        draft_kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None,
        moe_load_balancer: MoeLoadBalancer | None,
        enable_spec_decode: bool,
        gather_context_logits: bool,
        position_id_offsets: torch.Tensor | None,
        static_inputs_staged: int = 0,
    ) -> Any:
        breakable_runner = self.breakable_cuda_graph_runner
        if key is None:
            num_tokens = inputs["input_ids"].shape[0]
            can_run_breakable_graph = (
                breakable_runner is not None
                and get_per_request_prefill_cuda_graph_flag()
                and not gather_context_logits
                and breakable_runner.has_graph(num_tokens)
            )
            if can_run_breakable_graph and not breakable_runner.is_warming_up:
                outputs = breakable_runner.execute(num_tokens, lambda: run_model(inputs))
            else:
                # real eager or BCG warmup or PCG
                outputs = run_model(inputs)
        else:
            needs_capture = self.cuda_graph_runner.needs_capture(key)
            if needs_capture:

                def capture_model(model_inputs: dict[str, Any]) -> Any:
                    with self._torch_compile_bypass():
                        return run_model(model_inputs)

                capture_outputs = self.cuda_graph_runner.capture(
                    key,
                    capture_model,
                    inputs,
                    enable_spec_decode=enable_spec_decode,
                    postprocess_fn=postprocess_inputs,
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
                        key,
                        inputs,
                        position_id_offsets=position_id_offsets,
                        static_inputs_staged=static_inputs_staged,
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
                            key,
                            inputs,
                            position_id_offsets=position_id_offsets,
                            static_inputs_staged=static_inputs_staged,
                        )
                finally:
                    restore_attn_metadata_after_draft_replay(attn_metadata, saved_draft)
        return outputs

    def _get_graphs_to_capture(
        self, cuda_graph_batch_sizes: list[int]
    ) -> list[tuple[int, int] | tuple[int, int, int]]:
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
            logger.info(
                f"Dynamic draft length enabled for one-model path. "
                f"Capturing {len(graphs)} graphs: {graphs}"
            )
            return graphs

        if self._inputs is not None and self._inputs._dspark_confidence_enabled:
            max_draft_len = int(self._config.spec_config.max_draft_len)
            native_graphs = [(bs, max_draft_len) for bs in cuda_graph_batch_sizes]
            if not self._inputs._dspark_trims_submitted_tokens:
                return native_graphs
            from tensorrt_llm._torch.speculative.dspark_planner import ExactSpsCostTable

            exact_table = self._inputs._dspark_sps_cost_table
            if not isinstance(exact_table, ExactSpsCostTable):
                raise RuntimeError(
                    "DSpark confidence graph capture requires an authenticated exact SPS table"
                )
            ragged_graphs = [
                (bs, max_draft_len, verifier_budget)
                for bs in cuda_graph_batch_sizes
                for verifier_budget in exact_table.production_candidate_budgets(bs)
            ]
            # Full K keeps the ordinary native graph. Compact layouts have
            # distinct keys; they cannot alias its uniform attention layout.
            graphs = native_graphs + ragged_graphs
            logger.info(
                f"DSpark ragged verification: capturing {len(graphs)} graphs "
                f"({len(native_graphs)} native static-K{max_draft_len} graphs + "
                f"{len(ragged_graphs)} exact measured (G,V) cells)."
            )
            return graphs

        # Case 2: Static draft length
        # Match the runtime_draft_len semantics enforced in resolve_runtime_draft_len:
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

    def run_cuda_graph_warmup(self, resource_manager: ResourceManager) -> None:
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

                kv_cache_manager = resource_manager.get_resource_manager(
                    ResourceManagerType.KV_CACHE_MANAGER
                )
                context_only = (
                    getattr(kv_cache_manager, "context_swa_layer_limit", None) is not None
                )
                # Context-only workers only use the separate prefill capture.
                if not context_only:
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
                            (quant_method := getattr(module, "quant_method", None)),
                            MXFP8LinearMethod,
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
                    if (
                        is_warmup_only
                        and runtime_hooks_of(self.model).merge_tuned_tactics_across_ranks
                    ):
                        # Tactics tuned while warming the decode graphs agree across
                        # ranks before the capture pass reads them.
                        AutoTuner.get().post_tune_merge_tactics()
                    self._capture_additional_cuda_graphs(resource_manager)
        # Piecewise graphs have separate capture machinery and do not use the
        # whole-model attention workspace. Capture them only on the second pass.
        if not is_warmup_only:
            self._capture_prefill_cuda_graphs(resource_manager)

    def _capture_generation_cuda_graphs(self, resource_manager: ResourceManager) -> None:
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
        if self.mapping.has_cp_helix():
            effective_max_seq_len = self._config.max_seq_len // self.mapping.cp_size

        short_long_seq_len_threshold = self.cuda_graph_runner.short_long_seq_len_threshold
        if short_long_seq_len_threshold is not None:
            # For short sequences, subtract the maximum runtime tokens consumed
            # by a generation step so all current-step tokens stay within the
            # sequence length threshold. PARD uses 2K tokens here, not K+1.
            max_runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(
                self._config.max_draft_len
            )
            # For long sequences, use the default maximum sequence length.
            max_seq_len = short_long_seq_len_threshold - max_runtime_tokens_per_gen_step
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
            temperature_only: bool = False,
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
                for entry in graphs_to_capture:
                    bs, draft_len = entry[:2]
                    verify_bucket = entry[2] if len(entry) == 3 else None
                    if bs > self._config.max_batch_size:
                        continue

                    for max_seq_len in max_seq_len_list:
                        warmup_request = self._dummy_requests.create_cuda_graph_warmup_request(
                            resource_manager,
                            bs,
                            draft_len,
                            max_seq_len,
                            force_non_greedy=force_non_greedy,
                            temperature_only=temperature_only,
                        )
                        with self._dummy_requests.release_batch_context(
                            warmup_request, resource_manager
                        ) as batch:
                            if batch is not None and verify_bucket is not None:
                                self._set_warmup_ragged_windows(batch, verify_bucket, draft_len)
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
                            forward = (
                                self.forward_discarded_warmup
                                if self.cuda_graph_runner.is_warmup_only
                                else self._forward_warmup
                            )
                            forward(
                                batch,
                                resource_manager,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=draft_len,
                            )
                            torch.cuda.synchronize()
            finally:
                self._force_lora_graph_for_capture = None
                self.cuda_graph_runner.set_capture_sample_type(None)
                if self._inputs is not None and self._inputs._dspark_confidence_enabled:
                    self.cuda_graph_runner.agreed_ragged_bucket = None

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
        # The predicate is use_one_engine(): the drafter-based modes (NGram,
        # user-provided) get a plain TorchSampler from get_spec_decoder, so they
        # belong on the TorchSampler branch and do need the FAST graphs.
        uses_own_spec_decoder = (
            self._config.spec_config is not None
            and self._config.spec_config.spec_dec_mode.use_one_engine()
        )
        temperature_only_graphs = (
            uses_own_spec_decoder
            and self._metadata.spec_metadata is not None
            and self._metadata.spec_metadata.temperature_only_graphs
        )

        def _capture_variant(
            label: str,
            force_non_greedy: bool = False,
            sample_type: SampleType | None = None,
            temperature_only: bool = False,
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
                    temperature_only=temperature_only,
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
            if temperature_only_graphs:
                # Temperature-only batches replay the vocabulary-sharded Gumbel-max
                # target sampling (the advanced pass records the stock sampler: its
                # params enable top-k / top-p / min-p).
                _capture_variant(
                    "temperature-only sampling", force_non_greedy=True, temperature_only=True
                )
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
        if self._metadata.spec_metadata is not None:
            self._metadata.spec_metadata.is_all_greedy_sample = True
            self._metadata.spec_metadata.temperature_only = False

    def _capture_break_graphs(
        self,
        resource_manager: ResourceManager,
        num_tokens: int,
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
    ) -> None:
        """Capture the CUDA graphs of a prefill bucket's graphable breaks (see
        eager_on_graph) on warmup batches of each _BREAK_GRAPH_NUM_CONTEXTS
        request count, with and without in-forward Mamba snapshot splits when
        the cache manager plans them."""
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        variants = [contextlib.nullcontext]
        if (
            isinstance(kv_cache_manager, KVCacheManagerV2)
            and kv_cache_manager.inforward_snapshot_rows > 0
        ):
            variants.append(resolve_mamba_metadata_cls(self.model).inforward_split_for_warmup)
        for num_contexts in _BREAK_GRAPH_NUM_CONTEXTS:
            for variant in variants:
                warmup_request = self._dummy_requests.create_warmup_request(
                    resource_manager, num_tokens, 0, num_context_requests=num_contexts
                )
                with self._dummy_requests.release_batch_context(
                    warmup_request, resource_manager
                ) as batch:
                    self._agreement.assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
                    if batch is None:
                        continue
                    with variant():
                        self.breakable_cuda_graph_runner.capture_break_graphs(
                            lambda: self._forward_warmup(
                                batch,
                                resource_manager,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=runtime_draft_len,
                            ),
                            self._warmup_inputs,
                        )

    def _capture_prefill_cuda_graphs(self, resource_manager: ResourceManager) -> None:
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
        break_graph_bytes = 0
        with capture_context, self.no_cuda_graph():
            for num_tokens in prefill_cuda_graph_num_tokens:
                warmup_request = self._dummy_requests.create_warmup_request(
                    resource_manager, num_tokens, 0
                )
                with self._dummy_requests.release_batch_context(
                    warmup_request, resource_manager
                ) as batch:
                    self._agreement.assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
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
                                self._warmup_inputs,
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
                                self.forward_discarded_warmup(
                                    batch,
                                    resource_manager,
                                    enable_spec_decode=enable_spec_decode,
                                    runtime_draft_len=runtime_draft_len,
                                )
                            torch.cuda.synchronize()
                        with timing_metric("ctx_cuda_graph_capture_seconds", self._metrics):
                            self.forward_discarded_warmup(
                                batch,
                                resource_manager,
                                enable_spec_decode=enable_spec_decode,
                                runtime_draft_len=runtime_draft_len,
                            )
                            torch.cuda.synchronize()
                # Break graphs remove host time, which dominates only the
                # shorter prefill iterations.
                if (
                    self.breakable_cuda_graph_runner is not None
                    and num_tokens <= self._config.max_num_tokens // 4
                    and self.breakable_cuda_graph_runner.has_graphable_breaks(num_tokens)
                ):
                    free_bytes = torch.cuda.mem_get_info()[0]
                    with timing_metric("ctx_cuda_graph_capture_seconds", self._metrics):
                        self._capture_break_graphs(
                            resource_manager,
                            num_tokens,
                            enable_spec_decode=enable_spec_decode,
                            runtime_draft_len=runtime_draft_len,
                        )
                    break_graph_bytes += free_bytes - torch.cuda.mem_get_info()[0]
        if self.breakable_cuda_graph_runner is not None:
            logger.info(
                f"Captured {self.breakable_cuda_graph_runner.num_break_graphs} breakable "
                f"prefill graph break graphs, {break_graph_bytes / 2**20:.0f} MiB"
            )

        # The logits allocations grow with the number of requests and are not
        # part of the captured model body. Warm up the largest request count so
        # those allocations can be reused during stable inference.
        with timing_metric("post_ctx_cuda_graph_capture_warmup_seconds", self._metrics):
            for num_tokens in prefill_cuda_graph_num_tokens:
                warmup_request = self._dummy_requests.create_warmup_request(
                    resource_manager, num_tokens, 0, least_requests=False
                )
                with self._dummy_requests.release_batch_context(
                    warmup_request, resource_manager
                ) as batch:
                    self._agreement.assert_all_tp_ranks_have_warmup_batch(batch, num_tokens)
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
                                self._warmup_inputs,
                                steps=1,
                            )
                    else:
                        self.forward_discarded_warmup(
                            batch,
                            resource_manager,
                            enable_spec_decode=enable_spec_decode,
                            runtime_draft_len=runtime_draft_len,
                        )
                    torch.cuda.synchronize()

    def release(self, *, release_nccl_window_owners: bool = True) -> None:
        self.cuda_graph_runner.clear(release_nccl_window_owners=release_nccl_window_owners)
        if self.breakable_cuda_graph_runner is not None:
            self.breakable_cuda_graph_runner.clear(
                release_nccl_window_owners=release_nccl_window_owners
            )

    def _set_warmup_ragged_windows(self, batch, verify_bucket: int, draft_len: int) -> None:
        """Give a capture batch per-request windows summing to ``verify_bucket``.

        The warmup batch must hit the bucket total exactly, or the graph is
        keyed under a total no runtime batch produces and every ragged step
        silently falls back to eager. Any split with the right sum captures
        the right shape (the raggedness lives in the contents of qo_indptr).
        """
        requests = batch.generation_requests
        if not requests:
            return
        from tensorrt_llm._torch.speculative.ragged_helpers import RaggedVerifyLayout

        lens = torch.ones(len(requests), dtype=torch.int32)
        layout = RaggedVerifyLayout.from_verify_lens(
            lens, graph_num_tokens=int(verify_bucket), total_verify_tokens=len(requests)
        )
        filled = layout.fill_bucket(max_verify_len=1 + int(draft_len))
        for request, tokens in zip(requests, filled.verify_lens.tolist()):
            request.py_verify_len = int(tokens) - 1
        # Publish to the runner too: the graph key carries this value, and the
        # runtime fit never runs during capture.
        self.cuda_graph_runner.agreed_ragged_bucket = int(verify_bucket)
