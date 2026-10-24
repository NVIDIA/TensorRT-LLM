# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoder input preparation state."""

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any

import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.kernels.spec_step_copies import op as spec_step_copies
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.models.modeling_multimodal_mixin import _build_request_multimodal_input
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import get_mrope_dummy_seq_slot
from tensorrt_llm._torch.pyexecutor.guided_decoder import CapturableGuidedDecoder
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.resource_manager import (
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTensors
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import RemoteTailPhase
from tensorrt_llm._torch.speculative import SpecMetadata, get_num_extra_kv_tokens
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata
from tensorrt_llm._torch.speculative.interface import INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec
from tensorrt_llm._torch.speculative.utils import get_static_draft_len
from tensorrt_llm._torch.utils import helix_local_len, set_per_request_prefill_cuda_graph_flag
from tensorrt_llm._utils import maybe_pin_memory, nvtx_range, prefer_pinned
from tensorrt_llm.inputs.multimodal import (
    MultimodalParams,
    MultimodalRuntimeData,
    check_mm_embed_cumsum_if_needed,
    strip_mm_data_for_generation,
)
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, PrefillCudaGraphBackend
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType, Mapping

from ...input_buffers import InputBuffers
from ...lora import LoraParamBuilder
from ...multimodal import MultimodalItemScheduler
from ..common import (
    apply_position_id_offset,
    get_all_rank_num_tokens,
    get_position_id_offset,
    pad_input_tokens,
    prepare_multimodal_indices,
    resolve_mrope_position_deltas_cache,
    ship_multimodal_indices,
    token_model_inputs,
)
from ..interface import PreparedInputs
from .config import DecoderRunnerConfig
from .metadata import set_spec_metadata_all_rank_num_tokens


def resolve_runtime_draft_len(
    spec_config: DecodingBaseConfig | None, runtime_draft_len: int
) -> int:
    """Return the draft length a decoder step prepares and runs with.

    For tree decoding it matches the total tree tokens, not the tree depth;
    py_executor resets it every iteration.
    """
    if spec_config is not None and not spec_config.is_linear_tree:
        return get_static_draft_len(spec_config)
    return runtime_draft_len


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


def _get_context_prompt_lookahead_token(request: LlmRequest, chunk_end: int) -> int:
    """Prompt token immediately following a context chunk. Uses the live C++
    ``mPromptLen``; ``py_prompt_len`` goes stale after a non-recompute preemption.
    """
    if request.is_last_context_chunk:
        return INVALID_PROMPT_LOOKAHEAD_TOKEN
    return request.get_token(0, chunk_end)


def _zero_fill_row_spans(buffer: torch.Tensor, spans: list[tuple[int, int]]) -> None:
    """Zero ``buffer`` over half-open row spans, merging adjacent spans.

    Used for the input-id rows of CUDA-graph and attention-DP padding requests,
    which neither the host nor the device input-id copies write.
    """
    if not spans:
        return
    rows = sorted(spans)
    merged_begin, merged_end = rows[0]
    for row_begin, row_end in rows[1:]:
        if row_begin <= merged_end:
            merged_end = max(merged_end, row_end)
            continue
        buffer[merged_begin:merged_end].fill_(0)
        merged_begin, merged_end = row_begin, row_end
    buffer[merged_begin:merged_end].fill_(0)


@dataclass(frozen=True, kw_only=True)
class DecoderInputBuffers(InputBuffers):
    """Persistent buffers read by decoder execution.

    CUDA graphs record these addresses, so they are allocated once and only
    ever sliced or written in place.
    """

    previous_batch_indices_cuda: torch.Tensor
    # Host counter of the steady-state generation prepare cache.
    steady_gen_positions_pinned: torch.Tensor
    previous_pos_indices_cuda: torch.Tensor | None = None
    previous_pos_id_offsets_cuda: torch.Tensor | None = None
    previous_kv_lens_offsets_cuda: torch.Tensor | None = None
    mrope_position_ids_cuda: torch.Tensor | None = None

    @classmethod
    def for_config(
        cls, config: DecoderRunnerConfig, use_beam_search: bool
    ) -> "DecoderInputBuffers":
        """Allocate the buffers ``config`` needs before warmup or graph capture."""
        base = InputBuffers.allocate(
            max_num_tokens=config.max_num_tokens,
            max_batch_size=config.max_batch_size,
            max_beam_width=config.max_beam_width,
            max_seq_len=config.max_seq_len,
            use_cache_indirection=use_beam_search,
            max_num_draft_tokens=(
                config.max_draft_loop_tokens * config.max_batch_size
                if config.is_spec_decode
                else None
            ),
        )
        spec = config.is_spec_decode
        return cls(
            **{f.name: getattr(base, f.name) for f in fields(InputBuffers)},
            previous_batch_indices_cuda=torch.empty(
                (config.max_num_tokens,), dtype=torch.int, device="cuda"
            ),
            steady_gen_positions_pinned=torch.empty(
                (config.max_num_tokens,), dtype=torch.int, pin_memory=prefer_pinned()
            ),
            previous_pos_indices_cuda=(
                torch.empty((config.max_num_tokens,), dtype=torch.int, device="cuda")
                if spec
                else None
            ),
            previous_pos_id_offsets_cuda=(
                torch.zeros((config.max_num_tokens,), dtype=torch.int, device="cuda")
                if spec
                else None
            ),
            previous_kv_lens_offsets_cuda=(
                torch.zeros((config.max_batch_size,), dtype=torch.int, device="cuda")
                if spec
                else None
            ),
            mrope_position_ids_cuda=(
                torch.empty((3, 1, config.max_num_tokens), dtype=torch.int, device="cuda")
                if config.use_mrope
                else None
            ),
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


@dataclass(frozen=True, slots=True)
class HostLists:
    """Per-sequence and per-request lists of one full prepare pass.

    Stages fill them in place; the dataclass is frozen so none can be rebound
    away from the pass.
    """

    # if new_tensors_device exist, input_ids will only contain new context tokens
    input_ids: list[int] = field(default_factory=list)  # per sequence
    sequence_lengths: list[int] = field(default_factory=list)  # per sequence
    prompt_lengths: list[int] = field(default_factory=list)  # per sequence
    request_ids: list[int] = field(default_factory=list)  # per request
    gather_ids: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)  # per sequence
    num_cached_tokens_per_seq: list[int] = field(default_factory=list)  # per sequence
    draft_tokens: list[int] = field(default_factory=list)
    draft_lens: list[int] = field(default_factory=list)
    gen_request_seq_slots: list[int] = field(default_factory=list)  # per generation request
    # One-model rejection: slots of gen requests that produced 0 real draft
    # tokens this step (marked in _handle_dynamic_draft_len); their stale
    # draft_probs rows are one-hot'd after spec_metadata.prepare().
    padding_gen_slots: list[int] = field(default_factory=list)
    multimodal_params_list: list[MultimodalParams] = field(default_factory=list)
    # (start_idx, end_idx, (3,1,L) mrope_pos_ids) per multimodal request
    mrope_position_ids: list[tuple[int, int, torch.Tensor]] = field(default_factory=list)
    mrope_delta_write_seq_slots: list[int] = field(default_factory=list)
    mrope_delta_read_seq_slots: list[int] = field(default_factory=list)
    # will contain previous batch indices of generation requests
    previous_batch_indices: list[int] = field(default_factory=list)
    previous_pos_indices: list[int] = field(default_factory=list)
    # Half-open token-row spans of padding requests (CUDA-graph and
    # attention-DP dummies) whose input ids are not appended to the host
    # ``input_ids`` list; zero-filled after the input-id copies.
    unwritten_input_id_rows: list[tuple[int, int]] = field(default_factory=list)

    extend_requests: list[LlmRequest] = field(default_factory=list)
    extend_dummy_requests: list[LlmRequest] = field(default_factory=list)
    generation_requests: list[LlmRequest] = field(default_factory=list)
    # Collect generation request IDs during categorization to avoid
    # a separate iteration over scheduled_requests.generation_requests later.
    all_gen_request_ids: list[int] = field(default_factory=list)

    helix_is_inactive_rank: list[bool] = field(default_factory=list)
    helix_position_offsets: list[int] = field(default_factory=list)
    helix_owned_new_tokens: list[int] = field(default_factory=list)


@dataclass(slots=True)
class HostInputs:
    """Host-side inputs of one full prepare pass, built up stage by stage.

    Stages bind the fields they use to locals, since the per-request loops are
    hot, and write back only the scalars they rebind.
    """

    use_mrope: bool
    # Extra model-side cache slot reserved for CUDA graph / warmup dummy
    # requests, whose outputs are discarded, and for generation requests
    # that carry no MRoPE metadata at all. The cache is zero-initialized and
    # the write path only ever targets real ``py_seq_slot``s, so this slot
    # permanently reads back a zero delta.
    mrope_dummy_seq_slot: int
    extra_inputs: ExtraInputsCollector
    context_prompt_lookahead: list[int] | None
    lists: HostLists = field(default_factory=HostLists)

    num_ctx_requests: int = 0
    num_ctx_tokens: int = 0
    text_token_indices_ctx: Any = None
    mm_token_indices: Any = None

    has_cp_helix: bool = False
    helix_local_len_host: Callable[[int], int] | None = None
    helix_pack_extend: Callable[[LlmRequest, int], int] | None = None

    n_gen: int = 0
    has_any_multimodal_request: bool = False
    # Whether a verify group or generation request of this pass carries real
    # MRoPE metadata; without one, the delta read slots are dropped.
    has_gen_mrope_delta: bool = False
    beam_width: int = 1

    num_tokens: int = 0
    total_num_tokens: int = 0
    previous_batch_len: int = 0
    host_position_ids: torch.Tensor | None = None
    num_cached_tokens_snapshot: list[int] | None = None
    remote_tail_modes: set[Any] = field(default_factory=set)
    spec_all_rank_counts: list[list[int]] | None = None
    virtual_num_tokens: int = 0


class DecoderInputPreparer:
    """Prepares decoder model inputs into the persistent input buffers.

    A model family with its own per-sequence inputs or a specialized whole-batch
    path overrides ``_new_extra_inputs`` and ``_prepare_inputs_fast_path``.
    """

    def __init__(
        self,
        *,
        buffers: DecoderInputBuffers,
        config: DecoderRunnerConfig,
        mapping: Mapping,
        dist: Distributed | None,
        model: torch.nn.Module,
        get_runtime_tokens_per_gen_step: Callable[[int], int],
        iter_states: dict[str, Any],
        lora: LoraParamBuilder,
        use_beam_search: bool,
        steady_gen_cache_supported: bool,
        mm_item_scheduler: MultimodalItemScheduler | None = None,
    ) -> None:
        self._buffers = buffers
        self._mm_item_scheduler = mm_item_scheduler
        self._config = config
        self.mapping = mapping
        self.dist = dist
        self.model = model
        self.get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step
        self.iter_states = iter_states
        self._lora = lora
        self.use_beam_search = use_beam_search
        self._steady_gen_cache_supported = steady_gen_cache_supported
        self.guided_decoder: CapturableGuidedDecoder | None = None
        self._prepare_inputs_event: torch.cuda.Event | None = None
        self._gather_ids_buffer = (
            DeviceCopyBuffer(buffers.gather_ids_cuda, pin_memory=prefer_pinned())
            if config.is_spec_decode
            else None
        )
        # Reuse device indices while the scheduled batch stays unchanged.
        self._staged_previous_batch_indices: list[int] | None = None
        self._staged_previous_pos_indices: list[int] | None = None
        self._step_input_gather: spec_step_copies.StepInputGather | None = (
            spec_step_copies.StepInputGather()
            if config.is_spec_decode and spec_step_copies.is_supported()
            else None
        )
        # Steady-state generation-only prepare cache (non-speculative overlap
        # decode). Holds the per-request lists that are invariant while the
        # scheduled generation batch keeps the same composition, plus a pinned
        # cached-token counter advanced by one per step (host-side bookkeeping
        # only; the device position buffer is advanced in place and this
        # buffer is never the source of an async H2D). Invalidated (set to
        # None) by every full prepare_inputs pass.
        self._steady_gen_cache: dict[str, Any] | None = None
        # Log cached prefixes in model-input sequence order when enabled.
        self.log_cached_kv_tokens_per_req = (
            os.getenv("TLLM_LOG_CACHED_KV_TOKENS_PER_REQ", "0") == "1"
        )

    def _new_extra_inputs(self) -> ExtraInputsCollector:
        """Return the collector of this pass's model inputs beyond the decoder's own."""
        return _NO_EXTRA_INPUTS

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

    def pad_batch_seed_mrope_delta_cache(self, padded_requests: ScheduledRequests) -> None:
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

    def _should_use_full_generation_page_table(
        self, spec_config: DecodingBaseConfig | None, attn_metadata: AttentionMetadata
    ) -> bool:
        """Return whether overlap decode needs every reserved generation page."""
        return uses_full_generation_page_table(
            self._config.disable_overlap_scheduler, spec_config, attn_metadata
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
        counts = attn_metadata.get_adp_token_counts()
        if self.mapping.has_cp_helix():
            counts = [math.ceil(count / self.mapping.cp_size) for count in counts]
        gathered = self.dist.tp_cp_allgather_int64([*counts, *spec_counts])
        cols = gathered.T.tolist()
        return (attn_metadata.set_adp_token_counts(cols[: len(counts)]), cols[len(counts) :])

    def record_cached_kv_tokens_per_req(
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

    def pad_tokens(
        self,
        attn_metadata: AttentionMetadata,
        num_tokens: int,
        num_contexts: int,
        attn_all_rank_num_tokens: list[int] | None = None,
        *,
        allow_prefill_cuda_graph: bool = True,
    ) -> int:
        """Pad the batch to the token count its graph or attention-DP peers need.

        Zeroes the padded input rows and returns the token count the model runs on.
        """
        if attn_all_rank_num_tokens is None:
            attn_all_rank_num_tokens = get_all_rank_num_tokens(
                attn_metadata,
                enable_attention_dp=self._config.enable_attention_dp,
                mapping=self.mapping,
                dist=self.dist,
            )
        return pad_input_tokens(
            attn_metadata,
            num_tokens,
            num_contexts,
            attn_all_rank_num_tokens,
            input_ids=self._buffers.input_ids_cuda,
            position_ids=self._position_ids_buffer(),
            dist=self.dist,
            enable_attention_dp=self._config.enable_attention_dp,
            prefill_cuda_graph_backend=(
                self._config.prefill_cuda_graph_backend
                if allow_prefill_cuda_graph
                else PrefillCudaGraphBackend.DISABLED
            ),
            prefill_cuda_graph_num_tokens=self._config.prefill_cuda_graph_num_tokens,
        )

    def model_inputs(
        self,
        attn_metadata: AttentionMetadata,
        resource_manager: ResourceManager | None,
        num_tokens: int,
        multimodal_params: list[MultimodalParams],
    ) -> dict[str, Any]:
        """Return the model inputs over the first ``num_tokens`` buffer rows."""
        return token_model_inputs(
            attn_metadata,
            resource_manager,
            input_ids=self._buffers.input_ids_cuda,
            position_ids=self._position_ids_buffer(),
            num_tokens=num_tokens,
            multimodal_params=multimodal_params,
        )

    def _position_ids_buffer(self) -> torch.Tensor:
        # MRoPE models always use the (3,1,N) layout.
        if self._config.use_mrope:
            return self._buffers.mrope_position_ids_cuda
        return self._buffers.position_ids_cuda

    def _can_use_steady_gen_fast_prepare(
        self,
        scheduled_requests: ScheduledRequests,
        new_tokens_device: torch.Tensor | None,
        next_draft_tokens_device: torch.Tensor | None,
        spec_metadata: SpecMetadata | None,
        is_dummy: bool,
    ) -> bool:
        """Check whether the cached steady-state generation prepare applies.

        The cache is only recorded by a full prepare_inputs pass whose
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
    ) -> PreparedInputs:
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
        # because only prepare_inputs writes it and the cache validity
        # invariant guarantees the previous pass wrote these same rows. This
        # avoids reusing a mutated pinned buffer as the source of an async
        # H2D whose previous-step copy may still be pending under the overlap
        # scheduler (the nvbug 6293536 hazard class; see
        # KVCacheManager._stage_block_offsets_for_copy). The pinned buffer is
        # host-side bookkeeping only.
        use_mrope = cache["use_mrope"]
        positions = self._buffers.steady_gen_positions_pinned[:num_requests]
        positions.add_(1)
        if use_mrope:
            # Text-only batch on an mrope model: the recording pass broadcast
            # the scalar positions onto all three axes of the (3,1,N) buffer,
            # which is what the model (and any captured CUDA graph) reads, so
            # advance it in place. position_ids_cuda is reseeded by the next
            # full pass.
            self._buffers.mrope_position_ids_cuda[:, :, :num_requests].add_(1)
        else:
            self._buffers.position_ids_cuda[:num_requests].add_(1)
        num_cached_tokens_per_seq = positions.tolist()

        # Gather this step's input tokens from the previous iteration's device
        # sample buffer; the seq-slot indices in previous_batch_indices_cuda
        # are unchanged since the last full pass.
        previous_slots = self._buffers.previous_batch_indices_cuda[:num_requests]
        torch.index_select(
            new_tensors_device.new_tokens[0, :, : self._config.max_beam_width],
            0,
            previous_slots,
            out=self._buffers.input_ids_cuda[: num_requests * self._config.max_beam_width].view(
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

        virtual_num_tokens = self.pad_tokens(attn_metadata, num_requests, 0)

        self.iter_states["num_ctx_requests"] = 0
        self.iter_states["num_ctx_tokens"] = 0
        self.iter_states["num_generation_tokens"] = num_requests
        self.iter_states["cached_kv_tokens"] = sum(num_cached_tokens_per_seq)
        if self.log_cached_kv_tokens_per_req:
            self.record_cached_kv_tokens_per_req(num_cached_tokens_per_seq)

        return PreparedInputs(
            self.model_inputs(attn_metadata, resource_manager, virtual_num_tokens, [])
        )

    def _build_multimodal_data_for_llm(
        self,
        request: LlmRequest,
        runtime: MultimodalRuntimeData | None = None,
    ) -> dict[str, Any] | None:
        """Attach cached item outputs when item scheduling owns the request."""
        if self._mm_item_scheduler is None:
            return request.py_multimodal_data
        return self._mm_item_scheduler.build_multimodal_data_for_llm(request, runtime)

    @nvtx_range("_prepare_inputs")
    def prepare_inputs(
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
        gather_context_logits: bool = False,
    ) -> PreparedInputs:
        """Prepare decoder model inputs for one step.

        A fast path reuses earlier state when the batch allows it. Otherwise
        every request is packed on the host, copied into the persistent
        buffers, and the attention and speculative metadata are prepared.
        """
        set_per_request_prefill_cuda_graph_flag(False)
        if self.mapping is not None and "cp_type" in self.mapping.cp_config:
            cp_type = self.mapping.cp_config["cp_type"]
            if cp_type in (CpType.HELIX, CpType.ULYSSES):
                # Prepared like any other batch.
                pass
            else:
                raise NotImplementedError(
                    f"Unsupported cp_type {getattr(cp_type, 'name', cp_type)}."
                )

        self._register_suffix_automaton_requests(scheduled_requests, resource_manager)

        # Outside capture, before any steady-input early return.
        prepare_request_inputs = getattr(self.model, "prepare_request_inputs", None)
        if prepare_request_inputs is not None:
            prepare_request_inputs(scheduled_requests, attn_metadata, promoted_context_request_ids)

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

        spec_config = self._config.spec_config if enable_spec_decode else None
        runtime_draft_len = resolve_runtime_draft_len(spec_config, runtime_draft_len)

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
            return self._finish_prepared(
                fast_inputs, scheduled_requests, kv_cache_manager, runtime_draft_len
            )

        if not promoted_context_request_ids and self._can_use_steady_gen_fast_prepare(
            scheduled_requests, new_tokens_device, next_draft_tokens_device, spec_metadata, is_dummy
        ):
            return self._finish_prepared(
                self._apply_steady_gen_fast_prepare(
                    kv_cache_manager, attn_metadata, new_tensors_device, resource_manager
                ),
                scheduled_requests,
                kv_cache_manager,
                runtime_draft_len,
            )
        # Any full pass invalidates the steady-state cache; it is re-recorded
        # at the end of this pass when the batch qualifies.
        self._steady_gen_cache = None

        host = HostInputs(
            use_mrope=self._config.use_mrope,
            mrope_dummy_seq_slot=get_mrope_dummy_seq_slot(
                self._config.max_num_tokens, self.mapping.pp_size
            ),
            extra_inputs=self._new_extra_inputs(),
            context_prompt_lookahead=(
                []
                if spec_metadata is not None
                and spec_metadata.context_prompt_lookahead_tokens is not None
                else None
            ),
        )
        self._pack_context_requests(host, scheduled_requests)
        self._classify_generation_requests(
            host,
            scheduled_requests,
            promoted_context_request_ids,
            next_draft_tokens_device,
            enable_spec_decode,
        )
        self._prepare_helix_state(host, kv_cache_manager)

        if not self._config.disable_overlap_scheduler and spec_config is not None:
            assert spec_config.spec_dec_mode.support_overlap_scheduler(), (
                f"{spec_config.decoding_type} does not support overlap scheduler"
            )

        runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(runtime_draft_len)
        self._pack_extend_requests(
            host,
            promoted_context_request_ids,
            next_draft_tokens_device,
            runtime_tokens_per_gen_step,
            spec_config,
            enable_spec_decode=enable_spec_decode,
        )
        self._pack_generation_requests(host, promoted_context_request_ids, new_tokens_device)

        self._copy_token_ids_to_device(
            host,
            scheduled_requests,
            new_tokens_device,
            new_tokens_lens_device,
            next_draft_tokens_device,
            runtime_draft_len,
            attn_metadata=attn_metadata,
            enable_spec_decode=enable_spec_decode,
        )
        self._copy_position_ids_to_device(host)

        if enable_spec_decode:
            self._gather_ids_buffer.update(torch.tensor(host.lists.gather_ids, dtype=torch.int))

        self._prepare_attn_metadata(
            host,
            scheduled_requests,
            kv_cache_manager,
            attn_metadata,
            cache_indirection_buffer,
            spec_config,
            enable_spec_decode=enable_spec_decode,
            is_dummy=is_dummy,
        )
        extra_model_inputs = host.extra_inputs.build(attn_metadata, resource_manager)

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

        self._pad_inputs(
            host,
            scheduled_requests,
            attn_metadata,
            spec_metadata,
            maybe_graph,
            gather_context_logits,
        )
        inputs = self._build_inputs(
            host,
            scheduled_requests,
            attn_metadata,
            spec_metadata,
            resource_manager,
            extra_model_inputs,
            lora_params,
            runtime_draft_len,
        )
        self._record_iter_states(host, scheduled_requests)
        if not is_dummy:
            self._record_steady_gen_cache(
                host, kv_cache_manager, attn_metadata, spec_metadata, new_tokens_device, lora_params
            )

        gather_ids_device = (
            self._gather_ids_buffer.tensor[: len(host.lists.gather_ids)]
            if enable_spec_decode
            else None
        )
        return self._finish_prepared(
            PreparedInputs(inputs, gather_ids_device),
            scheduled_requests,
            kv_cache_manager,
            runtime_draft_len,
        )

    def _finish_prepared(
        self,
        prepared: PreparedInputs,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        runtime_draft_len: int,
    ) -> PreparedInputs:
        """Record the step's draft length, and pass the context requests to models
        that select rows or replay encoder tokens."""
        from tensorrt_llm._torch.pyexecutor.ced_replay import (
            encoder_replay_tokens,
            uses_private_decoder_cache,
        )

        if scheduled_requests.num_context_requests and (
            (
                self._config.enable_attention_dp
                and getattr(self.model, "prepare_adp_inputs", None) is not None
            )
            or uses_private_decoder_cache(getattr(self.model, "model", None), kv_cache_manager)
            or any(encoder_replay_tokens(req) for req in scheduled_requests.context_requests)
        ):
            prepared.kwargs["context_requests"] = scheduled_requests.context_requests
        return replace(prepared, runtime_draft_len=runtime_draft_len)

    def _register_suffix_automaton_requests(
        self, scheduled_requests: ScheduledRequests, resource_manager: ResourceManager | None
    ) -> None:
        """Initialize SA state for new requests (MTP+SA, EAGLE3+SA, PARD+SA, etc.)."""
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

    def _pack_context_requests(
        self, host: HostInputs, scheduled_requests: ScheduledRequests
    ) -> None:
        """Pack the current chunk of every context request."""
        input_ids = host.lists.input_ids
        sequence_lengths = host.lists.sequence_lengths
        prompt_lengths = host.lists.prompt_lengths
        request_ids = host.lists.request_ids
        gather_ids = host.lists.gather_ids
        position_ids = host.lists.position_ids
        num_cached_tokens_per_seq = host.lists.num_cached_tokens_per_seq
        draft_lens = host.lists.draft_lens
        multimodal_params_list = host.lists.multimodal_params_list
        extra_inputs = host.extra_inputs
        context_prompt_lookahead = host.context_prompt_lookahead

        for request in scheduled_requests.context_requests:
            request_ids.append(request.py_request_id)
            draft_lens.append(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            if getattr(request, "py_ced_replay", None) is not None:
                from tensorrt_llm._torch.pyexecutor.ced_replay import encoder_replay_tokens

                begin_compute -= encoder_replay_tokens(request)
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
            prompt_lengths.append(len(prompt_tokens))
            past_seen_token_num = begin_compute
            num_cached_tokens_per_seq.append(past_seen_token_num - request.py_num_compressed_tokens)
            request.cached_tokens = request.context_current_position
            extra_inputs.add_context_request(request)

            self._pack_context_multimodal(
                host,
                request,
                begin_compute,
                end_compute,
                past_seen_token_num,
                prompt_tokens,
                context_start_idx,
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

        host.num_ctx_requests = num_ctx_requests
        host.num_ctx_tokens = num_ctx_tokens
        host.text_token_indices_ctx = text_token_indices_ctx
        host.mm_token_indices = mm_token_indices

    def _pack_context_multimodal(
        self,
        host: HostInputs,
        request: LlmRequest,
        begin_compute: int,
        end_compute: int,
        past_seen_token_num: int,
        prompt_tokens: list[int],
        context_start_idx: int,
    ) -> None:
        """Attach a context request's multimodal inputs and MRoPE positions."""
        _use_mrope = host.use_mrope
        position_ids = host.lists.position_ids
        mrope_position_ids = host.lists.mrope_position_ids
        mrope_delta_write_seq_slots = host.lists.mrope_delta_write_seq_slots
        multimodal_params_list = host.lists.multimodal_params_list

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
            multimodal_data=self._build_multimodal_data_for_llm(request, py_multimodal_runtime),
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

            # Item scheduling builds a per-forward prompt-window copy with this chunk's
            # joined rows. Keep it off the request so chunked prefill does not retain rows
            # outside the encoder-cache budget, and keeps its full CPU payload for later
            # items and a restarted prefill. Legacy requests keep their generation payload.
            if request.py_mm_encoder_state is None:
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

    def _classify_generation_requests(
        self,
        host: HostInputs,
        scheduled_requests: ScheduledRequests,
        promoted_context_request_ids: frozenset[int],
        next_draft_tokens_device: torch.Tensor | None,
        enable_spec_decode: bool,
    ) -> None:
        """Sort generation requests into the extend and generation groups."""
        extend_requests = host.lists.extend_requests
        extend_dummy_requests = host.lists.extend_dummy_requests
        generation_requests = host.lists.generation_requests
        all_gen_request_ids = host.lists.all_gen_request_ids

        # Requests with draft tokens are treated like extend requests. Dummy extend requests should be
        # at the end of extend_requests.
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
            else:
                generation_requests.append(request)
        extend_requests += extend_dummy_requests

    def _prepare_helix_state(
        self, host: HostInputs, kv_cache_manager: KVCacheManager | KVCacheManagerV2 | None
    ) -> None:
        # Helix bookkeeping is needed by BOTH the extend (speculative verify
        # group) and the plain generation packing loops below, so initialize
        # it ahead of them. Positions are global; KV ownership follows the
        # round-robin ledger (page b -> rank b % cp); the host-side
        # provisional packing values come from the one shared definition in
        # _torch.utils.helix_local_len.
        helix_is_inactive_rank = host.lists.helix_is_inactive_rank
        helix_position_offsets = host.lists.helix_position_offsets
        position_ids = host.lists.position_ids
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

            host.helix_local_len_host = _helix_local_len_host
            host.helix_pack_extend = _helix_pack_extend
        host.has_cp_helix = _has_cp_helix

    @staticmethod
    def _decode_mrope_position_delta(request: LlmRequest) -> torch.Tensor | None:
        """The MRoPE position delta of a decode-phase request, or None.

        A request that ran its context phase on this worker already holds the
        delta in ``py_mrope_position_delta``. Otherwise (for example the first
        step of a generation-only request) it is read from the request's
        ``mrope_config``, moved to the device once, and cached on the request.
        None means the request carries no MRoPE metadata (a text-only prompt),
        whose delta is zero.
        """
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
        return mrope_position_delta

    def _pack_extend_requests(
        self,
        host: HostInputs,
        promoted_context_request_ids: frozenset[int],
        next_draft_tokens_device: torch.Tensor | None,
        runtime_tokens_per_gen_step: int,
        spec_config: DecodingBaseConfig | None,
        *,
        enable_spec_decode: bool,
    ) -> None:
        """Pack requests that verify draft tokens or take a promoted context row."""
        extend_requests = host.lists.extend_requests
        padding_gen_slots = host.lists.padding_gen_slots
        request_ids = host.lists.request_ids
        input_ids = host.lists.input_ids
        draft_tokens = host.lists.draft_tokens
        draft_lens = host.lists.draft_lens
        prompt_lengths = host.lists.prompt_lengths
        sequence_lengths = host.lists.sequence_lengths
        gather_ids = host.lists.gather_ids
        position_ids = host.lists.position_ids
        num_cached_tokens_per_seq = host.lists.num_cached_tokens_per_seq
        previous_batch_indices = host.lists.previous_batch_indices
        previous_pos_indices = host.lists.previous_pos_indices
        unwritten_input_id_rows = host.lists.unwritten_input_id_rows
        helix_owned_new_tokens = host.lists.helix_owned_new_tokens
        extra_inputs = host.extra_inputs
        _has_cp_helix = host.has_cp_helix
        _helix_local_len_host = host.helix_local_len_host
        _helix_pack_extend = host.helix_pack_extend
        mrope_position_ids = host.lists.mrope_position_ids
        mrope_delta_write_seq_slots = host.lists.mrope_delta_write_seq_slots
        mrope_delta_read_seq_slots = host.lists.mrope_delta_read_seq_slots
        multimodal_params_list = host.lists.multimodal_params_list
        _use_mrope = host.use_mrope
        mrope_dummy_seq_slot = host.mrope_dummy_seq_slot
        has_gen_mrope_delta = host.has_gen_mrope_delta

        runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
        # Under extend_ctx the attention kernel runs extend requests as chunked
        # context rows: their rotary comes from position_ids alone, and they take
        # no entry in the generation-indexed MRoPE delta read slots.
        extend_requests_are_context = (
            _use_mrope
            and enable_spec_decode
            and spec_config.spec_dec_mode.extend_ctx(self._config.attention_backend)
            and spec_config.is_linear_tree
        )
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
                gather_ids.extend(
                    list(range(len(position_ids), len(position_ids) + 1 + num_draft_tokens))
                )
                position_ids.extend(
                    list(range(past_seen_token_num, past_seen_token_num + 1 + num_draft_tokens))
                )
                if request.is_attention_dp_dummy or request.is_cuda_graph_dummy:
                    # One position per token row, in batch order, so the rows
                    # this request owns are the position_ids it just added.
                    unwritten_input_id_rows.append(
                        (len(position_ids) - (1 + num_draft_tokens), len(position_ids))
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

            if _use_mrope:
                mrope_position_delta = self._decode_mrope_position_delta(request)
                if mrope_position_delta is not None:
                    # Every row of a verify group lies past the prompt, so it is
                    # text: its three MRoPE coordinates are equal, the scalar
                    # position plus the request's delta. The overlap scheduler's
                    # accepted-token correction later adds to all three axes.
                    num_request_tokens = sequence_lengths[-1]
                    token_start = len(position_ids) - num_request_tokens
                    first_position = position_ids[token_start]
                    extend_mrope_position_ids = (
                        torch.arange(
                            first_position,
                            first_position + num_request_tokens,
                            dtype=torch.int32,
                            device=mrope_position_delta.device,
                        )
                        + mrope_position_delta.reshape(1, 1, 1)
                    ).expand(3, 1, num_request_tokens)
                    mrope_position_ids.append(
                        (token_start, len(position_ids), extend_mrope_position_ids)
                    )
                    if (
                        request.py_seq_slot is not None
                        and not request.is_dummy
                        and getattr(request, "py_mrope_delta_cache_slot", None)
                        != request.py_seq_slot
                    ):
                        multimodal_params_list.append(
                            MultimodalParams(
                                multimodal_data={
                                    "mrope_config": {"mrope_position_deltas": mrope_position_delta}
                                }
                            )
                        )
                        mrope_delta_write_seq_slots.append(request.py_seq_slot)
                        request.py_mrope_delta_cache_slot = request.py_seq_slot
                if not extend_requests_are_context:
                    # The kernel indexes deltas by generation request, so every
                    # verify group takes one read slot, in batch order; a dummy,
                    # unslotted, or text-only request reads the zero slot.
                    if mrope_position_delta is not None:
                        has_gen_mrope_delta = True
                    mrope_delta_read_seq_slots.append(
                        request.py_seq_slot
                        if mrope_position_delta is not None
                        and not request.is_dummy
                        and request.py_seq_slot is not None
                        else mrope_dummy_seq_slot
                    )

            extra_inputs.add_generation_request(request)

        host.has_gen_mrope_delta = has_gen_mrope_delta

    def _pack_generation_requests(
        self,
        host: HostInputs,
        promoted_context_request_ids: frozenset[int],
        new_tokens_device: torch.Tensor | None,
    ) -> None:
        """Pack one token per beam for every plain generation request."""
        generation_requests = host.lists.generation_requests
        draft_lens = host.lists.draft_lens
        sequence_lengths = host.lists.sequence_lengths
        request_ids = host.lists.request_ids
        input_ids = host.lists.input_ids
        previous_batch_indices = host.lists.previous_batch_indices
        helix_is_inactive_rank = host.lists.helix_is_inactive_rank
        helix_position_offsets = host.lists.helix_position_offsets
        helix_owned_new_tokens = host.lists.helix_owned_new_tokens
        position_ids = host.lists.position_ids
        unwritten_input_id_rows = host.lists.unwritten_input_id_rows
        num_cached_tokens_per_seq = host.lists.num_cached_tokens_per_seq
        prompt_lengths = host.lists.prompt_lengths
        gather_ids = host.lists.gather_ids
        mrope_position_ids = host.lists.mrope_position_ids
        mrope_delta_write_seq_slots = host.lists.mrope_delta_write_seq_slots
        mrope_delta_read_seq_slots = host.lists.mrope_delta_read_seq_slots
        multimodal_params_list = host.lists.multimodal_params_list
        extra_inputs = host.extra_inputs
        gen_request_seq_slots = host.lists.gen_request_seq_slots
        _use_mrope = host.use_mrope
        _has_cp_helix = host.has_cp_helix
        mrope_dummy_seq_slot = host.mrope_dummy_seq_slot
        # Whether any generation request in this batch, verify groups included,
        # carries real MRoPE metadata; see the post-loop cleanup below.
        has_gen_mrope_delta = host.has_gen_mrope_delta

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

                first_gather_id = len(position_ids)
                position_ids.extend([position_id] * beam_width)
                num_cached_tokens_per_seq.extend(
                    [past_seen_token_num - request.py_num_compressed_tokens] * beam_width
                )
                prompt_lengths.extend([request.py_prompt_len] * beam_width)
                gather_ids.extend(range(first_gather_id, first_gather_id + beam_width))

                if _use_mrope:
                    mrope_position_delta = self._decode_mrope_position_delta(request)
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
                if request.is_cuda_graph_dummy:
                    # beam_width token rows were added without input ids.
                    unwritten_input_id_rows.append(
                        (len(position_ids) - beam_width, len(position_ids))
                    )
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

        host.n_gen = _n_gen
        host.has_any_multimodal_request = _has_any_multimodal_request
        if _n_gen > 0:
            host.beam_width = beam_width

    def _copy_token_ids_to_device(
        self,
        host: HostInputs,
        scheduled_requests: ScheduledRequests,
        new_tokens_device: torch.Tensor | None,
        new_tokens_lens_device: torch.Tensor | None,
        next_draft_tokens_device: torch.Tensor | None,
        runtime_draft_len: int,
        *,
        attn_metadata: AttentionMetadata,
        enable_spec_decode: bool,
    ) -> None:
        input_ids = host.lists.input_ids
        draft_tokens = host.lists.draft_tokens
        position_ids = host.lists.position_ids
        draft_lens = host.lists.draft_lens
        previous_batch_indices = host.lists.previous_batch_indices
        previous_pos_indices = host.lists.previous_pos_indices
        extend_requests = host.lists.extend_requests
        extend_dummy_requests = host.lists.extend_dummy_requests

        previous_batch_len = len(previous_batch_indices)

        def previous_seq_slots_device() -> torch.Tensor:
            previous_slots = self._buffers.previous_batch_indices_cuda[:previous_batch_len]
            if previous_batch_indices != self._staged_previous_batch_indices:
                previous_batch_indices_host = torch.tensor(
                    previous_batch_indices, dtype=torch.int, pin_memory=prefer_pinned()
                )
                previous_slots.copy_(previous_batch_indices_host, non_blocking=True)
                self._staged_previous_batch_indices = list(previous_batch_indices)
            return previous_slots

        if scheduled_requests.is_attention_dp_phase_idle:
            # Graph padding normally reuses an existing input buffer. An idle
            # rank has no real row to initialize that buffer on its first step.
            input_ids = [1] * len(position_ids)
        num_tokens = len(input_ids)
        num_draft_tokens = len(draft_tokens)
        total_num_tokens = len(position_ids)
        step_gather = self._step_input_gather if enable_spec_decode else None
        assert total_num_tokens <= self._config.max_num_tokens, (
            f"total_num_tokens ({total_num_tokens}) should be less than or equal to max_num_tokens ({self._config.max_num_tokens})"  # noqa: E501
        )
        # if exist requests that do not have previous batch, copy input_ids and draft_tokens
        if num_tokens > 0:
            input_ids = torch.tensor(input_ids, dtype=torch.int, pin_memory=prefer_pinned())
            self._buffers.input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

        if num_draft_tokens > 0:
            draft_tokens = torch.tensor(draft_tokens, dtype=torch.int, pin_memory=prefer_pinned())
            self._buffers.draft_tokens_cuda[: len(draft_tokens)].copy_(
                draft_tokens, non_blocking=True
            )
        if next_draft_tokens_device is not None:
            runtime_tokens_per_gen_step = self.get_runtime_tokens_per_gen_step(runtime_draft_len)
            runtime_draft_token_buffer_width = runtime_tokens_per_gen_step - 1
            num_extend_reqeust_wo_dummy = len(extend_requests) - len(extend_dummy_requests)
            # The offsets of requests without a previous batch and of dummy
            # requests must read 0. On a CUDA graph step (no token padding)
            # where every row is a request with a previous batch, the gathers
            # below overwrite all rows _preprocess_inputs reads, so the buffers
            # need no zeroing first.
            if not (
                attn_metadata.is_cuda_graph
                and previous_batch_len == num_extend_reqeust_wo_dummy
                and not extend_dummy_requests
                and not host.lists.generation_requests
                and scheduled_requests.num_context_requests == 0
            ):
                self._buffers.previous_pos_id_offsets_cuda.zero_()
                self._buffers.previous_kv_lens_offsets_cuda.zero_()

            if previous_batch_len > 0:
                previous_slots = previous_seq_slots_device()
                previous_batch_tokens = previous_batch_len * runtime_tokens_per_gen_step
                if previous_pos_indices != self._staged_previous_pos_indices:
                    previous_pos_indices_host = torch.tensor(
                        previous_pos_indices, dtype=torch.int, pin_memory=prefer_pinned()
                    )
                    self._buffers.previous_pos_indices_cuda[0:previous_batch_tokens].copy_(
                        previous_pos_indices_host, non_blocking=True
                    )
                    self._staged_previous_pos_indices = list(previous_pos_indices)

                # The order of requests in a batch: [context requests, generation requests]
                # generation requests: ['requests that do not have previous batch', 'requests that
                # already have previous batch', 'dummy requests']
                #   1) 'requests that do not have previous batch': disable overlap scheduler or the
                #      first step in the generation server of disaggregated serving.
                #   2) 'requests that already have previous batch': previous iteration's requests.
                #   3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                # Therefore, both of self._buffers.previous_pos_id_offsets_cuda and
                # self._buffers.previous_kv_lens_offsets_cuda are also 3 segments.
                #   For 1) 'requests that do not have previous batch': disable overlap scheduler or
                #          the first step in the generation server of disaggregated serving.
                #       Set these requests' previous_pos_id_offsets and previous_kv_lens_offsets to
                #       '0' to skip the value changes in _preprocess_inputs.
                #       Zeroed above.
                #   For 2) 'requests that already have previous batch': enable overlap scheduler.
                #       Set their previous_pos_id_offsets and previous_kv_lens_offsets according to
                #       new_tokens_lens_device.
                #   For 3) 'dummy requests': pad dummy requests for CUDA graph or attention dp.
                #       Zeroed above.
                previous_begin = num_extend_reqeust_wo_dummy - previous_batch_len
                if step_gather is None or not step_gather.gather(
                    new_tokens_device,
                    next_draft_tokens_device,
                    new_tokens_lens_device,
                    self._buffers.previous_batch_indices_cuda,
                    self._buffers.previous_pos_indices_cuda,
                    previous_batch_len,
                    runtime_tokens_per_gen_step,
                    runtime_draft_token_buffer_width,
                    self._buffers.input_ids_cuda,
                    num_tokens,
                    self._buffers.draft_tokens_cuda,
                    num_draft_tokens,
                    self._buffers.previous_pos_id_offsets_cuda,
                    previous_begin * runtime_tokens_per_gen_step,
                    self._buffers.previous_kv_lens_offsets_cuda,
                    previous_begin,
                ):
                    # previous input ids
                    new_tokens = new_tokens_device.transpose(0, 1)[
                        previous_slots, :runtime_tokens_per_gen_step
                    ].flatten()
                    self._buffers.input_ids_cuda[
                        num_tokens : num_tokens + previous_batch_tokens
                    ].copy_(new_tokens, non_blocking=True)

                    # previous draft tokens
                    previous_batch_draft_tokens = (
                        previous_batch_len * runtime_draft_token_buffer_width
                    )
                    if runtime_draft_token_buffer_width > 0:
                        self._buffers.draft_tokens_cuda[
                            num_draft_tokens : num_draft_tokens + previous_batch_draft_tokens
                        ].copy_(
                            next_draft_tokens_device[
                                previous_slots, :runtime_draft_token_buffer_width
                            ].flatten(),
                            non_blocking=True,
                        )
                    # prepare data for the preprocess inputs
                    kv_len_offsets_device = new_tokens_lens_device - runtime_tokens_per_gen_step
                    self._buffers.previous_pos_id_offsets_cuda[
                        previous_begin * runtime_tokens_per_gen_step : num_extend_reqeust_wo_dummy
                        * runtime_tokens_per_gen_step
                    ].copy_(
                        new_tokens_lens_device[
                            self._buffers.previous_pos_indices_cuda[0:previous_batch_tokens]
                        ],
                        non_blocking=True,
                    )
                    self._buffers.previous_kv_lens_offsets_cuda[
                        previous_begin:num_extend_reqeust_wo_dummy
                    ].copy_(kv_len_offsets_device[previous_slots], non_blocking=True)

        elif new_tokens_device is not None:
            seq_slots_device = previous_seq_slots_device()
            max_draft_len = max(draft_lens)
            new_tokens = new_tokens_device[
                : max_draft_len + 1, seq_slots_device, : self._config.max_beam_width
            ]
            self._buffers.input_ids_cuda[
                num_tokens : num_tokens + previous_batch_len * self._config.max_beam_width
            ].copy_(new_tokens.flatten(), non_blocking=True)

        if (
            not self._config.disable_overlap_scheduler
            and next_draft_tokens_device is None
            and len(extend_requests) > 0
        ):
            # During warmup, for those generation requests, we don't have previous tensors,
            # so we need to set the previous_pos_id_offsets and previous_kv_lens_offsets to zeros
            # to skip the value changes in preprocess_inputs. Otherwise, there will be illegal memory access
            # when writing key/values to the KV cache.
            self._buffers.previous_pos_id_offsets_cuda.zero_()
            self._buffers.previous_kv_lens_offsets_cuda.zero_()

        # The host copy covers input_ids_cuda[:num_tokens] and the device copies
        # cover the previous-batch rows; padding-request rows are covered by
        # neither, so without this the embedding would gather stale contents
        # of the uninitialized buffer, which may exceed the vocabulary size.
        _zero_fill_row_spans(self._buffers.input_ids_cuda, host.lists.unwritten_input_id_rows)

        host.num_tokens = num_tokens
        host.total_num_tokens = total_num_tokens
        host.previous_batch_len = previous_batch_len

    def _copy_position_ids_to_device(self, host: HostInputs) -> None:
        total_num_tokens = host.total_num_tokens
        position_ids = apply_position_id_offset(host.lists.position_ids, model=self.model)
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
            self._buffers.position_ids_cuda[:total_num_tokens].copy_(
                host_position_ids, non_blocking=True
            )
            # Broadcast [N] to [3,1,N]: default for text-only tokens.
            self._buffers.mrope_position_ids_cuda[:, :, :total_num_tokens].copy_(
                self._buffers.position_ids_cuda[:total_num_tokens].view(1, 1, -1).expand(3, 1, -1),
                non_blocking=True,
            )
            # Overwrite multimodal spans with per-axis MRoPE positions.
            for start_idx, end_idx, segment in host.lists.mrope_position_ids:
                if segment.ndim != 3:
                    raise RuntimeError(
                        f"Expected 3D mrope_position_ids, got shape {tuple(segment.shape)}"
                    )
                if segment.shape[0] != 3 and segment.shape[-1] == 3:
                    logger.warning(
                        f"Transposing unexpected mrope_position_ids shape from {tuple(segment.shape)}"
                    )
                    segment = segment.transpose(0, 2).contiguous()
                if segment.shape[:2] != (3, 1):
                    raise RuntimeError(
                        f"Unexpected mrope_position_ids shape {tuple(segment.shape)} for span {start_idx}:{end_idx}"
                    )
                segment = segment.contiguous()
                if segment.device.type == "cpu":
                    segment = maybe_pin_memory(segment)
                self._buffers.mrope_position_ids_cuda[:, :, start_idx:end_idx].copy_(
                    segment[:, :, : end_idx - start_idx], non_blocking=True
                )
        else:
            self._buffers.position_ids_cuda[:total_num_tokens].copy_(
                host_position_ids, non_blocking=True
            )

        host.host_position_ids = host_position_ids

    def _prepare_attn_metadata(
        self,
        host: HostInputs,
        scheduled_requests: ScheduledRequests,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        attn_metadata: AttentionMetadata,
        cache_indirection_buffer: torch.Tensor | None,
        spec_config: DecodingBaseConfig | None,
        *,
        enable_spec_decode: bool,
        is_dummy: bool,
    ) -> None:
        num_cached_tokens_per_seq = host.lists.num_cached_tokens_per_seq
        gen_request_seq_slots = host.lists.gen_request_seq_slots
        extend_requests = host.lists.extend_requests

        if self.mapping.has_cp_helix():
            # A non-None owned-count list is what arms
            # _helix_spec_tokens_valid, and the per-token slots/bounds that
            # flag gates are only ever filled by recompute_helix_spec_buffers,
            # which preprocess_inputs runs under enable_spec_decode. Gate the
            # hand-off here, at the single choke point, so no packing loop can
            # arm the spec path for ordinary helix generation and send its
            # consumers to uninitialized buffers.
            helix_spec_active = bool(enable_spec_decode and host.lists.helix_owned_new_tokens)
            attn_metadata.update_helix_param(
                helix_position_offsets=host.lists.helix_position_offsets,
                helix_is_inactive_rank=host.lists.helix_is_inactive_rank,
                helix_owned_new_tokens=(
                    host.lists.helix_owned_new_tokens if helix_spec_active else None
                ),
            )

        if not attn_metadata.is_cuda_graph:
            # Assumes seq lens do not change between CUDA graph invocations. This applies
            # to draft sequences too. This means that all draft sequences must be padded.
            attn_metadata.seq_lens = torch.tensor(
                host.lists.sequence_lengths,
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
                # Gather straight into the persistent attention buffer: one
                # kernel and no temporary (the rows are [beam, max_seq_len]).
                torch.index_select(
                    cache_indirection_buffer,
                    0,
                    gen_request_seq_slots_tensor,
                    out=self._buffers.cache_indirection[:num_generation_requests],
                )
            if cache_indirection_buffer is not None or is_dummy:
                attn_metadata.beam_width = self._config.max_beam_width
        else:
            attn_metadata.beam_width = 1

        attn_metadata.request_ids = host.lists.request_ids
        attn_metadata.prompt_lens = host.lists.prompt_lengths
        attn_metadata.num_contexts = scheduled_requests.num_context_requests
        remote_tail_modes = {
            getattr(request, "py_csa2_remote_tail_mode", None)
            for request in scheduled_requests.context_requests
        }
        remote_tail_modes.discard(None)
        if remote_tail_modes:
            if len(remote_tail_modes) != 1 or any(
                getattr(request, "py_csa2_remote_tail_mode", None) is None
                for request in scheduled_requests.context_requests
            ):
                raise ValueError("CSA2 remote-tail context batches must be homogeneous")
            attn_metadata.csa2_remote_tail_mode = next(iter(remote_tail_modes))
            attn_metadata.csa2_remote_tail_starts = [
                request.py_csa2_remote_tail_start for request in scheduled_requests.context_requests
            ]
        else:
            attn_metadata.csa2_remote_tail_mode = None
            attn_metadata.csa2_remote_tail_starts = []
        # Use num_chunked_ctx_requests to record the number of extend context requests,
        # so that we can update the kv_lens_cuda correctly in preprocess_inputs.
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
        prepare_context_replay = getattr(attn_metadata, "prepare_context_replay", None)
        if prepare_context_replay is not None:
            prepare_context_replay(
                scheduled_requests.context_requests,
                allow_final_window=(
                    self._config.prefill_cuda_graph_backend == PrefillCudaGraphBackend.DISABLED
                ),
            )
        attn_metadata.prepare()
        host.num_cached_tokens_snapshot = num_cached_tokens_snapshot
        host.remote_tail_modes = remote_tail_modes

    def _pad_inputs(
        self,
        host: HostInputs,
        scheduled_requests: ScheduledRequests,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata | None,
        maybe_graph: bool,
        gather_context_logits: bool,
    ) -> None:
        total_num_tokens = host.total_num_tokens

        if spec_metadata is not None:
            # Set the per-batch counts here, before the attention-DP allgather
            # below: the allgather and prepare() must derive the DP token count
            # from the same fields (see SpecMetadata.dp_num_tokens). Use
            # scheduled_requests.num_generation_requests -- the same-named
            # local above excludes CUDA-graph dummies and would not match the
            # count prepare() uses.
            spec_metadata.num_tokens = total_num_tokens
            spec_metadata.num_generations = scheduled_requests.num_generation_requests
            spec_metadata.seq_lens = host.lists.sequence_lengths

        # Resolve row selection before the existing attention/spec count
        # exchange. Prefill graph padding is selected collectively below and
        # keeps the complete input shape on every rank.
        attn_metadata.padded_num_tokens = None
        prepare_adp_inputs = getattr(self.model, "prepare_adp_inputs", None)
        if self._config.enable_attention_dp and prepare_adp_inputs is not None:
            prepare_adp_inputs(
                attn_metadata,
                all_token_states_required=(
                    gather_context_logits
                    or (
                        spec_metadata is not None
                        and not isinstance(spec_metadata, DSparkSpecMetadata)
                    )
                ),
                requests=scheduled_requests.context_requests,
            )

        spec_all_rank_counts = None
        attn_all_rank_num_tokens = None
        if scheduled_requests.attention_dp_phase is RemoteTailPhase.DECODE:
            attn_all_rank_num_tokens = self._get_remote_tail_decode_rank_tokens(
                attn_metadata, maybe_graph
            )
        elif spec_metadata is not None and self._config.enable_attention_dp:
            (attn_all_rank_num_tokens, spec_all_rank_counts) = (
                self._get_all_rank_num_tokens_and_spec_counts(attn_metadata, spec_metadata)
            )
        virtual_num_tokens = self.pad_tokens(
            attn_metadata,
            total_num_tokens,
            host.num_ctx_requests,
            attn_all_rank_num_tokens,
            allow_prefill_cuda_graph=(
                not host.remote_tail_modes
                and scheduled_requests.attention_dp_phase is not RemoteTailPhase.CONTEXT
            ),
        )

        host.spec_all_rank_counts = spec_all_rank_counts
        host.virtual_num_tokens = virtual_num_tokens

    def _build_inputs(
        self,
        host: HostInputs,
        scheduled_requests: ScheduledRequests,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata | None,
        resource_manager: ResourceManager | None,
        extra_model_inputs: dict[str, Any],
        lora_params: dict[str, Any] | None,
        runtime_draft_len: int,
    ) -> dict[str, Any]:
        inputs = self.model_inputs(
            attn_metadata,
            resource_manager,
            host.virtual_num_tokens,
            host.lists.multimodal_params_list,
        )
        inputs.update(extra_model_inputs)

        if self._config.use_mrope:
            if host.lists.mrope_delta_write_seq_slots:
                delta_write_seq_slots = torch.tensor(
                    host.lists.mrope_delta_write_seq_slots,
                    dtype=torch.long,
                    pin_memory=prefer_pinned(),
                )
                inputs["mrope_delta_write_seq_slots"] = delta_write_seq_slots.to(
                    device="cuda", non_blocking=True
                )

            if host.lists.mrope_delta_read_seq_slots:
                delta_read_seq_slots = torch.tensor(
                    host.lists.mrope_delta_read_seq_slots,
                    dtype=torch.long,
                    pin_memory=prefer_pinned(),
                )
                inputs["mrope_delta_read_seq_slots"] = delta_read_seq_slots.to(
                    device="cuda", non_blocking=True
                )

        if bool(lora_params):
            inputs["lora_params"] = lora_params

        if spec_metadata is not None:
            total_draft_lens = sum(host.lists.draft_lens)
            spec_metadata.draft_tokens = self._buffers.draft_tokens_cuda[:total_draft_lens]
            spec_metadata.request_ids = host.lists.request_ids
            spec_metadata.gather_ids = self._gather_ids_buffer.tensor[: len(host.lists.gather_ids)]
            # num_generations / num_tokens / seq_lens are set above, before the
            # attention-DP allgather that must agree with prepare().
            spec_metadata.host_position_ids = host.host_position_ids
            if host.context_prompt_lookahead is not None:
                spec_metadata.populate_context_prompt_lookahead(host.context_prompt_lookahead)
            # No-op for non 1-model
            spec_metadata.populate_sampling_params_for_one_model(scheduled_requests.all_requests())
            spec_metadata.prepare()
            # One-model rejection: one-hot the stale draft_probs rows of gen
            # requests that produced no draft tokens this step, so the (possibly
            # captured) rejection kernel reads a legal placeholder distribution.
            spec_metadata.write_padding_onehot_draft_probs(
                host.lists.padding_gen_slots, runtime_draft_len
            )
            inputs["spec_metadata"] = spec_metadata

            if self._config.enable_attention_dp:
                set_spec_metadata_all_rank_num_tokens(spec_metadata, *host.spec_all_rank_counts)

        if host.mm_token_indices is not None:
            ship_multimodal_indices(
                inputs,
                mm_token_indices_cpu=host.mm_token_indices,
                text_token_indices_cpu=host.text_token_indices_ctx,
                num_ctx_tokens=host.num_ctx_tokens,
                total_num_tokens=host.total_num_tokens,
            )
        return inputs

    def _record_iter_states(self, host: HostInputs, scheduled_requests: ScheduledRequests) -> None:
        num_generation_tokens = (
            len(host.lists.generation_requests)
            + len(host.lists.extend_requests)
            + sum(host.lists.draft_lens)
        )
        self.iter_states["num_ctx_requests"] = host.num_ctx_requests
        self.iter_states["num_ctx_tokens"] = host.num_ctx_tokens
        self.iter_states["num_generation_tokens"] = num_generation_tokens
        # Count the already-cached prefix for the sequences scheduled this iteration.
        self.iter_states["cached_kv_tokens"] = sum(host.lists.num_cached_tokens_per_seq)
        if self.log_cached_kv_tokens_per_req:
            self.record_cached_kv_tokens_per_req(
                host.lists.num_cached_tokens_per_seq,
                (
                    (scheduled_requests.context_requests, 1),
                    (host.lists.extend_requests, 1),
                    (host.lists.generation_requests, host.beam_width),
                ),
            )

    def record_promoted_iter_states(self, scheduled_requests: ScheduledRequests) -> None:
        """Count promoted final-context rows as the context requests they were scheduled as."""
        self.iter_states["num_ctx_requests"] = scheduled_requests.num_context_requests
        self.iter_states["num_ctx_tokens"] = sum(
            request.context_chunk_size for request in scheduled_requests.context_requests
        )
        self.iter_states["num_generation_tokens"] = scheduled_requests.num_generation_requests

    def record_input_copy(self) -> None:
        """Mark the end of this pass's host-to-device input copies."""
        self._prepare_inputs_event = torch.cuda.Event()
        self._prepare_inputs_event.record()

    def wait_for_input_copy(self) -> None:
        if self._prepare_inputs_event is not None:
            self._prepare_inputs_event.synchronize()

    def _record_steady_gen_cache(
        self,
        host: HostInputs,
        kv_cache_manager: KVCacheManager | KVCacheManagerV2,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata | None,
        new_tokens_device: torch.Tensor | None,
        lora_params: dict[str, Any] | None,
    ) -> None:
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
            and not host.lists.mrope_position_ids
            and not host.lists.mrope_delta_write_seq_slots
            and not host.lists.mrope_delta_read_seq_slots
            and not self.use_beam_search
            and self._config.max_beam_width == 1
            and self._steady_gen_cache_supported
            and not host.has_cp_helix
            and host.num_ctx_requests == 0
            and not host.lists.extend_requests
            and host.n_gen > 0
            and host.previous_batch_len == host.n_gen
            and host.num_tokens == 0
            and not host.has_any_multimodal_request
            and not host.lists.multimodal_params_list
            and not lora_params
            and attn_metadata.padded_num_tokens is None
            and get_position_id_offset(self.model) == 0
            and not getattr(kv_cache_manager, "kv_compression_manages_history", False)
        ):
            self._buffers.steady_gen_positions_pinned[: host.n_gen].copy_(
                torch.as_tensor(host.num_cached_tokens_snapshot, dtype=torch.int)
            )
            self._steady_gen_cache = {
                "num_requests": host.n_gen,
                "request_ids": host.lists.all_gen_request_ids,
                "prompt_lens": host.lists.prompt_lengths,
                "seq_lens_ones": maybe_pin_memory(torch.ones(host.n_gen, dtype=torch.int)),
                "use_mrope": host.use_mrope,
            }

    def _get_remote_tail_decode_rank_tokens(
        self, attn_metadata: AttentionMetadata, can_run_graph: bool
    ) -> list[int]:
        """Require captured decode on every rank in the normal token-count vote."""
        gathered = self.dist.tp_allgather_int64([attn_metadata.num_tokens, int(can_run_graph)])
        if not gathered[:, 1].all():
            raise RuntimeError(
                "Conditional attention-DP decode requires a captured CUDA "
                "graph on every rank; at least one rank missed its graph"
            )
        return gathered[:, 0].tolist()

    def preprocess_inputs(
        self, inputs: dict[str, Any], *, enable_spec_decode: bool, runtime_draft_len: int
    ) -> dict[str, Any]:
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

    def postprocess_inputs(
        self, inputs: dict[str, Any], *, enable_spec_decode: bool, runtime_draft_len: int
    ) -> None:
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
                # used by `preprocess_inputs`.
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
                    # preprocess_inputs (capture symmetry, like position_ids
                    # above). The recompute's OVERWRITES (slots/bounds/
                    # kv_lens) need no reversal: every consumer buffer is
                    # rewritten from host state at the next step's prepare.
                    inputs["attn_metadata"].helix_position_offsets[:previous_batch_tokens] -= (
                        self._buffers.previous_pos_id_offsets_cuda[:previous_batch_tokens]
                    )
