# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 request metadata and bounded staging for TRTLLM sparse MLA."""

from __future__ import annotations

import functools
import itertools
import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from itertools import accumulate, chain
from typing import TYPE_CHECKING

import numpy as np
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.modules.multi_stream_utils import private_stream
from tensorrt_llm.math_utils import ceil_div

from .params import CSA2BackendForwardArgs, CSA2Layer, CSA2MetadataParams, CSA2Mode

if TYPE_CHECKING:
    from .....pyexecutor.llm_request import LlmRequest
    from .decoder_replay import DecoderReplayPlan

_SWA_TILE = 128
_HEAD_DIM = 512
_TMA_MIN_BACKING_BYTES = 128 << 10
# Decode tokens per request that DeepGEMM paged MQA logits score per request.
_BATCHED_DECODE_NEXT_N = range(2, 9)

# Forward serials are unique across metadata objects: graph copies of one metadata share its
# paged-indexer scratch and per-forward caches, and a per-object count would repeat between them.
_FORWARD_SERIALS = itertools.count(1)
# Buffer-bank entry: serial of the prepare that last wrote the bank's tables.
_BANK_WRITER = "prepare_serial"


def csa2_side_stream(device, index: int = 0) -> torch.cuda.ExternalStream:
    """Side stream ``index`` of a device for CSA2 work that only joins the model stream later."""
    return private_stream(torch.device(device).index, f"csa2_side_{index}")


@functools.cache
def _layout_layers(layout) -> tuple:
    """``layout.layer`` of every layer of the (immutable) ``layout``."""
    return tuple(layout.layer(i) for i in range(len(layout.compress_ratios)))


def _tma_backed_empty(shape, dtype, device):
    numel = math.prod(shape)
    element_size = torch.empty((), dtype=dtype).element_size()
    backing_numel = max(numel, (_TMA_MIN_BACKING_BYTES + element_size - 1) // element_size)
    # The TMA descriptors that read this buffer expect _TMA_MIN_BACKING_BYTES
    # of mapped memory from its base, even for smaller tensors.
    return torch.empty(backing_numel, dtype=dtype, device=device)[:numel].view(shape)


@dataclass
class CSA2SharedKVPlan:
    query_requests: torch.Tensor
    query_positions: torch.Tensor
    # First SWA position per request, written by the shared compact launch
    # from ``query_positions[swa_first]``, ``window`` and ``swa_floors``.
    swa_starts: torch.Tensor
    swa_offsets: torch.Tensor
    main_offsets: torch.Tensor
    swa_pages: torch.Tensor
    main_pages: torch.Tensor | None
    swa_page_size: int
    main_page_size: int
    num_requests: int
    swa_rows: int
    main_rows: int
    swa_first: torch.Tensor
    swa_floors: torch.Tensor
    window: int


@dataclass
class CSA2IndexerGroup:
    """One packed indexer MQA call over query rows ``[first, last)``.

    This rank scores rows ``[lo, hi)`` (TP-split prefill groups allgather with
    ``sizes``); ``descriptors`` are its int32 ``[pieces, 5]`` prefix-gather
    rows (request, key width, key offset, first query row, row in group) for
    ``keys`` gathered keys, ``width`` the widest piece. A group whose width is
    zero has no visible keys.
    """

    is_prefill: bool
    first: int
    last: int
    lo: int
    hi: int
    sizes: list[int] | None
    width: int
    keys: int
    descriptors: torch.Tensor | None


class _CoalescedUploads:
    """Batches the metadata H2D copies of one prepare into a single upload.

    ``copy`` returns the usual graph-stable device buffer and defers its fill;
    ``flush`` uploads all recorded host values at once and scatters them, and
    copies the recorded device page tables in one launch. The buffers must not
    be read before ``flush``.
    """

    _ALIGN = 16

    def __init__(self, metadata: CSA2TrtllmMetadata):
        self._metadata = metadata
        self._pending = []
        self._tables = []
        self._bytes = 0

    def copy(self, key: str, value) -> torch.Tensor:
        metadata = self._metadata
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                result = metadata._get_csa2_buffer(key, value.shape, value.dtype, like=value)
                self._tables.append((result, value))
                return result
            value = value.numpy()
        if not value.flags.c_contiguous:
            value = np.ascontiguousarray(value)
        host = torch.from_numpy(value)
        result = metadata._get_csa2_buffer(key, host.shape, host.dtype)
        if not result.is_cuda:
            result.copy_(host)
        elif value.size:
            offset = -(-self._bytes // self._ALIGN) * self._ALIGN
            self._pending.append((result, value, offset))
            self._bytes = offset + value.nbytes
        return result

    def flush(self) -> None:
        from . import kernel

        metadata = self._metadata
        if self._tables:
            targets, values = zip(*self._tables)
            if targets[0].is_cuda and kernel.dsl_available():
                rows = [kernel.table_copy_descriptor(*pair) for pair in self._tables]
                kernel.copy_tables_(
                    metadata._device_descriptors("prepared_page_tables", rows),
                    max(target.shape[0] for target in targets),
                )
            else:
                torch._foreach_copy_(targets, values)
            self._tables.clear()
        if not self._pending:
            return
        packed = np.empty(self._bytes, dtype=np.uint8)
        for _, value, offset in self._pending:
            packed[offset : offset + value.nbytes] = value.reshape(-1).view(np.uint8)
        device = self._pending[0][0].device
        slab = getattr(metadata, "_csa2_upload_slab", None)
        if slab is None or slab.device != device or slab.numel() < self._bytes:
            # Only the scatter below reads the slab, so graphs never see it.
            size = max(self._bytes, 2 * slab.numel() if slab is not None else 0)
            slab = metadata._csa2_upload_slab = torch.empty(size, dtype=torch.uint8, device=device)
        metadata._copy_host("coalesced_upload", slab[: self._bytes], torch.from_numpy(packed))
        groups = {}
        for result, value, offset in self._pending:
            source = slab[offset : offset + value.nbytes].view(result.dtype).view(result.shape)
            destinations, sources = groups.setdefault(result.dtype, ([], []))
            destinations.append(result)
            sources.append(source)
        for destinations, sources in groups.values():
            torch._foreach_copy_(destinations, sources)
        self._pending.clear()
        self._bytes = 0


class CSA2TrtllmMetadata(TrtllmAttentionMetadata):
    """Manager-backed request metadata with bounded native query staging.

    Normal prepare resolves manager pages for the packed model forward.
    for_query_tile creates a separate compute-only metadata object.
    Eager context retains real query groups with virtual staged KV lengths.
    Generation and captured context use independent one-query generation rows.
    Source causality, request isolation and windows are encoded in selections;
    these bounded compute pools are separate from persistent manager caches.
    """

    indexer_max_chunk_size: int = 32768
    indexer_q_split_threshold: int = 8192
    # Compute-tile metadata only: the shared converted-KV plan of an eager
    # context phase, or None for independent-query staging.
    shared_plan: CSA2SharedKVPlan | None = None
    _csa2_defer_decode_outputs: bool = False
    _csa2_deferred_decode_outputs: bool = False
    _csa2_pending_owner_tables: tuple | None = None
    _csa2_pending_prepare: tuple | None = None
    _csa2_model_driven: bool = False
    _csa2_graph_reserved: bool = False
    # "encoder" / "decoder" while a forward replays a source's SWA rows, else None.
    csa2_replay_mode: str | None = None

    # Request metadata is held directly on this object. Main/index physical
    # tables and write slots are owner-keyed; SWA and visibility are layer-keyed.
    csa2_indices: dict[int, torch.Tensor]
    csa2_candidates: dict[int, torch.Tensor]
    # DeepGEMM sparse-logits form of the candidates: sorted sparse block ids and
    # the number of valid leading candidate columns per query
    csa2_candidate_blocks: dict[int, torch.Tensor]
    csa2_candidate_counts: dict[int, torch.Tensor]
    _csa2_last_layer: int
    csa2_swa_indices: dict[int, torch.Tensor]
    csa2_swa_write_slots: dict[int, torch.Tensor]
    csa2_visible_lengths: dict[int, torch.Tensor]
    csa2_main_write_slots: dict[int, torch.Tensor]
    csa2_global_page_tables: dict[int, torch.Tensor]
    csa2_token_requests: torch.Tensor
    csa2_global_page_sizes: dict[int, int]
    csa2_global_max_positions: dict[int, int]
    csa2_kv_sources: dict[int, int | None]
    decoder_replay_plan: DecoderReplayPlan | None = None
    # Whole-prompt endpoints, frozen before overlap scheduling advances requests.
    # Unlike prompt_lens, these are not the current context chunk lengths.
    decoder_context_ends: tuple[int, ...] | None = None
    csa2_decoder_capture_lens: tuple[int, ...] | None = None

    def prepare_context_mla_with_cached_kv(
        self, cached_token_lens: torch.Tensor, kv_lens: torch.Tensor
    ) -> None:
        """Context statistics and indptrs without the dense-MLA chunked-context loop plan.

        CSA2 attends over its own staged sparse selections, so the per-loop KV
        chunk tensors that ``MLA`` reads for chunked dense context are never
        consumed.
        """
        features = self.runtime_features
        if features is None or not features.chunked_prefill:
            super().prepare_context_mla_with_cached_kv(cached_token_lens, kv_lens)
            return
        self.runtime_features = replace(features, chunked_prefill=False)
        try:
            super().prepare_context_mla_with_cached_kv(cached_token_lens, kv_lens)
        finally:
            self.runtime_features = features

    def get_adp_token_counts(self) -> list[int]:
        """Input rows, decoder rows, and whether this rank's replay plan must run.

        The last count is 1 when the local plan rewrites metadata (row compaction,
        SWA floors or a private decoder cache), so this rank enters the decoder
        replay whatever its peers' counts are.
        """
        plan = self.decoder_replay_plan
        input_tokens = self.padded_num_tokens or self.num_tokens
        decoder_tokens = (
            plan.num_replay_tokens if plan and plan.replays_local_tokens else input_tokens
        )
        required = int(plan is not None and plan.updates_local_metadata)
        return [input_tokens, decoder_tokens, required]

    def set_adp_token_counts(self, counts: list[list[int]]) -> list[int]:
        """Keep or drop the replay plan by one decision that every attention-DP rank shares.

        A replaying rank runs the decoder half's MoE collectives from the decoder-pass
        graph at its bucket's static shapes, so every peer -- an idle rank running a
        dummy step included -- must replay alongside it. An identity plan is dropped
        only when no rank requires its plan and no rank compacts rows.
        """
        input_counts, decoder_counts, required = counts
        plan = self.decoder_replay_plan
        if plan is None:
            if any(required):
                raise ValueError(
                    "An attention-DP peer enters the decoder replay but this rank has no "
                    "replay plan; the decoder-half collectives would diverge"
                )
            return input_counts
        plan.saved_all_rank_num_tokens = input_counts
        plan.replay_all_rank_num_tokens = decoder_counts
        if not any(required) and input_counts == decoder_counts:
            self.decoder_replay_plan = None
        return input_counts

    def __post_init__(self) -> None:
        # Graph clones rerun post-init: eligibility belongs to their own next
        # successful preparation, not the metadata object they were copied from.
        # Likewise the steady snapshot, whose in-place refresh would write the source's buffers.
        self._csa2_deferred_decode_outputs = False
        self._csa2_steady_snapshot = None
        self.decoder_replay_plan = None
        self.decoder_context_ends = None
        params = self.sparse_metadata_params
        if isinstance(params, CSA2MetadataParams):
            self.indexer_max_chunk_size = params.indexer_max_chunk_size
            self.indexer_q_split_threshold = params.q_split_threshold
        # CSA2 never reads the generic MLA chunked-context tables.
        self.enable_context_mla_with_cached_kv = False
        super().__post_init__()

    @property
    def tokens_per_block(self) -> int:
        return (
            self.kv_cache_manager.tokens_per_block
            if self.kv_cache_manager is not None
            else _SWA_TILE
        )

    @property
    def host_kv_cache_pool_pointers(self) -> torch.Tensor:
        return (
            self.pool_pointers
            if hasattr(self, "pool_pointers")
            else super().host_kv_cache_pool_pointers
        )

    @property
    def host_kv_cache_pool_mapping(self) -> torch.Tensor:
        return (
            self.pool_mapping
            if hasattr(self, "pool_mapping")
            else super().host_kv_cache_pool_mapping
        )

    @property
    def stage_kv_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The staging KV ``(orig_quant, quant_orig)`` pair this metadata derives.

        Both tensors are allocated once and rewritten in place by
        ``derive_stage_kv_scales``, so a caller may keep either handle across
        forwards and across CUDA Graph replay.
        """
        return self._csa2_stage_kv_quant, self._csa2_stage_kv_dequant

    def _allocate(
        self,
        capacity: int,
        heads: int,
        extra_capacity: int,
        device: torch.device,
        staging_dtype: torch.dtype = torch.bfloat16,
        shared_rows: int | None = None,
    ) -> None:
        if staging_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError("CSA2 native staging requires BF16 or E4M3")
        self.staging_dtype = staging_dtype
        if staging_dtype == torch.float8_e4m3fn:
            self.mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
            self.mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
            self.native_unit_scale = torch.ones(1, dtype=torch.float32, device=device)
            # Range-aware staging KV scale, rederived every forward from the
            # persistent pools' group-scale bytes. Written in place so CUDA Graph
            # replay observes the live value; the unit default keeps a metadata
            # that never derives a scale numerically identical to before.
            self._csa2_stage_kv_dequant = torch.ones(1, dtype=torch.float32, device=device)
            self._csa2_stage_kv_quant = torch.ones(1, dtype=torch.float32, device=device)
            # The KV quant scale the staged rows carry, alternating between two
            # slots so a launch never reads the slot it writes (``StageDequantKernel``).
            self._csa2_staged_scales = torch.ones(2, dtype=torch.float32, device=device)
            # Presence bitmap of the group-scale bytes each format's selected rows
            # carry, one 256-bit word group per format, then Q's amax bits.
            self._csa2_stage_scale_bitmap = torch.zeros(17, dtype=torch.int32, device=device)
            self.latent_placeholder = torch.empty(
                capacity, _HEAD_DIM, dtype=torch.bfloat16, device=device
            )
        self._csa2_context_geometry = None
        if shared_rows is None:
            self.swa_pool = _tma_backed_empty(
                (capacity, _SWA_TILE, _HEAD_DIM), staging_dtype, device
            )
            self.extra_pool = _tma_backed_empty(
                (capacity, max(extra_capacity, 1), _HEAD_DIM), staging_dtype, device
            )
        else:
            self.shared_pool = _tma_backed_empty((shared_rows, _HEAD_DIM), staging_dtype, device)
            self._csa2_shared_selected = torch.zeros(shared_rows, dtype=torch.int32, device=device)
            self._csa2_shared_main_scales = torch.zeros(2, dtype=torch.int32, device=device)
            self.swa_pool = self.extra_pool = self.shared_pool
        self.pool_pointers = torch.tensor(
            [[self.swa_pool.data_ptr(), 0]], dtype=torch.int64, device="cpu"
        )
        self.pool_mapping = torch.zeros((1, 2), dtype=torch.int32, device="cpu")
        self.num_sparse_topk = _SWA_TILE + extra_capacity
        self.max_seq_len = self.num_sparse_topk
        self.kv_cache_params = KVCacheParams(use_cache=True)
        self.kv_cache_block_offsets = torch.zeros(
            (1, capacity, 2, (self.num_sparse_topk + _SWA_TILE - 1) // _SWA_TILE),
            dtype=torch.int32,
            device=device,
        )
        self.prepared_indices = torch.full(
            (capacity, self.num_sparse_topk), -1, dtype=torch.int32, device=device
        )
        self.prepared_lens = torch.empty(capacity, dtype=torch.int32, device=device)
        self.prepared_counter = torch.zeros(1, dtype=torch.uint32, device=device)
        self.prepared_cu_q = torch.arange(capacity + 1, dtype=torch.int32, device=device) * heads
        self.prepared_cu_kv = (
            torch.arange(capacity + 1, dtype=torch.int32, device=device) * self.num_sparse_topk
        )
        self.query_lens_host = torch.ones(capacity, dtype=torch.int32, device="cpu")
        self.query_lens_device = torch.ones(capacity, dtype=torch.int32, device=device)
        self.kv_lens.fill_(self.num_sparse_topk)
        self.kv_lens_cuda.fill_(self.num_sparse_topk)
        self.prompt_lens_cpu.fill_(1)
        self.prompt_lens_cuda.fill_(1)
        self.host_request_types.fill_(1)
        self.host_total_kv_lens.zero_()
        # Each query count has independent native workspace storage.
        # Eager initialization and capture must use that same tensor.
        self.cuda_graph_workspace = self.workspace

    @classmethod
    def for_query_tile(
        cls,
        q: torch.Tensor,
        extra_width: int,
        *,
        staging_dtype=torch.bfloat16,
        context_lengths: list[int] | None = None,
        shared_rows: int | None = None,
    ) -> CSA2TrtllmMetadata:
        """Prepare fixed geometry before TrtllmAttention.forward allocates output.

        q is BF16 [queries, heads, 512]. Extra capacity is a selected-main-row
        bound, not the persistent cache size. Callers retain one metadata object
        per geometry and warm it with the normal forward before graph capture.
        As with TrtllmAttentionMetadata, set is_cuda_graph for captured calls.
        """
        count, heads, dim = q.shape
        if context_lengths and (
            any(length <= 0 for length in context_lengths) or sum(context_lengths) != count
        ):
            raise ValueError("CSA2 context groups must cover every query exactly")
        if shared_rows is not None and (not context_lengths or not 0 < shared_rows < 1 << 31):
            raise ValueError("CSA2 shared staging requires context groups and int32 row capacity")
        if count <= 0 or dim != _HEAD_DIM or extra_width < 0:
            raise ValueError(
                "CSA2 metadata requires a nonempty 512D query tile and nonnegative extra width"
            )
        with torch.cuda.device(q.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare and warm CSA2 metadata before CUDA Graph capture")
            metadata = cls(
                max_num_requests=len(context_lengths) if context_lengths else count,
                max_num_tokens=count,
            )
            metadata._allocate(
                count, heads, (extra_width + 127) // 128 * 128, q.device, staging_dtype, shared_rows
            )
        metadata.num_query_heads = heads
        metadata._seq_lens = metadata.query_lens_host
        metadata._seq_lens_cuda = metadata.query_lens_device
        metadata._num_contexts = metadata._num_ctx_tokens = 0
        metadata._num_generations = metadata._num_tokens = count
        metadata._bind_runtime_views(
            kv_lens_cuda=metadata.kv_lens_cuda,
            kv_lens=metadata.kv_lens,
            prompt_lens_cuda=metadata.prompt_lens_cuda,
            prompt_lens_cpu=metadata.prompt_lens_cpu,
            host_request_types=metadata.host_request_types,
        )
        metadata.host_total_kv_lens[1] = count * metadata.num_sparse_topk
        metadata.cu_q_seqlens = metadata.prepared_cu_q
        metadata.cu_kv_seqlens = metadata.prepared_cu_kv
        if context_lengths:
            metadata._bind_context_tile(context_lengths)
        return metadata

    def _prepare_indexer_plans(self, copy) -> None:
        """Packed indexer groups and prefix-gather descriptors, uploaded with the metadata.

        Full layers and the candidate source group the rows of each KV owner as
        in the DSA indexer: consecutive requests of a phase share one MQA call
        up to ``indexer_max_chunk_size`` rows and the logits budget, longer
        requests are split, decode rows join only when the native paged indexer
        cannot serve them, and decode or captured rows span the admitted width
        so replays see newly published keys. Reindex layers gather each context
        request's prefix. All descriptors ride one upload.
        """
        from tensorrt_llm._utils import get_sm_version

        from ..dsa.indexer import _INDEXER_MQA_LOGITS_ELEM_BUDGET, _split_prefill_queries

        layout = self.kv_cache_manager.layout
        ranges = self.csa2_request_query_ranges
        contexts = self.csa2_num_context_requests
        starts = self.csa2_request_start_positions
        sm = get_sm_version()
        decoded = layout.index_head_dim != 128 or sm < 100
        native_decode = 100 <= sm < 110 and 1 <= layout.index_heads <= 64 and not decoded
        requests = contexts if native_decode else len(ranges)
        chunk_size = self.indexer_max_chunk_size
        # The decoded fallback materializes per-head dot products over all keys.
        budget = _INDEXER_MQA_LOGITS_ELEM_BUDGET
        if decoded:
            budget = max(1, budget // layout.index_heads)

        def columns(width: int) -> int:
            # DeepGEMM pads each compressed logits row to 256 columns
            return -(-width // 256) * 256

        rows, slices = [], []
        modes = [layout.layer(index) for index in layout.index_source_layer_ids]
        self.csa2_indexer_groups = {}
        for owner in sorted({layer.kv_source for layer in modes if layer.mode == CSA2Mode.FULL}):
            ratio = layout.compress_ratios[owner]
            max_positions = self.csa2_global_max_positions[owner]
            raw = []
            for request in range(requests):
                begin, end = ranges[request]
                is_prefill = request < contexts
                fixed = self.is_cuda_graph or not is_prefill
                prefix = starts[request]
                widest = (
                    max_positions if fixed else min(max_positions, (prefix + end - begin) // ratio)
                )
                step = max(1, min(chunk_size, budget // max(1, columns(widest))))
                for start in range(begin, end, step):
                    stop = min(start + step, end)
                    width = (
                        max_positions
                        if fixed
                        else min(max_positions, (prefix + stop - begin) // ratio)
                    )
                    piece = (start, stop, width, request)
                    phase, group = raw[-1] if raw else (None, None)
                    if not decoded and phase == is_prefill and width > 0 and group[-1][2] > 0:
                        merged = max(width, *(block[2] for block in group))
                        if (
                            stop - group[0][0] <= chunk_size
                            and (stop - group[0][0]) * columns(merged) <= budget
                        ):
                            group.append(piece)
                            continue
                    raw.append((is_prefill, [piece]))
            groups = self.csa2_indexer_groups[owner] = []
            for is_prefill, pieces in raw:
                first, last = pieces[0][0], pieces[-1][1]
                if pieces[0][2] <= 0:
                    groups.append(
                        CSA2IndexerGroup(is_prefill, first, last, first, last, None, 0, 0, None)
                    )
                    continue
                local_start, local_end, sizes = _split_prefill_queries(
                    last - first, self.mapping, self.indexer_q_split_threshold if is_prefill else -1
                )
                lo, hi = first + local_start, first + local_end
                offset, keys, width = len(rows), 0, 0
                for start, stop, piece_width, request in pieces:
                    start, stop = max(start, lo), min(stop, hi)
                    if start < stop:
                        rows.append((request, piece_width, keys, start, start - lo))
                        keys += piece_width
                        width = max(width, piece_width)
                group = CSA2IndexerGroup(is_prefill, first, last, lo, hi, sizes, width, keys, None)
                groups.append(group)
                slices.append((group, offset, len(rows)))
        self.csa2_indexer_decode_planned = not native_decode
        self.csa2_indexer_prefixes = {}
        for owner in sorted({layer.kv_source for layer in modes if layer.mode == CSA2Mode.REINDEX}):
            max_positions = self.csa2_global_max_positions[owner]
            for request in range(contexts):
                begin, end = ranges[request]
                width = (
                    max_positions
                    if self.is_cuda_graph
                    else min(max_positions, starts[request] + end - begin)
                )
                slices.append(((owner, request), len(rows), len(rows) + 1))
                rows.append((request, width, 0, begin, 0))
        if not rows:
            return
        packed = copy("indexer_descriptors", np.asarray(rows, dtype=np.int32))
        for target, begin, end in slices:
            if isinstance(target, CSA2IndexerGroup):
                target.descriptors = packed[begin:end]
            else:
                self.csa2_indexer_prefixes[target] = (packed[begin:end], int(rows[begin][1]))

    def _prepare_shared_tables(self, copy) -> None:
        """Each owner's shared prefill staging tables, uploaded with the other prepare metadata.

        Per request: its first query row, its SWA bank row offset and its main
        bank row offset (``requests``, ``requests + 1`` and ``requests + 1``
        int64 values in one buffer), plus the owner's SWA and main row totals.
        """
        self._csa2_shared_tables = {}
        requests = self.csa2_num_context_requests
        lengths = self.csa2_request_lengths[:requests]
        window = self.kv_cache_manager.layout.window_size
        ends = accumulate(lengths)
        first = [end - length if length else 0 for end, length in zip(ends, lengths)]
        swa_offsets = list(
            accumulate((length + window - 1 if length else 0 for length in lengths), initial=1)
        )
        for owner in (None, *self._csa2_main_domain_counts):
            counts = [0] * requests if owner is None else self._csa2_main_domain_counts[owner]
            main_offsets = list(
                accumulate(
                    (end if length else 0 for end, length in zip(counts, lengths)),
                    initial=swa_offsets[-1],
                )
            )
            packed = copy(
                f"shared_plan:{owner}",
                np.fromiter(chain(first, swa_offsets, main_offsets), dtype=np.int64),
            )
            self._csa2_shared_tables[owner] = (
                *packed.split((requests, requests + 1, requests + 1)),
                swa_offsets[-1] - 1,
                main_offsets[-1] - swa_offsets[-1],
            )

    def _shared_context_plan(self, q, extra_width, layer_idx):
        # Graph frames are filtered by the caller. Without logical provenance,
        # keep the existing selected-row path.
        if (
            layer_idx is None
            or _SWA_TILE + (extra_width + 127) // 128 * 128 > 4096
            or not hasattr(self, "_csa2_main_domain_counts")
            or q.shape[0] != self.num_ctx_tokens
        ):
            return None
        manager = self.kv_cache_manager
        requests = self.csa2_num_context_requests
        lengths = self.csa2_request_lengths[:requests]
        if not lengths or sum(lengths) != q.shape[0]:
            return None
        layer = manager.layout.layer(layer_idx)
        owner = layer.kv_source
        window = manager.layout.window_size
        ratio = 1 if owner is None else manager.layout.compress_ratios[owner]
        first, swa_offsets, main_offsets, swa_rows, main_rows = self._csa2_shared_tables[owner]
        if swa_rows + main_rows > q.shape[0] * (window + extra_width):
            return None
        if 1 + swa_rows + main_rows >= 1 << 31:
            raise ValueError("CSA2 shared KV row indices exceed int32 capacity")
        return CSA2SharedKVPlan(
            self.csa2_token_requests[: q.shape[0]],
            self.csa2_positions[: q.shape[0]],
            self._get_csa2_buffer(f"shared_starts:{owner}", (requests,), torch.int64),
            swa_offsets,
            main_offsets,
            self._csa2_swa_page_tables[layer_idx][:requests],
            None if owner is None else self.csa2_global_page_tables[owner][:requests],
            manager.tokens_per_block,
            manager.tokens_per_block // ratio,
            requests,
            swa_rows,
            main_rows,
            first,
            self.csa2_replay_start_positions[:requests],
            window,
        )

    def get_query_tile_metadata(
        self,
        q: torch.Tensor,
        extra_width: int,
        query_start: int | None = None,
        *,
        staging_dtype=torch.bfloat16,
        layer_idx: int | None = None,
    ) -> CSA2TrtllmMetadata:
        """Share bounded staging, retaining real request groups for context."""
        context_lengths = []
        if query_start is not None and query_start < self.num_ctx_tokens:
            end = query_start + q.shape[0]
            if end > self.num_ctx_tokens:
                raise ValueError("CSA2 query tiles must not cross the context/generation boundary")
            for begin, stop in self.csa2_request_query_ranges:
                lo, hi = max(begin, query_start), min(stop, end)
                if lo < hi:
                    context_lengths.append(hi - lo)
            if sum(context_lengths) != q.shape[0]:
                raise ValueError("CSA2 context tile must be covered by its packed request ranges")
        # Native context grouping currently has host-bound launch metadata.
        # Captured contexts retain the existing independent-query generation
        # path; choose it during graph warmup too, so its workspace is ready.
        with torch.cuda.device(q.device):
            capturing = torch.cuda.is_current_stream_capturing()
        if context_lengths and (self.is_cuda_graph or capturing):
            context_lengths = []
        shared_plan = (
            self._shared_context_plan(q, extra_width, layer_idx)
            if context_lengths and query_start == 0
            else None
        )
        shared_rows = (
            None if shared_plan is None else 1 + shared_plan.swa_rows + shared_plan.main_rows
        )
        if not hasattr(self, "_csa2_query_tiles"):
            self._csa2_query_tiles = {}
        owner = None if shared_plan is None else self.csa2_kv_sources.get(layer_idx)
        key = (
            q.device,
            q.shape[1],
            q.shape[0],
            (extra_width + 127) // 128 * 128,
            bool(context_lengths),
            staging_dtype,
            shared_rows,
            len(context_lengths),
            owner,
        )
        workspace = None
        # Retain only the latest eager shape per phase, family and owner.
        # Graph warmup reserves storage permanently: shallow metadata clones
        # share this cache, and an eager use can clear a tile's is_cuda_graph.
        for previous, tile in tuple(self._csa2_query_tiles.items()):
            if (
                previous != key
                and not tile._csa2_graph_reserved
                and previous[:2] == key[:2]
                and previous[3:6] == key[3:6]
                and previous[8] == key[8]
            ):
                # Eager tiles run sequentially on the model stream. Keep their
                # native scratch high-water mark without retaining old pools;
                # graph-owned workspaces must never be borrowed or resized.
                workspace = self._csa2_query_tiles.pop(previous).workspace
        with torch.cuda.device(q.device):
            capturing = torch.cuda.is_current_stream_capturing()
            metadata = self._csa2_query_tiles.get(key)
            if metadata is None:
                metadata = self.for_query_tile(
                    q,
                    extra_width,
                    staging_dtype=staging_dtype,
                    context_lengths=context_lengths,
                    shared_rows=shared_rows,
                )
                if workspace is not None:
                    metadata.workspace = metadata.cuda_graph_workspace = workspace
                self._csa2_query_tiles[key] = metadata
            metadata.shared_plan = shared_plan
            metadata._csa2_graph_reserved |= self.is_cuda_graph or capturing
            metadata.is_cuda_graph = capturing
            if capturing:
                self._csa2_replay_capture_signature = getattr(self, "csa2_replay_signature", None)
            if context_lengths:
                metadata._bind_context_tile(context_lengths)
        return metadata

    def _bind_runtime_views(self, **kwargs) -> None:
        # Generic prepare/rebinding can replace the geometry-only child views.
        self._csa2_context_geometry = None
        super()._bind_runtime_views(**kwargs)

    def _bind_context_tile(self, query_lengths: list[int]) -> None:
        geometry = (self.num_sparse_topk, tuple(query_lengths))
        if getattr(self, "_csa2_context_geometry", None) == geometry:
            return
        # A failed partial update must not retain the previous successful key.
        self._csa2_context_geometry = None
        # Physical source causality is already encoded in selected rows. The
        # context kernel still applies a causal upper bound in staged-column
        # coordinates: give even the first query the full sparse capacity,
        # otherwise short source prefixes would clip valid compressed keys.
        kv_lengths = [self.num_sparse_topk + length - 1 for length in query_lengths]
        requests = len(query_lengths)
        query = torch.tensor(query_lengths, dtype=torch.int32, device="cpu")
        kv = torch.tensor(kv_lengths, dtype=torch.int32, device="cpu")
        self.query_lens_host[:requests].copy_(query)
        self._copy_host("context_query_lens", self.query_lens_device[:requests], query)
        self._seq_lens = self.query_lens_host[:requests]
        self._seq_lens_cuda = self.query_lens_device[:requests]
        self._num_contexts = requests
        self._num_ctx_tokens = self._num_tokens = sum(query_lengths)
        self._num_generations = 0
        self.kv_lens[:requests].copy_(kv)
        self._copy_host("context_kv_lens", self.kv_lens_cuda[:requests], kv)
        # THOP interprets context_lengths as Q lengths, independently of
        # past/current KV lengths. Keep them consistent with unfolded cuQ.
        self.prompt_lens_cpu[:requests].copy_(query)
        self._copy_host("context_prompt_lens", self.prompt_lens_cuda[:requests], query)
        self.host_request_types[:requests].zero_()
        self.host_total_kv_lens.zero_()
        self.host_total_kv_lens[0] = sum(kv_lengths)
        self.max_seq_len = max(self.num_sparse_topk, max(kv_lengths))
        self._copy_host(
            "context_cu_q",
            self.prepared_cu_q[: requests + 1],
            torch.cat((query.new_zeros(1), query.cumsum(0).int())),
        )
        self._copy_host(
            "context_cu_kv",
            self.prepared_cu_kv[: requests + 1],
            torch.cat((kv.new_zeros(1), kv.cumsum(0).int())),
        )
        self.cu_q_seqlens = self.prepared_cu_q[: requests + 1]
        self.cu_kv_seqlens = self.prepared_cu_kv[: requests + 1]
        self._bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda[:requests],
            kv_lens=self.kv_lens[:requests],
            prompt_lens_cuda=self.prompt_lens_cuda[:requests],
            prompt_lens_cpu=self.prompt_lens_cpu[:requests],
            host_request_types=self.host_request_types[:requests],
        )
        self._csa2_context_geometry = geometry

    def _stage_scratch(self, source, group, count: int, field: str = "serial"):
        """Compaction scratch for a staging group and whether this forward must refill it.

        Layers sharing a KV owner and an index source select the same rows,
        so the first layer of the group in a forward compacts and the others
        only decode. The scratch and the epoch live on the ``source`` request
        metadata (query tiles are separate objects); ``None`` groups always
        compact. A capture records its own compaction so replays follow the
        live selections. ``field`` names the per-forward record checked and
        updated: ``"serial"`` for the compaction, ``"scale_serial"`` for the
        group's recorded main-row scale bits (``main_bits``).
        """
        groups = source.__dict__.setdefault("_csa2_stage_groups", {})
        key = (group, self.prepared_lens.shape[0], self.num_sparse_topk)
        capturing = torch.cuda.is_current_stream_capturing()
        frame = (id(source), id(source.kv_cache_manager), getattr(source, "_csa2_stage_epoch", 0))
        entry = groups.get(key)
        if entry is None:
            if capturing:
                raise RuntimeError("Warm up CSA2 staging groups before graph capture")
            # A new geometry or phase offset retires obsolete eager scratch.
            # Keep this forward's groups for layers that reuse their selections,
            # plus every graph reservation shared by shallow metadata clones.
            for previous, scratch in tuple(groups.items()):
                serials = (scratch["serial"], scratch["scale_serial"])
                used = any(serial is not None and serial[:3] == frame for serial in serials)
                if not scratch["graph_reserved"] and not used:
                    del groups[previous]
            capacity, device = self.prepared_lens.shape[0], self.prepared_lens.device
            entry = groups[key] = {
                "tags": torch.empty(
                    (capacity, self.num_sparse_topk), dtype=torch.int32, device=device
                ),
                "counts": torch.empty(capacity, dtype=torch.int32, device=device),
                "refs": torch.empty(
                    (capacity, self.num_sparse_topk), dtype=torch.int32, device=device
                ),
                "own_blocks": torch.empty(capacity, dtype=torch.int32, device=device),
                "row_slots": torch.empty(
                    (capacity, self.num_sparse_topk), dtype=torch.int32, device=device
                ),
                "main_slots": torch.empty(
                    (capacity, max(self.num_sparse_topk - _SWA_TILE, 1)),
                    dtype=torch.int64,
                    device=device,
                ),
                "main_bits": torch.zeros(8, dtype=torch.int32, device=device),
                "serial": None,
                "scale_serial": None,
                "graph_reserved": False,
            }
        entry["graph_reserved"] |= self.is_cuda_graph or source.is_cuda_graph or capturing
        serial = (*frame, count, capturing)
        compact = group is None or entry[field] != serial
        entry[field] = serial
        return entry, compact

    def derive_stage_kv_scales(
        self,
        inputs: CSA2BackendForwardArgs,
        *,
        q: torch.Tensor | None = None,
        q_shared: bool = False,
        minimum: int = 0,
        maximum: int = 126,
        q_word: torch.Tensor | None = None,
        q_ready: torch.cuda.Event | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive this forward's range-aware staging KV scale from the packed pools.

        Staging expands the persistent per-group scales back to real magnitudes,
        so a unit per-tensor scale silently clamps every channel above the E4M3
        maximum of 448 -- the saturation that keeps native FP8 staging opt-in.
        Both persistent encoders clamp their payload, so the largest group scale
        of the selected rows yields an exact ceiling on what staging can decode,
        and the smallest power of two mapping that ceiling onto 448 is the tightest
        non-saturating scale. Reading only the scale bytes costs one extra pass
        over 32 or 16 bytes per row rather than its full payload.

        Returns ``(kv_scale_orig_quant, kv_scale_quant_orig)`` as metadata-owned
        device buffers. ``minimum`` and ``maximum`` bound the exponent, and the
        defaults only ever widen the range: shrinking it is an optimization, while
        saturation is a correctness problem, and an exponent of zero reproduces the
        historical unit scale byte for byte. ``q`` is the query tensor whose amax
        caps the exponent, because Q is scaled alongside KV and must not trade KV
        saturation for its own loss; ``q_shared`` selects which way it is scaled.
        Generation gives Q the reciprocal scale so BMM1's dequant product stays one,
        and the cap keeps Q inside the E4M3 range. The native context path cannot --
        the C++ op points ``dequant_scale_q`` and ``dequant_scale_kv`` at one tensor
        (``attentionOp.cpp:1917-1919``) -- so it passes ``q_shared``: Q is divided by
        this scale, which cannot saturate it and only costs resolution underneath, so
        the cap holds Q's amax at unit magnitude instead. See ``CSA2TrtllmAttention``
        for why the two phases can afford different exponents. ``q_word`` (int32)
        stands in for ``q`` when Q's producer already raised it to the FP32 bits of Q's
        amax; the derivation consumes and clears it, waiting for ``q_ready`` only
        before its finalize.
        """
        from .kernel import derive_stage_kv_scales as _derive_stage_kv_scales
        from .quantization import row_bytes

        if self.swa_pool.dtype != torch.float8_e4m3fn:
            raise ValueError("CSA2 staging KV scales require an E4M3 staging pool")
        if inputs.swa_pool is None or inputs.swa_indices is None:
            raise ValueError("CSA2 requires selected SWA pool inputs")
        sources = [(inputs.swa_pool, inputs.swa_indices, "swa")]
        if inputs.topk_indices is not None:
            if inputs.main_pool is None:
                raise ValueError("CSA2 selected main indices require a main pool")
            sources.append((inputs.main_pool, inputs.topk_indices, "main"))
        for pool, slots, cache_format in sources:
            if (
                pool.ndim != 2
                or pool.dtype != torch.uint8
                or pool.shape[1] != row_bytes(_HEAD_DIM, cache_format)
                or pool.device != self.swa_pool.device
                or not pool.is_cuda
                or pool.stride(1) != 1
            ):
                raise ValueError(
                    "CSA2 derived staging scales need CUDA packed rows with unit column stride"
                )
            if slots.dtype not in (torch.int32, torch.int64) or slots.device != pool.device:
                raise ValueError("CSA2 gather slots must be integer tensors on the pool device")
        if q is not None and q.device != self.swa_pool.device:
            raise ValueError("CSA2 staging scales need Q on the pool device")
        if self.shared_plan is not None:
            if q_word is not None:
                raise ValueError("CSA2 shared staging derives Q's amax from Q itself")
            # Shared staging reduces each marked bank row once, after compaction
            # marks them and before they are decoded (``stage_shared_rows``).
            self._csa2_deferred_scale = (q, q_shared, minimum, maximum)
            return self.stage_kv_scales
        main_bits, reuse_main = None, False
        if inputs.stage_group is not None and len(sources) > 1:
            # The layers of a staging group select the same main rows, so the
            # group's first layer in a forward records their scale bits and the
            # others reuse them.
            source = inputs.state.metadata if inputs.state is not None else None
            scratch, record = self._stage_scratch(
                source or self, inputs.stage_group, self.num_tokens, "scale_serial"
            )
            main_bits, reuse_main = scratch["main_bits"], not record
        _derive_stage_kv_scales(
            sources,
            self._csa2_stage_scale_bitmap,
            self._csa2_stage_kv_dequant,
            self._csa2_stage_kv_quant,
            main_mapping=inputs.main_mapping,
            q=q,
            q_shared=q_shared,
            minimum=minimum,
            maximum=maximum,
            main_bits=main_bits,
            reuse_main=reuse_main,
            q_word=q_word,
            q_ready=q_ready,
        )
        return self.stage_kv_scales

    def stage_selected(
        self,
        inputs: CSA2BackendForwardArgs,
        *,
        kv_scale_orig_quant=None,
        kv_scale_quant_orig=None,
        q_scale_quant_orig=None,
        softmax_scale=512**-0.5,
        main_logical=None,
        dedup=False,
    ) -> None:
        """Refresh bounded native staging pools and indices for one selected query tile.

        This prepares compute buffers only; the backend publishes native ABI
        arguments and performs attention after staging completes. ``dedup`` lets a
        query reference rows its request's anchor query stages; only consumers that
        read rows through the published indices may ask for it.
        """
        from .quantization import _fused_gather_supported, gather_rows, row_bytes

        count = self.num_tokens
        fp8 = self.swa_pool.dtype == torch.float8_e4m3fn
        if fp8:
            kv_scale_orig_quant = (
                self.native_unit_scale if kv_scale_orig_quant is None else kv_scale_orig_quant
            )
            kv_scale_quant_orig = (
                self.native_unit_scale if kv_scale_quant_orig is None else kv_scale_quant_orig
            )
            # Q keeps its own dequant scale: BMM1 folds dq_q * dq_kv, so a KV
            # scale chosen for the KV range must not move Q's own range.
            q_scale_quant_orig = (
                self.native_unit_scale if q_scale_quant_orig is None else q_scale_quant_orig
            )
            for scale in (kv_scale_orig_quant, kv_scale_quant_orig, q_scale_quant_orig):
                if (
                    scale.dtype != torch.float32
                    or scale.device != self.swa_pool.device
                    or scale.numel() != 1
                    or not scale.is_contiguous()
                ):
                    raise ValueError("CSA2 FP8 scales must be contiguous device FP32 scalars")
        if inputs.swa_pool is None or inputs.swa_indices is None:
            raise ValueError("CSA2 requires selected SWA pool inputs")
        if inputs.swa_pool.device != self.swa_pool.device:
            raise ValueError("CSA2 Q, packed pools and metadata must be on the same CUDA device")
        if (
            inputs.swa_indices.ndim != 2
            or inputs.swa_indices.shape[0] != count
            or inputs.swa_indices.shape[1] > _SWA_TILE
        ):
            raise ValueError("CSA2 SWA indices must match the query count and window <=128")
        if inputs.topk_indices is not None:
            if inputs.main_pool is None or inputs.main_pool.device != self.swa_pool.device:
                raise ValueError(
                    "CSA2 selected main indices require a main pool on the query device"
                )
            if (
                inputs.topk_indices.ndim != 2
                or inputs.topk_indices.shape[0] != count
                or inputs.topk_indices.shape[1] > self.num_sparse_topk - _SWA_TILE
            ):
                raise ValueError("CSA2 selected main indices exceed metadata geometry")
        # Preserve gather_rows validation even for zero-width selections; an
        # absent main selection continues to ignore an unused main pool.
        sources = [(inputs.swa_pool, inputs.swa_indices, "swa")]
        if inputs.topk_indices is not None:
            sources.append((inputs.main_pool, inputs.topk_indices, "main"))
        for pool, slots, cache_format in sources:
            if (
                pool.ndim != 2
                or pool.dtype != torch.uint8
                or pool.shape[1] != row_bytes(_HEAD_DIM, cache_format)
            ):
                raise ValueError("CSA2 gathered cache has the wrong row shape or dtype")
            if slots.dtype not in (torch.int32, torch.int64) or slots.device != pool.device:
                raise ValueError("CSA2 gather slots must be integer tensors on the pool device")
        mapping = inputs.main_mapping
        if self.shared_plan is not None:
            if any(not pool.is_cuda or pool.stride(1) != 1 for pool, _, _ in sources):
                raise ValueError(
                    "CSA2 shared staging requires CUDA packed rows with unit column stride"
                )
            if inputs.topk_indices is not None and (
                not isinstance(main_logical, torch.Tensor)
                or main_logical.shape != inputs.topk_indices.shape
                or main_logical.dtype not in (torch.int32, torch.int64)
                or main_logical.device != inputs.topk_indices.device
            ):
                raise ValueError("CSA2 shared MAIN logical indices must match selected indices")
            from .kernel import stage_shared_rows

            # Layers sharing a KV owner and an index source select the same bank
            # rows, so only the group's first layer of a forward compacts. Every
            # forward and prepare publishes a fresh ``csa2_indices``.
            source = inputs.state.metadata if inputs.state is not None else None
            serial = None
            if source is not None and inputs.stage_group is not None:
                serial = (
                    inputs.stage_group,
                    id(source.csa2_indices),
                    getattr(source, "_csa2_stage_epoch", 0),
                    count,
                )
            compact = serial is None or serial != getattr(self, "_csa2_shared_serial", None)
            self._csa2_shared_serial = serial
            self._csa2_shared_parity = 0 if compact else 1 - self._csa2_shared_parity
            stage_shared_rows(
                inputs.swa_pool,
                inputs.swa_indices,
                inputs.main_pool,
                None if mapping is not None else inputs.topk_indices,
                main_logical,
                self,
                self.shared_plan,
                main_mapping=mapping,
                kv_scale_orig_quant=kv_scale_orig_quant,
                kv_scale_quant_orig=kv_scale_quant_orig,
                q_scale_quant_orig=q_scale_quant_orig,
                softmax_scale=softmax_scale,
                scale=self.__dict__.pop("_csa2_deferred_scale", None),
                compact=compact,
                parity=self._csa2_shared_parity,
            )
            return
        native = (
            count > 0
            and all(pool.is_cuda and pool.stride(1) == 1 for pool, _, _ in sources)
            and _fused_gather_supported(inputs.swa_pool.device.index)
        )
        if native and self.num_sparse_topk <= 4096:
            from .kernel import stage_selected_rows

            source = inputs.state.metadata if inputs.state is not None else None
            scratch, compact = self._stage_scratch(source or self, inputs.stage_group, count)
            main_slots = inputs.topk_indices
            if mapping is not None and main_slots is not None:
                main_slots = scratch["main_slots"][:count, : main_slots.shape[1]]
            stage_selected_rows(
                inputs.swa_pool,
                inputs.swa_indices,
                inputs.main_pool,
                main_slots,
                self,
                scratch,
                compact=compact,
                main_logical=inputs.topk_indices if mapping is not None else None,
                main_mapping=mapping,
                kv_scale_orig_quant=kv_scale_orig_quant,
                kv_scale_quant_orig=kv_scale_quant_orig,
                q_scale_quant_orig=q_scale_quant_orig,
                softmax_scale=softmax_scale,
                dedup=dedup,
            )
            return
        topk_indices = inputs.topk_indices
        if mapping is not None and topk_indices is not None:
            table, page_size, max_positions, requests, visible = mapping
            topk_indices = self._map_global_slots(
                topk_indices, requests.long(), table, visible, page_size, max_positions
            )
        swa = gather_rows(inputs.swa_pool, inputs.swa_indices, _HEAD_DIM, "swa")
        swa_valid = (inputs.swa_indices >= 0) & (inputs.swa_indices < inputs.swa_pool.shape[0])
        extra = extra_valid = None
        if topk_indices is not None:
            extra = gather_rows(inputs.main_pool, topk_indices, _HEAD_DIM, "main")
            extra_valid = (topk_indices >= 0) & (topk_indices < inputs.main_pool.shape[0])
        self.prepared_counter.zero_()
        if fp8:
            bmm1 = q_scale_quant_orig * kv_scale_quant_orig * softmax_scale
            self.mla_bmm1_scale[:1].copy_(bmm1)
            self.mla_bmm1_scale[1:].copy_(bmm1 * 1.4426950408889634)
            self.mla_bmm2_scale.copy_(kv_scale_quant_orig)
        if extra is not None:
            if extra_valid is None:
                raise ValueError("Extra KV rows require a validity mask")
            rows = torch.cat((swa, extra), dim=1)
            valid = torch.cat((swa_valid, extra_valid), dim=1)
        else:
            rows, valid = swa, swa_valid
        # TG uses a dense valid prefix, split at slot 128 between its pools.
        # Compact the selected union before staging: -1 slots inside the
        # supplied extent can contribute zero logits in BF16 generation.
        # Physical source ownership no longer matters after dequantization.
        width = rows.shape[1]
        positions = torch.arange(width, device=swa.device).expand(count, -1)
        order = torch.where(valid, positions, width).argsort(dim=1, stable=True)
        packed = rows.gather(1, order[..., None].expand(-1, -1, _HEAD_DIM))
        lengths = valid.sum(1, dtype=torch.int32)
        packed = torch.where((positions < lengths[:, None])[..., None], packed, 0)
        # Zero selected rows reduce to the sink's zero value. Give TG one
        # zero KV row so it always launches a defined (nonempty) reduction.
        lengths = lengths.clamp_min(1)
        self.prepared_lens[:count].copy_(lengths)
        if fp8 and packed.numel():
            with torch.cuda.device(packed.device):
                packed, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    packed.contiguous(), kv_scale_quant_orig
                )
        self.swa_pool[:count].zero_()
        self.extra_pool[:count].zero_()
        swa_count = min(width, _SWA_TILE)
        self.swa_pool[:count, :swa_count].copy_(packed[:, :swa_count])
        if width > _SWA_TILE:
            self.extra_pool[:count, : width - _SWA_TILE].copy_(packed[:, _SWA_TILE:])
        indices = self.prepared_indices[:count]
        offsets = torch.arange(count, device=swa.device)[:, None]
        swa_positions = torch.arange(_SWA_TILE, device=swa.device)[None, :]
        indices[:, :_SWA_TILE].copy_(
            torch.where(swa_positions < lengths[:, None], offsets * _SWA_TILE + swa_positions, -1)
        )
        extra_capacity = self.num_sparse_topk - _SWA_TILE
        if extra_capacity:
            extra_positions = torch.arange(extra_capacity, device=swa.device)[None, :]
            indices[:, _SWA_TILE:].copy_(
                torch.where(
                    extra_positions + _SWA_TILE < lengths[:, None],
                    offsets * extra_capacity + extra_positions,
                    -1,
                )
            )

    def set_source_batch(self, seq_lengths: list[int], start_positions: list[int]) -> None:
        """Set encoder source rows independently of decoder query rows.

        This host-side input is consumed by the next prepare call. The caller
        must provision the manager for these source lengths before preparing.
        """
        if len(seq_lengths) != len(start_positions) or any(
            x < 0 for x in seq_lengths + start_positions
        ):
            raise ValueError("CSA2 source lengths and positions must be nonnegative and paired")
        self._csa2_source_batch = (list(seq_lengths), list(start_positions))

    def set_swa_bounded_replay(
        self, cached_prefix_lengths: list[int | None], *, decoder: bool = False
    ) -> None:
        """Prepare approximate reconstruction after an authoritative GLOBAL hit.

        The caller must allocate the manager's reported replay intervals and
        supply per-layer inputs for those absolute query positions. This does
        not perform prefix matching or change native V2 persistence policy.
        Encoder replay may include a new suffix; decoder replay assumes all
        prompt GLOBAL entries are ready and only reconstructs the last window.
        """
        if any(length is not None and length < 0 for length in cached_prefix_lengths):
            raise ValueError("CSA2 cached GLOBAL prefix lengths must be nonnegative")
        if decoder and any(length is None for length in cached_prefix_lengths):
            raise ValueError("Decoder replay requires a GLOBAL prefix for every request")
        if hasattr(self, "_csa2_source_batch"):
            raise ValueError(
                "Bounded replay source selection cannot be combined with set_source_batch"
            )
        self._csa2_pending_replay = (tuple(cached_prefix_lengths), decoder)

    def prepare_context_replay(
        self, requests: list[LlmRequest], *, allow_final_window: bool = True
    ) -> None:
        """Resolve recovery sources and read floors before preparing model inputs."""
        from .....pyexecutor.ced_replay import EncoderCheckpoint, EncoderReplay

        # Graph selection may vary between chunks. Keep every chunk's Decoder
        # history when a later chunk can enter a graph that expects full inputs.
        self.decoder_context_ends = (
            tuple(request.prompt_len for request in requests) if allow_final_window else ()
        )
        prefixes = [None] * self.num_seqs
        floors = [0] * self.num_seqs
        starts = self.kv_cache_params.num_cached_tokens_per_seq
        lengths = self.seq_lens.tolist()
        for row, request in enumerate(requests):
            plan = request.py_ced_replay
            if isinstance(plan, (EncoderReplay, EncoderCheckpoint)):
                cache = self.kv_cache_manager.kv_cache_map.get(request.py_request_id)
                if cache is None or plan.cache_identity != id(cache) or not cache.is_active:
                    raise ValueError("Encoder recovery has lost its active Global claim")
            if isinstance(plan, EncoderReplay):
                floors[row] = plan.start
                if not plan.consumed:
                    if (
                        plan.request_id != request.py_request_id
                        or starts[row] != plan.start
                        or plan.global_end != request.context_current_position
                        or lengths[row] != plan.num_tokens + request.context_chunk_size
                    ):
                        raise ValueError("Encoder recovery input does not match its Global claim")
                    prefixes[row] = plan.global_end
        if any(prefix is not None for prefix in prefixes):
            self.set_swa_bounded_replay(prefixes)
        self._csa2_pending_swa_floors = floors

    def set_decoder_query_boundary(self, first_layer: int) -> None:
        """Prepare only decoder working pages after full GLOBAL publication.

        Encoder pages before the current chunk may already be reclaimed. They
        are not dependencies of this query-only pass and must not be reserved
        again merely to build metadata for the longer decoder interval.
        """
        layout = self.kv_cache_manager.layout
        if (
            first_layer not in self.csa2_precomputed_kv_layers
            or first_layer != max(layout.kv_source_layer_ids)
            or layout.compress_ratios[first_layer] != 1
        ):
            raise ValueError(
                "Decoder query preparation requires the final precomputed GLOBAL owner"
            )
        if hasattr(self, "_csa2_pending_replay") or hasattr(self, "_csa2_source_batch"):
            raise ValueError("Decoder query preparation cannot override pending source selection")
        ends = [
            start + length
            for start, length in zip(
                self.kv_cache_params.num_cached_tokens_per_seq, self.seq_lens.tolist()
            )
        ]
        self.set_source_batch([0] * len(ends), ends)
        self._csa2_pending_query_layer_start = first_layer

    def select_global_source(
        self,
        owner: int,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select a declared source base; never infer it from tensor lengths."""
        mode = self.csa2_replay_mode
        if mode is None:
            return hidden_states if global_hidden_states is None else global_hidden_states
        if global_hidden_states is not None:
            raise ValueError(
                "CSA2 bounded replay uses query inputs or already prepared GLOBAL cache, not an external source batch"
            )
        indices = self.csa2_global_source_indices[owner]
        if indices.numel() == 0:
            return hidden_states[:0]
        if hidden_states.shape[0] != self.csa2_positions.numel():
            raise ValueError("CSA2 replay source selectors address the full packed query input")
        return hidden_states.index_select(0, indices)

    def _swa_replay_geometry(self, prefixes, decoder, starts, lengths) -> tuple:
        owner_shapes = []
        for owner in self.kv_cache_manager.layout.kv_source_layer_ids:
            ratio = self.kv_cache_manager.layout.compress_ratios[owner]
            source_lengths = tuple(
                length
                if prefix is None
                else (
                    0 if decoder else max(0, start + length - max(start, prefix - prefix % ratio))
                )
                for prefix, start, length in zip(prefixes, starts, lengths)
            )
            owner_shapes.append((owner, source_lengths, sum(source_lengths)))
        return ("decoder" if decoder else "encoder", tuple(lengths), tuple(owner_shapes))

    @contextmanager
    def defer_cuda_graph_decode_outputs(self):
        """Engine-only scope for a selected decode graph's preparation.

        The engine must run ``on_update_kv_lens`` before any model consumer,
        including warmup and capture, and after any acceptance correction of
        ``kv_lens_cuda``. Direct ``prepare`` callers retain fully initialized
        outputs; graph ownership alone is not sufficient proof.
        """
        previous = self._csa2_defer_decode_outputs
        self._csa2_defer_decode_outputs = True
        try:
            yield
        finally:
            self._csa2_defer_decode_outputs = previous

    def prepare(self) -> None:
        from .cache_manager import CSA2CacheManager

        self._csa2_deferred_decode_outputs = False
        self._csa2_context_geometry = None
        if self.is_cuda_graph and hasattr(self, "_csa2_replay_capture_signature"):
            pending = getattr(self, "_csa2_pending_replay", None)
            signature = (
                None
                if pending is None
                else self._swa_replay_geometry(
                    pending[0],
                    pending[1],
                    self.kv_cache_params.num_cached_tokens_per_seq,
                    self.seq_lens.tolist(),
                )
            )
            if signature != self._csa2_replay_capture_signature:
                raise ValueError(
                    "CSA2 replay mode/source geometry changed; use fresh graph metadata and recapture"
                )
        with torch.cuda.device(self.kv_lens_cuda.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare CSA2 request metadata before CUDA Graph capture")
        self._csa2_ready_for_kv_update = False
        manager = self.kv_cache_manager
        scope = (
            manager._swa_publication_scope()
            if isinstance(manager, CSA2CacheManager)
            else nullcontext(None)
        )
        # Model-driven metadata leaves an eager context batch's CSA2 prepare, block-offset
        # copy included, to its first CSA2 layer (_prepare_deferred).
        defer = (
            self._csa2_model_driven
            and isinstance(manager, CSA2CacheManager)
            and 0 < self.num_contexts == len(self.seq_lens)
            and not self.is_cuda_graph
            and not hasattr(self, "_csa2_pending_replay")
            and not hasattr(self, "_csa2_source_batch")
            and getattr(self, "csa2_remote_tail_mode", None) is None
        )
        self._csa2_pending_prepare = None
        if defer:
            # The batch's block offsets are copied with the deferred prepare.
            manager._deferred_block_offsets = []
            try:
                super().prepare()
            finally:
                block_offsets, manager._deferred_block_offsets = (
                    manager._deferred_block_offsets,
                    None,
                )
            # (block-offset copies, whether on_update_kv_lens ran meanwhile)
            self._csa2_pending_prepare = (block_offsets, False)
            return
        with scope as publication:
            super().prepare()
            self.prepare_csa2(swa_publication=publication)

    def _get_csa2_buffer(self, key, shape, dtype, *, like=None) -> torch.Tensor:
        # Preserve manager-local, shape-keyed addresses across graph replays.
        if not hasattr(self, "_csa2_buffers"):
            self._csa2_buffers = {}
        cache_key = (key, tuple(shape), dtype, self.is_cuda_graph)
        if not self.is_cuda_graph:
            # Eager buffers retain only the latest shape per name. Track that
            # shape per name so a prepare with many keys does not rescan the
            # whole bank for every lookup.
            latest = getattr(self, "_csa2_eager_latest", None)
            if latest is None or latest[0] is not self._csa2_buffers:
                # Shallow graph-metadata copies share the bank and its index.
                latest = self._csa2_eager_latest = (self._csa2_buffers, {})
            latest = latest[1]
            previous = latest.get(key)
            if previous is not None and previous != cache_key:
                self._csa2_buffers.pop(previous, None)
            latest[key] = cache_key
        result = self._csa2_buffers.get(cache_key)
        if result is None:
            result = (
                torch.empty_like(like, device=self.kv_lens_cuda.device)
                if like is not None
                else torch.empty(shape, dtype=dtype, device=self.kv_lens_cuda.device)
            )
            self._csa2_buffers[cache_key] = result
        return result

    def _copy_host(self, key: str, destination: torch.Tensor, value: torch.Tensor) -> None:
        """Async host-to-device copy through this metadata's pinned staging ring."""
        from tensorrt_llm._torch.host_staging import copy_host_to_device

        staging = getattr(self, "_csa2_host_staging", None)
        if staging is None:
            staging = self._csa2_host_staging = {}
        copy_host_to_device(staging, key, destination, value)

    def _copy_csa2_tensor(self, key: str, value) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(np.ascontiguousarray(value))
        result = self._get_csa2_buffer(key, value.shape, value.dtype, like=value)
        if value.device.type == "cpu":
            self._copy_host(key, result, value)
        else:
            result.copy_(value, non_blocking=True)
        return result

    @staticmethod
    def _swa_write_pages(starts, lengths, floors, allocated, block, table_shape):
        """Flat ``(requests, columns)`` of every SWA page the query intervals write.

        Raises when an interval is negative, starts below its replay floor or
        leaves the ``[requests, pages]`` table; the caller checks that the
        pages are mapped.
        """
        starts = np.asarray(starts, dtype=np.int64)
        lengths = np.asarray(lengths, dtype=np.int64)
        if np.any(lengths < 0):
            raise ValueError("CSA2 query/replay interval lengths must be nonnegative")
        # Positions beyond the allocated capacity are the overlap scheduler's
        # reservation and stay unmapped.
        lengths = np.minimum(lengths, np.maximum(0, np.asarray(allocated, dtype=np.int64) - starts))
        active = lengths > 0
        begin = starts // block
        end = (starts + lengths - 1) // block + 1
        if np.any(
            active
            & (
                (starts < np.maximum(0, np.asarray(floors, dtype=np.int64)))
                | (np.arange(starts.size) >= table_shape[0])
                | (end > table_shape[1])
            )
        ):
            raise ValueError(
                "CSA2 query/replay interval requires writable SWA pages; "
                "reserve the complete replay range before prepare"
            )
        return CSA2TrtllmMetadata._flat_page_ranges(begin, np.where(active, end - begin, 0))

    def _check_swa_write_pages(
        self, starts, lengths, floors, allocated_until, page_count, rows=None
    ):
        """Raise unless every SWA page the query intervals write is mapped.

        ``rows`` (int array) names the batch request of each interval when they
        are not the whole batch in order.
        """
        manager = self.kv_cache_manager
        write_rows, write_columns = self._swa_write_pages(
            starts,
            lengths,
            floors,
            allocated_until,
            manager.tokens_per_block,
            (len(lengths), page_count),
        )
        if rows is not None:
            write_rows = rows[write_rows]
        self._check_written_pages(
            manager.batch_pages_allocated(
                self._csa2_swa_specs, write_rows, write_columns, page_count
            )
        )

    def _check_owner_pages(
        self, owners, owner_starts, owner_ends, allocated_until, page_count, rows=None
    ):
        """Raise unless the GLOBAL output groups and compressor state rows inside
        the allocated capacity are mapped for ``owners``, which all read the same
        source rows. ``rows`` is as in ``_check_swa_write_pages``."""
        from .cache_manager import CSA2CacheRole

        manager = self.kv_cache_manager
        block = manager.tokens_per_block
        starts = np.asarray(owner_starts, dtype=np.int64)
        ends = np.asarray(owner_ends, dtype=np.int64)
        allocated_until = np.asarray(allocated_until, dtype=np.int64)
        by_ratio = {}
        for owner in owners:
            by_ratio.setdefault(manager.layout.compress_ratios[owner], []).append(owner)
        for ratio, ratio_owners in by_ratio.items():
            page_size = block // ratio
            group_starts = starts // ratio
            # Only groups inside the allocated capacity must be mapped.
            group_ends = np.minimum(ends // ratio, -(-allocated_until // ratio))
            group_first = group_starts // page_size
            group_rows, group_columns = self._flat_page_ranges(
                group_first,
                np.where(
                    group_ends > group_starts, (group_ends - 1) // page_size + 1 - group_first, 0
                ),
            )
            group_pages = (group_rows if rows is None else rows[group_rows], group_columns)
            specs = [(owner, CSA2CacheRole.GLOBAL) for owner in ratio_owners]
            if not np.all(manager.batch_pages_allocated(specs, *group_pages, page_count)):
                raise ValueError("CSA2 source output has no allocated GLOBAL page")
            if ratio != 2:
                continue
            state_first = (starts - starts % ratio) // block
            state_last = np.minimum(
                (np.minimum(ends, allocated_until) - 1) // block + 1, page_count
            )
            state_rows, state_columns = self._flat_page_ranges(
                state_first,
                np.where(ends > starts, np.maximum(state_last - state_first, 0), 0),
            )
            state_pages = (state_rows if rows is None else rows[state_rows], state_columns)
            specs = [
                (owner, role)
                for owner in ratio_owners
                for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE)
            ]
            if not np.all(manager.batch_pages_allocated(specs, *state_pages, page_count)):
                raise ValueError("CSA2 source rows have no allocated compressor state pages")

    @staticmethod
    def _flat_page_ranges(begin, counts):
        """Flat ``(rows, columns)`` covering columns ``[begin[r], begin[r] + counts[r])`` of each row."""
        rows = np.repeat(np.arange(counts.size), counts)
        columns = np.arange(rows.size) - np.repeat(np.cumsum(counts) - counts - begin, counts)
        return rows, columns

    @staticmethod
    def _check_written_pages(written) -> None:
        if not np.all(written):
            raise ValueError(
                "CSA2 query/replay interval requires writable SWA pages; "
                "reserve the complete replay range before prepare"
            )

    def _swa_table_descriptors(self, layers: int) -> torch.Tensor:
        """Device ``[layers, 4]`` int64 table of (pointer, row stride, rows, columns) per layer.

        Layers without a bound table (before a decoder-only query boundary)
        get an empty row, which resolves to invalid slots.
        """
        rows = []
        for layer in range(layers):
            table = self._csa2_swa_page_tables.get(layer)
            if table is None:
                rows.append((0, 0, 0, 0))
                continue
            if (
                table.dtype != torch.int32
                or table.ndim != 2
                or (table.shape[1] > 1 and table.stride(1) != 1)
            ):
                raise ValueError(
                    "CSA2 SWA page tables must be int32 [requests, pages] with unit column stride"
                )
            rows.append((table.data_ptr(), table.stride(0), table.shape[0], table.shape[1]))
        return self._device_descriptors("swa_table_desc", rows)

    def _device_descriptors(self, kind: str, rows) -> torch.Tensor:
        """Device int64 table of the host descriptor ``rows``, uploaded only when they change."""
        key = (kind, tuple(rows))
        cache = getattr(self, "_csa2_swa_table_cache", None)
        if cache is None or cache[0] is not self._csa2_buffers:
            # Shallow graph-metadata copies share the buffer bank; share the
            # descriptor index with it so every copy sees the same fills.
            cache = self._csa2_swa_table_cache = (self._csa2_buffers, {}, {})
        _, names, filled = cache
        if self.is_cuda_graph:
            # A captured launch reads its descriptor table at replay time, so
            # the content must never change after the fill: every distinct
            # binding (one per captured batch shape) owns its own buffer.
            name = names.setdefault(key, f"{kind}/{len(names)}")
        else:
            name = kind
        device_table = self._get_csa2_buffer(name, (len(rows), len(rows[0])), torch.int64)
        if filled.get(name) != (key, device_table.data_ptr()):
            self._copy_host(name, device_table, torch.tensor(rows, dtype=torch.int64, device="cpu"))
            filled[name] = (key, device_table.data_ptr())
        return device_table

    def _swa_ratio_tensor(self, ratios) -> torch.Tensor:
        key = tuple(int(r) for r in ratios)
        cached = getattr(self, "_csa2_swa_ratio_cache", None)
        device_ratios = self._get_csa2_buffer("swa_ratios", (len(key),), torch.int32)
        if cached is None or cached[0] != key or cached[1] is not device_ratios:
            self._copy_host("swa_ratios", device_ratios, torch.tensor(key, dtype=torch.int32))
            self._csa2_swa_ratio_cache = (key, device_ratios)
        return device_ratios

    def _ensure_swa_slots(self) -> None:
        """Resolve all layers' SWA slots and visible lengths from the bound page tables.

        Runs once per forward (one launch), like DeepSeek-V4's forward-time
        window conversion. ``reset_routing``, prepare and KV-length updates
        re-arm it, so a graph capture always records the launch.
        """
        if self._csa2_pending_prepare is not None:
            self._prepare_deferred()
        # None: not prepared by prepare_csa2.
        if getattr(self, "_csa2_swa_resolved", None) is not False:
            return
        from . import kernel

        manager = self.kv_cache_manager
        positions, requests = self.csa2_positions, self.csa2_token_requests
        layers, count, window = (
            # SWA layers are a prefix; a decoder-only pass binds its suffix.
            max(self._csa2_swa_page_tables, default=-1) + 1,
            positions.numel(),
            manager.layout.window_size,
        )
        num_layers = len(manager.layout.compress_ratios)
        reads = self._get_csa2_buffer("swa_reads", (layers, count, window), torch.int64)
        writes = self._get_csa2_buffer("swa_writes", (layers, count), torch.int64)
        visible = self._get_csa2_buffer("visible", (num_layers, count), torch.int64)
        self.csa2_swa_indices = dict(enumerate(reads.unbind(0)))
        self.csa2_swa_write_slots = dict(enumerate(writes.unbind(0)))
        self.csa2_visible_lengths = dict(enumerate(visible.unbind(0)))
        missing_slot = getattr(self, "_csa2_missing_swa_slot", None)
        if missing_slot is not None:
            # Context-only caches: layers without SWA pages read no window,
            # while visibility stays model-layer keyed for their GLOBAL KV.
            for layer in range(layers, num_layers):
                self.csa2_swa_indices[layer] = missing_slot.expand(count, window)
                self.csa2_swa_write_slots[layer] = missing_slot.expand(count)
        self._csa2_swa_resolved = True
        if count == 0:
            return
        floors = self.csa2_replay_start_positions
        read_floors = getattr(self, "_csa2_swa_read_floors", None)
        if read_floors is None:
            read_floors = floors
        descriptors = getattr(self, "_csa2_swa_descriptors", None)
        if positions.is_cuda and kernel.dsl_available() and descriptors is not None:
            kernel.refresh_swa_slots(
                positions,
                requests,
                floors,
                read_floors,
                descriptors[:layers],
                self._csa2_swa_ratios,
                reads,
                writes,
                visible,
                manager.tokens_per_block,
            )
            return
        # Layers before a decoder-only boundary bind no pages: they read and
        # write nothing, as the kernel's empty descriptors do.
        first = min(getattr(self, "csa2_query_layer_start", 0), layers)
        reads[:first].fill_(-1)
        writes[:first].fill_(-1)
        self._refresh_swa_tensor_outputs(
            positions,
            requests,
            floors,
            manager.tokens_per_block,
            window,
            tuple(
                (
                    self._csa2_swa_page_tables[layer],
                    reads[layer],
                    writes[layer],
                    visible[layer],
                    manager.layout.compress_ratios[layer],
                )
                for layer in range(first, layers)
            ),
            read_floors,
        )
        for layer in (*range(first), *range(layers, num_layers)):
            ratio = manager.layout.compress_ratios[layer]
            if ratio:
                torch.div(positions.long() + 1, ratio, rounding_mode="floor", out=visible[layer])
            else:
                visible[layer].zero_()

    @staticmethod
    def _refresh_swa_tensor_outputs(
        positions, requests, floors, block, window, layer_inputs, read_floors=None
    ):
        """PyTorch reference of the SWA refresh kernel (host metadata and tests)."""
        if not layer_inputs:
            return
        positions, requests = positions.long(), requests.long()
        logical = positions[:, None] - window + 1 + torch.arange(window, device=positions.device)
        columns = logical.clamp_min(0) // block
        offsets = logical.remainder(block)
        nonnegative = logical >= 0
        for pages, reads, writes, visible, ratio in layer_inputs:
            request_count = min(pages.shape[0], floors.numel())
            if request_count == 0 or pages.shape[1] == 0:
                reads.fill_(-1)
                writes.fill_(-1)
            else:
                safe_requests = requests.clamp(0, request_count - 1)
                physical = pages[safe_requests[:, None], columns.clamp(max=pages.shape[1] - 1)]
                valid = ((requests >= 0) & (requests < request_count))[:, None]
                valid = valid & (logical >= floors[safe_requests, None])
                valid &= nonnegative & (columns < pages.shape[1]) & (physical >= 0)
                slots = torch.where(valid, physical.long() * block + offsets, -1)
                writes.copy_(slots[:, -1])
                if read_floors is not None:
                    slots.masked_fill_(logical < read_floors[safe_requests, None], -1)
                reads.copy_(slots)
            if ratio:
                torch.div(positions + 1, ratio, rounding_mode="floor", out=visible)
            else:
                visible.zero_()

    def _bind_swa_pages(self, layer, pages, published_pages):
        # Select backing once for each existing shape/graph key. A direct or
        # draft fallback must update it, never redirect a captured consumer.
        if published_pages is not None and (
            published_pages.dtype != pages.dtype
            or published_pages.device != self.kv_lens_cuda.device
            or published_pages.shape != pages.shape
        ):
            published_pages = None
        key = (f"swa_pages/{layer}", tuple(pages.shape), pages.dtype, self.is_cuda_graph)
        if not hasattr(self, "_csa2_buffers"):
            self._csa2_buffers = {}
        if published_pages is not None and key not in self._csa2_buffers:
            self._csa2_buffers[key] = published_pages
        result = self._get_csa2_buffer(f"swa_pages/{layer}", pages.shape, pages.dtype, like=pages)
        if published_pages is None:
            result.copy_(pages, non_blocking=True)
        elif (
            result.data_ptr() != published_pages.data_ptr()
            or result.stride() != published_pages.stride()
        ):
            # Fallback-first preparation may already own a private allocation.
            result.copy_(published_pages, non_blocking=True)
        return result

    def prepare_csa2(self, *, swa_publication=None) -> None:
        """Resolve scheduler request IDs through the real manager's converters.

        Query and source metadata have independent packed lengths. All physical
        tables include the manager's sliding scratch mappings; no modulo ring
        addressing is synthesized here.
        """
        from . import kernel
        from .cache_manager import CSA2CacheManager, CSA2CacheRole

        self._csa2_deferred_decode_outputs = False
        self._csa2_ready_for_kv_update = False
        # Every prepared forward starts a new staging epoch: the first layer of
        # each staging group compacts again.
        self._csa2_stage_epoch = getattr(self, "_csa2_stage_epoch", 0) + 1
        manager = self.kv_cache_manager
        if not isinstance(manager, CSA2CacheManager):
            raise TypeError("CSA2 runtime metadata requires CSA2CacheManager")
        self.csa2_query_layer_start = getattr(self, "_csa2_pending_query_layer_start", 0)
        if hasattr(self, "_csa2_pending_query_layer_start"):
            del self._csa2_pending_query_layer_start
        if self.beam_width != 1 or self.is_spec_dec_dynamic_tree:
            raise NotImplementedError(
                "CSA2 supports contiguous-prefix verification, not beams or dynamic trees"
            )
        request_ids = list(self.request_ids)
        lengths = self.seq_lens.tolist()
        if manager.context_swa_layer_limit is not None and self.num_contexts != len(lengths):
            raise ValueError("CSA2 context-only cache cannot prepare generation requests")
        manager.validate_verification(lengths, self.num_contexts, self.is_spec_decoding_enabled)
        starts = list(self.kv_cache_params.num_cached_tokens_per_seq)
        if len(request_ids) != len(lengths) or len(starts) != len(lengths):
            raise ValueError("CSA2 requires one query length and cached length per request")
        replay = getattr(self, "_csa2_pending_replay", None)
        if hasattr(self, "_csa2_pending_replay"):
            del self._csa2_pending_replay
        self.csa2_replay_mode = None
        self.csa2_replay_cached_lengths = ()
        replay_starts = [0] * len(lengths)
        if replay is not None:
            prefixes, decoder = replay
            if len(prefixes) != len(lengths):
                raise ValueError(
                    "CSA2 bounded replay requires one prefix length per context request"
                )
            if hasattr(self, "_csa2_source_batch"):
                raise ValueError("CSA2 replay cannot use an unrelated external source batch")
            replay_starts = [
                0
                if prefix is None
                else manager.get_swa_replay_ranges([prefix], decoder=decoder)[0][0]
                for prefix in prefixes
            ]
            for row, (prefix, start, length) in enumerate(zip(prefixes, starts, lengths)):
                if prefix is None:
                    continue
                if row >= self.num_contexts:
                    raise ValueError("CSA2 bounded replay applies only to context requests")
                if start != replay_starts[row]:
                    raise ValueError(
                        "CSA2 replay queries must start at the reconstruction boundary"
                    )
                if start + length < prefix or (decoder and start + length != prefix):
                    raise ValueError("CSA2 replay query interval must cover its cached prefix tail")
            self.csa2_replay_mode = "decoder" if decoder else "encoder"
            self.csa2_replay_cached_lengths = prefixes
        if hasattr(self, "_csa2_pending_swa_floors"):
            replay_starts = [
                max(start, floor)
                for start, floor in zip(replay_starts, self._csa2_pending_swa_floors)
            ]
            del self._csa2_pending_swa_floors
        self.csa2_replay_signature = (
            None
            if replay is None
            else self._swa_replay_geometry(
                replay[0],
                replay[1],
                starts,
                lengths,
            )
        )
        external_source = hasattr(self, "_csa2_source_batch")
        # Only the engine's selected decode graph, plain or speculative, may leave derived
        # outputs to its captured preprocessing; source overrides and replay prepare fully.
        defer_outputs = (
            getattr(self, "_csa2_defer_decode_outputs", False)
            and self.is_cuda_graph
            and self.num_contexts == 0
            and bool(lengths)
            and replay is None
            and not external_source
            and getattr(self, "csa2_remote_tail_mode", None) is None
        )
        source_lengths, source_starts = getattr(self, "_csa2_source_batch", (lengths, starts))
        if external_source:
            del self._csa2_source_batch
        if len(source_lengths) != len(lengths):
            raise ValueError("CSA2 source and query batches must name the same requests")
        # Under the overlap scheduler, generation rows reserve the previous
        # step's full verification width; positions beyond the allocated
        # capacity stay unmapped and on_update_kv_lens narrows them later.
        allocated_until = np.full(len(lengths), manager.max_seq_len, dtype=np.int64)
        allocated_until[self.num_contexts :] = [
            manager.kv_cache_map[request_id].capacity
            for request_id in request_ids[self.num_contexts :]
        ]
        lengths_np = np.asarray(lengths, dtype=np.int64)
        starts_np = np.asarray(starts, dtype=np.int64)
        # Versions also catch compressor-state pages that change without a page crossing.
        page_versions = np.fromiter(
            (manager.kv_cache_map[request_id].page_storage_version for request_id in request_ids),
            dtype=np.uint64,
            count=len(request_ids),
        )
        snapshot = (
            tuple(request_ids),
            manager.request_epochs(request_ids),
            tuple(lengths),
            starts_np,
            allocated_until,
            lengths_np,
            page_versions,
        )
        block = manager.tokens_per_block
        page_count = (manager.max_seq_len + block - 1) // block
        if replay is None and not external_source:
            rows = self._steady_generation_step(snapshot)
            if rows is not None:
                # Same generation requests, each advanced by one committed
                # step: every geometry tensor of the previous prepare is still
                # valid, so only the host positions, the routing state and the
                # live device endpoints are refreshed, plus the page tables
                # when a request crossed a page. When requests left the batch,
                # the others moved rows: their start bases are uploaded again
                # and every table is re-read.
                moved = self._steady_pages_moved(snapshot, rows)
                remapped = not isinstance(rows, slice)
                refresh = remapped or moved.any()
                if refresh:
                    replay_starts_np = np.asarray(replay_starts, dtype=np.int64)
                if remapped:
                    uploads = _CoalescedUploads(self)
                    starts32 = starts_np.astype(np.int32)
                    self._csa2_query_base = uploads.copy("query_base", starts32)
                    uploads.copy("source_base", starts32)
                    self.csa2_replay_start_positions = uploads.copy(
                        "replay_starts", replay_starts_np
                    )
                    uploads.flush()
                if refresh:
                    self._refresh_steady_page_tables(
                        swa_publication,
                        request_ids,
                        starts_np,
                        np.where(moved, lengths_np, 0),
                        replay_starts_np,
                        allocated_until,
                        page_count,
                        page_versions,
                    )
                self.csa2_request_start_positions = tuple(starts)
                self.reset_routing()
                self._finish_prepare_csa2(snapshot, defer_outputs=defer_outputs)
                if not defer_outputs:
                    self._refresh_live_endpoints()
                return
        self._csa2_steady_snapshot = None
        self._csa2_steady_refresh = None
        if external_source:
            source_starts = np.asarray(source_starts, dtype=np.int64)
            source_lengths = np.asarray(source_lengths, dtype=np.int64)
        else:
            source_starts, source_lengths = starts_np, lengths_np
        ends = source_starts + source_lengths
        if np.any(ends > manager.max_seq_len):
            raise ValueError("CSA2 source rows exceed the manager context capacity")
        if np.any(
            (starts_np < 0) | (lengths_np < 0) | (starts_np + lengths_np > manager.max_seq_len)
        ):
            raise ValueError("CSA2 query positions exceed the manager context capacity")
        self.csa2_request_start_positions = tuple(starts)
        self.csa2_request_lengths = tuple(lengths)
        self.csa2_num_context_requests = self.num_contexts
        query_ends = np.cumsum(lengths_np)
        query_begins = query_ends - lengths_np
        packed_start = int(query_ends[-1]) if query_ends.size else 0
        query_ranges = tuple(zip(query_begins.tolist(), query_ends.tolist()))
        self.csa2_request_query_ranges = query_ranges
        uploads = _CoalescedUploads(self)
        copy = uploads.copy
        token_requests_np = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths_np)
        query_offsets_np = np.arange(packed_start, dtype=np.int64) - np.repeat(
            query_begins, lengths_np
        )
        positions_np = starts_np[token_requests_np] + query_offsets_np
        self.csa2_positions = (
            self._get_csa2_buffer("positions", (packed_start,), torch.int32)
            if defer_outputs
            else copy("positions", positions_np.astype(np.int32))
        )
        replay_starts_np = np.asarray(replay_starts, dtype=np.int64)
        self.csa2_replay_start_positions = copy("replay_starts", replay_starts_np)
        self._csa2_swa_read_floors = None
        self.csa2_global_source_indices = {}
        self.reset_routing()
        self.csa2_swa_indices = {}
        self.csa2_swa_write_slots = {}
        self.csa2_visible_lengths = {}
        self.csa2_main_write_slots = {}
        self.csa2_global_page_tables = {}
        self.csa2_global_page_sizes = {}
        self.csa2_global_max_positions = {}
        self.csa2_kv_sources = {}
        self._csa2_source_geometry = {}
        self._csa2_main_domain_counts = {}
        self._csa2_shared_tables = {}
        self._csa2_swa_page_tables = {}
        self._csa2_compression = {}
        self._csa2_compressed_positions = {}
        published_swa = manager._get_swa_publication(
            swa_publication, self.kv_cache_block_offsets, request_ids, self.num_contexts, page_count
        )
        if published_swa is None:
            manager.compute_batch_page_tables(
                request_ids, self.num_contexts, page_versions=page_versions.tolist()
            )
        # Owner slots are resolved from these host KV ends: a draft prepare
        # runs without the generic prepare, so kv_lens_cuda may hold live
        # lengths that differ from the host positions.
        prepared_ends = copy("prepared_kv_ends", (starts_np + lengths_np).astype(np.int32))
        # An eager context batch prepares the KV owner tables at its first layer with a KV owner.
        defer_owner_tables = (
            0 < self.num_contexts == len(lengths)
            and not self.is_cuda_graph
            and replay is None
            and not external_source
            and getattr(self, "csa2_remote_tail_mode", None) is None
        )
        owner_inputs = (
            request_ids,
            starts,
            lengths,
            source_starts,
            source_lengths,
            ends,
            allocated_until,
        )
        if not defer_owner_tables:
            self._prepare_owner_tables(copy, *owner_inputs)
        self.csa2_token_requests = copy("token_requests", token_requests_np)
        # Generation-row -> generation-request map for the sparse paged indexer kernels
        decode_start = (
            query_ranges[self.num_contexts][0]
            if self.num_contexts < len(query_ranges)
            else packed_start
        )
        decode_rows_np = (token_requests_np[decode_start:] - self.num_contexts).astype(np.int32)
        self.csa2_decode_row_requests = (
            copy("decode_row_requests", decode_rows_np) if decode_rows_np.size else None
        )
        self._csa2_query_base = copy("query_base", starts_np.astype(np.int32))
        self._csa2_query_lengths = copy("query_lengths", lengths_np.astype(np.int32))
        self._csa2_query_offsets = copy("query_offsets", query_offsets_np.astype(np.int32))
        # Bind SWA page tables; slots are resolved in _ensure_swa_slots.
        num_layers = len(manager.layout.compress_ratios)
        layers = _layout_layers(manager.layout)
        swa_layers = [i for i in range(num_layers) if manager.has_swa_cache(i)]
        # A decoder-only query pass binds no encoder pages: they may be reclaimed.
        first_layer = self.csa2_query_layer_start
        live_swa_layers = [i for i in swa_layers if i >= first_layer]
        swa_specs = self._csa2_swa_specs = [
            (layer_idx, CSA2CacheRole.SWA) for layer_idx in live_swa_layers
        ]
        self._check_swa_write_pages(
            starts_np, lengths_np, replay_starts_np, allocated_until, page_count
        )
        if published_swa is None:
            layer_pages = [manager.batch_page_table(spec, page_count) for spec in swa_specs]
            device_views = (None,) * len(live_swa_layers)
        else:
            local = published_swa.unbind(0)
            layer_pages = device_views = [local[manager.layer_offsets[i]] for i in live_swa_layers]
        for layer_idx, pages, device_pages in zip(live_swa_layers, layer_pages, device_views):
            self._csa2_swa_page_tables[layer_idx] = self._bind_swa_pages(
                layer_idx, pages, device_pages
            )
        # Context-only caches: layers without SWA pages share one -1 slot.
        self._csa2_missing_swa_slot = None
        if len(swa_layers) < num_layers:
            self._csa2_missing_swa_slot = self._get_csa2_buffer("missing_swa_slot", (), torch.int64)
            self._csa2_missing_swa_slot.fill_(-1)
        for layer_idx in range(first_layer, num_layers):
            self.csa2_kv_sources[layer_idx] = layers[layer_idx].kv_source
        if self.num_contexts:
            self._prepare_shared_tables(uploads.copy)
        if not defer_owner_tables:
            self._prepare_indexer_plans(uploads.copy)
        uploads.flush()
        self._csa2_pending_owner_tables = None
        if defer_owner_tables:
            self._csa2_owner_descriptors = None
            self._csa2_pending_owner_tables = (*owner_inputs, prepared_ends)
        else:
            self._resolve_owner_tables(prepared_ends, defer_outputs)
        self._csa2_swa_descriptors = self._csa2_swa_ratios = None
        if self.csa2_positions.is_cuda and kernel.dsl_available() and swa_layers:
            # Filled outside graph capture; the forward-time launch reads them.
            self._csa2_swa_descriptors = self._swa_table_descriptors(len(swa_layers))
            self._csa2_swa_ratios = self._swa_ratio_tensor(manager.layout.compress_ratios)
        self._csa2_swa_resolved = False
        if not hasattr(self, "csa2_precomputed_kv_layers"):
            self.csa2_precomputed_kv_layers = set()
            self.csa2_replay_query_rows = None
        # Only a plain generation prepare seeds the steady path: context,
        # replay, source-override and decoder-boundary geometry must be rebuilt.
        steady_seed = (
            not self.num_contexts
            and replay is None
            and not external_source
            and not self.csa2_query_layer_start
            and getattr(self, "csa2_remote_tail_mode", None) is None
        )
        self._finish_prepare_csa2(snapshot if steady_seed else None, defer_outputs=defer_outputs)

    def _prepare_owner_tables(
        self,
        copy,
        request_ids,
        starts,
        lengths,
        source_starts,
        source_lengths,
        ends,
        allocated_until,
    ) -> None:
        """Every KV owner's GLOBAL page tables, compressor batches and write buffers."""
        from .cache_manager import CSA2CacheRole
        from .compressor import CSA2CompressionBatch

        manager = self.kv_cache_manager
        block = manager.tokens_per_block
        page_count = (manager.max_seq_len + block - 1) // block
        query_ranges = self.csa2_request_query_ranges
        owners = manager.layout.kv_source_layer_ids
        if self.csa2_replay_mode is None:
            # Every owner reads the same source rows.
            shared_geometry = (
                copy("source_base", np.asarray(source_starts, dtype=np.int32)),
                copy("source_lengths", np.asarray(source_lengths, dtype=np.int32)),
            )
            shared_cu = copy(
                "source_cu", np.cumsum(np.concatenate(([0], source_lengths)), dtype=np.int32)
            )
            self._check_owner_pages(owners, source_starts, ends, allocated_until, page_count)
        if self.num_contexts:
            # Reserve every currently addressable logical row, including
            # upward live-end corrections within already published pages.
            # This is capacity only; selected indices retain visibility.
            columns = np.arange(page_count)
            allocated = manager.batch_pages_allocated(
                [(owner, CSA2CacheRole.GLOBAL) for owner in owners],
                np.repeat(np.arange(len(request_ids)), page_count),
                np.tile(columns, len(request_ids)),
                page_count,
            ).reshape(len(owners), len(request_ids), page_count)
            last_pages = np.where(allocated, columns + 1, 0).max(axis=2, initial=0).tolist()
        for index, owner in enumerate(owners):
            ratio = manager.layout.compress_ratios[owner]
            owner_starts, owner_lengths, owner_ends = source_starts, source_lengths, ends
            if self.csa2_replay_mode is not None:
                if self.csa2_replay_mode == "decoder":
                    owner_starts = list(self.csa2_replay_cached_lengths)
                    owner_lengths = [0] * len(lengths)
                    owner_ends = owner_starts
                    source_indices = []
                else:
                    owner_ends = [start + length for start, length in zip(starts, lengths)]
                    owner_starts = [
                        start if prefix is None else min(end, max(start, prefix - prefix % ratio))
                        for prefix, start, end in zip(
                            self.csa2_replay_cached_lengths, starts, owner_ends
                        )
                    ]
                    owner_lengths = [end - start for start, end in zip(owner_starts, owner_ends)]
                    source_indices = [
                        query_ranges[r][0] + position - starts[r]
                        for r, (start, end) in enumerate(zip(owner_starts, owner_ends))
                        for position in range(start, end)
                    ]
                self.csa2_global_source_indices[owner] = copy(
                    f"replay_source/{owner}",
                    torch.tensor(source_indices, dtype=torch.int64, device="cpu"),
                )
            self._csa2_source_geometry[owner] = (
                shared_geometry
                if self.csa2_replay_mode is None
                else (
                    copy(f"source_base/{owner}", np.asarray(owner_starts, dtype=np.int32)),
                    copy(f"source_lengths/{owner}", np.asarray(owner_lengths, dtype=np.int32)),
                )
            )
            capacity = int(np.sum(owner_lengths, dtype=np.int64))
            global_spec = (owner, CSA2CacheRole.GLOBAL)
            if self.num_contexts:
                self._csa2_main_domain_counts[owner] = tuple(
                    min(page * (block // ratio), manager.max_seq_len // ratio)
                    for page in last_pages[index]
                )
            if self.csa2_replay_mode is not None:
                for request_row, prefix in enumerate(self.csa2_replay_cached_lengths):
                    if prefix is None:
                        continue
                    prefix_pages = (prefix // ratio + block // ratio - 1) // (block // ratio)
                    if not np.all(
                        manager.batch_pages_allocated(
                            [global_spec],
                            np.full(prefix_pages, request_row),
                            np.arange(prefix_pages),
                            page_count,
                        )
                    ):
                        raise ValueError(
                            "CSA2 bounded replay requires the cached GLOBAL prefix pages to be ready"
                        )
            self.csa2_global_page_tables[owner] = copy(
                f"global_pages/{owner}", manager.batch_page_table(global_spec, page_count)
            )
            self.csa2_global_page_sizes[owner] = block // ratio
            self.csa2_global_max_positions[owner] = manager.max_seq_len // ratio
            if self.csa2_replay_mode is not None:
                self._check_owner_pages(
                    (owner,), owner_starts, owner_ends, allocated_until, page_count
                )
            # Compressed output groups; their write slots and positions are
            # resolved on the device after the uploads (_resolve_owner_slots).
            self.csa2_main_write_slots[owner] = self._get_csa2_buffer(
                f"writes/{owner}", (capacity,), torch.int64
            )
            self._csa2_compressed_positions[owner] = self._get_csa2_buffer(
                f"compressed_positions/{owner}", (capacity,), torch.int32
            )
            if ratio == 2:
                kv_spec = (owner, CSA2CacheRole.COMPRESSOR_KV)
                score_spec = (owner, CSA2CacheRole.COMPRESSOR_SCORE)
                kv_pages = copy(
                    f"kv_state_pages/{owner}", manager.batch_page_table(kv_spec, page_count)
                )
                score_pages = copy(
                    f"score_state_pages/{owner}", manager.batch_page_table(score_spec, page_count)
                )
                self._csa2_compression[owner] = CSA2CompressionBatch(
                    manager.get_buffers(owner, CSA2CacheRole.COMPRESSOR_KV),
                    manager.get_buffers(owner, CSA2CacheRole.COMPRESSOR_SCORE),
                    kv_pages,
                    score_pages,
                    # Ends, starts and compressed offsets: _resolve_owner_slots.
                    self._get_csa2_buffer(f"source_ends/{owner}", (len(lengths),), torch.int32),
                    self._get_csa2_buffer(f"source_starts/{owner}", (len(lengths),), torch.int32),
                    shared_cu
                    if self.csa2_replay_mode is None
                    else copy(f"source_cu/{owner}", np.cumsum([0, *owner_lengths], dtype=np.int32)),
                    self._get_csa2_buffer(
                        f"compressed_cu/{owner}", (len(lengths) + 1,), torch.int32
                    ),
                    capacity,
                    block,
                    max(1, int(np.max(owner_lengths, initial=0))),
                )

    def _resolve_owner_tables(self, prepared_ends, defer_outputs: bool) -> None:
        from . import kernel

        self._csa2_owner_descriptors = (
            self._owner_slot_descriptors()
            if self.csa2_positions.is_cuda and kernel.dsl_available()
            else None
        )
        if not defer_outputs:
            # A deferred decode graph resolves them in on_update_kv_lens.
            self._resolve_owner_slots(prepared_ends)

    def _prepare_pending_owner_tables(self) -> None:
        """Run the owner part of a deferred context prepare, uploaded in its own batch."""
        *owner_inputs, prepared_ends = self._csa2_pending_owner_tables
        self._csa2_pending_owner_tables = None
        uploads = _CoalescedUploads(self)
        self._prepare_owner_tables(uploads.copy, *owner_inputs)
        self._prepare_shared_tables(uploads.copy)
        self._prepare_indexer_plans(uploads.copy)
        uploads.flush()
        self._resolve_owner_tables(prepared_ends, False)

    def _steady_generation_step(self, snapshot):
        """Each request's row in the previous prepare when ``snapshot`` continues it by one
        committed step, or None.

        The same generation requests and cache lifetimes with the same query
        lengths, each start at most one query length after its previous start:
        verification commits up to its whole query, and padding rows do not
        move. The rows are ``slice(None)`` when the order is unchanged too; a
        batch of the same size and query lengths from which requests left (the
        others moved up, padding rows filled the end) maps each row to a previous
        row of its request. Positions and slots follow the live device lengths.
        """
        previous = getattr(self, "_csa2_steady_snapshot", None)
        if previous is None or self.num_contexts or snapshot[2] != previous[2]:
            return None
        # Graph copies of one metadata share the buffer bank by shape, so another
        # copy's prepare may have rewritten the tables this step would reuse.
        bank = getattr(self, "_csa2_buffers", {})
        if bank.get(_BANK_WRITER) != getattr(self, "_csa2_forward_serial", None):
            return None
        rows = slice(None)
        if snapshot[:2] != previous[:2]:
            # The query lengths are uniform, so a moved request keeps its length.
            # Padding rows repeat one request: any of its previous rows serves.
            if len(set(snapshot[2])) > 1:
                return None
            previous_rows = dict(zip(zip(previous[0], previous[1]), range(len(previous[0]))))
            try:
                rows = np.fromiter(
                    map(previous_rows.__getitem__, zip(snapshot[0], snapshot[1])),
                    dtype=np.int64,
                    count=len(snapshot[0]),
                )
            except KeyError:
                return None
        advance = snapshot[3] - previous[3][rows]
        return rows if np.all((advance >= 0) & (advance <= snapshot[5])) else None

    def _steady_pages_moved(self, snapshot, rows) -> np.ndarray:
        """Per request, whether a steady step's last position, SWA window start or
        allocated capacity crossed a page boundary since its previous ``rows``, or its
        page-storage version moved, so its page tables may have changed."""
        previous = self._csa2_steady_snapshot
        block = self.kv_cache_manager.tokens_per_block
        window = self.kv_cache_manager.layout.window_size
        starts, allocated, lengths, page_versions = snapshot[3:]
        last = starts + lengths - 1
        prev_last = last - (starts - previous[3][rows])
        return (
            (page_versions != previous[6][rows])
            | (last // block != prev_last // block)
            | ((last - window) // block != (prev_last - window) // block)
            | ((allocated - 1) // block != (previous[4][rows] - 1) // block)
        )

    def _refresh_steady_page_tables(
        self,
        swa_publication,
        request_ids,
        starts,
        lengths,
        floors,
        allocated_until,
        page_count,
        page_versions,
    ) -> None:
        """Re-read the page tables of a steady step that crossed a page.

        The device-converted tables changed while the geometry did not: check
        the pages written by requests with nonzero ``lengths`` like a full
        prepare (the others write inside pages an earlier prepare checked),
        then copy every table into the buffers the consumers already hold.
        ``starts``, ``lengths`` and ``floors`` are int64 arrays; only the
        requests with nonzero ``lengths`` are checked. The copy plan holds for
        every steady step that reads the same converted tables.
        """
        from . import kernel
        from .cache_manager import CSA2CacheRole

        manager = self.kv_cache_manager
        published_swa = manager._get_swa_publication(
            swa_publication, self.kv_cache_block_offsets, request_ids, self.num_contexts, page_count
        )
        if published_swa is None:
            manager.compute_batch_page_tables(
                request_ids, self.num_contexts, page_versions=page_versions.tolist()
            )
        rows = np.flatnonzero(lengths)
        starts, lengths, floors, allocated_until = (
            values[rows] for values in (starts, lengths, floors, allocated_until)
        )
        self._check_swa_write_pages(starts, lengths, floors, allocated_until, page_count, rows)
        self._check_owner_pages(
            manager.layout.kv_source_layer_ids,
            starts,
            starts + lengths,
            allocated_until,
            page_count,
            rows,
        )
        tables = manager._batch_page_tables
        key = (
            tables.data_ptr(),
            tables.shape,
            None
            if published_swa is None
            else (published_swa.data_ptr(), published_swa.shape, published_swa.stride()),
        )
        plan = self._csa2_steady_refresh
        if plan is None or plan[0] != key:
            targets, sources = [], []

            def refresh(target, spec, published=None):
                source = (
                    manager.batch_page_table(spec, page_count) if published is None else published
                )
                if target.data_ptr() != source.data_ptr() or target.stride() != source.stride():
                    targets.append(target)
                    sources.append(source)

            for owner, table in self.csa2_global_page_tables.items():
                refresh(table, (owner, CSA2CacheRole.GLOBAL))
            for owner, batch in self._csa2_compression.items():
                refresh(batch.kv_page_table, (owner, CSA2CacheRole.COMPRESSOR_KV))
                refresh(batch.score_page_table, (owner, CSA2CacheRole.COMPRESSOR_SCORE))
            for layer, table in self._csa2_swa_page_tables.items():
                refresh(
                    table,
                    (layer, CSA2CacheRole.SWA),
                    None if published_swa is None else published_swa[manager.layer_offsets[layer]],
                )
            descriptors = None
            if targets and targets[0].is_cuda and kernel.dsl_available():
                descriptors = [
                    kernel.table_copy_descriptor(*pair) for pair in zip(targets, sources)
                ]
            plan = self._csa2_steady_refresh = (
                key,
                targets,
                sources,
                descriptors,
                max((target.shape[0] for target in targets), default=0),
            )
        _, targets, sources, descriptors, max_rows = plan
        if descriptors is not None:
            kernel.copy_tables_(
                self._device_descriptors("page_table_refresh", descriptors), max_rows
            )
        elif targets:
            torch._foreach_copy_(targets, sources)

    def _finish_prepare_csa2(self, snapshot, *, defer_outputs: bool) -> None:
        manager = self.kv_cache_manager
        self._csa2_prepared_manager = manager
        self._csa2_forward_serial = next(_FORWARD_SERIALS)
        if not hasattr(self, "_csa2_buffers"):
            self._csa2_buffers = {}
        self._csa2_buffers[_BANK_WRITER] = self._csa2_forward_serial
        for layer_idx in getattr(self, "_csa2_priors", {}):
            self._refresh_indexer_prior(layer_idx)
        self._csa2_steady_snapshot = snapshot
        self._csa2_ready_for_kv_update = True
        self._csa2_deferred_decode_outputs = bool(defer_outputs)

    def _cache_dependent_fields(self) -> dict:
        shared = {"_csa2_query_tiles", "_csa2_indexer_workspaces", "_csa2_manager_states"}
        return {
            name: value
            for name, value in vars(self).items()
            if (name.startswith("csa2_") or name.startswith("_csa2_")) and name not in shared
        }

    def _restore_cache_fields(self, fields: dict) -> None:
        for name in self._cache_dependent_fields():
            delattr(self, name)
        for name, value in fields.items():
            setattr(self, name, value)

    def prepare_for_draft_forward(self) -> dict | None:
        """Rebuild draft-owned fields after the existing interface swaps managers.

        Only identical explicit layer layouts are representable here. Virtual
        model-layer mappings and CED draft scheduling belong to model integration.
        This hook runs before capture/replay, never inside a captured forward.
        """
        target = getattr(self, "_csa2_prepared_manager", None)
        draft = self.kv_cache_manager
        if target is None or target is draft:
            return None
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare CSA2 draft cache metadata before graph capture")
        if getattr(draft, "layout", None) != target.layout:
            raise NotImplementedError(
                "CSA2 draft cache requires an identical explicit layer mapping"
            )
        saved = self._cache_dependent_fields()
        if not hasattr(self, "_csa2_manager_states"):
            self._csa2_manager_states = {}
        self._restore_cache_fields(self._csa2_manager_states.get(id(draft), {}))
        try:
            self.prepare_csa2()
            # Even contiguous draft requests may follow target execution in
            # the same TopK module. Manager transitions always invalidate its
            # emission state, independently of the restored request identities.
            callbacks = list(saved.get("_csa2_indexer_resets", {}).values())
            callbacks += list(getattr(self, "_csa2_indexer_resets", {}).values())
            for callback in callbacks:
                callback()
        except (ValueError, TypeError, KeyError, NotImplementedError):
            self._restore_cache_fields(saved)
            raise
        return saved

    def restore_after_draft_forward(self, saved_state: dict | None) -> None:
        if saved_state is None:
            return
        draft = self._csa2_prepared_manager
        self._csa2_manager_states[id(draft)] = self._cache_dependent_fields()
        self._restore_cache_fields(saved_state)
        # Target and draft may use the same module's emission buffers, while
        # their request histories remain independent. Reset before target reuse.
        for callback in getattr(self, "_csa2_indexer_resets", {}).values():
            callback()

    def begin_model_forward(self) -> None:
        """Reset only per-model-forward publications, retaining prepared buffers."""
        self._csa2_model_driven = True
        self.reset_routing()
        self.csa2_precomputed_kv_layers = set()
        self.csa2_replay_query_rows = None
        self.csa2_decoder_capture_lens = None

    def on_update_kv_lens(self) -> None:
        """Refresh query/cache addresses after device-side acceptance correction.

        Host cached lengths reserve verification capacity; live device endpoints
        may be shorter. All address/ratio computations stay on device and retain
        the prepared tensor shapes for captured replay.
        """
        super().on_update_kv_lens()
        if self._csa2_pending_prepare is not None:
            self._csa2_pending_prepare = (self._csa2_pending_prepare[0], True)
        elif getattr(self, "_csa2_ready_for_kv_update", False):
            self._refresh_live_endpoints()

    def _refresh_live_endpoints(self) -> None:
        """Recompute positions and owner slots; SWA slots follow in the next forward."""
        from . import kernel

        self._csa2_swa_resolved = False
        if self._csa2_pending_owner_tables is not None:
            # Deferred owner slots resolve from the live ends too.
            self._csa2_pending_owner_tables = (
                *self._csa2_pending_owner_tables[:-1],
                self.kv_lens_cuda,
            )
        if (
            self.kv_lens_cuda.is_cuda
            and kernel.dsl_available()
            and self._csa2_owner_descriptors is not None
        ):
            kernel.refresh_owner_slots(
                self.kv_lens_cuda,
                self._csa2_query_lengths,
                self._csa2_query_base,
                *self._csa2_owner_descriptors,
                positions=(self.csa2_positions, self.csa2_token_requests, self._csa2_query_offsets),
            )
            return
        _, _, positions = self._refresh_query_positions(
            self.kv_lens_cuda,
            self._csa2_query_lengths,
            self._csa2_query_base,
            self.csa2_token_requests.long(),
            self._csa2_query_offsets,
        )
        self.csa2_positions.copy_(positions)
        self._resolve_owner_slots(self.kv_lens_cuda)

    def _resolve_owner_slots(self, kv_lens: torch.Tensor) -> None:
        """Every KV owner's GLOBAL write slots and compressed groups for the KV ends
        ``kv_lens``: one CuTe launch for all owners, or the PyTorch reference."""
        from . import kernel

        if kv_lens.is_cuda and kernel.dsl_available():
            if self._csa2_owner_descriptors is not None:
                kernel.refresh_owner_slots(
                    kv_lens,
                    self._csa2_query_lengths,
                    self._csa2_query_base,
                    *self._csa2_owner_descriptors,
                )
            return
        _, delta, _ = self._refresh_query_positions(
            kv_lens,
            self._csa2_query_lengths,
            self._csa2_query_base,
            self.csa2_token_requests.long(),
            self._csa2_query_offsets,
        )
        for _, args in self._owner_slot_arguments():
            source_base, source_lengths, pages, slots, compressed, ratio, page_size, *batch = args
            self._refresh_owner_slots(
                source_base,
                source_lengths,
                delta,
                ratio,
                page_size,
                pages,
                slots,
                compressed,
                *batch,
            )

    def _owner_slot_arguments(self):
        """``(owner, (source base, source lengths, pages, slots, compressed, ratio,
        page size, batch starts, batch lengths, batch cu))`` of every KV owner."""
        manager = self.kv_cache_manager
        for owner, (source_base, source_lengths) in self._csa2_source_geometry.items():
            ratio = manager.layout.compress_ratios[owner]
            batch = self._csa2_compression.get(owner) if ratio == 2 else None
            yield (
                owner,
                (
                    source_base,
                    source_lengths,
                    self.csa2_global_page_tables[owner],
                    self.csa2_main_write_slots[owner],
                    self._csa2_compressed_positions[owner],
                    ratio,
                    self.csa2_global_page_sizes[owner],
                    *(
                        (None, None, None)
                        if batch is None
                        else (batch.start_positions, batch.kv_lengths, batch.cu_compressed_lengths)
                    ),
                ),
            )

    def _owner_slot_descriptors(self):
        """Device descriptors and launch sizes of ``_resolve_owner_slots``'s kernel."""
        from . import kernel

        arguments = [args for _, args in self._owner_slot_arguments()]
        if not arguments:
            return None
        rows = [kernel.owner_slot_descriptor(*args) for args in arguments]
        return (
            self._device_descriptors("owner_slot_desc", rows),
            int(arguments[0][0].numel()),
            max(int(args[3].numel()) for args in arguments),
        )

    @staticmethod
    def _refresh_query_positions(kv_lens_cuda, lengths, query_base, requests, query_offsets):
        starts = kv_lens_cuda[: lengths.numel()].int() - lengths
        delta = starts - query_base
        positions = starts[requests] + query_offsets
        return starts, delta, positions

    @staticmethod
    def _refresh_owner_slots(
        source_base,
        source_lengths,
        delta,
        ratio,
        page_size,
        pages,
        slots,
        compressed_positions,
        batch_starts,
        batch_lengths,
        batch_cu_compressed,
    ):
        """Owner write slots and compressed groups for corrected live endpoints.

        One fused graph per owner: the previous per-op version launched about
        fifteen tiny kernels for each KV source layer on every decode step.
        """
        source_starts = source_base + delta
        source_ends = source_starts + source_lengths
        counts = source_ends // ratio - source_starts // ratio
        cumulative = counts.cumsum(0).int()
        offsets = torch.arange(slots.numel(), device=slots.device, dtype=torch.int32)
        output_requests = torch.searchsorted(cumulative, offsets, right=True)
        valid = output_requests < counts.numel()
        output_requests = output_requests.clamp(max=counts.numel() - 1).long()
        preceding = torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]))
        groups = source_starts[output_requests] // ratio + offsets - preceding[output_requests]
        columns = groups.clamp_min(0) // page_size
        physical = pages[output_requests, columns.clamp(max=pages.shape[1] - 1)]
        valid &= (groups >= 0) & (columns < pages.shape[1]) & (physical >= 0)
        slots.copy_(torch.where(valid, physical.long() * page_size + groups % page_size, -1))
        compressed_positions.copy_(torch.where(valid, groups * ratio, 0))
        if batch_starts is not None:
            batch_starts.copy_(source_starts)
            batch_lengths.copy_(source_ends)
            batch_cu_compressed[0].zero_()
            batch_cu_compressed[1:].copy_(cumulative)

    def reset_routing(self) -> None:
        """Begin one packed forward; graph replay recomputes captured producers."""
        self.csa2_indices = {}
        self.csa2_candidates = {}
        self.csa2_candidate_blocks = {}
        self.csa2_candidate_counts = {}
        self._csa2_last_layer = -1
        # Staging groups compact and record main scale bits again in every forward.
        self._csa2_stage_epoch = getattr(self, "_csa2_stage_epoch", 0) + 1
        if getattr(self, "_csa2_swa_resolved", None) is not None:
            self._csa2_swa_resolved = False
        # Standalone component callers may initialize routing before prepare.
        # Shape preparation must retain an existing CED prepass publication.
        if not hasattr(self, "csa2_precomputed_kv_layers"):
            self.csa2_precomputed_kv_layers = set()
        if not hasattr(self, "csa2_replay_query_rows"):
            self.csa2_replay_query_rows = None

    def enter_layer(self, layer: CSA2Layer) -> None:
        if layer.layer_idx < getattr(self, "csa2_query_layer_start", 0):
            raise ValueError("CSA2 encoder layer cannot consume decoder-only query metadata")
        if layer.layer_idx <= self._csa2_last_layer:
            raise ValueError("CSA2 routing cannot be reused across forwards or reordered layers")
        self._csa2_last_layer = layer.layer_idx
        self._ensure_swa_slots()

    def global_slot_tile(
        self,
        layer_idx: int,
        start: int,
        end: int,
        logical: torch.Tensor | None = None,
        *,
        visible_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Resolve one query tile through its owner's current physical pages.

        Page tables stay request-by-page; no persistent token-by-context mapping
        is allocated. Invalid logical entries, request rows and absent pages
        resolve to -1. A SWA-only layer has no global entries. Attention callers
        may apply their per-query visibility in the same mapping operation.
        """
        requests = self.csa2_token_requests[start:end].long()
        owner = self.csa2_kv_sources[layer_idx]
        if owner is None:
            return torch.empty((requests.shape[0], 0), dtype=torch.int64, device=requests.device)
        table = self.csa2_global_page_tables[owner]
        page_size = self.csa2_global_page_sizes[owner]
        max_positions = self.csa2_global_max_positions[owner]
        if logical is None:
            logical = torch.arange(max_positions, device=table.device)
            logical = logical.expand(requests.shape[0], -1)
        return self._map_global_slots(
            logical, requests, table, visible_lengths, page_size, max_positions
        )

    @staticmethod
    def _map_global_slots(logical, requests, table, visible_lengths, page_size, max_positions):
        logical = logical.long()
        if table.shape[0] == 0 or table.shape[1] == 0:
            slots = torch.full_like(logical, -1)
        else:
            pages = logical.clamp_min(0) // page_size
            valid = (logical >= 0) & (logical < max_positions) & (pages < table.shape[1])
            valid &= ((requests >= 0) & (requests < table.shape[0]))[:, None]
            physical = table[
                requests.clamp(0, table.shape[0] - 1)[:, None],
                pages.clamp(max=table.shape[1] - 1),
            ]
            slots = physical.long() * page_size + logical % page_size
            slots = torch.where(valid & (physical >= 0), slots, -1)
        if visible_lengths is not None:
            slots = torch.where(logical < visible_lengths[:, None], slots, -1)
        return slots

    def prepare_indexer(
        self, layer_idx: int, batch_queries: bool = False
    ) -> CSA2TrtllmMetadata | None:
        """Bind native paged-indexer descriptors for the real generation queries.

        The owner's index rows are read in place: a GLOBAL page is a run of
        consecutive native 64-row index pages, so the block table is derived
        from the owner's page table exactly as the DeepSeek-V4 indexer reads
        its INDEXER_COMPRESS cache. Descriptors are refreshed once per owner
        per forward and shared by every layer of that owner, and by every owner
        whose table converts identically. Eager scratch
        covers the currently visible pages; graph scratch reserves each
        admitted request's maximum context so pointers remain stable while
        request lengths and page mappings change between replays.

        With ``batch_queries``, requests that all decode the same number of
        tokens are scheduled as ``csa2_indexer_next_n`` queries per request, so
        the paged kernel reads each request's index pages once.
        """
        from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

        from .cache_manager import CSA2CacheRole
        from .quantization import INDEX_PAGE_ROWS

        manager = self.kv_cache_manager
        if manager is None:
            raise ValueError("CSA2 paged indexer requires a cache manager")
        if torch.cuda.is_current_stream_capturing() and not self.is_cuda_graph:
            raise RuntimeError("Set is_cuda_graph before warming CSA2 paged indexer metadata")
        owner = self.csa2_kv_sources[layer_idx]
        if owner is None:
            raise ValueError("SWA-only layers have no indexer cache")
        self._ensure_swa_slots()  # Visible lengths bound the descriptors.
        context_requests = self.csa2_num_context_requests
        generation_ranges = self.csa2_request_query_ranges[context_requests:]
        if not generation_ranges:
            return None
        if any(start == end for start, end in generation_ranges):
            raise ValueError("CSA2 paged indexer requires nonempty generation requests")
        decode_start = generation_ranges[0][0]
        decode_end = generation_ranges[-1][1]
        count = decode_end - decode_start
        request_count = len(generation_ranges)
        next_n = count // request_count
        if not (
            batch_queries
            and next_n in _BATCHED_DECODE_NEXT_N
            and all(end - start == next_n for start, end in generation_ranges)
        ):
            next_n = 1
        ratio = manager.layout.compress_ratios[owner]
        if self.is_cuda_graph:
            max_positions = max(1, self.csa2_global_max_positions[owner])
        else:
            max_positions = max(
                1,
                max(
                    (start + length) // ratio
                    for start, length in zip(
                        self.csa2_request_start_positions[context_requests:],
                        self.csa2_request_lengths[context_requests:],
                    )
                ),
            )
        # The DSL TopK reads 8-column vectors; an aligned logits width avoids
        # re-padding the whole logits tensor before every selection.
        max_positions = -(-max_positions // 8) * 8
        native_page_size = INDEX_PAGE_ROWS
        if self.is_cuda_graph:
            # Reserve the largest owner geometry during warmup so another
            # serialized owner cannot replace a buffer captured by this graph.
            required_pages = max(
                1,
                -(-max(self.csa2_global_max_positions.values()) // native_page_size),
            )
        else:
            # Grow geometrically, replacing the old eager arena rather than
            # retaining one allocation for every context length seen so far.
            required_pages = 1 << (-(-max_positions // native_page_size) - 1).bit_length()
        pages_per_source_page = manager.index_pages_per_global_page(owner)
        table = self.csa2_global_page_tables[owner][context_requests:]
        device = table.device
        draft_width = 1 + getattr(manager, "max_total_draft_tokens", 0)
        query_capacity = self.max_num_requests * draft_width if self.is_cuda_graph else count
        if self.is_cuda_graph and count > query_capacity:
            raise ValueError("CSA2 graph query count exceeds configured batch and draft width")
        key = (
            (device, manager.layout.index_topk, True)
            if self.is_cuda_graph
            else (device, request_count, count, manager.layout.index_topk, False)
        )
        if not hasattr(self, "_csa2_indexer_workspaces"):
            self._csa2_indexer_workspaces = {}
        if not self.is_cuda_graph:
            # Runtime metadata survives many batch geometries. Retain just
            # the latest eager arena per device; active descriptors still own
            # their tensors, and serialized stream work preserves their use.
            # Graph clones may share this dictionary, so never evict graph keys.
            for previous_key in tuple(self._csa2_indexer_workspaces):
                if previous_key[0] == device and not previous_key[-1] and previous_key != key:
                    del self._csa2_indexer_workspaces[previous_key]
        scratch = self._csa2_indexer_workspaces.get(key)
        if scratch is None or scratch["page_capacity"] < required_pages:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up CSA2 paged indexer metadata before graph capture")
            page_capacity = required_pages
            scratch = {
                "page_capacity": page_capacity,
                "block_table": torch.empty(
                    (query_capacity, page_capacity), dtype=torch.int32, device=device
                ),
                "context_lengths": torch.empty(
                    (query_capacity, 1), dtype=torch.int32, device=device
                ),
                "visible_lengths": torch.empty(query_capacity, dtype=torch.int32, device=device),
                "valid_pages": torch.empty(
                    (query_capacity, page_capacity),
                    dtype=torch.bool,
                    device=device,
                ),
                "radix_indices": torch.empty(
                    (query_capacity, 10, manager.layout.index_topk),
                    dtype=torch.int32,
                    device=device,
                ),
                "radix_logits": torch.empty(
                    (query_capacity, 10, manager.layout.index_topk),
                    dtype=torch.float32,
                    device=device,
                ),
                "schedules": {},
                "schedule_serials": {},
            }
            self._csa2_indexer_workspaces[key] = scratch
        page_capacity = scratch["page_capacity"]
        active_blocks = scratch["block_table"][:count]
        active_context = scratch["context_lengths"][:count]
        active_visible = scratch["visible_lengths"][:count]
        active_valid = scratch["valid_pages"][:count]
        # Owners of one compression ratio see identical decode context lengths,
        # so they share one paged-MQA schedule, refreshed once per forward as
        # the DSA indexer refreshes its own.
        schedule = scratch["schedules"].get(ratio)
        if schedule is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up CSA2 paged indexer metadata before graph capture")
            schedule = torch.empty_like(
                get_paged_mqa_logits_metadata(
                    torch.ones((1, 1), dtype=torch.int32, device=device),
                    native_page_size,
                    torch.cuda.get_device_properties(device).multi_processor_count,
                )
            )
            scratch["schedules"][ratio] = schedule
        serial = (
            id(manager),
            getattr(self, "_csa2_forward_serial", 0),
            count,
            next_n,
            self.is_cuda_graph,
            torch.cuda.is_current_stream_capturing(),
        )
        # Each layer of one owner sees identical requests, pages and visibility within a
        # forward, so the descriptors are derived once per owner and forward. Owners whose
        # GLOBAL page tables convert identically (same pool, scale and offset: the GLOBAL role
        # indexes pages per pool, not per layer) and whose ratios match share one derivation.
        group = owner
        if self.csa2_replay_mode is None and not getattr(self, "csa2_query_layer_start", 0):
            group = (
                ratio,
                pages_per_source_page,
                manager.batch_page_table_key((owner, CSA2CacheRole.GLOBAL)),
            )
        owner_pages = ceil_div(max_positions, native_page_size)
        # The arena holds one group's descriptors at a time: every owner, graph copy and
        # manager (target or draft) shares it. Keep them only if the arena provably holds this
        # group's derivation of this very forward (serials are unique per prepare).
        if scratch.get("holder") != (group, serial):
            # A capture always records its own refresh so replays follow the live page
            # tables and visible lengths. The upstream CuTe dispatcher already fuses these
            # integer outputs and records its launch directly in the decode CUDA graph.
            self._fill_indexer_descriptors(
                table,
                self.csa2_token_requests[decode_start:decode_end],
                self.csa2_visible_lengths[layer_idx][decode_start:decode_end],
                context_requests,
                pages_per_source_page,
                # The arena fits the largest owner; consumers of this owner
                # read at most max_positions columns.
                active_blocks[:, :owner_pages],
                active_context,
                active_visible,
                active_valid[:, :owner_pages],
            )
            if scratch["schedule_serials"].get(ratio) != serial:
                schedule.copy_(
                    get_paged_mqa_logits_metadata(
                        active_context.view(-1, next_n),
                        native_page_size,
                        torch.cuda.get_device_properties(device).multi_processor_count,
                    )
                )
                scratch["schedule_serials"][ratio] = serial
            scratch["holder"] = (group, serial)
        self.csa2_indexer_k_cache = manager.get_index_pages(owner)
        self.csa2_indexer_block_table = active_blocks
        self.csa2_indexer_context_lengths = active_context
        self.csa2_indexer_scheduler_metadata = schedule
        self.csa2_indexer_next_n = next_n
        self.csa2_indexer_max_seq_len = max_positions
        self.csa2_indexer_radix_aux_indices = scratch["radix_indices"][:count]
        self.csa2_indexer_radix_aux_logits = scratch["radix_logits"][:count]
        self.csa2_indexer_visible_lengths = active_visible
        self.csa2_indexer_valid_pages = active_valid
        return self

    def prepare_sparse_indexer(
        self, layer_idx: int, sparse_block: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Descriptors for DeepGEMM's paged *sparse* MQA logits, or None if unavailable.

        Reuses what the preceding ``prepare_indexer`` call bound for the generation
        rows (native 64-row index pages and per-query block table) plus
        the host-prepared generation-row -> request map, and builds the kernel
        schedule once per owner and forward: every candidate consumer of an owner
        scores the same published blocks over the same pages.
        """
        from tensorrt_llm.deep_gemm import get_paged_sparse_mqa_logits_metadata

        owner = self.csa2_kv_sources[layer_idx]
        inputs = self._sparse_schedule_inputs(owner)
        if inputs is None:
            return None
        pages, table, counts, row_requests, blocks, serial = inputs
        schedules = self.__dict__.setdefault("_csa2_sparse_schedules", {})
        cached = schedules.get(owner)
        if cached is None or cached[0] != serial:
            schedule = get_paged_sparse_mqa_logits_metadata(
                counts,
                table,
                row_requests,
                pages.shape[1],
                blocks,
                torch.int8,
                sparse_block,
            )
            schedules[owner] = cached = (serial, schedule, None)
        elif cached[2] is not None:
            # Built on the side stream by ``prebuild_sparse_schedule``.
            torch.cuda.current_stream().wait_event(cached[2])
        return pages, table, counts, row_requests, cached[1]

    def _sparse_schedule_inputs(self, owner: int):
        """``(pages, table, counts, row_requests, blocks, serial)`` of ``owner``'s sparse
        logits schedule for this forward, or None when the paged sparse path is unavailable."""
        row_requests = getattr(self, "csa2_decode_row_requests", None)
        table = getattr(self, "csa2_indexer_block_table", None)
        blocks = self.csa2_candidate_blocks.get(owner)
        counts = self.csa2_candidate_counts.get(owner)
        if table is None:
            return None
        count = table.shape[0]
        if (
            row_requests is None
            or blocks is None
            or counts is None
            or table.shape[1] == 0
            or row_requests.shape[0] != count
        ):
            return None
        pages = self.csa2_indexer_k_cache
        # DeepGEMM derives the active sparse slots from these lengths, not KV visibility.
        counts = counts[counts.shape[0] - count :].reshape(-1)
        blocks = blocks[blocks.shape[0] - count :]
        serial = (
            id(self.kv_cache_manager),
            getattr(self, "_csa2_forward_serial", 0),
            count,
            self.is_cuda_graph,
            torch.cuda.is_current_stream_capturing(),
        )
        return pages, table, counts, row_requests, blocks, serial

    def publish_candidates_event(self, source: int, event: torch.cuda.Event) -> None:
        """``source`` publishes this forward's candidates on a side stream; ``event`` marks
        them complete for its consumers (``wait_candidates``)."""
        events = self.__dict__.setdefault("_csa2_candidate_events", {})
        events[source] = (self._candidate_serial(), event)

    def wait_candidates(self, source: int) -> None:
        """Order the calling stream after ``source``'s side-stream candidate publication (and
        sparse logits schedule) of this forward, if it published there."""
        entry = self.__dict__.get("_csa2_candidate_events", {}).get(source)
        if entry is not None and entry[0] == self._candidate_serial():
            torch.cuda.current_stream().wait_event(entry[1])

    def _candidate_serial(self) -> tuple:
        return (
            id(self.kv_cache_manager),
            getattr(self, "_csa2_forward_serial", 0),
            self.is_cuda_graph,
            torch.cuda.is_current_stream_capturing(),
        )

    def prebuild_sparse_schedule(self, owner: int, sparse_block: int) -> None:
        """Build the sparse logits schedule ``owner``'s candidate consumers share this forward
        on the CSA2 side stream as soon as ``owner`` has published its candidates (the calling
        stream's work so far), so the first consumer layer only waits for it instead of building
        it on the model stream."""
        from tensorrt_llm.deep_gemm import get_paged_sparse_mqa_logits_metadata

        inputs = self._sparse_schedule_inputs(owner)
        if inputs is None:
            return
        pages, table, counts, row_requests, blocks, serial = inputs
        current = torch.cuda.current_stream()
        side = csa2_side_stream(current.device)
        published = torch.cuda.Event()
        published.record(current)
        with torch.cuda.stream(side):
            side.wait_event(published)
            for tensor in (table, counts, row_requests, blocks):
                tensor.record_stream(side)
            schedule = get_paged_sparse_mqa_logits_metadata(
                counts,
                table,
                row_requests,
                pages.shape[1],
                blocks,
                torch.int8,
                sparse_block,
            )
            done = torch.cuda.Event()
            done.record(side)
        schedule.record_stream(current)
        self.__dict__.setdefault("_csa2_sparse_schedules", {})[owner] = (serial, schedule, done)
        # The schedule follows the publication on the side stream: consumers wait for both.
        self.publish_candidates_event(owner, done)

    @staticmethod
    def _fill_indexer_descriptors(
        table,
        token_requests,
        decode_visible,
        context_requests,
        pages_per_source_page,
        active_blocks,
        active_context,
        active_visible,
        active_valid,
    ):
        """Derive native block tables and page validity from the owner's page table.

        Every GLOBAL page is ``pages_per_source_page`` consecutive native index
        pages. The paged kernel scores the whole visible prefix, reading page
        zero for unallocated source pages; ``active_valid`` marks the native
        pages backed by an allocated page, and readers bound it by the visible length.
        """
        request_count, source_pages = table.shape
        page_capacity = active_blocks.shape[1]
        device = table.device
        if table.is_cuda and request_count > 0:
            from . import kernel

            if kernel.dsl_available():
                kernel.fill_indexer_descriptors(
                    table,
                    token_requests,
                    decode_visible,
                    context_requests,
                    pages_per_source_page,
                    active_blocks,
                    active_context,
                    active_visible,
                    active_valid,
                )
                return
        if request_count == 0:
            active_blocks.zero_()
            active_visible.zero_()
            active_valid.fill_(False)
            active_context.fill_(1)
            return
        native = table.long()[:, :, None] * pages_per_source_page + torch.arange(
            pages_per_source_page, device=device
        )
        native = native.reshape(request_count, source_pages * pages_per_source_page)
        valid_pages = (table >= 0)[:, :, None].expand(-1, -1, pages_per_source_page)
        valid_pages = valid_pages.reshape(request_count, -1)
        if native.shape[1] < page_capacity:
            pad = page_capacity - native.shape[1]
            native = torch.nn.functional.pad(native, (0, pad))
            valid_pages = torch.nn.functional.pad(valid_pages, (0, pad), value=False)
        native = native[:, :page_capacity]
        valid_pages = valid_pages[:, :page_capacity]
        native = torch.where(valid_pages, native, 0)
        requests = token_requests - context_requests
        valid_requests = (requests >= 0) & (requests < request_count)
        requests = requests.clamp(0, request_count - 1).long()
        active_blocks.copy_(torch.where(valid_requests[:, None], native[requests].int(), 0))
        visible = torch.where(valid_requests, decode_visible.int(), 0)
        active_visible.copy_(visible)
        active_valid.copy_(valid_pages[requests] & valid_requests[:, None])
        # A zero-length row keeps one defined column for the paged kernel.
        active_context.copy_(visible.clamp_min(1)[:, None])

    @staticmethod
    def cache_gather_bytes_per_token(model_config) -> int:
        """Transient index gather of eager prefill, per logical raw source token.

        Decode reads index pages in place. Prefill gathers the compressed prefix
        of every query piece that ``_prepare_indexer_plans`` splits a request
        into: on the native path each key's E2M1 data, scale word and validity
        byte. The decoded fallback (index_head_dim != 128 or pre-SM100) and an
        unknown max_seq_len use 512 bytes per key.
        """
        from tensorrt_llm._utils import get_sm_version

        from ..dsa.indexer import _INDEXER_MQA_LOGITS_ELEM_BUDGET
        from .params import CSA2Layout
        from .quantization import INDEX_DATA_BYTES, INDEX_SCALE_BYTES

        layout = CSA2Layout.from_hf_config(model_config.pretrained_config)
        ratios = {layout.compress_ratios[owner] for owner in layout.kv_source_layer_ids}
        if not ratios:
            return 0
        max_seq_len = model_config.max_seq_len
        if layout.index_head_dim != 128 or get_sm_version() < 100 or max_seq_len is None:
            return (512 + min(ratios) - 1) // min(ratios)
        chunk_size = model_config.sparse_attention_config.indexer_max_chunk_size or 32768
        rate = 0
        for ratio in ratios:
            # Pieces are shortest at the widest prefix; logits rows pad to 256 columns.
            columns = -(-(max_seq_len // ratio) // 256) * 256
            step = max(1, min(chunk_size, _INDEXER_MQA_LOGITS_ELEM_BUDGET // columns))
            pieces = -(-model_config.max_num_tokens // step)
            key_bytes = (INDEX_DATA_BYTES + INDEX_SCALE_BYTES + 1) * pieces
            rate = max(rate, -(-key_bytes // ratio))
        return rate

    @staticmethod
    def workspace_reservation_bytes(
        request_capacity: int,
        query_capacity: int,
        max_positions: int,
        topk: int,
        num_sms: int,
        *,
        num_owners: int = 1,
    ) -> int:
        """Exact retained bytes for one native index descriptor arena, including schedule.

        Call with admitted generation capacities, not current token lengths.
        Index rows are read from the owner's cache pages in place, so no page
        copies are retained. This intentionally excludes model projections,
        temporal priors, generic metadata, FMHA native/provider workspaces and
        transient logits; report those separately with get_workspace_bytes.
        """
        if min(request_capacity, query_capacity, max_positions, topk, num_sms, num_owners) < 0:
            raise ValueError("CSA2 workspace capacities must be nonnegative")
        pages_per_request = max(1, (max_positions + 63) // 64)
        # Per query: int32 block table and one validity byte per page, context
        # and visible lengths, and the exact radix TopK scratch.
        per_query = pages_per_request * (4 + 1) + 8 + 80 * topk
        return query_capacity * per_query + num_owners * (num_sms + 1) * 2 * 4

    @staticmethod
    def _workspace_storage_bytes(values) -> int:
        """Count GPU storages once, including aliased descriptor views."""
        seen = set()
        total = 0

        def visit(value):
            nonlocal total
            if isinstance(value, torch.Tensor):
                if value.device.type != "cuda":
                    return
                storage = value.untyped_storage()
                key = (value.device, storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            elif isinstance(value, (CSA2SharedKVPlan, CSA2IndexerGroup)):
                for item in vars(value).values():
                    visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        for value in values:
            visit(value)
        return total

    def get_workspace_bytes(self) -> int:
        """Report retained GPU workspace once per storage, excluding KV pools."""
        values = [
            value
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor) and name != "position_ids"
        ]
        for name in (
            "workspace",
            "cuda_graph_workspace",
            "_csa2_buffers",
            "_csa2_indexer_workspaces",
            "_csa2_priors",
            "_csa2_manager_states",
            "_csa2_shared_tables",
            "csa2_indexer_groups",
            "csa2_indexer_prefixes",
        ):
            values.append(getattr(self, name, None))
        for metadata in getattr(self, "_csa2_query_tiles", {}).values():
            for name, value in vars(metadata).items():
                if name != "kv_cache_manager":
                    values.append(value)
        return self._workspace_storage_bytes(values)

    def register_indexer_reset(self, layer_idx: int, callback) -> None:
        """Register a host-prepare emission reset, before a captured replay."""
        if not hasattr(self, "_csa2_indexer_resets"):
            self._csa2_indexer_resets = {}
        self._csa2_indexer_resets[layer_idx] = callback

    def prepare_indexer_prior(self, layer_idx: int, topk: int) -> torch.Tensor:
        if not hasattr(self, "_csa2_priors"):
            self._csa2_priors = {}
            self.csa2_indexer_prior_capacity = {}
        record = self._csa2_priors.get(layer_idx)
        if record is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up CSA2 temporal priors before graph capture")
            capacity = self.max_num_tokens
            device = self.csa2_token_requests.device
            record = {
                "prior": torch.full((capacity, topk), -1, dtype=torch.int32, device=device),
                "published": torch.full((capacity, topk), -1, dtype=torch.int32, device=device),
                "published_valid": torch.zeros(capacity, dtype=torch.bool, device=device),
                "keys": (),
                "serial": -1,
            }
            self._csa2_priors[layer_idx] = record
            self.csa2_indexer_prior_capacity[layer_idx] = record["prior"]
        if record["prior"].shape[1] != topk:
            raise ValueError("CSA2 temporal prior width must remain fixed for a layer")
        if record["serial"] != getattr(self, "_csa2_forward_serial", 0):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Refresh CSA2 request prior identity before graph replay")
            self._refresh_indexer_prior(layer_idx)
        return record["prior"][: record["count"]]

    def _refresh_indexer_prior(self, layer_idx: int) -> None:
        record = self._csa2_priors[layer_idx]
        previous = {key: row for row, key in enumerate(record["keys"]) if key is not None}
        rows, identities = [], []
        publish_keys = [None] * sum(self.csa2_request_lengths)
        reset = self.csa2_num_context_requests > 0
        for request_row, length in enumerate(self.csa2_request_lengths):
            request_id = self.request_ids[request_row]
            start = self.csa2_request_start_positions[request_row]
            identity = (
                id(self.kv_cache_manager),
                self.kv_cache_manager.request_epoch(request_id),
                request_id,
            )
            begin, end = self.csa2_request_query_ranges[request_row]
            if request_row < self.csa2_num_context_requests:
                # Context source tokens are accepted; its last query safely
                # seeds the first decode after this prompt/chunk.
                if length:
                    publish_keys[end - 1] = (*identity, start + length)
                continue
            source = previous.get((*identity, start), -1) if length == 1 else -1
            reset |= source < 0
            rows.extend([source] if length == 1 else [-1] * length)
            identities.extend([identity] * length)
            if length == 1:
                publish_keys[begin] = (*identity, start + 1)
        reset |= tuple(identities) != record.get("decode_identities", ())
        record["decode_identities"] = tuple(identities)
        count = len(rows)
        if max(count, len(publish_keys)) > record["prior"].shape[0]:
            raise ValueError("CSA2 temporal prior query capacity exceeded")
        device = record["prior"].device
        record["prior"].fill_(-1)
        if count:
            source_rows = torch.tensor(rows, dtype=torch.int64, device=device)
            valid = (source_rows >= 0) & record["published_valid"][source_rows.clamp_min(0)]
            record["prior"][:count].copy_(
                torch.where(valid[:, None], record["published"][source_rows.clamp_min(0)], -1)
            )
        record["published_valid"].zero_()
        record["keys"] = tuple(publish_keys)
        record["count"] = count
        record["serial"] = getattr(self, "_csa2_forward_serial", 0)
        callbacks = getattr(self, "_csa2_indexer_resets", {})
        if reset and layer_idx in callbacks:
            callbacks[layer_idx]()

    def publish_indexer_prior(self, layer_idx: int, logical_fullbatch: torch.Tensor) -> None:
        record = self._csa2_priors[layer_idx]
        count = len(record["keys"])
        if logical_fullbatch.shape[0] != count:
            raise ValueError("CSA2 prior publication must contain the full packed batch")
        record["published"][:count].copy_(logical_fullbatch)
        record["published_valid"][:count].fill_(True)

    def _prepare_deferred(self) -> None:
        """Run the context prepare that ``prepare`` left to the first CSA2 layer."""
        block_offsets, kv_lens_updated = self._csa2_pending_prepare
        self._csa2_pending_prepare = None
        manager = self.kv_cache_manager
        with manager._swa_publication_scope() as publication:
            for args in block_offsets:
                manager.copy_batch_block_offsets(*args)
            self.prepare_csa2(swa_publication=publication)
        if kv_lens_updated:
            self._refresh_live_endpoints()

    def get_compression_batch(self, owner: int):
        if self._csa2_pending_prepare is not None:
            self._prepare_deferred()
        if (
            self._csa2_pending_owner_tables is not None
            and self.csa2_kv_sources.get(owner) is not None
        ):
            self._prepare_pending_owner_tables()
        return self._csa2_compression.get(owner)

    def get_compressed_positions(self, owner: int) -> torch.Tensor:
        return self._csa2_compressed_positions[owner]
