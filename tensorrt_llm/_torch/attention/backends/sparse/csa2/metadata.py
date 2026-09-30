# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 request metadata and bounded staging for TRTLLM sparse MLA."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams

from .params import CSA2BackendForwardArgs, CSA2Layer

if TYPE_CHECKING:
    from .....pyexecutor.llm_request import LlmRequest
    from .decoder_replay import DecoderReplayPlan

_SWA_TILE = 128
_HEAD_DIM = 512


@dataclass
class CSA2SharedKVPlan:
    query_requests: torch.Tensor
    query_positions: torch.Tensor
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


class _CoalescedUploads:
    """Batches the metadata H2D copies of one prepare into a single upload.

    ``copy`` returns the usual graph-stable device buffer and defers its fill;
    ``flush`` uploads all recorded host values at once and scatters them. The
    buffers must not be read before ``flush``.
    """

    _ALIGN = 16

    def __init__(self, metadata: CSA2TrtllmMetadata):
        self._metadata = metadata
        self._pending = []
        self._bytes = 0

    def copy(self, key: str, value) -> torch.Tensor:
        metadata = self._metadata
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                return metadata._copy_csa2_tensor(key, value)
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
        if not self._pending:
            return
        metadata = self._metadata
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

    indexer_max_chunk_size: int = 8192
    indexer_q_split_threshold: int = 8192
    # Compute-tile metadata only: the shared converted-KV plan of an eager
    # context phase, or None for independent-query staging.
    shared_plan: CSA2SharedKVPlan | None = None
    _csa2_defer_decode_outputs: bool = False
    _csa2_deferred_decode_outputs: bool = False

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

    def get_adp_token_counts(self) -> list[int]:
        plan = self.decoder_replay_plan
        input_tokens = self.padded_num_tokens or self.num_tokens
        decoder_tokens = (
            plan.num_replay_tokens if plan and plan.replays_local_tokens else input_tokens
        )
        return [input_tokens, decoder_tokens]

    def set_adp_token_counts(self, counts: list[list[int]]) -> list[int]:
        plan = self.decoder_replay_plan
        if plan is not None:
            plan.saved_all_rank_num_tokens, plan.replay_all_rank_num_tokens = counts
            if not plan.updates_local_metadata and counts[0] == counts[1]:
                self.decoder_replay_plan = None
        return counts[0]

    def __post_init__(self) -> None:
        # Graph clones rerun post-init: eligibility belongs to their own next
        # successful preparation, not the metadata object they were copied from.
        self._csa2_deferred_decode_outputs = False
        self.decoder_replay_plan = None
        self.decoder_context_ends = None
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
            # Presence bitmap of the group-scale bytes each format's selected rows
            # carry, one 256-bit word group per format.
            self._csa2_stage_scale_bitmap = torch.zeros(16, dtype=torch.int32, device=device)
            self.latent_placeholder = torch.empty(
                capacity, _HEAD_DIM, dtype=torch.bfloat16, device=device
            )
        self._csa2_context_geometry = None
        if shared_rows is None:
            self.swa_pool = torch.empty(
                capacity, _SWA_TILE, _HEAD_DIM, dtype=staging_dtype, device=device
            )
            self.extra_pool = torch.empty(
                capacity, max(extra_capacity, 1), _HEAD_DIM, dtype=staging_dtype, device=device
            )
        else:
            self.shared_pool = torch.empty(
                shared_rows, _HEAD_DIM, dtype=staging_dtype, device=device
            )
            self._csa2_shared_selected = torch.empty(shared_rows, dtype=torch.int32, device=device)
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
        swa_counts = [length + window - 1 if length else 0 for length in lengths]
        ratio = 1 if owner is None else manager.layout.compress_ratios[owner]
        main_counts = (
            [0] * requests
            if owner is None
            else [
                end if length else 0
                for end, length in zip(self._csa2_main_domain_counts[owner][:requests], lengths)
            ]
        )
        swa_rows, main_rows = sum(swa_counts), sum(main_counts)
        if swa_rows + main_rows > q.shape[0] * (window + extra_width):
            return None
        if 1 + swa_rows + main_rows >= 1 << 31:
            raise ValueError("CSA2 shared KV row indices exceed int32 capacity")
        cache = self._csa2_shared_domains
        geometry = (owner, requests, tuple(lengths), tuple(main_counts))
        if geometry not in cache:
            first, running = [], 0
            for length in lengths:
                first.append(running if length else 0)
                running += length
            first_device = self._copy_csa2_tensor(
                f"shared_first:{owner}", torch.tensor(first, dtype=torch.int64, device="cpu")
            )
            starts = self.csa2_positions[first_device].long() - window + 1
            starts = torch.maximum(starts.clamp_min(0), self.csa2_replay_start_positions[:requests])
            offsets, total = [1], 1
            for count in swa_counts:
                total += count
                offsets.append(total)
            main_offsets = [total]
            for count in main_counts:
                total += count
                main_offsets.append(total)
            cache[geometry] = (
                starts,
                self._copy_csa2_tensor(
                    f"shared_swa_offsets:{owner}",
                    torch.tensor(offsets, dtype=torch.int64, device="cpu"),
                ),
                self._copy_csa2_tensor(
                    f"shared_main_offsets:{owner}",
                    torch.tensor(main_offsets, dtype=torch.int64, device="cpu"),
                ),
            )
        starts, swa_offsets, main_offsets = cache[geometry]
        return CSA2SharedKVPlan(
            self.csa2_token_requests[: q.shape[0]],
            self.csa2_positions[: q.shape[0]],
            starts,
            swa_offsets,
            main_offsets,
            self._csa2_swa_page_tables[layer_idx][:requests],
            None if owner is None else self.csa2_global_page_tables[owner][:requests],
            manager.tokens_per_block,
            manager.tokens_per_block // ratio,
            requests,
            swa_rows,
            main_rows,
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
        if context_lengths:
            # Eager whole-phase buffers retain only the latest shape per family
            # and owner: owners with different shared-bank sizes alternate
            # within one forward and must not evict each other. GEN and
            # captured-context storage is never evicted or resized here.
            for previous in tuple(self._csa2_query_tiles):
                if (
                    previous != key
                    and previous[4]
                    and previous[:2] == key[:2]
                    and previous[3] == key[3]
                    and previous[5] == key[5]
                    and previous[8] == key[8]
                ):
                    # These eager tiles run sequentially on the model stream.
                    # Keep the native scratch high-water mark when replacing
                    # geometry, without retaining the old staging pools.
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

    def _stage_scratch(self, source, group, count: int):
        """Compaction scratch for a staging group and whether this forward must refill it.

        Layers sharing a KV owner and an index source select the same rows,
        so the first layer of the group in a forward compacts and the others
        only decode. The scratch and the epoch live on the ``source`` request
        metadata (query tiles are separate objects); ``None`` groups always
        compact. A capture records its own compaction so replays follow the
        live selections.
        """
        groups = source.__dict__.setdefault("_csa2_stage_groups", {})
        key = (group, self.prepared_lens.shape[0], self.num_sparse_topk)
        entry = groups.get(key)
        if entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up CSA2 staging groups before graph capture")
            capacity, device = self.prepared_lens.shape[0], self.prepared_lens.device
            entry = groups[key] = {
                "tags": torch.empty(
                    (capacity, self.num_sparse_topk), dtype=torch.int32, device=device
                ),
                "counts": torch.empty(capacity, dtype=torch.int32, device=device),
                "main_slots": torch.empty(
                    (capacity, max(self.num_sparse_topk - _SWA_TILE, 1)),
                    dtype=torch.int64,
                    device=device,
                ),
                "serial": None,
            }
        serial = (
            id(source.kv_cache_manager),
            getattr(source, "_csa2_stage_epoch", 0),
            count,
            torch.cuda.is_current_stream_capturing(),
        )
        compact = group is None or entry["serial"] != serial
        entry["serial"] = serial
        return entry, compact

    def derive_stage_kv_scales(
        self,
        inputs: CSA2BackendForwardArgs,
        *,
        q: torch.Tensor | None = None,
        q_shared: bool = False,
        minimum: int = 0,
        maximum: int = 126,
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
        for why the two phases can afford different exponents.
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
        # An infinity norm is amax(|Q|) in a single pass, with no temporary the size
        # of Q, and it stays inside the capture while a CUDA Graph is recording.
        q_amax = (
            None
            if q is None or not q.numel()
            else torch.linalg.vector_norm(q, float("inf"), dtype=torch.float32)
        )
        _derive_stage_kv_scales(
            sources,
            self._csa2_stage_scale_bitmap,
            self._csa2_stage_kv_dequant,
            self._csa2_stage_kv_quant,
            main_mapping=inputs.main_mapping,
            q_amax=q_amax,
            q_shared=q_shared,
            minimum=minimum,
            maximum=maximum,
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
    ) -> None:
        """Refresh bounded native staging pools and indices for one selected query tile.

        This prepares compute buffers only; the backend publishes native ABI
        arguments and performs attention after staging completes.
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
        mode = getattr(self, "csa2_replay_mode", None)
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
        """Engine-only scope for a selected ordinary decode graph's preparation.

        The engine must run ``on_update_kv_lens`` before any model consumer,
        including warmup and capture. Direct ``prepare`` callers retain fully
        initialized outputs; graph ownership alone is not sufficient proof.
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

    def _check_swa_write_pages(self, starts, lengths, floors, allocated_until, page_count):
        """Raise unless every SWA page the query intervals write is mapped."""
        manager = self.kv_cache_manager
        write_rows, write_columns = self._swa_write_pages(
            starts,
            lengths,
            floors,
            allocated_until,
            manager.tokens_per_block,
            (len(lengths), page_count),
        )
        self._check_written_pages(
            manager.batch_pages_allocated(
                self._csa2_swa_specs, write_rows, write_columns, page_count
            )
        )

    def _check_owner_pages(self, owner, owner_starts, owner_ends, allocated_until, page_count):
        """Raise unless the owner's GLOBAL output groups and compressor state rows
        inside the allocated capacity are mapped."""
        from .cache_manager import CSA2CacheRole

        manager = self.kv_cache_manager
        block = manager.tokens_per_block
        ratio = manager.layout.compress_ratios[owner]
        page_size = block // ratio
        group_starts = np.asarray(owner_starts, dtype=np.int64) // ratio
        group_ends = np.asarray(owner_ends, dtype=np.int64) // ratio
        counts = (group_ends - group_starts).tolist()
        group_requests = np.repeat(np.arange(len(counts)), counts)
        groups = np.arange(sum(counts), dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts - group_starts, counts
        )
        # Only positions inside the allocated capacity must be mapped.
        group_mapped = manager.batch_pages_allocated(
            [(owner, CSA2CacheRole.GLOBAL)], group_requests, groups // page_size, page_count
        )
        allocated_np = np.asarray(allocated_until, dtype=np.int64)[group_requests]
        if np.any(~group_mapped[0] & (groups * ratio < allocated_np)):
            raise ValueError("CSA2 source output has no allocated GLOBAL page")
        if ratio != 2:
            return
        state_starts = np.asarray(owner_starts, dtype=np.int64)
        state_ends = np.asarray(owner_ends, dtype=np.int64)
        state_first = (state_starts - state_starts % ratio) // block
        state_last = np.minimum(
            (np.minimum(state_ends, allocated_until) - 1) // block + 1, page_count
        )
        state_pages = self._flat_page_ranges(
            state_first,
            np.where(state_ends > state_starts, np.maximum(state_last - state_first, 0), 0),
        )
        specs = [(owner, CSA2CacheRole.COMPRESSOR_KV), (owner, CSA2CacheRole.COMPRESSOR_SCORE)]
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
        from .compressor import CSA2CompressionBatch

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
        # Only the engine's selected ordinary decode graph may leave derived
        # outputs for its captured preprocessing. Source overrides, replay and
        # speculative/draft metadata keep the complete preparation contract.
        defer_outputs = (
            getattr(self, "_csa2_defer_decode_outputs", False)
            and self.is_cuda_graph
            and self.num_contexts == 0
            and bool(lengths)
            and all(length == 1 for length in lengths)
            and not self.is_spec_decoding_enabled
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
        snapshot = (tuple(request_ids), tuple(lengths), starts_np, allocated_until)
        block = manager.tokens_per_block
        page_count = (manager.max_seq_len + block - 1) // block
        if replay is None and not external_source:
            if self._steady_generation_step(snapshot):
                # Same generation requests, each advanced by one committed
                # step: every geometry tensor of the previous prepare is still
                # valid, so only the host positions, the routing state and the
                # live device endpoints are refreshed, plus the page tables
                # when a request crossed a page.
                if self._steady_pages_moved(snapshot):
                    self._refresh_steady_page_tables(
                        swa_publication,
                        request_ids,
                        starts,
                        lengths,
                        replay_starts,
                        allocated_until,
                        page_count,
                    )
                self.csa2_request_start_positions = tuple(starts)
                self.reset_routing()
                self._finish_prepare_csa2(snapshot, defer_outputs=defer_outputs)
                if not defer_outputs:
                    self._refresh_live_endpoints()
                return
        self._csa2_steady_snapshot = None
        ends = np.asarray(source_starts, dtype=np.int64) + np.asarray(
            source_lengths, dtype=np.int64
        )
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
        self.csa2_replay_start_positions = copy(
            "replay_starts", np.asarray(replay_starts, dtype=np.int64)
        )
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
        self._csa2_shared_domains = {}
        self._csa2_swa_page_tables = {}
        self._csa2_compression = {}
        self._csa2_compressed_positions = {}
        published_swa = manager._get_swa_publication(
            swa_publication, self.kv_cache_block_offsets, request_ids, self.num_contexts, page_count
        )
        if published_swa is None:
            manager.compute_batch_page_tables(request_ids, self.num_contexts)
        # Owner slots are resolved from these host KV ends: a draft prepare
        # runs without the generic prepare, so kv_lens_cuda may hold live
        # lengths that differ from the host positions.
        prepared_ends = copy("prepared_kv_ends", (starts_np + lengths_np).astype(np.int32))
        if self.csa2_replay_mode is None:
            # Every owner reads the same source rows.
            shared_geometry = (
                copy("source_base", np.asarray(source_starts, dtype=np.int32)),
                copy("source_lengths", np.asarray(source_lengths, dtype=np.int32)),
            )
            shared_cu = copy("source_cu", np.cumsum([0, *source_lengths], dtype=np.int32))
        owners = manager.layout.kv_source_layer_ids
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
            capacity = sum(owner_lengths)
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
            # Compressed output groups; their write slots and positions are
            # resolved on the device after the uploads (_resolve_owner_slots).
            self._check_owner_pages(owner, owner_starts, owner_ends, allocated_until, page_count)
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
                    max(1, max(owner_lengths, default=0)),
                )
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
        layers = [manager.layout.layer(i) for i in range(num_layers)]
        swa_layers = [i for i in range(num_layers) if manager.has_swa_cache(i)]
        # A decoder-only query pass binds no encoder pages: they may be reclaimed.
        first_layer = self.csa2_query_layer_start
        live_swa_layers = [i for i in swa_layers if i >= first_layer]
        swa_specs = self._csa2_swa_specs = [
            (layer_idx, CSA2CacheRole.SWA) for layer_idx in live_swa_layers
        ]
        self._check_swa_write_pages(starts, lengths, replay_starts, allocated_until, page_count)
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
        uploads.flush()
        self._csa2_owner_descriptors = (
            self._owner_slot_descriptors()
            if self.csa2_positions.is_cuda and kernel.dsl_available()
            else None
        )
        if not defer_outputs:
            # A deferred decode graph resolves them in on_update_kv_lens.
            self._resolve_owner_slots(prepared_ends)
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

    def _steady_generation_step(self, snapshot) -> bool:
        """Whether ``snapshot`` continues the previous prepare by one committed step.

        The same generation requests, in the same order and with the same
        query lengths, each start where the previous query ended.
        """
        previous = getattr(self, "_csa2_steady_snapshot", None)
        if previous is None or self.num_contexts:
            return False
        ids, lengths, starts, _ = snapshot
        if (ids, lengths) != (previous[0], previous[1]):
            return False
        return np.array_equal(starts, previous[2] + np.asarray(lengths, dtype=np.int64))

    def _steady_pages_moved(self, snapshot) -> bool:
        """Whether a steady step's last position, SWA window start or allocated
        capacity crossed a page boundary, so its page tables may have changed."""
        previous = self._csa2_steady_snapshot
        block = self.kv_cache_manager.tokens_per_block
        window = self.kv_cache_manager.layout.window_size
        _, lengths, starts, allocated = snapshot
        last = starts + np.asarray(lengths, dtype=np.int64) - 1
        prev_last = last - (starts - previous[2])
        return bool(
            np.any(
                (last // block != prev_last // block)
                | ((last - window) // block != (prev_last - window) // block)
                | ((allocated - 1) // block != (previous[3] - 1) // block)
            )
        )

    def _refresh_steady_page_tables(
        self, swa_publication, request_ids, starts, lengths, floors, allocated_until, page_count
    ) -> None:
        """Re-read the page tables of a steady step that crossed a page.

        The device-converted tables changed while the geometry did not: check
        the pages the step writes like a full prepare, then copy every table
        into the buffers the consumers already hold.
        """
        from .cache_manager import CSA2CacheRole

        manager = self.kv_cache_manager
        published_swa = manager._get_swa_publication(
            swa_publication, self.kv_cache_block_offsets, request_ids, self.num_contexts, page_count
        )
        if published_swa is None:
            manager.compute_batch_page_tables(request_ids, self.num_contexts)
        self._check_swa_write_pages(starts, lengths, floors, allocated_until, page_count)
        ends = [start + length for start, length in zip(starts, lengths)]
        for owner in self.csa2_global_page_tables:
            self._check_owner_pages(owner, starts, ends, allocated_until, page_count)
        targets, sources = [], []

        def refresh(target, spec, published=None):
            source = manager.batch_page_table(spec, page_count) if published is None else published
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
        if targets:
            torch._foreach_copy_(targets, sources)

    def _finish_prepare_csa2(self, snapshot, *, defer_outputs: bool) -> None:
        manager = self.kv_cache_manager
        self._csa2_prepared_manager = manager
        self._csa2_forward_serial = getattr(self, "_csa2_forward_serial", 0) + 1
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
        if getattr(self, "_csa2_ready_for_kv_update", False):
            self._refresh_live_endpoints()

    def _refresh_live_endpoints(self) -> None:
        """Recompute positions and owner slots; SWA slots follow in the next forward."""
        self._csa2_shared_domains = {}
        self._csa2_swa_resolved = False
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

    def prepare_indexer(self, layer_idx: int) -> CSA2TrtllmMetadata | None:
        """Bind native paged-indexer descriptors for the real generation queries.

        The owner's index rows are read in place: a GLOBAL page is a run of
        consecutive native 64-row index pages, so the block table is derived
        from the owner's page table exactly as the DeepSeek-V4 indexer reads
        its INDEXER_COMPRESS cache. Descriptors are refreshed once per owner
        per forward and shared by every layer of that owner. Eager scratch
        covers the currently visible pages; graph scratch reserves each
        admitted request's maximum context so pointers remain stable while
        request lengths and page mappings change between replays.
        """
        from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

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
        if self.is_cuda_graph:
            resolved_cap = getattr(manager, "fp8_ctx_mla_kv_len_cap", None)
            if (
                resolved_cap is not None
                and resolved_cap < self.max_num_requests * manager.max_seq_len
            ):
                raise ValueError("CSA2 graph workspace requires the full admitted request KV bound")
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
                "valid_positions": torch.empty(
                    (query_capacity, page_capacity * native_page_size),
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
                "owners": {},
            }
            self._csa2_indexer_workspaces[key] = scratch
        page_capacity = scratch["page_capacity"]
        active_blocks = scratch["block_table"][:count]
        active_context = scratch["context_lengths"][:count]
        active_visible = scratch["visible_lengths"][:count]
        active_valid = scratch["valid_positions"][:count]
        schedule = scratch["schedules"].get(owner)
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
            scratch["schedules"][owner] = schedule
        serial = (
            id(manager),
            getattr(self, "_csa2_forward_serial", 0),
            count,
            self.is_cuda_graph,
            torch.cuda.is_current_stream_capturing(),
        )
        if scratch["owners"].get(owner) != serial:
            # Each layer of one owner sees identical requests, pages and
            # visibility within a forward, so the descriptors are derived once
            # per owner and forward. The arena is shared across target and
            # draft managers whose serials advance in lockstep, so the manager
            # identity is part of the key. A capture always records its own
            # refresh so replays follow the live page tables and visible lengths.
            # The upstream CuTe dispatcher already fuses these integer outputs
            # and records its launch directly in the decode CUDA graph.
            self._fill_indexer_descriptors(
                table,
                self.csa2_token_requests[decode_start:decode_end],
                self.csa2_visible_lengths[layer_idx][decode_start:decode_end],
                context_requests,
                pages_per_source_page,
                native_page_size,
                active_blocks,
                active_context,
                active_visible,
                active_valid,
            )
            schedule.copy_(
                get_paged_mqa_logits_metadata(
                    active_context,
                    native_page_size,
                    torch.cuda.get_device_properties(device).multi_processor_count,
                )
            )
            scratch["owners"][owner] = serial
        self.csa2_indexer_k_cache = manager.get_index_pages(owner)
        self.csa2_indexer_block_table = active_blocks
        self.csa2_indexer_context_lengths = active_context
        self.csa2_indexer_scheduler_metadata = schedule
        self.csa2_indexer_max_seq_len = max_positions
        self.csa2_indexer_radix_aux_indices = scratch["radix_indices"][:count]
        self.csa2_indexer_radix_aux_logits = scratch["radix_logits"][:count]
        self.csa2_indexer_visible_lengths = active_visible
        self.csa2_indexer_valid_positions = active_valid
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

        row_requests = getattr(self, "csa2_decode_row_requests", None)
        table = self.csa2_indexer_block_table
        owner = self.csa2_kv_sources[layer_idx]
        blocks = self.csa2_candidate_blocks.get(owner)
        counts = self.csa2_candidate_counts.get(owner)
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
            schedules[owner] = cached = (serial, schedule)
        return pages, table, counts, row_requests, cached[1]

    @staticmethod
    def _fill_indexer_descriptors(
        table,
        token_requests,
        decode_visible,
        context_requests,
        pages_per_source_page,
        native_page_size,
        active_blocks,
        active_context,
        active_visible,
        active_valid,
    ):
        """Derive native block tables and position validity from the owner's page table.

        Every GLOBAL page is ``pages_per_source_page`` consecutive native index
        pages. The paged kernel scores the whole visible prefix, reading page
        zero for unallocated source pages; ``active_valid`` marks exactly the
        positions that are both visible and backed by an allocated page, so
        holes are masked before selection as the logical-position path did.
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
                    native_page_size,
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
        offsets = torch.arange(active_valid.shape[1], device=device)
        valid = valid_pages[requests].repeat_interleave(native_page_size, dim=1)
        active_valid.copy_(valid & (offsets[None, :] < visible[:, None]))
        # A zero-length row keeps one defined column for the paged kernel.
        active_context.copy_(visible.clamp_min(1)[:, None])

    @staticmethod
    def cache_gather_bytes_per_token(model_config) -> int:
        """Transient index gather component for eager prefill only.

        Decode reads index pages in place. Prefill still gathers dequantized
        index rows and row-index intermediates per logical raw source token.
        This component rate must not be used as the generic backend workspace
        declaration or substituted for the fixed arena reservation below.
        """
        from .params import CSA2Layout

        layout = CSA2Layout.from_hf_config(model_config.pretrained_config)
        if not layout.kv_source_layer_ids:
            return 0
        ratio = min(layout.compress_ratios[owner] for owner in layout.kv_source_layer_ids)
        return (512 + ratio - 1) // ratio

    @staticmethod
    def workspace_reservation_bytes(
        request_capacity: int, query_capacity: int, max_positions: int, topk: int, num_sms: int
    ) -> int:
        """Exact retained bytes for one native index descriptor arena, including schedule.

        Call with admitted generation capacities, not current token lengths.
        Index rows are read from the owner's cache pages in place, so no page
        copies are retained. This intentionally excludes model projections,
        temporal priors, generic metadata, FMHA native/provider workspaces and
        transient logits; report those separately with get_workspace_bytes.
        """
        if min(request_capacity, query_capacity, max_positions, topk, num_sms) < 0:
            raise ValueError("CSA2 workspace capacities must be nonnegative")
        pages_per_request = max(1, (max_positions + 63) // 64)
        # Per query: int32 block table, context and visible lengths, one
        # validity byte per position, and the exact radix TopK scratch.
        per_query = pages_per_request * (4 + 64) + 8 + 80 * topk
        return query_capacity * per_query + (num_sms + 1) * 2 * 4

    def get_workspace_bytes(self) -> int:
        """Report retained GPU workspace once per storage, excluding KV pools."""
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
            elif isinstance(value, CSA2SharedKVPlan):
                for item in vars(value).values():
                    visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and name != "position_ids":
                visit(value)
        for name in (
            "workspace",
            "cuda_graph_workspace",
            "_csa2_buffers",
            "_csa2_indexer_workspaces",
            "_csa2_priors",
            "_csa2_manager_states",
            "_csa2_shared_domains",
        ):
            visit(getattr(self, name, None))
        for metadata in getattr(self, "_csa2_query_tiles", {}).values():
            for name, value in vars(metadata).items():
                if name != "kv_cache_manager":
                    visit(value)
        return total

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

    def get_compression_batch(self, owner: int):
        return self._csa2_compression.get(owner)

    def get_compressed_positions(self, owner: int) -> torch.Tensor:
        return self._csa2_compressed_positions[owner]
