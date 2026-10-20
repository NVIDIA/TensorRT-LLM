# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch golden reference for CSA2 sparse attention.

:class:`CSA2VanillaAttention` reproduces what :class:`~.backend.CSA2TrtllmAttention`
computes for one CSA2 layer, and :class:`CSA2VanillaIndexer` reproduces the
selection of :class:`~.indexer.CSA2Indexer`. A model forward reaches a layer as
one call per phase (all context queries, then all generation queries); the
``query_start == 0`` call additionally owns the layer's cache writes and
selection:

1. Cache append (first call). Every new token writes its SWA row; a FULL layer
   also writes every compressed group completed by this forward into its main
   and index caches. Rows pass through the module's ``RowTransform`` (RMSNorm
   and/or interleaved RoPE) and are packed with the cache encoders of
   ``quantization.py``.
2. Selection (first call). FULL and REINDEX layers score the KV owner's index
   rows, ``s(t, g) = sum_h w[t, h] * relu(Q[t, h] . K[g])`` on MXFP4-decoded
   128-wide rows, over the visible and mapped entries (REINDEX layers that
   consume candidates only over the published candidates), keep the
   ``index_topk`` best entries and publish them as
   ``metadata.csa2_indices[layer]``. The candidate source also publishes
   ``metadata.csa2_candidates[layer]``. REUSE layers read their index source's
   publication.
3. Attention (every call). Each query attends to the union of its SWA window
   and its selected main rows, with an attention sink:
   ``out = sum_j e^{z_j} v_j / (sum_j e^{z_j} + e^{sink})`` and
   ``z_j = q . k_j / (q_scaling * sqrt(512))``. Keys and values are the same
   decoded 512-wide rows; a query with no rows outputs zeros.

Independence and coverage:

- No CuTe DSL, DeepGEMM, Triton, FlashMLA or FlashInfer code runs here, and
  neither ``manager.write_*``, the ``CSA2Indexer`` stack nor
  ``TrtllmAttention.forward``. Packing, decoding and the row transforms use the
  pure-PyTorch helpers of ``quantization.py``; attention uses
  ``VanillaAttention._selected_mla_attention`` (FP32 logits, BF16
  probabilities, as the native kernel stages them).
- Geometry comes from independent sources: positions from
  ``kv_lens_cuda - seq_lens`` of the request metadata and pages from the
  manager's host ``get_cache_indices``. The golden never reads the
  kernel-produced ``csa2_swa_indices``, ``csa2_swa_write_slots``,
  ``csa2_main_write_slots``, ``csa2_global_page_tables`` or
  ``csa2_visible_lengths``.
- Tests construct this class directly: the CSA2 module builds its backend
  with ``create_attention("TRTLLM", ...)`` and the sparse registry rejects
  VANILLA for ``csa2``, so a model run never reaches it. It still exposes the
  module-facing attributes (``compute_backend``, ``staging_dtype``,
  ``Metadata``) the module reads from its backend.
- The module owns, and this golden does not validate, the projections, the
  compressor, the RoPE of Q and index Q, the index weights, the inverse output
  RoPE and the grouped output projection.
- Exact where production approximates: FP8 staging, heuristic or GVR TopK and
  pre-packed index queries are accepted, but the golden always decodes exactly,
  stages BF16 and selects with an exact top-k.
- Rejected: packed attention and fused output RoPE, replay geometry (replay
  mode, SWA floors, a decoder query boundary), remote-tail execution, CED
  precomputed global KV, external source batches, context-only caches,
  mixed-phase calls, CUDA graph capture and scaled or quantized outputs. The
  forward reads metadata on the host, so it runs eagerly.

Notation: T packed query rows of the model forward, request-major with context
requests first; H query heads; D = 512 = 448 + 64 RoPE channels; W the
sliding window; r the layer's compression ratio (0 for SWA-only layers); B
``tokens_per_block``; K ``index_topk``. A token sits at absolute position p of
its request. Compressed entry g covers positions ``g * r .. g * r + r - 1`` and
is visible to p when ``g < (p + 1) // r``.

Coordinates: SWA row of position p is ``page * B + p % B`` with
``page = swa_pages[p // B]`` of the layer. Main and index rows of entry g are
``page * (B / r) + g % (B / r)`` with ``page = global_pages[g // (B / r)]`` of
the KV owner; the index cache stores row s in 64-row page-footer pages (64 data
bytes per row, then 4 scale bytes per row). An unmapped page (``-1``) has no
row.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PredefinedAttentionMask,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttention
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

from .._vanilla_utils import phase_lens, token_positions
from ..dsa.vanilla_backend import DSAVanillaIndexer
from .backend import _CSA2_KV_CACHE_DTYPES, _HEAD_DIM, _csa2_kv_cache_algo
from .cache_manager import CSA2CacheRole
from .metadata import CSA2TrtllmMetadata
from .params import (
    CSA2BackendForwardArgs,
    CSA2ForwardState,
    CSA2Layer,
    CSA2Layout,
    CSA2Mode,
    CSA2Params,
    RowTransform,
)
from .quantization import (
    INDEX_DATA_BYTES,
    INDEX_PAGE_BYTES,
    INDEX_PAGE_ROWS,
    INDEX_SCALE_BYTES,
    apply_row_transform,
    index_slot_offsets,
    pack_rows,
    read_index_rows,
    row_bytes,
    unpack_rows,
)

# Width of a CSA2 index row (query head or key) and of its packed form.
_INDEX_HEAD_DIM = 128
_INDEX_PACKED_DATA_BYTES = _INDEX_HEAD_DIM // 2
# The MLA geometry CSA2TrtllmAttention requires: (kv_lora_rank, qk_nope_head_dim,
# qk_rope_head_dim, v_head_dim, rope_append).
_MLA_GEOMETRY = (448, 448, 64, 512, False)
# Work bounds of the golden's own transients: decoded attention rows per chunk
# and FP32 index-score elements per chunk.
_DECODE_ROW_BUDGET = 1 << 17
_SCORE_ELEMENT_BUDGET = 1 << 26


@dataclass(frozen=True)
class _ForwardBatch:
    """Request geometry of one model forward, from the request metadata.

    Attributes:
        request_ids: Scheduler request IDs, in packed order.
        seq_lens: New tokens per request.
        past_lens: Cached tokens per request, ``kv_lens_cuda - seq_lens``: the
            position of the request's first new token.
        query_starts: Packed row of each request's first new token.
        num_ctx_tokens: Rows of the context requests, which come first.
        positions: int64 [T] absolute position of every packed row.
        requests: int64 [T] batch index of every packed row's request.
    """

    request_ids: tuple[int, ...]
    seq_lens: tuple[int, ...]
    past_lens: tuple[int, ...]
    query_starts: tuple[int, ...]
    num_ctx_tokens: int
    positions: torch.Tensor
    requests: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return self.positions.numel()

    def rows(self, request: int) -> tuple[int, int]:
        """Packed rows ``[begin, end)`` of batch request ``request``."""
        begin = self.query_starts[request]
        return begin, begin + self.seq_lens[request]


def _forward_batch(metadata: CSA2TrtllmMetadata, device: torch.device) -> _ForwardBatch:
    """Derive the packed request geometry of ``metadata`` on ``device``.

    Raises:
        ValueError: If the request IDs do not match the request count.
    """
    _, _, seq_lens, past_lens = phase_lens(metadata, None)
    request_ids = tuple(metadata.request_ids)
    if len(request_ids) != len(seq_lens):
        raise ValueError("CSA2 request metadata needs one request ID per request")
    query_starts, start = [], 0
    for length in seq_lens:
        query_starts.append(start)
        start += length
    lengths = torch.tensor(seq_lens, dtype=torch.int64, device=device)
    return _ForwardBatch(
        request_ids=request_ids,
        seq_lens=tuple(seq_lens),
        past_lens=tuple(past_lens),
        query_starts=tuple(query_starts),
        num_ctx_tokens=sum(seq_lens[: metadata.num_contexts]),
        positions=token_positions(seq_lens, past_lens, device).long(),
        requests=torch.repeat_interleave(torch.arange(len(seq_lens), device=device), lengths),
    )


def _host_page_table(
    manager, request_ids: tuple[int, ...], layer_idx: int, role: CSA2CacheRole, width: int, device
) -> torch.Tensor:
    """Return int64 [requests, width] pages of ``role``, read from the manager on the host.

    Columns past a request's page list are unmapped (``BAD_PAGE_INDEX``).
    """
    rows = []
    for request_id in request_ids:
        pages = list(manager.get_cache_indices(request_id, layer_idx, role))[:width]
        rows.append(pages + [BAD_PAGE_INDEX] * (width - len(pages)))
    return torch.tensor(rows, dtype=torch.int64, device=device).reshape(len(request_ids), width)


def _paged_slots(
    table: torch.Tensor,
    requests: torch.Tensor,
    indices: torch.Tensor,
    rows_per_page: int,
    limit: int,
) -> torch.Tensor:
    """Map request-local row indices to pool rows through a page table.

    Index i of request ``req`` lives at
    ``table[req, i // rows_per_page] * rows_per_page + i % rows_per_page``.

    Args:
        table: int64 [requests, pages] page table; negative entries are unmapped.
        requests: int64 [N] batch request of every row of ``indices``.
        indices: int64 [N, M] request-local positions or compressed entries.
        rows_per_page: Rows one page holds.
        limit: Exclusive bound of a valid index.

    Returns:
        int64 [N, M] pool rows; ``-1`` where the index is negative, at or past
        ``limit``, past the page table or on an unmapped page.
    """
    if table.shape[1] == 0:
        return torch.full_like(indices, -1)
    columns = indices.clamp_min(0) // rows_per_page
    valid = (indices >= 0) & (indices < limit) & (columns < table.shape[1])
    pages = table[requests[:, None], columns.clamp(max=table.shape[1] - 1)]
    valid &= pages >= 0
    return torch.where(valid, pages * rows_per_page + indices % rows_per_page, -1)


def _write_rows(
    pool: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
    cache_format: str,
    transform: Optional[RowTransform],
) -> None:
    """Pack ``values`` [N, dim] into ``pool`` [rows, row_bytes] at ``slots`` [N].

    The transform is applied first, as the production quantize launch does;
    slots outside the pool, including ``-1``, write nothing.
    """
    packed = pack_rows(apply_row_transform(values, transform), cache_format)
    valid = (slots >= 0) & (slots < pool.shape[0])
    pool[slots[valid]] = packed[valid]


def _write_index_rows(pages: torch.Tensor, slots: torch.Tensor, packed: torch.Tensor) -> None:
    """Scatter packed [N, 68] index rows into page-footer pages at ``slots`` [N].

    Row s lives in 64-row native page ``s // 64``: its 64 data bytes start at
    byte ``(s % 64) * 64`` of the page and its 4 scale bytes at
    ``64 * 64 + (s % 64) * 4``. Slots outside the pages write nothing.
    """
    flat = pages.view(-1)
    capacity = flat.numel() // INDEX_PAGE_BYTES * INDEX_PAGE_ROWS
    valid = (slots >= 0) & (slots < capacity)
    data_offsets, scale_offsets = index_slot_offsets(slots[valid])
    packed = packed[valid]
    data = torch.arange(INDEX_DATA_BYTES, device=flat.device)
    scales = torch.arange(INDEX_SCALE_BYTES, device=flat.device)
    flat[data_offsets[:, None] + data] = packed[:, :INDEX_DATA_BYTES]
    flat[scale_offsets[:, None] + scales] = packed[:, INDEX_DATA_BYTES:]


def _check_transform_positions(
    transform: Optional[RowTransform],
    positions: torch.Tensor,
    mapped: torch.Tensor,
    name: str,
) -> None:
    """Require a RoPE transform to rotate rows ``0 .. N-1`` at ``positions`` [N].

    Only rows with ``mapped`` [N] set are compared: production rotates a group
    on an unmapped page at position 0, and that row is never stored.
    """
    if transform is None or transform.rope is None:
        return
    rope_positions = transform.rope[0].reshape(-1)
    count = positions.numel()
    if rope_positions.numel() < count:
        raise ValueError(f"CSA2 {name} RoPE transform covers fewer rows than the golden's groups")
    rope_positions = rope_positions[:count].to(device=positions.device, dtype=torch.int64)
    if not torch.equal(rope_positions[mapped], positions[mapped]):
        raise ValueError(
            f"CSA2 {name} RoPE positions differ from the golden's compressed-group positions"
        )


class CSA2VanillaIndexer:
    """Weightless FP32 golden of CSA2 selection and candidate publication.

    Mirrors :class:`~.indexer.CSA2Indexer` for one FULL or REINDEX layer:

    - queries: the layer's 128-wide index queries, MXFP4-quantized and decoded
      (BF16 queries are packed here; pre-packed queries are decoded as given);
    - keys: the KV owner's index rows of entries ``0 .. visible - 1``, decoded
      from the index cache;
    - scores: :meth:`scores`, ``sum_h w[t, h] * relu(Q[t, h] . K[g])`` in FP32;
    - domain of query t: entries ``g < (p + 1) // r`` on mapped pages, and for
      a candidate consumer only the candidates the candidate source published;
    - selection: :meth:`select`, the ``index_topk`` best entries of the domain
      (all of them when the domain is no larger), sorted, padded with ``-1``;
    - candidate source: :meth:`candidates` over the same scores.

    Production approximates where this golden is exact: TopK may run heuristic
    or GVR selection, scores are accumulated by DeepGEMM kernels, and ties may
    break differently.

    Args:
        layout: Model layer layout.
        layer_idx: The FULL or REINDEX layer this indexer serves.
        heads: Index heads of the queries, H_i.
        head_dim: Index head width; CSA2 index rows are 128 wide.
        options: Lowered CSA2 parameters, kept for signature parity; the golden
            ignores the TopK and kernel options.

    Raises:
        ValueError: For a layer without an indexer or a head width other than 128.
    """

    def __init__(
        self,
        layout: CSA2Layout,
        layer_idx: int,
        heads: int,
        head_dim: int,
        options: Optional[CSA2Params] = None,
    ) -> None:
        layer = layout.layer(layer_idx)
        if layer.mode not in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            raise ValueError("Only CSA2 Full/Reindex layers construct an indexer")
        if head_dim != _INDEX_HEAD_DIM or heads <= 0:
            raise ValueError("CSA2 index queries need positive heads of width 128")
        self.layout = layout
        self.layer = layer
        self.layer_idx = layer_idx
        self.heads = heads
        self.head_dim = head_dim
        self.options = options or CSA2Params(layout=layout)

    def __call__(self, state: CSA2ForwardState, query_start: int, count: int) -> torch.Tensor:
        """Select every query of the model forward and publish the results.

        Args:
            state: Forward state; its ``metadata`` is the request metadata, its
                ``index_q`` BF16 [T, H_i, 128] (or packed int8 [T, H_i, 64] with
                ``index_q_scale``, four UE8M0 bytes per head) and its
                ``index_weights`` [T, H_i].
            query_start: Must be 0: selection covers the complete forward.
            count: Must be T.

        Returns:
            int32 [T, index_topk] sorted logical entries per query, ``-1``
            padded, also published as ``metadata.csa2_indices[layer]``.

        Raises:
            ValueError: For missing inputs, partial batches, mismatched shapes
                or candidates the candidate source did not publish.
        """
        metadata = state.metadata
        manager = metadata.kv_cache_manager
        if manager is None or state.index_q is None or state.index_weights is None:
            raise ValueError("CSA2 indexer requires an owner cache, queries and head weights")
        total = state.swa_kv.shape[0]
        if query_start != 0 or count != total:
            raise ValueError("CSA2 indexer must run once for the complete model query batch")
        if state.index_weights.shape != (count, self.heads):
            raise ValueError("CSA2 index weights must match the layer's query rows and heads")
        device = state.swa_kv.device
        batch = _forward_batch(metadata, device)
        if batch.num_tokens != count:
            raise ValueError("CSA2 index queries must cover every row of the request metadata")
        queries = self._decode_queries(state, count)
        weights = state.index_weights.float()

        layer = self.layer
        owner, ratio = layer.kv_source, layer.compress_ratio
        block = manager.tokens_per_block
        rows_per_page = block // ratio
        max_positions = manager.max_seq_len // ratio
        table = _host_page_table(
            manager,
            batch.request_ids,
            owner,
            CSA2CacheRole.GLOBAL,
            -(-max_positions // rows_per_page),
            device,
        )
        index_pages = manager.get_index_pages(owner)
        visible = (batch.positions + 1) // ratio

        candidates = None
        if layer.candidate_source is not None and layer.candidate_source != self.layer_idx:
            candidates = metadata.csa2_candidates.get(layer.candidate_source)
            if candidates is None or candidates.ndim != 2 or candidates.shape[0] != count:
                raise ValueError(
                    "CSA2 candidate source must publish matching full-query rows first"
                )
        publish = layer.candidate_source == self.layer_idx
        block_size = self.layout.candidate_block_size
        candidate_width = (
            min(self.layout.candidate_topk_blocks, -(-max_positions // block_size)) * block_size
        )
        topk = self.layout.index_topk
        selected = torch.full((count, topk), -1, dtype=torch.int32, device=device)
        published = (
            torch.full((count, candidate_width), -1, dtype=torch.int32, device=device)
            if publish
            else None
        )
        for request in range(len(batch.request_ids)):
            begin, end = batch.rows(request)
            if begin == end:
                continue
            # Entries visible to the request's last query bound every query of it.
            width = min(
                max_positions, (batch.past_lens[request] + batch.seq_lens[request]) // ratio
            )
            entries = torch.arange(width, device=device)
            slots = _paged_slots(
                table, batch.requests[begin : begin + 1], entries[None, :], rows_per_page, width
            )[0]
            keys = unpack_rows(
                read_index_rows(index_pages, slots), _INDEX_HEAD_DIM, "index", torch.float32
            )
            chunk = max(1, _SCORE_ELEMENT_BUDGET // max(1, self.heads * width))
            for first in range(begin, end, chunk):
                last = min(first + chunk, end)
                domain = (entries[None, :] < visible[first:last, None]) & (slots[None, :] >= 0)
                if candidates is not None:
                    domain &= self._candidate_mask(candidates[first:last], width)
                scores = self.scores(queries[first:last], keys, weights[first:last], domain)
                selected[first:last] = self.select(scores, topk)
                if published is not None:
                    published[first:last] = self.candidates(
                        scores,
                        visible[first:last],
                        self.layout.candidate_topk_blocks,
                        block_size,
                        candidate_width,
                    )
        metadata.csa2_indices[self.layer_idx] = selected
        if published is not None:
            metadata.csa2_candidates[self.layer_idx] = published
        return selected

    def _decode_queries(self, state: CSA2ForwardState, count: int) -> torch.Tensor:
        """Return FP32 [T, H_i, 128] MXFP4-decoded index queries.

        BF16 queries are packed with the index encoder first, as production
        packs them before scoring; packed queries are decoded as given.
        """
        if state.index_q_scale is None:
            if state.index_q.shape != (count, self.heads, self.head_dim):
                raise ValueError("CSA2 index queries must match the layer's full-query geometry")
            rows = pack_rows(state.index_q, "index")
        else:
            if state.index_q.dtype not in (torch.int8, torch.uint8) or state.index_q.shape != (
                count,
                self.heads,
                _INDEX_PACKED_DATA_BYTES,
            ):
                raise ValueError("Packed CSA2 index queries require 128D E2M1 data")
            scales = state.index_q_scale.contiguous().view(torch.uint8)
            if scales.numel() != count * self.heads * INDEX_SCALE_BYTES:
                raise ValueError(
                    "Packed CSA2 index queries require four UE8M0 scale bytes per head"
                )
            rows = torch.cat(
                (
                    state.index_q.contiguous().view(torch.uint8),
                    scales.reshape(count, self.heads, INDEX_SCALE_BYTES),
                ),
                dim=-1,
            )
        return unpack_rows(rows, self.head_dim, "index", torch.float32)

    @staticmethod
    def _candidate_mask(candidates: torch.Tensor, width: int) -> torch.Tensor:
        """Return bool [n, width] marking each row's candidate entries below ``width``."""
        entries = candidates.long()
        inside = (entries >= 0) & (entries < width)
        mask = torch.zeros(
            (entries.shape[0], width + 1), dtype=torch.bool, device=candidates.device
        )
        mask.scatter_(1, torch.where(inside, entries, width), True)
        return mask[:, :width]

    @staticmethod
    def scores(
        queries: torch.Tensor,
        keys: torch.Tensor,
        weights: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return FP32 [n, S] index scores, ``-inf`` outside ``valid``.

        Args:
            queries: FP32 [n, H_i, 128] decoded index queries.
            keys: FP32 [S, 128] decoded index keys of entries ``0 .. S - 1``.
            weights: FP32 [n, H_i] index head weights.
            valid: bool [n, S] selection domain.
        """
        scores = DSAVanillaIndexer.weighted_relu_scores(queries, keys, weights, valid)
        return scores.masked_fill(~valid, -torch.inf)

    @staticmethod
    def select(scores: torch.Tensor, topk: int) -> torch.Tensor:
        """Return int32 [n, topk]: the best finite columns of each row, sorted, ``-1`` padded.

        Column s of ``scores`` [n, S] is logical entry s. A row with at most
        ``topk`` finite scores keeps all of them.
        """
        count, width = scores.shape
        result = torch.full((count, topk), -1, dtype=torch.int32, device=scores.device)
        keep = min(topk, width)
        if keep == 0:
            return result
        values, columns = scores.topk(keep, dim=-1)
        sentinel = torch.iinfo(torch.int64).max
        ordered = torch.where(values > -torch.inf, columns, sentinel).sort(dim=-1).values
        result[:, :keep] = torch.where(ordered == sentinel, -1, ordered).to(torch.int32)
        return result

    @staticmethod
    def candidates(
        scores: torch.Tensor,
        visible: torch.Tensor,
        topk_blocks: int,
        block_size: int,
        width: int,
    ) -> torch.Tensor:
        """Return int32 [n, width] candidate entries published by the candidate source.

        Entries form blocks ``b * block_size .. b * block_size + block_size - 1``.
        A block scores the maximum of its entries' scores [n, S], and the block
        holding the newest visible entry scores ``+inf``. The ``topk_blocks``
        best blocks with a finite score (every block when no more exist) are
        listed in ascending order, each expanded to its entries; entries at or
        past ``visible`` [n] are ``-1``, and the row is padded with ``-1``.
        """
        count, entries = scores.shape
        result = torch.full((count, width), -1, dtype=torch.int32, device=scores.device)
        num_blocks = -(-entries // block_size)
        if num_blocks == 0:
            return result
        padded = torch.nn.functional.pad(
            scores, (0, num_blocks * block_size - entries), value=-torch.inf
        )
        blocks = padded.reshape(count, num_blocks, block_size).amax(-1)
        block_ids = torch.arange(num_blocks, device=scores.device)
        newest = (visible.long() - 1) // block_size
        blocks = blocks.masked_fill(block_ids[None, :] == newest[:, None], torch.inf)
        keep = min(topk_blocks, num_blocks)
        values, chosen = blocks.topk(keep, dim=-1)
        sentinel = torch.iinfo(torch.int64).max
        ordered = torch.where(values > -torch.inf, chosen, sentinel).sort(dim=-1).values
        listed_block = ordered != sentinel
        positions = ordered.masked_fill(~listed_block, 0)[..., None] * block_size + torch.arange(
            block_size, device=scores.device
        )
        valid = listed_block[..., None] & (positions < visible[:, None, None])
        listed = torch.where(valid, positions, -1).flatten(1)
        result[:, : listed.shape[1]] = listed.to(torch.int32)
        return result


class CSA2VanillaAttention(VanillaAttention):
    """Pure-PyTorch golden of the CSA2 sparse attention backend.

    This is a drop-in replacement for :class:`~.backend.CSA2TrtllmAttention`:
    it keeps that class's constructor and ``forward`` signatures, its
    constructor guards, the request metadata class and the attributes the CSA2
    module reads (``compute_backend``, ``staging_dtype``, ``sparse_params``).
    ``compute_backend`` is ``"vanilla"``, so a module would call ``forward``
    rather than a Flash helper. Pipeline:

    - cache append on the ``query_start == 0`` call (:meth:`_write_layer_rows`);
    - selection on the same call: :class:`CSA2VanillaIndexer` for FULL and
      REINDEX layers, the index source's publication for REUSE layers;
    - SWA and main row addresses (:meth:`_swa_slots`, :meth:`_main_slots`);
    - sink attention over the decoded rows (:meth:`_attend_rows`, :meth:`_attend`).

    Caches are the persistent CSA2 pools of :class:`~.cache_manager.CSA2CacheManager`:
    SWA uint8 [rows, 528], main uint8 [rows, 288] and index page-footer pages
    uint8 [pages, 64, 1, 68]. Queries are BF16 [count, H * 512] after Q RoPE.

    Without ``state`` in the sparse arguments, ``forward`` follows the
    attention-only contract: ``swa_indices`` and ``topk_indices`` address the
    given pools directly (``-1`` padding), or ``main_mapping`` maps logical
    ``topk_indices`` through a page table as in production.
    """

    Metadata = CSA2TrtllmMetadata
    compute_backend = "vanilla"
    # The golden decodes exactly into BF16 for every requested staging dtype.
    staging_dtype = torch.bfloat16

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int = 1,
        sparse_params: Optional[CSA2Params] = None,
        mla_params: Optional[MLAParams] = None,
        **kwargs,
    ) -> None:
        """Build the golden for one CSA2 layer.

        Args:
            layer_idx: Global model layer index.
            num_heads: Query heads on this rank, H.
            head_dim: Must be 512.
            num_kv_heads: Must be 1.
            sparse_params: Lowered CSA2 parameters; their ``layout`` is required
                once a forward carries module state. Defaults to ``CSA2Params()``.
            mla_params: CSA2 MLA geometry (kv_lora_rank 448, qk_nope 448, RoPE
                64, V 512, ``rope_append=False``); defaults to it.
            **kwargs: Forwarded to :class:`VanillaAttention`. ``kv_cache_dtype``
                may be auto, bfloat16, fp8 or fp8_ds_mla and ``quant_config`` may
                request only FP8 KV-cache quantization; both select production's
                staging dtype, while the golden always stages exact BF16.

        Raises:
            ValueError: For a geometry, cache dtype, quantization or option
                production rejects, or for packed attention, which the golden
                does not reproduce.
        """
        if head_dim != _HEAD_DIM:
            raise ValueError("CSA2 TRTLLM sparse MLA requires head_dim=512")
        if num_kv_heads != 1:
            raise ValueError("CSA2 TRTLLM sparse MLA requires one KV head")
        if kwargs.get("kv_cache_dtype", "auto") not in _CSA2_KV_CACHE_DTYPES:
            raise ValueError("CSA2 KV cache dtype must be auto, bfloat16, fp8 or fp8_ds_mla")
        _csa2_kv_cache_algo(kwargs.get("quant_config"))
        sparse_params = sparse_params or CSA2Params()
        if sparse_params.compute_backend not in ("auto", "trtllm", "flash_mla", "flashinfer"):
            raise ValueError(f"Unknown CSA2 attention backend: {sparse_params.compute_backend}")
        if sparse_params.use_packed_sparse_attention:
            raise ValueError("CSA2 Vanilla attention does not reproduce packed sparse attention")
        if mla_params is None:
            mla_params = MLAParams(
                q_lora_rank=1280,
                kv_lora_rank=448,
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                v_head_dim=512,
                rope_append=False,
            )
        if (
            mla_params.kv_lora_rank,
            mla_params.qk_nope_head_dim,
            mla_params.qk_rope_head_dim,
            mla_params.v_head_dim,
            mla_params.rope_append,
        ) != _MLA_GEOMETRY:
            raise ValueError(
                "CSA2 TRTLLM MLA requires rank/nope 448, RoPE 64, V 512 and rope_append=False"
            )
        super().__init__(
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            sparse_params=sparse_params,
            mla_params=mla_params,
            **kwargs,
        )
        # Created by the first forward that carries index queries, as in production.
        self.indexer: Optional[CSA2VanillaIndexer] = None

    @property
    def native_fp8(self) -> bool:
        """Whether attention consumes E4M3 staging; never for the golden."""
        return False

    @classmethod
    def support_fused_rope(cls) -> bool:
        """Report the production capability; CSA2 Q arrives already rotated."""
        return True

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]) -> None:
        """Adopt ``new_quant_config`` after the KV-cache check production applies."""
        _csa2_kv_cache_algo(new_quant_config)
        super().update_quant_config(new_quant_config)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: CSA2TrtllmMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run CSA2 attention for one phase tile of the model forward.

        Args:
            q: BF16 [count, H * 512] rotated queries.
            k: Must be None; keys come from the caches.
            v: Must be None; values come from the caches.
            metadata: The tile's compute metadata, as ``get_query_tile_metadata``
                returns it; only ``num_tokens`` and ``num_query_heads`` are read.
            forward_args: ``attention_input_type`` (context_only or
                generation_only), ``sparse_backend_args`` (a
                :class:`~.params.CSA2BackendForwardArgs`; with ``state`` and
                ``query_start`` it addresses rows ``[query_start, query_start +
                count)`` of the model forward), optional float32 [H]
                ``attention_sinks`` and an optional BF16 [count, H * 512]
                ``output``.
            **kwargs: Merged into ``forward_args``.

        Returns:
            BF16 [count, H * 512] attention output: ``forward_args.output`` when
            given, otherwise a new tensor. A ``query_start == 0`` call with
            state also writes the layer's cache rows and publishes its selection.

        Raises:
            TypeError: If the sparse arguments are not CSA2 arguments.
            ValueError: For inconsistent inputs or an unsupported option.
            RuntimeError: Under CUDA graph capture.
        """
        args = merge_attention_forward_args(forward_args, kwargs)
        if k is not None or v is not None:
            raise ValueError("CSA2 attention reads keys and values from its caches")
        inputs = args.sparse_backend_args
        if not isinstance(inputs, CSA2BackendForwardArgs):
            raise TypeError("CSA2 requires CSA2BackendForwardArgs")
        count = metadata.num_tokens
        if (
            q.shape != (count, self.num_heads * _HEAD_DIM)
            or q.dtype != torch.bfloat16
            or metadata.num_query_heads != self.num_heads
        ):
            raise ValueError("CSA2 Q geometry/dtype must match its BF16 metadata")
        if args.attention_input_type not in (
            AttentionInputType.context_only,
            AttentionInputType.generation_only,
        ):
            raise ValueError("CSA2 compute tiles must contain one attention phase")
        self._check_output_args(q, args, inputs)
        if q.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CSA2 Vanilla attention reads metadata on the host and cannot be captured"
            )
        sink = args.attention_sinks
        if sink is not None and tuple(sink.shape) != (self.num_heads,):
            raise ValueError(f"CSA2 attention sinks must have shape [{self.num_heads}]")
        if inputs.state is None:
            swa_pool, swa_slots, main_pool, main_slots = self._pool_slots(inputs, count)
        else:
            swa_pool, swa_slots, main_pool, main_slots = self._model_slots(
                inputs.state, inputs.query_start, count
            )
        result = self._attend_rows(
            q.view(count, self.num_heads, _HEAD_DIM),
            swa_pool,
            swa_slots,
            main_pool,
            main_slots,
            sink,
            1.0 / ((self.q_scaling or 1.0) * _HEAD_DIM**0.5),
        ).view(count, self.num_heads * _HEAD_DIM)
        if args.output is None:
            return result
        args.output.copy_(result)
        return args.output

    @staticmethod
    def _check_output_args(
        q: torch.Tensor, args: AttentionForwardArgs, inputs: CSA2BackendForwardArgs
    ) -> None:
        """Reject masks, output formats and fused options the golden does not model."""
        if (
            args.attention_mask
            not in (PredefinedAttentionMask.CAUSAL, PredefinedAttentionMask.FULL)
            or args.attention_mask_data is not None
            or args.out_scale is not None
            or args.out_scale_sf is not None
            or args.output_sf is not None
            or args.quant_q_buffer is not None
            or args.quant_scale_qkv is not None
            or args.enable_dsv4_epilogue_fusion
        ):
            raise ValueError("CSA2 Vanilla attention does not support this mask or output format")
        state = inputs.state
        if (
            inputs.output_position_ids is not None
            or inputs.output_rotary_cos_sin is not None
            or (
                state is not None
                and (
                    state.output_position_ids is not None or state.output_rotary_cos_sin is not None
                )
            )
        ):
            raise ValueError("CSA2 fused output RoPE requires packed sparse attention")
        output = args.output
        if output is not None and (
            output.shape != q.shape
            or output.dtype != q.dtype
            or output.device != q.device
            or not output.is_contiguous()
        ):
            raise ValueError("CSA2 output must match contiguous BF16 query geometry")

    def _pool_slots(
        self, inputs: CSA2BackendForwardArgs, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return the attention-only contract's pools and pool rows.

        ``swa_indices`` [count, <= 128] and ``topk_indices`` [count, K] address
        their pools directly; with ``main_mapping`` the latter are logical
        entries mapped as production maps them.
        """
        swa_pool, swa_slots = inputs.swa_pool, inputs.swa_indices
        if swa_pool is None or swa_slots is None:
            raise ValueError("CSA2 requires selected SWA pool inputs")
        if swa_slots.ndim != 2 or swa_slots.shape[0] != count:
            raise ValueError("CSA2 SWA indices must match the query count")
        main_pool, main_slots = inputs.main_pool, inputs.topk_indices
        if main_slots is not None:
            if main_pool is None:
                raise ValueError("CSA2 selected main indices require a main pool")
            if main_slots.ndim != 2 or main_slots.shape[0] != count:
                raise ValueError("CSA2 selected main indices must match the query count")
            if inputs.main_mapping is not None:
                table, page_size, max_positions, requests, visible = inputs.main_mapping
                main_slots = CSA2TrtllmMetadata._map_global_slots(
                    main_slots, requests.long(), table, visible, page_size, max_positions
                )
        return swa_pool, swa_slots.long(), main_pool, main_slots

    def _model_slots(
        self, state: CSA2ForwardState, query_start: int, count: int
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return the pools and pool rows that queries ``[query_start, query_start + count)`` read.

        The ``query_start == 0`` call first writes the layer's cache rows and
        runs its selection.

        Returns:
            ``(swa_pool, swa_slots [count, W], main_pool, main_slots [count, K])``;
            the main entries are None on SWA-only layers.
        """
        source = state.metadata
        manager = source.kv_cache_manager
        layout = self.sparse_params.layout
        if layout is None:
            raise ValueError("CSA2 module inputs require the model layout")
        layer = layout.layer(self.layer_idx)
        self._check_supported_state(source, manager, layer)
        device = state.swa_kv.device
        batch = _forward_batch(source, device)
        total = batch.num_tokens
        end = query_start + count
        if state.swa_kv.shape != (total, _HEAD_DIM):
            raise ValueError("CSA2 SWA rows must cover every row of the request metadata")
        if query_start < 0 or end > total:
            raise ValueError("CSA2 query tile lies outside the model forward")
        if query_start < batch.num_ctx_tokens < end:
            raise ValueError("CSA2 query tiles must not cross the context/generation boundary")
        if query_start == 0:
            self._enter_layer(source, layer)
            self._write_layer_rows(state, layer, batch)
            if layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
                if state.index_q is None or state.index_weights is None:
                    raise ValueError("Full/Reindex mode requires index queries and weights")
                if self.indexer is None:
                    packed = state.index_q_scale is not None
                    self.indexer = CSA2VanillaIndexer(
                        layout,
                        self.layer_idx,
                        state.index_q.shape[1],
                        state.index_q.shape[-1] * (2 if packed else 1),
                        options=self.sparse_params,
                    )
                self.indexer(state, 0, total)
        positions = batch.positions[query_start:end]
        requests = batch.requests[query_start:end]
        swa_slots = self._swa_slots(manager, batch, requests, positions)
        if layer.mode == CSA2Mode.SWA:
            return manager.get_swa_buffer(self.layer_idx), swa_slots, None, None
        index_source = (
            self.layer_idx
            if layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX)
            else layer.index_source
        )
        if index_source not in source.csa2_indices:
            raise ValueError("CSA2 index source did not run in this forward")
        indices = source.csa2_indices[index_source]
        if indices.shape[0] != total:
            raise ValueError("CSA2 Reuse rows must match their source's packed query order")
        main_slots = self._main_slots(
            manager, batch, layer, requests, positions, indices[query_start:end]
        )
        return (
            manager.get_swa_buffer(self.layer_idx),
            swa_slots,
            manager.get_main_buffer(layer.kv_source),
            main_slots,
        )

    def _check_supported_state(self, source: CSA2TrtllmMetadata, manager, layer: CSA2Layer) -> None:
        """Reject request metadata whose geometry this golden does not reproduce.

        Replay and decoder-boundary geometry move SWA floors and GLOBAL sources,
        remote-tail and CED prepasses split cache ownership across passes, and
        context-only caches drop SWA layers; none of them is modeled.
        """
        if manager is None:
            raise ValueError("CSA2 module inputs require a KV cache manager")
        unsupported = []
        if getattr(source, "csa2_replay_mode", None) is not None:
            unsupported.append("bounded SWA replay")
        if getattr(source, "csa2_query_layer_start", 0):
            unsupported.append("a decoder query boundary")
        if getattr(source, "csa2_remote_tail_mode", None) is not None:
            unsupported.append("remote-tail execution")
        if getattr(source, "_csa2_swa_read_floors", None) is not None:
            unsupported.append("SWA read floors")
        floors = getattr(source, "csa2_replay_start_positions", None)
        if floors is not None and bool((floors > 0).any()):
            unsupported.append("SWA write floors")
        if getattr(manager, "context_swa_layer_limit", None) is not None:
            unsupported.append("context-only caches")
        if self.layer_idx in getattr(source, "csa2_precomputed_kv_layers", ()):
            unsupported.append("CED precomputed global KV")
        if unsupported:
            raise ValueError(f"CSA2 Vanilla attention does not model {', '.join(unsupported)}")

    @staticmethod
    def _enter_layer(source: CSA2TrtllmMetadata, layer: CSA2Layer) -> None:
        """Mirror ``CSA2TrtllmMetadata.enter_layer``'s ordering check.

        Layers of one forward must run in increasing order, so a REUSE layer
        never reads a selection published by an earlier forward. The SWA slot
        resolution ``enter_layer`` also launches serves only production.
        """
        if layer.layer_idx <= getattr(source, "_csa2_last_layer", -1):
            raise ValueError("CSA2 routing cannot be reused across forwards or reordered layers")
        source._csa2_last_layer = layer.layer_idx

    def _write_layer_rows(
        self, state: CSA2ForwardState, layer: CSA2Layer, batch: _ForwardBatch
    ) -> None:
        """Write this forward's SWA rows and, on a FULL layer, its completed groups.

        Every new token writes its SWA row (``swa_transform`` first). A FULL
        layer with ratio r writes each request's groups
        ``past // r .. (past + new) // r - 1``, packed request by request into
        the leading rows of ``main_kv`` [T, 512] and ``index_k`` [T, 128]; the
        remaining rows are padding. The first group of a request that resumes
        mid-group completes the group cached earlier. Rows on unmapped pages
        are not written.

        Raises:
            ValueError: If a FULL layer lacks main or index rows, another layer
                provides them, the rows do not match the request batch, or a
                RoPE transform disagrees with the group positions. Every check
                runs before the first write.
        """
        manager = state.metadata.kv_cache_manager
        global_slots = None
        if layer.mode == CSA2Mode.FULL:
            global_slots = self._group_slots(state, layer, batch)
        elif state.main_kv is not None or state.index_k is not None:
            raise ValueError("Only Full mode may write shared main/index caches")
        swa_slots = self._swa_slots(manager, batch, batch.requests, batch.positions[:, None])[:, 0]
        _write_rows(
            manager.get_swa_buffer(self.layer_idx),
            swa_slots,
            state.swa_kv,
            "swa",
            state.swa_transform,
        )
        if global_slots is None:
            return
        _write_rows(
            manager.get_main_buffer(layer.kv_source),
            global_slots,
            state.main_kv,
            "main",
            state.main_transform,
        )
        _write_index_rows(
            manager.get_index_pages(layer.kv_source),
            global_slots,
            pack_rows(apply_row_transform(state.index_k, state.index_transform), "index"),
        )

    @staticmethod
    def _group_slots(
        state: CSA2ForwardState, layer: CSA2Layer, batch: _ForwardBatch
    ) -> torch.Tensor:
        """Return int64 [T] main/index rows of a FULL layer's ``main_kv`` / ``index_k`` rows.

        Row i of the leading rows holds the i-th completed group in request
        order; padding rows and groups on unmapped pages map to ``-1``.

        Raises:
            ValueError: For missing or mis-sized rows, or RoPE transforms whose
                positions are not ``g * r`` of the derived groups on mapped
                pages.
        """
        if state.main_kv is None or state.index_k is None:
            raise ValueError("Full mode requires main and index rows, including empty rows")
        total = batch.num_tokens
        if state.main_kv.shape != (total, _HEAD_DIM) or state.index_k.shape != (
            total,
            _INDEX_HEAD_DIM,
        ):
            raise ValueError(
                "CSA2 Vanilla attention models global rows produced from the query batch: "
                f"main_kv [{total}, 512] and index_k [{total}, 128]"
            )
        manager = state.metadata.kv_cache_manager
        ratio = layer.compress_ratio
        groups, requests = [], []
        for request, (past, new) in enumerate(zip(batch.past_lens, batch.seq_lens)):
            first, last = past // ratio, (past + new) // ratio
            groups.extend(range(first, last))
            requests.extend([request] * (last - first))
        device = state.swa_kv.device
        group_ids = torch.tensor(groups, dtype=torch.int64, device=device)
        slots = torch.full((total,), -1, dtype=torch.int64, device=device)
        if groups:
            rows_per_page = manager.tokens_per_block // ratio
            max_positions = manager.max_seq_len // ratio
            table = _host_page_table(
                manager,
                batch.request_ids,
                layer.kv_source,
                CSA2CacheRole.GLOBAL,
                -(-max_positions // rows_per_page),
                device,
            )
            slots[: len(groups)] = _paged_slots(
                table,
                torch.tensor(requests, dtype=torch.int64, device=device),
                group_ids[:, None],
                rows_per_page,
                max_positions,
            )[:, 0]
        mapped = slots[: len(groups)] >= 0
        _check_transform_positions(state.main_transform, group_ids * ratio, mapped, "main")
        _check_transform_positions(state.index_transform, group_ids * ratio, mapped, "index")
        return slots

    def _swa_slots(
        self,
        manager,
        batch: _ForwardBatch,
        requests: torch.Tensor,
        logical: torch.Tensor,
    ) -> torch.Tensor:
        """Return int64 SWA pool rows of this layer for request-local positions.

        Args:
            manager: The request metadata's cache manager.
            batch: The forward's request geometry.
            requests: int64 [N] batch request of each row of ``logical``.
            logical: int64 [N, M] positions; a 1-D ``positions`` [N] becomes the
                query's window ``p - W + 1 .. p`` (oldest first).

        Returns:
            int64 [N, M] rows, ``-1`` for negative positions and unmapped pages.
        """
        if logical.ndim == 1:
            window = manager.layout.window_size
            logical = logical[:, None] - window + 1 + torch.arange(window, device=logical.device)
        block = manager.tokens_per_block
        table = _host_page_table(
            manager,
            batch.request_ids,
            self.layer_idx,
            CSA2CacheRole.SWA,
            -(-manager.max_seq_len // block),
            logical.device,
        )
        return _paged_slots(table, requests, logical, block, manager.max_seq_len)

    @staticmethod
    def _main_slots(
        manager,
        batch: _ForwardBatch,
        layer: CSA2Layer,
        requests: torch.Tensor,
        positions: torch.Tensor,
        logical: torch.Tensor,
    ) -> torch.Tensor:
        """Return int64 [n, K] main pool rows of the selected entries.

        Entries at or past the query's visible count ``(p + 1) // r`` and
        entries on unmapped GLOBAL pages of the KV owner resolve to ``-1``.
        """
        ratio = layer.compress_ratio
        rows_per_page = manager.tokens_per_block // ratio
        max_positions = manager.max_seq_len // ratio
        table = _host_page_table(
            manager,
            batch.request_ids,
            layer.kv_source,
            CSA2CacheRole.GLOBAL,
            -(-max_positions // rows_per_page),
            logical.device,
        )
        logical = logical.long()
        visible = (positions + 1) // ratio
        logical = torch.where(logical < visible[:, None], logical, -1)
        return _paged_slots(table, requests, logical, rows_per_page, max_positions)

    def _attend_rows(
        self,
        q: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_slots: torch.Tensor,
        main_pool: Optional[torch.Tensor],
        main_slots: Optional[torch.Tensor],
        sink: Optional[torch.Tensor],
        scale: float,
    ) -> torch.Tensor:
        """Decode each query's SWA and main rows and attend over the valid ones.

        Args:
            q: BF16 [n, H, 512] queries.
            swa_pool: uint8 [rows, 528] SWA pool.
            swa_slots: int64 [n, W] SWA rows; rows outside the pool are skipped.
            main_pool: uint8 [rows, 288] main pool, or None.
            main_slots: int64 [n, K] main rows, or None.
            sink: float32 [H] sink logits, or None.
            scale: Softmax scale.

        Returns:
            BF16 [n, H, 512] outputs; zero for a query without valid rows.
        """
        if swa_pool.ndim != 2 or swa_pool.dtype != torch.uint8:
            raise ValueError("CSA2 SWA pool must be uint8 [rows, 528]")
        if main_slots is not None and (main_pool.ndim != 2 or main_pool.dtype != torch.uint8):
            raise ValueError("CSA2 main pool must be uint8 [rows, 288]")
        sources = [(swa_pool, swa_slots, "swa")]
        if main_slots is not None:
            sources.append((main_pool, main_slots.long(), "main"))
        width = sum(slots.shape[1] for _, slots, _ in sources)
        output = torch.zeros_like(q)
        chunk = max(1, _DECODE_ROW_BUDGET // max(1, width))
        for first in range(0, q.shape[0], chunk):
            last = min(first + chunk, q.shape[0])
            rows, valid = [], []
            for pool, slots, cache_format in sources:
                if pool.shape[1] != row_bytes(_HEAD_DIM, cache_format):
                    raise ValueError(f"CSA2 {cache_format} pool has the wrong row width")
                part = slots[first:last]
                inside = (part >= 0) & (part < pool.shape[0])
                if pool.shape[0]:
                    packed = pool[torch.where(inside, part, 0)]
                else:
                    packed = pool.new_zeros((*part.shape, pool.shape[1]))
                rows.append(unpack_rows(packed, _HEAD_DIM, cache_format, q.dtype))
                valid.append(inside)
            rows, valid = torch.cat(rows, dim=1), torch.cat(valid, dim=1)
            for token in range(last - first):
                output[first + token] = self._attend(
                    q[first + token], rows[token][valid[token]], sink, scale
                )
        return output

    @staticmethod
    def _attend(
        q: torch.Tensor,
        rows: torch.Tensor,
        sink: Optional[torch.Tensor],
        scale: float,
    ) -> torch.Tensor:
        """Sink attention of one query over decoded rows that serve as keys and values.

        Args:
            q: BF16 [H, 512] query heads.
            rows: BF16 [N, 512] decoded rows.
            sink: float32 [H] sink logits, or None.
            scale: Softmax scale.

        Returns:
            BF16 [H, 512]; zeros when N is 0, as production's single zero row
            reduces to.
        """
        if rows.shape[0] == 0:
            return torch.zeros_like(q)
        return VanillaAttention._selected_mla_attention(
            q, rows, value_dim=_HEAD_DIM, scale=scale, attention_sink=sink
        )


__all__ = ["CSA2VanillaAttention", "CSA2VanillaIndexer"]
