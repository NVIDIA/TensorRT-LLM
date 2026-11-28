# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA attention metadata of the Kimi K3.1 Rubin targets (dep16, tep8, helix8).

``RubinDSAMetadata`` inherits the engine's DSA metadata (``attention/backends/sparse/dsa/metadata.py``)
with its fields and every method that is the engine's, with Helix context parallelism from the family's
``HelixDSAMetadataMixin`` (``dsa_metadata.py``). It overrides what the targets run differently
(``OVERRIDES``):

* ``prepare``: the indexer plan of the step. The prefill chunks carry this rank's query rows (the TP
  query split of a prefill under attention TP), their query tiles under the per-call logits budget
  and DeepGEMM's MQA-logits schedule of every tile, built once per step for all DSA layers. Both
  logits budgets are module constants. The decode schedule goes through the family entry.
* ``prepare_for_spec_decode``: the targets score a verify window as one DSL logits atom (next_n x
  32 index heads <= 256), so the metadata plans no DSL split. Nothing it writes depends on the
  step's lengths, so a captured graph and its replays agree, and no atom-offset tensor is built.
* ``on_update_kv_lens``: the generation step's decode metadata through the family entries, without
  the schedules the targets' indexer does not read.
* ``update_for_spec_dec``: a draft forward that reads no target metadata skips the device rebuild.
* ``__post_init__`` and ``warmup_selfsampling_topk``: the targets' Top-K is the deterministic
  self-sampling GVR on sm_107 too, whatever the sparse attention config's Top-K keys say; the
  warmup compiles that engine and the key-sharded Top-K's eligibility follows it.

``RubinDSAHelixMetadata`` adds Helix context parallelism (helix8): the declaration that one
KV-length refresh per input preparation suffices (``kv_lens_refresh_once``), the decode metadata
of a verify step in one launch, and the key-sharded Top-K's step inputs (``key_shard_next_n``, 0
when a step takes the CP all-gather merge instead). The metadata reads no environment variable and
holds no workspace a collective builds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import (
    Indexer,
    IndexerPrefillChunkMetadata,
    compute_cu_seqlen_kv_bounds_with_cache,
    plan_mla_chunked_prefill_chunks,
    select_indexer_chunk_size,
    split_prefill_chunks,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    DSAtrtllmAttentionMetadata,
    build_req_idx_per_token,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version, maybe_pin_memory
from tensorrt_llm.logger import logger

from ._rubin_dsa_indexer_plan import (
    compute_slot_mappings,
    effective_compress_ratio_divisor,
    select_indexer_compress_ratio,
    split_prefill_queries,
)
from .dsa_metadata import HelixDSAMetadataMixin
from .fused_dsa_decode_metadata import fused_dsa_decode_metadata
from .get_paged_mqa_logits_metadata import SCHEDULE_BLOCK_KV, get_paged_mqa_logits_metadata
from .rubin_fused_dsa_helix_decode_metadata import rubin_fused_dsa_helix_decode_metadata
from .rubin_gvr_topk_decode import rubin_gvr_topk_warmup
from .rubin_mqa_logits_metadata import rubin_mqa_logits_metadata

# Per-call cap on one indexer prefill MQA-logits call, in elements: a chunk's query rows are tiled
# so that rows x keys of one call stay within it. A serving configuration sets the KV cache
# estimate's reserve (TLLM_INDEXER_MQA_LOGITS_ELEM_BUDGET) to the same value.
_INDEXER_MQA_LOGITS_ELEM_BUDGET = 1 << 29
# Cap on the packed [sum(q) x sum(kv)] logits of the context requests one prefill chunk groups.
_INDEXER_PACKED_LOGITS_ELEM_BUDGET = 1 << 28
# Index heads of the K3.1 indexer.
_INDEX_N_HEADS = 32
# The FP8 DSL paged logits kernel's MMA N bound: next_n x index heads of one atom.
_DSL_MAX_ATOM_N = 256
# Page size the key-sharded Top-K kernels are built for (each CP rank holds whole pages).
_KEY_SHARD_TOKENS_PER_BLOCK = 64


def dsl_single_atom(next_n: int) -> bool:
    """Whether the DSL decode logits score a ``next_n``-row verify window as one atom."""
    return next_n * _INDEX_N_HEADS <= _DSL_MAX_ATOM_N


def packed_logits_elem_budget(max_num_tokens: int, max_seq_len: int) -> int:
    """The packed cap of a prefill chunk, clamped to one request's largest logits tile."""
    single_request_tile = min(_INDEXER_MQA_LOGITS_ELEM_BUDGET, max_num_tokens * max_seq_len)
    return min(_INDEXER_PACKED_LOGITS_ELEM_BUDGET, single_request_tile)


def query_tiles(q_start: int, q_end: int, num_k_tokens: int) -> Tuple[Tuple[int, int], ...]:
    """The ``[c0, c1)`` query tiles of rows ``[q_start, q_end)`` over ``num_k_tokens`` keys, each
    within the per-call logits budget (FP8 index keys: one logits element per row and key)."""
    rows = q_end - q_start
    q_tile = max(1, min(rows, _INDEXER_MQA_LOGITS_ELEM_BUDGET // max(1, num_k_tokens)))
    return tuple(
        (q_start + off, min(q_start + off + q_tile, q_end)) for off in range(0, rows, q_tile)
    )


def self_sampling_gvr(params, compress_ratio: int, sm_version: int) -> bool:
    """Whether the self-sampling GVR Top-K can run here (the engine's capability gate, with
    sm_107). The targets always run it, so the sparse attention config's Top-K keys
    (``enable_heuristic_topk``, ``use_self_sampling_topk``) do not decide."""
    return bool(
        IS_CUTLASS_DSL_AVAILABLE
        and sm_version in (100, 103, 107)
        and params.max_sparse_topk in (512, 1024, 2048)
        and compress_ratio in (1, 4)
    )


@dataclass
class RubinDsaPrefillChunk(IndexerPrefillChunkMetadata):
    """One indexer prefill chunk with this rank's query rows, their tiles and their schedules.

    ``cu_seqlen_ks`` / ``cu_seqlen_ke`` index the chunk's gathered keys. ``[q_start, q_end)`` are
    the chunk rows this rank scores; ``q_sizes`` every rank's row count when the chunk's rows are
    split over the TP group (the Top-K rows are then all-gathered), else None. ``tiles`` are the
    ``[c0, c1)`` chunk rows of one logits call and ``mqa_meta`` DeepGEMM's schedule of each (none
    when the step's context rows take the dense Top-K).
    """

    q_start: int = 0
    q_end: int = 0
    q_sizes: Optional[List[int]] = None
    tiles: Tuple[Tuple[int, int], ...] = ()
    mqa_meta: Tuple[torch.Tensor, ...] = ()


def _prefill_chunk(md, chunk_specs, q_split_threshold: int) -> RubinDsaPrefillChunk:
    """One chunk of the plan: the engine's bounds, then this rank's rows, tiles and schedules."""
    base = Indexer.prepare_one_prefill_chunk(md, chunk_specs)
    rows = base.token_end - base.token_start
    if md.indexer_prefill_q_split is None:
        q_start, q_end, q_sizes = split_prefill_queries(rows, md.mapping, q_split_threshold)
    else:
        # The batch plan already covers this rank's share of the queries only.
        q_start, q_end, q_sizes = 0, rows, None
    num_k_tokens = base.k_token_end - base.k_token_start
    tiles = query_tiles(q_start, q_end, num_k_tokens) if num_k_tokens > 0 else ()
    # A step whose context rows take the dense Top-K scores nothing: no schedule to build.
    metas = (
        ()
        if md.skip_indexer_for_ctx_reqs
        else tuple(
            rubin_mqa_logits_metadata(
                base.cu_seqlen_ks[c0:c1], base.cu_seqlen_ke[c0:c1], num_k_tokens, _INDEX_N_HEADS
            )
            for c0, c1 in tiles
        )
    )
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    return RubinDsaPrefillChunk(
        **fields, q_start=q_start, q_end=q_end, q_sizes=q_sizes, tiles=tiles, mqa_meta=metas
    )


def _plan_prefill(md, indexer_params) -> None:
    """The indexer's prefill chunks of the step's context requests.

    Under MLA chunked prefill the chunks group requests by the packed budget and, from
    ``q_split_threshold`` context rows on under attention TP, cover this rank's share of the
    queries only (``indexer_prefill_q_split``). Otherwise long requests split into query blocks of
    the indexer chunk size; a batch that fits one block runs as one pass (no chunks).
    """
    num_contexts = indexer_params.num_contexts
    seq_lens = indexer_params.seq_lens
    threshold = md.sparse_metadata_params.q_split_threshold
    md.indexer_prefill_q_split = None
    if md.enable_context_mla_with_cached_kv and md.runtime_features.chunked_prefill:
        ratio = effective_compress_ratio_divisor(select_indexer_compress_ratio(md.compress_ratios))
        q_lens = seq_lens[:num_contexts].tolist()
        cached = md.host_ctx_cached_token_indptr[: num_contexts + 1].tolist()
        kv_lens = [(cached[i + 1] - cached[i] + q_lens[i]) // ratio for i in range(num_contexts)]
        q_begin, q_end, q_sizes = split_prefill_queries(sum(q_lens), md.mapping, threshold)
        if q_sizes is not None:
            md.indexer_prefill_q_split = (q_begin, q_end, q_sizes)
        budget = packed_logits_elem_budget(md.max_num_tokens, md.kv_cache_manager.max_seq_len)
        specs = plan_mla_chunked_prefill_chunks(q_lens, kv_lens, budget, q_begin, q_end)
    else:
        max_k = int(indexer_params.kv_lens[:num_contexts].max().item()) if num_contexts else 0
        chunk_size = select_indexer_chunk_size(md.indexer_max_chunk_size, max_k)
        specs = split_prefill_chunks(seq_lens[:num_contexts], chunk_size, start_idx=0)
        if len(specs) <= 1 and not md.enable_context_mla_with_cached_kv:
            specs = None
    md.indexer_prefill_chunks = (
        None if specs is None else [_prefill_chunk(md, s, threshold) for s in specs]
    )
    # Chunked prefill and KV-cache reuse gather every cached and new index key of a request.
    Indexer.recompute_context_kv_gather_mappings(md, indexer_params)


def _decode_schedules(md, gen_indexer_kv_lens: torch.Tensor) -> None:
    """The decode logits' schedules of the generation rows, into the metadata's buffers.

    The targets' DSL logits read ``scheduler_metadata_buffer`` (one row per request, any verify
    width); per-token rows (Helix verify bounds, ragged verify) read the expanded schedule.
    """
    md.gen_indexer_kv_lens_cuda_runtime = gen_indexer_kv_lens
    md.update_indexer_kv_lens_2d(gen_indexer_kv_lens)
    get_paged_mqa_logits_metadata(
        gen_indexer_kv_lens.view(-1, 1), SCHEDULE_BLOCK_KV, md.num_sms, md.scheduler_metadata_buffer
    )
    if md.use_expanded_buffers_for_mtp:
        rows = sum(md.gen_token_repeat_list())
        get_paged_mqa_logits_metadata(
            md.kv_lens_expanded_cuda[:rows].view(-1, 1),
            SCHEDULE_BLOCK_KV,
            md.num_sms,
            md.scheduler_metadata_buffer_expanded,
        )


def _prepare_indexer(md) -> None:
    """The indexer's per-step metadata (the engine's ``Indexer.prepare`` with the family plan)."""
    indexer_params = Indexer.build_indexer_params(md)
    if indexer_params is None:
        return
    num_contexts = md.num_contexts
    num_ctx_tokens = md.num_ctx_tokens
    md.num_ctx_kv_tokens = indexer_params.new_kv_tokens[:num_contexts].sum().item()
    Indexer.prepare_for_update_k_cache(md, indexer_params)
    if num_contexts > 0:
        ratio = effective_compress_ratio_divisor(select_indexer_compress_ratio(md.compress_ratios))
        cu_ks, cu_ke = compute_cu_seqlen_kv_bounds_with_cache(
            md.seq_lens[:num_contexts],
            num_contexts,
            num_ctx_tokens,
            indexer_params._num_past_tokens_tensor[:num_contexts],
            indexer_params.kv_lens[:num_contexts],
            ratio,
        )
        md.cu_seqlen_ks[:num_ctx_tokens].copy_(maybe_pin_memory(cu_ks), non_blocking=True)
        md.cu_seqlen_ke[:num_ctx_tokens].copy_(maybe_pin_memory(cu_ke), non_blocking=True)
        _plan_prefill(md, indexer_params)
    else:
        md.indexer_prefill_chunks = None
        md.indexer_prefill_q_split = None
    if md.num_generations > 0:
        nc = num_contexts
        gen_kv_lens = md.kv_lens_cuda_runtime[nc : nc + md.num_generations]
        _decode_schedules(md, md.get_indexer_kv_lens(gen_kv_lens))


def _fused_decode_metadata(md) -> bool:
    """One launch for a pure generation step's request indices, slots and indptrs."""
    num_tokens, num_seqs = md.num_tokens, md.num_seqs
    if not (
        md.kv_cache_manager is not None
        and num_tokens > 0
        and md.num_generations > 0
        and md.num_contexts == 0
        and not md.use_fp8_ds_mla
        and not md.enable_helix
    ):
        return False
    index_head_dim = md.kv_cache_manager.index_head_dim
    fused_dsa_decode_metadata(
        md.seq_lens_cuda[:num_seqs],
        md.kv_lens_cuda[:num_seqs],
        md.indexer_k_cache_block_offsets[:num_seqs],
        md.req_idx_per_token[:num_tokens],
        md.slot_mapping_fp8[:num_tokens],
        md.slot_mapping_scale[:num_tokens],
        md.gen_kv_indptr[: num_seqs + 1],
        md.gen_cached_token_indptr[: num_seqs + 1],
        num_tokens=num_tokens,
        max_query_len=1 + md.max_draft_tokens,
        tokens_per_block=md._tokens_per_block,
        index_head_dim=index_head_dim,
        quant_block_size=md.indexer_quant_block_size,
        data_bytes_per_token=index_head_dim,
    )
    return True


def _eager_decode_metadata(md) -> None:
    """Request index and index-key slot of every row (a step with context rows, or Helix)."""
    num_tokens, num_seqs = md.num_tokens, md.num_seqs
    seq_lens = md.seq_lens_cuda[:num_seqs]
    md.req_idx_per_token[:num_tokens] = build_req_idx_per_token(seq_lens, num_tokens).to(
        md.req_idx_per_token.dtype
    )
    if md.kv_cache_manager is None:
        return
    # Runtime cached lengths after the overlap / speculative-decode correction.
    start_positions = md.kv_lens_cuda[:num_seqs] - seq_lens
    req_indices = md.req_idx_per_token[:num_tokens].to(dtype=torch.int64)
    seq_starts = torch.cumsum(seq_lens, dim=0, dtype=torch.int64) - seq_lens.to(torch.int64)
    token_offsets = (
        torch.arange(num_tokens, device=seq_lens.device, dtype=torch.int64)
        - seq_starts[req_indices]
    )
    global_positions = start_positions[req_indices] + token_offsets
    if md.use_fp8_ds_mla:
        md.token_positions_cuda[:num_tokens] = global_positions.to(torch.int32)
    index_head_dim = md.kv_cache_manager.index_head_dim
    fp8_indices, scale_indices = compute_slot_mappings(
        global_positions,
        md.indexer_k_cache_block_offsets,
        req_indices,
        index_head_dim,
        md._tokens_per_block,
        md.indexer_quant_block_size,
        data_bytes_per_token=index_head_dim,
    )
    md.slot_mapping_fp8[:num_tokens] = fp8_indices
    md.slot_mapping_scale[:num_tokens] = scale_indices


def prepare_generation_placeholder_rows(md: "RubinDSAMetadata") -> None:
    """Under forced acceptance, upload which generation rows verify placeholder drafts.

    The family cache manager decides it per request from the scheduled requests
    (``rubin_kda_dsa_cache_manager.placeholder_draft_requests``); the rows' flags go to
    ``md.generation_placeholder_rows`` in batch order, written here, eagerly, every step, before a captured
    graph replays. The worker gives those rows back their own accepted-token counts after the forcing
    (``ops/spec/rubin_dflash2_worker.py``). Rows past the scheduled requests (padding) are False.
    """
    manager = md.kv_cache_manager
    if getattr(manager, "first_step_pending", None) is None:
        return
    if md.generation_placeholder_rows is None:
        md.generation_placeholder_rows = torch.zeros(
            md.max_num_requests, dtype=torch.bool, device=md.kv_lens_cuda.device
        )
    if not md.request_ids:
        return
    placeholders = manager.placeholder_requests
    rows = [
        request_id in placeholders for request_id in md.request_ids[md.num_contexts : md.num_seqs]
    ]
    if rows:
        md.generation_placeholder_rows[: len(rows)].copy_(
            maybe_pin_memory(torch.tensor(rows, dtype=torch.bool)), non_blocking=True
        )


class RubinDSAMetadata(HelixDSAMetadataMixin, DSAtrtllmAttentionMetadata):
    """The DSA metadata of the dep16 and tep8 targets (see the module docstring)."""

    # Whether the class serves a Helix CP layout; the mapping must agree.
    HELIX: ClassVar[bool] = False
    # The methods this class defines; every other method is the Helix mixin's or the engine class's.
    OVERRIDES: ClassVar[Tuple[str, ...]] = (
        "__post_init__",
        "prepare",
        "prepare_for_spec_decode",
        "on_update_kv_lens",
        "update_for_spec_dec",
        "warmup_selfsampling_topk",
    )

    def __post_init__(self):
        super().__post_init__()
        if self.enable_helix != self.HELIX and not self.is_cross:
            raise ValueError(
                f"{type(self).__name__} serves {'a Helix' if self.HELIX else 'no Helix'} CP layout; "
                f"the mapping has Helix {'off' if self.HELIX else 'on'}"
            )
        self.use_self_sampling_topk = self_sampling_gvr(
            self.sparse_metadata_params, self._indexer_compress_ratio, get_sm_version()
        )
        # Helix key-sharded Top-K: CP ranks, verify width of the step's rows (0 = none).
        self.key_shard_world = 1
        self.key_shard_next_n = 0
        # Under forced acceptance: whether each generation row verifies placeholder drafts, in batch order
        # (see prepare_generation_placeholder_rows); allocated by the first step.
        self.generation_placeholder_rows: Optional[torch.Tensor] = None

    def prepare(self):
        """The engine's step preparation with the family's indexer plan."""
        self._validate_ragged_verification()
        TrtllmAttentionMetadata.prepare(self)
        self._invalidate_pool_view_cache()
        assert self.kv_cache_params.use_cache is True, "DSA requires use_cache to be True"
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq, dtype=torch.int, device="cpu"
        )
        prepare_generation_placeholder_rows(self)
        if self.enable_helix:
            kv_lens = self._helix_local_kv_lens(cached_token_lens)
        else:
            kv_lens = cached_token_lens + self.seq_lens_kv
        kv_lens = self._prepare_device_window_kv_lens(kv_lens)
        self.prepare_for_mla_rope_append(cached_token_lens, kv_lens)
        self.prepare_for_skip_indexer(kv_lens)
        self.prepare_for_indices_conversion()
        self.prepare_for_indexer_k_cache()
        self.prepare_for_spec_decode(kv_lens)
        _prepare_indexer(self)
        if self.enable_helix:
            self._prepare_helix_sparse_rows(kv_lens)

    def prepare_for_spec_decode(self, kv_lens: torch.Tensor):
        """Per-token decode rows where a verify window needs them; never a DSL split.

        Helix verify bounds and ragged verification score one row per token (the expanded
        buffers). A uniform window is one DSL atom on the targets, so the indexer reads the
        request-major schedule and no split buffers are planned.
        """
        use_dsl = self.use_cute_dsl_paged_mqa_logits
        if not self.is_ragged_verify:
            self._ragged_num_rows = 0
            self._attn_num_rows = 0
        else:
            use_dsl = False
            assert not self.is_spec_decoding_enabled, (
                "ragged verification requires dense spec-decoding metadata to be disabled"
            )
        helix_spec = self.enable_helix and self._helix_spec_tokens_valid
        if helix_spec:
            use_dsl = False
        self.use_expanded_buffers_for_mtp = not use_dsl and (helix_spec or self.is_ragged_verify)
        if self.use_expanded_buffers_for_mtp:
            nc, ns, ng = self.num_contexts, self.num_seqs, self.num_generations
            if self.is_ragged_verify:
                self.host_gen_token_repeats[:ng].copy_(
                    torch.tensor(self.gen_token_repeat_list(), dtype=torch.int64)
                )
                self.gen_token_repeats_cuda[:ng].copy_(
                    self.host_gen_token_repeats[:ng], non_blocking=True
                )
            gen_kv_lens = self.get_indexer_kv_lens(kv_lens[nc:ns])
            expanded, rows = self.expand_per_gen_token(gen_kv_lens)
            self.kv_lens_expanded_host[:rows].copy_(expanded)
            self.kv_lens_expanded_cuda[:rows].copy_(
                self.kv_lens_expanded_host[:rows], non_blocking=True
            )
            if self.kv_cache_manager is not None and ng > 0:
                width = self.host_indexer_k_cache_block_offsets.shape[1]
                blocks, _ = self.expand_per_gen_token(
                    self.host_indexer_k_cache_block_offsets[nc:ns, :width], dim=0
                )
                self.host_block_table_expanded[:rows, :width].copy_(blocks)
                self.block_table_expanded[:rows].copy_(
                    self.host_block_table_expanded[:rows], non_blocking=True
                )
                self.block_table_expanded.clamp_(min=0)
            if self.is_ragged_verify:
                self._prepare_ragged_row_kv_lens(kv_lens)
                self._prepare_token_major_gen_rows(kv_lens)
        if use_dsl and self.max_draft_tokens >= 1 and self.num_generations > 0:
            next_n = self.gen_token_stride
            if not dsl_single_atom(next_n):
                raise ValueError(
                    f"a {next_n}-row verify window does not fit one DSL logits atom "
                    f"({next_n} x {_INDEX_N_HEADS} index heads > {_DSL_MAX_ATOM_N}); the K3.1 "
                    "Rubin targets verify at most 7 draft tokens"
                )
        self.expand_for_dsl = False
        self.dsl_expand_factor = 1
        self.dsl_atom = self.gen_token_stride
        self.dsl_kernel_atoms = 1

    def on_update_kv_lens(self) -> None:
        """Rebuild the decode metadata from the current KV lengths (graph-safe: device work only).

        Called after every correction of ``kv_lens_cuda`` (overlap scheduler, speculative decode)
        and replayed by a captured decode graph. A pure generation step takes one fused launch for
        the request indices, index-key slots and generation indptrs.
        """
        TrtllmAttentionMetadata.on_update_kv_lens(self)
        self._invalidate_pool_view_cache()
        fused = _fused_decode_metadata(self)
        if not fused and self.num_tokens > 0:
            _eager_decode_metadata(self)
        if self.enable_ragged_verification:
            self.refresh_ragged_row_kv_lens()
            self.refresh_token_major_gen_rows()
        if self.num_generations > 0:
            nc, ns = self.num_contexts, self.num_seqs
            gen_kv_lens = self.kv_lens_cuda[nc:ns]
            if not fused:
                torch.cumsum(
                    gen_kv_lens, 0, dtype=torch.int64, out=self.gen_kv_indptr[1 : ns - nc + 1]
                )
                torch.cumsum(
                    gen_kv_lens - self.seq_lens_cuda[nc:ns],
                    0,
                    dtype=torch.int64,
                    out=self.gen_cached_token_indptr[1 : ns - nc + 1],
                )
            gen_indexer_kv_lens = self.get_indexer_kv_lens(gen_kv_lens)
            if self.use_expanded_buffers_for_mtp:
                expanded, rows = self.expand_per_gen_token(gen_indexer_kv_lens)
                self.kv_lens_expanded_cuda[:rows].copy_(expanded)
            _decode_schedules(self, gen_indexer_kv_lens)
        self._compute_kv_lens_row_reorder()
        self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)

    def update_for_spec_dec(self, refresh_device_metadata: bool = True):
        """``refresh_device_metadata=False``: the draft forward reads none of the device-side
        indexer or sparse-MLA metadata, so only the cached state is invalidated; the next step's
        ``on_update_kv_lens`` rebuilds all of it."""
        TrtllmAttentionMetadata.update_for_spec_dec(self)
        self.max_ctx_kv_len = 0
        self.num_ctx_cached_tokens = 0
        self.max_gen_seq_len = 1
        if refresh_device_metadata:
            self.on_update_kv_lens()
        else:
            TrtllmAttentionMetadata.on_update_kv_lens(self)
            self._invalidate_pool_view_cache()

    def warmup_selfsampling_topk(
        self, next_n: int, batch_sizes: Optional[List[int]] = None
    ) -> None:
        """Pre-compile the targets' Top-K, the deterministic self-sampling GVR, for the decode
        rows the step can have and the prefill (the engine's warmup, on sm_107 too)."""
        if not self.use_self_sampling_topk or self.kv_cache_manager is None:
            return
        next_n = int(next_n)
        # Small eager row counts and every CUDA-graph batch's rows.
        rows = set(range(next_n, 33, next_n)) or {next_n}
        rows.update(int(bs) * next_n for bs in batch_sizes or ())
        max_len = int(self.get_indexer_max_seq_len())
        if self.use_cute_dsl_paged_mqa_logits:
            row_stride = (max_len + 255) // 256 * 256  # the DSL logits arena's row stride
        else:
            # A row stride of whole float4s, else the decode takes the radix Top-K.
            row_stride = None if max_len % 4 else max_len
        rubin_gvr_topk_warmup(
            int(self.sparse_mla_topk),
            max_len,
            int(self._indexer_compress_ratio or 1),
            next_n,
            tuple(sorted(rows)),
            row_stride,
        )


def _key_shard_config(md) -> Tuple[int, int, int]:
    """``(CP ranks, rows, local width)`` of the key-sharded Top-K, or ``(1, 0, 0)`` without it.

    The CP ranks score their own pages (page g on rank g % cp) and merge their local Top-K. That
    needs tp 1, a power-of-two cp, no attention DP, FP8 index keys, the DSL paged logits, the
    self-sampling GVR, uncompressed index keys and its page size. ``rows`` is the largest decode
    step (max batch x (max draft + 1)), which the merge workspace is built for.
    """
    mapping = md.mapping
    cp = mapping.cp_size
    eligible = (
        cp > 1
        and cp & (cp - 1) == 0
        and mapping.tp_size == 1
        and not mapping.enable_attention_dp
        and getattr(md.kv_cache_manager, "indexer_k_dtype", "fp8") == "fp8"
        and md.use_cute_dsl_paged_mqa_logits
        and md.use_self_sampling_topk
        and md._indexer_compress_ratio == 1
        and md._tokens_per_block == _KEY_SHARD_TOKENS_PER_BLOCK
    )
    if not eligible:
        logger.warning_once(
            "Helix DSA indexer: no key-sharded Top-K (it needs tp 1, a power-of-two cp, no "
            "attention DP, FP8 index keys, the DSL logits, the self-sampling GVR, uncompressed "
            f"keys and {_KEY_SHARD_TOKENS_PER_BLOCK}-token pages); every step merges over CP",
            key="rubin_dsa_key_shard_ineligible",
        )
        return 1, 0, 0
    manager = md.kv_cache_manager
    server_draft = max(
        int(getattr(manager, "max_draft_len", 0) or 0),
        int(getattr(manager, "max_total_draft_tokens", 0) or 0),
    )
    rows = md.max_num_sequences * (1 + max(md._draft_sizing_cap, server_draft))
    return cp, rows, manager.max_blocks_per_seq * _KEY_SHARD_TOKENS_PER_BLOCK


def _key_shard_next_n(md) -> int:
    """The verify width the key-sharded Top-K serves this step, 0 when the CP merge serves it.

    A ragged step, rows that do not divide by the requests, and a window of more than one row
    without Helix verify bounds take the merge (eager only: a CUDA graph cannot change path).
    """
    num_gen = md.num_generations
    rows = md.num_tokens - md.num_ctx_tokens
    if md.key_shard_world <= 1 or num_gen <= 0:
        return 0
    next_n = 0
    if not md.is_ragged_verify and rows % num_gen == 0:
        next_n = rows // num_gen
        if rows > md.key_shard_rows_max:
            raise RuntimeError(
                f"DSA key-sharded Top-K: {rows} decode rows exceed the merge workspace's "
                f"{md.key_shard_rows_max} (max batch x (max draft + 1))"
            )
        if next_n != 1 and md.helix_spec_row_kv_lens(rows) is None:
            next_n = 0
    if next_n == 0 and md.is_cuda_graph:
        raise RuntimeError(
            "a CUDA-graph Helix step must take the key-sharded Top-K: "
            f"{num_gen} requests, {rows} rows, ragged {md.is_ragged_verify}, "
            f"verify bounds {md.helix_spec_row_kv_lens(rows) is not None}"
        )
    return next_n


def _helix_fused_applies(md) -> bool:
    """Whether a Helix step's decode metadata is one launch: a generation step of uniform verify
    blocks with Helix bounds and per-token rows."""
    return (
        md.kv_cache_manager is not None
        and md._helix_spec_tokens_valid
        and md.num_contexts == 0
        and md.num_ctx_tokens == 0
        and md.num_generations > 0
        and md.num_tokens == md.num_generations * md.gen_token_stride
        and md.use_expanded_buffers_for_mtp
        and not md.enable_ragged_verification
        and not md.is_ragged_verify
        and not md.use_fp8_ds_mla
        and md._indexer_compress_ratio <= 1
    )


def _helix_fused_decode_metadata(md) -> None:
    """The decode metadata of a Helix verify step: one launch, then the two schedules."""
    num_seqs, num_tokens = md.num_seqs, md.num_tokens
    index_head_dim = md.kv_cache_manager.index_head_dim
    gen_kv_lens = md.kv_lens_cuda[:num_seqs]
    rubin_fused_dsa_helix_decode_metadata(
        md.seq_lens_cuda[:num_seqs],
        gen_kv_lens,
        md.indexer_k_cache_block_offsets,
        md.helix_local_slots,
        md.helix_kv_bounds,
        md.req_idx_per_token,
        md.slot_mapping_fp8,
        md.slot_mapping_scale,
        md.gen_kv_indptr,
        md.gen_cached_token_indptr,
        md.kv_lens_cuda_2d,
        md.kv_lens_expanded_cuda,
        num_tokens=num_tokens,
        width=min(md.gen_token_stride, md.kv_lens_cuda_2d.shape[1]),
        tokens_per_block=md._tokens_per_block,
        index_head_dim=index_head_dim,
        quant_block_size=md.indexer_quant_block_size,
        data_bytes_per_token=index_head_dim,
    )
    gen_indexer_kv_lens = md.get_indexer_kv_lens(gen_kv_lens)
    md.gen_indexer_kv_lens_cuda_runtime = gen_indexer_kv_lens
    get_paged_mqa_logits_metadata(
        gen_indexer_kv_lens.view(-1, 1), SCHEDULE_BLOCK_KV, md.num_sms, md.scheduler_metadata_buffer
    )
    get_paged_mqa_logits_metadata(
        md.kv_lens_expanded_cuda[:num_tokens].view(-1, 1),
        SCHEDULE_BLOCK_KV,
        md.num_sms,
        md.scheduler_metadata_buffer_expanded,
    )


def _helix_gen_ownership(md) -> None:
    """Restrict the generation rows' index-key writes and scoring to this CP rank's tokens.

    A generation token's KV lives on the rank that owns its page: its row gets its rank-local
    slot, or -1 (the scatter skips it) where another rank owns it. Verify rows score against
    their rank-local causal bound (at least one key, the indexer drops empty rows).
    """
    num_ctx_tokens = md.num_ctx_tokens
    num_gen_tokens = md.num_tokens - num_ctx_tokens
    if num_gen_tokens <= 0:
        return
    if md._helix_spec_tokens_valid:
        local = md.helix_local_slots[:num_gen_tokens].to(torch.int64)
        owned = local >= 0
        req_indices = md.req_idx_per_token[num_ctx_tokens : md.num_tokens].to(torch.int64)
    else:
        # One token per generation row; the active rank appends it locally.
        local = md.kv_lens_cuda[md.num_contexts : md.num_seqs].to(torch.int64) - 1
        owned = ~md.helix_is_inactive_rank[: md.num_generations]
        req_indices = torch.arange(
            md.num_contexts, md.num_seqs, dtype=torch.int64, device=local.device
        )
    index_head_dim = md.kv_cache_manager.index_head_dim
    fp8_indices, scale_indices = compute_slot_mappings(
        local.clamp(min=0),
        md.indexer_k_cache_block_offsets,
        req_indices,
        index_head_dim,
        md._tokens_per_block,
        md.indexer_quant_block_size,
        data_bytes_per_token=index_head_dim,
    )
    rows = slice(num_ctx_tokens, md.num_tokens)
    md.slot_mapping_fp8[rows] = torch.where(owned, fp8_indices, -1)
    md.slot_mapping_scale[rows] = torch.where(owned, scale_indices, -1)
    if md._helix_spec_tokens_valid and md.use_expanded_buffers_for_mtp:
        md.kv_lens_expanded_cuda[:num_gen_tokens].copy_(
            md.helix_kv_bounds[:num_gen_tokens].clamp(min=1)
        )
        get_paged_mqa_logits_metadata(
            md.kv_lens_expanded_cuda[:num_gen_tokens].view(-1, 1),
            SCHEDULE_BLOCK_KV,
            md.num_sms,
            md.scheduler_metadata_buffer_expanded,
        )


class RubinDSAHelixMetadata(RubinDSAMetadata):
    """The DSA metadata of helix8 (Helix CP generation; see the module docstring)."""

    HELIX: ClassVar[bool] = True
    # One KV-length refresh per input preparation, after the last correction, suffices:
    # on_update_kv_lens rebuilds every output it writes from the current inputs.
    kv_lens_refresh_once: ClassVar[bool] = True
    OVERRIDES: ClassVar[Tuple[str, ...]] = (
        "__post_init__",
        "on_update_kv_lens",
        "helix_key_shard_inputs",
        "helix_sparse_row_lens_out",
    )

    def __post_init__(self):
        super().__post_init__()
        self.key_shard_world, self.key_shard_rows_max, self.key_shard_max_local_len = (
            _key_shard_config(self) if not self.is_cross else (1, 0, 0)
        )

    def on_update_kv_lens(self) -> None:
        """The base rebuild plus the Helix ownership, or one launch on a verify step with bounds;
        then the step's key-sharded Top-K width."""
        if _helix_fused_applies(self):
            TrtllmAttentionMetadata.on_update_kv_lens(self)
            self._invalidate_pool_view_cache()
            _helix_fused_decode_metadata(self)
            self._compute_kv_lens_row_reorder()
            self.prepare_dense_topk_indices(self.kv_lens_cuda, device=True)
        else:
            super().on_update_kv_lens()
            if self.kv_cache_manager is not None and self.num_generations > 0:
                _helix_gen_ownership(self)
        self.key_shard_next_n = _key_shard_next_n(self)

    def helix_key_shard_inputs(self, num_gen_tokens: int):
        """``(page table, local lengths, schedule, per-row local bounds)`` of the key-sharded
        Top-K: this rank's own pages, lengths and request-major schedule."""
        num_gen = self.num_generations
        ctx_lens = self.gen_indexer_kv_lens_cuda_runtime
        row_lens = self.helix_spec_row_kv_lens(num_gen_tokens)
        return (
            self.indexer_k_cache_block_offsets[self.num_contexts : self.num_seqs],
            ctx_lens,
            self.scheduler_metadata_buffer,
            ctx_lens[:num_gen] if row_lens is None else row_lens,
        )

    def helix_sparse_row_lens_out(self, num_gen_rows: int) -> Optional[torch.Tensor]:
        """The generation rows' token-sparse MLA row lengths, for the Top-K merge to write its
        kept counts into; None without Helix sparse rows this step."""
        if self._helix_num_rows <= 0:
            return None
        num_contexts = self.num_contexts
        return self.helix_row_kv_lens_cuda[num_contexts : num_contexts + num_gen_rows]


__all__ = [
    "RubinDSAHelixMetadata",
    "RubinDSAMetadata",
    "RubinDsaPrefillChunk",
    "dsl_single_atom",
    "packed_logits_elem_budget",
    "query_tiles",
    "self_sampling_gvr",
]
