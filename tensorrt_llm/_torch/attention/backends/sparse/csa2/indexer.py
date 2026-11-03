# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 candidate adaptation of the shared DSA/DeepSeek-V4 indexer workflow."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm._utils import get_sm_version, maybe_pin_memory

try:
    from tensorrt_llm.deep_gemm import (
        fp8_fp4_paged_sparse_mqa_logits,
        fp8_fp4_sparse_mqa_logits,
        get_mqa_logits_metadata,
        get_paged_sparse_mqa_logits_metadata,
        get_sparse_mqa_logits_metadata,
    )

    _HAS_SPARSE_MQA_LOGITS = True
except ImportError:  # The bundled DeepGEMM predates the sparse indexer kernels
    _HAS_SPARSE_MQA_LOGITS = False

from ..dsa.indexer import _INDEXER_MQA_LOGITS_ELEM_BUDGET, Indexer, _split_prefill_queries
from ..dsa.params import DSAParams
from ..params import use_self_sampling_gvr
from .decode_select import (
    fused_consumer_topk_supported,
    fused_decode_select_,
    fused_decode_select_supported,
    select_mapped_consumer_topk_,
)
from .kernel import (
    PREFIX_DESC_WIDTH,
    candidate_block_max_,
    dsl_available,
    expand_candidates_,
    finalize_selection_,
    gather_candidate_scores,
    gather_prefix_keys_,
    map_selection_,
    mask_logits_,
    mask_packed_logits_,
)
from .metadata import CSA2IndexerGroup, CSA2TrtllmMetadata, csa2_side_stream
from .params import CSA2ForwardState, CSA2Layout, CSA2Mode, CSA2Params
from .quantization import (
    INDEX_DATA_BYTES,
    INDEX_PAGE_ROWS,
    INDEX_SCALE_BYTES,
    pack_index_queries_split,
)

# Prefill scores narrower than this keep the exact radix TopK.
_SELF_SAMPLING_PREFILL_MIN_COLUMNS = 8192
# Batched candidate prefill gathers each request's prefix at a key offset aligned to this many keys
# (a multiple of every sparse block), with at most _BATCHED_CANDIDATE_KEYS keys per batch.
_BATCHED_CANDIDATE_ALIGN = 128
_BATCHED_CANDIDATE_KEYS = 4 << 20  # 4M
# Candidate publications with at most this many query rows use the cluster radix TopK.
_CANDIDATE_CLUSTER_TOPK_MAX_ROWS = 64
# The self-sampling GVR decode engine runs one CTA per row. On the candidate source, whose rows
# are twice as long, larger decode batches select with the exact radix Top-K.
_DECODE_GVR_SOURCE_MAX_ROWS = 256


def _sparse_block(candidate_block_size: int) -> int:
    """DeepGEMM sparse MQA-logits block (16 or 8 positions) dividing the candidate blocks, or 0."""
    return next((block for block in (16, 8) if candidate_block_size % block == 0), 0)


def candidate_consumer_requirement(
    heads: int, head_dim: int, candidate_block_size: int
) -> str | None:
    """What this GPU, build or index geometry lacks for candidate consumers, or None.

    Candidate consumers score the published candidate blocks only with DeepGEMM's sparse MQA logits.
    """
    sm = get_sm_version() if torch.cuda.is_available() else 0
    if not 100 <= sm < 110:
        return f"an SM100-family GPU (this is SM{sm})"
    if not _HAS_SPARSE_MQA_LOGITS:
        return "a DeepGEMM build with the sparse MQA-logits kernels"
    if (heads, head_dim) != (32, 128):
        return f"32 index heads of 128 dims (the checkpoint has {heads} of {head_dim})"
    if not _sparse_block(candidate_block_size):
        return f"a candidate_block_size that is a multiple of 8 (the checkpoint has {candidate_block_size})"
    return None


class CSA2Indexer(Indexer):
    """Prepare owner-cache candidates and publish CSA2 logical selections.

    This subclass owns prefix gathering, candidate workspace limits and routing
    publication. It reuses the base indexer's MQA dispatch, TP query partition
    calculation and TopK module. Cache writes and layer ordering belong to the
    backend.
    """

    def __init__(
        self,
        layout: CSA2Layout,
        layer_idx: int,
        heads: int,
        head_dim: int,
        options: CSA2Params | None = None,
    ) -> None:
        layer = layout.layer(layer_idx)
        if layer.mode not in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            raise ValueError("Only CSA2 Full/Reindex layers construct an indexer")
        super().__init__(
            quant_config=None,
            pos_embd_params=None,
            mla_params=None,
            skip_create_weights_in_init=True,
            sparse_params=DSAParams(
                index_n_heads=heads,
                index_head_dim=head_dim,
                index_topk=layout.index_topk,
                indexer_k_dtype="fp4",
            ),
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            projection_free=True,
        )
        self.layout = layout
        self.options = options or CSA2Params(layout=layout)
        sm = get_sm_version() if torch.cuda.is_available() else 0
        compatible = IS_CUTLASS_DSL_AVAILABLE and 100 <= sm < 110
        exact = (
            TopKImplementation.CUTE_DSL_RADIX
            if compatible and self.options.use_cute_dsl_topk
            else TopKImplementation.CUDA_RADIX
        )
        if not torch.cuda.is_available():
            exact = TopKImplementation.TORCH
        prefill, decode = exact, exact
        self_sampling_kwargs = dict(
            enable_heuristic_topk=True,
            use_self_sampling_topk=self.options.use_self_sampling_topk,
            index_topk=self.index_topk,
            compress_ratio=1,
            is_cute_dsl_available=IS_CUTLASS_DSL_AVAILABLE,
            sm_version=sm,
        )
        prefill_self_sampling = use_self_sampling_gvr(**self_sampling_kwargs, is_prefill=True)
        decode_self_sampling = use_self_sampling_gvr(**self_sampling_kwargs, is_prefill=False)
        if prefill_self_sampling:
            # The self-sampling engine is exact, so prefill uses it without heuristic decode.
            prefill = TopKImplementation.CUTE_DSL_GVR
        if compatible and self.options.enable_heuristic_topk:
            if decode_self_sampling:
                # Candidate consumers rank candidate-ordered columns, which are
                # not key positions; their decode TopK stays exact radix.
                if not layer.consumes_candidates:
                    decode = TopKImplementation.CUTE_DSL_GVR
            elif (
                not self.options.use_self_sampling_topk
                and layer.mode == CSA2Mode.FULL
                and head_dim == 128
                and self.index_topk in (512, 1024, 2048)
            ):
                decode = TopKImplementation.CUTE_DSL_GVR
        self.top_k.prefill_implementation = prefill
        self.top_k.decode_implementation = decode
        self.top_k.gvr_self_sampling = self.options.use_self_sampling_topk
        # CSA2's decode Top-K selectors opt into the CuTe DSL decode kernels' wide-row launches.
        self.top_k.wide_fp32_rows = True
        self.top_k.wide_cta_rows = True
        self.top_k.split_long_rows = True
        self._exact_top_k = exact
        self.narrow_prefill_top_k = (
            TopK(self.index_topk, prefill_implementation=exact) if prefill != exact else None
        )
        self.candidate_top_k: TopK | None = None
        # Candidate consumers score only the published candidate blocks with
        # DeepGEMM's sparse MQA logits.
        self.sparse_block = _sparse_block(layout.candidate_block_size)
        self.use_sparse_candidates = layer.consumes_candidates and (
            candidate_consumer_requirement(heads, head_dim, layout.candidate_block_size) is None
        )

    def _configure_candidate_topk(self, max_positions: int) -> None:
        if max_positions < 0:
            raise ValueError("CSA2 candidate position bound must be nonnegative")
        # Warmup and draft metadata may use different admitted KV bounds on
        # the same layer. Keep the selector at the immutable model budget;
        # only its caller-owned publication tensor follows the current bound.
        count = self.layout.candidate_topk_blocks
        if self.candidate_top_k is None:
            torch_reference = self.top_k.prefill_implementation == TopKImplementation.TORCH
            self.candidate_top_k = TopK(
                count,
                prefill_implementation=(
                    TopKImplementation.TORCH if torch_reference else TopKImplementation.CUDA_RADIX
                ),
                # With the CuTe DSL Top-K, up to _CANDIDATE_CLUSTER_TOPK_MAX_ROWS rows select with
                # its multi-CTA (cluster) radix kernel, which splits each wide row over several CTAs.
                decode_implementation=self._exact_top_k,
                wide_fp32_rows=True,
                split_long_rows=True,
            )
        elif self.candidate_top_k.top_k != count:
            raise ValueError("CSA2 candidate block budget must remain fixed for an indexer")

    def _constant_rows(self, count: int, value: int, device: torch.device) -> torch.Tensor:
        """Int32 ``[count]`` tensor filled with ``value``.

        A captured graph keeps addressing these rows on every replay, so rows
        created while capturing are retained for the indexer's lifetime (there
        are few graph geometries). Eager calls see many query counts and get a
        transient: an evictable cache entry could be freed and reused while a
        graph that captured it still reads it. Retained rows live in the graphs'
        shared memory pool, where earlier graphs' temporaries can overwrite them,
        so every call (eager or captured) refills them.
        """
        cache = self.__dict__.setdefault("_constant_row_cache", {})
        key = (str(device), count, value)
        rows = cache.get(key)
        if rows is None:
            rows = torch.empty((count,), dtype=torch.int32, device=device)
            if rows.is_cuda and torch.cuda.is_current_stream_capturing():
                cache[key] = rows
        return rows.fill_(value)

    def select_prepared_scores(
        self,
        scores: torch.Tensor,
        output_indices: torch.Tensor,
        row_starts: Optional[torch.Tensor],
        row_ends: Optional[torch.Tensor],
        *,
        is_prefill: bool = True,
        radix_aux_indices: Optional[torch.Tensor] = None,
        radix_aux_logits: Optional[torch.Tensor] = None,
        candidate: bool = False,
        gvr_prior_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Invoke the configured main TopK or the candidate-source block TopK.

        ``row_starts``/``row_ends`` default to the full score width; decode TopK
        only consumes the row ends.
        """
        if scores.shape[0] == 0 or scores.shape[1] == 0:
            return output_indices.fill_(-1)
        count, width = scores.shape
        if row_ends is None:
            row_ends = self._constant_rows(count, width, scores.device)
        if row_starts is None and is_prefill:
            row_starts = self._constant_rows(count, 0, scores.device)
        selector = self.candidate_top_k if candidate else self.top_k
        if (
            is_prefill
            and not candidate
            and self.narrow_prefill_top_k is not None
            and width < _SELF_SAMPLING_PREFILL_MIN_COLUMNS
        ):
            selector = self.narrow_prefill_top_k
        if selector is None or output_indices.shape[1] != selector.top_k:
            raise ValueError("CSA2 TopK output must match its configured selection domain")
        if (
            not is_prefill
            and selector.decode_implementation == TopKImplementation.CUTE_DSL_GVR
            and selector.gvr_self_sampling
            and count > _DECODE_GVR_SOURCE_MAX_ROWS
            and self.layout.layer(self.layer_idx).candidate_source == self.layer_idx
        ):
            selector = self._decode_radix_selector(selector)
        implementation = (
            selector.prefill_implementation if is_prefill else selector.decode_implementation
        )
        if implementation in (TopKImplementation.CUTE_DSL_RADIX, TopKImplementation.CUTE_DSL_GVR):
            # Exact DSL uses 32-byte vector copies; pad storage without extending
            # row bounds. Self-sampling also benefits from this native layout.
            vector = 32 // scores.element_size()
            if (
                scores.stride(1) != 1
                or scores.stride(0) % vector
                or scores.data_ptr() % 32
                or scores.shape[1] % vector
            ):
                scores = F.pad(
                    scores.contiguous(), (0, -scores.shape[1] % vector), value=-torch.inf
                )
        if not is_prefill:
            return selector(
                scores,
                output_indices,
                is_prefill=False,
                sequence_lengths=(
                    row_ends if selector.compress_ratio == 1 else row_ends * selector.compress_ratio
                ),
                scan_lengths=row_ends,
                next_n=1,
                max_seq_len=scores.shape[1],
                radix_aux_indices=radix_aux_indices,
                radix_aux_logits=radix_aux_logits,
                gvr_ext_kwargs={"gvr_prior_indices": gvr_prior_indices}
                if selector.needs_gvr_prior
                else None,
            )
        return selector(
            scores, output_indices, is_prefill=True, row_starts=row_starts, row_ends=row_ends
        )

    def _decode_radix_selector(self, selector: TopK) -> TopK:
        """Exact radix decode Top-K standing in for the self-sampling GVR engine on the candidate
        source's decode batches above ``_DECODE_GVR_SOURCE_MAX_ROWS`` rows."""
        radix = self.__dict__.get("_decode_radix_top_k")
        if radix is None or radix.top_k != selector.top_k:
            radix = TopK(
                selector.top_k,
                prefill_implementation=self._exact_top_k,
                decode_implementation=self._exact_top_k,
                compress_ratio=selector.compress_ratio,
                wide_fp32_rows=True,
                split_long_rows=True,
            )
            # A plain attribute: the stand-in holds no state worth registering.
            self.__dict__["_decode_radix_top_k"] = radix
        return radix

    def _select_all_visible(
        self, positions: torch.Tensor, visible: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """Select every valid logical candidate once its cardinality is bounded by K."""
        valid = (positions >= 0) & (positions < visible[:, None])
        sentinel = torch.iinfo(torch.int64).max
        selected = torch.where(valid, positions.long(), sentinel).sort(-1).values
        selected = torch.where(selected == sentinel, -1, selected).to(torch.int32)
        selected = selected[:, : self.index_topk]
        output.copy_(F.pad(selected, (0, self.index_topk - selected.shape[1]), value=-1))
        return output

    def _gather_prefix(
        self,
        metadata: CSA2TrtllmMetadata,
        kv_source: int,
        descriptors: torch.Tensor,
        keys: int,
        rows: int,
        visible: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Concatenate the prepared pieces' compressed prefixes in one launch.

        Returns ``keys`` gathered keys, their scales and validity and the
        ``[5, rows]`` row bounds of ``gather_prefix_keys_``.
        """
        device = visible.device
        data = torch.empty((keys, INDEX_DATA_BYTES), dtype=torch.int8, device=device)
        scales = torch.empty(keys, dtype=torch.int32, device=device)
        valid = torch.empty(keys, dtype=torch.bool, device=device)
        bounds = torch.empty((5, rows), dtype=torch.int32, device=device)
        gather_prefix_keys_(
            metadata.kv_cache_manager.get_index_pages(kv_source),
            metadata.csa2_global_page_tables[kv_source],
            visible,
            descriptors,
            metadata.csa2_global_page_sizes[kv_source],
            metadata.csa2_global_max_positions[kv_source],
            data,
            scales,
            valid,
            bounds,
        )
        return data, scales, valid, bounds

    def _packed_group_schedule(
        self,
        metadata: CSA2TrtllmMetadata,
        kv_source: int,
        group: CSA2IndexerGroup,
        visible: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        heads: int,
    ) -> Optional[torch.Tensor]:
        """DeepGEMM's per-SM split of a multi-request packed group's key ranges.

        Without it each SM scores fixed query blocks over their whole key range, which leaves
        most SMs idle when a group mixes short and long prefixes; a one-request group is
        balanced already and keeps the grid-stride order. Row bounds follow from the group and
        the visible lengths, so eager layers that share both reuse one schedule; captured graphs
        build it in each call.
        """
        if (
            group.descriptors.shape[0] < 2
            or not _HAS_SPARSE_MQA_LOGITS
            or self.head_dim != 128
            or heads not in (8, 16, 32, 64)
            or get_sm_version() not in (100, 103)
        ):
            return None
        if torch.cuda.is_current_stream_capturing():
            return get_mqa_logits_metadata(row_starts, row_ends, group.keys, heads)
        groups = metadata.csa2_indexer_groups
        cached = metadata.__dict__.get("_csa2_packed_group_schedules")
        if cached is None or cached[0] is not groups:
            cached = metadata._csa2_packed_group_schedules = (groups, {})
        key = (kv_source, group.lo, group.hi, visible.data_ptr(), visible.shape[0])
        schedule = cached[1].get(key)
        if schedule is None:
            schedule = cached[1][key] = get_mqa_logits_metadata(
                row_starts, row_ends, group.keys, heads
            )
        return schedule

    def _run_packed_group(
        self,
        metadata: CSA2TrtllmMetadata,
        kv_source: int,
        group: CSA2IndexerGroup,
        q_data: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor,
        visible: torch.Tensor,
        output: torch.Tensor,
        publish: Optional[Callable[..., None]],
        sharded: tuple[torch.Tensor, ...],
        radix: tuple[torch.Tensor, torch.Tensor] | None,
        prior: Optional[torch.Tensor],
        prior_start: int,
    ) -> None:
        """Score a prepared group's queries against their own compressed prefixes in one call.

        As in the DSA indexer, each query row is bounded to its request's keys
        up to its visible length, so DeepGEMM's compressed logits column ``j``
        is logical position ``j`` and TopK writes the selected positions
        directly. Attention staging skips a selection without an allocated
        page. ``publish`` receives a candidate source's logits before selection.
        TP-split prefill groups are gathered with the ``sharded`` rows.
        """
        first, last, lo, hi = group.first, group.last, group.lo, group.hi
        if group.width <= 0 and group.sizes is None:
            output[first:last].fill_(-1)
            return
        if group.keys > 0:
            keys, scales, valid, rows = self._gather_prefix(
                metadata, kv_source, group.descriptors, group.keys, hi - lo, visible
            )
            row_starts, row_ends, row_lengths, row_widths, zeros = rows.unbind(0)
            logits = self._call_mqa_logits(
                q_data[lo:hi],
                keys,
                scales,
                weights[lo:hi],
                row_starts,
                row_ends,
                q_scale[lo:hi],
                clean_logits=False,
                max_seqlen_k=-(-group.width // 8) * 8,
                schedule_meta=self._packed_group_schedule(
                    metadata, kv_source, group, visible, row_starts, row_ends, q_data.shape[1]
                ),
            )
            mask_packed_logits_(logits, valid, row_starts, row_lengths)
            if publish is not None:
                publish(logits, valid, 0, lo, row_starts, row_ends, row_widths)
            if group.is_prefill:
                self.select_prepared_scores(logits, output[lo:hi], zeros, row_lengths)
            else:
                local_prior = (
                    prior[lo - prior_start : hi - prior_start] if prior is not None else None
                )
                if local_prior is not None:
                    safe = local_prior.long().clamp(0, logits.shape[1] - 1)
                    backed = (local_prior >= 0) & (local_prior < row_lengths[:, None])
                    backed &= logits.gather(1, safe) > -torch.inf
                    local_prior = torch.where(backed, local_prior, -1)
                self.select_prepared_scores(
                    logits,
                    output[lo:hi],
                    None,
                    row_lengths,
                    is_prefill=False,
                    radix_aux_indices=radix[0][lo:hi],
                    radix_aux_logits=radix[1][lo:hi],
                    gvr_prior_indices=local_prior,
                )
            finalize_selection_(
                output[lo:hi],
                logits.shape[1],
                valid=valid,
                row_starts=row_starts,
                row_lengths=row_lengths,
            )
        if group.sizes is not None:
            tensors = (output, *sharded)
            gathered = allgather(
                [tensor[lo:hi] for tensor in tensors], metadata.mapping, dim=0, sizes=group.sizes
            )
            for tensor, value in zip(tensors, gathered):
                tensor[first:last] = value

    def _publish_block_candidates(
        self,
        logits: torch.Tensor,
        valid: torch.Tensor,
        valid_stride: int,
        visible: torch.Tensor,
        query_start: int,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        row_widths: torch.Tensor,
        output: torch.Tensor,
        sparse_ids: torch.Tensor | None,
        sparse_counts: torch.Tensor | None,
        page: int = 1,
    ) -> None:
        """Publish candidate blocks: block maxima, block TopK, ascending positions.

        Row ``q`` scores logits columns ``[0, row_ends[q] - row_starts[q])``
        where ``valid[q * valid_stride + (row_starts[q] + column) // page]`` is set;
        its blocks span ``row_widths[q]`` positions and the newest visible block is
        always selected. ``output`` and the sparse outputs are the rows'
        publication targets.
        """
        block_size = self.layout.candidate_block_size
        count = logits.shape[0]
        blocks = torch.empty(
            (count, -(-logits.shape[1] // block_size)), dtype=torch.float32, device=logits.device
        )
        block_counts = torch.empty(count, dtype=torch.int32, device=logits.device)
        candidate_block_max_(
            logits,
            valid,
            valid_stride,
            visible,
            query_start,
            row_starts,
            row_ends,
            row_widths,
            blocks,
            block_counts,
            block_size,
            page,
        )
        selected = torch.empty(
            (count, self.candidate_top_k.top_k), dtype=torch.int32, device=logits.device
        )
        self.select_prepared_scores(
            blocks,
            selected,
            None,
            block_counts,
            candidate=True,
            # The CUDA radix decode Top-K needs split-work buffers; its prefill form does not.
            is_prefill=(
                count > _CANDIDATE_CLUSTER_TOPK_MAX_ROWS
                or blocks.shape[1] <= self.candidate_top_k.top_k
                or self.candidate_top_k.decode_implementation == TopKImplementation.CUDA_RADIX
            ),
        )
        expand_candidates_(
            selected,
            blocks,
            visible,
            query_start,
            row_widths,
            output,
            block_size,
            sparse_ids,
            sparse_counts,
            self.sparse_block if sparse_ids is not None else 0,
        )

    def _invalidate_emission_holes(self, bad_rows: torch.Tensor) -> None:
        """Force stock masked-score GVR for rows whose raw emission saw page holes.

        The native emitter runs before logical masking. Invalidating just its
        prior or seeds is insufficient: candidate lists have independent
        admission controls. All side channels are reset on the device so a
        captured graph can change page validity without a host-side branch.
        """
        state = self.top_k._gvr_emission_state
        if state is None:
            return
        count = bad_rows.shape[0]
        mask = bad_rows[:, None]
        state.seed_row[:count, :3].masked_fill_(mask, torch.inf)
        state.seed_row[:count, 3:].masked_fill_(mask, 0)
        state.seed_rungs[:count].masked_fill_(mask, torch.inf)
        state.xstate[:count].masked_fill_(mask, 0)
        if state.cand_ctl is not None:
            rows = min(count, state.cand_ctl.shape[0])
            state.cand_ctl[:rows].masked_fill_(bad_rows[:rows, None], 0)
            state.cand_ctl[:rows, 1].masked_fill_(bad_rows[:rows], 1)
        if state.block_max is not None:
            state.block_max[:count].masked_fill_(mask, torch.inf)

    def _select_mapped_logits(
        self,
        logits: torch.Tensor,
        logical_positions: torch.Tensor,
        output_indices: torch.Tensor,
        valid_counts: torch.Tensor,
        is_prefill: bool,
        radix_aux_indices: Optional[torch.Tensor] = None,
        radix_aux_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select over sparse candidate logits whose columns map to ``logical_positions``.

        Each row's valid columns are its leading ``valid_counts`` ones and
        bound the TopK. Decode rows that one radix-select CTA holds run it and the mapping in one
        launch (``select_mapped_consumer_topk_``), ties at the k-th score lowest column first.
        """
        if (
            not is_prefill
            and self.top_k.decode_implementation == TopKImplementation.CUTE_DSL_RADIX
            and output_indices.shape[1] == self.top_k.top_k
            and logical_positions.is_contiguous()
            and output_indices.is_contiguous()
            and fused_consumer_topk_supported(logits, output_indices.shape[1])
        ):
            return select_mapped_consumer_topk_(
                logits, valid_counts, logical_positions, output_indices
            )
        self.select_prepared_scores(
            logits,
            output_indices,
            None,
            valid_counts,
            is_prefill=is_prefill,
            radix_aux_indices=radix_aux_indices,
            radix_aux_logits=radix_aux_logits,
        )
        return map_selection_(output_indices, logical_positions, valid_counts)

    def _select_paged_logits(
        self,
        metadata: CSA2TrtllmMetadata,
        logits: torch.Tensor,
        candidates: Optional[torch.Tensor],
        output_indices: torch.Tensor,
        gvr_prior_indices: Optional[torch.Tensor],
        after_mask: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """Decode selection over in-place paged logits, as the DSA decode indexer does.

        Column ``p`` of the paged logits is logical compressed position ``p``.
        Positions that are not visible or whose page is unallocated are masked
        by one CuTe launch; with ``candidates`` the same launch gathers the
        candidate columns instead. TopK runs over the masked scores, then one
        CuTe launch validates, sorts and pads the selection with ``-1``. Without
        candidates and with the CuTe DSL Top-K enabled, ``fused_decode_select_`` does
        all three in one launch for small decode tiles.
        """
        valid = metadata.csa2_indexer_valid_pages
        visible = metadata.csa2_indexer_visible_lengths
        width = logits.shape[1]
        if width == 0 or (candidates is not None and candidates.shape[1] == 0):
            return output_indices.fill_(-1)
        if candidates is None:
            if (
                gvr_prior_indices is None
                and not self.top_k._gvr_emission_armed
                and self._exact_top_k == TopKImplementation.CUTE_DSL_RADIX
                and fused_decode_select_supported(logits, output_indices.shape[1])
            ):
                # The candidate publication scores validated, visible columns only, so it reads
                # the logits unmasked as well.
                if after_mask is not None:
                    after_mask()
                return fused_decode_select_(logits, valid, visible, output_indices, INDEX_PAGE_ROWS)
            mask_logits_(logits, valid, row_lengths=visible)
            if after_mask is not None:
                after_mask()
            scores, positions, row_ends = logits, None, visible
            if gvr_prior_indices is not None:
                # A prior on a masked column would seed the GVR threshold with -inf.
                prior = gvr_prior_indices.long().clamp(0, width - 1)
                backed = (
                    valid.gather(1, prior // INDEX_PAGE_ROWS)
                    & (gvr_prior_indices >= 0)
                    & (gvr_prior_indices < width)
                    & (gvr_prior_indices < visible[:, None])
                )
                gvr_prior_indices = torch.where(backed, gvr_prior_indices, -1)
        else:
            scores = torch.empty(candidates.shape, dtype=torch.float32, device=logits.device)
            positions = torch.empty(candidates.shape, dtype=torch.int32, device=logits.device)
            gather_candidate_scores(logits, valid, candidates, visible, scores, positions)
            row_ends = None
        if self.top_k._gvr_emission_armed:
            # Raw emission ran over every visible column, including the ones
            # backed by unallocated pages that were masked above.
            offsets = torch.arange(width, device=logits.device)
            holes = (offsets[None, :] < visible[:, None]) & ~valid[:, offsets // INDEX_PAGE_ROWS]
            self._invalidate_emission_holes(holes.any(dim=1))
        self.select_prepared_scores(
            scores,
            output_indices,
            None,
            row_ends,
            is_prefill=False,
            radix_aux_indices=metadata.csa2_indexer_radix_aux_indices,
            radix_aux_logits=metadata.csa2_indexer_radix_aux_logits,
            gvr_prior_indices=gvr_prior_indices,
        )
        if positions is None:
            return finalize_selection_(output_indices, width, valid=valid, row_lengths=visible)
        return finalize_selection_(
            output_indices, positions.shape[1], scores=scores, positions=positions
        )

    def forward(self, state: CSA2ForwardState, query_start: int, count: int) -> torch.Tensor:
        """Predict the complete model batch through the shared phase dispatcher.

        Attention tiles consume metadata.csa2_indices after this call. The
        request ranges describe the real context/decode batch, independently of
        the synthetic one-query decode requests used by the attention kernel.
        """
        metadata = state.metadata
        manager = metadata.kv_cache_manager
        if manager is None or state.index_q is None or state.index_weights is None:
            raise ValueError("CSA2 indexer requires an owner cache, queries and head weights")
        # The model stream joins the layer's side-stream work (index weights, paged
        # descriptors) only where it first reads it (``_join_index_ready``).
        self.__dict__["_pending_index_ready"] = getattr(state, "index_ready", None)
        total_queries = state.swa_kv.shape[0]
        if query_start != 0 or count != total_queries:
            raise ValueError("CSA2 indexer must run once for the complete model query batch")
        if state.index_weights.shape != (count, self.n_heads):
            raise ValueError("CSA2 index weights must match the layer's query rows and heads")
        if state.index_q_scale is None:
            if state.index_q.shape != (count, self.n_heads, self.head_dim):
                raise ValueError("CSA2 index queries must match the layer's full-query geometry")
            q_data, q_scale = pack_index_queries_split(state.index_q)
        else:
            if (
                self.head_dim != 128
                or state.index_q.dtype not in (torch.int8, torch.uint8)
                or state.index_q.shape != (count, self.n_heads, 64)
            ):
                raise ValueError("Packed CSA2 index queries require 128D E2M1 data")
            if (
                state.index_q_scale.dtype not in (torch.int32, torch.uint8)
                or state.index_q_scale.numel() * state.index_q_scale.element_size()
                != count * self.n_heads * 4
            ):
                raise ValueError(
                    "Packed CSA2 index queries require four UE8M0 scale bytes per head"
                )
            q_data = state.index_q.contiguous().view(torch.int8)
            q_scale = state.index_q_scale.contiguous()
        weights = state.index_weights
        # Candidate consumers score with BF16 weights only (sparse MQA logits).
        if not self.layout.layer(self.layer_idx).consumes_candidates:
            if weights.dtype != torch.float32:
                self._join_index_ready()  # The FP32 cast reads the side stream's weights.
            weights = weights.float()
        logical = self.sparse_attn_indexer(
            metadata,
            state.swa_kv,
            q_data,
            None,
            None,
            weights,
            q_scale=q_scale,
        )
        self._join_index_ready()
        return logical

    def _join_index_ready(self) -> None:
        """Make the current stream wait for this layer's side-stream indexer inputs, once."""
        ready = self.__dict__.pop("_pending_index_ready", None)
        if ready is not None:
            torch.cuda.current_stream().wait_event(ready)

    def _paged_mqa_logits(
        self,
        metadata: CSA2TrtllmMetadata,
        q_data: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor,
        emission_kwargs: dict[str, torch.Tensor | int] | None = None,
    ) -> torch.Tensor:
        # Existing native paged specializations support 32/64 heads. Zero
        # weighted padding extends smaller CSA2 index geometries exactly.
        use_dsl = self.options.use_cute_dsl_paged_mqa_logits and IS_CUTLASS_DSL_AVAILABLE
        padded_heads = 64 if use_dsl or self.n_heads > 32 else 32
        if padded_heads != self.n_heads:
            q_data = F.pad(q_data, (0, 0, 0, padded_heads - self.n_heads))
            weights = F.pad(weights, (0, padded_heads - self.n_heads))
            q_scale = F.pad(q_scale, (0, padded_heads - self.n_heads))
        if use_dsl:
            return torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
                q_data.view(torch.uint8),
                q_scale,
                metadata.csa2_indexer_k_cache,
                weights,
                metadata.csa2_indexer_context_lengths.flatten(),
                metadata.csa2_indexer_block_table,
                metadata.csa2_indexer_scheduler_metadata,
                metadata.csa2_indexer_max_seq_len,
                **(emission_kwargs or {}),
            )
        # Queries of one request share its block table row.
        next_n = metadata.csa2_indexer_next_n
        requests = q_data.shape[0] // next_n
        if getattr(metadata, "csa2_indexer_split_heads", 0) == self.n_heads:
            # Two 32-head queries per 64-head row: one pass over each request's pages.
            rows = next_n * self.n_heads // 64
            return torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
                q_data.view(torch.uint8).view(requests, rows, 64, -1),
                q_scale.view(requests, rows, 64),
                metadata.csa2_indexer_k_cache,
                weights.view(requests * rows, 64),
                metadata.csa2_indexer_request_context,
                metadata.csa2_indexer_block_table.view(requests, next_n, -1)[:, 0],
                metadata.csa2_indexer_scheduler_metadata,
                metadata.csa2_indexer_max_seq_len,
                row_heads=self.n_heads,
            )
        return self._call_paged_mqa_logits(
            q_data.view(requests, next_n, *q_data.shape[2:]),
            metadata.csa2_indexer_k_cache,
            weights,
            metadata.csa2_indexer_context_lengths.view(requests, next_n),
            metadata.csa2_indexer_block_table.view(requests, next_n, -1)[:, 0],
            metadata.csa2_indexer_scheduler_metadata,
            metadata.csa2_indexer_max_seq_len,
            q_scale.view(requests, next_n, -1),
        )

    def _candidate_prefill_tile_size(self, candidate_width: int) -> int:
        """Query tile for sparse candidate scoring over a request's shared prefix.

        The sparse logits are candidate-wide, not prefix-wide, so the tile is
        bounded by the DSA logits budget over the candidate width and by the
        per-query transients (bf16 logits, their fp32 copy and int64 column
        positions), independently of the prefix length.
        """
        return max(
            1,
            min(
                _INDEXER_MQA_LOGITS_ELEM_BUDGET // max(1, candidate_width),
                (512 << 20) // max(1, 14 * candidate_width),
            ),
        )

    def _sparse_candidate_logits(
        self,
        q_data: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        weights: torch.Tensor,
        sparse_indices: torch.Tensor,
        *,
        keys: tuple[torch.Tensor, torch.Tensor] | None = None,
        valid_counts: Optional[torch.Tensor] = None,
        paged: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Score only each query's published candidate blocks with DeepGEMM.

        Prefill passes the request's packed prefix as ``keys`` (data, scales)
        with the rows' valid candidate-column counts. The scheduler must not
        merge duplicate padding blocks; visible KV lengths can include holes.
        Published block ids remain in packed-workspace coordinates.
        Decode passes the ``paged`` descriptors from ``prepare_sparse_indexer``
        (pages, block table, candidate counts, row -> request map, and the per-forward
        schedule shared by every consumer layer, or None to build one).
        Returns ``[rows, blocks * sparse_block]`` logits in candidate order: the
        kernel's BF16 for a CuTe DSL decode TopK, which selects over them directly,
        and FP32 otherwise.
        """
        count = q_data.shape[0]
        num_blocks = sparse_indices.shape[1]
        q_scale = q_scale.contiguous().view(torch.int32)
        weights = weights.to(torch.bfloat16)
        if paged is None:
            k_data, k_scale = keys
            schedule = get_sparse_mqa_logits_metadata(
                self._constant_rows(count, 0, q_data.device),
                valid_counts.to(torch.int32).contiguous(),
                k_data.shape[0],
                sparse_indices,
                q_data.dtype,
                self.sparse_block,
                False,
            )
            logits = fp8_fp4_sparse_mqa_logits(
                (q_data.view(torch.int8).contiguous(), q_scale.reshape(count, self.n_heads)),
                (
                    k_data.view(torch.int8).contiguous(),
                    k_scale.contiguous().view(torch.int32).reshape(-1),
                ),
                weights,
                schedule,
                num_blocks,
                self.sparse_block,
                False,
            )
        else:
            pages, block_table, valid_counts, row_requests, schedule = paged
            if schedule is None:
                schedule = get_paged_sparse_mqa_logits_metadata(
                    valid_counts,
                    block_table,
                    row_requests,
                    pages.shape[1],
                    sparse_indices,
                    q_data.dtype,
                    self.sparse_block,
                )
            logits = fp8_fp4_paged_sparse_mqa_logits(
                (q_data.unsqueeze(1).contiguous(), q_scale.reshape(count, 1, self.n_heads)),
                pages,
                weights,
                schedule,
                num_blocks,
                self.sparse_block,
            )
            if self.top_k.decode_implementation == TopKImplementation.CUTE_DSL_RADIX:
                return logits
        # A one-row .float() would keep DeepGEMM's padded row stride without the
        # padding; the self-sampling prefill TopK widens rows to their stride.
        return logits.to(torch.float32, memory_format=torch.contiguous_format)

    def sparse_attn_indexer(
        self,
        metadata: CSA2TrtllmMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor | None,
        k_scale: torch.Tensor | None,
        weights: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        is_generation: bool | None = None,
    ) -> torch.Tensor:
        """Schedule CSA2 phases using inherited native kernels and shared TP partitioning.

        Inputs span the complete model batch. K is read from the owner cache;
        the base signature is retained for the backend's indexer contract.
        """
        if is_generation is not None:
            raise ValueError("CSA2 prediction requires the complete model query batch")
        if not (q_fp8.is_cuda and dsl_available()):
            raise RuntimeError("CSA2 indexer requires CUDA and the CuTe DSL")
        count = hidden_states.shape[0]
        layer = self.layout.layer(self.layer_idx)
        visible = metadata.csa2_visible_lengths[self.layer_idx]
        max_positions = metadata.csa2_global_max_positions[layer.kv_source]
        device = q_fp8.device
        logical = torch.empty((count, self.index_topk), dtype=torch.int32, device=device)
        metadata.csa2_indices[self.layer_idx] = logical
        prior = None
        if self.top_k.needs_gvr_prior:
            metadata.register_indexer_reset(
                self.layer_idx, lambda: self.top_k.reset_gvr_emission_rows(slice(None))
            )
            prior = metadata.prepare_indexer_prior(self.layer_idx, self.index_topk)
        candidates = sparse_blocks = sparse_counts = None
        sharded = ()
        publish = None
        if layer.consumes_candidates:
            candidates = metadata.csa2_candidates.get(layer.candidate_source)
            if candidates is None or candidates.ndim != 2 or candidates.shape[0] != count:
                raise ValueError(
                    "CSA2 candidate source must publish matching full-query rows first"
                )
            sparse_blocks = metadata.csa2_candidate_blocks.get(layer.candidate_source)
            sparse_counts = metadata.csa2_candidate_counts.get(layer.candidate_source)
            metadata.wait_candidates(layer.candidate_source)
            if not (self.use_sparse_candidates and 0 < sparse_blocks.shape[1] <= 4096):
                raise ValueError(
                    "CSA2 candidate consumers require DeepGEMM sparse MQA logits over the "
                    "source's aligned sparse-block publication (SM100, 32 heads of 128D rows)"
                )
            sparse_counts = sparse_counts.reshape(-1)
        elif layer.candidate_source == self.layer_idx:
            self._configure_candidate_topk(max_positions)
            block_size = self.layout.candidate_block_size
            candidate_width = (
                min(self.layout.candidate_topk_blocks, -(-max_positions // block_size)) * block_size
            )
            if self.sparse_block:
                # Sparse MQA logits need a multiple of 4 sparse blocks per row; the
                # padded tail stays -1 with no valid columns.
                align = 4 * self.sparse_block
                candidate_width = -(-candidate_width // align) * align
            published = torch.full((count, candidate_width), -1, dtype=torch.int32, device=device)
            metadata.csa2_candidates[self.layer_idx] = published
            sharded = (published,)
            blocks = counts = None
            if self.sparse_block:
                # Rows without a scored key keep block 0 and zero valid columns.
                blocks = torch.zeros(
                    (count, candidate_width // self.sparse_block), dtype=torch.int32, device=device
                )
                counts = torch.zeros((count, 1), dtype=torch.int32, device=device)
                metadata.csa2_candidate_blocks[self.layer_idx] = blocks
                metadata.csa2_candidate_counts[self.layer_idx] = counts
                sharded = (published, blocks, counts)

            def publish(
                logits, valid, valid_stride, first, row_starts, row_ends, row_widths, page=1
            ):
                rows = slice(first, first + logits.shape[0])
                self._publish_block_candidates(
                    logits,
                    valid,
                    valid_stride,
                    visible,
                    first,
                    row_starts,
                    row_ends,
                    row_widths,
                    published[rows],
                    blocks[rows] if blocks is not None else None,
                    counts.view(-1)[rows] if counts is not None else None,
                    page,
                )

        # Host prefix bounds are valid for eager execution only. A graph must
        # admit every future replay's visibility, so use its configured bound.
        graph_mode = metadata.is_cuda_graph or torch.cuda.is_current_stream_capturing()
        short_bound = (
            max_positions
            if graph_mode
            else min(
                max_positions,
                max(
                    (prefix + end - begin) // layer.compress_ratio
                    for prefix, (begin, end) in zip(
                        metadata.csa2_request_start_positions, metadata.csa2_request_query_ranges
                    )
                )
                if metadata.csa2_request_query_ranges
                else 0,
            )
        )
        if (
            self.options.skip_indexer_for_short_seqs
            and publish is None
            and short_bound <= self.index_topk
        ):
            positions = (
                candidates
                if candidates is not None
                else torch.arange(short_bound, device=device).expand(count, -1)
            )
            slots = metadata.global_slot_tile(self.layer_idx, 0, count, positions)
            positions = torch.where(slots >= 0, positions, -1)
            self._select_all_visible(positions, visible, logical)
            if prior is not None:
                metadata.publish_indexer_prior(self.layer_idx, logical)
            return logical

        # Everything below may read the side stream's index weights and paged descriptors.
        self._join_index_ready()
        decode_start, native_decode = self._native_decode(metadata, count)
        paged = (
            metadata.prepare_indexer(self.layer_idx, batch_queries=self._batch_paged_queries())
            if native_decode
            else None
        )
        sparse_descriptors = None
        if paged is not None and candidates is not None:
            # Sparse paged logits read the native index pages in place
            sparse_descriptors = metadata.prepare_sparse_indexer(self.layer_idx, self.sparse_block)
            if sparse_descriptors is None:
                paged = None
        if candidates is None:
            # As in DSA: gathered prefixes, compressed MQA logits, row-bounded TopK.
            if decode_start < count and paged is None and not metadata.csa2_indexer_decode_planned:
                raise ValueError("CSA2 metadata planned no packed groups for these decode rows")
            # Exact radix decode scratch for decode rows outside the native paged path
            radix = None
            if decode_start < count and paged is None:
                radix_indices = torch.empty(
                    (count, 10, self.index_topk), dtype=torch.int32, device=device
                )
                radix = (radix_indices, torch.empty_like(radix_indices, dtype=torch.float32))
            for group in metadata.csa2_indexer_groups[layer.kv_source]:
                if group.is_prefill or paged is None:
                    self._run_packed_group(
                        metadata,
                        layer.kv_source,
                        group,
                        q_fp8,
                        weights,
                        q_scale,
                        visible,
                        logical,
                        publish,
                        sharded,
                        radix,
                        prior,
                        decode_start,
                    )
        else:
            if decode_start < count and paged is None:
                raise ValueError(
                    "CSA2 candidate-consumer decode rows need the native paged sparse decode"
                )
            self._run_candidate_prefill(
                metadata,
                layer.kv_source,
                candidates,
                sparse_blocks,
                sparse_counts,
                q_fp8,
                weights,
                q_scale,
                visible,
                logical,
            )
        if paged is not None:
            if candidates is not None:
                # Score only the published candidate blocks straight from the index pages
                scores = self._sparse_candidate_logits(
                    q_fp8[decode_start:],
                    q_scale[decode_start:],
                    weights[decode_start:],
                    sparse_blocks[decode_start:],
                    paged=sparse_descriptors,
                )
                self._select_mapped_logits(
                    scores,
                    candidates[decode_start:],
                    logical[decode_start:],
                    sparse_counts[decode_start:],
                    is_prefill=False,
                    radix_aux_indices=metadata.csa2_indexer_radix_aux_indices,
                    radix_aux_logits=metadata.csa2_indexer_radix_aux_logits,
                )
            else:
                self._paged_decode(
                    metadata, q_fp8, weights, q_scale, logical, decode_start, prior, publish
                )
        if publish is not None and paged is not None and self._feeds_sparse_consumers():
            # This forward's candidate consumers share one sparse logits schedule over the
            # candidates just published: build it on the side stream now, not in the first
            # consumer's chain.
            metadata.prebuild_sparse_schedule(self.layer_idx, self.sparse_block)
        if prior is not None:
            metadata.publish_indexer_prior(self.layer_idx, logical)
        return logical

    def _native_decode(self, metadata: CSA2TrtllmMetadata, count: int) -> tuple[int, bool]:
        """(first decode row, whether the decode rows take the native paged path)."""
        ranges = metadata.csa2_request_query_ranges
        num_contexts = metadata.csa2_num_context_requests
        decode_start = ranges[num_contexts][0] if num_contexts < len(ranges) else count
        native = (
            decode_start < count
            and 100 <= get_sm_version() < 110
            and 1 <= self.n_heads <= 64
            and self.head_dim == 128
        )
        return decode_start, native

    def _batch_paged_queries(self) -> bool:
        return not (self.options.use_cute_dsl_paged_mqa_logits and IS_CUTLASS_DSL_AVAILABLE)

    def prepare_paged_descriptors(self, metadata: CSA2TrtllmMetadata, count: int) -> None:
        """Bind, and once per owner and forward refresh, the paged indexer descriptors
        ``sparse_attn_indexer`` uses for ``count`` query rows, ahead of it: they depend only on
        the step's page tables and visible lengths, so a Full layer derives them on a side
        stream while it projects, and its indexer then finds them current."""
        if self._native_decode(metadata, count)[1]:
            metadata.prepare_indexer(self.layer_idx, batch_queries=self._batch_paged_queries())

    def _feeds_sparse_consumers(self) -> bool:
        """Whether this candidate source's consumers score its published blocks with the paged
        sparse logits (their ``use_sparse_candidates``)."""
        layers = range(len(self.layout.compress_ratios))
        return (
            _HAS_SPARSE_MQA_LOGITS
            and 100 <= get_sm_version() < 110
            and self.n_heads == 32
            and self.head_dim == 128
            and self.sparse_block > 0
            and any(
                self.layout.layer(i).mode == CSA2Mode.REINDEX
                and self.layout.layer(i).candidate_source == self.layer_idx
                for i in layers
            )
        )

    def _run_candidate_prefill(
        self,
        metadata: CSA2TrtllmMetadata,
        kv_source: int,
        candidates: torch.Tensor,
        sparse_blocks: torch.Tensor,
        sparse_counts: torch.Tensor,
        q_data: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor,
        visible: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Score each context request's candidate blocks in its gathered prefix.

        Query tiles of the request's TP-local rows score with DeepGEMM sparse
        logits; TP-split requests are gathered afterwards. Consecutive requests
        share one gather and their query tiles where ``_batched_candidate_plan``
        applies.
        """
        args = (candidates, sparse_blocks, sparse_counts, q_data, weights, q_scale, visible, output)
        if self._run_candidate_prefill_batched(metadata, kv_source, *args):
            return
        tile = self._candidate_prefill_tile_size(candidates.shape[1])
        for request in range(metadata.csa2_num_context_requests):
            begin, end = metadata.csa2_request_query_ranges[request]
            if begin == end:
                continue
            descriptors, width = metadata.csa2_indexer_prefixes[(kv_source, request)]
            local_start, local_end, sizes = _split_prefill_queries(
                end - begin, metadata.mapping, metadata.indexer_q_split_threshold
            )
            keys, scales, _, _ = self._gather_prefix(
                metadata, kv_source, descriptors, width, end - begin, visible
            )
            for first in range(begin + local_start, begin + local_end, tile):
                last = min(first + tile, begin + local_end)
                logits = self._sparse_candidate_logits(
                    q_data[first:last],
                    q_scale[first:last],
                    weights[first:last],
                    sparse_blocks[first:last],
                    keys=(keys, scales),
                    valid_counts=sparse_counts[first:last].reshape(-1),
                )
                self._select_mapped_logits(
                    logits,
                    candidates[first:last],
                    output[first:last],
                    sparse_counts[first:last],
                    True,
                )
            if sizes is not None:
                output[begin:end] = allgather(
                    output[begin + local_start : begin + local_end],
                    metadata.mapping,
                    dim=0,
                    sizes=sizes,
                )

    def _batched_candidate_plan(
        self, metadata: CSA2TrtllmMetadata, kv_source: int, device: torch.device
    ):
        """Batches of consecutive context requests whose candidate rows score together, or None.

        A batch ``(lo, hi, keys, descriptors, block_offsets)`` covers query rows ``[lo, hi)``:
        its requests' prefixes are gathered as on their own into ``keys`` keys, each at a key
        offset aligned to ``_BATCHED_CANDIDATE_ALIGN``, and ``block_offsets`` holds every row's
        request offset in sparse blocks. None with fewer than two non-empty requests, a TP-split
        request, an empty prefix or non-contiguous rows. Built once per prepare and sparse block.
        """
        prefixes = metadata.csa2_indexer_prefixes
        cached = metadata.__dict__.get("_csa2_batched_candidate_plans")
        if cached is None or cached[0] is not prefixes:
            cached = metadata._csa2_batched_candidate_plans = (prefixes, {})
        plans = cached[1]
        key = (kv_source, self.sparse_block)
        if key in plans:
            return plans[key]
        plans[key] = None
        pieces = []
        for request in range(metadata.csa2_num_context_requests):
            begin, end = metadata.csa2_request_query_ranges[request]
            if begin == end:
                continue
            local_start, local_end, sizes = _split_prefill_queries(
                end - begin, metadata.mapping, metadata.indexer_q_split_threshold
            )
            width = prefixes[(kv_source, request)][1]
            if sizes is not None or (local_start, local_end) != (0, end - begin) or width <= 0:
                return None
            if pieces and pieces[-1][2] != begin:
                return None
            pieces.append((request, begin, end, width))
        if len(pieces) < 2:
            return None
        align = _BATCHED_CANDIDATE_ALIGN
        groups, group, keys = [], [], 0
        for piece in pieces:
            span = -(-piece[3] // align) * align
            if group and keys + span > _BATCHED_CANDIDATE_KEYS:
                groups.append(group)
                group, keys = [], 0
            group.append(piece)
            keys += span
        groups.append(group)
        plan = []
        for group in groups:
            lo, hi = group[0][1], group[-1][2]
            rows, offsets, offset = [], [], 0
            for request, begin, end, width in group:
                rows.append((request, width, offset, begin, begin - lo))
                offsets.append(np.full(end - begin, offset // self.sparse_block, dtype=np.int32))
                offset += -(-width // align) * align
            host = maybe_pin_memory(
                torch.from_numpy(
                    np.concatenate((np.asarray(rows, dtype=np.int32).reshape(-1), *offsets))
                )
            )
            packed = host.to(device, non_blocking=True)
            split = len(rows) * PREFIX_DESC_WIDTH
            descriptors = packed[:split].view(len(rows), PREFIX_DESC_WIDTH)
            plan.append((lo, hi, offset, descriptors, packed[split:]))
        plans[key] = plan
        return plan

    def _run_candidate_prefill_batched(
        self,
        metadata: CSA2TrtllmMetadata,
        kv_source: int,
        candidates: torch.Tensor,
        sparse_blocks: torch.Tensor,
        sparse_counts: torch.Tensor,
        q_data: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor,
        visible: torch.Tensor,
        output: torch.Tensor,
    ) -> bool:
        """``_run_candidate_prefill`` over the batches of ``_batched_candidate_plan``.

        Every row scores the same keys against the same blocks as on its own request's path,
        and selection and mapping are row-wise. False (nothing launched) when no plan applies
        or under graph capture.
        """
        if metadata.is_cuda_graph or torch.cuda.is_current_stream_capturing():
            return False
        plan = self._batched_candidate_plan(metadata, kv_source, q_data.device)
        if plan is None:
            return False
        tile = self._candidate_prefill_tile_size(candidates.shape[1])
        for lo, hi, keys, descriptors, block_offsets in plan:
            data, scales, _, _ = self._gather_prefix(
                metadata, kv_source, descriptors, keys, hi - lo, visible
            )
            for first in range(lo, hi, tile):
                last = min(first + tile, hi)
                # DeepGEMM reads a row's sparse block b at key b * sparse_block and adds no row
                # key start, so each row's request-relative block ids move by its offset.
                blocks = sparse_blocks[first:last] + block_offsets[first - lo : last - lo, None]
                logits = self._sparse_candidate_logits(
                    q_data[first:last],
                    q_scale[first:last],
                    weights[first:last],
                    blocks,
                    keys=(data, scales),
                    valid_counts=sparse_counts[first:last].reshape(-1),
                )
                self._select_mapped_logits(
                    logits,
                    candidates[first:last],
                    output[first:last],
                    sparse_counts[first:last],
                    True,
                )
        return True

    def _paged_decode(
        self,
        metadata: CSA2TrtllmMetadata,
        q_data: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor,
        output: torch.Tensor,
        decode_start: int,
        prior: Optional[torch.Tensor],
        publish: Optional[Callable[..., None]],
    ) -> None:
        """Native paged decode, as in the DSA decode indexer.

        Column ``p`` of the paged logits is logical compressed position ``p``,
        so TopK writes the selected positions directly; attention staging skips
        any selection without a backed, visible row.
        """
        decode_count = q_data.shape[0] - decode_start
        decode_scale = (
            q_scale[decode_start:]
            .contiguous()
            .view(torch.int32)
            .reshape(decode_count, 1, self.n_heads)
        )
        emission_kwargs = {}
        if self.options.use_gvr_emission and prior is not None:
            emission_kwargs = self.top_k.prepare_gvr_emission(
                decode_count,
                metadata.csa2_indexer_max_seq_len,
                torch.cuda.get_device_properties(q_data.device).multi_processor_count,
                metadata.csa2_indexer_prior_capacity[self.layer_idx],
            )
        logits = self._paged_mqa_logits(
            metadata,
            q_data[decode_start:].unsqueeze(1),
            weights[decode_start:],
            decode_scale,
            emission_kwargs=emission_kwargs,
        )
        if logits.shape[1] == 0:
            output[decode_start:].fill_(-1)
            return
        after_mask = None
        if publish is not None:
            valid = metadata.csa2_indexer_valid_pages
            visible = metadata.csa2_indexer_visible_lengths
            zeros = self._constant_rows(decode_count, 0, logits.device)

            # Rows score and publish only their visible prefix, not the arena width.
            def publish_decode():
                publish(
                    logits,
                    valid,
                    valid.stride(0),
                    decode_start,
                    zeros,
                    visible,
                    visible,
                    INDEX_PAGE_ROWS,
                )

            # Only later consumer layers read the candidates: publish them on the side stream,
            # next to this layer's own TopK. The block maxima score validated, visible columns
            # only, which the selection's masking leaves as they are, and the TopK only reads
            # the logits.
            def after_mask():
                self._publish_on_side_stream(
                    metadata, publish_decode, (logits, valid, zeros, visible)
                )

        self._select_paged_logits(
            metadata, logits, None, output[decode_start:], prior, after_mask=after_mask
        )

    def _publish_on_side_stream(self, metadata, run, tensors) -> None:
        """Run this source's candidate publication on the CSA2 side stream after the calling
        stream's work so far; consumers of this forward wait for it (``wait_candidates``)."""
        current = torch.cuda.current_stream()
        side = csa2_side_stream(current.device)
        ready = torch.cuda.Event()
        ready.record(current)
        with torch.cuda.stream(side):
            side.wait_event(ready)
            for tensor in tensors:
                tensor.record_stream(side)
            for tensor in (
                metadata.csa2_candidates.get(self.layer_idx),
                metadata.csa2_candidate_blocks.get(self.layer_idx),
                metadata.csa2_candidate_counts.get(self.layer_idx),
            ):
                if tensor is not None:
                    tensor.record_stream(side)
            run()
            done = torch.cuda.Event()
            done.record(side)
        metadata.publish_candidates_event(self.layer_idx, done)


def prewarm_prefill_topk(top_ks, max_cols: int = 32768) -> list[int]:
    """Compile the self-sampling GVR prefill Top-K engines of ``top_ks`` ahead of serving.

    ``run_prefill`` compiles one engine per (row tier, Top-K width, width bucket) on first use,
    so a context pass narrower than every warm-up row would compile inside the forward while
    its attention-DP peers wait in the MoE all-to-all. Returns the Top-K widths compiled.
    """
    widths = sorted(
        {
            int(top_k.top_k)
            for top_k in top_ks
            if top_k.prefill_implementation == TopKImplementation.CUTE_DSL_GVR
            and top_k.gvr_self_sampling
        }
    )
    if widths:
        from tensorrt_llm._torch.kernels.blackwell.top_k import (
            gvr_topk_decode_self_sampling_host as host,
        )

        for width in widths:
            host.warmup_prefill(width, max_cols)
    return widths


def prewarm_sparse_candidate_kernels(
    layout: CSA2Layout, heads: int, head_dim: int, device: torch.device
) -> bool:
    """JIT-compile DeepGEMM's sparse MQA-logits kernels for the candidate consumers.

    Both the packed-prefix (prefill) and paged (decode) kernels and their
    schedule builders are specialized on the candidate geometry only, so one
    call with the model's block budget covers every consumer layer. Run it
    before the first collective-bearing forward so the JIT cost is not charged
    to the warm-up forward (and to the MoE all-to-all completion deadline);
    this also creates DeepGEMM's per-stream workspace on the launching stream.
    Returns False when the model has no sparse candidate consumer.
    """
    consumer = next(
        (
            index
            for index in layout.index_source_layer_ids
            if layout.layer(index).consumes_candidates
        ),
        None,
    )
    if consumer is None:
        return False
    indexer = CSA2Indexer(layout, consumer, heads, head_dim)
    if not indexer.use_sparse_candidates:
        return False
    rows, blocks = 2, layout.candidate_topk_blocks
    q = torch.zeros((rows, heads, head_dim // 2), dtype=torch.int8, device=device)
    q_scale = torch.zeros((rows, heads), dtype=torch.int32, device=device)
    weights = torch.zeros((rows, heads), dtype=torch.float32, device=device)
    indices = torch.zeros((rows, blocks), dtype=torch.int32, device=device)
    keys = torch.zeros((indexer.sparse_block, head_dim // 2), dtype=torch.int8, device=device)
    indexer._sparse_candidate_logits(
        q,
        q_scale,
        weights,
        indices,
        keys=(keys, torch.zeros(indexer.sparse_block, dtype=torch.int32, device=device)),
        valid_counts=torch.zeros(rows, dtype=torch.int32, device=device),
    )
    pages = torch.zeros(
        (1, INDEX_PAGE_ROWS, 1, INDEX_DATA_BYTES + INDEX_SCALE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    indexer._sparse_candidate_logits(
        q,
        q_scale,
        weights,
        indices,
        paged=(
            pages,
            torch.zeros((rows, 1), dtype=torch.int32, device=device),
            torch.ones(rows, dtype=torch.int32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device),
            None,
        ),
    )
    return True
