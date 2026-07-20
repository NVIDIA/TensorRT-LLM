# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TriAttention KV-cache compression: periodic physical KV eviction.

Every ``beta`` confirmed generation tokens TriAttention scores each cached token with a
trigonometric importance score (computed from offline-calibrated statistics of
the model's pre-RoPE query vectors) and physically deletes the tokens below the
top-B keep set. There is no context-phase work and no per-step attention mask:
the eviction runs in the compression manager's final
``on_generation_step_end`` hook.

TriAttention is a :class:`BaseKVCacheCompressionManager` and nothing more -- it
has no attention backend of its own; decode runs the model's standard dense
kernel over the compacted cache. TriAttention derives each request's effective
confirmed physical length after V2's native update/rewind and publishes the
cumulative evicted count on ``LlmRequest.py_num_compressed_tokens``; attention
metadata reads it back through the KV cache manager on the next step.
Physical reclaim uses V2's existing resize path directly after compaction. An
already-enqueued speculative suffix is excluded from scoring and appended
unchanged to the retained prefix by the same per-layer compact operation.
With one-model speculative decoding, the separate draft KV cache is compacted
in the same round with the target's kept token set (union mode only), so
target and draft always share one physical KV length.

KV layout: the decode kernel stores keys in HND layout
``[num_pages, kv_factor, num_kv_heads, tokens_per_block, head_dim]``. The Python
gather / score / compact code MUST read ``get_buffers`` with ``kv_layout="HND"``;
reading the default NHD silently swaps the token and head axes and scrambles the
cache.

Position handling: kept keys retain their original RoPE rotation (no re-RoPE on
compaction). The model engine keeps the decode query at its true absolute
position while the attention metadata uses the compacted physical length, so a
query against a kept key at its original rotation still yields the correct
relative distance.

Calibration is NOT computed here: the user calibrates with the official tool
(github.com/WeianMao/triattention) and passes that .pt via ``calibration_path``;
the manager converts it to our runtime schema at load (see _resolve_calibration).
The scoring math follows the same upstream reference (``methods/pruning_utils.py``).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseKVCacheCompressionManager
from tensorrt_llm._utils import nvtx_range, nvtx_range_debug, prefer_pinned
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    copy_batch_block_offsets_to_device,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler import ScheduledRequests


# Required keys for the calibration ``.pt`` consumed by TriAttention.
_REQUIRED_CALIBRATION_KEYS = frozenset({"E_q", "E_q_norm", "omega", "freq_scale_sq"})


def _build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    """Upstream pruning_utils.build_geometric_offsets: [1, 2, 4, ... <=max]."""
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


class _FixedScoreStreamMismatch(RuntimeError):
    """Raised when fixed score staging buffers are used from another CUDA stream."""


class _CrossRequestSelectionPlan(NamedTuple):
    """Selection dimensions used to allocate reusable fixed buffers."""

    eviction_mode: str
    dense_layers: Tuple[int, ...]
    num_query_heads: int
    num_kv_heads: int
    rows: int
    width: int
    keep_count: int
    dtype: torch.dtype
    device: torch.device
    max_requests: int


class _RuntimeKVLayout(NamedTuple):
    """Manager-lifetime layer and pool views used by every eviction."""

    manager: object
    num_layers: int
    global_layers: List[int]
    layer_pools: List[torch.Tensor]
    dense_layers: List[int]
    swa_layers: List[int]
    swa_window: Optional[int]
    storage_groups: Dict[object, List[int]]
    layer_group_representative: Dict[int, int]
    layer_pool_keys: Tuple[object, ...]
    pool_representatives: Tuple[int, ...]
    pool_page_counts: Tuple[int, ...]
    pool_view_fingerprint: Tuple[tuple, ...]


class _BatchedKeepSetSelectorBase:
    """Shared fixed buffers and row views for keep-set selectors."""

    def __init__(
        self,
        *,
        eviction_mode: str,
        dense_layers: Tuple[int, ...],
        num_query_heads: int,
        num_kv_heads: int,
        width: int,
        keep_count: int,
        selection_rows_per_request: int = 1,
        prompt_offsets_buffer: Optional[torch.Tensor] = None,
        dtype: torch.dtype,
        device: torch.device,
        max_requests: int,
    ) -> None:
        if width <= keep_count or keep_count <= 0:
            raise ValueError("keep-set selection requires width > keep_count > 0")
        if max_requests <= 0:
            raise ValueError("keep-set selection requires a positive request capacity")
        self.eviction_mode = eviction_mode
        self.dense_layers = tuple(int(layer) for layer in dense_layers)
        self.num_query_heads = int(num_query_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.width = int(width)
        self.keep_count = int(keep_count)
        self.device = device
        self.max_requests = int(max_requests)
        self.valid_widths = torch.full(
            (self.max_requests,), self.width, dtype=torch.int32, device=self.device
        )
        # Per-request pinned prompt lengths, refreshed each round: scores are
        # decode-relative and these offsets rebase emitted ordinals, so one
        # cohort may mix prompt lengths. ``row_prompt_offsets`` is the
        # row-major expansion consumed by the finalizer.
        self.selection_rows_per_request = int(selection_rows_per_request)
        # Optional compaction move packing fused into the settle launch; set
        # once the compaction buffers exist (see ``fuse_move_source_pack``).
        self._move_source_pack = None
        if prompt_offsets_buffer is not None:
            # Share the staging buffers' per-request prompt lengths so the
            # values are written once per round.
            if (
                prompt_offsets_buffer.shape != (self.max_requests,)
                or prompt_offsets_buffer.dtype != torch.int32
                or prompt_offsets_buffer.device != self.device
            ):
                raise ValueError("prompt offsets buffer does not match the selector geometry")
            self.prompt_offsets = prompt_offsets_buffer
        else:
            self.prompt_offsets = torch.zeros(
                (self.max_requests,), dtype=torch.int32, device=self.device
            )
        if self.selection_rows_per_request == 1:
            self.row_prompt_offsets = self.prompt_offsets
        else:
            self.row_prompt_offsets = torch.zeros(
                (self.max_requests * self.selection_rows_per_request,),
                dtype=torch.int32,
                device=self.device,
            )

    def refresh_row_prompt_offsets(self) -> None:
        """Re-expand the per-request prompt offsets into their row-major view.

        Called after the shared per-request buffer was staged externally.
        """
        if self.row_prompt_offsets is not self.prompt_offsets:
            self.row_prompt_offsets.view(self.max_requests, self.selection_rows_per_request).copy_(
                self.prompt_offsets.unsqueeze(1).expand(-1, self.selection_rows_per_request)
            )

    def _bind_selection_rows(
        self,
        scores_rows: torch.Tensor,
        row_lengths: torch.Tensor,
        provisional_indices: torch.Tensor,
        keep_rows: torch.Tensor,
    ) -> None:
        """Keep row-major views of the buffers the top-k selection reads."""
        self._selection_scores_rows = scores_rows
        self._selection_row_lengths = row_lengths
        self._provisional_rows = provisional_indices
        self._keep_rows = keep_rows

    def fuse_move_source_pack(self, pack_arguments) -> None:
        """Pack compaction move sources inside this selector's settle launch.

        ``pack_arguments`` is the dense/SWA packing description exported by
        ``BatchedKVCacheCompaction.hand_move_source_pack_to_selection``
        (fusion suggested by Fanrong Li, torch-graph review 2026-07-20). The
        fused kernel reads back the kept ordinals it just wrote, so the
        packing must read this selector's own keep buffer, and the packing
        geometry must match the selection rows this selector settles.
        """
        if (
            pack_arguments.kept_token_ordinals.data_ptr() != self._keep_rows.data_ptr()
            or pack_arguments.kept_token_ordinals.numel() != self._keep_rows.numel()
        ):
            raise ValueError("fused move packing must read this selector's keep buffer")
        if (
            pack_arguments.selection_rows != self.selection_rows_per_request
            or pack_arguments.keep_count != self.keep_count
            or pack_arguments.request_count * self.selection_rows_per_request
            != int(self._keep_rows.shape[0])
        ):
            raise ValueError("fused move packing does not match the selector geometry")
        self._move_source_pack = pack_arguments

    def _select_top_tokens(self) -> None:
        """Pick the top-k with the CuTE selector, then settle its output.

        The CuTE top-k is fast but breaks score ties arbitrarily and emits
        indices in arbitrary order; the settle kernel recomputes the threshold
        membership with lowest-index-wins ties, rebases each row by its prompt
        offset, and writes sorted ordinals. When a compaction move packing is
        fused in, the same launch also packs each request's dense/SWA move
        source indices from the ordinals it just settled.
        """
        from .triattention_kernels import _settle_ties_and_pack_compaction_sources_kernel

        rows = int(self._selection_scores_rows.shape[0])
        # The trailing 1 is next_n: decode scores one query token per request.
        torch.ops.trtllm.cute_dsl_indexer_topk_decode(
            self._selection_scores_rows,
            self._selection_row_lengths,
            self._provisional_rows,
            self.keep_count,
            1,
        )
        pack = self._move_source_pack
        if pack is None:
            # Settle only: the pack half is compiled away, so its tensor
            # parameters are never read; any resident tensor stands in.
            placeholder = self._selection_row_lengths
            pack_tensors = (placeholder,) * 5
            pack_shape = dict(
                DENSE_TOTAL=0,
                SWA_TOTAL=0,
                MOVE_CAPACITY=0,
                NUM_KV_HEADS=1,
                SWA_WINDOW=0,
                UNION=False,
                PER_LAYER=False,
                HAS_SWA=False,
                HAS_PACK=False,
            )
        else:
            pack_tensors = (
                pack.valid_sequence_lengths,
                pack.dense_offsets,
                pack.dense_indices,
                pack.swa_offsets,
                pack.swa_indices,
            )
            pack_shape = dict(
                DENSE_TOTAL=pack.dense_total,
                SWA_TOTAL=pack.swa_total,
                MOVE_CAPACITY=pack.move_capacity,
                NUM_KV_HEADS=pack.num_kv_heads,
                SWA_WINDOW=pack.swa_window,
                UNION=pack.union,
                PER_LAYER=pack.per_layer,
                HAS_SWA=pack.has_swa,
                HAS_PACK=True,
            )
        _settle_ties_and_pack_compaction_sources_kernel[
            (rows // self.selection_rows_per_request, self.selection_rows_per_request)
        ](
            self._selection_scores_rows,
            self._selection_row_lengths,
            self.row_prompt_offsets,
            self._provisional_rows,
            self._keep_rows,
            *pack_tensors,
            WIDTH=self.width,
            KEEP_COUNT=self.keep_count,
            OUTPUT_WIDTH=self.keep_count,
            SELECTION_ROWS=self.selection_rows_per_request,
            **pack_shape,
            BLOCK=256,
            num_warps=4,
        )


class _BatchedUnionKeepSetSelector(_BatchedKeepSetSelectorBase):
    """Persistent ``[request, ...]`` buffers for union selection."""

    def __init__(
        self,
        rows: int,
        width: int,
        keep_count: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        max_requests: int,
        dense_layers: Tuple[int, ...] = (),
        num_query_heads: int = 0,
        num_kv_heads: int = 0,
        input_scores: Optional[torch.Tensor] = None,
        normalize_scores: bool = True,
        prompt_offsets_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        if rows <= 0:
            raise ValueError("cross-request selection requires rows > 0")
        super().__init__(
            eviction_mode="union",
            dense_layers=dense_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            width=width,
            keep_count=keep_count,
            prompt_offsets_buffer=prompt_offsets_buffer,
            dtype=dtype,
            device=device,
            max_requests=max_requests,
        )
        if input_scores is None:
            raise ValueError("union selection requires its fixed score input")

        self.row_mean = torch.empty((max_requests, rows, 1), dtype=dtype, device=self.device)
        self.row_std = torch.empty_like(self.row_mean)
        self.combined = torch.empty((max_requests, width), dtype=dtype, device=self.device)
        self.final_indices = torch.empty(
            (max_requests, keep_count), dtype=torch.int32, device=self.device
        )
        # Kept decode ordinals only: rows are prompt-length independent, so
        # one selector serves cohorts with mixed prompt lengths.
        self.keep = torch.empty(
            (max_requests, self.keep_count), dtype=torch.int32, device=self.device
        )
        self._bind_selection_rows(self.combined, self.valid_widths, self.final_indices, self.keep)
        # Callers select from exactly this tensor with exactly this flag.
        self.input_scores = input_scores
        self.normalize_scores = bool(normalize_scores)

    def select_prepared_requests(self) -> None:
        """Select from the CUDA score tensor bound to this fixed selector."""
        from .triattention_kernels import prepare_union_scores

        prepare_union_scores(
            self.input_scores,
            self.valid_widths,
            self.row_mean,
            self.row_std,
            self.combined,
            self.max_requests,
            normalize_scores=self.normalize_scores,
        )
        self._select_top_tokens()


class _BatchedPerHeadKeepSetSelector(_BatchedKeepSetSelectorBase):
    """Fixed ``[request, ...]`` selector for both per-head modes."""

    def __init__(
        self,
        *,
        eviction_mode: str,
        dense_layers: Tuple[int, ...],
        num_query_heads: int,
        num_kv_heads: int,
        width: int,
        keep_count: int,
        dtype: torch.dtype,
        device: torch.device,
        max_requests: int,
        prompt_offsets_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        if eviction_mode not in ("per_head", "per_layer_perhead"):
            raise ValueError(f"unsupported per-head eviction mode: {eviction_mode}")
        if not dense_layers or min(num_query_heads, num_kv_heads, max_requests) <= 0:
            raise ValueError("per-head selection requires positive layer, head, and request counts")
        if num_query_heads % num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        selection_rows = (
            num_kv_heads if eviction_mode == "per_head" else len(dense_layers) * num_kv_heads
        )
        super().__init__(
            eviction_mode=eviction_mode,
            dense_layers=dense_layers,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            width=width,
            keep_count=keep_count,
            selection_rows_per_request=selection_rows,
            prompt_offsets_buffer=prompt_offsets_buffer,
            dtype=dtype,
            device=device,
            max_requests=max_requests,
        )
        self.num_layers = len(self.dense_layers)
        self.selection_rows = selection_rows

        score_shape = (self.max_requests, self.num_layers, self.num_query_heads, self.width)
        self.row_mean = torch.empty(score_shape[:-1] + (1,), dtype=dtype, device=self.device)
        self.row_std = torch.empty_like(self.row_mean)
        self.selection_scores = torch.empty(
            (self.max_requests, self.selection_rows, self.width),
            dtype=dtype,
            device=self.device,
        )
        self.row_seq_lens = torch.full(
            (self.max_requests, self.selection_rows),
            self.width,
            dtype=torch.int32,
            device=self.device,
        )
        selection_shape = (self.max_requests, self.selection_rows, self.keep_count)
        self.top_indices_i32 = torch.empty(selection_shape, dtype=torch.int32, device=self.device)
        # Kept decode ordinals only: rows are prompt-length independent, so
        # one selector serves cohorts with mixed prompt lengths.
        self.keep = torch.empty(
            (self.max_requests, self.selection_rows, self.keep_count),
            dtype=torch.int32,
            device=self.device,
        )
        self.selection_scores_flat = self.selection_scores.view(
            self.max_requests * self.selection_rows, self.width
        )
        self.row_seq_lens_flat = self.row_seq_lens.view(-1)
        self.top_indices_i32_flat = self.top_indices_i32.view(-1, self.keep_count)
        self.keep_flat = self.keep.view(-1, self.keep_count)
        self._bind_selection_rows(
            self.selection_scores_flat,
            self.row_seq_lens_flat,
            self.top_indices_i32_flat,
            self.keep_flat,
        )

    def select_requests(
        self,
        scores: torch.Tensor,
        *,
        normalize_scores: bool,
    ) -> None:
        from .triattention_kernels import prepare_per_head_scores

        expected_shape = (
            self.max_requests,
            self.num_layers,
            self.num_query_heads,
            self.width,
        )
        if tuple(scores.shape) != expected_shape or not scores.is_contiguous():
            raise ValueError("per-head scores do not match the selector geometry")
        prepare_per_head_scores(
            scores,
            self.valid_widths,
            self.row_mean,
            self.row_std,
            self.selection_scores,
            self.row_seq_lens,
            self.max_requests,
            num_kv_heads=self.num_kv_heads,
            per_layer=self.eviction_mode == "per_layer_perhead",
            normalize_scores=normalize_scores,
        )
        self._select_top_tokens()


class _FixedScoreStagingBuffers:
    """Pool-bound fixed score metadata with one nonblocking page-table upload."""

    @staticmethod
    def _page_table_slot_layout(
        page_representatives: List[int],
        page_table_keys: List[object],
    ) -> Tuple[Dict[int, int], int]:
        if len(page_table_keys) != len(page_representatives):
            raise ValueError("page-table keys must match the representative count")
        use_pool_ids = all(
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] == "pool"
            and isinstance(key[1], int)
            and key[1] >= 0
            for key in page_table_keys
        )
        unique_slots = []
        key_to_slot = {}
        representative_slots = {}
        for representative, key in zip(page_representatives, page_table_keys):
            slot = key_to_slot.get(key)
            if slot is None:
                slot = int(key[1]) if use_pool_ids else len(key_to_slot)
                key_to_slot[key] = slot
                unique_slots.append(slot)
            representative_slots[representative] = slot
        slot_count = max(unique_slots, default=-1) + 1
        return representative_slots, slot_count

    def __init__(
        self,
        layer_pools: List[torch.Tensor],
        dense_groups: List[List[int]],
        dense_layers: List[int],
        page_representatives: List[int],
        max_requests: int,
        seq_len: int,
        num_q_heads: int,
        num_freqs: int,
        q_real: torch.Tensor,
        q_imag: torch.Tensor,
        mlr_coef: torch.Tensor,
        freq_scale_sq: torch.Tensor,
        offsets: torch.Tensor,
        omega: torch.Tensor,
        page_table_keys: Optional[List[object]] = None,
        num_page_table_slots: Optional[int] = None,
        decode_width: Optional[int] = None,
        page_table_token_capacity: Optional[int] = None,
        draft_layer_pools: Optional[List[torch.Tensor]] = None,
        draft_page_representatives: Optional[List[int]] = None,
        draft_page_table_keys: Optional[List[object]] = None,
        draft_num_page_table_slots: Optional[int] = None,
        draft_page_table_token_capacity: Optional[int] = None,
    ) -> None:
        from .triattention_kernels import _FixedScoreGroup

        if not dense_groups or not dense_layers or not page_representatives or max_requests <= 0:
            raise ValueError("fixed score metadata requires non-empty positive geometry")
        grouped_layers = [layer for layers in dense_groups for layer in layers]
        if (
            len(grouped_layers) != len(dense_layers)
            or len(set(grouped_layers)) != len(grouped_layers)
            or len(set(dense_layers)) != len(dense_layers)
            or set(dense_layers) != set(grouped_layers)
        ):
            raise ValueError("dense layer order must cover every grouped layer exactly once")
        self.device = layer_pools[page_representatives[0]].device
        if self.device.type != "cuda":
            raise ValueError("fixed score metadata is CUDA-only")
        self.max_requests = max_requests
        self.bucket_seq_len = seq_len
        if page_table_token_capacity is None:
            page_table_token_capacity = seq_len
        if page_table_token_capacity < seq_len:
            raise ValueError("page-table capacity cannot be smaller than the score bucket")
        self.page_table_token_capacity = int(page_table_token_capacity)
        # Decode-width capacity of the score buffers; per-request prompt
        # lengths are staged runtime metadata. Default: the whole sequence
        # capacity is scorable.
        if decode_width is None:
            decode_width = int(seq_len)
        if decode_width <= 0 or decode_width > seq_len:
            raise ValueError("fixed score decode width exceeds the sequence capacity")
        self.decode_width = int(decode_width)
        q_real = q_real.to(device=self.device, dtype=torch.float32).contiguous()
        q_imag = q_imag.to(device=self.device, dtype=torch.float32).contiguous()
        mlr_coef = mlr_coef.to(device=self.device, dtype=torch.float32).contiguous()
        freq_scale_sq = freq_scale_sq.to(device=self.device, dtype=torch.float32).contiguous()
        offsets = offsets.to(device=self.device, dtype=torch.float32).contiguous()
        omega = omega.to(device=self.device, dtype=torch.float32).contiguous()
        if page_table_keys is None:
            page_table_keys = list(range(len(page_representatives)))
        self.representative_slots, minimum_page_table_slots = self._page_table_slot_layout(
            page_representatives, page_table_keys
        )
        if num_page_table_slots is None:
            num_page_table_slots = minimum_page_table_slots
        if num_page_table_slots < minimum_page_table_slots:
            raise ValueError("page-table slot capacity does not cover every V2 pool")
        tokens_per_block = int(layer_pools[page_representatives[0]].shape[3])
        if int(layer_pools[page_representatives[0]].shape[1]) != 2:
            raise ValueError("fixed score metadata requires an interleaved K/V pool")
        self.page_count = (
            self.page_table_token_capacity + tokens_per_block - 1
        ) // tokens_per_block
        self.copy_block_count = (self.page_count + 3) // 4 * 4
        if any(
            (self.page_table_token_capacity + int(layer_pools[layer].shape[3]) - 1)
            // int(layer_pools[layer].shape[3])
            != self.page_count
            for layer in page_representatives
        ):
            raise ValueError("fixed score metadata requires a uniform page count")
        device_page_shape = (
            num_page_table_slots,
            max_requests,
            2,
            self.copy_block_count,
        )
        # One table carries every per-round host value: three metadata rows
        # (logical position, valid length, prompt length) plus one move-offsets
        # row per compacted cache family, so each round pays exactly one
        # host-to-device copy. Offsets rows have request_capacity + 1 entries,
        # hence the extra column.
        self.request_metadata_host = torch.empty(
            (6, max_requests + 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self._bulk_copy_idx_src = torch.arange(
            max_requests,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self._bulk_offsets_src = torch.empty(
            device_page_shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.block_offsets_device = torch.empty(
            device_page_shape,
            dtype=torch.int32,
            device=self.device,
        )
        # Optional second block-offset staging plane for a co-compressed draft
        # KV cache. The draft is never scored: these offsets feed only the
        # draft compact launches.
        self.draft_block_offsets_device: Optional[torch.Tensor] = None
        self._draft_bulk_offsets_src: Optional[torch.Tensor] = None
        self.draft_representative_slots: Dict[int, int] = {}
        self.draft_copy_block_count = 0
        self.draft_page_table_token_capacity = 0
        if draft_layer_pools is not None:
            if (
                not draft_page_representatives
                or draft_page_table_keys is None
                or draft_page_table_token_capacity is None
                or draft_page_table_token_capacity <= 0
            ):
                raise ValueError(
                    "draft page-table staging requires representatives, keys, and capacity"
                )
            (
                self.draft_representative_slots,
                minimum_draft_slots,
            ) = self._page_table_slot_layout(draft_page_representatives, draft_page_table_keys)
            if draft_num_page_table_slots is None:
                draft_num_page_table_slots = minimum_draft_slots
            if draft_num_page_table_slots < minimum_draft_slots:
                raise ValueError("draft page-table slot capacity does not cover every V2 pool")
            draft_tokens_per_block = int(draft_layer_pools[draft_page_representatives[0]].shape[3])
            if int(draft_layer_pools[draft_page_representatives[0]].shape[1]) != 2:
                raise ValueError("draft page-table staging requires an interleaved K/V pool")
            self.draft_page_table_token_capacity = int(draft_page_table_token_capacity)
            draft_capacity = self.draft_page_table_token_capacity
            draft_page_count = (
                draft_capacity + draft_tokens_per_block - 1
            ) // draft_tokens_per_block
            if any(
                (draft_capacity + int(draft_layer_pools[layer].shape[3]) - 1)
                // int(draft_layer_pools[layer].shape[3])
                != draft_page_count
                for layer in draft_page_representatives
            ):
                raise ValueError("draft page-table staging requires a uniform page count")
            self.draft_copy_block_count = (draft_page_count + 3) // 4 * 4
            draft_page_shape = (
                draft_num_page_table_slots,
                max_requests,
                2,
                self.draft_copy_block_count,
            )
            self._draft_bulk_offsets_src = torch.empty(
                draft_page_shape,
                dtype=torch.int32,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            self.draft_block_offsets_device = torch.empty(
                draft_page_shape,
                dtype=torch.int32,
                device=self.device,
            )
        self.request_metadata_device = torch.empty(
            (6, max_requests + 1), dtype=torch.int32, device=self.device
        )
        self.round_starts_device = self.request_metadata_device[0, :max_requests]
        self.valid_seq_lens_device = self.request_metadata_device[1, :max_requests]
        # Per-request pinned prompt lengths: the score kernel starts each
        # request's decode window here, so one bucket may mix prompt lengths.
        self.token_starts_device = self.request_metadata_device[2, :max_requests]
        # Per-family move offsets consumed by the compaction pack kernel and
        # the C++ compact launches; refreshed with the metadata each round.
        self.dense_move_offsets = self.request_metadata_device[3]
        self.swa_move_offsets = self.request_metadata_device[4]
        self.draft_move_offsets = self.request_metadata_device[5]
        self.mean_cos = torch.empty(
            (max_requests, num_freqs), dtype=torch.float32, device=self.device
        )
        self.mean_sin = torch.empty_like(self.mean_cos)
        self.offsets = offsets
        self.omega = omega
        # ONE fused group across ALL dense layers: segments carry their own
        # layer base address and page-table slot, so distinct per-layer
        # storages/block tables no longer force one launch per storage group.
        _rep_of = {layer: layers[0] for layers in dense_groups for layer in layers}
        _page_table_slots = [self.representative_slots[_rep_of[layer]] for layer in dense_layers]
        self.fused_group = _FixedScoreGroup(
            layer_pools,
            dense_layers,
            max_requests,
            self.page_count,
            seq_len,
            num_q_heads,
            self.block_offsets_device,
            _page_table_slots,
            q_real,
            q_imag,
            mlr_coef,
            freq_scale_sq,
            omega,
            offsets,
            output_width=decode_width,
        )
        # Compile the optional SM100 CuTe score specialization here, outside
        # any CUDA graph capture (compilation allocates and synchronizes).
        # Default off: without TRTLLM_TRIATTENTION_CUTE_SCORE=1 this is a
        # no-op and scoring stays on the compiled C++ score ops.
        self.fused_group.prepare_cute_score(self.mean_cos, self.mean_sin)
        self.copy_done = torch.cuda.Event()
        # First record publishes constructor allocations to the V2 copy stream;
        # later records protect pinned metadata before the next cohort reuses it.
        self.copy_done.record(torch.cuda.current_stream(self.device))
        self.bulk_copy_done = torch.cuda.Event()
        self.bulk_consume_done = torch.cuda.Event()
        self.copy_pending = False
        self.page_tables_active = False
        self.stream = None
        self._score_valid_widths: Optional[torch.Tensor] = None
        self._score_aggregation: Optional[str] = None

    def bind_score_launcher(self, valid_widths: torch.Tensor, score_aggregation: str) -> None:
        """Bind the per-row score widths and aggregation for these buffers."""
        if self._score_aggregation is not None:
            raise RuntimeError("TriAttention score launcher is already bound")
        if score_aggregation not in ("mean", "max"):
            raise ValueError(f"unsupported score aggregation: {score_aggregation}")
        self._score_valid_widths = valid_widths
        self._score_aggregation = score_aggregation

    def launch_prepared_score(self) -> torch.Tensor:
        """Launch the phase and score kernels over these buffers."""
        from .triattention_kernels import prepare_mean_phase

        if self._score_aggregation is None:
            raise RuntimeError("TriAttention score launcher is not bound")
        stream = torch.cuda.current_stream(self.device)
        if self.stream is None:
            self.stream = stream
        elif (stream.device, stream.cuda_stream) != (
            self.stream.device,
            self.stream.cuda_stream,
        ):
            raise _FixedScoreStreamMismatch(
                "TriAttention score launches must stay on the staging CUDA stream"
            )
        if self._score_aggregation == "mean" and self.fused_group._cute_score_runner is not None:
            # mean_cos/mean_sin feed ONLY the opt-in CuTe score runner, whose
            # compiled kernel captured their device pointers, so they must be
            # refreshed before it launches. The default C++ mean path rotates
            # init-time phase tables inside the score kernels' own CTA
            # prologue instead, so production rounds launch zero phase or
            # coefficient kernels.
            prepare_mean_phase(
                self.round_starts_device,
                self.offsets,
                self.omega,
                self.mean_cos,
                self.mean_sin,
                self.max_requests,
            )
        return self.fused_group.launch(
            self.max_requests,
            self.valid_seq_lens_device,
            self._score_valid_widths,
            self.round_starts_device,
            self.token_starts_device,
            self.mean_cos,
            self.mean_sin,
            self._score_aggregation,
        )

    def stage(
        self,
        manager: KVCacheManagerV2,
        request_ids: List[int],
        round_starts: List[int],
        token_starts: List[int],
        seq_lens: Optional[List[int]] = None,
        page_table_seq_lens: Optional[List[int]] = None,
        draft_manager: Optional[KVCacheManagerV2] = None,
        dense_move_offsets: Optional[List[int]] = None,
        swa_move_offsets: Optional[List[int]] = None,
        draft_move_offsets: Optional[List[int]] = None,
    ) -> bool:
        """Copy one eviction cohort into reusable device buffers.

        ``token_starts`` carries each request's pinned prompt length; the
        score kernel starts that request's decode window there, so the cohort
        may mix prompt lengths.
        """
        request_count = len(request_ids)
        if (
            request_count == 0
            or request_count > self.max_requests
            or len(round_starts) != request_count
            or len(token_starts) != request_count
        ):
            return False
        if (draft_manager is None) != (self.draft_block_offsets_device is None):
            return False
        stream = torch.cuda.current_stream(self.device)
        if self.stream is None:
            self.stream = stream
        elif (stream.device, stream.cuda_stream) != (
            self.stream.device,
            self.stream.cuda_stream,
        ):
            raise _FixedScoreStreamMismatch(
                "TriAttention fixed score metadata is bound to its first CUDA stream"
            )
        if self.page_tables_active:
            raise RuntimeError("previous page-table cohort is still active")
        if seq_lens is None:
            seq_lens = [self.bucket_seq_len] * request_count
        if page_table_seq_lens is None:
            page_table_seq_lens = seq_lens
        if len(seq_lens) != request_count or len(page_table_seq_lens) != request_count:
            return False
        if manager.enable_swa_scratch_reuse:
            raise RuntimeError("TriAttention does not support V2 SWA scratch page-table remapping")
        try:
            request_metadata = torch.as_tensor(
                (round_starts, seq_lens, token_starts), dtype=torch.int32
            )
        except (OverflowError, RuntimeError, TypeError, ValueError):
            return False
        if not self._stage_page_tables_bulk(
            manager,
            request_ids,
            stream,
            self._bulk_offsets_src,
            self.block_offsets_device,
            self.copy_block_count,
        ):
            return False
        if draft_manager is not None:
            if draft_manager.enable_swa_scratch_reuse:
                raise RuntimeError(
                    "TriAttention does not support V2 SWA scratch page-table remapping"
                )
            assert self._draft_bulk_offsets_src is not None
            assert self.draft_block_offsets_device is not None
            if not self._stage_page_tables_bulk(
                draft_manager,
                request_ids,
                stream,
                self._draft_bulk_offsets_src,
                self.draft_block_offsets_device,
                self.draft_copy_block_count,
            ):
                return False
        self.request_metadata_host[:3, :request_count].copy_(request_metadata)
        # Rows past this cohort are padding: zero lengths keep the score
        # kernel and selection inert for them.
        self.request_metadata_host[:3, request_count:].zero_()
        # This round's per-family move offsets ride the same table, so the
        # single device copy below carries them too.
        for row, family_offsets in (
            (3, dense_move_offsets),
            (4, swa_move_offsets),
            (5, draft_move_offsets),
        ):
            if family_offsets is not None:
                self.request_metadata_host[row, : len(family_offsets)].copy_(
                    torch.as_tensor(family_offsets, dtype=torch.int32)
                )
        try:
            # Copy the fixed backing once. Only the first ``request_count``
            # columns are consumed by this cohort.
            self.request_metadata_device.copy_(self.request_metadata_host, non_blocking=True)
        finally:
            # Guard the pinned metadata until its asynchronous copies complete.
            # Page-table device-buffer reuse is guarded separately after compact.
            self.copy_done.record(stream)
            self.copy_pending = True
        self.page_tables_active = True
        return True

    def _stage_page_tables_bulk(
        self,
        manager: KVCacheManagerV2,
        request_ids: List[int],
        current_stream: torch.cuda.Stream,
        source: torch.Tensor,
        destination: torch.Tensor,
        copy_block_count: int,
    ) -> bool:
        """Copy one request group's V2 block offsets before live compaction.

        Uses the V2 block-offset kernel with an immutable pinned snapshot of the
        selected host-table rows. The snapshot is required because
        this method enqueues asynchronous host-memory reads; TriAttention later
        resizes the same cache, which mutates the manager's table in place.
        The IndexMapper synchronously resolves request slots and gathers only
        their beam-0 K block offsets, decoupling both live inputs before the
        native asynchronous copy consumes the snapshot with identity indices.
        ``dst[pool, r, 0(K), :]`` holds ``base_page * index_scales``. Score and
        compact decode that K plane inline, avoiding any conversion kernel.
        """
        if not request_ids or len(request_ids) > self.max_requests:
            return False

        host_table = manager.host_kv_cache_block_offsets
        num_pools, _, kv_planes, max_blocks = host_table.shape
        if (
            host_table.dtype != torch.int32
            or kv_planes != 2
            or copy_block_count > max_blocks
            or int(manager.kv_factor) != 2
            or num_pools != destination.shape[0]
        ):
            return False
        request_count = len(request_ids)
        submitted = False
        try:
            if self.copy_pending and not self.copy_done.query():
                self.copy_done.synchronize()
            # The native device copy reads only K and derives V with kv_offset.
            manager.index_mapper.gather_k_block_offsets(
                host_table,
                source,
                request_ids,
                copy_block_count,
            )
            manager._stream.wait_event(self.copy_done)
            copy_batch_block_offsets_to_device(
                source,
                destination,
                self._bulk_copy_idx_src[:request_count],
                manager.index_scales,
                manager.kv_offset,
                manager._stream.cuda_stream,
            )
            submitted = True
            self.bulk_copy_done.record(manager._stream)
            current_stream.wait_event(self.bulk_copy_done)
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            if submitted:
                raise RuntimeError(
                    "TriAttention bulk page-table copy failed after GPU submission"
                ) from exc
            logger.warning(f"TriAttention bulk page-table staging failed: {exc}")
            return False
        return True

    def mark_page_tables_consumed(self, *manager_streams: torch.cuda.Stream) -> None:
        """Order V2 page-table reuse and resize after this cohort's compact.

        Every passed manager stream (target, and the draft when co-compressed)
        waits on one event recorded after the compact launches, so neither
        cache can free or reallocate pages this cohort is still reading.
        """
        if not self.page_tables_active:
            raise RuntimeError("TriAttention page tables were not staged")
        self.bulk_consume_done.record(torch.cuda.current_stream(self.device))
        for manager_stream in manager_streams:
            manager_stream.wait_event(self.bulk_consume_done)
        self.page_tables_active = False


@dataclass(frozen=True, kw_only=True, slots=True)
class _PreparedEviction:
    """Request metadata validated before score, select, and compact."""

    request: "LlmRequest"
    request_id: int
    seq_len: int
    round_start: int
    prompt_len: int
    expected_keep_count: int
    protected_tail: int


@dataclass(kw_only=True, slots=True)
class _RequestCompressionState:
    """Mutable compression state owned by one live request."""

    generation_steps: int = 0
    evicted_tokens: int = 0
    confirmed_kv_length: Optional[int] = None


@dataclass(kw_only=True, slots=True)
class _PreparedGenerationBatch:
    """Target growth reserved by the most recently prepared generation batch."""

    batch: "ScheduledRequests"
    growth_by_request: Dict[int, int]


@dataclass(kw_only=True, slots=True)
class _EvictionBuffers:
    """Reusable fixed score and selection buffers for one runtime shape."""

    score_staging: _FixedScoreStagingBuffers
    keep_set_selector: Union[_BatchedUnionKeepSetSelector, _BatchedPerHeadKeepSetSelector]


class TriAttention(BaseKVCacheCompressionManager):
    """Periodic physical KV eviction driven by trigonometric importance scoring.

    Overrides ``on_generation_step_end``: every ``beta`` confirmed generation tokens it
    reads the cached keys through the ``KVCacheManagerV2``, scores each token
    with offline-calibrated stats, and physically evicts the tokens below the
    keep set. Full-attention layers are scored; kernel-masked SWA layers preserve
    their latest window in the same compacted prefix. Every layer ends with the
    same request-wide cached length.
    """

    adjusts_generation_kv_length = True

    def __init__(
        self,
        kv_cache_manager: KVCacheManagerV2,
        top_B: int,
        draft_kv_cache_manager: Optional[KVCacheManagerV2] = None,
        beta: int = 128,
        model_path: Optional[str] = None,
        calibration_path: Optional[str] = None,
        offset_max_length: int = 65536,
        score_aggregation: str = "mean",
        eviction_mode: str = "union",
        normalize_scores: bool = True,
        pin_prefill: bool = True,
        count_prompt_tokens: bool = False,
    ):
        super().__init__(kv_cache_manager, draft_kv_cache_manager)
        self.top_B = top_B
        self.beta = beta
        if self.top_B <= 0 or self.beta <= 0:
            raise ValueError("TriAttention top_B and beta must both be positive")
        # Which token set each eviction round keeps. The user-facing meaning of
        # each mode is documented on TriAttentionKvCacheCompressionConfig
        # (llm_args); implementation notes live above the selection helpers.
        self.eviction_mode = eviction_mode
        if self.eviction_mode not in ("union", "per_head", "per_layer_perhead"):
            raise ValueError(
                f"Unknown eviction_mode {self.eviction_mode!r}; expected one of "
                "'union', 'per_head', 'per_layer_perhead'"
            )
        self.normalize_scores = bool(normalize_scores)
        self.pin_prefill = bool(pin_prefill)
        # cpt=False (default): budget counts DECODE tokens only (pinned prompt is
        # extra). cpt=True: budget INCLUDES the pinned prompt.
        self.count_prompt_tokens = bool(count_prompt_tokens)
        if not self.pin_prefill or self.count_prompt_tokens:
            raise ValueError(
                "TriAttention physical KV reclaim requires pin_prefill=True and "
                "count_prompt_tokens=False so finalized prompt KV is preserved"
            )
        # All physical moves use the C++ V2 compaction operation.
        # No other compaction path exists.
        self.score_aggregation = score_aggregation
        # Calibration is the OFFICIAL TriAttention .pt (passed via
        # calibration_path), resolved + converted on the first request
        # (on_request_init). TRT-LLM does NOT compute calibration; model_path is
        # used for RoPE tables and local layer_types/sliding_window metadata.
        self.model_path = model_path
        if self.model_path is None:
            raise ValueError(
                "TriAttention requires model_path so kernel-masked "
                "sliding-attention layers can be classified safely"
            )
        self.calibration_path = calibration_path
        self.calibration: Optional[Dict[str, torch.Tensor]] = None
        self._calibrated = False
        # Calibration-derived dims + stats, filled in on_request_init.
        self._H: Optional[int] = None
        self._F: Optional[int] = None
        self._freq_scale_sq: Optional[torch.Tensor] = None

        # Geometric integration offsets (built lazily on first eviction so the
        # device matches the cache pool).
        self._offset_max_length = offset_max_length
        self._offsets: Optional[torch.Tensor] = None

        # Request presence records successful initialization. The record also
        # owns the counters and physical length cleared at request finish.
        self._request_states: Dict[int, _RequestCompressionState] = {}
        # The overlap executor prepares B(n) before finalizing B(n-1). Keep the
        # exact fixed-linear generation width for that currently in-flight batch;
        # the final hook treats those slots as an opaque suffix.
        self._prepared_generation_batch: Optional[_PreparedGenerationBatch] = None
        # Eviction buffers are built once at the first eviction, sized to
        # capacity bounds, and reused for the manager's lifetime.
        self._eviction_resources: Optional[_EvictionBuffers] = None
        self._eviction_pool_fingerprint: Optional[tuple] = None
        self._batched_compaction = None
        self._local_to_global_layers_cache: Optional[List[int]] = None
        self._attention_layer_partition_cache: Optional[
            Tuple[List[int], List[int], Optional[int]]
        ] = None
        self._runtime_kv_layout_cache: Optional[_RuntimeKVLayout] = None
        self._draft_runtime_kv_layout_cache: Optional[_RuntimeKVLayout] = None

    def on_request_init(self, request: "LlmRequest", **kwargs) -> None:
        """Mark capacity-only decode and resolve calibration once.

        Loads the user-supplied OFFICIAL calibration .pt and converts it to our
        runtime schema (see _resolve_calibration). TRT-LLM does not calibrate.
        """
        request_id = request.py_request_id
        if request_id not in self._request_states:
            self._validate_v2_compatibility()
            self._validate_request_capacity(request)
            num_layers = self._num_layers_from_manager()
            self._attention_layer_partition(num_layers)
            self._request_states[request_id] = _RequestCompressionState()
        self._ensure_calibrated()

    def _validate_request_capacity(self, request: "LlmRequest") -> None:
        """Require enough target page-table capacity to reach first eviction."""
        manager = self.kv_cache_manager
        # V2 mirrors the resolved speculative draft length (0 without spec).
        speculative_overshoot = int(manager.max_draft_len)
        first_eviction_decode_length = (
            self.top_B // self.beta + 1
        ) * self.beta + speculative_overshoot
        decode_capacity = min(int(request.py_max_new_tokens), first_eviction_decode_length)
        confirmed_capacity = int(request.py_prompt_len) + decode_capacity
        protected_tail_capacity = self._configured_protected_tail_capacity()
        required_capacity = confirmed_capacity + protected_tail_capacity
        pool_confirmed_capacity = manager.get_num_available_tokens(
            token_num_upper_bound=confirmed_capacity,
            max_num_draft_tokens=int(manager._kv_reserve_draft_tokens) + 1,
        )
        table_capacity = manager.max_blocks_per_seq * manager.tokens_per_block
        if confirmed_capacity > pool_confirmed_capacity or required_capacity > table_capacity:
            raise ValueError(
                "TriAttention target KV capacity is too small to reach its first "
                f"eviction: request requires {required_capacity} tokens "
                f"(prompt={request.py_prompt_len}, budget={self.top_B}, "
                f"beta={self.beta}, decode before eviction or completion="
                f"{decode_capacity}, speculative overshoot="
                f"{speculative_overshoot}, protected tail="
                f"{protected_tail_capacity}), "
                f"but the V2 pool covers {pool_confirmed_capacity + protected_tail_capacity} "
                f"tokens and its page table covers {table_capacity} tokens"
            )
        draft_manager = self.draft_kv_cache_manager
        if draft_manager is None:
            return
        draft_protected_tail = self._draft_protected_tail_capacity()
        draft_required_capacity = confirmed_capacity + draft_protected_tail
        draft_pool_capacity = draft_manager.get_num_available_tokens(
            token_num_upper_bound=confirmed_capacity,
            max_num_draft_tokens=int(draft_manager._kv_reserve_draft_tokens) + 1,
        )
        draft_table_capacity = draft_manager.max_blocks_per_seq * draft_manager.tokens_per_block
        if (
            confirmed_capacity > draft_pool_capacity
            or draft_required_capacity > draft_table_capacity
        ):
            raise ValueError(
                "TriAttention draft KV capacity is too small to reach the first "
                f"co-compression: request requires {draft_required_capacity} "
                f"tokens (prompt={request.py_prompt_len}, budget={self.top_B}, "
                f"beta={self.beta}, decode before eviction or completion="
                f"{decode_capacity}, draft protected tail={draft_protected_tail}), "
                f"but the draft V2 pool covers "
                f"{draft_pool_capacity + draft_protected_tail} tokens and its "
                f"page table covers {draft_table_capacity} tokens"
            )

    def _draft_protected_tail_capacity(self) -> int:
        """Return the draft tail moved and re-reserved by every co-compression."""
        draft_manager = self.draft_kv_cache_manager
        capacity = (
            int(draft_manager.num_extra_kv_tokens) + int(draft_manager._kv_reserve_draft_tokens) + 1
        )
        if capacity <= 0:
            raise RuntimeError("draft KVCacheManagerV2 exposes an invalid protected-tail capacity")
        return capacity

    def _ensure_calibrated(self) -> None:
        """Resolve calibration once for the first request."""
        if self._calibrated:
            return
        self.calibration = self._resolve_calibration()
        self._H = int(self.calibration["E_q"].shape[1])
        self._F = int(self.calibration["E_q"].shape[2])
        # Squared per-frequency RoPE scaling factor (required calibration key).
        self._freq_scale_sq = self.calibration["freq_scale_sq"].to(dtype=torch.float32)
        # Pre-split query stats + MLR coefficient for the score kernel so
        # it doesn't recompute (E_q_norm - |E_q|) per call. Shapes [L, H, F].
        _Eq = self.calibration["E_q"]
        self._triattn_q_real = _Eq.real.to(torch.float32).contiguous()
        self._triattn_q_imag = _Eq.imag.to(torch.float32).contiguous()
        self._triattn_mlr_coef = (
            self.calibration["E_q_norm"].to(torch.float32) - _Eq.abs().to(torch.float32)
        ).contiguous()
        self._calibrated = True

    def _validate_v2_compatibility(self) -> None:
        """Reject runtime modes outside the V2 physical-compaction contract."""
        manager = self.kv_cache_manager
        if not isinstance(manager, KVCacheManagerV2):
            raise ValueError("TriAttention physical eviction requires KVCacheManagerV2")
        if manager.kv_factor != 2:
            raise ValueError(
                "TriAttention requires a standard key/value KV cache; "
                "MLA/SELFKONLY caches are not supported"
            )
        if manager.mapping.enable_attention_dp:
            raise ValueError("TriAttention does not support attention DP")
        if manager.is_disagg:
            raise ValueError("TriAttention does not support disaggregated serving")
        if manager.max_beam_width != 1:
            raise ValueError("TriAttention requires beam-width-one decoding")
        # Speculative feature gates (resolved draft length, linear drafting,
        # mode whitelist) run in the factory, where spec_config lives. The
        # draft cache itself is validated here whenever one is attached.
        draft_manager = self.draft_kv_cache_manager
        if draft_manager is not None:
            if not draft_manager.is_draft:
                raise ValueError(
                    "TriAttention speculative compatibility requires the actual "
                    "separate draft KV cache manager"
                )
            if draft_manager.kv_factor != 2:
                raise ValueError(
                    "TriAttention compresses the draft KV cache together with "
                    "the target, so the draft cache must be a standard "
                    "key/value cache"
                )
            if self.eviction_mode != "union":
                raise ValueError(
                    "TriAttention draft KV co-compression supports only "
                    "eviction_mode='union'; per-head keep sets are not defined "
                    "for draft layers, which are never scored"
                )
            if any(window is not None for window in draft_manager.max_attention_window_vec) or any(
                not isinstance(layer, AttentionLayerConfig) or layer.sliding_window_size is not None
                for layer in draft_manager.kv_cache_manager_py_config.layers
            ):
                raise ValueError(
                    "TriAttention draft KV co-compression requires full-attention "
                    "draft V2 lifecycles"
                )
        if any(window is not None for window in manager.max_attention_window_vec) or any(
            not isinstance(layer, AttentionLayerConfig) or layer.sliding_window_size is not None
            for layer in manager.kv_cache_manager_py_config.layers
        ):
            raise ValueError(
                "TriAttention requires full-attention V2 lifecycles; native SWA, "
                "VSWA, and SSM pools are not supported"
            )

    # The framework drives all request-lifecycle hooks. TriAttention resolves
    # calibration on request init, evicts periodically at generation-step end,
    # and removes per-request state at finish. It scores from offline
    # calibration, not from live queries or attention scores, so it needs no
    # per-layer attention hook: the whole eviction runs once per period in
    # on_generation_step_end, which loops the layers and reads each layer's keys
    # straight from the KV pool.

    def on_generation_step_end(self, scheduled_batch: "ScheduledRequests", **kwargs) -> None:
        """Compact after native KV-cache updates have finalized this iteration.

        The compression manager is ordered after KVCacheManagerV2, so capacity
        already reflects the written token and any rewind. The overlap scheduler
        may already have enqueued the next forward; CUDA stream ordering keeps
        compaction after that reader. The resize happens only after compaction;
        it detaches the compacted tail without blocking the host, while V2's
        per-slot finish events prevent early page reuse.
        """
        with nvtx_range_debug("triattention.generation_step_end", color="blue"):
            self._periodic_evict(scheduled_batch)

    def prepare_resources(self, scheduled_batch: "ScheduledRequests") -> None:
        """Snapshot fixed-linear target growth; mutation remains in final update."""
        super().prepare_resources(scheduled_batch)
        generation_growth = {}
        for request in scheduled_batch.generation_requests:
            request_id = request.py_request_id
            growth = 1 + max(
                get_draft_token_length(request),
                self.kv_cache_manager._kv_reserve_draft_tokens,
            )
            generation_growth[request_id] = growth
        self._prepared_generation_batch = _PreparedGenerationBatch(
            batch=scheduled_batch,
            growth_by_request=generation_growth,
        )

    def _inflight_generation_growth(
        self, scheduled_batch: "ScheduledRequests", request_id: int
    ) -> int:
        """Return exact newer target allocation width under overlap scheduling."""
        prepared = self._prepared_generation_batch
        if prepared is None or scheduled_batch is prepared.batch:
            return 0
        return prepared.growth_by_request.get(request_id, 0)

    def _periodic_evict(
        self,
        scheduled_batch: "ScheduledRequests",
    ) -> None:
        """Count confirmed tokens; every ``beta`` tokens score the cache
        and physically evict to the pinned prompt plus top-B decode tokens."""
        gen_requests = scheduled_batch.generation_requests
        if not gen_requests:
            return
        mgr = self.kv_cache_manager
        resolved_requests = []
        for request in gen_requests:
            if request.is_dummy or request.state in (
                LlmRequestState.GENERATION_COMPLETE,
                LlmRequestState.CONTEXT_INIT,
            ):
                continue
            request_id = request.py_request_id
            kv_cache = mgr.kv_cache_map.get(request_id)
            if kv_cache is None:
                continue
            if not kv_cache.is_active:
                raise RuntimeError(
                    "TriAttention cannot finalize a suspended target KV cache; "
                    f"request {request_id} must be resumed before "
                    "the final update hook"
                )
            if request_id not in self._request_states:
                self.on_request_init(request)
            resolved_requests.append((request, request_id, kv_cache))
        if not resolved_requests or not self._calibrated:
            return
        protected_tails: Dict[int, int] = {}
        due_requests = []

        # Resolve every active target cache before changing cadence state. The
        # captured cache objects also avoid repeating the V2 map lookup here.
        for request, request_id, kv_cache in resolved_requests:
            raw_capacity = int(kv_cache.capacity)
            # One-engine speculative decoding keeps a fixed reserve E. Under
            # overlap, B(n) is allocated/enqueued before finalizing B(n-1), so
            # its exact scheduler growth Q is also opaque. Both spans are
            # contiguous after the stable target prefix and move byte-for-byte.
            protected_tail = int(mgr.num_extra_kv_tokens) + self._inflight_generation_growth(
                scheduled_batch, request_id
            )
            seq_len = raw_capacity - protected_tail
            if seq_len < 0 or protected_tail < 0:
                raise RuntimeError(
                    f"Request {request_id} has an inconsistent protected target tail: "
                    f"confirmed={seq_len}, capacity={raw_capacity}, "
                    f"protected_tail={protected_tail}"
                )
            if seq_len < kv_cache.history_length:
                raise RuntimeError(
                    f"Request {request_id} KV length {seq_len} is below finalized "
                    f"history {kv_cache.history_length}"
                )
            request_state = self._request_states[request_id]
            request_state.confirmed_kv_length = seq_len
            previous_step = request_state.generation_steps
            confirmed_delta = 1 + int(request.py_num_accepted_draft_tokens)
            step = previous_step + confirmed_delta
            request_state.generation_steps = step
            if previous_step // self.beta >= step // self.beta:
                continue
            if seq_len <= self._minimum_evictable_length(request, seq_len):
                continue
            if self.draft_kv_cache_manager is not None:
                draft_kv_cache = self.draft_kv_cache_manager.kv_cache_map.get(request_id)
                if draft_kv_cache is None or not draft_kv_cache.is_active:
                    raise RuntimeError(
                        "TriAttention cannot co-compress a missing or "
                        f"suspended draft KV cache; request {request_id} must "
                        "be resumed before the final update hook"
                    )
            protected_tails[request_id] = protected_tail
            due_requests.append((request, request_id))

        # (2) Compact all affected dense and kernel-masked SWA layers, then release
        # the unreachable tail directly through V2's public resize primitive.
        # Prompt lengths and tails are per-request metadata, so the whole due
        # cohort runs as one batched round (the workspace holds max_batch_size
        # requests, which bounds any generation batch).
        if not due_requests:
            return
        num_layers = self._num_layers_from_manager()
        with nvtx_range_debug("triattention.evict_request_group", color="purple"):
            capacity_targets = self._evict_requests(
                due_requests,
                num_layers,
                protected_tail_lengths=protected_tails,
            )
        self._resize_compacted_requests(capacity_targets, protected_tails)

    def _resize_compacted_requests(self, capacity_targets, protected_tails) -> None:
        if not capacity_targets:
            return
        mgr = self.kv_cache_manager
        draft_manager = self.draft_kv_cache_manager
        with nvtx_range("triattention.resize", color="red"):
            with nvtx_range_debug("triattention.v2_resize", color="red"):
                for rid, target_capacity in capacity_targets:
                    kv_cache = mgr.kv_cache_map.get(rid)
                    if kv_cache is None or not kv_cache.is_active:
                        continue
                    if target_capacity > kv_cache.capacity:
                        raise RuntimeError(
                            f"Request {rid} compacted capacity {target_capacity} exceeds "
                            f"current capacity {kv_cache.capacity}"
                        )
                    protected_tail = protected_tails[rid]
                    resized_capacity = target_capacity + protected_tail
                    if not kv_cache.resize(resized_capacity, None):
                        raise RuntimeError(
                            f"Failed to resize compacted KV cache for request {rid} "
                            f"to {resized_capacity} tokens"
                        )
                if draft_manager is not None:
                    # The draft cache was compacted with the same kept token
                    # set, so it shrinks to the same retained length plus its
                    # own protected tail.
                    draft_protected_tail = self._draft_protected_tail_capacity()
                    for rid, target_capacity in capacity_targets:
                        draft_kv_cache = draft_manager.kv_cache_map.get(rid)
                        if draft_kv_cache is None or not draft_kv_cache.is_active:
                            continue
                        draft_capacity = target_capacity + draft_protected_tail
                        if not draft_kv_cache.resize(draft_capacity, None):
                            raise RuntimeError(
                                "Failed to resize co-compressed draft KV cache "
                                f"for request {rid} to {draft_capacity} tokens"
                            )

    def _minimum_evictable_length(self, request: "LlmRequest", seq_len: int) -> int:
        """Return the largest cache length for which selection is an identity.

        With a decode-only budget, pinned prompt tokens do not consume ``top_B``.
        Selection therefore keeps every token until the cache exceeds
        ``prompt_len + top_B``. The constructor guarantees the decode-only
        budget (``pin_prefill=True``, ``count_prompt_tokens=False``).
        """
        prompt_len = min(int(request.py_prompt_len), seq_len)
        return prompt_len + self.top_B

    def _local_score_calibration(
        self,
        num_layers: int,
        global_layers: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return calibration tensors indexed in this PP rank's local layer order."""
        if global_layers and max(global_layers) >= self._triattn_q_real.shape[0]:
            raise ValueError(
                f"TriAttention calibration has {self._triattn_q_real.shape[0]} layers, "
                f"but this PP rank references global layer {max(global_layers)}"
            )
        if global_layers == list(range(global_layers[0], global_layers[0] + num_layers)):
            layer_slice = slice(global_layers[0], global_layers[0] + num_layers)
            return (
                self._triattn_q_real[layer_slice],
                self._triattn_q_imag[layer_slice],
                self._triattn_mlr_coef[layer_slice],
            )
        layer_ids = torch.as_tensor(
            global_layers,
            device=self._triattn_q_real.device,
            dtype=torch.long,
        )
        return (
            self._triattn_q_real.index_select(0, layer_ids),
            self._triattn_q_imag.index_select(0, layer_ids),
            self._triattn_mlr_coef.index_select(0, layer_ids),
        )

    def _configured_protected_tail_capacity(self) -> int:
        """Return the largest target tail reserved by the native V2 lifecycle."""
        capacity = (
            int(self.kv_cache_manager.num_extra_kv_tokens)
            + int(self.kv_cache_manager._kv_reserve_draft_tokens)
            + 1
        )
        if capacity <= 0:
            raise RuntimeError("KVCacheManagerV2 exposes an invalid protected-tail capacity")
        return capacity

    @staticmethod
    def _build_cross_request_keep_set_selector(
        plan: _CrossRequestSelectionPlan,
        *,
        input_scores: Optional[torch.Tensor] = None,
        normalize_scores: bool = True,
        prompt_offsets_buffer: Optional[torch.Tensor] = None,
    ) -> Union[_BatchedUnionKeepSetSelector, _BatchedPerHeadKeepSetSelector]:
        """Allocate one fixed ``[request, ...]`` keep-set selector."""
        if plan.eviction_mode == "union":
            return _BatchedUnionKeepSetSelector(
                plan.rows,
                plan.width,
                plan.keep_count,
                dtype=plan.dtype,
                device=plan.device,
                max_requests=plan.max_requests,
                dense_layers=plan.dense_layers,
                num_query_heads=plan.num_query_heads,
                num_kv_heads=plan.num_kv_heads,
                input_scores=input_scores,
                normalize_scores=normalize_scores,
                prompt_offsets_buffer=prompt_offsets_buffer,
            )
        return _BatchedPerHeadKeepSetSelector(
            eviction_mode=plan.eviction_mode,
            dense_layers=plan.dense_layers,
            num_query_heads=plan.num_query_heads,
            num_kv_heads=plan.num_kv_heads,
            width=plan.width,
            keep_count=plan.keep_count,
            dtype=plan.dtype,
            device=plan.device,
            max_requests=plan.max_requests,
            prompt_offsets_buffer=prompt_offsets_buffer,
        )

    def on_request_finish(self, request: "LlmRequest", **kwargs) -> None:
        """Drop this request's per-request length and eviction state."""
        request_id = request.py_request_id
        self._request_states.pop(request_id, None)
        prepared = self._prepared_generation_batch
        if prepared is not None:
            prepared.growth_by_request.pop(request_id, None)
        # The workspace stays resident across idle periods: its memory is a
        # deliberate one-time cost and rebuilding it per burst would reintroduce
        # allocation on the decode hot path.

    # ================================================================== #
    # Helpers (eviction / scoring / V2 cache access / calibration)       #
    # ================================================================== #

    # --- Upstream-faithful eviction modes (per_head / per_layer_perhead / union) ---
    #
    # These reproduce github.com/WeianMao/triattention's selection: scores are NOT
    # averaged over heads (each KV head keeps its own token set), they are
    # z-normalized per head over the decode region, the prompt (prefill) tokens are
    # pinned, and there is no recency window. The kept COUNT stays uniform (= top_B)
    # so paged attention
    # and the num_cached bookkeeping are unchanged; only the kept SET differs per
    # head. Kept K keeps its original RoPE rotation (scored post-RoPE), so a head
    # holding a different token set still scores the correct relative distance
    # and no per-head position tracking is needed.

    def _local_to_global_layers(self, num_layers: int) -> List[int]:
        """Return V2's global layer id for every local TriAttention layer slot."""
        cached = self._local_to_global_layers_cache
        if cached is not None:
            if len(cached) != num_layers:
                raise ValueError(
                    f"TriAttention layer count changed from {len(cached)} to {num_layers}"
                )
            return cached

        global_layers = [int(layer) for layer in self.kv_cache_manager.pp_layers]
        if len(global_layers) != num_layers:
            raise ValueError(
                f"KVCacheManagerV2 exposes {len(global_layers)} PP layers, "
                f"but TriAttention received {num_layers} local layers"
            )
        self._local_to_global_layers_cache = global_layers
        return global_layers

    @staticmethod
    def _has_sliding_window_signal(config: Dict[str, object]) -> bool:
        """Return whether config metadata hints at sliding attention."""
        use_sliding_window = config.get("use_sliding_window")
        if isinstance(use_sliding_window, bool):
            return use_sliding_window
        for field in (
            "sliding_window",
            "sliding_window_size",
            "sliding_window_pattern",
            "max_window_layers",
        ):
            value = config.get(field)
            if isinstance(value, bool):
                if value:
                    return True
            elif isinstance(value, (int, float)):
                if value > 0:
                    return True
            elif value:
                return True
        return False

    def _attention_layer_partition(
        self, num_layers: int
    ) -> Tuple[List[int], List[int], Optional[int]]:
        """Return dense layers, kernel-masked SWA layers, and the SWA window.

        TriAttention initialization has already rejected real V2 windowed
        lifecycles. A sliding layer found here is therefore stored at full length
        and applies its window only in the attention kernel.
        """
        cached = self._attention_layer_partition_cache
        if cached is not None:
            return cached

        model_path = self.model_path
        if model_path is None:
            raise ValueError("TriAttention requires model_path")

        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:
            raise ValueError(
                f"TriAttention could not load the local model config from {model_path!r}"
            ) from exc
        config_values = config.get_text_config().to_dict()
        layer_types = config_values.get("layer_types")
        if not layer_types:
            if self._has_sliding_window_signal(config_values):
                raise ValueError(
                    "Model config exposes sliding-window metadata but no layer_types; "
                    "TriAttention cannot classify kernel-masked SWA layers safely"
                )
            result = (list(range(num_layers)), [], None)
            self._attention_layer_partition_cache = result
            return result
        global_layers = self._local_to_global_layers(num_layers)
        if global_layers and max(global_layers) >= len(layer_types):
            raise ValueError(
                f"Model config has {len(layer_types)} layer_types entries, "
                f"but this PP rank references global layer {max(global_layers)}"
            )

        swa_layers = [
            local_layer
            for local_layer, global_layer in enumerate(global_layers)
            if "sliding" in str(layer_types[global_layer]).lower()
        ]
        swa_set = set(swa_layers)
        dense_layers = [layer for layer in range(num_layers) if layer not in swa_set]
        window_size = None
        if swa_layers:
            raw_window = config_values.get("sliding_window")
            if not isinstance(raw_window, int) or raw_window <= 0:
                raise ValueError(
                    "TriAttention requires a positive integer model sliding_window "
                    "when layer_types contains sliding attention"
                )
            if self.top_B < raw_window:
                raise ValueError(
                    f"TriAttention decode budget top_B={self.top_B} must be at least "
                    f"the kernel-masked SWA window size {raw_window}"
                )
            window_size = raw_window
        result = (dense_layers, swa_layers, window_size)
        self._attention_layer_partition_cache = result
        return result

    def _runtime_kv_layout(self, num_layers: int) -> _RuntimeKVLayout:
        """Return stable V2 pool views and layer groups for eviction.

        KVCacheManagerV2 keeps GPU virtual addresses and layer geometry stable,
        while opt-in pool rebalance can change the page dimension. Cache all
        layer views, then query the live page count for one representative per
        physical pool before reuse. This avoids rebuilding TensorWrapper views
        on every eviction while retaining the same fail-closed rebalance check.
        """
        cached = self._runtime_kv_layout_cache
        manager = self.kv_cache_manager
        if cached is not None:
            if cached.num_layers != num_layers:
                raise ValueError(
                    f"TriAttention layer count changed from {cached.num_layers} to {num_layers}"
                )
            if cached.manager is not manager:
                raise RuntimeError("TriAttention target KV cache manager changed at runtime")
            current_page_counts = self._pool_page_counts(
                manager,
                cached.global_layers,
                cached.pool_representatives,
            )
            if current_page_counts != cached.pool_page_counts:
                raise RuntimeError(
                    "TriAttention V2 pool layout changed after the layout was built; "
                    "KV pool rebalance is not supported"
                )
            return cached

        global_layers = self._local_to_global_layers(num_layers)
        dense_layers, swa_layers, swa_window = self._attention_layer_partition(num_layers)
        if not dense_layers:
            raise ValueError("TriAttention requires at least one full-attention layer")
        layout = self._build_runtime_kv_layout(
            manager,
            global_layers,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
            dense_storage_groups=self._dense_layer_pool_groups(dense_layers, global_layers),
            what="",
        )
        self._runtime_kv_layout_cache = layout
        return layout

    def _build_runtime_kv_layout(
        self,
        manager: KVCacheManagerV2,
        global_layers: List[int],
        *,
        dense_layers: List[int],
        swa_layers: List[int],
        swa_window: Optional[int],
        dense_storage_groups: Optional[Dict[object, List[int]]],
        what: str,
    ) -> _RuntimeKVLayout:
        """Build the manager-lifetime layer and pool views one eviction reads.

        ``dense_storage_groups`` restricts the compaction groups to the dense
        layers (target cache); None groups every layer (draft cache, which has
        no SWA partition). ``what`` prefixes error messages ("" or "draft ").
        """
        num_layers = len(global_layers)
        maybe_layer_pools = [manager.get_buffers(layer, kv_layout="HND") for layer in global_layers]
        if any(pool is None for pool in maybe_layer_pools):
            missing = [
                layer for layer, pool in zip(global_layers, maybe_layer_pools) if pool is None
            ]
            raise RuntimeError(f"Missing {what}KV pools for attention layers {missing}")
        layer_pools = [pool for pool in maybe_layer_pools if pool is not None]
        all_layers = list(range(num_layers))
        layer_pool_keys = tuple(
            self._page_table_pool_keys(all_layers, global_layers, manager=manager)
        )
        all_storage_groups: Dict[object, List[int]] = {}
        for layer, pool_key in zip(all_layers, layer_pool_keys):
            all_storage_groups.setdefault(pool_key, []).append(layer)
        storage_groups = (
            dense_storage_groups if dense_storage_groups is not None else all_storage_groups
        )
        layer_group_representative = {
            layer: layers[0] for layers in storage_groups.values() for layer in layers
        }
        pool_representatives = tuple(layers[0] for layers in all_storage_groups.values())
        return _RuntimeKVLayout(
            manager=manager,
            num_layers=num_layers,
            global_layers=global_layers,
            layer_pools=layer_pools,
            dense_layers=dense_layers,
            swa_layers=swa_layers,
            swa_window=swa_window,
            storage_groups=storage_groups,
            layer_group_representative=layer_group_representative,
            layer_pool_keys=layer_pool_keys,
            pool_representatives=pool_representatives,
            pool_page_counts=tuple(
                int(layer_pools[layer].shape[0]) for layer in pool_representatives
            ),
            pool_view_fingerprint=self._pool_view_fingerprint(
                [layer_pools[layer] for layer in pool_representatives]
            ),
        )

    def _draft_runtime_kv_layout(self) -> _RuntimeKVLayout:
        """Return stable draft V2 pool views, mirroring ``_runtime_kv_layout``.

        The draft cache is compacted with the target's kept token set, so its
        layout has no scoring role: every draft layer is dense and there is no
        SWA partition.
        """
        manager = self.draft_kv_cache_manager
        if manager is None:
            raise RuntimeError("TriAttention has no draft KV cache manager to lay out")
        cached = self._draft_runtime_kv_layout_cache
        if cached is not None:
            if cached.manager is not manager:
                raise RuntimeError("TriAttention draft KV cache manager changed at runtime")
            current_page_counts = self._pool_page_counts(
                manager,
                cached.global_layers,
                cached.pool_representatives,
            )
            if current_page_counts != cached.pool_page_counts:
                raise RuntimeError(
                    "TriAttention draft V2 pool layout changed after the layout "
                    "was built; KV pool rebalance is not supported"
                )
            return cached

        global_layers = [int(layer) for layer in manager.pp_layers]
        if not global_layers:
            raise RuntimeError("TriAttention draft KV cache manager exposes no layers")
        layout = self._build_runtime_kv_layout(
            manager,
            global_layers,
            dense_layers=list(range(len(global_layers))),
            swa_layers=[],
            swa_window=None,
            dense_storage_groups=None,
            what="draft ",
        )
        self._draft_runtime_kv_layout_cache = layout
        return layout

    @staticmethod
    def _pool_page_counts(
        manager: KVCacheManagerV2,
        global_layers: Sequence[int],
        pool_representatives: Sequence[int],
    ) -> Tuple[int, ...]:
        """Read the only pool-view dimension that V2 rebalance can change."""
        return tuple(
            int(
                manager.impl.get_page_index_upper_bound(
                    manager.layer_offsets[global_layers[layer]],
                    Role.KEY,
                )
            )
            // int(manager.kv_factor)
            for layer in pool_representatives
        )

    @staticmethod
    def _pool_view_fingerprint(pools: List[torch.Tensor]) -> Tuple[tuple, ...]:
        """Identify the V2 pool properties consumed by score and compact kernels."""
        return tuple(
            (
                pool.data_ptr(),
                tuple(int(value) for value in pool.shape),
                tuple(int(value) for value in pool.stride()),
                pool.dtype,
                pool.device,
            )
            for pool in pools
        )

    def _fixed_resources_for(
        self,
        layout: _RuntimeKVLayout,
        prepared: Sequence[_PreparedEviction],
    ) -> _EvictionBuffers:
        """Return the eviction buffers, building them once at first use.

        The request capacity follows the executor's max batch size (memory
        scales linearly with it) and the decode-width capacity follows the
        eviction bound (compaction keeps the scored decode region near
        ``top_B`` plus one period of growth), so one set of buffers serves
        every round. They are rebuilt only when the pool views change or a
        round outgrows them.
        """
        if not prepared:
            raise ValueError("TriAttention eviction requires at least one request")
        needed_width = max(item.seq_len - item.prompt_len for item in prepared)
        needed_page_tokens = max(item.seq_len + item.protected_tail for item in prepared)
        needed_requests = len(prepared)
        draft_fingerprint = None
        if self.draft_kv_cache_manager is not None:
            draft_layout = self._draft_runtime_kv_layout()
            draft_fingerprint = (
                draft_layout.pool_page_counts,
                draft_layout.pool_view_fingerprint,
            )
        fingerprint = (
            self.eviction_mode,
            self.top_B,
            tuple(layout.dense_layers),
            layout.pool_view_fingerprint,
            draft_fingerprint,
        )
        resources = self._eviction_resources
        if resources is not None:
            staging = resources.score_staging
            if (
                self._eviction_pool_fingerprint == fingerprint
                and needed_width <= staging.decode_width
                and needed_page_tokens <= staging.page_table_token_capacity
                and needed_requests <= staging.max_requests
            ):
                return resources
            # Pools changed or this round outgrew the buffers: rebuild.
            self._eviction_resources = None
            self._batched_compaction = None

        mgr = self.kv_cache_manager
        tail_capacity = self._configured_protected_tail_capacity()
        request_capacity = max(needed_requests, int(mgr.max_batch_size))
        decode_width = max(
            needed_width,
            self.top_B + 2 * self.beta + int(mgr.max_total_draft_tokens or 0),
        )
        seq_capacity = max(needed_page_tokens, int(mgr.max_seq_len))
        page_table_token_capacity = max(needed_page_tokens, seq_capacity + tail_capacity)

        dense_groups = list(layout.storage_groups.values())
        representatives = [group[0] for group in dense_groups]
        representatives.extend(layer for layer in layout.swa_layers if layer not in representatives)
        draft_kwargs = {}
        if self.draft_kv_cache_manager is not None:
            draft_layout = self._draft_runtime_kv_layout()
            draft_tail_capacity = self._draft_protected_tail_capacity()
            draft_representatives = list(draft_layout.pool_representatives)
            draft_kwargs = dict(
                draft_layer_pools=draft_layout.layer_pools,
                draft_page_representatives=draft_representatives,
                draft_page_table_keys=[
                    draft_layout.layer_pool_keys[layer] for layer in draft_representatives
                ],
                draft_num_page_table_slots=self.draft_kv_cache_manager.num_pools,
                draft_page_table_token_capacity=seq_capacity + draft_tail_capacity,
            )

        first_pool = layout.layer_pools[layout.dense_layers[0]]
        if self._offsets is None:
            self._offsets = _build_geometric_offsets(self._offset_max_length, first_pool.device)
        q_real, q_imag, mlr_coef = self._local_score_calibration(
            layout.num_layers, layout.global_layers
        )
        score_staging = _FixedScoreStagingBuffers(
            layout.layer_pools,
            dense_groups=dense_groups,
            dense_layers=layout.dense_layers,
            page_representatives=representatives,
            max_requests=request_capacity,
            seq_len=seq_capacity,
            num_q_heads=int(self._H),
            num_freqs=int(self._F),
            q_real=q_real,
            q_imag=q_imag,
            mlr_coef=mlr_coef,
            freq_scale_sq=self._freq_scale_sq,
            offsets=self._offsets,
            omega=self.calibration["omega"],
            page_table_keys=self._page_table_pool_keys(representatives, layout.global_layers),
            num_page_table_slots=layout.manager.num_pools,
            decode_width=decode_width,
            page_table_token_capacity=page_table_token_capacity,
            **draft_kwargs,
        )
        keep_set_selector = self._build_cross_request_keep_set_selector(
            _CrossRequestSelectionPlan(
                eviction_mode=self.eviction_mode,
                dense_layers=tuple(layout.dense_layers),
                num_query_heads=int(self._H),
                num_kv_heads=int(first_pool.shape[2]),
                rows=len(layout.dense_layers) * int(self._H),
                width=decode_width,
                keep_count=self.top_B,
                dtype=torch.float32,
                device=first_pool.device,
                max_requests=request_capacity,
            ),
            input_scores=score_staging.fused_group.output.view(
                request_capacity,
                len(layout.dense_layers) * int(self._H),
                decode_width,
            ),
            normalize_scores=self.normalize_scores,
            prompt_offsets_buffer=score_staging.token_starts_device,
        )
        # Padded rows carry zero valid width; their provisional TopK entries
        # must still be in-range ordinals for the finalizer's score gather.
        provisional = getattr(keep_set_selector, "final_indices", None)
        if provisional is None:
            provisional = keep_set_selector.top_indices_i32
        provisional.zero_()
        score_staging.bind_score_launcher(
            keep_set_selector.valid_widths,
            self.score_aggregation,
        )
        resources = _EvictionBuffers(
            score_staging=score_staging,
            keep_set_selector=keep_set_selector,
        )
        self._eviction_resources = resources
        self._eviction_pool_fingerprint = fingerprint
        return resources

    def _batched_compaction_for(
        self,
        *,
        layout: _RuntimeKVLayout,
        prepared: Sequence[_PreparedEviction],
        score_staging: _FixedScoreStagingBuffers,
        keep_set_selector: Union[_BatchedUnionKeepSetSelector, _BatchedPerHeadKeepSetSelector],
    ):
        """Build or reuse the C++ compaction launches for one cohort."""
        from .compaction import BatchedKVCacheCompaction

        if layout.swa_layers and layout.swa_window:
            # SWA landing positions are prompt-dependent; reject a request
            # whose retained span cannot cover the model window this round.
            for item in prepared:
                if item.prompt_len + self.top_B < int(layout.swa_window):
                    raise ValueError(
                        f"Request {item.request_id} retains "
                        f"{item.prompt_len + self.top_B} tokens, below the "
                        f"sliding window {layout.swa_window}"
                    )
        batched_compaction = self._batched_compaction
        if batched_compaction is None:
            draft_kwargs = {}
            if self.draft_kv_cache_manager is not None:
                draft_layout = self._draft_runtime_kv_layout()
                draft_kwargs = dict(
                    draft_layer_pools=draft_layout.layer_pools,
                    draft_layers=draft_layout.dense_layers,
                    draft_layer_group_representative=draft_layout.layer_group_representative,
                    draft_layer_pool_keys=list(draft_layout.layer_pool_keys),
                    draft_protected_tail_capacity=self._draft_protected_tail_capacity(),
                    draft_kv_block_offsets=score_staging.draft_block_offsets_device,
                    draft_page_table_slots=score_staging.draft_representative_slots,
                )
            batched_compaction = BatchedKVCacheCompaction(
                eviction_mode=self.eviction_mode,
                layer_pools=layout.layer_pools,
                dense_layers=layout.dense_layers,
                swa_layers=layout.swa_layers,
                layer_group_representative=layout.layer_group_representative,
                layer_pool_keys=list(layout.layer_pool_keys),
                kept_token_ordinals=keep_set_selector.keep,
                valid_sequence_lengths=score_staging.valid_seq_lens_device,
                kv_block_offsets=score_staging.block_offsets_device,
                page_table_slots=score_staging.representative_slots,
                request_count=score_staging.max_requests,
                prompt_offsets=score_staging.token_starts_device,
                decode_keep_count=self.top_B,
                swa_window=layout.swa_window,
                protected_tail_capacity=self._configured_protected_tail_capacity(),
                dense_move_offsets=score_staging.dense_move_offsets,
                swa_move_offsets=score_staging.swa_move_offsets,
                draft_move_offsets=score_staging.draft_move_offsets,
                **draft_kwargs,
            )
            # One launch settles the kept ordinals and packs the dense/SWA
            # move sources; the compaction keeps only its C++ moves (plus the
            # draft's own pack). Both caches are invalidated together, so the
            # fused packing always points at the live compaction buffers.
            keep_set_selector.fuse_move_source_pack(
                batched_compaction.hand_move_source_pack_to_selection()
            )
            self._batched_compaction = batched_compaction
        # Tails vary per round (in-flight growth), so the per-family move
        # offsets ride the staged metadata table each round.
        return batched_compaction

    def _move_offsets_for(
        self,
        layout: _RuntimeKVLayout,
        prepared: Sequence[_PreparedEviction],
        capacity: int,
    ) -> Tuple[List[int], Optional[List[int]], Optional[List[int]]]:
        """Build this round's per-family move offsets, padded to the capacity.

        Rows past the cohort repeat the final offset, so padded requests move
        nothing in the pack kernel and the C++ compact launches.
        """

        def padded_offsets(moves_per_request: List[int]) -> List[int]:
            offsets = [0]
            for moves in moves_per_request:
                offsets.append(offsets[-1] + moves)
            offsets.extend(offsets[-1:] * (capacity - len(moves_per_request)))
            return offsets

        tails = [item.protected_tail for item in prepared]
        dense = padded_offsets([self.top_B + tail for tail in tails])
        swa = None
        if layout.swa_layers and layout.swa_window:
            swa = padded_offsets([int(layout.swa_window) + tail for tail in tails])
        draft = None
        if self.draft_kv_cache_manager is not None:
            draft_tail = self._draft_protected_tail_capacity()
            draft = padded_offsets([self.top_B + draft_tail] * len(prepared))
        return dense, swa, draft

    def _page_table_pool_keys(
        self,
        representatives: List[int],
        global_layers: List[int],
        manager: Optional[KVCacheManagerV2] = None,
    ) -> List[object]:
        """Return stable V2-pool keys for the representative layers."""
        if manager is None:
            manager = self.kv_cache_manager
        layer_offsets = manager.layer_offsets
        layer_to_pool = manager.layer_to_pool_mapping_dict
        try:
            return [
                ("pool", int(layer_to_pool[layer_offsets[global_layers[layer]]]))
                for layer in representatives
            ]
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("KVCacheManagerV2 exposes an invalid layer-to-pool mapping") from exc

    def _dense_layer_pool_groups(
        self,
        dense_layers: List[int],
        global_layers: List[int],
    ) -> Dict[object, List[int]]:
        """Group layers that use the same V2 page table."""
        groups: Dict[object, List[int]] = {}
        for layer, pool_key in zip(
            dense_layers,
            self._page_table_pool_keys(dense_layers, global_layers),
        ):
            groups.setdefault(pool_key, []).append(layer)
        return groups

    def _attach_page_ids(
        self,
        prepared: Sequence[_PreparedEviction],
        staging: _FixedScoreStagingBuffers,
        layout: _RuntimeKVLayout,
    ) -> None:
        dense_offsets, swa_offsets, draft_offsets = self._move_offsets_for(
            layout, prepared, staging.max_requests
        )
        try:
            staged = staging.stage(
                self.kv_cache_manager,
                [item.request_id for item in prepared],
                [item.round_start for item in prepared],
                [item.prompt_len for item in prepared],
                [item.seq_len for item in prepared],
                [item.seq_len + item.protected_tail for item in prepared],
                draft_manager=self.draft_kv_cache_manager,
                dense_move_offsets=dense_offsets,
                swa_move_offsets=swa_offsets,
                draft_move_offsets=draft_offsets,
            )
        except _FixedScoreStreamMismatch:
            raise
        except Exception as exc:
            raise RuntimeError("TriAttention score staging failed") from exc
        if not staged:
            raise RuntimeError("TriAttention page-table staging rejected the cohort")

    def _evict_requests(
        self,
        evict_reqs,
        num_layers: int,
        protected_tail_lengths: Optional[Dict[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        """Score and compact requests, returning ``(request_id, capacity)`` targets.

        Only full-attention layers participate in scoring. For kernel-masked SWA
        layers, the latest model window is rebased to the tail of the common
        compacted prefix before the request-wide capacity is reduced.
        """
        if protected_tail_lengths is None:
            protected_tail_lengths = {}
        protected_tail_capacity = self._configured_protected_tail_capacity()
        with nvtx_range_debug("triattention.resolve_layout", color="blue"):
            layout = self._runtime_kv_layout(num_layers)

        # Resolve request length and page metadata before mutating any layer.
        prepared: List[_PreparedEviction] = []
        with nvtx_range("triattention.metadata", color="cyan"):
            for request, rid in evict_reqs:
                request_state = self._request_states.get(rid)
                seq_len = None if request_state is None else request_state.confirmed_kv_length
                if seq_len is None:
                    raise RuntimeError(f"Missing confirmed KV length for request {rid}")
                # Restore the uncompressed confirmed logical position from the
                # physical prefix and cumulative eviction count.
                round_start = seq_len + request_state.evicted_tokens
                minimum_evictable_length = self._minimum_evictable_length(request, seq_len)
                if seq_len <= minimum_evictable_length:
                    continue
                expected_keep_count = minimum_evictable_length
                protected_tail = int(protected_tail_lengths.get(rid, 0))
                if protected_tail < 0 or protected_tail > protected_tail_capacity:
                    raise RuntimeError(
                        f"Request {rid} protected tail {protected_tail} exceeds "
                        f"configured capacity {protected_tail_capacity}"
                    )
                prepared.append(
                    _PreparedEviction(
                        request=request,
                        request_id=rid,
                        seq_len=int(seq_len),
                        round_start=int(round_start),
                        prompt_len=min(int(request.py_prompt_len), int(seq_len)),
                        expected_keep_count=expected_keep_count,
                        protected_tail=protected_tail,
                    )
                )
        if not prepared:
            return []
        with nvtx_range_debug("triattention.staging_lookup", color="blue"):
            resources = self._fixed_resources_for(layout, prepared)
            score_staging = resources.score_staging
            keep_set_selector = resources.keep_set_selector
            batched_compaction = self._batched_compaction_for(
                layout=layout,
                prepared=prepared,
                score_staging=score_staging,
                keep_set_selector=keep_set_selector,
            )
        with nvtx_range_debug("triattention.page_table_stage", color="orange"):
            self._attach_page_ids(prepared, score_staging, layout)
            # The staged per-request prompt lengths are shared with the
            # selector; per-head modes re-expand them to selection rows here.
            keep_set_selector.refresh_row_prompt_offsets()

        try:
            with nvtx_range("triattention.score", color="blue"):
                per_head = score_staging.launch_prepared_score()
            with nvtx_range("triattention.select", color="yellow"):
                if isinstance(keep_set_selector, _BatchedUnionKeepSetSelector):
                    keep_set_selector.select_prepared_requests()
                else:
                    keep_set_selector.select_requests(
                        per_head,
                        normalize_scores=self.normalize_scores,
                    )
            with nvtx_range("triattention.compact", color="purple"):
                batched_compaction.compact()
        finally:
            consumer_streams = [self.kv_cache_manager._stream]
            if self.draft_kv_cache_manager is not None:
                consumer_streams.append(self.draft_kv_cache_manager._stream)
            score_staging.mark_page_tables_consumed(*consumer_streams)

        capacity_targets = []
        for item in prepared:
            keep_count = item.expected_keep_count
            evicted = item.seq_len - keep_count
            if evicted <= 0:
                raise RuntimeError("TriAttention attempted an identity compaction")
            request_state = self._request_states[item.request_id]
            request_state.evicted_tokens += evicted
            request_state.confirmed_kv_length = keep_count
            # Publish the cumulative count on the request: this is the
            # manager's only channel to the runtime. The model engine
            # reads it back where it builds num_cached_tokens_per_seq,
            # so the kernels see the compacted KV length next step.
            item.request.py_num_compressed_tokens = request_state.evicted_tokens
            capacity_targets.append((item.request_id, keep_count))
        return capacity_targets

    def _num_layers_from_manager(self) -> int:
        return len(self.kv_cache_manager.pp_layers)

    # ------------------------------------------------------------------ #
    # Helpers: calibration loading                                       #
    # ------------------------------------------------------------------ #

    def _resolve_calibration(self) -> Dict[str, torch.Tensor]:
        """Load the user-supplied calibration .pt and return our runtime schema.

        TriAttention does NOT compute calibration -- the user calibrates with the
        official tool (github.com/WeianMao/triattention) and passes that file via
        ``calibration_path``; we only run inference. Both the official R-KV layout
        (``{metadata, stats{"layerLL_headHH": {q_mean_real, q_mean_imag,
        q_abs_mean}}}``) and our already-converted flat layout are accepted -- the
        official one is converted here."""
        if self.calibration_path is None:
            raise ValueError(
                "TriAttention requires `calibration_path`: a calibration .pt from "
                "the official tool (github.com/WeianMao/triattention). TRT-LLM does "
                "not compute calibration -- see examples/ for the Qwen3-8B file and "
                "the official calibration instructions."
            )
        raw = torch.load(self.calibration_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and _REQUIRED_CALIBRATION_KEYS <= set(raw):
            calib = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in raw.items()}
            self._validate_calibration(calib)
            return calib
        if isinstance(raw, dict) and {"metadata", "stats"} <= set(raw):
            return self._convert_official_calibration(raw)
        got = sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
        raise ValueError(
            f"Unrecognized calibration at {self.calibration_path}: expected the "
            f"official {{metadata, stats}} layout or "
            f"{sorted(_REQUIRED_CALIBRATION_KEYS)}; got {got}."
        )

    def _convert_official_calibration(self, raw) -> Dict[str, torch.Tensor]:
        """Convert the official per-(layer, head) stats to our flat runtime schema.

        ``E_q[l,h] = q_mean_real + i*q_mean_imag`` and ``E_q_norm[l,h] =
        q_abs_mean`` are the same statistic, just restacked into ``[L, H, F]``.
        ``omega`` / ``freq_scale_sq`` are not in the official file (its runtime
        recomputes them from the model rotary), so we derive them from the model
        config -- model-intrinsic and corpus-independent."""
        stats = raw["stats"]
        meta = raw.get("metadata", {})
        if "sampled_heads" in meta:
            heads = [(int(a), int(b)) for a, b in meta["sampled_heads"]]
        else:
            heads = [
                (int(k[len("layer") : k.index("_head")]), int(k[k.index("_head") + len("_head") :]))
                for k in stats
            ]
        num_layers = max(layer for layer, _ in heads) + 1
        num_heads = max(h for _, h in heads) + 1
        freq_count = int(next(iter(stats.values()))["q_mean_real"].numel())
        E_q = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.complex64)
        E_q_norm = torch.zeros(num_layers, num_heads, freq_count, dtype=torch.float32)
        for layer, h in heads:
            s = stats[f"layer{layer:02d}_head{h:02d}"]
            E_q[layer, h] = torch.complex(s["q_mean_real"].float(), s["q_mean_imag"].float())
            E_q_norm[layer, h] = s["q_abs_mean"].float()
        omega, freq_scale_sq = self._rope_tables(freq_count)
        calib = {
            "E_q": E_q.to("cuda"),
            "E_q_norm": E_q_norm.to("cuda"),
            "omega": omega.to("cuda"),
            "freq_scale_sq": freq_scale_sq.to("cuda"),
        }
        self._validate_calibration(calib)
        logger.info(
            f"TriAttention: converted official calibration {self.calibration_path}"
            f" -> E_q[L={num_layers}, H={num_heads}, F={freq_count}]"
        )
        return calib

    def _rope_tables(self, freq_count: int):
        """RoPE ``omega`` (inv_freq) + ``freq_scale_sq`` (squared position-0
        amplitude) from the model config -- model-intrinsic, corpus-independent
        (the official file does not store them). transformers' rope-init handles
        plain and scaled RoPE; plain RoPE has attention_factor 1 so freq_scale_sq
        is all ones. Falls back to the analytic inv_freq if rope-init is absent."""
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True).get_text_config()
        config_values = cfg.to_dict()
        head_dim = freq_count * 2
        base = float(config_values.get("rope_theta", 10000.0))
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            scaling = config_values.get("rope_scaling") or {}
            rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
            inv_freq, attention_factor = ROPE_INIT_FUNCTIONS[rope_type](cfg, device="cpu")
            omega = inv_freq.to(torch.float32)[:freq_count].clone()
            scale_sq = float(attention_factor) ** 2
        except Exception:
            idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
            omega = (1.0 / (base ** (idx / head_dim)))[:freq_count].clone()
            scale_sq = 1.0
        return omega, torch.full((freq_count,), scale_sq, dtype=torch.float32)

    def _validate_calibration(self, calibration: Dict[str, torch.Tensor]) -> None:
        """Verify the calibration dict has the expected keys."""
        missing = _REQUIRED_CALIBRATION_KEYS - set(calibration.keys())
        if missing:
            raise ValueError(
                f"TriAttention calibration is missing keys: {sorted(missing)}; "
                f"got {sorted(calibration.keys())}."
            )
