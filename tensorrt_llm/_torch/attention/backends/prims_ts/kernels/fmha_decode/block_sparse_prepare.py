# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare exact-first sparse routes for the PrimTS FMHA route consumer.

The BSR frontend is shared by continuous exact/proxy and paged exact storage.
It validates each canonical row once and emits the same logical exact records;
a paged row resolves each atom's page locator in the pass that emits its
record, and a continuous row then appends its proxy records. The bitmask
frontend shares the record geometry and emitters but remains limited to
continuous storage. Proxy suffixes contain one stable record per summary
group; a fully exact group remains present with zero score words. One warp owns
one sparse row and four warps share a CTA.

Each lane owns one exact atom, so a warp pass emits
``32 / logical_origins_per_route`` complete exact records. A proxy record owns
an aligned group of ``max(logical_origins_per_route, token_words_per_route)``
lanes, so a proxy pass emits 32 divided by that group size. Route masks and
flags reduce lane-group ballots. A pass stages its consecutive records in
shared memory and copies them to the workspace with coalesced stores that skip
record padding. A BSR lane reads its atom's block from ``block_indices``; the
bitmask frontend lists the selected blocks of 32 exact words at a time in a
per-warp shared-memory list, and each pass gathers its atoms' blocks from the
list, moving on to later batches until the one holding its last block.

``row_route_offsets`` is a separate plan-owned immutable Int32 tensor.
``route_workspace`` contains only mutable row counts and route metadata
described by ``_BlockSparseRouteLayout``. Payload outside each live row
count is intentionally stale. ``max_blocks_per_row`` is the plan-declared
semantic BSR-block limit, which remains distinct from packed-route capacity.

A prepare launched with ``use_pdl`` is a programmatic dependent of the grid
launched immediately before it on the same stream, a Sage plan's scale
prepare, whose output the routes never read. Its CTAs may start while that
grid drains; each acquires the grid after its last store, so the prepare's
completion implies the producer's completion and memory visibility for the
attention launch that follows in stream order and reads both.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_drv
from cutlass.cute.testing import assert_ as runtime_assert
from cutlass.experimental import primitives as prims
from cutlass.utils.smem_allocator import SmemAllocator

from ..._block_sparse.prepared import (
    _PREPARED_ROUTE_IS_FULL_FLAG,
    _PREPARED_ROUTE_IS_PROXY_FLAG,
    _BlockSparseRouteLayout,
)
from .fmha_decode_resources.helpers_common import _warp_broadcast_i32
from .block_sparse_inspect import _validate_bsr_row_lane


_WARPS_PER_CTA = 4
_WARP_SIZE = 32
_THREADS_PER_CTA = _WARPS_PER_CTA * _WARP_SIZE
# One bitmask batch loads a word per lane, so it selects at most 32 * 32 blocks.
_BITMASK_BATCH_BLOCKS = _WARP_SIZE * _WARP_SIZE


@dataclass(frozen=True)
class _RouteConfig:
    """Compile-time route, warp-pass, and shared-memory geometry plus policy flags.

    It is shared across sparse input and storage modes.
    """

    num_pattern_heads: int
    num_q_blocks: int
    num_kv_blocks: int
    num_exact_words: int
    num_proxy_groups: int
    num_rows: int
    seq_len_kv: int
    kv_block_size: int
    atom_size: int
    atoms_per_block: int
    logical_origins_per_route: int
    token_words_per_route: int
    atom_valid_mask_word_offset: int
    route_flags_word_offset: int
    token_words_word_offset: int
    # ``None`` selects the contiguous record, which has no page-ID words.
    physical_page_ids_word_offset: int | None
    stores_score_words: bool
    apply_token_mask: bool
    use_proxy_routes: bool
    route_metadata_stride_words: int
    route_metadata_base_word_offset: int

    @staticmethod
    def create(
        *,
        layout: _BlockSparseRouteLayout,
        num_pattern_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        q_block_size: int,
        kv_block_size: int,
        apply_token_mask: bool,
        use_proxy_routes: bool,
    ) -> "_RouteConfig":
        """Build storage-independent route geometry and policy flags."""

        stores_score_words = layout.token_words_word_offset is not None
        if apply_token_mask and not stores_score_words:
            raise ValueError("token masking requires prepared score words")
        if use_proxy_routes and not stores_score_words:
            raise ValueError("proxy routes require prepared score words")
        num_q_blocks = (seq_len_q + q_block_size - 1) // q_block_size
        num_kv_blocks = (seq_len_kv + kv_block_size - 1) // kv_block_size
        cfg = _RouteConfig(
            num_pattern_heads=num_pattern_heads,
            num_q_blocks=num_q_blocks,
            num_kv_blocks=num_kv_blocks,
            num_exact_words=(num_kv_blocks + _WARP_SIZE - 1) // _WARP_SIZE,
            num_proxy_groups=(num_kv_blocks + layout.kv_route_size - 1)
            // layout.kv_route_size,
            num_rows=layout.num_rows,
            seq_len_kv=seq_len_kv,
            kv_block_size=kv_block_size,
            atom_size=layout.atom_size,
            atoms_per_block=kv_block_size // layout.atom_size,
            logical_origins_per_route=layout.logical_origins_per_route,
            token_words_per_route=layout.token_words_per_route,
            atom_valid_mask_word_offset=layout.atom_valid_mask_word_offset,
            route_flags_word_offset=layout.route_flags_word_offset,
            token_words_word_offset=(
                layout.token_words_word_offset
                if layout.token_words_word_offset is not None
                else 0
            ),
            physical_page_ids_word_offset=(
                layout.physical_page_ids_word_offset if layout.is_paged else None
            ),
            stores_score_words=stores_score_words,
            apply_token_mask=apply_token_mask,
            use_proxy_routes=use_proxy_routes,
            route_metadata_stride_words=layout.route_metadata_stride_words,
            route_metadata_base_word_offset=layout.route_metadata_base_word_offset,
        )
        # Exact passes give each route origin a lane; proxy passes give each
        # origin or score-word slot a lane. The layout's power-of-two route and
        # atom sizes make both lane groups tile a warp.
        assert _WARP_SIZE % cfg.proxy_lanes_per_route == 0
        return cfg

    @property
    def exact_routes_per_pass(self) -> int:
        """Exact records emitted by one warp pass holding one atom per lane."""

        return _WARP_SIZE // self.logical_origins_per_route

    @property
    def proxy_lanes_per_route(self) -> int:
        """Lanes owning one proxy record's origin and score-word slots."""

        return max(self.logical_origins_per_route, self.token_words_per_route)

    @property
    def proxy_routes_per_pass(self) -> int:
        """Proxy records emitted by one warp pass; never more than exact ones."""

        return _WARP_SIZE // self.proxy_lanes_per_route

    @property
    def route_record_words(self) -> int:
        """Words of one record that carry data; the rest of its stride is padding."""

        return (
            self.token_words_word_offset + self.token_words_per_route
            if self.stores_score_words
            else self.route_flags_word_offset + 1
        )

    @property
    def record_stage_words(self) -> int:
        """Shared-memory words one warp uses to stage a pass of records."""

        return self.exact_routes_per_pass * self.route_metadata_stride_words

    @property
    def selected_block_list_words(self) -> int:
        """Shared-memory words of one warp's compacted bitmask batch."""

        return min(_BITMASK_BATCH_BLOCKS, self.num_exact_words * _WARP_SIZE)


def _positive_i32_ceil_div(
    value: cutlass.Int32,
    divisor: cutlass.Constexpr[int],
) -> cutlass.Int32:
    """Ceil-divide a positive Int32 without overflowing its upper bound."""

    return (value - cutlass.Int32(1)) // cutlass.Int32(divisor) + cutlass.Int32(1)


@cute.jit
def _prepared_route_counts(
    selected_block_count: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """Return exact atoms, exact routes, and total prepared routes for one row."""

    exact_atom_count = selected_block_count * cutlass.Int32(cfg.atoms_per_block)
    exact_route_count = (
        exact_atom_count + cutlass.Int32(cfg.logical_origins_per_route - 1)
    ) // cutlass.Int32(cfg.logical_origins_per_route)
    total_route_count = exact_route_count
    if cutlass.const_expr(cfg.use_proxy_routes):
        total_route_count += cutlass.Int32(cfg.num_proxy_groups)
    return exact_atom_count, exact_route_count, total_route_count


@cute.jit
def _load_row_route_span(
    row_route_offsets: cute.Tensor,
    linear_row_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    row_is_valid: cutlass.Boolean,
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Load one row's plan-owned prepared-route span into lane zero.

    The span depends only on the row, so kernels issue this load before the
    row's routing metadata to overlap the two global-memory round trips.
    """

    row_route_begin = cutlass.Int32(0)
    row_route_end = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0) and row_is_valid:
        row_route_begin = cutlass.Int32(row_route_offsets[linear_row_idx])
        row_route_end = cutlass.Int32(row_route_offsets[linear_row_idx + 1])
    return row_route_begin, row_route_end


@cute.jit
def _prepared_row_route_begin(
    row_route_begin: cutlass.Int32,
    row_route_end: cutlass.Int32,
    lane_idx: cutlass.Int32,
    row_is_valid: cutlass.Boolean,
    total_route_count: cutlass.Int32,
) -> cutlass.Int32:
    """Validate one row's loaded route span and broadcast its first route."""

    if lane_idx == cutlass.Int32(0) and row_is_valid:
        row_capacity = row_route_end - row_route_begin
        runtime_assert(
            row_route_begin >= cutlass.Int32(0)
            and row_capacity >= cutlass.Int32(0)
            and total_route_count <= row_capacity,
            "prepared routes exceed planned row capacity",
        )
    return _warp_broadcast_i32(row_route_begin, 0)


@cute.jit
def _route_metadata_word_index(
    row_route_begin: cutlass.Int32,
    route_idx: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> cutlass.Int32:
    """Return the first workspace word of one row-relative route record."""

    return cutlass.Int32(cfg.route_metadata_base_word_offset) + (
        row_route_begin + route_idx
    ) * cutlass.Int32(cfg.route_metadata_stride_words)


@cute.jit
def _store_staged_records(
    route_workspace: cute.Tensor,
    record_stage: cute.Tensor,
    first_record_word_index: cutlass.Int32,
    live_record_count: cutlass.Int32,
    lane_idx: cutlass.Int32,
    records_per_pass: cutlass.Constexpr[int],
    cfg: cutlass.Constexpr[_RouteConfig],
) -> None:
    """Copy one pass of staged consecutive records to the workspace.

    Every store instruction covers 32 consecutive workspace words. Record
    padding and records at or beyond ``live_record_count`` are skipped, so
    exactly the data words of the live records are written.
    """

    cute.arch.sync_warp()
    stride = cfg.route_metadata_stride_words
    live_word_count = cutlass.Int32(records_per_pass * stride)
    if live_record_count < cutlass.Int32(records_per_pass):
        live_word_count = live_record_count * cutlass.Int32(stride)
    for step in cutlass.range_constexpr(
        (records_per_pass * stride + _WARP_SIZE - 1) // _WARP_SIZE
    ):
        word_idx = cutlass.Int32(step * _WARP_SIZE) + lane_idx
        word_in_record = word_idx % cutlass.Int32(stride)
        if word_idx < live_word_count and word_in_record < cutlass.Int32(
            cfg.route_record_words
        ):
            route_workspace[first_record_word_index + word_idx] = record_stage[word_idx]
    # The next pass restages only after every lane has copied this one.
    cute.arch.sync_warp()


@cute.jit
def _lane_group_bits(
    ballot: cutlass.Int32,
    lane_idx: cutlass.Int32,
    lanes_per_group: cutlass.Constexpr[int],
) -> cutlass.Uint32:
    """Return the ballot bits of the aligned lane group containing this lane."""

    group_bits = ballot.bitcast(cutlass.Uint32)
    if cutlass.const_expr(lanes_per_group < _WARP_SIZE):
        first_group_lane = lane_idx - lane_idx % cutlass.Int32(lanes_per_group)
        group_bits = (group_bits >> first_group_lane) & cutlass.Uint32(
            (1 << lanes_per_group) - 1
        )
    return group_bits


@cute.jit
def _lane_group_all(
    predicate: cutlass.Boolean,
    lane_idx: cutlass.Int32,
    lanes_per_group: cutlass.Constexpr[int],
) -> cutlass.Boolean:
    """Return whether ``predicate`` holds on every lane of this lane's group."""

    return cutlass.Boolean(
        _lane_group_bits(
            cute.arch.vote_ballot_sync(predicate), lane_idx, lanes_per_group
        )
        == cutlass.Uint32((1 << lanes_per_group) - 1)
    )


@cute.jit
def _resolve_route_logical_atom_origin(
    block_indices: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    flat_atom_idx: cutlass.Int32,
    seq_len_kv: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> tuple[cutlass.Int32, cutlass.Boolean]:
    """Resolve a BSR row's ``flat_atom_idx``-th exact atom to its KV-token origin."""

    bsr_entry_offset = flat_atom_idx // cutlass.Int32(cfg.atoms_per_block)
    atom_in_block = flat_atom_idx % cutlass.Int32(cfg.atoms_per_block)
    valid = cutlass.Boolean(bsr_entry_offset < row_end - row_begin)
    logical_origin = cutlass.Int32(-1)
    if valid:
        block_idx = cutlass.Int32(block_indices[row_begin + bsr_entry_offset])
        block_origin = block_idx * cutlass.Int32(cfg.kv_block_size)
        atom_offset = atom_in_block * cutlass.Int32(cfg.atom_size)
        valid = cutlass.Boolean(atom_offset < seq_len_kv - block_origin)
        if valid:
            logical_origin = block_origin + atom_offset
    return logical_origin, valid


@cute.jit
def _low_bits_mask(valid_bits: cutlass.Int32) -> cutlass.Uint32:
    """Return a Uint32 mask with its lowest clamped bit count set."""

    mask = cutlass.Uint32(0)
    if valid_bits >= cutlass.Int32(_WARP_SIZE):
        mask = cutlass.Uint32(0xFFFFFFFF)
    elif valid_bits > cutlass.Int32(0):
        mask = (cutlass.Uint32(1) << valid_bits) - cutlass.Uint32(1)
    return mask


@cute.jit
def _load_atom_score_word(
    kv_valid_bits: cute.Tensor,
    logical_origin: cutlass.Int32,
    word_in_atom: cutlass.Constexpr[int],
    batch_idx: cutlass.Int32,
    seq_len_kv: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> cutlass.Uint32:
    """Build the score bits of one atom's ``word_in_atom``-th 32-token word.

    An atom of at most 32 tokens has one word, returned in the low
    ``atom_size`` bits. Tokens at or past ``seq_len_kv`` and, with caller-token
    masking, masked tokens have no bit; an invalid atom (origin ``-1``) has
    none.
    """

    word_origin = logical_origin + cutlass.Int32(word_in_atom * _WARP_SIZE)
    token_word = cutlass.Uint32(0)
    if logical_origin >= cutlass.Int32(0) and word_origin < seq_len_kv:
        token_word = _low_bits_mask(seq_len_kv - word_origin) & cutlass.Uint32(
            (1 << min(cfg.atom_size, _WARP_SIZE)) - 1
        )
        if cutlass.const_expr(cfg.apply_token_mask):
            # Atoms start at multiples of their size, so only an atom narrower
            # than a word sits at a nonzero bit offset of its source word.
            bit_offset = word_origin & cutlass.Int32(
                _WARP_SIZE - min(cfg.atom_size, _WARP_SIZE)
            )
            source_word = cutlass.Uint32(
                kv_valid_bits[batch_idx, word_origin >> cutlass.Int32(5)]
            )
            token_word = token_word & (source_word >> bit_offset)
    return token_word


@cute.jit
def _resolve_prepared_bsr_row(
    block_indptr: cute.Tensor,
    block_indices: cute.Tensor,
    linear_row_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    row_is_valid: cutlass.Boolean,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """Resolve one trusted canonical runtime BSR row."""

    row_begin = cutlass.Int32(0)
    row_end = cutlass.Int32(0)
    batch_idx = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0) and row_is_valid:
        q_block_row_idx = linear_row_idx % cfg.num_q_blocks
        linear_batch_head_idx = linear_row_idx // cfg.num_q_blocks
        pattern_head_idx = linear_batch_head_idx % cfg.num_pattern_heads
        batch_idx = linear_batch_head_idx // cfg.num_pattern_heads
        row_begin = cutlass.Int32(
            block_indptr[batch_idx, pattern_head_idx, q_block_row_idx]
        )
        row_end = cutlass.Int32(
            block_indptr[batch_idx, pattern_head_idx, q_block_row_idx + 1]
        )
        num_indices = cutlass.Int32(cute.size(block_indices))
        runtime_assert(
            row_begin >= cutlass.Int32(0)
            and row_begin <= row_end
            and row_end <= num_indices,
            "block_indptr row must be bounded and monotone",
        )
    row_begin = _warp_broadcast_i32(row_begin, 0)
    row_end = _warp_broadcast_i32(row_end, 0)
    batch_idx = _warp_broadcast_i32(batch_idx, 0)

    row_error_code = cutlass.Int32(0)
    if row_is_valid:
        row_error_code = _validate_bsr_row_lane(
            block_indices,
            row_begin,
            row_end,
            lane_idx,
            cfg.num_kv_blocks,
        )
    row_error_code = cutlass.Int32(cute.arch.warp_redux_sync(row_error_code, "max"))
    if lane_idx == cutlass.Int32(0) and row_is_valid:
        runtime_assert(
            row_error_code == cutlass.Int32(0),
            "block_indices row must be canonical and in range",
        )
    return row_begin, row_end, batch_idx


@cute.jit
def _emit_exact_route_pass(
    route_workspace: cute.Tensor,
    record_stage: cute.Tensor,
    kv_valid_bits: cute.Tensor,
    row_route_begin: cutlass.Int32,
    first_route_idx: cutlass.Int32,
    exact_route_count: cutlass.Int32,
    lane_idx: cutlass.Int32,
    logical_origin: cutlass.Int32,
    physical_page_id: cutlass.Int32,
    batch_idx: cutlass.Int32,
    seq_len_kv: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> None:
    """Emit the exact records whose atoms the warp holds, one atom per lane.

    Lane ``i`` holds atom ``i % logical_origins_per_route`` of route
    ``first_route_idx + i // logical_origins_per_route``; an invalid atom has
    origin ``-1``. Records at or beyond ``exact_route_count`` are not stored.
    """

    origins_per_route = cfg.logical_origins_per_route
    atom_in_route = lane_idx % cutlass.Int32(origins_per_route)
    route_in_pass = lane_idx // cutlass.Int32(origins_per_route)
    record_word_index = route_in_pass * cutlass.Int32(cfg.route_metadata_stride_words)
    record_stage[record_word_index + atom_in_route] = logical_origin
    if cutlass.const_expr(cfg.physical_page_ids_word_offset is not None):
        record_stage[
            record_word_index
            + cutlass.Int32(cfg.physical_page_ids_word_offset)
            + atom_in_route
        ] = physical_page_id

    # A record is full when every atom is structurally full and every score
    # word it stores is full, so each lane folds its own words into its atom.
    atom_is_valid = cutlass.Boolean(logical_origin >= cutlass.Int32(0))
    atom_is_full = cutlass.Boolean(
        atom_is_valid and logical_origin <= seq_len_kv - cutlass.Int32(cfg.atom_size)
    )
    if cutlass.const_expr(cfg.stores_score_words):
        # Atoms narrower than a word share it: an aligned group of adjacent
        # lanes ORs its atoms' bits into one word. A wider atom fills several.
        atoms_per_word = max(_WARP_SIZE // cfg.atom_size, 1)
        words_per_atom = max(cfg.atom_size // _WARP_SIZE, 1)
        atom_in_word = lane_idx % cutlass.Int32(atoms_per_word)
        first_word_index = (
            record_word_index
            + cutlass.Int32(cfg.token_words_word_offset)
            + atom_in_route // cutlass.Int32(atoms_per_word) * words_per_atom
        )
        for word_in_atom in cutlass.range_constexpr(words_per_atom):
            score_word = _load_atom_score_word(
                kv_valid_bits, logical_origin, word_in_atom, batch_idx, seq_len_kv, cfg
            ) << (atom_in_word * cutlass.Int32(cfg.atom_size))
            for step in cutlass.range_constexpr(atoms_per_word.bit_length() - 1):
                score_word = score_word | cute.arch.shuffle_sync_bfly(
                    score_word, offset=1 << step
                )
            if atom_in_word == cutlass.Int32(0):
                record_stage[first_word_index + word_in_atom] = cutlass.Int32(
                    score_word
                )
            atom_is_full = cutlass.Boolean(
                atom_is_full and score_word == cutlass.Uint32(0xFFFFFFFF)
            )

    atom_valid_mask = _lane_group_bits(
        cute.arch.vote_ballot_sync(atom_is_valid), lane_idx, origins_per_route
    )
    route_is_full = _lane_group_all(atom_is_full, lane_idx, origins_per_route)
    if atom_in_route == cutlass.Int32(0):
        record_stage[
            record_word_index + cutlass.Int32(cfg.atom_valid_mask_word_offset)
        ] = atom_valid_mask.bitcast(cutlass.Int32)
        record_stage[record_word_index + cutlass.Int32(cfg.route_flags_word_offset)] = (
            cutlass.Int32(_PREPARED_ROUTE_IS_FULL_FLAG)
            if route_is_full
            else cutlass.Int32(0)
        )
    _store_staged_records(
        route_workspace,
        record_stage,
        _route_metadata_word_index(row_route_begin, first_route_idx, cfg),
        exact_route_count - first_route_idx,
        lane_idx,
        cfg.exact_routes_per_pass,
        cfg,
    )


@cute.jit
def _resolve_paged_route_atom_page_id(
    block_tables: cute.Tensor,
    batch_idx: cutlass.Int32,
    block_table_row_stride: cutlass.Int64,
    logical_origin: cutlass.Int32,
    logical_origin_is_valid: cutlass.Boolean,
    lane_idx: cutlass.Int32,
    page_size: cutlass.Constexpr[int],
    num_physical_kv_pages: cutlass.Int64,
) -> cutlass.Int32:
    """Resolve one trusted selected logical atom to its physical page ID."""

    physical_page_id = cutlass.Int32(-1)
    page_id_is_valid = cutlass.Boolean(True)
    if logical_origin_is_valid:
        logical_page_idx = logical_origin // cutlass.Int32(page_size)
        physical_page_id = cutlass.Int32(
            block_tables.iterator[
                cutlass.Int64(batch_idx) * block_table_row_stride
                + cutlass.Int64(logical_page_idx)
            ]
        )
        page_id_is_valid = cutlass.Boolean(
            physical_page_id >= cutlass.Int32(0)
            and cutlass.Int64(physical_page_id) < num_physical_kv_pages
        )
    page_ids_are_valid = cute.arch.vote_all_sync(page_id_is_valid)
    if lane_idx == cutlass.Int32(0):
        runtime_assert(
            page_ids_are_valid,
            "block_tables contains an out-of-range physical page ID",
        )
    return physical_page_id


@cute.jit
def _inclusive_warp_prefix_sum(
    value: cutlass.Int32,
    lane_idx: cutlass.Int32,
) -> cutlass.Int32:
    """Return the inclusive warp prefix sum of one Int32 per lane."""

    prefix_sum = value
    for step in cutlass.range_constexpr((_WARP_SIZE - 1).bit_length()):
        # A zero clamp keeps shfl.up lanes below the offset on their own value.
        lower_sum = cute.arch.shuffle_sync_up(
            prefix_sum, offset=1 << step, mask_and_clamp=0
        )
        if lane_idx >= cutlass.Int32(1 << step):
            prefix_sum = prefix_sum + lower_sum
    return prefix_sum


@cute.jit
def _load_bitmask_word(
    exact_block_bits: cute.Tensor,
    batch_idx: cutlass.Int32,
    pattern_head_idx: cutlass.Int32,
    q_block_idx: cutlass.Int32,
    logical_word_idx: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
    for_proxy: cutlass.Constexpr[bool],
) -> cutlass.Uint32:
    """Load one in-range exact or proxy semantic-block word."""

    valid_word = _low_bits_mask(
        cutlass.Int32(cfg.num_kv_blocks) - logical_word_idx * cutlass.Int32(_WARP_SIZE)
    )
    selected_word = cutlass.Uint32(
        exact_block_bits[batch_idx, pattern_head_idx, q_block_idx, logical_word_idx]
    )
    if cutlass.const_expr(for_proxy):
        selected_word = ~selected_word
    return valid_word & selected_word


@cute.jit
def _list_bitmask_batch(
    exact_word: cutlass.Uint32,
    batch_word_idx: cutlass.Int32,
    selected_blocks: cute.Tensor,
    lane_idx: cutlass.Int32,
) -> cutlass.Int32:
    """List one batch's selected blocks in rank order; return their count.

    Lane ``i`` holds exact word ``batch_word_idx + i``. An inclusive warp
    prefix sum of the per-word counts gives each lane its list slots, which it
    fills from its highest selected block downward. The count is warp-uniform.
    """

    word_rank_end = _inclusive_warp_prefix_sum(
        cutlass.Int32(cute.arch.popc(exact_word)), lane_idx
    )
    first_word_block = (batch_word_idx + lane_idx) * cutlass.Int32(_WARP_SIZE)
    remaining_word = exact_word
    block_slot = word_rank_end
    while remaining_word != cutlass.Uint32(0):
        bit_idx = cutlass.Int32(cute.arch.bfind(remaining_word))
        block_slot -= cutlass.Int32(1)
        selected_blocks[block_slot] = first_word_block + bit_idx
        remaining_word = remaining_word ^ (cutlass.Uint32(1) << bit_idx)
    # Lanes gather from the list only after every lane has filled its slots.
    cute.arch.sync_warp()
    return _warp_broadcast_i32(word_rank_end, _WARP_SIZE - 1)


@cute.jit
def _gather_listed_atom_origin(
    selected_blocks: cute.Tensor,
    flat_atom_idx: cutlass.Int32,
    exact_atom_count: cutlass.Int32,
    batch_rank_begin: cutlass.Int32,
    batch_rank_end: cutlass.Int32,
    logical_origin: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> cutlass.Int32:
    """Return the origin of an exact atom whose block is in the listed batch.

    Atoms past the row's exact atoms or outside the batch keep
    ``logical_origin``.
    """

    block_rank = flat_atom_idx // cutlass.Int32(cfg.atoms_per_block)
    if (
        flat_atom_idx < exact_atom_count
        and block_rank >= batch_rank_begin
        and block_rank < batch_rank_end
    ):
        semantic_block_idx = cutlass.Int32(
            selected_blocks[block_rank - batch_rank_begin]
        )
        candidate_origin = semantic_block_idx * cutlass.Int32(
            cfg.kv_block_size
        ) + flat_atom_idx % cutlass.Int32(cfg.atoms_per_block) * cutlass.Int32(
            cfg.atom_size
        )
        if candidate_origin < cutlass.Int32(cfg.seq_len_kv):
            logical_origin = candidate_origin
    return logical_origin


@cute.jit
def _emit_bitmask_exact_routes(
    exact_block_bits: cute.Tensor,
    kv_valid_bits: cute.Tensor,
    route_workspace: cute.Tensor,
    record_stage: cute.Tensor,
    selected_blocks: cute.Tensor,
    first_exact_word: cutlass.Uint32,
    batch_idx: cutlass.Int32,
    pattern_head_idx: cutlass.Int32,
    q_block_idx: cutlass.Int32,
    row_route_begin: cutlass.Int32,
    exact_atom_count: cutlass.Int32,
    exact_route_count: cutlass.Int32,
    lane_idx: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> None:
    """Emit one bitmask row's exact records, one pass of atoms at a time.

    The row's selected blocks are listed in rank order one batch of 32 exact
    words at a time, starting with the already loaded ``first_exact_word``.
    Each pass gathers its atoms' blocks from the listed batch and moves on to
    later batches, possibly empty ones, until the one holding the pass's last
    block; atoms gathered from an earlier batch keep their origins.
    """

    batch_word_idx = cutlass.Int32(0)
    batch_rank_begin = cutlass.Int32(0)
    batch_rank_end = _list_bitmask_batch(
        first_exact_word, batch_word_idx, selected_blocks, lane_idx
    )
    first_route_idx = cutlass.Int32(0)
    while first_route_idx < exact_route_count:
        first_flat_atom_idx = first_route_idx * cutlass.Int32(
            cfg.logical_origins_per_route
        )
        flat_atom_idx = first_flat_atom_idx + lane_idx
        last_flat_atom_idx = first_flat_atom_idx + cutlass.Int32(_WARP_SIZE - 1)
        if last_flat_atom_idx >= exact_atom_count:
            last_flat_atom_idx = exact_atom_count - cutlass.Int32(1)
        last_block_rank = last_flat_atom_idx // cutlass.Int32(cfg.atoms_per_block)
        logical_origin = _gather_listed_atom_origin(
            selected_blocks,
            flat_atom_idx,
            exact_atom_count,
            batch_rank_begin,
            batch_rank_end,
            cutlass.Int32(-1),
            cfg,
        )
        while last_block_rank >= batch_rank_end and batch_word_idx + cutlass.Int32(
            _WARP_SIZE
        ) < cutlass.Int32(cfg.num_exact_words):
            # The next batch overwrites the list only after every lane has read it.
            cute.arch.sync_warp()
            batch_word_idx += cutlass.Int32(_WARP_SIZE)
            word_idx = batch_word_idx + lane_idx
            exact_word = cutlass.Uint32(0)
            if word_idx < cutlass.Int32(cfg.num_exact_words):
                exact_word = _load_bitmask_word(
                    exact_block_bits,
                    batch_idx,
                    pattern_head_idx,
                    q_block_idx,
                    word_idx,
                    cfg,
                    for_proxy=False,
                )
            batch_rank_begin = batch_rank_end
            batch_rank_end += _list_bitmask_batch(
                exact_word, batch_word_idx, selected_blocks, lane_idx
            )
            logical_origin = _gather_listed_atom_origin(
                selected_blocks,
                flat_atom_idx,
                exact_atom_count,
                batch_rank_begin,
                batch_rank_end,
                logical_origin,
                cfg,
            )
        _emit_exact_route_pass(
            route_workspace,
            record_stage,
            kv_valid_bits,
            row_route_begin,
            first_route_idx,
            exact_route_count,
            lane_idx,
            logical_origin,
            cutlass.Int32(-1),
            batch_idx,
            cutlass.Int32(cfg.seq_len_kv),
            cfg,
        )
        first_route_idx += cutlass.Int32(cfg.exact_routes_per_pass)


@cute.jit
def _load_bsr_proxy_word(
    block_indices: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    logical_word_idx: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> cutlass.Uint32:
    """Build one proxy word from a canonical sorted-BSR interval."""

    word_begin = logical_word_idx * cutlass.Int32(_WARP_SIZE)
    valid_word = _low_bits_mask(cutlass.Int32(cfg.num_kv_blocks) - word_begin)
    selected_word = cutlass.Uint32(0)
    lower = row_begin
    upper = row_end
    while lower < upper:
        middle = lower + (upper - lower) // cutlass.Int32(2)
        if cutlass.Int32(block_indices[middle]) < word_begin:
            lower = middle + cutlass.Int32(1)
        else:
            upper = middle
    cursor = lower
    word_end = word_begin + cutlass.Int32(_WARP_SIZE)
    scanning = cutlass.Boolean(True)
    while cursor < row_end and scanning:
        block_idx = cutlass.Int32(block_indices[cursor])
        if block_idx < word_end:
            selected_word = selected_word | (
                cutlass.Uint32(1) << (block_idx - word_begin)
            )
            cursor += cutlass.Int32(1)
        else:
            scanning = cutlass.Boolean(False)
    return valid_word & ~selected_word


@cute.jit
def _emit_proxy_routes(
    route_workspace: cute.Tensor,
    record_stage: cute.Tensor,
    row_route_begin: cutlass.Int32,
    exact_route_count: cutlass.Int32,
    lane_idx: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
    load_proxy_word: cutlass.Constexpr,
) -> None:
    """Emit every fixed summary-group proxy record, including empty masks.

    ``load_proxy_word(logical_word_idx)`` returns one in-range word of the
    row's proxy blocks, the complement of its exact blocks. Each aligned group
    of ``proxy_lanes_per_route`` lanes owns one record per pass: its lane ``j``
    stores origin ``j`` and score word ``j`` when the record has them.
    """

    lanes_per_route = cfg.proxy_lanes_per_route
    route_slot = lane_idx % cutlass.Int32(lanes_per_route)
    route_in_pass = lane_idx // cutlass.Int32(lanes_per_route)
    record_word_index = route_in_pass * cutlass.Int32(cfg.route_metadata_stride_words)
    group_capacity = cfg.token_words_per_route * _WARP_SIZE
    first_group_idx = cutlass.Int32(0)
    while first_group_idx < cutlass.Int32(cfg.num_proxy_groups):
        group_idx = first_group_idx + route_in_pass
        logical_word_idx = group_idx * cutlass.Int32(cfg.token_words_per_route) + (
            route_slot
        )
        proxy_word = cutlass.Uint32(0)
        if route_slot < cutlass.Int32(
            cfg.token_words_per_route
        ) and logical_word_idx < cutlass.Int32(cfg.num_exact_words):
            proxy_word = load_proxy_word(logical_word_idx)

        group_start = group_idx * cutlass.Int32(group_capacity)
        origin_is_valid = cutlass.Boolean(False)
        if route_slot < cutlass.Int32(cfg.logical_origins_per_route):
            summary_origin = group_start + route_slot * cutlass.Int32(cfg.atom_size)
            origin_is_valid = cutlass.Boolean(summary_origin < cfg.num_kv_blocks)
            stored_origin = cutlass.Int32(-1)
            if origin_is_valid:
                stored_origin = summary_origin
            record_stage[record_word_index + route_slot] = stored_origin
        if route_slot < cutlass.Int32(cfg.token_words_per_route):
            record_stage[
                record_word_index
                + cutlass.Int32(cfg.token_words_word_offset)
                + route_slot
            ] = cutlass.Int32(proxy_word)
        atom_valid_mask = _lane_group_bits(
            cute.arch.vote_ballot_sync(origin_is_valid), lane_idx, lanes_per_route
        )
        words_are_full = _lane_group_all(
            route_slot >= cutlass.Int32(cfg.token_words_per_route)
            or proxy_word == cutlass.Uint32(0xFFFFFFFF),
            lane_idx,
            lanes_per_route,
        )
        if route_slot == cutlass.Int32(0):
            record_stage[
                record_word_index + cutlass.Int32(cfg.atom_valid_mask_word_offset)
            ] = atom_valid_mask.bitcast(cutlass.Int32)
            proxy_is_full = cutlass.Boolean(
                group_start + cutlass.Int32(group_capacity)
                <= cutlass.Int32(cfg.num_kv_blocks)
                and words_are_full
            )
            record_stage[
                record_word_index + cutlass.Int32(cfg.route_flags_word_offset)
            ] = cutlass.Int32(_PREPARED_ROUTE_IS_PROXY_FLAG) | (
                cutlass.Int32(proxy_is_full)
                * cutlass.Int32(_PREPARED_ROUTE_IS_FULL_FLAG)
            )
        _store_staged_records(
            route_workspace,
            record_stage,
            _route_metadata_word_index(
                row_route_begin, exact_route_count + first_group_idx, cfg
            ),
            cutlass.Int32(cfg.num_proxy_groups) - first_group_idx,
            lane_idx,
            cfg.proxy_routes_per_pass,
            cfg,
        )
        first_group_idx += cutlass.Int32(cfg.proxy_routes_per_pass)


def _allocate_warp_words(
    smem: SmemAllocator,
    words_per_warp: int,
    warp_idx: cutlass.Int32,
) -> cute.Tensor:
    """Allocate ``words_per_warp`` shared Int32 words per warp; return this warp's."""

    return smem.allocate_tensor(
        cutlass.Int32,
        cute.make_layout((words_per_warp, _WARPS_PER_CTA)),
        byte_alignment=16,
    )[None, warp_idx]


@cute.jit
def _acquire_pdl_producer(use_pdl: cutlass.Constexpr[bool]) -> None:
    """Acquire the preceding grid of a programmatic dependent launch before exit.

    The routes depend on nothing that grid writes, so the prepare stores its
    records first and acquires last; a prepare launched without ``use_pdl``
    has no dependency to acquire.
    """

    if cutlass.const_expr(use_pdl):
        prims.griddepcontrol(kind=prims.GridDepAction.WAIT)


class _PrepareRoutesBase:
    """Own shared route geometry and compile-time storage/policy flags.

    ``num_pattern_heads`` is the head axis of the BSR or bitmask inputs, one
    when every KV head shares a pattern, and sizes the route rows. ``use_pdl``
    launches the prepare as a programmatic dependent of the grid launched
    immediately before it on the same stream and acquires that grid before
    the kernel exits.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_pattern_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        q_block_size: int,
        kv_block_size: int,
        kv_route_size: int,
        use_proxy_routes: bool,
        use_causal_mask: bool = False,
        apply_token_mask: bool = False,
        store_score_words: bool = False,
        page_size: int | None = None,
        use_pdl: bool = False,
    ) -> None:
        if not isinstance(use_proxy_routes, bool):
            raise TypeError("use_proxy_routes must be a bool")
        if not isinstance(apply_token_mask, bool):
            raise TypeError("apply_token_mask must be a bool")
        if not isinstance(store_score_words, bool):
            raise TypeError("store_score_words must be a bool")
        if not isinstance(use_causal_mask, bool):
            raise TypeError("use_causal_mask must be a bool")
        if not isinstance(use_pdl, bool):
            raise TypeError("use_pdl must be a bool")
        if use_proxy_routes and page_size is not None:
            raise ValueError("paged KV does not support proxy routes")

        num_q_blocks = (seq_len_q + q_block_size - 1) // q_block_size
        num_rows = batch_size * num_pattern_heads * num_q_blocks
        # Structural score words (sequence tail, invalid atoms) can be stored
        # without a caller token mask; proxy routes and token masks require them.
        stores_score_words = use_proxy_routes or apply_token_mask or store_score_words
        layout = _BlockSparseRouteLayout.create(
            kv_route_size=kv_route_size,
            kv_block_size=kv_block_size,
            page_size=page_size,
            has_token_bits=stores_score_words,
            route_metadata_capacity=0,
            num_rows=num_rows,
        )
        self.route_layout = layout
        self.cfg = _RouteConfig.create(
            layout=layout,
            num_pattern_heads=num_pattern_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            apply_token_mask=apply_token_mask,
            use_proxy_routes=use_proxy_routes,
        )
        self.page_size = page_size if page_size is not None else 1
        self.minimum_seq_len_kv = seq_len_q if use_causal_mask else 1
        self.use_pdl = use_pdl
        self.route_metadata_base_word_offset = layout.route_metadata_base_word_offset

    @property
    def grid(self) -> list[int]:
        """Return the launch grid: one warp per route row, four rows per CTA."""

        return [(self.cfg.num_rows + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA, 1, 1]


class _PrepareBsrRoutes(_PrepareRoutesBase):
    """Prepare continuous exact/proxy or paged exact routes from one BSR flow."""

    @cute.jit
    def __call__(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        seq_lens_kv: cute.Tensor | None,
        block_tables: cute.Tensor | None,
        num_physical_kv_pages: cutlass.Int64,
        block_table_row_stride: cutlass.Int64,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        stream: cuda_drv.CUstream,
    ) -> None:
        """Launch four independent BSR row preparers per CTA."""

        self.kernel(
            block_indptr,
            block_indices,
            kv_valid_bits,
            seq_lens_kv,
            block_tables,
            num_physical_kv_pages,
            block_table_row_stride,
            row_route_offsets,
            route_workspace,
            max_blocks_per_row,
        ).launch(
            grid=self.grid,
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        seq_lens_kv: cute.Tensor | None,
        block_tables: cute.Tensor | None,
        num_physical_kv_pages: cutlass.Int64,
        block_table_row_stride: cutlass.Int64,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
    ) -> None:
        """Assert trusted inputs, emit routes, resolve storage, then publish."""

        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = thread_idx // _WARP_SIZE
        lane_idx = thread_idx % _WARP_SIZE
        linear_row_idx = block_idx * _WARPS_PER_CTA + warp_idx
        row_is_valid = linear_row_idx < self.cfg.num_rows
        record_stage = _allocate_warp_words(
            SmemAllocator(), self.cfg.record_stage_words, warp_idx
        )

        row_route_span_begin, row_route_span_end = _load_row_route_span(
            row_route_offsets,
            linear_row_idx,
            lane_idx,
            row_is_valid,
        )
        row_begin, row_end, batch_idx = _resolve_prepared_bsr_row(
            block_indptr,
            block_indices,
            linear_row_idx,
            lane_idx,
            row_is_valid,
            self.cfg,
        )

        live_seq_len_kv = cutlass.Int32(self.cfg.seq_len_kv)
        selected_block_count = row_end - row_begin
        if cutlass.const_expr(self.route_layout.is_paged):
            raw_seq_len_kv = cutlass.Int32(self.cfg.seq_len_kv)
            if lane_idx == cutlass.Int32(0) and row_is_valid:
                raw_seq_len_kv = cutlass.Int32(seq_lens_kv[batch_idx])
                runtime_assert(
                    raw_seq_len_kv >= cutlass.Int32(self.minimum_seq_len_kv)
                    and raw_seq_len_kv <= cutlass.Int32(self.cfg.seq_len_kv),
                    "seq_lens_kv is outside the planned live-length range",
                )
            live_seq_len_kv = _warp_broadcast_i32(raw_seq_len_kv, 0)

            if (
                lane_idx == cutlass.Int32(0)
                and row_is_valid
                and selected_block_count > cutlass.Int32(0)
            ):
                last_block_idx = cutlass.Int32(
                    block_indices[row_end - cutlass.Int32(1)]
                )
                runtime_assert(
                    last_block_idx * cutlass.Int32(self.cfg.kv_block_size)
                    < live_seq_len_kv,
                    "block_indices row exceeds the live KV block range",
                )

            if lane_idx == cutlass.Int32(0) and row_is_valid:
                required_pages = _positive_i32_ceil_div(
                    live_seq_len_kv,
                    self.page_size,
                )
                runtime_assert(
                    required_pages <= cutlass.Int32(block_tables.shape[1]),
                    "block_tables row lacks the required live page capacity",
                )

        _, exact_route_count, total_route_count = _prepared_route_counts(
            selected_block_count,
            self.cfg,
        )
        if lane_idx == cutlass.Int32(0) and row_is_valid:
            runtime_assert(
                selected_block_count <= max_blocks_per_row,
                "selected BSR blocks exceed planned semantic capacity",
            )
        row_route_begin = _prepared_row_route_begin(
            row_route_span_begin,
            row_route_span_end,
            lane_idx,
            row_is_valid,
            total_route_count,
        )

        if row_is_valid:
            first_route_idx = cutlass.Int32(0)
            while first_route_idx < exact_route_count:
                (
                    logical_origin,
                    logical_origin_is_valid,
                ) = _resolve_route_logical_atom_origin(
                    block_indices,
                    row_begin,
                    row_end,
                    first_route_idx * cutlass.Int32(self.cfg.logical_origins_per_route)
                    + lane_idx,
                    live_seq_len_kv,
                    self.cfg,
                )
                physical_page_id = cutlass.Int32(-1)
                if cutlass.const_expr(self.route_layout.is_paged):
                    physical_page_id = _resolve_paged_route_atom_page_id(
                        block_tables,
                        batch_idx,
                        block_table_row_stride,
                        logical_origin,
                        logical_origin_is_valid,
                        lane_idx,
                        self.page_size,
                        num_physical_kv_pages,
                    )
                _emit_exact_route_pass(
                    route_workspace,
                    record_stage,
                    kv_valid_bits,
                    row_route_begin,
                    first_route_idx,
                    exact_route_count,
                    lane_idx,
                    logical_origin,
                    physical_page_id,
                    batch_idx,
                    live_seq_len_kv,
                    self.cfg,
                )
                first_route_idx += cutlass.Int32(self.cfg.exact_routes_per_pass)

            if cutlass.const_expr(self.cfg.use_proxy_routes):
                _emit_proxy_routes(
                    route_workspace,
                    record_stage,
                    row_route_begin,
                    exact_route_count,
                    lane_idx,
                    self.cfg,
                    lambda word_idx: _load_bsr_proxy_word(
                        block_indices, row_begin, row_end, word_idx, self.cfg
                    ),
                )

        if lane_idx == cutlass.Int32(0) and row_is_valid:
            route_workspace[linear_row_idx] = total_route_count
        _acquire_pdl_producer(self.use_pdl)


class _PrepareBitmaskRoutes(_PrepareRoutesBase):
    """Lower packed exact-block bits to continuous exact-first routes."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_pattern_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        q_block_size: int,
        kv_block_size: int,
        kv_route_size: int,
        use_proxy_routes: bool,
        use_causal_mask: bool = False,
        apply_token_mask: bool = False,
        store_score_words: bool = False,
        use_pdl: bool = False,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_pattern_heads=num_pattern_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            kv_route_size=kv_route_size,
            use_proxy_routes=use_proxy_routes,
            use_causal_mask=use_causal_mask,
            apply_token_mask=apply_token_mask,
            store_score_words=store_score_words,
            use_pdl=use_pdl,
        )

    @cute.jit
    def __call__(
        self,
        exact_block_bits: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        stream: cuda_drv.CUstream,
    ) -> None:
        self.kernel(
            exact_block_bits,
            kv_valid_bits,
            row_route_offsets,
            route_workspace,
            max_blocks_per_row,
        ).launch(
            grid=self.grid,
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        exact_block_bits: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
    ) -> None:
        """Pack one bitmask row after proving its complete payload fits."""

        thread_idx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = thread_idx // _WARP_SIZE
        lane_idx = thread_idx % _WARP_SIZE
        linear_row_idx = block_idx * _WARPS_PER_CTA + warp_idx
        row_is_valid = linear_row_idx < self.cfg.num_rows
        q_block_idx = linear_row_idx % self.cfg.num_q_blocks
        linear_batch_head_idx = linear_row_idx // self.cfg.num_q_blocks
        pattern_head_idx = linear_batch_head_idx % self.cfg.num_pattern_heads
        batch_idx = linear_batch_head_idx // self.cfg.num_pattern_heads
        smem = SmemAllocator()
        record_stage = _allocate_warp_words(smem, self.cfg.record_stage_words, warp_idx)
        selected_blocks = _allocate_warp_words(
            smem, self.cfg.selected_block_list_words, warp_idx
        )

        row_route_span_begin, row_route_span_end = _load_row_route_span(
            row_route_offsets,
            linear_row_idx,
            lane_idx,
            row_is_valid,
        )
        # Count the first word after the loop so its load overlaps the others.
        first_exact_word = cutlass.Uint32(0)
        if row_is_valid and lane_idx < cutlass.Int32(self.cfg.num_exact_words):
            first_exact_word = _load_bitmask_word(
                exact_block_bits,
                batch_idx,
                pattern_head_idx,
                q_block_idx,
                lane_idx,
                self.cfg,
                for_proxy=False,
            )
        lane_exact_count = cutlass.Int32(0)
        word_idx = lane_idx + cutlass.Int32(_WARP_SIZE)
        while word_idx < cutlass.Int32(self.cfg.num_exact_words):
            if row_is_valid:
                exact_word = _load_bitmask_word(
                    exact_block_bits,
                    batch_idx,
                    pattern_head_idx,
                    q_block_idx,
                    word_idx,
                    self.cfg,
                    for_proxy=False,
                )
                lane_exact_count += cutlass.Int32(cute.arch.popc(exact_word))
            word_idx += cutlass.Int32(_WARP_SIZE)
        lane_exact_count += cutlass.Int32(cute.arch.popc(first_exact_word))
        exact_block_count = cutlass.Int32(
            cute.arch.warp_redux_sync(lane_exact_count, "add")
        )

        exact_atom_count, exact_route_count, total_route_count = _prepared_route_counts(
            exact_block_count,
            self.cfg,
        )
        if lane_idx == cutlass.Int32(0) and row_is_valid:
            runtime_assert(
                exact_block_count <= max_blocks_per_row,
                "selected bitmask blocks exceed planned semantic capacity",
            )
        row_route_begin = _prepared_row_route_begin(
            row_route_span_begin,
            row_route_span_end,
            lane_idx,
            row_is_valid,
            total_route_count,
        )

        if row_is_valid:
            _emit_bitmask_exact_routes(
                exact_block_bits,
                kv_valid_bits,
                route_workspace,
                record_stage,
                selected_blocks,
                first_exact_word,
                batch_idx,
                pattern_head_idx,
                q_block_idx,
                row_route_begin,
                exact_atom_count,
                exact_route_count,
                lane_idx,
                self.cfg,
            )

            if cutlass.const_expr(self.cfg.use_proxy_routes):
                _emit_proxy_routes(
                    route_workspace,
                    record_stage,
                    row_route_begin,
                    exact_route_count,
                    lane_idx,
                    self.cfg,
                    lambda word_idx: _load_bitmask_word(
                        exact_block_bits,
                        batch_idx,
                        pattern_head_idx,
                        q_block_idx,
                        word_idx,
                        self.cfg,
                        for_proxy=True,
                    ),
                )

        if lane_idx == cutlass.Int32(0) and row_is_valid:
            route_workspace[linear_row_idx] = total_route_count
        _acquire_pdl_producer(self.use_pdl)


__all__ = ["_PrepareBitmaskRoutes", "_PrepareBsrRoutes"]
