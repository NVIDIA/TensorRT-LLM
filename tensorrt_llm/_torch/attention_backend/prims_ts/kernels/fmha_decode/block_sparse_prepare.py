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
It validates each canonical row once, emits the same logical exact records,
then either resolves paged locators or appends continuous proxy records. The
bitmask frontend shares the record geometry and emitters but remains limited
to continuous storage. Proxy suffixes contain one stable record per summary
group; a fully exact group remains present with zero score words. One warp owns
one sparse row and four warps share a CTA.

``row_route_offsets`` is a separate plan-owned immutable Int32 tensor.
``route_workspace`` contains only mutable row counts and route metadata
described by ``_BlockSparseRouteLayout``. Payload outside each live row
count is intentionally stale. ``max_blocks_per_row`` is the plan-declared
semantic BSR-block limit, which remains distinct from packed-route capacity.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_drv
from cutlass.cute.testing import assert_ as runtime_assert

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


@dataclass(frozen=True)
class _RouteConfig:
    """Compile-time route geometry shared across sparse input/storage modes."""

    num_kv_heads: int
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
    stores_score_words: bool
    apply_token_mask: bool
    use_proxy_routes: bool
    route_metadata_stride_words: int
    route_metadata_base_word_offset: int

    @staticmethod
    def create(
        *,
        layout: _BlockSparseRouteLayout,
        num_kv_heads: int,
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
        return _RouteConfig(
            num_kv_heads=num_kv_heads,
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
            stores_score_words=stores_score_words,
            apply_token_mask=apply_token_mask,
            use_proxy_routes=use_proxy_routes,
            route_metadata_stride_words=layout.route_metadata_stride_words,
            route_metadata_base_word_offset=layout.route_metadata_base_word_offset,
        )


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
def _prepared_row_route_begin(
    row_route_offsets: cute.Tensor,
    linear_row_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    row_is_valid: cutlass.Boolean,
    total_route_count: cutlass.Int32,
) -> cutlass.Int32:
    """Load and validate one row's plan-owned prepared-route span."""

    row_route_begin = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0) and row_is_valid:
        row_route_begin = cutlass.Int32(row_route_offsets[linear_row_idx])
        row_route_end = cutlass.Int32(row_route_offsets[linear_row_idx + 1])
        row_capacity = row_route_end - row_route_begin
        runtime_assert(
            row_route_begin >= cutlass.Int32(0)
            and row_capacity >= cutlass.Int32(0)
            and total_route_count <= row_capacity,
            "prepared routes exceed planned row capacity",
        )
    return _warp_broadcast_i32(row_route_begin, 0)


@cute.jit
def _resolve_route_logical_atom_origin(
    block_indices: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    route_idx: cutlass.Int32,
    atom_in_route: cutlass.Int32,
    kv_block_size: cutlass.Constexpr[int],
    atom_size: cutlass.Constexpr[int],
    logical_origins_per_route: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Int32,
) -> tuple[cutlass.Int32, cutlass.Boolean]:
    """Resolve one route atom to its logical KV-token origin."""

    atoms_per_block = kv_block_size // atom_size
    flat_atom_idx = route_idx * cutlass.Int32(logical_origins_per_route) + atom_in_route
    bsr_entry_offset = flat_atom_idx // cutlass.Int32(atoms_per_block)
    atom_in_block = flat_atom_idx % cutlass.Int32(atoms_per_block)
    valid = cutlass.Boolean(bsr_entry_offset < row_end - row_begin)
    logical_origin = cutlass.Int32(-1)
    if valid:
        block_idx = cutlass.Int32(block_indices[row_begin + bsr_entry_offset])
        block_origin = block_idx * cutlass.Int32(kv_block_size)
        atom_offset = atom_in_block * cutlass.Int32(atom_size)
        valid = cutlass.Boolean(atom_offset < cutlass.Int32(seq_len_kv) - block_origin)
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
def _load_exact_score_word(
    route_workspace: cute.Tensor,
    kv_valid_bits: cute.Tensor,
    route_metadata_word_index: cutlass.Int32,
    logical_word_idx: cutlass.Int32,
    batch_idx: cutlass.Int32,
    seq_len_kv: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> cutlass.Uint32:
    """Build one exact score word with optional caller-token masking."""

    token_word = cutlass.Uint32(0)
    if cutlass.const_expr(cfg.atom_size <= _WARP_SIZE):
        atoms_per_word = _WARP_SIZE // cfg.atom_size
        first_atom_idx = logical_word_idx * cutlass.Int32(atoms_per_word)
        for atom_in_word in cutlass.range_constexpr(atoms_per_word):
            atom_idx = first_atom_idx + cutlass.Int32(atom_in_word)
            if atom_idx < cutlass.Int32(cfg.logical_origins_per_route):
                origin = cutlass.Int32(
                    route_workspace[route_metadata_word_index + atom_idx]
                )
                atom_word = cutlass.Uint32(0)
                if origin >= cutlass.Int32(0):
                    if cutlass.const_expr(cfg.apply_token_mask):
                        source_word_idx = origin >> cutlass.Int32(5)
                        atom_word = cutlass.Uint32(
                            kv_valid_bits[batch_idx, source_word_idx]
                        )
                        atom_word = atom_word >> (origin & cutlass.Int32(31))
                        atom_word = atom_word & cutlass.Uint32((1 << cfg.atom_size) - 1)
                        atom_word = atom_word & _low_bits_mask(seq_len_kv - origin)
                    else:
                        atom_word = _low_bits_mask(
                            seq_len_kv - origin
                        ) & cutlass.Uint32((1 << cfg.atom_size) - 1)
                token_word = token_word | (
                    atom_word << cutlass.Int32(atom_in_word * cfg.atom_size)
                )
    else:
        words_per_atom = cfg.atom_size // _WARP_SIZE
        atom_idx = logical_word_idx // cutlass.Int32(words_per_atom)
        word_in_atom = logical_word_idx % cutlass.Int32(words_per_atom)
        origin = cutlass.Int32(route_workspace[route_metadata_word_index + atom_idx])
        word_origin = origin + word_in_atom * cutlass.Int32(_WARP_SIZE)
        if origin >= cutlass.Int32(0):
            if cutlass.const_expr(cfg.apply_token_mask):
                if word_origin < seq_len_kv:
                    source_word_idx = word_origin >> cutlass.Int32(5)
                    token_word = cutlass.Uint32(
                        kv_valid_bits[batch_idx, source_word_idx]
                    ) & _low_bits_mask(seq_len_kv - word_origin)
            else:
                token_word = _low_bits_mask(seq_len_kv - word_origin)
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
        kv_head_idx = linear_batch_head_idx % cfg.num_kv_heads
        batch_idx = linear_batch_head_idx // cfg.num_kv_heads
        row_begin = cutlass.Int32(block_indptr[batch_idx, kv_head_idx, q_block_row_idx])
        row_end = cutlass.Int32(
            block_indptr[batch_idx, kv_head_idx, q_block_row_idx + 1]
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
def _finalize_exact_route(
    route_workspace: cute.Tensor,
    kv_valid_bits: cute.Tensor,
    route_metadata_word_index: cutlass.Int32,
    batch_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    atom_is_valid: cutlass.Boolean,
    seq_len_kv: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> None:
    """Finalize an exact record after its logical origins are stored."""

    atom_is_full = cutlass.Boolean(False)
    if lane_idx < cutlass.Int32(cfg.logical_origins_per_route):
        origin = cutlass.Int32(route_workspace[route_metadata_word_index + lane_idx])
        atom_is_full = cutlass.Boolean(
            atom_is_valid and origin <= seq_len_kv - cutlass.Int32(cfg.atom_size)
        )
    atom_valid_mask = cutlass.Int32(cute.arch.vote_ballot_sync(atom_is_valid))
    structural_full = cute.arch.vote_all_sync(
        lane_idx >= cutlass.Int32(cfg.logical_origins_per_route) or atom_is_full
    )

    score_words_are_full = cutlass.Boolean(True)
    if cutlass.const_expr(cfg.stores_score_words):
        score_word = cutlass.Uint32(0)
        if lane_idx < cutlass.Int32(cfg.token_words_per_route):
            score_word = _load_exact_score_word(
                route_workspace,
                kv_valid_bits,
                route_metadata_word_index,
                lane_idx,
                batch_idx,
                seq_len_kv,
                cfg,
            )
            route_workspace[
                route_metadata_word_index
                + cutlass.Int32(cfg.token_words_word_offset)
                + lane_idx
            ] = cutlass.Int32(score_word)
        score_words_are_full = cute.arch.vote_all_sync(
            lane_idx >= cutlass.Int32(cfg.token_words_per_route)
            or score_word == cutlass.Uint32(0xFFFFFFFF)
        )
    if lane_idx == cutlass.Int32(0):
        route_workspace[
            route_metadata_word_index + cutlass.Int32(cfg.atom_valid_mask_word_offset)
        ] = atom_valid_mask
        route_workspace[
            route_metadata_word_index + cutlass.Int32(cfg.route_flags_word_offset)
        ] = (
            cutlass.Int32(_PREPARED_ROUTE_IS_FULL_FLAG)
            if structural_full and score_words_are_full
            else cutlass.Int32(0)
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
def _exact_lane_rank(
    exact_ballot: cutlass.Uint32,
    lane_idx: cutlass.Int32,
    exact_prefix: cutlass.Int32,
) -> cutlass.Int32:
    """Return one exact lane's global semantic-block rank."""

    lower_lane_mask = (cutlass.Uint32(1) << lane_idx) - cutlass.Uint32(1)
    return exact_prefix + cutlass.Int32(cute.arch.popc(exact_ballot & lower_lane_mask))


@cute.jit
def _emit_exact_block_atoms(
    route_workspace: cute.Tensor,
    row_route_begin: cutlass.Int32,
    semantic_block_idx: cutlass.Int32,
    exact_block_rank: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> None:
    """Expand one bitmask-selected block into fixed row-global atom slots."""

    first_atom_rank = exact_block_rank * cutlass.Int32(cfg.atoms_per_block)
    atom_in_block = cutlass.Int32(0)
    while atom_in_block < cutlass.Int32(cfg.atoms_per_block):
        atom_rank = first_atom_rank + atom_in_block
        route_idx = atom_rank // cutlass.Int32(cfg.logical_origins_per_route)
        atom_in_route = atom_rank % cutlass.Int32(cfg.logical_origins_per_route)
        route_word_index = cutlass.Int32(cfg.route_metadata_base_word_offset) + (
            (row_route_begin + route_idx)
            * cutlass.Int32(cfg.route_metadata_stride_words)
        )
        logical_origin = semantic_block_idx * cutlass.Int32(
            cfg.kv_block_size
        ) + atom_in_block * cutlass.Int32(cfg.atom_size)
        stored_origin = cutlass.Int32(-1)
        if logical_origin < cutlass.Int32(cfg.seq_len_kv):
            stored_origin = logical_origin
        route_workspace[route_word_index + atom_in_route] = stored_origin
        atom_in_block += cutlass.Int32(1)


@cute.jit
def _load_bitmask_word(
    exact_block_bits: cute.Tensor,
    batch_idx: cutlass.Int32,
    kv_head_idx: cutlass.Int32,
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
        exact_block_bits[batch_idx, kv_head_idx, q_block_idx, logical_word_idx]
    )
    if cutlass.const_expr(for_proxy):
        selected_word = ~selected_word
    return valid_word & selected_word


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
def _emit_proxy_route(
    route_workspace: cute.Tensor,
    row_route_begin: cutlass.Int32,
    exact_route_count: cutlass.Int32,
    group_idx: cutlass.Int32,
    proxy_word: cutlass.Uint32,
    lane_idx: cutlass.Int32,
    cfg: cutlass.Constexpr[_RouteConfig],
) -> None:
    """Emit one fixed summary-group proxy record, including an empty mask."""

    route_metadata_word_index = cutlass.Int32(cfg.route_metadata_base_word_offset) + (
        row_route_begin + exact_route_count + group_idx
    ) * cutlass.Int32(cfg.route_metadata_stride_words)
    group_start = group_idx * cutlass.Int32(cfg.token_words_per_route * _WARP_SIZE)
    group_size = cutlass.Int32(cfg.num_kv_blocks) - group_start
    if group_size > cutlass.Int32(cfg.token_words_per_route * _WARP_SIZE):
        group_size = cutlass.Int32(cfg.token_words_per_route * _WARP_SIZE)
    origin_is_valid = cutlass.Boolean(False)
    if lane_idx < cutlass.Int32(cfg.logical_origins_per_route):
        summary_origin = group_start + lane_idx * cutlass.Int32(cfg.atom_size)
        origin_is_valid = cutlass.Boolean(summary_origin < cfg.num_kv_blocks)
        stored_origin = cutlass.Int32(-1)
        if origin_is_valid:
            stored_origin = summary_origin
        route_workspace[route_metadata_word_index + lane_idx] = stored_origin
    atom_valid_mask = cutlass.Int32(cute.arch.vote_ballot_sync(origin_is_valid))
    if lane_idx < cutlass.Int32(cfg.token_words_per_route):
        route_workspace[
            route_metadata_word_index
            + cutlass.Int32(cfg.token_words_word_offset)
            + lane_idx
        ] = cutlass.Int32(proxy_word)
    score_full = cute.arch.vote_all_sync(
        lane_idx >= cutlass.Int32(cfg.token_words_per_route)
        or proxy_word == cutlass.Uint32(0xFFFFFFFF)
    )
    if lane_idx == cutlass.Int32(0):
        route_workspace[
            route_metadata_word_index + cutlass.Int32(cfg.atom_valid_mask_word_offset)
        ] = atom_valid_mask
        proxy_is_full = cutlass.Boolean(
            group_size == cutlass.Int32(cfg.token_words_per_route * _WARP_SIZE)
            and score_full
        )
        route_workspace[
            route_metadata_word_index + cutlass.Int32(cfg.route_flags_word_offset)
        ] = cutlass.Int32(_PREPARED_ROUTE_IS_PROXY_FLAG) | (
            cutlass.Int32(proxy_is_full) * cutlass.Int32(_PREPARED_ROUTE_IS_FULL_FLAG)
        )


class _PrepareRoutesBase:
    """Own shared route geometry and compile-time storage/policy flags."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        q_block_size: int,
        kv_block_size: int,
        kv_route_size: int,
        use_proxy_routes: bool,
        use_causal_mask: bool = False,
        apply_token_mask: bool = False,
        page_size: int | None = None,
    ) -> None:
        if not isinstance(use_proxy_routes, bool):
            raise TypeError("use_proxy_routes must be a bool")
        if not isinstance(apply_token_mask, bool):
            raise TypeError("apply_token_mask must be a bool")
        if not isinstance(use_causal_mask, bool):
            raise TypeError("use_causal_mask must be a bool")
        if use_proxy_routes and page_size is not None:
            raise ValueError("paged KV does not support proxy routes")

        num_q_blocks = (seq_len_q + q_block_size - 1) // q_block_size
        num_rows = batch_size * num_kv_heads * num_q_blocks
        stores_score_words = use_proxy_routes or apply_token_mask
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
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            apply_token_mask=apply_token_mask,
            use_proxy_routes=use_proxy_routes,
        )
        self.page_size = page_size if page_size is not None else 1
        self.minimum_seq_len_kv = seq_len_q if use_causal_mask else 1
        self.physical_page_ids_word_offset = (
            layout.physical_page_ids_word_offset if layout.is_paged else 0
        )
        self.route_metadata_base_word_offset = layout.route_metadata_base_word_offset


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
            grid=[
                (self.cfg.num_rows + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA,
                1,
                1,
            ],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
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
            row_route_offsets,
            linear_row_idx,
            lane_idx,
            row_is_valid,
            total_route_count,
        )

        if row_is_valid:
            route_idx = cutlass.Int32(0)
            while route_idx < exact_route_count:
                route_word_index = cutlass.Int32(
                    self.cfg.route_metadata_base_word_offset
                ) + (row_route_begin + route_idx) * cutlass.Int32(
                    self.cfg.route_metadata_stride_words
                )
                logical_origin = cutlass.Int32(-1)
                logical_origin_is_valid = cutlass.Boolean(False)
                physical_page_id = cutlass.Int32(-1)
                if lane_idx < cutlass.Int32(self.cfg.logical_origins_per_route):
                    (
                        logical_origin,
                        logical_origin_is_valid,
                    ) = _resolve_route_logical_atom_origin(
                        block_indices,
                        row_begin,
                        row_end,
                        route_idx,
                        lane_idx,
                        self.cfg.kv_block_size,
                        self.cfg.atom_size,
                        self.cfg.logical_origins_per_route,
                        live_seq_len_kv,
                    )
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
                if lane_idx < cutlass.Int32(self.cfg.logical_origins_per_route):
                    route_workspace[route_word_index + lane_idx] = logical_origin
                    if cutlass.const_expr(self.route_layout.is_paged):
                        route_workspace[
                            route_word_index
                            + cutlass.Int32(self.physical_page_ids_word_offset)
                            + lane_idx
                        ] = physical_page_id
                cute.arch.sync_warp()

                _finalize_exact_route(
                    route_workspace,
                    kv_valid_bits,
                    route_word_index,
                    batch_idx,
                    lane_idx,
                    logical_origin_is_valid,
                    live_seq_len_kv,
                    self.cfg,
                )
                route_idx += cutlass.Int32(1)

            if cutlass.const_expr(self.cfg.use_proxy_routes):
                group_idx = cutlass.Int32(0)
                while group_idx < cutlass.Int32(self.cfg.num_proxy_groups):
                    proxy_word = cutlass.Uint32(0)
                    logical_word_idx = (
                        group_idx * cutlass.Int32(self.cfg.token_words_per_route)
                        + lane_idx
                    )
                    if lane_idx < cutlass.Int32(self.cfg.token_words_per_route):
                        if logical_word_idx < cutlass.Int32(self.cfg.num_exact_words):
                            proxy_word = _load_bsr_proxy_word(
                                block_indices,
                                row_begin,
                                row_end,
                                logical_word_idx,
                                self.cfg,
                            )
                    _emit_proxy_route(
                        route_workspace,
                        row_route_begin,
                        exact_route_count,
                        group_idx,
                        proxy_word,
                        lane_idx,
                        self.cfg,
                    )
                    group_idx += cutlass.Int32(1)

        if lane_idx == cutlass.Int32(0) and row_is_valid:
            route_workspace[linear_row_idx] = total_route_count


class _PrepareBitmaskRoutes(_PrepareRoutesBase):
    """Lower packed exact-block bits to continuous exact-first routes."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        q_block_size: int,
        kv_block_size: int,
        kv_route_size: int,
        use_proxy_routes: bool,
        use_causal_mask: bool = False,
        apply_token_mask: bool = False,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            kv_route_size=kv_route_size,
            use_proxy_routes=use_proxy_routes,
            use_causal_mask=use_causal_mask,
            apply_token_mask=apply_token_mask,
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
            grid=[
                (self.cfg.num_rows + _WARPS_PER_CTA - 1) // _WARPS_PER_CTA,
                1,
                1,
            ],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
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
        kv_head_idx = linear_batch_head_idx % self.cfg.num_kv_heads
        batch_idx = linear_batch_head_idx // self.cfg.num_kv_heads

        lane_exact_count = cutlass.Int32(0)
        word_idx = lane_idx
        while word_idx < cutlass.Int32(self.cfg.num_exact_words):
            if row_is_valid:
                exact_word = _load_bitmask_word(
                    exact_block_bits,
                    batch_idx,
                    kv_head_idx,
                    q_block_idx,
                    word_idx,
                    self.cfg,
                    for_proxy=False,
                )
                lane_exact_count += cutlass.Int32(cute.arch.popc(exact_word))
            word_idx += cutlass.Int32(_WARP_SIZE)
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
            row_route_offsets,
            linear_row_idx,
            lane_idx,
            row_is_valid,
            total_route_count,
        )

        if row_is_valid:
            exact_prefix = cutlass.Int32(0)
            word_idx = cutlass.Int32(0)
            while word_idx < cutlass.Int32(self.cfg.num_exact_words):
                exact_word_i32 = cutlass.Int32(0)
                if lane_idx == cutlass.Int32(0):
                    exact_word_i32 = _load_bitmask_word(
                        exact_block_bits,
                        batch_idx,
                        kv_head_idx,
                        q_block_idx,
                        word_idx,
                        self.cfg,
                        for_proxy=False,
                    ).bitcast(cutlass.Int32)
                exact_word = _warp_broadcast_i32(exact_word_i32, 0).bitcast(
                    cutlass.Uint32
                )
                is_exact = cutlass.Boolean(
                    (exact_word & (cutlass.Uint32(1) << lane_idx)) != cutlass.Uint32(0)
                )
                exact_ballot = cute.arch.vote_ballot_sync(is_exact).bitcast(
                    cutlass.Uint32
                )
                exact_rank = _exact_lane_rank(exact_ballot, lane_idx, exact_prefix)
                if is_exact:
                    _emit_exact_block_atoms(
                        route_workspace,
                        row_route_begin,
                        word_idx * cutlass.Int32(_WARP_SIZE) + lane_idx,
                        exact_rank,
                        self.cfg,
                    )
                exact_prefix += cutlass.Int32(cute.arch.popc(exact_ballot))
                word_idx += cutlass.Int32(1)

            final_route_atom_count = exact_atom_count % cutlass.Int32(
                self.cfg.logical_origins_per_route
            )
            if final_route_atom_count != cutlass.Int32(0):
                if lane_idx >= final_route_atom_count and lane_idx < cutlass.Int32(
                    self.cfg.logical_origins_per_route
                ):
                    final_route_word_index = cutlass.Int32(
                        self.cfg.route_metadata_base_word_offset
                    ) + (row_route_begin + exact_route_count - cutlass.Int32(1)) * (
                        cutlass.Int32(self.cfg.route_metadata_stride_words)
                    )
                    route_workspace[final_route_word_index + lane_idx] = cutlass.Int32(
                        -1
                    )
            cute.arch.sync_warp()

            route_idx = cutlass.Int32(0)
            while route_idx < exact_route_count:
                route_word_index = cutlass.Int32(
                    self.cfg.route_metadata_base_word_offset
                ) + (row_route_begin + route_idx) * cutlass.Int32(
                    self.cfg.route_metadata_stride_words
                )
                atom_is_valid = cutlass.Boolean(False)
                if lane_idx < cutlass.Int32(self.cfg.logical_origins_per_route):
                    atom_is_valid = cutlass.Boolean(
                        cutlass.Int32(route_workspace[route_word_index + lane_idx])
                        >= cutlass.Int32(0)
                    )
                _finalize_exact_route(
                    route_workspace,
                    kv_valid_bits,
                    route_word_index,
                    batch_idx,
                    lane_idx,
                    atom_is_valid,
                    cutlass.Int32(self.cfg.seq_len_kv),
                    self.cfg,
                )
                route_idx += cutlass.Int32(1)

            if cutlass.const_expr(self.cfg.use_proxy_routes):
                group_idx = cutlass.Int32(0)
                while group_idx < cutlass.Int32(self.cfg.num_proxy_groups):
                    proxy_word = cutlass.Uint32(0)
                    logical_word_idx = (
                        group_idx * cutlass.Int32(self.cfg.token_words_per_route)
                        + lane_idx
                    )
                    if lane_idx < cutlass.Int32(self.cfg.token_words_per_route):
                        if logical_word_idx < cutlass.Int32(self.cfg.num_exact_words):
                            proxy_word = _load_bitmask_word(
                                exact_block_bits,
                                batch_idx,
                                kv_head_idx,
                                q_block_idx,
                                logical_word_idx,
                                self.cfg,
                                for_proxy=True,
                            )
                    _emit_proxy_route(
                        route_workspace,
                        row_route_begin,
                        exact_route_count,
                        group_idx,
                        proxy_word,
                        lane_idx,
                        self.cfg,
                    )
                    group_idx += cutlass.Int32(1)

        if lane_idx == cutlass.Int32(0) and row_is_valid:
            route_workspace[linear_row_idx] = total_route_count


__all__ = ["_PrepareBitmaskRoutes", "_PrepareBsrRoutes"]
