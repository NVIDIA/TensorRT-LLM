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

"""Prepare per-run canonical BSR rows for the PrimTS FMHA route consumer.

The kernel converts caller-owned semantic KV blocks into fixed-stride route
metadata on every run. Route origins are logical KV-token coordinates;
the paged specialization also resolves each origin to a physical page ID for
the attention load path. One warp handles one BSR row and iterates only that
row's active routes, while four warps share a CTA.

``row_route_offsets`` is a separate plan-owned immutable Int32 tensor.
``route_workspace`` contains only mutable row counts and route metadata
described by ``_BlockSparseRouteLayout``. Payload outside each active row
count is intentionally stale. ``max_blocks_per_row`` is the plan-declared
semantic BSR-block limit, which remains distinct from packed-route capacity.
"""

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_drv
from cutlass.cute.testing import assert_ as runtime_assert
from cutlass.experimental import primitives as prims

from ..._block_sparse.prepared import (
    _PREPARED_ROUTE_IS_FULL_FLAG,
    _BlockSparseRouteLayout,
)
from .block_sparse_inspect import _validate_bsr_row_lane
from .fmha_decode_resources.helpers_common import _warp_broadcast_i32


_WARPS_PER_CTA = 4
_WARP_SIZE = 32
_THREADS_PER_CTA = _WARPS_PER_CTA * _WARP_SIZE


@dataclass(frozen=True)
class _PreparedRouteConfig:
    """Compile-time geometry shared by contiguous and paged route packing."""

    num_kv_heads: int
    num_q_block_rows: int
    num_kv_blocks: int
    num_rows: int
    seq_len_kv: int
    kv_block_size: int
    atom_size: int
    logical_origins_per_route: int
    token_words_per_route: int
    atom_valid_mask_word_offset: int
    route_flags_word_offset: int
    token_words_word_offset: int
    has_token_bits: bool
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
    ) -> "_PreparedRouteConfig":
        """Build shared prepare geometry without adding a storage-mode flag."""

        num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
        return _PreparedRouteConfig(
            num_kv_heads=num_kv_heads,
            num_q_block_rows=num_q_block_rows,
            num_kv_blocks=(seq_len_kv + kv_block_size - 1) // kv_block_size,
            num_rows=layout.num_rows,
            seq_len_kv=seq_len_kv,
            kv_block_size=kv_block_size,
            atom_size=layout.atom_size,
            logical_origins_per_route=layout.logical_origins_per_route,
            token_words_per_route=layout.token_words_per_route,
            atom_valid_mask_word_offset=layout.atom_valid_mask_word_offset,
            route_flags_word_offset=layout.route_flags_word_offset,
            token_words_word_offset=(
                layout.token_words_word_offset
                if layout.token_words_word_offset is not None
                else 0
            ),
            has_token_bits=layout.has_token_bits,
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
def _retained_atom_count(
    block_indices: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    kv_block_size: cutlass.Constexpr[int],
    atom_size: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Int32,
) -> cutlass.Int32:
    """Count selected atoms whose logical origin precedes ``seq_len_kv``."""

    row_nnz = row_end - row_begin
    retained_atoms = cutlass.Int32(0)
    if row_nnz > cutlass.Int32(0):
        atoms_per_block = kv_block_size // atom_size
        retained_atoms = (row_nnz - cutlass.Int32(1)) * cutlass.Int32(atoms_per_block)
        last_block_idx = cutlass.Int32(block_indices[row_end - cutlass.Int32(1)])
        last_block_origin = last_block_idx * cutlass.Int32(kv_block_size)
        remaining_tokens = cutlass.Int32(seq_len_kv) - last_block_origin
        runtime_assert(
            remaining_tokens > cutlass.Int32(0),
            "block_indices row exceeds the active KV block range",
        )
        retained_last_atoms = (remaining_tokens - cutlass.Int32(1)) // cutlass.Int32(
            atom_size
        ) + cutlass.Int32(1)
        if retained_last_atoms > cutlass.Int32(atoms_per_block):
            retained_last_atoms = cutlass.Int32(atoms_per_block)
        retained_atoms = retained_atoms + retained_last_atoms
    return retained_atoms


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
def _load_coarse_token_word(
    block_indices: cute.Tensor,
    kv_valid_bits: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    route_idx: cutlass.Int32,
    logical_word_idx: cutlass.Int32,
    batch_idx: cutlass.Int32,
    kv_block_size: cutlass.Constexpr[int],
    atom_size: cutlass.Constexpr[int],
    logical_origins_per_route: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Int32,
) -> cutlass.Uint32:
    """Load one logical K32 word from a coarse atom larger than K32."""

    logical_word = cutlass.Uint32(0)
    words_per_atom = atom_size // 32
    atom_in_route = logical_word_idx // cutlass.Int32(words_per_atom)
    word_in_atom = logical_word_idx % cutlass.Int32(words_per_atom)
    logical_origin, valid = _resolve_route_logical_atom_origin(
        block_indices,
        row_begin,
        row_end,
        route_idx,
        atom_in_route,
        kv_block_size,
        atom_size,
        logical_origins_per_route,
        seq_len_kv,
    )
    logical_word_origin = logical_origin + word_in_atom * cutlass.Int32(32)
    if valid and logical_word_origin < cutlass.Int32(seq_len_kv):
        valid_bits_word_idx = logical_word_origin >> cutlass.Int32(5)
        logical_word = cutlass.Uint32(kv_valid_bits[batch_idx, valid_bits_word_idx])
        remaining_tokens = cutlass.Int32(seq_len_kv) - logical_word_origin
        if remaining_tokens < cutlass.Int32(32):
            logical_word = logical_word & (
                (cutlass.Uint32(1) << remaining_tokens) - cutlass.Uint32(1)
            )
    return logical_word


@cute.jit
def _load_atom_token_chunk(
    kv_valid_bits: cute.Tensor,
    batch_idx: cutlass.Int32,
    logical_origin: cutlass.Int32,
    origin_is_valid: cutlass.Boolean,
    atom_size: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Int32,
) -> cutlass.Uint32:
    """Load the <=K32 mask chunk owned by one resolved-origin lane."""

    token_chunk = cutlass.Uint32(0)
    if origin_is_valid:
        valid_bits_word_idx = logical_origin >> cutlass.Int32(5)
        source_word = cutlass.Uint32(kv_valid_bits[batch_idx, valid_bits_word_idx])
        token_chunk = source_word >> (logical_origin & cutlass.Int32(31))
        token_chunk = token_chunk & cutlass.Uint32((1 << atom_size) - 1)
        remaining_tokens = cutlass.Int32(seq_len_kv) - logical_origin
        if remaining_tokens < cutlass.Int32(atom_size):
            token_chunk = token_chunk & (
                (cutlass.Uint32(1) << remaining_tokens) - cutlass.Uint32(1)
            )
    return token_chunk


@cute.jit
def _resolve_prepared_bsr_row(
    block_indptr: cute.Tensor,
    block_indices: cute.Tensor,
    linear_row_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    row_is_valid: cutlass.Boolean,
    cfg: cutlass.Constexpr[_PreparedRouteConfig],
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """Resolve one trusted canonical runtime BSR row."""

    row_begin = cutlass.Int32(0)
    row_end = cutlass.Int32(0)
    batch_idx = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0) and row_is_valid:
        q_block_row_idx = linear_row_idx % cfg.num_q_block_rows
        linear_batch_head_idx = linear_row_idx // cfg.num_q_block_rows
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
def _publish_prepared_route_count(
    block_indices: cute.Tensor,
    row_route_offsets: cute.Tensor,
    route_workspace: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    linear_row_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    row_is_valid: cutlass.Boolean,
    max_blocks_per_row: cutlass.Int32,
    seq_len_kv: cutlass.Int32,
    cfg: cutlass.Constexpr[_PreparedRouteConfig],
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Assert semantic capacity, publish the header, and return its active span."""

    row_route_begin = cutlass.Int32(0)
    required_route_count = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0) and row_is_valid:
        row_route_begin = cutlass.Int32(row_route_offsets[linear_row_idx])
        selected_block_count = row_end - row_begin
        runtime_assert(
            selected_block_count <= max_blocks_per_row,
            "selected BSR blocks exceed planned semantic capacity",
        )
        retained_atom_count = _retained_atom_count(
            block_indices,
            row_begin,
            row_end,
            cfg.kv_block_size,
            cfg.atom_size,
            seq_len_kv,
        )
        required_route_count = (
            retained_atom_count + cutlass.Int32(cfg.logical_origins_per_route - 1)
        ) // cutlass.Int32(cfg.logical_origins_per_route)
        route_workspace[linear_row_idx] = required_route_count
    row_route_begin = _warp_broadcast_i32(row_route_begin, 0)
    required_route_count = _warp_broadcast_i32(required_route_count, 0)
    return required_route_count, row_route_begin


@cute.jit
def _store_prepared_route_validity(
    block_indices: cute.Tensor,
    kv_valid_bits: cute.Tensor,
    route_workspace: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    route_idx: cutlass.Int32,
    batch_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    logical_origin: cutlass.Int32,
    logical_origin_is_valid: cutlass.Boolean,
    stored_atom_is_full: cutlass.Boolean,
    route_metadata_word_index: cutlass.Int32,
    seq_len_kv: cutlass.Int32,
    cfg: cutlass.Constexpr[_PreparedRouteConfig],
) -> None:
    """Store storage-independent atom, token, and route validity metadata."""

    stored_atom_valid_mask = cutlass.Int32(
        cute.arch.vote_ballot_sync(logical_origin_is_valid)
    )
    structural_route_is_full = cute.arch.vote_all_sync(
        lane_idx >= cutlass.Int32(cfg.logical_origins_per_route) or stored_atom_is_full
    )
    route_is_full = structural_route_is_full
    if cutlass.const_expr(cfg.has_token_bits):
        token_word = cutlass.Uint32(0)
        if cutlass.const_expr(cfg.atom_size <= 32):
            token_chunk = _load_atom_token_chunk(
                kv_valid_bits,
                batch_idx,
                logical_origin,
                logical_origin_is_valid,
                cfg.atom_size,
                seq_len_kv,
            )
            atoms_per_word = 32 // cfg.atom_size
            if lane_idx < cutlass.Int32(cfg.logical_origins_per_route):
                atom_in_word = lane_idx % cutlass.Int32(atoms_per_word)
                token_word = token_chunk << (
                    atom_in_word * cutlass.Int32(cfg.atom_size)
                )
                active_origin_lanes = (1 << cfg.logical_origins_per_route) - 1
                for shuffle_step in cutlass.range_constexpr(
                    int(math.log2(atoms_per_word))
                ):
                    peer_word = cutlass.Uint32(
                        prims.shfl_sync(
                            thread_mask=active_origin_lanes,
                            val=token_word,
                            offset=1 << shuffle_step,
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.BFLY,
                        )
                    )
                    token_word = token_word | peer_word
                if atom_in_word == cutlass.Int32(0):
                    logical_word_idx = lane_idx // cutlass.Int32(atoms_per_word)
                    route_workspace[
                        route_metadata_word_index
                        + cutlass.Int32(cfg.token_words_word_offset)
                        + logical_word_idx
                    ] = cutlass.Int32(token_word)
            full_atom_mask = cutlass.Uint32((1 << cfg.atom_size) - 1)
            token_route_is_full = cute.arch.vote_all_sync(
                lane_idx >= cutlass.Int32(cfg.logical_origins_per_route)
                or token_chunk == full_atom_mask
            )
        else:
            if lane_idx < cutlass.Int32(cfg.token_words_per_route):
                token_word = _load_coarse_token_word(
                    block_indices,
                    kv_valid_bits,
                    row_begin,
                    row_end,
                    route_idx,
                    lane_idx,
                    batch_idx,
                    cfg.kv_block_size,
                    cfg.atom_size,
                    cfg.logical_origins_per_route,
                    seq_len_kv,
                )
                route_workspace[
                    route_metadata_word_index
                    + cutlass.Int32(cfg.token_words_word_offset)
                    + lane_idx
                ] = cutlass.Int32(token_word)
            token_route_is_full = cute.arch.vote_all_sync(
                lane_idx >= cutlass.Int32(cfg.token_words_per_route)
                or token_word == cutlass.Uint32(0xFFFFFFFF)
            )
        route_is_full = cutlass.Boolean(
            structural_route_is_full and token_route_is_full
        )

    if lane_idx == cutlass.Int32(0):
        route_workspace[
            route_metadata_word_index + cutlass.Int32(cfg.atom_valid_mask_word_offset)
        ] = stored_atom_valid_mask
        route_workspace[
            route_metadata_word_index + cutlass.Int32(cfg.route_flags_word_offset)
        ] = (
            cutlass.Int32(_PREPARED_ROUTE_IS_FULL_FLAG)
            if route_is_full
            else cutlass.Int32(0)
        )


@cute.jit
def _paged_request_page_range_is_valid(
    request_begin: cutlass.Int32,
    request_end: cutlass.Int32,
    num_indices: cutlass.Int32,
    required_pages: cutlass.Int32,
) -> cutlass.Boolean:
    """Validate one request's page-table range before any index load."""

    return cutlass.Boolean(
        request_begin >= cutlass.Int32(0)
        and request_begin <= request_end
        and request_end <= num_indices
        and request_end - request_begin >= required_pages
    )


@cute.jit
def _resolve_paged_route_atom_page_id(
    paged_kv_indices: cute.Tensor,
    request_begin: cutlass.Int32,
    logical_origin: cutlass.Int32,
    logical_origin_is_valid: cutlass.Boolean,
    lane_idx: cutlass.Int32,
    page_size: cutlass.Constexpr[int],
    num_physical_kv_pages: cutlass.Int64,
) -> cutlass.Int32:
    """Resolve one trusted selected logical atom to its raw physical page ID."""

    physical_page_id = cutlass.Int32(-1)
    page_id_is_valid = cutlass.Boolean(True)
    if logical_origin_is_valid:
        logical_page_idx = logical_origin // cutlass.Int32(page_size)
        candidate_page_id = cutlass.Int32(
            paged_kv_indices[request_begin + logical_page_idx]
        )
        physical_page_id = candidate_page_id
        page_id_is_valid = cutlass.Boolean(
            candidate_page_id >= cutlass.Int32(0)
            and cutlass.Int64(candidate_page_id) < num_physical_kv_pages
        )
    page_ids_are_valid = cute.arch.vote_all_sync(page_id_is_valid)
    if lane_idx == cutlass.Int32(0):
        runtime_assert(
            page_ids_are_valid,
            "paged_kv_indices contains an out-of-range physical page ID",
        )
    return physical_page_id


class _PrepareBlockSparseRoutes:
    """Prepare contiguous or paged sparse routes for one static geometry."""

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
        has_token_bits: bool,
        page_size: int | None = None,
        mask_type: str,
    ) -> None:
        if mask_type not in ("dense", "causal"):
            raise ValueError(f"unsupported mask_type: {mask_type}")
        num_q_block_rows = (seq_len_q + q_block_size - 1) // q_block_size
        num_rows = batch_size * num_kv_heads * num_q_block_rows
        layout = _BlockSparseRouteLayout.create(
            kv_route_size=kv_route_size,
            kv_block_size=kv_block_size,
            page_size=page_size,
            has_token_bits=has_token_bits,
            route_metadata_capacity=0,
            num_rows=num_rows,
        )
        self.cfg = _PreparedRouteConfig.create(
            layout=layout,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
        )
        self.route_layout = layout
        self.page_size = page_size if page_size is not None else 1
        self.minimum_seq_len_kv = seq_len_q if mask_type == "causal" else 1
        self.physical_page_ids_word_offset = (
            layout.physical_page_ids_word_offset if layout.is_paged else 0
        )
        self.route_metadata_base_word_offset = layout.route_metadata_base_word_offset

    @cute.jit
    def __call__(
        self,
        block_indptr: cute.Tensor,
        block_indices: cute.Tensor,
        kv_valid_bits: cute.Tensor,
        seq_lens_kv: cute.Tensor | None,
        paged_kv_indptr: cute.Tensor | None,
        paged_kv_indices: cute.Tensor | None,
        num_physical_kv_pages: cutlass.Int64,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
        stream: cuda_drv.CUstream,
    ) -> None:
        """Launch four independent row preparers per CTA."""

        self.kernel(
            block_indptr,
            block_indices,
            kv_valid_bits,
            seq_lens_kv,
            paged_kv_indptr,
            paged_kv_indices,
            num_physical_kv_pages,
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
        paged_kv_indptr: cute.Tensor | None,
        paged_kv_indices: cute.Tensor | None,
        num_physical_kv_pages: cutlass.Int64,
        row_route_offsets: cute.Tensor,
        route_workspace: cute.Tensor,
        max_blocks_per_row: cutlass.Int32,
    ) -> None:
        """Pack logical routes and, when paged, translate physical locators."""

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

        request_begin = cutlass.Int32(0)
        runtime_seq_len_kv = cutlass.Int32(self.cfg.seq_len_kv)
        if cutlass.const_expr(self.route_layout.is_paged):
            raw_seq_len_kv = cutlass.Int32(self.cfg.seq_len_kv)
            if lane_idx == cutlass.Int32(0) and row_is_valid:
                raw_seq_len_kv = cutlass.Int32(seq_lens_kv[batch_idx])
                runtime_assert(
                    raw_seq_len_kv >= cutlass.Int32(self.minimum_seq_len_kv)
                    and raw_seq_len_kv <= cutlass.Int32(self.cfg.seq_len_kv),
                    "seq_lens_kv is outside the planned length range",
                )
            raw_seq_len_kv = _warp_broadcast_i32(raw_seq_len_kv, 0)
            runtime_seq_len_kv = raw_seq_len_kv

            if lane_idx == cutlass.Int32(0) and row_is_valid:
                required_pages = _positive_i32_ceil_div(
                    runtime_seq_len_kv,
                    self.page_size,
                )
                request_begin = cutlass.Int32(paged_kv_indptr[batch_idx])
                request_end = cutlass.Int32(
                    paged_kv_indptr[batch_idx + cutlass.Int32(1)]
                )
                metadata_starts_at_zero = cutlass.Boolean(
                    paged_kv_indptr[cutlass.Int32(0)] == cutlass.Int32(0)
                )
                runtime_assert(
                    metadata_starts_at_zero
                    and _paged_request_page_range_is_valid(
                        request_begin,
                        request_end,
                        cutlass.Int32(cute.size(paged_kv_indices)),
                        required_pages,
                    ),
                    "paged_kv_indptr row lacks the required active page capacity",
                )
            request_begin = _warp_broadcast_i32(request_begin, 0)

        route_count, row_route_begin = _publish_prepared_route_count(
            block_indices,
            row_route_offsets,
            route_workspace,
            row_begin,
            row_end,
            linear_row_idx,
            lane_idx,
            row_is_valid,
            max_blocks_per_row,
            runtime_seq_len_kv,
            self.cfg,
        )

        route_idx = cutlass.Int32(0)
        while route_idx < route_count:
            route_ordinal = row_route_begin + route_idx
            route_metadata_word_index = cutlass.Int32(
                self.cfg.route_metadata_base_word_offset
            ) + route_ordinal * cutlass.Int32(self.cfg.route_metadata_stride_words)
            logical_origin = cutlass.Int32(-1)
            logical_origin_is_valid = cutlass.Boolean(False)
            physical_page_id = cutlass.Int32(-1)
            atom_is_full = cutlass.Boolean(False)
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
                    runtime_seq_len_kv,
                )
            if cutlass.const_expr(self.route_layout.is_paged):
                physical_page_id = _resolve_paged_route_atom_page_id(
                    paged_kv_indices,
                    request_begin,
                    logical_origin,
                    logical_origin_is_valid,
                    lane_idx,
                    self.page_size,
                    num_physical_kv_pages,
                )
            if logical_origin_is_valid:
                atom_is_full = cutlass.Boolean(
                    logical_origin
                    <= runtime_seq_len_kv - cutlass.Int32(self.cfg.atom_size)
                )
            if lane_idx < cutlass.Int32(self.cfg.logical_origins_per_route):
                route_workspace[route_metadata_word_index + lane_idx] = logical_origin
                if cutlass.const_expr(self.route_layout.is_paged):
                    route_workspace[
                        route_metadata_word_index
                        + cutlass.Int32(self.physical_page_ids_word_offset)
                        + lane_idx
                    ] = physical_page_id

            _store_prepared_route_validity(
                block_indices,
                kv_valid_bits,
                route_workspace,
                row_begin,
                row_end,
                route_idx,
                batch_idx,
                lane_idx,
                logical_origin,
                logical_origin_is_valid,
                atom_is_full,
                route_metadata_word_index,
                runtime_seq_len_kv,
                self.cfg,
            )
            route_idx = route_idx + cutlass.Int32(1)
