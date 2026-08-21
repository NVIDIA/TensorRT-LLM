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

"""Prepare live canonical BSR rows for the PrimTS FMHA route consumer.

The kernel converts caller-owned semantic KV blocks into fixed-stride route
metadata on every run. Route origins are logical KV-token coordinates;
the paged specialization also resolves each origin to a physical page ID for
the attention load path. One warp handles one BSR row and iterates only that
row's live routes, while four warps share a CTA.

``row_route_offsets`` is a separate plan-owned immutable Int32 tensor.
``route_workspace`` contains only mutable row counts and route metadata
described by ``_BlockSparseRouteLayout``. Payload outside each live row
count is intentionally stale. ``max_blocks_per_row`` is a run-time Int32
scalar: ``-1`` disables the semantic bound, while a non-negative value keeps a
declared BSR-block limit distinct from packed-route capacity.
"""

import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_drv
from cutlass.experimental import primitives as prims

from ..._block_sparse.common import _SIGNED_INT32_MAX
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
    num_kv_valid_words: int
    route_metadata_stride_words: int
    route_metadata_base_word_offset: int
    route_capacity_block_scale_num: int
    route_capacity_block_scale_den: int

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
        capacity_gcd = math.gcd(layout.kv_route_size, kv_block_size)
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
            num_kv_valid_words=(seq_len_kv + 31) // 32,
            route_metadata_stride_words=layout.route_metadata_stride_words,
            route_metadata_base_word_offset=layout.route_metadata_base_word_offset,
            route_capacity_block_scale_num=layout.kv_route_size // capacity_gcd,
            route_capacity_block_scale_den=kv_block_size // capacity_gcd,
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
        num_kv_blocks = _positive_i32_ceil_div(seq_len_kv, kv_block_size)
        retained_last_atoms = cutlass.Int32(atoms_per_block)
        last_block_is_in_range = cutlass.Boolean(
            last_block_idx >= cutlass.Int32(0)
            and last_block_idx < cutlass.Int32(num_kv_blocks)
        )
        if last_block_is_in_range:
            last_block_origin = last_block_idx * cutlass.Int32(kv_block_size)
            remaining_tokens = cutlass.Int32(seq_len_kv) - last_block_origin
            retained_last_atoms = cutlass.Int32(0)
            if remaining_tokens > cutlass.Int32(0):
                retained_last_atoms = (
                    remaining_tokens - cutlass.Int32(1)
                ) // cutlass.Int32(atom_size) + cutlass.Int32(1)
                if retained_last_atoms > cutlass.Int32(atoms_per_block):
                    retained_last_atoms = cutlass.Int32(atoms_per_block)
        retained_atoms = retained_atoms + retained_last_atoms
    return retained_atoms


@cute.jit
def _bsr_row_fits_live_k(
    block_indices: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    kv_block_size: cutlass.Constexpr[int],
    seq_len_kv: cutlass.Int32,
) -> cutlass.Boolean:
    """Check a canonical sorted row's live bound from its final block."""

    row_fits = cutlass.Boolean(True)
    if row_end > row_begin:
        last_block_idx = cutlass.Int32(block_indices[row_end - cutlass.Int32(1)])
        last_block_origin = last_block_idx * cutlass.Int32(kv_block_size)
        row_fits = cutlass.Boolean(last_block_origin < seq_len_kv)
    return row_fits


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
        num_kv_blocks = _positive_i32_ceil_div(seq_len_kv, kv_block_size)
        valid = cutlass.Boolean(
            block_idx >= cutlass.Int32(0) and block_idx < cutlass.Int32(num_kv_blocks)
        )
        if valid:
            block_origin = block_idx * cutlass.Int32(kv_block_size)
            atom_offset = atom_in_block * cutlass.Int32(atom_size)
            valid = cutlass.Boolean(
                atom_offset < cutlass.Int32(seq_len_kv) - block_origin
            )
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
    num_kv_valid_words: cutlass.Int32,
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
    if (
        valid
        and logical_word_origin >= cutlass.Int32(0)
        and logical_word_origin < cutlass.Int32(seq_len_kv)
    ):
        valid_bits_word_idx = logical_word_origin >> cutlass.Int32(5)
        if valid_bits_word_idx >= cutlass.Int32(
            0
        ) and valid_bits_word_idx < cutlass.Int32(num_kv_valid_words):
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
    num_kv_valid_words: cutlass.Int32,
) -> cutlass.Uint32:
    """Load the <=K32 mask chunk owned by one resolved-origin lane."""

    token_chunk = cutlass.Uint32(0)
    if origin_is_valid and logical_origin >= cutlass.Int32(0):
        valid_bits_word_idx = logical_origin >> cutlass.Int32(5)
        if valid_bits_word_idx >= cutlass.Int32(
            0
        ) and valid_bits_word_idx < cutlass.Int32(num_kv_valid_words):
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
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32, cutlass.Boolean]:
    """Resolve and validate one canonical runtime BSR row."""

    row_begin = cutlass.Int32(0)
    row_end = cutlass.Int32(0)
    batch_idx = cutlass.Int32(0)
    row_range_is_valid = cutlass.Int32(0)
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
        row_range_is_valid = cutlass.Int32(
            row_begin >= cutlass.Int32(0)
            and row_begin <= row_end
            and row_end <= num_indices
        )
    row_begin = _warp_broadcast_i32(row_begin, 0)
    row_end = _warp_broadcast_i32(row_end, 0)
    batch_idx = _warp_broadcast_i32(batch_idx, 0)
    row_range_is_valid = _warp_broadcast_i32(row_range_is_valid, 0)

    row_error_code = cutlass.Int32(0)
    if row_is_valid and row_range_is_valid != cutlass.Int32(0):
        row_error_code = _validate_bsr_row_lane(
            block_indices,
            row_begin,
            row_end,
            lane_idx,
            cfg.num_kv_blocks,
        )
    row_error_code = cutlass.Int32(cute.arch.warp_redux_sync(row_error_code, "max"))
    row_is_canonical = cutlass.Boolean(
        row_range_is_valid != cutlass.Int32(0) and row_error_code == cutlass.Int32(0)
    )
    return row_begin, row_end, batch_idx, row_is_canonical


@cute.jit
def _publish_prepared_route_count(
    block_indices: cute.Tensor,
    row_route_offsets: cute.Tensor,
    route_workspace: cute.Tensor,
    row_begin: cutlass.Int32,
    row_end: cutlass.Int32,
    row_is_canonical: cutlass.Boolean,
    linear_row_idx: cutlass.Int32,
    lane_idx: cutlass.Int32,
    row_is_valid: cutlass.Boolean,
    max_blocks_per_row: cutlass.Int32,
    seq_len_kv: cutlass.Int32,
    cfg: cutlass.Constexpr[_PreparedRouteConfig],
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Validate row capacity, publish its header, and return its live span."""

    route_count = cutlass.Int32(0)
    row_route_begin = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0) and row_is_valid:
        row_route_begin = cutlass.Int32(row_route_offsets[linear_row_idx])
        row_route_end = cutlass.Int32(
            row_route_offsets[linear_row_idx + cutlass.Int32(1)]
        )
        row_route_capacity = row_route_end - row_route_begin
        # An invalid row retains this negative header and emits no payload;
        # the attention consumer clamps it to an empty row after synchronization.
        stored_route_count = cutlass.Int32(-1)
        if row_is_canonical:
            selected_block_count = row_end - row_begin
            route_block_capacity = (
                row_route_capacity * cutlass.Int32(cfg.route_capacity_block_scale_num)
            ) // cutlass.Int32(cfg.route_capacity_block_scale_den)
            block_count_fits = cutlass.Boolean(
                selected_block_count <= route_block_capacity
                and (
                    max_blocks_per_row < cutlass.Int32(0)
                    or selected_block_count <= max_blocks_per_row
                )
            )
            if block_count_fits:
                retained_atom_count = _retained_atom_count(
                    block_indices,
                    row_begin,
                    row_end,
                    cfg.kv_block_size,
                    cfg.atom_size,
                    seq_len_kv,
                )
                required_route_count = (
                    retained_atom_count
                    + cutlass.Int32(cfg.logical_origins_per_route - 1)
                ) // cutlass.Int32(cfg.logical_origins_per_route)
                if required_route_count <= row_route_capacity:
                    route_count = required_route_count
                    stored_route_count = required_route_count
                else:
                    # Keep a capacity violation visible without an OOB write.
                    stored_route_count = -required_route_count
            else:
                # Preserve the semantic BSR-block bound even when fine blocks
                # pack several entries into one prepared route.
                stored_route_count = -selected_block_count
        route_workspace[linear_row_idx] = stored_route_count
    route_count = _warp_broadcast_i32(route_count, 0)
    row_route_begin = _warp_broadcast_i32(row_route_begin, 0)
    return route_count, row_route_begin


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
    stored_atom_is_valid: cutlass.Boolean,
    stored_atom_is_full: cutlass.Boolean,
    route_metadata_word_index: cutlass.Int32,
    seq_len_kv: cutlass.Int32,
    num_kv_valid_words: cutlass.Int32,
    cfg: cutlass.Constexpr[_PreparedRouteConfig],
) -> cutlass.Int32:
    """Store storage-independent atom, token, and route validity metadata."""

    stored_atom_valid_mask = cutlass.Int32(
        cute.arch.vote_ballot_sync(stored_atom_is_valid)
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
                num_kv_valid_words,
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
                    num_kv_valid_words,
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
    return stored_atom_valid_mask


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
    request_page_count: cutlass.Int32,
    logical_origin: cutlass.Int32,
    logical_origin_is_valid: cutlass.Boolean,
    page_size: cutlass.Constexpr[int],
    num_physical_kv_pages: cutlass.Int64,
) -> tuple[cutlass.Int32, cutlass.Boolean]:
    """Resolve one selected logical atom without an out-of-range index load."""

    physical_page_id = cutlass.Int32(-1)
    page_id_is_valid = cutlass.Boolean(False)
    if logical_origin_is_valid and logical_origin >= cutlass.Int32(0):
        logical_page_idx = logical_origin // cutlass.Int32(page_size)
        if (
            logical_page_idx >= cutlass.Int32(0)
            and logical_page_idx < request_page_count
        ):
            candidate_page_id = cutlass.Int64(
                paged_kv_indices[request_begin + logical_page_idx]
            )
            page_id_is_valid = cutlass.Boolean(
                candidate_page_id >= cutlass.Int64(0)
                and candidate_page_id < num_physical_kv_pages
                and candidate_page_id <= cutlass.Int64(_SIGNED_INT32_MAX)
            )
            if page_id_is_valid:
                physical_page_id = cutlass.Int32(candidate_page_id)
    return physical_page_id, page_id_is_valid


@cute.jit
def _invalid_paged_route_count(route_count: cutlass.Int32) -> cutlass.Int32:
    """Encode a fail-closed paged row without the negative-zero ambiguity."""

    return -route_count - cutlass.Int32(1)


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
        mask_type: str = "dense",
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

        row_begin, row_end, batch_idx, row_is_canonical = _resolve_prepared_bsr_row(
            block_indptr,
            block_indices,
            linear_row_idx,
            lane_idx,
            row_is_valid,
            self.cfg,
        )

        request_begin = cutlass.Int32(0)
        live_seq_len_kv = cutlass.Int32(self.cfg.seq_len_kv)
        request_page_count = cutlass.Int32(0)
        live_num_kv_valid_words = cutlass.Int32(self.cfg.num_kv_valid_words)
        if cutlass.const_expr(self.route_layout.is_paged):
            required_pages = cutlass.Int32(0)
            request_page_range_is_valid = cutlass.Int32(0)
            row_fits_live_k = cutlass.Int32(0)
            live_seq_len_is_valid = cutlass.Int32(0)
            if lane_idx == cutlass.Int32(0) and row_is_valid:
                raw_seq_len_kv = cutlass.Int32(seq_lens_kv[batch_idx])
                live_seq_len_kv = cutlass.Int32(self.minimum_seq_len_kv)
                live_seq_len_is_valid = cutlass.Int32(
                    raw_seq_len_kv >= cutlass.Int32(self.minimum_seq_len_kv)
                    and raw_seq_len_kv <= cutlass.Int32(self.cfg.seq_len_kv)
                )
                if live_seq_len_is_valid != cutlass.Int32(0):
                    live_seq_len_kv = raw_seq_len_kv

                metadata_starts_at_zero = cutlass.Boolean(
                    paged_kv_indptr[cutlass.Int32(0)] == cutlass.Int32(0)
                )
                if (
                    live_seq_len_is_valid != cutlass.Int32(0)
                    and metadata_starts_at_zero
                ):
                    required_pages = _positive_i32_ceil_div(
                        live_seq_len_kv,
                        self.page_size,
                    )
                    if row_is_canonical and _bsr_row_fits_live_k(
                        block_indices,
                        row_begin,
                        row_end,
                        self.cfg.kv_block_size,
                        live_seq_len_kv,
                    ):
                        row_fits_live_k = cutlass.Int32(1)
                        page_indptr_size = cutlass.Int32(cute.size(paged_kv_indptr))
                        page_indptr_entry_is_valid = cutlass.Boolean(
                            batch_idx >= cutlass.Int32(0)
                            and batch_idx + cutlass.Int32(1) < page_indptr_size
                        )
                        if page_indptr_entry_is_valid:
                            request_begin = cutlass.Int32(paged_kv_indptr[batch_idx])
                            request_end = cutlass.Int32(
                                paged_kv_indptr[batch_idx + cutlass.Int32(1)]
                            )
                            num_page_indices = cutlass.Int32(
                                cute.size(paged_kv_indices)
                            )
                            request_page_range_is_valid = cutlass.Int32(
                                _paged_request_page_range_is_valid(
                                    request_begin,
                                    request_end,
                                    num_page_indices,
                                    required_pages,
                                )
                            )
                            if request_page_range_is_valid != cutlass.Int32(0):
                                request_page_count = required_pages
            request_begin = _warp_broadcast_i32(request_begin, 0)
            live_seq_len_kv = _warp_broadcast_i32(live_seq_len_kv, 0)
            request_page_count = _warp_broadcast_i32(request_page_count, 0)
            row_fits_live_k = _warp_broadcast_i32(row_fits_live_k, 0)
            live_seq_len_is_valid = _warp_broadcast_i32(
                live_seq_len_is_valid,
                0,
            )
            request_page_range_is_valid = _warp_broadcast_i32(
                request_page_range_is_valid,
                0,
            )
            row_is_canonical = cutlass.Boolean(
                row_is_canonical
                and live_seq_len_is_valid != cutlass.Int32(0)
                and row_fits_live_k != cutlass.Int32(0)
                and request_page_range_is_valid != cutlass.Int32(0)
            )
            live_num_kv_valid_words = _positive_i32_ceil_div(live_seq_len_kv, 32)

        route_count, row_route_begin = _publish_prepared_route_count(
            block_indices,
            row_route_offsets,
            route_workspace,
            row_begin,
            row_end,
            row_is_canonical,
            linear_row_idx,
            lane_idx,
            row_is_valid,
            max_blocks_per_row,
            live_seq_len_kv,
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
            load_origin_is_valid = cutlass.Boolean(False)
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
                    live_seq_len_kv,
                )
                load_origin_is_valid = logical_origin_is_valid
                if cutlass.const_expr(self.route_layout.is_paged):
                    (
                        physical_page_id,
                        load_origin_is_valid,
                    ) = _resolve_paged_route_atom_page_id(
                        paged_kv_indices,
                        request_begin,
                        request_page_count,
                        logical_origin,
                        logical_origin_is_valid,
                        self.page_size,
                        num_physical_kv_pages,
                    )
                if load_origin_is_valid:
                    atom_is_full = cutlass.Boolean(
                        logical_origin
                        <= live_seq_len_kv - cutlass.Int32(self.cfg.atom_size)
                    )
                route_workspace[route_metadata_word_index + lane_idx] = logical_origin
                if cutlass.const_expr(self.route_layout.is_paged):
                    route_workspace[
                        route_metadata_word_index
                        + cutlass.Int32(self.physical_page_ids_word_offset)
                        + lane_idx
                    ] = physical_page_id

            if cutlass.const_expr(self.route_layout.is_paged):
                logical_origin_valid_mask = cutlass.Int32(
                    cute.arch.vote_ballot_sync(logical_origin_is_valid)
                )
            stored_atom_valid_mask = _store_prepared_route_validity(
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
                load_origin_is_valid,
                atom_is_full,
                route_metadata_word_index,
                live_seq_len_kv,
                live_num_kv_valid_words,
                self.cfg,
            )

            if cutlass.const_expr(self.route_layout.is_paged):
                if lane_idx == cutlass.Int32(0):
                    # Page validity is a strict subset of logical validity. Any
                    # mismatch means a selected atom had a bad locator.
                    if stored_atom_valid_mask != logical_origin_valid_mask:
                        route_workspace[linear_row_idx] = _invalid_paged_route_count(
                            route_count
                        )
            route_idx = route_idx + cutlass.Int32(1)
