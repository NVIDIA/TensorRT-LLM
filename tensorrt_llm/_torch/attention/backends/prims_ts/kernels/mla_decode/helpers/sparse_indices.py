# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select a source index buffer from a tile's logical KV offset.

    row = indices.bind_query_row(query_row)
    tile = row.select_source_tile(tile_start)
    quad = tile.load_quad(kv_offset)

The primary source's logical span is rounded to 128 so BK64/BK128 tiles never
cross pools, even when physical rows are random. Caller arrays/live lengths
need no such padding; bounded loads synthesize invalid tails. A 32-entry
boundary would suffice for the grouped TMA descriptors alone, but not for
this whole-tile source selection. These compiler-side objects carry existing
pointers/scalars through CuTe calls; they allocate no merged metadata buffer.
Only SparseKVLengths retains tensor indexing for the shared scheduler.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64


class SparseKVIndices:
    """Caller-owned index lists; optionally cache one query row's live lengths."""

    def __init__(self, values, capacity, assume_valid_prefix=False, has_extra=True):
        self.values = values
        (
            self.primary_indices,
            self.extra_indices,
            self.primary_lengths,
            self.extra_lengths,
            self.query_row,
            self.primary_length,
            self.extra_length,
        ) = values
        self.capacity = capacity
        self.assume_valid_prefix = assume_valid_prefix
        self.has_extra = has_extra
        self.is_bound = self.query_row is not None
        if self.is_bound:
            self.primary_span = (
                (self.primary_length + Int32(127)) // Int32(128) * Int32(128)
            )

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseKVIndices(
            cutlass.new_from_mlir_values(self.values, values),
            self.capacity,
            self.assume_valid_prefix,
            self.has_extra,
        )

    @cute.jit
    def bind_query_row(self, query_row, primary_length=None, extra_length=None):
        if cutlass.const_expr(self.is_bound):
            return self
        row = Int64(query_row)
        # Once bound, carry scalar lengths instead of the two length tensors.
        return SparseKVIndices(
            (
                self.primary_indices,
                self.extra_indices,
                None,
                None,
                row,
                Int32(self.primary_lengths[row])
                if cutlass.const_expr(primary_length is None)
                else primary_length,
                (
                    Int32(self.extra_lengths[row])
                    if cutlass.const_expr(extra_length is None)
                    else extra_length
                )
                if cutlass.const_expr(self.has_extra)
                else Int32(0),
            ),
            self.capacity,
            self.assume_valid_prefix,
            self.has_extra,
        )

    @cute.jit
    def select_source_tile(
        self, tile_start, query_row=None, *, for_mask: cutlass.Constexpr[bool] = False
    ):
        row = self.bind_query_row(query_row)
        if cutlass.const_expr(not self.has_extra):
            # A single-pool plan needs no second length, pointer selection or tag.
            if cutlass.const_expr(for_mask and self.assume_valid_prefix):
                return SparseSourceTile(
                    (None, Int32(0), row.primary_length, None), valid_prefix=True
                )
            return SparseSourceTile(
                (
                    row.primary_indices[row.query_row, None],
                    Int32(0),
                    row.primary_length,
                    Int32(0),
                ),
                valid_prefix=self.assume_valid_prefix,
            )
        use_extra = tile_start >= row.primary_span
        origin = row.primary_span if use_extra else Int32(0)
        length = row.extra_length if use_extra else row.primary_length
        if cutlass.const_expr(for_mask and self.assume_valid_prefix):
            # Analytic masks need no index pointer or index loads.
            return SparseSourceTile((None, origin, length, None), valid_prefix=True)
        primary_ptr = row.primary_indices.iterator.toint() + row.query_row * Int64(
            row.primary_indices.stride[0]
        ) * Int64(4)
        extra_ptr = row.extra_indices.iterator.toint() + row.query_row * Int64(
            row.extra_indices.stride[0]
        ) * Int64(4)
        # Source tensors can have different layouts; select numeric addresses.
        address = extra_ptr if use_extra else primary_ptr
        indices = cute.make_tensor(
            cute.make_ptr(Int32, address, assumed_align=4),
            cute.make_layout((self.capacity,), stride=(1,)),
        )
        # The first coordinate chooses a shared TMA descriptor, even when
        # invalid. Preserve this source tag on every synthesized tail entry.
        tag = Int32(-2147483648) if use_extra else Int32(0)
        return SparseSourceTile(
            (indices, origin, length, tag), valid_prefix=self.assume_valid_prefix
        )


class SparseSourceTile:
    """Selected source and bounds; analytic prefix masks need no index pointer."""

    def __init__(self, values, valid_prefix=False):
        self.values = values
        self.indices, self.origin, self.length, self.tag = values
        self.valid_prefix = valid_prefix

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseSourceTile(
            cutlass.new_from_mlir_values(self.values, values), self.valid_prefix
        )

    @cute.jit
    def _bounded_index(self, token):
        position = token - self.origin
        # A zero-capacity source binds one dummy index. Even masked scalar
        # loads stay inside its backing; validity is determined separately.
        safe = cute.math.min(
            cute.math.max(position, Int32(0)),
            cute.math.max(self.length - Int32(1), Int32(0)),
        )
        index = Int32(self.indices[safe])
        valid = (position >= Int32(0)) & (position < self.length)
        if cutlass.const_expr(not self.valid_prefix):
            valid = valid & (index >= Int32(0))
        return index, valid

    @cute.jit
    def load_row(self, token):
        index, valid = self._bounded_index(token)
        return (index if valid else Int32(0x7FFFFFFF)) | self.tag

    @cute.jit
    def load_full_quad(self, token):
        """Load an in-bounds quad; callers establish its source-local bounds."""
        values = cutlass.Pointer(
            self.indices.iterator + token - self.origin, dtype=Int32
        ).load(count=4, alignment=4)
        rows = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(4):
            index = Int32(values[j])
            if cutlass.const_expr(self.valid_prefix):
                rows[j] = index | self.tag
            else:
                rows[j] = (index if index >= Int32(0) else Int32(0x7FFFFFFF)) | self.tag
        return rows

    @cute.jit
    def load_quad(self, token):
        """One vector load for an active quad; bounded scalar loads for tails."""
        rows = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(4):
            rows[j] = Int32(0x7FFFFFFF) | self.tag
        position = token - self.origin
        if position >= Int32(0) and position + Int32(3) < self.length:
            values = self.load_full_quad(token)
            for j in cutlass.range_constexpr(4):
                rows[j] = values[j]
        elif position + Int32(3) >= Int32(0) and position < self.length:
            for j in cutlass.range_constexpr(4):
                rows[j] = self.load_row(token + Int32(j))
        return rows

    @cute.jit
    def is_invalid(self, token, request):
        del request
        if cutlass.const_expr(self.valid_prefix):
            return (token < self.origin) | (token >= self.origin + self.length)
        _, valid = self._bounded_index(token)
        return ~valid

    @cute.jit
    def tile_is_full(self, start, live_length):
        return start + Int32(128) <= cute.math.min(
            self.origin + self.length, live_length
        )

    @cute.jit
    def prefix_mask32(self, start, live_length):
        # Source/tile boundaries are aligned; start is within this source.
        end = cute.math.min(self.origin + self.length, live_length)
        valid = cute.math.min(cute.math.max(end - start, Int32(0)), Int32(32))
        # Avoid an undefined shift by 32, including on an unselected SSA arm.
        bits = Int32(-1) << (valid & Int32(31))
        return bits if valid < Int32(32) else Int32(0)


class SparseKVLengths:
    """Adapter for the scheduler's lengths[query_row] interface only."""

    def __init__(self, values, has_extra=True):
        self.values = values
        self.has_extra = has_extra
        self.primary_lengths, self.extra_lengths, self.bound_length, rows = values
        self.shape = (
            (rows,) if self.bound_length is not None else self.primary_lengths.shape
        )

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return SparseKVLengths(
            cutlass.new_from_mlir_values(self.values, values), self.has_extra
        )

    @cute.jit
    def __getitem__(self, query_row):
        if cutlass.const_expr(self.bound_length is not None):
            return self.bound_length
        return _sparse_execution_length(
            Int32(self.primary_lengths[query_row]),
            Int32(self.extra_lengths[query_row])
            if cutlass.const_expr(self.has_extra)
            else Int32(0),
        )


@cute.jit
def _sparse_execution_length(primary_length, extra_length):
    primary_tiles = (primary_length + Int32(127)) // Int32(128)
    extra_tiles = (extra_length + Int32(127)) // Int32(128)
    # Empty rows still execute a masked tile to publish O=0 and LSE=-inf.
    return cute.math.max((primary_tiles + extra_tiles) * Int32(128), Int32(1))


@cute.jit
def bind_sparse_kv_indices(
    primary_indices,
    extra_indices,
    primary_lengths,
    extra_lengths,
    capacity: cutlass.Constexpr[int],
    *,
    query_row=None,
    assume_valid_prefix: cutlass.Constexpr[bool] = False,
    has_extra: cutlass.Constexpr[bool] = True,
):
    """Bind index readers and scheduler lengths without copying metadata."""
    indices = SparseKVIndices(
        (
            primary_indices,
            extra_indices,
            primary_lengths,
            extra_lengths,
            None,
            None,
            None,
        ),
        capacity,
        assume_valid_prefix,
        has_extra,
    )
    if cutlass.const_expr(query_row is not None):
        indices = indices.bind_query_row(query_row)
        length = _sparse_execution_length(indices.primary_length, indices.extra_length)
        lengths = SparseKVLengths(
            (None, None, length, primary_lengths.shape[0]), has_extra
        )
    else:
        lengths = SparseKVLengths(
            (primary_lengths, extra_lengths, None, None), has_extra
        )
    return indices, lengths
