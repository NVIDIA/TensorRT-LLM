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

    # Bind source tensors and optional cached row lengths; capacity counts padded KV
    # slots.
    # assume_valid_prefix promises nonnegative indices within each live prefix;
    # has_extra enables the second source, whose tile domain follows the primary span.
    def __init__(self, values, capacity, assume_valid_prefix=False, has_extra=True):
        """Carry caller-owned source metadata and an optional FP8-primary scale through JIT calls."""
        self.values = (*values, capacity)
        (
            self.primary_indices,
            self.extra_indices,
            self.primary_lengths,
            self.extra_lengths,
            self.query_row,
            self.primary_length,
            self.extra_length,
            self.primary_tensor_scale,
        ) = values
        self.capacity = capacity
        self.assume_valid_prefix = assume_valid_prefix
        self.has_extra = has_extra
        self.is_bound = self.query_row is not None
        if self.is_bound:
            self.primary_span = (
                (self.primary_length + Int32(127)) // Int32(128) * Int32(128)
            )

    # Flatten source tensors, row state and runtime capacity for CuTe calls.
    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    # Rebind source values and live capacity while preserving format promises.
    def __new_from_mlir_values__(self, values):
        rebound = cutlass.new_from_mlir_values(self.values, values)
        return SparseKVIndices(
            rebound[:-1],
            rebound[-1],
            self.assume_valid_prefix,
            self.has_extra,
        )

    # Load one query row's lengths once, or use the supplied token-count overrides.
    @cute.jit
    def bind_query_row(self, query_row, primary_length=None, extra_length=None):
        """Bind one query to live source lengths, optionally reusing register-cached lengths.

        Round the primary span to 128 slots so a BK64/BK128 tile never mixes
        pools. Padding is logical: caller index arrays and cache rows are not copied."""
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
                self.primary_tensor_scale,
            ),
            self.capacity,
            self.assume_valid_prefix,
            self.has_extra,
        )

    # Select a source at a logical KV-token offset; for_mask may omit index loads.
    @cute.jit
    def tensor_scale_is_extra(self, tile_start, query_row=None):
        """Return whether a logical KV tile starts in the extra pool.

        The primary list occupies ceil(primary_length/128)*128 logical slots.
        Later slots select extra NVFP4 rows and therefore have tensor scale one.
        Use the current graph-live length, not the planned list capacity."""
        if cutlass.const_expr(not self.has_extra):
            return cutlass.Boolean(False)
        length = (
            self.primary_length
            if cutlass.const_expr(self.is_bound)
            else Int32(self.primary_lengths[Int64(query_row)])
        )
        return tile_start >= (length + Int32(127)) // Int32(128) * Int32(128)

    @cute.jit
    def tensor_scale(self, tile_start, query_row=None):
        """Return K/V's descale for this tile: primary FP8 scale or one for NVFP4.

        NVFP4 block scales have already been applied in the loader. The FP8
        primary bytes remain unchanged, so score/output arithmetic consumes
        the host-runtime FP32 scalar instead of rounding scaled K/V back into E4M3.
        """
        scale = cutlass.Float32(1)
        if not self.tensor_scale_is_extra(tile_start, query_row):
            scale = cutlass.Float32(self.primary_tensor_scale)
        return scale

    @cute.jit
    def tensor_scale_ratio(self, current, previous, query_row=None):
        """Convert accumulated O from the previous tile's V units to the current's.

        PV uses raw FP8 primary values and already-decoded NVFP4 extra values.
        Therefore O_new = alpha * (s_previous / s_current) * O_old + P @ V_raw,
        where alpha is the ordinary online-softmax correction. Primary tiles
        precede extra tiles, so the only transition multiplies O by the FP8
        primary scale (the extra scale is one). Stay exactly one within a pool.
        This factor changes O only; the softmax denominator and LSE use alpha.
        """
        ratio = cutlass.Float32(1)
        if cutlass.const_expr(self.has_extra):
            if self.tensor_scale_is_extra(
                current, query_row
            ) and not self.tensor_scale_is_extra(previous, query_row):
                ratio = cutlass.Float32(self.primary_tensor_scale)
        return ratio

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

    # Bind one source's indices, token origin/length and descriptor tag; valid_prefix
    # skips index validity checks.
    def __init__(self, values, valid_prefix=False):
        self.values = values
        self.indices, self.origin, self.length, self.tag = values
        self.valid_prefix = valid_prefix

    # Flatten the selected source pointer and token bounds for CuTe calls.
    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    # Rebind a source tile while retaining its valid-prefix promise.
    def __new_from_mlir_values__(self, values):
        return SparseSourceTile(
            cutlass.new_from_mlir_values(self.values, values), self.valid_prefix
        )

    # Clamp an index load inside source storage and return its separate live-token
    # validity.
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

    # Return one physical row with its source tag, or the tagged invalid-row sentinel.
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

    # Test one logical KV token; request is unused because the source is row-bound.
    @cute.jit
    def is_invalid(self, token, request):
        del request
        if cutlass.const_expr(self.valid_prefix):
            return (token < self.origin) | (token >= self.origin + self.length)
        _, valid = self._bounded_index(token)
        return ~valid

    # Check that a 128-token tile fits both the selected source and live execution span.
    @cute.jit
    def tile_is_full(self, start, live_length):
        return start + Int32(128) <= cute.math.min(
            self.origin + self.length, live_length
        )

    # Return one invalid-bit mask for 32 tokens starting within the selected source.
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

    # Bind per-query lengths or a cached scalar execution span; has_extra selects the
    # second pool.
    def __init__(self, values, has_extra=True):
        self.values = values
        self.has_extra = has_extra
        self.primary_lengths, self.extra_lengths, self.bound_length, rows = values
        self.shape = (
            (rows,) if self.bound_length is not None else self.primary_lengths.shape
        )

    # Flatten scheduler length tensors or the bound scalar for CuTe calls.
    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    # Rebind scheduler lengths while retaining the source-count contract.
    def __new_from_mlir_values__(self, values):
        return SparseKVLengths(
            cutlass.new_from_mlir_values(self.values, values), self.has_extra
        )

    # Return the padded execution span in tokens for one virtual query request.
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


# Round each source to 128-token tiles; empty rows still execute one masked token.
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
    capacity: Int32,
    *,
    query_row=None,
    assume_valid_prefix: cutlass.Constexpr[bool] = False,
    has_extra: cutlass.Constexpr[bool] = True,
    primary_tensor_scale=None,
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
            primary_tensor_scale,
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


@cute.jit
def cache_sparse_source_lengths(
    page_offsets,
    query_row,
    lengths,
    *,
    include_primary_tiles: cutlass.Constexpr[bool] = False,
):
    """Snapshot graph-live source lengths once per scheduled query work tile.

    Bound readers already carry scalar lengths. Unbound persistent readers
    cache primary/extra lengths in registers; loaders can additionally retain
    ceil(primary_length/128), the source boundary in full KV tiles.
    """
    if cutlass.const_expr(not page_offsets.is_bound):
        lengths[0] = Int32(page_offsets.primary_lengths[query_row])
        lengths[1] = (
            Int32(page_offsets.extra_lengths[query_row])
            if page_offsets.has_extra
            else Int32(0)
        )
        if cutlass.const_expr(include_primary_tiles):
            lengths[2] = (lengths[0] + Int32(127)) // Int32(128)
