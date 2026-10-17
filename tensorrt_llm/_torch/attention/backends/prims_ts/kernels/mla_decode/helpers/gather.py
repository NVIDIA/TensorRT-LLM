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

"""Blackwell four-row TMA gather with explicit five-coordinate PTX operands.

The CUTLASS DSL 4.7 experimental wrapper validates gather coordinates as a pair;
it cannot express four independent row coordinates through that interface.
Keep the native instruction isolated here instead of patching the dependency.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from .constants import TMA_GATHER_ROWS


@dsl_user_op
def _address(pointer, dtype, *, loc=None, ip=None):
    if hasattr(pointer, "data_ptr"):
        pointer = pointer.data_ptr(loc=loc, ip=ip)
    if hasattr(pointer, "toint"):
        pointer = pointer.toint(loc=loc, ip=ip)
    return dtype(pointer)


@cute.jit
def select_gather_map(primary, extra, use_extra):
    """Select numeric addresses: Pointer-object rebinding is not an SSA select."""
    primary_addr = _address(primary, Int64)
    extra_addr = _address(extra, Int64)
    return extra_addr if use_extra else primary_addr


@cute.jit
def load_sparse_rows(routes, request, offset, *, fragments: cutlass.Constexpr[int]):
    """Coalesced row window, distributed as one coordinate per lane/fragment."""
    lane = cute.arch.thread_idx()[0] & Int32(31)
    rows = cutlass.Array(Int32, fragments, space=cutlass.AddressSpace.rmem)
    for fragment in cutlass.range_constexpr(fragments):
        index = offset + lane + Int32(fragment * 32)
        value = Int32(0x7FFFFFFF)
        if index < routes.shape[0]:
            value = Int32(routes[index, request])
        rows[fragment] = value
    return rows


@cute.jit
def load_sparse_quad(routes, request, offset):
    """Keep one contiguous row quadruple in its issuing lane's registers."""
    rows = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
    if cutlass.const_expr(hasattr(routes, "routes_for_tile")):
        tile_routes = routes.routes_for_tile(offset, request)
        return tile_routes.mapped_quad(offset)
    for j in cutlass.range_constexpr(4):
        rows[j] = Int32(0x7FFFFFFF)
    if offset + Int32(3) < routes.shape[0]:
        if cutlass.const_expr(routes.stride[0] == 1):
            values = cutlass.Pointer(
                routes.iterator + offset + request * routes.stride[1], dtype=Int32
            ).load(count=4, alignment=4)
            for j in cutlass.range_constexpr(4):
                rows[j] = values[j]
        else:
            for j in cutlass.range_constexpr(4):
                rows[j] = Int32(routes[offset + j, request])
    else:
        for j in cutlass.range_constexpr(4):
            if offset + Int32(j) < routes.shape[0]:
                rows[j] = Int32(routes[offset + j, request])
    return rows


@cute.jit
def decode_sparse_quad(primary, extra, raw):
    """Select a source and map masked row sentinels to TMA zero-fill rows."""
    rows = cutlass.Array(Int32, TMA_GATHER_ROWS, space=cutlass.AddressSpace.rmem)
    for j in cutlass.range_constexpr(TMA_GATHER_ROWS):
        # Native sparse pools have fewer than INT32_MAX storage rows. The
        # reserved sentinel is already an out-of-bounds TMA coordinate;
        # retaining it gives zero fill without a compare/select per row.
        rows[j] = raw[j] & Int32(0x7FFFFFFF)
    return select_gather_map(primary, extra, raw[0] < 0), rows


@cute.jit
def gather4_quad(
    dst, primary, extra, column, raw, barrier, *, cta_group: cutlass.Constexpr[int] = 1
):
    """Issue a quad from the owning lane, without election or warp shuffles."""
    descriptor, rows = decode_sparse_quad(primary, extra, raw)
    gather4(
        dst,
        descriptor,
        column,
        rows[0],
        rows[1],
        rows[2],
        rows[3],
        barrier,
        cta_group=cta_group,
    )


@cute.jit
def gather4_cached(
    dst,
    primary,
    extra,
    column,
    cached_rows,
    token_offset,
    barrier,
    *,
    cache_base: cutlass.Constexpr[int] = 0,
    cta_group: cutlass.Constexpr[int] = 1,
):
    """Assemble a row quadruple using only warp-local register shuffles."""
    raw0 = Int32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=cached_rows[cache_base + token_offset // 32],
            offset=token_offset % 32,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.IDX,
        )
    )
    rows = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
    for j in cutlass.range_constexpr(4):
        token = token_offset + Int32(j)
        raw = Int32(
            prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=cached_rows[cache_base + token // 32],
                offset=token % 32,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.IDX,
            )
        )
        row = raw & Int32(0x7FFFFFFF)
        rows[j] = row if row != Int32(0x7FFFFFFF) else Int32(-1)
    descriptor = select_gather_map(primary, extra, raw0 < 0)
    if prims.elect_sync():
        gather4(
            dst,
            descriptor,
            column,
            rows[0],
            rows[1],
            rows[2],
            rows[3],
            barrier,
            cta_group=cta_group,
        )


@cute.jit
def invalid_sparse_token(
    token, length, routes, request, *, sparse: cutlass.Constexpr[bool]
):
    """Mask prefix tails and tagged invalid routes before softmax.

    Bit 31 selects the source; bits 0..30 hold its TMA row coordinate.
    All-one low bits denote an invalid row for either source.
    """
    if cutlass.const_expr(sparse and hasattr(routes, "is_invalid")):
        return (token >= length) | routes.is_invalid(token, request)
    invalid = token >= length
    if cutlass.const_expr(sparse):
        if token < length:
            invalid = (Int32(routes[token, request]) & Int32(0x7FFFFFFF)) == Int32(
                0x7FFFFFFF
            )
    return invalid


@cute.jit
def gather4(
    dst,
    tensor_map,
    column,
    row0,
    row1,
    row2,
    row3,
    barrier,
    *,
    cta_group: cutlass.Constexpr[int] = 1,
):
    """Issue one four-row copy from the calling thread.

    Multiple lanes may issue disjoint copies. The caller accounts for all
    copies in the shared completion barrier, including masked/OOB rows.

    The descriptor is 2D with a one-row box and no interleave. Destination
    alignment/swizzle and coordinate bounds follow its tensor-map contract.
    Group 2 delivers completion to the even CTA of the issuing CTA pair.
    """
    dst_addr = _address(dst, Int32)
    bar_addr = _address(barrier, Int32)
    if cutlass.const_expr(cta_group == 2):
        bar_addr = bar_addr & Int32(-16777217)  # Clear peer-CTA address bit 24.
        instruction = (
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4."
            "mbarrier::complete_tx::bytes.cta_group::2 "
        )
    else:
        assert cta_group == 1
        instruction = (
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
            "mbarrier::complete_tx::bytes.cta_group::1 "
        )
    cute.arch.inline_ptx(
        instruction + "[{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}];",
        read_only_args=[
            dst_addr,
            _address(tensor_map, Int64),
            Int32(column),
            Int32(row0),
            Int32(row1),
            Int32(row2),
            Int32(row3),
            bar_addr,
        ],
    )


class CachedSparseMask:
    """One/two per-warp bitsets reused across all score registers and heads."""

    def __init__(self, values, swaps):
        self.values = values
        self.swaps = swaps
        self.low, self.high = values

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    def __new_from_mlir_values__(self, values):
        return CachedSparseMask(
            cutlass.new_from_mlir_values(self.values, values), self.swaps
        )

    @cute.jit
    def is_invalid(self, token, request):
        del request
        word = self.low if self.swaps or (token & Int32(32)) == Int32(0) else self.high
        return ((word >> (token & Int32(31))) & Int32(1)) != Int32(0)


@cute.jit
def cache_sparse_mask(
    routes,
    tile_offset,
    length,
    request,
    *,
    swaps: cutlass.Constexpr[bool],
    column_half=None,
    tile_size_kv: cutlass.Constexpr[int] = 128,
):
    lane = cute.arch.thread_idx()[0] % 32
    warp = cute.arch.thread_idx()[0] % 128 // 32
    if cutlass.const_expr(hasattr(routes, "prefix_mask32")):
        low, high = Int32(0), Int32(0)
        if not routes.tile_is_full(tile_offset, length):
            if cutlass.const_expr(swaps):
                low = routes.prefix_mask32(tile_offset + warp * Int32(32), length)
                high = low
            else:
                if cutlass.const_expr(column_half is None):
                    column_half = lane >> Int32(4)
                start = tile_offset
                if cutlass.const_expr(tile_size_kv == 128):
                    start += column_half * Int32(64)
                low = routes.prefix_mask32(start, length)
                high = routes.prefix_mask32(start + Int32(32), length)
        return CachedSparseMask((low, high), swaps)
    if cutlass.const_expr(swaps):
        token = tile_offset + warp * Int32(32) + lane
        invalid = invalid_sparse_token(token, length, routes, request, sparse=True)
        bits = Int32(cute.arch.vote_ballot_sync(invalid))
        return CachedSparseMask((bits, bits), True)
    else:
        masks = cutlass.Array(
            Int32, tile_size_kv // 32, space=cutlass.AddressSpace.rmem
        )
        for part in cutlass.range_constexpr(tile_size_kv // 32):
            token = tile_offset + Int32(part * 32) + lane
            invalid = invalid_sparse_token(token, length, routes, request, sparse=True)
            masks[part] = Int32(cute.arch.vote_ballot_sync(invalid))
        if cutlass.const_expr(column_half is None):
            column_half = lane >> Int32(4)
        if cutlass.const_expr(tile_size_kv == 64):
            low, high = masks[0], masks[1]
        else:
            low = masks[0] if column_half == Int32(0) else masks[2]
            high = masks[1] if column_half == Int32(0) else masks[3]
        return CachedSparseMask((low, high), False)


@dsl_user_op
def _gather4_slices(
    dst,
    tensor_map,
    column,
    rows,
    barrier,
    *,
    num_slices,
    dst_stride,
    col_stride,
    loc=None,
    ip=None,
):
    # All callers broadcast the quad before entering this collective. Keep
    # feature slices in one asm block so election/uniformization is shared.
    instruction = "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2 "
    asm = [
        "{ .reg .b32 slice_dst, slice_col; .reg .pred issue; elect.sync _|issue, 0xffffffff;",
        "mov.b32 slice_dst, {$r0};",
        "mov.b32 slice_col, {$r2};",
    ]
    for i in range(num_slices):
        asm.append(
            "@issue "
            + instruction
            + "[slice_dst], [{$r1}, {slice_col, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}];"
        )
        if i + 1 < num_slices:
            asm.append(f"add.s32 slice_dst, slice_dst, {dst_stride};")
            asm.append(f"add.s32 slice_col, slice_col, {col_stride};")
    asm.append("}")
    cute.arch.inline_ptx(
        "\n".join(asm),
        read_only_args=[
            _address(dst, Int32),
            _address(tensor_map, Int64),
            Int32(column),
            rows[0],
            rows[1],
            rows[2],
            rows[3],
            _address(barrier, Int32) & Int32(-16777217),
        ],
        loc=loc,
        ip=ip,
    )


@cute.jit
def gather4_uniform_quad_slices(
    dst,
    primary,
    extra,
    column,
    raw,
    barrier,
    *,
    num_slices: cutlass.Constexpr[int],
    dst_stride: cutlass.Constexpr[int],
    col_stride: cutlass.Constexpr[int],
):
    """Issue 2CTA feature slices from a warp-uniform cached row quadruple.

    Every lane must participate with identical arguments. One elected lane
    issues all copies; dst_stride is bytes and col_stride is source elements.
    The caller accounts for all copied bytes in the completion barrier.
    """
    descriptor, rows = decode_sparse_quad(primary, extra, raw)
    _gather4_slices(
        dst,
        descriptor,
        column,
        rows,
        barrier,
        num_slices=num_slices,
        dst_stride=dst_stride,
        col_stride=col_stride,
    )


@dsl_user_op
def gather4_cached_warp(
    dst,
    tensor_map,
    column,
    cached_rows,
    barrier,
    *,
    cache_start,
    num_quads,
    num_slices,
    quad_stride,
    slice_stride,
    col_stride,
    loc=None,
    ip=None,
):
    """Issue one warp's cached quads with one election for every feature slice.

    All lanes participate. Rows are already mapped, cleaned and warp-uniform;
    the caller retains them across the head-dimension stages of one K tile.
    Strides for destinations are bytes, and the column stride is elements.
    """
    assert num_quads <= 8
    args = [
        _address(dst, Int32),
        _address(tensor_map, Int64),
        Int32(column),
        _address(barrier, Int32),
    ]
    args += [cached_rows[cache_start + i] for i in range(num_quads * 4)]
    instruction = "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::1 "
    asm = [
        "{ .reg .b32 copy_dst, copy_col; .reg .pred issue; elect.sync _|issue, 0xffffffff;"
    ]
    for part in range(num_slices):
        asm.append(f"add.s32 copy_col, {{$r2}}, {part * col_stride};")
        for quad in range(num_quads):
            asm.append(
                f"add.s32 copy_dst, {{$r0}}, {part * slice_stride + quad * quad_stride};"
            )
            rows = ", ".join("{$r" + str(4 + quad * 4 + j) + "}" for j in range(4))
            asm.append(
                "@issue "
                + instruction
                + "[copy_dst], [{$r1}, {copy_col, "
                + rows
                + "}], [{$r3}];"
            )
    asm.append("}")
    cute.arch.inline_ptx("\n".join(asm), read_only_args=args, loc=loc, ip=ip)
