# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CSA2 packed-row producer for the PrimTS single-pipe Keep-AB kernel."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import SignalingThreads
from cutlass.experimental.task_scheduling.resources import PipelineConfig, StageInfo

from ...prims_ts.kernels.fmha_decode.fmha_decode_resources.smem_transformed_kv import (
    _cvt_e2m1_word_to_e4m3_words,
    _cvt_e2m1x2_byte_to_bf16x2,
    _cvt_e4m3x2_byte_to_bf16x2,
    _cvt_e4m3x2_word_to_bf16x2,
)
from ...prims_ts.kernels.mla_decode.helpers.layout import head_dim_cta_offset_v
from ...prims_ts.kernels.mla_decode.helpers.math import fmul2
from ...prims_ts.kernels.mla_decode.helpers.ops import pack_float4_to_fp8_e4m3
from ...prims_ts.kernels.mla_decode.helpers.sparse_indices import (
    SparseKVIndices,
    SparseKVLengths,
    SparseSourceTile,
    _sparse_execution_length,
)
from ...prims_ts.kernels.mla_decode.helpers.stage import MlaStage
from ...prims_ts.kernels.mla_decode.helpers.tile import (
    cta_idx_head_dim_v_for_stage,
    staged_kv_head_dim_call_idx,
)
from ...prims_ts.kernels.mla_decode.throughput_latency_1cta.resources.smem_resources import (
    SmemKvResource,
)
from .params import PrimsTSMainFormat, SwaCacheFormat
from .quantization import row_bytes

__all__ = ["PackedKvProducer", "main_row_bytes"]


def main_row_bytes(main_format: PrimsTSMainFormat) -> int:
    """Row width of a MAIN source: a packed NVFP4 record or a decoded E4M3 bank row."""
    if main_format == "main":
        return row_bytes(512, "main")
    if main_format == "main_fp8":
        return 512
    raise ValueError("CSA2 PrimTS MAIN format must be 'main' or 'main_fp8'")


class _BoundedSourceTile(SparseSourceTile):
    """Apply physical-pool bounds to the existing sparse gather and score mask."""

    def __init__(self, values: tuple, pool_rows: Int32) -> None:
        super().__init__(values, valid_prefix=False)
        self.pool_rows = pool_rows
        self._packed_values = (values, pool_rows)

    def __extract_mlir_values__(self) -> list:
        return cutlass.extract_mlir_values(self._packed_values)

    def __new_from_mlir_values__(self, values: list) -> "_BoundedSourceTile":
        source_values, rows = cutlass.new_from_mlir_values(self._packed_values, values)
        return _BoundedSourceTile(source_values, rows)

    @cute.jit
    def _bounded_index(self, token: Int32) -> tuple:
        position = token - self.origin
        index = Int32(-1)
        # Inactive entries need not even dereference the index list.
        if position >= Int32(0) and position < self.length:
            index = Int32(self.indices[position])
        valid = (index >= Int32(0)) & (index < self.pool_rows)
        return index, valid

    @cute.jit
    def load_full_quad(self, token: Int32) -> cutlass.Array:
        rows = SparseSourceTile.load_full_quad(self, token)
        for j in cutlass.range_constexpr(4):
            index = rows[j] & Int32(0x7FFFFFFF)
            valid = index < self.pool_rows
            rows[j] = (index if valid else Int32(0x7FFFFFFF)) | self.tag
        return rows


class _BoundedKVIndices(SparseKVIndices):
    """Retain routing and the physical-pool limits of both sources."""

    def __init__(self, source: SparseKVIndices, pool_rows: tuple) -> None:
        # SparseKVIndices.values ends with the capacity it carries through JIT calls.
        super().__init__(source.values[:-1], source.capacity, False, source.has_extra)
        self.pool_rows = pool_rows
        self._packed_values = (self.values, pool_rows)

    def __extract_mlir_values__(self) -> list:
        return cutlass.extract_mlir_values(self._packed_values)

    def __new_from_mlir_values__(self, values: list) -> "_BoundedKVIndices":
        source_values, pool_rows = cutlass.new_from_mlir_values(self._packed_values, values)
        source = SparseKVIndices(source_values[:-1], source_values[-1], False, self.has_extra)
        return _BoundedKVIndices(source, pool_rows)

    @cute.jit
    def bind_query_row(
        self, query_row: Int32, primary_length=None, extra_length=None
    ) -> "_BoundedKVIndices":
        source = SparseKVIndices.bind_query_row(self, query_row, primary_length, extra_length)
        return _BoundedKVIndices(source, self.pool_rows)

    @cute.jit
    def select_source_tile(
        self, tile_start: Int32, query_row=None, *, for_mask: cutlass.Constexpr[bool] = False
    ) -> _BoundedSourceTile:
        row = self.bind_query_row(query_row)
        source = SparseKVIndices.select_source_tile(row, tile_start, for_mask=for_mask)
        rows = self.pool_rows[0]
        if cutlass.const_expr(self.has_extra):
            rows = self.pool_rows[1] if tile_start >= row.primary_span else rows
        return _BoundedSourceTile(source.values, rows)


@cute.jit
def _pow2_f32(exponent: Int32) -> Float32:
    """Construct an exact FP32 power of two, including representable subnormals."""
    bits = Int32(0)
    if exponent > Int32(127):
        bits = Int32(0x7F800000)
    elif exponent >= Int32(-126):
        bits = (exponent + Int32(127)) << Int32(23)
    elif exponent >= Int32(-149):
        bits = Int32(1) << (exponent + Int32(149))
    return prims.mov_b32(bits, target_type=Float32)


@cute.jit
def _ue8m0_scale(scale: Int32) -> Float32:
    """Decode a UE8M0 group scale, with byte zero denoting 2**-127 and 255 NaN."""
    value = _pow2_f32(scale - Int32(127))
    if scale == Int32(255):
        value = prims.mov_b32(Int32(0x7FC00000), target_type=Float32)
    return value


@cute.jit
def _e4m3_scale(scale: Int32) -> Float32:
    """Decode an E4M3 group scale exactly; BF16 holds every E4M3 value, NaN included."""
    bits = _cvt_e4m3x2_byte_to_bf16x2(scale) & Int32(0xFFFF)
    return prims.mov_b32(bits << Int32(16), target_type=Float32)


@cute.jit
def _scale_bf16_pairs_to_e4m3(lo: Int32, hi: Int32, scale: Float32) -> Int32:
    """Multiply four exactly decoded values in FP32 and pack RN/satfinite E4M3."""
    scale_pair = (scale, scale)
    values_lo = fmul2(
        (
            prims.mov_b32(lo << Int32(16), target_type=Float32),
            prims.mov_b32(lo & Int32(-65536), target_type=Float32),
        ),
        scale_pair,
    )
    values_hi = fmul2(
        (
            prims.mov_b32(hi << Int32(16), target_type=Float32),
            prims.mov_b32(hi & Int32(-65536), target_type=Float32),
        ),
        scale_pair,
    )
    return pack_float4_to_fp8_e4m3(values_lo[0], values_lo[1], values_hi[0], values_hi[1])


@cute.jit
def _load_packed_words(
    pool: cute.Tensor, row: Int32, column: Int32, count: cutlass.Constexpr[int]
) -> cutlass.Array:
    """Read an already bounded payload chunk, with byte loads for unaligned rows."""
    words = cutlass.Array(Int32, count, space=cutlass.AddressSpace.rmem)
    byte_offset = Int64(row) * Int64(pool.stride[0]) + Int64(column)
    # Pool bases are 16B aligned, but a dynamic outer stride may misalign rows.
    if (byte_offset & Int64(4 * count - 1)) == Int64(0):
        packed = cutlass.Pointer(pool.iterator + byte_offset, dtype=Int32).load(
            count=count, alignment=4 * count
        )
        for word in cutlass.range_constexpr(count):
            words[word] = Int32(packed[word])
    else:
        for word in cutlass.range_constexpr(count):
            value = Int32(0)
            for byte in cutlass.range_constexpr(4):
                value = value | (
                    Int32(pool[Int64(row), column + Int32(word * 4 + byte)]) << Int32(byte * 8)
                )
            words[word] = value
    return words


@dataclass(kw_only=True)
class _PackedSmemKvResource(SmemKvResource):
    """Write decoded E4M3 bytes into the native swizzled K/V operand ring.

    Operands carry the cache values at unit tensor scale: group scales are
    applied while decoding and the result is rounded RN/satfinite to E4M3.
    ``main_fp8`` MAIN rows already hold those operand bytes and are copied.
    """

    swa: object = None
    main: object = None
    has_extra: bool = False
    swa_cache_format: SwaCacheFormat = "swa"
    main_cache_format: PrimsTSMainFormat = "main"
    aligned_strides: bool = True

    # Fetches only issue a chunk's global loads and return its scale byte (-1
    # for a masked row) and payload words as SSA values, never local arrays, so
    # the producer can keep several chunks' loads in flight in registers.
    @cute.jit
    def _fetch_swa_chunk(self, row: Int32, dimension: Int32) -> tuple:
        scale, w0, w1, w2, w3 = Int32(-1), Int32(0), Int32(0), Int32(0), Int32(0)
        if row >= Int32(0) and row < self.swa.shape[0] and row != Int32(0x7FFFFFFF):
            packed = _load_packed_words(self.swa, row, dimension, 4)
            w0, w1, w2, w3 = packed[0], packed[1], packed[2], packed[3]
            scale = Int32(0)
            if cutlass.const_expr(self.swa_cache_format != "swa_fp8"):
                scale = Int32(self.swa[Int64(row), Int32(512) + dimension // Int32(32)])
        return scale, (w0, w1, w2, w3)

    @cute.jit
    def _convert_swa_chunk(self, scale_byte: Int32, payload: tuple, values: cutlass.Array) -> None:
        if scale_byte >= Int32(0):
            if cutlass.const_expr(self.swa_cache_format == "swa_fp8"):
                # A unit persistent scale leaves the payload as the operand.
                for word in cutlass.range_constexpr(4):
                    values[word] = payload[word]
            else:
                scale = _ue8m0_scale(scale_byte)
                for word in cutlass.range_constexpr(4):
                    values[word] = _scale_bf16_pairs_to_e4m3(
                        _cvt_e4m3x2_word_to_bf16x2(payload[word], 0),
                        _cvt_e4m3x2_word_to_bf16x2(payload[word], 16),
                        scale,
                    )

    @cute.jit
    def _fetch_main_chunk(self, row: Int32, dimension: Int32) -> tuple:
        if cutlass.const_expr(self.main_cache_format == "main_fp8"):
            scale, w0, w1, w2, w3 = Int32(-1), Int32(0), Int32(0), Int32(0), Int32(0)
            if row >= Int32(0) and row < self.main.shape[0] and row != Int32(0x7FFFFFFF):
                packed = _load_packed_words(self.main, row, dimension, 4)
                w0, w1, w2, w3 = packed[0], packed[1], packed[2], packed[3]
                scale = Int32(0)
            return scale, (w0, w1, w2, w3)
        scale, w0, w1 = Int32(-1), Int32(0), Int32(0)
        if row >= Int32(0) and row < self.main.shape[0] and row != Int32(0x7FFFFFFF):
            packed = _load_packed_words(self.main, row, dimension // Int32(2), 2)
            w0, w1 = packed[0], packed[1]
            scale = Int32(self.main[Int64(row), Int32(256) + dimension // Int32(16)])
        return scale, (w0, w1)

    @cute.jit
    def _convert_main_chunk(self, scale_byte: Int32, payload: tuple, values: cutlass.Array) -> None:
        if scale_byte >= Int32(0):
            if cutlass.const_expr(self.main_cache_format == "main_fp8"):
                # Bank rows were decoded at unit scale: the payload is the operand.
                for word in cutlass.range_constexpr(4):
                    values[word] = payload[word]
            else:
                scale = _e4m3_scale(scale_byte)
                for word in cutlass.range_constexpr(4):
                    pair = payload[word // 2] >> Int32((word % 2) * 16)
                    values[word] = _scale_bf16_pairs_to_e4m3(
                        _cvt_e2m1x2_byte_to_bf16x2(pair),
                        _cvt_e2m1x2_byte_to_bf16x2(pair >> Int32(8)),
                        scale,
                    )

    @cute.jit
    def _decode_swa_chunk(self, row: Int32, dimension: Int32, values: cutlass.Array) -> None:
        scale, payload = self._fetch_swa_chunk(row, dimension)
        self._convert_swa_chunk(scale, payload, values)

    @cute.jit
    def _decode_main_chunk(self, row: Int32, dimension: Int32, values: cutlass.Array) -> None:
        scale, payload = self._fetch_main_chunk(row, dimension)
        self._convert_main_chunk(scale, payload, values)

    @cute.jit
    def _prefetch_row(self, pool: cute.Tensor, row: Int32, width: cutlass.Constexpr[int]) -> None:
        """Bring every cache line of one packed row into L2, whatever its alignment."""
        start = pool.iterator + Int64(row) * Int64(pool.stride[0])
        for line in cutlass.range_constexpr((width + 127) // 128):
            prims.prefetch_l2(start + line * 128)
        prims.prefetch_l2(start + (width - 1))

    @cute.jit
    def _prefetch_tile_rows(
        self, cached_page_ids, base: int, lane: Int32, quads_per_warp: cutlass.Constexpr[int]
    ) -> None:
        """Warm L2 with every row of this warp's quads before their serial decode.

        Lane ``l`` takes entry ``l % 4`` of the quad its lane ``l // 4`` owns, so
        all of a tile's DRAM misses are in flight before the first row is decoded.
        """
        raw = Int32(0)
        for j in cutlass.range_constexpr(4):
            candidate = prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=cached_page_ids[base + j],
                offset=lane >> Int32(2),
                mask_and_clamp=0x1F,
                kind=prims.Shfl.IDX,
            )
            raw = candidate if (lane & Int32(3)) == Int32(j) else raw
        row = raw & Int32(0x7FFFFFFF)
        if lane < Int32(4 * quads_per_warp) and row != Int32(0x7FFFFFFF):
            if cutlass.const_expr(self.has_extra):
                if raw < Int32(0):
                    if row < self.main.shape[0]:
                        self._prefetch_row(self.main, row, main_row_bytes(self.main_cache_format))
                elif row < self.swa.shape[0]:
                    self._prefetch_row(self.swa, row, row_bytes(512, self.swa_cache_format))
            elif row < self.swa.shape[0]:
                self._prefetch_row(self.swa, row, row_bytes(512, self.swa_cache_format))

    @cute.jit
    def _owner_row(self, cached_page_ids, base: int, owner: Int32, row_in_quad: Int32) -> Int32:
        """This lane's tagged row of quad ``owner``, broadcast from the lane holding it."""
        raw = Int32(0)
        for j in cutlass.range_constexpr(4):
            candidate = prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=cached_page_ids[base + j],
                offset=owner,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.IDX,
            )
            raw = candidate if row_in_quad == Int32(j) else raw
        return raw

    @cute.jit
    def _chunk_offset(
        self,
        warp: Int32,
        quads_per_warp: cutlass.Constexpr[int],
        owner: Int32,
        row_in_quad: Int32,
        chunk_in_row: Int32,
    ) -> Int32:
        """Byte offset of this lane's 16-byte chunk in the 128B-swizzled operand stage."""
        token = (warp * Int32(quads_per_warp) + owner) * Int32(4) + row_in_quad
        byte_offset = token * Int32(128) + chunk_in_row * Int32(16)
        return byte_offset ^ ((token & Int32(7)) << Int32(4))

    @cute.jit
    def _serial_quads(
        self,
        words: cutlass.Array,
        cached_page_ids,
        base: int,
        warp: Int32,
        quads_per_warp: cutlass.Constexpr[int],
        row_in_quad: Int32,
        chunk_in_row: Int32,
        dimension: Int32,
    ) -> None:
        """Decode quad by quad with synchronous loads; any row alignment or tag."""
        for owner in cutlass.range(quads_per_warp, unroll=1):
            raw = self._owner_row(cached_page_ids, base, owner, row_in_quad)
            row = raw & Int32(0x7FFFFFFF)
            values = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            for word in cutlass.range_constexpr(4):
                values[word] = Int32(0)
            if cutlass.const_expr(self.has_extra):
                if raw < Int32(0):
                    self._decode_main_chunk(row, dimension, values)
                else:
                    self._decode_swa_chunk(row, dimension, values)
            else:
                self._decode_swa_chunk(row, dimension, values)
            offset = self._chunk_offset(warp, quads_per_warp, owner, row_in_quad, chunk_in_row)
            words.data_ptr(offset // Int32(4)).store(values.load(0, 4), alignment=16)

    @cute.jit
    def _async_quads(
        self,
        is_main: cutlass.Constexpr[bool],
        words: cutlass.Array,
        cached_page_ids,
        base: int,
        warp: Int32,
        quads_per_warp: cutlass.Constexpr[int],
        row_in_quad: Int32,
        chunk_in_row: Int32,
        dimension: Int32,
    ) -> None:
        """Copy raw chunks into their own operand slots asynchronously, then decode in place.

        The 128B swizzle permutes 16-byte chunks only within a row, so a lane's raw
        bytes and its decoded chunk share one slot no other lane touches. A packed
        MAIN slot receives the chunk's 8 payload bytes and the 4-byte word holding
        its E4M3 scale and is decoded in place; an ordinary FP8 SWA slot or an
        E4M3 MAIN bank slot receives its final 16 payload bytes, which are already
        the unit-scale operand. Masked rows copy zero bytes, which decode to zero.
        Every copy of the warp is in flight before the first wait, and the loops
        stay rolled.
        """
        pool = self.main if is_main else self.swa
        decode = is_main and self.main_cache_format == "main"
        stride = Int64(pool.stride[0])
        for owner in cutlass.range(quads_per_warp, unroll=1):
            row = self._owner_row(cached_page_ids, base, owner, row_in_quad) & Int32(0x7FFFFFFF)
            # Pools hold fewer than 0x7FFFFFFF rows, so this also rejects the sentinel.
            valid = row < pool.shape[0]
            start = pool.iterator + Int64(row if valid else Int32(0)) * stride
            slot = self._chunk_offset(warp, quads_per_warp, owner, row_in_quad, chunk_in_row)
            if cutlass.const_expr(decode):
                prims.cp_async_shared_global(
                    words.data_ptr(slot // Int32(4)),
                    start + Int64(dimension // Int32(2)),
                    8,
                    "ca",
                    cp_size=Int32(8) if valid else Int32(0),
                )
                prims.cp_async_shared_global(
                    words.data_ptr(slot // Int32(4) + Int32(2)),
                    start + Int64(Int32(256) + ((dimension // Int32(16)) & Int32(-4))),
                    4,
                    "ca",
                    cp_size=Int32(4) if valid else Int32(0),
                )
            else:
                prims.cp_async_shared_global(
                    words.data_ptr(slot // Int32(4)),
                    start + Int64(dimension),
                    16,
                    "cg",
                    cp_size=Int32(16) if valid else Int32(0),
                )
        prims.cp_async_commit_group()
        prims.cp_async_wait_group(0)
        if cutlass.const_expr(decode):
            self._decode_main_slots(
                words, warp, quads_per_warp, row_in_quad, chunk_in_row, dimension
            )

    @cute.jit
    def _decode_main_slots(
        self,
        words: cutlass.Array,
        warp: Int32,
        quads_per_warp: cutlass.Constexpr[int],
        row_in_quad: Int32,
        chunk_in_row: Int32,
        dimension: Int32,
    ) -> None:
        """Decode MAIN slots in place with FP16x2 arithmetic.

        Every E2M1 x E4M3 product has at most seven significant bits within
        [2**-10, 2688], so FP16 holds it exactly and the single E4M3 rounding
        matches the synchronous FP32 path bit for bit, including signed zeros
        and NaN.
        """
        shift = ((dimension // Int32(16)) & Int32(3)) * Int32(8)
        for owner in cutlass.range(quads_per_warp, unroll=1):
            slot = self._chunk_offset(warp, quads_per_warp, owner, row_in_quad, chunk_in_row)
            chunk = words.data_ptr(slot // Int32(4)).load(count=4, alignment=16)
            scale = Int32(chunk[2]) >> shift
            values = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            values[0], values[1] = _cvt_e2m1_word_to_e4m3_words(Int32(chunk[0]), scale)
            values[2], values[3] = _cvt_e2m1_word_to_e4m3_words(Int32(chunk[1]), scale)
            words.data_ptr(slot // Int32(4)).store(values.load(0, 4), alignment=16)

    @cute.jit
    def _swa_quads(self, *quads) -> None:
        # Scaled legacy SWA rows keep synchronous loads.
        if cutlass.const_expr(self.swa_cache_format == "swa_fp8"):
            self._async_quads(False, *quads)
        else:
            self._serial_quads(*quads)

    @cute.jit
    def _stage_quads(self, *quads) -> None:
        """Copy asynchronously when the compiled variant guarantees aligned pool rows."""
        if cutlass.const_expr(not self.aligned_strides):
            self._serial_quads(*quads)
        elif cutlass.const_expr(self.has_extra):
            words, cached_page_ids, base = quads[:3]
            first = prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=cached_page_ids[base],
                offset=0,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.IDX,
            )
            # A source tile never mixes SWA and MAIN rows, and every coordinate,
            # masked ones included, keeps its source tag. Lane 0 owns the warp's
            # first quad, so its tag selects one specialized path per warp.
            if first < Int32(0):
                self._async_quads(True, *quads)
            else:
                self._swa_quads(*quads)
        else:
            self._swa_quads(*quads)

    @cute.jit
    def _producer_load_kv_stage(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ) -> None:
        cfg = self.cfg
        head_stage = staged_kv_head_dim_call_idx(
            cfg, stage_info, inst_id, is_v, stage_idx=stage_idx, section=section
        )
        dim_offset = Int32(head_stage * cfg.head_dim_per_stage_kv)
        if cutlass.const_expr(is_v):
            v_partition = cta_idx_head_dim_v_for_stage(self.cta_idx_head_dim_v, stage_info)
            dim_offset = head_dim_cta_offset_v(cfg, v_partition) + Int32(
                head_stage * cfg.head_dim_per_stage_v
            )
        lane = cute.arch.thread_idx()[0] % 32
        warp = cute.arch.thread_idx()[0] // 32 - cfg.load_warp_idx
        quads_per_warp = cfg.tile_size_kv // (4 * cfg.load_num_warps)
        base = page_id_slot * cfg.sparse_cache_words
        stage = self._stage_base(stage_info)
        words = cutlass.Array(
            stage.data_ptr(), dtype=Int32, shape=(cfg.kv_smem_tile_bytes // 4,), addrspace=3
        )
        # Eight adjacent lanes write one D128 FP8 row in a quad's four rows.
        row_in_quad = lane >> Int32(3)
        chunk_in_row = lane & Int32(7)
        dimension = dim_offset + chunk_in_row * Int32(16)
        if cutlass.const_expr(not is_v and head_stage == 0):
            # Later D128 slices of the same rows then hit L2.
            self._prefetch_tile_rows(cached_page_ids, base, lane, quads_per_warp)
        self._stage_quads(
            words,
            cached_page_ids,
            base,
            warp,
            quads_per_warp,
            row_in_quad,
            chunk_in_row,
            dimension,
        )
        # The async/UMMA pipeline's commit publishes thread arrivals after all
        # generic SMEM stores are visible to tcgen05. There are no TMA bytes.
        cute.arch.fence_view_async_shared()


@dataclass(frozen=True)
class PackedKvProducer:
    """CSA2-only producer ABI; packed pools and their row strides stay dynamic.

    ``aligned_strides`` compiles only the asynchronous row copies, which require
    pool row strides that are multiples of 16 bytes; otherwise only synchronous
    loads are compiled. Each kernel carries a single path to keep its code small.
    ``main_cache_format`` selects persistent NVFP4 MAIN records (``main``) or
    unit-scale E4M3 rows already decoded from them (``main_fp8``).
    """

    has_extra: bool
    swa_cache_format: SwaCacheFormat = "swa"
    aligned_strides: bool = True
    main_cache_format: PrimsTSMainFormat = "main"

    def __post_init__(self) -> None:
        if self.swa_cache_format not in ("swa", "swa_fp8"):
            raise ValueError("CSA2 PrimTS SWA format must be 'swa' or 'swa_fp8'")
        main_row_bytes(self.main_cache_format)

    def compile_signature(self) -> tuple:
        swa_layout = (
            ("e4m3", "unit", 512, 0)
            if self.swa_cache_format == "swa_fp8"
            else ("e4m3", "ue8m0", 528, 32)
        )
        main_layout = (
            ("e4m3", "unit", 512, 0)
            if self.main_cache_format == "main_fp8"
            else ("e2m1x2", "e4m3", 288, 16)
        )
        return (
            "csa2_packed_rows_e4m3_vector_quad_v4",
            self.swa_cache_format,
            swa_layout,
            self.main_cache_format,
            main_layout,
            "unit_kv_tensor_scale",
            "dynamic_outer_byte_stride",
            self.has_extra,
            self.aligned_strides,
        )

    def pipeline_config(
        self, cfg: object, load_group: object, umma_group: object, cta_layout_vmnk: tuple
    ) -> PipelineConfig:
        if (
            not cfg.single_kv_pipe
            or cfg.kernel_variant != "keeps_mma_ab"
            or cfg.qkv_dtype != "e4m3"
            or cfg.qkv_dtype_bytes != 1
            or cfg.rope_dim != 0
            or cfg.head_dim_qk != 512
            or cfg.head_dim_v != 512
            or cfg.head_dim_per_cta_v not in (128, 256, 512)
            or cfg.head_dim_per_stage_kv != 128
            or cfg.head_dim_per_stage_v != 128
            or cfg.tile_size_q != 64
            or cfg.tile_size_kv != 128
            or cfg.num_tokens_per_page != 1
            or cfg.load_num_warps not in (1, 2, 4, 8)
            or cfg.q_stages != 2
            or cfg.kv_stages < 8
            # A single V owner reads V from its retained K stages; a V split may
            # read its slices there too or load them separately.
            or (cfg.head_dim_per_cta_v == 512 and not cfg.sparse_reuse_kv)
        ):
            raise ValueError(
                "CSA2 packed KV requires E4M3 D512/D128 single-pipe Keep-AB "
                "with M64/BK128/page1, Q2, at least eight KV stages and retained KV "
                "whenever one CTA owns all of V"
            )
        return PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.kv_stages,
            producer_group=load_group,
            consumer_group=umma_group,
            cta_layout_vmnk=cta_layout_vmnk,
            consumer_signaling_threads=SignalingThreads.CtaLeader,
        )

    def make_resource(self, sources: tuple, **kwargs: object) -> SmemKvResource:
        swa, main = sources
        return _PackedSmemKvResource(
            swa=swa,
            main=main,
            has_extra=self.has_extra,
            swa_cache_format=self.swa_cache_format,
            main_cache_format=self.main_cache_format,
            aligned_strides=self.aligned_strides,
            **kwargs,
        )

    @cute.jit
    def bind_sparse_views(
        self,
        indices: tuple,
        swa_lengths: cute.Tensor,
        main_lengths,
        sources: tuple,
        capacity: cutlass.Constexpr[int],
        *,
        request=None,
    ) -> tuple:
        """Bind format-tagged UINT8 SWA/MAIN pools to both physical-row lists."""
        # The first sparse-input slot carries both lists; the two source slots
        # remain opaque packed UINT8 pools. Operands are decoded at unit tensor
        # scale, so no FP8-primary tensor scale is bound.
        swa_indices, main_indices = indices
        source = SparseKVIndices(
            (swa_indices, main_indices, swa_lengths, main_lengths, None, None, None, None),
            capacity,
            assume_valid_prefix=False,
            has_extra=self.has_extra,
        )
        routes = _BoundedKVIndices(source, (Int32(sources[0].shape[0]), Int32(sources[1].shape[0])))
        if cutlass.const_expr(request is not None):
            routes = routes.bind_query_row(request)
            length = _sparse_execution_length(routes.primary_length, routes.extra_length)
            lengths = SparseKVLengths((None, None, length, swa_lengths.shape[0]), self.has_extra)
        else:
            lengths = SparseKVLengths((swa_lengths, main_lengths, None, None), self.has_extra)
        return routes, lengths
