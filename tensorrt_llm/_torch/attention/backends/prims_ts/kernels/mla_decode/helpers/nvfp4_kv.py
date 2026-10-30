# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Storage and loader helpers shared by the 1CTA/2CTA NVFP4 MLA paths.

External records are token-interleaved uint8 bytes, with no global NVFP4 scale:

    D512 NVFP4: [ packed E2M1: 256 B ][ E4M3 block16 scales: 32 B ]
    D576 NVFP4: [ packed E2M1: 256 B ][ E4M3 RoPE: 64 B ][ scales: 32 B ]
    D512 FP8:   [ E4M3 K/V: 512 B ]   (primary SWA pool only)

D512 scale block b is at byte 256+b. D576 scale block b is at
320+8*(b%4)+b//4; its RoPE bytes are unscaled. Each NVFP4 byte stores the
even-indexed value in its low nibble. FP8 primary storage uses one separate
runtime tensor scale; QK/output arithmetic applies it without changing bytes.

Supported pairs are native/native, NVFP4/NVFP4, D576 NVFP4+FP8-RoPE in both
pools, and D512 FP8 primary/NVFP4 extra. External physical-page strides are
preserved when callers expose the internal [1, record_bytes, storage_rows]
view. Source-tagged indices select either descriptor: bit31 means extra,
low31 bits select a storage row, and all-one low bits represent an invalid row.
Both families gather the same records, then consume ordinary FP8 MMA tiles.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.experimental import cuda, primitives as prims

from ...tensor_map import create_tensor_map_tiled_from_view
from .gather import _address, gather4, select_gather_map
from ...fmha_decode.fmha_decode_resources.tmem_transformed_kv import (
    _has_packed_e4m3_e2m1_mul,
)
from ...fmha_decode.fmha_decode_resources.smem_transformed_kv import (
    _cvt_e2m1_word_to_e4m3_words,
)


# ===========================================================================
# Storage formats and configuration
# ===========================================================================


def cache_record_size(cache_format: str, rope_dim: int) -> int:
    """Elements per token; quantized formats use uint8 elements."""
    return {
        "native": 512 + rope_dim,
        "nvfp4": 288,
        "nvfp4.fp8rope": 352,
        "fp8": 512,
    }[cache_format]


def validate_nvfp4_kv_formats(
    primary_format, extra_format, rope_dim, qkv_dtype, page_size
):
    """Reject unsupported storage/compute pairs before constructing tensor maps.

    Page size one is the internal physical-row view, not an external cache-page
    restriction. Native/native retains the caller's ordinary paged-cache path.
    A mixed pair has FP8 only in the primary SWA pool and NVFP4 in the extra.
    """
    formats = (primary_format, extra_format)
    allowed = (
        ("native", "native"),
        ("nvfp4", "nvfp4"),
        ("nvfp4.fp8rope", "nvfp4.fp8rope"),
        ("fp8", "nvfp4"),
    )
    if formats not in allowed:
        raise ValueError(
            "cache formats must be native/native, nvfp4/nvfp4, nvfp4.fp8rope/nvfp4.fp8rope, or fp8 primary/nvfp4 extra"
        )
    if primary_format == "native":
        return
    if qkv_dtype != "e4m3":
        raise ValueError("NVFP4 KV formats require E4M3 Q")
    if page_size != 1:
        raise ValueError("NVFP4 KV requires an internal page-size-one storage-row view")
    if primary_format == "nvfp4.fp8rope":
        if rope_dim != 64:
            raise ValueError("nvfp4.fp8rope requires 512 NoPE and 64 RoPE channels")
    elif rope_dim != 0:
        raise ValueError("nvfp4/fp8 require D512 QK/V without an extra RoPE tail")


class Nvfp4KVConfig:
    """Common format predicates for both attention families."""

    @property
    def has_nvfp4_kv(self) -> bool:
        """Select row-record loaders for NVFP4 pools or the FP8-primary/NVFP4-extra pair."""
        return self.kv_cache_format != "native"

    @property
    def has_fp8_primary(self) -> bool:
        """Return whether primary K/V bytes need a separate per-tensor descale."""
        return self.kv_cache_format == "fp8"

    @property
    def max_cache_record_size(self) -> int:
        """Return the largest storage-row extent, in bytes for the NVFP4 loader family."""
        return max(
            cache_record_size(self.kv_cache_format, self.rope_dim),
            cache_record_size(self.extra_kv_cache_format, self.rope_dim),
        )

    @property
    def raw_record_stride(self) -> int:
        # gather4 requires every four-row destination to be 128-byte aligned.
        """Return the raw SMEM row pitch in bytes, aligned for four-row TMA gathers."""
        return (self.max_cache_record_size + 31) // 32 * 32


# ===========================================================================
# Shared-memory addressing and publication
# ===========================================================================


@cute.jit
def publish_nvfp4_kv(barrier, count, *, cta_group: cutlass.Constexpr[int] = 1):
    """Publish ordinary shared stores before signaling the MMA pipeline."""
    cute.arch.fence_view_async_shared()
    # The elected publisher must observe every lane's converted stores.
    # Proxy fencing alone does not make a relaxed manual complete_tx a
    # release operation. Join the warp, then publish through a release fence.
    cute.arch.sync_warp()
    if prims.elect_sync():
        if cutlass.const_expr(cta_group == 2):
            prims.fence_sync_restrict(prims.MemOrder.RELEASE)
            # Use the cluster address mapping API, not encoded CTA-rank bits.
            leader = prims.mapa(barrier, Int32(0))
            prims.mbarrier_complete_tx(
                leader, Int32(count), scope=prims.MemScope.CLUSTER
            )
        else:
            cute.arch.inline_ptx("fence.acq_rel.cta;")
            prims.mbarrier_complete_tx(barrier, Int32(count))


@cute.jit
def swizzle_byte_offset(offset, width: cutlass.Constexpr[int]):
    """Map an unswizzled byte offset into the matching 32/64/128-byte TMA swizzle.

    The selected XOR mask changes the 16-byte bank group using row bits;
    callers must use the same swizzle width as the tensor map and MMA view."""
    mask = {32: 0x10, 64: 0x30, 128: 0x70}[width]
    return offset ^ ((offset >> Int32(3)) & Int32(mask))


@cute.jit
def store_shared_u32x4(dst, offset, a, b, c, d):
    """Store four packed 32-bit words at dst + byte offset with one 16-byte SMEM store.

    The destination must be 16-byte aligned. Publication is a separate
    operation so a loader can finish its complete tile before signaling MMA."""
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[_address(dst, Int32) + offset, a, b, c, d],
    )


# ===========================================================================
# Primary-pool FP8 selection
# ===========================================================================


@cute.jit
def source_is_fp8(cfg, use_extra):
    """Select unchanged FP8 primary bytes from a source-tagged gather group.

    The extra pool is always NVFP4 in a mixed pair; the tag selects a pool,
    not an element dtype within a record. Broadcast from lane zero because
    only metadata-issuing lanes have valid source coordinates."""
    if cutlass.const_expr(cfg.kv_cache_format != "fp8"):
        return cutlass.Boolean(False)
    # Only the primary pool can be per-tensor FP8. The extra-source tag means
    # NVFP4, even when the first selected row is a masked hole. Only issuing
    # lanes own metadata, so broadcast that tag to every conversion lane.
    extra = Int32(
        prims.shfl_sync(
            thread_mask=0xFFFFFFFF,
            val=Int32(use_extra),
            offset=0,
            mask_and_clamp=0x1F,
            kind=prims.Shfl.IDX,
        )
    )
    return extra == Int32(0)


# ===========================================================================
# NVFP4 conversion
# ===========================================================================

NVFP4_DATA_BYTES = 256


NVFP4_SCALE_BYTES = 32


@cute.jit
def nvfp4_scale_byte_offset(block, rope_dim: cutlass.Constexpr[int] = 64):
    """Offset within the scale region; only the mixed format permutes it."""
    if cutlass.const_expr(rope_dim == 0):
        return block
    return 8 * (block % 4) + block // 4


def nvfp4_row_bytes(rope_dim: int) -> int:
    """Packed D512, optional unscaled FP8 tail, and block16 E4M3 scales."""
    return NVFP4_DATA_BYTES + rope_dim + NVFP4_SCALE_BYTES


@cute.jit
def convert_nvfp4x8(word: Int32, scale: Int32):
    """Convert eight packed E2M1 values with one E4M3 block-scale byte.

    ``word`` contains eight consecutive four-bit values; ``scale`` stores the
    E4M3 byte in its low eight bits. Return two words containing eight E4M3
    bytes, rounded once with saturation for ordinary FP8 MMA. No tensor-wide
    NVFP4 scale is applied.
    """
    if cutlass.const_expr(not _has_packed_e4m3_e2m1_mul()):
        # Older DSL builds lack packed multiply even with a newer system
        # toolkit. Convert dense nibbles directly instead of unpacking to
        # bytes only to repack them inside the multiply fallback.
        return _cvt_e2m1_word_to_e4m3_words(word, scale)
    scales = (scale & Int32(255)) * Int32(0x01010101)
    # Each 16-bit half holds four contiguous E2M1 nibbles. CUDA 13.4's
    # dense packed operand avoids padded-nibble masks and output permutations.
    values = cute.arch.inline_ptx(
        "{ .reg .b16 lo, hi;\n"
        "  mov.b32 {lo, hi}, {$r0};\n"
        "  mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite {$w0}, lo, {$r1};\n"
        "  mul.rn.e4m3x4.e2m1x4.e4m3x4.satfinite {$w1}, hi, {$r1};\n"
        "}",
        write_only_types=[Int32, Int32],
        read_only_args=[word, scales],
    )
    return values[0], values[1]


# ===========================================================================
# TMA descriptors, gathering and staging
# ===========================================================================


@cute.jit
def make_whole_record_map(pool, record_bytes: cutlass.Constexpr[int]):
    """Describe a complete byte record as Uint16 values for unswizzled gather4.

    All supported records fit the 256-element TMA box limit using Uint16.
    A 32-byte raw row pitch makes each four-row destination 128-byte aligned;
    global rows retain their original record stride and are not repacked.
    """
    word_bytes = 2
    raw_bytes = (record_bytes + 31) // 32 * 32
    packed = cute.make_tensor(
        cute.make_ptr(
            cutlass.Uint16,
            pool.iterator.toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        cute.make_layout(
            (record_bytes // word_bytes, pool.shape[2]),
            stride=(1, pool.stride[2] // word_bytes),
        ),
    )
    return create_tensor_map_tiled_from_view(
        packed,
        box_dims=(raw_bytes // word_bytes, 1),
        stride_order=(0, 1),
        swizzle=cuda.TensorMapSwizzle.none,
        l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
    )


@cute.jit
def make_nvfp4_kv_maps(
    primary,
    extra,
    rope_dim: cutlass.Constexpr[int],
    whole_record: cutlass.Constexpr[bool] = False,
    primary_format: cutlass.Constexpr[str] = "nvfp4",
    extra_format: cutlass.Constexpr[str] = "nvfp4",
):
    """Build paired payload, block-scale and optional RoPE tensor maps.

    Each pool is the internal [1, record_bytes, physical_rows] byte view.
    Return primary/extra maps in payload, scales, RoPE order. Whole-record
    maps preserve the token-interleaved representation; panel maps place
    128 logical values in the MMA swizzle. FP8 primary data has no trailer,
    so its unused scale/RoPE descriptors are safe aliases inside that pool."""
    nope = 512
    maps = ()
    for pool, fmt in ((primary, primary_format), (extra, extra_format)):
        fp8 = fmt == "fp8"
        data = cute.make_tensor(
            pool.iterator,
            cute.make_layout(
                (((nope + 127) // 128) * 128, pool.shape[2]),
                stride=(1, pool.stride[2] * (1 if fp8 else 2)),
            ),
        )
        scales = cute.make_tensor(
            pool.iterator + (0 if fp8 else nope // 2 + rope_dim),
            cute.make_layout(
                (16 if fp8 else 32, pool.shape[2]), stride=(1, pool.stride[2])
            ),
        )
        rope = cute.make_tensor(
            pool.iterator + (nope // 2 if rope_dim else 0),
            cute.make_layout((64, pool.shape[2]), stride=(1, pool.stride[2])),
        )
        if cutlass.const_expr(whole_record):
            data_map = make_whole_record_map(pool, cache_record_size(fmt, rope_dim))
        elif cutlass.const_expr(fp8):
            data_map = create_tensor_map_tiled_from_view(
                data,
                box_dims=(128, 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s128b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
        else:
            data_map = create_tensor_map_tiled_from_view(
                data,
                box_dims=(128, 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s128b,
                dtype=cutlass.Float4E2M1FNx2,
                tma_format=cuda.TensorMapDataFormat.B4X16_P64,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            )
        maps += (
            data_map,
            create_tensor_map_tiled_from_view(
                scales,
                box_dims=(32, 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s32b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            ),
            create_tensor_map_tiled_from_view(
                rope,
                box_dims=(64, 1),
                stride_order=(0, 1),
                swizzle=cuda.TensorMapSwizzle.s64b,
                l2_promotion=cuda.TensorMapL2Promotion.l2_128b,
            ),
        )
    return maps[0], maps[3], maps[1], maps[4], maps[2], maps[5]


@cute.jit
def convert_nvfp4_panel(
    dst,
    scales,
    column,
    *,
    rows: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
):
    """Expand a gathered 128-column E2M1 panel in place into E4M3 MMA bytes.

    TMA B4X16_P64 leaves each eight packed bytes in a sixteen-byte slot.
    One lane reads sixteen values and their block16 scale, rounds once to
    E4M3, then fills that slot; D576 uses its permuted scale-byte order."""
    lane = cute.arch.thread_idx()[0] % 32
    for chunk in cutlass.range(lane, rows * 8, 32, unroll=4):
        row, col = chunk // 8, chunk % 8 * 16
        logical_col = column + col
        if logical_col < 512:
            byte = swizzle_byte_offset(row * 128 + col, 128)
            packed = cutlass.Pointer(dst + byte, dtype=Int32).load(count=2, alignment=8)
            block = logical_col // 16
            scale_index = nvfp4_scale_byte_offset(block, rope_dim)
            scale_byte = cutlass.Array(scales, dtype=cutlass.Uint8)[
                swizzle_byte_offset(row * 32 + scale_index, 32)
            ]
            a, b = convert_nvfp4x8(packed[0], Int32(scale_byte))
            c, d = convert_nvfp4x8(packed[1], Int32(scale_byte))
            store_shared_u32x4(dst, byte, a, b, c, d)


@cute.jit
def begin_raw_load(barrier, count):
    """Have lane zero arm a raw TMA barrier for count bytes and return its epoch token.

    The warp joins before issuing gathers. Only lane zero later consumes
    the returned token; other lanes wait through the matching warp join."""
    state = Int64(0)
    if cute.arch.thread_idx()[0] % 32 == 0:
        state = prims.mbarrier_arrive_expect_tx(barrier, Int32(count))
    cute.arch.sync_warp()
    return state


@cute.jit
def wait_raw_load(barrier, state):
    """Wait in lane zero for the armed raw TMA epoch, then release the whole warp."""
    if cute.arch.thread_idx()[0] % 32 == 0:
        while not prims.mbarrier_try_wait(barrier, state):
            pass
    cute.arch.sync_warp()


@cute.jit
def gather_cached(
    dst,
    primary_map,
    extra_map,
    column,
    barrier,
    cached,
    base,
    *,
    rows: cutlass.Constexpr[int],
    width,
):
    """Issue four-row TMA gathers from register-cached, source-tagged storage indices.

    Each issuing lane owns one quad; width is its destination bytes per row.
    The high source bit chooses the pool map and is stripped from row IDs.
    Invalid rows become the TMA out-of-bounds sentinel and are zero-filled."""
    lane = cute.arch.thread_idx()[0] % 32
    if lane < rows // 4:
        raw = cached[base]
        coords = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(4):
            row = cached[base + j] & Int32(0x7FFFFFFF)
            coords[j] = row if row != Int32(0x7FFFFFFF) else Int32(-1)
        descriptor = select_gather_map(primary_map, extra_map, raw < 0)
        gather4(
            dst + lane * 4 * width,
            descriptor,
            column,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            barrier,
        )


@cute.jit
def load_nvfp4_panel(
    dst,
    scratch,
    barrier,
    cached,
    cache_base,
    column,
    data_map,
    extra_data_map,
    scale_map,
    extra_scale_map,
    rope_map,
    extra_rope_map,
    *,
    rows: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    rope_only: cutlass.Constexpr[bool],
    load_scales: cutlass.Constexpr[bool],
    fp8=False,
):
    """Gather one feature panel, applying NVFP4 block scales only for packed storage.

    The caller owns the raw barrier and scratch. Payload and scale byte
    credits match their tensor-map transfers; FP8 primary and unscaled
    RoPE bypass conversion. MMA publication remains the caller's job."""
    if cutlass.const_expr(rope_only):
        state = begin_raw_load(barrier, rows * 64)
        gather_cached(
            dst,
            rope_map,
            extra_rope_map,
            Int32(0),
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=64,
        )
        wait_raw_load(barrier, state)
    else:
        # B4X16_P64 writes 8 packed bytes followed by 8 pad bytes per 16
        # values. Its completion count is the packed bytes, not padded SMEM.
        state = begin_raw_load(
            barrier, rows * (128 if fp8 else 64 + (32 if load_scales else 0))
        )
        gather_cached(
            dst,
            data_map,
            extra_data_map,
            column,
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=128,
        )
        if load_scales and not fp8:
            gather_cached(
                scratch,
                scale_map,
                extra_scale_map,
                Int32(0),
                barrier,
                cached,
                cache_base,
                rows=rows,
                width=32,
            )
        wait_raw_load(barrier, state)
        if not fp8:
            convert_nvfp4_panel(dst, scratch, column, rows=rows, rope_dim=rope_dim)


@cute.jit
def load_nvfp4_panels(
    destinations,
    columns,
    scratch,
    barrier,
    cached,
    cache_base,
    data_map,
    extra_data_map,
    scale_map,
    extra_scale_map,
    rope_map,
    extra_rope_map,
    *,
    rows: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    separate_rope: cutlass.Constexpr[bool],
    fp8=False,
):
    """Overlap all feature-panel gathers before one raw-data wait."""
    panels = len(columns)
    state = begin_raw_load(
        barrier,
        rows
        * (
            panels * (128 if fp8 else 64)
            + (0 if fp8 else 32)
            + (64 if separate_rope else 0)
        ),
    )
    for part in cutlass.range_constexpr(panels):
        gather_cached(
            destinations[part],
            data_map,
            extra_data_map,
            columns[part],
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=128,
        )
    if not fp8:
        gather_cached(
            scratch,
            scale_map,
            extra_scale_map,
            Int32(0),
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=32,
        )
    if cutlass.const_expr(separate_rope):
        gather_cached(
            destinations[panels],
            rope_map,
            extra_rope_map,
            Int32(0),
            barrier,
            cached,
            cache_base,
            rows=rows,
            width=64,
        )
    wait_raw_load(barrier, state)
    for part in cutlass.range_constexpr(panels):
        if not fp8:
            convert_nvfp4_panel(
                destinations[part], scratch, columns[part], rows=rows, rope_dim=rope_dim
            )


@cute.jit
def issue_whole_record(
    dst, barrier, cached, data_map, extra_data_map, *, issuer, row_bytes
):
    """One gather per quad for a complete NVFP4 or per-tensor FP8 storage record."""
    begin_raw_load(barrier, Int32(64) * row_bytes)
    gather_cached(
        dst + issuer * Int32(64) * row_bytes,
        data_map,
        extra_data_map,
        Int32(0),
        barrier,
        cached,
        0,
        rows=64,
        width=row_bytes,
    )


@cute.jit
def convert_nvfp4_record(
    dst,
    thread_idx,
    *,
    stage_bytes: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    raw=None,
    raw_empty=None,
):
    """Snapshot two16-value rows per lane, then write conflict-free panels."""
    row_bytes = nvfp4_row_bytes(rope_dim)
    if cutlass.const_expr(raw is None):
        raw = dst
    packed = cutlass.Array(Int32, 32, space=cutlass.AddressSpace.rmem)
    scales = cutlass.Array(Int32, 8, space=cutlass.AddressSpace.rmem)
    rope = cutlass.Array(Int32, 8, space=cutlass.AddressSpace.rmem)
    lane = thread_idx & Int32(31)
    first_row = thread_idx >> Int32(5)
    column = lane * Int32(16)
    sf_column = nvfp4_scale_byte_offset(lane, rope_dim)
    for chunk in cutlass.range_constexpr(8):
        row0 = first_row + Int32(chunk * 16)
        row1 = row0 + Int32(8)
        words0 = cutlass.Pointer(
            raw + row0 * Int32(row_bytes) + column // Int32(2), dtype=Int32
        ).load(count=2, alignment=8)
        words1 = cutlass.Pointer(
            raw + row1 * Int32(row_bytes) + column // Int32(2), dtype=Int32
        ).load(count=2, alignment=8)
        packed[chunk * 4], packed[chunk * 4 + 1] = words0[0], words0[1]
        packed[chunk * 4 + 2], packed[chunk * 4 + 3] = words1[0], words1[1]
        sf0 = cutlass.Array(raw, dtype=cutlass.Uint8)[
            row0 * Int32(row_bytes) + Int32(NVFP4_DATA_BYTES + rope_dim) + sf_column
        ]
        sf1 = cutlass.Array(raw, dtype=cutlass.Uint8)[
            row1 * Int32(row_bytes) + Int32(NVFP4_DATA_BYTES + rope_dim) + sf_column
        ]
        scales[chunk] = Int32(sf0) | (Int32(sf1) << Int32(8))
        if cutlass.const_expr(rope_dim):
            rope0 = cutlass.Pointer(
                raw
                + row0 * Int32(row_bytes)
                + Int32(NVFP4_DATA_BYTES)
                + lane * Int32(2),
                dtype=cutlass.Uint16,
            ).load()
            rope1 = cutlass.Pointer(
                raw
                + row1 * Int32(row_bytes)
                + Int32(NVFP4_DATA_BYTES)
                + lane * Int32(2),
                dtype=cutlass.Uint16,
            ).load()
            rope[chunk] = Int32(rope0) | (Int32(rope1) << Int32(16))
    if cutlass.const_expr(raw_empty is not None):
        # Raw storage is disjoint from every expanded destination. Each
        # warp snapshots all of its bytes before its release-arrival. The
        # raw-empty barrier has eight arrivals; the metadata TMA producer
        # acquires it and proxy-fences before any overwrite.
        cute.arch.sync_warp()
        if prims.elect_sync():
            prims.mbarrier_arrive(raw_empty, scope=prims.MemScope.CTA)
    else:
        # In-place expansion must join all readers before any output store.
        prims.barrier_cta_sync(barrier_id=12, thread_count=256)
    panel = lane >> Int32(3)
    local_column = (lane & Int32(7)) * Int32(16)
    for chunk in cutlass.range_constexpr(8):
        for pair in cutlass.range_constexpr(2):
            row = first_row + Int32(chunk * 16 + pair * 8)
            sf = (scales[chunk] >> Int32(pair * 8)) & Int32(255)
            a, b = convert_nvfp4x8(packed[chunk * 4 + pair * 2], sf)
            c, d = convert_nvfp4x8(packed[chunk * 4 + pair * 2 + 1], sf)
            store_shared_u32x4(
                dst + panel * Int32(stage_bytes),
                swizzle_byte_offset(row * Int32(128) + local_column, 128),
                a,
                b,
                c,
                d,
            )
            if cutlass.const_expr(rope_dim):
                packed_rope = cutlass.Uint16(
                    (rope[chunk] >> Int32(pair * 16)) & Int32(65535)
                )
                cutlass.Pointer(
                    dst
                    + Int32(4 * stage_bytes)
                    + swizzle_byte_offset(row * Int32(64) + lane * Int32(2), 64),
                    dtype=cutlass.Uint16,
                ).store(packed_rope)


@cute.jit
def copy_fp8_record(dst, raw, thread_idx, raw_empty):
    """Reorder unmodified FP8 bytes into four retained MMA panels."""
    lane = thread_idx % Int32(32)
    warp = thread_idx // Int32(32)
    for i in cutlass.range(16, unroll=2):
        row = warp + i * Int32(8)
        offset = (lane // Int32(8)) * Int32(16384) + swizzle_byte_offset(
            row * Int32(128) + (lane % Int32(8)) * Int32(16), 128
        )
        words = cutlass.Pointer(
            raw + row * Int32(512) + lane * Int32(16), dtype=Int32
        ).load(count=4, alignment=16)
        store_shared_u32x4(dst, offset, words[0], words[1], words[2], words[3])
    cute.arch.sync_warp()
    if prims.elect_sync():
        prims.mbarrier_arrive(raw_empty, scope=prims.MemScope.CTA)


@cute.jit
def prefetch_nvfp4_kv_row(primary, extra, route, cfg):
    """Warm the complete selected record while metadata runs ahead of conversion."""
    row = route & Int32(0x7FFFFFFF)
    address = primary.iterator.toint()
    capacity = Int32(primary.shape[2])
    stride = Int32(primary.stride[2])
    width = Int32(cache_record_size(cfg.kv_cache_format, cfg.rope_dim))
    if route < Int32(0):
        address = extra.iterator.toint()
        capacity = Int32(extra.shape[2])
        stride = Int32(extra.stride[2])
        width = Int32(cache_record_size(cfg.extra_kv_cache_format, cfg.rope_dim))
    if row < capacity:
        address += Int64(row) * Int64(stride)
        for line in cutlass.range_constexpr(3):
            cute.arch.inline_ptx(
                "prefetch.global.L2 [{$r0}];",
                read_only_args=[address + Int64(line * 128)],
            )
        if cutlass.const_expr(cfg.rope_dim):
            cute.arch.inline_ptx(
                "prefetch.global.L2 [{$r0}];", read_only_args=[address + Int64(351)]
            )
        if cutlass.const_expr(cfg.has_fp8_primary):
            if width == Int32(512):
                for line in cutlass.range_constexpr(3, 4):
                    cute.arch.inline_ptx(
                        "prefetch.global.L2 [{$r0}];",
                        read_only_args=[address + Int64(line * 128)],
                    )
