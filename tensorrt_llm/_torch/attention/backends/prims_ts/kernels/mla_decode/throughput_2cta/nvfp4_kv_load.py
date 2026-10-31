# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Convert disjoint K rows once, then exchange peer V features through DSMEM.

Each CTA owns 64 of 128 token rows. It writes its local V half and sends the
other 256 features per row to its peer. Register stores permit unequal K/V
stage counts; bulk copies retain their K source through coupled K/V reuse.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.pipeline import PipelineState

from ..helpers.gather import gather4_quad
from ..helpers.nvfp4_kv import (
    convert_nvfp4x8,
    nvfp4_row_bytes,
    nvfp4_scale_byte_offset,
    publish_nvfp4_kv,
    store_shared_u32x4,
    swizzle_byte_offset,
)


@cute.jit
def shared_byte_view(ptr, count: cutlass.Constexpr[int]):
    """Expose count bytes of CTA-local raw scratch for packed scale/payload reads."""
    return cutlass.Array(ptr, dtype=cutlass.Uint8, shape=(count,), addrspace=3)


@cute.jit
def arm_raw_load(barrier, tx_bytes: cutlass.Constexpr[int], reset=False):
    """Initialize once per work tile, then advance the raw barrier every KV tile."""
    if reset:
        prims.mbarrier_init(barrier, 1)
        prims.fence_mbarrier_init()
    prims.mbarrier_arrive_expect_tx(barrier, tx_bytes)


@cute.jit
def wait_raw_load(barrier, phase):
    """Acquire completion of the current per-warp TMA epoch before reading raw bytes."""
    while not prims.mbarrier_wait_parity(barrier, phase, prims.MBarrierWait.TRY):
        pass


@cute.jit
def load_shared_row_quad(smem_sparse_offsets, offset):
    """Four storage rows of one gather quad from the offsets ring; masked rows keep the OOB sentinel."""
    raw = smem_sparse_offsets.data_ptr(offset).load(count=4, alignment=16)
    rows = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
    for j in cutlass.range_constexpr(4):
        rows[j] = raw[j]
    return rows


@cute.jit
def load_shared_words(addr, count: cutlass.Constexpr[int]):
    """Vector shared load of ``count`` words (2 or 4) from a suitably aligned address."""
    return cutlass.inttoptr(addr, 3, Int32).load(count=count, alignment=4 * count)


def nvfp4_kv_scratch_bytes(rope_dim):
    """Return bytes for the 64 NVFP4 rows owned by one CTA; FP8 gathers directly into K."""
    return 64 * nvfp4_row_bytes(rope_dim)


@cute.jit
def issue_k_gathers(k, raw, dst, barrier, rows, q: cutlass.Constexpr[int], warp, fp8):
    """Gather one warp-owned four-row group into raw NVFP4 scratch or FP8 K panels.

    FP8 primary emits four 128-byte panel gathers. NVFP4 emits one whole
    record gather and converts later; both routes use the same source tag."""
    if fp8:
        quad = warp * Int32(2) + Int32(q)
        for part in cutlass.range_constexpr(4):
            gather4_quad(
                dst + part * 8192 + quad * Int32(512),
                k.tma_desc_c_latent,
                k.tma_desc_c_rope,
                Int32(part * 128),
                rows,
                barrier,
            )
    else:
        gather4_quad(
            raw + q * 4 * nvfp4_row_bytes(k.cfg.rope_dim),
            k.tma_desc_c_latent,
            k.tma_desc_c_rope,
            Int32(0),
            rows,
            barrier,
        )


@dsl_user_op
def store_v_remote(dst, offset, done, owner, a, b, c, d, *, loc=None, ip=None):
    """Send sixteen converted V bytes and credit the peer completion barrier.

    Offset is in bytes within an acquired peer V stage. Prefer the native
    vector store for compiler scheduling. DSL 4.7 validates tuples but then
    converts them to a scalar; use inline PTX only for that wrapper failure.
    """
    destination = cute.make_ptr(
        cutlass.Int32,
        dst.toint(Int32) + offset,
        cute.AddressSpace.smem,
        assumed_align=16,
        loc=loc,
        ip=ip,
    )
    completion = cute.make_ptr(
        cutlass.Int64,
        done.toint(Int32),
        cute.AddressSpace.smem,
        assumed_align=8,
        loc=loc,
        ip=ip,
    )
    try:
        cute.arch.store_async_dsmem(
            destination, (a, b, c, d), completion, owner, loc=loc, ip=ip
        )
        return
    except DSLRuntimeError as error:
        if "to integer conversion is not supported" not in str(error):
            raise

    remote = prims.mapa(dst + offset, owner, loc=loc, ip=ip)
    remote_completion = prims.mapa(done, owner, loc=loc, ip=ip)
    cute.arch.inline_ptx(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}}, [{$r5}];",
        read_only_args=[remote.ir_value(), a, b, c, d, remote_completion.ir_value()],
        loc=loc,
        ip=ip,
    )


@cute.jit
def convert_k_and_exchange_v(
    kraw,
    kdst,
    vdst,
    lane,
    warp,
    cta,
    barrier,
    done,
    *,
    fp8: cutlass.Constexpr[bool],
    register_exchange: cutlass.Constexpr[bool],
    rope_dim: cutlass.Constexpr[int],
):
    """Convert eight owned token rows and distribute their V features across the CTA pair.

    Each lane handles sixteen columns. Its CTA keeps one 256-feature V
    half and sends the other half to its peer. FP8 primary K is already in
    place; NVFP4 is expanded once and reused for both QK and PV. Register
    exchange permits unequal K/V stage counts; bulk exchange retains K
    as the copy source until both V consumers finish."""
    stride = nvfp4_row_bytes(rope_dim)
    for i in cutlass.range(8, unroll=2 if fp8 else 8):
        row = warp * Int32(8) + Int32(i)
        offset = (lane >> Int32(3)) * Int32(8192) + swizzle_byte_offset(
            row * Int32(128) + (lane & Int32(7)) * Int32(16), 128
        )
        a, b, c, d = Int32(0), Int32(0), Int32(0), Int32(0)
        if cutlass.const_expr(fp8):
            k8 = load_shared_words(kdst.toint(Int32) + offset, 4)
            a, b, c, d = k8[0], k8[1], k8[2], k8[3]
        else:
            k4 = load_shared_words(
                kraw.toint(Int32) + Int32(i * stride) + lane * Int32(8), 2
            )
            sf = Int32(
                shared_byte_view(kraw, 8 * stride)[
                    Int32(i * stride + 256 + rope_dim)
                    + nvfp4_scale_byte_offset(lane, rope_dim)
                ]
            )
            a, b = convert_nvfp4x8(k4[0], sf)
            c, d = convert_nvfp4x8(k4[1], sf)
        # FP8 panel TMA already populated K. Only NVFP4 needs a K write;
        # both formats reuse these registers for local/remote V publication.
        if cutlass.const_expr(not fp8):
            store_shared_u32x4(kdst, offset, a, b, c, d)
        owner = (lane >> Int32(3)) & Int32(1)
        voffset = (cta * Int32(2) + (lane >> Int32(4))) * Int32(8192)
        voffset += swizzle_byte_offset(
            row * Int32(128) + (lane & Int32(7)) * Int32(16), 128
        )
        if owner == cta:
            store_shared_u32x4(vdst, voffset, a, b, c, d)
        elif cutlass.const_expr(register_exchange):
            store_v_remote(vdst, voffset, done, owner, a, b, c, d)
    if cutlass.const_expr(rope_dim):
        r, c = lane >> Int32(2), lane & Int32(3)
        wr = load_shared_words(
            kraw.toint(Int32) + Int32(256) + r * Int32(stride) + c * Int32(16), 4
        )
        row = warp * Int32(8) + r
        store_shared_u32x4(
            kdst,
            Int32(32768) + swizzle_byte_offset(row * Int32(64) + c * Int32(16), 64),
            wr[0],
            wr[1],
            wr[2],
            wr[3],
        )
    publish_nvfp4_kv(barrier, 8 * (512 + rope_dim), cta_group=2)
    if cutlass.const_expr(not register_exchange):
        # This warp owns all eight source rows; K publication supplied its
        # proxy fence and warp join. Coupled K/V slots retain the copy source.
        if prims.elect_sync():
            peer = cta ^ Int32(1)
            for part in cutlass.range_constexpr(2):
                source = (
                    kdst + (peer + Int32(part * 2)) * Int32(8192) + warp * Int32(1024)
                )
                destination = (
                    vdst
                    + (cta * Int32(2) + Int32(part)) * Int32(8192)
                    + warp * Int32(1024)
                )
                prims.cp_async_bulk_shared_cluster_shared_cta(
                    prims.mapa(destination, peer),
                    source,
                    prims.mapa(done, peer),
                    Int32(1024),
                )


@cute.jit
def load_nvfp4_source_stage(
    k, v, stage_info, v_stage, page_stage, *, fp8_source: cutlass.Constexpr[bool]
):
    """Specialize the complete producer so source-format branches stay together."""
    cfg = k.cfg
    assert cfg.load_num_warps == 8
    assert cfg.sparse_offset_stages >= 2
    register_exchange = cfg.load_v_stage > cfg.load_k_stage
    lane = cute.arch.thread_idx()[0] & Int32(31)
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(cfg.load_k_warp_id)
    cta = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    kdst = k.smem_k.data_ptr(stage_info.stage_idx * cfg.smem_k_stage_elems)
    vdst = v.smem_v.data_ptr(v_stage * cfg.smem_v_stage_elems)
    raw_bytes = 8 * nvfp4_row_bytes(cfg.rope_dim)
    raw = k.smem_raw.data_ptr(warp * Int32(raw_bytes))
    raw_bar = k.raw_mbar.data_ptr(warp)
    done = k.raw_mbar.data_ptr(Int32(cfg.load_num_warps) + v_stage)
    phase = v.current_slot[1]
    # The group-2 V empty barrier completes only after both CTAs' V readers
    # retire. Local acquire therefore permits writes to either V destination.
    # Completion may precede expect_tx: the pending arrival keeps this phase
    # open, and the signed transaction counter permits that ordering (PTX9.7.15.16).
    if warp == Int32(0) and prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(done, Int32(16384))
    base = page_stage * Int32(128) + cta * Int32(64)
    loop = Int32(stage_info.loop_offset)
    if fp8_source or loop == Int32(0):
        if prims.elect_sync():
            if fp8_source:
                arm_raw_load(raw_bar, 8 * 512, loop == Int32(0))
            else:
                arm_raw_load(raw_bar, raw_bytes, True)
            for q in cutlass.range_constexpr(2):
                rows = load_shared_row_quad(
                    k.smem_sparse_offsets, base + warp * Int32(8) + Int32(q * 4)
                )
                issue_k_gathers(k, raw, kdst, raw_bar, rows, q, warp, fp8_source)
    has_next = loop + Int32(1) < Int32(stage_info.work_tile.k_tile_count)
    next_stage = Int32(0)
    if has_next:
        # Read ahead under the same future consumer token. This wait does not
        # advance/release it; the normal next iteration still owns its release.
        # Construct new scalar values: DSL 4.7 can alias the staged fields
        # returned by clone(), so advance() also advances the live consumer
        # and releases the next ring slot instead of the one being read.
        current = k.page_window.state_src.consumer_state
        next_stage = (current.index + Int32(1)) % Int32(cfg.sparse_offset_stages)
        upcoming = PipelineState(
            cfg.sparse_offset_stages,
            current.count + Int32(1),
            next_stage,
            current.phase ^ Int32(next_stage == Int32(0)),
        )
        k.page_window.pipeline.consumer_wait(upcoming)
    wait_raw_load(raw_bar, loop & Int32(1))
    if fp8_source:
        convert_k_and_exchange_v(
            raw,
            kdst,
            vdst,
            lane,
            warp,
            cta,
            stage_info.barrier,
            done,
            fp8=True,
            register_exchange=register_exchange,
            rope_dim=cfg.rope_dim,
        )
    else:
        convert_k_and_exchange_v(
            raw,
            kdst,
            vdst,
            lane,
            warp,
            cta,
            stage_info.barrier,
            done,
            fp8=False,
            register_exchange=register_exchange,
            rope_dim=cfg.rope_dim,
        )

    # Prefetch only raw NVFP4 K; per-tensor FP8 writes the next acquired K stage directly.
    if has_next:
        # K publication already joined all raw readers.
        next_base = next_stage * Int32(128) + cta * Int32(64) + warp * Int32(8)
        next_fp8 = tile_is_fp8(k, stage_info, loop + Int32(1))
        if not next_fp8 and prims.elect_sync():
            arm_raw_load(raw_bar, raw_bytes)
            rows0 = load_shared_row_quad(k.smem_sparse_offsets, next_base)
            rows1 = load_shared_row_quad(k.smem_sparse_offsets, next_base + Int32(4))
            issue_k_gathers(k, raw, kdst, raw_bar, rows0, 0, warp, False)
            issue_k_gathers(k, raw, kdst, raw_bar, rows1, 1, warp, False)
    # For bulk copies, the next coupled K/V acquisitions protect K until both
    # V destinations have consumed the previous copies. Register stores need
    # no retained K source after their payload has been issued.
    # Only one publisher needs to gate the V full barrier on incoming data.
    # Other publishers can advance; the group-2 MMA cannot start until this
    # publisher contributes its final bytes after observing completion.
    if warp == Int32(0):
        while not prims.mbarrier_try_wait_parity(
            done, phase, scope=prims.MBarrierScope.CLUSTER, order=prims.MemOrder.ACQUIRE
        ):
            pass


@cute.jit
def publish_nvfp4_v(barrier, count):
    # K publication has already proxy-fenced and warp-joined every ordinary
    # local V store. The remaining writes are async DSMEM transactions.
    # Warp0's incoming completion acquire is ordered before this release;
    # its withheld credit keeps V-full closed until all received bytes arrive.
    """Publish V only after local stores and incoming DSMEM transfers are complete.

    K publication already fenced the local writes. The designated publisher
    acquired peer completion before releasing the cluster-wide V-full credit."""
    if prims.elect_sync():
        prims.fence_sync_restrict(prims.MemOrder.RELEASE)
        prims.mbarrier_complete_tx(
            prims.mapa(barrier, Int32(0)), Int32(count), scope=prims.MemScope.CLUSTER
        )


@cute.jit
def tile_is_fp8(k, stage_info, offset):
    """Select the primary FP8 producer for the current or prefetched KV tile.

    Compare logical tile position with the live primary boundary cached
    by the metadata task. The extra source always selects NVFP4."""
    cfg = k.cfg
    if cutlass.const_expr(cfg.kv_cache_format != "fp8"):
        return cutlass.Boolean(False)
    extra = cutlass.Boolean(False)
    if cutlass.const_expr(k.page_offsets.has_extra):
        primary_tiles = Int32(0)
        if cutlass.const_expr(k.page_offsets.is_bound):
            primary_tiles = (
                Int32(k.page_offsets.primary_length) + Int32(127)
            ) // Int32(128)
        else:
            primary_tiles = k.request_lengths[2]
        extra = Int32(stage_info.work_tile.k_index_base) + offset >= primary_tiles
    return not extra


@cute.jit
def load_nvfp4_kv_stage(k, v, stage_info, v_stage, page_stage):
    """Select the format once before the specialized gather/conversion path."""
    if tile_is_fp8(k, stage_info, Int32(stage_info.loop_offset)):
        load_nvfp4_source_stage(k, v, stage_info, v_stage, page_stage, fp8_source=True)
    else:
        load_nvfp4_source_stage(k, v, stage_info, v_stage, page_stage, fp8_source=False)
