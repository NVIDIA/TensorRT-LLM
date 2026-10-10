"""VC-Attention operand preparation kernels (CuTe DSL).

Small kernels turn bf16/fp16 ``[B, S, H, D]`` K/V (and Q under
VC-Attention-QK8) into the prims_ts VC-Attention operands with each tensor read
once (V twice) and written once:

* :class:`VcKvPass1`: per (batch*head, 128-token K/V tile) gathers the permuted
  tokens, writes the permuted K rows, and writes the V tile mean and the
  residual amax;
* :class:`VcKvPass2`: after the per-channel V residual scale is known, writes
  the E4M3 V residuals and the packed bf16 tile-mean UMMA operand;
* :class:`VcQKPass` (VC-Attention-QK8 only): per 128-token block, the
  Hadamard-rotated E4M3 Q, and the permuted, centred, rotated E4M3 K, with one
  scale per block in the sage flat scale layout.

With ``repair`` the token order is the input order, the partial last tile
moves ``tail_shift`` rows down to leave room for the V repair tiles, and pass
2 writes each token's residual energy instead of the mean operand.

Following the paper, V residuals get one E4M3 scale per (batch, head,
channel) and the block means are stored divided by that scale. Under QK16 Q
and K stay in their input dtype.

Every kernel runs 256 threads. Thread ``t`` owns head_dim columns
``(t % 16) * 8 .. + 8`` (one 16-byte vector) of rows ``t // 16 + 16 * i``
for ``i < 8``. Rows stream through registers, never a tile-sized slab, and the
per-column state is 8 wide: the pre-pass is HBM-latency-bound and needs the
occupancy (a 128-value fp32 slab per thread capped it at 2 CTAs/SM and 17% of
DRAM bandwidth; 16-wide column state at 96 registers still at 30%).
"""

from __future__ import annotations

import functools
import math

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from ..vc_attention import (
    E4M3_MAX,
    VC_K_BLOCK_SIZE,
    VC_MEAN_MMA_K,
    VC_MEAN_GROUP_TILES,
    VC_MEAN_OPERANDS,
    vc_mean_group_shape,
    vc_mean_operand_shape,
)
from .fmha_decode.fmha_decode_resources.helpers_common import _pack_float2_to_bf16
from .fmha_decode.fmha_decode_resources.helpers_softmax import _pack_float4_to_fp8_e4m3
from ..vc_attention import vc_repair_kv_len

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
# The kernels are specialized for head_dim 128.
_D = 128
_THREADS = 256
_TILE = VC_K_BLOCK_SIZE
_COLS_PER_THREAD = 8
_COL_GROUPS = _D // _COLS_PER_THREAD
_ROW_GROUPS = _THREADS // _COL_GROUPS
_ROWS_PER_THREAD = _TILE // _ROW_GROUPS
_MU_ELEMS = math.prod(vc_mean_operand_shape(_D))
_HADAMARD_NORM = 1.0 / math.sqrt(_D)


@cute.jit
def _abs_f32(x: Float32) -> Float32:
    return cute.arch.fmax(x, -x)


@cute.jit
def _hadamard128_row(vals, base: cutlass.Constexpr[int], cg: Int32):
    """In-place *unnormalised* 128-point Walsh-Hadamard transform of one row
    whose 8 columns ``vals[base : base + 8]`` live in this thread and whose
    other 120 columns live in the 15 neighbouring lanes ``cg ^ {1, 2, 4, 8}``.
    The caller folds the 1/sqrt(128) normalisation into its scale. Butterflies
    run as packed f32x2 pairs: the pass is instruction-issue-bound."""
    for h in (1, 2, 4):
        # Pairs (i, i ^ h) for i with bit h clear, two pairs per packed op.
        lo = [i for i in range(_COLS_PER_THREAD) if (i & h) == 0]
        for q in cutlass.range_constexpr(0, len(lo), 2):
            i0, i1 = lo[q], lo[q + 1]
            a = (vals[base + i0], vals[base + i1])
            b = (vals[base + (i0 ^ h)], vals[base + (i1 ^ h)])
            vals[base + i0], vals[base + i1] = cute.arch.add_packed_f32x2(a, b)
            vals[base + (i0 ^ h)], vals[base + (i1 ^ h)] = cute.arch.sub_packed_f32x2(
                a, b
            )
    for sbit in cutlass.range_constexpr(4):
        stride = 1 << sbit
        # Lanes with the bit clear take (mine + other); lanes with it set take (other - mine).
        sign = Float32(1.0) - Float32(2.0) * Float32((cg >> sbit) & 1)
        for j in cutlass.range_constexpr(0, _COLS_PER_THREAD, 2):
            m0 = vals[base + j]
            m1 = vals[base + j + 1]
            o0 = cute.arch.shuffle_sync_bfly(m0, stride)
            o1 = cute.arch.shuffle_sync_bfly(m1, stride)
            vals[base + j], vals[base + j + 1] = cute.arch.fma_packed_f32x2(
                (m0, m1), (sign, sign), (o0, o1)
            )


@cute.jit
def _block_max(v: Float32, red_smem, tidx: Int32) -> Float32:
    """Max over the threads of the CTA."""
    for shift in cutlass.range_constexpr(5):
        v = cute.arch.fmax(v, cute.arch.shuffle_sync_bfly(v, 1 << shift))
    if tidx % 32 == 0:
        red_smem[tidx // 32] = v
    cute.arch.sync_threads()
    r = red_smem[0]
    for w in cutlass.range_constexpr(1, _THREADS // 32):
        r = cute.arch.fmax(r, red_smem[w])
    cute.arch.sync_threads()
    return r


@cute.jit
def _flat_scale_offset(
    bh: Int32, num_heads: Int32, num_batch_heads: Int32, seq_len: Int32, block: Int32
) -> Int64:
    """Element offset of block ``block`` of sequence ``bh // num_heads`` in the flat
    ``[H, ceil(B*S / 128) + B - 1]`` scale layout (``sage.flat_scale_slot``)."""
    b_idx = bh // num_heads
    h_idx = bh % num_heads
    n_batch = num_batch_heads // num_heads
    numel = ((n_batch * seq_len + _TILE - 1) // _TILE) + n_batch - 1
    slot = ((b_idx * seq_len) // _TILE) + b_idx + block
    return Int64(h_idx) * Int64(numel) + Int64(slot)


@cute.jit
def _load_row(base_addr: Int64, is_bf16: cutlass.Constexpr[bool]):
    """This thread's 8 bf16/fp16 elements at ``base_addr`` as a Vector of Float32."""
    regs = cutlass.inttoptr(base_addr, mem_space=1, dtype=Int32).load(
        count=_COLS_PER_THREAD // 2, alignment=16
    )
    if cutlass.const_expr(is_bf16):
        return regs.bitcast(cutlass.BFloat16).to(Float32)
    return regs.bitcast(cutlass.Float16).to(Float32)


@cute.jit
def _store_row_fp8(
    base_addr: Int64, vals, base: cutlass.Constexpr[int], scale_inv: Float32
):
    """Quantize this thread's 8 Float32 values ``vals[base : base + 8]`` (Array
    or Vector) by ``scale_inv`` and store 8 E4M3 bytes."""
    packed = cutlass.Array(
        Int32, _COLS_PER_THREAD // 4, space=cutlass.AddressSpace.rmem
    )
    for j in cutlass.range_constexpr(_COLS_PER_THREAD // 4):
        packed[j] = _pack_float4_to_fp8_e4m3(
            vals[base + 4 * j] * scale_inv,
            vals[base + 4 * j + 1] * scale_inv,
            vals[base + 4 * j + 2] * scale_inv,
            vals[base + 4 * j + 3] * scale_inv,
        )
    cutlass.inttoptr(base_addr, mem_space=1, dtype=Int32).store(
        packed.data_ptr().load(count=_COLS_PER_THREAD // 4, alignment=8), alignment=8
    )


class VcKvPass1:
    def __init__(self, is_bf16: bool):
        self.is_bf16 = is_bf16

    @cute.kernel
    def kernel(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mKPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVAmax: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        out_len: Int32,
        tail_start: Int32,
        tail_shift: Int32,
        demean: cutlass.Constexpr[bool],
        repair: cutlass.Constexpr[bool],
        qk_fp8: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        t, bh, _ = cute.arch.block_idx()
        b = bh // num_heads
        h = bh % num_heads
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS_PER_THREAD

        smem = cutlass.utils.SmemAllocator()
        col_sums = smem.allocate_array(Float32, _ROW_GROUPS * _TILE)

        k_base = mK.iterator.toint()
        v_base = mV.iterator.toint()
        kp_base = mKPerm.iterator.toint()
        perm_base = mPerm.iterator.toint() + Int64(bh) * Int64(seq_len) * 4
        row_bytes = Int64(num_heads) * (_D * 2)

        toks = cutlass.Array(Int32, _ROWS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            tok = pos
            if cutlass.const_expr(not repair):
                tok = Int32(0)
                if pos < seq_len:
                    tok = cutlass.inttoptr(
                        perm_base + Int64(pos) * 4, mem_space=1, dtype=Int32
                    ).load()
            toks[i] = tok

        if cutlass.const_expr(not qk_fp8):
            # K, copy the permuted rows (QK8 quantizes K in VcQKPass instead).
            for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
                pos = t * _TILE + rg + _ROW_GROUPS * i
                if pos < seq_len:
                    src = (
                        k_base
                        + (Int64(b) * Int64(seq_len) + Int64(toks[i])) * row_bytes
                        + (Int64(h) * _D + col0) * 2
                    )
                    dst = (
                        kp_base
                        + (
                            Int64(b) * Int64(out_len)
                            + Int64(pos + tail_shift * Int32(pos >= tail_start))
                        )
                        * row_bytes
                        + (Int64(h) * _D + col0) * 2
                    )
                    regs = cutlass.inttoptr(src, mem_space=1, dtype=Int32).load(
                        count=_COLS_PER_THREAD // 2, alignment=16
                    )
                    cutlass.inttoptr(dst, mem_space=1, dtype=Int32).store(
                        regs, alignment=16
                    )

        # V, tile mean and residual amax. The rows stream through registers:
        # per-column sum, max and min are enough, since
        # max_i |v_i - mean| == max(vmax - mean, mean - vmin) exactly (fl() is
        # monotonic, negation exact).
        psum = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        pmax = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        pmin = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            psum[j] = Float32(0.0)
            pmax[j] = -Float32.inf
            pmin[j] = Float32.inf
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                addr = (
                    v_base
                    + (Int64(b) * Int64(seq_len) + Int64(toks[i])) * row_bytes
                    + (Int64(h) * _D + col0) * 2
                )
                row = _load_row(addr, self.is_bf16)
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    psum[j] = psum[j] + row[j]
                    pmax[j] = cute.arch.fmax(pmax[j], row[j])
                    pmin[j] = cute.arch.fmin(pmin[j], row[j])
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            col_sums[rg * _TILE + col0 + j] = psum[j]
        cute.arch.sync_threads()
        valid_rows = seq_len - t * _TILE
        if valid_rows > _TILE:
            valid_rows = Int32(_TILE)
        inv_count = Float32(1.0) / Float32(valid_rows)
        mean = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            acc = Float32(0.0)
            for g in cutlass.range_constexpr(_ROW_GROUPS):
                acc = acc + col_sums[g * _TILE + col0 + j]
            mean[j] = acc * inv_count
            if cutlass.const_expr(not demean):
                mean[j] = Float32(0.0)
        if rg == 0:
            mean_addr = (
                mMean.iterator.toint()
                + ((Int64(bh) * Int64(num_tiles) + t) * _D + col0) * 4
            )
            cutlass.inttoptr(mean_addr, mem_space=1, dtype=Float32).store(
                mean.data_ptr().load(count=_COLS_PER_THREAD, alignment=32), alignment=32
            )
        # Per-channel residual amax of this tile (reduced over tiles on the host).
        # Threads without a valid row keep -inf/inf and contribute 0.
        cmax = cutlass.Array(Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            cmax[j] = Float32(0.0)
            if pmax[j] >= pmin[j]:
                cmax[j] = cute.arch.fmax(pmax[j] - mean[j], mean[j] - pmin[j])
        cute.arch.sync_threads()  # everyone is done reading col_sums as sums
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            col_sums[rg * _TILE + col0 + j] = cmax[j]
        cute.arch.sync_threads()
        if rg == 0:
            for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                acc = Float32(0.0)
                for g in cutlass.range_constexpr(_ROW_GROUPS):
                    acc = cute.arch.fmax(acc, col_sums[g * _TILE + col0 + j])
                cmax[j] = acc
            amax_addr = (
                mVAmax.iterator.toint()
                + ((Int64(bh) * Int64(num_tiles) + t) * _D + col0) * 4
            )
            cutlass.inttoptr(amax_addr, mem_space=1, dtype=Float32).store(
                cmax.data_ptr().load(count=_COLS_PER_THREAD, alignment=32), alignment=32
            )

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mKPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVAmax: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        num_batch_heads: Int32,
        out_len: Int32,
        tail_start: Int32,
        tail_shift: Int32,
        demean: cutlass.Constexpr[bool],
        repair: cutlass.Constexpr[bool],
        qk_fp8: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(
            mK,
            mV,
            mPerm,
            mKPerm,
            mMean,
            mVAmax,
            seq_len,
            num_heads,
            num_tiles,
            out_len,
            tail_start,
            tail_shift,
            demean,
            repair,
            qk_fp8,
        ).launch(
            grid=[num_tiles, num_batch_heads, 1],
            block=[_THREADS, 1, 1],
            smem=_ROW_GROUPS * _TILE * 4,
            stream=stream,
        )


@cute.jit
def _load_row_f32(base_addr: Int64):
    """This thread's 8 Float32 elements at ``base_addr``."""
    return cutlass.inttoptr(base_addr, mem_space=1, dtype=Float32).load(
        count=_COLS_PER_THREAD, alignment=32
    )


class VcKvPass2:
    def __init__(self, is_bf16: bool):
        self.is_bf16 = is_bf16

    @cute.kernel
    def kernel(
        self,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVScale: cute.Tensor,
        mV8: cute.Tensor,
        mMu: cute.Tensor,
        mEnergy: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        out_len: Int32,
        tail_start: Int32,
        tail_shift: Int32,
        repair: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        t, bh, _ = cute.arch.block_idx()
        b = bh // num_heads
        h = bh % num_heads
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS_PER_THREAD
        v_base = mV.iterator.toint()
        v8_base = mV8.iterator.toint()
        perm_base = mPerm.iterator.toint() + Int64(bh) * Int64(seq_len) * 4
        row_bytes = Int64(num_heads) * (_D * 2)
        v8_row_bytes = Int64(num_heads) * _D
        mean_base = mMean.iterator.toint() + (Int64(bh) * Int64(num_tiles) + t) * (
            _D * 4
        )

        smem = cutlass.utils.SmemAllocator()
        # V repair, per-thread partial residual energies of 8 channels for each of 8 rows.
        energy_part = smem.allocate_array(Float32, _THREADS * _ROWS_PER_THREAD)

        vscale_base = mVScale.iterator.toint() + Int64(bh) * (_D * 4)  # [B, H, D] fp32
        vs = _load_row_f32(vscale_base + col0 * 4)
        vs_inv = cutlass.Array(
            Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
        )
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            vs_inv[j] = Float32(1.0) / vs[j]
        mean = _load_row_f32(mean_base + col0 * 4)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = t * _TILE + rg + _ROW_GROUPS * i
            energy = Float32(0.0)
            if pos < seq_len:
                tok = pos
                if cutlass.const_expr(not repair):
                    tok = cutlass.inttoptr(
                        perm_base + Int64(pos) * 4, mem_space=1, dtype=Int32
                    ).load()
                addr = (
                    v_base
                    + (Int64(b) * Int64(seq_len) + Int64(tok)) * row_bytes
                    + (Int64(h) * _D + col0) * 2
                )
                row = _load_row(addr, self.is_bf16)
                res = cutlass.Array(
                    Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
                )
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    res[j] = (row[j] - mean[j]) * vs_inv[j]
                out_addr = (
                    v8_base
                    + (
                        Int64(b) * Int64(out_len)
                        + Int64(pos + tail_shift * Int32(pos >= tail_start))
                    )
                    * v8_row_bytes
                    + Int64(h) * _D
                    + col0
                )
                if cutlass.const_expr(repair):
                    # Residual energy of the E4M3 rounding in value units.
                    packed = cutlass.Array(
                        Int32, _COLS_PER_THREAD // 4, space=cutlass.AddressSpace.rmem
                    )
                    for j in cutlass.range_constexpr(_COLS_PER_THREAD // 4):
                        packed[j] = _pack_float4_to_fp8_e4m3(
                            res[4 * j], res[4 * j + 1], res[4 * j + 2], res[4 * j + 3]
                        )
                    words = packed.data_ptr().load(
                        count=_COLS_PER_THREAD // 4, alignment=8
                    )
                    codes = words.bitcast(cutlass.Float8E4M3FN).to(Float32)
                    for c in cutlass.range_constexpr(_COLS_PER_THREAD):
                        err = (res[c] - codes[c]) * vs[c]
                        energy = energy + err * err
                    cutlass.inttoptr(out_addr, mem_space=1, dtype=Int32).store(
                        words, alignment=8
                    )
                else:
                    _store_row_fp8(out_addr, res, 0, Float32(1.0))
            if cutlass.const_expr(repair):
                energy_part[(rg + _ROW_GROUPS * i) * _COL_GROUPS + cg] = energy
        if cutlass.const_expr(repair):
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
                pos = t * _TILE + rg + _ROW_GROUPS * i
                if (cg == 0) & (pos < seq_len):
                    acc = Float32(0.0)
                    for c in cutlass.range_constexpr(_COL_GROUPS):
                        acc = (
                            acc + energy_part[(rg + _ROW_GROUPS * i) * _COL_GROUPS + c]
                        )
                    cutlass.inttoptr(
                        mEnergy.iterator.toint()
                        + (Int64(bh) * Int64(seq_len) + Int64(pos)) * 4,
                        mem_space=1,
                        dtype=Float32,
                    ).store(acc)

        # Mean operands of this tile's group in core-matrix order, g = 8-row
        # group, c = K core matrix, r = row, k = K index. Tile 8o+i of the group
        # owns K slots 2i and 2i+1 of every row of operand o; the operands are
        # zeroed by the host, so each thread stores only its two rows' slots.
        # The operand mapping below was written for 128 threads (idx0 spans
        # exactly _MU_ELEMS); the upper half of the CTA has nothing to store.
        k_per_core = VC_MEAN_MMA_K // 2
        idx0 = tidx * (2 * k_per_core)
        g = idx0 // (8 * VC_MEAN_MMA_K)
        c = (idx0 // (8 * k_per_core)) % 2
        r0 = (idx0 // k_per_core) % 8
        d0 = g * 8 + r0
        tile_in_group = t % VC_MEAN_GROUP_TILES
        operand = tile_in_group // k_per_core
        slot = 2 * (tile_in_group % k_per_core)
        c_slot = slot // k_per_core
        if cutlass.const_expr(not repair) and tidx < 128:
            m0 = cutlass.inttoptr(
                mean_base + Int64(d0) * 4, mem_space=1, dtype=Float32
            ).load()
            m1 = cutlass.inttoptr(
                mean_base + Int64(d0 + 1) * 4, mem_space=1, dtype=Float32
            ).load()
            s0 = cutlass.inttoptr(
                vscale_base + Int64(d0) * 4, mem_space=1, dtype=Float32
            ).load()
            s1 = cutlass.inttoptr(
                vscale_base + Int64(d0 + 1) * 4, mem_space=1, dtype=Float32
            ).load()
            if c == c_slot:
                # Means are stored divided by the per-channel value scale (paper, Appendix B).
                num_groups = (
                    num_tiles + VC_MEAN_GROUP_TILES - 1
                ) // VC_MEAN_GROUP_TILES
                dst = (
                    mMu.iterator.toint()
                    + (
                        (
                            (Int64(bh) * Int64(num_groups) + t // VC_MEAN_GROUP_TILES)
                            * VC_MEAN_OPERANDS
                            + operand
                        )
                        * _MU_ELEMS
                        + idx0
                        + slot % k_per_core
                    )
                    * 2
                )
                cutlass.inttoptr(dst, mem_space=1, dtype=Int32).store(
                    _pack_float2_to_bf16(m0 / s0, m0 / s0)
                )
                cutlass.inttoptr(dst + k_per_core * 2, mem_space=1, dtype=Int32).store(
                    _pack_float2_to_bf16(m1 / s1, m1 / s1)
                )

    @cute.jit
    def __call__(
        self,
        mV: cute.Tensor,
        mPerm: cute.Tensor,
        mMean: cute.Tensor,
        mVScale: cute.Tensor,
        mV8: cute.Tensor,
        mMu: cute.Tensor,
        mEnergy: cute.Tensor,
        seq_len: Int32,
        num_heads: Int32,
        num_tiles: Int32,
        num_batch_heads: Int32,
        out_len: Int32,
        tail_start: Int32,
        tail_shift: Int32,
        repair: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(
            mV,
            mPerm,
            mMean,
            mVScale,
            mV8,
            mMu,
            mEnergy,
            seq_len,
            num_heads,
            num_tiles,
            out_len,
            tail_start,
            tail_shift,
            repair,
        ).launch(
            grid=[num_tiles, num_batch_heads, 1],
            block=[_THREADS, 1, 1],
            smem=_THREADS * _ROWS_PER_THREAD * 4,
            stream=stream,
        )


class VcQKPass:
    """VC-Attention-QK8 Q or K pass: per (batch*head, 128-token block), quantize
    a ``[B, S, H, D]`` bf16/fp16 tensor to E4M3 after the Hadamard rotation,
    one scale per block in the flat scale layout. ``gather`` reads the rows
    through a row map (the K/V permutation; under V repair the whole tiles,
    the repair tiles' selected tokens and the shifted tail, with ``-1`` for
    zero rows) and ``centre`` subtracts the per-head channel mean first (both
    for K; Q uses neither). ``seq_len`` counts the output rows, ``src_len``
    the input rows per batch, and the launch covers ``num_blocks`` blocks from
    ``block0``. The rotated block stays in registers between the amax and the
    store (8 fp32 per row per thread), so the tensor is read once and rotated
    once."""

    def __init__(self, is_bf16: bool, gather: bool, centre: bool):
        self.is_bf16 = is_bf16
        self.gather = gather
        self.centre = centre

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mPerm: cute.Tensor,
        mCentre: cute.Tensor,
        mX8: cute.Tensor,
        mScale: cute.Tensor,
        seq_len: Int32,
        src_len: Int32,
        block0: Int32,
        num_heads: Int32,
        num_batch_heads: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        nb, bh, _ = cute.arch.block_idx()
        nb = nb + block0
        b = bh // num_heads
        h = bh % num_heads
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS_PER_THREAD
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_array(Float32, _THREADS // 32)
        row_bytes = Int64(num_heads) * (_D * 2)
        x_base = (
            mX.iterator.toint()
            + Int64(b) * Int64(src_len) * row_bytes
            + (Int64(h) * _D + col0) * 2
        )
        perm_base = mPerm.iterator.toint() + Int64(bh) * Int64(seq_len) * 4
        centre = cutlass.Array(
            Float32, _COLS_PER_THREAD, space=cutlass.AddressSpace.rmem
        )
        for j in cutlass.range_constexpr(_COLS_PER_THREAD):
            centre[j] = Float32(0.0)
        if cutlass.const_expr(self.centre):
            c = _load_row_f32(mCentre.iterator.toint() + (Int64(bh) * _D + col0) * 4)
            for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                centre[j] = c[j]
        vals = cutlass.Array(
            Float32,
            _ROWS_PER_THREAD * _COLS_PER_THREAD,
            space=cutlass.AddressSpace.rmem,
        )
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = nb * _TILE + rg + _ROW_GROUPS * i
            tok = pos
            if cutlass.const_expr(self.gather):
                tok = Int32(-1)
                if pos < seq_len:
                    tok = cutlass.inttoptr(
                        perm_base + Int64(pos) * 4, mem_space=1, dtype=Int32
                    ).load()
            if (pos < seq_len) & (tok >= 0):
                row = _load_row(x_base + Int64(tok) * row_bytes, self.is_bf16)
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    vals[i * _COLS_PER_THREAD + j] = row[j] - centre[j]
            else:
                # The rotation shuffles need all 16 lanes of a row, zero rows too.
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    vals[i * _COLS_PER_THREAD + j] = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            _hadamard128_row(vals, i * _COLS_PER_THREAD, cg)
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = nb * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                for j in cutlass.range_constexpr(_COLS_PER_THREAD):
                    amax = cute.arch.fmax(
                        amax, _abs_f32(vals[i * _COLS_PER_THREAD + j])
                    )
        amax = _block_max(amax, red, tidx)
        # vals hold the unnormalised transform; the 1/sqrt(128) goes into the scale.
        scale = cute.arch.fmax(
            amax * Float32(_HADAMARD_NORM / E4M3_MAX), Float32(1e-12)
        )
        if tidx == 0:
            cutlass.inttoptr(
                mScale.iterator.toint()
                + _flat_scale_offset(bh, num_heads, num_batch_heads, seq_len, nb) * 4,
                mem_space=1,
                dtype=Float32,
            ).store(scale)
        scale_inv = Float32(_HADAMARD_NORM) / scale
        x8_row_bytes = Int64(num_heads) * _D
        x8_base = (
            mX8.iterator.toint()
            + Int64(b) * Int64(seq_len) * x8_row_bytes
            + Int64(h) * _D
            + col0
        )
        for i in cutlass.range_constexpr(_ROWS_PER_THREAD):
            pos = nb * _TILE + rg + _ROW_GROUPS * i
            if pos < seq_len:
                _store_row_fp8(
                    x8_base + Int64(pos) * x8_row_bytes,
                    vals,
                    i * _COLS_PER_THREAD,
                    scale_inv,
                )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mPerm: cute.Tensor,
        mCentre: cute.Tensor,
        mX8: cute.Tensor,
        mScale: cute.Tensor,
        seq_len: Int32,
        src_len: Int32,
        block0: Int32,
        num_heads: Int32,
        num_blocks: Int32,
        num_batch_heads: Int32,
        stream,
    ):
        self.kernel(
            mX,
            mPerm,
            mCentre,
            mX8,
            mScale,
            seq_len,
            src_len,
            block0,
            num_heads,
            num_batch_heads,
        ).launch(
            grid=[num_blocks, num_batch_heads, 1],
            block=[_THREADS, 1, 1],
            smem=(_THREADS // 32) * 4,
            stream=stream,
        )


# ---------------------------------------------------------------------------
# Compilation cache (TVM-FFI entry points taking torch tensors directly)
# ---------------------------------------------------------------------------
def _fake1d(dtype, align=32):
    return cute.runtime.make_fake_compact_tensor(
        dtype, (cute.sym_int(),), assumed_align=align
    )


def _in_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16, True
    if dtype == torch.float16:
        return cutlass.Float16, False
    raise TypeError(f"VC fused preparation needs bf16/fp16 inputs, got {dtype}")


@functools.lru_cache(maxsize=None)
def _compiled(
    dtype: torch.dtype, demean: bool, repair: bool = False, qk_fp8: bool = False
):
    cdt, is_bf16 = _in_dtype(dtype)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    f32 = lambda: _fake1d(Float32, 64)  # noqa: E731
    i32 = lambda: _fake1d(Int32, 16)  # noqa: E731
    e4m3 = lambda: _fake1d(cutlass.Float8E4M3FN, 16)  # noqa: E731
    p1 = cute.compile(
        VcKvPass1(is_bf16),
        _fake1d(cdt),
        _fake1d(cdt),
        i32(),
        _fake1d(cdt),
        f32(),
        f32(),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        demean,
        repair,
        qk_fp8,
        stream,
        options=_COMPILE_OPTIONS,
    )
    pq = pk = None
    if qk_fp8:
        pq, pk = (
            cute.compile(
                VcQKPass(is_bf16, gather, centre),
                _fake1d(cdt),
                i32(),
                f32(),
                e4m3(),
                f32(),
                Int32(1),
                Int32(1),
                Int32(1),
                Int32(1),
                Int32(1),
                Int32(1),
                stream,
                options=_COMPILE_OPTIONS,
            )
            for gather, centre in ((False, False), (True, True))
        )
    p2 = cute.compile(
        VcKvPass2(is_bf16),
        _fake1d(cdt),
        i32(),
        f32(),
        f32(),
        e4m3(),
        _fake1d(cutlass.BFloat16),
        f32(),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        repair,
        stream,
        options=_COMPILE_OPTIONS,
    )
    return p1, p2, pq, pk


@torch.no_grad()
def vc_prepare(
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor | None,
    *,
    demean: bool = True,
    repair_tiles: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Run the two preparation kernels.

    ``k``, ``v``: contiguous ``[B, S, H, D]`` bf16/fp16; ``perm``:
    ``[B, H, S_k]`` int32/int64. ``demean=False`` keeps the tile means at zero
    (V-Smooth off, plain per-channel E4M3 V). Returns ``(k_perm, v8, v_scale,
    mean, mu, energy)`` in the ``VCAttentionOperands`` layouts; ``v_scale`` is
    ``[B, H, D]``. With ``repair_tiles`` the token order is the input order,
    ``k_perm`` and ``v8`` use the V repair layout with the repair tiles left
    unwritten, and ``energy`` holds each token's residual energy ``[B, H, S]``
    in place of the mean operand.
    """
    return _prepare(None, k, v, perm, demean=demean, repair_tiles=repair_tiles)[:6]


@torch.no_grad()
def vc_prepare_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor | None,
    *,
    demean: bool = True,
    repair_tiles: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Run the VC-Attention-QK8 preparation kernels.

    ``q``, ``k``, ``v``: contiguous ``[B, S, H, D]`` bf16/fp16 (one Q block scale
    per 128 tokens); ``perm``: ``[B, H, S_k]`` int32/int64. Returns ``(q8, k8,
    v8, q_scale, k_scale, v_scale, mean, mu, energy)`` in the
    ``VCAttentionOperands`` layouts, the Q/K scales in the flat scale layout
    (the K layout over the K/V rows, repair tiles included). With
    ``repair_tiles`` the K repair tiles are left unwritten for
    :func:`vc_prepare_repair_keys_fp8`.
    """
    if q.shape[-1] != _D or not q.is_contiguous():
        raise ValueError(f"VC fused preparation needs contiguous head_dim {_D} Q")
    k8, v8, v_scale, mean, mu, energy, q8, q_scale, k_scale = _prepare(
        q, k, v, perm, demean=demean, repair_tiles=repair_tiles
    )
    return q8, k8, v8, q_scale, k_scale, v_scale, mean, mu, energy


def _repair_row_map(
    b: int, h: int, s_k: int, repair_tiles: int, device: torch.device
) -> torch.Tensor:
    """Output row -> source token of a V repair K layout: the whole tiles, the
    repair tiles (``-1``, filled by :func:`vc_prepare_repair_keys_fp8`) and the
    shifted partial tail; ``-1`` marks zero rows."""
    rows = vc_repair_kv_len(s_k, repair_tiles)
    whole = s_k // _TILE * _TILE
    shift = repair_tiles * _TILE
    row_map = torch.full((b, h, rows), -1, dtype=torch.int32, device=device)
    row_map[:, :, :whole] = torch.arange(whole, dtype=torch.int32, device=device)
    row_map[:, :, whole + shift : s_k + shift] = torch.arange(
        whole, s_k, dtype=torch.int32, device=device
    )
    return row_map


@torch.no_grad()
def vc_prepare_repair_keys_fp8(
    k: torch.Tensor,
    k8: torch.Tensor,
    k_scale: torch.Tensor,
    index: torch.Tensor,
    repair_tiles: int,
) -> None:
    """Write the K repair tiles of a VC-Attention-QK8 V repair run: the selected
    tokens ``index`` (``[B, H, selected]``) re-centred, rotated and quantized
    with each repair tile's own scale into ``k8`` and ``k_scale`` from
    :func:`vc_prepare_fp8`."""
    b, s_k, h, d = k.shape
    rows = k8.shape[1]
    whole = s_k // _TILE * _TILE
    _, _, _, pk = _compiled(k.dtype, False, True, True)
    k_mean = k.mean(dim=1, dtype=torch.float32).contiguous()  # [B, H, D]
    row_map = torch.full((b, h, rows), -1, dtype=torch.int32, device=k.device)
    row_map[:, :, whole : whole + index.shape[2]] = index.to(torch.int32)
    pk(
        k.view(-1),
        row_map.view(-1),
        k_mean.view(-1),
        k8.view(-1),
        k_scale.view(-1),
        rows,
        s_k,
        whole // _TILE,
        h,
        repair_tiles,
        b * h,
    )


def _prepare(
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor | None,
    *,
    demean: bool,
    repair_tiles: int,
) -> tuple[torch.Tensor, ...]:
    """Run the passes; ``q`` selects the QK8 recipe. Returns ``(k_out, v8,
    v_scale, mean, mu, energy, q8, q_scale, k_scale)`` with the QK8 outputs
    ``None`` under QK16."""
    b, s_k, h, d = k.shape
    if d != _D:
        raise ValueError(f"VC fused preparation supports head_dim {_D} only")
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("k and v must be contiguous")
    dev = k.device
    t = (s_k + _TILE - 1) // _TILE
    bh = b * h
    repair = repair_tiles > 0
    rows = vc_repair_kv_len(s_k, repair_tiles) if repair else s_k
    tail_start = s_k // _TILE * _TILE if repair else s_k
    qk_fp8 = q is not None
    p1, p2, pq, pk = _compiled(k.dtype, bool(demean), repair, qk_fp8)
    perm32 = (
        torch.zeros((1,), dtype=torch.int32, device=dev)
        if repair
        else perm.to(torch.int32).contiguous()
    )
    dummy = torch.zeros((1,), dtype=torch.float32, device=dev)
    if qk_fp8:
        k_perm = torch.empty((1,), dtype=k.dtype, device=dev)
        # bf16/fp16 input, fp32 accumulation; no fp32 copy of K.
        k_mean = k.mean(dim=1, dtype=torch.float32).contiguous()  # [B, H, D]
        k8 = torch.empty((b, rows, h, d), dtype=torch.float8_e4m3fn, device=dev)
        # Flat scale layout (sage.flat_scale_numel), [H, ceil(B*S / 128) + B - 1].
        k_scale = torch.ones(
            (h, (b * rows + _TILE - 1) // _TILE + b - 1),
            dtype=torch.float32,
            device=dev,
        )
        k_rows = _repair_row_map(b, h, s_k, repair_tiles, dev) if repair else perm32
    else:
        k_perm = torch.empty((b, rows, h, d), dtype=k.dtype, device=dev)
        k_mean = dummy
        k8 = torch.empty((1,), dtype=torch.float8_e4m3fn, device=dev)
        k_scale = dummy
    v8 = torch.empty((b, rows, h, d), dtype=torch.float8_e4m3fn, device=dev)
    mean = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    vamax = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    mu = torch.zeros(
        (1,)
        if repair
        else (
            b,
            h,
            (t + VC_MEAN_GROUP_TILES - 1) // VC_MEAN_GROUP_TILES,
            *vc_mean_group_shape(d),
        ),
        dtype=torch.bfloat16,
        device=dev,
    )
    energy = torch.empty(
        (b, h, s_k) if repair else (1,), dtype=torch.float32, device=dev
    )
    p1(
        k.view(-1),
        v.view(-1),
        perm32.view(-1),
        k_perm.view(-1),
        mean.view(-1),
        vamax.view(-1),
        s_k,
        h,
        t,
        bh,
        rows,
        tail_start,
        repair_tiles * _TILE,
    )
    v_scale = (vamax.amax(dim=2) / E4M3_MAX).clamp_min(1e-12).contiguous()  # [B, H, D]
    p2(
        v.view(-1),
        perm32.view(-1),
        mean.view(-1),
        v_scale.view(-1),
        v8.view(-1),
        mu.view(-1),
        energy.view(-1),
        s_k,
        h,
        t,
        bh,
        rows,
        tail_start,
        repair_tiles * _TILE,
    )
    if not qk_fp8:
        return k_perm, v8, v_scale, mean, mu, energy, None, None, None
    s_q = q.shape[1]
    q8 = torch.empty((b, s_q, h, d), dtype=torch.float8_e4m3fn, device=dev)
    q_scale = torch.ones(
        (h, (b * s_q + _TILE - 1) // _TILE + b - 1), dtype=torch.float32, device=dev
    )
    pk(
        k.view(-1),
        k_rows.view(-1),
        k_mean.view(-1),
        k8.view(-1),
        k_scale.view(-1),
        rows,
        s_k,
        0,
        h,
        (rows + _TILE - 1) // _TILE,
        bh,
    )
    pq(
        q.view(-1),
        perm32.view(-1),
        dummy,
        q8.view(-1),
        q_scale.view(-1),
        s_q,
        s_q,
        0,
        h,
        (s_q + _TILE - 1) // _TILE,
        bh,
    )
    return k8, v8, v_scale, mean, mu, energy, q8, q_scale, k_scale
