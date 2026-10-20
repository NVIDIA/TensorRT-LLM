"""VC-Attention operand preparation kernels.

``_prepare`` turns bf16 or fp16 ``[B, S, H, D]`` K, V and under QK8 Q into the
VC-Attention operands with one CTA per batch, head and 128-token tile. Shapes
are compile-time constants of the kernels and the launches take tensors only.
An operand a recipe does not use is ``None``, and the kernels take their mode
from that.

* ``VcKvPass1`` writes the V tile means and the per-channel residual amax, plus
  the permuted K rows under QK16 or the K tile channel sums under QK8.
* ``VcKvPass2`` writes the V scale, the E4M3 V residuals and the tile-mean
  operand, the K centre under QK8, the residual energies under repair.
* ``VcQKPass`` writes the centred, Hadamard-rotated E4M3 Q and K with one scale
  per block under QK8.
* ``VcSelectPass`` and ``VcRepairRowsPass`` pick the tokens of largest residual
  energy and write their repair rows under repair.

Mean recipes take tokens in ``perm`` order. Repair recipes keep input order and
shift the partial last tile past the repair tiles. Each thread owns 8 columns of
8 rows and issues all its loads before the guarded math, since the kernels are
latency-bound and need the occupancy.
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
    VC_HEAD_DIM,
    VC_K_BLOCK_SIZE,
    VC_MEAN_GROUP_TILES,
    VC_MEAN_MMA_K,
    VC_MEAN_OPERANDS,
    vc_mean_group_shape,
    vc_mean_operand_shape,
    vc_repair_kv_len,
)
from .fmha_decode.fmha_decode_resources.helpers_common import _pack_float2_to_bf16
from .fmha_decode.fmha_decode_resources.helpers_softmax import _pack_float4_to_fp8_e4m3

_COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"
_THREADS = 256
_COLS = 8
_COL_GROUPS = VC_HEAD_DIM // _COLS
_ROW_GROUPS = _THREADS // _COL_GROUPS
_ROWS = VC_K_BLOCK_SIZE // _ROW_GROUPS
_WORDS = _COLS // 2
_MU_ELEMS = math.prod(vc_mean_operand_shape(VC_HEAD_DIM))
_HADAMARD_NORM = 1.0 / math.sqrt(VC_HEAD_DIM)
_SELECT_BINS = 256
assert _THREADS == _SELECT_BINS, "VcSelectPass owns one histogram bin per thread"


@cute.jit
def _abs_f32(x: Float32) -> Float32:
    return cute.arch.fmax(x, -x)


@cute.jit
def _hadamard128_row(vals, base: cutlass.Constexpr[int], cg: Int32):
    """Unnormalised 128-point Walsh-Hadamard transform of one row in place.

    This thread holds 8 columns and the 15 lanes of its column group hold the
    rest. The caller folds the normalisation into its scale. The butterflies run
    as packed f32x2 pairs because the pass is issue-bound.
    """
    for h in (1, 2, 4):
        lo = [i for i in range(_COLS) if (i & h) == 0]
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
        sign = Float32(1.0) - Float32(2.0) * Float32((cg >> sbit) & 1)
        for j in cutlass.range_constexpr(0, _COLS, 2):
            m0 = vals[base + j]
            m1 = vals[base + j + 1]
            o0 = cute.arch.shuffle_sync_bfly(m0, stride)
            o1 = cute.arch.shuffle_sync_bfly(m1, stride)
            vals[base + j], vals[base + j + 1] = cute.arch.fma_packed_f32x2(
                (m0, m1), (sign, sign), (o0, o1)
            )


@cute.jit
def _block_max(v: Float32, red_smem, tidx: Int32) -> Float32:
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
def _load_raw(raw, base: cutlass.Constexpr[int], addr: Int64):
    """Load this thread's 16 bytes at ``addr`` into four words of ``raw``."""
    regs = cutlass.inttoptr(addr, mem_space=1, dtype=Int32).load(
        count=_WORDS, alignment=16
    )
    for q in cutlass.range_constexpr(_WORDS):
        raw[base + q] = regs[q]


@cute.jit
def _row(raw, base: cutlass.Constexpr[int], is_bf16: cutlass.Constexpr[bool]):
    """The 8 bf16 or fp16 values held in four words of ``raw`` as Float32."""
    regs = (raw.data_ptr() + base).load(count=_WORDS, alignment=16)
    if cutlass.const_expr(is_bf16):
        return regs.bitcast(cutlass.BFloat16).to(Float32)
    return regs.bitcast(cutlass.Float16).to(Float32)


@cute.jit
def _load_row_f32(addr: Int64):
    return cutlass.inttoptr(addr, mem_space=1, dtype=Float32).load(
        count=_COLS, alignment=32
    )


@cute.jit
def _pack_row_fp8(vals, base: cutlass.Constexpr[int], scale_inv: Float32):
    """This thread's 8 values from ``base`` times ``scale_inv`` as two E4M3 words."""
    packed = cutlass.Array(Int32, _COLS // 4, space=cutlass.AddressSpace.rmem)
    for j in cutlass.range_constexpr(_COLS // 4):
        packed[j] = _pack_float4_to_fp8_e4m3(
            vals[base + 4 * j] * scale_inv,
            vals[base + 4 * j + 1] * scale_inv,
            vals[base + 4 * j + 2] * scale_inv,
            vals[base + 4 * j + 3] * scale_inv,
        )
    return packed.data_ptr().load(count=_COLS // 4, alignment=8)


@cute.jit
def _coords(num_heads: cutlass.Constexpr[int]):
    """This CTA's tile, batch and head, this thread's row group and first column."""
    tidx, _, _ = cute.arch.thread_idx()
    t, bh, _ = cute.arch.block_idx()
    return (
        t,
        bh,
        bh // num_heads,
        bh % num_heads,
        tidx // _COL_GROUPS,
        (tidx % _COL_GROUPS) * _COLS,
    )


@cute.jit
def _row_base(
    base: Int64,
    b: Int32,
    rows: cutlass.Constexpr[int],
    h: Int32,
    col0: Int32,
    elem_bytes: cutlass.Constexpr[int],
    row_bytes: cutlass.Constexpr[int],
) -> Int64:
    """Address of this thread's columns of row 0 of batch ``b`` in a ``[B, rows, H, D]`` tensor."""
    return (
        base
        + Int64(b) * rows * row_bytes
        + (Int64(h) * VC_HEAD_DIM + col0) * elem_bytes
    )


@cute.jit
def _store_tile_cols(
    base: Int64,
    bh: Int32,
    num_tiles: cutlass.Constexpr[int],
    t: Int32,
    col0: Int32,
    vals,
):
    """Store this thread's 8 columns of tile ``t`` of a ``[B, H, T, D]`` fp32 tensor."""
    cutlass.inttoptr(
        base + ((Int64(bh) * num_tiles + t) * VC_HEAD_DIM + col0) * 4,
        mem_space=1,
        dtype=Float32,
    ).store(vals.data_ptr().load(count=_COLS, alignment=32), alignment=32)


@cute.jit
def _cta_col_reduce(
    col_sums, vals, rg: Int32, col0: Int32, is_max: cutlass.Constexpr[bool]
):
    """Reduce this thread's 8 column values over the row groups through shared memory."""
    cute.arch.sync_threads()
    for j in cutlass.range_constexpr(_COLS):
        col_sums[rg * VC_K_BLOCK_SIZE + col0 + j] = vals[j]
    cute.arch.sync_threads()
    for j in cutlass.range_constexpr(_COLS):
        acc = col_sums[col0 + j]
        for g in cutlass.range_constexpr(1, _ROW_GROUPS):
            if cutlass.const_expr(is_max):
                acc = cute.arch.fmax(acc, col_sums[g * VC_K_BLOCK_SIZE + col0 + j])
            else:
                acc = acc + col_sums[g * VC_K_BLOCK_SIZE + col0 + j]
        vals[j] = acc


@cute.jit
def _v_scale(vmax: Float32) -> Float32:
    return cute.arch.fmax(vmax * Float32(1.0 / E4M3_MAX), Float32(1e-12))


@cute.jit
def _stats_addr(
    mStats: cute.Tensor, bh: Int32, slot: cutlass.Constexpr[int], col0: Int32
) -> Int64:
    """Slot 0 holds the K channel sums and slot 1 the V residual amax."""
    return mStats.iterator.toint() + ((Int64(bh) * 2 + slot) * VC_HEAD_DIM + col0) * 4


@cute.jit
def _scale_slot(
    bh: Int32,
    block: Int32,
    num_heads: cutlass.Constexpr[int],
    batch: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
) -> Int64:
    """Element of block ``block`` of sequence ``bh`` in the ``[H, B, blocks]`` scale table."""
    blocks = (rows + VC_K_BLOCK_SIZE - 1) // VC_K_BLOCK_SIZE
    return Int64(bh % num_heads) * (batch * blocks) + Int64(
        (bh // num_heads) * blocks + block
    )


class _Shape:
    """Compile-time geometry shared by the kernels of one preparation."""

    def __init__(
        self,
        is_bf16: bool,
        batch: int,
        seq_len_kv: int,
        num_heads: int,
        repair_tiles: int,
    ):
        self.is_bf16 = is_bf16
        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.num_heads = num_heads
        self.num_tiles = (seq_len_kv + VC_K_BLOCK_SIZE - 1) // VC_K_BLOCK_SIZE
        self.repair_tiles = repair_tiles
        self.repair = repair_tiles > 0
        self.rows = (
            vc_repair_kv_len(seq_len_kv, repair_tiles) if self.repair else seq_len_kv
        )
        self.tail = (
            seq_len_kv // VC_K_BLOCK_SIZE * VC_K_BLOCK_SIZE
            if self.repair
            else seq_len_kv
        )
        self.shift = repair_tiles * VC_K_BLOCK_SIZE
        self.row_bytes = num_heads * VC_HEAD_DIM * 2
        self.row_bytes8 = num_heads * VC_HEAD_DIM


class VcKvPass1(_Shape):
    def __init__(
        self, is_bf16, batch, seq_len_kv, num_heads, repair_tiles, demean: bool
    ):
        super().__init__(is_bf16, batch, seq_len_kv, num_heads, repair_tiles)
        self.demean = demean

    @cute.kernel
    def kernel(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPerm: cute.Tensor | None,
        mKPerm: cute.Tensor | None,
        mMean: cute.Tensor,
        mKSum: cute.Tensor | None,
        mStats: cute.Tensor,
    ):
        t, bh, b, h, rg, col0 = _coords(self.num_heads)
        smem = cutlass.utils.SmemAllocator()
        col_sums = smem.allocate_array(Float32, _ROW_GROUPS * VC_K_BLOCK_SIZE)
        k_base = _row_base(
            mK.iterator.toint(), b, self.seq_len_kv, h, col0, 2, self.row_bytes
        )
        v_base = _row_base(
            mV.iterator.toint(), b, self.seq_len_kv, h, col0, 2, self.row_bytes
        )
        if cutlass.const_expr(mKPerm is not None):
            kp_base = _row_base(
                mKPerm.iterator.toint(), b, self.rows, h, col0, 2, self.row_bytes
            )

        # Rows past the sequence read token 0 and skip the math, so every load
        # of a thread is in flight together.
        toks = cutlass.Array(Int32, _ROWS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            pos = t * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
            tok = Int32(0)
            if pos < self.seq_len_kv:
                tok = pos
                if cutlass.const_expr(mPerm is not None):
                    tok = cutlass.inttoptr(
                        mPerm.iterator.toint()
                        + (Int64(bh) * self.seq_len_kv + pos) * 4,
                        mem_space=1,
                        dtype=Int32,
                    ).load()
            toks[i] = tok
        raw = cutlass.Array(Int32, _ROWS * _WORDS, space=cutlass.AddressSpace.rmem)

        if cutlass.const_expr(mKPerm is not None):
            for i in cutlass.range_constexpr(_ROWS):
                _load_raw(raw, i * _WORDS, k_base + Int64(toks[i]) * self.row_bytes)
            for i in cutlass.range_constexpr(_ROWS):
                pos = t * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
                if pos < self.seq_len_kv:
                    out_row = pos + self.shift * Int32(pos >= self.tail)
                    cutlass.inttoptr(
                        kp_base + Int64(out_row) * self.row_bytes,
                        mem_space=1,
                        dtype=Int32,
                    ).store(
                        (raw.data_ptr() + i * _WORDS).load(count=_WORDS, alignment=16),
                        alignment=16,
                    )
        else:
            # K channel sums of the tile in input order. Pass 2 sums the tiles
            # in a fixed order so the centre is deterministic.
            ksum = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
            for j in cutlass.range_constexpr(_COLS):
                ksum[j] = Float32(0.0)
            for i in cutlass.range_constexpr(_ROWS):
                pos = t * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
                kpos = Int32(0)
                if pos < self.seq_len_kv:
                    kpos = pos
                _load_raw(raw, i * _WORDS, k_base + Int64(kpos) * self.row_bytes)
            for i in cutlass.range_constexpr(_ROWS):
                pos = t * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
                if pos < self.seq_len_kv:
                    row = _row(raw, i * _WORDS, self.is_bf16)
                    for j in cutlass.range_constexpr(_COLS):
                        ksum[j] = ksum[j] + row[j]
            _cta_col_reduce(col_sums, ksum, rg, col0, False)
            if rg == 0:
                _store_tile_cols(
                    mKSum.iterator.toint(), bh, self.num_tiles, t, col0, ksum
                )

        # Per-column sum, max and min of the tile's V rows give the mean and the
        # residual amax exactly.
        psum = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        pmax = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        pmin = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS):
            psum[j] = Float32(0.0)
            pmax[j] = -Float32.inf
            pmin[j] = Float32.inf
        for i in cutlass.range_constexpr(_ROWS):
            _load_raw(raw, i * _WORDS, v_base + Int64(toks[i]) * self.row_bytes)
        for i in cutlass.range_constexpr(_ROWS):
            pos = t * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
            if pos < self.seq_len_kv:
                row = _row(raw, i * _WORDS, self.is_bf16)
                for j in cutlass.range_constexpr(_COLS):
                    psum[j] = psum[j] + row[j]
                    pmax[j] = cute.arch.fmax(pmax[j], row[j])
                    pmin[j] = cute.arch.fmin(pmin[j], row[j])
        valid_rows = self.seq_len_kv - t * VC_K_BLOCK_SIZE
        if valid_rows > VC_K_BLOCK_SIZE:
            valid_rows = Int32(VC_K_BLOCK_SIZE)
        inv_count = Float32(1.0) / Float32(valid_rows)
        _cta_col_reduce(col_sums, psum, rg, col0, False)
        mean = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS):
            mean[j] = psum[j] * inv_count
            if cutlass.const_expr(not self.demean):
                mean[j] = Float32(0.0)
        if rg == 0:
            _store_tile_cols(mMean.iterator.toint(), bh, self.num_tiles, t, col0, mean)
        cmax = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS):
            cmax[j] = Float32(0.0)
            if pmax[j] >= pmin[j]:
                cmax[j] = cute.arch.fmax(pmax[j] - mean[j], mean[j] - pmin[j])
        _cta_col_reduce(col_sums, cmax, rg, col0, True)
        if rg == 0:
            for j in cutlass.range_constexpr(_COLS):
                cute.arch.atomic_fmax(
                    cutlass.inttoptr(
                        _stats_addr(mStats, bh, 1, col0) + j * 4,
                        mem_space=1,
                        dtype=Float32,
                    ),
                    cmax[j],
                    sign_bit=False,
                )

    @cute.jit
    def __call__(self, mK, mV, mPerm, mKPerm, mMean, mKSum, mStats, stream):
        self.kernel(mK, mV, mPerm, mKPerm, mMean, mKSum, mStats).launch(
            grid=[self.num_tiles, self.batch * self.num_heads, 1],
            block=[_THREADS, 1, 1],
            smem=_ROW_GROUPS * VC_K_BLOCK_SIZE * 4,
            stream=stream,
        )


class VcKvPass2(_Shape):
    @cute.kernel
    def kernel(
        self,
        mV: cute.Tensor,
        mPerm: cute.Tensor | None,
        mMean: cute.Tensor,
        mKSum: cute.Tensor | None,
        mStats: cute.Tensor,
        mVScale: cute.Tensor,
        mV8: cute.Tensor,
        mMu: cute.Tensor | None,
        mEnergy: cute.Tensor | None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        t, bh, _ = cute.arch.block_idx()
        b = bh // self.num_heads
        h = bh % self.num_heads
        cg = tidx % _COL_GROUPS
        rg = tidx // _COL_GROUPS
        col0 = cg * _COLS
        v_base = (
            mV.iterator.toint()
            + Int64(b) * self.seq_len_kv * self.row_bytes
            + (Int64(h) * VC_HEAD_DIM + col0) * 2
        )
        mean_base = mMean.iterator.toint() + (Int64(bh) * self.num_tiles + t) * (
            VC_HEAD_DIM * 4
        )
        smem = cutlass.utils.SmemAllocator()
        energy_part = smem.allocate_array(Float32, _THREADS * _ROWS)

        vmax_base = _stats_addr(mStats, bh, 1, 0)
        vmax = _load_row_f32(vmax_base + col0 * 4)
        vs = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        vs_inv = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS):
            vs[j] = _v_scale(vmax[j])
            vs_inv[j] = Float32(1.0) / vs[j]
        if (t == 0) & (rg == 0):
            cutlass.inttoptr(
                mVScale.iterator.toint() + (Int64(bh) * VC_HEAD_DIM + col0) * 4,
                mem_space=1,
                dtype=Float32,
            ).store(vs.data_ptr().load(count=_COLS, alignment=32), alignment=32)
        if cutlass.const_expr(mKSum is not None):
            if (t == 0) & (tidx < VC_HEAD_DIM):
                ksum_base = (
                    mKSum.iterator.toint()
                    + (Int64(bh) * self.num_tiles * VC_HEAD_DIM + tidx) * 4
                )
                acc = Float32(0.0)
                for i in cutlass.range(self.num_tiles):
                    acc = (
                        acc
                        + cutlass.inttoptr(
                            ksum_base + Int64(i) * (VC_HEAD_DIM * 4),
                            mem_space=1,
                            dtype=Float32,
                        ).load()
                    )
                cutlass.inttoptr(
                    _stats_addr(mStats, bh, 0, 0) + tidx * 4, mem_space=1, dtype=Float32
                ).store(acc)

        mean = _load_row_f32(mean_base + col0 * 4)
        raw = cutlass.Array(Int32, _ROWS * _WORDS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            pos = t * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
            tok = Int32(0)
            if pos < self.seq_len_kv:
                tok = pos
                if cutlass.const_expr(mPerm is not None):
                    tok = cutlass.inttoptr(
                        mPerm.iterator.toint()
                        + (Int64(bh) * self.seq_len_kv + pos) * 4,
                        mem_space=1,
                        dtype=Int32,
                    ).load()
            _load_raw(raw, i * _WORDS, v_base + Int64(tok) * self.row_bytes)
        for i in cutlass.range_constexpr(_ROWS):
            pos = t * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
            energy = Float32(0.0)
            if pos < self.seq_len_kv:
                row = _row(raw, i * _WORDS, self.is_bf16)
                res = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
                for j in cutlass.range_constexpr(_COLS):
                    res[j] = (row[j] - mean[j]) * vs_inv[j]
                out_row = pos + self.shift * Int32(pos >= self.tail)
                out_addr = (
                    mV8.iterator.toint()
                    + (Int64(b) * self.rows + out_row) * self.row_bytes8
                    + Int64(h) * VC_HEAD_DIM
                    + col0
                )
                if cutlass.const_expr(mEnergy is not None):
                    # The squared E4M3 rounding error in value units ranks the
                    # tokens for repair.
                    words = _pack_row_fp8(res, 0, Float32(1.0))
                    codes = words.bitcast(cutlass.Float8E4M3FN).to(Float32)
                    for c in cutlass.range_constexpr(_COLS):
                        err = (res[c] - codes[c]) * vs[c]
                        energy = energy + err * err
                    cutlass.inttoptr(out_addr, mem_space=1, dtype=Int32).store(
                        words, alignment=8
                    )
                else:
                    cutlass.inttoptr(out_addr, mem_space=1, dtype=Int32).store(
                        _pack_row_fp8(res, 0, Float32(1.0)), alignment=8
                    )
            if cutlass.const_expr(mEnergy is not None):
                energy_part[(rg + _ROW_GROUPS * i) * _COL_GROUPS + cg] = energy
        if cutlass.const_expr(mEnergy is not None):
            cute.arch.sync_threads()
            for i in cutlass.range_constexpr(_ROWS):
                pos = t * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
                if (cg == 0) & (pos < self.seq_len_kv):
                    acc = Float32(0.0)
                    for c in cutlass.range_constexpr(_COL_GROUPS):
                        acc = (
                            acc + energy_part[(rg + _ROW_GROUPS * i) * _COL_GROUPS + c]
                        )
                    cutlass.inttoptr(
                        mEnergy.iterator.toint()
                        + (Int64(bh) * self.seq_len_kv + pos) * 4,
                        mem_space=1,
                        dtype=Float32,
                    ).store(acc)

        if cutlass.const_expr(mMu is not None):
            # Mean operand of this tile's group in core-matrix order. Tile 8o+i
            # of a group owns K slots 2i and 2i+1 of operand o. Each of the first
            # 128 threads stores its two rows' slots, and the last tile stores
            # zero slots for the tiles its group lacks. Means are stored divided
            # by the per-channel scale.
            k_per_core = VC_MEAN_MMA_K // 2
            idx0 = tidx * (2 * k_per_core)
            g = idx0 // (8 * VC_MEAN_MMA_K)
            c = (idx0 // (8 * k_per_core)) % 2
            d0 = g * 8 + (idx0 // k_per_core) % 8
            num_groups = (
                self.num_tiles + VC_MEAN_GROUP_TILES - 1
            ) // VC_MEAN_GROUP_TILES
            group0 = t - t % VC_MEAN_GROUP_TILES
            if tidx < 128:
                m0 = cutlass.inttoptr(
                    mean_base + Int64(d0) * 4, mem_space=1, dtype=Float32
                ).load()
                m1 = cutlass.inttoptr(
                    mean_base + Int64(d0 + 1) * 4, mem_space=1, dtype=Float32
                ).load()
                s0 = _v_scale(
                    cutlass.inttoptr(
                        vmax_base + Int64(d0) * 4, mem_space=1, dtype=Float32
                    ).load()
                )
                s1 = _v_scale(
                    cutlass.inttoptr(
                        vmax_base + Int64(d0 + 1) * 4, mem_space=1, dtype=Float32
                    ).load()
                )
                own0 = _pack_float2_to_bf16(m0 / s0, m0 / s0)
                own1 = _pack_float2_to_bf16(m1 / s1, m1 / s1)
                for i in cutlass.range_constexpr(VC_MEAN_GROUP_TILES):
                    operand = i // k_per_core
                    slot = 2 * (i % k_per_core)
                    if cutlass.const_expr(slot // k_per_core == 0):
                        mine = c == 0
                    else:
                        mine = c == 1
                    tile = group0 + i
                    if mine & ((tile == t) | ((t == self.num_tiles - 1) & (tile > t))):
                        val0 = Int32(0)
                        val1 = Int32(0)
                        if tile == t:
                            val0 = own0
                            val1 = own1
                        dst = (
                            mMu.iterator.toint()
                            + (
                                (
                                    (Int64(bh) * num_groups + t // VC_MEAN_GROUP_TILES)
                                    * VC_MEAN_OPERANDS
                                    + operand
                                )
                                * _MU_ELEMS
                                + idx0
                                + slot % k_per_core
                            )
                            * 2
                        )
                        cutlass.inttoptr(dst, mem_space=1, dtype=Int32).store(val0)
                        cutlass.inttoptr(
                            dst + k_per_core * 2, mem_space=1, dtype=Int32
                        ).store(val1)

    @cute.jit
    def __call__(
        self, mV, mPerm, mMean, mKSum, mStats, mVScale, mV8, mMu, mEnergy, stream
    ):
        self.kernel(mV, mPerm, mMean, mKSum, mStats, mVScale, mV8, mMu, mEnergy).launch(
            grid=[self.num_tiles, self.batch * self.num_heads, 1],
            block=[_THREADS, 1, 1],
            smem=_THREADS * _ROWS * 4,
            stream=stream,
        )


class VcQKPass(_Shape):
    """QK8 quantization of Q and K in one launch, one CTA per 128-token block.

    The first ``blocks_q`` blocks are Q, the rest are K through ``perm`` or the
    repair row map, with the repair tiles through the selection index. K is
    centred with the channel sums of the statistics. The rotated block stays
    in registers between the amax and the store.
    """

    def __init__(
        self,
        is_bf16,
        batch,
        seq_len_kv,
        num_heads,
        repair_tiles,
        seq_len_q: int,
        repair_tokens: int = 0,
    ):
        super().__init__(is_bf16, batch, seq_len_kv, num_heads, repair_tiles)
        self.seq_len_q = seq_len_q
        self.repair_tokens = repair_tokens
        self.blocks_q = (seq_len_q + VC_K_BLOCK_SIZE - 1) // VC_K_BLOCK_SIZE

    @cute.jit
    def _block(
        self,
        nb: Int32,
        co,
        mX: cute.Tensor,
        mPerm: cute.Tensor | None,
        mIndex: cute.Tensor | None,
        mStats: cute.Tensor | None,
        mX8: cute.Tensor,
        mScale: cute.Tensor,
        red,
        rows_out: cutlass.Constexpr[int],
        src: cutlass.Constexpr[int],
    ):
        """One block of Q, of K when ``mStats`` is given, or of the repair tiles
        when ``mIndex`` is given. ``co`` carries the CTA and thread coordinates."""
        _, bh, b, h, rg, col0 = co
        cg = col0 // _COLS
        tidx = rg * _COL_GROUPS + cg
        x_base = _row_base(mX.iterator.toint(), b, src, h, col0, 2, self.row_bytes)
        centre = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for j in cutlass.range_constexpr(_COLS):
            centre[j] = Float32(0.0)
        if cutlass.const_expr(mStats is not None):
            c = _load_row_f32(_stats_addr(mStats, bh, 0, col0))
            for j in cutlass.range_constexpr(_COLS):
                centre[j] = c[j] * Float32(1.0 / src)

        # Source token per row, with -1 for rows outside the sequence and for
        # zero rows. Loads read token 0 for those and the math is skipped.
        toks = cutlass.Array(Int32, _ROWS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            pos = nb * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
            tok = Int32(-1)
            if pos < rows_out:
                tok = pos
                if cutlass.const_expr(mIndex is not None):
                    tok = Int32(-1)
                    sel = pos - self.tail
                    if (sel >= 0) & (sel < self.repair_tokens):
                        tok = cutlass.inttoptr(
                            mIndex.iterator.toint()
                            + (Int64(bh) * self.repair_tokens + sel) * 4,
                            mem_space=1,
                            dtype=Int32,
                        ).load()
                elif cutlass.const_expr(mPerm is not None):
                    tok = cutlass.inttoptr(
                        mPerm.iterator.toint()
                        + (Int64(bh) * self.seq_len_kv + pos) * 4,
                        mem_space=1,
                        dtype=Int32,
                    ).load()
                elif cutlass.const_expr(mStats is not None and self.repair):
                    if pos >= self.tail:
                        tok = Int32(-1)
                        if (pos >= self.tail + self.shift) & (
                            pos < self.seq_len_kv + self.shift
                        ):
                            tok = pos - self.shift
            toks[i] = tok
        raw = cutlass.Array(Int32, _ROWS * _WORDS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            tok = toks[i]
            if tok < 0:
                tok = Int32(0)
            _load_raw(raw, i * _WORDS, x_base + Int64(tok) * self.row_bytes)
        vals = cutlass.Array(Float32, _ROWS * _COLS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            if toks[i] >= 0:
                row = _row(raw, i * _WORDS, self.is_bf16)
                for j in cutlass.range_constexpr(_COLS):
                    vals[i * _COLS + j] = row[j] - centre[j]
            else:
                # The rotation shuffles need all 16 lanes of a row.
                for j in cutlass.range_constexpr(_COLS):
                    vals[i * _COLS + j] = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS):
            _hadamard128_row(vals, i * _COLS, cg)
        amax = Float32(0.0)
        for i in cutlass.range_constexpr(_ROWS):
            pos = nb * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
            if pos < rows_out:
                for j in cutlass.range_constexpr(_COLS):
                    amax = cute.arch.fmax(amax, _abs_f32(vals[i * _COLS + j]))
        amax = _block_max(amax, red, tidx)
        scale = cute.arch.fmax(
            amax * Float32(_HADAMARD_NORM / E4M3_MAX), Float32(1e-12)
        )
        if tidx == 0:
            cutlass.inttoptr(
                mScale.iterator.toint()
                + _scale_slot(bh, nb, self.num_heads, self.batch, rows_out) * 4,
                mem_space=1,
                dtype=Float32,
            ).store(scale)
        scale_inv = Float32(_HADAMARD_NORM) / scale
        x8_base = _row_base(
            mX8.iterator.toint(), b, rows_out, h, col0, 1, self.row_bytes8
        )
        for i in cutlass.range_constexpr(_ROWS):
            pos = nb * VC_K_BLOCK_SIZE + rg + _ROW_GROUPS * i
            if pos < rows_out:
                cutlass.inttoptr(
                    x8_base + Int64(pos) * self.row_bytes8, mem_space=1, dtype=Int32
                ).store(_pack_row_fp8(vals, i * _COLS, scale_inv), alignment=8)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPerm: cute.Tensor | None,
        mIndex: cute.Tensor | None,
        mStats: cute.Tensor,
        mQ8: cute.Tensor,
        mK8: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
    ):
        co = _coords(self.num_heads)
        nb = co[0]
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_array(Float32, _THREADS // 32)
        kb = nb - self.blocks_q
        if nb < self.blocks_q:
            self._block(
                nb,
                co,
                mQ,
                None,
                None,
                None,
                mQ8,
                mQScale,
                red,
                self.seq_len_q,
                self.seq_len_q,
            )
        elif cutlass.const_expr(mIndex is not None) and (
            (kb >= self.tail // VC_K_BLOCK_SIZE)
            & (kb < self.tail // VC_K_BLOCK_SIZE + self.repair_tiles)
        ):
            self._block(
                kb,
                co,
                mK,
                None,
                mIndex,
                mStats,
                mK8,
                mKScale,
                red,
                self.rows,
                self.seq_len_kv,
            )
        else:
            self._block(
                kb,
                co,
                mK,
                mPerm,
                None,
                mStats,
                mK8,
                mKScale,
                red,
                self.rows,
                self.seq_len_kv,
            )

    @cute.jit
    def __call__(
        self, mQ, mK, mPerm, mIndex, mStats, mQ8, mK8, mQScale, mKScale, stream
    ):
        self.kernel(mQ, mK, mPerm, mIndex, mStats, mQ8, mK8, mQScale, mKScale).launch(
            grid=[
                self.blocks_q + (self.rows + VC_K_BLOCK_SIZE - 1) // VC_K_BLOCK_SIZE,
                self.batch * self.num_heads,
                1,
            ],
            block=[_THREADS, 1, 1],
            smem=(_THREADS // 32) * 4,
            stream=stream,
        )


class VcSelectPass(_Shape):
    """The ``repair_tokens`` tokens of largest residual energy per sequence.

    A radix select over the four bytes of the non-negative fp32 bits with one
    histogram per warp, then the indices above the threshold and the first
    ones at it, in arbitrary order.
    """

    def __init__(
        self, is_bf16, batch, seq_len_kv, num_heads, repair_tiles, repair_tokens: int
    ):
        super().__init__(is_bf16, batch, seq_len_kv, num_heads, repair_tiles)
        self.repair_tokens = repair_tokens

    @cute.kernel
    def kernel(self, mEnergy: cute.Tensor, mIndex: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bh, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        smem = cutlass.utils.SmemAllocator()
        hist = smem.allocate_array(Int32, (_THREADS // 32) * _SELECT_BINS)
        warp_tot = smem.allocate_array(Int32, _THREADS // 32)
        # The threshold prefix, the count still wanted at it, the count stored
        # above it and the count seen at it.
        state = smem.allocate_array(Int32, 4)
        e_base = mEnergy.iterator.toint() + Int64(bh) * self.seq_len_kv * 4
        out_base = mIndex.iterator.toint() + Int64(bh) * self.repair_tokens * 4
        # One CTA streams the whole row, so four loads per thread and
        # iteration set the memory parallelism.
        chunks = (self.seq_len_kv + 4 * _THREADS - 1) // (4 * _THREADS)
        if tidx == 0:
            state[0] = Int32(0)
            state[1] = Int32(self.repair_tokens)
            state[2] = Int32(0)
            state[3] = Int32(0)
        cute.arch.sync_threads()
        for p in cutlass.range_constexpr(4):
            shift = 24 - 8 * p
            for w in cutlass.range_constexpr(_THREADS // 32):
                hist[w * _SELECT_BINS + tidx] = Int32(0)
            prefix = state[0]
            rem = state[1]
            cute.arch.sync_threads()
            for it in cutlass.range(chunks):
                vals = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
                for u in cutlass.range_constexpr(4):
                    i = tidx + (it * 4 + u) * _THREADS
                    vals[u] = Int32(-1)
                    if i < self.seq_len_kv:
                        vals[u] = cutlass.inttoptr(
                            e_base + Int64(i) * 4, mem_space=1, dtype=Int32
                        ).load()
                for u in cutlass.range_constexpr(4):
                    bits = vals[u]
                    hit = bits >= 0
                    if cutlass.const_expr(p > 0):
                        hit = hit & ((bits >> (shift + 8)) == prefix)
                    if hit:
                        cute.arch.atomic_add(
                            hist
                            + (
                                warp * _SELECT_BINS
                                + ((bits >> shift) & (_SELECT_BINS - 1))
                            ),
                            Int32(1),
                        )
            cute.arch.sync_threads()
            # Thread tidx owns bin 255 - tidx and the scan runs in descending bin order.
            mybin = _SELECT_BINS - 1 - tidx
            cnt = Int32(0)
            for w in cutlass.range_constexpr(_THREADS // 32):
                cnt = cnt + hist[w * _SELECT_BINS + mybin]
            incl = cnt
            for sbit in cutlass.range_constexpr(5):
                other = cute.arch.shuffle_sync_up(incl, 1 << sbit)
                if lane >= (1 << sbit):
                    incl = incl + other
            if lane == 31:
                warp_tot[warp] = incl
            cute.arch.sync_threads()
            excl = incl - cnt
            for w in cutlass.range_constexpr(_THREADS // 32):
                if w < warp:
                    excl = excl + warp_tot[w]
            if (excl < rem) & (rem <= excl + cnt):
                state[0] = (prefix << 8) | mybin
                state[1] = rem - excl
            cute.arch.sync_threads()
        threshold = state[0]
        rem = state[1]
        n_above = self.repair_tokens - rem
        for it in cutlass.range(chunks):
            vals = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            for u in cutlass.range_constexpr(4):
                i = tidx + (it * 4 + u) * _THREADS
                vals[u] = Int32(-1)
                if i < self.seq_len_kv:
                    vals[u] = cutlass.inttoptr(
                        e_base + Int64(i) * 4, mem_space=1, dtype=Int32
                    ).load()
            for u in cutlass.range_constexpr(4):
                i = tidx + (it * 4 + u) * _THREADS
                bits = vals[u]
                if bits > threshold:
                    slot = cute.arch.atomic_add(state + 2, Int32(1))
                    cutlass.inttoptr(
                        out_base + Int64(slot) * 4, mem_space=1, dtype=Int32
                    ).store(i)
                elif bits == threshold:
                    slot = cute.arch.atomic_add(state + 3, Int32(1))
                    if slot < rem:
                        cutlass.inttoptr(
                            out_base + Int64(n_above + slot) * 4,
                            mem_space=1,
                            dtype=Int32,
                        ).store(i)

    @cute.jit
    def __call__(self, mEnergy, mIndex, stream):
        self.kernel(mEnergy, mIndex).launch(
            grid=[self.batch * self.num_heads, 1, 1],
            block=[_THREADS, 1, 1],
            smem=((_THREADS // 32) * _SELECT_BINS + _THREADS // 32 + 4) * 4,
            stream=stream,
        )


class VcRepairRowsPass(_Shape):
    """The repair tiles and the padding after the shifted tail.

    One CTA per repair tile writes the selected tokens' E4M3 value residuals
    and, when ``mKOut`` is given, their keys, with zero rows for unused slots.
    One more CTA zeroes the padding rows.
    """

    def __init__(
        self, is_bf16, batch, seq_len_kv, num_heads, repair_tiles, repair_tokens: int
    ):
        super().__init__(is_bf16, batch, seq_len_kv, num_heads, repair_tiles)
        self.repair_tokens = repair_tokens

    @cute.kernel
    def kernel(
        self,
        mK: cute.Tensor | None,
        mV: cute.Tensor,
        mIndex: cute.Tensor,
        mStats: cute.Tensor,
        mKOut: cute.Tensor | None,
        mV8: cute.Tensor,
    ):
        j, bh, b, h, rg, col0 = _coords(self.num_heads)
        v_base = _row_base(
            mV.iterator.toint(), b, self.seq_len_kv, h, col0, 2, self.row_bytes
        )
        v8_base = _row_base(
            mV8.iterator.toint(), b, self.rows, h, col0, 1, self.row_bytes8
        )
        if cutlass.const_expr(mKOut is not None):
            k_base = _row_base(
                mK.iterator.toint(), b, self.seq_len_kv, h, col0, 2, self.row_bytes
            )
            kout_base = _row_base(
                mKOut.iterator.toint(), b, self.rows, h, col0, 2, self.row_bytes
            )
        vmax = _load_row_f32(_stats_addr(mStats, bh, 1, col0))
        vs_inv = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
        for c in cutlass.range_constexpr(_COLS):
            vs_inv[c] = Float32(1.0) / _v_scale(vmax[c])
        zeros = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
        for c in cutlass.range_constexpr(4):
            zeros[c] = Int32(0)
        raw = cutlass.Array(Int32, _WORDS, space=cutlass.AddressSpace.rmem)
        for i in cutlass.range_constexpr(_ROWS):
            rr = rg + _ROW_GROUPS * i
            tok = Int32(-1)
            out_row = self.seq_len_kv + self.shift + rr
            if j < self.repair_tiles:
                out_row = self.tail + j * VC_K_BLOCK_SIZE + rr
                sel = j * VC_K_BLOCK_SIZE + rr
                if sel < self.repair_tokens:
                    tok = cutlass.inttoptr(
                        mIndex.iterator.toint()
                        + (Int64(bh) * self.repair_tokens + sel) * 4,
                        mem_space=1,
                        dtype=Int32,
                    ).load()
            if out_row < self.rows:
                v_dst = v8_base + Int64(out_row) * self.row_bytes8
                if tok >= 0:
                    src_off = Int64(tok) * self.row_bytes
                    _load_raw(raw, 0, v_base + src_off)
                    row = _row(raw, 0, self.is_bf16)
                    # The residual of the E4M3 code pass 2 stored for this token.
                    res = cutlass.Array(Float32, _COLS, space=cutlass.AddressSpace.rmem)
                    for c in cutlass.range_constexpr(_COLS):
                        res[c] = row[c] * vs_inv[c]
                    codes = (
                        _pack_row_fp8(res, 0, Float32(1.0))
                        .bitcast(cutlass.Float8E4M3FN)
                        .to(Float32)
                    )
                    for c in cutlass.range_constexpr(_COLS):
                        res[c] = res[c] - codes[c]
                    cutlass.inttoptr(v_dst, mem_space=1, dtype=Int32).store(
                        _pack_row_fp8(res, 0, Float32(1.0)), alignment=8
                    )
                    if cutlass.const_expr(mKOut is not None):
                        _load_raw(raw, 0, k_base + src_off)
                        cutlass.inttoptr(
                            kout_base + Int64(out_row) * self.row_bytes,
                            mem_space=1,
                            dtype=Int32,
                        ).store(
                            raw.data_ptr().load(count=_WORDS, alignment=16),
                            alignment=16,
                        )
                else:
                    cutlass.inttoptr(v_dst, mem_space=1, dtype=Int32).store(
                        zeros.data_ptr().load(count=2, alignment=8), alignment=8
                    )
                    if cutlass.const_expr(mKOut is not None):
                        cutlass.inttoptr(
                            kout_base + Int64(out_row) * self.row_bytes,
                            mem_space=1,
                            dtype=Int32,
                        ).store(
                            zeros.data_ptr().load(count=4, alignment=16), alignment=16
                        )

    @cute.jit
    def __call__(self, mK, mV, mIndex, mStats, mKOut, mV8, stream):
        self.kernel(mK, mV, mIndex, mStats, mKOut, mV8).launch(
            grid=[self.repair_tiles + 1, self.batch * self.num_heads, 1],
            block=[_THREADS, 1, 1],
            smem=0,
            stream=stream,
        )


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
    dtype: torch.dtype,
    demean: bool,
    batch: int,
    seq_len_kv: int,
    num_heads: int,
    repair_tiles: int,
    seq_len_q: int,
    repair_tokens: int,
):
    """The kernels of one geometry. ``seq_len_q`` is 0 under QK16 and ``repair_tokens`` is 0
    without repair. Absent operands are ``None`` at compile time as at run time."""
    cdt, is_bf16 = _in_dtype(dtype)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    f32 = lambda: _fake1d(Float32, 64)  # noqa: E731
    i32 = lambda: _fake1d(Int32, 16)  # noqa: E731
    e4m3 = lambda: _fake1d(cutlass.Float8E4M3FN, 16)  # noqa: E731
    x16 = lambda: _fake1d(cdt)  # noqa: E731
    qk8, repair = seq_len_q > 0, repair_tiles > 0
    shape = (is_bf16, batch, seq_len_kv, num_heads, repair_tiles)
    perm = None if repair else i32()
    kperm = None if qk8 else x16()
    ksum = f32() if qk8 else None
    k = {
        "p1": cute.compile(
            VcKvPass1(*shape, demean),
            x16(),
            x16(),
            perm,
            kperm,
            f32(),
            ksum,
            f32(),
            stream,
            options=_COMPILE_OPTIONS,
        ),
        "p2": cute.compile(
            VcKvPass2(*shape),
            x16(),
            perm,
            f32(),
            ksum,
            f32(),
            f32(),
            e4m3(),
            None if repair else _fake1d(cutlass.BFloat16),
            f32() if repair else None,
            stream,
            options=_COMPILE_OPTIONS,
        ),
    }
    if repair:
        k["select"] = cute.compile(
            VcSelectPass(*shape, repair_tokens),
            f32(),
            i32(),
            stream,
            options=_COMPILE_OPTIONS,
        )
        k["rows"] = cute.compile(
            VcRepairRowsPass(*shape, repair_tokens),
            None if qk8 else x16(),
            x16(),
            i32(),
            f32(),
            None if qk8 else x16(),
            e4m3(),
            stream,
            options=_COMPILE_OPTIONS,
        )
    if qk8:
        k["qk"] = cute.compile(
            VcQKPass(*shape, seq_len_q, repair_tokens),
            x16(),
            x16(),
            perm,
            i32() if repair else None,
            f32(),
            e4m3(),
            e4m3(),
            f32(),
            f32(),
            stream,
            options=_COMPILE_OPTIONS,
        )
    return k


@torch.no_grad()
def vc_prepare(
    k: torch.Tensor, v: torch.Tensor, perm: torch.Tensor, *, demean: bool = True
):
    """VC-Attention-QK16 operands ``(k_perm, v8, v_scale, mean, mu)`` of
    contiguous ``[B, S, H, D]`` K and V in ``perm`` order."""
    return _prepare(None, k, v, perm, demean=demean, repair_tiles=0, repair_tokens=0)[
        :5
    ]


@torch.no_grad()
def vc_prepare_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    perm: torch.Tensor,
    *,
    demean: bool = True,
):
    """VC-Attention-QK8 operands ``(q8, k8, v8, q_scale, k_scale, v_scale, mean,
    mu)`` of contiguous ``[B, S, H, D]`` Q, K and V, K and V in ``perm`` order."""
    if q.shape[-1] != VC_HEAD_DIM or not q.is_contiguous():
        raise ValueError(
            f"VC fused preparation needs contiguous head_dim {VC_HEAD_DIM} Q"
        )
    k8, v8, v_scale, mean, mu, q8, q_scale, k_scale = _prepare(
        q, k, v, perm, demean=demean, repair_tiles=0, repair_tokens=0
    )
    return q8, k8, v8, q_scale, k_scale, v_scale, mean, mu


@torch.no_grad()
def vc_prepare_repair(
    q: torch.Tensor | None,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    repair_tiles: int,
    repair_tokens: int,
):
    """V repair operands ``(q8, k_all, v_all, q_scale, k_scale, v_scale)`` in
    input order plus ``repair_tiles`` tiles of the ``repair_tokens`` largest-residual
    tokens. The QK8 entries are ``None`` under QK16."""
    if q is not None and (q.shape[-1] != VC_HEAD_DIM or not q.is_contiguous()):
        raise ValueError(
            f"VC fused preparation needs contiguous head_dim {VC_HEAD_DIM} Q"
        )
    if not 0 < repair_tokens <= min(repair_tiles * VC_K_BLOCK_SIZE, k.shape[1]):
        raise ValueError("repair_tokens must fit the repair tiles and the sequence")
    k_all, v_all, v_scale, _, _, q8, q_scale, k_scale = _prepare(
        q,
        k,
        v,
        None,
        demean=False,
        repair_tiles=repair_tiles,
        repair_tokens=repair_tokens,
    )
    return q8, k_all, v_all, q_scale, k_scale, v_scale


def _prepare(q, k, v, perm, *, demean: bool, repair_tiles: int, repair_tokens: int):
    """Run the passes and return ``(k_out, v8, v_scale, mean, mu, q8, q_scale,
    k_scale)``. ``q`` selects QK8 and the QK8 outputs are ``None`` without it."""
    b, s_k, h, d = k.shape
    if d != VC_HEAD_DIM:
        raise ValueError(f"VC fused preparation supports head_dim {VC_HEAD_DIM} only")
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("k and v must be contiguous")
    dev = k.device
    t = (s_k + VC_K_BLOCK_SIZE - 1) // VC_K_BLOCK_SIZE
    repair = repair_tiles > 0
    rows = vc_repair_kv_len(s_k, repair_tiles) if repair else s_k
    s_q = 0 if q is None else q.shape[1]
    ker = _compiled(k.dtype, bool(demean), b, s_k, h, repair_tiles, s_q, repair_tokens)
    if not repair and (
        perm is None or tuple(perm.shape) != (b, h, s_k) or perm.device != dev
    ):
        raise ValueError(f"perm must be a [B, H, S_k] index tensor on {dev}")
    perm32 = None if repair else perm.to(torch.int32).contiguous()
    stats = torch.zeros((b, h, 2, d), dtype=torch.float32, device=dev)
    mean = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    v8 = torch.empty((b, rows, h, d), dtype=torch.float8_e4m3fn, device=dev)
    v_scale = torch.empty((b, h, d), dtype=torch.float32, device=dev)
    k_perm = k_sum = mu = energy = index = None
    if q is None:
        k_perm = torch.empty((b, rows, h, d), dtype=k.dtype, device=dev)
    else:
        k_sum = torch.empty((b, h, t, d), dtype=torch.float32, device=dev)
    if repair:
        energy = torch.empty((b, h, s_k), dtype=torch.float32, device=dev)
        index = torch.empty((b, h, repair_tokens), dtype=torch.int32, device=dev)
    else:
        groups = (t + VC_MEAN_GROUP_TILES - 1) // VC_MEAN_GROUP_TILES
        mu = torch.empty(
            (b, h, groups, *vc_mean_group_shape(d)), dtype=torch.bfloat16, device=dev
        )
    flat = lambda x: None if x is None else x.view(-1)  # noqa: E731
    ker["p1"](
        flat(k),
        flat(v),
        flat(perm32),
        flat(k_perm),
        flat(mean),
        flat(k_sum),
        flat(stats),
    )
    ker["p2"](
        flat(v),
        flat(perm32),
        flat(mean),
        flat(k_sum),
        flat(stats),
        flat(v_scale),
        flat(v8),
        flat(mu),
        flat(energy),
    )
    if repair:
        ker["select"](flat(energy), flat(index))
        ker["rows"](
            flat(k) if q is None else None,
            flat(v),
            flat(index),
            flat(stats),
            flat(k_perm),
            flat(v8),
        )
    if q is None:
        return k_perm, v8, v_scale, mean, mu, None, None, None
    k8 = torch.empty((b, rows, h, d), dtype=torch.float8_e4m3fn, device=dev)
    q8 = torch.empty((b, s_q, h, d), dtype=torch.float8_e4m3fn, device=dev)
    k_scale = torch.empty(
        (h, b, (rows + VC_K_BLOCK_SIZE - 1) // VC_K_BLOCK_SIZE),
        dtype=torch.float32,
        device=dev,
    )
    q_scale = torch.empty(
        (h, b, (s_q + VC_K_BLOCK_SIZE - 1) // VC_K_BLOCK_SIZE),
        dtype=torch.float32,
        device=dev,
    )
    ker["qk"](
        flat(q),
        flat(k),
        flat(perm32),
        flat(index),
        flat(stats),
        flat(q8),
        flat(k8),
        flat(q_scale),
        flat(k_scale),
    )
    return k8, v8, v_scale, mean, mu, q8, q_scale, k_scale
