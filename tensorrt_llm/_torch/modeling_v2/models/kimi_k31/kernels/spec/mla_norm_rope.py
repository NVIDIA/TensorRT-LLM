# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA drafter latent RMSNorm and adjacent-pair RoPE for Blackwell (CuTe DSL).

Plain SIMT kernels, one warp per row. The ``kv_lora_rank`` (512) latent is RMS-normalized and
scaled by the norm weight; the ``rope_dim`` tail is rotated as (real, imag) pairs at
``freqs[position]`` (``rope_interleave``) with no norm and no weight.

The norm reproduces the torch reference bit for bit: each square is rounded on its own, the
512 squares are summed in the order torch's CUDA mean takes for a contiguous 512-wide FP32 row
(``_sum_squares_torch_order``), then ``* 2**-9``, ``+ eps``, ``rsqrt`` and the two products,
each rounded on its own. The rotation is the same expression as torch's complex product.

``K31DraftMlaCtxNormRoPEKernel`` normalizes the context latents of every drafter layer in one
launch; ``K31DraftMlaBlockRoPEKernel`` does one layer's block work in one launch: the query's rope
half and the block latents stored straight into the paged BF16 pool.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


@dsl_user_op
def _add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a + b`` rounded to nearest, never contracted."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """``a * b`` rounded to nearest, never contracted."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _rsqrt(a: Float32, *, loc=None, ip=None) -> Float32:
    """``rsqrt.approx.f32``: what ``torch.rsqrt`` of an FP32 tensor runs on the GPU."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


#: The latent width the torch-order norm is written for (``kv_lora_rank``).
_NORM_DIM = 512
#: Latent columns per lane: four 4-wide vectors at ``128 j + 4 lane + i``.
_NORM_VECS = 4
_NORM_VEC = 4


def _norm_column(lane, k: int):
    """Latent column of a lane's item ``k = 4 j + i``: ``128 j + 4 lane + i``."""
    return (k // _NORM_VEC) * (_NORM_VEC * cute.arch.WARP_SIZE) + lane * _NORM_VEC + k % _NORM_VEC


def _sum_squares_torch_order(values, num_rows) -> Float32:
    """The sum of the 512 squares of a row, in the order torch's CUDA mean takes, on every lane.

    torch reduces a contiguous ``[num_rows, 512]`` FP32 tensor over its last dim with 4-wide
    vector loads and ``bw`` threads per row (``bw`` = 32 from 16 rows up, 64 from 8, else 128):
    thread ``l`` accumulates vectors ``l, l + bw, ...`` into four running sums (one per vector
    element), folds them ``((s0 + s1) + s2) + s3``, the threads past 32 are folded in by halves
    through shared memory, and a warp tree (offsets 16 .. 1) finishes. Lane ``p`` here holds the
    vectors ``p, p + 32, p + 64, p + 96`` (``values[4 j + i]``, column ``128 j + 4 p + i``), so
    every ``bw`` is a fixed regrouping of its 16 squares; a butterfly leaves lane 0's tree sum
    on every lane.
    """
    sq = [_mul_rn(values[k], values[k]) for k in range(_NORM_VECS * _NORM_VEC)]

    def fold(acc):
        return _add_rn(_add_rn(_add_rn(acc[0], acc[1]), acc[2]), acc[3])

    def lane_sum(vectors):
        acc = []
        for i in range(_NORM_VEC):
            total = sq[vectors[0] * _NORM_VEC + i]
            for j in vectors[1:]:
                total = _add_rn(total, sq[j * _NORM_VEC + i])
            acc.append(total)
        return fold(acc)

    sum32 = lane_sum((0, 1, 2, 3))
    sum64 = _add_rn(lane_sum((0, 2)), lane_sum((1, 3)))
    single = [lane_sum((j,)) for j in range(_NORM_VECS)]
    sum128 = _add_rn(_add_rn(single[0], single[2]), _add_rn(single[1], single[3]))
    total = Float32(cutlass.select_(num_rows < 16, sum64, sum32))
    total = Float32(cutlass.select_(num_rows < 8, sum128, total))
    for offset in (16, 8, 4, 2, 1):
        total = _add_rn(total, cute.arch.shuffle_sync_bfly(total, offset=offset))
    return total


def _inverse_rms_torch_order(values, num_rows, eps: float) -> Float32:
    """``rsqrt(mean(x ** 2) + eps)`` of a 512-wide row, bitwise torch's FP32 result."""
    mean = _mul_rn(_sum_squares_torch_order(values, num_rows), Float32(1.0 / _NORM_DIM))
    return _rsqrt(_add_rn(mean, Float32(eps)))


class K31DraftMlaCtxNormRoPEKernel:
    """MLA context latents of every drafter layer in one launch: RMSNorm + RoPE.

    Row ``r`` of the work is layer ``r // N``, token ``r % N``: the ``kv_lora_rank`` latent of
    ``x[token, layer]`` RMS-normalized and scaled by ``weight[layer]``, then the ``rope_dim``
    tail rotated at ``freqs[positions[token]]`` (adjacent pairs, no norm, no weight), written to
    ``output[layer, token]`` (any layer / token strides, unit last stride).

    Every load of a row is issued before its first store, and the RoPE table is indexed by
    position in the kernel instead of through a gathered copy. The torch reference normalizes
    the latents as ``N * L`` rows, which picks its summation order.
    """

    num_warps = 4
    num_threads = num_warps * cute.arch.WARP_SIZE

    def __init__(self, num_layers: int, kv_lora_rank: int, rope_dim: int, eps: float):
        warp = cute.arch.WARP_SIZE
        if kv_lora_rank % warp != 0 or (rope_dim // 2) % warp != 0 or rope_dim % 2 != 0:
            raise ValueError(
                f"kv_lora_rank ({kv_lora_rank}) and rope_dim / 2 ({rope_dim // 2}) must be "
                f"multiples of {warp}"
            )
        if kv_lora_rank != _NORM_DIM:
            raise ValueError(f"the torch-order norm is written for kv_lora_rank {_NORM_DIM}")
        self.num_layers = num_layers
        self.norm_dim = kv_lora_rank
        self.head_dim = kv_lora_rank + rope_dim
        self.eps = eps
        self.norm_items = kv_lora_rank // warp
        self.pair_items = rope_dim // 2 // warp

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows = x.shape[0] * self.num_layers
        self.kernel(x, weight, positions, freqs, output).launch(
            grid=[cute.ceil_div(rows, self.num_warps), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % cute.arch.WARP_SIZE
        row = bidx * self.num_warps + tidx // cute.arch.WARP_SIZE
        num_tokens = x.shape[0]
        if row < num_tokens * self.num_layers:
            layer = row // num_tokens
            token = row - layer * num_tokens
            col0 = layer * self.head_dim
            values = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            scales = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            reals = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            imags = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            coss = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            sins = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            position = positions[token]
            for item in cutlass.range_constexpr(self.norm_items):
                dim = _norm_column(lane, item)
                values[item] = cutlass.Float32(x[token, col0 + dim])
                scales[item] = cutlass.Float32(weight[layer, dim])
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                reals[item] = cutlass.Float32(x[token, col0 + self.norm_dim + pair * 2])
                imags[item] = cutlass.Float32(x[token, col0 + self.norm_dim + pair * 2 + 1])
                coss[item] = cutlass.Float32(freqs[position, pair, 0])
                sins[item] = cutlass.Float32(freqs[position, pair, 1])

            # torch normalizes the [N, L, 512] latents as N * L rows.
            inverse_rms = _inverse_rms_torch_order(values, num_tokens * self.num_layers, self.eps)

            for item in cutlass.range_constexpr(self.norm_items):
                dim = _norm_column(lane, item)
                value = _mul_rn(_mul_rn(values[item], inverse_rms), scales[item])
                output[layer, token, dim] = value.to(output.element_type)
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                real = reals[item]
                imag = imags[item]
                cos = coss[item]
                sin = sins[item]
                output[layer, token, self.norm_dim + pair * 2] = (real * cos - imag * sin).to(
                    output.element_type
                )
                output[layer, token, self.norm_dim + pair * 2 + 1] = (imag * cos + real * sin).to(
                    output.element_type
                )


class K31DraftMlaCtxNormRoPEStoreKernel(K31DraftMlaCtxNormRoPEKernel):
    """:class:`K31DraftMlaCtxNormRoPEKernel` storing straight into the paged latent pool: row ``r`` of the
    work (layer ``r // N``, token ``r % N``) goes to ``pool[layer, rows[token]]`` (``page * page_size +
    offset``), and a token whose ``keep[token]`` is 0 stores its rounded values times zero (the masked
    multiply ``latent.mul_(keep)`` of the torch path, signs and NaNs included)."""

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        pool: cute.Tensor,
        rows: cute.Tensor,
        keep: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num = x.shape[0] * self.num_layers
        self.kernel(x, weight, positions, freqs, pool, rows, keep).launch(
            grid=[cute.ceil_div(num, self.num_warps), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.jit
    def _put(self, pool: cute.Tensor, layer, row, dim, value: Float32, keep):
        out = value.to(pool.element_type)
        if keep == 0:
            out = (Float32(out) * Float32(0.0)).to(pool.element_type)
        pool[layer, row, dim] = out

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        pool: cute.Tensor,
        rows: cute.Tensor,
        keep: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % cute.arch.WARP_SIZE
        row = bidx * self.num_warps + tidx // cute.arch.WARP_SIZE
        num_tokens = x.shape[0]
        if row < num_tokens * self.num_layers:
            layer = row // num_tokens
            token = row - layer * num_tokens
            col0 = layer * self.head_dim
            values = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            scales = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            reals = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            imags = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            coss = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            sins = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
            position = positions[token]
            dest = rows[token]
            kept = cutlass.Int32(keep[token])
            for item in cutlass.range_constexpr(self.norm_items):
                dim = _norm_column(lane, item)
                values[item] = cutlass.Float32(x[token, col0 + dim])
                scales[item] = cutlass.Float32(weight[layer, dim])
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                reals[item] = cutlass.Float32(x[token, col0 + self.norm_dim + pair * 2])
                imags[item] = cutlass.Float32(x[token, col0 + self.norm_dim + pair * 2 + 1])
                coss[item] = cutlass.Float32(freqs[position, pair, 0])
                sins[item] = cutlass.Float32(freqs[position, pair, 1])

            # torch normalizes the [N, L, 512] latents as N * L rows.
            inverse_rms = _inverse_rms_torch_order(values, num_tokens * self.num_layers, self.eps)

            for item in cutlass.range_constexpr(self.norm_items):
                dim = _norm_column(lane, item)
                value = _mul_rn(_mul_rn(values[item], inverse_rms), scales[item])
                self._put(pool, layer, dest, dim, value, kept)
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                real = reals[item]
                imag = imags[item]
                cos = coss[item]
                sin = sins[item]
                self._put(
                    pool, layer, dest, self.norm_dim + pair * 2, real * cos - imag * sin, kept
                )
                self._put(
                    pool, layer, dest, self.norm_dim + pair * 2 + 1, imag * cos + real * sin, kept
                )


class K31DraftMlaBlockRoPEKernel:
    """One drafter layer's block RoPE work in one launch, all loads up front.

    Rows ``[0, T * H)`` rotate the ``rope_dim`` tail of the query row ``q[r]`` at
    ``freqs[positions[r // H]]`` into ``query[r, kv_lora_rank:]`` (the rope half of the absorbed
    query; the latent half is left to the absorption GEMM). Rows ``[T * H, T * H + T)``
    RMS-normalize the latent of the block token ``t`` (``kv[t, :kv_lora_rank]``, scaled by
    ``weight``), rotate its ``k_pe`` at ``freqs[positions[t]]`` and store the row straight into
    the paged pool at ``pool[pages[t], offsets[t]]`` in the pool's dtype.
    """

    num_warps = 4
    num_threads = num_warps * cute.arch.WARP_SIZE

    def __init__(
        self,
        num_heads: int,
        qk_nope_dim: int,
        kv_lora_rank: int,
        rope_dim: int,
        eps: float,
    ):
        warp = cute.arch.WARP_SIZE
        if kv_lora_rank % warp != 0 or (rope_dim // 2) % warp != 0 or rope_dim % 2 != 0:
            raise ValueError(
                f"kv_lora_rank ({kv_lora_rank}) and rope_dim / 2 ({rope_dim // 2}) must be "
                f"multiples of {warp}"
            )
        if kv_lora_rank != _NORM_DIM:
            raise ValueError(f"the torch-order norm is written for kv_lora_rank {_NORM_DIM}")
        self.num_heads = num_heads
        self.qk_nope_dim = qk_nope_dim
        self.norm_dim = kv_lora_rank
        self.eps = eps
        self.norm_items = kv_lora_rank // warp
        self.pair_items = rope_dim // 2 // warp

    @cute.jit
    def _store(self, pool: cute.Tensor, page, offset, dim, value: Float32):
        pool[page, offset, dim] = value.to(pool.element_type)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        query: cute.Tensor,
        kv: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        offsets: cute.Tensor,
        stream: cuda.CUstream,
    ):
        rows = q.shape[0] + kv.shape[0]
        self.kernel(q, query, kv, weight, positions, freqs, pool, pages, offsets).launch(
            grid=[cute.ceil_div(rows, self.num_warps), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        query: cute.Tensor,
        kv: cute.Tensor,
        weight: cute.Tensor,
        positions: cute.Tensor,
        freqs: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        offsets: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % cute.arch.WARP_SIZE
        row = bidx * self.num_warps + tidx // cute.arch.WARP_SIZE
        num_q_rows = q.shape[0]
        reals = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
        imags = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
        coss = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
        sins = cute.make_rmem_tensor((self.pair_items,), cutlass.Float32)
        if row < num_q_rows:
            position = positions[row // self.num_heads]
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                reals[item] = cutlass.Float32(q[row, self.qk_nope_dim + pair * 2])
                imags[item] = cutlass.Float32(q[row, self.qk_nope_dim + pair * 2 + 1])
                coss[item] = cutlass.Float32(freqs[position, pair, 0])
                sins[item] = cutlass.Float32(freqs[position, pair, 1])
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                real = reals[item]
                imag = imags[item]
                cos = coss[item]
                sin = sins[item]
                query[row, self.norm_dim + pair * 2] = (real * cos - imag * sin).to(
                    query.element_type
                )
                query[row, self.norm_dim + pair * 2 + 1] = (imag * cos + real * sin).to(
                    query.element_type
                )
        elif row < num_q_rows + kv.shape[0]:
            token = row - num_q_rows
            values = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            scales = cute.make_rmem_tensor((self.norm_items,), cutlass.Float32)
            position = positions[token]
            page = pages[token]
            offset = offsets[token]
            for item in cutlass.range_constexpr(self.norm_items):
                dim = _norm_column(lane, item)
                values[item] = cutlass.Float32(kv[token, dim])
                scales[item] = cutlass.Float32(weight[dim])
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                reals[item] = cutlass.Float32(kv[token, self.norm_dim + pair * 2])
                imags[item] = cutlass.Float32(kv[token, self.norm_dim + pair * 2 + 1])
                coss[item] = cutlass.Float32(freqs[position, pair, 0])
                sins[item] = cutlass.Float32(freqs[position, pair, 1])

            # torch normalizes the [T, 512] block latents as T rows.
            inverse_rms = _inverse_rms_torch_order(values, kv.shape[0], self.eps)

            for item in cutlass.range_constexpr(self.norm_items):
                dim = _norm_column(lane, item)
                value = _mul_rn(_mul_rn(values[item], inverse_rms), scales[item])
                self._store(pool, page, offset, dim, value)
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                real = reals[item]
                imag = imags[item]
                cos = coss[item]
                sin = sins[item]
                self._store(pool, page, offset, self.norm_dim + pair * 2, real * cos - imag * sin)
                self._store(
                    pool, page, offset, self.norm_dim + pair * 2 + 1, imag * cos + real * sin
                )


#: Largest finite FP8 E4M3 value: the drafter's FP8 latents saturate there.
_E4M3_MAX = 448.0


@dsl_user_op
def _e4m3_byte(value: Float32, *, loc=None, ip=None) -> Uint8:
    """``value`` (already clamped to the E4M3 range) as the E4M3 byte, rounded to nearest even."""
    pair = llvm.inline_asm(
        T.i16(),
        [value.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, 0f00000000, $1;",
        "=h,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Uint8(cutlass.Int16(pair))


class K31DraftMlaBlockRoPEFp8Kernel(K31DraftMlaBlockRoPEKernel):
    """``K31DraftMlaBlockRoPEKernel`` storing into a BF16 pool, or with ``fp8_pool`` into an FP8 E4M3 one.

    With ``fp8_pool`` (the pool passed as its bytes) each stored value is the BF16 value saturated to +-448 and
    rounded to E4M3: the drafter history at a unit scale that the sm_107 targets keep.
    """

    def __init__(
        self,
        num_heads: int,
        qk_nope_dim: int,
        kv_lora_rank: int,
        rope_dim: int,
        eps: float,
        fp8_pool: bool = False,
    ):
        super().__init__(num_heads, qk_nope_dim, kv_lora_rank, rope_dim, eps)
        self.fp8_pool = fp8_pool

    @cute.jit
    def _store(self, pool: cute.Tensor, page, offset, dim, value: Float32):
        if cutlass.const_expr(self.fp8_pool):
            rounded = Float32(value.to(cutlass.BFloat16))
            # Saturate like torch.clamp: comparisons are false for NaN, which passes through.
            if rounded > Float32(_E4M3_MAX):
                rounded = Float32(_E4M3_MAX)
            if rounded < Float32(-_E4M3_MAX):
                rounded = Float32(-_E4M3_MAX)
            pool[page, offset, dim] = _e4m3_byte(rounded)
        else:
            pool[page, offset, dim] = value.to(pool.element_type)


__all__ = [
    "K31DraftMlaBlockRoPEFp8Kernel",
    "K31DraftMlaBlockRoPEKernel",
    "K31DraftMlaCtxNormRoPEKernel",
]
