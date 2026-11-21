# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 MLA drafter's fused RMSNorm + RoPE kernels (the ``dspark_tail_v2`` constant).

Per row they compute what ``DSparkRMSNormRoPEKernel`` (``attention/kernels/blackwell/
dspark_rmsnorm_rope.py``) computes, with every load of a row issued before its first store, so the
values are bitwise those of the launches they replace:

* :class:`DSparkMlaCtxNormRoPEKernel`: the context latents of every drafter layer in one launch;
* :class:`DSparkMlaBlockRoPEKernel`: one layer's query RoPE and block-latent RMSNorm + RoPE, the
  latents stored straight into the paged pool.

Launchers: ``ops/spec/_dspark_mla_rope.py``.
"""

import cutlass
import cutlass.cute as cute

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda


class DSparkMlaCtxNormRoPEKernel:
    """MLA context latents of every drafter layer in one launch: RMSNorm + RoPE.

    Row ``r`` of the output is layer ``r // N``, token ``r % N``: the
    ``kv_lora_rank`` latent of ``x[token, layer]`` RMS-normalized and scaled by
    ``weight[layer]``, then the ``rope_dim`` tail rotated at
    ``freqs[positions[token]]`` (adjacent pairs, no norm, no weight).

    One warp per row, and per row the arithmetic of
    ``DSparkRMSNormRoPEKernel(kv_lora_rank + rope_dim, rope_dim, 1, eps, True,
    True, False, norm_dim=kv_lora_rank)``: lane ``l`` owns columns ``l + 32 i``,
    accumulates their squares in ``i`` order, the warp reduction and ``rsqrt``
    are the same calls, and each output is the same chain of FP32 products. The
    results are therefore bitwise those of the per-layer launches. What differs:
    every load of a row is issued before its first store (the per-layer kernel
    interleaves loads with stores the compiler cannot reorder, so each of its
    16 weight loads waits for the previous store), the layers share one launch,
    and the RoPE table is indexed by position in the kernel instead of through a
    gathered copy.
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
                dim = lane + item * cute.arch.WARP_SIZE
                values[item] = cutlass.Float32(x[token, col0 + dim])
                scales[item] = cutlass.Float32(weight[layer, dim])
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                reals[item] = cutlass.Float32(x[token, col0 + self.norm_dim + pair * 2])
                imags[item] = cutlass.Float32(x[token, col0 + self.norm_dim + pair * 2 + 1])
                coss[item] = cutlass.Float32(freqs[position, pair, 0])
                sins[item] = cutlass.Float32(freqs[position, pair, 1])

            sum_sq = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(self.norm_items):
                value = values[item]
                sum_sq += value * value
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            inverse_rms = cute.math.rsqrt(sum_sq / self.norm_dim + self.eps)

            for item in cutlass.range_constexpr(self.norm_items):
                dim = lane + item * cute.arch.WARP_SIZE
                value = values[item] * inverse_rms
                value *= scales[item]
                output[row, dim] = value.to(output.element_type)
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                real = reals[item]
                imag = imags[item]
                cos = coss[item]
                sin = sins[item]
                output[row, self.norm_dim + pair * 2] = (real * cos - imag * sin).to(
                    output.element_type
                )
                output[row, self.norm_dim + pair * 2 + 1] = (imag * cos + real * sin).to(
                    output.element_type
                )


class DSparkMlaBlockRoPEKernel:
    """One drafter layer's block RoPE work in one launch, all loads up front.

    Rows ``[0, T * H)`` rotate the ``rope_dim`` tail of the 192-wide query row
    ``q[r]`` at ``freqs[positions[r // H]]`` into ``query[r, kv_lora_rank:]``
    (the rope half of the 576-wide absorbed query; the latent half is left to
    the absorption GEMM). Rows ``[T * H, T * H + T)`` RMS-normalize the latent of
    the block token ``t`` (``kv[t, :kv_lora_rank]``, scaled by ``weight``),
    rotate its ``k_pe`` at ``freqs[positions[t]]`` and store the 576-wide row
    straight into the paged pool at ``pool[pages[t], offsets[t]]``.

    Per row the arithmetic is that of the separate launches it replaces --
    ``cute_dsl_dspark_rope_into`` for the query rows and
    ``cute_dsl_dspark_rmsnorm_rope(norm_dim=kv_lora_rank)`` followed by the
    pool scatter for the latent rows -- so the values written are bitwise the
    same.
    """

    num_warps = 4
    num_threads = num_warps * cute.arch.WARP_SIZE

    def __init__(
        self, num_heads: int, qk_nope_dim: int, kv_lora_rank: int, rope_dim: int, eps: float
    ):
        warp = cute.arch.WARP_SIZE
        if kv_lora_rank % warp != 0 or (rope_dim // 2) % warp != 0 or rope_dim % 2 != 0:
            raise ValueError(
                f"kv_lora_rank ({kv_lora_rank}) and rope_dim / 2 ({rope_dim // 2}) must be "
                f"multiples of {warp}"
            )
        self.num_heads = num_heads
        self.qk_nope_dim = qk_nope_dim
        self.norm_dim = kv_lora_rank
        self.eps = eps
        self.norm_items = kv_lora_rank // warp
        self.pair_items = rope_dim // 2 // warp

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
                dim = lane + item * cute.arch.WARP_SIZE
                values[item] = cutlass.Float32(kv[token, dim])
                scales[item] = cutlass.Float32(weight[dim])
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                reals[item] = cutlass.Float32(kv[token, self.norm_dim + pair * 2])
                imags[item] = cutlass.Float32(kv[token, self.norm_dim + pair * 2 + 1])
                coss[item] = cutlass.Float32(freqs[position, pair, 0])
                sins[item] = cutlass.Float32(freqs[position, pair, 1])

            sum_sq = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(self.norm_items):
                value = values[item]
                sum_sq += value * value
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            inverse_rms = cute.math.rsqrt(sum_sq / self.norm_dim + self.eps)

            for item in cutlass.range_constexpr(self.norm_items):
                dim = lane + item * cute.arch.WARP_SIZE
                value = values[item] * inverse_rms
                value *= scales[item]
                pool[page, offset, dim] = value.to(pool.element_type)
            for item in cutlass.range_constexpr(self.pair_items):
                pair = lane + item * cute.arch.WARP_SIZE
                real = reals[item]
                imag = imags[item]
                cos = coss[item]
                sin = sins[item]
                pool[page, offset, self.norm_dim + pair * 2] = (real * cos - imag * sin).to(
                    pool.element_type
                )
                pool[page, offset, self.norm_dim + pair * 2 + 1] = (imag * cos + real * sin).to(
                    pool.element_type
                )
