# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The wide MoE tail's up projection fused with its output (``tail_rows``: in place of
the FP32 GEMM and ``OutRows``): BF16(xnorm @ W_shard^T + shared) of this rank's output
columns, multicast into every rank's output ring row.

``SplitKTmemGemm64`` with the weights as the MMA's M operand: each 128-column tile of
the shard's weight rows (448 at TP16: four tiles, the last half empty) splits K over a
cluster of ``split`` CTAs whose K slices sit in tensor memory before the PDL wait;
xnorm streams after it in ``n``-token tiles (grid y), so every column tile reads the
activations once per token tile. The cluster's FP32 partials meet in one DSMEM
reduce-scatter (each owner sums the ranks in order); the owner adds the shared rows
(FP32), rounds once (-0 made +0) and multicasts 8 columns of a token per store. Versus
the FP32 GEMM + ``OutRows`` this reorders the FP32 accumulation over K only."""

from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common.primitives import (
    asm,
    at,
    bits_float,
    load_words,
    multicast_words,
    named_barrier,
    pack_pair,
)
from ..gemm.splitk import SplitKTactic
from ..gemm.splitk_tmem64 import SplitKTmemGemm64, _ld_shared_v4

H, LATENT = 7168, 3584


@asm()
def _st_shared_f32(address, value):
    return "st.shared.f32 [$0], $1;", "r,f,~{memory}", (Int32(address), Float32(value))


class UpTmem64(SplitKTmemGemm64):
    """``columns`` output columns (the shard's weight rows) in 128-row tiles; K 3584
    split over ``split`` CTAs (each its own owner); ``n``-token tiles; ``slots`` x
    tiles in flight."""

    def __init__(self, *, columns, split=4, n=64, slots=10):
        tactic = SplitKTactic(128, n, split, slots)
        super().__init__(tactic=tactic, k=LATENT, owners=split)
        self.columns, self.tiles = columns, -(-columns // 128)

    @cute.experimental.jit
    def __call__(
        self, weight: cute.Tensor, x: cute.Tensor, shared: cute.Tensor,
        out_mc: cutlass.Int64, column: cutlass.Int32, w_map: cutlass.Int64,
        stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(weight, x, shared, out_mc, column, w_map).launch(
            grid=(
                self.tiles,
                cute.ceil_div(x.layout.shape[0], self.n) * self.split_k,
                1,
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, shared: cute.Tensor,
        out_mc: cutlass.Int64, column: cutlass.Int32, w_map: cutlass.Int64,
    ):  # fmt: skip
        out = (shared, out_mc, column, mB.layout.shape[0])
        self.body(mA, mB, shared, w_map, out)

    @cute.experimental.jit
    def finish_inputs(self, epi_tid, out, row0, col0):
        """The shared rows of this thread's ``finish`` items (8 FP32 columns each),
        loaded while the cluster reduces; ``shared`` comes from the predecessor."""
        shared, _, _, tokens = out
        cute.arch.griddepcontrol_wait()
        parts: cutlass.Constexpr = self.share // 8
        items: cutlass.Constexpr = self.n * parts // 128
        adds = cute.make_rmem_tensor(8 * items, Float32)
        for i in cutlass.range_constexpr(items):
            item = epi_tid + 128 * i
            token, row = col0 + item // parts, row0 + 8 * (item % parts)
            for k in cutlass.range_constexpr(8):
                adds[8 * i + k] = Float32(0)
            if (token < tokens) & (row < self.columns):
                for h in cutlass.range_constexpr(2):
                    pointer = at(shared, token, row) + 4 * h
                    words = load_words(pointer, count=4, mutable=True)
                    for k in cutlass.range_constexpr(4):
                        adds[8 * i + 4 * h + k] = bits_float(words[k])
        return adds

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """An owner's share [share columns] x n tokens transposed through ``tile`` [n
        tokens][share]; then (token, 8 columns) items: + the prefetched shared rows,
        one rounding, multicast into the token's output ring row."""
        _, out_mc, column, tokens, adds = out
        share = self.share
        rr, g = epi_tid % share, epi_tid // share
        width: cutlass.Constexpr = self.n * share // 128  # tokens per thread
        for t in cutlass.range_constexpr(width):
            _st_shared_f32(tile + ((width * g + t) * share + rr) * 4, totals[t])
        named_barrier(3, 128)
        parts: cutlass.Constexpr = share // 8
        for i in cutlass.range_constexpr(self.n * parts // 128):
            item = epi_tid + 128 * i
            token, part = item // parts, item % parts
            row = row0 + 8 * part
            if (col0 + token < tokens) & (row < self.columns):
                chunk = tile + (token * share + 8 * part) * 4
                values = [bits_float(w) for w in _ld_shared_v4(chunk)]
                values += [bits_float(w) for w in _ld_shared_v4(chunk + 16)]
                words = tuple(
                    pack_pair(
                        values[2 * j] + adds[8 * i + 2 * j],
                        values[2 * j + 1] + adds[8 * i + 2 * j + 1],
                        policy="latent",
                    )
                    for j in range(4)
                )
                offset = (col0 + token) * H + column + row
                multicast_words(out_mc + Int64(offset) * 2, words, kind="bf16x2")


@cache
def compile_up_tmem(columns, device, split=4, n=64, slots=10):
    """``UpTmem64`` for any token count: (weight, x) BF16 views (``matrix_view``),
    shared [tokens, columns] FP32 (``dynamic_view``), the output ring's multicast
    address, this rank's first output column, the weight's {64, 128} tensor map."""
    kernel = UpTmem64(columns=columns, split=split, n=n, slots=slots)

    def matrix():
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    shared = make_fake_tensor(
        Float32, (cute.sym_int(), cute.sym_int()), (cute.sym_int(64), 1),
        assumed_align=16,
    )  # fmt: skip
    args = (matrix(), matrix(), shared, Int64(0), Int32(0), Int64(0))
    with torch.accelerator.device_index(device):
        arch = "sm_{}{}a".format(*torch.cuda.get_device_capability(device))
        return cute.experimental.compile(
            kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}"
        )
