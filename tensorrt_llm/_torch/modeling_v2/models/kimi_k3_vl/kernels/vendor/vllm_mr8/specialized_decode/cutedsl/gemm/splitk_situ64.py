# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark drafter's dense gate/up of 33-64 rows (B5-B8) with vLLM's silu_and_mul in
its epilogue, on ``SplitKTmemGemm64`` (weights on chip before x lands).

Each 128-row A tile is 64 gate rows and their 64 up rows (two {64, 64} boxes per K
tile instead of one {64, 128} box), so every output row keeps the K split, the MMA K
order and the owners' rank-order partial sums of the gate/up GEMM with the same
tactic. Owner o of a tile reduces gate and up features 16 o .. 16 o + 15 (its 32-row
share: the 16 gate rows, then their 16 up rows), stages the FP32 sums in shared
memory, rounds gate and up to BF16 as the GEMM output would be, and applies
``SplitKSiluMul``'s silu_and_mul (silu(gate) rounded to BF16, times up in FP32, BF16
h): bitwise the GEMM followed by silu_and_mul."""

from functools import cache

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ..common.primitives import asm, at, bf16x2, div_rn, named_barrier, store_words
from .splitk import SplitKTactic
from .splitk_tmem64 import SHARE, SplitKTmemGemm64, _tma_box


@asm()
def _st_shared_f32(address, value):
    return "st.shared.f32 [$0], $1;", "r,f,~{memory}", (Int32(address), Float32(value))


@asm(Float32)
def _ld_shared_f32(address):
    return "ld.shared.f32 $0, [$1];", "=f,r,~{memory}", (Int32(address),)


class SplitKTmemSiluMul64(SplitKTmemGemm64):
    """h [B, half] = silu_and_mul(x W_gate^T, x W_up^T) for a [2 half, K] gate/up
    weight; ``a_map`` must have {64, 64} boxes (``weight_map(weight, 64)``). Default
    owners (4 x 32-row shares)."""

    def __init__(self, *, tactic, k, half):
        super().__init__(tactic=tactic, k=k)
        assert self.share == SHARE and half % 64 == 0
        self.half = half

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, h: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        # h is (half, N, 1) M-major: grid-x walks 64-feature tiles.
        self.kernel(a, b, h, bias, a_map).launch(
            grid=(
                self.half // 64,
                cute.ceil_div(h.layout.shape[1], self.cta_n) * self.split_k,
                h.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    def issue_a(self, dst, barrier, a_map, k0, row0):
        """Tile ``row0 // 128``'s gate rows, then its up rows (64 each)."""
        first = row0 // 128 * 64
        _tma_box(dst, barrier, a_map, k0, first)
        _tma_box(dst + 64 * 128, barrier, a_map, k0, first + self.half)

    def owner_of(self, row, row0):
        """Accumulator row ``row`` (gate feature f < 64, then up feature f) -> (owner
        f // 16, share row: 16 gate rows, then 16 up rows)."""
        feature = row % 64
        return feature // 16, row // 64 * 16 + feature % 16

    @cute.jit
    def activate(self, gate, up):
        one = Float32(1)
        silu = div_rn(gate, one + cute.math.exp(-gate, fastmath=False))
        return Float32(cutlass.BFloat16(silu)) * (up + Float32(0))

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """Owner (row0 % 128) // 32 of tile row0 // 128: thread (share row rr, tokens
        16 g .. 16 g + 15) holds ``totals``; staged FP32 [64 tokens][32 rows], then
        thread (token, half) writes h[token, 8 features] (``out`` = (h,))."""
        mH = out[0]
        rr, g = epi_tid % SHARE, epi_tid // SHARE
        for t in cutlass.range_constexpr(16):
            _st_shared_f32(tile + ((16 * g + t) * SHARE + rr) * 4, totals[t])
        named_barrier(3, 128)
        token, part = epi_tid // 2, epi_tid % 2
        feature0 = row0 // 128 * 64 + (row0 % 128) // SHARE * 16 + 8 * part
        if col0 + token < mH.layout.shape[1]:
            words = []
            for p in cutlass.range_constexpr(4):
                pair = []
                for e in cutlass.range_constexpr(2):
                    f = 8 * part + 2 * p + e
                    raw_gate = _ld_shared_f32(tile + (token * SHARE + f) * 4)
                    raw_up = _ld_shared_f32(tile + (token * SHARE + 16 + f) * 4)
                    gate = Float32(cutlass.BFloat16(raw_gate))
                    up = Float32(cutlass.BFloat16(raw_up))
                    pair.append(self.activate(gate, up))
                words.append(bf16x2(pair[0], pair[1]))
            store_words(at(mH, feature0, col0 + token, l_idx), tuple(words))


@cache
def compile_situ64(tactic, k, half, arch):
    """Code-only cache: ``SplitKTmemSiluMul64`` for ``tactic`` (128, 64, split, x's
    slots, TMEM_TILES) and reduction length ``k``."""

    def fake(leading):
        shape = tuple(cute.sym_int() for _ in range(3))
        stride = (1, cute.sym_int(64), 0) if leading == 0 else (cute.sym_int(64), 1, 0)
        return make_fake_tensor(cutlass.BFloat16, shape, stride, assumed_align=16)

    kernel = SplitKTmemSiluMul64(tactic=SplitKTactic(*tactic[:4]), k=k, half=half)
    args = (fake(1), fake(1), fake(0), fake(0), Int64(0), make_fake_stream())
    return cute.experimental.compile(kernel, *args, options=f"--gpu-arch={arch}")
