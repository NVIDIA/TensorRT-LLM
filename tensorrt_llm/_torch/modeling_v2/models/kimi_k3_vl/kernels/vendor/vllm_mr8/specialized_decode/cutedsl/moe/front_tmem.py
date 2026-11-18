# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin MoE front at 64 tokens (spec verify B8 x q8): the front GEMM rows and the
shared expert's gate/up in one ``SplitKTmemGemm64`` grid.

The merged gate kernel (``SharedUmma`` with the front rows) streams x after the PDL
wait through a ring that still holds weights. Here each 128-row tile's K slice is in
tensor memory before x lands (clusters of 8, one CTA per SM, 14 K tiles each):
tiles 0 .. 2 are this rank's packed front rows [router (``valid``, padded to
``split``); latent-down shard], tiles 3 .. 8 hold 64 gate rows and the matching 64
up rows each. Front owners publish like ``WsGemm.epilogue`` (FP32 logits and BF16
latent columns, -0 made +0, multicast into the front mailbox); gate/up owners take
8 gate rows and their 8 up rows, round both to BF16, apply SiTU like the merged
kernel and store h for ``SharedDown``. Every rank owns a 16-row share (8 owners:
the SiTU work spreads over every CTA). The split partials are summed in rank order
(an FP32 reorder of the merged kernel's six partials)."""

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64

from ..common.primitives import asm, bits_float, multicast_words, named_barrier
from ..gemm.splitk import SplitKTactic
from ..gemm.splitk_tmem64 import SplitKTmemGemm64, _ld_shared_v4, _tma_box
from ..gemm.ws_gemm import positive_zero, positive_zero32
from .shared import situ

__all__ = ["MoeFrontRows", "MoeFrontTmem64"]

FRONT_TILES = 3  # 288 packed front rows in 128-row tiles


@asm()
def _st_shared_f32(address, value):
    return "st.shared.f32 [$0], $1;", "r,f,~{memory}", (Int32(address), Float32(value))


@asm()
def _tma_half(dst, barrier, tensor_map, k0, row0):
    """One {64, 64} box at (k0, row0) to shared address ``dst`` (half an A tile)."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
    code += "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes.L2::cache_hint [$0], [$1, {$2, $3}], [$4], pol; }"
    operands = (Int32(dst), tensor_map, Int32(k0), Int32(row0), barrier)
    return code, "r,l,r,r,r,~{memory}", operands


class MoeFrontTmem64(SplitKTmemGemm64):
    """Front rows (``valid`` router rows padded to ``split_rows``, then ``shard``
    latent rows) and ``hidden`` gate/up pairs at 64 tokens; split K 8."""

    def __init__(self, *, valid, split_rows, shard, hidden, stamps=False):
        super().__init__(
            tactic=SplitKTactic(128, 64, 8, 14), k=7168, stamps=stamps, owners=8
        )
        self.valid, self.split_rows, self.shard, self.hidden = (
            valid, split_rows, shard, hidden,
        )  # fmt: skip
        self.gate_tiles = hidden // 64
        self.tiles = FRONT_TILES + self.gate_tiles

    @cute.experimental.jit
    def __call__(
        self, packed: cute.Tensor, x: cute.Tensor, h: cute.Tensor, box: cute.Tensor,
        mc: cutlass.Int64, slot: cutlass.Int32, base: cutlass.Int32,
        beta: cutlass.Float32, linear_beta: cutlass.Float32, front_map: cutlass.Int64,
        gate_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            packed, x, h, box, mc, slot, base, beta, linear_beta, front_map, gate_map
        ).launch(
            grid=(self.tiles, cute.ceil_div(x.layout.shape[0], 64) * self.split_k, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, h: cute.Tensor, box: cute.Tensor,
        mc: cutlass.Int64, slot: cutlass.Int32, base: cutlass.Int32,
        beta: cutlass.Float32, linear_beta: cutlass.Float32, front_map: cutlass.Int64,
        gate_map: cutlass.Int64,
    ):  # fmt: skip
        tokens = mB.layout.shape[0]
        self.gate_map = gate_map
        out = (box, mc, slot, base, tokens, h, beta, linear_beta)
        self.body(mA, mB, box, front_map, out)

    @cute.experimental.jit
    def issue_a(self, dst, barrier, a_map, k0, row0):
        """Front tiles: one {64, 128} box of the packed rows; gate/up tile g: gate
        rows [64 g, + 64) and up rows [hidden + 64 g, + 64) as two {64, 64} boxes."""
        tile = row0 // 128
        if tile < FRONT_TILES:
            _tma_box(dst, barrier, a_map, k0, row0)
        else:
            gate = (tile - FRONT_TILES) * 64
            _tma_half(dst, barrier, self.gate_map, k0, gate)
            _tma_half(dst + 8192, barrier, self.gate_map, k0, self.hidden + gate)

    def owner_of(self, row, row0):
        """Front tiles: 16-row shares; gate/up tiles: owner o takes gate rows
        [8 o, + 8) and their up rows (share rows 0-7 gate, 8-15 up)."""
        gate = Int32(row0 >= FRONT_TILES * 128)
        owner = gate * (row % 64 // 8) + (1 - gate) * (row // 16)
        share_row = gate * (row // 64 * 8 + row % 8) + (1 - gate) * (row % 16)
        return owner, share_row

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """An owner's FP32 share [16 rows] x 64 tokens (thread: row rr, tokens
        8 g .. 8 g + 7), transposed through ``tile`` [64 tokens][16]; front tiles
        publish one 8-row chunk per thread, gate/up tiles store 4 SiTU columns of h
        per thread."""
        box, mc, slot, base, tokens, h, beta, linear_beta = out
        rr, g = epi_tid % 16, epi_tid // 16
        for t in cutlass.range_constexpr(8):
            _st_shared_f32(tile + ((8 * g + t) * 16 + rr) * 4, totals[t])
        named_barrier(3, 128)
        m, owner = row0 // 128, row0 % 128 // 16
        token, part = epi_tid // 2, epi_tid % 2
        if col0 + token < tokens:
            chunk = tile + token * 64
            if m < FRONT_TILES:
                # 64 tokens: box row = token (``MoeFrontRows``: col0 + token)
                self.publish(box, mc, slot, base, chunk, part, row0, token)
            else:
                column = (m - FRONT_TILES) * 64 + 8 * owner + 4 * part
                gates = [bits_float(w) for w in _ld_shared_v4(chunk + 16 * part)]
                ups = [bits_float(w) for w in _ld_shared_v4(chunk + 32 + 16 * part)]
                for c in cutlass.range_constexpr(4):
                    gate, up = Float32(BFloat16(gates[c])), Float32(BFloat16(ups[c]))
                    value = BFloat16(situ(gate, up, beta, linear_beta))
                    h[col0 + token, column + c] = value

    @cute.experimental.jit
    def publish(
        self, box: cute.Tensor, mc: cutlass.Int64, slot: cutlass.Int32,
        base: cutlass.Int32, chunk: Int32, part: Int32, row0: cutlass.Int32,
        token: Int32,
    ):  # fmt: skip
        """Front rows ``row0 + 8 part`` .. + 7 of ``token`` (FP32 at shared address
        ``chunk``) into its front-box row: router rows as FP32 logit words, latent
        rows as BF16 (-0 made +0), multicast."""
        pitch = box.layout.shape[2]
        row = row0 + 8 * part
        values = []
        for v in cutlass.range_constexpr(2):
            values += [bits_float(w) for w in _ld_shared_v4(chunk + 32 * part + 16 * v)]
        if row < self.valid:
            for v in cutlass.range_constexpr(2):
                quad = tuple(positive_zero32(values[4 * v + i]) for i in range(4))
                offset = token * pitch + 3584 + 2 * (slot + row + 4 * v)
                multicast_words(mc + Int64(offset) * 2, quad, kind="bf16x2")
        elif (row >= self.split_rows) & (row - self.split_rows < self.shard):
            words = tuple(
                positive_zero(values[i], values[i + 1]) for i in range(0, 8, 2)
            )
            offset = token * pitch + base + row - self.split_rows
            multicast_words(mc + Int64(offset) * 2, words, kind="bf16x2")


class MoeFrontRows(MoeFrontTmem64):
    """``MoeFrontTmem64`` at any token count: the three front tiles and ``hidden`` // 64
    gate/up tiles (0: the front alone) per ``n``-token tile (grid y; ``n`` 64, 96, 128 or
    160: 13 / 12 / 11 weight tiles in tensor memory beside the wider accumulator); each token is
    published into its own front-box row (a box of at least ``tokens`` rows) and its
    h row stored. Every row keeps the K split and the owners' summation order: per
    token, bitwise the 64-token kernel's outputs at either N."""

    def __init__(self, *, valid, split_rows, shard, hidden=0, n=64):
        tactic = SplitKTactic(128, n, 8, 14 if n == 64 else 8)  # N96 / N128 / N160: 8 x slots
        SplitKTmemGemm64.__init__(self, tactic=tactic, k=7168, owners=8)
        self.valid, self.split_rows, self.shard, self.hidden = (
            valid, split_rows, shard, hidden,
        )  # fmt: skip
        self.gate_tiles = hidden // 64
        self.tiles = FRONT_TILES + self.gate_tiles

    @cute.experimental.jit
    def __call__(
        self, packed: cute.Tensor, x: cute.Tensor, h: cute.Tensor, box: cute.Tensor,
        mc: cutlass.Int64, slot: cutlass.Int32, base: cutlass.Int32,
        beta: cutlass.Float32, linear_beta: cutlass.Float32, front_map: cutlass.Int64,
        gate_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            packed, x, h, box, mc, slot, base, beta, linear_beta, front_map, gate_map
        ).launch(
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

    @cute.experimental.jit
    def finish(
        self, epi_tid: cutlass.Int32, totals: cute.Tensor, tile: Int32, out,
        row0: cutlass.Int32, col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """``MoeFrontTmem64.finish`` over ``n`` tokens: the share [16 rows] x n tokens
        transposed through ``tile`` [n tokens][16], then n / 64 (token, half) items per
        thread: front tiles publish 8 rows, gate/up tiles store 4 SiTU columns of h."""
        box, mc, slot, base, tokens, h, beta, linear_beta = out
        rr, g = epi_tid % 16, epi_tid // 16
        width: cutlass.Constexpr = self.n // 8  # tokens per thread
        for t in cutlass.range_constexpr(width):
            _st_shared_f32(tile + ((width * g + t) * 16 + rr) * 4, totals[t])
        named_barrier(3, 128)
        m, owner = row0 // 128, row0 % 128 // 16
        for i in cutlass.range_constexpr(-(-self.n // 64)):
            token, part = (epi_tid + 128 * i) // 2, epi_tid % 2
            if cutlass.const_expr(self.n % 64 != 0):  # N96 / N160: the tile's last items
                token = cutlass.select_(token < self.n, token, tokens)  # past the tile: skip
            if col0 + token < tokens:
                chunk = tile + token * 64
                if m < FRONT_TILES:
                    self.publish(box, mc, slot, base, chunk, part, row0, col0 + token)
                elif cutlass.const_expr(self.gate_tiles > 0):
                    column = (m - FRONT_TILES) * 64 + 8 * owner + 4 * part
                    gates = [bits_float(w) for w in _ld_shared_v4(chunk + 16 * part)]
                    ups = [bits_float(w) for w in _ld_shared_v4(chunk + 32 + 16 * part)]
                    for c in cutlass.range_constexpr(4):
                        gate = Float32(BFloat16(gates[c]))
                        up = Float32(BFloat16(ups[c]))
                        value = BFloat16(situ(gate, up, beta, linear_beta))
                        h[col0 + token, column + c] = value
