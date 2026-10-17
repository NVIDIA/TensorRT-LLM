# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BF16 shared expert of a GLM-5.2 MoE layer in two small kernels (CuTe DSL, SM100 family); an
alternative to the native shared expert, used by ``shared_prefetch_se.py``.

The per-rank shared expert is ``y = down(silu(gate(x)) * up(x))`` with
``gate_up`` ``[512, 6144]`` (rows 0..255 gate, 256..511 up) and ``down``
``[6144, 256]``, all BF16, for ``R <= 8`` rows. Rounding points are the
native ones: gate and up rounded to BF16 after an FP32 accumulation, the
SiLU-and-mul in FP32 from those BF16 values (the native Triton instruction
sequence, ``ptx.silu_mul_triton``) rounded to BF16, the down projection
accumulated in FP32 and rounded to BF16. Only the FP32 summation order inside
each GEMM differs from cuBLAS (numerical variant ``se-reorder``).

``SharedExpertTwoKernelUp`` (``ctas`` CTAs of 8 warps): CTA ``c`` owns the
intermediate columns ``[c * cols, (c + 1) * cols)``; its gate and up rows form
``cols / 8`` tiles of 16 rows (8 gate rows over the 8 up rows of the same
columns), so the ``mma.sync`` m16n8k16 accumulator of a thread holds the gate
and the up value of one column for two tokens. The 8 warps split K = 6144
(768 each) and their FP32 partial tiles are summed in shared memory in warp
order 0..7. It writes ``a`` (BF16 ``[R, 256]``, a per-module scratch buffer).

``SharedExpertTwoKernelDown`` (``ctas`` CTAs of 8 warps): CTA ``c`` owns output
features ``[c * F, (c + 1) * F)``, ``F = 6144 / ctas``; each warp computes
16-feature tiles over the full K = 256 from ``a`` and writes ``y`` (BF16
``[R, 6144]``).

Both kernels read every weight with 16-byte non-coherent loads straight into
``mma.sync`` fragments: thread ``(g, t)`` loads 8 consecutive K elements of
its rows per 32-wide K chunk and the activation row ``g`` at the same 8 K
positions, so A and B use the same permutation of K inside the chunk.
Activation columns of tokens ``>= R`` read row ``R - 1`` and their outputs are
never stored (MMA columns are independent).

With ``stage`` the kernels read the same weight bytes from shared memory
instead: before ``griddepcontrol.wait`` each warp bulk-copies
(``cp.async.bulk``, completion on its own mbarrier) the weight rows it reads
into a padded shared-memory image (row pitch + 16 bytes, so that the 16-byte
fragment loads of a warp are bank-conflict free), waits for its mbarrier after
the dependency wait, and runs the same ``mma.sync`` sequence on ``ld.shared``
fragments. The arithmetic and its order are unchanged, so the outputs are
bit-identical to the unstaged kernels. The up kernel stages its 16 rows
(192 KB, ``ctas = 32`` only); the down kernel its ``F`` rows (64 KB at 48
CTAs, 128 KB at 24). With ``trigger = "early"`` the up kernel releases its
dependent (the down kernel) right after its dependency wait, so that the down
kernel can stage its rows while the up kernel computes.

Memory order and progress (programmatic dependent launch as served):

* Before ``griddepcontrol.wait`` the up kernel reads nothing and writes
  nothing; with ``prefetch`` it issues ``cp.async.bulk.prefetch.L2`` of its
  own gate/up rows (bit 1) and of one 1/ctas slice of ``down`` (bit 2), all
  immutable weights. The down kernel likewise prefetches its own ``down``
  rows (bit 4).
* After the wait, the up kernel reads ``x`` (written by the pre-MoE hop,
  plain coherent loads) and writes ``a``; the down kernel reads ``a`` (written
  by the up kernel, complete when its wait returns) and writes ``y``. ``a``'s
  previous reader is the down kernel of the module's previous call, complete
  before this call's predecessors ran; ``y`` is a fresh output tensor.
* With ``stage``, the pre-wait bulk copies read only immutable weights and
  write only the CTA's own shared memory; ``griddepcontrol.wait`` still
  precedes every read of ``x`` or ``a``. The early trigger changes only when
  the down kernel is scheduled: its own wait returns after the up kernel has
  completed and its writes of ``a`` are visible.
* No CTA waits for another CTA: the only cross-CTA dependency is the kernel
  boundary between the two launches, so progress does not depend on
  co-residency.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import shared_prefetch_se_ptx as ptx
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
    ptx as wptx,
)

HIDDEN = 6144
INTER = 256  # per-rank shared intermediate size (2048 / TP8)
NTOK = 8
THREADS = 256
WARPS = THREADS // 32
W1_ROW_BYTES = HIDDEN * 2
W2_ROW_BYTES = INTER * 2
PF_W1 = 1
PF_W2_SLICE = 2
PF_W2_OWN = 4
PAD = 16  # shared-memory row padding of the staged weight images
ROWP1 = W1_ROW_BYTES + PAD
ROWP2 = W2_ROW_BYTES + PAD
SLICE1 = HIDDEN // WARPS * 2  # bytes of one gate/up row in one warp's K range


class SharedExpertTwoKernelUp:
    """gate/up GEMM + SiLU-and-mul (see the module docstring)."""

    def __init__(
        self,
        ctas: int = 32,
        prefetch: int = PF_W1 | PF_W2_SLICE,
        unroll: int = 0,
        silu: int = ptx.SILU_DEFAULT,
        carveout: int = 100,
        stage: bool = False,
        trigger: str = "end",
    ):
        assert ctas in (16, 32), ctas
        assert not stage or ctas == 32, "the staged up kernel holds 16 rows per CTA"
        assert trigger in ("end", "early"), trigger
        self.stage = stage
        self.trigger = trigger
        self.ctas = ctas
        self.cols = INTER // ctas
        self.tiles = self.cols // 8
        self.chunks = HIDDEN // (WARPS * 32)
        unroll = unroll or 8 // self.tiles
        assert self.chunks % unroll == 0
        self.unroll = unroll
        self.prefetch = prefetch
        self.silu = silu
        self.carveout = carveout

    @cute.jit
    def run(
        self,
        x_addr: Int64,
        w1_addr: Int64,
        w2_addr: Int64,
        a_addr: Int64,
        num_tokens: Int32,
        stream,
    ):
        @cute.struct
        class SharedStorage:
            part: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, WARPS * self.tiles * 128], 16
            ]

        @cute.struct
        class StagedStorage:
            part: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, WARPS * self.tiles * 128], 16
            ]
            mbar: cute.struct.MemRange[cutlass.Int64, WARPS]
            wsm: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, 2 * self.cols * ROWP1 // 8], 128
            ]

        if cutlass.const_expr(self.stage):
            self.shared_storage = StagedStorage
        else:
            self.shared_storage = SharedStorage
        self.kernel(x_addr, w1_addr, w2_addr, a_addr, num_tokens).launch(
            grid=[self.ctas, 1, 1],
            block=[THREADS, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=True,
            preferred_smem_carveout=self.carveout,
        )

    @cute.kernel
    def kernel(
        self, x_addr: Int64, w1_addr: Int64, w2_addr: Int64, a_addr: Int64, num_tokens: Int32
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = Int32(tidx) // 32
        lane = Int32(tidx) % 32
        g = lane // 4
        t = lane % 4
        c = Int32(bidx)
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        part = Int32(storage.part.data_ptr().toint())
        if cutlass.const_expr(self.stage):
            # Pre-wait: each warp bulk-copies its K range of the CTA's 2 * cols weight rows (immutable) into the
            # padded image; row r < cols is gate row c * cols + r, row cols + r is up row INTER + c * cols + r.
            mbar = storage.mbar.data_ptr()
            wsm = Int32(storage.wsm.data_ptr().toint())
            if tidx == 0:
                for w in cutlass.range_constexpr(WARPS):
                    cute.arch.mbarrier_init(mbar + w, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            if lane == 0:
                bar = Int32((mbar + warp).toint())
                wptx.mbar_arrive_expect_tx(bar, Int32(2 * self.cols * SLICE1))
                for r in cutlass.range_constexpr(2 * self.cols):
                    src_row = c * self.cols + r
                    if cutlass.const_expr(r >= self.cols):
                        src_row = INTER + c * self.cols + (r - self.cols)
                    wptx.bulk_g2s(
                        wsm + r * ROWP1 + warp * SLICE1,
                        w1_addr + Int64(src_row) * W1_ROW_BYTES + Int64(warp * SLICE1),
                        Int32(SLICE1),
                        bar,
                    )

        if cutlass.const_expr(self.prefetch & (PF_W1 | PF_W2_SLICE)):
            if tidx == 0:
                if cutlass.const_expr(self.prefetch & PF_W1):
                    wptx.prefetch_l2_bulk(
                        w1_addr + Int64(c * self.cols) * W1_ROW_BYTES,
                        Int32(self.cols * W1_ROW_BYTES),
                    )
                    wptx.prefetch_l2_bulk(
                        w1_addr + Int64(INTER + c * self.cols) * W1_ROW_BYTES,
                        Int32(self.cols * W1_ROW_BYTES),
                    )
                if cutlass.const_expr(self.prefetch & PF_W2_SLICE):
                    rows2 = HIDDEN // self.ctas
                    wptx.prefetch_l2_bulk(
                        w2_addr + Int64(c * rows2) * W2_ROW_BYTES, Int32(rows2 * W2_ROW_BYTES)
                    )
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.trigger == "early"):
            cute.arch.griddepcontrol_launch_dependents()

        xrow = g
        if xrow >= num_tokens:
            xrow = num_tokens - 1
        xbase = x_addr + Int64(xrow * HIDDEN + warp * (HIDDEN // WARPS) + 8 * t) * 2
        wbase = w1_addr + Int64((c * self.cols + g) * HIDDEN + warp * (HIDDEN // WARPS) + 8 * t) * 2
        sbase = Int32(0)
        if cutlass.const_expr(self.stage):
            sbase = wsm + g * ROWP1 + warp * SLICE1 + 16 * t
            cute.arch.mbarrier_wait(mbar + warp, 0)
        acc = cute.make_rmem_tensor((self.tiles * 4,), Float32)
        for i in cutlass.range_constexpr(self.tiles * 4):
            acc[i] = Float32(0.0)
        xv = cute.make_rmem_tensor((self.unroll * 4,), Uint32)
        wv = cute.make_rmem_tensor((self.unroll * self.tiles * 8,), Uint32)
        q0 = Int32(0)
        while q0 < self.chunks:
            for u in cutlass.range_constexpr(self.unroll):
                off = Int64((q0 + u) * 64)
                v = wptx.ld_v4_u32(xbase + off)
                for i in cutlass.range_constexpr(4):
                    xv[u * 4 + i] = v[i]
                for j in cutlass.range_constexpr(self.tiles):
                    if cutlass.const_expr(self.stage):
                        soff = Int32((q0 + u) * 64)
                        rg = wptx.lds_v4_u32(sbase + j * 8 * ROWP1 + soff)
                        ru = wptx.lds_v4_u32(sbase + (self.cols + j * 8) * ROWP1 + soff)
                    else:
                        rg = wptx.ld_nc_v4_u32(wbase + Int64(j * 8 * W1_ROW_BYTES) + off)
                        ru = wptx.ld_nc_v4_u32(wbase + Int64((INTER + j * 8) * W1_ROW_BYTES) + off)
                    for i in cutlass.range_constexpr(4):
                        wv[(u * self.tiles + j) * 8 + i] = rg[i]
                        wv[(u * self.tiles + j) * 8 + 4 + i] = ru[i]
            for u in cutlass.range_constexpr(self.unroll):
                for j in cutlass.range_constexpr(self.tiles):
                    b = (u * self.tiles + j) * 8
                    for h in cutlass.range_constexpr(2):
                        d = ptx.mma_bf16_16816(
                            wv[b + 2 * h],
                            wv[b + 4 + 2 * h],
                            wv[b + 2 * h + 1],
                            wv[b + 4 + 2 * h + 1],
                            xv[u * 4 + 2 * h],
                            xv[u * 4 + 2 * h + 1],
                            acc[j * 4],
                            acc[j * 4 + 1],
                            acc[j * 4 + 2],
                            acc[j * 4 + 3],
                        )
                        for i in cutlass.range_constexpr(4):
                            acc[j * 4 + i] = d[i]
            q0 = q0 + self.unroll

        for j in cutlass.range_constexpr(self.tiles):
            for i in cutlass.range_constexpr(4):
                wptx.sts_u32(
                    part + (((warp * self.tiles + j) * 32 + lane) * 4 + i) * 4,
                    wptx.f32_bits(acc[j * 4 + i]),
                )
        cute.arch.sync_threads()
        if warp < self.tiles:
            j = warp
            # warp order 0..7, starting from warp 0's partial (no +0.0 term)
            s = cute.make_rmem_tensor((4,), Float32)
            for i in cutlass.range_constexpr(4):
                s[i] = wptx.bits_f32(wptx.lds_u32(part + ((j * 32 + lane) * 4 + i) * 4))
            for w in cutlass.range_constexpr(WARPS - 1):
                for i in cutlass.range_constexpr(4):
                    s[i] = s[i] + wptx.bits_f32(
                        wptx.lds_u32(part + ((((w + 1) * self.tiles + j) * 32 + lane) * 4 + i) * 4)
                    )
            col = c * self.cols + j * 8 + g
            for h in cutlass.range_constexpr(2):
                tok = 2 * t + h
                gate = wptx.bf16_round(s[h])
                up = wptx.bf16_round(s[2 + h])
                av = ptx.silu_mul_triton(gate, up, self.silu)
                if tok < num_tokens:
                    ptx.st_u16(a_addr + Int64(tok * INTER + col) * 2, wptx.bf16_bits(av))
        if cutlass.const_expr(self.trigger == "end"):
            cute.arch.griddepcontrol_launch_dependents()


class SharedExpertTwoKernelDown:
    """down GEMM (see the module docstring)."""

    def __init__(self, ctas: int = 48, prefetch: int = 0, carveout: int = 100, stage: bool = False):
        assert ctas in (24, 48), ctas
        self.ctas = ctas
        self.feats = HIDDEN // ctas
        self.tpw = self.feats // (WARPS * 16)
        self.prefetch = prefetch
        self.carveout = carveout
        self.stage = stage

    @cute.jit
    def run(self, a_addr: Int64, w2_addr: Int64, y_addr: Int64, num_tokens: Int32, stream):
        if cutlass.const_expr(self.stage):

            @cute.struct
            class StagedStorage:
                mbar: cute.struct.MemRange[cutlass.Int64, WARPS]
                wsm: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Int64, self.feats * ROWP2 // 8], 128
                ]

            self.shared_storage = StagedStorage
            self.kernel(a_addr, w2_addr, y_addr, num_tokens).launch(
                grid=[self.ctas, 1, 1],
                block=[THREADS, 1, 1],
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=True,
                preferred_smem_carveout=self.carveout,
            )
        else:
            self.kernel(a_addr, w2_addr, y_addr, num_tokens).launch(
                grid=[self.ctas, 1, 1],
                block=[THREADS, 1, 1],
                stream=stream,
                use_pdl=True,
                preferred_smem_carveout=self.carveout,
            )

    @cute.kernel
    def kernel(self, a_addr: Int64, w2_addr: Int64, y_addr: Int64, num_tokens: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = Int32(tidx) // 32
        lane = Int32(tidx) % 32
        g = lane // 4
        t = lane % 4
        c = Int32(bidx)
        rows_w = 16 * self.tpw
        if cutlass.const_expr(self.stage):
            # Pre-wait: each warp bulk-copies its 16 * tpw down rows (immutable) into the padded image.
            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            mbar = storage.mbar.data_ptr()
            wsm = Int32(storage.wsm.data_ptr().toint())
            if tidx == 0:
                for w in cutlass.range_constexpr(WARPS):
                    cute.arch.mbarrier_init(mbar + w, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            if lane == 0:
                bar = Int32((mbar + warp).toint())
                wptx.mbar_arrive_expect_tx(bar, Int32(rows_w * W2_ROW_BYTES))
                for r in cutlass.range_constexpr(rows_w):
                    wptx.bulk_g2s(
                        wsm + (warp * rows_w + r) * ROWP2,
                        w2_addr + Int64(c * self.feats + warp * rows_w + r) * W2_ROW_BYTES,
                        Int32(W2_ROW_BYTES),
                        bar,
                    )
        if cutlass.const_expr(self.prefetch & PF_W2_OWN):
            if tidx == 0:
                wptx.prefetch_l2_bulk(
                    w2_addr + Int64(c * self.feats) * W2_ROW_BYTES, Int32(self.feats * W2_ROW_BYTES)
                )
        cute.arch.griddepcontrol_wait()

        arow = g
        if arow >= num_tokens:
            arow = num_tokens - 1
        abase = a_addr + Int64(arow * INTER + 8 * t) * 2
        f0 = c * self.feats + warp * (16 * self.tpw)
        wbase = w2_addr + Int64((f0 + g) * INTER + 8 * t) * 2
        sbase = Int32(0)
        if cutlass.const_expr(self.stage):
            sbase = wsm + (warp * rows_w + g) * ROWP2 + 16 * t
            cute.arch.mbarrier_wait(mbar + warp, 0)
        acc = cute.make_rmem_tensor((self.tpw * 4,), Float32)
        for i in cutlass.range_constexpr(self.tpw * 4):
            acc[i] = Float32(0.0)
        half = 8 // self.tpw
        av = cute.make_rmem_tensor((half * 4,), Uint32)
        wv = cute.make_rmem_tensor((half * self.tpw * 8,), Uint32)
        for p in cutlass.range_constexpr(self.tpw):
            for u in cutlass.range_constexpr(half):
                off = Int64((p * half + u) * 64)
                v = wptx.ld_v4_u32(abase + off)
                for i in cutlass.range_constexpr(4):
                    av[u * 4 + i] = v[i]
                for j in cutlass.range_constexpr(self.tpw):
                    if cutlass.const_expr(self.stage):
                        soff = Int32((p * half + u) * 64)
                        r0 = wptx.lds_v4_u32(sbase + j * 16 * ROWP2 + soff)
                        r1 = wptx.lds_v4_u32(sbase + (j * 16 + 8) * ROWP2 + soff)
                    else:
                        r0 = wptx.ld_nc_v4_u32(wbase + Int64(j * 16 * W2_ROW_BYTES) + off)
                        r1 = wptx.ld_nc_v4_u32(wbase + Int64((j * 16 + 8) * W2_ROW_BYTES) + off)
                    for i in cutlass.range_constexpr(4):
                        wv[(u * self.tpw + j) * 8 + i] = r0[i]
                        wv[(u * self.tpw + j) * 8 + 4 + i] = r1[i]
            for u in cutlass.range_constexpr(half):
                for j in cutlass.range_constexpr(self.tpw):
                    b = (u * self.tpw + j) * 8
                    for h in cutlass.range_constexpr(2):
                        d = ptx.mma_bf16_16816(
                            wv[b + 2 * h],
                            wv[b + 4 + 2 * h],
                            wv[b + 2 * h + 1],
                            wv[b + 4 + 2 * h + 1],
                            av[u * 4 + 2 * h],
                            av[u * 4 + 2 * h + 1],
                            acc[j * 4],
                            acc[j * 4 + 1],
                            acc[j * 4 + 2],
                            acc[j * 4 + 3],
                        )
                        for i in cutlass.range_constexpr(4):
                            acc[j * 4 + i] = d[i]
        for j in cutlass.range_constexpr(self.tpw):
            for h in cutlass.range_constexpr(2):
                tok = 2 * t + h
                if tok < num_tokens:
                    row = y_addr + Int64(tok * HIDDEN + f0 + j * 16 + g) * 2
                    ptx.st_u16(row, wptx.bf16_bits(acc[j * 4 + h]))
                    ptx.st_u16(row + 16, wptx.bf16_bits(acc[j * 4 + 2 + h]))
        cute.arch.griddepcontrol_launch_dependents()
