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
"""Kimi K3 latent all-reduce for decode (M <= 8 tokens) as the consumer of k3_moe's pushed partials.

The push-only k3_moe (``trtllm::k3_fused_moe_push`` / ``trtllm::k3_fused_moe_front_push``) stores every rank's routed
partial through the multicast mapping into slot [rank] of half ``flags[0] & 1`` of every rank's buffer, int32
``[2][8][world][1792]`` (bf16 pairs, ``0x80000000`` not written, -0.0 stored as +0.0), and writes no flag. This kernel
sums the slots of the live rows in the MNNVL one-shot's order (``reduceOneshotLamport``: fp32 over chunks of 8 ranks
in rank order, each from 0, the chunks added in order, then bf16), so its row equals the one-shot all-reduce of the
partials bit for bit. It owns the protocol's state: ``flags[0]`` counts its calls (the half), ``flags[2]`` counts the
CTAs of the running call in.

Grid M x CTAS, 448 / CTAS threads, one 16-byte vector of a token's 3584-wide row per thread. Before the grid-dependency
wait thread 0 reads the count and counts its CTA in: the predecessor (the push) read the same count after its own
wait, and this kernel's previous call ended before that. After the wait every thread loads its vector from all slots
in one pass, repeated until no word is empty, sums, stores the row and empties the words it read (the next push into
this half comes two calls later, after this grid). CTA 0's thread 0 advances the count once every CTA is in.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

HIDDEN_SIZE = 3584
ROW_WORDS = HIDDEN_SIZE // 2  # int32 words (bf16 pairs) of one row
ROW_VECS = HIDDEN_SIZE // 8  # 16-byte vectors of one row
MAX_TOKENS = 8
BUFFERS = 2
RANK_CHUNK = 8
EMPTY_WORD = -(2**31)  # 0x80000000
FLAG_COUNT = 0
FLAG_ARRIVED = 2


def buffer_words(world: int) -> int:
    """Int32 words of one rank's buffer (both halves)."""
    return BUFFERS * MAX_TOKENS * world * ROW_WORDS


@dsl_user_op
def _pack_bf16x2(hi, lo, *, loc=None, ip=None):
    """(bf16(hi) << 16) | bf16(lo), round to nearest even."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [hi.ir_value(loc=loc, ip=ip), lo.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _red_add_release(addr_i64, val, *, loc=None, ip=None):
    """red.release.gpu.global.add.u32: this thread's earlier reads are performed before the add."""
    _llvm.inline_asm(
        None, [addr_i64.ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _ld_acquire_gpu(addr_i64, *, loc=None, ip=None):
    """ld.acquire.gpu.global.u32."""
    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip)], "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l",
            has_side_effects=True, is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _lo(word):
    return (word << cutlass.Int32(16)).bitcast(cutlass.Float32)


def _hi(word):
    return (word & cutlass.Int32(-65536)).bitcast(cutlass.Float32)


@cute.kernel
def k3_latent_reduce_kernel(
    buf: cutlass.Array,  # int32 words of this rank's buffer, [2][8][world][1792]
    flags: cutlass.Array,  # int32 [4]: [0] the call count, [2] the CTAs of this call counted in
    out: cutlass.Array,  # int32 view of the bf16 output [M, 3584]: [M * 1792]
    world: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    tok, cta, _ = cute.arch.block_idx()
    n_tok, n_cta, _ = cute.arch.grid_dim()
    s_count = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    if tidx == 0:
        count = cutlass.Int32(flags.load(idx=FLAG_COUNT, is_volatile=True))
        s_count.store(count, idx=0)
        # Released after the read: CTA 0 advances the count only once every CTA has read it.
        _red_add_release(flags.data_ptr(FLAG_ARRIVED).toint(), cutlass.Int32(1))
    cute.arch.barrier()
    count = s_count.load(idx=0)

    prims.griddepcontrol(prims.GridDepAction.WAIT)
    # The dependents read this grid's output only after their own grid-dependency wait.
    prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)

    vec = cta * cutlass.Int32(threads) + tidx
    base = (
        ((count & cutlass.Int32(1)) * cutlass.Int32(MAX_TOKENS) + tok) * cutlass.Int32(world)
    ) * cutlass.Int32(ROW_WORDS) + vec * cutlass.Int32(4)
    a0 = cutlass.Float32(0.0)
    a1 = cutlass.Float32(0.0)
    a2 = cutlass.Float32(0.0)
    a3 = cutlass.Float32(0.0)
    a4 = cutlass.Float32(0.0)
    a5 = cutlass.Float32(0.0)
    a6 = cutlass.Float32(0.0)
    a7 = cutlass.Float32(0.0)
    pending = cutlass.Boolean(True)
    while pending:
        dirty = cutlass.Boolean(False)
        total = [cutlass.Float32(0.0)] * 8
        for rb in cutlass.range_constexpr(0, world, RANK_CHUNK):
            chunk = [cutlass.Float32(0.0)] * 8
            for rr in cutlass.range_constexpr(min(RANK_CHUNK, world - rb)):
                v = buf.load(idx=base + cutlass.Int32((rb + rr) * ROW_WORDS), vector_size=4, alignment=16,
                             is_volatile=True)  # fmt: skip
                for q in cutlass.range_constexpr(4):
                    word = cutlass.Int32(v[q])
                    dirty = dirty | (word == cutlass.Int32(EMPTY_WORD))
                    chunk[2 * q] = chunk[2 * q] + _lo(word)
                    chunk[2 * q + 1] = chunk[2 * q + 1] + _hi(word)
            for e in cutlass.range_constexpr(8):
                total[e] = total[e] + chunk[e]
        a0, a1, a2, a3, a4, a5, a6, a7 = total
        pending = dirty

    out.store(
        (_pack_bf16x2(a1, a0), _pack_bf16x2(a3, a2), _pack_bf16x2(a5, a4), _pack_bf16x2(a7, a6)),
        idx=tok * cutlass.Int32(ROW_WORDS) + vec * cutlass.Int32(4),
        alignment=16,
    )
    empty = cutlass.Int32(EMPTY_WORD)
    for r in cutlass.range_constexpr(world):
        buf.store(
            (empty, empty, empty, empty), idx=base + cutlass.Int32(r * ROW_WORDS), alignment=16
        )

    if (tok == 0) & (cta == 0) & (tidx == 0):
        everyone = n_tok * n_cta
        arrived = _ld_acquire_gpu(flags.data_ptr(FLAG_ARRIVED).toint())
        while arrived < cutlass.Int32(everyone):
            arrived = _ld_acquire_gpu(flags.data_ptr(FLAG_ARRIVED).toint())
        flags.store(cutlass.Int32(0), idx=FLAG_ARRIVED)
        flags.store(count + cutlass.Int32(1), idx=FLAG_COUNT)


@cute.jit
def k3_latent_reduce(
    buf: cute.Tensor,
    flags: cute.Tensor,
    out: cute.Tensor,
    num_tokens: cutlass.Int32,
    world: cutlass.Constexpr[int],
    ctas: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    threads = ROW_VECS // ctas
    k3_latent_reduce_kernel(buf, flags, out, world, threads).launch(
        grid=[num_tokens, ctas, 1],
        block=[threads, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )
