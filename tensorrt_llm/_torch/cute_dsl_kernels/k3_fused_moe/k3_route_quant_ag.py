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
"""Kimi K3 MoE head all-gather + routing + MXFP8 input quantization for decode, in CuTe DSL.

The MoE head is row-sharded over the TP group of W ranks: rank r's GEMV gives fp32
``[M, H/W + E/W]``, the latent down's columns ``[r*H/W, (r+1)*H/W)`` followed by the router
logits of experts ``[r*E/W, (r+1)*E/W)``. This kernel gathers every rank's slice and returns
what ``trtllm::k3_route_quant`` returns for the gathered logits and latent (top-16 ids, bf16
weights, the MXFP8 latent, its UE8M0 scales), bit for bit what ``mnnvl_allgather_split``
followed by ``k3_route_quant`` give: the pushed values are rounded and sanitized as that
all-gather does (latent to bf16, -0.0 to +0.0), and the routing and quantization are
``k3_route_quant``'s device code.

Grid 2M CTAs of 448 threads:
- CTA t < M pushes this rank's 14 logit vectors of token t, through the multicast mapping,
  into every rank's Lamport buffer, polls token t's logit vectors of all W ranks, and routes
  token t (keys in shared memory, warp 0 selects).
- CTA M + t pushes this rank's 28 latent vectors of token t (8 bf16 each), polls token t's
  latent vectors of all ranks (one per thread: the thread's 8-element quantization vector)
  and quantizes them.
Every reader writes the empty word (0x80000000) back over what it read. Two buffers alternate
between calls: flags[0] holds the buffer of the next call, flipped by the grid's last CTA to
count itself in flags[1].

With ``publish``, each CTA also releases a per-token ready word once its outputs are written
(ready[t] = epoch + 1 for the ids and weights, ready[8 + t] for the MXFP8 row, the epoch being
flags[2]), so that k3_moe (built with head_flags, which advances flags[2]) can acquire them
instead of waiting for this grid.

Buffer words: ``[buffer][token < 8][rank < W][latent 4*H/(8W) words | logits 4*E/(4W) words]``.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
from cutlass import dsl_user_op
from cutlass.experimental import primitives as prims

from tensorrt_llm._torch.cute_dsl_kernels.k3_route_quant import k3_route_quant_kernel as _rq

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN_SIZE = 3584
SF_VEC_SIZE = 32
THREADS = HIDDEN_SIZE // 8  # one 8-element vector per thread in a quantization CTA
M_MAX = 8
BUFFERS = 2
EMPTY_WORD = -(2**31)  # 0x80000000: fp32 -0.0, never a pushed word


def slot_words(world: int) -> int:
    """Int32 words of one (token, rank) slot: the latent's 8-element vectors, then the logits'."""
    return (HIDDEN_SIZE // world // 8 + NUM_EXPERTS // world // 4) * 4


def buffer_words(world: int) -> int:
    return BUFFERS * M_MAX * world * slot_words(world)


FRONT_SPLIT = 8  # k3_moe_front's cluster size: its router rows reach the route CTAs as this many split-K partials


def partial_words() -> int:
    """Int32 words after the buffers: k3_moe_front's router logits as split-K partials,
    [buffer][token][rank][cluster rank][expert of the rank] (the same size for every world)."""
    return BUFFERS * M_MAX * FRONT_SPLIT * NUM_EXPERTS


def workspace_words(world: int) -> int:
    """The head workspace: the all-gather's buffers, then the front's router partials."""
    return buffer_words(world) + partial_words()


@dsl_user_op
def _atomic_add_acq_rel(addr_i64, val, *, loc=None, ip=None):
    """atom.acq_rel.gpu.global.add.u32, returning the old value."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r", has_side_effects=True,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


@dsl_user_op
def _store_release(addr_i64, val, *, loc=None, ip=None):
    """st.release.gpu.global.u32."""
    from cutlass._mlir.dialects import llvm as _llvm

    _llvm.inline_asm(
        None, [addr_i64.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.u32 [$0], $1;", "l,r", has_side_effects=True,
        is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _pack_bf16x2(hi, lo, *, loc=None, ip=None):
    """(bf16(hi) << 16) | bf16(lo), round to nearest even (``__floats2bfloat162_rn``)."""
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass._mlir.extras import types as _T

    return cutlass.Int32(
        _llvm.inline_asm(
            _T.i32(), [hi.ir_value(loc=loc, ip=ip), lo.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False,
            is_align_stack=False, asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
        )
    )  # fmt: skip


def _sanitize_bf16x2(word):
    """-0.0 halves become +0.0 (the all-gather's sanitizeBf16Pair), so no word is 0x80000000."""
    word = cutlass.Int32(
        cutlass.select_(
            (word & cutlass.Int32(0xFFFF)) == cutlass.Int32(0x8000),
            word & cutlass.Int32(-65536),
            word,
        )
    )
    return cutlass.Int32(
        cutlass.select_(
            (word & cutlass.Int32(-65536)) == cutlass.Int32(-2147483648),
            word & cutlass.Int32(0xFFFF),
            word,
        )
    )


def _sanitize_f32(word):
    return cutlass.Int32(cutlass.select_(word == cutlass.Int32(EMPTY_WORD), cutlass.Int32(0), word))


@cute.jit
def _poll4(arr, idx):
    """Spin (no back-off) until none of the 4 words at idx is empty; returns them."""
    w0 = cutlass.Int32(EMPTY_WORD)
    w1 = cutlass.Int32(EMPTY_WORD)
    w2 = cutlass.Int32(EMPTY_WORD)
    w3 = cutlass.Int32(EMPTY_WORD)
    pending = cutlass.Boolean(True)
    while pending:
        v = arr.load(idx=idx, vector_size=4, alignment=16, is_volatile=True)
        w0 = cutlass.Int32(v[0])
        w1 = cutlass.Int32(v[1])
        w2 = cutlass.Int32(v[2])
        w3 = cutlass.Int32(v[3])
        empty = cutlass.Int32(EMPTY_WORD)
        pending = (w0 == empty) | (w1 == empty) | (w2 == empty) | (w3 == empty)
    return w0, w1, w2, w3


@cute.kernel
def k3_route_quant_ag_kernel(
    head_words: cutlass.Array,  # int32 view of this rank's fp32 head [M, WL + WE]
    bias: cutlass.Array,  # fp32 [896]
    buf_uc: cutlass.Array,  # int32 words of this rank's Lamport buffers
    buf_mc: cutlass.Array,  # int32 words of their multicast mapping
    flags: cutlass.Array,  # int32 [0] buffer of this call, [1] CTAs of this call counted so far
    topk_ids: cutlass.Array,  # int32 [M * 16]
    topk_weight_bits: cutlass.Array,  # int16 view of bf16 [M * 16]
    quant_words: cutlass.Array,  # int32 view of e4m3 [M, 3584]: [M * 896]
    scales: cutlass.Array,  # uint8 [M * 112]
    ready: cutlass.Array,  # int32 [16] per-token ready words (publish)
    num_tokens: cutlass.Int32,
    rank: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    world: cutlass.Constexpr[int],
    early_trigger: cutlass.Constexpr[bool],
    publish: cutlass.Constexpr[bool],
):
    WL = HIDDEN_SIZE // world  # latent columns per rank
    WE = NUM_EXPERTS // world  # logits per rank
    LV = WL // 8  # latent vectors per slot
    EV = WE // 4  # logit vectors per slot
    SLOT = (LV + EV) * 4
    HEAD_ROW = WL + WE  # fp32 words per head row
    BUF = M_MAX * world * SLOT
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    s_key = cutlass.Array(cutlass.Int32, NUM_EXPERTS, space=cutlass.AddressSpace.smem, alignment=16)
    s_sigmoid = cutlass.Array(
        cutlass.Float32, NUM_EXPERTS, space=cutlass.AddressSpace.smem, alignment=16
    )
    s_buf = cutlass.Array(cutlass.Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    routes = bidx < num_tokens
    tok = cutlass.select_(routes, bidx, bidx - num_tokens)

    # The bias is a weight: this thread's 4 experts (the gathered logits are rank-major, so
    # route thread i polls experts 4i .. 4i+3), read before the grid dependency.
    polls_logits = tidx < cutlass.Int32(NUM_EXPERTS // 4)
    e0 = cutlass.select_(polls_logits, tidx * cutlass.Int32(4), cutlass.Int32(0))
    bias4 = bias.load(idx=e0, vector_size=4, alignment=16)

    prims.griddepcontrol(prims.GridDepAction.WAIT)
    if tidx == 0:
        s_buf.store(flags.load(idx=0, is_volatile=True), idx=0)
        if cutlass.const_expr(publish):
            s_buf.store(flags.load(idx=2, is_volatile=True), idx=1)
    cute.arch.barrier()
    b = s_buf.load(idx=0)
    epoch = s_buf.load(idx=1)
    if cutlass.const_expr(early_trigger):
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    # Push this rank's slot of token tok: logits (route CTAs) or latent (quantization CTAs).
    slot_base = b * BUF + (tok * world + rank) * SLOT
    head_row = tok * HEAD_ROW
    if routes:
        if tidx < EV:
            w = head_words.load(idx=head_row + WL + tidx * 4, vector_size=4, alignment=16)
            buf_mc.store(
                (
                    _sanitize_f32(cutlass.Int32(w[0])),
                    _sanitize_f32(cutlass.Int32(w[1])),
                    _sanitize_f32(cutlass.Int32(w[2])),
                    _sanitize_f32(cutlass.Int32(w[3])),
                ),
                idx=slot_base + LV * 4 + tidx * 4,
                alignment=16,
            )
            # Drain the posted multicast store now: without a fence it can sit in the SM's write path
            # until this thread's next release, and the polls below have none (a cluster-scope fence
            # does not wait for the remote ranks' acknowledgement).
            prims.fence_acq_rel(prims.MemScope.CLUSTER)
    else:
        if tidx < LV:
            lo = head_words.load(idx=head_row + tidx * 8, vector_size=4, alignment=16)
            hi = head_words.load(idx=head_row + tidx * 8 + 4, vector_size=4, alignment=16)
            f = [cutlass.Int32(lo[q]).bitcast(cutlass.Float32) for q in range(4)]
            f += [cutlass.Int32(hi[q]).bitcast(cutlass.Float32) for q in range(4)]
            buf_mc.store(
                (
                    _sanitize_bf16x2(_pack_bf16x2(f[1], f[0])),
                    _sanitize_bf16x2(_pack_bf16x2(f[3], f[2])),
                    _sanitize_bf16x2(_pack_bf16x2(f[5], f[4])),
                    _sanitize_bf16x2(_pack_bf16x2(f[7], f[6])),
                ),
                idx=slot_base + tidx * 4,
                alignment=16,
            )
            prims.fence_acq_rel(prims.MemScope.CLUSTER)

    # Every CTA has read the buffer index before counting itself (a thread of the last warp, which
    # never pushes): the grid's last one hands the other buffer to the next call, which reads it
    # after its own grid-dependency wait.
    if tidx == cutlass.Int32(THREADS - 32):
        arrived = _atomic_add_acq_rel(flags.data_ptr(1).toint(), cutlass.Int32(1))
        if arrived == num_tokens * cutlass.Int32(2) - cutlass.Int32(1):
            flags.store(cutlass.Int32(0), idx=1)
            flags.store(b ^ cutlass.Int32(1), idx=0)

    tok_base = b * BUF + tok * world * SLOT
    empty = cutlass.Int32(EMPTY_WORD)
    if routes:
        # Poll every rank's logit vectors of this token (one per thread), keys and sigmoids into
        # shared memory, empty what was read.
        if polls_logits:
            addr = tok_base + (tidx // EV) * SLOT + LV * 4 + (tidx % EV) * 4
            w0, w1, w2, w3 = _poll4(buf_uc, addr)
            buf_uc.store((empty, empty, empty, empty), idx=addr, alignment=16)
            words = [w0, w1, w2, w3]
            for q in cutlass.range_constexpr(4):
                sig = _rq.sigmoid_accurate(words[q].bitcast(cutlass.Float32))
                s_sigmoid.store(sig, idx=e0 + q)
                s_key.store(_rq.selection_key(sig + cutlass.Float32(bias4[q])), idx=e0 + q)
        cute.arch.barrier()
        if tidx < cutlass.Int32(32):
            expert, weight_bits = _rq.top16_warp(s_key, s_sigmoid, tidx, routed_scaling_factor)
            if tidx < cutlass.Int32(TOP_K):
                out = tok * cutlass.Int32(TOP_K) + tidx
                topk_ids.store(expert, idx=out)
                topk_weight_bits.store(weight_bits, idx=out)
                if cutlass.const_expr(publish):
                    cute.arch.fence_acq_rel_gpu()
            if cutlass.const_expr(publish):
                cute.arch.sync_warp()
                if tidx == cutlass.Int32(0):
                    _store_release(ready.data_ptr(tok).toint(), epoch + cutlass.Int32(1))
    else:
        # Poll this thread's latent vector (rank tidx // LV, columns 8 * tidx ..), quantize it with
        # its 4-lane scale group, empty what was read.
        addr = tok_base + (tidx // LV) * SLOT + (tidx % LV) * 4
        w0, w1, w2, w3 = _poll4(buf_uc, addr)
        buf_uc.store((empty, empty, empty, empty), idx=addr, alignment=16)
        q_lo, q_hi, sf_byte = _rq.mxfp8_quant_vec8([w0, w1, w2, w3])
        quant_words.store(
            (q_lo, q_hi),
            idx=tok * cutlass.Int32(HIDDEN_SIZE // 4) + tidx * cutlass.Int32(2),
            alignment=8,
        )
        if tidx % cutlass.Int32(SF_VEC_SIZE // 8) == cutlass.Int32(0):
            scales.store(
                cutlass.Uint8(sf_byte),
                idx=tok * cutlass.Int32(HIDDEN_SIZE // SF_VEC_SIZE)
                + tidx // cutlass.Int32(SF_VEC_SIZE // 8),
            )
        if cutlass.const_expr(publish):
            # The row is read by k3_moe's TMA gathers (async proxy).
            prims.fence_proxy("async_global")
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier()
            if tidx == cutlass.Int32(0):
                _store_release(
                    ready.data_ptr(tok + cutlass.Int32(M_MAX)).toint(), epoch + cutlass.Int32(1)
                )

    if cutlass.const_expr(not early_trigger):
        cute.arch.fence_acq_rel_gpu()
        cute.arch.barrier()
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)


@cute.jit
def k3_route_quant_ag(
    head_words: cute.Tensor,
    bias: cute.Tensor,
    buf_uc: cute.Tensor,
    buf_mc: cute.Tensor,
    flags: cute.Tensor,
    topk_ids: cute.Tensor,
    topk_weight_bits: cute.Tensor,
    quant_words: cute.Tensor,
    scales: cute.Tensor,
    ready: cute.Tensor,
    num_tokens: cutlass.Int32,
    rank: cutlass.Int32,
    routed_scaling_factor: cutlass.Float64,
    world: cutlass.Constexpr[int],
    early_trigger: cutlass.Constexpr[bool],
    publish: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda_driver.CUstream,
) -> None:
    k3_route_quant_ag_kernel(
        head_words, bias, buf_uc, buf_mc, flags, topk_ids, topk_weight_bits, quant_words, scales, ready,
        num_tokens, rank, routed_scaling_factor, world, early_trigger, publish,
    ).launch(
        grid=[num_tokens * 2, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=use_pdl,
    )  # fmt: skip
