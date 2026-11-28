# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP MoE reduce-scatter fused with the next hyper-connection site's injection and
grouped norm, and optionally with the routed experts' finalize: one Lamport launch.

A trunk MoE block of an attention-DP rank ends with the reduce-scatter of its window partials
(`dep_lamport_round_trip`, `reduce=True`: `[W * R, H]` partials -> this rank's `[R, H]` summed rows
`routed`), and the next hyper-connection site starts with `qwen4_exp_hc_inject_norm(hc, routed,
inject, norm_weight, eps, C, shared, gate_logits)`:

    block[t]      = bf16(fma(sigmoid(gate[t]), shared[t], routed[t]))
    hc_out[t, s]  = bf16(fma(inject[t, s], block[t], hc[t, s]))              s = 0 .. C - 1
    normed[t, s]  = bf16(hc_out[t, s] * rstd(t, s) * (1 + norm_weight[s]))
    rstd(t, s)    = rsqrt(sum_i(hc_out[t, s, i]^2) / H + eps)

Here the two are ONE launch: a grid of `(rows, C / SPP)` CTAs of `H / 8` threads, a thread a vector
of 8 bf16 lanes; CTA `(row, c)` holds `SPP` streams of the row (`streams_per_cta`: as many CTAs a
row as stay resident together). The `c == 0` CTAs publish their vector of every peer's slice; every
CTA polls its row's vectors in its own slot as the reduce-scatter kernel does (same slots, same
offsets, same sum order with a bf16 rounding after every add): a rank running this launch and a rank
running `reduce_scatter` exchange the same bytes, so each rank may pick either. Between the publish
and the poll a thread issues the loads of its columns of `hc`, `norm_weight`, `shared`, the row's
gate logit and inject gates. After the receive it combines, injects, takes the CTA-wide sum of
squares of each of its streams and stores `hc_out` and `normed`; `routed` is never stored.

Numerics: the Lamport sum is the reduce-scatter kernel's statements (its helpers, imported). The
combine, the injection and the norm are `qwen4_exp_hc_inject_norm`'s CuTe instructions (its
`_fma` / `_mul` / `_add` / `_sigmoid` / `_rstd` / `_round_bf16`, imported), and the sum of squares
is reduced in the Triton kernel's tree at its 4096 block and eight warps (the form it runs below
`MAX_ROWS + 1` rows): thread `t < 256` there holds columns `[8t, 8t + 8)` and `[2048 + 8t, ...)`;
here thread `256 + u` hands its two lane partials of columns `[2048 + 8u, ...)` to thread `u` through
shared memory before the warp reduction of threads 0..255. That is `_FoldNormKernel`'s tree, which
that op certifies bitwise against the Triton kernel. So `(hc_out, normed)` are bitwise
`reduce_scatter` + `qwen4_exp_hc_inject_norm`.

`fin` (compile-time): the launch reads the trtllm-gen runner's UNFINALIZED output instead of the
partials -- the permuted FC2 rows `g2 [P, H]` bf16, the routing weights `w [W * R, K]` bf16 and the
expanded -> permuted map `e2p [W * R, K]` int32 (-1: a pick outside this rank's window) -- and forms
each partial vector it publishes or keeps with the runner's finalize: an fp32 accumulator from +0
over k = 0 .. K - 1 in order, skipping -1, `fma(w, x, acc)`, one rounding to bf16 (a bf16 x bf16
product is exact in fp32, so the fma and a multiply then an add agree). The runner's finalize launch
and its `[W * R, H]` store and reload are gone; the bytes exchanged, the slots and the sum order
are unchanged. Every CTA forms this rank's own row (its sum needs it); peer ordinal `j` (1 .. W - 1)
goes to CTA `(j - 1) % (C / SPP)` of the row, so each peer vector is published once.

What the kernel reads before its poll: operands whose writers completed when it starts. A plain
launch is no PDL launch, on a stream position behind every writer (`hc` and `inject`: the MoE site's
launches; `shared` / `gate_logits`: the side stream, which the caller joins in front of this
launch; `norm_weight`: static; the partials or the runner's output: the expert chain). A `fin`
launch with `pdl` is a PDL launch whose `griddepcontrol.wait` is its first statement, and its
runner-output loads are coherent, side-effecting loads: ptxas moves `ld.global.nc` above the wait
once anything precedes it, so no `.nc` load is used. Nothing the peers write is read except
through the poll. The launch releases its own PDL dependents (the site's first projection, which
streams its static weight before its own wait) right after its publish.

Slots, by the workspace's protocol (`DepRoundTrip.steps`):
* device counter: the reduce-scatter kernel's acquire load, re-arm pass and last-CTA advance over
  the whole `(rows, C / SPP)` grid.
* host index (`LamportSteps`): the slot is `(state[1] + launch_index) mod 3`, with no acquire load,
  atomic or release store. The `C / SPP` CTAs of a row all read the row's vectors, so no single
  reader can re-arm them: CTA 0 records the slot's vector count with a plain store, and the re-arm
  pass of the next launch on the workspace (which every launch of the workspace runs) clears
  them, as the device-counter form does. The workspace's other host-index launches clean their own
  vectors and record 0. Every receive issues its first probe of every peer before it waits on any.

Collective: every rank of the group launches a reduce-scatter (this one, its `fin` form or the
plain one) at the same point with the same row count. The plain launch publishes from its `c == 0`
CTAs, which come first in launch order, so a grid past its resident wave (`streams_per_cta`'s
second layout) still progresses; every CTA of a `fin` row publishes, so a `fin` launch is served
only where its whole grid is resident at once (`fin_resident`).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.hc import qwen4_exp_hc_inject_norm as _hn
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_round_trip as _rt

_PING_PONG = _rt._PING_PONG
_VEC = _rt._VEC
_ALIGN = _rt._ALIGN
_NEG_ZERO = _rt._NEG_ZERO
#: The Triton inject-norm kernel's eight-warp layout at a hidden in (2048, 4096]: a thread's second
#: chunk starts at this column.
_SPLIT = _hn._CUTE_SPLIT
_SPLIT_T = _SPLIT // _VEC
_RED_W = 16
#: Rows a rank this launch serves: the inject-norm kernel runs its eight-warp form below
#: `_FEW_WARPS_MIN_ROWS` rows, and that form's reduction tree is the one reproduced here.
MAX_ROWS = _hn._FEW_WARPS_MIN_ROWS - 1
#: `(streams a CTA, most CTAs a launch)`, tried in order by `streams_per_cta`; past the last one every
#: stream of a row shares one CTA. GB300 (152 SMs, 320-thread CTAs), us a layer saved against
#: reduce-scatter + inject: 4 CTAs a row pays to 48 rows a rank and equals 2 a row from 48 to 96; past
#: 608 CTAs the launch runs in two waves. dep4 kept `(1, 192)`; dep8 measured `(1, 304)` (4 CTAs a row
#: up to 76 rows) at -0.098 / -0.058 / -0.048 ms a step at 14 / 16 / 18 requests a rank.
LAYOUTS = {4: ((1, 192), (2, 608)), 8: ((1, 304), (2, 608))}
_raw_stream = torch._C._cuda_getCurrentRawStream


def _hidden_hex(hidden: int) -> str:
    """`hidden` as a PTX fp32 immediate (the norm's divisor)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", float(hidden)))[0]


# ------------------------------------------------------------- the finalize's operand loads
# Coherent and side-effecting: a `fin` launch under PDL reads the runner's output only after its
# `griddepcontrol.wait`, which LLVM keeps in front of these (memory-clobbering) statements and ptxas
# keeps in front of coherent loads (it may move `ld.global.nc` across it).
@dsl_user_op
def _ld_s32(addr, *, loc=None, ip=None):
    return cutlass.Int32(
        _hn._asm(
            T.i32(),
            [cutlass.Int64(addr)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _ld_bf16_f32(addr, *, loc=None, ip=None):
    """A bf16 in memory as fp32 (exact)."""
    return cutlass.Float32(
        _hn._asm(
            T.f32(),
            [cutlass.Int64(addr)],
            "{ .reg .b16 h; ld.global.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
            "=f,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _selp_valid(slot, a, b, *, loc=None, ip=None):
    """`a` where the permuted index `slot` is not -1 (a pick in this rank's window), else `b`."""
    return cutlass.Float32(
        _hn._asm(
            T.f32(),
            [cutlass.Int32(slot), cutlass.Float32(a), cutlass.Float32(b)],
            "{ .reg .pred q; setp.ne.s32 q, $1, -1; selp.f32 $0, $2, $3, q; }",
            "=f,r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _valid(slot, *, loc=None, ip=None):
    """1 where the permuted index `slot` is not -1, else 0."""
    return cutlass.Int32(
        _hn._asm(
            T.i32(),
            [cutlass.Int32(slot)],
            "{ .reg .pred q; setp.ne.s32 q, $1, -1; selp.b32 $0, 1, 0, q; }",
            "=r,r",
            loc,
            ip,
        )
    )


def _finalized(g2_base, e2p_base, w_base, g, col, N: int, TOPK: int):
    """The runner's finalize of gathered row `g` (Int64) at columns `[col, col + 8)` as a bf16
    vector (traced inline: loops over constants only). Every load is issued before the chain."""
    slots = [_ld_s32(e2p_base + (g * TOPK + k) * 4) for k in range(TOPK)]
    weights = [_ld_bf16_f32(w_base + (g * TOPK + k) * 2) for k in range(TOPK)]
    rows = [
        _hn._ldg_v4(g2_base + (cutlass.Int64(slots[k]) * N + col) * 2, _valid(slots[k]))
        for k in range(TOPK)
    ]
    acc = [cutlass.Float32(0.0) for _ in range(_VEC)]
    for k in range(TOPK):
        xs = _hn._vals(rows[k])
        for e in range(_VEC):
            acc[e] = _selp_valid(slots[k], _hn._fma(weights[k], xs[e], acc[e]), acc[e])
    out = cutlass.Array(cutlass.BFloat16, _VEC, alignment=_ALIGN)
    for e in range(_VEC):
        out[e] = _hn._round_bf16(acc[e]).to(cutlass.BFloat16)
    return out


def _rearm(local_buffer, offset, neg_zero_vec) -> None:
    """Store the not-ready marker into this rank's slot at `offset` (traced inline)."""
    local_buffer.store(
        neg_zero_vec[0:_VEC, cutlass.align(_ALIGN)],
        offset,
        _VEC,
        alignment=_ALIGN,
        is_volatile=True,
    )


@cute.kernel
def _rs_inject_kernel(
    input_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    signal_state: cute.Tensor,
    launch_index: cutlass.Int32,
    mInj: cute.Tensor,
    mGate: cute.Tensor,
    hc_base: cutlass.Int64,
    w_base: cutlass.Int64,
    sh_base: cutlass.Int64,
    out_base: cutlass.Int64,
    nrm_base: cutlass.Int64,
    rows: cutlass.Int32,
    s_hc: cutlass.Int32,
    s_sh: cutlass.Int32,
    eps: cutlass.Float32,
    g2_base: cutlass.Int64,
    e2p_base: cutlass.Int64,
    fw_base: cutlass.Int64,
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    SUM_ORDER: cutlass.Constexpr[tuple[int, ...]],
    HC: cutlass.Constexpr[int],
    SPP: cutlass.Constexpr[int],
    STEP: cutlass.Constexpr[bool],
    FIN: cutlass.Constexpr[bool],
    TOPK: cutlass.Constexpr[int],
    PDL: cutlass.Constexpr[bool],
):
    """Grid `(rows, HC / SPP)`, N / 8 threads a CTA: CTA `(row, c)` holds streams
    `[c * SPP, (c + 1) * SPP)` of the row."""
    if cutlass.const_expr(PDL):
        # A PDL launch waits for its predecessors before anything else: what it gains is its
        # launch and CTA ramp under the predecessor's tail. Keep it the first statement.
        cute.arch.griddepcontrol_wait()
    tidx, _, _ = cute.arch.thread_idx()
    row, cta, _ = cute.arch.block_idx()
    grid_x, grid_y, _ = cute.arch.grid_dim()
    bdim, _, _ = cute.arch.block_dim()
    # The launch's CTAs in launch order and their count (the last-arrival advance counts them).
    bidx = cta * grid_x + row
    grid_n = grid_x * grid_y

    input_array = cutlass.make_array_view(input_tensor)
    buffer_arrays = [cutlass.make_array_view(buffer) for buffer in buffers]
    local_buffer = buffer_arrays[LOCAL_RANK]
    # Threads [_SPLIT_T, N / 8) hold the columns from _SPLIT on; their lane partials of the sum of
    # squares go to threads [0, N / 8 - _SPLIT_T) through `pb`.
    n_b = N // _VEC - _SPLIT_T
    pb = cutlass.Array(
        cutlass.Float32, 2 * SPP * n_b, space=cutlass.AddressSpace.smem, alignment=_ALIGN
    )
    red = cutlass.Array(
        cutlass.Float32, SPP * _RED_W, space=cutlass.AddressSpace.smem, alignment=_ALIGN
    )

    # ---- the reduce-scatter kernel's statements: thread `tidx` of every CTA of `row` owns vector
    # `row * bdim + tidx` of every slice.
    active_numel = rows * N
    active_numel_i64 = cutlass.Int64(active_numel)
    workspace_numel_i64 = cutlass.Int64(cute.size(buffers[0], mode=[2]))
    state_counter_ptr = (signal_state.iterator + cutlass.Int64(_rt._SIGNAL_COUNTER_INDEX)).llvm_ptr
    state_signal_ptr = (signal_state.iterator + cutlass.Int64(_rt._SIGNAL_VALUE_INDEX)).llvm_ptr
    if cutlass.const_expr(STEP):
        signal_i32 = (
            cute.arch.load(state_signal_ptr, cutlass.Int32, sem="relaxed", scope="gpu")
            + launch_index
        )
    else:
        signal_i32 = cute.arch.load(state_signal_ptr, cutlass.Int32, sem="acquire", scope="gpu")
        # Every thread of this CTA holds the signal before thread 0 counts the CTA as arrived.
        cute.arch.sync_threads()
    signal_i64 = cutlass.Int64(signal_i32)
    ping_slot = signal_i64 % _PING_PONG
    clear_slot = (signal_i64 + cutlass.Int64(_rt._CLEAR_AHEAD)) % _PING_PONG
    ping_base = cutlass.Int64(ping_slot) * cutlass.Int64(WORLD_SIZE) * workspace_numel_i64
    clear_base = cutlass.Int64(clear_slot) * cutlass.Int64(WORLD_SIZE) * workspace_numel_i64

    neg_zero_vec = cutlass.Array(cutlass.BFloat16, _VEC, alignment=_ALIGN)
    neg_zero_bits = cutlass.Int16(_NEG_ZERO)
    for lane in cutlass.range_constexpr(_VEC):
        neg_zero_vec[lane] = neg_zero_bits.bitcast(cutlass.BFloat16)

    # Re-arm the previous launch's slot (`_rt._CLEAR_AHEAD`, the round trip's protocol: every launch
    # of the workspace must re-arm the same slot), for as many vectors as its peers wrote into it:
    # every peer finished writing it before this rank's previous launch completed, and none writes
    # it again before this rank's next launch has published.
    clear_count_ptr = (
        signal_state.iterator
        + cutlass.Int64(_rt._SIGNAL_SLOT_COUNTS_INDEX)
        + cutlass.Int64(clear_slot)
    ).llvm_ptr
    if cutlass.const_expr(STEP):
        clear_vectors = cute.arch.load(clear_count_ptr, cutlass.Int32, sem="relaxed", scope="gpu")
    else:
        clear_vectors = cute.arch.load(clear_count_ptr, cutlass.Int32, sem="acquire", scope="gpu")
    clear_vector_idx = row * bdim + tidx
    clear_stride = grid_x * bdim
    while clear_vector_idx < clear_vectors:
        clear_vector_idx_i64 = cutlass.Int64(clear_vector_idx)
        for peer_ordinal in cutlass.range_constexpr(1, WORLD_SIZE):
            if cta == ((peer_ordinal - 1) % (HC // SPP) if FIN else 0):
                _rearm(
                    local_buffer,
                    clear_base
                    + cutlass.Int64(PEER_ORDER[peer_ordinal]) * workspace_numel_i64
                    + clear_vector_idx_i64 * cutlass.Int64(_VEC),
                    neg_zero_vec,
                )
        clear_vector_idx = clear_vector_idx + clear_stride

    base_idx_i64 = cutlass.Int64((row * bdim + tidx) * _VEC)
    write_offset = ping_base + cutlass.Int64(LOCAL_RANK) * workspace_numel_i64 + base_idx_i64
    if cutlass.const_expr(not FIN):
        if cta == 0:
            # Publish every peer's slice of this rank's partials into that peer's slot.
            for peer_ordinal in cutlass.range_constexpr(WORLD_SIZE):
                target_rank = PEER_ORDER[peer_ordinal]
                if target_rank != LOCAL_RANK:
                    partial = input_array.load(
                        cutlass.Int64(target_rank) * active_numel_i64 + base_idx_i64,
                        _VEC,
                        alignment=_ALIGN,
                    )
                    clean = _rt._sanitize(partial)
                    buffer_arrays[target_rank].store(
                        clean[0:_VEC, cutlass.align(_ALIGN)],
                        write_offset,
                        _VEC,
                        alignment=_ALIGN,
                        is_volatile=True,
                    )
        own = input_array.load(
            cutlass.Int64(LOCAL_RANK) * active_numel_i64 + base_idx_i64, _VEC, alignment=_ALIGN
        )
    else:
        col = cutlass.Int64(tidx * _VEC)
        own = _finalized(
            g2_base,
            e2p_base,
            fw_base,
            cutlass.Int64(LOCAL_RANK) * cutlass.Int64(rows) + cutlass.Int64(row),
            col,
            N,
            TOPK,
        )
        for c in cutlass.range_constexpr(HC // SPP):
            if cta == c:
                for peer_ordinal in cutlass.range_constexpr(1, WORLD_SIZE):
                    if cutlass.const_expr((peer_ordinal - 1) % (HC // SPP) == c):
                        target_rank = PEER_ORDER[peer_ordinal]
                        partial = _finalized(
                            g2_base,
                            e2p_base,
                            fw_base,
                            cutlass.Int64(target_rank) * cutlass.Int64(rows) + cutlass.Int64(row),
                            col,
                            N,
                            TOPK,
                        )
                        clean = _rt._sanitize(partial)
                        buffer_arrays[target_rank].store(
                            clean[0:_VEC, cutlass.align(_ALIGN)],
                            write_offset,
                            _VEC,
                            alignment=_ALIGN,
                            is_volatile=True,
                        )
    parts = cutlass.Array(cutlass.BFloat16, WORLD_SIZE * _VEC, alignment=_ALIGN)
    for lane in cutlass.range_constexpr(_VEC):
        parts[LOCAL_RANK * _VEC + lane] = own[lane]
    # The site's first projection (launched with PDL) may start streaming its weight; every CTA
    # of this grid is resident once every CTA has issued this.
    cute.arch.griddepcontrol_launch_dependents()

    # ---- the site's operands, issued before the poll.
    one = cutlass.Int32(1)
    row64 = cutlass.Int64(row)
    col = tidx * _VEC
    width = HC * N
    xw = []
    ww = []
    gs = []
    for j in cutlass.range_constexpr(SPP):
        stream = cta * SPP + j
        xw.append(_hn._ldg_v4(hc_base + (row64 * s_hc + stream * N) * 2 + col * 2, one))
        ww.append(_hn._ldg_v4(w_base + cutlass.Int64(stream * N) * 2 + col * 2, one))
        gs.append(mInj[row, stream].to(cutlass.Float32))
    shw = _hn._ldg_v4(sh_base + (row64 * s_sh) * 2 + col * 2, one)
    gate_v = mGate[row].to(cutlass.Float32)

    # ---- the receive and the ring's chain: bf16 after every add (the reduce-scatter's).
    first = []
    if cutlass.const_expr(STEP):
        first = _rt._first_probes(
            local_buffer, ping_base, workspace_numel_i64, base_idx_i64, PEER_ORDER, WORLD_SIZE
        )
    for peer_ordinal in cutlass.range_constexpr(1, WORLD_SIZE):
        source_rank = PEER_ORDER[peer_ordinal]
        offset = ping_base + cutlass.Int64(source_rank) * workspace_numel_i64 + base_idx_i64
        if cutlass.const_expr(STEP):
            got = _rt._poll_from(local_buffer, offset, first[peer_ordinal - 1])
        else:
            got = _rt._poll(local_buffer, offset)
        for lane in cutlass.range_constexpr(_VEC):
            parts[source_rank * _VEC + lane] = got[lane]
    routed = []
    for lane in cutlass.range_constexpr(_VEC):
        acc = parts[SUM_ORDER[0] * _VEC + lane]
        for k in cutlass.range_constexpr(1, WORLD_SIZE):
            acc = (
                acc.to(cutlass.Float32) + parts[SUM_ORDER[k] * _VEC + lane].to(cutlass.Float32)
            ).to(cutlass.BFloat16)
        routed.append(acc.to(cutlass.Float32))

    # ---- the combine `fma(sigmoid(gate), shared, routed)` rounded once, then per stream
    # `fma(inject, block, hc)` rounded once into `hc_out` and the lane partials of its sum of
    # squares.
    sg = _hn._sigmoid(gate_v)
    sh = _hn._vals(shw)
    bv = [_hn._round_bf16(_hn._fma(sg, sh[e], routed[e])) for e in range(_VEC)]
    ras = []
    p0s = []
    p1s = []
    for j in cutlass.range_constexpr(SPP):
        stream = cta * SPP + j
        xv = _hn._vals(xw[j])
        ra = [_hn._round_bf16(_hn._fma(gs[j], bv[e], xv[e])) for e in range(_VEC)]
        _hn._stg_v4(
            out_base + (row64 * width + stream * N) * 2 + col * 2,
            one,
            *[_hn._bf16x2(ra[2 * i], ra[2 * i + 1]) for i in range(4)],
        )
        ras.append(ra)
        # Lane L of the bf16 pairs (L = 0 the even columns): fma(a0, a0, a1 * a1) +
        # fma(a2, a2, a3 * a3) over the chunk's four pairs.
        p0 = _hn._add(
            _hn._fma(ra[0], ra[0], _hn._mul(ra[2], ra[2])),
            _hn._fma(ra[4], ra[4], _hn._mul(ra[6], ra[6])),
        )
        p1 = _hn._add(
            _hn._fma(ra[1], ra[1], _hn._mul(ra[3], ra[3])),
            _hn._fma(ra[5], ra[5], _hn._mul(ra[7], ra[7])),
        )
        p0s.append(p0)
        p1s.append(p1)
        if tidx >= _SPLIT_T:
            pb[(2 * j) * n_b + (tidx - _SPLIT_T)] = p0
            pb[(2 * j + 1) * n_b + (tidx - _SPLIT_T)] = p1
    cute.arch.sync_threads()
    warp = tidx // 32
    for j in cutlass.range_constexpr(SPP):
        # The Triton thread's value: (first chunk + second chunk) per lane, then the lanes. A thread
        # without a second chunk adds +0.0 (the Triton kernel's masked columns).
        q0 = cutlass.Float32(0.0)
        q1 = cutlass.Float32(0.0)
        if tidx < n_b:
            q0 = pb[(2 * j) * n_b + tidx]
            q1 = pb[(2 * j + 1) * n_b + tidx]
        v = _hn._add(_hn._add(p0s[j], q0), _hn._add(p1s[j], q1))
        for off in cutlass.range_constexpr(5):
            v = _hn._add(v, cute.arch.shuffle_sync_bfly(v, offset=16 >> off))
        # Warps from _SPLIT_T / 32 on write entries nobody reads.
        if tidx % 32 == 0:
            red[j * _RED_W + warp] = v
    cute.arch.sync_threads()
    for j in cutlass.range_constexpr(SPP):
        stream = cta * SPP + j
        v = red[j * _RED_W + tidx % (_SPLIT_T // 32)]
        for off in cutlass.range_constexpr(3):
            v = _hn._add(v, cute.arch.shuffle_sync_bfly(v, offset=4 >> off))
        rstd = _hn._rstd(v, eps, _hidden_hex(N))
        wv = _hn._vals(ww[j])
        ra = ras[j]
        ya = [
            _hn._mul(_hn._mul(rstd, ra[e]), _hn._add(wv[e], cutlass.Float32(1.0)))
            for e in range(_VEC)
        ]
        _hn._stg_v4(
            nrm_base + (row64 * width + stream * N) * 2 + col * 2,
            one,
            *[_hn._bf16x2(ya[2 * i], ya[2 * i + 1]) for i in range(4)],
        )

    if cutlass.const_expr(STEP):
        # The vectors the peers wrote into this slot, for the re-arm pass of the next launch (a
        # completed grid by then): a plain store by CTA 0.
        if bidx + tidx == 0:
            slot_count_ptr = (
                signal_state.iterator
                + cutlass.Int64(_rt._SIGNAL_SLOT_COUNTS_INDEX)
                + cutlass.Int64(ping_slot)
            ).llvm_ptr
            cute.arch.store(
                slot_count_ptr,
                cutlass.Int32(active_numel) // cutlass.Int32(_VEC),
                sem="relaxed",
                scope="gpu",
            )
    elif tidx == 0:
        # The last CTA to arrive records how many vectors the peers wrote into this slot and
        # rotates the signal; every CTA has read it above.
        _rt._advance_signal_if_last_cta(
            state_counter_ptr,
            state_signal_ptr,
            signal_state,
            signal_i32,
            ping_slot,
            cutlass.Int32(active_numel) // cutlass.Int32(_VEC),
            grid_n,
        )


@cute.jit
def _rs_inject(
    input_tensor: cute.Tensor,
    buffers: list[cute.Tensor],
    signal_state: cute.Tensor,
    launch_index: cutlass.Int32,
    inj_ptr: cute.Pointer,
    gate_ptr: cute.Pointer,
    hc_ptr: cute.Pointer,
    w_ptr: cute.Pointer,
    sh_ptr: cute.Pointer,
    out_ptr: cute.Pointer,
    nrm_ptr: cute.Pointer,
    rows: cutlass.Int32,
    s_inj: cutlass.Int32,
    s_gate: cutlass.Int32,
    s_hc: cutlass.Int32,
    s_sh: cutlass.Int32,
    eps: cutlass.Float32,
    e2p_base: cutlass.Int64,
    fw_base: cutlass.Int64,
    stream: cuda.CUstream,
    N: cutlass.Constexpr[int],
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    SUM_ORDER: cutlass.Constexpr[tuple[int, ...]],
    HC: cutlass.Constexpr[int],
    SPP: cutlass.Constexpr[int],
    STEP: cutlass.Constexpr[bool],
    FIN: cutlass.Constexpr[bool],
    TOPK: cutlass.Constexpr[int],
    PDL: cutlass.Constexpr[bool],
) -> None:
    inj = cute.make_tensor(inj_ptr, cute.make_layout((rows, HC), stride=(s_inj, 1)))
    gate = cute.make_tensor(gate_ptr, cute.make_layout((rows,), stride=(s_gate,)))
    kernel = _rs_inject_kernel(
        input_tensor,
        buffers,
        signal_state,
        launch_index,
        inj,
        gate,
        hc_ptr.toint(),
        w_ptr.toint(),
        sh_ptr.toint(),
        out_ptr.toint(),
        nrm_ptr.toint(),
        rows,
        s_hc,
        s_sh,
        eps,
        input_tensor.iterator.toint(),
        e2p_base,
        fw_base,
        N,
        LOCAL_RANK,
        WORLD_SIZE,
        PEER_ORDER,
        SUM_ORDER,
        HC,
        SPP,
        STEP,
        FIN,
        TOPK,
        PDL,
    )
    kernel.launch(grid=(rows, HC // SPP, 1), block=(N // _VEC, 1, 1), stream=stream, use_pdl=PDL)


#: Compiled launches by `(rank, world, device, hidden, hc, sum order, spp, step, fin, topk, pdl)`.
_COMPILED: dict = {}


def _compile(key):
    rank, world, device_index, hidden, hc, order, spp, step, fin, topk, pdl = key
    fn = _COMPILED.get(key)
    if fn is not None:
        return fn
    cutlass.cuda.initialize_cuda_context(device_index)
    fake_in = make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int32(), hidden), stride_order=(1, 0), assumed_align=_ALIGN
    )
    numel = cute.sym_int32(divisibility=_VEC)
    fake_buffers = [
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (_PING_PONG, world, numel),
            stride_order=(2, 1, 0),
            assumed_align=_ALIGN,
        )
        for _ in range(world)
    ]
    fake_state = make_fake_compact_tensor(
        cutlass.Int32, (_rt._SIGNAL_STATE_SIZE,), assumed_align=_rt._STATE_ALIGN
    )
    gm = cute.AddressSpace.gmem
    bf = cutlass.BFloat16
    # Only the pointer types matter to the compile; any 16-byte aligned address will do.
    p = 1 << 20
    fn = cute.compile(
        _rs_inject,
        fake_in,
        fake_buffers,
        fake_state,
        cutlass.Int32(0),
        make_ptr(bf, p, gm, assumed_align=2),
        make_ptr(bf, p, gm, assumed_align=2),
        make_ptr(bf, p, gm, assumed_align=16),
        make_ptr(bf, p, gm, assumed_align=16),
        make_ptr(bf, p, gm, assumed_align=16),
        make_ptr(bf, p, gm, assumed_align=16),
        make_ptr(bf, p, gm, assumed_align=16),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Float32(1e-6),
        cutlass.Int64(p),
        cutlass.Int64(p),
        make_fake_stream(),
        hidden,
        rank,
        world,
        tuple((rank + k) % world for k in range(world)),
        tuple(order),
        hc,
        spp,
        bool(step),
        bool(fin),
        int(topk),
        bool(pdl),
        options="--enable-tvm-ffi",
    )
    _COMPILED[key] = fn
    return fn


def supported(hidden: int, hc_count: int) -> bool:
    """The geometry the reduction tree is written for: a hidden in (2048, 4096] of whole 16-lane
    pairs, one thread a vector, up to 8 streams."""
    return (
        _SPLIT < hidden <= 2 * _SPLIT
        and hidden % (2 * _VEC) == 0
        and hidden // _VEC <= 1024
        and 1 <= hc_count <= 8
    )


def streams_per_cta(rows: int, hc_count: int, layouts) -> int:
    """Streams a CTA of the launch at `rows` rows: the first of `layouts` whose grid stays under its
    CTA limit, else every stream of a row in one CTA. Host arithmetic on the row count alone."""
    for spp, limit in layouts:
        if hc_count % spp == 0 and rows * (hc_count // spp) <= limit:
            return spp
    return hc_count


#: CTAs of a `fin` launch one SM holds: 320 threads at up to 96 registers (79 / 91 / 96 at 1 / 2 /
#: 4 streams a CTA, sm_103a, CuTe DSL 4.8.0), two of them in the 64K register file.
FIN_CTAS_PER_SM = 2


def fin_resident(rows: int, hc_count: int, layouts, sm_count: int) -> bool:
    """Whether the `fin` launch at `rows` rows is one resident wave on `sm_count` SMs. Every CTA of
    a `fin` row publishes to some peers (peer ordinal `j` goes to CTA `(j - 1) % (C / SPP)`), so a
    CTA that is not resident holds up its row on every peer; the plain launch publishes from the
    `c == 0` CTAs only, which the hardware dispatches first. Host arithmetic on the row count."""
    spp = streams_per_cta(rows, hc_count, layouts)
    return rows * (hc_count // spp) <= FIN_CTAS_PER_SM * sm_count


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


@dataclass(frozen=True)
class RsInject:
    """One rank's fused launches over an existing `DepRoundTrip` (its peer buffers, slot state and
    host-index registry: the reduce-scatter this launch replaces and the gathers around it share
    them)."""

    trip: object
    hc_count: int
    layouts: tuple
    #: `{streams a CTA: compiled launch}` for every layout `streams_per_cta` can pick.
    fns: dict
    #: The `fin` launches by the same keys; empty when not built.
    fin_fns: dict
    topk: int
    pdl: bool

    def spp(self, rows: int) -> int:
        return streams_per_cta(rows, self.hc_count, self.layouts)

    def _site_refuses(self, rows, hc, inject, norm_weight, shared, gate_logits) -> str | None:
        trip = self.trip
        bf = torch.bfloat16
        if hc.dim() != 2 or hc.shape[1] != self.hc_count * trip.hidden:
            return f"hc {tuple(hc.shape)}"
        if hc.shape[0] != rows or not 1 <= rows <= min(trip.max_rows, MAX_ROWS):
            return f"{hc.shape[0]} hc rows, {rows} rows (up to {min(trip.max_rows, MAX_ROWS)})"
        if not all(t.dtype is bf for t in (hc, inject, norm_weight, shared, gate_logits)):
            return "an operand is not bf16"
        if tuple(inject.shape) != (rows, self.hc_count) or inject.stride(1) != 1:
            return f"inject {tuple(inject.shape)} strides {tuple(inject.stride())}"
        if tuple(shared.shape) != (rows, trip.hidden) or shared.stride(1) != 1:
            return f"shared {tuple(shared.shape)} strides {tuple(shared.stride())}"
        if tuple(gate_logits.shape) not in ((rows,), (rows, 1)):
            return f"gate logits {tuple(gate_logits.shape)}"
        if rows > 1 and gate_logits.stride(0) == 0:
            return "one gate logit for every row"
        if (
            tuple(norm_weight.shape) != (self.hc_count * trip.hidden,)
            or not norm_weight.is_contiguous()
        ):
            return f"norm weight {tuple(norm_weight.shape)}"
        if hc.stride(1) != 1 or hc.stride(0) % 8 or shared.stride(0) % 8:
            return f"row strides hc {tuple(hc.stride())} shared {tuple(shared.stride())}"
        if not all(_aligned(t) for t in (hc, norm_weight, shared)):
            return "an operand is not 16-byte aligned"
        if len({t.device for t in (hc, inject, norm_weight, shared, gate_logits)}) != 1:
            return "operands on different devices"
        return None

    def refuses(self, partial, hc, inject, norm_weight, shared, gate_logits) -> str | None:
        """Why this call cannot take the fused launch (None: it can). Host side: shapes, dtypes,
        strides and addresses only. A rank that refuses runs the plain reduce-scatter, which
        exchanges the same bytes with its peers."""
        trip = self.trip
        rows = hc.shape[0] if hc.dim() == 2 else 0
        why = self._site_refuses(rows, hc, inject, norm_weight, shared, gate_logits)
        if why is not None:
            return why
        if (
            partial.dtype is not torch.bfloat16
            or tuple(partial.shape) != (trip.world * rows, trip.hidden)
            or not partial.is_contiguous()
            or not _aligned(partial)
            or partial.device != hc.device
        ):
            return f"partials {tuple(partial.shape)} strides {tuple(partial.stride())}"
        return None

    def refuses_fin(self, hc, inject, norm_weight, shared, gate_logits) -> str | None:
        """`refuses` of the `fin` launch, asked before the runner is entered (every operand but
        the runner's output). None: it can."""
        if not self.fin_fns:
            return "no finalize launch was built"
        rows = hc.shape[0] if hc.dim() == 2 else 0
        return self._site_refuses(rows, hc, inject, norm_weight, shared, gate_logits)

    def fin_operands_refuse(self, rows: int, g2, w, e2p) -> str | None:
        """Why the runner's unfinalized output `(g2, w, e2p)` is not in the launch's layout."""
        trip = self.trip
        m = trip.world * rows
        if (
            g2.dtype is not torch.bfloat16
            or g2.dim() != 2
            or g2.shape[1] != trip.hidden
            or not g2.is_contiguous()
            or not _aligned(g2)
        ):
            return f"FC2 rows {tuple(g2.shape)} {g2.dtype} strides {tuple(g2.stride())}"
        if w.dtype is not torch.bfloat16 or w.numel() != m * self.topk or not w.is_contiguous():
            return f"weights {tuple(w.shape)} {w.dtype}"
        if e2p.dtype is not torch.int32 or e2p.numel() != m * self.topk or not e2p.is_contiguous():
            return f"index map {tuple(e2p.shape)} {e2p.dtype}"
        return None

    def _launch(self, fn, src, hc, inject, norm_weight, eps, shared, gate_logits, e2p, w):
        rows, width = hc.shape
        hc_out = torch.empty((rows, width), dtype=hc.dtype, device=hc.device)
        normed = torch.empty((rows, width), dtype=hc.dtype, device=hc.device)
        trip = self.trip
        stream = _raw_stream(hc.device.index)
        # The launch's index on the workspace, as a reduce-scatter would take it.
        index = None if trip.steps is None else trip.steps.next(trip, stream)
        gate = gate_logits.reshape(rows, -1) if gate_logits.dim() == 1 else gate_logits
        fn(
            src,
            trip.peer_buffers,
            trip.state,
            0 if index is None else index,
            inject.data_ptr(),
            gate.data_ptr(),
            hc.data_ptr(),
            norm_weight.data_ptr(),
            shared.data_ptr(),
            hc_out.data_ptr(),
            normed.data_ptr(),
            rows,
            inject.stride(0),
            gate.stride(0),
            hc.stride(0),
            shared.stride(0),
            float(eps),
            0 if e2p is None else e2p.data_ptr(),
            0 if w is None else w.data_ptr(),
            stream,
        )
        if trip.steps is not None and index is None:
            trip.advance(1, stream)
        return hc_out, normed

    def run(self, partial, hc, inject, norm_weight, eps: float, shared, gate_logits) -> tuple:
        """`[W * R, H]` partials on every rank + the next site's operands -> `(hc_out, normed)`,
        both fresh `[R, C * H]`: what `reduce_scatter(partial)` followed by
        `qwen4_exp_hc_inject_norm(hc, routed, inject, norm_weight, eps, C, shared, gate_logits)`
        returns. The caller has asked `refuses` and joined the stream that wrote `shared` /
        `gate_logits`."""
        fn = self.fns[self.spp(hc.shape[0])]
        return self._launch(
            fn, partial, hc, inject, norm_weight, eps, shared, gate_logits, None, None
        )

    def run_fin(self, g2, w, e2p, hc, inject, norm_weight, eps: float, shared, gate_logits):
        """`run` over the runner's unfinalized output of the group's `W * R` gathered rows (`g2` its
        permuted FC2 rows, `w` the routing weights, `e2p` the expanded -> permuted map): the same
        `(hc_out, normed)`."""
        fn = self.fin_fns[self.spp(hc.shape[0])]
        return self._launch(fn, g2, hc, inject, norm_weight, eps, shared, gate_logits, e2p, w)


def build(trip, hc_count: int, layouts, fin: bool = False, topk: int = 10, pdl: bool = False):
    """Compile this rank's fused launches over `trip` (a `dep_lamport_round_trip.DepRoundTrip`):
    one a layout `streams_per_cta` can pick, and with `fin` the finalize forms too (`pdl`: those
    launch with programmatic dependent launch). Rank-local: it allocates nothing shared and
    exchanges nothing. Call it outside graph capture."""
    hidden = trip.hidden
    assert supported(hidden, hc_count), (hidden, hc_count)
    step = trip.steps is not None
    fns, fin_fns = {}, {}
    for spp in sorted({spp for spp, _ in layouts if hc_count % spp == 0} | {hc_count}):
        base = (trip.rank, trip.world, trip.buffer.device.index, hidden, hc_count, trip.order, spp)
        fns[spp] = _compile((*base, step, False, int(topk), False))
        if fin:
            fin_fns[spp] = _compile((*base, step, True, int(topk), bool(pdl)))
    return RsInject(
        trip=trip,
        hc_count=hc_count,
        layouts=tuple(layouts),
        fns=fns,
        fin_fns=fin_fns,
        topk=int(topk),
        pdl=bool(pdl and fin),
    )


def finalize_reference(g2, w, e2p) -> torch.Tensor:
    """The runner's finalize of one rank's window: per gathered row, fp32 FMAs of `w * g2[e2p]` in
    slot order from +0 (a slot of -1 skipped), rounded once to bf16."""
    rows, topk = e2p.shape
    acc = torch.zeros((rows, g2.shape[1]), dtype=torch.float32, device=g2.device)
    for k in range(topk):
        slot = e2p[:, k].long()
        picked = g2[slot.clamp(min=0)].float() * w[:, k : k + 1].float()
        acc = torch.where((slot >= 0).unsqueeze(1), acc + picked, acc)
    return acc.to(torch.bfloat16)


class DepLamportRsInject(OpWrapper):
    """`dep_lamport_rs_inject(partial, rs, hc, inject, norm_weight, eps, shared, gate_logits, fin)`:
    the attention-DP MoE reduce-scatter of `partial` (`[W * R, H]` bf16 window partials) over the
    workspace of `rs` (`build`), with the next hyper-connection site's combine, injection and
    grouped norm: `(hc_out, normed)`, each `[R, C * H]` bf16, bitwise
    `dep_lamport_round_trip(partial, trip, True)` followed by `qwen4_exp_hc_inject_norm(hc, routed,
    inject, norm_weight, eps, C, shared, gate_logits)`. With `fin = (g2, w, e2p)` the partial is the
    runner's finalize of that unfinalized output (`partial` is then None). Every rank calls a
    reduce-scatter of the workspace at the same point with the same `R`."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(world=world, hidden=2560, hc=4, rows=rows, step=step, fin=fin))
        for world, rows, step, fin, why in (
            (4, 12, False, False, "dep4 MTP-3 verify, 3 requests a rank, device-counter slots"),
            (4, 12, True, False, "dep4 MTP-3 verify, 3 requests a rank, host-index slots"),
            (4, 208, True, False, "dep4 52 requests a rank (widest measured cap), 2 streams a CTA"),
            (8, 12, True, False, "dep8 3 requests a rank, 4 CTAs a row"),
            (8, 72, True, False, "dep8 18 requests a rank: 4 CTAs a row only under dep8's layouts"),
            (8, 112, True, False, "dep8 28 requests a rank, 2 streams a CTA"),
            (4, 12, True, True, "dep4 3 requests a rank, the finalize folded in (a tray runs it)"),
            (4, 12, False, True, "dep4 3 requests a rank, finalize folded in, device counter"),
            (4, 128, True, True, "dep4 32 requests a rank, finalize folded in, 2 streams a CTA"),
            (8, 12, True, True, "dep8 3 requests a rank, the runner's finalize folded in"),
            (8, 112, True, True, "dep8 28 requests a rank, finalize folded in, 2 streams a CTA"),
        )
    )

    note = """
    Source: one CuTe DSL kernel (`_rs_inject_kernel`) in two slot protocols (device counter; host
    index with `LamportSteps`) and two input forms (finalized partials;
    with `fin` the runner's permuted FC2 rows, weights and index map), compiled per rank and layout
    at build (`--enable-tvm-ffi`), over the workspace of a `dep_lamport_round_trip.DepRoundTrip`.
    Written for GB300 at 4 ranks (one node) and 8 (two nodes of one rack), hidden 2560, 4 streams,
    1 to 256 rows a rank; its prototype ran the same statements at 4 and 8 ranks bitwise against
    the plain reduce-scatter + inject-norm (1..256 rows). Not certified here (no GPU run yet).
    """

    def __call__(
        self,
        partial,
        rs: RsInject,
        hc: torch.Tensor,
        inject: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        shared: torch.Tensor,
        gate_logits: torch.Tensor,
        fin=None,
    ) -> tuple:
        if fin is not None:
            return rs.run_fin(*fin, hc, inject, norm_weight, eps, shared, gate_logits)
        return rs.run(partial, hc, inject, norm_weight, eps, shared, gate_logits)

    def reference(
        self,
        partials_in_rank_order,
        trip_rank: int,
        hc,
        inject,
        norm_weight,
        eps: float,
        shared,
        gate_logits,
        sum_order=None,
    ) -> tuple:
        """Collective (the documented exception to mirroring `__call__`): every rank's finalized
        partials in rank order and the rank whose result to form; the plain reduce-scatter's
        reference then `qwen4_exp_hc_inject_norm`'s float64 reference."""
        routed = _rt.dep_lamport_round_trip.reference(
            partials_in_rank_order, trip_rank, True, sum_order
        )
        return _hn.qwen4_exp_hc_inject_norm.reference(
            hc, routed, inject, norm_weight, eps, inject.shape[1], shared, gate_logits
        )

    def is_valid(
        self,
        partial,
        rs: RsInject,
        hc: torch.Tensor,
        inject: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        shared: torch.Tensor,
        gate_logits: torch.Tensor,
        fin=None,
    ) -> None:
        assert (partial is None) == (fin is not None), "partials or the runner's output, not both"
        if fin is None:
            why = rs.refuses(partial, hc, inject, norm_weight, shared, gate_logits)
        else:
            why = rs.refuses_fin(hc, inject, norm_weight, shared, gate_logits)
            why = why or rs.fin_operands_refuse(hc.shape[0], *fin)
        assert why is None, why


dep_lamport_rs_inject = DepLamportRsInject()
