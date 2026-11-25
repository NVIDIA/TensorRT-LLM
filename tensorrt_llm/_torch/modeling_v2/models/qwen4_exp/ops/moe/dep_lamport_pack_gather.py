# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP MoE gather of each row's NVFP4 form and router logits: one Lamport launch.

Today an attention-DP MoE block gathers the group's bf16 rows (`dep_lamport_round_trip`, 2560
lanes a row), runs the router over all of them and the route front quantizes all of them to NVFP4.
Here every rank quantizes and routes its OWN rows, and the launch gathers what the routing and the
runner read: a row travels as 16-byte vectors

    [0, H / 32)                  E2M1 codes, 32 elements a vector (two NVFP4 blocks)
    [H / 32, H / 32 + H / 256)   E4M3 linear block scales, 16 blocks a vector
    [.., + E / 8)                the row's bf16 router logits, 8 a vector

(1,232 bf16 lanes a row at H 2560, E 512, instead of 2,560). The codes and scales are the route
front's NVFP4 arithmetic (`moe_decode_front._nvfp4_block`, linear scales at the experts' input
global scale) over the row with its -0.0 lanes as +0.0, as the route front reads the gathered rows
today. A code or logit lane equal to the transport's marker (0x8000) travels as 0x0000: for a logit
that is the gather's own sanitizing; for a code lane it is an E2M1 -0 nibble after three +0
nibbles, which becomes +0, the same zero. Scale bytes are positive E4M3, never 0x8000.

Outputs, rank-major over the group's `W * R` rows: the codes `[W R, H / 2]` uint8 and scales
`[W R, H / 16]` uint8 in the route front's layouts, and the logits `[W R, E]` bf16.

`ownk` (owner top-k): the row's owner routes it instead, and the row carries its routing rather
than its logits. One warp a row loads the owner's logits, sanitized as above, and runs the route
front's own `route_one_token` (the statements every routing warp runs over the gathered logits);
lanes 0..2 then publish 3 vectors of 16-bit slots: the 10 experts, their 10 BF16 weights' bits, 4
zeros (`moe_route_front__decode.IDS_ROW_SLOTS`; experts < 512 and weights >= +0.0, never the
marker). A row is 93 vectors instead of 154, and the routing-only front reads the slots
(`route_front_only(..., ids_in=True)`) instead of routing all `W R` rows on every rank. The third
output is then the routing `[W R, 24]` int16.

Transport: the host-index gather of `dep_lamport_round_trip`'s workspace, statement for statement
on an int32 view of the same buffers (a vector is 4 int32 lanes; the marker is any 16-bit half
equal to 0x8000): the slot is `(state[1] + launch_index) mod 3`, each reader re-arms the vector it
took, the re-arm pass of the launch's slot count stays, every receive issues its first probe of
every peer before it waits on any, and CTA 0 records a count of 0. One CTA a row; each warp holds
one kind of vector (code units, then scale receivers, then logit units). The launch releases its
PDL dependents at its start (the routing front behind it waits for this grid before it reads).

Collective: every rank of the group launches it at the same point with the same row count; every
CTA must be resident at once on every rank (one 192-thread CTA a row).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, const_expr
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op

from tensorrt_llm._torch.kernels.blackwell import moe_decode_front as _mdf
from tensorrt_llm._torch.kernels.blackwell import moe_route_front as _mrf
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import dep_lamport_round_trip as _rt

_ALIGN = _rt._ALIGN
_PING_PONG = _rt._PING_PONG
#: Two marker lanes in one word.
_NZ_WORD = -2147450880  # 0x80008000
_raw_stream = torch._C._cuda_getCurrentRawStream


@dsl_user_op
def _clean(w, *, loc=None, ip=None):
    """`w` with every 16-bit half equal to 0x8000 (the marker) as 0x0000."""
    return Int32(
        _mdf._asm(
            _mdf.T.i32(),
            [_mdf._i32(w, loc, ip)],
            "{\n\t.reg .b32 l, h;\n\t.reg .pred p;\n\t"
            "and.b32 l, $1, 0x0000FFFF;\n\tsetp.eq.u32 p, l, 0x00008000;\n\tselp.b32 l, 0, l, p;\n\t"
            "and.b32 h, $1, 0xFFFF0000;\n\tsetp.eq.u32 p, h, 0x80000000;\n\tselp.b32 h, 0, h, p;\n\t"
            "or.b32 $0, l, h;\n\t}",
            "=r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _marker(w0, w1, w2, w3, *, loc=None, ip=None):
    """1 when any 16-bit half of the four words is the marker (the vector is not ready), else 0."""
    body = ["{", ".reg .b32 t;", ".reg .pred p, q;", "mov.b32 $0, 0;"]
    for i in range(1, 5):
        body += [
            f"and.b32 t, ${i}, 0x0000FFFF;",
            "setp.eq.u32 p, t, 0x00008000;",
            f"and.b32 t, ${i}, 0xFFFF0000;",
            "setp.eq.u32 q, t, 0x80000000;",
            "or.pred p, p, q;",
            "@p mov.b32 $0, 1;",
        ]
    body.append("}")
    return Int32(
        _mdf._asm(
            _mdf.T.i32(),
            [_mdf._i32(w, loc, ip) for w in (w0, w1, w2, w3)],
            "\n\t".join(body),
            "=r,r,r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl(x, src, *, loc=None, ip=None):
    return Int32(
        _mdf._asm(
            _mdf.T.i32(),
            [_mdf._i32(x, loc, ip), _mdf._i32(src, loc, ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _routing_words(sel, w, lane):
    """The 4 words of routing vector `lane` (meaningful in lanes 0..2; every lane of the warp calls
    it): 16-bit slot `s` holds the expert of pick `s` (s < 10), the weight bits of pick `s - 10`
    (s < 20), else 0; word `j` = slot `2 j` | slot `2 j + 1` << 16 of the vector's 8."""
    ids16 = sel & Int32(65535)
    w16 = w & Int32(65535)
    words = []
    for j in cutlass.range_constexpr(4):
        half = []
        for h in cutlass.range_constexpr(2):
            slot = lane * Int32(8) + Int32(2 * j + h)
            a = _shfl(ids16, slot)
            b = _shfl(w16, slot - Int32(10))
            half.append(
                _mdf._sel_lt(slot, Int32(10), a, _mdf._sel_lt(slot, Int32(20), b, Int32(0)))
            )
        words.append(half[0] | (half[1] << Int32(16)))
    return words[0], words[1], words[2], words[3]


@cute.jit
def _poll_from(local, offset, first):
    """The vector at `offset` (int32 lanes) of this rank's slot once no 16-bit lane holds the
    marker, from the already-issued load `first`."""
    values = first
    while _marker(values[0], values[1], values[2], values[3]) != Int32(0):
        values = local.load(offset, 4, alignment=_ALIGN, is_volatile=True)
    copy = cutlass.Array(cutlass.Int32, 4, alignment=_ALIGN)
    for lane in cutlass.range_constexpr(4):
        copy[lane] = values[lane]
    return copy


@cute.jit
def _receive(
    local,
    nz_vec,
    ping_base,
    ws,
    off,
    out_ptr,
    out_row_bytes: cutlass.Constexpr,
    out_col,
    rows,
    row,
    WORLD_SIZE: cutlass.Constexpr,
    PEER_ORDER: cutlass.Constexpr,
):
    """Vector `off` of every peer's region: every first load issued before any wait, each taken,
    re-armed and stored at byte `out_col` of that peer's row of `out_ptr`."""
    first = []
    for k in cutlass.range_constexpr(1, WORLD_SIZE):
        first.append(
            local.load(
                ping_base + Int64(PEER_ORDER[k]) * ws + off, 4, alignment=_ALIGN, is_volatile=True
            )
        )
    for k in cutlass.range_constexpr(1, WORLD_SIZE):
        o = ping_base + Int64(PEER_ORDER[k]) * ws + off
        got = _poll_from(local, o, first[k - 1])
        local.store(nz_vec[0:4, cutlass.align(_ALIGN)], o, 4, alignment=_ALIGN, is_volatile=True)
        orow = Int64(PEER_ORDER[k]) * Int64(rows) + Int64(row)
        _mdf._st_v4(out_ptr + orow * Int64(out_row_bytes) + out_col, got[0], got[1], got[2], got[3])


@cute.kernel
def _pack_kernel(
    x_ptr: Int64,
    a_stride: Int32,
    lg_ptr: Int64,
    gs_ptr: Int64,
    xq_ptr: Int64,
    xsf_ptr: Int64,
    lgo_ptr: Int64,
    buffers: list[cute.Tensor],
    signal_state: cute.Tensor,
    launch_index: Int32,
    rows: Int32,
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    HIDDEN: cutlass.Constexpr[int],
    EXPERTS: cutlass.Constexpr[int],
    OWNK: cutlass.Constexpr[bool],
):
    """One CTA a row: warps [0, s0 / 32) code units (thread u < H / 32), then scale receivers,
    then logit units (E / 8 lanes), or with `OWNK` one routing warp (its lanes 0..2 the routing
    units); the other lanes only re-arm."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    grid_x, _, _ = cute.arch.grid_dim()
    bdim, _, _ = cute.arch.block_dim()
    # The routing front behind it may be scheduled now; its own wait holds it until this grid has
    # completed. This grid is no PDL launch: everything before it has completed.
    cute.arch.griddepcontrol_launch_dependents()
    cv: cutlass.Constexpr = HIDDEN // 32
    sv: cutlass.Constexpr = HIDDEN // 256
    lv: cutlass.Constexpr = 3 if OWNK else EXPERTS // 8
    vpr: cutlass.Constexpr = HIDDEN // 32 + HIDDEN // 256 + lv
    s0: cutlass.Constexpr = -(-cv // 32) * 32
    l0: cutlass.Constexpr = s0 + -(-sv // 32) * 32
    buffer_arrays = [cutlass.make_array_view(buffer) for buffer in buffers]
    local = buffer_arrays[LOCAL_RANK]
    ws = Int64(cute.size(buffers[0], mode=[2]))
    state_signal_ptr = (signal_state.iterator + Int64(_rt._SIGNAL_VALUE_INDEX)).llvm_ptr
    signal_i64 = Int64(
        cute.arch.load(state_signal_ptr, Int32, sem="relaxed", scope="gpu") + launch_index
    )
    ping_slot = signal_i64 % _PING_PONG
    clear_slot = (signal_i64 + Int64(1)) % _PING_PONG
    ping_base = Int64(ping_slot) * Int64(WORLD_SIZE) * ws
    clear_base = Int64(clear_slot) * Int64(WORLD_SIZE) * ws
    nz_vec = cutlass.Array(cutlass.Int32, 4, alignment=_ALIGN)
    for lane in cutlass.range_constexpr(4):
        nz_vec[lane] = Int32(_NZ_WORD)
    gs = _mdf._ld_f32(gs_ptr)
    row = bidx
    u = tidx
    lane_id = tidx % Int32(32)
    rbase = Int64(row) * Int64(vpr * 4)
    own_row = Int64(LOCAL_RANK) * Int64(rows) + Int64(row)
    mine = ping_base + Int64(LOCAL_RANK) * ws
    # The payload first: its loads and quantization overlap the slot-count load.
    c0 = Int32(0)
    c1 = Int32(0)
    c2 = Int32(0)
    c3 = Int32(0)
    s16 = Int32(0)
    if u < Int32(cv):
        src = x_ptr + (Int64(row) * Int64(a_stride) + Int64(u) * Int64(32)) * Int64(2)
        a0, a1, a2, a3 = _mdf._ld_v4(src)
        a4, a5, a6, a7 = _mdf._ld_v4(src + Int64(16))
        e0, e1, e2, e3 = _mdf._ld_v4(src + Int64(32))
        e4, e5, e6, e7 = _mdf._ld_v4(src + Int64(48))
        # The route front quantizes the gathered rows, whose -0.0 lanes travel as +0.0.
        lo0, hi0, sf0 = _mdf._nvfp4_block(
            _clean(a0),
            _clean(a1),
            _clean(a2),
            _clean(a3),
            _clean(a4),
            _clean(a5),
            _clean(a6),
            _clean(a7),
            gs,
        )
        lo1, hi1, sf1 = _mdf._nvfp4_block(
            _clean(e0),
            _clean(e1),
            _clean(e2),
            _clean(e3),
            _clean(e4),
            _clean(e5),
            _clean(e6),
            _clean(e7),
            gs,
        )
        c0 = _clean(lo0)
        c1 = _clean(hi0)
        c2 = _clean(lo1)
        c3 = _clean(hi1)
        s16 = sf0 | (sf1 << Int32(8))
    elif (u >= Int32(l0)) & (u < Int32(l0 + (32 if OWNK else lv))):
        if const_expr(OWNK):
            # The whole warp routes the owner's row over its sanitized logits (16 a lane).
            src = lg_ptr + (Int64(row) * Int64(EXPERTS) + Int64(lane_id) * Int64(16)) * Int64(2)
            m0, m1, m2, m3 = _mdf._ld_v4(src)
            m4, m5, m6, m7 = _mdf._ld_v4(src + Int64(16))
            packed = tuple(_clean(m) for m in (m0, m1, m2, m3, m4, m5, m6, m7))
            sel, w = _mrf.route_one_token(packed, lane_id, 10)
            c0, c1, c2, c3 = _routing_words(sel, w, lane_id)
        else:
            m0, m1, m2, m3 = _mdf._ld_v4(
                lg_ptr + (Int64(row) * Int64(EXPERTS) + Int64(u - Int32(l0)) * Int64(8)) * Int64(2)
            )
            c0 = _clean(m0)
            c1 = _clean(m1)
            c2 = _clean(m2)
            c3 = _clean(m3)
    # The scale vector of an 8-lane group (every lane takes part; non-code lanes hold 0).
    g = lane_id - lane_id % Int32(8)
    q0 = _shfl(s16, g) | (_shfl(s16, g + Int32(1)) << Int32(16))
    q1 = _shfl(s16, g + Int32(2)) | (_shfl(s16, g + Int32(3)) << Int32(16))
    q2 = _shfl(s16, g + Int32(4)) | (_shfl(s16, g + Int32(5)) << Int32(16))
    q3 = _shfl(s16, g + Int32(6)) | (_shfl(s16, g + Int32(7)) << Int32(16))
    # Re-arm the slot the launch after next uses, as the host-index gather does.
    clear_count_ptr = (
        signal_state.iterator + Int64(_rt._SIGNAL_SLOT_COUNTS_INDEX) + Int64(clear_slot)
    ).llvm_ptr
    clear_vectors = cute.arch.load(clear_count_ptr, Int32, sem="relaxed", scope="gpu")
    clear_vector_idx = bidx * bdim + tidx
    clear_stride = grid_x * bdim
    while clear_vector_idx < clear_vectors:
        for source_rank in cutlass.range_constexpr(WORLD_SIZE):
            if source_rank != LOCAL_RANK:
                local.store(
                    nz_vec[0:4, cutlass.align(_ALIGN)],
                    clear_base + Int64(source_rank) * ws + Int64(clear_vector_idx) * Int64(4),
                    4,
                    alignment=_ALIGN,
                    is_volatile=True,
                )
        clear_vector_idx = clear_vector_idx + clear_stride

    if u < Int32(cv):
        code = cutlass.Array(cutlass.Int32, 4, alignment=_ALIGN)
        code[0] = c0
        code[1] = c1
        code[2] = c2
        code[3] = c3
        for k in cutlass.range_constexpr(1, WORLD_SIZE):
            buffer_arrays[PEER_ORDER[k]].store(
                code[0:4, cutlass.align(_ALIGN)],
                mine + rbase + Int64(u) * Int64(4),
                4,
                alignment=_ALIGN,
                is_volatile=True,
            )
        if u % Int32(8) == Int32(0):
            sc = cutlass.Array(cutlass.Int32, 4, alignment=_ALIGN)
            sc[0] = q0
            sc[1] = q1
            sc[2] = q2
            sc[3] = q3
            off_s = rbase + (Int64(cv) + Int64(u // Int32(8))) * Int64(4)
            for k in cutlass.range_constexpr(1, WORLD_SIZE):
                buffer_arrays[PEER_ORDER[k]].store(
                    sc[0:4, cutlass.align(_ALIGN)],
                    mine + off_s,
                    4,
                    alignment=_ALIGN,
                    is_volatile=True,
                )
            _mdf._st_v4(
                xsf_ptr + own_row * Int64(HIDDEN // 16) + Int64(u // Int32(8)) * Int64(16),
                q0,
                q1,
                q2,
                q3,
            )
        _mdf._st_v4(xq_ptr + own_row * Int64(HIDDEN // 2) + Int64(u) * Int64(16), c0, c1, c2, c3)
        _receive(
            local,
            nz_vec,
            ping_base,
            ws,
            rbase + Int64(u) * Int64(4),
            xq_ptr,
            HIDDEN // 2,
            Int64(u) * Int64(16),
            rows,
            row,
            WORLD_SIZE,
            PEER_ORDER,
        )
    elif (u >= Int32(s0)) & (u < Int32(s0 + sv)):
        j = u - Int32(s0)
        _receive(
            local,
            nz_vec,
            ping_base,
            ws,
            rbase + (Int64(cv) + Int64(j)) * Int64(4),
            xsf_ptr,
            HIDDEN // 16,
            Int64(j) * Int64(16),
            rows,
            row,
            WORLD_SIZE,
            PEER_ORDER,
        )
    elif (u >= Int32(l0)) & (u < Int32(l0 + lv)):
        kk = u - Int32(l0)
        lgv = cutlass.Array(cutlass.Int32, 4, alignment=_ALIGN)
        lgv[0] = c0
        lgv[1] = c1
        lgv[2] = c2
        lgv[3] = c3
        off_l = rbase + (Int64(cv + sv) + Int64(kk)) * Int64(4)
        for k in cutlass.range_constexpr(1, WORLD_SIZE):
            buffer_arrays[PEER_ORDER[k]].store(
                lgv[0:4, cutlass.align(_ALIGN)], mine + off_l, 4, alignment=_ALIGN, is_volatile=True
            )
        # 16-bit lanes a row of the third output: the logits, or with OWNK the routing slots
        out_lanes: cutlass.Constexpr = _mrf.IDS_ROW_SLOTS if OWNK else EXPERTS
        _mdf._st_v4(
            lgo_ptr + (own_row * Int64(out_lanes) + Int64(kk) * Int64(8)) * Int64(2), c0, c1, c2, c3
        )
        _receive(
            local,
            nz_vec,
            ping_base,
            ws,
            off_l,
            lgo_ptr,
            out_lanes * 2,
            Int64(kk) * Int64(16),
            rows,
            row,
            WORLD_SIZE,
            PEER_ORDER,
        )
    # Every vector was re-armed by its reader: the slot's count is 0 (CTA 0, a plain store).
    if bidx + tidx == 0:
        slot_count_ptr = (
            signal_state.iterator + Int64(_rt._SIGNAL_SLOT_COUNTS_INDEX) + Int64(ping_slot)
        ).llvm_ptr
        cute.arch.store(slot_count_ptr, Int32(0), sem="relaxed", scope="gpu")


def threads(hidden: int, experts: int, ownk: bool = False) -> int:
    """Threads a CTA: whole warps of code units, scale receivers and logit units (`ownk`: one
    routing warp)."""
    tail = 1 if ownk else -(-(experts // 8) // 32)
    return (-(-(hidden // 32) // 32) + -(-(hidden // 256) // 32) + tail) * 32


@cute.jit
def _pack_launch(
    x_ptr: Int64,
    a_stride: Int32,
    lg_ptr: Int64,
    gs_ptr: Int64,
    xq_ptr: Int64,
    xsf_ptr: Int64,
    lgo_ptr: Int64,
    buffers: list[cute.Tensor],
    signal_state: cute.Tensor,
    launch_index: Int32,
    rows: Int32,
    stream: cuda.CUstream,
    LOCAL_RANK: cutlass.Constexpr[int],
    WORLD_SIZE: cutlass.Constexpr[int],
    PEER_ORDER: cutlass.Constexpr[tuple[int, ...]],
    HIDDEN: cutlass.Constexpr[int],
    EXPERTS: cutlass.Constexpr[int],
    OWNK: cutlass.Constexpr[bool],
) -> None:
    _pack_kernel(
        x_ptr,
        a_stride,
        lg_ptr,
        gs_ptr,
        xq_ptr,
        xsf_ptr,
        lgo_ptr,
        buffers,
        signal_state,
        launch_index,
        rows,
        LOCAL_RANK,
        WORLD_SIZE,
        PEER_ORDER,
        HIDDEN,
        EXPERTS,
        OWNK,
    ).launch(grid=(rows, 1, 1), block=(threads(HIDDEN, EXPERTS, OWNK), 1, 1), stream=stream)


_COMPILED: dict = {}


def transported_codes(codes: torch.Tensor) -> torch.Tensor:
    """NVFP4 `codes` (uint8, an even byte count a row) as the gather delivers them: a 16-bit half
    equal to the transport's marker 0x8000 (three E2M1 +0 nibbles, then a -0 nibble) arrives as
    0x0000, the same four zeros. Every other byte is unchanged."""
    halves = codes.contiguous().view(torch.int16)
    clean = torch.where(halves == _rt._NEG_ZERO, torch.zeros_like(halves), halves)
    return clean.view(torch.uint8).view(codes.shape)


def vectors_a_row(hidden: int, experts: int, ownk: bool = False) -> int:
    """16-byte vectors a row travels as."""
    return hidden // 32 + hidden // 256 + (3 if ownk else experts // 8)


class PackGather:
    """The packed gather over a host-index `dep_lamport_round_trip.DepRoundTrip` (`trip`), compiled
    for this rank at `hidden` / `experts` (`ownk`: rows carry their owner's routing). Rank-local to
    build; outside graph capture."""

    def __init__(self, trip, hidden: int, experts: int, ownk: bool = False) -> None:
        assert trip.steps is not None, "the packed gather serves the host-index workspace"
        assert hidden % 256 == 0 and experts % 8 == 0, (hidden, experts)
        assert not ownk or experts == 512, "the owner's top-k is the route front's (512 experts)"
        self.trip, self.hidden, self.experts, self.ownk = trip, hidden, experts, bool(ownk)
        self.vpr = vectors_a_row(hidden, experts, ownk)
        dev = trip.buffer.device.index
        key = (trip.rank, trip.world, dev, hidden, experts, self.ownk)
        fn = _COMPILED.get(key)
        if fn is None:
            cutlass.cuda.initialize_cuda_context(dev)
            numel = cute.sym_int32(divisibility=4)
            fake_buffers = [
                make_fake_compact_tensor(
                    cutlass.Int32,
                    (_PING_PONG, trip.world, numel),
                    stride_order=(2, 1, 0),
                    assumed_align=_ALIGN,
                )
                for _ in range(trip.world)
            ]
            fake_state = make_fake_compact_tensor(
                cutlass.Int32, (_rt._SIGNAL_STATE_SIZE,), assumed_align=_rt._STATE_ALIGN
            )
            z, zi = Int64(0), Int32(0)
            order = tuple((trip.rank + k) % trip.world for k in range(trip.world))
            fn = cute.compile(
                _pack_launch,
                z,
                zi,
                z,
                z,
                z,
                z,
                z,
                fake_buffers,
                fake_state,
                zi,
                zi,
                make_fake_stream(),
                trip.rank,
                trip.world,
                order,
                hidden,
                experts,
                self.ownk,
                options="--enable-tvm-ffi",
            )
            _COMPILED[key] = fn
        self.fn = fn
        self.buffers = [b.view(torch.int32) for b in trip.peer_buffers]

    def takes(self, rows: int, x: torch.Tensor) -> bool:
        """Every rank answers the same for the group's rows: in the workspace, bf16 `[R, H]` rows."""
        trip = self.trip
        return (
            1 <= rows <= trip.max_rows
            and rows * self.vpr * 8 <= int(trip.buffer.shape[2])
            and x.dtype is torch.bfloat16
            and tuple(x.shape) == (rows, self.hidden)
            and x.stride(1) == 1
            and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0
        )

    def __call__(self, x: torch.Tensor, logits: torch.Tensor, global_scale: torch.Tensor):
        trip = self.trip
        rows, w, dev = x.shape[0], trip.world, x.device
        xq = torch.empty((w * rows, self.hidden // 2), dtype=torch.uint8, device=dev)
        xsf = torch.empty((w * rows, self.hidden // 16), dtype=torch.uint8, device=dev)
        if self.ownk:
            lg = torch.empty((w * rows, _mrf.IDS_ROW_SLOTS), dtype=torch.int16, device=dev)
        else:
            lg = torch.empty((w * rows, self.experts), dtype=torch.bfloat16, device=dev)
        stream = _raw_stream(dev.index)
        index = trip.steps.next(trip, stream)
        self.fn(
            x.data_ptr(),
            x.stride(0),
            logits.data_ptr(),
            global_scale.data_ptr(),
            xq.data_ptr(),
            xsf.data_ptr(),
            lg.data_ptr(),
            self.buffers,
            trip.state,
            0 if index is None else index,
            rows,
            stream,
        )
        if index is None:
            trip.advance(1, stream)
        return xq, xsf, lg


class DepLamportPackGather(OpWrapper):
    """`xq, xsf, logits = dep_lamport_pack_gather(x, logits, global_scale, pack)`: every rank's
    `[R, H]` bf16 rows and their `[R, E]` bf16 router logits (contiguous) gathered rank-major over
    `pack`'s workspace (`PackGather`) as the NVFP4 codes `[W R, H / 2]` uint8 and linear block
    scales `[W R, H / 16]` uint8 at `global_scale` (fp32 `[1]`) of the rows with -0.0 lanes as +0.0,
    and the logits `[W R, E]` with -0.0 as +0.0, or with `pack.ownk` the rows' routing (top-10 and
    Renormalize weights of those logits, `[W R, 24]` int16 in `route_front_only(ids_in=True)`'s
    form). Every rank calls it at the same point with the same `R` (`pack.takes`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(world=world, hidden=2560, experts=512, rows=rows, ownk=ownk))
        for world, rows, ownk, why in (
            (4, 12, False, "dep4 3 requests a rank (a 4-GPU tray runs it)"),
            (8, 12, False, "dep8 3 requests a rank"),
            (8, 48, False, "dep8 12 requests a rank"),
            (8, 112, False, "dep8 28 requests a rank"),
            (4, 12, True, "owner top-k, dep4 3 requests a rank (a tray runs it)"),
            (8, 48, True, "owner top-k, dep8 12 requests a rank (a token a routing warp)"),
            (8, 72, True, "owner top-k, dep8 18 requests a rank"),
            (8, 96, True, "owner top-k, dep8 24 requests a rank"),
            (8, 112, True, "owner top-k, dep8 28 requests a rank"),
        )
    )

    note = """
    Source: one CuTe DSL kernel (`_pack_kernel`) over the host-index workspace of
    `dep_lamport_round_trip` (an int32 view of its buffers), compiled per rank at build. Its
    prototype ran the same statements at 8 ranks over two nodes of one rack, bitwise against the
    bf16 gather + route front quantization (0 differing bytes). The owner top-k form's prototype
    ran 8 ranks over two nodes the same way, its routing bitwise the route front's over the
    gathered logits (0 differing bytes, eagerly and replayed). Not certified here (no GPU run
    yet); the device-counter workspace is not served.
    """

    def __call__(self, x, logits, global_scale, pack: PackGather):
        return pack(x, logits, global_scale)

    def reference(self, xs_in_rank_order, logits_in_rank_order, global_scale, ownk=False):
        """Collective (the documented exception to mirroring `__call__`): every rank's rows and
        logits in rank order; `fp4_quantize` (linear scales) of the sanitized gathered rows, its
        codes as the transport delivers them (`transported_codes`), and the sanitized gathered
        logits, or with `ownk` their `renorm_moe_routing_op` routing as `routed_rows`."""
        from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import (
            moe_route_front__decode as _front,
        )

        rows = torch.cat([_rt._sanitized(x) for x in xs_in_rank_order])
        xq, xsf = _front.fp4_quantize(rows, global_scale, 16, False, False)
        lg = torch.cat([_rt._sanitized(g) for g in logits_in_rank_order])
        if ownk:
            ids, wts = _front.renorm_moe_routing_op(lg, _front.TOP_K, torch.bfloat16)
            lg = _front.routed_rows(ids, wts)
        return transported_codes(xq), xsf.reshape(rows.shape[0], -1), lg

    def is_valid(self, x, logits, global_scale, pack: PackGather) -> None:
        rows = x.shape[0]
        assert pack.takes(rows, x), (tuple(x.shape), x.dtype, x.stride())
        assert logits.dtype is torch.bfloat16 and logits.is_contiguous()
        assert tuple(logits.shape) == (rows, pack.experts)
        assert global_scale.dtype is torch.float32 and global_scale.numel() == 1


dep_lamport_pack_gather = DepLamportPackGather()
