# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tep4 decode MoE block's finalize and TP all-reduce in one fp32 Lamport launch.

Under attention TP every rank of the group holds every row, runs its expert window over them,
and the window partials are summed over the group. The stock chain is the trtllm-gen runner's
finalize (`finalizeKernel`: per row, the routing-weighted sum of the row's FC2 outputs in slot
order) and then the all-reduce. Here the runner is entered without its finalize, and one launch
does both on every rank of the 4-rank group, for `rows` rows of hidden 2560:

    local[t] = bf16( sum over slots k = 0..9 in order, skipping -1, of fp32(w[t, k]) * fp32(fc2[perm[t, k]]) )
    out[t]   = bf16( ((fp32(local_0[t]) + fp32(local_1[t])) + fp32(local_2[t])) + fp32(local_3[t]) )

The first line is `finalizeKernel`'s expression (a bf16 x bf16 product is exact in fp32, so the
FMA and the multiply-then-add give the same bits); the second is the one-shot all-reduce with
fp32 accumulation in rank order. A -0.0 lane is published as +0.0 (it is the not-ready marker),
as the stock Lamport kernels do. So the output is bitwise the finalize followed by that
all-reduce, except where every rank's partial of a lane is -0.0.

With a shared addend (`shared`, the launch compiled `with_shared`), each rank also adds a gated
fp32 term before its one rounding:

    local[t] = bf16( fma(sigmoid(fp32(g[t])), shared[t], <the slot-order sum above>) )

`shared[t]` `[rows, 2560]` fp32 is this rank's quarter of the shared expert (its down projection
over a quarter of the intermediate, fp32 out) and `g[t]` the bf16 gate logit; sigmoid is
`1 / (1 + ex2.approx(-g log2 e))` with an IEEE division. Summed over the group, the four quarters
are the whole shared expert gated once per rank: the layer's `routed + sigmoid(g) * shared`, in
fp32 up to the one rounding a rank, in place of the next site's bf16 shared expert and its
gate. Same precision, another work split (the down projection's K is split over the ranks).

Transport: three slots of `[4 ranks, 16 rows, 2560]` bf16 a rank, symmetric over the group,
filled with bf16 -0.0. The slot is chosen on the host (`slot`: the block's index among the
step's trunk MoE launches, mod 3) and so is baked into a captured graph; there is no device
counter and no last-CTA atomic. A writer stores its 16-byte vector into slot `[slot][my rank]
[row]` of the three peers (its own vector stays in registers); a reader polls its own slot for
the three peers' vectors (volatile loads of the high lane of each 32-bit word: a written word
never holds -0.0 there) and writes the marker back into what it read. Any three consecutive
launches use three different slots, and a rank can be at most one launch ahead of a peer's
read, so a slot is never rewritten before its reader has cleaned it.

Grid: one CTA a (row, `chunk` columns), `chunk / 8` threads of 8 bf16 lanes. PDL:
`griddepcontrol.wait` before any read; the dependents are released right after the publish, or
with `early` right after the wait (they still wait for this grid's completion before reading
its output).

Collective: every rank of the group launches the same kernel at the same point with the same
rows and slot, and every CTA of a launch must be resident on every rank at once (5 x 16 CTAs of
64 threads at most). A rank whose window holds no FC2 row (`[0, 2560]`, every slot -1) still
takes part: it launches over a one-row zero stand-in that no slot names, so its partial is 0 and
no load reads past an FC2 allocation. Routing weights at a 2-byte offset are copied to aligned storage
first (the kernel reads them in 32-bit pairs); the copy is rank-local, every rank still launches.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

HIDDEN = 2560
TOP_K = 10
WORLD = 4
#: Rows a launch serves: 1..4 requests of four verify rows.
MAX_ROWS = 16
SLOTS = 3
CHUNK = 512
CHUNKS = (256, 512, 1280)
#: Two bf16 -0.0 lanes (the not-ready marker) as one 32-bit word, and one as a 16-bit word.
_SENTINEL_WORD = -2147450880  # 0x80008000
_NEG_ZERO_BF16_I16 = -32768  # 0x8000


# ---------------------------------------------------------------------------- inline PTX ---
def _asm(res, operands, text, constraints, loc, ip):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


def _struct(n, ty):
    return llvm.StructType.get_literal([ty] * n)


def _unpack(res, n, cls, ty, loc, ip):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _ld_b32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [_v(addr, loc, ip)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def _ld_v4(addr: Int64, *, loc=None, ip=None):
    res = _asm(
        _struct(4, T.i32()),
        [_v(addr, loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return _unpack(res, 4, Int32, T.i32(), loc, ip)


@dsl_user_op
def _st_v4(addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, loc, ip)] + [_v(x, loc, ip) for x in (a, b, c, d)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _st_v4_volatile(
    addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None
) -> None:
    _asm(
        None,
        [_v(addr, loc, ip)] + [_v(x, loc, ip) for x in (a, b, c, d)],
        "st.volatile.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def _poll3(a0: Int64, a1: Int64, a2: Int64, *, loc=None, ip=None):
    """Spin until the three 16-byte vectors at a0 / a1 / a2 hold no marker (the high bf16 lane
    of every 32-bit word is checked: 32-bit stores are single-copy atomic and a written lane is
    never -0.0). Returns the 12 words."""
    loads = "".join(
        f"ld.volatile.global.v4.b32 {{${4 * i}, ${4 * i + 1}, ${4 * i + 2}, ${4 * i + 3}}}, "
        f"[${12 + i}];\n"
        for i in range(3)
    )
    checks = "".join(
        f"and.b32 t, ${w}, 0xFFFF0000;\nsetp.eq.u32 p, t, 0x80000000;\n"
        "selp.b32 t, 1, 0, p;\nor.b32 d, d, t;\n"
        for w in range(12)
    )
    text = (
        "{\n.reg .pred p;\n.reg .b32 t, d;\n"
        "TEPFINAR_POLL:\nmov.b32 d, 0;\n" + loads + checks + "setp.ne.u32 p, d, 0;\n"
        "@p bra TEPFINAR_POLL;\n}\n"
    )
    res = _asm(
        _struct(12, T.i32()),
        [_v(a0, loc, ip), _v(a1, loc, ip), _v(a2, loc, ip)],
        text,
        ",".join(["=r"] * 12 + ["l"] * 3),
        loc,
        ip,
    )
    return _unpack(res, 12, Int32, T.i32(), loc, ip)


@dsl_user_op
def _unpack_bf16x2(word: Int32, *, loc=None, ip=None):
    res = _asm(
        _struct(2, T.f32()),
        [_v(word, loc, ip)],
        "{\n.reg .b32 t;\nshl.b32 t, $2, 16;\nmov.b32 $0, t;\n"
        "and.b32 t, $2, 0xFFFF0000;\nmov.b32 $1, t;\n}\n",
        "=f,=f,r",
        loc,
        ip,
    )
    return _unpack(res, 2, Float32, T.f32(), loc, ip)


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [_v(lo, loc, ip), _v(hi, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _sanitize(word: Int32, *, loc=None, ip=None) -> Int32:
    """Each bf16 -0.0 lane of `word` as +0.0."""
    return Int32(
        _asm(
            T.i32(),
            [_v(word, loc, ip)],
            "{\n.reg .b32 t;\n.reg .pred p, q;\n"
            "and.b32 t, $1, 0xFFFF;\nsetp.eq.u32 p, t, 0x8000;\n"
            "and.b32 t, $1, 0xFFFF0000;\nsetp.eq.u32 q, t, 0x80000000;\n"
            "mov.b32 $0, $1;\n@p and.b32 $0, $0, 0xFFFF0000;\n@q and.b32 $0, $0, 0xFFFF;\n}\n",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(a, loc, ip), _v(b, loc, ip), _v(c, loc, ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _add(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [_v(a, loc, ip), _v(b, loc, ip)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip)
    )


@dsl_user_op
def _select(pred: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(pred, loc, ip), _v(a, loc, ip), _v(b, loc, ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}\n",
            "=f,r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_bf16_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """One bf16 at `addr` (2-byte aligned) as fp32."""
    return Float32(
        _asm(
            T.f32(),
            [_v(addr, loc, ip)],
            "{\n.reg .b16 h, z;\nld.global.b16 h, [$1];\nmov.b16 z, 0;\nmov.b32 $0, {z, h};\n}\n",
            "=f,l",
            loc,
            ip,
        )
    )


@dsl_user_op
def _sigmoid(g: Float32, *, loc=None, ip=None) -> Float32:
    """1 / (1 + 2^(-g log2 e)) in fp32 (ex2.approx, IEEE division)."""
    return Float32(
        _asm(
            T.f32(),
            [_v(g, loc, ip)],
            "{\n.reg .f32 t;\nmul.rn.f32 t, $1, 0fBFB8AA3B;\nex2.approx.ftz.f32 t, t;\n"
            "add.rn.f32 t, t, 0f3F800000;\ndiv.rn.f32 $0, 0f3F800000, t;\n}\n",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bits_f32(word: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(word, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc, ip))


@dsl_user_op
def _perm_check(index: Int32, rows: Int32, *, loc=None, ip=None):
    """`(valid, safe_index)`: valid = 0 <= index < rows; safe_index = index when valid, else 0."""
    res = _asm(
        _struct(2, T.i32()),
        [_v(index, loc, ip), _v(rows, loc, ip)],
        "{\n.reg .pred p;\nsetp.lt.u32 p, $2, $3;\nselp.b32 $0, 1, 0, p;\nselp.b32 $1, $2, 0, p;\n}\n",
        "=r,=r,r,r",
        loc,
        ip,
    )
    return _unpack(res, 2, Int32, T.i32(), loc, ip)


# ---------------------------------------------------------------------------- trace time ---
def _local_partial(
    k, fc2, fc2_row_bytes, fc2_rows, wts, perm, shared, gate, gate_row_bytes, row, col
):
    """This rank's finalized vector (4 words = 8 bf16 lanes) of (row, col..col+7), with the gated
    shared addend when compiled with one, sanitized."""
    top_k = k.top_k
    pbase = perm + Int64(row) * Int64(top_k * 4)
    wbase = wts + Int64(row) * Int64(top_k * 2)
    idx = [_ld_b32(pbase + Int64(4 * s)) for s in range(top_k)]
    wpairs = [_ld_b32(wbase + Int64(4 * p)) for p in range(top_k // 2)]
    weights = []
    for wp in wpairs:
        lo, hi = _unpack_bf16x2(wp)
        weights += [lo, hi]
    checks = [_perm_check(idx[s], fc2_rows) for s in range(top_k)]
    # Every slot's 16-byte load issues before the first add (an invalid slot loads row 0 and is
    # not added).
    xs = [
        _ld_v4(fc2 + Int64(checks[s][1]) * fc2_row_bytes + Int64(col) * Int64(2))
        for s in range(top_k)
    ]
    acc = [Float32(0.0)] * 8
    for s in range(top_k):
        for j in range(4):
            x_lo, x_hi = _unpack_bf16x2(xs[s][j])
            acc[2 * j] = _select(checks[s][0], _fma(weights[s], x_lo, acc[2 * j]), acc[2 * j])
            acc[2 * j + 1] = _select(
                checks[s][0], _fma(weights[s], x_hi, acc[2 * j + 1]), acc[2 * j + 1]
            )
    if k.has_shared:  # trace time: the compiled launch has the addend or not
        sbase = shared + Int64(row) * Int64(k.hidden * 4) + Int64(col) * Int64(4)
        sw = _ld_v4(sbase) + _ld_v4(sbase + Int64(16))
        sig = _sigmoid(_ld_bf16_f32(gate + Int64(row) * gate_row_bytes))
        acc = [_fma(sig, _bits_f32(sw[i]), acc[i]) for i in range(8)]
    return [_sanitize(_pack_bf16x2(acc[2 * j], acc[2 * j + 1])) for j in range(4)]


def _offset(k, slot, src_rank: int, row, col):
    return (
        (Int64(slot) * Int64(k.world) + Int64(src_rank)) * Int64(k.max_rows * k.hidden * 2)
        + Int64(row) * Int64(k.hidden * 2)
        + Int64(col) * Int64(2)
    )


def _publish(k, local, bufs, slot, row, col):
    off = _offset(k, slot, k.rank, row, col)
    for p in range(k.world):
        if p != k.rank:
            _st_v4_volatile(bufs[p] + off, local[0], local[1], local[2], local[3])


def _reduce_store_clean(k, local, bufs, slot, row, col, out):
    own = bufs[k.rank]
    peers = [p for p in range(k.world) if p != k.rank]
    addrs = [own + _offset(k, slot, p, row, col) for p in peers]
    got = _poll3(addrs[0], addrs[1], addrs[2])
    words = {k.rank: list(local)}
    for i, p in enumerate(peers):
        words[p] = [got[4 * i + j] for j in range(4)]
    red = [Float32(0.0)] * 8
    for r in range(k.world):  # rank order 0..3, the one-shot all-reduce's fp32 order
        for j in range(4):
            v_lo, v_hi = _unpack_bf16x2(words[r][j])
            red[2 * j] = _add(red[2 * j], v_lo)
            red[2 * j + 1] = _add(red[2 * j + 1], v_hi)
    o = [_pack_bf16x2(red[2 * j], red[2 * j + 1]) for j in range(4)]
    _st_v4(out + Int64(row) * Int64(k.hidden * 2) + Int64(col) * Int64(2), o[0], o[1], o[2], o[3])
    s = Int32(_SENTINEL_WORD)
    for a in addrs:  # self-clean: this thread is the one reader of these three vectors
        _st_v4_volatile(a, s, s, s, s)


class _Kernel:
    """Compile-time shape of one launch: the rank, the columns a CTA, PDL and its release
    point, and whether a gated fp32 shared addend joins the local sum."""

    def __init__(self, rank: int, chunk: int, use_pdl: bool, early: bool, has_shared: bool):
        assert HIDDEN % chunk == 0 and chunk % 8 == 0
        self.rank = int(rank)
        self.world = WORLD
        self.hidden = HIDDEN
        self.top_k = TOP_K
        self.max_rows = MAX_ROWS
        self.chunk = int(chunk)
        self.threads = self.chunk // 8
        self.use_pdl = bool(use_pdl)
        self.early = bool(use_pdl) and bool(early)
        self.has_shared = bool(has_shared)

    @cute.jit
    def __call__(
        self,
        fc2: Int64,
        fc2_row_bytes: Int64,
        fc2_rows: Int32,
        wts: Int64,
        perm: Int64,
        shared: Int64,
        gate: Int64,
        gate_row_bytes: Int64,
        out: Int64,
        b0: Int64,
        b1: Int64,
        b2: Int64,
        b3: Int64,
        slot: Int32,
        state: cute.Tensor,
        num_rows: Int32,
        stream: CUstream,
    ):
        # `state` (an unused int32 tensor) lets the TVM-FFI launcher bind the caller's stream.
        self.kernel(
            fc2,
            fc2_row_bytes,
            fc2_rows,
            wts,
            perm,
            shared,
            gate,
            gate_row_bytes,
            out,
            b0,
            b1,
            b2,
            b3,
            slot,
            state,
            num_rows,
        ).launch(
            grid=[self.hidden // self.chunk, num_rows, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        fc2: Int64,
        fc2_row_bytes: Int64,
        fc2_rows: Int32,
        wts: Int64,
        perm: Int64,
        shared: Int64,
        gate: Int64,
        gate_row_bytes: Int64,
        out: Int64,
        b0: Int64,
        b1: Int64,
        b2: Int64,
        b3: Int64,
        slot: Int32,
        state_tensor: cute.Tensor,
        num_rows: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, row, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        col = (cta * Int32(self.threads) + tidx) * Int32(8)
        bufs = [b0, b1, b2, b3]
        local = _local_partial(
            self, fc2, fc2_row_bytes, fc2_rows, wts, perm, shared, gate, gate_row_bytes, row, col
        )
        _publish(self, local, bufs, slot, row, col)
        if cutlass.const_expr(self.use_pdl and not self.early):
            cute.arch.griddepcontrol_launch_dependents()
        _reduce_store_clean(self, local, bufs, slot, row, col, out)


_COMPILED: dict = {}


def _compile(rank: int, chunk: int, use_pdl: bool, early: bool, has_shared: bool = False):
    """The compiled launch of one shape, once a process."""
    key = (int(rank), int(chunk), bool(use_pdl), bool(use_pdl) and bool(early), bool(has_shared))
    fn = _COMPILED.get(key)
    if fn is None:
        from cutlass.cute import runtime as cute_runtime

        kern = _Kernel(rank, chunk, use_pdl, early, has_shared)
        z64, z32 = Int64(0), Int32(0)
        state_fake = cute_runtime.make_fake_compact_tensor(
            Int32, (4,), stride_order=(0,), assumed_align=16
        )
        fn = cute.compile(
            kern,
            z64, z64, z32, z64, z64, z64, z64, z64, z64, z64, z64, z64, z64, z32, state_fake, z32,
            stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )  # fmt: skip
        _COMPILED[key] = fn
    return fn


@dataclass
class TepFinar:
    """One rank's workspace and compiled launches: `buffer` this rank's `[3, 4, 16 * 2560]` bf16
    symmetric slots, `peers` the four ranks' slot base addresses in rank order, `fn_shared` the
    launch with the gated shared addend (None unless built `with_shared`)."""

    rank: int
    chunk: int
    use_pdl: bool
    early: bool
    buffer: torch.Tensor
    peer_buffers: list
    peers: tuple
    state: torch.Tensor
    fn: object
    #: `[1, 2560]` bf16 zeros: the FC2 operand of a rank whose window is empty.
    empty_fc2: torch.Tensor | None = None
    fn_shared: object = None

    @property
    def workspace_bytes(self) -> int:
        return self.buffer.numel() * self.buffer.element_size()

    def takes(self, fc2: torch.Tensor, wts: torch.Tensor, perm: torch.Tensor, rows: int) -> bool:
        """Whether the launch takes these operands (host values every rank shares at a step)."""
        return (
            1 <= rows <= MAX_ROWS
            and fc2.dtype == torch.bfloat16
            and fc2.dim() == 2
            and fc2.shape[1] == HIDDEN
            and fc2.stride(1) == 1
            and (fc2.stride(0) * 2) % 16 == 0
            and fc2.data_ptr() % 16 == 0
            and wts.dtype == torch.bfloat16
            and tuple(wts.shape) == (rows, TOP_K)
            and wts.is_contiguous()
            and perm.dtype == torch.int32
            and tuple(perm.shape) == (rows, TOP_K)
            and perm.is_contiguous()
        )

    def takes_shared(self, shared, gate, rows: int) -> bool:
        """Whether the shared-addend launch takes `shared` `[rows, 2560]` fp32 (dense) and its
        gate logit `gate` `[rows, 1]` bf16 at any row pitch."""
        return (
            self.fn_shared is not None
            and shared.dtype == torch.float32
            and tuple(shared.shape) == (rows, HIDDEN)
            and shared.is_contiguous()
            and shared.data_ptr() % 16 == 0
            and gate.dtype == torch.bfloat16
            and gate.dim() == 2
            and tuple(gate.shape) == (rows, 1)
            and gate.data_ptr() % 2 == 0
        )

    def __call__(self, fc2, wts, perm, rows: int, slot: int, shared=None, gate=None):
        with_shared = shared is not None or gate is not None
        if not (
            self.takes(fc2, wts, perm, rows)
            and 0 <= slot < SLOTS
            and (not with_shared or self.takes_shared(shared, gate, rows))
        ):
            raise RuntimeError(
                "tep_finalize_allreduce: operands outside the compiled layout: fc2 "
                f"{fc2.dtype} {tuple(fc2.shape)} strides {tuple(fc2.stride())}, wts {wts.dtype} "
                f"{tuple(wts.shape)}, perm {perm.dtype} {tuple(perm.shape)}, rows {rows}, slot {slot}"
                + (
                    f", shared {shared.dtype} {tuple(shared.shape)}, gate {gate.dtype} "
                    f"{tuple(gate.shape)}, shared launch built {self.fn_shared is not None}"
                    if with_shared and shared is not None and gate is not None
                    else (", shared and gate must be given together" if with_shared else "")
                )
            )
        if fc2.shape[0] == 0:
            # No FC2 row to read: every slot is -1, the partial is 0, and the rank still
            # publishes and reduces with its peers.
            if self.empty_fc2 is None or self.empty_fc2.device != fc2.device:
                self.empty_fc2 = torch.zeros((1, HIDDEN), dtype=torch.bfloat16, device=fc2.device)
            fc2 = self.empty_fc2
        if wts.data_ptr() % 4:
            wts = wts.clone()  # the kernel loads the weights in 32-bit pairs
        out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=fc2.device)
        (self.fn_shared if with_shared else self.fn)(
            fc2.data_ptr(),
            fc2.stride(0) * 2,
            fc2.shape[0],
            wts.data_ptr(),
            perm.data_ptr(),
            shared.data_ptr() if with_shared else 0,
            gate.data_ptr() if with_shared else 0,
            gate.stride(0) * 2 if with_shared else 0,
            out.data_ptr(),
            self.peers[0],
            self.peers[1],
            self.peers[2],
            self.peers[3],
            int(slot),
            self.state,
            int(rows),
        )
        return out


def build(
    rank: int, use_pdl: bool, early: bool = False, chunk: int = CHUNK, with_shared: bool = False
) -> TepFinar:
    """Pin the three slots over the 4-rank torch process group (WORLD, which must be the TP group
    of one NVLink domain) and compile the launch (and with `with_shared` the shared-addend launch,
    which shares the slots). Collective: every rank calls it with the same arguments."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    assert chunk in CHUNKS, chunk
    assert dist.is_initialized() and dist.get_world_size() == WORLD and dist.get_rank() == rank
    dev = torch.device("cuda", torch.cuda.current_device())
    buffer = symm_mem.empty((SLOTS, WORLD, MAX_ROWS * HIDDEN), dtype=torch.bfloat16, device=dev)
    # Every slot starts not-ready, or the first exchange reads uninitialized memory as data.
    buffer.view(torch.int16).fill_(_NEG_ZERO_BF16_I16)
    group = dist.group.WORLD
    handle = symm_mem.rendezvous(buffer, group=getattr(group, "group_name", group))
    peer_buffers = [handle.get_buffer(p, buffer.shape, buffer.dtype) for p in range(WORLD)]
    peers = tuple(int(b.data_ptr()) for b in peer_buffers)
    assert all(p % 16 == 0 for p in peers), peers
    state = torch.zeros(4, dtype=torch.int32, device=dev)
    fn = _compile(rank, chunk, use_pdl, early)
    fn_shared = _compile(rank, chunk, use_pdl, early, True) if with_shared else None
    torch.cuda.synchronize(dev)
    # Every rank holds its buffers and the launch before any of them exchanges a row.
    ok = torch.ones(1, dtype=torch.int32, device=dev)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    torch.cuda.synchronize(dev)
    early = bool(use_pdl) and bool(early)
    empty_fc2 = torch.zeros((1, HIDDEN), dtype=torch.bfloat16, device=dev)
    return TepFinar(
        rank,
        int(chunk),
        bool(use_pdl),
        early,
        buffer,
        peer_buffers,
        peers,
        state,
        fn,
        empty_fc2=empty_fc2,
        fn_shared=fn_shared,
    )


def _finalize(fc2, wts, perm, shared=None, gate=None) -> torch.Tensor:
    """`finalizeKernel` over one rank's window: fp32, slot order, then (with `shared`) plus
    sigmoid(gate) x shared in fp32, one bf16 rounding, -0.0 as +0.0 (the transport's marker).
    The sigmoid is torch's: the launch's `ex2.approx` form gives the same fp32 at the gate logits
    the tests drive (0, large positive and large negative: 0.5, 1 and 0)."""
    acc = torch.zeros((perm.shape[0], fc2.shape[1]), dtype=torch.float32, device=fc2.device)
    for k in range(perm.shape[1] if fc2.shape[0] else 0):  # an empty window adds nothing
        idx = perm[:, k].long()
        live = (idx >= 0) & (idx < fc2.shape[0])
        rows = fc2[idx.clamp(0, fc2.shape[0] - 1)].float()
        acc = torch.where(live[:, None], acc + wts[:, k : k + 1].float() * rows, acc)
    if shared is not None:
        acc = acc + torch.sigmoid(gate.float()) * shared.float()
    local = acc.to(torch.bfloat16)
    return torch.where(
        local.view(torch.int16) == _NEG_ZERO_BF16_I16, torch.zeros_like(local), local
    )


class TepFinalizeAllReduce(OpWrapper):
    """`tep_finalize_allreduce(fc2, wts, perm, rows, slot, finar, shared=None, gate=None)`: over
    `finar` (`build`), every rank's window FC2 rows `fc2` `[P, 2560]` bf16 (the runner's permuted
    rows, entered without its finalize), the routing weights `wts` `[rows, 10]` bf16 and the slot
    -> FC2 row table `perm` `[rows, 10]` int32 (-1 for a slot another rank's window serves), the
    finalized rows summed over the 4-rank group in fp32, rank order, `[rows, 2560]` bf16, on every
    rank. With `shared` `[rows, 2560]` fp32 and its gate logit `gate` `[rows, 1]` bf16 (a column
    view; `finar` built `with_shared`) each rank adds sigmoid(gate) x shared to its partial before
    its rounding. Every rank calls it at the same point with the same `rows`, `slot` (cycling 0,
    1, 2 over consecutive launches) and the same choice of the shared addend."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(world=WORLD, hidden=HIDDEN, rows=rows))
        for rows, why in (
            (4, "tep4 MTP-3 verify: 1 request"),
            (8, "tep4 MTP-3 verify: 2 requests"),
            (12, "tep4 MTP-3 verify: 3 requests"),
            (MAX_ROWS, "tep4 MTP-3 verify: 4 requests, the launch's last"),
        )
    ) + tuple(
        Cell(why=why, spec=dict(world=WORLD, hidden=HIDDEN, rows=rows, shared=True))
        for rows, why in (
            (12, "tep4 MTP-3 verify: 3 requests, shared-expert quarter in the launch"),
            (MAX_ROWS, "tep4 MTP-3 verify: 4 requests, shared-expert quarter in the launch"),
        )
    )

    note = """
    Source: one CuTe DSL kernel compiled per rank at `build` (`--enable-tvm-ffi`), over torch
    symmetric memory on the 4-rank torch group of one NVLink domain. Certified on tep4 GB300
    only: 4 ranks, hidden 2560, top-10, 1 to `MAX_ROWS` rows. Not certified: other world sizes,
    groups across nodes, a stream holding SMs that a launch's resident CTAs wait for.

    The shared-addend launch (cells with `shared`) reads `shared` and `gate` after its PDL wait,
    like every other operand. Its sigmoid is `ex2.approx`-based: bitwise torch's at the gate
    logits the cells drive (0, large positive, large negative), within a few fp32 ulp elsewhere,
    which the one bf16 rounding a rank mostly absorbs.
    """

    def __call__(
        self, fc2, wts, perm, rows: int, slot: int, finar: TepFinar, shared=None, gate=None
    ) -> torch.Tensor:
        return finar(fc2, wts, perm, rows, slot, shared, gate)

    def reference(self, operands_in_rank_order) -> torch.Tensor:
        """Collective (the documented exception to mirroring `__call__`): every rank's
        `(fc2, wts, perm)` or `(fc2, wts, perm, shared, gate)` in rank order; the sum every rank
        gets."""
        parts = [_finalize(*ops) for ops in operands_in_rank_order]
        acc = torch.zeros_like(parts[0], dtype=torch.float32)
        for p in parts:
            acc = acc + p.float()
        return acc.to(torch.bfloat16)

    def is_valid(
        self, fc2, wts, perm, rows: int, slot: int, finar: TepFinar, shared=None, gate=None
    ) -> None:
        assert finar.takes(fc2, wts, perm, rows) and 0 <= slot < SLOTS
        assert (shared is None) == (gate is None), "shared and gate come together"
        assert shared is None or finar.takes_shared(shared, gate, rows)


tep_finalize_allreduce = TepFinalizeAllReduce()
