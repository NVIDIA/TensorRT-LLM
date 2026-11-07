# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decode attention-residual mix + RMSNorm (+ residual add) with a PDL prologue.

Runs on SM100, SM103 and SM107: the kernel uses no tensor-core instructions and portable cluster
sizes.

For ``M`` tokens of hidden size ``H``, ``N - 1`` snapshot rows ``block[n]`` and the layer row
``x``, one launch computes, per token,

* ``rsigma_n = rsqrt(mean(v_n^2) + eps)``, ``logit_n = rsigma_n * <v_n, res_w * rms_w>``,
* ``p = softmax(logit)``, ``mixed = bf16(sum_n p_n v_n)``,
* ``out = bf16(bf16(mixed * rsqrt(mean(mixed^2) + out_eps)) * out_w)``,

where ``x`` is the given row, ``bf16(prefix + addend)`` (``trtllm::attn_res_add_rmsnorm_fwd``'s
fused add), or ``bf16(residual + bf16(routed + shared))`` (a MoE layer's residual update, the
two BF16 adds of its routed and shared outputs); the updated prefix ``x`` is stored as well in
the last two cases, bit-identical to those adds.

The kernel is launched with programmatic dependent launch and does everything that does not
depend on earlier kernels of the step before ``griddepcontrol.wait``: it loads the weights and
the snapshot rows, reduces the snapshots' statistics over the cluster and folds their share of
the softmax and of the mixture (``sum_n exp(logit_n - m) v_n``). After the wait it loads the
layer row (and, with an addend, the prefix sum and the addends), reduces its two statistics (one
cluster round), rescales the mixture, reduces its sum of squares (a second round) and stores.
Cluster partials travel by ``st.async`` into every peer's shared memory and are counted by a
per-round mbarrier (one one-way message per round).

A cluster of ``c`` CTAs (:func:`cluster_size`) serves one token, one 16-byte vector per thread.
The statistics are summed in another order than the C++ attention-residual kernels, so the
normalized output can differ from theirs by one or two BF16 rounding steps in a small fraction
of the elements; the updated prefix sum, an elementwise add, is bit-identical.

The addend of ``bf16(prefix + addend)`` may also be the result of a pushed two-shot MNNVL
all-reduce left in its workspace (``mnnvl_pushed_allreduce.run_finish_in_place``): the kernel
then polls each 16-byte vector from the broadcast stage of that call's buffer after
``griddepcontrol.wait`` (``mnnvl_pushed_allreduce.poll_result_v4``), the bits the gather would
have stored, so the results are bitwise those of the gathered tensor. Or the kernel finishes a
pushed call itself (``finish``, its finish left pending): one-shot transport, it sums its token's
slots of every rank in the C++ order; two-shot, on each owned row it reduces the scatter slots and
multicasts the row (the owned rows' CTA groups first in the grid), on the others it polls the
broadcast row. It also re-arms the previous call's buffer and advances the rotation state
(``mnnvl_pushed_allreduce``), so the addend has the bits of the finished all-reduce. With
``early_poll`` (one-shot) the kernel finishes the call and computes the whole mix before its
``griddepcontrol.wait``: it becomes resident once the pushing producer has passed its own wait,
and only its stores and the state advance follow the wait.

Optionally the kernel also stores the MXFP8 form of its normalized output: E4M3 codes and one
UE8M0 exponent per 1x32 group in the R128c4 scale layout, the bits the FP8 low-M GEMM's
in-kernel quantization makes of the BF16 output, so a projection can consume it instead of
quantizing (``fp8_block_scale_low_m_gemm``).

PDL contract (with ``TRTLLM_ENABLE_PDL``, the default; without it the kernel is launched and
runs like any other). Before its wait the kernel reads only the weights and, by default, the
snapshot rows (with ``early_poll`` also the prefix sum and the pushed slots). That is safe while
the snapshot rows were not written by the kernel launched immediately before this one: some
kernel in between must start its dependents only after its predecessors' writes are visible (a
launch without PDL, or a PDL kernel whose ``launch_dependents`` follows its
``griddepcontrol.wait``), so that this grid cannot start before the writer's grid has completed.
Where the immediately preceding kernel writes the snapshot rows (the copy that slices the
snapshot bank at the first token-sliced layer of ``helix_a2a``), the caller selects
``snapshots_after_wait``: the kernel then loads the snapshot rows after its wait. With
``early_poll`` the producer releases this grid only after its own wait, so every earlier grid,
the prefix sum's writer and the previous call's readers included, has completed by then. The
kernel releases its dependent grid right after its own ``griddepcontrol.wait``, so the next
kernel (the decode projection GEMM or the router) can become resident and stream its weights
while this one finishes; that is safe for every dependent that, as PDL requires, writes no global
memory and reads no output of this kernel before its own ``griddepcontrol.wait``.

``entry_trigger`` releases the dependent at kernel entry instead and prefers the largest
shared-memory carve-out, so that a projection which holds its whole weight slice on chip before
its wait (``fp8_block_scale_low_m_gemm.weights_on_chip``) becomes resident and streams its weights
while this kernel and the grid before it still run, also on the SMs of this kernel's CTAs. The
dependent, and the grids it releases before its own wait, must then read before their wait only
memory no grid still running writes (weights, and for the KDA verify kernel the state, conv
windows and replay rows of the previous step).
"""

from __future__ import annotations

import functools
import itertools
import struct
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from . import fp8_block_scale_low_m_gemm as low_m_gemm
from . import mnnvl_pushed_allreduce as pushed_ar
from .raw_ptx import asm as _asm_se
from .raw_ptx import asm_pure as _asm
from .raw_ptx import cluster_arrive_relaxed as _cluster_arrive_relaxed
from .raw_ptx import fence_mbar_init as _fence_mbar_init
from .raw_ptx import fv as _fv
from .raw_ptx import iv as _iv
from .raw_ptx import lds_f32 as _lds_f32
from .raw_ptx import lv as _lv
from .raw_ptx import mbar_expect_tx as _mbar_arrive_expect_tx
from .raw_ptx import mbar_init as _mbar_init
from .utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait

#: Candidate counts (snapshots + the layer row) the kernel is instantiated for.
MIN_CANDIDATES, MAX_CANDIDATES = 2, 12
#: Largest token count the kernel serves.
MAX_TOKENS = 192
#: CTAs per token (:func:`cluster_size`).
CLUSTER_SIZES = (4, 8)
#: Inputs of the layer row: the row itself; ``bf16(prefix + addend)`` (the C++ fused add);
#: ``bf16(residual + bf16(routed + shared))`` (the MoE residual update).
ADD_NONE, ADD_PREFIX, ADD_MOE = 0, 1, 2

_LOG2E = "0f3FB8AA3B"
_NEG_FLT_MAX = "0fFF7FFFFF"


def _f32_immediate(value: float) -> str:
    """PTX hexadecimal immediate of the float32 ``value``."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", value))[0]


# ---------------------------------------------------------------------------
# Inline PTX helpers (the fast-math instructions of the C++ attention-residual kernels)
# ---------------------------------------------------------------------------


def _f32_op(text, *args, loc=None, ip=None):
    constraints = "=f," + ",".join("f" for _ in args)
    return Float32(
        _asm(T.f32(), [_fv(a, loc=loc, ip=ip) for a in args], text, constraints, loc=loc, ip=ip)
    )


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return _f32_op("fma.rn.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    return _f32_op("add.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _add_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("add.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _mul_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("mul.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _zero_f32(*, loc=None, ip=None):
    return Float32(_asm(T.f32(), [], "mov.b32 $0, 0f00000000;", "=f", loc=loc, ip=ip))


@dsl_user_op
def _rsigma(sum_sq, eps, hidden: int, *, loc=None, ip=None):
    """``rsqrtf(sum_sq / hidden + eps)`` as nvcc --use_fast_math emits it."""
    return _f32_op(
        "{\n.reg .f32 t;\n"
        f"div.approx.ftz.f32 t, $1, {_f32_immediate(hidden)};\n"
        "add.ftz.f32 t, t, $2;\nrsqrt.approx.ftz.f32 $0, t;\n}",
        sum_sq,
        eps,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _max_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("max.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _neg_flt_max(*, loc=None, ip=None):
    return Float32(_asm(T.f32(), [], f"mov.b32 $0, {_NEG_FLT_MAX};", "=f", loc=loc, ip=ip))


@dsl_user_op
def _rcp_ftz(a, *, loc=None, ip=None):
    return _f32_op("rcp.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def _round_bf16(a, *, loc=None, ip=None):
    """``float(bf16_rn(a))``."""
    return _f32_op(
        "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\ncvt.f32.bf16 $0, t;\n}", a, loc=loc, ip=ip
    )


@dsl_user_op
def _shfl_idx(value, lane: int, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.idx.b32 $0, $1, {lane}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


# Unique PTX labels of the spin loops in :func:`_mbar_wait_cluster`.
_wait_labels = itertools.count(1)


@dsl_user_op
def _mbar_wait_cluster(addr, *, loc=None, ip=None):
    """Wait for phase 0 of a local mbarrier completed by peers' ``st.async`` (cluster acquire)."""
    lab = f"K3AR_WAIT{next(_wait_labels)}"
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\n"
        f"{lab}:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$0], 0;\n"
        f"@!p bra {lab};\n}}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_b32(local_addr, rank, local_mbar, a, *, loc=None, ip=None):
    _asm_se(
        None,
        [
            _iv(local_addr, loc=loc, ip=ip),
            _iv(rank, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
            _fv(a, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra, rm;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "mapa.shared::cluster.u32 rm, $2, $1;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], $3, [rm];\n}",
        "r,r,r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cluster_wait(*, loc=None, ip=None):
    _asm_se(None, [], "barrier.cluster.wait.acquire.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _ldg_v4(addr, nc: bool, *, loc=None, ip=None):
    """16-byte global load of four packed bf16x2 words."""
    space = "global.nc" if nc else "global"
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        f"ld.{space}.v4.b32 {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ldg_v4_if(valid, addr, nc: bool, *, loc=None, ip=None):
    """:func:`_ldg_v4` predicated on ``valid != 0`` (zeros otherwise)."""
    space = "global.nc" if nc else "global"
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $4, 0;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        f"@p ld.{space}.v4.b32 {{$0, $1, $2, $3}}, [$5];\n}}",
        "=&r,=&r,=&r,=&r,r,l",
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _stg_v4_if(valid, addr, words, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)]
        + [_iv(w, loc=loc, ip=ip) for w in words],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.v4.b32 [$1], {$2, $3, $4, $5};\n}",
        "r,l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _unpack_bf16x2(word, *, loc=None, ip=None):
    """``(float(lo), float(hi))`` of a packed bf16x2 word."""
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nshl.b32 t, $2, 16;\nmov.b32 $0, t;\n"
        "and.b32 t, $2, 0xffff0000;\nmov.b32 $1, t;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """bf16x2 word ``{bf16_rn(hi), bf16_rn(lo)}``."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(hi, loc=loc, ip=ip), _fv(lo, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_xor_b32(value, offset: int, *, loc=None, ip=None):
    return Int32(
        _asm_se(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _stg_v2_if(valid, addr, w0, w1, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)]
        + [_iv(w, loc=loc, ip=ip) for w in (w0, w1)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.v2.b32 [$1], {$2, $3};\n}",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u8_if(valid, addr, value, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.u8 [$1], $2;\n}",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _shfl_xor(value, offset: int, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _selp_b32(cond, a, b, *, loc=None, ip=None):
    """32-bit ``a if cond != 0 else b``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(cond, loc=loc, ip=ip), _iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _selp(cond, a, b, *, loc=None, ip=None):
    """``a if cond != 0 else b``."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(cond, loc=loc, ip=ip), _fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}",
            "=f,r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ex2_scaled(x, *, loc=None, ip=None):
    """``exp(x)`` as ``ex2.approx.ftz(x * log2(e))``."""
    return _f32_op(
        f"{{\n.reg .f32 t;\nmul.ftz.f32 t, $1, {_LOG2E};\nex2.approx.ftz.f32 $0, t;\n}}",
        x,
        loc=loc,
        ip=ip,
    )


def _warp_transpose_sum(values, lane):
    """Sums every value over the warp in ``len(values) - 1 + 5 - log2(len)`` shuffles.

    ``len(values)`` must be a power of two <= 32. Afterwards lane ``l`` holds the warp total of
    ``values[l >> (5 - log2(len(values)))]``.
    """
    offset = 16
    while len(values) > 1:
        half = len(values) // 2
        upper = Int32((lane & offset) != 0)
        reduced = []
        for i in range(half):
            mine = _selp(upper, values[i + half], values[i])
            send = _selp(upper, values[i], values[i + half])
            reduced.append(_add(mine, _shfl_xor(send, offset)))
        values = reduced
        offset //= 2
    total = values[0]
    while offset >= 1:
        total = _add(total, _shfl_xor(total, offset))
        offset //= 2
    return total


def _warp_sum(value):
    for offset in (16, 8, 4, 2, 1):
        value = _add(value, _shfl_xor(value, offset))
    return value


def _warp_max(value):
    for offset in (16, 8, 4, 2, 1):
        value = _max_ftz(value, _shfl_xor(value, offset))
    return value


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def supports_hidden_size(hidden: int) -> bool:
    """Hidden sizes the kernel serves with every cluster size: each CTA owns a multiple of four
    16-byte vectors (whole 1x32 MXFP8 groups) and a token has at most 32 warp partials."""
    if hidden <= 0 or hidden % (8 * 4 * max(CLUSTER_SIZES)):
        return False
    return all(c * -(-hidden // (8 * c * 32)) <= 32 for c in CLUSTER_SIZES)


class KimiK3AttnResKernel:
    """A cluster of ``c`` CTAs per token, one 16-byte vector per thread.

    Each CTA owns ``hidden / c`` contiguous elements and each thread eight of them (one 16-byte
    load per row). With ``in_place_addend`` the addend is the in-place result of a pushed MNNVL
    all-reduce, ``add_addr`` / ``add2_addr`` the workspace's local buffer and state flags; with
    ``finish`` the kernel finishes that pushed call (``mc_addr`` / ``ar_rank``: the workspace's
    multicast base and this rank), see :class:`AttnResVariant`.

    Args:
        variant: The compiled variant (:class:`AttnResVariant`).
        use_pdl: Launch with programmatic dependent launch (``griddepcontrol`` wait and
            release); without it the kernel waits for the previous grid like any launch.
    """

    def __init__(self, variant: AttnResVariant, use_pdl: bool = TRTLLM_ENABLE_PDL):
        variant.validate()
        finish = variant.finish
        self.mxfp8 = variant.mxfp8
        self.in_place_addend = variant.in_place_addend
        self.finish_world = finish.world if finish is not None else 0
        self.finish_two_shot = finish is not None and finish.two_shot
        self.early_poll = finish is not None and finish.early_poll
        self.finish_chunk = (
            (finish.chunk or pushed_ar.reduce_chunk(finish.world, True))
            if finish is not None
            else 0
        )
        self.finish_sanitize = finish is not None and finish.sanitize
        self.finish_all_gather = finish is not None and finish.all_gather
        self.entry_trigger = variant.entry_trigger
        self.snapshots_after_wait = variant.snapshots_after_wait
        self.use_pdl = use_pdl
        num_candidates, cluster, hidden = variant.num_candidates, variant.cluster, variant.hidden
        self.n = num_candidates
        self.k = num_candidates - 1
        self.c = cluster
        self.hidden = hidden
        self.add_mode = variant.add_mode
        self.vecs = hidden // 8 // cluster  # 16-byte vectors per CTA, one per thread
        self.threads = (self.vecs + 31) // 32 * 32
        self.warps = self.threads // 32
        self.parts = cluster * self.warps  # warp partials per token
        assert self.parts <= 32
        # Snapshot statistics (sq, dot) per warp, padded to a power of two for the transpose sum.
        self.nvals = 2 * self.k
        self.width = 1 << (self.nvals - 1).bit_length()
        self.shift = 5 - (self.width.bit_length() - 1)

    @cute.jit
    def __call__(
        self,
        layer_addr: Int64,
        add_addr: Int64,
        add2_addr: Int64,
        block_addr: Int64,
        res_addr: Int64,
        rms_addr: Int64,
        out_w_addr: Int64,
        out_addr: Int64,
        upd_addr: Int64,
        xq_addr: Int64,
        xs_addr: Int64,
        mc_addr: Int64,
        ar_rank: Int32,
        m: Int32,
        eps: Float32,
        out_eps: Float32,
        stream: cuda.CUstream,
    ):
        parts, nvals = self.parts, self.nvals

        @cute.struct
        class SharedStorage:
            snap: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, parts * nvals], 16]
            last: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, parts * 2], 16]
            outsq: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, parts], 16]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 3], 16]

        self.shared_storage = SharedStorage
        kernel = self.kernel(
            layer_addr,
            add_addr,
            add2_addr,
            block_addr,
            res_addr,
            rms_addr,
            out_w_addr,
            out_addr,
            upd_addr,
            xq_addr,
            xs_addr,
            mc_addr,
            ar_rank,
            m,
            eps,
            out_eps,
        )
        kernel.launch(
            grid=(m * self.c, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.c, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
            preferred_smem_carveout=100 if self.entry_trigger else None,
        )

    @cute.kernel
    def kernel(
        self,
        layer_addr: Int64,
        add_addr: Int64,
        add2_addr: Int64,
        block_addr: Int64,
        res_addr: Int64,
        rms_addr: Int64,
        out_w_addr: Int64,
        out_addr: Int64,
        upd_addr: Int64,
        xq_addr: Int64,
        xs_addr: Int64,
        mc_addr: Int64,
        ar_rank: Int32,
        m: Int32,
        eps: Float32,
        out_eps: Float32,
    ):
        c, k, parts, nvals = self.c, self.k, self.parts, self.nvals
        warps, shift, hidden = self.warps, self.shift, self.hidden
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        snap_smem = storage.snap.data_ptr().toint()
        last_smem = storage.last.data_ptr().toint()
        outsq_smem = storage.outsq.data_ptr().toint()
        mbar_smem = storage.mbar.data_ptr().toint()

        if cutlass.const_expr(self.entry_trigger and self.use_pdl):
            griddepcontrol_launch_dependents()
        token = bidx // c
        rank = bidx - token * c
        if cutlass.const_expr(self.finish_two_shot):
            token = pushed_ar.two_shot_token(token, m, ar_rank, self.finish_world)
        if tidx == 0:
            for r in cutlass.range_constexpr(3):
                _mbar_init(mbar_smem + r * 8, 1)
            _fence_mbar_init()
        _cluster_arrive_relaxed()

        valid = Int32(1)
        if cutlass.const_expr(self.vecs != self.threads):
            valid = Int32(tidx < self.vecs)
        col = Int64(rank * self.vecs + tidx) * 16
        row = Int64(token) * (hidden * 2)

        def load(addr, nc):
            if cutlass.const_expr(self.vecs != self.threads):
                return _ldg_v4_if(valid, addr, nc)
            return _ldg_v4(addr, nc)

        def unpack(words):
            out = []
            for w in words:
                lo, hi = _unpack_bf16x2(w)
                out.append(lo)
                out.append(hi)
            return out

        def wait_and_release():
            # Wait for the previous grid, then release the dependent grid (entry_trigger: released
            # at kernel entry).
            if cutlass.const_expr(self.use_pdl):
                griddepcontrol_wait()
                if cutlass.const_expr(not self.entry_trigger):
                    griddepcontrol_launch_dependents()

        # ---- prologue: everything that does not depend on the previous grid ----
        res_w = unpack(load(res_addr + col, True))
        rms_w = unpack(load(rms_addr + col, True))
        out_w = load(out_w_addr + col, True)
        if cutlass.const_expr(self.snapshots_after_wait):
            wait_and_release()
        plane = Int64(m) * (hidden * 2)
        # Written by the previous grid with ``snapshots_after_wait``: coherent loads then.
        snap_nc = not self.snapshots_after_wait
        snaps = [load(block_addr + plane * n + row + col, snap_nc) for n in range(k)]
        q = [_mul_ftz(res_w[e], rms_w[e]) for e in range(8)]

        snap_f = [unpack(snaps[n]) for n in range(k)]
        stats = []
        for n in cutlass.range_constexpr(k):
            sq = _zero_f32()
            dot = _zero_f32()
            for e in cutlass.range_constexpr(8):
                sq = _fma(snap_f[n][e], snap_f[n][e], sq)
                dot = _fma(snap_f[n][e], q[e], dot)
            stats.append(sq)
            stats.append(dot)
        for _ in cutlass.range_constexpr(self.width - nvals):
            stats.append(_zero_f32())
        warp_stat = _warp_transpose_sum(stats, lane)
        vidx = lane >> shift
        sub = lane & ((1 << shift) - 1)
        _cluster_wait()
        if tidx == 0:
            _mbar_arrive_expect_tx(mbar_smem, parts * nvals * 4)
            _mbar_arrive_expect_tx(mbar_smem + 8, parts * 2 * 4)
            _mbar_arrive_expect_tx(mbar_smem + 16, parts * 4)
        slot = snap_smem + ((rank * warps + warp) * nvals + vidx) * 4
        for r in cutlass.range_constexpr(c):
            if vidx < nvals:
                if sub == r % (1 << shift):
                    _st_async_b32(slot, r, mbar_smem, warp_stat)
        _mbar_wait_cluster(mbar_smem)

        # Snapshot totals (lane n < k), logits and the snapshot part of the softmax.
        logit = _neg_flt_max()
        if lane < k:
            tot_sq = _zero_f32()
            tot_dot = _zero_f32()
            for j in cutlass.range_constexpr(parts):
                tot_sq = _add(tot_sq, _lds_f32(snap_smem + (j * nvals + 2 * lane) * 4))
                tot_dot = _add(tot_dot, _lds_f32(snap_smem + (j * nvals + 2 * lane + 1) * 4))
            logit = _mul_ftz(tot_dot, _rsigma(tot_sq, eps, hidden))
        snap_max = _warp_max(logit)
        weight = _zero_f32()
        if lane < k:
            weight = _ex2_scaled(_add(logit, -snap_max))
        snap_sum = _warp_sum(weight)
        premix = [_zero_f32() for _ in range(8)]
        for n in cutlass.range_constexpr(k):
            w_n = _shfl_idx(weight, n)
            for e in cutlass.range_constexpr(8):
                premix[e] = _fma(w_n, snap_f[n][e], premix[e])

        # ---- wait for the previous grid, then release the dependent grid (early_poll: after
        # the mix, before the first store; snapshots_after_wait: in the prologue) ----
        if cutlass.const_expr(not self.early_poll and not self.snapshots_after_wait):
            wait_and_release()

        # The prefix sum is older than the previous grid: its load is issued before the slot
        # polls and lands while they spin (before the grid wait only with early_poll).
        prefix = []
        if cutlass.const_expr(self.add_mode != ADD_NONE):
            prefix = load(layer_addr + row + col, False)
        state = None
        finished = []
        arrivals = Int32(0)
        if cutlass.const_expr(self.finish_world > 0):
            # This kernel finishes the pushed call: the state is this call's (its producer read it
            # after its own wait), the previous call's buffer is re-armed while the rows travel.
            # Threads past the row work on the row's first vector, do not store and use zeros.
            world = self.finish_world
            state = pushed_ar.lamport_state(add2_addr)
            cur, dirty, nbytes, dstages, c0, c1, c2, c3 = state
            pushed_ar.clear_dirty_stages(
                add_addr,
                dirty,
                nbytes,
                dstages,
                c0,
                c1,
                c2,
                c3,
                bidx * self.threads + tidx,
                m * (c * self.threads),
            )
            off = _selp_b32(valid, Int32(rank * self.vecs + tidx) * 16, Int32(0))
            if cutlass.const_expr(self.finish_two_shot):
                finished = pushed_ar.finish_two_shot_v4(
                    add_addr, mc_addr, cur, nbytes, token, off, ar_rank, valid, world, hidden
                )
            else:
                acc = pushed_ar.reduce_slots_v4(
                    pushed_ar.one_shot_slots(add_addr, cur, nbytes, token, world, hidden)
                    + Int64(off),
                    hidden * 2,
                    world,
                    self.finish_chunk,
                )
                finished = pushed_ar.round_sum_v4(acc, self.finish_sanitize)
            finished = [_selp_b32(valid, w, Int32(0)) for w in finished]
            # Every thread of this CTA has read the state (its polls used it): count the CTA
            # now with a relaxed atomic whose result thread 0 reads only after its stores, so the
            # round trip overlaps the mix; the last CTA advances the state after the grid wait.
            cute.arch.barrier()
            if tidx == 0:
                arrivals = pushed_ar.arrive_count(add2_addr)
        if cutlass.const_expr(self.add_mode == ADD_PREFIX):
            pre = unpack(prefix)
            if cutlass.const_expr(self.finish_world > 0):
                addend = unpack(finished)
            elif cutlass.const_expr(self.in_place_addend):
                # Threads past the row poll the row's first vector and use zeros.
                off = _selp_b32(valid, Int32(rank * self.vecs + tidx) * 16, Int32(0))
                polled = pushed_ar.poll_result_v4(add_addr, add2_addr, Int32(token), off, hidden)
                addend = unpack([_selp_b32(valid, w, Int32(0)) for w in polled])
            else:
                addend = unpack(load(add_addr + row + col, False))
            x = [_round_bf16(_add_ftz(pre[e], addend[e])) for e in range(8)]
        elif cutlass.const_expr(self.add_mode == ADD_MOE):
            pre = unpack(prefix)
            if cutlass.const_expr(self.finish_all_gather):
                # routed: this token's all-gather row; shared: the finished slot sum.
                routed_words = pushed_ar.poll_v4(
                    pushed_ar.all_gather_row(add_addr, cur, nbytes, m, token, world, hidden)
                    + Int64(off)
                )
                routed = unpack([_selp_b32(valid, w, Int32(0)) for w in routed_words])
                shared = unpack(finished)
            else:
                routed = unpack(load(add_addr + row + col, False))
                shared = unpack(load(add2_addr + row + col, False))
            x = [
                _round_bf16(_add(pre[e], _round_bf16(_add(routed[e], shared[e])))) for e in range(8)
            ]
        else:
            x = unpack(load(layer_addr + row + col, False))
        upd_words = []
        if cutlass.const_expr(self.add_mode != ADD_NONE):
            upd_words = [_pack_bf16x2(x[2 * i], x[2 * i + 1]) for i in range(4)]
            if cutlass.const_expr(not self.early_poll):
                _stg_v4_if(valid, upd_addr + row + col, upd_words)

        sq = _zero_f32()
        dot = _zero_f32()
        for e in cutlass.range_constexpr(8):
            sq = _fma(x[e], x[e], sq)
            dot = _fma(x[e], q[e], dot)
        # Lanes 0-15 hold the warp's sum of squares, lanes 16-31 its dot product.
        pair = _warp_transpose_sum([sq, dot], lane)
        half = lane & 15
        if half < c:
            _st_async_b32(
                last_smem + ((rank * warps + warp) * 2 + (lane >> 4)) * 4,
                half,
                mbar_smem + 8,
                pair,
            )
        _mbar_wait_cluster(mbar_smem + 8)
        part_sq = _zero_f32()
        part_dot = _zero_f32()
        if lane < parts:
            part_sq = _lds_f32(last_smem + lane * 8)
            part_dot = _lds_f32(last_smem + lane * 8 + 4)
        tot = _warp_transpose_sum([part_sq, part_dot], lane)
        tot_sq = _shfl_idx(tot, 0)
        tot_dot = _shfl_idx(tot, 16)
        last_logit = _mul_ftz(tot_dot, _rsigma(tot_sq, eps, hidden))

        # Softmax over all candidates: p_n = exp(logit_n - m_s) * a / den, p_last = b / den.
        new_max = _max_ftz(snap_max, last_logit)
        a = _ex2_scaled(_add(snap_max, -new_max))
        b = _ex2_scaled(_add(last_logit, -new_max))
        inv = _rcp_ftz(_fma(snap_sum, a, b))
        scale = _mul_ftz(a, inv)
        p_last = _mul_ftz(b, inv)
        mixed = [_round_bf16(_fma(p_last, x[e], _mul_ftz(scale, premix[e]))) for e in range(8)]
        out_sq = _zero_f32()
        for e in cutlass.range_constexpr(8):
            out_sq = _fma(mixed[e], mixed[e], out_sq)
        out_sq = _warp_sum(out_sq)
        if lane < c:
            _st_async_b32(outsq_smem + (rank * warps + warp) * 4, lane, mbar_smem + 16, out_sq)
        _mbar_wait_cluster(mbar_smem + 16)
        part = _zero_f32()
        if lane < parts:
            part = _lds_f32(outsq_smem + lane * 4)
        out_rs = _rsigma(_warp_sum(part), out_eps, hidden)

        w_out = unpack(out_w)
        y = [_mul_ftz(_round_bf16(_mul_ftz(mixed[e], out_rs)), w_out[e]) for e in range(8)]
        words = [_pack_bf16x2(y[2 * i], y[2 * i + 1]) for i in range(4)]
        if cutlass.const_expr(self.early_poll):
            # The output buffers may reuse memory the producer, or a grid before it, still reads.
            if cutlass.const_expr(not self.snapshots_after_wait):
                wait_and_release()
            _stg_v4_if(valid, upd_addr + row + col, upd_words)
        _stg_v4_if(valid, out_addr + row + col, words)
        if cutlass.const_expr(self.mxfp8):
            # MXFP8 form of the stored BF16 row: four consecutive lanes hold one 1x32 group.
            amax = low_m_gemm.amax_bf16x8(*words)
            amax = low_m_gemm.amax_pair(amax, _shfl_xor_b32(amax, 1))
            amax = low_m_gemm.amax_pair(amax, _shfl_xor_b32(amax, 2))
            e_a = low_m_gemm.ue8m0_of_amax(amax)
            q0, q1 = low_m_gemm.quant_bf16x8_to_e4m3(*words, low_m_gemm.bf16x2_inv_scale(e_a))
            chan = Int64(rank * self.vecs + tidx) * 8
            _stg_v2_if(valid, xq_addr + Int64(token) * hidden + chan, q0, q1)
            # R128c4: atom (token // 128, group // 4) of 512 bytes, byte
            # (token % 32) * 16 + (token % 128 // 32) * 4 + group % 4.
            group = chan >> 5
            sf_off = (Int64(token >> 7) * (hidden // 128) + (group >> 2)) * 512
            sf_off = sf_off + Int64((token & 31) * 16 + ((token & 127) >> 5) * 4) + (group & 3)
            _stg_u8_if(valid & Int32((tidx & 3) == 0), xs_addr + sf_off, e_a)
        if cutlass.const_expr(self.finish_world > 0):
            # The producer has completed (the wait above): the last CTA to count advances the
            # workspace.
            if tidx == 0:
                pushed_ar.arrive_rotate(
                    add2_addr,
                    arrivals,
                    m * c,
                    state[0],
                    state[2],
                    m,
                    self.finish_world,
                    hidden,
                    not self.finish_two_shot,
                    self.finish_all_gather,
                )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


class AllReduceFinish(NamedTuple):
    """A pushed MNNVL all-reduce the kernel finishes (``finish`` of :func:`run_attn_res`):
    ``world`` ranks, this ``rank``, the workspace's multicast base ``mc_base``, the transport
    (``two_shot``) and whether the kernel finishes it before its grid wait (``early_poll``).
    One-shot calls may sum ``chunk`` ranks at a time and ``sanitize`` the sum (the order of
    another C++ transport, 0 / False: their own) and be ``all_gather`` calls (``(routed,
    shared)`` addends: the all-gather rows and the slot sum)."""

    world: int
    rank: int
    mc_base: int
    two_shot: bool
    early_poll: bool
    chunk: int = 0
    sanitize: bool = False
    all_gather: bool = False


class AttnResVariant(NamedTuple):
    """The compiled variant of :class:`KimiK3AttnResKernel` (also the key of its compile cache).

    ``num_candidates``: ``N``, the snapshot rows plus the layer row, in ``[MIN_CANDIDATES,
    MAX_CANDIDATES]``; ``add_mode``: inputs of the layer row (``ADD_NONE``, ``ADD_PREFIX``,
    ``ADD_MOE``); ``cluster``: CTAs per token (:func:`cluster_size`); ``hidden``: ``H``
    (:func:`supports_hidden_size`); ``mxfp8``: also store the MXFP8 form of the output;
    ``snapshots_after_wait``: load the snapshot rows after ``griddepcontrol.wait`` (the module's
    PDL contract); ``in_place_addend``: the addend is the in-place result of a pushed MNNVL
    all-reduce (``ADD_PREFIX``, or ``ADD_MOE`` for an all-gather call); ``finish``: the pushed
    call the kernel finishes (with ``in_place_addend``; ``rank`` and ``mc_base`` are launch
    arguments and 0 here); ``entry_trigger``: release the dependent grid at kernel entry (module
    docstring).
    """

    num_candidates: int
    add_mode: int
    cluster: int
    hidden: int
    mxfp8: bool = False
    snapshots_after_wait: bool = False
    in_place_addend: bool = False
    finish: AllReduceFinish | None = None
    entry_trigger: bool = False

    def validate(self) -> None:
        if not MIN_CANDIDATES <= self.num_candidates <= MAX_CANDIDATES:
            raise ValueError(f"num_candidates={self.num_candidates} not in [2, 12]")
        if (
            self.add_mode not in (ADD_NONE, ADD_PREFIX, ADD_MOE)
            or self.cluster not in CLUSTER_SIZES
        ):
            raise ValueError(f"add_mode={self.add_mode} cluster={self.cluster}")
        if not supports_hidden_size(self.hidden):
            raise ValueError(f"hidden={self.hidden} is not supported")
        finish = self.finish
        all_gather = finish is not None and finish.all_gather
        if self.in_place_addend and self.add_mode != (ADD_MOE if all_gather else ADD_PREFIX):
            raise ValueError("an in-place addend is ADD_PREFIX, or ADD_MOE for an all-gather call")
        if finish is None:
            return
        if not self.in_place_addend or finish.world not in pushed_ar.WORLD_SIZES:
            raise ValueError(f"cannot finish an in-place addend over {finish.world} ranks")
        if finish.two_shot and (finish.early_poll or finish.chunk or finish.sanitize or all_gather):
            raise ValueError("early_poll, sum orders and all-gather calls are one-shot only")


def cluster_size(num_tokens: int, num_candidates: int) -> int:
    """CTAs per token: eight for up to 16 tokens with at least four candidates (a smaller
    per-CTA prologue), else four (eight-CTA clusters would oversubscribe the GPU)."""
    return 8 if num_tokens <= 16 and num_candidates >= 4 else 4


@functools.cache
def _compiled(variant: AttnResVariant):
    kernel = KimiK3AttnResKernel(variant)
    addr = Int64(1 << 20)
    return cute.compile(
        kernel,
        *([addr] * 12),
        Int32(0),
        Int32(1),
        Float32(1e-6),
        Float32(1e-6),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def is_supported_device(device_index: int) -> bool:
    """Whether the kernel runs on this GPU (SM100, SM103; SM107 with a Rubin-capable CuTe DSL)."""
    from ...cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    capability = torch.cuda.get_device_capability(device_index)
    return capability in ((10, 0), (10, 3)) or (
        capability == (10, 7) and IS_CUTLASS_DSL_RUBIN_AVAILABLE
    )


def supports(num_tokens: int, hidden: int, num_candidates: int) -> bool:
    """Whether the kernel serves ``num_tokens`` rows of ``hidden`` with ``num_candidates``
    candidates (snapshots + the layer row) on a supported device."""
    return (
        1 <= num_tokens <= MAX_TOKENS
        and MIN_CANDIDATES <= num_candidates <= MAX_CANDIDATES
        and supports_hidden_size(hidden)
    )


def can_run(
    layer: torch.Tensor,
    addends: tuple[torch.Tensor, ...],
    block: torch.Tensor,
    *weights: torch.Tensor,
) -> bool:
    """Whether :func:`run_attn_res` serves ``layer [M, H]``, 0-2 addends ``[M, H]``,
    ``block [N - 1, M, H]`` and the ``[H]`` weights: BF16, contiguous, 16-byte aligned,
    :func:`supports` ``(M, H, N)``, on SM100 / SM103 / SM107."""
    if not layer.is_cuda or not is_supported_device(layer.device.index or 0):
        return False
    if layer.dim() != 2 or block.dim() != 3 or len(addends) > 2:
        return False
    m, hidden = layer.shape
    if not supports(m, hidden, block.shape[0] + 1) or tuple(block.shape[1:]) != (m, hidden):
        return False
    for t in (layer, block, *weights, *addends):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.device != layer.device:
            return False
        if t.data_ptr() % 16:
            return False
    if any(a.shape != layer.shape for a in addends):
        return False
    return all(w.numel() == hidden for w in weights)


def precompile(
    num_candidates: int,
    add_mode: int,
    cluster: int,
    hidden: int,
    mxfp8: bool = False,
    snapshots_after_wait: bool = False,
) -> None:
    """JIT-compile the variant :func:`run_attn_res` launches for these parameters (``cluster``
    as :func:`cluster_size` picks it), so its first launch does not compile."""
    _compiled(
        AttnResVariant(num_candidates, add_mode, cluster, hidden, mxfp8, snapshots_after_wait)
    )


def mxfp8_scale_numel(num_tokens: int, hidden: int) -> int:
    """Bytes of the R128c4 UE8M0 scales of ``num_tokens`` MXFP8 rows of ``hidden`` (tokens padded
    to 128)."""
    return (num_tokens + 127) // 128 * 128 * (hidden // 32)


def run_attn_res(
    layer: torch.Tensor,
    addends: tuple[torch.Tensor, ...],
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
    output: torch.Tensor,
    updated: torch.Tensor | None,
    *,
    mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    in_place_addend: tuple[int, int] | None = None,
    finish: AllReduceFinish | None = None,
    entry_trigger: bool = False,
    snapshots_after_wait: bool = False,
) -> None:
    """Launch on the current stream, with PDL unless ``TRTLLM_ENABLE_PDL=0`` (inputs are not
    re-validated; see :func:`can_run`).

    Args:
        layer: ``bf16 [M, H]``: the layer row without addends, else the prefix sum.
        addends: ``()``; ``(addend,)`` for ``bf16(layer + addend)``; ``(routed, shared)`` for
            ``bf16(layer + bf16(routed + shared))`` (all ``bf16 [M, H]``).
        block: ``bf16 [N - 1, M, H]`` snapshot rows, ``2 <= N <= 12``.
        res_weight, rms_weight: Score projection and its RMSNorm weight, ``bf16 [H]``.
        out_weight: Weight of the trailing RMSNorm, ``bf16 [H]``.
        rms_eps, out_eps: Epsilons of the two norms.
        output: ``bf16 [M, H]`` normalized mixture.
        updated: ``bf16 [M, H]`` updated prefix sum; required with addends.
        mxfp8_out: ``(xq, xs)``: also store the MXFP8 form of ``output``, E4M3 ``xq [M, H]``
            and UE8M0 ``xs`` (uint8, :func:`mxfp8_scale_numel` bytes) in the R128c4 layout.
        in_place_addend: ``(uc_local, flags)`` device addresses of an MNNVL all-reduce workspace
            (``distributed.ops.LamportWorkspaceView``) instead of ``addends`` (then
            ``()``): ``bf16(layer + addend)`` with the ``[M, H]`` result of the pushed two-shot
            all-reduce finished in place last on the workspace. The next all-reduce on the
            workspace re-arms that buffer, so it must not start before this kernel completed.
        finish: with ``in_place_addend``, the addend is instead the pushed call whose finish
            was left to this kernel (``distributed.ops.MnnvlPendingResult``): the kernel
            finishes it (see :class:`AllReduceFinish`). It must be the call's only finish, run
            on every rank before the next all-reduce on the workspace. For an ``all_gather``
            call the mix is the MoE residual update ``bf16(layer + bf16(routed + shared))``
            with the call's all-gather rows as ``routed`` and its slot sum as ``shared``.
        entry_trigger: release the dependent at kernel entry (see the module docstring).
        snapshots_after_wait: Load ``block`` after the PDL wait: required where the kernel
            launched immediately before this one writes ``block`` (module docstring).
    """
    if in_place_addend is not None and addends:
        raise ValueError("pass either addends or in_place_addend")
    if finish is not None and in_place_addend is None:
        raise ValueError("finish needs in_place_addend")
    if in_place_addend is not None:
        add_mode = ADD_MOE if finish is not None and finish.all_gather else ADD_PREFIX
    else:
        add_mode = len(addends)
    num_tokens, hidden = layer.shape
    num_candidates = block.shape[0] + 1
    compiled = _compiled(
        AttnResVariant(
            num_candidates,
            add_mode,
            cluster_size(num_tokens, num_candidates),
            hidden,
            mxfp8_out is not None,
            snapshots_after_wait,
            in_place_addend is not None,
            finish._replace(rank=0, mc_base=0) if finish is not None else None,
            entry_trigger,
        )
    )
    if in_place_addend is not None:
        add_addr, add2_addr = in_place_addend
    else:
        add_addr = addends[0].data_ptr() if add_mode >= 1 else 0
        add2_addr = addends[1].data_ptr() if add_mode == 2 else 0
    compiled(
        Int64(layer.data_ptr()),
        Int64(add_addr),
        Int64(add2_addr),
        Int64(block.data_ptr()),
        Int64(res_weight.data_ptr()),
        Int64(rms_weight.data_ptr()),
        Int64(out_weight.data_ptr()),
        Int64(output.data_ptr()),
        Int64(updated.data_ptr() if add_mode else 0),
        Int64(mxfp8_out[0].data_ptr() if mxfp8_out is not None else 0),
        Int64(mxfp8_out[1].data_ptr() if mxfp8_out is not None else 0),
        Int64(finish.mc_base if finish is not None else 0),
        Int32(finish.rank if finish is not None else 0),
        Int32(layer.shape[0]),
        Float32(rms_eps),
        Float32(out_eps),
        cuda.CUstream(torch.cuda.current_stream(layer.device).cuda_stream),
    )


__all__ = [
    "ADD_MOE",
    "AllReduceFinish",
    "AttnResVariant",
    "ADD_NONE",
    "ADD_PREFIX",
    "CLUSTER_SIZES",
    "MAX_CANDIDATES",
    "MAX_TOKENS",
    "MIN_CANDIDATES",
    "KimiK3AttnResKernel",
    "can_run",
    "cluster_size",
    "is_supported_device",
    "mxfp8_scale_numel",
    "precompile",
    "run_attn_res",
    "supports",
    "supports_hidden_size",
]
