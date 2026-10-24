# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decode attention-residual mix + RMSNorm (+ residual add) with a PDL prologue
(SM100/SM103/SM107).

For ``M`` tokens of hidden size 7168, ``N - 1`` snapshot rows ``block[n]`` and the layer row
``x``, one launch computes, per token,

* ``rsigma_n = rsqrt(mean(v_n^2) + eps)``, ``logit_n = rsigma_n * <v_n, res_w * rms_w>``,
* ``p = softmax(logit)``, ``mixed = bf16(sum_n p_n v_n)``,
* ``out = bf16(bf16(mixed * rsqrt(mean(mixed^2) + out_eps)) * out_w)``,

where ``x`` is the given row, ``bf16(prefix + addend)`` (``trtllm::attn_res_add_rmsnorm_fwd``'s
fused add), or ``bf16(residual + bf16(routed + shared))`` (the MoE residual update of the
previous layer, ``kimi_k3_moe_residual._fused_moe_residual_add``); the updated prefix ``x`` is
stored as well in the last two cases, bit-identical to those adds.

The kernels are launched with programmatic dependent launch and do everything that does not
depend on the previous kernel before ``griddepcontrol.wait``: they load the weights and the
snapshot rows (and the prefix sum, which is older than the previous kernel) and reduce the
snapshots' statistics over the CTA or the cluster. After the wait they only load the dependent
row, reduce its two statistics, finish the softmax and the mix, reduce the output sum of squares
and store. Cluster partials travel by ``st.async`` into every peer's shared memory and are
counted by a per-round mbarrier (one one-way message per round).

Two kernels share that schedule:

* :class:`K3AttnResFastKernel` (default): a cluster of ``c`` CTAs per token (see
  :func:`fast_cluster_size`), one 16-byte vector per thread. It sums the statistics in another
  order than the C++ kernels, so the normalized output can differ from theirs by one BF16
  rounding step in a small fraction of the elements; the snapshots' share of the softmax and of
  the mixture is also folded before the wait.
* :class:`K3AttnResKernel` (``TRTLLM_K3_ATTN_RES_PDL_EXACT=1``): the per-element layout and the
  arithmetic of ``trtllm::attn_res_rmsnorm_fwd`` / ``trtllm::attn_res_add_rmsnorm_fwd``
  instruction for instruction (one 256-thread CTA per token for ``N <= 4``, an eight-CTA cluster
  for ``N >= 5``), so its results are bit-identical to theirs.

Snapshot rows and the prefix sum may be read before the wait because they are older than the
previous kernel: the bank rows are written by plain copy kernels (no early trigger) at least one
layer earlier, and the prefix sum before the previous kernel's own predecessor, which waits for
its predecessor before it triggers its dependents.

``TRTLLM_K3_ATTN_RES_EARLY_TRIGGER`` (MK1, SM107 only; see ``kimi_k3_mk1_gate``) moves this
kernel's ``griddepcontrol.launch_dependents``: 0 (default) after the last store, 1 right after
its own ``griddepcontrol.wait``, 2 at entry. The dependent's own wait still blocks until this
grid has completed, so every mode computes the same results; an early trigger only lets the
dependent launch and run its pre-wait part (e.g. stage its weights) while this kernel runs. Mode 2
lets the dependent launch before this kernel has waited for its own predecessor: use it only when
every kernel that can then start reads nothing but constants (weights) before its own wait.

**Lamport slots** (``slots = W``): instead of reading the addend (``ADD_PREFIX``) or the
layer row (``ADD_NONE``) from memory, the kernel takes ``bf16(sum_r slot[r])`` from a one-shot
Lamport mailbox region ``[M][W][7168]`` (``k3_lamport_mailbox``) into which each of the ``W``
ranks of the TP group published its BF16 partial (the o_proj GEMM's Lamport epilogue). The sum
is the MNNVL one-shot order (FP32 from ``+0``, one BF16 rounding), so the result is bit-identical
to ``attn_res(prefix + MNNVLAllReduce(partials))``; with ``ADD_NONE`` the summed row is stored as
the updated prefix (it is the all-reduce output the caller would have kept). Every polled fragment
is re-armed by the thread that polled it. The exact kernel polls its slice as contiguous 16-byte
fragments, thread ``t`` fragment ``t`` (a warp reads 512 contiguous bytes of each slot), and the
sums meet in shared memory (``c = 8``); or (``c = 1``) it runs the per-token CTA on ``poll_ctas``
CTAs of a cluster that each poll a share of the row that way, broadcast the sums into every CTA's
shared memory and store a share of the outputs (the arithmetic of the single CTA is unchanged);
the fast kernel polls its own vector. ``poll_early`` polls before ``griddepcontrol.wait``
(outputs are still stored after it).
"""

from __future__ import annotations

import functools
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..rubin.k3_lamport import poll_sum_slots_v4

#: Hidden size the kernel is specialized for.
HIDDEN_SIZE = 7168
#: Candidate counts (snapshots + the layer row) the kernel is instantiated for.
MIN_CANDIDATES, MAX_CANDIDATES = 2, 12
#: Largest token count of the decode path (one CTA or cluster per token).
MAX_TOKENS = 1024
#: Inputs of the layer row: the row itself; ``bf16(prefix + addend)`` (the C++ fused add);
#: ``bf16(residual + bf16(routed + shared))`` (the MoE residual update).
ADD_NONE, ADD_PREFIX, ADD_MOE = 0, 1, 2

#: Programmatic dependent launch unless TRTLLM_ENABLE_PDL=0 (the stack-wide switch).
_USE_PDL = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
#: Where the kernels trigger their dependents (see the module docstring), read at use time.
EARLY_TRIGGER_ENV = "TRTLLM_K3_ATTN_RES_EARLY_TRIGGER"
TRIGGER_AT_END, TRIGGER_AFTER_WAIT, TRIGGER_AT_ENTRY = 0, 1, 2
#: Largest TP group of the Lamport slots delta source (one-shot regions of ``k3_lamport_mailbox``).
SLOTS_MAX_WORLD = 16
#: Default CTAs per token of the exact ``c = 1`` kernel with slots (one 16-byte fragment per thread).
SLOTS_POLL_CTAS = 4

_THREADS = 256
_WARPS = _THREADS // 32
_LOG2E = "0f3FB8AA3B"
_NEG_FLT_MAX = "0fFF7FFFFF"
_HIDDEN_F32 = "0f45E00000"  # 7168.0

# ---------------------------------------------------------------------------
# Inline PTX helpers (fixed instructions: the C++ kernels are built with --use_fast_math)
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _asm_se(res, operands, text, constraints, *, loc=None, ip=None):
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


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _f32_op(text, *args, loc=None, ip=None):
    constraints = "=f," + ",".join("f" for _ in args)
    return Float32(
        _asm(T.f32(), [_fv(a, loc=loc, ip=ip) for a in args], text, constraints, loc=loc, ip=ip)
    )


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return _f32_op("fma.rn.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def _fma_ftz(a, b, c, *, loc=None, ip=None):
    return _f32_op("fma.rn.ftz.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


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
def _rsigma(sum_sq, eps, *, loc=None, ip=None):
    """``rsqrtf(sum_sq / 7168 + eps)`` as nvcc --use_fast_math emits it."""
    return _f32_op(
        "{\n.reg .f32 t;\n"
        f"div.approx.ftz.f32 t, $1, {_HIDDEN_F32};\n"
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
def _softmax_term(logit, max_logit, *, loc=None, ip=None):
    """``exp2f((logit - max_logit) * LOG2_E)`` (fast math)."""
    return _f32_op(
        "{\n.reg .f32 t;\nsub.ftz.f32 t, $1, $2;\n"
        f"mul.ftz.f32 t, t, {_LOG2E};\nex2.approx.ftz.f32 $0, t;\n}}",
        logit,
        max_logit,
        loc=loc,
        ip=ip,
    )


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
def _ld_bf16(addr, *, loc=None, ip=None):
    """bf16 global load (coherent: the data may come from the previous grid)."""
    return Float32(
        _asm_se(
            T.f32(),
            [_lv(addr, loc=loc, ip=ip)],
            "{\n.reg .b16 t;\nld.global.b16 t, [$1];\ncvt.f32.bf16 $0, t;\n}",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_bf16_nc(addr, *, loc=None, ip=None):
    """bf16 load through the read-only path (data older than this grid)."""
    return Float32(
        _asm_se(
            T.f32(),
            [_lv(addr, loc=loc, ip=ip)],
            "{\n.reg .b16 t;\nld.global.nc.b16 t, [$1];\ncvt.f32.bf16 $0, t;\n}",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_bf16_if(valid, addr, nc: bool, *, loc=None, ip=None):
    """bf16 global load predicated on ``valid != 0``; 0.0 otherwise."""
    space = "global.nc" if nc else "global"
    return Float32(
        _asm_se(
            T.f32(),
            [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)],
            "{\n.reg .b16 t;\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b16 t, 0;\n"
            f"@p ld.{space}.b16 t, [$2];\ncvt.f32.bf16 $0, t;\n}}",
            "=f,r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_bf16(addr, value, *, loc=None, ip=None):
    """Store ``bf16_rn(value)``."""
    _asm_se(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\nst.global.b16 [$0], t;\n}",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_bf16_if(valid, addr, value, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 t;\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\ncvt.rn.bf16.f32 t, $2;\n"
        "@p st.global.b16 [$1], t;\n}",
        "r,l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _shfl_down(value, offset: int, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.down.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
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


@dsl_user_op
def _warp_sum_bfly_ftz(value, *, loc=None, ip=None):
    """``warp_reduce_sum`` of the C++ kernels: xor butterfly 16..1, ``add.ftz``."""
    body = "".join(
        f"shfl.sync.bfly.b32 t, x, {off}, 0x1f, 0xffffffff;\nadd.ftz.f32 x, x, t;\n"
        for off in (16, 8, 4, 2, 1)
    )
    return Float32(
        _asm_se(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            "{\n.reg .f32 x, t;\nmov.b32 x, $1;\n" + body + "mov.b32 $0, x;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_v2(addr, a, b, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v2(addr, *, loc=None, ip=None):
    res = _asm_se(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _sts_f32(addr, a, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(a, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _bar_sync(barrier_id: int, num_threads: int = _THREADS, *, loc=None, ip=None):
    _asm_se(None, [], f"bar.sync {barrier_id}, {num_threads};", "", loc=loc, ip=ip)


@dsl_user_op
def _mbar_init(addr, count: int, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.init.shared::cta.b64 [$0], {count};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_mbar_init(*, loc=None, ip=None):
    _asm_se(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def _mbar_arrive_expect_tx(addr, nbytes: int, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {nbytes};",
        "r",
        loc=loc,
        ip=ip,
    )


_WAIT_LABEL = [0]


@dsl_user_op
def _mbar_wait_cluster(addr, *, loc=None, ip=None):
    """Wait for phase 0 of a local mbarrier completed by peers' ``st.async`` (cluster acquire)."""
    _WAIT_LABEL[0] += 1
    lab = f"K3AR_WAIT{_WAIT_LABEL[0]}"
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
def _st_async_v2(local_addr, rank, local_mbar, a, b, *, loc=None, ip=None):
    """8-byte ``st.async`` into CTA ``rank``'s copy of ``local_addr``, counted by its mbarrier."""
    _asm_se(
        None,
        [
            _iv(local_addr, loc=loc, ip=ip),
            _iv(rank, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
            _fv(a, loc=loc, ip=ip),
            _fv(b, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra, rm;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "mapa.shared::cluster.u32 rm, $2, $1;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [ra], {$3, $4}, [rm];\n}",
        "r,r,r,f,f",
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
def _cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm_se(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait(*, loc=None, ip=None):
    _asm_se(None, [], "barrier.cluster.wait.acquire.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm_se(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm_se(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


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
def _sts_v4_if(valid, addr, words, *, loc=None, ip=None):
    """16-byte shared store of four words where ``valid != 0``."""
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)]
        + [_iv(w, loc=loc, ip=ip) for w in words],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.shared.v4.b32 [$1], {$2, $3, $4, $5};\n}",
        "r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_v4_if(valid, local_addr, rank, local_mbar, words, *, loc=None, ip=None):
    """16-byte ``st.async`` into CTA ``rank`` of the cluster (same shared-memory offsets),
    completing 16 bytes of its mbarrier at ``local_mbar``; nothing where ``valid == 0``."""
    _asm_se(
        None,
        [
            _iv(valid, loc=loc, ip=ip),
            _iv(local_addr, loc=loc, ip=ip),
            _iv(rank, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
        ]
        + [_iv(w, loc=loc, ip=ip) for w in words],
        "{\n.reg .b32 ra, rm;\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n"
        "mapa.shared::cluster.u32 ra, $1, $2;\nmapa.shared::cluster.u32 rm, $3, $2;\n"
        "@p st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [ra], {$4, $5, $6, $7}, [rm];\n}",
        "r,r,r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_arrive_expect_tx_if(valid, addr, nbytes: int, *, loc=None, ip=None):
    """``mbarrier.arrive.expect_tx`` of ``nbytes`` on a local mbarrier where ``valid != 0``."""
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n"
        f"@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], {int(nbytes)};\n}}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_bf16(addr, *, loc=None, ip=None):
    """``float`` of the BF16 at shared address ``addr``."""
    return Float32(
        _asm_se(
            T.f32(),
            [_iv(addr, loc=loc, ip=ip)],
            "{\n.reg .b16 t;\nld.shared.b16 t, [$1];\ncvt.f32.bf16 $0, t;\n}",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_bf16_if(valid, addr, *, loc=None, ip=None):
    """``float`` of the BF16 at shared address ``addr`` where ``valid != 0``, else 0."""
    return Float32(
        _asm_se(
            T.f32(),
            [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
            "{\n.reg .b16 t;\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b16 t, 0;\n"
            "@p ld.shared.b16 t, [$2];\ncvt.f32.bf16 $0, t;\n}",
            "=f,r,r",
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


# ---------------------------------------------------------------------------
# FP8 1x128 output (the next GEMM's activation, R128c4 UE8M0 scales)
#
# The arithmetic is that of fp8_quantize_1x128_cutedsl_kernel_impl (fp8_blockscale_quant_packed.cu)
# and of the MXFP8 quantizer with 128-element scale vectors (quantization.cuh), instruction for
# instruction, applied to the BF16 values this kernel stores: amax over the 128-element block
# (floored at 1e-10), dequant scale amax * rcp.approx.ftz(448) rounded up to a power of two
# (UE8M0, satfinite), quant scale 2^(127 - e), codes cvt.rn.satfinite.e4m3 of value * scale. Each
# scale byte is replicated into the four K32 slots of CuTe's 128x4 swizzled layout, and the scale
# rows of the padding up to the next multiple of 128 tokens are zero, as those quantizers write them.
# ---------------------------------------------------------------------------

#: Bytes of one 128-row x 128-K tile of the R128c4 scale layout (32 x 4 x 4 slots).
_SF_TILE_BYTES = 512
#: 128-element scale blocks per row of 7168.
_SF_BLOCKS = HIDDEN_SIZE // 128


def fp8_scale_numel(num_tokens: int) -> int:
    """Bytes of the R128c4 UE8M0 scale buffer of ``[num_tokens, 7168]`` (rows padded to 128)."""
    return (num_tokens + 127) // 128 * 128 * (HIDDEN_SIZE // 32)


@dsl_user_op
def _abs_max(a, b, *, loc=None, ip=None):
    """``max(a, |b|)`` (``a`` is already non-negative)."""
    return _f32_op("{\n.reg .f32 t;\nabs.f32 t, $2;\nmax.f32 $0, $1, t;\n}", a, b, loc=loc, ip=ip)


@dsl_user_op
def _max_f32(a, b, *, loc=None, ip=None):
    return _f32_op("max.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _ue8m0_scales(amax, *, loc=None, ip=None):
    """``(e, q)``: UE8M0 byte of ``max(amax, 1e-10) / 448`` rounded up, and ``q = 2^(127 - e)``.

    ``e = exponent + (mantissa != 0)`` is round-toward-+inf to a power of two for the positive
    normal values reachable here (1e-10 / 448 <= x <= bf16 max / 448), clamped to 254
    (satfinite); ``q`` is built from its exponent bits (1.0 for ``e == 0``), the exact value of the
    C++ quantizers' ``exp2f(127 - e)`` / ``rcp(2^(e - 127))``.
    """
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [_fv(amax, loc=loc, ip=ip)],
        "{\n.reg .f32 a, d;\n.reg .b32 bits, ex, mt, e8, qb;\n.reg .pred pm, pz;\n"
        "max.f32 a, $2, 0f2EDBE6FF;\n"  # 1e-10f
        "rcp.approx.ftz.f32 d, 0f43E00000;\n"  # 448.0f
        "mul.f32 d, a, d;\n"
        "mov.b32 bits, d;\n"
        "shr.u32 ex, bits, 23;\nand.b32 ex, ex, 255;\n"
        "and.b32 mt, bits, 8388607;\nsetp.ne.u32 pm, mt, 0;\n"
        "selp.u32 e8, 1, 0, pm;\nadd.u32 e8, ex, e8;\nmin.u32 e8, e8, 254;\n"
        "sub.u32 qb, 254, e8;\nshl.b32 qb, qb, 23;\n"
        "setp.eq.u32 pz, e8, 0;\nselp.b32 qb, 1065353216, qb, pz;\n"
        "mov.b32 $0, e8;\nmov.b32 $1, qb;\n}",
        "=r,=f,f",
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _e4m3x8(vals, q, *, loc=None, ip=None):
    """Two ``b32`` words holding ``e4m3(vals[i] * q)``, ``vals[0]`` in the lowest byte."""
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [_fv(v, loc=loc, ip=ip) for v in vals] + [_fv(q, loc=loc, ip=ip)],
        "{\n.reg .f32 t0, t1, t2, t3, t4, t5, t6, t7;\n.reg .b16 h0, h1, h2, h3;\n"
        + "".join(f"mul.f32 t{i}, ${i + 2}, $10;\n" for i in range(8))
        + "cvt.rn.satfinite.e4m3x2.f32 h0, t1, t0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, t3, t2;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h2, t5, t4;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h3, t7, t6;\n"
        "mov.b32 $0, {h0, h1};\nmov.b32 $1, {h2, h3};\n}",
        "=r,=r," + ",".join("f" for _ in range(9)),
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _e4m3_byte(val, q, *, loc=None, ip=None):
    """``e4m3(val * q)`` in the low byte of a ``b32``."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(val, loc=loc, ip=ip), _fv(q, loc=loc, ip=ip)],
            "{\n.reg .f32 t, z;\n.reg .b16 h;\nmul.f32 t, $1, $2;\nmov.b32 z, 0f00000000;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h, z, t;\ncvt.u32.u16 $0, h;\n}",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _stg_v2_if(valid, addr, w0, w1, *, loc=None, ip=None):
    _asm_se(
        None,
        [
            _iv(valid, loc=loc, ip=ip),
            _lv(addr, loc=loc, ip=ip),
            _iv(w0, loc=loc, ip=ip),
            _iv(w1, loc=loc, ip=ip),
        ],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.v2.b32 [$1], {$2, $3};\n}",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_b32_if(valid, addr, word, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip), _iv(word, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.b32 [$1], $2;\n}",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_b8_if(valid, addr, byte, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip), _iv(byte, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.b8 [$1], $2;\n}",
        "r,l,r",
        loc=loc,
        ip=ip,
    )


def _sf_offset(token, block):
    """Byte offset of (token, 128-element block) in the R128c4 scale layout of a 7168-wide row."""
    return (
        Int64(token >> 7) * (_SF_BLOCKS * _SF_TILE_BYTES)
        + Int64(block) * _SF_TILE_BYTES
        + Int64((token & 31) * 16 + ((token & 127) >> 5) * 4)
    )


def _pad_scale_rows(m, token):
    """Padding scale rows ``m + token + j * m < roundup(m, 128)`` that token ``token`` clears.

    Every token's CTAs cover every 128-element block of the row once, so token ``t`` zeroes the
    padding rows congruent to ``t`` modulo ``m``; together they zero all of them, as the C++
    quantizers do. Returns the trip count (``<= 0``: none); the loop itself stays in the kernel
    body, where the DSL turns it into a device loop.
    """
    return ((m + 127) // 128 * 128 - m - token + m - 1) // m


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


def _warp_transpose_max(values, lane):
    """:func:`_warp_transpose_sum` with ``max`` (``len(values)`` a power of two <= 32)."""
    offset = 16
    while len(values) > 1:
        half = len(values) // 2
        upper = Int32((lane & offset) != 0)
        reduced = []
        for i in range(half):
            mine = _selp(upper, values[i + half], values[i])
            send = _selp(upper, values[i], values[i + half])
            reduced.append(_max_f32(mine, _shfl_xor(send, offset)))
        values = reduced
        offset //= 2
    total = values[0]
    while offset >= 1:
        total = _max_f32(total, _shfl_xor(total, offset))
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


def _check_slots(slots: int, add_mode: int, poll_early: bool, use_pdl: bool) -> None:
    """Validate the Lamport slots options of the attn_res kernels (module docstring)."""
    if slots == 0:
        if poll_early:
            raise ValueError("poll_early needs slots")
        return
    if not 2 <= slots <= SLOTS_MAX_WORLD:
        raise ValueError(f"slots={slots} not in [2, {SLOTS_MAX_WORLD}]")
    if add_mode not in (ADD_NONE, ADD_PREFIX):
        raise ValueError(
            f"slots take the place of the addend or the layer row, not add_mode={add_mode}"
        )
    if poll_early and not use_pdl:
        raise ValueError("poll_early only applies to a PDL launch")


def cluster_size(num_candidates: int) -> int:
    """CTAs per token: the topology of the C++ kernel with the same candidate count."""
    return 1 if num_candidates <= 4 else 8


class K3AttnResKernel:
    """One CTA (``c = 1``) or one cluster of eight CTAs (``c = 8``) per token."""

    def __init__(
        self,
        num_candidates: int,
        add_mode: int,
        use_pdl: bool,
        cluster: int | None = None,
        fp8_out: bool = False,
        trigger: int = TRIGGER_AT_END,
        slots: int = 0,
        poll_early: bool = False,
        poll_ctas: int | None = None,
    ):
        if not MIN_CANDIDATES <= num_candidates <= MAX_CANDIDATES:
            raise ValueError(f"num_candidates={num_candidates} not in [2, 12]")
        if add_mode not in (ADD_NONE, ADD_PREFIX, ADD_MOE):
            raise ValueError(f"add_mode={add_mode}")
        if trigger not in (TRIGGER_AT_END, TRIGGER_AFTER_WAIT, TRIGGER_AT_ENTRY):
            raise ValueError(f"trigger={trigger}")
        _check_slots(slots, add_mode, poll_early, use_pdl)
        self.trigger = trigger
        self.n = num_candidates
        self.k = num_candidates - 1
        # ``cluster`` overrides the topology for experiments; results then follow the C++
        # kernel of that topology rather than of ``num_candidates``.
        self.c = cluster if cluster is not None else cluster_size(num_candidates)
        if self.c not in (1, 2, 4, 8):
            raise ValueError(f"cluster={self.c}")
        # Lamport slots (module docstring): ``slots`` ranks publish into a one-shot region.
        self.slots = slots
        self.poll_early = bool(poll_early)
        # c = 1 with slots: the per-token CTA runs on ``poll_ctas`` CTAs of a cluster, each
        # polling HIDDEN_SIZE / 8 / poll_ctas fragments (one per thread) and storing that share
        # of the outputs. Otherwise one CTA (c = 1) or the c-CTA cluster per token.
        self.poll_ctas = 1
        if slots and self.c == 1:
            self.poll_ctas = SLOTS_POLL_CTAS if poll_ctas is None else poll_ctas
            if self.poll_ctas not in (1, 2, 4, 8):
                raise ValueError(f"poll_ctas={self.poll_ctas}")
        self.launch_c = self.c if self.c > 1 else self.poll_ctas
        # Fragments each polling CTA of a c = 1 cluster sums (one per thread when it fits).
        self.poll_frags = HIDDEN_SIZE // 8 // self.poll_ctas
        if slots and self.c == 1 and self.poll_ctas > 1 and self.poll_frags > _THREADS:
            raise ValueError(f"poll_ctas={self.poll_ctas}: {self.poll_frags} fragments per CTA")
        self.add_mode = add_mode
        self.use_pdl = use_pdl
        self.slice = HIDDEN_SIZE // self.c
        self.items = (self.slice + _THREADS - 1) // _THREADS
        # Only the last item can be partial (c = 8: 896 = 3.5 x 256).
        self.partial_last = self.slice % _THREADS != 0
        # FP8 1x128 output: item j of the CTA covers 256 elements, i.e. two 128-element blocks
        # (warps 0-3 and 4-7); block maxima travel through shared memory.
        self.fp8_out = fp8_out
        self.amax_items = 1 << (self.items - 1).bit_length()  # padded to a power of two
        assert self.amax_items <= 32

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
        fp8_addr: Int64,
        sf_addr: Int64,
        m: Int32,
        eps: Float32,
        out_eps: Float32,
        stream: cuda.CUstream,
    ):
        c, k = self.c, self.k
        amax_slots = self.items * _WARPS

        # The structs live in this jit function: the module postpones annotations, and a plain
        # method would hand cute.struct strings (the DSL's rewrite of this function does not).
        @cute.struct
        class SharedStorage:
            # [c][warp][k] (sq, dot) snapshot partials, [c][warp] for the layer row, [c][warp]
            # output sums of squares; under c = 8 every CTA of the cluster fills its own slot g.
            snap: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * k * 2], 16]
            last: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * 2], 16]
            outsq: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS], 16]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 4], 16]

        @cute.struct
        class SharedStorageFp8:
            snap: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * k * 2], 16]
            last: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * 2], 16]
            outsq: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS], 16]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 4], 16]
            # [item][warp] |output| maxima of the FP8 output.
            amax: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, amax_slots], 16]

        # Lamport slots: the summed slots in shared memory (BF16), this CTA's slice, or on a poll
        # cluster (c = 1) the whole row, broadcast into every CTA's ``delta`` (mbarrier slot 3).
        delta_words = HIDDEN_SIZE // 2 if self._poll_cluster() else self.slice // 2

        @cute.struct
        class SharedStorageDelta:
            snap: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * k * 2], 16]
            last: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * 2], 16]
            outsq: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS], 16]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 4], 16]
            delta: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, delta_words], 16]

        @cute.struct
        class SharedStorageFp8Delta:
            snap: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * k * 2], 16]
            last: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * 2], 16]
            outsq: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS], 16]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 4], 16]
            amax: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, amax_slots], 16]
            delta: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, delta_words], 16]

        # Only the FP8 output allocates the maxima. On SM107 the BF16-output kernel faulted in
        # grids of more clusters than are resident at once when given 3472-3488 or 3984-4032 B of
        # dynamic shared memory (cause not found), sizes that a 128 B slot moved it onto for
        # N = 6 and 7; its own sizes and those of the FP8-output kernel ran clean. The c = 8
        # slots kernel adds its 1792 B slice (4640-8224 B for N = 5-12, clear of both ranges).
        if cutlass.const_expr(self.slots):
            if cutlass.const_expr(self.fp8_out):
                self.shared_storage = SharedStorageFp8Delta
            else:
                self.shared_storage = SharedStorageDelta
        elif cutlass.const_expr(self.fp8_out):
            self.shared_storage = SharedStorageFp8
        else:
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
            fp8_addr,
            sf_addr,
            m,
            eps,
            out_eps,
        )
        if cutlass.const_expr(c > 1):
            kernel.launch(
                grid=(m * c, 1, 1),
                block=(_THREADS, 1, 1),
                cluster=(c, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
                # <= 64 registers: four CTAs per SM keep up to 76 eight-CTA clusters resident.
                min_blocks_per_mp=4,
            )
        elif cutlass.const_expr(self.launch_c > 1):
            # c = 1 with slots on a poll cluster: the per-token CTA's registers, launch_c times.
            kernel.launch(
                grid=(m * self.launch_c, 1, 1),
                block=(_THREADS, 1, 1),
                cluster=(self.launch_c, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:
            kernel.launch(
                grid=(m, 1, 1),
                block=(_THREADS, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
            )

    def _poll_cluster(self) -> bool:
        """Whether the per-token CTA runs on a poll cluster (slots, c = 1, poll_ctas > 1)."""
        return bool(self.slots) and self.c == 1 and self.poll_ctas > 1

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
        fp8_addr: Int64,
        sf_addr: Int64,
        m: Int32,
        eps: Float32,
        out_eps: Float32,
    ):
        if cutlass.const_expr(self.use_pdl and self.trigger == TRIGGER_AT_ENTRY):
            _griddepcontrol_launch_dependents()
        c, k, n_cand, items = self.c, self.k, self.n, self.items
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        snap_smem = storage.snap.data_ptr().toint()
        last_smem = storage.last.data_ptr().toint()
        outsq_smem = storage.outsq.data_ptr().toint()
        mbar_smem = storage.mbar.data_ptr().toint()

        # ``launch_c`` CTAs per token: the c-CTA cluster, or (c = 1 with slots) the poll cluster,
        # whose CTAs all compute the whole row (rank 0's slice) and store a share of it.
        lc = self.launch_c
        token = bidx // lc
        cta = bidx - token * lc
        rank = cta if c > 1 else 0
        poll_cluster = self._poll_cluster()
        delta_smem = 0
        if cutlass.const_expr(self.slots):
            delta_smem = storage.delta.data_ptr().toint()

        if cutlass.const_expr(c > 1):
            if tidx == 0:
                for r in cutlass.range_constexpr(3):
                    _mbar_init(mbar_smem + r * 8, 1)
                _fence_mbar_init()
            _cluster_arrive_relaxed()
        if cutlass.const_expr(poll_cluster):
            if tidx == 0:
                _mbar_init(mbar_smem + 24, 1)
                _fence_mbar_init()
            _cluster_arrive_relaxed()

        # Element offsets (bytes) of this thread's items within a row.
        row = Int64(token) * (HIDDEN_SIZE * 2)
        h_off = [Int64(rank * self.slice + tidx + _THREADS * j) * 2 for j in range(items)]
        valid_last = Int32(1)
        if cutlass.const_expr(self.partial_last):
            valid_last = Int32(tidx + _THREADS * (items - 1) < self.slice)

        def ld_item(base, j, nc):
            if cutlass.const_expr(self.partial_last and j == items - 1):
                return _ld_bf16_if(valid_last, base + h_off[j], nc)
            if cutlass.const_expr(nc):
                return _ld_bf16_nc(base + h_off[j])
            return _ld_bf16(base + h_off[j])

        def store_pred(j):
            """Whether this CTA stores item ``j`` (a poll cluster splits the items over its CTAs)."""
            pred = Int32(1)
            if cutlass.const_expr(self.partial_last and j == items - 1):
                pred = valid_last
            if cutlass.const_expr(poll_cluster):
                pred = pred * Int32(cta == j % lc)
            return pred

        def st_item(base, j, value):
            if cutlass.const_expr(poll_cluster or (self.partial_last and j == items - 1)):
                _st_bf16_if(store_pred(j), base + h_off[j], value)
            else:
                _st_bf16(base + h_off[j], value)

        # ---- prologue: everything that does not depend on the previous grid ----
        q = [_mul_ftz(ld_item(res_addr, j, True), ld_item(rms_addr, j, True)) for j in range(items)]
        snaps = []
        plane = Int64(m) * (HIDDEN_SIZE * 2)
        for n in cutlass.range_constexpr(k):
            base = block_addr + plane * n + row
            snaps.append([ld_item(base, j, True) for j in range(items)])
        prefix = []
        if cutlass.const_expr(self.add_mode != ADD_NONE):
            prefix = [ld_item(layer_addr + row, j, True) for j in range(items)]
        out_w = [ld_item(out_w_addr, j, True) for j in range(items)]

        # Per-thread (sum of squares, dot) partials of the snapshots, in item order.
        fma = _fma if c == 1 else _fma_ftz
        add_tree = _add if c == 1 else _add_ftz
        snap_part = []
        for n in cutlass.range_constexpr(k):
            sq = _zero_f32()
            dot = _zero_f32()
            for j in cutlass.range_constexpr(items):
                sq = fma(snaps[n][j], snaps[n][j], sq)
                dot = fma(snaps[n][j], q[j], dot)
            for off in (16, 8, 4, 2, 1):
                sq = add_tree(sq, _shfl_down(sq, off))
                dot = add_tree(dot, _shfl_down(dot, off))
            snap_part.append((sq, dot))

        if cutlass.const_expr(c == 1):
            if lane == 0:
                for n in cutlass.range_constexpr(k):
                    _sts_v2(snap_smem + (warp * k + n) * 8, snap_part[n][0], snap_part[n][1])
        else:
            # Every peer's mbarriers are initialized once the cluster barrier completes.
            _cluster_wait()
            if tidx == 0:
                _mbar_arrive_expect_tx(mbar_smem, c * _WARPS * k * 8)
                _mbar_arrive_expect_tx(mbar_smem + 8, c * _WARPS * 8)
                _mbar_arrive_expect_tx(mbar_smem + 16, c * _WARPS * 4)
            # Lane r < c of each warp sends the warp's partials to CTA r (slot [rank][warp]).
            for n in cutlass.range_constexpr(k):
                sq = _shfl_idx(snap_part[n][0], 0)
                dot = _shfl_idx(snap_part[n][1], 0)
                if lane < c:
                    _st_async_v2(
                        snap_smem + ((rank * _WARPS + warp) * k + n) * 8, lane, mbar_smem, sq, dot
                    )

        # Lamport slots: the summed row (``add_addr`` is the mailbox region). Polled before the
        # wait with ``poll_early``; every store below stays after the wait.
        delta = []
        if cutlass.const_expr(self.slots and self.poll_early):
            delta = self._slots_delta(add_addr, token, cta, rank, tidx, mbar_smem, delta_smem)

        # ---- wait for the previous grid ----
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
            if cutlass.const_expr(self.trigger == TRIGGER_AFTER_WAIT):
                _griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.slots and not self.poll_early):
            delta = self._slots_delta(add_addr, token, cta, rank, tidx, mbar_smem, delta_smem)

        if cutlass.const_expr(self.add_mode == ADD_PREFIX):
            # bf16(prefix + addend): the C++ kernels' fused add (fast-math add.ftz). All loads
            # are issued before the first store (a store may alias a later load, so they are
            # not reordered).
            if cutlass.const_expr(self.slots):
                addend = delta
            else:
                addend = [ld_item(add_addr + row, j, False) for j in range(items)]
            x = []
            for j in cutlass.range_constexpr(items):
                v = _round_bf16(_add_ftz(prefix[j], addend[j]))
                st_item(upd_addr + row, j, v)
                x.append(v)
        elif cutlass.const_expr(self.add_mode == ADD_MOE):
            # bf16(residual + bf16(routed + shared)): the MoE residual update of
            # kimi_k3_moe_residual._moe_residual_add_kernel (Triton, add.rn).
            routed = [ld_item(add_addr + row, j, False) for j in range(items)]
            shared = [ld_item(add2_addr + row, j, False) for j in range(items)]
            x = []
            for j in cutlass.range_constexpr(items):
                moe = _round_bf16(_add(routed[j], shared[j]))
                v = _round_bf16(_add(prefix[j], moe))
                st_item(upd_addr + row, j, v)
                x.append(v)
        elif cutlass.const_expr(self.slots):
            # The summed row is the layer row and the updated prefix (the all-reduce output).
            x = delta
            for j in cutlass.range_constexpr(items):
                st_item(upd_addr + row, j, x[j])
        else:
            x = [ld_item(layer_addr + row, j, False) for j in range(items)]

        sq = _zero_f32()
        dot = _zero_f32()
        for j in cutlass.range_constexpr(items):
            sq = fma(x[j], x[j], sq)
            dot = fma(x[j], q[j], dot)
        for off in (16, 8, 4, 2, 1):
            sq = add_tree(sq, _shfl_down(sq, off))
            dot = add_tree(dot, _shfl_down(dot, off))

        # Totals: lane n < N of every warp reduces candidate n in the C++ order
        # (0 + sum over warps, then, for c = 8, 0 + sum over CTAs).
        tot_sq = _zero_f32()
        tot_dot = _zero_f32()
        if cutlass.const_expr(c == 1):
            if lane == 0:
                _sts_v2(last_smem + warp * 8, sq, dot)
            _bar_sync(1)
            if lane < k:
                for w in cutlass.range_constexpr(_WARPS):
                    a, b = _lds_v2(snap_smem + (w * k + lane) * 8)
                    tot_sq = _add(tot_sq, a)
                    tot_dot = _add(tot_dot, b)
            if lane == k:
                for w in cutlass.range_constexpr(_WARPS):
                    a, b = _lds_v2(last_smem + w * 8)
                    tot_sq = _add(tot_sq, a)
                    tot_dot = _add(tot_dot, b)
        else:
            sq = _shfl_idx(sq, 0)
            dot = _shfl_idx(dot, 0)
            if lane < c:
                _st_async_v2(last_smem + (rank * _WARPS + warp) * 8, lane, mbar_smem + 8, sq, dot)
            _mbar_wait_cluster(mbar_smem)
            _mbar_wait_cluster(mbar_smem + 8)
            if lane < k:
                for g in cutlass.range_constexpr(c):
                    cta_sq = _zero_f32()
                    cta_dot = _zero_f32()
                    for w in cutlass.range_constexpr(_WARPS):
                        a, b = _lds_v2(snap_smem + ((g * _WARPS + w) * k + lane) * 8)
                        cta_sq = _add(cta_sq, a)
                        cta_dot = _add(cta_dot, b)
                    tot_sq = _add(tot_sq, cta_sq)
                    tot_dot = _add(tot_dot, cta_dot)
            if lane == k:
                for g in cutlass.range_constexpr(c):
                    cta_sq = _zero_f32()
                    cta_dot = _zero_f32()
                    for w in cutlass.range_constexpr(_WARPS):
                        a, b = _lds_v2(last_smem + (g * _WARPS + w) * 8)
                        cta_sq = _add(cta_sq, a)
                        cta_dot = _add(cta_dot, b)
                    tot_sq = _add(tot_sq, cta_sq)
                    tot_dot = _add(tot_dot, cta_dot)

        # Softmax over the N candidates (every lane ends with all N weights).
        rs = _rsigma(tot_sq, eps)
        logit = _mul_ftz(tot_dot, rs)
        logits = [_shfl_idx(logit, n) for n in range(n_cand)]
        max_logit = _neg_flt_max()
        for n in cutlass.range_constexpr(n_cand):
            max_logit = _max_ftz(max_logit, logits[n])
        terms = [_softmax_term(logits[n], max_logit) for n in range(n_cand)]
        den = _zero_f32()
        for n in cutlass.range_constexpr(n_cand):
            den = _add_ftz(den, terms[n])
        inv = _rcp_ftz(den)
        probs = [_mul_ftz(terms[n], inv) for n in range(n_cand)]

        # Mix, BF16 rounding, output sum of squares.
        mixed = []
        out_sq = _zero_f32()
        for j in cutlass.range_constexpr(items):
            val = _zero_f32()
            for n in cutlass.range_constexpr(k):
                val = _fma_ftz(probs[n], snaps[n][j], val)
            val = _fma_ftz(probs[k], x[j], val)
            val = _round_bf16(val)
            mixed.append(val)
            out_sq = _fma_ftz(val, val, out_sq)
        out_sq = _warp_sum_bfly_ftz(out_sq)

        total = _zero_f32()
        if cutlass.const_expr(c == 1):
            if lane == 0:
                _sts_f32(outsq_smem + warp * 4, out_sq)
            _bar_sync(2)
            for w in cutlass.range_constexpr(_WARPS):
                total = _add_ftz(total, _lds_f32(outsq_smem + w * 4))
        else:
            if lane < c:
                _st_async_b32(outsq_smem + (rank * _WARPS + warp) * 4, lane, mbar_smem + 16, out_sq)
            _mbar_wait_cluster(mbar_smem + 16)
            for g in cutlass.range_constexpr(c):
                cta = _zero_f32()
                for w in cutlass.range_constexpr(_WARPS):
                    cta = _add_ftz(cta, _lds_f32(outsq_smem + (g * _WARPS + w) * 4))
                total = _add_ftz(total, cta)
        out_rs = _rsigma(total, out_eps)

        stored = []
        for j in cutlass.range_constexpr(items):
            y = _round_bf16(_mul_ftz(mixed[j], out_rs))
            out = _mul_ftz(y, out_w[j])
            st_item(out_addr + row, j, out)
            if cutlass.const_expr(self.fp8_out):
                stored.append(_round_bf16(out))  # the value st_item stored

        if cutlass.const_expr(self.fp8_out):
            # Per item: the warp's |output| maximum (one transpose-max over all items), then the
            # maximum of the four warps that share a 128-element block (warps 0-3 | 4-7).
            amax_smem = storage.amax.data_ptr().toint()
            maxima = [_abs_max(_zero_f32(), v) for v in stored]
            for _ in cutlass.range_constexpr(self.amax_items - items):
                maxima.append(_zero_f32())
            shift = 5 - (self.amax_items.bit_length() - 1)
            warp_max = _warp_transpose_max(maxima, lane)
            if (lane & ((1 << shift) - 1)) == 0:
                if (lane >> shift) < items:
                    _sts_f32(amax_smem + ((lane >> shift) * _WARPS + warp) * 4, warp_max)
            _bar_sync(5)
            half = warp >> 2
            token_row = Int64(token) * HIDDEN_SIZE
            for j in cutlass.range_constexpr(items):
                block_max = _zero_f32()
                for w in cutlass.range_constexpr(4):
                    block_max = _max_f32(
                        block_max, _lds_f32(amax_smem + (j * _WARPS + half * 4 + w) * 4)
                    )
                e8, qscale = _ue8m0_scales(block_max)
                elem = rank * self.slice + tidx + _THREADS * j
                valid_j = store_pred(j)
                _stg_b8_if(
                    valid_j, fp8_addr + token_row + Int64(elem), _e4m3_byte(stored[j], qscale)
                )
                # Thread 128 * half + j stores the scale word of (item j, half): one store per
                # thread rather than ``items`` stores by one thread per half.
                lead = valid_j * Int32((tidx & 127) == j)
                sf_block = (rank * self.slice + _THREADS * j) // 128 + half
                _stg_b32_if(lead, sf_addr + _sf_offset(token, sf_block), e8 * 0x01010101)
            # Padding scale rows of this token (``_pad_scale_rows``) x this CTA's blocks, spread
            # over all threads (one word each per round; on a poll cluster by its first CTA).
            pad_lead = Int32(1)
            if cutlass.const_expr(poll_cluster):
                pad_lead = Int32(cta == 0)
            blocks = self.slice // 128
            pad_words = _pad_scale_rows(m, token) * blocks
            for it in cutlass.range(0, (pad_words - tidx + _THREADS - 1) // _THREADS, 1, unroll=1):
                word = tidx + it * _THREADS
                pad_row = word // blocks
                _stg_b32_if(
                    pad_lead,
                    sf_addr
                    + _sf_offset(m + token + pad_row * m, rank * blocks + word - pad_row * blocks),
                    Int32(0),
                )

        if cutlass.const_expr(self.use_pdl and self.trigger == TRIGGER_AT_END):
            _bar_sync(3)
            _griddepcontrol_launch_dependents()

    def _slots_delta(self, mbox, token, cta, rank, tidx, mbar_smem, delta_smem):
        """This thread's items of ``bf16(sum_r slot[r])`` (module docstring), as FP32 values.

        Plain Python at trace time (no dynamic control flow: predicated PTX only). ``mbox`` is
        this rank's copy of the one-shot region ``[M][W][7168]`` (BF16). The CTA polls its slice
        as contiguous 16-byte fragments, thread ``t`` fragments ``t, t + 256, ...``
        (``poll_sum_slots_v4``: MNNVL order, re-armed by the poller), so that a warp reads 512
        contiguous bytes of each slot, and the sums meet in the shared ``delta`` slice
        (``delta_smem``); each thread then reads its items from it. A poll cluster (``c = 1``)
        polls one fragment per thread on each CTA and broadcasts the sums into every CTA's
        ``delta`` row (counted by mbarrier slot 3). Keep each 128-byte line of a slot within one
        warp's poll: splitting lines between warps made the c = 8 chain slower on
        alternate sites."""
        world = self.slots
        stride = HIDDEN_SIZE * 2
        tok = mbox + Int64(token) * (world * HIDDEN_SIZE * 2)
        out = []
        if self._poll_cluster():
            mbar_delta = mbar_smem + 24
            frag = cta * self.poll_frags + tidx
            valid = Int32(tidx < self.poll_frags)
            words = poll_sum_slots_v4(valid, tok + Int64(frag) * 16, world, stride)
            # Every CTA's barrier is initialized once the cluster barrier completes.
            _cluster_wait()
            _mbar_arrive_expect_tx_if(Int32(tidx == 0), mbar_delta, HIDDEN_SIZE * 2)
            for r in range(self.launch_c):
                _st_async_v4_if(valid, delta_smem + frag * 16, r, mbar_delta, words)
            _mbar_wait_cluster(mbar_delta)
            for j in range(self.items):
                out.append(_lds_bf16(delta_smem + (tidx + _THREADS * j) * 2))
        else:
            frags = self.slice // 8
            first = tok + Int64(rank * self.slice) * 2
            for f0 in range(0, frags, _THREADS):
                valid = Int32(tidx + f0 < frags)
                words = poll_sum_slots_v4(valid, first + Int64(tidx + f0) * 16, world, stride)
                _sts_v4_if(valid, delta_smem + (tidx + f0) * 16, words)
            _bar_sync(4)
            for j in range(self.items):
                addr = delta_smem + (tidx + _THREADS * j) * 2
                if self.partial_last and j == self.items - 1:
                    out.append(_lds_bf16_if(Int32(tidx + _THREADS * j < self.slice), addr))
                else:
                    out.append(_lds_bf16(addr))
        return out


class K3AttnResFastKernel:
    """Vectorized variant: a cluster of ``c`` CTAs per token, one 16-byte vector per thread.

    Each CTA owns ``7168 / c`` contiguous elements and each thread eight of them (one 16-byte
    load per row), so a token's rows are read with ``8 x`` fewer instructions than by the
    per-element layout of the C++ kernels, and the statistics are summed in a different order:
    the results are not bit-identical to the C++ kernels (the updated prefix sum, an elementwise
    add, is). Before ``griddepcontrol.wait`` the snapshots' statistics, logits and their part of
    the softmax and of the mixture (``sum_n exp(logit_n - m) v_n``) are complete; after it the
    kernel reduces two numbers of the layer row (one cluster round), rescales the mixture and
    reduces its sum of squares (a second round).
    """

    def __init__(
        self,
        num_candidates: int,
        add_mode: int,
        use_pdl: bool,
        cluster: int = 4,
        fp8_out: bool = False,
        trigger: int = TRIGGER_AT_END,
        slots: int = 0,
        poll_early: bool = False,
    ):
        if not MIN_CANDIDATES <= num_candidates <= MAX_CANDIDATES:
            raise ValueError(f"num_candidates={num_candidates} not in [2, 12]")
        if add_mode not in (ADD_NONE, ADD_PREFIX, ADD_MOE) or cluster not in (1, 2, 4, 8):
            raise ValueError(f"add_mode={add_mode} cluster={cluster}")
        if trigger not in (TRIGGER_AT_END, TRIGGER_AFTER_WAIT, TRIGGER_AT_ENTRY):
            raise ValueError(f"trigger={trigger}")
        _check_slots(slots, add_mode, poll_early, use_pdl)
        self.trigger = trigger
        # Lamport slots (module docstring): each thread polls its own 16-byte vector.
        self.slots = slots
        self.poll_early = bool(poll_early)
        self.n = num_candidates
        self.k = num_candidates - 1
        self.c = cluster
        self.add_mode = add_mode
        self.use_pdl = use_pdl
        # Also store the output as FP8 1x128 codes + R128c4 UE8M0 scales (sixteen threads per
        # 128-element block: a CTA slice of 7168 / c elements is a whole number of blocks).
        self.fp8_out = fp8_out
        self.vecs = HIDDEN_SIZE // 8 // cluster  # 16-byte vectors per CTA, one per thread
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
        fp8_addr: Int64,
        sf_addr: Int64,
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
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 4], 16]

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
            fp8_addr,
            sf_addr,
            m,
            eps,
            out_eps,
        )
        if cutlass.const_expr(self.c > 1):
            kernel.launch(
                grid=(m * self.c, 1, 1),
                block=(self.threads, 1, 1),
                cluster=(self.c, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:
            kernel.launch(
                grid=(m, 1, 1),
                block=(self.threads, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
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
        fp8_addr: Int64,
        sf_addr: Int64,
        m: Int32,
        eps: Float32,
        out_eps: Float32,
    ):
        if cutlass.const_expr(self.use_pdl and self.trigger == TRIGGER_AT_ENTRY):
            _griddepcontrol_launch_dependents()
        c, k, parts, nvals = self.c, self.k, self.parts, self.nvals
        warps, shift = self.warps, self.shift
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        snap_smem = storage.snap.data_ptr().toint()
        last_smem = storage.last.data_ptr().toint()
        outsq_smem = storage.outsq.data_ptr().toint()
        mbar_smem = storage.mbar.data_ptr().toint()

        token = bidx // c
        rank = bidx - token * c
        if cutlass.const_expr(c > 1):
            if tidx == 0:
                for r in cutlass.range_constexpr(3):
                    _mbar_init(mbar_smem + r * 8, 1)
                _fence_mbar_init()
            _cluster_arrive_relaxed()

        valid = Int32(1)
        if cutlass.const_expr(self.vecs != self.threads):
            valid = Int32(tidx < self.vecs)
        col = Int64(rank * self.vecs + tidx) * 16
        row = Int64(token) * (HIDDEN_SIZE * 2)

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

        # ---- prologue: everything that does not depend on the previous grid ----
        res_w = unpack(load(res_addr + col, True))
        rms_w = unpack(load(rms_addr + col, True))
        out_w = load(out_w_addr + col, True)
        plane = Int64(m) * (HIDDEN_SIZE * 2)
        snaps = [load(block_addr + plane * n + row + col, True) for n in range(k)]
        prefix = []
        if cutlass.const_expr(self.add_mode != ADD_NONE):
            prefix = load(layer_addr + row + col, True)
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
        if cutlass.const_expr(c == 1):
            if sub == 0:
                if vidx < nvals:
                    _sts_f32(snap_smem + (warp * nvals + vidx) * 4, warp_stat)
            _bar_sync(1, self.threads)
        else:
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
            logit = _mul_ftz(tot_dot, _rsigma(tot_sq, eps))
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

        # Lamport slots: this thread's vector of the summed row (``add_addr`` is the mailbox
        # region); polled before the wait with ``poll_early``, stored after it.
        summed = []
        if cutlass.const_expr(self.slots and self.poll_early):
            summed = self._slots_vector(add_addr, token, rank, tidx, valid)

        # ---- wait for the previous grid ----
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
            if cutlass.const_expr(self.trigger == TRIGGER_AFTER_WAIT):
                _griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.slots and not self.poll_early):
            summed = self._slots_vector(add_addr, token, rank, tidx, valid)

        if cutlass.const_expr(self.add_mode == ADD_PREFIX):
            if cutlass.const_expr(self.slots):
                addend = unpack(summed)
            else:
                addend = unpack(load(add_addr + row + col, False))
            pre = unpack(prefix)
            x = [_round_bf16(_add_ftz(pre[e], addend[e])) for e in range(8)]
        elif cutlass.const_expr(self.add_mode == ADD_MOE):
            routed = unpack(load(add_addr + row + col, False))
            shared = unpack(load(add2_addr + row + col, False))
            pre = unpack(prefix)
            x = [
                _round_bf16(_add(pre[e], _round_bf16(_add(routed[e], shared[e])))) for e in range(8)
            ]
        elif cutlass.const_expr(self.slots):
            x = unpack(summed)
        else:
            x = unpack(load(layer_addr + row + col, False))
        if cutlass.const_expr(self.add_mode != ADD_NONE or self.slots):
            # The updated prefix; with slots and ADD_NONE the summed row (the all-reduce output).
            words = [_pack_bf16x2(x[2 * i], x[2 * i + 1]) for i in range(4)]
            _stg_v4_if(valid, upd_addr + row + col, words)

        sq = _zero_f32()
        dot = _zero_f32()
        for e in cutlass.range_constexpr(8):
            sq = _fma(x[e], x[e], sq)
            dot = _fma(x[e], q[e], dot)
        # Lanes 0-15 hold the warp's sum of squares, lanes 16-31 its dot product.
        pair = _warp_transpose_sum([sq, dot], lane)
        half = lane & 15
        if cutlass.const_expr(c == 1):
            if half == 0:
                _sts_f32(last_smem + (warp * 2 + (lane >> 4)) * 4, pair)
            _bar_sync(2, self.threads)
        else:
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
        last_logit = _mul_ftz(tot_dot, _rsigma(tot_sq, eps))

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
        if cutlass.const_expr(c == 1):
            if lane == 0:
                _sts_f32(outsq_smem + warp * 4, out_sq)
            _bar_sync(3, self.threads)
        else:
            if lane < c:
                _st_async_b32(outsq_smem + (rank * warps + warp) * 4, lane, mbar_smem + 16, out_sq)
            _mbar_wait_cluster(mbar_smem + 16)
        part = _zero_f32()
        if lane < parts:
            part = _lds_f32(outsq_smem + lane * 4)
        out_rs = _rsigma(_warp_sum(part), out_eps)

        w_out = unpack(out_w)
        y = [_mul_ftz(_round_bf16(_mul_ftz(mixed[e], out_rs)), w_out[e]) for e in range(8)]
        words = [_pack_bf16x2(y[2 * i], y[2 * i + 1]) for i in range(4)]
        _stg_v4_if(valid, out_addr + row + col, words)

        if cutlass.const_expr(self.fp8_out):
            # The stored BF16 values; lanes 16i..16i+15 hold one 128-element block.
            stored = unpack(words)
            amax = _zero_f32()
            for e in cutlass.range_constexpr(8):
                amax = _abs_max(amax, stored[e])
            for off in (8, 4, 2, 1):
                amax = _max_f32(amax, _shfl_xor(amax, off))
            e8, qscale = _ue8m0_scales(amax)
            w0, w1 = _e4m3x8(stored, qscale)
            vec = rank * self.vecs + tidx
            _stg_v2_if(valid, fp8_addr + Int64(token) * HIDDEN_SIZE + Int64(vec) * 8, w0, w1)
            lead = valid * Int32((tidx & 15) == 0)
            sf_block = vec >> 4
            _stg_b32_if(lead, sf_addr + _sf_offset(token, sf_block), e8 * 0x01010101)
            # Padding scale rows of this token: the block's sixteen threads take every 16th.
            pad_lane = tidx & 15
            pad_rows = _pad_scale_rows(m, token)
            for it in cutlass.range(0, (pad_rows - pad_lane + 15) // 16, 1, unroll=1):
                pad_row = pad_lane + it * 16
                _stg_b32_if(
                    valid, sf_addr + _sf_offset(m + token + pad_row * m, sf_block), Int32(0)
                )

        if cutlass.const_expr(self.use_pdl and self.trigger == TRIGGER_AT_END):
            _bar_sync(4, self.threads)
            _griddepcontrol_launch_dependents()

    def _slots_vector(self, mbox, token, rank, tidx, valid):
        """The four BF16x2 words of this thread's vector of ``bf16(sum_r slot[r])`` (plain Python
        at trace time; ``poll_sum_slots_v4`` polls, sums in the MNNVL order and re-arms)."""
        world = self.slots
        tok = mbox + Int64(token) * (world * HIDDEN_SIZE * 2)
        col = Int64(rank * self.vecs + tidx) * 16
        return poll_sum_slots_v4(valid, tok + col, world, HIDDEN_SIZE * 2)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


#: The bit-identical per-element kernel instead of the vectorized one.
_EXACT = os.environ.get("TRTLLM_K3_ATTN_RES_PDL_EXACT", "0") == "1"


def fast_cluster_size(num_tokens: int, num_candidates: int) -> int:
    """CTAs per token of :class:`K3AttnResFastKernel`.

    Measured on GB300 behind a PDL producer: eight CTAs per token are fastest for a handful of
    tokens with at least three snapshots (smaller per-CTA prologue), four from ~32 tokens on
    (eight-CTA clusters then fill the machine) and for two or three candidates.
    """
    return 8 if num_tokens <= 16 and num_candidates >= 4 else 4


def early_trigger_mode() -> int:
    """``TRTLLM_K3_ATTN_RES_EARLY_TRIGGER`` (0, 1 or 2; 0 where the MK1 ports are unsupported)."""
    from ...modules.kimi_k3_mk1_gate import mk1_value

    return mk1_value(EARLY_TRIGGER_ENV, (TRIGGER_AT_END, TRIGGER_AFTER_WAIT, TRIGGER_AT_ENTRY))


@functools.cache
def _compiled(
    num_candidates: int,
    add_mode: int,
    use_pdl: bool,
    cluster: int | None,
    exact: bool,
    fp8_out: bool = False,
    trigger: int = TRIGGER_AT_END,
    slots: int = 0,
    poll_early: bool = False,
    poll_ctas: int | None = None,
    spin_limit: int = 0,
):
    # ``spin_limit`` keys the cache on the poll trap compiled into slots kernels
    # (TRTLLM_K3_LAMPORT_SPIN_LIMIT, read by the Lamport primitives at trace time).
    del spin_limit
    slot_kw = dict(trigger=trigger, slots=slots, poll_early=poll_early)
    if exact:
        kernel = K3AttnResKernel(
            num_candidates,
            add_mode,
            use_pdl,
            cluster,
            fp8_out=fp8_out,
            poll_ctas=poll_ctas,
            **slot_kw,
        )
    else:
        kernel = K3AttnResFastKernel(
            num_candidates, add_mode, use_pdl, cluster, fp8_out=fp8_out, **slot_kw
        )
    addr = Int64(1 << 20)
    return cute.compile(
        kernel,
        addr,
        addr,
        addr,
        addr,
        addr,
        addr,
        addr,
        addr,
        addr,
        addr,
        addr,
        Int32(1),
        Float32(1e-6),
        Float32(1e-6),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    # SM100 / SM103 (Blackwell) and SM107 (Rubin): the kernels only use clusters, st.async and
    # mbarriers.
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def can_use_k3_attn_res(
    layer: torch.Tensor,
    addends: tuple[torch.Tensor, ...],
    block: torch.Tensor,
    *weights: torch.Tensor,
) -> bool:
    """Whether :func:`run_k3_attn_res` accepts ``layer [M, H]``, 0-2 addends ``[M, H]`` and
    ``block [N - 1, M, H]``."""
    if layer.dim() != 2 or len(addends) > 2:
        return False
    return _can_use_rows(layer.shape[0], (layer, *addends), block, weights)


def can_use_k3_attn_res_slots(
    m: int,
    prefix: torch.Tensor | None,
    block: torch.Tensor,
    *weights: torch.Tensor,
) -> bool:
    """Whether :func:`run_k3_attn_res_slots` accepts ``m`` rows with ``prefix [M, H]`` (or None)
    and ``block [N - 1, M, H]``."""
    return _can_use_rows(m, () if prefix is None else (prefix,), block, weights)


def _can_use_rows(
    m: int,
    rows: tuple[torch.Tensor, ...],
    block: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
) -> bool:
    if not block.is_cuda or not _is_supported_device(block.device.index or 0):
        return False
    if not 1 <= m <= MAX_TOKENS or any(tuple(r.shape) != (m, HIDDEN_SIZE) for r in rows):
        return False
    if block.dim() != 3 or tuple(block.shape[1:]) != (m, HIDDEN_SIZE):
        return False
    if not MIN_CANDIDATES <= block.shape[0] + 1 <= MAX_CANDIDATES:
        return False
    for t in (*rows, block, *weights):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.device != block.device:
            return False
    return all(w.numel() == HIDDEN_SIZE for w in weights)


def _check_fp8_buffers(layer: torch.Tensor, fp8_output: torch.Tensor, fp8_scale: torch.Tensor):
    """The FP8 outputs are written by the kernel: reject any buffer it would overrun."""
    m = layer.shape[0]
    if (
        fp8_output.dtype is not torch.float8_e4m3fn
        or fp8_output.device != layer.device
        or not fp8_output.is_contiguous()
        or tuple(fp8_output.shape) != (m, HIDDEN_SIZE)
    ):
        raise ValueError(
            f"fp8_output must be a contiguous float8_e4m3fn [{m}, {HIDDEN_SIZE}] tensor on "
            f"{layer.device}, got {fp8_output.dtype} {tuple(fp8_output.shape)} on {fp8_output.device}"
        )
    if (
        fp8_scale.dtype is not torch.uint8
        or fp8_scale.device != layer.device
        or not fp8_scale.is_contiguous()
        or fp8_scale.numel() != fp8_scale_numel(m)
    ):
        raise ValueError(
            f"fp8_scale must be a contiguous uint8 tensor of {fp8_scale_numel(m)} elements on "
            f"{layer.device}, got {fp8_scale.dtype} {fp8_scale.numel()} on {fp8_scale.device}"
        )


def run_k3_attn_res(
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
    use_pdl: bool = _USE_PDL,
    cluster: int | None = None,
    exact: bool | None = None,
    fp8_output: torch.Tensor | None = None,
    fp8_scale: torch.Tensor | None = None,
    trigger: int | None = None,
) -> None:
    """Launch on the current stream (inputs are not re-validated; see :func:`can_use_k3_attn_res`).

    Args:
        layer: ``bf16 [M, 7168]``: the layer row without addends, else the prefix sum.
        addends: ``()``; ``(addend,)`` for ``bf16(layer + addend)``; ``(routed, shared)`` for
            ``bf16(layer + bf16(routed + shared))`` (all ``bf16 [M, 7168]``).
        block: ``bf16 [N - 1, M, 7168]`` snapshot rows, ``2 <= N <= 12``.
        res_weight, rms_weight: Score projection and its RMSNorm weight, ``bf16 [7168]``.
        out_weight: Weight of the trailing RMSNorm, ``bf16 [7168]``.
        rms_eps, out_eps: Epsilons of the two norms.
        output: ``bf16 [M, 7168]`` normalized mixture.
        updated: ``bf16 [M, 7168]`` updated prefix sum; required with addends.
        use_pdl: Launch with programmatic dependent launch.
        cluster: CTAs per token (default: :func:`cluster_size` / :func:`fast_cluster_size`); for
            ``exact``, another value than the default is for experiments.
        exact: The per-element kernel (bit-identical to the C++ ops) instead of the vectorized
            one (identical updated prefix; the statistics are summed in another order); default
            ``TRTLLM_K3_ATTN_RES_PDL_EXACT``.
        fp8_output: ``float8_e4m3fn [M, 7168]``: also store ``output`` quantized for an FP8
            block-scale GEMM, bit-identical to ``fp8_quantize_1x128_cutedsl_ue8m0(output)`` /
            ``fp8_quantize_1x128_packed_ue8m0(output)`` (1x128 blocks, UE8M0 scales).
        fp8_scale: ``uint8 [fp8_scale_numel(M)]``: the R128c4 scales of ``fp8_output`` (padding
            rows written as zero); required with ``fp8_output``.
        trigger: Where the kernel triggers its dependents (``TRIGGER_*``; the results do not
            depend on it); default ``TRTLLM_K3_ATTN_RES_EARLY_TRIGGER`` (0 off SM107).
    """
    if exact is None:
        exact = _EXACT
    if trigger is None:
        trigger = early_trigger_mode()
    if (fp8_output is None) != (fp8_scale is None):
        raise ValueError("fp8_output and fp8_scale go together")
    fp8_out = fp8_output is not None
    if fp8_out:
        _check_fp8_buffers(layer, fp8_output, fp8_scale)
    add_mode = len(addends)
    if cluster is None and not exact:
        cluster = fast_cluster_size(layer.shape[0], block.shape[0] + 1)
    compiled = _compiled(block.shape[0] + 1, add_mode, use_pdl, cluster, exact, fp8_out, trigger)
    compiled(
        Int64(layer.data_ptr()),
        Int64(addends[0].data_ptr() if add_mode >= 1 else 0),
        Int64(addends[1].data_ptr() if add_mode == 2 else 0),
        Int64(block.data_ptr()),
        Int64(res_weight.data_ptr()),
        Int64(rms_weight.data_ptr()),
        Int64(out_weight.data_ptr()),
        Int64(output.data_ptr()),
        Int64(updated.data_ptr() if add_mode else 0),
        Int64(fp8_output.data_ptr() if fp8_out else 0),
        Int64(fp8_scale.data_ptr() if fp8_out else 0),
        Int32(layer.shape[0]),
        Float32(rms_eps),
        Float32(out_eps),
        cuda.CUstream(torch.cuda.current_stream(layer.device).cuda_stream),
    )


def run_k3_attn_res_slots(
    prefix: torch.Tensor | None,
    mailbox: int,
    world: int,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
    output: torch.Tensor,
    updated: torch.Tensor,
    *,
    use_pdl: bool = _USE_PDL,
    cluster: int | None = None,
    exact: bool | None = None,
    fp8_output: torch.Tensor | None = None,
    fp8_scale: torch.Tensor | None = None,
    trigger: int | None = None,
    poll_early: bool = False,
    poll_ctas: int | None = None,
) -> None:
    """:func:`run_k3_attn_res` with the addend (or the layer row) summed from Lamport slots.

    ``mailbox`` is this rank's address of a one-shot region ``[M][world][7168]`` (BF16) into
    which every rank of the group published its partial for the ``M = output.shape[0]`` rows
    (``K3LamportArena.local_addr(parity)``); the kernel polls, sums in the MNNVL one-shot order
    and re-arms it. ``prefix`` given: ``updated = bf16(prefix + sum)`` (``ADD_PREFIX``);
    ``prefix`` None: the sum is the layer row and is stored as ``updated`` (``ADD_NONE``). The
    outputs are bit-identical to :func:`run_k3_attn_res` given ``sum`` as the addend or the layer
    (the same kernel, topology and arithmetic). ``poll_early`` polls before
    ``griddepcontrol.wait``; ``poll_ctas`` sets the ``c = 1`` poll cluster (exact kernel).
    """
    if exact is None:
        exact = _EXACT
    if trigger is None:
        trigger = early_trigger_mode()
    if (fp8_output is None) != (fp8_scale is None):
        raise ValueError("fp8_output and fp8_scale go together")
    fp8_out = fp8_output is not None
    m = output.shape[0]
    weights = (res_weight, rms_weight, out_weight)
    if not can_use_k3_attn_res_slots(m, prefix, block, *weights) or not _can_use_rows(
        m, (output, updated), block, weights
    ):
        raise ValueError(
            f"k3_attn_res slots: prefix / block / weights / outputs do not fit {m} rows of "
            f"{HIDDEN_SIZE} (contiguous bf16 on one supported device)"
        )
    if fp8_out:
        _check_fp8_buffers(output, fp8_output, fp8_scale)
    add_mode = ADD_NONE if prefix is None else ADD_PREFIX
    if cluster is None and not exact:
        cluster = fast_cluster_size(m, block.shape[0] + 1)
    from ..rubin import k3_lamport

    compiled = _compiled(
        block.shape[0] + 1,
        add_mode,
        use_pdl,
        cluster,
        exact,
        fp8_out,
        trigger,
        world,
        poll_early,
        poll_ctas,
        k3_lamport.SPIN_LIMIT,
    )
    compiled(
        Int64(prefix.data_ptr() if prefix is not None else 0),
        Int64(mailbox),
        Int64(0),
        Int64(block.data_ptr()),
        Int64(res_weight.data_ptr()),
        Int64(rms_weight.data_ptr()),
        Int64(out_weight.data_ptr()),
        Int64(output.data_ptr()),
        Int64(updated.data_ptr()),
        Int64(fp8_output.data_ptr() if fp8_out else 0),
        Int64(fp8_scale.data_ptr() if fp8_out else 0),
        Int32(m),
        Float32(rms_eps),
        Float32(out_eps),
        cuda.CUstream(torch.cuda.current_stream(output.device).cuda_stream),
    )


@torch.library.custom_op("trtllm::k3_attn_res_rmsnorm", mutates_args=())
def k3_attn_res_rmsnorm(
    layer: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> torch.Tensor:
    """Attention-residual mix of ``block`` and ``layer`` followed by RMSNorm.

    Bit-identical to ``trtllm::attn_res_rmsnorm_fwd`` (``layer [M, 7168]``, ``block [N - 1, M,
    7168]``).
    """
    output = torch.empty_like(layer)
    run_k3_attn_res(
        layer, (), block, res_weight, rms_weight, out_weight, rms_eps, out_eps, output, None
    )
    return output


@k3_attn_res_rmsnorm.register_fake
def _(layer, block, res_weight, rms_weight, out_weight, rms_eps, out_eps):
    return torch.empty_like(layer)


@torch.library.custom_op("trtllm::k3_attn_res_add_rmsnorm", mutates_args=())
def k3_attn_res_add_rmsnorm(
    prefix: torch.Tensor,
    addend: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(bf16(prefix + addend), output)``: the residual add, then the mix and RMSNorm.

    Bit-identical to ``trtllm::attn_res_add_rmsnorm_fwd``.
    """
    output = torch.empty_like(prefix)
    updated = torch.empty_like(prefix)
    run_k3_attn_res(
        prefix,
        (addend,),
        block,
        res_weight,
        rms_weight,
        out_weight,
        rms_eps,
        out_eps,
        output,
        updated,
    )
    return updated, output


@k3_attn_res_add_rmsnorm.register_fake
def _(prefix, addend, block, res_weight, rms_weight, out_weight, rms_eps, out_eps):
    return torch.empty_like(prefix), torch.empty_like(prefix)


@torch.library.custom_op("trtllm::k3_attn_res_moe_add_rmsnorm", mutates_args=())
def k3_attn_res_moe_add_rmsnorm(
    residual: torch.Tensor,
    routed: torch.Tensor,
    shared: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(prefix, output)`` with ``prefix = bf16(residual + bf16(routed + shared))``.

    The MoE residual update of the previous layer (bit-identical to
    ``kimi_k3_moe_residual._fused_moe_residual_add``) followed by the mix and RMSNorm of
    :func:`k3_attn_res_rmsnorm` on ``prefix``.
    """
    output = torch.empty_like(residual)
    prefix = torch.empty_like(residual)
    run_k3_attn_res(
        residual,
        (routed, shared),
        block,
        res_weight,
        rms_weight,
        out_weight,
        rms_eps,
        out_eps,
        output,
        prefix,
    )
    return prefix, output


@k3_attn_res_moe_add_rmsnorm.register_fake
def _(residual, routed, shared, block, res_weight, rms_weight, out_weight, rms_eps, out_eps):
    return torch.empty_like(residual), torch.empty_like(residual)


# ---------------------------------------------------------------------------
# FP8-output variants: the normalized mixture is also returned as the FP8 1x128 activation of the
# next block-scale GEMM (codes, R128c4 UE8M0 scales), so that GEMM's separate quantize kernel goes
# away. The codes and scales are bit-identical to quantizing the BF16 output with the production
# quantizers (see run_k3_attn_res); the BF16 outputs are those of the non-FP8 ops.
# ---------------------------------------------------------------------------


def _fp8_buffers(layer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = layer.shape[0]
    fp8 = torch.empty((m, HIDDEN_SIZE), dtype=torch.float8_e4m3fn, device=layer.device)
    scale = torch.empty((fp8_scale_numel(m),), dtype=torch.uint8, device=layer.device)
    return fp8, scale


@torch.library.custom_op("trtllm::k3_attn_res_rmsnorm_fp8", mutates_args=())
def k3_attn_res_rmsnorm_fp8(
    layer: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`k3_attn_res_rmsnorm` plus ``(fp8 [M, 7168], scale uint8)`` of its output."""
    output = torch.empty_like(layer)
    fp8, scale = _fp8_buffers(layer)
    run_k3_attn_res(
        layer,
        (),
        block,
        res_weight,
        rms_weight,
        out_weight,
        rms_eps,
        out_eps,
        output,
        None,
        fp8_output=fp8,
        fp8_scale=scale,
    )
    return output, fp8, scale


@k3_attn_res_rmsnorm_fp8.register_fake
def _(layer, block, res_weight, rms_weight, out_weight, rms_eps, out_eps):
    return (torch.empty_like(layer), *_fp8_buffers(layer))


@torch.library.custom_op("trtllm::k3_attn_res_add_rmsnorm_fp8", mutates_args=())
def k3_attn_res_add_rmsnorm_fp8(
    prefix: torch.Tensor,
    addend: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`k3_attn_res_add_rmsnorm` plus ``(fp8, scale)`` of its normalized output."""
    output = torch.empty_like(prefix)
    updated = torch.empty_like(prefix)
    fp8, scale = _fp8_buffers(prefix)
    run_k3_attn_res(
        prefix,
        (addend,),
        block,
        res_weight,
        rms_weight,
        out_weight,
        rms_eps,
        out_eps,
        output,
        updated,
        fp8_output=fp8,
        fp8_scale=scale,
    )
    return updated, output, fp8, scale


@k3_attn_res_add_rmsnorm_fp8.register_fake
def _(prefix, addend, block, res_weight, rms_weight, out_weight, rms_eps, out_eps):
    return (torch.empty_like(prefix), torch.empty_like(prefix), *_fp8_buffers(prefix))


@torch.library.custom_op("trtllm::k3_attn_res_moe_add_rmsnorm_fp8", mutates_args=())
def k3_attn_res_moe_add_rmsnorm_fp8(
    residual: torch.Tensor,
    routed: torch.Tensor,
    shared: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`k3_attn_res_moe_add_rmsnorm` plus ``(fp8, scale)`` of its normalized output."""
    output = torch.empty_like(residual)
    prefix = torch.empty_like(residual)
    fp8, scale = _fp8_buffers(residual)
    run_k3_attn_res(
        residual,
        (routed, shared),
        block,
        res_weight,
        rms_weight,
        out_weight,
        rms_eps,
        out_eps,
        output,
        prefix,
        fp8_output=fp8,
        fp8_scale=scale,
    )
    return prefix, output, fp8, scale


@k3_attn_res_moe_add_rmsnorm_fp8.register_fake
def _(residual, routed, shared, block, res_weight, rms_weight, out_weight, rms_eps, out_eps):
    return (torch.empty_like(residual), torch.empty_like(residual), *_fp8_buffers(residual))


__all__ = [
    "ADD_MOE",
    "ADD_NONE",
    "ADD_PREFIX",
    "HIDDEN_SIZE",
    "MAX_CANDIDATES",
    "MAX_TOKENS",
    "MIN_CANDIDATES",
    "K3AttnResKernel",
    "can_use_k3_attn_res",
    "cluster_size",
    "fp8_scale_numel",
    "k3_attn_res_add_rmsnorm",
    "k3_attn_res_add_rmsnorm_fp8",
    "k3_attn_res_moe_add_rmsnorm",
    "k3_attn_res_moe_add_rmsnorm_fp8",
    "k3_attn_res_rmsnorm",
    "k3_attn_res_rmsnorm_fp8",
    "run_k3_attn_res",
]
