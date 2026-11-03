# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decode attention-residual mix + RMSNorm (+ residual add) with a PDL prologue.

Runs on SM100, SM103 and SM107: the kernels use no tensor-core instructions, at most five named
barriers per CTA and portable cluster sizes.

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

The fast kernel can take the addend from an MNNVL Lamport workspace (``lamport_addend``): the
result of the two-shot all-reduce that completed last, or (``lamport_finish``) the one-shot
all-reduce whose input every rank pushed, which the kernel then finishes itself (rank-order sum
of the slots, rotation state, re-arm of the previous buffer; :mod:`k3_lamport_ar`). With
``early_poll`` it finishes that all-reduce and computes the mix before its ``griddepcontrol.wait``
and only stores after it (see :func:`run_k3_attn_res`).
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

from . import k3_lamport_ar as _lam

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
#: Early trigger row limit (0: never; the Kimi K3.1 TEP16 decode servers set 8 at model build): calls
#: of at most that many rows release the dependent grid right after ``griddepcontrol.wait`` instead
#: of after the last store, so the next kernel (a decode projection) can become resident and stream its weights while
#: this one finishes.
#:
#: Why it is safe in the Kimi-K3 decode graph. Since the trigger follows this kernel's own wait,
#: every grid upstream of it has completed when a dependent starts; only this grid may still run.
#: The grids that can start before it completes are its programmatic dependents (under stream
#: capture that includes cross-stream event joins) and, through dependents that trigger before their
#: own wait, theirs: the decode projections (qkvg, q_a / kv_a, latent down, shared gate-up), the
#: router GEMM, ``k3_kda_fab`` (warp 0) and behind them the KDA MTP decode and the MoE routing /
#: expert kernels. Before their wait these read only weights, weight scales, routing metadata and
#: (KDA decode) the layer's state pool, conv windows and replay rows, and write no global memory;
#: this kernel's outputs (normed row, updated prefix) are read only after a wait. The next attn_res
#: launch, which reads the prefix this kernel stores (and snapshot rows) before its wait, is
#: separated from this one on every path by an MNNVL all-reduce (the o_proj all-reduce between A_pre
#: and A_mid; the latent and shared-expert all-reduces of the MoE block between A_mid and the next
#: layer's A_pre). Those kernels call ``cudaGridDependencySynchronize`` before they trigger, so the
#: next attn_res starts only after this grid completed.
#:
#: The row limit: with more rows the grid has more CTAs, and a dependent GEMM resident beside it
#: early takes SM slots its CTAs still need. The release point does not change any result.
_EARLY_TRIGGER_MAX_M = 0

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
def _selp_b32(cond, a, b, *, loc=None, ip=None):
    """32-bit ``a if cond != 0 else b``."""
    return Int32(
        _asm_se(
            T.i32(),
            [_iv(cond, loc=loc, ip=ip), _iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _poll_mailbox_v4(valid, addr, *, loc=None, ip=None):
    """16 bytes of a fused latent-MoE tail mailbox (``modules/kimi_k3_moe_tail``: the local copy
    of the up-projection's multicast all-gather, every empty BF16 lane holds ``-0``), polled until
    no lane is empty, then re-armed to ``-0`` in every lane; zeros (no access) where ``valid`` is 0.

    The producer stores ``+0`` for a ``-0`` result, so a filled lane never reads as empty; each
    16-byte vector has one writer (the rank that owns its columns)."""
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred pv, pe;\n.reg .b32 t, s;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        "setp.ne.u32 pv, $4, 0;\n@!pv bra MBX_DONE;\n"
        "MBX_POLL:\n"
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$5];\n"
        "and.b32 t, $0, 0xffff;\nsetp.eq.b32 pe, t, 0x8000;\n"
        "and.b32 t, $0, 0xffff0000;\nsetp.eq.or.b32 pe, t, 0x80000000, pe;\n"
        "and.b32 t, $1, 0xffff;\nsetp.eq.or.b32 pe, t, 0x8000, pe;\n"
        "and.b32 t, $1, 0xffff0000;\nsetp.eq.or.b32 pe, t, 0x80000000, pe;\n"
        "and.b32 t, $2, 0xffff;\nsetp.eq.or.b32 pe, t, 0x8000, pe;\n"
        "and.b32 t, $2, 0xffff0000;\nsetp.eq.or.b32 pe, t, 0x80000000, pe;\n"
        "and.b32 t, $3, 0xffff;\nsetp.eq.or.b32 pe, t, 0x8000, pe;\n"
        "and.b32 t, $3, 0xffff0000;\nsetp.eq.or.b32 pe, t, 0x80000000, pe;\n"
        "@pe bra MBX_POLL;\n"
        "mov.b32 s, 0x80008000;\nst.global.v4.b32 [$5], {s, s, s, s};\n"
        "MBX_DONE:\n}",
        "=&r,=&r,=&r,=&r,r,l",
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


def _lamport_ops():
    """``k3_lamport_ar`` (the pushed MNNVL all-reduce), imported by the Lamport-addend variant."""
    from . import k3_lamport_ar

    return k3_lamport_ar


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
        early_trigger: bool = False,
    ):
        if not MIN_CANDIDATES <= num_candidates <= MAX_CANDIDATES:
            raise ValueError(f"num_candidates={num_candidates} not in [2, 12]")
        if add_mode not in (ADD_NONE, ADD_PREFIX, ADD_MOE):
            raise ValueError(f"add_mode={add_mode}")
        self.n = num_candidates
        self.k = num_candidates - 1
        # Release the dependent grid right after the wait (early trigger, _EARLY_TRIGGER_MAX_M).
        self.early_trigger = early_trigger and use_pdl
        # ``cluster`` overrides the topology for experiments; results then follow the C++
        # kernel of that topology rather than of ``num_candidates``.
        self.c = cluster if cluster is not None else cluster_size(num_candidates)
        if self.c not in (1, 2, 4, 8):
            raise ValueError(f"cluster={self.c}")
        self.add_mode = add_mode
        self.use_pdl = use_pdl
        self.slice = HIDDEN_SIZE // self.c
        self.items = (self.slice + _THREADS - 1) // _THREADS
        # Only the last item can be partial (c = 8: 896 = 3.5 x 256).
        self.partial_last = self.slice % _THREADS != 0

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
        lam_mc: Int64,
        lam_rank: Int32,
        m: Int32,
        eps: Float32,
        out_eps: Float32,
        stream: cuda.CUstream,
    ):
        c, k = self.c, self.k

        @cute.struct
        class SharedStorage:
            # [c][warp][k] (sq, dot) snapshot partials, [c][warp] for the layer row, [c][warp]
            # output sums of squares; under c = 8 every CTA of the cluster fills its own slot g.
            snap: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * k * 2], 16]
            last: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS * 2], 16]
            outsq: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, c * _WARPS], 16]
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
        else:
            kernel.launch(
                grid=(m, 1, 1),
                block=(_THREADS, 1, 1),
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
        m: Int32,
        eps: Float32,
        out_eps: Float32,
    ):
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

        token = bidx // c
        rank = bidx - token * c

        if cutlass.const_expr(c > 1):
            if tidx == 0:
                for r in cutlass.range_constexpr(3):
                    _mbar_init(mbar_smem + r * 8, 1)
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

        def st_item(base, j, value):
            if cutlass.const_expr(self.partial_last and j == items - 1):
                _st_bf16_if(valid_last, base + h_off[j], value)
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

        # ---- wait for the previous grid ----
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
            if cutlass.const_expr(self.early_trigger):
                _griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.add_mode == ADD_PREFIX):
            # bf16(prefix + addend): the C++ kernels' fused add (fast-math add.ftz). All loads
            # are issued before the first store (a store may alias a later load, so they are
            # not reordered).
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

        for j in cutlass.range_constexpr(items):
            y = _round_bf16(_mul_ftz(mixed[j], out_rs))
            st_item(out_addr + row, j, _mul_ftz(y, out_w[j]))

        if cutlass.const_expr(self.use_pdl and not self.early_trigger):
            _bar_sync(3)
            _griddepcontrol_launch_dependents()


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
        lamport: bool = False,
        finish_world: int = 0,
        finish_chunk: int = 0,
        add_rn: bool = False,
        finish_two_shot: bool = False,
        early_poll: bool = False,
        mailbox: bool = False,
        early_trigger: bool = False,
    ):
        if not MIN_CANDIDATES <= num_candidates <= MAX_CANDIDATES:
            raise ValueError(f"num_candidates={num_candidates} not in [2, 12]")
        if mailbox and (add_mode != ADD_PREFIX or lamport or not add_rn):
            raise ValueError("the mailbox addend is an ADD_PREFIX variant with add_rn, no Lamport")
        # The addend is the fused latent-MoE tail's output in its symmetric mailbox: polled
        # until every BF16 lane has arrived, then re-armed (see _poll_mailbox_v4).
        self.mailbox = mailbox
        if early_poll and not finish_world:
            raise ValueError("early_poll finishes a pushed all-reduce: it needs finish_world")
        if add_mode not in (ADD_NONE, ADD_PREFIX, ADD_MOE) or cluster not in (1, 2, 4, 8):
            raise ValueError(f"add_mode={add_mode} cluster={cluster}")
        if lamport and add_mode != ADD_PREFIX:
            raise ValueError("the Lamport addend is an ADD_PREFIX variant")
        if finish_world and (not lamport or finish_world < 2 or finish_chunk < 1):
            raise ValueError(
                f"the finish needs the Lamport addend, finish_world={finish_world} >= 2 "
                f"and finish_chunk={finish_chunk} >= 1"
            )
        if finish_two_shot and not finish_world:
            raise ValueError("finish_two_shot needs finish_world")
        if add_rn and add_mode != ADD_PREFIX:
            raise ValueError("add_rn selects the rounding of the ADD_PREFIX add")
        self.lamport = lamport
        # Finish: the addend is the pushed all-reduce over ``finish_world`` ranks, which this
        # kernel completes with the one-shot or (``finish_two_shot``) the two-shot transport (0:
        # the result of a finished two-shot all-reduce, polled from its broadcast stage).
        self.finish_world = finish_world
        self.finish_chunk = finish_chunk
        self.finish_two_shot = finish_two_shot
        # bf16(prefix + addend) with a round-to-nearest add (the MoE residual update's
        # arithmetic) instead of the C++ kernels' fast-math add.
        self.add_rn = add_rn
        # Finish the pushed all-reduce before ``griddepcontrol.wait`` (see ``early_poll`` of
        # run_k3_attn_res); without PDL the grid starts after its predecessor anyway.
        self.early_poll = early_poll and use_pdl
        # Release the dependent grid right after the wait (early trigger, _EARLY_TRIGGER_MAX_M).
        self.early_trigger = early_trigger and use_pdl
        self.n = num_candidates
        self.k = num_candidates - 1
        self.c = cluster
        self.add_mode = add_mode
        self.use_pdl = use_pdl
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
        lam_mc: Int64,
        lam_rank: Int32,
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
            lam_mc,
            lam_rank,
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
        lam_mc: Int64,
        lam_rank: Int32,
        m: Int32,
        eps: Float32,
        out_eps: Float32,
    ):
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
        if cutlass.const_expr(self.finish_two_shot):
            # The rows this rank owns in the two-shot call go to the first CTA groups.
            token = _lam.two_shot_token(token, m, lam_rank, self.finish_world)
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

        # ---- wait for the previous grid (early_poll: after the mix, before the first store) ----
        if cutlass.const_expr(self.use_pdl and not self.early_poll):
            _griddepcontrol_wait()
            if cutlass.const_expr(self.early_trigger):
                _griddepcontrol_launch_dependents()

        upd_words = []
        arr_old = Int32(0)  # early_poll: thread 0's arrival count (arrive_count / arrive_rotate)
        if cutlass.const_expr(self.add_mode == ADD_PREFIX):
            if cutlass.const_expr(self.lamport and self.finish_world > 0):
                # The addend is the all-reduce whose input every rank pushed into the next call on a
                # Lamport workspace (add_addr: its local unicast base, add2_addr: its state flags,
                # lam_mc: its multicast base, lam_rank: this rank), and this kernel is the call's
                # finish: it reads the rotation state (final: the producer completed before the
                # wait above), counts its CTA (the last one advances the state) and re-arms the
                # previous call's buffer. One-shot transport: it sums the token's slot of every
                # rank in rank order like k3_lamport_ar.run_lamport_finish. Two-shot transport: on
                # the token's owner it sums the scatter slots and multicasts the row, elsewhere it
                # polls the broadcast row. Threads past the row work on the row's first vector
                # (and do not store) and use zeros.
                # early_poll: all of it before the wait (the producer released this grid after
                # its own wait, so the state is final and the dirty buffer's readers are done),
                # except the state advance, which follows the wait (below).
                W = self.finish_world
                cur, dirty, nbytes, dstages, c0, c1, c2, c3 = _lam.lamport_state(add2_addr)
                _bar_sync(1, self.threads)
                if cutlass.const_expr(not self.early_poll):
                    if tidx == 0:
                        _lam.arrive_finish(
                            add2_addr,
                            m * c,
                            cur,
                            nbytes,
                            m,
                            W,
                            HIDDEN_SIZE,
                            not self.finish_two_shot,
                            0,
                        )
                _lam.clear_dirty_stages(
                    add_addr,
                    dirty,
                    nbytes,
                    dstages,
                    c0,
                    c1,
                    c2,
                    c3,
                    Int32(bidx * self.threads + tidx),
                    m * (c * self.threads),
                )
                off = _selp_b32(valid, Int32(rank * self.vecs + tidx) * 16, Int32(0))
                # early_poll: every thread polls its slots from the start, while the producer
                # still runs.
                if cutlass.const_expr(self.finish_two_shot):
                    words = _lam.finish_two_shot_v4(
                        add_addr,
                        lam_mc,
                        cur,
                        nbytes,
                        token,
                        off,
                        lam_rank,
                        valid,
                        W,
                        self.finish_chunk,
                        HIDDEN_SIZE,
                    )
                    addend = unpack([_selp_b32(valid, w, Int32(0)) for w in words])
                else:
                    acc = _lam.poll_reduce_v4(
                        _lam.stage_addr(add_addr, cur, Int32(0), 1, nbytes)
                        + Int64(token * W) * (HIDDEN_SIZE * 2)
                        + Int64(off),
                        HIDDEN_SIZE * 2,
                        W,
                        self.finish_chunk,
                    )
                    addend = [_selp(valid, _round_bf16(a), _zero_f32()) for a in acc]
                if cutlass.const_expr(self.early_poll):
                    # Every thread of this CTA has seen its vectors of every rank, so every
                    # producer CTA that pushed them has read the rotation state (the push address
                    # depends on it); once every CTA of this grid has counted itself, the last
                    # one may advance it. The count is a relaxed atomic issued here whose result
                    # thread 0 reads only after its stores (arrive_rotate below): its round trip
                    # overlaps the mix instead of delaying warp 0 into the cluster rounds.
                    _bar_sync(1, self.threads)
                    if tidx == 0:
                        arr_old = _lam.arrive_count(add2_addr)
            elif cutlass.const_expr(self.mailbox):
                # The addend is the fused MoE tail's [M, 7168] output in the local copy of its
                # multicast mailbox (add_addr), written by every rank's up-projection shard:
                # polled per vector, then re-armed for the next tail call. Threads past the row
                # neither poll nor store (their vector is another thread's).
                addend = unpack(_poll_mailbox_v4(valid, add_addr + row + col))
            elif cutlass.const_expr(self.lamport):
                # The addend is the result of the MNNVL all-reduce that completed last on a Lamport
                # workspace (add_addr: its local unicast base, add2_addr: its state flags), finished
                # without the gather: polled from the broadcast stage of the call's buffer (see
                # k3_lamport_ar.poll_result_v4). Threads past the row poll the row's first vector
                # and use zeros.
                off = _selp_b32(valid, Int32(rank * self.vecs + tidx) * 16, Int32(0))
                polled = _lamport_ops().poll_result_v4(
                    add_addr, add2_addr, Int32(token), off, HIDDEN_SIZE
                )
                addend = unpack([_selp_b32(valid, w, Int32(0)) for w in polled])
            else:
                addend = unpack(load(add_addr + row + col, False))
            pre = unpack(prefix)
            if cutlass.const_expr(self.add_rn):
                x = [_round_bf16(_add(pre[e], addend[e])) for e in range(8)]
            else:
                x = [_round_bf16(_add_ftz(pre[e], addend[e])) for e in range(8)]
        elif cutlass.const_expr(self.add_mode == ADD_MOE):
            routed = unpack(load(add_addr + row + col, False))
            shared = unpack(load(add2_addr + row + col, False))
            pre = unpack(prefix)
            x = [
                _round_bf16(_add(pre[e], _round_bf16(_add(routed[e], shared[e])))) for e in range(8)
            ]
        else:
            x = unpack(load(layer_addr + row + col, False))
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
        if cutlass.const_expr(self.early_poll):
            # Every store of this grid follows the wait: its output buffers may reuse memory that
            # the producer, or a grid between it and this one, still reads. The dependents are
            # released after the stores (below).
            _griddepcontrol_wait()
            _stg_v4_if(valid, upd_addr + row + col, upd_words)
        _stg_v4_if(valid, out_addr + row + col, words)
        if cutlass.const_expr(self.early_poll):
            # The last CTA to count advances the workspace (its count was issued after the poll).
            if tidx == 0:
                _lam.arrive_rotate(
                    add2_addr,
                    arr_old,
                    m * c,
                    cur,
                    nbytes,
                    m,
                    self.finish_world,
                    HIDDEN_SIZE,
                    not self.finish_two_shot,
                    0,
                )

        # Release at the end unless released after the wait (early_trigger, not with early_poll).
        if cutlass.const_expr(self.use_pdl and (self.early_poll or not self.early_trigger)):
            _bar_sync(4, self.threads)
            _griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


#: The bit-identical per-element kernel instead of the vectorized one.
_EXACT = os.environ.get("TRTLLM_K3_ATTN_RES_PDL_EXACT", "0") == "1"


def fast_cluster_size(num_tokens: int, num_candidates: int) -> int:
    """CTAs per token of :class:`K3AttnResFastKernel`.

    Measured on GB300 behind a PDL producer: eight CTAs per token are fastest for a handful of
    tokens with at least three snapshots (smaller per-CTA prologue), four from ~32 tokens on
    (eight-CTA clusters then fill the machine) and for two or three candidates. On SM107 (212
    SMs) this choice is within 0.25 us of the fastest cluster size for up to 64 tokens.
    """
    return 8 if num_tokens <= 16 and num_candidates >= 4 else 4


#: Where :func:`fast_cluster_size` picks four CTAs per token, run :func:`tuned_cluster_size` instead:
#: TRTLLM_K3_ATTN_RES_TUNED_CLUSTER=1 for every model, =0 nowhere; unset, only in processes whose
#: model enabled it (Kimi K3.1, :func:`enable_tuned_cluster`), so Kimi K3 keeps four by default.
_TUNED_CLUSTER_ENV = os.environ.get("TRTLLM_K3_ATTN_RES_TUNED_CLUSTER")
if _TUNED_CLUSTER_ENV not in (None, "0", "1"):
    raise ValueError(f"TRTLLM_K3_ATTN_RES_TUNED_CLUSTER must be 0 or 1, got {_TUNED_CLUSTER_ENV!r}")
_TUNED_CLUSTER = _TUNED_CLUSTER_ENV == "1"


def enable_tuned_cluster() -> None:
    """Make :func:`tuned_cluster_size` the default of this process (called when a Kimi K3.1 model
    is built); an explicit TRTLLM_K3_ATTN_RES_TUNED_CLUSTER keeps its value."""
    global _TUNED_CLUSTER
    if _TUNED_CLUSTER_ENV is None:
        _TUNED_CLUSTER = True


@functools.cache
def _tuned_cluster_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3))


def tuned_cluster_size(num_tokens: int, num_candidates: int, add: bool) -> int:
    """CTAs per token on SM100 / SM103 where :func:`fast_cluster_size` picks four: one or two CTAs
    give the same bits as four (each warp sums the same 32 vectors and the 28 warp partials are
    added in the same order; only eight CTAs, 4-warp CTAs with a partial last warp, differ) and
    are faster at these shapes (GB300, behind a BF16 GEMM, us per call, M=112: N=2 3.53 vs 4.25,
    N=4 4.38 vs 4.94, N=8 5.72 vs 7.26; M=56 N=3 2.98 vs 3.89). With more tokens the per-token
    clusters of four no longer fit one wave once the snapshots fill the registers (from about 84
    tokens at seven candidates, six with the residual add), where one CTA per token stays at one
    wave. ``add``: the layer row is ``bf16(prefix + addend)``."""
    if num_tokens <= 21:
        return 4
    if num_tokens <= 28:
        return 1 if num_candidates <= 6 else 4
    if num_tokens < 84:
        return 1 if num_candidates <= 5 else 2 if num_candidates == 7 and not add else 4
    if num_candidates <= 4 or num_candidates >= 7:
        return 1
    return 1 if num_candidates == 6 and add else 4


@functools.cache
def _compiled(
    num_candidates: int,
    add_mode: int,
    use_pdl: bool,
    cluster: int | None,
    exact: bool,
    lamport: bool = False,
    finish_world: int = 0,
    finish_chunk: int = 0,
    add_rn: bool = False,
    finish_two_shot: bool = False,
    early_poll: bool = False,
    mailbox: bool = False,
    early_trigger: bool = False,
):
    if exact:
        kernel = K3AttnResKernel(num_candidates, add_mode, use_pdl, cluster, early_trigger)
    else:
        kernel = K3AttnResFastKernel(
            num_candidates,
            add_mode,
            use_pdl,
            cluster,
            lamport,
            finish_world,
            finish_chunk,
            add_rn,
            finish_two_shot,
            early_poll,
            mailbox,
            early_trigger,
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
        Int32(0),
        Int32(1),
        Float32(1e-6),
        Float32(1e-6),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def can_use_k3_attn_res(
    layer: torch.Tensor,
    addends: tuple[torch.Tensor, ...],
    block: torch.Tensor,
    *weights: torch.Tensor,
) -> bool:
    """Whether :func:`run_k3_attn_res` accepts ``layer [M, H]``, 0-2 addends ``[M, H]`` and
    ``block [N - 1, M, H]``."""
    if not layer.is_cuda or not _is_supported_device(layer.device.index or 0):
        return False
    if layer.dim() != 2 or layer.shape[1] != HIDDEN_SIZE or not 1 <= layer.shape[0] <= MAX_TOKENS:
        return False
    m = layer.shape[0]
    if block.dim() != 3 or tuple(block.shape[1:]) != (m, HIDDEN_SIZE):
        return False
    if not MIN_CANDIDATES <= block.shape[0] + 1 <= MAX_CANDIDATES or len(addends) > 2:
        return False
    for t in (layer, block, *weights, *addends):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.device != layer.device:
            return False
    if any(a.shape != layer.shape for a in addends):
        return False
    return all(w.numel() == HIDDEN_SIZE for w in weights)


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
    lamport_addend: tuple[int, int] | None = None,
    lamport_finish: tuple[int, int, bool, int, int] | None = None,
    add_rn: bool = False,
    early_poll: bool = False,
    mailbox_addend: int | None = None,
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
        lamport_addend: ``(uc_local, flags)`` device addresses of an MNNVL Lamport workspace
            (``k3_lamport_ar.LamportWorkspace``) instead of ``addends`` (which must be ``()``):
            ``bf16(layer + addend)`` with the ``[M, 7168]`` result of the all-reduce that
            completed last on the workspace, finished without its gather
            (``run_lamport_finish_in_place``), polled from its broadcast stage after
            ``griddepcontrol.wait``; the same bits as the gathered tensor, so the same results
            (vectorized kernel only). The next all-reduce on the workspace re-arms that buffer:
            it must not start before this kernel completed (its producer must trigger only
            after its own ``griddepcontrol.wait``, as the push producers and the C++ kernels do).
        lamport_finish: ``(world, chunk, two_shot, mc_base, rank)`` with ``lamport_addend``: the
            addend is the all-reduce whose input every rank pushed into the next call on the
            workspace (``mc_base``: its multicast base, ``rank``: this rank of ``world``) with
            the one-shot or (``two_shot``) the two-shot transport, and this kernel finishes that
            call as ``k3_lamport_ar.run_lamport_finish`` does (rank-order sums, ``chunk`` ranks per
            partial sum; two-shot: the owner of each row reduces and multicasts it, the others
            poll it), advances the rotation state and re-arms the previous call's buffer, so that
            no finish kernel runs in front of it. The addend has the bits of the finished
            all-reduce. The producer must trigger only after its own ``griddepcontrol.wait``; with
            the two-shot transport every rank's CTAs must become co-resident (the grid is small:
            ``M * cluster`` CTAs, the owned rows first).
        add_rn: ``bf16(layer + addend)`` with a round-to-nearest FP32 add (the MoE residual
            update's arithmetic) instead of the C++ kernels' fast-math add (vectorized kernel,
            one addend).
        early_poll: (with ``lamport_finish``, PDL) finish the pushed all-reduce before
            ``griddepcontrol.wait``: as soon as it is resident the kernel reads the rotation
            state, re-arms the previous buffer, polls its slots, counts its CTA (a relaxed
            atomic: every producer CTA that pushed those slots has read the state) and computes
            the mix and the norm; then it waits for the previous grid, stores its outputs (which
            may reuse memory that grids still running read) and the last CTA advances the state;
            it releases its dependents after its stores.
            This is the post-projection half of a collective "sandwich" as two PDL grids: the
            producer (the push GEMM) releases this grid after its own wait, every rank's slots
            are polled while the producer runs, and the kernel boundary leaves the critical path.
            Requires that this grid is launched (programmatically or not) after the producer
            and that the producer triggers only after its own ``griddepcontrol.wait`` (the push
            variants do), so every earlier call on the workspace has completed at launch. Same
            bits as without it.
        mailbox_addend: device address of the fused latent-MoE tail's mailbox rows (the local
            copy of its multicast all-gather, ``bf16 [M, 7168]``, ``-0`` marks an empty lane)
            instead of ``addends`` (which must be ``()``): ``bf16(layer + addend)`` with a
            round-to-nearest add (``add_rn`` is implied: the BF16 add of the tail output to the
            residual), each vector polled after ``griddepcontrol.wait`` and re-armed once read,
            as the tail's copy kernel does (vectorized kernel only). The next tail call on the
            mailbox must not start before this kernel completed (its producers wait before they
            trigger).
    """
    if exact is None:
        exact = _EXACT
    mailbox = mailbox_addend is not None
    if mailbox:
        if addends or lamport_addend is not None or exact:
            raise ValueError("the mailbox addend replaces addends; vectorized kernel only")
        add_rn = True
    if early_poll and lamport_finish is None:
        raise ValueError("early_poll needs lamport_finish")
    if (lamport_addend is not None or add_rn) and exact:
        raise ValueError("the Lamport addend and add_rn need the vectorized kernel")
    lamport = lamport_addend is not None
    if lamport and addends:
        raise ValueError("pass either addends or lamport_addend")
    if lamport_finish is not None and not lamport:
        raise ValueError("lamport_finish needs lamport_addend")
    add_mode = ADD_PREFIX if lamport or mailbox else len(addends)
    if add_rn and add_mode != ADD_PREFIX:
        raise ValueError("add_rn needs exactly one addend")
    if cluster is None and not exact:
        cluster = fast_cluster_size(layer.shape[0], block.shape[0] + 1)
        if (
            cluster == 4
            and _TUNED_CLUSTER
            and not lamport
            and _tuned_cluster_device(layer.device.index or 0)
        ):
            # Bit-identical to four CTAs per token (see tuned_cluster_size).
            cluster = tuned_cluster_size(layer.shape[0], block.shape[0] + 1, add_mode != ADD_NONE)
    finish_world, finish_chunk, finish_two_shot, lam_mc, lam_rank = (
        lamport_finish if lamport_finish is not None else (0, 0, False, 0, 0)
    )
    compiled = _compiled(
        block.shape[0] + 1,
        add_mode,
        use_pdl,
        cluster,
        exact,
        lamport,
        finish_world,
        finish_chunk,
        add_rn,
        finish_two_shot,
        early_poll and use_pdl,
        mailbox,
        use_pdl and layer.shape[0] <= _EARLY_TRIGGER_MAX_M,
    )
    if lamport:
        add_addr, add2_addr = lamport_addend
    elif mailbox:
        add_addr, add2_addr = mailbox_addend, 0
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
        Int64(lam_mc),
        Int32(lam_rank),
        Int32(layer.shape[0]),
        Float32(rms_eps),
        Float32(out_eps),
        cuda.CUstream(torch.cuda.current_stream(layer.device).cuda_stream),
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


@torch.library.custom_op("trtllm::k3_attn_res_mailbox_add_rmsnorm", mutates_args=("mailbox",))
def k3_attn_res_mailbox_add_rmsnorm(
    prefix: torch.Tensor,
    mailbox: torch.Tensor,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(bf16(prefix + addend), output)`` with the addend the fused latent-MoE tail's output in
    the first ``M`` rows of its mailbox (``bf16 [..., 7168]``, contiguous), read as it arrives
    and re-armed (see ``mailbox_addend`` of :func:`run_k3_attn_res`).

    Bit-identical to the tail's copy kernel, then ``prefix + addend`` (a BF16 add), then
    ``trtllm::k3_attn_res_rmsnorm`` on the sum.
    """
    if (
        mailbox.dtype != torch.bfloat16
        or not mailbox.is_contiguous()
        or mailbox.shape[-1] != prefix.shape[1]
        or mailbox.numel() < prefix.numel()
    ):
        raise ValueError("mailbox must be a contiguous BF16 tensor of at least M rows of 7168")
    output = torch.empty_like(prefix)
    updated = torch.empty_like(prefix)
    run_k3_attn_res(
        prefix,
        (),
        block,
        res_weight,
        rms_weight,
        out_weight,
        rms_eps,
        out_eps,
        output,
        updated,
        mailbox_addend=mailbox.data_ptr(),
    )
    return updated, output


@k3_attn_res_mailbox_add_rmsnorm.register_fake
def _(prefix, mailbox, block, res_weight, rms_weight, out_weight, rms_eps, out_eps):
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


@torch.library.custom_op("trtllm::k3_attn_res_add_rmsnorm_lamport", mutates_args=())
def k3_attn_res_add_rmsnorm_lamport(
    prefix: torch.Tensor,
    uc_local: int,
    flags: int,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`k3_attn_res_add_rmsnorm` whose addend is the result of the all-reduce that
    completed last on the Lamport workspace ``(uc_local, flags)``, read from its broadcast stage
    (see ``lamport_addend`` of :func:`run_k3_attn_res`); vectorized kernel."""
    updated = torch.empty_like(prefix)
    output = torch.empty_like(prefix)
    run_k3_attn_res(
        prefix,
        (),
        block,
        res_weight,
        rms_weight,
        out_weight,
        rms_eps,
        out_eps,
        output,
        updated,
        exact=False,
        lamport_addend=(uc_local, flags),
    )
    return updated, output


@k3_attn_res_add_rmsnorm_lamport.register_fake
def _(prefix, uc_local, flags, block, res_weight, rms_weight, out_weight, rms_eps, out_eps):
    return torch.empty_like(prefix), torch.empty_like(prefix)


def _run_lamport_add(
    prefix,
    uc_local,
    flags,
    block,
    weights,
    rms_eps,
    out_eps,
    finish,
    add_rn,
    early_poll=False,
):
    updated = torch.empty_like(prefix)
    output = torch.empty_like(prefix)
    run_k3_attn_res(
        prefix,
        (),
        block,
        *weights,
        rms_eps,
        out_eps,
        output,
        updated,
        exact=False,
        lamport_addend=(uc_local, flags),
        lamport_finish=finish if finish[0] else None,
        add_rn=add_rn,
        early_poll=early_poll and bool(finish[0]),
    )
    return updated, output


_LAMPORT_ADD_DOC = """``(bf16(prefix + addend), output)``: the mix and RMSNorm of
    :func:`k3_attn_res_add_rmsnorm` with the addend read from the Lamport workspace ``(uc_local,
    flags)``. ``finish_world == 0``: the result of the two-shot all-reduce that completed last on
    the workspace (as :func:`k3_attn_res_add_rmsnorm_lamport`); else the pushed all-reduce over
    ``finish_world`` ranks that this kernel finishes, ``finish_chunk`` ranks per partial sum, with
    the one-shot or (``finish_two_shot``) the two-shot transport (``mc_base``: the workspace's
    multicast base, ``rank``: this rank; ``lamport_finish`` of :func:`run_k3_attn_res`).
    ``add_rn``: round-to-nearest add; ``early_poll`` (with a finish): finish the call before the
    grid wait and store after it (``early_poll`` of :func:`run_k3_attn_res`, same bits); vectorized
    kernel."""


@torch.library.custom_op("trtllm::k3_attn_res_lamport_add_rmsnorm", mutates_args=())
def k3_attn_res_lamport_add_rmsnorm(
    prefix: torch.Tensor,
    uc_local: int,
    flags: int,
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
    finish_world: int,
    finish_chunk: int,
    finish_two_shot: bool,
    mc_base: int,
    rank: int,
    add_rn: bool,
    early_poll: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_lamport_add(
        prefix,
        uc_local,
        flags,
        block,
        (res_weight, rms_weight, out_weight),
        rms_eps,
        out_eps,
        (finish_world, finish_chunk, finish_two_shot, mc_base, rank),
        add_rn,
        early_poll,
    )


k3_attn_res_lamport_add_rmsnorm.__doc__ = _LAMPORT_ADD_DOC


@k3_attn_res_lamport_add_rmsnorm.register_fake
def _(prefix, uc_local, flags, block, res_weight, rms_weight, out_weight, rms_eps, out_eps, *args):
    return torch.empty_like(prefix), torch.empty_like(prefix)


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
    "k3_attn_res_add_rmsnorm",
    "k3_attn_res_add_rmsnorm_lamport",
    "k3_attn_res_lamport_add_rmsnorm",
    "k3_attn_res_moe_add_rmsnorm",
    "k3_attn_res_rmsnorm",
    "run_k3_attn_res",
]
