# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ROUTE_IDS16: the MiniMax2 top-4 of a 9..16-row generation step's MoE, from the router's fp32 logits, in one
PDL-launched CuTe DSL kernel, so the TRTLLM-Gen NVFP4 MoE runner routes from the top-k ids (its single-CTA routing
kernel, DynBlock, then skips the per-token top-k of its Phase 1).

One 32-thread CTA per token: lane ``l`` holds experts ``l``, ``l + 32``, ``l + 64`` and ``l + 96``. The arithmetic is
TRTLLM-Gen's MiniMax2 routing (``RoutingCustomPolicy.cuh``: SigmoidBiasPreprocess + ScaledSumNormalizePostprocess,
sumEpsilon 1e-20, routed scaling) operation for operation; the same top-4 code is the in-kernel routing of the
<= 8-row MoE engine this target runs at batch 1 (``kernels/m3_moe/m3_moe_kernel.py``):

* key[e] = fma(0.5, tanh(0.5 x[e]), 0.5) + bias[e]: ``sigmoid_accurate`` (``0.5f * tanhf(0.5f * x) + 0.5f``) as nvcc
  contracts it, plus the fp32 bias;
* the top 4 by ``(TwiddleIn(key) << 32) | (65535 - e)``, the largest first (``TopKRedType::makeCmpVal``): equal keys
  go to the lower expert id, and slot k holds the k-th largest, the order DynBlock writes;
* w[k] = bf16((key[k] - bias[e_k]) * routed_scaling / (sum + 1e-20)), the sum by ``cg::reduce``'s butterfly
  (masks 16, 8, 4, 2, 1) over lanes 0..3's values and zeros elsewhere.

Outputs: expert ids int32 [rows, 4] and weights bf16 [rows, 4] (row-major), the ``topk_ids`` / ``topk_weights`` the
runner takes. The bias is a weight, read before the grid wait; the kernel releases its dependent (the runner's
routing kernel, which waits on this grid before it reads the ids) right after its own grid wait. Kernels are compiled
per PDL launch flag with a symbolic row count; ``prebuild`` compiles and loads them before the executor's warmup, so
a decode step never compiles one. A variant is usable (``is_compiled``) only after its dummy launch completed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.extras import types as _T
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import cute_aot
from tensorrt_llm._torch.kernels.blackwell.utils import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)

EXPERTS = 128  # capacity: fixed: the checkpoint's routed experts
TOP_K = 4  # capacity: fixed: the checkpoint's top-k
PER_LANE = EXPERTS // 32  # capacity: fixed: experts per lane of the token's warp
SUM_EPS = 1e-20  # the MiniMax2 routing's sumEpsilon
MIN_ROWS = 9  # capacity: fixed: past the MoE engine's 8 rows
MAX_ROWS = 16  # capacity: fixed: DynBlock's token bound (DynBlockKernelMaxNumTokens)


# =============================================================================
# Inline-PTX helpers (the MoE16 engine's: round-to-nearest, no flush-to-zero, as nvcc without fast math).
# =============================================================================
def _asm(ret, args, ptx, cons, loc=None, ip=None):
    return _llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        ptx,
        cons,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    args = [cutlass.Float32(a), cutlass.Float32(b), cutlass.Float32(c)]
    return cutlass.Float32(_asm(_T.f32(), args, "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", loc, ip))


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None):
    args = [cutlass.Float32(a), cutlass.Float32(b)]
    return cutlass.Float32(_asm(_T.f32(), args, "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def _sub_rn(a, b, *, loc=None, ip=None):
    args = [cutlass.Float32(a), cutlass.Float32(b)]
    return cutlass.Float32(_asm(_T.f32(), args, "sub.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    args = [cutlass.Float32(a), cutlass.Float32(b)]
    return cutlass.Float32(_asm(_T.f32(), args, "mul.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def _div_rn(a, b, *, loc=None, ip=None):
    args = [cutlass.Float32(a), cutlass.Float32(b)]
    return cutlass.Float32(_asm(_T.f32(), args, "div.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def _redux_max_u32(v, *, loc=None, ip=None):
    args = [cutlass.Int32(v)]
    return cutlass.Int32(
        _asm(_T.i32(), args, "redux.sync.max.u32 $0, $1, 0xffffffff;", "=r,r", loc, ip)
    )


@dsl_user_op
def _bf16_bits(x, *, loc=None, ip=None):
    """The bf16 (nearest even) bits of x in the low 16 bits (``__float2bfloat16``)."""
    args = [cutlass.Float32(x)]
    ptx = "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\ncvt.u32.u16 $0, t;\n}"
    return cutlass.Int32(_asm(_T.i32(), args, ptx, "=r,f", loc, ip))


def _twiddle(score):
    """Float bits mapped so that unsigned order is float order (cub ``TwiddleIn``)."""
    bits = score.bitcast(cutlass.Int32)
    return cutlass.select_(bits < cutlass.Int32(0), ~bits, bits | cutlass.Int32(-2147483648))


# =============================================================================
# The kernel.
# =============================================================================
class _RouteIds16Kernel:
    """One token per 32-thread CTA: the grid wait (with PDL), the dependent's release, then the top-4."""

    def __init__(self, use_pdl: bool):
        self.use_pdl = use_pdl

    @cute.kernel
    def kernel(
        self,
        logits: cute.Tensor,  # fp32 [rows, 128], row-major
        bias: cute.Tensor,  # fp32 [128]
        ids: cute.Tensor,  # int32 [rows, 4], row-major
        wbits: cute.Tensor,  # int16 [rows, 4] (the bf16 weights' bits), row-major
        route_scale: cutlass.Float32,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % cutlass.Int32(32)
        # The bias is a weight (never written by a step): read before the grid wait.
        bv = [cutlass.Float32(bias[lane + cutlass.Int32(i * 32)]) for i in range(PER_LANE)]
        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()
        keys = []
        for i in cutlass.range_constexpr(PER_LANE):
            x = cutlass.Float32(logits[row, lane + cutlass.Int32(i * 32)])
            sg = _fma_rn(
                cutlass.Float32(0.5),
                cute.math.tanh(_mul_rn(cutlass.Float32(0.5), x)),
                cutlass.Float32(0.5),
            )
            keys.append(_add_rn(sg, bv[i]))
        hk = [_twiddle(keys[i]) for i in range(PER_LANE)]
        taken = [cutlass.Boolean(False) for _ in range(PER_LANE)]
        ids_t = []
        key_t = []
        bias_t = []
        for k in cutlass.range_constexpr(TOP_K):
            # This lane's best (key, 65535 - expert) not yet taken, then the warp's: max of the high words, then of
            # the low words among the lanes holding that high word (TopKRedType::reduce on SM 10.x).
            bh = cutlass.Int32(0)
            bl = cutlass.Int32(0)
            for i in cutlass.range_constexpr(PER_LANE):
                lo_i = cutlass.Int32(65535 - i * 32) - lane
                hv = cutlass.select_(taken[i], cutlass.Int32(0), hk[i])
                lv = cutlass.select_(taken[i], cutlass.Int32(0), lo_i)
                better = (cutlass.Uint32(hv) > cutlass.Uint32(bh)) | (
                    (hv == bh) & (cutlass.Uint32(lv) > cutlass.Uint32(bl))
                )
                bh = cutlass.select_(better, hv, bh)
                bl = cutlass.select_(better, lv, bl)
            mh = _redux_max_u32(bh)
            ml = _redux_max_u32(cutlass.select_(bh == mh, bl, cutlass.Int32(0)))
            e_k = cutlass.Int32(65535) - ml
            kv = cutlass.Float32(0.0)
            bk = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(PER_LANE):
                mine = e_k == cutlass.Int32(i * 32) + lane
                taken[i] = taken[i] | mine
                kv = cutlass.select_(mine, keys[i], kv)
                bk = cutlass.select_(mine, bv[i], bk)
            src = e_k % cutlass.Int32(32)
            ids_t.append(e_k)
            key_t.append(cute.arch.shuffle_sync(kv, src))
            bias_t.append(cute.arch.shuffle_sync(bk, src))
        # ScaledSumNormalize: lane k < 4 holds slot k's un-biased score; every other lane adds +0.
        sig = cutlass.Float32(0.0)
        e_l = cutlass.Int32(0)
        for k in cutlass.range_constexpr(TOP_K):
            sk = _sub_rn(key_t[k], bias_t[k])
            sig = cutlass.select_(lane == cutlass.Int32(k), sk, sig)
            e_l = cutlass.select_(lane == cutlass.Int32(k), ids_t[k], e_l)
        total = sig
        for off in (16, 8, 4, 2, 1):
            total = _add_rn(total, cute.arch.shuffle_sync_bfly(total, off))
        w = _div_rn(_mul_rn(sig, route_scale), _add_rn(total, cutlass.Float32(SUM_EPS)))
        if lane < cutlass.Int32(TOP_K):
            ids[row, lane] = e_l
            wbits[row, lane] = cutlass.Int16(_bf16_bits(w))

    @cute.jit
    def __call__(self, logits, bias, ids, wbits, route_scale: cutlass.Float32, stream):
        self.kernel(logits, bias, ids, wbits, route_scale).launch(
            grid=(logits.shape[0], 1, 1),
            block=(32, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )


#: Kernels by PDL launch flag, each compiled AND launched once on dummy rows (``prebuild``): ``route`` runs only these.
kernel_cache: Dict[bool, object] = {}


def _compile(use_pdl: bool):
    """The compiled (or AOT-loaded) kernel of this PDL launch flag; not yet usable by ``route`` (see ``prebuild``)."""
    use_pdl = bool(use_pdl)
    if use_pdl in kernel_cache:
        return kernel_cache[use_pdl]
    rows = cute.sym_int()
    logits = make_fake_compact_tensor(cutlass.Float32, (rows, EXPERTS), stride_order=(1, 0))
    bias = make_fake_compact_tensor(cutlass.Float32, (EXPERTS,), stride_order=(0,))
    ids = make_fake_compact_tensor(cutlass.Int32, (rows, TOP_K), stride_order=(1, 0))
    wbits = make_fake_compact_tensor(cutlass.Int16, (rows, TOP_K), stride_order=(1, 0))
    return cute_aot.compile_or_load(
        f"m31_route_ids16 pdl={use_pdl}",
        lambda: cute.compile(
            _RouteIds16Kernel(use_pdl),
            logits,
            bias,
            ids,
            wbits,
            cutlass.Float32(1.0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        ),
        sources=_aot_sources(),
        tvm_ffi=True,
    )


def is_compiled(use_pdl: bool) -> bool:
    return bool(use_pdl) in kernel_cache


def operands_ok(logits: torch.Tensor, bias: torch.Tensor) -> bool:
    """Whether the kernel takes these operands: fp32 [9..16, 128] logits and an fp32 [128] bias, contiguous and
    16-byte aligned on one CUDA device."""
    return (
        logits.dim() == 2
        and MIN_ROWS <= logits.shape[0] <= MAX_ROWS
        and logits.shape[1] == EXPERTS
        and logits.dtype == torch.float32
        and logits.is_cuda
        and logits.is_contiguous()
        and logits.data_ptr() % 16 == 0
        and bias.dtype == torch.float32
        and bias.numel() == EXPERTS
        and bias.is_contiguous()
        and bias.device == logits.device
        and bias.data_ptr() % 16 == 0
    )


def route(
    logits: torch.Tensor, bias: torch.Tensor, routed_scaling: float, use_pdl: bool
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(top-4 expert ids int32 [rows, 4], bf16 weights [rows, 4]) of fp32 router logits [rows, 128], or None (operands
    outside the kernel's, or its PDL variant not compiled before warmup): the caller keeps the logits path."""
    if not operands_ok(logits, bias) or bool(use_pdl) not in kernel_cache:
        return None
    rows = logits.shape[0]
    ids = torch.empty(rows, TOP_K, dtype=torch.int32, device=logits.device)
    weights = torch.empty(rows, TOP_K, dtype=torch.bfloat16, device=logits.device)
    kernel_cache[bool(use_pdl)](
        logits, bias.view(EXPERTS), ids, weights.view(torch.int16), float(routed_scaling)
    )
    return ids, weights


def prebuild(use_pdl: bool, device) -> bool:
    """Compile the ``use_pdl`` kernel and launch it once on dummy rows (loading its module), before warmup. The
    variant enters ``kernel_cache`` (so ``route`` runs it) only after that launch completed; a compile or launch
    error propagates and leaves it out."""
    use_pdl = bool(use_pdl)
    fn = _compile(use_pdl)
    kernel_cache.pop(use_pdl, None)
    logits = torch.zeros(MAX_ROWS, EXPERTS, dtype=torch.float32, device=device)
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device=device)
    ids = torch.full((MAX_ROWS, TOP_K), -1, dtype=torch.int32, device=device)
    weights = torch.empty(MAX_ROWS, TOP_K, dtype=torch.int16, device=device)
    fn(logits, bias, ids, weights, 1.0)
    torch.cuda.synchronize(device)
    # All-equal keys: every row takes experts 0..3 (ties to the lower id).
    if not bool((ids == torch.arange(TOP_K, dtype=torch.int32, device=device)).all()):
        raise RuntimeError(
            f"m31_route_ids16 pdl={use_pdl}: the dummy launch did not write experts 0..3"
        )
    kernel_cache[use_pdl] = fn
    return True


def _aot_sources():
    """The Python files this module's CuTe DSL kernel is traced from (the target package and the shared Blackwell
    kernel helpers): the ``sources`` of its shipped AOT object."""
    target = Path(__file__).resolve().parents[1]
    shared = cute_aot.BLACKWELL_DIR
    return cute_aot.py_files(str(target), str(shared))
