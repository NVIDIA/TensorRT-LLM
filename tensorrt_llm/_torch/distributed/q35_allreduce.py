# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot MNNVL all-reduce + residual + RMSNorm in Triton, sharing the C++ MNNVL workspace protocol.

``Q35FusedAllReduce`` performs exactly what ``torch.ops.trtllm.mnnvl_fusion_allreduce`` does in its
one-shot form (RESIDUAL_RMS_NORM, optionally with the FP8 quantized output), on the same fabric
workspace and the same Lamport triple buffer / control flags (``MNNVLAllReduce``'s
``buffer_flags``), so its launches interleave freely with the library kernel's (one-shot and
two-shot) in one CUDA graph. On top of the library kernel it can:

* fold the Qwen3-Next shared-expert combine into the all-reduce input
  (``x = routed + sigmoid(gate) * shared``, the arithmetic of ``fused_sigmoid_gate_mul_add``),
  removing that kernel from every MoE boundary;
* stay one-shot beyond 16 rows (the library switches to two-shot + a separate RMSNorm kernel above
  1 MiB of payload, i.e. at the 28-row b4 verify).

Protocol (``cpp/tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.cu`` one-shot, and
``lamportUtils.cuh``): flags (uint32) = [current buffer, dirty buffer, bytes per buffer, dirty
stages, 4 x dirty bytes per stage, arrival counter]. Every program reads the flags after
``griddepcontrol.wait`` and then counts its arrival; it multicasts its row (BF16 -0.0 sent as +0.0)
into slot [token][rank] of the current buffer, clears its share of the previous call's (dirty)
buffer to the -0.0 sentinel, polls its own copy until no 32-bit word holds the sentinel, sums the
ranks in rank order in FP32 (bitwise the library's sum), adds the residual (BF16, bitwise the
library's) and normalizes. Program 0 finally waits for every arrival and rotates the buffers.
The RMSNorm sum of squares (BF16 squares, FP32 accumulation, as the library) is reduced in a
different order than the library's cluster reduction: the norm output can differ by one BF16 ulp.

Dependents are released (``griddepcontrol.launch_dependents``) after the reduction, as the library
kernel does. One program (512 threads) per token; hidden % 2048 == 0; TP group of 2, 4 or 8 ranks.
"""

from __future__ import annotations

import os

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_SENTINEL = tl.constexpr(-2147483648)  # 0x80000000: Lamport "not written" (two BF16: +0.0, -0.0)
_HI_MASK = tl.constexpr(-65536)  # 0xFFFF0000


@triton.jit
def _gdc_wait_clobber():
    tl.inline_asm_elementwise("griddepcontrol.wait; // dummy $0", "=r,~{memory}", [], dtype=tl.int32,
                              is_pure=False, pack=1)


@triton.jit
def _lo_f32(w):
    """FP32 value of the BF16 in the low half of each int32 word."""
    return (w << 16).to(tl.float32, bitcast=True)


@triton.jit
def _hi_f32(w):
    """FP32 value of the BF16 in the high half of each int32 word."""
    return (w & _HI_MASK).to(tl.float32, bitcast=True)


@triton.jit
def _bf16_bits(x):
    """FP32 -> BF16 (round to nearest even) bits in the low 16 bits of an int32."""
    return x.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535


@triton.jit
def _pack(lo, hi):
    return _bf16_bits(lo) | (_bf16_bits(hi) << 16)


@triton.jit
def _sanitize(w):
    """BF16 -0.0 halves -> +0.0, so no written word equals the sentinel."""
    w = tl.where((w & 65535) == 32768, w & _HI_MASK, w)
    return tl.where((w & _HI_MASK) == _SENTINEL, w & 65535, w)


@triton.jit
def _load_rank(base, row_words, r: tl.constexpr, W: tl.constexpr, cols):
    if r < W:
        return tl.load(base + r * row_words + cols, volatile=True)
    else:
        return tl.zeros_like(cols)


@triton.jit
def _pending(w, r: tl.constexpr, W: tl.constexpr):
    if r < W:
        return tl.sum((w == _SENTINEL).to(tl.int32))
    else:
        return 0


@triton.jit(do_not_specialize=["m", "rank", "eps"])
def _q35_oneshot_ar_kernel(
    x_ptr,  # int32 view [m, H/2] of the BF16 input (unused with SE_FOLD)
    f_ptr,  # int32 view [m, H/2]: routed MoE output (SE_FOLD)
    sh_ptr,  # int32 view [m, H/2]: shared-expert output (SE_FOLD)
    g_ptr,  # [m] shared-expert gate logits (SE_FOLD)
    res_ptr,  # int32 view [m, H/2]: residual in
    gamma_ptr,  # int32 view [H/2]: RMSNorm weight (BF16)
    scale_ptr,  # fp32 [1]: FP8 scale (QUANT)
    norm_ptr,  # int32 view [m, H/2]: RMSNorm out (NORM_OUT)
    resout_ptr,  # int32 view [m, H/2]: residual out
    q_ptr,  # int16 view [m, H/2]: FP8 out, two E4M3 per element (QUANT)
    uc_ptr,  # int32 view of this rank's workspace (3 buffers)
    mc_ptr,  # int32 view of the multicast workspace
    flags_ptr,  # int32 view of buffer_flags [9]
    m,
    rank,
    eps,
    W: tl.constexpr,
    H: tl.constexpr,
    SE_FOLD: tl.constexpr,
    NORM_OUT: tl.constexpr,
    QUANT: tl.constexpr,
    WAIT: tl.constexpr,
):
    HW: tl.constexpr = H // 2  # int32 words per row
    t = tl.program_id(0)
    cols = tl.arange(0, HW)
    gamma = tl.load(gamma_ptr + cols)  # a parameter: safe ahead of the dependency wait
    if WAIT:
        _gdc_wait_clobber()

    # ---- Lamport state of this launch; count the arrival only after reading it ----
    cur = tl.load(flags_ptr + 0, volatile=True)
    dirty = tl.load(flags_ptr + 1, volatile=True)
    # Buffers are multiples of 8 MiB: the hint lets the row accesses vectorize (16 B).
    bpb_words = tl.multiple_of(tl.load(flags_ptr + 2, volatile=True) // 4, 1024)
    dstages = tl.load(flags_ptr + 3, volatile=True)
    tl.atomic_add(flags_ptr + 8, 1, sem="release", scope="gpu")

    # ---- this rank's row (optionally the shared-expert combine), multicast to every rank ----
    row = t.to(tl.int64) * HW + cols
    if SE_FOLD:
        s = tl.sigmoid(tl.load(g_ptr + t).to(tl.float32))
        fw = tl.load(f_ptr + row)
        shw = tl.load(sh_ptr + row)
        xw = _pack(_lo_f32(fw) + s * _lo_f32(shw), _hi_f32(fw) + s * _hi_f32(shw))
    else:
        xw = tl.load(x_ptr + row)
    xw = _sanitize(xw)
    cur_base = cur.to(tl.int64) * bpb_words
    slot = cur_base + (t.to(tl.int64) * W + rank) * HW
    tl.store(mc_ptr + slot + cols, xw)

    # ---- clear this program's share of the previous call's buffer (all its stages) ----
    for s_i in range(0, dstages):
        stage_words = tl.multiple_of(bpb_words // dstages, 1024)
        stage_base = (dirty.to(tl.int64) * dstages + s_i) * stage_words
        n_words = tl.multiple_of((tl.load(flags_ptr + 4 + s_i, volatile=True) + 15) // 16 * 4, 4)
        for start in range(t * HW, n_words, m * HW):
            tl.store(uc_ptr + stage_base + start + cols, tl.full([HW], _SENTINEL, tl.int32),
                     mask=start + cols < n_words)

    # ---- wait for the token's row of every rank (one load of all ranks per round) ----
    tok_base = uc_ptr + cur_base + t.to(tl.int64) * W * HW
    w0 = _load_rank(tok_base, HW, 0, W, cols)
    w1 = _load_rank(tok_base, HW, 1, W, cols)
    w2 = _load_rank(tok_base, HW, 2, W, cols)
    w3 = _load_rank(tok_base, HW, 3, W, cols)
    w4 = _load_rank(tok_base, HW, 4, W, cols)
    w5 = _load_rank(tok_base, HW, 5, W, cols)
    w6 = _load_rank(tok_base, HW, 6, W, cols)
    w7 = _load_rank(tok_base, HW, 7, W, cols)
    pending = (_pending(w0, 0, W) + _pending(w1, 1, W) + _pending(w2, 2, W) + _pending(w3, 3, W) +
               _pending(w4, 4, W) + _pending(w5, 5, W) + _pending(w6, 6, W) + _pending(w7, 7, W))
    while pending > 0:
        w0 = _load_rank(tok_base, HW, 0, W, cols)
        w1 = _load_rank(tok_base, HW, 1, W, cols)
        w2 = _load_rank(tok_base, HW, 2, W, cols)
        w3 = _load_rank(tok_base, HW, 3, W, cols)
        w4 = _load_rank(tok_base, HW, 4, W, cols)
        w5 = _load_rank(tok_base, HW, 5, W, cols)
        w6 = _load_rank(tok_base, HW, 6, W, cols)
        w7 = _load_rank(tok_base, HW, 7, W, cols)
        pending = (_pending(w0, 0, W) + _pending(w1, 1, W) + _pending(w2, 2, W) + _pending(w3, 3, W) +
                   _pending(w4, 4, W) + _pending(w5, 5, W) + _pending(w6, 6, W) + _pending(w7, 7, W))

    # ---- FP32 sum in rank order, rounded to BF16 once (the library's deterministic order) ----
    zero = tl.zeros([HW], tl.float32)
    lo = zero + _lo_f32(w0)
    hi = zero + _hi_f32(w0)
    lo = lo + _lo_f32(w1)
    hi = hi + _hi_f32(w1)
    if W > 2:
        lo = lo + _lo_f32(w2)
        hi = hi + _hi_f32(w2)
        lo = lo + _lo_f32(w3)
        hi = hi + _hi_f32(w3)
    if W > 4:
        lo = lo + _lo_f32(w4)
        hi = hi + _hi_f32(w4)
        lo = lo + _lo_f32(w5)
        hi = hi + _hi_f32(w5)
        lo = lo + _lo_f32(w6)
        hi = hi + _hi_f32(w6)
        lo = lo + _lo_f32(w7)
        hi = hi + _hi_f32(w7)
    tl.extra.cuda.gdc_launch_dependents()

    # ---- residual (BF16 add) and RMSNorm (BF16 squares, FP32 sum) ----
    sw = _pack(lo, hi)
    rw = tl.load(res_ptr + row)
    outw = _pack(_lo_f32(sw) + _lo_f32(rw), _hi_f32(sw) + _hi_f32(rw))
    tl.store(resout_ptr + row, outw)
    o_lo = _lo_f32(outw)
    o_hi = _hi_f32(outw)
    sq = _lo_f32(_pack(o_lo * o_lo, o_hi * o_hi)) + _hi_f32(_pack(o_lo * o_lo, o_hi * o_hi))
    total = tl.sum(sq)
    rstd = tl.math.rsqrt(total / H + eps)
    y_lo = o_lo * rstd * _lo_f32(gamma)
    y_hi = o_hi * rstd * _hi_f32(gamma)
    yw = _pack(y_lo, y_hi)
    if NORM_OUT:
        tl.store(norm_ptr + row, yw)
    if QUANT:
        inv = tl.math.div_rn(1.0, tl.load(scale_ptr))
        q_lo = (_lo_f32(yw) * inv).to(tl.float8e4nv).to(tl.int8, bitcast=True).to(tl.int16) & 255
        q_hi = (_hi_f32(yw) * inv).to(tl.float8e4nv).to(tl.int8, bitcast=True).to(tl.int16) & 255
        tl.store(q_ptr + row, q_lo | (q_hi << 8))

    # ---- hand the triple buffer to the next call once every program has read the flags ----
    if t == 0:
        arrived = tl.atomic_add(flags_ptr + 8, 0, sem="acquire", scope="gpu")
        while arrived < m:
            arrived = tl.atomic_add(flags_ptr + 8, 0, sem="acquire", scope="gpu")
        tl.store(flags_ptr + 0, (cur + 1) % 3)
        tl.store(flags_ptr + 1, cur)
        tl.store(flags_ptr + 3, 1)
        tl.store(flags_ptr + 4, m * W * H * 2)
        tl.store(flags_ptr + 5, 0)
        tl.store(flags_ptr + 6, 0)
        tl.store(flags_ptr + 7, 0)
        tl.store(flags_ptr + 8, 0)


_NUM_WARPS = 16


class Q35FusedAllReduce:
    """One-shot all-reduce + residual + RMSNorm (+ shared-expert fold, + FP8 output) on the
    ``MNNVLAllReduce`` workspace of ``mapping``'s TP group (see the module docstring)."""

    def __init__(self, mapping, dtype: torch.dtype = torch.bfloat16):
        from .ops import MNNVLAllReduce

        if dtype != torch.bfloat16 or mapping.tp_size not in (2, 4, 8):
            raise ValueError("Q35FusedAllReduce supports BF16 and TP 2/4/8")
        self.mapping = mapping
        self.dtype = dtype
        self.mnnvl = MNNVLAllReduce(mapping, dtype)  # creates / shares the workspace
        self._mc_views = {}
        from ..flashinfer_utils import get_env_enable_pdl
        self._pdl = get_env_enable_pdl()

    def _workspace(self, rows: int, hidden: int):
        from .ops import get_or_scale_allreduce_mnnvl_workspace
        ws = get_or_scale_allreduce_mnnvl_workspace(self.mapping, self.dtype,
                                                    buffer_size_bytes=rows * hidden * self.mapping.tp_size * 2)
        if ws is None:
            raise RuntimeError(f"MNNVL workspace sized once (TRTLLM_Q35_MNNVL_PRESIZE) is too small for {rows} rows")
        uc = ws["uc_buffer"].view(torch.int32)
        key = id(ws["handle"])
        mc = self._mc_views.get(key)
        if mc is None:
            mc = ws["handle"].get_mc_buffer([uc.numel()], torch.int32, 0)
            self._mc_views = {key: mc}
        return uc, mc, ws["buffer_flags"].view(torch.int32)

    def __call__(self,
                 residual: torch.Tensor,
                 norm_weight: torch.Tensor,
                 eps: float,
                 x: Optional[torch.Tensor] = None,
                 shared_fold: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                 scale: Optional[torch.Tensor] = None,
                 norm_out: bool = True) -> Tuple[torch.Tensor, ...]:
        """``x`` [M, H] BF16 (or ``shared_fold`` = (routed [M, H], shared [M, H], gate logits [M] /
        [M, 1])); returns ``(norm, residual_out)``, ``(norm, fp8, residual_out)`` with ``scale``, or
        ``(fp8, residual_out)`` with ``scale`` and ``norm_out=False`` -- the outputs of the library
        op for RESIDUAL_RMS_NORM / _OUT_QUANT_FP8 / _QUANT_FP8."""
        src = x if x is not None else shared_fold[0]
        m, hidden = src.shape
        assert hidden % 2048 == 0 and residual.shape == src.shape
        uc, mc, flags = self._workspace(m, hidden)
        norm = torch.empty_like(residual) if norm_out else None
        res_out = torch.empty_like(residual)
        quant = (torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
                 if scale is not None else None)
        i32 = lambda t: t.contiguous().view(torch.int32)  # noqa: E731
        dummy = flags
        if shared_fold is not None:
            routed, shared, gate = shared_fold
            f, sh, g, xin = i32(routed), i32(shared), gate.contiguous().reshape(-1), dummy
        else:
            f = sh = g = dummy
            xin = i32(x)
        _q35_oneshot_ar_kernel[(m, )](
            xin,
            f,
            sh,
            g,
            i32(residual),
            i32(norm_weight),
            scale if scale is not None else dummy,
            norm.view(torch.int32) if norm is not None else dummy,
            res_out.view(torch.int32),
            quant.view(torch.int16) if quant is not None else dummy,
            uc,
            mc,
            flags,
            m,
            self.mapping.tp_rank,
            eps,
            W=self.mapping.tp_size,
            H=hidden,
            SE_FOLD=shared_fold is not None,
            NORM_OUT=norm_out,
            QUANT=scale is not None,
            WAIT=self._pdl,
            num_warps=_NUM_WARPS,
            launch_pdl=self._pdl,
        )
        if scale is None:
            return norm, res_out
        if norm_out:
            return norm, quant, res_out
        return quant, res_out


_INSTANCES = {}
MAX_ROWS = 64  # one program per row; decode / spec-verify shapes only


def get_q35_fused_allreduce(mapping) -> Optional[Q35FusedAllReduce]:
    """The group's Triton all-reduce (lazy; None where the MNNVL workspace is unavailable).

    Every rank of the group decides the same way (same mapping, same shapes)."""
    key = (tuple(mapping.tp_group), mapping.tp_rank)
    if key not in _INSTANCES:
        from .ops import MNNVLAllReduce
        inst = None
        if (mapping.tp_size in (2, 4, 8) and not mapping.enable_attention_dp
                and MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16)):
            inst = Q35FusedAllReduce(mapping)
        _INSTANCES[key] = inst
    return _INSTANCES[key]


# ------------------------------------------------------------------------------------------------
# CuTe DSL kernel (cute_dsl_kernels/blackwell/q35_allreduce_norm.py): the library one-shot's thread
# layout (8-CTA cluster per token) and arithmetic order -> bitwise the library op's outputs.


class _GraphSafeDlpack:
    """DLPack export without a stream sync (safe under CUDA-graph capture)."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor.detach()

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _cute(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack
    return from_dlpack(_GraphSafeDlpack(t), assumed_align=16)


def _ctas_per_token(rows: int, hidden: int) -> int:
    """CTAs per token: the library one-shot's ``adjustGridConfig`` for 16-byte vectors -- the widest
    cluster (<= 8) whose CTAs keep >= 64 threads and whose ``rows x cluster`` fits the SMs (so its
    reduction tree, and the norm output, match the library bit for bit where both are one-shot).
    ``TRTLLM_Q35_FUSED_AR_CTAS`` overrides it."""
    env = os.environ.get("TRTLLM_Q35_FUSED_AR_CTAS")
    if env:
        return int(env)
    threads = hidden // 8
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    ctas = 8
    while ctas > 1 and (threads // ctas < 64 or (threads // ctas) % 32 or rows * ctas > sms):
        ctas //= 2
    return ctas


import functools  # noqa: E402


# Rows from which the CuTe kernel uses its fused two-shot mode (0: never). Default by group size from GB300
# microbenchmarks: TP16 from 12 rows (one-shot ingress doubles: 14 rows one-shot 6.68 / two-shot 6.51 us, 28 rows
# 9.65 / 7.52), never at TP8 and below (28 rows: one-shot 6.93 / two-shot 7.28). TRTLLM_Q35_FUSED_AR_TWOSHOT_ROWS
# overrides.
_TWO_SHOT_ROWS_ENV = os.environ.get("TRTLLM_Q35_FUSED_AR_TWOSHOT_ROWS")


def _two_shot_rows(nranks: int) -> int:
    if _TWO_SHOT_ROWS_ENV is not None:
        return int(_TWO_SHOT_ROWS_ENV)
    return 12 if nranks >= 16 else 0


@functools.lru_cache(maxsize=None)
def _compiled_cute_ar(m: int, hidden: int, eps: float, nranks: int, ctas: int, use_pdl: bool, se_fold: bool,
                      norm_out: bool, quant: bool, moe_topk: int = 0, fc2_row: int = 0, two_shot: bool = False):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.q35_allreduce_norm import Q35AllReduceNormKernel

    kernel = Q35AllReduceNormKernel(hidden=hidden, norm_eps=eps, nranks=nranks, ctas=ctas, use_pdl=use_pdl,
                                    se_fold=se_fold, norm_out=norm_out, quant=quant, moe_topk=moe_topk,
                                    fc2_row=fc2_row, two_shot=two_shot)
    dev = torch.device("cuda")
    x = torch.empty((m, hidden), dtype=torch.bfloat16, device=dev)
    gate = torch.empty((m, ), dtype=torch.bfloat16, device=dev)
    w = torch.empty((hidden, ), dtype=torch.bfloat16, device=dev)
    scale = torch.empty((1, ), dtype=torch.float32, device=dev)
    q = torch.empty((m, hidden), dtype=torch.uint8, device=dev)
    table = torch.empty((3 + nranks, ), dtype=torch.int64, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel, _cute(x), _cute(x), _cute(x), _cute(gate), _cute(x), _cute(w), _cute(scale),
                        _cute(x), _cute(x), _cute(q), _cute(table), cutlass.Int64(0), cutlass.Int64(0),
                        cutlass.Int64(0), cutlass.Int32(0), stream)


class Q35CuteAllReduce:
    """Same interface and outputs as :class:`Q35FusedAllReduce`, on the CuTe DSL kernel (bitwise the
    library op; also one-shot above 16 rows and with the shared-expert fold)."""

    def __init__(self, mapping, dtype: torch.dtype = torch.bfloat16):
        from .ops import MNNVLAllReduce

        if dtype != torch.bfloat16 or mapping.tp_size not in (2, 4, 8, 16):
            raise ValueError("Q35CuteAllReduce supports BF16 and TP 2/4/8/16")
        self.mapping = mapping
        self.dtype = dtype
        self.mnnvl = MNNVLAllReduce(mapping, dtype)
        self._ws = None
        self._views = None
        # (local buffer, multicast buffer, flags, every rank's unicast buffer)
        self.table = torch.zeros((3 + mapping.tp_size, ),
                                 dtype=torch.int64,
                                 device=torch.device("cuda", torch.cuda.current_device()))
        from ..flashinfer_utils import get_env_enable_pdl
        self._pdl = get_env_enable_pdl()
        self._dummies = {}

    def _dummy(self, m: int, hidden: int):
        """Never-read operands of the compiled signature (gate [m], FP8 out [m, H])."""
        d = self._dummies.get((m, hidden))
        if d is None:
            dev = self.table.device
            d = (torch.empty((m, ), dtype=torch.bfloat16, device=dev),
                 torch.empty((m, hidden), dtype=torch.uint8, device=dev), self.table.view(torch.float32)[:1])
            self._dummies[(m, hidden)] = d
        return d

    def _table(self, nbytes: int) -> torch.Tensor:
        from .ops import get_or_scale_allreduce_mnnvl_workspace
        ws = get_or_scale_allreduce_mnnvl_workspace(self.mapping, self.dtype, buffer_size_bytes=nbytes)
        if ws is None:
            raise RuntimeError(f"MNNVL workspace sized once (TRTLLM_Q35_MNNVL_PRESIZE) is too small for {nbytes} B")
        if ws is not self._ws:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the MNNVL workspace changed during CUDA graph capture")
            local = ws["uc_buffer"]
            mcast = ws["handle"].get_mc_buffer(tuple(local.shape), local.dtype, 0)
            flags = ws["buffer_flags"]
            peers = [ws["handle"].get_uc_buffer(r, (1, ), torch.float32, 0) for r in range(self.mapping.tp_size)]
            self.table.copy_(
                torch.tensor([local.data_ptr(), mcast.data_ptr(), flags.data_ptr()] + [p.data_ptr() for p in peers],
                             dtype=torch.int64))
            self._ws = ws
            self._views = (local, mcast, flags, peers)
        return self.table

    @staticmethod
    def moe_fold_supported(fc2: torch.Tensor, topk_weights: torch.Tensor, perm: torch.Tensor, hidden: int) -> bool:
        """Can ``shared_fold``'s routed operand be this un-finalized trtllm-gen MoE output?"""
        return (fc2.dtype == torch.bfloat16 and fc2.dim() == 2 and fc2.stride(1) == 1 and fc2.shape[1] >= hidden
                and fc2.stride(0) % 8 == 0 and fc2.data_ptr() % 16 == 0 and topk_weights.dtype == torch.bfloat16
                and perm.dtype == torch.int32 and topk_weights.shape == perm.shape and perm.dim() == 2
                and 0 < perm.shape[1] <= 16)

    def __call__(self,
                 residual: torch.Tensor,
                 norm_weight: torch.Tensor,
                 eps: float,
                 x: Optional[torch.Tensor] = None,
                 shared_fold: Optional[Tuple] = None,
                 scale: Optional[torch.Tensor] = None,
                 norm_out: bool = True,
                 two_shot: Optional[bool] = None) -> Tuple[torch.Tensor, ...]:
        """``shared_fold`` = (routed, shared, gate); ``routed`` may be the un-finalized trtllm-gen MoE
        output ``(fc2 [P, >= H] BF16, topk_weights [M, K] BF16, expanded_idx_to_permuted_idx [M, K]
        int32)`` (do_finalize=False), finalized in-kernel. ``two_shot``: None = by rows and group size
        (``_two_shot_rows``)."""
        import cuda.bindings.driver as cuda
        import cutlass

        moe = None
        if shared_fold is not None and isinstance(shared_fold[0], (tuple, list)):
            moe = shared_fold[0]
        src = x if x is not None else shared_fold[1]
        m, hidden = src.shape
        nranks = self.mapping.tp_size
        if two_shot is None:
            thr = _two_shot_rows(nranks)
            two_shot = 0 < thr <= m
        # one-shot: [rank][token][H] in one stage; two-shot: two stages of max(scatter, broadcast) each
        table = self._table(max(m * hidden * nranks * 2, 2 * ((m + nranks - 1) // nranks) * nranks * hidden * 2))
        ctas = _ctas_per_token(m, hidden)
        topk = moe[2].shape[1] if moe is not None else 0
        fc2_row = moe[0].stride(0) if moe is not None else 0
        fn = _compiled_cute_ar(m, hidden, float(eps), nranks, ctas, self._pdl, shared_fold is not None, norm_out,
                               scale is not None, topk, fc2_row, bool(two_shot))
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        quant = (torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
                 if scale is not None else None)
        d_gate, d_quant, d_scale = self._dummy(m, hidden)
        fc2_ptr = topw_ptr = perm_ptr = 0
        if shared_fold is not None:
            routed, shared, gate = shared_fold
            if moe is not None:
                fc2, topw, perm = moe
                assert perm.shape[0] == m and self.moe_fold_supported(fc2, topw, perm, hidden)
                fc2_ptr, topw_ptr, perm_ptr = fc2.data_ptr(), topw.contiguous().data_ptr(), perm.contiguous().data_ptr()
                routed = shared  # unread
            xin, gate = routed, gate.reshape(-1)
        else:
            xin, routed, shared, gate = x, x, x, d_gate
        w = norm_weight.reshape(-1)
        fn(_cute(xin.contiguous()), _cute(routed.contiguous()), _cute(shared.contiguous()), _cute(gate.contiguous()),
           _cute(residual.contiguous()), _cute(w), _cute(scale.reshape(1) if scale is not None else d_scale),
           _cute(res_out), _cute(norm), _cute(quant.view(torch.uint8) if quant is not None else d_quant),
           _cute(table), cutlass.Int64(fc2_ptr), cutlass.Int64(topw_ptr), cutlass.Int64(perm_ptr),
           cutlass.Int32(self.mapping.tp_rank), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        if scale is None:
            return norm, res_out
        if norm_out:
            return norm, quant, res_out
        return quant, res_out


_CUTE_INSTANCES = {}


def get_q35_cute_allreduce(mapping) -> Optional[Q35CuteAllReduce]:
    key = (tuple(mapping.tp_group), mapping.tp_rank)
    if key not in _CUTE_INSTANCES:
        from .ops import MNNVLAllReduce
        inst = None
        if (mapping.tp_size in (2, 4, 8, 16) and not mapping.enable_attention_dp
                and MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16)):
            inst = Q35CuteAllReduce(mapping)
        _CUTE_INSTANCES[key] = inst
    return _CUTE_INSTANCES[key]


# ---------------------------------------------------------------------------------------------------------------
# Row-parallel projection (o_proj / out_proj) + all-reduce + residual + RMSNorm in one launch
# (cute_dsl_kernels/blackwell/q35_gemm_ar_norm.py), TRTLLM_Q35_GEMM_AR.
# ---------------------------------------------------------------------------------------------------------------
_GEMM_AR_MAX_CTAS = 1024


@functools.lru_cache(maxsize=None)
def _compiled_gemm_ar(m: int, k: int, hidden: int, k_warps: int, n_tiles: int, swap: bool, fp8_act: str,
                      a_stride: int, eps: float, nranks: int, use_pdl: bool, min_blocks: int, norm_out: bool,
                      quant: bool, stamps: bool = False, row_ar: bool = True, two_shot: bool = False):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.q35_gemm_ar_norm import PART_SLOT_ROWS, Q35GemmArNormKernel

    kernel = Q35GemmArNormKernel(num_rows=m, k_extent=k, k_warps=k_warps, n_tiles=n_tiles, swap_ab=swap,
                                 fp8_act=fp8_act, hidden=hidden, norm_eps=eps, nranks=nranks, use_pdl=use_pdl,
                                 min_blocks=min_blocks, norm_out=norm_out, quant=quant,
                                 max_ctas=_GEMM_AR_MAX_CTAS, stamps=stamps, row_ar=row_ar, two_shot=two_shot)
    dev = torch.device("cuda")
    a16 = torch.empty((m, a_stride), dtype=torch.bfloat16, device=dev)[:, :k] if fp8_act != "fp8" else \
        torch.empty((1, 8), dtype=torch.bfloat16, device=dev)
    a8 = torch.empty((m, a_stride), dtype=torch.uint8, device=dev)[:, :k] if fp8_act == "fp8" else \
        torch.empty((1, 16), dtype=torch.uint8, device=dev)
    b = torch.empty((hidden, k), dtype=torch.uint8, device=dev)
    s = torch.empty((1, ), dtype=torch.float32, device=dev)
    x = torch.empty((m, hidden), dtype=torch.bfloat16, device=dev)
    w = torch.empty((hidden, ), dtype=torch.bfloat16, device=dev)
    qo = torch.empty((m, hidden), dtype=torch.uint8, device=dev)
    table = torch.empty((3 + nranks, ), dtype=torch.int64, device=dev)
    part = torch.empty((2 * PART_SLOT_ROWS * _GEMM_AR_MAX_CTAS, ), dtype=torch.int64, device=dev)
    epoch = torch.empty((1, ), dtype=torch.int32, device=dev)
    dbg = torch.empty((kernel.ctas * 8 if stamps else 8, ), dtype=torch.int64, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel, _cute(a16), _cute(a8), _cute(b), _cute(s), _cute(s), _cute(x), _cute(w), _cute(s),
                        _cute(x), _cute(x), _cute(qo), _cute(table), _cute(part), _cute(epoch), _cute(dbg),
                        cutlass.Int32(0), stream)


class Q35GemmArNorm:
    """``x @ W.T`` (this rank's row-parallel partial, FP8 per-tensor W8A8 on the low-M warp-MMA GEMM) +
    all-reduce + residual + RMSNorm (+ FP8 out) in one launch on the library's MNNVL workspace."""

    def __init__(self, mapping):
        from ..cute_dsl_kernels.blackwell.q35_gemm_ar_norm import PART_SLOT_ROWS
        self.mapping = mapping
        self.ar = Q35CuteAllReduce(mapping)  # owns the workspace table refresh
        dev = self.ar.table.device
        self.part = torch.zeros((2 * PART_SLOT_ROWS * _GEMM_AR_MAX_CTAS, ), dtype=torch.int64, device=dev)
        self.epoch = torch.ones((1, ), dtype=torch.int32, device=dev)
        self._sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self._dummy_q = {}
        self._dbg_dummy = torch.zeros((8, ), dtype=torch.int64, device=dev)

    def tactic(self, m: int, hidden: int, k: int, fp8_act: str):
        """(k_warps, n_tiles, swap, min_blocks) of the GEMM part, or None when the fused kernel cannot run
        this shape. The K split (k_warps, hence every output's FP32 summation order) is gemm's measured LowM
        tactic; the column tiles per CTA are raised until the grid fits one CTA per SM: the partial-sum
        exchange is a cross-CTA wait, so every CTA must be resident (TRTLLM_Q35_GEMM_AR_TACTIC=kw x nt
        overrides)."""
        from ..cute_dsl_kernels.blackwell.q35_gemm_ar_norm import MAX_ROWS
        from ..modules.lowm_linear import default_tactic
        if not 0 < m <= MAX_ROWS:
            return None
        env = os.environ.get("TRTLLM_Q35_GEMM_AR_TACTIC")
        if env:
            k_warps, n_tiles = (int(v) for v in env.lower().split("x"))
        else:
            k_warps, n_tiles, ksplit, _, _ = default_tactic(m, hidden, k, True, False, fp8_act)
            if ksplit != 1:
                return None
            while n_tiles < 4 and hidden // (8 * n_tiles) > self._sms:
                n_tiles *= 2
        ctas = hidden // (8 * n_tiles)
        if (hidden % (8 * n_tiles) or ctas > self._sms or ctas > _GEMM_AR_MAX_CTAS or m * n_tiles > 32 * k_warps
                or k % (64 * k_warps)):
            return None
        vpt = (hidden // 8) // (32 * k_warps)  # row all-reduce: 16-byte vectors per thread
        if (hidden // 8) % (32 * k_warps) or not 1 <= vpt <= 4 or m > ctas:
            return None
        # TP16 (K = 512): the fused one-shot (< 12 rows) and the 4-vector-per-thread row CTAs (k_warps 4)
        # lose to LowM + CuTe AR by 1.7-2.1 us; the two-shot with 256-thread CTAs wins (fp8 14 / 16 rows: 7.22 / 7.29
        # vs 7.94 / 7.99 us).
        thr = _two_shot_rows(self.mapping.tp_size)
        if self.mapping.tp_size > 8 and not (0 < thr <= m and k_warps >= 8):
            return None
        return k_warps, n_tiles, m <= 8 and n_tiles % 2 == 0, 1

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor,
                 input_scale: torch.Tensor, residual: torch.Tensor, norm_weight: torch.Tensor, eps: float,
                 scale: Optional[torch.Tensor] = None, norm_out: bool = True,
                 tactic=None, dbg: Optional[torch.Tensor] = None,
                 row_ar: Optional[bool] = None) -> Optional[Tuple[torch.Tensor, ...]]:
        """x [M, K] BF16 (``quant``: quantized in registers with ``input_scale``) or E4M3 (``fp8``); weight
        [H, K] E4M3. Returns (norm, residual_out), (norm, quant, residual_out) or (quant, residual_out)
        like the fused all-reduce; None when the shape is not supported. ``dbg`` (debug): int64 [CTAs * 8]
        receiving every CTA's %globaltimer at its 8 phase boundaries. ``row_ar`` (default: not
        TRTLLM_Q35_GEMM_AR_MODE=col): the all-reduce + norm by row clusters after the GEMM, else by column
        slices with an all-CTA norm exchange."""
        import cuda.bindings.driver as cuda
        import cutlass

        m, k = x.shape
        hidden = weight.shape[0]
        fp8_act = "fp8" if x.dtype == torch.float8_e4m3fn else "quant"
        tac = tactic or self.tactic(m, hidden, k, fp8_act)
        if tac is None:
            return None
        k_warps, n_tiles, swap, min_blocks = tac
        nranks = self.mapping.tp_size
        if row_ar is None:
            row_ar = os.environ.get("TRTLLM_Q35_GEMM_AR_MODE", "row") != "col"
        # two-shot (row mode) where the CuTe all-reduce goes two-shot (TP16 from 12 rows)
        thr = _two_shot_rows(nranks)
        two_shot = bool(row_ar) and 0 < thr <= m
        table = self.ar._table(max(m * hidden * nranks * 2, 2 * ((m + nranks - 1) // nranks) * nranks * hidden * 2))
        fn = _compiled_gemm_ar(m, k, hidden, k_warps, n_tiles, swap, fp8_act, x.stride(0), float(eps), nranks,
                               self.ar._pdl, min_blocks, norm_out, scale is not None, dbg is not None, bool(row_ar),
                               two_shot)
        res_out = torch.empty_like(residual)
        norm = torch.empty_like(residual) if norm_out else res_out
        quant = (torch.empty(residual.shape, dtype=torch.float8_e4m3fn, device=residual.device)
                 if scale is not None else None)
        d_gate, d_quant, d_scale = self.ar._dummy(m, hidden)
        dev = x.device
        if fp8_act == "fp8":
            a16 = torch.empty((1, 8), dtype=torch.bfloat16, device=dev)
            a8 = x.view(torch.uint8)
        else:
            a16 = x
            a8 = torch.empty((1, 16), dtype=torch.uint8, device=dev)
        fn(_cute(a16), _cute(a8), _cute(weight.view(torch.uint8)), _cute(weight_scale.reshape(1).float()),
           _cute(input_scale.reshape(1).float()), _cute(residual.contiguous()), _cute(norm_weight.reshape(-1)),
           _cute(scale.reshape(1) if scale is not None else d_scale), _cute(res_out), _cute(norm),
           _cute(quant.view(torch.uint8) if quant is not None else d_quant), _cute(table), _cute(self.part),
           _cute(self.epoch), _cute(dbg if dbg is not None else self._dbg_dummy), cutlass.Int32(self.mapping.tp_rank),
           cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        if scale is None:
            return norm, res_out
        if norm_out:
            return norm, quant, res_out
        return quant, res_out


_GEMM_AR_INSTANCES = {}


def get_q35_gemm_ar(mapping) -> Optional[Q35GemmArNorm]:
    key = (tuple(mapping.tp_group), mapping.tp_rank)
    if key not in _GEMM_AR_INSTANCES:
        inst = None
        if get_q35_cute_allreduce(mapping) is not None:
            inst = Q35GemmArNorm(mapping)
        _GEMM_AR_INSTANCES[key] = inst
    return _GEMM_AR_INSTANCES[key]
