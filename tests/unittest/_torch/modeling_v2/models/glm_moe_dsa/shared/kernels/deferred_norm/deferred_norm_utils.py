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
"""Shared helpers of the deferred_norm tests and benchmarks.

* ``hop_sumsq_partials``: torch emulation of the deferred_norm / post_moe_hop's square-sum
  partials (per thread: 8 columns, sequential fp32 sum of bf16-rounded
  squares; warp butterfly 16, 8, 4, 2, 1; one value per warp), bit-exact by
  construction because every step is one IEEE fp32 operation.
* ``fp64_oracle``: exact projection of the un-rounded RMSNorm output.
* ``ProxyProducer``: a CuTe DSL stand-in for a hop kernel. It is launched
  with programmatic dependent launch, releases its dependents at entry (or
  at the end), spins for a given time and only then writes the packet
  tensors; a dependent that reads them before its dependency wait would see
  stale data.
* error statistics (normalized RMS error, bf16 ulp distance).
"""

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

H = 6144
EPS = 1e-5
HOP_CTAS = 8
HOP_THREADS = H // (8 * HOP_CTAS)  # 96
HOP_WARPS = HOP_THREADS // 32  # 3
PARTIALS = HOP_CTAS * HOP_WARPS  # 24


def hop_sumsq_partials(x: torch.Tensor) -> torch.Tensor:
    """fp32 ``[R, 24]`` partials exactly as the hop computes them from bf16 ``x``."""
    rows = x.shape[0]
    xf = x.float().view(rows, HOP_CTAS, HOP_WARPS, 32, 8)
    sq = (xf * xf).bfloat16().float()
    s = torch.zeros(rows, HOP_CTAS, HOP_WARPS, 32, dtype=torch.float32, device=x.device)
    for i in range(8):
        s = s + sq[..., i]
    lanes = torch.arange(32, device=x.device)
    for mask in (16, 8, 4, 2, 1):
        s = s + s[..., lanes ^ mask]
    return s[..., 0].reshape(rows, PARTIALS).contiguous()


def square_sum_in_order(partials: torch.Tensor) -> torch.Tensor:
    """fp32 sum of the partials in index order (the consumer's order)."""
    s = torch.zeros(partials.shape[0], dtype=torch.float32, device=partials.device)
    for i in range(partials.shape[1]):
        s = s + partials[:, i]
    return s


def make_frag_packet(x: torch.Tensor, gamma: torch.Tensor, eps: float = EPS):
    """Packet of the hop computed in torch (bit-exact ``a`` and partials).

    ``sumsq`` is a view of an 8-row buffer (``sumsq_full(packet)`` is the
    full buffer) so a proxy producer can copy whole 8-row units.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHidden,
        materialize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        frag_from_rows,
    )

    rows = x.shape[0]
    a = (x.float() * gamma.float()).bfloat16()
    ss = torch.zeros(8, PARTIALS, dtype=torch.float32, device=x.device)
    ss[:rows] = hop_sumsq_partials(x)
    return DeferredNormHidden(frag_from_rows(a), ss[:rows], x, gamma, eps, materialize)


def sumsq_full(p):
    """The 8-row partial buffer behind a test packet's ``sumsq`` view (test
    packets allocate 8 rows; inference-mode views do not keep ``_base``)."""
    return torch.as_strided(p.sumsq, (8, PARTIALS), (PARTIALS, 1))


def empty_like_packet(p):
    """Same-shaped packet with fresh (uninitialized) a_frag / partials buffers."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHidden,
    )

    ss = torch.empty(8, PARTIALS, dtype=torch.float32, device=p.sumsq.device)
    return DeferredNormHidden(
        torch.empty_like(p.a_frag), ss[: p.num_rows], p.residual, p.gamma, p.eps, p.materializer
    )


def fp64_oracle(
    x: torch.Tensor, gamma: torch.Tensor, weight: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """``y* = r* · Σ_k x_k γ_k W_jk`` in fp64 (``r*`` from exact squares)."""
    xd = x.double()
    r = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
    return (xd * gamma.double() * r) @ weight.double().t()


def norm_rms_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.double()
    return float((y.double() - ref).norm() / ref.norm().clamp_min(1e-300))


def bf16_ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def ordered(t):
        v = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(v < 0, -(v + 32768), v)

    return (ordered(a) - ordered(b)).abs()


def compare(cand: torch.Tensor, native: torch.Tensor, oracle: torch.Tensor) -> dict:
    ulp = bf16_ulp_distance(cand, native)
    return {
        "allclose_native": bool(torch.allclose(cand.float(), native.float(), rtol=0.1, atol=1e-2)),
        "e_cand": norm_rms_err(cand, oracle),
        "e_native": norm_rms_err(native, oracle),
        "max_ulp_vs_native": int(ulp.max()),
        "frac_diff_vs_native": float((ulp > 0).float().mean()),
        "max_abs_vs_native": float((cand.float() - native.float()).abs().max()),
    }


def within_budget(stats: dict) -> bool:
    return stats["allclose_native"] and stats["e_cand"] <= 1.25 * stats["e_native"] + 2.0**-12


def _asm(res, args, text, cons, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        args,
        text,
        cons,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _spin_ns(ns: Int64, sleep_ns: int = 0, *, loc=None, ip=None):
    nap = f"\t@p nanosleep.u32 {int(sleep_ns)};\n" if sleep_ns else ""
    _asm(
        None,
        [ns.ir_value()],
        "{\n\t.reg .u64 t0, t;\n\t.reg .pred p;\n\tmov.u64 t0, %globaltimer;\n\t"
        "GLM_MOE_DSA_DEFERRED_NORM_SPIN:\n\tmov.u64 t, %globaltimer;\n\tsub.u64 t, t, t0;\n\t"
        "setp.lt.u64 p, t, $0;\n" + nap + "\t@p bra GLM_MOE_DSA_DEFERRED_NORM_SPIN;\n\t}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy16(dst: Int64, src: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [dst.ir_value(), src.ir_value()],
        "{\n\t.reg .b32 a, b, c, d;\n\tld.global.v4.b32 {a, b, c, d}, [$1];\n\t"
        "st.global.v4.b32 [$0], {a, b, c, d};\n\t}",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy4(dst: Int64, src: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [dst.ir_value(), src.ir_value()],
        "{\n\t.reg .b32 a;\n\tld.global.b32 a, [$1];\n\tst.global.b32 [$0], a;\n\t}",
        "l,l",
        loc=loc,
        ip=ip,
    )


class _ProxyKernel:
    def __init__(self, trigger_at_start: bool, sleep_ns: int = 0, carveout: int = -1):
        self.trigger_at_start = trigger_at_start
        self.sleep_ns = sleep_ns
        self.carveout = carveout

    @cute.jit
    def __call__(
        self,
        act_src: Int64,
        act_dst: Int64,
        ss_src: Int64,
        ss_dst: Int64,
        delay_ns: Int64,
        anchor: cute.Tensor,
        rows: Int32,
        stream: CUstream,
    ):
        if cutlass.const_expr(self.carveout >= 0):
            self.kernel(act_src, act_dst, ss_src, ss_dst, delay_ns, anchor).launch(
                grid=[HOP_CTAS, rows, 1],
                block=[HOP_THREADS, 1, 1],
                stream=stream,
                use_pdl=True,
                preferred_smem_carveout=self.carveout,
            )
        else:
            self.kernel(act_src, act_dst, ss_src, ss_dst, delay_ns, anchor).launch(
                grid=[HOP_CTAS, rows, 1], block=[HOP_THREADS, 1, 1], stream=stream, use_pdl=True
            )

    @cute.kernel
    def kernel(
        self,
        act_src: Int64,
        act_dst: Int64,
        ss_src: Int64,
        ss_dst: Int64,
        delay_ns: Int64,
        anchor: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        cta, row, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.trigger_at_start):
            cute.arch.griddepcontrol_launch_dependents()
        _spin_ns(delay_ns, self.sleep_ns)
        off = Int64(row) * Int64(H * 2) + Int64((cta * Int32(HOP_THREADS) + tid) * Int32(16))
        _copy16(act_dst + off, act_src + off)
        if tid < Int32(HOP_WARPS):
            o4 = Int64(row) * Int64(PARTIALS * 4) + Int64((cta * Int32(HOP_WARPS) + tid) * 4)
            _copy4(ss_dst + o4, ss_src + o4)
        if cutlass.const_expr(not self.trigger_at_start):
            cute.arch.griddepcontrol_launch_dependents()


_PROXY = {}


def proxy_producer(
    act_src, act_dst, ss_src, ss_dst, delay_ns: int, trigger_at_start: bool = True, rows: int = None
):
    """Launch the proxy producer on the current stream (PDL).

    Copies ``rows`` x 12,288 bytes of ``act_src`` and ``rows`` x 24 fp32 of
    ``ss_src`` (``rows`` defaults to ``act_src.shape[0]``; an ``a_frag``
    buffer is 8 rows' worth).
    """
    import os

    sleep_ns = int(os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_PROXY_SLEEP_NS", "0"))
    carveout = int(os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_PROXY_CARVEOUT", "-1"))
    key = (trigger_at_start, sleep_ns, carveout)
    if key not in _PROXY:
        from cutlass.cute import runtime as cute_runtime

        z = Int64(0)
        anchor = cute_runtime.make_fake_compact_tensor(
            Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
        )
        _PROXY[key] = cute.compile(
            _ProxyKernel(trigger_at_start, sleep_ns, carveout),
            z,
            z,
            z,
            z,
            z,
            anchor,
            Int32(1),
            stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _PROXY[key](
        act_src.data_ptr(),
        act_dst.data_ptr(),
        ss_src.data_ptr(),
        ss_dst.data_ptr(),
        int(delay_ns),
        ss_dst.view(torch.int32).view(-1),
        rows if rows is not None else act_src.shape[0],
    )
