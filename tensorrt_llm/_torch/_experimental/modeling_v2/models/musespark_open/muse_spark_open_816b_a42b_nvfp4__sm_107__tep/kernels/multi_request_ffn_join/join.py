# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""Lever ``multi_request_ffn_join``: the post-FFN join of 33..768-row capture-layer calls as a multi-row
cluster kernel that also stores the DFlash capture.

Above 32 rows the built-in post-FFN join (``MuseSparkOpenGatedResidual._add_after_rmsnorm_and_norm``
with the next layer's pre-QKV norm) runs the Triton ``_gated_residual_after_rmsnorm_kernel``: one
512-thread program a row, which re-reads the four 8192-wide parameter vectors (the ``1 + w`` gain,
alpha, beta and the next norm's weight: 96 KB) for every row and, at the largest row counts, runs in
more than one wave of two programs per SM on the device's SM count. On the DFlash capture layers the
built-in layer then casts the FP32 residual to BF16 and copies it into the capture buffer
(``maybe_capture_hidden_states``): two more launches on the serial chain before the next QKV GEMM.

This kernel computes the same function with the same arithmetic, operation for operation, as the
bitwise cluster join of ``ops/comm/projection_publish`` (``JOIN_CLUSTER``: the PTX Triton emits for
this kernel at BLOCK_H 8192 and 16 warps -- the BF16 gain product, the fma / add in-thread tree, the
xor-16..1 warp butterfly, the 16 warp sums reduced in the order of Triton's cross-warp xor-8..1
butterfly, ``div.full.f32`` by 8192, ``rsqrt.approx.ftz.f32``, ``fma(residual, alpha, normed * beta)``
and both next-variance trees), on a dense ``[rows, 8192]`` branch instead of a transport:

* a row is still 512 threads, each owning elements ``[8T, 8T + 8)`` and ``[4096 + 8T, 4096 + 8T + 8)``,
  split over a cluster of ``CL`` CTAs whose warp sums meet through DSMEM (``st.async`` + mbarrier);
* a cluster joins ``rows_per_cluster`` consecutive rows in a loop, holding the parameter vectors in
  registers and loading the next row's residual and branch before the current row's reductions; by
  default the grid is sized to one resident wave (``rows_per_cluster`` = ceil(rows / the device's
  co-resident clusters));
* on a capture layer (``capture_fold``'s ``_capture_fold`` view) it also stores
  ``cvt.rn.bf16.f32(updated)`` into the DFlash capture, the bytes the built-in cast writes.

Every per-row value is computed by the same instructions in the same order as one Triton program,
so every output (the FP32 residual, the BF16 next input and the capture) is expected to be the
Triton join's bits; the GPU parity test checks that bitwise. The loop changes only which CTA
computes a row and when.

The barriers: each CTA has two mbarriers (one per reduction), each expecting the 16 warp sums of the
row (64 bytes) per phase; row ``i`` of a cluster uses phase ``i`` (parity ``i & 1``) of both, and
thread 0 re-arms a barrier right after its own wait on the phase. A CTA sends its warp sums of row
``i + 1``'s first reduction only after its second reduction of row ``i`` completed, which needs every
warp of every CTA of the cluster to have passed its first-reduction read of row ``i``; the same chain
holds for the second reduction. So no phase's sums land before the previous phase completed on the
receiver, and no warp-sum slot is overwritten before every warp read it.

Launched with PDL when the model launches with PDL (``dispatch.LAUNCH_PDL``): every CTA waits for
its prerequisite grid (``griddepcontrol.wait``) before reading anything, and releases its
dependents at exit (``trigger`` ``exit``, the default) or right after the wait (``entry``).
Compiled with NVRTC for sm_107 when installed (and on the first eager call for another variant); a
call inside a CUDA-graph capture before its variant compiled keeps the built-in join (logged once).

Which calls it takes (``capture_only``, on by default and as the lever installs it): only the joins
that carry a DFlash capture view (``capture_fold``'s ``_capture_fold``: the capture layers' joins of a
33..768-row decode step, five a step for the served drafter). There the kernel replaces the built-in
join and the built-in capture's cast and 2-D copy; every other join keeps the built-in Triton join,
which is as fast as this kernel behind the unpool GEMM (and faster at 768 rows). A capture call the
kernel declines (operands, capture view, an uncompiled variant in a capture) keeps the built-in join
and the built-in capture. With ``capture_only`` off the kernel takes every served call, with or
without a capture.

Local only: no collective, so a rank may take or leave this path on its own without affecting its
peers.
"""

from __future__ import annotations

import ctypes
import math
import threading
from typing import Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

#: Width of the residual stream.
HIDDEN = 8192
#: The decode rows the lever serves by default: five to 96 requests at K = 7.
MIN_ROWS, MAX_ROWS = 33, 768
#: The cluster sizes the kernel takes (512 / CL threads a CTA).
CLUSTERS = (1, 2, 4, 8)
#: The default cluster size: 128 threads a CTA (``join_cluster``'s size above 128 rows).
DEFAULT_CLUSTER = 4
#: When the kernel releases its dependent: ``exit`` (after its last store) or ``entry``.
TRIGGERS = ("exit", "entry")
#: The operands' alignment the vector loads and stores need.
_ALIGN = 16

_SRC = r"""
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
#ifndef NORM_FP32
#define NORM_FP32 0
#endif
#ifndef HAS_GAIN
#define HAS_GAIN 1
#endif
#ifndef CL
#define CL 4  // CTAs a row; each runs 512 / CL of the Triton join's 512 threads
#endif
#define NT (512 / CL)
#define NW (NT / 32)
__device__ __forceinline__ float bf2f(u32 h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ u32 f2bf(float f) { unsigned short r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r; }
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ float div_full(float a, float b) { float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float rsqrt_ftz(float a) { float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ void unpack8(uint4 q, float* v) {
  v[0] = bf2f(q.x & 0xffffu); v[1] = bf2f(q.x >> 16); v[2] = bf2f(q.y & 0xffffu); v[3] = bf2f(q.y >> 16);
  v[4] = bf2f(q.z & 0xffffu); v[5] = bf2f(q.z >> 16); v[6] = bf2f(q.w & 0xffffu); v[7] = bf2f(q.w >> 16); }
__device__ __forceinline__ uint4 pack8(const float* v) {
  return make_uint4(f2bf(v[0]) | (f2bf(v[1]) << 16), f2bf(v[2]) | (f2bf(v[3]) << 16),
                    f2bf(v[4]) | (f2bf(v[5]) << 16), f2bf(v[6]) | (f2bf(v[7]) << 16)); }
__device__ __forceinline__ uint4 ld16(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void ld8f(const float* p, float* v) {
  *reinterpret_cast<float4*>(v) = *reinterpret_cast<const float4*>(p);
  *reinterpret_cast<float4*>(v + 4) = *reinterpret_cast<const float4*>(p + 4); }
__device__ __forceinline__ void st8f(float* p, const float* v) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]); }
// Triton's in-thread order (fma tree: the first variance, and the BF16-first next variance)
__device__ __forceinline__ float thread_sum_fma(const float* x, const float* y) {
  const float ax = __fmaf_rn(x[0], x[0], __fmul_rn(x[2], x[2])), ay = __fmaf_rn(x[1], x[1], __fmul_rn(x[3], x[3]));
  const float bx = __fmaf_rn(x[4], x[4], __fmul_rn(x[6], x[6])), by = __fmaf_rn(x[5], x[5], __fmul_rn(x[7], x[7]));
  const float cx = __fmaf_rn(y[0], y[0], __fmul_rn(y[2], y[2])), cy = __fmaf_rn(y[1], y[1], __fmul_rn(y[3], y[3]));
  const float dx = __fmaf_rn(y[4], y[4], __fmul_rn(y[6], y[6])), dy = __fmaf_rn(y[5], y[5], __fmul_rn(y[7], y[7]));
  return __fadd_rn(__fadd_rn(__fadd_rn(ax, bx), __fadd_rn(cx, dx)), __fadd_rn(__fadd_rn(ay, by), __fadd_rn(cy, dy))); }
// the FP32-residual next variance: squares then an add tree, no fma (the PTX Triton emits there)
__device__ __forceinline__ float thread_sum_mul(const float* x, const float* y) {
  float q[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) { q[i] = __fmul_rn(x[i], x[i]); q[8 + i] = __fmul_rn(y[i], y[i]); }
  const float ax = __fadd_rn(q[0], q[2]), ay = __fadd_rn(q[1], q[3]), bx = __fadd_rn(q[4], q[6]), by = __fadd_rn(q[5], q[7]);
  const float cx = __fadd_rn(q[8], q[10]), cy = __fadd_rn(q[9], q[11]), dx = __fadd_rn(q[12], q[14]), dy = __fadd_rn(q[13], q[15]);
  return __fadd_rn(__fadd_rn(__fadd_rn(ax, bx), __fadd_rn(cx, dx)), __fadd_rn(__fadd_rn(ay, by), __fadd_rn(cy, dy))); }
__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }
// the row total: warp butterfly, the 16 warp sums to every CTA of the row's cluster (st.async + mbarrier
// phase ``parity``), then the tree every lane of Triton's cross-warp xor-8/4/2/1 butterfly computes.
// Thread 0 re-arms the barrier for its next phase once its own wait returned.
__device__ __forceinline__ float row_sum(float s, float* wsum, u64* bar, int rank, u32 parity) {
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, off));
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane < CL) {  // lane q sends this warp's sum to CTA q
    u32 raddr, rbar;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(raddr) : "r"(smem_u32(wsum + NW * rank + warp)), "r"(lane));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbar) : "r"(smem_u32(bar)), "r"(lane));
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                 :: "r"(raddr), "r"(__float_as_uint(s)), "r"(rbar) : "memory");
  }
  asm volatile("{\n .reg .pred p;\n W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra W;\n}"
               :: "r"(smem_u32(bar)), "r"(parity) : "memory");
  float W[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = wsum[i];
  if (threadIdx.x == 0)
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 64;" :: "r"(smem_u32(bar)) : "memory");
  const float a0 = __fadd_rn(__fadd_rn(W[0], W[8]), __fadd_rn(W[4], W[12]));
  const float a2 = __fadd_rn(__fadd_rn(W[2], W[10]), __fadd_rn(W[6], W[14]));
  const float a1 = __fadd_rn(__fadd_rn(W[1], W[9]), __fadd_rn(W[5], W[13]));
  const float a3 = __fadd_rn(__fadd_rn(W[3], W[11]), __fadd_rn(W[7], W[15]));
  return __fadd_rn(__fadd_rn(a0, a2), __fadd_rn(a1, a3)); }

extern "C" __global__ void __launch_bounds__(NT)
mr_ffn_join(float* out, const float* residual, const u16* branch, const u16* gain, const float* alpha, const float* beta,
            float eps, u16* next_out, const u16* next_w, float next_eps, u16* capture, long long capture_stride,
            int rows, int rows_per_cluster, int trigger) {
  __shared__ __align__(16) float wsum0[16];
  __shared__ __align__(16) float wsum1[16];
  __shared__ __align__(8) u64 bars[2];
  const int t = threadIdx.x, rank = blockIdx.x % CL, group = blockIdx.x / CL;
  const int T = rank * NT + t;
  const int c0 = 8 * T, c1 = 4096 + 8 * T;
  const int row0 = group * rows_per_cluster;
  const int n = min(rows_per_cluster, rows - row0);  // the same on every CTA of the cluster; >= 1
  if (t == 0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(bars + i)) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 64;" :: "r"(smem_u32(bars + i)) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if (trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // the parameter vectors, once for the cluster's rows
  const uint4 g0q = ld16(gain + c0), g1q = ld16(gain + c1), w0q = ld16(next_w + c0), w1q = ld16(next_w + c1);
  float a0[8], a1[8], b0[8], b1[8];
  ld8f(alpha + c0, a0); ld8f(alpha + c1, a1); ld8f(beta + c0, b0); ld8f(beta + c1, b1);
  // the first row's residual and branch; each iteration loads the next row's before its reductions
  float nr0[8], nr1[8];
  long long rb = (long long)row0 * 8192;
  ld8f(residual + rb + c0, nr0); ld8f(residual + rb + c1, nr1);
  uint4 nq0 = ld16(branch + rb + c0), nq1 = ld16(branch + rb + c1);
  for (int i = 0; i < n; ++i) {
    rb = (long long)(row0 + i) * 8192;
    float r0[8], r1[8], x[8], y[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { r0[k] = nr0[k]; r1[k] = nr1[k]; }
    unpack8(nq0, x); unpack8(nq1, y);
    if (i + 1 < n) {
      const long long nb = rb + 8192;
      ld8f(residual + nb + c0, nr0); ld8f(residual + nb + c1, nr1);
      nq0 = ld16(branch + nb + c0); nq1 = ld16(branch + nb + c1);
    }
    const u32 parity = (u32)(i & 1);
#if HAS_GAIN
    {
      float g0[8], g1[8];
      unpack8(g0q, g0); unpack8(g1q, g1);
#pragma unroll
      for (int k = 0; k < 8; ++k) { x[k] = rbf(__fmul_rn(x[k], g0[k])); y[k] = rbf(__fmul_rn(y[k], g1[k])); }
    }
#endif
    const float inv = rsqrt_ftz(__fadd_rn(div_full(row_sum(thread_sum_fma(x, y), wsum0, bars, rank, parity), 8192.0f), eps));
    float u0[8], u1[8], n0[8], n1[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      u0[k] = __fmaf_rn(r0[k], a0[k], __fmul_rn(rbf(__fmul_rn(inv, x[k])), b0[k]));
      u1[k] = __fmaf_rn(r1[k], a1[k], __fmul_rn(rbf(__fmul_rn(inv, y[k])), b1[k]));
#if NORM_FP32
      n0[k] = u0[k]; n1[k] = u1[k];
#else
      n0[k] = rbf(u0[k]); n1[k] = rbf(u1[k]);
#endif
    }
    st8f(out + rb + c0, u0); st8f(out + rb + c1, u1);
    if (capture != nullptr) {  // the DFlash capture: the built-in cast's round-to-nearest-even BF16
      u16* cp = capture + (long long)(row0 + i) * capture_stride;
      *reinterpret_cast<uint4*>(cp + c0) = pack8(u0);
      *reinterpret_cast<uint4*>(cp + c1) = pack8(u1);
    }
#if NORM_FP32
    const float sum2 = thread_sum_mul(n0, n1);
#else
    const float sum2 = thread_sum_fma(n0, n1);
#endif
    const float inv2 = rsqrt_ftz(__fadd_rn(div_full(row_sum(sum2, wsum1, bars + 1, rank, parity), 8192.0f), next_eps));
    float w0[8], w1[8];
    unpack8(w0q, w0); unpack8(w1q, w1);
    u32 p0[4], p1[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      p0[k] = f2bf(__fmul_rn(__fmul_rn(inv2, n0[2 * k]), __fadd_rn(w0[2 * k], 1.0f)))
            | (f2bf(__fmul_rn(__fmul_rn(inv2, n0[2 * k + 1]), __fadd_rn(w0[2 * k + 1], 1.0f))) << 16);
      p1[k] = f2bf(__fmul_rn(__fmul_rn(inv2, n1[2 * k]), __fadd_rn(w1[2 * k], 1.0f)))
            | (f2bf(__fmul_rn(__fmul_rn(inv2, n1[2 * k + 1]), __fadd_rn(w1[2 * k + 1], 1.0f))) << 16);
    }
    u16* no = next_out + rb;
    *reinterpret_cast<uint4*>(no + c0) = make_uint4(p0[0], p0[1], p0[2], p0[3]);
    *reinterpret_cast<uint4*>(no + c1) = make_uint4(p1[0], p1[1], p1[2], p1[3]);
  }
  if (!trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
"""


def rows_per_cluster(rows: int, max_clusters: int, fixed: int = 0) -> int:
    """The rows each cluster joins: ``fixed`` when set, else the fewest that fit ``rows`` into
    ``max_clusters`` co-resident clusters (one wave)."""
    if rows < 1:
        raise ValueError(f"rows must be positive, got {rows}")
    if fixed:
        return int(fixed)
    return max(1, math.ceil(rows / max(1, int(max_clusters))))


def grid_clusters(rows: int, per_cluster: int) -> int:
    """The clusters of a call: every row in exactly one cluster's range ``[g * per, (g + 1) * per)``."""
    return math.ceil(rows / per_cluster)


def capture_fits(capture: torch.Tensor, rows: int, device: torch.device) -> bool:
    """Whether ``capture`` is a view the kernel's capture stores address correctly."""
    return (
        capture.dtype == torch.bfloat16
        and capture.dim() == 2
        and tuple(capture.shape) == (rows, HIDDEN)
        and capture.stride(1) == 1
        and capture.stride(0) >= HIDDEN
        and capture.stride(0) % 8 == 0
        and capture.data_ptr() % _ALIGN == 0
        and capture.device == device
    )


def _span(t: torch.Tensor) -> Tuple[int, int]:
    """``[first, last)`` byte addresses ``t`` covers."""
    if t.numel() == 0:
        return t.data_ptr(), t.data_ptr()
    last = sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    return t.data_ptr(), t.data_ptr() + (last + 1) * t.element_size()


def overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the byte spans of ``a`` and ``b`` intersect (a conservative aliasing test)."""
    a0, a1 = _span(a)
    b0, b1 = _span(b)
    return a0 < b1 and b0 < a1


def operand_refusal(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gain: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    weight: torch.Tensor,
) -> Optional[str]:
    """Why the kernel cannot join these operands, or ``None`` when it can.

    The kernel is the built-in Triton join's function only where the built-in would run that
    kernel (``_add_after_rmsnorm_and_norm``'s fused predicate) on these shapes and dtypes, and its
    vector accesses need 16-byte alignment.
    """
    if residual.dim() != 2 or residual.shape[1] != HIDDEN:
        return f"residual {tuple(residual.shape)} is not [rows, {HIDDEN}]"
    if not residual.is_cuda or residual.dtype != torch.float32 or not residual.is_contiguous():
        return (
            f"residual {residual.dtype} on {residual.device} is not a contiguous CUDA FP32 tensor"
        )
    if (
        branch.dtype != torch.bfloat16
        or tuple(branch.shape) != tuple(residual.shape)
        or not branch.is_contiguous()
        or branch.device != residual.device
    ):
        return f"branch {tuple(branch.shape)} {branch.dtype} is not a contiguous BF16 residual-shaped tensor"
    vectors = {
        "alpha": (alpha, torch.float32),
        "beta": (beta, torch.float32),
        "weight": (weight, torch.bfloat16),
    }
    if gain is not None:
        vectors["gain"] = (gain, torch.bfloat16)
    for name, (t, dtype) in vectors.items():
        if (
            not isinstance(t, torch.Tensor)
            or t.dtype != dtype
            or tuple(t.shape) != (HIDDEN,)
            or not t.is_contiguous()
            or t.device != residual.device
        ):
            return f"{name} is not a contiguous [{HIDDEN}] {dtype} tensor on {residual.device}"
    for name, t in (
        ("residual", residual),
        ("branch", branch),
        *((k, v[0]) for k, v in vectors.items()),
    ):
        if t.data_ptr() % _ALIGN:
            return f"{name} is not {_ALIGN}-byte aligned"
    return None


def _check(r):
    if int(r[0]) != 0:
        raise RuntimeError(f"multi_request_ffn_join: CUDA/NVRTC error {r!r}")
    return r[1] if len(r) == 2 else r[1:]


class MultiRequestFfnJoin:
    """The post-FFN join of ``min_rows..max_rows``-row calls, shared by every layer's join.

    Args:
        min_rows: the fewest rows of a call the kernel takes (``MIN_ROWS``).
        max_rows: the most (``MAX_ROWS``).
        cluster: CTAs a row (``CLUSTERS``).
        rows_per_cluster: rows a cluster joins; 0 sizes the grid to one resident wave.
        trigger: ``exit`` or ``entry`` (``TRIGGERS``).
        pdl: launch with programmatic dependent launch.
        capture_only: take only the calls whose join carries a DFlash capture view.
    """

    def __init__(
        self,
        *,
        min_rows: int = MIN_ROWS,
        max_rows: int = MAX_ROWS,
        cluster: int = DEFAULT_CLUSTER,
        rows_per_cluster: int = 0,
        trigger: str = "exit",
        pdl: bool = True,
        capture_only: bool = False,
    ) -> None:
        if not 1 <= min_rows <= max_rows:
            raise ValueError(f"multi_request_ffn_join: rows {min_rows}..{max_rows} is not a range")
        if cluster not in CLUSTERS:
            raise ValueError(f"multi_request_ffn_join: cluster {cluster} is not one of {CLUSTERS}")
        if rows_per_cluster < 0:
            raise ValueError(f"multi_request_ffn_join: rows_per_cluster {rows_per_cluster} < 0")
        if trigger not in TRIGGERS:
            raise ValueError(
                f"multi_request_ffn_join: trigger {trigger!r} is not one of {TRIGGERS}"
            )
        self.min_rows = int(min_rows)
        self.max_rows = int(max_rows)
        self.cluster = int(cluster)
        self.rows_per_cluster = int(rows_per_cluster)
        self.trigger = trigger
        self.pdl = bool(pdl)
        self.capture_only = bool(capture_only)
        #: (device index, has gain, FP32 next-norm residual) -> (function, module, max clusters).
        self._functions: Dict[Tuple[int, bool, bool], Tuple[object, object, int]] = {}
        self._lock = threading.Lock()
        self._noted: set = set()
        #: Calls the kernel joined, and of those the ones that stored the capture (host-side counts;
        #: under a CUDA graph, per capture).
        self.calls = 0
        self.captures = 0

    # -- host-side choice ---------------------------------------------------------------------

    def serves(self, join, rows: int) -> bool:
        """Whether a call of ``rows`` rows on ``join`` reaches this kernel when its operands fit.

        Host-side reads only (the row count, the published transport's arming): under a CUDA graph
        a per-capture constant. ``capture_fold`` asks this before it hands the join a capture view.
        """
        if not self.min_rows <= rows <= self.max_rows:
            return False
        published = getattr(join, "_published", None)
        if published is not None and getattr(published, "is_armed", False):
            # The published transport's join takes the call first (and stores no capture).
            return False
        return bool(getattr(join, "_fused_enabled", False)) and not torch.compiler.is_compiling()

    def _note(self, reason: str) -> None:
        if reason not in self._noted:
            self._noted.add(reason)
            logger.warning(
                f"[musespark_open] multi_request_ffn_join: built-in join kept ({reason})"
            )

    # -- compilation --------------------------------------------------------------------------

    def prepare(self, device: torch.device, has_gain: bool, norm_fp32: bool) -> None:
        """Compile the variant a served join launches, before any CUDA-graph capture.

        Raises:
            RuntimeError: when the device is not sm_107 or NVRTC / the driver refuses the kernel.
        """
        self._function(device, has_gain, norm_fp32, allow_capture=True)

    def _function(
        self, device: torch.device, has_gain: bool, norm_fp32: bool, allow_capture: bool = False
    ):
        index = device.index if device.index is not None else torch.cuda.current_device()
        key = (int(index), bool(has_gain), bool(norm_fp32))
        found = self._functions.get(key)
        if found is not None:
            return found
        if not allow_capture and torch.cuda.is_current_stream_capturing():
            self._note(
                f"first call of variant gain={has_gain} norm_fp32={norm_fp32} on device {index} "
                "inside a CUDA-graph capture; the kernel compiles on an eager call"
            )
            return None
        with self._lock:
            found = self._functions.get(key)
            if found is None:
                found = self._compile(int(index), bool(has_gain), bool(norm_fp32))
                self._functions[key] = found
        return found

    def _compile(self, index: int, has_gain: bool, norm_fp32: bool):
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc

        major, minor = torch.cuda.get_device_capability(index)
        if (major, minor) != (10, 7):
            raise RuntimeError(
                f"multi_request_ffn_join: built for sm_107 only, device {index} is sm_{major}{minor}"
            )
        cl = self.cluster
        prog = _check(nvrtc.nvrtcCreateProgram(_SRC.encode(), b"mr_ffn_join.cu", 0, [], []))
        opts = [
            f"--gpu-architecture=sm_{major}{minor}".encode(),
            b"-std=c++17",
            b"-default-device",
            f"-DHAS_GAIN={int(has_gain)}".encode(),
            f"-DNORM_FP32={int(norm_fp32)}".encode(),
            f"-DCL={cl}".encode(),
        ]
        r = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if int(r[0]) != 0:
            size = _check(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(
                "multi_request_ffn_join: NVRTC compile failed:\n" + log.decode(errors="replace")
            )
        size = _check(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        _check(nvrtc.nvrtcGetCUBIN(prog, cubin))
        with torch.cuda.device(index):
            # The driver calls below need the device's primary context current on this thread; a
            # process that has not touched the device yet (an install before any tensor) has none.
            torch.zeros(1, device=torch.device("cuda", index))
            module = _check(cu.cuModuleLoadData(cubin))
            fn = _check(cu.cuModuleGetFunction(module, b"mr_ffn_join"))
            regs = int(
                _check(
                    cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, fn)
                )
            )
            local = int(
                _check(
                    cu.cuFuncGetAttribute(
                        cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn
                    )
                )
            )
            clusters = self._max_clusters(fn, index)
        logger.info(
            f"[musespark_open] multi_request_ffn_join: compiled for sm_{major}{minor} on device {index} "
            f"(cluster {cl} x {512 // cl} threads, gain {has_gain}, FP32 next-norm residual {norm_fp32}, "
            f"regs {regs}, local {local} B, {clusters} co-resident clusters)"
        )
        if local:
            logger.warning(
                f"[musespark_open] multi_request_ffn_join: the kernel spills {local} B a thread to local memory"
            )
        return fn, module, clusters

    def _max_clusters(self, fn, index: int) -> int:
        """The clusters of this kernel the device holds at once (the one-wave grid)."""
        from cuda.bindings import driver as cu

        cl = self.cluster
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = cl, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 512 // cl, 1, 1
        cfg.sharedMemBytes = 0
        attr = cu.CUlaunchAttribute()
        attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = cl, 1, 1
        cfg.attrs = [attr]
        cfg.numAttrs = 1
        err, clusters = cu.cuOccupancyMaxActiveClusters(fn, cfg)
        if int(err) == 0 and int(clusters) > 0:
            return int(clusters)
        # The block-level bound when the cluster query is unavailable.
        per_sm = int(_check(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, 512 // cl, 0)))
        sms = torch.cuda.get_device_properties(index).multi_processor_count
        return max(1, per_sm * sms // cl)

    # -- the call -----------------------------------------------------------------------------

    def join(self, owner, residual, branch, eps, next_norm, gain):
        """``owner``'s post-FFN join with ``next_norm`` as this kernel: ``(updated, next_input)``,
        or ``None`` to keep the built-in join.

        Stores ``owner._capture_fold`` (when set and it fits) and then sets
        ``owner._capture_stored``. With ``capture_only`` a call without a capture view, or whose
        view the kernel cannot store, keeps the built-in join (and then the built-in capture).
        """
        capture = getattr(owner, "_capture_fold", None)
        if self.capture_only and capture is None:
            return None
        rows = residual.shape[0] if residual.dim() == 2 else 0
        if not self.serves(owner, rows):
            return None
        if not (
            getattr(next_norm, "use_gemma", False)
            and isinstance(getattr(next_norm, "weight", None), torch.Tensor)
        ):
            self._note("the next norm is not a Gemma RMSNorm with a weight tensor")
            return None
        alpha, beta = owner.coefficients()
        weight = next_norm.weight
        reason = operand_refusal(residual, branch, gain, alpha, beta, weight)
        if reason is not None:
            self._note(reason)
            return None
        function = self._function(residual.device, gain is not None, bool(owner._preqkv_norm_fp32))
        if function is None:
            return None
        fn, _, max_clusters = function
        if capture is not None:
            refusal = None
            if not capture_fits(capture, rows, residual.device):
                refusal = (
                    f"capture view {tuple(capture.shape)} stride {tuple(capture.stride())} "
                    f"{capture.dtype} does not fit the kernel's store"
                )
            elif overlaps(capture, residual) or overlaps(capture, branch):
                refusal = "the capture view overlaps the residual or the branch"
            if refusal is not None:
                self._note(f"{refusal}; the built-in capture follows")
                if self.capture_only:
                    return None
                capture = None
        updated = torch.empty_like(residual)
        next_input = torch.empty_like(branch)
        per = rows_per_cluster(rows, max_clusters, self.rows_per_cluster)
        self._launch(
            fn,
            grid_clusters(rows, per),
            per,
            rows,
            updated,
            residual,
            branch,
            gain if gain is not None else weight,  # unread without the gain
            alpha,
            beta,
            float(eps),
            next_input,
            weight,
            float(next_norm.variance_epsilon),
            capture,
        )
        self.calls += 1
        if capture is not None:
            owner._capture_stored = True
            self.captures += 1
        return updated, next_input

    def _launch(
        self,
        fn,
        clusters,
        per,
        rows,
        updated,
        residual,
        branch,
        gain,
        alpha,
        beta,
        eps,
        next_input,
        weight,
        next_eps,
        capture,
    ) -> None:
        from cuda.bindings import driver as cu

        cl = self.cluster
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = cl * clusters, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = 512 // cl, 1, 1
        cfg.sharedMemBytes = 0
        cfg.hStream = cu.CUstream(torch.cuda.current_stream(residual.device).cuda_stream)
        a1 = cu.CUlaunchAttribute()
        a1.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        a1.value.clusterDim.x, a1.value.clusterDim.y, a1.value.clusterDim.z = cl, 1, 1
        attrs = [a1]
        if self.pdl:
            a2 = cu.CUlaunchAttribute()
            a2.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            a2.value.programmaticStreamSerializationAllowed = 1
            attrs.append(a2)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        vp, ll, fl, ii = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float, ctypes.c_int
        args = (
            (
                updated.data_ptr(),
                residual.data_ptr(),
                branch.data_ptr(),
                gain.data_ptr(),
                alpha.data_ptr(),
                beta.data_ptr(),
                eps,
                next_input.data_ptr(),
                weight.data_ptr(),
                next_eps,
                capture.data_ptr() if capture is not None else 0,
                int(capture.stride(0)) if capture is not None else 0,
                int(rows),
                int(per),
                int(self.trigger == "entry"),
            ),
            (vp, vp, vp, vp, vp, vp, fl, vp, vp, fl, vp, ll, ii, ii, ii),
        )
        _check((cu.cuLaunchKernelEx(cfg, fn, args, 0)[0], None))


def install(
    model,
    *,
    min_rows: int = MIN_ROWS,
    max_rows: int = MAX_ROWS,
    cluster: int = DEFAULT_CLUSTER,
    rows_per_cluster: int = 0,
    trigger: str = "exit",
    pdl: bool = True,
    capture_only: bool = True,
    join_class=None,
) -> MultiRequestFfnJoin:
    """Hand every decoder layer's post-FFN join (``join_class``: the projection-gather join, whose
    built-in fallback calls this kernel) the shared op, and compile the served variant.

    Run after ``projection_gathers.install``. The served variant is the post-FFN join's: with the
    gain, and the pre-QKV norm's FP32-residual ordering the joins were constructed with. With
    ``capture_only`` (the default) the op takes only the joins ``capture_fold`` hands a capture view.

    Raises:
        ValueError: when a layer's post-FFN join is not ``join_class``, or the joins disagree on the
            FP32-residual ordering.
        RuntimeError: when the kernel cannot be compiled on the model's device.
    """
    layers = list(model.layers)
    if join_class is not None:
        for layer in layers:
            if not isinstance(layer.add_ffn, join_class):
                raise ValueError(
                    f"[musespark_open] multi_request_ffn_join hooks the projection-gather post-FFN join; layer "
                    f"{layer.layer_idx}'s is {type(layer.add_ffn).__name__}"
                )
    orders = {bool(layer.add_ffn._preqkv_norm_fp32) for layer in layers}
    if len(orders) != 1:
        raise ValueError(
            "[musespark_open] multi_request_ffn_join: the layers' post-FFN joins disagree on norm_fp32"
        )
    op = MultiRequestFfnJoin(
        min_rows=min_rows,
        max_rows=max_rows,
        cluster=cluster,
        rows_per_cluster=rows_per_cluster,
        trigger=trigger,
        pdl=pdl,
        capture_only=capture_only,
    )
    device = layers[0].add_ffn.gate.device
    if device.type == "cuda":
        op.prepare(device, True, orders.pop())
    for layer in layers:
        layer.add_ffn._multi_request_join = op
    scope = (
        "capture-layer calls (the other joins keep the built-in join)"
        if capture_only
        else "calls (on capture_fold layers it also writes the DFlash capture)"
    )
    logger.info(
        f"[musespark_open] multi_request_ffn_join: the post-FFN join of {min_rows}..{max_rows}-row {scope} "
        f"runs as the multi-row cluster kernel that stores the DFlash capture (cluster {cluster}, rows per "
        f"cluster {rows_per_cluster or 'one wave'}, trigger {trigger}, PDL {pdl}, capture_only {capture_only})"
    )
    return op
