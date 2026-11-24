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
#
# Adapted from vllm-project/vllm #61867 @642c18d (vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/mla/query.py):
# the structure only (one cluster per head, each CTA owns a slice of the head's Q_B rows, the q_nope rows are
# exchanged through distributed shared memory, static weights are read before the PDL wait). The kernel below is
# CUDA C++ (NVRTC) written for this tree's FP8 128x128-scaled Q_B and FP8 query; it shares no code with the PR.
"""Fused MLA query of the Kimi K3 decode (``TRTLLM_K3_MLA_FUSED_Q``), SM100/103, one kernel.

Replaces the q_a RMSNorm, the 1x128 FP8 quantization, the ``q_b_proj`` DeepGEMM and the FP8 q_absorb kernels of the
decode chain by one cluster launch. For ``M <= 32`` rows it takes the fused projection output ``q_a`` (BF16
``[M, 1536]``, any row stride) and writes the FP8 query ``[M, 12, 576]`` of the CuTe DSL decode: columns 0..511 of head
``h`` are ``e4m3(sat(q_nope[h] @ W_UK[h]^T))`` (FP32 accumulation, one rounding from FP32 like the Triton kernel it
replaces) and columns 512..575 ``e4m3(sat(q_pe[h]))``.

* RMSNorm: the arithmetic of FlashInfer's CuTe RMSNorm (as reproduced for H <= 3072 by
  ``cute_dsl_kernels/rubin/k3_mla_fused_qkv.py``): per-lane FP32 squares in value order (lane ``l`` holds columns
  ``8 l + 256 b + j``), a 5-step butterfly, ``rsqrt.approx(sum / H + eps)``, ``(x * rstd) * w`` rounded to BF16.
* Quantization: ``fp8_quantize_1x128_packed_kernel`` (UE8M0 scale per row and 128 K values, RNE, saturating).
* Q_B: ``mma.sync.m16n8k32`` E4M3 per 128-K block, then the exact power-of-two promotion by the weight block scale
  times the activation scale and an FP32 accumulation in K-block order (equal to DeepGEMM up to the FP32 summation
  order of the four K slices, which are added in a fixed order), BF16 rounding of the result.
* Absorption: ``mma.sync.m16n8k16`` BF16 with FP32 accumulation over the 128 nope values.

Grid: 12 clusters x 6 CTAs (256 threads) per row tile of 8 or 16 rows (batches of more rows launch several tiles,
each cluster independent). CTA ``p`` of cluster ``h`` computes the 32 Q_B rows ``32 p .. 32 p + 31`` of
head ``h`` (CTAs 0-3: q_nope, CTAs 4-5: q_pe); the four q_nope CTAs exchange their rows through distributed shared
memory and each absorbs 128 of the 512 latent columns. The Q_B and W_UK weights are loaded into registers before
``griddepcontrol.wait``.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Dict, Optional, Tuple

import torch

NUM_HEADS = 12
Q_LORA = 1536
NOPE = 128
ROPE = 64
LATENT = 512
HEAD_DIM = NOPE + ROPE
QROW = LATENT + ROPE
MAX_ROWS = 32
CLUSTER = 6
SHARD_FROM_MT = int(__import__("os").environ.get("K3_MLA_Q_SHARD_FROM_MT", "3"))  # batches above 8 * (this - 1) rows shard the RMSNorm / quantization over the cluster

_SOURCE = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define NH 12
#define QL 1536
#define NOPE 128
#define HD 192
#define LAT 512
#define QROW 576
#define KB 12
#define CL 6
#define MP (MT * 8)
#define XP (QL + 8)
#define QNP 136
#define LDS_XQ_BYTES (MP * XP)
#define LDS_XS_BYTES (MP * KB * 4)
#define LDS_QN_BYTES (MP * QNP * 2)
#define LDS_QL_BYTES (MP * 32 * 2)
#define LDS_OT_BYTES (MP * 128)

struct __align__(16) U4 { u32 x, y, z, w; };
struct __align__(8) U2 { u32 x, y; };

__device__ __forceinline__ U4 ldg_stream(const void* p) {
  U4 r;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ float rcp_approx_ftz(float v) {
  float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v)); return r;
}
__device__ __forceinline__ float rsqrt_approx_ftz(float v) {
  float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v)); return r;
}
__device__ __forceinline__ u32 e4m3x4(float a, float b, float c, float d) {
  unsigned short lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(b), "f"(a));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(d), "f"(c));
  return (u32)lo | ((u32)hi << 16);
}
__device__ __forceinline__ u32 e4m3_1(float a) {
  unsigned short lo; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(0.f), "f"(a));
  return (u32)(lo & 0xFFu);
}
__device__ __forceinline__ u16 f2bf(float f) {
  u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f)); return r;
}
__device__ __forceinline__ float lo_bf(u32 v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float hi_bf(u32 v) { return __uint_as_float(v & 0xFFFF0000u); }
__device__ __forceinline__ float bf2f(u16 v) { return __uint_as_float(((u32)v) << 16); }
__device__ __forceinline__ u32 pick(const U4& a, const U4& b, int i) {
  return i == 0 ? a.x : i == 1 ? a.y : i == 2 ? a.z : i == 3 ? a.w : i == 4 ? b.x : i == 5 ? b.y : i == 6 ? b.z : b.w;
}
__device__ __forceinline__ void mma_e4m3(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u32 f16x2_of_e4m3x2(u32 v16) {
  u32 r; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)v16)); return r;
}
__device__ __forceinline__ void mma_f16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_bf16(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

extern "C" __global__ void __cluster_dims__(CL, 1, 1) __launch_bounds__(256, 1)
k3_mla_query(const u16* __restrict__ fa, int ld_fa, const u16* __restrict__ qln_w, float eps,
             const u8* __restrict__ wqb, const float* __restrict__ wsc, const u16* __restrict__ wkc,
             u8* __restrict__ out, int M) {
  extern __shared__ __align__(16) unsigned char smem[];
  u8* xq = smem;
  float* xs = (float*)(smem + LDS_XQ_BYTES);
  u16* qn = (u16*)(smem + LDS_XQ_BYTES + LDS_XS_BYTES);
  u16* ql = (u16*)(smem + LDS_XQ_BYTES + LDS_XS_BYTES + LDS_QN_BYTES);
  u8* ot = smem + LDS_XQ_BYTES + LDS_XS_BYTES + LDS_QN_BYTES + LDS_QL_BYTES;
  float* red = (float*)smem;  // aliases xq once the MMAs are done

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  u32 rank;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const int cid = blockIdx.x / CL;
  const int head = cid % NH;
  const int tile = cid / NH;  // row tile: rows [tile * MP, tile * MP + MP) of the batch
  const int p = (int)rank;
  fa += (size_t)tile * MP * ld_fa;
  out += (size_t)tile * MP * NH * QROW;
  M = min(M - tile * MP, MP);
  const int rel0 = p * 32;
  const int rg = warp & 1, ks = warp >> 1;

  // 1. Static weights into registers before the dependency wait.
  U4 wr[3][4];
  {
    const u8* p0 = wqb + (size_t)(head * HD + rel0 + rg * 16 + g) * QL + (ks * 3) * 128 + 32 * t;
    const u8* p1 = p0 + (size_t)8 * QL;
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
      wr[kk][0] = ldg_stream(p0 + kk * 128);
      wr[kk][1] = ldg_stream(p0 + kk * 128 + 16);
      wr[kk][2] = ldg_stream(p1 + kk * 128);
      wr[kk][3] = ldg_stream(p1 + kk * 128 + 16);
    }
  }
  U4 wk[4][2];
  if (p < 4) {
    const u16* k0 = wkc + ((size_t)head * LAT + p * 128 + warp * 16 + g) * NOPE;
    const u16* k1 = k0 + 8 * NOPE;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      wk[c][0] = ldg_stream(k0 + c * 32 + 8 * t);
      wk[c][1] = ldg_stream(k1 + c * 32 + 8 * t);
    }
  }
#if USE_PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
#endif

#if SHARD
  // 2'. Row-sharded RMSNorm + quantization: CTA p of the cluster normalizes and quantizes rows p, p + 6, ... (one per
  //     warp) and writes the codes and scales into every CTA's xq / xs through distributed shared memory.
  {
    const float r448 = rcp_approx_ftz(448.f);
    const u32 hmask = 0xFFFFu << (lane & 16);
    for (int r = tid >> 5; r < MP; r += 8)  // rows past M: zero codes, unit scales (local, never written remotely)
      if (r >= M) {
#pragma unroll
        for (int b = 0; b < 6; ++b) *(U2*)(xq + r * XP + 256 * b + 8 * lane) = U2{0u, 0u};
        if (lane < KB) xs[r * KB + lane] = 1.f;
      }
    const int r = p + CL * warp;
    if (r < M) {
      U4 raw[6];
      float s = 0.f;
#pragma unroll
      for (int b = 0; b < 6; ++b) {
        raw[b] = *(const U4*)(fa + (size_t)r * ld_fa + 8 * lane + 256 * b);
        const u32 w4[4] = {raw[b].x, raw[b].y, raw[b].z, raw[b].w};
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          const float lo = lo_bf(w4[q4]), hi = hi_bf(w4[q4]);
          s = __fadd_rn(s, __fmul_rn(lo, lo));
          s = __fadd_rn(s, __fmul_rn(hi, hi));
        }
      }
#pragma unroll
      for (int i = 0; i < 5; ++i) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1 << i));
      const float mean = __fdiv_rn(s, (float)QL);
      const float rstd = rsqrt_approx_ftz(__fadd_rn(mean, eps));
#pragma unroll
      for (int b = 0; b < 6; ++b) {
        const U4 gw = *(const U4*)(qln_w + 8 * lane + 256 * b);
        const u32 xw[4] = {raw[b].x, raw[b].y, raw[b].z, raw[b].w};
        const u32 gwv[4] = {gw.x, gw.y, gw.z, gw.w};
        float y[8];
#pragma unroll
        for (int q4 = 0; q4 < 4; ++q4) {
          y[2 * q4] = bf2f(f2bf(__fmul_rn(__fmul_rn(lo_bf(xw[q4]), rstd), lo_bf(gwv[q4]))));
          y[2 * q4 + 1] = bf2f(f2bf(__fmul_rn(__fmul_rn(hi_bf(xw[q4]), rstd), hi_bf(gwv[q4]))));
        }
        float amax = fabsf(y[0]);
#pragma unroll
        for (int j = 1; j < 8; ++j) amax = fmaxf(amax, fabsf(y[j]));
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(hmask, amax, o));
        amax = fmaxf(amax, 1e-10f);
        const u32 sb = __float_as_uint(amax * r448);
        int e8 = (int)((sb >> 23) & 0xFFu) + ((sb & 0x7FFFFFu) != 0u ? 1 : 0);
        e8 = min(e8, 254);
        const float q = __uint_as_float((u32)(254 - e8) << 23);
        const u32 c0 = e4m3x4(y[0] * q, y[1] * q, y[2] * q, y[3] * q);
        const u32 c1 = e4m3x4(y[4] * q, y[5] * q, y[6] * q, y[7] * q);
        const u32 scale_bits = (u32)e8 << 23;
        const u32 sa = (u32)__cvta_generic_to_shared(xq + r * XP + 256 * b + 8 * lane);
        const u32 ssa = (u32)__cvta_generic_to_shared(xs + r * KB + 2 * b + (lane >> 4));
#pragma unroll
        for (int dst = 0; dst < CL; ++dst) {
          u32 ra, rs;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(sa), "r"(dst));
          asm volatile("st.shared::cluster.v2.b32 [%0], {%1,%2};" :: "r"(ra), "r"(c0), "r"(c1) : "memory");
          if ((lane & 15) == 0) {
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rs) : "r"(ssa), "r"(dst));
            asm volatile("st.shared::cluster.b32 [%0], %1;" :: "r"(rs), "r"(scale_bits) : "memory");
          }
        }
      }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  }
#else
#ifndef SKIP_NORM
  // 2. RMSNorm (FlashInfer arithmetic) + 1x128 quantization: warp w takes rows w, w + 8, ...; the MT rows of a warp
  //    are processed together so their dependent chains overlap. Rows past M are zeros.
  const float r448 = rcp_approx_ftz(448.f);
  const u32 hmask = 0xFFFFu << (lane & 16);
  U4 raw[MT][6];
  float ss[MT];
#pragma unroll
  for (int rr = 0; rr < MT; ++rr) {
    const int r = warp + 8 * rr;
    float s = 0.f;
#pragma unroll
    for (int b = 0; b < 6; ++b) {
      raw[rr][b] = r < M ? *(const U4*)(fa + (size_t)r * ld_fa + 8 * lane + 256 * b) : U4{0u, 0u, 0u, 0u};
      const u32 w4[4] = {raw[rr][b].x, raw[rr][b].y, raw[rr][b].z, raw[rr][b].w};
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        const float lo = lo_bf(w4[q4]), hi = hi_bf(w4[q4]);
        s = __fadd_rn(s, __fmul_rn(lo, lo));
        s = __fadd_rn(s, __fmul_rn(hi, hi));
      }
    }
    ss[rr] = s;
  }
#pragma unroll
  for (int i = 0; i < 5; ++i)
#pragma unroll
    for (int rr = 0; rr < MT; ++rr) ss[rr] = __fadd_rn(ss[rr], __shfl_xor_sync(0xFFFFFFFFu, ss[rr], 1 << i));
#pragma unroll
  for (int rr = 0; rr < MT; ++rr) {
    const int r = warp + 8 * rr;
    const float mean = __fdiv_rn(ss[rr], (float)QL);
    const float rstd = rsqrt_approx_ftz(__fadd_rn(mean, eps));
#pragma unroll
    for (int b = 0; b < 6; ++b) {
      const U4 gw = *(const U4*)(qln_w + 8 * lane + 256 * b);
      const u32 xw[4] = {raw[rr][b].x, raw[rr][b].y, raw[rr][b].z, raw[rr][b].w};
      const u32 gwv[4] = {gw.x, gw.y, gw.z, gw.w};
      float y[8];
#pragma unroll
      for (int q4 = 0; q4 < 4; ++q4) {
        y[2 * q4] = bf2f(f2bf(__fmul_rn(__fmul_rn(lo_bf(xw[q4]), rstd), lo_bf(gwv[q4]))));
        y[2 * q4 + 1] = bf2f(f2bf(__fmul_rn(__fmul_rn(hi_bf(xw[q4]), rstd), hi_bf(gwv[q4]))));
      }
      float amax = fabsf(y[0]);
#pragma unroll
      for (int j = 1; j < 8; ++j) amax = fmaxf(amax, fabsf(y[j]));
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(hmask, amax, o));
      amax = fmaxf(amax, 1e-10f);
      const u32 sb = __float_as_uint(amax * r448);
      int e8 = (int)((sb >> 23) & 0xFFu) + ((sb & 0x7FFFFFu) != 0u ? 1 : 0);
      e8 = min(e8, 254);
      const float q = __uint_as_float((u32)(254 - e8) << 23);
      U2 codes;
      codes.x = e4m3x4(y[0] * q, y[1] * q, y[2] * q, y[3] * q);
      codes.y = e4m3x4(y[4] * q, y[5] * q, y[6] * q, y[7] * q);
      *(U2*)(xq + r * XP + 256 * b + 8 * lane) = codes;
      if ((lane & 15) == 0) xs[r * KB + 2 * b + (lane >> 4)] = __uint_as_float((u32)e8 << 23);
    }
  }
#endif
#endif
  __syncthreads();

  // 3. Q_B rows of this CTA: warp (rg, ks) = 16 weight rows x K blocks 3 ks .. 3 ks + 2.
  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  const float* sp = wsc + (size_t)((head * HD + rel0 + rg * 16) / 128) * KB + ks * 3;
#ifndef SKIP_MMA
#pragma unroll
  for (int kk = 0; kk < 3; ++kk) {
    const int kb = ks * 3 + kk;
    const float sw = sp[kk];
#pragma unroll
#if F16_MMA
    // E4M3 -> FP16 is exact; native m16n8k16 FP16 MMAs (FP32 accumulation) instead of the emulated E4M3 ones.
    u32 af[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const u32 wa = pick(wr[kk][0], wr[kk][1], i), wb = pick(wr[kk][2], wr[kk][3], i);
      af[i][0] = f16x2_of_e4m3x2(wa & 0xFFFFu);
      af[i][1] = f16x2_of_e4m3x2(wb & 0xFFFFu);
      af[i][2] = f16x2_of_e4m3x2(wa >> 16);
      af[i][3] = f16x2_of_e4m3x2(wb >> 16);
    }
#endif
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      float tmp[4] = {0.f, 0.f, 0.f, 0.f};
#if F16_MMA
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const U2 b = *(const U2*)(xq + (j * 8 + g) * XP + kb * 128 + 32 * t + 8 * s);
        mma_f16(tmp, af[2 * s][0], af[2 * s][1], af[2 * s][2], af[2 * s][3],
                f16x2_of_e4m3x2(b.x & 0xFFFFu), f16x2_of_e4m3x2(b.x >> 16));
        mma_f16(tmp, af[2 * s + 1][0], af[2 * s + 1][1], af[2 * s + 1][2], af[2 * s + 1][3],
                f16x2_of_e4m3x2(b.y & 0xFFFFu), f16x2_of_e4m3x2(b.y >> 16));
      }
#else
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const U2 b = *(const U2*)(xq + (j * 8 + g) * XP + kb * 128 + 32 * t + 8 * s);
        mma_e4m3(tmp, pick(wr[kk][0], wr[kk][1], 2 * s), pick(wr[kk][2], wr[kk][3], 2 * s),
                 pick(wr[kk][0], wr[kk][1], 2 * s + 1), pick(wr[kk][2], wr[kk][3], 2 * s + 1), b.x, b.y);
      }
#endif
      const float s0 = sw * xs[(j * 8 + 2 * t) * KB + kb];
      const float s1 = sw * xs[(j * 8 + 2 * t + 1) * KB + kb];
      acc[j][0] = fmaf(tmp[0], s0, acc[j][0]);
      acc[j][1] = fmaf(tmp[1], s1, acc[j][1]);
      acc[j][2] = fmaf(tmp[2], s0, acc[j][2]);
      acc[j][3] = fmaf(tmp[3], s1, acc[j][3]);
    }
  }
#endif
  __syncthreads();
#pragma unroll
  for (int j = 0; j < MT; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) red[(warp * MT * 4 + j * 4 + i) * 32 + lane] = acc[j][i];
  __syncthreads();
  if (ks == 0) {  // warps 0 and 1: add the four K slices in a fixed order, round to BF16
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      float tot[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float sum = red[((0 * 2 + rg) * MT * 4 + j * 4 + i) * 32 + lane];
#pragma unroll
        for (int k2 = 1; k2 < 4; ++k2) sum += red[((k2 * 2 + rg) * MT * 4 + j * 4 + i) * 32 + lane];
        tot[i] = sum;
      }
      const int cl = rg * 16 + g;
      ql[(j * 8 + 2 * t) * 32 + cl] = f2bf(tot[0]);
      ql[(j * 8 + 2 * t + 1) * 32 + cl] = f2bf(tot[1]);
      ql[(j * 8 + 2 * t) * 32 + cl + 8] = f2bf(tot[2]);
      ql[(j * 8 + 2 * t + 1) * 32 + cl + 8] = f2bf(tot[3]);
    }
  }
  __syncthreads();

  if (p >= 4) {  // q_pe rows: FP8 copy straight to the query
    for (int i = tid; i < M * 8; i += 256) {
      const int m = i >> 3, c4 = (i & 7) * 4;
      const U2 raw = *(const U2*)(ql + m * 32 + c4);
      *(u32*)(out + ((size_t)m * NH + head) * QROW + LAT + (p - 4) * 32 + c4) =
          e4m3x4(lo_bf(raw.x), hi_bf(raw.x), lo_bf(raw.y), hi_bf(raw.y));
    }
  } else {  // q_nope rows: to the smem of the four absorbing CTAs of the cluster
    for (int i = tid; i < MP * 4 * 4; i += 256) {
      const int m = i >> 4, c = (i >> 2) & 3, dst = i & 3;
      const U4 val = *(const U4*)(ql + m * 32 + c * 8);
      u32 saddr = (u32)__cvta_generic_to_shared(qn + m * QNP + p * 32 + c * 8);
      u32 raddr;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(raddr) : "r"(saddr), "r"(dst));
      asm volatile("st.shared::cluster.v4.b32 [%0], {%1,%2,%3,%4};"
                   :: "r"(raddr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
    }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

#ifndef SKIP_ABS
  // 4. Absorption: latent columns 128 p + 16 warp .. +15, K = 128 nope values.
  if (p < 4) {
    float a2[MT][4];
#pragma unroll
    for (int j = 0; j < MT; ++j) a2[j][0] = a2[j][1] = a2[j][2] = a2[j][3] = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        const U4 bq = *(const U4*)(qn + (j * 8 + g) * QNP + c * 32 + 8 * t);
        mma_bf16(a2[j], wk[c][0].x, wk[c][1].x, wk[c][0].y, wk[c][1].y, bq.x, bq.y);
        mma_bf16(a2[j], wk[c][0].z, wk[c][1].z, wk[c][0].w, wk[c][1].w, bq.z, bq.w);
      }
    }
    const int lc = warp * 16 + g;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      ot[(j * 8 + 2 * t) * 128 + lc] = (u8)e4m3_1(a2[j][0]);
      ot[(j * 8 + 2 * t + 1) * 128 + lc] = (u8)e4m3_1(a2[j][1]);
      ot[(j * 8 + 2 * t) * 128 + lc + 8] = (u8)e4m3_1(a2[j][2]);
      ot[(j * 8 + 2 * t + 1) * 128 + lc + 8] = (u8)e4m3_1(a2[j][3]);
    }
    __syncthreads();
    for (int i = tid; i < M * 8; i += 256) {
      const int m = i >> 3, c16 = i & 7;
      *(U4*)(out + ((size_t)m * NH + head) * QROW + p * 128 + c16 * 16) = *(const U4*)(ot + m * 128 + c16 * 16);
    }
  }
#endif
}

// latent_cache[m] = [RMSNorm(kv_a[m]) (512) | k_pe[m] (64)] in BF16: the kv_a RMSNorm (FlashInfer arithmetic, H = 512)
// and the concatenation of the stock chain in one kernel; one warp per row, 8 rows per CTA.
extern "C" __global__ void __launch_bounds__(256)
k3_mla_kv_prep(const u16* __restrict__ ckv, int ld, const u16* __restrict__ w, float eps, u16* __restrict__ latent,
               int M) {
#if USE_PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  const int lane = threadIdx.x & 31;
  const int r = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (r >= M) return;
  const u16* row = ckv + (size_t)r * ld;
  U4 raw[2];
  float s = 0.f;
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    raw[b] = *(const U4*)(row + 8 * lane + 256 * b);
    const u32 w4[4] = {raw[b].x, raw[b].y, raw[b].z, raw[b].w};
#pragma unroll
    for (int q4 = 0; q4 < 4; ++q4) {
      const float lo = lo_bf(w4[q4]), hi = hi_bf(w4[q4]);
      s = __fadd_rn(s, __fmul_rn(lo, lo));
      s = __fadd_rn(s, __fmul_rn(hi, hi));
    }
  }
#pragma unroll
  for (int i = 0; i < 5; ++i) s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1 << i));
  const float rstd = rsqrt_approx_ftz(__fadd_rn(__fdiv_rn(s, 512.f), eps));
#pragma unroll
  for (int b = 0; b < 2; ++b) {
    const U4 gw = *(const U4*)(w + 8 * lane + 256 * b);
    const u32 xw[4] = {raw[b].x, raw[b].y, raw[b].z, raw[b].w};
    const u32 gwv[4] = {gw.x, gw.y, gw.z, gw.w};
    u32 o[4];
#pragma unroll
    for (int q4 = 0; q4 < 4; ++q4) {
      const u16 y0 = f2bf(__fmul_rn(__fmul_rn(lo_bf(xw[q4]), rstd), lo_bf(gwv[q4])));
      const u16 y1 = f2bf(__fmul_rn(__fmul_rn(hi_bf(xw[q4]), rstd), hi_bf(gwv[q4])));
      o[q4] = (u32)y0 | ((u32)y1 << 16);
    }
    *(U4*)(latent + (size_t)r * 576 + 8 * lane + 256 * b) = U4{o[0], o[1], o[2], o[3]};
  }
  if (lane < 8) *(U4*)(latent + (size_t)r * 576 + 512 + 8 * lane) = *(const U4*)(row + 512 + 8 * lane);
}
"""

_LOCK = threading.Lock()
_FUNCTIONS: Dict[Tuple, object] = {}


def _check(status, what: str) -> None:
    import cuda.bindings.driver as cuda

    code = status[0] if isinstance(status, tuple) else status
    if code != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {code}")


def smem_bytes(mt: int) -> int:
    mp = 8 * mt
    return mp * (Q_LORA + 8) + mp * 12 * 4 + mp * 136 * 2 + mp * 32 * 2 + mp * 128


def _compile(mt: int, use_pdl: bool, device: torch.device, f16: bool = True, shard: bool = False):
    key = (mt, use_pdl, device.index, f16, shard)
    with _LOCK:
        hit = _FUNCTIONS.get(key)
        if hit is not None:
            return hit
        import cuda.bindings.driver as cuda
        import cuda.bindings.nvrtc as nvrtc

        major, minor = torch.cuda.get_device_capability(device)
        err, prog = nvrtc.nvrtcCreateProgram(_SOURCE.encode(), b"k3_mla_query.cu", 0, [], [])
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"nvrtcCreateProgram failed: {err}")
        arch = f"sm_{major}{minor}" + ("a" if major >= 10 else "")
        opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17", b"-fmad=false"]
        import os
        opts += [f"-D{d}".encode() for d in os.environ.get("K3_MLA_Q_DEFS", "").split(",") if d]
        opts += [f"-DMT={mt}".encode(), f"-DUSE_PDL={int(use_pdl)}".encode(), f"-DF16_MMA={int(f16)}".encode(),
                 f"-DSHARD={int(shard)}".encode()]
        if os.environ.get("K3_MLA_Q_VERBOSE"):
            opts.append(b"--ptxas-options=--verbose")
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        if os.environ.get("K3_MLA_Q_VERBOSE"):
            _, size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            print(f"NVRTC MT={mt}: {log.decode(errors='replace')}", flush=True)
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, size = nvrtc.nvrtcGetProgramLogSize(prog)
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"NVRTC failed for MT={mt}:\n{log.decode(errors='replace')}")
        _, size = nvrtc.nvrtcGetCUBINSize(prog)
        cubin = b" " * size
        nvrtc.nvrtcGetCUBIN(prog, cubin)
        nvrtc.nvrtcDestroyProgram(prog)
        with torch.cuda.device(device):
            torch.zeros(1, device=device)
            status, module = cuda.cuModuleLoadData(cubin)
            _check(status, "cuModuleLoadData")
            status, function = cuda.cuModuleGetFunction(module, b"k3_mla_query")
            _check(status, "cuModuleGetFunction")
            status, kv_function = cuda.cuModuleGetFunction(module, b"k3_mla_kv_prep")
            _check(status, "cuModuleGetFunction(kv_prep)")
            _check(
                cuda.cuFuncSetAttribute(
                    function,
                    cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_bytes(mt),
                ),
                "cuFuncSetAttribute(smem)",
            )
        _FUNCTIONS[key] = (module, function, kv_function)
        return _FUNCTIONS[key]


def block_scale_from_deepgemm(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """FP32 ``[N / 128, K / 128]`` block scales of a 128x128-quantized weight from DeepGEMM's packed
    (INT32, 1x128, MN-major) scale; raises if a 128-row block does not carry one scale."""
    from tensorrt_llm.quantization.utils.fp8_utils import unpack_col_major_tma_aligned_packed_tensor

    per_row = unpack_col_major_tma_aligned_packed_tensor(weight_scale, n, k // 128)
    blocks = per_row.view(n // 128, 128, k // 128)
    if not torch.equal(blocks, blocks[:, :1, :].expand_as(blocks)):
        raise ValueError("weight scale is not constant over 128-row blocks")
    return blocks[:, 0, :].contiguous()


class K3MlaQuery:
    """One MLA layer's fused query kernel (weights bound at construction, outside CUDA graph capture)."""

    def __init__(
        self,
        q_b_weight: torch.Tensor,
        q_b_scale: torch.Tensor,
        q_a_ln_weight: torch.Tensor,
        eps: float,
        w_kc: torch.Tensor,
        kv_a_ln_weight: torch.Tensor,
        kv_eps: float,
        use_pdl: bool = True,
        f16: bool = True,
    ) -> None:
        n, k = q_b_weight.shape
        if (n, k) != (NUM_HEADS * HEAD_DIM, Q_LORA) or q_b_weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"q_b weight must be E4M3 [{NUM_HEADS * HEAD_DIM}, {Q_LORA}], got {q_b_weight.shape}")
        if not q_b_weight.is_contiguous() or q_b_weight.data_ptr() % 16:
            raise ValueError("q_b weight must be contiguous and 16-byte aligned")
        if tuple(w_kc.shape) != (NUM_HEADS, LATENT, NOPE) or w_kc.dtype != torch.bfloat16 or not w_kc.is_contiguous():
            raise ValueError("w_kc must be contiguous BF16 [12, 512, 128]")
        if q_a_ln_weight.dtype != torch.bfloat16 or q_a_ln_weight.numel() != Q_LORA:
            raise ValueError("q_a layernorm weight must be BF16 [1536]")
        if kv_a_ln_weight.dtype != torch.bfloat16 or kv_a_ln_weight.numel() != LATENT or kv_a_ln_weight.data_ptr() % 16:
            raise ValueError("kv_a layernorm weight must be 16-byte aligned BF16 [512]")
        if w_kc.data_ptr() % 16 or q_a_ln_weight.data_ptr() % 16:
            raise ValueError("weights must be 16-byte aligned")
        self.q_b_weight = q_b_weight
        self.block_scale = block_scale_from_deepgemm(q_b_scale, n, k)
        self.ln_weight = q_a_ln_weight
        self.w_kc = w_kc
        self.eps = float(eps)
        self.use_pdl = use_pdl
        self.device = q_b_weight.device
        self._fn = {mt: _compile(mt, use_pdl, self.device, f16, mt >= SHARD_FROM_MT)[1] for mt in range(1, MAX_ROWS // 8 + 1)}
        self._kv_fn = _compile(1, use_pdl, self.device, f16, False)[2]
        self.kv_ln_weight = kv_a_ln_weight
        self.kv_eps = float(kv_eps)

    def __call__(self, q_a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """``q_a``: BF16 ``[M, 1536]`` rows (16-byte aligned, row stride a multiple of 8); ``out``: contiguous
        uint8 / E4M3 ``[M, 12, 576]``."""
        import cuda.bindings.driver as cuda

        m = q_a.shape[0]
        if not 0 < m <= MAX_ROWS:
            raise ValueError(f"rows {m} outside 1..{MAX_ROWS}")
        if (
            q_a.dtype != torch.bfloat16
            or q_a.shape[1] != Q_LORA
            or q_a.stride(1) != 1
            or q_a.stride(0) % 8
            or q_a.data_ptr() % 16
        ):
            raise ValueError("q_a must be 16-byte aligned BF16 [M, 1536] rows")
        if out.shape != (m, NUM_HEADS, QROW) or out.element_size() != 1 or not out.is_contiguous():
            raise ValueError("out must be contiguous 1-byte [M, 12, 576]")
        mt = (m + 7) // 8
        tiles = 1
        config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = NUM_HEADS * CLUSTER * tiles, 1, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = 256, 1, 1
        config.sharedMemBytes = smem_bytes(mt)
        attrs = []
        if self.use_pdl:
            attr = cuda.CUlaunchAttribute()
            attr.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            attr.value.programmaticStreamSerializationAllowed = 1
            attrs.append(attr)
        config.attrs = attrs
        config.numAttrs = len(attrs)
        config.hStream = cuda.CUstream(torch.cuda.current_stream(q_a.device).cuda_stream)
        holders = (
            ctypes.c_void_p(q_a.data_ptr()),
            ctypes.c_int(q_a.stride(0)),
            ctypes.c_void_p(self.ln_weight.data_ptr()),
            ctypes.c_float(self.eps),
            ctypes.c_void_p(self.q_b_weight.data_ptr()),
            ctypes.c_void_p(self.block_scale.data_ptr()),
            ctypes.c_void_p(self.w_kc.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(m),
        )
        args = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(h) for h in holders])
        _check(
            cuda.cuLaunchKernelEx(config, self._fn[mt], ctypes.addressof(args), 0),
            "cuLaunchKernelEx(k3_mla_query)",
        )
        return out

    def kv_prep(self, kv_a: torch.Tensor, latent_cache: torch.Tensor) -> torch.Tensor:
        """``latent_cache[m] = [RMSNorm(kv_a[m]) | k_pe[m]]`` (BF16 ``[M, 576]``) from ``kv_a``: the 576 BF16 columns
        ``[kv_a (512) | k_pe (64)]`` of the fused projection output (16-byte aligned rows)."""
        import cuda.bindings.driver as cuda

        m = kv_a.shape[0]
        if (
            kv_a.dtype != torch.bfloat16
            or kv_a.shape[1] != LATENT + ROPE
            or kv_a.stride(1) != 1
            or kv_a.stride(0) % 8
            or kv_a.data_ptr() % 16
        ):
            raise ValueError("kv_a must be 16-byte aligned BF16 [M, 576] rows")
        if latent_cache.shape != (m, LATENT + ROPE) or latent_cache.dtype != torch.bfloat16 or not latent_cache.is_contiguous():
            raise ValueError("latent_cache must be contiguous BF16 [M, 576]")
        config = cuda.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = -(-m // 8), 1, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = 256, 1, 1
        config.sharedMemBytes = 0
        attrs = []
        if self.use_pdl:
            attr = cuda.CUlaunchAttribute()
            attr.id = cuda.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            attr.value.programmaticStreamSerializationAllowed = 1
            attrs.append(attr)
        config.attrs = attrs
        config.numAttrs = len(attrs)
        config.hStream = cuda.CUstream(torch.cuda.current_stream(kv_a.device).cuda_stream)
        holders = (
            ctypes.c_void_p(kv_a.data_ptr()),
            ctypes.c_int(kv_a.stride(0)),
            ctypes.c_void_p(self.kv_ln_weight.data_ptr()),
            ctypes.c_float(self.kv_eps),
            ctypes.c_void_p(latent_cache.data_ptr()),
            ctypes.c_int(m),
        )
        args = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(h) for h in holders])
        _check(cuda.cuLaunchKernelEx(config, self._kv_fn, ctypes.addressof(args), 0), "cuLaunchKernelEx(k3_mla_kv_prep)")
        return latent_cache
