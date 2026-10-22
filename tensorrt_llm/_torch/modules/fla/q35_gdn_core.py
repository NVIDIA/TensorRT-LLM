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
"""GDN target-verify core for Qwen3.5 low latency: cached replay + gated RMSNorm (+ FP8) as one CUDA kernel.

Drop-in for ``q35_gdn_fused_update(..., norm_weight=...)`` on convolved q|k|v rows; gdn_mixer.py selects it.
Compiled at first use with NVRTC; CUDA-graph capturable, PDL-launched. One CTA per (request, value head), so the
gated RMSNorm of a head reduces inside the CTA. Supported: K = V = 128, history 16, 2 <= T <= 8, BF16 state.

PDL: before ``griddepcontrol.wait`` the kernel reads only data from earlier steps or static data (slots, history,
SSM checkpoint state, gating / norm parameters); the in_proj outputs and every global store follow the wait.

Numerics: the Triton kernel's bf16 roundings are kept but FP32 sums run in other orders, so outputs equal the
Triton path up to rounding.
"""

from __future__ import annotations

import ctypes
import threading
from typing import Optional

import torch

Q35_GDN_CORE_MAX_T = 8

_SRC = r"""
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define KD 128
#define VD 128
#define HIST 16
#define NW 4
#define FULL 0xffffffffu
#ifndef WARM
#define WARM 0
#endif

struct alignas(16) U4 { u32 x, y, z, w; };
struct alignas(8) F2 { float x, y; };

__device__ __forceinline__ float bflo(u32 x) { return __int_as_float(x << 16); }
__device__ __forceinline__ float bfhi(u32 x) { return __int_as_float(x & 0xffff0000u); }
__device__ __forceinline__ float bf2f(u16 x) { return __int_as_float(((u32)x) << 16); }
__device__ __forceinline__ u16 f2bf(float x) {
  u16 r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float bfround(float x) { return bf2f(f2bf(x)); }
// (lo, hi) -> bf16x2 with lo in the low half
__device__ __forceinline__ u32 pack2(float lo, float hi) {
  u32 r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ u32 word(const U4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
__device__ __forceinline__ U4 ld16(const void* p) { return *reinterpret_cast<const U4*>(p); }
__device__ __forceinline__ void st16(void* p, U4 v) { *reinterpret_cast<U4*>(p) = v; }
__device__ __forceinline__ U4 zero4() { U4 z; z.x = z.y = z.z = z.w = 0u; return z; }
__device__ __forceinline__ float sqrt_ap(float x) {
    float r; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float rcp_ap(float x) {
    float r; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float rsqrt_ap(float x) {
    float r; asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float rcp_rn(float x) {
  float r;
  asm("div.rn.f32 %0, 0f3F800000, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ u8 f2e4m3(float x) {
  u16 r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(r) : "f"(x));
  return (u8)(r & 0xff);
}
__device__ __forceinline__ float pick8(const float (&a)[8], int i) {
  float r = a[0];
#pragma unroll
  for (int k = 1; k < 8; ++k) r = (i == k) ? a[k] : r;
  return r;
}

__device__ __forceinline__ void mma16816(float (&d)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// K-reduction over 128-wide rows: A = rows r (rows r + 8 are zero), B = 8 columns (rows of B). The lane (r, q) holds
// the physical elements [32 c + 8 q, 32 c + 8 q + 8) of its row in chunk c; k-step 2c + e uses words (2e, 2e + 1)
// for both operands.
__device__ __forceinline__ void kred(float (&acc)[4], const U4 (&A)[4], const U4 (&B)[4]) {
#pragma unroll
  for (int c = 0; c < 4; ++c) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      mma16816(acc, word(A[c], 2 * e), 0u, word(A[c], 2 * e + 1), 0u, word(B[c], 2 * e), word(B[c], 2 * e + 1));
    }
  }
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// Pointers to data that another kernel writes (the in_proj outputs x / z / a / b, the per-step metadata) are not
// __restrict__: const + __restrict__ lets the compiler use ld.global.nc, which PTX allows only for data that stays
// read-only for the kernel's whole lifetime, and ptxas may then schedule those loads ahead of griddepcontrol.wait
// (seen: stale z / v / b under a delayed PDL producer). Only the static parameters (A_log, dt_bias, norm weight, FP8
// scale) keep __restrict__.
extern "C" __global__ void __launch_bounds__(NW * 32, 1) q35_gdn_core(
    const u16* x, int x_stride, const u16* z, int z_stride,
    const u16* a_in, const u16* b_in, int ab_stride,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    u16* state, long long state_slot_stride, const int* state_indices,
    u16* old_u, u16* old_k, float* old_G,
    const int* cache_buf_idx, const int* pnat,
    const u16* __restrict__ norm_w, const float* __restrict__ fp8_scale, float eps, float qscale,
    u8* y, int y_stride, u16* o_out) {
  constexpr int GRP = HV / HK;  // value heads per key head
  const int cta = blockIdx.x;
  const int n = cta / HV;
  const int hv = cta - n * HV;
  const int hk = hv / GRP;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int r = lane >> 2;
  const int q = lane & 3;
  const int vb = warp * 32;  // value columns vb .. vb + 31: m-tile m, half h -> v = vb + 16 m + 8 h + r
  const int t0 = 2 * q;
  const int t1 = 2 * q + 1;
  const long long row0 = (long long)n * TT;

  __shared__ F2 s_coef[2][2][32];  // [k / q][j-tile][lane]: raw history product x decay (scaled by 1/|row| on use)
  __shared__ float s_inv[2][8];    // 1 / |k_t|, scale / |q_t|
  __shared__ float s_A[8][8];      // A[t][s] (final)
  __shared__ float s_Q[8][8];      // raw q_t k_s x decay, t >= s (scaled by inv_q[t] inv_k[s] on use)
  __shared__ float s_norm[NW][8];
  __shared__ U4 s_hc[VD][17];      // checkpoint commit staging (row v, 16-byte slot 4 c + q), padded

#if TRIGGER == 2
  pdl_launch();
#endif
  const int slot = state_indices[n];
  if (slot < 0) {
    // Padded request: defined zeros for out_proj (after the wait, like every global store).
    pdl_wait();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = (i & 1) ? t1 : t0;
      const int vs = vb + 16 * (i >> 2) + 8 * ((i >> 1) & 1) + r;
      if (t < TT) {
#if OUT_FP8
        y[(row0 + t) * y_stride + hv * VD + vs] = 0;
#else
        reinterpret_cast<u16*>(y)[(row0 + t) * y_stride + hv * VD + vs] = 0;
#endif
#if STORE_O
        o_out[((row0 + t) * HV + hv) * VD + vs] = 0;
#endif
      }
    }
#if TRIGGER == 3
    __syncthreads();
    pdl_launch();
#endif
    return;
  }

  // ---------------- before the wait: earlier-step and static data ----------------
  const int pn = pnat[slot];
  const int buf = cache_buf_idx[slot];
  const bool is_write = pn + TT > HIST;
  const int w_buf = is_write ? 1 - buf : buf;
  const int w_off = is_write ? 0 : pn;
  const long long rb = (long long)slot * 2 + buf;
  const long long wb = (long long)slot * 2 + w_buf;
  const u16* khp = old_k + rb * (HIST * HK * KD) + hk * KD;
  const u16* uhp = old_u + rb * (HIST * HV * VD) + hv * VD;
  const float* ghp = old_G + (rb * HV + hv) * HIST;
  u16* sp = state + (long long)slot * state_slot_stride + (long long)hv * VD * KD;

  U4 H[2][2][4];  // checkpoint state rows (h^T) of this lane: [m-tile][half][chunk]
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int c = 0; c < 4; ++c) H[m][h][c] = ld16(sp + (vb + 16 * m + 8 * h + r) * KD + 32 * c + 8 * q);
  // history columns of this lane: j = 2q, 2q + 1, 2q + 8, 2q + 9 (rows j >= pn are masked to zero)
  const int jj[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
  bool hj[4];
  float gh[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    hj[i] = jj[i] < pn;
    gh[i] = hj[i] ? ghp[jj[i]] : 0.f;
  }
  const float g_start = pn > 0 ? ghp[pn - 1] : 0.f;
  // uh^T as the A operand of the history outputs:
  // [m-tile][a0..a3] = (v0, j 2q..2q+1), (v1, ..), (v0, j + 8), (v1, j + 8)
  u32 UA[2][4];
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int va = vb + 16 * m + r, vc = va + 8;
    u16 e[8];
    e[0] = hj[0] ? uhp[jj[0] * (HV * VD) + va] : (u16)0;
    e[1] = hj[1] ? uhp[jj[1] * (HV * VD) + va] : (u16)0;
    e[2] = hj[0] ? uhp[jj[0] * (HV * VD) + vc] : (u16)0;
    e[3] = hj[1] ? uhp[jj[1] * (HV * VD) + vc] : (u16)0;
    e[4] = hj[2] ? uhp[jj[2] * (HV * VD) + va] : (u16)0;
    e[5] = hj[3] ? uhp[jj[3] * (HV * VD) + va] : (u16)0;
    e[6] = hj[2] ? uhp[jj[2] * (HV * VD) + vc] : (u16)0;
    e[7] = hj[3] ? uhp[jj[3] * (HV * VD) + vc] : (u16)0;
#pragma unroll
    for (int i = 0; i < 4; ++i) UA[m][i] = (u32)e[2 * i] | ((u32)e[2 * i + 1] << 16);
  }
  // history keys (warps 1 and 2: the history products), rows j = r + 8 jt, rows >= pn zero
  U4 KH[2][4];
#pragma unroll
  for (int jt = 0; jt < 2; ++jt) {
    const int j = r + 8 * jt;
    const bool ok = (warp == 1 || warp == 2) && j < pn;
#pragma unroll
    for (int c = 0; c < 4; ++c) KH[jt][c] = ok ? ld16(khp + j * (HK * KD) + 32 * c + 8 * q) : zero4();
  }
  const float negAexp = -__expf(A_log[hv]);
  const float dtb = dt_bias[hv];
  float nw[2][2];
#pragma unroll
  for (int m = 0; m < 2; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) nw[m][h] = bf2f(norm_w[vb + 16 * m + 8 * h + r]);
#if OUT_FP8
  const float rscale = rcp_rn(fp8_scale[0]);
#endif
  // post-wait addresses
  const u16* xq = x + (row0 + (r < TT ? r : 0)) * x_stride + hk * KD + 8 * q;
  const u16* xk = xq + HK * KD;
  const u16* xv0 = x + (row0 + t0) * x_stride + 2 * HK * KD + hv * VD + vb + r;
  const u16* xv1 = xv0 + x_stride;
  const u16* zv0 = z + (row0 + t0) * z_stride + hv * VD + vb + r;
  const u16* zv1 = zv0 + z_stride;
  const u16* pa = a_in + (row0 + (lane < TT ? lane : 0)) * ab_stride + hv;
  const u16* pb = b_in + (row0 + (lane < TT ? lane : 0)) * ab_stride + hv;

  // Checkpoint commit (history full): h' = h e^{g_start} + kh^T Uc, Uc[j, v] = u[j, v] e^{g_start - G_j} (j < pn),
  // Uc split into bf16 hi + lo. Transposed: h'^T[v, kd] = h^T e^{g_start} + Uc^T[v, j] kh[j, kd]; the n-tile (c, wd)
  // covers the physical kd of word wd of chunk c of this lane's state rows. Results are staged in shared memory and
  // stored after the wait.
  if (is_write) {
    const float eg = __expf(g_start);
    float cd[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) cd[i] = hj[i] ? __expf(g_start - gh[i]) : 0.f;
    u32 AH[2][4], AL[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        // a_i: (half i & 1, j pair i >> 1): elements j = jj[2 (i >> 1)], jj[2 (i >> 1) + 1]
        const float lo = bflo(UA[m][i]) * cd[2 * (i >> 1)], hi = bfhi(UA[m][i]) * cd[2 * (i >> 1) + 1];
        AH[m][i] = pack2(lo, hi);
        AL[m][i] = pack2(lo - bflo(AH[m][i]), hi - bfhi(AH[m][i]));
      }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      u32 out[2][2][4];
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
        // B column n = r holds kd = 32c + 8 (r >> 1) + 2 wd + (r & 1)
        const int kd = 32 * c + 8 * (r >> 1) + 2 * wd + (r & 1);
        const u16 k0 = hj[0] ? khp[jj[0] * (HK * KD) + kd] : (u16)0;
        const u16 k1 = hj[1] ? khp[jj[1] * (HK * KD) + kd] : (u16)0;
        const u16 k2 = hj[2] ? khp[jj[2] * (HK * KD) + kd] : (u16)0;
        const u16 k3 = hj[3] ? khp[jj[3] * (HK * KD) + kd] : (u16)0;
        const u32 b0 = (u32)k0 | ((u32)k1 << 16);
        const u32 b1 = (u32)k2 | ((u32)k3 << 16);
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          float acc[4] = {0.f, 0.f, 0.f, 0.f};
          mma16816(acc, AH[m][0], AH[m][1], AH[m][2], AH[m][3], b0, b1);
          mma16816(acc, AL[m][0], AL[m][1], AL[m][2], AL[m][3], b0, b1);
          const u32 hw0 = word(H[m][0][c], wd), hw1 = word(H[m][1][c], wd);
          out[m][0][wd] = pack2(bflo(hw0) * eg + acc[0], bfhi(hw0) * eg + acc[1]);
          out[m][1][wd] = pack2(bflo(hw1) * eg + acc[2], bfhi(hw1) * eg + acc[3]);
        }
      }
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          U4 v4;
          v4.x = out[m][h][0]; v4.y = out[m][h][1]; v4.z = out[m][h][2]; v4.w = out[m][h][3];
          s_hc[vb + 16 * m + 8 * h + r][4 * c + q] = v4;
        }
    }
  }

  // WARM=1: the post-wait path runs once before the wait on zero inputs with every global access off (pass 0), so
  // that its instructions are cached on this SM when the in_proj outputs arrive (pass 1).
#pragma unroll 1
  for (int pass = WARM ? 0 : 1; pass < 2; ++pass) {
    const bool real = pass == 1;
    if (real) {
      // ---------------- the wait: in_proj outputs from here on ----------------
      pdl_wait();
#if TRIGGER == 1
      pdl_launch();
#endif
    }
    // raw q / k rows (row r = token; rows >= T zero), v / z of this lane's (v, t), a / b of token = lane
    U4 QR[4], KR[4];
    const bool rowok = real && r < TT;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      QR[c] = rowok ? ld16(xq + 32 * c) : zero4();
      KR[c] = rowok ? ld16(xk + 32 * c) : zero4();
    }
    float vv[2][4], zz[2][4];  // [m][(h, t)]: 0 (h0, t0), 1 (h0, t1), 2 (h1, t0), 3 (h1, t1)
    const bool ok0 = real && t0 < TT, ok1 = real && t1 < TT;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int off = 16 * m + 8 * h;
        vv[m][2 * h] = ok0 ? bf2f(xv0[off]) : 0.f;
        vv[m][2 * h + 1] = ok1 ? bf2f(xv1[off]) : 0.f;
        zz[m][2 * h] = ok0 ? bf2f(zv0[off]) : 0.f;
        zz[m][2 * h + 1] = ok1 ? bf2f(zv1[off]) : 0.f;
      }
    }
    float ga = 0.f, gb = 0.f;
    if (real && lane < TT) {
      ga = bf2f(*pa);
      gb = bf2f(*pb);
    }
    if (real && is_write) {
      // checkpoint commit stores (issued after this pass's loads)
#pragma unroll
      for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int v = vb + 16 * m + 8 * h + r;
            st16(sp + v * KD + 32 * c + 8 * q, s_hc[v][4 * c + q]);
          }
    }

    // gating (lane t < T), cumulative log-decay by an 8-lane scan
    float gl = 0.f, beta = 0.f;
    if (lane < TT) {
      const float xg = ga + dtb;
      const float spl = xg <= 20.f ? 0.6931471805599453f * __log2f(1.f + __expf(xg)) : xg;
      gl = negAexp * spl;
      beta = __fdividef(1.f, 1.f + __expf(-gb));
    }
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      const float o = __shfl_up_sync(FULL, gl, d);
      if (lane >= d) gl += o;
    }
    const float Gv = g_start + gl;  // lane t: G[t]
    const float Gt0 = __shfl_sync(FULL, Gv, t0), Gt1 = __shfl_sync(FULL, Gv, t1);
    const float bt0 = __shfl_sync(FULL, beta, t0), bt1 = __shfl_sync(FULL, beta, t1);
    const float Gr = __shfl_sync(FULL, Gv, r), br = __shfl_sync(FULL, beta, r);
    const float eG0 = __expf(Gt0), eG1 = __expf(Gt1);

    // state products on the raw rows (all warps): rows v, columns = tokens; scaled by 1 / |k_t|, scale / |q_t| later
    float KS[2][4], QS[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int i = 0; i < 4; ++i) KS[m][i] = QS[m][i] = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const u32 a0 = word(H[m][0][c], 2 * e), a1 = word(H[m][1][c], 2 * e);
          const u32 a2 = word(H[m][0][c], 2 * e + 1), a3 = word(H[m][1][c], 2 * e + 1);
          mma16816(KS[m], a0, a1, a2, a3, word(KR[c], 2 * e), word(KR[c], 2 * e + 1));
          mma16816(QS[m], a0, a1, a2, a3, word(QR[c], 2 * e), word(QR[c], 2 * e + 1));
        }
      }
    }

    // head-common products: warp 3: k k^T, q q^T (norms, A); warp 0: q k^T; warps 1 / 2: k / q x history keys
    if (warp == 3) {
      float kk[4] = {0.f, 0.f, 0.f, 0.f}, qq[4] = {0.f, 0.f, 0.f, 0.f};
      kred(kk, KR, KR);
      kred(qq, QR, QR);
      // |k_t|^2 = (k k^T)[t, t] sits in lane 4t + (t >> 1), element t & 1
      const float dk = (r & 1) ? kk[1] : kk[0];
      const float dq = (r & 1) ? qq[1] : qq[0];
      const float ik = rcp_ap(sqrt_ap(dk + 1e-6f));
      const float iq = qscale * rcp_ap(sqrt_ap(dq + 1e-6f));
      const float ik_r = __shfl_sync(FULL, ik, 4 * r + (r >> 1));
      const float ik_0 = __shfl_sync(FULL, ik, 4 * t0 + q);
      const float ik_1 = __shfl_sync(FULL, ik, 4 * t1 + q);
      if (q == (r >> 1)) {
        s_inv[0][r] = ik;
        s_inv[1][r] = iq;
      }
      s_A[r][t0] = (r > t0 && r < TT) ? br * (kk[0] * ik_r * ik_0) * __expf(Gr - Gt0) : 0.f;
      s_A[r][t1] = (r > t1 && r < TT) ? br * (kk[1] * ik_r * ik_1) * __expf(Gr - Gt1) : 0.f;
    } else if (warp == 0) {
      float qk[4] = {0.f, 0.f, 0.f, 0.f};
      kred(qk, QR, KR);
      s_Q[r][t0] = (r >= t0 && r < TT) ? qk[0] * __expf(Gr - Gt0) : 0.f;
      s_Q[r][t1] = (r >= t1 && r < TT) ? qk[1] * __expf(Gr - Gt1) : 0.f;
    } else {
      // (t = r, j = 8 jt + 2q + i): raw product x e^{G_t - G_j}, j < pn
#pragma unroll
      for (int jt = 0; jt < 2; ++jt) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if (warp == 1) kred(acc, KR, KH[jt]); else kred(acc, QR, KH[jt]);
        F2 cf;
        cf.x = hj[2 * jt] ? acc[0] * __expf(Gr - gh[2 * jt]) : 0.f;
        cf.y = hj[2 * jt + 1] ? acc[1] * __expf(Gr - gh[2 * jt + 1]) : 0.f;
        s_coef[warp - 1][jt][lane] = cf;
      }
    }
    __syncthreads();

    const float ik0 = s_inv[0][t0], ik1 = s_inv[0][t1], iq0 = s_inv[1][t0], iq1 = s_inv[1][t1];
    const float ikr = s_inv[0][r], iqr = s_inv[1][r];
    // history of this step's tokens: normalized keys (one value head per key head, warp 0) and G (warp 1)
    if (real && warp == 0 && (hv % GRP) == 0 && r < TT) {
      u16* wk = old_k + wb * (HIST * HK * KD) + (long long)(w_off + r) * (HK * KD) + hk * KD + 8 * q;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        U4 v4;
        v4.x = pack2(bflo(KR[c].x) * ikr, bfhi(KR[c].x) * ikr);
        v4.y = pack2(bflo(KR[c].y) * ikr, bfhi(KR[c].y) * ikr);
        v4.z = pack2(bflo(KR[c].z) * ikr, bfhi(KR[c].z) * ikr);
        v4.w = pack2(bflo(KR[c].w) * ikr, bfhi(KR[c].w) * ikr);
        st16(wk + 32 * c, v4);
      }
    }
    if (real && warp == 1 && lane < TT) old_G[(wb * HV + hv) * HIST + w_off + lane] = is_write ? gl : Gv;

    // history outputs: uh^T (16 x 16) x coeff^T (16 x 8), coefficient = raw x decay x (1 / |row t|) in bf16
    u32 bk0, bk1, bq0, bq1;
    {
      const F2 ck0 = s_coef[0][0][lane], ck1 = s_coef[0][1][lane];
      const F2 cq0 = s_coef[1][0][lane], cq1 = s_coef[1][1][lane];
      bk0 = pack2(ck0.x * ikr, ck0.y * ikr);
      bk1 = pack2(ck1.x * ikr, ck1.y * ikr);
      bq0 = pack2(cq0.x * iqr, cq0.y * iqr);
      bq1 = pack2(cq1.x * iqr, cq1.y * iqr);
    }
    float KHI[2][4], QHI[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int i = 0; i < 4; ++i) KHI[m][i] = QHI[m][i] = 0.f;
      mma16816(KHI[m], UA[m][0], UA[m][1], UA[m][2], UA[m][3], bk0, bk1);
      mma16816(QHI[m], UA[m][0], UA[m][1], UA[m][2], UA[m][3], bq0, bq1);
    }
    const float eGk0 = eG0 * ik0, eGk1 = eG1 * ik1, eGq0 = eG0 * iq0, eGq1 = eG1 * iq1;
    // rhs = beta (v - e^G kh0 - k_hist); C layout per m-tile: [0] (h0, t0), [1] (h0, t1), [2] (h1, t0), [3] (h1, t1)
    float rhs[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      rhs[m][0] = bt0 * (vv[m][0] - eGk0 * KS[m][0] - KHI[m][0]);
      rhs[m][1] = bt1 * (vv[m][1] - eGk1 * KS[m][1] - KHI[m][1]);
      rhs[m][2] = bt0 * (vv[m][2] - eGk0 * KS[m][2] - KHI[m][2]);
      rhs[m][3] = bt1 * (vv[m][3] - eGk1 * KS[m][3] - KHI[m][3]);
    }

    // forward substitution (I + A) U = rhs per value column; every lane of a quad solves its four columns
    float U[4][8];
    {
      float R[4][8];
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int src = (lane & ~3) | qq;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            R[2 * m + h][2 * qq] = __shfl_sync(FULL, rhs[m][2 * h], src);
            R[2 * m + h][2 * qq + 1] = __shfl_sync(FULL, rhs[m][2 * h + 1], src);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float u[4] = {0.f, 0.f, 0.f, 0.f};
        if (i < TT) {
#pragma unroll
          for (int col = 0; col < 4; ++col) u[col] = R[col][i];
#pragma unroll
          for (int s = 0; s < i; ++s) {
            const float a = s_A[i][s];
#pragma unroll
            for (int col = 0; col < 4; ++col) u[col] -= a * U[col][s];
          }
        }
#pragma unroll
        for (int col = 0; col < 4; ++col) U[col][i] = u[col];
      }
    }
    // intra-chunk output of this lane's tokens: q_new[v, t] = sum_{s <= t} inv_q[t] inv_k[s] Qraw[t][s] U[v][s]
    float qn[2][4];
    {
      float c0[8], c1[8];
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const float iks = s < TT ? s_inv[0][s] : 0.f;
        c0[s] = s < TT ? s_Q[t0][s] * iks * iq0 : 0.f;
        c1[s] = s < TT ? s_Q[t1][s] * iks * iq1 : 0.f;
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float a0 = 0.f, a1 = 0.f;
#pragma unroll
          for (int s = 0; s < TT; ++s) {
            a0 += c0[s] * U[2 * m + h][s];
            a1 += c1[s] * U[2 * m + h][s];
          }
          qn[m][2 * h] = a0;
          qn[m][2 * h + 1] = a1;
        }
      }
    }
    float ov[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      ov[m][0] = eGq0 * QS[m][0] + QHI[m][0] + qn[m][0];
      ov[m][1] = eGq1 * QS[m][1] + QHI[m][1] + qn[m][1];
      ov[m][2] = eGq0 * QS[m][2] + QHI[m][2] + qn[m][2];
      ov[m][3] = eGq1 * QS[m][3] + QHI[m][3] + qn[m][3];
    }

    // history of this step's tokens: U
    {
      u16* wu = old_u + wb * (HIST * HV * VD) + hv * VD + vb + r;
#pragma unroll
      for (int col = 0; col < 4; ++col) {
        const int off = 16 * (col >> 1) + 8 * (col & 1);
        if (ok0) wu[(long long)(w_off + t0) * (HV * VD) + off] = f2bf(pick8(U[col], t0));
        if (ok1) wu[(long long)(w_off + t1) * (HV * VD) + off] = f2bf(pick8(U[col], t1));
      }
    }

    // gated RMSNorm over the 128 columns of the head (bf16-rounded replay output, as the norm kernel reads it)
    float xb[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int i = 0; i < 4; ++i) xb[m][i] = bfround(ov[m][i]);
#if STORE_O
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int t = (i & 1) ? t1 : t0;
        const int vs = vb + 16 * m + 8 * (i >> 1) + r;
        if (real && t < TT) o_out[((row0 + t) * HV + hv) * VD + vs] = f2bf(ov[m][i]);
      }
#endif
    float p0 = xb[0][0] * xb[0][0] + xb[0][2] * xb[0][2] + xb[1][0] * xb[1][0] + xb[1][2] * xb[1][2];
    float p1 = xb[0][1] * xb[0][1] + xb[0][3] * xb[0][3] + xb[1][1] * xb[1][1] + xb[1][3] * xb[1][3];
#pragma unroll
    for (int msk = 4; msk < 32; msk <<= 1) {
      p0 += __shfl_xor_sync(FULL, p0, msk);
      p1 += __shfl_xor_sync(FULL, p1, msk);
    }
    if (r == 0) {
      s_norm[warp][t0] = p0;
      s_norm[warp][t1] = p1;
    }
    __syncthreads();
    float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      ss0 += s_norm[w][t0];
      ss1 += s_norm[w][t1];
    }
    const float rstd0 = rsqrt_ap(ss0 * (1.f / VD) + eps);
    const float rstd1 = rsqrt_ap(ss1 * (1.f / VD) + eps);
    u8* yr0 = y + (row0 + t0) * y_stride + hv * VD + vb + r;
    u8* yr1 = yr0 + y_stride;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int h = i >> 1;
        float yv = xb[m][i] * ((i & 1) ? rstd1 : rstd0) * nw[m][h];
        const float zv = zz[m][i];
        yv *= zv * __fdividef(1.f, 1.f + __expf(-zv));
        const bool ok = (i & 1) ? ok1 : ok0;
        const int off = 16 * m + 8 * h;
        if (ok) {
#if OUT_FP8
          ((i & 1) ? yr1 : yr0)[off] = f2e4m3(bfround(yv) * rscale);
#else
          reinterpret_cast<u16*>(y)[(row0 + ((i & 1) ? t1 : t0)) * y_stride + hv * VD + vb + r + off] = f2bf(yv);
#endif
        }
      }
    }
  }  // pass
#if TRIGGER == 3
  __syncthreads();
  pdl_launch();
#endif
}
"""


class _NvrtcKernel:
    """One compiled instantiation of the core kernel (NVRTC -> cubin -> driver module)."""

    def __init__(self, defines: dict, device: int):
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc

        self._cu = cu
        torch.cuda.init()
        if int(self._check(cu.cuCtxGetCurrent())) == 0:
            # No context on this thread yet: the device's primary context (the one the CUDA runtime / torch use).
            pctx = self._check(cu.cuDevicePrimaryCtxRetain(self._check(cu.cuDeviceGet(device))))
            self._check(cu.cuCtxSetCurrent(pctx))
        src = "".join(f"#define {k} {v}\n" for k, v in defines.items()) + _SRC
        prog = self._check(nvrtc.nvrtcCreateProgram(src.encode(), b"q35_gdn_core.cu", 0, [], []))
        major, minor = torch.cuda.get_device_capability()
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode(), b"-std=c++17", b"-lineinfo"]
        if nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            size = self._check(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(
                "q35_gdn_core: NVRTC compile failed:\n" + log.decode(errors="replace")
            )
        cubin = b" " * self._check(nvrtc.nvrtcGetCUBINSize(prog))
        self._check(nvrtc.nvrtcGetCUBIN(prog, cubin))
        ptx = b" " * self._check(nvrtc.nvrtcGetPTXSize(prog))
        self._check(nvrtc.nvrtcGetPTX(prog, ptx))
        self.ptx = ptx.rstrip(b"\x00 ").decode()
        nvrtc.nvrtcDestroyProgram(prog)
        self.cubin = cubin
        self.module = self._check(cu.cuModuleLoadData(cubin))
        self.fn = self._check(cu.cuModuleGetFunction(self.module, b"q35_gdn_core"))

    def _check(self, res):
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc

        err = res[0]
        ok = (
            err == cu.CUresult.CUDA_SUCCESS
            if isinstance(err, cu.CUresult)
            else err == nvrtc.nvrtcResult.NVRTC_SUCCESS
        )
        if not ok:
            raise RuntimeError(f"q35_gdn_core: {err}")
        return res[1] if len(res) == 2 else res[1:]

    def attr(self, which: str) -> int:
        cu = self._cu
        return self._check(cu.cuFuncGetAttribute(getattr(cu.CUfunction_attribute, which), self.fn))

    def launch(self, grid: int, block: int, args, pdl: bool) -> None:
        cu = self._cu
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block, 1, 1
        cfg.sharedMemBytes = 0
        cfg.hStream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        attrs = []
        if pdl:
            a = cu.CUlaunchAttribute()
            a.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            a.value.programmaticStreamSerializationAllowed = 1
            attrs.append(a)
        cfg.attrs = attrs
        cfg.numAttrs = len(attrs)
        ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(c) for c in args])
        self._check(cu.cuLaunchKernelEx(cfg, self.fn, ctypes.addressof(ptrs), 0))


_LOCK = threading.Lock()
_KERNELS: dict = {}


def _kernel(
    device: int,
    t: int,
    hk: int,
    hv: int,
    out_fp8: bool,
    store_o: bool,
    trigger: int,
    warm: bool = False,
) -> _NvrtcKernel:
    key = (device, t, hk, hv, bool(out_fp8), bool(store_o), int(trigger)) + (
        ("warm",) if warm else ()
    )
    kern = _KERNELS.get(key)
    if kern is None:
        with _LOCK, torch.cuda.device(device):
            kern = _KERNELS.get(key)
            if kern is None:
                kern = _NvrtcKernel(
                    dict(
                        TT=t,
                        HK=hk,
                        HV=hv,
                        OUT_FP8=int(out_fp8),
                        STORE_O=int(store_o),
                        TRIGGER=int(trigger),
                        WARM=int(warm),
                    ),
                    device,
                )
                _KERNELS[key] = kern
    return kern


def q35_gdn_core_ready(device: int, t: int, hk: int, hv: int, out_fp8: bool, trigger: int) -> bool:
    """Whether the instantiation can be launched now: compiled already, or compilable (not inside a stream capture,
    where loading a module is not allowed)."""
    key = (device, t, hk, hv, bool(out_fp8), False, int(trigger))
    return key in _KERNELS or not torch.cuda.is_current_stream_capturing()


def q35_gdn_core_supported(
    num_decodes: int,
    draft_token_num: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    history_size: int,
    ssm_dtype: torch.dtype,
) -> bool:
    """Shapes the CUDA core kernel handles (Qwen3.5 TP8 / TP16 decode at T <= 8); others keep the Triton kernel."""
    return (
        2 <= draft_token_num <= Q35_GDN_CORE_MAX_T
        and history_size == 16
        and head_k_dim == 128
        and head_v_dim == 128
        and num_v_heads % num_k_heads == 0
        and ssm_dtype == torch.bfloat16
        and num_decodes >= 1
    )


def _ptr(t: Optional[torch.Tensor]) -> ctypes.c_uint64:
    return ctypes.c_uint64(0 if t is None else t.data_ptr())


def q35_gdn_core_update(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    state_indices: torch.Tensor,
    old_u: torch.Tensor,
    old_k: torch.Tensor,
    old_G: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    history_size: int,
    num_decodes: int,
    draft_token_num: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float = 1e-6,
    fp8_scale: Optional[torch.Tensor] = None,
    norm_out: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = True,
    trigger: int = 2,
    warm: bool = False,
) -> torch.Tensor:
    """Cached replay + gated RMSNorm (+ FP8) of the MTP target-verify step on convolved q|k|v rows.

    mixed_qkv: [N*T, >= conv_dim] convolved q|k|v rows (unit column stride, 16-byte aligned rows); a, b: [N*T, HV]
    row-strided views with a common row stride; z: [N*T, HV, V] token-major (z.stride(1) == V); ssm_states
    [slots, HV, V, K] BF16; old_u / old_k / old_G: the layer's replay history caches (contiguous).
    Returns the gated RMSNorm output [N*T, HV*V] (FP8 E4M3 when ``fp8_scale`` is given, else the input dtype).
    ``output``: optional [N, T, HV, V] buffer that also receives the replay output o (checks only).
    ``trigger``: 2 = release the dependents at entry, 3 = after the final CTA barrier, 0 = at exit.
    ``warm``: run the post-wait path once before the wait on zero inputs (instruction-cache warm-up).
    """
    N, T = num_decodes, draft_token_num
    H, HV, K, V = num_k_heads, num_v_heads, head_k_dim, head_v_dim
    conv_dim = 2 * H * K + HV * V
    assert (
        K == 128
        and V == 128
        and history_size == 16
        and 2 <= T <= Q35_GDN_CORE_MAX_T
        and HV % H == 0
    )
    assert (
        mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.shape[0] == N * T
        and mixed_qkv.shape[1] >= conv_dim
    )
    assert (
        mixed_qkv.stride(1) == 1 and mixed_qkv.stride(0) % 8 == 0 and mixed_qkv.data_ptr() % 16 == 0
    )
    assert (
        a.shape == (N * T, HV) and b.shape == (N * T, HV) and a.stride(1) == 1 and b.stride(1) == 1
    )
    assert a.stride(0) == b.stride(0)
    assert z.shape == (N * T, HV, V) and z.stride(2) == 1 and z.stride(1) == V
    assert ssm_states.dtype == torch.bfloat16
    s0, s1, s2, s3 = ssm_states.stride()
    assert s3 == 1 and s2 == K and s1 == V * K
    for name, tensor in (("old_u", old_u), ("old_k", old_k), ("old_G", old_G)):
        assert tensor.is_contiguous(), f"{name} must be contiguous"
    assert (
        norm_weight.numel() == V
        and norm_weight.is_contiguous()
        and norm_weight.dtype == torch.bfloat16
    )
    assert state_indices.dtype == torch.int32 and state_indices.is_contiguous()
    out_fp8 = fp8_scale is not None
    out_dtype = torch.float8_e4m3fn if out_fp8 else mixed_qkv.dtype
    if norm_out is None:
        norm_out = torch.empty(N * T, HV * V, dtype=out_dtype, device=mixed_qkv.device)
    else:
        assert (
            norm_out.dtype == out_dtype
            and norm_out.shape == (N * T, HV * V)
            and norm_out.stride(1) == 1
        )
    if output is not None:
        assert (
            output.is_contiguous()
            and output.numel() == N * T * HV * V
            and output.dtype == torch.bfloat16
        )
    if not launch_with_pdl:
        trigger = 0
    kern = _kernel(
        mixed_qkv.device.index
        if mixed_qkv.device.index is not None
        else torch.cuda.current_device(),
        T,
        H,
        HV,
        out_fp8,
        output is not None,
        trigger,
        warm=warm,
    )
    args = [
        _ptr(mixed_qkv),
        ctypes.c_int32(mixed_qkv.stride(0)),
        _ptr(z),
        ctypes.c_int32(z.stride(0)),
        _ptr(a),
        _ptr(b),
        ctypes.c_int32(a.stride(0)),
        _ptr(A_log),
        _ptr(dt_bias),
        _ptr(ssm_states),
        ctypes.c_int64(s0),
        _ptr(state_indices),
        _ptr(old_u),
        _ptr(old_k),
        _ptr(old_G),
        _ptr(cache_buf_idx),
        _ptr(prev_num_accepted_tokens),
        _ptr(norm_weight),
        _ptr(fp8_scale),
        ctypes.c_float(norm_eps),
        ctypes.c_float(K**-0.5),
        _ptr(norm_out),
        ctypes.c_int32(norm_out.stride(0)),
        _ptr(output),
    ]
    kern.launch(N * HV, 128, args, pdl=launch_with_pdl)
    return norm_out
