# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 decode routing + TRTLLM-Gen routing tables in one 8-block cluster (NVRTC, first
eager call); the CuTe DSL route kernel's bits except the row order inside an expert's block."""

from __future__ import annotations

import ctypes
import os
import threading
from typing import Optional

import torch

from tensorrt_llm._utils import get_sm_version

MAX_TOKENS = 8  # a block per token in the 8-block cluster
NUM_EXPERTS = 384
TOP_K = 6
MAX_TILES = 3
_CLUSTER = 8
_THREADS = 384

_SRC = r"""
#include <cuda_bf16.h>

#define CLUSTER 8
#define THREADS 384                 // one thread per expert
#define NWARPS (THREADS / 32)
#define NEXP 384
#define TOPK 6
#define KPAD 512                    // keys order equal selections by KPAD - 1 - expert: the lower expert first
#define NCAND (NWARPS * TOPK)       // every warp's top 6
#define MAX_ROWS CLUSTER            // a token per block
#define MAX_PAIRS (MAX_ROWS * TOPK)
#define KEY_NONE ((long long)0x8000000000000000ll)
#define FULL 0xFFFFFFFFu

namespace route {

__device__ __forceinline__ float mul_rn(float a, float b) {
  float r;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float add_rn(float a, float b) {
  float r;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float div_rn(float a, float b) {
  float r;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

__device__ __forceinline__ float rcp_approx_ftz(float v) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

__device__ __forceinline__ float abs_f32(float v) {
  float r;
  asm("abs.f32 %0, %1;" : "=f"(r) : "f"(v));
  return r;
}

__device__ __forceinline__ float max_f32(float a, float b) {
  float r;
  asm("max.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

// sqrt(softplus(l)), threshold 20: libdevice's expf / log1pf instruction sequences (flush to zero, no contraction),
// branch-free, then sqrt.rn.
__device__ __forceinline__ float route_score(float l) {
  float s;
  asm("{\n\t"
      ".reg .pred p1, p2, p3, p4;\n\t"
      ".reg .b32 c1, c2, c3, c4, c5, c6, t5, t6, t9, t10, t11, t13, t15, t16, t17, e, t19, t20, t21, t22, t23, t26;\n\t"
      ".reg .b32 t27, t28, t29, t32, t34, t36, t38, t40, t42, t44, t46, t47, t48, lp, t51, t52, t53, sp;\n\t"
      "mov.b32 c1, 0f3F000000;\n\t"
      "mov.b32 c2, 0f3BBB989D;\n\t"
      "fma.rn.ftz.f32 t5, %1, c2, c1;\n\t"
      "cvt.ftz.sat.f32.f32 t6, t5;\n\t"
      "mov.b32 c3, 0f4B400001;\n\t"
      "mov.b32 c4, 0f437C0000;\n\t"
      "fma.rm.ftz.f32 t9, t6, c4, c3;\n\t"
      "add.rn.ftz.f32 t10, t9, 0fCB40007F;\n\t"
      "neg.ftz.f32 t11, t10;\n\t"
      "mov.b32 c5, 0f3FB8AA3B;\n\t"
      "fma.rn.ftz.f32 t13, %1, c5, t11;\n\t"
      "mov.b32 c6, 0f32A57060;\n\t"
      "fma.rn.ftz.f32 t15, %1, c6, t13;\n\t"
      "shl.b32 t16, t9, 23;\n\t"
      "ex2.approx.ftz.f32 t17, t15;\n\t"
      "mul.rn.ftz.f32 e, t17, t16;\n\t"
      "mov.b32 c1, 0f3F800000;\n\t"
      "add.rz.ftz.f32 t19, e, c1;\n\t"
      "add.s32 t20, t19, -1061158912;\n\t"
      "and.b32 t21, t20, -8388608;\n\t"
      "sub.s32 t22, e, t21;\n\t"
      "sub.s32 t23, 1082130432, t21;\n\t"
      "mov.b32 c1, 0fBF800000;\n\t"
      "mov.b32 c2, 0f3E800000;\n\t"
      "fma.rn.ftz.f32 t26, c2, t23, c1;\n\t"
      "add.rn.ftz.f32 t27, t26, t22;\n\t"
      "cvt.rn.f32.s32 t28, t21;\n\t"
      "mul.rn.ftz.f32 t29, t28, 0f34000000;\n\t"
      "mov.b32 c1, 0f3DD80012;\n\t"
      "mov.b32 c2, 0fBD39BF78;\n\t"
      "fma.rn.ftz.f32 t32, c2, t27, c1;\n\t"
      "mov.b32 c1, 0fBE0778E0;\n\t"
      "fma.rn.ftz.f32 t34, t32, t27, c1;\n\t"
      "mov.b32 c1, 0f3E146475;\n\t"
      "fma.rn.ftz.f32 t36, t34, t27, c1;\n\t"
      "mov.b32 c1, 0fBE2A68DD;\n\t"
      "fma.rn.ftz.f32 t38, t36, t27, c1;\n\t"
      "mov.b32 c1, 0f3E4CAF9E;\n\t"
      "fma.rn.ftz.f32 t40, t38, t27, c1;\n\t"
      "mov.b32 c1, 0fBE800042;\n\t"
      "fma.rn.ftz.f32 t42, t40, t27, c1;\n\t"
      "mov.b32 c1, 0f3EAAAAE6;\n\t"
      "fma.rn.ftz.f32 t44, t42, t27, c1;\n\t"
      "mov.b32 c1, 0fBF000000;\n\t"
      "fma.rn.ftz.f32 t46, t44, t27, c1;\n\t"
      "mul.rn.ftz.f32 t47, t27, t46;\n\t"
      "fma.rn.ftz.f32 t48, t47, t27, t27;\n\t"
      "mov.b32 c1, 0f3F317218;\n\t"
      "fma.rn.ftz.f32 lp, t29, c1, t48;\n\t"
      "setp.lt.u32 p2, e, 2139095040;\n\t"
      "setp.gt.s32 p3, e, -1082130432;\n\t"
      "mov.b32 c1, 0f7F800000;\n\t"
      "fma.rn.ftz.f32 t51, e, c1, c1;\n\t"
      "selp.f32 t52, t51, lp, p3;\n\t"
      "setp.eq.ftz.f32 p4, e, 0f00000000;\n\t"
      "selp.f32 t53, 0f80000000, t52, p4;\n\t"
      "selp.f32 lp, lp, t53, p2;\n\t"
      "setp.gt.ftz.f32 p1, %1, 0f41A00000;\n\t"
      "selp.f32 sp, %1, lp, p1;\n\t"
      "sqrt.rn.f32 %0, sp;\n\t"
      "}"
      : "=f"(s)
      : "f"(l));
  return s;
}

// Float bits in signed-integer order (NaN above +inf, as the route kernel's keys).
__device__ __forceinline__ int ordered_bits(float v) {
  const int bits = __float_as_int(v);
  return bits < 0 ? (bits ^ 0x7FFFFFFF) : bits;
}

__device__ __forceinline__ unsigned short to_bf16_rn(float v) {
  unsigned short b;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(v));
  return b;
}

// One 16-element NVFP4 block of BF16 inputs (fp4_quantize: E2M1 codes, element 2i in the low nibble of byte i, and the
// UE4M3 block scale).
__device__ __forceinline__ void quant16(uint4 lo, uint4 hi, float global_scale, uint2* codes, unsigned char* sf) {
  const float rcp6 = rcp_approx_ftz(6.0f);
  const float rcp_gs = rcp_approx_ftz(mul_rn(1.0f, global_scale));
  const unsigned w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  float v[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[2 * i] = __uint_as_float(w[i] << 16);
    v[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
  }
  float vmax = abs_f32(v[0]);
#pragma unroll
  for (int i = 1; i < 16; ++i) vmax = max_f32(vmax, abs_f32(v[i]));
  const float s = mul_rn(global_scale, mul_rn(vmax, rcp6));
  unsigned short s2;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(s2) : "f"(0.0f), "f"(s));
  const unsigned char s8 = (unsigned char)(s2 & 0xFF);
  float sfv;
  asm("{\n\t.reg .b32 h2;\n\t.reg .b16 l16, h16;\n\t"
      "cvt.rn.f16x2.e4m3x2 h2, %1;\n\t"
      "mov.b32 {l16, h16}, h2;\n\t"
      "cvt.f32.f16 %0, l16;\n\t}"
      : "=f"(sfv)
      : "h"((unsigned short)s8));
  float out_scale;
  asm("{\n\t.reg .pred nz;\n\t.reg .f32 r;\n\t"
      "setp.neu.f32 nz, %1, 0f00000000;\n\t"
      "mul.rn.f32 r, %2, %3;\n\t"
      "rcp.approx.ftz.f32 r, r;\n\t"
      "selp.f32 %0, r, 0f00000000, nz;\n\t}"
      : "=f"(out_scale)
      : "f"(vmax), "f"(sfv), "f"(rcp_gs));
  unsigned q[2] = {0u, 0u};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float a = mul_rn(v[2 * j], out_scale), b = mul_rn(v[2 * j + 1], out_scale);
    unsigned byte;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(byte) : "f"(b), "f"(a));
    q[j >> 2] |= byte << (8 * (j & 3));
  }
  *codes = make_uint2(q[0], q[1]);
  *sf = s8;
}

}  // namespace route

__device__ __forceinline__ unsigned smem_u32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned cluster_addr(unsigned local, unsigned rank) {
  unsigned r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(rank));
  return r;
}

// The largest of the warp's keys (selection bits, then the lower expert), in every lane.
__device__ __forceinline__ long long warp_max_key(long long key) {
  const int hi = (int)(key >> 32);
  const int top = __reduce_max_sync(FULL, hi);
  const unsigned low = __reduce_max_sync(FULL, hi == top ? (unsigned)(key & 0xFFFFFFFFll) : 0u);
  return ((long long)top << 32) | low;
}

// The routing tables by one warp (an expert's picks fit one CTA: m <= every tile): a pick per lane and pass, s_cnt
// [NEXP] and s_bits [NEXP / 32] zeroed. A pick's row is its expert's pick count before it (any order is valid); this
// rank's experts in use get CTAs in expert order.
__device__ __forceinline__ void tables_warp(const int* s_pick, int* s_cnt, unsigned* s_bits, int m, int* tables,
                                            int width, int num_tiles, int tile0, int tile1, int tile2,
                                            int local_start, int num_local, int dyn_rank, int dyn_size) {
  const int lane = threadIdx.x & 31, n = m * TOPK;
  int e[2], rank[2] = {0, 0};
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    e[h] = lane + 32 * h < n ? s_pick[lane + 32 * h] : -1;
    if (e[h] >= 0) {
      rank[h] = atomicAdd(&s_cnt[e[h]], 1);
      atomicOr(&s_bits[e[h] >> 5], 1u << (e[h] & 31));
    }
  }
  __syncwarp();
  // Lane w holds used-expert bits w; excl: experts in use below them.
  const unsigned bits = lane < NEXP / 32 ? s_bits[lane] : 0u;
  int incl = __popc(bits);
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int t = __shfl_up_sync(FULL, incl, o);
    incl += lane >= o ? t : 0;
  }
  const int excl = incl - __popc(bits);
  const int total = __shfl_sync(FULL, incl, 31);
  // Experts in use below x, for any x in [0, NEXP]; every lane shuffles.
  auto used_before = [&](int x) {
    const int w = x >> 5;
    const unsigned below = __shfl_sync(FULL, bits, w) & ((1u << (x & 31)) - 1u);
    return __shfl_sync(FULL, excl, w) + __popc(below);
  };
  // As the CuTe kernel's home_first_owner for two ranks: a rank keeps its range's first ceil(total / 2) experts in use.
  const int lower = used_before(NEXP / 2), cap = (total + 1) >> 1;
  const int mine_lower = dyn_rank == 0 ? min(lower, cap) : max(lower - cap, 0);
  const int mine_upper = dyn_rank == 1 ? min(total - lower, cap) : max(total - lower - cap, 0);
  const int first = used_before(local_start);
  const int ctas = dyn_size == 2 ? mine_lower + mine_upper : used_before(local_start + num_local) - first;
  bool mine[2];
  int cta[2], cnt[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int below = used_before(max(e[h], 0));
    if (dyn_size == 2) {
      const int home = e[h] >= NEXP / 2, idx = below - (home ? lower : 0), moved = idx >= cap;
      mine[h] = e[h] >= 0 && (moved ? 1 - home : home) == dyn_rank;
      cta[h] = (home ? mine_lower : 0) + (moved ? idx - cap : idx);
    } else {
      mine[h] = e[h] >= local_start && e[h] < local_start + num_local;
      cta[h] = below - first;
    }
    cnt[h] = mine[h] ? s_cnt[e[h]] : 0;
  }
  for (int t = 0; t < num_tiles; ++t) {
    const int tile = t == 0 ? tile0 : (t == 1 ? tile1 : tile2);
    int* const e2p = tables + (long long)t * 5 * width;
    int* const p2t = e2p + width;
    int* const cta_expert = e2p + 2 * width;
    int* const cta_limit = e2p + 3 * width;
    int* const totals = e2p + 4 * width;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int i = lane + 32 * h;
      if (i < n) {
        if (mine[h]) {
          const int permuted = cta[h] * tile + rank[h];
          e2p[i] = permuted;
          p2t[permuted] = i / TOPK;
          if (rank[h] == 0) {
            cta_expert[cta[h]] = e[h] - local_start;
            cta_limit[cta[h]] = cta[h] * tile + cnt[h];
          }
        } else {
          e2p[i] = -1;
        }
      }
    }
    if (lane == 0) {
      totals[0] = ctas;
      totals[1] = ctas * tile;
    }
  }
}

// tables: [num_tiles, 5, width] (expanded -> permuted, permuted -> token, CTA -> expert, CTA -> row limit,
// (CTAs, padded rows)), experts numbered from local_start; null routes (and quantizes) only. With dyn_size == 2 both
// ranks hold every expert and the tables cover rank dyn_rank's share of the used ones (the CuTe kernel's
// home_first_owner). Static inputs (bias, global scale) are read before the dependency wait, the logits and x after it.
extern "C" __global__ void __cluster_dims__(CLUSTER, 1, 1) __launch_bounds__(THREADS, 1) dsv4_route_one(
    const float* logits, const float* __restrict__ bias, const __nv_bfloat16* x,
    const float* __restrict__ global_scale, int* ids, void* weights_out, unsigned char* xq, unsigned char* xsf,
    float rsf, long long stride_x, int m, int hidden, int weights_bf16, int* tables, int width, int num_tiles,
    int tile0, int tile1, int tile2, int local_start, int num_local, int dyn_rank, int dyn_size) {
  __shared__ float s_sc[NEXP];
  __shared__ long long s_ck[NCAND];
  __shared__ __align__(8) unsigned long long s_bar;  // block 0: completes once every pick has landed
  __shared__ int s_pick[MAX_PAIRS];
  __shared__ int s_cnt[NEXP];
  __shared__ unsigned s_bits[NEXP / 32];
  const int cta = blockIdx.x;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const bool with_tables = tables != nullptr;
  const int n = m * TOPK;
  const float my_bias = bias[tid];
  const float gs = xq != nullptr ? *global_scale : 0.0f;
  if (with_tables) {
    if (cta == 0) {
      s_cnt[tid] = 0;
      if (tid < NEXP / 32) s_bits[tid] = 0;
      if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&s_bar)));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&s_bar)), "r"(n * 4)
                     : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
  }
#ifdef USE_PDL
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  // The expert GEMM that follows reads nothing before its own dependency wait: it may launch now.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int row = cta;
  if (row < m) {
    const float sc = route::route_score(logits[(long long)row * NEXP + tid]);
    s_sc[tid] = sc;
    long long key = ((long long)route::ordered_bits(route::add_rn(sc, my_bias)) << 32) | (unsigned)(KPAD - 1 - tid);
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const long long win = warp_max_key(key);
      if (lane == 0) s_ck[warp * TOPK + k] = win;
      key = key == win ? KEY_NONE : key;
    }
    __syncthreads();
    if (warp == 0) {
      // Lane l holds candidates l, l + 32, l + 64 in descending order; each maximum's winner drops its first.
      long long c3[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) c3[j] = lane + 32 * j < NCAND ? s_ck[lane + 32 * j] : KEY_NONE;
#pragma unroll
      for (int i = 1; i < 3; ++i) {
#pragma unroll
        for (int j = i; j > 0; --j) {
          const long long a = c3[j - 1], b = c3[j];
          c3[j - 1] = a > b ? a : b;
          c3[j] = a > b ? b : a;
        }
      }
      int ek[TOPK];
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        const long long win = warp_max_key(c3[0]);
        ek[k] = KPAD - 1 - (int)(unsigned)(win & 0xFFFFFFFFll);
        if (c3[0] == win) {
          c3[0] = c3[1];
          c3[1] = c3[2];
          c3[2] = KEY_NONE;
        }
      }
      float p[TOPK];
#pragma unroll
      for (int k = 0; k < TOPK; ++k) p[k] = s_sc[ek[k]];
      int e = ek[0];
      float sk = p[0];
#pragma unroll
      for (int k = 1; k < TOPK; ++k) {
        if (lane == k) {
          e = ek[k];
          sk = p[k];
        }
      }
      if (lane < TOPK) {
        // The route kernel's warp sum of the six scores: ((p0 + p4) + (p2 + 0)) + ((p1 + p5) + (p3 + 0)).
        const float total = route::add_rn(route::add_rn(route::add_rn(p[0], p[4]), route::add_rn(p[2], 0.0f)),
                                          route::add_rn(route::add_rn(p[1], p[5]), route::add_rn(p[3], 0.0f)));
        const float w = route::mul_rn(route::div_rn(sk, route::add_rn(total, 1e-20f)), rsf);
        if (with_tables) {
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.s32 [%0], %1, [%2];" ::"r"(
                           cluster_addr(smem_u32(&s_pick[row * TOPK + lane]), 0)),
                       "r"(e), "r"(cluster_addr(smem_u32(&s_bar), 0))
                       : "memory");
        }
        ids[row * TOPK + lane] = e;
        if (weights_bf16) {
          reinterpret_cast<unsigned short*>(weights_out)[row * TOPK + lane] = route::to_bf16_rn(w);
        } else {
          reinterpret_cast<float*>(weights_out)[row * TOPK + lane] = w;
        }
      }
    }
  }
  if (cta > 0) {
    if (xq != nullptr) {
      const int nb = hidden / 16;
      for (int q = (cta - 1) * THREADS + tid; q < m * nb; q += (CLUSTER - 1) * THREADS) {
        const int r = q / nb, b = q - r * nb;
        const uint4* src = reinterpret_cast<const uint4*>(x + r * stride_x + (long long)b * 16);
        const uint4 lo = src[0], hi = src[1];
        uint2 codes;
        unsigned char sf;
        route::quant16(lo, hi, gs, &codes, &sf);
        *reinterpret_cast<uint2*>(xq + (long long)r * (hidden / 2) + b * 8) = codes;
        xsf[(long long)r * nb + b] = sf;
      }
    }
    return;
  }
  if (!with_tables) return;
  // ---- block 0: the tables, from shared memory ----
  asm volatile(
      "{\n\t.reg .pred p;\n\tW_%=:\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n\t"
      "@!p bra W_%=;\n\t}" ::"r"(smem_u32(&s_bar))
      : "memory");
  if (warp == 0) {
    tables_warp(s_pick, s_cnt, s_bits, m, tables, width, num_tiles, tile0, tile1, tile2, local_start, num_local,
                dyn_rank, dyn_size);
  }
}
"""


class _Kernel:
    def __init__(self, device_index: int, pdl: bool):
        from cuda.bindings import driver as cu
        from cuda.bindings import nvrtc

        def check(res):
            err = res[0]
            ok = (
                err == cu.CUresult.CUDA_SUCCESS
                if isinstance(err, cu.CUresult)
                else err == nvrtc.nvrtcResult.NVRTC_SUCCESS
            )
            if not ok:
                raise RuntimeError(f"dsv4_route_one: {err}")
            return res[1] if len(res) == 2 else res[1:]

        self._cu, self._check = cu, check
        prog = check(nvrtc.nvrtcCreateProgram(_SRC.encode(), b"dsv4_route_one.cu", 0, [], []))
        major, minor = torch.cuda.get_device_capability(device_index)
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
        # The architecture-specific target has the E2M1 / E4M3 conversions.
        opts = [
            f"--gpu-architecture=sm_{major}{minor}a".encode(),
            b"-std=c++17",
            f"--include-path={cuda_home}/include".encode(),
        ]
        if pdl:
            opts.append(b"-DUSE_PDL")
        if nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            size = check(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError("dsv4_route_one: NVRTC failed:\n" + log.decode(errors="replace"))
        cubin = b" " * check(nvrtc.nvrtcGetCUBINSize(prog))
        check(nvrtc.nvrtcGetCUBIN(prog, cubin))
        nvrtc.nvrtcDestroyProgram(prog)
        # cuModuleLoadData needs the device's primary context current.
        torch.zeros(1, device=torch.device("cuda", device_index))
        self.module = check(cu.cuModuleLoadData(cubin))
        self.fn = check(cu.cuModuleGetFunction(self.module, b"dsv4_route_one"))
        self.pdl = pdl

    def launch(self, args) -> None:
        cu = self._cu
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = _CLUSTER, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = _THREADS, 1, 1
        cfg.sharedMemBytes = 0
        cfg.hStream = cu.CUstream(torch.cuda.current_stream().cuda_stream)
        cfg.numAttrs = 0
        if self.pdl:
            attr = cu.CUlaunchAttribute()
            attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
            attr.value.programmaticStreamSerializationAllowed = 1
            cfg.attrs = [attr]
            cfg.numAttrs = 1
        ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
        self._check(cu.cuLaunchKernelEx(cfg, self.fn, ctypes.addressof(ptrs), 0))


_kernels: dict[tuple[int, bool], _Kernel] = {}
_lock = threading.Lock()
_unavailable = False


def _kernel(device_index: int, pdl: bool) -> Optional[_Kernel]:
    """The device's kernel, compiled at the first eager call (never inside a graph capture)."""
    global _unavailable
    key = (device_index, pdl)
    kernel = _kernels.get(key)
    if kernel is not None or _unavailable or torch.cuda.is_current_stream_capturing():
        return kernel
    with _lock:
        kernel = _kernels.get(key)
        if kernel is None:
            try:
                kernel = _Kernel(*key)
            except (RuntimeError, ImportError) as e:
                _unavailable = True
                from tensorrt_llm.logger import logger

                logger.warning(f"DeepSeek-V4 one-cluster route kernel unavailable ({e})")
                return None
            _kernels[key] = kernel
    return kernel


def applies(num_tokens: int, num_experts: int, top_k: int, routing_tiles: tuple[int, ...]) -> bool:
    """Whether a routing-table launch of this shape takes the one-cluster kernel."""
    return (
        0 < num_tokens <= MAX_TOKENS
        and num_experts == NUM_EXPERTS
        and top_k == TOP_K
        and 0 < len(routing_tiles) <= MAX_TILES
        and get_sm_version() in (100, 103)
    )


def route_one(
    logits: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    ids: torch.Tensor,
    weights: torch.Tensor,
    tables: torch.Tensor,
    routing_tiles: tuple[int, ...],
    local_start: int,
    num_local: int,
    pdl: bool,
    dynamic_ep: Optional[tuple[int, int]] = None,
    x: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    x_fp4: Optional[torch.Tensor] = None,
    x_sf: Optional[torch.Tensor] = None,
) -> bool:
    """Route (and with ``x`` quantize) as ``dsv4_route`` / ``dsv4_route_nvfp4_quant`` with tables;
    False leaves the call to them (an input layout it does not take, or no compiled kernel)."""
    num_tokens = logits.shape[0]
    if (
        logits.dtype != torch.float32
        or not logits.is_contiguous()
        or bias.dtype != torch.float32
        or not bias.is_contiguous()
        or weights.dtype not in (torch.float32, torch.bfloat16)
        or not tables.is_contiguous()
    ):
        return False
    if x is not None and (
        x.dtype != torch.bfloat16
        or x.stride(1) != 1
        or (x.stride(0) * x.element_size()) % 16
        or x.data_ptr() % 16
        or x.shape[1] % 16
        or global_scale.dtype != torch.float32
    ):
        return False
    kernel = _kernel(logits.device.index, pdl)
    if kernel is None:
        return False

    def ptr(t: Optional[torch.Tensor]) -> ctypes.c_uint64:
        return ctypes.c_uint64(0 if t is None else t.data_ptr())

    tiles = tuple(routing_tiles) + (routing_tiles[-1],) * (MAX_TILES - len(routing_tiles))
    args = [
        ptr(logits), ptr(bias), ptr(x), ptr(global_scale), ptr(ids), ptr(weights), ptr(x_fp4), ptr(x_sf),
        ctypes.c_float(routed_scaling_factor), ctypes.c_int64(0 if x is None else x.stride(0)),
        ctypes.c_int32(num_tokens), ctypes.c_int32(0 if x is None else x.shape[1]),
        ctypes.c_int32(1 if weights.dtype == torch.bfloat16 else 0), ptr(tables),
        ctypes.c_int32(tables.shape[2]), ctypes.c_int32(len(routing_tiles)),
        *(ctypes.c_int32(t) for t in tiles), ctypes.c_int32(local_start), ctypes.c_int32(num_local),
        *(ctypes.c_int32(v) for v in (dynamic_ep or (0, 0))),
    ]  # fmt: skip
    kernel.launch(args)
    return True
