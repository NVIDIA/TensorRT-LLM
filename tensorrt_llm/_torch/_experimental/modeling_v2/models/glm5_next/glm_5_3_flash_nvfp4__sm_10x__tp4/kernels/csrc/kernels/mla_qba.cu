/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// The MLA decode query path in one launch: q_b (q_nope = q_resid x W_qb^T, bf16) and the absorb
// (q_lat[h] = q_nope[h] x w_kc[h], over w_kcT[h] = w_kc[h]^T) for T <= 8 tokens of the local heads
// (q_lora 1536, nope 256, kv_lora 512), instead of gemv_pre(q_b) + the head-batched gemv_pre(absorb):
// one launch and one dependent hop fewer.
//
// One cluster of 4 CTAs per head. CTA c owns the head's q_nope rows [64 c, 64 c + 64) (their q_b weight
// rows, 192 KB, copied into shared memory with one bulk copy per row before the PDL wait: rows padded
// by 16 bytes, conflict-free ldmatrix without a swizzle) and the head's q_lat columns [128 c, 128 c +
// 128) (their w_kcT rows, 64 KB, bulk-copied into the q_b weights' space once the q_b products are
// done, landing during the q_b epilogue and the exchange). After the wait: the q_b products exactly as
// gemv_pre computes them (warp w: k-slice w of every row, mma.sync m16n8k16, the 8 warps' partials
// summed in warp order, rounded to bf16), each CTA's 64 q_nope values per token stored into every
// cluster CTA's q_nope tile (DSMEM), one cluster barrier, then the absorb with the head-batched
// gemv_pre's products and sums (warp w: 16 rows, each of gemv_pre's 8 k-slices into its own
// accumulator, the slices summed in order in registers): BITWISE gemv_pre(q_b) + gemv_pre(absorb).
// 64 CTAs (gemv_pre: 86 + 128), so the GEMVs fit beside the top-k's CTAs (one SM each).
#include <cooperative_groups.h>
#include <cuda.h>
#include <cudaTypedefs.h>

#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "mla.h"
#include "pdl.cuh"
#include "reduce.cuh"

namespace glm {
namespace k {

namespace {
__device__ __forceinline__ uint32_t q_s32_(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
constexpr int kQbaCtas = 4;      // CTAs per head (one cluster)
constexpr int kQbaThreads = 256;  // 8 warps
constexpr int kQbaWarps = kQbaThreads / 32;
constexpr int kQl = 1536, kNope = 256, kKl = kLatent;
constexpr int kQbRows = kNope / kQbaCtas;  // 64 q_nope rows per CTA
constexpr int kAbRows = kKl / kQbaCtas;    // 128 q_lat columns per CTA
constexpr int kRsW = kQl * 2 + 16;         // padded row strides (bytes)
constexpr int kRsK = kNope * 2 + 16;
constexpr int kMT = 8;                     // MMA token columns (gemv_pre's MT for M <= 8)
constexpr int kRedS = kMT + 1;                               // partials' row stride (floats)
constexpr int kOffR1 = kAbRows * kRsK;                       // q_b partials [8][64][kRedS] f32
constexpr int kOffA = kQbRows * kRsW;                        // activations [M][kRsW], then q_nope [M][kRsK]
static_assert(kOffR1 + kQbaWarps * kQbRows * kRedS * 4 <= kOffA, "mla_qba: partials overlap");
static_assert(kQl % (16 * kQbaWarps) == 0 && kNope % (16 * kQbaWarps) == 0, "mla_qba: k-slices");

// (TM) the absorb weights' 4 swizzled boxes at the first 1024-byte aligned offset of the q_b weights' space
__device__ __forceinline__ uint32_t qba_tm_off(const uint8_t* sm) {
  return (1024u - (q_s32_(sm) & 1023u)) & 1023u;
}
static_assert(1023 + 4 * kAbRows * 128 <= kOffR1, "mla_qba: TM absorb boxes overlap the q_b partials");

__host__ __device__ inline size_t qba_smem(int M) {
  return (size_t)kOffA + (size_t)M * (kRsW + kRsK) + 4 * sizeof(uint64_t);
}

__device__ __forceinline__ uint32_t q_s32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void q_ldsm_x4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void q_ldsm_x2(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}
__device__ __forceinline__ void q_mma(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void q_bar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(q_s32(b)), "r"(n));
}
__device__ __forceinline__ void q_bar_expect(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(q_s32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void q_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
          q_s32(dst)),
      "l"(src), "r"(bytes), "r"(q_s32(b))
      : "memory");
}
// 2-D tensor copy (TM form): box {64 columns, 128 rows} of the absorb weights, 128-byte swizzle
__device__ __forceinline__ void q_tmap(void* dst, const CUtensorMap* m, int c0, int c1, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n" ::"r"(
          q_s32(dst)),
      "l"(reinterpret_cast<uint64_t>(m)), "r"(c0), "r"(c1), "r"(q_s32(b))
      : "memory");
}
__device__ __forceinline__ void q_bar_wait(uint64_t* b, uint32_t ph) {
  uint32_t ok = 0;
  while (!ok)
    asm volatile(
        "{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }\n"
        : "=r"(ok)
        : "r"(q_s32(b)), "r"(ph)
        : "memory");
}
// (QX) a 32-bit value to cluster CTA q's shared memory at local address a's offset, completing 4 bytes on that CTA's
// mbarrier at local address mb's offset (no fence: the receiver waits on its mbarrier)
__device__ __forceinline__ void q_st_async(uint32_t a, uint32_t v, uint32_t mb, int q) {
  uint32_t ra, rm;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(a), "r"(q));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rm) : "r"(mb), "r"(q));
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(ra), "r"(v),
               "r"(rm)
               : "memory");
}
__device__ __forceinline__ uint32_t q_pack(float a, float b) {
  return (uint32_t)__bfloat16_as_ushort(__float2bfloat16(a)) |
         ((uint32_t)__bfloat16_as_ushort(__float2bfloat16(b)) << 16);
}
}  // namespace

// QX: the q_nope exchange as st.async stores completing on each receiver's mbarrier
// (expecting the M x 256 values of the head) instead of DSMEM stores and a cluster barrier, whose release half is a
// GPU-scope MEMBAR.ALL.GPU in SASS that waits for the CTA's stores and in-flight absorb copies; the cluster barrier's
// arrive moves to the start (after the mbarriers' init, relaxed) and its wait before the first store (the same values
// in the same places: bitwise)
// TM: the absorb weights as 4 tensor copies (boxes of 64 columns x 128 rows, 128-byte swizzle) issued by one thread
// instead of 128 row copies issued by warps 0-3 (a 32-lane bulk-copy instruction is slow to issue: those warps would
// reach the exchange late); the same values in the same MMA operands: bitwise
// SIG (sparse_mla_attn_sel's q path): every CTA adds 1 to *done once its q_lat stores are visible GPU-wide (each
// thread's fence, the block barrier, one atomic)
// NQ: see QbaNorm (mla.h); the norms after the rows land, the latent rows by CTA (0, 0) after its exchange stores
template <bool QX, bool TM, bool SIG = false, bool NQ = false>
__global__ void __cluster_dims__(kQbaCtas, 1, 1) __launch_bounds__(kQbaThreads, 1)
    mla_qba_kernel(const bf16* __restrict__ qr, int ldq, const bf16* __restrict__ wqb,
                   const bf16* __restrict__ wkcT, bf16* __restrict__ out, int ldo, int M, int late,
                   const __grid_constant__ CUtensorMap tm_kc, unsigned* done, const QbaNorm nq) {
  if (!late) pdl_trigger();
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int c = blockIdx.x, h = blockIdx.y;  // (cluster rank = blockIdx.x: clusters of 4 along x)
  uint8_t* sA = sm + kOffA;
  uint8_t* sQ = sA + (size_t)M * kRsW;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sQ + (size_t)M * kRsK);
  uint64_t* wbar = bars;      // q_b weights
  uint64_t* abar = bars + 1;  // activations
  uint64_t* kbar = bars + 2;  // absorb weights
  uint64_t* xbar = bars + 3;  // (QX) the head's q_nope values from the cluster
  // the CTA's q_b weight rows, before the wait (weights never depend on earlier kernels): warp 0
  if (warp == 0) {
    if (lane == 0) {
      q_bar_init(wbar, 1);
      q_bar_init(abar, 1);
      q_bar_init(kbar, 1);
      if constexpr (QX) q_bar_init(xbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
      q_bar_expect(wbar, (uint32_t)kQbRows * kQl * 2);
      if constexpr (QX) q_bar_expect(xbar, (uint32_t)M * kNope * 2);
    }
    __syncwarp();
    for (int r = lane; r < kQbRows; r += 32)
      q_bulk(sm + (size_t)r * kRsW, wqb + ((size_t)h * kNope + c * kQbRows + r) * kQl, kQl * 2, wbar);
  }
  __syncthreads();  // (the barriers are initialized before any warp waits on them)
  // (QX) the cluster's mbarriers visible before any CTA's first store into them: this arrive's wait comes just before
  // the stores (by then long complete)
  if constexpr (QX) asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  // (NQ) the norms' weights (constants) before the wait: mla_qkv_norm's per-thread elements t + 256 k; CTA (0, 0)'s
  // latent cache rows with the metadata uploaded / final before the step (cm.pre)
  constexpr int kNqQ = NQ ? kQl / kQbaThreads : 1, kNqK = NQ ? kKl / kQbaThreads : 1;
  float nwq[kNqQ], nwk[kNqK];
  int64_t ncrow[kMT];
  const bool nkv = NQ && blockIdx.x == 0 && blockIdx.y == 0;
  if constexpr (NQ) {
#pragma unroll
    for (int k2 = 0; k2 < kNqQ; ++k2) nwq[k2] = __bfloat162float(nq.wq[tid + k2 * kQbaThreads]);
#pragma unroll
    for (int k2 = 0; k2 < kNqK; ++k2) nwk[k2] = nkv ? __bfloat162float(nq.wkv[tid + k2 * kQbaThreads]) : 0.f;
  }
  auto nq_rows = [&]() {
#pragma unroll
    for (int m = 0; m < kMT; ++m) {
      if (m < M && nkv) {
        const int pos = nq.cm.positions[m];
        ncrow[m] = nq.cm.block_tables[(size_t)nq.cm.token_seq[m] * nq.cm.max_blocks + pos / kMlaBlock] *
                       nq.cm.lat_page_rows + pos % kMlaBlock;
      }
    }
  };
  if constexpr (NQ)
    if (nq.cm.pre) nq_rows();
  pdl_wait();
  if (late) pdl_trigger();  // (the next kernel launches once this one's wait is over)
  if (tid == 0) q_bar_expect(abar, (uint32_t)M * kQl * 2);
  if (tid < M) q_bulk(sA + (size_t)tid * kRsW, qr + (size_t)tid * ldq, kQl * 2, abar);
  if constexpr (NQ)
    if (!nq.cm.pre) nq_rows();
  q_bar_wait(abar, 0);
  if constexpr (NQ) {  // (mla_qkv_norm's q part of each row, in place)
    __shared__ float nsh[32];
    for (int m = 0; m < M; ++m) {
      __nv_bfloat16* row = reinterpret_cast<__nv_bfloat16*>(sA + (size_t)m * kRsW);
      float ss = 0.f;
      for (int i = tid; i < kQl; i += kQbaThreads) {
        const float v = __bfloat162float(row[i]);
        ss += v * v;
      }
      ss = block_sum(ss, nsh);
      const float r = rsqrtf(ss / kQl + nq.eps);
#pragma unroll
      for (int k2 = 0; k2 < kNqQ; ++k2) {
        const int i = tid + k2 * kQbaThreads;
        row[i] = __float2bfloat16(__bfloat162float(row[i]) * r * nwq[k2]);
      }
    }
    __syncthreads();  // (the normalized rows before the products' ldmatrix reads)
  }
  q_bar_wait(wbar, 0);
  // q_b: gemv_pre's products (warp w: k-steps [12 w, 12 w + 12) of every row)
  {
    float acc[kQbRows / 16][4];
#pragma unroll
    for (int mi = 0; mi < kQbRows / 16; ++mi) acc[mi][0] = acc[mi][1] = acc[mi][2] = acc[mi][3] = 0.f;
    constexpr int ks_per = kQl / (16 * kQbaWarps);
    const int ks0 = warp * ks_per;
    const uint32_t a0 = q_s32(sA) + ((lane & 7) % M) * kRsW + ((lane >> 3) & 1) * 16;
    const uint32_t w0 = q_s32(sm) + (lane & 15) * kRsW + (lane >> 4) * 16;
#pragma unroll 4
    for (int ks = ks0; ks < ks0 + ks_per; ++ks) {
      uint32_t bfr[2];
      q_ldsm_x2(bfr, a0 + ks * 32);
#pragma unroll
      for (int mi = 0; mi < kQbRows / 16; ++mi) {
        uint32_t af[4];
        q_ldsm_x4(af, w0 + mi * 16 * kRsW + ks * 32);
        q_mma(acc[mi], af, bfr);
      }
    }
    __syncthreads();  // every warp is done with the q_b weights
    // the absorb weight rows into their space (landing during the epilogue and the exchange)
    if (tid == 0) q_bar_expect(kbar, (uint32_t)kAbRows * kNope * 2);
    if constexpr (TM) {
      if (tid == 0) {
#pragma unroll
        for (int b = 0; b < kNope / 64; ++b)
          q_tmap(sm + qba_tm_off(sm) + b * (kAbRows * 128), &tm_kc, b * 64, h * kKl + c * kAbRows, kbar);
      }
    } else if (tid < kAbRows) {
      q_bulk(sm + (size_t)tid * kRsK, wkcT + ((size_t)h * kKl + c * kAbRows + tid) * kNope, kNope * 2, kbar);
    }
    float* red = reinterpret_cast<float*>(sm + kOffR1);
#pragma unroll
    for (int mi = 0; mi < kQbRows / 16; ++mi)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int row = mi * 16 + (lane >> 2) + (e >> 1) * 8, tok = (lane & 3) * 2 + (e & 1);
        red[((size_t)warp * kQbRows + row) * kRedS + tok] = acc[mi][e];
      }
  }
  __syncthreads();
  // q_nope: the 8 warps' partials summed in warp order (gemv_pre's sums), bf16, a pair of rows per
  // thread (the token fastest: few bank conflicts), stored into every cluster CTA's q_nope tile
  {
    const float* red = reinterpret_cast<const float*>(sm + kOffR1);
    if constexpr (QX) asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    for (int i = tid; i < M * (kQbRows / 2); i += kQbaThreads) {
      const int tok = i % M, r = 2 * (i / M);
      float v0 = 0.f, v1 = 0.f;
#pragma unroll
      for (int w = 0; w < kQbaWarps; ++w) {
        v0 += red[((size_t)w * kQbRows + r) * kRedS + tok];
        v1 += red[((size_t)w * kQbRows + r + 1) * kRedS + tok];
      }
      const uint32_t pk = q_pack(v0, v1);
      const uint32_t dst = q_s32(sQ + (size_t)tok * kRsK + (c * kQbRows + r) * 2);
      if constexpr (QX) {
#pragma unroll
        for (int q = 0; q < kQbaCtas; ++q) q_st_async(dst, pk, q_s32(xbar), q);
      } else {
#pragma unroll
        for (int q = 0; q < kQbaCtas; ++q) {
          uint32_t ra;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(dst), "r"(q));
          asm volatile("st.shared::cluster.u32 [%0], %1;\n" ::"r"(ra), "r"(pk) : "memory");
        }
      }
    }
    if constexpr (NQ) {  // (CTA (0, 0), its q_nope values sent: mla_qkv_norm's latent part of each row into the cache)
      if (nkv) {
        __shared__ float nsk[32];
        for (int m = 0; m < M; ++m) {
          const bf16* xr = nq.x_kv + (size_t)m * ldq;
          float xv[kNqK], ss = 0.f;
#pragma unroll
          for (int k2 = 0; k2 < kNqK; ++k2) xv[k2] = __bfloat162float(xr[tid + k2 * kQbaThreads]);
#pragma unroll
          for (int k2 = 0; k2 < kNqK; ++k2) ss += xv[k2] * xv[k2];
          ss = block_sum(ss, nsk);
          const float r = rsqrtf(ss / kKl + nq.eps);
          bf16* orow = nq.kv_cache + ncrow[m] * kKl;
#pragma unroll
          for (int k2 = 0; k2 < kNqK; ++k2) orow[tid + k2 * kQbaThreads] = __float2bfloat16(xv[k2] * r * nwk[k2]);
        }
      }
    }
    if constexpr (QX) {
      q_bar_wait(xbar, 0);  // every CTA's q_nope values of the head in this CTA's tile
    } else {
      // every CTA's q_nope values in every tile (release / acquire at cluster scope)
      asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
      asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    }
  }
  q_bar_wait(kbar, 0);
  // absorb: warp w computes the CTA's rows [16 w, 16 w + 16) for each of the head-batched gemv_pre's 8
  // k-slices (its warps' k-steps [2 s, 2 s + 2)) and sums the slices' partials in slice order in
  // registers: gemv_pre's products and sums
  {
    float acc[kQbaWarps][4];
#pragma unroll
    for (int s2 = 0; s2 < kQbaWarps; ++s2) acc[s2][0] = acc[s2][1] = acc[s2][2] = acc[s2][3] = 0.f;
    constexpr int ks_per = kNope / (16 * kQbaWarps);
    const uint32_t a0 = q_s32(sQ) + ((lane & 7) % M) * kRsK + ((lane >> 3) & 1) * 16;
    const uint32_t w0 = q_s32(sm) + (warp * 16 + (lane & 15)) * kRsK + (lane >> 4) * 16;
    // (TM: row r = 16 warp + lane % 16 of box b = chunk / 8, its 16-byte chunk q = chunk % 8 at (q ^ r % 8))
    const uint32_t t0 = q_s32(sm) + qba_tm_off(sm) + (warp * 16 + (lane & 15)) * 128;
#pragma unroll
    for (int s2 = 0; s2 < kQbaWarps; ++s2)
#pragma unroll
      for (int kk = 0; kk < ks_per; ++kk) {
        const int ks = s2 * ks_per + kk;
        uint32_t bfr[2], af[4];
        q_ldsm_x2(bfr, a0 + ks * 32);
        if constexpr (TM) {
          const int ch = 2 * ks + (lane >> 4);
          q_ldsm_x4(af, t0 + (ch >> 3) * (kAbRows * 128) + (((ch & 7) ^ (lane & 7)) << 4));
        } else {
          q_ldsm_x4(af, w0 + ks * 32);
        }
        q_mma(acc[s2], af, bfr);
      }
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      v[e] = 0.f;
#pragma unroll
      for (int s2 = 0; s2 < kQbaWarps; ++s2) v[e] += acc[s2][e];
    }
    // element e: row 16 w + lane / 4 (+ 8 for e >= 2), token 2 (lane % 4) (+ 1 for odd e); rows r, r + 1
    // of a token (lanes l, l + 4) packed, stored by the lanes of even lane / 4
    const bool st = ((lane >> 2) & 1) == 0;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint32_t hb = __bfloat16_as_ushort(__float2bfloat16(v[e]));
      const uint32_t nb = __shfl_down_sync(0xffffffffu, hb, 4);
      const int row = warp * 16 + (lane >> 2) + (e >> 1) * 8, tok = (lane & 3) * 2 + (e & 1);
      if (st && tok < M)
        *reinterpret_cast<uint32_t*>(out + (size_t)tok * ldo + (size_t)h * kKl + c * kAbRows + row) =
            hb | (nb << 16);
    }
  }
  if constexpr (SIG) {
    __threadfence();
    __syncthreads();
    if (tid == 0) atomicAdd(done, 1u);
  }
}

bool mla_qba_fits(int M, int H, int ql, int nope, int kl) {
  return M >= 1 && M <= kMT && H >= 1 && ql == kQl && nope == kNope && kl == kKl;
}

namespace {
// (TM) the absorb weights w_kcT ([H x kl rows, nope columns] bf16) as a 2-D tensor map: boxes of 64 columns x 128
// rows, 128-byte swizzle; one per weight pointer (built once, outside any capture's stream work)
const CUtensorMap& qba_kc_map(const bf16* w_kcT, int H) {
  static std::mutex mu;
  static std::map<std::pair<const void*, int>, CUtensorMap> maps;
  std::lock_guard<std::mutex> lk(mu);
  const auto key = std::make_pair(static_cast<const void*>(w_kcT), H);
  auto it = maps.find(key);
  if (it != maps.end()) return it->second;
  static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q));
    CHECK(fn != nullptr && q == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)kNope, (cuuint64_t)H * kKl};
  cuuint64_t strides[1] = {(cuuint64_t)kNope * sizeof(bf16)};
  cuuint32_t box[2] = {64, (cuuint32_t)kAbRows};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<bf16*>(w_kcT), dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CHECK(r == CUDA_SUCCESS, "mla_qba: cuTensorMapEncodeTiled failed: " + std::to_string((int)r));
  return maps.emplace(key, m).first->second;
}
}  // namespace

// (also the instances' module load: lazily loaded on first use, a load during sparse_mla_attn_sel's wait for the SIG
// instance would wait for that kernel)
void mla_qba_preload() {
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    for (auto kern :
         {mla_qba_kernel<true, true>, mla_qba_kernel<true, true, true>, mla_qba_kernel<true, true, false, true>})
      CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)qba_smem(kMT)));
    configured[dev] = true;
  }
}

bool mla_qba(const bf16* q_resid, int ldq, const bf16* w_qb, const bf16* w_kcT, bf16* q_lat, int ldo,
             int M, int H, cudaStream_t st, unsigned* done, bool early, const QbaNorm* nq) {
  if (!mla_qba_fits(M, H, kQl, kNope, kKl)) return false;
  CHECK(!nq || (!done && nq->wq && nq->x_kv && nq->wkv && nq->kv_cache && nq->cm.positions &&
                nq->cm.block_tables && nq->cm.token_seq && ldq % 8 == 0),
        "mla_qba: the norms (NQ) need the norms' weights and the cache metadata");
  mla_qba_preload();
  const CUtensorMap tmk = qba_kc_map(w_kcT, H);
  const int late = early ? 0 : 1;  // (late: the next kernel launched once this one's PDL wait is over)
  if (nq)
    launch_pdl(mla_qba_kernel<true, true, false, true>, dim3(kQbaCtas, H), dim3(kQbaThreads), qba_smem(M), st, q_resid,
               ldq, w_qb, w_kcT, q_lat, ldo, M, late, tmk, (unsigned*)nullptr, *nq);
  else if (done)
    launch_pdl(mla_qba_kernel<true, true, true>, dim3(kQbaCtas, H), dim3(kQbaThreads), qba_smem(M), st, q_resid, ldq,
               w_qb, w_kcT, q_lat, ldo, M, late, tmk, done, QbaNorm{});
  else
    launch_pdl(mla_qba_kernel<true, true>, dim3(kQbaCtas, H), dim3(kQbaThreads), qba_smem(M), st, q_resid, ldq, w_qb,
               w_kcT, q_lat, ldo, M, late, tmk, (unsigned*)nullptr, QbaNorm{});
  return true;
}

}  // namespace k
}  // namespace glm
