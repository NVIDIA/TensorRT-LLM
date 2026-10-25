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
// BF16 GEMV for few rows (decode) streaming the weights with TMA bulk copies: D[M, N] = A[M, K] x W[N, K]^T with
// M <= 8 and any N (attention DP's KDA in_proj: N 24896, K 4096, 204 MB per rank and call). Each CTA owns a
// contiguous range of weight rows (equal bytes per CTA, no wave quantization) and walks
// it in tiles of 16 rows and k-chunks of KC. A producer warp issues one 1D cp.async.bulk per (row, chunk) into a
// ring of NS stages whose rows are padded by 16 bytes (conflict-free ldmatrix without a swizzle); per stage a full
// mbarrier (the producer's expected bytes) and an empty one (one arrival per consumer warp), so the producer runs
// NS stages ahead and the consumers meet (named barrier) only at tile ends. The first stages are issued before the
// PDL wait (weights never depend on earlier kernels); the activations, loaded after it, stay resident. Eight
// consumer warps split each chunk's k-steps on the tensor cores (swap-AB: weight rows are the MMA M dimension,
// tokens N) and their partials are summed in warp order (deterministic; the same sums whatever the CTA count).
#include <algorithm>

#include "pdl.cuh"
#include "gemv_pre.h"
#include "lamport.cuh"
#include "reduce.cuh"
#include "sampling.h"

namespace glm {
namespace k {

namespace {
constexpr int kTWarps = 8, kTThreads = (kTWarps + 1) * 32;  // + the producer warp
constexpr int kTSmemMax = 227 * 1024;
constexpr int kTMaxRows = 8;

__device__ __forceinline__ uint32_t s32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void t_ldsm_x4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void t_ldsm_x2(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(a));
}
__device__ __forceinline__ void t_mma(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void t_bar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(s32(b)), "r"(n));
}
__device__ __forceinline__ void t_bar_expect(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(s32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void t_bar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(s32(b)) : "memory");
}
__device__ __forceinline__ void t_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
          s32(dst)),
      "l"(src), "r"(bytes), "r"(s32(b))
      : "memory");
}
// (EF: the weight copies with an L2 evict_first policy -- read once per step, so that they leave the other L2 lines in
// place; the same data)
__device__ __forceinline__ uint64_t t_ef_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void t_bulk_ef(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(
          s32(dst)),
      "l"(src), "r"(bytes), "r"(s32(b)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ bool t_bar_test(uint64_t* b, uint32_t ph) {
  uint32_t ok;
  asm volatile(
      "{ .reg .pred P; mbarrier.test_wait.parity.shared.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }\n"
      : "=r"(ok)
      : "r"(s32(b)), "r"(ph)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void t_bar_wait(uint64_t* b, uint32_t ph) {
  uint32_t ok = 0;
  while (!ok)
    asm volatile(
        "{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }\n"
        : "=r"(ok)
        : "r"(s32(b)), "r"(ph)
        : "memory");
}
__device__ __forceinline__ void t_consumers_sync() {  // the 8 consumer warps only
  asm volatile("bar.sync 1, 256;\n" ::: "memory");
}

// GemvEhArgs' rows: the producer warp's lane m (after the wait) finds row m's token and bulk-copies its embedding
// row and hidden row into sA row m (the A load's barrier and bytes); the 8 consumer warps (256 threads, 16 columns
// each; hidden 4096) then normalize them in place with mtp_eh_row's products and block_sum order (each warp's sum,
// then the 8 warp sums in one warp; red: scratch of 16 floats per row). Compact row loops (cold code).
__device__ __forceinline__ void t_unpack16(const uint8_t* p, float* f) {  // (16 bf16 of shared memory, as load8)
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const uint4 u = *reinterpret_cast<const uint4*>(p + 16 * j);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 t = __bfloat1622float2(b[i]);
      f[8 * j + 2 * i] = t.x;
      f[8 * j + 2 * i + 1] = t.y;
    }
  }
}
__device__ __forceinline__ void t_pack16(uint8_t* p, const float* f) {  // (as store8)
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    uint4 u;
    __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = __floats2bfloat162_rn(f[8 * j + 2 * i], f[8 * j + 2 * i + 1]);
    *reinterpret_cast<uint4*>(p + 16 * j) = u;
  }
}
// (consumers, after the rows' bulk copies landed)
__device__ __forceinline__ void t_eh_norm(const GemvEhArgs& eh, int M, const uint4* wv, uint8_t* sA, int AS,
                                          float* red) {
  constexpr int H = 4096;
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, c0 = tid * 16;
#pragma unroll 1
  for (int m = 0; m < M; ++m) {
    const uint8_t* row = sA + (size_t)m * AS;
    float e[16], h[16];
    t_unpack16(row + (size_t)c0 * 2, e);
    t_unpack16(row + (size_t)(H + c0) * 2, h);
    float se = 0.f, shh = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      se += e[i] * e[i];
      shh += h[i] * h[i];
    }
    se = warp_sum(se);
    shh = warp_sum(shh);
    if (lane == 0) {
      red[m * 16 + wid] = se;
      red[m * 16 + 8 + wid] = shh;
    }
  }
  t_consumers_sync();
#pragma unroll 1
  for (int m = 0; m < M; ++m) {
    uint8_t* row = sA + (size_t)m * AS;
    const float se = warp_sum(lane < 8 ? red[m * 16 + lane] : 0.f);
    const float shh = warp_sum(lane < 8 ? red[m * 16 + 8 + lane] : 0.f);
    const float re = rsqrtf(se / H + eh.eps), rh = rsqrtf(shh / H + eh.eps);
    float x[16], w[16];
    t_unpack16(row + (size_t)c0 * 2, x);
    t_unpack16(reinterpret_cast<const uint8_t*>(wv), w);
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = x[i] * re * w[i];
    t_pack16(row + (size_t)c0 * 2, x);
    t_unpack16(row + (size_t)(H + c0) * 2, x);
    t_unpack16(reinterpret_cast<const uint8_t*>(wv + 2), w);
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = x[i] * rh * w[i];
    t_pack16(row + (size_t)(H + c0) * 2, x);
  }
  t_consumers_sync();  // (the rows complete before any warp's MMAs)
}

__host__ __device__ inline size_t t_a_bytes(int M, int K) {
  return ((size_t)M * (K * 2 + 16) + 127) / 128 * 128;
}
template <int KC, int NS, int RT = 16>
__host__ __device__ inline size_t t_smem(int M, int K) {
  return t_a_bytes(M, K) + (size_t)NS * RT * (KC * 2 + 16) + (size_t)kTWarps * 16 * 8 * 4 +
         (2 * NS + 1) * 8;
}
}  // namespace

// UNEVEN: the first ne CTAs own rpc rows each, the rest rpc2 each (the TP KDA in_proj: its last CTAs become resident
// only when the previous boundary's CTAs exit, with nothing preloaded, and take fewer rows)
// EH (1): the activation rows built from GemvEhArgs instead of loaded. RT: weight rows per tile (16, or 8: a
// stage of 8 rows, the MMA's rows 8-15 duplicates of 0-7 whose sums are dropped; the same sums per row)
// PRE (< NS): only the first PRE stages issued before the PDL wait, the others right after the activation rows' copies:
// the stages issued at the CTA's start are still streaming at the wait (the TP KDA in_proj's CTAs become resident
// shortly before it, less HBM time than the 19 MB of 4 stages), so the activation rows would queue behind them while
// the full ring holds no new request; issued after the activations, the last stages keep requests in flight while the
// consumers start (the same sums)
// PACE (the attention's pushed o_proj; rpc2 = the depth D): before the PDL wait the producer keeps at most D ring
// stages in flight (stage i issued once stage i - D landed) and issues the rest as soon as the wait has passed;
// consumer warp 0 issues the activation rows at its wait. A CTA resident well before the wait (on the attention's SMs)
// still loads its whole ring before it; one that becomes resident shortly before it (as the w_vc's CTAs exit) does not
// queue its activation rows behind a full ring (128 KB) issued at its entry, where they would set the o_proj's end (the
// same sums)
// PD (> 0; the TP KDA in_proj, UNEVEN, launched early by the boundary before it): PACE's issue order with
// depth PD (rpc2 keeps the late CTAs' rows): all NS stages before the PDL wait, at most PD of them in flight, the
// activation rows by consumer warp 0 at its wait -- the ring is filled at a bounded rate while the boundary's push / poll
// runs, and the activations do not queue behind it (the same sums)
// RW (> 0; the same in_proj): the consumer warps copy the A fragments (their k-steps) of the CTA's first RW stages
// from the ring into registers before their PDL wait, so that the ring's slots refill with the next stages: RW + NS
// stages on chip at the wait (the smem ring alone holds NS); a 10th warp waits and issues the activation rows (the same
// fragments, MMAs and sums as the ring path: bitwise)
template <int KC, int NS, bool PUSH = false, bool UNEVEN = false, int EH = 0, int RT = 16, int PRE = NS,
          bool PACE = false, int PD = 0, int RW = 0, bool EF = false>
__global__ void __launch_bounds__(RW > 0 ? kTThreads + 32 : kTThreads, 1) gemv_tma_kernel(const GemvPreArgs p, int rpc,
                                                                                            int ne, int rpc2,
                                                                                            const GemvEhArgs eh) {
  if (!p.late_trigger) pdl_trigger();
  extern __shared__ __align__(128) uint8_t sm[];
  constexpr int MT = 8;           // token columns (one MMA n-tile)
  constexpr int RS = KC * 2 + 16;  // ring row stride (bytes)
  constexpr int STAGE = RT * RS;
  const int AS = p.K * 2 + 16;  // activation row stride
  uint8_t* sA = sm;
  uint8_t* ring = sm + t_a_bytes(p.M, p.K);
  float* red = reinterpret_cast<float*>(ring + (size_t)NS * STAGE);
  uint64_t* full = reinterpret_cast<uint64_t*>(red + kTWarps * 16 * MT);
  uint64_t* empty = full + NS;
  uint64_t* abar = empty + NS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int r0 = blockIdx.x * rpc, r1 = min(p.N, r0 + rpc);
  if constexpr (UNEVEN) {
    if ((int)blockIdx.x >= ne) {
      r0 = min(p.N, ne * rpc + ((int)blockIdx.x - ne) * rpc2);
      r1 = min(p.N, r0 + rpc2);
    }
  }
  const int ntile = r1 > r0 ? (r1 - r0 + RT - 1) / RT : 0, nch = p.K / KC, total = ntile * nch;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) {
      t_bar_init(&full[s], 1);
      t_bar_init(&empty[s], kTWarps);
    }
    t_bar_init(abar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  // (PACE: consumer warp 0 has passed the PDL wait and issued the activation rows -- the last word of red, which the
  // consumers first write at the end of the first tile, after the producer's paced issues: no static shared memory, so
  // the dynamic part keeps its 227 KB maximum)
  volatile int* wpassed = reinterpret_cast<volatile int*>(red + kTWarps * 16 * MT - 1);
  constexpr bool kPaced = PACE || PD > 0;
  if constexpr (kPaced || RW > 0) {
    if (tid == 0) *wpassed = 0;
  }
  __syncthreads();
  if constexpr (RW > 0) {
    static_assert(!PUSH && EH == 0 && RT == 16 && UNEVEN, "gemv_tma RW: the uneven in_proj form");
    constexpr int KS = KC / 16 / kTWarps;  // 16-element k-steps per warp and chunk
    if (warp == kTWarps + 1) {  // the activation rows at the PDL wait
      pdl_wait();
      if (p.late_trigger) pdl_trigger();
      if (lane == 0) t_bar_expect(abar, (uint32_t)p.M * p.K * 2);
      __syncwarp();
      for (int m = lane; m < p.M; m += 32)
        t_bulk(sA + (size_t)m * AS, p.A + (size_t)m * p.lda, p.K * 2, abar);
      __syncwarp();
      if (lane == 0) *wpassed = 1;
      return;
    }
    if (warp == kTWarps) {  // the producer: every stage through the ring, at most PD in flight before the wait
      [[maybe_unused]] uint64_t pol = 0;
      if constexpr (EF) pol = t_ef_policy();
      auto issue = [&](int i) {
        const int s = i % NS, t = i / nch, c = i % nch;
        const int n0 = r0 + t * RT, rows = min(RT, r1 - n0);
        if (lane == 0) t_bar_expect(&full[s], (uint32_t)rows * KC * 2);
        __syncwarp();
        if (lane < rows) {
          if constexpr (EF)
            t_bulk_ef(ring + (size_t)s * STAGE + lane * RS, p.W + (size_t)(n0 + lane) * p.ldw + (size_t)c * KC, KC * 2,
                      &full[s], pol);
          else
            t_bulk(ring + (size_t)s * STAGE + lane * RS, p.W + (size_t)(n0 + lane) * p.ldw + (size_t)c * KC, KC * 2,
                   &full[s]);
        }
      };
      // (PD 1-5: the first NS stages paced; PD 11-15: every stage before the wait, depth PD - 10 -- the refills of the
      // slots the consumers moved into registers too)
      constexpr int depth = PD > 10 ? PD - 10 : (PD > 0 ? PD : NS);
      for (int i = 0; i < NS && i < total; ++i) {
        if (i >= depth)
          while (!t_bar_test(&full[i - depth], 0) && *wpassed == 0) {
          }
        issue(i);
      }
      for (int i = NS; i < total; ++i) {
        t_bar_wait(&empty[i % NS], ((i / NS) - 1) & 1);
        if constexpr (PD > 10)
          while (!t_bar_test(&full[(i - depth) % NS], ((i - depth) / NS) & 1) && *wpassed == 0) {
          }
        issue(i);
      }
      return;
    }
    // consumers: stages 0 .. RW - 1 into registers before the wait (each warp its own k-steps), their slots released
    uint32_t wr[RW][KS][4];
    const uint32_t lofs = (lane & (RT - 1)) * RS + (lane >> 4) * 16;
#pragma unroll
    for (int i = 0; i < RW; ++i) {
      if (i < total) {
        const int s = i % NS;
        t_bar_wait(&full[s], (i / NS) & 1);
        const uint32_t sw = s32(ring + (size_t)s * STAGE) + lofs;
#pragma unroll
        for (int q = 0; q < KS; ++q) t_ldsm_x4(wr[i][q], sw + (warp * KS + q) * 32);
        __syncwarp();
        if (lane == 0) t_bar_arrive(&empty[s]);
      }
    }
    pdl_wait();
    t_bar_wait(abar, 0);
    float acc[4];
    const int tok = lane & 7;
    const uint32_t arow = s32(sA) + (tok < p.M ? tok : 0) * AS + ((lane >> 3) & 1) * 16;
    auto tile_end = [&](int t) {  // the warps' partials summed in warp order (the ring path's epilogue)
      float* rw = red + (size_t)warp * 16 * MT;
      const int row = lane >> 2, col = (lane & 3) * 2;
      rw[row * MT + col] = acc[0];
      rw[row * MT + col + 1] = acc[1];
      rw[(row + 8) * MT + col] = acc[2];
      rw[(row + 8) * MT + col + 1] = acc[3];
      t_consumers_sync();
      const int n0 = r0 + t * RT;
      for (int e = tid; e < RT * p.M; e += kTWarps * 32) {
        const int r = e / p.M, m = e % p.M;
        if (n0 + r >= r1) continue;
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kTWarps; ++w) v += red[(size_t)w * 16 * MT + r * MT + m];
        if (p.d_f32)
          static_cast<float*>(p.D)[(size_t)m * p.ldd + n0 + r] = v;
        else
          static_cast<bf16*>(p.D)[(size_t)m * p.ldd + n0 + r] = __float2bfloat16(v);
      }
      t_consumers_sync();
    };
#pragma unroll
    for (int i = 0; i < RW; ++i) {
      if (i < total) {
        const int t = i / nch, c = i % nch;
        if (c == 0) acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
#pragma unroll
        for (int q = 0; q < KS; ++q) {
          const int ks = warp * KS + q;
          uint32_t bfr[2];
          t_ldsm_x2(bfr, arow + (c * (KC / 8) + ks * 2) * 16);
          t_mma(acc, wr[i][q], bfr);
        }
        if (c == nch - 1) tile_end(t);
      }
    }
    for (int i = RW; i < total; ++i) {
      const int s = i % NS, t = i / nch, c = i % nch;
      if (c == 0) acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
      t_bar_wait(&full[s], (i / NS) & 1);
      const uint32_t sw = s32(ring + (size_t)s * STAGE) + lofs;
#pragma unroll
      for (int q = 0; q < KS; ++q) {
        const int ks = warp * KS + q;
        uint32_t af[4], bfr[2];
        t_ldsm_x4(af, sw + ks * 32);
        t_ldsm_x2(bfr, arow + (c * (KC / 8) + ks * 2) * 16);
        t_mma(acc, af, bfr);
      }
      __syncwarp();
      if (lane == 0) t_bar_arrive(&empty[s]);
      if (c == nch - 1) tile_end(t);
    }
    return;
  }
  if (warp == kTWarps) {  // the producer: one row per lane
    [[maybe_unused]] uint64_t pol = 0;
    if constexpr (EF) pol = t_ef_policy();
    auto issue = [&](int i) {
      const int s = i % NS, t = i / nch, c = i % nch;
      const int n0 = r0 + t * RT, rows = min(RT, r1 - n0);
      if (lane == 0) t_bar_expect(&full[s], (uint32_t)rows * KC * 2);
      __syncwarp();
      if (lane < rows) {
        if constexpr (EF)
          t_bulk_ef(ring + (size_t)s * STAGE + lane * RS, p.W + (size_t)(n0 + lane) * p.ldw + (size_t)c * KC, KC * 2,
                    &full[s], pol);
        else
          t_bulk(ring + (size_t)s * STAGE + lane * RS,
                 p.W + (size_t)(n0 + lane) * p.ldw + (size_t)c * KC, KC * 2, &full[s]);
      }
    };
    if constexpr (kPaced) {  // (no PDL wait here: the weights do not depend on earlier kernels)
      const int depth = PD > 0 ? PD : rpc2;
      for (int i = 0; i < NS && i < total; ++i) {
        if (i >= depth)
          while (!t_bar_test(&full[i - depth], 0) && *wpassed == 0) {
          }
        issue(i);
      }
      for (int i = NS; i < total; ++i) {
        t_bar_wait(&empty[i % NS], ((i / NS) - 1) & 1);
        issue(i);
      }
      return;
    }
    for (int i = 0; i < PRE && i < total; ++i) issue(i);
    pdl_wait();
    if (p.late_trigger) pdl_trigger();
    if (lane == 0) t_bar_expect(abar, (uint32_t)p.M * p.K * 2);
    __syncwarp();
    if constexpr (EH == 0) {
      for (int m = lane; m < p.M; m += 32)
        t_bulk(sA + (size_t)m * AS, p.A + (size_t)m * p.lda, p.K * 2, abar);
    } else if (lane < p.M) {  // (M <= 8) row lane: its embedding row, then its hidden row
      constexpr int H = 4096;
      const int id = argmax_key_token(__ldcg(eh.keys + lane));
      t_bulk(sA + (size_t)lane * AS, eh.embed + (size_t)id * H, H * 2, abar);
      t_bulk(sA + (size_t)lane * AS + H * 2, eh.hid + (size_t)lane * H, H * 2, abar);
    }
    if constexpr (PRE < NS)
      for (int i = PRE; i < NS && i < total; ++i) issue(i);
    for (int i = NS; i < total; ++i) {
      t_bar_wait(&empty[i % NS], ((i / NS) - 1) & 1);
      issue(i);
    }
    return;
  }
  uint4 wv[4];  // (EH) enorm, hnorm: weights, before the wait
  if constexpr (EH != 0) {
    const int c0 = tid * 16;
    wv[0] = __ldg(reinterpret_cast<const uint4*>(eh.enorm + c0));
    wv[1] = __ldg(reinterpret_cast<const uint4*>(eh.enorm + c0 + 8));
    wv[2] = __ldg(reinterpret_cast<const uint4*>(eh.hnorm + c0));
    wv[3] = __ldg(reinterpret_cast<const uint4*>(eh.hnorm + c0 + 8));
  }
  pdl_wait();
  if (p.late_trigger) pdl_trigger();
  if constexpr (kPaced) {
    if (warp == 0) {
      if (lane == 0) t_bar_expect(abar, (uint32_t)p.M * p.K * 2);
      __syncwarp();
      for (int m = lane; m < p.M; m += 32)
        t_bulk(sA + (size_t)m * AS, p.A + (size_t)m * p.lda, p.K * 2, abar);
      __syncwarp();
      if (lane == 0) *wpassed = 1;
    }
  }
  t_bar_wait(abar, 0);
  if constexpr (EH != 0) t_eh_norm(eh, p.M, wv, sA, AS, red);
  constexpr int KS = KC / 16 / kTWarps;  // 16-element k-steps per warp and chunk
  float acc[4];
  const int tok = lane & 7;
  const uint32_t arow = s32(sA) + (tok < p.M ? tok : 0) * AS + ((lane >> 3) & 1) * 16;
  for (int i = 0; i < total; ++i) {
    const int s = i % NS, t = i / nch, c = i % nch;
    if (c == 0) acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    t_bar_wait(&full[s], (i / NS) & 1);
    // (RT 8: the 8 rows' two k-halves only, x2; the MMA's rows 8-15 take the same registers -- the duplicates the x4
    // load read again from shared memory, half the stage's ldmatrix traffic)
    const uint32_t sw = RT == 8 ? s32(ring + (size_t)s * STAGE) + (lane & 7) * RS + ((lane >> 3) & 1) * 16
                                : s32(ring + (size_t)s * STAGE) + (lane & (RT - 1)) * RS + (lane >> 4) * 16;
#pragma unroll
    for (int q = 0; q < KS; ++q) {
      const int ks = warp * KS + q;  // 16-element k-step of the chunk
      uint32_t af[4], bfr[2];
      if constexpr (RT == 8) {
        t_ldsm_x2(af, sw + ks * 32);
        af[3] = af[1];
        af[2] = af[1];
        af[1] = af[0];
      } else {
        t_ldsm_x4(af, sw + ks * 32);
      }
      t_ldsm_x2(bfr, arow + (c * (KC / 8) + ks * 2) * 16);
      t_mma(acc, af, bfr);
    }
    __syncwarp();
    if (lane == 0) t_bar_arrive(&empty[s]);
    if (c == nch - 1) {  // tile done: the warps' partials summed in warp order
      float* rw = red + (size_t)warp * 16 * MT;
      const int row = lane >> 2, col = (lane & 3) * 2;
      rw[row * MT + col] = acc[0];
      rw[row * MT + col + 1] = acc[1];
      rw[(row + 8) * MT + col] = acc[2];
      rw[(row + 8) * MT + col + 1] = acc[3];
      t_consumers_sync();
      const int n0 = r0 + t * RT;
      if constexpr (PUSH) {
        // (p.push: rows per CTA a multiple of 16) 16-byte chunks of 8 rows of one token, the same sums,
        // stored and pushed into every rank's Lamport buffer of the next boundary's call (gemv_store's
        // protocol: epoch next_epoch(state[0]), -0.0 mapped to +0.0)
        const uint32_t ep = next_epoch(__ldcg(p.push.state));
        const size_t off = (size_t)(ep % 3) * 4 * p.push.max_elems + (size_t)p.push.rank * p.push.max_elems;
        for (int q = tid; q < (RT / 8) * p.M; q += kTWarps * 32) {
          const int rb = q / p.M, m = q % p.M;
          uint32_t h[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < kTWarps; ++w) v += red[(size_t)w * 16 * MT + (rb * 8 + i) * MT + m];
            h[i] = __bfloat16_as_ushort(__float2bfloat16(v));
          }
          uint4 u = make_uint4(h[0] | (h[1] << 16), h[2] | (h[3] << 16), h[4] | (h[5] << 16), h[6] | (h[7] << 16));
          const size_t ee = (size_t)m * p.ldd + n0 + rb * 8;
          *reinterpret_cast<uint4*>(static_cast<bf16*>(p.D) + ee) = u;
          u.x = clear_neg_zero(u.x);
          u.y = clear_neg_zero(u.y);
          u.z = clear_neg_zero(u.z);
          u.w = clear_neg_zero(u.w);
#pragma unroll
          for (int r = 0; r < 4; ++r) *reinterpret_cast<uint4*>(p.push.buf[r] + off + ee) = u;
        }
        t_consumers_sync();
        continue;
      }
      for (int e = tid; e < RT * p.M; e += kTWarps * 32) {
        const int r = e / p.M, m = e % p.M;
        if (n0 + r >= r1) continue;
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kTWarps; ++w) v += red[(size_t)w * 16 * MT + r * MT + m];
        if (p.d_f32)
          static_cast<float*>(p.D)[(size_t)m * p.ldd + n0 + r] = v;
        else
          static_cast<bf16*>(p.D)[(size_t)m * p.ldd + n0 + r] = __float2bfloat16(v);
      }
      t_consumers_sync();
    }
  }
}

namespace {
template <int KC, int NS, bool PUSH, int EH = 0, int RT = 16, bool PACE = false, bool EF = false>
void launch_tma_p(const GemvPreArgs& a, int ctas, cudaStream_t st, const GemvEhArgs& eh = GemvEhArgs{}, int depth = 0) {
  int rpc = (a.N + ctas - 1) / ctas;
  if (PUSH) {  // (whole tiles per CTA: the pushed 8-row chunks never straddle CTAs)
    rpc = (rpc + RT - 1) / RT * RT;
    ctas = (a.N + rpc - 1) / rpc;
  }
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_tma_kernel<KC, NS, PUSH, false, EH, RT, NS, PACE, 0, 0, EF>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kTSmemMax));
    // (the pushed o_proj: the largest carveout, so that a CTA of the 16-CTA cluster boundary that follows fits beside)
    if (PUSH) {
      if constexpr (PACE)
        CUDA_CHECK(cudaFuncSetAttribute(gemv_tma_kernel<KC, NS, PUSH, false, EH, RT, NS, PACE, 0, 0, EF>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      else
        CUDA_CHECK(
            cudaFuncSetAttribute(gemv_tma_kernel<KC, NS, PUSH>, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    configured[dev] = true;
  }
  launch_pdl(gemv_tma_kernel<KC, NS, PUSH, false, EH, RT, NS, PACE, 0, 0, EF>, dim3(ctas), dim3(kTThreads),
             t_smem<KC, NS, RT>(a.M, a.K), st, a, rpc, ctas, PACE ? depth : 0, eh);
}
// the pushed form with GemvEhArgs' rows
template <int KC, int NS>
void launch_tma_eh(const GemvPreArgs& a, int ctas, cudaStream_t st, const GemvEhArgs& eh) {
  launch_tma_p<KC, NS, true, 1>(a, ctas, st, eh);
}
// (the MTP eh_proj's (8-row-tile pushed forms) and the MLA o_proj's (pushed PACE form) weight copies take an L2
// evict_first policy, EF -- read once per pass / layer)
// the pushed form in 8-row tiles
template <int KC, int NS>
void launch_tma_r8(const GemvPreArgs& a, int ctas, cudaStream_t st) {
  launch_tma_p<KC, NS, true, 0, 8, false, true>(a, ctas, st);
}
// the first ctas - nl CTAs own the rows left after the last nl CTAs' rl rows each
template <int KC, int NS, int PRE = NS, int PD = 0, int RW = 0, bool EF = false>
void launch_tma_uneven(const GemvPreArgs& a, int ctas, int nl, int rl, cudaStream_t st) {
  const int ne = ctas - nl;
  const int rpc = (std::max(0, a.N - nl * rl) + ne - 1) / ne;
  static bool configured[64] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  if (!configured[dev]) {
    CUDA_CHECK(cudaFuncSetAttribute(gemv_tma_kernel<KC, NS, false, true, 0, 16, PRE, false, PD, RW, EF>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kTSmemMax));
    // (the TP KDA in_proj: the largest carveout, which its SMs keep when its CTAs exit, so that the KDA decode's co-form
    // CTAs landing there (58 KB) leave room for an o_proj CTA beside them)
    CUDA_CHECK(cudaFuncSetAttribute(gemv_tma_kernel<KC, NS, false, true, 0, 16, PRE, false, PD, RW, EF>,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    configured[dev] = true;
  }
  launch_pdl(gemv_tma_kernel<KC, NS, false, true, 0, 16, PRE, false, PD, RW, EF>, dim3(ctas),
             dim3(RW > 0 ? kTThreads + 32 : kTThreads), t_smem<KC, NS>(a.M, a.K), st, a, rpc, ne, rl, GemvEhArgs{});
}
template <int KC, int NS>
void launch_tma(const GemvPreArgs& a, int ctas, cudaStream_t st, int pace = 0) {
  if constexpr (NS == 4) {  // (PACE: the o_proj's 4-stage ring only)
    if (a.push.state && pace > 0) {
      launch_tma_p<KC, NS, true, 0, 16, true, true>(a, ctas, st, GemvEhArgs{}, pace);
      return;
    }
  }
  if (a.push.state)
    launch_tma_p<KC, NS, true>(a, ctas, st);
  else
    launch_tma_p<KC, NS, false>(a, ctas, st);
}
}  // namespace

bool gemv_tma_fits(int M, int N, int K) {
  if (M < 1 || M > kTMaxRows || N < 1 || K % 1024 != 0) return false;
  return t_smem<1024, 2>(M, K) <= (size_t)kTSmemMax;
}

void gemv_tma_uneven(const GemvPreArgs& a, int ctas, int nl, int rl, cudaStream_t st, int pd, int rw) {
  CHECK(gemv_tma_fits(a.M, a.N, a.K) && (t_smem<1024, 4>(a.M, a.K) <= (size_t)kTSmemMax),
        "gemv_tma_uneven: unsupported shape");
  CHECK(a.batch == 1 && !a.push.state && nl >= 0 && nl < ctas && rl >= 0, "gemv_tma_uneven: unsupported form");
  // (rw 3-5: the register-staged form, RW above; ring paced at 3 (pd 1-3), every pre-wait stage paced at 2 / 3 (pd 12 /
  // 13), or unpaced. The RW 4 / PD 1-3 form's weight copies -- 52.6 MB per KDA layer -- take an L2 evict_first policy,
  // EF above)
  if (rw == 4 && pd >= 1 && pd <= 3 && t_smem<1024, 5>(a.M, a.K) <= (size_t)kTSmemMax) {
    launch_tma_uneven<1024, 5, 5, 3, 4, true>(a, ctas, nl, rl, st);
    return;
  }
  if (rw >= 3 && t_smem<1024, 5>(a.M, a.K) <= (size_t)kTSmemMax) {
#define TMA_UNEVEN_RW(R)                                                                                    \
  do {                                                                                                      \
    if (pd >= 1 && pd <= 3) launch_tma_uneven<1024, 5, 5, 3, R>(a, ctas, nl, rl, st);                       \
    else if (pd == 12) launch_tma_uneven<1024, 5, 5, 12, R>(a, ctas, nl, rl, st);                           \
    else if (pd == 13) launch_tma_uneven<1024, 5, 5, 13, R>(a, ctas, nl, rl, st);                           \
    else launch_tma_uneven<1024, 5, 5, 0, R>(a, ctas, nl, rl, st);                                          \
  } while (0)
    if (rw == 3)
      TMA_UNEVEN_RW(3);
    else if (rw == 4)
      TMA_UNEVEN_RW(4);
    else
      TMA_UNEVEN_RW(5);
#undef TMA_UNEVEN_RW
    return;
  }
  // (pd 1-5: the paced form of a 5-stage ring, PD above: for the in_proj launched by an early-triggering boundary)
  if (pd >= 1 && t_smem<1024, 5>(a.M, a.K) <= (size_t)kTSmemMax) {
    switch (pd) {
      case 1: launch_tma_uneven<1024, 5, 5, 1>(a, ctas, nl, rl, st); return;
      case 2: launch_tma_uneven<1024, 5, 5, 2>(a, ctas, nl, rl, st); return;
      case 3: launch_tma_uneven<1024, 5, 5, 3>(a, ctas, nl, rl, st); return;
      case 4: launch_tma_uneven<1024, 5, 5, 4>(a, ctas, nl, rl, st); return;
      default: launch_tma_uneven<1024, 5, 5, 5>(a, ctas, nl, rl, st); return;
    }
  }
  // a 5-stage ring where it fits (<= 4 rows), 2 of its stages issued after the PDL wait, behind the activation rows
  // (PRE 3); else 4 stages, all before the wait
  if (t_smem<1024, 5>(a.M, a.K) <= (size_t)kTSmemMax)
    launch_tma_uneven<1024, 5, 3>(a, ctas, nl, rl, st);
  else
    launch_tma_uneven<1024, 4>(a, ctas, nl, rl, st);
}

void gemv_tma(const GemvPreArgs& a, int ctas, cudaStream_t st, int ns_max, const GemvEhArgs* eh, int rt, int pace) {
  CHECK(gemv_tma_fits(a.M, a.N, a.K), "gemv_tma: unsupported shape");
  CHECK(a.batch == 1, "gemv_tma: no batched form");
  CHECK(!a.push.state || (!a.d_f32 && a.ldd % 8 == 0 && a.late_trigger), "gemv_tma: pushed form shapes");
  ctas = std::max(1, std::min(ctas, a.N));
  if (rt == 8) {
    CHECK(a.push.state && (!eh || !eh->mode || eh->mode == 1), "gemv_tma: 8-row tiles (the pushed form)");
    if (eh && eh->mode) {  // (built rows: every k-chunk of a CTA's 8 rows loaded before the wait at 1-4 rows)
      const bool fits = t_smem<1024, 8, 8>(a.M, a.K) <= (size_t)kTSmemMax;
      CHECK(a.K == 8192 && eh->keys && eh->embed && eh->hid && eh->enorm && eh->hnorm && fits,
            "gemv_tma: built rows in 8-row tiles (the pushed form, hidden 4096, at most 4 rows)");
      launch_tma_p<1024, 8, true, 1, 8, false, true>(a, ctas, st, *eh);
      return;
    }
    if (t_smem<1024, 8, 8>(a.M, a.K) <= (size_t)kTSmemMax)
      launch_tma_r8<1024, 8>(a, ctas, st);
    else if (t_smem<1024, 6, 8>(a.M, a.K) <= (size_t)kTSmemMax)
      launch_tma_r8<1024, 6>(a, ctas, st);
    else if (t_smem<1024, 5, 8>(a.M, a.K) <= (size_t)kTSmemMax)
      launch_tma_r8<1024, 5>(a, ctas, st);
    else
      launch_tma_r8<1024, 4>(a, ctas, st);
    return;
  }
  if (eh && eh->mode) {
    CHECK(eh->mode == 1 && a.push.state && a.K == 8192 && eh->keys && eh->embed && eh->hid && eh->enorm && eh->hnorm,
          "gemv_tma: built rows (the pushed form, hidden 4096)");
    if (ns_max >= 6 && t_smem<1024, 6>(a.M, a.K) <= (size_t)kTSmemMax)
      launch_tma_eh<1024, 6>(a, ctas, st, *eh);
    else if (ns_max >= 5 && t_smem<1024, 5>(a.M, a.K) <= (size_t)kTSmemMax)
      launch_tma_eh<1024, 5>(a, ctas, st, *eh);
    else if (t_smem<1024, 4>(a.M, a.K) <= (size_t)kTSmemMax)
      launch_tma_eh<1024, 4>(a, ctas, st, *eh);
    else if (t_smem<1024, 3>(a.M, a.K) <= (size_t)kTSmemMax)
      launch_tma_eh<1024, 3>(a, ctas, st, *eh);
    else
      launch_tma_eh<1024, 2>(a, ctas, st, *eh);
    return;
  }
  // (ns_max > 4: a deeper ring where the activations leave room, more of the weights loaded before the PDL wait; the
  // sums do not change)
  if (ns_max >= 6 && t_smem<1024, 6>(a.M, a.K) <= (size_t)kTSmemMax)
    launch_tma<1024, 6>(a, ctas, st);
  else if (ns_max >= 5 && t_smem<1024, 5>(a.M, a.K) <= (size_t)kTSmemMax)
    launch_tma<1024, 5>(a, ctas, st);
  else if (t_smem<1024, 4>(a.M, a.K) <= (size_t)kTSmemMax)
    launch_tma<1024, 4>(a, ctas, st, ns_max == 4 ? pace : 0);
  else if (t_smem<1024, 3>(a.M, a.K) <= (size_t)kTSmemMax)
    launch_tma<1024, 3>(a, ctas, st);
  else
    launch_tma<1024, 2>(a, ctas, st);
}

}  // namespace k
}  // namespace glm
