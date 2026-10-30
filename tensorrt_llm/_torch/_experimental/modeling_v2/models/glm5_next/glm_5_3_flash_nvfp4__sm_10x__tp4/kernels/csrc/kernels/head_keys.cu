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
// The vocab-parallel LM head of few rows fused with its argmax keys (head_keys in sampling.h): gemv_tma's weight
// stream (each CTA owns an equal range of this rank's vocabulary rows and walks it in tiles of 16 rows and k-chunks
// of KC, cp.async.bulk into a ring of NS stages fed by a producer warp, 8 consumer warps on the tensor cores with the
// token rows as the MMA's n = 8 columns, warp partials summed in warp order), but every tile's logits become
// argmax_keys' keys in registers instead of an fp32 logits row: no logits store, no split-K reduction, no argmax
// kernel. Each consumer thread keeps one row's best key; the CTA's best keys go to a scratch row per CTA and the
// last CTA to finish (a counter it resets) reduces them, then runs custom_max_u64's Lamport exchange with the other
// ranks itself (no separate kernel).
#include <algorithm>

#include "pdl.cuh"
#include "lamport.cuh"
#include "sampling.h"
#include "sampling_dev.cuh"

namespace glm {
namespace k {

namespace {
constexpr int kHWarps = 8, kHThreads = (kHWarps + 1) * 32;  // + the producer warp
constexpr int kHSmemMax = 227 * 1024;

__device__ __forceinline__ uint32_t h_s32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void h_ldsm_x4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void h_ldsm_x2(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void h_mma(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void h_bar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(h_s32(b)), "r"(n));
}
__device__ __forceinline__ void h_bar_expect(uint64_t* b, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(h_s32(b)), "r"(tx) : "memory");
}
__device__ __forceinline__ void h_bar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(h_s32(b)) : "memory");
}
__device__ __forceinline__ void h_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
          h_s32(dst)),
      "l"(src), "r"(bytes), "r"(h_s32(b))
      : "memory");
}
// (EF: the vocab rows' bulk copies with an L2 evict_first policy -- read once per head; the same data)
__device__ __forceinline__ uint64_t h_ef_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ void h_bulk_ef(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" ::"r"(
          h_s32(dst)),
      "l"(src), "r"(bytes), "r"(h_s32(b)), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void h_bar_wait(uint64_t* b, uint32_t ph) {
  uint32_t ok = 0;
  while (!ok)
    asm volatile(
        "{ .reg .pred P; mbarrier.try_wait.parity.shared.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }\n"
        : "=r"(ok)
        : "r"(h_s32(b)), "r"(ph)
        : "memory");
}
__device__ __forceinline__ void h_consumers_sync() {  // the 8 consumer warps only
  asm volatile("bar.sync 1, 256;\n" ::: "memory");
}
__device__ __forceinline__ unsigned h_ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__host__ __device__ inline size_t h_a_bytes(int M, int K) { return ((size_t)M * (K * 2 + 16) + 127) / 128 * 128; }
template <int KC, int NS>
__host__ __device__ inline size_t h_smem(int M, int K) {  // (+ the CTA's best keys and the last-CTA flag)
  return h_a_bytes(M, K) + (size_t)NS * 16 * (KC * 2 + 16) + (size_t)kHWarps * 16 * 8 * 4 + (2 * NS + 1) * 8 +
         kHeadKeysMaxRows * 8 + 8;
}
}  // namespace

// RDY (p.ready set): the rows' readiness polled by the producer instead of the PDL wait (see HeadKeysArgs::ready)
template <int KC, int NS, bool RDY = false, bool EF = false>
__global__ void __launch_bounds__(kHThreads, 1) head_keys_kernel(const HeadKeysArgs p, int rpc) {
  if (!p.late_trigger) pdl_trigger();
  extern __shared__ __align__(128) uint8_t sm[];
  constexpr int MT = 8;            // token columns (one MMA n-tile)
  constexpr int RS = KC * 2 + 16;  // ring row stride (bytes)
  constexpr int STAGE = 16 * RS;
  const int AS = p.K * 2 + 16;  // activation row stride
  uint8_t* sA = sm;
  uint8_t* ring = sm + h_a_bytes(p.M, p.K);
  float* red = reinterpret_cast<float*>(ring + (size_t)NS * STAGE);
  uint64_t* full = reinterpret_cast<uint64_t*>(red + kHWarps * 16 * MT);
  uint64_t* empty = full + NS;
  uint64_t* abar = empty + NS;
  unsigned long long* s_best = reinterpret_cast<unsigned long long*>(abar + 1);  // [kHeadKeysMaxRows]
  int* s_last = reinterpret_cast<int*>(s_best + kHeadKeysMaxRows);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int r0 = blockIdx.x * rpc, r1 = min(p.N, r0 + rpc);
  const int ntile = r1 > r0 ? (r1 - r0 + 15) / 16 : 0, nch = p.K / KC, total = ntile * nch;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) {
      h_bar_init(&full[s], 1);
      h_bar_init(&empty[s], kHWarps);
    }
    h_bar_init(abar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (tid < kHeadKeysMaxRows) s_best[tid] = 0ull;
  __syncthreads();
  if (warp == kHWarps) {  // the producer: one row per lane
    [[maybe_unused]] uint64_t pol = 0;
    if constexpr (EF) pol = h_ef_policy();
    auto issue = [&](int i) {
      const int s = i % NS, t = i / nch, c = i % nch;
      const int n0 = r0 + t * 16, rows = min(16, r1 - n0);
      if (lane == 0) h_bar_expect(&full[s], (uint32_t)rows * KC * 2);
      __syncwarp();
      if (lane < rows) {
        if constexpr (EF)
          h_bulk_ef(ring + (size_t)s * STAGE + lane * RS, p.W + (size_t)(n0 + lane) * p.ldw + (size_t)c * KC, KC * 2,
                    &full[s], pol);
        else
          h_bulk(ring + (size_t)s * STAGE + lane * RS, p.W + (size_t)(n0 + lane) * p.ldw + (size_t)c * KC, KC * 2,
                 &full[s]);
      }
    };
    for (int i = 0; i < NS && i < total; ++i) issue(i);
    if constexpr (RDY) {  // (the rows stored by the norm's row CTAs; read below through the async proxy)
      if (lane == 0) {
        while (h_ld_acquire(p.ready) < p.ready_target) {
        }
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
      __syncwarp();
    } else {
      pdl_wait();
    }
    if (p.late_trigger) pdl_trigger();
    if (lane == 0) h_bar_expect(abar, (uint32_t)p.M * p.K * 2);
    __syncwarp();
    for (int m = lane; m < p.M; m += 32) h_bulk(sA + (size_t)m * AS, p.A + (size_t)m * p.lda, p.K * 2, abar);
    for (int i = NS; i < total; ++i) {
      h_bar_wait(&empty[i % NS], ((i / NS) - 1) & 1);
      issue(i);
    }
    if constexpr (RDY) pdl_wait();
    return;
  }
  if constexpr (!RDY) {
    pdl_wait();
    if (p.late_trigger) pdl_trigger();
  }
  // this thread's (tile row, token) in every tile's epilogue: e = tid < 16 M (M <= 8: one element per tile)
  const int em = tid % p.M, er = tid / p.M;
  const bool ep = tid < 16 * p.M;
  bool noise = false;
  float invT = 1.f;
  uint2 rkey = make_uint2(0, 0);
  uint32_t st_lo = 0, st_hi = 0;
  const SampleParams* const params = RDY ? nullptr : p.params;  // (RDY: greedy rows only, the drafts)
  if (ep && params) {
    const SampleParams sp = params[em];
    noise = sp.temperature > 0.f && sp.top_k != 1;
    invT = noise ? 1.f / sp.temperature : 1.f;
    rkey = make_uint2((uint32_t)sp.seed, (uint32_t)(sp.seed >> 32));
    st_lo = (uint32_t)sp.step;
    st_hi = (uint32_t)(sp.step >> 32);
  }
  unsigned long long best = 0ull;
  h_bar_wait(abar, 0);
  if constexpr (RDY)
    if (p.late_trigger) pdl_trigger();
  constexpr int KS = KC / 16 / kHWarps;  // 16-element k-steps per warp and chunk
  float acc[4];
  const int tok = lane & 7;
  const uint32_t arow = h_s32(sA) + (tok < p.M ? tok : 0) * AS + ((lane >> 3) & 1) * 16;
  for (int i = 0; i < total; ++i) {
    const int s = i % NS, t = i / nch, c = i % nch;
    if (c == 0) acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    h_bar_wait(&full[s], (i / NS) & 1);
    const uint32_t sw = h_s32(ring + (size_t)s * STAGE) + (lane & 15) * RS + (lane >> 4) * 16;
#pragma unroll
    for (int q = 0; q < KS; ++q) {
      const int ks = warp * KS + q;  // 16-element k-step of the chunk
      uint32_t af[4], bfr[2];
      h_ldsm_x4(af, sw + ks * 32);
      h_ldsm_x2(bfr, arow + (c * (KC / 8) + ks * 2) * 16);
      h_mma(acc, af, bfr);
    }
    __syncwarp();
    if (lane == 0) h_bar_arrive(&empty[s]);
    if (c == nch - 1) {  // tile done: the warps' partials summed in warp order, then the key of this thread's logit
      float* rw = red + (size_t)warp * 16 * MT;
      const int row = lane >> 2, col = (lane & 3) * 2;
      rw[row * MT + col] = acc[0];
      rw[row * MT + col + 1] = acc[1];
      rw[(row + 8) * MT + col] = acc[2];
      rw[(row + 8) * MT + col + 1] = acc[3];
      h_consumers_sync();
      const int n = r0 + t * 16 + er;  // this thread's vocabulary row (local)
      if (ep && n < r1) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kHWarps; ++w) v += red[(size_t)w * 16 * MT + er * MT + em];
        if (noise) {  // argmax_keys' Gumbel noise of global index vocab_offset + n
          const uint4 rn = philox4x32_10(gumbel_counter((uint32_t)((p.vocab_offset + n) >> 2), st_lo, st_hi), rkey);
          const uint32_t x = (n & 3) == 0 ? rn.x : (n & 3) == 1 ? rn.y : (n & 3) == 2 ? rn.z : rn.w;
          v = fmaf(v, invT, gumbel_of(x));
        }
        const unsigned long long key = argmax_key_of(v, p.vocab_offset + n);
        best = key > best ? key : best;
      }
      h_consumers_sync();
    }
  }
  // the CTA's best key per row -> its scratch row (thread 0, then counted with an acq_rel atomic: no separate
  // fence); the last CTA to count reduces every CTA's row (one per thread, warp shuffles, then shared atomics)
  if (ep) atomicMax(&s_best[em], best);
  h_consumers_sync();
  if (tid == 0) {
    for (int m = 0; m < p.M; ++m) p.part[(size_t)blockIdx.x * kHeadKeysMaxRows + m] = s_best[m];
    unsigned int c;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;\n" : "=r"(c) : "l"(p.count) : "memory");
    *s_last = c == gridDim.x - 1;
  }
  h_consumers_sync();
  if constexpr (RDY) pdl_wait();  // (the predecessor complete before this grid is: see HeadKeysArgs::ready)
  if (!*s_last) return;
  if (tid == 0) {
    *p.count = 0u;  // (every CTA has counted)
    if constexpr (RDY) *p.ready = 0u;  // (and seen the rows)
  }
  unsigned long long bk[kHeadKeysMaxRows];
#pragma unroll
  for (int m = 0; m < kHeadKeysMaxRows; ++m) bk[m] = 0ull;
  for (int b = tid; b < (int)gridDim.x; b += kHWarps * 32)
#pragma unroll
    for (int m = 0; m < kHeadKeysMaxRows; ++m)
      if (m < p.M) {
        const unsigned long long x = __ldcg(p.part + (size_t)b * kHeadKeysMaxRows + m);
        bk[m] = x > bk[m] ? x : bk[m];
      }
#pragma unroll
  for (int m = 0; m < kHeadKeysMaxRows; ++m)
    if (m < p.M) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long y = __shfl_xor_sync(0xffffffffu, bk[m], o);
        bk[m] = y > bk[m] ? y : bk[m];
      }
      if (lane == 0 && bk[m]) atomicMax(&s_best[m], bk[m]);
    }
  h_consumers_sync();
  const unsigned long long v = tid < p.M ? s_best[tid] : 0ull;
  if (p.tp <= 1) {
    if (tid < p.M) {
      p.keys[tid] = v;
      if (p.tokens) p.tokens[tid * p.tokens_ld] = argmax_key_token(v);
    }
    return;
  }
  // custom_max_u64's exchange (max_u64_kernel): slot buffer e % 3 receives every rank's keys, the receiver polls
  // until all are nonzero, and the previous call's buffer is reset to zero
  const uint32_t epoch = next_epoch(p.state[0]);
  const int prev_n = (int)p.state[1];
  const int cur = epoch % 3, old = (epoch + 2) % 3;
  const size_t buf = (size_t)p.tp * kMaxKeyRows;
  if (tid < p.M) {
    for (int r = 0; r < p.tp; ++r)
      *(volatile unsigned long long*)(p.peers.keys[r] + cur * buf + (size_t)p.rank * kMaxKeyRows + tid) = v;
    unsigned long long m = 0;
    for (int r = 0; r < p.tp; ++r) {
      unsigned long long x;
      do {
        x = *(volatile unsigned long long*)(p.peers.keys[p.rank] + cur * buf + (size_t)r * kMaxKeyRows + tid);
      } while (x == 0ull);
      m = m > x ? m : x;
    }
    p.keys[tid] = m;
    if (p.tokens) p.tokens[tid * p.tokens_ld] = argmax_key_token(m);
  }
  for (int i = tid; i < prev_n; i += kHWarps * 32)
    for (int r = 0; r < p.tp; ++r) p.peers.keys[p.rank][old * buf + (size_t)r * kMaxKeyRows + i] = 0ull;
  h_consumers_sync();
  if (tid == 0) {
    p.state[0] = epoch;
    p.state[1] = (uint32_t)p.M;
  }
}

namespace {
// the EF instances: the vocab rows with an L2 evict_first policy
template <int KC, int NS>
void launch_head(const HeadKeysArgs& a, int ctas, cudaStream_t st) {
  const int rpc = (a.N + ctas - 1) / ctas;
  static bool configured[64][2] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  const int rdy = a.ready ? 1 : 0;
  if (!configured[dev][rdy]) {
    CUDA_CHECK(cudaFuncSetAttribute(rdy ? head_keys_kernel<KC, NS, true, true> : head_keys_kernel<KC, NS, false, true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kHSmemMax));
    configured[dev][rdy] = true;
  }
  if (rdy)
    launch_pdl(head_keys_kernel<KC, NS, true, true>, dim3(ctas), dim3(kHThreads), h_smem<KC, NS>(a.M, a.K), st, a,
               rpc);
  else
    launch_pdl(head_keys_kernel<KC, NS, false, true>, dim3(ctas), dim3(kHThreads), h_smem<KC, NS>(a.M, a.K), st, a,
               rpc);
}
}  // namespace

bool head_keys_fits(int M, int N, int K) {
  if (M < 1 || M > kHeadKeysMaxRows || N < 1 || K % 1024 != 0) return false;
  return h_smem<1024, 2>(M, K) <= (size_t)kHSmemMax;
}

void head_keys(const HeadKeysArgs& a, int ctas, cudaStream_t st, int ns_max) {
  CHECK(head_keys_fits(a.M, a.N, a.K), "head_keys: unsupported shape");
  CHECK(!a.ready || (!a.params && a.ready_target > 0), "head_keys: the ready count is for greedy rows (the drafts)");
  CHECK(a.vocab_offset % 4 == 0 && a.N % 4 == 0, "head_keys: vocab shards of 4k tokens");
  CHECK(a.tp <= 1 || (a.state && a.tp <= kMaxTp), "head_keys: the cross-rank max needs the peers' key slots");
  ctas = std::max(1, std::min(std::min(ctas, kHeadKeysMaxCtas), a.N));
  if (ns_max >= 6 && h_smem<1024, 6>(a.M, a.K) <= (size_t)kHSmemMax)
    launch_head<1024, 6>(a, ctas, st);
  else if (h_smem<1024, 4>(a.M, a.K) <= (size_t)kHSmemMax)
    launch_head<1024, 4>(a, ctas, st);
  else if (h_smem<1024, 3>(a.M, a.K) <= (size_t)kHSmemMax)
    launch_head<1024, 3>(a, ctas, st);
  else
    launch_head<1024, 2>(a, ctas, st);
}

}  // namespace k
}  // namespace glm
