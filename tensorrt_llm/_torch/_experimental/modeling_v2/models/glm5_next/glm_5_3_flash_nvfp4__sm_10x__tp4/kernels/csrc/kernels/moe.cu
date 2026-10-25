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
// MoE router logits, routing, small-batch dispatch and combine kernels.
#include "pdl.cuh"
#include <cooperative_groups.h>

#include "moe.h"
#include "moe_route.cuh"
#include "reduce.cuh"

namespace glm {
namespace k {

// ------------------------------------------------------------------ decode router
// Router logits for decode-sized T (moe_router_logits): clusters of 4 CTAs. Cluster c owns
// experts [8c, 8c + 8), CTA q of the cluster the k-quarter q: warp w computes expert 8c + w over
// that quarter, its lanes holding the weights of k = q H/4 + j 256 + lane 8 (+0..7), j < NV,
// loaded before the PDL wait (weights never depend on earlier kernels); the quarter of every
// token's x is staged in shared memory with cp.async. The 4 quarter sums of a logit are added in
// quarter order through distributed shared memory: deterministic, no atomics.
constexpr int kRlCluster = 4;
constexpr int kRlExperts = 8;   // experts per CTA (one per warp)
constexpr int kRlThreads = 32 * kRlExperts;

__device__ __forceinline__ void bf16x8_to_f32(const uint4& u, float* f) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

template <int NV>
__global__ void __cluster_dims__(kRlCluster, 1, 1) __launch_bounds__(kRlThreads)
    moe_router_logits_kernel(float* __restrict__ logits, const bf16* __restrict__ x,
                             const bf16* __restrict__ W, int T, int E, int H) {
  pdl_trigger();
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  extern __shared__ __align__(16) uint8_t rl_smem[];
  bf16* xs = reinterpret_cast<bf16*>(rl_smem);  // [T][H / 4]
  __shared__ float part[kRouterMaxT][kRlExperts];
  const int q = (int)cluster.block_rank(), c = blockIdx.x / kRlCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int e = c * kRlExperts + warp;
  const int Hq = NV * 256;  // = H / 4
  uint4 wv[NV];
#pragma unroll
  for (int j = 0; j < NV; ++j)
    wv[j] = e < E ? __ldg(reinterpret_cast<const uint4*>(W + (size_t)e * H + q * Hq + j * 256 +
                                                         lane * 8))
                  : make_uint4(0, 0, 0, 0);
  pdl_wait();
  const int cpr = Hq / 8;  // 16-byte chunks per token row of the quarter
  for (int i = tid; i < T * cpr; i += kRlThreads) {
    const int t = i / cpr, cc = i - t * cpr;
    const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(xs + (size_t)t * Hq + cc * 8));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
                 "l"(x + (size_t)t * H + q * Hq + cc * 8));
  }
  asm volatile("cp.async.commit_group;\n" ::);
  float wf[NV][8];
#pragma unroll
  for (int j = 0; j < NV; ++j) bf16x8_to_f32(wv[j], wf[j]);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  for (int t0 = 0; t0 < T; t0 += 4) {
    float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (t0 + u < T) {
        const bf16* xr = xs + (size_t)(t0 + u) * Hq + lane * 8;
#pragma unroll
        for (int j = 0; j < NV; ++j) {
          float xf[8];
          bf16x8_to_f32(*reinterpret_cast<const uint4*>(xr + j * 256), xf);
#pragma unroll
          for (int i = 0; i < 8; ++i) a[u] = fmaf(xf[i], wf[j][i], a[u]);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) a[u] = warp_sum(a[u]);
    if (lane == 0) {
#pragma unroll
      for (int u = 0; u < 4; ++u)
        if (t0 + u < T) part[t0 + u][warp] = a[u];
    }
  }
  cluster.sync();
  // CTA q writes the logits of tokens t = q (mod 4)
  for (int i = tid; i < T * kRlExperts; i += kRlThreads) {
    const int t = i / kRlExperts, w = i - t * kRlExperts, ee = c * kRlExperts + w;
    if (t % kRlCluster != q || ee >= E) continue;
    const float p0 = *cluster.map_shared_rank(&part[t][w], 0);
    const float p1 = *cluster.map_shared_rank(&part[t][w], 1);
    const float p2 = *cluster.map_shared_rank(&part[t][w], 2);
    const float p3 = *cluster.map_shared_rank(&part[t][w], 3);
    logits[(size_t)t * E + ee] = (p0 + p1) + (p2 + p3);
  }
  cluster.sync();  // the other CTAs' reads of this CTA's partials are done
}

bool moe_router_logits(float* logits, const bf16* x, const bf16* W, int T, int E, int H,
                       cudaStream_t st) {
  if (T <= 0 || T > kRouterMaxT || H % 1024 != 0) return false;
  const int grid = (int)cdiv(E, kRlExperts) * kRlCluster;
  const size_t smem = (size_t)T * (H / 4) * sizeof(bf16);
#define GLM_RL_LAUNCH(NV)                                                                      \
  do {                                                                                         \
    static bool configured[64] = {};                                                           \
    int dev = 0;                                                                               \
    CUDA_CHECK(cudaGetDevice(&dev));                                                           \
    if (!configured[dev]) {                                                                    \
      CUDA_CHECK(cudaFuncSetAttribute(moe_router_logits_kernel<NV>,                            \
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,             \
                                      (int)(kRouterMaxT * NV * 256 * sizeof(bf16))));          \
      configured[dev] = true;                                                                  \
    }                                                                                          \
    launch_pdl(moe_router_logits_kernel<NV>, dim3(grid), dim3(kRlThreads), smem, st, logits,   \
               x, W, T, E, H);                                                                 \
  } while (0)
  switch (H / 1024) {
    case 2: GLM_RL_LAUNCH(2); break;
    case 4: GLM_RL_LAUNCH(4); break;
    case 8: GLM_RL_LAUNCH(8); break;
    default: return false;
  }
#undef GLM_RL_LAUNCH
  return true;
}

// ------------------------------------------------------------------ decode router, tensor cores
// Router logits for 1-32 tokens on the tensor cores (moe_router_mma): clusters of 8 CTAs; cluster
// g owns experts [16 g, 16 g + 16), CTA c of the cluster the k-range [512 c, 512 c + 512), warp w
// the 128-wide k-slice w of it. mma.sync m16n8k16 with the expert rows as A (held in registers,
// loaded before the PDL wait: weights never depend on earlier kernels) and 8 tokens per n-tile as
// B. Inside a warp's slice the k order is permuted so that each lane's operands are 32 contiguous
// values of its two rows and of its token (A and B use the same permutation: the dot products are
// unchanged, only the order of the fp32 sums differs from cuBLASLt's). Every warp stores its
// partials straight into the shared memory of the CTA that reduces them (lane l's outputs: CTA
// l % 8), one cluster barrier, then each CTA adds its outputs' 32 partials in (CTA, warp) order:
// deterministic, no atomics, one barrier. 128 threads and at most 8 KB of shared memory per CTA,
// so the fused MoE kernel that this one launches early (219.9 KB of shared memory, 192 threads)
// still fits on the same SMs.
// KS (moe_router_mma_ks): each output's routing choice key and score (route_key_score with the expert's bias,
// loaded before the PDL wait) instead of the logit, as {key, score} 64-bit words: the fused MoE kernel's
// in-kernel routing then runs no score arithmetic on its critical path (16 threads per CTA run it here).
// XB (the default exchange): the partials travel as st.async stores that complete bytes on an mbarrier of the
// reducing CTA (initialized and armed before the PDL wait, one cluster barrier there), whose reducing threads wait
// on it, instead of plain DSMEM stores and a cluster barrier after the MMAs (that barrier waits for every thread of
// the 8 CTAs). The same partials, the same sums.
constexpr int kRmCluster = 8, kRmRows = 16, kRmWarps = 4, kRmThreads = 32 * kRmWarps;
constexpr int kRmKc = 512;  // k per CTA; H = kRmCluster * kRmKc

__device__ __forceinline__ uint32_t rm_smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint4 ld_cg_u4(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ void rm_mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                       uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int NT, bool KS = false, bool XB = false>  // n-tiles of 8 tokens
__global__ void __cluster_dims__(kRmCluster, 1, 1) __launch_bounds__(kRmThreads)
    moe_router_mma_kernel(float* __restrict__ logits, const bf16* __restrict__ x,
                          const bf16* __restrict__ W, int T, int E, int H, const float* __restrict__ bias) {
  pdl_trigger();
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  // partials this CTA reduces: [source CTA x warp][n-tile][lane / 8][4] (lanes l = c + 8 m)
  __shared__ __align__(16) float recv[kRmCluster * kRmWarps][NT][4][4];
  __shared__ __align__(8) uint64_t xbar;  // (XB) the partials' bytes arrived
  const int c = (int)cluster.block_rank(), g = blockIdx.x / kRmCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if constexpr (XB) {
    if (tid == 0) {  // one arrival (this one), the partials' 2 KB per n-tile as transaction bytes
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(rm_smem(&xbar)) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(rm_smem(&xbar)),
                   "r"((uint32_t)sizeof(recv))
                   : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
  }
  const int r = lane >> 2, q = lane & 3;
  const int k0 = c * kRmKc + warp * 128 + q * 32;  // this lane's 32 contiguous k
  const int e0 = g * kRmRows + r, e1 = e0 + 8;
  uint32_t wa[16], wb[16];  // rows e0 / e1, k0 .. k0 + 32 (bf16 pairs)
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 va = e0 < E ? __ldg(reinterpret_cast<const uint4*>(W + (size_t)e0 * H + k0) + i)
                            : make_uint4(0, 0, 0, 0);
    const uint4 vb = e1 < E ? __ldg(reinterpret_cast<const uint4*>(W + (size_t)e1 * H + k0) + i)
                            : make_uint4(0, 0, 0, 0);
    wa[4 * i] = va.x, wa[4 * i + 1] = va.y, wa[4 * i + 2] = va.z, wa[4 * i + 3] = va.w;
    wb[4 * i] = vb.x, wb[4 * i + 1] = vb.y, wb[4 * i + 2] = vb.z, wb[4 * i + 3] = vb.w;
  }
  float4* dst = reinterpret_cast<float4*>(
      cluster.map_shared_rank(&recv[c * kRmWarps + warp][0][lane >> 3][0], lane & 7));
  // (KS) this thread's output expert's bias (a weight: before the wait)
  [[maybe_unused]] float be = 0.f;
  if constexpr (KS) {
    const int l = c + 8 * ((tid >> 2) & 3), e = g * kRmRows + (l >> 2) + 8 * ((tid & 3) >> 1);
    if (tid < NT * 16 && e < E) be = __ldg(bias + e);
  }
  // (XB) every CTA's barrier initialized and armed before any partial is sent (off the critical path: the router
  // waits for the kernel before it)
  [[maybe_unused]] uint32_t xslot = 0, xrbar = 0;
  if constexpr (XB) {
    cluster.sync();
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
                 : "=r"(xslot)
                 : "r"(rm_smem(&recv[c * kRmWarps + warp][0][lane >> 3][0])), "r"(lane & 7));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(xrbar) : "r"(rm_smem(&xbar)), "r"(lane & 7));
  }
  pdl_wait();
  uint32_t xb[NT][16];  // token 8 n + r, the same 32 k
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    const int t = 8 * n + r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 v = t < T ? ld_cg_u4(reinterpret_cast<const uint4*>(x + (size_t)t * H + k0) + i)
                            : make_uint4(0, 0, 0, 0);
      xb[n][4 * i] = v.x, xb[n][4 * i + 1] = v.y, xb[n][4 * i + 2] = v.z, xb[n][4 * i + 3] = v.w;
    }
  }
  float acc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  // k-step s: virtual k 2q, 2q + 1 -> word 2 s (k0 + 4 s + 0, 1), 2q + 8, 2q + 9 -> word 2 s + 1;
  // the n-tiles' accumulator chains interleaved
#pragma unroll
  for (int s = 0; s < 8; ++s)
#pragma unroll
    for (int n = 0; n < NT; ++n)
      rm_mma(acc[n], wa[2 * s], wb[2 * s], wa[2 * s + 1], wb[2 * s + 1], xb[n][2 * s],
             xb[n][2 * s + 1]);
  // this lane's outputs (n, l, j) go to CTA l % 8, slot [c, warp][n][l / 8][j]
  if constexpr (XB) {
#pragma unroll
    for (int n = 0; n < NT; ++n)
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n"
                   ::"r"(xslot + (uint32_t)(n * 64)), "f"(acc[n][0]), "f"(acc[n][1]), "f"(acc[n][2]), "f"(acc[n][3]),
                   "r"(xrbar)
                   : "memory");
  } else {
#pragma unroll
    for (int n = 0; n < NT; ++n) dst[n * 4] = make_float4(acc[n][0], acc[n][1], acc[n][2], acc[n][3]);
    cluster.sync();  // every partial of this CTA's outputs is here
  }
  // outputs (n, l = c + 8 m, j): thread (n, m, j)
  if (tid < NT * 16) {
    if constexpr (XB) {  // (every partial of this CTA's outputs is here)
      asm volatile("{\n.reg .pred p;\nXB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra XB_WAIT;\n}\n"
                   ::"r"(rm_smem(&xbar))
                   : "memory");
    }
    const int n = tid >> 4, m = (tid >> 2) & 3, j = tid & 3, l = c + 8 * m;
    float v = 0.f;
#pragma unroll
    for (int src = 0; src < kRmCluster * kRmWarps; ++src) v += recv[src][n][m][j];
    const int e = g * kRmRows + (l >> 2) + 8 * (j >> 1), t = 8 * n + 2 * (l & 3) + (j & 1);
    if constexpr (KS) {
      if (t < T && e < E) {
        uint32_t key;
        float sc;
        route_key_score(v, be, key, sc);
        reinterpret_cast<uint2*>(logits)[(size_t)t * E + e] = make_uint2(key, __float_as_uint(sc));
      }
    } else {
      if (t < T && e < E) logits[(size_t)t * E + e] = v;
    }
  }
}

static bool router_xb(int xb) { return xb != 0; }  // (the default xb -1: XB)

bool moe_router_mma(float* logits, const bf16* x, const bf16* W, int T, int E, int H,
                    cudaStream_t st, int xb) {
  if (T <= 0 || T > kRouterMaxT || H != kRmCluster * kRmKc || E % kRmRows != 0) return false;
  const dim3 grid((E / kRmRows) * kRmCluster), block(kRmThreads);
  const float* const nb = nullptr;
  if (router_xb(xb)) {
    switch ((T + 7) / 8) {
      case 1: launch_pdl(moe_router_mma_kernel<1, false, true>, grid, block, 0, st, logits, x, W, T, E, H, nb); break;
      case 2: launch_pdl(moe_router_mma_kernel<2, false, true>, grid, block, 0, st, logits, x, W, T, E, H, nb); break;
      case 3: launch_pdl(moe_router_mma_kernel<3, false, true>, grid, block, 0, st, logits, x, W, T, E, H, nb); break;
      default: launch_pdl(moe_router_mma_kernel<4, false, true>, grid, block, 0, st, logits, x, W, T, E, H, nb); break;
    }
    return true;
  }
  switch ((T + 7) / 8) {
    case 1: launch_pdl(moe_router_mma_kernel<1>, grid, block, 0, st, logits, x, W, T, E, H, nb); break;
    case 2: launch_pdl(moe_router_mma_kernel<2>, grid, block, 0, st, logits, x, W, T, E, H, nb); break;
    case 3: launch_pdl(moe_router_mma_kernel<3>, grid, block, 0, st, logits, x, W, T, E, H, nb); break;
    default: launch_pdl(moe_router_mma_kernel<4>, grid, block, 0, st, logits, x, W, T, E, H, nb); break;
  }
  return true;
}

bool moe_router_mma_ks(unsigned long long* ks, const bf16* x, const bf16* W, const float* bias, int T, int E,
                       int H, cudaStream_t st, int xb) {
  if (T <= 0 || T > 8 || H != kRmCluster * kRmKc || E % kRmRows != 0) return false;
  if (router_xb(xb))
    launch_pdl(moe_router_mma_kernel<1, true, true>, dim3((E / kRmRows) * kRmCluster), dim3(kRmThreads), 0, st,
               reinterpret_cast<float*>(ks), x, W, T, E, H, bias);
  else
    launch_pdl(moe_router_mma_kernel<1, true>, dim3((E / kRmRows) * kRmCluster), dim3(kRmThreads), 0, st,
               reinterpret_cast<float*>(ks), x, W, T, E, H, bias);
  return true;
}

constexpr int kRouteWarpMaxPer = 16;  // E <= 512

// One warp per token; the biases are loaded before the PDL wait.
template <int MAXPER>
__global__ void __launch_bounds__(256) moe_route_warp_kernel(
    int* __restrict__ ids, float* __restrict__ wout, const float* __restrict__ logits,
    const float* __restrict__ bias, int T, int E, int top_k, float routed_scale, int shared_id,
    const int* __restrict__ tok_slot, int skip_slot) {
  pdl_trigger();
  const int lane = threadIdx.x & 31, t = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  float bl[MAXPER];
#pragma unroll
  for (int i = 0; i < MAXPER; ++i) {
    const int e = lane + 32 * i;
    bl[i] = e < E ? __ldg(bias + e) : 0.f;
  }
  pdl_wait();
  if (t >= T) return;
  const int K1 = top_k + (shared_id >= 0);
  if (tok_slot && tok_slot[t] == skip_slot) {  // padding token: no expert
    if (lane < K1) {
      ids[(size_t)t * K1 + lane] = -1;
      wout[(size_t)t * K1 + lane] = 0.f;
    }
    return;
  }
  route_token_warp<MAXPER>(logits + (size_t)t * E, bl, E, top_k, routed_scale, shared_id,
                           ids + (size_t)t * K1, wout + (size_t)t * K1, lane);
}

void moe_route_warp(int* topk_ids, float* topk_w, const float* logits, const float* bias, int T,
                    int E, int top_k, float routed_scale, cudaStream_t st, int shared_id,
                    const int* tok_slot, int skip_slot) {
  if (T == 0) return;
  CHECK(E <= 32 * kRouteWarpMaxPer && E >= top_k && top_k <= 31, "routing kernel limits");
  const dim3 grid(cdiv(T, 8)), block(256);
  if (E <= 32 * 9)
    launch_pdl(moe_route_warp_kernel<9>, grid, block, 0, st, topk_ids, topk_w, logits, bias, T,
               E, top_k, routed_scale, shared_id, tok_slot, skip_slot);
  else
    launch_pdl(moe_route_warp_kernel<kRouteWarpMaxPer>, grid, block, 0, st, topk_ids, topk_w,
               logits, bias, T, E, top_k, routed_scale, shared_id, tok_slot, skip_slot);
}

// ------------------------------------------------------------------ small-batch dispatch
// Single CTA (1024 threads, G <= 1024 groups): shared-memory histogram of the (token, slot)
// pairs, block-wide exclusive scan -> offsets and the compacted active-group list, then each
// pair claims a row with a shared-memory cursor. Rows inside a group are in arbitrary order,
// which does not change any result (every grouped row is computed independently and the
// combine sums a token's slots in slot order). row_src maps grouped rows back to tokens so the
// grouped GEMM can gather its A rows without replicating them.
// MAXPER > 0: the routing of the (T <= 32) tokens is computed first from ra.logits, one warp per
// token (route_token_warp, as moe_route_warp), into ra.ids_out / ra.w_out and the ids used here.
template <int MAXPER>
__global__ void __launch_bounds__(1024) moe_dispatch_small_kernel(
    int* __restrict__ pair_pos, int* __restrict__ row_src, int* __restrict__ offsets,
    int* __restrict__ active, int* __restrict__ num_active, const int* __restrict__ ids_in, int T,
    int K1, const MoeGroupMap map, int mt_rows, const MoeRouteArgs ra) {
  pdl_trigger();
  __shared__ int cnt[1024], cursor[1024], wsum[32], wact[32];
  __shared__ int rload[8], owner_s;
  __shared__ int ids_s[MAXPER > 0 ? kRouterMaxT * 32 : 1];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float bl[MAXPER > 0 ? MAXPER : 1];
  if constexpr (MAXPER > 0) {
#pragma unroll
    for (int i = 0; i < MAXPER; ++i) {
      const int e = lane + 32 * i;
      bl[i] = e < map.E ? __ldg(ra.bias + e) : 0.f;
    }
  }
  pdl_wait();
  const int* ids = ids_in;
  if constexpr (MAXPER > 0) {
    if (warp < T) {
      const int id = route_token_warp<MAXPER>(ra.logits + (size_t)warp * map.E, bl, map.E,
                                              ra.top_k, ra.scale, map.E,
                                              ra.ids_out + (size_t)warp * K1,
                                              ra.w_out + (size_t)warp * K1, lane);
      if (lane < K1) ids_s[warp * K1 + lane] = id;
    }
    ids = ids_s;
    __syncthreads();
  }
  const int n = T * K1, G = map.groups();
  cnt[tid] = 0;
  // shared expert owner: the rank holding the fewest distinct routed experts of the batch
  int owner = -1;
  if (map.shared_dyn && map.ep > 1) {
    cursor[tid] = 0;  // (as the used-expert flags until the scan below)
    if (tid < 8) rload[tid] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += 1024) {
      const int id = ids[i];
      if (id >= 0 && id < map.E) cursor[id] = 1;
    }
    __syncthreads();
    if (tid < map.E && cursor[tid]) atomicAdd(&rload[tid / (map.E / map.ep)], 1);
    __syncthreads();
    if (tid == 0) {
      int best = 0;
      for (int r = 1; r < map.ep; ++r)
        if (rload[r] < rload[best]) best = r;
      owner_s = best;
    }
  }
  __syncthreads();
  if (map.shared_dyn && map.ep > 1) owner = owner_s;
  auto gmap = [&](int id, int t) {
    if (owner >= 0 && id >= map.E) return owner == map.ep_rank ? map.n : -1;
    return map(id, t);
  };
  for (int i = tid; i < n; i += 1024) {
    const int g = gmap(ids[i], i / K1);
    if (g >= 0) atomicAdd(&cnt[g], 1);
  }
  __syncthreads();
  // two-level exclusive scan of counts and of the m-tile work items
  const int c = tid < G ? cnt[tid] : 0;
  const int items = (c + mt_rows - 1) / mt_rows;
  int v = c, f = items;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up_sync(0xffffffff, v, o), uf = __shfl_up_sync(0xffffffff, f, o);
    if (lane >= o) {
      v += u;
      f += uf;
    }
  }
  if (lane == 31) {
    wsum[warp] = v;
    wact[warp] = f;
  }
  __syncthreads();
  if (warp == 0) {
    int a = wsum[lane], b = wact[lane];
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffff, a, o), ub = __shfl_up_sync(0xffffffff, b, o);
      if (lane >= o) {
        a += u;
        b += ub;
      }
    }
    wsum[lane] = a;  // inclusive per-warp prefix
    wact[lane] = b;
  }
  __syncthreads();
  const int base = warp > 0 ? wsum[warp - 1] : 0, abase = warp > 0 ? wact[warp - 1] : 0;
  const int excl = base + v - c;
  if (tid < G) {
    offsets[tid] = excl;
    cursor[tid] = excl;
    for (int m = 0; m < items; ++m) active[abase + f - items + m] = tid | (m << 16);
  }
  if (tid == 0) {
    offsets[G] = wsum[31];
    *num_active = wact[31];
  }
  __syncthreads();
  for (int i = tid; i < n; i += 1024) {
    const int g = gmap(ids[i], i / K1);
    int pos = -1;
    if (g >= 0) {
      pos = atomicAdd(&cursor[g], 1);
      row_src[pos] = i / K1;
    } else if (map.shared_mode == 1 && ids[i] >= map.E) {
      pos = map.shared_rows + i / K1;  // the separate shared expert's row of this token
    }
    pair_pos[i] = pos;
  }
}

void moe_dispatch_small(int* pair_pos, int* row_src, int* offsets, int* active,
                        int* num_active, const int* ids, int T, int K1, const MoeGroupMap& map,
                        int mt_rows, cudaStream_t st, const MoeRouteArgs* route) {
  if (T == 0) return;
  CHECK(map.groups() <= 1024, "moe_dispatch_small: at most 1024 groups");
  if (route) {
    CHECK(T <= kRouterMaxT && K1 == route->top_k + 1 && map.E <= 32 * kRouteWarpMaxPer &&
              K1 <= 32,
          "moe_dispatch_small: routing limits");
    if (map.E <= 32 * 9)
      launch_pdl(moe_dispatch_small_kernel<9>, dim3(1), dim3(1024), 0, st, pair_pos, row_src,
                 offsets, active, num_active, ids, T, K1, map, mt_rows, *route);
    else
      launch_pdl(moe_dispatch_small_kernel<kRouteWarpMaxPer>, dim3(1), dim3(1024), 0, st,
                 pair_pos, row_src, offsets, active, num_active, ids, T, K1, map, mt_rows,
                 *route);
    return;
  }
  launch_pdl(moe_dispatch_small_kernel<0>, dim3(1), dim3(1024), 0, st, pair_pos, row_src, offsets,
             active, num_active, ids, T, K1, map, mt_rows, MoeRouteArgs{});
}

// ------------------------------------------------------------------ combine
__global__ void moe_combine_kernel(bf16* __restrict__ out, const bf16* __restrict__ y,
                                   const float* __restrict__ w, const int* __restrict__ pair_pos,
                                   const bf16* __restrict__ shared, int top_k, int hidden) {
  pdl_entry();
  const int t = blockIdx.x;
  __shared__ int pos[32];
  __shared__ float ws[32];
  if (threadIdx.x < top_k) {
    pos[threadIdx.x] = pair_pos[(size_t)t * top_k + threadIdx.x];
    ws[threadIdx.x] = w[(size_t)t * top_k + threadIdx.x];
  }
  __syncthreads();
  for (int d8 = threadIdx.x; d8 < hidden / 8; d8 += blockDim.x) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s = 0; s < top_k; ++s) {
      if (pos[s] < 0) continue;
      float f[8];
      load8(y + (size_t)pos[s] * hidden + d8 * 8, f);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += ws[s] * f[j];
    }
    if (shared) {
      float f[8];
      load8(shared + (size_t)t * hidden + d8 * 8, f);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] += f[j];
    }
    store8(out + (size_t)t * hidden + d8 * 8, acc);
  }
}

void moe_combine(bf16* out, const bf16* y, const float* topk_w, const int* pair_pos,
                 const bf16* shared, int T, int top_k, int hidden, cudaStream_t st) {
  if (T == 0) return;
  launch_pdl(moe_combine_kernel, dim3(T), dim3(256), 0, st, out, y, topk_w, pair_pos, shared, top_k, hidden);
}

}  // namespace k
}  // namespace glm
