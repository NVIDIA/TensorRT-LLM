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
// The MTP layer's chained passes: k::mtp_lamport_norm's summing form folded into the tensor-core MoE router (SR).
#include <cooperative_groups.h>

#include "allreduce.h"
#include "lamport.cuh"
#include "pdl.cuh"
#include "reduce.cuh"

namespace glm {
namespace k {

namespace {
constexpr int kGnTP = 4;

__device__ __forceinline__ uint32_t gn_s32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint4 gn_pack8(const float* f) {
  uint4 u;
  __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) b[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return u;
}
// The call state of the folded consumer (SR): thread 0 of every CTA reads it into shared memory and counts its
// read on state[2] (release); the recorder (one thread of CTA 0) waits for every CTA's count early -- the arrivals are
// microseconds away: the producer lets the grid launch at its PDL wait -- and, after its own PDL wait (the producer,
// which reads the call state until its last push, has completed), records the call with plain stores and resets the
// count: no atomic round trip before the grid's end (the counted form takes an atomicAdd per CTA there).
__device__ __forceinline__ void gn_state_read(const uint32_t* state, uint32_t* sst) {
  sst[0] = __ldcg(state);
  sst[1] = __ldcg(state + 1);
  sst[2] = __ldcg(state + 3);
}
__device__ __forceinline__ void gn_state_count(uint32_t* state) {
  asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" ::"l"(state + 2) : "memory");
}
__device__ __forceinline__ void gn_state_wait(const uint32_t* state, unsigned n) {
  uint32_t v;
  do {
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(state + 2) : "memory");
  } while (v < n);
}
__device__ __forceinline__ void gn_state_record(uint32_t* state, uint32_t epoch, int n) {
  state[0] = epoch;
  state[1] = (uint32_t)n;
  state[3] = 1u;
  state[2] = 0u;
}
}  // namespace

// SR (the MTP layer's chained passes): the o_proj's pushed all-reduce + the residual add + the
// post-attention norm (k::mtp_lamport_norm's summing form) folded into the MoE router (moe_router_mma_kernel<1, false,
// XB>: 8-CTA clusters, cluster g the experts [16 g, 16 g + 16), CTA c the k slice [512 c, 512 c + 512)): one launch and
// one dependent hop (the norm grid's completion, the router's PDL release and its loads of x) fewer per pass.
// The o_proj triggers its dependents only after its own PDL wait, so when this grid starts every earlier kernel has
// completed (the residual, the call state are final). Warp t of every CTA is the summing form's warp c for row t: lane l
// its thread 32 c + l (columns [512 c + 16 l, + 16)): the residual + the 4 ranks' partials polled in their slots, summed
// in rank order (bf16r(v + bf16r(sum))), its squares; the warp's sum of squares travels to every CTA of the cluster by
// st.async onto an mbarrier, and the 8 warp sums of the row are summed as block_sum's last step (lanes 0-7, the same
// tree): the same rr, x = bf16r(v * rr * w) into the CTA's slice of x in shared memory, from which the router's lanes
// take their 32 k (its products, its st.async partials and (CTA, warp)-order sums: bitwise moe_router_mma). Cluster 0
// stores x (the MoE's input) and the residual. The call state: see gn_state_read / gn_state_record.
namespace {
constexpr int kSrCluster = 8, kSrRows = 16, kSrWarps = 4, kSrThreads = 32 * kSrWarps, kSrKc = 512, kSrMaxT = 4;
__device__ __forceinline__ void sr_mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// the summing form's thread for row e / H, columns e % H .. + 16 (mtp_lamport_norm_kernel, !gather): v = the residual +
// the rank-order sum of the 4 slots' partials; returns its sum of squares
__device__ __forceinline__ float sr_row(const bf16* __restrict__ resid, const bf16* mine, size_t max_elems, size_t e,
                                        float* v) {
  {
    const uint4 r0 = __ldcg(reinterpret_cast<const uint4*>(resid + e));
    const uint4 r1 = __ldcg(reinterpret_cast<const uint4*>(resid + e + 8));
    const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&r0);
    const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 a = __bfloat1622float2(h0[q]), b = __bfloat1622float2(h1[q]);
      v[2 * q] = a.x;
      v[2 * q + 1] = a.y;
      v[8 + 2 * q] = b.x;
      v[8 + 2 * q + 1] = b.y;
    }
  }
  uint4 us[2][kGnTP];
  unsigned need = (1u << (2 * kGnTP)) - 1u;
  while (need) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int r = 0; r < kGnTP; ++r)
        if (need >> (h * kGnTP + r) & 1u) us[h][r] = ld_volatile16(mine + (size_t)r * max_elems + e + 8 * h);
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int r = 0; r < kGnTP; ++r)
        if ((need >> (h * kGnTP + r) & 1u) && !(has_sentinel(us[h][r].x) || has_sentinel(us[h][r].y) ||
                                                has_sentinel(us[h][r].z) || has_sentinel(us[h][r].w)))
          need &= ~(1u << (h * kGnTP + r));
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < kGnTP; ++r) {
      const __nv_bfloat162* hq = reinterpret_cast<const __nv_bfloat162*>(&us[h][r]);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 f = __bfloat1622float2(hq[q]);
        acc[2 * q] += f.x;
        acc[2 * q + 1] += f.y;
      }
    }
#pragma unroll
    for (int q = 0; q < 8; ++q) v[8 * h + q] = bf16r(v[8 * h + q] + bf16r(acc[q]));
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) ss += v[i] * v[i];
  return ss;
}
}  // namespace

__global__ void __cluster_dims__(kSrCluster, 1, 1) __launch_bounds__(kSrThreads)
    mtp_sr_router_kernel(float* __restrict__ logits, const bf16* __restrict__ W, int T, int E, int H,
                         const MtpGnArgs g) {
  pdl_trigger();
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  extern __shared__ __align__(16) uint8_t sr_dyn[];
  bf16* xs = reinterpret_cast<bf16*>(sr_dyn);  // [T][512]: this CTA's slice of x
  __shared__ __align__(16) float recv[kSrCluster * kSrWarps][4][4];  // the router's partials (one n-tile)
  __shared__ float ssx[kSrMaxT][kSrCluster];                          // each row's 8 warp sums
  __shared__ uint32_t sst[3];
  __shared__ __align__(8) uint64_t xbar, sbar;
  const int c = (int)cluster.block_rank(), gi = blockIdx.x / kSrCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const bool rec = blockIdx.x == 0 && tid == kSrThreads - 1;  // (the recorder)
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(gn_s32(&xbar)) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(gn_s32(&xbar)),
                 "r"((uint32_t)sizeof(recv))
                 : "memory");
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(gn_s32(&sbar)) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(gn_s32(&sbar)),
                 "r"((uint32_t)(T * kSrCluster * 4))
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    gn_state_read(g.state, sst);
  }
  // (the router) rows e0 / e1 of the CTA's cluster, this lane's 32 contiguous k of the slice (weights: before the wait)
  const int r = lane >> 2, q = lane & 3;
  const int k0l = warp * 128 + q * 32, k0 = c * kSrKc + k0l;
  const int e0 = gi * kSrRows + r, e1 = e0 + 8;
  uint32_t wa[16], wb[16];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 va = e0 < E ? __ldg(reinterpret_cast<const uint4*>(W + (size_t)e0 * H + k0) + i)
                            : make_uint4(0, 0, 0, 0);
    const uint4 vb = e1 < E ? __ldg(reinterpret_cast<const uint4*>(W + (size_t)e1 * H + k0) + i)
                            : make_uint4(0, 0, 0, 0);
    wa[4 * i] = va.x, wa[4 * i + 1] = va.y, wa[4 * i + 2] = va.z, wa[4 * i + 3] = va.w;
    wb[4 * i] = vb.x, wb[4 * i + 1] = vb.y, wb[4 * i + 2] = vb.z, wb[4 * i + 3] = vb.w;
  }
  // (the norm) this lane's columns of the slice and their weights
  const int cl = lane * 16, cg0 = c * kSrKc + cl;
  float wv[16];
  if (warp < T) {
    load8(g.w + cg0, wv);
    load8(g.w + cg0 + 8, wv + 8);
  }
  cluster.sync();  // (every CTA's barriers initialized and armed before any st.async; the call state shared)
  if (tid == 0) gn_state_count(g.state);
  if (rec) gn_state_wait(g.state, gridDim.x);
  const uint32_t epoch = next_epoch(sst[0]);
  const int cur = epoch % 3, old = (epoch + 2) % 3;
  uint32_t xslot, xrbar;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
               : "=r"(xslot)
               : "r"(gn_s32(&recv[c * kSrWarps + warp][lane >> 3][0])), "r"(lane & 7));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(xrbar) : "r"(gn_s32(&xbar)), "r"(lane & 7));
  // the previous call's region back to the sentinel (spread over the grid) while this call's partials are in flight
  lamport_clear<kGnTP>(g.peers, g.rank, old, (int)sst[1], sst[2], g.max_elems, blockIdx.x, gridDim.x, tid,
                       kSrThreads);
  if (warp < T) {  // row t = warp
    const int t = warp;
    const size_t e = (size_t)t * H + cg0;
    float v[16];
    const float ss = warp_sum(sr_row(g.resid, g.peers.buf[g.rank] + (size_t)cur * kGnTP * g.max_elems, g.max_elems,
                                     e, v));
    if (gi == 0) {  // (write_resid)
      store8(g.resid + e, v);
      store8(g.resid + e + 8, v + 8);
    }
    if (lane < kSrCluster) {  // (lane d: to CTA d)
      uint32_t ra, rm;
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(gn_s32(&ssx[t][c])), "r"(lane));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rm) : "r"(gn_s32(&sbar)), "r"(lane));
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n" ::"r"(ra),
                   "r"(__float_as_uint(ss)), "r"(rm)
                   : "memory");
    }
    asm volatile("{\n.reg .pred p;\nSR_SWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra SR_SWAIT;\n}\n"
                 ::"r"(gn_s32(&sbar))
                 : "memory");
    const float rr = rsqrtf(warp_sum(lane < kSrCluster ? ssx[t][lane] : 0.f) / H + g.eps);
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = bf16r(v[i] * rr * wv[i]);
    *reinterpret_cast<uint4*>(xs + t * kSrKc + cl) = gn_pack8(v);
    *reinterpret_cast<uint4*>(xs + t * kSrKc + cl + 8) = gn_pack8(v + 8);
    if (gi == 0) {
      store8(g.out + e, v);
      store8(g.out + e + 8, v + 8);
    }
  }
  __syncthreads();  // (the slice of x)
  // the router's products: token r of the n-tile, this lane's 32 k
  uint32_t xb[16];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = r < T ? *reinterpret_cast<const uint4*>(xs + r * kSrKc + k0l + 8 * i) : make_uint4(0, 0, 0, 0);
    xb[4 * i] = v.x, xb[4 * i + 1] = v.y, xb[4 * i + 2] = v.z, xb[4 * i + 3] = v.w;
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s2 = 0; s2 < 8; ++s2)
    sr_mma(acc, wa[2 * s2], wb[2 * s2], wa[2 * s2 + 1], wb[2 * s2 + 1], xb[2 * s2], xb[2 * s2 + 1]);
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n" ::"r"(
                   xslot),
               "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]), "r"(xrbar)
               : "memory");
  if (tid < 16) {  // outputs (m, j), l = c + 8 m
    asm volatile("{\n.reg .pred p;\nSR_XWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra SR_XWAIT;\n}\n"
                 ::"r"(gn_s32(&xbar))
                 : "memory");
    const int m = (tid >> 2) & 3, j = tid & 3, l = c + 8 * m;
    float v = 0.f;
#pragma unroll
    for (int src = 0; src < kSrCluster * kSrWarps; ++src) v += recv[src][m][j];
    const int e = gi * kSrRows + (l >> 2) + 8 * (j >> 1), t = 2 * (l & 3) + (j & 1);
    if (t < T && e < E) logits[(size_t)t * E + e] = v;
  }
  pdl_wait();
  if (rec) gn_state_record(g.state, epoch, T * H);
  // (no CTA exits while a peer may still store into its shared memory: every st.async into this CTA has completed
  // once its barriers completed -- sbar by the rows' warps, xbar by threads 0-15)
}

bool mtp_sr_ok(int T, int hidden, int tp, size_t max_elems, int E) {
  return T >= 1 && T <= kSrMaxT && hidden == kSrCluster * kSrKc && tp == kGnTP && E % kSrRows == 0 &&
         mtp_lamport_norm_ok(T, hidden, tp, max_elems);
}

void mtp_sr_router(float* logits, const bf16* W, int T, int E, int H, const MtpGnArgs& g, cudaStream_t st) {
  CHECK(mtp_sr_ok(T, H, kGnTP, g.max_elems, E) && g.state && g.w && g.out && g.resid,
        "mtp_sr_router: unsupported shape / arguments");
  launch_pdl(mtp_sr_router_kernel, dim3((E / kSrRows) * kSrCluster), dim3(kSrThreads), (size_t)T * kSrKc * 2, st,
             logits, W, T, E, H, g);
}

}  // namespace k
}  // namespace glm
