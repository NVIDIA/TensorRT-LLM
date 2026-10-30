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
// The tensor-core router's RF form (moe_router_mma_ks_rf), in a translation unit of its own so that the code generated
// for moe.cu's router kernels does not depend on it.
#include "pdl.cuh"
#include <cooperative_groups.h>

#include "allreduce.h"
#include "moe.h"
#include "moe_route.cuh"

namespace glm {
namespace k {

// (moe.cu's moe_router_mma_kernel geometry: clusters of 8 CTAs, 16 experts per cluster, 4 warps, 512 k per CTA)
constexpr int kRmCluster = 8, kRmRows = 16, kRmWarps = 4, kRmThreads = 32 * kRmWarps;
constexpr int kRmKc = 512;
static_assert(kRouterRfGroups * kRmRows == 288, "RF: one copy per expert group of 288 experts");

static __device__ __forceinline__ uint32_t rf_smem(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
static __device__ __forceinline__ void rf_mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// RF (moe_router_mma_ks_rf): moe_router_mma_kernel<1, KS, XB> with its x rows taken by the Lamport
// protocol from the attention-side boundary's copies (k::ar_rf_x: one copy [kRouterRfT][H] per expert group, filled
// with the bf16 sentinel -0.0 except while a call's rows are in it; the boundary writes -0.0 as +0.0) instead of after
// the PDL wait: the MMA starts once this lane's 64 bytes are in (one L2 round trip after the boundary's stores), not
// when the boundary's grid has completed. Each lane resets the bytes it read to the sentinel (its group's copy has no
// other reader). The same weights, MMAs, partials, sums and keys as moe_router_mma_kernel<1, true, true> (a +0.0 for a
// -0.0 operand changes no fp32 sum here: every accumulator starts at +0.0), so the {key, score} words are bitwise its
// own. RF 1: the key / score stores wait for the boundary's grid (griddepcontrol.wait: the PDL chain to the MoE kernel
// unchanged); RF 2: no grid wait (the stores as soon as the sums are done).
static __device__ __forceinline__ uint4 rf_ld16(const void* p) {  // (every poll a new L2 load)
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
// nonzero iff a bf16 lane of u holds the sentinel 0x8000 (a zero half of u ^ sentinel; exact for "any lane")
static __device__ __forceinline__ uint32_t rf_sent(const uint4& u) {
  const uint32_t a = u.x ^ 0x80008000u, b = u.y ^ 0x80008000u, c = u.z ^ 0x80008000u, d = u.w ^ 0x80008000u;
  const uint32_t m = ((a - 0x00010001u) & ~a) | ((b - 0x00010001u) & ~b) | ((c - 0x00010001u) & ~c) |
                     ((d - 0x00010001u) & ~d);
  return m & 0x80008000u;
}
template <int RF>
__global__ void __cluster_dims__(kRmCluster, 1, 1) __launch_bounds__(kRmThreads)
    moe_router_rf_kernel(unsigned long long* __restrict__ ks, bf16* __restrict__ xr, const bf16* __restrict__ W,
                         int T, int E, int H, const float* __restrict__ bias) {
  pdl_trigger();
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  __shared__ __align__(16) float recv[kRmCluster * kRmWarps][1][4][4];
  __shared__ __align__(8) uint64_t xbar;
  const int c = (int)cluster.block_rank(), g = blockIdx.x / kRmCluster;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(rf_smem(&xbar)) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(rf_smem(&xbar)),
                 "r"((uint32_t)sizeof(recv))
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  const int r = lane >> 2, q = lane & 3;
  const int k0 = c * kRmKc + warp * 128 + q * 32;
  const int e0 = g * kRmRows + r, e1 = e0 + 8;
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
  float be = 0.f;
  {
    const int l = c + 8 * ((tid >> 2) & 3), e = g * kRmRows + (l >> 2) + 8 * ((tid & 3) >> 1);
    if (tid < 16 && e < E) be = __ldg(bias + e);
  }
  uint32_t xslot, xrbar;
  cluster.sync();
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
               : "=r"(xslot)
               : "r"(rf_smem(&recv[c * kRmWarps + warp][0][lane >> 3][0])), "r"(lane & 7));
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(xrbar) : "r"(rf_smem(&xbar)), "r"(lane & 7));
  // token r's 32 k of this lane: polled in its group's copy, then reset to the sentinel
  uint32_t xb[16];
  if (r < T) {
    uint4* const src = reinterpret_cast<uint4*>(xr + ((size_t)g * kRouterRfT + r) * H + k0);
    uint4 v0, v1, v2, v3;
    do {
      v0 = rf_ld16(src);
      v1 = rf_ld16(src + 1);
      v2 = rf_ld16(src + 2);
      v3 = rf_ld16(src + 3);
    } while ((rf_sent(v0) | rf_sent(v1) | rf_sent(v2) | rf_sent(v3)) != 0u);
    const uint4 sent = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
    src[0] = sent;
    src[1] = sent;
    src[2] = sent;
    src[3] = sent;
    xb[0] = v0.x, xb[1] = v0.y, xb[2] = v0.z, xb[3] = v0.w;
    xb[4] = v1.x, xb[5] = v1.y, xb[6] = v1.z, xb[7] = v1.w;
    xb[8] = v2.x, xb[9] = v2.y, xb[10] = v2.z, xb[11] = v2.w;
    xb[12] = v3.x, xb[13] = v3.y, xb[14] = v3.z, xb[15] = v3.w;
  } else {
#pragma unroll
    for (int i = 0; i < 16; ++i) xb[i] = 0u;
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s = 0; s < 8; ++s)
    rf_mma(acc, wa[2 * s], wb[2 * s], wa[2 * s + 1], wb[2 * s + 1], xb[2 * s], xb[2 * s + 1]);
  // (this lane's outputs (l, j) to CTA l % 8, slot [c, warp][l / 8][j]: moe_router_mma_kernel's XB exchange)
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n"
               ::"r"(xslot), "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]), "r"(xrbar)
               : "memory");
  if (tid < 16) {  // (outputs (l = c + 8 m, j): thread (m, j); the sums in (CTA, warp) order)
    asm volatile("{\n.reg .pred p;\nRF_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
                 "@!p bra RF_WAIT;\n}\n" ::"r"(rf_smem(&xbar))
                 : "memory");
    const int m = (tid >> 2) & 3, j = tid & 3, l = c + 8 * m;
    float v = 0.f;
#pragma unroll
    for (int src = 0; src < kRmCluster * kRmWarps; ++src) v += recv[src][0][m][j];
    const int e = g * kRmRows + (l >> 2) + 8 * (j >> 1), t = 2 * (l & 3) + (j & 1);
    if (t < T && e < E) {
      uint32_t key;
      float sc;
      route_key_score(v, be, key, sc);
      if constexpr (RF == 1) pdl_wait();
      reinterpret_cast<uint2*>(ks)[(size_t)t * E + e] = make_uint2(key, __float_as_uint(sc));
    }
  }
}

bool moe_router_mma_ks_rf(unsigned long long* ks, bf16* xr, const bf16* W, const float* bias, int T, int E, int H,
                          cudaStream_t st, int mode) {
  if (T <= 0 || T > kRouterRfT || H != kRmCluster * kRmKc || E != kRouterRfGroups * kRmRows) return false;
  const dim3 grid((E / kRmRows) * kRmCluster), block(kRmThreads);
  if (mode == 2)
    launch_pdl(moe_router_rf_kernel<2>, grid, block, 0, st, ks, xr, W, T, E, H, bias);
  else
    launch_pdl(moe_router_rf_kernel<1>, grid, block, 0, st, ks, xr, W, T, E, H, bias);
  return true;
}

__global__ void moe_router_rf_fill_kernel(uint4* __restrict__ p, size_t n16) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
    p[i] = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
}
void moe_router_rf_init(bf16* xr, size_t n, cudaStream_t st) {
  CHECK(n % 8 == 0, "moe_router_rf_init: whole 16-byte chunks");
  moe_router_rf_fill_kernel<<<64, 256, 0, st>>>(reinterpret_cast<uint4*>(xr), n / 8);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace k
}  // namespace glm
