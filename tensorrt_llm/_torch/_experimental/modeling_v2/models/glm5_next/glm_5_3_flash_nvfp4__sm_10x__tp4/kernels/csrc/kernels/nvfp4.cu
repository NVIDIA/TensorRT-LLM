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
// NVFP4 activation quantization and the MoE row permutation for the grouped GEMMs (see nvfp4.h).
#include "nvfp4.h"
#include "nvfp4_q.cuh"
#include "pdl.cuh"

namespace glm {
namespace k {

namespace {

using nvq::load16_bf16;
using nvq::quant16;

__global__ void nvfp4_quant_kernel(uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                   const bf16* __restrict__ x, int ldx, int K, float sfs,
                                   bool swz) {
  pdl_entry();
  const int m = blockIdx.x, nb = K / 16;
  for (int kb = threadIdx.x; kb < nb; kb += blockDim.x) {
    float v[16];
    load16_bf16(x + (size_t)m * ldx + kb * 16, v);
    uint2 o;
    const uint8_t s = quant16(v, sfs, o);
    *reinterpret_cast<uint2*>(q + (size_t)m * (K / 2) + kb * 8) = o;
    sf[swz ? nvfp4_sf_off(m, kb, K) : (size_t)m * nb + kb] = s;
  }
}

// as swiglu1 in elementwise.cu (clamp, SiLU, product rounded to bf16), with the approximate
// division (div.approx: x * rcp(y)): the IEEE division makes the grouped kernel instruction-bound
__device__ __forceinline__ float swiglu1_fast(float g, float u, float limit) {
  g = fminf(g, limit);
  u = fminf(fmaxf(u, -limit), limit);
  return __bfloat162float(__float2bfloat16(__fdividef(g, 1.f + __expf(-g)) * u));
}

// Dense form (the dense MLP layers: no groups): one CTA per row, one thread per 16-feature block
// (the warp-per-row kernel below has too few warps for the decode steps' 4-32 rows).
__global__ void swiglu_quant_nvfp4_dense_kernel(uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                                const bf16* __restrict__ gu, int I, float limit,
                                                float sfs, const int* __restrict__ m_dev,
                                                bool gu64) {
  pdl_entry();
  const int r = blockIdx.x;
  if (m_dev && r >= *m_dev) return;
  const int nb = I / 16;
  for (int kb = threadIdx.x; kb < nb; kb += blockDim.x) {
    float g[16], u[16], v[16];
    // (16 features never straddle a 64-feature interleave block)
    const int gc = gu64 ? (kb * 16 / kGu64) * 2 * kGu64 + kb * 16 % kGu64 : kb * 16;
    const int uc = gu64 ? gc + kGu64 : I + kb * 16;
    load16_bf16(gu + (size_t)r * 2 * I + gc, g);
    load16_bf16(gu + (size_t)r * 2 * I + uc, u);
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = swiglu1_fast(g[i], u[i], limit);
    uint2 o;
    const uint8_t s = quant16(v, sfs, o);
    *reinterpret_cast<uint2*>(q + (size_t)r * (I / 2) + kb * 8) = o;
    sf[nvfp4_sf_off(r, kb, I)] = s;
  }
}

// Grouped form (the MoE's permuted rows): one warp per row (grid-stride), the groups' offsets /
// padded scale offsets staged in shared memory (a launch over every possible grouped row with a
// binary search in global memory per row is latency-bound)
__global__ void __launch_bounds__(256) swiglu_quant_nvfp4_kernel(
    uint8_t* __restrict__ q, uint8_t* __restrict__ sf, const bf16* __restrict__ gu, int M, int I,
    float limit, float sfs, const int* __restrict__ offsets, const int* __restrict__ pad, int G,
    const int* __restrict__ m_dev, bool gu64) {
  pdl_entry();
  extern __shared__ int s_tab[];  // [G + 1] offsets, then [G + 1] padded offsets
  int n_rows = M;
  if (offsets) {
    for (int i = threadIdx.x; i <= G; i += blockDim.x) {
      s_tab[i] = offsets[i];
      s_tab[G + 1 + i] = pad[i];
    }
    __syncthreads();
    n_rows = min(n_rows, s_tab[G]);
  }
  if (m_dev) n_rows = min(n_rows, *m_dev);
  const int lane = threadIdx.x & 31;
  const int wpb = blockDim.x >> 5, warps = gridDim.x * wpb;
  const int nb = I / 16;
  for (int r = blockIdx.x * wpb + (threadIdx.x >> 5); r < n_rows; r += warps) {
    int rr = r;
    size_t sf_base = 0;
    if (offsets) {
      int lo = 0, hi = G - 1;  // last group with offsets[g] <= r
      while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (s_tab[mid] <= r)
          lo = mid;
        else
          hi = mid - 1;
      }
      rr = r - s_tab[lo];
      sf_base = (size_t)s_tab[G + 1 + lo] * nb;
    }
    for (int kb = lane; kb < nb; kb += 32) {
      float g[16], u[16], v[16];
      // (16 features never straddle a 64-feature interleave block)
      const int gc = gu64 ? (kb * 16 / kGu64) * 2 * kGu64 + kb * 16 % kGu64 : kb * 16;
      const int uc = gu64 ? gc + kGu64 : I + kb * 16;
      load16_bf16(gu + (size_t)r * 2 * I + gc, g);
      load16_bf16(gu + (size_t)r * 2 * I + uc, u);
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = swiglu1_fast(g[i], u[i], limit);
      uint2 o;
      const uint8_t s = quant16(v, sfs, o);
      *reinterpret_cast<uint2*>(q + (size_t)r * (I / 2) + kb * 8) = o;
      sf[sf_base + nvfp4_sf_off(rr, kb, I)] = s;
    }
  }
}

// One CTA per 256 pairs: a shared-memory histogram, then one global atomic per (CTA, group) (the
// per-pair global atomics on ~72 counters serialized at prefill sizes)
__global__ void permute_hist_kernel(int* __restrict__ counts, const int* __restrict__ ids, int n,
                                    int K1, const MoeGroupMap map, int G) {
  pdl_entry();
  extern __shared__ int s_hist[];
  for (int i = threadIdx.x; i < G; i += blockDim.x) s_hist[i] = 0;
  __syncthreads();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int g = map(ids[i], i / K1);
    if (g >= 0) atomicAdd(&s_hist[g], 1);
  }
  __syncthreads();
  for (int g = threadIdx.x; g < G; g += blockDim.x)
    if (s_hist[g]) atomicAdd(&counts[g], s_hist[g]);
}

// Single block: offsets / pad = exclusive scans of counts / counts rounded up to 128; counts reset
// to 0 for use as cursors.
__global__ void permute_scan_kernel(int* __restrict__ counts, int* __restrict__ offsets,
                                    int* __restrict__ pad, int G) {
  pdl_entry();
  __shared__ int a[1024], b[1024];
  const int t = threadIdx.x;
  const int c = t < G ? counts[t] : 0;
  a[t] = c;
  b[t] = (c + 127) & ~127;
  __syncthreads();
  for (int o = 1; o < 1024; o <<= 1) {
    const int va = t >= o ? a[t - o] : 0;
    const int vb = t >= o ? b[t - o] : 0;
    __syncthreads();
    a[t] += va;
    b[t] += vb;
    __syncthreads();
  }
  if (t < G) {
    offsets[t + 1] = a[t];
    pad[t + 1] = b[t];
    counts[t] = 0;
  }
  if (t == 0) offsets[0] = pad[0] = 0;
}

// One warp per (token, slot) pair: claim a row of the pair's group and copy the token's row
// (NVFP4: packed values + scales into the group's CUTLASS-layout scales; BF16: the row).
__global__ void permute_scatter_kernel(uint8_t* __restrict__ x_perm, uint8_t* __restrict__ sf_perm,
                                       int* __restrict__ pair_pos, int* __restrict__ cursor,
                                       const int* __restrict__ offsets,
                                       const int* __restrict__ pad, const uint8_t* __restrict__ x,
                                       const uint8_t* __restrict__ sx,
                                       const int* __restrict__ ids, int n_pairs, int K1, int K,
                                       bool nvfp4, const MoeGroupMap map) {
  pdl_entry();
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n_pairs) return;
  const int t = warp / K1;
  int pos = -1, g = -1;
  if (lane == 0) {
    const int id = ids[warp];
    g = map(id, t);
    if (g >= 0)
      pos = offsets[g] + atomicAdd(&cursor[g], 1);
    else if (id >= map.E && map.shared_mode == 1)
      pos = map.shared_rows + t;
    pair_pos[warp] = pos;
  }
  pos = __shfl_sync(0xffffffff, pos, 0);
  g = __shfl_sync(0xffffffff, g, 0);
  if (g < 0) return;
  const size_t row_bytes = nvfp4 ? (size_t)K / 2 : (size_t)K * 2;
  const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)t * row_bytes);
  uint4* dst = reinterpret_cast<uint4*>(x_perm + (size_t)pos * row_bytes);
  for (int i = lane; i < (int)(row_bytes / 16); i += 32) dst[i] = src[i];
  if (!nvfp4) return;
  const int nb = K / 16, rr = pos - offsets[g];
  uint8_t* sfg = sf_perm + (size_t)pad[g] * nb;
  if ((nb & 3) == 0) {
    // the 4 consecutive scale bytes of blocks 4 j .. 4 j + 3 (one 32-bit word in both layouts)
    for (int j = lane; j < nb / 4; j += 32)
      *reinterpret_cast<uint32_t*>(sfg + nvfp4_sf_off(rr, 4 * j, K)) =
          *reinterpret_cast<const uint32_t*>(sx + nvfp4_sf_off(t, 4 * j, K));
    return;
  }
  for (int kb = lane; kb < nb; kb += 32) sfg[nvfp4_sf_off(rr, kb, K)] = sx[nvfp4_sf_off(t, kb, K)];
}

}  // namespace

void nvfp4_quant(uint8_t* q, uint8_t* sf, const bf16* x, int ldx, int M, int K, float gs,
                 bool swz, cudaStream_t st) {
  if (M == 0) return;
  CHECK(K % 16 == 0 && ldx % 8 == 0, "nvfp4_quant: K % 16, ldx % 8");
  const int threads = std::min(256, std::max(32, K / 16));
  launch_pdl(nvfp4_quant_kernel, dim3(M), dim3(threads), 0, st, q, sf, x, ldx, K, 1.f / gs, swz);
}

void swiglu_quant_nvfp4(uint8_t* q, uint8_t* sf, const bf16* gu, int M, int I, float limit,
                        float gs, cudaStream_t st, const int* offsets, const int* pad, int G,
                        const int* m_dev, bool gu64) {
  if (M == 0) return;
  CHECK(I % 64 == 0, "swiglu_quant_nvfp4: I % 64");
  if (!offsets) {
    const int threads = std::min(256, std::max(32, I / 16));
    launch_pdl(swiglu_quant_nvfp4_dense_kernel, dim3(M), dim3(threads), 0, st, q, sf, gu, I, limit,
               1.f / gs, m_dev, gu64);
    return;
  }
  static int sms = 0;
  if (sms == 0) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  }
  const size_t smem = offsets ? 2 * (size_t)(G + 1) * sizeof(int) : 0;
  // up to 8 rows per CTA (one per warp), at most 8 CTAs per SM
  const int blocks = (int)std::min<int64_t>(cdiv(M, 8), 8 * (int64_t)sms);
  launch_pdl(swiglu_quant_nvfp4_kernel, dim3(blocks), dim3(256), smem, st, q, sf, gu, M, I, limit,
             1.f / gs, offsets, pad, G, m_dev, gu64);
}

void moe_permute_rows(uint8_t* x_perm, uint8_t* sf_perm, int* pair_pos, int* counts,
                      int* offsets, int* pad, const uint8_t* x, const uint8_t* sx,
                      const int* topk_ids, int T, int K1, const MoeGroupMap& map, int G, int K,
                      bool nvfp4, cudaStream_t st) {
  CHECK(G <= 1024, "moe_permute_rows: too many groups");
  CUDA_CHECK(cudaMemsetAsync(counts, 0, G * sizeof(int), st));
  const int n = T * K1;
  if (n > 0)
    launch_pdl(permute_hist_kernel, dim3(cdiv(n, 256)), dim3(256), (size_t)G * sizeof(int), st,
               counts, topk_ids, n, K1, map, G);
  launch_pdl(permute_scan_kernel, dim3(1), dim3(1024), 0, st, counts, offsets, pad, G);
  if (n > 0)
    launch_pdl(permute_scatter_kernel, dim3(cdiv(n, 8)), dim3(256), 0, st, x_perm, sf_perm,
               pair_pos, counts, offsets, pad, x, sx, topk_ids, n, K1, K, nvfp4, map);
}

}  // namespace k
}  // namespace glm
