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
// Sequence-parallel boundary collectives over peer memory (see sp_comm.h).
//
// Protocol: block 0 of each call signals every rank (one epoch flag per source rank, after a
// system fence) that this rank's input rows are complete; every block waits for every rank's
// signal, then pulls. The pulled rows are not overwritten before every reader finished: a rank
// rewrites its out rows only after its next all-gather, which waits for every rank's signal of a
// later call.
#include "pdl.cuh"
#include "reduce.cuh"
#include "sp_comm.h"
#include "sp_sync.cuh"

namespace glm {
namespace k {

namespace {
constexpr int kThreads = 512;
constexpr int kBlocks = 132;

__device__ __forceinline__ void st_release_sys(uint32_t* p, uint32_t v) { sp_st_release_sys(p, v); }
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) { return sp_ld_acquire_sys(p); }
__device__ __forceinline__ void red_release_sys_add(uint32_t* p, uint32_t v) {
  sp_red_release_sys_add(p, v);
}

// signal (block 0) and wait for every rank's flag of kind `kind` at the call's epoch
__device__ __forceinline__ void sp_barrier(const SpPeers& p, int tp, int rank, int kind,
                                           uint32_t epoch) {
  if (blockIdx.x == 0 && threadIdx.x < tp) {
    __threadfence_system();
    st_release_sys(p.flags[threadIdx.x] + kind * kSpMaxTp + rank, epoch);
  }
  if (threadIdx.x < tp)
    while (ld_acquire_sys(p.flags[rank] + kind * kSpMaxTp + threadIdx.x) < epoch) {
    }
  __syncthreads();
}

// the last block to finish records the call's epoch
__device__ __forceinline__ void finish(uint32_t* state, int which, uint32_t epoch) {
  __shared__ bool last;
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(&state[2 + which], 1u) == gridDim.x - 1;
  __syncthreads();
  if (last && threadIdx.x == 0) {
    state[which] = epoch;
    state[2 + which] = 0;
    __threadfence();
  }
}
}  // namespace

__global__ void __launch_bounds__(kThreads) sp_rs_pull_kernel(const SpPeers p, int tp, int rank,
                                                              int q0, int nrows, int H,
                                                              uint32_t* state) {
  pdl_entry();
  const uint32_t epoch = state[0] + 1;  // (state[0] changes after every block counted in)
  sp_barrier(p, tp, rank, 0, epoch);
  const int V = H / 8;
  const size_t n = (size_t)nrows * V;
  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t nth = (size_t)gridDim.x * blockDim.x;
  bf16* const mine = p.out[rank];
  for (size_t idx = tid; idx < n; idx += nth) {
    const size_t off = ((size_t)q0 + idx / V) * H + (idx % V) * 8;
    uint4 u[kSpMaxTp];
#pragma unroll
    for (int r = 0; r < kSpMaxTp; ++r)
      if (r < tp) u[r] = __ldcg(reinterpret_cast<const uint4*>(p.out[r] + off));
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int r = 0; r < kSpMaxTp; ++r) {
      if (r >= tp) break;
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u[r]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h[e]);
        acc[2 * e] += f.x;
        acc[2 * e + 1] += f.y;
      }
    }
    store8(mine + off, acc);
  }
  finish(state, 0, epoch);
}

// sp_rs_pull_kernel's sum (rank order, fp32, one bf16 rounding) feeding mhc_post_kernel's
// arithmetic for the same 8 elements of every stream (the same expressions, so bitwise both):
// kU vectors per thread and iteration, every load of them issued before the first use.
template <int kU>
__global__ void __launch_bounds__(kThreads) sp_rs_post_kernel(const SpPeers p, int tp, int rank,
                                                              int q0, int nrows, int H,
                                                              bf16* __restrict__ sq,
                                                              const float* __restrict__ post,
                                                              const float* __restrict__ comb,
                                                              uint32_t* state) {
  pdl_entry();
  constexpr int hc = 4;
  const uint32_t epoch = state[0] + 1;
  sp_barrier(p, tp, rank, 0, epoch);
  const int V = H / 8;
  const size_t n = (size_t)nrows * V;
  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t nth = (size_t)gridDim.x * blockDim.x;
  for (size_t idx0 = tid; idx0 < n; idx0 += kU * nth) {
    uint4 u[kU][kSpMaxTp], sv[kU][hc];
    size_t row[kU];
    int d8[kU];
    bool ok[kU];
#pragma unroll
    for (int w = 0; w < kU; ++w) {
      const size_t idx = idx0 + w * nth;
      ok[w] = idx < n;
      row[w] = ok[w] ? idx / V : 0;
      d8[w] = ok[w] ? (int)(idx % V) : 0;
      const size_t off = ((size_t)q0 + row[w]) * H + d8[w] * 8;
      if (ok[w]) {
#pragma unroll
        for (int r = 0; r < kSpMaxTp; ++r)
          if (r < tp) u[w][r] = __ldcg(reinterpret_cast<const uint4*>(p.out[r] + off));
        const bf16* s = sq + row[w] * hc * H + d8[w] * 8;
#pragma unroll
        for (int i = 0; i < hc; ++i)
          sv[w][i] = __ldcg(reinterpret_cast<const uint4*>(s + (size_t)i * H));
      }
    }
#pragma unroll
    for (int w = 0; w < kU; ++w) {
      if (!ok[w]) continue;
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int r = 0; r < kSpMaxTp; ++r) {
        if (r >= tp) break;
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u[w][r]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 f = __bfloat1622float2(h[e]);
          acc[2 * e] += f.x;
          acc[2 * e + 1] += f.y;
        }
      }
      // mhc_post_kernel with x = bf16(acc)
      const size_t t = row[w];
      float c[hc][hc], pp[hc];
#pragma unroll
      for (int i = 0; i < hc; ++i) {
        pp[i] = post[t * hc + i];
#pragma unroll
        for (int j = 0; j < hc; ++j) c[i][j] = comb[t * hc * hc + i * hc + j];
      }
      float r[hc][8], xv[8];
#pragma unroll
      for (int i = 0; i < hc; ++i) load8(reinterpret_cast<const bf16*>(&sv[w][i]), r[i]);
#pragma unroll
      for (int e = 0; e < 8; ++e) xv[e] = bf16r(acc[e]);
      bf16* s = sq + t * hc * H + d8[w] * 8;
#pragma unroll
      for (int j = 0; j < hc; ++j) {
        float o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float a = 0.f;
#pragma unroll
          for (int i = 0; i < hc; ++i) a += c[i][j] * r[i][e];
          o[e] = a + pp[j] * xv[e];
        }
        store8(s + (size_t)j * H, o);
      }
    }
  }
  finish(state, 0, epoch);
}

// The peers' rows are free to overwrite: every peer read its x for the last time in the sublayer
// before this boundary's reduce-scatter, which this rank passed only after every peer arrived in it.
__global__ void __launch_bounds__(kThreads) sp_ag_push_kernel(const SpPeers p, int tp, int rank,
                                                              int Tq, int T, int H,
                                                              uint32_t* state) {
  pdl_entry();
  const uint32_t epoch = state[1] + 1;
  const int V = H / 8;
  const int r0 = min(T, rank * Tq), r1 = min(T, r0 + Tq);
  const size_t n = (size_t)(r1 - r0) * V, base = (size_t)r0 * V;
  const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const size_t nth = (size_t)gridDim.x * blockDim.x;
  const uint4* const src = reinterpret_cast<const uint4*>(p.x[rank]) + base;
  for (size_t i = tid; i < n; i += nth) {
    const uint4 v = __ldcg(src + i);
#pragma unroll
    for (int k = 1; k < kSpMaxTp; ++k) {
      if (k >= tp) break;
      reinterpret_cast<uint4*>(p.x[(rank + k) % tp])[base + i] = v;
    }
  }
  // this block's stores, counted in at every peer; then every peer's blocks, counted in here
  __syncthreads();
  if (threadIdx.x < tp && threadIdx.x != rank) {
    __threadfence_system();
    red_release_sys_add(p.flags[threadIdx.x] + 2 * kSpMaxTp + rank, 1u);
    const uint32_t want = gridDim.x * epoch;
    while ((int)(ld_acquire_sys(p.flags[rank] + 2 * kSpMaxTp + threadIdx.x) - want) < 0) {
    }
  }
  finish(state, 1, epoch);
}

void sp_rs_pull(const SpPeers& peers, int tp, int rank, int q0, int nrows, int H, uint32_t* state,
                cudaStream_t st) {
  CHECK(H % 8 == 0 && tp <= kSpMaxTp, "sp_rs_pull: shape");
  launch_pdl(sp_rs_pull_kernel, dim3(kBlocks), dim3(kThreads), 0, st, peers, tp, rank, q0,
             std::max(nrows, 0), H, state);
}

void sp_rs_post(const SpPeers& peers, int tp, int rank, int q0, int nrows, int H, bf16* sq,
                const float* post, const float* comb, uint32_t* state, cudaStream_t st) {
  CHECK(H % 8 == 0 && tp <= kSpMaxTp, "sp_rs_post: shape");
  // one vector per thread and iteration, on twice the pulls' blocks
  constexpr int kRsPostBlocks = 2 * kBlocks;
  launch_pdl(sp_rs_post_kernel<1>, dim3(kRsPostBlocks), dim3(kThreads), 0, st, peers, tp, rank, q0,
             std::max(nrows, 0), H, sq, post, comb, state);
}

void sp_ag_push(const SpPeers& peers, int tp, int rank, int Tq, int T, int H, uint32_t* state,
                cudaStream_t st) {
  CHECK(H % 8 == 0 && tp <= kSpMaxTp, "sp_ag_push: shape");
  launch_pdl(sp_ag_push_kernel, dim3(kBlocks), dim3(kThreads), 0, st, peers, tp, rank, Tq, T, H,
             state);
}

}  // namespace k
}  // namespace glm
