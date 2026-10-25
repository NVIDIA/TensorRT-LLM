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
// Token-split MLA prefill attention exchanges (see mla_ts.h).
//
// Protocol (as sp_ag_push): every block of a call stores its share of the rows into the
// destination ranks' buffers, then counts itself in at every rank (one counter per source rank,
// after a system fence); every block then waits until every source's blocks of this call arrived
// here. The destination buffers are free when the stores land: the q staging (b.out) was last read
// by the peers' reduce-scatter pulls of the boundary before this sublayer, which every rank passed
// before its all-gather completed; the output rows (b.x) were last read by the destination's own
// projections, which precede its q push (that every attention waits for).
#include "mla_ts.h"
#include "pdl.cuh"

namespace glm {
namespace k {

namespace {
constexpr int kThreads = 512;
constexpr int kBlocks = 132;

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_sys_add(uint32_t* p, uint32_t v) {
  asm volatile("red.release.sys.global.add.u32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}

// count this block in at every rank, wait for every rank's blocks here; the last block to finish
// records the call's epoch
__device__ __forceinline__ void arrive_wait_finish(const TsPeers& p, int tp, int rank, int kind,
                                                   uint32_t epoch, uint32_t* state) {
  __syncthreads();
  if (threadIdx.x < tp) {
    __threadfence_system();
    red_release_sys_add(p.flags[threadIdx.x] + kind * kSpMaxTp + rank, 1u);
    const uint32_t want = gridDim.x * epoch;
    while ((int)(ld_acquire_sys(p.flags[rank] + kind * kSpMaxTp + threadIdx.x) - want) < 0) {
    }
  }
  __shared__ bool last;
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(&state[2 + kind], 1u) == gridDim.x - 1;
  __syncthreads();
  if (last && threadIdx.x == 0) {
    state[kind] = epoch;
    state[2 + kind] = 0;
    __threadfence();
  }
}
}  // namespace

__global__ void __launch_bounds__(kThreads) mla_ts_q_push_kernel(const TsPeers p, int tp, int rank,
                                                                 const bf16* __restrict__ q, int T,
                                                                 int Tq, int hn, uint32_t* state) {
  pdl_entry();
  const uint32_t epoch = state[0] + 1;
  const int V = hn / 8;  // 16-byte vectors of one row piece
  // the blocks take the destinations in turn (all of them at once rather than one peer at a time)
  const int k = blockIdx.x % tp, r = (rank + 1 + k) % tp;
  const int nb = ((int)gridDim.x - k + tp - 1) / tp;
  const size_t tid = (size_t)(blockIdx.x / tp) * blockDim.x + threadIdx.x;
  const size_t nth = (size_t)nb * blockDim.x;
  const uint4* const src = reinterpret_cast<const uint4*>(q);
  const int r0 = r * Tq, rows = max(0, min(Tq, T - r0));
  uint4* const dst = reinterpret_cast<uint4*>(p.out[r]) + (size_t)rank * V;
  const size_t n = (size_t)rows * V;
  size_t idx = tid;
  for (; idx + nth < n; idx += 2 * nth) {  // two vectors in flight per thread
    const size_t i0 = idx, i1 = idx + nth;
    const uint4 v0 = __ldcg(src + (size_t)(r0 + i0 / V) * V + i0 % V);
    const uint4 v1 = __ldcg(src + (size_t)(r0 + i1 / V) * V + i1 % V);
    dst[(i0 / V) * (size_t)(tp * V) + i0 % V] = v0;
    dst[(i1 / V) * (size_t)(tp * V) + i1 % V] = v1;
  }
  for (; idx < n; idx += nth)
    dst[(idx / V) * (size_t)(tp * V) + idx % V] = __ldcg(src + (size_t)(r0 + idx / V) * V + idx % V);
  arrive_wait_finish(p, tp, rank, 0, epoch, state);
}

__global__ void __launch_bounds__(kThreads) mla_ts_o_push_kernel(const TsPeers p, int tp, int rank,
                                                                 const bf16* __restrict__ o, int Tr,
                                                                 int Tq, int hv, uint32_t* state,
                                                                 int wait_ready) {
  pdl_entry();
  const uint32_t epoch = state[1] + 1;
  const int V = hv / 8;
  const int k = blockIdx.x % tp, d = (rank + 1 + k) % tp;
  if (wait_ready && d != rank) {  // rank d's x is free once its projections ran
    if (threadIdx.x == 0) {
      const uint32_t want = state[4];
      while ((int)(ld_acquire_sys(p.flags[rank] + 2 * kSpMaxTp + d) - want) < 0) {
      }
    }
    __syncthreads();
  }
  const int nb = ((int)gridDim.x - k + tp - 1) / tp;
  const size_t tid = (size_t)(blockIdx.x / tp) * blockDim.x + threadIdx.x;
  const size_t nth = (size_t)nb * blockDim.x;
  const uint4* const src = reinterpret_cast<const uint4*>(o) + (size_t)d * V;
  uint4* const dst = reinterpret_cast<uint4*>(p.x[d]) + (size_t)rank * Tq * V;
  const size_t n = (size_t)Tr * V;
  size_t idx = tid;
  for (; idx + nth < n; idx += 2 * nth) {
    const size_t i0 = idx, i1 = idx + nth;
    const uint4 v0 = __ldcg(src + (i0 / V) * (size_t)(tp * V) + i0 % V);
    const uint4 v1 = __ldcg(src + (i1 / V) * (size_t)(tp * V) + i1 % V);
    dst[i0] = v0;
    dst[i1] = v1;
  }
  for (; idx < n; idx += nth) dst[idx] = __ldcg(src + (idx / V) * (size_t)(tp * V) + idx % V);
  arrive_wait_finish(p, tp, rank, 1, epoch, state);
}

void mla_ts_q_push(const TsPeers& peers, int tp, int rank, const bf16* q, int T, int Tq, int hn,
                   uint32_t* state, cudaStream_t st) {
  CHECK(hn % 8 == 0 && tp <= kSpMaxTp, "mla_ts_q_push: shape");
  launch_pdl(mla_ts_q_push_kernel, dim3(kBlocks), dim3(kThreads), 0, st, peers, tp, rank, q, T, Tq,
             hn, state);
}

void mla_ts_o_push(const TsPeers& peers, int tp, int rank, const bf16* o, int Tr, int Tq, int hv,
                   uint32_t* state, cudaStream_t st, bool wait_ready) {
  CHECK(hv % 8 == 0 && tp <= kSpMaxTp, "mla_ts_o_push: shape");
  launch_pdl(mla_ts_o_push_kernel, dim3(kBlocks), dim3(kThreads), 0, st, peers, tp, rank, o,
             std::max(Tr, 0), Tq, hv, state, wait_ready ? 1 : 0);
}

__global__ void mla_ts_ready_kernel(const TsPeers p, int tp, int rank, uint32_t* state) {
  pdl_entry();
  const uint32_t epoch = state[4] + 1;
  if (threadIdx.x < tp) {
    __threadfence_system();
    red_release_sys_add(p.flags[threadIdx.x] + 2 * kSpMaxTp + rank, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0) state[4] = epoch;
}

void mla_ts_ready(const TsPeers& peers, int tp, int rank, uint32_t* state, cudaStream_t st) {
  CHECK(tp <= kSpMaxTp, "mla_ts_ready: tp");
  launch_pdl(mla_ts_ready_kernel, dim3(1), dim3(32), 0, st, peers, tp, rank, state);
}

__global__ void mla_ts_wait_ready_kernel(const TsPeers p, int tp, int rank, uint32_t* state) {
  pdl_entry();
  const uint32_t want = state[4];
  if (threadIdx.x < tp && threadIdx.x != rank)
    while ((int)(ld_acquire_sys(p.flags[rank] + 2 * kSpMaxTp + threadIdx.x) - want) < 0) {
    }
  __syncthreads();
}

void mla_ts_wait_ready(const TsPeers& peers, int tp, int rank, uint32_t* state, cudaStream_t st) {
  CHECK(tp <= kSpMaxTp, "mla_ts_wait_ready: tp");
  launch_pdl(mla_ts_wait_ready_kernel, dim3(1), dim3(32), 0, st, peers, tp, rank, state);
}

}  // namespace k
}  // namespace glm
