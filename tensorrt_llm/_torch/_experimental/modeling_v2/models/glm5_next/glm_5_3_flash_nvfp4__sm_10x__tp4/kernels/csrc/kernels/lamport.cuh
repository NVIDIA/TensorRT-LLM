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
// Lamport-protocol device helpers shared by the one-shot all-reduce kernels (allreduce.cu) and
// the kernels that fuse an all-reduce in front of their own work (the mHC boundary).
#pragma once

#include "allreduce.h"

namespace glm {
namespace k {

// Lamport-style one-shot all-reduce: receive slots are pre-filled with the bf16 sentinel -0.0
// (0x8000); senders push their chunk (with -0.0 mapped to +0.0) into every rank's slot and the
// receiver polls the 16-byte chunks until no lane holds the sentinel, so the data itself is the
// synchronization (no flags or system fences). Three buffers rotate with a per-rank call epoch
// (state[0] = epoch, state[1] = elements of the previous call, state[2] = finished blocks):
// call e uses buffer e % 3 and resets the previous call's region of buffer (e - 1) % 3, which
// every rank has finished writing (no rank starts call e + 2 before this rank's call e + 1
// data exists). The last block to finish advances the epoch.
constexpr uint32_t kSentinel2 = 0x80008000u;  // two bf16 -0.0

// The epoch of the call after call e (every kernel of the protocol takes its call's epoch from
// state[0] with it): e + 1, wrapping to 0 at kEpochWrap. The wrap is a multiple of 3 so that the
// buffers keep their order across it; at 2^32, which is not, e % 3 would repeat the last call's
// buffer before its reset, and the next calls would read its data as theirs.
constexpr uint32_t kEpochWrap = 3u << 30;
__device__ __forceinline__ uint32_t next_epoch(uint32_t e) { return e + 1u == kEpochWrap ? 0u : e + 1u; }

__device__ __forceinline__ uint32_t clear_neg_zero(uint32_t w) {
  // map each bf16 lane equal to 0x8000 to 0x0000
  const uint32_t lo = (w & 0xffffu) == 0x8000u ? 0u : (w & 0xffffu);
  const uint32_t hi = (w >> 16) == 0x8000u ? 0u : (w & 0xffff0000u);
  return lo | hi;
}
__device__ __forceinline__ bool has_sentinel(uint32_t w) {
  return (w & 0xffffu) == 0x8000u || (w >> 16) == 0x8000u;
}
__device__ __forceinline__ uint4 ld_volatile16(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// Reset this block's share of the region the previous call (prev_n elements, protocol prev_kind:
// 1 one-shot, 2 two-shot) wrote in buffer `old` back to the sentinel.
// (b / nb: this block's index among the nb blocks sharing the clearing; tid / nthr: the threads
// of the block that take part)
template <int TP>
__device__ __forceinline__ void lamport_clear(const ArPeers& peers, int rank, int old, int prev_n,
                                              uint32_t prev_kind, size_t max_elems, int b, int nb,
                                              int tid, int nthr) {
  const uint4 sent = make_uint4(kSentinel2, kSentinel2, kSentinel2, kSentinel2);
  if (prev_kind == 1) {
    const int chunk = (prev_n / 8 + nb - 1) / nb * 8;
    const int p0 = b * chunk, p1 = min(prev_n, p0 + chunk);
    bf16* clr = peers.buf[rank] + (size_t)old * TP * max_elems;
    for (int r = 0; r < TP; ++r)
      for (int i = p0 + tid * 8; i < p1; i += nthr * 8)
        *reinterpret_cast<uint4*>(clr + (size_t)r * max_elems + i) = sent;
  } else if (prev_kind == 2) {
    const size_t sl = max_elems / TP;
    const int pS = prev_n / TP;
    const int chunk = (pS / 8 + nb - 1) / nb * 8;
    const int p0 = b * chunk, p1 = min(pS, p0 + chunk);
    bf16* clr = peers.buf[rank] + (size_t)5 * TP * max_elems + (size_t)old * 2 * max_elems;
    for (int r = 0; r < 2 * TP; ++r)
      for (int i = p0 + tid * 8; i < p1; i += nthr * 8)
        *reinterpret_cast<uint4*>(clr + (size_t)r * sl + i) = sent;
  }
}

// lamport_clear with its loops kept rolled (the mHC boundaries' clearing, off their chains: one copy of the store
// loop instead of the unrolled ones, so the boundary's code per call fits the SM's instruction cache better); the same
// stores
template <int TP>
__device__ __forceinline__ void lamport_clear_small(const ArPeers& peers, int rank, int old, int prev_n,
                                                    uint32_t prev_kind, size_t max_elems, int b, int nb, int tid,
                                                    int nthr) {
  const uint4 sent = make_uint4(kSentinel2, kSentinel2, kSentinel2, kSentinel2);
  if (prev_kind == 1) {
    const int chunk = (prev_n / 8 + nb - 1) / nb * 8;
    const int p0 = b * chunk, p1 = min(prev_n, p0 + chunk);
    bf16* clr = peers.buf[rank] + (size_t)old * TP * max_elems;
#pragma unroll 1
    for (int r = 0; r < TP; ++r)
#pragma unroll 1
      for (int i = p0 + tid * 8; i < p1; i += nthr * 8)
        *reinterpret_cast<uint4*>(clr + (size_t)r * max_elems + i) = sent;
  } else if (prev_kind == 2) {
    const size_t sl = max_elems / TP;
    const int pS = prev_n / TP;
    const int chunk = (pS / 8 + nb - 1) / nb * 8;
    const int p0 = b * chunk, p1 = min(pS, p0 + chunk);
    bf16* clr = peers.buf[rank] + (size_t)5 * TP * max_elems + (size_t)old * 2 * max_elems;
#pragma unroll 1
    for (int r = 0; r < 2 * TP; ++r)
#pragma unroll 1
      for (int i = p0 + tid * 8; i < p1; i += nthr * 8)
        *reinterpret_cast<uint4*>(clr + (size_t)r * sl + i) = sent;
  }
}

// The last of the nb blocks to finish records the call (epoch, size, protocol) for the next one
// (one thread per block, after the block's clearing).
__device__ __forceinline__ void lamport_finish_thread(uint32_t* state, uint32_t epoch, int n,
                                                      uint32_t kind, int nb) {
  __threadfence();
  if (atomicAdd(&state[2], 1u) == (uint32_t)nb - 1) {
    state[0] = epoch;
    state[1] = (uint32_t)n;
    state[3] = kind;
    state[2] = 0;
    __threadfence();
  }
}


}  // namespace k
}  // namespace glm
