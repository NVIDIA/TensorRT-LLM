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
// Device helpers of the vocab-parallel argmax keys (argmax_keys in sampling.cu, the fused LM head in head_keys.cu):
// both must produce the same key for the same logit, so they share these.
#pragma once

#include <cstdint>

namespace glm {
namespace k {

// The max-reducible key of value v at global vocabulary index i: the value's order-preserving bits above, the
// inverted index below (the larger key wins; equal values: the lower index; never 0).
__device__ __forceinline__ unsigned long long argmax_key_of(float v, int i) {
  const uint32_t u = __float_as_uint(v);
  return ((unsigned long long)((u & 0x80000000u) ? ~u : (u | 0x80000000u)) << 32) |
         (unsigned long long)(0xffffffffu - (uint32_t)i);
}

__device__ __forceinline__ uint4 philox4x32_10(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

// Gumbel(0, 1) from 32 random bits: U = (x >> 9 + 1/2) 2^-23 in (0, 1), E = -log(U) (accurate
// logf: E keeps its relative precision for U near 1), G = -log(E). 23 bits, so that the half is exact in fp32: from
// 24 bits the largest value rounds to U = 1, a G of +inf.
__device__ __forceinline__ float gumbel_of(uint32_t x) {
  const float u = ((float)(x >> 9) + 0.5f) * 0x1p-23f;
  return -__logf(-logf(u));
}

// The Philox counter of the Gumbel noise of the 4-token vocabulary group g (global index / 4) at (step lo, hi)
__device__ __forceinline__ uint4 gumbel_counter(uint32_t g, uint32_t st_lo, uint32_t st_hi) {
  return make_uint4(g, st_lo, st_hi, 0x67756d62u);
}

// uniform in [0, 1) from (slot, position): the same on every rank, a fresh draw every step
__device__ __forceinline__ float accept_uniform(int slot, int pos) {
  unsigned long long x = ((unsigned long long)(uint32_t)slot << 32) ^ (uint32_t)pos;
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x ^= x >> 31;
  return (float)(x >> 40) * (1.f / 16777216.f);
}

// accepted drafts of the sequence with sample rows [r0, r0 + n): the longest prefix of fed drafts
// equal to the target's samples, or (force_accept >= 0, forced acceptance:
// TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS) floor(f) drafts plus one more with probability frac(f),
// whatever the drafts were (the target's samples are emitted). mtp_accept and mtp_pick both count
// with it.
template <bool CG = false>
__device__ __forceinline__ int mtp_accept_count(float force_accept, int r0, int n,
                                                const int* __restrict__ sampled,
                                                const int* __restrict__ ids,
                                                const int* __restrict__ sample_idx,
                                                const int* __restrict__ positions,
                                                const int* __restrict__ token_slot) {
  // (CG: the loads through L2, for a caller that reads them before its PDL wait)
  auto ld = [](const int* p) { return CG ? __ldcg(p) : *p; };
  int a = 0;
  if (force_accept >= 0.f) {
    const int base = (int)force_accept;
    a = min(base, n - 1);
    const int t0 = ld(sample_idx + r0);
    if (a < n - 1 && accept_uniform(ld(token_slot + t0), ld(positions + t0)) < force_accept - (float)base) ++a;
  } else {
    while (a < n - 1 && ld(sampled + r0 + a) == ld(ids + ld(sample_idx + r0 + a + 1))) ++a;
  }
  return a;
}

}  // namespace k
}  // namespace glm
