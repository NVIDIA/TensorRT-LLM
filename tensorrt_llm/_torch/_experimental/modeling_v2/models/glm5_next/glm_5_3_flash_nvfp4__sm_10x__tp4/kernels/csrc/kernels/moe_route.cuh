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
// The decode routing of one token by one warp (moe_route_warp's routing, see moe.h), shared by the
// routing / dispatch kernels (moe.cu) and the fused MoE kernel's in-kernel routing (moe_fp4.cu).
#pragma once

#include "reduce.cuh"

namespace glm {
namespace k {

// Order-preserving key of a routing choice value (finite): unsigned compare = float compare, and
// -0 maps to +0 so that choices equal as floats tie (lower index first). Key 0 is below every
// finite value (removed / absent experts).
__device__ __forceinline__ uint32_t route_key(float f) {
  const uint32_t b = __float_as_uint(f + 0.f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// 1 / d for d in [1, 2^126): the reciprocal approximation and one Newton step, bitwise the IEEE
// quotient there (every float of the range checked); outside it the caller divides
__device__ __forceinline__ float route_rcp_ge1(float d) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(d));
  const float e = __fmaf_rn(-d, r, 1.f);
  return __fmaf_rn(r, e, r);
}

// a lane's three best (key, slot, score) so far, key descending then slot ascending: inserts slot i
// (slots arrive in increasing order, so a key equal to a cached one goes after it)
struct RouteTop3 {
  uint32_t k1, k2, k3;
  int i1, i2, i3;
  float s1, s2, s3;
  __device__ __forceinline__ void clear() {
    k1 = k2 = k3 = 0u;
    i1 = i2 = i3 = 0;
    s1 = s2 = s3 = 0.f;
  }
  __device__ __forceinline__ void insert(uint32_t k, int i, float s) {
    const bool c1 = k > k1, c2 = k > k2, c3 = k > k3;
    k3 = c2 ? k2 : (c3 ? k : k3);
    i3 = c2 ? i2 : (c3 ? i : i3);
    s3 = c2 ? s2 : (c3 ? s : s3);
    k2 = c1 ? k1 : (c2 ? k : k2);
    i2 = c1 ? i1 : (c2 ? i : i2);
    s2 = c1 ? s1 : (c2 ? s : s2);
    k1 = c1 ? k : k1;
    i1 = c1 ? i : i1;
    s1 = c1 ? s : s1;
  }
};

// the routing score of logit x: 1 / (1 + exp(-x)), bitwise; far: the reciprocal's range exceeded
// (a logit below -87: the caller divides)
__device__ __forceinline__ float route_score(float x, bool& far) {
  const float d = 1.f + expf(-x);
  far |= !(d < 0x1p126f);
  return route_rcp_ge1(d);
}

// (a register array's next slot in a rolled loop: rotated by one, back in order after MAXPER)
template <int MAXPER>
__device__ __forceinline__ float route_rot(float* a) {
  const float x = a[0];
#pragma unroll
  for (int j = 0; j + 1 < MAXPER; ++j) a[j] = a[j + 1];
  a[MAXPER - 1] = x;
  return x;
}

// Routes one token (lv[i]: the logit of expert lane + 32 i, loaded by the caller) with the calling
// warp: the top_k experts by choice = sigmoid + bias (descending, lower index first), weights =
// score / (the selected scores' butterfly warp sum + 1e-20) * routed_scale. bias_l[i]: the bias of
// expert lane + 32 i.
// Compact code (the in-kernel routing runs once per launch: its instruction fetch is most of its
// latency): rolled loops over the lane's experts (register arrays rotated), each lane keeping its
// three best; per round one redux max over the lanes' first keys and one redux min over the
// indices holding it, the winning lane shifts its three (a lane out of cached experts with experts
// left rescans them: rare). The scores' reciprocals without the division routine (the division
// only if a logit is below -87). ROLL: the experts' scores in a rolled loop too (less code, less
// instruction-level parallelism: the fused kernel's routing within each CTA takes it, its routing
// over the CTAs does not).
// Writes ids_out / w_out [K1]; returns the lane's id (-1 for lanes >= K1).
// NULLABLE: ids_out / w_out may be null (nothing written)
template <int MAXPER, bool NULLABLE = false, bool ROLL = false>
__device__ __forceinline__ int route_token_warp_lv(const float* lv, const float* bias_l, int E,
                                                   int top_k, float routed_scale, int shared_id,
                                                   int* ids_out, float* w_out, int lane) {
  float lw[MAXPER], bw[MAXPER];
#pragma unroll
  for (int i = 0; i < MAXPER; ++i) lw[i] = lv[i], bw[i] = bias_l[i];
  // the lane's experts (slots i with lane + 32 i < E) and their keys: three cached
  const int nval = E > lane ? min(MAXPER, (E - lane + 31) >> 5) : 0;
  RouteTop3 c;
  c.clear();
  bool far = false;
  if constexpr (ROLL) {
#pragma unroll 1
    for (int i = 0; i < MAXPER; ++i) {
      const float x = route_rot<MAXPER>(lw), bb = route_rot<MAXPER>(bw);
      bool fi = false;
      const float sc = route_score(x, fi);
      if (i < nval) {
        far |= fi;
        c.insert(route_key(sc + bb), i, sc);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < MAXPER; ++i) {
      bool fi = false;
      const float sc = route_score(lw[i], fi);
      if (i < nval) {
        far |= fi;
        c.insert(route_key(sc + bw[i]), i, sc);
      }
    }
  }
  if (__any_sync(0xffffffffu, far)) {  // (logits below -87: the division)
    c.clear();
#pragma unroll 1
    for (int i = 0; i < MAXPER; ++i) {
      const float x = route_rot<MAXPER>(lw), bb = route_rot<MAXPER>(bw);
      const float sc = 1.f / (1.f + expf(-x));
      if (i < nval) c.insert(route_key(sc + bb), i, sc);
    }
  }
  int npop = 0;      // the lane's experts taken
  uint32_t lk = 0u;  // the lane's last taken key and slot (a rescan's threshold)
  int li = -1;
  float my_s = 0.f;
  int my_id = 0;
  // one round: the best first key over the lanes (lowest index among equal ones) is taken
  auto round = [&](int k) {
    const uint32_t mk = __reduce_max_sync(0xffffffffu, c.k1);
    const uint32_t mi =
        __reduce_min_sync(0xffffffffu, c.k1 == mk ? (uint32_t)(lane + 32 * c.i1) : 0xffffffffu);
    const int owner = (int)(mi & 31u);
    const float s = __shfl_sync(0xffffffffu, c.s1, owner);
    my_id = lane == k ? (int)mi : my_id;
    my_s = lane == k ? s : my_s;
    const bool own = lane == owner;
    lk = own ? c.k1 : lk;
    li = own ? c.i1 : li;
    npop += own;
    c.k1 = own ? c.k2 : c.k1, c.i1 = own ? c.i2 : c.i1, c.s1 = own ? c.s2 : c.s1;
    c.k2 = own ? c.k3 : c.k2, c.i2 = own ? c.i3 : c.i2, c.s2 = own ? c.s3 : c.s2;
    c.k3 = own ? 0u : c.k3;
    return owner;
  };
  // rounds until a lane gives its third expert (warp-uniform bookkeeping); from then on (rare) a
  // lane whose cache ran out with experts left rescans its experts after the last taken one
  unsigned m1 = 0u, m2 = 0u;  // lanes taken from at least once / twice
  int k = 0;
  bool third = false;
  for (; k < top_k && !third; ++k) {
    const unsigned ob = 1u << round(k);
    third = (m2 & ob) != 0u;
    m2 |= m1 & ob;
    m1 |= ob;
  }
  if (third) {
    for (;; ++k) {
      if (c.k1 == 0u && npop < nval) {
#pragma unroll 1
        for (int i = 0; i < MAXPER; ++i) {
          const float x = route_rot<MAXPER>(lw), bb = route_rot<MAXPER>(bw);
          bool fi = false;
          float sc = route_score(x, fi);
          if (fi) sc = 1.f / (1.f + expf(-x));
          const uint32_t key = route_key(sc + bb);
          if (i < nval && (key < lk || (key == lk && i > li))) c.insert(key, i, sc);
        }
      }
      if (k >= top_k) break;
      round(k);
    }
  }
  const float v = lane < top_k ? my_s : 0.f;
  const float sum = warp_sum(v);
  if (lane < top_k) {
    if (!NULLABLE || ids_out) {
      ids_out[lane] = my_id;
      w_out[lane] = v / (sum + 1e-20f) * routed_scale;
    }
    return my_id;
  }
  if (lane == top_k && shared_id >= 0) {
    if (!NULLABLE || ids_out) {
      ids_out[lane] = shared_id;
      w_out[lane] = 1.f;
    }
    return shared_id;
  }
  return -1;
}

// The routing choice key and score of one logit x with its bias b, as route_token_warp_lv makes them (bitwise): the
// score's reciprocal by route_rcp_ge1, the IEEE division where the logit is below -87 (route_token_warp_lv divides
// every score of the warp then: route_rcp_ge1 equals the quotient elsewhere)
__device__ __forceinline__ void route_key_score(float x, float b, uint32_t& key, float& sc) {
  bool far = false;
  sc = route_score(x, far);
  if (far) sc = 1.f / (1.f + expf(-x));
  key = route_key(sc + b);
}

// route_token_warp_lv's selection and weights from keys / scores that route_key_score made (kv[i] / sv[i]: expert
// lane + 32 i), bitwise; no score arithmetic left (the in-kernel routing's critical path: the router kernel made them)
template <int MAXPER, bool NULLABLE = false>
__device__ __forceinline__ int route_token_warp_ks(const uint32_t* kv, const float* sv, int E, int top_k,
                                                   float routed_scale, int shared_id, int* ids_out, float* w_out,
                                                   int lane, bool skip_sum = false) {
  float kw[MAXPER], sw[MAXPER];  // (the keys' bits as floats: route_rot's register rotation in the rescan)
#pragma unroll
  for (int i = 0; i < MAXPER; ++i) kw[i] = __uint_as_float(kv[i]), sw[i] = sv[i];
  const int nval = E > lane ? min(MAXPER, (E - lane + 31) >> 5) : 0;
  RouteTop3 c;
  c.clear();
#pragma unroll
  for (int i = 0; i < MAXPER; ++i)
    if (i < nval) c.insert(kv[i], i, sv[i]);
  int npop = 0;
  uint32_t lk = 0u;
  int li = -1;
  float my_s = 0.f;
  int my_id = 0;
  auto round = [&](int k) {
    const uint32_t mk = __reduce_max_sync(0xffffffffu, c.k1);
    const uint32_t mi =
        __reduce_min_sync(0xffffffffu, c.k1 == mk ? (uint32_t)(lane + 32 * c.i1) : 0xffffffffu);
    const int owner = (int)(mi & 31u);
    const float s = __shfl_sync(0xffffffffu, c.s1, owner);
    my_id = lane == k ? (int)mi : my_id;
    my_s = lane == k ? s : my_s;
    const bool own = lane == owner;
    lk = own ? c.k1 : lk;
    li = own ? c.i1 : li;
    npop += own;
    c.k1 = own ? c.k2 : c.k1, c.i1 = own ? c.i2 : c.i1, c.s1 = own ? c.s2 : c.s1;
    c.k2 = own ? c.k3 : c.k2, c.i2 = own ? c.i3 : c.i2, c.s2 = own ? c.s3 : c.s2;
    c.k3 = own ? 0u : c.k3;
    return owner;
  };
  unsigned m1 = 0u, m2 = 0u;
  int k = 0;
  bool third = false;
  for (; k < top_k && !third; ++k) {
    const unsigned ob = 1u << round(k);
    third = (m2 & ob) != 0u;
    m2 |= m1 & ob;
    m1 |= ob;
  }
  if (third) {
    for (;; ++k) {
      if (c.k1 == 0u && npop < nval) {
#pragma unroll 1
        for (int i = 0; i < MAXPER; ++i) {
          const uint32_t key = __float_as_uint(route_rot<MAXPER>(kw));
          const float sc = route_rot<MAXPER>(sw);
          if (i < nval && (key < lk || (key == lk && i > li))) c.insert(key, i, sc);
        }
      }
      if (k >= top_k) break;
      round(k);
    }
  }
  const float v = lane < top_k ? my_s : 0.f;
  // (skip_sum: nothing written -- a warp-uniform choice -- so the weights' butterfly sum is skipped)
  const float sum = NULLABLE && skip_sum && !ids_out ? 0.f : warp_sum(v);
  if (lane < top_k) {
    if (!NULLABLE || ids_out) {
      ids_out[lane] = my_id;
      w_out[lane] = v / (sum + 1e-20f) * routed_scale;
    }
    return my_id;
  }
  if (lane == top_k && shared_id >= 0) {
    if (!NULLABLE || ids_out) {
      ids_out[lane] = shared_id;
      w_out[lane] = 1.f;
    }
    return shared_id;
  }
  return -1;
}

template <int MAXPER, bool NULLABLE = false>
__device__ __forceinline__ int route_token_warp(const float* lg, const float* bias_l, int E,
                                                int top_k, float routed_scale, int shared_id,
                                                int* ids_out, float* w_out, int lane) {
  float lv[MAXPER];
#pragma unroll
  for (int i = 0; i < MAXPER; ++i) {
    const int e = lane + 32 * i;
    lv[i] = e < E ? __ldcg(lg + e) : 0.f;
  }
  return route_token_warp_lv<MAXPER, NULLABLE>(lv, bias_l, E, top_k, routed_scale, shared_id, ids_out,
                                               w_out, lane);
}


}  // namespace k
}  // namespace glm
