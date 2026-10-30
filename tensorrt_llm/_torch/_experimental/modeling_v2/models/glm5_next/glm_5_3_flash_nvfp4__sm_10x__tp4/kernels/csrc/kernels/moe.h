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
// Mixture-of-experts: router logits, routing, the small-batch dispatch and combine.
#pragma once

#include "../common.h"

namespace glm {
namespace k {

// Routing id -> group of this rank's grouped expert GEMMs (-1: not computed here). Routed experts
// [lo, lo + n) are groups 0..n-1; routing id E (the shared expert) is group n, computed for
// (gathered) token t where t % ep == ep_rank. ep = 1, lo = 0, n = E: every expert is local.
struct MoeGroupMap {
  int E = 0, lo = 0, n = 0, ep = 1, ep_rank = 0;
  int t0 = 0;  // index of the first token of this batch among the gathered tokens
  // moe_dispatch_small only: the shared expert's tokens all go to the rank with the fewest
  // distinct routed experts in this batch (every rank sees the same routing ids, so all agree)
  // instead of round-robin by token, so that the slowest rank does not also stream the shared expert.
  int shared_dyn = 0;
  // the shared expert (routing id E): 0: group n here, as above; 1: computed apart from the
  // grouped GEMMs into rows shared_rows + t (moe_permute_rows sets those pair rows; every rank
  // holds a partial of it); 2: not computed by this call
  int shared_mode = 0;
  int shared_rows = 0;
  __host__ __device__ __forceinline__ int operator()(int id, int t) const {
    if (id < 0) return -1;
    if (id >= E) return shared_mode == 0 && (t + t0) % ep == ep_rank ? n : -1;
    const int g = id - lo;
    return g >= 0 && g < n ? g : -1;
  }
  __host__ __device__ int groups() const { return n + 1; }
};

// Decode-sized router logits: logits[T, E] (fp32) = x[T, H] (bf16) . W[E, H]^T (bf16), fp32 sums
// in a fixed order (deterministic). Returns false (nothing launched) unless T <= kRouterMaxT and
// H / 1024 is 2, 4 or 8.
constexpr int kRouterMaxT = 32;
bool moe_router_logits(float* logits, const bf16* x, const bf16* W, int T, int E, int H,
                       cudaStream_t st);
// The same logits on the tensor cores (clusters of 8 CTAs of 128 threads, 8 KB of shared memory):
// returns false unless 1 <= T <= kRouterMaxT, H == 4096 and E % 16 == 0.
// xb (here and in moe_router_mma_ks): the partials' exchange, nonzero (the default) st.async stores (moe.cu's XB),
// 0 plain DSMEM stores and a cluster barrier
bool moe_router_mma(float* logits, const bf16* x, const bf16* W, int T, int E, int H,
                    cudaStream_t st, int xb = -1);
// The same sums handed to the fused MoE kernel's in-kernel routing (MoeFp4Args::rt_ks): per (token, expert) the
// routing choice key and score (route_key_score with bias[e]) as a 64-bit word {key, score bits} at ks[t * E + e];
// 1 <= T <= 8 (one 8-token n-tile), else false (nothing launched)
bool moe_router_mma_ks(unsigned long long* ks, const bf16* x, const bf16* W, const float* bias, int T, int E,
                       int H, cudaStream_t st, int xb = -1);
// RF: moe_router_mma_ks (XB) with its x rows polled in xr, the attention-side boundary's copies
// (ar_rf_x: kRouterRfGroups x [kRouterRfT][H] bf16, the sentinel -0.0 where no rows are pending; -0.0 written as
// +0.0) instead of read after the PDL wait; every lane resets what it read. Bitwise the same words as
// moe_router_mma_ks. mode 1: the key / score stores after griddepcontrol.wait; 2: without a grid wait. False (nothing
// launched) unless 1 <= T <= kRouterRfT, H == 4096 and E == kRouterRfGroups * 16.
bool moe_router_mma_ks_rf(unsigned long long* ks, bf16* xr, const bf16* W, const float* bias, int T, int E, int H,
                          cudaStream_t st, int mode = 1);
// fills n bf16 (n % 8 == 0) of RF copies with the sentinel -0.0
void moe_router_rf_init(bf16* xr, size_t n, cudaStream_t st);
// noaux_tc sigmoid routing (n_group = 1), one warp per token: per token pick top_k experts by
// sigmoid(logit) + bias, weights = normalized sigmoid scores * routed_scale; E <= 512, top_k <= 31.
// With shared_id >= 0 every token gets an extra slot (index top_k) = (shared_id, weight 1).
// tok_slot / skip_slot (optional): tokens whose state slot is skip_slot (CUDA-graph padding and
// idle attention-DP ranks' dummy sequences) get no expert: ids -1, weights 0 in every slot.
void moe_route_warp(int* topk_ids, float* topk_w, const float* logits, const float* bias, int T,
                    int E, int top_k, float routed_scale, cudaStream_t st, int shared_id = -1,
                    const int* tok_slot = nullptr, int skip_slot = -1);
// Routing done by moe_dispatch_small itself (T <= kRouterMaxT): moe_route_warp's routing of the
// logits with shared_id = map.E, written to ids_out / w_out [T, top_k + 1].
struct MoeRouteArgs {
  const float* logits = nullptr;  // [T, E]
  const float* bias = nullptr;    // [E]
  int top_k = 0;
  float scale = 1.f;
  int* ids_out = nullptr;
  float* w_out = nullptr;
  bool ks = false;  // logits holds moe_router_mma_ks's {key, score} words (the fused kernel's routing only)
};

// Small-batch dispatch (T * K1 <= 4096 pairs): counting sort of the (token, slot) pairs by group
// (map(id, token), G = map.groups() groups). Outputs: pair_pos [T*K1] (grouped row of each pair,
// -1 if not computed on this rank), row_src [T*K1] (token of each grouped row, for the GEMM's A
// gather), offsets [G + 1], and the GEMM work items: active[i] = group | (m-tile << 16) for every
// mt_rows-row m-tile of every non-empty group, num_active = item count
// (<= min(G, T*K1) + T*K1 / mt_rows).
// route: compute the ids (ignoring `ids`) from route->logits first (see MoeRouteArgs).
void moe_dispatch_small(int* pair_pos, int* row_src, int* offsets, int* active,
                        int* num_active, const int* ids, int T, int K1, const MoeGroupMap& map,
                        int mt_rows, cudaStream_t st, const MoeRouteArgs* route = nullptr);

// out[t, :] = bf16(sum_s w[t, s] * y[pair_pos[t*top_k + s], :] + shared[t, :]) over the pairs
// with pair_pos >= 0
void moe_combine(bf16* out, const bf16* y, const float* topk_w, const int* pair_pos,
                 const bf16* shared, int T, int top_k, int hidden, cudaStream_t st);

}  // namespace k
}  // namespace glm
