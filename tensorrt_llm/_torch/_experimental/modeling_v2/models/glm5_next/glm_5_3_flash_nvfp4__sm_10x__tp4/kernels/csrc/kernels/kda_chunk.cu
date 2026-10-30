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
// Chunked KDA prefill (gated delta rule with per-channel decay), chunk size 64.
//
// Per head, with S the [K, V] state, D_t = diag(exp(g_t)) and Gamma_t the in-chunk cumulative
// log decay (per channel):
//   S_t = (I - b_t k_t k_t^T) D_t S_{t-1} + b_t k_t v_t^T,    o_t = S_t^T q_t
// Inside a chunk starting from state S0:
//   U = (I + diag(b) L)^-1 diag(b) (V - Bk S0)      L[t,s] = sum_c k_s k_t e^{G_t - G_s} (s < t)
//                                                    Bk[t]  = k_t e^{G_t}
//   O = Qh S0 + M U                                  Qh[t]  = q_t e^{G_t}
//                                                    M[t,s] = sum_c q_t k_s e^{G_t - G_s} (s <= t)
//   S_end = diag(e^{G_C}) S0 + Kd^T U                Kd[s]  = k_s e^{G_C - G_s}
// Every exponent is a decay difference (<= 0), so nothing overflows. A prep pass computes
// W = T^-1 diag(b) Bk, Qh, Kd, M and e^{G_C} for all chunks in parallel; a scan walks each
// sequence's chunks with the state in registers:
//   U = T^-1 diag(b) V - W S;  O = Qh S + M U;  S = e^{G_C} * S + Kd^T U.
// The passes are in kda_chunk_v3.cu.
#include <algorithm>

#include "kda.h"

namespace glm {
namespace k {

namespace {
constexpr int kC = 64;    // chunk length
constexpr int kD = 128;   // head dim
}  // namespace

// kda_chunk_v3.cu: BF16 tensor-core prep + fused state / output scan
size_t kda_chunk3_block_floats();
void kda_chunked3(bf16* o, const bf16* qkv, const bf16* beta_raw, int ld_beta, const bf16* g1,
                  const float* a_log, const float* dt_bias, float* state, const KdaSeqMeta& meta,
                  const int2* chunks, const int* seq_chunks, int num_chunks, float* scratch, int B,
                  int H, float lower_bound, cudaStream_t st, const KdaSnap* snaps, int n_snaps,
                  int b_long);

const float* kda_chunk_states(const float* scratch, int num_chunks, int H) {
  return scratch + (size_t)num_chunks * H * kda_chunk3_block_floats();
}

size_t kda_chunk_scratch_floats(int num_chunks, int H) {
  // per (chunk, head): a block of 5 x 64 x 128 + 64 x 64 + 128 + 128 x 128 floats (more than the v3
  // tile image and the chunk's fp32 initial state take); then the v3 prep / scan overlap's
  // per-(chunk, head) ready flags and two counters
  constexpr size_t kBlkScratch = 5 * kC * kD + kC * kC + kD + kD * kD;
  return (size_t)num_chunks * H * std::max(kBlkScratch, kda_chunk3_block_floats() + kD * kD) +
         (size_t)num_chunks * H + 64;
}

void kda_chunked(bf16* o, const bf16* qkv, const bf16* beta_raw, int ld_beta, const bf16* g1,
                 const float* a_log, const float* dt_bias, float* state, const KdaSeqMeta& meta_in,
                 const int2* chunks, const int* seq_chunks, int num_chunks, float* scratch, int B,
                 int H, int D, float lower_bound, cudaStream_t st, const KdaSnap* snaps,
                 int n_snaps, int b_long) {
  if (B == 0 || num_chunks == 0) return;
  CHECK(D == kD, "kda chunked specialized for head_dim 128");
  const KdaSeqMeta meta = kda_dense_strides(meta_in, H, D, 3 * H * D, 4, 0);
  kda_chunked3(o, qkv, beta_raw, ld_beta, g1, a_log, dt_bias, state, meta, chunks, seq_chunks,
               num_chunks, scratch, B, H, lower_bound, st, snaps, n_snaps, b_long);
}

}  // namespace k
}  // namespace glm
