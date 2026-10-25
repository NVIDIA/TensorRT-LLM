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
// FlashMLA's SM100 sparse prefill attention (third_party/flashmla, MIT) behind a plain-pointer interface:
// the vendored headers define their own bf16 type, so they are compiled in fmla_fwd.cu only.
#pragma once

#include <cuda_runtime.h>

namespace glm {
namespace k {

struct FmlaFwdArgs {
  int s_q = 0;            // query tokens (one CTA each)
  int s_kv = 0;           // rows of the latent cache (indices must be < s_kv)
  int h_q = 0;            // query heads: 64 or 128
  int topk = 0;           // index row width, a multiple of 64
  float sm_scale = 0.f;
  const void* q = nullptr;  // bf16 [s_q, h_q, 512] (strides in elements)
  int stride_q_s_q = 0, stride_q_h_q = 0;
  const void* kv = nullptr;  // bf16 cache rows of 512 values
  int stride_kv_s_kv = 0;
  const int* indices = nullptr;      // [s_q, topk] cache row ids, -1 = none
  int stride_indices_s_q = 0;
  const int* topk_length = nullptr;  // [s_q] valid prefix of each index row (nullptr: topk)
  void* out = nullptr;               // bf16 [s_q, h_q, 512]
  float* max_logits = nullptr;       // [s_q, h_q]
  float* lse = nullptr;              // [s_q, h_q]
  int num_sm = 148;
  cudaStream_t stream = nullptr;
};

void fmla_sparse_fwd(const FmlaFwdArgs& a);

// Query tiles (fmla_union.cu): a CTA per tile of 64 / H consecutive tokens of one sequence x H
// heads, over the union of their visible pools from sparse_mla_union's workspace (one gather4 per pool, a
// validity mask per token). Writes out rows [t0 H, (t0 + n) H) of each tile only.
struct FmlaUnionArgs {
  const int2* tiles = nullptr;  // {first token, tokens}
  int ntiles = 0;
  int H = 16;                   // local heads (8, 16, 32 or 64)
  float sm_scale = 0.f;
  const void* q = nullptr;      // bf16 [q_rows, 512] (token-major x heads)
  long long q_rows = 0;
  const void* kv = nullptr;     // bf16 cache rows of 512 values
  long long kv_rows = 0;
  const void* ws = nullptr;     // sparse_mla_union workspace
  int union_bytes = 0, union_cap = 0;
  const int* positions = nullptr;  // [T]
  void* out = nullptr;          // bf16 [T x H, 512]
  cudaStream_t stream = nullptr;
  // split-KV (H <= 16): nsplit CTAs per tile, each over a range of the union's 64-row blocks; out
  // unused, the partials into partial_ws (mla_attn_combine's layout, [token][split][16][514] fp32)
  int nsplit = 1;
  float* partial_ws = nullptr;
  // split-KV: the CTAs' prologue (query tile, barriers, tensor memory) before the PDL wait -- only
  // when the grid before it triggers its dependents after its own wait (the query rows complete)
  bool pre = false;
};
void fmla_sparse_union_fwd(const FmlaUnionArgs& a);

}  // namespace k
}  // namespace glm
