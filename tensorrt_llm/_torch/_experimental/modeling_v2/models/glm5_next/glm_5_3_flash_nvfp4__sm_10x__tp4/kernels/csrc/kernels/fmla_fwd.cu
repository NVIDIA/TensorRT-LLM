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
// FlashMLA SM100 sparse prefill (third_party/flashmla: deepseek-ai/FlashMLA at 15f13e5, MIT license),
// d_qk = d_v = 512 (no RoPE part), 64 query heads per CTA (one query token per CTA).
// This translation unit holds only the vendored kernel: its headers define a global bf16 type that
// differs from common.h's.
#include "fmla.h"

#include "sm100/prefill/sparse/fwd/head64/phase1.cuh"
#include "sm100/prefill/sparse/fwd/head128/phase1.cuh"

namespace glm {
namespace k {

static SparseAttnFwdParams fmla_params(const FmlaFwdArgs& a) {
  SparseAttnFwdParams p{};
  p.s_q = a.s_q;
  p.s_kv = a.s_kv;
  p.h_q = a.h_q;
  p.h_kv = 1;
  p.d_qk = 512;
  p.d_v = 512;
  p.topk = a.topk;
  p.sm_scale = a.sm_scale;
  p.sm_scale_div_log2 = a.sm_scale * 1.44269504088896340736f;
  p.q = (cutlass::bfloat16_t*)a.q;
  p.kv = (cutlass::bfloat16_t*)a.kv;
  p.indices = (int*)a.indices;
  p.attn_sink = nullptr;
  p.topk_length = (int*)a.topk_length;
  p.stride_q_s_q = a.stride_q_s_q;
  p.stride_q_h_q = a.stride_q_h_q;
  p.stride_kv_s_kv = a.stride_kv_s_kv;
  p.stride_kv_h_kv = a.stride_kv_s_kv;
  p.stride_indices_s_q = a.stride_indices_s_q;
  p.stride_indices_h_kv = a.topk;
  p.out = (cutlass::bfloat16_t*)a.out;
  p.max_logits = a.max_logits;
  p.lse = a.lse;
  p.num_sm = a.num_sm;
  p.stream = a.stream;
  return p;
}

void fmla_sparse_fwd(const FmlaFwdArgs& a) {
  if (a.s_q <= 0) return;
  const SparseAttnFwdParams p = fmla_params(a);
  if (a.h_q == 128)
    sm100::fwd::head128::run_fwd_phase1_kernel<512>(p);
  else
    sm100::fwd::head64::run_fwd_phase1_kernel<512>(p);
}

}  // namespace k
}  // namespace glm
