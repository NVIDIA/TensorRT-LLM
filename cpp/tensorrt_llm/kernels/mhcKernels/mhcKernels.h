/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mhc
{

void mhcBigFuseLaunch(float const* y_acc, float const* r_acc, __nv_bfloat16 const* residual, float const* hc_scale,
    float const* hc_base, float* post_mix, float* comb_mix, __nv_bfloat16* layer_input, int M, int K, int hidden_size,
    float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat,
    int num_splits, int block_size, cudaStream_t stream);

void mhcGemmSqrsumFmaLaunch(__nv_bfloat16 const* x, float const* w_t, float* y, float* r, int M, int N, int K,
    int tile_n, int tile_m, cudaStream_t stream);

void mhcHcHeadApplyLaunch(float const* mixes, float const* sqrsum, __nv_bfloat16 const* x, __nv_bfloat16* out,
    float const* scale, float const* base, int M, int mult, int hidden_size, int K, float norm_eps, float eps,
    cudaStream_t stream);

void mhcPostMappingLaunch(__nv_bfloat16 const* residual, __nv_bfloat16 const* x, float const* post_mix,
    float const* comb_mix, __nv_bfloat16* out, int B, int hidden_size, cudaStream_t stream);

// Single-launch fused hyper-connection boundary op (SM100 only).
//
// Produces (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur) in two
// kernel launches: (1) tcgen05 TF32 GEMM fused with post-mapping and sqr-sum
// that emits D, sqr_sum, and residual_cur, and (2) big-fuse postlogue that
// consumes those to emit post_mix_cur, comb_mix_cur, layer_input_cur.
//
// Workspace tensors are caller-allocated and zeroed by this launcher:
//   y_acc_workspace: fp32 [M, 24]
//   r_acc_workspace: fp32 [M]
//
// Shape constraints (B200/B300 / sm_100a):
//   hc_mult == 4
//   hidden_size in {4096, 7168} for SM100/tcgen05 MMA fused-HC paths.
// FMA fused-HC paths use runtime hidden_size but still require hidden_size % 64 == 0.
// Passing num_k_splits == 0 or bigfuse_block_size == 0 falls back to the
// internal heuristics.
void mhcFusedHcLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev, float const* post_mix_prev,
    float const* comb_mix_prev, float const* w_t, float const* hc_scale, float const* hc_base,
    __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur, __nv_bfloat16* layer_input_cur,
    float* y_acc_workspace, float* r_acc_workspace, int M, int hidden_size, int hc_mult, int num_k_splits,
    int bigfuse_block_size, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value,
    int sinkhorn_repeat, cudaStream_t stream);

// FMA-path fused hyper-connection boundary launcher.
//
// Composes fused_pmap_gemm_fma_ksplit (writes Yp[ks, M, 24], Rp[ks, M], and
// residual_cur[M, HC_MULT, HIDDEN]) with mhcBigFuseKernel<ks, bs> which
// reduces across the split axis internally.  Used as the small-M backend in
// the mhcFusedHc autotuner.
//
// Supported tactics (hit by the autotuner); see pickFhcFma for the exact
// (tile_n, num_k_splits) table:
//   tile_n ∈ {1, 2, 3, 4, 6, 8, 12, 24}
//   num_k_splits ∈ {1, 2, 4, 8} (ks=1 ⇔ seq FMA, no HIDDEN split). Larger ks
//     requires smaller tile_n: {1,2}→{1,2,4,8}, {3}→{1,2,4}, {4}→{1,2},
//     {6,8,12,24}→{1}.
//   bigfuse_block_size ∈ {128, 256, 512}
//
// Workspace shapes (caller-allocated):
//   y_acc_workspace: fp32 [num_k_splits, M, 24]
//   r_acc_workspace: fp32 [num_k_splits, M]
// No pre-zeroing required (ksplit kernel writes = not +=).
void mhcFusedHcFmaLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev, float const* post_mix_prev,
    float const* comb_mix_prev, float const* w_t, float const* hc_scale, float const* hc_base,
    __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur, __nv_bfloat16* layer_input_cur,
    float* y_acc_workspace, float* r_acc_workspace, int M, int hidden_size, int hc_mult, int tile_n, int num_k_splits,
    int bigfuse_block_size, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, float hc_post_mult_value,
    int sinkhorn_repeat, cudaStream_t stream);

// Single-kernel all-in-one fused hyper-connection boundary launcher (TF32 tcgen05 path).
//
// Wraps fused_allinone_tf32_pmap_gemm_atomic_impl: pmap + TF32 GEMM + bigfuse
// are all fused into ONE kernel launch.  Phase 3 uses an atomic done-counter
// to elect the last-home CTA per m-block, which then executes the bigfuse
// epilogue inline (no second kernel, no launch-latency stall).  Preferred for
// mid/large M where the extra kernel launch overhead of the 2-kernel path is
// visible (see design_doc_fused_hc.md).
//
// Workspace tensors are caller-allocated and zeroed by this launcher:
//   y_acc_workspace: fp32 [M, 24]
//   r_acc_workspace: fp32 [M]
//   done_counter_workspace: int32 [ceil(M / 64)]
//
// Shape constraints (B200/B300 / sm_100a):
//   hc_mult == 4
//   hidden_size in {4096, 7168} for SM100/tcgen05 MMA fused-HC paths.
// FMA fused-HC paths use runtime hidden_size but still require hidden_size % 64 == 0.
// Passing num_k_splits == 0 falls back to internal heuristics.
void mhcFusedHcAllInOneLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev,
    float const* post_mix_prev, float const* comb_mix_prev, float const* w_t, float const* hc_scale,
    float const* hc_base, __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur,
    __nv_bfloat16* layer_input_cur, float* y_acc_workspace, float* r_acc_workspace, int* done_counter_workspace, int M,
    int hidden_size, int hc_mult, int num_k_splits, float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps,
    float hc_post_mult_value, int sinkhorn_repeat, cudaStream_t stream);

// Single-kernel all-in-one fused hyper-connection boundary launcher (FMA path).
//
// Wraps fused_pmap_gemm_fma_allinone<TN, KS, TM>: pmap + fp32 FMA GEMM +
// bigfuse are all fused into ONE kernel launch.  Uses the same atomic
// last-home election pattern as the TF32 all-in-one path.  Preferred for
// small-M (M <= 32) where the TF32 tcgen05 path cannot saturate the MMA pipe.
//
// Supported tactics; see pickFhcFmaAllInOne for the exact table:
//   tile_n ∈ {1, 2, 3, 4, 6, 8, 12, 24}
//   num_k_splits ∈ {1, 2}. ks=2 is only valid with tile_n ∈ {1, 2}; all
//     other tile_n require ks=1.
//   tile_m ∈ {1, 2, 4} (tokens per CTA)
//
// Workspace shapes (caller-allocated, zeroed by this launcher):
//   y_acc_workspace: fp32 [M, 24]
//   r_acc_workspace: fp32 [M]
//   done_counter_workspace: int32 [ceil(M / tile_m)]
void mhcFusedHcFmaAllInOneLaunch(__nv_bfloat16 const* x_prev, __nv_bfloat16 const* residual_prev,
    float const* post_mix_prev, float const* comb_mix_prev, float const* w_t, float const* hc_scale,
    float const* hc_base, __nv_bfloat16* residual_cur, float* post_mix_cur, float* comb_mix_cur,
    __nv_bfloat16* layer_input_cur, float* y_acc_workspace, float* r_acc_workspace, int* done_counter_workspace, int M,
    int hidden_size, int hc_mult, int tile_n, int num_k_splits, int tile_m, float rms_eps, float hc_pre_eps,
    float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat, cudaStream_t stream);

} // namespace kernels::mhc

TRTLLM_NAMESPACE_END
