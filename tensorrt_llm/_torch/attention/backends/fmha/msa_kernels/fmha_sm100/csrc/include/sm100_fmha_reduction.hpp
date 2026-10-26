/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cutlass/cutlass.h"
#include "fmha_common.hpp"

namespace cutlass::fmha::kernel {

// Reduction kernel for split-KV FMHA.
// Merges partial bf16 O using float32 LSE (log-sum-exp) from multiple KV splits.
// ptr_lse stores LSE = max + log2(sum) / scale_softmax_log2.
//
// Grid:  (num_qo_heads, total_qo_len, 1)
// Block: 128 threads. Each thread handles one d element.
template <class ElementPartial,  // bf16 (workspace O type)
          class ElementOut,      // output dtype (fp8 or bf16)
          int kMaxSplits = 128>
struct Sm100FmhaReductionKernel {
  static const int MaxThreadsPerBlock = 128;

  struct Params {
    const ElementPartial* ptr_O_partial;  // [total_qo_len * num_splits, num_heads, head_dim]
    ElementOut* ptr_O;                    // [total_qo_len, num_heads, head_dim]
    const float* ptr_lse;                // [num_splits, total_qo_len, num_heads]
    const int* num_kv_splits_per_row;     // [total_qo_len]: actual split count per row
    float scale_softmax_log2;
    float inv_scale_o;
    int num_kv_splits;
    int total_qo_len;
    int num_qo_heads;
    int head_dim_vo;
    int stride_o_n;       // output: token stride
    int stride_o_h;       // output: head stride
    int stride_partial_n; // partial: token stride
    int stride_partial_h; // partial: head stride
    // Fused unpack-GQA: when ptr_O_direct != nullptr, write to unpacked layout
    // [nnz_qo, num_qo_heads_orig, head_dim_vo] instead of packed ptr_O.
    ElementOut* ptr_O_direct = nullptr;
    int num_qo_heads_orig = 0;
    int num_kv_heads = 0;
    int pack_factor = 1;
  };

  static dim3 get_grid_shape(Params const& params) {
    return dim3(params.num_qo_heads, params.total_qo_len, 1);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE void operator()(Params const& params, char* /* smem */) {
    int head_idx = blockIdx.x;
    int abs_row = blockIdx.y;
    int d = threadIdx.x;

    if (abs_row >= params.total_qo_len || d >= params.head_dim_vo) return;

    int num_splits = warp_uniform(params.num_kv_splits_per_row[abs_row]);
    int partial_off = abs_row * params.stride_partial_n +
                      head_idx * params.stride_partial_h + d;

    // Decide output base + offset. Fused unpack remaps packed (abs_row, head_idx)
    // to unpacked (actual_tok, unpacked_head); plain path uses packed strides.
    ElementOut* o_ptr_used;
    int64_t dst_off;
    if (params.ptr_O_direct != nullptr) {
      int pf            = params.pack_factor;
      int rem_hr        = params.num_qo_heads / params.num_kv_heads;  // packed h_r
      int kv_head       = head_idx / rem_hr;
      int rhr_idx       = head_idx % rem_hr;
      int h_r_orig      = rem_hr * pf;
      int pf_idx        = abs_row % pf;
      int actual_tok    = abs_row / pf;
      int unpacked_head = kv_head * h_r_orig + rhr_idx * pf + pf_idx;
      dst_off = (int64_t)actual_tok * params.num_qo_heads_orig * params.head_dim_vo
              + (int64_t)unpacked_head * params.head_dim_vo
              + d;
      o_ptr_used = params.ptr_O_direct;
    } else {
      dst_off = (int64_t)abs_row * params.stride_o_n
              + (int64_t)head_idx * params.stride_o_h
              + d;
      o_ptr_used = params.ptr_O;
    }

    // Fast path: single split — just scale and copy, no merge needed
    if (num_splits == 1) {
      int stats_base = warp_uniform(abs_row * params.num_qo_heads + head_idx);
      float lse = params.ptr_lse[stats_base];
      if (lse == -INFINITY) {
        o_ptr_used[dst_off] = ElementOut(0.f);
      } else {
        float o_s = static_cast<float>(params.ptr_O_partial[partial_off]);
        o_ptr_used[dst_off] = ElementOut(o_s * params.inv_scale_o);
      }
      return;
    }

    int stats_stride = warp_uniform(params.total_qo_len * params.num_qo_heads);
    int stats_base = warp_uniform(abs_row * params.num_qo_heads + head_idx);

    float running_lse = -__FLT_MAX__;
    float running_w = 0.f;
    float running_o = 0.f;

    int partial_stride = params.total_qo_len * params.stride_partial_n;
    int stats_off = stats_base;

    // Prologue: load first split
    float lse_cur = -INFINITY, o_s_cur = 0.f;
    if (num_splits > 0) {
      lse_cur = params.ptr_lse[stats_off];
      o_s_cur = static_cast<float>(params.ptr_O_partial[partial_off]);
    }

    // Main loop: prefetch next, merge current using LSE-based weights
    // Empty splits have LSE=-INF, exp2(scale*(-INF - ref)) = 0 → natural no-op
    #pragma unroll 2
    for (int s = 0; s < num_splits - 1; ++s) {
      float lse_s = lse_cur, o_s = o_s_cur;
      stats_off += stats_stride;
      partial_off += partial_stride;
      lse_cur = params.ptr_lse[stats_off];
      o_s_cur = static_cast<float>(params.ptr_O_partial[partial_off]);

      o_s = (lse_s != -INFINITY) ? o_s : 0.f;

      if (lse_s > running_lse) {
        float rescale = exp2f(params.scale_softmax_log2 * (running_lse - lse_s));
        running_o = fmaf(running_o, rescale, o_s);
        running_w = fmaf(running_w, rescale, 1.f);
        running_lse = lse_s;
      } else {
        float rescale = exp2f(params.scale_softmax_log2 * (lse_s - running_lse));
        running_o = fmaf(o_s, rescale, running_o);
        running_w += rescale;
      }
    }

    // Epilogue: last split
    if (num_splits > 0) {
      float o_last = (lse_cur != -INFINITY) ? o_s_cur : 0.f;
      if (lse_cur > running_lse) {
        float rescale = exp2f(params.scale_softmax_log2 * (running_lse - lse_cur));
        running_o = fmaf(running_o, rescale, o_last);
        running_w = fmaf(running_w, rescale, 1.f);
        running_lse = lse_cur;
      } else {
        float rescale = exp2f(params.scale_softmax_log2 * (lse_cur - running_lse));
        running_o = fmaf(o_last, rescale, running_o);
        running_w += rescale;
      }
    }

    if (running_w == 0.f) {
      o_ptr_used[dst_off] = ElementOut(0.f);
      return;
    }

    float inv_w = warp_uniform(1.f / running_w);
    o_ptr_used[dst_off] = ElementOut(running_o * inv_w * params.inv_scale_o);
  }
};

}  // namespace cutlass::fmha::kernel
