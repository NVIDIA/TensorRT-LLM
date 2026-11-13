// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "gmem_bounds_check.h"

struct FMHACutlassSM100Params {
  void* workspace_buffer_ptr;
  void* q_ptr;
  void* k_ptr;
  void* v_ptr;
  int* qo_segment_lens_ptr;
  int* kv_segment_lens_ptr;
  int* qo_segment_offsets_ptr;
  int* kv_segment_offsets_ptr;
  uint64_t* packed_work_range_ptr;
  uint64_t* packed_work_info_ptr;
  void* o_ptr;
  int mask_mode_code;
  float sm_scale;
  float scale_q;
  float scale_k;
  float scale_v;
  float o_scale;
  int num_qo_heads;
  int num_kv_heads;
  int head_dim_qk;
  int head_dim_vo;
  int q_stride_n;
  int q_stride_h;
  int k_stride_n;
  int k_stride_h;
  int v_stride_n;
  int v_stride_h;
  int batch_size;
  int total_qo_len;
  int total_kv_len;
  int max_qo_len;
  int* qo_offsets_ptr;
  cudaStream_t stream;
  int num_kv_splits;
  int* kv_tile_begin_ptr;
  int* kv_tile_end_ptr;
  int* kv_split_ptr;
  void* workspace_o_ptr;
  float* workspace_lse_ptr;
  int* num_kv_splits_per_row_ptr;
  int* kv_indices_ptr;
  int* kv_page_indptr_ptr;
  int total_page_num;
  float* max_score_ptr;
  int max_k_tiles;
  int* kv_block_indexes_ptr;
  int kv_block_num;
  int64_t kv_block_stride_n = 0;
  int64_t kv_block_stride_h = 0;
  int64_t kv_block_stride_b = 1;
  int pack_factor = 1;
  int h_r_original = 0;
  int q_stride_n_original = 0;
  int q_stride_h_original = 0;
  float* max_score_direct_ptr = nullptr;
  int total_qo_len_orig = 0;
  void* o_direct_ptr = nullptr;
  int num_qo_heads_orig = 0;
  int num_ctas = 0;
  // TMA fused direct-O unpack: enabled when pack_factor > 1, ptr_O_direct present,
  // qo_len uniform across batches, and FMHA_DISABLE_TMA_DIRECT_O env var not set.
  // When false, epilogue falls back to vec16 software scatter.
  bool tma_direct_o_enabled = false;
  GMEM_BOUNDS_FIELD
};

using FMHAVariantFn = cudaError_t (*)(const FMHACutlassSM100Params&);

cudaError_t fmha_reduction_bf16(const void* ptr_O_partial, void* ptr_O,
                                const float* ptr_lse,
                                const int* num_kv_splits_per_row,
                                float scale_softmax_log2, float inv_scale_o,
                                int num_kv_splits, int total_qo_len, int num_qo_heads,
                                int head_dim_vo, int stride_o_n, int stride_o_h,
                                int stride_partial_n, int stride_partial_h,
                                cudaStream_t stream);
