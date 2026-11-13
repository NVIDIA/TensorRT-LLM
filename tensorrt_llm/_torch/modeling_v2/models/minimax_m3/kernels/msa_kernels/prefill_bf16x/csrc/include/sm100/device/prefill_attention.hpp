// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/bfloat16.h"

// Data-path selector (see prefill_attention_config.cuh): 0 = E4M3, 1 = BF16. Passed by the JIT
// for the BF16 extension only.
#if !defined(FMHA_SM100_PREFILL_BF16X_DTYPE)
#define FMHA_SM100_PREFILL_BF16X_DTYPE 0
#endif

namespace fmha_sm100::prefill_bf16x {

// One side (K or V) of the NVFP4 cache, data or block scales: a [pages, heads, 128, row_bytes]
// view whose token rows are contiguous within a head; the head and page strides are free
// (bytes, multiples of 16).
struct CacheView {
  uint8_t const *ptr = nullptr;
  int64_t head_stride = 0;
  int64_t page_stride = 0;
};

struct PrefillArguments {
  uint8_t const *q_ptr = nullptr;
  CacheView packed_k;
  CacheView packed_v;
  CacheView k_scale;
  CacheView v_scale;
  // Physical pages: a flat list where request b owns entries [kv_indptr[b], kv_indptr[b + 1]),
  // or, without kv_indptr, a [batch, page_table_width] table with row stride page_table_stride.
  int32_t const *kv_indices_ptr = nullptr;
  int32_t const *kv_indptr_ptr = nullptr;
  int page_table_stride = 0;
  int page_table_width = 0;
  int32_t const *cu_seqlens_q_ptr = nullptr;
  int32_t const *cu_seqlens_k_ptr = nullptr;
  // Optional [batch] KV lengths used for masking and causal alignment instead of cu_seqlens_k.
  int32_t const *seqused_k_ptr = nullptr;
  int32_t const *k2q_row_ptr = nullptr;
  int32_t const *qsplit_indices_ptr = nullptr;
  int32_t const *scheduler_metadata_ptr = nullptr;
  int32_t const *work_count_ptr = nullptr;
  cutlass::bfloat16_t *o_partial_ptr = nullptr;
  // Elements between consecutive split slabs of o_partial: total_q * num_q_heads * 128 for a
  // contiguous buffer, more when the slabs are padded (locality-domain striping).
  int64_t o_partial_split_stride = 0;
  float *lse_partial_ptr = nullptr;
  // Optional one-element fp32 per-tensor scales (value = code x block_scale x global_scale),
  // read on the device so captured graphs follow updates. BF16 data path: the K global scale is
  // folded into the K dequant (bf16 x bf16 block scale) and v_global_scale must be null (the
  // combine applies it).
  float const *k_global_scale_ptr = nullptr;
  float const *v_global_scale_ptr = nullptr;
  // Optional int32 counters the combine claims its tiles from; this kernel zeroes them for the
  // combine that follows it in the stream (the previous combine has completed when it starts).
  int32_t *combine_tile_counters_ptr = nullptr;
  int combine_tile_counter_count = 0;

  int total_q = 0;
  int num_q_heads = 0;
  int num_kv_heads = 0;
  int physical_pages = 0;
  int total_rows = 0;
  int qsplit_stride = 0;
  int work_capacity = 0;
  // Host factors; the kernel multiplies in the K and V global scales respectively (E4M3). The
  // BF16 data path ignores output_scale (its forward applies none).
  float softmax_scale_log2 = 0.0f;
  float output_scale = 1.0f;
};

cudaError_t launch_prefill_attention(PrefillArguments const &arguments, cudaStream_t stream);

} // namespace fmha_sm100::prefill_bf16x
