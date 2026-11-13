// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include "plan.cuh"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

namespace minfer {
namespace ops {

void fmha_sm100_plan_forward(
    TensorView qo_segment_offsets,
    TensorView qo_segment_lens, TensorView kv_segment_lens,
    TensorView packed_work_range, TensorView packed_work_info,
    int64_t qo_tile_size, int64_t kv_tile_size,
    int64_t num_heads, int64_t num_buckets, bool causal,
    Optional<TensorView> maybe_qo_offsets,
    int64_t num_kv_splits,
    Optional<TensorView> maybe_kv_tile_begin_indices,
    Optional<TensorView> maybe_kv_tile_end_indices,
    Optional<TensorView> maybe_kv_split_indices,
    int64_t adaptive_chunk_size,
    Optional<TensorView> maybe_out_max_sm_cost,
    Optional<TensorView> maybe_num_kv_splits_per_row,
    int64_t stream_ptr,
    Optional<TensorView> maybe_workspace_lse,
    int64_t lse_total_size,
    int64_t pack_factor) {
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  int batch_size = qo_segment_lens.size(0);

  int* qo_offsets_ptr = maybe_qo_offsets.has_value()
                            ? static_cast<int*>(maybe_qo_offsets.value().data_ptr())
                            : nullptr;
  int* kv_tile_begin_ptr = maybe_kv_tile_begin_indices.has_value()
                               ? static_cast<int*>(maybe_kv_tile_begin_indices.value().data_ptr())
                               : nullptr;
  int* kv_tile_end_ptr = maybe_kv_tile_end_indices.has_value()
                             ? static_cast<int*>(maybe_kv_tile_end_indices.value().data_ptr())
                             : nullptr;
  int* kv_split_ptr = maybe_kv_split_indices.has_value()
                          ? static_cast<int*>(maybe_kv_split_indices.value().data_ptr())
                          : nullptr;
  float* out_max_sm_cost_ptr = maybe_out_max_sm_cost.has_value()
                                   ? static_cast<float*>(maybe_out_max_sm_cost.value().data_ptr())
                                   : nullptr;
  int* num_kv_splits_per_row_ptr = maybe_num_kv_splits_per_row.has_value()
                                       ? static_cast<int*>(maybe_num_kv_splits_per_row.value().data_ptr())
                                       : nullptr;
  float* workspace_lse_ptr = maybe_workspace_lse.has_value()
                                 ? static_cast<float*>(maybe_workspace_lse.value().data_ptr())
                                 : nullptr;

  auto status = flashinfer::plan_kernel_wrapper(
      static_cast<int*>(qo_segment_offsets.data_ptr()),
      static_cast<int*>(qo_segment_lens.data_ptr()),
      static_cast<int*>(kv_segment_lens.data_ptr()),
      static_cast<uint64_t*>(packed_work_range.data_ptr()),
      static_cast<uint64_t*>(packed_work_info.data_ptr()),
      qo_tile_size, kv_tile_size, batch_size,
      num_heads, num_buckets, causal, qo_offsets_ptr,
      /*enable_pdl=*/true, stream,
      num_kv_splits, kv_tile_begin_ptr, kv_tile_end_ptr, kv_split_ptr,
      adaptive_chunk_size, out_max_sm_cost_ptr,
      num_kv_splits_per_row_ptr,
      workspace_lse_ptr, static_cast<int>(lse_total_size),
      static_cast<int>(pack_factor));
  if (status != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError) << "Failed to plan fmha_sm100: " << cudaGetErrorString(status);
  }
}

}  // namespace ops
}  // namespace minfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, minfer::ops::fmha_sm100_plan_forward);
