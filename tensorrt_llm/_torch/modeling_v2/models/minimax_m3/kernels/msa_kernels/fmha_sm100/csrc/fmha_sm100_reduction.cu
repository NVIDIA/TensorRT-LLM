// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-License-Identifier: MIT

#include "fmha_cutlass_sm100.cuh"

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

using tvm::ffi::TensorView;

namespace minfer {
namespace ops {

static cudaError_t fmha_reduction_bf16_impl(
    const void* ptr_O_partial, void* ptr_O,
    const float* ptr_lse,
    const int* num_kv_splits_per_row,
    float scale_softmax_log2, float inv_scale_o,
    int num_kv_splits, int total_qo_len, int num_qo_heads,
    int head_dim_vo, int stride_o_n, int stride_o_h,
    int stride_partial_n, int stride_partial_h,
    void* ptr_O_direct, int num_qo_heads_orig,
    int num_kv_heads, int pack_factor,
    cudaStream_t stream) {
  return flashinfer::launch_fmha_reduction<cutlass::bfloat16_t, cutlass::bfloat16_t>(
      static_cast<const cutlass::bfloat16_t*>(ptr_O_partial),
      static_cast<cutlass::bfloat16_t*>(ptr_O),
      ptr_lse, num_kv_splits_per_row,
      scale_softmax_log2, inv_scale_o,
      num_kv_splits, total_qo_len, num_qo_heads, head_dim_vo,
      stride_o_n, stride_o_h, stride_partial_n, stride_partial_h,
      static_cast<cutlass::bfloat16_t*>(ptr_O_direct),
      num_qo_heads_orig, num_kv_heads, pack_factor,
      stream);
}

void fmha_sm100_reduction_forward(
    TensorView o_partial, TensorView o,
    TensorView lse,
    TensorView num_kv_splits_per_row_tensor,
    double scale_softmax_log2, double inv_scale_o,
    int64_t num_kv_splits, int64_t total_qo_len, int64_t num_qo_heads,
    int64_t head_dim_vo,
    int64_t stride_o_n, int64_t stride_o_h,
    int64_t stride_partial_n, int64_t stride_partial_h,
    int64_t num_qo_heads_orig,
    int64_t num_kv_heads,
    int64_t pack_factor,
    int64_t stream_ptr) {
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  // When direct unpack is enabled, route o.data_ptr() to ptr_O_direct and leave
  // packed ptr_O nullptr. The kernel branches on ptr_O_direct.
  void* o_data = o.data_ptr();
  void* o_packed_ptr = (pack_factor > 1) ? nullptr : o_data;
  void* o_direct_ptr = (pack_factor > 1) ? o_data : nullptr;

  auto status = fmha_reduction_bf16_impl(
      o_partial.data_ptr(), o_packed_ptr,
      static_cast<const float*>(lse.data_ptr()),
      static_cast<const int*>(num_kv_splits_per_row_tensor.data_ptr()),
      static_cast<float>(scale_softmax_log2),
      static_cast<float>(inv_scale_o),
      static_cast<int>(num_kv_splits),
      static_cast<int>(total_qo_len),
      static_cast<int>(num_qo_heads),
      static_cast<int>(head_dim_vo),
      static_cast<int>(stride_o_n),
      static_cast<int>(stride_o_h),
      static_cast<int>(stride_partial_n),
      static_cast<int>(stride_partial_h),
      o_direct_ptr,
      static_cast<int>(num_qo_heads_orig),
      static_cast<int>(num_kv_heads),
      static_cast<int>(pack_factor),
      stream);
  if (status != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError)
        << "FMHA split-KV reduction failed: " << cudaGetErrorString(status);
  }
}

}  // namespace ops
}  // namespace minfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(reduction, minfer::ops::fmha_sm100_reduction_forward);
