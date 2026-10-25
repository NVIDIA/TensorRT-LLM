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

// ST-01-SM: FlashInfer softmax(logits, temperature) Path A with the Reduce kernel's
// write pass spread over many CTAs per row. See softmaxMcSm103.h for the contract. Compiled with --use_fast_math (set
// on this source in kernels/CMakeLists.txt), as FlashInfer's JIT compiles it.

#include "tensorrt_llm/kernels/minimaxM31/softmaxMcSm103.h"

#include <cuda/std/limits>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cub/cub.cuh>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

// ---------------------------------------------------------------------------------------------------------------------
// Vendored from FlashInfer (v0.6.15; identical in v0.6.18): include/flashinfer/utils.cuh (ceil_div, round_up),
// include/flashinfer/sampling.cuh (MaxReduceOp, Float2SoftmaxReduceOp, PartialSoftmaxResult) and
// include/flashinfer/vec_dtypes.cuh (vec_t<float, vec_size>: only the members the kernels below use; cast_load /
// cast_store are the same-type branch of FlashInfer's cast_load_impl / cast_store_impl, i.e. load / store).
// Copyright (c) 2023-2026 by FlashInfer team, licensed under the Apache License, Version 2.0. Copied verbatim
// (formatting kept) so that the element order, the float4 accesses and the reduction operators are FlashInfer's own;
// FlashInfer's headers are not on the TensorRT-LLM C++ include path.
// ---------------------------------------------------------------------------------------------------------------------
namespace fi
{
// clang-format off
#define FLASHINFER_INLINE inline __attribute__((always_inline)) __device__

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 ceil_div(const T1 x, const T2 y) noexcept {
  return (x + y - 1) / y;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ constexpr T1 round_up(const T1 x, const T2 y) noexcept {
  return ceil_div(x, y) * y;
}

using MaxReduceOp = cuda::maximum<>;

struct Float2SoftmaxReduceOp {
  __device__ __forceinline__ float2 operator()(const float2& a, const float2& b) const {
    if (isinf(a.x)) return b;
    if (isinf(b.x)) return a;

    float new_max = max(a.x, b.x);
    float new_denom = a.y * __expf(a.x - new_max) + b.y * __expf(b.x - new_max);
    return make_float2(new_max, new_denom);
  }
};

struct PartialSoftmaxResult {
  float max_val;
  float denominator;
};

template <typename float_t, size_t vec_size>
struct vec_t;

// float x 4 or more
template <size_t vec_size>
struct vec_t<float, vec_size> {
  static_assert(vec_size % 4 == 0, "Invalid vector size");
  float4 data[vec_size / 4];

  FLASHINFER_INLINE float& operator[](size_t i) { return ((float*)(data))[i]; }
  FLASHINFER_INLINE const float& operator[](size_t i) const { return ((const float*)(data))[i]; }
  FLASHINFER_INLINE void fill(float val) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = make_float4(val, val, val, val);
    }
  }
  FLASHINFER_INLINE void load(const float* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      data[i] = ((float4*)ptr)[i];
    }
  }
  FLASHINFER_INLINE void store(float* ptr) const {
#pragma unroll
    for (size_t i = 0; i < vec_size / 4; ++i) {
      ((float4*)ptr)[i] = data[i];
    }
  }
  template <typename T>
  FLASHINFER_INLINE void cast_load(const T* ptr) {
    static_assert(std::is_same_v<T, float>, "only the same-type branch of cast_load_impl is vendored");
    load(ptr);
  }
  template <typename T>
  FLASHINFER_INLINE void cast_store(T* ptr) const {
    static_assert(std::is_same_v<T, float>, "only the same-type branch of cast_store_impl is vendored");
    store(ptr);
  }
};

#undef FLASHINFER_INLINE
// clang-format on
} // namespace fi

// FlashInfer's OnlineSoftmax on sm_80+ (DISPATCH_COMPUTE_CAP_NUM_THREADS) and vec_size = gcd(16 / sizeof(float), d)
// = 4 for d % 4 == 0.
constexpr uint32_t kThreads = 1024;
constexpr uint32_t kVec = 4;

// FlashInfer OnlineSoftmaxMapKernel<1024, 4, float>: the same body (the shared temp storage is static instead of
// FlashInfer's dynamic OnlineSoftmaxTempStorage union; no PDL griddepcontrol, as the launches here are plain).
__global__ void __launch_bounds__(kThreads) softmaxMcMapKernel(float const* logits, fi::PartialSoftmaxResult* partials,
    float const* temperature_arr, float temperature_val, uint32_t d, uint32_t num_slices)
{
    // clang-format off
  const uint32_t bx = blockIdx.x;
  const uint32_t by = blockIdx.y;  // slice index
  const uint32_t tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;
  const uint32_t vec_alignment_elems = alignof(fi::vec_t<float, kVec>) / sizeof(float);
  const uint32_t slice_stride = fi::round_up(fi::ceil_div(d, num_slices), vec_alignment_elems);
  const uint32_t slice_start = by * slice_stride;
  const uint32_t slice_size = min((by + 1) * slice_stride, d) - slice_start;
  if (slice_start >= d) return;
  __shared__ typename cub::BlockReduce<float, kThreads>::TempStorage reduce;
  __shared__ float s_max, s_den;
  fi::vec_t<float, kVec> logits_vec;
  float running_max = -cuda::std::numeric_limits<float>::infinity();
  float running_denominator = 0.0f;
  float threadlocal_running_denominator = 0.0f;
#pragma unroll 2
  for (uint32_t i = 0; i < fi::ceil_div(slice_size, kThreads * kVec); ++i) {
    logits_vec.fill(-cuda::std::numeric_limits<float>::infinity());
    if ((i * kThreads + tx) * kVec < slice_size) {
      logits_vec.cast_load(logits + bx * d + slice_start + (i * kThreads + tx) * kVec);
    }
    float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
    for (uint32_t j = 0; j < kVec; ++j) {
      logits_vec[j] *= inv_temp;
      thread_max = max(thread_max, logits_vec[j]);
    }
    float block_max = cub::BlockReduce<float, kThreads>(reduce).Reduce(thread_max, fi::MaxReduceOp{});
    if (tx == 0) s_max = block_max;
    __syncthreads();
    block_max = s_max;
    // if block_max is -inf, then this block contains all -inf values, so we can skip updating
    if (!isinf(block_max)) {
      float threadlocal_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < kVec; ++j) {
        threadlocal_sum += __expf(logits_vec[j] - block_max);
      }
      float new_max = max(running_max, block_max);
      threadlocal_running_denominator = threadlocal_running_denominator * __expf(running_max - new_max) +
                                        threadlocal_sum * __expf(block_max - new_max);
      running_max = new_max;
    }
  }
  running_denominator = cub::BlockReduce<float, kThreads>(reduce).Sum(threadlocal_running_denominator);
  if (tx == 0) s_den = running_denominator;
  __syncthreads();
  running_denominator = s_den;
  if (tx == 0) {
    partials[bx * num_slices + by] = {running_max, running_denominator};
  }
    // clang-format on
}

// FlashInfer OnlineSoftmaxReduceKernel<1024, 4, float> with its write loop split over gridDim.y CTAs per row: the
// prologue (the partials' reduction) is FlashInfer's, unchanged, in every CTA of the row.
__global__ void __launch_bounds__(kThreads)
    softmaxMcReduceWriteKernel(float const* logits, float* output, fi::PartialSoftmaxResult const* partials,
        float const* temperature_arr, float temperature_val, uint32_t d, uint32_t num_slices)
{
    // clang-format off
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;
  __shared__ typename cub::BlockReduce<float2, kThreads>::TempStorage reduce_pair;
  __shared__ float s_max, s_den;
  const fi::Float2SoftmaxReduceOp reduce_op;
  float2 thread_aggregate = make_float2(-cuda::std::numeric_limits<float>::infinity(), 0.0f);
  for (uint32_t i = tx; i < num_slices; i += kThreads) {
    fi::PartialSoftmaxResult partial = partials[bx * num_slices + i];
    float2 partial_pair = make_float2(partial.max_val, partial.denominator);
    thread_aggregate = reduce_op(thread_aggregate, partial_pair);
  }
  float2 block_result = cub::BlockReduce<float2, kThreads>(reduce_pair).Reduce(thread_aggregate, reduce_op);
  if (tx == 0) {
    s_max = block_result.x;
    s_den = block_result.y;
  }
  __syncthreads();
  const float final_max = s_max;
  const float inv_denominator = 1.0f / s_den;
  // FlashInfer's write loop iterates i over ceil_div(d, 1024 * 4) tiles of 4096 elements; this CTA takes tiles
  // i = blockIdx.y, blockIdx.y + gridDim.y, ... (elementwise, so the split does not change any value).
  fi::vec_t<float, kVec> logits_vec;
  fi::vec_t<float, kVec> prob_vec;
  for (uint32_t i = blockIdx.y; i < fi::ceil_div(d, kThreads * kVec); i += gridDim.y) {
    logits_vec.fill(-cuda::std::numeric_limits<float>::infinity());
    if ((i * kThreads + tx) * kVec < d) {
      logits_vec.cast_load(logits + bx * d + (i * kThreads + tx) * kVec);
    }
#pragma unroll
    for (uint32_t j = 0; j < kVec; ++j) {
      logits_vec[j] *= inv_temp;
      float p = __expf(static_cast<float>(logits_vec[j]) - final_max) * inv_denominator;
      prob_vec[j] = static_cast<float>(p);
    }
    if ((i * kThreads + tx) * kVec < d) {
      prob_vec.cast_store(output + bx * d + (i * kThreads + tx) * kVec);
    }
  }
    // clang-format on
}

uint32_t numSlices(int d)
{
    return fi::ceil_div(static_cast<uint32_t>(d), static_cast<uint32_t>(kSoftmaxMcSliceSize));
}

} // namespace

cudaError_t invokeSoftmaxMc(float const* logits, float* out, float const* temperatureArr, float temperatureVal,
    int rows, int d, int writeCtas, void* partials, cudaStream_t stream)
{
    if (rows <= 0 || d <= 0)
    {
        return cudaSuccess;
    }
    if (d % static_cast<int>(kVec) != 0 || writeCtas < 1 || writeCtas > 65535 || rows > 65535)
    {
        return cudaErrorInvalidValue;
    }
    uint32_t const slices = numSlices(d);
    auto* part = static_cast<fi::PartialSoftmaxResult*>(partials);
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>(rows), slices);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaError_t const err = cudaLaunchKernelEx(
        &cfg, softmaxMcMapKernel, logits, part, temperatureArr, temperatureVal, static_cast<uint32_t>(d), slices);
    if (err != cudaSuccess)
    {
        return err;
    }
    cfg.gridDim = dim3(static_cast<unsigned>(rows), static_cast<unsigned>(writeCtas));
    return cudaLaunchKernelEx(&cfg, softmaxMcReduceWriteKernel, logits, out,
        static_cast<fi::PartialSoftmaxResult const*>(part), temperatureArr, temperatureVal, static_cast<uint32_t>(d),
        slices);
}

size_t softmaxMcPartialsBytes(int rows, int d)
{
    if (rows <= 0 || d <= 0)
    {
        return 0;
    }
    return static_cast<size_t>(rows) * numSlices(d) * sizeof(fi::PartialSoftmaxResult);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
