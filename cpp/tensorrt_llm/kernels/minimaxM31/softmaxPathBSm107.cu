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

#include "tensorrt_llm/kernels/minimaxM31/softmaxPathBSm107.h"
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/std/limits>
#include <type_traits>
TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{
namespace fi
{
// Helpers and ordered softmax arithmetic adapted from FlashInfer 0.6.18, Copyright (c) 2023-2026 FlashInfer team,
// Apache-2.0. Float4 helpers are the same subset used by softmaxMcSm103.cu. Keep the source arithmetic form so the
// denominator recurrence and final CUB sum preserve the reference's rounding. Only the chunk maximum is replaced
// with unsigned hardware warp reductions and separate ping-pong shared storage.
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
template <uint32_t BLOCK_THREADS>
struct OnlineSoftmaxTempStorage {
  union {
    typename cub::BlockReduce<float, BLOCK_THREADS>::TempStorage reduce;
    typename cub::BlockReduce<float2, BLOCK_THREADS>::TempStorage reduce_pair;
  } block_prim;

  struct {
    float max_val;
    float denominator;
  } shared_state;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1070)
#define M31_PATHB_LAUNCH_BOUNDS(block_threads) __launch_bounds__(block_threads)
#else
#define M31_PATHB_LAUNCH_BOUNDS(block_threads)
#endif
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, typename DType, bool CACHE_INPUT>
__global__ M31_PATHB_LAUNCH_BOUNDS(BLOCK_THREADS) void PathBWarpMaxKernel(
    DType* logits, DType* output, DType* temperature_arr, DType temperature_val) {
  constexpr uint32_t d = 200064;
  static_assert(BLOCK_THREADS == 1024 && VEC_SIZE == 4 && !CACHE_INPUT);
  // Separate from final CUB storage: it may be used while slower warps finish the last max load.
  // Buffer i is not overwritten until all warps have crossed the next iteration's barrier.
  __shared__ uint32_t warp_max_words[2][32];
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  float temperature = temperature_arr == nullptr ? temperature_val : temperature_arr[bx];
  const float inv_temp = (temperature == 0.f) ? 0.f : 1.f / temperature;

  using TempStorage = OnlineSoftmaxTempStorage<BLOCK_THREADS>;
  extern __shared__ __align__(alignof(TempStorage)) uint8_t smem[];
  auto& temp_storage = reinterpret_cast<TempStorage&>(smem);

  DType* smem_vec_base = nullptr;
  if constexpr (CACHE_INPUT) {
    constexpr size_t vec_alignment = alignof(vec_t<DType, VEC_SIZE>);
    size_t aligned_offset = round_up(sizeof(TempStorage), vec_alignment);
    smem_vec_base = reinterpret_cast<DType*>(smem + aligned_offset);
  }

  vec_t<DType, VEC_SIZE> logits_vec;

  float running_max = -cuda::std::numeric_limits<float>::infinity();
  float running_denominator = 0.0f;
  float threadlocal_running_denominator = 0.0f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // Pass 1: Compute running max and denominator
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      logits_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);

#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        logits_vec[j] *= inv_temp;
      }

      if constexpr (CACHE_INPUT) {
        logits_vec.store(smem_vec_base + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }
    }

    float thread_max = -cuda::std::numeric_limits<float>::infinity();
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = max(thread_max, logits_vec[j]);
    }
    // Monotonic float keys preserve finite maxima and the -inf padding lanes.
    const uint32_t lane = tx & 31u;
    const uint32_t warp = tx >> 5;
    const uint32_t bits = __float_as_uint(thread_max);
    uint32_t key = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    key = __reduce_max_sync(0xffffffffu, key);
    if (lane == 0) warp_max_words[i & 1u][warp] = key;
    __syncthreads();
    key = __reduce_max_sync(0xffffffffu, warp_max_words[i & 1u][lane]);
    float block_max = __uint_as_float(key ^ ((key & 0x80000000u) ? 0x80000000u : 0xffffffffu));
    // if block_max is -inf, then this block contains all -inf values, so we can skip updating
    if (!isinf(block_max)) {
      float threadlocal_sum = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        threadlocal_sum += __expf(logits_vec[j] - block_max);
      }
      float new_max = max(running_max, block_max);
      threadlocal_running_denominator =
          threadlocal_running_denominator * __expf(running_max - new_max) +
          threadlocal_sum * __expf(block_max - new_max);
      running_max = new_max;
    }
  }

  running_denominator = cub::BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                            .Sum(threadlocal_running_denominator);
  if (tx == 0) {
    temp_storage.shared_state.denominator = running_denominator;
  }
  __syncthreads();
  running_denominator = temp_storage.shared_state.denominator;

  const float final_max = running_max;
  const float inv_denominator = 1.0f / running_denominator;

  // Pass 2: Normalize in place
  vec_t<DType, VEC_SIZE> prob_vec;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    if constexpr (CACHE_INPUT) {
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        logits_vec.load(smem_vec_base + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }
    } else {
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        logits_vec.cast_load(logits + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);

#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          logits_vec[j] *= inv_temp;
        }
      }
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      float p = __expf(static_cast<float>(logits_vec[j]) - final_max) * inv_denominator;
      prob_vec[j] = static_cast<DType>(p);
    }

    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      prob_vec.cast_store(output + bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

#undef M31_PATHB_LAUNCH_BOUNDS
// clang-format on
} // namespace fi
} // namespace

cudaError_t invokeSoftmaxPathB(
    float* logits, float* output, float* temperature, float temperatureValue, int rows, cudaStream_t stream)
{
    if (rows < kSoftmaxPathBMinRows || rows > kSoftmaxPathBMaxRows)
    {
        return cudaErrorInvalidValue;
    }
    constexpr int kThreads = 1024;
    constexpr int kVectorSize = 4;
    auto kernel = fi::PathBWarpMaxKernel<kThreads, kVectorSize, float, false>;
    constexpr size_t kSharedBytes = sizeof(fi::OnlineSoftmaxTempStorage<kThreads>);
    static_assert(kSharedBytes + 256 < 48 * 1024);
    void* arguments[] = {&logits, &output, &temperature, &temperatureValue};
    return cudaLaunchKernel(reinterpret_cast<void const*>(kernel), dim3(static_cast<unsigned>(rows)), dim3(kThreads),
        arguments, kSharedBytes, stream);
}
} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
