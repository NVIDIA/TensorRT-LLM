/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector_functions.h>
#include <type_traits>

namespace tensorrt_llm::kernels::nccl_device {

// Helper struct to distinguish half vector operations
struct HalfVector {
  uint4 data;
  using accType = float; // Accumulation type for precision
  
  __device__ __forceinline__ HalfVector(uint4 val) : data(val) {}
  __device__ __forceinline__ HalfVector() : data({0,0,0,0}) {}
  __device__ __forceinline__ HalfVector(int) : data({0,0,0,0}) {} // For {0} initialization
};

// Helper struct to distinguish bfloat16 vector operations
struct BFloat16Vector {
  uint4 data;
  using accType = float; // Accumulation type for precision
  
  __device__ __forceinline__ BFloat16Vector(uint4 val) : data(val) {}
  __device__ __forceinline__ BFloat16Vector() : data({0,0,0,0}) {}
  __device__ __forceinline__ BFloat16Vector(int) : data({0,0,0,0}) {} // For {0} initialization
};

// Helper struct to distinguish FP8 e5m2x4 vector operations
// e5m2x4 means 4 e5m2 elements packed into uint4 (32 bits total)
struct FP8E5M2x4Vector {
  uint4 data;
  using accType = float; // Accumulation type for precision
  
  __device__ __forceinline__ FP8E5M2x4Vector(uint4 val) : data(val) {}
  __device__ __forceinline__ FP8E5M2x4Vector() : data({0,0,0,0}) {}
  __device__ __forceinline__ FP8E5M2x4Vector(int) : data({0,0,0,0}) {} // For {0} initialization
};

// Helper struct to distinguish FP8 e4m3x4 vector operations
// e4m3x4 means 4 e4m3 elements packed into uint4 (32 bits total)
struct FP8E4M3x4Vector {
  uint4 data;
  using accType = float; // Accumulation type for precision
  
  __device__ __forceinline__ FP8E4M3x4Vector(uint4 val) : data(val) {}
  __device__ __forceinline__ FP8E4M3x4Vector() : data({0,0,0,0}) {}
  __device__ __forceinline__ FP8E4M3x4Vector(int) : data({0,0,0,0}) {} // For {0} initialization
};

// Vector type mapping
template<typename T>
struct VectorType {
  using type = T; // Default to scalar (elementsPerVector = 1)
  using accType = T;
};

// Specializations for vectorized types
template<>
struct VectorType<float> {
  using type = float4; // Use float4 for best vectorization (elementsPerVector = 4)
  using accType = float;
};

template<>
struct VectorType<double> {
  using type = double; // Use double for vectorization since that is the only multimem supported version
  using accType = double;
};

template <>
struct VectorType<half> {
  using type=HalfVector;  // Use HalfVector for proper half arithmetic
  using accType=float;  // Always use float for accumulation for numerical stability
};

template <>
struct VectorType<__nv_bfloat16> {
  using type=BFloat16Vector;  // Use BFloat16Vector for proper bfloat16 arithmetic
  using accType=float;  // Always use float for accumulation for numerical stability
};

template <>
struct VectorType<__nv_fp8_e5m2> {
  using type=FP8E5M2x4Vector;  // Use FP8E5M2x4Vector for FP8 e5m2x4 arithmetic
  using accType=float;
};

template <>
struct VectorType<__nv_fp8_e4m3> {
  using type=FP8E4M3x4Vector;  // Use FP8E4M3x4Vector for FP8 e4m3x4 arithmetic
  using accType=float;
};

} // namespace tensorrt_llm::kernels::nccl_device 
