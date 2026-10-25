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
// Shared utilities of the GLM-5.3-Flash kernels.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                                          \
  do {                                                                                            \
    cudaError_t _e = (expr);                                                                      \
    if (_e != cudaSuccess) {                                                                      \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorString(_e), __FILE__, __LINE__, \
              #expr);                                                                             \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e));            \
    }                                                                                             \
  } while (0)

#define CHECK(cond, msg)                                                                 \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      fprintf(stderr, "Check failed %s at %s:%d: %s\n", #cond, __FILE__, __LINE__,       \
              std::string(msg).c_str());                                                 \
      throw std::runtime_error(std::string("Check failed: ") + #cond + ": " + (msg));   \
    }                                                                                    \
  } while (0)

namespace glm {

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;

__host__ __device__ inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
__host__ __device__ inline int64_t round_up(int64_t a, int64_t b) { return cdiv(a, b) * b; }

}  // namespace glm
