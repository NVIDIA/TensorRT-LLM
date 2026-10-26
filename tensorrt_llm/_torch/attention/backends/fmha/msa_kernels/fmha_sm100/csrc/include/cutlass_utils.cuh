/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_CUTLASS_UTILS_CUH_
#define FLASHINFER_CUTLASS_UTILS_CUH_

#include <cuda_fp8.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#if defined(FLASHINFER_ENABLE_FP4_E2M1)
#if (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 12080)
#include <cuda_fp4.h>
#endif
#endif

namespace flashinfer {

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <>
struct cutlass_dtype<__nv_fp8_e4m3> {
  using type = cutlass::float_e4m3_t;
};

template <>
struct cutlass_dtype<__nv_fp8_e5m2> {
  using type = cutlass::float_e5m2_t;
};

#if CUDA_VERSION >= 12080
template <>
struct cutlass_dtype<__nv_fp8_e8m0> {
  using type = cutlass::float_ue8m0_t;
};

#if defined(FLASHINFER_ENABLE_FP4_E2M1)
template <>
struct cutlass_dtype<__nv_fp4_e2m1> {
  using type = cutlass::float_e2m1_t;
};
#endif
#endif

template <typename T>
using cutlass_dtype_t = typename cutlass_dtype<T>::type;

template <typename T>
void compileTimeDebug(T&&) {
  static_assert(sizeof(T) == 0, "Compile time debug");
}

#define CUTLASS_CHECK(cmd)                                                            \
  do {                                                                                \
    auto status = cmd;                                                                \
    if (status != cutlass::Status::kSuccess) {                                        \
      std::ostringstream err_msg;                                                     \
      err_msg << "cutlass " << #cmd << " failed: " << cutlassGetStatusString(status); \
      FLASHINFER_ERROR(err_msg.str());                                                \
    }                                                                                 \
  } while (0)

}  // namespace flashinfer

#endif  // FLASHINFER_CUTLASS_UTILS_CUH_
