/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Conversion Utility to convert RMEM from one type to another. Used for conversion from AccumType
// to input/output type.
template <typename To_type, typename From_type, typename Fragment>
inline __device__ auto convert_type(Fragment const& tensor) {
  // The number of the elements in the source.
  constexpr int numel = decltype(size(tensor))::value;
  // The converter.
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // The data of the input.
  auto const* data = reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data());
  // Create the destination tensor (at least the array in registers). The src must be contiguous.
  auto dst = convert_op(*data);
  // Reconstruct the tensor.
  return cute::make_tensor(cute::make_rmem_ptr<To_type>(&dst), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
