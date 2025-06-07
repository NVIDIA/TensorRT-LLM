/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <optional>
#include <torch/extension.h>

namespace th = torch;

namespace torch_ext
{
th::Tensor& cublas_mm_out(
    th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias, th::Tensor& out);

th::Tensor cublas_mm(th::Tensor const& mat_a, th::Tensor const& mat_b, std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> out_dtype);

th::Tensor cublas_scaled_mm(th::Tensor const& mat_a, th::Tensor const& mat_b, th::Tensor const& scale_a,
    th::Tensor const& scale_b, std::optional<at::Tensor> const& bias, std::optional<c10::ScalarType> out_dtype,
    bool to_userbuffers = false);

th::Tensor cublas_scaled_mm_out(th::Tensor const& mat_a, th::Tensor const& mat_b, th::Tensor const& scale_a,
    th::Tensor const& scale_b, std::optional<at::Tensor> const& bias, th::Tensor& out);
} // namespace torch_ext
