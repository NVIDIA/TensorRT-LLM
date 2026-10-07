/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "cublasScaledMM.h"

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "cublas_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, Tensor? bias,"
        " ScalarType? out_dtype, int output_buffer_kind=0, int[]? group=None)"
        " -> (Tensor out)");
    m.def(
        "cublas_mm(Tensor mat_a, Tensor mat_b, Tensor? bias, ScalarType? out_dtype,"
        " int output_buffer_kind=0, int[]? group=None) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cublas_scaled_mm", &tensorrt_llm::torch_ext::cublas_scaled_mm);
    m.impl("cublas_mm", &tensorrt_llm::torch_ext::cublas_mm);
}
