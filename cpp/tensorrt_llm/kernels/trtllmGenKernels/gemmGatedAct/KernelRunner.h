/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda.h>
#include <optional>

#include "tensorrt_llm/kernels/trtllmGenKernels/common/Dtype.h"

namespace tensorrt_llm
{
namespace kernels
{

struct TrtllmGenGemmGatedActRunnerOptions
{
    Dtype eltType;
    Dtype outputType;
    bool deepSeekFp8{false};
    bool transposeMmaOutput{false};
};

class TrtllmGenGemmGatedActRunner
{
public:
    explicit TrtllmGenGemmGatedActRunner(TrtllmGenGemmGatedActRunnerOptions const& options);

    [[nodiscard]] size_t getWorkspaceSizeInBytes(int32_t m, int32_t n, int32_t k);

    void run(int32_t m, int32_t n, int32_t k, void const* a, float const* aScale, void const* b, float const* bScale,
        void* c, float* cScale, float* cScaleGate, void* workspace, CUstream stream, int device);

    void run(int32_t m, int32_t n, int32_t k, void const* a, void const* b, void* c, float* cScale, float* cScaleGate,
        void* workspace, CUstream stream, int device);

private:
    void selectGemmConfig(int32_t m, int32_t n, int32_t k);

private:
    TrtllmGenGemmGatedActRunnerOptions mOptions;
    std::optional<int> mSelectedConfigIndex;
    std::vector<int32_t> mPassingConfigIndices;
};
} // namespace kernels
} // namespace tensorrt_llm
