/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "kernelList.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"

namespace tensorrt_llm
{
namespace kernels
{
class TrtllmGenBlockScaleGemmRunner
{
public:
    explicit TrtllmGenBlockScaleGemmRunner(Data_type outputType);

    void run(int32_t m, int32_t n, int32_t k, void const* a, float const* aScale, void const* b, float const* bScale,
        void* c, float* cScale, CUstream stream);

private:
    Data_type mOutputType;
    TrtllmGenBlockScaleGemmInfo const* mKernelInfo;
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    CUmodule mModule;
    CUfunction mFunction;
};
} // namespace kernels
} // namespace tensorrt_llm
