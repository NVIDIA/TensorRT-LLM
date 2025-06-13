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

#include "NvInferPlugin.h"

#if defined(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
#include "tensorrt_llm/kernels/cutlass_kernels/include/allreduce_gemm_runner.h"
#else
#include "allreduce_gemm_runner.h"
#endif
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/plugins/common/plugin.h"

using namespace nvinfer1;

namespace tensorrt_llm::plugins
{

#if defined(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
namespace cutlass_kernels = ::tensorrt_llm::kernels::opened_cutlass_kernels;
#else
namespace cutlass_kernels = ::tensorrt_llm::kernels::cutlass_kernels;
#endif
class GemmAllReducePersistentWorkspace : public IPluginResource
{
public:
    GemmAllReducePersistentWorkspace(std::shared_ptr<cutlass_kernels::PersistentWorkspaceInterface> workspace)
        : mWorkspace(workspace)
    {
    }

    //////////////////////////////////
    // IPluginResource Methods
    //////////////////////////////////
    IPluginResource* clone() noexcept override
    {
        auto copy = new GemmAllReducePersistentWorkspace(mWorkspace);
        // Resource initialization (if any) may be skipped for non-cloned objects
        // since only clones will be registered by TensorRT.
        try
        {
            copy->mWorkspace->allocate();
            return copy;
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR(e.what());
            return nullptr;
        }
    }

    int32_t release() noexcept override
    {
        try
        {
            return mWorkspace->free();
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_ERROR(e.what());
            return -1;
        }
    }

    std::shared_ptr<cutlass_kernels::PersistentWorkspaceInterface> mWorkspace;
};

} // namespace tensorrt_llm::plugins
