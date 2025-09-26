/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include <NvInferRuntime.h>

#include <cassert>
#include <vector>

namespace tensorrt_llm::kernels
{

using CublasGemmWrapper = tensorrt_llm::common::CublasMMWrapper;
using CublasGemmWrapperPtr = std::shared_ptr<CublasGemmWrapper>;
using Config = cublasLtMatmulHeuristicResult_t;

class LoraImpl
{
public:
    LoraImpl(int in_hidden_size, std::vector<int> out_hidden_sizes, bool transA, bool transB, int num_lora_modules,
        nvinfer1::DataType type, int max_low_rank, std::shared_ptr<CublasGemmWrapper> cublasWrapper);

    [[nodiscard]] size_t getWorkspaceSize(int64_t numTokens, int64_t numReqs, nvinfer1::DataType type) const noexcept;
    void setBestTactic(std::optional<Config> config);
    int run(int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
        void const* const* loraWeightsPtr, int weightIndex, void* const* outputs, void* workspace, cudaStream_t stream);

    void setGemmConfig();

    [[nodiscard]] CublasGemmWrapperPtr getCublasWrapper() const
    {
        return mCublasWrapper;
    }

private:
    bool mTransA;
    bool mTransB;
    nvinfer1::DataType mType;
    int mNumLoraModules;

    // @fixme: seems this is shared across multiple clones.
    // If we deep copy the wrapper inside clone(), then we may avoid the mutex inside the wrapper?
    CublasGemmWrapperPtr mCublasWrapper;

    int mInHiddenSize;
    std::vector<int> mOutHiddenSizes;
    int mMaxLowRank;
    static int constexpr mSplitKSlices = 16;

    std::optional<Config> mBestConfig;
};

// Change to following declarations must sync with moe_kernels.h in internal kernel repo
int Lora_run(LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
    void const* const* loraWeightsPtr, int weightIndex, void* const* outputs, void* workspace, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
