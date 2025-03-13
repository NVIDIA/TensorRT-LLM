/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"

namespace
{
void validateCudaGraphCacheSize(bool cudaGraphMode, tensorrt_llm::executor::SizeType32 cudaGraphCacheSize)
{
    TLLM_CHECK_WITH_INFO(cudaGraphCacheSize >= 0, "CUDA graph cache size must be greater or equal to 0.");
    if (!cudaGraphMode && cudaGraphCacheSize > 0)
    {
        TLLM_LOG_WARNING(
            "Setting cudaGraphCacheSize to a value greater than 0 without enabling cudaGraphMode has no effect.");
    }
}

} // namespace

namespace tensorrt_llm::executor
{

ExtendedRuntimePerfKnobConfig::ExtendedRuntimePerfKnobConfig(
    bool multiBlockMode, bool enableContextFMHAFP32Acc, bool cudaGraphMode, SizeType32 cudaGraphCacheSize)
    : mMultiBlockMode(multiBlockMode)
    , mEnableContextFMHAFP32Acc(enableContextFMHAFP32Acc)
    , mCudaGraphMode(cudaGraphMode)
    , mCudaGraphCacheSize(cudaGraphCacheSize)
{
    validateCudaGraphCacheSize(mCudaGraphMode, mCudaGraphCacheSize);
}

bool ExtendedRuntimePerfKnobConfig::getMultiBlockMode() const
{
    return mMultiBlockMode;
}

bool ExtendedRuntimePerfKnobConfig::getEnableContextFMHAFP32Acc() const
{
    return mEnableContextFMHAFP32Acc;
}

bool ExtendedRuntimePerfKnobConfig::getCudaGraphMode() const
{
    return mCudaGraphMode;
}

SizeType32 ExtendedRuntimePerfKnobConfig::getCudaGraphCacheSize() const
{
    return mCudaGraphCacheSize;
}

void ExtendedRuntimePerfKnobConfig::setMultiBlockMode(bool multiBlockMode)
{
    mMultiBlockMode = multiBlockMode;
}

void ExtendedRuntimePerfKnobConfig::setEnableContextFMHAFP32Acc(bool enableContextFMHAFP32Acc)
{
    mEnableContextFMHAFP32Acc = enableContextFMHAFP32Acc;
}

void ExtendedRuntimePerfKnobConfig::setCudaGraphMode(bool cudaGraphMode)
{
    mCudaGraphMode = cudaGraphMode;
}

void ExtendedRuntimePerfKnobConfig::setCudaGraphCacheSize(SizeType32 cudaGraphCacheSize)
{
    mCudaGraphCacheSize = cudaGraphCacheSize;
    validateCudaGraphCacheSize(mCudaGraphMode, mCudaGraphCacheSize);
}

} // namespace tensorrt_llm::executor
