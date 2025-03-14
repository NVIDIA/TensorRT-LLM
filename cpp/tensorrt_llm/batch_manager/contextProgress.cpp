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

#include "tensorrt_llm/batch_manager/contextProgress.h"

namespace tensorrt_llm::batch_manager
{

ContextProgress::ContextProgress(int numLayers)
{
    mCudaEventsRecorded = std::make_unique<std::atomic_bool[]>(numLayers);
    mCudaEvents.reserve(numLayers);
    for (int i = 0; i < numLayers; i++)
    {
        mCudaEventsRecorded[i] = false;
        mCudaEvents.emplace_back(cudaEventBlockingSync | cudaEventDisableTiming);
    }
    TLLM_LOG_DEBUG("ContextProgress created - expect %d layers", numLayers);
}

void ContextProgress::recordEvent(int layerIdx, cudaStream_t stream)
{
    TLLM_CHECK(layerIdx < getNumLayers());
    TLLM_CHECK_WITH_INFO(layerIdx == 0 || mCudaEventsRecorded[layerIdx - 1], "Layer %d is skipped", layerIdx - 1);
    TLLM_CHECK_WITH_INFO(!mCudaEventsRecorded[layerIdx], "Layer %d is recorded twice", layerIdx);
    TLLM_CUDA_CHECK(cudaEventRecord(mCudaEvents[layerIdx].get(), stream));
    mCudaEventsRecorded[layerIdx] = true;
    mConditionVariable.notify_all();
}

void ContextProgress::wait(int layerIdx)
{
    TLLM_CHECK(layerIdx < getNumLayers());
    while (!mCudaEventsRecorded[layerIdx])
    {
        std::unique_lock lock(mMutex);
        auto const timeout = std::chrono::milliseconds(10);
        mConditionVariable.wait_for(lock, timeout);
    }
    mCudaEvents[layerIdx].synchronize();
}

} // namespace tensorrt_llm::batch_manager
