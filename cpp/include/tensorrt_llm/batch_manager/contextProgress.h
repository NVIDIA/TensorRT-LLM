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

#pragma once

#include "tensorrt_llm/runtime/cudaEvent.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace tensorrt_llm::batch_manager
{

// Use to track progress of context phase in dist-serving
class ContextProgress
{
public:
    ContextProgress(int numLayers);

    void recordEvent(int layerIdx, cudaStream_t stream);

    void wait(int layerIdx);

    int getNumLayers() const
    {
        return mCudaEvents.size();
    }

    cudaEvent_t getEvent(int layerIdx)
    {
        return mCudaEvents.at(layerIdx).get();
    }

private:
    std::mutex mMutex;
    std::condition_variable mConditionVariable;
    std::unique_ptr<std::atomic_bool[]> mCudaEventsRecorded;
    std::vector<runtime::CudaEvent> mCudaEvents;
};

} // namespace tensorrt_llm::batch_manager
