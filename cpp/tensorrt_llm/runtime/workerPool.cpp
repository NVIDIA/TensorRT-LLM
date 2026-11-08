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

#include "workerPool.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::runtime
{
WorkerPool::WorkerPool(std::size_t numWorkers, std::int32_t deviceId)
{
    for (std::size_t i = 0; i < numWorkers; ++i)
    {
        mWorkers.emplace_back(
            [this, deviceId]
            {
                if (deviceId >= 0)
                {
                    TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
                }
                else
                {
                    TLLM_LOG_WARNING("WorkerPool did not set cuda device");
                }

                while (true)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->mQueueMutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->mTasks.empty(); });
                        if (this->stop && this->mTasks.empty())
                        {
                            return;
                        }
                        task = std::move(this->mTasks.front());
                        this->mTasks.pop();
                    }

                    task();
                }
            });
    }
}

WorkerPool::~WorkerPool()
{
    {
        std::unique_lock<std::mutex> lock(mQueueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : mWorkers)
    {
        worker.join();
    }
}
} // namespace tensorrt_llm::runtime
