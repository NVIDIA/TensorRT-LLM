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

#include "staticThreadPool.h"

namespace tensorrt_llm::batch_manager::utils
{

StaticThreadPool::StaticThreadPool(std::size_t numThreads)
{
    TLLM_CHECK_WITH_INFO(numThreads > 0, "The number of threads must be greater than 0.");
    try
    {
        for (std::size_t i = 0; i < numThreads; ++i)
        {
            mThreads.emplace_back(std::thread(&StaticThreadPool::workerThread, this));
        }
    }
    catch (...)
    {
        requestStop();
        join();
    }
}

void StaticThreadPool::join()
{
    for (auto& thread : mThreads)
    {
        thread.join();
    }
}

StaticThreadPool::~StaticThreadPool()
{
    requestStop();
    join();
}

void StaticThreadPool::requestStop()
{
    mTerminate = true;
}

void StaticThreadPool::workerThread()
{
    while (!mTerminate)
    {
        std::unique_lock lock(mQueueMutex);
        if (mQueue.size())
        {
            auto task = std::move(mQueue.front());
            mQueue.pop();
            lock.unlock();
            task();
        }
        else
        {
            std::this_thread::yield();
        }
    }
}

} // namespace tensorrt_llm::batch_manager::utils
