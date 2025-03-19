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

#pragma once

#include "tensorrt_llm/common/assert.h"
#include <atomic>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <iostream>

namespace tensorrt_llm::batch_manager::utils
{

// A simple static thread pool to avoid the overhead caused by having too many threads.
class StaticThreadPool
{
public:
    explicit StaticThreadPool(std::size_t numThreads);

    StaticThreadPool(StaticThreadPool const&) = delete;
    StaticThreadPool& operator=(StaticThreadPool const&) = delete;

    ~StaticThreadPool();

    // TODO: Performance optimization.
    template <typename TFunction, typename... TArgs>
    [[nodiscard]] std::future<std::invoke_result_t<TFunction, TArgs...>> execute(TFunction&& f, TArgs&&... args)
    {
        TLLM_CHECK(!mTerminate);
        auto task = std::make_shared<std::packaged_task<std::invoke_result_t<TFunction, TArgs...>()>>(
            std::bind(std::forward<TFunction>(f), std::forward<TArgs>(args)...));
        auto res = task->get_future();
        {
            std::unique_lock lock(mQueueMutex);
            mQueue.push([taskCapture = std::move(task)] { (*taskCapture)(); });
        }
        return res;
    }

    void requestStop();

private:
    void workerThread();

    void join();

    std::atomic<bool> mTerminate{false};
    std::queue<std::function<void()>> mQueue;
    std::mutex mQueueMutex;
    std::vector<std::thread> mThreads;
};

} // namespace tensorrt_llm::batch_manager::utils
