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

#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace tensorrt_llm::runtime
{

class WorkerPool
{
public:
    explicit WorkerPool(std::size_t numWorkers = 1, std::int32_t deviceId = -1);

    WorkerPool(WorkerPool const&) = delete;
    WorkerPool(WorkerPool&&) = delete;
    WorkerPool& operator=(WorkerPool const&) = delete;
    WorkerPool& operator=(WorkerPool&&) = delete;
    ~WorkerPool();

    template <class F>
    auto enqueue(F&& task) -> std::future<typename std::invoke_result<F>::type>
    {
        using returnType = typename std::invoke_result<F>::type;
        auto const taskPromise = std::make_shared<std::promise<returnType>>();
        {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            mTasks.push(
                [task = std::forward<F>(task), taskPromise]()
                {
                    try
                    {
                        if constexpr (std::is_void_v<returnType>)
                        {
                            task();
                            taskPromise->set_value();
                        }
                        else
                        {
                            taskPromise->set_value(task());
                        }
                    }
                    catch (...)
                    {
                        taskPromise->set_exception(std::current_exception());
                    }
                });
        }
        condition.notify_one();
        return taskPromise->get_future();
    }

private:
    std::vector<std::thread> mWorkers;
    std::queue<std::function<void()>> mTasks;

    std::mutex mQueueMutex;
    std::condition_variable condition;
    bool stop{};
};

} // namespace tensorrt_llm::runtime
