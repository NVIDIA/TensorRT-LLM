/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace tensorrt_llm::runtime
{

class WorkerPool
{
public:
    explicit WorkerPool(std::size_t numWorkers = 1, int device = -1)
        : mNumWorkers(numWorkers)
        , mShutdown(false)
        , mDevice(device)
    {
        initThreads();
    }

    ~WorkerPool()
    {
        shutdown();
    }

    template <typename Function, typename Return = std::invoke_result_t<std::decay_t<Function>>>
    std::future<Return> enqueue(Function&& task)
    {
        if (mShutdown)
        {
            throw std::runtime_error("WorkerPool is shutdown cannot enqueue new tasks");
        }

        auto const taskPromise = std::make_shared<std::promise<Return>>();
        std::lock_guard<std::mutex> lock(mTasksMutex);
        mTasks.push(
            [task = std::forward<Function>(task), taskPromise]()
            {
                try
                {
                    if constexpr (std::is_void_v<Return>)
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
        mTasksCv.notify_one();
        return taskPromise->get_future();
    }

private:
    static constexpr size_t kMaxNumWorkers = 128;
    std::size_t mNumWorkers;

    std::queue<std::function<void()>> mTasks{};
    mutable std::mutex mTasksMutex;
    std::condition_variable mTasksCv;

    std::atomic<bool> mShutdown = false;

    std::thread mThreads[kMaxNumWorkers];

    int mDevice{-1};

    void shutdown()
    {
        if (mShutdown)
        {
            return;
        }
        mShutdown = true;
        mTasksCv.notify_all();
        for (std::size_t i = 0; i < mNumWorkers; ++i)
        {
            mThreads[i].join();
        }
    }

    void initThreads()
    {
        if (mNumWorkers > kMaxNumWorkers)
        {
            throw std::runtime_error(
                "numWorker > maxNumWorkers " + std::to_string(mNumWorkers) + " > " + std::to_string(kMaxNumWorkers));
        }
        for (std::size_t i = 0; i < mNumWorkers; ++i)
        {
            mThreads[i] = std::thread(&WorkerPool::doWork, this);
        }
    }

    void doWork()
    {
        if (mDevice >= 0)
        {
            TLLM_CUDA_CHECK(cudaSetDevice(mDevice));
        }
        else
        {
            TLLM_LOG_WARNING("WorkerPool did not set cuda device");
        }
        while (!mShutdown)
        {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mTasksMutex);
                mTasksCv.wait(lock, [this]() { return !mTasks.empty() || mShutdown; });
                if (mTasks.empty())
                {
                    continue;
                }
                task = mTasks.front();
                mTasks.pop();
            }

            task();
        }
    }
};
} // namespace tensorrt_llm::runtime
