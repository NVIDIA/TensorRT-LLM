/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include <nanobind/nanobind.h>

namespace nb = nanobind;
namespace tle = tensorrt_llm::executor;

namespace tensorrt_llm::nanobind::executor
{

class Executor
{
public:
    Executor(
        std::filesystem::path const& modelPath, tle::ModelType modelType, tle::ExecutorConfig const& executorConfig);

    Executor(std::filesystem::path const& encoderModelPath, std::filesystem::path const& decoderModelPath,
        tle::ModelType modelType, tle::ExecutorConfig const& executorConfig);

    Executor(nb::bytes const& engineBuffer, std::string const& jsonConfigStr, tle::ModelType modelType,
        tle::ExecutorConfig const& executorConfig, std::optional<nb::dict> managedWeights);

    Executor(std::string const& encoderEngineBuffer, std::string const& encoderJsonConfigStr,
        std::string const& decoderEngineBuffer, std::string const& decoderJsonConfigStr, tle::ModelType modelType,
        tle::ExecutorConfig const& executorConfig);

    nb::object enter();
    void exit(
        [[maybe_unused]] nb::handle type, [[maybe_unused]] nb::handle value, [[maybe_unused]] nb::handle traceback);
    void shutdown();

    [[nodiscard]] tle::IdType enqueueRequest(tle::Request const& request)
    {
        return mExecutor->enqueueRequest(request);
    }

    [[nodiscard]] std::vector<tle::IdType> enqueueRequests(std::vector<tle::Request> const& requests)
    {
        return mExecutor->enqueueRequests(requests);
    }

    [[nodiscard]] std::vector<tle::Response> awaitResponses(
        std::optional<std::chrono::milliseconds> const& timeout = std::nullopt)
    {
        // Await responses blocks until a response is received. Release GIL so that it can be ran in a background
        // thread.
        nb::gil_scoped_release release;
        return mExecutor->awaitResponses(timeout);
    }

    [[nodiscard]] std::vector<tle::Response> awaitResponses(
        tle::IdType const& requestId, std::optional<std::chrono::milliseconds> const& timeout = std::nullopt)
    {
        // Await responses blocks until a response is received. Release GIL so that it can be ran in a background
        // thread.
        nb::gil_scoped_release release;
        return mExecutor->awaitResponses(requestId, timeout);
    }

    [[nodiscard]] std::vector<std::vector<tle::Response>> awaitResponses(std::vector<tle::IdType> const& requestIds,
        std::optional<std::chrono::milliseconds> const& timeout = std::nullopt)
    {
        // Await responses blocks until a response is received. Release GIL so that it can be ran in a background
        // thread.
        nb::gil_scoped_release release;
        return mExecutor->awaitResponses(requestIds, timeout);
    }

    [[nodiscard]] tle::SizeType32 getNumResponsesReady(std::optional<tle::IdType> const& requestId = std::nullopt) const
    {
        return mExecutor->getNumResponsesReady(requestId);
    }

    void cancelRequest(tle::IdType requestId)
    {
        mExecutor->cancelRequest(requestId);
    }

    std::deque<tle::IterationStats> getLatestIterationStats()
    {
        return mExecutor->getLatestIterationStats();
    }

    std::deque<tle::RequestStatsPerIteration> getLatestRequestStats()
    {
        return mExecutor->getLatestRequestStats();
    }

    std::deque<tle::DebugTensorsPerIteration> getLatestDebugTensors()
    {
        return mExecutor->getLatestDebugTensors();
    }

    [[nodiscard]] bool canEnqueueRequests() const
    {
        return mExecutor->canEnqueueRequests();
    }

    [[nodiscard]] std::optional<std::shared_ptr<tle::KVCacheEventManager>> getKVCacheEventManager() const
    {
        return mExecutor->getKVCacheEventManager();
    }

    static void initBindings(nb::module_& m);

private:
    std::unique_ptr<tle::Executor> mExecutor;
};

} // namespace tensorrt_llm::nanobind::executor
