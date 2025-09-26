/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::executor::disagg_executor
{

namespace texec = tensorrt_llm::executor;

struct ResponseWithId
{

    tensorrt_llm::executor::Response response;
    IdType gid;

    ResponseWithId(tensorrt_llm::executor::Response&& response, IdType gid)
        : response(std::move(response))
        , gid(gid)
    {
    }

    ResponseWithId(tensorrt_llm::executor::Response const& response, IdType gid)
        : response(response)
        , gid(gid)
    {
    }

    ResponseWithId(ResponseWithId&& other) noexcept
        : response(std::move(other.response))
        , gid(other.gid)
    {
        other.gid = {};
    }

    ResponseWithId(ResponseWithId const& other) = default;

    ResponseWithId& operator=(ResponseWithId&& other) noexcept
    {
        if (this != &other)
        {
            response = std::move(other.response);
            gid = other.gid;
            other.gid = {};
        }
        return *this;
    }

    ResponseWithId& operator=(ResponseWithId const& other)
    {

        if (this != &other)
        {
            response = other.response;
            gid = other.gid;
        }
        return *this;
    }

    ~ResponseWithId() = default;
};

class DisaggExecutorOrchestrator
{
public:
    /// @brief Constructs a DisaggExecutorOrchestrator object.
    ///
    /// @param ctxEnginePaths A vector of file paths to context engine files.
    /// @param genEnginePaths A vector of file paths to generation engine files.
    /// @param ctxExecutorConfigs A vector of ExecutorConfig  for context executors.
    /// @param genExecutorConfigs A vector of ExecutorConfig  for generation executors.
    /// @param hasContextAwaitThreads Whether or not there are threads that receive response for each generation
    /// executor.
    /// @param hasGenAwaitThreads Whether or not there are threads that receive response for each generation executor.

    DisaggExecutorOrchestrator(std::vector<std::filesystem::path> const& ctxEnginePaths,
        std::vector<std::filesystem::path> const& genEnginePaths,
        std::vector<executor::ExecutorConfig> const& ctxExecutorConfigs,
        std::vector<executor::ExecutorConfig> const& genExecutorConfigs, bool hasContextAwaitThreads,
        bool hasGenAwaitThreads);

    /// @brief Enqueue context-only requests to context executors.
    /// @param requests A vector of context-only requests.
    /// @param selectContextId The index of the context executor to use. If `std::nullopt`, the executor that has the
    /// smallest number of inflight requests will be used.
    /// @param batch If true,enqueue requests in same context executor.If false, will try to use a different executor
    /// for each request.
    /// @return A vector of global request ids, corresponding to the order of the requests in `requests`, the id
    /// returned may be different from the request id in each executor.
    [[nodiscard]] std::vector<IdType> enqueueContext(std::vector<texec::Request> const& requests,
        std::optional<int> selectContextId = std::nullopt, bool batch = false);

    /// @brief Enqueue generation-only requests to generation executors.
    /// @param requests A vector of generation-only requests.
    /// @param globalRequestIds A vector of global request ids, corresponding to the order of the requests,and must be
    /// the ids returned by the enqueueContext function.
    /// @param selectGenIdx The index of the generation executor to use. If `std::nullopt`, the executor that has the
    /// smallest number of inflight requests will be used.
    /// @param batch If true,enqueue requests in same generation executor.If false, will try to use a different executor
    /// for each request.

    void enqueueGeneration(std::vector<texec::Request> const& requests, std::vector<IdType> const& globalRequestIds,
        std::optional<int> selectGenIdx = std::nullopt, bool batch = false);

    /// @brief Await for context responses
    /// @param timeout The maximum time to wait for new responses
    /// @param contextIdx The index of the context executor to use. If `std::nullopt`, return ready responses in all
    /// context executors,if `hasContextAwaitThreads` is true, then this parameter must be std::nullopt.
    /// @return A vector of responses with corresponding global request ids

    [[nodiscard]] std::vector<ResponseWithId> awaitContextResponses(
        std::optional<std::chrono::milliseconds> const& timeout, std::optional<int> contextIdx = std::nullopt);

    /// @brief Await for generation responses
    /// @param timeout The maximum time to wait for new responses.
    /// @param genIdx The index of the generation executor to use. If `std::nullopt`, return ready responses in all
    /// generation executors,if `hasGenAwaitThreads` is true, then this parameter must be std::nullopt.
    /// @return A vector of responses with corresponding global request ids.
    [[nodiscard]] std::vector<ResponseWithId> awaitGenerationResponses(
        std::optional<std::chrono::milliseconds> const& timeout, std::optional<int> genIdx = std::nullopt);

    /// @brief  Indicates if the current process is allowed to enqueueRequests
    [[nodiscard]] bool canEnqueue() const;

    /// @brief Get context executors
    [[nodiscard]] std::vector<std::unique_ptr<texec::Executor>> const& getContextExecutors() const;

    /// @brief Get generation executors
    [[nodiscard]] std::vector<std::unique_ptr<texec::Executor>> const& getGenExecutors() const;

    ~DisaggExecutorOrchestrator();

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};
} // namespace tensorrt_llm::executor::disagg_executor
