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

#include "tensorrt_llm/executor/disaggServerUtil.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <mutex>

namespace tensorrt_llm::executor::disagg_executor
{

class DisaggExecutorOrchestrator::Impl
{
public:
    Impl(std::vector<std::filesystem::path> const& ctxEnginePaths,
        std::vector<std::filesystem::path> const& genEnginePaths,
        std::vector<texec::ExecutorConfig> const& ctxExecutorConfigs,
        std::vector<texec::ExecutorConfig> const& genExecutorConfigs, bool hasContextAwaitThreads,
        bool hasGenAwaitThreads)
        : mhasContextAwaitThreads(hasContextAwaitThreads)
        , mhasGenAwaitThreads(hasGenAwaitThreads)
    {
        TLLM_CHECK(ctxEnginePaths.size() == ctxExecutorConfigs.size());
        TLLM_CHECK(genEnginePaths.size() == genExecutorConfigs.size());
        TLLM_CHECK(!(ctxEnginePaths.empty() || genEnginePaths.empty()));
        int worldRank = tensorrt_llm::mpi::MpiComm::world().getRank();
        mIsOrchestrator = (worldRank == 0);
        auto contextNum = ctxEnginePaths.size();
        mContextReqIdToGlobalId = std::vector<std::unordered_map<IdType, IdType>>(contextNum);
        mContextMapMutexs = std::vector<std::mutex>(contextNum);
        auto genNum = genEnginePaths.size();
        mGenerationReqIdToGlobalId = std::vector<std::unordered_map<IdType, IdType>>(genNum);
        mGenerationMapMutexs = std::vector<std::mutex>(genNum);

        for (size_t cN = 0; cN < contextNum; cN++)
        {
            mContextExecutors.push_back(std::make_unique<texec::Executor>(
                ctxEnginePaths[cN], texec::ModelType::kDECODER_ONLY, ctxExecutorConfigs[cN]));
        }

        for (size_t gN = 0; gN < genNum; gN++)
        {
            mGenerationExecutors.push_back(std::make_unique<texec::Executor>(
                genEnginePaths[gN], texec::ModelType::kDECODER_ONLY, genExecutorConfigs[gN]));
        }

        if (mIsOrchestrator)
        {
            if (mhasContextAwaitThreads)
            {
                for (size_t contextIdx = 0; contextIdx < contextNum; contextIdx++)
                {
                    mContextThreads.emplace_back(
                        [this, contextIdx]() { this->waitResponseAndAppendThreadFun(true, contextIdx); });
                }
            }
            if (mhasGenAwaitThreads)
            {

                for (size_t genIdx = 0; genIdx < genNum; genIdx++)
                {
                    mGenerationThreads.emplace_back(
                        [this, genIdx]() { this->waitResponseAndAppendThreadFun(false, genIdx); });
                }
            }
        }
        tensorrt_llm::mpi::MpiComm::world().barrier();
    }

    std::vector<IdType> enqueueContext(std::vector<texec::Request> const& requests,
        std::optional<int> selectContextId = std::nullopt, bool batch = false)
    {

        std::vector<IdType> globalReqIds;
        for (auto const& request : requests)
        {
            globalReqIds.push_back(generatedGlobalId());
            TLLM_CHECK(request.getRequestType() == tensorrt_llm::executor::RequestType::REQUEST_TYPE_CONTEXT_ONLY);
        }

        if (batch)
        {
            size_t contextId = selectContextId.has_value() ? selectContextId.value() : selectContextExecutor();
            auto contextReqIds = mContextExecutors[contextId]->enqueueRequests(requests);
            {
                std::scoped_lock<std::mutex> lock{mContextMapMutexs[contextId]};
                for (size_t i = 0; i < requests.size(); ++i)
                {
                    mContextReqIdToGlobalId[contextId][contextReqIds[i]] = globalReqIds[i];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < requests.size(); ++i)
            {
                size_t contextId = selectContextId.has_value() ? selectContextId.value() : selectContextExecutor();

                auto contextReqId = mContextExecutors[contextId]->enqueueRequest(requests[i]);
                {
                    std::scoped_lock<std::mutex> lock{mContextMapMutexs[contextId]};
                    mContextReqIdToGlobalId[contextId][contextReqId] = globalReqIds[i];
                }
            }
        }
        return globalReqIds;
    }

    void enqueueGeneration(std::vector<texec::Request> const& requests, std::vector<IdType> const& globalRequestIds,
        std::optional<int> selectGenIdx = std::nullopt, bool batch = false)
    {

        TLLM_CHECK(globalRequestIds.size() == requests.size());

        for (auto const& request : requests)
        {

            TLLM_CHECK(request.getRequestType() == tensorrt_llm::executor::RequestType::REQUEST_TYPE_GENERATION_ONLY);
        }
        if (batch)
        {
            size_t genIdx = selectGenIdx.has_value() ? selectGenIdx.value() : selectGenerationExecutor();
            auto genReqIds = mGenerationExecutors[genIdx]->enqueueRequests(requests);
            {
                std::scoped_lock<std::mutex> lock{mGenerationMapMutexs[genIdx]};
                for (size_t i = 0; i < requests.size(); ++i)
                {
                    mGenerationReqIdToGlobalId[genIdx][genReqIds[i]] = globalRequestIds[i];
                }
            }
        }
        else
        {
            for (size_t i = 0; i < requests.size(); ++i)
            {
                size_t genIdx = selectGenIdx.has_value() ? selectGenIdx.value() : selectGenerationExecutor();

                auto genReqId = mGenerationExecutors[genIdx]->enqueueRequest(requests[i]);
                {
                    std::scoped_lock<std::mutex> lock{mGenerationMapMutexs[genIdx]};
                    mGenerationReqIdToGlobalId[genIdx][genReqId] = globalRequestIds[i];
                }
            }
        }
    }

    std::vector<ResponseWithId> awaitContextResponses(
        std::optional<int> contextIdx, std::optional<std::chrono::milliseconds> const& timeout)
    {

        std::vector<ResponseWithId> responses;

        if (mhasContextAwaitThreads)
        {

            std::unique_lock<std::mutex> lock(mResponsesContextMtx);
            auto pred = [&mShutdown = mShutdown, &resp = this->mContextResponses]() -> bool
            { return !resp.empty() || mShutdown; };
            auto storeResponses = [&resp = this->mContextResponses, &responses]()
            {
                responses = std::move(resp);
                resp.clear();
            };
            if (timeout)
            {
                if (mContextResponsesCV.wait_for(lock, timeout.value(), pred))
                {
                    storeResponses();
                }
            }
            else
            {
                mContextResponsesCV.wait(lock, pred);
                storeResponses();
            }
            TLLM_CHECK_WITH_INFO(
                !contextIdx.has_value(), "contextIdx should not be provided when mhasContextAwaitThreads is true");

            return responses;
        }

        if (contextIdx.has_value())
        {
            TLLM_CHECK(!mhasContextAwaitThreads);
            auto responseFromExecutor = mContextExecutors[contextIdx.value()]->awaitResponses(timeout);
            for (auto&& resp : responseFromExecutor)
            {

                auto reqId = resp.getRequestId();
                IdType globalId{0};

                {
                    std::scoped_lock<std::mutex> lock{mContextMapMutexs.at(contextIdx.value())};
                    globalId = mContextReqIdToGlobalId.at(contextIdx.value()).at(reqId);
                }
                TLLM_CHECK(globalId != 0);
                responses.emplace_back(std::move(resp), globalId);
            }
            return responses;
        }
        TLLM_CHECK(timeout.has_value());
        auto timeouP = timeout.value() / mContextExecutors.size();
        for (size_t ci = 0; ci < mContextExecutors.size(); ci++)
        {
            auto responseFromExecutor = mContextExecutors.at(ci)->awaitResponses(timeouP);
            for (auto&& resp : responseFromExecutor)
            {
                auto reqId = resp.getRequestId();
                IdType globalId{0};

                {
                    std::scoped_lock<std::mutex> lock{mContextMapMutexs.at(ci)};
                    globalId = mContextReqIdToGlobalId.at(ci).at(reqId);
                }
                TLLM_CHECK(globalId != 0);
                responses.emplace_back(std::move(resp), globalId);
            }
        }

        return responses;
    };

    std::vector<ResponseWithId> awaitGenerationResponses(
        std::optional<int> genIdx, std::optional<std::chrono::milliseconds> const& timeout)
    {

        std::vector<ResponseWithId> responses;

        if (mhasGenAwaitThreads)
        {

            std::unique_lock<std::mutex> lock(mResponseGenerationMtx);
            auto pred = [&mShutdown = mShutdown, &resp = this->mGenerationResponses]() -> bool
            { return !resp.empty() || mShutdown; };
            auto storeResponses = [&resp = this->mGenerationResponses, &responses]()
            {
                responses = std::move(resp);
                resp.clear();
            };
            if (timeout)
            {
                if (mGenerationResponsesCv.wait_for(lock, timeout.value(), pred))
                {
                    storeResponses();
                }
            }
            else
            {
                mGenerationResponsesCv.wait(lock, pred);
                storeResponses();
            }
            TLLM_CHECK_WITH_INFO(!genIdx.has_value(), "genIdx should not be provided when mhasGenAwaitThreads is true");
            return responses;
        }

        if (genIdx.has_value())
        {
            TLLM_CHECK(!mhasGenAwaitThreads);
            auto responseFromExecutor = mGenerationExecutors[genIdx.value()]->awaitResponses(timeout);
            for (auto&& resp : responseFromExecutor)
            {

                auto reqId = resp.getRequestId();
                IdType globalId{0};

                {
                    std::scoped_lock<std::mutex> lock{mGenerationMapMutexs.at(genIdx.value())};
                    globalId = mGenerationReqIdToGlobalId.at(genIdx.value()).at(reqId);
                }
                TLLM_CHECK(globalId != 0);
                responses.emplace_back(std::move(resp), globalId);
            }
            return responses;
        }
        TLLM_CHECK(timeout.has_value());
        auto timeouP = timeout.value() / mGenerationExecutors.size();

        for (size_t gi = 0; gi < mGenerationExecutors.size(); gi++)
        {
            auto responseFromExecutor = mGenerationExecutors.at(gi)->awaitResponses(timeouP);
            for (auto&& resp : responseFromExecutor)
            {

                auto reqId = resp.getRequestId();
                IdType globalId{0};

                {
                    std::scoped_lock<std::mutex> lock{mGenerationMapMutexs.at(gi)};
                    globalId = mGenerationReqIdToGlobalId.at(gi).at(reqId);
                }
                TLLM_CHECK(globalId != 0);
                responses.emplace_back(std::move(resp), globalId);
            }
        }

        return responses;
    };

    [[nodiscard]] bool canEnqueue() const
    {
        return mIsOrchestrator;
    }

    [[nodiscard]] std::vector<std::unique_ptr<texec::Executor>> const& getContextExecutors() const
    {
        return mContextExecutors;
    }

    [[nodiscard]] std::vector<std::unique_ptr<texec::Executor>> const& getGenExecutors() const
    {
        return mGenerationExecutors;
    }

    ~Impl()
    {

        mShutdown = true;

        mContextResponsesCV.notify_all();
        mGenerationResponsesCv.notify_all();
        for (auto&& executor : mContextExecutors)
        {
            executor->shutdown();
        }
        for (auto&& executor : mGenerationExecutors)
        {
            executor->shutdown();
        }

        if (mIsOrchestrator)
        {
            if (mhasContextAwaitThreads)
            {
                for (auto&& contextThread : mContextThreads)
                {
                    if (contextThread.joinable())
                    {
                        contextThread.join();
                    }
                }
            }
            if (mhasGenAwaitThreads)
            {
                for (auto&& genThread : mGenerationThreads)
                {
                    if (genThread.joinable())
                    {
                        genThread.join();
                    }
                }
            }
        }
    }

private:
    IdType generatedGlobalId()
    {
        return (++mLastId % UINT64_MAX);
    };

    size_t selectContextExecutor()
    {
        static size_t selectContextId = 0;
        auto contextId = (selectContextId++) % mContextExecutors.size();
        if (selectContextId >= mContextExecutors.size())
        {
            selectContextId = 0;
        }
        return contextId;
    }

    size_t selectGenerationExecutor()
    {
        static size_t selectGenerationId = 0;
        auto generationIdx = (selectGenerationId++) % mGenerationExecutors.size();
        if (selectGenerationId >= mGenerationExecutors.size())
        {
            selectGenerationId = 0;
        }
        return generationIdx;
    }

    void appendNewContextResponse(std::vector<ResponseWithId>&& newResponses)
    {
        {
            std::scoped_lock<std::mutex> lock(mResponsesContextMtx);
            for (auto&& response : newResponses)
            {
                mContextResponses.emplace_back(std::move(response));
            }
        }
        mContextResponsesCV.notify_all();
    }

    void appendNewGenerationResponse(std::vector<ResponseWithId>&& newResponses)
    {
        {
            std::scoped_lock<std::mutex> lock(mResponseGenerationMtx);
            for (auto&& response : newResponses)
            {
                mGenerationResponses.emplace_back(std::move(response));
            }
        }
        mGenerationResponsesCv.notify_all();
    }

    void waitResponseAndAppendThreadFun(bool isContext, int executorIdx)
    {

        tensorrt_llm::common::setThreadName("waitResponseAndAppendThreadFun");

        auto& executor = isContext ? mContextExecutors[executorIdx] : mGenerationExecutors[executorIdx];

        while (!mShutdown)
        {
            auto responses = executor->awaitResponses();

            if (responses.empty())
            {
                continue;
            }
            std::vector<ResponseWithId> responseWithIds;
            if (isContext)
            {
                for (auto&& response : responses)
                {
                    auto reqId = response.getRequestId();
                    IdType globalId{0};

                    {
                        std::scoped_lock<std::mutex> lock{mContextMapMutexs.at(executorIdx)};
                        globalId = mContextReqIdToGlobalId.at(executorIdx).at(reqId);
                    }
                    TLLM_CHECK(globalId != 0);
                    responseWithIds.emplace_back(std::move(response), globalId);
                }
                if (responseWithIds.size() > 0)
                {
                    appendNewContextResponse(std::move(responseWithIds));
                }
            }
            else
            {

                for (auto&& response : responses)
                {
                    auto reqId = response.getRequestId();
                    IdType globalId{0};

                    {
                        std::scoped_lock<std::mutex> lock{mGenerationMapMutexs.at(executorIdx)};
                        globalId = mGenerationReqIdToGlobalId.at(executorIdx).at(reqId);
                    }
                    TLLM_CHECK(globalId != 0);
                    responseWithIds.emplace_back(std::move(response), globalId);
                }
                if (responseWithIds.size() > 0)
                {
                    appendNewGenerationResponse(std::move(responseWithIds));
                }
            }
        }
    };

    std::vector<std::unique_ptr<texec::Executor>> mContextExecutors;
    std::vector<std::unique_ptr<texec::Executor>> mGenerationExecutors;
    std::vector<std::thread> mContextThreads;
    std::vector<std::thread> mGenerationThreads;

    std::atomic<IdType> mLastId{0};
    std::vector<std::unordered_map<IdType, IdType>> mContextReqIdToGlobalId;
    std::vector<std::unordered_map<IdType, IdType>> mGenerationReqIdToGlobalId;
    std::vector<std::mutex> mContextMapMutexs;
    std::vector<std::mutex> mGenerationMapMutexs;
    std::vector<ResponseWithId> mContextResponses;
    std::condition_variable mContextResponsesCV;
    std::mutex mResponsesContextMtx;

    std::vector<ResponseWithId> mGenerationResponses;
    std::condition_variable mGenerationResponsesCv;
    std::mutex mResponseGenerationMtx;
    std::atomic<bool> mShutdown{false};
    std::atomic<bool> mhasContextAwaitThreads{false};
    std::atomic<bool> mhasGenAwaitThreads{false};
    bool mIsOrchestrator{false};
};

DisaggExecutorOrchestrator::DisaggExecutorOrchestrator(std::vector<std::filesystem::path> const& ctxEnginePaths,
    std::vector<std::filesystem::path> const& genEnginePaths,
    std::vector<executor::ExecutorConfig> const& ctxExecutorConfigs,
    std::vector<executor::ExecutorConfig> const& genExecutorConfigs, bool hasContextAwaitThreads,
    bool hasGenAwaitThreads)
    : mImpl(std::make_unique<DisaggExecutorOrchestrator::Impl>(ctxEnginePaths, genEnginePaths, ctxExecutorConfigs,
        genExecutorConfigs, hasContextAwaitThreads, hasGenAwaitThreads))
{
}

std::vector<IdType> DisaggExecutorOrchestrator::enqueueContext(
    std::vector<texec::Request> const& requests, std::optional<int> selectContextId, bool batch)
{
    return mImpl->enqueueContext(requests, selectContextId, batch);
}

void DisaggExecutorOrchestrator::enqueueGeneration(std::vector<texec::Request> const& requests,
    std::vector<IdType> const& globalRequestIds, std::optional<int> selectGenIdx, bool batch)
{
    mImpl->enqueueGeneration(requests, globalRequestIds, selectGenIdx, batch);
}

std::vector<ResponseWithId> DisaggExecutorOrchestrator::awaitContextResponses(
    std::optional<std::chrono::milliseconds> const& timeout, std::optional<int> contextIdx)
{
    return mImpl->awaitContextResponses(contextIdx, timeout);
}

std::vector<ResponseWithId> DisaggExecutorOrchestrator::awaitGenerationResponses(
    std::optional<std::chrono::milliseconds> const& timeout, std::optional<int> genIdx)
{
    return mImpl->awaitGenerationResponses(genIdx, timeout);
}

bool DisaggExecutorOrchestrator::canEnqueue() const
{
    return mImpl->canEnqueue();
};

std::vector<std::unique_ptr<texec::Executor>> const& DisaggExecutorOrchestrator::getContextExecutors() const
{
    return mImpl->getContextExecutors();
}

std::vector<std::unique_ptr<texec::Executor>> const& DisaggExecutorOrchestrator::getGenExecutors() const
{
    return mImpl->getGenExecutors();
}

DisaggExecutorOrchestrator::~DisaggExecutorOrchestrator() = default;

} // namespace tensorrt_llm::executor::disagg_executor
