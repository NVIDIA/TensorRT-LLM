/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/disaggServerUtil.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/requestWithId.h"
#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace tensorrt_llm::executor;
using namespace tensorrt_llm::executor::disagg_executor;
namespace su = tensorrt_llm::executor::serialize_utils;

namespace tensorrt_llm::testing::disaggexecutor
{

constexpr int32_t kM_INSTANCE_ID_TAG{12024};
constexpr int32_t kM_CONTROLLER_ID_TAG{22024};
constexpr int32_t kM_INSTANCE_DATA_TAG{32024};
constexpr int32_t kM_CONTROLLER_DATA_TAG{42024};

enum class MessageID : uint64_t
{
    PENDING_CONTEXT_REQUEST = 1,
    PENDING_GENERATION_REQUEST = 2,
    CONTEXT_RESPONSE = 3,
    GENERATION_RESPONSE = 4,

    TERMINATION = 5,
};

struct RequestsData
{
    std::vector<RequestWithId> requests;
};

static std::vector<char> serializeResponseWithIds(std::vector<ResponseWithId> const& responseWithIds)
{
    size_t totalSize = 0;
    totalSize += sizeof(size_t);
    for (auto const& responseWithId : responseWithIds)
    {
        totalSize += su::serializedSize(responseWithId.gid);
        totalSize += su::serializedSize(responseWithId.response);
    }

    std::vector<char> buffer(totalSize);
    std::stringbuf strbuf{std::ios_base::out | std::ios_base::in};
    strbuf.pubsetbuf(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    std::ostream ostream{&strbuf};

    su::serialize(responseWithIds.size(), ostream);
    for (auto const& responseWithId : responseWithIds)
    {
        su::serialize(responseWithId.gid, ostream);
        su::serialize(responseWithId.response, ostream);
    }
    return buffer;
}

static std::vector<ResponseWithId> deserializeResponseWithIds(std::vector<char>& buffer)
{
    std::vector<ResponseWithId> responseWithIds;
    su::VectorWrapBuf<char> strbuf{buffer};
    std::istream istream{&strbuf};
    auto numReq = su::deserialize<std::int64_t>(istream);
    for (int64_t req = 0; req < numReq; ++req)
    {
        auto const id = su::deserialize<std::uint64_t>(istream);
        responseWithIds.emplace_back(ResponseWithId{Serialization::deserializeResponse(istream), id});
    }
    return responseWithIds;
}

struct ResponsesData
{
    std::vector<ResponseWithId> response;
};

using MessageData = std::variant<RequestsData, ResponsesData>;

struct Message
{

    MessageID id;

    MessageData data;
};

class MessageQueue
{
public:
    void push(Message&& message)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mQueue.push(std::move(message));
        mCv.notify_one();
    }

    Message pop()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mCv.wait(lock, [this] { return !mQueue.empty(); });
        Message message = std::move(mQueue.front());
        mQueue.pop();
        return message;
    }

private:
    std::queue<Message> mQueue;
    std::mutex mMutex;
    std::condition_variable mCv;
};

class DisaggExecutorLeader
{
public:
    DisaggExecutorLeader(std::filesystem::path const& modelPath, ModelType modelType,
        ExecutorConfig const& executorConfig, bool isController, bool isContext, int numRequests,
        std::vector<int>& participatIds, std::vector<int> const& participantDeviceIdsThisInstance, int worldRank)
        : mIsContext(isContext)
        , mNumRequests(numRequests)
        , mWorldRanksInstances(participatIds)
        , mDeviceIdsThisInstance(participantDeviceIdsThisInstance)
        , mWorldRank(worldRank)
        , mIsLeaderInstance(false)
        , mShutdown(false)
        , mWorldComm(tensorrt_llm::mpi::MpiComm::world())

    {

#if ENABLE_MULTI_DEVICE

        auto world_size = mWorldComm.getSize();
        mRolesPerRank.resize(world_size);

        if (!mWorldRanksInstances.empty())
        {
            mIsLeaderInstance = mWorldRank == mWorldRanksInstances.front();
        };

        bool needExecutor = (std::find(mWorldRanksInstances.begin(), mWorldRanksInstances.end(), worldRank)
            != mWorldRanksInstances.end());
        if (needExecutor)
        {
            ExecutorConfig executorConfigC = executorConfig;

            auto parallelConfig = executorConfigC.getParallelConfig().value_or(ParallelConfig{});
            std::vector<int> participantIds = mWorldRanksInstances;

            parallelConfig.setParticipantIds(participantIds);
            TLLM_CHECK(parallelConfig.getCommunicationMode() == tensorrt_llm::executor::CommunicationMode::kLEADER);
            parallelConfig.setCommunicationType(tensorrt_llm::executor::CommunicationType::kMPI);
            parallelConfig.setDeviceIds(mDeviceIdsThisInstance);
            executorConfigC.setParallelConfig(parallelConfig);

            mExecutor = std::make_unique<Executor>(modelPath, modelType, executorConfigC);
        }

        mIsController = false;
        uint32_t role = 0;
        if (mIsLeaderInstance)
        {
            role |= 0b001;
        }
        if (mIsContext)
        {
            role |= 0b010;
        }
        if (isController)
        {
            mIsController = true;
            role |= 0b100;
        }

        TLLM_CHECK(mWorldRanksInstances.size() == mDeviceIdsThisInstance.size());

        mWorldComm.allgather(&role, mRolesPerRank.data(), 1, tensorrt_llm::mpi::MpiType::kUINT32);

        generatedRoles();

        if (mIsController)
        {
            mControllerSendThread = std::thread(&DisaggExecutorLeader::ControllerSendThread, this);
            mControllerRecvThread = std::thread(&DisaggExecutorLeader::ControllerRecvThread, this);
        }
        if (mIsLeaderInstance)
        {
            mInstanceRecvThread = std::thread(&DisaggExecutorLeader::InstanceLeaderRecvThread, this);
            mInstanceSendThread = std::thread(&DisaggExecutorLeader::InstanceLeaderSendThread, this);
            mInstanceLoopThread = std::thread(&DisaggExecutorLeader::InstanceLeaderLoopThread, this);
        }
#else
        TLLM_THROW("DisaggExecutor only support being compiled with ENABLE_MULTI_DEVICE");

#endif
    }

    bool isController() const
    {
        return mIsController;
    }

    std::vector<IdType> enqueueRequests(std::vector<Request> const& llmRequests)

    {
        if (!mIsController)
        {
            return {};
        }

        std::vector<RequestWithId> requestWithIds;
        std::vector<IdType> reqIds;
        for (auto const& req : llmRequests)
        {
            IdType id = generatedControlId();
            reqIds.push_back(id);

            RequestWithId reqWithId{req, id};
            reqWithId.req.setRequestType(RequestType::REQUEST_TYPE_CONTEXT_ONLY);

            requestWithIds.push_back(std::move(reqWithId));

            mRequestMap.insert(std::make_pair(id, req));
        }

        Message message{MessageID::PENDING_CONTEXT_REQUEST, MessageData{RequestsData{requestWithIds}}};

        mControllerSendQueue.push(std::move(message));

        return reqIds;
    }

    std::vector<Response> awaitResponses(std::optional<std::chrono::milliseconds> const& timeout)
    {
        // wait for responseQueue , modify reqid-
        std::vector<Response> responses;
        std::unique_lock<std::mutex> lck(mResponsesMtx);
        auto pred = [&mShutdown = mShutdown, &resp = this->mResponses]() -> bool { return !resp.empty() || mShutdown; };
        auto storeResponses = [this, &resp = this->mResponses, &responses]()
        {
            for (auto it = resp.cbegin(); it != resp.cend();)
            {
                responses.insert(responses.end(), it->second.begin(), it->second.end());
                resp.erase(it++);
            }
        };

        if (timeout)
        {
            if (mResponsesCv.wait_for(lck, timeout.value(), pred))
            {
                storeResponses();
            }
        }
        else
        {
            mResponsesCv.wait(lck, pred);
            storeResponses();
        }
        return responses;
    }

    std::deque<RequestStatsPerIteration> getLatestRequestStats()
    {
        if (mExecutor && mExecutor->canEnqueueRequests())
        {
            return mExecutor->getLatestRequestStats();
        }
        return {};
    }

    bool isContextRank() const
    {
        return mIsContext;
    }

    bool isGenerationRank() const
    {
        return !mIsContext;
    }

    void shutDown()
    {
        if (mShutdown)
        {
            return;
        }

        if (mIsController)
        {
            std::call_once(mHasSendTerminFlag,
                [&]()
                {
                    MessageID terminationMessage = MessageID::TERMINATION;
                    for (auto&& leaderRanks : {mContextLeaderRanks, mGenerationLeaderRanks})
                    {
                        for (auto&& leaderRank : leaderRanks)
                        {

                            mWorldComm.send(&terminationMessage, 1, tensorrt_llm::mpi::MpiType::kUINT64, leaderRank,
                                kM_CONTROLLER_ID_TAG);
                        }
                    }

                    mWorldComm.send(&terminationMessage, 1, tensorrt_llm::mpi::MpiType::kUINT64, mControllerRank,
                        kM_INSTANCE_ID_TAG);
                });
            // end recv thread;
        }
        mShutdown = true;

        // end send thread
        if (mIsController)
        {
            mControllerSendQueue.push({MessageID::TERMINATION, {}});
        }
        mInstanceSendQueue.push({MessageID::TERMINATION, {}});
    }

    ~DisaggExecutorLeader()
    {

        if (mIsController)
        {
            shutDown();
        }

        if (mIsLeaderInstance)
        {
            if (mInstanceSendThread.joinable())
            {
                mInstanceSendThread.join();
            }
            if (mInstanceRecvThread.joinable())
            {
                mInstanceRecvThread.join();
            }
            if (mInstanceLoopThread.joinable())
            {
                mInstanceLoopThread.join();
            }
        }

        if (mIsController)
        {
            if (mControllerSendThread.joinable())
            {
                mControllerSendThread.join();
            }
            if (mControllerRecvThread.joinable())
            {
                mControllerRecvThread.join();
            }
        }

        if (!mIsController)
        {
            mExecutor->shutdown();
        }
        if (mIsController && mIsLeaderInstance)
        {
            mExecutor->shutdown();
        }

        shutDown();
    }

private:
    bool mIsContext;
    tensorrt_llm::mpi::MpiComm const& mWorldComm;
    std::unique_ptr<Executor> mExecutor;
    std::thread mInstanceSendThread;
    std::thread mInstanceRecvThread;
    std::thread mInstanceLoopThread;
    std::thread mControllerSendThread;
    std::thread mControllerRecvThread;
    int mNumRequests;
    std::map<std::uint64_t, Request> mRequestMap;
    std::map<IdType, DataTransceiverState> mGenIdToContextPhase;
    std::unordered_map<IdType, IdType> mInstanceIdToGlobalId;
    std::mutex mIdToGlbalMutex;

    std::vector<int> mWorldRanksInstances;

    int mWorldRank;
    int mControllerRank = 0;
    bool mIsController;
    bool mIsLeaderInstance;
    std::vector<uint32_t> mRolesPerRank;
    std::vector<int> mContextLeaderRanks;
    std::vector<int> mGenerationLeaderRanks;

    IdType mLastId = 1;
    MessageQueue mControllerSendQueue;
    MessageQueue mInstanceSendQueue;

    std::atomic<bool> mShutdown;

    // Ready responses
    std::unordered_map<IdType, std::vector<Response>> mResponses;
    mutable std::mutex mResponsesMtx;
    std::condition_variable mResponsesCv;

    std::vector<int> mDeviceIdsThisInstance;
    std::once_flag mHasSendTerminFlag;

    void appendNewResponses(std::vector<ResponseWithId>& newResponses)
    {
        {
            std::scoped_lock<std::mutex> lck(mResponsesMtx);
            for (auto& responseWithId : newResponses)
            {
                // global id to Result
                responseWithId.response = Response(responseWithId.gid, responseWithId.response.getResult());

                mResponses[responseWithId.gid].emplace_back(responseWithId.response);
            }
        }
        mResponsesCv.notify_all();
    }

    void generatedRoles()
    {
        int contextNum = 0;
        int genrationNum = 0;
        int controllerNum = 0;
        for (int rank = 0; rank < mRolesPerRank.size(); rank++)
        {
            uint32_t role = mRolesPerRank[rank];
            if ((role & 0b001) != 0u)
            {
                if ((role & 0b010) != 0u)
                {
                    contextNum++;
                    mContextLeaderRanks.push_back(rank);
                }
                else
                {
                    genrationNum++;
                    mGenerationLeaderRanks.push_back(rank);
                }
            }
            if ((role & 0b100) != 0u)
            {
                controllerNum++;
                mControllerRank = rank;
            }
        }
        TLLM_CHECK_WITH_INFO(controllerNum == 1, "only one rank is controller but get %d controllerNum", controllerNum);
    }

    IdType generatedControlId()
    {
        return (mLastId++ % UINT64_MAX);
    }

    int selectContextLeaderRank()
    {
        static int leaderRank = 0;
        leaderRank = (leaderRank + 1) % mContextLeaderRanks.size();
        return mContextLeaderRanks[leaderRank];
    }

    int selectGenerationLeaderRank()
    {

        // TODO: for same reqId , need select specific generationLeader
        static int leaderRank = 0;
        leaderRank = (leaderRank + 1) % mGenerationLeaderRanks.size();
        return mGenerationLeaderRanks[leaderRank];
    }

    void ControllerSendThread()
    {
        // send request to context reqid
        // and send context pahse to generation

        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));
        tensorrt_llm::common::setThreadName("ControllerSendThread");

        while (!mShutdown)
        {
            auto message = mControllerSendQueue.pop();
            if (message.id == MessageID::TERMINATION)
            {

                TLLM_LOG_DEBUG("controller get termination message in sendQueue");
                break;
            }
            if (message.id == MessageID::PENDING_CONTEXT_REQUEST)
            {

                auto& reqWithIds = std::get<RequestsData>(message.data);
                auto packed = RequestWithId::serializeReqWithIds(reqWithIds.requests);
                int contextRank = selectContextLeaderRank();

                mWorldComm.send(&message.id, 1, tensorrt_llm::mpi::MpiType::kUINT64, contextRank, kM_CONTROLLER_ID_TAG);

                mWorldComm.send(packed.data(), packed.size(), tensorrt_llm::mpi::MpiType::kCHAR, contextRank,
                    kM_CONTROLLER_DATA_TAG);
            }
            else if (message.id == MessageID::PENDING_GENERATION_REQUEST)
            {

                auto& reqWithIds = std::get<RequestsData>(message.data);
                auto packed = RequestWithId::serializeReqWithIds(reqWithIds.requests);
                int generationRank = selectGenerationLeaderRank();

                mWorldComm.send(
                    &message.id, 1, tensorrt_llm::mpi::MpiType::kUINT64, generationRank, kM_CONTROLLER_ID_TAG);

                mWorldComm.send(packed.data(), packed.size(), tensorrt_llm::mpi::MpiType::kCHAR, generationRank,
                    kM_CONTROLLER_DATA_TAG);
            }
            else
            {
                TLLM_THROW("rank:%d, size:%d controller send Invalid message id:%ld", mWorldComm.getRank(),
                    mWorldComm.getSize(), static_cast<uint64_t>(message.id));
            }
        }
    }

    void ControllerRecvThread()
    {
#if ENABLE_MULTI_DEVICE
        tensorrt_llm::common::setThreadName("ControllerRecvThread");

        // recv response from context and push to sendQueue
        // recv response from generation and push to responseQueue and notify awaitResponse
        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));

        while (!mShutdown)
        {

            MPI_Message msg = nullptr;
            MPI_Status status;

            mWorldComm.mprobe(MPI_ANY_SOURCE, kM_INSTANCE_ID_TAG, &msg, &status);

            auto sourceRank{status.MPI_SOURCE};
            int32_t count = 0;
            MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
            TLLM_CHECK(count == 1);

            MessageID messageId;
            MPICHECK(MPI_Mrecv(&messageId, count, MPI_UINT64_T, &msg, &status));

            if (messageId == MessageID::TERMINATION)
            {
                TLLM_LOG_DEBUG("controller received termination message***************\n");
                break;
            }
            if (messageId == MessageID::CONTEXT_RESPONSE)
            {
                mWorldComm.mprobe(sourceRank, kM_INSTANCE_DATA_TAG, &msg, &status);
                MPICHECK(MPI_Get_count(&status, MPI_CHAR, &count));
                std::vector<char> buffer(count);
                MPICHECK(MPI_Mrecv(buffer.data(), count, MPI_CHAR, &msg, &status));
                auto responseWithIds = deserializeResponseWithIds(buffer);
                //  enqueueTo sendQueue like enqueuRequest. . modify requestType and set ContextPhaseParams
                //  and push to sendQueue.
                std::vector<RequestWithId> requestWithIds;
                for (auto&& responseWithId : responseWithIds)
                {
                    auto reqId = responseWithId.gid;
                    auto& request = mRequestMap.at(reqId);

                    request.setRequestType(RequestType::REQUEST_TYPE_GENERATION_ONLY);
                    request.setContextPhaseParams(responseWithId.response.getResult().contextPhaseParams.value());
                    requestWithIds.push_back(RequestWithId{request, reqId});
                }
                mControllerSendQueue.push({MessageID::PENDING_GENERATION_REQUEST, RequestsData{requestWithIds}});
            }

            else if (messageId == MessageID::GENERATION_RESPONSE)
            {

                mWorldComm.mprobe(sourceRank, kM_INSTANCE_DATA_TAG, &msg, &status);
                MPICHECK(MPI_Get_count(&status, MPI_CHAR, &count));
                std::vector<char> buffer(count);
                MPICHECK(MPI_Mrecv(buffer.data(), count, MPI_CHAR, &msg, &status));

                auto responseWithIds = deserializeResponseWithIds(buffer);
                appendNewResponses(responseWithIds);
            }
            else
            {
                TLLM_THROW("rank:%d, size:%d controller recv Invalid message id:%ld", mWorldComm.getRank(),
                    mWorldComm.getSize(), static_cast<uint64_t>(messageId));
            }
        }
#endif
    }

    void InstanceLeaderSendThread()
    {
        tensorrt_llm::common::setThreadName("InstanceLeaderSendThread");

        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));

        // pop senQueue and send response to controller

        while (!mShutdown)
        {
            auto message = mInstanceSendQueue.pop();
            if (message.id == MessageID::CONTEXT_RESPONSE || message.id == MessageID::GENERATION_RESPONSE)
            {
                auto& responseWithIds = std::get<ResponsesData>(message.data);
                auto packed = serializeResponseWithIds(responseWithIds.response);

                mWorldComm.send(
                    &message.id, 1, tensorrt_llm::mpi::MpiType::kUINT64, mControllerRank, kM_INSTANCE_ID_TAG);
                mWorldComm.send(packed.data(), packed.size(), tensorrt_llm::mpi::MpiType::kCHAR, mControllerRank,
                    kM_INSTANCE_DATA_TAG);
            }
            else if (message.id == MessageID::TERMINATION)
            {
                // break; no send
                TLLM_LOG_DEBUG(
                    "ranK:%d ,size:%d ,isContext:%d... Context or Generation leader get termination message in "
                    "sendQueue***************\n",
                    mWorldComm.getRank(), mWorldComm.getSize(), int(mIsContext));
                break;
            }
            else
            {
                TLLM_THROW("rank:%d, size:%d InstanceLeaderSendThread send Invalid message id:%ld",
                    mWorldComm.getRank(), mWorldComm.getSize(), static_cast<uint64_t>(message.id));
            }
        }
    }

    void InstanceLeaderRecvThread()
    {

#if ENABLE_MULTI_DEVICE
        tensorrt_llm::common::setThreadName("InstanceLeaderRecvThread");

        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));

        // recv request from controller and enqueRequest to executor
        while (!mShutdown)
        {
            MPI_Message msg;
            MPI_Status status;
            auto sourceRank{mControllerRank};
            mWorldComm.mprobe(sourceRank, kM_CONTROLLER_ID_TAG, &msg, &status);

            int32_t count;
            MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
            TLLM_CHECK(count == 1);

            MessageID messageId;
            MPICHECK(MPI_Mrecv(&messageId, count, MPI_UINT64_T, &msg, &status));

            if (messageId == MessageID::TERMINATION)
            {
                TLLM_LOG_DEBUG(
                    "ranK:%d ,size:%d ,isContext:%d ... Context or Generation leader revb termination message in "
                    "InstanceLeaderRecvThread***************\n",
                    mWorldComm.getRank(), mWorldComm.getSize(), int(mIsContext));
                shutDown();
                break;
            }
            if (messageId == MessageID::PENDING_CONTEXT_REQUEST || messageId == MessageID::PENDING_GENERATION_REQUEST)
            {
                mWorldComm.mprobe(sourceRank, kM_CONTROLLER_DATA_TAG, &msg, &status);
                MPICHECK(MPI_Get_count(&status, MPI_CHAR, &count));
                std::vector<char> buffer(count);
                MPICHECK(MPI_Mrecv(buffer.data(), count, MPI_CHAR, &msg, &status));
                auto requestWithIds = RequestWithId::deserializeReqWithIds(buffer);
                for (auto&& requestWithId : requestWithIds)
                {

                    auto globalReqId = requestWithId.id;
                    if (mIsContext)
                    {
                        TLLM_CHECK(messageId == MessageID::PENDING_CONTEXT_REQUEST);
                        TLLM_CHECK(requestWithId.req.getRequestType() == RequestType::REQUEST_TYPE_CONTEXT_ONLY);
                    }
                    if (!mIsContext)
                    {
                        TLLM_CHECK(messageId == MessageID::PENDING_GENERATION_REQUEST);
                        TLLM_CHECK(requestWithId.req.getRequestType() == RequestType::REQUEST_TYPE_GENERATION_ONLY);
                    }
                    auto reqId = mExecutor->enqueueRequest(requestWithId.req);
                    {
                        std::scoped_lock<std::mutex> lock{mIdToGlbalMutex};
                        mInstanceIdToGlobalId[reqId] = globalReqId;
                    }
                }
            }
            else
            {
                TLLM_THROW("rank:%d, size:%d InstanceLeaderRecvThread send Invalid message id:%ld",
                    mWorldComm.getRank(), mWorldComm.getSize(), static_cast<uint64_t>(messageId));
            }
        }
#endif
    }

    void InstanceLeaderLoopThread()
    {

        tensorrt_llm::common::setThreadName("InstanceLeaderLoopThread");

        TLLM_CUDA_CHECK(
            cudaSetDevice(mDeviceIdsThisInstance.at(COMM_SESSION.getRank() % (mDeviceIdsThisInstance.size()))));

        // loop awaitResponse and enqueue into sendQueue
        while (!mShutdown)
        {
            std::chrono::milliseconds waitTime(1);

            auto responses = mExecutor->awaitResponses(waitTime);
            if (responses.empty())
            {
                continue;
            }
            std::vector<ResponseWithId> responseWithIds;
            for (auto&& response : responses)
            {
                auto reqId = response.getRequestId();
                IdType globalId{0};
                {
                    std::scoped_lock<std::mutex> lock{mIdToGlbalMutex};
                    globalId = mInstanceIdToGlobalId[reqId];
                }
                TLLM_CHECK(globalId != 0);
                responseWithIds.push_back(ResponseWithId{response, globalId});
            }

            if (mIsContext)
            {
                mInstanceSendQueue.push({MessageID::CONTEXT_RESPONSE, ResponsesData{responseWithIds}});
            }
            if ((!mIsContext))
            {
                mInstanceSendQueue.push({MessageID::GENERATION_RESPONSE, ResponsesData{responseWithIds}});
            }
        }
    }
};
} // namespace tensorrt_llm::testing::disaggexecutor
