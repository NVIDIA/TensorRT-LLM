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
#include "tensorrt_llm/common/stringUtils.h"
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
    PENDING_FULL_REQUEST = 3,
    CONTEXT_RESPONSE = 4,
    GENERATION_RESPONSE = 5,

    TERMINATION = 6,
};

enum DisaggRole : uint32_t
{
    DISAGG_CONTEXT = 1,
    DISAGG_GENERATION = 2,
    DISAGG_MIXED = DISAGG_CONTEXT | DISAGG_GENERATION,
    DISAGG_LEADER = 4,
    DISAGG_CONTROLLER = 8,
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
        ExecutorConfig const& executorConfig, bool isController, bool isContext, bool isGeneration, int numRequests,
        std::vector<int>& participatIds, std::vector<int> const& participantDeviceIdsThisInstance, int worldRank)
        : mNumRequests(numRequests)
        , mWorldRanksInstances(participatIds)
        , mDeviceIdsThisInstance(participantDeviceIdsThisInstance)
        , mWorldRank(worldRank)
        , mShutdown(false)
        , mWorldComm(tensorrt_llm::mpi::MpiComm::world())

    {

#if ENABLE_MULTI_DEVICE

        auto world_size = mWorldComm.getSize();
        mRolesPerRank.resize(world_size);

        if (isContext)
        {
            mRole |= DisaggRole::DISAGG_CONTEXT;
        }
        if (isGeneration)
        {
            mRole |= DisaggRole::DISAGG_GENERATION;
        }

        if (!mWorldRanksInstances.empty() && mWorldRank == mWorldRanksInstances.front())
        {
            mRole |= DisaggRole::DISAGG_LEADER;
        }

        if (isController)
        {
            mRole |= DisaggRole::DISAGG_CONTROLLER;
        }

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

        TLLM_CHECK(mWorldRanksInstances.size() == mDeviceIdsThisInstance.size());

        mWorldComm.allgather(&mRole, mRolesPerRank.data(), 1, tensorrt_llm::mpi::MpiType::kUINT32);

        generateRoles();

        if (isController)
        {
            mControllerSendThread = std::thread(&DisaggExecutorLeader::ControllerSendThread, this);
            mControllerRecvThread = std::thread(&DisaggExecutorLeader::ControllerRecvThread, this);
        }
        if (isLeaderInstance())
        {
            mInstanceRecvThread = std::thread(&DisaggExecutorLeader::InstanceLeaderRecvThread, this);
            mInstanceSendThread = std::thread(&DisaggExecutorLeader::InstanceLeaderSendThread, this);
            mInstanceLoopThread = std::thread(&DisaggExecutorLeader::InstanceLeaderLoopThread, this);
        }
#else
        TLLM_THROW("DisaggExecutor only support being compiled with ENABLE_MULTI_DEVICE");

#endif
    }

    bool isControllerRank() const
    {
        return mRole & DISAGG_CONTROLLER;
    }

    bool isContextRank() const
    {
        return mRole & DISAGG_CONTEXT;
    }

    bool isGenerationRank() const
    {
        return mRole & DISAGG_GENERATION;
    }

    bool isLeaderInstance() const
    {
        return mRole & DISAGG_LEADER;
    }

    std::vector<IdType> enqueueRequests(std::vector<Request> const& llmRequests)

    {
        if (!isControllerRank())
        {
            return {};
        }

        std::vector<RequestWithId> requestWithIds;
        std::vector<RequestWithId> requestWithIdsFull; // full request, not disaggregated
        std::vector<IdType> reqIds;
        for (auto const& req : llmRequests)
        {
            IdType id = generatedControlId();
            reqIds.push_back(id);

            RequestWithId reqWithId{req, id};
            if (req.getRequestType() == RequestType::REQUEST_TYPE_CONTEXT_ONLY)
            {
                requestWithIds.push_back(std::move(reqWithId));
            }
            else
            {
                TLLM_CHECK(req.getRequestType() == RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION);
                requestWithIdsFull.push_back(std::move(reqWithId));
            }

            mRequestMap.insert(std::make_pair(id, req));
        }

        if (!requestWithIds.empty())
        {
            Message message{MessageID::PENDING_CONTEXT_REQUEST, MessageData{RequestsData{requestWithIds}}};
            mControllerSendQueue.push(std::move(message));
        }
        if (!requestWithIdsFull.empty())
        {
            Message message{MessageID::PENDING_FULL_REQUEST, MessageData{RequestsData{requestWithIdsFull}}};
            mControllerSendQueue.push(std::move(message));
        }

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

    void shutDown()
    {
        if (mShutdown)
        {
            return;
        }

        if (isControllerRank())
        {
            std::call_once(mHasSendTerminFlag,
                [&]()
                {
                    MessageID terminationMessage = MessageID::TERMINATION;
                    std::vector<bool> isSend(mWorldComm.getSize(), false);
                    for (auto&& leaderRanks : {mContextLeaderRanks, mGenerationLeaderRanks})
                    {
                        for (auto&& leaderRank : leaderRanks)
                        {
                            if (isSend[leaderRank])
                            {
                                continue;
                            }
                            mWorldComm.sendRawTag(&terminationMessage, 1, tensorrt_llm::mpi::MpiType::kUINT64,
                                leaderRank, kM_CONTROLLER_ID_TAG);
                            isSend[leaderRank] = true;
                        }
                    }

                    mWorldComm.sendRawTag(&terminationMessage, 1, tensorrt_llm::mpi::MpiType::kUINT64, mControllerRank,
                        kM_INSTANCE_ID_TAG);
                });
            // end recv thread;
        }
        mShutdown = true;

        // end send thread
        if (isControllerRank())
        {
            mControllerSendQueue.push({MessageID::TERMINATION, {}});
        }
        mInstanceSendQueue.push({MessageID::TERMINATION, {}});
    }

    ~DisaggExecutorLeader()
    {

        if (isControllerRank())
        {
            shutDown();
        }

        if (isLeaderInstance())
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

        if (isControllerRank())
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

        if (!isControllerRank())
        {
            mExecutor->shutdown();
        }
        if (isControllerRank() && isLeaderInstance())
        {
            mExecutor->shutdown();
        }

        shutDown();
    }

private:
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
    uint32_t mRole = 0;
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

    void generateRoles()
    {
        int contextNum = 0;
        int genrationNum = 0;
        int controllerNum = 0;
        for (int rank = 0; rank < mRolesPerRank.size(); rank++)
        {
            uint32_t role = mRolesPerRank[rank];
            if (role & DISAGG_LEADER)
            {
                if (role & DISAGG_CONTEXT)
                {
                    contextNum++;
                    mContextLeaderRanks.push_back(rank);
                }
                if (role & DISAGG_GENERATION)
                {
                    genrationNum++;
                    mGenerationLeaderRanks.push_back(rank);
                }
            }
            if (role & DISAGG_CONTROLLER)
            {
                controllerNum++;
                mControllerRank = rank;
            }
        }
        TLLM_CHECK_WITH_INFO(controllerNum == 1, "only one rank is controller but get %d controllerNum", controllerNum);
        TLLM_LOG_INFO("leader ctx: %s, gen: %s", common::vec2str(mContextLeaderRanks).c_str(),
            common::vec2str(mGenerationLeaderRanks).c_str());
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

                mWorldComm.sendRawTag(
                    &message.id, 1, tensorrt_llm::mpi::MpiType::kUINT64, contextRank, kM_CONTROLLER_ID_TAG);

                mWorldComm.sendRawTag(packed.data(), packed.size(), tensorrt_llm::mpi::MpiType::kCHAR, contextRank,
                    kM_CONTROLLER_DATA_TAG);
            }
            else if (message.id == MessageID::PENDING_GENERATION_REQUEST
                || message.id == MessageID::PENDING_FULL_REQUEST)
            {

                auto& reqWithIds = std::get<RequestsData>(message.data);
                auto packed = RequestWithId::serializeReqWithIds(reqWithIds.requests);
                int generationRank = selectGenerationLeaderRank();

                mWorldComm.sendRawTag(
                    &message.id, 1, tensorrt_llm::mpi::MpiType::kUINT64, generationRank, kM_CONTROLLER_ID_TAG);

                mWorldComm.sendRawTag(packed.data(), packed.size(), tensorrt_llm::mpi::MpiType::kCHAR, generationRank,
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

            mWorldComm.mprobeRawTag(MPI_ANY_SOURCE, kM_INSTANCE_ID_TAG, &msg, &status);

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
                mWorldComm.mprobeRawTag(sourceRank, kM_INSTANCE_DATA_TAG, &msg, &status);
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

                mWorldComm.mprobeRawTag(sourceRank, kM_INSTANCE_DATA_TAG, &msg, &status);
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

                mWorldComm.sendRawTag(
                    &message.id, 1, tensorrt_llm::mpi::MpiType::kUINT64, mControllerRank, kM_INSTANCE_ID_TAG);
                mWorldComm.sendRawTag(packed.data(), packed.size(), tensorrt_llm::mpi::MpiType::kCHAR, mControllerRank,
                    kM_INSTANCE_DATA_TAG);
            }
            else if (message.id == MessageID::TERMINATION)
            {
                // break; no send
                TLLM_LOG_DEBUG(
                    "ranK:%d ,size:%d ,isContext:%d... Context or Generation leader get termination message in "
                    "sendQueue***************\n",
                    mWorldComm.getRank(), mWorldComm.getSize(), int(isContextRank()));
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
            mWorldComm.mprobeRawTag(sourceRank, kM_CONTROLLER_ID_TAG, &msg, &status);

            int32_t count;
            MPICHECK(MPI_Get_count(&status, MPI_UINT64_T, &count));
            TLLM_CHECK(count == 1);

            MessageID messageId;
            MPICHECK(MPI_Mrecv(&messageId, count, MPI_UINT64_T, &msg, &status));

            if (messageId == MessageID::TERMINATION)
            {
                TLLM_LOG_DEBUG(
                    "ranK:%d ,size:%d ,isContext:%d ... Context or Generation leader recv termination message in "
                    "InstanceLeaderRecvThread***************\n",
                    mWorldComm.getRank(), mWorldComm.getSize(), int(isContextRank()));
                shutDown();
                break;
            }
            if (messageId == MessageID::PENDING_CONTEXT_REQUEST || messageId == MessageID::PENDING_GENERATION_REQUEST
                || messageId == MessageID::PENDING_FULL_REQUEST)
            {
                mWorldComm.mprobeRawTag(sourceRank, kM_CONTROLLER_DATA_TAG, &msg, &status);
                MPICHECK(MPI_Get_count(&status, MPI_CHAR, &count));
                std::vector<char> buffer(count);
                MPICHECK(MPI_Mrecv(buffer.data(), count, MPI_CHAR, &msg, &status));
                auto requestWithIds = RequestWithId::deserializeReqWithIds(buffer);
                for (auto&& requestWithId : requestWithIds)
                {

                    auto globalReqId = requestWithId.id;
                    if (isContextRank() && messageId == MessageID::PENDING_CONTEXT_REQUEST)
                    {
                        TLLM_CHECK(requestWithId.req.getRequestType() == RequestType::REQUEST_TYPE_CONTEXT_ONLY);
                    }
                    else if (isGenerationRank()
                        && (messageId == MessageID::PENDING_GENERATION_REQUEST
                            || messageId == MessageID::PENDING_FULL_REQUEST))
                    {
                        if (messageId == MessageID::PENDING_GENERATION_REQUEST)
                        {
                            TLLM_CHECK(requestWithId.req.getRequestType() == RequestType::REQUEST_TYPE_GENERATION_ONLY);
                        }
                        else // PENDING_FULL_REQUEST
                        {
                            TLLM_CHECK(
                                requestWithId.req.getRequestType() == RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION);
                        }
                    }
                    else
                    {
                        TLLM_THROW("rank:%d, size:%d InstanceLeaderRecvThread recv Invalid message id:%ld",
                            mWorldComm.getRank(), mWorldComm.getSize(), static_cast<uint64_t>(messageId));
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
            std::vector<ResponseWithId> responseWithIdsContext;
            std::vector<ResponseWithId> responseWithIdsGeneration;
            for (auto&& response : responses)
            {
                auto reqId = response.getRequestId();
                IdType globalId{0};
                {
                    std::scoped_lock<std::mutex> lock{mIdToGlbalMutex};
                    globalId = mInstanceIdToGlobalId[reqId];
                }
                TLLM_CHECK(globalId != 0);
                auto const& result = response.getResult();
                if (result.contextPhaseParams.has_value())
                {
                    responseWithIdsContext.emplace_back(response, globalId);
                }
                else
                {
                    responseWithIdsGeneration.emplace_back(response, globalId);
                }
            }

            if (isContextRank())
            {
                mInstanceSendQueue.push({MessageID::CONTEXT_RESPONSE, ResponsesData{responseWithIdsContext}});
            }
            if (isGenerationRank())
            {
                mInstanceSendQueue.push({MessageID::GENERATION_RESPONSE, ResponsesData{responseWithIdsGeneration}});
            }
        }
    }
};
} // namespace tensorrt_llm::testing::disaggexecutor
