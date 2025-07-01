/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 */

#include "tensorrt_llm/batch_manager/metaTransceiver.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include "gtest/gtest.h"

#include <chrono>
#include <memory>
#include <string>
#include <thread>

using namespace tensorrt_llm::batch_manager;

// ---------------------------------------
//     RealMetaTransceiverTest
// ---------------------------------------

class RealMetaTransceiverTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override {}

    size_t setUpCommunicator()
    {
        // Initialize MPI for multi-process testing
        tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);
        mComm = std::addressof(tensorrt_llm::mpi::MpiComm::world());
        mWorldSize = mComm->getSize();
        mRank = mComm->getRank();

        // Determine if this process is sender or receiver
        isSender = mRank % 2 == 0; // Even ranks are senders

        return mWorldSize;
    }

    void setUpMetaTransceiver()
    {
        // Set up ZMQ endpoints based on rank
        if (isSender)
        {
            // Sender process - create MetaTransceiver and connect to receivers
            mTransceiver = std::make_unique<MetaTransceiver>(false);
            mTransceiver->connect("tcp://10.78.7.63:5555");
        }
        else
        {
            // Receiver process - create MetaTransceiver and bind to port
            mTransceiver = std::make_unique<MetaTransceiver>(true, "tcp://10.78.7.63:5555");
        }
    }

    std::string createPlainTestMessage(std::string const& messageType, int messageId)
    {
        return "{\"type\":\"" + messageType + "\",\"id\":" + std::to_string(messageId) + "}";
    }

    RequestInfo createRequestInfo()
    {
        auto state = std::make_unique<tensorrt_llm::executor::DataTransceiverState>();
        state->setCommState(tensorrt_llm::executor::kv_cache::CommState{12, "127.0.0.1"});
        state->setCacheState(
            tensorrt_llm::executor::kv_cache::CacheState{10, 12, 128, 128, 8, 8, nvinfer1::DataType::kFLOAT});
        return RequestInfo{1, *state};
    }

    bool isSender{false};
    tensorrt_llm::mpi::MpiComm const* mComm;
    size_t mWorldSize{0};
    int mRank{0};
    std::unique_ptr<MetaTransceiver> mTransceiver;
};

TEST_F(RealMetaTransceiverTest, BasicSendReceive)
{
    auto worldSize = setUpCommunicator();
    if (worldSize != 2)
    {
        GTEST_SKIP() << "mpirun with 2 processes is required to run this test.";
    }

    setUpMetaTransceiver();

    // Synchronize processes before starting communication
    mComm->barrier();

    std::string testMessage = createPlainTestMessage("test_request", 1);

    if (isSender)
    {
        // Wait for receiver to be ready
        std::this_thread::sleep_for(std::chrono::seconds(5));

        // Sender process (rank 0)
        TLLM_LOG_INFO("Sender (rank %d): Sending message to %s", mRank, mTransceiver->getEndpoint().c_str());

        // Send message to receiver
        mTransceiver->send(testMessage.c_str(), testMessage.size());
    }
    else
    {
        // Receiver process (rank 1)
        TLLM_LOG_INFO("Receiver (rank %d): Waiting for message", mRank);

        // Wait for message from sender
        std::string receivedMessage(testMessage.size(), '\0');
        mTransceiver->recv(receivedMessage.data(), testMessage.size());

        // Verify received message
        EXPECT_TRUE(testMessage == receivedMessage);
    }

    mComm->barrier();
}

TEST_F(RealMetaTransceiverTest, SerializedSendReceive)
{
    auto worldSize = setUpCommunicator();
    if (worldSize != 2)
    {
        GTEST_SKIP() << "mpirun with 2 processes is required to run this test.";
    }

    setUpMetaTransceiver();

    // Synchronize processes before starting communication
    mComm->barrier();

    RequestInfo testMessage = createRequestInfo();

    if (isSender)
    {
        // Wait for receiver to be ready
        std::this_thread::sleep_for(std::chrono::seconds(5));

        // Sender process (rank 0)
        TLLM_LOG_INFO("Sender (rank %d): Sending message to %s", mRank, mTransceiver->getEndpoint().c_str());

        // Send message to receiver
        std::ostringstream oss;
        RequestInfo::serialize(testMessage, oss);
        auto const& serializedInfo = oss.str();
        mTransceiver->send(serializedInfo.c_str(), serializedInfo.size());
    }
    else
    {
        // Receiver process (rank 1)
        TLLM_LOG_INFO("Receiver (rank %d): Waiting for message", mRank);

        // Wait for message from sender
        std::ostringstream oss;
        RequestInfo::serialize(testMessage, oss);
        size_t messageSize = RequestInfo::serializedSize(testMessage);

        std::string serializedInfo;
        serializedInfo.resize(messageSize);
        mTransceiver->recv(serializedInfo.data(), messageSize);
        std::istringstream iss(serializedInfo);
        auto receivedMessage = RequestInfo::deserialize(iss);

        // Verify received message
        EXPECT_TRUE(testMessage == receivedMessage);
    }

    mComm->barrier();
}

// TEST_F(RealMetaTransceiverTest, AsyncCommunication)
// {
//     auto worldSize = setUpCommunicator();
//     if (worldSize != 2)
//     {
//         GTEST_SKIP() << "mpirun with 2 processes is required to run this test.";
//     }

//     setUpMetaTransceiver();
//     mComm->barrier();

//     if (isSender)
//     {
//         // Launch async send operations
//         std::string executorId = "executor_1";
//         std::vector<std::future<void>> sendFutures;
//         std::vector<std::future<std::string>> receiveFutures;

//         for (int i = 0; i < 3; ++i)
//         {
//             // Async send
//             sendFutures.push_back(std::async(std::launch::async, [this, executorId, i]()
//             {
//                 std::string message = createTestMessage("async_request", i);
//                 mTransceiver->send(executorId, message);
//             }));

//             // Async receive
//             receiveFutures.push_back(std::async(std::launch::async, [this, executorId]() -> std::string
//             {
//                 std::string response;
//                 return mTransceiver->receive(executorId, response);
//             }));
//         }

//         // Wait for all operations to complete
//         for (auto& future : sendFutures)
//         {
//             future.get();
//         }

//         for (int i = 0; i < 3; ++i)
//         {
//             auto response = receiveFutures[i].get();
//             EXPECT_TRUE(response.find("async_response") != std::string::npos);
//         }
//     }
//     else
//     {
//         // Handle async requests
//         for (int i = 0; i < 3; ++i)
//         {
//             std::string receivedMessage;
//             auto message = mTransceiver->receive("", receivedMessage);

//             EXPECT_TRUE(message.find("async_request") != std::string::npos);

//             // Send async response
//             std::string responseMessage = createTestMessage("async_response", i);
//             std::string senderExecutorId = "executor_0";
//             mTransceiver->send(senderExecutorId, responseMessage);
//         }
//     }

//     mComm->barrier();
// }

// // Test with 4 processes (2 senders, 2 receivers)
// TEST_F(RealMetaTransceiverTest, MultiProcessCommunication)
// {
//     auto worldSize = setUpCommunicator();
//     if (worldSize != 4)
//     {
//         GTEST_SKIP() << "mpirun with 4 processes is required to run this test.";
//     }

//     setUpMetaTransceiver();
//     mComm->barrier();

//     if (isSender)
//     {
//         // Sender processes (rank 0, 2)
//         int receiverRank = mRank + 1;  // Send to next rank
//         std::string executorId = "executor_" + std::to_string(receiverRank);
//         std::string testMessage = createTestMessage("multi_request", mRank);

//         mTransceiver->send(executorId, testMessage);

//         std::string response;
//         auto receivedMessage = mTransceiver->receive(executorId, response);

//         EXPECT_TRUE(receivedMessage.find("multi_response") != std::string::npos);
//         EXPECT_TRUE(receivedMessage.find("sender_rank\":" + std::to_string(receiverRank)) != std::string::npos);
//     }
//     else
//     {
//         // Receiver processes (rank 1, 3)
//         std::string receivedMessage;
//         auto message = mTransceiver->receive("", receivedMessage);

//         EXPECT_TRUE(message.find("multi_request") != std::string::npos);

//         // Send response back to sender
//         int senderRank = mRank - 1;  // Send to previous rank
//         std::string senderExecutorId = "executor_" + std::to_string(senderRank);
//         std::string responseMessage = createTestMessage("multi_response", mRank);
//         mTransceiver->send(senderExecutorId, responseMessage);
//     }

//     mComm->barrier();
// }
