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

#define UCX_WRAPPER_LIB_NAME "tensorrt_llm_ucx_wrapper"

#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary(name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "gtest/gtest.h"
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <gmock/gmock.h>
#include <memory>
#include <random>
#include <tensorrt_llm/batch_manager/mlaCacheFormatter.h>
#include <tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h>

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager;
namespace texec = tensorrt_llm::executor;

namespace
{
std::mutex mDllMutex;

std::unique_ptr<texec::kv_cache::ConnectionManager> makeOneUcxConnectionManager()
{
    std::lock_guard<std::mutex> lock(mDllMutex);
    void* WrapperLibHandle{nullptr};
    WrapperLibHandle = dllOpen(UCX_WRAPPER_LIB_NAME);
    TLLM_CHECK_WITH_INFO(WrapperLibHandle != nullptr, "UCX wrapper library is not open correctly.");
    auto load_sym = [](void* handle, char const* name)
    {
        void* ret = dllGetSym(handle, name);

        TLLM_CHECK_WITH_INFO(ret != nullptr,
            "Unable to load UCX wrapper library symbol, possible cause is that TensorRT LLM library is not "
            "built with UCX support, please rebuild in UCX-enabled environment.");
        return ret;
    };
    std::unique_ptr<tensorrt_llm::executor::kv_cache::ConnectionManager> (*makeUcxConnectionManager)();
    *(void**) (&makeUcxConnectionManager) = load_sym(WrapperLibHandle, "makeUcxConnectionManager");
    return makeUcxConnectionManager();
}

class UcxCommTest : public ::testing::Test
{
};

using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
using TransceiverTag = tensorrt_llm::batch_manager::TransceiverTag;

TEST_F(UcxCommTest, Basic)
{

    try
    {
        TransceiverTag::Id id1;
        TransceiverTag::Id id2;

        auto connectionManager1 = makeOneUcxConnectionManager();
        EXPECT_NE(connectionManager1, nullptr);
        auto connectionManager2 = makeOneUcxConnectionManager();
        EXPECT_NE(connectionManager2, nullptr);
        auto CommState1 = connectionManager1->getCommState();
        auto CommState2 = connectionManager2->getCommState();
        ASSERT_EQ(CommState1.isSocketState(), true);
        ASSERT_EQ(CommState2.isSocketState(), true);

        auto connections1 = connectionManager2->getConnections(CommState1);
        ASSERT_EQ(connections1.size(), 1);
        auto connection1 = connections1[0];
        id1 = TransceiverTag::Id::REQUEST_SEND;
        connection1->send(DataContext{TransceiverTag::kID_TAG}, &id1, sizeof(id1));

        auto connection1Peer = connectionManager1->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id2, sizeof(id2));
        ASSERT_EQ(id2, id1);
        constexpr size_t bufferSize = 1024;
        std::vector<char> buffer(bufferSize);
        // Fill buffer with random data
        std::generate(buffer.begin(), buffer.end(), []() { return static_cast<char>(std::rand()); });

        connection1->send(DataContext{0x74}, buffer.data(), buffer.size());

        std::vector<char> recvBuffer(buffer.size());

        connection1Peer->recv(DataContext{0x74}, recvBuffer.data(), recvBuffer.size());

        ASSERT_EQ(memcmp(buffer.data(), recvBuffer.data(), buffer.size()), 0);

        // Test with CUDA memory
        tensorrt_llm::runtime::BufferManager bufferManager{std::make_shared<tensorrt_llm::runtime::CudaStream>()};

        // Create and fill source CUDA buffer with random data
        auto srcBuffer = bufferManager.gpu(buffer.size(), nvinfer1::DataType::kINT8);
        bufferManager.copy(buffer.data(), *srcBuffer);
        bufferManager.getStream().synchronize();

        auto dstBuffer = bufferManager.gpu(buffer.size(), nvinfer1::DataType::kINT8);

        // Send CUDA buffer using connection1
        connection1->send(DataContext{0x75}, srcBuffer->data(), srcBuffer->getSizeInBytes());

        // Receive into CUDA buffer using connection1Peer
        connection1Peer->recv(DataContext{0x75}, dstBuffer->data(), dstBuffer->getSizeInBytes());

        std::vector<char> recvCudaBuffer(buffer.size());
        bufferManager.copy(*dstBuffer, recvCudaBuffer.data(), dstBuffer->getMemoryType());
        bufferManager.getStream().synchronize();

        ASSERT_EQ(memcmp(buffer.data(), recvCudaBuffer.data(), buffer.size()), 0);
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        if (error.find("UCX wrapper library is not open correctly") != std::string::npos
            || error.find("Unable to load UCX wrapper library symbol") != std::string::npos)
        {
            GTEST_SKIP() << "UCX wrapper library is not open correctly. Skip this test case.";
        }

        throw e;
    }
}

TEST_F(UcxCommTest, multiSend)
{
    try
    {
        TransceiverTag::Id id1;
        TransceiverTag::Id id2;
        TransceiverTag::Id id1Peer;
        TransceiverTag::Id id2Peer;

        auto manager1 = makeOneUcxConnectionManager();
        auto manager2 = makeOneUcxConnectionManager();
        auto managerRecv = makeOneUcxConnectionManager();

        auto connection1 = managerRecv->getConnections(manager1->getCommState())[0];
        auto connection2 = managerRecv->getConnections(manager2->getCommState())[0];
        id1 = TransceiverTag::Id::REQUEST_SEND;
        id2 = TransceiverTag::Id::REQUEST_SEND;
        connection1->send(DataContext{TransceiverTag::kID_TAG}, &id1, sizeof(id1));
        connection2->send(DataContext{TransceiverTag::kID_TAG}, &id2, sizeof(id2));
        auto connection1Peer = manager1->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id1Peer, sizeof(id1Peer));
        auto connection2Peer = manager2->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id2Peer, sizeof(id2Peer));
        ASSERT_EQ(id1Peer, id1);
        ASSERT_EQ(id2Peer, id2);
        constexpr size_t bufferSize = 1024;
        std::vector<char> buffer1(bufferSize);
        std::vector<char> buffer2(bufferSize);
        std::generate(buffer1.begin(), buffer1.end(), []() { return static_cast<char>(std::rand()); });
        std::generate(buffer2.begin(), buffer2.end(), []() { return static_cast<char>(std::rand()); });

        connection1Peer->send(DataContext{0x74}, buffer1.data(), buffer1.size());
        connection2Peer->send(DataContext{0x74}, buffer2.data(), buffer2.size());

        std::vector<char> recvBuffer1(buffer1.size());
        std::vector<char> recvBuffer2(buffer2.size());
        connection2->recv(DataContext{0x74}, recvBuffer2.data(), recvBuffer2.size());
        connection1->recv(DataContext{0x74}, recvBuffer1.data(), recvBuffer1.size());
        ASSERT_EQ(memcmp(buffer1.data(), recvBuffer1.data(), buffer1.size()), 0);
        ASSERT_EQ(memcmp(buffer2.data(), recvBuffer2.data(), buffer2.size()), 0);

        tensorrt_llm::runtime::BufferManager bufferManager{std::make_shared<tensorrt_llm::runtime::CudaStream>()};

        auto srcBuffer1 = bufferManager.gpu(buffer1.size(), nvinfer1::DataType::kINT8);
        auto srcBuffer2 = bufferManager.gpu(buffer2.size(), nvinfer1::DataType::kINT8);
        bufferManager.copy(buffer1.data(), *srcBuffer1);
        bufferManager.copy(buffer2.data(), *srcBuffer2);
        bufferManager.getStream().synchronize();

        auto dstBuffer1 = bufferManager.gpu(buffer1.size(), nvinfer1::DataType::kINT8);
        auto dstBuffer2 = bufferManager.gpu(buffer2.size(), nvinfer1::DataType::kINT8);

        connection1Peer->send(DataContext{0x75}, srcBuffer1->data(), srcBuffer1->getSizeInBytes());
        connection2Peer->send(DataContext{0x75}, srcBuffer2->data(), srcBuffer2->getSizeInBytes());
        connection2->recv(DataContext{0x75}, dstBuffer2->data(), dstBuffer2->getSizeInBytes());

        connection1->recv(DataContext{0x75}, dstBuffer1->data(), dstBuffer1->getSizeInBytes());
        std::vector<char> recvCudaBuffer1(buffer1.size());
        std::vector<char> recvCudaBuffer2(buffer2.size());
        bufferManager.copy(*dstBuffer1, recvCudaBuffer1.data(), dstBuffer1->getMemoryType());
        bufferManager.copy(*dstBuffer2, recvCudaBuffer2.data(), dstBuffer2->getMemoryType());
        bufferManager.getStream().synchronize();

        ASSERT_EQ(memcmp(buffer1.data(), recvCudaBuffer1.data(), buffer1.size()), 0);
        ASSERT_EQ(memcmp(buffer2.data(), recvCudaBuffer2.data(), buffer2.size()), 0);
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        if (error.find("UCX wrapper library is not open correctly") != std::string::npos
            || error.find("Unable to load UCX wrapper library symbol") != std::string::npos)
        {
            GTEST_SKIP() << "UCX wrapper library is not open correctly. Skip this test case.";
        }

        throw e;
    }
}

TEST_F(UcxCommTest, CommCache)
{

    try
    {
        TransceiverTag::Id id1;
        TransceiverTag::Id id2;

        auto connectionManager1 = makeOneUcxConnectionManager();
        EXPECT_NE(connectionManager1, nullptr);
        auto connectionManager2 = makeOneUcxConnectionManager();
        EXPECT_NE(connectionManager2, nullptr);
        auto CommState1 = connectionManager1->getCommState();
        auto CommState2 = connectionManager2->getCommState();
        ASSERT_EQ(CommState1.isSocketState(), true);
        ASSERT_EQ(CommState2.isSocketState(), true);

        auto connections1 = connectionManager2->getConnections(CommState1);
        ASSERT_EQ(connections1.size(), 1);
        auto connection1 = connections1[0];
        id1 = TransceiverTag::Id::REQUEST_SEND;
        connection1->send(DataContext{TransceiverTag::kID_TAG}, &id1, sizeof(id1));

        auto connection1Peer = connectionManager1->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id2, sizeof(id2));
        ASSERT_EQ(id2, id1);
        auto connection1Cached = connectionManager2->getConnections(CommState1)[0];
        ASSERT_EQ(connection1Cached, connection1);
        id1 = TransceiverTag::Id::REQUEST_SEND;
        connection1Cached->send(DataContext{TransceiverTag::kID_TAG}, &id1, sizeof(id1));

        auto connection1PeerCached
            = connectionManager1->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id2, sizeof(id2));
        ASSERT_EQ(id2, id1);

        ASSERT_EQ(connection1PeerCached, connection1Peer);
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        if (error.find("UCX wrapper library is not open correctly") != std::string::npos
            || error.find("Unable to load UCX wrapper library symbol") != std::string::npos)
        {
            GTEST_SKIP() << "UCX wrapper library is not open correctly. Skip this test case.";
        }

        throw e;
    }
}

}; // namespace
