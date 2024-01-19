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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

#include <algorithm>

namespace mpi = tensorrt_llm::mpi;
namespace tr = tensorrt_llm::runtime;

TEST(MPIUtils, RankAndSize)
{
    auto& comm = mpi::MpiComm::world();
    auto const rank = comm.getRank();
    EXPECT_LE(0, rank);
    auto const size = comm.getSize();
    EXPECT_LE(rank, size);
}

template <typename T>
void testBroadcast()
{
    auto& comm = mpi::MpiComm::world();
    auto const rank = comm.getRank();
    auto constexpr expectedValue = static_cast<T>(42);
    auto constexpr root = 0;
    auto value = rank == root ? expectedValue : T{};
    comm.bcast(value, root);
    EXPECT_EQ(value, expectedValue);
}

TEST(MPIUtils, Broadcast)
{
    testBroadcast<std::byte>();
    testBroadcast<float>();
    testBroadcast<double>();
    testBroadcast<bool>();
    testBroadcast<std::int8_t>();
    testBroadcast<std::uint8_t>();
    testBroadcast<std::int32_t>();
    testBroadcast<std::uint32_t>();
    testBroadcast<std::int64_t>();
    testBroadcast<std::uint64_t>();
}

#if ENABLE_MULTI_DEVICE
TEST(MPIUtils, BroadcastNcclId)
{
    auto& comm = mpi::MpiComm::world();
    auto const rank = comm.getRank();
    auto constexpr root = 0;
    ncclUniqueId id;
    if (rank == root)
    {
        ncclGetUniqueId(&id);
    }
    else
    {
        std::memset(&id, 0, sizeof(id));
    }
    comm.bcast(id, root);
    EXPECT_TRUE(std::any_of(
        id.internal, id.internal + sizeof(id.internal) / sizeof(id.internal[0]), [](auto x) { return x != 0; }));
}
#endif // ENABLE_MULTI_DEVICE

template <typename T>
void testBroadcastBuffer()
{
    using BufferType = T;
    auto& comm = mpi::MpiComm::world();
    auto const rank = comm.getRank();
    auto constexpr root = 0;
    auto constexpr expectedValue = static_cast<BufferType>(42);
    auto const value = rank == root ? expectedValue : BufferType{};
    auto constexpr bufferSize = 1024;
    auto buffer = tr::BufferManager::cpu(bufferSize, tr::TRTDataType<BufferType>::value);
    auto* data = tr::bufferCast<BufferType>(*buffer);
    std::fill(data, data + bufferSize, value);
    comm.bcast(*buffer, root);
    EXPECT_TRUE(std::all_of(data, data + bufferSize, [&](auto x) { return x == expectedValue; }));
}

TEST(MPIUtils, BroadcastBuffer)
{
    testBroadcastBuffer<float>();
    testBroadcastBuffer<bool>();
    testBroadcastBuffer<std::int8_t>();
    testBroadcastBuffer<std::uint8_t>();
    testBroadcastBuffer<std::int32_t>();
    testBroadcastBuffer<std::uint32_t>();
    testBroadcastBuffer<std::int64_t>();
    testBroadcastBuffer<std::uint64_t>();
}

template <typename T>
void testSendRecv()
{
    auto& comm = mpi::MpiComm::world();
    auto const rank = comm.getRank();
    auto constexpr expectedValue = static_cast<T>(42);
    auto constexpr tag = 0;
    if (rank == 0)
    {
        comm.send(expectedValue, 1, tag);
    }
    else if (rank == 1)
    {
        T value{};
        comm.recv(value, 0, tag);
        EXPECT_EQ(value, expectedValue);
    }
}

TEST(MPIUtils, SendRecv)
{
    auto& comm = mpi::MpiComm::world();
    if (comm.getSize() < 2)
    {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    testSendRecv<float>();
    testSendRecv<bool>();
    testSendRecv<std::int8_t>();
    testSendRecv<std::uint8_t>();
    testSendRecv<std::int32_t>();
    testSendRecv<std::uint32_t>();
    testSendRecv<std::int64_t>();
    testSendRecv<std::uint64_t>();
}

TEST(MPIUtils, SessionCommunicator)
{
    auto& world = mpi::MpiComm::world();
    if (world.getSize() < 2)
    {
        GTEST_SKIP() << "Test requires at least 2 processes";
    }

    auto const rank = world.getRank();
    auto const size = world.getSize();
    auto const sessionSize = (size + 1) / 2;
    auto const sessionColor = rank / sessionSize;
    auto sessionRank = rank % sessionSize;
    auto& session = mpi::MpiComm::session();
    EXPECT_EQ(session, world);
    session = world.split(sessionColor, sessionRank);
    EXPECT_EQ(session, mpi::MpiComm::session());
    EXPECT_NE(session, world);
    EXPECT_EQ(session.getRank(), sessionRank);
    EXPECT_LE(sessionSize - 1, session.getSize());
    EXPECT_LE(session.getSize(), sessionSize);

    session = session.split(sessionRank, 0);
    EXPECT_EQ(session, mpi::MpiComm::session());
    EXPECT_EQ(session.getRank(), 0);
    EXPECT_EQ(session.getSize(), 1);
}
