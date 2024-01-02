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

#include "tensorrt_llm/common/mpiUtils.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/common.h"

#include <csignal>
#include <mpi.h>
#include <mutex>
#include <type_traits>

// We rely on SizeType being int32_t in some places with weak type checking,
// i.e. we're passing void ptr to some function. To prevent mysterious errors
// in the future, we trigger a compilation error here if SizeType isn't int32_t.
static_assert(std::is_same<tensorrt_llm::runtime::SizeType, std::int32_t>::value);

namespace tensorrt_llm::mpi
{

MPI_Datatype getMpiDtype(MpiType dtype)
{
    static const std::unordered_map<MpiType, MPI_Datatype> dtype_map{

        {MpiType::kBYTE, MPI_BYTE},
        {MpiType::kHALF, MPI_UINT16_T},
        {MpiType::kFLOAT, MPI_FLOAT},
        {MpiType::kDOUBLE, MPI_DOUBLE},
        {MpiType::kBOOL, MPI_C_BOOL},
        {MpiType::kINT8, MPI_INT8_T},
        {MpiType::kUINT8, MPI_UINT8_T},
        {MpiType::kINT32, MPI_INT32_T},
        {MpiType::kUINT32, MPI_UINT32_T},
        {MpiType::kINT64, MPI_INT64_T},
        {MpiType::kUINT64, MPI_UINT64_T},
        {MpiType::kFP8, MPI_UINT8_T},
        {MpiType::kBF16, MPI_UINT16_T},
    };
    return dtype_map.at(dtype);
}

MPI_Op getMpiOp(MpiOp op)
{
    static const std::unordered_map<MpiOp, MPI_Op> op_map{
        {MpiOp::NULLOP, MPI_OP_NULL},
        {MpiOp::MAX, MPI_MAX},
        {MpiOp::MIN, MPI_MIN},
        {MpiOp::SUM, MPI_SUM},
        {MpiOp::PROD, MPI_PROD},
        {MpiOp::LAND, MPI_LAND},
        {MpiOp::BAND, MPI_BAND},
        {MpiOp::LOR, MPI_LOR},
        {MpiOp::BOR, MPI_BOR},
        {MpiOp::LXOR, MPI_LXOR},
        {MpiOp::BXOR, MPI_BXOR},
        {MpiOp::MINLOC, MPI_MINLOC},
        {MpiOp::MAXLOC, MPI_MAXLOC},
        {MpiOp::REPLACE, MPI_REPLACE},
    };
    return op_map.at(op);
}

namespace
{

bool mpiInitialized = false;
std::mutex mpiMutex;

} // namespace

void initialize(MpiThreadSupport threadMode)
{
    std::lock_guard<std::mutex> lk(mpiMutex);
    if (mpiInitialized)
    {
        return;
    }

    int initialized = 0;
    TLLM_MPI_CHECK(MPI_Initialized(&initialized));
    if (!initialized)
    {
        TLLM_LOG_INFO("Initializing MPI with thread mode %d", threadMode);
        int providedMode;
        auto requiredMode = static_cast<int>(threadMode);
        MPICHECK(MPI_Init_thread(nullptr, nullptr, requiredMode, &providedMode));
        TLLM_CHECK_WITH_INFO(providedMode >= requiredMode, "MPI_Init_thread failed");
        std::atexit([]() { MPI_Finalize(); });

        auto previousHandler = std::signal(SIGABRT, [](int signal) { MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); });
        TLLM_CHECK_WITH_INFO(previousHandler != SIG_ERR, "Signal handler setup failed");
    }

    mpiInitialized = true;
}

void MpiComm::barrier() const
{
    MPICHECK(MPI_Barrier(mComm));
}

std::shared_ptr<MpiRequest> MpiComm::bcastAsync(void* buffer, size_t size, MpiType dtype, int root) const
{
    std::shared_ptr<MpiRequest> r = std::make_shared<MpiRequest>();
    MPICHECK(MPI_Ibcast(buffer, size, getMpiDtype(dtype), root, mComm, &r->mRequest));
    return r;
}

void MpiComm::bcast(void* buffer, size_t size, MpiType dtype, int root) const
{
    MPICHECK(MPI_Bcast(buffer, size, getMpiDtype(dtype), root, mComm));
}

void MpiComm::bcast(std::vector<int64_t>& packed, int root) const
{
    int64_t nWords1;
    auto const rank = getRank();
    if (rank == root)
    {
        nWords1 = static_cast<int64_t>(packed.size());
    }
    auto const mpiInt64 = MpiTypeConverter<int64_t>::value;
    bcast(&nWords1, 1, mpiInt64, root);
    if (rank != root)
    {
        packed.resize(nWords1);
    }
    bcast(packed.data(), packed.size(), mpiInt64, root);
}

void MpiComm::send(void const* buffer, size_t size, MpiType dtype, int dest, int tag) const
{
    MPICHECK(MPI_Send(buffer, size, getMpiDtype(dtype), dest, tag, mComm));
}

MPI_Status MpiComm::recv(void* buffer, size_t size, MpiType dtype, int source, int tag) const
{
    MPI_Status status{};
    MPICHECK(MPI_Recv(buffer, size, getMpiDtype(dtype), source, tag, mComm, &status));
    return status;
}

MpiComm MpiComm::split(int color, int key) const
{
    MPI_Comm splitComm;
    MPICHECK(MPI_Comm_split(mComm, color, key, &splitComm));
    return MpiComm{splitComm, true};
}

void MpiComm::allreduce(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiOp op) const
{
    MPICHECK(MPI_Allreduce(sendbuf, recvbuf, count, getMpiDtype(dtype), getMpiOp(op), mComm));
}

void MpiComm::allgather(const void* sendbuf, void* recvbuf, int count, MpiType dtype) const
{
    MPICHECK(MPI_Allgather(sendbuf, count, getMpiDtype(dtype), recvbuf, count, getMpiDtype(dtype), mComm));
}

int MpiComm::getRank() const
{
    int rank = 0;
    MPICHECK(MPI_Comm_rank(mComm, &rank));
    return rank;
}

int MpiComm::getSize() const
{
    int world_size = 1;
    MPICHECK(MPI_Comm_size(mComm, &world_size));
    return world_size;
}

MpiComm const& MpiComm::world()
{
    static MpiComm commWorld{MPI_COMM_WORLD, false};
    return commWorld;
}

MpiComm& MpiComm::session()
{
    static MpiComm commSession{world(), false};
    return commSession;
}

MpiComm::MpiComm(MPI_Comm g, bool freeComm)
    : mComm{g}
    , mFreeComm{freeComm}
{
    TLLM_CHECK(mComm != MPI_COMM_NULL);
    if (g == MPI_COMM_WORLD)
    {
        initialize();
    }
}

MpiComm::~MpiComm() noexcept
{
    if (mFreeComm && mComm && MPI_Comm_free(&mComm) != MPI_SUCCESS)
    {
        TLLM_LOG_ERROR("MPI_Comm_free failed");
    }
}

MpiComm::MpiComm(MpiComm&& comm) noexcept
    : mComm{comm.mComm}
    , mFreeComm{comm.mFreeComm}
{
    comm.mFreeComm = false;
}

MpiComm& MpiComm::operator=(MpiComm&& comm) noexcept
{
    this->~MpiComm();
    mComm = comm.mComm;
    mFreeComm = comm.mFreeComm;
    comm.mFreeComm = false;
    return *this;
}

} // namespace tensorrt_llm::mpi
