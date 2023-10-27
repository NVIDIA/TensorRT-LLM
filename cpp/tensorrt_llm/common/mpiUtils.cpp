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

#include "tensorrt_llm/common/mpiUtils.h"
#include "mpi.h"
#include "tensorrt_llm/runtime/common.h"
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
        {MPI_TYPE_BYTE, MPI_BYTE},
        {MPI_TYPE_CHAR, MPI_CHAR},
        {MPI_TYPE_INT, MPI_INT},
        {MPI_TYPE_FLOAT, MPI_FLOAT},
        {MPI_TYPE_DOUBLE, MPI_DOUBLE},
        {MPI_TYPE_INT64_T, MPI_INT64_T},
        {MPI_TYPE_INT32_T, MPI_INT32_T},
        {MPI_TYPE_UINT64_T, MPI_UINT64_T},
        {MPI_TYPE_UINT32_T, MPI_UINT32_T},
        {MPI_TYPE_UNSIGNED_LONG_LONG, MPI_UNSIGNED_LONG_LONG},
        {MPI_TYPE_SIZETYPE, MPI_INT32_T},
    };
    return dtype_map.at(dtype);
}

MPI_Op getMpiOp(MpiOp op)
{
    static const std::unordered_map<MpiOp, MPI_Op> op_map{
        {MPI_OP_NULLOP, MPI_OP_NULL},
        {MPI_OP_MAX, MPI_MAX},
        {MPI_OP_MIN, MPI_MIN},
        {MPI_OP_SUM, MPI_SUM},
        {MPI_OP_PROD, MPI_PROD},
        {MPI_OP_LAND, MPI_LAND},
        {MPI_OP_BAND, MPI_BAND},
        {MPI_OP_LOR, MPI_LOR},
        {MPI_OP_BOR, MPI_BOR},
        {MPI_OP_LXOR, MPI_LXOR},
        {MPI_OP_BXOR, MPI_BXOR},
        {MPI_OP_MINLOC, MPI_MINLOC},
        {MPI_OP_MAXLOC, MPI_MAXLOC},
        {MPI_OP_REPLACE, MPI_REPLACE},
    };
    return op_map.at(op);
}

void initialize(int* argc, char*** argv)
{
    MPICHECK(MPI_Init(argc, argv));
}

void finalize()
{
    MPICHECK(MPI_Finalize());
}

bool isInitialized()
{
    int mpi_initialized = 0;
    MPICHECK(MPI_Initialized(&mpi_initialized));
    return static_cast<bool>(mpi_initialized);
}

void initThread(int* argc, char*** argv, MpiThreadSupport required, int* provided)
{
    switch (required)
    {
    case THREAD_SINGLE: MPICHECK(MPI_Init_thread(argc, argv, MPI_THREAD_SINGLE, provided)); break;
    case THREAD_FUNNELED: MPICHECK(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, provided)); break;
    case THREAD_SERIALIZED: MPICHECK(MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, provided)); break;
    case THREAD_MULTIPLE: MPICHECK(MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, provided)); break;
    default: break;
    }
}

int getCommWorldRank()
{
    int rank = 0;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    return rank;
}

int getCommWorldSize()
{
    int world_size = 1;
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    return world_size;
}

void barrier(MpiComm comm)
{
    MPICHECK(MPI_Barrier(comm.group));
}

void barrier()
{
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
}

std::shared_ptr<MpiRequest> bcast_async(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm)
{
    std::shared_ptr<MpiRequest> r = std::make_shared<MpiRequest>();
    MPICHECK(MPI_Ibcast(buffer, size, getMpiDtype(dtype), root, comm.group, &r->mRequest));
    return r;
}

void bcast(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm)
{
    MPICHECK(MPI_Bcast(buffer, size, getMpiDtype(dtype), root, comm.group));
}

void bcast(std::vector<int64_t>& packed, int root, MpiComm comm)
{
    int64_t nWords1;
    if (getCommWorldRank() == root)
    {
        nWords1 = static_cast<int64_t>(packed.size());
    }
    bcast(&nWords1, 1, MPI_TYPE_INT64_T, root, comm);
    if (getCommWorldRank() != root)
    {
        packed.resize(nWords1);
    }
    bcast(packed.data(), packed.size(), MPI_TYPE_INT64_T, root, comm);
}

void comm_split(MpiComm comm, int color, int key, MpiComm* newcomm)
{
    MPICHECK(MPI_Comm_split(comm.group, color, key, &newcomm->group));
}

void allreduce(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiOp op, MpiComm comm)
{
    MPICHECK(MPI_Allreduce(sendbuf, recvbuf, count, getMpiDtype(dtype), getMpiOp(op), comm.group));
}

void allgather(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiComm comm)
{
    MPICHECK(MPI_Allgather(sendbuf, count, getMpiDtype(dtype), recvbuf, count, getMpiDtype(dtype), comm.group));
}

} // namespace tensorrt_llm::mpi
