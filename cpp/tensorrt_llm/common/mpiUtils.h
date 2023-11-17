/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#include <cstdlib>
#include <memory>
#include <mpi.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#define COMM_WORLD MpiComm(MPI_COMM_WORLD)
#define MPICHECK(cmd) TLLM_MPI_CHECK(cmd)

// A wrapper module of the MPI library.
namespace tensorrt_llm::mpi
{

// A wrapper of MPI data type. MPI_TYPE_{data_type}
enum MpiType
{
    MPI_TYPE_BYTE,
    MPI_TYPE_CHAR,
    MPI_TYPE_INT,
    MPI_TYPE_FLOAT,
    MPI_TYPE_DOUBLE,
    MPI_TYPE_INT64_T,
    MPI_TYPE_INT32_T,
    MPI_TYPE_UINT64_T,
    MPI_TYPE_UINT32_T,
    MPI_TYPE_UNSIGNED_LONG_LONG,
    MPI_TYPE_SIZETYPE,
};

// A wrapper of MPI_Op type.
enum MpiOp
{
    MPI_OP_NULLOP,
    MPI_OP_MAX,
    MPI_OP_MIN,
    MPI_OP_SUM,
    MPI_OP_PROD,
    MPI_OP_LAND,
    MPI_OP_BAND,
    MPI_OP_LOR,
    MPI_OP_BOR,
    MPI_OP_LXOR,
    MPI_OP_BXOR,
    MPI_OP_MINLOC,
    MPI_OP_MAXLOC,
    MPI_OP_REPLACE,
};

// A wrapper of the level of MPI thread support
enum MpiThreadSupport
{
    THREAD_SINGLE,
    THREAD_FUNNELED,
    THREAD_SERIALIZED,
    THREAD_MULTIPLE
};

struct MpiComm
{
    MPI_Comm group;
    MpiComm(){};
    MpiComm(MPI_Comm g)
        : group(g){};
};

class MpiRequest
{
public:
    MpiRequest() {}

    ~MpiRequest() {}

    void wait()
    {
        // TODO: Don't ignore return status
        MPI_Wait(&mRequest, MPI_STATUS_IGNORE);
    }

    MPI_Request mRequest;
};

MPI_Datatype getMpiDtype(MpiType dtype);

void initialize(int* argc, char*** argv);
void initThread(int* argc, char*** argv, MpiThreadSupport required, int* provided);
void finalize();
bool isInitialized();
void barrier(MpiComm comm);
void barrier();

int getCommWorldRank();
int getCommWorldSize();

std::shared_ptr<MpiRequest> bcast_async(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm);
void bcast(void* buffer, size_t size, MpiType dtype, int root, MpiComm comm);
void bcast(std::vector<int64_t>& packed, int root, MpiComm comm);
void comm_split(MpiComm comm, int color, int key, MpiComm* newcomm);
void allreduce(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiOp op, MpiComm comm);
void allgather(const void* sendbuf, void* recvbuf, int count, MpiType dtype, MpiComm comm);

} // namespace tensorrt_llm::mpi
