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

#include "ipcsocket.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "userbuffers.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

namespace tensorrt_llm::runtime::ub
{
#define UB_ONESHOT_DEFAULT_VALUE 1
#define UB_FORCE_ENABLE_ONESHOT_TOKEN_NUM_THRESHOLD 8

#define MULTICAST_GB_TOTAL 512

#ifdef MNNVL
#if CUDART_VERSION < 12030
// MNNVL: FABRIC handle support lifted from CUDA 12.3
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType) 0x8ULL)
#define CU_IPC_HANDLE_SIZE 64

typedef struct CUmemFabricHandle_st
{
    unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;

typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif
#endif

#define CUCHECK(cmd)                                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        CUresult const retval = cmd;                                                                                   \
        if (retval != CUDA_SUCCESS)                                                                                    \
        {                                                                                                              \
            const char* error_string;                                                                                  \
            cuGetErrorString(retval, &error_string);                                                                   \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, error_string);                               \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0);

#define NCCLCHECK(cmd)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        ipcSocketResult_t const r = cmd;                                                                               \
        if (r != ipcSocketSuccess)                                                                                     \
        {                                                                                                              \
            printf("Failed, NCCL error %s:%d ''\n", __FILE__, __LINE__);                                               \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define NCCLCHECKGOTO(call, RES, label)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        RES = call;                                                                                                    \
        if (RES != ipcSocketSuccess && RES != ipcSocketInProgress)                                                     \
        {                                                                                                              \
            /* Print the back trace*/                                                                                  \
            goto label;                                                                                                \
        }                                                                                                              \
    } while (0);

namespace
{
void ub_alloc_copy_allgather(void** globaldata, size_t data_bytes, void* localdata, MPI_Comm c)
{
    int myrank = 0;
    int nranks = 0;
    MPI_Comm_rank(c, &myrank);
    MPI_Comm_size(c, &nranks);
    *globaldata = malloc(nranks * data_bytes); // peer addresses
    memcpy(reinterpret_cast<uint8_t*>(*globaldata) + myrank * data_bytes, localdata, data_bytes);
    MPI_Allgather(reinterpret_cast<uint8_t*>(*globaldata) + myrank * data_bytes, data_bytes, MPI_BYTE, *globaldata,
        data_bytes, MPI_BYTE, c);
}

void ub_bcast(void* ptr, size_t data_bytes, int root, MPI_Comm c)
{
    MPI_Bcast(ptr, data_bytes, MPI_BYTE, root, c);
}

void ub_allreduce_longmax(void* in, void* out, MPI_Comm c)
{
    MPI_Allreduce(in, out, 1, MPI_LONG, MPI_MAX, c);
}

void ub_gather(void* sbuf, size_t data_bytes, void* rbuf, int root, MPI_Comm c)
{
    MPI_Gather(sbuf, data_bytes, MPI_BYTE, rbuf, data_bytes, MPI_BYTE, root, c);
}

void ub_barrier(MPI_Comm c)
{
    MPI_Barrier(c);
}

void ub_free(void* ptr)
{
    free(ptr);
}
} // namespace

int create_communicator_grouped2(communicator** comm, tensorrt_llm::runtime::WorldConfig const& world_config)
{
    *comm = (communicator*) malloc(sizeof(communicator));

    int myrank = 0;
    int nranks = 0;
    int cur_dev = 0;
    int ndev = 0;
    MPI_Comm_dup(MPI_COMM_WORLD, &(*comm)->comm_world);
    myrank = world_config.getRank();
    nranks = world_config.getSize();
    (*comm)->nranks = nranks;
    (*comm)->myrank = myrank;
    (*comm)->free_region = 0;
    (*comm)->pdl_launch = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;

    cudaDeviceProp device_prop{};
    TLLM_CUDA_CHECK(cudaGetDevice(&cur_dev));
    TLLM_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, cur_dev));
    (*comm)->sm_arch = device_prop.major;
    (*comm)->use_rr_kernel = device_prop.major == 8;
    if (getenv("OVERRIDERR"))
        (*comm)->use_rr_kernel = atoi(getenv("OVERRIDERR"));
    (*comm)->cga_size = getenv("CGASIZE")                     ? atoi(getenv("CGASIZE"))
        : (device_prop.major == 9 && !(*comm)->use_rr_kernel) ? 4
                                                              : 1;
    (*comm)->oneshot = getenv("UB_ONESHOT") ? atoi(getenv("UB_ONESHOT")) : UB_ONESHOT_DEFAULT_VALUE;
    (*comm)->oneshot_force_enable_threshold = getenv("UB_ONESHOT_FORCE_ENABLE_THRESHOLD")
        ? atoi(getenv("UB_ONESHOT_FORCE_ENABLE_THRESHOLD"))
        : UB_FORCE_ENABLE_ONESHOT_TOKEN_NUM_THRESHOLD;

    if (ndev > 1)
    {
        auto device_id = world_config.getDevice();
        if (cur_dev != device_id)
        {
            TLLM_LOG_INFO(
                "[UserBuffer] rank %d: device used %d[%d] ,resetting device to %d", myrank, cur_dev, ndev, device_id);
        }
        TLLM_CUDA_CHECK(cudaSetDevice(device_id));
        cur_dev = device_id;
    }
    (*comm)->mydev = cur_dev;
    (*comm)->nvrank = myrank;
    (*comm)->nvsize = nranks;

    (*comm)->tp_size = world_config.getTensorParallelism();
    (*comm)->tp_rank = world_config.getTensorParallelRank();
    (*comm)->tp_first_rank = (*comm)->nvrank - (*comm)->tp_rank;

    MPI_Comm_dup(MPI_COMM_WORLD, &(*comm)->comm_intra);

#define NBUF 1

    constexpr bool nvls_supported = true;

    if (nvls_supported && (*comm)->sm_arch >= 9 && (*comm)->tp_size > 1 && !getenv("UB_SKIPMC"))
    {
        size_t mc_maxsize = MULTICAST_GB_TOTAL * (1ull << 30);
        (*comm)->mc_offset = 0;
        (*comm)->use_mc = 1;
        size_t gran;
        CUmulticastObjectProp mcProp = {};
        mcProp.numDevices = (*comm)->tp_size;
        mcProp.size = (*comm)->mc_maxsize;
#ifdef MNNVL
        mcProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
#else
        mcProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
        CUCHECK(cuMulticastGetGranularity(&gran, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
        mc_maxsize = ((mc_maxsize + gran - 1) / gran) * gran;
        mcProp.size = mc_maxsize;
        (*comm)->mc_maxsize = mc_maxsize;
        if ((*comm)->tp_rank == 0)
            CUCHECK(cuMulticastCreate(&(*comm)->mc_handle, &mcProp));

#ifdef MNNVL
        CUmemFabricHandle* exphndl = (CUmemFabricHandle*) malloc(sizeof(CUmemFabricHandle));
        CUmemFabricHandle* tmphndl = (CUmemFabricHandle*) malloc(sizeof(CUmemFabricHandle));
        if ((*comm)->tp_rank == 0)
            CUCHECK(cuMemExportToShareableHandle(
                static_cast<void*>(tmphndl), (*comm)->mc_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
        for (int tp_group = 0; tp_group < (*comm)->nvsize / (*comm)->tp_size; ++tp_group)
        {
            int root = tp_group * (*comm)->tp_size;
            ub_bcast(tmphndl, sizeof(CUmemFabricHandle), root, (*comm)->comm_intra);
            if ((*comm)->tp_first_rank == root)
                memcpy(exphndl, tmphndl, sizeof(CUmemFabricHandle));
        }
        if ((*comm)->tp_rank != 0)
            CUCHECK(cuMemImportFromShareableHandle(
                &(*comm)->mc_handle, reinterpret_cast<void*>(exphndl), CU_MEM_HANDLE_TYPE_FABRIC));
        free(exphndl);
        free(tmphndl);
#else
        int fd = 0;
        uint32_t volatile abortFlag = 0;
        IpcSocketHandle ipcSock{};
        srand(time(NULL));
        uint64_t opId = (uint64_t) (rand()) ^ ((uint64_t) (rand()) << 32);
        ub_bcast(&opId, sizeof(uint64_t), 0, (*comm)->comm_world);
        ipcSocketResult_t ret = ipcSocketSuccess;
        NCCLCHECK(ipcSocketInit(&ipcSock, (*comm)->nvrank, (uint64_t) opId, &abortFlag));
        ub_barrier((*comm)->comm_world);
        if ((*comm)->tp_rank == 0)
        {
            CUCHECK(cuMemExportToShareableHandle(
                &fd, (*comm)->mc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
            for (int p = 1; p < (*comm)->tp_size; p++)
            {
                ub_barrier((*comm)->comm_intra);
                NCCLCHECKGOTO(ipcSocketSendFd(&ipcSock, fd, p + (*comm)->tp_first_rank, (uint64_t) opId), ret, error);
            }
        }
        else
        {
            for (int i = 0; i < (*comm)->tp_rank; i++)
                ub_barrier((*comm)->comm_intra);
            NCCLCHECKGOTO(ipcSocketRecvFd(&ipcSock, &fd), ret, error);
            for (int i = 0; i < (*comm)->tp_size - (*comm)->tp_rank - 1; i++)
                ub_barrier((*comm)->comm_intra);
            CUCHECK(cuMemImportFromShareableHandle(&(*comm)->mc_handle,
                reinterpret_cast<void*>(static_cast<uintptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        }
    error:
        NCCLCHECK(ipcSocketClose(&ipcSock));
        close(fd);
#endif
        CUCHECK(cuMulticastAddDevice((*comm)->mc_handle, (*comm)->mydev));

        CUdeviceptr mc_va;
        CUCHECK(cuMemAddressReserve(&mc_va, mc_maxsize, 0, 0U, 0));
        CUCHECK(cuMemMap(mc_va, mc_maxsize, 0, (*comm)->mc_handle, 0));

        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = (*comm)->mydev;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess(mc_va, mc_maxsize, &accessDesc, 1));

        (*comm)->mc_baseptr = reinterpret_cast<void*>(mc_va);
        ub_barrier((*comm)->comm_world);
        if ((*comm)->tp_rank == 0)
        {
            TLLM_LOG_INFO(
                "[UserBuffer] rank %d, MC initialized successfully, window size = %ld", (*comm)->nvrank, mc_maxsize);
        }
    }
    else
    {
        if ((*comm)->tp_rank == 0)
            TLLM_LOG_WARNING("[UserBuffer] rank %d, MC NOT initialized and used", (*comm)->nvrank);
        (*comm)->mc_maxsize = 0;
        (*comm)->mc_offset = 0;
        (*comm)->use_mc = 0;
    }

#define LOCALSIZE 4 * (REG0_OFFSET(*comm) + REG0_FLAGS + REG0_COMMBUFFER * NBUF)
    register_user_buffer_collective(&((*comm)->gpu_ptrs), LOCALSIZE, *comm); // will use handler 0

    TLLM_CUDA_CHECK(cudaMalloc(&(*comm)->send_id, (*comm)->nranks * sizeof(int)));
    TLLM_CUDA_CHECK(cudaMalloc(&(*comm)->recv_id, MAX_REGIONS * (*comm)->nranks * sizeof(int)));
    TLLM_CUDA_CHECK(cudaMemset((*comm)->send_id, 0, (*comm)->nranks * sizeof(int)));
    TLLM_CUDA_CHECK(cudaMemset((*comm)->recv_id, 0, MAX_REGIONS * (*comm)->nranks * sizeof(int)));
    (*comm)->sms = getenv("MAXSMS") ? atoi(getenv("MAXSMS")) : 16;
    (*comm)->threads = getenv("MAXTHREADS") ? atoi(getenv("MAXTHREADS")) : 1024;

    return 0;
}

void destroy_communicator(communicator* comm)
{
    // WIP
}

int register_user_buffer_collective(void** gpubuff, size_t bytes, communicator* comm)
{
    if (comm->free_region > MAX_REGIONS)
        return -1;
    int const hndl = comm->free_region;
    comm->peer_ptr[hndl] = (void**) malloc(sizeof(void*) * (comm->nvsize));
    size_t aligned_size = bytes;
    comm->memflags[hndl] = 0;

    int const nranks = comm->nvsize;
    int const myrank = comm->nvrank;
    auto** remptrs = static_cast<void**>(malloc(nranks * sizeof(void*)));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = comm->mydev;
#ifdef MNNVL
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
#else
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#endif
    size_t granularity = 0;
    CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    aligned_size = (bytes + granularity - 1) / granularity * granularity;

    if (comm->use_mc)
    {
        CUmulticastObjectProp mcProp = {};
        mcProp.numDevices = comm->tp_size;
        mcProp.size = aligned_size;
        mcProp.handleTypes = prop.requestedHandleTypes;
        CUCHECK(cuMulticastGetGranularity(&granularity, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
        aligned_size = (aligned_size + granularity - 1) / granularity * granularity;
    }

    prop.location.id = comm->mydev;
    prop.allocFlags.gpuDirectRDMACapable = 1;

    comm->uchandles[hndl] = (CUmemGenericAllocationHandle*) malloc(nranks * sizeof(CUmemGenericAllocationHandle));
    CUCHECK(cuMemCreate(&(comm->uchandles[hndl][myrank]), aligned_size, &prop, 0));

#ifdef MNNVL
    CUmemFabricHandle* exphndl;
    CUmemFabricHandle myhndl;
    CUCHECK(cuMemExportToShareableHandle(&myhndl, comm->uchandles[hndl][myrank], CU_MEM_HANDLE_TYPE_FABRIC, 0));
    ub_alloc_copy_allgather((void**) &exphndl, sizeof(CUmemFabricHandle), (void*) &myhndl, comm->comm_intra);
    for (int p = 0; p < nranks; p++)
        if (p != myrank)
            CUCHECK(cuMemImportFromShareableHandle(
                &comm->uchandles[hndl][p], reinterpret_cast<void*>(&exphndl[p]), CU_MEM_HANDLE_TYPE_FABRIC));
    ub_free(exphndl);
#else
    auto* peerfd = static_cast<int*>(malloc(nranks * sizeof(int)));
    CUCHECK(cuMemExportToShareableHandle(
        &peerfd[myrank], comm->uchandles[hndl][myrank], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));

    uint32_t volatile abortFlag = 0;
    IpcSocketHandle ipcSock{};
    uint64_t opId = (uint64_t) (rand()) ^ ((uint64_t) (rand()) << 32);
    ub_bcast(&opId, sizeof(uint64_t), 0, comm->comm_world);
    ipcSocketResult_t ret = ipcSocketSuccess;

    NCCLCHECK(ipcSocketInit(&ipcSock, myrank, (uint64_t) opId, &abortFlag));
    for (int p = 1; p < nranks; p++)
    {
        ub_barrier(comm->comm_intra);
        NCCLCHECKGOTO(ipcSocketSendFd(&ipcSock, peerfd[myrank], (myrank + p) % nranks, (uint64_t) opId), ret, error);
        NCCLCHECKGOTO(ipcSocketRecvFd(&ipcSock, &peerfd[(myrank + nranks - p) % nranks]), ret, error);
    }
error:
    NCCLCHECK(ipcSocketClose(&ipcSock));

    for (int p = 0; p < nranks; p++)
    {
        if (p != myrank)
            CUCHECK(cuMemImportFromShareableHandle(&comm->uchandles[hndl][p],
                reinterpret_cast<void*>(static_cast<uintptr_t>(peerfd[p])), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        close(peerfd[p]);
    }
    free(peerfd);
#endif
    CUdeviceptr ptr;
    CUCHECK(cuMemAddressReserve(&ptr, aligned_size * nranks, 0, 0, 0));
    comm->ucbase_ptr[hndl] = reinterpret_cast<void*>(ptr);
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDesc.location.id = comm->mydev;

    for (int i = 0; i < nranks; i++)
    {
        CUCHECK(cuMemMap(ptr + (aligned_size * i), aligned_size, 0, comm->uchandles[hndl][i], 0));
        remptrs[i] = reinterpret_cast<void*>(ptr + (aligned_size * i));
        if (i == comm->nvrank)
        {
            if (hndl)
                *gpubuff = remptrs[i];
            else
                comm->gpu_ptrs = remptrs[i];
        }
        comm->peer_ptr[hndl][i] = remptrs[i];
    }
    CUCHECK(cuMemSetAccess(ptr, aligned_size * nranks, &accessDesc, 1));

    if (hndl == 0)
        TLLM_CUDA_CHECK(cudaMemset(comm->gpu_ptrs, 0, aligned_size));
    TLLM_CUDA_CHECK(cudaMemcpy(((char*) (comm->gpu_ptrs)) + (hndl * nranks * sizeof(void*)), remptrs,
        nranks * sizeof(void*), cudaMemcpyHostToDevice));
    free(remptrs);

    comm->memflags[hndl] = UB_MEM_ALLOCATED;
    if (!getenv("UB_SKIPUCPTR"))
        comm->memflags[hndl] |= UB_MEM_UC_CONTIG;

    if (comm->use_mc && comm->mc_maxsize >= comm->mc_offset + aligned_size)
    {
        CUCHECK(cuMulticastBindMem(
            comm->mc_handle, comm->mc_offset, comm->uchandles[hndl][myrank], 0 /*memOffset*/, aligned_size, 0));
        comm->memflags[hndl] |= UB_MEM_MC_CREATED;
        comm->mc_ptr[hndl] = reinterpret_cast<uint8_t*>(comm->mc_baseptr) + comm->mc_offset;
        comm->mc_offset += aligned_size;
    }
    else if (comm->myrank == 0)
    {
        TLLM_LOG_WARNING("[UserBuffer] warning region %d size %ld MB registered without MC access", hndl,
            aligned_size / 1024 / 1024);
    }
    comm->mem_size[hndl] = aligned_size;
    comm->mem_ptr[hndl] = *gpubuff;

    return comm->free_region++;
}
} // namespace tensorrt_llm::runtime::ub
