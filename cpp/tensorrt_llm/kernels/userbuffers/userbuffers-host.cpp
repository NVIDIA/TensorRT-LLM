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
#include "userbuffers.h"
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

namespace tensorrt_llm::runtime::ub
{
#define UB_ONESHOT_DEFAULT_VALUE 1

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
        CUresult retval = cmd;                                                                                         \
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
        ipcSocketResult_t r = cmd;                                                                                     \
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

void ub_alloc_copy_allgather(void** globaldata, size_t data_bytes, void* localdata, MPI_Comm c)
{
    int myrank, nranks;
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

int pipe_rank(communicator* comm, int step)
{
    int mynode = comm->myrank / comm->nvsize;
    int mylocal = comm->nvrank;
    int numlocal = comm->nvsize;

    int newlocal1 = mylocal + step * comm->ar_nvsize * comm->ar2_nvsize;
    int newlocal = (numlocal + (newlocal1 % numlocal)) % numlocal;
    int newnode = mynode;
    return newnode * numlocal + newlocal;
}

int create_communicator_grouped2(communicator** comm, int pipegpus, int pipenodes, int tensorgpus, int tensornodes)
{
    *comm = (communicator*) malloc(sizeof(communicator));

    int myrank, nranks, cur_dev, ndev;
    MPI_Comm_dup(MPI_COMM_WORLD, &(*comm)->comm_world);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    (*comm)->nranks = nranks;
    (*comm)->myrank = myrank;
    (*comm)->free_region = 0;
    (*comm)->launch_mode = LAUNCH_GPU | LAUNCH_CPU;
    (*comm)->pdl_launch = 0;

    cudaDeviceProp device_prop;
    TLLM_CUDA_CHECK(cudaGetDevice(&cur_dev));
    TLLM_CUDA_CHECK(cudaGetDeviceCount(&ndev));
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, cur_dev));
    (*comm)->sm_arch = device_prop.major;
    (*comm)->use_rr_kernel = device_prop.major == 8;
    if (getenv("OVERRIDERR"))
        (*comm)->use_rr_kernel = atoi(getenv("OVERRIDERR"));
    (*comm)->push = device_prop.major != 8;
    (*comm)->use_ce = getenv("USECE") ? 1 : 0;
    (*comm)->cga_size = getenv("CGASIZE")                     ? atoi(getenv("CGASIZE"))
        : (device_prop.major == 9 && !(*comm)->use_rr_kernel) ? 4
                                                              : 1;
    (*comm)->oneshot = getenv("UB_ONESHOT") ? atoi(getenv("UB_ONESHOT")) : UB_ONESHOT_DEFAULT_VALUE;

    if (ndev > 1)
    { // all visible devices
        if (cur_dev != myrank % ndev)
            printf("%d: device used %d[%d] ,resetting device to %d\n", myrank, cur_dev, ndev, myrank);
        TLLM_CUDA_CHECK(cudaSetDevice(myrank % ndev));
        cur_dev = myrank % ndev;
    }
    (*comm)->mydev = cur_dev;
    (*comm)->nvrank = myrank;
    (*comm)->nvsize = nranks;

    int divgpus = pipegpus * tensorgpus;
    int datagpus = nranks / divgpus;
    (*comm)->ar_nvsize = datagpus;
    (*comm)->ar_firstgpu = myrank - ((myrank / tensorgpus) % datagpus) * tensorgpus;
    (*comm)->ar_nvrank = (myrank - (*comm)->ar_firstgpu) / tensorgpus;
    // ar2 is tensor
    (*comm)->ar2_nvsize = tensorgpus;
    (*comm)->ar2_firstgpu = myrank - myrank % tensorgpus;
    (*comm)->ar2_nvrank = myrank - (*comm)->ar2_firstgpu;

    (*comm)->pipe_id = myrank / (datagpus * tensorgpus);
    MPI_Comm_dup(MPI_COMM_WORLD, &(*comm)->comm_intra);

    (*comm)->ibnvsize = getenv("IBNVSIZE") ? atoi(getenv("IBNVSIZE")) : (*comm)->nvsize;

#define NBUF 1

    int nvls_supported = 1;

    if (nvls_supported && (*comm)->sm_arch >= 9 && (*comm)->ar2_nvsize > 1 && !getenv("UB_SKIPMC"))
    { // multicast init only for TP ops (____2 operations)
        size_t mc_maxsize = MULTICAST_GB_TOTAL * (1ull << 30);
        (*comm)->mc_offset = 0;
        (*comm)->use_mc = 1;
        size_t gran;
        CUmulticastObjectProp mcProp = {};
        mcProp.numDevices = (*comm)->ar2_nvsize;
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
        if ((*comm)->ar2_nvrank == 0)
            CUCHECK(cuMulticastCreate(&(*comm)->mc_handle, &mcProp));

#ifdef MNNVL
        CUmemFabricHandle* exphndl = (CUmemFabricHandle*) malloc(sizeof(CUmemFabricHandle));
        CUmemFabricHandle* tmphndl = (CUmemFabricHandle*) malloc(sizeof(CUmemFabricHandle));
        if ((*comm)->ar2_nvrank == 0)
            CUCHECK(cuMemExportToShareableHandle(
                static_cast<void*>(tmphndl), (*comm)->mc_handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
        for (int grp = 0; grp < (*comm)->ar_nvsize; grp++)
        { // we do N broadcasts for N TP groups in NVL domain
            int root = grp * (*comm)->ar2_nvsize;
            ub_bcast(tmphndl, sizeof(CUmemFabricHandle), root, (*comm)->comm_intra);
            // save data if broadcast was from rank 0 in our group
            if ((*comm)->ar2_firstgpu == root)
                memcpy(exphndl, tmphndl, sizeof(CUmemFabricHandle));
        }
        if ((*comm)->ar2_nvrank != 0)
            CUCHECK(cuMemImportFromShareableHandle(
                &(*comm)->mc_handle, reinterpret_cast<void*>(exphndl), CU_MEM_HANDLE_TYPE_FABRIC));
        free(exphndl);
        free(tmphndl);
#else
        int fd;
        volatile uint32_t abortFlag = 0;
        struct IpcSocketHandle ipcSock = {0};
        srand(time(NULL));
        uint64_t opId = (uint64_t) (rand()) ^ ((uint64_t) (rand()) << 32);
        ub_bcast(&opId, sizeof(uint64_t), 0, (*comm)->comm_world);
        ipcSocketResult_t ret = ipcSocketSuccess;
        NCCLCHECK(ipcSocketInit(&ipcSock, (*comm)->ar2_nvrank + (*comm)->ar2_firstgpu, (uint64_t) opId, &abortFlag));
        ub_barrier((*comm)->comm_world);
        if ((*comm)->ar2_nvrank == 0)
        {
            CUCHECK(cuMemExportToShareableHandle(
                &fd, (*comm)->mc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));
            for (int p = 1; p < (*comm)->ar2_nvsize; p++)
            {
                ub_barrier((*comm)->comm_intra);
                NCCLCHECKGOTO(ipcSocketSendFd(&ipcSock, fd, p + (*comm)->ar2_firstgpu, (uint64_t) opId), ret, error);
            }
        }
        else
        {
            for (int i = 0; i < (*comm)->ar2_nvrank; i++)
                ub_barrier((*comm)->comm_intra);
            NCCLCHECKGOTO(ipcSocketRecvFd(&ipcSock, &fd), ret, error);
            for (int i = 0; i < (*comm)->ar2_nvsize - (*comm)->ar2_nvrank - 1; i++)
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
        if (!(*comm)->myrank)
            printf("MC initialized successfully, window size = %ld\n", mc_maxsize);
    }
    else
    {
        if (!(*comm)->myrank)
            printf("MC NOT initialized and used\n");
        (*comm)->mc_maxsize = 0;
        (*comm)->mc_offset = 0;
        (*comm)->use_mc = 0;
    }

#define LOCALSIZE 4 * (REG0_OFFSET(*comm) + REG0_FLAGS + REG0_COMMBUFFER * NBUF)
    // peer pointers + op flags + comm buffer
    register_user_buffer_collective(&((*comm)->gpu_ptrs), LOCALSIZE, *comm, true); // will use handler 0

    TLLM_CUDA_CHECK(cudaMalloc(&(*comm)->send_id, (*comm)->nranks * sizeof(int)));
    TLLM_CUDA_CHECK(cudaMalloc(&(*comm)->recv_id, MAX_REGIONS * (*comm)->nranks * sizeof(int)));
    TLLM_CUDA_CHECK(cudaMemset((*comm)->send_id, 0, (*comm)->nranks * sizeof(int)));
    TLLM_CUDA_CHECK(cudaMemset((*comm)->recv_id, 0, MAX_REGIONS * (*comm)->nranks * sizeof(int)));
    (*comm)->sms = getenv("MAXSMS") ? atoi(getenv("MAXSMS")) : 16;
    (*comm)->threads = getenv("MAXTHREADS") ? atoi(getenv("MAXTHREADS")) : 1024;

    return 0;
}

int create_communicator_grouped(communicator** comm, int pipegpus, int pipenodes)
{
    return create_communicator_grouped2(comm, pipegpus, pipenodes, 1, 1);
}

int create_communicator(communicator** comm)
{
    return create_communicator_grouped2(comm, 1, 1, 1, 1);
}

void destroy_communicator(communicator* comm)
{
    MPI_Comm_free(&comm->comm_intra);
    MPI_Comm_free(&comm->comm_world);
}

int register_user_buffer_collective(void** gpubuff, size_t bytes, communicator* comm, bool alloc)
{
    if (comm->free_region > MAX_REGIONS)
        return -1;
    int hndl = comm->free_region;
    comm->peer_ptr[hndl] = (void**) malloc(sizeof(void*) * (comm->nvsize));
    size_t aligned_size = bytes;
    comm->memflags[hndl] = 0;

    if (alloc)
    {
        int nranks = comm->nvsize; // total GPUs in NVLINK domain
        int myrank = comm->nvrank;
        void** remptrs = (void**) malloc(nranks * sizeof(void*));

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
        // MPI_Allreduce MAX of granularity check
        aligned_size = (bytes + granularity - 1) / granularity * granularity;

        if (comm->use_mc)
        {
            CUmulticastObjectProp mcProp = {};
            mcProp.numDevices = nranks;
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
        int* peerfd = (int*) malloc(nranks * sizeof(int));
        CUCHECK(cuMemExportToShareableHandle(
            &peerfd[myrank], comm->uchandles[hndl][myrank], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0 /*flags*/));

        volatile uint32_t abortFlag = 0;
        struct IpcSocketHandle ipcSock = {0};
        uint64_t opId = (uint64_t) (rand()) ^ ((uint64_t) (rand()) << 32);
        ub_bcast(&opId, sizeof(uint64_t), 0, comm->comm_world);
        ipcSocketResult_t ret = ipcSocketSuccess;

        NCCLCHECK(ipcSocketInit(&ipcSock, myrank, (uint64_t) opId, &abortFlag));
        for (int p = 1; p < nranks; p++)
        {
            ub_barrier(comm->comm_intra);
            NCCLCHECKGOTO(
                ipcSocketSendFd(&ipcSock, peerfd[myrank], (myrank + p) % nranks, (uint64_t) opId), ret, error);
            NCCLCHECKGOTO(ipcSocketRecvFd(&ipcSock, &peerfd[(myrank + nranks - p) % nranks]), ret, error);
        }
    error:
        NCCLCHECK(ipcSocketClose(&ipcSock));

        for (int p = 0; p < nranks; p++)
        {
            if (p != myrank)
                CUCHECK(cuMemImportFromShareableHandle(&comm->uchandles[hndl][p],
                    reinterpret_cast<void*>(static_cast<uintptr_t>(peerfd[p])),
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
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
        else if (!comm->myrank)
            printf(
                "UB: warning region %d size %ld MB registered without MC access\n", hndl, aligned_size / 1024 / 1024);
    }
    else
    {
        if (!comm->myrank)
            printf("UB: warning region %d size %ld MB allocated using cudaMalloc - deprecated(no MC available)\n", hndl,
                aligned_size / 1024 / 1024);
#ifdef MNNVL
        exit(1);
#endif
        // legacy/deprecated code for cudaMalloc user-allocated memory: might even print a warning about it
        assert(comm->nvsize <= 8); // FIXME check its single node
        cudaIpcMemHandle_t* memhndl;
        cudaIpcMemHandle_t myhndl;
        TLLM_CUDA_CHECK(cudaIpcGetMemHandle(&myhndl, *gpubuff));
        ub_alloc_copy_allgather((void**) &memhndl, sizeof(cudaIpcMemHandle_t), (void*) &myhndl, comm->comm_intra);

        for (int i = 0; i < comm->nvsize; i++)
            if (i != comm->nvrank)
                TLLM_CUDA_CHECK(cudaIpcOpenMemHandle(
                    (void**) &(comm->peer_ptr[hndl][i]), memhndl[i], cudaIpcMemLazyEnablePeerAccess));
        comm->peer_ptr[hndl][comm->nvrank] = *gpubuff;
        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        TLLM_CUDA_CHECK(cudaMemcpy((char*) (comm->gpu_ptrs) + (hndl * comm->nvsize * sizeof(void*)),
            comm->peer_ptr[hndl], comm->nvsize * sizeof(void*), cudaMemcpyHostToDevice));

        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        ub_free(memhndl);
    }
    comm->mem_size[hndl] = aligned_size;
    comm->mem_ptr[hndl] = *gpubuff;

    return comm->free_region++;
}
} // namespace tensorrt_llm::runtime::ub
