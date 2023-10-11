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

#include "tensorrt_llm/plugins/ncclPlugin/FTCustomAR.h"
#include "NvInferRuntimeBase.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/mpiUtils.h"

namespace tensorrt_llm
{

CustomAllReduceComm::CustomAllReduceComm(size_t TPSize, size_t PPSize, int deviceId, size_t bufferSize)
    : mTPSize(TPSize)
    , mPPSize(PPSize)
    , mTPRank(deviceId % TPSize)
    , mPPRank(deviceId / TPSize)
    , mDeviceId(deviceId)
    , mBufferSize(bufferSize)
{
    if (mPPSize == 0)
    {
        group_comm_ = mpi::COMM_WORLD;
    }
    else
    {
        mpi::comm_split(mpi::COMM_WORLD, mPPRank, mTPRank, &group_comm_);
    }
    param_.barrier_flag = 0;
    // NOTE: assume All Reduce happens within the node (DGX A100)
    param_.ranks_per_node = mTPSize;
    param_.rank = mTPRank;
    param_.local_rank = mTPRank;
    param_.node_id = 0;
    allocate();
    IpcSyncMemHandle();
}

CustomAllReduceComm::~CustomAllReduceComm()
{
    if (is_ipc_handle_opened_)
    {
        IpcCloseMemHandle();
    }
    mpi::barrier(); // wait for others to stop using resources before freeing them
    if (mTPRank == 0)
    {
        for (int rank = 0; rank < mTPSize; rank++)
        {
            size_t device_id = mPPRank * mTPSize + rank;
            cudaSetDevice(device_id);
            cudaPointerAttributes comm_buffer_attributes, barrier_attributes;
            common::check_cuda_error(
                cudaPointerGetAttributes(&comm_buffer_attributes, param_.peer_comm_buffer_ptrs[rank]));
            common::check_cuda_error(cudaPointerGetAttributes(&barrier_attributes, param_.peer_barrier_ptrs[rank]));
            if (comm_buffer_attributes.type == 2)
            {
                common::check_cuda_error(cudaFree(param_.peer_comm_buffer_ptrs[rank]));
            }
            if (barrier_attributes.type == 2)
            {
                common::check_cuda_error(cudaFree(param_.peer_barrier_ptrs[rank]));
            }
        }
        cudaSetDevice(mDeviceId);
        setP2P(false);
    }
}

void CustomAllReduceComm::IpcGetMemHandle()
{
    for (int rank = 0; rank < mTPSize; rank++)
    {
        common::check_cuda_error(cudaIpcGetMemHandle(
            &(param_.ipc_mem_handles.peer_barrier_ipc_handles[rank]), param_.peer_barrier_ptrs[rank]));
        common::check_cuda_error(cudaIpcGetMemHandle(
            &(param_.ipc_mem_handles.peer_comm_buffer_ipc_handles[rank]), param_.peer_comm_buffer_ptrs[rank]));
    }
}

void CustomAllReduceComm::IpcSyncMemHandle()
{
    if (mTPRank == 0)
    {
        IpcGetMemHandle();
    }
    mpi::bcast(reinterpret_cast<char*>(&(param_.ipc_mem_handles)), sizeof(kernels::AllReduceIpcMemHandles),
        mpi::MPI_TYPE_CHAR, 0, group_comm_);
    if (mTPRank != 0)
    {
        IpcOpenMemHandle();
    }
    common::check_cuda_error(cudaSetDevice(mDeviceId));
}

void CustomAllReduceComm::IpcOpenMemHandle()
{
    if (is_ipc_handle_opened_)
    {
        IpcCloseMemHandle();
        is_ipc_handle_opened_ = false;
    }
    if (!is_ipc_handle_opened_)
    {
        for (int rank = 0; rank < mTPSize; rank++)
        {
            common::check_cuda_error(cudaIpcOpenMemHandle((void**) (&(param_.peer_barrier_ptrs[rank])),
                param_.ipc_mem_handles.peer_barrier_ipc_handles[rank], cudaIpcMemLazyEnablePeerAccess));
            common::check_cuda_error(cudaIpcOpenMemHandle((void**) (&(param_.peer_comm_buffer_ptrs[rank])),
                param_.ipc_mem_handles.peer_comm_buffer_ipc_handles[rank], cudaIpcMemLazyEnablePeerAccess));
        }
        param_.local_output_buffer_ptr = param_.peer_comm_buffer_ptrs[mTPRank];
        is_ipc_handle_opened_ = true;
    }
}

void CustomAllReduceComm::IpcCloseMemHandle()
{
    if (is_ipc_handle_opened_)
    {
        for (int rank = 0; rank < mTPSize; rank++)
        {
            common::check_cuda_error(cudaIpcCloseMemHandle(param_.peer_barrier_ptrs[rank]));
            common::check_cuda_error(cudaIpcCloseMemHandle(param_.peer_comm_buffer_ptrs[rank]));
        }
        is_ipc_handle_opened_ = false;
    }
}

void CustomAllReduceComm::customAllReduce(
    void* data, size_t elts, size_t size_per_elem, nvinfer1::DataType dataType, cudaStream_t stream)
{
    param_.local_output_buffer_ptr = data;
    param_.elts_total = elts;
    param_.barrier_flag = FLAG(param_.barrier_flag + 1);

    if (dataType == nvinfer1::DataType::kFLOAT)
    {
        using T = tensorrt_llm::CustomARCommTypeConverter<float>::Type;
        kernels::invokeOneOrTwoShotAllReduceKernel<T>(param_, stream);
    }
    else if (dataType == nvinfer1::DataType::kHALF)
    {
        using T = tensorrt_llm::CustomARCommTypeConverter<half>::Type;
        kernels::invokeOneOrTwoShotAllReduceKernel<T>(param_, stream);
    }
    else if (dataType == nvinfer1::DataType::kBF16)
    {
        using T = tensorrt_llm::CustomARCommTypeConverter<__nv_bfloat16>::Type;
        kernels::invokeOneOrTwoShotAllReduceKernel<T>(param_, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported dataType for customAllReduce");
    }
}

void CustomAllReduceComm::allocate()
{
    if (mTPRank != 0)
        return;
    setP2P();
    for (size_t i = 0; i < mTPSize; i++)
    {
        size_t device_id = mPPRank * mTPSize + i;
        common::check_cuda_error(cudaSetDevice(device_id));
        common::check_cuda_error(cudaMalloc(&(param_.peer_comm_buffer_ptrs[i]), mBufferSize));
        common::check_cuda_error(
            cudaMalloc(&(param_.peer_barrier_ptrs[i]), mTPSize * (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t)));
        common::check_cuda_error(
            cudaMemset(param_.peer_barrier_ptrs[i], 0, mTPSize * (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t)));
    }
    cudaSetDevice(mDeviceId);
}

bool CustomAllReduceComm::isAvailable()
{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11020
#else
    return false;
#endif

    if (!mpi::isInitialized())
    {
        return false;
    }

    auto worldSize = mpi::getCommWorldSize();
    auto rank = mpi::getCommWorldRank();
    if ((worldSize % 2 != 0) || (worldSize > MAX_RANKS_PER_NODE) || (worldSize == 0))
    {
        return false;
    }

    return true;
}

void CustomAllReduceComm::setP2P(bool activate)
{
    int peer_access_available = 0;
    size_t device_offset = mPPRank * mTPSize;
    for (int i = 0; i < mTPSize; i++)
    {
        cudaSetDevice(device_offset + i);
        for (int j = 0; j < mTPSize; j++)
        {
            if (i == j)
            {
                continue;
            }
            cudaDeviceCanAccessPeer(&peer_access_available, device_offset + i, device_offset + j);
            assert(peer_access_available);
            if (activate)
            {
                cudaDeviceEnablePeerAccess(device_offset + j, 0);
                cudaError_t result = cudaGetLastError();
                if (result == cudaErrorPeerAccessAlreadyEnabled)
                {
                    result = cudaSuccess;
                }
                common::check_cuda_error(result);
            }
            else
            {
                cudaDeviceDisablePeerAccess(device_offset + j);
            }
        }
    }
    cudaSetDevice(mDeviceId);
}

void* CustomAllReduceComm::getShareBuffer()
{
    return reinterpret_cast<void*>(param_.peer_comm_buffer_ptrs[mTPRank]);
}

} // namespace tensorrt_llm
