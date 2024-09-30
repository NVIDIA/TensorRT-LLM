/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_runtime_api.h>

#include "cumsumLastDim.h"

#include <cub/cub.cuh>

namespace tensorrt_llm
{
namespace kernels
{

///////////////

template <typename T>
size_t invokeComputeCumsumLastDimWorkspaceSize(SizeType32 inputLength)
{
    T* iodata = nullptr;
    size_t tempStorageBytes;
    cub::DeviceScan::InclusiveSum(nullptr, tempStorageBytes, iodata, iodata, inputLength);
    return tempStorageBytes;
}

#define INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(T)                                                 \
    template size_t invokeComputeCumsumLastDimWorkspaceSize<T>(int inputLength)

INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(int);
INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(float);
INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE

///////////////

template <typename T, int THREADS_PER_BLOCK, int ITEMS_PER_THREAD, cub::BlockScanAlgorithm ALGORITHM>
__global__ void cumsum_last_dim(T const* d_in, T* d_out, int length)
{
    typedef cub::BlockLoad<T, THREADS_PER_BLOCK, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    typedef cub::BlockStore<T, THREADS_PER_BLOCK, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
    typedef cub::BlockScan<T, THREADS_PER_BLOCK, ALGORITHM> BlockScanT;

    int const row_idx = blockIdx.x;
    T const* local_d_in = d_in + row_idx * length;
    T* local_d_out = d_out + row_idx * length;

    // Shared memory
    __shared__ union TempStorage
    {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
        typename BlockScanT::TempStorage scan;
    } temp_storage;

    int tile_size = THREADS_PER_BLOCK * ITEMS_PER_THREAD;
    T aggregate = static_cast<T>(0);
    T const* cur_d_in = local_d_in;
    T* cur_d_out = local_d_out;
    for (int tile_start = 0; tile_start < length;
         tile_start += tile_size, cur_d_in += tile_size, cur_d_out += tile_size)
    {
        int cur_tile_size = (tile_start + tile_size) <= length ? tile_size : (length - tile_start);
        T data[ITEMS_PER_THREAD]; // Per-thread tile data

        // Load items into a blocked arrangement
        BlockLoadT(temp_storage.load).Load(cur_d_in, data, cur_tile_size, static_cast<T>(0));
        if (threadIdx.x == 0)
        {
            data[0] += aggregate;
        }
        __syncthreads();

        BlockScanT(temp_storage.scan).InclusiveSum(data, data, aggregate);
        __syncthreads();

        // Store items from a blocked arrangement
        BlockStoreT(temp_storage.store).Store(cur_d_out, data, cur_tile_size);
        __syncthreads();
    }
}

///////////////

template <typename T>
void invokeDeviceScan(SizeType32 batchSize, SizeType32 inputLength, void const* __restrict__ input,
    void* __restrict__ output, void* d_temp_storage, size_t tempStorageBytes, cudaStream_t stream)
{
    for (SizeType32 i = 0; i < batchSize; i++)
    {
        T const* inputPtr = reinterpret_cast<T const*>(input) + i * inputLength;
        T* outputPtr = reinterpret_cast<T*>(output) + i * inputLength;
        cub::DeviceScan::InclusiveSum(d_temp_storage, tempStorageBytes, inputPtr, outputPtr, inputLength, stream);
    }
}

///////////////

template <typename T>
void invokeCumsumLastDim(SizeType32 batchSize, SizeType32 inputLength, void const* __restrict__ input,
    void* __restrict__ output, void* deviceTempStorage, size_t tempStorageBytes, cudaStream_t stream)
{
    // For empty tensor support
    if (batchSize == 0)
    {
        return;
    }

    if (deviceTempStorage != nullptr) // we need to use DeviceScan
    {
        invokeDeviceScan<T>(batchSize, inputLength, input, output, deviceTempStorage, tempStorageBytes, stream);
        return;
    }

    T const* inputPtr = reinterpret_cast<T const*>(input);
    T* outputPtr = reinterpret_cast<T*>(output);

    // Launch the kernel
    if (inputLength <= 64)
    {
        int const ITP = 1;
        int const TPB = 32;
        const size_t SHMEM = sizeof(T) * TPB * ITP;
        const cub::BlockScanAlgorithm ALG = cub::BLOCK_SCAN_WARP_SCANS;
        cumsum_last_dim<T, TPB, ITP, ALG><<<batchSize, TPB, SHMEM, stream>>>(inputPtr, outputPtr, inputLength);
    }
    else if (inputLength < 512)
    {
        int const ITP = 2;
        int const TPB = 64;
        const size_t SHMEM = sizeof(T) * TPB * ITP;
        const cub::BlockScanAlgorithm ALG = cub::BLOCK_SCAN_WARP_SCANS;
        cumsum_last_dim<T, TPB, ITP, ALG><<<batchSize, TPB, SHMEM, stream>>>(inputPtr, outputPtr, inputLength);
    }
    else // if ()
    {
        int const ITP = 8;
        int const TPB = 256;
        const size_t SHMEM = sizeof(T) * TPB * ITP;
        const cub::BlockScanAlgorithm ALG = cub::BLOCK_SCAN_WARP_SCANS;
        cumsum_last_dim<T, TPB, ITP, ALG><<<batchSize, TPB, SHMEM, stream>>>(inputPtr, outputPtr, inputLength);
    }
}

#define INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(T)                                                                        \
    template void invokeCumsumLastDim<T>(SizeType32 batchSize, SizeType32 inputLength, const void* __restrict__ input, \
        void* __restrict__ output, void* workspace, size_t tempStorageBytes, cudaStream_t stream)

INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(int);
INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(float);
INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_CUMSUM_LastDim_DATA_TYPE

} // namespace kernels
} // namespace tensorrt_llm
