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

#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>

#include "convertSpecDecodingMaskToPackedMask.h"

namespace tensorrt_llm
{
namespace kernels
{

size_t invokeScanSpecDecodingGenerationLenghtsTempStorageBytes(int batch_size, cudaStream_t stream)
{
    int* iodata = nullptr;
    size_t temp_storage_bytes;
    cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, iodata, iodata, batch_size, stream);
    return temp_storage_bytes;
}

size_t invokeReduceMaxSpecDecodingGenerationLengthsTempStorageBytes(int batch_size, cudaStream_t stream)
{
    int* iodata = nullptr;
    size_t reduce_temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr, reduce_temp_storage_bytes, iodata, iodata, batch_size, stream);
    return reduce_temp_storage_bytes;
}

// inclusive prefix sum spec_decoding_generation_lengths and reduce max spec_decoding_generation_lengths
void invokeScanSpecDecodingGenerationLenghths(int batch_size, int* __restrict__ const spec_decoding_generation_lengths,
    void* __restrict__ scan_temp_storage, size_t scan_temp_storage_bytes,
    int* __restrict__ scaned_spec_decoding_generation_lengths, void* __restrict__ reduce_max_temp_storage,
    size_t reduce_max_temp_storage_bytes, int* max_spec_decoding_generation_lengths, cudaStream_t stream)
{
    cub::DeviceScan::InclusiveSum(scan_temp_storage, scan_temp_storage_bytes, spec_decoding_generation_lengths,
        scaned_spec_decoding_generation_lengths, batch_size, stream);
    cub::DeviceReduce::Max(reduce_max_temp_storage, reduce_max_temp_storage_bytes, spec_decoding_generation_lengths,
        max_spec_decoding_generation_lengths, batch_size, stream);
}

////////////////////////

__device__ int myPositivePowerOfTwo(int n)
{
    if (n == 0)
        return 1;
    if (n == 1)
        return 2;
    int res = 1;
    int i = n;
    int x = 2;
    while (i)
    {
        if (i & 0x1)
        {
            res *= x;
        }
        x *= x;
        i >>= 1;
    }
    return res;
}

__global__ void get_spec_decoding_packed_mask(int const* __restrict__ spec_decoding_cum_generation_lengths,
    bool const* __restrict__ spec_decoding_mask, int max_draft_tokens, int max_generation_length,
    int* __restrict__ spec_decoding_packed_mask)
{
    int batch_id = blockIdx.y;
    int token_id = blockIdx.x;

    int num_tokens = batch_id == 0
        ? spec_decoding_cum_generation_lengths[0]
        : spec_decoding_cum_generation_lengths[batch_id] - spec_decoding_cum_generation_lengths[batch_id - 1];
    if (token_id >= num_tokens)
        return;
    int num_packed_masks = (max_draft_tokens + 1 + 31) / 32;
    int output_start_id = batch_id == 0 ? 0 : spec_decoding_cum_generation_lengths[batch_id - 1];
    int* output_ptr = spec_decoding_packed_mask + (output_start_id + token_id) * num_packed_masks;

    if (token_id == 0)
    {
        for (int mask_id = threadIdx.x; mask_id < num_packed_masks; mask_id += blockDim.x)
        {
            output_ptr[mask_id] = mask_id == 0 ? 1 : 0;
        }
        return;
    }
    else
    {
        bool const* spec_decoding_mask_ptr = spec_decoding_mask
            + batch_id * max_generation_length * max_generation_length + token_id * max_generation_length + 1;
        extern __shared__ char sh_spec_decoding_mask[];
        if (threadIdx.x == 0)
            sh_spec_decoding_mask[max_generation_length - 1] = '1';
        for (int i = threadIdx.x; i < max_generation_length - 1; i += blockDim.x)
        {
            int sh_index = max_generation_length - 1 - i - 1;
            sh_spec_decoding_mask[sh_index] = spec_decoding_mask_ptr[i] ? '1' : '0';
        }
        __syncthreads();
        for (int mask_id = threadIdx.x; mask_id < num_packed_masks; mask_id += blockDim.x)
        {
            if (mask_id * 32 >= max_generation_length)
            {
                output_ptr[mask_id] = 0;
                return;
            }
            else
            {
                int sh_spec_decoding_mask_index_start
                    = (max_generation_length - (mask_id + 1) * 32) < 0 ? 0 : max_generation_length - (mask_id + 1) * 32;
                int sh_spec_decoding_mask_index_end = max_generation_length - (mask_id * 32 + 1) + 1;

                int valid_num_bits = sh_spec_decoding_mask_index_end - sh_spec_decoding_mask_index_start;
                bool first_bit1 = sh_spec_decoding_mask[sh_spec_decoding_mask_index_start] == '1' ? true : false;
                int mask_31bits = 0;
                if (valid_num_bits != 1)
                {
                    for (int i = sh_spec_decoding_mask_index_start + 1; i < sh_spec_decoding_mask_index_end; i++)
                    {
                        int index = (valid_num_bits - 1) - (i - sh_spec_decoding_mask_index_start - 1) - 1;
                        mask_31bits += sh_spec_decoding_mask[i] == '1' ? myPositivePowerOfTwo(index) : 0;
                    }
                }
                int mask_32bits;
                if (valid_num_bits == 32)
                {
                    mask_32bits = first_bit1 ? mask_31bits - myPositivePowerOfTwo(valid_num_bits - 1) : mask_31bits;
                }
                else
                {
                    mask_32bits = first_bit1 ? mask_31bits + myPositivePowerOfTwo(valid_num_bits - 1) : mask_31bits;
                }
                output_ptr[mask_id] = mask_32bits;
            }
        }
    }
}

void invokeConvertSpecDecodingMaskToPackedMask(int batch_size,
    int const* __restrict__ spec_decoding_cum_generation_lengths, bool const* __restrict__ spec_decoding_mask,
    int max_draft_tokens, int max_generation_length, int* __restrict__ spec_decoding_packed_mask, cudaStream_t stream)
{
    dim3 block(32);
    dim3 grid(max_generation_length, batch_size);
    size_t shm_size = max_generation_length * sizeof(char);
    get_spec_decoding_packed_mask<<<grid, block, shm_size, stream>>>(spec_decoding_cum_generation_lengths,
        spec_decoding_mask, max_draft_tokens, max_generation_length, spec_decoding_packed_mask);
}

} // namespace kernels
} // namespace tensorrt_llm
