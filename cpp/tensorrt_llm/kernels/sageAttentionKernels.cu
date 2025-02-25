/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaFp8Utils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/sageAttentionKernels.h"
#include <cub/cub.cuh>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template void sage_quant<128, 64, 64, 256, __nv_bfloat16, __nv_fp8_e4m3, float>(
    // host var
    unsigned int batch_size, unsigned int head_num, unsigned int max_seq_len, bool smooth_k, bool is_padded,
    // device input
    void const* q, void const* k, void const* v, int const stride_q, int const stride_k, int const stride_v,
    int const* cu_seqlens_q, int const* cu_seqlens_kv, void* workspace,
    // device output
    void* quant_q, void* quant_k, void* quant_v, float* scales_q, float* scales_k, float* scales_v,
    cudaStream_t stream);

template <int HeadSize, int kThreadCount, int kTokensPerThreadBlock, typename T, typename TSmoothK>
__global__ void k_mean_kernel(bool const is_padded, int const max_seq_len, int const head_num, void const* k,
    int const stride_k, int const* cu_seqlens_kv, void* k_mean)
{
    int batch_id = blockIdx.y / head_num;
    int head_id = blockIdx.y % head_num;
    int channel_id = blockIdx.x * kThreadCount + threadIdx.x;

    if (channel_id >= HeadSize)
        return;

    int seq_start = cu_seqlens_kv[batch_id];
    int seq_len = cu_seqlens_kv[batch_id + 1] - seq_start;
    if (is_padded)
        seq_start = batch_id * max_seq_len;
    int seq_end = seq_start + seq_len;

    seq_start += blockIdx.z * kTokensPerThreadBlock;
    if (seq_start >= seq_end)
        return;

    seq_end = min(seq_start + kTokensPerThreadBlock, seq_end);

    float channel_mean = 0.f;

    for (int seq_id = seq_start; seq_id < seq_end; seq_id++)
    {
        T const* input = reinterpret_cast<T const*>(k) + seq_id * stride_k + head_id * HeadSize + channel_id;
        channel_mean += static_cast<float>(*input);
        input += stride_k;
    }

    channel_mean /= static_cast<float>(seq_len);

    TSmoothK* output
        = reinterpret_cast<TSmoothK*>(k_mean) + batch_id * head_num * HeadSize + head_id * HeadSize + channel_id;

    atomicAdd(output, channel_mean);
}

template <int HeadSize, int BlockSizeQ, int BlockSizeK, int BlockSizeV, typename T, typename TQuant, typename TSmooth>
__global__ void sage_quant_kernel(void const* q, void const* k, void const* v, int const stride_q, int const stride_k,
    int const stride_v, int const* cu_seqlens_q, int const* cu_seqlens_kv, void const* k_mean, int max_seq_len,
    bool smooth_k, bool is_padded,
    // output
    void* quant_q, void* quant_k, void* quant_v, float* scales_q, float* scales_k, float* scales_v)
{

    int batch_id = blockIdx.z;
    int head_id = blockIdx.y / 3;
    int qkv_id = blockIdx.y % 3;
    int qblock_id = blockIdx.x;

    constexpr int kElementsAccess = sizeof(float4) / sizeof(T);
    constexpr int tbDimx = 128 / sizeof(float4);
    constexpr int tbDimy = 128 / tbDimx;
    constexpr int tbIterx = HeadSize / tbDimx / kElementsAccess;
    int col_id = threadIdx.x % tbDimx;
    int row_id = threadIdx.x / tbDimx;

    if (qkv_id == 0)
    {
        // Q

        int seq_start = cu_seqlens_q[batch_id];
        int seq_end = cu_seqlens_q[batch_id + 1];

        if (seq_start + qblock_id * BlockSizeQ >= seq_end)
            return;

        if (is_padded)
        {
            int seq_len = seq_end - seq_start;
            seq_start = batch_id * max_seq_len;
            seq_end = seq_start + seq_len;
        }

        int seq_id = seq_start + qblock_id * BlockSizeQ + row_id;
        constexpr int tbItery = BlockSizeQ / tbDimy;

        T const* input
            = reinterpret_cast<T const*>(q) + seq_id * stride_q + head_id * HeadSize + col_id * kElementsAccess;

        T local_input[tbItery * tbIterx * kElementsAccess];
        T local_amax = T(0);

        int seq_id_ = seq_id;
        for (int y_ = 0; y_ < tbItery; y_++)
        {

            T* local_input_ptr = local_input + y_ * tbIterx * kElementsAccess;
            T const* input_ptr = input + y_ * tbDimy * stride_q;

            if (seq_id_ < seq_end)
            {
                for (int x_ = 0; x_ < tbIterx; x_++)
                {

                    *reinterpret_cast<float4*>(local_input_ptr) = *reinterpret_cast<float4 const*>(input_ptr);

                    for (int i = 0; i < kElementsAccess; i++)
                    {
                        T value = __habs(local_input_ptr[i]);
                        if (value > local_amax)
                            local_amax = value;
                    }

                    local_input_ptr += kElementsAccess;
                    input_ptr += tbDimx * kElementsAccess;
                }
            }
            else
            {
                break;
            }

            seq_id_ += tbDimy;
        }

        /// CUB block level max
        using BlockReduce = cub::BlockReduce<T, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float s_block_amax;

        // Compute the block-wide max for thread0
        // cuda::maximum<>{}
        float aggregate = BlockReduce(temp_storage).Reduce(local_amax, cub::Max{});

        if (row_id == 0 && col_id == 0)
            s_block_amax = static_cast<float>(aggregate);

        __syncthreads();

        float block_scale = s_block_amax / 448 + 1e-4;

        int max_qblock_per_seq = (max_seq_len + BlockSizeQ - 1) / BlockSizeQ;
        float* scales_q_ptr
            = scales_q + batch_id * (gridDim.y / 3) * max_qblock_per_seq + head_id * max_qblock_per_seq + qblock_id;
        *scales_q_ptr = block_scale;

        TQuant local_input_fp8[tbItery * tbIterx * kElementsAccess];

        for (int i = 0; i < tbItery * tbIterx * kElementsAccess; i++)
        {
            local_input_fp8[i] = static_cast<TQuant>(static_cast<float>(local_input[i]) / block_scale);
        }

        TQuant* output
            = reinterpret_cast<TQuant*>(quant_q) + seq_id * stride_q + head_id * HeadSize + col_id * kElementsAccess;

        for (int y_ = 0; y_ < tbItery; y_++)
        {

            TQuant* local_output_ptr = local_input_fp8 + y_ * tbIterx * kElementsAccess;
            TQuant* output_ptr = output + y_ * tbDimy * stride_q;

            if (seq_id >= seq_end)
                break;

            for (int x_ = 0; x_ < tbIterx; x_++)
            {

                *reinterpret_cast<float2*>(output_ptr) = *reinterpret_cast<float2*>(local_output_ptr);

                local_output_ptr += kElementsAccess;
                output_ptr += tbDimx * kElementsAccess;
            }

            seq_id += tbDimy;
        }
    }
    else if (qkv_id == 1)
    {
        // K

        int seq_start = cu_seqlens_kv[batch_id];
        int seq_end = cu_seqlens_kv[batch_id + 1];

        if (seq_start + qblock_id * BlockSizeK >= seq_end)
            return;

        if (is_padded)
        {
            int seq_len = seq_end - seq_start;
            seq_start = batch_id * max_seq_len;
            seq_end = seq_start + seq_len;
        }

        int seq_id = seq_start + qblock_id * BlockSizeK + row_id;
        constexpr int tbItery = BlockSizeK / tbDimy;

        T const* input
            = reinterpret_cast<T const*>(k) + seq_id * stride_k + head_id * HeadSize + col_id * kElementsAccess;

        TSmooth local_k_mean[tbIterx * kElementsAccess];

        if (smooth_k)
        {
            int head_num = gridDim.y / 3;
            TSmooth const* k_mean_ptr = reinterpret_cast<TSmooth const*>(k_mean) + batch_id * head_num * HeadSize
                + head_id * HeadSize + col_id * kElementsAccess;
            for (int x_ = 0; x_ < tbIterx; x_++)
            {
                for (int i = 0; i < sizeof(TSmooth) / sizeof(T); i++)
                {
                    *(reinterpret_cast<float4*>(local_k_mean + x_ * kElementsAccess) + i)
                        = *(reinterpret_cast<float4 const*>(k_mean_ptr) + i);
                }

                k_mean_ptr += tbDimx * kElementsAccess;
            }
        }

        T local_input[tbItery * tbIterx * kElementsAccess];
        T local_amax = T(0);

        int seq_id_ = seq_id;
        for (int y_ = 0; y_ < tbItery; y_++)
        {

            T* local_input_ptr = local_input + y_ * tbIterx * kElementsAccess;
            T const* input_ptr = input + y_ * tbDimy * stride_k;

            if (seq_id_ < seq_end)
            {
                for (int x_ = 0; x_ < tbIterx; x_++)
                {

                    *reinterpret_cast<float4*>(local_input_ptr) = *reinterpret_cast<float4 const*>(input_ptr);

                    for (int i = 0; i < kElementsAccess; i++)
                    {
                        if (smooth_k)
                        {
                            local_input_ptr[i] -= local_k_mean[x_ * kElementsAccess + i];
                        }

                        T value = __habs(local_input_ptr[i]);
                        if (value > local_amax)
                            local_amax = value;
                    }

                    local_input_ptr += kElementsAccess;
                    input_ptr += tbDimx * kElementsAccess;
                }
            }
            else
            {
                break;
            }

            seq_id_ += tbDimy;
        }

        /// CUB block level max
        using BlockReduce = cub::BlockReduce<T, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float s_block_amax;

        // Compute the block-wide max for thread0
        // cuda::maximum<>{}
        float aggregate = BlockReduce(temp_storage).Reduce(local_amax, cub::Max{});

        if (row_id == 0 && col_id == 0)
            s_block_amax = static_cast<float>(aggregate);

        __syncthreads();

        float block_scale = s_block_amax / 448 + 1e-4;

        int max_qblock_per_seq = (max_seq_len + BlockSizeK - 1) / BlockSizeK;

        float* scales_ptr
            = scales_k + batch_id * (gridDim.y / 3) * max_qblock_per_seq + head_id * max_qblock_per_seq + qblock_id;
        *scales_ptr = block_scale;

        TQuant* output
            = reinterpret_cast<TQuant*>(quant_k) + seq_id * stride_k + head_id * HeadSize + col_id * kElementsAccess;

        TQuant local_input_fp8[tbItery * tbIterx * kElementsAccess];

        for (int i = 0; i < tbItery * tbIterx * kElementsAccess; i++)
        {
            local_input_fp8[i] = static_cast<TQuant>(static_cast<float>(local_input[i]) / block_scale);
        }

        for (int y_ = 0; y_ < tbItery; y_++)
        {

            TQuant* local_output_ptr = local_input_fp8 + y_ * tbIterx * kElementsAccess;
            TQuant* output_ptr = output + y_ * tbDimy * stride_k;

            if (seq_id >= seq_end)
                break;

            for (int x_ = 0; x_ < tbIterx; x_++)
            {

                *reinterpret_cast<float2*>(output_ptr) = *reinterpret_cast<float2*>(local_output_ptr);

                local_output_ptr += kElementsAccess;
                output_ptr += tbDimx * kElementsAccess;
            }

            seq_id += tbDimy;
        }
    }
    else if (qkv_id == 2)
    {
        // V

        int seq_start = cu_seqlens_kv[batch_id];
        int seq_end = cu_seqlens_kv[batch_id + 1];

        if (seq_start + qblock_id * BlockSizeV >= seq_end)
            return;

        if (is_padded)
        {
            int seq_len = seq_end - seq_start;
            seq_start = batch_id * max_seq_len;
            seq_end = seq_start + seq_len;
        }

        int seq_id = seq_start + qblock_id * BlockSizeV + row_id;
        constexpr int tbItery = BlockSizeV / tbDimy;

        T const* input
            = reinterpret_cast<T const*>(v) + seq_id * stride_v + head_id * HeadSize + col_id * kElementsAccess;

        T local_input[tbItery * tbIterx * kElementsAccess];
        T local_amax = T(0);

        int seq_id_ = seq_id;
        for (int y_ = 0; y_ < tbItery; y_++)
        {

            T* local_input_ptr = local_input + y_ * tbIterx * kElementsAccess;
            T const* input_ptr = input + y_ * tbDimy * stride_v;

            if (seq_id_ < seq_end)
            {
                for (int x_ = 0; x_ < tbIterx; x_++)
                {

                    *reinterpret_cast<float4*>(local_input_ptr) = *reinterpret_cast<float4 const*>(input_ptr);

                    for (int i = 0; i < kElementsAccess; i++)
                    {
                        T value = __habs(local_input_ptr[i]);
                        if (value > local_amax)
                            local_amax = value;
                    }

                    local_input_ptr += kElementsAccess;
                    input_ptr += tbDimx * kElementsAccess;
                }
            }
            else
            {
                break;
            }

            seq_id_ += tbDimy;
        }

        /// CUB block level max
        using BlockReduce = cub::BlockReduce<T, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ float s_block_amax;

        // Compute the block-wide max for thread0
        // cuda::maximum<>{}
        float aggregate = BlockReduce(temp_storage).Reduce(local_amax, cub::Max{});

        if (row_id == 0 && col_id == 0)
            s_block_amax = static_cast<float>(aggregate);

        __syncthreads();

        float block_scale = s_block_amax / 448 + 1e-4;

        int max_qblock_per_seq = (max_seq_len + BlockSizeV - 1) / BlockSizeV;

        float* scales_ptr
            = scales_v + batch_id * (gridDim.y / 3) * max_qblock_per_seq + head_id * max_qblock_per_seq + qblock_id;
        *scales_ptr = block_scale;

        TQuant* output
            = reinterpret_cast<TQuant*>(quant_v) + seq_id * stride_v + head_id * HeadSize + col_id * kElementsAccess;

        TQuant local_input_fp8[tbItery * tbIterx * kElementsAccess];

        for (int i = 0; i < tbItery * tbIterx * kElementsAccess; i++)
        {
            local_input_fp8[i] = static_cast<TQuant>(static_cast<float>(local_input[i]) / block_scale);
        }

        for (int y_ = 0; y_ < tbItery; y_++)
        {

            TQuant* local_output_ptr = local_input_fp8 + y_ * tbIterx * kElementsAccess;
            TQuant* output_ptr = output + y_ * tbDimy * stride_v;

            if (seq_id >= seq_end)
                break;

            for (int x_ = 0; x_ < tbIterx; x_++)
            {

                *reinterpret_cast<float2*>(output_ptr) = *reinterpret_cast<float2*>(local_output_ptr);

                local_output_ptr += kElementsAccess;
                output_ptr += tbDimx * kElementsAccess;
            }

            seq_id += tbDimy;
        }
    }
}

template <int HeadSize, int BlockSizeQ, int BlockSizeK, int BlockSizeV, typename T, typename TQuant, typename TSmoothK>
void sage_quant(
    // host var
    unsigned int batch_size, unsigned int head_num, unsigned int max_seq_len, bool smooth_k, bool is_padded,
    // device input
    void const* q, void const* k, void const* v, int const stride_q, int const stride_k, int const stride_v,
    int const* cu_seqlens_q, int const* cu_seqlens_kv, void* workspace,
    // device output
    void* quant_q, void* quant_k, void* quant_v, float* scales_q, float* scales_k, float* scales_v, cudaStream_t stream)
{
    void* k_mean = workspace;

    if (smooth_k)
    {

        int const tokens_per_block = 1024;
        int const block = 128;
        dim3 grid((HeadSize + block - 1) / block, batch_size * head_num,
            (max_seq_len + tokens_per_block - 1) / tokens_per_block);

        cudaMemsetAsync(k_mean, 0, batch_size * head_num * HeadSize * sizeof(TSmoothK), stream);
        k_mean_kernel<HeadSize, block, tokens_per_block, T, TSmoothK>
            <<<grid, block, 0, stream>>>(is_padded, max_seq_len, head_num, k, stride_k, cu_seqlens_kv, k_mean);
    }

    constexpr int BlockSize_ = (BlockSizeQ > BlockSizeK) ? BlockSizeK : BlockSizeQ;
    constexpr int BlockSize = (BlockSizeV > BlockSize_) ? BlockSize_ : BlockSizeV;

    dim3 grid((max_seq_len + BlockSize - 1) / BlockSize, head_num * 3, batch_size);

    sage_quant_kernel<HeadSize, BlockSizeQ, BlockSizeK, BlockSizeV, T, TQuant, TSmoothK><<<grid, 128, 0, stream>>>(q, k,
        v, stride_q, stride_k, stride_v, cu_seqlens_q, cu_seqlens_kv, k_mean, max_seq_len, smooth_k, is_padded, quant_q,
        quant_k, quant_v, scales_q, scales_k, scales_v);
}

} // namespace kernels
} // namespace tensorrt_llm
