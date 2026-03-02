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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/kernels/arcquantFP4.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace
{

#define FP4_MAX 6
#define SCALE_EPS 0.001953125f
#define GROUP_NUM(x) ((x) / 16)

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// PTX-based vectorized FP4 quantization - quantize only using hardware instructions
// Converts 4 floats to 4 e2m1 values (packed into uint16_t) using PTX
inline __device__ uint16_t fp32_vec4_to_e2m1(float (&scaled_inputs)[4])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint16_t packed_e2m1;

    // Use PTX hardware cvt.rn.satfinite.e2m1x2.f32 instruction (bypasses ALU pipeline)
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1;\n"

        // Quantize: 4 scaled floats -> 4 e2m1 (2 e2m1 values per byte)
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"

        // Pack 2 bytes into uint16_t output
        "mov.b16 %0, {byte0, byte1};\n"
        "}"
        : "=h"(packed_e2m1)
        : "f"(scaled_inputs[0]), "f"(scaled_inputs[1]), "f"(scaled_inputs[2]), "f"(scaled_inputs[3]));

    return packed_e2m1;
#else
    return 0;
#endif
}

// PTX-based vectorized FP4 quantization - quantize only using hardware instructions
// Converts 4 e2m1 values (packed into uint16_t) to 4 floats using PTX
inline __device__ float4 e2m1_to_float(uint16_t const& packed_e2m1)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t out_fp16[2];
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1;\n"
        "mov.b16 {byte0, byte1}, %2;\n"
        "cvt.rn.f16x2.e2m1x2 %0, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, byte1;\n"
        "}\n"
        : "=r"(out_fp16[0]), "=r"(out_fp16[1])
        : "h"(packed_e2m1));

    float2 res0 = __half22float2(reinterpret_cast<__half2&>(out_fp16[0]));
    float2 res1 = __half22float2(reinterpret_cast<__half2&>(out_fp16[1]));
    return {res0.x, res0.y, res1.x, res1.y};
#else
    return {0.0f, 0.0f, 0.0f, 0.0f};
#endif
}

__forceinline__ __device__ int64_t get_sf_offset(int row_id, int pos, int K)
{
    int64_t sf_offset = 0;
    sf_offset += (row_id % 32) * 16;
    sf_offset += ((row_id / 32) % 4) * 4;
    sf_offset += (row_id / 128) * (32 * 16 * K / 64);
    sf_offset += (pos % 4) * 1;
    sf_offset += (pos / 4) * 512;
    return sf_offset;
}

} // namespace

namespace kernels
{

// Modified from ARCQuant.
template <typename T, int GROUP_SIZE, ArcQuantType arcquant_type>
__global__ void quantize_reorder_nvfp4_kernel(
    T* hidden_states, float* input_scale, int16_t* reorder_index, uint8_t* q_out, uint8_t* q_scale, int KQ, int KE)
{
    int const hidden_dim = KQ;
    int const K = KQ + KE;
    int const bdx = hidden_dim / GROUP_SIZE;
    constexpr int elements_per_thread = GROUP_SIZE;

    T* input = reinterpret_cast<T*>(hidden_states);
    __nv_fp8_e4m3* q_scale_tensor = reinterpret_cast<__nv_fp8_e4m3*>(q_scale);
    // One block solves one row of hidden states.
    extern __shared__ uint8_t smem[];
    T* input_smem = reinterpret_cast<T*>(smem);
    // Local memory stores the reordered hidden states.
    __nv_bfloat16 input_frag[elements_per_thread];
    int8_t output_frag[elements_per_thread];
    // Row are independent
    int row_id = blockIdx.x;
    input = input + row_id * hidden_dim;
    q_out = q_out + row_id * K / 2;
    // Load input scale for FP8 dequantization
    float global_scale = 1.0f;
    // FP8_Scale = FP4_Scale / 6.0
    // input = input / FP8_Scale * FP4_Scale
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        global_scale = 6.0f;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        global_scale = *input_scale;
    }
    // Coalesced access global memory
    int tid = threadIdx.x;
    int const bytes_per_iter = bdx * sizeof(float4);
    int const iters = hidden_dim * sizeof(T) / bytes_per_iter;

    for (int i = 0; i < iters; ++i)
    {
        // Each thread loads 16 bytes
        int offset = i * bytes_per_iter + tid * sizeof(float4);
        *(float4*) (reinterpret_cast<uint8_t*>(input_smem) + offset)
            = *(float4*) (reinterpret_cast<uint8_t*>(input) + offset);
    }
    __syncthreads();
    // Reorder and convert to BF16

    for (int i = 0; i < elements_per_thread; ++i)
    {
        int offset = tid * elements_per_thread + i;
        // Convert to BF16 and apply FP8 scale if needed
        input_frag[i] = __float2bfloat16_rn((float) input_smem[reorder_index[offset]] * global_scale);
    }
    // Reduce to get max
    float maxv = 0, scale = 1.0, r_scale = 1.0;

    for (int i = 0; i < elements_per_thread; ++i)
    {
        maxv = cuda_max(maxv, __bfloat162float(cuda_abs(input_frag[i])));
    }
    // Q quantize
    scale = cuda_max(maxv / FP4_MAX, SCALE_EPS);
    int pos = tid + max(0, tid - GROUP_NUM(KQ - KE));
    int64_t sf_offset = get_sf_offset(row_id, pos, K);
    __nv_fp8_e4m3 scale_ue4m3;
    scale_ue4m3.__x = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
    q_scale_tensor[sf_offset] = scale_ue4m3;
    // Use reverse scale to replace division by multiplication
    float qdq_scale = (float) scale_ue4m3;
    r_scale = reciprocal_approximate_ftz(qdq_scale);
    // Quantize each thread's value using PTX hardware instructions
    // Each iteration processes 4 elements using vectorized PTX operations
    for (int i = 0; i < elements_per_thread; i += 4)
    {
        // Prepare scaled inputs for quantization
        float scaled_inputs[4];
        scaled_inputs[0] = __bfloat162float(input_frag[i + 0]) * r_scale;
        scaled_inputs[1] = __bfloat162float(input_frag[i + 1]) * r_scale;
        scaled_inputs[2] = __bfloat162float(input_frag[i + 2]) * r_scale;
        scaled_inputs[3] = __bfloat162float(input_frag[i + 3]) * r_scale;

        // PTX-based quantization: converts 4 floats -> 4 e2m1 using hardware instruction
        // Uses cvt.rn.satfinite.e2m1x2.f32 which bypasses ALU pipeline
        uint16_t packed_e2m1 = fp32_vec4_to_e2m1(scaled_inputs);

        // Dequantize e2m1 to float and compute residuals using PTX instructions
        float4 e2m1_float = e2m1_to_float(packed_e2m1);
        input_frag[i + 0] = __float2bfloat16_rn(__bfloat162float(input_frag[i + 0]) - e2m1_float.x * qdq_scale);
        input_frag[i + 1] = __float2bfloat16_rn(__bfloat162float(input_frag[i + 1]) - e2m1_float.y * qdq_scale);
        input_frag[i + 2] = __float2bfloat16_rn(__bfloat162float(input_frag[i + 2]) - e2m1_float.z * qdq_scale);
        input_frag[i + 3] = __float2bfloat16_rn(__bfloat162float(input_frag[i + 3]) - e2m1_float.w * qdq_scale);

        reinterpret_cast<uint16_t*>(output_frag)[i / 4] = packed_e2m1;
    }
    int const ke_thread_count = GROUP_NUM(KE);
    int const kq_thread_count = bdx - ke_thread_count;
    if (tid >= kq_thread_count)
    {
        if constexpr (arcquant_type == ArcQuantType::ACT)
        {
            maxv = 0;

            for (int i = 0; i < elements_per_thread; ++i)
            {
                maxv = cuda_max(maxv, __bfloat162float(cuda_abs(input_frag[i])));
            }
            scale = cuda_max(maxv / FP4_MAX, SCALE_EPS);
            sf_offset = get_sf_offset(row_id, pos + 1, K);
            __nv_fp8_e4m3 scale_ue4m3_res;
            scale_ue4m3_res.__x = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
            q_scale_tensor[sf_offset] = scale_ue4m3_res;
            r_scale = reciprocal_approximate_ftz((float) scale_ue4m3_res);
            for (int i = 0; i < elements_per_thread; i += 4)
            {
                // Prepare scaled residuals for quantization
                float scaled_inputs[4];
                scaled_inputs[0] = __bfloat162float(input_frag[i + 0]) * r_scale;
                scaled_inputs[1] = __bfloat162float(input_frag[i + 1]) * r_scale;
                scaled_inputs[2] = __bfloat162float(input_frag[i + 2]) * r_scale;
                scaled_inputs[3] = __bfloat162float(input_frag[i + 3]) * r_scale;

                // PTX-based quantization of residuals
                uint16_t packed_e2m1 = fp32_vec4_to_e2m1(scaled_inputs);
                reinterpret_cast<uint16_t*>(output_frag)[(i + elements_per_thread) / 4] = packed_e2m1;
            }
        }
        else if constexpr (arcquant_type == ArcQuantType::WEIGHT)
        {
            sf_offset = get_sf_offset(row_id, pos + 1, K);
            q_scale_tensor[sf_offset] = scale_ue4m3;

            for (int i = 0; i < elements_per_thread; i += 4)
            {
                reinterpret_cast<uint16_t*>(output_frag)[(i + elements_per_thread) / 4]
                    = reinterpret_cast<uint16_t*>(output_frag)[i / 4];
            }
        }

        int const kq_region_bytes = kq_thread_count * 8;
        int const ke_thread_idx = tid - kq_thread_count;
        int const ke_thread_offset = kq_region_bytes + ke_thread_idx * 16;

        float4* q_out_ptr = reinterpret_cast<float4*>(q_out + ke_thread_offset);
        *q_out_ptr = *(reinterpret_cast<float4*>(output_frag));
    }
    else
    {
        float2* q_out_ptr = reinterpret_cast<float2*>(q_out + tid * 8);
        *q_out_ptr = *(reinterpret_cast<float2*>(output_frag));
    }
}

template <typename T, int GROUP_SIZE, ArcQuantType arcquant_type>
void run_quantize_reorder_nvfp4(int16_t* hidden_states, float* input_scale, int16_t* reorder_index, uint8_t* q_out,
    uint8_t* q_scale, int seq_len, int KQ, int KE, cudaStream_t stream)
{
    int hidden_dim = KQ;
    dim3 grids(seq_len);
    dim3 blocks(hidden_dim / GROUP_SIZE);
    size_t smem_size = hidden_dim * sizeof(T);
    quantize_reorder_nvfp4_kernel<T, GROUP_SIZE, arcquant_type>
        <<<grids, blocks, smem_size, stream>>>((T*) hidden_states, input_scale, reorder_index, q_out, q_scale, KQ, KE);
}

// Explicit template instantiation for the specific types used
template void run_quantize_reorder_nvfp4<__nv_bfloat16, 16, ArcQuantType::ACT>(int16_t* hidden_states,
    float* input_scale, int16_t* reorder_index, uint8_t* q_out, uint8_t* q_scale, int seq_len, int KQ, int KE,
    cudaStream_t stream);

template void run_quantize_reorder_nvfp4<__nv_fp8_e4m3, 16, ArcQuantType::ACT>(int16_t* hidden_states,
    float* input_scale, int16_t* reorder_index, uint8_t* q_out, uint8_t* q_scale, int seq_len, int KQ, int KE,
    cudaStream_t stream);

template void run_quantize_reorder_nvfp4<__nv_bfloat16, 16, ArcQuantType::WEIGHT>(int16_t* hidden_states,
    float* input_scale, int16_t* reorder_index, uint8_t* q_out, uint8_t* q_scale, int seq_len, int KQ, int KE,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
