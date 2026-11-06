/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// The in-place BF16 QK norm + RoPE of torch.ops.trtllm.fused_qk_norm_rope (cpp/tensorrt_llm/kernels/
// fusedQKNormRopeKernel.cu, fusedQKNormRopeKernel with OutT = BF16, process_v = false, plain RoPE) launched
// as a programmatic dependent: every CTA runs griddepcontrol.wait before its first global read and then
// releases its own dependent (griddepcontrol.launch_dependents), so the kernel launches while its producer
// (the qkv projection) is still running and its consumer launches as soon as every CTA has passed the
// wait. The body is the shipped one statement for statement; built with the shipped library's flags (no
// --use_fast_math) it writes the shipped kernel's bits.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace musespark_open_qk_norm_rope_pdl
{

__device__ __forceinline__ void gridDependencyWait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void gridDependencyTrigger()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// tensorrt_llm::common::warpReduceSum<float>.
__device__ __forceinline__ float warpReduceSum(float val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = val + __shfl_xor_sync(0xffffffffu, val, mask, 32);
    return val;
}

// tensorrt_llm::common::packed_as<uint, N>.
template <int N>
struct PackedUint;

template <>
struct PackedUint<1>
{
    using type = uint;
};

template <>
struct PackedUint<2>
{
    using type = uint2;
};

template <>
struct PackedUint<4>
{
    using type = uint4;
};

template <int head_dim, bool interleave>
__global__ void qkNormRopePdlKernel(__nv_bfloat16* qkv, int const num_heads_q, int const num_heads_k,
    int const num_heads_v, int const rotary_dim, float const eps, __nv_bfloat16 const* q_weight,
    __nv_bfloat16 const* k_weight, float const base, int const* position_ids, int const num_tokens, float factor,
    float low, float high, float attention_factor, bool is_qk_norm, bool use_gemma)
{
    // Before any global read: qkv is the previous kernel's output. Every CTA (including the ones past the last
    // token) waits, so the dependent's launch implies the producer grid completed.
    gridDependencyWait();
    gridDependencyTrigger();

    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    int const total_qk_heads = num_heads_q + num_heads_k;
    int const total_proc_heads = total_qk_heads;

    int const tokenIdx = globalWarpIdx / total_proc_heads;
    int const localHeadIdx = globalWarpIdx % total_proc_heads;

    if (tokenIdx >= num_tokens)
        return;

    bool const isQ = localHeadIdx < num_heads_q;
    int headIdx;
    int segStart;
    if (isQ)
    {
        headIdx = localHeadIdx;
        segStart = 0;
    }
    else
    {
        headIdx = localHeadIdx - num_heads_q;
        segStart = num_heads_q * head_dim;
    }

    int const num_heads = num_heads_q + num_heads_k + num_heads_v;

    static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
    constexpr int numElemsPerThread = head_dim / 32;
    float elements[numElemsPerThread];
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(elemSizeBytes % 4 == 0, "numSizeBytes must be a multiple of 4");
    constexpr int vecSize = elemSizeBytes / 4;
    using vec_T = typename PackedUint<vecSize>::type;

    int const offsetWarp = tokenIdx * num_heads * head_dim + segStart + headIdx * head_dim;
    int offsetThread = offsetWarp + laneId * numElemsPerThread;

    float sumOfSquares = 0.0f;

    {
        vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
#pragma unroll
        for (int i = 0; i < vecSize; i++)
        {
            float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
            sumOfSquares += vals.x * vals.x;
            sumOfSquares += vals.y * vals.y;

            elements[2 * i] = vals.x;
            elements[2 * i + 1] = vals.y;
        }
    }

    if (is_qk_norm)
    {
        sumOfSquares = warpReduceSum(sumOfSquares);

        float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

        for (int i = 0; i < numElemsPerThread; i++)
        {
            int dim = laneId * numElemsPerThread + i;
            float weight = isQ ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
            elements[i] *= rms_rcp * (use_gemma ? (1.0f + weight) : weight);
        }
    }
    float elements2[numElemsPerThread];
    float cos_vals[numElemsPerThread];
    float sin_vals[numElemsPerThread];

    float const neg2_log2base_over_rd = -2.0f * __log2f(base) / static_cast<float>(rotary_dim);
    int const rd_mask = rotary_dim - 1;
    bool const rd_is_pow2 = ((rotary_dim & rd_mask) == 0);
    if constexpr (interleave)
    {
        for (int i = 0; i < numElemsPerThread; i++)
        {
            if (i % 2 == 0)
            {
                elements2[i] = -elements[i + 1];
            }
            else
            {
                elements2[i] = elements[i - 1];
            }

            int dim_idx = laneId * numElemsPerThread + i;
            int half_dim = dim_idx / 2;
            float freq = exp2f(static_cast<float>(half_dim) * neg2_log2base_over_rd);

            if (factor != 1.0f)
            {
                float inv_freq_extrapolation = freq;
                float inv_freq_interpolation = freq / factor;

                if (fabsf(low - high) <= 1e-6f)
                {
                    high += 0.001;
                }
                float linear_func = (static_cast<float>(half_dim) - low) / (high - low);
                float ramp_func = fmin(fmax(linear_func, 0.0f), 1.0f);
                float inv_freq_extrapolation_factor = 1.0f - ramp_func;
                freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor)
                    + inv_freq_extrapolation * inv_freq_extrapolation_factor;
            }

            float pos_id = static_cast<float>(position_ids[tokenIdx]);
            float theta = pos_id * freq;
            __sincosf(theta, &sin_vals[i], &cos_vals[i]);
        }
    }
    else
    {
        __syncwarp();
        int pairOffset = (rotary_dim / 2) / numElemsPerThread;
        for (int i = 0; i < numElemsPerThread; i++)
        {
            elements2[i] = __shfl_xor_sync(0xffffffff, elements[i], pairOffset);
            if (laneId < pairOffset)
            {
                elements2[i] = -elements2[i];
            }

            int dim_idx = laneId * numElemsPerThread + i;
            dim_idx = rd_is_pow2 ? ((dim_idx * 2) & rd_mask) : ((dim_idx * 2) % rotary_dim);
            int half_dim = dim_idx / 2;
            float freq = exp2f(static_cast<float>(half_dim) * neg2_log2base_over_rd);

            if (factor != 1.0f)
            {
                float inv_freq_extrapolation = freq;
                float inv_freq_interpolation = freq / factor;

                if (fabsf(low - high) <= 1e-6f)
                {
                    high += 0.001;
                }
                float linear_func = (static_cast<float>(half_dim) - low) / (high - low);
                float ramp_func = fmin(fmax(linear_func, 0.0f), 1.0f);
                float inv_freq_extrapolation_factor = 1.0f - ramp_func;
                freq = inv_freq_interpolation * (1.0f - inv_freq_extrapolation_factor)
                    + inv_freq_extrapolation * inv_freq_extrapolation_factor;
            }

            float pos_id = static_cast<float>(position_ids[tokenIdx]);
            float theta = pos_id * freq;
            __sincosf(theta, &sin_vals[i], &cos_vals[i]);
        }
        __syncwarp();
    }

    bool const is_full_rope = (rotary_dim == head_dim);
    if (is_full_rope)
    {
        for (int i = 0; i < numElemsPerThread; i++)
        {
            elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
        }
    }
    else
    {
        for (int i = 0; i < numElemsPerThread; i++)
        {
            int dim_idx = laneId * numElemsPerThread + i;

            if (dim_idx < rotary_dim)
            {
                elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
            }
        }
    }

    // storeHeadElements<__nv_bfloat16, numElemsPerThread, vecSize>.
    vec_T vec;
#pragma unroll
    for (int i = 0; i < vecSize; i++)
    {
        __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(elements[2 * i], elements[2 * i + 1]));
        reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<uint*>(&vec) + i)) = vals;
    }
    *reinterpret_cast<vec_T*>(&qkv[offsetThread]) = vec;
}

template <typename Kernel, typename... Args>
void launch(Kernel kernel, dim3 grid, dim3 block, bool pdl, Args... args)
{
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

template <int head_dim>
void dispatchInterleave(bool interleave, dim3 grid, dim3 block, bool pdl, __nv_bfloat16* qkv, int nq, int nk, int nv,
    int rotary_dim, float eps, __nv_bfloat16 const* qw, __nv_bfloat16 const* kw, float base, int const* pos,
    int num_tokens, float factor, float low, float high, float attention_factor, bool is_qk_norm, bool use_gemma)
{
    if (interleave)
    {
        launch(qkNormRopePdlKernel<head_dim, true>, grid, block, pdl, qkv, nq, nk, nv, rotary_dim, eps, qw, kw, base,
            pos, num_tokens, factor, low, high, attention_factor, is_qk_norm, use_gemma);
    }
    else
    {
        launch(qkNormRopePdlKernel<head_dim, false>, grid, block, pdl, qkv, nq, nk, nv, rotary_dim, eps, qw, kw, base,
            pos, num_tokens, factor, low, high, attention_factor, is_qk_norm, use_gemma);
    }
}

// torch.ops.trtllm.fused_qk_norm_rope's contract (in place on qkv, no mRoPE), plus pdl.
void run(at::Tensor qkv, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim,
    int64_t rotary_dim, double eps, at::Tensor q_weight, at::Tensor k_weight, double base, bool is_neox,
    at::Tensor position_ids, double factor, double low, double high, double attention_factor, bool is_qk_norm,
    bool use_gemma, bool pdl)
{
    TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous() && qkv.dim() == 2, "qkv must be a contiguous 2D CUDA tensor");
    TORCH_CHECK(qkv.scalar_type() == at::kBFloat16, "qkv must be BF16");
    TORCH_CHECK(qkv.size(1) == (num_heads_q + num_heads_k + num_heads_v) * head_dim, "qkv width mismatch");
    TORCH_CHECK(position_ids.is_cuda() && position_ids.is_contiguous() && position_ids.scalar_type() == at::kInt
            && position_ids.dim() == 1 && position_ids.size(0) == qkv.size(0),
        "position_ids must be a contiguous [num_tokens] int32 CUDA tensor");
    TORCH_CHECK(q_weight.scalar_type() == at::kBFloat16 && k_weight.scalar_type() == at::kBFloat16
            && q_weight.is_contiguous() && k_weight.is_contiguous() && q_weight.numel() == head_dim
            && k_weight.numel() == head_dim,
        "q_weight / k_weight must be contiguous [head_dim] BF16");
    TORCH_CHECK(rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim % 2 == 0, "bad rotary_dim");
    TORCH_CHECK(is_neox == false || (rotary_dim * 16) % head_dim == 0, "unsupported rotary_dim");
    TORCH_CHECK(factor != 1.0 || attention_factor == 1.0, "attention_factor without yarn");
    at::cuda::CUDAGuard guard(qkv.device());

    int const numTokens = static_cast<int>(qkv.size(0));
    if (numTokens == 0)
    {
        return;
    }
    constexpr int blockSize = 256;
    int const totalWarps = numTokens * static_cast<int>(num_heads_q + num_heads_k);
    dim3 grid((totalWarps + blockSize / 32 - 1) / (blockSize / 32));
    dim3 block(blockSize);
    auto* q = reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr());
    auto const* qw = reinterpret_cast<__nv_bfloat16 const*>(q_weight.data_ptr());
    auto const* kw = reinterpret_cast<__nv_bfloat16 const*>(k_weight.data_ptr());
    auto const* pos = position_ids.data_ptr<int>();
    bool const interleave = !is_neox;
    int const nq = static_cast<int>(num_heads_q);
    int const nk = static_cast<int>(num_heads_k);
    int const nv = static_cast<int>(num_heads_v);
    int const rd = static_cast<int>(rotary_dim);
    switch (head_dim)
    {
    case 64:
        dispatchInterleave<64>(interleave, grid, block, pdl, q, nq, nk, nv, rd, static_cast<float>(eps), qw, kw,
            static_cast<float>(base), pos, numTokens, static_cast<float>(factor), static_cast<float>(low),
            static_cast<float>(high), static_cast<float>(attention_factor), is_qk_norm, use_gemma);
        break;
    case 128:
        dispatchInterleave<128>(interleave, grid, block, pdl, q, nq, nk, nv, rd, static_cast<float>(eps), qw, kw,
            static_cast<float>(base), pos, numTokens, static_cast<float>(factor), static_cast<float>(low),
            static_cast<float>(high), static_cast<float>(attention_factor), is_qk_norm, use_gemma);
        break;
    case 256:
        dispatchInterleave<256>(interleave, grid, block, pdl, q, nq, nk, nv, rd, static_cast<float>(eps), qw, kw,
            static_cast<float>(base), pos, numTokens, static_cast<float>(factor), static_cast<float>(low),
            static_cast<float>(high), static_cast<float>(attention_factor), is_qk_norm, use_gemma);
        break;
    default: TORCH_CHECK(false, "unsupported head_dim ", head_dim);
    }
}

} // namespace musespark_open_qk_norm_rope_pdl

TORCH_LIBRARY(musespark_open_qk_norm_rope_pdl, m)
{
    m.def(
        "run(Tensor(a!) qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, int rotary_dim, "
        "float eps, Tensor q_weight, Tensor k_weight, float base, bool is_neox, Tensor position_ids, float factor, "
        "float low, float high, float attention_factor, bool is_qk_norm, bool use_gemma, bool pdl) -> ()");
}

TORCH_LIBRARY_IMPL(musespark_open_qk_norm_rope_pdl, CUDA, m)
{
    m.impl("run", &musespark_open_qk_norm_rope_pdl::run);
}
