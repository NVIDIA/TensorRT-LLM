/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "fused_qk_norm_rope_kernel.h"
#include <stdio.h>
#include <string.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

namespace llm::common
{
// Primary template declaration for packed_as.
template <typename T, int N>
struct packed_as;

// Specialization for packed_as used in this kernel.
template <>
struct packed_as<uint, 1>
{
    using type = uint;
};

template <>
struct packed_as<uint, 2>
{
    using type = uint2;
};

template <>
struct packed_as<uint, 4>
{
    using type = uint4;
};

inline int divUp(int a, int b)
{
    return (a + b - 1) / b;
}

template <typename T>
inline __device__ T add(T a, T b)
{
    return a + b;
}

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = add<T>(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32)); //__shfl_sync bf16 return float when sm < 80
    return val;
}

} // namespace llm::common

////////////////////////////////////////////////////////////////////////////////////////////////////

// Perform per-head QK Norm and RoPE in a single kernel.
// head_dim: the dimension of each head
// interleave: interleave=!is_neox.
template <int head_dim, bool interleave>
__global__ void fusedQKNormRopeKernel(
    __nv_bfloat16* qkv,            // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int const num_heads_q,         // Number of query heads
    int const num_heads_k,         // Number of key heads
    int const num_heads_v,         // Number of value heads
    float const eps,               // Epsilon for RMS normalization
    __nv_bfloat16 const* q_weight, // RMSNorm weights for query
    __nv_bfloat16 const* k_weight, // RMSNorm weights for key
    __nv_bfloat16 const* q_add_weight, // RMSNorm weights for query
    __nv_bfloat16 const* k_add_weight, // RMSNorm weights for key
    float const* cos_emb,           // RoPE cos embeddings [num_tokens, head_dim] - float32, range [0, 1]
    float const* sin_emb,           // RoPE sin embeddings [num_tokens, head_dim] - float32, range [0, 1]
    int const num_tokens,             // Number of tokens
    int const seqlen_per_bs         // bs=1, the len of seq.
)
{
    int const seqlenBS1 = seqlen_per_bs; //4608
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;

    // Calculate global warp index to determine which head/token this warp processes
    int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;

    // Total number of attention heads (Q and K)
    int const total_qk_heads = num_heads_q + num_heads_k;

    // Determine which token and head type (Q or K) this warp processes
    int const tokenIdx = globalWarpIdx / total_qk_heads;
    int const localHeadIdx = globalWarpIdx % total_qk_heads;

    // Skip if this warp is assigned beyond the number of tokens
    if (tokenIdx >= num_tokens)
        return;

    bool const isQ = localHeadIdx < num_heads_q;
    int const headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;

    int const num_heads = num_heads_q + num_heads_k + num_heads_v;

    static_assert(head_dim % (32 * 2) == 0,
        "head_dim must be divisible by 64 (each warp processes one head, and each thread gets even number of "
        "elements)");
    constexpr int numElemsPerThread = head_dim / 32;
    float elements[numElemsPerThread];
    constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
    static_assert(elemSizeBytes % 4 == 0, "numSizeBytes must be a multiple of 4");
    constexpr int vecSize = elemSizeBytes / 4;
    using vec_T = typename llm::common::packed_as<uint, vecSize>::type;

    int offsetWarp; // Offset for the warp
    if (isQ)
    {
        // Q segment: token offset + head offset within Q segment
        offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    }
    else
    {
        // K segment: token offset + entire Q segment + head offset within K segment
        offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
    }
    int offsetThread = offsetWarp + laneId * numElemsPerThread;

    // Sum of squares for RMSNorm
    float sumOfSquares = 0.0f;

    // Load.
    {
        vec_T vec = *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
        for (int i = 0; i < vecSize; i++)
        {
            float2 vals = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
            sumOfSquares += vals.x * vals.x;
            sumOfSquares += vals.y * vals.y;

            elements[2 * i] = vals.x;
            elements[2 * i + 1] = vals.y;
        }
    }

    // Reduce sum across warp using the utility function
    sumOfSquares = llm::common::warpReduceSum(sumOfSquares);

    // Compute RMS normalization factor
    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

    // Normalize elements
    for (int i = 0; i < numElemsPerThread; i++)
    {
        int dim = laneId * numElemsPerThread + i;

        float weight=0.f;
        int localtokenIdx = tokenIdx % seqlenBS1;
        if (localtokenIdx < 512 )
            if (q_add_weight == nullptr)
                weight = isQ ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
            else
                weight = isQ ? __bfloat162float(q_add_weight[dim]) : __bfloat162float(k_add_weight[dim]);
        else
            weight = isQ ? __bfloat162float(q_weight[dim]) : __bfloat162float(k_weight[dim]);
        elements[i] *= rms_rcp * weight;
    }

    // Apply RoPE to normalized elements
    float elements2[numElemsPerThread];
    float cos_vals[numElemsPerThread];
    float sin_vals[numElemsPerThread];

    // TODO: cos sin calculation could be halved.
    if constexpr (interleave)
    {
        // Perform interleaving. Fill cos_vals and sin_vals.
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
            int localtokenIdx = tokenIdx % seqlenBS1;
            int dim_idx = localtokenIdx * 128 + laneId * numElemsPerThread + i;
            sin_vals[i] = sin_emb[dim_idx];
            cos_vals[i] = cos_emb[dim_idx];
        }
    }

    for (int i = 0; i < numElemsPerThread; i++)
    {
        elements[i] = elements[i] * cos_vals[i] + elements2[i] * sin_vals[i];
    }

    // Store.
    {
        vec_T vec;
        for (int i = 0; i < vecSize; i++)
        {
            __nv_bfloat162 vals = __float22bfloat162_rn(make_float2(elements[2 * i], elements[2 * i + 1]));
            reinterpret_cast<__nv_bfloat162&>(*(reinterpret_cast<uint*>(&vec) + i)) = vals;
        }
        vec_T* outputPtr = reinterpret_cast<vec_T*>(&qkv[offsetThread]);
        *outputPtr = vec;
    }
}

// Borrowed from
// https://github.com/flashinfer-ai/flashinfer/blob/8125d079a43e9a0ba463a4ed1b639cefd084cec9/include/flashinfer/pos_enc.cuh#L568
#define DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...)                                                               \
    if (interleave)                                                                                                    \
    {                                                                                                                  \
        const bool INTERLEAVE = true;                                                                                  \
        __VA_ARGS__                                                                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        const bool INTERLEAVE = false;                                                                                 \
        __VA_ARGS__                                                                                                    \
    }

void launchFusedQKNormRope(void* qkv, int const num_tokens, int const seqlen_per_bs, int const num_heads_q, int const num_heads_k,
    int const num_heads_v, int const head_dim, float const eps, void const* q_weight, void const* k_weight, void const* q_add_weight, void const* k_add_weight,
    float const* cos_emb, float const* sin_emb, cudaStream_t stream)
{
    constexpr int blockSize = 256; //512; //

    int const warpsPerBlock = blockSize / 32;
    int const totalQKHeads = num_heads_q + num_heads_k;
    int const totalWarps = num_tokens * totalQKHeads;

    int const gridSize = llm::common::divUp(totalWarps, warpsPerBlock);
    dim3 gridDim(gridSize);
    dim3 blockDim(blockSize);

    switch (head_dim)
    {
    case 64:
        DISPATCH_INTERLEAVE(true, INTERLEAVE, {
            fusedQKNormRopeKernel<64, INTERLEAVE>
                <<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv), num_heads_q, num_heads_k,
                    num_heads_v, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),
                    reinterpret_cast<__nv_bfloat16 const*>(k_weight), reinterpret_cast<__nv_bfloat16 const*>(q_add_weight), reinterpret_cast<__nv_bfloat16 const*>(k_add_weight), cos_emb, sin_emb, num_tokens, seqlen_per_bs);
        });
        break;
    case 128:
        DISPATCH_INTERLEAVE(true, INTERLEAVE, {
            fusedQKNormRopeKernel<128, INTERLEAVE>
                <<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv), num_heads_q, num_heads_k,
                    num_heads_v, eps, reinterpret_cast<__nv_bfloat16 const*>(q_weight),
                    reinterpret_cast<__nv_bfloat16 const*>(k_weight), reinterpret_cast<__nv_bfloat16 const*>(q_add_weight), reinterpret_cast<__nv_bfloat16 const*>(k_add_weight), cos_emb, sin_emb, num_tokens, seqlen_per_bs);
        });
        break;
    default: /* TLLM_THROW("Unsupported head dimension for fusedQKNormRope: %d", head_dim); */ break;
    }
}

inline void launchFusedQKNormRopeFromTensors(
    torch::Tensor const& qkv,
    int num_tokens,
    int seqlen_per_bs,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    int head_dim,
    float eps,
    torch::Tensor const& q_weight,
    torch::Tensor const& k_weight,
    torch::Tensor const& q_add_weight,
    torch::Tensor const& k_add_weight,
    c10::optional<torch::Tensor> const& cos_emb = c10::nullopt,
    c10::optional<torch::Tensor> const& sin_emb = c10::nullopt,
    cudaStream_t stream = nullptr)
{

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream();
    }

    float const* cos_ptr = nullptr;
    float const* sin_ptr = nullptr;

    if (cos_emb.has_value()) {
        TORCH_CHECK(cos_emb->is_cuda(), "cos_emb must be a CUDA tensor");
        TORCH_CHECK(cos_emb->dtype() == torch::kFloat32, "cos_emb must be float32");
        cos_ptr = cos_emb->data_ptr<float>();
    }
    if (sin_emb.has_value()) {
        TORCH_CHECK(sin_emb->is_cuda(), "sin_emb must be a CUDA tensor");
        TORCH_CHECK(sin_emb->dtype() == torch::kFloat32, "sin_emb must be float32");
        sin_ptr = sin_emb->data_ptr<float>();
    }

    launchFusedQKNormRope(
        qkv.data_ptr(),
        num_tokens,
        seqlen_per_bs,	
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight.data_ptr(),
        k_weight.data_ptr(),
        q_add_weight.data_ptr(),
        k_add_weight.data_ptr(),
        cos_ptr,
        sin_ptr,
        stream);
}

torch::Tensor fused_qk_norm_rope_op(
    torch::Tensor qkv,
    int64_t num_tokens,
    int64_t seqlen_per_bs,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor const& q_add_weight,
    torch::Tensor const& k_add_weight,
    c10::optional<torch::Tensor> image_rotary_emb_cos,
    c10::optional<torch::Tensor> image_rotary_emb_sin)
{
    TORCH_CHECK(qkv.is_cuda(), "fused_qk_norm_rope_op: qkv must be a CUDA tensor");
    TORCH_CHECK(qkv.dim() == 2, "fused_qk_norm_rope_op: expect 2D tensor [num_tokens, hidden_size]");
    TORCH_CHECK(q_weight.is_cuda(), "fused_qk_norm_rope_op: q_weight must be a CUDA tensor");
    TORCH_CHECK(k_weight.is_cuda(), "fused_qk_norm_rope_op: k_weight must be a CUDA tensor");

    launchFusedQKNormRopeFromTensors(
        qkv,
	static_cast<int>(num_tokens),
        static_cast<int>(seqlen_per_bs),
	static_cast<int>(num_heads_q),
        static_cast<int>(num_heads_k),
        static_cast<int>(num_heads_v),
        static_cast<int>(head_dim),
        static_cast<float>(eps),
        q_weight,
        k_weight,
        q_add_weight,
        k_add_weight,
        image_rotary_emb_cos,
        image_rotary_emb_sin);

    return qkv;
}

TORCH_LIBRARY(fused_qk_norm_rope, m)
{
    m.def("fused_qk_norm_rope(Tensor qkv, int num_tokens, int seqlen_per_bs, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, Tensor q_add_weight, Tensor k_add_weight, Tensor? image_rotary_emb_cos, Tensor? image_rotary_emb_sin) -> Tensor");
}

TORCH_LIBRARY_IMPL(fused_qk_norm_rope, CUDA, m)
{
    m.impl("fused_qk_norm_rope", fused_qk_norm_rope_op);
}


inline void launchFusedQKNormRopeFromTensors_0(
    torch::Tensor const& qkv,
    int num_tokens,
    int seqlen_per_bs,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    int head_dim,
    float eps,
    torch::Tensor const& q_weight,
    torch::Tensor const& k_weight,
    c10::optional<torch::Tensor> const& cos_emb = c10::nullopt,
    c10::optional<torch::Tensor> const& sin_emb = c10::nullopt,
    cudaStream_t stream = nullptr)
{

    if (stream == nullptr) {
        stream = at::cuda::getCurrentCUDAStream();
    }

    float const* cos_ptr = nullptr;
    float const* sin_ptr = nullptr;

    if (cos_emb.has_value()) {
        TORCH_CHECK(cos_emb->is_cuda(), "cos_emb must be a CUDA tensor");
        TORCH_CHECK(cos_emb->dtype() == torch::kFloat32, "cos_emb must be float32");
        cos_ptr = cos_emb->data_ptr<float>();
    }
    if (sin_emb.has_value()) {
        TORCH_CHECK(sin_emb->is_cuda(), "sin_emb must be a CUDA tensor");
        TORCH_CHECK(sin_emb->dtype() == torch::kFloat32, "sin_emb must be float32");
        sin_ptr = sin_emb->data_ptr<float>();
    }

    launchFusedQKNormRope(
        qkv.data_ptr(),
        num_tokens,
	seqlen_per_bs,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight.data_ptr(),
        k_weight.data_ptr(),
        nullptr,
        nullptr,
        cos_ptr,
        sin_ptr,
        stream);
}

torch::Tensor fused_qk_norm_rope_0(
    torch::Tensor qkv,
    int64_t num_tokens,
    int64_t seqlen_per_bs,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    c10::optional<torch::Tensor> image_rotary_emb_cos,
    c10::optional<torch::Tensor> image_rotary_emb_sin)
{
    TORCH_CHECK(qkv.is_cuda(), "fused_qk_norm_rope_op: qkv must be a CUDA tensor");
    TORCH_CHECK(qkv.dim() == 2, "fused_qk_norm_rope_op: expect 2D tensor [num_tokens, hidden_size]");
    TORCH_CHECK(q_weight.is_cuda(), "fused_qk_norm_rope_op: q_weight must be a CUDA tensor");
    TORCH_CHECK(k_weight.is_cuda(), "fused_qk_norm_rope_op: k_weight must be a CUDA tensor");

    launchFusedQKNormRopeFromTensors_0(
        qkv,
        static_cast<int>(num_tokens),
        static_cast<int>(seqlen_per_bs),
	static_cast<int>(num_heads_q),
        static_cast<int>(num_heads_k),
        static_cast<int>(num_heads_v),
        static_cast<int>(head_dim),
        static_cast<float>(eps),
        q_weight,
        k_weight,
        image_rotary_emb_cos,
        image_rotary_emb_sin);

    return qkv;
}

TORCH_LIBRARY(fused_qk_norm_rope_0, m)
{
    m.def("fused_qk_norm_rope_0(Tensor qkv, int num_tokens, int seqlen_per_bs, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, Tensor? image_rotary_emb_cos, Tensor? image_rotary_emb_sin) -> Tensor");
}

TORCH_LIBRARY_IMPL(fused_qk_norm_rope_0, CUDA, m)
{
    m.impl("fused_qk_norm_rope_0", fused_qk_norm_rope_0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}
