#include "fusedQKNormRopeKernel.h"
#include "tensorrt_llm/common/mathUtils.h"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to compute RoPE rotation
__host__ __device__ void applyRoPE(float& x, float& y, float cosTheta, float sinTheta)
{
    float xNew = x * cosTheta - y * sinTheta;
    float yNew = x * sinTheta + y * cosTheta;
    x = xNew;
    y = yNew;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Perform per-head QK Norm and RoPE in a single kernel.
template <int head_dim>
__global__ void fusedQKNormRopeKernel(
    __nv_bfloat16* qkv,         // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int const* position_ids,    // Position IDs for RoPE [num_tokens]
    int const num_tokens,       // Number of tokens
    int const num_heads_q,      // Number of query heads
    int const num_heads_k,      // Number of key heads
    int const num_heads_v,      // Number of value heads
    float const eps = 1e-5f,    // Epsilon for RMS normalization
    float const base = 10000.0f // Base for RoPE computation
)
{
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
    static_assert(head_dim % 32 == 0, "head_dim must be divisible by 32 (each warp processes one head)");
    constexpr int numElemsPerThread = head_dim / 32;
    constexpr int numVecPerThread = numElemsPerThread / 4;
    float elements[numElemsPerThread];

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

    // Load elements and compute sum of squares
    int i = 0;

#pragma unroll
    for (; i < numVecPerThread * 4; i += 4)
    {
        // Load 4 bfloat16 elements using LDG.64
        uint2 data;
        data = *reinterpret_cast<uint2 const*>(&qkv[offsetThread + i]);

        // Convert bfloat16 to float
        auto vals0 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 const&>(data.x));
        auto vals1 = __bfloat1622float2(reinterpret_cast<__nv_bfloat162 const&>(data.y));

        // Store and compute sum of squares
        elements[i] = vals0.x;
        elements[i + 1] = vals0.y;
        elements[i + 2] = vals1.x;
        elements[i + 3] = vals1.y;

        sumOfSquares += elements[i] * elements[i];
        sumOfSquares += elements[i + 1] * elements[i + 1];
        sumOfSquares += elements[i + 2] * elements[i + 2];
        sumOfSquares += elements[i + 3] * elements[i + 3];
    }

    for (; i < numElemsPerThread; i++)
    {
        elements[i] = __bfloat162float(qkv[offsetThread + i]);
        sumOfSquares += elements[i] * elements[i];
    }

    // Reduce sum across warp
    for (int mask = 16; mask > 0; mask /= 2)
    {
        sumOfSquares += __shfl_xor_sync(0xffffffff, sumOfSquares, mask);
    }

    // Compute RMS normalization factor
    float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

    // Normalize elements
    for (i = 0; i < numElemsPerThread; i++)
    {
        elements[i] *= rms_rcp;
    }

    // Apply RoPE to normalized elements
    int pos_id = position_ids[tokenIdx];

    for (i = 0; i < numElemsPerThread; i += 2)
    {
        // Calculate the actual dimension index in the head
        int dim_idx = laneId * numElemsPerThread + i;

        // Proper RoPE frequency calculation
        int half_dim = dim_idx / 2;
        float freq = powf(base, -2.0f * half_dim / static_cast<float>(head_dim));
        float theta = pos_id * freq;

        float cosTheta = cosf(theta);
        float sinTheta = sinf(theta);

        // Apply rotation
        applyRoPE(elements[i], elements[i + 1], cosTheta, sinTheta);
    }

    // Write back to original tensor
    for (i = 0; i < numElemsPerThread; i += 4)
    {
        // Convert back to bfloat16 format
        __nv_bfloat162 vals0 = __float22bfloat162_rn(float2(elements[i], elements[i + 1]));
        __nv_bfloat162 vals1 = __float22bfloat162_rn(float2(elements[i + 2], elements[i + 3]));

        uint2 data;
        data.x = *reinterpret_cast<uint32_t*>(&vals0);
        data.y = *reinterpret_cast<uint32_t*>(&vals1);

        // Calculate the correct offset for writing back
        int writeOffset = offsetThread + i;
        uint2* outputPtr = reinterpret_cast<uint2*>(&qkv[writeOffset]);
        *outputPtr = data;
    }
}

namespace tensorrt_llm
{
namespace kernels
{

// Launch wrapper for the fusedQKNormRope kernel with different head dimensions
void launchFusedQKNormRope(void* qkv, int const* position_ids, int const num_tokens, int const num_heads_q,
    int const num_heads_k, int const num_heads_v, int const head_dim, float const eps, float const base,
    cudaStream_t stream)
{
    constexpr int blockSize = 256;

    int const warpsPerBlock = blockSize / 32;
    int const totalQKHeads = num_heads_q + num_heads_k;
    int const totalWarps = num_tokens * totalQKHeads;

    int const gridSize = divUp(totalWarps, warpsPerBlock);
    dim3 gridDim(gridSize);
    dim3 blockDim(blockSize);

    // Head dimensions should be a multiple of 32
    // Add more cases as needed
    switch (head_dim)
    {
    case 64:
        fusedQKNormRopeKernel<64><<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv), position_ids,
            num_tokens, num_heads_q, num_heads_k, num_heads_v, eps, base);
        break;
    case 128:
        fusedQKNormRopeKernel<128><<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv),
            position_ids, num_tokens, num_heads_q, num_heads_k, num_heads_v, eps, base);
        break;
    case 256:
        fusedQKNormRopeKernel<256><<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(qkv),
            position_ids, num_tokens, num_heads_q, num_heads_k, num_heads_v, eps, base);
        break;
    default:
        // Unsupported head dimension
        TLLM_THROW("Unsupported head dimension for fusedQKNormRope: %d", head_dim);
    }
}

} // namespace kernels
} // namespace tensorrt_llm
