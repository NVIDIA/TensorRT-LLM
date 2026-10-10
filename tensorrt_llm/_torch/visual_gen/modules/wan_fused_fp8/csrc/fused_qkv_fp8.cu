// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Cross-head QK RMSNorm + RoPE + FP8 quant of packed QKV (from fusedDiTQKNormFullDimRopeKernel).

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <tuple>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace
{

template <typename T>
__device__ __forceinline__ T warpReduceSumT(T v)
{
#pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float warpReduceMaxF(float v)
{
#pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// 8 values: bf16 round, scale, saturating e4m3 store; tracks amax.
__device__ __forceinline__ void storeFp8x8(float const (&x)[8], __nv_fp8_e4m3* dst, float mul, float& amax)
{
    __nv_fp8x2_storage_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float a = __bfloat162float(__float2bfloat16_rn(x[2 * i]));
        float b = __bfloat162float(__float2bfloat16_rn(x[2 * i + 1]));
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
        packed[i] = __nv_cvt_float2_to_fp8x2(make_float2(a * mul, b * mul), __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<uint2 const*>(packed);
}

} // namespace

namespace tensorrt_llm::common
{
__device__ __forceinline__ float warpReduceSum(float v)
{
    return warpReduceSumT(v);
}

__device__ __forceinline__ float warpReduceMax(float v)
{
    return warpReduceMaxF(v);
}
} // namespace tensorrt_llm::common

template <int HEAD_DIM, bool INTERLEAVE, bool PER_HEAD_COS, typename CosT>
__global__ void fusedDiTQKNormRopeFp8QuantKernel(__nv_bfloat16 const* qkv, __nv_fp8_e4m3* q_out, __nv_fp8_e4m3* k_out,
    __nv_fp8_e4m3* v_out, float const* quant_mul, float* amax_out, int const num_heads_q, int const num_heads_k,
    int const num_heads_v, float const eps, __nv_bfloat16 const* q_weight, __nv_bfloat16 const* k_weight,
    CosT const* cos_emb, CosT const* sin_emb, int const num_tokens, int const cos_seq_per_batch)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int ROWS_PER_BLOCK = 2;
    constexpr int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK; // 128
    constexpr int WARPS_PER_ROW = THREADS_PER_ROW / 32;          // 4
    constexpr int CHUNK_ELEMS = 8;                               // uint4 = 8 bf16
    static_assert(HEAD_DIM % CHUNK_ELEMS == 0, "HEAD_DIM must be divisible by the vector width");
    // MAX_N = max(num_heads_q) * HEAD_DIM. 64 covers WAN-14B (40 heads) and
    // future ≤64-head models. SMEM budget at 64h × 128 = 128 KB (bf16 cos) / 192 KB
    // (fp32 cos), both within B200's 227 KB dynamic SMEM cap.
    constexpr int MAX_N = 64 * HEAD_DIM;
    constexpr int MAX_CHUNKS = (MAX_N + THREADS_PER_ROW * CHUNK_ELEMS - 1) / (THREADS_PER_ROW * CHUNK_ELEMS);

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif

    int const tid = threadIdx.x;
    int const row_in_block = tid / THREADS_PER_ROW;
    int const lane_in_row = tid % THREADS_PER_ROW;
    int const row_warp = lane_in_row >> 5;
    int const row_lane = lane_in_row & 31;

    int const tokenIdx = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    if (tokenIdx >= num_tokens)
        return;

    int const N = num_heads_q * HEAD_DIM; // num_heads_q == num_heads_k (enforced by launcher)
    int const chunks_per_row = (N + THREADS_PER_ROW * CHUNK_ELEMS - 1) / (THREADS_PER_ROW * CHUNK_ELEMS);
    int const num_heads_total = num_heads_q + num_heads_k + num_heads_v;
    int64_t const tokenBaseQ = static_cast<int64_t>(tokenIdx) * num_heads_total * HEAD_DIM;
    int64_t const tokenBaseK = tokenBaseQ + N;
    int const cos_tokenIdx = (cos_seq_per_batch > 0) ? (tokenIdx % cos_seq_per_batch) : tokenIdx;
    int64_t const embBase = PER_HEAD_COS ? static_cast<int64_t>(cos_tokenIdx) * num_heads_q * HEAD_DIM
                                         : static_cast<int64_t>(cos_tokenIdx) * HEAD_DIM;

    // SMEM layout: [Q row0][Q row1][K row0][K row1] bf16, [cos row0..1][sin row0..1] CosT, warp_sums.
    // Only the shared fp32 path is deduplicated. The bf16 and per-head paths keep their existing layout.
    constexpr bool kDeduplicateSharedCos = !PER_HEAD_COS && std::is_same_v<CosT, float>;
    int const cosStride = kDeduplicateSharedCos ? HEAD_DIM : N;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* smem_k = smem_q + ROWS_PER_BLOCK * N;
    CosT* smem_cos = reinterpret_cast<CosT*>(smem_raw + 2 * ROWS_PER_BLOCK * N * sizeof(__nv_bfloat16));
    CosT* smem_sin = smem_cos + ROWS_PER_BLOCK * cosStride;
    float* warp_sums = reinterpret_cast<float*>(
        smem_raw + 2 * ROWS_PER_BLOCK * N * sizeof(__nv_bfloat16) + 2 * ROWS_PER_BLOCK * cosStride * sizeof(CosT));

    // Phase 0a: cp.async Q + K + cos + sin -> SMEM (all in one commit group).
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        __pipeline_memcpy_async(smem_q + row_in_block * N + elemBase, qkv + tokenBaseQ + elemBase, 16);
        __pipeline_memcpy_async(smem_k + row_in_block * N + elemBase, qkv + tokenBaseK + elemBase, 16);
        int const headIdx = elemBase / HEAD_DIM;
        int const baseDim = elemBase - headIdx * HEAD_DIM;
        int const cosHeadOff = PER_HEAD_COS ? headIdx * HEAD_DIM : 0;
        if constexpr (kDeduplicateSharedCos)
        {
            // Only the threads assigned to the first head stage the shared row.
            if (elemBase < HEAD_DIM)
            {
                __pipeline_memcpy_async(
                    smem_cos + row_in_block * HEAD_DIM + elemBase, cos_emb + embBase + elemBase, 16);
                __pipeline_memcpy_async(
                    smem_cos + row_in_block * HEAD_DIM + elemBase + 4, cos_emb + embBase + elemBase + 4, 16);
                __pipeline_memcpy_async(
                    smem_sin + row_in_block * HEAD_DIM + elemBase, sin_emb + embBase + elemBase, 16);
                __pipeline_memcpy_async(
                    smem_sin + row_in_block * HEAD_DIM + elemBase + 4, sin_emb + embBase + elemBase + 4, 16);
            }
        }
        else if constexpr (std::is_same_v<CosT, float>)
        {
            __pipeline_memcpy_async(
                smem_cos + row_in_block * N + elemBase, cos_emb + embBase + cosHeadOff + baseDim, 16);
            __pipeline_memcpy_async(
                smem_cos + row_in_block * N + elemBase + 4, cos_emb + embBase + cosHeadOff + baseDim + 4, 16);
            __pipeline_memcpy_async(
                smem_sin + row_in_block * N + elemBase, sin_emb + embBase + cosHeadOff + baseDim, 16);
            __pipeline_memcpy_async(
                smem_sin + row_in_block * N + elemBase + 4, sin_emb + embBase + cosHeadOff + baseDim + 4, 16);
        }
        else
        {
            __pipeline_memcpy_async(
                smem_cos + row_in_block * N + elemBase, cos_emb + embBase + cosHeadOff + baseDim, 16);
            __pipeline_memcpy_async(
                smem_sin + row_in_block * N + elemBase, sin_emb + embBase + cosHeadOff + baseDim, 16);
        }
    }
    __pipeline_commit();

    // Phase 0b: sync load q_weight + k_weight -> regs (overlaps cp.async transfers).
    uint4 q_w_cache[MAX_CHUNKS], k_w_cache[MAX_CHUNKS];
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        int const headIdx = elemBase / HEAD_DIM;
        int const baseDim = elemBase - headIdx * HEAD_DIM;
        q_w_cache[chunk] = *reinterpret_cast<uint4 const*>(&q_weight[headIdx * HEAD_DIM + baseDim]);
        k_w_cache[chunk] = *reinterpret_cast<uint4 const*>(&k_weight[headIdx * HEAD_DIM + baseDim]);
    }

    // Phase 0c: wait + sync. The barrier makes the shared row visible to threads assigned to every head.
    __pipeline_wait_prior(0);
    __syncthreads();

    // Phase 1: compute sum²_Q and sum²_K together from SMEM.
    float q_sum2 = 0.0f, k_sum2 = 0.0f;
#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            continue;
        uint4 const qv = *reinterpret_cast<uint4 const*>(&smem_q[row_in_block * N + elemBase]);
        uint4 const kv = *reinterpret_cast<uint4 const*>(&smem_k[row_in_block * N + elemBase]);
        uint const* qu = reinterpret_cast<uint const*>(&qv);
        uint const* ku = reinterpret_cast<uint const*>(&kv);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 qv2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&qu[i]));
            float2 kv2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&ku[i]));
            q_sum2 += qv2.x * qv2.x + qv2.y * qv2.y;
            k_sum2 += kv2.x * kv2.x + kv2.y * kv2.y;
        }
    }

    // Per-row reduce both at once: pack (q_sum, k_sum) per warp slot.
    q_sum2 = tensorrt_llm::common::warpReduceSum(q_sum2);
    k_sum2 = tensorrt_llm::common::warpReduceSum(k_sum2);
    // warp_sums layout: [row_in_block][2 * warp + (0=Q, 1=K)]
    if (row_lane == 0)
    {
        warp_sums[row_in_block * (2 * WARPS_PER_ROW) + 2 * row_warp + 0] = q_sum2;
        warp_sums[row_in_block * (2 * WARPS_PER_ROW) + 2 * row_warp + 1] = k_sum2;
    }
    __syncthreads();
    float q_total = 0.0f, k_total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS_PER_ROW; w++)
    {
        q_total += warp_sums[row_in_block * (2 * WARPS_PER_ROW) + 2 * w + 0];
        k_total += warp_sums[row_in_block * (2 * WARPS_PER_ROW) + 2 * w + 1];
    }
    float const q_rms_rcp = rsqrtf(q_total / static_cast<float>(N) + eps);
    float const k_rms_rcp = rsqrtf(k_total / static_cast<float>(N) + eps);

    float const q_mul = quant_mul[0], k_mul = quant_mul[1], v_mul = quant_mul[2];
    float q_amax = 0.f, k_amax = 0.f, v_amax = 0.f;

    // Phase 2: norm + RoPE on Q/K, FP8 quant of Q/K/V.
    // Cos/sin loaded from SMEM (same stage as Q+K), converted to fp32 at use.
    auto apply_chunk = [&](int chunk, __nv_bfloat16 const* smem_input, uint4 const* w_cache, __nv_fp8_e4m3* out,
                           float mul, float& amax, float rms_rcp)
    {
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase >= N)
            return;
        uint4 const in_vec = *reinterpret_cast<uint4 const*>(&smem_input[row_in_block * N + elemBase]);
        uint4 const w_vec = w_cache[chunk];

        // CHUNK_ELEMS divides HEAD_DIM, so a vector never crosses a head boundary.
        int const cosIdx
            = kDeduplicateSharedCos ? (row_in_block * HEAD_DIM + elemBase % HEAD_DIM) : (row_in_block * N + elemBase);

        float cos_vals[CHUNK_ELEMS];
        float sin_vals[CHUNK_ELEMS];
        if constexpr (std::is_same_v<CosT, float>)
        {
            float4 const* cs = reinterpret_cast<float4 const*>(&smem_cos[cosIdx]);
            float4 const* ss = reinterpret_cast<float4 const*>(&smem_sin[cosIdx]);
            float4 c0 = cs[0], c1 = cs[1];
            float4 s0 = ss[0], s1 = ss[1];
            cos_vals[0] = c0.x;
            cos_vals[1] = c0.y;
            cos_vals[2] = c0.z;
            cos_vals[3] = c0.w;
            cos_vals[4] = c1.x;
            cos_vals[5] = c1.y;
            cos_vals[6] = c1.z;
            cos_vals[7] = c1.w;
            sin_vals[0] = s0.x;
            sin_vals[1] = s0.y;
            sin_vals[2] = s0.z;
            sin_vals[3] = s0.w;
            sin_vals[4] = s1.x;
            sin_vals[5] = s1.y;
            sin_vals[6] = s1.z;
            sin_vals[7] = s1.w;
        }
        else
        {
            uint4 const cp = *reinterpret_cast<uint4 const*>(&smem_cos[cosIdx]);
            uint4 const sp = *reinterpret_cast<uint4 const*>(&smem_sin[cosIdx]);
            uint const* cu = reinterpret_cast<uint const*>(&cp);
            uint const* su = reinterpret_cast<uint const*>(&sp);
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 cv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&cu[i]));
                float2 sv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&su[i]));
                cos_vals[2 * i] = cv.x;
                cos_vals[2 * i + 1] = cv.y;
                sin_vals[2 * i] = sv.x;
                sin_vals[2 * i + 1] = sv.y;
            }
        }

        float elements[CHUNK_ELEMS];
        float w_vals[CHUNK_ELEMS];
        uint const* x_uints = reinterpret_cast<uint const*>(&in_vec);
        uint const* w_uints = reinterpret_cast<uint const*>(&w_vec);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 xv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]));
            float2 wv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&w_uints[i]));
            elements[2 * i] = xv.x;
            elements[2 * i + 1] = xv.y;
            w_vals[2 * i] = wv.x;
            w_vals[2 * i + 1] = wv.y;
        }

#pragma unroll
        for (int i = 0; i < CHUNK_ELEMS; i++)
            elements[i] *= rms_rcp * w_vals[i];

        if constexpr (INTERLEAVE)
        {
#pragma unroll
            for (int i = 0; i < CHUNK_ELEMS; i += 2)
            {
                float const x = elements[i], y = elements[i + 1];
                elements[i] = x * cos_vals[i] + (-y) * sin_vals[i];
                elements[i + 1] = y * cos_vals[i + 1] + x * sin_vals[i + 1];
            }
        }
        else
        {
            constexpr int xor_mask = HEAD_DIM / 16;
            bool const negate = ((row_lane & xor_mask) == 0);
            unsigned const activeMask = __activemask();
#pragma unroll
            for (int i = 0; i < CHUNK_ELEMS; i++)
            {
                float p = __shfl_xor_sync(activeMask, elements[i], xor_mask);
                if (negate)
                {
                    p = -p;
                }
                elements[i] = elements[i] * cos_vals[i] + p * sin_vals[i];
            }
        }

        // Round to bf16 first to match the unfused path.
        storeFp8x8(elements, out + static_cast<int64_t>(tokenIdx) * N + elemBase, mul, amax);
    };

#pragma unroll
    for (int chunk = 0; chunk < MAX_CHUNKS; chunk++)
    {
        if (chunk >= chunks_per_row)
            continue;
        apply_chunk(chunk, smem_q, q_w_cache, q_out, q_mul, q_amax, q_rms_rcp);
        apply_chunk(chunk, smem_k, k_w_cache, k_out, k_mul, k_amax, k_rms_rcp);
        // V: quantize only.
        int const elemBase = chunk * THREADS_PER_ROW * CHUNK_ELEMS + lane_in_row * CHUNK_ELEMS;
        if (elemBase < N)
        {
            uint4 const in_vec = *reinterpret_cast<uint4 const*>(&qkv[tokenBaseK + N + elemBase]);
            uint const* x_uints = reinterpret_cast<uint const*>(&in_vec);
            float elements[CHUNK_ELEMS];
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 xv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]));
                elements[2 * i] = xv.x;
                elements[2 * i + 1] = xv.y;
            }
            storeFp8x8(elements, v_out + static_cast<int64_t>(tokenIdx) * N + elemBase, v_mul, v_amax);
        }
    }

    // Record this call's per-tensor amax.
    if (amax_out != nullptr)
    {
        q_amax = tensorrt_llm::common::warpReduceMax(q_amax);
        k_amax = tensorrt_llm::common::warpReduceMax(k_amax);
        v_amax = tensorrt_llm::common::warpReduceMax(v_amax);
        if (row_lane == 0)
        {
            // Non-negative floats compare like their int bits.
            atomicMax(reinterpret_cast<int*>(&amax_out[0]), __float_as_int(q_amax));
            atomicMax(reinterpret_cast<int*>(&amax_out[1]), __float_as_int(k_amax));
            atomicMax(reinterpret_cast<int*>(&amax_out[2]), __float_as_int(v_amax));
        }
    }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Ring depth of the bulk-copy pipeline (rows in flight per block).
constexpr int kNrqStages = 3;

namespace
{

// float to bf16 (RN) and back in one cvt: zero low half.
__device__ __forceinline__ float roundBf16(float v)
{
    __nv_bfloat162 const p = __floats2bfloat162_rn(0.f, v);
    return __uint_as_float(*reinterpret_cast<uint32_t const*>(&p));
}

// storeFp8x8 with the single-cvt bf16 rounding.
__device__ __forceinline__ void storeFp8x8Fast(float const (&x)[8], __nv_fp8_e4m3* dst, float mul, float& amax)
{
    __nv_fp8x2_storage_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float a = roundBf16(x[2 * i]);
        float b = roundBf16(x[2 * i + 1]);
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
        packed[i] = __nv_cvt_float2_to_fp8x2(make_float2(a * mul, b * mul), __NV_SATFINITE, __NV_E4M3);
    }
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<uint2 const*>(packed);
}

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemAddr(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t phase)
{
    uint32_t done = 0;
    while (!done)
    {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(smemAddr(bar)), "r"(phase)
            : "memory");
    }
}

// Async bulk global-to-shared copy; completion counted on the mbarrier.
__device__ __forceinline__ void bulkLoad(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemAddr(dst)),
        "l"(src), "r"(bytes), "r"(smemAddr(bar))
        : "memory");
}

// Q/K chunk: same math as apply_chunk, fp32 cos/sin.
template <bool INTERLEAVE>
__device__ __forceinline__ void normRopeChunk(uint4 const in_vec, uint4 const w_vec, float const* cosp,
    float const* sinp, int row_lane, float rms_rcp, __nv_fp8_e4m3* dst, float mul, float& amax)
{
    constexpr int HEAD_DIM = 128;
    constexpr int CHUNK_ELEMS = 8;
    float cos_vals[CHUNK_ELEMS];
    float sin_vals[CHUNK_ELEMS];
    float4 const* cs = reinterpret_cast<float4 const*>(cosp);
    float4 const* ss = reinterpret_cast<float4 const*>(sinp);
    float4 c0 = cs[0], c1 = cs[1];
    float4 s0 = ss[0], s1 = ss[1];
    cos_vals[0] = c0.x;
    cos_vals[1] = c0.y;
    cos_vals[2] = c0.z;
    cos_vals[3] = c0.w;
    cos_vals[4] = c1.x;
    cos_vals[5] = c1.y;
    cos_vals[6] = c1.z;
    cos_vals[7] = c1.w;
    sin_vals[0] = s0.x;
    sin_vals[1] = s0.y;
    sin_vals[2] = s0.z;
    sin_vals[3] = s0.w;
    sin_vals[4] = s1.x;
    sin_vals[5] = s1.y;
    sin_vals[6] = s1.z;
    sin_vals[7] = s1.w;

    float elements[CHUNK_ELEMS];
    float w_vals[CHUNK_ELEMS];
    uint const* x_uints = reinterpret_cast<uint const*>(&in_vec);
    uint const* w_uints = reinterpret_cast<uint const*>(&w_vec);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 xv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]));
        elements[2 * i] = xv.x;
        elements[2 * i + 1] = xv.y;
        float2 wv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&w_uints[i]));
        w_vals[2 * i] = wv.x;
        w_vals[2 * i + 1] = wv.y;
    }

#pragma unroll
    for (int i = 0; i < CHUNK_ELEMS; i++)
        elements[i] *= rms_rcp * w_vals[i];

    if constexpr (INTERLEAVE)
    {
#pragma unroll
        for (int i = 0; i < CHUNK_ELEMS; i += 2)
        {
            float const x = elements[i], y = elements[i + 1];
            elements[i] = x * cos_vals[i] + (-y) * sin_vals[i];
            elements[i + 1] = y * cos_vals[i + 1] + x * sin_vals[i + 1];
        }
    }
    else
    {
        constexpr int xor_mask = HEAD_DIM / 16;
        bool const negate = ((row_lane & xor_mask) == 0);
        unsigned const activeMask = __activemask();
#pragma unroll
        for (int i = 0; i < CHUNK_ELEMS; i++)
        {
            float p = __shfl_xor_sync(activeMask, elements[i], xor_mask);
            if (negate)
            {
                p = -p;
            }
            elements[i] = elements[i] * cos_vals[i] + p * sin_vals[i];
        }
    }

    // Round to bf16 first to match the unfused path.
    storeFp8x8Fast(elements, dst, mul, amax);
}

// V chunk: bf16 to FP8 with the V multiplier.
__device__ __forceinline__ void quantChunk(uint4 const in_vec, __nv_fp8_e4m3* dst, float mul, float& amax)
{
    uint const* x_uints = reinterpret_cast<uint const*>(&in_vec);
    float elements[8];
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 xv = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&x_uints[i]));
        elements[2 * i] = xv.x;
        elements[2 * i + 1] = xv.y;
    }
    storeFp8x8Fast(elements, dst, mul, amax);
}

// Per-row sum of squares over this thread's Q and K chunks (original order).
template <int NCH>
__device__ __forceinline__ void rowSumSq(
    __nv_bfloat16 const* sq, __nv_bfloat16 const* sk, int lane_in_row, int N, float& q_sum2, float& k_sum2)
{
#pragma unroll
    for (int chunk = 0; chunk < NCH; chunk++)
    {
        int const elemBase = chunk * 1024 + lane_in_row * 8;
        if (elemBase >= N)
            continue;
        uint4 const qv = *reinterpret_cast<uint4 const*>(&sq[elemBase]);
        uint4 const kv = *reinterpret_cast<uint4 const*>(&sk[elemBase]);
        uint const* qu = reinterpret_cast<uint const*>(&qv);
        uint const* ku = reinterpret_cast<uint const*>(&kv);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 qv2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&qu[i]));
            float2 kv2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&ku[i]));
            q_sum2 += qv2.x * qv2.x + qv2.y * qv2.y;
            k_sum2 += kv2.x * kv2.x + kv2.y * kv2.y;
        }
    }
}

} // namespace

// Shared fp32 cos only; persistent 128-thread blocks, rows streamed by bulk copies.
template <bool INTERLEAVE, int NCH>
__global__ void __launch_bounds__(128) fusedDiTQKNormRopeFp8QuantPipeKernel(__nv_bfloat16 const* qkv,
    __nv_fp8_e4m3* q_out, __nv_fp8_e4m3* k_out, __nv_fp8_e4m3* v_out, float const* quant_mul, float* amax_out,
    int const num_heads, float const eps, __nv_bfloat16 const* q_weight, __nv_bfloat16 const* k_weight,
    float const* cos_emb, float const* sin_emb, int const num_tokens, int const cos_seq_per_batch)
{
    constexpr int HEAD_DIM = 128;
    constexpr int WARPS_PER_ROW = 4;
    constexpr int NST = kNrqStages;
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;");
#endif
    int const tid = threadIdx.x;
    int const row_warp = tid >> 5;
    int const row_lane = tid & 31;
    int const N = num_heads * HEAD_DIM;
    uint32_t const rowBytes = 3 * N * sizeof(__nv_bfloat16);
    uint32_t const stageBytes = rowBytes + 2 * HEAD_DIM * sizeof(float);
    extern __shared__ __align__(128) unsigned char ring[];
    __shared__ __align__(8) uint64_t bars[NST];
    __shared__ float warp_sums[2 * WARPS_PER_ROW];
    int const stride = gridDim.x;
    auto issue = [&](int st, int t)
    {
        unsigned char* dst = ring + st * stageBytes;
        int const ct = (cos_seq_per_batch > 0) ? (t % cos_seq_per_batch) : t;
        mbarExpectTx(&bars[st], stageBytes);
        bulkLoad(dst, qkv + static_cast<int64_t>(t) * 3 * N, rowBytes, &bars[st]);
        bulkLoad(dst + rowBytes, cos_emb + static_cast<int64_t>(ct) * HEAD_DIM, HEAD_DIM * sizeof(float), &bars[st]);
        bulkLoad(dst + rowBytes + HEAD_DIM * sizeof(float), sin_emb + static_cast<int64_t>(ct) * HEAD_DIM,
            HEAD_DIM * sizeof(float), &bars[st]);
    };
    if (tid == 0)
    {
#pragma unroll
        for (int st = 0; st < NST; st++)
            mbarInit(&bars[st], 1);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#pragma unroll
        for (int st = 0; st < NST; st++)
        {
            int const t = blockIdx.x + st * stride;
            if (t < num_tokens)
                issue(st, t);
        }
    }
    // Norm weights stay in registers across rows.
    uint4 q_w_cache[NCH], k_w_cache[NCH];
#pragma unroll
    for (int chunk = 0; chunk < NCH; chunk++)
    {
        int const elemBase = chunk * 1024 + tid * 8;
        if (elemBase >= N)
            continue;
        uint4 const qw = *reinterpret_cast<uint4 const*>(&q_weight[elemBase]);
        uint4 const kw = *reinterpret_cast<uint4 const*>(&k_weight[elemBase]);
        q_w_cache[chunk] = qw;
        k_w_cache[chunk] = kw;
    }
    float const q_mul = quant_mul[0], k_mul = quant_mul[1], v_mul = quant_mul[2];
    float q_amax = 0.f, k_amax = 0.f, v_amax = 0.f;
    __syncthreads();
    int it = 0;
    for (int tokenIdx = blockIdx.x; tokenIdx < num_tokens; tokenIdx += stride, it++)
    {
        int const st = it % NST;
        mbarWait(&bars[st], (it / NST) & 1);
        __nv_bfloat16 const* srow = reinterpret_cast<__nv_bfloat16 const*>(ring + st * stageBytes);
        float const* scos = reinterpret_cast<float const*>(ring + st * stageBytes + rowBytes);
        float const* ssin = scos + HEAD_DIM;
        float q_sum2 = 0.0f, k_sum2 = 0.0f;
        rowSumSq<NCH>(srow, srow + N, tid, N, q_sum2, k_sum2);
        q_sum2 = tensorrt_llm::common::warpReduceSum(q_sum2);
        k_sum2 = tensorrt_llm::common::warpReduceSum(k_sum2);
        if (row_lane == 0)
        {
            warp_sums[2 * row_warp + 0] = q_sum2;
            warp_sums[2 * row_warp + 1] = k_sum2;
        }
        __syncthreads();
        float q_total = 0.0f, k_total = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_ROW; w++)
        {
            q_total += warp_sums[2 * w + 0];
            k_total += warp_sums[2 * w + 1];
        }
        float const q_rms_rcp = rsqrtf(q_total / static_cast<float>(N) + eps);
        float const k_rms_rcp = rsqrtf(k_total / static_cast<float>(N) + eps);
        int64_t const outBase = static_cast<int64_t>(tokenIdx) * N;
#pragma unroll
        for (int chunk = 0; chunk < NCH; chunk++)
        {
            int const elemBase = chunk * 1024 + tid * 8;
            if (elemBase >= N)
                continue;
            int const cosOff = elemBase % HEAD_DIM;
            normRopeChunk<INTERLEAVE>(*reinterpret_cast<uint4 const*>(srow + elemBase), q_w_cache[chunk], scos + cosOff,
                ssin + cosOff, row_lane, q_rms_rcp, q_out + outBase + elemBase, q_mul, q_amax);
            normRopeChunk<INTERLEAVE>(*reinterpret_cast<uint4 const*>(srow + N + elemBase), k_w_cache[chunk],
                scos + cosOff, ssin + cosOff, row_lane, k_rms_rcp, k_out + outBase + elemBase, k_mul, k_amax);
            quantChunk(
                *reinterpret_cast<uint4 const*>(srow + 2 * N + elemBase), v_out + outBase + elemBase, v_mul, v_amax);
        }
        // Stage and warp sums fully consumed; refill the stage.
        __syncthreads();
        if (tid == 0 && tokenIdx + NST * stride < num_tokens)
            issue(st, tokenIdx + NST * stride);
    }

    // Record this call's per-tensor amax.
    if (amax_out != nullptr)
    {
        q_amax = tensorrt_llm::common::warpReduceMax(q_amax);
        k_amax = tensorrt_llm::common::warpReduceMax(k_amax);
        v_amax = tensorrt_llm::common::warpReduceMax(v_amax);
        if (row_lane == 0)
        {
            // Non-negative floats compare like their int bits.
            atomicMax(reinterpret_cast<int*>(&amax_out[0]), __float_as_int(q_amax));
            atomicMax(reinterpret_cast<int*>(&amax_out[1]), __float_as_int(k_amax));
            atomicMax(reinterpret_cast<int*>(&amax_out[2]), __float_as_int(v_amax));
        }
    }

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Per-block |V| amax partials; one warp per token, kVU loads in flight.
constexpr int kVU = 10;

__global__ void __launch_bounds__(256) vAmaxRowKernel(__nv_bfloat16 const* qkv, int64_t tokens, int N, float* out)
{
    int const lane = threadIdx.x & 31;
    int64_t const nWarps = (static_cast<int64_t>(gridDim.x) * blockDim.x) >> 5;
    int const cpr = N / 8;
    float m = 0.f;
    for (int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5; t < tokens; t += nWarps)
    {
        uint4 const* src = reinterpret_cast<uint4 const*>(qkv + t * 3 * N + 2 * N);
        for (int c0 = 0; c0 < cpr; c0 += 32 * kVU)
        {
            uint4 v[kVU];
#pragma unroll
            for (int u = 0; u < kVU; u++)
            {
                int const c = c0 + u * 32 + lane;
                v[u] = c < cpr ? __ldg(src + c) : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int u = 0; u < kVU; u++)
            {
                __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&v[u]);
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    float2 f = __bfloat1622float2(h[j]);
                    m = fmaxf(m, fmaxf(fabsf(f.x), fabsf(f.y)));
                }
            }
        }
    }
    m = warpReduceMaxF(m);
    __shared__ float sm[8];
    if ((threadIdx.x & 31) == 0)
        sm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        m = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.f;
        m = warpReduceMaxF(m);
        if (threadIdx.x == 0)
        {
            out[blockIdx.x] = m;
        }
    }
}

// Max of per-block partials, then the V scale and its reciprocal.
__global__ void vScaleReduceKernel(float const* partial, int n, float* amax, float* mul_v, float* scale_v)
{
    float m = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        m = fmaxf(m, partial[i]);
    m = warpReduceMaxF(m);
    __shared__ float sm[32];
    if ((threadIdx.x & 31) == 0)
        sm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        m = threadIdx.x < (blockDim.x >> 5) ? sm[threadIdx.x] : 0.f;
        m = warpReduceMaxF(m);
        if (threadIdx.x == 0)
        {
            amax[0] = m;
            float const s = fmaxf(m, 1e-12f) / 448.f;
            *scale_v = s;
            *mul_v = 1.f / s;
        }
    }
}

// Resident-block grid for the amax kernel.
int vAmaxGrid(int sms)
{
    static int perSm = 0;
    if (perSm == 0)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, vAmaxRowKernel, 256, 0);
    return std::max(1, perSm) * sms;
}

template <typename K, typename... Args>
void launchPdl(K kernel, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args)
{
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaLaunchKernelEx(&cfg, kernel, args...);
}

// Fast path: shared fp32 cos rows, chunk count as a template argument.
template <bool IL, int NCH>
void launchNrqFast(int tokens, int H, cudaStream_t stream, __nv_bfloat16 const* qkv, __nv_fp8_e4m3* q8,
    __nv_fp8_e4m3* k8, __nv_fp8_e4m3* v8, float const* mul, float* amax, float eps, __nv_bfloat16 const* qw,
    __nv_bfloat16 const* kw, float const* cosp, float const* sinp, int cspb)
{
    int const N = H * 128;
    auto* kern = fusedDiTQKNormRopeFp8QuantPipeKernel<IL, NCH>;
    size_t const smem = kNrqStages * (3 * N * sizeof(__nv_bfloat16) + 256 * sizeof(float));
    cudaFuncSetAttribute(reinterpret_cast<void const*>(kern), cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int perSm = 0, sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kern, 128, smem);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int const grid = std::min(tokens, std::max(1, perSm) * sms);
    launchPdl(kern, dim3(grid), dim3(128), smem, stream, qkv, q8, k8, v8, mul, amax, H, eps, qw, kw, cosp, sinp, tokens,
        cspb);
}

template <bool IL>
void dispatchNrqFast(int nch, int tokens, int H, cudaStream_t stream, __nv_bfloat16 const* qkv, __nv_fp8_e4m3* q8,
    __nv_fp8_e4m3* k8, __nv_fp8_e4m3* v8, float const* mul, float* amax, float eps, __nv_bfloat16 const* qw,
    __nv_bfloat16 const* kw, float const* cosp, float const* sinp, int cspb)
{
#define NRQ_CASE(C)                                                                                                    \
    case C: launchNrqFast<IL, C>(tokens, H, stream, qkv, q8, k8, v8, mul, amax, eps, qw, kw, cosp, sinp, cspb); break
    switch (nch)
    {
        NRQ_CASE(1);
        NRQ_CASE(2);
        NRQ_CASE(3);
        NRQ_CASE(4);
        NRQ_CASE(5);
        NRQ_CASE(6);
        NRQ_CASE(7);
        NRQ_CASE(8);
    default: TORCH_CHECK(false, "bad chunk count");
    }
#undef NRQ_CASE
}

// qkv [T, 3*H*D] bf16 to q8/k8/v8 [T, H*D] fp8.
void norm_rope_quant(torch::Tensor qkv, int64_t num_heads, double eps, torch::Tensor q_weight, torch::Tensor k_weight,
    torch::Tensor cos_emb, torch::Tensor sin_emb, bool interleave, int64_t cos_seq_per_batch, torch::Tensor quant_mul,
    torch::Tensor amax, torch::Tensor q8, torch::Tensor k8, torch::Tensor v8)
{
    TORCH_CHECK(qkv.is_contiguous() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2);
    TORCH_CHECK(cos_emb.scalar_type() == at::kFloat && sin_emb.scalar_type() == at::kFloat, "fp32 cos/sin only");
    int const H = static_cast<int>(num_heads);
    int64_t const D = qkv.size(1) / (3 * H);
    TORCH_CHECK(D == 128 && H <= 64, "head_dim 128, <= 64 heads");
    int const N = H * 128;
    int const tokens = static_cast<int>(qkv.size(0));
    bool const per_head_cos = cos_emb.size(-1) == N;
    auto const stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    auto aligned = [](void const* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
    bool const fast
        = !per_head_cos && aligned(qkv.data_ptr()) && aligned(cos_emb.data_ptr()) && aligned(sin_emb.data_ptr());
    if (fast)
    {
        if (tokens == 0)
            return;
        int const nch = (N + 1023) / 1024;
        auto args
            = std::make_tuple(nch, tokens, H, stream.stream(), reinterpret_cast<__nv_bfloat16 const*>(qkv.data_ptr()),
                reinterpret_cast<__nv_fp8_e4m3*>(q8.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(k8.data_ptr()),
                reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()), quant_mul.data_ptr<float>(), amax.data_ptr<float>(),
                static_cast<float>(eps), reinterpret_cast<__nv_bfloat16 const*>(q_weight.data_ptr()),
                reinterpret_cast<__nv_bfloat16 const*>(k_weight.data_ptr()), cos_emb.data_ptr<float>(),
                sin_emb.data_ptr<float>(), static_cast<int>(cos_seq_per_batch));
        if (interleave)
            std::apply(dispatchNrqFast<true>, args);
        else
            std::apply(dispatchNrqFast<false>, args);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    constexpr int ROWS = 2;
    int const cosStride = per_head_cos ? N : 128;
    size_t const smem
        = 2 * ROWS * N * sizeof(__nv_bfloat16) + 2 * ROWS * cosStride * sizeof(float) + ROWS * 8 * sizeof(float);
    cudaLaunchAttribute attrs[1] = {};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((tokens + ROWS - 1) / ROWS);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
#define LAUNCH(IL, PH)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        auto* kptr = fusedDiTQKNormRopeFp8QuantKernel<128, IL, PH, float>;                                             \
        cudaFuncSetAttribute(reinterpret_cast<void const*>(kptr), cudaFuncAttributeMaxDynamicSharedMemorySize, smem);  \
        cudaLaunchKernelEx(&cfg, kptr, reinterpret_cast<__nv_bfloat16 const*>(qkv.data_ptr()),                         \
            reinterpret_cast<__nv_fp8_e4m3*>(q8.data_ptr()), reinterpret_cast<__nv_fp8_e4m3*>(k8.data_ptr()),          \
            reinterpret_cast<__nv_fp8_e4m3*>(v8.data_ptr()), quant_mul.data_ptr<float>(), amax.data_ptr<float>(), H,   \
            H, H, static_cast<float>(eps), reinterpret_cast<__nv_bfloat16 const*>(q_weight.data_ptr()),                \
            reinterpret_cast<__nv_bfloat16 const*>(k_weight.data_ptr()), cos_emb.data_ptr<float>(),                    \
            sin_emb.data_ptr<float>(), tokens, static_cast<int>(cos_seq_per_batch));                                   \
    } while (0)
    if (interleave)
    {
        if (per_head_cos)
            LAUNCH(true, true);
        else
            LAUNCH(true, false);
    }
    else
    {
        if (per_head_cos)
            LAUNCH(false, true);
        else
            LAUNCH(false, false);
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Writes the V scale and its reciprocal into mul[2].
void v_scale(torch::Tensor qkv, int64_t num_heads, torch::Tensor amax_v, torch::Tensor mul, torch::Tensor scale_v)
{
    int const N = static_cast<int>(num_heads) * 128;
    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, qkv.get_device());
    auto const* src = reinterpret_cast<__nv_bfloat16 const*>(qkv.data_ptr());
    // Per-block partials replace the memset and the atomics.
    int const grid = vAmaxGrid(sms);
    auto partial = torch::empty({grid}, qkv.options().dtype(at::kFloat));
    vAmaxRowKernel<<<grid, 256, 0, stream>>>(src, qkv.size(0), N, partial.data_ptr<float>());
    vScaleReduceKernel<<<1, 1024, 0, stream>>>(partial.data_ptr<float>(), grid, amax_v.data_ptr<float>(),
        mul.data_ptr<float>() + 2, scale_v.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("norm_rope_quant", &norm_rope_quant);
    m.def("v_scale", &v_scale);
}
