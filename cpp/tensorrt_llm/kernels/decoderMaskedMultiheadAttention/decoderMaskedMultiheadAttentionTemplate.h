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
#pragma once

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/math.h"
#include <assert.h>
#include <float.h>
#include <type_traits>

// Multi-block mmha kernel can only be selected when CUDA >= 11.7
#if (CUDART_VERSION >= 11070)
#define ENABLE_MULTI_BLOCK_OPTION
#endif

#ifdef ENABLE_MULTI_BLOCK_OPTION
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/std/bit>
#endif // ENABLE_MULTI_BLOCK_OPTION

namespace tensorrt_llm
{
namespace kernels
{

// Use HMMA to compute with FP16/BF16 inputs and FP32 accumulators.
// #define MMHA_USE_HMMA

// Pre-scale Q or P to reduce number of instructions for dequantizing KV cache.
// If you notice a decrease in accuracy when the fp8 kv cache is enabled,
//  consider disabling the two flags.
#ifdef ENABLE_FP8
// Apply the FP8 scaling to Q instead of K.
#define MMHA_FP8_SCALE_Q_INSTEAD_OF_K
// Apply the FP8 scaling to P instead of V.
#define MMHA_FP8_SCALE_P_INSTEAD_OF_V
#endif // !defined ENABLE_FP8

// Below are knobs to extend FP32 accumulation for higher FP16 accuracy

// Does not seem to affect the accuracy that much
#define MMHA_USE_FP32_ACCUM_FOR_FMA

// Seems to slightly improve the accuracy
#define MMHA_USE_FP32_ACCUM_FOR_OUT

#if 0 && defined(MMHA_USE_FP32_ACCUM_FOR_OUT)
 // Does not seem to improve the accuracy
 //#define MMHA_USE_FP32_ACCUM_FOR_LOGITS
#endif

namespace mmha
{

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// We use the following terminology to describe the different dimensions.
//
// B:  Batch size (number of sequences),
// L:  Sequence length,
// D:  Hidden dimension,
// H:  Number of heads,
// Dh: Hidden dimension per head - Dh = D / H.
//
// The different kernels assign a threadblock for B x H pair. The grid has size (1, B, H). We use
// 256 threads per block to maximum occupancy and performance.
//
// Each threadblock loads Dh values from Q and its associated bias. The kernels run a loop to
// compute Q * K^T where K is loaded from a cache buffer -- except for the current timestep. The
// cache buffer helps with memory accesses and contains keys with bias.
//
// The layout of the cache buffer for the keys/values is [B, H, L, Dh]
// where the fastest moving dimension (contiguous data) is the rightmost one.
// Contiguous threads will read one hidden_dimension per LDG unless we need more than 32 threads.
//
// The different kernels use 1 ~ 32 threads per key (THREADS_PER_KEY). The size of the LDGs
// is always 16bytes (8 bytes for 8bit cache). Each thread sums Dh / THREADS_PER_KEY elements. At
// the end of each iteration of the Q * K^T loop, we perform a reduction between lanes using an
// HMMA instruction (Tensor Core). Each Q * K^T value is stored in shared memory in FP32.
//
// After that loop, a parallel softmax is computed across the different Q * K^T values stored in
// shared memory.
//
// The kernel ends with a loop over the values in V. We use THREADS_PER_VALUE to control how many
// timesteps are computed by loop iteration. As with the keys, the values are read from a cache
// except for the current timestep. The layout of the cache buffer for the values is same as the key,
// which is [B, H, L, Dh].
//
// Note that we have remapped key layout to make sure it shares the same pattern as value [B, H, L, Dh].
// It helps coalescing memory access, and reducing register pressure.

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int Dh_MAX>
struct Qk_vec_m_
{
};

template <>
struct Qk_vec_m_<float, 32>
{
    using Type = float;
};

template <>
struct Qk_vec_m_<float, 64>
{
    using Type = float2;
};

template <>
struct Qk_vec_m_<float, 128>
{
    using Type = float4;
};

template <>
struct Qk_vec_m_<float, 256>
{
    using Type = float4;
};

template <>
struct Qk_vec_m_<uint16_t, 32>
{
    using Type = uint32_t;
};

template <>
struct Qk_vec_m_<uint16_t, 64>
{
    using Type = uint32_t;
};

template <>
struct Qk_vec_m_<uint16_t, 128>
{
    using Type = uint2;
};

template <>
struct Qk_vec_m_<uint16_t, 256>
{
    using Type = uint4;
};
#ifdef ENABLE_BF16
template <>
struct Qk_vec_m_<__nv_bfloat16, 32>
{
    using Type = __nv_bfloat162;
};

template <>
struct Qk_vec_m_<__nv_bfloat16, 64>
{
    using Type = __nv_bfloat162;
};

template <>
struct Qk_vec_m_<__nv_bfloat16, 128>
{
    using Type = bf16_4_t;
};

template <>
struct Qk_vec_m_<__nv_bfloat16, 256>
{
    using Type = bf16_8_t;
};
#endif // ENABLE_BF16

#ifdef ENABLE_FP8
template <>
struct Qk_vec_m_<__nv_fp8_e4m3, 32>
{
    using Type = fp8_4_t;
};

template <>
struct Qk_vec_m_<__nv_fp8_e4m3, 64>
{
    using Type = fp8_4_t;
};

template <>
struct Qk_vec_m_<__nv_fp8_e4m3, 128>
{
    using Type = fp8_4_t;
};

template <>
struct Qk_vec_m_<__nv_fp8_e4m3, 256>
{
    using Type = fp8_4_t;
};
#endif // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int Dh>
struct Qk_vec_k_
{
    using Type = typename Qk_vec_m_<T, Dh>::Type;
};
#ifdef ENABLE_FP8
template <>
struct Qk_vec_k_<__nv_fp8_e4m3, 32>
{
    using Type = float4;
};

template <>
struct Qk_vec_k_<__nv_fp8_e4m3, 64>
{
    using Type = float4;
};

template <>
struct Qk_vec_k_<__nv_fp8_e4m3, 128>
{
    using Type = float4;
};

template <>
struct Qk_vec_k_<__nv_fp8_e4m3, 256>
{
    using Type = float4;
};
#endif // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int V_VEC_SIZE>
struct V_vec_m_
{
};

template <>
struct V_vec_m_<float, 1>
{
    using Type = float;
};

template <>
struct V_vec_m_<float, 2>
{
    using Type = float2;
};

template <>
struct V_vec_m_<float, 4>
{
    using Type = float4;
};

template <>
struct V_vec_m_<float, 8>
{
    using Type = Float8_;
};

template <>
struct V_vec_m_<uint16_t, 2>
{
    using Type = uint32_t;
};

template <>
struct V_vec_m_<uint16_t, 4>
{
    using Type = uint2;
};

template <>
struct V_vec_m_<uint16_t, 8>
{
    using Type = uint4;
};
#ifdef ENABLE_BF16
template <>
struct V_vec_m_<__nv_bfloat16, 2>
{
    using Type = __nv_bfloat162;
};

template <>
struct V_vec_m_<__nv_bfloat16, 4>
{
    using Type = bf16_4_t;
};

template <>
struct V_vec_m_<__nv_bfloat16, 8>
{
    using Type = bf16_8_t;
};
#endif // ENABLE_BF16

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int V_VEC_SIZE>
struct V_vec_k_
{
    using Type = typename V_vec_m_<T, V_VEC_SIZE>::Type;
};
#ifdef ENABLE_FP8
template <>
struct V_vec_k_<__nv_fp8_e4m3, 4>
{
    using Type = float4;
};

template <>
struct V_vec_k_<__nv_fp8_e4m3, 8>
{
    using Type = float4;
};

template <>
struct V_vec_k_<__nv_fp8_e4m3, 16>
{
    using Type = float4;
};
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Reuse V_vec traits as key and value share the same layout.
template <typename T, int K_VEC_SIZE>
struct K_vec_m_
{
    using Type = typename V_vec_m_<T, K_VEC_SIZE>::Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int K_VEC_SIZE>
struct K_vec_k_
{
    using Type = typename K_vec_m_<T, K_VEC_SIZE>::Type;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
template <typename T>
struct Qk_vec_accum_fp32_
{
};

template <>
struct Qk_vec_accum_fp32_<float>
{
    using Type = float;
};

template <>
struct Qk_vec_accum_fp32_<float2>
{
    using Type = float2;
};

template <>
struct Qk_vec_accum_fp32_<float4>
{
    using Type = float4;
};

// template<> struct Qk_vec_accum_fp32_<uint16_t> { using Type = float;        };
template <>
struct Qk_vec_accum_fp32_<uint32_t>
{
    using Type = float2;
};

template <>
struct Qk_vec_accum_fp32_<uint2>
{
    using Type = Float4_;
};

template <>
struct Qk_vec_accum_fp32_<uint4>
{
    using Type = Float8_;
};

template <>
struct Qk_vec_accum_fp32_<__nv_bfloat16>
{
    using Type = float;
};

template <>
struct Qk_vec_accum_fp32_<__nv_bfloat162>
{
    using Type = float2;
};

template <>
struct Qk_vec_accum_fp32_<bf16_4_t>
{
    using Type = Float4_;
};

template <>
struct Qk_vec_accum_fp32_<bf16_8_t>
{
    using Type = Float8_;
};

#ifdef ENABLE_FP8
// template<>
// struct Qk_vec_accum_fp32_<fp8_2_t> {
//     using Type = float2;
// };
template <>
struct Qk_vec_accum_fp32_<fp8_4_t>
{
    using Type = Float4_;
};

// template<>
// struct Qk_vec_accum_fp32_<fp8_8_t> {
//     using Type = Float4_;
// };
#endif // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct K_vec_accum_fp32_
{
};

template <>
struct K_vec_accum_fp32_<float>
{
    using Type = float;
};

template <>
struct K_vec_accum_fp32_<float2>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<float4>
{
    using Type = float4;
};

template <>
struct K_vec_accum_fp32_<Float8_>
{
    using Type = Float8_;
};

template <>
struct K_vec_accum_fp32_<uint32_t>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<uint2>
{
    using Type = Float4_;
};

template <>
struct K_vec_accum_fp32_<uint4>
{
    using Type = Float8_;
};

template <>
struct K_vec_accum_fp32_<__nv_bfloat16>
{
    using Type = float;
};

template <>
struct K_vec_accum_fp32_<__nv_bfloat162>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<bf16_4_t>
{
    using Type = Float4_;
};

template <>
struct K_vec_accum_fp32_<bf16_8_t>
{
    using Type = Float8_;
};
#ifdef ENABLE_FP8
template <>
struct K_vec_accum_fp32_<__nv_fp8_e4m3>
{
    using Type = float;
};

template <>
struct K_vec_accum_fp32_<fp8_2_t>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<fp8_4_t>
{
    using Type = Float4_;
};

template <>
struct K_vec_accum_fp32_<fp8_8_t>
{
    using Type = Float8_;
};
#endif // ENABLE_FP8

template <>
struct K_vec_accum_fp32_<int8_t>
{
    using Type = float;
};

template <>
struct K_vec_accum_fp32_<int16_t>
{
    using Type = float2;
};

template <>
struct K_vec_accum_fp32_<int32_t>
{
    using Type = Float4_;
};

template <>
struct K_vec_accum_fp32_<int64_t>
{
    using Type = Float8_;
};

#endif // MMHA_USE_FP32_ACCUM_FOR_FMA

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
template <typename T>
struct V_vec_accum_fp32_
{
};

template <>
struct V_vec_accum_fp32_<float>
{
    using Type = float;
};

template <>
struct V_vec_accum_fp32_<float2>
{
    using Type = float2;
};

template <>
struct V_vec_accum_fp32_<float4>
{
    using Type = float4;
};

template <>
struct V_vec_accum_fp32_<uint32_t>
{
    using Type = float2;
};

template <>
struct V_vec_accum_fp32_<uint2>
{
    using Type = Float4_;
};

template <>
struct V_vec_accum_fp32_<uint4>
{
    using Type = Float8_;
};
#ifdef ENABLE_BF16
template <>
struct V_vec_accum_fp32_<__nv_bfloat162>
{
    using Type = float2;
};

template <>
struct V_vec_accum_fp32_<bf16_4_t>
{
    using Type = Float4_;
};

template <>
struct V_vec_accum_fp32_<bf16_8_t>
{
    using Type = Float8_;
};
#endif // ENABLE_BF16
#ifdef ENABLE_FP8
// template<>
// struct V_vec_accum_fp32_<fp8_2_t> {
//     using Type = float2;
// };
template <>
struct V_vec_accum_fp32_<fp8_4_t>
{
    using Type = Float4_;
};

// template<>
// struct V_vec_accum_fp32_<fp8_8_t> {
//     using Type = Float4_;
// };
#endif // ENABLE_FP8
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tout, typename Tin>
__inline__ __device__ constexpr Tout vec_conversion(Tin const& x)
{
    static_assert(std::is_same<Tout, Tin>::value, "Type mismatch");
    return x;
}

template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint4>(uint4 const& a)
{
    Float8_ fc;
    fc.x = half2_to_float2(a.x);
    fc.y = half2_to_float2(a.y);
    fc.z = half2_to_float2(a.z);
    fc.w = half2_to_float2(a.w);
    return fc;
}

#ifdef ENABLE_BF16
template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, bf16_8_t>(bf16_8_t const& a)
{
    Float8_ fc;
    fc.x = bf1622float2(a.x);
    fc.y = bf1622float2(a.y);
    fc.z = bf1622float2(a.z);
    fc.w = bf1622float2(a.w);
    return fc;
}
#endif // ENABLE_BF16

#ifdef ENABLE_FP8
// fp8_t
template <>
__inline__ __device__ float vec_conversion<float, __nv_fp8_e4m3>(__nv_fp8_e4m3 const& a)
{
    return float(a);
}

template <>
__inline__ __device__ __nv_fp8_e4m3 vec_conversion<__nv_fp8_e4m3, float>(float const& a)
{
    return __nv_fp8_e4m3(a);
}

// fp8_2_t
template <>
__inline__ __device__ float2 vec_conversion<float2, fp8_2_t>(fp8_2_t const& a)
{
    return float2(a);
}

template <>
__inline__ __device__ fp8_2_t vec_conversion<fp8_2_t, float2>(float2 const& a)
{
    return fp8_2_t(a);
}

// fp8_4_t
template <>
__inline__ __device__ float4 vec_conversion<float4, fp8_4_t>(fp8_4_t const& a)
{
    return float4(a);
}

template <>
__inline__ __device__ fp8_4_t vec_conversion<fp8_4_t, float4>(float4 const& a)
{
    return fp8_4_t(a);
}
#endif // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS_PER_KEY, typename Q_vec, typename K_vec, int N>
inline __device__ float qk_dot_(const Q_vec (&q)[N], const K_vec (&k)[N])
{
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename K_vec_accum_fp32_<K_vec>::Type;
#else
    using K_vec_accum = K_vec;
#endif
    // Compute the parallel products for Q*K^T (treat vector lanes separately).
    K_vec_accum qk_vec = mul<K_vec_accum, Q_vec, K_vec>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii)
    {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }

    // Finalize the reduction across lanes.
    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2)
    {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

template <int THREADS_PER_KEY, typename Q_vec, typename K_vec, int N>
inline __device__ float qk_scale_dot_(const Q_vec (&q)[N], const K_vec (&k)[N], float const k_scale)
{
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename K_vec_accum_fp32_<K_vec>::Type;
#else
    using K_vec_accum = K_vec;
#endif
    // Compute the parallel products for Q*K^T (treat vector lanes separately).
    K_vec_accum k_vec = mul<K_vec_accum, float, K_vec>(k_scale, k[0]);
    K_vec_accum qk_vec = mul<K_vec_accum, Q_vec, K_vec_accum>(q[0], k_vec);
#pragma unroll
    for (int ii = 1; ii < N; ++ii)
    {
        K_vec_accum k_vec = mul<K_vec_accum, float, K_vec>(k_scale, k[ii]);
        qk_vec = fma(q[ii], k_vec, qk_vec);
    }

    // Finalize the reduction across lanes.
    float qk = sum(qk_vec);
#pragma unroll
    for (int mask = THREADS_PER_KEY / 2; mask >= 1; mask /= 2)
    {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int THREADS_PER_KEY>
struct Qk_dot
{
    template <typename Q_vec, typename K_vec, int N>
    static inline __device__ float dot(const Q_vec (&q)[N], const K_vec (&k)[N])
    {
        return qk_dot_<THREADS_PER_KEY>(q, k);
    }

    template <typename Q_vec, typename K_vec, int N>
    static inline __device__ float scale_dot(const Q_vec (&q)[N], const K_vec (&k)[N], float const k_scale)
    {
#ifdef MMHA_USE_HMMA
        static_assert("HMMA doesn't support k scales");
#endif // MMHA_USE_HMMA
        return qk_scale_dot_<THREADS_PER_KEY>(q, k, k_scale);
    }

    template <int WARP_SIZE = 32>
    static inline __device__ bool is_leader(int const tidx)
    {
        return (tidx % THREADS_PER_KEY) == 0;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename K_vec>
inline __device__ void hmma_fp32(float4& c, K_vec const& a, K_vec b)
{
    // Not supported.
    assert(false);
}

template <>
inline __device__ void hmma_fp32(float4& c, uint32_t const& a, uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
        "    {%0, %1, %2, %3}, \n"
        "    {%4, %5}, \n"
        "    {%6}, \n"
        "    {%0, %1, %2, %3}; \n"
        : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w)
        : "r"(a), "r"(a), "r"(b));
}

template <>
inline __device__ void hmma_fp32(float4& c, uint2 const& a, uint2 b)
{
    hmma_fp32(c, a.x, b.x);
    hmma_fp32(c, a.y, b.y);
}

template <>
inline __device__ void hmma_fp32(float4& c, uint4 const& a, uint4 b)
{
    hmma_fp32(c, a.x, b.x);
    hmma_fp32(c, a.y, b.y);
    hmma_fp32(c, a.z, b.z);
    hmma_fp32(c, a.w, b.w);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename K_vec, int THREADS_PER_KEY, int N>
inline __device__ float qk_hmma_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750

    // Each quad computes its partial result.
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        hmma_fp32(acc, q[ii], k[ii]);
    }

    // The position inside the warp.
    int lane = threadIdx.x % 32;

    // The position inside the HMMA instruction.
    int row = lane / 4;
    int col = lane % 4 * 2;

    // The result. Only 1 thread in each quad owns a valid value.
    //
    // Row 0, it's lane  0 (col 0) in acc.x.
    // Row 1, it's lane  4 (col 0) in acc.y.
    // Row 2, it's lane  9 (col 2) in acc.x.
    // Row 3, it's lane 13 (col 2) in acc.y.
    // Row 4, it's lane 18 (col 4) in acc.x.
    // Row 5, it's lane 22 (col 4) in acc.y.
    // Row 6, it's lane 27 (col 6) in acc.x.
    // Row 7, it's lane 31 (col 6) in acc.y.
    //
    float result = (row == col) ? acc.x : acc.y;

    // Do the reduction inside the warp.
    if (THREADS_PER_KEY > 4)
    {
        result += __shfl_xor_sync(unsigned(-1), result, 4);
    }
    if (THREADS_PER_KEY > 8)
    {
        result += __shfl_xor_sync(unsigned(-1), result, 9);
    }
    if (THREADS_PER_KEY > 16)
    {
        result += __shfl_xor_sync(unsigned(-1), result, 18);
    }

    // The warp leader has the correct value.
    return result;

#else // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 750
    return 0.f;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS_PER_KEY>
struct Qk_dot<uint16_t, THREADS_PER_KEY>
{
    template <typename Q_vec, typename K_vec, int N>
    static inline __device__ float dot(const Q_vec (&q)[N], const K_vec (&k)[N])
    {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
        return qk_hmma_dot_<K_vec, THREADS_PER_KEY, N>(q, k);
#else
        return qk_dot_<THREADS_PER_KEY>(q, k);
#endif // defined MMHA_USE_HMMA
    }

    template <typename Q_vec, typename K_vec, int N>
    static inline __device__ float scale_dot(const Q_vec (&q)[N], const K_vec (&k)[N], float const k_scale)
    {
#ifdef MMHA_USE_HMMA
        static_assert("HMMA doesn't support k scales");
#endif // MMHA_USE_HMMA
        return qk_scale_dot_<THREADS_PER_KEY>(q, k, k_scale);
    }

    template <int WARP_SIZE = 32>
    static inline __device__ bool is_leader(int const tidx)
    {
        // Use HMMA.FP32, leader threads are in the diagonal roughly (0, 4, 9, 13, 18, 22, 27, 31).
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
        int leader = 0;
        // The thread position inside the warp.
        int lane = tidx % WARP_SIZE;
        if (THREADS_PER_KEY == 4)
        {
            leader = int(lane / 8);
        }
        else
        {
            leader = int(lane / THREADS_PER_KEY) * int(THREADS_PER_KEY / 8);
        }
#else
        bool const leader = 0;
#endif // defined MMHA_USE_HMMA
        return (tidx % THREADS_PER_KEY) == leader;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tk, typename V_vec_accum, typename V_vec_m, bool INT8_KV_CACHE, bool FP8_KV_CACHE>
inline __device__ void Logit_value_fma(
    V_vec_accum& out, Tk const* logits_smem, V_vec_m const& v_vec, float const v_scale, bool const is_mask)
{
#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)
    float logit = is_mask ? 0.f : reinterpret_cast<float*>(logits_smem)[0];
    if constexpr (INT8_KV_CACHE)
    {
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out = fma(logit, cast_to_float(v_vec_), out);
    }
    else if constexpr (FP8_KV_CACHE)
    {
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
        out = fma(logit, cast_to_float(v_vec), out);
#else
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out = fma(logit, cast_to_float(v_vec_), out);
#endif // MMHA_FP8_SCALE_P_INSTEAD_OF_V
    }
    else
    {
        out = fma(logit, cast_to_float(v_vec), out);
    }
#else // MMHA_USE_FP32_ACCUM_FOR_LOGITS
    Tk logit = is_mask ? Tk(0.f) : logits_smem[0];
    if constexpr (INT8_KV_CACHE)
    {
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out = fma(logit, v_vec_, out);
    }
    else if constexpr (FP8_KV_CACHE)
    {
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
        out = fma(logit, v_vec, out);
#else
        V_vec_accum v_vec_ = mul<V_vec_accum, float, V_vec_m>(v_scale, v_vec);
        out = fma(logit, v_vec_, out);
#endif // MMHA_FP8_SCALE_P_INSTEAD_OF_V
    }
    else
    {
        out = fma(logit, v_vec, out);
    }
#endif // MMHA_USE_FP32_ACCUM_FOR_LOGITS
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int WARPS_PER_BLOCK, int WARP_SIZE = 32>
inline __device__ float block_sum(float* red_smem, float sum)
{

    // Decompose the thread index into warp / lane.
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
    {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Warp leaders store the data to shared memory.
    if (lane == 0)
    {
        red_smem[warp] = sum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The warps compute the final sums.
    if (lane < WARPS_PER_BLOCK)
    {
        sum = red_smem[lane];
    }

// Parallel reduction inside the warp.
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2)
    {
        sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
    }

    // Broadcast to other threads.
    return __shfl_sync(uint32_t(-1), sum, 0);
}

#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float cast_to_float(float u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(float2 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 cast_to_float(float4 u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(Float4_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(Float8_ u)
{
    return u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(uint32_t u)
{
    return half2_to_float2(u);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(uint2 u)
{
    Float4_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(uint4 u)
{
    Float8_ tmp;
    tmp.x = half2_to_float2(u.x);
    tmp.y = half2_to_float2(u.y);
    tmp.z = half2_to_float2(u.z);
    tmp.w = half2_to_float2(u.w);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 cast_to_float(__nv_bfloat162 u)
{
    float2 tmp;
    tmp = __bfloat1622float2(u);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ cast_to_float(bf16_4_t u)
{
    Float4_ tmp;
    tmp.x = __bfloat1622float2(u.x);
    tmp.y = __bfloat1622float2(u.y);
    return tmp;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ cast_to_float(bf16_8_t u)
{
    Float8_ tmp;
    tmp.x = __bfloat1622float2(u.x);
    tmp.y = __bfloat1622float2(u.y);
    tmp.z = __bfloat1622float2(u.z);
    tmp.w = __bfloat1622float2(u.w);
    return tmp;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ __host__ T div(T m, T n)
{
    return m / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct kernel_type_t
{
    using Type = T;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Compute the largest supported head size (dh_max). It must be the smallest power-of-2 that is not strictly smaller
// than the head size (dh).
inline __device__ __host__ constexpr unsigned dh_max(unsigned dh)
{
    return next_power_of_two(mmha::const_max(dh, 32u));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ __host__ constexpr unsigned threads_per_value(unsigned dh_max)
{
    return dh_max * sizeof(T) / 16;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, unsigned Dh_MAX>
inline __device__ __host__ constexpr unsigned threads_per_key()
{
    // Since we want to perform the reduction entirely within a warp, the number of threads per key
    // is capped at 32.
    constexpr unsigned threads = (unsigned) (Dh_MAX * sizeof(T) / 16u);
    if ((threads & (threads - 1)) != 0)
    {
        assert(false); // Not a power of two.
    }
    return std::min(32u, threads);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Specialized launch bounds for certain cases, which helps increase occupancy.
// Keep other cases untouched as there might be register spilling.

template <typename T, typename Tcache, unsigned THREADS_PER_BLOCK, unsigned Dh_MAX, bool DO_CROSS_ATTENTION,
    bool HAS_BEAMS, bool POS_SHIFT>
struct Launch_bounds_config
{
    // By default, we will not use launch bounds.
    static constexpr int MAX_THREADS_PER_BLOCK = 0;
    static constexpr int MIN_BLOCKS_PER_SM = 0;
};

template <>
struct Launch_bounds_config<uint16_t, __nv_fp8_e4m3, 256u, 64u, false, false, false>
{
    static constexpr int MAX_THREADS_PER_BLOCK = 256u;
    static constexpr int MIN_BLOCKS_PER_SM = 4u;
};

// Llama with FP8 KV Cache.
template <>
struct Launch_bounds_config<uint16_t, __nv_fp8_e4m3, 256u, 128u, false, false, false>
{
    static constexpr int MAX_THREADS_PER_BLOCK = 256u;
    static constexpr int MIN_BLOCKS_PER_SM = 4u;
};

// GPTJ With Beam Searching and FP8 KV Cache.
template <>
struct Launch_bounds_config<uint16_t, __nv_fp8_e4m3, 256u, 256u, false, true, false>
{
    static constexpr int MAX_THREADS_PER_BLOCK = 256u;
    static constexpr int MIN_BLOCKS_PER_SM = 3u;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ constexpr uint32_t shfl_mask(int threads)
{
    assert(threads <= 32);
    return threads == 32 ? -1u : (1u << threads) - 1u;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename T_VEC, unsigned VECS_PER_CHUNK>
__device__ inline constexpr uint2 chunk_index(unsigned tidx)
{
    // The chunk associated with the thread.
    auto const idx_chunk = tidx / VECS_PER_CHUNK;

    // The position of the T_VEC vector in that chunk associated with the thread.
    static_assert(sizeof(T_VEC) % sizeof(T) == 0);
    unsigned constexpr kVecSize{sizeof(T_VEC) / sizeof(T)};
    auto const idx_vec = (tidx % VECS_PER_CHUNK) * kVecSize;

    return uint2{idx_chunk, idx_vec};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The type of the inputs. Supported types: float, uint16_t, nv_bfloat16.
    typename T,
    // The type of the cache.
    typename Tcache,
    // The type of the shift key cache.
    typename TKcache,
    // Type of struct containing KV cache
    typename KVCacheBuffer,
    // Type of struct containing K cache to read past keys
    typename KCacheBuffer,
    // The hidden dimension per head.
    unsigned Dh,
    // The number of threads in a threadblock.
    unsigned THREADS_PER_BLOCK,
    // Whether cross attention is enabled
    bool DO_CROSS_ATTENTION,
    // Whether has beams.
    bool HAS_BEAMS,
    // Whether enable multi-block mode for long-sequence-length.
    bool DO_MULTI_BLOCK = false,
    // Whether enable position shift for streamingllm
    bool POS_SHIFT = false,
    // Whether to compute and apply block sparse attention mask
    bool BLOCK_SPARSE_ATTN = false,
    // Whether compute implicit relative attention bias on the fly.
    bool IMPLICIT_REL_ATTN_BIAS = false,
    // Whether enable attention logit softcapping scale.
    bool ATTN_LOGIT_SOFTCAPPING = false,
    // The number of threads per key.
    unsigned THREADS_PER_KEY = threads_per_key<T, dh_max(Dh)>(),
    // The number of threads per value.
    unsigned THREADS_PER_VALUE = threads_per_value<T>(dh_max(Dh)),
    // The unroll factor for loading from K cache.
    // Set it default to 4 for higher occupancy (by reducing registers usage).
    unsigned K_LOOP_UNROLL = 4,
    // The unroll factor for loading from V cache.
    unsigned V_LOOP_UNROLL = 8,
    // Launch bounds
    unsigned MAX_THEADS_PER_BLOCK
    = Launch_bounds_config<T, Tcache, THREADS_PER_BLOCK, dh_max(Dh), DO_CROSS_ATTENTION, HAS_BEAMS, POS_SHIFT>()
          .MAX_THREADS_PER_BLOCK,
    unsigned MIN_BLOCKS_PER_SM
    = Launch_bounds_config<T, Tcache, THREADS_PER_BLOCK, dh_max(Dh), DO_CROSS_ATTENTION, HAS_BEAMS, POS_SHIFT>()
          .MIN_BLOCKS_PER_SM>
__global__ void __launch_bounds__(MAX_THEADS_PER_BLOCK, MIN_BLOCKS_PER_SM) masked_multihead_attention_kernel(
    Multihead_attention_params<T, DO_CROSS_ATTENTION> params, KVCacheBuffer kvCacheBuffer, KCacheBuffer pastKCache)
{
    using Tk = typename kernel_type_t<T>::Type;
    // Use 8bit cache.
    static constexpr bool ENABLE_8BITS_K_CACHE = sizeof(TKcache) == 1;
    static constexpr bool ENABLE_8BITS_KV_CACHE = sizeof(Tcache) == 1;
    // FP8 KV Cache.
    static constexpr bool FP8_K_CACHE = std::is_same<TKcache, __nv_fp8_e4m3>::value;
    static constexpr bool FP8_KV_CACHE = std::is_same<Tcache, __nv_fp8_e4m3>::value;
    // INT8 KV Cache.
    static constexpr bool INT8_KV_CACHE = std::is_same<Tcache, int8_t>::value;

    // The size of a warp.
    constexpr unsigned WARP_SIZE{32};
    // The number of warps in a threadblock.
    constexpr unsigned WARPS_PER_BLOCK{THREADS_PER_BLOCK / WARP_SIZE};

    // The maximum hidden size per head.
    constexpr auto Dh_MAX = dh_max(Dh);
    constexpr bool IS_Dh_MAX = Dh == Dh_MAX;
    static_assert(Dh_MAX >= WARP_SIZE);
    static_assert(Dh_MAX >= Dh);
    // Only instantiate few head sizes for implicit relative attention bias in order to save compilation time.
    static_assert(!IMPLICIT_REL_ATTN_BIAS || Dh == 32 || Dh == 64 || Dh == 128);

    // The maximum sequence length in the cyclic kv_cache, i.e., an upper bound on L.
    // Note that the maximum sequence length supported by the model might be greater than this.
    // Note max_attention_window_size is maximum of cyclic_attention_window_size among all layers.
    // By default, you can assume that they are the same.
    auto const cyclic_kv_cache_len = params.cyclic_attention_window_size;
    // The chunked attention size.
    auto const chunked_attention_size = static_cast<unsigned>(params.chunked_attention_size);
    // The number of sink tokens in kv cache to support streamingllm
    auto const sink_token_len = static_cast<unsigned>(params.sink_token_length);
    // The current timestep (including paddings).
    // It is only used to calculate the smem stride.
    auto const timestep = static_cast<unsigned>(DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep);

#ifdef ENABLE_MULTI_BLOCK_OPTION
    constexpr bool MULTI_BLOCK_FLAG = DO_MULTI_BLOCK;
#else
    constexpr bool MULTI_BLOCK_FLAG = false;
#endif

    // Use smem_size_in_bytes (above) to determine the amount of shared memory.
    extern __shared__ char smem_[];

    // The shared memory for the Q*K^T values and partial logits in softmax.
    auto qk_smem = reinterpret_cast<float*>(smem_);

    __shared__ float qk_current_smem[1];

    // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
    char* logits_smem_ = smem_;
#ifndef MMHA_USE_FP32_ACCUM_FOR_LOGITS
    if (sizeof(Tk) != 4)
    {
        auto const max_timesteps
            = min(timestep, min(chunked_attention_size, static_cast<unsigned>(cyclic_kv_cache_len)));
        logits_smem_ += divUp(max_timesteps + 1, 4u) * 16;
    }
    Tk* logits_smem = reinterpret_cast<Tk*>(logits_smem_);
#else
    float* logits_smem = reinterpret_cast<float*>(logits_smem_);
#endif

    __shared__ Tk logits_current_smem[1];

    // The shared memory to do the final reduction for the output values. Reuse qk_smem.
    Tk* out_smem = reinterpret_cast<Tk*>(smem_);

    // The shared memory buffers for the block-wide reductions. One for max, one for sum.
    __shared__ float red_smem[WARPS_PER_BLOCK * 2];

    // A vector of Q or K elements for the current timestep.
    using Qk_vec_m = typename Qk_vec_m_<T, Dh_MAX>::Type; // with memory-used precision
    using Qk_vec_k = typename Qk_vec_k_<T, Dh_MAX>::Type; // with kernel-used precision
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using Qk_vec_accum = typename Qk_vec_accum_fp32_<Qk_vec_k>::Type;
#else
    using Qk_vec_accum = Qk_vec_k;
#endif

    // Make sure the hidden dimension per head is a multiple of the number of threads per key.
    static_assert(Dh_MAX % THREADS_PER_KEY == 0); // trivially satisfied since THREADS_PER_KEY in {1, 2, 4}

    // The number of elements per vector.
    // Each thread will handle 16 bytes.
    constexpr int K_VEC_SIZE = 16u / sizeof(T);
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % K_VEC_SIZE == 0);
    // The type of queries and keys for the math in the Q*K^T product.
    using K_vec_k = typename K_vec_k_<T, K_VEC_SIZE>::Type;
    // Only used when key cache is quantized to 8 bits.
    using K_vec_m = typename packed_type<TKcache, num_elems<K_vec_k>::value>::type;
#ifdef MMHA_USE_FP32_ACCUM_FOR_FMA
    using K_vec_accum = typename Qk_vec_accum_fp32_<K_vec_k>::Type;
#else
    using K_vec_accum = K_vec_k;
#endif

    // Use alignment for safely casting the shared buffers as Qk_vec_k and K_vec_k.
    // Shared memory to store Q inputs.
    __shared__ __align__(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk q_smem[Dh_MAX];
    __shared__ __align__(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk k_smem[Dh_MAX];

    // Make sure the hidden dimension per head is a multiple of the number of threads per value.
    static_assert(Dh_MAX % THREADS_PER_VALUE == 0); // trivially satisfied since THREADS_PER_VALUE == Dh_MAX / p

    // The number of elements per vector.
    constexpr int V_VEC_SIZE = Dh_MAX / THREADS_PER_VALUE;
    // A vector of V elements for the current timestep.
    using V_vec_k = typename V_vec_k_<T, V_VEC_SIZE>::Type;
    // Only used when value cache is quantized to 8 bits.
    using V_vec_m = typename packed_type<Tcache, num_elems<V_vec_k>::value>::type;
    static_assert(V_VEC_SIZE == sizeof(V_vec_k) / sizeof(T));

    // This could be one of the reasons to have a separate kernel for cross attention
    constexpr auto bias_smem_size = DO_CROSS_ATTENTION ? Dh_MAX : 1u;
    __shared__ __align__(mmha::const_max(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k)), sizeof(V_vec_k)))
        [[maybe_unused]] Tk bias_smem[bias_smem_size];

    // The number of elements per vector.
    constexpr unsigned QK_VEC_SIZE{sizeof(Qk_vec_m) / sizeof(T)};
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % QK_VEC_SIZE == 0);
    // We will use block wide reduction if needed
    // The number of vectors per Dh_MAX.
    constexpr unsigned QK_VECS_PER_Dh_MAX{Dh_MAX / QK_VEC_SIZE};
    static_assert(THREADS_PER_BLOCK >= QK_VECS_PER_Dh_MAX);

    // The batch/beam idx
    auto const batch_beam_idx = blockIdx.y;
    if (params.finished != nullptr && params.finished[batch_beam_idx])
    {
        return;
    }

    // The head.
    unsigned const hi{blockIdx.x};
    // The number of heads.
    auto const num_heads = static_cast<unsigned>(params.num_heads);
    // The number of heads for keys and values adjusted for MQA/GQA.
    auto const num_heads_kv = static_cast<unsigned>(params.num_kv_heads);
    // The head index of keys and values adjusted for MQA/GQA.
    auto const qhead_per_kv{num_heads / num_heads_kv};
    // The head index of keys and values adjusted for MQA/GQA.
    unsigned const hi_kv{(hi / qhead_per_kv)};

    // The thread in the block.
    unsigned const tidx{threadIdx.x};

    // The column tile along L dimension on K^T -- noted as T_c in flash-attention paper
    unsigned const c_tile{MULTI_BLOCK_FLAG ? blockIdx.z : 0};

    // Indicate if we need to compute the K/V cache element (add KV bias, IA3, RoPE, etc.) and update the cache.
    // For Self-Attention, it's always required.
    // For Cross-Attention, as everything is pre-computed,
    // in the context phase of the encoder, it's not needed in that kernel.
    // Therefore, HANDLE_KV is !DO_CROSS_ATTENTION and irrelevant of timestep.
    static constexpr bool HANDLE_KV{!DO_CROSS_ATTENTION};

    // While doing the product Q*K^T for the different keys we track the max.
    float qk_max = -FLT_MAX;

    float qk = 0.0F;

    // Do we have a relative attention bias?
    bool has_relative_attention_bias = params.relative_attention_bias != nullptr;
    // Do we have a logn scale ptr?
    bool has_logn_scaling = params.logn_scaling_ptr != nullptr;
    // IMPLICIT_REL_ATTN_BIAS:
    // Compute relative attention bias on the fly, with relative attention table [head_num/TP, num_buckets] passed in.
    // num_buckets passed as relative_attention_bias_stride, max_distance passed as params.max_distance
    // this is a common optimization for both self attention and cross attention
    int relative_attention_bias_stride
        = params.relative_attention_bias_stride; // num_buckets might be modified below, save it beforehand
    [[maybe_unused]] int max_distance = params.max_distance;

    // The actual sequence length excluding the paddings.
    // minus 1 because it includes the current timestep while tlength denotes the kv cache length.
    int const tlength = DO_CROSS_ATTENTION
        ? params.memory_length_per_sample[batch_beam_idx] - 1
        : (params.length_per_sample ? (params.length_per_sample[batch_beam_idx] - 1) : static_cast<int>(timestep));
    // When enable cyclic kv cache and one more block mode, we need to shift the index to the actual index in the
    // sequence. Otherwise, if the token is not the sink token, we need to add the bubblen length to the index.
    bool const enable_use_seq_idx_kv = kvCacheBuffer.mEnableOneMoreBlock && tlength > cyclic_kv_cache_len;
    int const shift_for_cyclic_kv = (enable_use_seq_idx_kv) ? tlength - cyclic_kv_cache_len : kvCacheBuffer.mBubbleLen;
    int const shift_for_cyclic_k = (enable_use_seq_idx_kv) ? tlength - cyclic_kv_cache_len : pastKCache.mBubbleLen;
    // The actual kv cache length.
    // Minus 1 because the current token is also included in the attention window.
    int kv_loop_length = min(tlength, cyclic_kv_cache_len - 1);
    // The bound of the kv token idx (tlength = 0 should not happen ideally, but add here for safety).
    int const kv_token_idx_bound = max(tlength - 1, 0);
    // The kv_token_start_offset. All tokens before kv_token_start_offset will be fully masked.
    int kv_token_start_offset = max(tlength - cyclic_kv_cache_len + 1, 0);
    // Only consider the current attention chunk if the chunked attention is used.
    if (params.chunked_attention_size_log2 > 0)
    {
        kv_token_start_offset = (tlength >> params.chunked_attention_size_log2) << params.chunked_attention_size_log2;
        kv_loop_length -= kv_token_start_offset;
    }
    // The shared context length for beam searching optimization (all points to beam 0).
    // TODO: with cyclic kv cache, we set it 0 for now (will optimize in the future)
    // as context kv cache might be overwritten by the new kv cache
    int const beam0_context_length
        = HAS_BEAMS && tlength > cyclic_kv_cache_len ? 0 : params.input_lengths[batch_beam_idx];

    // The offset in the Q and K buffer also accounts for the batch.
    auto const qk_vec_idx = tidx * QK_VEC_SIZE;
    auto const is_valid_qk_vec = qk_vec_idx < Dh;

    bool const load_qkv_quant = params.qkv_scale_quant_orig != nullptr;
    bool const write_attention_quant = params.attention_out_scale_orig_quant != nullptr;

    // Quant/Dequant scales for 8bits kv cache.
    using T_scale = typename kv_cache_scale_type_t<T, Tcache>::Type;
    T_scale kv_scale_orig_quant, k_scale_quant_orig;
    float const k_scale_quant_orig_f = (ENABLE_8BITS_K_CACHE ? params.kv_scale_quant_orig[0] : 1.0f);
    float const kv_scale_quant_orig_f = (ENABLE_8BITS_KV_CACHE ? params.kv_scale_quant_orig[0] : 1.0f);
    convert_from_float(&k_scale_quant_orig, k_scale_quant_orig_f);
    convert_from_float(&kv_scale_orig_quant, (ENABLE_8BITS_KV_CACHE ? params.kv_scale_orig_quant[0] : 1.0f));

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // Up to QK_VECS_PER_Dh_MAX threads load Q and K + the bias values for the current timestep.
    // Trigger the loads from the Q and K buffers.
    Qk_vec_k q, k, q_bias, k_bias;
    // key without position embedding
    Qk_vec_k k_wo_pos;
    zero(q);
    zero(k);
    zero(q_bias);
    zero(k_bias);
    zero(k_wo_pos);
    float rotary_embedding_base = params.rotary_embedding_base;
    float rotary_embedding_scale = params.rotary_embedding_scale;
    // Need to recompute the inv freq if it is dynamic scaling.
    float const* rotary_embedding_inv_freq_cache = params.rotary_embedding_scale_type != RotaryScalingType::kDYNAMIC
        ? params.rotary_embedding_inv_freq_cache
        : nullptr;
    if (is_valid_qk_vec)
    {
        mmha::update_rotary_base_n_scale(rotary_embedding_base, rotary_embedding_scale,
            params.rotary_embedding_scale_type, params.rotary_embedding_dim, params.rotary_embedding_max_positions,
            tlength);
        // Query
        // The stride between tokens. We may be able to always use params.stride.
        uint32_t q_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads * Dh);
        // The offset.
        auto const q_offset = tensorrt_llm::common::flat_index_strided3(batch_beam_idx, hi, qk_vec_idx, q_stride, Dh);

        if (load_qkv_quant)
        {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
            using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
            auto const q_scaling = params.qkv_scale_quant_orig[0];
            auto const q_quant
                = *reinterpret_cast<Packed_Int8_t const*>(&reinterpret_cast<int8_t const*>(params.q)[q_offset]);
            convert_from_float(&q, mul<Packed_Float_t, float>(q_scaling, float_from_int8(q_quant)));
        }
        else
        {
            q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&params.q[q_offset]));
        }

        if constexpr (DO_CROSS_ATTENTION)
        {
            auto const k_idx = QK_VEC_SIZE * tidx;
            int const inBlockIdx = pastKCache.getKVLocalIdx(tlength, hi_kv, Dh, k_idx);
            Tcache* k_cache = reinterpret_cast<Tcache*>(pastKCache.getKBlockPtr(batch_beam_idx, tlength));

            if constexpr (ENABLE_8BITS_K_CACHE)
            {
                load_8bits_kv_cache_vec(&k, k_cache, inBlockIdx, k_scale_quant_orig_f);
            }
            else
            {
                k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&k_cache[inBlockIdx]));
            }
        }
        else
        {
            // Key
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t k_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            // The offset.
            auto const k_offset
                = tensorrt_llm::common::flat_index_strided3(batch_beam_idx, hi_kv, qk_vec_idx, k_stride, Dh);

            if (load_qkv_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
                auto const k_scaling = params.qkv_scale_quant_orig[1];
                auto const k_quant
                    = *reinterpret_cast<Packed_Int8_t const*>(&reinterpret_cast<int8_t const*>(params.k)[k_offset]);

                convert_from_float(&k, mul<Packed_Float_t, float>(k_scaling, float_from_int8(k_quant)));
            }
            else
            {
                k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&params.k[k_offset]));
            }
        }

        if (params.q_bias != nullptr)
        {
            auto const q_bias_offset = tensorrt_llm::common::flat_index2(hi, qk_vec_idx, Dh);
            q_bias
                = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&params.q_bias[q_bias_offset]));
        }
        if (HANDLE_KV && params.k_bias != nullptr)
        {
            auto const k_bias_offset = tensorrt_llm::common::flat_index2(hi_kv, qk_vec_idx, Dh);
            k_bias
                = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(&params.k_bias[k_bias_offset]));
        }
    }

    // Computes the Q/K values with bias.
    q = add(q, q_bias);
    if (HANDLE_KV)
    {
        k = add(k, k_bias);
    }

    // The width of the beam.
    auto const beam_width = static_cast<unsigned>(params.beam_width);
    // The batch idx.
    int const batch_idx = batch_beam_idx / beam_width;
    // Do we apply IA3?
    bool const do_ia3 = HANDLE_KV && params.ia3_tasks != nullptr;
    // Compute the IA3 task. One per batch index.
    auto const ia3_ti_hi = do_ia3
        ? tensorrt_llm::common::flat_index2(static_cast<unsigned>(params.ia3_tasks[batch_idx]), hi, num_heads)
        : 0;

    if (do_ia3 && is_valid_qk_vec)
    {
        k = mul<Qk_vec_k, Qk_vec_k, Qk_vec_k>(k,
            vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<Qk_vec_m const*>(
                &params.ia3_key_weights[tensorrt_llm::common::flat_index2(ia3_ti_hi, qk_vec_idx, Dh)])));
    }
    k_wo_pos = k;

    // Note we have no paddings in KV cache now.
    switch (params.position_embedding_type)
    {
    case PositionEmbeddingType::kLEARNED_ABSOLUTE:
    case PositionEmbeddingType::kRELATIVE:
    case PositionEmbeddingType::kALIBI:
    case PositionEmbeddingType::kALIBI_WITH_SCALE:
    {
        break;
    }
    case PositionEmbeddingType::kROPE_GPTJ:
    {
        if (HANDLE_KV)
        {
            apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, rotary_embedding_base,
                rotary_embedding_scale, tlength, rotary_embedding_inv_freq_cache);
        }
        else
        {
            apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, rotary_embedding_base, rotary_embedding_scale,
                tlength, rotary_embedding_inv_freq_cache);
        }
        break;
    }
    case PositionEmbeddingType::kLONG_ROPE:
    case PositionEmbeddingType::kROPE_M:
    case PositionEmbeddingType::kROPE_GPT_NEOX:
    case PositionEmbeddingType::kYARN:
    {
        bool const do_rotary = is_valid_qk_vec && QK_VEC_SIZE * tidx < params.rotary_embedding_dim;

        T* q_smem_ = reinterpret_cast<T*>(smem_);
        T* k_smem_ = q_smem_ + params.rotary_embedding_dim;

        int const half_rotary_dim = params.rotary_embedding_dim / 2;
        int const half_idx = qk_vec_idx / half_rotary_dim;
        int const intra_half_idx = qk_vec_idx % half_rotary_dim;
        int const smem_pitch = half_rotary_dim; // TODO: adjust for bank conflicts

        assert(half_rotary_dim % QK_VEC_SIZE == 0);
        int position_idx = tlength;
        if (params.position_embedding_type == PositionEmbeddingType::kROPE_M && params.mrope_position_deltas != nullptr)
        {
            position_idx += params.mrope_position_deltas[batch_idx];
        }

        if (do_rotary)
        {
            *reinterpret_cast<Qk_vec_k*>(q_smem_ + half_idx * smem_pitch + intra_half_idx) = q;
            if (HANDLE_KV)
            {
                *reinterpret_cast<Qk_vec_k*>(k_smem_ + half_idx * smem_pitch + intra_half_idx) = k;
            }
        }

        __syncthreads();

        int const transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
        constexpr int tidx_factor = (QK_VEC_SIZE > 1) ? QK_VEC_SIZE / 2 : 1;
        if (do_rotary)
        {
            float rotary_embedding_m_scale = tlength <= params.rotary_embedding_original_max_positions
                ? params.rotary_embedding_short_m_scale
                : params.rotary_embedding_long_m_scale;
            // The rotary cos_sin cache for the current timestep
            float2 const* cos_sin_cache = params.rotary_embedding_cos_sin_cache;
            if (cos_sin_cache)
            {
                cos_sin_cache += (static_cast<int64_t>(position_idx) * params.rotary_embedding_dim / 2);
            }

            mmha::vec_from_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
            if (HANDLE_KV)
            {
                mmha::vec_from_smem_transpose(k, k_smem_, transpose_idx, smem_pitch);

                mmha::apply_rotary_embedding(q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                    rotary_embedding_base, rotary_embedding_scale, position_idx, rotary_embedding_inv_freq_cache,
                    cos_sin_cache, rotary_embedding_m_scale, params.rotary_cogvlm_vision_start,
                    params.rotary_cogvlm_vision_length);

                mmha::write_smem_transpose(k, k_smem_, transpose_idx, smem_pitch);
            }
            else
            {
                mmha::apply_rotary_embedding(q, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                    rotary_embedding_base, rotary_embedding_scale, position_idx, rotary_embedding_inv_freq_cache,
                    cos_sin_cache, rotary_embedding_m_scale, params.rotary_cogvlm_vision_start,
                    params.rotary_cogvlm_vision_length);
            }
            mmha::write_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
        }

        __syncthreads();

        if (do_rotary)
        {
            q = *reinterpret_cast<Qk_vec_k*>(q_smem_ + half_idx * smem_pitch + intra_half_idx);
            if (HANDLE_KV)
            {
                k = *reinterpret_cast<Qk_vec_k*>(k_smem_ + half_idx * smem_pitch + intra_half_idx);
            }
        }

        __syncthreads();
        break;
    }
    }

    // For the same reason as HANDLE_KV, no compute needed in Cross-Attention's 1st step
    // Store Q K vectors to shared memory, and calculate QK.
    if (qk_vec_idx < Dh_MAX)
    {

        // Store the Q values to shared memory.
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
        if constexpr (FP8_K_CACHE)
        {
            // There are many more elements from K than elements from Q so we pre-scale Q instead
            // of scaling all the elements from K. It helps reduce the number of ops.
            Qk_vec_k scaled_q;
            zero(scaled_q);
            if (is_valid_qk_vec)
            {
                scaled_q = mul<Qk_vec_k, Tk, Qk_vec_k>(k_scale_quant_orig, q);
            }
            reinterpret_cast<Qk_vec_k*>(&q_smem[qk_vec_idx])[0] = scaled_q;
        }
        else
#endif
        {
            // Set padded Dh to 0 for the correctness of QK (when Dh != Dh_Max).
            Qk_vec_k zero_q;
            zero(zero_q);
            reinterpret_cast<Qk_vec_k*>(&q_smem[qk_vec_idx])[0] = is_valid_qk_vec ? q : zero_q;
        }

        // Store the K values to shared memory.
        // We store K values from shared memory to global memory
        //  when the target position of K cache in global memory has been accessed (in the case of cyclic kv cache)
        if (POS_SHIFT && !DO_CROSS_ATTENTION)
        {
            reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx])[0] = k_wo_pos;
        }
        else
        {
            reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx])[0] = k;
        }

        // Compute \sum_i Q[i] * K^T[i] for the current timestep.
        qk = dot<Qk_vec_accum, Qk_vec_k>(q, k);
        if (QK_VECS_PER_Dh_MAX <= WARP_SIZE)
        {
#pragma unroll
            for (int mask = QK_VECS_PER_Dh_MAX / 2; mask >= 1; mask /= 2)
            {
                qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_Dh_MAX), qk, mask);
            }
        }
    }

    if (QK_VECS_PER_Dh_MAX > WARP_SIZE)
    {
        constexpr int WARPS_PER_RED = (QK_VECS_PER_Dh_MAX + WARP_SIZE - 1) / WARP_SIZE;
        qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
    }

    // Pre-compute the pointer for the relative attention bias.
    T const* relative_attention_bias_ptr = nullptr;
    [[maybe_unused]] T const* relative_attention_bias_ptr_fixed = nullptr; // record the base for offset
    if (has_relative_attention_bias)
    {
        // "hi" is unsigned, subtracting int from unsigned int causes underflow. Cast to int
        int64_t offset = IMPLICIT_REL_ATTN_BIAS
            ? ((int64_t) hi * relative_attention_bias_stride - tlength)
            : ((int64_t) hi * relative_attention_bias_stride + tlength) * relative_attention_bias_stride;
        relative_attention_bias_ptr = &params.relative_attention_bias[offset];
        relative_attention_bias_ptr_fixed = &params.relative_attention_bias[offset];
    }

    // Pre-compute the pointer for the attention mask.
    bool const* attention_mask_ptr = nullptr;
    // Do we have attention mask ?
    bool has_attention_mask = params.attention_mask != nullptr;
    if (has_attention_mask)
    {
        attention_mask_ptr = params.attention_mask + batch_idx * params.attention_mask_stride;
    }

    // Load the value.
    float relative_attention_bias = 0.f;
    if (has_relative_attention_bias && tidx == 0)
    {
        relative_attention_bias = convert_to_float(relative_attention_bias_ptr[tlength]);
    }
    if (has_attention_mask && tidx == 0)
    {
        // Note: reuse the relative_attention_bias variable.
        // attention_mask = 1.0 means that the position is not masked.
        relative_attention_bias += (FLT_MAX * (float(attention_mask_ptr[tlength]) - 1.0f));
    }

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if (tidx == 0)
    {
        // Normalize qk.
        qk = qk * params.inv_sqrt_dh + relative_attention_bias;

        // Apply attention logit softcapping scale.
        if constexpr (ATTN_LOGIT_SOFTCAPPING)
        {
            qk = params.attn_logit_softcapping_scale * __tanhf(qk * params.attn_logit_softcapping_inverse_scale);
        }

        // We don't need to apply the linear position bias here since qi - ki = 0 yields the position bias 0.
        qk_max = qk;

        // Store Q*K^T to shared memory.
        if (MULTI_BLOCK_FLAG)
        {
            qk_current_smem[0] = qk;
        }
        else
        {
            // We need to store the qk result to the end of the qk_smem for cyclic kv cache (+ 1 for smem memory
            // allocation) because the previous cache will still write to the new_cache_pos of qk_smem.
            qk_smem[kv_loop_length] = qk;
        }
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    constexpr unsigned K_ELTS_PER_CHUNK{THREADS_PER_KEY * K_VEC_SIZE};

    // The positions of the cache buffer (for this B * H) and the vector within that chunk associated with this
    // thread.
    auto const k_idx = chunk_index<T, K_vec_k, THREADS_PER_KEY>(tidx);

    // The number of vectors per thread.
    constexpr unsigned K_VECS_PER_THREAD{Dh_MAX / K_ELTS_PER_CHUNK};
    static_assert(Dh_MAX == K_ELTS_PER_CHUNK * K_VECS_PER_THREAD);

    float logn_scale = 1.f;
    if (has_logn_scaling)
    {
        logn_scale = params.logn_scaling_ptr[tlength];
    }

    // Load the Q values from shared memory. The values are reused during the loop on K.
    K_vec_accum q_vec[K_VECS_PER_THREAD];
#pragma unroll
    for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii)
    {
        q_vec[ii] = vec_conversion<K_vec_accum, K_vec_k>(*reinterpret_cast<K_vec_k const*>(
            &q_smem[tensorrt_llm::common::flat_index2(ii, k_idx.y, K_ELTS_PER_CHUNK)]));
        if (has_logn_scaling)
        {
            q_vec[ii] = mmha::mul<K_vec_accum, float, K_vec_accum>(logn_scale, q_vec[ii]);
        }
    }

    // The number of timesteps loaded per iteration, i.e., (THREADS_PER_BLOCK * THREADS_PER_BLOCK) / 256 <= 256
    constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
    // The number of keys per warp.
    constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
    // The number of unrolled keys per warp.
    constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
    // The number of unrolled keys per ieration.
    constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

    auto const timesteps_per_block = static_cast<unsigned>(params.timesteps_per_block);

    // Clarifications:
    // - in self attn, input_length is input text length, tlength is current timestep
    // - in cross attn, input_length is *decoder* input length (usually 1), tlength is *encoder* input context length

    // Take all previous cache as context in order to batch as many LDGs as possible.
    auto const k_loop_end = MULTI_BLOCK_FLAG
        ? divUp(timesteps_per_block, UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP
        : divUp(static_cast<unsigned>(kv_loop_length), UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;

    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    // Note max_attention_window_size is maximum of cyclic_attention_window_size among all layers.
    // By default, you can assume that they are the same.
    auto const bi_seq_len_offset = static_cast<std::size_t>(batch_beam_idx) * params.max_attention_window_size;
    // Beam indices are based on the max_attention_window_size while each layer may have different
    // cyclic_attention_window_size So we need to rebuild the beam_indices if max_attention_window_size is not equal to
    // cyclic_attention_window_size.
    int const* beam_indices = HAS_BEAMS ? &params.cache_indir[bi_seq_len_offset] : nullptr;

    auto const c_tile_times_timesteps_per_block = c_tile * timesteps_per_block; // 0 if !MULTI_BLOCK_FLAG

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Key cache loops for dot(Q, K).

    // Is it the leader?
    bool const is_leader = Qk_dot<T, THREADS_PER_KEY>::is_leader(tidx);

    // The slope for ALiBi.
    float linear_bias_slope = 0.f;
    if (params.linear_bias_slopes != nullptr)
    {
        // TODO: Use a cleaner code to convert from T to float.
        linear_bias_slope = mul<float>(params.linear_bias_slopes[hi], 1.f);
    }

    // Handle only context key cache with beam searching.
    // Handle both context and generation key cache without beam searching.
    // Explicit batching of LDGs (by K_LOOP_UNROLL) as it doesn't depend on indirection tables.
    for (int ti = k_idx.x; ti < k_loop_end; ti += UNROLLED_K_PER_ITER)
    {
        int const time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        // The keys loaded from the key cache.
        K_vec_m k_vec_cache[K_LOOP_UNROLL][K_VECS_PER_THREAD];

#pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
        {
#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
            {
                // Make sure we read data within the bound.
                // Dh OOB values will be handled by zero_q.
                // Seq OOB values will be masked out when storing back to smem.
                auto const jj = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);
                int valid_time_now = min(time_now + kv_token_start_offset + k_loop * K_PER_ITER, kv_token_idx_bound);
                // The beam offset is always 0 either when beam_width = 1
                // or the time_idx < kv_loop_length (all beams share the same context kv cache).
                int beam_offset
                    = (HAS_BEAMS && valid_time_now >= beam0_context_length) ? beam_indices[valid_time_now] : 0;
                if (POS_SHIFT && valid_time_now >= sink_token_len)
                {
                    // If one more block mode is enabled, we use the index in sequence as tokenIdx.
                    // Otherwise, we need to add the bubble length to the index
                    valid_time_now += shift_for_cyclic_k;
                    if (enable_use_seq_idx_kv)
                    {
                        // Convert the token index in sequence to token index in K cache.
                        valid_time_now = pastKCache.getKVTokenIdx(valid_time_now);
                    }
                }
                int const seqIdx = batch_idx * beam_width + beam_offset;

                // Base pointer to k cache block for beam's batch
                TKcache* k_cache_batch = reinterpret_cast<TKcache*>(pastKCache.getKBlockPtr(seqIdx, valid_time_now));

                int inBlockIdx = pastKCache.getKVLocalIdx(valid_time_now, hi_kv, Dh, jj);
                k_vec_cache[k_loop][k_vec_i] = *reinterpret_cast<K_vec_m const*>(&k_cache_batch[inBlockIdx]);
            }
        }

#pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
        {
            int const local_time_now = time_now + k_loop * K_PER_ITER;
            int const local_ti = ti + k_loop * K_PER_ITER;

            // Perform the dot product and normalize qk.
            //
            // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
            K_vec_m k_vec[K_VECS_PER_THREAD];
#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
            {
                k_vec[k_vec_i] = *reinterpret_cast<K_vec_m*>(&k_vec_cache[k_loop][k_vec_i]);
            }

            // Is it active?
            bool const is_active = local_time_now < kv_loop_length;

            if constexpr (IMPLICIT_REL_ATTN_BIAS)
            {
                // Compute bias value on the fly (See bert_preprocess_kernels.cu::buildRelativeAttentionBias)
                int relative_buckets = 0;
                int relative_position = local_time_now - tlength;
                int num_buckets = relative_attention_bias_stride;
                // Special logic in T5 relative attention, both encoder & decoder use this, because
                // relative_attention_bias is pre-computed once and passed around.
                // T5 decoder attention now only uses bidirectional=False relative position logic
                // (ref: tensorrt_llm/layers/attention.py compute_relative_bias())
                relative_position = relative_position >= 0 ? 0 : -relative_position;

                int max_exact = num_buckets / 2;
                bool is_small = relative_position < max_exact;
                int relative_position_if_large = max_exact
                    + (int) (logf(relative_position * 1.0f / max_exact) / logf((float) max_distance / max_exact)
                        * (num_buckets - max_exact));
                relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                relative_buckets += is_small ? relative_position : relative_position_if_large;
                relative_attention_bias_ptr
                    = relative_attention_bias_ptr_fixed + (tlength - local_time_now) + relative_buckets;
            }

            // Prefetch the relative attention bias.
            float relative_attention_bias = 0.f;
            if (is_active && has_relative_attention_bias)
            {
                relative_attention_bias = convert_to_float(relative_attention_bias_ptr[local_time_now]);
            }
            if (is_active && has_attention_mask)
            {
                // Note: reuse the relative_attention_bias variable.
                // attention_mask = 1.0 means that the position is not masked.
                relative_attention_bias += (FLT_MAX * (float(attention_mask_ptr[tlength]) - 1.0f));
            }

            // Compute the dot product between Q and K.
            // Note that dot will convert 8bit vec to the accumulation data type (float by default).
            float qk_ = 0.f;
#ifdef MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            if constexpr (FP8_K_CACHE)
            {
                qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
            }
            else
#endif // MMHA_FP8_SCALE_Q_INSTEAD_OF_K
            {
                if constexpr (ENABLE_8BITS_K_CACHE)
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::scale_dot(q_vec, k_vec, k_scale_quant_orig_f)
                        * params.inv_sqrt_dh;
                }
                else
                {
                    qk_ = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh;
                }
            }

            // Apply attention logit softcapping scale.
            if constexpr (ATTN_LOGIT_SOFTCAPPING)
            {
                qk_ = params.attn_logit_softcapping_scale * __tanhf(qk_ * params.attn_logit_softcapping_inverse_scale);
            }

            // For multi-block mode, we need to make sure it will not be OOB.
            if (MULTI_BLOCK_FLAG && local_ti >= timesteps_per_block)
            {
                continue;
            }

            // Add the ALiBi bias. (ki - qi) * slope[hi].
            //
            // The padding tokens are located between the input context and the generated tokens.
            // We need to remove the correct number of padding tokens in the distance computation.
            //
            //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
            //   token: i i i i p p p o o o where i=input, p=pad, o=output.
            // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
            //
            // All the threads do the work even if it's not relevant to avoid divergence.
            qk_ += linear_bias_slope * (local_time_now - tlength) + relative_attention_bias;

            if constexpr (BLOCK_SPARSE_ATTN)
            {
                float mask_val
                    = params.block_sparse_params.computeMask(tlength, local_time_now, tlength + 1, num_heads, hi) ? 1.f
                                                                                                                  : 0.f;
                qk_ += (1.0f - mask_val) * -10000.0f;
            }

            // There's one qk value per timestep.
            // Make sure only leader threads stores qk value within the bound.
            if (is_active && is_leader)
            {
                // Calculate the max for softmax.
                qk_max = fmaxf(qk_max, qk_);
                // Store the product to shared memory.
                qk_smem[local_ti] = qk_;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Softmax.

    // Perform the final reduction to compute the max inside each warp.
    //
    // NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
    // group so it's not needed to run the reduction inside the group (again).

#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA)
    // Leader threads will be in the dignonal when using HMMA.
    if (THREADS_PER_KEY <= 4)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 4));
    }
    if (THREADS_PER_KEY <= 8)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 9));
    }
    if (THREADS_PER_KEY <= 16)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(unsigned(-1), qk_max, 18));
    }
#else
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }
#endif // defined MMHA_USE_HMMA

    // Decompose the thread index into warp and lane.
    auto const warp = tidx / WARP_SIZE;
    auto const lane = tidx % WARP_SIZE;

    // The warp leader writes the max to shared memory.
    if (lane == 0)
    {
        red_smem[warp] = qk_max;
    }

    // Make sure the products are in shared memory.
    __syncthreads();

    // After the syncthreads, the target k position (cyclic kv cache) should also have been used by the k loop.
    // Write the K values to the global memory cache.
    //
    // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
    // system. We designed it this way as it allows much better memory loads (and there are many
    // more loads) + the stores are really "write and forget" since we won't need the ack before
    // the end of the kernel. There's plenty of time for the transactions to complete.

    // For MQA/GQA mode, write only with the first Q head of each group per KV head.

    // Get the c_tile_id that handles the current timestep.
    int current_step_ctile_idx = kv_loop_length / timesteps_per_block;
    if (HANDLE_KV && hi == (hi_kv * qhead_per_kv) && qk_vec_idx < Dh
        && (!MULTI_BLOCK_FLAG || c_tile == current_step_ctile_idx))
    {
        // Trigger the stores to global memory.
        Qk_vec_k k_vec = *reinterpret_cast<Qk_vec_k*>(&k_smem[qk_vec_idx]);
        auto const k_idx = QK_VEC_SIZE * tidx;
        int const inBlockIdx = kvCacheBuffer.getKVLocalIdx(tlength, hi_kv, Dh, k_idx);
        // The base pointer for the value in the cache buffer.
        Tcache* k_cache = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(batch_beam_idx, tlength));

        if constexpr (ENABLE_8BITS_KV_CACHE)
        {
            store_8bits_vec(reinterpret_cast<Tcache*>(k_cache), k_vec, inBlockIdx, kv_scale_orig_quant);
        }
        else
        {
            *reinterpret_cast<Qk_vec_m*>(&k_cache[inBlockIdx]) = vec_conversion<Qk_vec_m, Qk_vec_k>(k_vec);
        }
    }

    // The warps finalize the reduction.
    qk_max = lane < WARPS_PER_BLOCK ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    // Broadcast to all the threads in the warp.
    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

    // Compute the logits and start the sum.
    float sum = 0.f;

    // Each thread will handle one float (either qk_smem/logit).
    int const logit_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= logit_loop_end; ti += THREADS_PER_BLOCK)
    {

        int const time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        // For single-block mode, we don't need the mask since it has been skipped.
        if (!MULTI_BLOCK_FLAG)
        {
            float logit = __expf(qk_smem[time_now] - qk_max);
            sum += logit;
            qk_smem[time_now] = logit;
        }
        else
        {
            // Not supported yet: multi-block mode with FP8_MHA
            if (time_now < kv_loop_length && ti != timesteps_per_block)
            {
                float logit = __expf(qk_smem[ti] - qk_max);
                sum += logit;
                qk_smem[ti] = logit;
            }
            else if (time_now == kv_loop_length)
            {
                float logit = __expf(qk_current_smem[0] - qk_max);
                sum += logit;
                qk_current_smem[0] = logit;
            }
        }
    }

    // Compute the sum.
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

    // Add the attention sinks.
    // It has been moved to the end of the kernel if the multi-block mode is enabled.
    if (!MULTI_BLOCK_FLAG && params.attention_sinks != nullptr)
    {
        sum += expf(params.attention_sinks[hi] - qk_max);
    }

// Normalize the logits.
#ifdef MMHA_FP8_SCALE_P_INSTEAD_OF_V
    float logit_scale = (FP8_KV_CACHE ? kv_scale_quant_orig_f : 1.0f);
#else
    float logit_scale = 1.f;
#endif // MMHA_FP8_SCALE_P_INSTEAD_OF_V
    float inv_sum = __fdividef(logit_scale, sum + 1.e-6f);

    int const normlization_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
    for (int ti = tidx; ti <= normlization_loop_end; ti += THREADS_PER_BLOCK)
    {
        int const time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        if (!MULTI_BLOCK_FLAG)
        {
            convert_from_float(&logits_smem[ti], qk_smem[ti] * inv_sum);
        }
        else
        {
            // no scaling factor inv_sum applied here, will apply the scaling factor after all blocks finished
            if (time_now < kv_loop_length && ti != timesteps_per_block)
            {
                convert_from_float(&logits_smem[ti], qk_smem[ti]);
            }
            else if (time_now == kv_loop_length)
            {
                convert_from_float(&logits_current_smem[0], qk_current_smem[0]);
            }
        }
    }

    // Put Values part below so we leverage __syncthreads
    // from the previous step

    auto const v_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
    // The value computed by this thread.
    auto const vo = v_idx.x;
    // The hidden dimensions computed by this particular thread.
    auto const vi = v_idx.y;

    // The number of values processed per iteration of the loop.
    constexpr unsigned V_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_VALUE};
    // The number of unrolled keys per ieration.
    constexpr unsigned UNROLLED_V_PER_ITER = V_PER_ITER * V_LOOP_UNROLL;

    bool const is_valid_vi = IS_Dh_MAX || vi < Dh;

    // One group of threads computes the product(s) for the current timestep.
    V_vec_k v_bias;
    zero(v_bias);
    // if( vo == params.timestep % V_PER_ITER ) {
    if (is_valid_vi && HANDLE_KV && vo == kv_loop_length % V_PER_ITER)
    {
        // Trigger the loads from the V bias buffer.
        if (params.v_bias != nullptr)
        {
            auto const v_bias_offset = tensorrt_llm::common::flat_index2(hi_kv, vi, Dh);
            v_bias = *reinterpret_cast<V_vec_k const*>(&params.v_bias[v_bias_offset]);
        }

        if (DO_CROSS_ATTENTION)
        {
            *reinterpret_cast<V_vec_k*>(&bias_smem[vi]) = v_bias;
        }
    }

    // From previous, before values, step
    // Also make sure the logits are in shared memory.
    __syncthreads();

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Value cache loops.

#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
    using V_vec_accum = typename V_vec_accum_fp32_<V_vec_k>::Type;
#else
    using V_vec_accum = V_vec_k;
#endif
    // The partial outputs computed by each thread.
    V_vec_accum out;
    zero(out);

    // Loop over the timesteps to compute the partial outputs.
    if (is_valid_vi)
    {
        // Explicit batching of LDGs (by V_LOOP_UNROLL) as it doesn't depend on indirection tables.
        // Take all previous kv cache as context in order to batch as many LDGs as possible
        int v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : kv_loop_length;
        for (int ti = vo; ti < v_loop_end; ti += UNROLLED_V_PER_ITER)
        {
            V_vec_m v_vec_cache[V_LOOP_UNROLL];
#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
            {
                // Fetch offset based on cache_indir when beam sampling
                int time_idx = ti + v_loop * V_PER_ITER + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                time_idx += kv_token_start_offset;
                time_idx = min(time_idx, kv_token_idx_bound);
                // The beam offset is always 0 either when beam_width = 1
                // or the time_idx < kv_loop_length (all beams share the same context kv cache).
                int beam_offset = (HAS_BEAMS && time_idx >= beam0_context_length) ? beam_indices[time_idx] : 0;
                if (POS_SHIFT && time_idx >= sink_token_len)
                {
                    // If one more block mode is enabled, we use the index in sequence as tokenIdx.
                    // Otherwise, we need to add the bubble length to the index
                    time_idx += shift_for_cyclic_kv;
                    if (enable_use_seq_idx_kv)
                    {
                        // Convert the token index in sequence to token index in V cache.
                        time_idx = kvCacheBuffer.getKVTokenIdx(time_idx);
                    }
                }
                int rowIdx = batch_idx * beam_width + beam_offset;

                int const inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                // The base pointer for the value in the cache buffer.
                Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));

                v_vec_cache[v_loop] = *reinterpret_cast<V_vec_m const*>(&v_cache_batch[inBlockIdx]);
            }

#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
            {
                V_vec_m v_vec = reinterpret_cast<V_vec_m*>(&v_vec_cache[v_loop])[0];

                int local_time_idx = ti + v_loop * V_PER_ITER;
                int time_idx = local_time_idx + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);

                bool const is_mask
                    = (MULTI_BLOCK_FLAG && local_time_idx >= timesteps_per_block) || (time_idx >= kv_loop_length);

                // Load the logits from shared memory.
                // Note that fma will convert 8bit vec to the accumulation data type (float by default).
                Logit_value_fma<Tk, V_vec_accum, V_vec_m, INT8_KV_CACHE, FP8_KV_CACHE>(
                    out, reinterpret_cast<Tk*>(logits_smem + local_time_idx), v_vec, kv_scale_quant_orig_f, is_mask);
            }
        }
    }

    // Make sure we can overwrite the v cache if using cyclic kv cache.
    __syncthreads();

    // One group of threads computes the product(s) for the current timestep.
    if (vo == kv_loop_length % V_PER_ITER && is_valid_vi && (!MULTI_BLOCK_FLAG || (c_tile == current_step_ctile_idx)))
    {
        int const inBlockIdx = kvCacheBuffer.getKVLocalIdx(tlength, hi_kv, Dh, vi);
        // The base pointer for the value in the cache buffer.
        Tcache* v_cache_base = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(batch_beam_idx, tlength));

        V_vec_k v;
        if (DO_CROSS_ATTENTION)
        {
            if constexpr (ENABLE_8BITS_KV_CACHE)
            {
                load_8bits_kv_cache_vec(&v, v_cache_base, inBlockIdx, kv_scale_quant_orig_f);
            }
            else
            {
                v = vec_conversion<V_vec_k, V_vec_k>(*reinterpret_cast<V_vec_k const*>(&v_cache_base[inBlockIdx]));
            }
        }
        else
        {
            // Trigger the loads from the V buffer.
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t v_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            // The offset.
            auto const v_offset = tensorrt_llm::common::flat_index_strided3(batch_beam_idx, hi_kv, vi, v_stride, Dh);

            if (load_qkv_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_k>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<V_vec_k>::value>::type;
                auto const v_scaling = params.qkv_scale_quant_orig[2];
                auto const v_quant
                    = *reinterpret_cast<Packed_Int8_t const*>(&reinterpret_cast<int8_t const*>(params.v)[v_offset]);

                convert_from_float(&v, mul<Packed_Float_t, float>(v_scaling, float_from_int8(v_quant)));
            }
            else
            {
                v = *reinterpret_cast<V_vec_k const*>(&params.v[v_offset]);
            }
        }

        if (HANDLE_KV)
        {
            // Compute the V values with bias.
            v = add(v, v_bias);

            if (do_ia3)
            {
                v = mul<V_vec_k, V_vec_k, V_vec_k>(v,
                    *reinterpret_cast<V_vec_k const*>(
                        &params.ia3_value_weights[tensorrt_llm::common::flat_index2(ia3_ti_hi, vi, Dh)]));
            }
        }

        // Store the values with bias back to global memory in the cache for V.
        //*reinterpret_cast<V_vec_k*>(&v_cache[params.timestep*Dh]) = v;
        // For MQA/GQA mode, write only with the first Q head of each group per KV head.
        if ((hi == (hi_kv * qhead_per_kv)) && !DO_CROSS_ATTENTION)
        {
            if (ENABLE_8BITS_KV_CACHE)
            {
                store_8bits_vec(v_cache_base, v, inBlockIdx, kv_scale_orig_quant);
            }
            else
            {
                *reinterpret_cast<V_vec_k*>(&v_cache_base[inBlockIdx]) = v;
            }
        }

        // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACCUM_FOR_LOGITS)
        // out = fma(logits_smem[params.timestep], cast_to_float(v), out);
        if (!MULTI_BLOCK_FLAG)
        {
            out = fma(logits_smem[kv_loop_length], cast_to_float(v), out);
        }
        else
        {
            out = fma(logits_current_smem[0], cast_to_float(v), out);
        }
#else  // MMHA_USE_FP32_ACCUM_FOR_LOGITS
       // out = fma(logits_smem[params.timestep], v, out);
        if (!MULTI_BLOCK_FLAG)
        {
            out = fma(logits_smem[kv_loop_length], v, out);
        }
        else
        { // MULTI_BLOCK_FLAG // Not supported yet: multi-block mode with FP8_MHA
            out = fma(logits_current_smem[0], v, out);
        }
#endif // MMHA_USE_FP32_ACCUM_FOR_LOGITS
    }
    // Make sure we can start writing to shared memory.
    __syncthreads();

    // Run the final reduction amongst the different groups computing different partial outputs.
#pragma unroll
    for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2)
    {

        // The midpoint in the number of active groups.
        int midpoint = active_groups / 2;

        // The upper part of active threads store to shared memory.
        if (vo >= midpoint && vo < active_groups && (Dh == Dh_MAX || vi < Dh))
        {
#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
            convert_from_float(reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
            *reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
        }
        __syncthreads();

        // The bottom warps update their values.
        if (vo < midpoint && (Dh == Dh_MAX || vi < Dh))
        {
            out = add(*reinterpret_cast<V_vec_k const*>(&out_smem[vo * Dh + vi]), out);
        }
        __syncthreads();
    }

    // Quantized output only supports fp8 currently, which should be used together with FP8 Context FMHA.
    using Quantized_t = __nv_fp8_e4m3;
    using Quantized_vec = typename packed_type<__nv_fp8_e4m3, num_elems<V_vec_accum>::value>::type;
    auto const bhi = tensorrt_llm::common::flat_index2(batch_beam_idx, hi, num_heads);
    auto const bhi_seq_len_tile = bhi * params.seq_len_tile;
    // Output the final values.
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh))
    {
        auto const bhvi = tensorrt_llm::common::flat_index2(bhi, vi, Dh);
#ifdef MMHA_USE_FP32_ACCUM_FOR_OUT
        if (!MULTI_BLOCK_FLAG)
        {
            if (write_attention_quant)
            {
                out = mul<V_vec_accum, float>(*params.attention_out_scale_orig_quant, out);
                Quantized_vec final_out;
                convert_to_fp8(&final_out, out);
                *reinterpret_cast<Quantized_vec*>(reinterpret_cast<Quantized_t*>(params.out) + bhvi) = final_out;
            }
            else
            {
                // This makes sure we have coalesced memory access.
                V_vec_k final_out;
                convert_from_float(&final_out, out);
                *reinterpret_cast<V_vec_k*>(static_cast<T*>(params.out) + bhvi) = final_out;
            }
        }
        else
        {
            // for write partial output to partial_out
            int partial_out_offset = c_tile * params.batch_size * num_heads * params.hidden_size_per_head;
            // for write partial statistics to partial_max and partial_sum
            int partial_stats_offset = bhi_seq_len_tile + c_tile;

            // This makes sure we have coalesced memory access.
            V_vec_k partial_out;
            convert_from_float(&partial_out, out);
            *reinterpret_cast<V_vec_k*>(&params.partial_out[partial_out_offset + bhvi]) = partial_out;
            convert_from_float(reinterpret_cast<float*>(&params.partial_max[partial_stats_offset]), qk_max);
            convert_from_float(reinterpret_cast<float*>(&params.partial_sum[partial_stats_offset]), sum);
        }
#else  // MMHA_USE_FP32_ACCUM_FOR_OUT
        *reinterpret_cast<V_vec_accum*>(static_cast<T*>(params.out) + bhvi) = out;
#endif // MMHA_USE_FP32_ACCUM_FOR_OUT
    }

#ifdef ENABLE_MULTI_BLOCK_OPTION
    if (MULTI_BLOCK_FLAG)
    {

        cuda::atomic_ref<int, cuda::thread_scope_device> count_ref{params.block_counter[bhi]};
        bool last_block{false};
        if (tidx == 0)
        {
            if (count_ref.fetch_add(1, cuda::memory_order_acq_rel) == (gridDim.z - 1))
            {
                last_block = true;
            }
        }

        ////////////////////
        ////////////////////
        // Make sure every threadblock finishes the previous computation, and enter the last threadblock in the
        // following (for each B and H) Do the final computation in the last threadblock Final reduction computation
        // by combining all the partial max/sum and outputs
        ////////////////////
        ////////////////////
        if (__syncthreads_or(last_block))
        {

            ////////////////////
            // Find the global max from all partial max -> use CUB BlockReduce
            ////////////////////

            float final_max = -FLT_MAX;
            float thread_partial_max = -FLT_MAX;
            thread_partial_max = params.partial_max[bhi_seq_len_tile + min(tidx, gridDim.z - 1)];

            // Make sure we can start writing to shared memory.
            __syncthreads();

            // Specialize BlockReduce for a 1D block of THREADS_PER_BLOCK threads of type int
            typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
            // Allocate shared memory for BlockReduce
            __shared__ typename BlockReduce::TempStorage temp_storage;
            // Obtain a segment of consecutive items that are blocked across threads (final_max from above)
            // Compute the block-wide max for thread0
            final_max = BlockReduce(temp_storage).Reduce(thread_partial_max, cuda::maximum(), gridDim.z);

            __shared__ float final_max_smem;
            if (tidx == 0)
            {
                final_max_smem = final_max;
            }
            __syncthreads();

            // Finish the final_max computation
            final_max = final_max_smem;

            ////////////////////
            // Reduction for global sum over all partial sum (scaled by the exponential term from global max) -> use
            // gridDim.z threads
            ////////////////////

            float final_sum = 0.f;
            if (tidx < gridDim.z)
            {
                thread_partial_max = params.partial_max[bhi_seq_len_tile + tidx];
                auto const thread_partial_sum = params.partial_sum[bhi_seq_len_tile + tidx];
                final_sum += __expf(thread_partial_max - final_max) * thread_partial_sum;
            }

            // Compute the final_sum.
            final_sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], final_sum);

            ////////////////////
            // Reduction for final output (scaled by the exponential term from global max) -> use THREADS_PER_VALUE
            // * gridDim.z threads
            ////////////////////

            // Shared memory to store partial outputs for each oi. -> size: gridDim.z * Dh * 4 Bytes. Reuse qk_smem.
            T* out_oi_smem = reinterpret_cast<T*>(smem_);

            auto const o_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);

            // Init partial out for accumulation.
            V_vec_k zero_k;
            zero(zero_k);
            V_vec_k thread_accumulated_out = zero_k;

            // The hidden dimensions computed by this particular thread. (refer to vi)
            auto const oi = o_idx.y;

            // The partial output region this thread takes care of
            auto const oo = o_idx.x;

            // Each thread may handle more than one partial output.
            for (int tile_idx = o_idx.x; tile_idx < gridDim.z; tile_idx += V_PER_ITER)
            {
                // Load partial output
                int thread_partial_out_offset = tile_idx * params.batch_size * num_heads * params.hidden_size_per_head;
                // Load partial max (different to thread_partial_max since the threadIdx rule changes here)
                float thread_partial_max_for_out = params.partial_max[bhi_seq_len_tile + tile_idx];
                // Load the partial outputs.
                V_vec_k thread_partial_out
                    = *reinterpret_cast<V_vec_k const*>(&params.partial_out[thread_partial_out_offset + bhi * Dh + oi]);
                // Apply the correction factor.
                Tk factor_compute;
                convert_from_float(&factor_compute, __expf(thread_partial_max_for_out - final_max));
                thread_partial_out = mul<V_vec_k, Tk, V_vec_k>(factor_compute, thread_partial_out);
                thread_accumulated_out = add(thread_partial_out, thread_accumulated_out);
            }

            // Run the final reduction amongst the different groups computing different partial outputs.
#pragma unroll
            for (int active_groups = V_PER_ITER; active_groups >= 2; active_groups /= 2)
            {

                // The midpoint in the number of active groups.
                int midpoint = active_groups / 2;

                // The upper part of active threads store to shared memory.
                if (oo >= midpoint && oo < active_groups && (Dh == Dh_MAX || oi < Dh))
                {
                    *reinterpret_cast<V_vec_k*>(&out_oi_smem[(oo - midpoint) * Dh + oi]) = thread_accumulated_out;
                }
                __syncthreads();

                // The bottom warps update their values.
                if (oo < midpoint && (Dh == Dh_MAX || oi < Dh))
                {
                    thread_accumulated_out
                        = add(thread_accumulated_out, *reinterpret_cast<V_vec_k const*>(&out_oi_smem[oo * Dh + oi]));
                }
                __syncthreads();
            }

            ////////////////////
            // Final output O * inv_sum
            ////////////////////

            if (oo == 0 && (Dh == Dh_MAX || oi < Dh))
            {
                // Add the attention sinks.
                if (params.attention_sinks != nullptr)
                {
                    final_sum += expf(params.attention_sinks[hi] - final_max);
                }

                auto const inv_sum = __fdividef(
                    write_attention_quant ? *params.attention_out_scale_orig_quant : 1.f, final_sum + 1.e-6f);

                Tk inv_sum_compute;
                convert_from_float(&inv_sum_compute, inv_sum);

                thread_accumulated_out = mul<V_vec_k, Tk, V_vec_k>(inv_sum_compute, thread_accumulated_out);

                if (write_attention_quant)
                {
                    Quantized_vec final_out;
                    convert_to_fp8(&final_out, thread_accumulated_out);
                    *reinterpret_cast<Quantized_vec*>(reinterpret_cast<Quantized_t*>(params.out) + bhi * Dh + oi)
                        = final_out;
                }
                else
                {
                    *reinterpret_cast<V_vec_k*>(static_cast<T*>(params.out) + (bhi * Dh + oi)) = thread_accumulated_out;
                }
            }

            // Reset qk_current_smem and block_counter for the next timestep
            if (tidx == 0)
            {
                params.block_counter[bhi] = 0;
            }
        }
    }
#endif // ENABLE_MULTI_BLOCK_OPTION

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

} // namespace mmha

} // namespace kernels
} // namespace tensorrt_llm
