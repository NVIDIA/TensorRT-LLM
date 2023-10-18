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

// #define MMHA_USE_HMMA_FOR_REDUCTION

// Below are knobs to extend FP32 accumulation for higher FP16 accuracy

// Does not seem to affect the accuracy that much
#define MMHA_USE_FP32_ACUM_FOR_FMA

// Seems to slightly improve the accuracy
#define MMHA_USE_FP32_ACUM_FOR_OUT

#if 0 && defined(MMHA_USE_FP32_ACUM_FOR_OUT)
 // Does not seem to improve the accuracy
 //#define MMHA_USE_FP32_ACUM_FOR_LOGITS
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

#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
template <typename T>
struct Qk_vec_acum_fp32_
{
};

template <>
struct Qk_vec_acum_fp32_<float>
{
    using Type = float;
};

template <>
struct Qk_vec_acum_fp32_<float2>
{
    using Type = float2;
};

template <>
struct Qk_vec_acum_fp32_<float4>
{
    using Type = float4;
};

// template<> struct Qk_vec_acum_fp32_<uint16_t> { using Type = float;        };
template <>
struct Qk_vec_acum_fp32_<uint32_t>
{
    using Type = float2;
};

template <>
struct Qk_vec_acum_fp32_<uint2>
{
    using Type = Float4_;
};

template <>
struct Qk_vec_acum_fp32_<uint4>
{
    using Type = Float8_;
};

template <>
struct Qk_vec_acum_fp32_<__nv_bfloat16>
{
    using Type = float;
};

template <>
struct Qk_vec_acum_fp32_<__nv_bfloat162>
{
    using Type = float2;
};

template <>
struct Qk_vec_acum_fp32_<bf16_4_t>
{
    using Type = Float4_;
};

template <>
struct Qk_vec_acum_fp32_<bf16_8_t>
{
    using Type = Float8_;
};

#ifdef ENABLE_FP8
// template<>
// struct Qk_vec_acum_fp32_<fp8_2_t> {
//     using Type = float2;
// };
template <>
struct Qk_vec_acum_fp32_<fp8_4_t>
{
    using Type = Float4_;
};

// template<>
// struct Qk_vec_acum_fp32_<fp8_8_t> {
//     using Type = Float4_;
// };
#endif // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct K_vec_acum_fp32_
{
};

template <>
struct K_vec_acum_fp32_<float>
{
    using Type = float;
};

template <>
struct K_vec_acum_fp32_<float2>
{
    using Type = float2;
};

template <>
struct K_vec_acum_fp32_<float4>
{
    using Type = float4;
};

template <>
struct K_vec_acum_fp32_<Float8_>
{
    using Type = Float8_;
};

template <>
struct K_vec_acum_fp32_<uint32_t>
{
    using Type = float2;
};

template <>
struct K_vec_acum_fp32_<uint2>
{
    using Type = Float4_;
};

template <>
struct K_vec_acum_fp32_<uint4>
{
    using Type = Float8_;
};

template <>
struct K_vec_acum_fp32_<__nv_bfloat16>
{
    using Type = float;
};

template <>
struct K_vec_acum_fp32_<__nv_bfloat162>
{
    using Type = float2;
};

template <>
struct K_vec_acum_fp32_<bf16_4_t>
{
    using Type = Float4_;
};

template <>
struct K_vec_acum_fp32_<bf16_8_t>
{
    using Type = Float8_;
};
#ifdef ENABLE_FP8
// template<>
// struct K_vec_acum_fp32_<fp8_2_t> {
//     using Type = float2;
// };
template <>
struct K_vec_acum_fp32_<fp8_4_t>
{
    using Type = Float4_;
};

// template<>
// struct K_vec_acum_fp32_<fp8_8_t> {
//     using Type = Float4_;
// };
#endif // ENABLE_FP8
#endif // MMHA_USE_FP32_ACUM_FOR_FMA

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
template <typename T>
struct V_vec_acum_fp32_
{
};

template <>
struct V_vec_acum_fp32_<float>
{
    using Type = float;
};

template <>
struct V_vec_acum_fp32_<float2>
{
    using Type = float2;
};

template <>
struct V_vec_acum_fp32_<float4>
{
    using Type = float4;
};

template <>
struct V_vec_acum_fp32_<uint32_t>
{
    using Type = float2;
};

template <>
struct V_vec_acum_fp32_<uint2>
{
    using Type = Float4_;
};

template <>
struct V_vec_acum_fp32_<uint4>
{
    using Type = Float8_;
};
#ifdef ENABLE_BF16
template <>
struct V_vec_acum_fp32_<__nv_bfloat162>
{
    using Type = float2;
};

template <>
struct V_vec_acum_fp32_<bf16_4_t>
{
    using Type = Float4_;
};

template <>
struct V_vec_acum_fp32_<bf16_8_t>
{
    using Type = Float8_;
};
#endif // ENABLE_BF16
#ifdef ENABLE_FP8
// template<>
// struct V_vec_acum_fp32_<fp8_2_t> {
//     using Type = float2;
// };
template <>
struct V_vec_acum_fp32_<fp8_4_t>
{
    using Type = Float4_;
};

// template<>
// struct V_vec_acum_fp32_<fp8_8_t> {
//     using Type = Float4_;
// };
#endif // ENABLE_FP8
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tout, typename Tin>
__inline__ __device__ constexpr Tout vec_conversion(const Tin& x)
{
    static_assert(std::is_same<Tout, Tin>::value, "Type mismatch");
    return x;
}
#ifdef ENABLE_FP8
// fp8_t
template <>
__inline__ __device__ float vec_conversion<float, __nv_fp8_e4m3>(const __nv_fp8_e4m3& a)
{
    return float(a);
}

template <>
__inline__ __device__ __nv_fp8_e4m3 vec_conversion<__nv_fp8_e4m3, float>(const float& a)
{
    return __nv_fp8_e4m3(a);
}

// fp8_2_t
template <>
__inline__ __device__ float2 vec_conversion<float2, fp8_2_t>(const fp8_2_t& a)
{
    return float2(a);
}

template <>
__inline__ __device__ fp8_2_t vec_conversion<fp8_2_t, float2>(const float2& a)
{
    return fp8_2_t(a);
}

// fp8_4_t
template <>
__inline__ __device__ float4 vec_conversion<float4, fp8_4_t>(const fp8_4_t& a)
{
    return float4(a);
}

template <>
__inline__ __device__ fp8_4_t vec_conversion<fp8_4_t, float4>(const float4& a)
{
    return fp8_4_t(a);
}
#endif // ENABLE_FP8

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS_PER_KEY, typename K_vec, int N>
inline __device__ float qk_dot_(const K_vec (&q)[N], const K_vec (&k)[N])
{
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    using K_vec_acum = typename K_vec_acum_fp32_<K_vec>::Type;
#else
    using K_vec_acum = K_vec;
#endif
    // Compute the parallel products for Q*K^T (treat vector lanes separately).
    K_vec_acum qk_vec = mul<K_vec_acum, K_vec, K_vec>(q[0], k[0]);
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

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int THREADS_PER_KEY>
struct Qk_dot
{
    template <typename K_vec, int N>
    static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
    {
        return qk_dot_<THREADS_PER_KEY>(q, k);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 hmma_fp32(const uint2& a, uint32_t b)
{
    float4 c;
    float zero = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \n"
        "    {%0, %1, %2, %3}, \n"
        "    {%4, %5}, \n"
        "    {%6}, \n"
        "    {%7, %7, %7, %7}; \n"

        : "=f"(c.x), "=f"(c.y), "=f"(c.z), "=f"(c.w)
        : "r"(a.x) "r"(a.y), "r"(b), "f"(zero));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
inline __device__ float qk_hmma_dot_(const uint32_t (&q)[N], const uint32_t (&k)[N])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    using K_vec_acum = typename K_vec_acum_fp32_<uint32_t>::Type;
#else
    using K_vec_acum = uint32_t;
#endif
    K_vec_acum qk_vec = mul<K_vec_acum, uint32_t, uint32_t>(q[0], k[0]);
#pragma unroll
    for (int ii = 1; ii < N; ++ii)
    {
        qk_vec = fma(q[ii], k[ii], qk_vec);
    }
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
    uint32_t qk_vec_ = float2_to_half2(qk_vec);
    return hmma_fp32(make_uint2(qk_vec_, 0u), 0x3c003c00u).x;
#else
    return hmma_fp32(make_uint2(qk_vec, 0u), 0x3c003c00u).x;
#endif
#else
    return 0.f;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Qk_dot<uint16_t, 4>
{
    template <typename K_vec, int N>
    static inline __device__ float dot(const K_vec (&q)[N], const K_vec (&k)[N])
    {
        return qk_dot_<4>(q, k);
    }

    template <int N>
    static inline __device__ float dot(const uint32_t (&q)[N], const uint32_t (&k)[N])
    {
#if __CUDA_ARCH__ >= 750 && defined(MMHA_USE_HMMA_FOR_REDUCTION)
        return qk_hmma_dot_(q, k);
#else
        return qk_dot_<4>(q, k);
#endif // defined MMHA_USE_HMMA_FOR_REDUCTION
    }
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

#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)

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
    // Type of struct containing KV cache
    typename KVCacheBuffer,
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
    // The number of threads per key.
    unsigned THREADS_PER_KEY = threads_per_key<T, dh_max(Dh)>(),
    // The number of threads per value.
    unsigned THREADS_PER_VALUE = threads_per_value<T>(dh_max(Dh)),
    // The unroll factor for loading from K cache.
    unsigned K_LOOP_UNROLL = 8,
    // The unroll factor for loading from V cache.
    // Set it default to 4 for higher occupancy (by reducing registers usage).
    unsigned V_LOOP_UNROLL = 4>
__global__ void masked_multihead_attention_kernel(
    Multihead_attention_params<T, DO_CROSS_ATTENTION> params, KVCacheBuffer kvCacheBuffer)
{

    using Tk = typename kernel_type_t<T>::Type;
    // Use 8bit cache.
    static constexpr bool ENABLE_8BITS_CACHE = sizeof(Tcache) == 1;

    // The size of a warp.
    constexpr unsigned WARP_SIZE{32};
    // The number of warps in a threadblock.
    constexpr unsigned WARPS_PER_BLOCK{THREADS_PER_BLOCK / WARP_SIZE};

    // The maximum hidden size per head.
    constexpr auto Dh_MAX = dh_max(Dh);
    constexpr bool IS_Dh_MAX = Dh == Dh_MAX;
    static_assert(Dh_MAX >= WARP_SIZE);
    static_assert(Dh_MAX >= Dh);

    // The maximum sequence length in the kv_cache, i.e., an upper bound on L.
    // Note that the maximum sequence length supported by the model might be greater than this.
    const auto max_seq_len = static_cast<unsigned>(params.memory_max_len);
    assert(max_seq_len > 0);
    // The current timestep (including paddings).
    // It is only used to calculate the smem stride.
    const auto timestep = static_cast<unsigned>(DO_MULTI_BLOCK ? params.timesteps_per_block : params.timestep);

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
#ifndef MMHA_USE_FP32_ACUM_FOR_LOGITS
    if (sizeof(Tk) != 4)
    {
        // TODO - change to tlength
        const auto max_timesteps = DO_CROSS_ATTENTION ? max_seq_len : min(timestep, max_seq_len);
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
    using K_vec_m = typename packed_type<Tcache, num_elems<K_vec_k>::value>::type;

    // Use alignment for safely casting the shared buffers as Qk_vec_k and K_vec_k.
    // Shared memory to store Q inputs.
    __shared__ __align__(mmha::const_max(sizeof(Qk_vec_k), sizeof(K_vec_k))) Tk q_smem[Dh_MAX];

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
        Tk bias_smem[bias_smem_size];

    // The number of elements per vector.
    constexpr unsigned QK_VEC_SIZE{sizeof(Qk_vec_m) / sizeof(T)};
    // Make sure the hidden size per head is a multiple of the vector size.
    static_assert(Dh_MAX % QK_VEC_SIZE == 0);
    // We will use block wide reduction if needed
    // The number of vectors per Dh_MAX.
    constexpr unsigned QK_VECS_PER_Dh_MAX{Dh_MAX / QK_VEC_SIZE};
    static_assert(THREADS_PER_BLOCK >= QK_VECS_PER_Dh_MAX);

    // The batch/beam idx
    const auto bi = blockIdx.y;
    if (params.finished != nullptr && params.finished[bi])
    {
        return;
    }
    // The head.
    const unsigned hi{blockIdx.x};
    // The head index of keys and values adjusted for MQA/GQA.
    const int qhead_per_kv{params.num_heads / params.num_kv_heads};
    const unsigned hi_kv{hi / qhead_per_kv};
    // The number of heads.
    const auto num_heads = static_cast<unsigned>(params.num_heads);
    // The number of heads for keys and values adjusted for MQA/GQA.
    const auto num_heads_kv = static_cast<unsigned>(params.num_kv_heads);

    // The thread in the block.
    const unsigned tidx{threadIdx.x};

    // The column tile along L dimension on K^T -- noted as T_c in flash-attention paper
    const unsigned c_tile{MULTI_BLOCK_FLAG ? blockIdx.z : 0};

    // Indicate if we need to compute the K/V cache element (add KV bias, IA3, RoPE, etc.) and update the cache.
    // For Self-Attention, it's always required.
    // For Cross-Attention, as everything is pre-computed,
    // in the context phase of the encoder, it's not needed in that kernel.
    // Therefore, handle_kv is !DO_CROSS_ATTENTION and irrelevant of timestep.
    const bool handle_kv{!DO_CROSS_ATTENTION};

    // While doing the product Q*K^T for the different keys we track the max.
    float qk_max = -FLT_MAX;

    float qk = 0.0F;

    // Compute relative attention bias on the fly, with relative attention table [head_num/TP, num_buckets] passed in.
    // num_buckets passed as params.relative_attention_bias_stride, max_distance passed as params.max_distance
    bool implicit_rel_attn_bias = params.max_distance != 0;
    int relative_attention_bias_stride
        = params.relative_attention_bias_stride; // num_buckets might be modified below, save it beforehand
    int max_distance = params.max_distance;

    // The actual sequence length excluding the paddings.
    // minus 1 because it includes the current timestep while tlength denotes the kv cache length.
    const int tlength = DO_CROSS_ATTENTION
        ? params.memory_length_per_sample[bi] - 1
        : (params.length_per_sample ? (params.length_per_sample[bi] - 1) : static_cast<int>(timestep));
    // The context length for beam searching optimization (all points to beam 0).
    const int input_length = params.input_lengths[bi];

    // The offset in the Q and K buffer also accounts for the batch.
    const auto qk_vec_idx = tidx * QK_VEC_SIZE;
    const auto is_valid_qk_vec = qk_vec_idx < Dh;

    const bool load_qkv_quant = params.qkv_scale_quant_orig != nullptr;
    const bool write_attention_quant = params.attention_out_scale_orig_quant != nullptr;

    // Quant/Dequant scales for 8bits kv cache.
    using T_scale = typename kv_cache_scale_type_t<T, Tcache>::Type;
    T_scale kv_scale_quant_orig, kv_scale_orig_quant;
    convert_from_float(&kv_scale_quant_orig, (ENABLE_8BITS_CACHE ? params.kv_scale_quant_orig[0] : 1.0f));
    convert_from_float(&kv_scale_orig_quant, (ENABLE_8BITS_CACHE ? params.kv_scale_orig_quant[0] : 1.0f));

    // Up to QK_VECS_PER_Dh_MAX threads load Q and K + the bias values for the current timestep.
    // Trigger the loads from the Q and K buffers.
    Qk_vec_k q, k, q_bias, k_bias;
    zero(q);
    zero(k);
    zero(q_bias);
    zero(k_bias);
    float rotary_embedding_base = params.rotary_embedding_base;
    float rotary_embedding_scale = params.rotary_embedding_scale;
    if (is_valid_qk_vec)
    {
        mmha::update_rotary_base_n_scale(rotary_embedding_base, rotary_embedding_scale,
            params.rotary_embedding_scale_type, params.rotary_embedding_dim, params.rotary_embedding_max_positions,
            tlength);
        // Query
        // The stride between tokens. We may be able to always use params.stride.
        uint32_t q_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads * Dh);
        // The offset.
        const auto q_offset = tensorrt_llm::common::flat_index_strided3(bi, hi, qk_vec_idx, q_stride, Dh);

        if (load_qkv_quant)
        {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
            using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
            const auto q_scaling = params.qkv_scale_quant_orig[0];
            const auto q_quant
                = *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.q)[q_offset]);
            convert_from_float(&q, mul<Packed_Float_t, float>(q_scaling, float_from_int8(q_quant)));
        }
        else
        {
            q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.q[q_offset]));
        }

        if constexpr (DO_CROSS_ATTENTION)
        {
            const auto k_idx = QK_VEC_SIZE * tidx;
            const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(tlength, hi, Dh, k_idx);
            Tcache* k_cache = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(bi, tlength));

            k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&k_cache[inBlockIdx]));
        }
        else
        {
            // Key
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t k_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            // The offset.
            const auto k_offset = tensorrt_llm::common::flat_index_strided3(bi, hi_kv, qk_vec_idx, k_stride, Dh);

            if (load_qkv_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<Qk_vec_m>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<Qk_vec_m>::value>::type;
                const auto k_scaling = params.qkv_scale_quant_orig[1];
                const auto k_quant
                    = *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.k)[k_offset]);

                convert_from_float(&k, mul<Packed_Float_t, float>(k_scaling, float_from_int8(k_quant)));
            }
            else
            {
                k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.k[k_offset]));
            }
        }

        if (params.q_bias != nullptr)
        {
            const auto q_bias_offset = tensorrt_llm::common::flat_index2(hi, qk_vec_idx, Dh);
            q_bias
                = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.q_bias[q_bias_offset]));
        }
        if (handle_kv && params.k_bias != nullptr)
        {
            const auto k_bias_offset = tensorrt_llm::common::flat_index2(hi_kv, qk_vec_idx, Dh);
            k_bias
                = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(&params.k_bias[k_bias_offset]));
        }
    }

    // Computes the Q/K values with bias.
    q = add(q, q_bias);
    if (handle_kv)
    {
        k = add(k, k_bias);
    }

    const bool do_ia3 = handle_kv && params.ia3_tasks != nullptr;
    const auto beam_width = static_cast<unsigned>(params.beam_width);
    const auto ia3_ti_hi = do_ia3
        ? tensorrt_llm::common::flat_index2(static_cast<unsigned>(params.ia3_tasks[bi / beam_width]), hi, num_heads)
        : 0;

    if (do_ia3 && is_valid_qk_vec)
    {
        k = mul<Qk_vec_k, Qk_vec_k, Qk_vec_k>(k,
            vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(
                &params.ia3_key_weights[tensorrt_llm::common::flat_index2(ia3_ti_hi, qk_vec_idx, Dh)])));
    }

    // Note we have no paddings in KV cache now.
    switch (params.position_embedding_type)
    {
    case PositionEmbeddingType::kLEARNED_ABSOLUTE:
    case PositionEmbeddingType::kRELATIVE:
    case PositionEmbeddingType::kALIBI:
    case PositionEmbeddingType::kALIBI_WITH_SCALE: break;
    case PositionEmbeddingType::kROPE_GPTJ:
    {
        if (handle_kv)
        {
            apply_rotary_embedding(q, k, tidx, params.rotary_embedding_dim, params.rotary_embedding_base,
                params.rotary_embedding_scale, tlength);
        }
        else
        {
            apply_rotary_embedding(q, tidx, params.rotary_embedding_dim, params.rotary_embedding_base,
                params.rotary_embedding_scale, tlength);
        }
        break;
    }
    case PositionEmbeddingType::kROPE_GPT_NEOX:
    {
        const bool do_rotary = is_valid_qk_vec && QK_VEC_SIZE * tidx < params.rotary_embedding_dim;

        T* q_smem_ = reinterpret_cast<T*>(smem_);
        T* k_smem = q_smem_ + params.rotary_embedding_dim;

        const int half_rotary_dim = params.rotary_embedding_dim / 2;
        const int half_idx = qk_vec_idx / half_rotary_dim;
        const int intra_half_idx = qk_vec_idx % half_rotary_dim;
        const int smem_pitch = half_rotary_dim; // TODO: adjust for bank conflicts

        assert(half_rotary_dim % QK_VEC_SIZE == 0);

        if (do_rotary)
        {
            *reinterpret_cast<Qk_vec_k*>(q_smem_ + half_idx * smem_pitch + intra_half_idx) = q;
            if (handle_kv)
            {
                *reinterpret_cast<Qk_vec_k*>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
            }
        }

        __syncthreads();

        const int transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
        constexpr int tidx_factor = (QK_VEC_SIZE > 1) ? QK_VEC_SIZE / 2 : 1;
        if (do_rotary)
        {
            mmha::vec_from_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
            if (handle_kv)
            {
                mmha::vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);

                mmha::apply_rotary_embedding(q, k, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                    rotary_embedding_base, rotary_embedding_scale, tlength);

                mmha::write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
            }
            else
            {
                mmha::apply_rotary_embedding(q, transpose_idx / tidx_factor, params.rotary_embedding_dim,
                    rotary_embedding_base, rotary_embedding_scale, tlength);
            }
            mmha::write_smem_transpose(q, q_smem_, transpose_idx, smem_pitch);
        }

        __syncthreads();

        if (do_rotary)
        {
            q = *reinterpret_cast<Qk_vec_k*>(q_smem_ + half_idx * smem_pitch + intra_half_idx);
            if (handle_kv)
            {
                k = *reinterpret_cast<Qk_vec_k*>(k_smem + half_idx * smem_pitch + intra_half_idx);
            }
        }

        __syncthreads();
        break;
    }
    }

    // For the same reason as handle_kv, no compute needed in Cross-Attention's 1st step

    if (qk_vec_idx < Dh_MAX)
    {

        // Store the Q values to shared memory.
        // Set padded Dh to 0 for the correctness of QK (when Dh != Dh_Max).
        Qk_vec_k zero_q;
        zero(zero_q);

        *reinterpret_cast<Qk_vec_k*>(&q_smem[qk_vec_idx]) = is_valid_qk_vec ? q : zero_q;

        // Write the K values to the global memory cache.
        //
        // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
        // system. We designed it this way as it allows much better memory loads (and there are many
        // more loads) + the stores are really "write and forget" since we won't need the ack before
        // the end of the kernel. There's plenty of time for the transactions to complete.

        // For MQA/GQA mode, write only with the first Q head of each group per KV head.
        if (handle_kv && hi == (hi_kv * qhead_per_kv) && (IS_Dh_MAX || is_valid_qk_vec))
        {
            // Trigger the stores to global memory.
            const auto k_idx = QK_VEC_SIZE * tidx;
            const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(tlength, hi_kv, Dh, k_idx);
            // The base pointer for the value in the cache buffer.
            Tcache* k_cache = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(bi, tlength));

            if constexpr (ENABLE_8BITS_CACHE)
            {
                store_8bits_kv_cache_vec(reinterpret_cast<Tcache*>(k_cache), k, inBlockIdx, kv_scale_orig_quant);
            }
            else
            {
                *reinterpret_cast<Qk_vec_m*>(&k_cache[inBlockIdx]) = vec_conversion<Qk_vec_m, Qk_vec_k>(k);
            }
        }

        // Compute \sum_i Q[i] * K^T[i] for the current timestep.
#ifdef MMHA_USE_FP32_ACUM_FOR_FMA
        using Qk_vec_acum = typename Qk_vec_acum_fp32_<Qk_vec_k>::Type;
#else
        using Qk_vec_acum = Qk_vec_k;
#endif
        qk = dot<Qk_vec_acum, Qk_vec_k>(q, k);
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

    // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
    if (tidx == 0)
    {
        // Normalize qk.
        qk *= params.inv_sqrt_dh;
        if (params.relative_attention_bias != nullptr)
        {
            if (implicit_rel_attn_bias)
            {
                // Here i == j == tlength, so relative_position = 0 --> relative_buckets = 0.
                T rel_attn_bias = params.relative_attention_bias[hi * relative_attention_bias_stride + 0];
                qk = add(qk, rel_attn_bias);
            }
            else
            {
                qk = add(qk,
                    params.relative_attention_bias[hi * params.relative_attention_bias_stride
                            * params.relative_attention_bias_stride
                        + tlength * params.relative_attention_bias_stride + tlength]);
            }
        }
        // We don't need to apply the linear position bias here since qi - ki = 0 yields the position bias 0.

        qk_max = qk;
        // qk_smem[params.timestep] = qk;
        if (MULTI_BLOCK_FLAG)
        {
            qk_current_smem[0] = qk;
        }
        else
        {
            qk_smem[tlength] = qk;
        }
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    constexpr unsigned K_ELTS_PER_CHUNK{THREADS_PER_KEY * K_VEC_SIZE};

    // The positions of the cache buffer (for this B * H) and the vector within that chunk associated with this
    // thread.
    const auto k_idx = chunk_index<T, K_vec_k, THREADS_PER_KEY>(tidx);

    // The number of vectors per thread.
    constexpr unsigned K_VECS_PER_THREAD{Dh_MAX / K_ELTS_PER_CHUNK};
    static_assert(Dh_MAX == K_ELTS_PER_CHUNK * K_VECS_PER_THREAD);

    // Load the Q values from shared memory. The values are reused during the loop on K.
    K_vec_k q_vec[K_VECS_PER_THREAD];
#pragma unroll
    for (unsigned ii = 0; ii < K_VECS_PER_THREAD; ++ii)
    {
        q_vec[ii] = *reinterpret_cast<const K_vec_k*>(
            &q_smem[tensorrt_llm::common::flat_index2(ii, k_idx.y, K_ELTS_PER_CHUNK)]);
    }

    // The number of timesteps loaded per iteration, i.e., (THREADS_PER_BLOCK * THREADS_PER_BLOCK) / 256 <= 256
    constexpr unsigned K_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_KEY};
    // The number of keys per warp.
    constexpr unsigned K_PER_WARP{WARP_SIZE / THREADS_PER_KEY};
    // The number of unrolled keys per warp.
    constexpr unsigned UNROLLED_K_PER_WARP = K_PER_WARP * K_LOOP_UNROLL;
    // The number of unrolled keys per ieration.
    constexpr unsigned UNROLLED_K_PER_ITER = K_PER_ITER * K_LOOP_UNROLL;

    // Base pointer for the row of pointers to k cache blocks
    void** k_cache_base_row_ptr = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::K_IDX, bi));

    const auto timesteps_per_block = static_cast<unsigned>(params.timesteps_per_block);

    // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
    // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.
    const int context_length = HAS_BEAMS ? input_length : tlength;
    const auto context_ti_end = MULTI_BLOCK_FLAG
        ? divUp(timesteps_per_block, UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP
        : divUp(static_cast<unsigned>(context_length), UNROLLED_K_PER_WARP) * UNROLLED_K_PER_WARP;

    // The generation ti_end.
    const auto generation_ti_end = MULTI_BLOCK_FLAG ? divUp(timesteps_per_block, K_PER_WARP) * K_PER_WARP
                                                    : divUp(static_cast<unsigned>(tlength), K_PER_WARP) * K_PER_WARP;

    // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
    const auto bi_seq_len_offset = static_cast<std::size_t>(bi) * max_seq_len;
    const int* beam_indices = HAS_BEAMS ? &params.cache_indir[bi_seq_len_offset] : nullptr;

    const auto c_tile_times_timesteps_per_block = c_tile * timesteps_per_block; // 0 if !MULTI_BLOCK_FLAG

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Key cache loops for dot(Q, K).

    // Handle only context key cache with beam searching.
    // Handle both context and generation key cache without beam searching.
    // Explict batching of LDGs (by K_LOOP_UNROLL) as it doesn't depend on indirection tables.
    for (int ti = k_idx.x; ti < context_ti_end; ti += UNROLLED_K_PER_ITER)
    {
        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

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
                const int valid_time_now = min(time_now + k_loop * K_PER_ITER, context_length - 1);
                const int seqIdx = bi / beam_width * beam_width;
                // Base pointer to k cache block for beam's batch
                Tcache* k_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(seqIdx, valid_time_now));

                int inBlockIdx = kvCacheBuffer.getKVLocalIdx(valid_time_now, hi_kv, Dh, jj);
                k_vec_cache[k_loop][k_vec_i] = *reinterpret_cast<const K_vec_m*>(&k_cache_batch[inBlockIdx]);
            }
        }

#pragma unroll
        for (int k_loop = 0; k_loop < K_LOOP_UNROLL; ++k_loop)
        {
            const int local_time_now = time_now + k_loop * K_PER_ITER;
            const int local_ti = ti + k_loop * K_PER_ITER;

            K_vec_k k_vec[K_VECS_PER_THREAD];
#pragma unroll
            for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
            {
                // we move quantization to here for better batching of inflight LDGs.
                if constexpr (ENABLE_8BITS_CACHE)
                {
                    convert_from_8bit_kv_cache<K_vec_m, K_vec_k, Tcache, T_scale>(
                        &k_vec[k_vec_i], k_vec_cache[k_loop][k_vec_i], kv_scale_quant_orig);
                }
                else
                {
                    // K_vek is same as K_vec_cache in this case.
                    k_vec[k_vec_i] = *reinterpret_cast<K_vec_k*>(&k_vec_cache[k_loop][k_vec_i]);
                }
            }

            // Perform the dot product and normalize qk.
            //
            // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
            float qk_{Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh};

            // For multi-block mode, we still need to make sure it will not be OOB.
            if (MULTI_BLOCK_FLAG && local_ti >= timesteps_per_block)
            {
                continue;
            }

            // Store the product to shared memory. There's one qk value per timestep. Update the max.
            if (local_time_now < context_length && tidx % THREADS_PER_KEY == 0)
            {
                if (params.relative_attention_bias != nullptr)
                {
                    if (implicit_rel_attn_bias)
                    {
                        // Compute bias value on the fly (See bert_preprocess_kernels.cu::buildRelativeAttentionBias)
                        int relative_buckets = 0;
                        int relative_position = local_time_now - tlength;
                        int num_buckets = relative_attention_bias_stride;
                        // Special logic in T5 relative attention, both encoder & decoder use this, because
                        // relative_attention_bias is pre-computed once and passed around.
                        num_buckets /= 2;
                        relative_buckets += relative_position > 0 ? num_buckets : 0;
                        relative_position = abs(relative_position);
                        int max_exact = num_buckets / 2;
                        bool is_small = relative_position < max_exact;
                        int relative_position_if_large = max_exact
                            + (int) (logf(relative_position * 1.0f / max_exact) / logf((float) max_distance / max_exact)
                                * (num_buckets - max_exact));
                        relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                        relative_buckets += is_small ? relative_position : relative_position_if_large;
                        T rel_attn_bias
                            = params.relative_attention_bias[hi * relative_attention_bias_stride + relative_buckets];
                        qk_ = add(qk_, rel_attn_bias);
                    }
                    else
                    {
                        qk_ = add(qk_,
                            params.relative_attention_bias[hi * params.relative_attention_bias_stride
                                    * params.relative_attention_bias_stride
                                + tlength * params.relative_attention_bias_stride + local_time_now]);
                    }
                }
                if (params.linear_bias_slopes != nullptr)
                {
                    // Apply the linear position bias: (ki - qi) * slope[hi].
                    // The padding token locates between the input context and the generated tokens.
                    // We need to remove the number of padding tokens in the distance computation.
                    //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
                    //   token: i i i i p p p o o o where i=input, p=pad, o=output.
                    // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
                    float dist = local_time_now - tlength;

                    qk_ += mul<float, T, float>(params.linear_bias_slopes[hi], dist);
                }

                // Calculate the max for softmax, and store qk back to smem.
                // Don't need mask here as we remove paddings in kv cache.
                qk_max = fmaxf(qk_max, qk_);
                qk_smem[local_ti] = qk_;
            }
        }
    }

    // Handle generation key cache with beam searching.
    // Note that it may be overlapped with the context key loop, but it won't impact the corretness.
    if (HAS_BEAMS)
    {
        // For multi-block mode, the last few blocks will handle the generation key cache.
        if (!MULTI_BLOCK_FLAG || (c_tile + 1) * timesteps_per_block > input_length)
        {
            const int generation_start_ti = k_idx.x
                + ((MULTI_BLOCK_FLAG ? input_length % timesteps_per_block : input_length) / K_PER_WARP) * K_PER_WARP;
            for (int ti = generation_start_ti; ti < generation_ti_end; ti += K_PER_ITER)
            {
                const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

                // The keys loaded from the key cache.
                K_vec_k k_vec[K_VECS_PER_THREAD];

#pragma unroll
                for (int k_vec_i = 0; k_vec_i < K_VECS_PER_THREAD; ++k_vec_i)
                {
                    const int jj = min(k_idx.y + k_vec_i * K_ELTS_PER_CHUNK, Dh - K_VEC_SIZE);
                    const int valid_time_now = min(time_now, tlength - 1);
                    int beam_offset = beam_indices[valid_time_now];
                    const int seqIdx = bi / beam_width * beam_width + beam_offset;
                    // Base pointer to k cache block for beam's batch, before offsetting with indirection buffer
                    Tcache* k_cache_batch
                        = reinterpret_cast<Tcache*>(kvCacheBuffer.getKBlockPtr(seqIdx, valid_time_now));

                    int inBlockIdx = kvCacheBuffer.getKVLocalIdx(valid_time_now, hi_kv, Dh, jj);
                    if constexpr (ENABLE_8BITS_CACHE)
                    {
                        load_8bits_kv_cache_vec(&k_vec[k_vec_i], k_cache_batch, inBlockIdx, kv_scale_quant_orig);
                    }
                    else
                    {
                        k_vec[k_vec_i] = (*reinterpret_cast<const K_vec_k*>(&k_cache_batch[inBlockIdx]));
                    }
                }

                // Perform the dot product and normalize qk.
                //
                // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
                float qk_{Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k_vec) * params.inv_sqrt_dh};

                // Store the product to shared memory. There's one qk value per timestep. Update the max.
                if (time_now >= input_length && time_now < tlength && tidx % THREADS_PER_KEY == 0)
                {
                    if (params.relative_attention_bias != nullptr)
                    {
                        if (implicit_rel_attn_bias)
                        {
                            // Compute bias value on the fly (See
                            // bert_preprocess_kernels.cu::buildRelativeAttentionBias)
                            int relative_buckets = 0;
                            int relative_position = time_now - tlength;
                            int num_buckets = relative_attention_bias_stride;
                            // Special logic in T5 relative attention, both encoder & decoder use this, because
                            // relative_attention_bias is pre-computed once and passed around.
                            num_buckets /= 2;
                            relative_buckets += relative_position > 0 ? num_buckets : 0;
                            relative_position = abs(relative_position);
                            int max_exact = num_buckets / 2;
                            bool is_small = relative_position < max_exact;
                            int relative_position_if_large = max_exact
                                + (int) (logf(relative_position * 1.0f / max_exact)
                                    / logf((float) max_distance / max_exact) * (num_buckets - max_exact));
                            relative_position_if_large = min(relative_position_if_large, num_buckets - 1);
                            relative_buckets += is_small ? relative_position : relative_position_if_large;
                            T rel_attn_bias
                                = params
                                      .relative_attention_bias[hi * relative_attention_bias_stride + relative_buckets];
                            qk_ = add(qk_, rel_attn_bias);
                        }
                        else
                        {
                            qk_ = add(qk_,
                                params.relative_attention_bias[hi * params.relative_attention_bias_stride
                                        * params.relative_attention_bias_stride
                                    + tlength * params.relative_attention_bias_stride + time_now]);
                        }
                    }
                    if (params.linear_bias_slopes != nullptr)
                    {
                        // Apply the linear position bias: (ki - qi) * slope[hi].
                        // The padding token locates between the input context and the generated tokens.
                        // We need to remove the number of padding tokens in the distance computation.
                        //   ti   : 0 1 2 3 4 5 6 7 8 9(tlength)
                        //   token: i i i i p p p o o o where i=input, p=pad, o=output.
                        // e.g. ti = 2, dist = (9 - 3) - 2 = 4.
                        float dist = time_now - tlength;

                        qk_ += mul<float, T, float>(params.linear_bias_slopes[hi], dist);
                    }

                    // Calculate the max for softmax, and store qk back to smem.
                    qk_max = fmaxf(qk_max, qk_);
                    qk_smem[ti] = qk_;
                }
            }
        }
    }

////////////////////////////////////////////////////////////////////////////////////////////////
// Softmax.

// Perform the final reduction to compute the max inside each warp.
//
// NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
// group so it's not needed to run the reduction inside the group (again).
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2)
    {
        qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }

    // Decompose the thread index into warp and lane.
    const auto warp = tidx / WARP_SIZE;
    const auto lane = tidx % WARP_SIZE;

    // The warp leader writes the max to shared memory.
    if (lane == 0)
    {
        red_smem[warp] = qk_max;
    }

    // Make sure the products are in shared memory.
    __syncthreads();

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
    const int logit_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : tlength;
    for (int ti = tidx; ti <= logit_loop_end; ti += THREADS_PER_BLOCK)
    {

        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

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
            if (time_now < tlength && ti != timesteps_per_block)
            {
                float logit = __expf(qk_smem[ti] - qk_max);
                sum += logit;
                qk_smem[ti] = logit;
            }
            else if (time_now == tlength)
            {
                float logit = __expf(qk_current_smem[0] - qk_max);
                sum += logit;
                qk_current_smem[0] = logit;
            }
        }
    }

    // Compute the sum.
    sum = block_sum<WARPS_PER_BLOCK>(&red_smem[WARPS_PER_BLOCK], sum);

    // Normalize the logits.
    float inv_sum = __fdividef(1.f, sum + 1.e-6f);

    const int normlization_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : tlength;
    for (int ti = tidx; ti <= normlization_loop_end; ti += THREADS_PER_BLOCK)
    {

        const int time_now = MULTI_BLOCK_FLAG ? ti + c_tile_times_timesteps_per_block : ti;

        if (!MULTI_BLOCK_FLAG)
        {
            convert_from_float(&logits_smem[ti], qk_smem[ti] * inv_sum);
        }
        else
        {
            // no scaling factor inv_sum applied here, will apply the scaling factor after all blocks finished
            if (time_now < tlength && ti != timesteps_per_block)
            {
                convert_from_float(&logits_smem[ti], qk_smem[ti]);
            }
            else if (time_now == tlength)
            {
                convert_from_float(&logits_current_smem[0], qk_current_smem[0]);
            }
        }
    }

    // Put Values part below so we leverage __syncthreads
    // from the previous step

    const auto v_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
    // The value computed by this thread.
    const auto vo = v_idx.x;
    // The hidden dimensions computed by this particular thread.
    const auto vi = v_idx.y;
    // Base pointer for the row of pointers to v cache blocks
    void** v_cache_base_row_ptr = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, bi));
    // Base pointer for the row of pointers to v cache blocks for beam's batch, before offsetting with indirection
    // buffer
    void** v_cache_batch_row_ptr
        = reinterpret_cast<void**>(kvCacheBuffer.getRowPtr(KVIdxType::V_IDX, bi / beam_width * beam_width));

    // The number of values processed per iteration of the loop.
    constexpr unsigned V_PER_ITER{THREADS_PER_BLOCK / THREADS_PER_VALUE};
    // The number of unrolled keys per ieration.
    constexpr unsigned UNROLLED_V_PER_ITER = V_PER_ITER * V_LOOP_UNROLL;

    bool const is_valid_vi = IS_Dh_MAX || vi < Dh;

    // One group of threads computes the product(s) for the current timestep.
    V_vec_k v_bias;
    zero(v_bias);
    // if( vo == params.timestep % V_PER_ITER ) {
    if (is_valid_vi && handle_kv && vo == tlength % V_PER_ITER)
    {
        // Trigger the loads from the V bias buffer.
        if (params.v_bias != nullptr)
        {
            const auto v_bias_offset = tensorrt_llm::common::flat_index2(hi_kv, vi, Dh);
            v_bias = *reinterpret_cast<const V_vec_k*>(&params.v_bias[v_bias_offset]);
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

#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
    using V_vec_acum = typename V_vec_acum_fp32_<V_vec_k>::Type;
#else
    using V_vec_acum = V_vec_k;
#endif
    // The partial outputs computed by each thread.
    V_vec_acum out;
    zero(out);

    // Loop over the timesteps to compute the partial outputs.
    if (is_valid_vi)
    {
        // Handle only context value cache with beam searching.
        // Handle both context and generation value cache without beam searching.
        // Explict batching of LDGs (by V_LOOP_UNROLL) as it doesn't depend on indirection tables.
        // Take all previous cache as context when we have no beam searching in order to batch as many LDGs as possible.
        const int context_length = HAS_BEAMS ? input_length : tlength;
        int context_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : context_length;
        int generation_v_loop_end = MULTI_BLOCK_FLAG ? timesteps_per_block : tlength;
        for (int ti = vo; ti < context_v_loop_end; ti += UNROLLED_V_PER_ITER)
        {
            V_vec_m v_vec_cache[V_LOOP_UNROLL];
#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
            {
                // Fetch offset based on cache_indir when beam sampling
                int time_idx = ti + v_loop * V_PER_ITER + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                time_idx = min(time_idx, tlength - 1);
                int rowIdx = bi / beam_width * beam_width;

                const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                // The base pointer for the value in the cache buffer.
                Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));

                v_vec_cache[v_loop] = *reinterpret_cast<const V_vec_m*>(&v_cache_batch[inBlockIdx]);
            }

#pragma unroll
            for (int v_loop = 0; v_loop < V_LOOP_UNROLL; v_loop++)
            {
                V_vec_k v_vec;
                // we move quantization to here for better batching of inflight LDGs.
                if constexpr (ENABLE_8BITS_CACHE)
                {
                    convert_from_8bit_kv_cache<V_vec_m, V_vec_k, Tcache, T_scale>(
                        &v_vec, v_vec_cache[v_loop], kv_scale_quant_orig);
                }
                else
                {
                    // V_vek is same as V_vec_cache in this case.
                    v_vec = *reinterpret_cast<V_vec_k*>(&v_vec_cache[v_loop]);
                }

                int local_time_idx = ti + v_loop * V_PER_ITER;
                int time_idx = local_time_idx + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);

                const bool is_mask
                    = (MULTI_BLOCK_FLAG && local_time_idx >= timesteps_per_block) || (time_idx >= context_length);
                // Load the logits from shared memory.
                if (!is_mask)
                {
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
                    float logit = logits_smem[local_time_idx];
                    out = fma(logit, cast_to_float(v_vec), out);
#else  // MMHA_USE_FP32_ACUM_FOR_LOGITS
                    Tk logit = logits_smem[local_time_idx];
                    out = fma(logit, v_vec, out);
#endif // MMHA_USE_FP32_ACUM_FOR_LOGITS
                }
            }
        }

        // Handle generation value cache with beam searching.
        if (HAS_BEAMS)
        {
            const auto generation_start_ti = MULTI_BLOCK_FLAG ? vo : (vo + (input_length / V_PER_ITER) * V_PER_ITER);
            // Only the last few blocks need to handle the generation value cache.
            if (!MULTI_BLOCK_FLAG || (c_tile + 1) * timesteps_per_block > input_length)
            {
                for (int ti = generation_start_ti; ti < generation_v_loop_end; ti += V_PER_ITER)
                {
                    // Fetch offset based on cache_indir when beam sampling
                    int time_idx = ti + (MULTI_BLOCK_FLAG ? c_tile_times_timesteps_per_block : 0);
                    int local_time_idx = ti;
                    if (time_idx < input_length || (MULTI_BLOCK_FLAG && time_idx >= tlength))
                    {
                        continue;
                    }
                    int rowIdx = bi / beam_width * beam_width + beam_indices[time_idx];

                    V_vec_k v;
                    const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(time_idx, hi_kv, Dh, vi);
                    // The base pointer for the value in the cache buffer.
                    Tcache* v_cache_batch = reinterpret_cast<Tcache*>(kvCacheBuffer.getVBlockPtr(rowIdx, time_idx));

                    if (ENABLE_8BITS_CACHE)
                    {
                        load_8bits_kv_cache_vec(&v, v_cache_batch, inBlockIdx, kv_scale_quant_orig);
                    }
                    else
                    {
                        v = *reinterpret_cast<const V_vec_k*>(&v_cache_batch[inBlockIdx]);
                    }

                    // Load the logits from shared memory.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
                    float logit = logits_smem[local_time_idx];
                    out = fma(logit, cast_to_float(v), out);
#else  // MMHA_USE_FP32_ACUM_FOR_LOGITS
                    Tk logit = logits_smem[local_time_idx];
                    out = fma(logit, v, out);
#endif // MMHA_USE_FP32_ACUM_FOR_LOGITS
                }
            }
        }
    }

    // One group of threads computes the product(s) for the current timestep.
    if (vo == tlength % V_PER_ITER && is_valid_vi && (!MULTI_BLOCK_FLAG || (c_tile == gridDim.z - 1)))
    {
        const int tokenIdx = tlength;
        const int inBlockIdx = kvCacheBuffer.getKVLocalIdx(tokenIdx, hi_kv, Dh, vi);
        // The base pointer for the value in the cache buffer.
        Tcache* v_cache_base = reinterpret_cast<Tcache*>(kvCacheBuffer.getBlockPtr(v_cache_base_row_ptr, tokenIdx));

        V_vec_k v;
        if (DO_CROSS_ATTENTION)
        {
            if constexpr (ENABLE_8BITS_CACHE)
            {
                // To verify
                load_8bits_kv_cache_vec(&v, v_cache_base, inBlockIdx, kv_scale_quant_orig);
            }
            else
            {
                v = vec_conversion<V_vec_k, V_vec_m>(*reinterpret_cast<const V_vec_m*>(&v_cache_base[inBlockIdx]));
            }
        }
        else
        {
            // Trigger the loads from the V buffer.
            // The stride between tokens. We may be able to always use params.stride.
            uint32_t v_stride = params.stride ? static_cast<uint32_t>(params.stride) : (num_heads_kv * Dh);
            // The offset.
            const auto v_offset = tensorrt_llm::common::flat_index_strided3(bi, hi_kv, vi, v_stride, Dh);

            if (load_qkv_quant)
            {
                using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_k>::value>::type;
                using Packed_Float_t = typename packed_type<float, num_elems<V_vec_k>::value>::type;
                const auto v_scaling = params.qkv_scale_quant_orig[2];
                const auto v_quant
                    = *reinterpret_cast<const Packed_Int8_t*>(&reinterpret_cast<const int8_t*>(params.v)[v_offset]);

                convert_from_float(&v, mul<Packed_Float_t, float>(v_scaling, float_from_int8(v_quant)));
            }
            else
            {
                v = *reinterpret_cast<const V_vec_k*>(&params.v[v_offset]);
            }
        }

        if (handle_kv)
        {
            // Compute the V values with bias.
            v = add(v, v_bias);

            if (do_ia3)
            {
                v = mul<V_vec_k, V_vec_k, V_vec_k>(v,
                    *reinterpret_cast<const V_vec_k*>(
                        &params.ia3_value_weights[tensorrt_llm::common::flat_index2(ia3_ti_hi, vi, Dh)]));
            }
        }

        // Store the values with bias back to global memory in the cache for V.
        //*reinterpret_cast<V_vec_k*>(&v_cache[params.timestep*Dh]) = v;
        // For MQA/GQA mode, write only with the first Q head of each group per KV head.
        if (hi == (hi_kv * qhead_per_kv))
        {
            if (ENABLE_8BITS_CACHE)
            {
                store_8bits_kv_cache_vec(v_cache_base, v, inBlockIdx, kv_scale_orig_quant);
            }
            else
            {
                *reinterpret_cast<V_vec_k*>(&v_cache_base[inBlockIdx]) = v;
            }
        }

        // Initialize the output value with the current timestep.
#if defined(MMHA_USE_FP32_ACUM_FOR_LOGITS)
        // out = fma(logits_smem[params.timestep], cast_to_float(v), out);
        if (!MULTI_BLOCK_FLAG)
        {
            out = fma(logits_smem[tlength], cast_to_float(v), out);
        }
        else
        {
            out = fma(logits_current_smem[0], cast_to_float(v), out);
        }
#else  // MMHA_USE_FP32_ACUM_FOR_LOGITS
       // out = fma(logits_smem[params.timestep], v, out);
        if (!MULTI_BLOCK_FLAG)
        {
            out = fma(logits_smem[tlength], v, out);
        }
        else
        { // MULTI_BLOCK_FLAG // Not supported yet: multi-block mode with FP8_MHA
            out = fma(logits_current_smem[0], v, out);
        }
#endif // MMHA_USE_FP32_ACUM_FOR_LOGITS
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
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
            convert_from_float(reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]), out);
#else
            *reinterpret_cast<V_vec_k*>(&out_smem[(vo - midpoint) * Dh + vi]) = out;
#endif
        }
        __syncthreads();

        // The bottom warps update their values.
        if (vo < midpoint && (Dh == Dh_MAX || vi < Dh))
        {
            out = add(*reinterpret_cast<const V_vec_k*>(&out_smem[vo * Dh + vi]), out);
        }
        __syncthreads();
    }

    const auto bhi = tensorrt_llm::common::flat_index2(bi, hi, num_heads);
    const auto bhi_seq_len_tile = bhi * params.max_seq_len_tile;
    // Output the final values.
    if (vo == 0 && (Dh == Dh_MAX || vi < Dh))
    {
        const auto bhvi = tensorrt_llm::common::flat_index2(bhi, vi, Dh);
#ifdef MMHA_USE_FP32_ACUM_FOR_OUT
        if (write_attention_quant)
        {
            using Packed_Int8_t = typename packed_type<int8_t, num_elems<V_vec_acum>::value>::type;
            out = mul<V_vec_acum, float>(*params.attention_out_scale_orig_quant, out);
            *reinterpret_cast<Packed_Int8_t*>(&(reinterpret_cast<int8_t*>(params.out)[bhvi])) = cast_to_int8(out);
        }
        else
        {
            if (!MULTI_BLOCK_FLAG)
            {
                // This makes sure we have coalesced memory access.
                V_vec_k final_out;
                convert_from_float(&final_out, out);
                *reinterpret_cast<V_vec_k*>(&params.out[bhvi]) = final_out;
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
        }
#else  // MMHA_USE_FP32_ACUM_FOR_OUT
        *reinterpret_cast<V_vec_acum*>(&params.out[bhvi]) = out;
#endif // MMHA_USE_FP32_ACUM_FOR_OUT
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
            if (tidx < gridDim.z)
                thread_partial_max = params.partial_max[bhi_seq_len_tile + tidx];
            // final_max = fmaxf(final_max, thread_partial_max);

            // Make sure we can start writing to shared memory.
            __syncthreads();

            // Specialize BlockReduce for a 1D block of THREADS_PER_BLOCK threads of type int
            typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
            // Allocate shared memory for BlockReduce
            __shared__ typename BlockReduce::TempStorage temp_storage;
            // Obtain a segment of consecutive items that are blocked across threads (final_max from above)
            // Compute the block-wide max for thread0
            final_max = BlockReduce(temp_storage).Reduce(thread_partial_max, cub::Max(), gridDim.z);

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
                const auto thread_partial_sum = params.partial_sum[bhi_seq_len_tile + tidx];
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

            // Number of threads to utilize: THREADS_PER_VALUE * gridDim.z (THREADS_PER_VALUE for vectorized output
            // and gridDim.z for all the partial outputs)
            int threads_boundary = THREADS_PER_VALUE * gridDim.z; // should be smaller than THREADS_PER_BLOCK
            assert(threads_boundary <= THREADS_PER_BLOCK);

            const auto o_idx = chunk_index<T, V_vec_k, THREADS_PER_VALUE>(tidx);
            // The partial output region this thread takes care of
            const auto oo = o_idx.x;
            // The hidden dimensions computed by this particular thread. (refer to vi)
            const auto oi = o_idx.y;

            // Load partial output
            int thread_partial_out_offset = oo * params.batch_size * num_heads * params.hidden_size_per_head;
            // Load partial max (different to thread_partial_max since the threadIdx rule changes here)
            float thread_partial_max_for_out = params.partial_max[bhi_seq_len_tile + oo];

            // Load the partial outputs.
            V_vec_k thread_partial_out
                = *reinterpret_cast<const V_vec_k*>(&params.partial_out[thread_partial_out_offset + bhi * Dh + oi]);

            if (tidx >= threads_boundary)
            {
                zero(thread_partial_out);
            }

            Tk factor_compute;
            convert_from_float(&factor_compute, __expf(thread_partial_max_for_out - final_max));

            thread_partial_out = mul<V_vec_k, Tk, V_vec_k>(factor_compute, thread_partial_out);

            // Make sure we can start writing to shared memory.
            __syncthreads();

            // The reduction iteration should start with a number which is a power of 2
            const auto reduction_iteration = static_cast<int>(cuda::std::bit_ceil(gridDim.z));

            // Run the final reduction amongst the different groups computing different partial outputs.
#pragma unroll
            for (int active_groups = reduction_iteration; active_groups >= 2; active_groups /= 2)
            {

                // The midpoint in the number of active groups.
                int midpoint = active_groups / 2;

                // The upper part of active threads store to shared memory.
                if (oo >= midpoint && oo < active_groups && (Dh == Dh_MAX || oi < Dh))
                {
                    *reinterpret_cast<V_vec_k*>(&out_oi_smem[(oo - midpoint) * Dh + oi]) = thread_partial_out;
                }
                __syncthreads();

                // The bottom warps update their values.
                if (oo < midpoint && (Dh == Dh_MAX || oi < Dh))
                {
                    thread_partial_out
                        = add(thread_partial_out, *reinterpret_cast<const V_vec_k*>(&out_oi_smem[oo * Dh + oi]));
                }
                __syncthreads();
            }

            ////////////////////
            // Final output O * inv_sum
            ////////////////////

            if (oo == 0 && (Dh == Dh_MAX || oi < Dh))
            {
                const auto inv_sum = __fdividef(1.f, final_sum + 1.e-6f);

                Tk inv_sum_compute;
                convert_from_float(&inv_sum_compute, inv_sum);

                thread_partial_out = mul<V_vec_k, Tk, V_vec_k>(inv_sum_compute, thread_partial_out);

                *reinterpret_cast<V_vec_k*>(&params.out[bhi * Dh + oi]) = thread_partial_out;
            }

            // Reset qk_current_smem and block_counter for the next timestep
            if (tidx == 0)
            {
                params.block_counter[bhi] = 0;
            }
        }
    }
#endif // ENABLE_MULTI_BLOCK_OPTION
}

} // namespace mmha

} // namespace kernels
} // namespace tensorrt_llm
