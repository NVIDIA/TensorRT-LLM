/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace tensorrt_llm::runtime::ub
{
#define ENABLE_FP8 1
#define ENABLE_BF16 1

template <typename T>
struct packed_type;

template <>
struct packed_type<float>
{
    using type = float;
}; // we don't need to pack float by default

template <>
struct packed_type<half>
{
    using type = half2;
};

#ifdef ENABLE_BF16
template <>
struct packed_type<__nv_bfloat16>
{
    using type = __nv_bfloat162;
};

inline __device__ float2 bf1622float2(const __nv_bfloat162 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float2 f_val;
    f_val.x = __low2float(val);
    f_val.y = __high2float(val);
    return f_val;
#else
    return __bfloat1622float2(val);
#endif
}

inline __device__ int16_t bf1622int16(__nv_bfloat162 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    float2 f_val;
    f_val.x = max(min(__low2float(val), 127.f), -128.f);
    f_val.y = max(min(__high2float(val), 127.f), -128.f);

    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = static_cast<int8_t>(static_cast<short>(f_val.x));
    int8[1] = static_cast<int8_t>(static_cast<short>(f_val.y));
    return int16;
#else
    val = __hmin2(val, make_bfloat162(127., 127.));
    val = __hmax2(val, make_bfloat162(-128., -128.));

    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = static_cast<int8_t>(static_cast<short>(val.x));
    int8[1] = static_cast<int8_t>(static_cast<short>(val.y));
    return int16;
#endif
}

inline __device__ __nv_bfloat162 float22bf162(const float2 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __floats2bfloat162_rn(val.x, val.y);
#else
    return __float22bfloat162_rn(val);
#endif
}

inline __device__ __nv_bfloat162 bf162bf162(const __nv_bfloat16 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __nv_bfloat162 val2;
    val2.x = val;
    val2.y = val;
    return val2;
#else
    return __bfloat162bfloat162(val);
#endif
}

#endif

#ifdef ENABLE_FP8
template <>
struct packed_type<__nv_fp8_e4m3>
{
    using type = __nv_fp8x2_e4m3;
};

__inline__ __device__ __nv_bfloat162 fp8x2_e4m3_to_bfloat2(__nv_fp8x2_e4m3 const* in)
{
    const char2 tmp_val = reinterpret_cast<char2 const*>(in)[0];
    __nv_bfloat162 out = __nv_bfloat162((float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.x)[0],
        (float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.y)[0]);
    return out;
}

__inline__ __device__ half2 fp8x2_e4m3_to_half2(__nv_fp8x2_e4m3 const* in)
{
    const char2 tmp_val = reinterpret_cast<char2 const*>(in)[0];
    half2 out = half2((float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.x)[0],
        (float) reinterpret_cast<__nv_fp8_e4m3 const*>(&tmp_val.y)[0]);
    return out;
}

#endif

template <typename T>
struct num_elems;

template <>
struct num_elems<float>
{
    static constexpr int value = 1;
};

template <>
struct num_elems<float2>
{
    static constexpr int value = 2;
};

template <>
struct num_elems<float4>
{
    static constexpr int value = 4;
};

template <>
struct num_elems<half>
{
    static constexpr int value = 1;
};

template <>
struct num_elems<half2>
{
    static constexpr int value = 2;
};
#ifdef ENABLE_BF16
template <>
struct num_elems<__nv_bfloat16>
{
    static constexpr int value = 1;
};

template <>
struct num_elems<__nv_bfloat162>
{
    static constexpr int value = 2;
};
#endif
#ifdef ENABLE_FP8
template <>
struct num_elems<__nv_fp8_e4m3>
{
    static constexpr int value = 1;
};

template <>
struct num_elems<__nv_fp8x2_e4m3>
{
    static constexpr int value = 2;
};
#endif

template <typename T, int num>
struct packed_as;

template <typename T>
struct packed_as<T, 1>
{
    using type = T;
};

template <>
struct packed_as<half, 2>
{
    using type = half2;
};

template <>
struct packed_as<float, 2>
{
    using type = float2;
};

template <>
struct packed_as<int8_t, 2>
{
    using type = int16_t;
};

template <>
struct packed_as<int32_t, 2>
{
    using type = int2;
};

template <>
struct packed_as<half2, 1>
{
    using type = half;
};

template <>
struct packed_as<float2, 1>
{
    using type = float;
};
#ifdef ENABLE_BF16
template <>
struct packed_as<__nv_bfloat16, 2>
{
    using type = __nv_bfloat162;
};

template <>
struct packed_as<__nv_bfloat162, 1>
{
    using type = __nv_bfloat16;
};
#endif
#ifdef ENABLE_FP8
template <>
struct packed_as<__nv_fp8_e4m3, 2>
{
    using type = __nv_fp8x2_e4m3;
};

template <>
struct packed_as<__nv_fp8x2_e4m3, 1>
{
    using type = __nv_fp8_e4m3;
};

template <>
struct packed_as<__nv_fp8_e5m2, 2>
{
    using type = __nv_fp8x2_e5m2;
};

template <>
struct packed_as<__nv_fp8x2_e5m2, 1>
{
    using type = __nv_fp8_e5m2;
};
#endif

inline __device__ float2 operator*(float2 a, float2 b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}

inline __device__ float2 operator+(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float2 operator*(float2 a, float b)
{
    return make_float2(a.x * b, a.y * b);
}

inline __device__ float2 operator+(float2 a, float b)
{
    return make_float2(a.x + b, a.y + b);
}

inline __device__ float2 operator-(float2 a, float b)
{
    return make_float2(a.x - b, a.y - b);
}

template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val)
{
    return val;
}

template <>
__device__ inline float2 cuda_cast<float2, int2>(int2 val)
{
    return make_float2(val.x, val.y);
}

template <>
__device__ inline float2 cuda_cast<float2, float>(float val)
{
    return make_float2(val, val);
}

template <>
__device__ inline float2 cuda_cast<float2, half2>(half2 val)
{
    return __half22float2(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float2>(float2 val)
{
    return __float22half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, float>(float val)
{
    return __float2half2_rn(val);
}

template <>
__device__ inline half2 cuda_cast<half2, half>(half val)
{
    return __half2half2(val);
}

template <>
__device__ inline int8_t cuda_cast<int8_t, half>(half val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    union
    {
        half fp16;
        int16_t int16_in;
    };

    fp16 = val;
    asm volatile("cvt.rni.sat.s8.f16 %0, %1;" : "=h"(int16) : "h"(int16_in));
    return int8[0];
}

template <>
__device__ inline int16_t cuda_cast<int16_t, half2>(half2 val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = cuda_cast<int8_t>(val.x);
    int8[1] = cuda_cast<int8_t>(val.y);
    return int16;
}

template <>
__device__ inline int8_t cuda_cast<int8_t, float>(float val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=h"(int16) : "f"(val));
    return int8[0];
}

template <>
__device__ inline int16_t cuda_cast<int16_t, float2>(float2 val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int8[0] = cuda_cast<int8_t>(val.x);
    int8[1] = cuda_cast<int8_t>(val.y);
    return int16;
}

template <>
__device__ inline half2 cuda_cast<half2, int16_t>(int16_t val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int16 = val;
    return make_half2(int8[0], int8[1]);
}

template <>
__device__ inline float2 cuda_cast<float2, int16_t>(int16_t val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int16 = val;
    return make_float2(int8[0], int8[1]);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat16 cuda_cast(int32_t val)
{
    return static_cast<float>(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast(int8_t val)
{
    return static_cast<float>(val);
}

template <>
__device__ inline int8_t cuda_cast(__nv_bfloat16 val)
{
    return static_cast<float>(val);
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val)
{
    return __bfloat162float(val);
}

template <>
__device__ inline float2 cuda_cast<float2, __nv_bfloat162>(__nv_bfloat162 val)
{
    return bf1622float2(val);
}

template <>
__device__ inline half cuda_cast<half, __nv_bfloat16>(__nv_bfloat16 val)
{
    return __float2half(__bfloat162float(val));
}

template <>
__device__ inline int16_t cuda_cast<int16_t, __nv_bfloat162>(__nv_bfloat162 val)
{
    return bf1622int16(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, float>(float val)
{
    return __float2bfloat16(val);
}

template <>
__device__ inline __nv_bfloat16 cuda_cast<__nv_bfloat16, half>(half val)
{
    return __float2bfloat16(__half2float(val));
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_bfloat16>(__nv_bfloat16 val)
{
    return bf162bf162(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float>(float val)
{
    return __float2bfloat162_rn(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, float2>(float2 val)
{
    return float22bf162(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, int16_t>(int16_t val)
{
    union
    {
        int8_t int8[2];
        int16_t int16;
    };

    int16 = val;
    __nv_bfloat162 res;
    res.x = cuda_cast<__nv_bfloat16>(int8[0]);
    res.y = cuda_cast<__nv_bfloat16>(int8[1]);
    return res;
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, half2>(half2 val)
{
    return float22bf162(__half22float2(val));
}

#endif // ENABLE BF16

#ifdef ENABLE_FP8
template <>
__device__ inline float2 cuda_cast<float2, __nv_fp8x2_e4m3>(__nv_fp8x2_e4m3 val)
{
    return bf1622float2(fp8x2_e4m3_to_bfloat2(&val));
}

template <>
__device__ inline half2 cuda_cast<half2, __nv_fp8x2_e4m3>(__nv_fp8x2_e4m3 val)
{
    return fp8x2_e4m3_to_half2(&val);
}

template <>
__device__ inline __nv_fp8x2_e4m3 cuda_cast<__nv_fp8x2_e4m3, float2>(float2 val)
{
    return __nv_fp8x2_e4m3(bf1622float2(float22bf162(val)));
}

template <>
__device__ inline __nv_fp8x2_e4m3 cuda_cast<__nv_fp8x2_e4m3, half2>(half2 val)
{
    return __nv_fp8x2_e4m3(cuda_cast<float2>(val));
}

template <>
__device__ inline __nv_fp8x2_e4m3 cuda_cast<__nv_fp8x2_e4m3, __nv_bfloat162>(__nv_bfloat162 val)
{
    return __nv_fp8x2_e4m3(cuda_cast<float2>(val));
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, half>(half val)
{
    return __nv_fp8_e4m3(val);
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, __nv_bfloat16>(__nv_bfloat16 val)
{
    return __nv_fp8_e4m3(val);
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, float>(float val)
{
    return __nv_fp8_e4m3(val);
}

template <>
__device__ inline float cuda_cast<float, __nv_fp8_e4m3>(__nv_fp8_e4m3 val)
{
    return (float) val;
}

template <>
__device__ inline __nv_bfloat162 cuda_cast<__nv_bfloat162, __nv_fp8x2_e4m3>(__nv_fp8x2_e4m3 val)
{
    return fp8x2_e4m3_to_bfloat2(&val);
}

template <>
__device__ inline int8_t cuda_cast<int8_t, __nv_fp8_e4m3>(__nv_fp8_e4m3 val)
{
    // no impl
    return 0;
}

template <>
__device__ inline __nv_fp8_e4m3 cuda_cast<__nv_fp8_e4m3, int8_t>(int8_t val)
{
    return cuda_cast<__nv_fp8_e4m3>(cuda_cast<__nv_bfloat16>(cuda_cast<float>(val)));
}

#endif // ENABLE_FP8

#define FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T) (0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val)
{
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = is_mask ? shared[i][lane] : (T) (0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T) 0.0f;
}

static bool const kDISABLE_FP32_ACCUMULATION = getenv("TRTLLM_UB_AR_DISABLE_FP32_ACCUMULATION") != nullptr;

} // namespace tensorrt_llm::runtime::ub
