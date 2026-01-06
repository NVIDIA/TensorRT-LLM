// Adapted from https://github.com/nunchaku-tech/nunchaku
// @article{
//   li2024svdquant,
//   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
//   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
//   journal={arXiv preprint arXiv:2411.05007},
//   year={2024}
// }

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include "common.h"
#include "utils.cuh"

namespace nunchaku::kernels {

static constexpr int clamp(int val, int min, int max) {
    if (val < min)
        return min;
    if (val > max)
        return max;
    return val;
}

template<bool shmem = false, typename T>
__device__ __forceinline__ static T load(const T *addr) {
    if constexpr (shmem) {
        if constexpr (sizeof(T) == 8) {
            uint2 data;
            asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];"
                         : "=r"(data.x), "=r"(data.y)
                         : "l"(__cvta_generic_to_shared(addr)));
            return *reinterpret_cast<T *>(&data);
        }
        if constexpr (sizeof(T) == 16) {
            uint4 data;
            asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                         : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                         : "l"(__cvta_generic_to_shared(addr)));
            return *reinterpret_cast<T *>(&data);
        }
        return *addr;
    }

    if constexpr (sizeof(T) == 8) {
        uint2 data = __ldg(reinterpret_cast<const uint2 *>(addr));
        return *reinterpret_cast<T *>(&data);
    }
    if constexpr (sizeof(T) == 16) {
        uint4 data = __ldg(reinterpret_cast<const uint4 *>(addr));
        return *reinterpret_cast<T *>(&data);
    }

    return *addr;
}

template<typename T>
__device__ __forceinline__ static T load_pred(const T *addr, bool pred) {
    if constexpr (sizeof(T) == 4) {
        uint32_t data;
        asm volatile("{ .reg .pred loadpred; setp.ne.b32 loadpred, %2, 0;"
                     "@loadpred ld.global.nc.b32 %0, [%1];"
                     "}"
                     : "=r"(data)
                     : "l"(addr), "r"((int)pred));
        return *reinterpret_cast<T *>(&data);
    }
    if constexpr (sizeof(T) == 8) {
        uint2 data;
        asm volatile("{ .reg .pred loadpred; setp.ne.b32 loadpred, %3, 0;"
                     "@loadpred ld.global.nc.v2.b32 {%0, %1}, [%2];"
                     "}"
                     : "=r"(data.x), "=r"(data.y)
                     : "l"(addr), "r"((int)pred));
        return *reinterpret_cast<T *>(&data);
    }
    if constexpr (sizeof(T) == 16) {
        uint4 data;
        asm volatile("{ .reg .pred loadpred; setp.ne.b32 loadpred, %5, 0;"
                     "@loadpred ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                     "}"
                     : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                     : "l"(addr), "r"((int)pred));
        return *reinterpret_cast<T *>(&data);
    }

    T result;
    if (pred) {
        result = *addr;
    }
    return result;
}

template<bool shmem = false, typename T>
__device__ __forceinline__ static void store(T *addr, T val) {
    if constexpr (shmem) {
        if constexpr (sizeof(T) == 8) {
            uint2 data = *reinterpret_cast<uint2 *>(&val);
            asm volatile(
                "st.shared.v2.b32 [%0], {%1, %2};" ::"l"(__cvta_generic_to_shared(addr)), "r"(data.x), "r"(data.y));
            return;
        }
        if constexpr (sizeof(T) == 16) {
            uint4 data = *reinterpret_cast<uint4 *>(&val);
            asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(__cvta_generic_to_shared(addr)),
                         "r"(data.x),
                         "r"(data.y),
                         "r"(data.z),
                         "r"(data.w));
            return;
        }
        *addr = val;
        return;
    }

    if constexpr (sizeof(T) == 4) {
        __stcg(reinterpret_cast<unsigned int *>(addr), *reinterpret_cast<unsigned int *>(&val));
        return;
    }
    if constexpr (sizeof(T) == 8) {
        __stcg(reinterpret_cast<uint2 *>(addr), *reinterpret_cast<uint2 *>(&val));
        return;
    }
    if constexpr (sizeof(T) == 16) {
        __stcg(reinterpret_cast<uint4 *>(addr), *reinterpret_cast<uint4 *>(&val));
        return;
    }
    *addr = val;
}

template<typename T>
__device__ __forceinline__ static void store_pred(T *addr, T val, bool pred) {
    if constexpr (sizeof(T) == 4) {
        uint32_t data = *reinterpret_cast<uint32_t *>(&val);
        asm volatile("{ .reg .pred storepred; setp.ne.b32 storepred, %0, 0;"
                     "@storepred st.global.cg.b32 [%1], %2;"
                     "}" ::"r"((int)pred),
                     "l"(addr),
                     "r"(data));
        return;
    }
    if constexpr (sizeof(T) == 8) {
        uint2 data = *reinterpret_cast<uint2 *>(&val);
        asm volatile("{ .reg .pred storepred; setp.ne.b32 storepred, %0, 0;"
                     "@storepred st.global.cg.v2.b32 [%1], {%2, %3};"
                     "}" ::"r"((int)pred),
                     "l"(addr),
                     "r"(data.x),
                     "r"(data.y));
        return;
    }
    if constexpr (sizeof(T) == 16) {
        uint4 data = *reinterpret_cast<uint4 *>(&val);
        asm volatile("{ .reg .pred storepred; setp.ne.b32 storepred, %0, 0;"
                     "@storepred st.global.cg.v4.b32 [%1], {%2, %3, %4, %5};"
                     "}" ::"r"((int)pred),
                     "l"(addr),
                     "r"(data.x),
                     "r"(data.y),
                     "r"(data.z),
                     "r"(data.w));
        return;
    }

    if (pred) {
        *addr = val;
    }
}

__device__ __forceinline__ static float2 half22float2(half2 val) {
    return __half22float2(val);
}

__device__ __forceinline__ static float2 half22float2(__nv_bfloat162 val) {
    return __bfloat1622float2(val);
}

template<typename T>
__device__ __forceinline__ static T float22half2(float2 val) = delete;

template<>
__device__ __forceinline__ half2 float22half2<half2>(float2 val) {
    return __float22half2_rn(val);
}

template<>
__device__ __forceinline__ __nv_bfloat162 float22half2<__nv_bfloat162>(float2 val) {
    return __float22bfloat162_rn(val);
}

template<typename T>
__device__ __forceinline__ static void unused_var(T &val, bool alwaysfalse) {
    volatile T *ptr = nullptr;
    if (alwaysfalse) {
        *ptr = val;
    }
}

__device__ __forceinline__ static void ldmatrix(const void *ptr, uint4 &out) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
                 : "l"(__cvta_generic_to_shared(ptr)));
}

template<typename T>
__device__ __forceinline__ static T movmatrix(T x) {
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
                 : "=r"(*reinterpret_cast<uint32_t *>(&x))
                 : "r"(*reinterpret_cast<uint32_t *>(&x)));
    return x;
}

// x in low bit, y in high bit
template<int bitwidth, bool use_unsigned>
__device__ __forceinline__ uint32_t quantize_float2(float2 value) = delete;

template<>
__device__ __forceinline__ uint32_t quantize_float2<4, false>(float2 value) {
    int v1, v2;
    uint32_t result;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v1) : "f"(value.x));
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v2) : "f"(value.y));
    asm volatile("cvt.pack.sat.s4.s32.b32 %0, %1, %2, 0;" : "=r"(result) : "r"(v2), "r"(v1));
    return result;
}

template<>
__device__ __forceinline__ uint32_t quantize_float2<4, true>(float2 value) {
    int v1, v2;
    uint32_t result;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v1) : "f"(value.x));
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v2) : "f"(value.y));
    asm volatile("cvt.pack.sat.u4.s32.b32 %0, %1, %2, 0;" : "=r"(result) : "r"(v2), "r"(v1));
    return result;
}

template<>
__device__ __forceinline__ uint32_t quantize_float2<8, false>(float2 value) {
    int v1, v2;
    uint32_t result;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v1) : "f"(value.x));
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v2) : "f"(value.y));
    asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;" : "=r"(result) : "r"(v2), "r"(v1));
    return result;
}

__device__ __forceinline__ uint32_t quantize_float2_fp4(float2 value) {
    uint32_t result;
    asm volatile("{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; cvt.u32.u8 %0, tmp; }"
                 : "=r"(result)
                 : "f"(value.y), "f"(value.x));
    return result;
}

__device__ __forceinline__ uint32_t quantize_float4_fp8(float4 value) {
    uint16_t lo, hi;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(value.y), "f"(value.x));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(value.w), "f"(value.z));
    return uint32_t(lo) | (uint32_t(hi) << 16);
}

__device__ __forceinline__ static float cuda_tanhf(float x) {
    float result;
    asm("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ static float cuda_frcp(float x) {
    float result;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ static float cuda_frsqrt(float x) {
    float result;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ static float cuda_sin(float x) {
    float result;
    asm("sin.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ static float cuda_cos(float x) {
    float result;
    asm("cos.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ static float cuda_exp2(float x) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// https://forums.developer.nvidia.com/t/hardware-accelerated-computation-of-the-sigmoid-logistic-function/266206
__forceinline__ __device__ static float cuda_sigmoidf(float a) {
#if USE_TANH
    return fmaf(0.5, __tanhf(0.5f * a), 0.5f);
#else  // USE_TANH
    const float L2E = 1.442695041f; // log2(exp(1))
    float t, d, e, r;
    t = -L2E * a;
    asm("ex2.approx.ftz.f32 %0,%1;\n\t" : "=f"(e) : "f"(t));
    d = e + 1.0f;
    asm("rcp.approx.ftz.f32 %0,%1;\n\t" : "=f"(r) : "f"(d));
    return r;
#endif // USE_TANH
}

template<typename T>
__device__ __forceinline__ static T gelu_half2(T x) {
    float2 xf  = half22float2(x);
    float2 x3f = xf * xf * xf;
    float t1   = 0.5f + 0.5f * cuda_tanhf(0.79788456f * (xf.x + (0.044715f * x3f.x)));
    float t2   = 0.5f + 0.5f * cuda_tanhf(0.79788456f * (xf.y + (0.044715f * x3f.y)));
    return float22half2<T>(xf * make_float2(t1, t2));
}

template<typename T>
__device__ __forceinline__ static T gelu_half(T x) {
    float xf  = float(x);
    float x3f = xf * xf * xf;
    float t   = 0.5f + 0.5f * cuda_tanhf(0.79788456f * (xf + (0.044715f * x3f)));
    return (T)(xf * t);
}

template<typename T>
__device__ __forceinline__ static T silu(const T &x) {
    // x * sigmoid(x)
    return (T)((float)x * cuda_sigmoidf((float)x));
    // return (T)__fdividef((float)x, 1.0f + __expf((float)-x));
}

__device__ __forceinline__ static half2 h2div(half2 a, half2 b) {
    float2 af = half22float2(a);
    float2 bf = half22float2(b);
    float2 of;
    of.x = __fdividef(af.x, bf.x);
    of.y = __fdividef(af.y, bf.y);
    return float22half2<half2>(of);
};
__device__ __forceinline__ static __nv_bfloat162 h2div(__nv_bfloat162 a, __nv_bfloat162 b) {
    float2 af = half22float2(a);
    float2 bf = half22float2(b);
    float2 of;
    of.x = __fdividef(af.x, bf.x);
    of.y = __fdividef(af.y, bf.y);
    return float22half2<__nv_bfloat162>(of);
};

__device__ __forceinline__ static void reduce_add(float *addr, float val) {
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" ::"l"(addr), "f"(val));
}

__device__ __forceinline__ static void reduce_add_pred(float *addr, float val, bool pred) {
    asm volatile("{ .reg .pred storepred; setp.ne.b32 storepred, %0, 0;"
                 "@storepred red.relaxed.gpu.global.add.f32 [%1], %2;"
                 "}" ::"r"((int)pred),
                 "l"(addr),
                 "f"(val));
}

template<int cnt, typename F>
__device__ __forceinline__ static void unrolled_loop(F &&lambda) {
    auto call = [&]<int... Is>(std::integer_sequence<int, Is...>) { (lambda.template operator()<Is>(), ...); };
    call(std::make_integer_sequence<int, cnt>());
}

// int2float is slow on sm_80 and before
// val in [-4194304, 4194303]
__device__ __forceinline__ static float int2float_fast(int val) {
    float fval;
    // fval = (val & 0x7FFFFF) ^ 0x4B400000
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=f"(fval)
                 : "r"(val), "n"(0x7FFFFF), "n"(0x4B400000), "n"((0xF0 & 0xCC) ^ 0xAA));
    return fval - 12582912.0f;
}

template<typename To, typename From>
__device__ __forceinline__ static To bit_cast(const From &input) {
    static_assert(sizeof(To) == sizeof(From));
    // not safe but anyway
    return *reinterpret_cast<const To *>(&input);
}

// both int2float and float2half are slow on sm_75 and before
// val in [-8192, 8191], steps of 16, round to negative inf
__device__ __forceinline__ static half2 int2half2_fast_8192(int x, int y) {
    uint32_t ival;
    uint32_t hval;
    // ival.lo = x.lo; ival.hi = y.lo;
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(ival) : "r"(x), "r"(y), "n"(0x5410));
    ival = ival >> 4;
    // (val & 0x03FF03FF) ^ 0x76007600
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(hval)
                 : "r"(ival), "n"(0x03FF03FF), "n"(0x76007600), "n"((0xF0 & 0xCC) ^ 0xAA));
    return __hadd2(kernels::bit_cast<half2>(hval), half2(-24576.0f, -24576.0f));
}
// val in [-4096, 4095], steps of 8, round to nearest
__device__ __forceinline__ static half2 int2half2_fast_4096_rn(int x, int y) {
    // x = max(min(x, 4095), -4096);
    // y = max(min(y, 4095), -4096);
    // TODO: round to even?
    x = x * 8192 + 32768;
    y = y * 8192 + 32768;
    uint32_t ival;
    uint32_t hval;
    // ival.lo = x.hi; ival.hi = y.hi;
    // <=> divide x and y by 65536 and pack them
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(ival) : "r"(x), "r"(y), "n"(0x7632));
    // (val & 0x03FF03FF) ^ 0x72007200
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(hval)
                 : "r"(ival), "n"(0x03FF03FF), "n"(0x72007200), "n"((0xF0 & 0xCC) ^ 0xAA));
    return __hadd2(kernels::bit_cast<half2>(hval), half2(-12288.0f, -12288.0f));
}
// val in [-512, 511]
__device__ __forceinline__ static half2 int2half2_fast_512(int x, int y) {
    uint32_t ival;
    uint32_t hval;
    // ival.lo = x.lo; ival.hi = y.lo;
    // <=> divide x and y by 65536 and pack them
    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(ival) : "r"(x), "r"(y), "n"(0x5410));
    // (val & 0x03FF03FF) ^ 0x66006600
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(hval)
                 : "r"(ival), "n"(0x03FF03FF), "n"(0x66006600), "n"((0xF0 & 0xCC) ^ 0xAA));
    return __hadd2(kernels::bit_cast<half2>(hval), half2(-1536.0f, -1536.0f));
}

}; // namespace nunchaku::kernels
