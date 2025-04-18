#ifndef LLAMA4_UTILS_CUH
#define LLAMA4_UTILS_CUH

#include <cuda_fp8.h>

#define ENABLE_ACQBULK 1
#define ENABLE_PREFETCH 1
#define ENABLE_PREEXIT 0

__device__ __forceinline__ float2 ffma2(float2 x, float2 y, float2 acc) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    return __ffma2_rn(x, y, acc);
#else
    return make_float2(x.x * y.x + acc.x, x.y * y.y + acc.y);
#endif
}

struct __align__(8) aligned_fp8x8 {
    __align__(8) __nv_fp8x4_e4m3 data[2];
};

#endif
