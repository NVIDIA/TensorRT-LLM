#pragma once

#include <cuda_fp8.h>

namespace tensorrt_llm::kernels
{

namespace llama4_router_gemm
{
constexpr size_t GEMM_K = 5120;
constexpr size_t BLOCK_SIZE = 256;
constexpr size_t NUM_EXPERTS = 128;
constexpr size_t VEC_SIZE = 4;

} // namespace llama4_router_gemm

namespace llama4_qkv_gemm
{

constexpr size_t HIDDEN_IN = 5120;
constexpr size_t HIDDEN_OUT = 896;

constexpr size_t Q_HIDDEN_OUT = 640;
constexpr size_t FLOOR_SCALE = 8192.0;
constexpr size_t ATTN_SCALE = 0.1;

constexpr size_t BLOCK_SIZE = 128;
constexpr size_t WARP_SIZE = 32;
constexpr size_t WARP_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
// Use 8 for now, which results in LDG.64.
constexpr size_t VEC_SIZE = 8;

constexpr size_t ENABLE_ACQBULK = 1;
constexpr size_t ENABLE_PREFETCH = 1;
constexpr size_t ENABLE_PREEXIT = 0;

} // namespace llama4_qkv_gemm

namespace llama4_fc_swiglu
{

constexpr size_t HIDDEN_IN = 5120;

constexpr size_t BLOCK_SIZE = 128;
constexpr size_t WARP_SIZE = 32;

constexpr size_t ENABLE_ACQBULK = 1;
constexpr size_t ENABLE_PREFETCH = 1;
constexpr size_t ENABLE_PREEXIT = 1;

// Use 8 for now, which results in LDG.64.
constexpr size_t VEC_SIZE = 8;

} // namespace llama4_fc_swiglu

__device__ __forceinline__ float2 ffma2(float2 x, float2 y, float2 acc)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
    return __ffma2_rn(x, y, acc);
#else
    return make_float2(x.x * y.x + acc.x, x.y * y.y + acc.y);
#endif
}

struct __align__(8) aligned_fp8x8
{
    __align__(8) __nv_fp8x4_e4m3 data[2];
};

} // namespace tensorrt_llm::kernels
