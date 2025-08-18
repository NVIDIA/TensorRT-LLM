/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cfloat>
#include <cstdio>
#include <fmha/numeric_types.h>
#include <fmha/utils.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

// The number of threads per warp.
enum
{
    THREADS_PER_WARP = 32
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dst_type, typename Src_type>
struct Softmax_params
{
    // Output pointer.
    Dst_type* dst;
    // Source pointer.
    Src_type const* src;
    // Masks.
    int8_t const* mask;
    // Attention sinks (per head).
    float const* attention_sinks;
    // Softmax sum pointer.
    float* softmax_sum;
    // ALiBi
    bool has_alibi;
    // Dimensions of the problem.
    size_t b, h;
    // Precomputed constants.
    size_t bhs, hs, bs;
    // The scaling factors to apply when we convert to/from float.
    float scale_bmm1, softcapping_scale_bmm1, scale_softmax;
    // The number of reduction warps used by the fused kernel.
    int warps_n;
    int* cu_q_seqlens;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float to_float(uint16_t const& src, float)
{
    return fmha::half_to_float(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Disable warning #177-D because this function has not been used elsewhere
#pragma nv_diag_suppress 177

static inline __device__ float to_float(fmha::bf16_t const& src, float)
{
    return __bfloat162float(src);
}

#pragma nv_diag_default 177

////////////////////////////////////////////////////////////////////////////////////////////////////

// Disable warning #177-D because this function has not been used elsewhere
#pragma nv_diag_suppress 177

static inline __device__ float to_float(fmha::e4m3_t const& src, float)
{
    return float(src);
}

#pragma nv_diag_default 177

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float to_float(float const& src, float)
{
    return src;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float to_float(int const& src, float scale)
{
    float dst;

    // Convert from int to float.
    dst = static_cast<float>(src);

    // Scale.
    dst *= scale;

    return dst;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void from_float(uint16_t& dst, float const& src, float)
{
    dst = fmha::float_to_half(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void from_float(fmha::bf16_t& dst, float const& src, float)
{
    dst = fmha::float_to_bf16(src);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ int8_t float_to_int8_rn(float x)
{
    uint32_t dst;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
    return reinterpret_cast<int8_t const&>(dst);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void from_float(int8_t& dst, float const& src, float scale)
{
    dst = float_to_int8_rn(src * scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void from_float(fmha::e4m3_t& dst, float const& src, float scale)
{
    dst = fmha::e4m3_t(src * scale);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float apply_exp_(float x, float max)
{
    return isinf(x) ? 0.f : __expf(x - max);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
static inline __device__ void reduce(float (&data_fp32)[N][1], int8_t const (&mask)[N][1], int warps_n, float& sum_fp32,
    float& max_fp32, float const attention_sink)
{

// Apply the masks.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        data_fp32[ii][0] = mask[ii][0] ? data_fp32[ii][0] : -HUGE_VALF;
    }

    // Compute the max inside the thread.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        max_fp32 = fmaxf(max_fp32, data_fp32[ii][0]);
    }

// Compute inside the warp.
#pragma unroll
    for (int xor_mask = THREADS_PER_WARP / 2; xor_mask > 0; xor_mask /= 2)
    {
        max_fp32 = fmaxf(max_fp32, __shfl_xor_sync(uint32_t(-1), max_fp32, xor_mask));
    }

// Transform the elements.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        data_fp32[ii][0] = apply_exp_(data_fp32[ii][0], max_fp32);
    }

    // Compute the max inside the thread.
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)

#pragma unroll
    for (int ii = 0; ii < N; ii++)
    {
        sum_fp32 += data_fp32[ii][0]; //+0    +64    +128
    }

    // Emulate tmp[0] + tmp[1]
    sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 4);
    __syncwarp();

    // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
    sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 1);
    __syncwarp();
    // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
    sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 2);
    __syncwarp();

    // Emulate final reduction
    sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 8);
    __syncwarp();
    sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 16);
    __syncwarp();

#else
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        sum_fp32 += data_fp32[ii][0];
    }

// Compute inside the warp.
#pragma unroll
    for (int xor_mask = THREADS_PER_WARP / 2; xor_mask > 0; xor_mask /= 2)
    {
        sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, xor_mask);
    }
#endif

    // // DEBUG.
    // if( blockIdx.z == 1 && threadIdx.y == 0 && threadIdx.x == 5 ) {
    //   printf("elt=%12.8f sum_fp32=%12.8f\n", data_fp32[0].x, sum_fp32);
    // }

    // Fix the sum if needed.
    if (sum_fp32 == 0.f || sum_fp32 != sum_fp32)
    {
        sum_fp32 = 1.f;
    }

    // Normalize.
    float inv_sum_fp32 = 1.f / (sum_fp32 + expf(attention_sink - max_fp32));
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        data_fp32[ii][0] *= inv_sum_fp32;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
static inline __device__ void reduce(float (&data_fp32)[N][2], int8_t const (&mask)[N][2], int warps_n, float& sum_fp32,
    float& max_fp32, float const attention_sink)
{
// Apply the masks.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        data_fp32[ii][0] = mask[ii][0] ? data_fp32[ii][0] : -HUGE_VALF;
        data_fp32[ii][1] = mask[ii][1] ? data_fp32[ii][1] : -HUGE_VALF;
    }

    // Compute the max inside the thread.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        max_fp32 = fmaxf(max_fp32, data_fp32[ii][0]);
        max_fp32 = fmaxf(max_fp32, data_fp32[ii][1]);
    }

// Compute inside the warp.
#pragma unroll
    for (int xor_mask = THREADS_PER_WARP / 2; xor_mask > 0; xor_mask /= 2)
    {
        max_fp32 = fmaxf(max_fp32, __shfl_xor_sync(uint32_t(-1), max_fp32, xor_mask));
    }

// // DEBUG.
// if( blockIdx.z == 1 && threadIdx.y == 0 && threadIdx.x == 5 ) {
//   printf("elt=%12.8f max_fp32=%12.8f\n", data_fp32[0][0], max_fp32);
// }
// // END OF DEBUG.

// Transform the elements.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        data_fp32[ii][0] = apply_exp_(data_fp32[ii][0], max_fp32);
        data_fp32[ii][1] = apply_exp_(data_fp32[ii][1], max_fp32);
    }

    // Compute the max inside the thread.
    // float sum_fp32 = 0.f;
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
    if (warps_n == 1)
    {
        // TODO not sure if we can improve this on the gmma side without using additional regs.

        // this is intentionally o(n) instead of o(log n)
        // lanes 0 and 1 here represent the first quad.

        // need to account for offset of l0 when addressing absolute lanes.
        int const ti = threadIdx.x % 4;
        float tmp = 0.f;

        for (int ni = 0; ni < N; ni++)
        {
            float x = data_fp32[ni][0] + data_fp32[ni][1];
            tmp += x;

            for (int it = 1; it < 8; it++)
            {
                tmp += __shfl_sync(uint32_t(-1), x, 4 * it + ti);
                __syncwarp();
            }
        }

        // emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
        tmp += __shfl_xor_sync(uint32_t(-1), tmp, 1);
        __syncwarp();
        // emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
        tmp += __shfl_xor_sync(uint32_t(-1), tmp, 2);
        __syncwarp();
        sum_fp32 = __shfl_sync(uint32_t(-1), tmp, 0);
    }
    else if (warps_n == 8)
    {
        // Accumulate warp 0 and warp 4
        float tmp[2] = {0.f, 0.f};
#pragma unroll
        for (int ii = 0; ii < N; ii += 2)
        {
            tmp[0] += data_fp32[ii + 0][0];
            tmp[0] += data_fp32[ii + 0][1];
            tmp[1] += data_fp32[ii + 1][0];
            tmp[1] += data_fp32[ii + 1][1];
        }

        // Emulate tmp[0] + tmp[1]
        tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 4);
        tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 4);

        // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
        tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 1);
        tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 1);
        // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
        tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 2);
        tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 2);
        // Emulate final reduction
        tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 8);
        tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 8);

        tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 16);
        tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 16);
        sum_fp32 = tmp[0] + tmp[1];

        sum_fp32 = __shfl_sync(uint32_t(-1), sum_fp32, 0);
    }
    else
    {

#pragma unroll
        for (int ii = 0; ii < N; ii++)
        {
            sum_fp32 += data_fp32[ii][0] + data_fp32[ii][1];
        }

        // Emulate tmp[0] + tmp[1]
        sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 4);
        // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
        sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 1);
        // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
        sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 2);
        // Emulate final reduction
        sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 8);
        sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, 16);

        sum_fp32 = __shfl_sync(uint32_t(-1), sum_fp32, 0);
    }

#else
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        sum_fp32 += data_fp32[ii][0];
        sum_fp32 += data_fp32[ii][1];
    }

// Compute inside the warp.
#pragma unroll
    for (int xor_mask = THREADS_PER_WARP / 2; xor_mask > 0; xor_mask /= 2)
    {
        sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, xor_mask);
    }
#endif

    // // DEBUG.
    // if( blockIdx.z == 1 && threadIdx.y == 0 && threadIdx.x == 5 ) {
    //   printf("elt=%12.8f sum_fp32=%12.8f\n", data_fp32[0][0], sum_fp32);
    // }

    // Fix the sum if needed.
    if (sum_fp32 == 0.f || sum_fp32 != sum_fp32)
    {
        sum_fp32 = 1.f;
    }

    // Normalize.
    float inv_sum_fp32 = 1.f / (sum_fp32 + expf(attention_sink - max_fp32));
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        data_fp32[ii][0] *= inv_sum_fp32;
        data_fp32[ii][1] *= inv_sum_fp32;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
static inline __device__ void reduce(float (&data_fp32)[N][4], int8_t const (&mask)[N][4], int warps_n, float& sum_fp32,
    float& max_fp32, float const attention_sink)
{

// Apply the masks.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        data_fp32[ii][0] = mask[ii][0] ? data_fp32[ii][0] : -HUGE_VALF;
        data_fp32[ii][1] = mask[ii][1] ? data_fp32[ii][1] : -HUGE_VALF;
        data_fp32[ii][2] = mask[ii][2] ? data_fp32[ii][2] : -HUGE_VALF;
        data_fp32[ii][3] = mask[ii][3] ? data_fp32[ii][3] : -HUGE_VALF;
    }

    // Compute the max inside the thread.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        max_fp32 = fmaxf(max_fp32, data_fp32[ii][0]);
        max_fp32 = fmaxf(max_fp32, data_fp32[ii][1]);
        max_fp32 = fmaxf(max_fp32, data_fp32[ii][2]);
        max_fp32 = fmaxf(max_fp32, data_fp32[ii][3]);
    }

// Compute inside the warp.
#pragma unroll
    for (int xor_mask = THREADS_PER_WARP / 2; xor_mask > 0; xor_mask /= 2)
    {
        max_fp32 = fmaxf(max_fp32, __shfl_xor_sync(uint32_t(-1), max_fp32, xor_mask));
    }

// // DEBUG.
// if( blockIdx.z == 1 && threadIdx.y == 0 && threadIdx.x == 5 ) {
//   printf("elt=%12.8f max_fp32=%12.8f\n", data_fp32[0][0], max_fp32);
// }

// Transform the elements.
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        data_fp32[ii][0] = apply_exp_(data_fp32[ii][0], max_fp32);
        data_fp32[ii][1] = apply_exp_(data_fp32[ii][1], max_fp32);
        data_fp32[ii][2] = apply_exp_(data_fp32[ii][2], max_fp32);
        data_fp32[ii][3] = apply_exp_(data_fp32[ii][3], max_fp32);
    }

    // Compute the max inside the thread.
    // float sum_fp32 = 0.f;

    // TODO needs refactoring...

#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
    // Within a thread it should correspond to the operation done in the tmp[0]/[1] loop.

    if (warps_n == 1)
    { // E.g. 4x1: 4 threads iterate over all cores.
        // TODO not sure if we can improve this on the gmma side without using additional regs.

        // this is intentionally o(n) instead of o(log n)
        // lanes 0 and 1 here represent the first quad.

        // need to account for offset of l0 when addressing absolute lanes.
        int const ti = threadIdx.x % 2;
        float tmp[2] = {0.f, 0.f};

        for (int ni = 0; ni < N; ni++)
        {
            // +1
            float x = data_fp32[ni][0] + data_fp32[ni][1];
            float y = data_fp32[ni][2] + data_fp32[ni][3];
            tmp[0] += x;
            tmp[1] += y;

            for (int it = 1; it < 16; it++)
            {
                tmp[0] += __shfl_sync(uint32_t(-1), x, 2 * it + ti);
                __syncwarp();
                tmp[1] += __shfl_sync(uint32_t(-1), y, 2 * it + ti);
                __syncwarp();
            }
        }

        // emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
        tmp[0] += tmp[1];
        // emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
        tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 1);
        __syncwarp();
        sum_fp32 = __shfl_sync(uint32_t(-1), tmp[0], 0);
    }
    else
    {

        // SEQLEN == 128.
        if (N == 1)
        {

            float tmp[2] = {0.f, 0.f};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 // GV100
            // The thread local reduction.
            tmp[0] += data_fp32[0][0];
            tmp[0] += data_fp32[0][1];
            tmp[0] += data_fp32[0][2];
            tmp[0] += data_fp32[0][3];

            // Add threads 0 and 2. Inside a thread in the impl.
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 2);
            __syncwarp();
            // Add threads 0 and 8. Inside the thread.
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 8);
            __syncwarp();
            // Add threads 0 and 16. Inside the thread.
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 16);
            __syncwarp();

            // Add threads 0 and 1.
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 1);
            __syncwarp();

            // Add threads 0 and 4. Inter-warp in the code.
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 4);
            __syncwarp();
#else
            if (warps_n == 2)
            { // 2x2
                tmp[0] += data_fp32[0][0] + data_fp32[0][1];
                tmp[1] += data_fp32[0][2] + data_fp32[0][3];

                // Emulate a_01 += a_23...
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 2);
                __syncwarp();
                tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 2);
                __syncwarp();

                // Emulate a_01 += a_45...
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 8);
                __syncwarp();
                tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 8);
                __syncwarp();

                // Emulate a_01 += a_89...
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 16);
                __syncwarp();
                tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 16);
                __syncwarp();

                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
                tmp[0] += tmp[1];

                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 1);
                __syncwarp();

                // Emulate the final reduction in smem.
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 4);
                __syncwarp();
            }
            else
            { // 1x4
                tmp[0] += data_fp32[0][0] + data_fp32[0][1];
                tmp[1] += data_fp32[0][2] + data_fp32[0][3];

                // Add +64.
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 16);
                __syncwarp();
                tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 16);
                __syncwarp();

                // T0: Emulate dst[mi] = tmp[mi][0] + tmp[mi][1];
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 2);
                __syncwarp();
                // T1: Emulate dst[mi] = tmp[mi][0] + tmp[mi][1];
                tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 2);
                __syncwarp();

                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
                tmp[0] += tmp[1];

                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 1);
                __syncwarp();
                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 4); __syncwarp();
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 4);
                __syncwarp();
                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 8); __syncwarp();
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 8);
                __syncwarp();
            }

#endif // ! GV100

            // Don't forget to put the value in sum_fp32 :)
            // sum_fp32 = tmp[0];
            sum_fp32 = __shfl_sync(uint32_t(-1), tmp[0], 0);

            // SEQLEN == 256 - compare with 1x4.
        }
        else if (N == 2 || N == 8)
        {

#pragma unroll
            for (int step = 0; step < N; step += 2)
            {

                float tmp[2] = {0.f, 0.f};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 // GV100

                // The thread local reduction.
                tmp[0] += data_fp32[step + 0][0];
                tmp[0] += data_fp32[step + 0][1];
                tmp[0] += data_fp32[step + 0][2];
                tmp[0] += data_fp32[step + 0][3];

                tmp[1] += data_fp32[step + 1][0];
                tmp[1] += data_fp32[step + 1][1];
                tmp[1] += data_fp32[step + 1][2];
                tmp[1] += data_fp32[step + 1][3];

                // Sum offset 0 and 128 (and so on).
                tmp[0] += tmp[1];

                // Add threads 0 and 2. Inside a thread in the impl.
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 2);
                __syncwarp();
                // Add threads 0 and 16. Inside the thread.
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 16);
                __syncwarp();

                // Add threads 0 and 1.
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 1);
                __syncwarp();

                // Add threads 0 and 4. Inter-warp in the code.
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 4);
                __syncwarp();
                // Add threads 0 and 8. Inter-warp in the code.
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 8);
                __syncwarp();
#else
                // 0.
                tmp[0] += data_fp32[step + 0][0] + data_fp32[step + 0][1];
                tmp[1] += data_fp32[step + 0][2] + data_fp32[step + 0][3];

                // Add +64.
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 16);
                __syncwarp();
                tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 16);
                __syncwarp();

                // Add +128 but use temp storage due to the next round of shfl.
                float xy = data_fp32[step + 1][0] + data_fp32[step + 1][1];
                float zw = data_fp32[step + 1][2] + data_fp32[step + 1][3];

                // Add +128.
                tmp[0] += xy;
                tmp[1] += zw;

                // Add +192.
                tmp[0] += __shfl_xor_sync(uint32_t(-1), xy, 16);
                __syncwarp();
                tmp[1] += __shfl_xor_sync(uint32_t(-1), zw, 16);
                __syncwarp();

                // T0: Emulate dst[mi] = tmp[mi][0] + tmp[mi][1];
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 2);
                __syncwarp();
                // T1: Emulate dst[mi] = tmp[mi][0] + tmp[mi][1];
                tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 2);
                __syncwarp();

                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
                tmp[0] += tmp[1];

                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 1);
                __syncwarp();
                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 4); __syncwarp();
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 4);
                __syncwarp();
                // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 8); __syncwarp();
                tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 8);
                __syncwarp();
#endif // ! GV100

                // Don't forget to put the value in sum_fp32 :)
                sum_fp32 += tmp[0];
            }
            // Emulate taking warp results from position 0, 16, 32, 48, etc.
            sum_fp32 = __shfl_sync(uint32_t(-1), sum_fp32, 0);

            // SEQLEN == 384.
        }
        else if (N == 3)
        {

            float tmp[2] = {0.f, 0.f};

// The reduction inside the thread.
#pragma unroll
            for (int ii = 0; ii < N; ++ii)
            {
                tmp[0] += data_fp32[ii][0];
                tmp[0] += data_fp32[ii][1];
                tmp[1] += data_fp32[ii][2];
                tmp[1] += data_fp32[ii][3];
            }

            // Emulate dst[mi] = tmp[mi][0] + tmp[mi][1];
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 2);
            __syncwarp();
            tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 2);
            __syncwarp();

            // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
            tmp[0] += tmp[1];

            // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 1);
            __syncwarp();

            // Emulate the final summation.
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 4);
            __syncwarp();
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 8);
            __syncwarp();
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 16);
            __syncwarp();

            // Don't forget to put the value in sum_fp32 :)
            sum_fp32 += tmp[0];
            // SEQLEN == 512 - compare with 1x8.
        }
        else if (N >= 4)
        {
            // Emulate thread local
            float tmp[2] = {0.f, 0.f}; // T0, T1
#pragma unroll
            for (int step = 0; step < N; step++)
            {
                tmp[0] += data_fp32[step][0]; // + 0
                tmp[0] += data_fp32[step][1]; // + 1
                tmp[1] += data_fp32[step][2]; // + 2
                tmp[1] += data_fp32[step][3]; // + 3
            }

            // T0: Emulate dst[mi] = tmp[mi][0] + tmp[mi][1];
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 2);
            __syncwarp();
            // T1: Emulate dst[mi] = tmp[mi][0] + tmp[mi][1];
            tmp[1] += __shfl_xor_sync(uint32_t(-1), tmp[1], 2);
            __syncwarp();

            // Emulate intra-thread
            // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 1); __syncwarp();
            tmp[0] += tmp[1];
            // Emulate dst[mi] += __shfl_xor_sync(uint32_t(-1), dst[mi], 2); __syncwarp();
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 1);
            __syncwarp();

            // Emulate inter-thread
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 4);
            __syncwarp();
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 8);
            __syncwarp();
            tmp[0] += __shfl_xor_sync(uint32_t(-1), tmp[0], 16);
            __syncwarp();

            // Don't forget to put the value in sum_fp32 :)
            // sum_fp32 = tmp[0];

            // Emulate taking warp results from position 0, 16, 32, 48, etc.
            sum_fp32 = __shfl_sync(uint32_t(-1), tmp[0], 0);
            // Not supported.
        }
        else
        {
            assert(false);
        }
    } // warps_n ==  1
#else
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        sum_fp32 += data_fp32[ii][0];
        sum_fp32 += data_fp32[ii][1];
        sum_fp32 += data_fp32[ii][2];
        sum_fp32 += data_fp32[ii][3];
    }

// Compute inside the warp.
#pragma unroll
    for (int xor_mask = THREADS_PER_WARP / 2; xor_mask > 0; xor_mask /= 2)
    {
        sum_fp32 += __shfl_xor_sync(uint32_t(-1), sum_fp32, xor_mask);
    }
#endif

    // // DEBUG.
    // if( blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.x == 0 ) {
    //     printf("elt=%12.8f sum_fp32=%12.8f\n", data_fp32[0][0], sum_fp32);
    // }
    // // END OF DEBUG.

    // Fix the sum if needed.
    if (sum_fp32 == 0.f || sum_fp32 != sum_fp32)
    {
        sum_fp32 = 1.f;
    }

    // Normalize.
    float inv_sum_fp32 = 1.f / (sum_fp32 + expf(attention_sink - max_fp32));
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        data_fp32[ii][0] *= inv_sum_fp32;
        data_fp32[ii][1] *= inv_sum_fp32;
        data_fp32[ii][2] *= inv_sum_fp32;
        data_fp32[ii][3] *= inv_sum_fp32;
    }
}

template <typename Data_type, int X>
struct VecX
{

    using Type = typename fmha::Uint_from_size_in_bytes<X * sizeof(Data_type)>::Type;
    static_assert(sizeof(Type) == X * sizeof(Data_type));

    union Alias
    {
        Type raw;
        Data_type elt[X];
    };

    static __device__ inline void to_floatX(
        float (&dst)[X], Type const& src, float const scale, float const attn_logit_softcapping_scale)
    {
        Alias tmp;
        tmp.raw = src;
#pragma unroll
        for (int it = 0; it < X; it++)
        {
            dst[it] = to_float(tmp.elt[it], scale);
            if (attn_logit_softcapping_scale != 0.f)
            {
                dst[it] = attn_logit_softcapping_scale * fmha::__tanhf(dst[it] / attn_logit_softcapping_scale);
            }
        }
    }

    static __device__ inline void from_floatX(Type& dst, float const (&src)[X], float const scale)
    {
        Alias tmp;
#pragma unroll
        for (int it = 0; it < X; it++)
        {
            from_float(tmp.elt[it], src[it], scale);
        }
        dst = tmp.raw;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float get_alibi_head_scaling_factor(int const head_id, int const num_heads)
{
    // Round down to power of 2
    int const num_heads_pow2 = (1u << (31 - __clz(num_heads)));
    if (head_id < num_heads_pow2)
    {
        return exp2f((head_id + 1) * -8.0f / num_heads_pow2);
    }
    else
    {
        float const adjusted_head_id = 2 * (head_id - num_heads_pow2) + 1;
        return exp2f(adjusted_head_id * -4.0f / num_heads_pow2);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dst_type, typename Src_type, int SEQLEN, int WARPS_PER_CTA, int X = 4>
static __global__ void softmax_kernel(Softmax_params<Dst_type, Src_type> params)
{

    // By default, use LDG.64 for the loads and STG.64 for the stores.
    enum
    {
        ELEMENTS_PER_LDG = X,
        ELEMENTS_PER_STG = X
    };

    // The number of Vec_type per thread.
    enum
    {
        VECs_PER_THREAD = SEQLEN / THREADS_PER_WARP / ELEMENTS_PER_LDG
    };

    // DEBUG.
    static_assert(VECs_PER_THREAD * THREADS_PER_WARP * ELEMENTS_PER_LDG == SEQLEN, "");
    // END OF DEBUG.

    using VecO = VecX<Dst_type, X>;
    using VecI = VecX<Src_type, X>;
    using VecM = VecX<int8_t, X>;
    // The vector types.
    using DstX_type = typename VecO::Type;
    using SrcX_type = typename VecI::Type;

    // Make sure the sizes match our expectations.
    static_assert(sizeof(DstX_type) == X * sizeof(Dst_type));
    static_assert(sizeof(SrcX_type) == X * sizeof(Src_type));

    // The type of the mask.
    using MaskX_type = typename VecM::Type;

    // One warp per sequence.
    size_t hi = blockIdx.y * WARPS_PER_CTA + threadIdx.y;
    size_t bi = blockIdx.z;
    size_t si = blockIdx.x;

    // The data offset. Layout is S * B * H * S.
    size_t src_offset = si * params.bhs + bi * params.hs + hi * SEQLEN + threadIdx.x * ELEMENTS_PER_LDG;

    // Load the input elements.
    SrcX_type const* src_ptr = reinterpret_cast<SrcX_type const*>(&params.src[src_offset]);
    SrcX_type data_src[VECs_PER_THREAD];
#pragma unroll
    for (int ii = 0; ii < VECs_PER_THREAD; ++ii)
    {
        if (hi < params.h)
        {
            data_src[ii] = src_ptr[ii * THREADS_PER_WARP];
        }
    }

    // The mask offset. Layout is S * B * S.
    size_t mask_offset = si * params.bs + bi * SEQLEN + threadIdx.x * ELEMENTS_PER_LDG;

    // Load the masks.
    MaskX_type const* mask_ptr = reinterpret_cast<MaskX_type const*>(&params.mask[mask_offset]);
    MaskX_type mask[VECs_PER_THREAD];
#pragma unroll
    for (int ii = 0; ii < VECs_PER_THREAD; ++ii)
    {
        mask[ii] = mask_ptr[ii * THREADS_PER_WARP];
    }

    // Convert the data to float.
    float data_fp32[VECs_PER_THREAD][X];
    int8_t mask_[VECs_PER_THREAD][X];
#pragma unroll
    for (int ii = 0; ii < VECs_PER_THREAD; ++ii)
    {
        VecI::to_floatX(data_fp32[ii], data_src[ii], params.scale_bmm1, params.softcapping_scale_bmm1);

        typename VecM::Alias tmp;
        tmp.raw = mask[ii];
#pragma unroll
        for (int it = 0; it < X; it++)
        {
            mask_[ii][it] = tmp.elt[it];
        }
    }

    if (params.has_alibi)
    {
        float const alibi_factor = get_alibi_head_scaling_factor(hi, params.h);
#pragma unroll
        for (int ii = 0; ii < VECs_PER_THREAD; ii++)
        {
#pragma unroll
            for (int jj = 0; jj < X; jj++)
            {
                int col = ii * THREADS_PER_WARP * X + threadIdx.x * X + jj;
                data_fp32[ii][jj] += alibi_factor * col;
            }
        }
    }

    // The attention sink value.
    float attention_sink = -FLT_MAX;
    if (params.attention_sinks != nullptr)
    {
        attention_sink = params.attention_sinks[hi];
    }

    // Do the reduction.
    float sum_fp32 = 0.f;
    float max_fp32 = -HUGE_VALF;
    reduce(data_fp32, mask_, params.warps_n, sum_fp32, max_fp32, attention_sink);
    if (threadIdx.x == 0)
    {
        int sum_s = params.cu_q_seqlens[bi];
        // [B, S, H, 2] {max, sum} float
        if (hi < params.h)
        {
            params.softmax_sum[(sum_s + si) * params.h * 2 + hi * 2] = max_fp32;
            params.softmax_sum[(sum_s + si) * params.h * 2 + hi * 2 + 1] = sum_fp32;
        }
    }
    // Reconvert to half.
    DstX_type data_dst[VECs_PER_THREAD];
#pragma unroll
    for (int ii = 0; ii < VECs_PER_THREAD; ++ii)
    {
        VecO::from_floatX(data_dst[ii], data_fp32[ii], params.scale_softmax);
    }

    // Store the output elements.
    DstX_type* dst_ptr = reinterpret_cast<DstX_type*>(&params.dst[src_offset]);
#pragma unroll
    for (int ii = 0; ii < VECs_PER_THREAD; ++ii)
    {
        if (hi < params.h)
        {
            dst_ptr[ii * THREADS_PER_WARP] = data_dst[ii];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dst_type, typename Src_type>
void run_softmax(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum,
    void* cu_q_seqlens, int s_inner, int s_outer, int b, int h, float scale_bmm1, float scale_softmax,
    float softcapping_scale_bmm1, int warps_n, bool has_alibi)
{

    Softmax_params<Dst_type, Src_type> params;
    memset(&params, 0, sizeof(params));

    // The different pointers.
    params.dst = reinterpret_cast<Dst_type*>(dst);
    params.src = reinterpret_cast<Src_type const*>(src);
    params.softmax_sum = reinterpret_cast<float*>(softmax_sum);
    params.cu_q_seqlens = reinterpret_cast<int*>(cu_q_seqlens);
    params.mask = reinterpret_cast<int8_t const*>(mask);
    params.attention_sinks = reinterpret_cast<float const*>(attention_sinks);
    params.has_alibi = has_alibi;

    // The dimensions and precomputed values.
    params.b = b;
    params.h = h;
    params.bhs = b * h * s_inner;
    params.hs = h * s_inner;
    params.bs = b * s_inner;

    // The scaling factors for the int8 version to convert to/from float.
    params.scale_bmm1 = scale_bmm1;
    params.softcapping_scale_bmm1 = softcapping_scale_bmm1;
    params.scale_softmax = scale_softmax;
    // The number of warps_n used to identify the reduction strategy.
    params.warps_n = warps_n;

    // Compute the grid size.
    enum
    {
        WARPS_PER_CTA = 4
    };

    dim3 grid(s_outer, (h + WARPS_PER_CTA - 1) / WARPS_PER_CTA, b);
    dim3 threads_per_cta(THREADS_PER_WARP, WARPS_PER_CTA);

    // Launch the kernel.
    if (s_inner == 32)
    {
        softmax_kernel<Dst_type, Src_type, 32, WARPS_PER_CTA, 1><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 64)
    {
        softmax_kernel<Dst_type, Src_type, 64, WARPS_PER_CTA, 2><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 96)
    {
        softmax_kernel<Dst_type, Src_type, 96, WARPS_PER_CTA, 1><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 128)
    {
        softmax_kernel<Dst_type, Src_type, 128, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 192)
    {
        softmax_kernel<Dst_type, Src_type, 192, WARPS_PER_CTA, 2><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 256)
    {
        softmax_kernel<Dst_type, Src_type, 256, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 384)
    {
        softmax_kernel<Dst_type, Src_type, 384, WARPS_PER_CTA, 2><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 512)
    {
        softmax_kernel<Dst_type, Src_type, 512, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 1024)
    {
        softmax_kernel<Dst_type, Src_type, 1024, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 2048)
    {
        softmax_kernel<Dst_type, Src_type, 2048, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 4096)
    {
        softmax_kernel<Dst_type, Src_type, 4096, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 8192)
    {
        softmax_kernel<Dst_type, Src_type, 8192, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 16384)
    {
        softmax_kernel<Dst_type, Src_type, 16384, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 32768)
    {
        softmax_kernel<Dst_type, Src_type, 32768, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else if (s_inner == 65536)
    {
        softmax_kernel<Dst_type, Src_type, 65536, WARPS_PER_CTA><<<grid, threads_per_cta>>>(params);
    }
    else
    {
        assert(false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
