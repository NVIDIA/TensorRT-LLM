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

#pragma once

#include "sm89_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace ada_blockwise_gemm
{

using Fp8 = cutlass::float_e4m3_t;
using Bf16 = cutlass::bfloat16_t;

using nvcuda::wmma::accumulator;
using nvcuda::wmma::col_major;
using nvcuda::wmma::fill_fragment;
using nvcuda::wmma::fragment;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::mem_row_major;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::store_matrix_sync;

template <typename Traits>
__device__ __forceinline__ void run_gemm_tile(DeepGemmLaunchConfig cfg, const Fp8* __restrict__ A,
    const Fp8* __restrict__ B, Bf16* __restrict__ D, const float* __restrict__ SFA,
    const float* __restrict__ SFB, half* shared_a, half* shared_b, float* warp_store_base)
{
    const int M = cfg.M;
    const int N = cfg.N;
    const int K = cfg.K;
    if (M <= 0 || N <= 0 || K <= 0)
    {
        return;
    }

    const int block_row = blockIdx.y * kBlockM;
    const int block_col = blockIdx.x * Traits::kBlockN;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m_idx = warp_id / Traits::kWarpTilesN;
    const int warp_n_idx = warp_id % Traits::kWarpTilesN;
    const int warp_row = warp_m_idx * Traits::kWarpM;
    const int warp_col = warp_n_idx * Traits::kWarpN;

    const int scale_m_stride = cfg.scale_m_stride > 0 ? cfg.scale_m_stride : align_to(M, 4);
    const int scale_n_stride = cfg.scale_n_stride > 0 ? cfg.scale_n_stride : ceil_div(K, kScaleGranularityK);
    const int scale_k_tiles = cfg.scale_k_tiles > 0 ? cfg.scale_k_tiles : ceil_div(K, kScaleGranularityK);

    constexpr int kVecsPerRow = kBlockK / kVectorWidth;
    constexpr int kVecsPerCol = kBlockK / kVectorWidth;
    constexpr int kVecsPerBlockA = (kBlockM * kBlockK) / kVectorWidth;
    constexpr int kVecsPerBlockB = (Traits::kBlockN * kBlockK) / kVectorWidth;

    const cutlass::NumericArrayConverter<float, Fp8, kVectorWidth> fp8_to_float;

    fragment<accumulator, kFragM, kFragN, kFragK, float> acc[Traits::kWarpM / kFragM][Traits::kWarpN / kFragN];
#pragma unroll
    for (int mi = 0; mi < Traits::kWarpM / kFragM; ++mi)
    {
#pragma unroll
        for (int ni = 0; ni < Traits::kWarpN / kFragN; ++ni)
        {
            fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    const int total_k_tiles = (K + kBlockK - 1) / kBlockK;
    for (int tile_k = 0; tile_k < total_k_tiles; ++tile_k)
    {
        const int global_k_base = tile_k * kBlockK;
        int scale_tile = global_k_base / kScaleGranularityK;
        if (scale_tile >= scale_k_tiles)
        {
            scale_tile = scale_k_tiles - 1;
        }

        for (int vec_idx = threadIdx.x; vec_idx < kVecsPerBlockA; vec_idx += Traits::kThreadsPerBlock)
        {
            const int local_row = vec_idx / kVecsPerRow;
            const int vec_col = vec_idx - local_row * kVecsPerRow;
            const int global_row = block_row + local_row;
            const int global_col = global_k_base + vec_col * kVectorWidth;

            alignas(16) cutlass::Array<half, kVectorWidth> converted;
#pragma unroll
            for (int ii = 0; ii < kVectorWidth; ++ii)
            {
                converted[ii] = __float2half(0.0f);
            }

            if (global_row < M)
            {
                const float scale = SFA ? SFA[static_cast<size_t>(global_row)
                        + static_cast<size_t>(scale_m_stride) * static_cast<size_t>(scale_tile)]
                                        : 1.0f;
                const int remaining = K - global_col;
                if (remaining > 0)
                {
                    if (remaining >= kVectorWidth)
                    {
                        const auto* src_ptr = reinterpret_cast<const cutlass::Array<Fp8, kVectorWidth>*>(
                            A + static_cast<size_t>(global_row) * K + global_col);
                        const auto float_vals = fp8_to_float(*src_ptr);
#pragma unroll
                        for (int ii = 0; ii < kVectorWidth; ++ii)
                        {
                            converted[ii] = __float2half(float_vals[ii] * scale);
                        }
                    }
                    else
                    {
                        const Fp8* src_ptr = A + static_cast<size_t>(global_row) * K + global_col;
#pragma unroll
                        for (int ii = 0; ii < kVectorWidth; ++ii)
                        {
                            const bool valid = ii < remaining;
                            const float val = valid ? static_cast<float>(src_ptr[ii]) * scale : 0.0f;
                            converted[ii] = __float2half(val);
                        }
                    }
                }
            }

            auto* dst = reinterpret_cast<uint4*>(shared_a + local_row * kPaddedK + vec_col * kVectorWidth);
            const auto* src = reinterpret_cast<const uint4*>(converted.data());
            dst[0] = src[0];
            dst[1] = src[1];
        }

        for (int vec_idx = threadIdx.x; vec_idx < kVecsPerBlockB; vec_idx += Traits::kThreadsPerBlock)
        {
            const int local_col = vec_idx / kVecsPerCol;
            const int vec_row = vec_idx - local_col * kVecsPerCol;
            const int global_col = block_col + local_col;
            const int global_row = global_k_base + vec_row * kVectorWidth;

            alignas(16) cutlass::Array<half, kVectorWidth> converted;
#pragma unroll
            for (int ii = 0; ii < kVectorWidth; ++ii)
            {
                converted[ii] = __float2half(0.0f);
            }

            if (global_col < N)
            {
                int scale_n_block = global_col / kScaleGranularityN;
                if (scale_n_block >= scale_n_stride)
                {
                    scale_n_block = scale_n_stride - 1;
                }
                const float scale = SFB ? SFB[static_cast<size_t>(scale_n_block)
                        + static_cast<size_t>(scale_n_stride) * static_cast<size_t>(scale_tile)]
                                        : 1.0f;
                const int remaining = K - global_row;
                if (remaining > 0)
                {
                    if (remaining >= kVectorWidth)
                    {
                        const auto* src_ptr = reinterpret_cast<const cutlass::Array<Fp8, kVectorWidth>*>(
                            B + static_cast<size_t>(global_col) * K + global_row);
                        const auto float_vals = fp8_to_float(*src_ptr);
#pragma unroll
                        for (int ii = 0; ii < kVectorWidth; ++ii)
                        {
                            converted[ii] = __float2half(float_vals[ii] * scale);
                        }
                    }
                    else
                    {
                        const Fp8* src_ptr = B + static_cast<size_t>(global_col) * K + global_row;
#pragma unroll
                        for (int ii = 0; ii < kVectorWidth; ++ii)
                        {
                            const bool valid = ii < remaining;
                            const float val = valid ? static_cast<float>(src_ptr[ii]) * scale : 0.0f;
                            converted[ii] = __float2half(val);
                        }
                    }
                }
            }

            auto* dst = reinterpret_cast<uint4*>(shared_b + local_col * kPaddedK + vec_row * kVectorWidth);
            const auto* src = reinterpret_cast<const uint4*>(converted.data());
            dst[0] = src[0];
            dst[1] = src[1];
        }

        __syncthreads();

        for (int kk = 0; kk < kBlockK; kk += kFragK)
        {
            fragment<matrix_a, kFragM, kFragN, kFragK, half, mem_row_major> frag_a[Traits::kWarpM / kFragM];
            fragment<matrix_b, kFragM, kFragN, kFragK, half, col_major> frag_b[Traits::kWarpN / kFragN];

#pragma unroll
            for (int mi = 0; mi < Traits::kWarpM / kFragM; ++mi)
            {
                const int row_offset = warp_row + mi * kFragM;
                const half* tile_ptr = shared_a + row_offset * kPaddedK + kk;
                load_matrix_sync(frag_a[mi], tile_ptr, kPaddedK);
            }

#pragma unroll
            for (int ni = 0; ni < Traits::kWarpN / kFragN; ++ni)
            {
                const int col_offset = warp_col + ni * kFragN;
                const half* tile_ptr = shared_b + col_offset * kPaddedK + kk;
                load_matrix_sync(frag_b[ni], tile_ptr, kPaddedK);
            }

#pragma unroll
            for (int mi = 0; mi < Traits::kWarpM / kFragM; ++mi)
            {
#pragma unroll
                for (int ni = 0; ni < Traits::kWarpN / kFragN; ++ni)
                {
                    mma_sync(acc[mi][ni], frag_a[mi], frag_b[ni], acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    float* warp_tile = warp_store_base + warp_id * (kFragM * kFragN);
    const int out_row_base = block_row + warp_row;
    const int out_col_base = block_col + warp_col;

#pragma unroll
    for (int mi = 0; mi < Traits::kWarpM / kFragM; ++mi)
    {
#pragma unroll
        for (int ni = 0; ni < Traits::kWarpN / kFragN; ++ni)
        {
            const int tile_row = out_row_base + mi * kFragM;
            const int tile_col = out_col_base + ni * kFragN;
            store_matrix_sync(warp_tile, acc[mi][ni], kFragN, mem_row_major);
            __syncwarp();
#pragma unroll
            for (int elem = lane_id; elem < kFragM * kFragN; elem += 32)
            {
                const int r = elem / kFragN;
                const int c = elem - r * kFragN;
                const int global_row = tile_row + r;
                const int global_col = tile_col + c;
                if (global_row < M && global_col < N)
                {
                    D[static_cast<size_t>(global_row) * N + global_col] = Bf16(warp_tile[elem]);
                }
            }
            __syncwarp();
        }
    }
}

template <typename GemmKernel>
__global__ __launch_bounds__(kThreadsPerBlock, 2) void sm89_fp8_gemm_1d1d_impl(DeepGemmLaunchConfig cfg,
    const Fp8* __restrict__ A, const Fp8* __restrict__ B, Bf16* __restrict__ D, const float* __restrict__ SFA,
    const float* __restrict__ SFB)
{
    __shared__ half shared_a[kBlockM * kPaddedK];
    __shared__ half shared_b[kMaxBlockN * kPaddedK];
    __shared__ float warp_store[kMaxWarps][kFragM * kFragN];

    if (cfg.use_wide)
    {
        run_gemm_tile<WideTile>(cfg, A, B, D, SFA, SFB, shared_a, shared_b, &warp_store[0][0]);
    }
    else
    {
        run_gemm_tile<NarrowTile>(cfg, A, B, D, SFA, SFB, shared_a, shared_b, &warp_store[0][0]);
    }
}

template <typename GemmKernel>
__global__ __launch_bounds__(kThreadsPerBlock, 2) void sm89_fp8_bmm_1d1d_impl(DeepGemmLaunchConfig cfg, Fp8* A,
    Fp8* B, Bf16* D, float* scales_a, float* scales_b, uint64_t stride_a, uint64_t stride_b, uint64_t stride_d,
    uint64_t stride_scales_a, uint64_t stride_scales_b)
{
    const int problem_idx = static_cast<int>(blockIdx.z);
    const Fp8* A_ptr = A + static_cast<size_t>(problem_idx) * stride_a;
    const Fp8* B_ptr = B + static_cast<size_t>(problem_idx) * stride_b;
    Bf16* D_ptr = D + static_cast<size_t>(problem_idx) * stride_d;
    float* SFA_ptr = scales_a + static_cast<size_t>(problem_idx) * stride_scales_a;
    float* SFB_ptr = scales_b + static_cast<size_t>(problem_idx) * stride_scales_b;

    __shared__ half shared_a[kBlockM * kPaddedK];
    __shared__ half shared_b[kMaxBlockN * kPaddedK];
    __shared__ float warp_store[kMaxWarps][kFragM * kFragN];

    if (cfg.use_wide)
    {
        run_gemm_tile<WideTile>(cfg, A_ptr, B_ptr, D_ptr, SFA_ptr, SFB_ptr, shared_a, shared_b, &warp_store[0][0]);
    }
    else
    {
        run_gemm_tile<NarrowTile>(cfg, A_ptr, B_ptr, D_ptr, SFA_ptr, SFB_ptr, shared_a, shared_b, &warp_store[0][0]);
    }
}

} // namespace ada_blockwise_gemm
