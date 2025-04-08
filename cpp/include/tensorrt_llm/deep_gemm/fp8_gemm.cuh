/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/MIT
 *
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "compiler.cuh"
#include "fp8_gemm_impl.cuh"
#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm
{
template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups,
    uint32_t kNumStages, uint32_t kNumTMAMulticast, GemmType kGemmType>
class Gemm
{
private:
    using Barrier = cuda::barrier<cuda::thread_scope_block>;

public:
    Gemm() = default;

    // DeepGEMM
    template <typename LayoutIndexType>
    static void run(__nv_bfloat16* gmem_d, float* scales_b, LayoutIndexType* grouped_layout, uint32_t shape_m,
        CUtensorMap const& tma_a_desc, CUtensorMap const& tma_b_desc, CUtensorMap const& tma_scales_a_desc,
        CUtensorMap const& tma_d_desc, cudaStream_t stream, int num_sms, uint32_t smem_size)
    {
        using SchedulerType = typename SchedulerSelector<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
            kNumGroups, kNumTMAMulticast>::type;
        using InputType = typename SchedulerType::Input;
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups, kNumStages,
            kNumTMAThreads, kNumMathThreadsPerGroup, kNumTMAMulticast, SchedulerType>;
        DG_HOST_ASSERT(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;

        InputType input;
        input.shape_m = shape_m;
        input.grouped_layout = grouped_layout;

        // Launch
        auto status = cudaLaunchKernelEx(
            &config, kernel, gmem_d, scales_b, input, tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    // Grouped GEMM with Offset
    // `problem_m_padded_offsets` is used for reading scales, to satisfy the alignment requirements of TMA,
    // each problem offset in problem_m_padded_offsets must be padded to multiple for 4.
    template <typename LayoutIndexType>
    static void run(__nv_bfloat16* gmem_d, float* scales_b, LayoutIndexType* problem_m_offsets,
        LayoutIndexType* problem_m_padded_offsets, CUtensorMap const& tma_a_desc, CUtensorMap const& tma_b_desc,
        CUtensorMap const& tma_scales_a_desc, CUtensorMap const& tma_d_desc, cudaStream_t stream, int num_sms,
        uint32_t smem_size)
    {
        using SchedulerType = typename SchedulerSelector<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
            kNumGroups, kNumTMAMulticast>::type;
        using InputType = typename SchedulerType::Input;
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups, kNumStages,
            kNumTMAThreads, kNumMathThreadsPerGroup, kNumTMAMulticast, SchedulerType>;
        DG_HOST_ASSERT(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;

        InputType input;
        input.problem_m_offsets = problem_m_offsets;
        input.problem_m_padded_offsets = problem_m_padded_offsets;

        // Launch
        auto status = cudaLaunchKernelEx(
            &config, kernel, gmem_d, scales_b, input, tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }

    // Batched Strided GEMM
    static void run(__nv_bfloat16* gmem_d, float* scales_b, uint32_t shape_m, CUtensorMap const& tma_a_desc,
        CUtensorMap const& tma_b_desc, CUtensorMap const& tma_scales_a_desc, CUtensorMap const& tma_d_desc,
        uint64_t ld_a, uint64_t stride_a, uint64_t ld_b, uint64_t stride_b, uint64_t ld_d, uint64_t stride_d,
        cudaStream_t stream, int num_sms, uint32_t smem_size)
    {
        using SchedulerType = typename SchedulerSelector<kGemmType, SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K,
            kNumGroups, kNumTMAMulticast>::type;
        using InputType = typename SchedulerType::Input;
        // NOTES: we must use 4 warps to do TMA, because `setmaxnreg.aligned` requires 4 warps
        constexpr uint32_t kNumTMAThreads = 128;
        constexpr uint32_t kNumMathThreadsPerGroup = 128;
        auto kernel = fp8_gemm_kernel<SHAPE_N, SHAPE_K, BLOCK_M, BLOCK_N, BLOCK_K, kNumGroups, kNumStages,
            kNumTMAThreads, kNumMathThreadsPerGroup, kNumTMAMulticast, SchedulerType>;
        DG_HOST_ASSERT(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

        // Cluster launch
        cudaLaunchConfig_t config;
        config.gridDim = num_sms;
        config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;

        // Clusters for TMA multicast
        // NOTES: `>= 4` cluster size will cause performance degradation
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {kNumTMAMulticast, 1, 1};
        config.attrs = &attr;
        config.numAttrs = 1;

        InputType input{shape_m, ld_a, stride_a, ld_b, stride_b, ld_d, stride_d};
        // Launch
        auto status = cudaLaunchKernelEx(
            &config, kernel, gmem_d, scales_b, input, tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
        DG_HOST_ASSERT(status == cudaSuccess);
    }
};

template <typename T>
static CUtensorMap make_2d_tma_a_desc(T* global_address, uint32_t shape_m, uint32_t shape_k, uint32_t block_m,
    uint32_t block_k, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::RowMajor,
        shape_m * (gemm_type == GemmType::GroupedMasked ? num_groups : 1), shape_k, block_m, block_k,
        global_stride_in_bytes);
}

template <typename T>
CUtensorMap make_2d_tma_b_desc(T* global_address, uint32_t shape_n, uint32_t shape_k, uint32_t block_n,
    uint32_t block_k, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::ColMajor, shape_k,
        shape_n * (gemm_type != GemmType::Normal ? num_groups : 1), block_k, block_n, global_stride_in_bytes);
}

template <typename T>
CUtensorMap make_2d_tma_d_desc(T* global_address, uint32_t shape_m, uint32_t shape_n, uint32_t block_m,
    uint32_t block_n, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::RowMajor,
        shape_m * (gemm_type == GemmType::GroupedMasked ? num_groups : 1), shape_n, min(block_m, shape_m),
        min(block_n, shape_n), global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
CUtensorMap make_2d_tma_scales_a_desc(T* global_address, uint32_t shape_m, uint32_t shape_k, uint32_t block_m,
    uint32_t block_k, uint32_t num_groups, GemmType gemm_type, uint64_t global_stride_in_bytes = 0)
{
    // Make TMA aligned to 16 bytes
    constexpr uint32_t kAlignment = 16 / sizeof(T);
    shape_m = ceil_div(shape_m, kAlignment) * kAlignment;

    return make_2d_tma_desc(global_address, Layout::ColMajor, shape_m,
        ceil_div(shape_k, block_k)
            * ((gemm_type == GemmType::GroupedMasked || gemm_type == GemmType::StridedBatched) ? num_groups : 1),
        block_m, 1, global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
CUtensorMap make_tma_scales_a_offset_desc(T* global_address, int64_t max_m_padded_total, uint32_t shape_k,
    uint32_t block_m, uint32_t block_k, uint64_t global_stride_in_bytes = 0)
{
    return make_2d_tma_desc(global_address, Layout::ColMajor, max_m_padded_total, ceil_div(shape_k, block_k), block_m,
        1, global_stride_in_bytes, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);
}

template <typename T>
CUtensorMap make_2d_tma_desc(T* global_address, Layout layout, uint32_t gmem_rows, uint32_t gmem_cols,
    uint32_t smem_rows, uint32_t smem_cols, uint64_t global_stride_in_bytes,
    CUtensorMapSwizzle swizzle_type = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B)
{
    if (layout == Layout::RowMajor)
    {
        uint64_t gmem_dim[2] = {gmem_cols, gmem_rows};
        uint32_t smem_dim[2] = {smem_cols, smem_rows};
        global_stride_in_bytes = global_stride_in_bytes ? global_stride_in_bytes : gmem_cols * sizeof(T);
        return make_2d_tma_copy_desc(global_address, gmem_dim, global_stride_in_bytes, smem_dim, swizzle_type);
    }
    else
    {
        uint64_t gmem_dim[2] = {gmem_rows, gmem_cols};
        uint32_t smem_dim[2] = {smem_rows, smem_cols};
        global_stride_in_bytes = global_stride_in_bytes ? global_stride_in_bytes : gmem_rows * sizeof(T);
        return make_2d_tma_copy_desc(global_address, gmem_dim, gmem_rows * sizeof(T), smem_dim, swizzle_type);
    }
}

template <typename LayoutIndexType>
void runGemm(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d, float* scales_a,
    float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n,
    uint32_t block_k, uint32_t num_groups, uint32_t num_tma_multicast, GemmType gemm_type,
    LayoutIndexType* grouped_layout, cudaStream_t stream, int num_sms, uint32_t smem_size)
{
    auto tma_a_desc = make_2d_tma_a_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m, shape_k, block_m, block_k, num_groups, gemm_type, ld_a);
    auto tma_b_desc = make_2d_tma_b_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type, ld_b);
    auto tma_scales_a_desc
        = make_2d_tma_scales_a_desc(scales_a, shape_m, shape_k, block_m, block_k, num_groups, gemm_type);
    auto tma_d_desc = make_2d_tma_d_desc(
        reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, num_groups, gemm_type, ld_d * 2);

    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    // Cluster launch
    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(static_cast<int32_t>(block_m));
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Clusters for TMA multicast
    // NOTES: `>= 4` cluster size will cause performance degradation
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {num_tma_multicast, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    NormalSchedulerInput input;
    input.shape_m = shape_m;
    input.grouped_layout = grouped_layout;

    // Launch
    auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_b, input,
        tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
    DG_HOST_ASSERT(status == cudaSuccess);
}

template <typename LayoutIndexType>
void runGemm(cudaKernel_t kernel, void* mat_a, int ld_a, void* mat_b, int ld_b, void* mat_d, int ld_d, float* scales_a,
    float* scales_b, uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n,
    uint32_t block_k, uint32_t num_groups, uint32_t num_tma_multicast, GemmType gemm_type,
    LayoutIndexType* problem_m_offsets, LayoutIndexType* problem_m_padded_offsets, cudaStream_t stream, int num_sms,
    uint32_t smem_size, uint32_t max_shape_m_padded)
{
    auto tma_a_desc = make_2d_tma_a_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_a), max_shape_m_padded, shape_k, block_m, block_k, num_groups, gemm_type);
    auto tma_b_desc = make_2d_tma_b_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type);
    auto tma_scales_a_desc = make_tma_scales_a_offset_desc(scales_a, max_shape_m_padded, shape_k, block_m, block_k);
    auto tma_d_desc = make_2d_tma_d_desc(
        reinterpret_cast<__nv_bfloat16*>(mat_d), max_shape_m_padded, shape_n, block_m, block_n, num_groups, gemm_type);
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    // Cluster launch
    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(static_cast<int32_t>(block_m));
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Clusters for TMA multicast
    // NOTES: `>= 4` cluster size will cause performance degradation
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {num_tma_multicast, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    GroupedWithOffsetSchedulerInput input;
    input.problem_m_offsets = problem_m_offsets;
    input.problem_m_padded_offsets = problem_m_padded_offsets;

    // Launch
    auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_b, input,
        tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
    DG_HOST_ASSERT(status == cudaSuccess);
}

void runGemm(cudaKernel_t kernel, void* mat_a, uint64_t ld_a, uint64_t stride_a, void* mat_b, uint64_t ld_b,
    uint64_t stride_b, void* mat_d, uint64_t ld_d, uint64_t stride_d, float* scales_a, float* scales_b,
    uint32_t shape_m, uint32_t shape_n, uint32_t shape_k, uint32_t block_m, uint32_t block_n, uint32_t block_k,
    uint32_t num_groups, uint32_t num_tma_multicast, GemmType gemm_type, cudaStream_t stream, int num_sms,
    uint32_t smem_size)
{
    auto tma_a_desc = make_2d_tma_a_desc(reinterpret_cast<__nv_fp8_e4m3*>(mat_a), shape_m * num_groups, shape_k,
        block_m, block_k, num_groups, gemm_type, ld_a);
    auto tma_b_desc = make_2d_tma_b_desc(
        reinterpret_cast<__nv_fp8_e4m3*>(mat_b), shape_n, shape_k, block_n, block_k, num_groups, gemm_type, ld_b);
    auto tma_scales_a_desc
        = make_2d_tma_scales_a_desc(scales_a, shape_m, shape_k, block_m, block_k, num_groups, gemm_type);
    auto tma_d_desc = make_2d_tma_d_desc(
        reinterpret_cast<__nv_bfloat16*>(mat_d), shape_m, shape_n, block_m, block_n, num_groups, gemm_type, ld_d * 2);
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumMathThreadsPerGroup = 128;
    DG_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess);

    // Cluster launch
    cudaLaunchConfig_t config;
    config.gridDim = num_sms;
    config.blockDim = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(static_cast<int32_t>(block_m));
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Clusters for TMA multicast
    // NOTES: `>= 4` cluster size will cause performance degradation
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {num_tma_multicast, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;

    StridedBatchedSchedulerInput input{shape_m, ld_a, stride_a, ld_b, stride_b, ld_d, stride_d};
    // Launch
    auto status = cudaLaunchKernelEx(&config, kernel, reinterpret_cast<__nv_bfloat16*>(mat_d), scales_b, input,
        tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc);
    DG_HOST_ASSERT(status == cudaSuccess);
}

}; // namespace deep_gemm

#pragma clang diagnostic pop
