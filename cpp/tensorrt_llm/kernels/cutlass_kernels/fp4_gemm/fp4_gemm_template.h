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

#include <type_traits>
#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass_extensions/gemm_configs.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include "../include/fp4_gemm.h"
#include "mxfp8_mxfp4_gemm_template_sm100.h"
#include "nvfp4_nvfp4_gemm_template_sm100.h"
#include "nvfp4_nvfp4_gemm_template_sm120.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{
using namespace cute;

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

template <typename Arch, typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_>
size_t dispatchNVFP4xNVFP4GemmClusterShapeSm10x(T* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (gemmConfig.cluster_shape)
    {
    case tkc::ClusterShape::ClusterShape_1x1x1:
        return genericFp4GemmKernelLauncher<Arch, T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<1>, cute::Int<1>,
            _1SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x1x1:
        return genericFp4GemmKernelLauncher<Arch, T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<1>, cute::Int<1>,
            _2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_1x2x1:
        return genericFp4GemmKernelLauncher<Arch, T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<2>, cute::Int<1>,
            _1SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
        return genericFp4GemmKernelLauncher<Arch, T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<2>, cute::Int<1>,
            _2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_1x4x1:
        return genericFp4GemmKernelLauncher<Arch, T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<4>, cute::Int<1>,
            _1SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_4x2x1:
        return genericFp4GemmKernelLauncher<Arch, T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<2>, cute::Int<1>,
            _2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x4x1:
        return genericFp4GemmKernelLauncher<Arch, T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<4>, cute::Int<1>,
            _2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_4x4x1:
        return genericFp4GemmKernelLauncher<Arch, T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<4>, cute::Int<1>,
            _2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][FP4][dispatch_gemm_cluster_shape] Config is invalid for FP4 GEMM.");
        break;
    }
}

template <typename Arch, typename T>
size_t dispatchNVFP4xNVFP4GemmCTAShapeSm10x(T* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // Several constraints:
    // Cta N should be one of 128/192/256.
    // M-mode size should be 128 or 256 for 2 CTA cluster MMA;
    // M-mode size should be 128 for 1 CTA cluster OMMA.
    // K256 looks to be better than K128
#define CTA_CASE(M, N, K)                                                                                              \
    case tkc::CutlassTileConfigSM100::CtaShape##M##x##N##x##K##B:                                                      \
        return dispatchNVFP4xNVFP4GemmClusterShapeSm10x<Arch, T, cute::Int<M>, cute::Int<N>, cute::Int<K>>(D, A, B,    \
            input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,       \
            occupancy);
#define CTA_CASE_DEFAULT                                                                                               \
    case tkc::CutlassTileConfigSM100::Undefined:                                                                       \
        throw std::runtime_error("[TensorRT-LLM Error][FP4][dispatch_gemm_cta_shape] Gemm config undefined.");         \
        break;                                                                                                         \
    case tkc::CutlassTileConfigSM100::ChooseWithHeuristic:                                                             \
        throw std::runtime_error(                                                                                      \
            "[TensorRT-LLM Error][FP4][dispatch_gemm_cta_shape] Gemm config should have already been set by "          \
            "heuristic.");                                                                                             \
        break;                                                                                                         \
    default:                                                                                                           \
        throw std::runtime_error(                                                                                      \
            "[TensorRT-LLM Error][FP4][dispatch_gemm_cta_shape] Config is invalid for FP4 GEMM.");                     \
        break;
    if constexpr (std::is_same_v<Arch, cutlass::arch::Sm100>)
    {
        switch (gemmConfig.tile_config_sm100)
        {
            CTA_CASE(128, 64, 128)
            CTA_CASE(128, 256, 128)
            CTA_CASE(128, 128, 256)
            CTA_CASE(128, 256, 256)
            CTA_CASE_DEFAULT
        }
    }
    else if constexpr (std::is_same_v<Arch, cutlass::arch::Sm103>)
    {
        switch (gemmConfig.tile_config_sm100)
        {
            CTA_CASE(128, 128, 256)
            CTA_CASE(128, 256, 256)
            CTA_CASE_DEFAULT
        }
    }
    else
    {
        throw std::runtime_error(
            "[TensorRT-LLM Error][FP4][dispatch_gemm_cta_shape] Architecture not supported for FP4 GEMM.");
    }
}

template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_>
size_t dispatchNVFP4xNVFP4GemmClusterShapeSm120(T* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (gemmConfig.cluster_shape)
    {
    case tkc::ClusterShape::ClusterShape_1x1x1:
        return genericFp4GemmKernelLauncherSm120<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<1>, cute::Int<1>>(D,
            A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][FP4][dispatch_gemm_cluster_shape] Config is invalid for FP4 GEMM.");
        break;
    }
}

template <typename T>
size_t dispatchNVFP4xNVFP4GemmCTAShapeSm120(T* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TLLM_LOG_DEBUG("gemmConfig.tile_config_sm120: %d", gemmConfig.tile_config_sm120);

    switch (gemmConfig.tile_config_sm120)
    {
    case tkc::CutlassTileConfigSM120::CtaShape128x128x256B:
        return dispatchNVFP4xNVFP4GemmClusterShapeSm120<T, cute::Int<128>, cute::Int<128>, cute::Int<256>>(D, A, B,
            input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;
    case tkc::CutlassTileConfigSM120::CtaShape256x128x128B:
        return dispatchNVFP4xNVFP4GemmClusterShapeSm120<T, cute::Int<256>, cute::Int<128>, cute::Int<128>>(D, A, B,
            input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;
    case tkc::CutlassTileConfigSM120::Undefined:
        throw std::runtime_error("[TensorRT LLM Error][FP4][sm120][dispatch_gemm_cta_shape] Gemm config undefined.");
        break;
    case tkc::CutlassTileConfigSM120::ChooseWithHeuristic:
        throw std::runtime_error(
            "[TensorRT LLM Error][FP4][sm120][dispatch_gemm_cta_shape] Gemm config should have already been set by "
            "heuristic.");
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][FP4][sm120][dispatch_gemm_cta_shape] Config is invalid for FP4 GEMM.");
        break;
    }
}

template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_>
size_t dispatchMXFP8xMXFP4GemmClusterShapeSm100(T* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (gemmConfig.cluster_shape)
    {
    case tkc::ClusterShape::ClusterShape_2x1x1:
        return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<1>, cute::Int<1>,
            __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
        return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<2>, cute::Int<1>,
            __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_4x2x1:
        return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<2>, cute::Int<1>,
            __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_2x4x1:
        return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<4>, cute::Int<1>,
            __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    case tkc::ClusterShape::ClusterShape_4x4x1:
        return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<4>, cute::Int<1>,
            __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
            stream, occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][FP4][dispatch_gemm_cluster_shape] Config is invalid for FP4 GEMM.");
        break;
    }
}

template <typename T>
size_t dispatchMXFP8xMXFP4GemmCTAShapeSm100(T* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
    int* occupancy = nullptr)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemmConfig.tile_config_sm100)
    {
    case tkc::CutlassTileConfigSM100::CtaShape128x64x128B:
        return dispatchMXFP8xMXFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<64>, cute::Int<128>>(D, A, B,
            input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;
    case tkc::CutlassTileConfigSM100::CtaShape128x256x128B:
        return dispatchMXFP8xMXFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<256>, cute::Int<128>>(D, A, B,
            input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;
    case tkc::CutlassTileConfigSM100::CtaShape128x128x256B:
        return dispatchMXFP8xMXFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<128>, cute::Int<256>>(D, A, B,
            input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;
    case tkc::CutlassTileConfigSM100::CtaShape128x256x256B:
        return dispatchMXFP8xMXFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<256>, cute::Int<256>>(D, A, B,
            input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
            occupancy);
        break;
    case tkc::CutlassTileConfigSM100::Undefined:
        throw std::runtime_error("[TensorRT LLM Error][FP4][dispatch_gemm_cta_shape] Gemm config undefined.");
        break;
    case tkc::CutlassTileConfigSM100::ChooseWithHeuristic:
        throw std::runtime_error(
            "[TensorRT LLM Error][FP4][dispatch_gemm_cta_shape] Gemm config should have already been set by "
            "heuristic.");
        break;
    default:
        throw std::runtime_error("[TensorRT LLM Error][FP4][dispatch_gemm_cta_shape] Config is invalid for FP4 GEMM.");
        break;
    }
}

template <typename T, FP4GemmType fp4GemmType>
CutlassFp4GemmRunner<T, fp4GemmType>::CutlassFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device{-1};
    tk::check_cuda_error(cudaGetDevice(&device));
    mSm = tk::getSMVersion();
    tk::check_cuda_error(cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, FP4GemmType fp4GemmType>
CutlassFp4GemmRunner<T, fp4GemmType>::~CutlassFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template <typename T, FP4GemmType fp4GemmType>
size_t CutlassFp4GemmRunner<T, fp4GemmType>::dispatchToArch(T* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
    int* occupancy)
{
    if constexpr (fp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8)
    {
        if (mSm == 100 || mSm == 103)
        {
            return dispatchMXFP8xMXFP4GemmCTAShapeSm100<T>(D, A, B, input_sf, weight_sf, global_sf, m, n, k,
                batch_count, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        }
        else
        {
            throw std::runtime_error(
                "[TensorRT LLM Error][CutlassFp4GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS FP4 GEMM");
        }
    }
    else if constexpr (fp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4)
    {
        if (mSm == 103)
        {
#ifdef COMPILE_BLACKWELL_SM103_TMA_GEMMS
            return dispatchNVFP4xNVFP4GemmCTAShapeSm10x<cutlass::arch::Sm103, T>(D, A, B, input_sf, weight_sf,
                global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream, occupancy);
#else
            return dispatchNVFP4xNVFP4GemmCTAShapeSm10x<cutlass::arch::Sm100, T>(D, A, B, input_sf, weight_sf,
                global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream, occupancy);
#endif
        }
        else if (mSm == 100)
        {
            return dispatchNVFP4xNVFP4GemmCTAShapeSm10x<cutlass::arch::Sm100, T>(D, A, B, input_sf, weight_sf,
                global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        }
        else if (mSm == 120 || mSm == 121)
        {
            return dispatchNVFP4xNVFP4GemmCTAShapeSm120<T>(D, A, B, input_sf, weight_sf, global_sf, m, n, k,
                batch_count, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        }
        else
        {
            throw std::runtime_error(
                "[TensorRT LLM Error][CutlassFp4GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS FP4 GEMM");
        }
    }
    else
    {
        throw std::runtime_error(
            "[TensorRT LLM Error][CutlassFp4GemmRunner][GEMM Dispatch] FP4 Gemm type unsupported for CUTLASS FP4 GEMM");
    }
}

template <typename T, FP4GemmType fp4GemmType>
void CutlassFp4GemmRunner<T, fp4GemmType>::gemm(void* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    CutlassFp4GemmRunner<T, fp4GemmType>::dispatchToArch(reinterpret_cast<T*>(D), A, B, input_sf, weight_sf, global_sf,
        m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream);
}

template <typename T, FP4GemmType fp4GemmType>
std::vector<tkc::CutlassGemmConfig> CutlassFp4GemmRunner<T, fp4GemmType>::getConfigs() const
{
    using tkc::CutlassTileConfig;
    using tkc::CutlassGemmConfig;

    std::vector<CutlassGemmConfig> candidateConfigs;

    if (mSm == 100 || mSm == 103)
    {
        std::vector<tkc::CutlassTileConfigSM100> tilesSm10x = {
            tkc::CutlassTileConfigSM100::CtaShape128x128x256B,
            tkc::CutlassTileConfigSM100::CtaShape128x256x256B,
        };
        if (mSm == 100)
        {
            tilesSm10x.push_back(tkc::CutlassTileConfigSM100::CtaShape128x64x128B);
            tilesSm10x.push_back(tkc::CutlassTileConfigSM100::CtaShape128x256x128B);
        }
        std::vector<tkc::ClusterShape> clusterShapes = {
            tkc::ClusterShape::ClusterShape_1x1x1,
            tkc::ClusterShape::ClusterShape_1x2x1,
            tkc::ClusterShape::ClusterShape_2x1x1,
            tkc::ClusterShape::ClusterShape_2x2x1,
            tkc::ClusterShape::ClusterShape_1x4x1,
            tkc::ClusterShape::ClusterShape_4x2x1,
            tkc::ClusterShape::ClusterShape_2x4x1,
            tkc::ClusterShape::ClusterShape_4x4x1,
        };
        for (auto const& tile_config : tilesSm10x)
        {
            for (auto const& cluster_config : clusterShapes)
            {
                if constexpr (fp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8)
                {
                    // Skip for high smem usage.
                    if (cluster_config == tkc::ClusterShape::ClusterShape_1x1x1
                        || cluster_config == tkc::ClusterShape::ClusterShape_1x2x1
                        || cluster_config == tkc::ClusterShape::ClusterShape_1x4x1)
                    {
                        continue;
                    }
                }
                CutlassGemmConfig config(tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                    cluster_config, tkc::ClusterShape::Undefined, tkc::ClusterShape::Undefined, mSm);
                candidateConfigs.push_back(config);
            }
        }
    }
    else if (mSm == 120 || mSm == 121)
    {
        std::vector<tkc::CutlassTileConfigSM120> tilesSm120 = {
            // tkc::CutlassTileConfigSM120::CtaShape128x128x128B,
            tkc::CutlassTileConfigSM120::CtaShape128x128x256B,
            tkc::CutlassTileConfigSM120::CtaShape256x128x128B,

        };
        tkc::ClusterShape clusterShape = tkc::ClusterShape::ClusterShape_1x1x1;
        for (auto const& tile_config : tilesSm120)
        {
            CutlassGemmConfig config(
                tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO, clusterShape);
            candidateConfigs.push_back(config);
        }
    }

    return candidateConfigs;
}

template <typename T, FP4GemmType fp4GemmType>
size_t CutlassFp4GemmRunner<T, fp4GemmType>::getWorkspaceSizeImpl(
    int const m, int const n, int const k, int const batch_count)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t workspace_size = 0;
    auto gemmConfigs = CutlassFp4GemmRunner<T, fp4GemmType>{}.getConfigs();
    for (auto const& gemmConfig : gemmConfigs)
    {
        try
        {
            size_t curr_workspace_size = CutlassFp4GemmRunner<T, fp4GemmType>::dispatchToArch(
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, batch_count, gemmConfig, nullptr, 0, 0);
            workspace_size = std::max(workspace_size, curr_workspace_size);
        }
        catch (std::runtime_error& e)
        {
            // Swallow errors when SMEM exceeds maximum allowed
            continue;
        }
    }
    return workspace_size;
}

template <typename T, FP4GemmType fp4GemmType>
size_t CutlassFp4GemmRunner<T, fp4GemmType>::getWorkspaceSize(
    int const m, int const n, int const k, int const batch_count)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Custom hash function for the MNKB type
    using MNK = std::tuple<int, int, int, int>;

    struct MNKHash
    {
        size_t operator()(const MNK& mnk) const
        {
            auto h1 = std::hash<int>{}(std::get<0>(mnk));
            auto h2 = std::hash<int>{}(std::get<1>(mnk));
            auto h3 = std::hash<int>{}(std::get<2>(mnk));
            auto h4 = std::hash<int>{}(std::get<3>(mnk));
            return h1 ^ h2 ^ h3 ^ h4;
        }
    };

    static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;

    size_t workspace_size = 0;
    if (workspace_hashmap.find(std::make_tuple(m, n, k, batch_count)) == workspace_hashmap.end())
    {
        workspace_size = CutlassFp4GemmRunner<T, fp4GemmType>::getWorkspaceSizeImpl(m, n, k, batch_count);
        workspace_hashmap[std::make_tuple(m, n, k, batch_count)] = workspace_size;
    }
    else
    {
        workspace_size = workspace_hashmap[std::make_tuple(m, n, k, batch_count)];
    }
    return workspace_size;
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
