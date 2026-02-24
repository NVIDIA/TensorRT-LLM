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

#include "cute/numeric/integral_constant.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"

#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm100.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels_oss
{
namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

using namespace cute;

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape, typename ClusterShape,
    typename MainloopSchedule>
void sm100_dispatch_epilogue_schedules(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
    ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
    int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace, size_t workspace_bytes,
    cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemm_config.epilogue_schedule)
    {
    case tkc::EpilogueScheduleType::AUTO:
        // TODO: use heuristics to select the epilogue schedule, depending on the CTA shape and cluster shape
        using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
        sm100_generic_mixed_gemm_kernelLauncher<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType,
            QuantOp, EpilogueTag, CTAShape, ClusterShape, MainloopSchedule, EpilogueSchedule>(A, B, weight_scales,
            weight_zero_points, biases, alpha, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream,
            occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][fpA_intB][sm100_dispatch_epilogue_schedules] Unsupported epilogue schedule for mixed "
            "type GEMM.");
        break;
    }
}

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape, typename ClusterShape>
void sm100_dispatch_mainloop_schedules(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
    ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
    int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace, size_t workspace_bytes,
    cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (gemm_config.mainloop_schedule)
    {
    case tkc::MainloopScheduleType::AUTO:
        // TODO: use heuristics to select the mainloop schedule, depending on the CTA shape and cluster shape
        using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
        sm100_dispatch_epilogue_schedules<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
            EpilogueTag, CTAShape, ClusterShape, MainloopSchedule>(A, B, weight_scales, weight_zero_points, biases,
            alpha, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][fpA_intB][sm100_dispatch_mainloop_schedules] Unsupported mainloop schedule for mixed "
            "type GEMM.");
        break;
    }
}

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape>
void sm100_dispatch_gemm_config(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
    ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
    int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char* workspace, size_t workspace_bytes,
    cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemm_config.cluster_shape)
    {
    // TODO: add support for other cluster shapes
    case tkc::ClusterShape::ClusterShape_1x1x1:
        sm100_dispatch_mainloop_schedules<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
            EpilogueTag, CTAShape, Shape<_1, _1, _1>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n,
            k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][fpA_intB][sm100_dispatch_gemm_config] Unsupported CTA and Cluster shapes for mixed "
            "type GEMM.");
        break;
    }
}

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag>
void sm100_dispatch_gemm_to_cutlass(ActivationType const* A, WeightType const* B, ScaleZeroType const* weight_scales,
    ScaleZeroType const* weight_zero_points, BiasType const* biases, float const alpha, OutputType* C, int m, int n,
    int k, int const group_size, char* workspace, size_t workspace_bytes, tkc::CutlassGemmConfig gemm_config,
    cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // Note that SIMT configs are omitted here since they are not supported for fpA_intB.
    // We also only instantiate configs here where threadblockShapeM == warpShapeM since those usually perform the best
    // for mixed type gemms.

    constexpr int Ktile = 128 / sizeof(ActivationType);
    using _Ktile = Int<Ktile>;
    switch (gemm_config.tile_config_sm100)
    {
    case tkc::CutlassTileConfigSM100::CtaShape64x128x128B:
        sm100_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
            EpilogueTag, Shape<_64, _128, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k,
            group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfigSM100::CtaShape128x128x128B:
        sm100_dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
            EpilogueTag, Shape<_128, _128, _Ktile>>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k,
            group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][fpA_intB][sm100_dispatch_gemm_to_cutlass] Unsupported tile shape for mixed type "
            "GEMM.");
        break;
    }
}

} // namespace cutlass_kernels_oss
} // namespace kernels
} // namespace tensorrt_llm
