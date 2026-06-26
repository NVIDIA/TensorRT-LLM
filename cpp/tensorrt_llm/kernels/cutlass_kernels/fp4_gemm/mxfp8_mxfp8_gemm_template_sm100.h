/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/archCondition.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp4_gemm/mxfp8_mxfp4_gemm_template_sm100.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

using namespace cute;
using namespace tensorrt_llm::kernels::cutlass_kernels;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cutlass_kernels
{

#ifdef PLACEHOLDER_KERNELS

template <typename T, typename CTA_M, typename CTA_N, typename CTA_K, typename CGA_M, typename CGA_N, typename CGA_K,
    typename XSM_>
size_t genericMXFP8xMXFP8GemmKernelLauncher(void* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
    int* occupancy)
{
    throw std::runtime_error(
        "[TensorRT LLM Error][MXFP8 gemm Runner] TensorRT LLM is not compiled with support for this Architecture.");
}

#else

template <typename T, typename CTA_M, typename CTA_N, typename CTA_K, typename CGA_M, typename CGA_N, typename CGA_K,
    typename XSM>
struct DeviceGemmMXFP8xMXFP8GemmSm100
{
    using OutElementType = typename TllmToCutlassTypeAdapter<T>::type;
    using ClusterShape = cute::Shape<int, int, _1>;
    using Arch = cutlass::arch::Sm100;
    /* // Input A: MXFP8 (e4m3 + UE8M0 block scales) */
    using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using LayoutA = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 16;
    /* // Input B: MXFP8 (e4m3 + UE8M0 block scales) -- new vs the MXFP4 template */
    using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
    using LayoutB = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 16;
    /* // Input C */
    using ElementC = void;
    using LayoutC = cutlass::layout::RowMajor;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;

    using SFType = cutlass::float_ue8m0_t;
    using ElementCompute = float;
    using ElementAccumulator = float;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueSchedule = typename MXSMTypeAdapter<XSM>::EpilogueSchedule;
    using MainloopSchedule = typename MXSMTypeAdapter<XSM>::MainloopSchedule;
    using TileScheduler = cutlass::gemm::PersistentScheduler;
    using MmaTileShape = cute::Shape<cute::Int<CTA_M{} * MXSMTypeAdapter<XSM>::Scale>, CTA_N, CTA_K>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<Arch, OperatorClass,
        MmaTileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
        OutElementType, LayoutC, AlignmentC, EpilogueSchedule,
        cutlass::epilogue::fusion::LinearCombination<OutElementType, float, void, float>>::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<Arch, cutlass::arch::OpClassBlockScaledTensorOp, ElementA,
            LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, MmaTileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            MainloopSchedule>::CollectiveOp;

    template <typename Base>
    struct Sm10xOnly : Base
    {
        using typename Base::Params;

        CUTLASS_DEVICE
        void operator()(Params const& params, char* smem_buf)
        {
            if constexpr (tensorrt_llm::kernels::arch::is_major_v<10>)
            {
                this->Base::operator()(params, smem_buf);
            }
            else
            {
                if (cute::thread0())
                {
                    printf("%s : This kernel shall only run on SM10x devices.\n", __PRETTY_FUNCTION__);
                    __trap();
                }
            }
        }
    };

    using GemmKernel = Sm10xOnly<cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
        CollectiveMainloop, CollectiveEpilogue, TileScheduler>>;

    using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename T, typename CTA_M, typename CTA_N, typename CTA_K, typename CGA_M, typename CGA_N, typename CGA_K,
    typename XSM_>
size_t genericMXFP8xMXFP8GemmKernelLauncher(void* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
    int* occupancy)
{
    using ElementOutput__ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
    using ElementOutput_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput__, float>::value, float,
            ElementOutput__>::type;
    using ElementOutput =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, SafeBF16>::value,
            cutlass::bfloat16_t, ElementOutput_>::type;

    using MXFP8xMXFP8GemmOperator =
        typename DeviceGemmMXFP8xMXFP8GemmSm100<T, CTA_M, CTA_N, CTA_K, CGA_M, CGA_N, CGA_K, XSM_>::Gemm;
    MXFP8xMXFP8GemmOperator gemm;
    // Reuse the MXFP8xMXFP4 argument preparation helper -- the argument layout
    // is identical (block-scaled SFA/SFB, UE8M0 scales, same problem shape +
    // strides). Only the B element type differs.
    auto args = prepareGemmArgsSm100<MXFP8xMXFP8GemmOperator>(D, A, B, input_sf, weight_sf, global_sf, m, n, k,
        batch_count, dim3(CGA_M{}, CGA_N{}, CGA_K{}), MXSMTypeAdapter<XSM_>::Scale);
    /* // Check shared memory size; throw when SMEM exceeds */
    int smem_size = int(sizeof(typename MXFP8xMXFP8GemmOperator::GemmKernel::SharedStorage));
    static int mMaxSmemSize = tk::getMaxSharedMemoryPerBlockOptin();
    if (smem_size > mMaxSmemSize)
    {
        std::string errMsg = "SMEM size exceeds maximum allowed. Required " + std::to_string(smem_size) + ", got "
            + std::to_string(mMaxSmemSize);
        throw std::runtime_error("[TensorRT LLM Error][MXFP8 gemm Runner] " + errMsg);
    }
    /* // Return workspace size */
    if (!A && !B && !D)
    {
        return gemm.get_workspace_size(args);
    }
    if (gemm.get_workspace_size(args) > workspaceBytes)
    {
        std::string errMsg("Requested workspace size insufficient. Required "
            + std::to_string(gemm.get_workspace_size(args)) + ", got " + std::to_string(workspaceBytes));
        throw std::runtime_error("[TensorRT LLM Error][MXFP8 gemm Runner] " + errMsg);
    }
    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string errMsg = "MXFP8xMXFP8 Gemm cutlass kernel will fail for params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[TensorRT LLM Error][MXFP8 gemm Runner] " + errMsg);
    }
    auto initStatus = gemm.initialize(args, workspace, stream);
    if (initStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Failed to initialize cutlass MXFP8xMXFP8 gemm. Error: "
            + std::string(cutlassGetStatusString(initStatus));
        throw std::runtime_error("[TensorRT LLM Error][MXFP8xMXFP8 gemm Runner] " + errMsg);
    }
    auto runStatus = gemm.run(args, workspace, stream, nullptr, tensorrt_llm::common::getEnvEnablePDL());
    if (runStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg
            = "Failed to run cutlass MXFP8xMXFP8 gemm. Error: " + std::string(cutlassGetStatusString(runStatus));
        throw std::runtime_error("[TensorRT LLM Error][MXFP8xMXFP8 gemm Runner] " + errMsg);
    }
    return gemm.get_workspace_size(args);
}

#endif

} // namespace cutlass_kernels
} // namespace kernels

TRTLLM_NAMESPACE_END
