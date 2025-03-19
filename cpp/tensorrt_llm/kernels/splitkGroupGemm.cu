/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "splitkGroupGemm.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"

#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/device/splitk_gemm_grouped.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/default_splitk_gemm_grouped.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/splitk_gemm_grouped.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"

namespace tensorrt_llm::kernels
{

int64_t inline getGemmCoordSize(int64_t problemCount)
{
    return (int64_t) (tensorrt_llm::common::divUp(problemCount * sizeof(cutlass::gemm::GemmCoord), 16) * 16);
}

int64_t inline getPtrSize(int64_t problemCount)
{
    return (int64_t) (tensorrt_llm::common::divUp(problemCount * sizeof(half*), 16) * 16);
}

int64_t inline getLddSize(int64_t problemCount)
{
    return (int64_t) (tensorrt_llm::common::divUp(problemCount * sizeof(int64_t), 16) * 16);
}

int64_t inline getOffsetSize(int64_t problemCount)
{
    return (int64_t) (tensorrt_llm::common::divUp(problemCount * sizeof(int64_t), 16) * 16);
}

int64_t getSplitkGroupedGemmParamsWorkSpaceSize(int64_t problemCount)
{
    auto gemm_coord_size = getGemmCoordSize(problemCount);
    auto ptr_size = 4 * getPtrSize(problemCount);
    auto ldd_size = 4 * getLddSize(problemCount);
    auto offset_size = getOffsetSize(problemCount);

    return gemm_coord_size + ptr_size + ldd_size + offset_size;
}

template <int M1, int N1, int K1, int M2, int N2, int K2, typename cutlassType, int kAlignmentAB, int kAlignmentC,
    int kStages>
void splitkGroupedGemm_(std::vector<cutlass::gemm::GemmCoord> problemSizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkSpace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkSpaceSize,
    int splitKSlices, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    using ElementA = cutlassType;
    using ElementB = cutlassType;
    using ElementOutput = float;
    using ElementAccumulator = float;
    using ElementFinalOutput = cutlassType;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultSplitkGemmGrouped<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, kAlignmentAB, ElementB, LayoutB, cutlass::ComplexTransform::kNone,
        kAlignmentAB, ElementOutput, LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<M1, N1, K1>, cutlass::gemm::GemmShape<M2, N2, K2>, cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<ElementOutput, kAlignmentC, ElementAccumulator,
            ElementAccumulator>,
        // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
        // This parameter is passed in at present to match the APIs of other kernels. The parameter
        // is unused within the kernel.
        cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle, kStages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

    using Gemm = cutlass::gemm::device::SplitkGemmGrouped<GemmKernel>;

    float alpha = 1.0f;
    float beta = 0.0f;
    typename Gemm::EpilogueOutputOp::Params epilogue_op(alpha, beta);

    int problemCount = problemSizes.size();
    auto gemm_coord_size = getGemmCoordSize(problemCount);
    auto ptr_size = getPtrSize(problemCount);
    auto ldd_size = getLddSize(problemCount);
    auto offset_size = getOffsetSize(problemCount);
    auto out_ptr_size = ptr_size;

    char* host_workspace = (char*) std::malloc(gemmParamsWorkSpaceSize);
    cutlass::gemm::GemmCoord* problem_sizes_host = reinterpret_cast<cutlass::gemm::GemmCoord*>(host_workspace);
    ElementA** ptr_A_host = reinterpret_cast<ElementA**>(host_workspace + gemm_coord_size);
    ElementB** ptr_B_host = reinterpret_cast<ElementB**>(host_workspace + gemm_coord_size + ptr_size);
    ElementFinalOutput** ptr_C_host
        = reinterpret_cast<ElementFinalOutput**>(host_workspace + gemm_coord_size + 2 * ptr_size);
    ElementFinalOutput** ptr_D_host
        = reinterpret_cast<ElementFinalOutput**>(host_workspace + gemm_coord_size + 2 * ptr_size + out_ptr_size);
    int64_t* lda_host
        = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 0 * ldd_size);
    int64_t* ldb_host
        = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 1 * ldd_size);
    int64_t* ldc_host
        = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 2 * ldd_size);
    int64_t* ldd_host
        = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 3 * ldd_size);
    int64_t* offset_host
        = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 4 * ldd_size);

    int64_t cumulative_offsets = 0;
    for (int32_t i = 0; i < problemCount; ++i)
    {
        problem_sizes_host[i] = problemSizes.at(i);
        ptr_A_host[i] = (ElementA*) ptrA.at(i);
        ptr_B_host[i] = (ElementB*) ptrB.at(i);
        ptr_C_host[i] = (ElementFinalOutput*) ptrC.at(i);
        ptr_D_host[i] = (ElementFinalOutput*) ptrD.at(i);

        auto const& problem = problemSizes.at(i);
        lda_host[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
        TLLM_CHECK(lda_host[i] % kAlignmentAB == 0);
        ldb_host[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
        TLLM_CHECK(ldb_host[i] % kAlignmentAB == 0);
        ldc_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
        TLLM_CHECK(ldc_host[i] % kAlignmentC == 0);
        ldd_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
        TLLM_CHECK(ldd_host[i] % kAlignmentC == 0);

        offset_host[i] = cumulative_offsets;
        cumulative_offsets += problem.m() * problem.n();
    }

    cutlass::gemm::GemmCoord* problem_sizes_device = reinterpret_cast<cutlass::gemm::GemmCoord*>(gemmParamsWorkSpace);
    ElementA** ptr_A = reinterpret_cast<ElementA**>((char*) gemmParamsWorkSpace + gemm_coord_size);
    ElementB** ptr_B = reinterpret_cast<ElementB**>((char*) gemmParamsWorkSpace + gemm_coord_size + ptr_size);
    ElementFinalOutput** ptr_C
        = reinterpret_cast<ElementFinalOutput**>((char*) gemmParamsWorkSpace + gemm_coord_size + 2 * ptr_size);
    ElementFinalOutput** ptr_D = reinterpret_cast<ElementFinalOutput**>(
        (char*) gemmParamsWorkSpace + gemm_coord_size + 2 * ptr_size + out_ptr_size);
    int64_t* lda = reinterpret_cast<int64_t*>(
        (char*) gemmParamsWorkSpace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 0 * ldd_size);
    int64_t* ldb = reinterpret_cast<int64_t*>(
        (char*) gemmParamsWorkSpace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 1 * ldd_size);
    int64_t* ldc = reinterpret_cast<int64_t*>(
        (char*) gemmParamsWorkSpace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 2 * ldd_size);
    int64_t* ldd = reinterpret_cast<int64_t*>(
        (char*) gemmParamsWorkSpace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 3 * ldd_size);
    int64_t* offset = reinterpret_cast<int64_t*>(
        (char*) gemmParamsWorkSpace + gemm_coord_size + 2 * ptr_size + 2 * out_ptr_size + 4 * ldd_size);

    TLLM_CHECK(((char*) ldc_host - (char*) host_workspace) == ((char*) ldc - (char*) gemmParamsWorkSpace));
    tensorrt_llm::common::cudaAutoCpy(
        (int8_t*) gemmParamsWorkSpace, (int8_t*) host_workspace, gemmParamsWorkSpaceSize, stream);

    int threadblock_count = Gemm::sufficient(problemSizes.data(), problemCount);

    typename Gemm::Arguments args(problem_sizes_device, problemCount, threadblock_count, epilogue_op, ptr_A, ptr_B,
        ptr_C, ptr_D, lda, ldb, ldc, ldd, problemSizes.data(), splitKSlices, offset);

    // Initialize the GEMM object
    Gemm gemm;

    size_t workSpaceSize = gemm.get_workspace_size(args);
    TLLM_CHECK_WITH_INFO(workSpaceSize <= gemmWorkSpaceSize,
        "workSpaceSize (%lu) is smaller than required gemmWorkSpaceSize (%lu).", workSpaceSize,
        (size_t) gemmWorkSpaceSize);

    cutlass::Status status = gemm.initialize(args, gemmWorkSpace);

    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "Failed to initialize CUTLASS Grouped GEMM kernel.");

    // Run the grouped GEMM object
    status = gemm.run(stream);

    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "Failed to run CUTLASS Grouped GEMM kernel.");
    sync_check_cuda_error(stream);

    std::free(host_workspace);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <int M1, int N1, int K1, int M2, int N2, int K2, int kAlignmentAB, int kAlignmentC, int kStages>
void splitkGroupedGemmType_(std::vector<cutlass::gemm::GemmCoord> const& problemSizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkSpace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkSpaceSize,
    nvinfer1::DataType dataType, int splitKSlices, cudaStream_t stream)
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        splitkGroupedGemm_<M1, N1, K1, M2, N2, K2, cutlass::half_t, kAlignmentAB, kAlignmentC, kStages>(problemSizes,
            ptrA, ptrB, ptrC, ptrD, gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkSpaceSize,
            splitKSlices, stream);
    }
    else if (dataType == nvinfer1::DataType::kFLOAT)
    {
        TLLM_CHECK_WITH_INFO(false, "not support float input/output");
    }
#ifdef ENABLE_BF16
    else if (dataType == nvinfer1::DataType::kBF16)
    {
        splitkGroupedGemm_<M1, N1, K1, M2, N2, K2, cutlass::bfloat16_t, kAlignmentAB, kAlignmentC, kStages>(
            problemSizes, ptrA, ptrB, ptrC, ptrD, gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace,
            gemmWorkSpaceSize, splitKSlices, stream);
    }
#endif
}

void splitkGroupedGemm(std::vector<cutlass::gemm::GemmCoord> const& problemSizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkSpace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkSpaceSize,
    bool isLoraIn, nvinfer1::DataType dataType, int splitKSlices, int minKN, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start, isLoraIn: %d, minKN = %d", __PRETTY_FUNCTION__, static_cast<int>(isLoraIn), minKN);
    if (isLoraIn)
    {
        // K >> N, like K = 1024, N = 8
        // Use larger K tile and smaller N tile
        if (minKN >= 8)
        {
            splitkGroupedGemmType_<16, 32, 64, 16, 32, 64, 8, 8, 4>(problemSizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkSpaceSize, dataType, splitKSlices,
                stream);
        }
        else if (minKN >= 4)
        {
            splitkGroupedGemmType_<16, 32, 64, 16, 32, 64, 8, 4, 4>(problemSizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkSpaceSize, dataType, splitKSlices,
                stream);
        }
        else if (minKN >= 2)
        {
            splitkGroupedGemmType_<16, 32, 64, 16, 32, 64, 8, 2, 2>(problemSizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkSpaceSize, dataType, splitKSlices,
                stream);
        }
        else if (minKN >= 1)
        {
            splitkGroupedGemmType_<16, 32, 64, 16, 32, 64, 8, 1, 2>(problemSizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkSpaceSize, dataType, splitKSlices,
                stream);
        }
    }
    else
    {
        // N >> K, like K = 8, N = 1024
        // User larger N tile and smaller K tile
        if (minKN >= 8)
        {
            splitkGroupedGemmType_<32, 128, 32, 32, 32, 32, 8, 8, 4>(problemSizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkSpaceSize, dataType, splitKSlices,
                stream);
        }
        else if (minKN >= 4)
        {
            splitkGroupedGemmType_<32, 128, 32, 32, 32, 32, 4, 8, 4>(problemSizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkSpaceSize, dataType, splitKSlices,
                stream);
        }
        else if (minKN >= 2)
        {
            splitkGroupedGemmType_<32, 128, 32, 32, 32, 32, 2, 8, 2>(problemSizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkSpaceSize, dataType, splitKSlices,
                stream);
        }
        else if (minKN >= 1)
        {
            splitkGroupedGemmType_<32, 128, 32, 32, 32, 32, 1, 8, 2>(problemSizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkSpaceSize, dataType, splitKSlices,
                stream);
        }
    }
}

} // namespace tensorrt_llm::kernels
