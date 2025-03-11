/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"

#include "groupGemm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/memoryUtils.h"

#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace kernels
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

int64_t getGroupedGemmParamsWorkSpaceSize(int64_t problemCount)
{
    auto gemm_coord_size = getGemmCoordSize(problemCount);
    auto ptr_size = 4 * getPtrSize(problemCount);
    auto ldd_size = 4 * getLddSize(problemCount);

    return gemm_coord_size + ptr_size + ldd_size;
}

template <int M1, int N1, int K1, int M2, int N2, int K2, typename cutlassType, int kAlignmentAB, int kAlignmentC,
    int kStages>
void groupedGemm_(std::vector<cutlass::gemm::GemmCoord> problem_sizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkSpace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkspaceSize,
    nvinfer1::DataType dataType, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    using ElementA = cutlassType;
    using ElementB = cutlassType;
    using ElementOutput = cutlassType;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    int problemCount = problem_sizes.size();

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, kAlignmentAB, ElementB, LayoutB, cutlass::ComplexTransform::kNone,
        kAlignmentAB, ElementOutput, LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<M1, N1, K1>, cutlass::gemm::GemmShape<M2, N2, K2>, cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<ElementOutput, kAlignmentC, ElementAccumulator,
            ElementAccumulator>,
        // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
        // This parameter is passed in at present to match the APIs of other kernels. The parameter
        // is unused within the kernel.
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, kStages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    float alpha = 1.0f;
    float beta = 0.0f;
    typename Gemm::EpilogueOutputOp::Params epilogue_op(alpha, beta);

    auto gemm_coord_size = getGemmCoordSize(problemCount);
    auto ptr_size = getPtrSize(problemCount);
    auto ldd_size = getLddSize(problemCount);

    char* host_workspace = (char*) std::malloc(gemmParamsWorkSpaceSize);
    cutlass::gemm::GemmCoord* problem_sizes_host = reinterpret_cast<cutlass::gemm::GemmCoord*>(host_workspace);
    ElementA** ptr_A_host = reinterpret_cast<ElementA**>(host_workspace + gemm_coord_size);
    ElementB** ptr_B_host = reinterpret_cast<ElementB**>(host_workspace + gemm_coord_size + ptr_size);
    ElementOutput** ptr_C_host = reinterpret_cast<ElementOutput**>(host_workspace + gemm_coord_size + 2 * ptr_size);
    ElementOutput** ptr_D_host = reinterpret_cast<ElementOutput**>(host_workspace + gemm_coord_size + 3 * ptr_size);
    int64_t* lda_host = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 4 * ptr_size + 0 * ldd_size);
    int64_t* ldb_host = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 4 * ptr_size + 1 * ldd_size);
    int64_t* ldc_host = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 4 * ptr_size + 2 * ldd_size);
    int64_t* ldd_host = reinterpret_cast<int64_t*>(host_workspace + gemm_coord_size + 4 * ptr_size + 3 * ldd_size);

    for (int32_t i = 0; i < problemCount; ++i)
    {
        problem_sizes_host[i] = problem_sizes.at(i);
        ptr_A_host[i] = (ElementA*) ptrA.at(i);
        ptr_B_host[i] = (ElementB*) ptrB.at(i);
        ptr_C_host[i] = (ElementOutput*) ptrC.at(i);
        ptr_D_host[i] = (ElementOutput*) ptrD.at(i);

        auto problem = problem_sizes.at(i);
        lda_host[i] = LayoutA::packed({problem.m(), problem.k()}).stride(0);
        TLLM_CHECK(lda_host[i] % kAlignmentAB == 0);
        ldb_host[i] = LayoutB::packed({problem.k(), problem.n()}).stride(0);
        TLLM_CHECK(ldb_host[i] % kAlignmentAB == 0);
        ldc_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
        TLLM_CHECK(ldc_host[i] % kAlignmentC == 0);
        ldd_host[i] = LayoutC::packed({problem.m(), problem.n()}).stride(0);
        TLLM_CHECK(ldd_host[i] % kAlignmentC == 0);
    }

    cutlass::gemm::GemmCoord* problem_sizes_device = reinterpret_cast<cutlass::gemm::GemmCoord*>(gemmParamsWorkSpace);
    ElementA** ptr_A = reinterpret_cast<ElementA**>((char*) gemmParamsWorkSpace + gemm_coord_size);
    ElementB** ptr_B = reinterpret_cast<ElementB**>((char*) gemmParamsWorkSpace + gemm_coord_size + ptr_size);
    ElementOutput** ptr_C
        = reinterpret_cast<ElementOutput**>((char*) gemmParamsWorkSpace + gemm_coord_size + 2 * ptr_size);
    ElementOutput** ptr_D
        = reinterpret_cast<ElementOutput**>((char*) gemmParamsWorkSpace + gemm_coord_size + 3 * ptr_size);
    int64_t* lda
        = reinterpret_cast<int64_t*>((char*) gemmParamsWorkSpace + gemm_coord_size + 4 * ptr_size + 0 * ldd_size);
    int64_t* ldb
        = reinterpret_cast<int64_t*>((char*) gemmParamsWorkSpace + gemm_coord_size + 4 * ptr_size + 1 * ldd_size);
    int64_t* ldc
        = reinterpret_cast<int64_t*>((char*) gemmParamsWorkSpace + gemm_coord_size + 4 * ptr_size + 2 * ldd_size);
    int64_t* ldd
        = reinterpret_cast<int64_t*>((char*) gemmParamsWorkSpace + gemm_coord_size + 4 * ptr_size + 3 * ldd_size);

    TLLM_CHECK(((char*) ldc_host - (char*) host_workspace) == ((char*) ldc - (char*) gemmParamsWorkSpace));
    tensorrt_llm::common::cudaAutoCpy(
        (int8_t*) gemmParamsWorkSpace, (int8_t*) host_workspace, gemmParamsWorkSpaceSize, stream);

    int threadblock_count = Gemm::sufficient(problem_sizes.data(), problemCount);

    typename Gemm::Arguments args(problem_sizes_device, problemCount, threadblock_count, epilogue_op, ptr_A, ptr_B,
        ptr_C, ptr_D, lda, ldb, ldc, ldd, problem_sizes.data());

    // Initialize the GEMM object
    Gemm gemm;

    size_t workspace_size = gemm.get_workspace_size(args);
    TLLM_CHECK(gemm.get_workspace_size(args) <= gemmWorkspaceSize);

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
void groupedGemmType_(std::vector<cutlass::gemm::GemmCoord> problem_sizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkSpace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkspaceSize,
    nvinfer1::DataType dataType, cudaStream_t stream)
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        groupedGemm_<M1, N1, K1, M2, N2, K2, cutlass::half_t, kAlignmentAB, kAlignmentC, kStages>(problem_sizes, ptrA,
            ptrB, ptrC, ptrD, gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType,
            stream);
    }
    else if (dataType == nvinfer1::DataType::kFLOAT)
    {
        TLLM_CHECK_WITH_INFO(false, "not support float input/output");
    }
#ifdef ENABLE_BF16
    else if (dataType == nvinfer1::DataType::kBF16)
    {
        groupedGemm_<M1, N1, K1, M2, N2, K2, cutlass::bfloat16_t, kAlignmentAB, kAlignmentC, kStages>(problem_sizes,
            ptrA, ptrB, ptrC, ptrD, gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize,
            dataType, stream);
    }
#endif
}

void groupedGemm(std::vector<cutlass::gemm::GemmCoord> problem_sizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkSpace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkspaceSize,
    bool isLoraIn, nvinfer1::DataType dataType, int minKN, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start, isLoraIn: %d, minKN = %d", __PRETTY_FUNCTION__, static_cast<int>(isLoraIn), minKN);
    if (isLoraIn)
    {
        // K >> N, like K = 1024, N = 8
        // Use larger K tile and smaller N tile
        if (minKN >= 8)
        {
            groupedGemmType_<16, 32, 64, 16, 32, 64, 8, 8, 4>(problem_sizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType, stream);
        }
        else if (minKN >= 4)
        {
            groupedGemmType_<16, 32, 64, 16, 32, 64, 8, 4, 4>(problem_sizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType, stream);
        }
        else if (minKN >= 2)
        {
            groupedGemmType_<16, 32, 64, 16, 32, 64, 8, 2, 2>(problem_sizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType, stream);
        }
        else if (minKN >= 1)
        {
            groupedGemmType_<16, 32, 64, 16, 32, 64, 8, 1, 2>(problem_sizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType, stream);
        }
    }
    else
    {
        // N >> K, like K = 8, N = 1024
        // User larger N tile and smaller K tile
        if (minKN >= 8)
        {
            groupedGemmType_<32, 128, 32, 32, 32, 32, 8, 8, 4>(problem_sizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType, stream);
        }
        else if (minKN >= 4)
        {
            groupedGemmType_<32, 128, 32, 32, 32, 32, 4, 8, 4>(problem_sizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType, stream);
        }
        else if (minKN >= 2)
        {
            groupedGemmType_<32, 128, 32, 32, 32, 32, 2, 8, 2>(problem_sizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType, stream);
        }
        else
        {
            groupedGemmType_<32, 128, 32, 32, 32, 32, 1, 8, 2>(problem_sizes, ptrA, ptrB, ptrC, ptrD,
                gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType, stream);
        }
    }
}

} // namespace kernels

} // namespace tensorrt_llm
