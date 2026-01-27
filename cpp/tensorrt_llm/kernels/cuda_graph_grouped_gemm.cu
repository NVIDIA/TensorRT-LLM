/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "cuda_graph_grouped_gemm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <ATen/ATen.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/device/splitk_gemm_grouped.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/default_splitk_gemm_grouped.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/**
 * Template for CUDA Graph compatible grouped GEMM that directly uses GPU tensors
 */
template <int M1, int N1, int K1, int M2, int N2, int K2, typename cutlassType, int kAlignmentAB, int kAlignmentC,
    int kStages>
void cudaGraphGroupedGemmTemplate(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu,
    void** ptrBGpu, void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu,
    cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, cudaStream_t stream)
{
    using ElementA = cutlassType;
    using ElementB = cutlassType;
    using ElementOutput = cutlassType;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, kAlignmentAB, ElementB, LayoutB, cutlass::ComplexTransform::kNone,
        kAlignmentAB, ElementOutput, LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<M1, N1, K1>, cutlass::gemm::GemmShape<M2, N2, K2>, cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<ElementOutput, kAlignmentC, ElementAccumulator,
            ElementAccumulator>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, kStages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    float alpha = 1.0f;
    float beta = 0.0f;
    typename Gemm::EpilogueOutputOp::Params epilogueOp(alpha, beta);

    auto ptrA = reinterpret_cast<ElementA**>(ptrAGpu);
    auto ptrB = reinterpret_cast<ElementB**>(ptrBGpu);
    auto ptrC = reinterpret_cast<ElementOutput**>(ptrCGpu);
    auto ptrD = reinterpret_cast<ElementOutput**>(ptrDGpu);

    Gemm gemmOp;

    int threadblockCount = Gemm::sufficient(nullptr, problemCount);

    typename Gemm::Arguments args(problemSizesPtr, // GPU problem sizes
        problemCount,                              // Problem count
        threadblockCount,                          // Threadblock count
        epilogueOp,                                // Epilogue operation
        ptrA,                                      // GPU pointer array A
        ptrB,                                      // GPU pointer array B
        ptrC,                                      // GPU pointer array C (can be nullptr)
        ptrD,                                      // GPU pointer array D
        ldaGpu,                                    // Precomputed leading dimension A (on GPU)
        ldbGpu,                                    // Precomputed leading dimension B (on GPU)
        ldcGpu,                                    // Precomputed leading dimension C (on GPU)
        lddGpu,                                    // Precomputed leading dimension D (on GPU)
        hostMaxProblemSizesPtr);

    static_assert(Gemm::BaseKernel::ProblemVisitor::kRequiresPrecomputation == false,
        "Grouped GEMM with CUDA Graph cannot use precompution.");
    {
        cutlass::Status status = gemmOp.can_implement(args);
        TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
            "Grouped GEMM cannot be implemented with the given arguments, Error: %s",
            cutlass::cutlassGetStatusString(status));
    }

    at::Tensor workspace;
    void* gemmWorkspace = nullptr;
    size_t const requiredWorkspace = gemmOp.get_workspace_size(args);
    if (requiredWorkspace > 0)
    {
        auto const workspaceTensorOptions = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
        workspace = at::empty({static_cast<int64_t>(requiredWorkspace)}, workspaceTensorOptions);
        gemmWorkspace = workspace.data_ptr();
    }

    cutlass::Status status = gemmOp.initialize(args, gemmWorkspace);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "Failed to initialize grouped GEMM");

    status = gemmOp.run(stream);
    sync_check_cuda_error(stream);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "Failed to execute grouped GEMM");
}

template <int M1, int N1, int K1, int M2, int N2, int K2, int kAlignmentAB, int kAlignmentC, int kStages>
void cudaGraphGroupedGemmType(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu,
    void** ptrBGpu, void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu,
    nvinfer1::DataType dataType, cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, cudaStream_t stream)
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        cudaGraphGroupedGemmTemplate<M1, N1, K1, M2, N2, K2, cutlass::half_t, kAlignmentAB, kAlignmentC, kStages>(
            problemSizesPtr, problemCount, ptrAGpu, ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu,
            hostMaxProblemSizesPtr, stream);
    }
#ifdef ENABLE_BF16
    else if (dataType == nvinfer1::DataType::kBF16)
    {
        cudaGraphGroupedGemmTemplate<M1, N1, K1, M2, N2, K2, cutlass::bfloat16_t, kAlignmentAB, kAlignmentC, kStages>(
            problemSizesPtr, problemCount, ptrAGpu, ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu,
            hostMaxProblemSizesPtr, stream);
    }
#endif
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported data type for CUDA Graph grouped GEMM");
    }
}

void cudaGraphGroupedGemm(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu, void** ptrBGpu,
    void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu, bool isLoraIn,
    nvinfer1::DataType dataType, int minKN, cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, cudaStream_t stream)
{
    if (isLoraIn)
    {
        if (minKN >= 8)
        {
            cudaGraphGroupedGemmType<16, 32, 64, 16, 32, 64, 8, 8, 4>(problemSizesPtr, problemCount, ptrAGpu, ptrBGpu,
                ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, hostMaxProblemSizesPtr, stream);
        }
        else if (minKN >= 4)
        {
            cudaGraphGroupedGemmType<16, 32, 64, 16, 32, 64, 8, 4, 4>(problemSizesPtr, problemCount, ptrAGpu, ptrBGpu,
                ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, hostMaxProblemSizesPtr, stream);
        }
        else if (minKN >= 2)
        {
            cudaGraphGroupedGemmType<16, 32, 64, 16, 32, 64, 8, 2, 2>(problemSizesPtr, problemCount, ptrAGpu, ptrBGpu,
                ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, hostMaxProblemSizesPtr, stream);
        }
        else if (minKN >= 1)
        {
            cudaGraphGroupedGemmType<16, 32, 64, 16, 32, 64, 8, 1, 2>(problemSizesPtr, problemCount, ptrAGpu, ptrBGpu,
                ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, hostMaxProblemSizesPtr, stream);
        }
    }
    else
    {
        if (minKN >= 8)
        {
            cudaGraphGroupedGemmType<32, 128, 32, 32, 32, 32, 8, 8, 4>(problemSizesPtr, problemCount, ptrAGpu, ptrBGpu,
                ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, hostMaxProblemSizesPtr, stream);
        }
        else if (minKN >= 4)
        {
            cudaGraphGroupedGemmType<32, 128, 32, 32, 32, 32, 4, 8, 4>(problemSizesPtr, problemCount, ptrAGpu, ptrBGpu,
                ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, hostMaxProblemSizesPtr, stream);
        }
        else if (minKN >= 2)
        {
            cudaGraphGroupedGemmType<32, 128, 32, 32, 32, 32, 2, 8, 2>(problemSizesPtr, problemCount, ptrAGpu, ptrBGpu,
                ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, hostMaxProblemSizesPtr, stream);
        }
        else
        {
            cudaGraphGroupedGemmType<32, 128, 32, 32, 32, 32, 1, 8, 2>(problemSizesPtr, problemCount, ptrAGpu, ptrBGpu,
                ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, hostMaxProblemSizesPtr, stream);
        }
    }
}

/**
 * Template for CUDA Graph compatible split-K grouped GEMM
 */
template <int M1, int N1, int K1, int M2, int N2, int K2, typename cutlassType, int kAlignmentAB, int kAlignmentC,
    int kStages>
void cudaGraphSplitKGroupedGemmTemplate(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu,
    void** ptrBGpu, void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu,
    int splitKSlices, cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, int64_t* splitKOffsetsGpu, cudaStream_t stream)
{
    using ElementA = cutlassType;
    using ElementB = cutlassType;
    using ElementOutput = cutlassType;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultSplitkGemmGrouped<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, kAlignmentAB, ElementB, LayoutB, cutlass::ComplexTransform::kNone,
        kAlignmentAB, ElementOutput, LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<M1, N1, K1>, cutlass::gemm::GemmShape<M2, N2, K2>, cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<ElementOutput, kAlignmentC, ElementAccumulator,
            ElementAccumulator>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, kStages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

    using Gemm = cutlass::gemm::device::SplitkGemmGrouped<GemmKernel>;

    float alpha = 1.0f;
    float beta = 0.0f;
    typename Gemm::EpilogueOutputOp::Params epilogueOp(alpha, beta);

    auto ptrA = reinterpret_cast<ElementA**>(ptrAGpu);
    auto ptrB = reinterpret_cast<ElementB**>(ptrBGpu);
    auto ptrC = reinterpret_cast<ElementOutput**>(ptrCGpu);
    auto ptrD = reinterpret_cast<ElementOutput**>(ptrDGpu);

    Gemm gemmOp;

    int threadblockCount = Gemm::sufficient(nullptr, problemCount);

    // Setup arguments for split-K grouped GEMM - using precomputed leading dimensions from GPU tensors
    typename Gemm::Arguments args(problemSizesPtr, // GPU problem sizes
        problemCount,                              // Problem count
        threadblockCount,                          // Threadblock count
        epilogueOp,                                // Epilogue operation
        ptrA,                                      // GPU pointer array A
        ptrB,                                      // GPU pointer array B
        ptrC,                                      // GPU pointer array C
        ptrD,                                      // GPU pointer array D
        ldaGpu,                                    // Precomputed leading dimension A (on GPU)
        ldbGpu,                                    // Precomputed leading dimension B (on GPU)
        ldcGpu,                                    // Precomputed leading dimension C (on GPU)
        lddGpu,                                    // Precomputed leading dimension D (on GPU)
        hostMaxProblemSizesPtr,                    // Host problem sizes
        splitKSlices,                              // Split-K factor
        splitKOffsetsGpu);

    {
        cutlass::Status status = gemmOp.can_implement(args);
        TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
            "Split-K grouped GEMM cannot be implemented with the given arguments. Problem count: %d, Split-K slices: "
            "%d, Error: %s",
            problemCount, splitKSlices, cutlass::cutlassGetStatusString(status));
    }

    at::Tensor workspace;
    void* gemmWorkspace = nullptr;
    size_t const requiredWorkspace = gemmOp.get_workspace_size(args);
    if (requiredWorkspace > 0)
    {
        workspace = at::empty(
            {static_cast<int64_t>(requiredWorkspace)}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
        gemmWorkspace = workspace.data_ptr();
    }

    cutlass::Status status = gemmOp.initialize(args, gemmWorkspace);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
        "Failed to initialize split-K grouped GEMM. Problem count: %d, Split-K slices: %d, Error: %s", problemCount,
        splitKSlices, cutlass::cutlassGetStatusString(status));

    status = gemmOp.run(stream);
    sync_check_cuda_error(stream);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
        "Failed to execute split-K grouped GEMM. Problem count: %d, Split-K slices: %d, Error: %s", problemCount,
        splitKSlices, cutlass::cutlassGetStatusString(status));
}

template <int M1, int N1, int K1, int M2, int N2, int K2, int kAlignmentAB, int kAlignmentC, int kStages>
void cudaGraphSplitKGroupedGemmType(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu,
    void** ptrBGpu, void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu,
    nvinfer1::DataType dataType, int splitKSlices, cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr,
    int64_t* splitKOffsetsGpu, cudaStream_t stream)
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        cudaGraphSplitKGroupedGemmTemplate<M1, N1, K1, M2, N2, K2, cutlass::half_t, kAlignmentAB, kAlignmentC, kStages>(
            problemSizesPtr, problemCount, ptrAGpu, ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu,
            splitKSlices, hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
    }
#ifdef ENABLE_BF16
    else if (dataType == nvinfer1::DataType::kBF16)
    {
        cudaGraphSplitKGroupedGemmTemplate<M1, N1, K1, M2, N2, K2, cutlass::bfloat16_t, kAlignmentAB, kAlignmentC,
            kStages>(problemSizesPtr, problemCount, ptrAGpu, ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu,
            splitKSlices, hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
    }
#endif
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported data type for CUDA Graph split-K grouped GEMM");
    }
}

void cudaGraphSplitKGroupedGemm(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu,
    void** ptrBGpu, void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu,
    bool isLoraIn, nvinfer1::DataType dataType, int splitKSlices, int minKN,
    cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, int64_t* splitKOffsetsGpu, cudaStream_t stream)
{
    if (isLoraIn)
    {
        if (minKN >= 8)
        {
            cudaGraphSplitKGroupedGemmType<16, 32, 64, 16, 32, 64, 8, 8, 4>(problemSizesPtr, problemCount, ptrAGpu,
                ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, splitKSlices,
                hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
        }
        else if (minKN >= 4)
        {
            cudaGraphSplitKGroupedGemmType<16, 32, 64, 16, 32, 64, 8, 4, 4>(problemSizesPtr, problemCount, ptrAGpu,
                ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, splitKSlices,
                hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
        }
        else if (minKN >= 2)
        {
            cudaGraphSplitKGroupedGemmType<16, 32, 64, 16, 32, 64, 8, 2, 2>(problemSizesPtr, problemCount, ptrAGpu,
                ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, splitKSlices,
                hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
        }
        else if (minKN >= 1)
        {
            cudaGraphSplitKGroupedGemmType<16, 32, 64, 16, 32, 64, 8, 1, 2>(problemSizesPtr, problemCount, ptrAGpu,
                ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, splitKSlices,
                hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
        }
    }
    else
    {
        if (minKN >= 8)
        {
            cudaGraphSplitKGroupedGemmType<32, 128, 32, 32, 32, 32, 8, 8, 4>(problemSizesPtr, problemCount, ptrAGpu,
                ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, splitKSlices,
                hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
        }
        else if (minKN >= 4)
        {
            cudaGraphSplitKGroupedGemmType<32, 128, 32, 32, 32, 32, 4, 8, 4>(problemSizesPtr, problemCount, ptrAGpu,
                ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, splitKSlices,
                hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
        }
        else if (minKN >= 2)
        {
            cudaGraphSplitKGroupedGemmType<32, 128, 32, 32, 32, 32, 2, 8, 2>(problemSizesPtr, problemCount, ptrAGpu,
                ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, splitKSlices,
                hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
        }
        else
        {
            cudaGraphSplitKGroupedGemmType<32, 128, 32, 32, 32, 32, 1, 8, 2>(problemSizesPtr, problemCount, ptrAGpu,
                ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, dataType, splitKSlices,
                hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
        }
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
