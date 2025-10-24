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

namespace tensorrt_llm
{
namespace kernels
{

/**
 * Template for CUDA Graph compatible grouped GEMM that directly uses GPU tensors
 */
template <int M1, int N1, int K1, int M2, int N2, int K2, typename cutlassType, int kAlignmentAB, int kAlignmentC,
    int kStages>
void cuda_graph_grouped_gemm_template(cutlass::gemm::GemmCoord* problem_sizes_ptr, int problem_count, void** ptrA_gpu,
    void** ptrB_gpu, void** ptrC_gpu, void** ptrD_gpu, int64_t* lda_gpu, int64_t* ldb_gpu, int64_t* ldc_gpu,
    int64_t* ldd_gpu, cutlass::gemm::GemmCoord* host_max_problem_sizes_ptr, cudaStream_t stream)
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
    typename Gemm::EpilogueOutputOp::Params epilogue_op(alpha, beta);

    auto ptr_A = reinterpret_cast<ElementA**>(ptrA_gpu);
    auto ptr_B = reinterpret_cast<ElementB**>(ptrB_gpu);
    auto ptr_C = reinterpret_cast<ElementOutput**>(ptrC_gpu);
    auto ptr_D = reinterpret_cast<ElementOutput**>(ptrD_gpu);

    Gemm gemm_op;

    int threadblock_count = Gemm::sufficient(nullptr, problem_count);

    typename Gemm::Arguments args(problem_sizes_ptr, // GPU problem sizes
        problem_count,                               // Problem count
        threadblock_count,                           // Threadblock count
        epilogue_op,                                 // Epilogue operation
        ptr_A,                                       // GPU pointer array A
        ptr_B,                                       // GPU pointer array B
        ptr_C,                                       // GPU pointer array C (can be nullptr)
        ptr_D,                                       // GPU pointer array D
        lda_gpu,                                     // Precomputed leading dimension A (on GPU)
        ldb_gpu,                                     // Precomputed leading dimension B (on GPU)
        ldc_gpu,                                     // Precomputed leading dimension C (on GPU)
        ldd_gpu,                                     // Precomputed leading dimension D (on GPU)
        host_max_problem_sizes_ptr);

    static_assert(Gemm::BaseKernel::ProblemVisitor::kRequiresPrecomputation == false,
        "Grouped GEMM with CUDA Graph cannot use precompution.");
    {
        cutlass::Status status = gemm_op.can_implement(args);
        TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
            "Grouped GEMM cannot be implemented with the given arguments, Error: %s",
            cutlass::cutlassGetStatusString(status));
    }

    at::Tensor workspace;
    void* gemm_workspace = nullptr;
    size_t const required_workspace = gemm_op.get_workspace_size(args);
    if (required_workspace > 0)
    {
        auto const workspaceTensorOptions = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);
        workspace = at::empty({static_cast<int64_t>(required_workspace)}, workspaceTensorOptions);
        gemm_workspace = workspace.data_ptr();
    }

    cutlass::Status status = gemm_op.initialize(args, gemm_workspace);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "Failed to initialize grouped GEMM");

    status = gemm_op.run(stream);
    sync_check_cuda_error(stream);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "Failed to execute grouped GEMM");
}

template <int M1, int N1, int K1, int M2, int N2, int K2, int kAlignmentAB, int kAlignmentC, int kStages>
void cuda_graph_grouped_gemm_type(cutlass::gemm::GemmCoord* problem_sizes_ptr, int problem_count, void** ptrA_gpu,
    void** ptrB_gpu, void** ptrC_gpu, void** ptrD_gpu, int64_t* lda_gpu, int64_t* ldb_gpu, int64_t* ldc_gpu,
    int64_t* ldd_gpu, nvinfer1::DataType dataType, cutlass::gemm::GemmCoord* host_max_problem_sizes_ptr,
    cudaStream_t stream)
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        cuda_graph_grouped_gemm_template<M1, N1, K1, M2, N2, K2, cutlass::half_t, kAlignmentAB, kAlignmentC, kStages>(
            problem_sizes_ptr, problem_count, ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu,
            ldd_gpu, host_max_problem_sizes_ptr, stream);
    }
#ifdef ENABLE_BF16
    else if (dataType == nvinfer1::DataType::kBF16)
    {
        cuda_graph_grouped_gemm_template<M1, N1, K1, M2, N2, K2, cutlass::bfloat16_t, kAlignmentAB, kAlignmentC,
            kStages>(problem_sizes_ptr, problem_count, ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu,
            ldc_gpu, ldd_gpu, host_max_problem_sizes_ptr, stream);
    }
#endif
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported data type for CUDA Graph grouped GEMM");
    }
}

void cuda_graph_grouped_gemm(cutlass::gemm::GemmCoord* problem_sizes_ptr, int problem_count, void** ptrA_gpu,
    void** ptrB_gpu, void** ptrC_gpu, void** ptrD_gpu, int64_t* lda_gpu, int64_t* ldb_gpu, int64_t* ldc_gpu,
    int64_t* ldd_gpu, bool isLoraIn, nvinfer1::DataType dataType, int minKN,
    cutlass::gemm::GemmCoord* host_max_problem_sizes_ptr, cudaStream_t stream)
{
    if (isLoraIn)
    {
        if (minKN >= 8)
        {
            cuda_graph_grouped_gemm_type<16, 32, 64, 16, 32, 64, 8, 8, 4>(problem_sizes_ptr, problem_count, ptrA_gpu,
                ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, host_max_problem_sizes_ptr,
                stream);
        }
        else if (minKN >= 4)
        {
            cuda_graph_grouped_gemm_type<16, 32, 64, 16, 32, 64, 8, 4, 4>(problem_sizes_ptr, problem_count, ptrA_gpu,
                ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, host_max_problem_sizes_ptr,
                stream);
        }
        else if (minKN >= 2)
        {
            cuda_graph_grouped_gemm_type<16, 32, 64, 16, 32, 64, 8, 2, 2>(problem_sizes_ptr, problem_count, ptrA_gpu,
                ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, host_max_problem_sizes_ptr,
                stream);
        }
        else if (minKN >= 1)
        {
            cuda_graph_grouped_gemm_type<16, 32, 64, 16, 32, 64, 8, 1, 2>(problem_sizes_ptr, problem_count, ptrA_gpu,
                ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, host_max_problem_sizes_ptr,
                stream);
        }
    }
    else
    {
        if (minKN >= 8)
        {
            cuda_graph_grouped_gemm_type<32, 128, 32, 32, 32, 32, 8, 8, 4>(problem_sizes_ptr, problem_count, ptrA_gpu,
                ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, host_max_problem_sizes_ptr,
                stream);
        }
        else if (minKN >= 4)
        {
            cuda_graph_grouped_gemm_type<32, 128, 32, 32, 32, 32, 4, 8, 4>(problem_sizes_ptr, problem_count, ptrA_gpu,
                ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, host_max_problem_sizes_ptr,
                stream);
        }
        else if (minKN >= 2)
        {
            cuda_graph_grouped_gemm_type<32, 128, 32, 32, 32, 32, 2, 8, 2>(problem_sizes_ptr, problem_count, ptrA_gpu,
                ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, host_max_problem_sizes_ptr,
                stream);
        }
        else
        {
            cuda_graph_grouped_gemm_type<32, 128, 32, 32, 32, 32, 1, 8, 2>(problem_sizes_ptr, problem_count, ptrA_gpu,
                ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, host_max_problem_sizes_ptr,
                stream);
        }
    }
}

/**
 * Template for CUDA Graph compatible split-K grouped GEMM
 */
template <int M1, int N1, int K1, int M2, int N2, int K2, typename cutlassType, int kAlignmentAB, int kAlignmentC,
    int kStages>
void cuda_graph_splitk_grouped_gemm_template(cutlass::gemm::GemmCoord* problem_sizes_ptr, int problem_count,
    void** ptrA_gpu, void** ptrB_gpu, void** ptrC_gpu, void** ptrD_gpu, int64_t* lda_gpu, int64_t* ldb_gpu,
    int64_t* ldc_gpu, int64_t* ldd_gpu, int splitKSlices, cutlass::gemm::GemmCoord* host_max_problem_sizes_ptr,
    int64_t* splitk_offsets_gpu, cudaStream_t stream)
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
    typename Gemm::EpilogueOutputOp::Params epilogue_op(alpha, beta);

    auto ptr_A = reinterpret_cast<ElementA**>(ptrA_gpu);
    auto ptr_B = reinterpret_cast<ElementB**>(ptrB_gpu);
    auto ptr_C = reinterpret_cast<ElementOutput**>(ptrC_gpu);
    auto ptr_D = reinterpret_cast<ElementOutput**>(ptrD_gpu);

    Gemm gemm_op;

    int threadblock_count = Gemm::sufficient(nullptr, problem_count);

    // Setup arguments for split-K grouped GEMM - using precomputed leading dimensions from GPU tensors
    typename Gemm::Arguments args(problem_sizes_ptr, // GPU problem sizes
        problem_count,                               // Problem count
        threadblock_count,                           // Threadblock count
        epilogue_op,                                 // Epilogue operation
        ptr_A,                                       // GPU pointer array A
        ptr_B,                                       // GPU pointer array B
        ptr_C,                                       // GPU pointer array C
        ptr_D,                                       // GPU pointer array D
        lda_gpu,                                     // Precomputed leading dimension A (on GPU)
        ldb_gpu,                                     // Precomputed leading dimension B (on GPU)
        ldc_gpu,                                     // Precomputed leading dimension C (on GPU)
        ldd_gpu,                                     // Precomputed leading dimension D (on GPU)
        host_max_problem_sizes_ptr,                  // Host problem sizes
        splitKSlices,                                // Split-K factor
        splitk_offsets_gpu);

    {
        cutlass::Status status = gemm_op.can_implement(args);
        TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
            "Split-K grouped GEMM cannot be implemented with the given arguments. Problem count: %d, Split-K slices: "
            "%d, Error: %s",
            problem_count, splitKSlices, cutlass::cutlassGetStatusString(status));
    }

    at::Tensor workspace;
    void* gemm_workspace = nullptr;
    size_t const required_workspace = gemm_op.get_workspace_size(args);
    if (required_workspace > 0)
    {
        workspace = at::empty(
            {static_cast<int64_t>(required_workspace)}, at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
        gemm_workspace = workspace.data_ptr();
    }

    cutlass::Status status = gemm_op.initialize(args, gemm_workspace);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
        "Failed to initialize split-K grouped GEMM. Problem count: %d, Split-K slices: %d, Error: %s", problem_count,
        splitKSlices, cutlass::cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    sync_check_cuda_error(stream);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
        "Failed to execute split-K grouped GEMM. Problem count: %d, Split-K slices: %d, Error: %s", problem_count,
        splitKSlices, cutlass::cutlassGetStatusString(status));
}

template <int M1, int N1, int K1, int M2, int N2, int K2, int kAlignmentAB, int kAlignmentC, int kStages>
void cuda_graph_splitk_grouped_gemm_type(cutlass::gemm::GemmCoord* problem_sizes_ptr, int problem_count,
    void** ptrA_gpu, void** ptrB_gpu, void** ptrC_gpu, void** ptrD_gpu, int64_t* lda_gpu, int64_t* ldb_gpu,
    int64_t* ldc_gpu, int64_t* ldd_gpu, nvinfer1::DataType dataType, int splitKSlices,
    cutlass::gemm::GemmCoord* host_max_problem_sizes_ptr, int64_t* splitk_offsets_gpu, cudaStream_t stream)
{
    if (dataType == nvinfer1::DataType::kHALF)
    {
        cuda_graph_splitk_grouped_gemm_template<M1, N1, K1, M2, N2, K2, cutlass::half_t, kAlignmentAB, kAlignmentC,
            kStages>(problem_sizes_ptr, problem_count, ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu,
            ldc_gpu, ldd_gpu, splitKSlices, host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
    }
#ifdef ENABLE_BF16
    else if (dataType == nvinfer1::DataType::kBF16)
    {
        cuda_graph_splitk_grouped_gemm_template<M1, N1, K1, M2, N2, K2, cutlass::bfloat16_t, kAlignmentAB, kAlignmentC,
            kStages>(problem_sizes_ptr, problem_count, ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu,
            ldc_gpu, ldd_gpu, splitKSlices, host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
    }
#endif
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported data type for CUDA Graph split-K grouped GEMM");
    }
}

void cuda_graph_splitk_grouped_gemm(cutlass::gemm::GemmCoord* problem_sizes_ptr, int problem_count, void** ptrA_gpu,
    void** ptrB_gpu, void** ptrC_gpu, void** ptrD_gpu, int64_t* lda_gpu, int64_t* ldb_gpu, int64_t* ldc_gpu,
    int64_t* ldd_gpu, bool isLoraIn, nvinfer1::DataType dataType, int splitKSlices, int minKN,
    cutlass::gemm::GemmCoord* host_max_problem_sizes_ptr, int64_t* splitk_offsets_gpu, cudaStream_t stream)
{
    if (isLoraIn)
    {
        if (minKN >= 8)
        {
            cuda_graph_splitk_grouped_gemm_type<16, 32, 64, 16, 32, 64, 8, 8, 4>(problem_sizes_ptr, problem_count,
                ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, splitKSlices,
                host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
        }
        else if (minKN >= 4)
        {
            cuda_graph_splitk_grouped_gemm_type<16, 32, 64, 16, 32, 64, 8, 4, 4>(problem_sizes_ptr, problem_count,
                ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, splitKSlices,
                host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
        }
        else if (minKN >= 2)
        {
            cuda_graph_splitk_grouped_gemm_type<16, 32, 64, 16, 32, 64, 8, 2, 2>(problem_sizes_ptr, problem_count,
                ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, splitKSlices,
                host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
        }
        else if (minKN >= 1)
        {
            cuda_graph_splitk_grouped_gemm_type<16, 32, 64, 16, 32, 64, 8, 1, 2>(problem_sizes_ptr, problem_count,
                ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, splitKSlices,
                host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
        }
    }
    else
    {
        if (minKN >= 8)
        {
            cuda_graph_splitk_grouped_gemm_type<32, 128, 32, 32, 32, 32, 8, 8, 4>(problem_sizes_ptr, problem_count,
                ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, splitKSlices,
                host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
        }
        else if (minKN >= 4)
        {
            cuda_graph_splitk_grouped_gemm_type<32, 128, 32, 32, 32, 32, 4, 8, 4>(problem_sizes_ptr, problem_count,
                ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, splitKSlices,
                host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
        }
        else if (minKN >= 2)
        {
            cuda_graph_splitk_grouped_gemm_type<32, 128, 32, 32, 32, 32, 2, 8, 2>(problem_sizes_ptr, problem_count,
                ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, splitKSlices,
                host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
        }
        else
        {
            cuda_graph_splitk_grouped_gemm_type<32, 128, 32, 32, 32, 32, 1, 8, 2>(problem_sizes_ptr, problem_count,
                ptrA_gpu, ptrB_gpu, ptrC_gpu, ptrD_gpu, lda_gpu, ldb_gpu, ldc_gpu, ldd_gpu, dataType, splitKSlices,
                host_max_problem_sizes_ptr, splitk_offsets_gpu, stream);
        }
    }
}

} // namespace kernels
} // namespace tensorrt_llm
