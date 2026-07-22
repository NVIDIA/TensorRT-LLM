/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/device/splitk_gemm_grouped.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm/kernel/default_splitk_gemm_grouped.h"

#ifdef ENABLE_FP8
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "tensorrt_llm/common/memoryUtils.h"
#endif // ENABLE_FP8

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kFp8TmaAlignment = 16;

void checkFp8CudaGraphAlignment(
    cutlass::gemm::GemmCoord const* hostMaxProblemSizesPtr, int problemCount, int minKN, char const* kernelName)
{
    TLLM_CHECK_WITH_INFO(minKN >= kFp8TmaAlignment && minKN % kFp8TmaAlignment == 0,
        "%s requires active LoRA ranks to be multiples of %d elements for 128-bit TMA alignment. "
        "The minimum K/N dimension is %d.",
        kernelName, kFp8TmaAlignment, minKN);

    for (int problemIdx = 0; problemIdx < problemCount; ++problemIdx)
    {
        auto const& problem = hostMaxProblemSizesPtr[problemIdx];
        if (problem.m() == 0 || problem.n() == 0 || problem.k() == 0)
        {
            continue;
        }

        TLLM_CHECK_WITH_INFO(problem.n() % kFp8TmaAlignment == 0 && problem.k() % kFp8TmaAlignment == 0,
            "%s requires GEMM N and K dimensions to be multiples of %d elements for 128-bit TMA alignment. "
            "Max problem %d has M=%d, N=%d, K=%d.",
            kernelName, kFp8TmaAlignment, problemIdx, problem.m(), problem.n(), problem.k());
    }
}

} // namespace

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
    tensorrt_llm::DataType dataType, cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, cudaStream_t stream)
{
    if (dataType == tensorrt_llm::DataType::kHALF)
    {
        cudaGraphGroupedGemmTemplate<M1, N1, K1, M2, N2, K2, cutlass::half_t, kAlignmentAB, kAlignmentC, kStages>(
            problemSizesPtr, problemCount, ptrAGpu, ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu,
            hostMaxProblemSizesPtr, stream);
    }
#ifdef ENABLE_BF16
    else if (dataType == tensorrt_llm::DataType::kBF16)
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

#ifdef ENABLE_FP8
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

// ====================================================================
// FP8 CUDA-graph-compatible grouped GEMM using CUTLASS 3.x.
//
// The CUDA graph variant receives live problem sizes, pointers, and leading
// dimensions already on the GPU. The 3.x API needs cute stride arrays on
// device, so a small setup kernel converts the existing graph parameters into
// the CUTLASS 3.x argument layout on the same stream.
// ====================================================================

template <typename ProblemShape, typename StrideA, typename StrideB, typename StrideC, typename StrideD>
__global__ void fillFp8CudaGraphGroupedGemmParams(cutlass::gemm::GemmCoord const* problemSizesPtr, int problemCount,
    int64_t const* ldaGpu, int64_t const* ldbGpu, int64_t const* ldcGpu, int64_t const* lddGpu,
    typename ProblemShape::UnderlyingProblemShape* problemShapes, StrideA* strideA, StrideB* strideB, StrideC* strideC,
    StrideD* strideD)
{
    int const problemIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (problemIdx >= problemCount)
    {
        return;
    }

    int const m = problemSizesPtr[problemIdx].m();
    int const n = problemSizesPtr[problemIdx].n();
    int const k = problemSizesPtr[problemIdx].k();

    problemShapes[problemIdx] = cute::make_shape(m, n, k);
    strideA[problemIdx]
        = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, static_cast<int>(ldaGpu[problemIdx]), 1));
    strideB[problemIdx]
        = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, static_cast<int>(ldbGpu[problemIdx]), 1));
    strideC[problemIdx]
        = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, static_cast<int>(ldcGpu[problemIdx]), 1));
    strideD[problemIdx]
        = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, static_cast<int>(lddGpu[problemIdx]), 1));
}

void fp8CudaGraphGroupedGemm(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu,
    void** ptrBGpu, void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu,
    cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    using namespace cute;

    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using ElementC = cutlass::float_e4m3_t;
    using ElementD = cutlass::float_e4m3_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int kAlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _2, _1>;

    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;

    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass, TileShape, ClusterShape,
            cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, ElementC, LayoutC*,
            kAlignmentC, ElementD, LayoutD*, kAlignmentD, EpilogueSchedule>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA,
        LayoutA*, kAlignmentA, ElementB, LayoutB*, kAlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    using StrideA = typename GemmKernel::InternalStrideA;
    using StrideB = typename GemmKernel::InternalStrideB;
    using StrideC = typename GemmKernel::InternalStrideC;
    using StrideD = typename GemmKernel::InternalStrideD;

    // Allocate device memory for problem shapes, pointer arrays, and strides.
    auto const tensorOpts = at::TensorOptions().dtype(at::kByte).device(at::kCUDA);

    auto const align16 = [](int64_t bytes) -> int64_t { return (bytes + 15) / 16 * 16; };

    int64_t const szProblemShapes = align16(problemCount * sizeof(typename ProblemShape::UnderlyingProblemShape));
    int64_t const szPtrA = align16(problemCount * sizeof(ElementA const*));
    int64_t const szPtrB = align16(problemCount * sizeof(ElementB const*));
    int64_t const szPtrC = align16(problemCount * sizeof(ElementC const*));
    int64_t const szPtrD = align16(problemCount * sizeof(ElementD*));
    int64_t const szStrideA = align16(problemCount * sizeof(StrideA));
    int64_t const szStrideB = align16(problemCount * sizeof(StrideB));
    int64_t const szStrideC = align16(problemCount * sizeof(StrideC));
    int64_t const szStrideD = align16(problemCount * sizeof(StrideD));

    int64_t const totalParamsBytes
        = szProblemShapes + szPtrA + szPtrB + szPtrC + szPtrD + szStrideA + szStrideB + szStrideC + szStrideD;

    at::Tensor paramsTensor = at::empty({totalParamsBytes}, tensorOpts);
    char* devBase = static_cast<char*>(paramsTensor.data_ptr());

    int64_t devOff = 0;
    auto devPtr = [&devBase, &devOff](int64_t sz) -> void*
    {
        void* p = devBase + devOff;
        devOff += (sz + 15) / 16 * 16;
        return p;
    };

    auto* devProblemShapes = static_cast<typename ProblemShape::UnderlyingProblemShape*>(devPtr(szProblemShapes));
    auto* devPtrA = static_cast<ElementA const**>(devPtr(szPtrA));
    auto* devPtrB = static_cast<ElementB const**>(devPtr(szPtrB));
    auto* devPtrC = static_cast<ElementC const**>(devPtr(szPtrC));
    auto* devPtrD = static_cast<ElementD**>(devPtr(szPtrD));
    auto* devStrideA = static_cast<StrideA*>(devPtr(szStrideA));
    auto* devStrideB = static_cast<StrideB*>(devPtr(szStrideB));
    auto* devStrideC = static_cast<StrideC*>(devPtr(szStrideC));
    auto* devStrideD = static_cast<StrideD*>(devPtr(szStrideD));

    cudaMemcpyAsync(devPtrA, ptrAGpu, problemCount * sizeof(void*), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(devPtrB, ptrBGpu, problemCount * sizeof(void*), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(devPtrC, ptrCGpu, problemCount * sizeof(void*), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(devPtrD, ptrDGpu, problemCount * sizeof(void*), cudaMemcpyDeviceToDevice, stream);

    int constexpr kThreads = 256;
    int const blocks = (problemCount + kThreads - 1) / kThreads;
    fillFp8CudaGraphGroupedGemmParams<ProblemShape, StrideA, StrideB, StrideC, StrideD>
        <<<blocks, kThreads, 0, stream>>>(problemSizesPtr, problemCount, ldaGpu, ldbGpu, ldcGpu, lddGpu,
            devProblemShapes, devStrideA, devStrideB, devStrideC, devStrideD);

    cutlass::KernelHardwareInfo hwInfo;
    hwInfo.device_id = 0;
    cudaGetDevice(&hwInfo.device_id);
    hwInfo.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hwInfo.device_id);

    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
        {problemCount, devProblemShapes, nullptr}, {devPtrA, devStrideA, devPtrB, devStrideB},
        {{1.0f, 0.0f}, devPtrC, devStrideC, devPtrD, devStrideD}, hwInfo};

    Gemm gemm;

    size_t const requiredWorkspace = Gemm::get_workspace_size(arguments);
    at::Tensor workspace;
    void* gemmWorkspace = nullptr;
    if (requiredWorkspace > 0)
    {
        workspace = at::empty({static_cast<int64_t>(requiredWorkspace)}, tensorOpts);
        gemmWorkspace = workspace.data_ptr();
    }

    cutlass::Status status = gemm.can_implement(arguments);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "FP8 CUDA graph grouped GEMM can_implement failed: %s",
        cutlass::cutlassGetStatusString(status));

    status = gemm.initialize(arguments, gemmWorkspace, stream);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "FP8 CUDA graph grouped GEMM initialize failed: %s",
        cutlass::cutlassGetStatusString(status));

    status = gemm.run(stream);
    sync_check_cuda_error(stream);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "FP8 CUDA graph grouped GEMM run failed: %s",
        cutlass::cutlassGetStatusString(status));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
#endif // ENABLE_FP8

void cudaGraphGroupedGemm(cutlass::gemm::GemmCoord* problemSizesPtr, int problemCount, void** ptrAGpu, void** ptrBGpu,
    void** ptrCGpu, void** ptrDGpu, int64_t* ldaGpu, int64_t* ldbGpu, int64_t* ldcGpu, int64_t* lddGpu, bool isLoraIn,
    tensorrt_llm::DataType dataType, int minKN, cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, cudaStream_t stream)
{
#ifdef ENABLE_FP8
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
    if (dataType == tensorrt_llm::DataType::kFP8)
    {
        checkFp8CudaGraphAlignment(hostMaxProblemSizesPtr, problemCount, minKN, "FP8 CUDA graph grouped GEMM");
        fp8CudaGraphGroupedGemm(
            problemSizesPtr, problemCount, ptrAGpu, ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, stream);
        return;
    }
#else
    if (dataType == tensorrt_llm::DataType::kFP8)
    {
        TLLM_CHECK_WITH_INFO(false,
            "FP8 CUDA graph grouped GEMM requires CUTLASS modifiable TMA support (CUDA 12.3+ and Hopper sm90+ "
            "kernels).");
    }
#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
#endif // ENABLE_FP8

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
    tensorrt_llm::DataType dataType, int splitKSlices, cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr,
    int64_t* splitKOffsetsGpu, cudaStream_t stream)
{
    if (dataType == tensorrt_llm::DataType::kHALF)
    {
        cudaGraphSplitKGroupedGemmTemplate<M1, N1, K1, M2, N2, K2, cutlass::half_t, kAlignmentAB, kAlignmentC, kStages>(
            problemSizesPtr, problemCount, ptrAGpu, ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu,
            splitKSlices, hostMaxProblemSizesPtr, splitKOffsetsGpu, stream);
    }
#ifdef ENABLE_BF16
    else if (dataType == tensorrt_llm::DataType::kBF16)
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
    bool isLoraIn, tensorrt_llm::DataType dataType, int splitKSlices, int minKN,
    cutlass::gemm::GemmCoord* hostMaxProblemSizesPtr, int64_t* splitKOffsetsGpu, cudaStream_t stream)
{
#ifdef ENABLE_FP8
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
    if (dataType == tensorrt_llm::DataType::kFP8)
    {
        // Reuse the non-split-K fp8 path; CUTLASS 3.x cooperative schedule handles large-K efficiently.
        checkFp8CudaGraphAlignment(hostMaxProblemSizesPtr, problemCount, minKN, "FP8 CUDA graph split-K grouped GEMM");
        fp8CudaGraphGroupedGemm(
            problemSizesPtr, problemCount, ptrAGpu, ptrBGpu, ptrCGpu, ptrDGpu, ldaGpu, ldbGpu, ldcGpu, lddGpu, stream);
        return;
    }
#else
    if (dataType == tensorrt_llm::DataType::kFP8)
    {
        TLLM_CHECK_WITH_INFO(false,
            "FP8 CUDA graph split-K grouped GEMM requires CUTLASS modifiable TMA support (CUDA 12.3+ and Hopper sm90+ "
            "kernels).");
    }
#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
#endif // ENABLE_FP8

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
