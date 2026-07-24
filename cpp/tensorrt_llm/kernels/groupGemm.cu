/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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

#ifdef ENABLE_FP8
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#endif // ENABLE_FP8

#include "groupGemm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int kFp8TmaAlignment = 16;

void checkFp8GroupedGemmAlignment(std::vector<cutlass::gemm::GemmCoord> const& problemSizes, char const* kernelName)
{
    for (size_t problemIdx = 0; problemIdx < problemSizes.size(); ++problemIdx)
    {
        auto const& problem = problemSizes[problemIdx];
        if (problem.m() == 0 || problem.n() == 0 || problem.k() == 0)
        {
            continue;
        }

        TLLM_CHECK_WITH_INFO(problem.n() % kFp8TmaAlignment == 0 && problem.k() % kFp8TmaAlignment == 0,
            "%s requires GEMM N and K dimensions to be multiples of %d elements for 128-bit TMA alignment. "
            "Problem %zu has M=%d, N=%d, K=%d.",
            kernelName, kFp8TmaAlignment, problemIdx, problem.m(), problem.n(), problem.k());
    }
}

} // namespace

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

int64_t getFp8GroupedGemmParamsWorkSpaceSize(int64_t problemCount)
{
    // The FP8 grouped GEMM (CUTLASS 3.x) stores on device:
    //   - problem shapes:  problemCount * 12 bytes (Shape<int,int,int>)
    //   - 4 pointer arrays: 4 * problemCount * 8 bytes
    //   - 4 stride arrays:  4 * problemCount * 16 bytes (conservative upper bound for cute strides)
    // All 16-byte aligned.
    auto align16 = [](int64_t bytes) -> int64_t { return (bytes + 15) / 16 * 16; };

    int64_t const szProblemShapes = align16(problemCount * 12);
    int64_t const szPtrs = 4 * align16(problemCount * 8);
    int64_t const szStrides = 4 * align16(problemCount * 16);

    return szProblemShapes + szPtrs + szStrides;
}

template <int M1, int N1, int K1, int M2, int N2, int K2, typename cutlassType, int kAlignmentAB, int kAlignmentC,
    int kStages>
void groupedGemm_(std::vector<cutlass::gemm::GemmCoord> problem_sizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkSpace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkspaceSize,
    tensorrt_llm::DataType dataType, cudaStream_t stream)
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
    tensorrt_llm::DataType dataType, cudaStream_t stream)
{
    if (dataType == tensorrt_llm::DataType::kHALF)
    {
        groupedGemm_<M1, N1, K1, M2, N2, K2, cutlass::half_t, kAlignmentAB, kAlignmentC, kStages>(problem_sizes, ptrA,
            ptrB, ptrC, ptrD, gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize, dataType,
            stream);
    }
    else if (dataType == tensorrt_llm::DataType::kFLOAT)
    {
        TLLM_CHECK_WITH_INFO(false, "not support float input/output");
    }
#ifdef ENABLE_BF16
    else if (dataType == tensorrt_llm::DataType::kBF16)
    {
        groupedGemm_<M1, N1, K1, M2, N2, K2, cutlass::bfloat16_t, kAlignmentAB, kAlignmentC, kStages>(problem_sizes,
            ptrA, ptrB, ptrC, ptrD, gemmParamsWorkSpace, gemmParamsWorkSpaceSize, gemmWorkSpace, gemmWorkspaceSize,
            dataType, stream);
    }
#endif
}

#ifdef ENABLE_FP8

// ====================================================================
// FP8 grouped GEMM using CUTLASS 3.x collective API (Hopper sm90+).
//
// The legacy CUTLASS 2.x DefaultGemmGrouped does NOT support fp8 element
// types. This implementation uses the CUTLASS 3.x CollectiveBuilder and
// GemmUniversal kernel with PtrArray schedules, following the pattern of
// cutlass/examples/57_hopper_grouped_gemm.
//
// Layout convention:
//   A: RowMajor    (M x K)
//   B: ColumnMajor (K x N, stored column-major)
//   D: RowMajor    (M x N, output; C is aliased to D for beta=0)
//
// The 3.x grouped API uses pointer-array arguments: arrays of per-group
// pointers and strides on the GPU. We build these on the host and copy
// them to device memory via the provided workspace.
// ====================================================================

#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)

void fp8GroupedGemm(std::vector<cutlass::gemm::GemmCoord> const& problemSizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkSpace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkspaceSize,
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

    // 128-bit TMA alignment: 128 bits / 8 bits per fp8 element = 16 elements
    static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int kAlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int kAlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using ArchTag = cutlass::arch::Sm90;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    // Tile and cluster shapes chosen for fp8 on Hopper.
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _2, _1>;

    // Kernel and epilogue schedule for fp8 grouped GEMM with PtrArray TMA.
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

    int const problemCount = static_cast<int>(problemSizes.size());

    // Build host-side arrays for problem shapes, pointers, and strides.
    std::vector<typename ProblemShape::UnderlyingProblemShape> problemShapesHost(problemCount);
    std::vector<ElementA const*> ptrAHost(problemCount);
    std::vector<ElementB const*> ptrBHost(problemCount);
    std::vector<ElementC const*> ptrCHost(problemCount);
    std::vector<ElementD*> ptrDHost(problemCount);
    std::vector<StrideA> strideAHost(problemCount);
    std::vector<StrideB> strideBHost(problemCount);
    std::vector<StrideC> strideCHost(problemCount);
    std::vector<StrideD> strideDHost(problemCount);

    for (int i = 0; i < problemCount; ++i)
    {
        auto const& coord = problemSizes[i];
        int const m = coord.m();
        int const n = coord.n();
        int const k = coord.k();

        problemShapesHost[i] = cute::make_shape(m, n, k);
        ptrAHost[i] = static_cast<ElementA const*>(ptrA[i]);
        ptrBHost[i] = static_cast<ElementB const*>(ptrB[i]);
        ptrCHost[i] = static_cast<ElementC const*>(ptrC[i]);
        ptrDHost[i] = static_cast<ElementD*>(ptrD[i]);

        strideAHost[i] = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
        strideBHost[i] = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
        strideCHost[i] = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
        strideDHost[i] = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
    }

    // Compute workspace layout sizes (all 16-byte aligned).
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
    TLLM_CHECK_WITH_INFO(totalParamsBytes <= gemmParamsWorkSpaceSize,
        "FP8 grouped GEMM params workspace too small: need %ld, have %ld", totalParamsBytes, gemmParamsWorkSpaceSize);

    // Stage everything into a single host buffer, then memcpy to device.
    std::vector<char> hostBuf(totalParamsBytes);
    char* cursor = hostBuf.data();

    auto copyToHostBuf = [&cursor](void const* src, int64_t bytes)
    {
        std::memcpy(cursor, src, bytes);
        char* base = cursor;
        cursor += (bytes + 15) / 16 * 16;
        return base;
    };

    copyToHostBuf(problemShapesHost.data(), problemCount * sizeof(typename ProblemShape::UnderlyingProblemShape));
    copyToHostBuf(ptrAHost.data(), problemCount * sizeof(ElementA const*));
    copyToHostBuf(ptrBHost.data(), problemCount * sizeof(ElementB const*));
    copyToHostBuf(ptrCHost.data(), problemCount * sizeof(ElementC const*));
    copyToHostBuf(ptrDHost.data(), problemCount * sizeof(ElementD*));
    copyToHostBuf(strideAHost.data(), problemCount * sizeof(StrideA));
    copyToHostBuf(strideBHost.data(), problemCount * sizeof(StrideB));
    copyToHostBuf(strideCHost.data(), problemCount * sizeof(StrideC));
    copyToHostBuf(strideDHost.data(), problemCount * sizeof(StrideD));

    tensorrt_llm::common::cudaAutoCpy(reinterpret_cast<int8_t*>(gemmParamsWorkSpace),
        reinterpret_cast<int8_t const*>(hostBuf.data()), totalParamsBytes, stream);

    // Compute device pointers from the workspace base.
    char* devBase = static_cast<char*>(gemmParamsWorkSpace);
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

    // Hardware info for kernel launch.
    cutlass::KernelHardwareInfo hwInfo;
    hwInfo.device_id = 0;
    cudaGetDevice(&hwInfo.device_id);
    hwInfo.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hwInfo.device_id);

    // Build CUTLASS 3.x arguments. alpha=1, beta=0 for LoRA GEMM.
    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
        {problemCount, devProblemShapes, nullptr}, {devPtrA, devStrideA, devPtrB, devStrideB},
        {{1.0f, 0.0f}, devPtrC, devStrideC, devPtrD, devStrideD}, hwInfo};

    Gemm gemm;

    size_t const workspaceSize = Gemm::get_workspace_size(arguments);
    TLLM_CHECK_WITH_INFO(static_cast<int64_t>(workspaceSize) <= gemmWorkspaceSize,
        "FP8 grouped GEMM workspace too small: need %lu, have %ld", workspaceSize, gemmWorkspaceSize);

    cutlass::Status status = gemm.can_implement(arguments);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "FP8 grouped GEMM can_implement failed: %s",
        cutlass::cutlassGetStatusString(status));

    status = gemm.initialize(arguments, gemmWorkSpace, stream);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "FP8 grouped GEMM initialize failed: %s",
        cutlass::cutlassGetStatusString(status));

    status = gemm.run(stream);
    TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess, "FP8 grouped GEMM run failed: %s",
        cutlass::cutlassGetStatusString(status));

    sync_check_cuda_error(stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED

#endif // ENABLE_FP8

void groupedGemm(std::vector<cutlass::gemm::GemmCoord> problem_sizes, std::vector<void*> const& ptrA,
    std::vector<void*> const& ptrB, std::vector<void*> const& ptrC, std::vector<void*> const& ptrD,
    void* gemmParamsWorkSpace, int64_t gemmParamsWorkSpaceSize, void* gemmWorkSpace, int64_t gemmWorkspaceSize,
    bool isLoraIn, tensorrt_llm::DataType dataType, int minKN, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start, isLoraIn: %d, minKN = %d", __PRETTY_FUNCTION__, static_cast<int>(isLoraIn), minKN);

#ifdef ENABLE_FP8
#if defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
    if (dataType == tensorrt_llm::DataType::kFP8)
    {
        checkFp8GroupedGemmAlignment(problem_sizes, "FP8 grouped GEMM");

        fp8GroupedGemm(problem_sizes, ptrA, ptrB, ptrC, ptrD, gemmParamsWorkSpace, gemmParamsWorkSpaceSize,
            gemmWorkSpace, gemmWorkspaceSize, stream);
        return;
    }
#else
    if (dataType == tensorrt_llm::DataType::kFP8)
    {
        TLLM_CHECK_WITH_INFO(
            false, "FP8 grouped GEMM requires CUTLASS modifiable TMA support (CUDA 12.3+ and Hopper sm90+ kernels).");
    }
#endif // CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED
#endif // ENABLE_FP8

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

TRTLLM_NAMESPACE_END
