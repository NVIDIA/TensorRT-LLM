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

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__

#include "cutlass/cutlass.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

#include "63_hopper_gemm_with_weight_prefetch/collective/builder.hpp"
#include "63_hopper_gemm_with_weight_prefetch/collective/dispatch_policy_extra.hpp"
#include "63_hopper_gemm_with_weight_prefetch/kernel/sm90_gemm_tma_warpspecialized_with_prefetch.hpp"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC__

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"

#include "../include/low_latency_gemm.h"
#include "cutlass/float8.h"
#include "cutlass_extensions/gemm_configs.h"
#include <cstddef>
#include <stdexcept>

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{

namespace kernels
{

namespace cutlass_kernels
{
using ConfigType = CutlassLowLatencyFp8GemmRunnerInterface::ConfigType;

// only support fp8 TN gemm
// batch GEMM
// return require_workspace_size
template <typename T, typename arch, typename ThreadblockShape, typename ClusterShape_, typename KernelSchedule_>
size_t genericFp8LowLatencyGemmKernelLauncherSm90(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float alpha,
    float beta, T const* C, T* D, int m, int n, int k, int b, float pdl_overlap_ratio, float prefetch_ratio,
    ConfigType config, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

#ifdef COMPILE_HOPPER_TMA_GEMMS
    using namespace cute;

    // A matrix configuration
    using ElementA = cutlass::float_e4m3_t;            // Element type for A matrix operand
    using LayoutA = cutlass::layout::RowMajor;         // Layout type for A matrix operand
    constexpr int AlignmentA
        = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A matrix in units of
                                                       // elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = cutlass::float_e4m3_t;            // Element type for B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;      // Layout type for B matrix operand
    constexpr int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B matrix in units of
                                                       // elements (up to 16 bytes)

    using ElementOutput_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementOutput =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, ElementOutput_>::type;
#else
    using ElementOutput = ElementOutput_;
#endif

    // C matrix configuration
    using ElementC = ElementOutput;                    // Element type for C and D matrix operands
    using LayoutC = cutlass::layout::ColumnMajor;      // Layout type for C and D matrix operands
    constexpr int AlignmentC
        = 128 / cutlass::sizeof_bits<ElementC>::value; // Memory access granularity/alignment of C matrix in units of
                                                       // elements (up to 16 bytes)

    // D matrix configuration
    using ElementD = ElementC;
    using LayoutD = LayoutC;
    constexpr int AlignmentD = AlignmentC;

    // / Core kernel configurations
    using ElementAccumulator = float;    // Element type for internal accumulation
    using ElementCompute = float;        // Element type for epilogue computation
    using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
    using TileShape = ThreadblockShape;                   // Threadblock-level tile size
    using ClusterShape = ClusterShape_;                   // Shape of the threadblocks in a cluster
    using KernelSchedule = KernelSchedule_;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
        TileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementCompute, ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD, EpilogueSchedule>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA,
        LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, // Indicates ProblemShape
        CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Extract information from Gemm kernel.
    using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
    using ElementScalar = typename EpilogueOutputOp::ElementScalar;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    using LayoutScalar = cutlass::layout::PackedVectorLayout;

    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, b));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, b));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, b));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, b));

    auto a_coord = cutlass::make_Coord(m * b, k);
    auto c_coord = cutlass::make_Coord(m * b, n);
    auto b_coord = cutlass::make_Coord(k, n * b);

    typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, b},
        {reinterpret_cast<cutlass::float_e4m3_t const*>(A), stride_A, reinterpret_cast<cutlass::float_e4m3_t const*>(B),
            stride_B},
        {{}, // epilogue.thread
            reinterpret_cast<ElementOutput const*>(C), stride_C, reinterpret_cast<ElementOutput*>(D), stride_D}};

    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha = alpha;
    fusion_args.beta = beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;

    //

    //   fusion_args.beta_ptr = scalar_beta.device_data();

    arguments.mainloop.overlap_ratio = pdl_overlap_ratio;
    arguments.mainloop.prefetch_ratio = prefetch_ratio;

    Gemm gemm;
    int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));
    static int mMaxSmemSize = tk::getMaxSharedMemoryPerBlockOptin();
    if (smem_size > mMaxSmemSize)
    {
        std::string errMsg = "SMEM size exceeds maximum allowed. Required " + std::to_string(smem_size) + ", got "
            + std::to_string(mMaxSmemSize);
        throw std::runtime_error("[TensorRT LLM Error][Fp8LowLatencyGemm Runner] " + errMsg);
    }

    // Return workspace size
    if (!A && !B && !C && !D)
    {
        return Gemm::get_workspace_size(arguments);
    }

    if (Gemm::get_workspace_size(arguments) > workspaceBytes)
    {
        std::string errMsg("Requested workspace size insufficient. Required "
            + std::to_string(gemm.get_workspace_size(arguments)) + ", got " + std::to_string(workspaceBytes));
        throw std::runtime_error("[TensorRT LLM Error][Fp8LowLatencyGemm Runner] " + errMsg);
    }

    auto can_implement = gemm.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Fp8LowLatencyGemm cutlass kernel not implemented given the params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[TensorRT LLM Error][Fp8LowLatencyGemm Runner] " + errMsg);
    }

    auto initStatus = gemm.initialize(arguments, workspacePtr);
    if (initStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Failed to initialize. Error: " + std::string(cutlassGetStatusString(initStatus));
        throw std::runtime_error("[TensorRT LLM Error][Fp8LowLatencyGemm Runner] " + errMsg);
    }

    auto runStatus = gemm.run(stream, nullptr, pdl_overlap_ratio >= 0);
    if (runStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg = "Failed to run gemm. Error: " + std::string(cutlassGetStatusString(runStatus));
        throw std::runtime_error("[TensorRT LLM Error][Fp8LowLatencyGemm Runner] " + errMsg);
    }
    return gemm.get_workspace_size(arguments);
#else  // COMPILE_HOPPER_TMA_GEMMS
    throw std::runtime_error(
        "[TensorRT LLM Error][genericFp8LowLatencyGemmKernelLauncherSm90] Please recompile with support for hopper by "
        "passing 90-real as an arch to build_wheel.py.");
#endif // COMPILE_HOPPER_TMA_GEMMS
}

// TODO:
template <typename T, typename arch, typename ThreadblockShape, typename ClusterShape>
size_t dispatchLowLatencyGemmCultassKernelSchedSm90(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float alpha,
    float beta, T const* C, T* D, int m, int n, int k, int b, float pdl_overlap_ratio, float prefetch_ratio,
    ConfigType config, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
{

    using WS_PREFETECH_TYPE = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccumWithPrefetch;
    using WS_PREFETECH_SPLIT_WARP_TYPE = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccumWithPrefetchAndSplitDMA;
    switch (config.kernel_schedule)
    {

    case KernelScheduleType::WS_PREFETECH:
        return genericFp8LowLatencyGemmKernelLauncherSm90<T, arch, ThreadblockShape, ClusterShape, WS_PREFETECH_TYPE>(A,
            B, alpha, beta, C, D, m, n, k, b, pdl_overlap_ratio, prefetch_ratio, config, workspacePtr, workspaceBytes,
            stream);
        break;
    case KernelScheduleType::WS_SPLIT_PREFETECH:
        return genericFp8LowLatencyGemmKernelLauncherSm90<T, arch, ThreadblockShape, ClusterShape,
            WS_PREFETECH_SPLIT_WARP_TYPE>(A, B, alpha, beta, C, D, m, n, k, b, pdl_overlap_ratio, prefetch_ratio,
            config, workspacePtr, workspaceBytes, stream);
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][CutlassLowLatencyFp8GemmRunner][dispatchLowLatencyGemmCultassKernelSchedSm90] Config "
            "is "
            "invalid for low latency fp8 gemm");
        break;
    }
}

template <typename T, typename arch, typename ThreadblockShape>
size_t dispatchLowLatencyGemmClusterShapeSm90(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float alpha, float beta,
    T const* C, T* D, int m, int n, int k, int b, float pdl_overlap_ratio, float prefetch_ratio, ConfigType gemmConfig,
    char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    using namespace cute;

    switch (gemmConfig.cutlass_gemm_config.cluster_shape)
    {
    case tkc::ClusterShape::ClusterShape_1x1x1:
        return dispatchLowLatencyGemmCultassKernelSchedSm90<T, arch, ThreadblockShape, Shape<_1, _1, _1>>(A, B, alpha,
            beta, C, D, m, n, k, b, pdl_overlap_ratio, prefetch_ratio, gemmConfig, workspacePtr, workspaceBytes,
            stream);
        break;

    case tkc::ClusterShape::ClusterShape_2x1x1:
        return dispatchLowLatencyGemmCultassKernelSchedSm90<T, arch, ThreadblockShape, Shape<_2, _1, _1>>(A, B, alpha,
            beta, C, D, m, n, k, b, pdl_overlap_ratio, prefetch_ratio, gemmConfig, workspacePtr, workspaceBytes,
            stream);
        break;
    case tkc::ClusterShape::ClusterShape_8x1x1:
        return dispatchLowLatencyGemmCultassKernelSchedSm90<T, arch, ThreadblockShape, Shape<_8, _1, _1>>(A, B, alpha,
            beta, C, D, m, n, k, b, pdl_overlap_ratio, prefetch_ratio, gemmConfig, workspacePtr, workspaceBytes,
            stream);
        break;

    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][CutlassLowLatencyFp8GemmRunner][dispatchLowLatencyGemmClusterShapeSm90] Config is "
            "invalid for low latency fp8 gemm");
        break;
    }
}

template <typename T>
size_t dispatchLowLatencyGemmToCutlassSm90(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float alpha, float beta,
    void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio, float prefetch_ratio, ConfigType gemmConfig,
    char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    using namespace cute;

    using arch = cutlass::arch::Sm90;

    // swap M, N and A,B
    //  change m-> n , n-> m
    // A rowmajor, B colmajor , C, D  rowmajor
    switch (gemmConfig.cutlass_gemm_config.tile_config_sm90)
    {
    case tkc::CutlassTileConfigSM90::CtaShape64x16x128B:
        return dispatchLowLatencyGemmClusterShapeSm90<T, arch, Shape<_64, _16, _128>>(B, A, alpha, beta,
            static_cast<T const*>(C), static_cast<T*>(D), n, m, k, 1, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
            workspacePtr, workspaceBytes, stream);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x32x128B:
        return dispatchLowLatencyGemmClusterShapeSm90<T, arch, Shape<_64, _32, _128>>(B, A, alpha, beta,
            static_cast<T const*>(C), static_cast<T*>(D), n, m, k, 1, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
            workspacePtr, workspaceBytes, stream);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x64x128B:
        return dispatchLowLatencyGemmClusterShapeSm90<T, arch, Shape<_64, _64, _128>>(B, A, alpha, beta,
            static_cast<T const*>(C), static_cast<T*>(D), n, m, k, 1, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
            workspacePtr, workspaceBytes, stream);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x128x128B:
        return dispatchLowLatencyGemmClusterShapeSm90<T, arch, Shape<_64, _128, _128>>(B, A, alpha, beta,
            static_cast<T const*>(C), static_cast<T*>(D), n, m, k, 1, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
            workspacePtr, workspaceBytes, stream);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape64x256x128B:
        return dispatchLowLatencyGemmClusterShapeSm90<T, arch, Shape<_64, _256, _128>>(B, A, alpha, beta,
            static_cast<T const*>(C), static_cast<T*>(D), n, m, k, 1, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
            workspacePtr, workspaceBytes, stream);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x16x128B:
        return dispatchLowLatencyGemmClusterShapeSm90<T, arch, Shape<_128, _16, _128>>(B, A, alpha, beta,
            static_cast<T const*>(C), static_cast<T*>(D), n, m, k, 1, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
            workspacePtr, workspaceBytes, stream);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x32x128B:
        return dispatchLowLatencyGemmClusterShapeSm90<T, arch, Shape<_128, _32, _128>>(B, A, alpha, beta,
            static_cast<T const*>(C), static_cast<T*>(D), n, m, k, 1, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
            workspacePtr, workspaceBytes, stream);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x64x128B:
        return dispatchLowLatencyGemmClusterShapeSm90<T, arch, Shape<_128, _64, _128>>(B, A, alpha, beta,
            static_cast<T const*>(C), static_cast<T*>(D), n, m, k, 1, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
            workspacePtr, workspaceBytes, stream);
        break;
    case tkc::CutlassTileConfigSM90::CtaShape128x128x128B:
        return dispatchLowLatencyGemmClusterShapeSm90<T, arch, Shape<_128, _128, _128>>(B, A, alpha, beta,
            static_cast<T const*>(C), static_cast<T*>(D), n, m, k, 1, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
            workspacePtr, workspaceBytes, stream);
        break;
    case tkc::CutlassTileConfigSM90::Undefined:
        throw std::runtime_error(
            "[TensorRT LLM Error][CutlassLowLatencyFp8GemmRunner][dispatchLowLatencyGemmToCutlassSm90] gemm config "
            "undefined.");
        break;
    case tkc::CutlassTileConfigSM90::ChooseWithHeuristic:
        throw std::runtime_error(
            "[TensorRT LLM Error][CutlassLowLatencyFp8GemmRunner][dispatchLowLatencyGemmToCutlassSm90] gemm config "
            "should have "
            "already been set by "
            "heuristic.");
        break;
    default:
        throw std::runtime_error(
            "[TensorRT LLM Error][CutlassLowLatencyFp8GemmRunner][dispatchLowLatencyGemmToCutlassSm90] Config is "
            "invalid for low latency fp8 gemm");
        break;
    }
}

template <typename T>
CutlassLowLatencyFp8GemmRunner<T>::CutlassLowLatencyFp8GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mSm = tk::getSMVersion();
}

template <typename T>
size_t CutlassLowLatencyFp8GemmRunner<T>::dispatchToArch(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float alpha,
    float beta, void const* C, void* D, int m, int n, int k, float pdl_overlap_ratio, float preftch_ratio,
    ConfigType gemmConfig, char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
#ifndef PLACEHOLDER_KERNELS
    if (mSm == 90)
    {
        return dispatchLowLatencyGemmToCutlassSm90<T>(static_cast<__nv_fp8_e4m3 const*>(A),
            static_cast<__nv_fp8_e4m3 const*>(B), alpha, beta, static_cast<T const*>(C), static_cast<T*>(D), m, n, k,
            pdl_overlap_ratio, preftch_ratio, gemmConfig, workspacePtr, workspaceBytes, stream);
    }
    else
#endif
    {

        throw std::runtime_error(
            "[TensorRT LLM Error][CutlassLowLatencyFp8GemmRunner][GEMM Dispatch] dtype unsupported for CUTLASS Low "
            "Latency Gemm");
    }
    return 0;
}

template <typename T>
void CutlassLowLatencyFp8GemmRunner<T>::gemm(__nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float alpha, float beta, void const* C,
    void* D, int m, int n, int k, float pdl_overlap_ratio, float prefetch_ratio, ConfigType gemmConfig,
    char* workspacePtr, size_t const workspaceBytes, cudaStream_t stream)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatchToArch(static_cast<__nv_fp8_e4m3 const*>(A), static_cast<__nv_fp8_e4m3 const*>(B), alpha, beta,
        static_cast<T const*>(C), static_cast<T*>(D), m, n, k, pdl_overlap_ratio, prefetch_ratio, gemmConfig,
        workspacePtr, workspaceBytes, stream);
}

// Note: can be quite heavyweight; when possible, call once
template <typename T>
size_t CutlassLowLatencyFp8GemmRunner<T>::getWorkspaceSizeImpl(int const m, int const n, int const k)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    size_t workspace_size = 0;
    auto gemmConfigs = CutlassLowLatencyFp8GemmRunner<T>{}.getConfigs();
    for (auto const& gemmConfig : gemmConfigs)
    {
        try
        {
            size_t curr_workspace_size
                = CutlassLowLatencyFp8GemmRunner<T>::dispatchToArch(static_cast<__nv_fp8_e4m3*>(nullptr),
                    static_cast<__nv_fp8_e4m3*>(nullptr), 1.0f, 0.0f, static_cast<T const*>(nullptr),
                    static_cast<T*>(nullptr), m, n, k, 0.1F, 0.1F, gemmConfig, static_cast<char*>(nullptr), 0, nullptr);

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

template <typename T>
size_t CutlassLowLatencyFp8GemmRunner<T>::getWorkspaceSize(int const m, int const n, int const k)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // Custom hash function for the MNK type
    using MNK = std::tuple<int, int, int>;

    struct MNKHash
    {
        size_t operator()(const MNK& mnk) const
        {
            auto h1 = std::hash<int>{}(std::get<0>(mnk));
            auto h2 = std::hash<int>{}(std::get<1>(mnk));
            auto h3 = std::hash<int>{}(std::get<2>(mnk));
            return h1 ^ h2 ^ h3;
        }
    };

    static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;

    size_t workspace_size = 0;
    if (workspace_hashmap.find(std::make_tuple(m, n, k)) == workspace_hashmap.end())
    {
        workspace_size = CutlassLowLatencyFp8GemmRunner<T>::getWorkspaceSizeImpl(m, n, k);
        workspace_hashmap[std::make_tuple(m, n, k)] = workspace_size;
    }
    else
    {
        workspace_size = workspace_hashmap[std::make_tuple(m, n, k)];
    }
    return workspace_size;
}

template <typename T>
std::vector<ConfigType> CutlassLowLatencyFp8GemmRunner<T>::getConfigs() const
{
    using namespace cutlass_extensions;
    if (mSm != 90)
    {
        throw std::runtime_error(
            "[TensorRT LLM Error][CutlassLowLatencyFp8GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS FP8 Low "
            "Latency GEMM");
    }
    tkc::CutlassGemmConfig::CandidateConfigTypeParam config_type_param
        = tkc::CutlassGemmConfig::CandidateConfigTypeParam::HOPPER;

    std::vector<ConfigType> candidateConfigs;
    auto comm_configs = kernels::cutlass_kernels::get_candidate_configs(mSm, 1, config_type_param);

    // registers are not enough, remove some configs
    comm_configs.erase(std::remove_if(comm_configs.begin(), comm_configs.end(),
                           [](auto const& config)
                           { return config.tile_config_sm90 == tkc::CutlassTileConfigSM90::CtaShape128x256x128B; }),
        comm_configs.end());
    for (auto const& config : comm_configs)
    {
        if ((config.cluster_shape == ClusterShape::ClusterShape_1x1x1)
            || (config.cluster_shape == ClusterShape::ClusterShape_2x1x1)
            || (config.cluster_shape == ClusterShape::ClusterShape_8x1x1))
        {
            candidateConfigs.push_back(ConfigType{config, KernelScheduleType::WS_PREFETECH});
            candidateConfigs.push_back(ConfigType{config, KernelScheduleType::WS_SPLIT_PREFETECH});
        }
    }

    std::vector<ConfigType> clusterShape8x1x1Configs;
    for (auto const& config : candidateConfigs)
    {
        CutlassTileConfigSM90 tile = config.cutlass_gemm_config.tile_config_sm90;
        if (config.cutlass_gemm_config.cluster_shape == ClusterShape::ClusterShape_1x1x1)
        {
            if (config.kernel_schedule == KernelScheduleType::WS_PREFETECH)
            {
                clusterShape8x1x1Configs.push_back(
                    ConfigType{CutlassGemmConfig{tile, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                                   tkc::ClusterShape::ClusterShape_8x1x1},
                        KernelScheduleType::WS_PREFETECH});
            }
            else if (config.kernel_schedule == KernelScheduleType::WS_SPLIT_PREFETECH)
            {
                clusterShape8x1x1Configs.push_back(
                    ConfigType{CutlassGemmConfig{tile, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                                   tkc::ClusterShape::ClusterShape_8x1x1},
                        KernelScheduleType::WS_SPLIT_PREFETECH});
            }
        }
    }
    candidateConfigs.insert(candidateConfigs.end(), clusterShape8x1x1Configs.begin(), clusterShape8x1x1Configs.end());

    return candidateConfigs;
}

}; // namespace cutlass_kernels
}; // namespace kernels

}; // namespace tensorrt_llm
