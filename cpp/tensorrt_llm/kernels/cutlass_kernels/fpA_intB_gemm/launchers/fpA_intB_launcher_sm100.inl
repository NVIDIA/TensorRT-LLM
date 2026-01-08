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

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/packed_stride.hpp"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm_configs.h"

#include "cutlass_extensions/gemm/collective/collective_builder_sm100_weightonly.hpp"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC__

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/launchers/fpA_intB_launcher_sm100.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels_oss
{
using namespace tensorrt_llm::kernels::cutlass_kernels;
namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

using namespace cute;

template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
    cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename CTAShape, typename ClusterShape,
    typename MainloopScheduleType, typename EpilogueScheduleType>
void sm100_generic_mixed_gemm_kernelLauncher(ActivationType const* A, WeightType const* B,
    ScaleZeroType const* weight_scales, ScaleZeroType const* weight_zero_points, BiasType const* biases,
    float const alpha, OutputType* C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemm_config,
    char* workspace, size_t workspace_bytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    using CutlassActivationType = typename TllmToCutlassTypeAdapter<ActivationType>::type;

#ifdef COMPILE_BLACKWELL_TMA_GEMMS
    if constexpr (!should_filter_tma_warp_specialized_gemm_problem_shape_v<cutlass::arch::Sm100, CTAShape, ClusterShape,
                      false, ActivationType>)
    {
        using CutlassWeightType__ = typename TllmToCutlassTypeAdapter<WeightType>::type;
        // We need to remap this since SM100 uses a different layout for the weight matrix.
        using CutlassWeightType_ = std::conditional_t<std::is_same_v<CutlassWeightType__, cutlass::uint4b_t>,
            cutlass::int4b_t, CutlassWeightType__>;

        using CutlassWeightType
            = std::conditional_t<std::is_same_v<CutlassWeightType_, uint8_t>, int8_t, CutlassWeightType_>;

        using CutlassScaleZeroType = typename TllmToCutlassTypeAdapter<ScaleZeroType>::type;
        using CutlassBiasType = typename TllmToCutlassTypeAdapter<BiasType>::type;
        using CutlassOutputType = typename TllmToCutlassTypeAdapter<OutputType>::type;

        static_assert(std::is_same_v<CutlassActivationType, cutlass::half_t>
                || std::is_same_v<CutlassActivationType, cutlass::bfloat16_t>
                || std::is_same_v<CutlassActivationType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassActivationType, cutlass::float_e5m2_t>,
            "Activation type must be bfloat16, half, FP8");

        static_assert(std::is_same_v<CutlassWeightType, int8_t> || std::is_same_v<CutlassWeightType, cutlass::int4b_t>
                || std::is_same_v<CutlassWeightType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassWeightType, cutlass::float_e5m2_t>,
            "Weight type must be fp8, int8_t or int4_t");

        static_assert(!std::is_same_v<CutlassActivationType, cutlass::float_e4m3_t>
                || std::is_same_v<CutlassScaleZeroType, cutlass::half_t>,
            "Scale/Zero type must be half for fp8 activation");

        using LayoutA = cutlass::layout::RowMajor; // Layout type for A matrix operand
        constexpr int AlignmentA = 128 / cutlass::sizeof_bits<CutlassActivationType>::value;

        using LayoutB = cutlass::layout::ColumnMajor; // Layout type for B matrix operand
        constexpr int AlignmentB = 128 / cutlass::sizeof_bits<CutlassWeightType>::value;

        // This example manually swaps and transposes, so keep transpose of input layouts
        using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
        using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

        using ElementZero = CutlassScaleZeroType;
        using ElementScale = CutlassScaleZeroType;

        // C/D matrix configuration. We reuse the C operand for the bias and set the stride for broadcast.
        using LayoutBias = cutlass::layout::RowMajor;
        constexpr int AlignmentBias = 128 / cutlass::sizeof_bits<CutlassBiasType>::value;

        // D matrix configuration
        using LayoutOutput = cutlass::layout::RowMajor;
        constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<CutlassOutputType>::value;

        // Core kernel configurations
        using ElementAccumulator = float;     // Element type for internal accumulation
        using ElementCompute = float;         // Element type for epilogue computation
        using ArchTag = cutlass::arch::Sm100; // Tag indicating the minimum SM that supports the intended feature
        using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
        // using TileShape = CTAShape;                           // Threadblock-level tile size
        constexpr static bool Is2SM = cute::size<0>(ClusterShape{}) == 2;
        constexpr static int TileM = cute::size<0>(CTAShape{}) * (Is2SM ? 2 : 1);
        constexpr static int TileN = cute::size<1>(CTAShape{});
        constexpr static int TileK = cute::size<2>(CTAShape{});
        using TileShape = cute::Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
        using MainloopSchedule = std::conditional_t<Is2SM, cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100,
            cutlass::gemm::KernelTmaWarpSpecialized1SmMixedInputSm100>;
        using EpilogueSchedule = std::conditional_t<Is2SM, cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::epilogue::TmaWarpSpecialized1Sm>;

        static_assert(std::is_same_v<EpilogueTag, tensorrt_llm::cutlass_extensions::EpilogueOpBias>, "");

        using CollectiveEpilogue =
            typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass, TileShape, ClusterShape,
                cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
                // Transpose layout of D here since we use the explicit swap + transpose trick
                // Void C since we don't use it. Prevents smem allocation.
                CutlassBiasType, typename cutlass::layout::LayoutTranspose<LayoutBias>::type, AlignmentBias,
                CutlassOutputType, typename cutlass::layout::LayoutTranspose<LayoutOutput>::type, AlignmentOutput,
                EpilogueSchedule>::CollectiveOp;

        using PackedScaleZero = cute::tuple<CutlassWeightType, ElementScale, ElementZero>;
        using PackedScale = cute::tuple<CutlassWeightType, ElementScale>;
        using ElementBCollectiveInfo = std::conditional_t<cutlass::hasZero(QuantOp), PackedScaleZero, PackedScale>;

        constexpr int ScaleGranularityN = 1;                              // Should be less than or equal to GEMM_N
        constexpr int ScaleGranularityK = size<2>(TileShape{});           // Should be less than or equal to GEMM_K
        using ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<ScaleGranularityN, ScaleGranularityK>;
        using LayoutScale = decltype(ScaleConfig::deduce_layout_scale()); // Layout type for SFA matrix operand
        LayoutScale layout_S = ScaleConfig::tile_atom_to_shape_scale(make_shape(n, k, 1));

        // We swap A and B operands to the builder here
        using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderSm100WeightOnly<ArchTag,
            OperatorClass, ElementBCollectiveInfo, cute::tuple<LayoutB_Transpose, LayoutScale>, AlignmentB,
            CutlassActivationType, LayoutA_Transpose, AlignmentA, ElementAccumulator, TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            MainloopSchedule>::CollectiveOp;

        using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, // Indicates ProblemShape
            CollectiveMainloop, CollectiveEpilogue>;

        using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

        using StrideA = typename GemmKernel::StrideA;
        using StrideB = typename GemmKernel::StrideB;
        using StrideC = typename GemmKernel::StrideC;
        using StrideD = typename GemmKernel::StrideD;

        if (weight_scales == nullptr)
        {
            throw std::runtime_error("Weight scales must always be set to a non-null value.");
        }

        if constexpr (cutlass::isFinegrained(QuantOp))
        {
            int cta_shape_k = cute::size<2>(TileShape{});
            if (group_size % cta_shape_k != 0)
            {
                std::string err_msg = "The group size must a multiple of " + std::to_string(cta_shape_k);
                throw std::runtime_error("[TensorRT LLM Error][fpA_intB Runner]" + err_msg);
            }

            if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY)
            {
                if (weight_zero_points != nullptr)
                {
                    throw std::runtime_error("Weight zero pointer must be a nullptr for scale only fine grained");
                }
            }
            else if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
            {
                if (weight_zero_points == nullptr)
                {
                    throw std::runtime_error("Weight zero pointer must be valid for scale and bias fine grained");
                }
            }
        }
        else
        {
            if (group_size != k)
            {
                throw std::runtime_error("Invalid group size for per column scaling kernels.");
            }

            if (weight_zero_points != nullptr)
            {
                throw std::runtime_error("Weight zero-points must be null when running per column scaling");
            }
        }

        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(0, m, 1));

        typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1},
            {reinterpret_cast<CutlassWeightType const*>(B), stride_B, reinterpret_cast<CutlassActivationType const*>(A),
                stride_A, reinterpret_cast<ElementScale const*>(weight_scales), layout_S, group_size,
                reinterpret_cast<ElementZero const*>(weight_zero_points)},
            {{alpha}, reinterpret_cast<CutlassBiasType const*>(biases), stride_C,
                reinterpret_cast<CutlassOutputType*>(C), stride_D}};

        Gemm gemm;
        if (gemm.get_workspace_size(args) > workspace_bytes)
        {
            TLLM_LOG_ERROR("[TensorRT LLM Error][fpA_intB Runner] given workspace size insufficient.");
        }

        auto can_implement = gemm.can_implement(args);
        if (can_implement != cutlass::Status::kSuccess)
        {
            std::string err_msg = "fpA_intB cutlass kernel will fail for params. Error: "
                + std::string(cutlass::cutlassGetStatusString(can_implement));
            std::cout << err_msg << std::endl;
            throw std::runtime_error("[TensorRT LLM Error][fpA_intB Runner] " + err_msg);
        }

        auto init_status = gemm.initialize(args, workspace, stream);
        if (init_status != cutlass::Status::kSuccess)
        {
            std::string err_msg = "Failed to initialize cutlass fpA_intB gemm. Error: "
                + std::string(cutlassGetStatusString(init_status));
            throw std::runtime_error("[TensorRT LLM Error][fpA_intB Runner] " + err_msg);
        }

        auto run_status = gemm.run(stream);
        if (run_status != cutlass::Status::kSuccess)
        {
            std::string err_msg
                = "Failed to run cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(run_status));
            throw std::runtime_error("[TensorRT LLM Error][fpA_intB Runner] " + err_msg);
        }
    }
    else
    {
        std::stringstream ss;
        ss << "[TensorRT LLM Error][fpA_intB Runner] Config (" << (int64_t) cute::size<0>(CTAShape{}) << ","
           << (int64_t) cute::size<1>(CTAShape{}) << "," << (int64_t) cute::size<2>(CTAShape{}) << ") ("
           << (int64_t) cute::size<0>(ClusterShape{}) << "," << (int64_t) cute::size<1>(ClusterShape{}) << ","
           << (int64_t) cute::size<2>(ClusterShape{}) << ") not compiled with FAST_BUILD.";

        throw std::runtime_error(ss.str());
    }

#else  // COMPILE_BLACKWELL_TMA_GEMMS
    throw std::runtime_error(
        "[TensorRT LLM Error][fpA_intB Runner] Please recompile with support for blackwell by passing 100-real as an "
        "arch "
        "to build_wheel.py.");
#endif // COMPILE_BLACKWELL_TMA_GEMMS
}

} // namespace cutlass_kernels_oss
} // namespace kernels
} // namespace tensorrt_llm
