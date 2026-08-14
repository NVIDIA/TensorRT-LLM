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

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__
#include "tensorrt_llm/common/config.h"

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "tensorrt_llm/kernels/archCondition.h"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC__

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{
using namespace cute;

template <typename ElementType, typename OutElementType, typename AccumElementType, typename CTAShape,
    typename ClusterShape, typename MainloopScheduleType, typename EpilogueScheduleType,
    typename TileSchedulerType = void>
struct DeviceGemmFp8RowwiseSm100
{
    static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");

    // A matrix configuration
    using ElementA = ElementType;                      // Element type for A matrix operand
    using LayoutA = cutlass::layout::RowMajor;         // Layout type for A matrix operand
    static constexpr int AlignmentA
        = 128 / cutlass::sizeof_bits<ElementA>::value; // Memory access granularity/alignment of A
                                                       // matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using ElementB = ElementType;                      // Element type for B matrix operand
    using LayoutB = cutlass::layout::ColumnMajor;      // Layout type for B matrix operand
    static constexpr int AlignmentB
        = 128 / cutlass::sizeof_bits<ElementB>::value; // Memory access granularity/alignment of B
                                                       // matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using ElementC = void;                                   // Element type for C matrix operands
    using LayoutC = cutlass::layout::RowMajor;               // Layout type for C matrix operands
    static constexpr int AlignmentC
        = 128 / cutlass::sizeof_bits<OutElementType>::value; // Memory access granularity/alignment of C matrices in
                                                             // units of elements (up to 16 bytes)

    // Output matrix configuration
    using ElementOutput = OutElementType;           // Element type for output matrix operands
    using LayoutOutput = cutlass::layout::RowMajor; // Layout type for output matrix operands
    static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

    // Auxiliary matrix configuration and other fusion types
    using ElementBias = float;

    // Multiply-accumulate blocking/pipelining details
    using ElementAccumulator = AccumElementType; // Element type for internal accumulation
    using ElementCompute = float;                // Element type for compute
    using ElementComputeEpilogue = float;
    using ArchTag = cutlass::arch::Sm100;        // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
    using TileShape = CTAShape;                           // Threadblock-level tile size
    using TileScheduler = TileSchedulerType;

    using Multiply = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementComputeEpilogue,
        ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
    using Add = cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, ElementComputeEpilogue, ElementComputeEpilogue,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using Cast = cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Identity, OutElementType,
        ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;

    // Implement rowwise scaling epilogue.
    using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, ElementComputeEpilogue,
        ElementComputeEpilogue, cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

    using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementComputeEpilogue,
        ElementComputeEpilogue, cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

    using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementBias, ElementBias,
        cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

    using Compute0 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies,
        ElementComputeEpilogue, // First stage output type.
        ElementComputeEpilogue, // First stage input types.
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

    using Compute1 = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementOutput,
        ElementComputeEpilogue, // Second stage input types.
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

    using ComputeBias = cutlass::epilogue::fusion::Sm90Compute<cutlass::plus,
        ElementOutput, // Final (optional) stage output type.
        ElementBias,   // Final stage input types.
        cutlass::FloatRoundStyle::round_to_nearest>;

    using EVTComputeBias = cutlass::epilogue::fusion::Sm90EVT<ComputeBias, Bias, EVTCompute1>;

    using EpilogueEVT = EVTCompute1;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag, OperatorClass,
        TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementComputeEpilogue, ElementC, LayoutC, AlignmentC, ElementOutput, LayoutOutput, AlignmentOutput,
        EpilogueScheduleType, EpilogueEVT>::CollectiveOp;

    using MainLoopSchedule = cutlass::gemm::collective::KernelScheduleAuto;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<ArchTag, OperatorClass, ElementA,
        LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainLoopSchedule>::CollectiveOp;

    template <typename Base>
    struct Sm100Only : Base
    {
        using typename Base::Params;

        CUTLASS_DEVICE
        void operator()(Params const& params, char* smem_buf)
        {
            if constexpr (tensorrt_llm::kernels::arch::is_match_v<100>)
            {
                this->Base::operator()(params, smem_buf);
            }
            else
            {
                if (cute::thread0())
                {
                    printf("%s : This kernel shall only run on SM100 devices.\n", __PRETTY_FUNCTION__);
                    __trap();
                }
            }
        }
    };

    using GemmKernel
        = Sm100Only<typename cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, // Indicates
                                                                                                   // ProblemShape
            CollectiveMainloop, CollectiveEpilogue, TileScheduler>>;

    using Gemm = typename cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
