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
#include "./allreduce_gemm_impl_sm100.h"
#include "./allreduce_gemm_impl_sm90.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/half.h"

namespace tensorrt_llm::kernels::opened_cutlass_kernels
{
/////////////////////////////////////////////////
// GemmAllReduce implementation specializations
/////////////////////////////////////////////////
template <int SmVersion, GemmAllReduceImpl Impl, typename SmXGemmTraits>
struct GemmImpl
{
    using Type = void;
};

// Hopper
template <typename SmXGemmTraits>
struct GemmImpl<90, GemmAllReduceImpl::kNVLS_2SHOT, SmXGemmTraits>
{
    using Type = GemmAllReduceImplTwoshot_Sm90<SmXGemmTraits, false>;
};

// Blackwell
template <typename SmXGemmTraits>
struct GemmImpl<100, GemmAllReduceImpl::kNVLS_2SHOT, SmXGemmTraits>
{
    using Type = GemmAllReduceImplTwoshot_Sm100<SmXGemmTraits>;
};

///////////////////////////////////////////////
// Builder for GemmAllReduce implementations
///////////////////////////////////////////////
template <typename KeyType, typename ValueType>
class GemmAllReduceRegistryBuilder
{
public:
    template <typename GemmTraits, GemmAllReduceImpl Impl, MainloopScheduleType Schedule, TileShape TileShape_MNK,
        ClusterShape ClusterShape_MNK>
    void addSm90()
    {
        using TransposedGemmTraits = decltype(swapAndTranspose<GemmTraits>());

        using MainloopSchedule = cute::conditional_t<Schedule == MainloopScheduleType::PINGPONG,
            cute::conditional_t<isFP8<TransposedGemmTraits>(),
                cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
                cutlass::gemm::KernelTmaWarpSpecializedPingpong>,
            cute::conditional_t<isFP8<TransposedGemmTraits>(),
                cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum,
                cutlass::gemm::KernelTmaWarpSpecializedCooperative>>;

        using EpilogueSchedule = cute::conditional_t<Schedule == MainloopScheduleType::PINGPONG,
            cutlass::epilogue::TmaWarpSpecialized, cutlass::epilogue::TmaWarpSpecializedCooperative>;

        using Sm90GemmTraits = Sm90GemmTypes<typename TransposedGemmTraits::ElementA,
            typename TransposedGemmTraits::ElementB, typename TransposedGemmTraits::ElementC,
            typename TransposedGemmTraits::ElementD, typename TransposedGemmTraits::ElementSFA,
            typename TransposedGemmTraits::ElementSFB, typename TransposedGemmTraits::LayoutA,
            typename TransposedGemmTraits::LayoutB, typename TransposedGemmTraits::LayoutC,
            typename TransposedGemmTraits::LayoutD, decltype(get_tile_shape<TileShape_MNK>()),
            decltype(get_cluster_shape<ClusterShape_MNK>()), MainloopSchedule, EpilogueSchedule>;

        using GemmType = typename GemmImpl<90, Impl, Sm90GemmTraits>::Type;
        static_assert(not std::is_same_v<GemmType, void>);

        const GemmAllReduceImplInterface::LaunchConfig key(
            {Impl, Schedule, TileShape_MNK, ClusterShape_MNK, 1, true /* transposed*/});
        auto value = std::make_shared<GemmType>();

        mGemmRegistry.insert({key, value});
    }

    template <typename GemmTraits, GemmAllReduceImpl Impl, typename MmaType, TileShape TileShape_MNK,
        ClusterShape ClusterShape_MNK>
    void addSm100()
    {
        using Sm100GemmTraits = Sm100GemmTypes<typename GemmTraits::ElementA, typename GemmTraits::ElementB,
            typename GemmTraits::ElementC, typename GemmTraits::ElementD, typename GemmTraits::ElementSFA,
            typename GemmTraits::ElementSFB, typename GemmTraits::LayoutA, typename GemmTraits::LayoutB,
            typename GemmTraits::LayoutC, typename GemmTraits::LayoutD, decltype(get_tile_shape<TileShape_MNK>()),
            decltype(get_cluster_shape<ClusterShape_MNK>()), MmaType>;

        using GemmType = typename GemmImpl<100, Impl, Sm100GemmTraits>::Type;
        static_assert(not std::is_same_v<GemmType, void>);

        constexpr int MMA_SMs = std::is_same_v<MmaType, _1SM> ? 1 : 2;

        const GemmAllReduceImplInterface::LaunchConfig key({Impl, MainloopScheduleType::WARPSPECIALIZED, TileShape_MNK,
            ClusterShape_MNK, MMA_SMs, false /* transposed */});
        auto value = std::make_shared<GemmType>();

        mGemmRegistry.insert({key, value});
    }

    auto build()
    {
        return mGemmRegistry;
    }

private:
    template <typename GemmTraits>
    static constexpr auto swapAndTranspose()
    {
        constexpr bool SwapAB = true;

        using ElementA = cute::conditional_t<SwapAB, typename GemmTraits::ElementB, typename GemmTraits::ElementA>;
        using ElementB = cute::conditional_t<SwapAB, typename GemmTraits::ElementA, typename GemmTraits::ElementB>;
        using ElementSFA
            = cute::conditional_t<SwapAB, typename GemmTraits::ElementSFB, typename GemmTraits::ElementSFA>;
        using ElementSFB
            = cute::conditional_t<SwapAB, typename GemmTraits::ElementSFA, typename GemmTraits::ElementSFB>;

        using LayoutA
            = cute::conditional_t<SwapAB, typename cutlass::layout::LayoutTranspose<typename GemmTraits::LayoutB>::type,
                typename GemmTraits::LayoutA>;

        using LayoutB
            = cute::conditional_t<SwapAB, typename cutlass::layout::LayoutTranspose<typename GemmTraits::LayoutA>::type,
                typename GemmTraits::LayoutB>;

        using LayoutC
            = cute::conditional_t<SwapAB, typename cutlass::layout::LayoutTranspose<typename GemmTraits::LayoutC>::type,
                typename GemmTraits::LayoutC>;

        using LayoutD
            = cute::conditional_t<SwapAB, typename cutlass::layout::LayoutTranspose<typename GemmTraits::LayoutD>::type,
                typename GemmTraits::LayoutD>;

        return GemmTypes<ElementA, ElementB, typename GemmTraits::ElementC, typename GemmTraits::ElementD, ElementSFA,
            ElementSFB, LayoutA, LayoutB, LayoutC, LayoutD>{};
    }

    template <typename GemmTraits>
    static constexpr bool isFP8()
    {
        return std::is_same_v<typename GemmTraits::ElementA, cutlass::float_e4m3_t>
            and std::is_same_v<typename GemmTraits::ElementB, cutlass::float_e4m3_t>;
    }

    std::map<KeyType, ValueType> mGemmRegistry;
};

//////////////////////////////////////////
// GemmAllReduceImplRunner methods
//////////////////////////////////////////
template <typename GemmTraits>
GemmAllReduceImplRunner<GemmTraits>::GemmAllReduceImplRunner()
{
    GemmAllReduceRegistryBuilder<KeyType, ValueType> registry_builder;
    constexpr int bits_input = cutlass::sizeof_bits<typename GemmTraits::ElementA>::value;

    // Instantiate GEMMs for each config
    switch (tensorrt_llm::common::getSMVersion())
    {
    // Hopper
    case 90:
        // Sub-byte GEMMs not supported
        if constexpr (bits_input >= 8)
        {
            registry_builder.addSm90<GemmTraits, GemmAllReduceImpl::kNVLS_2SHOT, MainloopScheduleType::PINGPONG,
                TileShape::TileShape_128x16x128, ClusterShape::ClusterShape_2x1x1>();
            registry_builder.addSm90<GemmTraits, GemmAllReduceImpl::kNVLS_2SHOT, MainloopScheduleType::PINGPONG,
                TileShape::TileShape_128x32x128, ClusterShape::ClusterShape_2x1x1>();
            registry_builder.addSm90<GemmTraits, GemmAllReduceImpl::kNVLS_2SHOT, MainloopScheduleType::PINGPONG,
                TileShape::TileShape_128x64x128, ClusterShape::ClusterShape_2x1x1>();
            registry_builder.addSm90<GemmTraits, GemmAllReduceImpl::kNVLS_2SHOT, MainloopScheduleType::PINGPONG,
                TileShape::TileShape_128x128x128, ClusterShape::ClusterShape_2x1x1>();
        }
        break;
    // Blackwell
    case 100:
        registry_builder.addSm100<GemmTraits, GemmAllReduceImpl::kNVLS_2SHOT, _2SM, TileShape::TileShape_128x256x128,
            ClusterShape::ClusterShape_4x1x1>();
        break;
    default: TLLM_THROW("SM architecture not supported for GEMM+AR fusion.");
    }

    mGemmRegistry = registry_builder.build();
}

template <typename GemmTraits>
std::shared_ptr<PersistentWorkspaceInterface> GemmAllReduceImplRunner<GemmTraits>::getPersistentWorkspace(
    ProblemArgs const& max_problem)
{
    auto swapped_problem = swapAB(max_problem);
    std::shared_ptr<PersistentWorkspaceInterface> pworkspace;
    // Iterate over all launch configs and return workspace with largest allocation size so that it
    // will work for all launch configs.
    for (auto launch_config : getSupportedLaunchConfigs())
    {
        TLLM_CHECK_WITH_INFO(mGemmRegistry.count(launch_config) > 0, "No cutlass gemm impl found.");
        auto gemm_impl = mGemmRegistry[launch_config];

        swapped_problem.launch_config = launch_config;
        auto impl_pworkspace = gemm_impl->getPersistentWorkspace(swapped_problem);
        // Ensure we return workspace with largest allocation so all configs work.
        if (!pworkspace || impl_pworkspace->size() > pworkspace->size())
        {
            pworkspace = impl_pworkspace;
        }
    }
    return pworkspace;
}

template <typename GemmTraits>
int GemmAllReduceImplRunner<GemmTraits>::run(ProblemArgs const& problem, cudaStream_t stream)
{
    auto swapped_problem = swapAB(problem);
    TLLM_CHECK_WITH_INFO(mGemmRegistry.count(swapped_problem.launch_config) > 0, "No cutlass gemm impl found.");
    auto gemm_impl = mGemmRegistry.at(swapped_problem.launch_config);
    return gemm_impl->run(swapped_problem, stream);
}

template <typename GemmTraits>
std::vector<GemmAllReduceImplInterface::LaunchConfig>
GemmAllReduceImplRunner<GemmTraits>::getSupportedLaunchConfigs() const
{
    std::vector<LaunchConfig> configs;
    for (auto [key, value] : mGemmRegistry)
    {
        configs.emplace_back(key);
    }
    return configs;
}

///////////////////////////////////////
// Private Methods
///////////////////////////////////////
template <typename GemmTraits>
GemmAllReduceImplInterface::ProblemArgs GemmAllReduceImplRunner<GemmTraits>::swapAB(
    GemmAllReduceImplInterface::ProblemArgs const& problem) const
{
    if (not problem.launch_config.transposed)
    {
        return problem;
    }
    auto [M, N, K, L] = problem.problem_size;
    ProblemArgs swapped_problem = problem;
    swapped_problem.A = problem.B;
    swapped_problem.B = problem.A;
    swapped_problem.A_scale = problem.B_scale;
    swapped_problem.B_scale = problem.B_scale;
    swapped_problem.problem_size = std::make_tuple(N, M, K, L);
    return swapped_problem;
}

// fp16xfp16=fp16
template class GemmAllReduceImplRunner<
    GemmTypes<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, void, void, cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

// bf16xbf16=bf16
template class GemmAllReduceImplRunner<
    GemmTypes<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, void, void,
        cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

// fp8xfp8=bf16
template class GemmAllReduceImplRunner<
    GemmTypes<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, cutlass::bfloat16_t, void, void,
        cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

// fp8xfp8=fp16
template class GemmAllReduceImplRunner<
    GemmTypes<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t, cutlass::half_t, void, void,
        cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

// fp4xfp4=fp16
template class GemmAllReduceImplRunner<GemmTypes<cutlass::float_e2m1_t, cutlass::float_e2m1_t, cutlass::half_t,
    cutlass::half_t, cutlass::float_ue4m3_t, cutlass::float_ue4m3_t, cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

// fp4xfp4=bf16
template class GemmAllReduceImplRunner<GemmTypes<cutlass::float_e2m1_t, cutlass::float_e2m1_t, cutlass::bfloat16_t,
    cutlass::bfloat16_t, cutlass::float_ue4m3_t, cutlass::float_ue4m3_t, cutlass::layout::RowMajor,
    cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

} // namespace tensorrt_llm::kernels::opened_cutlass_kernels
