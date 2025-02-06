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
#include "allreduce_gemm_impl_sm90.h"
#include "allreduce_gemm_runner.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "cutlass/half.h"

namespace tensorrt_llm::kernels::cutlass_kernels
{
using namespace cute;
using namespace tensorrt_llm::cutlass_extensions;
namespace tc = tensorrt_llm::common;

template <typename GemmTraits>
static constexpr bool isFP8()
{
    return std::is_same_v<typename GemmTraits::ElementA, cutlass::float_e4m3_t>
        and std::is_same_v<typename GemmTraits::ElementB, cutlass::float_e4m3_t>;
}

template <typename GemmTraits, MainloopScheduleType Schedule>
static constexpr auto Sm90UpdateTraits()
{
    using MainloopSchedule = cute::conditional_t<Schedule == MainloopScheduleType::PINGPONG,
        cute::conditional_t<isFP8<GemmTraits>(), cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
            cutlass::gemm::KernelTmaWarpSpecializedPingpong>,
        cute::conditional_t<isFP8<GemmTraits>(), cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum,
            cutlass::gemm::KernelTmaWarpSpecializedCooperative>>;

    using EpilogueSchedule = cute::conditional_t<Schedule == MainloopScheduleType::PINGPONG,
        cutlass::epilogue::TmaWarpSpecialized, cutlass::epilogue::TmaWarpSpecializedCooperative>;

    using CutlassGemmTraits = CutlassGemmTypes<typename GemmTraits::ElementA, typename GemmTraits::ElementB,
        typename GemmTraits::ElementC, typename GemmTraits::ElementD, typename GemmTraits::LayoutA,
        typename GemmTraits::LayoutB, typename GemmTraits::LayoutC, typename GemmTraits::LayoutD,
        typename GemmTraits::TileShape_MNK, typename GemmTraits::ClusterShape_MNK, MainloopSchedule, EpilogueSchedule>;

    return CutlassGemmTraits{};
}

/////////////////////////////////////////////////
// GemmAllReduce implementation specializations
/////////////////////////////////////////////////
template <int SmVersion, GemmAllReduceImpl Impl, MainloopScheduleType Schedule, typename GemmTraits>
struct GemmImpl
{
    using Type = void;
};

template <typename GemmTraits>
struct GemmImpl<90, GemmAllReduceImpl::NVLS_1SHOT, MainloopScheduleType::PINGPONG, GemmTraits>
{
    using Sm90GemmTraits = decltype(Sm90UpdateTraits<GemmTraits, MainloopScheduleType::PINGPONG>());
    using Type = GemmAllReduceImplTwoshot_Sm90<Sm90GemmTraits, true>;
};

template <typename GemmTraits>
struct GemmImpl<90, GemmAllReduceImpl::NVLS_1SHOT, MainloopScheduleType::COOPERATIVE, GemmTraits>
{
    using Sm90GemmTraits = decltype(Sm90UpdateTraits<GemmTraits, MainloopScheduleType::COOPERATIVE>());
    using Type = GemmAllReduceImplTwoshot_Sm90<Sm90GemmTraits, true>;
};

template <typename GemmTraits>
struct GemmImpl<90, GemmAllReduceImpl::NVLS_2SHOT, MainloopScheduleType::PINGPONG, GemmTraits>
{
    using Sm90GemmTraits = decltype(Sm90UpdateTraits<GemmTraits, MainloopScheduleType::PINGPONG>());
    using Type = GemmAllReduceImplTwoshot_Sm90<Sm90GemmTraits, false>;
};

template <typename GemmTraits>
struct GemmImpl<90, GemmAllReduceImpl::NVLS_2SHOT, MainloopScheduleType::COOPERATIVE, GemmTraits>
{
    using Sm90GemmTraits = decltype(Sm90UpdateTraits<GemmTraits, MainloopScheduleType::COOPERATIVE>());
    using Type = GemmAllReduceImplTwoshot_Sm90<Sm90GemmTraits, false>;
};

template <typename GemmTraits, typename KeyType, typename ValueType>
class GemmAllReduceRegistryBuilder
{
public:
    template <int SmVersion, GemmAllReduceImpl Impl, MainloopScheduleType Schedule, TileShape TileShape_MNK,
        ClusterShape ClusterShape_MNK>
    void add()
    {
        // We transpose & swap AB such that M is always mapped to WGMMA N.
        // This allows us to get better tensor core utilization when M is < 64.
        constexpr bool SwapAB = true;

        using ElementA = cute::conditional_t<SwapAB, typename GemmTraits::ElementB, typename GemmTraits::ElementA>;

        using ElementB = cute::conditional_t<SwapAB, typename GemmTraits::ElementA, typename GemmTraits::ElementB>;

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

        using CutlassGemmTraits = CutlassGemmTypes<ElementA, ElementB, typename GemmTraits::ElementC,
            typename GemmTraits::ElementD, LayoutA, LayoutB, LayoutC, LayoutD,
            decltype(get_tile_shape<TileShape_MNK>()), decltype(get_cluster_shape<ClusterShape_MNK>()), void, void>;

        using GemmType = typename GemmImpl<SmVersion, Impl, Schedule, CutlassGemmTraits>::Type;
        static_assert(not std::is_same_v<GemmType, void>);

        auto key = std::make_tuple(Impl, Schedule, TileShape_MNK, ClusterShape_MNK);
        auto value = std::make_shared<GemmType>();

        mGemmRegistry.insert({key, value});
    }

    auto build()
    {
        return mGemmRegistry;
    }

private:
    std::map<KeyType, ValueType> mGemmRegistry;
};

//////////////////////////////////////////
// GemmAllReduceImplRunner methods
//////////////////////////////////////////
template <typename GemmTraits>
GemmAllReduceImplRunner<GemmTraits>::GemmAllReduceImplRunner()
{
    GemmAllReduceRegistryBuilder<GemmTraits, KeyType, ValueType> registry_builder;

    // Instantiate GEMMs for each config
    switch (tc::getSMVersion())
    {
    case 90:
        registry_builder.template add<90, GemmAllReduceImpl::NVLS_2SHOT, MainloopScheduleType::PINGPONG,
            TileShape::TileShape_128x16x128, ClusterShape::ClusterShape_1x1x1>();
        registry_builder.template add<90, GemmAllReduceImpl::NVLS_2SHOT, MainloopScheduleType::PINGPONG,
            TileShape::TileShape_128x32x128, ClusterShape::ClusterShape_1x1x1>();
        registry_builder.template add<90, GemmAllReduceImpl::NVLS_2SHOT, MainloopScheduleType::PINGPONG,
            TileShape::TileShape_128x64x128, ClusterShape::ClusterShape_1x1x1>();
        registry_builder.template add<90, GemmAllReduceImpl::NVLS_2SHOT, MainloopScheduleType::PINGPONG,
            TileShape::TileShape_128x128x128, ClusterShape::ClusterShape_2x1x1>();
    }

    mGemmRegistry = registry_builder.build();
}

template <typename GemmTraits>
std::shared_ptr<PersistentWorkspaceInterface> GemmAllReduceImplRunner<GemmTraits>::getPersistentWorkspace(
    ProblemArgs const& max_problem)
{
    auto swapped_problem = swapAB(max_problem);
    auto key = std::make_tuple(swapped_problem.launch_config.impl, swapped_problem.launch_config.schedule,
        swapped_problem.launch_config.tile_shape, swapped_problem.launch_config.cluster_shape);
    TLLM_CHECK_WITH_INFO(mGemmRegistry.count(key) > 0, "No cutlass gemm impl found.");
    auto gemm_impl = mGemmRegistry[key];
    return gemm_impl->getPersistentWorkspace(swapped_problem);
}

template <typename GemmTraits>
int GemmAllReduceImplRunner<GemmTraits>::run(ProblemArgs const& problem, cudaStream_t stream)
{
    auto swapped_problem = swapAB(problem);
    auto key = std::make_tuple(swapped_problem.launch_config.impl, swapped_problem.launch_config.schedule,
        swapped_problem.launch_config.tile_shape, swapped_problem.launch_config.cluster_shape);
    TLLM_CHECK_WITH_INFO(mGemmRegistry.count(key) > 0, "No cutlass gemm impl found.");
    auto gemm_impl = mGemmRegistry.at(key);
    return gemm_impl->run(swapped_problem, stream);
}

template <typename GemmTraits>
std::vector<GemmAllReduceImplInterface::LaunchConfig>
GemmAllReduceImplRunner<GemmTraits>::getSupportedLaunchConfigs() const
{
    std::vector<LaunchConfig> configs;
    for (auto [key, value] : mGemmRegistry)
    {
        auto [impl, schedule, tile_shape, cluster_shape] = key;
        configs.emplace_back(LaunchConfig{impl, schedule, tile_shape, cluster_shape});
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
    auto [M, N, K, L] = problem.problem_size;
    ProblemArgs swapped_problem = problem;
    swapped_problem.A = problem.B;
    swapped_problem.B = problem.A;
    swapped_problem.problem_size = std::make_tuple(N, M, K, L);
    return swapped_problem;
}

template class GemmAllReduceImplRunner<GemmTypes<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t,
    cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

template class GemmAllReduceImplRunner<
    GemmTypes<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
        cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

template class GemmAllReduceImplRunner<
    GemmTypes<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
        cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

template class GemmAllReduceImplRunner<
    GemmTypes<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t, cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>>;

} // namespace tensorrt_llm::kernels::cutlass_kernels
