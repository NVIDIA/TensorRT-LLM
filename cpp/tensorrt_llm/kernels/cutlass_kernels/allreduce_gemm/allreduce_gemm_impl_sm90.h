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
#pragma once

#include "allreduce_gemm_runner.h"

#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_extensions/gemm_configs.h"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass_extensions/communication/collective/sm90_allreduce_nvls_warpspecialized.hpp"
#include "cutlass_extensions/epilogue/fusion/sm90_visitor_allreduce_tma_warpspecialized.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_allreduce_tma_warpspecialized.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_allreduce_tma_warpspecialized_cooperative.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_allreduce_tma_warpspecialized_pingpong.hpp"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels::cutlass_kernels
{
constexpr int kGemmAllReduceOneShotMaxSizeBytes = 128 * 1024; // 128KiB

//////////////////////////////////////////////
// Sm90 Two-shot fusion
//////////////////////////////////////////////
template <typename ElementA_, typename ElementB_, typename ElementC_, typename ElementD_, typename LayoutA_,
    typename LayoutB_, typename LayoutC_, typename LayoutD_, typename TileShape_MNK_, typename ClusterShape_MNK_,
    typename MainLoopScheduleType_, typename EpilogueScheduleType_>
struct CutlassGemmTypes
{
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementD = ElementD_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutD = LayoutD_;
    using TileShape_MNK = TileShape_MNK_;
    using ClusterShape_MNK = ClusterShape_MNK_;
    using MainLoopScheduleType = MainLoopScheduleType_;
    using EpilogueScheduleType = EpilogueScheduleType_;
};

template <typename GemmTraits, bool OneShot>
class GemmAllReduceImplTwoshot_Sm90 : public GemmAllReduceImplInterface
{
public:
    using ElementA = typename GemmTraits::ElementA;
    using ElementB = typename GemmTraits::ElementB;
    using ElementC = typename GemmTraits::ElementC;
    using ElementD = typename GemmTraits::ElementD;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementScalar = float;

    using LayoutA = typename GemmTraits::LayoutA;
    using LayoutB = typename GemmTraits::LayoutB;
    using LayoutC = typename GemmTraits::LayoutC;
    using LayoutD = typename GemmTraits::LayoutD;
    using TileShape_MNK = typename GemmTraits::TileShape_MNK;
    using ClusterShape_MNK = typename GemmTraits::ClusterShape_MNK;

    using MainLoopScheduleType = typename GemmTraits::MainLoopScheduleType;
    using EpilogueScheduleType = typename GemmTraits::EpilogueScheduleType;
    using TileSchedulerType = cutlass::gemm::PersistentScheduler;

    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

    // 16B alignment for TMA
    static constexpr int AlignmentA = 16 / sizeof(ElementA);
    static constexpr int AlignmentB = 16 / sizeof(ElementB);
    static constexpr int AlignmentC = 16 / sizeof(ElementC);
    static constexpr int AlignmentD = 16 / sizeof(ElementD);

    ////////////////
    // AuxStore EVT
    ////////////////
    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using EVT_D = cutlass::epilogue::fusion::Sm90LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar,
        RoundStyle>; // beta * C + (alpha * acc)

    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueDescriptor = cutlass::epilogue::collective::detail::EpilogueDescriptor<TileShape_MNK,
        EpilogueTileType, ElementC, ElementD, EpilogueScheduleType>;

    using AuxStoreDescriptor
        = cutlass::epilogue::collective::detail::AuxStoreDescriptor<EpilogueDescriptor, LayoutD, ElementD>;

    using TileBarrierType = cutlass::MulticastSystemBarrier<cutlass::detail::SyncNoOp, true /* Safe across phases */>;

    using AuxStore = cutlass::epilogue::fusion::Sm90AuxStoreReduceWarpSpecialized<AuxStoreDescriptor::Stages,
        typename EpilogueDescriptor::EpilogueTile, typename AuxStoreDescriptor::Element,
        typename AuxStoreDescriptor::Stride, typename AuxStoreDescriptor::SmemLayoutAtom, RoundStyle,
        typename AuxStoreDescriptor::CopyOpR2S, TileShape_MNK, TileBarrierType, OneShot>;

    using FusionCallbacks = cutlass::epilogue::fusion::Sm90EVT<AuxStore, EVT_D>;

    ////////////////
    // Epilogue
    ////////////////
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp, TileShape_MNK, ClusterShape_MNK, EpilogueTileType, ElementAccumulator,
        ElementCompute, ElementC, LayoutC, AlignmentC, void, LayoutD, AlignmentD, // set to void because using EVT
        EpilogueScheduleType, FusionCallbacks>::CollectiveOp;

    ////////////////
    // AllReduce
    ////////////////
    static constexpr int CollectiveThreads
        = cute::is_base_of_v<cutlass::gemm::KernelTmaWarpSpecializedCooperative, MainLoopScheduleType> ? 256 : 128;

    using CollectiveAllReduce = cutlass::communication::collective::CollectiveAllReduceMulticastWarpSpecialized<
        typename AuxStoreDescriptor::Element, CollectiveThreads, TileShape_MNK, typename AuxStoreDescriptor::Stride,
        TileBarrierType, LayoutD, OneShot>;

    ////////////////
    // Mainloop
    ////////////////
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
        ElementAccumulator, TileShape_MNK, ClusterShape_MNK,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainLoopScheduleType>::CollectiveOp;

    ////////////////
    // GemmKernel
    ////////////////
    using GemmKernel = cutlass::gemm::kernel::GemmARUniversal<cute::Shape<int, int, int>, CollectiveMainloop,
        CollectiveEpilogue, CollectiveAllReduce, TileSchedulerType>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

    class PersistentWorkspace : public PersistentWorkspaceInterface
    {
    public:
        using BarrierT = typename TileBarrierType::T;

        PersistentWorkspace(int64_t M, int64_t N, std::set<int> ranks)
            : _ranks(ranks)
        {
            assert(M > 0 && "M is 0.");
            assert(N > 0 && "N is 0.");
            // barriers to know when each tile ready to be consumed by AR
            _num_tile_barriers = CollectiveAllReduce::get_num_barrier_flags(M, N);
            // need barrier per warpgroup to indicate broadcast complete, this is safe.
            _num_completion_barriers = _num_tile_barriers;
        }

        /////////////////////////////////////////
        // PersistentWorkspaceInterface Methods
        /////////////////////////////////////////
        void allocate() override
        {
            _tile_barriers.reset(_num_tile_barriers, _ranks);
            _completion_barriers.reset(_num_completion_barriers, _ranks);

            TLLM_CUDA_CHECK(
                cudaMemset(_tile_barriers.getUnicastPointer(), 0, _tile_barriers.getCapacity() * sizeof(BarrierT)));
            TLLM_CUDA_CHECK(cudaMemset(
                _completion_barriers.getUnicastPointer(), 0, _completion_barriers.getCapacity() * sizeof(BarrierT)));

            // Ensure local memset is visible across all processes
            if (_ranks.size() > 1)
            {
                MPI_group_barrier(_ranks);
            }

            TLLM_CUDA_CHECK(cudaStreamCreate(&_memcpy_stream));
            TLLM_CUDA_CHECK(cudaEventCreate(&_fork_join_event));
        }

        int free() override
        {
            _tile_barriers.free();
            _completion_barriers.free();
            return 0;
        }

        /////////////////////////////////////////
        // Methods used by GemmAllReduceImpl
        /////////////////////////////////////////
        auto getTileBarrierParams()
        {
            return typename CollectiveAllReduce::SystemBarrier::Params{
                _tile_barriers.getMulticastPointer(), _tile_barriers.getUnicastPointer()};
        };

        auto getCompletionBarrierParams()
        {
            return typename CollectiveAllReduce::SystemBarrier::Params{
                _completion_barriers.getMulticastPointer(), _completion_barriers.getUnicastPointer()};
        };

        size_t _num_tile_barriers = 0;
        size_t _num_completion_barriers = 0;
        std::set<int> _ranks;
        DeviceAllocationNvls<BarrierT> _tile_barriers;
        DeviceAllocationNvls<BarrierT> _completion_barriers;
        cudaStream_t _memcpy_stream;
        cudaEvent_t _fork_join_event;
    };

    GemmAllReduceImplTwoshot_Sm90()
    {
        int device_id = -1;
        TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
        _hw_info.device_id = device_id;
        _hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(_hw_info.device_id);
    }

    std::shared_ptr<PersistentWorkspaceInterface> getPersistentWorkspace(ProblemArgs const& max_problem)
    {
        auto [M, N, K, L] = max_problem.problem_size;
        assert(L == 1 && "batched GEMM not supported yet.");
        return std::make_shared<PersistentWorkspace>(M, N, max_problem.ranks);
    }

    int run(ProblemArgs const& problem, cudaStream_t stream)
    {
        Gemm gemm;
        auto arguments = getArgs(problem);

        size_t workspace_size = gemm.get_workspace_size(arguments);
        TLLM_CHECK_WITH_INFO(workspace_size == 0, "Gemm workspace is not 0 bytes.");

        auto status = gemm.can_implement(arguments);
        TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
            "This kernel is not supported. Last CUDA error is: %s", cutlassGetStatusString(status));

        status = gemm.initialize(arguments, nullptr, stream);
        TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
            "Failed to initialize the CUTLASS kernel. Last CUDA error is: %s", cutlassGetStatusString(status));

        status = gemm.run(stream);
        TLLM_CHECK_WITH_INFO(status == cutlass::Status::kSuccess,
            "Failed to run the CUTLASS kernel. Last CUDA error is: %s", cutlassGetStatusString(status));

        return 0;
    }

private:
    auto getArgs(ProblemArgs const& problem)
    {
        bool skip_AR = problem.ranks.size() == 1; // single-GPU doesn't need AR

        auto [M, N, K, L] = problem.problem_size;
        auto workspace = static_cast<PersistentWorkspace*>(problem.workspace);
        assert(L == 1);
        assert((skip_AR && workspace == nullptr) || workspace != nullptr);

        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

        auto arguments = typename Gemm::Arguments{};

        auto get_aux_store_args = [&](/* int output_stage */)
        {
            return typename AuxStore::Arguments{(ElementD*) problem.D_mc, (ElementD*) problem.D, stride_D,
                skip_AR ? typename TileBarrierType::Params{} : workspace->getTileBarrierParams(), problem.rank,
                int(problem.ranks.size())};
        };

        auto get_epilogue_args = [&](/* int output_stage */)
        {
            typename GemmKernel::EpilogueArguments epilogue{{}, (ElementD*) problem.C, stride_C, nullptr, stride_D};

            epilogue.thread = {// unary op: aux store D
                {
                    // ternary op : beta * C + (alpha * acc)
                    {{problem.beta}}, // leaf op+args : beta
                    {},               // leaf op+args : C
                    {
                        // binary op : alpha * acc
                        {{problem.alpha}}, // leaf op+args : alpha
                        {},                // leaf op+args : acc
                        {}                 // binary args : multiplies
                    },                     // end binary op
                    {}                     // ternary args : multiply_add
                },
                // aux store D
                get_aux_store_args()};

            return epilogue;
        };

        auto get_collective_allreduce_args = [&](/* int output_stage */)
        {
            return typename CollectiveAllReduce::Arguments{(ElementD*) problem.D_mc, (ElementD*) problem.D, stride_D,
                skip_AR ? typename TileBarrierType::Params{} : workspace->getTileBarrierParams(),
                skip_AR ? typename TileBarrierType::Params{} : workspace->getCompletionBarrierParams(), problem.rank,
                int(problem.ranks.size())};
        };

        arguments.mode = cutlass::gemm::GemmUniversalMode::kGemm;
        arguments.problem_shape = ProblemShapeType{M, N, K};
        arguments.mainloop = {(ElementA*) problem.A, stride_A, (ElementB*) problem.B, stride_B};
        arguments.epilogue = get_epilogue_args();
        arguments.hw_info = _hw_info;
        arguments.all_reduce = get_collective_allreduce_args();
        arguments.scheduler = {};
        arguments.scheduler.raster_order = RasterOrderOptions::AlongN;
        // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
        arguments.scheduler.max_swizzle_size = 1;

        return arguments;
    }

    // Holds the number of SMs on the GPU.
    // This information is used by the underlying kernel.
    cutlass::KernelHardwareInfo _hw_info;
};

} // namespace tensorrt_llm::kernels::cutlass_kernels
