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
#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cutlass_extensions/gemm_configs.h"

#include "../include/allreduce_gemm_runner.h"
#include "./communication/sm90_allreduce_nvls_warpspecialized.hpp"
#include "./epilogue/sm90_visitor_allreduce_tma_warpspecialized.hpp"
#include "./kernel/sm90_gemm_allreduce_tma_warpspecialized_pingpong.hpp"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels::opened_cutlass_kernels
{
//////////////////////////////////////////////
// Sm90 Two-shot fusion
//////////////////////////////////////////////
template <typename ElementA_, typename ElementB_, typename ElementC_, typename ElementD_, typename ElementSFA_,
    typename ElementSFB_, typename LayoutA_, typename LayoutB_, typename LayoutC_, typename LayoutD_,
    typename TileShape_MNK_, typename ClusterShape_MNK_, typename MainLoopScheduleType_, typename EpilogueScheduleType_>
struct Sm90GemmTypes
{
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementD = ElementD_;
    using ElementSFA = ElementSFA_;
    using ElementSFB = ElementSFB_;
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

    using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
    using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
    using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

    using TileShape_MNK = typename GemmTraits::TileShape_MNK;
    using ClusterShape_MNK = typename GemmTraits::ClusterShape_MNK;

    using MainLoopScheduleType = typename GemmTraits::MainLoopScheduleType;
    using EpilogueScheduleType = typename GemmTraits::EpilogueScheduleType;
    using TileSchedulerType = cutlass::gemm::PersistentScheduler;

    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

    using TileBarrierType = cutlass::MulticastSystemBarrier<cutlass::detail::SyncNoOp, true /* Safe across phases */>;

    // 16B alignment for TMA
    static constexpr int AlignmentA = 16 / sizeof(ElementA);
    static constexpr int AlignmentB = 16 / sizeof(ElementB);
    static constexpr int AlignmentC = 16 / sizeof(ElementC);
    static constexpr int AlignmentD = 16 / sizeof(ElementD);

    ////////////////
    // Epilogue
    ////////////////
    using FusionOp = cutlass::epilogue::fusion::Sm90LinCombAuxAllReduce<OneShot, TileBarrierType, LayoutD, ElementD,
        ElementCompute, ElementC>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp, TileShape_MNK, ClusterShape_MNK,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC, LayoutC,
        AlignmentC, void, LayoutD, AlignmentD, EpilogueScheduleType, FusionOp>::CollectiveOp;

    ////////////////
    // AllReduce
    ////////////////
    static constexpr int CollectiveThreads
        = cute::is_base_of_v<cutlass::gemm::KernelTmaWarpSpecializedCooperative, MainLoopScheduleType> ? 256 : 128;

    using CollectiveAllReduce
        = cutlass::communication::collective::CollectiveAllReduceMulticastWarpSpecialized<ElementD, CollectiveThreads,
            4 /* Unroll */, TileShape_MNK, StrideD, TileBarrierType, LayoutD, OneShot>;

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
    using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

    class PersistentWorkspace : public PersistentWorkspaceInterface
    {
    public:
        using BarrierT = typename TileBarrierType::T;

        PersistentWorkspace(int64_t M, int64_t N, std::set<int> ranks)
            : _ranks(ranks)
            , _num_elements(M * N)
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
            if (_ranks.size() == 2)
            {
                _stage_buf.reset(_num_elements, _ranks);
            }

            TLLM_CUDA_CHECK(
                cudaMemset(_tile_barriers.getUnicastPointer(), 0, _tile_barriers.getCapacity() * sizeof(BarrierT)));
            TLLM_CUDA_CHECK(cudaMemset(
                _completion_barriers.getUnicastPointer(), 0, _completion_barriers.getCapacity() * sizeof(BarrierT)));

            // Ensure local memset is visible across all processes
            if (_ranks.size() > 1)
            {
                MPI_group_barrier(_ranks);
            }
        }

        int free() override
        {
            _tile_barriers.free();
            _completion_barriers.free();
            return 0;
        }

        size_t size() override
        {
            size_t size_bytes = 0;
            size_bytes += _num_tile_barriers * sizeof(BarrierT);
            size_bytes += _num_completion_barriers * sizeof(BarrierT);
            if (_ranks.size() == 2)
            {
                size_bytes += _num_elements * sizeof(ElementD);
            }
            return size_bytes;
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
        size_t _num_elements = 0;
        std::set<int> _ranks;
        DeviceAllocationNvls<BarrierT> _tile_barriers;
        DeviceAllocationNvls<BarrierT> _completion_barriers;
        DeviceAllocationNvls<ElementD> _stage_buf;
    };

    GemmAllReduceImplTwoshot_Sm90()
    {
        int device_id = -1;
        TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
        _hw_info.device_id = device_id;
        _hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(_hw_info.device_id);
    }

    std::shared_ptr<PersistentWorkspaceInterface> getPersistentWorkspace(ProblemArgs const& max_problem) override
    {
        auto [M, N, K, L] = max_problem.problem_size;
        assert(L == 1 && "batched GEMM not supported yet.");
        return std::make_shared<PersistentWorkspace>(M, N, max_problem.ranks);
    }

    int run(ProblemArgs const& problem, cudaStream_t stream) override
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
    auto getArgs(ProblemArgs const& problem) const
    {
        bool const skip_AR = problem.ranks.size() == 1; // single-GPU doesn't need AR

        auto [M, N, K, L] = problem.problem_size;
        auto workspace = static_cast<PersistentWorkspace*>(problem.workspace);
        assert(L == 1);
        assert((skip_AR && workspace == nullptr) || workspace != nullptr);

        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

        ElementD* ptr_D = reinterpret_cast<ElementD*>(problem.D);
        ElementD* mc_ptr_D = reinterpret_cast<ElementD*>(problem.D_mc);
        ElementD* mc_ptr_out = reinterpret_cast<ElementD*>(problem.D_mc);
        ElementD** ipc_ptr_D = reinterpret_cast<ElementD**>(problem.D_ipc);
        ElementD** ipc_ptr_out = reinterpret_cast<ElementD**>(problem.D_ipc);

        // Pointers to GEMM output
        if (workspace->_stage_buf.getCapacity() > 0)
        {
            ptr_D = workspace->_stage_buf.getUnicastPointer();
            mc_ptr_D = workspace->_stage_buf.getMulticastPointer();
            ipc_ptr_D = workspace->_stage_buf.getIpcUnicastPointers();
        }

        return typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K},
            {// Mainloop arguments
                reinterpret_cast<ElementA const*>(problem.A), stride_A, reinterpret_cast<ElementB const*>(problem.B),
                stride_B},
            {// Epilogue arguments
                {problem.alpha, problem.beta, problem.alpha_ptr, nullptr, mc_ptr_D, ptr_D, stride_D,
                    skip_AR ? typename TileBarrierType::Params{} : workspace->getTileBarrierParams(), problem.rank,
                    static_cast<int>(problem.ranks.size())},
                reinterpret_cast<ElementC const*>(problem.C), stride_C, (ElementD*) nullptr, stride_D},
            {// AllReduce arguments
                mc_ptr_D, mc_ptr_out, ipc_ptr_D, ipc_ptr_out, stride_D,
                skip_AR ? typename TileBarrierType::Params{} : workspace->getTileBarrierParams(),
                skip_AR ? typename TileBarrierType::Params{} : workspace->getCompletionBarrierParams(), problem.rank,
                static_cast<int>(problem.ranks.size())},
            _hw_info,
            {// TileScheduler arguments
                1, RasterOrderOptions::AlongN}};
    }

    // Holds the number of SMs on the GPU.
    // This information is used by the underlying kernel.
    cutlass::KernelHardwareInfo _hw_info;
};

} // namespace tensorrt_llm::kernels::opened_cutlass_kernels
