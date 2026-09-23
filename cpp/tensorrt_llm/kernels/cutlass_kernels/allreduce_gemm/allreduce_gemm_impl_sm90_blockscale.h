/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include "../fp8_blockscale_gemm/ada_blockwise_gemm/sm89_fp8_gemm_1d1d.cuh"
#include "../include/allreduce_gemm_runner.h"
#include "./communication/sm90_allreduce_nvls_warpspecialized.hpp"

#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::opened_cutlass_kernels
{
namespace detail
{

template <typename GemmKernel, typename CollectiveAllReduce>
CUTLASS_GLOBAL void sm90Fp8BlockScaleGemmAllReduceKernel(uint32_t shapeM, uint32_t shapeN, uint32_t shapeK,
    void const* a, void const* b, void* d, float const* scalesA, float const* scalesB,
    typename CollectiveAllReduce::Params allReduceParams)
{
    GemmKernel gemm;
    gemm.invoke(shapeM, shapeN, shapeK, a, b, d, scalesA, scalesB);

    if (allReduceParams.world_size <= 1)
    {
        return;
    }

    // The blockwise GEMM epilogue uses ordinary global stores. Every thread
    // fences its stores before thread 0 publishes this tile to the system-wide
    // ready barrier.
    __threadfence_system();
    __syncthreads();

    int const tileIndex = allReduceParams.tile_layout(blockIdx.x, blockIdx.y);
    using SystemBarrier = typename CollectiveAllReduce::SystemBarrier;
    SystemBarrier::template arrive_inc<cuda::thread_scope::thread_scope_device>(
        allReduceParams.barrier_params, threadIdx.x, tileIndex, allReduceParams.rank, allReduceParams.world_size);

    constexpr uint32_t kNamedBarrier = 0;
    CollectiveAllReduce collectiveAllReduce(allReduceParams, kNamedBarrier);
    auto const problemShape = cute::make_shape(shapeM, shapeN, shapeK, 1);
    auto const tileCoord = cute::make_coord(blockIdx.x, blockIdx.y, 0, 0);
    collectiveAllReduce.gather_reduce_broadcast(problemShape, tileCoord, threadIdx.x);
    collectiveAllReduce.tile_global_sync(problemShape, tileCoord, threadIdx.x);
}

} // namespace detail

//! Hopper FP8 block-scaled GEMM followed by a tile-granular NVLS all-reduce.
//!
//! This consumes the existing Hopper FP8_BLOCK_SCALES representation: E4M3
//! operands, FP32 1x128 activation scales, and FP32 128x128 weight scales.
//! Hopper has no native MXFP8 block-scaled MMA, so each K=128 partial is scaled
//! before it is accumulated by AdaBlockwiseGemmKernel.
template <typename GemmTraits>
class GemmAllReduceImplBlockScaleSm90 : public GemmAllReduceImplInterface
{
public:
    using ElementA = typename GemmTraits::ElementA;
    using ElementB = typename GemmTraits::ElementB;
    using ElementD = typename GemmTraits::ElementD;
    using ElementSFA = typename GemmTraits::ElementSFA;
    using ElementSFB = typename GemmTraits::ElementSFB;
    using LayoutD = typename GemmTraits::LayoutD;
    using TileShapeMNK = typename GemmTraits::TileShape_MNK;

    static_assert(std::is_same_v<ElementA, cutlass::float_e4m3_t>);
    static_assert(std::is_same_v<ElementB, cutlass::float_e4m3_t>);
    static_assert(std::is_same_v<ElementD, cutlass::bfloat16_t>);
    static_assert(std::is_same_v<ElementSFA, float>);
    static_assert(std::is_same_v<ElementSFB, float>);
    static_assert(std::is_same_v<LayoutD, cutlass::layout::RowMajor>);

    static constexpr int kTileM = cute::size<0>(TileShapeMNK{});
    static constexpr int kTileN = cute::size<1>(TileShapeMNK{});
    static constexpr int kTileK = cute::size<2>(TileShapeMNK{});
    static constexpr int kStages = 3;

    using BlockScaleTraits
        = ada_blockwise_gemm::AdaBlockwiseGemmTraits<ElementA, ElementD, float, float, kStages, kTileM, kTileN, kTileK>;
    using BlockScaleGemm = ada_blockwise_gemm::AdaBlockwiseGemmKernel<BlockScaleTraits>;
    using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;
    using TileBarrier = cutlass::MulticastSystemBarrier<cutlass::detail::SyncNoOp, true>;
    using CollectiveAllReduce
        = cutlass::communication::collective::CollectiveAllReduceMulticastWarpSpecialized<ElementD,
            BlockScaleTraits::kThreadCount, 4, TileShapeMNK, StrideD, TileBarrier, LayoutD, false>;

    class PersistentWorkspace : public PersistentWorkspaceInterface
    {
    public:
        using BarrierT = typename TileBarrier::T;

        PersistentWorkspace(int64_t m, int64_t n, runtime::IpcNvlsRendezvousPtr rendezvous)
            : mRendezvous(std::move(rendezvous))
            , mNumElements(m * n)
            , mNumTileBarriers(CollectiveAllReduce::get_num_barrier_flags(m, n))
            , mNumCompletionBarriers(mNumTileBarriers)
        {
            TLLM_CHECK_WITH_INFO(m > 0 && n > 0, "GEMM dimensions must be positive");
            TLLM_CHECK_WITH_INFO(mRendezvous != nullptr, "NVLS rendezvous must not be null");
        }

        void allocate() override
        {
            mTileBarriers.reset(mNumTileBarriers, mRendezvous);
            mCompletionBarriers.reset(mNumCompletionBarriers, mRendezvous);
            if (mRendezvous->size() == 2)
            {
                mStageBuffer.reset(mNumElements, mRendezvous);
            }

            TLLM_CUDA_CHECK(
                cudaMemset(mTileBarriers.getUnicastPointer(), 0, mTileBarriers.getCapacity() * sizeof(BarrierT)));
            TLLM_CUDA_CHECK(cudaMemset(
                mCompletionBarriers.getUnicastPointer(), 0, mCompletionBarriers.getCapacity() * sizeof(BarrierT)));
            mRendezvous->barrier();
        }

        int free() override
        {
            mTileBarriers.free();
            mCompletionBarriers.free();
            mStageBuffer.free();
            return 0;
        }

        size_t size() override
        {
            size_t bytes = (mNumTileBarriers + mNumCompletionBarriers) * sizeof(BarrierT);
            if (mRendezvous->size() == 2)
            {
                bytes += mNumElements * sizeof(ElementD);
            }
            return bytes;
        }

        auto getTileBarrierParams()
        {
            return typename TileBarrier::Params{mTileBarriers.getMulticastPointer(), mTileBarriers.getUnicastPointer()};
        }

        auto getCompletionBarrierParams()
        {
            return typename TileBarrier::Params{
                mCompletionBarriers.getMulticastPointer(), mCompletionBarriers.getUnicastPointer()};
        }

        runtime::IpcNvlsRendezvousPtr mRendezvous;
        size_t mNumElements;
        size_t mNumTileBarriers;
        size_t mNumCompletionBarriers;
        runtime::DeviceAllocationNvls<BarrierT> mTileBarriers;
        runtime::DeviceAllocationNvls<BarrierT> mCompletionBarriers;
        runtime::DeviceAllocationNvls<ElementD> mStageBuffer;
    };

    explicit GemmAllReduceImplBlockScaleSm90(runtime::IpcNvlsRendezvousPtr rendezvous = nullptr)
        : mRendezvous(std::move(rendezvous))
    {
    }

    std::shared_ptr<PersistentWorkspaceInterface> getPersistentWorkspace(ProblemArgs const& maxProblem) override
    {
        auto const [m, n, k, l] = maxProblem.problem_size;
        TLLM_CHECK_WITH_INFO(l == 1, "Batched GEMM is not supported");
        auto rendezvous = mRendezvous ? mRendezvous : runtime::makeMpiIpcNvlsRendezvous(maxProblem.ranks);
        return std::make_shared<PersistentWorkspace>(m, n, std::move(rendezvous));
    }

    int run(ProblemArgs const& problem, cudaStream_t stream) override
    {
        auto const [m, n, k, l] = problem.problem_size;
        TLLM_CHECK_WITH_INFO(l == 1, "Batched GEMM is not supported");
        auto const worldSize = static_cast<int>(problem.ranks.size());
        TLLM_CHECK_WITH_INFO(worldSize >= 1 && worldSize <= CollectiveAllReduce::MaxRanksPerCollective,
            "SM90 FP8 block-scaled GEMM+allreduce supports between 1 and %d ranks",
            CollectiveAllReduce::MaxRanksPerCollective);
        TLLM_CHECK_WITH_INFO(problem.A != nullptr && problem.B != nullptr && problem.D != nullptr,
            "SM90 FP8 block-scaled GEMM requires A, B, and D");
        TLLM_CHECK_WITH_INFO(problem.A_scale != nullptr && problem.B_scale != nullptr,
            "SM90 FP8 block-scaled GEMM requires A and B scales");
        TLLM_CHECK_WITH_INFO(
            problem.C == nullptr && problem.alpha == 1.0F && problem.beta == 0.0F && problem.alpha_ptr == nullptr,
            "SM90 FP8 block-scaled GEMM supports only D = GEMM(A, B)");
        TLLM_CHECK_WITH_INFO(k % BlockScaleTraits::ScaleGranularityK == 0,
            "SM90 FP8 block-scaled GEMM requires K to be a multiple of %d", BlockScaleTraits::ScaleGranularityK);

        auto* workspace = static_cast<PersistentWorkspace*>(problem.workspace);
        TLLM_CHECK_WITH_INFO(workspace != nullptr, "Persistent workspace must not be null");

        ElementD* gemmOutput = reinterpret_cast<ElementD*>(problem.D);
        ElementD* multicastGemmOutput = reinterpret_cast<ElementD*>(problem.D_mc);
        ElementD** ipcGemmOutput = reinterpret_cast<ElementD**>(problem.D_ipc);
        if (workspace->mStageBuffer.getCapacity() > 0)
        {
            gemmOutput = workspace->mStageBuffer.getUnicastPointer();
            multicastGemmOutput = workspace->mStageBuffer.getMulticastPointer();
            ipcGemmOutput = workspace->mStageBuffer.getIpcUnicastPointers();
        }

        auto const strideD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
        typename CollectiveAllReduce::Arguments allReduceArguments{multicastGemmOutput,
            reinterpret_cast<ElementD*>(problem.D_mc), ipcGemmOutput, reinterpret_cast<ElementD**>(problem.D_ipc),
            strideD, workspace->getTileBarrierParams(), workspace->getCompletionBarrierParams(), problem.rank,
            static_cast<int>(problem.ranks.size())};
        auto const allReduceParams
            = CollectiveAllReduce::to_underlying_arguments(cute::make_shape(m, n, k, 1), allReduceArguments);

        dim3 const grid(cute::ceil_div(m, kTileM), cute::ceil_div(n, kTileN), 1);
        dim3 const block(BlockScaleTraits::kThreadCount, 1, 1);
        constexpr int kSmemSize = BlockScaleTraits::kSmemSize;
        auto kernel = detail::sm90Fp8BlockScaleGemmAllReduceKernel<BlockScaleGemm, CollectiveAllReduce>;
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        kernel<<<grid, block, kSmemSize, stream>>>(m, n, k, problem.A, problem.B, gemmOutput,
            reinterpret_cast<float const*>(problem.A_scale), reinterpret_cast<float const*>(problem.B_scale),
            allReduceParams);
        TLLM_CUDA_CHECK(cudaGetLastError());
        return 0;
    }

private:
    runtime::IpcNvlsRendezvousPtr mRendezvous;
};

} // namespace kernels::opened_cutlass_kernels

TRTLLM_NAMESPACE_END
