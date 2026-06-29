/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
// SVDQuant fused NVFP4 GEMM launcher (SM100): build the standard block-scaled NVFP4 mainloop via
// the CUTLASS CollectiveBuilder, then re-instantiate the custom CollectiveMmaLoRA (a renamed copy
// of the SM100 block-scaled warp-specialized collective extended with the rank-r LoRA-up MMA) with
// the Builder's 15 template typedefs, and run it via GemmUniversal. The stock nvfp4 GEMM template
// (nvfp4_nvfp4_gemm_template_sm100.h) is untouched.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/archCondition.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

#include "nvfp4_svdquant_gemm_collective_sm100.h" // CollectiveMmaLoRA (must precede the Builder use)
#include "tensorrt_llm/kernels/cutlass_kernels/include/nvfp4_svdquant_gemm.h"
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{
namespace
{
using namespace cute;

using Arch = cutlass::arch::Sm100;
using ElementType = cutlass::float_e2m1_t; // NVFP4 operands
using SFType = cutlass::float_ue4m3_t;
using OutElementType = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementC = void;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
static constexpr int AlignA = 32;
static constexpr int AlignB = 32;
static constexpr int AlignC = 8;

using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

// The rank-r LoRA-up is fused in the mainloop. Match the stock NVFP4 epilogue so the Qwen
// per-column bias is added without a separate bandwidth-bound kernel.
using FusionOperation
    = cutlass::epilogue::fusion::LinCombPerColBias<OutElementType, float, OutElementType, ElementC, float>;

// Every kernel shape uses a dynamic cluster type so each tile can benchmark stock-compatible
// runtime cluster shapes without multiplying the generated kernel variants.
template <class MmaTileShape_, class EpilogueSchedule_, class MainloopSchedule_>
struct SvdquantGemmConfig
{
    using MmaTileShape = MmaTileShape_;
    using EpilogueSchedule = EpilogueSchedule_;
    using MainloopSchedule = MainloopSchedule_;
    using ClusterShape = Shape<int, int, _1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<Arch,
        cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
        ElementCompute, ElementC, LayoutC, AlignC, OutElementType, LayoutC, AlignC, EpilogueSchedule,
        FusionOperation>::CollectiveOp;

    // Build the standard block-scaled mainloop, then re-instantiate CollectiveMmaLoRA with the
    // builder's extracted template arguments. D/L1 overlay the residual A/B stage buffers, so no
    // extra shared-memory carveout is needed for either the 1SM or 2SM tactic.
    using CollectiveMainloopBase = typename cutlass::gemm::collective::CollectiveBuilder<Arch,
        cutlass::arch::OpClassBlockScaledTensorOp, cute::tuple<ElementType, SFType>, LayoutA, AlignA,
        cute::tuple<ElementType, SFType>, LayoutB, AlignB, ElementAccumulator, MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

    using CollectiveMainloop
        = cutlass::gemm::collective::CollectiveMmaLoRA<typename CollectiveMainloopBase::DispatchPolicy,
            typename CollectiveMainloopBase::TileShape, typename CollectiveMainloopBase::ElementPairA,
            typename CollectiveMainloopBase::StridePairA, typename CollectiveMainloopBase::ElementPairB,
            typename CollectiveMainloopBase::StridePairB, typename CollectiveMainloopBase::TiledMma,
            typename CollectiveMainloopBase::GmemTiledCopyPairA,
            typename CollectiveMainloopBase::SmemLayoutAtomPairA, typename CollectiveMainloopBase::SmemCopyAtomA,
            typename CollectiveMainloopBase::TransformA, typename CollectiveMainloopBase::GmemTiledCopyPairB,
            typename CollectiveMainloopBase::SmemLayoutAtomPairB, typename CollectiveMainloopBase::SmemCopyAtomB,
            typename CollectiveMainloopBase::TransformB>;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
        CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Tactic1Sm128x256x128Config
    = SvdquantGemmConfig<Shape<_128, _256, _128>, cutlass::epilogue::TmaWarpSpecialized1Sm,
        cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>;
using Tactic2Sm256x256x128Config
    = SvdquantGemmConfig<Shape<_256, _256, _128>, cutlass::epilogue::TmaWarpSpecialized2Sm,
        cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>;
using Tactic1Sm128x128x128Config
    = SvdquantGemmConfig<Shape<_128, _128, _128>, cutlass::epilogue::TmaWarpSpecialized1Sm,
        cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>;
using Tactic2Sm256x192x128Config
    = SvdquantGemmConfig<Shape<_256, _192, _128>, cutlass::epilogue::TmaWarpSpecialized2Sm,
        cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100>;

enum class KernelShape
{
    k1Sm128x256x128,
    k2Sm256x256x128,
    k1Sm128x128x128,
    k2Sm256x192x128,
};

struct RuntimeTactic
{
    KernelShape kernel_shape;
    dim3 cluster_shape;
    dim3 cluster_shape_fallback;
};

RuntimeTactic resolve_tactic(int tactic)
{
    // A 1SM kernel may fall back to 1x1. A 2SM instruction always requires at least its 2x1 MMA
    // group, so every 2SM cluster uses 2x1 as its fallback.
    switch (static_cast<Nvfp4SvdquantGemmTactic>(tactic))
    {
    case Nvfp4SvdquantGemmTactic::k1Sm128x256x128:
        return {KernelShape::k1Sm128x256x128, dim3(1, 1, 1), dim3(1, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k2Sm256x256x128:
        return {KernelShape::k2Sm256x256x128, dim3(2, 1, 1), dim3(2, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k1Sm128x256x128Cluster1x2:
        return {KernelShape::k1Sm128x256x128, dim3(1, 2, 1), dim3(1, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k1Sm128x256x128Cluster1x4:
        return {KernelShape::k1Sm128x256x128, dim3(1, 4, 1), dim3(1, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k2Sm256x256x128Cluster2x2:
        return {KernelShape::k2Sm256x256x128, dim3(2, 2, 1), dim3(2, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k2Sm256x256x128Cluster2x4:
        return {KernelShape::k2Sm256x256x128, dim3(2, 4, 1), dim3(2, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k2Sm256x256x128Cluster4x2:
        return {KernelShape::k2Sm256x256x128, dim3(4, 2, 1), dim3(2, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k2Sm256x256x128Cluster4x4:
        return {KernelShape::k2Sm256x256x128, dim3(4, 4, 1), dim3(2, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k2Sm256x256x128Cluster4x1:
        return {KernelShape::k2Sm256x256x128, dim3(4, 1, 1), dim3(2, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k1Sm128x128x128:
        return {KernelShape::k1Sm128x128x128, dim3(1, 1, 1), dim3(1, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k1Sm128x128x128Cluster1x2:
        return {KernelShape::k1Sm128x128x128, dim3(1, 2, 1), dim3(1, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k2Sm256x192x128:
        return {KernelShape::k2Sm256x192x128, dim3(2, 1, 1), dim3(2, 1, 1)};
    case Nvfp4SvdquantGemmTactic::k2Sm256x192x128Cluster2x2:
        return {KernelShape::k2Sm256x192x128, dim3(2, 2, 1), dim3(2, 1, 1)};
    default: throw std::invalid_argument("nvfp4_svdquant_gemm: invalid tactic");
    }
}

template <class Config>
size_t workspace_size_for_tactic(int m, int n, int k, RuntimeTactic const& tactic)
{
    using Gemm = typename Config::Gemm;
    Gemm gemm;
    typename Gemm::Arguments args{};
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = cute::make_shape(m, n, k, 1);
    args.hw_info.cluster_shape = tactic.cluster_shape;
    args.hw_info.cluster_shape_fallback = tactic.cluster_shape_fallback;
    return gemm.get_workspace_size(args);
}

template <class Config>
void run_tactic(void* out, void const* A, void const* B, void const* sfa, void const* sfb, float const* alpha,
    void const* D, int64_t dStride, void const* L1, void const* bias, int m, int n, int k, char* ws, size_t wsBytes,
    cudaStream_t stream, RuntimeTactic const& tactic)
{
    using Gemm = typename Config::Gemm;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    typename Gemm::Arguments args{};
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = cute::make_shape(m, n, k, 1);

    args.mainloop.ptr_A = static_cast<ElementType const*>(A);
    args.mainloop.ptr_B = static_cast<ElementType const*>(B);
    args.mainloop.ptr_SFA = static_cast<SFType const*>(sfa);
    args.mainloop.ptr_SFB = static_cast<SFType const*>(sfb);
    args.mainloop.dA = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideA>(k, 0);
    args.mainloop.dB = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideB>(k, 0);
    args.mainloop.layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(args.problem_shape);
    args.mainloop.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(args.problem_shape);
    // LoRA D[M,LoRaK] / L1[N,LoRaK] (bf16, row-major K-contiguous). 1/alpha is folded into L1.
    constexpr int64_t LoRaK = 32; // must match CollectiveMmaLoRA::LoRaK
    args.mainloop.ptr_D = static_cast<cutlass::bfloat16_t const*>(D);
    args.mainloop.dD = cute::make_stride(dStride, cute::_1{}, int64_t(m) * dStride);
    args.mainloop.ptr_L1 = static_cast<cutlass::bfloat16_t const*>(L1);
    args.mainloop.dL1 = cute::make_stride(LoRaK, cute::_1{}, int64_t(n) * LoRaK);

    args.epilogue.ptr_C = nullptr;
    args.epilogue.ptr_D = static_cast<OutElementType*>(out);
    args.epilogue.dC = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideC>(n, 0);
    args.epilogue.dD = args.epilogue.dC;
    // out = alpha * acc + bias[N] (alpha via device scalar pointer). A null bias is a no-op.
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    args.epilogue.thread.alpha_ptr = alpha;
    args.epilogue.thread.beta_ptr = nullptr;
    args.epilogue.thread.bias_ptr = static_cast<OutElementType const*>(bias);

    if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>)
        args.scheduler.max_swizzle_size = 1;
    if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>)
    {
        using Enum_t = decltype(args.scheduler.raster_order);
        args.scheduler.raster_order = Enum_t::Heuristic;
    }
    args.hw_info.cluster_shape = tactic.cluster_shape;
    args.hw_info.cluster_shape_fallback = tactic.cluster_shape_fallback;

    Gemm gemm;
    size_t const requiredWorkspaceBytes = gemm.get_workspace_size(args);
    if (wsBytes < requiredWorkspaceBytes)
        throw std::invalid_argument("svdquant fused_mma: insufficient workspace");
    auto st = gemm.can_implement(args);
    if (st != cutlass::Status::kSuccess)
        throw std::runtime_error("svdquant fused_mma: can_implement failed");
    st = gemm.initialize(args, ws, stream);
    if (st != cutlass::Status::kSuccess)
        throw std::runtime_error("svdquant fused_mma: initialize failed");
    st = gemm.run(args, ws, stream, nullptr, tensorrt_llm::common::getEnvEnablePDL());
    if (st != cutlass::Status::kSuccess)
        throw std::runtime_error("svdquant fused_mma: run failed");
}
} // namespace

size_t nvfp4_svdquant_gemm_workspace_size(int m, int n, int k, int tactic)
{
    RuntimeTactic const runtime_tactic = resolve_tactic(tactic);
    switch (runtime_tactic.kernel_shape)
    {
    case KernelShape::k1Sm128x256x128:
        return workspace_size_for_tactic<Tactic1Sm128x256x128Config>(m, n, k, runtime_tactic);
    case KernelShape::k2Sm256x256x128:
        return workspace_size_for_tactic<Tactic2Sm256x256x128Config>(m, n, k, runtime_tactic);
    case KernelShape::k1Sm128x128x128:
        return workspace_size_for_tactic<Tactic1Sm128x128x128Config>(m, n, k, runtime_tactic);
    case KernelShape::k2Sm256x192x128:
        return workspace_size_for_tactic<Tactic2Sm256x192x128Config>(m, n, k, runtime_tactic);
    }
    throw std::invalid_argument("nvfp4_svdquant_gemm_workspace_size: invalid kernel shape");
}

// Fused residual NVFP4 GEMM + rank-r LoRA-up: D @ L1ᵀ via the 2nd bf16 tcgen05 MMA in the custom
// collective. 1/alpha folded into L1 so the epilogue yields alpha*residual + D@L1ᵀ + bias.
void nvfp4_svdquant_gemm_run(void* out, void const* A, void const* B, void const* sfa, void const* sfb,
    float const* alpha, void const* D, void const* L1, void const* bias, int m, int n, int k, char* ws,
    size_t wsBytes, cudaStream_t stream, int tactic, int64_t dStride)
{
    RuntimeTactic const runtime_tactic = resolve_tactic(tactic);
    switch (runtime_tactic.kernel_shape)
    {
    case KernelShape::k1Sm128x256x128:
        return run_tactic<Tactic1Sm128x256x128Config>(
            out, A, B, sfa, sfb, alpha, D, dStride, L1, bias, m, n, k, ws, wsBytes, stream, runtime_tactic);
    case KernelShape::k2Sm256x256x128:
        return run_tactic<Tactic2Sm256x256x128Config>(
            out, A, B, sfa, sfb, alpha, D, dStride, L1, bias, m, n, k, ws, wsBytes, stream, runtime_tactic);
    case KernelShape::k1Sm128x128x128:
        return run_tactic<Tactic1Sm128x128x128Config>(
            out, A, B, sfa, sfb, alpha, D, dStride, L1, bias, m, n, k, ws, wsBytes, stream, runtime_tactic);
    case KernelShape::k2Sm256x192x128:
        return run_tactic<Tactic2Sm256x192x128Config>(
            out, A, B, sfa, sfb, alpha, D, dStride, L1, bias, m, n, k, ws, wsBytes, stream, runtime_tactic);
    }
    throw std::invalid_argument("nvfp4_svdquant_gemm_run: invalid kernel shape");
}
} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
