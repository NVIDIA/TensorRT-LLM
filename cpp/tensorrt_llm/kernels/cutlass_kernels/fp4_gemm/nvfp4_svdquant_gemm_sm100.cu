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

namespace tensorrt_llm::kernels::cutlass_kernels
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
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
static constexpr int AlignA = 32;
static constexpr int AlignB = 32;
static constexpr int AlignC = 8;

// Tile 128x256x128, 1-CTA cluster -- the residual uses the stock nvfp4 GEMM's fast tile.
using MmaTileShape = Shape<_128, _256, _128>;
using ClusterShape = Shape<int, int, _1>;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100;

// LinearCombination epilogue (out = alpha*acc); the rank-r LoRA-up is fused in the mainloop, not
// the epilogue.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<Arch,
    cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape, EpilogueTileType, ElementAccumulator, ElementCompute,
    void, LayoutC, AlignC, OutElementType, LayoutC, AlignC, EpilogueSchedule>::CollectiveOp;

// Build the STANDARD blockscaled mainloop via the Builder, then re-instantiate our
// renamed CollectiveMmaLoRA with its 15 (extracted) template args.
using CollectiveMainloopBase = typename cutlass::gemm::collective::CollectiveBuilder<Arch,
    cutlass::arch::OpClassBlockScaledTensorOp, cute::tuple<ElementType, SFType>, LayoutA, AlignA,
    cute::tuple<ElementType, SFType>, LayoutB, AlignB, ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage)
        /* No extra LoRA carveout: D/L1 reinterpret the residual stage buffers smem_A/smem_B
           (byte-exact overlay, no dedicated smem), so this collective's SharedStorage matches the
           base nvfp4 collective and the residual keeps its full mainloop stage count. */)>,
    MainloopSchedule>::CollectiveOp;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMmaLoRA<typename CollectiveMainloopBase::DispatchPolicy,
    typename CollectiveMainloopBase::TileShape, typename CollectiveMainloopBase::ElementPairA,
    typename CollectiveMainloopBase::StridePairA, typename CollectiveMainloopBase::ElementPairB,
    typename CollectiveMainloopBase::StridePairB, typename CollectiveMainloopBase::TiledMma,
    typename CollectiveMainloopBase::GmemTiledCopyPairA, typename CollectiveMainloopBase::SmemLayoutAtomPairA,
    typename CollectiveMainloopBase::SmemCopyAtomA, typename CollectiveMainloopBase::TransformA,
    typename CollectiveMainloopBase::GmemTiledCopyPairB, typename CollectiveMainloopBase::SmemLayoutAtomPairB,
    typename CollectiveMainloopBase::SmemCopyAtomB, typename CollectiveMainloopBase::TransformB>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
    CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
} // namespace

size_t nvfp4_svdquant_gemm_workspace_size(int m, int n, int k)
{
    Gemm gemm;
    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = cute::make_shape(m, n, k, 1);
    return gemm.get_workspace_size(args);
}

// Fused residual NVFP4 GEMM + rank-r LoRA-up: D @ L1ᵀ via the 2nd bf16 tcgen05 MMA in the custom
// collective. 1/alpha folded into L1 so the epilogue out = alpha*acc yields alpha*residual + D@L1ᵀ.
void nvfp4_svdquant_gemm_run(void* out, void const* A, void const* B, void const* sfa, void const* sfb,
    float const* alpha, void const* D, void const* L1, int m, int n, int k, char* ws, size_t wsBytes,
    cudaStream_t stream)
{
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    typename Gemm::Arguments args;
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
    // LoRA D[M,LoRaK] / L1[N,LoRaK] (bf16, row-major K-contiguous). 1/alpha folded into L1 host-side.
    constexpr int64_t LoRaK = 32; // must match CollectiveMmaLoRA::LoRaK
    args.mainloop.ptr_D = static_cast<cutlass::bfloat16_t const*>(D);
    args.mainloop.dD = cute::make_stride(LoRaK, cute::_1{}, int64_t(m) * LoRaK);
    args.mainloop.ptr_L1 = static_cast<cutlass::bfloat16_t const*>(L1);
    args.mainloop.dL1 = cute::make_stride(LoRaK, cute::_1{}, int64_t(n) * LoRaK);

    args.epilogue.ptr_C = nullptr;
    args.epilogue.ptr_D = static_cast<OutElementType*>(out);
    args.epilogue.dC = cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideC>(n, 0);
    args.epilogue.dD = args.epilogue.dC;
    // LinearCombination: out = alpha*acc (alpha via device scalar ptr, beta=0).
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    args.epilogue.thread.alpha_ptr = alpha;
    args.epilogue.thread.beta_ptr = nullptr;

    if constexpr (!std::is_const_v<decltype(args.scheduler.max_swizzle_size)>)
        args.scheduler.max_swizzle_size = 1;
    if constexpr (!std::is_const_v<decltype(args.scheduler.raster_order)>)
    {
        using Enum_t = decltype(args.scheduler.raster_order);
        args.scheduler.raster_order = Enum_t::Heuristic;
    }
    args.hw_info.cluster_shape = dim3(1, 1, 1);
    args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);

    Gemm gemm;
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
} // namespace tensorrt_llm::kernels::cutlass_kernels
