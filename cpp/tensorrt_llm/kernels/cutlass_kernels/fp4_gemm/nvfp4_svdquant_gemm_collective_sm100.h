/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective
{
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// SVDQuant LoRA-fused mainloop: a copy of the SM100 block-scaled warp-specialized CollectiveMma,
// renamed CollectiveMmaLoRA so it coexists with the unmodified original (a distinct type, not a
// CollectiveMma specialization, so there is no ODR clash). The stock nvfp4 GEMM template is
// untouched. It is instantiated via extract-from-Builder: each of the 15 template args is a public
// typedef on the standard Builder's CollectiveOp (DispatchPolicy / TileShape / ElementPairA / ...),
// so the launcher re-instantiates CollectiveMmaLoRA<BC::DispatchPolicy, BC::TileShape, ...>.
// The only functional change vs the base collective is the rank-r LoRA-up: D @ L1ᵀ is accumulated
// by a 2nd bf16 tcgen05 MMA (K = r) into the same TMEM accumulator after the NVFP4 K-loop, with
// D/L1 riding the residual stage buffers (no dedicated smem). The LoRA-up is always applied.
template <class DispatchPolicy, class TileShape_, class ElementPairA_, class StridePairA_, class ElementPairB_,
    class StridePairB_, class TiledMma_, class GmemTiledCopyPairA_, class SmemLayoutAtomPairA_, class SmemCopyAtomA_,
    class TransformA_, class GmemTiledCopyPairB_, class SmemLayoutAtomPairB_, class SmemCopyAtomB_, class TransformB_>
struct CollectiveMmaLoRA;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
// Both DMA Load and MMA methods of this class must be run by a single thread that's picked by elect_one
template <int Stages, int SchedulerPipelineStageCount, int AccumulatorPipelineStageCount,
    class ClusterShape, // Static cluster shape or dynamic (int, int, _1)
    class TileShape_,   // (MmaAtomShapeM, MmaAtomShapeN, TileK)
    class ElementPairA_, class StridePairA_, class ElementPairB_, class StridePairB_, class TiledMma_,
    class GmemTiledCopyPairA_, class SmemLayoutAtomPairA_, class SmemCopyAtomA_, class TransformA_,
    class GmemTiledCopyPairB_, class SmemLayoutAtomPairB_, class SmemCopyAtomB_, class TransformB_>
struct CollectiveMmaLoRA<MainloopSm100TmaUmmaWarpSpecializedBlockScaled<Stages, SchedulerPipelineStageCount,
                             AccumulatorPipelineStageCount, ClusterShape>,
    TileShape_, ElementPairA_, StridePairA_, ElementPairB_, StridePairB_, TiledMma_, GmemTiledCopyPairA_,
    SmemLayoutAtomPairA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyPairB_, SmemLayoutAtomPairB_, SmemCopyAtomB_,
    TransformB_>
{
    //
    // Type Aliases
    //
    using TiledMma = TiledMma_;
    using AtomThrShapeMNK = Shape<decltype(shape<0>(typename TiledMma::ThrLayoutVMNK{})), _1, _1>;

    using DispatchPolicy = MainloopSm100TmaUmmaWarpSpecializedBlockScaled<Stages, SchedulerPipelineStageCount,
        AccumulatorPipelineStageCount, ClusterShape>;
    using TileShape = TileShape_;
    using TiledMMA_SF = TiledMMA<MMA_Atom<typename TiledMma::MMA_ScaleFactor>, Layout<Shape<_1, _1, _1>>,
        Tile<Underscore, Underscore, Underscore>>;

    static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;
    static constexpr int SFVecSize = TiledMma::SFVecSize;
    static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;

    CUTE_STATIC_ASSERT_V(evenly_divides(TileShape{}, tile_shape(TiledMma{})),
        "Static cluster shape used: TileShape should be evenly divided by TiledMma");

    using CtaShape_MNK = decltype(shape_div(TileShape{}, AtomThrShapeMNK{}));

    // === LoRA-up: a bf16 tcgen05 MMA of D@L1ᵀ (K=rank) accumulated into the
    // SAME f32 TMEM accumulator after the NVFP4 K-loop. Build the bf16 TiledMMA via CUTLASS's
    // own SM100 helper (matches the residual cta_group; SS = both operands SMEM-sourced, as
    // the residual mainloop requires). The MMA tile is the full CTA-group tile, while the
    // SMEM layouts below are explicitly reduced to each CTA's partition. ===
    static constexpr int LoRaK = 32; // SVDQuant rank r=32 = 2 bf16 16-wide K-atoms
    using LoRaMmaTileShape
        = decltype(make_shape(cute::size<0>(TileShape{}), cute::size<1>(TileShape{}), cute::Int<LoRaK>{}));
    static constexpr auto make_lora_mma()
    {
        static_assert(cute::size(AtomThrShapeMNK{}) == 1 || cute::size(AtomThrShapeMNK{}) == 2,
            "LoRaMma only supports one- or two-CTA MMA groups");
        if constexpr (cute::size(AtomThrShapeMNK{}) == 2)
        {
            return cutlass::gemm::collective::detail::sm100_make_2sm_trivial_tiled_mma<cutlass::bfloat16_t,
                cutlass::bfloat16_t, float, LoRaMmaTileShape, ClusterShape, cute::UMMA::Major::K,
                cute::UMMA::Major::K>();
        }
        else
        {
            return cutlass::gemm::collective::detail::sm100_make_1sm_trivial_tiled_mma<cutlass::bfloat16_t,
                cutlass::bfloat16_t, float, LoRaMmaTileShape, ClusterShape, cute::UMMA::Major::K,
                cute::UMMA::Major::K>();
        }
    }
    using LoRaMma = decltype(make_lora_mma());
    static_assert(cute::size(typename LoRaMma::ThrLayoutVMNK{}) >= 1, "LoRaMma constructed");
    static_assert(cute::size(typename LoRaMma::AtomThrID{}) == cute::size(typename TiledMma::AtomThrID{}),
        "LoRA and residual MMA CTA groups must match");
    static_assert(shape<1>(CtaShape_MNK{}) == 192 or shape<1>(CtaShape_MNK{}) == 64 or shape<1>(CtaShape_MNK{}) == 128
            or shape<1>(CtaShape_MNK{}) == 256,
        "Cta N should be one of 64/128/192/256");

    using ClusterTileShape = decltype(make_shape(get<0>(TileShape{}) * get<0>(ClusterShape{}),
        get<1>(TileShape{}) * get<1>(ClusterShape{}), get<2>(TileShape{}) * get<2>(ClusterShape{})));
    using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
    using Blk_MN = typename Sm1xxBlkScaledConfig::Blk_MN;
    static constexpr int IsCtaN192 = shape<1>(CtaShape_MNK{}) == 192;
    static constexpr int IsCtaN64 = shape<1>(CtaShape_MNK{}) == 64;
    static int constexpr CTA_N_SF = cutlass::ceil_div(size<1>(CtaShape_MNK{}), Blk_MN{}) * Blk_MN{};
    // Tile shape used for partitioning Scale Factor B.
    // The M-dim does not affect the SFB, so just set it as the original TileShape;
    using TileShape_SF = decltype(make_shape(
        get<0>(CtaShape_MNK{}), Int<CTA_N_SF>{} * shape<2>(typename TiledMma::ThrLayoutVMNK()), get<2>(TileShape{})));

    // Define A and B block shapes for reduced size TMA_LOADs
    using MmaShapeA_MK
        = decltype(partition_shape_A(TiledMma{}, make_shape(size<0>(TileShape{}), size<2>(TileShape{}))));
    using MmaShapeB_NK
        = decltype(partition_shape_B(TiledMma{}, make_shape(size<1>(TileShape{}), size<2>(TileShape{}))));

    using ElementPairA = ElementPairA_;
    using ElementPairB = ElementPairB_;
    using ElementAMma = typename TiledMma::ValTypeA;
    using ElementBMma = typename TiledMma::ValTypeB;
    using StridePairA = StridePairA_;
    using StridePairB = StridePairB_;
    using SmemLayoutAtomPairA = SmemLayoutAtomPairA_;
    using SmemLayoutAtomPairB = SmemLayoutAtomPairB_;
    static_assert(cute::is_same_v<remove_cvref_t<decltype(get<1>(ElementPairA{}))>,
                      remove_cvref_t<decltype(get<1>(ElementPairB{}))>>,
        "SFA and SFB data types should be the same");

    // A and B matrices
    using ElementA = remove_cvref_t<decltype(get<0>(ElementPairA{}))>;
    using StrideA = remove_cvref_t<decltype(get<0>(StridePairA{}))>;

    using ElementB = remove_cvref_t<decltype(get<0>(ElementPairB{}))>;
    using StrideB = remove_cvref_t<decltype(get<0>(StridePairB{}))>;

    static constexpr bool IsRuntimeDataTypeA = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementA>();
    static constexpr bool IsRuntimeDataTypeB = cutlass::gemm::collective::detail::is_sm10x_runtime_f8f6f4<ElementB>();

    static_assert((IsRuntimeDataTypeA && IsRuntimeDataTypeB) || (!IsRuntimeDataTypeA && !IsRuntimeDataTypeB),
        "ElementA and ElementB should be both runtime or both static.");

    static constexpr bool IsRuntimeDataType = IsRuntimeDataTypeA && IsRuntimeDataTypeB;

    // SFA and SFB
    using ElementSF = remove_cvref_t<decltype(get<1>(ElementPairA{}))>;
    using LayoutSFA = remove_cvref_t<decltype(get<1>(StridePairA{}))>;
    using LayoutSFB = remove_cvref_t<decltype(get<1>(StridePairB{}))>;

    using ElementAccumulator = typename TiledMma::ValTypeC;
    using GmemTiledCopyPairA = GmemTiledCopyPairA_;
    using GmemTiledCopyPairB = GmemTiledCopyPairB_;
    using GmemTiledCopyA = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairA{}))>;
    using GmemTiledCopySFA = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairA{}))>;
    using GmemTiledCopyB = remove_cvref_t<decltype(get<0>(GmemTiledCopyPairB{}))>;
    using GmemTiledCopySFB = remove_cvref_t<decltype(get<1>(GmemTiledCopyPairB{}))>;

    using SmemLayoutAtomA = remove_cvref_t<decltype(get<0>(SmemLayoutAtomPairA{}))>;
    using SmemLayoutAtomSFA = remove_cvref_t<decltype(get<1>(SmemLayoutAtomPairA{}))>;
    using SmemLayoutAtomB = remove_cvref_t<decltype(get<0>(SmemLayoutAtomPairB{}))>;
    using SmemLayoutAtomSFB = remove_cvref_t<decltype(get<1>(SmemLayoutAtomPairB{}))>;

    using SmemCopyAtomA = SmemCopyAtomA_;
    using SmemCopyAtomB = SmemCopyAtomB_;
    using TransformA = TransformA_;
    using TransformB = TransformB_;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<DispatchPolicy::Stages, ClusterShape, AtomThrShapeMNK>;
    using MainloopPipelineState = typename MainloopPipeline::PipelineState;

    static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert(
        (size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtomA must evenly divide the tile shape.");
    static_assert(
        (size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtomA must evenly divide the tile shape.");
    static_assert(
        cute::is_void_v<SmemCopyAtomA>, "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

    static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert(
        (size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtomB must evenly divide the tile shape.");
    static_assert(
        (size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtomB must evenly divide the tile shape.");
    static_assert(
        cute::is_void_v<SmemCopyAtomB>, "SM100 UMMA cannot have a non-void copy atom for smem sourced instructions.");

    // Tile along K mode first before tiling over MN. PIPE mode last as usual.
    // This maximizes TMA boxes due to better smem-K vectorization, reducing total issued TMAs.
    // (MMA_TILE_M,MMA_TILE_K),MMA_M,MMA_K,PIPE)
    using SmemLayoutA
        = decltype(UMMA::tile_to_mma_shape(SmemLayoutAtomA{}, append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
            cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_2, _1, _3>, Step<_1, _2, _3>>{}));
    // (MMA_TILE_N,MMA_TILE_K),MMA_N,MMA_K,PIPE)
    using SmemLayoutB
        = decltype(UMMA::tile_to_mma_shape(SmemLayoutAtomB{}, append(MmaShapeB_NK{}, Int<DispatchPolicy::Stages>{}),
            cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideB>(), Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

    // --- LoRA operand smem layouts: bf16 D[M,LoRaK] (A-side) + L1[N,LoRaK] (B-side).
    // D/L1 are loaded once per output tile, not per k_tile. Build the SMEM selector from the
    // partitioned MMA shape so a 2SM 256x256 MMA stores only its 128x256 per-CTA partition. ---
    using MmaShapeD_MK = decltype(partition_shape_A(
        LoRaMma{}, make_shape(cute::size<0>(LoRaMmaTileShape{}), cute::size<2>(LoRaMmaTileShape{}))));
    using MmaShapeL1_NK = decltype(partition_shape_B(
        LoRaMma{}, make_shape(cute::size<1>(LoRaMmaTileShape{}), cute::size<2>(LoRaMmaTileShape{}))));
    using BlockTileD_M = decltype(cute::size<0, 0>(MmaShapeD_MK{}) * cute::size<1>(MmaShapeD_MK{}));
    using BlockTileD_K = decltype(cute::size<0, 1>(MmaShapeD_MK{}) * cute::size<2>(MmaShapeD_MK{}));
    using BlockTileL1_N = decltype(cute::size<0, 0>(MmaShapeL1_NK{}) * cute::size<1>(MmaShapeL1_NK{}));
    using BlockTileL1_K = decltype(cute::size<0, 1>(MmaShapeL1_NK{}) * cute::size<2>(MmaShapeL1_NK{}));
    using SmemLayoutAtomD = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<cute::UMMA::Major::K,
        cutlass::bfloat16_t, BlockTileD_M, BlockTileD_K>());
    using SmemLayoutAtomL1 = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<cute::UMMA::Major::K,
        cutlass::bfloat16_t, BlockTileL1_N, BlockTileL1_K>());
    // D/L1 smem layouts are MULTI-STAGE (Stages, == smem_A/smem_B) so D/L1 RIDE the
    // residual stage buffers (smem_A[ws]/smem_B[ws]) instead of dedicated sD/sL1 -> no carveout,
    // residual regains its full stages. Byte-exact overlay: D-tile = M*LoRaK*bf16 == A-stage
    // (M*K*fp4/2) and L1-tile = N*LoRaK*bf16 == B-stage; stage stride matches smem_A/B.
    using SmemLayoutD = decltype(UMMA::tile_to_mma_shape(
        SmemLayoutAtomD{}, append(MmaShapeD_MK{}, cute::Int<DispatchPolicy::Stages>{})));
    using SmemLayoutL1 = decltype(UMMA::tile_to_mma_shape(
        SmemLayoutAtomL1{}, append(MmaShapeL1_NK{}, cute::Int<DispatchPolicy::Stages>{})));
    static_assert(cute::cosize(SmemLayoutD{}) > 0 && cute::cosize(SmemLayoutL1{}) > 0, "LoRA smem layouts constructed");
    static_assert(cute::cosize(take<0, 3>(SmemLayoutD{})) * cute::sizeof_bits_v<cutlass::bfloat16_t>
            == cute::cosize(take<0, 3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>,
        "Each CTA's LoRA D tile must exactly overlay one residual A stage");
    static_assert(cute::cosize(take<0, 3>(SmemLayoutL1{})) * cute::sizeof_bits_v<cutlass::bfloat16_t>
            == cute::cosize(take<0, 3>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>,
        "Each CTA's LoRA L1 tile must exactly overlay one residual B stage");
    using StrideD = cute::Stride<int64_t, cute::_1, int64_t>;  // D  [M, LoRaK, L] (K-contig)
    using StrideL1 = cute::Stride<int64_t, cute::_1, int64_t>; // L1 [N, LoRaK, L] (K-contig)
    // CTA-group bytes for one stage -- the post-loop TMA loads one D/L1 tile per output tile.
    // (Unused in the post-loop, which relies on D/L1 arriving the armed A/B byte budget, but
    // kept correct in case of an expect_transaction path.)
    static constexpr uint32_t LoRaTmaBytes
        = cutlass::bits_to_bytes(cute::size(AtomThrShapeMNK{}) * cute::cosize(take<0, 3>(SmemLayoutD{}))
              * cute::sizeof_bits_v<cutlass::bfloat16_t>)
        + cutlass::bits_to_bytes(cute::size(AtomThrShapeMNK{}) * cute::cosize(take<0, 3>(SmemLayoutL1{}))
            * cute::sizeof_bits_v<cutlass::bfloat16_t>);

    // SmemLayoutAtomSFA and SmemLayoutAtomSFB are for whole CTA tiles. We add the number of pipeline stages here.
    // The number of pipeline stages is the same as the number of pipeline stages from AB Load <-> MainLoop
    using SmemLayoutSFA = decltype(make_layout(append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
        append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))));
    using SmemLayoutSFB = decltype(make_layout(append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
        append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))));

    static_assert(cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value
            && cute::is_base_of<cute::UMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
        "MMA atom must source both A and B operand from smem_desc for this mainloop.");
    static_assert((size(AtomThrShapeMNK{}) == 1
                      && (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>
                          || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) )
            || (size(AtomThrShapeMNK{}) == 2
                && (cute::is_same_v<GmemTiledCopyA, SM100_TMA_2SM_LOAD>
                    || cute::is_same_v<GmemTiledCopyA, SM100_TMA_2SM_LOAD_MULTICAST>) ),
        "GmemTiledCopy - invalid TMA copy atom specified.");
    static_assert((size(AtomThrShapeMNK{}) == 1
                      && (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>
                          || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) )
            || (size(AtomThrShapeMNK{}) == 2
                && (cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD>
                    || cute::is_same_v<GmemTiledCopyB, SM100_TMA_2SM_LOAD_MULTICAST>) ),
        "GmemTiledCopy -  invalid TMA copy atom specified.");

    static constexpr bool IsF8F6F4 = detail::is_sm100_mma_f8f6f4<TiledMma, ElementA, ElementB>();

    using TmaInternalElementA = cute::conditional_t<IsF8F6F4, ElementAMma, ElementA>;
    using TmaInternalElementB = cute::conditional_t<IsF8F6F4, ElementBMma, ElementB>;

    using SmemAllocTypeA = cute::conditional_t < IsF8F6F4&& cute::sizeof_bits_v<ElementAMma><8, uint8_t, ElementAMma>;
    using SmemAllocTypeB = cute::conditional_t < IsF8F6F4&& cute::sizeof_bits_v<ElementBMma><8, uint8_t, ElementBMma>;

    using BitTypeElementA = cute::uint_bit_t<cute::sizeof_bits_v<ElementA>>;
    using BitTypeElementB = cute::uint_bit_t<cute::sizeof_bits_v<ElementB>>;

    using ArrayElementA = cute::conditional_t<IsRuntimeDataTypeA, BitTypeElementA, ElementA>;
    using ArrayElementB = cute::conditional_t<IsRuntimeDataTypeB, BitTypeElementB, ElementB>;

    using RuntimeDataTypeA = typename detail::sm10x_block_scale_runtime_input_t<ElementAMma, IsRuntimeDataTypeA>::Type;
    using RuntimeDataTypeB = typename detail::sm10x_block_scale_runtime_input_t<ElementBMma, IsRuntimeDataTypeB>::Type;

    struct SharedStorage
    {
        struct TensorStorage : cute::aligned_struct<128, _0>
        {
            cute::ArrayEngine<SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
            cute::ArrayEngine<SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
            cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
            cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
            // NO dedicated LoRA smem -- D/L1 reinterpret smem_A/smem_B's stage buffers
            // (byte-exact overlay) post-K-loop. This frees the ~24KB + carveout -> +1 mainloop stage.
        } tensors;

        using PipelineStorage = typename MainloopPipeline::SharedStorage;
        PipelineStorage pipeline;
    };

    // Expose shared storage for tensors/pipelines separately to allow kernel layer to reorder them.
    using TensorStorage = typename SharedStorage::TensorStorage;
    using PipelineStorage = typename SharedStorage::PipelineStorage;

    // Only one thread issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly
    static constexpr uint32_t SFTransactionBytes
        = cutlass::bits_to_bytes(
              size(AtomThrShapeMNK{}) * cosize(take<0, 3>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>)
        + cutlass::bits_to_bytes(
            size(AtomThrShapeMNK{}) * cosize(take<0, 3>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF>);
    static constexpr uint32_t ABTmaTransactionBytes
        = cutlass::bits_to_bytes(
              size(AtomThrShapeMNK{}) * cosize(take<0, 3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>)
        + cutlass::bits_to_bytes(
            size(AtomThrShapeMNK{}) * cosize(take<0, 3>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>);
    static constexpr uint32_t TmaTransactionBytes = ABTmaTransactionBytes + SFTransactionBytes;

    template <class AccTensor, class SfaTensor, class SfbTensor>
    struct TmemStorage
    {
        AccTensor accumulators;
        SfaTensor tCtSFA;
        SfbTensor tCtSFB;
    };

    template <class KTileCount, class GTensorPartitionedA, class GTensorPartitionedB, class STensorA, class STensorB,
        class GTensorPartitionedSFA, class GTensorPartitionedSFB, class STensorSFA, class STensorSFB,
        class GTensorPartitionedD, class STensorD, class GTensorPartitionedL1, class STensorL1>
    struct LoadParams
    {
        // for scheduler
        KTileCount k_tiles;
        // for input tensor values
        GTensorPartitionedA tAgA_mkl;
        GTensorPartitionedB tBgB_nkl;
        STensorA tAsA;
        STensorB tBsB;
        // for scale factor tensor values
        GTensorPartitionedSFA tAgSFA_mkl;
        GTensorPartitionedSFB tBgSFB_nkl;
        STensorSFA tAsSFA;
        STensorSFB tBsSFB;
        // the TMA multicast masks
        uint16_t mcast_mask_a;
        uint16_t mcast_mask_b;
        uint16_t mcast_mask_sfa;
        uint16_t mcast_mask_sfb;
        // LoRA D/L1 partitioned gmem + smem + masks (D like A [M-indexed], L1 like B [N-indexed]).
        GTensorPartitionedD tDgD_mkl;
        STensorD tDsD;
        GTensorPartitionedL1 tL1gL1_nkl;
        STensorL1 tL1sL1;
        uint16_t mcast_mask_d;
        uint16_t mcast_mask_l1;

        CUTLASS_DEVICE
        LoadParams(KTileCount k_tiles_, GTensorPartitionedA tAgA_mkl_, GTensorPartitionedB tBgB_nkl_, STensorA tAsA_,
            STensorB tBsB_, GTensorPartitionedSFA tAgSFA_mkl_, GTensorPartitionedSFB tBgSFB_nkl_, STensorSFA tAsSFA_,
            STensorSFB tBsSFB_, uint16_t mcast_mask_a_, uint16_t mcast_mask_b_, uint16_t mcast_mask_sfa_,
            uint16_t mcast_mask_sfb_, GTensorPartitionedD tDgD_mkl_, STensorD tDsD_, GTensorPartitionedL1 tL1gL1_nkl_,
            STensorL1 tL1sL1_, uint16_t mcast_mask_d_, uint16_t mcast_mask_l1_)
            : k_tiles(k_tiles_)
            , tAgA_mkl(tAgA_mkl_)
            , tBgB_nkl(tBgB_nkl_)
            , tAsA(tAsA_)
            , tBsB(tBsB_)
            , tAgSFA_mkl(tAgSFA_mkl_)
            , tBgSFB_nkl(tBgSFB_nkl_)
            , tAsSFA(tAsSFA_)
            , tBsSFB(tBsSFB_)
            , mcast_mask_a(mcast_mask_a_)
            , mcast_mask_b(mcast_mask_b_)
            , mcast_mask_sfa(mcast_mask_sfa_)
            , mcast_mask_sfb(mcast_mask_sfb_)
            , tDgD_mkl(tDgD_mkl_)
            , tDsD(tDsD_)
            , tL1gL1_nkl(tL1gL1_nkl_)
            , tL1sL1(tL1sL1_)
            , mcast_mask_d(mcast_mask_d_)
            , mcast_mask_l1(mcast_mask_l1_)
        {
        }
    };

    template <class TiledMma, class FragmentA, class FragmentB, class FragmentSFA, class FragmentSFB,
        class SFATiledCopy, class SmemFrgSFA, class TmemFrgSFA, class SFBTiledCopy, class SmemFrgSFB, class TmemFrgSFB,
        class FragmentD, class FragmentL1>
    struct MmaParams
    {
        TiledMma tiled_mma;
        FragmentA tCrA;
        FragmentB tCrB;
        FragmentSFA tCtSFA;
        FragmentSFB tCtSFB;
        SFATiledCopy tiled_copy_s2t_SFA;
        SmemFrgSFA thr_tCsSFA_s2t;
        TmemFrgSFA thr_tCtSFA_s2t;
        SFBTiledCopy tiled_copy_s2t_SFB;
        SmemFrgSFB thr_tCsSFB_s2t;
        TmemFrgSFB thr_tCtSFB_s2t;
        FragmentD tCrD;   // LoRA D fragment (MMA,MMA_M,MMA_K)
        FragmentL1 tCrL1; // LoRA L1 fragment (MMA,MMA_N,MMA_K)

        CUTLASS_DEVICE
        MmaParams(TiledMma tiled_mma_, FragmentA tCrA_, FragmentB tCrB_, FragmentSFA tCtSFA_, FragmentSFB tCtSFB_,
            SFATiledCopy tiled_copy_s2t_SFA_, SmemFrgSFA thr_tCsSFA_s2t_, TmemFrgSFA thr_tCtSFA_s2t_,
            SFBTiledCopy tiled_copy_s2t_SFB_, SmemFrgSFB thr_tCsSFB_s2t_, TmemFrgSFB thr_tCtSFB_s2t_, FragmentD tCrD_,
            FragmentL1 tCrL1_)
            : tiled_mma(tiled_mma_)
            , tCrA(tCrA_)
            , tCrB(tCrB_)
            , tCtSFA(tCtSFA_)
            , tCtSFB(tCtSFB_)
            , tiled_copy_s2t_SFA(tiled_copy_s2t_SFA_)
            , thr_tCsSFA_s2t(thr_tCsSFA_s2t_)
            , thr_tCtSFA_s2t(thr_tCtSFA_s2t_)
            , tiled_copy_s2t_SFB(tiled_copy_s2t_SFB_)
            , thr_tCsSFB_s2t(thr_tCsSFB_s2t_)
            , thr_tCtSFB_s2t(thr_tCtSFB_s2t_)
            , tCrD(tCrD_)
            , tCrL1(tCrL1_)
        {
        }
    };

    // Host side kernel arguments
    struct Arguments
    {
        ArrayElementA const* ptr_A{nullptr};
        StrideA dA{};
        ArrayElementB const* ptr_B{nullptr};
        StrideB dB{};
        ElementSF const* ptr_SFA{nullptr};
        LayoutSFA layout_SFA{};
        ElementSF const* ptr_SFB{nullptr};
        LayoutSFB layout_SFB{};
        RuntimeDataTypeA runtime_data_type_a{};
        RuntimeDataTypeB runtime_data_type_b{};
        // LoRA-up: D [M, LoRaK] (1/alpha NOT here), L1 [N, LoRaK] (1/alpha folded in). bf16.
        cutlass::bfloat16_t const* ptr_D{nullptr};
        StrideD dD{};
        cutlass::bfloat16_t const* ptr_L1{nullptr};
        StrideL1 dL1{};
    };

    // Device side kernel params
    struct Params
    {
        using ClusterLayout_VMNK
            = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(
                                        make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                make_tile(typename TiledMma::AtomThrID{})));

        using ClusterLayoutSfb_VMNK
            = decltype(tiled_divide(make_layout(conditional_return<IsDynamicCluster>(
                                        make_shape(uint32_t(0), uint32_t(0), Int<1>{}), ClusterShape{})),
                make_tile(typename TiledMMA_SF::AtomThrID{})));

        using TMA_A = decltype(make_tma_atom_A_sm100<TmaInternalElementA>(GmemTiledCopyA{},
            make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
            SmemLayoutA{}(_, _, _, cute::Int<0>{}), TileShape{}, TiledMma{}, ClusterLayout_VMNK{}));

        using TMA_B = decltype(make_tma_atom_B_sm100<TmaInternalElementB>(GmemTiledCopyB{},
            make_tensor(recast_ptr<TmaInternalElementB>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
            SmemLayoutB{}(_, _, _, cute::Int<0>{}), TileShape{}, TiledMma{}, ClusterLayout_VMNK{}));

        using TMA_SFA = decltype(make_tma_atom_A_sm100<uint16_t>(GmemTiledCopySFA{},
            make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}), SmemLayoutSFA{}(_, _, _, cute::Int<0>{}),
            TileShape{}, TiledMma{}, ClusterLayout_VMNK{}));

        using TMA_SFB = decltype(make_tma_atom_B_sm100<uint16_t>(GmemTiledCopySFB{},
            make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFB{}), SmemLayoutSFB{}(_, _, _, cute::Int<0>{}),
            TileShape_SF{}, TiledMMA_SF{}, ClusterLayoutSfb_VMNK{}));

        // LoRA D/L1 TMA atoms (bf16; D mcast like A [M-indexed], L1 like B [N-indexed]).
        using TMA_D = decltype(make_tma_atom_A_sm100<cutlass::bfloat16_t>(GmemTiledCopyA{},
            make_tensor(
                static_cast<cutlass::bfloat16_t const*>(nullptr), repeat_like(StrideD{}, int32_t(0)), StrideD{}),
            SmemLayoutD{}(_, _, _, cute::Int<0>{}), LoRaMmaTileShape{}, LoRaMma{}, ClusterLayout_VMNK{}));
        using TMA_L1 = decltype(make_tma_atom_B_sm100<cutlass::bfloat16_t>(GmemTiledCopyB{},
            make_tensor(
                static_cast<cutlass::bfloat16_t const*>(nullptr), repeat_like(StrideL1{}, int32_t(0)), StrideL1{}),
            SmemLayoutL1{}(_, _, _, cute::Int<0>{}), LoRaMmaTileShape{}, LoRaMma{}, ClusterLayout_VMNK{}));

        TMA_A tma_load_a;
        TMA_B tma_load_b;
        TMA_SFA tma_load_sfa;
        TMA_SFB tma_load_sfb;
        TMA_A tma_load_a_fallback;
        TMA_B tma_load_b_fallback;
        TMA_SFA tma_load_sfa_fallback;
        TMA_SFB tma_load_sfb_fallback;
        TMA_D tma_load_d;
        TMA_L1 tma_load_l1;
        TMA_D tma_load_d_fallback;
        TMA_L1 tma_load_l1_fallback;
        LayoutSFA layout_SFA;
        LayoutSFB layout_SFB;
        dim3 cluster_shape_fallback;
        RuntimeDataTypeA runtime_data_type_a;
        RuntimeDataTypeB runtime_data_type_b;
    };

    CUTLASS_DEVICE
    CollectiveMmaLoRA(Params const& params, ClusterShape cluster_shape, uint32_t block_rank_in_cluster)
        : cluster_shape_(cluster_shape)
        , block_rank_in_cluster_(block_rank_in_cluster)
        , layout_SFA_(params.layout_SFA)
        , layout_SFB_(params.layout_SFB)
        , runtime_data_type_a_(params.runtime_data_type_a)
        , runtime_data_type_b_(params.runtime_data_type_b)
    {
        if constexpr (IsDynamicCluster)
        {
            bool const is_fallback_cluster = (cute::size<0>(cluster_shape_) == params.cluster_shape_fallback.x
                && cute::size<1>(cluster_shape_) == params.cluster_shape_fallback.y);
            observed_tma_load_a_ = is_fallback_cluster ? &params.tma_load_a_fallback : &params.tma_load_a;
            observed_tma_load_b_ = is_fallback_cluster ? &params.tma_load_b_fallback : &params.tma_load_b;
            observed_tma_load_sfa_ = is_fallback_cluster ? &params.tma_load_sfa_fallback : &params.tma_load_sfa;
            observed_tma_load_sfb_ = is_fallback_cluster ? &params.tma_load_sfb_fallback : &params.tma_load_sfb;
            observed_tma_load_d_ = is_fallback_cluster ? &params.tma_load_d_fallback : &params.tma_load_d;
            observed_tma_load_l1_ = is_fallback_cluster ? &params.tma_load_l1_fallback : &params.tma_load_l1;
        }
        else
        {
            observed_tma_load_a_ = &params.tma_load_a;
            observed_tma_load_b_ = &params.tma_load_b;
            observed_tma_load_sfa_ = &params.tma_load_sfa;
            observed_tma_load_sfb_ = &params.tma_load_sfb;
            observed_tma_load_d_ = &params.tma_load_d;
            observed_tma_load_l1_ = &params.tma_load_l1;
        }
    }

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
        [[maybe_unused]] void* workspace, cutlass::KernelHardwareInfo const& hw_info = cutlass::KernelHardwareInfo{})
    {

        // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
        auto problem_shape_MNKL = append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;

        auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
        auto ptr_B = recast_ptr<TmaInternalElementB>(args.ptr_B);

        Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M, K, L), args.dA));
        Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N, K, L), args.dB));
        auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape);

        // Cluster layout for TMA construction
        auto cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMma::AtomThrID{}));
        auto cluster_shape_fallback
            = cutlass::detail::select_cluster_shape(ClusterShape{}, hw_info.cluster_shape_fallback);
        auto cluster_layout_vmnk_fallback
            = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMma::AtomThrID{}));
        Tensor tensor_sfa = make_tensor(args.ptr_SFA, args.layout_SFA);
        Tensor tensor_sfb = make_tensor(args.ptr_SFB, args.layout_SFB);

        // Cluster layout for TMA construction of SFB
        auto cluster_layout_sfb_vmnk
            = tiled_divide(make_layout(cluster_shape), make_tile(typename TiledMMA_SF::AtomThrID{}));
        auto cluster_layout_sfb_vmnk_fallback
            = tiled_divide(make_layout(cluster_shape_fallback), make_tile(typename TiledMMA_SF::AtomThrID{}));

        typename Params::TMA_A tma_load_a = make_tma_atom_A_sm100<TmaInternalElementA>(GmemTiledCopyA{}, tensor_a,
            SmemLayoutA{}(_, _, _, cute::Int<0>{}), TileShape{}, TiledMma{}, cluster_layout_vmnk);

        typename Params::TMA_B tma_load_b = make_tma_atom_B_sm100<TmaInternalElementB>(GmemTiledCopyB{}, tensor_b,
            SmemLayoutB{}(_, _, _, cute::Int<0>{}), TileShape{}, TiledMma{}, cluster_layout_vmnk);

        typename Params::TMA_A tma_load_a_fallback = make_tma_atom_A_sm100<TmaInternalElementA>(GmemTiledCopyA{},
            tensor_a, SmemLayoutA{}(_, _, _, cute::Int<0>{}), TileShape{}, TiledMma{}, cluster_layout_vmnk_fallback);

        typename Params::TMA_B tma_load_b_fallback = make_tma_atom_B_sm100<TmaInternalElementB>(GmemTiledCopyB{},
            tensor_b, SmemLayoutB{}(_, _, _, cute::Int<0>{}), TileShape{}, TiledMma{}, cluster_layout_vmnk_fallback);

        typename Params::TMA_SFA tma_load_sfa = make_tma_atom_A_sm100<uint16_t>(GmemTiledCopySFA{}, tensor_sfa,
            SmemLayoutSFA{}(_, _, _, cute::Int<0>{}), TileShape{}, TiledMma{}, cluster_layout_vmnk);

        typename Params::TMA_SFB tma_load_sfb = make_tma_atom_B_sm100<uint16_t>(GmemTiledCopySFB{}, tensor_sfb,
            SmemLayoutSFB{}(_, _, _, cute::Int<0>{}), TileShape_SF{}, TiledMMA_SF{}, cluster_layout_sfb_vmnk);

        typename Params::TMA_SFA tma_load_sfa_fallback = make_tma_atom_A_sm100<uint16_t>(GmemTiledCopySFA{}, tensor_sfa,
            SmemLayoutSFA{}(_, _, _, cute::Int<0>{}), TileShape{}, TiledMma{}, cluster_layout_vmnk_fallback);

        typename Params::TMA_SFB tma_load_sfb_fallback = make_tma_atom_B_sm100<uint16_t>(GmemTiledCopySFB{}, tensor_sfb,
            SmemLayoutSFB{}(_, _, _, cute::Int<0>{}), TileShape_SF{}, TiledMMA_SF{}, cluster_layout_sfb_vmnk_fallback);

        // LoRA D/L1 TMA atoms (bf16; [M,LoRaK,L] / [N,LoRaK,L], K-contiguous).
        Tensor tensor_d = make_tensor(args.ptr_D, make_layout(make_shape(M, cute::Int<LoRaK>{}, L), args.dD));
        Tensor tensor_l1 = make_tensor(args.ptr_L1, make_layout(make_shape(N, cute::Int<LoRaK>{}, L), args.dL1));
        typename Params::TMA_D tma_load_d = make_tma_atom_A_sm100<cutlass::bfloat16_t>(GmemTiledCopyA{}, tensor_d,
            SmemLayoutD{}(_, _, _, cute::Int<0>{}), LoRaMmaTileShape{}, LoRaMma{}, cluster_layout_vmnk);
        typename Params::TMA_L1 tma_load_l1 = make_tma_atom_B_sm100<cutlass::bfloat16_t>(GmemTiledCopyB{}, tensor_l1,
            SmemLayoutL1{}(_, _, _, cute::Int<0>{}), LoRaMmaTileShape{}, LoRaMma{}, cluster_layout_vmnk);
        typename Params::TMA_D tma_load_d_fallback = make_tma_atom_A_sm100<cutlass::bfloat16_t>(GmemTiledCopyA{},
            tensor_d, SmemLayoutD{}(_, _, _, cute::Int<0>{}), LoRaMmaTileShape{}, LoRaMma{},
            cluster_layout_vmnk_fallback);
        typename Params::TMA_L1 tma_load_l1_fallback = make_tma_atom_B_sm100<cutlass::bfloat16_t>(GmemTiledCopyB{},
            tensor_l1, SmemLayoutL1{}(_, _, _, cute::Int<0>{}), LoRaMmaTileShape{}, LoRaMma{},
            cluster_layout_vmnk_fallback);

        return {tma_load_a, tma_load_b, tma_load_sfa, tma_load_sfb, tma_load_a_fallback, tma_load_b_fallback,
            tma_load_sfa_fallback, tma_load_sfb_fallback, tma_load_d, tma_load_l1, tma_load_d_fallback,
            tma_load_l1_fallback, args.layout_SFA, args.layout_SFB, hw_info.cluster_shape_fallback,
            args.runtime_data_type_a, args.runtime_data_type_b};
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const& problem_shape, [[maybe_unused]] Arguments const& args)
    {
        auto problem_shape_MNKL = append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;

        constexpr int tma_alignment_bits_A = cutlass::detail::get_input_alignment_bits<ElementA, IsF8F6F4>();
        constexpr int tma_alignment_bits_B = cutlass::detail::get_input_alignment_bits<ElementB, IsF8F6F4>();

        bool implementable = true;
        constexpr int min_tma_aligned_elements_A = tma_alignment_bits_A / cute::sizeof_bits<ElementA>::value;
        implementable = implementable
            && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M, K, L), StrideA{});
        constexpr int min_tma_aligned_elements_B = tma_alignment_bits_B / cute::sizeof_bits<ElementB>::value;
        implementable = implementable
            && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N, K, L), StrideB{});

        // Check for SFA SFB layout requirement
        auto const layout_sfa_ref = take<0, 2>(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(problem_shape_MNKL));
        auto const layout_sfb_ref = take<0, 2>(Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(problem_shape_MNKL));

        implementable = implementable && (layout_sfa_ref == take<0, 2>(args.layout_SFA));
        if (!implementable)
        {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: layout_SFA mismatch, layout_SFA needs to be K-major\n");
        }

        implementable = implementable && (layout_sfb_ref == take<0, 2>(args.layout_SFB));
        if (!implementable)
        {
            CUTLASS_TRACE_HOST("  CAN IMPLEMENT: layout_SFB mismatch, layout_SFB needs to be K-major\n");
        }

        if (!implementable)
        {
            CUTLASS_TRACE_HOST(
                "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
        }
        return implementable;
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE void prefetch_tma_descriptors()
    {
        cute::prefetch_tma_descriptor(observed_tma_load_a_->get_tma_descriptor());
        cute::prefetch_tma_descriptor(observed_tma_load_b_->get_tma_descriptor());
        cute::prefetch_tma_descriptor(observed_tma_load_sfa_->get_tma_descriptor());
        cute::prefetch_tma_descriptor(observed_tma_load_sfb_->get_tma_descriptor());
        cute::prefetch_tma_descriptor(observed_tma_load_d_->get_tma_descriptor());
        cute::prefetch_tma_descriptor(observed_tma_load_l1_->get_tma_descriptor());
    }

    /// Construct A Single Stage's Accumulator Shape
    CUTLASS_DEVICE static auto partition_accumulator_shape()
    {
        auto acc_shape
            = partition_shape_C(TiledMma{}, take<0, 2>(TileShape{})); // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

        return acc_shape;
    }

    template <class TmemStorage>
    CUTLASS_DEVICE static auto slice_accumulator(TmemStorage tmem_storage, int stage)
    {
        return cute::make_tuple(tmem_storage.accumulators(_, _, _, stage));
    }

    template <class EpilogueTile, bool IsOverlappingAccum = false>
    CUTLASS_DEVICE static auto init_tmem_tensors(EpilogueTile epi_tile)
    {
        TiledMma tiled_mma;
        auto acc_shape = partition_accumulator_shape();
        // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N,ACC_PIPE) where ACC_PIPE=2 so we can double buffer our accumulators for
        // mainloop and epilogue.
        Tensor accumulators
            = cutlass::detail::make_sm100_accumulator<AccumulatorPipelineStageCount, IsOverlappingAccum>(
                tiled_mma, acc_shape, EpilogueTile{});
        Tensor tCtSFA = make_tensor<typename TiledMma::FrgTypeSFA>(shape(SmemLayoutAtomSFA{}));
        Tensor tCtSFB = make_tensor<typename TiledMma::FrgTypeSFB>(shape(SmemLayoutAtomSFB{}));

        TmemStorage<decltype(accumulators), decltype(tCtSFA), decltype(tCtSFB)> tmem_storage;
        tmem_storage.accumulators = accumulators;
        tmem_storage.tCtSFA = tCtSFA;
        tmem_storage.tCtSFB = tCtSFB;

        return tmem_storage;
    }

    template <class TmemStorage>
    CUTLASS_DEVICE static void set_tmem_offsets(TmemStorage& tmem_storage, uint32_t tmem_base_addr)
    {
        tmem_storage.accumulators.data() = tmem_base_addr;
        tmem_storage.tCtSFA.data() = tmem_storage.accumulators.data().get()
            + cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.accumulators);
        tmem_storage.tCtSFB.data()
            = tmem_storage.tCtSFA.data().get() + cutlass::detail::find_tmem_tensor_col_offset(tmem_storage.tCtSFA);
    }

    /// Set up the data needed by this collective for load.
    /// Return tuple element contain
    /// gA_mkl - The tiled tma tensor for input A
    /// gB_nkl - The tiled tma tensor for input B
    /// tAgA_mkl - partitioned gmem tensor for A
    /// tBgB_nkl - partitioned gmem tensor for B
    /// tAsA - partitioned smem tensor for A
    /// tBsB - partitioned smem tensor for B
    /// tAgSFA_mkl - partitioned gmem tensor for SFA
    /// tBgSFB_nkl - partitioned gmem tensor for SFB
    /// tAsSFA - partitioned tmem tensor for SFA
    /// tAsSFB - partitioned tmem tensor for SFB
    /// mcast_mask_a - tma multicast mask for A
    /// mcast_mask_b - tma multicast mask for B
    /// mcast_mask_sfa - tma multicast mask for SFA
    /// mcast_mask_sfb - tma multicast mask for SFB
    template <class ProblemShape_MNKL>
    CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL, TensorStorage& shared_tensors) const
    {
        using X = Underscore;

        // Separate out problem shape for convenience
        auto [M, N, K, L] = problem_shape_MNKL;

        // Represent the full tensors -- get these from TMA
        Tensor mA_mkl = observed_tma_load_a_->get_tma_tensor(make_shape(M, K, L));
        Tensor mB_nkl = observed_tma_load_b_->get_tma_tensor(make_shape(N, K, L));

        // Tile the tensors and defer the slice
        Tensor gA_mkl
            = local_tile(mA_mkl, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{}); // (BLK_M, BLK_K, m, k, l)
        Tensor gB_nkl
            = local_tile(mB_nkl, TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (BLK_N, BLK_K, n, k, l)

        // Represent the full tensor of Scale factors
        Tensor mSFA_mkl = observed_tma_load_sfa_->get_tma_tensor(shape(layout_SFA_));
        auto mSFB_nkl = [=]()
        {
            if constexpr (IsCtaN192)
            {
                Tensor mSFB_tmp = observed_tma_load_sfb_->get_tma_tensor(shape(layout_SFB_));
                auto x = stride<0, 1>(mSFB_tmp);
                auto y = ceil_div(shape<0, 1>(mSFB_tmp), 4);
                auto new_shape = make_shape(make_shape(shape<0, 0>(mSFB_tmp), make_shape(make_shape(_2{}, _2{}), y)),
                    shape<1>(mSFB_tmp), shape<2>(mSFB_tmp));
                auto new_stride
                    = make_stride(make_stride(stride<0, 0>(mSFB_tmp), make_stride(make_stride(x, x), x * 3)),
                        stride<1>(mSFB_tmp), stride<2>(mSFB_tmp));
                return make_tensor(mSFB_tmp.data(), make_layout(new_shape, new_stride));
            }
            else if constexpr (IsCtaN64)
            {
                Tensor mSFB_tmp = observed_tma_load_sfb_->get_tma_tensor(shape(layout_SFB_));
                auto new_shape = make_shape(make_shape(shape<0, 0>(mSFB_tmp), make_shape(_2{}, shape<0, 1>(mSFB_tmp))),
                    shape<1>(mSFB_tmp), shape<2>(mSFB_tmp));
                auto new_stride
                    = make_stride(make_stride(stride<0, 0>(mSFB_tmp), make_stride(_0{}, stride<0, 1>(mSFB_tmp))),
                        stride<1>(mSFB_tmp), stride<2>(mSFB_tmp));
                return make_tensor(mSFB_tmp.data(), make_layout(new_shape, new_stride));
            }
            else
            {
                return observed_tma_load_sfb_->get_tma_tensor(shape(layout_SFB_));
            }
        }();

        Tensor gSFA_mkl
            = local_tile(mSFA_mkl, TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});    // (TILE_M,TILE_K,m,k,l)
        Tensor gSFB_nkl
            = local_tile(mSFB_nkl, TileShape_SF{}, make_coord(_, _, _), Step<X, _1, _1>{}); // (TILE_N,TILE_K,n,k,l)

        // Partition for this CTA
        ThrMMA cta_mma = TiledMma{}.get_slice(blockIdx.x % size(typename TiledMma::AtomThrID{}));

        Tensor tCgA_mkl = cta_mma.partition_A(gA_mkl); // (MMA, MMA_M, MMA_K, m, k, l)
        Tensor tCgB_nkl = cta_mma.partition_B(gB_nkl); // (MMA, MMA_N, MMA_K, n, k, l)

        Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{}); // (MMA,MMA_M,MMA_K,PIPE)
        Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{}); // (MMA,MMA_N,MMA_K,PIPE)

        ThrMMA cta_mma_sfb = TiledMMA_SF{}.get_slice(blockIdx.x % size(typename TiledMMA_SF::AtomThrID{}));
        Tensor tCgSFA_mkl = cta_mma.partition_A(gSFA_mkl);     // (MMA, MMA_M, MMA_K, m, k, l)
        Tensor tCgSFB_nkl = cta_mma_sfb.partition_B(gSFB_nkl); // (MMA, MMA_N, MMA_K, n, k, l)

        Tensor sSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
        Tensor sSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

        // Define the CTA-in-cluster Layout and Coord
        Layout cta_layout_mnk = make_layout(cluster_shape_);
        Layout cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMma::AtomThrID{}));
        auto cta_coord_vmnk = cta_layout_vmnk.get_flat_coord(block_rank_in_cluster_);

        Layout cta_layout_sfb_vmnk = tiled_divide(cta_layout_mnk, make_tile(typename TiledMMA_SF::AtomThrID{}));
        auto cta_coord_sfb_vmnk = cta_layout_sfb_vmnk.get_flat_coord(block_rank_in_cluster_);

        // Project the cta_layout for tma_a along the n-modes
        auto [tAgA_mkl, tAsA] = tma_partition(*observed_tma_load_a_, get<2>(cta_coord_vmnk),
            make_layout(size<2>(cta_layout_vmnk)), group_modes<0, 3>(sA), group_modes<0, 3>(tCgA_mkl));

        // Project the cta_layout for tma_b along the m-modes
        auto [tBgB_nkl, tBsB] = tma_partition(*observed_tma_load_b_, get<1>(cta_coord_vmnk),
            make_layout(size<1>(cta_layout_vmnk)), group_modes<0, 3>(sB), group_modes<0, 3>(tCgB_nkl));

        // Project the cta_layout for tma_a along the n-modes
        auto [tAgSFA_mkl, tAsSFA] = tma_partition(*observed_tma_load_sfa_, get<2>(cta_coord_vmnk),
            make_layout(size<2>(cta_layout_vmnk)), group_modes<0, 3>(sSFA), group_modes<0, 3>(tCgSFA_mkl));

        // Project the cta_layout for tma_b along the m-modes
        auto [tBgSFB_nkl, tBsSFB] = tma_partition(*observed_tma_load_sfb_, get<1>(cta_coord_sfb_vmnk),
            make_layout(size<1>(cta_layout_sfb_vmnk)), group_modes<0, 3>(sSFB), group_modes<0, 3>(tCgSFB_nkl));

        // TMA Multicast Masks
        uint16_t mcast_mask_a = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
        uint16_t mcast_mask_b = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);
        uint16_t mcast_mask_sfa = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
        uint16_t mcast_mask_sfb = create_tma_multicast_mask<1>(cta_layout_sfb_vmnk, cta_coord_sfb_vmnk);

        // LoRA D[M,LoRaK]/L1[N,LoRaK]: partition like A/B (D along M, L1 along N), single k-tile.
        Tensor mD_mkl = observed_tma_load_d_->get_tma_tensor(make_shape(M, cute::Int<LoRaK>{}, L));
        Tensor mL1_nkl = observed_tma_load_l1_->get_tma_tensor(make_shape(N, cute::Int<LoRaK>{}, L));
        Tensor gD_mkl = local_tile(mD_mkl, LoRaMmaTileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
        Tensor gL1_nkl = local_tile(mL1_nkl, LoRaMmaTileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
        ThrMMA cta_mma_lora = LoRaMma{}.get_slice(blockIdx.x % size(typename LoRaMma::AtomThrID{}));
        Tensor tCgD_mkl = cta_mma_lora.partition_A(gD_mkl);
        Tensor tCgL1_nkl = cta_mma_lora.partition_B(gL1_nkl);
        // D/L1 SMEM = the residual stage buffers reinterpreted as bf16 (byte-exact, multi-stage).
        Tensor sD
            = make_tensor(recast_ptr<cutlass::bfloat16_t>(make_smem_ptr(shared_tensors.smem_A.begin())), SmemLayoutD{});
        Tensor sL1 = make_tensor(
            recast_ptr<cutlass::bfloat16_t>(make_smem_ptr(shared_tensors.smem_B.begin())), SmemLayoutL1{});
        auto [tDgD_mkl, tDsD] = tma_partition(*observed_tma_load_d_, get<2>(cta_coord_vmnk),
            make_layout(size<2>(cta_layout_vmnk)), group_modes<0, 3>(sD), group_modes<0, 3>(tCgD_mkl));
        auto [tL1gL1_nkl, tL1sL1] = tma_partition(*observed_tma_load_l1_, get<1>(cta_coord_vmnk),
            make_layout(size<1>(cta_layout_vmnk)), group_modes<0, 3>(sL1), group_modes<0, 3>(tCgL1_nkl));
        uint16_t mcast_mask_d = create_tma_multicast_mask<2>(cta_layout_vmnk, cta_coord_vmnk);
        uint16_t mcast_mask_l1 = create_tma_multicast_mask<1>(cta_layout_vmnk, cta_coord_vmnk);

        return LoadParams{size<3>(gA_mkl),                                    // for scheduler
            tAgA_mkl, tBgB_nkl, tAsA, tBsB,                                   // for input tensor values
            tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB,                           // for input scale factor tensor values
            mcast_mask_a, mcast_mask_b, mcast_mask_sfa, mcast_mask_sfb,       // multicast masks
            tDgD_mkl, tDsD, tL1gL1_nkl, tL1sL1, mcast_mask_d, mcast_mask_l1}; // LoRA D/L1
    }

    /// Set up the data needed by this collective for mma compute.
    template <class TmemStorage>
    CUTLASS_DEVICE auto mma_init(TmemStorage tmem_storage, TensorStorage& shared_tensors) const
    {

        // Allocate "fragments/descriptors" for A and B matrices
        Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
        Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

        // Allocate "fragments/descriptors" for A and B matrices
        Tensor tCrA = TiledMma::make_fragment_A(sA);                        // (MMA,MMA_M,MMA_K,PIPE)
        Tensor tCrB = TiledMma::make_fragment_B(sB);                        // (MMA,MMA_N,MMA_K,PIPE)

        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sA)); // PIPE
        CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<3>(sB)); // PIPE

        //
        // Scale Factor
        //
        Tensor tCtSFA = tmem_storage.tCtSFA;
        Tensor tCtSFB = tmem_storage.tCtSFB;
        // Setup smem descriptors for UTCCP
        Tensor tCsSFA = make_tensor(make_smem_ptr(shared_tensors.smem_SFA.begin()), SmemLayoutSFA{});
        Tensor tCsSFB = make_tensor(make_smem_ptr(shared_tensors.smem_SFB.begin()), SmemLayoutSFB{});

        // Make SMEM and TMEM tensors compact removing the zero strides to eliminate unnecessary copy instructions.
        auto tCsSFA_compact = make_tensor(tCsSFA.data(), filter_zeros(tCsSFA.layout()));
        auto tCtSFA_compact = make_tensor(tCtSFA.data(), filter_zeros(tCtSFA.layout()));
        auto tCsSFB_compact = make_tensor(tCsSFB.data(), filter_zeros(tCsSFB.layout()));
        auto tCtSFB_compact = make_tensor(tCtSFB.data(), filter_zeros(tCtSFB.layout()));

        // Create the SMEM to TMEM copy operations based on the MMA atom used (1CTA vs 2CTA)
        using AtomThrID = typename TiledMma::AtomThrID;
        using UtccpOp = cute::conditional_t<(decltype(cute::size(AtomThrID{}) == Int<2>{})::value),
            SM100_UTCCP_4x32dp128bit_2cta, SM100_UTCCP_4x32dp128bit_1cta>;
        auto tiled_copy_s2t_SFA = make_utccp_copy(UtccpOp{}, tCtSFA_compact);
        auto tiled_copy_s2t_SFB = make_utccp_copy(UtccpOp{}, tCtSFB_compact);

        auto thr_copy_s2t_SFA = tiled_copy_s2t_SFA.get_slice(0);
        auto thr_tCsSFA_compact_s2t_ = thr_copy_s2t_SFA.partition_S(tCsSFA_compact);
        // SMEM to TMEM copy operation requires source SMEM operand to be an SMEM descriptor
        auto thr_tCsSFA_compact_s2t = get_utccp_smem_desc_tensor<UtccpOp>(thr_tCsSFA_compact_s2t_);
        auto thr_tCtSFA_compact_s2t = thr_copy_s2t_SFA.partition_D(tCtSFA_compact);

        auto thr_copy_s2t_SFB = tiled_copy_s2t_SFB.get_slice(0);
        auto thr_tCsSFB_compact_s2t_ = thr_copy_s2t_SFB.partition_S(tCsSFB_compact);
        // SMEM to TMEM copy operation requires source SMEM operand to be an SMEM descriptor
        auto thr_tCsSFB_compact_s2t = get_utccp_smem_desc_tensor<UtccpOp>(thr_tCsSFB_compact_s2t_);
        auto thr_tCtSFB_compact_s2t = thr_copy_s2t_SFB.partition_D(tCtSFB_compact);

        TiledMma tiled_mma;

        if constexpr (IsRuntimeDataType)
        {
            // Update instruction descriptor according to runtime argument.
            // Applying bitmask (0b111) to help compiler deduce that the conversion and assignment are safe.
            tiled_mma.idesc_.a_format_ = uint8_t(runtime_data_type_a_) & 0b111;
            tiled_mma.idesc_.b_format_ = uint8_t(runtime_data_type_b_) & 0b111;
        }

        // LoRA fragments for the post-K-loop D@L1ᵀ MMA into the SAME f32 accumulator.
        // D/L1 SMEM = reinterpreted residual stage buffers (multi-stage) -> tCrD/tCrL1 are
        // stage-moded (MMA,MMA_M/N,MMA_K,Stages); apply_lora indexes the consumed stage read_stage.
        Tensor sD
            = make_tensor(recast_ptr<cutlass::bfloat16_t>(make_smem_ptr(shared_tensors.smem_A.begin())), SmemLayoutD{});
        Tensor sL1 = make_tensor(
            recast_ptr<cutlass::bfloat16_t>(make_smem_ptr(shared_tensors.smem_B.begin())), SmemLayoutL1{});
        Tensor tCrD = LoRaMma::make_fragment_A(sD);   // (MMA,MMA_M,MMA_K,Stages)
        Tensor tCrL1 = LoRaMma::make_fragment_B(sL1); // (MMA,MMA_N,MMA_K,Stages)

        return MmaParams{tiled_mma, tCrA, tCrB, tCtSFA, tCtSFB, tiled_copy_s2t_SFA, thr_tCsSFA_compact_s2t,
            thr_tCtSFA_compact_s2t, tiled_copy_s2t_SFB, thr_tCsSFB_compact_s2t, thr_tCtSFB_compact_s2t, tCrD, tCrL1};
    }

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Producer Perspective
    template <class LoadParams, class TileCoordMNKL, class KTileIterator>
    CUTLASS_DEVICE auto load(MainloopPipeline mainloop_pipeline, MainloopPipelineState mainloop_pipe_producer_state,
        LoadParams const& load_inputs, TileCoordMNKL const& cta_coord_mnkl, KTileIterator k_tile_iter, int k_tile_count)
    {

        auto [unused_k_tiles, tAgA_mkl, tBgB_nkl, tAsA, tBsB, tAgSFA_mkl, tBgSFB_nkl, tAsSFA, tBsSFB, mcast_mask_a,
            mcast_mask_b, mcast_mask_sfa, mcast_mask_sfb, tDgD_mkl, tDsD, tL1gL1_nkl, tL1sL1, mcast_mask_d,
            mcast_mask_l1]
            = load_inputs;

        // slice out the work coord from partitioned tensors
        Tensor tAgA
            = tAgA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
        Tensor tBgB = tBgB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
        Tensor tAgSFA
            = tAgSFA_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
        Tensor tBgSFB = tBgSFB_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));
        Tensor tDgD
            = tDgD_mkl(_, get<0>(cta_coord_mnkl) / size(typename TiledMma::AtomThrID{}), _, get<3>(cta_coord_mnkl));
        Tensor tL1gL1 = tL1gL1_nkl(_, get<1>(cta_coord_mnkl), _, get<3>(cta_coord_mnkl));

        // The kernel calls load() TWICE per work-tile: prologue (starts at k-tile 0) then mainloop
        // (starts at k_tile_prologue>0). For a simple-GEMM scalar k-tile iterator, *k_tile_iter is
        // that start index. We emit the post-K-loop D/L1 step ONLY in the mainloop call (start!=0)
        // so it lands at the END of the global k-tile sequence (exactly once), matching mma()'s
        // single post-loop consumer step. (Case k_tiles<=Stages: prologue loads all, mainloop loads
        // 0 but still starts at k_tiles>0 -> it emits, still after all tiles. Correct either way.)
        int lora_start_k = *k_tile_iter;

        auto barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

        // Issue the Mainloop loads
        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > 0)
        {
            // LOCK mainloop_pipe_producer_state for _writing_
            mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);
            // Note: We don't synchronize the sf_pipeline for "Buffer_Empty". We use mainloop pipeline
            // to do the synchronization at once.

            using BarrierType = typename MainloopPipeline::ProducerBarrierType;
            BarrierType* tma_barrier = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);

            int write_stage = mainloop_pipe_producer_state.index();
            ++mainloop_pipe_producer_state;
            barrier_token = mainloop_pipeline.producer_try_acquire(mainloop_pipe_producer_state);

            if (cute::elect_one_sync())
            {
                copy(observed_tma_load_a_->with(*tma_barrier, mcast_mask_a), tAgA(_, *k_tile_iter),
                    tAsA(_, write_stage));
                copy(observed_tma_load_b_->with(*tma_barrier, mcast_mask_b), tBgB(_, *k_tile_iter),
                    tBsB(_, write_stage));
                copy(observed_tma_load_sfa_->with(*tma_barrier, mcast_mask_sfa), tAgSFA(_, *k_tile_iter),
                    tAsSFA(_, write_stage));
                copy(observed_tma_load_sfb_->with(*tma_barrier, mcast_mask_sfb), tBgSFB(_, *k_tile_iter),
                    tBsSFB(_, write_stage));
            }

            --k_tile_count;
            ++k_tile_iter;
        }

        // === load D/L1 in ONE extra producer step AFTER the K-loop, INTO the
        // freed residual stage buffers smem_A[ws]/smem_B[ws] (no dedicated sD/sL1, no carveout), ONLY
        // in the mainloop call (lora_start_k != 0) so it lands at the END of the k-tile sequence (else
        // the prologue's copy would inject mid-sequence -> consumer misreads it as a tile -> hang). The
        // producer_acquire arms the barrier for the A+B+SF byte budget; D arrives EXACTLY the A-stage
        // bytes (D-tile == A-stage) into smem_A[ws], L1 the B-stage bytes into smem_B[ws], and SFA/SFB
        // are DUMMY-reloaded to arrive the SF bytes -> D+L1+SF == armed A+B+SF, so NO expect_transaction
        // and NO A/B re-load needed. (D/L1 in pipeline-managed stage buffers also removes the earlier
        // dedicated-single-buffer cross-tile clobber caveat.) ===
        if (lora_start_k != 0)
        {
            mainloop_pipeline.producer_acquire(mainloop_pipe_producer_state, barrier_token);
            using BarrierTypePost = typename MainloopPipeline::ProducerBarrierType;
            BarrierTypePost* tb = mainloop_pipeline.producer_get_barrier(mainloop_pipe_producer_state);
            int ws = mainloop_pipe_producer_state.index();
            ++mainloop_pipe_producer_state;
            if (cute::elect_one_sync())
            {
                copy(observed_tma_load_sfa_->with(*tb, mcast_mask_sfa), tAgSFA(_, cute::_0{}),
                    tAsSFA(_, ws)); // dummy: arrive SFA bytes
                copy(observed_tma_load_sfb_->with(*tb, mcast_mask_sfb), tBgSFB(_, cute::_0{}),
                    tBsSFB(_, ws)); // dummy: arrive SFB bytes
                copy(observed_tma_load_d_->with(*tb, mcast_mask_d), tDgD(_, cute::_0{}),
                    tDsD(_, ws));   // D -> smem_A[ws] (== A-stage bytes)
                copy(observed_tma_load_l1_->with(*tb, mcast_mask_l1), tL1gL1(_, cute::_0{}),
                    tL1sL1(_, ws)); // L1 -> smem_B[ws] (== B-stage bytes)
            }
        }

        return cute::make_tuple(mainloop_pipe_producer_state, k_tile_iter);
    }

    /// Perform a Producer Epilogue to prevent early exit of ctas in a Cluster
    CUTLASS_DEVICE void load_tail(
        MainloopPipeline mainloop_pipeline, MainloopPipelineState mainloop_pipe_producer_state)
    {
        // Issue the epilogue waits
        // This helps avoid early exit of ctas in Cluster
        // Waits for all stages to either be released (all
        // Consumer UNLOCKs), or if the stage was never used
        // then would just be acquired since the phase was
        // still inverted from make_producer_start_state
        mainloop_pipeline.producer_tail(mainloop_pipe_producer_state);
    }

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Consumer Perspective
    template <class AccumulatorPipeline, class FrgEngine, class FrgLayout, class MmaParams, class CtaTileCoord>
    CUTLASS_DEVICE auto mma(cute::tuple<MainloopPipeline, AccumulatorPipeline> pipelines,
        cute::tuple<MainloopPipelineState, typename AccumulatorPipeline::PipelineState> pipeline_states,
        cute::tuple<cute::Tensor<FrgEngine, FrgLayout>> const& accumulators_pair, MmaParams const& mma_inputs,
        CtaTileCoord cta_tile_coord, int k_tile_count)
    {
        static_assert(is_tmem<FrgEngine>::value, "Accumulator must be tmem resident.");
        static_assert(rank(FrgLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA, MMA_M, MMA_N)");

        auto accumulators = get<0>(accumulators_pair);
        auto [tiled_mma, tCrA, tCrB, tCtSFA, tCtSFB, tiled_copy_s2t_SFA, thr_tCsSFA_s2t, thr_tCtSFA_s2t,
            tiled_copy_s2t_SFB, thr_tCsSFB_s2t, thr_tCtSFB_s2t, tCrD, tCrL1]
            = mma_inputs;

        auto [mainloop_pipeline, accumulator_pipeline] = pipelines;
        auto [mainloop_pipe_consumer_state, accumulator_pipe_producer_state] = pipeline_states;

        auto tCtSFB_mma = [tCtSFB = tCtSFB, cta_tile_coord]()
        {
            if constexpr (IsCtaN192)
            {
                // If this is an ODD tile, shift the TMEM start address for N=192 case by two words (ignores first 64
                // columns of SFB)
                auto tCtSFB_tmp = tCtSFB;
                if (size<1>(cta_tile_coord) % 2 == 1)
                {
                    tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + 2;
                }
                return tCtSFB_tmp;
            }
            else if constexpr (IsCtaN64)
            {
                // Move in increments of 64 columns of SFB
                auto tCtSFB_tmp = tCtSFB;
                tCtSFB_tmp.data() = tCtSFB_tmp.data().get() + (size<1>(cta_tile_coord) % 2) * 2;
                return tCtSFB_tmp;
            }
            else
            {
                return tCtSFB;
            }
        }();

        uint32_t skip_wait = k_tile_count <= 0;
        auto barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

        //
        // PIPELINED MAIN LOOP
        //
        tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

        // LoRA-up D@L1ᵀ applied in ONE post-K-loop consumer step (see load(): D/L1 are TMA'd into the
        // residual stage buffers smem_A[ws]/smem_B[ws] in an extra producer step after the K-loop). The
        // residual accumulator is still held (producer_acquire'd, not yet committed), and f32 acc adds
        // are commutative, so adding D@L1ᵀ after the full residual == folding it in. accumulate_=One
        // adds. rstage = the consumed pipeline stage holding THIS output tile's D/L1 (tCrD/tCrL1 are
        // stage-moded since SmemLayoutD/L1 overlay the multi-stage smem_A/smem_B).
        auto lora_mma = LoRaMma{};
        lora_mma.accumulate_ = UMMA::ScaleOut::One;
        auto apply_lora = [&](int rstage)
        {
            CUTLASS_PRAGMA_UNROLL
            for (int kb = 0; kb < size<2>(tCrD); ++kb)
            {
                cute::gemm(lora_mma, tCrD(_, _, kb, rstage), tCrL1(_, _, kb, rstage), accumulators);
            }
        };

        if constexpr (IsOverlappingAccum)
        {
            // first iteration manual unroll for tmem overlap kernel
            if (k_tile_count > 0)
            {
                // WAIT on mainloop_pipe_consumer_state until its data are available
                // (phase bit flips from mainloop_pipe_consumer_state.phase() value)
                mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);

                // Compute on k_tile
                int read_stage = mainloop_pipe_consumer_state.index();
                // Save current mainlop pipeline read state
                auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

                // Advance mainloop_pipe
                ++mainloop_pipe_consumer_state;
                --k_tile_count;
                skip_wait = k_tile_count <= 0;
                // Peek at next iteration
                barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

                if (cute::elect_one_sync())
                {
                    copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_, _, _, _, read_stage), thr_tCtSFA_s2t);
                    copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_, _, _, _, read_stage), thr_tCtSFB_s2t);
                }

                // Wait for tmem accumulator buffer to become empty with a flipped phase
                accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);

                // Unroll the K mode manually so we can set scale C to 1
                CUTLASS_PRAGMA_UNROLL
                for (int k_block = 0; k_block < size<2>(tCrA); ++k_block)
                {
                    // (V,M) x (V,N) => (V,M,N)
                    cute::gemm(tiled_mma.with(tiled_mma.accumulate_, tCtSFA(_, _, k_block), tCtSFB_mma(_, _, k_block)),
                        tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accumulators);
                    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
                }
                mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
            }
        }
        else
        {
            // Wait for tmem accumulator buffer to become empty with a flipped phase
            accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
        }

        CUTLASS_PRAGMA_NO_UNROLL
        while (k_tile_count > 0)
        {
            // WAIT on mainloop_pipe_consumer_state until its data are available
            // (phase bit flips from mainloop_pipe_consumer_state.phase() value)
            mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);

            // Compute on k_tile
            int read_stage = mainloop_pipe_consumer_state.index();
            // Save current mainlop pipeline read state
            auto curr_mainloop_pipe_consumer_state = mainloop_pipe_consumer_state;

            // Advance mainloop_pipe
            ++mainloop_pipe_consumer_state;
            --k_tile_count;
            skip_wait = k_tile_count <= 0;
            // Peek at next iteration
            barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);

            if (cute::elect_one_sync())
            {
                copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_, _, _, _, read_stage), thr_tCtSFA_s2t);
                copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_, _, _, _, read_stage), thr_tCtSFB_s2t);
            }

            // Unroll the K mode manually so we can set scale C to 1
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < size<2>(tCrA); ++k_block)
            {
                // (V,M) x (V,N) => (V,M,N)
                cute::gemm(tiled_mma.with(tiled_mma.accumulate_, tCtSFA(_, _, k_block), tCtSFB_mma(_, _, k_block)),
                    tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage), accumulators);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);
        }

        // === consume the SINGLE post-K-loop D/L1 producer step (load() emits it
        // only in the mainloop call, gated by lora_start_k != 0, so exactly one occurs at the end of
        // the k-tile sequence), then run the LoRA MMA reading D/L1 from the consumed stage buffer
        // (read_stage = the smem_A[stage]/smem_B[stage] this step's D/L1 landed in) into the still-held
        // residual accumulator. One consumer step balances the one producer step. ===
        {
            auto lora_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state);
            mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, lora_token);
            int lora_read_stage = mainloop_pipe_consumer_state.index();
            apply_lora(lora_read_stage);
            mainloop_pipeline.consumer_release(mainloop_pipe_consumer_state);
            ++mainloop_pipe_consumer_state;
        }

        return mainloop_pipe_consumer_state;
    }

protected:
    typename Params::TMA_A const* observed_tma_load_a_{nullptr};
    typename Params::TMA_B const* observed_tma_load_b_{nullptr};
    typename Params::TMA_SFA const* observed_tma_load_sfa_{nullptr};
    typename Params::TMA_SFB const* observed_tma_load_sfb_{nullptr};
    typename Params::TMA_D const* observed_tma_load_d_{nullptr};
    typename Params::TMA_L1 const* observed_tma_load_l1_{nullptr};

    LayoutSFA layout_SFA_;
    LayoutSFB layout_SFB_;
    RuntimeDataTypeA runtime_data_type_a_{};
    RuntimeDataTypeB runtime_data_type_b_{};

    ClusterShape cluster_shape_;
    uint32_t block_rank_in_cluster_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
