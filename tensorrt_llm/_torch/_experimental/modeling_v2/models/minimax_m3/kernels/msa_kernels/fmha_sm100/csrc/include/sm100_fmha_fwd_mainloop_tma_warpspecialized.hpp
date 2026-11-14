/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <climits>

#include "gmem_bounds_check.h"
#include "cute/arch/simd_sm100.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "fmha_common.hpp"
#include "fmha_fusion.hpp"
#include "gpu_trace.h"
#include "sm100_fmha_load_tma_warpspecialized.hpp"

GPU_TRACE_SCOPE_DEC(SOFTMAX_Mask);
GPU_TRACE_SCOPE_DEC(SOFTMAX_GetMax);
GPU_TRACE_SCOPE_DEC(SOFTMAX_Exp);
GPU_TRACE_SCOPE_DEC(SOFTMAX_Sum);
GPU_TRACE_SCOPE_DEC(GEMM_QK0);
GPU_TRACE_SCOPE_DEC(GEMM_QK1);
GPU_TRACE_SCOPE_DEC(GEMM_PV0);
GPU_TRACE_SCOPE_DEC(GEMM_PV1);
GPU_TRACE_SCOPE_DEC(CORRECTION);
GPU_TRACE_SCOPE_DEC(CORRECTION_MRG);
GPU_TRACE_SCOPE_DEC(CORRECTION_EPI);
GPU_TRACE_SCOPE_DEC(WAIT_Q);
GPU_TRACE_SCOPE_DEC(DUP_Q);

GPU_TRACE_SCOPE_DEC(WAIT_QK);
GPU_TRACE_SCOPE_DEC(WAIT_K);
GPU_TRACE_SCOPE_DEC(WAIT_V);

GPU_TRACE_SCOPE_DEC(RELEASE_Q);

#define TRACE_MMA 1
#define TRACE_SOFTMAX 0
#define TRACE_CORR 0

namespace cutlass::fmha::collective {

using namespace cute;

// Unidirectional async pipeline: only consumer→producer direction (empty barrier).
// Producer calls wait(), consumer calls arrive(). Index auto-advances internally.
// Saves shared memory and initialization cost vs bidirectional PipelineAsync/PipelineUmmaAsync
// when producer_commit/consumer_wait are unused.
template <int Stages_>
class PipelineAsyncWaitArrive {
public:
  static constexpr uint32_t Stages = Stages_;
  using BarrierType = cutlass::arch::ClusterBarrier;

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t consumer_arv_count = 1;
    uint32_t dst_blockid = cute::block_rank_in_cluster();
    int initializing_warp = 0;
  };

  struct SharedStorage {
    BarrierType barrier_[Stages];
  };

  CUTLASS_DEVICE
  PipelineAsyncWaitArrive(SharedStorage& storage, Params const& params,
                          cute::true_type /*init_barriers*/ = {})
      : params_(params)
      , barrier_ptr_(&storage.barrier_[0])
      , index_(0)
      , phase_((params.role == ThreadCategory::Producer ||
                params.role == ThreadCategory::ProducerConsumer) ? 1u : 0u) {
    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == params_.initializing_warp) {
      if (cute::elect_one_sync()) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Stages; i++) {
          storage.barrier_[i].init(params_.consumer_arv_count);
        }
      }
    }
    cutlass::arch::fence_barrier_init();
  }

  // Producer waits for consumer to release a stage
  CUTLASS_DEVICE
  void wait() {
    barrier_ptr_[index_].wait(phase_);
    advance();
  }

  // Consumer signals that a stage is available for the producer
  CUTLASS_DEVICE
  void arrive() {
    barrier_ptr_[index_].arrive(params_.dst_blockid);
    advance();
  }

private:
  Params params_;
  BarrierType* barrier_ptr_;
  int index_;
  uint32_t phase_;

  CUTLASS_DEVICE
  void advance() {
    if (++index_ == Stages) {
      index_ = 0;
      phase_ ^= 1;
    }
  }
};

template <class Element_, class ElementQK_, class ElementPV_, class TileShapeQK_,
          class TileShapePV_, class StrideQ_, class StrideK_, class StrideV_, class Mask_,
          // shape here is QG K H
          // and referes to the two softmax warps
          // (2, 1, 1) means that they are stacked (best for large Q since it loads the least K/V)
          // (1, 2, 1) means they sit side by side (best for small Q / large K)
          class ThreadShape = Shape<_2, _1, _1>,
          bool IsSplitKV_ = false,
          int KVPageSize_ = -1,
          SparseAttnMode kSparseAttnMode = SparseAttnMode::Off,
          bool UnitSelectorColumn = false>
struct Sm100FmhaFwdMainloopTmaWarpspecialized {
  static constexpr bool IsSplitKV = IsSplitKV_;
  static constexpr int KVPageSize = KVPageSize_;

  static constexpr bool kNeedMaxScore = (kSparseAttnMode == SparseAttnMode::OnlyScore || kSparseAttnMode == SparseAttnMode::Full);
  static constexpr bool kNeedOutput = kSparseAttnMode != SparseAttnMode::OnlyScore;
  static constexpr bool kNeedSparse = kSparseAttnMode == SparseAttnMode::Sparse;
  // OnlyScore mode: correction's only real work is writing max_score to GMEM
  // (rescale paths are kNeedOutput-gated). Let softmax do the GMEM write
  // directly from its tile_max register, and have correction skip its own
  // GMEM write. All TMEM round-trip and PipelineC handshakes are preserved
  // for safety (no sync elision in v1 — those are TODO for v2).
  static constexpr bool kFuseMaxScoreIntoSoftmax =
      (kSparseAttnMode == SparseAttnMode::OnlyScore);
  // OnlyScore K-split: correction has no per-tile work (softmax writes the max scores itself), so the per-tile
  // softmax -> correction -> MMA handshake (pipeline_c commit/acquire, correction's s0_c/s1_c wait/release and
  // pipeline_corr arrive/wait) only serializes the two softmax warpgroups. It is skipped; pipeline_s (S0/S1 TMEM
  // reuse between MMA and softmax) and the final pipeline_c handshake into the split-KV merge/epilogue are kept.
  // Defining FMHA_ONLYSCORE_PER_TILE_CORRECTION (jit.py: TLLM_FMHA_PROXY_CORRECTION_CHAIN=1) keeps the handshake.
#ifdef FMHA_ONLYSCORE_PER_TILE_CORRECTION
  static constexpr bool kSkipPerTileCorrection = false;
#else
  static constexpr bool kSkipPerTileCorrection = kFuseMaxScoreIntoSoftmax && (get<1>(ThreadShape{}) > 1);
#endif

  struct SplitKVParams {
    float scale_output_splitkv = 1.0f;
    float scale_output_nosplit = 1.0f;
    float* ptr_lse_accum = nullptr;
    int* kv_tile_begin_indices = nullptr;
    int* kv_tile_end_indices = nullptr;
    int* kv_split_indices = nullptr;
    int total_qo_len = 0;
    int num_qo_heads = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int split_kv_buf_size = 0;
#endif
  };

  using Element = Element_;
  using ElementQK = ElementQK_;
  using ElementPV = ElementPV_;
  using TileShape = decltype(select<0, 1>(TileShapeQK_{}));
  using TileShapeQK = decltype(shape_div(TileShapeQK_{}, ThreadShape{}));
  // PV has dims (Q, D_VO, KV) — KV is at index 2, not 1.
  // Swap ThreadShape indices 1,2 so division falls on KV (index 2), not D_VO (index 1).
  using ThreadShapePV = decltype(select<0, 2, 1>(ThreadShape{}));
  using TileShapePV = decltype(shape_div(TileShapePV_{}, ThreadShapePV{}));
  using StrideQ = StrideQ_;
  using StrideK = StrideK_;
  using StrideV = StrideV_;
  using Mask = Mask_;

  static constexpr bool AggressiveLoopInvariantCodeMotionOfSoftmax = true;//get<1>(ThreadShape{}) == 1; // meet compiler error in decode
  static constexpr bool EnableEmuExp2 = true;
  // exp2 emulation tuning: replace a fraction of hardware exp2f with FMA polynomial.
  //   Freq: period length (in elements, step=2). Must divide fragment size.
  //   Res: how many elements at the end of each period use emulation.
  //   StartFrg: first eligible fragment (0-based). SkipLastFrg: skip last fragment.
  // Replacement rate ≈ (Res/Freq) × (eligible_frags/total_frags).
  // Current: (4/16) × (2/4) = 12.5%.  Try Res=8 for 25%, or StartFrg=0 for 18.75%.
  static constexpr int kExp2EmuFreq = 16;
  static constexpr int kExp2EmuRes = 4;
  static constexpr int kExp2EmuStartFrg = 1;
  static constexpr bool kExp2EmuSkipLastFrg = true;
  // Q-split (2,1,1): 2 Q stages (Q0 and Q1 in smem pipeline)
  // K-split (1,2,1): 1 Q stage (only Q0 needed, same Q for both stages)
  static constexpr int StageCountQ = (get<0>(ThreadShape{}) > 1) ? 2 : 1;
  // static constexpr int StageCountQ = (get<0>(ThreadShape{}) > 1) ? 2 : (sizeof(Element_) == 1 ? 2 : 1);
  // K-split uses dynamic softmax warp count (controlled by kernel-level warp guard)
  static constexpr bool kEnablePaddingSkip = (get<0>(ThreadShape{}) == 1);

  template <class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE static int get_full_trip_count(BlkCoord const& blk_coord,
                                                 ProblemShape const& problem_shape,
                                                 int kv_block_num = 0) {
    if constexpr (kNeedSparse) {
      constexpr int tile_kv = get<1>(TileShape{});
      return (kv_block_num * KVPageSize + tile_kv - 1) / tile_kv;
    } else {
      return Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
    }
  }

  template <class BlkCoord, class ProblemShape>
  CUTLASS_DEVICE static int get_effective_trip_count(BlkCoord const& blk_coord,
                                                      ProblemShape const& problem_shape,
                                                      int kv_tile_begin, int kv_tile_end,
                                                      int kv_block_num = 0) {
    int full = get_full_trip_count(blk_coord, problem_shape, kv_block_num);
    int eff_end = full < kv_tile_end ? full : kv_tile_end;
    int count = eff_end - kv_tile_begin;
    return count > 0 ? count : 0;
  }

  // Compute StageCountKV dynamically based on smem budget (FA4 style).
  // For FP8 d=128: (224KB - Q_smem - O_smem) / KV_per_stage ≈ 8 stages (vs hardcoded 4).
  // Fallback to the original formula if dynamic computation is not feasible for a config.
  static constexpr int StageCountKV_base =
      (sizeof(Element_) == 1)
          ? (get<2>(TileShapeQK{}) == 128 ? 4 : 2)
          : (get<2>(TileShapeQK{}) == 128 || get<2>(TileShapeQK{}) == 64 ? 2 : 1);
  // For dynamic: estimate smem per KV stage, Q total, O total, then maximize stages
  static constexpr int SmemPerStageKV_bytes =
      get<1>(TileShapeQK{}) * get<2>(TileShapeQK{}) * sizeof(Element_);  // N * D * element_size
  static constexpr int SmemQ_bytes =
      StageCountQ * get<0>(TileShapeQK{}) * get<2>(TileShapeQK{}) * sizeof(Element_);  // stages * M * D
  // OnlyScore mode: epilogue's smem_o is trimmed to 1 byte (NeedOutput_ gate),
  // so we don't pay the StageCountQ * kQRows * D output buffer cost.
  static constexpr int SmemO_bytes =
      kNeedOutput
          ? (StageCountQ * get<0>(TileShapeQK{}) * get<2>(TileShapeQK{}) *
             ((sizeof(Element_) == 1) ? 2 : (int)sizeof(Element_)))
          : 0;
  static constexpr int SmemBudget = 224 * 1024;
  static constexpr int StageCountKV_dynamic =
      (SmemBudget - SmemQ_bytes - SmemO_bytes) / SmemPerStageKV_bytes;
  // Use dynamic if it gives more stages, otherwise use base
  static constexpr int StageCountKV = // values was found by profiling
      sizeof(Element_) == 1 ?
        ( (get<0>(ThreadShape{}) == 1) ? 8 : 2 )
        : ( (get<0>(ThreadShape{}) == 1) ? 4 : 2 );
      // (StageCountKV_dynamic > StageCountKV_base) ? StageCountKV_dynamic : StageCountKV_base;

  using StagesQ = cutlass::gemm::collective::StageCount<StageCountQ>;
  using StagesKV = cutlass::gemm::collective::StageCount<StageCountKV>;

  static_assert(StageCountKV_dynamic >= StageCountKV);

  using ClusterShape = Shape<_1, _1, _1>;

  static const int Alignment = 128 / sizeof_bits_v<Element>;

  using CollectiveMmaQK = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Element, StrideQ, Alignment, Element,
      StrideK, Alignment, ElementQK, TileShapeQK, ClusterShape,
      cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using CollectiveMmaPV = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      // the stride for A does not matter since we do not load from smem at all
      Element, StrideK, Alignment, Element, StrideV, Alignment, ElementPV, TileShapePV,
      ClusterShape, cutlass::gemm::collective::StageCount<3> /* we change it later anyways*/,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using SmemLayoutQ =
      decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutA{}, Int<StageCountQ>{}));
  using SmemLayoutK =
      decltype(unstageSmemLayout(typename CollectiveMmaQK::SmemLayoutB{}, Int<StageCountKV>{}));
  using SmemLayoutV =
      decltype(unstageSmemLayout(typename CollectiveMmaPV::SmemLayoutB{}, Int<StageCountKV>{}));

  // K and V share the same physical smem buffer (FA4 style).
  // Use the larger cosize for the shared buffer.
  static constexpr size_t kSmemKCosize = cute::cosize_v<SmemLayoutK>;
  static constexpr size_t kSmemVCosize = cute::cosize_v<SmemLayoutV>;
  static constexpr size_t kSmemKVCosize = (kSmemKCosize > kSmemVCosize) ? kSmemKCosize : kSmemVCosize;

  // TensorStorage: shared K/V smem buffer + Q (no separate smem_v or smem_k)
  // smem_o is allocated separately in the kernel (no union overlap)
  struct TensorStorage {
    cute::array_aligned<Element, kSmemKVCosize> smem_kv;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
  };

  enum class TmemAllocation : uint32_t {
    kSizeS = 128,
    kSizeO = 128,
    kSizeP = 32,
    S0 = 0,
    S1 = S0 + kSizeS,   // 128
    V0 = S0,            // 0   // stats storage from softmax to correction
    V1 = S1,            // 128
    P0 = S0 + kSizeP,   // 32
    P1 = S1 + kSizeP,   // 160
    O0 = S1 + kSizeS,   // 256
    O1 = O0 + kSizeO,   // 384
    kEnd = O1 + kSizeO  // 512
  };

  // indices for V0 / V1
  enum : int { kIdxOldRowMax = 0, kIdxNewRowMax = 1, kIdxFinalRowSum = 0, kIdxFinalRowMax = 1 };
  static constexpr int kVStatsWidth = 2;

  // from load to mma warp, protects q in smem
  using PipelineQ =
      cutlass::PipelineTmaUmmaAsync<StageCountQ, typename CollectiveMmaQK::AtomThrShapeMNK>;

  // from load to mma warp, protects k/v in smem (merged: K and V share one pipeline)
  using PipelineKV =
      cutlass::PipelineTmaUmmaAsync<StageCountKV, typename CollectiveMmaQK::AtomThrShapeMNK>;

  // from mma to softmax0/1 warp, protects S in tmem
  // (not sure yet about the reverse direction)
  // there is one pipe per softmax warp, and the mma warp alternates between them
  // OnlyScore K-split keeps no O accumulator in TMEM, so each Q stage double-buffers S: buffer 1 of S0 / S1 sits in
  // the O0 / O1 columns (+256), and the MMA fills tile i+1 while the softmax warps still read tile i. Nothing in
  // OnlyScore reads or writes O (the O-touching correction paths static_assert kNeedOutput).
  // Defining FMHA_ONLYSCORE_SINGLE_S (jit.py: TLLM_FMHA_PROXY_S_DOUBLE_BUFFER=0) keeps one S buffer per Q stage.
#ifdef FMHA_ONLYSCORE_SINGLE_S
  static constexpr bool kSDoubleBuf = false;
#else
  static constexpr bool kSDoubleBuf = kSkipPerTileCorrection;
#endif
  static_assert(!kSDoubleBuf || !kNeedOutput, "S double-buffering reuses the O TMEM columns");
  static constexpr uint32_t kSBuf1Offset = uint32_t(TmemAllocation::O0) - uint32_t(TmemAllocation::S0);
  using PipelineS = cutlass::PipelineUmmaAsync<kSDoubleBuf ? 2 : 1>;

  // from softmax0/1/ to correction wg
  using PipelineC = cutlass::PipelineAsync<1>;

  // from correction to mma
  using PipelineO = PipelineAsyncWaitArrive<2>;

  // from corr to epilogue
  // Use num_q_sub stages: Q-split (2,1,1) → 2, K-split (1,2,1) → 1
  // K-split only writes smem_o[0], so PipelineAsync<2> would allow producer to
  // overwrite smem_o[0] before consumer reads it. PipelineAsync<1> prevents this.
  static constexpr int NumQSub = get<0>(ThreadShape{});
  using PipelineE = cutlass::PipelineAsync<NumQSub>;

  static const int TransactionBytesLoadQ =
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);

  static const int TransactionBytesLoadK =
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);

  static const int TransactionBytesLoadV =
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutV{})) * cute::sizeof_bits_v<Element>);

  // For interleaved KV pipeline, each stage holds only K or V (not both)
  // transaction_bytes = K bytes (K and V are same size for FP8 d=128)
  static const int TransactionBytesLoadKV = TransactionBytesLoadK;

  using Load = Sm100FmhaLoadTmaWarpspecialized<Element, CollectiveMmaQK, CollectiveMmaPV,
                                               SmemLayoutQ, SmemLayoutK, SmemLayoutV, TensorStorage,
                                               PipelineQ, PipelineKV, Mask, TileShape, KVPageSize,
                                               kSparseAttnMode, UnitSelectorColumn>;
  using LayoutQ = typename Load::LayoutQ;
  using LayoutK = typename Load::LayoutK;
  using LayoutV = typename Load::LayoutV;

  struct Arguments {
    typename Load::Arguments load;

    float scale_softmax;

    // scaling factors to dequantize QKV
    float scale_q = 1.0f;
    float scale_k = 1.0f;
    float scale_v = 1.0f;

    // scaling factor to quantize O
    float inv_scale_o = 1.0f;
  };

  struct Params {
    typename Load::Params load;

    float scale_softmax;
    float scale_softmax_log2;

    float scale_output;
  };

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
                                        void* workspace) {
    float scale_softmax = args.scale_softmax;
    float log2_e = static_cast<float>(std::log2(std::exp(1.0)));

    return Params{Load::to_underlying_arguments(problem_shape, args.load, workspace),
                  args.scale_q * args.scale_k * scale_softmax,
                  args.scale_q * args.scale_k * log2_e * scale_softmax,
                  args.scale_v * args.inv_scale_o};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    Load::prefetch_tma_descriptors(params.load);
  }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape, class TensorStorageType>
  CUTLASS_DEVICE void load(BlkCoord const& blk_coord, ProblemShape const& problem_shape,
                           Params const& params, ParamsProblemShape const& params_problem_shape,
                           int const& work_idx,
                           TensorStorageType& storage, PipelineQ& pipeline_q,
                           typename PipelineQ::PipelineState& pipeline_q_producer_state,
                           PipelineKV& pipeline_kv,
                           typename PipelineKV::PipelineState& pipeline_kv_producer_state
                           #ifdef GPU_TRACE_ENABLED
                             , gpu_trace::Recorder& _gt_rec
                           #endif
                           , int kv_tile_begin = 0, int kv_tile_end = INT_MAX
                          ) {
    Load load_impl;
    load_impl.template load<IsSplitKV, kNeedOutput>(blk_coord, problem_shape, params.load, params_problem_shape, work_idx,
              storage, pipeline_q, pipeline_q_producer_state, pipeline_kv,
              pipeline_kv_producer_state, kv_tile_begin, kv_tile_end
              #ifdef GPU_TRACE_ENABLED
                , _gt_rec
              #endif
            );
  }

  template <bool kSingleWG = false, class BlkCoord, class ProblemShape, class TensorStorageType>
  CUTLASS_DEVICE auto mma(
      BlkCoord const& blk_coord, Params const& params, ProblemShape const& problem_shape,
      TensorStorageType& storage, PipelineQ& pipeline_q,
      typename PipelineQ::PipelineState& pipeline_q_consumer_state, PipelineKV& pipeline_kv,
      typename PipelineKV::PipelineState& pipeline_kv_consumer_state, PipelineS& pipeline_s0,
      typename PipelineS::PipelineState& pipeline_s0_producer_state, PipelineS& pipeline_s1,
      typename PipelineS::PipelineState& pipeline_s1_producer_state, PipelineO& pipeline_corr,
      int kv_tile_begin = 0, int kv_tile_end = INT_MAX) {
    GET_GPU_TRACE(TRACE_MMA);

    auto pipeline_q_release_state = pipeline_q_consumer_state;
    auto pipeline_kv_release_state = pipeline_kv_consumer_state;

    int mask_tile_count;
    if constexpr (IsSplitKV || kNeedSparse) {
      mask_tile_count = get_effective_trip_count(blk_coord, problem_shape,
                                                 kv_tile_begin, kv_tile_end, params.load.kv_block_num);
    } else {
      mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
    }

    typename CollectiveMmaQK::TiledMma mma_qk;
    ThrMMA thr_mma_qk = mma_qk.get_slice(0);

    typename CollectiveMmaPV::TiledMma mma_pv;
    TiledMMA mma_pv_ts = to_tiled_mma_sm100_ts(mma_pv);
    ThrMMA thr_mma_pv = mma_pv_ts.get_slice(0);

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    // K and V share the same physical smem buffer (smem_kv), interpreted with different layouts
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_kv.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_kv.data()), SmemLayoutV{});

    Tensor tSrQ = thr_mma_qk.make_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.make_fragment_B(sK);
    Tensor tOrV = thr_mma_pv.make_fragment_B(sV);

    Tensor tStS = partition_fragment_C(mma_qk, select<0, 1>(TileShapeQK{}));
    Tensor tOtO = partition_fragment_C(mma_pv_ts, select<0, 1>(TileShapePV{}));

    Tensor tStS0 = tStS;
    tStS0.data() = tStS.data().get() + uint32_t(TmemAllocation::S0);
    Tensor tStS1 = tStS;
    tStS1.data() = tStS.data().get() + uint32_t(TmemAllocation::S1);
    Tensor tStS0b = tStS0;
    tStS0b.data() = tStS0.data().get() + kSBuf1Offset;
    Tensor tStS1b = tStS1;
    tStS1b.data() = tStS1.data().get() + kSBuf1Offset;
    auto s0_buf = [&](auto const& state) { return (kSDoubleBuf && state.index() == 1) ? tStS0b : tStS0; };
    auto s1_buf = [&](auto const& state) { return (kSDoubleBuf && state.index() == 1) ? tStS1b : tStS1; };

    Tensor tOtO0 = tOtO;
    tOtO0.data() = tOtO.data().get() + uint32_t(TmemAllocation::O0);
    Tensor tOtO1 = tOtO;
    tOtO1.data() = tOtO.data().get() + uint32_t(TmemAllocation::O1);

    Tensor sP =
        make_tensor(make_smem_ptr((Element*)nullptr), typename CollectiveMmaPV::SmemLayoutA{});
    Tensor tOrP = thr_mma_pv.make_fragment_A(sP)(_, _, _, _0{});

    Tensor tOrP0 = tOrP;
    tOrP0.data() = tOrP0.data().get() + uint32_t(TmemAllocation::P0);
    Tensor tOrP1 = tOrP;
    tOrP1.data() = tOrP1.data().get() + uint32_t(TmemAllocation::P1);

    int k_index = 0;
    int v_index = 0;
    int v_odd_index = 0;  // K-split: V index for odd KV tile (used by PV→O1)
    int q_index = 0;

    // =========== Init: QK0, QK1, then consume first V ===========

    // wait for Q0
    {GPU_TRACE_SCOPE(WAIT_Q);
    q_index = pipeline_q_consumer_state.index();
    pipeline_q.consumer_wait(pipeline_q_consumer_state);
    ++pipeline_q_consumer_state;
    }

    // duplicate Q[0:64] → Q[64:128] in SMEM
    // Row r and row r+64 occupy the same column positions in SMEM but are offset
    // by a fixed physical byte distance (the "segment size"). This distance is
    // derived at compile time from SmemLayoutQ: ql(row=64,col=0) - ql(row=0,col=0).
    // For SW128 swizzle this equals 4096 elems = 8192 bytes; for other swizzle
    // modes the layout auto-adapts.
    // When stage_bytes > 2*seg_bytes, the stage contains N pairs of segments
    // (row 0..63 in even segments, row 64..127 in odd segments); each pair is
    // copied independently.
    if constexpr (kSingleWG) {
      GPU_TRACE_SCOPE(DUP_Q);
      constexpr int kSegElems = int(SmemLayoutQ{}(
          cute::make_tuple(cute::make_tuple(cute::Int<64>{}, cute::Int<0>{}),
                           cute::Int<0>{}, cute::Int<0>{}, cute::Int<0>{})));
      constexpr int kSegBytes = kSegElems * int(sizeof(Element));
      constexpr int kSegU4 = kSegBytes / 16;
      constexpr int kStageBytes =
          int(cute::cosize_v<SmemLayoutQ>) * int(sizeof(Element)) / int(StageCountQ);
      constexpr int kNumPairs = kStageBytes / (2 * kSegBytes);
      static_assert(kSegBytes > 0 && kSegBytes % 16 == 0,
                    "DUP_Q segment size must be positive and uint4-aligned");
      static_assert(kStageBytes % (2 * kSegBytes) == 0,
                    "stage byte size must be multiple of 2 * segment size");
      int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
      auto* q_base = reinterpret_cast<uint4*>(storage.smem_q.data())
                     + q_index * (kStageBytes / 16);
      CUTLASS_PRAGMA_UNROLL
      for (int p = 0; p < kNumPairs; p++) {
        int base = p * 2 * kSegU4;
        for (int i = lane; i < kSegU4; i += cutlass::NumThreadsPerWarp) {
          q_base[base + kSegU4 + i] = q_base[base + i];
        }
      }
      cutlass::arch::fence_view_async_shared();
    }
    Tensor tSrQ0 = tSrQ(_, _, _, q_index);

    // wait for K0 (stage 0 in interleaved pipeline)
    {GPU_TRACE_SCOPE(WAIT_K);
    k_index = pipeline_kv_consumer_state.index();
    pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
    ++pipeline_kv_consumer_state;
    }

    // QK_s0: Q0 * K0 -> S0
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);
    {
      // GPU_TRACE_SCOPE(GEMM_QK0);
      { auto tS_cur = s0_buf(pipeline_s0_producer_state); gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tS_cur); }
    }
    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;

    // K-split: release K_even, consume K_odd
    if constexpr (get<1>(ThreadShape{}) > 1) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
      GPU_TRACE_SCOPE(WAIT_K);
      k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    if constexpr (get<0>(ThreadShape{}) > 1 || get<2>(ThreadShape{}) > 1) {
      q_index = pipeline_q_consumer_state.index();
      pipeline_q.consumer_wait(pipeline_q_consumer_state);
      ++pipeline_q_consumer_state;
    }
    Tensor tSrQ1 = tSrQ(_, _, _, q_index);

    // QK_s1: Q-split: Q1 * K0 -> S1; K-split: Q0 * K1 -> S1
    pipeline_s1.producer_acquire(pipeline_s1_producer_state);
    {//GPU_TRACE_SCOPE(GEMM_QK1);
    { auto tS_cur = s1_buf(pipeline_s1_producer_state); gemm_zero_acc(mma_qk, tSrQ1, tSrK(_, _, _, k_index), tS_cur); }
    }
    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // release K (Q-split: release K0; K-split: release K1)
    pipeline_kv.consumer_release(pipeline_kv_release_state);
    ++pipeline_kv_release_state;

    mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero;


    // =========== Main loop: PV0 → QK0 → PV1 → QK1 ===========
    // K-split FIFO order: V_even, K_even, V_odd, K_odd per group (matching PV0→QK0→PV1→QK1)
    // Q-split FIFO order: K, V per group
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {

      // consume V_even
      if constexpr (kNeedOutput) {
        GPU_TRACE_SCOPE(WAIT_V);
        v_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      // PV0: P0 * V_even -> O0
      if constexpr (!kSkipPerTileCorrection) { pipeline_corr.wait(); }
      pipeline_s0.producer_acquire(pipeline_s0_producer_state);
      if constexpr(kNeedOutput) {
        // GPU_TRACE_SCOPE(GEMM_PV0);
        auto save_acc = mma_pv_ts.accumulate_;
        gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_, _, _, v_index), tOtO0);
        mma_pv_ts.accumulate_ = save_acc;
      }

      // K-split: V_even only used by PV0, release immediately
      if constexpr (get<1>(ThreadShape{}) > 1 && kNeedOutput) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }

      // consume K_even
      {GPU_TRACE_SCOPE(WAIT_K);
      k_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
      }

      // QK_s0: Q0 * K_even -> S0
      {
        // GPU_TRACE_SCOPE(GEMM_QK0);
        { auto tS_cur = s0_buf(pipeline_s0_producer_state); gemm_zero_acc(mma_qk, tSrQ0, tSrK(_, _, _, k_index), tS_cur); }
        pipeline_s0.producer_commit(pipeline_s0_producer_state);
        ++pipeline_s0_producer_state;
      }

      // K-split: release K_even, then consume V_odd
      if constexpr (get<1>(ThreadShape{}) > 1) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;

        if constexpr (kNeedOutput) {
          GPU_TRACE_SCOPE(WAIT_V);
          v_odd_index = pipeline_kv_consumer_state.index();
          pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
          ++pipeline_kv_consumer_state;
        }
      }

      if constexpr (!kSkipPerTileCorrection) { pipeline_corr.wait(); }
      pipeline_s1.producer_acquire(pipeline_s1_producer_state);
      if constexpr(kNeedOutput) {
        // GPU_TRACE_SCOPE(GEMM_PV1);
        int pv1_v_index;
        if constexpr (get<1>(ThreadShape{}) > 1) { pv1_v_index = v_odd_index; }
        else { pv1_v_index = v_index; }
        gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_, _, _, pv1_v_index), tOtO1);
      }

      // Q-split: release V (after PV1, both PV0 and PV1 used same V)
      // K-split: release V_odd (after PV1), then consume K_odd
      if constexpr (kNeedOutput) {
        pipeline_kv.consumer_release(pipeline_kv_release_state);
        ++pipeline_kv_release_state;
      }
      if constexpr (get<1>(ThreadShape{}) > 1) {
        GPU_TRACE_SCOPE(WAIT_K);
        k_index = pipeline_kv_consumer_state.index();
        pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
        ++pipeline_kv_consumer_state;
      }

      // QK_s1: Q-split: Q1 * K_even -> S1; K-split: Q0 * K_odd -> S1
      // Must happen BEFORE releasing K!
      {
        // GPU_TRACE_SCOPE(GEMM_QK1);
        { auto tS_cur = s1_buf(pipeline_s1_producer_state); gemm_zero_acc(mma_qk, tSrQ1, tSrK(_, _, _, k_index), tS_cur); }
        pipeline_s1.producer_commit(pipeline_s1_producer_state);
        ++pipeline_s1_producer_state;
      }

      // release K (Q-split: K_even after QK1; K-split: K_odd after QK1)
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    // =========== Tail: final PV0 and PV1 ===========

    // release Q
    {
      GPU_TRACE_SCOPE(RELEASE_Q);
      pipeline_q.consumer_release(pipeline_q_release_state);
      ++pipeline_q_release_state;
      if constexpr (get<0>(ThreadShape{}) > 1) {
        pipeline_q.consumer_release(pipeline_q_release_state);
        ++pipeline_q_release_state;
      }
    }

    // consume V_even for next iteration
    if constexpr (kNeedOutput) {
      GPU_TRACE_SCOPE(WAIT_V);
      v_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    // final PV0: P0 * V_last_even -> O0
    if constexpr (!kSkipPerTileCorrection) { pipeline_corr.wait(); }
    pipeline_s0.producer_acquire(pipeline_s0_producer_state);
    if constexpr(kNeedOutput) {
      GPU_TRACE_SCOPE(GEMM_PV0);
      auto save_acc = mma_pv_ts.accumulate_;
      gemm_reset_zero_acc(mma_pv_ts, tOrP0, tOrV(_, _, _, v_index), tOtO0);
      mma_pv_ts.accumulate_ = save_acc;
    }

    // K-split: release V_even (PV0 done, PV1 uses separate V_odd), then consume V_odd
    // Q-split: do NOT release V_even yet — PV1 reuses the same V buffer
    if constexpr (get<1>(ThreadShape{}) > 1 && kNeedOutput) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
      GPU_TRACE_SCOPE(WAIT_V);
      v_odd_index = pipeline_kv_consumer_state.index();
      pipeline_kv.consumer_wait(pipeline_kv_consumer_state);
      ++pipeline_kv_consumer_state;
    }

    // final PV1: P1 * V_last -> O1
    if constexpr (!kSkipPerTileCorrection) { pipeline_corr.wait(); }
    pipeline_s1.producer_acquire(pipeline_s1_producer_state);
    if constexpr(kNeedOutput) {
      GPU_TRACE_SCOPE(GEMM_PV1);
      int pv1_v_index;
      if constexpr (get<1>(ThreadShape{}) > 1) { pv1_v_index = v_odd_index; }
      else { pv1_v_index = v_index; }
      gemm_reset_zero_acc(mma_pv_ts, tOrP1, tOrV(_, _, _, pv1_v_index), tOtO1);
    }

    // release last V
    // Q-split: releases V_even (after both PV0 and PV1 used it)
    // K-split: releases V_odd (V_even was released above before PV1)
    if constexpr (kNeedOutput) {
      pipeline_kv.consumer_release(pipeline_kv_release_state);
      ++pipeline_kv_release_state;
    }

    // final S signals for softmax's final_call
    pipeline_s0.producer_commit(pipeline_s0_producer_state);
    ++pipeline_s0_producer_state;
    pipeline_s1.producer_commit(pipeline_s1_producer_state);
    ++pipeline_s1_producer_state;

    // T0 S00 B1, T0 S10 B1, T0 S00 B2, T0 S01 B1, T0 S10 B2, T0 S11 B1, T0 S01 B2, T1 S00 B1, T0
    // S11 B2, ... Q1 * K1  , Q2 * K1  , S11 * V1 , Q1 * K2  , S21 * V1  , Q2 * K2 , S12 * V2 , Q1 *
    // K3  , S22 * K2 , ...

    RELEASE_GPU_TRACE;
  }

  // Pre-computed TMEM context for softmax, initialized once per tile.
  // All members including derived ones (thread slices, partitions) are computed
  // in the constructor using this->, so internal references point to this object's
  // own members. Must not be copied/moved after construction.
  struct SoftmaxCtx {
    // Type helpers: compute types from a dummy construction
    static auto _dummy_tStS() {
      return partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0, 1>(TileShapeQK{}));
    }
    static auto _dummy_tStS_v() { return _dummy_tStS().compose(make_layout(make_shape(_128{}, Int<kVStatsWidth>{}))); }
    static constexpr auto _tilePlikeFP32() {
      return get<1>(TileShapeQK{}) / Int<sizeof(float)>{} * Int<sizeof(Element)>{};
    }
    static auto _dummy_tStS_P() { return _dummy_tStS().compose(make_layout(make_shape(_128{}, _tilePlikeFP32()))); }

    using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b32x;
    using TMEM_STORE = SM100_TMEM_STORE_32dp32b32x;
    using TMEM_STORE_V = SM100_TMEM_STORE_32dp32b2x;

    static auto _dummy_load() { return make_tmem_copy(TMEM_LOAD{}, _dummy_tStS()); }
    static auto _dummy_storev() { return make_tmem_copy(TMEM_STORE_V{}, _dummy_tStS_v()); }
    static auto _dummy_store() { return make_tmem_copy(TMEM_STORE{}, _dummy_tStS_P()); }
    static auto _dummy_thr_load() { return _dummy_load().get_slice(0); }
    static auto _dummy_thr_storev() { return _dummy_storev().get_slice(0); }
    static auto _dummy_thr_store() { return _dummy_store().get_slice(0); }

    // Root members
    decltype(_dummy_tStS()) tStS;
    decltype(_dummy_tStS_v()) tStS_v;
    decltype(_dummy_tStS_P()) tStS_P;
    decltype(_dummy_load()) tiled_tmem_load;
    decltype(_dummy_storev()) tiled_tmem_storev;
    decltype(_dummy_store()) tiled_tmem_store;
    int thread_idx;

    // Hoisted partitions (Q-split only; K-split triggers nvcc miscompile)
    decltype(_dummy_thr_load().partition_S(_dummy_tStS())) tTMEM_LOADtS;
    decltype(_dummy_thr_storev().partition_D(_dummy_tStS_v())) tTMEM_STOREVtS;
    decltype(_dummy_thr_store().partition_D(_dummy_tStS_P())) tTMEM_STOREtS_x4;

    // Constructor: order matters! compute each region fully before compose
    template <class Stage>
    CUTLASS_DEVICE SoftmaxCtx(Stage stage) {
      thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

      tStS = partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0, 1>(TileShapeQK{}));
      tStS.data() = warp_uniform(uint32_t(stage == _0{} ? TmemAllocation::S0 : TmemAllocation::S1));
      tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tStS);
      tTMEM_LOADtS = tiled_tmem_load.get_slice(thread_idx).partition_S(tStS);

      tStS_v = tStS.compose(make_layout(make_shape(_128{}, Int<kVStatsWidth>{})));
      tStS_v.data() = warp_uniform(uint32_t(stage == _0{} ? TmemAllocation::V0 : TmemAllocation::V1));
      tiled_tmem_storev = make_tmem_copy(TMEM_STORE_V{}, tStS_v);
      tTMEM_STOREVtS = tiled_tmem_storev.get_slice(thread_idx).partition_D(tStS_v);

      tStS_P = tStS.compose(make_layout(make_shape(_128{}, _tilePlikeFP32())));
      tStS_P.data() = warp_uniform(uint32_t(stage == _0{} ? TmemAllocation::P0 : TmemAllocation::P1));
      tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tStS_P);
      tTMEM_STOREtS_x4 = tiled_tmem_store.get_slice(thread_idx).partition_D(tStS_P);
      tTMEM_STOREtS_x4.data() = warp_uniform(tTMEM_STOREtS_x4.data().get());
    }

    template <bool need_apply_mask, bool skip_computation = false, class Stage, class BlkCoord,
              class CountingTensor, class ProblemShape>
    CUTLASS_DEVICE void step(float& row_max, float& row_sum, Stage stage, bool final_call,
                             BlkCoord const& blk_coord, CountingTensor const& cS,
                             Params const& params, ProblemShape const& problem_shape,
                             PipelineS& pipeline_s,
                             typename PipelineS::PipelineState& pipeline_s_consumer_state,
                             PipelineC& pipeline_c,
                             typename PipelineC::PipelineState& pipeline_c_producer_state,
                             float* ms_base, int ms_stride_k,
                             int& k_tile_idx, int k_tile_idx_step
#ifdef FMHA_GMEM_BOUNDS_CHECK
                             , const float* ms_check_base, int ms_check_numel
#endif
                            #ifdef GPU_TRACE_ENABLED
                             , gpu_trace::Recorder& _gt_rec
                            #endif
                             )
    {
      // cS-dependent coordinate partitions (always per-iteration)
      static constexpr auto kLayoutV = make_layout(make_shape(_128{}, Int<kVStatsWidth>{}));
      static constexpr auto kLayoutP = make_layout(make_shape(_128{}, _tilePlikeFP32()));

      Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);
      Tensor tTMEM_LOADcS = tiled_tmem_load.get_slice(thread_idx).partition_D(tScS);
      Tensor tScS_v = tScS.compose(kLayoutV);
      Tensor tTMEM_STOREVcS = tiled_tmem_storev.get_slice(thread_idx).partition_S(tScS_v);
      Tensor tScS_P = tScS.compose(kLayoutP);
      Tensor tTMEM_STOREcS = tiled_tmem_store.get_slice(thread_idx).partition_S(tScS_P);

      // tStS-dependent partitions: hoisted or recomputed based on AggressiveLoopInvariantCodeMotionOfSoftmax
      auto get_LOADtS = [&]() {
        if constexpr (!AggressiveLoopInvariantCodeMotionOfSoftmax) {
          return tiled_tmem_load.get_slice(thread_idx).partition_S(tStS);
        } else { return tTMEM_LOADtS; }
      };
      auto get_STOREVtS = [&]() {
        if constexpr (!AggressiveLoopInvariantCodeMotionOfSoftmax) {
          return tiled_tmem_storev.get_slice(thread_idx).partition_D(tStS_v);
        } else { return tTMEM_STOREVtS; }
      };
      auto get_STOREtS_x4 = [&]() {
        if constexpr (!AggressiveLoopInvariantCodeMotionOfSoftmax) {
          auto t = tiled_tmem_store.get_slice(thread_idx).partition_D(tStS_P);
          t.data() = warp_uniform(t.data().get());
          return t;
        } else { return tTMEM_STOREtS_x4; }
      };
      auto l_tTMEM_LOADtS = get_LOADtS();
      auto l_tTMEM_STOREVtS = get_STOREVtS();
      auto l_tTMEM_STOREtS_x4 = get_STOREtS_x4();

      // read all of S from tmem into reg mem
      Tensor tTMEM_LOADrS = make_tensor<ElementQK>(shape(tTMEM_LOADcS));

      // wait on tensor core pipe
      pipeline_s.consumer_wait(pipeline_s_consumer_state);

      {GPU_TRACE_SCOPE(SOFTMAX_Mask);
      if constexpr (!skip_computation) {
        auto l_ld = l_tTMEM_LOADtS;
        if constexpr (kSDoubleBuf) {
          if (pipeline_s_consumer_state.index() == 1) {
            l_ld.data() = l_ld.data().get() + kSBuf1Offset;
          }
        }
        copy(tiled_tmem_load, l_ld, tTMEM_LOADrS);
        if constexpr (need_apply_mask) {
          Mask{}.apply_mask(tTMEM_LOADrS, tTMEM_LOADcS, problem_shape);
        }
      }}

      ElementQK old_row_max = row_max;
      ElementQK tile_max = -INFINITY;
      {GPU_TRACE_SCOPE(SOFTMAX_GetMax);
      if constexpr (!skip_computation) {
        // Off: start from row_max (fused running max, identical to original code)
        // OnlyScore/Full: start from -inf to get per-tile max separately
        float init_val;
        if constexpr (!kNeedMaxScore) {
          init_val = row_max;
        } else {
          init_val = -INFINITY;
        }
        float tile_max_0 = init_val;
        float tile_max_1 = init_val;
        float tile_max_2 = init_val;
        float tile_max_3 = init_val;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTMEM_LOADrS); i += 4) {
          tile_max_0 = ::fmax(tile_max_0, tTMEM_LOADrS(i));
          tile_max_1 = ::fmax(tile_max_1, tTMEM_LOADrS(i + 1));
          tile_max_2 = ::fmax(tile_max_2, tTMEM_LOADrS(i + 2));
          tile_max_3 = ::fmax(tile_max_3, tTMEM_LOADrS(i + 3));
        }
        tile_max = ::fmax(tile_max_0, tile_max_1);
        tile_max = ::fmax(tile_max, tile_max_2);
        tile_max = ::fmax(tile_max, tile_max_3);
        if constexpr (kNeedMaxScore) {
          row_max = ::fmax(row_max, tile_max);
        } else {
          row_max = tile_max;
        }
      }}

      ElementQK row_max_safe = row_max == -INFINITY ? 0 : row_max;

      // OnlyScore fuse: softmax thread already holds tile_max in its register
      // (matching what correction would have read from V0/V1 TMEM). Write to
      // GMEM directly here; correction will skip its own write (no double
      // write, no TMEM round-trip for the value itself — though the V-channel
      // STSM still happens because correction's loop body unconditionally
      // reads V0/V1 for the rescale path's old/new max math, which is also
      // kept for sync safety in v1).
      if constexpr (kFuseMaxScoreIntoSoftmax) {
        if (ms_base != nullptr) {
#ifdef FMHA_GMEM_BOUNDS_CHECK
          gmem_stwt_checked(ms_base + k_tile_idx * ms_stride_k, tile_max,
                            ms_check_base, ms_check_numel, "max_score_stwt");
#else
          __stwt(ms_base + k_tile_idx * ms_stride_k, tile_max);
#endif
        }
        k_tile_idx += k_tile_idx_step;
      }

      // P2 (v3): OnlyScore fuse fully retires the V-channel TMEM round-trip in
      // the main-loop iterations. The correction warp's only consumer of these
      // STOREV values are (a) write_max_score (now no-op in OnlyScore: GMEM
      // write fused into softmax, max() update unused), and (b) rescale-path
      // exp2(old-new) which is kNeedOutput-gated. Both dead, so skip the store
      // and let DCE remove old_row_max / row_max_safe wiring. final_call's
      // STOREV (kIdxFinalRowMax/Sum) is preserved below for splitKV merge.
      if constexpr (!kFuseMaxScoreIntoSoftmax) {
        Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
        tTMEM_STOREVrS(kIdxOldRowMax) = old_row_max;
        if constexpr (kNeedMaxScore) {
          tTMEM_STOREVrS(kIdxNewRowMax) = tile_max;
        } else {
          tTMEM_STOREVrS(kIdxNewRowMax) = row_max_safe;
        }
        copy(tiled_tmem_storev, tTMEM_STOREVrS, l_tTMEM_STOREVtS);
      }

      if constexpr (!kSkipPerTileCorrection) {
        pipeline_c.producer_commit(pipeline_c_producer_state);
        ++pipeline_c_producer_state;
      }

      Tensor tTMEM_STORErS_x4 = make_tensor<uint32_t>(shape(tTMEM_STOREcS));

      if constexpr (!skip_computation && kNeedOutput) {
        ElementQK scale = params.scale_softmax_log2;
        ElementQK row_max_scale = row_max_safe * scale;

        float2 scale_fp32x2 = make_float2(scale, scale);
        float2 minus_row_max_scale_fp32x2 = make_float2(-row_max_scale, -row_max_scale);

        constexpr int kConversionsPerStep = 2;

        Tensor tTMEM_STORErS_x4_e = recast<Array<Element, kConversionsPerStep>>(tTMEM_STORErS_x4);

        NumericArrayConverter<Element, ElementQK, kConversionsPerStep> convert;

        {
          GPU_TRACE_SCOPE(SOFTMAX_Exp);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tTMEM_LOADrS); i += 2) {
            float2 in = make_float2(tTMEM_LOADrS(i + 0), tTMEM_LOADrS(i + 1));
            float2 out;
            cute::fma(out, scale_fp32x2, in, minus_row_max_scale_fp32x2);
            tTMEM_LOADrS(i + 0) = out.x;
            tTMEM_LOADrS(i + 1) = out.y;

            if constexpr (EnableEmuExp2 && !need_apply_mask) {
              constexpr int frg_cnt = decltype(size(tTMEM_LOADrS))::value / 32;
              if (i >= (kExp2EmuStartFrg * 32) &&
                  (!kExp2EmuSkipLastFrg || i < (32 * (frg_cnt - 1))) &&
                  (i % kExp2EmuFreq) >= (kExp2EmuFreq - kExp2EmuRes)) {
                exp2_emulated_2(tTMEM_LOADrS(i + 0), tTMEM_LOADrS(i + 1));
              } else {
                tTMEM_LOADrS(i + 0) = ::exp2f(tTMEM_LOADrS(i + 0));
                tTMEM_LOADrS(i + 1) = ::exp2f(tTMEM_LOADrS(i + 1));
              }
            } else {
              tTMEM_LOADrS(i + 0) = ::exp2f(tTMEM_LOADrS(i + 0));
              tTMEM_LOADrS(i + 1) = ::exp2f(tTMEM_LOADrS(i + 1));
            }

            Array<ElementQK, kConversionsPerStep> in_conv;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < kConversionsPerStep; j++) {
              in_conv[j] = tTMEM_LOADrS(i + j);
            }
            tTMEM_STORErS_x4_e[i / kConversionsPerStep] = convert(in_conv);

            if constexpr (size<2>(tTMEM_STORErS_x4) == _2{}) {
              if (i == size(tTMEM_LOADrS) - 6) {
                copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, 0), l_tTMEM_STOREtS_x4(_, _, 0));
              }
            }
          }
        }
      } else if constexpr(kNeedOutput) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTMEM_STORErS_x4); i++) {
          tTMEM_STORErS_x4(i) = 0u;
        }
      }

      CUTE_STATIC_ASSERT_V(size<2>(tTMEM_STORErS_x4) <= _2{});
      CUTE_STATIC_ASSERT_V(size<1>(tTMEM_STORErS_x4) == _1{});
      if constexpr (skip_computation && !kSDoubleBuf) {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tTMEM_STORErS_x4); ++k) {
          copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, k), l_tTMEM_STOREtS_x4(_, _, k));
        }
      } else if constexpr(kNeedOutput) {
        copy(tiled_tmem_store, tTMEM_STORErS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1),
            l_tTMEM_STOREtS_x4(_, _, size<2>(tTMEM_STORErS_x4) - 1));
      }

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_s.consumer_release(pipeline_s_consumer_state);
      ++pipeline_s_consumer_state;

      if constexpr (!kSkipPerTileCorrection) {
        pipeline_c.producer_acquire(pipeline_c_producer_state);
      }

      {GPU_TRACE_SCOPE(SOFTMAX_Sum);
      if constexpr (!skip_computation && kNeedOutput) {
        ElementQK scale = params.scale_softmax_log2;
        ElementQK acc_scale = 0.5f * ::exp2f(scale * (old_row_max - row_max_safe));
        row_sum *= acc_scale;
        float2 local_row_sum_f32x2 = make_float2(row_sum, row_sum);
        float2 local_row_sum_1 = make_float2(0, 0);
        float2 local_row_sum_2 = make_float2(0, 0);
        float2 local_row_sum_3 = make_float2(0, 0);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tTMEM_LOADrS); i += 8) {
          float2 in = make_float2(tTMEM_LOADrS(i), tTMEM_LOADrS(i + 1));
          cute::add(local_row_sum_f32x2, local_row_sum_f32x2, in);

          in = make_float2(tTMEM_LOADrS(i + 2), tTMEM_LOADrS(i + 2 + 1));
          cute::add(local_row_sum_1, local_row_sum_1, in);

          in = make_float2(tTMEM_LOADrS(i + 4), tTMEM_LOADrS(i + 4 + 1));
          cute::add(local_row_sum_2, local_row_sum_2, in);

          in = make_float2(tTMEM_LOADrS(i + 6), tTMEM_LOADrS(i + 6 + 1));
          cute::add(local_row_sum_3, local_row_sum_3, in);
        }

        cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_1);
        cute::add(local_row_sum_2, local_row_sum_2, local_row_sum_3);
        cute::add(local_row_sum_f32x2, local_row_sum_f32x2, local_row_sum_2);
        float local_row_sum = local_row_sum_f32x2.x + local_row_sum_f32x2.y;

        row_sum = local_row_sum;
      }

      if (final_call) {
        pipeline_s.consumer_wait(pipeline_s_consumer_state);

        Tensor tTMEM_STOREVrS = make_tensor<ElementQK>(shape(tTMEM_STOREVcS));
        if constexpr (!skip_computation && kNeedOutput) {
          tTMEM_STOREVrS(kIdxFinalRowMax) = row_max;
          tTMEM_STOREVrS(kIdxFinalRowSum) = row_sum;
        } else {
          tTMEM_STOREVrS(kIdxFinalRowMax) = 0;
          tTMEM_STOREVrS(kIdxFinalRowSum) = 0;
        }
        if constexpr (!kSDoubleBuf) {
          copy(tiled_tmem_storev, tTMEM_STOREVrS, l_tTMEM_STOREVtS);
        }
      }}
    }  // end step()
  };

  template <bool kSingleWG = false, class Stage, class BlkCoord, class ProblemShape,
            class ParamsProblemShape, class CollectiveEpilogue>
  CUTLASS_DEVICE auto softmax(Stage stage, BlkCoord const& blk_coord, Params const& params,
                              ParamsProblemShape const& params_problem_shape,
                              ProblemShape const& problem_shape,
                              CollectiveEpilogue const& epilogue,
                              PipelineS& pipeline_s,
                              typename PipelineS::PipelineState& pipeline_s_consumer_state,
                              PipelineC& pipeline_c,
                              typename PipelineC::PipelineState& pipeline_c_producer_state,
                              int kv_tile_begin = 0, int kv_tile_end = INT_MAX,
                              int q_offset_correction = 0) {
    GET_GPU_TRACE(TRACE_SOFTMAX && (kSingleWG ? (cutlass::canonical_warp_idx_sync() % 2 == 0) : (cutlass::canonical_warp_idx_sync() % 4 == 0)));

    int full_trip_count = get_full_trip_count(blk_coord, problem_shape, params.load.kv_block_num);

    int total_trip_count;
    int sub_q_offset = int(stage % get<0>(ThreadShape{})) * int(get<0>(TileShapeQK{}));
    int sub_q_size = int(get<0>(TileShapeQK{}));
    int skip_count, real_masked_count, unmasked_count;
    int effective_end;

    int64_t kbi_off_s = 0;
    if constexpr (kNeedSparse) {
      int h_r_s = get<3, 0, 0>(problem_shape);
      int kv_head_idx_s = get<2, 0>(blk_coord) / h_r_s;
      int batch_idx_s = get<2, 1>(blk_coord);
      kbi_off_s = batch_idx_s * params.load.kv_block_stride_n
                + kv_head_idx_s * params.load.kv_block_stride_h;
    }

    if constexpr (kNeedSparse) {
      constexpr int full_tile_kv = get<1>(TileShape{});
      if constexpr (IsSplitKV) {
        effective_end = full_trip_count < kv_tile_end ? full_trip_count : kv_tile_end;
        total_trip_count = effective_end - kv_tile_begin;
        if (total_trip_count <= 0) total_trip_count = 0;
      } else {
        total_trip_count = full_trip_count;
        effective_end = total_trip_count;
      }

      // Count padding blocks (backward scan for -1)
      int valid_blks = params.load.kv_block_num;
      while (valid_blks > 0 && (UnitSelectorColumn
          ? KV_BLOCK_INDEX_LOAD(params.load.kv_block_indexes,
                kbi_off_s + valid_blks - 1, params.load.kv_block_indexes_span)
          : KV_BLOCK_INDEX_LOAD(params.load.kv_block_indexes,
                kbi_off_s + (valid_blks - 1) * params.load.kv_block_stride_b,
                params.load.kv_block_indexes_span)) < 0)
        valid_blks--;

      int valid_tiles = (valid_blks * KVPageSize + full_tile_kv - 1) / full_tile_kv;
      if constexpr (IsSplitKV) {
        skip_count = max(0, effective_end - max(valid_tiles, kv_tile_begin));
        real_masked_count = max(0, min(valid_tiles, effective_end) - kv_tile_begin);
      } else {
        skip_count = total_trip_count - valid_tiles;
        real_masked_count = valid_tiles;
      }
      unmasked_count = 0;
    } else if constexpr (IsSplitKV) {
      effective_end = full_trip_count < kv_tile_end ? full_trip_count : kv_tile_end;
      int local_trip = effective_end - kv_tile_begin;
      if (local_trip <= 0) local_trip = 0;
      total_trip_count = local_trip;

      int full_per_warp = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape,
                                                sub_q_offset, sub_q_size);
      int full_masked = Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape,
                                                      sub_q_offset, sub_q_size);
      int unmasked_boundary = full_trip_count - full_masked;

      skip_count     = max(0, min(effective_end, full_trip_count) - max(kv_tile_begin, full_per_warp));
      real_masked_count = max(0, min(effective_end, full_per_warp) - max(kv_tile_begin, unmasked_boundary));
      unmasked_count = max(0, min(effective_end, unmasked_boundary) - kv_tile_begin);
    } else {
      total_trip_count = full_trip_count;
      effective_end = full_trip_count;

      int per_warp_trip = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape,
                                                sub_q_offset, sub_q_size);
      int masked_count = Mask{}.get_masked_trip_count(blk_coord, TileShape{}, problem_shape,
                                                      sub_q_offset, sub_q_size);
      skip_count = total_trip_count - per_warp_trip;
      real_masked_count = masked_count - skip_count;
      unmasked_count = total_trip_count - masked_count;
    }

    ElementQK row_max = -INFINITY;
    ElementQK row_sum = 0;

    // Reverse iteration: start cS at the highest KV tile position in our range
    Tensor cS_base = make_identity_tensor(select<0, 1>(TileShapeQK{}));
    auto logical_offset = make_coord(get<0>(blk_coord) * get<0>(TileShape{}) +
                                         (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{})
                                         + (kSingleWG ? q_offset_correction : 0),
                                     (effective_end - 1) * get<1>(ThreadShape{}) * get<1>(TileShapeQK{})
                                         + (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{}));
    Tensor cS = domain_offset(logical_offset, cS_base);

    int sparse_tile_counter = 0;
    // Returns true if tile is fully unmasked (sorted blocks → all remaining are too)
    auto sparse_update_cS = [&]() -> bool {
      if constexpr (kNeedSparse) {
        int tile_from_end = effective_end - 1 - sparse_tile_counter;
        int q_s = get<0>(blk_coord) * get<0>(TileShape{})
                + (stage % get<0>(ThreadShape{})) * get<0>(TileShapeQK{})
                + (kSingleWG ? q_offset_correction : 0);
        int kv_sub = (stage % get<1>(ThreadShape{})) * get<1>(TileShapeQK{});
        int sparse_linear_pos = tile_from_end * int(get<1>(TileShape{})) + kv_sub;
        uint page_idx = uint(sparse_linear_pos / KVPageSize);
        int offset_in_page = sparse_linear_pos % KVPageSize;
        sparse_tile_counter++;
        if (page_idx >= params.load.kv_block_num) {
          cS = domain_offset(make_coord(q_s, INT_MAX / 2), cS_base);
          return false;
        }
        int pos = UnitSelectorColumn
            ? KV_BLOCK_INDEX_LOAD(params.load.kv_block_indexes,
                  kbi_off_s + page_idx, params.load.kv_block_indexes_span)
            : KV_BLOCK_INDEX_LOAD(params.load.kv_block_indexes,
                  kbi_off_s + page_idx * params.load.kv_block_stride_b,
                  params.load.kv_block_indexes_span);
        if (pos < 0) {
          cS = domain_offset(make_coord(q_s, INT_MAX / 2), cS_base);
          return false;
        }
        int kv_coord = pos * KVPageSize + offset_in_page;
        cS = domain_offset(make_coord(q_s, kv_coord), cS_base);
        int causal_bound = Mask{}.sparse_causal_bound(q_s, problem_shape);
        constexpr int sub_tile_kv = get<1>(TileShapeQK{});
        return kv_coord + sub_tile_kv <= causal_bound + 1;
      }
      return false;
    };

    SoftmaxCtx softmax_ctx(stage);

    // OnlyScore fuse v1: precompute per-thread GMEM max_score addr.
    // Same row→thread mapping as correction (32dp32b2x partition is
    // symmetric for store and load), so softmax thread t writes the row
    // correction thread t would have written. v1 keeps all TMEM and
    // pipeline_c sync intact; correction only skips its own GMEM
    // write_max_score under kFuseMaxScoreIntoSoftmax.
    float* ms_base = nullptr;
    int ms_stride_k = 0;
    int k_tile_idx = 0;
    static constexpr int kTilesPerMacro =
        (get<1>(ThreadShape{}) > 1) ? 2 : 1;
    int k_tile_idx_step = -kTilesPerMacro;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    const float* ms_check_base = nullptr;
    int ms_check_numel = 0;
#endif
    if constexpr (kFuseMaxScoreIntoSoftmax) {
      int thread_idx_local = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);
      Tensor cS_for_addr = make_identity_tensor(select<0, 1>(TileShapeQK{}));
      Tensor tScS_for_addr =
          typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS_for_addr);
      Tensor tScS_v_for_addr =
          tScS_for_addr.compose(make_layout(make_shape(_128{}, Int<kVStatsWidth>{})));
      Tensor tTMEM_STOREVcS_addr =
          softmax_ctx.tiled_tmem_storev.get_slice(thread_idx_local)
                                       .partition_S(tScS_v_for_addr);
      int local_row = get<0>(tTMEM_STOREVcS_addr(_0{}));
      int abs_row = get<0>(blk_coord) * int(get<0>(TileShape{}))
                  + (int(stage) % int(get<0>(ThreadShape{}))) * int(get<0>(TileShapeQK{}))
                  + (kSingleWG ? q_offset_correction : 0)
                  + local_row;

      uint qo_len = uint(get<0>(problem_shape));
      if (uint(abs_row) < qo_len) {
        int qo_head_idx = get<2, 0>(blk_coord);
        int batch_idx = get<2, 1>(blk_coord);
        int _seg_off_sz = get<3, 1>(params_problem_shape) + 1;
        int segment_offset = SEG_OFF_LOAD(get<0>(params_problem_shape), batch_idx, _seg_off_sz);

        static constexpr int kPackFactor = pack_factor_of<Mask>::value;
        auto packed_maxscore_path = [&]() {
          Tensor gMaxScore = make_tensor(
              make_gmem_ptr(epilogue.params.ptr_MaxScore),
              epilogue.params.layout_MaxScore);
          ms_stride_k = int(&gMaxScore(0, 0, 1) - &gMaxScore(0, 0, 0));
          ms_base = &gMaxScore(segment_offset + abs_row, qo_head_idx, 0);
        };

        if constexpr (kPackFactor > 1) {
          if (epilogue.params.ptr_MaxScore_direct != nullptr) {
            int rem_hr = get<3, 0, 0>(params_problem_shape);
            int h_r_orig = rem_hr * kPackFactor;
            int kv_head = qo_head_idx / rem_hr;
            int rem_hr_idx = qo_head_idx % rem_hr;
            int max_k = cute::size<2>(epilogue.params.layout_MaxScore);
            int tql_orig = epilogue.params.total_qo_len_orig;
            int seg_off_orig = segment_offset / kPackFactor;
            ms_stride_k = tql_orig;
            int actual_tok = abs_row / kPackFactor;
            int pf_idx = abs_row % kPackFactor;
            int unpacked_head = kv_head * h_r_orig + rem_hr_idx * kPackFactor + pf_idx;
            ms_base = epilogue.params.ptr_MaxScore_direct
                    + unpacked_head * (max_k * tql_orig)
                    + seg_off_orig + actual_tok;
          } else {
            packed_maxscore_path();
          }
        } else {
          packed_maxscore_path();
        }
      }

#ifdef FMHA_GMEM_BOUNDS_CHECK
      ms_check_base = epilogue.params.ptr_MaxScore_direct
                        ? epilogue.params.ptr_MaxScore_direct
                        : epilogue.params.ptr_MaxScore;
      ms_check_numel = epilogue.params.max_score_numel;
#endif

      int stage_k_off = int(stage) % int(get<1>(ThreadShape{}));
      k_tile_idx = (effective_end - 1) * kTilesPerMacro + stage_k_off;
    }

    pipeline_c.producer_acquire(pipeline_c_producer_state);

    sparse_update_cS();

    // Skip fully-masked tiles (padding in sparse mode, or above causal boundary)
    CUTLASS_PRAGMA_NO_UNROLL
    for (int i = 0; i < skip_count; i++) {
      bool is_last_step = (i == skip_count - 1) && (real_masked_count == 0) && (unmasked_count == 0);
      softmax_ctx.template step<true, true /* skip_computation */>(
          row_max, row_sum, stage, is_last_step,
          blk_coord, cS, params, problem_shape, pipeline_s, pipeline_s_consumer_state, pipeline_c,
          pipeline_c_producer_state, ms_base, ms_stride_k, k_tile_idx, k_tile_idx_step
#ifdef FMHA_GMEM_BOUNDS_CHECK
          , ms_check_base, ms_check_numel
#endif
        #ifdef GPU_TRACE_ENABLED
          , _gt_rec
        #endif
        );

      if constexpr (!kNeedSparse) {
        cS.data() = cS.data() + E<1>{} * (-(int)(get<1>(ThreadShape{}) * get<1>(TileShapeQK{})));
      }
      else {
        sparse_update_cS();
      }
    }

    if constexpr (kNeedSparse) {
      // Masked loop: step<true> for skip-above + near-diagonal tiles.
      // Blocks are sorted so once we hit cls==1 (unmasked), all remaining are unmasked.
      int valid_remaining = real_masked_count;
      CUTLASS_PRAGMA_NO_UNROLL
      for (; valid_remaining > 0; ) {
        softmax_ctx.template step<true /* masked */>(
            row_max, row_sum, stage,
            (valid_remaining == 1),
            blk_coord, cS, params, problem_shape, pipeline_s, pipeline_s_consumer_state, pipeline_c,
            pipeline_c_producer_state, ms_base, ms_stride_k, k_tile_idx, k_tile_idx_step
#ifdef FMHA_GMEM_BOUNDS_CHECK
          , ms_check_base, ms_check_numel
#endif
          #ifdef GPU_TRACE_ENABLED
            , _gt_rec
          #endif
          );
          valid_remaining--;
          if (sparse_update_cS()) break;
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (; valid_remaining > 0; valid_remaining--) {
        softmax_ctx.template step<false /* unmasked */>(
            row_max, row_sum, stage,
            (valid_remaining == 1),
            blk_coord, cS, params, problem_shape, pipeline_s, pipeline_s_consumer_state, pipeline_c,
            pipeline_c_producer_state, ms_base, ms_stride_k, k_tile_idx, k_tile_idx_step
#ifdef FMHA_GMEM_BOUNDS_CHECK
          , ms_check_base, ms_check_numel
#endif
          #ifdef GPU_TRACE_ENABLED
            , _gt_rec
          #endif
          );
      }
    } else {
      // Masked iterations (near causal diagonal, need element-wise masking)
      int mask_tile_count = real_masked_count;
      CUTLASS_PRAGMA_NO_UNROLL
      for (; mask_tile_count > 0; mask_tile_count -= 1) {
        softmax_ctx.template step<true /* need_apply_mask */>(
            row_max, row_sum, stage,
            (mask_tile_count == 1) && (unmasked_count == 0),
            blk_coord, cS, params, problem_shape, pipeline_s, pipeline_s_consumer_state, pipeline_c,
            pipeline_c_producer_state, ms_base, ms_stride_k, k_tile_idx, k_tile_idx_step
#ifdef FMHA_GMEM_BOUNDS_CHECK
          , ms_check_base, ms_check_numel
#endif
          #ifdef GPU_TRACE_ENABLED
            , _gt_rec
          #endif
          );

        cS.data() = cS.data() + E<1>{} * (-(int)(get<1>(ThreadShape{}) * get<1>(TileShapeQK{})));
      }

      // Unmasked iterations (low KV index, far from diagonal)
      mask_tile_count = unmasked_count;
      CUTLASS_PRAGMA_NO_UNROLL
      for (; mask_tile_count > 0; mask_tile_count -= 1) {
        softmax_ctx.template step<false /* need_apply_mask */>(
            row_max, row_sum, stage,
            mask_tile_count == 1,
            blk_coord, cS, params, problem_shape, pipeline_s, pipeline_s_consumer_state, pipeline_c,
            pipeline_c_producer_state, ms_base, ms_stride_k, k_tile_idx, k_tile_idx_step
#ifdef FMHA_GMEM_BOUNDS_CHECK
          , ms_check_base, ms_check_numel
#endif
          #ifdef GPU_TRACE_ENABLED
            , _gt_rec
          #endif
          );

        cS.data() = cS.data() + E<1>{} * (-(int)(get<1>(ThreadShape{}) * get<1>(TileShapeQK{})));
      }
    }

    pipeline_c.producer_commit(pipeline_c_producer_state);
    ++pipeline_c_producer_state;

    pipeline_c.producer_acquire(pipeline_c_producer_state);
    // empty step to sync against pipe s
    pipeline_s.consumer_release(pipeline_s_consumer_state);
    ++pipeline_s_consumer_state;
    RELEASE_GPU_TRACE;
  }

  template <class Stage, class TensorO>
  CUTLASS_DEVICE auto correction_epilogue(float scale, Stage stage, TensorO const& sO_01) {
    static_assert(kNeedOutput, "O TMEM columns are S buffer 1 in OnlyScore");
    using ElementOut = typename TensorO::value_type;

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor sO = sO_01(_, _, stage);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    const int kCorrectionTileSize = 32 / sizeof(ElementOut);

    using TMEM_LOAD =
        std::conditional_t<kCorrectionTileSize == 32, SM100_TMEM_LOAD_32dp32b32x,
                           SM100_TMEM_LOAD_32dp32b16x>;  // 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO = make_identity_tensor(select<0, 1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0, 1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);
    Tensor tOsO = mma.get_slice(0).partition_C(sO);

    Tensor tOtO_i =
        logical_divide(tOtO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i =
        logical_divide(tOcO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOsO_i =
        logical_divide(tOsO, make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    if constexpr (decltype(stage == _0{})::value) {
      tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAllocation::O0);
    } else {
      static_assert(decltype(stage == _1{})::value, "stage is either 0 or 1");
      tOtO_i.data() = tOtO_i.data().get() + uint32_t(TmemAllocation::O1);
    }

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i(make_coord(_, _), _0{}));
    auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);

    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i(make_coord(_, _), _));
    Tensor tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i(make_coord(_, _), _));

    float2 scale_f32x2 = make_float2(scale, scale);

    // loop:
    //   TMEM_LOAD, FMUL2 scale, TMEM_STORE
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < get<1>(TileShapePV{}) / kCorrectionTileSize; i++) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO(_, _0{}, _0{}, i);
      Tensor tTMEM_LOADsO_i = tTMEM_LOADsO(_, _0{}, _0{}, i);

      Tensor tTMrO = make_tensor<ElementPV>(shape(tTMEM_LOADcO(_, _0{}, _0{}, i)));

      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO);

#ifndef ONLY_SOFTMAX
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO); j += 2) {
        float2 in = make_float2(tTMrO(j), tTMrO(j + 1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO(j) = out.x;
        tTMrO(j + 1) = out.y;
      }
#endif

      constexpr int N = 4 / sizeof(ElementOut);
      NumericArrayConverter<ElementOut, ElementPV, N> convert;

      Tensor tSMrO = make_tensor_like<ElementOut>(tTMrO);

      Tensor tCs = recast<decltype(convert)::source_type>(tTMrO);
      Tensor tCd = recast<decltype(convert)::result_type>(tSMrO);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tCs); j++) {
        tCd(j) = convert.convert(tCs(j));
      }

      Tensor tSMsO_i = recast<uint32_t>(tTMEM_LOADsO_i);
      Tensor tSMrO_i = recast<uint32_t>(tSMrO);

      copy(AutoVectorizingCopyWithAssumedAlignment<128>{}, tSMrO_i, tSMsO_i);
    }

    cutlass::arch::fence_view_async_shared();
  }

  CUTLASS_DEVICE auto correction_rescale(float scale, uint32_t tmem_O) {
    static_assert(kNeedOutput, "O TMEM columns are S buffer 1 in OnlyScore");
    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    // As opposed to the softmax, we do not have enough registers here
    // to load all of the values (for tile kv = 128), so we loop
    // good values would be either 32 or 64
    const int kCorrectionTileSize = 16;

    using TMEM_LOAD = SM100_TMEM_LOAD_32dp32b16x;    // 4x32 threads with 64 cols of 32b elem
    using TMEM_STORE = SM100_TMEM_STORE_32dp32b16x;  // 4x32 threads with 64 cols of 32b elem

    typename CollectiveMmaPV::TiledMma mma;
    Tensor cO = make_identity_tensor(select<0, 1>(TileShapePV{}));
    Tensor tOtO = partition_fragment_C(mma, select<0, 1>(TileShapePV{}));
    Tensor tOcO = mma.get_slice(0).partition_C(cO);

    Tensor tOtO_i = tOtO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));
    Tensor tOcO_i = tOcO.compose(make_layout(make_shape(_128{}, Int<kCorrectionTileSize>{})));

    tOtO_i.data() = tOtO_i.data().get() + tmem_O;

    auto tiled_tmem_load = make_tmem_copy(TMEM_LOAD{}, tOtO_i);
    auto thr_tmem_load = tiled_tmem_load.get_slice(thread_idx);
    auto tiled_tmem_store = make_tmem_copy(TMEM_STORE{}, tOtO_i);
    auto thr_tmem_store = tiled_tmem_store.get_slice(thread_idx);

    Tensor tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i);
    Tensor tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i);
    Tensor tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i);
    Tensor tTMEM_STOREcO = thr_tmem_store.partition_S(tOcO_i);
    static_assert(shape(tTMEM_STOREcO) == shape(tTMEM_LOADcO));

    float2 scale_f32x2 = make_float2(scale, scale);

    Tensor tTMrO = make_tensor<ElementPV>(
        make_shape(shape(tTMEM_LOADcO), Int<get<1>(TileShapePV{}) / kCorrectionTileSize>{}));

    auto copy_in = [&](int i) {
      Tensor tTMEM_LOADtO_i = tTMEM_LOADtO;
      tTMEM_LOADtO_i.data() = tTMEM_LOADtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i);
    };

    auto copy_out = [&](int i) {
      Tensor tTMEM_STOREtO_i = tTMEM_STOREtO;
      tTMEM_STOREtO_i.data() = tTMEM_STOREtO_i.data().get() + uint32_t(i * kCorrectionTileSize);
      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i);
    };

    // sequence: LLMSLMSLMSS

    // loop:
    //   TMEM_LOAD, FMUL2 scale, TMEM_STORE
    copy_in(0);

    int count = get<1>(TileShapePV{}) / kCorrectionTileSize;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < count; i++) {
      if (i != count - 1) {
        copy_in(i + 1);
      }

      Tensor tTMrO_i = tTMrO(_, i).compose(make_layout(shape<0>(tTMrO)));
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size(tTMrO_i); j += 2) {
        float2 in = make_float2(tTMrO_i(j), tTMrO_i(j + 1));
        float2 out;
        cute::mul(out, scale_f32x2, in);
        tTMrO_i(j) = out.x;
        tTMrO_i(j + 1) = out.y;
      }

      copy_out(i);
    }
  }


  // K-split: rescale O0/O1 by online softmax factors and accumulate O1 into O0 in TMEM.
  // Returns total_sum for the final epilogue scaling.
  template <bool kSingleWG = false, class TensorV>
  CUTLASS_DEVICE auto correction_merge_ksplit(
      Params const& params, TensorV const& tTMEM_LOADVrS0, TensorV const& tTMEM_LOADVrS1,
      float* smem_scratch = nullptr) {
    static_assert(kNeedOutput, "O TMEM columns are S buffer 1 in OnlyScore");
    ElementPV row_max_0 = tTMEM_LOADVrS0(kIdxFinalRowMax);
    ElementPV row_sum_0 = tTMEM_LOADVrS0(kIdxFinalRowSum);
    ElementPV row_max_1 = tTMEM_LOADVrS1(kIdxFinalRowMax);
    ElementPV row_sum_1 = tTMEM_LOADVrS1(kIdxFinalRowSum);

    // Phase 2: exchange V stats between partner threads via SMEM
    // Thread k has V0[k], thread k+64 has V1[k+64]. Each needs the partner's stats.
    if constexpr (kSingleWG) {
      int tidx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);
      bool is_lower = (tidx < 64);
      int partner = is_lower ? tidx : (tidx - 64);
      // Correction warpgroup barrier (not CTA-wide — softmax warps may have exited)
      constexpr int kCorrBarrierId = 8;
      constexpr int kCorrThreads = 4 * cutlass::NumThreadsPerWarp;
      auto corr_sync = [&]() { cutlass::arch::NamedBarrier::sync(kCorrThreads, kCorrBarrierId); };
      // smem_scratch layout: [0..63] lower's V0 max, [64..127] lower's V0 sum,
      //                      [128..191] upper's V1 max, [192..255] upper's V1 sum
      if (is_lower) {
        smem_scratch[partner] = row_max_0;
        smem_scratch[64 + partner] = row_sum_0;
      } else {
        smem_scratch[128 + partner] = row_max_1;
        smem_scratch[192 + partner] = row_sum_1;
      }
      corr_sync();
      // Each half reads the partner's stats from the OTHER region
      if (is_lower) {
        row_max_1 = smem_scratch[128 + tidx];
        row_sum_1 = smem_scratch[192 + tidx];
      } else {
        row_max_0 = smem_scratch[tidx - 64];
        row_sum_0 = smem_scratch[64 + tidx - 64];
      }
      corr_sync();
    }

    ElementPV total_max = ::fmaxf(row_max_0, row_max_1);
    // Guard against -inf - (-inf) = NaN when all positions are masked
    ElementPV rescale_0 = (total_max == -INFINITY) ? 0.f
        : ::exp2f(params.scale_softmax_log2 * (row_max_0 - total_max));
    ElementPV rescale_1 = (total_max == -INFINITY) ? 0.f
        : ::exp2f(params.scale_softmax_log2 * (row_max_1 - total_max));
    ElementPV total_sum = row_sum_0 * rescale_0 + row_sum_1 * rescale_1;

    // Fused rescale + accumulate: O0 = O0 * rescale_0 + O1 * rescale_1
    {
      int thread_idx_corr = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);
      constexpr int kAccTile = 16;
      using TMEM_LOAD_ACC = SM100_TMEM_LOAD_32dp32b16x;
      using TMEM_STORE_ACC = SM100_TMEM_STORE_32dp32b16x;
      typename CollectiveMmaPV::TiledMma mma_acc;
      Tensor tOtO_base = partition_fragment_C(mma_acc, select<0, 1>(TileShapePV{}));
      Tensor tOtO_tile = tOtO_base.compose(make_layout(make_shape(_128{}, Int<kAccTile>{})));

      Tensor tOtO0_tile = tOtO_tile;
      tOtO0_tile.data() = tOtO0_tile.data().get() + uint32_t(TmemAllocation::O0);
      Tensor tOtO1_tile = tOtO_tile;
      tOtO1_tile.data() = tOtO1_tile.data().get() + uint32_t(TmemAllocation::O1);

      auto tmem_load_acc = make_tmem_copy(TMEM_LOAD_ACC{}, tOtO0_tile);
      auto tmem_store_acc = make_tmem_copy(TMEM_STORE_ACC{}, tOtO0_tile);
      auto thr_load = tmem_load_acc.get_slice(thread_idx_corr);
      auto thr_store = tmem_store_acc.get_slice(thread_idx_corr);

      Tensor tSrcO0 = thr_load.partition_S(tOtO0_tile);
      Tensor tSrcO1 = thr_load.partition_S(tOtO1_tile);
      Tensor tDstO0 = thr_store.partition_D(tOtO0_tile);
      Tensor cO_tile = make_identity_tensor(make_shape(_128{}, Int<kAccTile>{}));

      constexpr int n_acc_tiles = get<1>(TileShapePV{}) / kAccTile;

      if constexpr (!kSingleWG) {
        float2 rescale_0_f2 = make_float2(rescale_0, rescale_0);
        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < n_acc_tiles; t++) {
          Tensor src0 = tSrcO0; src0.data() = src0.data().get() + uint32_t(t * kAccTile);
          Tensor src1 = tSrcO1; src1.data() = src1.data().get() + uint32_t(t * kAccTile);
          Tensor dst0 = tDstO0; dst0.data() = dst0.data().get() + uint32_t(t * kAccTile);

          auto rO0 = make_tensor<ElementPV>(shape(thr_load.partition_D(cO_tile)));
          auto rO1 = make_tensor<ElementPV>(shape(rO0));
          copy(tmem_load_acc, src0, rO0);
          copy(tmem_load_acc, src1, rO1);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size(rO0); j += 2) {
            float2 v0 = make_float2(rO0(j), rO0(j + 1));
            float2 v1 = make_float2(rO1(j), rO1(j + 1));
            float2 out;
            cute::mul(out, rescale_0_f2, v0);
            out.x = __fmaf_rn(v1.x, rescale_1, out.x);
            out.y = __fmaf_rn(v1.y, rescale_1, out.y);
            rO0(j) = out.x;
            rO0(j + 1) = out.y;
          }
          copy(tmem_store_acc, rO0, dst0);
        }
      } else {
        bool is_lower = (thread_idx_corr < 64);
        int partner = is_lower ? thread_idx_corr : (thread_idx_corr - 64);
        float* smem_o_xchg = smem_scratch + 256;
        constexpr int kCorrBarrierId2 = 9;
        constexpr int kCorrThreads = 4 * cutlass::NumThreadsPerWarp;
        auto corr_sync = [&]() { cutlass::arch::NamedBarrier::sync(kCorrThreads, kCorrBarrierId2); };

        CUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < n_acc_tiles; t++) {
          auto rO = make_tensor<ElementPV>(shape(thr_load.partition_D(cO_tile)));
          int n_elem = size(rO);

          if (!is_lower) {
            Tensor src1 = tSrcO1; src1.data() = src1.data().get() + uint32_t(t * kAccTile);
            copy(tmem_load_acc, src1, rO);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < n_elem; j++) {
              smem_o_xchg[partner * n_elem + j] = rO(j) * rescale_1;
            }
          }
          corr_sync();

          if (is_lower) {
            Tensor src0 = tSrcO0; src0.data() = src0.data().get() + uint32_t(t * kAccTile);
            Tensor dst0 = tDstO0; dst0.data() = dst0.data().get() + uint32_t(t * kAccTile);
            copy(tmem_load_acc, src0, rO);
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < n_elem; j++) {
              rO(j) = __fmaf_rn(rO(j), rescale_0, smem_o_xchg[thread_idx_corr * n_elem + j]);
            }
            copy(tmem_store_acc, rO, dst0);
          }
          corr_sync();
        }
      }

      cutlass::arch::fence_view_async_tmem_store();
    }

    struct MergeResult { ElementPV total_sum; ElementPV total_max; };
    return MergeResult{total_sum, total_max};
  }

  template <bool kSingleWG = false, class BlkCoord, class ParamsProblemShape, class ProblemShape, class TensorStorageEpi,
            class CollectiveEpilogue>
  CUTLASS_DEVICE auto correction(
      BlkCoord const& blk_coord, Params const& params,
      ParamsProblemShape const& params_problem_shape, ProblemShape const& problem_shape,
      TensorStorageEpi& shared_storage_epi, PipelineC& pipeline_s0_c,
      typename PipelineC::PipelineState& pipeline_s0_c_consumer_state, PipelineC& pipeline_s1_c,
      typename PipelineC::PipelineState& pipeline_s1_c_consumer_state, PipelineO& pipeline_corr,
      PipelineE& pipeline_epi,
      typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue,
      int kv_tile_begin = 0, int kv_tile_end = INT_MAX,
      int kv_split_idx = 0,
      SplitKVParams split_kv = {}) {

    GET_GPU_TRACE(TRACE_CORR && (cutlass::canonical_warp_idx_sync() % 4 == 0));

    int mask_tile_count = get_effective_trip_count(blk_coord, problem_shape,
                                                   kv_tile_begin, kv_tile_end, params.load.kv_block_num);
    int effective_end = kv_tile_begin + mask_tile_count;

    int thread_idx = threadIdx.x % (4 * cutlass::NumThreadsPerWarp);

    Tensor tStS =
        partition_fragment_C(typename CollectiveMmaQK::TiledMma{}, select<0, 1>(TileShapeQK{}));

    Tensor cS = make_identity_tensor(select<0, 1>(TileShapeQK{}));
    Tensor tScS = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_C(cS);

    Tensor tStS_v = tStS.compose(make_layout(make_shape(_128{}, Int<kVStatsWidth>{})));
    Tensor tScS_v = tScS.compose(make_layout(make_shape(_128{}, Int<kVStatsWidth>{})));

    using TMEM_LOAD_V = SM100_TMEM_LOAD_32dp32b2x;

    auto tiled_tmem_loadv = make_tmem_copy(TMEM_LOAD_V{}, tStS_v);
    auto thr_tmem_loadv = tiled_tmem_loadv.get_slice(thread_idx);

    Tensor tTMEM_LOADVtS = thr_tmem_loadv.partition_S(tStS_v);
    Tensor tTMEM_LOADVcS = thr_tmem_loadv.partition_D(tScS_v);

    Tensor tTMEM_LOADVtS0 = tTMEM_LOADVtS;
    tTMEM_LOADVtS0.data() = tTMEM_LOADVtS0.data().get() + uint32_t(TmemAllocation::V0);
    Tensor tTMEM_LOADVtS1 = tTMEM_LOADVtS;
    tTMEM_LOADVtS1.data() = tTMEM_LOADVtS1.data().get() + uint32_t(TmemAllocation::V1);

    // Determine if this correction warp handles all-padding Q rows
    int corr_warp_in_wg = thread_idx / cutlass::NumThreadsPerWarp;
    int corr_first_row = corr_warp_in_wg * 32;
    if constexpr (kSingleWG) {
      corr_first_row = corr_first_row % 64;  // rows 64-127 are duplicates of 0-63
    }
    bool is_padding_corr = (corr_first_row >= int(get<0>(problem_shape)));
    static constexpr bool kIsKSplit = get<1>(ThreadShape{}) > 1;
    static constexpr int kS1RowOffset = kIsKSplit ? (kSingleWG ? -64 : 0) : int(get<0>(TileShapeQK{}));

    // K tiles are iterated in REVERSE order (last to first) for causal.
    static constexpr int kTilesPerMacro = kIsKSplit ? 2 : 1;
    int k_tile_idx;

    float* ms_base_s0 = nullptr;
    float* ms_base_s1 = nullptr;
    int ms_stride_k = 0;
    if constexpr (kNeedMaxScore) {
      k_tile_idx = (effective_end - 1) * kTilesPerMacro;
      int qo_head_idx = get<2, 0>(blk_coord);
      int batch_idx = get<2, 1>(blk_coord);
      uint qo_len = uint(get<0>(problem_shape));
      int segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];
      int row_s0 = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * get<0>(blk_coord);
      int row_s1 = row_s0 + kS1RowOffset;

      static constexpr int kPackFactor = pack_factor_of<Mask>::value;

      auto packed_maxscore_path = [&]() {
        Tensor gMaxScore = make_tensor(
            make_gmem_ptr(epilogue.params.ptr_MaxScore),
            epilogue.params.layout_MaxScore);
        ms_stride_k = int(&gMaxScore(0, 0, 1) - &gMaxScore(0, 0, 0));
        if (!is_padding_corr && uint(row_s0) < qo_len)
          ms_base_s0 = &gMaxScore(segment_offset + row_s0, qo_head_idx, 0);
        if (!is_padding_corr && uint(row_s1) < qo_len)
          ms_base_s1 = &gMaxScore(segment_offset + row_s1, qo_head_idx, 0);
      };

      if constexpr (kPackFactor > 1) {
        if (epilogue.params.ptr_MaxScore_direct != nullptr) {
          int rem_hr = get<3, 0, 0>(params_problem_shape);
          int h_r_orig = rem_hr * kPackFactor;
          int kv_head = qo_head_idx / rem_hr;
          int rem_hr_idx = qo_head_idx % rem_hr;
          int max_k = cute::size<2>(epilogue.params.layout_MaxScore);
          int tql_orig = epilogue.params.total_qo_len_orig;
          int seg_off_orig = segment_offset / kPackFactor;

          ms_stride_k = tql_orig;

          auto compute_ms_base = [&](int row) -> float* {
            if (uint(row) >= qo_len) return nullptr;
            int actual_tok = row / kPackFactor;
            int pf_idx = row % kPackFactor;
            int unpacked_head = kv_head * h_r_orig + rem_hr_idx * kPackFactor + pf_idx;
            return epilogue.params.ptr_MaxScore_direct
                   + unpacked_head * (max_k * tql_orig)
                   + seg_off_orig + actual_tok;
          };

          if (!is_padding_corr) {
            ms_base_s0 = compute_ms_base(row_s0);
            ms_base_s1 = compute_ms_base(row_s1);
          }
        } else {
          packed_maxscore_path();
        }
      } else {
        packed_maxscore_path();
      }
    }

    // Inline MaxScore write and correct scale
    // OnlyScore fuse v1: softmax already __stwt'd this value to GMEM directly
    // from its tile_max register. Skip correction-side GMEM write to avoid a
    // redundant store (~1 GMEM transaction per row per tile). The
    // max_score(kIdxNewRowMax) update below is kept because it feeds the
    // rescale path under kNeedOutput — dead in OnlyScore but harmless.
    auto write_max_score = [&](float* ms_base, auto& max_score, int k_idx) {
      if constexpr (!kFuseMaxScoreIntoSoftmax) {
        if (ms_base) ms_base[k_idx * ms_stride_k] = max_score(kIdxNewRowMax);
      }
      max_score(kIdxNewRowMax) = max(max_score(kIdxOldRowMax), max_score(kIdxNewRowMax));
      max_score(kIdxNewRowMax) = max_score(kIdxNewRowMax) == -INFINITY ? 0 : max_score(kIdxNewRowMax);
    };

    if constexpr (!kSkipPerTileCorrection) {
    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);
    if constexpr (kNeedMaxScore) {
      // P2 (v3): in OnlyScore fuse mode, the V-channel STOREV on the softmax
      // side is gated off (dead store). Skip the matching LOADV + dead
      // write_max_score path here. Sync (consumer_wait/arrive/release) is
      // preserved verbatim — only the per-iteration TMEM round-trip is dropped.
      if constexpr (!kFuseMaxScoreIntoSoftmax) {
        Tensor tTMEM_LOADVrS_first = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
        if (!is_padding_corr) {
          copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS_first);
          write_max_score(ms_base_s0, tTMEM_LOADVrS_first, k_tile_idx);
        }
      }
      pipeline_corr.arrive();
      if constexpr (kIsKSplit) k_tile_idx++;
    }
    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);
    if constexpr (kNeedMaxScore) {
      if constexpr (!kFuseMaxScoreIntoSoftmax) {
        Tensor tTMEM_LOADVrS_first = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
        if (!is_padding_corr) {
          copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS_first);
          write_max_score(ms_base_s1, tTMEM_LOADVrS_first, k_tile_idx);
        }
      }
      pipeline_corr.arrive();
      k_tile_idx -= (kTilesPerMacro + (kIsKSplit ? 1 : 0));
    }

    // handle the last iteration differently (i.e. tmem_load/stsm for epi)
    mask_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

      Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));

      if (!is_padding_corr) {
        GPU_TRACE_SCOPE(CORRECTION);
        // P2 (v3): skip dead LOADV + dead write_max_score in OnlyScore fuse mode.
        // softmax already wrote max_score to GMEM; rescale path is NeedOutput-gated.
        if constexpr (!kFuseMaxScoreIntoSoftmax) {
          // read row_wise new global max
          copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);
          if constexpr (kNeedMaxScore) {
            write_max_score(ms_base_s0, tTMEM_LOADVrS, k_tile_idx);
            if constexpr (kIsKSplit) k_tile_idx++;
          }
        }

        if constexpr (kNeedOutput) {
          // e^(scale * (old_max - new_max)
          float scale = ::exp2f(params.scale_softmax_log2 *
                                (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));
          bool warp_should_rescale = __any_sync(0xffffffff, scale != 1.f);
          if (warp_should_rescale) {
            correction_rescale(scale, uint32_t(TmemAllocation::O0));
          }
        }
      }

      pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
      ++pipeline_s1_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_corr.arrive();

      pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

      if (!is_padding_corr) {
        GPU_TRACE_SCOPE(CORRECTION);
        if constexpr (!kFuseMaxScoreIntoSoftmax) {
          copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);
          if constexpr (kNeedMaxScore) {
            write_max_score(ms_base_s1, tTMEM_LOADVrS, k_tile_idx);
            k_tile_idx -= (kTilesPerMacro + (kIsKSplit ? 1 : 0));
          }
        }

        if constexpr (kNeedOutput) {
          float scale = ::exp2f(params.scale_softmax_log2 *
                          (tTMEM_LOADVrS(kIdxOldRowMax) - tTMEM_LOADVrS(kIdxNewRowMax)));
          bool warp_should_rescale = __any_sync(0xffffffff, scale != 1.f);
          if (warp_should_rescale) {
            correction_rescale(scale, uint32_t(TmemAllocation::O1));
          }
        }
      }

      pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
      ++pipeline_s0_c_consumer_state;

      cutlass::arch::fence_view_async_tmem_store();

      pipeline_corr.arrive();
    }

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    }

    // do the final correction to O1
    // better to somehow special-case it in the loop above
    // doesn't matter for non-persistent code, but if it were
    // persistent we do not want to release O too early

    Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()),
                            typename TensorStorageEpi::SmemLayoutO{});
    Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), epilogue.params.layout_LSE);

    if constexpr (get<1>(ThreadShape{}) > 1) {
      // ========== K-split: merge O0+O1 with online softmax combine ==========
      pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);
      Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
      if constexpr (kNeedOutput)
        copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);
      pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
      ++pipeline_s0_c_consumer_state;

      pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);
      Tensor tTMEM_LOADVrS1 = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
      if constexpr (kNeedOutput)
        copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS1);
      pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
      ++pipeline_s1_c_consumer_state;

      // SingleSoftmaxWarpGroup merge uses smem_o as scratch — must wait for previous epilogue TMA
      if constexpr (kSingleWG) {
        pipeline_epi.producer_acquire(pipeline_epi_producer_state);
      }

      float total_sum = 0;
      float total_max = 0;

      if constexpr (kNeedOutput)
      {
        GPU_TRACE_SCOPE(CORRECTION_MRG);
        auto merge_result = correction_merge_ksplit<kSingleWG>(
            params, tTMEM_LOADVrS, tTMEM_LOADVrS1,
            kSingleWG ? reinterpret_cast<float*>(shared_storage_epi.smem_o.data()) : nullptr);
        total_sum = merge_result.total_sum;
        total_max = merge_result.total_max;
      }

      // Write merged O0 to smem
      if constexpr (!kSingleWG) {
        pipeline_epi.producer_acquire(pipeline_epi_producer_state);
      }

      if constexpr (kNeedOutput)
      {
        GPU_TRACE_SCOPE(CORRECTION_EPI);
        if constexpr (IsSplitKV) {
          float corr_scale = (total_sum == 0.f) ? 0.f : split_kv.scale_output_splitkv / total_sum;
          correction_epilogue(corr_scale, _0{}, sO);
        } else {
          correction_epilogue(params.scale_output / total_sum, _0{}, sO);
        }
        cutlass::arch::fence_view_async_tmem_load();
      }

      // Release O0 and O1
      if constexpr (!kNeedMaxScore) {
        pipeline_corr.arrive();
        pipeline_corr.arrive();
      }

      if constexpr (kNeedOutput && IsSplitKV)
      {
        // Write per-row LSE to smem; epilogue will flush to global memory
        int local_row = get<0>(tTMEM_LOADVcS(_0{}));
        float lse = (total_sum == 0.f) ? -INFINITY
                    : total_max + __log2f(total_sum) / params.scale_softmax_log2;
        shared_storage_epi.smem_lse[local_row] = lse;
      }

      pipeline_epi.producer_commit(pipeline_epi_producer_state);
      ++pipeline_epi_producer_state;

    } else {
    // ========== Q-split: write O0 and O1 independently (interleaved) ==========

    pipeline_s0_c.consumer_wait(pipeline_s0_c_consumer_state);

    Tensor tTMEM_LOADVrS = make_tensor<ElementQK>(shape(tTMEM_LOADVcS));
   
    if constexpr (kNeedOutput)
      copy(tiled_tmem_loadv, tTMEM_LOADVtS0, tTMEM_LOADVrS);

    pipeline_s0_c.consumer_release(pipeline_s0_c_consumer_state);
    ++pipeline_s0_c_consumer_state;

    // Wait for O0 and process it immediately
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);
    if constexpr (kNeedOutput) {
      GPU_TRACE_SCOPE(CORRECTION_EPI);
      correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _0{}, sO);
      if (epilogue.params.ptr_LSE != nullptr) {
        int qo_tile_idx = get<0>(blk_coord);
        int qo_head_idx = get<2, 0>(blk_coord);
        int batch_idx = get<2, 1>(blk_coord);
        int qo_len = get<0>(problem_shape);
        int _seg_off_sz = get<3, 1>(params_problem_shape) + 1;
        int segment_offset = SEG_OFF_LOAD(get<0>(params_problem_shape), batch_idx, _seg_off_sz);
        int row_idx = get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * qo_tile_idx;

        ElementPV lse = __log2f(tTMEM_LOADVrS(kIdxFinalRowSum)) +
                        params.scale_softmax_log2 * tTMEM_LOADVrS(kIdxFinalRowMax);

        if (row_idx < qo_len) {
          gLSE(segment_offset + row_idx, qo_head_idx) = lse;
        }
      }
      cutlass::arch::fence_view_async_tmem_load();
    }

    if constexpr (!kNeedMaxScore) {
      pipeline_corr.arrive();
    }

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    // Now read V1 and wait for O1
    pipeline_s1_c.consumer_wait(pipeline_s1_c_consumer_state);

    if constexpr (kNeedOutput)
      copy(tiled_tmem_loadv, tTMEM_LOADVtS1, tTMEM_LOADVrS);

    pipeline_s1_c.consumer_release(pipeline_s1_c_consumer_state);
    ++pipeline_s1_c_consumer_state;

    if constexpr (kNeedOutput) {
      GPU_TRACE_SCOPE(CORRECTION_EPI);
      correction_epilogue(params.scale_output / tTMEM_LOADVrS(kIdxFinalRowSum), _1{}, sO);
      if (epilogue.params.ptr_LSE != nullptr) {
        int qo_tile_idx = get<0>(blk_coord);
        int qo_head_idx = get<2, 0>(blk_coord);
        int batch_idx = get<2, 1>(blk_coord);
        int qo_len = get<0>(problem_shape);
        int _seg_off_sz = get<3, 1>(params_problem_shape) + 1;
        int segment_offset = SEG_OFF_LOAD(get<0>(params_problem_shape), batch_idx, _seg_off_sz);
        int row_idx =
            get<0>(tTMEM_LOADVcS(_0{})) + get<0>(TileShape{}) * qo_tile_idx + get<0>(TileShapeQK{});

        ElementPV lse = __log2f(tTMEM_LOADVrS(kIdxFinalRowSum)) +
                        params.scale_softmax_log2 * tTMEM_LOADVrS(kIdxFinalRowMax);

        if (row_idx < qo_len) {
          gLSE(segment_offset + row_idx, qo_head_idx) = lse;
        }
      }
      cutlass::arch::fence_view_async_tmem_load();
    }

    if constexpr (!kNeedMaxScore) {
      pipeline_corr.arrive();
    }

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;
    } // end Q-split/K-split
    RELEASE_GPU_TRACE;
  }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape, class TensorStorageEpi,
            class CollectiveEpilogue>
  CUTLASS_DEVICE auto correction_empty(
      BlkCoord const& blk_coord, Params const& params, ProblemShape const& problem_shape,
      ParamsProblemShape const& params_problem_shape, TensorStorageEpi& shared_storage_epi,
      PipelineE& pipeline_epi, typename PipelineE::PipelineState& pipeline_epi_producer_state,
      CollectiveEpilogue& epilogue) {
    pipeline_epi.producer_acquire(pipeline_epi_producer_state);

    if constexpr (kNeedOutput) {
      Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()),
                              typename TensorStorageEpi::SmemLayoutO{});
      int thread_idx = threadIdx.x % (4 * NumThreadsPerWarp);

      using ElementOut = typename CollectiveEpilogue::ElementOut;
      auto tiled_copy = make_cotiled_copy(
          Copy_Atom<UniversalCopy<uint32_t>, ElementOut>{},
          make_ordered_layout(make_shape(_128{}, Int<sizeof(uint32_t) / sizeof(ElementOut)>{}),
                              Step<_1, _0>{}),
          sO.layout());

      auto thr_copy = tiled_copy.get_slice(thread_idx);
      auto tOgO = thr_copy.partition_D(sO);
      auto tOrO = make_tensor<ElementOut>(shape(tOgO(_, _, _, _0{})));
      clear(tOrO);

      copy(tiled_copy, tOrO, tOgO(_, _, _, _0{}));

      if (epilogue.params.ptr_LSE != nullptr) {
        Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), epilogue.params.layout_LSE);
        int qo_tile_idx = get<0>(blk_coord);
        int qo_head_idx = get<2, 0>(blk_coord);
        int batch_idx = get<2, 1>(blk_coord);
        int qo_len = get<0>(problem_shape);
        int _seg_off_sz = get<3, 1>(params_problem_shape) + 1;
        int segment_offset = SEG_OFF_LOAD(get<0>(params_problem_shape), batch_idx, _seg_off_sz);
        int row_idx = thread_idx + get<0>(TileShape{}) * qo_tile_idx;

        if (row_idx < qo_len) {
          gLSE(segment_offset + row_idx, qo_head_idx) = -cuda::std::numeric_limits<float>::infinity();
        }
      }

      if constexpr (kNeedOutput && IsSplitKV) {
        if (thread_idx < int(get<0>(TileShape{}))) {
          shared_storage_epi.smem_lse[thread_idx] = -INFINITY;
        }
      }
    }

    pipeline_epi.producer_commit(pipeline_epi_producer_state);
    ++pipeline_epi_producer_state;

    if constexpr (get<1>(ThreadShape{}) <= 1) {
      if constexpr (kNeedOutput) {
        Tensor sO = make_tensor(make_smem_ptr(shared_storage_epi.smem_o.data()),
                                typename TensorStorageEpi::SmemLayoutO{});
        int thread_idx = threadIdx.x % (4 * NumThreadsPerWarp);
        using ElementOut = typename CollectiveEpilogue::ElementOut;
        auto tiled_copy = make_cotiled_copy(
            Copy_Atom<UniversalCopy<uint32_t>, ElementOut>{},
            make_ordered_layout(make_shape(_128{}, Int<sizeof(uint32_t) / sizeof(ElementOut)>{}),
                                Step<_1, _0>{}),
            sO.layout());
        auto thr_copy = tiled_copy.get_slice(thread_idx);
        auto tOgO = thr_copy.partition_D(sO);
        auto tOrO = make_tensor<ElementOut>(shape(tOgO(_, _, _, _0{})));
        clear(tOrO);
        copy(tiled_copy, tOrO, tOgO(_, _, _, _1{}));
        cutlass::arch::fence_view_async_shared();
      }

      pipeline_epi.producer_acquire(pipeline_epi_producer_state);

      if constexpr (kNeedOutput) {
        if (epilogue.params.ptr_LSE != nullptr) {
          Tensor gLSE = make_tensor(make_gmem_ptr(epilogue.params.ptr_LSE), epilogue.params.layout_LSE);
          int thread_idx = threadIdx.x % (4 * NumThreadsPerWarp);
          int qo_tile_idx = get<0>(blk_coord);
          int qo_head_idx = get<2, 0>(blk_coord);
          int batch_idx = get<2, 1>(blk_coord);
          int qo_len = get<0>(problem_shape);
          int _seg_off_sz = get<3, 1>(params_problem_shape) + 1;
        int segment_offset = SEG_OFF_LOAD(get<0>(params_problem_shape), batch_idx, _seg_off_sz);
          int row_idx = thread_idx + get<0>(TileShape{}) * qo_tile_idx
                      + int(get<0>(TileShapeQK{}));
          if (row_idx < qo_len) {
            gLSE(segment_offset + row_idx, qo_head_idx) = -cuda::std::numeric_limits<float>::infinity();
          }
        }
        cutlass::arch::fence_view_async_shared();
      }

      pipeline_epi.producer_commit(pipeline_epi_producer_state);
      ++pipeline_epi_producer_state;
    }
    // K-split: only 1 epi commit (merged output), skip O1
  }
};

}  // namespace cutlass::fmha::collective
