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
#include "cutlass_utils.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "fmha_common.hpp"
#include "gpu_trace.h"
#include "fmha_fusion.hpp"

#if (__CUDACC_VER_MAJOR__ >= 12) && !defined(__CUDACC_RTC__)
#include <cuda.h>
#endif

GPU_TRACE_SCOPE_DEC(LOAD_Q);
GPU_TRACE_SCOPE_DEC(LOAD_Q_WAIT);
GPU_TRACE_SCOPE_DEC(LOAD_K);
GPU_TRACE_SCOPE_DEC(LOAD_V);

namespace cutlass::fmha::collective {

using namespace cute;

template <typename T>
struct MaskPackFactor { static constexpr int value = 1; };
template <int P>
struct MaskPackFactor<PackedCausalMask<P>> { static constexpr int value = P; };

template <class Element, class CollectiveMmaQK, class CollectiveMmaPV, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class TensorStorage, class PipelineQ,
          class PipelineKV, class Mask, class TileShape, int KVPageSize = -1,
          SparseAttnMode kSparseAttnMode = SparseAttnMode::Off,
          bool UnitSelectorColumn = false>
struct Sm100FmhaLoadTmaWarpspecialized {
  // P0: OnlyScore mode never reads V — skip every V slot in the KV pipeline.
  // mma() gates the matching wait_V/release_V under `kNeedOutput` so producer
  // (loader) and consumer (mma) stay step-locked on the pipeline state.
  static constexpr bool kNeedV = (kSparseAttnMode != SparseAttnMode::OnlyScore);

  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
  static constexpr uint32_t NumStagesQ = PipelineQ::Stages;
  static constexpr int kTransactionBytesKV =
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutK{})) * cute::sizeof_bits_v<Element>);

  // (N, D, (H_R, H_G))
  using ShapeQ = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideQ = cute::Shape<int32_t, _1, cute::Shape<int32_t, int32_t>>;
  using LayoutQ = cute::Layout<ShapeQ, StrideQ>;

  // Paged: (page_size, D, H_kv, total_page_num) — 4-mode flat
  // Non-paged: (N, D, (H_R, H_G)) — 3-mode hierarchical
  using ShapeK = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, int32_t, int32_t, int32_t>,
      cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>>;
  using StrideK = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, _1, int32_t, int32_t>,
      cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>>;
  using ShapeV = std::conditional_t<(KVPageSize > 0),
      cute::Shape<int32_t, int32_t, int32_t, int32_t>,
      cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>>;
  using StrideV = std::conditional_t<(KVPageSize > 0),
      cute::Shape<_1, int32_t, int32_t, int32_t>,
      cute::Shape<_1, int32_t, cute::Shape<_0, int32_t>>>;
  using LayoutK = cute::Layout<ShapeK, StrideK>;
  using LayoutV = cute::Layout<ShapeV, StrideV>;
  struct Arguments {
    const Element* ptr_Q;
    LayoutQ layout_Q;
    const Element* ptr_K;
    LayoutK layout_K;
    const Element* ptr_V;
    LayoutV layout_V;
    int* kv_indices = nullptr;
    int* kv_page_indptr = nullptr;
    int* kv_block_indexes = nullptr;
    int kv_block_num = 0;
    int64_t kv_block_stride_n = 0;
    int64_t kv_block_stride_h = 0;
    int64_t kv_block_stride_b = 1;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int kv_page_indptr_size = 0;
    int kv_indices_size = 0;
    int64_t kv_block_indexes_span = 0;
#endif
    int pack_factor = 1;
    int q_stride_n_original = 0;
    int q_stride_h_original = 0;
    int h_r_original = 0;
  };

  // using ShapeLseT = cute::Shape<int32_t, int32_t>;
  // using StrideLseT = cute::Shape<_1, int64_t>;
  // using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  using ClusterLayout_VMNK =
      decltype(tiled_divide(make_layout(Shape<_1, _1, _1>{}),
                            make_tile(typename CollectiveMmaQK::TiledMma::AtomThrID{})));
  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

  static constexpr int kPackFactor = MaskPackFactor<Mask>::value;
  static constexpr bool kEnablePackedQTMA = (kPackFactor > 1) && ((kPackFactor & (kPackFactor - 1)) == 0);

  struct Params {
    TMA_Q tma_load_Q;
    LayoutQ layout_Q;
    TMA_K tma_load_K;
    LayoutK layout_K;
    TMA_V tma_load_V;
    LayoutV layout_V;
    int* kv_indices = nullptr;
    int* kv_page_indptr = nullptr;
    int* kv_block_indexes = nullptr;
    int kv_block_num = 0;
    int64_t kv_block_stride_n = 0;
    int64_t kv_block_stride_h = 0;
    int64_t kv_block_stride_b = 1;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int kv_page_indptr_size = 0;
    int kv_indices_size = 0;
    int64_t kv_block_indexes_span = 0;
#endif
    const Element* ptr_Q_orig = nullptr;
    int q_stride_n_orig = 0;
    int q_stride_h_orig = 0;
    int h_r_orig = 0;
    cute::TmaDescriptor tma_desc_q_pack;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
                                        void* workspace) {
    static_assert(is_variable_length_v<tuple_element_t<0, ProblemShape>>);
    static_assert(is_variable_length_v<tuple_element_t<1, ProblemShape>>);
    auto ptr_Q = args.ptr_Q;
    auto ptr_K = args.ptr_K;
    auto ptr_V = args.ptr_V;
    LayoutQ layout_Q = args.layout_Q;
    LayoutK layout_K = args.layout_K;
    LayoutV layout_V = args.layout_V;

    auto mQ = make_tensor(make_gmem_ptr(ptr_Q), layout_Q);
    auto mK = make_tensor(make_gmem_ptr(ptr_K), layout_K);
    auto mV = make_tensor(make_gmem_ptr(ptr_V), layout_V);

    auto cluster_layout_vmnk =
        tiled_divide(make_layout(Shape<_1, _1, _1>{}),
                     make_tile(typename CollectiveMmaQK::TiledMma::AtomThrID{}));
    TMA_Q tma_load_Q = make_tma_atom_A_sm100<Element>(
        GmemTiledCopyQ{}, mQ, SmemLayoutQ{}(_, _, _, _0{}), TileShapeQK{},
        typename CollectiveMmaQK::TiledMma{}, cluster_layout_vmnk);
    TMA_K tma_load_K = make_tma_atom_B_sm100<Element>(
        GmemTiledCopyKV{}, mK, SmemLayoutK{}(_, _, _, _0{}), TileShapeQK{},
        typename CollectiveMmaQK::TiledMma{}, cluster_layout_vmnk);
    TMA_V tma_load_V = make_tma_atom_B_sm100<Element>(
        GmemTiledCopyKV{}, mV, SmemLayoutV{}(_, _, _, _0{}), TileShapePV{},
        typename CollectiveMmaPV::TiledMma{}, cluster_layout_vmnk);

    cute::TmaDescriptor tma_desc_q_pack{};
    if constexpr (kEnablePackedQTMA) {
      constexpr int tile_m = get<0>(TileShapeQK{});
      constexpr int tile_k = get<2>(TileShapeQK{});
      int total_seq = get<0>(shape(layout_Q)) / kPackFactor;
      int total_heads = args.h_r_original * get<1>(get<2>(shape(layout_Q)));
      int dim = get<1>(shape(layout_Q));

      auto tma_dtype = cute::TMA::to_CUtensorMapDataType<Element>();
      constexpr int box_dim0 = 128 / (int)sizeof(Element);
      uint64_t gDim[3] = {(uint64_t)dim, (uint64_t)total_heads, (uint64_t)total_seq};
      uint64_t gStride[2] = {
          (uint64_t)(args.q_stride_h_original * (int)sizeof(Element)),
          (uint64_t)(args.q_stride_n_original * (int)sizeof(Element))};
      uint32_t bDim[3] = {(uint32_t)box_dim0, (uint32_t)kPackFactor, (uint32_t)(tile_m / kPackFactor)};
      uint32_t eStride[3] = {1, 1, 1};

      cuTensorMapEncodeTiled(
          reinterpret_cast<CUtensorMap*>(&tma_desc_q_pack),
          tma_dtype, 3, (void*)args.ptr_Q,
          gDim, gStride, bDim, eStride,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_128B,
          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    Params p{tma_load_Q, layout_Q, tma_load_K, layout_K, tma_load_V, layout_V,
                  args.kv_indices, args.kv_page_indptr, args.kv_block_indexes, args.kv_block_num,
                  args.kv_block_stride_n, args.kv_block_stride_h, args.kv_block_stride_b,
#ifdef FMHA_GMEM_BOUNDS_CHECK
                  args.kv_page_indptr_size, args.kv_indices_size, args.kv_block_indexes_span,
#endif
                  args.ptr_Q, args.q_stride_n_original, args.q_stride_h_original, args.h_r_original,
                  tma_desc_q_pack};
    return p;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    if constexpr (kEnablePackedQTMA)
      cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&params.tma_desc_q_pack));
    else if constexpr (kPackFactor == 1)
      cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
  }

  static constexpr int kTransactionBytesQ =
      cutlass::bits_to_bytes(cosize(take<0, 3>(SmemLayoutQ{})) * cute::sizeof_bits_v<Element>);

  CUTLASS_DEVICE
  static void load_Q_cp_async(
      int q_tile_index, int qo_head_idx, int qo_segment_offset, int qo_len,
      Params const& params, auto const& params_problem_shape,
      TensorStorage& storage, PipelineQ& pipeline_q,
      typename PipelineQ::PipelineState& pipeline_q_producer_state,
      uint32_t lane_predicate
      #ifdef GPU_TRACE_ENABLED
        , gpu_trace::Recorder& _gt_rec
      #endif
      ) {

    constexpr int tile_m = get<0>(TileShapeQK{});
    constexpr int tile_k = get<2>(TileShapeQK{});
    int stage = pipeline_q_producer_state.index();

    int h_r_orig = params.h_r_orig;
    int h_r_packed = h_r_orig / kPackFactor;
    int kv_head = qo_head_idx / h_r_packed;
    int rem_h = qo_head_idx % h_r_packed;

    int tile_base = q_tile_index * tile_m;

    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    auto sQ_grouped = group_modes<0, rank(SmemLayoutQ{}) - 1>(sQ);

    int lane_id = threadIdx.x % 32;
    int packed_len = qo_segment_offset + qo_len;

    {
      GPU_TRACE_SCOPE(LOAD_Q);
      constexpr int kVecElems = 16 / sizeof(Element);
      constexpr int k_iters = tile_k / kVecElems;
      int valid_m = min(tile_m, packed_len - (qo_segment_offset + tile_base));
      if (valid_m < 0) valid_m = 0;
      int total_ops = valid_m * k_iters;

      for (int idx = lane_id; idx < total_ops; idx += 32) {
        int m = idx / k_iters;
        int k = (idx % k_iters) * kVecElems;

        int packed_pos = qo_segment_offset + tile_base + m;
        int orig_pos = packed_pos / kPackFactor;
        int pack_idx = packed_pos % kPackFactor;
        int orig_head = kv_head * h_r_orig + rem_h * kPackFactor + pack_idx;

        const Element* gmem_src = params.ptr_Q_orig
                                + orig_pos * params.q_stride_n_orig
                                + orig_head * params.q_stride_h_orig
                                + k;

        Element* smem_dst = reinterpret_cast<Element*>(&sQ_grouped(m + tile_m * k, stage));
        cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
            smem_dst, gmem_src);
      }
    }
  }

  CUTLASS_DEVICE
  static void load_Q_wait(PipelineQ& pipeline_q,
      typename PipelineQ::PipelineState& pipeline_q_producer_state,
      uint32_t lane_predicate
      #ifdef GPU_TRACE_ENABLED
        , gpu_trace::Recorder& _gt_rec
      #endif
      ) {

    GPU_TRACE_SCOPE(LOAD_Q_WAIT);
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();

    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      cutlass::arch::ClusterTransactionBarrier::complete_transaction(
          tma_barrier, cute::block_rank_in_cluster(), kTransactionBytesQ);
    }
  }

  template <bool IsSplitKV = false, bool LoadV = true, class BlkCoord, class ProblemShape, class ParamsProblemShape, class TensorStorageType>
  CUTLASS_DEVICE void load(BlkCoord const& blk_coord, ProblemShape const& problem_shape,
                           Params const& params, ParamsProblemShape const& params_problem_shape,
                           int const& work_idx,
                           TensorStorageType& storage, PipelineQ& pipeline_q,
                           typename PipelineQ::PipelineState& pipeline_q_producer_state,
                           PipelineKV& pipeline_kv,
                           typename PipelineKV::PipelineState& pipeline_kv_producer_state,
                           int kv_tile_begin, int kv_tile_end
                           #ifdef GPU_TRACE_ENABLED
                             , gpu_trace::Recorder& _gt_rec
                           #endif
                          ) {
    // GET_GPU_TRACE(true);

    int qo_tile_idx = get<0>(blk_coord);
    int qo_head_idx = get<2, 0>(blk_coord);
    int batch_idx = get<2, 1>(blk_coord);
    int qo_len = get<0>(problem_shape);
    int kv_len = get<1>(problem_shape);
    int qo_segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];
    int kv_segment_offset = get<1>(params_problem_shape).segment_offsets[batch_idx];

    int full_trip;
    if constexpr (kSparseAttnMode == SparseAttnMode::Sparse) {
      constexpr int full_tile_kv = get<1>(TileShape{});
      full_trip = (params.kv_block_num * KVPageSize + full_tile_kv - 1) / full_tile_kv;
    } else {
      full_trip = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);
    }
    int effective_end = full_trip < kv_tile_end ? full_trip : kv_tile_end;
    int mask_tile_count = effective_end - kv_tile_begin;
    if constexpr (IsSplitKV) {
      if (mask_tile_count <= 0) return;
    }
    int start_kv_index = effective_end - 1;

    using X = Underscore;

    Tensor mQ = params.tma_load_Q.get_tma_tensor(params.layout_Q.shape());

    ThrMMA mma_qk = typename CollectiveMmaQK::TiledMma{}.get_slice(0);
    ThrMMA mma_pv = typename CollectiveMmaPV::TiledMma{}.get_slice(0);
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_kv.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_kv.data()), SmemLayoutV{});

    auto gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShapeQK{}), qo_head_idx, qo_segment_offset,
                                    qo_len);
    Tensor tSgQ_qdl = mma_qk.partition_A(gQ);
    auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 3>(sQ),
                                      group_modes<0, 3>(tSgQ_qdl));

    uint32_t lane_predicate = cute::elect_one_sync();

    static constexpr int num_q_sub = get<0>(TileShape{}) / get<0>(TileShapeQK{});
    static constexpr int num_kv_sub = get<1>(TileShape{}) / get<1>(TileShapeQK{});

    int q0_index = num_q_sub * get<0>(blk_coord);
    int q1_index = num_q_sub * get<0>(blk_coord) + 1;
    int kv_tile_index = start_kv_index * num_kv_sub;

    auto run_loads = [&](auto load_K_tile, auto load_V_tile) {

      if constexpr (kEnablePackedQTMA) {
        static_assert(num_kv_sub > 1);
        static_assert(num_q_sub == 1);

        pipeline_q.producer_acquire(pipeline_q_producer_state);
        {
          GPU_TRACE_SCOPE(LOAD_Q);
          if (lane_predicate) {
            constexpr int tile_m = get<0>(TileShapeQK{});
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            uint32_t smem_int_mbar = cute::cast_smem_ptr_to_uint(&(*tma_barrier));
            uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&params.tma_desc_q_pack);

            int h_r_packed = params.h_r_orig / kPackFactor;
            int kv_head = qo_head_idx / h_r_packed;
            int rem_h = qo_head_idx % h_r_packed;
            int first_head = kv_head * params.h_r_orig + rem_h * kPackFactor;
            int tile_base = q0_index * tile_m;
            int orig_token_start = (qo_segment_offset + tile_base) / kPackFactor;

            int stage = pipeline_q_producer_state.index();
            constexpr int stage_bytes = kTransactionBytesQ;
            uint32_t smem_base = cute::cast_smem_ptr_to_uint(storage.smem_q.data())
                                  + stage * stage_bytes;

            constexpr int box_dim0 = 128 / (int)sizeof(Element);
            constexpr int dim_iters = get<2>(TileShapeQK{}) / box_dim0;
            constexpr int chunk_bytes = box_dim0 * tile_m * (int)sizeof(Element);
            for (int di = 0; di < dim_iters; di++) {
              uint32_t smem_int_ptr = smem_base + di * chunk_bytes;
              asm volatile(
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
                  "cp.async.bulk.tensor.3d.shared::cta.global.tile"
#else
                  "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
#endif
                  ".mbarrier::complete_tx::bytes.L2::cache_hint"
                  " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                  :
                  : "r"(smem_int_ptr), "l"(gmem_int_desc),
                    "r"(di * box_dim0), "r"(first_head), "r"(orig_token_start),
                    "r"(smem_int_mbar), "l"((uint64_t)0)
                  : "memory");
            }
          }
        }
        ++pipeline_q_producer_state;

        load_K_tile(kv_tile_index);
        load_K_tile(kv_tile_index + 1);
      } else if constexpr (kPackFactor > 1) {
        static_assert(num_kv_sub > 1);
        static_assert(num_q_sub == 1);

        load_K_tile(kv_tile_index);
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        load_Q_cp_async(q0_index, qo_head_idx, qo_segment_offset, qo_len,
                        params, params_problem_shape, storage, pipeline_q,
                        pipeline_q_producer_state, lane_predicate
                        #ifdef GPU_TRACE_ENABLED
                          , _gt_rec
                        #endif
                        );
        load_K_tile(kv_tile_index + 1);
        load_Q_wait(pipeline_q, pipeline_q_producer_state, lane_predicate
          #ifdef GPU_TRACE_ENABLED
            , _gt_rec
          #endif
          );
        ++pipeline_q_producer_state;
      } else {
        pipeline_q.producer_acquire(pipeline_q_producer_state);
        {
          GPU_TRACE_SCOPE(LOAD_Q);
          if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            copy(params.tma_load_Q.with(*tma_barrier, 0), tQgQ(_, q0_index),
                 tQsQ(_, pipeline_q_producer_state.index()));
          }
        }
        ++pipeline_q_producer_state;

        if constexpr (num_q_sub > 1) {
          pipeline_q.producer_acquire(pipeline_q_producer_state);
          GPU_TRACE_SCOPE(LOAD_Q);
          if (lane_predicate) {
            auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
            copy(params.tma_load_Q.with(*tma_barrier, 0), tQgQ(_, q1_index),
                 tQsQ(_, pipeline_q_producer_state.index()));
          }
          ++pipeline_q_producer_state;
        }
        load_K_tile(kv_tile_index);
        if constexpr (num_kv_sub > 1) load_K_tile(kv_tile_index + 1);
      }

      if constexpr (num_kv_sub > 1) {
        // K-split: pre(KK) already done above; loop body is VKVK; epi is VV
        int v_tile_index = kv_tile_index;
        kv_tile_index -= 2;

        mask_tile_count -= 1;
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
          if constexpr (kNeedV) load_V_tile(v_tile_index);
          load_K_tile(kv_tile_index);
          if constexpr (kNeedV) load_V_tile(v_tile_index + 1);
          load_K_tile(kv_tile_index + 1);
          v_tile_index = kv_tile_index;
          kv_tile_index -= 2;
        }

        if constexpr (kNeedV) {
          load_V_tile(v_tile_index);
          load_V_tile(v_tile_index + 1);
        }
      } else {
        if constexpr (kNeedV) load_V_tile(kv_tile_index);
        kv_tile_index -= 1;

        mask_tile_count -= 1;
        for (; mask_tile_count > 0; mask_tile_count -= 1) {
          load_K_tile(kv_tile_index);
          if constexpr (kNeedV) load_V_tile(kv_tile_index);
          kv_tile_index -= 1;
        }
      }
    };

    if constexpr (KVPageSize > 0) {
      // Paged KV: K/V shape (page_size, D, num_kv_heads, total_page_num)
      Tensor mK = params.tma_load_K.get_tma_tensor(params.layout_K.shape());
      Tensor mV = params.tma_load_V.get_tma_tensor(params.layout_V.shape());

      int h_r = get<3, 0, 0>(params_problem_shape);
      int kv_head_idx = qo_head_idx / h_r;
      int kv_page_start = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx, params.kv_page_indptr_size);
      int num_pages_batch = 0;
      if constexpr (kSparseAttnMode != SparseAttnMode::Sparse) {
        int kv_page_end = KV_INDPTR_LOAD(params.kv_page_indptr, batch_idx + 1, params.kv_page_indptr_size);
        num_pages_batch = kv_page_end - kv_page_start;
      }
      constexpr int effective_tile_kv = get<1>(TileShapeQK{});
      constexpr int tiles_per_page = KVPageSize / effective_tile_kv;

      int64_t kv_block_offset = 0;
      if constexpr (kSparseAttnMode == SparseAttnMode::Sparse) {
        kv_block_offset = batch_idx * params.kv_block_stride_n
                        + kv_head_idx * params.kv_block_stride_h;
      }

      // Keep full 4D tensor (page_size, D, H_kv, P) — select head at copy time
      auto gK = local_tile(mK, select<1, 2>(TileShapeQK{}), make_coord(_, _0{}));
      auto gV = local_tile(mV, select<1, 2>(TileShapePV{}), make_coord(_0{}, _));

      Tensor tSgK_kdl = mma_qk.partition_B(gK);
      Tensor tOgV_dkl = mma_pv.partition_B(gV);
      auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                         group_modes<0, 3>(sK), group_modes<0, 3>(tSgK_kdl));
      auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                         group_modes<0, 3>(sV), group_modes<0, 3>(tOgV_dkl));

      auto load_K_tile = [&](int tile_idx) {
        int logical_page = tile_idx / tiles_per_page;
        int sub_tile = tile_idx % tiles_per_page;
        int page_for_lookup;
        if constexpr (kSparseAttnMode == SparseAttnMode::Sparse) {
          int sparse_idx = (logical_page < params.kv_block_num)
              ? (UnitSelectorColumn
                    ? KV_BLOCK_INDEX_LOAD(params.kv_block_indexes,
                          kv_block_offset + logical_page, params.kv_block_indexes_span)
                    : KV_BLOCK_INDEX_LOAD(params.kv_block_indexes,
                          kv_block_offset + logical_page * params.kv_block_stride_b,
                          params.kv_block_indexes_span))
              : -1;
          page_for_lookup = (sparse_idx >= 0) ? sparse_idx : 0;
        } else {
          page_for_lookup = logical_page;
          page_for_lookup = min(page_for_lookup, num_pages_batch - 1);
        }
        int physical_page = KV_INDICES_LOAD(params.kv_indices, kv_page_start + page_for_lookup, params.kv_indices_size);
        { GPU_TRACE_SCOPE(LOAD_K); pipeline_kv.producer_acquire(pipeline_kv_producer_state); }
        if (lane_predicate) {
          auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
          copy(params.tma_load_K.with(*tma_barrier, 0),
               tKgK(_, sub_tile, kv_head_idx, physical_page),
               tKsK(_, pipeline_kv_producer_state.index()));
        }
        ++pipeline_kv_producer_state;
      };

      auto load_V_tile = [&](int tile_idx) {
        int logical_page = tile_idx / tiles_per_page;
        int sub_tile = tile_idx % tiles_per_page;
        int page_for_lookup;
        if constexpr (kSparseAttnMode == SparseAttnMode::Sparse) {
          int sparse_idx = (logical_page < params.kv_block_num)
              ? (UnitSelectorColumn
                    ? KV_BLOCK_INDEX_LOAD(params.kv_block_indexes,
                          kv_block_offset + logical_page, params.kv_block_indexes_span)
                    : KV_BLOCK_INDEX_LOAD(params.kv_block_indexes,
                          kv_block_offset + logical_page * params.kv_block_stride_b,
                          params.kv_block_indexes_span))
              : -1;
          page_for_lookup = (sparse_idx >= 0) ? sparse_idx : 0;
        } else {
          page_for_lookup = logical_page;
          page_for_lookup = min(page_for_lookup, num_pages_batch - 1);
        }
        int physical_page = KV_INDICES_LOAD(params.kv_indices, kv_page_start + page_for_lookup, params.kv_indices_size);
        { GPU_TRACE_SCOPE(LOAD_V); pipeline_kv.producer_acquire(pipeline_kv_producer_state); }
        if (lane_predicate) {
          auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
          if constexpr (LoadV) {
            copy(params.tma_load_V.with(*tma_barrier, 0),
                 tVgV(_, sub_tile, kv_head_idx, physical_page),
                 tVsV(_, pipeline_kv_producer_state.index()));
          } else {
            cutlass::arch::ClusterTransactionBarrier::complete_transaction(
                tma_barrier, cute::block_rank_in_cluster(), kTransactionBytesKV);
          }
        }
        ++pipeline_kv_producer_state;
      };

      run_loads(load_K_tile, load_V_tile);
    } else {
      // Non-paged KV
      Tensor mK = params.tma_load_K.get_tma_tensor(params.layout_K.shape());
      Tensor mV = params.tma_load_V.get_tma_tensor(params.layout_V.shape());

      auto gK = get_local_tile_tensor(mK, select<1, 2>(TileShapeQK{}), qo_head_idx,
                                      kv_segment_offset, kv_len);
      auto gV = get_local_tile_t_tensor(mV, select<1, 2>(TileShapePV{}), qo_head_idx,
                                        kv_segment_offset, kv_len);

      Tensor tSgK_kdl = mma_qk.partition_B(gK);
      Tensor tOgV_dkl = mma_pv.partition_B(gV);
      auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                         group_modes<0, 3>(sK), group_modes<0, 3>(tSgK_kdl));
      auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                         group_modes<0, 3>(sV), group_modes<0, 3>(tOgV_dkl));

      auto load_K_tile = [&](int tile_idx) {
        { GPU_TRACE_SCOPE(LOAD_K); pipeline_kv.producer_acquire(pipeline_kv_producer_state); }
        if (lane_predicate) {
          auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
          copy(params.tma_load_K.with(*tma_barrier, 0), tKgK(_, tile_idx),
               tKsK(_, pipeline_kv_producer_state.index()));
        }
        ++pipeline_kv_producer_state;
      };

      auto load_V_tile = [&](int tile_idx) {
        { GPU_TRACE_SCOPE(LOAD_V); pipeline_kv.producer_acquire(pipeline_kv_producer_state); }
        if (lane_predicate) {
          auto tma_barrier = pipeline_kv.producer_get_barrier(pipeline_kv_producer_state);
          if constexpr (LoadV) {
            copy(params.tma_load_V.with(*tma_barrier, 0), tVgV(_, tile_idx),
                 tVsV(_, pipeline_kv_producer_state.index()));
          } else {
            cutlass::arch::ClusterTransactionBarrier::complete_transaction(
                tma_barrier, cute::block_rank_in_cluster(), kTransactionBytesKV);
          }
        }
        ++pipeline_kv_producer_state;
      };

      run_loads(load_K_tile, load_V_tile);
    }
    // RELEASE_GPU_TRACE;
  }
};

}  // namespace cutlass::fmha::collective
