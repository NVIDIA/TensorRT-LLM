/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "fmha_common.hpp"
#include "fmha_fusion.hpp"
#include "gmem_bounds_check.h"
#include "gpu_trace.h"
#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/layout.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "fmha_options.hpp"
#include "fmha_tile_scheduler.hpp"

GPU_TRACE_SCOPE_DEC(KERNEL_LAUNCHED);

namespace cutlass::fmha::kernel {

using namespace cute;
using namespace cutlass::fmha::collective;

template <bool SingleSoftmaxWarpGroup_ = false>
struct Sm100FmhaCtxKernelWarpspecializedSchedule {
  static constexpr bool SingleSoftmaxWarpGroup = SingleSoftmaxWarpGroup_;
  enum class WarpRole { Softmax0, Softmax1, Correction, MMA, Load, Epilogue, Empty };

  static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    if (warp_idx == 0) return WarpRole::Load;
    if (warp_idx == 1) return WarpRole::MMA;
    if (warp_idx == 2) return WarpRole::Epilogue;
    if (warp_idx == 3) return WarpRole::Empty;
    if constexpr (SingleSoftmaxWarpGroup) {
      // Merged: warps 0-1 = Softmax0, 2-3 = Softmax1, no separate warpgroups
      if (warp_idx < 6) return WarpRole::Softmax0;
      if (warp_idx < 8) return WarpRole::Softmax1;
      if (warp_idx < 12) return WarpRole::Correction;
    } else {
      int wg_idx = warp_idx / 4;
      if (wg_idx == 1) return WarpRole::Softmax0;
      if (wg_idx == 2) return WarpRole::Softmax1;
      if (wg_idx == 3) return WarpRole::Correction;
    }
    return WarpRole::Empty;
  }

  static const int NumWarpsSoftmaxPerStage = SingleSoftmaxWarpGroup ? 2 : 4;
  static const int NumWarpsCorrection = 4;
  static const int NumWarpsEpilogue = 1;
  static const int NumWarpsLoad = 1;

  static const bool kDebugUsingPrintf = false;
  static const int NumRegsSoftmax = SingleSoftmaxWarpGroup ? 232 : 192;
  static const int NumRegsCorrection = SingleSoftmaxWarpGroup ? 96 : 64;
  static const int NumRegsOther = SingleSoftmaxWarpGroup ? 72 : 64;

  static const int NumWarps = SingleSoftmaxWarpGroup ? 12 : 16;
};

template <class ProblemShapeIn, class CollectiveMainloop, class CollectiveEpilogue,
          class TileScheduler, class KernelSchedule = Sm100FmhaCtxKernelWarpspecializedSchedule<false>>
struct Sm100FmhaFwdKernelTmaWarpspecialized {
  using TileShape = typename CollectiveMainloop::TileShape;
  using ProblemShape = ProblemShapeIn;

  using WarpRole = typename KernelSchedule::WarpRole;

  constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    return KernelSchedule::warp_idx_to_WarpRole(warp_idx);
  }

  static const int NumWarpsCorrection = KernelSchedule::NumWarpsCorrection;
  static const int NumWarpsEpilogue = KernelSchedule::NumWarpsEpilogue;
  static const int NumWarpsLoad = KernelSchedule::NumWarpsLoad;

  static const int NumRegsSoftmax = KernelSchedule::NumRegsSoftmax;
  static const int NumRegsCorrection = KernelSchedule::NumRegsCorrection;
  static const int NumRegsOther = KernelSchedule::NumRegsOther;

  static const int NumWarps = KernelSchedule::NumWarps;

  static constexpr bool kSingleSoftmaxWarpGroup = KernelSchedule::SingleSoftmaxWarpGroup;
  static constexpr int kNumWarpsSoftmaxPerStage = KernelSchedule::NumWarpsSoftmaxPerStage;

  using ClusterShape = typename CollectiveMainloop::ClusterShape;

  // No smem_v padding needed — K/V share smem_kv which is separate from epilogue smem_o
  using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;

  using TmemAllocator = cute::TMEM::Allocator1Sm;

  struct SharedStorage {
    // FA4 style: mainloop and epilogue are independent (no union)
    MainloopTensorStorage mainloop;
    typename CollectiveEpilogue::TensorStorage epilogue;

    struct PipelineStorage {
      alignas(16) typename CollectiveMainloop::PipelineQ::SharedStorage load_q;
      alignas(16) typename CollectiveMainloop::PipelineKV::SharedStorage load_kv;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage mma_s0;
      alignas(16) typename CollectiveMainloop::PipelineS::SharedStorage mma_s1;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage s0_corr;
      alignas(16) typename CollectiveMainloop::PipelineC::SharedStorage s1_corr;
      alignas(16) typename CollectiveMainloop::PipelineO::SharedStorage mma_corr;
      alignas(16) typename CollectiveMainloop::PipelineE::SharedStorage corr_epi;
    } pipelines;

    uint32_t tmem_base_ptr;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr bool IsSplitKV = CollectiveMainloop::IsSplitKV;

  using SplitKVParams = typename CollectiveMainloop::SplitKVParams;

  struct ArgumentsBase {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    typename TileScheduler::Arguments tile_scheduler;
    cutlass::KernelHardwareInfo hw_info;
  };

  struct ArgumentsSplitKV : ArgumentsBase {
    SplitKVParams split_kv{};
  };

  using Arguments = std::conditional_t<IsSplitKV, ArgumentsSplitKV, ArgumentsBase>;

  struct ParamsBase {
    ProblemShape problem_shape;
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    typename TileScheduler::Params tile_scheduler;
  };

  struct ParamsSplitKV : ParamsBase {
    SplitKVParams split_kv;
  };

  using Params = std::conditional_t<IsSplitKV, ParamsSplitKV, ParamsBase>;

  static const int MinBlocksPerMultiprocessor = 1;
  static const int MaxThreadsPerBlock = NumWarps * cutlass::NumThreadsPerWarp;
  using ArchTag = cutlass::arch::Sm100;

  static size_t get_workspace_size(Arguments const& args) { return 0; }
  static cutlass::Status initialize_workspace(Arguments const&, void*, cudaStream_t) {
    return cutlass::Status::kSuccess;
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.tile_scheduler);
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    if constexpr (IsSplitKV) {
      return Params{
          {args.problem_shape,
           CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
           CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
           TileScheduler::to_underlying_arguments(args.tile_scheduler, args.hw_info)},
          args.split_kv};
    } else {
      return Params{
          args.problem_shape,
          CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
          CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
          TileScheduler::to_underlying_arguments(args.tile_scheduler, args.hw_info)};
    }
  }

  CUTLASS_DEVICE auto apply_batch(const Params& params, ProblemShape const& problem_shape,
                                  int batch_idx) {
    return apply_variable_length(params.problem_shape, batch_idx);
  }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;" ::: "memory");  // t0fix (Indexer): PDL-launched; wait before any input read
#endif

    GPU_TRACE_INIT;


    int warp_idx = cutlass::canonical_warp_idx_sync();
    auto role = warp_idx_to_WarpRole(warp_idx);
    uint32_t lane_predicate = cute::elect_one_sync();

    GET_GPU_TRACE(role == WarpRole::MMA || role == WarpRole::Load);
    {GPU_TRACE_SCOPE(KERNEL_LAUNCHED);}
    TileScheduler tile_scheduler{params.tile_scheduler};
    
#ifdef SM_TIMING_ENABLED
    long long __sm_start_clock = clock64();
#endif

    if (role == WarpRole::Load && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);

    typename CollectiveMainloop::PipelineQ::Params pipeline_load_q_params;
    if (role == WarpRole::Load) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Producer;
    }
    if (role == WarpRole::MMA) {
      pipeline_load_q_params.role = CollectiveMainloop::PipelineQ::ThreadCategory::Consumer;
    }
    pipeline_load_q_params.is_leader = lane_predicate && (role == WarpRole::Load);
    pipeline_load_q_params.transaction_bytes = CollectiveMainloop::TransactionBytesLoadQ;
    pipeline_load_q_params.initializing_warp = 1;
    typename CollectiveMainloop::PipelineQ pipeline_load_q(
        shared_storage.pipelines.load_q, pipeline_load_q_params, ClusterShape{}, cute::true_type{},
        /*mask calc*/ cute::false_type{});

    typename CollectiveMainloop::PipelineKV::Params pipeline_load_kv_params;
    if (role == WarpRole::Load) {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Producer;
    }
    if (role == WarpRole::MMA) {
      pipeline_load_kv_params.role = CollectiveMainloop::PipelineKV::ThreadCategory::Consumer;
    }
    pipeline_load_kv_params.is_leader = lane_predicate && (role == WarpRole::Load);
    pipeline_load_kv_params.transaction_bytes = CollectiveMainloop::TransactionBytesLoadKV;
    pipeline_load_kv_params.initializing_warp = 1;
    typename CollectiveMainloop::PipelineKV pipeline_load_kv(
        shared_storage.pipelines.load_kv, pipeline_load_kv_params, ClusterShape{},
        /*barrier init*/ cute::true_type{}, /*mask calc*/ cute::false_type{});
    
    __syncthreads();

    pipeline_load_q.init_masks(ClusterShape{});
    pipeline_load_kv.init_masks(ClusterShape{});

    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_consumer_state;
    typename CollectiveMainloop::PipelineQ::PipelineState pipeline_load_q_producer_state =
        cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineQ>();

    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_consumer_state;
    typename CollectiveMainloop::PipelineKV::PipelineState pipeline_load_kv_producer_state =
        cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineKV>();

    CollectiveMainloop mainloop;

    if (role == WarpRole::Load) {
      warpgroup_reg_set<NumRegsOther>();

      int work_idx = 0;

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape =
            apply_batch(params, params.problem_shape, get<2, 1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {  // kv_len == 0
          work_idx++;
          continue;
        }

        if constexpr (IsSplitKV) {
          int wp = tile_scheduler.get_work_ptr();
          int kv_tile_begin = SPLIT_KV_BEGIN(params.split_kv, wp);
          int kv_tile_end = SPLIT_KV_END(params.split_kv, wp);
          if (CollectiveMainloop::get_effective_trip_count(
                  blk_coord, logical_problem_shape,
                  kv_tile_begin, kv_tile_end, params.mainloop.load.kv_block_num) <= 0) { work_idx++; continue; }
          mainloop.load(blk_coord, logical_problem_shape, params.mainloop, params.problem_shape,
                        work_idx, shared_storage.mainloop,
                        pipeline_load_q, pipeline_load_q_producer_state,
                        pipeline_load_kv, pipeline_load_kv_producer_state
                        #ifdef GPU_TRACE_ENABLED
                          , _gt_rec
                        #endif
                        , kv_tile_begin, kv_tile_end
                      );
        } else {
          mainloop.load(blk_coord, logical_problem_shape, params.mainloop, params.problem_shape,
                        work_idx, shared_storage.mainloop,
                        pipeline_load_q, pipeline_load_q_producer_state,
                        pipeline_load_kv, pipeline_load_kv_producer_state
                        #ifdef GPU_TRACE_ENABLED
                          , _gt_rec
                        #endif
                      );
        }

        work_idx++;
      }
    }

    // Dynamic: how many warps per stage are actually needed (based on max Q length)
    int active_warps_per_stage = kNumWarpsSoftmaxPerStage;
    if constexpr (CollectiveMainloop::kEnablePaddingSkip) {
      int active_q = min(params.epilogue.max_qo_len, (int)get<0>(TileShape{}));
      active_warps_per_stage = min((int)kNumWarpsSoftmaxPerStage, (active_q + 31) / 32);
      if (active_warps_per_stage < 1) active_warps_per_stage = 1;
    }
    int softmax_arv_count = warp_uniform(active_warps_per_stage * cutlass::NumThreadsPerWarp);

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s0_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Softmax0) {
      pipeline_mma_s0_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s0_params.consumer_arv_count = softmax_arv_count;
    pipeline_mma_s0_params.initializing_warp = 1;
    typename CollectiveMainloop::PipelineS pipeline_mma_s0(
        shared_storage.pipelines.mma_s0, pipeline_mma_s0_params, ClusterShape{},
        /*barrier init*/ cute::true_type{}, /*mask calc*/ cute::false_type{});

    typename CollectiveMainloop::PipelineS::Params pipeline_mma_s1_params;
    if (role == WarpRole::MMA) {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Producer;
    }
    if (role == WarpRole::Softmax1) {
      pipeline_mma_s1_params.role = CollectiveMainloop::PipelineS::ThreadCategory::Consumer;
    }
    pipeline_mma_s1_params.consumer_arv_count = softmax_arv_count;
    pipeline_mma_s1_params.initializing_warp = 1;
    typename CollectiveMainloop::PipelineS pipeline_mma_s1(
        shared_storage.pipelines.mma_s1, pipeline_mma_s1_params, ClusterShape{},
        /*barrier init*/ cute::true_type{}, /*mask calc*/ cute::false_type{});

    typename CollectiveMainloop::PipelineC::Params pipeline_s0_corr_params;
    if (role == WarpRole::Softmax0) {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_s0_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s0_corr_params.producer_arv_count = softmax_arv_count;
    pipeline_s0_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    pipeline_s0_corr_params.initializing_warp = 1;
    typename CollectiveMainloop::PipelineC pipeline_s0_corr(shared_storage.pipelines.s0_corr,
                                                            pipeline_s0_corr_params,
                                                            /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::PipelineC::Params pipeline_s1_corr_params;
    if (role == WarpRole::Softmax1) {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Producer;
    }
    if (role == WarpRole::Correction) {
      pipeline_s1_corr_params.role = CollectiveMainloop::PipelineC::ThreadCategory::Consumer;
    }
    pipeline_s1_corr_params.producer_arv_count = softmax_arv_count;
    pipeline_s1_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    pipeline_s1_corr_params.initializing_warp = 1;
    typename CollectiveMainloop::PipelineC pipeline_s1_corr(shared_storage.pipelines.s1_corr,
                                                            pipeline_s1_corr_params,
                                                            /*barrier init*/ cute::true_type{});

    typename CollectiveMainloop::PipelineO::Params pipeline_mma_corr_params;
    if (role == WarpRole::MMA) {
      if constexpr (CollectiveMainloop::kNeedMaxScore) {
        // consumer role to wait for writing out of MaxScore
        pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Consumer;
      }
      else{
        pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Producer;
      }
    }
    if (role == WarpRole::Correction) {
        pipeline_mma_corr_params.role = CollectiveMainloop::PipelineO::ThreadCategory::Consumer;
    }
    pipeline_mma_corr_params.consumer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    pipeline_mma_corr_params.initializing_warp = 1;
    typename CollectiveMainloop::PipelineO pipeline_corr(
        shared_storage.pipelines.mma_corr, pipeline_mma_corr_params);

    typename CollectiveMainloop::PipelineE::Params pipeline_corr_epi_params;
    if (role == WarpRole::Correction) {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Producer;
    }
    if (role == WarpRole::Epilogue) {
      pipeline_corr_epi_params.role = CollectiveMainloop::PipelineE::ThreadCategory::Consumer;
    }
    pipeline_corr_epi_params.producer_arv_count = NumWarpsCorrection * cutlass::NumThreadsPerWarp;
    pipeline_corr_epi_params.consumer_arv_count = NumWarpsEpilogue * cutlass::NumThreadsPerWarp;
    pipeline_corr_epi_params.initializing_warp = 1;
    typename CollectiveMainloop::PipelineE pipeline_corr_epi(shared_storage.pipelines.corr_epi,
                                                             pipeline_corr_epi_params,
                                                             /*barrier init*/ cute::true_type{});

    if (role != WarpRole::Load) {
      if constexpr (kSingleSoftmaxWarpGroup)
        asm volatile("bar.sync 8, 352;");
      else
        asm volatile("bar.sync 8, 480;");

      pipeline_mma_s0.init_masks(ClusterShape{});
      pipeline_mma_s1.init_masks(ClusterShape{});
    }

    TmemAllocator tmem_allocator;
    bool has_work = tile_scheduler.is_valid();

    if (role == WarpRole::Epilogue && lane_predicate) {
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }


    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s0_producer_state =
        cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_consumer_state;
    typename CollectiveMainloop::PipelineS::PipelineState pipeline_mma_s1_producer_state =
        cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineS>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s0_corr_producer_state =
        cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_consumer_state;
    typename CollectiveMainloop::PipelineC::PipelineState pipeline_s1_corr_producer_state =
        cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineC>();

    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_consumer_state;
    typename CollectiveMainloop::PipelineE::PipelineState pipeline_corr_epi_producer_state =
        cutlass::make_producer_start_state<typename CollectiveMainloop::PipelineE>();

    RELEASE_GPU_TRACE;

    CollectiveEpilogue epilogue{params.epilogue};

    if (role == WarpRole::Softmax0 || role == WarpRole::Softmax1) {
      warpgroup_reg_set<NumRegsSoftmax>();

      // Idle warps within a stage: early return, don't touch pipelines
      int warp_in_stage = cutlass::canonical_warp_idx_sync() % kNumWarpsSoftmaxPerStage;
      if (warp_in_stage >= active_warps_per_stage) return;

      bool is_softmax_0 = (role == WarpRole::Softmax0);
      int q_offset_correction = 0;
      if constexpr (kSingleSoftmaxWarpGroup) {
        if (!is_softmax_0) q_offset_correction = -64;
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape =
            apply_batch(params, params.problem_shape, get<2, 1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        if constexpr (IsSplitKV) {
          int wp = tile_scheduler.get_work_ptr();
          int kv_tile_begin = SPLIT_KV_BEGIN(params.split_kv, wp);
          int kv_tile_end = SPLIT_KV_END(params.split_kv, wp);
          if (CollectiveMainloop::get_effective_trip_count(
                  blk_coord, logical_problem_shape,
                  kv_tile_begin, kv_tile_end, params.mainloop.load.kv_block_num) <= 0) {
            continue;
          }
          mainloop.template softmax<kSingleSoftmaxWarpGroup>(
              is_softmax_0 ? 0 : 1, blk_coord, params.mainloop,
              params.problem_shape, logical_problem_shape, epilogue,
              is_softmax_0 ? pipeline_mma_s0 : pipeline_mma_s1,
              is_softmax_0 ? pipeline_mma_s0_consumer_state : pipeline_mma_s1_consumer_state,
              is_softmax_0 ? pipeline_s0_corr : pipeline_s1_corr,
              is_softmax_0 ? pipeline_s0_corr_producer_state : pipeline_s1_corr_producer_state,
              kv_tile_begin, kv_tile_end, q_offset_correction);
        } else {
          mainloop.template softmax<kSingleSoftmaxWarpGroup>(
              is_softmax_0 ? 0 : 1, blk_coord, params.mainloop,
              params.problem_shape, logical_problem_shape, epilogue,
              is_softmax_0 ? pipeline_mma_s0 : pipeline_mma_s1,
              is_softmax_0 ? pipeline_mma_s0_consumer_state : pipeline_mma_s1_consumer_state,
              is_softmax_0 ? pipeline_s0_corr : pipeline_s1_corr,
              is_softmax_0 ? pipeline_s0_corr_producer_state : pipeline_s1_corr_producer_state,
              0, INT_MAX, q_offset_correction);
        }
      }
    } else if (role == WarpRole::Correction) {
      cutlass::arch::warpgroup_reg_dealloc<NumRegsCorrection>();
      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape =
            apply_batch(params, params.problem_shape, get<2, 1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {
          mainloop.correction_empty(blk_coord, params.mainloop, logical_problem_shape,
                                    params.problem_shape, shared_storage.epilogue,
                                    pipeline_corr_epi, pipeline_corr_epi_producer_state, epilogue);
          continue;
        }

        if constexpr (IsSplitKV) {
          int wp = tile_scheduler.get_work_ptr();
          int kv_tile_begin = SPLIT_KV_BEGIN(params.split_kv, wp);
          int kv_tile_end = SPLIT_KV_END(params.split_kv, wp);
          int kv_split_idx = SPLIT_KV_SPLIT(params.split_kv, wp);
          if (CollectiveMainloop::get_effective_trip_count(
                  blk_coord, logical_problem_shape,
                  kv_tile_begin, kv_tile_end, params.mainloop.load.kv_block_num) <= 0) {
            mainloop.correction_empty(blk_coord, params.mainloop, logical_problem_shape,
                                      params.problem_shape, shared_storage.epilogue,
                                      pipeline_corr_epi, pipeline_corr_epi_producer_state, epilogue);
            continue;
          }
          mainloop.template correction<kSingleSoftmaxWarpGroup>(blk_coord, params.mainloop, params.problem_shape,
                              logical_problem_shape, shared_storage.epilogue, pipeline_s0_corr,
                              pipeline_s0_corr_consumer_state, pipeline_s1_corr,
                              pipeline_s1_corr_consumer_state, pipeline_corr,
                              pipeline_corr_epi, pipeline_corr_epi_producer_state, epilogue,
                              kv_tile_begin, kv_tile_end, kv_split_idx, params.split_kv);
        } else {
          mainloop.template correction<kSingleSoftmaxWarpGroup>(blk_coord, params.mainloop, params.problem_shape,
                              logical_problem_shape, shared_storage.epilogue, pipeline_s0_corr,
                              pipeline_s0_corr_consumer_state, pipeline_s1_corr,
                              pipeline_s1_corr_consumer_state, pipeline_corr,
                              pipeline_corr_epi, pipeline_corr_epi_producer_state, epilogue);
        }
      }

      if constexpr (NumWarpsEpilogue == 0) {
        static_assert(NumWarpsCorrection == 1);
        if (has_work) {
          uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
          tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
        }
      }

    } else if (role == WarpRole::MMA) {
      warpgroup_reg_set<NumRegsOther>();
      // Only allocate TMEM if this SM has work to do
      if (has_work) {
        tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                                &shared_storage.tmem_base_ptr);
      }

      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape =
            apply_batch(params, params.problem_shape, get<2, 1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        if (get<1>(logical_problem_shape) == 0) {
          continue;
        }

        if constexpr (IsSplitKV) {
          int wp = tile_scheduler.get_work_ptr();
          int kv_tile_begin = SPLIT_KV_BEGIN(params.split_kv, wp);
          int kv_tile_end = SPLIT_KV_END(params.split_kv, wp);
          if (CollectiveMainloop::get_effective_trip_count(
                  blk_coord, logical_problem_shape,
                  kv_tile_begin, kv_tile_end, params.mainloop.load.kv_block_num) <= 0) {
            continue;
          }
          mainloop.template mma<kSingleSoftmaxWarpGroup>(
              blk_coord, params.mainloop, logical_problem_shape, shared_storage.mainloop,
              pipeline_load_q, pipeline_load_q_consumer_state, pipeline_load_kv,
              pipeline_load_kv_consumer_state,
              pipeline_mma_s0, pipeline_mma_s0_producer_state, pipeline_mma_s1,
              pipeline_mma_s1_producer_state, pipeline_corr,
              kv_tile_begin, kv_tile_end);
        } else {
          mainloop.template mma<kSingleSoftmaxWarpGroup>(
              blk_coord, params.mainloop, logical_problem_shape, shared_storage.mainloop,
              pipeline_load_q, pipeline_load_q_consumer_state, pipeline_load_kv,
              pipeline_load_kv_consumer_state,
              pipeline_mma_s0, pipeline_mma_s0_producer_state, pipeline_mma_s1,
              pipeline_mma_s1_producer_state, pipeline_corr);
        }
      }
    } else if (role == WarpRole::Epilogue) {
      warpgroup_reg_set<NumRegsOther>();

      int work_idx = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (; tile_scheduler.is_valid(); ++tile_scheduler) {
        auto blk_coord = tile_scheduler.get_block_coord();

        auto logical_problem_shape =
            apply_batch(params, params.problem_shape, get<2, 1>(blk_coord));

        if (get<0>(blk_coord) * get<0>(TileShape{}) >= get<0>(logical_problem_shape)) {
          continue;
        }

        if constexpr (IsSplitKV) {
          int wp = tile_scheduler.get_work_ptr();
          int kv_split_idx = SPLIT_KV_SPLIT(params.split_kv, wp);
          epilogue.store(blk_coord, logical_problem_shape, params.epilogue, params.problem_shape,
                         shared_storage.epilogue, pipeline_corr_epi,
                         pipeline_corr_epi_consumer_state, kv_split_idx,
                         params.split_kv.total_qo_len,
                         params.split_kv.ptr_lse_accum,
                         params.split_kv.num_qo_heads);
        } else {
          epilogue.store(blk_coord, logical_problem_shape, params.epilogue, params.problem_shape,
                         shared_storage.epilogue, pipeline_corr_epi,
                         pipeline_corr_epi_consumer_state);
        }

        work_idx++;
      }

      static_assert(NumWarpsEpilogue <= 1);
      if constexpr (NumWarpsEpilogue == 1) {
        if (has_work) {
          uint32_t free_stage_ptr = shared_storage.tmem_base_ptr;
          tmem_allocator.free(free_stage_ptr, TmemAllocator::Sm100TmemCapacityColumns);
        }
      }

    } else if (role == WarpRole::Empty) {
      warpgroup_reg_set<NumRegsOther>();

      /* no-op, donate regs and exit */
    }

    GPU_TRACE_EXIT;
#ifdef SM_TIMING_ENABLED
    long long __sm_end_clock = clock64();
    long long __sm_duration = __sm_end_clock - __sm_start_clock;
    if (threadIdx.x % 32 == 0 && role == WarpRole::Epilogue) {
      TileScheduler tile_scheduler_out{params.tile_scheduler};
      int num_tiles = tile_scheduler_out.work_ptr_end - tile_scheduler_out.work_ptr;
      printf("SM_TIMING block=%d duration=%lld start=%lld end=%lld num_tiles=%d\n",
             blockIdx.x, __sm_duration, __sm_start_clock, __sm_end_clock, num_tiles);
      for (; tile_scheduler_out.is_valid(); ++tile_scheduler_out) {
        auto blk_coord = tile_scheduler_out.get_block_coord();
        auto logical_problem_shape =
            apply_batch(params, params.problem_shape, get<2, 1>(blk_coord));
        int q_len = get<0>(logical_problem_shape);
        int kv_len = get<1>(logical_problem_shape);
        int kv_iters;
        if constexpr (IsSplitKV) {
          int wp = tile_scheduler_out.get_work_ptr();
          int kb = SPLIT_KV_BEGIN(params.split_kv, wp);
          int ke = SPLIT_KV_END(params.split_kv, wp);
          kv_iters = CollectiveMainloop::get_effective_trip_count(
              blk_coord, logical_problem_shape, kb, ke, params.mainloop.load.kv_block_num);
        } else {
          kv_iters = CollectiveMainloop::get_full_trip_count(
              blk_coord, logical_problem_shape, params.mainloop.load.kv_block_num);
        }
        printf("  SM_TILE block=%d qo_tile=%d batch=%d head=%d q_len=%d kv_len=%d kv_iters=%d\n",
               blockIdx.x, get<0>(blk_coord), get<2, 1>(blk_coord), get<2, 0>(blk_coord),
               q_len, kv_len, kv_iters);
      }
    }
#endif
  }
};

}  // namespace cutlass::fmha::kernel
