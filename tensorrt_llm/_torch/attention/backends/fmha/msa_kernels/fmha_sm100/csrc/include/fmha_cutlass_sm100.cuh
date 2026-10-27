// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>

#include "allocator.h"
#include "fmha_fusion.hpp"
#include "gpu_trace.h"
#include "sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "fmha.hpp"
#include "fmha_tile_scheduler.hpp"
#include "sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "sm100_fmha_reduction.hpp"

namespace flashinfer {

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::device;

struct PackGQAUnpackParams {
  float* max_score_direct = nullptr;
  int total_qo_len_orig = 0;
  void* o_direct = nullptr;
  int num_qo_heads_orig = 0;
  bool tma_direct_o_enabled = false;
#ifdef FMHA_GMEM_BOUNDS_CHECK
  GmemBounds gmem_bounds = {};
#endif
};

template <typename DTypeIn, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask,
          class ThreadShape = Shape<_2, _1, _1>,
          bool IsSplitKV = false,
          bool SingleSoftmaxWarpGroup = false,
          int KVPageSize = -1,
          cutlass::fmha::collective::SparseAttnMode kSparseAttnMode =
              cutlass::fmha::collective::SparseAttnMode::Off,
          bool UnitSelectorColumn = false>
struct FwdRunner {
  using Element = DTypeIn;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = DTypeOut;

  // Q K D ((H_R, H_KV), B)
  using ProblemShapeVarlen =
      cute::tuple<VariableLength, VariableLength, int, cute::tuple<cute::tuple<int, int>, int>,
                  PerBatchOffset>;

  using StrideQ = cute::tuple<int, _1, cute::tuple<int, int>>;  // Q D (H_G H_R)
  // Paged: 4-mode flat (entry, dim, head, page). Non-paged: 3-mode with GQA broadcast
  using StrideK = std::conditional_t<(KVPageSize > 0),
      cute::tuple<int, _1, int, int>,
      cute::tuple<int, _1, cute::tuple<_0, int>>>;
  using StrideV = std::conditional_t<(KVPageSize > 0),
      cute::tuple<_1, int, int, int>,
      cute::tuple<_1, int, cute::tuple<_0, int>>>;
  // NOTE(Zihao): use markus's trick for tma store
  using StrideO =
      cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;  // Q D (H_G H_R) CUMULATIVE_Q
  using StrideLSE = cute::tuple<int, cute::tuple<_1, int>>;           // Q (H_G H_R)

  // NumQStages: 2 for Q-split (2,1,1), 1 for K-split (1,2,1)
  static constexpr int NumQStages = cute::size<0>(ThreadShape{});
  static constexpr bool bNeedOutput = kSparseAttnMode != cutlass::fmha::collective::SparseAttnMode::OnlyScore;
  static constexpr int kPackFactor = cutlass::fmha::collective::pack_factor_of<ActiveMask>::value;

  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShapeQK, TileShapePV, StrideQ,
      StrideK, StrideV, ActiveMask, ThreadShape, IsSplitKV, KVPageSize, kSparseAttnMode,
      UnitSelectorColumn>;
  using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
      ElementOut, ElementAccumulatorPV, typename Mainloop::TileShapePV, NumQStages, IsSplitKV, bNeedOutput, kPackFactor>;
  using Operation =
      cutlass::fmha::device::FMHA<cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          ProblemShapeVarlen, Mainloop, Epilogue,
          cutlass::fmha::kernel::HostPrecomputedTileScheduler,
          cutlass::fmha::kernel::Sm100FmhaCtxKernelWarpspecializedSchedule<SingleSoftmaxWarpGroup>>>;
  using LayoutQ = typename Mainloop::LayoutQ;
  using LayoutK = typename Mainloop::LayoutK;
  using LayoutV = typename Mainloop::LayoutV;
  using LayoutO = typename Epilogue::LayoutO;
  using LayoutLSE = typename Epilogue::LayoutLSE;

  static cudaError_t run(void* workspace_buffer, DTypeIn* q, DTypeIn* k, DTypeIn* v,
                         IdType* qo_segment_lens, IdType* kv_segment_lens,
                         IdType* qo_segment_offsets, IdType* kv_segment_offsets,
                         uint64_t* packed_work_range, uint64_t* packed_work_info,
                         DTypeOut* o, int mask_mode_code,
                         float sm_scale, float q_scale, float k_scale, float v_scale,
                         float o_scale, int num_qo_heads, int num_kv_heads, int head_dim_qk,
                         int head_dim_vo, int q_stride_n, int q_stride_h, int k_stride_n,
                         int k_stride_h, int v_stride_n, int v_stride_h, int batch_size,
                         int total_qo_len, int total_kv_len, int max_qo_len, int* qo_offsets,
                         cudaStream_t stream,
                         int num_kv_splits = 1,
                         int* kv_tile_begin_indices = nullptr,
                         int* kv_tile_end_indices = nullptr,
                         int* kv_split_indices = nullptr,
                         float* ptr_lse_accum = nullptr,
                         int* kv_indices = nullptr,
                         int* kv_page_indptr = nullptr,
                         int total_page_num = 0,
                         float* maybe_max_score = nullptr,
                         int max_k_tiles = 0,
                         int* kv_block_indexes = nullptr,
                         int kv_block_num = 0,
                         int64_t kv_block_stride_n = 0,
                         int64_t kv_block_stride_h = 0,
                         int64_t kv_block_stride_b = 1,
                         int pack_factor = 1,
                         int q_stride_n_original = 0,
                         int q_stride_h_original = 0,
                         int h_r_original = 0,
                         PackGQAUnpackParams pack_gqa = {},
                         int num_ctas = 0) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = (num_ctas > 0)
        ? num_ctas
        : cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    [[maybe_unused]] auto sched_args = typename cutlass::fmha::kernel::HostPrecomputedTileScheduler::Arguments{
        packed_work_range, packed_work_info};
#ifdef FMHA_GMEM_BOUNDS_CHECK
    sched_args.packed_work_info_size = pack_gqa.gmem_bounds.packed_work_info_size;
#endif

    StrideQ stride_Q;
    StrideK stride_K;
    StrideV stride_V;
    StrideO stride_O;
    StrideLSE stride_LSE;

    int h_r = num_qo_heads / num_kv_heads;
    assert(num_qo_heads % num_kv_heads == 0);
    ProblemShapeVarlen problem_shape = cute::make_tuple(
        VariableLength{qo_segment_lens, qo_segment_offsets},
        VariableLength{kv_segment_lens, kv_segment_offsets}, head_dim_qk,
        cute::make_tuple(cute::make_tuple(h_r, num_kv_heads), batch_size),
        PerBatchOffset{qo_offsets});

    stride_Q = make_stride(q_stride_n, _1{}, make_stride(q_stride_h, h_r * q_stride_h));
    stride_O = make_stride(
        num_qo_heads * head_dim_vo, _1{},
        make_stride(make_stride(head_dim_vo, h_r * head_dim_vo), num_qo_heads * head_dim_vo));
    stride_LSE = make_stride(num_qo_heads, make_stride(_1{}, h_r));

    if constexpr (KVPageSize > 0) {
      int k_stride_page = k_stride_n;
      int k_stride_head = k_stride_h;
      int v_stride_page = v_stride_n;
      int v_stride_head = v_stride_h;
      stride_K = make_stride(head_dim_qk, _1{}, k_stride_head, k_stride_page);
      stride_V = make_stride(_1{}, head_dim_vo, v_stride_head, v_stride_page);
      auto shape_K = make_shape(KVPageSize, head_dim_qk, num_kv_heads, total_page_num);
      auto shape_V = make_shape(head_dim_vo, KVPageSize, num_kv_heads, total_page_num);

      LayoutK layout_K = make_layout(shape_K, stride_K);
      LayoutV layout_V = make_layout(shape_V, stride_V);
      auto shape_Q = make_shape(total_qo_len, head_dim_qk, make_shape(h_r, num_kv_heads));
      auto shape_O = make_shape(max_qo_len, head_dim_vo,
                                make_shape(make_shape(h_r, num_kv_heads),
                                           max_qo_len + total_qo_len * num_kv_splits));
      LayoutQ layout_Q = make_layout(shape_Q, stride_Q);
      LayoutO layout_O = make_layout(shape_O, stride_O);

      auto shape_MaxScore = make_shape(total_qo_len, make_shape(h_r, num_kv_heads), max_k_tiles);
      auto stride_MaxScore = make_stride(
          _1{},
          make_stride(total_qo_len * max_k_tiles, h_r * total_qo_len * max_k_tiles),
          total_qo_len);
      typename Epilogue::LayoutMaxScore layout_MaxScore = make_layout(shape_MaxScore, stride_MaxScore);

      typename Epilogue::Arguments epi_args;
      epi_args.ptr_O = o ? (o - max_qo_len * get<0>(stride_O)) : nullptr;
      epi_args.layout_O = layout_O;
      epi_args.max_qo_len = max_qo_len;
      epi_args.ptr_MaxScore = maybe_max_score;
      epi_args.layout_MaxScore = layout_MaxScore;
      epi_args.ptr_MaxScore_direct = pack_gqa.max_score_direct;
      epi_args.total_qo_len_orig = pack_gqa.total_qo_len_orig;
#ifdef FMHA_GMEM_BOUNDS_CHECK
      epi_args.max_score_numel = pack_gqa.gmem_bounds.max_score_numel;
#endif
      // Direct-O unpack is incompatible with split-KV: epilogue would overwrite
      // user O with per-split partials, racing with the reduction kernel.
      if constexpr (IsSplitKV) {
        epi_args.ptr_O_direct = nullptr;
      } else {
        epi_args.ptr_O_direct = static_cast<ElementOut*>(pack_gqa.o_direct);
        if (epi_args.ptr_O_direct) {
          epi_args.ptr_O = nullptr;
        }
      }
      epi_args.num_qo_heads_orig = pack_gqa.num_qo_heads_orig;
      epi_args.head_dim_vo = head_dim_vo;
      epi_args.remaining_h_r = h_r;
      // TMA fused direct-O unpack: enabled only when host eligibility check passes
      // AND non-split-KV path. max_qo_len_orig is derived in epilogue from
      // max_qo_len / kPackFactor_ (compile-time constant).
      epi_args.qo_len_uniform = pack_gqa.tma_direct_o_enabled;

      typename Operation::Arguments arguments;
      if constexpr (IsSplitKV) {
        arguments = {
          {problem_shape,
           {{.ptr_Q = q,
             .layout_Q = layout_Q,
             .ptr_K = k,
             .layout_K = layout_K,
             .ptr_V = v,
             .layout_V = layout_V,
             .kv_indices = kv_indices,
             .kv_page_indptr = kv_page_indptr,
             .kv_block_indexes = kv_block_indexes,
             .kv_block_num = kv_block_num,
             .kv_block_stride_n = kv_block_stride_n,
             .kv_block_stride_h = kv_block_stride_h,
             .kv_block_stride_b = kv_block_stride_b,
#ifdef FMHA_GMEM_BOUNDS_CHECK
             .kv_page_indptr_size = pack_gqa.gmem_bounds.kv_page_indptr_size,
             .kv_indices_size = pack_gqa.gmem_bounds.kv_indices_size,
             .kv_block_indexes_span = pack_gqa.gmem_bounds.kv_block_indexes_span,
#endif
             .pack_factor = pack_factor,
             .q_stride_n_original = q_stride_n_original,
             .q_stride_h_original = q_stride_h_original,
             .h_r_original = h_r_original},
            sm_scale, q_scale, k_scale, v_scale, o_scale},
           epi_args,
           sched_args,
           hw_info},
          {static_cast<float>(v_scale),
           static_cast<float>(v_scale * o_scale),
           ptr_lse_accum,
           kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
           total_qo_len, num_qo_heads
#ifdef FMHA_GMEM_BOUNDS_CHECK
           , pack_gqa.gmem_bounds.split_kv_size
#endif
           }};
      } else {
        arguments = {
          problem_shape,
          {{.ptr_Q = q,
             .layout_Q = layout_Q,
             .ptr_K = k,
             .layout_K = layout_K,
             .ptr_V = v,
             .layout_V = layout_V,
             .kv_indices = kv_indices,
             .kv_page_indptr = kv_page_indptr,
             .kv_block_indexes = kv_block_indexes,
             .kv_block_num = kv_block_num,
             .kv_block_stride_n = kv_block_stride_n,
             .kv_block_stride_h = kv_block_stride_h,
             .kv_block_stride_b = kv_block_stride_b,
#ifdef FMHA_GMEM_BOUNDS_CHECK
             .kv_page_indptr_size = pack_gqa.gmem_bounds.kv_page_indptr_size,
             .kv_indices_size = pack_gqa.gmem_bounds.kv_indices_size,
             .kv_block_indexes_span = pack_gqa.gmem_bounds.kv_block_indexes_span,
#endif
             .pack_factor = pack_factor,
             .q_stride_n_original = q_stride_n_original,
             .q_stride_h_original = q_stride_h_original,
             .h_r_original = h_r_original},
           sm_scale, q_scale, k_scale, v_scale, o_scale},
          epi_args,
          sched_args,
          hw_info};
      }

      Operation op;
      size_t workspace_size = Operation::get_workspace_size(arguments);
      AlignedAllocator allocator(workspace_buffer, workspace_size);
      uint8_t* workspace_ptr =
          allocator.aligned_alloc<uint8_t>(workspace_size, 16, "fmha_cutlass_sm100_workspace");

      cutlass::Status status = op.can_implement(arguments);
      if (status != cutlass::Status::kSuccess) {
        cudaError_t err = cudaGetLastError();
        std::cerr << "This kernel is not supported. Last CUDA error is: "
                  << cudaGetErrorString(err) << std::endl;
        return err != cudaSuccess ? err : cudaErrorNotSupported;
      }
      status = op.initialize(arguments, workspace_ptr);
      if (status != cutlass::Status::kSuccess) {
        cudaError_t err = cudaGetLastError();
        std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(err) << std::endl;
        return err != cudaSuccess ? err : cudaErrorLaunchFailure;
      }
      GPUTraceParam _gt_param;
      gpu_trace::setup_from_env(_gt_param, stream);
      status = op.run(stream);
      if (status != cutlass::Status::kSuccess) {
        cudaError_t err = cudaGetLastError();
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(err) << std::endl;
        gpu_trace::teardown(_gt_param, stream);
        return err != cudaSuccess ? err : cudaErrorLaunchFailure;
      }
      gpu_trace::teardown(_gt_param, stream);
      return cudaSuccess;
    } else {
      stride_K = make_stride(k_stride_n, _1{}, make_stride(_0{}, k_stride_h));
      stride_V = make_stride(_1{}, v_stride_n, make_stride(_0{}, v_stride_h));

      auto shape_Q = make_shape(total_qo_len, head_dim_qk, make_shape(h_r, num_kv_heads));
      auto shape_O = make_shape(max_qo_len, head_dim_vo,
                                make_shape(make_shape(h_r, num_kv_heads),
                                           max_qo_len + total_qo_len * num_kv_splits));
      auto shape_K = make_shape(total_kv_len, head_dim_qk, make_shape(h_r, num_kv_heads));
      auto shape_V = make_shape(head_dim_vo, total_kv_len, make_shape(h_r, num_kv_heads));

      LayoutQ layout_Q = make_layout(shape_Q, stride_Q);
      LayoutK layout_K = make_layout(shape_K, stride_K);
      LayoutV layout_V = make_layout(shape_V, stride_V);
      LayoutO layout_O = make_layout(shape_O, stride_O);

      auto shape_MaxScore = make_shape(total_qo_len, make_shape(h_r, num_kv_heads), max_k_tiles);
      auto stride_MaxScore = make_stride(
          _1{},
          make_stride(total_qo_len * max_k_tiles, h_r * total_qo_len * max_k_tiles),
          total_qo_len);
      typename Epilogue::LayoutMaxScore layout_MaxScore = make_layout(shape_MaxScore, stride_MaxScore);

      typename Epilogue::Arguments epi_args;
      epi_args.ptr_O = o ? (o - max_qo_len * get<0>(stride_O)) : nullptr;
      epi_args.layout_O = layout_O;
      epi_args.max_qo_len = max_qo_len;
      epi_args.ptr_MaxScore = maybe_max_score;
      epi_args.layout_MaxScore = layout_MaxScore;
      epi_args.ptr_MaxScore_direct = pack_gqa.max_score_direct;
      epi_args.total_qo_len_orig = pack_gqa.total_qo_len_orig;
#ifdef FMHA_GMEM_BOUNDS_CHECK
      epi_args.max_score_numel = pack_gqa.gmem_bounds.max_score_numel;
#endif
      // Direct-O unpack is incompatible with split-KV: epilogue would overwrite
      // user O with per-split partials, racing with the reduction kernel.
      if constexpr (IsSplitKV) {
        epi_args.ptr_O_direct = nullptr;
      } else {
        epi_args.ptr_O_direct = static_cast<ElementOut*>(pack_gqa.o_direct);
        if (epi_args.ptr_O_direct) {
          epi_args.ptr_O = nullptr;
        }
      }
      epi_args.num_qo_heads_orig = pack_gqa.num_qo_heads_orig;
      epi_args.head_dim_vo = head_dim_vo;
      epi_args.remaining_h_r = h_r;
      // TMA fused direct-O unpack: enabled only when host eligibility check passes
      // AND non-split-KV path. max_qo_len_orig is derived in epilogue from
      // max_qo_len / kPackFactor_ (compile-time constant).
      epi_args.qo_len_uniform = pack_gqa.tma_direct_o_enabled;

      typename Operation::Arguments arguments;
      if constexpr (IsSplitKV) {
        arguments = {
          {problem_shape,
           {{.ptr_Q = q,
             .layout_Q = layout_Q,
             .ptr_K = k,
             .layout_K = layout_K,
             .ptr_V = v,
             .layout_V = layout_V,
             .kv_indices = nullptr,
             .kv_page_indptr = nullptr,
             .kv_block_indexes = nullptr,
             .kv_block_num = 0,
             .kv_block_stride_n = 0,
             .kv_block_stride_h = 0,
             .kv_block_stride_b = 1,
#ifdef FMHA_GMEM_BOUNDS_CHECK
             .kv_page_indptr_size = 0,
             .kv_indices_size = 0,
             .kv_block_indexes_span = 0,
#endif
             .pack_factor = pack_factor,
             .q_stride_n_original = q_stride_n_original,
             .q_stride_h_original = q_stride_h_original,
             .h_r_original = h_r_original},
            sm_scale, q_scale, k_scale, v_scale, o_scale},
           epi_args,
           sched_args,
           hw_info},
          {static_cast<float>(v_scale),
           static_cast<float>(v_scale * o_scale),
           ptr_lse_accum,
           kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
           total_qo_len, num_qo_heads
#ifdef FMHA_GMEM_BOUNDS_CHECK
           , pack_gqa.gmem_bounds.split_kv_size
#endif
           }};
      } else {
        arguments = {
          problem_shape,
          {{.ptr_Q = q,
             .layout_Q = layout_Q,
             .ptr_K = k,
             .layout_K = layout_K,
             .ptr_V = v,
             .layout_V = layout_V,
             .kv_indices = nullptr,
             .kv_page_indptr = nullptr,
             .kv_block_indexes = nullptr,
             .kv_block_num = 0,
             .kv_block_stride_n = 0,
             .kv_block_stride_h = 0,
             .kv_block_stride_b = 1,
#ifdef FMHA_GMEM_BOUNDS_CHECK
             .kv_page_indptr_size = 0,
             .kv_indices_size = 0,
             .kv_block_indexes_span = 0,
#endif
             .pack_factor = pack_factor,
             .q_stride_n_original = q_stride_n_original,
             .q_stride_h_original = q_stride_h_original,
             .h_r_original = h_r_original},
           sm_scale, q_scale, k_scale, v_scale, o_scale},
          epi_args,
          sched_args,
          hw_info};
      }

      Operation op;
      size_t workspace_size = Operation::get_workspace_size(arguments);
      AlignedAllocator allocator(workspace_buffer, workspace_size);
      uint8_t* workspace_ptr =
          allocator.aligned_alloc<uint8_t>(workspace_size, 16, "fmha_cutlass_sm100_workspace");

      cutlass::Status status = op.can_implement(arguments);
      if (status != cutlass::Status::kSuccess) {
        cudaError_t err = cudaGetLastError();
        std::cerr << "This kernel is not supported. Last CUDA error is: "
                  << cudaGetErrorString(err) << std::endl;
        return err != cudaSuccess ? err : cudaErrorNotSupported;
      }
      status = op.initialize(arguments, workspace_ptr);
      if (status != cutlass::Status::kSuccess) {
        cudaError_t err = cudaGetLastError();
        std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(err) << std::endl;
        return err != cudaSuccess ? err : cudaErrorLaunchFailure;
      }
      GPUTraceParam _gt_param;
      gpu_trace::setup_from_env(_gt_param, stream);
      status = op.run(stream);
      if (status != cutlass::Status::kSuccess) {
        cudaError_t err = cudaGetLastError();
        std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                  << cudaGetErrorString(err) << std::endl;
        gpu_trace::teardown(_gt_param, stream);
        return err != cudaSuccess ? err : cudaErrorLaunchFailure;
      }
      gpu_trace::teardown(_gt_param, stream);
      return cudaSuccess;
    }
  }
};

template <typename DTypeIn, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask,
          class ThreadShape = Shape<_2, _1, _1>,
          bool IsSplitKV = false,
          bool SingleSoftmaxWarpGroup = false,
          int KVPageSize = -1,
          cutlass::fmha::collective::SparseAttnMode kSparseAttnMode =
              cutlass::fmha::collective::SparseAttnMode::Off,
          bool UnitSelectorColumn = false>
cudaError_t run_fmha_fwd(void* workspace_buffer, DTypeIn* q, DTypeIn* k, DTypeIn* v,
                         IdType* qo_segment_lens, IdType* kv_segment_lens,
                         IdType* qo_segment_offsets, IdType* kv_segment_offsets,
                         uint64_t* packed_work_range, uint64_t* packed_work_info,
                         DTypeOut* o, int mask_mode_code,
                         double sm_scale, double q_scale, double k_scale, double v_scale,
                         double o_scale, int num_qo_heads, int num_kv_heads, int head_dim_qk,
                         int head_dim_vo, int q_stride_n, int q_stride_h, int k_stride_n,
                         int k_stride_h, int v_stride_n, int v_stride_h, int batch_size,
                         int total_qo_len, int total_kv_len, int max_qo_len, int* qo_offsets,
                         cudaStream_t stream,
                         int num_kv_splits = 1,
                         int* kv_tile_begin_indices = nullptr,
                         int* kv_tile_end_indices = nullptr,
                         int* kv_split_indices = nullptr,
                         float* ptr_lse_accum = nullptr,
                         int* kv_indices = nullptr,
                         int* kv_page_indptr = nullptr,
                         int total_page_num = 0,
                         float* maybe_max_score = nullptr,
                         int max_k_tiles = 0,
                         int* kv_block_indexes = nullptr,
                         int kv_block_num = 0,
                         int64_t kv_block_stride_n = 0,
                         int64_t kv_block_stride_h = 0,
                         int64_t kv_block_stride_b = 1,
                         int pack_factor = 1,
                         int q_stride_n_original = 0,
                         int q_stride_h_original = 0,
                         int h_r_original = 0,
                         PackGQAUnpackParams pack_gqa = {},
                         int num_ctas = 0) {
  return FwdRunner<DTypeIn, DTypeOut, IdType, TileShapeQK, TileShapePV, ActiveMask,
                   ThreadShape, IsSplitKV, SingleSoftmaxWarpGroup, KVPageSize, kSparseAttnMode,
                   UnitSelectorColumn>::run(
      workspace_buffer, q, k, v, qo_segment_lens, kv_segment_lens,
      qo_segment_offsets, kv_segment_offsets, packed_work_range,
      packed_work_info, o, mask_mode_code, sm_scale,
      q_scale, k_scale, v_scale, o_scale, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
      q_stride_n, q_stride_h, k_stride_n, k_stride_h, v_stride_n, v_stride_h, batch_size,
      total_qo_len, total_kv_len, max_qo_len, qo_offsets, stream,
      num_kv_splits, kv_tile_begin_indices, kv_tile_end_indices, kv_split_indices,
      ptr_lse_accum,
      kv_indices, kv_page_indptr, total_page_num,
      maybe_max_score, max_k_tiles,
      kv_block_indexes, kv_block_num,
      kv_block_stride_n, kv_block_stride_h, kv_block_stride_b,
      pack_factor, q_stride_n_original, q_stride_h_original, h_r_original,
      pack_gqa, num_ctas);
}

};  // namespace flashinfer

namespace flashinfer {

// Launch the split-KV reduction kernel
template <typename ElementPartial, typename ElementOut>
__global__ void fmha_reduction_kernel(
    typename Sm100FmhaReductionKernel<ElementPartial, ElementOut>::Params params) {
  Sm100FmhaReductionKernel<ElementPartial, ElementOut> kernel;
  kernel(params, nullptr);
}

template <typename ElementPartial, typename ElementOut>
cudaError_t launch_fmha_reduction(
    const ElementPartial* ptr_O_partial, ElementOut* ptr_O,
    const float* ptr_lse,
    const int* num_kv_splits_per_row,
    float scale_softmax_log2, float inv_scale_o,
    int num_kv_splits, int total_qo_len, int num_qo_heads, int head_dim_vo,
    int stride_o_n, int stride_o_h, int stride_partial_n, int stride_partial_h,
    ElementOut* ptr_O_direct, int num_qo_heads_orig,
    int num_kv_heads, int pack_factor,
    cudaStream_t stream) {
  if (total_qo_len <= 0) return cudaSuccess;
  using ReductionKernel = Sm100FmhaReductionKernel<ElementPartial, ElementOut>;
  typename ReductionKernel::Params params{
      ptr_O_partial, ptr_O, ptr_lse,
      num_kv_splits_per_row,
      scale_softmax_log2, inv_scale_o,
      num_kv_splits, total_qo_len, num_qo_heads, head_dim_vo,
      stride_o_n, stride_o_h, stride_partial_n, stride_partial_h,
      ptr_O_direct, num_qo_heads_orig, num_kv_heads, pack_factor};
  dim3 grid = ReductionKernel::get_grid_shape(params);
  dim3 block = ReductionKernel::get_block_shape();
  fmha_reduction_kernel<ElementPartial, ElementOut><<<grid, block, 0, stream>>>(params);
  return cudaGetLastError();
}

}  // namespace flashinfer
