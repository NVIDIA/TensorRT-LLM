// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

// Auto-generated FMHA variant: 1_0_0_2_1_0_1
// DTypeIn=__nv_fp8_e4m3, Tile=_128x_256, Sparse=OnlyScore, Page=128, SplitKV=false
#include "fmha_cutlass_sm100.cuh"
#include "mask.cuh"
#include "cutlass_utils.cuh"
#include "fmha_sm100_params.h"

using namespace flashinfer;
using namespace cute;

cudaError_t fmha_sm100_1_0_0_2_1_0_1(const FMHACutlassSM100Params& p) {
  using cutlass_type_in = cutlass_dtype_t<__nv_fp8_e4m3>;
  using cutlass_type_out = cutlass::bfloat16_t;
  using TileShapeQK = Shape<_128, _256, cute::Int<128>>;
  using TileShapePV = Shape<_128, cute::Int<128>, _256>;
  using FmhaThreadShape = Shape<_1, _2, _1>;


  auto* out_ptr = static_cast<cutlass_type_out*>(p.o_ptr);
  int effective_max_qo_len = p.max_qo_len;


  using ActiveMaskType = PackedCausalMask<2>;

  auto run = [&](auto unit_selector_column) {
    return run_fmha_fwd<cutlass_type_in, cutlass_type_out, int32_t,
                      TileShapeQK, TileShapePV, ActiveMaskType, FmhaThreadShape,
                      /*IsSplitKV=*/false,
                      /*SingleSoftmaxWarpGroup=*/true,
                      128,
                      cutlass::fmha::collective::SparseAttnMode::OnlyScore,
                      decltype(unit_selector_column)::value>(
      p.workspace_buffer_ptr,
      static_cast<cutlass_type_in*>(p.q_ptr),
      static_cast<cutlass_type_in*>(p.k_ptr),
      static_cast<cutlass_type_in*>(p.v_ptr),
      p.qo_segment_lens_ptr, p.kv_segment_lens_ptr,
      p.qo_segment_offsets_ptr, p.kv_segment_offsets_ptr,
      p.packed_work_range_ptr, p.packed_work_info_ptr,
      out_ptr,
      p.mask_mode_code,
      p.sm_scale, p.scale_q, p.scale_k, p.scale_v, p.o_scale,
      p.num_qo_heads, p.num_kv_heads,
      p.head_dim_qk, p.head_dim_vo,
      p.q_stride_n, p.q_stride_h,
      p.k_stride_n, p.k_stride_h,
      p.v_stride_n, p.v_stride_h,
      p.batch_size, p.total_qo_len, p.total_kv_len,
      effective_max_qo_len,
      p.qo_offsets_ptr, p.stream,
      p.num_kv_splits,
      p.kv_tile_begin_ptr, p.kv_tile_end_ptr, p.kv_split_ptr,
      p.workspace_lse_ptr,
      p.kv_indices_ptr, p.kv_page_indptr_ptr, p.total_page_num,
      p.max_score_ptr, p.max_k_tiles,
      p.kv_block_indexes_ptr, p.kv_block_num,
      p.kv_block_stride_n, p.kv_block_stride_h, p.kv_block_stride_b,
      p.pack_factor, p.q_stride_n_original, p.q_stride_h_original, p.h_r_original,
      flashinfer::PackGQAUnpackParams{p.max_score_direct_ptr, p.total_qo_len_orig,
                                      p.o_direct_ptr, p.num_qo_heads_orig,
                                      p.tma_direct_o_enabled
#ifdef FMHA_GMEM_BOUNDS_CHECK
                                      , p.gmem_bounds
#endif
                                      },
      p.num_ctas);
  };

  return run(cute::false_type{});
}