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
#pragma once

#include <cuda.h>  // CUtensorMap, cuTensorMapEncodeTiled
#include "cute/layout.hpp"
#include "cute/swizzle.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "fmha_common.hpp"
#include "gpu_trace.h"

GPU_TRACE_SCOPE_DEC(EPILOGUE);

namespace cutlass::fmha::collective {

namespace detail {

// constexpr cute::Swizzle<B,M,S> → CUtensorMapSwizzle
// Mirrors cute::TMA::to_CUtensorMapSwizzle but is usable in compile-time contexts.
// Only the swizzle patterns produced by sm100_smem_selector for K-major atoms.
template <int B, int M, int S>
constexpr CUtensorMapSwizzle to_cu_tensor_map_swizzle(cute::Swizzle<B, M, S>) {
  if constexpr (B == 0)                          return CU_TENSOR_MAP_SWIZZLE_NONE;
  else if constexpr (B == 1 && M == 4 && S == 3) return CU_TENSOR_MAP_SWIZZLE_32B;
  else if constexpr (B == 2 && M == 4 && S == 3) return CU_TENSOR_MAP_SWIZZLE_64B;
  else if constexpr (B == 3 && M == 4 && S == 3) return CU_TENSOR_MAP_SWIZZLE_128B;
  else {
    // Dependent-name static_assert so it only fires on bad instantiations.
    static_assert(B + M + S < 0,
                  "Unsupported cute::Swizzle pattern for TMA descriptor "
                  "(only SW128/SW64/SW32/NONE produced by sm100_smem_selector "
                  "are mapped).");
    return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

// constexpr dtype → CUtensorMapDataType for our supported epilogue outputs.
// Production paths only ever instantiate Element = bf16, but keep bf16/fp16/fp8
// branches so fallback / future use stay sound.
template <class T>
constexpr CUtensorMapDataType to_cu_tensor_map_dtype() {
  if constexpr (cute::is_same_v<T, cutlass::bfloat16_t>) return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  else if constexpr (cute::is_same_v<T, cutlass::half_t>) return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  else if constexpr (cute::is_same_v<T, cutlass::float_e4m3_t>) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  else if constexpr (cute::is_same_v<T, cutlass::float_e5m2_t>) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  else if constexpr (sizeof(T) == 1) return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  else if constexpr (sizeof(T) == 2) return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  else {
    static_assert(sizeof(T) < 0, "Unsupported epilogue Element type for TMA fused unpack");
    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
  }
}

}  // namespace detail

template <class Element, class ElementAcc, class TileShape, int NumQStages_ = 2, bool IsSplitKV_ = false, bool NeedOutput_ = true, int kPackFactor_ = 1>
struct Sm100FmhaFwdEpilogueTmaWarpspecialized {
  static constexpr bool IsSplitKV = IsSplitKV_;
  static constexpr int  kPackFactor = kPackFactor_;
  // Direct-O unpack lives in this epilogue ONLY for non-split path. Split-KV
  // routes unpack into the reduction kernel, so split variants must elide this
  // entire codepath at compile time.
  static constexpr bool kEnableDirectO = !IsSplitKV_ && (kPackFactor_ > 1);
  using ElementOut = Element;
  using Pipeline = cutlass::PipelineAsync<NumQStages_>;
  // using ShapeT = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  // using StrideO = cute::Shape<int32_t, _1, cute::Shape<int32_t, int32_t>>;
  // using LayoutO = cute::Layout<ShapeT, StrideO>;
  using ShapeT = cute::Shape<int32_t, int32_t, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t>>;
  using StrideO = cute::Shape<int32_t, _1, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t>>;
  using LayoutO = cute::Layout<ShapeT, StrideO>;

  using ShapeLSE = cute::Shape<int32_t, cute::Shape<int32_t, int32_t>>;
  using StrideLSE = cute::Shape<int32_t, cute::Shape<_1, int32_t>>;
  using LayoutLSE = cute::Layout<ShapeLSE, StrideLSE>;

  using ShapeMaxScore = cute::Shape<int32_t, cute::Shape<int32_t, int32_t>, int32_t>;
  using StrideMaxScore = cute::Shape<_1, cute::Shape<int32_t, int32_t>, int32_t>;
  using LayoutMaxScore = cute::Layout<ShapeMaxScore, StrideMaxScore>;

  //  using SmemLayoutO = decltypa(make_layout(append<3>(select<0,1>(TileShape_WG{}), _2{})));
  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
                                   cute::UMMA::Major::K, Element, tuple_element_t<0, TileShape>,
                                   tuple_element_t<1, TileShape>>());
  //  using SmemLayoutAtomO = decltype(make_ordered_layout(select<0,1>(TileShape{}), Step<_1,
  //  _0>{}));
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtomO{}, replace<2>(TileShape{}, cute::Int<NumQStages_>{}), Step<_2, _1, _3>{}));
  using SmemLayoutO_ = SmemLayoutO;
  static constexpr size_t kSmemOCosize = cute::cosize_v<SmemLayoutO>;

  static constexpr int kQRows = cute::size<0>(TileShape{});

  // ---- Compile-time atom + swizzle extraction (TMA fused unpack path) ----
  // SmemLayoutAtomO is cute::ComposedLayout<Swizzle, Offset, InnerLayout>.
  // InnerLayout shape is (atom_rows, atom_cols); Swizzle is e.g. Swizzle<3,4,3> (SW128).
  using AtomInnerLayout = decltype(cute::get_nonswizzle_portion(SmemLayoutAtomO{}));
  using AtomSwizzle     = decltype(cute::get_swizzle_portion(SmemLayoutAtomO{}));
  static constexpr int kAtomRows = cute::size<0>(AtomInnerLayout{});
  static constexpr int kAtomCols = cute::size<1>(AtomInnerLayout{});

  struct TensorStorageBase {
    using SmemLayoutO = SmemLayoutO_;
    // NeedOutput_=false (OnlyScore mode): all real smem_o accesses (TMA store
    // at 549/561, direct-O PTX at 395/417) are gated by `if constexpr (NeedOutput_)`
    // or `if constexpr (kEnableDirectO && NeedOutput_)`. Line 520 `make_tensor`
    // and partition_S/D only need a valid pointer + shape, no byte access, so
    // a 1-byte placeholder is safe and frees ~32-64KB SMEM for K-pipeline.
    cute::array_aligned<Element, NeedOutput_ ? cute::cosize_v<SmemLayoutO> : 1> smem_o;
  };

  struct TensorStorageSplitKV : TensorStorageBase {
    float smem_lse[kQRows];
  };

  using TensorStorage = std::conditional_t<IsSplitKV, TensorStorageSplitKV, TensorStorageBase>;
  struct Arguments {
    Element* ptr_O;
    LayoutO layout_O;

    int max_qo_len;
    ElementAcc* ptr_MaxScore = nullptr;
    LayoutMaxScore layout_MaxScore;

    ElementAcc* ptr_MaxScore_direct = nullptr;
    int total_qo_len_orig = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int max_score_numel = 0;
#endif

    Element* ptr_O_direct = nullptr;
    int num_qo_heads_orig = 0;
    int head_dim_vo = 0;
    int remaining_h_r = 0;

    // TMA fused direct-O unpack: host eligibility flag (qo_len equal across
    // batches + env var ok). Per-batch unpacked qo_len is derived as
    // max_qo_len / kPackFactor_ at descriptor build time (kPackFactor_ is a
    // compile-time template constant, max_qo_len is the packed length).
    bool qo_len_uniform = false;
  };

  using TMA_O = decltype(make_tma_copy(
      SM90_TMA_STORE{}, make_tensor((Element*)nullptr, repeat_like(StrideO{}, 0), StrideO{}),
      SmemLayoutO{}(_, _, _0{})));

  // ---- TMA fused direct-O unpack (Phase 2: hand-rolled CUtensorMap + PTX) ----
  // Replaces cuTe TMA descriptor (which had unsolvable composition + partition
  // contract issues) with raw NVIDIA driver API + PTX. The hardware path is
  // identical; we just skip the cuTe layout abstraction layer.
  //
  // Descriptor encoded host-side via cuTensorMapEncodeTiled with a 5D box that
  // mirrors SMEM atom tiling exactly:
  //   GMEM rank 5 = (atom_cols, num_heads_orig, num_horiz_atoms,
  //                  num_vert_atoms_per_tok, total_qo_len_orig)
  //   Box        = (atom_cols, atom_rows, num_horiz_atoms,
  //                 num_vert_atoms_per_tok, max_qo_len/kPackFactor)
  //   Swizzle    = matches SmemLayoutAtomO (e.g. K-major SW128 for bf16 hd=128)
  //
  // Device-side: single PTX cp.async.bulk.tensor.5d.
  static constexpr int kNumActualTokMax = (kPackFactor_ > 0) ? (kQRows / kPackFactor_) : kQRows;
  static constexpr int kHeadDimCT = cute::size<1>(TileShape{});

  // 5D split parameters derived purely from atom shape + (pf, head_dim, dtype).
  // - kNumHorizAtoms: head_dim spans this many atom-cols-wide tiles
  // - kNumVertAtomsPerTok: one actual_tok covers pf packed_rows = this many atom-rows tiles
  static constexpr int kNumHorizAtoms       = (kAtomCols > 0) ? (kHeadDimCT / kAtomCols) : 0;
  static constexpr int kNumVertAtomsPerTok  = (kAtomRows > 0) ? (kPackFactor_ / kAtomRows) : 0;

  // Compile-time TMA path eligibility. If any condition fails the entire host
  // descriptor build and device PTX path are pruned and we silently fall back
  // to the vec16 software scatter loop. Conditions:
  //   1) pf > 1                              -- pf=1 has no pack to unpack
  //   2) IsSplitKV_ = false                  -- split-KV routes unpack elsewhere
  //   3) pf % atom_rows == 0                 -- avoids fractional vert atoms
  //   4) head_dim % atom_cols == 0           -- avoids fractional horiz atoms
  //   5) pf >= atom_rows                     -- ensures >=1 vert atom per actual_tok
  //   6) atom_cols * sizeof(Element) == 128  -- SW128 inner == 128 bytes
  static constexpr bool kTmaPathCompileEligible =
      kEnableDirectO &&
      (kAtomRows > 0) && (kAtomCols > 0) &&
      (kPackFactor_ % kAtomRows == 0) &&
      (kHeadDimCT  % kAtomCols == 0) &&
      (kPackFactor_ >= kAtomRows) &&
      (kAtomCols * (int)sizeof(Element) == 128);

  struct Params {
    TMA_O tma_store_o;
    LayoutO layout_O;
    ElementAcc* ptr_LSE = nullptr;
    LayoutLSE layout_LSE;
    int max_qo_len;
    ElementAcc* ptr_MaxScore = nullptr;
    LayoutMaxScore layout_MaxScore;

    ElementAcc* ptr_MaxScore_direct = nullptr;
    int total_qo_len_orig = 0;
#ifdef FMHA_GMEM_BOUNDS_CHECK
    int max_score_numel = 0;
#endif

    Element* ptr_O_direct = nullptr;
    int num_qo_heads_orig = 0;
    int head_dim_vo = 0;
    int remaining_h_r = 0;

    // Hand-rolled TMA descriptor for unpacked O write. Always 128 bytes / 16 QWORDs.
    // valid=true means descriptor was successfully built host-side.
    CUtensorMap tma_desc_o_direct;
    bool tma_store_o_direct_valid = false;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
                                        void* workspace = nullptr) {
    static_assert(is_variable_length_v<tuple_element_t<0, ProblemShape>>);
    auto ptr_O = args.ptr_O;
    LayoutO layout_O = args.layout_O;

    TMA_O tma_store_o;
    if (ptr_O != nullptr) {
      tma_store_o =
          make_tma_copy(SM90_TMA_STORE{}, make_tensor(ptr_O, layout_O), SmemLayoutO{}(_, _, _0{}));
    }

    // Phase 2: construct TMA descriptor via NVIDIA driver API cuTensorMapEncodeTiled.
    //
    // CRITICAL: SmemLayoutO uses tile_to_shape with atom (kAtomRows × kAtomCols).
    // SMEM bytes are NOT laid out as (packed_row, head_dim) row-major. Atom tiling
    // order (with Step<_2,_1,_3> meaning head_dim innermost, kQRows next):
    //   atom 0: rows 0..kAtomRows-1, cols 0..kAtomCols-1
    //   atom 1: rows 0..kAtomRows-1, cols kAtomCols..2*kAtomCols-1
    //   atom 2: rows kAtomRows..2*kAtomRows-1, cols 0..kAtomCols-1
    //   ... so on
    //
    // To make TMA descriptor match this SMEM byte order, the box must be 5D:
    //   dim 0 (innermost): kAtomCols head_dim cols within atom row    → GMEM stride 1 elem
    //   dim 1:             kAtomRows atom rows                         → GMEM stride head_dim (1 head)
    //   dim 2:             kNumHorizAtoms horizontal atom halves       → GMEM stride kAtomCols elem
    //   dim 3:             kNumVertAtomsPerTok vertical atom halves    → GMEM stride kAtomRows * head_dim (kAtomRows heads)
    //   dim 4:             max_qo_len/kPackFactor (num actual_tok)      → GMEM stride num_heads_orig * head_dim (1 token)
    //
    // 5D total elements = kAtomCols * kAtomRows * kNumHorizAtoms * kNumVertAtomsPerTok * kNumActualTokMax
    //                   = kQRows * kHeadDimCT = one CTA's SMEM smem_o size.
    CUtensorMap tma_desc_o_direct{};
    bool tma_store_o_direct_valid = false;
    if constexpr (kTmaPathCompileEligible) {
      // max_qo_len is the packed length; unpacked per-batch length is
      // max_qo_len / kPackFactor_. Require >= kPackFactor_ so the unpacked
      // length is at least 1.
      if (args.ptr_O_direct != nullptr
          && args.qo_len_uniform
          && args.max_qo_len >= kPackFactor_
          && args.num_qo_heads_orig > 0
          && args.head_dim_vo == kHeadDimCT
          && args.total_qo_len_orig > 0) {
        constexpr CUtensorMapDataType tma_dtype = detail::to_cu_tensor_map_dtype<Element>();
        constexpr CUtensorMapSwizzle  tma_swizzle = detail::to_cu_tensor_map_swizzle(AtomSwizzle{});

        cuuint64_t global_dim[5] = {
            (cuuint64_t)kAtomCols,                                // dim 0 = atom inner head_dim cols
            (cuuint64_t)args.num_qo_heads_orig,                   // dim 1 = full num_heads
            (cuuint64_t)kNumHorizAtoms,                           // dim 2 = head_dim outer atoms
            (cuuint64_t)kNumVertAtomsPerTok,                      // dim 3 = vertical atom halves per actual_tok
            (cuuint64_t)args.total_qo_len_orig};                  // dim 4 = full token count
        cuuint64_t global_stride[4] = {
            (cuuint64_t)args.head_dim_vo * sizeof(Element),                                  // dim 0 → dim 1 : 1 head row
            (cuuint64_t)kAtomCols * sizeof(Element),                                         // dim 1 → dim 2 : kAtomCols elems
            (cuuint64_t)kAtomRows * args.head_dim_vo * sizeof(Element),                      // dim 2 → dim 3 : kAtomRows heads
            (cuuint64_t)args.num_qo_heads_orig * args.head_dim_vo * sizeof(Element)};        // dim 3 → dim 4 : 1 token
        cuuint32_t box_dim[5] = {
            (cuuint32_t)kAtomCols,
            (cuuint32_t)kAtomRows,
            (cuuint32_t)kNumHorizAtoms,
            (cuuint32_t)kNumVertAtomsPerTok,
            (cuuint32_t)(args.max_qo_len / kPackFactor_)};
        cuuint32_t element_stride[5] = {1, 1, 1, 1, 1};
        CUresult result = cuTensorMapEncodeTiled(
            &tma_desc_o_direct,
            tma_dtype,
            /*tensorRank=*/5,
            args.ptr_O_direct,
            global_dim, global_stride,
            box_dim, element_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            tma_swizzle,
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        tma_store_o_direct_valid = (result == CUDA_SUCCESS);
        if (!tma_store_o_direct_valid) {
          std::cerr << "[fmha_sm100] cuTensorMapEncodeTiled failed (code=" << result
                    << "), falling back to vec16 scatter." << std::endl;
        }
      }
    }

    Params p = {tma_store_o, layout_O, nullptr, {}, args.max_qo_len,
            args.ptr_MaxScore, args.layout_MaxScore,
            args.ptr_MaxScore_direct, args.total_qo_len_orig,
#ifdef FMHA_GMEM_BOUNDS_CHECK
            args.max_score_numel,
#endif
            args.ptr_O_direct, args.num_qo_heads_orig, args.head_dim_vo,
            args.remaining_h_r,
            tma_desc_o_direct, tma_store_o_direct_valid};
    return p;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    // P4 (v3): OnlyScore mode (NeedOutput_=false) never uses tma_store_o
    // (all consumers are gated under NeedOutput_), so the descriptor prefetch
    // is dead — skip it. Saves one prefetch_tma_descriptor PTX hint per CTA.
    if constexpr (!NeedOutput_) return;
    bool prefetch = true;
    if constexpr (kEnableDirectO) {
      if (params.ptr_O_direct != nullptr) prefetch = false;
    }
    if (prefetch) {
      cute::prefetch_tma_descriptor(params.tma_store_o.get_tma_descriptor());
    }
  }

  const Params& params;

  CUTLASS_DEVICE Sm100FmhaFwdEpilogueTmaWarpspecialized(const Params& params) : params(params) {}

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE auto store(BlkCoord const& blk_coord, ProblemShape const& problem_shape,
                            Params const& params, ParamsProblemShape const& params_problem_shape,
                            TensorStorage& shared_storage, Pipeline& pipeline,
                            typename Pipeline::PipelineState& pipeline_consumer_state,
                            int kv_split_idx = 0, int total_qo_len_splitkv = 0,
                            float* ptr_lse_accum = nullptr,
                            int split_kv_num_qo_heads = 0) {
    GET_GPU_TRACE(true);
    int qo_tile_idx = get<0>(blk_coord);
    int qo_head_idx = get<2, 0>(blk_coord);
    int batch_idx = get<2, 1>(blk_coord);
    int qo_len = get<0>(problem_shape);
    int qo_segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];
    uint32_t lane_predicate = cute::elect_one_sync();

    // Direct O write: read from smem (written by correction warpgroup),
    // write to unpacked global O layout. Runs in epilogue warpgroup,
    // overlapping with next tile's mainloop. Decode-only path: pack_factor>1
    // implies max_qo_len<=32, which forces tile_q=128 / NumQStages_=1.
    if constexpr (kEnableDirectO && NeedOutput_ && !IsSplitKV) {
    static_assert(NumQStages_ == 1,
                  "Direct-O unpack only supports NumQStages_=1 (decode path with tile_q=128)");
    // Hot path: pack>1 + non-split + production env always has ptr_O_direct.
    // The nullptr fallback exists only for FMHA_DISABLE_DIRECT_O_UNPACK=1.
    if (__builtin_expect(params.ptr_O_direct != nullptr, 1)) {
      static constexpr int pf = kPackFactor;

      int rem_hr = params.remaining_h_r;
      int kv_head = qo_head_idx / rem_hr;
      int rhr_idx = qo_head_idx % rem_hr;
      int h_r_orig = rem_hr * pf;
      int seg_off_orig = qo_segment_offset / pf;
      int head_dim = params.head_dim_vo;
      int num_heads_orig = params.num_qo_heads_orig;
      int base_head = kv_head * h_r_orig + rhr_idx * pf;

      // ---- Phase 2 TMA fused direct-O fast path (5D hand-rolled PTX) ----
      // 5D tensor layout matches SMEM atom tiling exactly:
      //   dim 0: kAtomCols head_dim cols within atom (innermost)
      //   dim 1: kAtomRows atom rows (packed_rows)
      //   dim 2: kNumHorizAtoms horizontal atoms (head_dim halves)
      //   dim 3: kNumVertAtomsPerTok vertical atoms per actual_tok (pf_idx halves)
      //   dim 4: actual_tok (token index)
      //
      // CTA writes box at coord (0, base_head, 0, 0, seg_off_orig).
      // base_head maps onto dim 1 (stride = 1 head); dim 3 then advances by
      // kAtomRows heads, covering pf = kAtomRows * kNumVertAtomsPerTok heads
      // for the current CTA.
      if constexpr (kTmaPathCompileEligible) {
        if (params.tma_store_o_direct_valid) {
          pipeline.consumer_wait(pipeline_consumer_state);
          {
            GPU_TRACE_SCOPE(EPILOGUE);
            if (cute::elect_one_sync()) {
              uint64_t desc_addr = reinterpret_cast<uint64_t>(&params.tma_desc_o_direct);
              uint32_t smem_int = cute::cast_smem_ptr_to_uint(shared_storage.smem_o.data());
              int32_t crd0 = 0;
              int32_t crd1 = base_head;
              int32_t crd2 = 0;
              int32_t crd3 = 0;
              int32_t crd4 = seg_off_orig;
              asm volatile(
                  "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%2, %3, %4, %5, %6}], [%1];\n"
                  "cp.async.bulk.commit_group;\n"
                  "cp.async.bulk.wait_group 0;\n"
                  :: "l"(desc_addr), "r"(smem_int),
                     "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
                  : "memory");
            }
          }
          pipeline.consumer_release(pipeline_consumer_state);
          ++pipeline_consumer_state;
          RELEASE_GPU_TRACE;
          return;
        }
      }

      Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
      int tid = threadIdx.x % 32;

      pipeline.consumer_wait(pipeline_consumer_state);
      {
        GPU_TRACE_SCOPE(EPILOGUE);
        if constexpr (kQRows % pf == 0) {
        // Vec16 fast path: kQRows splits cleanly into pf-row segments.
        // For each actual_tok the pf rows map to a contiguous (pf * head_dim)
        // byte segment in unpacked GMEM. K-major SMEM atom guarantees
        // head_dim direction is byte-contiguous within a row, so each
        // thread does one uint4 load (within a row) + one uint4 store.
        static constexpr int num_segs = kQRows / pf;
        int base_head = kv_head * h_r_orig + rhr_idx * pf;
        using VecT = uint4;
        static constexpr int VEC = sizeof(VecT);  // 16 bytes
        // sizeof(Element) is 1 for fp8 / 2 for bf16; operate on bytes.
        int vec_per_row = (head_dim * (int)sizeof(Element)) / VEC;
        int vecs_per_seg = pf * vec_per_row;

        int tile_base = qo_tile_idx * kQRows;
        int tile_base_tok = tile_base / pf;

        CUTLASS_PRAGMA_NO_UNROLL
        for (int seg = 0; seg < num_segs; seg++) {
          int seg_first_packed_row = tile_base + seg * pf;
          if (seg_first_packed_row >= qo_len) break;

          int actual_tok = tile_base_tok + seg;
          int64_t base_offset =
              (int64_t)(seg_off_orig + actual_tok) * num_heads_orig * head_dim
              + (int64_t)base_head * head_dim;
          VecT* dst_seg = reinterpret_cast<VecT*>(params.ptr_O_direct + base_offset);

          for (int v = tid; v < vecs_per_seg; v += 32) {
            int r_local = v / vec_per_row;
            int d_chunk = v % vec_per_row;
            int r_global = seg * pf + r_local;
            if (seg_first_packed_row + r_local >= qo_len) continue;
            VecT data = *reinterpret_cast<const VecT*>(
                &sO(r_global, d_chunk * (VEC / (int)sizeof(Element)), _0{}));
            dst_seg[v] = data;
          }
        }
      } else {
        // Per-row fallback: kQRows not divisible by pf (e.g. pack_factor=6
        // with tile_q=128 → kQRows=128, 128 % 6 != 0). Each thread handles
        // its own packed row, computes unpacked head with pf_idx, and
        // scalar-stores head_dim elements.
        int tile_base = qo_tile_idx * kQRows;
        for (int r = tid; r < kQRows; r += 32) {
          int packed_row = tile_base + r;
          if (packed_row >= qo_len) break;
          int pf_idx = packed_row % pf;
          int actual_tok = packed_row / pf;
          int unpacked_head = kv_head * h_r_orig + rhr_idx * pf + pf_idx;
          Element* dst = params.ptr_O_direct
                       + (int64_t)(seg_off_orig + actual_tok) * num_heads_orig * head_dim
                       + (int64_t)unpacked_head * head_dim;
          for (int d = 0; d < head_dim; d++) {
            dst[d] = sO(r, d, _0{});
          }
        }
        pipeline.consumer_release(pipeline_consumer_state);
        ++pipeline_consumer_state;
        RELEASE_GPU_TRACE;
        return;
      }
      }  // close GPU_TRACE_SCOPE block
      pipeline.consumer_release(pipeline_consumer_state);
      ++pipeline_consumer_state;
      RELEASE_GPU_TRACE;
      return;
    }
    }  // if constexpr (kEnableDirectO)

    // Fallthrough = packed TMA store path. Reached when:
    //   - kEnableDirectO=false (pack=1 or split-KV), OR
    //   - kEnableDirectO=true but ptr_O_direct==nullptr (disable env).
    // In all cases, host-side (fmha_cutlass_sm100.cuh:182,197-199) guarantees
    // params.tma_store_o is constructed from a valid ptr_O whenever
    // NeedOutput_ is true. Contract: ptr_O nulled iff ptr_O_direct provided.

    using X = Underscore;

    int o0_index = NumQStages_ * get<0>(blk_coord);
    int o1_index = NumQStages_ * get<0>(blk_coord) + 1;

    int offs_0 = params.max_qo_len - qo_len;
    int offs_2_1 = qo_segment_offset + qo_len;
    if (total_qo_len_splitkv > 0 && kv_split_idx > 0) {
      offs_2_1 += kv_split_idx * total_qo_len_splitkv;
    }
    BlkCoord blk_coord_updated = blk_coord;
    get<2, 1>(blk_coord_updated) = 0;

    Tensor mO = params.tma_store_o.get_tma_tensor(params.layout_O.shape());

    Tensor mO_qdl = domain_offset(make_coord(offs_0, _0{}, make_coord(_0{}, offs_2_1)), mO);

    Tensor gO_qdl = local_tile(mO_qdl, TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{});
    Tensor gO = gO_qdl(_, _, _, _0{}, get<2>(blk_coord_updated));

    Tensor sO = make_tensor(make_smem_ptr(shared_storage.smem_o.data()), SmemLayoutO{});
    auto block_tma = params.tma_store_o.get_slice(0);
    Tensor tOsO = block_tma.partition_S(sO);
    Tensor tOgO = block_tma.partition_D(gO);

    auto pipeline_release_state = pipeline_consumer_state;

    // O1 O2
    // one pipeline: O
    // wait from corr, issue tma store on smem
    pipeline.consumer_wait(pipeline_consumer_state);
    ++pipeline_consumer_state;

    // Write split-KV stats from smem to global memory
    if constexpr (IsSplitKV) {
      int tile_row_base = qo_tile_idx * kQRows;
      int lane = threadIdx.x % 32;
      for (int r = lane; r < kQRows; r += 32) {
        int row_idx = tile_row_base + r;
        if (row_idx < qo_len) {
          int abs_row = qo_segment_offset + row_idx;
          int actual_split = kv_split_idx < 0 ? 0 : kv_split_idx;
          int stats_offset = actual_split * total_qo_len_splitkv * split_kv_num_qo_heads
                           + abs_row * split_kv_num_qo_heads + qo_head_idx;
          ptr_lse_accum[stats_offset] = shared_storage.smem_lse[r];
        }
      }
    }

    if constexpr (NeedOutput_) {
      if (lane_predicate) {
        GPU_TRACE_SCOPE(EPILOGUE);
        copy(params.tma_store_o, tOsO(_, _, _, _0{}), tOgO(_, _, _, o0_index));
      }
      tma_store_arrive();
    }

    if constexpr (NumQStages_ > 1) {
      pipeline.consumer_wait(pipeline_consumer_state);
      ++pipeline_consumer_state;

      if constexpr (NeedOutput_) {
        if (lane_predicate) {
          GPU_TRACE_SCOPE(EPILOGUE);
          copy(params.tma_store_o, tOsO(_, _, _, _1{}), tOgO(_, _, _, o1_index));
        }
        tma_store_arrive();

        tma_store_wait<1>();
      }

      pipeline.consumer_release(pipeline_release_state);
      ++pipeline_release_state;

      if constexpr (NeedOutput_) {
        tma_store_wait<0>();
      }
    } else {
      if constexpr (NeedOutput_) {
        tma_store_wait<0>();
      }

      pipeline.consumer_release(pipeline_release_state);
      ++pipeline_release_state;
    }

    if constexpr (NumQStages_ > 1) {
      pipeline.consumer_release(pipeline_release_state);
      ++pipeline_release_state;
    }
    RELEASE_GPU_TRACE;
  }
};

}  // namespace cutlass::fmha::collective
