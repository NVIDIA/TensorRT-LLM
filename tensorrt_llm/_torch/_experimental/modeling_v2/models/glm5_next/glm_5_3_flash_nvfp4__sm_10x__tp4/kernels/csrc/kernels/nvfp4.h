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
// NVFP4 (ModelOpt W4A4): e2m1 values with one e4m3 scale per 16 elements and an fp32 global scale
// per tensor. Activation quantization, the MoE permutation for the grouped GEMM, and the
// CUTLASS SM100 block-scaled GEMMs (dense and grouped), plus the BF16 grouped GEMM used by the
// experts a checkpoint leaves unquantized (the MTP layer of the NVFP4 checkpoint).
#pragma once

#include <memory>

#include "../common.h"
#include "moe.h"

namespace glm {
namespace k {

// Byte offset of the scale of (row m, 16-element block kb) of a [rows, K] operand in the CUTLASS
// SM100 K-major scale layout: 512-byte atoms of 128 rows x 4 blocks (row m of an atom at
// (m % 32) * 16 + (m / 32 % 4) * 4, block kb at + kb % 4), atoms along K first, then along rows.
__host__ __device__ __forceinline__ size_t nvfp4_sf_off(int m, int kb, int K) {
  const int katoms = (K / 16 + 3) / 4;
  return ((size_t)(m >> 7) * katoms + (kb >> 2)) * 512 + (m & 31) * 16 + ((m >> 5) & 3) * 4 +
         (kb & 3);
}
// bytes of the scales of a [rows, K] operand in that layout (rows padded to 128, blocks to 4)
__host__ __device__ __forceinline__ size_t nvfp4_sf_bytes(int rows, int K) {
  return (size_t)round_up(rows, 128) * round_up(K / 16, 4);
}

// x [M, K] bf16 (row stride ldx) -> NVFP4 with global scale gs (the checkpoint's input_scale):
// q [M, K/2] packed e2m1 (element 2i in the low nibble), scales e4m3 into sf: row-major
// [M, K/16] (swz = false) or the CUTLASS layout of [M, K] (swz = true).
// TRT-LLM's recipe (cvt_warp_fp16_to_fp4): s = e4m3(amax16 / 6 / gs), q = e2m1(x / (s * gs)).
void nvfp4_quant(uint8_t* q, uint8_t* sf, const bf16* x, int ldx, int M, int K, float gs,
                 bool swz, cudaStream_t st);

// NVFP4 expert w13 row order: gate / up rows interleaved by kGu64 (rows [gate f..f+63 |
// up f..f+63] per 128-row block), so one 128-row tile holds both halves of 64 features.
constexpr int kGu64 = 64;

// clamp-SwiGLU of gu [M, 2I] (gate | up, bf16-rounded as swiglu_bf16) quantized to NVFP4 with
// global scale gs: q [M, I/2]. Scales in the CUTLASS layout, dense (offsets null: one [M, I]
// operand) or per group (offsets / pad [G + 1]: row r of group g, offsets[g] <= r < offsets[g+1],
// at row r - offsets[g] of the group's operand whose scales start at sf + pad[g] * I / 16).
// m_dev (optional): device row count; rows past it are skipped. gu64: gu's columns in the kGu64
// interleave (gate j at (j / 64) * 128 + j % 64, up j at + 64) instead of [gate | up].
void swiglu_quant_nvfp4(uint8_t* q, uint8_t* sf, const bf16* gu, int M, int I, float limit,
                        float gs, cudaStream_t st, const int* offsets = nullptr,
                        const int* pad = nullptr, int G = 0, const int* m_dev = nullptr,
                        bool gu64 = false);

// Counting sort of (token, slot) pairs by group (map(id, token), G groups) for the grouped GEMM,
// copying each pair's token row: NVFP4 rows (x [T, K/2], scales sx in the CUTLASS layout of
// [T, K] (nvfp4_quant swz)) to
// x_perm [R, K/2] with the scales in the CUTLASS layout per group (sf_perm + pad[g] * K/16), or
// BF16 rows (x [T, K], sx null, row_bytes = 2K) to x_perm [R, K].
// Outputs offsets / pad [G + 1] (rows, rows padded to 128 per group) and pair_pos [T * K1]
// (row of each pair, -1 when not computed here). map.shared_mode 1: the shared slot's pair_pos is
// map.shared_rows + t (rows the separate shared expert writes).
// counts: [G] scratch.
void moe_permute_rows(uint8_t* x_perm, uint8_t* sf_perm, int* pair_pos, int* counts,
                      int* offsets, int* pad, const uint8_t* x, const uint8_t* sx,
                      const int* topk_ids, int T, int K1, const MoeGroupMap& map, int G, int K,
                      bool nvfp4, cudaStream_t st);

// D [M, N] (bf16, row stride ldd) = alpha * (A [M, K] . B [N, K]^T) with NVFP4 A / B: packed
// e2m1 rows and scales in the CUTLASS layout (A: [M, K], B: [N, K]). alpha = gs_A * gs_B.
void nvfp4_gemm(const uint8_t* A, const uint8_t* sfa, const uint8_t* B, const uint8_t* sfb,
                bf16* D, int ldd, int M, int N, int K, float alpha, void* workspace,
                size_t ws_bytes, cudaStream_t st);

// Grouped NVFP4 GEMM over experts (CUTLASS SM100 ptr-array block-scaled kernel; per-group
// arguments filled on device from offsets / pad, CUDA-graph capturable): for each group g with
// rows [offsets[g], offsets[g+1]): D[r, :N] = alpha[g] * A[r, :K] . W_g[N, K]^T.
// A [rows, K/2], its scales per group at sfa + pad[g] * K/16; W [G, N, K/2], scales [G] x
// nvfp4_sf_bytes(N, K); alpha: device [G]. rows_hint: the expected rows per group (tokens x top_k /
// experts; < 0: unknown), which picks the tile shape (see nvfp4_gemm.cu).
class Nvfp4GroupedGemm {
 public:
  Nvfp4GroupedGemm(int groups, int N, int K, int device);
  ~Nvfp4GroupedGemm();
  void run(const uint8_t* A, const uint8_t* sfa, const uint8_t* W, const uint8_t* sfw,
           const float* alpha, bf16* D, const int* offsets, const int* pad, cudaStream_t st,
           int rows_hint = -1);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Grouped BF16 GEMM over experts: D[r, :N] = A[r, :K] . W_g[N, K]^T for rows of group g.
class Bf16GroupedGemm {
 public:
  Bf16GroupedGemm(int groups, int N, int K, int device);
  ~Bf16GroupedGemm();
  void run(const bf16* A, const bf16* W, bf16* D, const int* offsets, cudaStream_t st);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace k
}  // namespace glm
