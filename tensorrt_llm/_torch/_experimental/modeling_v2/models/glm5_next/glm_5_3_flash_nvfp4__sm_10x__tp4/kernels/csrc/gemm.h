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
// cuBLASLt GEMM wrappers. Row-major convention: C[M, N] = A[M, K] * B[N, K]^T
// (A = activations, B = weights stored [out, in]).
#pragma once

#include <cublasLt.h>

#include <map>
#include <tuple>
#include <vector>

#include "common.h"

namespace glm {

class Gemm {
 public:
  Gemm(size_t workspace_bytes = 64ull << 20);
  ~Gemm();
  // bf16 x bf16 -> bf16 or fp32. lda/ldb/ldc in elements (row strides). beta accumulates into C.
  void mm_bf16(const bf16* A, int lda, const bf16* B, int ldb, void* C, int ldc, bool c_f32, int M,
            int N, int K, cudaStream_t st, float beta = 0.f);
  // Batched bf16: for b < batch: C_b[M, N] = A_b[M, K] * op(B_b) where B_b is [N, K] (b_trans
  // = true) or [K, N] (b_trans = false). Strides in elements.
  void mm_bf16_batched(const bf16* A, int lda, long long strideA, const bf16* B, int ldb,
                    long long strideB, bool b_trans, bf16* C, int ldc, long long strideC, int M,
                    int N, int K, int batch, cudaStream_t st);
  // Autotune of the bf16 algorithm: while enabled, the first call of a new bf16 shape with M <= 128
  // times cuBLASLt's heuristic candidates with the weights cold in L2 and keeps the fastest (the
  // model enables it for the eager run before a decode step's CUDA-graph capture)
  void set_tune(bool on) { tune_on_ = on; }

 private:
  struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo;
    bool ok = false;
    bool tuned = false;
    std::vector<cublasLtMatmulAlgo_t> cands;  // heuristic candidates, in heuristic order
  };
  void tune_bf16(Plan& p, const bf16* A, const bf16* B, void* C, float beta, int M, int N, int K,
                 cudaStream_t st);
  bool tune_on_ = false;
  using Key = std::tuple<int, int, int, int, int, int, int, int, long long, long long, int, int>;
  Plan& plan(const Key& key, int kind);
  cublasLtHandle_t lt_ = nullptr;
  void* ws_ = nullptr;
  size_t ws_bytes_ = 0;
  std::map<Key, Plan> plans_;
};

}  // namespace glm
