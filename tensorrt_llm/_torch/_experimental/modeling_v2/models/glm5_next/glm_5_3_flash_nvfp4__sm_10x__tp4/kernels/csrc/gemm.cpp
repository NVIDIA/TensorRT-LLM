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
#include "gemm.h"

#include "kernels/kernels.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace glm {

#define LT_CHECK(expr)                                                                   \
  do {                                                                                   \
    cublasStatus_t _s = (expr);                                                          \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                                   \
      fprintf(stderr, "cuBLASLt error %d at %s:%d: %s\n", (int)_s, __FILE__, __LINE__, #expr); \
      throw std::runtime_error("cuBLASLt error");                                        \
    }                                                                                    \
  } while (0)

namespace {
enum Kind { kBf16 = 0, kBf16Batched = 1 };
}  // namespace

Gemm::Gemm(size_t workspace_bytes) : ws_bytes_(workspace_bytes) {
  LT_CHECK(cublasLtCreate(&lt_));
  CUDA_CHECK(cudaMalloc(&ws_, ws_bytes_));
}

Gemm::~Gemm() {
  for (auto& [k, p] : plans_) {
    if (p.op) cublasLtMatmulDescDestroy(p.op);
    if (p.a) cublasLtMatrixLayoutDestroy(p.a);
    if (p.b) cublasLtMatrixLayoutDestroy(p.b);
    if (p.c) cublasLtMatrixLayoutDestroy(p.c);
  }
  if (ws_) cudaFree(ws_);
  if (lt_) cublasLtDestroy(lt_);
}

Gemm::Plan& Gemm::plan(const Key& key, int kind) {
  auto it = plans_.find(key);
  if (it != plans_.end()) return it->second;
  Plan p;
  const int M = std::get<1>(key), N = std::get<2>(key), K = std::get<3>(key);
  const int lda = std::get<4>(key), ldb = std::get<5>(key), ldc = std::get<6>(key);
  const int flag = std::get<7>(key);
  const long long strideA = std::get<8>(key), strideB = std::get<9>(key);
  const int batch = std::get<10>(key);
  const long long strideC = (long long)std::get<11>(key);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  LT_CHECK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  if (kind == kBf16) {
    LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, ldb));
    LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, lda));
    LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, flag ? CUDA_R_32F : CUDA_R_16BF, N, M, ldc));
  } else {  // batched bf16
    if (flag) {  // B_b stored [N, K]
      ta = CUBLAS_OP_T;
      LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, K, N, ldb));
    } else {  // B_b stored [K, N]
      ta = CUBLAS_OP_N;
      LT_CHECK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, N, K, ldb));
    }
    LT_CHECK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, K, M, lda));
    LT_CHECK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, N, M, ldc));
    int32_t bc = batch;
    for (auto [lay, stride] : {std::pair{p.a, strideB}, std::pair{p.b, strideA},
                               std::pair{p.c, strideC}}) {
      LT_CHECK(cublasLtMatrixLayoutSetAttribute(lay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc,
                                                sizeof(bc)));
      int64_t s = stride;
      LT_CHECK(cublasLtMatrixLayoutSetAttribute(lay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                &s, sizeof(s)));
    }
  }
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
  LT_CHECK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
  cublasLtMatmulPreference_t pref;
  LT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  LT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                &ws_bytes_, sizeof(ws_bytes_)));
  cublasLtMatmulHeuristicResult_t res[16];
  int n = 0;
  cublasStatus_t s =
      cublasLtMatmulAlgoGetHeuristic(lt_, p.op, p.a, p.b, p.c, p.c, pref, 16, res, &n);
  cublasLtMatmulPreferenceDestroy(pref);
  if (s != CUBLAS_STATUS_SUCCESS || n == 0) {
    fprintf(stderr, "no cuBLASLt algo: kind=%d M=%d N=%d K=%d status=%d\n", kind, M, N, K,
            (int)s);
    throw std::runtime_error("no cuBLASLt algorithm");
  }
  p.algo = res[0].algo;
  for (int i = 0; i < n; ++i) p.cands.push_back(res[i].algo);
  p.ok = true;
  return plans_.emplace(key, p).first->second;
}

namespace {
constexpr int kTuneMaxM = 128;  // the largest M the autotune times
}  // namespace

// Times every candidate algorithm on this call's operands (results overwritten by the real call
// afterwards): per candidate R x [flush L2, GEMM] minus R x [flush], median of 5, all ranks alike.
void Gemm::tune_bf16(Plan& p, const bf16* A, const bf16* B, void* C, float beta, int M, int N,
                     int K, cudaStream_t st) {
  p.tuned = true;
  if (p.cands.size() < 2) return;
  constexpr size_t kFlushBytes = 192ull << 20;
  int R = 8;  // repetitions per timing (fewer for long GEMMs, below)
  static thread_local void* flush = nullptr;
  static thread_local float* sink = nullptr;
  if (!flush) {
    CUDA_CHECK(cudaMalloc(&flush, kFlushBytes));
    CUDA_CHECK(cudaMemsetAsync(flush, 0, kFlushBytes, st));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
  }
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  const float alpha = 1.f;
  auto run = [&](int ci) -> float {  // ci < 0: flushes only; returns us per repetition
    std::vector<float> ts;
    for (int it = 0; it < 6; ++it) {
      CUDA_CHECK(cudaEventRecord(e0, st));
      for (int r = 0; r < R; ++r) {
        k::l2_flush(flush, kFlushBytes, sink, st);
        if (ci >= 0)
          cublasLtMatmul(lt_, p.op, &alpha, B, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.cands[ci],
                         ws_, ws_bytes_, st);
      }
      CUDA_CHECK(cudaEventRecord(e1, st));
      CUDA_CHECK(cudaEventSynchronize(e1));
      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
      if (it > 0) ts.push_back(ms * 1000.f / R);  // (the first: warm-up)
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
  };
  {  // one plain run of heuristic 0: repetitions for ~0.4 ms of GEMM time per timing
    CUDA_CHECK(cudaEventRecord(e0, st));
    cublasLtMatmul(lt_, p.op, &alpha, B, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.cands[0], ws_,
                   ws_bytes_, st);
    CUDA_CHECK(cudaEventRecord(e1, st));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    R = std::max(1, std::min(8, (int)(0.4f / std::max(ms, 1e-3f))));
  }
  const float base = run(-1);
  // heuristic 0 unless a candidate beats the best so far by 3%
  const float t0 = run(0) - base;
  int best = 0;
  float best_t = t0;
  for (int ci = 1; ci < (int)p.cands.size(); ++ci) {
    if (cublasLtMatmul(lt_, p.op, &alpha, B, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.cands[ci], ws_,
                       ws_bytes_, st) != CUBLAS_STATUS_SUCCESS) {
      cudaGetLastError();
      continue;
    }
    const float t = run(ci) - base;
    if (t < best_t * 0.97f) {
      best_t = t;
      best = ci;
    }
  }
  CUDA_CHECK(cudaEventDestroy(e0));
  CUDA_CHECK(cudaEventDestroy(e1));
  p.algo = p.cands[best];
}

void Gemm::mm_bf16(const bf16* A, int lda, const bf16* B, int ldb, void* C, int ldc, bool c_f32,
                int M, int N, int K, cudaStream_t st, float beta) {
  if (M == 0) return;
  Key key{kBf16, M, N, K, lda, ldb, ldc, c_f32 ? 1 : 0, 0, 0, 1, 0};
  Plan& p = plan(key, kBf16);
  if (!p.tuned && tune_on_ && beta == 0.f && M <= kTuneMaxM) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(st, &cs));
    if (cs == cudaStreamCaptureStatusNone) tune_bf16(p, A, B, C, beta, M, N, K, st);
  }
  const float alpha = 1.f;
  LT_CHECK(cublasLtMatmul(lt_, p.op, &alpha, B, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.algo, ws_,
                          ws_bytes_, st));
}

void Gemm::mm_bf16_batched(const bf16* A, int lda, long long strideA, const bf16* B, int ldb,
                        long long strideB, bool b_trans, bf16* C, int ldc, long long strideC,
                        int M, int N, int K, int batch, cudaStream_t st) {
  if (M == 0) return;
  Key key{kBf16Batched, M, N, K, lda, ldb, ldc, b_trans ? 1 : 0, strideA, strideB, batch,
          (int)strideC};
  Plan& p = plan(key, kBf16Batched);
  const float alpha = 1.f, beta = 0.f;
  LT_CHECK(cublasLtMatmul(lt_, p.op, &alpha, B, p.a, A, p.b, &beta, C, p.c, C, p.c, &p.algo, ws_,
                          ws_bytes_, st));
}

}  // namespace glm
