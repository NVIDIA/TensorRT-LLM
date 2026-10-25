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
// Sequence-parallel boundary collectives over NVLink peer memory (tensor parallelism, one process per rank):
// the reduce-scatter of a sublayer's partial output (pulls of the peers' rows) and the all-gather of the
// layer input (stores into the peers' rows), with per-source arrival flags instead of NCCL ring collectives.
#pragma once

#include "../common.h"

namespace glm {
namespace k {

constexpr int kSpMaxTp = 8;

// every rank's full-row buffers (sublayer output b.out, layer input b.x, [rows][hidden] bf16) and
// arrival flags ([2 kinds][kSpMaxTp sources] epochs, zeroed)
struct SpPeers {
  bf16* out[kSpMaxTp];
  bf16* x[kSpMaxTp];
  uint32_t* flags[kSpMaxTp];
  fp8* xq[kSpMaxTp];    // the layer input's FP8 1x128 quantization (b.xq / b.xs), all-gathered
  float* xs[kSpMaxTp];  // with x by sp_pre_push when the boundary emits it
};
constexpr int kSpFlagWords = 4 * kSpMaxTp;  // [rs epochs, ag epochs, ag arrival counters, sp_pre_push
                                            //  arrival counters][src]
constexpr int kSpStateWords = 6;  // [rs epoch, ag epoch, rs block counter, ag block counter,
                                  //  sp_pre_push epoch, sp_pre_push block counter]

// Reduce-scatter in place: rows [q0, q0 + nrows) of this rank's out = the sum (fp32, rank order)
// of every rank's out rows; returns when done. Every rank's out must be complete (stream order).
void sp_rs_pull(const SpPeers& peers, int tp, int rank, int q0, int nrows, int H, uint32_t* state,
                cudaStream_t st);
// All-gather in place as stores: this rank's rows [rank * Tq, min((rank + 1) * Tq, T)) of x into every
// peer's x, then a wait until every peer's rows arrived (with all ranks moving data, stores to peers
// reach a higher bandwidth than loads of one source at a time).
void sp_ag_push(const SpPeers& peers, int tp, int rank, int Tq, int T, int H, uint32_t* state,
                cudaStream_t st);

// The reduce-scatter fused with the mHC post-mapping of this rank's rows (bitwise sp_rs_pull, then
// mhc_post): streams rows [0, nrows) at sq (= the streams of global rows q0..) become
// comb^T streams + post * (the rank-order fp32 sum of every rank's out row, rounded to bf16), in
// one pass in which the pulls of the peers' rows and the streams' reads and writes overlap. This
// rank's out rows are not written (the sum lives only in the streams).
void sp_rs_post(const SpPeers& peers, int tp, int rank, int q0, int nrows, int H, bf16* sq,
                const float* post, const float* comb, uint32_t* state, cudaStream_t st);
// mhc_pre_apply of this rank's rows (bitwise its outputs: x, the optional FP8 xq / xs, post and
// comb) fused with the all-gather of the layer input as stores into every peer's x (and xq / xs),
// row by row as each row is done, then a wait until every peer's rows arrived. sq / mixes / post /
// comb / rs: this rank's rows (from global row q0); x / xq / xs: the full-row buffers. Shares
// sp_ag_push's epoch (one of the two per boundary). hidden 4096.
void sp_pre_push(const SpPeers& peers, int tp, int rank, int q0, int nrows, int H, float* post,
                 float* comb, bf16* x, fp8* xq, float* xs, const bf16* sq, const float* mixes,
                 const float* base, const float* scale, const bf16* norm_w, float rms_eps,
                 float hc_eps, int iters, float post_mult, float norm_eps, float* rs,
                 uint32_t* state, cudaStream_t st);

}  // namespace k
}  // namespace glm
