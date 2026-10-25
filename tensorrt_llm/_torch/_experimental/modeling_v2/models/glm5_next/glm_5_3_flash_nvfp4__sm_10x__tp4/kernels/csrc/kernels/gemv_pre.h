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
// BF16 GEMV for few rows (decode) with the weights preloaded into shared memory before the PDL
// wait: D[M, N] = A[M, K] x W[N, K]^T.
#pragma once

#include "../common.h"

namespace glm {
namespace k {

// Producer side of the next mHC boundary's fused one-shot all-reduce (ArFuse.pushed; tp == 4,
// D bf16 [M][hidden] with ldd == hidden): the outputs, stored as 16-byte chunks of 8 columns of one
// row, are also pushed (-0.0 mapped to +0.0) into every rank's Lamport buffer of the call that
// follows (epoch next_epoch(state[0])) as the boundary would push them, so the boundary starts by polling.
struct GemvArPush {
  bf16* buf[4] = {};
  const uint32_t* state = nullptr;  // null: off
  int rank = 0;
  size_t max_elems = 0;
};

struct GemvPreArgs {
  const bf16* A = nullptr;  // [M, lda]
  int lda = 0;
  const bf16* W = nullptr;  // [N, ldw]
  int ldw = 0;
  void* D = nullptr;        // [M, ldd] bf16, or fp32 with d_f32
  int ldd = 0;
  bool d_f32 = false;
  int M = 0, N = 0, K = 0;
  // let the next kernel launch only once this one's wait is over (else at its start): a kernel
  // launched that early stays resident for the whole previous kernel
  bool late_trigger = false;
  // head-batched (gemv_pre only): CTA row blockIdx.y = batch b, whose A / W / D start b x sA /
  // sW / sD elements further (the MLA's per-head decode projections)
  int batch = 1;
  long long sA = 0, sW = 0, sD = 0;
  GemvArPush push{};
};

constexpr int kGemvPreMaxRows = 16;
// whether gemv_pre handles these shapes (rows <= 16, K % 64 == 0, the weights fit the shared
// memory of at most max_ctas CTAs)
bool gemv_pre_fits(int M, int N, int K, int max_ctas, int batch = 1);
// (carveout >= 0: the launch's preferred shared-memory carveout, a launch attribute; -1: the instance's own)
void gemv_pre(const GemvPreArgs& a, int max_ctas, cudaStream_t st, int carveout = -1);
// gemv_pre's products and warp-order sums for M <= 8 and K = 2048 in CTAs of 32 weight rows, staged by TMA bulk copies
// (one per row, padded) before the PDL wait, 141 KB + the activation rows: the KDA o_proj (TP4, N 4096) in 128 CTAs
// (gemv_pre: 86 CTAs of 48 rows, 2 of which find no SM beside the KDA decode). Bitwise gemv_pre.
bool gemv_pre_lean_fits(int M, int N, int K);
void gemv_pre_lean(const GemvPreArgs& a, cudaStream_t st);
// Weights too large for that: the first k-chunks preloaded before the PDL wait, the rest streamed
// after it (rows <= 16, K % 512 == 0, at most min(max_ctas, 116) CTAs).
// (cap: the CTA count limit, default kStreamMaxCtas = one wave beside the neighbouring kernels' CTAs;
// a call may pass up to the SM count)
bool gemv_stream_fits(int M, int N, int K, int max_ctas, int cap = 0);
void gemv_stream(const GemvPreArgs& a, int max_ctas, cudaStream_t st, int cap = 0);
// Pipelined GEMV for rows <= 16 of any size: CTAs of 16-64 weight rows (one wave of at most
// max_ctas), each warp streaming an eighth of K through its own cp.async ring (K % 512 == 0).
bool gemv_pipe_fits(int M, int N, int K, int max_ctas);
void gemv_pipe(const GemvPreArgs& a, int max_ctas, cudaStream_t st);
// gemv_pipe for at most 8 rows: 8-row activation stages, more ring stages (bitwise gemv_pipe)
bool gemv_pipe8_fits(int M, int N, int K, int max_ctas);
void gemv_pipe8(const GemvPreArgs& a, int max_ctas, cudaStream_t st);
// gemv_pipe's pushed form with its push epoch loaded before the PDL wait (bitwise gemv_pipe)
void gemv_pipe_ep(const GemvPreArgs& a, int max_ctas, cudaStream_t st);
// Weights streamed with TMA bulk copies (gemv_tma.cu): rows <= 8, any N (each of `ctas` CTAs owns an equal range of
// weight rows), K % 1024 == 0; bf16 or fp32 output; no batched / pushed forms.
bool gemv_tma_fits(int M, int N, int K);
// The MTP layer's chained passes' input rows built by gemv_tma itself (mode 1; A unused): after the wait the producer
// warp's lane m bulk-copies row m's embedding row (token argmax_key_token(keys[m])) and hidden row into the activation
// rows, which the consumer warps normalize in place, [rmsnorm(embed[id_m]) * enorm, rmsnorm(hid[m]) * hnorm] with
// mtp_eh_row's arithmetic and thread mapping (16 columns per thread, hidden = 16 x 256; the sums in block_sum's order):
// bitwise the rows mtp_next / mtp_accept write. K = 2 hidden.
struct GemvEhArgs {
  int mode = 0;
  const unsigned long long* keys = nullptr;
  const bf16* embed = nullptr;
  const bf16* hid = nullptr;
  const bf16* enorm = nullptr;
  const bf16* hnorm = nullptr;
  float eps = 0.f;
};
// ns_max: the ring's stages at most (4 by default; 5 / 6 where the activations leave room); eh (mode 1): the rows
// built in the kernel (GemvEhArgs; hidden 4096, K 8192, the pushed form); rt 8 (the pushed form, loaded rows): tiles
// of 8 weight rows (twice the CTAs, half-size stages: 4-8 of them; the same sums); pace D > 0 (the pushed form on its
// 4-stage ring, loaded rows): before the PDL wait at most D stages in flight, the activation rows issued by a consumer
// warp at its wait (the attention's o_proj; the same sums)
void gemv_tma(const GemvPreArgs& a, int ctas, cudaStream_t st, int ns_max = 4, const GemvEhArgs* eh = nullptr,
              int rt = 16, int pace = 0);
// gemv_tma (4-stage ring) with the last nl of `ctas` CTAs owning rl rows each and the others the rest in equal ranges
// (the TP KDA in_proj at <= 4 rows: the last 32 CTAs become resident only when the MoE-side boundary's 32 CTAs exit,
// with nothing preloaded). Bitwise gemv_tma (rows per CTA do not enter a sum). Ring: 5 stages where they fit, 3 issued
// before the PDL wait and 2 after the activation rows (see gemv_tma_kernel's PRE). pd 1-5: the paced form instead (all
// 5 stages before the wait, at most pd in flight, the activation rows by a consumer warp at its wait; for an in_proj
// launched early by the boundary before it; see gemv_tma_kernel's PD). rw 3-5: the register-staged
// form (the first rw stages copied into the consumer warps' registers before the wait, so that rw + 5 stages are on chip;
// ring pd 1-3 -> paced at 3, else unpaced; gemv_tma_kernel's RW).
void gemv_tma_uneven(const GemvPreArgs& a, int ctas, int nl, int rl, cudaStream_t st, int pd = 0, int rw = 0);

}  // namespace k
}  // namespace glm
