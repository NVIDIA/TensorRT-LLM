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
// Custom one-shot all-reduce over NVLink peer memory (one process per rank; comm.py IPC-maps every rank's buffers).
#pragma once

#include "../common.h"

namespace glm {
namespace k {

constexpr int kMaxTp = 8;

struct ArPeers {
  // per rank (allreduce_buf_elems): [3 one-shot Lamport buffers (sentinel-filled) + 2
  // flag-protocol parities][tp sources][max_elems], then 3 two-shot Lamport buffers
  bf16* buf[kMaxTp];
  uint32_t* flags[kMaxTp];  // per rank: [2 phases][2 parities][tp sources][max_blocks] epochs
  unsigned long long* keys[kMaxTp];  // per rank: [3][tp][kMaxKeyRows] max-reduce slots (0 = empty)
};

// Max all-reduce of nonzero 64-bit keys (e.g. argmax keys), n <= kMaxKeyRows, Lamport-style over
// the peers' key slots (zero-initialized). state: this rank's [2] counters (zero-initialized).
constexpr int kMaxKeyRows = 1024;
// tokens (optional): argmax_key_token of each reduced key.
void custom_max_u64(unsigned long long* keys, int n, const ArPeers& peers, uint32_t* state,
                    int rank, int tp, cudaStream_t st, int* tokens = nullptr);

// In-place sum of `data` (n bf16 elements, n % 8 == 0) across all ranks. `epochs` is this
// rank's [max_blocks + 4] counter array (zero-initialized, same on all ranks). Returns false if the
// message does not fit (caller falls back to NCCL).
bool custom_allreduce(bf16* data, size_t n, const ArPeers& peers, uint32_t* epochs, int rank,
                      int tp, size_t max_elems, int max_blocks, cudaStream_t st);
// Lamport protocol for messages up to 64 x 4096 elements: its 3 * tp * max_elems receive elements
// must be filled with the sentinel before first use.

// MoE layer output fused into the one-shot Lamport all-reduce: the pushed value is
// bf16(sum_s w[t, s] * y[pair_pos[t * K1 + s], :]) (= moe_combine) and `out` receives the
// all-reduced result. Returns false (nothing launched) if the message does not use the one-shot
// Lamport protocol; the caller then runs moe_combine + custom_allreduce.
struct MoeCombineArgs {
  const bf16* y = nullptr;
  const float* w = nullptr;
  const int* pair_pos = nullptr;
  int K1 = 0;
  int hidden = 0;
};
// pairs per token (top_k + 1) up to which the fused combine loads a token's pairs at once
constexpr int kArMaxK1 = 12;
// A one-shot Lamport all-reduce fused in front of another kernel (the mHC boundary): that
// kernel pushes its blocks' slices of the partial sums (data, or the MoE combine of cm) into every
// rank's buffer, sums the arrivals in rank order (the result of allreduce_lamport_kernel) and
// takes part in the buffer rotation (clearing, epoch). state = epochs + max_blocks of the
// custom all-reduce.
struct ArFuse {
  ArPeers peers{};
  uint32_t* state = nullptr;  // null: no fused all-reduce
  int rank = 0, tp = 0;
  size_t max_elems = 0;
  MoeCombineArgs cm{};
  // the producer of the partial sums (the attention's o_proj GEMV, GemvArPush) has already pushed
  // them into every rank's buffer: the consumer only polls, sums and takes part in the rotation
  int pushed = 0;
  // deferred Sinkhorn between consecutive cluster boundaries (mhc_sk_ok): 1 this boundary writes
  // post and the comb LOGITS (its Sinkhorn left to the next boundary); 2 comb holds the previous
  // boundary's logits: this boundary's spare warp runs that Sinkhorn while its all-reduce is in
  // flight and hands comb to the post-mapping through shared memory
  int sk = 0;
  // (RF: an attention-side boundary -- no MoE combine, cm.y null -- may carry in cm.w the address of the tensor-core
  // router's x copies, ar_rf_x)
};
// RF (the router's x by the Lamport protocol): the attention-side boundary before a MoE layer also writes its x_out rows
// into kRouterRfGroups copies [group][kRouterRfT][hidden] (every bf16 -0.0 as +0.0), which the tensor-core router's RF
// form (moe_router_mma_ks_rf: the CTAs of expert group g read copy g; E = 288 experts, 16 per group) polls instead of
// waiting for the boundary's grid
constexpr int kRouterRfGroups = 18, kRouterRfT = 4;
__host__ __device__ inline bf16* ar_rf_x(const ArFuse& a) {
  return a.cm.y ? nullptr : reinterpret_cast<bf16*>(const_cast<float*>(a.cm.w));
}
bool custom_allreduce_moe(bf16* out, const MoeCombineArgs& cm, size_t n, const ArPeers& peers,
                          uint32_t* epochs, int rank, int tp, size_t max_elems, int max_blocks,
                          cudaStream_t st);
// whether a one-shot Lamport all-reduce of n elements may be fused (ArFuse) into a consumer
bool allreduce_fusable(size_t n, int tp, size_t max_elems);
// The MTP layer's one-shot Lamport call of T rows of `hidden` whose data the producer GEMVs pushed (GemvArPush, a late
// trigger), consumed by the add_rmsnorm after it (tp 4): gather = false, every rank pushed partial sums of whole rows
// (the o_proj): add_rmsnorm(resid, <the rank-order sums>, w, out, xq, xs, write_resid) -- bitwise allreduce_lamport +
// add_rmsnorm; gather = true, rank q pushed its columns [q hidden / 4, (q + 1) hidden / 4) (a column-parallel GEMV): the
// gathered rows go to resid (the other ranks' columns; the producer stored its own) and out = add_rmsnorm of them
// without a delta. The arrivals are polled before the PDL wait. late: the next kernel launches after the rows.
// pick (optional, the summing form; k::mtp_pick's job): each of the B sequences' last accepted row t (counted as
// mtp_accept does) also goes to row s of hp (the residual), xp (out) and xqp / xsp (the FP8 rows).
struct MtpPickArgs {
  bf16* xp = nullptr;
  bf16* hp = nullptr;
  fp8* xqp = nullptr;
  float* xsp = nullptr;
  float force_accept = -1.f;
  const int* row0 = nullptr;
  const int* nrows = nullptr;
  const int* sampled = nullptr;
  const int* ids = nullptr;
  const int* sample_idx = nullptr;
  const int* positions = nullptr;
  const int* token_slot = nullptr;
  int B = 0;
  // (optional) mtp_accept's bookkeeping, done by the clearing CTAs (off the rows' path; one sequence each) for every
  // sequence s: accepted[s] (the accepted drafts), next_pos[j B + s] = position + 1 + j (j < npos: the chained
  // passes' positions), next_slot[s]; sel_dst / vis_dst (optional): the last accepted token's selection row of
  // sel_src ([T, sel_n]) and its complete pools, the chained passes' shared selection
  int* accepted = nullptr;
  int* next_pos = nullptr;
  int npos = 0;
  int* next_slot = nullptr;
  const int* sel_src = nullptr;
  int* sel_dst = nullptr;
  int* vis_dst = nullptr;
  int sel_n = 0, kpool = 1;
};
bool mtp_lamport_norm_ok(int T, int hidden, int tp, size_t max_elems);
void mtp_lamport_norm(int T, int hidden, bool gather, const ArPeers& peers, uint32_t* epochs, int rank, int tp,
                      size_t max_elems, int max_blocks, bf16* resid, bool write_resid, const bf16* w, bf16* out,
                      fp8* xq, float* xs, float eps, bool late, cudaStream_t st, const MtpPickArgs* pick = nullptr);
// The MTP MoE's one-shot Lamport all-reduce of the combine of cm (T rows; the chunks combined and pushed by every CTA of
// the grid after its PDL wait) consumed by the head norm: out = add_rmsnorm(resid, <the rank-order sums>, w) (resid
// not written) -- bitwise custom_allreduce_moe + add_rmsnorm. late: the next kernel launches at this grid's start (0),
// after the rows (1) or once the rows have arrived, before their arithmetic (2).
// ready (optional): each row's CTA adds 1 once its out row is stored (fenced for the generic and async proxies): the
// MTP head's rows-ready count (HeadKeysArgs::ready), polled instead of its PDL wait.
void mtp_lamport_combine_norm(int T, int hidden, const MoeCombineArgs& cm, const ArPeers& peers, uint32_t* epochs,
                              int rank, int tp, size_t max_elems, int max_blocks, bf16* resid, const bf16* w,
                              bf16* out, float eps, int late, cudaStream_t st, unsigned* ready = nullptr);
// SR (mtp_gn.cu; the MTP layer's chained passes): the summing form of mtp_lamport_norm (the o_proj's pushed
// all-reduce + the residual add + the post-attention norm; resid written, out = x) folded into the tensor-core router
// (moe_router_mma's logits of x for E experts): bitwise the two launches; 1 <= T <= 4 rows
struct MtpGnArgs {
  ArPeers peers{};
  uint32_t* state = nullptr;  // the one-shot calls' state (epochs + max_blocks)
  int rank = 0;
  size_t max_elems = 0;
  bf16* resid = nullptr;     // the residual rows (the sums written back by cluster 0)
  const bf16* w = nullptr;   // the post-attention norm's weights
  bf16* out = nullptr;       // the normalized rows x (cluster 0)
  float eps = 0.f;
};
bool mtp_sr_ok(int T, int hidden, int tp, size_t max_elems, int E);
void mtp_sr_router(float* logits, const bf16* W, int T, int E, int H, const MtpGnArgs& g, cudaStream_t st);
// receive buffer elements per rank and its one-time initialization
size_t allreduce_buf_elems(int tp, size_t max_elems);
void allreduce_lamport_init(bf16* buf, int tp, size_t max_elems, cudaStream_t st);

}  // namespace k
}  // namespace glm
