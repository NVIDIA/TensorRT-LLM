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
// KDA (linear attention) kernels.
#pragma once

#include <vector>

#include "../common.h"

namespace glm {
namespace k {

// Per-sequence metadata of a packed batch (device pointers, B sequences).
struct KdaSeqMeta {
  const int* cu_seqlens;  // [B + 1] token offsets of each sequence's new tokens
  const int* slots;       // [B] state slot of each sequence
  const int* has_state;   // [B] 1 if the slot holds this sequence's earlier state, 0 = fresh
  const int* replay = nullptr;  // [B] fused decode: accepted drafts of the previous step to
                                // re-apply from the replay records before the new tokens
  // Slot strides (elements) of the runtime's per-layer pools, 0 = dense: the fp32 recurrent state
  // [H, D, D] of a slot (state_ss) and its bf16 conv window [C, W-1] (conv_ss); the fused
  // decode's replay records rec [ndr][H][kKdaRec] fp32 of a slot (rec_ss); the drafts' raw conv
  // inputs: draft i's input of channel hc of section sec (q | k | v, crec_cw channels each) at
  // slot * crec_ss + sec * crec_sec + i * crec_rs + hc (kda_crec_off).
  int64_t state_ss = 0, conv_ss = 0, rec_ss = 0, crec_ss = 0, crec_sec = 0, crec_rs = 0;
};

// Replay record of one draft token for one head: normalized k | u | decay (kD each), with u the
// delta rule's update vector of the token (beta (v - (decayed state)^T k), from its step's pass):
// re-applying an accepted draft is state = decay * state + k u^T (no dot products with the state).
constexpr int kKdaRec = 3 * 128;

// meta with every zero stride replaced by the dense layout's (H heads of head dim D, C conv
// channels of kernel W, ndr replay records per slot)
inline KdaSeqMeta kda_dense_strides(KdaSeqMeta m, int H, int D, int C, int W, int ndr) {
  if (!m.state_ss) m.state_ss = (int64_t)H * D * D;
  if (!m.conv_ss) m.conv_ss = (int64_t)C * (W - 1);
  if (!m.rec_ss) m.rec_ss = (int64_t)ndr * H * kKdaRec;
  if (!m.crec_ss) m.crec_ss = (int64_t)ndr * C;
  if (!m.crec_sec) m.crec_sec = C / 3;
  if (!m.crec_rs) m.crec_rs = C;
  return m;
}

// element offset of draft i's raw conv input of channel hc of section sec in a slot's crec
__host__ __device__ __forceinline__ int64_t kda_crec_off(const KdaSeqMeta& m, int slot, int i, int sec, int hc) {
  return (int64_t)slot * m.crec_ss + (int64_t)sec * m.crec_sec + (int64_t)i * m.crec_rs + hc;
}

// Causal depthwise conv (kernel W) with SiLU over the C = 3*H*D q|k|v channels of `in`
// (row stride ld_in). conv_state: [slots, C, W-1] bf16, updated in place.
// The B sequences cover the T tokens [t_begin, t_begin + T) (= [cu_seqlens[0], cu_seqlens[B])).
void kda_conv(bf16* out, const bf16* in, int ld_in, const bf16* w, bf16* conv_state,
              const KdaSeqMeta& meta, int B, int C, int W, cudaStream_t st, int T,
              int t_begin = 0);

// Gated delta-rule recurrence. qkv: [T, 3*H*D] (post-conv), beta_raw: logits at column h of
// rows with stride ld_beta, g1: [T, H*D] raw gate (f_b output), state: [slots, H, D, D] fp32.
// o: [T, H*D] bf16.
void kda_recurrent(bf16* o, const bf16* qkv, const bf16* beta_raw, int ld_beta, const bf16* g1,
                   const float* a_log, const float* dt_bias, float* state,
                   const KdaSeqMeta& meta, int B, int H, int D, float lower_bound,
                   cudaStream_t st);

// Fused decode step (tokens_per_seq = 1..4 new tokens per sequence): short conv + f_b / g_b
// gate projections + gated delta rule + gated RMSNorm for every (sequence, head). proj: in_proj
// output rows (columns q | k | v | beta | f_a | g_a); out: [T, H*D] bf16 (o_proj input), rows
// at the tokens' absolute indices. Only the state / conv window after the first token is
// written; draft tokens (i >= 1) leave replay records rec (ndr [H][kKdaRec] fp32 records per slot)
// and crec (their raw conv inputs, bf16) at the meta's slot strides (dense: rec
// [slots][ndr][H][kKdaRec], crec [slots][ndr][3*H*D]), re-applied by the next step for the
// meta.replay[b] accepted drafts.
// g1 / g2 (optional, rows [T, H*D]): the gate projections f_b . f_a and g_b . g_a of every
// token computed beforehand (GEMMs, as in prefill) instead of inside the kernel.
// l2pf: the one-CTA-per-SM form prefetches each head's state and gate tiles into L2 at entry
// (attention DP, behind the in_proj GEMV).
void kda_decode_fused(bf16* out, const bf16* proj, int ld, const bf16* conv_w, bf16* conv_state,
                      const bf16* f_b, const bf16* g_b, const float* a_log, const float* dt_bias,
                      const bf16* o_norm, float* state, const KdaSeqMeta& meta, int B, int H,
                      int D, float lower_bound, float eps, cudaStream_t st,
                      int tokens_per_seq = 1, float* rec = nullptr, bf16* crec = nullptr,
                      int ndr = 0, const bf16* g1 = nullptr, const bf16* g2 = nullptr,
                      bool l2pf = false, bf16* lpfg = nullptr);
// (LP, the TP in_proj's Lamport copy of the projections at one sequence: GlmStep::layer_kda) kda_decode_fused with lpfg
// non-null reads proj as that copy: the in_proj's words polled (sentinel 0xFFFF), reset by their readers; lpfg = the other
// copy's f_a / g_a columns (rows 0-3), reset here (their readers were the previous KDA layer's CTAs). Only on the late co
// instance's G8 form (16-byte aligned projection rows): kda_decode_lp_path says whether kda_decode_fused takes it for
// these arguments
bool kda_decode_lp_path(int B, int H, int D, int tokens_per_seq, int ld, const void* proj, bool l2pf, bool pre);

struct KdaSnap {  // a prefix-reuse snapshot (fields: see kda_snapshot)
  int chunk, slot, dst, conv_row;
};

// Chunked prefill (64-token chunks): same inputs as kda_recurrent plus the chunk list
// (chunks[c] = {first token, tokens}, consecutive per sequence; seq_chunks [B + 1] offsets) and
// a scratch buffer of kda_chunk_scratch_floats(num_chunks, H) floats.
size_t kda_chunk_scratch_floats(int num_chunks, int H);
void kda_chunked(bf16* o, const bf16* qkv, const bf16* beta_raw, int ld_beta, const bf16* g1,
                 const float* a_log, const float* dt_bias, float* state, const KdaSeqMeta& meta,
                 const int2* chunks, const int* seq_chunks, int num_chunks, float* scratch, int B,
                 int H, int D, float lower_bound, cudaStream_t st, const KdaSnap* snaps = nullptr,
                 int n_snaps = 0, int b_long = 0);
// b_long (> 0): the sequences long enough to set the scan's critical path (the others take one or
// two chunks, e.g. attention DP's dummy prefill rows): the v3 scan sizes its column split for these
// snaps: this step's prefix-reuse snapshots (the chunks whose fp32 initial states they copy).

// Initial state of every chunk of the last kda_chunked call ([num_chunks][H][D][D] fp32, inside
// its scratch buffer).
const float* kda_chunk_states(const float* scratch, int num_chunks, int H);

// Prefix-reuse snapshots. KDA state of all L layers lives in [L][slots] arrays (layer strides
// state_ls floats / conv_ls elements; per slot H*D*D fp32 state and C*(W-1) bf16 conv window).
// kda_state_copy: for every pair {src slot, dst slot}, copy the state of every layer.
void kda_state_copy(float* state, size_t state_ls, bf16* conv, size_t conv_ls, int L,
                    const int2* pairs, int npairs, int H, int D, int C, cudaStream_t st);
// One layer's state at page boundaries inside this step's prefill tokens: e[i].chunk = chunk
// whose initial state is taken (-1: the sequence's slot state after the step), e[i].slot = the
// sequence's state slot, e[i].dst = snapshot slot, e[i].conv_row = token row (in `in`, row stride
// ld_in, raw conv inputs in columns [0, C)) of the first of the W-1 inputs before the boundary
// (-1: the slot's conv window after the step).
void kda_snapshot(float* state, bf16* conv_state, const float* chunk_states, const bf16* in,
                  int ld_in, const KdaSnap* e, int n, int H, int D, int C, int W,
                  cudaStream_t st);

// Attention DP, KDA prefill split over the ranks by heads: every rank's state / conv-state pools
// (layer 0 base; the layers follow at the pools' common layer strides), mapped into this process.
constexpr int kKdaMaxPeers = 8;
struct KdaPoolPeers {
  float* state[kKdaMaxPeers] = {};
  bf16* conv[kKdaMaxPeers] = {};
};
// One sequence's head slice moved between an owner rank's pools (slot oslot, Hf heads per slot)
// and a local slice pool (slot lslot, Hs heads per slot).
struct KdaXfer {
  int owner, oslot, lslot;
};
// dir 0: owner -> local, dir 1: local -> owner. The state slice is heads [h0, h0 + Hs) of the
// owner's slot (fp32 [Hf][D][D]); the conv slice is the q / k / v channel ranges of those heads
// (owner [3][Hf * D][Wm1] per slot, local [3][Hs * D][Wm1]). state_off / conv_off: the layer's
// offset into the owners' pools (elements).
void kda_slice_xfer(const KdaPoolPeers& peers, size_t state_off, size_t conv_off, float* lstate,
                    bf16* lconv, const KdaXfer* x, int n, int dir, int h0, int Hf, int Hs, int D,
                    int Wm1, cudaStream_t st);

// out = bf16(rmsnorm(o per head) * w) * sigmoid(g2); o, g2, out: [T, H*D].
void kda_out_norm(bf16* out, const bf16* o, const bf16* g2, const bf16* w, int T, int H, int D,
                  float eps, cudaStream_t st);

}  // namespace k
}  // namespace glm
