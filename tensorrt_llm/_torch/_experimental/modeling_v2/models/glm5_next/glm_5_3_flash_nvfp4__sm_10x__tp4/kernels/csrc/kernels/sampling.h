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
// Token sampling over vocab-sharded logits.
#pragma once

#include "../common.h"
#include "allreduce.h"

namespace glm {
namespace k {

struct SampleParams {
  float temperature;  // <= 0 -> greedy
  float top_p;        // 1 -> off
  int top_k;          // <= 0 -> off
  float min_p;        // 0 -> off
  uint64_t seed;
  uint64_t step;      // advances the RNG stream per generated token
  int num_logprobs;   // -1 -> none, else top-N alternatives (<= 20)
};

// Per-row argmax over a vocab shard [rows, shard_vocab] encoded as a max-reducible key; the
// global token after a max all-reduce over shards is argmax_key_token(key). With params, rows
// with temperature > 0 (and top_k != 1) take the argmax of logits / T + Gumbel noise keyed by
// (seed, step, vocab index): an exact sample of softmax(logits / T), the same on every rank --
// for rows without top-p / top-k / min-p (callers check). shard_vocab, vocab_offset: multiples of 4.
void argmax_keys(unsigned long long* keys, const float* logits, int rows, int shard_vocab,
                 int vocab_offset, cudaStream_t st, const SampleParams* params = nullptr);
// tokens[i] = argmax_key_token(keys[i])
void keys_to_tokens(int* tokens, const unsigned long long* keys, int n, cudaStream_t st);
__host__ __device__ inline int argmax_key_token(unsigned long long key) {
  return (int)(0xffffffffu - (uint32_t)(key & 0xffffffffull));
}

// The vocab-parallel LM head of at most kHeadKeysMaxRows rows fused with argmax_keys (and with peers, custom_max_u64):
// keys[m] = the argmax_keys key of row m over logits A[m] x W^T of this rank's vocabulary slice (W [N, K], global
// indices vocab_offset + n; with params, Gumbel-max keys exactly as argmax_keys draws them), then the max over the
// ranks. The logits are those of gemv_tma (fp32 tensor-core sums in a fixed order, not cuBLAS's); none are stored.
// part / count: scratch of head_keys_scratch_bytes(), zeroed once (the last CTA resets the counter).
constexpr int kHeadKeysMaxRows = 8, kHeadKeysMaxCtas = 256;
struct HeadKeysArgs {
  const bf16* A = nullptr;  // [M, lda]
  int lda = 0;
  const bf16* W = nullptr;  // [N, ldw]
  int ldw = 0;
  int M = 0, N = 0, K = 0;
  int vocab_offset = 0;
  const SampleParams* params = nullptr;  // per row, or null: greedy
  unsigned long long* keys = nullptr;    // [M]
  int* tokens = nullptr;                 // optional [M]: argmax_key_token of the reduced keys
  unsigned long long* part = nullptr;    // scratch [kHeadKeysMaxCtas][kHeadKeysMaxRows]
  unsigned int* count = nullptr;         // scratch
  // the cross-rank max (tp > 1): custom_max_u64's peers and state
  ArPeers peers{};
  uint32_t* state = nullptr;
  int rank = 0, tp = 1;
  bool late_trigger = true;  // let the next kernel launch once this one's wait is over (else at its start)
  int tokens_ld = 1;         // tokens[m * tokens_ld] (the MTP drafts' layout [B, ndraft])
  // (the MTP passes' heads after the combine norm) the rows A are complete once *ready reaches
  // ready_target (each of the norm's row CTAs adds 1 after its stores): the producer polls it instead of the PDL wait,
  // which every thread still passes before it exits (the grid completes after its predecessor); the last CTA resets it
  unsigned int* ready = nullptr;
  unsigned int ready_target = 0;
};
inline size_t head_keys_scratch_bytes() {
  return (size_t)kHeadKeysMaxCtas * kHeadKeysMaxRows * sizeof(unsigned long long) + 256;
}
// rows <= kHeadKeysMaxRows, K % 1024 == 0
bool head_keys_fits(int M, int N, int K);
void head_keys(const HeadKeysArgs& a, int ctas, cudaStream_t st, int ns_max = 4);

// MTP speculative decoding, one CTA per sampled sequence s with sample rows
// [row0[s], row0[s] + nrows[s]) (the last context token, then the fed drafts). The accepted
// draft count a is the longest prefix with sampled[r0 + i] == fed token at row r0 + i + 1. The
// first new draft is the greedy MTP token of row r0 + a (keys); it, the position after row
// r0 + a and that row's MTP hidden state (g) seed the chained draft pass (next_*; next_hidden
// may be null when no further pass runs). force_accept >= 0 (forced acceptance, TRT-LLM's
// TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS): a = floor(force_accept), plus one with probability
// frac(force_accept) drawn from (slot, position), capped at the fed drafts.
// sel_src (optional): the step's indexer selections ([T, sel_n]) of the MTP layer; each sequence's
// row of its last accepted token goes to sel_dst[s] and that token's complete pools
// ((position + 1) / kpool) to vis_dst[s] (the chained passes' shared selection)
// With eh: the next pass's MTP input row of every sequence, exactly as mtp_eh_norm computes it from
// (the new draft token, the hidden row this kernel would copy to next_hidden), written to eh.out
// [B, 2 hidden] (the next pass then skips mtp_eh_norm; 256 threads per sequence at hidden 4096).
struct MtpEhArgs {
  bf16* out = nullptr;
  const bf16* embed = nullptr;
  const bf16* enorm = nullptr;
  const bf16* hnorm = nullptr;
  float eps = 0.f;
};
// compact: the pass-1 head ran at each sequence's last accepted row only (mtp_pick): keys and g
// rows are per sequence (row s) instead of per sample row.
void mtp_accept(int* accepted, int* drafts, int ndraft, int* next_ids, int* next_pos,
                int* next_slot, bf16* next_hidden, const unsigned long long* keys, const bf16* g,
                const int* row0, const int* nrows, const int* sampled, const int* ids,
                const int* sample_idx, const int* positions, const int* token_slot, int B,
                int hidden, cudaStream_t st, float force_accept = -1.f,
                const int* sel_src = nullptr, int* sel_dst = nullptr, int* vis_dst = nullptr,
                int sel_n = 0, int kpool = 1, const MtpEhArgs* eh = nullptr,
                bool compact = false);
// Pass 1 of the MTP layer runs its MoE, head norm and draft head at each sampled sequence's last
// accepted row only (the other rows' outputs feed nothing: the attention above already wrote every
// row's MTP KV): sequence s's last accepted row t (counted as mtp_accept does, from the same
// inputs) is copied from rows t of x / h (the MoE input and the residual; xq / xs: the FP8 input,
// optional) to rows s of xp / hp (xqp / xsp); slotp (optional): its state slot (attention DP: the
// MoE exchange skips the dummy slot's rows).
void mtp_pick(bf16* xp, bf16* hp, fp8* xqp, float* xsp, int* slotp, const bf16* x, const bf16* h,
              const fp8* xq, const float* xs, float force_accept, const int* row0,
              const int* nrows, const int* sampled, const int* ids, const int* sample_idx,
              const int* positions, const int* token_slot, int B, int hidden, cudaStream_t st);
// After chained draft pass j (j >= 2, one token per sequence): drafts[s * ndraft + j - 1] and
// the inputs of the next pass (position + 1, hidden = g of this pass; next_hidden may be null).
void mtp_next(int* drafts, int ndraft, int j, int* next_ids, int* next_pos, bf16* next_hidden,
              const unsigned long long* keys, const bf16* g, int B, int hidden, cudaStream_t st,
              const MtpEhArgs* eh = nullptr);

}  // namespace k
}  // namespace glm
