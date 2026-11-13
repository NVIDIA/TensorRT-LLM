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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
//! One pass of the fused greedy Eagle3 draft loop of the MiniMax-M3 NVFP4 TP4 decode path (sm_103a): the BF16
//! vocab-sharded draft head, the cross-rank argmax and the draft loop's per-pass bookkeeping in one kernel. See
//! draftStep.cu.
namespace draftStep
{

constexpr int kMaxTokens = 8;       // head rows (requests) of one launch
constexpr int kWarps = 16;          // warps per CTA
constexpr int kMaxTilesPerCta = 32; // 16-row weight tiles one CTA can own
//! Shared-memory weight prologue per CTA, in KB (capped by the free shared memory).
constexpr int kPrefetchKb = 112;

// Draft-pass epilogue (last CTA, one thread per request row b): stores the token as draft token `step`, stages the
// next pass's input id and position, and advances the draft KV lengths the way the in-tree draft loop does between
// passes (pass 0 with num_acc: drop the rejected verify rows and switch to one token per request; later passes: +1).
struct StepArgs
{
    int* draft_out; // [nreq, draft_stride]: column `step` receives the token
    int draft_stride;
    int step;
    int* ids_next;      // [nreq]
    int const* pos_src; // positions of the rows the head read (indexed like x)
    int* pos_next;      // [nreq] = pos_src[x_rows[b] or b] + 1
    int* kv_lens;       // [nreq] draft KV lengths, updated in place
    int const* num_acc; // pass 0: accepted counts (nullptr: no rollback)
    int* seq_lens;      // pass 0: set to 1
    int draft_len;
};

// The 12-bit coded head (see draftStep.cu): the per-tile exception lists and the exponent base. exc_off == nullptr:
// the head is the plain BF16 fragment-ordered copy.
struct CodedArgs
{
    int const* exc_off;      // [n_tiles + 1] exception list offsets
    uint32_t const* exc_pos; // row in tile << 16 | k, per exception
    float2 const* exc_w;     // {w, 0} per exception: the exact weight
    uint32_t base;           // exponent base << 7 in both BF16 halves (base << 7 | base << 23)
};

// Everything a draft-step launch takes (see invokeDraftStep).
struct Params
{
    __nv_bfloat16 const* x;  // [*, K] head input rows
    long long const* x_rows; // optional [M]: head input row of request b (pass 0: the last accepted verify row)
    int M;                   // requests, <= kMaxTokens
    uint4 const* wf;         // [n_tiles][K / 16][32 lanes] 16-byte BF16 A fragments, or the 12-bit coded stream
                             // [n_tiles][K / 64][3][32 lanes] when cd.exc_off is set (see draftStep.cu)
    int n_tiles;
    int vocab_offset;        // global id of this rank's first head row
    unsigned long long* keys;
    unsigned int* counters;
    unsigned long long* ws_uc; // this rank's exchange slots [2][kMaxTokens][world]
    unsigned long long* ws_mc; // their multicast address
    int rank;
    int world;
    unsigned long long* out_keys; // optional [M]: the winning global key of each row
    StepArgs st;
    CodedArgs cd;
};

//! Launch the draft step over `grid` 512-thread CTAs (K = the hidden size, a multiple of 16; of 64 for a coded head).
cudaError_t invokeDraftStep(Params const& p, int K, int grid, bool pdl, cudaStream_t stream);

} // namespace draftStep
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
