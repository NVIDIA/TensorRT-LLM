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
//! Fused sampled (temperature + top-p) verification of the MiniMax-M3 Eagle3 target head over the TP-sharded
//! vocabulary, decode rows <= kMaxRows; see sampledAccept.cu.
namespace sampledAccept
{

constexpr int kMaxSplits = 8; // CTAs per verify row (at most)
constexpr int kMaxRows = 48;  // verify rows per call
constexpr int kMaxWorld = 8;  // TP ranks
constexpr int kMaxVs = 51200; // vocab shard width
constexpr int kIdBits = 18;   // token ids < 2^18

struct Args
{
    __nv_bfloat16 const* logits; // [rows, Vs] this rank's shard
    long long ld;
    int rows, Vs, vocab_offset;
    float const* temperatures;
    float const* top_ps;
    long long const* seeds;
    long long const* offsets;
    unsigned long long* ws_uc;
    unsigned long long* ws_mc;
    unsigned int* counters; // [0] finished rows, [1] call
    int* tokens;            // [rows] sampled token per row
    int* extra_attempts;    // optional [1]: rounds beyond the second, summed over rows
    int rank, world;
    int const* draft;
    int const* batch_idx;
    int* acc_tokens;
    int* num_acc;
    long long* gather_ids;
    int draft_len, nreq, force_total;
    int* next_new;
    int next_stride;
    int const* force_vec;
};

//! Words (uint64) of the multicast exchange workspace for `world` ranks: two call halves.
long long sampledAcceptWsWords(int world);

//! Launch the sampled verification over a.rows * (8 or 4) co-resident CTAs (8 per row up to 32 rows).
cudaError_t invokeSampledAccept(Args const& a, bool usePdl, cudaStream_t stream);

} // namespace sampledAccept
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
