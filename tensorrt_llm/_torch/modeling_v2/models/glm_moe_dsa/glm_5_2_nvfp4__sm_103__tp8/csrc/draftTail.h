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

// Greedy draft-token selection over a vocab-sharded LM head.
//
// For R hidden rows and this rank's vocab shard W [V_local, H] the kernels
// produce, on every rank, the token id of the global maximum of the BF16
// logits h @ W^T over all tensor-parallel shards; ties go to the lowest global
// token id and NaN ranks above every number. That is the selection rule of
// SpecWorkerBase.greedy_sample_draft_with_tp_gather.
//
// The logits kernel selects from BF16 logits computed by the caller (the
// native LM-head GEMM), so its tokens are the native ones. The fused kernels
// compute the logits themselves (FP32 accumulation, then BF16 rounding);
// where the largest logits are within one BF16 rounding step of each other,
// the selected token depends on the exact FP32 sums, i.e. on the accumulation
// order. Fused variants 11 to 14 accumulate like the native cuBLAS LM-head
// GEMM selected for this shape on SM103 (K in 16-value tensor-core steps from
// 0 to H into one FP32 accumulator), so their BF16 logits and tokens equal the
// native ones bit for bit (checked by the draft_tail unit tests against F.linear;
// re-check whenever cuBLAS or the GPU changes). The other fused variants
// accumulate in other orders and are decision-level variants at BF16
// near-ties.
//
// Each candidate is packed into one 64-bit key, (ordered BF16 bits << 32) |
// (0xFFFFFFFF - global id), so the selection is a plain unsigned max that is
// associative, commutative and independent of reduction order. The keys of
// all ranks are exchanged through a multicast (or unicast peer) buffer with a
// 0-sentinel protocol over three rotating slots; see draftTail.cu.

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace draft_tail
{

constexpr int kMaxRows = 8;
constexpr int kMaxRanks = 8;
constexpr int kNumSlots = 3;
// Exchange buffer: kNumSlots x kMaxRanks x kMaxRows 64-bit keys.
constexpr int kSlotWords = kMaxRanks * kMaxRows;
constexpr int kExchangeWords = kNumSlots * kSlotWords;
// Rank-local scratch, in 64-bit words: kMaxRows partial keys, then the CTA
// arrival counter and the exchange epoch (32 bits each in one word), then an
// error word.
constexpr int kScratchWords = kMaxRows + 2;

enum class ExchangeMode : int
{
    kNone = 0,      // tpSize == 1, or debug: tokens from the rank-local maximum
    kMulticast = 1, // one multimem.st per row into the multicast slot
    kUnicast = 2,   // one store per (peer, row) into each peer's slot
    // Direct variants: every CTA publishes its own partial key (no rank-local
    // atomics or last-CTA election); CTA 0 of each rank polls all
    // tpSize x numCtas entries, writes the tokens and recycles the slot.
    kMulticastDirect = 3,
    kUnicastDirect = 4,
};

constexpr int kMaxDirectCtas = 512;
constexpr int kDirectSlotWords = kMaxRows * kMaxRanks * kMaxDirectCtas;
// The direct region follows the last-CTA region in the same buffer.
constexpr int kDirectOffsetWords = kExchangeWords;
constexpr int kExchangeBufferWords = kExchangeWords + kNumSlots * kDirectSlotWords;

struct DraftTailParams
{
    // Fused mode: hidden [rows, hiddenDim] (row stride hiddenStride elements)
    // and the vocab shard weight [vocabLocal, hiddenDim], row-major.
    __nv_bfloat16 const* hidden = nullptr;
    int64_t hiddenStride = 0;
    __nv_bfloat16 const* weight = nullptr;
    // Logits mode: BF16 logits [rows, vocabLocal] (row stride logitsStride).
    __nv_bfloat16 const* logits = nullptr;
    int64_t logitsStride = 0;

    int rows = 0;
    int vocabLocal = 0;
    int hiddenDim = 0;
    int64_t vocabOffset = 0;     // global id of local column 0

    int32_t* tokens = nullptr;   // [rows] output
    uint64_t* keysOut = nullptr; // optional [rows] output: the winning key

    uint64_t* scratch = nullptr; // [kScratchWords], zero-initialized once

    int tpRank = 0;
    int tpSize = 1;
    ExchangeMode mode = ExchangeMode::kNone;
    uint64_t* mcSlots = nullptr;             // multicast address of the exchange buffer
    uint64_t* ucSlotsLocal = nullptr;        // this rank's unicast address of the buffer
    uint64_t* const* ucSlotsPeers = nullptr; // device array [tpSize] of unicast addresses
    long long timeoutCycles = 0;             // 0: wait forever
    // Optional BF16 logits [rows, vocabLocal] (row stride logitsOutStride),
    // written by the tensor-core fused kernels only (tests compare them with
    // the native GEMM bit for bit); nullptr in production.
    __nv_bfloat16* logitsOut = nullptr;
    int64_t logitsOutStride = 0;
    // Fused kernel, L2-hinted variants: each warp loads the first
    // keepPermille / 1000 of its rows with an L2 evict_last policy (the rest
    // evict_first), so that part of the shard tends to stay in L2 across the
    // draft steps that re-read the same weight.
    int keepPermille = 0;
};

struct FusedLaunchConfig
{
    int numCtas = 0; // 0: one per SM
    int variant = 0; // see draftTail.cu: kernel shape and load flavour
    bool pdl = true;
};

int numFusedVariants();

// GEMV + rank-local max + exchange in one launch.
void invokeDraftTailFused(DraftTailParams const& params, FusedLaunchConfig const& config, cudaStream_t stream);

// Rank-local max over precomputed BF16 logits + exchange in one launch.
void invokeDraftTailFromLogits(DraftTailParams const& params, bool pdl, cudaStream_t stream);

} // namespace draft_tail

TRTLLM_MV2_NAMESPACE_END
