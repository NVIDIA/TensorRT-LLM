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
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace lm_head_argmax
{

//! \brief Greedy tokens of a vocabulary-parallel LM head in one launch.
//!
//! For few rows (draft heads, small verify batches) the LM head is a weight-bandwidth-bound GEMV over this rank's
//! vocabulary shard. One kernel streams the shard's rows into shared memory, computes every 16-row tile's logits on
//! the tensor cores and folds them straight into per-row argmax keys (no logits are stored, no split-K reduction, no
//! separate argmax). The last CTA to finish reduces the CTAs' keys and, with several ranks, exchanges them with the
//! other ranks of the tensor-parallel group through peer-mapped buffers, so every rank writes the same global token.
//!
//! A key is ``(order-preserving bits of the fp32 logit) << 32 | (0xFFFFFFFF - global index)``: the largest key is
//! the largest logit, and among equal logits the lowest index (``torch.argmax`` over the gathered vocabulary).

//! Rows one call takes (the token columns of one m16n8k16 MMA).
constexpr int kMaxRows = 8;
//! Ranks of the tensor-parallel group.
constexpr int kMaxRanks = 8;
//! CTAs one call may use.
constexpr int kMaxCtas = 256;
//! uint64 slots of one rank's exchange buffer: three rotating epochs of kMaxRanks x kMaxRows keys.
constexpr int kExchangeSlots = 3 * kMaxRanks * kMaxRows;
//! uint64 words of a rank's scratch: the CTAs' partial keys, then the CTA counter and the exchange state
//! (epoch, rows of the previous call). Zero-initialized once; every call leaves it ready for the next.
constexpr int kScratchWords = kMaxCtas * kMaxRows + 2;

struct Params
{
    void const* input{nullptr};  //!< bf16 [m, k] rows (row stride inputStride elements)
    int64_t inputStride{0};
    void const* weight{nullptr}; //!< bf16 [n, k] this rank's vocabulary shard (row stride weightStride elements)
    int64_t weightStride{0};
    int m{0};
    int n{0};
    int k{0};
    int vocabOffset{0};           //!< global vocabulary index of the shard's first row
    int32_t* tokens{nullptr};     //!< [m] output tokens, at tokens[row * tokensStride]
    int tokensStride{1};
    uint64_t* scratch{nullptr};   //!< this rank's kScratchWords words
    uint64_t* peers[kMaxRanks]{}; //!< every rank's exchange buffer (kExchangeSlots words each), by rank
    int rank{0};
    int ranks{1};
};

//! Whether a call of m rows, an n-row shard and hidden size k is supported (on an SM90+ device).
bool supports(int m, int n, int k);

//! Launch the fused head on stream. ctas: the grid size (typically the SM count).
void launch(Params const& params, int ctas, cudaStream_t stream, bool enablePdl);

} // namespace lm_head_argmax
} // namespace kernels

TRTLLM_NAMESPACE_END
