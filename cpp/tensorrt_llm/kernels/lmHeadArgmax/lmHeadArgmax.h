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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace lm_head_argmax
{

//! \brief Greedy tokens of a vocabulary-parallel BF16 or FP16 LM head in one launch.
//!
//! For few rows (draft heads, small verify batches) the LM head is a weight-bandwidth-bound GEMV over this rank's
//! vocabulary shard. One kernel streams the shard's rows into shared memory, computes every 16-row tile's logits on
//! the tensor cores and folds them straight into per-row argmax keys (no logits are stored, no split-K reduction, no
//! separate argmax). The last CTA to finish reduces the CTAs' keys and, with several ranks, exchanges them with the
//! other ranks of the tensor-parallel group, so every rank writes the same global token.
//!
//! The exchange speaks the protocol of the Python ``ll_exchange.TPKeyExchange`` on one of its channels (same receive
//! buffers, same epoch counter): every 32-bit half of a rank's key travels as one 8-byte packet
//! ``(epoch << 32) | word`` stored into every rank's receive slot ``epoch & 1``; a reader polls its own slot until
//! every rank's packets carry the current epoch, then the channel's epoch advances. Launches of this kernel and of
//! the Triton exchange kernels may therefore share a channel in any order, as long as every rank issues the same
//! sequence. Like those kernels, this one never releases its dependents early, so the next launch on the channel
//! always reads the epoch this one wrote.
//!
//! Keys order like ``torch.argmax`` over the gathered vocabulary: the largest logit, and among equal logits the
//! lowest global index. -0.0 compares equal to +0.0. With ``roundLogits`` each fp32 dot product is first rounded to
//! the operand dtype, the dtype of the logits the GEMM path would store, so ties of the stored logits are ties here
//! too.
//!
//! Requires SM90+ (``cp.async.bulk``, ``mbarrier`` transaction counts, programmatic dependent launch) and enough
//! opt-in shared memory for the rows and the hidden size (see ``maxRows``); on older architectures the kernel body
//! is empty and ``maxRows`` returns 0.

//! Rows one call takes (the token columns of one m16n8k16 MMA).
constexpr int kMaxRows = 8;
//! Ranks of the tensor-parallel group (``TPKeyExchange`` takes 2, 4, 8 or 16).
constexpr int kMaxRanks = 16;
//! CTAs one call may use.
constexpr int kMaxCtas = 256;
//! Hidden-size granularity: a pipeline stage holds the widest of 1280, 1024, 768, 512 or 256 k elements that divides
//! the hidden size.
constexpr int kHiddenGranularity = 256;
//! uint64 words of a head's scratch: the CTAs' partial keys, then the CTA counter. Zero-initialized once; every call
//! leaves it ready for the next.
constexpr int kScratchWords = kMaxCtas * kMaxRows + 1;

struct Params
{
    void const* input{nullptr};  //!< bf16 / fp16 [m, k] rows (row stride inputStride elements)
    int64_t inputStride{0};
    void const* weight{nullptr}; //!< [n, k] this rank's vocabulary shard, input's dtype (row stride weightStride)
    int64_t weightStride{0};
    int m{0};
    int n{0};
    int k{0};
    int vocabOffset{0};         //!< global vocabulary index of the shard's first row
    bool fp16{false};           //!< fp16 operands (else bf16)
    bool roundLogits{false};    //!< compare the logits rounded to the operand dtype (see above)
    int32_t* tokens{nullptr};   //!< [m] output tokens
    uint64_t* scratch{nullptr}; //!< this rank's kScratchWords words
    // The exchange (ranks > 1): the channel's epoch counter and every rank's receive region of the channel, by rank;
    // a region holds two slots of ranks x exchangeWords 8-byte packets.
    int32_t* exchangeEpoch{nullptr};
    int64_t* peers[kMaxRanks]{};
    int exchangeWords{0}; //!< packets per rank and slot (2 per key; at least 2 * m)
    int rank{0};
    int ranks{1};
};

//! Rows (at most kMaxRows) one call takes for hidden size k on the current device; 0 when the device or k is not
//! supported.
int maxRows(int k);

//! Launch the fused head on stream. ctas: the grid size (typically the SM count).
void launch(Params const& params, int ctas, cudaStream_t stream, bool enablePdl);

} // namespace lm_head_argmax
} // namespace kernels

TRTLLM_NAMESPACE_END
