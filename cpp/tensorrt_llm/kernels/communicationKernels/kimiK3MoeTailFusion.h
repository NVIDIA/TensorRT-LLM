/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include <cuda_bf16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimi_k3_moe
{

//! Fused Kimi K3 latent-MoE tail head for TEP decode: one kernel does
//!   oneshot-lamport AllReduce of the routed latent partial [M, L]
//!     + RMSNorm epilogue (bf16 rounding of the fp32 sum, then fp32-variance
//!       norm — matching the baseline AR -> rmsnorm semantics)
//!   concurrently with a ReduceScatter of the shared-expert partial [M, H]:
//!     every rank pushes segment j of its shared row to segment-owner j; the
//!     owner accumulates its fully-reduced segment s_shard [M, H/nranks].
//! Replaces the baseline [cat -> combined twoshot AR over [M, H+L] -> split
//! -> contiguous copy -> rmsnorm] chain: both cross-node landings overlap
//! inside one kernel and the AR payload shrinks to the latent width.
//! One CTA per token (L/8 threads); lamport 3-phase rotation protocol
//! (allReduceFusionKernels workspace layout), consumer-side sentinel restore.
struct KimiK3MoeTailParams
{
    int nranks;
    int rank;
    int latent_dim;    // L (3584): AR + norm width
    int hidden_dim;    // H (7168) = stripe_dim * nranks: RS width
    int token_num;     // M in [1, 64]
    void** workspace;
    void* latent_in;   // [M, L] bf16 contiguous routed partial
    void* shared_in;   // [M, H] bf16 contiguous shared-expert partial
    void* norm_weight; // [L] bf16 (routed_expert_norm gamma)
    void* z_out;       // [M, L] bf16 reduced+normed latent (full, every rank)
    void* sshard_out;  // [M, H/nranks] bf16 this rank's reduced shared segment
    float rms_eps;
    cudaStream_t stream;
    bool trigger_completion_at_end = true;
};

void kimi_k3_moe_tail_op(KimiK3MoeTailParams const& params);

//! Fused [oneshot-lamport AllGather + elementwise add] for the column-striped
//! up-projection that follows the tail head: rank r computes only its output
//! columns out[:, r*S:(r+1)*S] = z @ W_up[rS:(r+1)S, :]^T from a bf16 weight
//! stripe; this kernel gathers the NRanks stripes into the full hidden row.
//! Optional operands: shard_add [M, S] is folded into the pushed stripe (fp32
//! + single rounding) — the RS'd shared segment — and shared [M, H] (strided
//! rows OK) is added on the poll side. Same workspace/rotation protocol as
//! the tail head; calls interleave safely on one stream.
struct KimiK3StripeAgParams
{
    int nranks;
    int rank;
    int size;        // token_num * stripe_dim (elements pushed per rank)
    int stripe_dim;  // hidden_dim / nranks (896 for K3 tep8)
    int hidden_dim;  // 7168
    int shared_ld;   // row stride of shared_in, in elements (16B-aligned)
    void** workspace;
    void* shard_in;  // [M, stripe] bf16 contiguous, this rank's stripe
    void* shard_add; // optional [M, stripe] bf16 added into the pushed stripe
    void* shared_in; // optional [M, hidden] bf16 added on the poll side
    void* out;       // [M, hidden] bf16 contiguous
    cudaStream_t stream;
    bool trigger_completion_at_end = true;
};

void kimi_k3_stripe_ag_add_op(KimiK3StripeAgParams const& params);

} // namespace kernels::kimi_k3_moe

TRTLLM_NAMESPACE_END
