/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3Mnnvl
{

/**
 * \brief Kimi K3 attention-residual epilogue of oneshotAllreduceAttnResOp (BF16 tensors).
 */
struct AttnResEpilogueParams
{
    void const* blockResidual;   //!< [numCandidates - 1, numTokens, tokenDim] snapshots; unused when numCandidates == 1
    void const* resWeight;       //!< [tokenDim] attention-residual projection
    void const* rmsWeight;       //!< [tokenDim] weight of the RMSNorm inside the attention-residual score
    void const* outputRmsWeight; //!< [tokenDim] weight of the RMSNorm applied to the selected residual
    float rmsEps;
    float outputRmsEps;
    int numCandidates; //!< Snapshots plus the running prefix sum, 1..12
};

/**
 * \brief One-shot all-reduce with Kimi K3's residual add, attention-residual selection and RMSNorm as the epilogue.
 *
 * Per token, with r = bf16(allreduce(input)):
 *   residualOut = bf16(residualIn + r), or r when residualIn is null
 *   output      = RMSNorm(attn_res(blockResidual[0 .. numCandidates-2], residualOut), outputRmsWeight)
 * with the rounding of the unfused sequence all-reduce -> attn_res_add_rmsnorm_fwd. The reduction order is fixed, so
 * the result is deterministic and identical on every rank.
 *
 * Requirements: BF16; tokenDim a multiple of 1024 and at most 8192; the one-shot footprint
 * numTokens * tokenDim * nRanks elements fits in one Lamport buffer; nRanks in {2, 4, 8, 16}.
 */
void oneshotAllreduceAttnResOp(mnnvl::AllReduceFusionParams const& params, AttnResEpilogueParams const& epilogue);

} // namespace kernels::kimiK3Mnnvl

TRTLLM_NAMESPACE_END
