/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/kernels/kdaDecode/kdaDecode.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kdaDecode
{

//! Launches the legacy compact-heads KDA decode kernel.
void launchKdaDecodeLegacyCompactHeads(KdaDecodeParams const& params, cudaStream_t stream);

//! Launches the legacy many-heads KDA decode kernel.
void launchKdaDecodeLegacyManyHeads(KdaDecodeParams const& params, cudaStream_t stream);

//! Launches the Blackwell single-CTA cp.async KDA decode kernel.
void launchKdaDecodeBlackwellSingleCta(KdaDecodeParams const& params, cudaStream_t stream);

//! Launches the Blackwell single-CTA two-stage cp.async.bulk KDA decode kernel.
void launchKdaDecodeBlackwellTwoStageBulk(KdaDecodeParams const& params, cudaStream_t stream);

//! Launches the Blackwell single-CTA four-stage cp.async.bulk KDA decode kernel.
void launchKdaDecodeBlackwellFourStageBulk(KdaDecodeParams const& params, cudaStream_t stream);

//! Launches the Blackwell four-CTA cluster cp.async KDA decode kernel.
void launchKdaDecodeBlackwellFourCtaCluster(KdaDecodeParams const& params, cudaStream_t stream);

} // namespace kernels::kdaDecode

TRTLLM_NAMESPACE_END
