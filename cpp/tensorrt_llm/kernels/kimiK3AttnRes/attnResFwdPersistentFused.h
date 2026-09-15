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

#include "attnResFwd.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimiK3AttnRes
{

//! Persistent-grid counterpart to invokeAttnResAddRmsNormFwd.
//!
//! Same contract and same AttnResFwdParams, different topology: one CTA per SM
//! looping over tokens, instead of one CTA (or one cluster) per token. That
//! makes it the shape to reach for when there are far more tokens than SMs --
//! prefill -- where the per-token kernels stop being able to amortize the score
//! projection load.
//!
//! Requires H == 7168 and N in [2, 9]. Unlike the per-token kernels there is no
//! N == 1 case: the persistent kernel always has at least one snapshot plus the
//! layer residual.
//!
//! layerResidualAdd and outputRmsWeight are both required -- this entry point
//! exists for the fully fused form. updatedLayerResidual must alias
//! layerResidual: the kernel folds the residual add in place.
void invokeAttnResPersistentFusedFwd(AttnResFwdParams const& params, cudaStream_t stream);

//! True when invokeAttnResPersistentFusedFwd can serve this shape, so callers
//! can route without duplicating the constraints above.
bool attnResPersistentFusedSupported(AttnResFwdParams const& params);

} // namespace kernels::kimiK3AttnRes

TRTLLM_NAMESPACE_END
