/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/xqaParams.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

template <typename KVCacheBuffer>
CUtensorMap makeTensorMapForHopperXqaKVCache(std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> const& driver,
    XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer);

template <typename KVCacheBuffer>
CUtensorMap makeTensorMapForXqaMlaKVCache(std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> const& driver,
    XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer, bool forK);

CUtensorMap makeTensorMapForXqaMlaQ(
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> const& driver, XQAParams const& xqaParams, void const* q);

// 2D per-token gather descriptor for DSV4 dynamic-sparse MLA dual-pool KV on SM120. `poolBase` is the
// SWA pool or the compressed pool base; the descriptor addresses the pool as [token_slot, headElems]
// (nbKHeads==1 MLA latent) so the kernel gathers any token by its absolute slot (a sparse_attn_indices
// entry) via tma.gather4. Swizzle matches the dense MLA KV descriptor so gathered rows land in the
// layout the MMA consumer reads.
CUtensorMap makeTensorMapForXqaMlaKVCacheGather(std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> const& driver,
    XQAParams const& xqaParams, void const* poolBase, bool forK);

} // namespace kernels

TRTLLM_NAMESPACE_END
