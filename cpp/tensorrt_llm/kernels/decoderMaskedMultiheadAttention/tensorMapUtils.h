/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/xqaParams.h"

namespace tensorrt_llm
{
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

} // namespace kernels
} // namespace tensorrt_llm
