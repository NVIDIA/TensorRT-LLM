/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeIndexerKCacheScatter(uint8_t const* k_fp8_bytes, uint8_t const* k_scale_bytes, uint8_t* k_cache,
    int64_t const* slot_mapping_fp8, int64_t const* slot_mapping_scale, int32_t num_tokens, int32_t head_dim,
    int32_t scale_size, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3,
    int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2, int64_t cache_stride_3,
    cudaStream_t stream = 0);

}

TRTLLM_NAMESPACE_END
