/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "tensorrt_llm/common/config.h"
#include "unfusedAttentionKernels_2_template.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

#ifdef ENABLE_BF16
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(__nv_bfloat16, __nv_fp8_e4m3, KVBlockArray);
INSTANTIATE_ATTENTION_INPUT_OUTPUT_PROCESSING(__nv_bfloat16, __nv_fp8_e4m3, KVLinearBuffer);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END

// TLLM_GPTOSS_ROPEKV_TOKEN_MAJOR, for in-process A/B tests (ctypes): mode < 0 follows the environment variable (the
// default), 0 keeps applyBiasRopeUpdateKVCacheV2 for every launch, 1..3 runs that variant of the token-major kernel on
// the context-phase launches it covers (larger values: off). The BF16 x FP8 paged-KV instantiation in this file is the
// only one that reads it.
extern "C" void tllm_gptoss_ropekv_token_major_override(int mode)
{
    tensorrt_llm::kernels::ropeKvTokenMajorOverride().store(mode < 0 ? -1 : mode);
}

// TLLM_GPTOSS_ROPEKV_GEN, for in-process A/B tests (ctypes): mode < 0 follows the environment variable (the default),
// 0 keeps applyBiasRopeUpdateKVCacheV2 for every generation-phase launch, 1..3 runs that variant of the split-token
// kernel on the generation-phase launches it covers (larger values: off). Read only by the BF16 x FP8 paged-KV
// instantiation in this file.
extern "C" void tllm_gptoss_ropekv_gen_override(int mode)
{
    tensorrt_llm::kernels::ropeKvGenOverride().store(mode < 0 ? -1 : mode);
}
