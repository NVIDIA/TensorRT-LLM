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

#include "tensorrt_llm/kernels/kimiK3AttnRes/attnResFwd.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::kimiK3DecodeAttnRes
{

struct AttnResFwdParams : ::tensorrt_llm::kernels::kimiK3AttnRes::AttnResFwdParams
{
    bool earlyTrigger = false;
};

void invokeAttnResRmsNormFwd(AttnResFwdParams const& params, cudaStream_t stream);
void invokeAttnResAddRmsNormFwd(AttnResFwdParams const& params, cudaStream_t stream);

}

TRTLLM_MV2_NAMESPACE_END
