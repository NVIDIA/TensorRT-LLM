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

#include "tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::kimiK3MnnvlOneShot
{

struct AllReduceFusionParams : ::tensorrt_llm::kernels::mnnvl::AllReduceFusionParams
{
    bool earlyTrigger = false;
};

void oneshotAllreduceFusionOp(AllReduceFusionParams const& params);

}

TRTLLM_MV2_NAMESPACE_END
