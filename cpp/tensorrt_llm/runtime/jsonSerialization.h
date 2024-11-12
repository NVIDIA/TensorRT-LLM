/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/jsonSerializeOptional.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/runtimeDefaults.h"
#include <nlohmann/json.hpp>

namespace tensorrt_llm::runtime
{

// RuntimeDefaults
void to_json(nlohmann::json& json, RuntimeDefaults const& runtimeDefaults)
{
    json = nlohmann::json{{"max_attention_window", runtimeDefaults.maxAttentionWindowVec},
        {"sink_token_length", runtimeDefaults.sinkTokenLength}};
}

void from_json(nlohmann::json const& json, RuntimeDefaults& runtimeDefaults)
{
    runtimeDefaults.maxAttentionWindowVec = json.value("max_attention_window", runtimeDefaults.maxAttentionWindowVec);
    runtimeDefaults.sinkTokenLength = json.value("sink_token_length", runtimeDefaults.sinkTokenLength);
}

// End RuntimeDefaults

} // namespace tensorrt_llm::runtime
