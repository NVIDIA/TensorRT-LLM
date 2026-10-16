/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/common/config.h"

#include <iostream>
#include <string>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

std::unordered_set<std::string> str2set(std::string const& input, char delimiter)
{
    std::unordered_set<std::string> values;
    if (!input.empty())
    {
        std::stringstream valStream(input);
        std::string val;
        while (std::getline(valStream, val, delimiter))
        {
            if (!val.empty())
            {
                values.insert(val);
            }
        }
    }
    return values;
};

} // namespace common

TRTLLM_NAMESPACE_END
