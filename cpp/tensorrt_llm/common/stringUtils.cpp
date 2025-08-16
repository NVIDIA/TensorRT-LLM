/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/assert.h"

#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <string>

namespace tensorrt_llm::common
{

void fmtstr_(char const* format, fmtstr_allocator alloc, void* target, va_list args)
{
    va_list args0;
    va_copy(args0, args);

    size_t constexpr init_size = 2048;
    char fixed_buffer[init_size];
    auto const size = std::vsnprintf(fixed_buffer, init_size, format, args0);
    va_end(args0);

    TLLM_CHECK_WITH_INFO(size >= 0, std::string(std::strerror(errno)));
    if (size == 0)
    {
        return;
    }

    auto* memory = alloc(target, size);

    if (static_cast<size_t>(size) < init_size)
    {
        std::memcpy(memory, fixed_buffer, size + 1);
    }
    else
    {
        auto const size2 = std::vsnprintf(memory, size + 1, format, args);
        TLLM_CHECK_WITH_INFO(size2 == size, std::string(std::strerror(errno)));
    }
}

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

} // namespace tensorrt_llm::common
