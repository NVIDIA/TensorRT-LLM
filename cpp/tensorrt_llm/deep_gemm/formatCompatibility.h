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

#include <array>
#include <charconv>
#include <cmath>
#include <string>
#include <system_error>

namespace tensorrt_llm::deepGemm
{

// Format a finite float as a CUDA literal, matching std::format's hexadecimal
// representation, including signed zero and subnormal values.
inline std::string formatFloatLiteral(float const value)
{
    constexpr int kBufferSize = 32;
    std::array<char, kBufferSize> buffer{};
    // Every finite float is represented exactly as a normal double (or zero),
    // giving subnormal floats the same normalized exponent across libraries.
    auto const result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), static_cast<double>(std::abs(value)), std::chars_format::hex);
    if (result.ec != std::errc{})
    {
        throw std::system_error(std::make_error_code(result.ec));
    }
    return std::string(std::signbit(value) ? "-0x" : "0x") + std::string(buffer.data(), result.ptr) + "f";
}

} // namespace tensorrt_llm::deepGemm
