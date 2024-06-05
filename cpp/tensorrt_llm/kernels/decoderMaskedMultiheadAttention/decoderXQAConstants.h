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
 *
 * This file contains constants that decoderXQA*.{h,cpp} need.
 */
#pragma once
#include <cstdint>
#include <optional>

namespace tensorrt_llm
{
namespace kernels
{
inline constexpr int kMinHistoryTokensPerBlock = 128;

inline constexpr float kEnableMinBlockFactor = 4.0;
inline constexpr int kTargetWaveFactor = 8;

} // namespace kernels
} // namespace tensorrt_llm
