/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted from Baseten's sa_spec library (Apache-2.0)
 * https://github.com/basetenlabs/sa_spec
 */

#pragma once

#include <cstdint>

#include "saNamedType.h"

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

using Token = NamedType<int32_t, struct TokenTag>;
using BatchIndex = NamedType<int, struct BatchIndexTag>;
using RequestID = NamedType<uint64_t, struct RequestIDTag>;
using NumTokens = NamedType<int, struct NumTokensTag>;
using NodeIndex = NamedType<int, struct NodeIndexTag>;
using TextIndex = NamedType<int, struct TextIndexTag>;

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
