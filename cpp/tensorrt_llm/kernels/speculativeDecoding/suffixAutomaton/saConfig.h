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

#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

#ifndef SA_MAX_SEQUENCE_LENGTH
#define SA_MAX_SEQUENCE_LENGTH 262144
#endif

#ifndef SA_MAX_SLOTS
#define SA_MAX_SLOTS 256
#endif

class SAConfig
{
public:
    static constexpr size_t MAX_SEQUENCE_LENGTH = SA_MAX_SEQUENCE_LENGTH;
    static constexpr size_t MAX_SLOTS = SA_MAX_SLOTS;

private:
    SAConfig() = default;
};

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
