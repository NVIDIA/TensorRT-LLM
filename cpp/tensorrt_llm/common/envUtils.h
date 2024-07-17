/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#pragma once
#include <cstdint>
#include <optional>

namespace tensorrt_llm::common
{

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels();

// max number of CTAs for each KV head, multiple CTAs for one KV head is multi-block mode.
// this number defines the maximum number when reaches both max_batch_size and max_beam_width.
// If batch_size or beam_width doesn't reach maximum value, it is possible to have more CTAs per KV head than this
// value.
int32_t xqaMaxNbCtaPerKVHeadFactor();

std::optional<int32_t> envXqaNbCtaPerKVHead();

// Whether XQA JIT is enabled.
//
// Returns the value of TRTLLM_ENABLE_XQA_JIT env var. If such env var doesn't exist, std::nullopt is returned.
std::optional<bool> getEnvEnableXQAJIT();

// Tune the number of blocks per sequence for accuracy/performance purpose.
bool getEnvMmhaMultiblockDebug();

int getEnvMmhaBlocksPerSequence();

int getEnvMmhaKernelBlockSize();

} // namespace tensorrt_llm::common
