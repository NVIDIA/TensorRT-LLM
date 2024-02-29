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

namespace tensorrt_llm
{
namespace kernels
{

// max number of CTAs for each KV head, multiple CTAs for one KV head is multi-block mode.
// this number defines the maximum number when reaches both max_batch_size and max_beam_width.
// If batch_size or beam_width doesn't reach maximum value, it is possible to have more CTAs per KV head than this
// value.
static constexpr int kMaxNbCtaPerKVHeadFactor = 8;
static constexpr int kMinHistoryTokensPerBlock = 512;

static constexpr float kEnableMinBlockFactor = 4.0;
static constexpr int kTargetWaveFactor = 8;

} // namespace kernels
} // namespace tensorrt_llm
