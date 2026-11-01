/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/minimaxM3Sm107/sampledAccept.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{
//! Fused sampled (temperature + top-p) verification of the MiniMax-M3 Eagle3 target head over the TP-sharded
//! vocabulary, decode rows <= kMaxRows; see sampledAccept.cu.
namespace sampledAccept
{

constexpr int kMaxSplits = 8;  // CTAs per verify row (at most)
constexpr int kMaxRows = 48;   // verify rows per call
constexpr int kMaxWorld = 8;   // TP ranks
constexpr int kMaxVs = 100352; // vocab shard width (TP2: 200064 / 2 = 100032, padded to a multiple of 512)
constexpr int kIdBits = 18;    // token ids < 2^18

using Args = minimaxM3Sm107::sampledAccept::Args;

//! Words (uint64) of the multicast exchange workspace for `world` ranks and SA-1R variant `oneRoundK`: two call halves.
long long sampledAcceptWsWords(int world, int oneRoundK = 0);

//! Launch the sampled verification over a.rows * (8 or 4) co-resident CTAs (8 per row up to 32 rows).
cudaError_t invokeSampledAccept(Args const& a, bool usePdl, cudaStream_t stream);

} // namespace sampledAccept
} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_NAMESPACE_END
