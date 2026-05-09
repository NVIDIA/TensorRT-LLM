/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::runtime::decoder
{
class DecoderState;
}

namespace tensorrt_llm::batch_manager
{

/// @brief Reconstruct coherent per-beam token histories by tracing parentIds.
///
/// During beam search, LlmRequest::getTokens() returns slot-accumulated histories
/// that are NOT coherent after beam reassignment.  This function traces
/// DecoderState::parentIds backwards for each beam to recover the true ancestral
/// token path, and stores the result in
/// inputBuffers.gatheredBeamTokensForCallback[i] for each request that has a
/// LogitsPostProcessor (per-request or batched).
///
/// Requests with beam_width <= 1 or without any callback are skipped (the
/// LogitsPostProcessor falls back to llmReq->getTokens() in that case).
///
/// @param inputBuffers   DecoderInputBuffers whose decoderRequests are populated.
/// @param decoderState   DecoderState containing parentIds on GPU.
/// @param bufferManager  A BufferManager for the D2H copy of parentIds.
void buildGatheredBeamTokensForCallback(
    DecoderInputBuffers& inputBuffers,
    runtime::decoder::DecoderState const& decoderState,
    runtime::BufferManager const& bufferManager);

} // namespace tensorrt_llm::batch_manager
