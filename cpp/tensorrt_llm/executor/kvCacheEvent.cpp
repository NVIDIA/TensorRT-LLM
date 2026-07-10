/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Definition of the (backend-agnostic) KVCacheEvent constructor. It previously
// lived in executor.cpp alongside the legacy TensorRT-engine Executor class
// implementation; that file was removed with the TensorRT backend, so the
// constructor is relocated here so the retained KV-cache event path
// (kvCacheEventManager, serialization, nanobind) continues to link.

#include <tensorrt_llm/executor/executor.h>
#include <tensorrt_llm/executor/version.h>

namespace tensorrt_llm::executor
{

char const* version() noexcept
{
    return kTensorRtLlmVersion;
}

KVCacheEvent::KVCacheEvent(
    size_t eventId, KVCacheEventData data, SizeType32 windowSize, std::optional<SizeType32> attentionDpRank)
    : eventId{eventId}
    , data{std::move(data)}
    , windowSize{windowSize}
    , attentionDpRank{attentionDpRank}
{
}

} // namespace tensorrt_llm::executor
