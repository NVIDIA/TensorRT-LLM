/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"

namespace tensorrt_llm::testing
{

/// @brief Test utilities for KV cache manager unit tests. NEVER use in production code.
class KvCacheManagerTestUtil
{
public:
    /// @brief Simulate completion of the prefill stage on an LlmRequest.
    ///
    /// NEVER CALL FROM PRODUCTION CODE. This is solely for use in unit tests.
    ///
    /// Most BlockManager/KVCacheManager functions (storeContextBlocks, releaseBlocks,
    /// removeSequence) require prefill to be complete before they are called. This
    /// method updates llmRequest state as if prefill has just finished, allowing unit
    /// tests to invoke those functions correctly.
    static void simulatePrefillCompletion(batch_manager::LlmRequest& llmRequest)
    {
        llmRequest.setContextCurrentPosition(llmRequest.getPromptLen());
    }

    /// @brief Return mutable BlockManager access for tests that need to force internal block state.
    ///
    /// NEVER CALL FROM PRODUCTION CODE. This keeps const-cast based test setup in one explicit test-only helper.
    static batch_manager::kv_cache_manager::BlockManager& mutableBlockManager(
        batch_manager::kv_cache_manager::BaseKVCacheManager& kvCacheManager)
    {
        return const_cast<batch_manager::kv_cache_manager::BlockManager&>(kvCacheManager.getBlockManager());
    }
};

} // namespace tensorrt_llm::testing
