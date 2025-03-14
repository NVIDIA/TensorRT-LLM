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

#include "tensorrt_llm/batch_manager/generateRequestOptions.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/peftCacheManager.h"
#include "tensorrt_llm/batch_manager/sequenceSlotManager.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"

namespace tensorrt_llm::batch_manager::utils
{

void draftModelSendLogitsThread(int device, std::atomic<bool>* draftModelThreadShouldExit,
    RequestVector* draftRequestsWaitingToSendLogits, std::shared_ptr<SequenceSlotManager> seqSlotManager,
    SizeType32 maxInputLen, std::shared_ptr<kv_cache_manager::BaseKVCacheManager> kvCacheManager,
    std::shared_ptr<kv_cache_manager::BaseKVCacheManager> crossKvCacheManager,
    std::shared_ptr<BasePeftCacheManager> peftCacheManager);

std::optional<GenerateRequestOptions::TensorPtr> targetModelReceiveLogits(
    executor::SpeculativeDecodingFastLogitsInfo const& fastLogitsInfo, runtime::ModelConfig const& modelConfig);

} // namespace tensorrt_llm::batch_manager::utils
