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

#include "rnnStateBuffers.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

RnnStateBuffers::RnnStateBuffers(SizeType32 maxBatchSize, runtime::TllmRuntime const& runtime)
{
    auto const& manager = runtime.getBufferManager();

    slotMappingHost = BufferManager::cpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
    slotMappingDevice = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
}

void RnnStateBuffers::reshape(SizeType32 numSequences)
{
    slotMappingHost->reshape(ITensor::makeShape({numSequences}));
    slotMappingDevice->reshape(ITensor::makeShape({numSequences}));
}

void RnnStateBuffers::fillSlotMappings(
    RequestVector const& contextRequests, rnn_state_manager::RnnStateManager* rnnStateManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(rnnStateBuffersFillSlotMappings);

    SizeType32 batchIdx{0};
    for (auto const& llmReq : contextRequests)
    {
        auto const seqSlot = llmReq->mSeqSlot.value();
        auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
        rnnStateManager->fillSlotMapping(*slotMappingHost, batchIdx, seqSlot, reqBeamWidth);
        ++batchIdx;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RnnStateBuffers::copySlotMappingH2D(runtime::TllmRuntime const& runtime)
{
    auto const& manager = runtime.getBufferManager();
    manager.copy(*slotMappingHost, *slotMappingDevice);
}

void RnnStateBuffers::getBuffers(TensorMap& inputBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(rnnStateBuffersGetBuffers);

    inputBuffers.insert_or_assign("slot_mapping", slotMappingDevice);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
