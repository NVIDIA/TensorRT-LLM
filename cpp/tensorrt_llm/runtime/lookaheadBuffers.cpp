/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 */

#include "tensorrt_llm/runtime/lookaheadBuffers.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/tllmDataType.h"

namespace tensorrt_llm::runtime
{

LookaheadDecodingBuffers::LookaheadDecodingBuffers(
    SizeType32 maxNumSequences, SizeType32 maxTokensPerStep, BufferManager const& bufferManager)
    : generationLengths(bufferManager.gpu(ITensor::makeShape({maxNumSequences}), tensorrt_llm::DataType::kINT32))
    , positionOffsets(
          bufferManager.gpu(ITensor::makeShape({maxNumSequences, maxTokensPerStep}), tensorrt_llm::DataType::kINT32))
    , packedMasks(bufferManager.gpu(ITensor::makeShape({maxNumSequences, maxTokensPerStep,
                                        static_cast<ITensor::DimType64>(common::divUp(maxTokensPerStep, 32))}),
          tensorrt_llm::DataType::kINT32))
    , positionIds(
          bufferManager.gpu(ITensor::makeShape({maxNumSequences, maxTokensPerStep}), tensorrt_llm::DataType::kINT32))
{
}

} // namespace tensorrt_llm::runtime
