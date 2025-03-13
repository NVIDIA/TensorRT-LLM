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

#include "promptTuningBuffers.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"

namespace tensorrt_llm::batch_manager
{

PromptTuningBuffers::PromptTuningBuffers(SizeType32 maxBatchSize, runtime::BufferManager const& manager,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    auto maxPromptEmbeddingTableSize = modelConfig.getMaxPromptEmbeddingTableSize();
    auto const hiddenSize = modelConfig.getHiddenSize() * worldConfig.getTensorParallelism();

    // vocabSize and mMaxPromptVocabSize
    mPromptTuningParams.vocabSize = manager.gpu(runtime::ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    mMaxPromptVocabSize = maxPromptEmbeddingTableSize / maxBatchSize;

    auto promptVocabSizeHost
        = runtime::BufferManager::pinned(runtime::ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    auto promptVocabSizeHostData = runtime::bufferCast<SizeType32>(*promptVocabSizeHost);
    promptVocabSizeHostData[0] = mMaxPromptVocabSize;
    manager.copy(*promptVocabSizeHost, *mPromptTuningParams.vocabSize);

    // embeddingTable
    mPromptTuningParams.embeddingTable = manager.gpu(
        runtime::ITensor::makeShape({maxPromptEmbeddingTableSize, hiddenSize}), modelConfig.getDataType());

    // tasks
    mPromptTuningParams.tasks = manager.emptyTensor(runtime::MemoryType::kGPU, nvinfer1::DataType::kINT32);
}

void PromptTuningBuffers::validate(
    std::optional<TensorPtr> const& optReqPromptEmbeddingTable, std::optional<SizeType32> const& optReqPromptVocabSize)
{
    // Need to copy request embeddingTable to promptEmbeddingTable
    if (optReqPromptEmbeddingTable.has_value())
    {

        auto reqPromptEmbeddingTable = optReqPromptEmbeddingTable.value();
        auto reqPromptVocabSize = optReqPromptVocabSize.value();

        if (reqPromptVocabSize > mMaxPromptVocabSize)
        {
            std::string errStr = "Prompt vocab size" + std::to_string(reqPromptVocabSize)
                + " is larger than max prompt vocab size of " + std::to_string(mMaxPromptVocabSize)
                + ". Max prompt vocab size is computed from max_prompt_embedding_table_size / max_batch_size. ";
            TLLM_LOG_ERROR(errStr);
            throw std::runtime_error(errStr);
        }
        else
        {
            // Check that type matches model weights
            if (reqPromptEmbeddingTable->getDataType() != mPromptTuningParams.embeddingTable->getDataType())
            {
                std::string errStr = "Request embedding table data type doesn't match model weight data type.";
                TLLM_LOG_ERROR(errStr);
                throw std::runtime_error(errStr);
            }

            if (reqPromptEmbeddingTable->getShape().d[1] != reqPromptVocabSize)
            {
                std::string errStr
                    = "First dimension of request embedding table is expected to be equal to prompt vocab size";
                TLLM_LOG_ERROR(errStr);
                throw std::runtime_error(errStr);
            }
        }
    }
}

void PromptTuningBuffers::fill(RequestVector const& contextRequests, RequestVector const& genRequests,
    runtime::BufferManager const& manager, bool packed)
{
    manager.setZero(*mPromptTuningParams.embeddingTable);

    auto const numContextRequests = static_cast<SizeType32>(contextRequests.size());

    std::vector<SizeType32> reqBeamWidths;
    std::vector<SizeType32> reqPromptLengths;
    mPromptTuningParams.promptTuningEnabled.clear();

    SizeType32 batchIdx{0};
    for (auto const& requests : {contextRequests, genRequests})
    {
        for (auto const& llmReq : requests)
        {
            reqBeamWidths.push_back(llmReq->mSamplingConfig.beamWidth);
            if (batchIdx < numContextRequests)
            {
                SizeType32 numContextTokens = 0;
                auto const draftLength = llmReq->isLastContextChunk() ? llmReq->getNumDraftTokens() : 0;
                auto const contextChunkSize = llmReq->getContextChunkSize();
                numContextTokens += contextChunkSize + draftLength;
                reqPromptLengths.push_back(numContextTokens);
            }
            auto optReqPromptEmbeddingTable = llmReq->getPromptEmbeddingTable();
            auto const optReqPromptVocabSize = llmReq->getPromptVocabSize();
            mPromptTuningParams.promptTuningEnabled.push_back(optReqPromptEmbeddingTable.has_value());

            // If context request & has embedding table, validate it
            if (optReqPromptEmbeddingTable.has_value())
            {
                // If a context request, validate prompt tensors and move to GPU
                if (batchIdx < numContextRequests)
                {
                    // Move to GPU
                    llmReq->movePromptEmbeddingTableToGpu(manager);
                    optReqPromptEmbeddingTable = llmReq->getPromptEmbeddingTable();

                    // Validate the table, prompt_vocab_size
                    validate(optReqPromptEmbeddingTable, optReqPromptVocabSize);
                }

                auto const reqPromptEmbeddingTable = optReqPromptEmbeddingTable.value();
                auto const reqPromptVocabSize = optReqPromptVocabSize.value();

                // TODO: Use invokeCopyBatch to avoid multiple bs1 copies
                // Copy into large prompt embedding table
                TensorPtr reqPromptEmbeddingTableView = runtime::ITensor::view(reqPromptEmbeddingTable);
                reqPromptEmbeddingTableView->squeeze(0);
                auto const promptEmbeddingTableSlice = runtime::ITensor::slice(
                    mPromptTuningParams.embeddingTable, batchIdx * mMaxPromptVocabSize, reqPromptVocabSize);
                manager.copy(*reqPromptEmbeddingTable, *promptEmbeddingTableSlice);
                // TODO:       src: 2007040 (llmReq->getPromptEmbeddingTable()) != dst: 1003520 (reqPromptVocabSize)
                //                                                                      (original shape passed from
                //                                                                      python == 196 * 5120, fp16)
                // VILA mode 1 , 2 images in one request
            }
            ++batchIdx;
        }
    }

    auto const batchSize = batchIdx;
    TensorPtr tasksHost = runtime::BufferManager::pinned(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    auto* tasksHostPtr = runtime::bufferCast<SizeType32>(*tasksHost);
    std::iota(tasksHostPtr, tasksHostPtr + batchSize, 0);
    mPromptTuningParams.fillTasksTensor(
        tasksHost, batchSize, numContextRequests, reqBeamWidths, reqPromptLengths, manager, packed);
}

} // namespace tensorrt_llm::batch_manager
