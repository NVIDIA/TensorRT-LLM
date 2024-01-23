/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/loraModule.h"
#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{

class GptModelConfig
{
public:
    enum class ModelVariant : std::int32_t
    {
        kGpt = 0,
        kGlm = 1, // https://github.com/THUDM/GLM and https://github.com/THUDM/ChatGLM-6B
    };

    explicit GptModelConfig(
        SizeType vocabSize, SizeType nbLayers, SizeType nbHeads, SizeType hiddenSize, nvinfer1::DataType dtype)
        : mVocabSize(vocabSize)
        , mNbLayers(nbLayers)
        , mNbHeads(nbHeads)
        , mNbKvHeads(nbHeads)
        , mHiddenSize(hiddenSize)
        , mDataType(dtype)
        , mUseGptAttentionPlugin(false)
        , mInputPacked{false}
        , mPagedKvCache{false}
        , mTokensPerBlock{64}
        , mQuantMode{common::QuantMode::none()}
        , mMaxBatchSize(0)
        , mMaxBeamWidth(0)
        , mMaxInputLen(0)
        , mMaxSequenceLen(0)
        , mMaxNumTokens(std::nullopt)
        , mComputeContextLogits(false)
        , mComputeGenerationLogits(false)
        , mModelVariant(ModelVariant::kGpt)
        , mUseCustomAllReduce(false)
        , mMaxPromptEmbeddingTableSize(0)
        , mMaxDraftLen(0)
        , mUseContextFMHAForGeneration(false)
        , mPagedContextFMHA(false)
        , mUseLoraPlugin(false)
        , mLoraModules(std::vector<LoraModule>{})
        , mMlpHiddenSize(0)
    {
    }

    [[nodiscard]] SizeType constexpr getVocabSize() const noexcept
    {
        return mVocabSize;
    }

    [[nodiscard]] SizeType constexpr getVocabSizePadded(SizeType worldSize) const noexcept
    {
        return (mVocabSize + worldSize - 1) / worldSize * worldSize;
    }

    [[nodiscard]] SizeType constexpr getNbLayers(SizeType pipelineParallelism = 1) const
    {
        TLLM_CHECK(mNbLayers % pipelineParallelism == 0);
        return mNbLayers / pipelineParallelism;
    }

    [[nodiscard]] SizeType constexpr getNbHeads() const noexcept
    {
        return mNbHeads;
    }

    [[nodiscard]] SizeType constexpr getNbKvHeads() const noexcept
    {
        return mNbKvHeads;
    }

    void constexpr setNbKvHeads(SizeType nbKvHeads) noexcept
    {
        mNbKvHeads = nbKvHeads;
    }

    [[nodiscard]] SizeType constexpr getHiddenSize() const noexcept
    {
        return mHiddenSize;
    }

    [[nodiscard]] SizeType constexpr getSizePerHead() const noexcept
    {
        return mHiddenSize / mNbHeads;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept
    {
        return mDataType;
    }

    [[nodiscard]] bool constexpr useGptAttentionPlugin() const noexcept
    {
        return mUseGptAttentionPlugin;
    }

    void constexpr useGptAttentionPlugin(bool useGptAttentionPlugin) noexcept
    {
        mUseGptAttentionPlugin = useGptAttentionPlugin;
    }

    [[nodiscard]] bool constexpr usePackedInput() const noexcept
    {
        return mInputPacked;
    }

    void constexpr usePackedInput(bool inputPacked) noexcept
    {
        mInputPacked = inputPacked;
    }

    [[nodiscard]] bool constexpr usePagedKvCache() const noexcept
    {
        return mPagedKvCache;
    }

    void constexpr usePagedKvCache(bool pagedKvCache) noexcept
    {
        mPagedKvCache = pagedKvCache;
    }

    [[nodiscard]] SizeType constexpr getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    void constexpr setTokensPerBlock(SizeType TokensPerBlock) noexcept
    {
        mTokensPerBlock = TokensPerBlock;
    }

    [[nodiscard]] common::QuantMode constexpr getQuantMode() const noexcept
    {
        return mQuantMode;
    }

    void constexpr setQuantMode(common::QuantMode QuantMode) noexcept
    {
        mQuantMode = QuantMode;
    }

    [[nodiscard]] bool constexpr supportsInflightBatching() const noexcept
    {
        return mUseGptAttentionPlugin && mInputPacked && mPagedKvCache;
    }

    [[nodiscard]] SizeType constexpr getMaxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    void constexpr setMaxBatchSize(SizeType maxBatchSize) noexcept
    {
        mMaxBatchSize = maxBatchSize;
    }

    [[nodiscard]] SizeType constexpr getMaxBeamWidth() const noexcept
    {
        return mMaxBeamWidth;
    }

    void constexpr setMaxBeamWidth(SizeType maxBeamWidth) noexcept
    {
        mMaxBeamWidth = maxBeamWidth;
    }

    [[nodiscard]] SizeType constexpr getMaxInputLen() const noexcept
    {
        return mMaxInputLen;
    }

    void constexpr setMaxInputLen(SizeType maxInputLen) noexcept
    {
        mMaxInputLen = maxInputLen;
    }

    [[nodiscard]] SizeType constexpr getMaxSequenceLen() const noexcept
    {
        return mMaxSequenceLen;
    }

    void constexpr setMaxSequenceLen(SizeType maxSequenceLen) noexcept
    {
        mMaxSequenceLen = maxSequenceLen;
    }

    [[nodiscard]] std::optional<SizeType> constexpr getMaxNumTokens() const noexcept
    {
        return mMaxNumTokens;
    }

    void constexpr setMaxNumTokens(std::optional<SizeType> maxNumTokens) noexcept
    {
        mMaxNumTokens = maxNumTokens;
    }

    [[nodiscard]] bool constexpr usePromptTuning() const noexcept
    {
        return mMaxPromptEmbeddingTableSize > 0;
    }

    [[nodiscard]] SizeType constexpr getMaxPromptEmbeddingTableSize() const noexcept
    {
        return mMaxPromptEmbeddingTableSize;
    }

    void constexpr setMaxPromptEmbeddingTableSize(SizeType maxPromptEmbeddingTableSize) noexcept
    {
        mMaxPromptEmbeddingTableSize = maxPromptEmbeddingTableSize;
    }

    [[nodiscard]] bool constexpr computeContextLogits() const noexcept
    {
        return mComputeContextLogits;
    }

    void constexpr computeContextLogits(bool computeContextLogits) noexcept
    {
        mComputeContextLogits = computeContextLogits;
    }

    [[nodiscard]] bool constexpr computeGenerationLogits() const noexcept
    {
        return mComputeGenerationLogits;
    }

    void constexpr computeGenerationLogits(bool computeGenerationLogits) noexcept
    {
        mComputeGenerationLogits = computeGenerationLogits;
    }

    [[nodiscard]] ModelVariant getModelVariant() const
    {
        return mModelVariant;
    }

    void setModelVariant(ModelVariant modelVariant)
    {
        mModelVariant = modelVariant;
    }

    [[nodiscard]] bool constexpr useCustomAllReduce() const noexcept
    {
        return mUseCustomAllReduce;
    }

    void constexpr useCustomAllReduce(bool customAllReduce) noexcept
    {
        mUseCustomAllReduce = customAllReduce;
    }

    void constexpr setMaxDraftLen(SizeType maxDraftLen) noexcept
    {
        mMaxDraftLen = maxDraftLen;
    }

    [[nodiscard]] SizeType getMaxDraftLen() const
    {
        return mMaxDraftLen;
    }

    [[nodiscard]] SizeType constexpr getMaxTokensPerStep() const noexcept
    {
        return mMaxDraftLen + 1;
    }

    void constexpr setUseContextFMHAForGeneration(bool useContextFMHAForGeneration) noexcept
    {
        mUseContextFMHAForGeneration = useContextFMHAForGeneration;
    }

    [[nodiscard]] bool constexpr getContextFMHAForGeneration() const noexcept
    {
        return mUseContextFMHAForGeneration;
    }

    void constexpr setPagedContextFMHA(bool pagedContextFMHA) noexcept
    {
        mPagedContextFMHA = pagedContextFMHA;
    }

    [[nodiscard]] bool constexpr getPagedContextFMHA() const noexcept
    {
        return mPagedContextFMHA;
    }

    [[nodiscard]] bool constexpr useLoraPlugin() const noexcept
    {
        return mUseLoraPlugin;
    }

    void constexpr useLoraPlugin(bool useLoraPlugin) noexcept
    {
        mUseLoraPlugin = useLoraPlugin;
    }

    std::vector<LoraModule> const& getLoraModules() const noexcept
    {
        return mLoraModules;
    }

    void setLoraModules(std::vector<LoraModule> const& loraModules) noexcept
    {
        mLoraModules = loraModules;
    }

    [[nodiscard]] SizeType constexpr getMlpHiddenSize() const noexcept
    {
        return mMlpHiddenSize;
    }

    void constexpr setMlpHiddenSize(SizeType mlpHiddenSize) noexcept
    {
        mMlpHiddenSize = mlpHiddenSize;
    }

private:
    SizeType mVocabSize;
    SizeType mNbLayers;
    SizeType mNbHeads;
    SizeType mNbKvHeads;
    SizeType mHiddenSize;
    nvinfer1::DataType mDataType;
    bool mUseGptAttentionPlugin;
    bool mInputPacked;
    bool mPagedKvCache;
    SizeType mTokensPerBlock;
    common::QuantMode mQuantMode;
    SizeType mMaxBatchSize;
    SizeType mMaxBeamWidth;
    SizeType mMaxInputLen;
    SizeType mMaxSequenceLen;
    std::optional<SizeType> mMaxNumTokens;

    bool mComputeContextLogits;
    bool mComputeGenerationLogits;
    ModelVariant mModelVariant;
    bool mUseCustomAllReduce;

    SizeType mMaxPromptEmbeddingTableSize;
    SizeType mMaxDraftLen;

    bool mUseContextFMHAForGeneration;
    bool mPagedContextFMHA;

    bool mUseLoraPlugin;
    std::vector<LoraModule> mLoraModules;
    SizeType mMlpHiddenSize;
};
} // namespace tensorrt_llm::runtime
