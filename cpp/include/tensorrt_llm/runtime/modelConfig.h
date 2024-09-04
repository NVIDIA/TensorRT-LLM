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
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"
#include "tensorrt_llm/runtime/speculativeDecodingModule.h"

#include <NvInferRuntime.h>
#include <array>

namespace tensorrt_llm::runtime
{

class ModelConfig
{
public:
    // See `split_point` defined in `tensorrt_llm/models/generation_mixin.py`.
    // The split points are tuned to get better perf, if we need to let
    // users tune that, we can support that by writing and reading the
    // points in `config.json`.
    static constexpr std::array kOPT_PROFILES_SPLIT_POINTS{64, 128, 256, 512, 1024};

    enum class ModelVariant : std::int32_t
    {
        kGpt = 0,
        kChatGlm = 1,        // https://github.com/THUDM/ChatGLM-6B
        kGlm = 2,            // https://github.com/THUDM/GLM
        kMamba = 3,          // https://github.com/state-spaces/mamba
        kRecurrentGemma = 4, // https://github.com/google-deepmind/recurrentgemma
        kEncDec = 5,
    };

    struct RnnConfig
    {
        SizeType32 stateSize = 0;
        SizeType32 convKernel = 0;
        SizeType32 rnnHiddenSize = 0;
        SizeType32 rnnHeadSize = 0;
        SizeType32 rnnConvDimSize = 0;
    };

    enum class LayerType : std::int32_t
    {
        kATTENTION,
        kRECURRENT,
    };

    enum class KVCacheType : std::int32_t
    {
        kCONTINUOUS,
        kPAGED,
        kDISABLED,
    };

    static KVCacheType KVCacheTypeFromString(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(), ::toupper);

        if (value == "CONTINUOUS")
        {
            return KVCacheType::kCONTINUOUS;
        }
        else if (value == "PAGED")
        {
            return KVCacheType::kPAGED;
        }
        else if (value == "DISABLED")
        {
            return KVCacheType::kDISABLED;
        }
        else
        {
            throw std::invalid_argument("Invalid KV cache type: " + value);
        }
    }

    enum class ManageWeightsType : std::int32_t
    {
        kDisabled,
        kEnabled,
    };

    explicit ModelConfig(SizeType32 vocabSize, SizeType32 nbAttentionLayers, SizeType32 nbRnnLayers, SizeType32 nbHeads,
        SizeType32 hiddenSize, nvinfer1::DataType dtype)
        : mVocabSize(vocabSize)
        , mNbAttentionLayers(nbAttentionLayers)
        , mNbRnnLayers(nbRnnLayers)
        , mNbHeads(nbHeads)
        , mNbKvHeads(nbHeads)
        , mHiddenSize(hiddenSize)
        , mSizePerHead(mHiddenSize / mNbHeads)
        , mDataType(dtype)
        , mUseGptAttentionPlugin(false)
        , mUseMambaConv1dPlugin(false)
        , mInputPacked{false}
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
        , mMaxPromptEmbeddingTableSize(0)
        , mContextFMHA(false)
        , mPagedContextFMHA(false)
        , mUseXQA{false}
        , mUseLoraPlugin(false)
        , mMlpHiddenSize(0)
        , mUseCrossAttention(false)
        , mUsePositionEmbedding(false)
        , mUseTokenTypeEmbedding(false)
        , mSpeculativeDecodingMode(SpeculativeDecodingMode::None())
        , mLogitsDtype(nvinfer1::DataType::kFLOAT)
        , mUseShapeInference(true)
        , mManageWeightsType(ManageWeightsType::kDisabled)
    {
    }

    [[nodiscard]] static std::vector<SizeType32> getOptProfilesSplitPoints() noexcept
    {
        return {kOPT_PROFILES_SPLIT_POINTS.begin(), kOPT_PROFILES_SPLIT_POINTS.end()};
    }

    [[nodiscard]] SizeType32 constexpr getVocabSize() const noexcept
    {
        return mVocabSize;
    }

    [[nodiscard]] SizeType32 constexpr getVocabSizePadded(SizeType32 worldSize) const noexcept
    {
        return (mVocabSize + worldSize - 1) / worldSize * worldSize;
    }

    [[nodiscard]] SizeType32 constexpr getNbAttentionLayers(SizeType32 pipelineParallelism = 1) const
    {
        return mNbAttentionLayers / pipelineParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getNbRnnLayers(SizeType32 pipelineParallelism = 1) const
    {
        return mNbRnnLayers / pipelineParallelism;
    }

    [[nodiscard]] SizeType32 constexpr getNbHeads() const noexcept
    {
        return mNbHeads;
    }

    [[nodiscard]] SizeType32 constexpr getNbKvHeads() const noexcept
    {
        return mNbKvHeads;
    }

    void constexpr setNbKvHeads(SizeType32 nbKvHeads) noexcept
    {
        mNbKvHeads = nbKvHeads;
    }

    [[nodiscard]] SizeType32 constexpr getHiddenSize() const noexcept
    {
        return mHiddenSize;
    }

    [[nodiscard]] SizeType32 constexpr getEncoderHiddenSize() const noexcept
    {
        return mEncoderHiddenSize;
    }

    void constexpr setEncoderHiddenSize(SizeType32 encoderHiddenSize) noexcept
    {
        mEncoderHiddenSize = encoderHiddenSize;
    }

    [[nodiscard]] SizeType32 constexpr getSizePerHead() const noexcept
    {
        return mSizePerHead;
    }

    void constexpr setSizePerHead(SizeType32 sizePerHead) noexcept
    {
        mSizePerHead = sizePerHead;
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

    [[nodiscard]] bool constexpr useMambaConv1dPlugin() const noexcept
    {
        return mUseMambaConv1dPlugin;
    }

    void constexpr useMambaConv1dPlugin(bool useMambaConv1dPlugin) noexcept
    {
        mUseMambaConv1dPlugin = useMambaConv1dPlugin;
    }

    [[nodiscard]] bool constexpr usePackedInput() const noexcept
    {
        return mInputPacked;
    }

    void constexpr usePackedInput(bool inputPacked) noexcept
    {
        mInputPacked = inputPacked;
    }

    [[nodiscard]] bool constexpr usePagedState() const noexcept
    {
        return mPagedState;
    }

    void constexpr usePagedState(bool pagedState) noexcept
    {
        mPagedState = pagedState;
    }

    [[nodiscard]] SizeType32 constexpr getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    void constexpr setTokensPerBlock(SizeType32 TokensPerBlock) noexcept
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
        return (isTransformerBased() && mUseGptAttentionPlugin && mInputPacked
                   && (mKVCacheType == KVCacheType::kDISABLED || mKVCacheType == KVCacheType::kPAGED))
            || (isRnnBased() && mUseMambaConv1dPlugin && mInputPacked && mPagedState);
    }

    [[nodiscard]] SizeType32 constexpr getMaxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    void constexpr setMaxBatchSize(SizeType32 maxBatchSize) noexcept
    {
        mMaxBatchSize = maxBatchSize;
    }

    [[nodiscard]] SizeType32 constexpr getMaxBeamWidth() const noexcept
    {
        return mMaxBeamWidth;
    }

    void constexpr setMaxBeamWidth(SizeType32 maxBeamWidth) noexcept
    {
        mMaxBeamWidth = maxBeamWidth;
    }

    [[nodiscard]] SizeType32 constexpr getMaxInputLen() const noexcept
    {
        return mMaxInputLen;
    }

    void constexpr setMaxInputLen(SizeType32 maxInputLen) noexcept
    {
        mMaxInputLen = maxInputLen;
    }

    [[nodiscard]] SizeType32 constexpr getMaxSequenceLen() const noexcept
    {
        return mMaxSequenceLen;
    }

    void constexpr setMaxSequenceLen(SizeType32 maxSequenceLen) noexcept
    {
        mMaxSequenceLen = maxSequenceLen;
    }

    [[nodiscard]] std::optional<SizeType32> constexpr getMaxNumTokens() const noexcept
    {
        return mMaxNumTokens;
    }

    void constexpr setMaxNumTokens(std::optional<SizeType32> maxNumTokens) noexcept
    {
        mMaxNumTokens = maxNumTokens;
    }

    [[nodiscard]] SizeType32 constexpr getMaxEncoderLen() const noexcept
    {
        return mMaxEncoderLen;
    }

    void constexpr setMaxEncoderLen(SizeType32 maxEncoderLen) noexcept
    {
        mMaxEncoderLen = maxEncoderLen;
    }

    [[nodiscard]] bool constexpr usePromptTuning() const noexcept
    {
        return mMaxPromptEmbeddingTableSize > 0;
    }

    [[nodiscard]] SizeType32 constexpr getMaxPromptEmbeddingTableSize() const noexcept
    {
        return mMaxPromptEmbeddingTableSize;
    }

    void constexpr setMaxPromptEmbeddingTableSize(SizeType32 maxPromptEmbeddingTableSize) noexcept
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

    [[nodiscard]] SizeType32 getMaxDecodingDraftTokens() const
    {
        return getSpeculativeDecodingMode().isNone() ? 0 : getSpeculativeDecodingModule().getMaxDecodingDraftTokens();
    }

    [[nodiscard]] SizeType32 constexpr getMaxDecodingTokens() const noexcept
    {
        return getSpeculativeDecodingMode().isNone() ? 1 : getSpeculativeDecodingModule().getMaxDecodingTokens();
    }

    void constexpr setContextFMHA(bool contextFMHA) noexcept
    {
        mContextFMHA = contextFMHA;
    }

    [[nodiscard]] bool constexpr getContextFMHA() const noexcept
    {
        return mContextFMHA;
    }

    void constexpr setPagedContextFMHA(bool pagedContextFMHA) noexcept
    {
        mPagedContextFMHA = pagedContextFMHA;
    }

    [[nodiscard]] bool constexpr getPagedContextFMHA() const noexcept
    {
        return mPagedContextFMHA;
    }

    void constexpr useXQA(bool useXQA) noexcept
    {
        mUseXQA = useXQA;
    }

    [[nodiscard]] bool constexpr useXQA() const noexcept
    {
        return mUseXQA;
    }

    [[nodiscard]] bool constexpr useLoraPlugin() const noexcept
    {
        return mUseLoraPlugin;
    }

    void constexpr useLoraPlugin(bool useLoraPlugin) noexcept
    {
        mUseLoraPlugin = useLoraPlugin;
    }

    [[nodiscard]] std::vector<LoraModule> const& getLoraModules() const noexcept
    {
        return mLoraModules;
    }

    void setLoraModules(std::vector<LoraModule> const& loraModules) noexcept
    {
        mLoraModules = loraModules;
    }

    [[nodiscard]] SizeType32 constexpr getMlpHiddenSize() const noexcept
    {
        return mMlpHiddenSize;
    }

    void constexpr setMlpHiddenSize(SizeType32 mlpHiddenSize) noexcept
    {
        mMlpHiddenSize = mlpHiddenSize;
    }

    // Utility functions for fast KVCacheType checking.
    [[nodiscard]] bool constexpr isKVCacheEnabled() const noexcept
    {
        return mKVCacheType != KVCacheType::kDISABLED;
    }

    [[nodiscard]] bool constexpr isPagedKVCache() const noexcept
    {
        return mKVCacheType == KVCacheType::kPAGED;
    }

    [[nodiscard]] bool constexpr isContinuousKVCache() const noexcept
    {
        return mKVCacheType == KVCacheType::kCONTINUOUS;
    }

    [[nodiscard]] KVCacheType constexpr getKVCacheType() const noexcept
    {
        return mKVCacheType;
    }

    void constexpr setKVCacheType(KVCacheType kvCacheType) noexcept
    {
        mKVCacheType = kvCacheType;
    }

    [[nodiscard]] bool constexpr useCrossAttention() const noexcept
    {
        return mUseCrossAttention;
    }

    void constexpr setUseCrossAttention(bool useCrossAttention) noexcept
    {
        mUseCrossAttention = useCrossAttention;
    }

    [[nodiscard]] bool constexpr usePositionEmbedding() const noexcept
    {
        return mUsePositionEmbedding;
    }

    void constexpr setUsePositionEmbedding(bool usePositionEmbedding) noexcept
    {
        mUsePositionEmbedding = usePositionEmbedding;
    }

    [[nodiscard]] bool constexpr useTokenTypeEmbedding() const noexcept
    {
        return mUseTokenTypeEmbedding;
    }

    void constexpr setUseTokenTypeEmbedding(bool useTokenTypeEmbedding) noexcept
    {
        mUseTokenTypeEmbedding = useTokenTypeEmbedding;
    }

    [[nodiscard]] SizeType32 constexpr getMaxLoraRank() const noexcept
    {
        return mMaxLoraRank;
    }

    void constexpr setMaxLoraRank(SizeType32 maxLoraRank) noexcept
    {
        mMaxLoraRank = maxLoraRank;
    }

    void setSpeculativeDecodingMode(SpeculativeDecodingMode mode) noexcept
    {
        mSpeculativeDecodingMode = mode;
    }

    [[nodiscard]] bool hasSpeculativeDecodingModule() const noexcept
    {
        return mSpeculativeDecodingModule != nullptr;
    }

    [[nodiscard]] SpeculativeDecodingModule const& getSpeculativeDecodingModule() const noexcept
    {
        TLLM_CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set");
        return *mSpeculativeDecodingModule;
    }

    [[nodiscard]] std::shared_ptr<SpeculativeDecodingModule const> getSpeculativeDecodingModulePtr() const noexcept
    {
        TLLM_CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set");
        return mSpeculativeDecodingModule;
    }

    [[nodiscard]] std::shared_ptr<SpeculativeDecodingModule> getSpeculativeDecodingModulePtr() noexcept
    {
        TLLM_CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set");
        return mSpeculativeDecodingModule;
    }

    void setSpeculativeDecodingModule(
        std::shared_ptr<SpeculativeDecodingModule> const& speculativeDecodingModule) noexcept
    {
        mSpeculativeDecodingModule = speculativeDecodingModule;
    }

    [[nodiscard]] nvinfer1::DataType getKvDataType() const noexcept
    {
        if (getQuantMode().hasFp8KvCache())
        {
            return nvinfer1::DataType::kFP8;
        }
        else if (getQuantMode().hasInt8KvCache())
        {
            return nvinfer1::DataType::kINT8;
        }
        else
        {
            return getDataType();
        }
    }

    [[nodiscard]] bool constexpr isTransformerBased() const noexcept
    {
        return mModelVariant == ModelVariant::kGpt || mModelVariant == ModelVariant::kGlm
            || mModelVariant == ModelVariant::kChatGlm || mModelVariant == ModelVariant::kRecurrentGemma;
    }

    [[nodiscard]] bool hasRnnConfig() const noexcept
    {
        return mRnnConfig.has_value();
    }

    [[nodiscard]] std::optional<RnnConfig> getRnnConfig() const noexcept
    {
        return mRnnConfig;
    }

    void setRnnConfig(RnnConfig const& rnnConfig) noexcept
    {
        mRnnConfig = rnnConfig;
    }

    [[nodiscard]] bool constexpr isRnnBased() const noexcept
    {
        return mModelVariant == ModelVariant::kMamba || mModelVariant == ModelVariant::kRecurrentGemma;
    }

    [[nodiscard]] std::vector<LayerType> const& getLayerTypes() const noexcept
    {
        return mLayerTypes;
    }

    void setLayerTypes(std::vector<LayerType> const& layerTypes) noexcept
    {
        mLayerTypes = layerTypes;
    }

    [[nodiscard]] SpeculativeDecodingMode constexpr getSpeculativeDecodingMode() const noexcept
    {
        return mSpeculativeDecodingMode;
    }

    void setLogitsDtype(nvinfer1::DataType inputDtype) noexcept
    {
        mLogitsDtype = inputDtype;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getLogitsDtype() const noexcept
    {
        return mLogitsDtype;
    }

    void setUseShapeInference(bool useShapeInference) noexcept
    {
        mUseShapeInference = useShapeInference;
    }

    [[nodiscard]] bool useShapeInference() const noexcept
    {
        return mUseShapeInference;
    }

    [[nodiscard]] ManageWeightsType getManageWeightsType() const noexcept
    {
        return mManageWeightsType;
    }

    void setManageWeightsType(const ManageWeightsType manageWeightType) noexcept
    {
        mManageWeightsType = manageWeightType;
    }

    [[nodiscard]] std::string const& getModelName() const noexcept
    {
        return mModelName;
    }

    void setModelName(std::string const& modelName)
    {
        mModelName = modelName;
    }

private:
    SizeType32 mVocabSize;
    SizeType32 mNbAttentionLayers;
    SizeType32 mNbRnnLayers;
    SizeType32 mNbHeads;
    SizeType32 mNbKvHeads;
    SizeType32 mHiddenSize;
    SizeType32 mSizePerHead;
    nvinfer1::DataType mDataType;
    bool mUseGptAttentionPlugin;
    bool mUseMambaConv1dPlugin;
    bool mInputPacked;
    bool mPagedState;
    SizeType32 mTokensPerBlock;
    common::QuantMode mQuantMode;
    SizeType32 mMaxBatchSize;
    SizeType32 mMaxBeamWidth;
    SizeType32 mMaxInputLen;
    SizeType32 mMaxSequenceLen;
    std::optional<SizeType32> mMaxNumTokens;

    bool mComputeContextLogits;
    bool mComputeGenerationLogits;
    ModelVariant mModelVariant;

    SizeType32 mMaxPromptEmbeddingTableSize;

    bool mContextFMHA;
    bool mPagedContextFMHA;
    bool mUseXQA;

    bool mUseLoraPlugin;
    std::vector<LoraModule> mLoraModules;
    SizeType32 mMlpHiddenSize;
    SizeType32 mMaxLoraRank;

    std::optional<RnnConfig> mRnnConfig;

    // Whether kv_cache is enabled. In kv_cache is disabled, it is only intended for context phase only now.
    KVCacheType mKVCacheType = KVCacheType::kCONTINUOUS;

    // Configs related to encoder / enc-dec models
    SizeType32 mMaxEncoderLen{};
    SizeType32 mEncoderHiddenSize{};
    bool mUseCrossAttention;
    bool mUsePositionEmbedding;
    bool mUseTokenTypeEmbedding;

    std::vector<LayerType> mLayerTypes;
    // Speculative decoding members
    std::shared_ptr<SpeculativeDecodingModule> mSpeculativeDecodingModule;
    SpeculativeDecodingMode mSpeculativeDecodingMode;

    // Logits datatype
    nvinfer1::DataType mLogitsDtype;
    bool mUseShapeInference;
    ManageWeightsType mManageWeightsType;
    std::string mModelName;
};

} // namespace tensorrt_llm::runtime
