/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvInfer.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"

#include <filesystem>
#include <vector>

namespace tensorrt_llm::testing
{

using tensorrt_llm::runtime::SizeType32;
using tensorrt_llm::runtime::SpeculativeDecodingMode;
using KVCacheType = tensorrt_llm::runtime::ModelConfig::KVCacheType;

enum class QuantMethod
{
    kNONE,
    kSMOOTH_QUANT,
};

enum class OutputContentType
{
    kNONE,
    kCONTEXT_LOGITS,
    kGENERATION_LOGITS,
    kLOG_PROBS,
    kCUM_LOG_PROBS
};

class ModelSpec
{
public:
    ModelSpec(std::string const& inputFile, nvinfer1::DataType dtype,
        std::shared_ptr<ModelSpec> otherModelSpecToCompare = nullptr)
        : mInputFile{std::move(inputFile)}
        , mDataType{dtype}
        , mOtherModelSpecToCompare(otherModelSpecToCompare)
    {
    }

    ModelSpec& setInputFile(std::string const& inputFile)
    {
        mInputFile = inputFile;
        return *this;
    }

    ModelSpec& useGptAttentionPlugin()
    {
        mUseGptAttentionPlugin = true;
        return *this;
    }

    ModelSpec& usePackedInput()
    {
        mUsePackedInput = true;
        return *this;
    }

    ModelSpec& setKVCacheType(KVCacheType kvCacheType)
    {
        mKVCacheType = kvCacheType;
        return *this;
    }

    ModelSpec& setKVCacheReuse(bool kvCacheReuse)
    {
        mKVCacheReuse = kvCacheReuse;
        return *this;
    }

    ModelSpec& useDecoderPerRequest()
    {
        mDecoderPerRequest = true;
        return *this;
    }

    ModelSpec& useTensorParallelism(int tensorParallelism)
    {
        mTPSize = tensorParallelism;
        return *this;
    }

    ModelSpec& usePipelineParallelism(int pipelineParallelism)
    {
        mPPSize = pipelineParallelism;
        return *this;
    }

    ModelSpec& useContextParallelism(int contextParallelism)
    {
        mCPSize = contextParallelism;
        return *this;
    }

    ModelSpec& setDraftTokens(SizeType32 maxDraftTokens)
    {
        mMaxDraftTokens = maxDraftTokens;
        return *this;
    }

    ModelSpec& useAcceptByLogits()
    {
        mAcceptDraftByLogits = true;
        return *this;
    }

    ModelSpec& useMambaPlugin()
    {
        mUseMambaPlugin = true;
        return *this;
    }

    ModelSpec& gatherLogits()
    {
        mGatherLogits = true;
        return *this;
    }

    ModelSpec& replaceLogits()
    {
        mReplaceLogits = true;
        return *this;
    }

    ModelSpec& returnLogProbs()
    {
        mReturnLogProbs = true;
        return *this;
    }

    ModelSpec& smokeTest()
    {
        mSmokeTest = true;
        return *this;
    }

    ModelSpec& useMedusa()
    {
        mSpecDecodingMode = SpeculativeDecodingMode::Medusa();
        return *this;
    }

    ModelSpec& useEagle()
    {
        mSpecDecodingMode = SpeculativeDecodingMode::Eagle();
        return *this;
    }

    ModelSpec& useLookaheadDecoding()
    {
        mSpecDecodingMode = SpeculativeDecodingMode::LookaheadDecoding();
        return *this;
    }

    ModelSpec& useExplicitDraftTokensDecoding()
    {
        mSpecDecodingMode = SpeculativeDecodingMode::ExplicitDraftTokens();
        return *this;
    }

    ModelSpec& useDraftTokensExternalDecoding()
    {
        mSpecDecodingMode = SpeculativeDecodingMode::DraftTokensExternal();
        return *this;
    }

    [[nodiscard]] bool useLogits() const
    {
        return mGatherLogits || mReplaceLogits;
    }

    ModelSpec& useMultipleProfiles()
    {
        mUseMultipleProfiles = true;
        return *this;
    }

    ModelSpec& enableContextFMHAFp32Acc()
    {
        mEnableContextFMHAFp32Acc = true;
        return *this;
    }

    [[nodiscard]] bool getEnableContextFMHAFp32Acc() const
    {
        return mEnableContextFMHAFp32Acc;
    }

    ModelSpec& setMaxInputLength(SizeType32 maxInputLength)
    {
        mMaxInputLength = maxInputLength;
        return *this;
    }

    ModelSpec& setMaxOutputLength(SizeType32 maxOutputLength)
    {
        mMaxOutputLength = maxOutputLength;
        return *this;
    }

    ModelSpec& setQuantMethod(QuantMethod quantMethod)
    {
        mQuantMethod = quantMethod;
        return *this;
    }

    ModelSpec& useLoraPlugin()
    {
        mUseLoraPlugin = true;
        return *this;
    }

    ModelSpec& collectGenerationLogitsFile()
    {
        mCollectGenerationLogits = true;
        return *this;
    }

    ModelSpec& collectContextLogitsFile()
    {
        mCollectContextLogits = true;
        return *this;
    }

    ModelSpec& collectCumLogProbsFile()
    {
        mCollectCumLogProbs = true;
        return *this;
    }

    ModelSpec& collectLogProbsFile()
    {
        mCollectLogProbs = true;
        return *this;
    }

    ModelSpec& capacitySchedulerPolicy(tensorrt_llm::executor::CapacitySchedulerPolicy policy)
    {
        mCapacitySchedulerPolicy = policy;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, ModelSpec const& modelSpec)
    {
        return os << modelSpec.getModelPath();
    }

    // Computed properties
    [[nodiscard]] std::string getInputFile() const;

    [[nodiscard]] std::string getModelPath() const;

    [[nodiscard]] std::string getResultsFileInternal(
        OutputContentType outputContentType = OutputContentType::kNONE) const;

    [[nodiscard]] std::string getResultsFile() const;
    [[nodiscard]] std::string getGenerationLogitsFile() const;

    [[nodiscard]] std::string getContextLogitsFile() const;

    [[nodiscard]] std::string getCumLogProbsFile() const;

    [[nodiscard]] std::string getLogProbsFile() const;

    [[nodiscard]] std::string getDtypeString() const;

    [[nodiscard]] std::string getQuantMethodString() const;

    [[nodiscard]] std::string getKVCacheTypeString() const;

    [[nodiscard]] std::string getSpeculativeDecodingModeString() const;

    [[nodiscard]] std::string getCapacitySchedulerString() const;

    static ModelSpec getDefaultModelSpec()
    {
        static ModelSpec modelSpec{"input_tokens.npy", nvinfer1::DataType::kHALF};
        modelSpec.useGptAttentionPlugin().setKVCacheType(KVCacheType::kPAGED).usePackedInput();

        return modelSpec;
    }

    std::string mInputFile;
    nvinfer1::DataType mDataType;

    bool mUseGptAttentionPlugin{false};
    bool mUsePackedInput{false};
    KVCacheType mKVCacheType{KVCacheType::kCONTINUOUS};
    bool mKVCacheReuse{false};
    bool mDecoderPerRequest{false};
    int mPPSize{1};
    int mTPSize{1};
    int mCPSize{1};
    int mMaxDraftTokens{0};
    bool mAcceptDraftByLogits{false};
    bool mUseMambaPlugin{false};
    bool mGatherLogits{false};
    bool mReplaceLogits{false};
    bool mReturnLogProbs{false};
    bool mSmokeTest{false};
    bool mUseMultipleProfiles{false};
    int mMaxInputLength{0};
    int mMaxOutputLength{0};
    bool mUseLoraPlugin{false};
    bool mEnableContextFMHAFp32Acc{false};

    // Flags to store whether model spec wants collect these outputs, you could call getXXXFile() if you need the name.
    bool mCollectGenerationLogits{false};
    bool mCollectContextLogits{false};
    bool mCollectCumLogProbs{false};
    bool mCollectLogProbs{false};
    QuantMethod mQuantMethod{QuantMethod::kNONE};

    SpeculativeDecodingMode mSpecDecodingMode{SpeculativeDecodingMode::None()};

    std::optional<tensorrt_llm::executor::CapacitySchedulerPolicy> mCapacitySchedulerPolicy{std::nullopt};

    // Sometimes, we need to compare with another model spec for golden results.
    std::shared_ptr<ModelSpec> mOtherModelSpecToCompare{nullptr};
};

}; // namespace tensorrt_llm::testing
