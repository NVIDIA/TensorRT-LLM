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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <tensorrt_llm/runtime/common.h>
#include <tensorrt_llm/runtime/speculativeDecodingModule.h>

#include <optional>
#include <utility>
#include <vector>

namespace tensorrt_llm::layers
{

using TensorPtr = runtime::ITensor::SharedPtr;
using TensorConstPtr = runtime::ITensor::SharedConstPtr;
using BufferPtr = runtime::IBuffer::SharedPtr;
using BufferConstPtr = runtime::IBuffer::SharedConstPtr;
template <typename T>
using OptVec = std::optional<std::vector<T>>;

//!
//! \brief In a DecodingLayer's life cycle, it is constructed once;
//! `setup` repeatedly, but once per request; `forward*` repeatedly, many times per request.
//! A possible sequence would be, construct(maxBatchSize) -> setup({1,3}) -> forward({1, 3})
//! -> forward({1, 3}) -> setup({2,4}) -> forward({1, 3, 2, 4}) -> forward({1, 3, 2, 4})
//! -> forward({1, 2, 4}), where {a,b} are batchSlots, and {3} ends at last step.
//! As a result there are three types of batches.
//! 1. `maxBatchSize` for each layers to reserve resources.
//!    It is passed through class constructor, in DecoderDomain.getBatchSize().
//! 2. `setupBatchSize` for setting up layers for a batch of new requests.
//!    It is passed through `setup` method.
//! 3. `forwardBatchSize` for layers forwarding for a batch of existing active requests.
//!    it is passed through `forwardAsync` and `forwardSync` methods.
//! `setup` and `forward` always provide `batchSlots` indexed by
//! local batch index ranging in [0, setupBatchSize) or [0, forwardBatchSize),
//! holding the global batch index ranging in [0, maxBatchSize).
//! In case of beam search, maxBatchSize = forwardBatchSize = 1.

class DecoderDomain
{
public:
    DecoderDomain(runtime::SizeType32 batchSize, runtime::SizeType32 beamWidth, runtime::SizeType32 vocabSize,
        std::optional<runtime::SizeType32> vocabSizePadded = std::nullopt,
        std::shared_ptr<runtime::SpeculativeDecodingModule const> speculativeDecodingModule = nullptr)
        : mBatchSize(batchSize)
        , mBeamWidth(beamWidth)
        , mVocabSize(vocabSize)
        , mVocabSizePadded(vocabSizePadded.value_or(vocabSize))
        , mSpeculativeDecodingModule(std::move(speculativeDecodingModule))
    {
    }

    [[nodiscard]] runtime::SizeType32 getBatchSize() const
    {
        return mBatchSize;
    }

    [[nodiscard]] runtime::SizeType32 getBeamWidth() const
    {
        return mBeamWidth;
    }

    void setBeamWidth(runtime::SizeType32 beamWidth)
    {
        mBeamWidth = beamWidth;
    }

    [[nodiscard]] runtime::SizeType32 getVocabSize() const
    {
        return mVocabSize;
    }

    [[nodiscard]] runtime::SizeType32 getVocabSizePadded() const
    {
        return mVocabSizePadded;
    }

    [[nodiscard]] runtime::SizeType32 getMaxDecodingTokens() const
    {
        return mSpeculativeDecodingModule ? mSpeculativeDecodingModule->getMaxDecodingTokens() : 1;
    }

    [[nodiscard]] std::shared_ptr<runtime::SpeculativeDecodingModule const> getSpeculativeDecodingModule() const
    {
        TLLM_CHECK_WITH_INFO(mSpeculativeDecodingModule, "Speculative decoding module is not set to decoder domain");
        return mSpeculativeDecodingModule;
    }

    [[nodiscard]] std::shared_ptr<runtime::SpeculativeDecodingModule const> getSpeculativeDecodingModulePtr() const
    {
        return mSpeculativeDecodingModule;
    }

private:
    runtime::SizeType32 mBatchSize;
    runtime::SizeType32 mBeamWidth;
    runtime::SizeType32 mVocabSize;
    runtime::SizeType32 mVocabSizePadded;
    std::shared_ptr<runtime::SpeculativeDecodingModule const> mSpeculativeDecodingModule;
};

class BaseSetupParams
{
public:
    virtual ~BaseSetupParams() = default;
};

// Penalty layer
class PenaltySetupParams : public BaseSetupParams
{
public:
    OptVec<float> temperature;             // [1] or [setupBatchSize]
    OptVec<runtime::SizeType32> minLength; // [1] or [setupBatchSize]
    OptVec<float> repetitionPenalty;       // [1] or [setupBatchSize]
    OptVec<float> presencePenalty;         // [1] or [setupBatchSize]
    OptVec<float> frequencyPenalty;        // [1] or [setupBatchSize]
};

// Ban words layer
class BanWordsSetupParams : public BaseSetupParams
{
public:
    OptVec<runtime::SizeType32> noRepeatNgramSize; // [1] or [setupBatchSize]
};

class DecodingSetupParams : public BaseSetupParams
{
public:
    virtual ~DecodingSetupParams() = default;

    OptVec<uint64_t> randomSeed; // [1] or [setupBatchSize]
    OptVec<bool> outputLogProbs; // [setupBatchSize]
    OptVec<bool> cumLogProbs;    // [setupBatchSize]
};

class SamplingSetupParams : public DecodingSetupParams
{
public:
    // baseSamplingLayer
    OptVec<runtime::SizeType32> runtimeTopK; // [1] or [setupBatchSize]
    OptVec<float> runtimeTopP;               // [1] or [setupBatchSize]
    OptVec<float> runtimeMinP;               // [1] or [setupBatchSize]

    // topPSamplingLayer
    OptVec<float> topPDecay;                   // [setupBatchSize], between [0, 1]
    OptVec<float> topPMin;                     // [setupBatchSize], between [0, 1]
    OptVec<runtime::TokenIdType> topPResetIds; // [setupBatchSize]
    std::optional<bool> normalizeLogProbs;
};

class BeamSearchSetupParams : public DecodingSetupParams
{
public:
    // BeamSearchLayer
    OptVec<float> beamSearchDiversityRate;                   // [setupBatchSize]
    OptVec<float> lengthPenalty;                             // [setupBatchSize]
    OptVec<int> earlyStopping;                               // [setupBatchSize]
    OptVec<std::vector<runtime::SizeType32>> beamWidthArray; // [setupBatchSize, nMaxBeamWidthArray]
    bool hasDiffRuntimeArgs{false};
};

class MedusaSetupParams : public DecodingSetupParams
{
public:
    // Medusa params
    OptVec<runtime::SizeType32> runtimeTopK;                   // [setupBatchSize]
    OptVec<std::vector<runtime::SizeType32>> runtimeHeadsTopK; // [setupBatchSize, maxMedusaHeads]
};

class ExplicitDraftTokensSetupParams : public DecodingSetupParams
{
public:
    OptVec<float> temperature; // [setupBatchSize]
    // Hack to init some data for the context phase in the setup.
    TensorPtr randomDataSample; // [maxBatchSize], on gpu
    TensorPtr temperatures;     // [maxBatchSize], on gpu
    nvinfer1::DataType dtype;   // [1]
};

class EagleSetupParams : public DecodingSetupParams
{
public:
    OptVec<float> temperature; // [setupBatchSize]
    // Hack to init some data for the context phase in the setup.
    TensorPtr randomDataSample; // [maxBatchSize], on gpu
    TensorPtr temperatures;     // [maxBatchSize], on gpu
    nvinfer1::DataType dtype;   // [1]
};

class DynamicDecodeSetupParams : public BaseSetupParams
{
public:
    std::shared_ptr<PenaltySetupParams> penaltyParams;
    std::shared_ptr<BanWordsSetupParams> banWordsParams;
    std::shared_ptr<DecodingSetupParams> decodingParams;
};

struct LookaheadSetupParams : public DecodingSetupParams
{
    using TensorPtr = runtime::ITensor::SharedPtr;

    std::vector<runtime::ITensor::SharedConstPtr> prompt;       // [batchSize][maxSeqLen], on cpu
    std::vector<executor::LookaheadDecodingConfig> algoConfigs; // [1] or [batchSize]

    //! see class LookaheadDecodingOutputs
    TensorPtr generationLengths;    // [maxBatchSize], on gpu
    TensorPtr positionOffsets;      // [maxBatchSize, maxDecodingTokens], on gpu
    TensorPtr attentionPackedMasks; // [maxBatchSize, maxDecodingTokens], on gpu
};

class ExternalDraftTokensSetupParams : public DecodingSetupParams
{
public:
    OptVec<runtime::SizeType32> runtimeTopK; // [1] or [setupBatchSize]
    OptVec<float> runtimeTopP;               // [1] or [setupBatchSize]
};

class BaseDecodingInputs
{
public:
    BaseDecodingInputs(runtime::SizeType32 localBatchSize)
        : localBatchSize(localBatchSize)
    {
    }

    virtual ~BaseDecodingInputs() = default;

    runtime::SizeType32 localBatchSize;
};

// Ban words inputs
class BanWordsDecodingInputs : public BaseDecodingInputs
{
public:
    BanWordsDecodingInputs(runtime::SizeType32 localBatchSize)
        : BaseDecodingInputs(localBatchSize)
    {
    }

    runtime::SizeType32 maxBadWordsLen{0};
    std::optional<TensorConstPtr> badWordsPtr;     // [maxBatchSize][2, bad_words_length], on gpu
    std::optional<TensorConstPtr> badWordsLengths; // [maxBatchSize], on gpu
};

// Stop criteria inputs
class StopCriteriaDecodingInputs : public BaseDecodingInputs
{
public:
    StopCriteriaDecodingInputs(runtime::SizeType32 localBatchSize)
        : BaseDecodingInputs(localBatchSize)
    {
    }

    runtime::SizeType32 maxStopWordsLen{0};
    std::optional<TensorConstPtr> sequenceLimitLength; // [maxBatchSize], on gpu
    std::optional<TensorConstPtr> stopWordsPtr;        // [maxBatchSize][2, stop_words_length], on pinned
    std::optional<TensorConstPtr> stopWordsLengths;    // [maxBatchSize], on pinned
};

class DecodingInputs : public BaseDecodingInputs
{
public:
    DecodingInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 step = 0,
        runtime::SizeType32 ite = 0, runtime::SizeType32 localBatchSize = 0, runtime::SizeType32 maxAttentionWindow = 0,
        runtime::SizeType32 sinkTokenLength = 0)
        : BaseDecodingInputs(localBatchSize)
        , endIds{std::move(endIds)}
        , step{step}
        , ite{ite}
        , maxAttentionWindow{maxAttentionWindow}
        , sinkTokenLength{sinkTokenLength}
        , batchSlots{std::move(batchSlots)}
    {
    }

    TensorConstPtr endIds; // [maxBatchSize]

    // used only for python runtime
    runtime::SizeType32 step;
    runtime::SizeType32 ite;

    // mandatory parameters
    runtime::SizeType32 maxAttentionWindow;
    runtime::SizeType32 sinkTokenLength;

    //! One of `logits` and `logitsVec` has to be set
    //! DynamicDecodeLayer::forward checks for it
    //! Need both of these fields to support legacy code during transition period to the batched decoder
    std::optional<TensorConstPtr> logits; // [forwardBatchSize, beamWidth, vocabSizePadded], on gpu
    OptVec<TensorConstPtr> logitsVec;     // [forwardBatchSize][beamWidth, vocabSizePadded], on gpu
    TensorConstPtr batchSlots;            // [forwardBatchSize], on pinned

    // optional parameters
    std::optional<TensorPtr>
        srcCacheIndirection;                     // [forwardBatchSize, maxBeamWidth, maxSeqLen], on gpu, for Beam Search
    std::optional<TensorConstPtr> embeddingBias; // [vocabSizePadded], on gpu
    std::optional<TensorConstPtr> inputLengths;  // [maxBatchSize, maxBeamWidth], on gpu
    std::optional<TensorConstPtr> finished;      // [maxBatchSize, maxBeamWidth]
    std::optional<TensorPtr> curTokensPerStep;   // [maxBatchSize], on gpu
    std::shared_ptr<BanWordsDecodingInputs> banWordsInputs;
    std::shared_ptr<StopCriteriaDecodingInputs> stopCriteriaInputs;
    OptVec<runtime::SizeType32> beamSearchSteps; // [forwardBatchSize], for Variable-Beam-Width-Search
};

class SamplingInputs : public DecodingInputs
{
public:
    explicit SamplingInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 step,
        runtime::SizeType32 ite, runtime::SizeType32 localBatchSize)
        : DecodingInputs{std::move(endIds), std::move(batchSlots), step, ite, localBatchSize}
    {
    }

    //! optional parameters
    curandState_t* curandStates{}; // [localBatchSize]

    //! Flag to mark that logits tensor contains probabilities
    bool probsComputed{};
};

class ExternalDraftTokensInputs : public DecodingInputs
{
public:
    explicit ExternalDraftTokensInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 step,
        runtime::SizeType32 ite, runtime::SizeType32 localBatchSize)
        : DecodingInputs{std::move(endIds), std::move(batchSlots), step, ite, localBatchSize}
    {
    }

    TensorPtr draftLogits;
    TensorPtr draftProbs;
    TensorPtr targetProbs;
    TensorPtr numDraftTokens;
    TensorPtr numDraftTokensHost;
    TensorPtr draftTokenIds;
    TensorPtr useDraftLogits;
    TensorPtr useDraftLogitsHost;

    runtime::SizeType32 step{};
    float constantThreshold{};
    bool useRandomAcceptanceThreshold{};

    //! optional parameters
    curandState_t* curandStates{}; // [localBatchSize]

    //! Flag to mark that logits tensor contains probabilities
    bool probsComputed{};
};

// Medusa inputs
class MedusaDecodingInputs : public DecodingInputs
{
public:
    explicit MedusaDecodingInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 localBatchSize)
        : DecodingInputs(std::move(endIds), std::move(batchSlots), 0, 0, localBatchSize)
    {
    }

    TensorConstPtr targetTokensPerStep; // [maxBatchSize], on gpu
    TensorConstPtr paths;               // [maxBatchSize, maxPathLen, maxPathLen], on gpu
    TensorConstPtr treeIds;             // [maxBatchSize, maxDecodingTokens], on gpu

    // [maxBatchSize][maxDraftPathLen][maxDecodingTokens, vocabSizePadded], on gpu
    std::vector<std::vector<TensorPtr>> medusaLogits;
};

// Explicit draft tokens inputs
class ExplicitDraftTokensInputs : public DecodingInputs
{
public:
    explicit ExplicitDraftTokensInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 batchSize)
        : DecodingInputs(std::move(endIds), std::move(batchSlots), 0, 0, batchSize)
    {
    }

    //! Draft tokens for the next iteration. The first token in each path is the last accepted token at current
    //! iteration. E.g. if forwardBatchSize == 1, maxNumPaths == 2, maxPathLen== 3, [[[0, 1, 2], [0, 1, 10]]]
    TensorConstPtr nextDraftTokens; // [forwardBatchSize, maxNumPaths, maxPathLen], gpu
    //! Compressed form of `nextDraftTokens`, where common prefixes and collapsed.
    //! Using example above [0, 1, 2, 10]
    TensorConstPtr nextFlatTokens; // [forwardBatchSize * maxDecodingTokens], gpu
    //! Indices of draft tokens in the compressed `nextFlatTokens` for the next iteration.
    //! Using example above, [[[0, 1, 2], [0, 1, 3]]]
    TensorConstPtr nextDraftIndices; // [forwardBatchSize, maxNumPaths, maxPathLen], gpu
    //! Probabilities of the next draft tokens.
    TensorConstPtr nextDraftProbs; // [forwardBatchSize, maxNumPaths, maxDraftPathLen, vocabSize], gpu
    //! Same as `nextDraftTokens`, but for current iteration.
    //! Current accepted tokens obtained as `lastDraftTokens[bi][bestPathIndices[bi]][1:bestPathLengths[bi]]`.
    TensorConstPtr lastDraftTokens; // [forwardBatchSize, maxNumPaths, maxPathLen], gpu
    //! Same as `nextDraftIndices`, but for current iteration.
    TensorConstPtr lastDraftIndices; // [forwardBatchSize, maxNumPaths, maxPathLen], gpu
    //! Boolean attention masks.
    //! maxDecodingTokens' = generationLengths.max()
    TensorConstPtr masks; // [forwardBatchSize, maxDecodingTokens', maxDecodingTokens'], gpu
    //! Relative to `positionIdsBase` position ids. Same as `nextFlatTokens` for next draft indices.
    //! Using example above, [0, 1, 2, 3]
    TensorConstPtr packedPosIds; // [forwardBatchSize * maxDecodingTokens], gpu
    //! Lengths of the accepted paths for each request. It is 1 for context phase (Only 1 primary tokens is accepted).
    TensorConstPtr bestPathLengths; // [forwardBatchSize], gpu
    //! Indices of the accepted paths for each request. It is 0 for context phase.
    TensorConstPtr bestPathIndices; // [forwardBatchSize], gpu
    //! Number of the draft tokens for the next iteration.
    TensorConstPtr generationLengths; // [forwardBatchSize], gpu
    //! Baseline for the position ids.
    TensorConstPtr positionIdsBase; // [forwardBatchSize], gpu
    //! Generation length for the previous stage.
    TensorConstPtr lastGenerationLengths; // [forwardBatchSize], gpu
    //! Maximum number of generated tokens for the next step across whole batch
    TensorConstPtr maxGenLengthDevice; // [1], on gpu
    //! Address map to map from linear indices of the engine outputs to seqSlot.
    //! It is not the same as batchSlots because it maps the ordered engine outputs to the respective seqSlot,
    //! while batchSlots is just a a list of active seqSlots.
    TensorConstPtr seqSlots; // [forwardBatchSize], on gpu
};

class LookaheadDecodingInputs : public DecodingInputs
{
public:
    explicit LookaheadDecodingInputs(TensorConstPtr endIds, TensorConstPtr batchSlots)
        : DecodingInputs{std::move(endIds), std::move(batchSlots)}
    {
    }
};

// Explicit draft tokens inputs
class EagleInputs : public DecodingInputs
{
public:
    explicit EagleInputs(TensorConstPtr endIds, TensorConstPtr batchSlots, runtime::SizeType32 batchSize,
        TensorConstPtr nextDraftTokens, TensorConstPtr nextDraftLens, TensorConstPtr nextDraftPaths,
        TensorConstPtr lastDraftTokens, TensorConstPtr lastDraftLens, TensorConstPtr lastDraftPaths,
        TensorConstPtr acceptedTokens, TensorConstPtr acceptedLens, TensorConstPtr acceptedPathIds,
        TensorConstPtr chunkedContextNextTokens, TensorConstPtr seqSlots)
        : DecodingInputs(std::move(endIds), std::move(batchSlots), 0, 0, batchSize)
        , nextDraftTokens(nextDraftTokens)
        , nextDraftLens(nextDraftLens)
        , nextDraftPaths(nextDraftPaths)
        , lastDraftTokens(lastDraftTokens)
        , lastDraftLens(lastDraftLens)
        , lastDraftPaths(lastDraftPaths)
        , acceptedTokens(acceptedTokens)
        , acceptedLens(acceptedLens)
        , acceptedPathIds(acceptedPathIds)
        , chunkedContextNextTokens(chunkedContextNextTokens)
        , seqSlots(seqSlots)
    {
    }

    //! Draft tokens for the next iteration.
    TensorConstPtr nextDraftTokens; // [forwardBatchSize, maxDecodingDraftTokens], gpu
    //! Number of the draft tokens for the next iteration.
    TensorConstPtr nextDraftLens; // [forwardBatchSize], gpu
    //! Draft paths for the next iteration.
    TensorConstPtr nextDraftPaths; // [forwardBatchSize, maxDecodingTokens, maxPathLen], gpu
    //! Same as `nextDraftTokens`, but for current iteration.
    TensorConstPtr lastDraftTokens; // [forwardBatchSize, maxNumPaths, maxPathLen], gpu
    //! Number of the draft tokens input to the previous TRT iteration.
    TensorConstPtr lastDraftLens; // [forwardBatchSize], gpu
    //! Same as `nextDraftPaths`, but for current iteration.
    TensorConstPtr lastDraftPaths; // [forwardBatchSize, maxDecodingTokens, maxPathLen], gpu
    //! Lastly accepted tokens (including golden token).
    TensorConstPtr acceptedTokens; // [forwardBatchSize, maxPathLen]
    //! Number of accepted tokens (at least 1).
    TensorConstPtr acceptedLens; // [forwardBatchSize]
    //! Ids of the accepted path.
    TensorConstPtr acceptedPathIds; // [forwardBatchSize]
    //! Indicator whether the context request last chunk or not.
    TensorConstPtr chunkedContextNextTokens; // [forwardBatchSize]
    //!
    TensorConstPtr seqSlots; // [forwardBatchSize], on gpu
};

class BaseDecodingOutputs
{
public:
    explicit BaseDecodingOutputs(TensorPtr outputIds)
        : outputIds{std::move(outputIds)}
    {
    }

    virtual ~BaseDecodingOutputs() = default;

    //! Mandatory parameters
    TensorPtr outputIds; // [maxBatchSize, maxSeqLen]

    //! Optional parameters
    std::optional<TensorPtr> finished;       // [maxBatchSize * maxBeamWidth], on pinned
    std::optional<TensorPtr> sequenceLength; // [maxBatchSize * maxBeamWidth], on gpu
    std::optional<TensorPtr> cumLogProbs;    // [maxBatchSize * maxBeamWidth], on gpu, for Beam Search
    std::optional<TensorPtr> outputLogProbs; // [maxBatchSize, maxBeamWidth, maxSeqLen], on gpu
    std::optional<TensorPtr> parentIds;      // [maxBatchSize, maxBeamWidth, maxSeqLen], on gpu, for Beam Search

    TensorPtr outputIdsPtr;     // [maxBatchSize][maxBeamWidth, maxSeqLen], on gpu and outputIdsPtr[i], on gpu
    TensorPtr outputIdsPtrHost; // [maxBatchSize][maxBeamWidth, maxSeqLen], on cpu but outputIdsPtrHost[i], on gpu
    TensorPtr parentIdsPtr;     // [maxBatchSize][maxBeamWidth, maxSeqLen], on cpu but parentIdsPtr[i], on gpu
    TensorPtr newTokens;        // [maxBatchSize, maxBeamWidth], on gpu, tokens predicted at current iteration.

    // optional parameters
    std::optional<TensorPtr> numNewTokens; // [maxBatchSize], on pinned, number of tokens predicted at current iteration
    std::optional<TensorPtr> finishedSum;  // [1], on pinned
    std::optional<TensorPtr> outputLogProbsTiled; // [maxSeqLen, maxBatchSize, maxBeamWidth], on gpu

    // Beam width might change in Variable-Beam-Width-Search mode.
    // So the beam width is updated in beam search layer for the later layers.
    runtime::SizeType32 beamWidth{1};
};

class BeamSearchOutputs : public BaseDecodingOutputs
{
public:
    explicit BeamSearchOutputs(TensorPtr outputIds)
        : BaseDecodingOutputs{std::move(outputIds)}
    {
    }

    TensorPtr tgtCacheIndirection; //[forwardBatchSize, maxBeamWidth, maxSeqLen], on gpu, the k/v cache index

    std::unique_ptr<kernels::BeamHypotheses> beamHypotheses; // Structure maintains variables of Beam Search
};

//!
//! \brief SpeculativeDecodingOutputs outputs.
//!
//! For one example sequence [a, b] [c] <x, y, z>, where, [a, b, c] is the accepted sequence,
//! [c] is the last accepted token, and <x, y, z> is the draft tokens from `nextDraftTokens` saved by last step.
//! [c]'s position id is known, only position ids for <x, y, z> need to be provided in `nextDraftPosIds`.
//! LLM inputs {c, x, y, z} and generates {c', x', y', z'}.
//!
//! {c'} is always accepted and {x', z'} is supposed to be accepted.
//! The accepted tokens [c', x', z'] is saved in `outputIds` in-place, starting from `sequenceLength`.
//! The `acceptedLength` is 3, and the accepted draft tokens length is 2.
//! `sequenceLength` is also increased by `acceptedLength` in-place.
//! The pathsOffset is {0, 1, 3} for {c', x', z'}.
//! [] for accepted, <> for draft, {} for input/output.
//!
//! For a batchSlots {1, 3}, `numNewTokensCumSum` is an exclusive sum of `numNewTokens` over the batch,
//! the `numNewTokens` may be {3, 5}, `numNewTokensCumSum` is {0, 3, 8}.
//!
//! `nextDraftLengths` and `prevDraftLengths` are needed for methods that support if variable
//! draft length. `nextDraftLengths` must contain the number of draft tokens per request for the next iteration.
//! `prevDraftLengths` must contain the number of draft tokens used in the current iteraiton.
//!
//! `pathsOffsets` is needed for KV cache rewind. It contains the positions of the accepted draft tokens in the
//! flattened tensor of draft tokens. E.g. if for sequence {c, x, y, z} only `y` and `z` were accepted,
//! `pathsOffsets` contains [1, 2]. `pathsOffsets` is flattened tensor for whole batch.
//!
//! The order of `pathsOffsets` and `numNewTokensCumSum` must be aligned. Such that
//! `pathsOffset[numNewTokensCumSum[bi]:numNewTokensCumSum[bi+1]]` is the slice of offsets for `bi`th request.
//! Furthermore, the order of requests is important and must be aligned with sorted `RuntimeBuffers::seqSlots`
//! such that the request with smaller `seqSlot` stays earlier in the tensors.
//! However, this condition usually holds if method does not expect from the engine anything else, but logits.
class SpeculativeDecodingOutputs : public BaseDecodingOutputs
{
public:
    explicit SpeculativeDecodingOutputs(TensorPtr outputIds)
        : BaseDecodingOutputs{std::move(outputIds)}
    {
    }

    //! Draft tokens for the next step
    TensorPtr nextDraftTokens; // [maxBatchSize, maxDecodingDraftTokens]
    //! Draft token position IDs
    TensorPtr nextDraftPosIds; // [maxBatchSize, maxDecodingDraftTokens]
    //! Prev step draft tokens lengths, should be filled only for variable draft length speculative decoding mode
    TensorPtr prevDraftLengths; // [maxBatchSize]
    //! Next step draft tokens lengths, should be filled only for variable draft length speculative decoding mode
    TensorPtr nextDraftLengths; // [maxBatchSize]
    //! Accumulative sum along batchSlots.
    TensorPtr numNewTokensCumSum; // [maxBatchSize + 1]
    TensorPtr pathsOffsets;       // [maxBatchSize * maxPathLen]
    TensorPtr packedMasks;        // [maxBatchSize, maxDecodingTokens, divUp(maxDecodingTokens, 32)]
};

class LookaheadDecodingOutputs : public SpeculativeDecodingOutputs
{
    using TensorPtr = runtime::ITensor::SharedPtr;

public:
    explicit LookaheadDecodingOutputs(TensorPtr outputIds)
        : SpeculativeDecodingOutputs{std::move(outputIds)}
    {
    }

    //! for TLLM engine input "spec_decoding_generation_lengths", indicating how many tokens to be generated.
    //! currently, the 1st step of generation is 1, set at `setup`, others are maxDecodingTokens, set at `forward`.
    TensorPtr generationLengths; //  [maxBatchSize]
    //! for TLLM engine input "spec_decoding_position_offsets",
    //! indicating each token position offset base on the last golden token = 0.
    //! ABC<D>efgxyz--- // sequence tokens, ABCD: golden; efg, xyz: draft; ---: padding.
    //! ***<0>123123--- // positionOffsets.
    //! 012<3>456456--- // positionIds.
    TensorPtr positionOffsets; // [maxBatchSize, maxDecodingTokens]
    TensorPtr positionIds;     // [maxBatchSize, maxDecodingTokens]
};

class ExplicitDraftTokensOutputs : public SpeculativeDecodingOutputs
{
public:
    explicit ExplicitDraftTokensOutputs(TensorPtr outputIds)
        : SpeculativeDecodingOutputs{std::move(outputIds)}
    {
    }

    //! Draft tokens for the next iteration. The first token in each path is the last accepted token at current
    //! iteration. E.g. if batchSize == 1, maxNumPaths == 2, maxPathLen== 3, [[[0, 1, 2], [0, 1, 10]]]
    TensorPtr unpackedNextDraftTokens; // [maxBatchSize, maxNumPaths, maxPathLen], on gpu
    //! Indices of draft tokens in the compressed `nextFlatTokens` for the next iteration.
    //! Using example above, [[[0, 1, 2], [0, 1, 3]]]
    TensorPtr unpackedNextDraftIndices; // [maxBatchSize, maxNumPaths, maxPathLen], on gpu
    //! Probabilities of the next draft tokens.
    TensorPtr nextDraftProbs; // [maxBatchSize, maxNumPaths, maxPathDraftLen, vocabSize], on gpu
    //! Baseline for the position ids.
    TensorPtr positionIdsBase; // [maxBatchSize], on gpu
    //! Randomly sampled data (between 0.f and 1.f)
    TensorPtr randomDataSample; // [maxBatchSize], on gpu
    //! Randomly sampled data (between 0.f and 1.f)
    TensorPtr randomDataValidation; // [maxBatchSize, maxNumPaths, maxDraftPathLen], on gpu
    //! Sampling temperature.
    TensorPtr temperatures; // [maxBatchSize], on gpu
    //! Next generation lengths.
    TensorPtr generationLengths; // [maxBatchSize], on gpu
    //! Next generation lengths on host.
    TensorPtr generationLengthsHost; // [maxBatchSize], on pinned
    //! Maximum number of generated tokens for the next step across whole batch
    TensorPtr maxGenLengthHost; // [1], on pinned
};

class EagleOutputs : public SpeculativeDecodingOutputs
{
public:
    explicit EagleOutputs(TensorPtr outputIds)
        : SpeculativeDecodingOutputs{std::move(outputIds)}
    {
    }

    //! Unpacked draft tokens
    TensorPtr unpackedNextDraftTokens; // [maxBatchSize, maxDecodingDraftTokens], on gpu
    //! Draft paths for the next iteration.
    TensorPtr nextDraftPaths; // [maxBatchSize, maxDecodingTokens, maxPathLen], on gpu
    //! Randomly sampled data (between 0.f and 1.f)
    TensorPtr randomDataSample; // [maxBatchSize], on gpu
    //! Randomly sampled data (between 0.f and 1.f)
    TensorPtr randomDataValidation; // [maxBatchSize], on gpu
    //! Sampling temperature.
    TensorPtr temperatures; // [maxBatchSize], on gpu
    //! Next generation lengths.
    TensorPtr generationLengths; // [maxBatchSize], on gpu
    //! Next generation lengths.
    TensorPtr generationLengthsHost; // [maxBatchSize], on pinned
    //! Request types for ctx stage of the EagleNet0 (filled with 0s).
    TensorPtr eagleNetCtxRequestTypesHost; // [maxBatchSize], on pinned
    //! Context lengths of the context EagleNet0.
    TensorPtr eagleNetCtxContextLengthsHost; // [maxBatchSize], on pinned
    //! Past kv lengths of the context EagleNet0.
    TensorPtr eagleNetCtxPastKeyValueLengthsHost; // [maxBatchSize], on pinned
    //! Request types for ctx stage of the EagleNetX (filled with 1s).
    TensorPtr eagleNetGenRequestTypesHost; // [maxBatchSize], on pinned
    //! Context lengths of the generation EagleNetX.
    TensorPtr eagleNetGenContextLengthsHost; // [maxBatchSize], on pinned
    //! Past kv lengths of the generation EagleNetX.
    TensorPtr eagleNetGenPastKeyValueLengthsHost; // [maxBatchSize], on pinned
};

} // namespace tensorrt_llm::layers
