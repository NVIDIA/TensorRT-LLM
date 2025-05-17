// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/executor/types.h"
#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <optional>

#include "utils.h"

using namespace triton::backend::inflight_batcher_llm;
using namespace triton::backend::inflight_batcher_llm::utils;

TEST(UtilsTest, csvStrToVecInt)
{
    {
        std::string str = "0";
        auto out = csvStrToVecInt(str);
        EXPECT_THAT(out, testing::ElementsAre(0));
    }
    {
        std::string str = "0, 1, 2";
        auto out = csvStrToVecInt(str);
        EXPECT_THAT(out, testing::ElementsAre(0, 1, 2));
    }
    {
        std::string str = "0,1,2";
        auto out = csvStrToVecInt(str);
        EXPECT_THAT(out, testing::ElementsAre(0, 1, 2));
    }
    // Trailing comma is ok
    {
        std::string str = "0, 1, 2, ";
        auto out = csvStrToVecInt(str);
        EXPECT_THAT(out, testing::ElementsAre(0, 1, 2));
    }
    {
        std::vector<std::string> invalidStr{"a", "bbdfdsa", ",,", ","};
        for (auto const& str : invalidStr)
        {
            try
            {
                auto out = csvStrToVecInt(str);
                for (auto val : out)
                {
                    std::cout << val << std::endl;
                }
                FAIL() << "Expected exception for input: " << str;
            }
            catch (std::invalid_argument const& e)
            {
                EXPECT_THAT(e.what(), testing::HasSubstr("stoi"));
            }
            catch (tensorrt_llm::common::TllmException const& e)
            {
                EXPECT_THAT(
                    e.what(), testing::AnyOf(testing::HasSubstr("empty string"), testing::HasSubstr("Empty vector")));
            }
            catch (std::exception const& e)
            {
                FAIL() << "Expected invalid argument of input: " << str;
            }
        }
    }
}

TEST(UtilsTest, csvStrToVecVecInt)
{
    {
        std::string str = "{0}";
        auto out = csvStrToVecVecInt(str);
        EXPECT_THAT(out.size(), 1);
        EXPECT_THAT(out.at(0), testing::ElementsAre(0));
    }
    {
        std::vector<std::string> strs = {"{1, 5}, {1, 2, 3}", "{1,5},{1,2,3}", "{1, 5}, {1,2,3,}, "};
        for (auto str : strs)
        {
            auto out = csvStrToVecVecInt(str);
            EXPECT_THAT(out.size(), 2);
            EXPECT_THAT(out.at(0), testing::ElementsAre(1, 5));
            EXPECT_THAT(out.at(1), testing::ElementsAre(1, 2, 3));
        }
    }

    {
        std::vector<std::string> invalidStr{"a", "0,1,", "{a}", "{bdfda,,}", "", "", "{{}}", "{}"};
        for (auto const& str : invalidStr)
        {
            try
            {
                auto out = csvStrToVecVecInt(str);
                FAIL() << "Expected exception for input: " << str;
            }
            catch (std::invalid_argument const& e)
            {
                EXPECT_THAT(e.what(), testing::HasSubstr("stoi"));
            }
            catch (tensorrt_llm::common::TllmException const& e)
            {
                EXPECT_THAT(
                    e.what(), testing::AnyOf(testing::HasSubstr("empty string"), testing::HasSubstr("Empty vector")));
            }
            catch (std::exception const& e)
            {
                FAIL() << "Expected invalid argument of input: " << str << " got " << e.what();
            }
        }
    }
}

template <typename Value>
void pushTensor(std::unordered_map<std::string, NamedTensor>& inputsTensors, std::string name, nvinfer1::DataType type,
    std::vector<int64_t> shape, std::vector<Value> data)
{
    NamedTensor tensor(type, shape, name, data.data());
    inputsTensors.insert(make_pair(name, std::move(tensor)));
}

TEST(UtilsTest, extractSingleton)
{
    std::unordered_map<std::string, NamedTensor> inputsTensors;
    pushTensor<int32_t>(inputsTensors, "int32", nvinfer1::DataType::kINT32, {1}, {2});
    pushTensor<int64_t>(inputsTensors, "int64", nvinfer1::DataType::kINT64, {1, 1}, {4294967296ll});
    pushTensor<float>(inputsTensors, "float32", nvinfer1::DataType::kFLOAT, {1, 2}, {0.5, 0.6});

    // extractSingleton
    {
        int32_t int32Value = 0;
        EXPECT_THAT(extractSingleton(inputsTensors, "int32", int32Value), true);
        EXPECT_THAT(int32Value, 2);
    }
    {
        int64_t int64Value = 0;
        EXPECT_THAT(extractSingleton(inputsTensors, "int64_typo", int64Value), false);
        EXPECT_THAT(extractSingleton(inputsTensors, "int64", int64Value), true);
        EXPECT_THAT(int64Value, 4294967296ll);
    }
    {
        float floatValue = 0;
        try
        {
            extractSingleton(inputsTensors, "float32", floatValue);
            FAIL() << "Expected exception";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("Invalid size"));
        }
    }

    // extractOptionalSingleton
    {
        std::optional<int64_t> int64Value;
        extractOptionalSingleton(inputsTensors, "int64_typo", int64Value);
        EXPECT_THAT(int64Value.has_value(), false);
        extractOptionalSingleton(inputsTensors, "int64", int64Value);
        EXPECT_THAT(int64Value.has_value(), true);
        EXPECT_THAT(int64Value.value(), 4294967296ll);
    }

    // extractVector
    {
        std::vector<float> float32Values;
        EXPECT_THAT(extractVector(inputsTensors, "float32", float32Values), true);
        EXPECT_THAT(float32Values, testing::ElementsAre(0.5, 0.6));
    }
}

TEST(UtilsTest, flatten)
{
    // single vector pass
    {
        std::vector<int32_t> original{1, 2, 3};
        std::vector<int32_t> out(3, -1);
        flatten(original, out.data(), {3});
        EXPECT_THAT(out, testing::ElementsAre(1, 2, 3));
    }
    // single vector fail
    {
        std::vector<int32_t> original{1, 2, 3};
        std::vector<int32_t> out(3, -1);
        try
        {
            flatten(original, out.data(), {4});
            FAIL() << "Expected exception for mismatched shape";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("unexpected size"));
        }
    }
    // vector of vector pass
    {
        std::vector<std::vector<int32_t>> original{
            {1, 1},
            {2, 3},
            {5, 8},
        };
        std::vector<int32_t> out(6, -1);
        flatten(original, out.data(), {6});
        EXPECT_THAT(out, testing::ElementsAre(1, 1, 2, 3, 5, 8));
    }
    // vector of vector fail
    {
        std::vector<std::vector<int32_t>> original{
            {1, 1},
            {2, 3},
            {5, 8, 8},
        };
        std::vector<int32_t> out(7, -1);
        try
        {
            flatten(original, out.data(), {7});
            FAIL() << "Expected exception for mismatched inner vectors";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("mismatched sizes"));
        }
    }
    // executor tensor pass
    {
        std::vector<int32_t> tensorUnderlying{6, 7, 8};
        auto original = executor::Tensor::of(tensorUnderlying.data(), {3});
        std::vector<int32_t> out(3, -1);
        flatten<int32_t>(original, out.data(), {3});
        EXPECT_THAT(out, testing::ElementsAre(6, 7, 8));
    }
    // executor tensor fail
    {
        std::vector<int32_t> tensorUnderlying{6, 7, 8};
        auto original = executor::Tensor::of(tensorUnderlying.data(), {3});
        std::vector<int32_t> out(3, -1);
        try
        {
            flatten<int32_t>(original, out.data(), {4});
            FAIL() << "Expected exception for mismatched inner vectors";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("unexpected size"));
        }
    }
}

TEST(UtilsTest, convertWordList)
{
    // fail
    {
        executor::VecTokens before{1, 2, 3};
        std::list<executor::VecTokens> after;
        try
        {
            after = convertWordList(before);
            FAIL() << "Expected exception";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("odd length"));
        }
    }
    // fail
    {
        executor::VecTokens before{1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 9, -1, -1, -1, -1, 666, -1};
        std::list<executor::VecTokens> after;
        try
        {
            after = convertWordList(before);
            FAIL() << "Expected exception";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("additional -1s"));
        }
    }
    // fail
    {
        executor::VecTokens before{1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 10, -1, -1, -1, -1, -1, -1};
        std::list<executor::VecTokens> after;
        try
        {
            after = convertWordList(before);
            FAIL() << "Expected exception";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("out-of-bound offsets"));
        }
    }
    // fail
    {
        executor::VecTokens before{1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 2, -1, -1, -1, -1, -1, -1};
        std::list<executor::VecTokens> after;
        try
        {
            after = convertWordList(before);
            FAIL() << "Expected exception";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("non-increasing offsets"));
        }
    }
    // pass
    {
        executor::VecTokens before{1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 9, -1, -1, -1, -1, -1, -1};
        std::list<executor::VecTokens> after;
        after = convertWordList(before);
        EXPECT_THAT(after.size(), 3);
        auto iter = after.begin();
        EXPECT_THAT(*(iter++), testing::ElementsAre(1, 2));
        EXPECT_THAT(*(iter++), testing::ElementsAre(3, 4));
        EXPECT_THAT(*(iter++), testing::ElementsAre(5, 6, 7, 8, 9));
    }
}

std::optional<tensorrt_llm::executor::Request> getRequest(
    executor::ModelType modelType = executor::ModelType::kDECODER_ONLY, std::optional<int32_t> padId = std::nullopt,
    std::vector<int32_t> const& inputTokens = {1, 2, 3, 4, 5}, std::vector<int32_t> const& decoderInputTokens = {})
{
    std::unordered_map<std::string, NamedTensor> inputsTensors;

    pushTensor<int32_t>(inputsTensors, InputFieldsNames::inputTokens, nvinfer1::DataType::kINT32,
        {static_cast<int64_t>(inputTokens.size())}, inputTokens);
    if (!decoderInputTokens.empty())
    {
        pushTensor<int32_t>(inputsTensors, InputFieldsNames::decoderInputTokens, nvinfer1::DataType::kINT32,
            {static_cast<int64_t>(decoderInputTokens.size())}, decoderInputTokens);
    }
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::maxNewTokens, nvinfer1::DataType::kINT32, {1}, {8});
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::numReturnSequences, nvinfer1::DataType::kINT32, {1}, {29});
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::endId, nvinfer1::DataType::kINT32, {1}, {11});
    if (padId)
    {
        pushTensor<int32_t>(inputsTensors, InputFieldsNames::padId, nvinfer1::DataType::kINT32, {1}, {padId.value()});
    }
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::badWords, nvinfer1::DataType::kINT32, {6}, {1, 2, 3, 2, 3, -1});
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::stopWords, nvinfer1::DataType::kINT32, {6}, {1, 2, 3, 3, -1, -1});
    pushTensor<float>(
        inputsTensors, InputFieldsNames::embeddingBias, nvinfer1::DataType::kFLOAT, {1, 3}, {0.5, 0.6, 0.7});

    // OutputConfig
    using MyBool = char; // prevent std::vector<bool>
    pushTensor<MyBool>(inputsTensors, InputFieldsNames::returnLogProbs, nvinfer1::DataType::kBOOL, {1}, {true});
    pushTensor<MyBool>(inputsTensors, InputFieldsNames::returnGenerationLogits, nvinfer1::DataType::kBOOL, {1}, {true});
    pushTensor<MyBool>(inputsTensors, InputFieldsNames::returnContextLogits, nvinfer1::DataType::kBOOL, {1}, {true});

    // SamplingConfig
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::beamWidth, nvinfer1::DataType::kINT32, {1}, {1023});
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::topK, nvinfer1::DataType::kINT32, {1}, {25});
    pushTensor<float>(inputsTensors, InputFieldsNames::topP, nvinfer1::DataType::kFLOAT, {1}, {0.7});
    pushTensor<float>(inputsTensors, InputFieldsNames::topPMin, nvinfer1::DataType::kFLOAT, {1}, {0.8});
    pushTensor<float>(inputsTensors, InputFieldsNames::topPDecay, nvinfer1::DataType::kFLOAT, {1}, {0.9});
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::topPResetIds, nvinfer1::DataType::kINT32, {1}, {25});
    pushTensor<float>(inputsTensors, InputFieldsNames::temperature, nvinfer1::DataType::kFLOAT, {1}, {0.3});
    pushTensor<float>(inputsTensors, InputFieldsNames::lengthPenalty, nvinfer1::DataType::kFLOAT, {1}, {0.4});
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::earlyStopping, nvinfer1::DataType::kINT32, {1}, {4});
    pushTensor<float>(inputsTensors, InputFieldsNames::repetitionPenalty, nvinfer1::DataType::kFLOAT, {1}, {0.8});
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::minTokens, nvinfer1::DataType::kINT32, {1}, {45});
    pushTensor<float>(inputsTensors, InputFieldsNames::beamSearchDiversityRate, nvinfer1::DataType::kFLOAT, {1}, {0.1});
    pushTensor<float>(inputsTensors, InputFieldsNames::presencePenalty, nvinfer1::DataType::kFLOAT, {1}, {0.2});
    pushTensor<float>(inputsTensors, InputFieldsNames::frequencyPenalty, nvinfer1::DataType::kFLOAT, {1}, {0.3});
    pushTensor<uint64_t>(inputsTensors, InputFieldsNames::seed, nvinfer1::DataType::kINT64, {1}, {3456});

    // PromptTuningConfig
    pushTensor<float>(inputsTensors, InputFieldsNames::promptEmbeddingTable, nvinfer1::DataType::kFLOAT, {1, 2, 2},
        {0.5, 0.6, 0.7, 0.8});

    // LoraConfig
    pushTensor<uint64_t>(inputsTensors, InputFieldsNames::loraTaskId, nvinfer1::DataType::kINT64, {1}, {87654});
    pushTensor<float>(inputsTensors, InputFieldsNames::loraWeights, nvinfer1::DataType::kFLOAT, {1, 3, 2},
        {0.5, 0.6, 0.7, 0.8, 0.1, 0.1});
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::loraConfig, nvinfer1::DataType::kINT32, {1, 3, 2}, {1, 1, 2, 3, 5, 8});

    // KvCacheRetentionConfig
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::retentionTokenRangeStarts, nvinfer1::DataType::kINT32, {3}, {0, 100, 200});
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::retentionTokenRangeEnds, nvinfer1::DataType::kINT32, {3}, {100, 200, 300});
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::retentionTokenRangePriorities, nvinfer1::DataType::kINT32, {3}, {70, 50, 20});
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::retentionTokenRangeDurations, nvinfer1::DataType::kINT32, {3},
        {3000, 2000, 1000});
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::retentionDecodePriority, nvinfer1::DataType::kINT32, {1}, {10});
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::retentionDecodeDuration, nvinfer1::DataType::kINT32, {1}, {500});

    // ExternalDraftTokensConfig
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::draftInputs, nvinfer1::DataType::kINT32, {4}, {1, 2, 3, 3});
    pushTensor<float>(
        inputsTensors, InputFieldsNames::draftLogits, nvinfer1::DataType::kFLOAT, {1, 4, 1}, {1.1, 2.1, 3.1, 3.1});
    pushTensor<float>(
        inputsTensors, InputFieldsNames::draftAcceptanceThreshold, nvinfer1::DataType::kFLOAT, {1}, {0.222F});

    // getGuidedDecodingParams
    // NOTE
    // guideType: 'regex', guide: '\d+'
    // Add 4 dummy 0(null)'s to mimic Triton StringType input.
    pushTensor<int8_t>(inputsTensors, InputFieldsNames::guidedDecodingGuideType, nvinfer1::DataType::kINT8, {1, 9},
        {0, 0, 0, 0, 114, 101, 103, 101, 120});
    pushTensor<int8_t>(inputsTensors, InputFieldsNames::guidedDecodingGuide, nvinfer1::DataType::kINT8, {1, 7},
        {0, 0, 0, 0, 92, 100, 43});

    // Request LookaheadDecodingConfig
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::requestLookaheadDecodingWindowSize, nvinfer1::DataType::kINT32, {1}, {2});
    pushTensor<int32_t>(
        inputsTensors, InputFieldsNames::requestLookaheadDecodingNgramSize, nvinfer1::DataType::kINT32, {1}, {2});
    pushTensor<int32_t>(inputsTensors, InputFieldsNames::requestLookaheadDecodingVerificationSetSize,
        nvinfer1::DataType::kINT32, {1}, {1});

    std::optional<executor::Request> request;
    try
    {
        request = createRequestsFromInputTensors({inputsTensors}, /*paramExcludeInputFromOutput*/ true,
            /*isDecoupled*/ true, /*streaming*/ true, modelType,
            executor::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION, false /* isOrchestrator */,
            false /* specDecFastLogits */, executor::LookaheadDecodingConfig{2, 2, 1})
                      .at(0);
        if (modelType == executor::ModelType::kENCODER_DECODER && decoderInputTokens.empty() && !padId)
        {
            EXPECT_TRUE(false) << "Should have failed";
        }
        return request;
    }
    catch (std::exception const& e)
    {
        if (modelType == executor::ModelType::kENCODER_DECODER && decoderInputTokens.empty() && !padId)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("Assertion failed: !mInputTokenIds.empty"));
            return std::nullopt;
        }
        else
        {
            EXPECT_TRUE(false) << "Should not fail but failed with: " << e.what();
        }
    }
    return request;
}

void checkWords(
    std::list<tensorrt_llm::executor::VecTokens> const& words, std::vector<std::vector<int32_t>> const& reference)
{
    EXPECT_EQ(words.size(), reference.size());
    auto iter1 = words.begin();
    auto iter2 = reference.begin();
    for (int32_t i = 0; i < words.size(); ++i)
    {
        EXPECT_EQ(*(iter1++), *(iter2++));
    }
}

template <class T>
void checkTensor(tensorrt_llm::executor::Tensor const& tensor, std::vector<T> reference)
{
    EXPECT_EQ(tensor.getSizeInBytes(), reference.size() * sizeof(T));
    for (int i = 0; i < reference.size(); ++i)
    {
        EXPECT_EQ(reference[i], static_cast<T const*>(tensor.getData())[i]);
    }
}

void checkRequest(tensorrt_llm::executor::Request const& request,
    executor::ModelType modelType = executor::ModelType::kDECODER_ONLY, std::optional<int32_t> padId = std::nullopt,
    std::vector<int32_t> const& inputTokens = {1, 2, 3, 4, 5}, std::vector<int32_t> const& decoderInputTokens = {})
{
    if (modelType == executor::ModelType::kENCODER_ONLY)
    {
        EXPECT_EQ(request.getEncoderInputTokenIds(), inputTokens);
    }
    else if (modelType == executor::ModelType::kENCODER_DECODER)
    {
        EXPECT_EQ(request.getEncoderInputTokenIds(), inputTokens);
        if (!decoderInputTokens.empty())
        {
            EXPECT_EQ(request.getInputTokenIds(), decoderInputTokens);
        }
        else
        {
            if (padId)
            {
                EXPECT_EQ(request.getInputTokenIds(), std::vector<int32_t>{padId.value()});
            }
            else
            {
                FAIL() << "Should have failed already";
            }
        }
    }
    EXPECT_EQ(request.getMaxTokens(), 8);
    EXPECT_EQ(request.getEndId().value(), 11);
    if (padId)
    {
        EXPECT_EQ(request.getPadId().value(), padId.value());
    }
    checkWords(request.getBadWords().value(), {{1, 2}, {3}});
    checkWords(request.getStopWords().value(), {{1, 2, 3}});
    checkTensor<float>(request.getEmbeddingBias().value(), {0.5, 0.6, 0.7});

    // OutputConfig
    auto outputConfig = request.getOutputConfig();
    EXPECT_TRUE(outputConfig.returnLogProbs);
    EXPECT_TRUE(outputConfig.returnGenerationLogits);
    EXPECT_TRUE(outputConfig.returnContextLogits);
    EXPECT_TRUE(outputConfig.excludeInputFromOutput);

    // ExternalDraftTokensConfig
    auto externalDraftTokensConfig = request.getExternalDraftTokensConfig().value();
    EXPECT_THAT(externalDraftTokensConfig.getTokens(), testing::ElementsAre(1, 2, 3, 3));
    checkTensor<float>(externalDraftTokensConfig.getLogits().value(), {1.1, 2.1, 3.1, 3.1});
    EXPECT_TRUE(externalDraftTokensConfig.getAcceptanceThreshold().has_value());
    EXPECT_FLOAT_EQ(externalDraftTokensConfig.getAcceptanceThreshold().value(), 0.222F);

    // PromptTuningConfig
    auto promptTuningConfig = request.getPromptTuningConfig().value();
    checkTensor<float>(promptTuningConfig.getEmbeddingTable(), {0.5, 0.6, 0.7, 0.8});

    // LoraConfig
    auto loraConfig = request.getLoraConfig().value();
    EXPECT_EQ(loraConfig.getTaskId(), 87654);
    checkTensor<float>(loraConfig.getWeights().value(), {0.5, 0.6, 0.7, 0.8, 0.1, 0.1});
    checkTensor<int32_t>(loraConfig.getConfig().value(), {1, 1, 2, 3, 5, 8});

    // KvCacheRetentionConfig
    auto kvCacheRetentionConfig = request.getKvCacheRetentionConfig().value();

    EXPECT_THAT(kvCacheRetentionConfig.getTokenRangeRetentionConfigs(),
        testing::ContainerEq(std::vector<executor::KvCacheRetentionConfig::TokenRangeRetentionConfig>{
            executor::KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 100, 70, std::chrono::milliseconds(3000)),
            executor::KvCacheRetentionConfig::TokenRangeRetentionConfig(100, 200, 50, std::chrono::milliseconds(2000)),
            executor::KvCacheRetentionConfig::TokenRangeRetentionConfig(
                200, 300, 20, std::chrono::milliseconds(1000))}));

    EXPECT_EQ(kvCacheRetentionConfig.getDecodeRetentionPriority(), 10);
    EXPECT_EQ(kvCacheRetentionConfig.getDecodeDurationMs(), std::chrono::milliseconds(500));

    auto requestLookaheadConfig = request.getLookaheadConfig().value();
    EXPECT_THAT(requestLookaheadConfig.getWindowSize(), 2);
    EXPECT_THAT(requestLookaheadConfig.getNgramSize(), 2);
    EXPECT_THAT(requestLookaheadConfig.getVerificationSetSize(), 1);

    // SamplingConfig
    auto samplingConfig = request.getSamplingConfig();
    EXPECT_EQ(samplingConfig.getBeamWidth(), 1023);
    EXPECT_EQ(samplingConfig.getTopK().value(), 25);
    EXPECT_EQ(samplingConfig.getTopP().value(), 0.7f);
    EXPECT_EQ(samplingConfig.getTopPMin().value(), 0.8f);
    EXPECT_EQ(samplingConfig.getTopPDecay().value(), 0.9f);
    EXPECT_EQ(samplingConfig.getTopPResetIds().value(), 25);
    EXPECT_EQ(samplingConfig.getTemperature().value(), 0.3f);
    EXPECT_EQ(samplingConfig.getLengthPenalty().value(), 0.4f);
    EXPECT_EQ(samplingConfig.getEarlyStopping().value(), 4);
    EXPECT_EQ(samplingConfig.getRepetitionPenalty().value(), 0.8f);
    EXPECT_EQ(samplingConfig.getMinTokens().value(), 45);
    EXPECT_EQ(samplingConfig.getBeamSearchDiversityRate().value(), 0.1f);
    EXPECT_EQ(samplingConfig.getPresencePenalty().value(), 0.2f);
    EXPECT_EQ(samplingConfig.getFrequencyPenalty().value(), 0.3f);
    EXPECT_EQ(samplingConfig.getSeed().value(), 3456);
    EXPECT_EQ(samplingConfig.getNumReturnSequences().value(), 29);

    // getGuidedDecodingParams
    auto guidedDecodingParams = request.getGuidedDecodingParams().value();
    EXPECT_EQ(guidedDecodingParams.getGuideType(), executor::GuidedDecodingParams::GuideType::kREGEX);
    EXPECT_EQ(guidedDecodingParams.getGuide().value(), "\\d+");
}

TEST(UtilsTest, createRequestFromInputTensors)
{
    auto request = getRequest(executor::ModelType::kDECODER_ONLY, 2);
    checkRequest(request.value(), executor::ModelType::kDECODER_ONLY, 2);
}

TEST(UtilsTest, createRequestFromInputTensorsEncoderOnly)
{
    auto modelType = executor::ModelType::kENCODER_ONLY;
    int32_t padId = 2;
    auto request = getRequest(modelType, padId);
    checkRequest(request.value(), modelType, padId);
}

TEST(UtilsTest, createRequestFromInputTensorsEncoderDecoder)
{
    auto modelType = executor::ModelType::kENCODER_DECODER;
    auto inputTokensVec = std::vector<std::vector<int32_t>>{{1, 2, 3, 4, 5}};
    auto decoderInputTokensVec = std::vector<std::vector<int32_t>>{{1, 2, 3, 4, 5}, {}};
    auto padIds = std::vector<std::optional<int32_t>>{2, std::nullopt};

    for (auto const& inputTokens : inputTokensVec)
    {
        for (auto const& decoderInputTokens : decoderInputTokensVec)
        {
            for (auto padId : padIds)
            {
                auto request = getRequest(modelType, padId, inputTokens, decoderInputTokens);
                // If not request returned, means we were expecting failure during request creation
                // Just continue in this case.
                if (!request)
                {
                    continue;
                }
                else
                {
                    checkRequest(request.value(), modelType, padId, inputTokens, decoderInputTokens);
                }
            }
        }
    }
}

TEST(UtilsTest, splitBatchInputsTensorsBS1)
{
    // Batch size 1
    auto modelType = executor::ModelType::kDECODER_ONLY;
    {
        std::vector<int32_t> const& inputTokens = {1, 2, 3, 4, 5};
        InputTensors inputTensors;
        pushTensor<int32_t>(inputTensors, InputFieldsNames::inputTokens, nvinfer1::DataType::kINT32,
            {1, static_cast<int64_t>(inputTokens.size())}, inputTokens);
        pushTensor<int32_t>(inputTensors, InputFieldsNames::maxNewTokens, nvinfer1::DataType::kINT32, {1}, {8});

        auto inputsTensors = splitBatchInputsTensors(inputTensors);

        // No length specified should work ok with batch size 1
        {
            EXPECT_EQ(inputsTensors.size(), 1);
            for (int batchId = 0; batchId < inputsTensors.size(); batchId++)
            {
                auto& inputTensors = inputsTensors.at(batchId);
                EXPECT_EQ(inputTensors.size(), 2);
                EXPECT_EQ(inputTensors.count(InputFieldsNames::inputTokens), 1);
                auto inputTokensTensor = inputTensors.at(InputFieldsNames::inputTokens).tensor;
                EXPECT_EQ(inputTokensTensor->getShape().nbDims, 2);
                EXPECT_EQ(inputTokensTensor->getShape().d[0], 1);
                EXPECT_EQ(inputTokensTensor->getShape().d[1], inputTokens.size());
                auto inputTokensData = reinterpret_cast<int32_t*>(inputTokensTensor->data());
                EXPECT_EQ(inputTokensData[0], 1);
                EXPECT_EQ(inputTokensData[1], 2);
                EXPECT_EQ(inputTokensData[2], 3);
                EXPECT_EQ(inputTokensData[3], 4);
                EXPECT_EQ(inputTokensData[4], 5);

                EXPECT_EQ(inputTensors.count(InputFieldsNames::maxNewTokens), 1);
                auto maxTokensTensor = inputTensors.at(InputFieldsNames::maxNewTokens).tensor;
                EXPECT_EQ(maxTokensTensor->getShape().nbDims, 1);
                EXPECT_EQ(maxTokensTensor->getShape().d[0], 1);
                auto maxTokensData = reinterpret_cast<int32_t*>(maxTokensTensor->data());
                EXPECT_EQ(maxTokensData[0], 8);
            }
        }

        // Create requests from batch size 1 tensor
        {
            auto requests = createRequestsFromInputTensors(inputsTensors, true, true, true, modelType,
                executor::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION, false /* isOrchestrator */,
                false /* specDecFastLogits */, std::nullopt);

            EXPECT_EQ(requests.size(), 1);
            EXPECT_EQ(requests.at(0).getInputTokenIds(), inputTokens);
            EXPECT_EQ(requests.at(0).getMaxTokens(), 8);
        }
    }
}

TEST(UtilsTest, splitBatchInputsTensorsBS3)
{
    auto modelType = executor::ModelType::kDECODER_ONLY;
    // Batch size 3
    std::vector<int32_t> const& inputTokens = {1, 2, -1, 3, 4, 5, 6, -1, -1};
    std::vector<int32_t> const& inputLengths = {2, 3, 1};
    std::vector<int32_t> const& maxTokens = {8, 10, 11};
    InputTensors inputTensors;
    pushTensor<int32_t>(inputTensors, InputFieldsNames::inputTokens, nvinfer1::DataType::kINT32, {3, 3}, inputTokens);
    pushTensor<int32_t>(inputTensors, InputFieldsNames::maxNewTokens, nvinfer1::DataType::kINT32, {3}, maxTokens);

    {
        // W/o lengths, it should throw an error
        try
        {
            auto inputsTensors = splitBatchInputsTensors(inputTensors);
            FAIL() << "Expected exception";
        }
        catch (tensorrt_llm::common::TllmException const& e)
        {
            EXPECT_THAT(e.what(), testing::HasSubstr("input lengths tensor not provided"));
        }
    }

    pushTensor<int32_t>(inputTensors, InputFieldsNames::inputLengths, nvinfer1::DataType::kINT32, {3}, inputLengths);
    auto inputsTensors = splitBatchInputsTensors(inputTensors);

    EXPECT_EQ(inputsTensors.size(), 3);
    for (int batchId = 0; batchId < inputsTensors.size(); batchId++)
    {
        auto& inputTensors = inputsTensors.at(batchId);
        EXPECT_EQ(inputTensors.size(), 3);
        EXPECT_EQ(inputTensors.count(InputFieldsNames::inputTokens), 1);
        auto inputTokensTensor = inputTensors.at(InputFieldsNames::inputTokens).tensor;
        EXPECT_EQ(inputTokensTensor->getShape().nbDims, 2);
        EXPECT_EQ(inputTokensTensor->getShape().d[0], 1);
        EXPECT_EQ(inputTokensTensor->getShape().d[1], inputLengths[batchId]);
        auto inputTokensData = reinterpret_cast<int32_t*>(inputTokensTensor->data());
        EXPECT_EQ(inputTokensData[0], inputTokens[batchId * 3 + 0]);
        EXPECT_EQ(inputTokensData[1], inputTokens[batchId * 3 + 1]);
        EXPECT_EQ(inputTokensData[2], inputTokens[batchId * 3 + 2]);

        EXPECT_EQ(inputTensors.count(InputFieldsNames::maxNewTokens), 1);
        auto maxTokensTensor = inputTensors.at(InputFieldsNames::maxNewTokens).tensor;
        EXPECT_EQ(maxTokensTensor->getShape().nbDims, 1);
        EXPECT_EQ(maxTokensTensor->getShape().d[0], 1);
        auto maxTokensData = reinterpret_cast<int32_t*>(maxTokensTensor->data());
        EXPECT_EQ(maxTokensData[0], maxTokens[batchId]);

        EXPECT_EQ(inputTensors.count(InputFieldsNames::inputLengths), 1);
        auto inputLengthsTensor = inputTensors.at(InputFieldsNames::inputLengths).tensor;
        EXPECT_EQ(maxTokensTensor->getShape().nbDims, 1);
        EXPECT_EQ(maxTokensTensor->getShape().d[0], 1);
        auto inputLengthsData = reinterpret_cast<int32_t*>(inputLengthsTensor->data());
        EXPECT_EQ(inputLengthsData[0], inputLengths[batchId]);
    }

    // Get the requests
    {
        auto requests = createRequestsFromInputTensors(inputsTensors, true, true, true, modelType,
            executor::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION, false /* isOrchestrator */,
            false /* specDecFastLogits */, std::nullopt);
        EXPECT_EQ(requests.size(), 3);
        for (int batchId = 0; batchId < inputsTensors.size(); batchId++)
        {
            auto const& request = requests[batchId];
            EXPECT_EQ(request.getInputTokenIds(),
                std::vector<int32_t>(
                    inputTokens.begin() + batchId * 3, inputTokens.begin() + batchId * 3 + inputLengths[batchId]));
            EXPECT_EQ(request.getMaxTokens(), maxTokens[batchId]);
        }
    }
}

TEST(UtilsTest, split)
{
    std::string str = "a b c";
    std::vector<std::string> expected = {"a", "b", "c"};
    EXPECT_EQ(split(str, ' '), expected);

    str = "a b c ";
    expected = {"a", "b", "c"};
    EXPECT_EQ(split(str, ' '), expected);

    str = "";
    expected = {};
    EXPECT_EQ(split(str, ' '), expected);
}
