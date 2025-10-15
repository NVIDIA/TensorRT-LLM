/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/executor/serializeUtils.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <gtest/gtest.h>

#include <optional>
#include <type_traits>
#include <variant>

namespace su = tensorrt_llm::executor::serialize_utils;
namespace texec = tensorrt_llm::executor;
using VecTokens = texec::VecTokens;

void compareRequestPerfMetrics(texec::RequestPerfMetrics const& lh, texec::RequestPerfMetrics const& rh)
{
    EXPECT_EQ(lh.timingMetrics.arrivalTime, rh.timingMetrics.arrivalTime);
    EXPECT_EQ(lh.timingMetrics.firstScheduledTime, rh.timingMetrics.firstScheduledTime);
    EXPECT_EQ(lh.timingMetrics.firstTokenTime, rh.timingMetrics.firstTokenTime);
    EXPECT_EQ(lh.timingMetrics.lastTokenTime, rh.timingMetrics.lastTokenTime);
    EXPECT_EQ(lh.timingMetrics.kvCacheTransferStart, rh.timingMetrics.kvCacheTransferStart);
    EXPECT_EQ(lh.timingMetrics.kvCacheTransferEnd, rh.timingMetrics.kvCacheTransferEnd);

    EXPECT_EQ(lh.kvCacheMetrics.numTotalAllocatedBlocks, rh.kvCacheMetrics.numTotalAllocatedBlocks);
    EXPECT_EQ(lh.kvCacheMetrics.numNewAllocatedBlocks, rh.kvCacheMetrics.numNewAllocatedBlocks);
    EXPECT_EQ(lh.kvCacheMetrics.numReusedBlocks, rh.kvCacheMetrics.numReusedBlocks);
    EXPECT_EQ(lh.kvCacheMetrics.numMissedBlocks, rh.kvCacheMetrics.numMissedBlocks);
    EXPECT_EQ(lh.kvCacheMetrics.kvCacheHitRate, rh.kvCacheMetrics.kvCacheHitRate);

    EXPECT_EQ(lh.speculativeDecoding.acceptanceRate, rh.speculativeDecoding.acceptanceRate);
    EXPECT_EQ(lh.speculativeDecoding.totalAcceptedDraftTokens, rh.speculativeDecoding.totalAcceptedDraftTokens);
    EXPECT_EQ(lh.speculativeDecoding.totalDraftTokens, rh.speculativeDecoding.totalDraftTokens);

    EXPECT_EQ(lh.firstIter, rh.firstIter);
    EXPECT_EQ(lh.lastIter, rh.lastIter);
    EXPECT_EQ(lh.iter, rh.iter);
}

void compareKvCacheStats(texec::KvCacheStats const& lh, texec::KvCacheStats const& rh)
{
    EXPECT_TRUE(std::make_tuple(lh.maxNumBlocks, lh.freeNumBlocks, lh.usedNumBlocks, lh.tokensPerBlock)
        == std::make_tuple(rh.maxNumBlocks, rh.freeNumBlocks, rh.usedNumBlocks, rh.tokensPerBlock));
}

void compareStaticBatchingStats(texec::StaticBatchingStats const& lh, texec::StaticBatchingStats const& rh)
{
    EXPECT_TRUE(std::make_tuple(
                    lh.numScheduledRequests, lh.numContextRequests, lh.numCtxTokens, lh.numGenTokens, lh.emptyGenSlots)
        == std::make_tuple(
            rh.numScheduledRequests, rh.numContextRequests, rh.numCtxTokens, rh.numGenTokens, rh.emptyGenSlots));
}

void compareInflightBatchingStats(texec::InflightBatchingStats const& lh, texec::InflightBatchingStats const& rh)
{
    EXPECT_TRUE(std::make_tuple(lh.numScheduledRequests, lh.numContextRequests, lh.numGenRequests, lh.numPausedRequests,
                    lh.numCtxTokens, lh.microBatchId)
        == std::make_tuple(rh.numScheduledRequests, rh.numContextRequests, rh.numGenRequests, rh.numPausedRequests,
            rh.numCtxTokens, rh.microBatchId));
}

void compareIterationStats(texec::IterationStats const& lh, texec::IterationStats const& rh)
{
    EXPECT_EQ(lh.timestamp, rh.timestamp);
    EXPECT_EQ(lh.iter, rh.iter);
    EXPECT_EQ(lh.numActiveRequests, rh.numActiveRequests);
    EXPECT_EQ(lh.maxNumActiveRequests, rh.maxNumActiveRequests);
    EXPECT_EQ(lh.gpuMemUsage, rh.gpuMemUsage);
    EXPECT_EQ(lh.cpuMemUsage, rh.cpuMemUsage);
    EXPECT_EQ(lh.pinnedMemUsage, rh.pinnedMemUsage);
    EXPECT_EQ(lh.kvCacheStats.has_value(), rh.kvCacheStats.has_value());
    if (lh.kvCacheStats.has_value())
    {
        compareKvCacheStats(lh.kvCacheStats.value(), rh.kvCacheStats.value());
    }
    EXPECT_EQ(lh.staticBatchingStats.has_value(), rh.staticBatchingStats.has_value());
    if (lh.staticBatchingStats.has_value())
    {
        compareStaticBatchingStats(lh.staticBatchingStats.value(), rh.staticBatchingStats.value());
    }
    EXPECT_EQ(lh.inflightBatchingStats.has_value(), rh.inflightBatchingStats.has_value());
    if (lh.inflightBatchingStats.has_value())
    {
        compareInflightBatchingStats(lh.inflightBatchingStats.value(), rh.inflightBatchingStats.value());
    }
}

void compareDisServingRequestStats(texec::DisServingRequestStats const& lh, texec::DisServingRequestStats const& rh)
{
    EXPECT_EQ(lh.kvCacheTransferMS, rh.kvCacheTransferMS);
    EXPECT_EQ(lh.kvCacheSize, rh.kvCacheSize);
}

void compareRequestStats(texec::RequestStats const& lh, texec::RequestStats const& rh)
{
    EXPECT_EQ(lh.id, rh.id);
    EXPECT_EQ(lh.stage, rh.stage);
    EXPECT_EQ(lh.contextPrefillPosition, rh.contextPrefillPosition);
    EXPECT_EQ(lh.numGeneratedTokens, rh.numGeneratedTokens);
    EXPECT_EQ(lh.avgNumDecodedTokensPerIter, rh.avgNumDecodedTokensPerIter);
    EXPECT_EQ(lh.scheduled, rh.scheduled);
    EXPECT_EQ(lh.paused, rh.paused);
    EXPECT_EQ(lh.disServingStats.has_value(), rh.disServingStats.has_value());
    if (lh.disServingStats.has_value())
    {
        compareDisServingRequestStats(lh.disServingStats.value(), rh.disServingStats.value());
    }
    EXPECT_EQ(lh.allocTotalBlocksPerRequest, rh.allocTotalBlocksPerRequest);
    EXPECT_EQ(lh.allocNewBlocksPerRequest, rh.allocNewBlocksPerRequest);
    EXPECT_EQ(lh.reusedBlocksPerRequest, rh.reusedBlocksPerRequest);
    EXPECT_EQ(lh.missedBlocksPerRequest, rh.missedBlocksPerRequest);
    EXPECT_EQ(lh.kvCacheHitRatePerRequest, rh.kvCacheHitRatePerRequest);
}

void compareRequestStatsPerIteration(
    texec::RequestStatsPerIteration const& lh, texec::RequestStatsPerIteration const& rh)
{
    EXPECT_EQ(lh.iter, rh.iter);
    EXPECT_EQ(lh.requestStats.size(), rh.requestStats.size());
    for (size_t i = 0; i < lh.requestStats.size(); i++)
    {
        compareRequestStats(lh.requestStats.at(i), rh.requestStats.at(i));
    }
}

void compareResult(texec::Result res, texec::Result res2)
{
    EXPECT_EQ(res.isFinal, res2.isFinal);
    EXPECT_EQ(res.outputTokenIds, res2.outputTokenIds);
    EXPECT_EQ(res.cumLogProbs, res2.cumLogProbs);
    EXPECT_EQ(res.logProbs, res2.logProbs);
    EXPECT_EQ(res.finishReasons, res2.finishReasons);
    EXPECT_EQ(res.decodingIter, res2.decodingIter);
    EXPECT_EQ(res.sequenceIndex, res2.sequenceIndex);
    EXPECT_EQ(res.isSequenceFinal, res2.isSequenceFinal);
}

void compareResponse(texec::Response res, texec::Response res2)
{
    EXPECT_EQ(res.hasError(), res2.hasError());
    EXPECT_EQ(res.getRequestId(), res2.getRequestId());
    if (res.hasError())
    {
        EXPECT_EQ(res.getErrorMsg(), res2.getErrorMsg());
    }
    else
    {
        compareResult(res.getResult(), res2.getResult());
    }
    EXPECT_EQ(res.getClientId(), res2.getClientId());
}

template <typename T>
T serializeDeserialize(T val)
{
    auto size = su::serializedSize(val);
    std::ostringstream oss;
    su::serialize(val, oss);
    EXPECT_EQ(oss.str().size(), size);

    std::istringstream iss(oss.str());
    return su::deserialize<T>(iss);
}

template <typename T>
void testSerializeDeserialize(T val)
{
    auto val2 = serializeDeserialize(val);
    if constexpr (std::is_same<T, texec::Result>::value)
    {
        compareResult(val, val2);
    }
    else if constexpr (std::is_same<T, texec::Response>::value)
    {
        compareResponse(val, val2);
    }
    else if constexpr (std::is_same<T, texec::KvCacheStats>::value)
    {
        compareKvCacheStats(val, val2);
    }
    else if constexpr (std::is_same<T, texec::StaticBatchingStats>::value)
    {
        compareStaticBatching(val, val2);
    }
    else if constexpr (std::is_same<T, texec::InflightBatchingStats>::value)
    {
        compareInflightBatchingStats(val, val2);
    }
    else if constexpr (std::is_same<T, texec::IterationStats>::value)
    {
        compareIterationStats(val, val2);
    }
    else if constexpr (std::is_same<T, texec::RequestStatsPerIteration>::value)
    {
        compareRequestStatsPerIteration(val, val2);
    }
    else if constexpr (std::is_same<T, texec::RequestPerfMetrics>::value)
    {
        compareRequestPerfMetrics(val, val2);
    }
    else
    {
        EXPECT_EQ(val2, val) << typeid(T).name();
    }
}

template <typename T, typename T2>
void testSerializeDeserializeVariant(T val)
{
    auto val2 = serializeDeserialize(val);
    EXPECT_TRUE(std::holds_alternative<T2>(val2));
    if constexpr (std::is_same<T2, texec::Result>::value)
    {
        compareResult(std::get<T2>(val), std::get<T2>(val2));
    }
    else
    {
        EXPECT_EQ(std::get<T2>(val), std::get<T2>(val2));
    }
}

TEST(SerializeUtilsTest, FundamentalTypes)
{
    testSerializeDeserialize(int32_t(99));
    testSerializeDeserialize(int64_t(99));
    testSerializeDeserialize(uint32_t(99));
    testSerializeDeserialize(uint64_t(99));
    testSerializeDeserialize(float(99.f));
    testSerializeDeserialize(double(99.));
    testSerializeDeserialize(char('c'));
}

TEST(SerializeUtilsTest, Vector)
{
    {
        std::vector<int32_t> vec{1, 2, 3, 4};
        testSerializeDeserialize(vec);
    }
    {
        std::vector<char> vec{'a', 'b', 'c', 'd'};
        testSerializeDeserialize(vec);
    }
    {
        std::vector<float> vec{1.f, 2.f, 3.f, 4.f};
        testSerializeDeserialize(vec);
    }
}

TEST(SerializeUtilsTest, List)
{
    {
        std::list<int32_t> list{1, 2, 3, 4};
        testSerializeDeserialize(list);
    }
    {
        std::list<char> list{'a', 'b', 'c', 'd'};
        testSerializeDeserialize(list);
    }
    {
        std::list<float> list{9.0f, 3.333f};
        testSerializeDeserialize(list);
    }
}

TEST(SerializeUtilsTest, String)
{
    {
        std::string str{"abcdefg"};
        testSerializeDeserialize(str);
    }
}

TEST(SerializeUtilsTest, Optional)
{
    {
        std::optional<int32_t> opt = std::nullopt;
        testSerializeDeserialize(opt);
    }
    {
        std::optional<int32_t> opt = 1;
        testSerializeDeserialize(opt);
    }
    {
        std::optional<char> opt = std::nullopt;
        testSerializeDeserialize(opt);
    }
    {
        std::optional<char> opt = 'c';
        testSerializeDeserialize(opt);
    }
    {
        std::optional<float> opt = std::nullopt;
        testSerializeDeserialize(opt);
    }
    {
        std::optional<float> opt = 1.f;
        testSerializeDeserialize(opt);
    }
}

TEST(SerializeUtilsTest, Variant)
{
    {
        std::variant<bool, int32_t> val = int32_t(10);
        testSerializeDeserializeVariant<std::variant<bool, int32_t>, int32_t>(val);
    }
    {
        std::variant<bool, texec::Result> val = texec::Result{false, {{1, 2, 3}}};
        testSerializeDeserializeVariant<std::variant<bool, texec::Result>, texec::Result>(val);
    }
    {
        std::variant<bool, texec::Result> val = true;
        testSerializeDeserializeVariant<std::variant<bool, texec::Result>, bool>(val);
    }
}

static texec::RequestPerfMetrics::TimePoint generateTimePoint(uint64_t microseconds)
{
    std::chrono::microseconds duration(microseconds);
    return std::chrono::steady_clock::time_point(duration);
}

TEST(SerializeUtilsTest, RequestPerfMetrics)
{
    uint64_t x = 42;
    texec::RequestPerfMetrics::TimingMetrics timingMetrics{
        generateTimePoint(x++),
        generateTimePoint(x++),
        generateTimePoint(x++),
        generateTimePoint(x++),
        generateTimePoint(x++),
        generateTimePoint(x++),
    };

    texec::RequestPerfMetrics::KvCacheMetrics kvCacheMetrics{1, 2, 3, 4, 5};

    texec::RequestPerfMetrics val{timingMetrics, kvCacheMetrics, 1000, 1001, 1002, 0.6f};

    testSerializeDeserialize(val);
}

TEST(SerializeUtilsTest, SamplingConfig)
{
    {
        texec::SamplingConfig val(2);
        testSerializeDeserialize(val);
    }
    {
        texec::SamplingConfig val(4);
        val.setNumReturnSequences(3);
        testSerializeDeserialize(val);
    }
    {
        texec::SamplingConfig val(1);
        val.setNumReturnSequences(3);
        testSerializeDeserialize(val);
    }
}

TEST(SerializeUtilsTest, Nested)
{
    {
        std::optional<std::vector<int32_t>> val = std::nullopt;
        testSerializeDeserialize(val);
    }
    {
        std::optional<std::vector<int32_t>> val = std::vector<int32_t>{1, 2, 3, 5};
        testSerializeDeserialize(val);
    }
    {
        std::list<std::vector<int32_t>> val = {{2, 3}, {5, 6, 7}};
        testSerializeDeserialize(val);
    }
    {
        std::list<std::vector<std::optional<float>>> val = {{2.f, 3.f}, {5.f, 6.f, 7.f}, {std::nullopt, 3.f}};
        testSerializeDeserialize(val);
    }
    // Unsupported won't build
    //{
    // std::map<int, int> val;
    // val[1] = 1;
    // testSerializeDeserialize(val);
    //}
    {
        auto const val = std::make_optional(std::vector<texec::SamplingConfig>{
            texec::SamplingConfig{1, 1, 0.05, 0.2}, texec::SamplingConfig{2, std::nullopt}});
        testSerializeDeserialize(val);
    }
    {
        auto const val = std::make_optional(texec::ExternalDraftTokensConfig({1, 1}));
        auto const size = su::serializedSize(val);
        std::ostringstream oss;
        su::serialize(val, oss);
        EXPECT_EQ(oss.str().size(), size);

        std::istringstream iss(oss.str());
        auto const val2 = su::deserialize<std::optional<texec::ExternalDraftTokensConfig>>(iss);
        EXPECT_EQ(val2.value().getTokens(), val.value().getTokens());
    }
}

TEST(SerializeUtilsTest, ResultResponse)
{
    texec::Result res = texec::Result{false, {{1, 2, 3}}, texec::VecLogProbs{1.0, 2.0},
        std::vector<texec::VecLogProbs>{{1.1, 2.2}, {3.3, 4.4}}, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
        std::vector<texec::FinishReason>{texec::FinishReason::kLENGTH},
        texec::ContextPhaseParams({9, 37}, 0, VecTokens{1, 2}), 3, 2, true};
    {
        testSerializeDeserialize(res);
    }
    {
        auto val = texec::Response(1, res);
        testSerializeDeserialize(val);
    }
    {
        auto val = texec::Response(1, "my error msg");
        testSerializeDeserialize(val);
    }
    {
        auto val = texec::Response(1, "my error msg", 2);
        testSerializeDeserialize(val);
    }
}

TEST(SerializeUtilsTest, VectorResponses)
{
    int numResponses = 15;
    std::vector<texec::Response> responsesIn;
    for (int i = 0; i < numResponses; ++i)
    {
        if (i < 5)
        {
            texec::Result res = texec::Result{false, {{i + 1, i + 2, i + 3}}, texec::VecLogProbs{1.0, 2.0},
                std::vector<texec::VecLogProbs>{{1.1, 2.2}, {3.3, 4.4}}, std::nullopt, std::nullopt, std::nullopt,
                std::nullopt, std::vector<texec::FinishReason>{texec::FinishReason::kEND_ID}};
            responsesIn.emplace_back(i, res);
        }
        else if (i < 10)
        {
            std::string errMsg = "my_err_msg" + std::to_string(i);
            responsesIn.emplace_back(i, errMsg);
        }
        else
        {
            std::string errMsg = "my_err_msg" + std::to_string(i);
            responsesIn.emplace_back(i, errMsg, i + 1);
        }
    }

    auto buffer = texec::Serialization::serialize(responsesIn);
    auto responsesOut = texec::Serialization::deserializeResponses(buffer);

    EXPECT_EQ(responsesIn.size(), responsesOut.size());

    for (int i = 0; i < numResponses; ++i)
    {
        compareResponse(responsesIn.at(i), responsesOut.at(i));
    }
}

TEST(SerializeUtilsTest, KvCacheConfig)
{
    texec::KvCacheConfig kvCacheConfig(
        true, 10, std::vector(1, 100), 2, 0.1, 10000, false, 0.5, 50, 1024, false, false, true, 77);
    auto kvCacheConfig2 = serializeDeserialize(kvCacheConfig);

    EXPECT_EQ(kvCacheConfig.getEnableBlockReuse(), kvCacheConfig2.getEnableBlockReuse());
    EXPECT_EQ(kvCacheConfig.getEnablePartialReuse(), kvCacheConfig2.getEnablePartialReuse());
    EXPECT_EQ(kvCacheConfig.getCopyOnPartialReuse(), kvCacheConfig2.getCopyOnPartialReuse());
    EXPECT_EQ(kvCacheConfig.getMaxTokens(), kvCacheConfig2.getMaxTokens());
    EXPECT_EQ(kvCacheConfig.getMaxAttentionWindowVec(), kvCacheConfig2.getMaxAttentionWindowVec());
    EXPECT_EQ(kvCacheConfig.getSinkTokenLength(), kvCacheConfig2.getSinkTokenLength());
    EXPECT_EQ(kvCacheConfig.getFreeGpuMemoryFraction(), kvCacheConfig2.getFreeGpuMemoryFraction());
    EXPECT_EQ(kvCacheConfig.getHostCacheSize(), kvCacheConfig2.getHostCacheSize());
    EXPECT_EQ(kvCacheConfig.getOnboardBlocks(), kvCacheConfig2.getOnboardBlocks());
    EXPECT_EQ(kvCacheConfig.getCrossKvCacheFraction(), kvCacheConfig2.getCrossKvCacheFraction());
    EXPECT_EQ(kvCacheConfig.getSecondaryOffloadMinPriority(), kvCacheConfig2.getSecondaryOffloadMinPriority());
    EXPECT_EQ(kvCacheConfig.getEventBufferMaxSize(), kvCacheConfig2.getEventBufferMaxSize());
    EXPECT_EQ(kvCacheConfig.getUseUvm(), kvCacheConfig2.getUseUvm());
    EXPECT_EQ(kvCacheConfig.getAttentionDpEventsGatherPeriodMs(), kvCacheConfig2.getAttentionDpEventsGatherPeriodMs());
}

TEST(SerializeUtilsTest, SchedulerConfig)
{
    texec::SchedulerConfig schedulerConfig(
        texec::CapacitySchedulerPolicy::kMAX_UTILIZATION, texec::ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED);
    auto schedulerConfig2 = serializeDeserialize(schedulerConfig);
    EXPECT_EQ(schedulerConfig.getCapacitySchedulerPolicy(), schedulerConfig2.getCapacitySchedulerPolicy());
    EXPECT_EQ(schedulerConfig.getContextChunkingPolicy(), schedulerConfig2.getContextChunkingPolicy());
}

TEST(SerializeUtilsTest, ParallelConfig)
{
    texec::ParallelConfig parallelConfig(texec::CommunicationType::kMPI, texec::CommunicationMode::kLEADER,
        std::vector<texec::SizeType32>{1, 2, 7}, std::vector<texec::SizeType32>{0, 1, 4});

    auto parallelConfig2 = serializeDeserialize(parallelConfig);
    EXPECT_EQ(parallelConfig.getCommunicationType(), parallelConfig2.getCommunicationType());
    EXPECT_EQ(parallelConfig.getCommunicationMode(), parallelConfig2.getCommunicationMode());
    EXPECT_EQ(parallelConfig.getDeviceIds(), parallelConfig2.getDeviceIds());
    EXPECT_EQ(parallelConfig.getParticipantIds(), parallelConfig2.getParticipantIds());
}

TEST(SerializeUtilsTest, PeftCacheConfig)
{
    auto peftCacheConfig = texec::PeftCacheConfig(10, 9, 8, 7, 6, 5, 4, 3, 2, 0.9, 1000);
    testSerializeDeserialize(peftCacheConfig);
}

TEST(SerializeUtilsTest, LookaheadDecodingConfig)
{
    auto lookaheadDecodingConfig = texec::LookaheadDecodingConfig(3, 5, 7);
    auto lookaheadDecodingConfig2 = serializeDeserialize(lookaheadDecodingConfig);
    EXPECT_EQ(lookaheadDecodingConfig.getNgramSize(), lookaheadDecodingConfig2.getNgramSize());
    EXPECT_EQ(lookaheadDecodingConfig.getWindowSize(), lookaheadDecodingConfig2.getWindowSize());
    EXPECT_EQ(lookaheadDecodingConfig.getVerificationSetSize(), lookaheadDecodingConfig2.getVerificationSetSize());
}

TEST(SerializeUtilsTest, EagleConfig)
{
    texec::EagleChoices eagleChoices{{{0, 1, 2}}};
    auto eagleConfig = texec::EagleConfig(eagleChoices);
    auto eagleConfig2 = serializeDeserialize(eagleConfig);
    EXPECT_EQ(eagleConfig.getEagleChoices(), eagleConfig2.getEagleChoices());
}

TEST(SerializeUtilsTest, KvCacheRetentionConfig)
{

    using namespace std::chrono_literals;

    auto kvCacheRetentionConfig = texec::KvCacheRetentionConfig();
    auto kvCacheRetentionConfig2 = serializeDeserialize(kvCacheRetentionConfig);
    EXPECT_EQ(kvCacheRetentionConfig.getTokenRangeRetentionConfigs(),
        kvCacheRetentionConfig2.getTokenRangeRetentionConfigs());
    EXPECT_EQ(
        kvCacheRetentionConfig.getDecodeRetentionPriority(), kvCacheRetentionConfig2.getDecodeRetentionPriority());
    EXPECT_EQ(kvCacheRetentionConfig.getDecodeDurationMs(), kvCacheRetentionConfig2.getDecodeDurationMs());

    kvCacheRetentionConfig
        = texec::KvCacheRetentionConfig({texec::KvCacheRetentionConfig::TokenRangeRetentionConfig(0, 1, 80),
                                            texec::KvCacheRetentionConfig::TokenRangeRetentionConfig(1, 2, 50),
                                            texec::KvCacheRetentionConfig::TokenRangeRetentionConfig(2, 3, 30)},
            5, 30s);
    kvCacheRetentionConfig2 = serializeDeserialize(kvCacheRetentionConfig);
    EXPECT_EQ(kvCacheRetentionConfig.getTokenRangeRetentionConfigs(),
        kvCacheRetentionConfig2.getTokenRangeRetentionConfigs());
    EXPECT_EQ(
        kvCacheRetentionConfig.getDecodeRetentionPriority(), kvCacheRetentionConfig2.getDecodeRetentionPriority());
    EXPECT_EQ(kvCacheRetentionConfig.getDecodeDurationMs(), kvCacheRetentionConfig2.getDecodeDurationMs());
}

TEST(SerializeUtilsTest, DecodingConfig)
{
    {
        texec::DecodingMode decodingMode{texec::DecodingMode::Lookahead()};
        texec::LookaheadDecodingConfig laConfig{3, 5, 7};
        auto specDecodingConfig = texec::DecodingConfig(decodingMode, laConfig);
        auto specDecodingConfig2 = serializeDeserialize(specDecodingConfig);
        EXPECT_EQ(specDecodingConfig.getDecodingMode(), specDecodingConfig2.getDecodingMode());
        EXPECT_EQ(specDecodingConfig.getLookaheadDecodingConfig(), specDecodingConfig2.getLookaheadDecodingConfig());
    }

    {
        texec::DecodingMode decodingMode{texec::DecodingMode::Medusa()};
        texec::MedusaChoices medusaChoices{{{0, 1, 2}}};
        auto specDecodingConfig = texec::DecodingConfig(decodingMode, std::nullopt, medusaChoices);
        auto specDecodingConfig2 = serializeDeserialize(specDecodingConfig);
        EXPECT_EQ(specDecodingConfig.getDecodingMode(), specDecodingConfig2.getDecodingMode());
        EXPECT_EQ(specDecodingConfig.getMedusaChoices(), specDecodingConfig2.getMedusaChoices());
    }

    {
        texec::DecodingMode decodingMode{texec::DecodingMode::Eagle()};
        texec::EagleChoices eagleChoices{{{0, 1, 2}}};
        texec::EagleConfig eagleConfig{eagleChoices};
        auto specDecodingConfig = texec::DecodingConfig(decodingMode, std::nullopt, std::nullopt, eagleConfig);
        auto specDecodingConfig2 = serializeDeserialize(specDecodingConfig);
        EXPECT_EQ(specDecodingConfig.getDecodingMode(), specDecodingConfig2.getDecodingMode());
        EXPECT_EQ(specDecodingConfig.getEagleConfig()->getEagleChoices(),
            specDecodingConfig2.getEagleConfig()->getEagleChoices());
    }
}

TEST(SerializeUtilsTest, DebugConfig)
{
    texec::DebugConfig debugConfig(true, true, {"test"}, 3);
    auto debugConfig2 = serializeDeserialize(debugConfig);
    EXPECT_EQ(debugConfig.getDebugInputTensors(), debugConfig2.getDebugInputTensors());
    EXPECT_EQ(debugConfig.getDebugOutputTensors(), debugConfig2.getDebugOutputTensors());
    EXPECT_EQ(debugConfig.getDebugTensorNames(), debugConfig2.getDebugTensorNames());
    EXPECT_EQ(debugConfig.getDebugTensorsMaxIterations(), debugConfig2.getDebugTensorsMaxIterations());
}

TEST(SerializeUtilsTest, OrchestratorConfig)
{
    auto orchConfig = texec::OrchestratorConfig(false, std::filesystem::current_path().string(), nullptr, false);
    auto orchConfig2 = serializeDeserialize(orchConfig);
    EXPECT_EQ(orchConfig.getIsOrchestrator(), orchConfig2.getIsOrchestrator());
    EXPECT_EQ(orchConfig.getWorkerExecutablePath(), orchConfig2.getWorkerExecutablePath());
    EXPECT_EQ(orchConfig.getSpawnProcesses(), orchConfig2.getSpawnProcesses());
}

TEST(SerializeUtilsTest, KvCacheStats)
{
    auto stats = texec::KvCacheStats{10, 20, 30, 40, 50, 60, 70};
    auto stats2 = serializeDeserialize(stats);
    compareKvCacheStats(stats, stats2);
}

TEST(SerializeUtilsTest, StaticBatchingStats)
{
    auto stats = texec::StaticBatchingStats{10, 20, 30, 40, 50};
    auto stats2 = serializeDeserialize(stats);
    compareStaticBatchingStats(stats, stats2);
}

TEST(SerializeUtilsTest, InflightBatchingStats)
{
    auto stats = texec::InflightBatchingStats{10, 20, 30, 40, 50, 60};
    auto stats2 = serializeDeserialize(stats);
    compareInflightBatchingStats(stats, stats2);
}

TEST(SerializeUtilsTest, IterationStats)
{
    auto timestamp = std::string{"05:01:00"};
    auto iter = texec::IterationType{10};
    auto iterLatencyMS = double{100};
    auto newActiveRequestsQueueLatencyMS = double{1000};
    auto numNewActiveRequests = texec::SizeType32{10};
    auto numActiveRequests = texec::SizeType32{20};
    auto numQueuedRequests = texec::SizeType32{30};
    auto numCompletedRequests = texec::SizeType32{10};
    auto maxNumActiveRequests = texec::SizeType32{30};
    auto maxBatchSizeStatic = texec::SizeType32{100};
    auto maxBatchSizeTunerRecommended = texec::SizeType32{50};
    auto maxBatchSizeRuntime = texec::SizeType32{50};
    auto maxNumTokensStatic = texec::SizeType32{100};
    auto maxNumTokensTunerRecommended = texec::SizeType32{50};
    auto maxNumTokensRuntime = texec::SizeType32{50};
    auto gpuMemUsage = size_t{1024};
    auto cpuMemUsage = size_t{2048};
    auto pinnedMemUsage = size_t{4096};
    auto kvCacheStats = texec::KvCacheStats{10, 20, 30, 40, 50, 60, 70};
    auto staticBatchingStats = texec::StaticBatchingStats{10, 20, 30, 40, 50};
    auto ifbBatchingStats = texec::InflightBatchingStats{10, 20, 30, 40, 50, 60};
    {
        {
            auto stats = texec::IterationStats{timestamp, iter, iterLatencyMS, newActiveRequestsQueueLatencyMS,
                numNewActiveRequests, numActiveRequests, numQueuedRequests, numCompletedRequests, maxNumActiveRequests,
                maxBatchSizeStatic, maxBatchSizeTunerRecommended, maxBatchSizeRuntime, maxNumTokensStatic,
                maxNumTokensTunerRecommended, maxNumTokensRuntime, gpuMemUsage, cpuMemUsage, pinnedMemUsage,
                kvCacheStats, kvCacheStats, staticBatchingStats, ifbBatchingStats};

            // serialize and deserialize using std::vector<char>
            {
                auto buffer = texec::Serialization::serialize(stats);
                auto stats2 = texec::Serialization::deserializeIterationStats(buffer);
                compareIterationStats(stats, stats2);
            }
            // serialize deserialize using is, os
            {
                auto stats2 = serializeDeserialize(stats);
                compareIterationStats(stats, stats2);
            }
        }
    }

    for (auto kvStats : std::vector<std::optional<texec::KvCacheStats>>{std::nullopt, kvCacheStats})
    {
        for (auto staticBatchStats :
            std::vector<std::optional<texec::StaticBatchingStats>>{std::nullopt, staticBatchingStats})
        {
            for (auto ifbBatchStats :
                std::vector<std::optional<texec::InflightBatchingStats>>{std::nullopt, ifbBatchingStats})
            {
                auto stats = texec::IterationStats{timestamp, iter, iterLatencyMS, newActiveRequestsQueueLatencyMS,
                    numNewActiveRequests, numActiveRequests, numQueuedRequests, numCompletedRequests,
                    maxNumActiveRequests, maxBatchSizeStatic, maxBatchSizeTunerRecommended, maxBatchSizeRuntime,
                    maxNumTokensStatic, maxNumTokensTunerRecommended, maxNumTokensRuntime, gpuMemUsage, cpuMemUsage,
                    pinnedMemUsage, kvStats, kvStats, staticBatchStats, ifbBatchStats};
                {
                    auto buffer = texec::Serialization::serialize(stats);
                    auto stats2 = texec::Serialization::deserializeIterationStats(buffer);
                    compareIterationStats(stats, stats2);
                }
                {
                    auto stats2 = serializeDeserialize(stats);
                    compareIterationStats(stats, stats2);
                }
            }
        }
    }
}

TEST(SerializeUtilsTest, ContextPhaseParams)
{
    {
        auto stats = texec::ContextPhaseParams({1}, 0, std::nullopt);
        auto stats2 = serializeDeserialize(stats);
        EXPECT_EQ(stats, stats2);
    }

    {
        auto state = std::make_unique<texec::DataTransceiverState>();
        state->setCommState(texec::kv_cache::CommState{{10, 20}});
        auto stats = texec::ContextPhaseParams({10, 20, 30, 40, 50, 60}, 1, state.release(), VecTokens{10, 20});
        auto stats2 = serializeDeserialize(stats);
        EXPECT_EQ(stats, stats2);
    }

    {
        auto state = std::make_unique<texec::DataTransceiverState>();
        state->setCommState(texec::kv_cache::CommState{12, "127.0.0.1"});
        state->setCacheState(texec::kv_cache::CacheState{10, 12, 128, 128, 8, 8, 8, {4}, nvinfer1::DataType::kFLOAT});
        auto stats = texec::ContextPhaseParams({10, 20, 30, 40, 50, 60}, 0, state.release(), VecTokens{10, 20});
        auto stats2 = serializeDeserialize(stats);
        EXPECT_EQ(stats, stats2);
    }

    {
        auto state = std::make_unique<texec::DataTransceiverState>();
        state->setCommState(texec::kv_cache::CommState{{10, 20}});
        auto state2 = *state;
        auto contextPhaseParams
            = texec::ContextPhaseParams({10, 20, 30, 40, 50, 60}, 1, state.release(), VecTokens{10, 20});

        auto serializedState = contextPhaseParams.getSerializedState();
        auto stateCopy = texec::Serialization::deserializeDataTransceiverState(serializedState);

        EXPECT_EQ(state2, stateCopy);
    }
}

TEST(SerializeUtilsTest, SpeculativeDecodingFastLogitsInfo)
{
    auto logitsInfo = texec::SpeculativeDecodingFastLogitsInfo{10, 20};
    auto logitsInfo2 = serializeDeserialize(logitsInfo);
    EXPECT_EQ(logitsInfo.draftRequestId, logitsInfo2.draftRequestId);
    EXPECT_EQ(logitsInfo.draftParticipantId, logitsInfo2.draftParticipantId);
}

TEST(SerializeUtilsTest, GuidedDecodingConfig)
{
    std::vector<std::string> encodedVocab{"eos", "a", "b", "c", "d"};
    std::vector<texec::TokenIdType> stopTokenIds{0};
    texec::GuidedDecodingConfig guidedDecodingConfig(
        texec::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR, encodedVocab, std::nullopt, stopTokenIds);
    auto guidedDecodingConfig2 = serializeDeserialize(guidedDecodingConfig);
    EXPECT_EQ(guidedDecodingConfig.getBackend(), guidedDecodingConfig2.getBackend());
    EXPECT_EQ(guidedDecodingConfig.getEncodedVocab(), guidedDecodingConfig2.getEncodedVocab());
    EXPECT_EQ(guidedDecodingConfig.getTokenizerStr(), guidedDecodingConfig2.getTokenizerStr());
    EXPECT_EQ(guidedDecodingConfig.getStopTokenIds(), guidedDecodingConfig2.getStopTokenIds());
}

TEST(SerializeUtilsTest, GuidedDecodingParams)
{
    texec::GuidedDecodingParams guidedDecodingParams(texec::GuidedDecodingParams::GuideType::kREGEX, R"(\d+)");
    auto guidedDecodingParams2 = serializeDeserialize(guidedDecodingParams);
    EXPECT_EQ(guidedDecodingParams.getGuideType(), guidedDecodingParams2.getGuideType());
    EXPECT_EQ(guidedDecodingParams.getGuide(), guidedDecodingParams2.getGuide());
}

TEST(SerializeUtilsTest, ExecutorConfig)
{
    texec::ExecutorConfig executorConfig(2, texec::SchedulerConfig(texec::CapacitySchedulerPolicy::kMAX_UTILIZATION),
        texec::KvCacheConfig(true), true, false, 500, 200, texec::BatchingType::kSTATIC, 128, 64,
        texec::ParallelConfig(texec::CommunicationType::kMPI, texec::CommunicationMode::kORCHESTRATOR),
        texec::PeftCacheConfig(10), std::nullopt,
        texec::DecodingConfig(texec::DecodingMode::Lookahead(), texec::LookaheadDecodingConfig(3, 5, 7)), false, 0.5f,
        8, texec::ExtendedRuntimePerfKnobConfig(true), texec::DebugConfig(true), 60000000, 180000000,
        texec::SpeculativeDecodingConfig(true),
        texec::GuidedDecodingConfig(
            texec::GuidedDecodingConfig::GuidedDecodingBackend::kXGRAMMAR, std::initializer_list<std::string>{"eos"}),
        std::vector{tensorrt_llm::executor::AdditionalModelOutput{"output_name"}},
        texec::CacheTransceiverConfig(std::nullopt, 1024), true, true, true);
    auto executorConfig2 = serializeDeserialize(executorConfig);

    EXPECT_EQ(executorConfig.getMaxBeamWidth(), executorConfig2.getMaxBeamWidth());
    EXPECT_EQ(executorConfig.getSchedulerConfig(), executorConfig2.getSchedulerConfig());
    EXPECT_EQ(executorConfig.getKvCacheConfig().getEnableBlockReuse(),
        executorConfig2.getKvCacheConfig().getEnableBlockReuse());
    EXPECT_EQ(executorConfig.getEnableChunkedContext(), executorConfig2.getEnableChunkedContext());
    EXPECT_EQ(executorConfig.getNormalizeLogProbs(), executorConfig2.getNormalizeLogProbs());
    EXPECT_EQ(executorConfig.getIterStatsMaxIterations(), executorConfig2.getIterStatsMaxIterations());
    EXPECT_EQ(executorConfig.getRequestStatsMaxIterations(), executorConfig2.getRequestStatsMaxIterations());
    EXPECT_EQ(executorConfig.getBatchingType(), executorConfig2.getBatchingType());
    EXPECT_EQ(executorConfig.getMaxBatchSize(), executorConfig2.getMaxBatchSize());
    EXPECT_EQ(executorConfig.getMaxNumTokens(), executorConfig2.getMaxNumTokens());
    EXPECT_EQ(executorConfig.getParallelConfig().value().getCommunicationMode(),
        executorConfig2.getParallelConfig().value().getCommunicationMode());
    EXPECT_EQ(executorConfig.getPeftCacheConfig(), executorConfig2.getPeftCacheConfig());
    EXPECT_EQ(executorConfig.getDecodingConfig(), executorConfig2.getDecodingConfig());
    EXPECT_EQ(executorConfig.getUseGpuDirectStorage(), executorConfig2.getUseGpuDirectStorage());
    EXPECT_EQ(executorConfig.getGpuWeightsPercent(), executorConfig2.getGpuWeightsPercent());
    EXPECT_EQ(executorConfig.getMaxQueueSize(), executorConfig2.getMaxQueueSize());
    EXPECT_EQ(executorConfig.getExtendedRuntimePerfKnobConfig(), executorConfig2.getExtendedRuntimePerfKnobConfig());
    EXPECT_EQ(executorConfig.getDebugConfig(), executorConfig2.getDebugConfig());
    EXPECT_EQ(executorConfig.getRecvPollPeriodMs(), executorConfig2.getRecvPollPeriodMs());
    EXPECT_EQ(executorConfig.getMaxSeqIdleMicroseconds(), executorConfig2.getMaxSeqIdleMicroseconds());
    EXPECT_EQ(executorConfig.getSpecDecConfig(), executorConfig2.getSpecDecConfig());
    EXPECT_EQ(executorConfig.getGuidedDecodingConfig(), executorConfig2.getGuidedDecodingConfig());
    EXPECT_EQ(executorConfig.getAdditionalModelOutputs(), executorConfig2.getAdditionalModelOutputs());
    EXPECT_EQ(executorConfig.getCacheTransceiverConfig(), executorConfig2.getCacheTransceiverConfig());
    EXPECT_EQ(executorConfig.getGatherGenerationLogits(), executorConfig2.getGatherGenerationLogits());
    EXPECT_EQ(executorConfig.getPromptTableOffloading(), executorConfig2.getPromptTableOffloading());
    EXPECT_EQ(executorConfig.getEnableTrtOverlap(), executorConfig2.getEnableTrtOverlap());
}

TEST(SerializeUtilsTest, RequestStats)
{
    tensorrt_llm::executor::DisServingRequestStats disServingRequestStats{0.56222, 78643200};
    texec::RequestStats requestStats{123, tensorrt_llm::executor::RequestStage::kQUEUED, 56, 25, 135, true, false,
        disServingRequestStats, 33, 22, 6, 1, 8};
    auto requestStats2 = serializeDeserialize(requestStats);
    compareRequestStats(requestStats, requestStats2);
    requestStats.disServingStats = std::nullopt;
    requestStats2 = serializeDeserialize(requestStats);
    compareRequestStats(requestStats, requestStats2);
}

TEST(SerializeUtilsTest, RequestStatsPerIteration)
{

    tensorrt_llm::executor::DisServingRequestStats disServingRequestStats{0.56222, 78643200};
    texec::RequestStats requestStats1{123, tensorrt_llm::executor::RequestStage::kQUEUED, 56, 25, 135, true, false,
        disServingRequestStats, 33, 22, 6, 1, 8};
    texec::RequestStats requestStats2{
        899, texec::RequestStage::kGENERATION_IN_PROGRESS, 98, 78, 77, false, true, std::nullopt, 7, 14, 65, 61, 78};

    texec::RequestStatsPerIteration requestStatsPerIteration{0, {requestStats1, requestStats2}};
    auto requestStatsPerIteration2 = serializeDeserialize(requestStatsPerIteration);
    compareRequestStatsPerIteration(requestStatsPerIteration, requestStatsPerIteration2);
}

void compareKvCacheEvents(texec::KVCacheEvent const& kvCacheEvent, texec::KVCacheEvent const& kvCacheEvent2)
{
    EXPECT_EQ(kvCacheEvent.eventId, kvCacheEvent2.eventId);
    EXPECT_EQ(kvCacheEvent.windowSize, kvCacheEvent2.windowSize);
    EXPECT_EQ(kvCacheEvent.attentionDpRank, kvCacheEvent2.attentionDpRank);

    if (std::holds_alternative<texec::KVCacheCreatedData>(kvCacheEvent.data))
    {
        EXPECT_TRUE(std::holds_alternative<texec::KVCacheCreatedData>(kvCacheEvent2.data));
        auto data = std::get<texec::KVCacheCreatedData>(kvCacheEvent.data);
        auto data2 = std::get<texec::KVCacheCreatedData>(kvCacheEvent2.data);
        EXPECT_EQ(data.numBlocksPerCacheLevel, data2.numBlocksPerCacheLevel);
    }
    else if (std::holds_alternative<texec::KVCacheRemovedData>(kvCacheEvent.data))
    {
        EXPECT_TRUE(std::holds_alternative<texec::KVCacheRemovedData>(kvCacheEvent2.data));
        auto data = std::get<texec::KVCacheRemovedData>(kvCacheEvent.data);
        auto data2 = std::get<texec::KVCacheRemovedData>(kvCacheEvent2.data);
        EXPECT_EQ(data.blockHashes, data2.blockHashes);
    }
    else if (std::holds_alternative<texec::KVCacheStoredData>(kvCacheEvent.data))
    {
        EXPECT_TRUE(std::holds_alternative<texec::KVCacheStoredData>(kvCacheEvent2.data));
        auto data = std::get<texec::KVCacheStoredData>(kvCacheEvent.data);
        auto data2 = std::get<texec::KVCacheStoredData>(kvCacheEvent2.data);
        EXPECT_EQ(data.parentHash, data2.parentHash);
        EXPECT_EQ(data.blocks.size(), data2.blocks.size());
        for (size_t i = 0; i < data.blocks.size(); ++i)
        {
            auto blockData = data.blocks[i];
            auto blockData2 = data2.blocks[i];
            EXPECT_EQ(blockData.blockHash, blockData2.blockHash);
            EXPECT_EQ(blockData.loraId, blockData2.loraId);
            EXPECT_EQ(blockData.cacheLevel, blockData2.cacheLevel);
            EXPECT_EQ(blockData.priority, blockData2.priority);
            EXPECT_EQ(blockData.tokens.size(), blockData2.tokens.size());
            for (size_t j = 0; j < blockData.tokens.size(); ++j)
            {
                EXPECT_EQ(blockData.tokens[j].tokenId, blockData2.tokens[j].tokenId);
                EXPECT_EQ(blockData.tokens[j].tokenExtraId, blockData2.tokens[j].tokenExtraId);
            }
        }
    }
    else if (std::holds_alternative<texec::KVCacheUpdatedData>(kvCacheEvent.data))
    {
        EXPECT_TRUE(std::holds_alternative<texec::KVCacheUpdatedData>(kvCacheEvent2.data));
        auto data = std::get<texec::KVCacheUpdatedData>(kvCacheEvent.data);
        auto data2 = std::get<texec::KVCacheUpdatedData>(kvCacheEvent2.data);
        EXPECT_EQ(data.blockHash, data2.blockHash);
        if (data.cacheLevel)
        {
            EXPECT_TRUE(data2.cacheLevel);
            EXPECT_EQ(data.cacheLevel.value().oldValue, data2.cacheLevel.value().oldValue);
            EXPECT_EQ(data.cacheLevel.value().newValue, data2.cacheLevel.value().newValue);
        }
        if (data.priority)
        {
            EXPECT_TRUE(data2.priority);
            EXPECT_EQ(data.priority.value().oldValue, data2.priority.value().oldValue);
            EXPECT_EQ(data.priority.value().newValue, data2.priority.value().newValue);
        }
    }
    else
    {
        FAIL() << "Unknown KVCacheEvent data type";
    }
}

TEST(SerializeUtilsTest, KvCacheEventsDeque)
{
    // Created event
    texec::KVCacheCreatedData kvCacheCreatedData{{1, 2}};
    texec::KVCacheEvent kvCacheCreatedEvent(1, kvCacheCreatedData, 32);

    // Removed event
    texec::KVCacheEvent kvCacheRemovedEvent(1, texec::KVCacheRemovedData{{3, 4}}, 32);

    // Stored event
    auto storedBlockData1 = texec::KVCacheStoredBlockData(77, {{1, 2}, {3, 4}, {5, 6}}, 88, 0, 99);
    auto storedBlockData2 = texec::KVCacheStoredBlockData(99, {{11, 12}, {3, 4}, {15, 6}}, 77, 1, 101);
    texec::KVCacheStoredData kvCacheStoredData{177, {storedBlockData1, storedBlockData2}};
    texec::KVCacheEvent kvCacheStoredEvent(1, kvCacheStoredData, 32);

    // Updated event
    texec::KVCacheEventDiff<texec::SizeType32> diff{0, 1};
    texec::KVCacheEventDiff<texec::SizeType32> diff2{90, 99};
    texec::KVCacheUpdatedData kvCacheUpdatedData(999, diff, diff2);
    texec::KVCacheEvent kvCacheEvent(1, kvCacheUpdatedData, 32);

    std::deque<texec::KVCacheEvent> kvCacheEvents{
        kvCacheCreatedEvent, kvCacheRemovedEvent, kvCacheStoredEvent, kvCacheEvent};

    auto serializedEvents = texec::Serialization::serialize(kvCacheEvents);
    auto kvCacheEvents2 = texec::Serialization::deserializeKVCacheEvents(serializedEvents);

    EXPECT_EQ(kvCacheEvents.size(), kvCacheEvents2.size());
    for (size_t i = 0; i < kvCacheEvents.size(); ++i)
    {
        compareKvCacheEvents(kvCacheEvents[i], kvCacheEvents2[i]);
    }
}

// Test for KVCacheEvent with KVCacheCreatedData
TEST(SerializeUtilsTest, KVCacheCreatedEvent)
{
    texec::KVCacheCreatedData kvCacheCreatedData{{1, 2}};
    texec::KVCacheEvent kvCacheEvent(1, kvCacheCreatedData, 32);
    auto kvCacheEvent2 = serializeDeserialize(kvCacheEvent);
    compareKvCacheEvents(kvCacheEvent, kvCacheEvent2);
}

// Test for KVCacheEvent with KVCacheRemovedData
TEST(SerializeUtilsTest, KVCacheRemovedEvents)
{
    texec::KVCacheEvent kvCacheEvent(1, texec::KVCacheRemovedData{{3, 4}}, 32);
    auto kvCacheEvent2 = serializeDeserialize(kvCacheEvent);
    compareKvCacheEvents(kvCacheEvent, kvCacheEvent2);
}

// Test for KVCacheEvent with KVCacheStoredData
TEST(SerializeUtilsTest, KVCacheStoredEvent)
{
    auto storedBlockData1 = texec::KVCacheStoredBlockData(77, {{1, 2}, {3, 4}, {5, 6}}, 88, 0, 99);
    auto storedBlockData2 = texec::KVCacheStoredBlockData(99, {{11, 12}, {3, 4}, {15, 6}}, 77, 1, 101);

    texec::KVCacheStoredData kvCacheStoredData{177, {storedBlockData1, storedBlockData2}};
    texec::KVCacheEvent kvCacheEvent(1, kvCacheStoredData, 32);
    auto kvCacheEvent2 = serializeDeserialize(kvCacheEvent);
    compareKvCacheEvents(kvCacheEvent, kvCacheEvent2);
}

// Test for KVCacheEvent with KVCacheUpdatedData
TEST(SerializeUtilsTest, KVCacheUpdatedEvent)
{
    texec::KVCacheEventDiff<texec::SizeType32> diff{0, 1};
    texec::KVCacheEventDiff<texec::SizeType32> diff2{90, 99};
    texec::KVCacheUpdatedData kvCacheUpdatedData(999, diff, diff2);
    texec::KVCacheEvent kvCacheEvent(1, kvCacheUpdatedData, 32);
    auto kvCacheEvent2 = serializeDeserialize(kvCacheEvent);
    compareKvCacheEvents(kvCacheEvent, kvCacheEvent2);
}

TEST(SerializeUtilsTest, UniqueToken)
{
    tensorrt_llm::runtime::UniqueToken token{1, 2};
    auto token2 = serializeDeserialize(token);
    EXPECT_EQ(token.tokenId, token2.tokenId);
    EXPECT_EQ(token.tokenExtraId, token2.tokenExtraId);
}

TEST(SerializeUtilsTest, UniqueTokenVector)
{
    std::vector<tensorrt_llm::runtime::UniqueToken> tokens{{1, 2}, {3, 4}, {5, 6}};
    auto tokens2 = serializeDeserialize(tokens);
    EXPECT_EQ(tokens.size(), tokens2.size());
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        EXPECT_EQ(tokens[i].tokenId, tokens2[i].tokenId);
        EXPECT_EQ(tokens[i].tokenExtraId, tokens2[i].tokenExtraId);
    }
}

TEST(SerializeUtilsTest, MethodReturnType)
{
    struct S
    {
        void foo() const;
        [[nodiscard]] int bar() const;
        [[nodiscard]] float const& baz() const;
    };

    static_assert(std::is_same_v<void, su::method_return_type_t<decltype(&S::foo)>>);
    static_assert(std::is_same_v<int, su::method_return_type_t<decltype(&S::bar)>>);
    static_assert(std::is_same_v<float const&, su::method_return_type_t<decltype(&S::baz)>>);
}

TEST(SerializeUtilsTest, CacheTransceiverConfig)
{
    texec::CacheTransceiverConfig cacheTransceiverConfig(
        tensorrt_llm::executor::CacheTransceiverConfig::BackendType::UCX, 1024);
    auto cacheTransceiverConfig2 = serializeDeserialize(cacheTransceiverConfig);
    EXPECT_EQ(cacheTransceiverConfig.getBackendType(), cacheTransceiverConfig2.getBackendType());
    EXPECT_EQ(cacheTransceiverConfig.getMaxTokensInBuffer(), cacheTransceiverConfig2.getMaxTokensInBuffer());
}

TEST(SerializeUtilsTest, BlockKeyBasic)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;

    VecUniqueTokens uniqueTokens{UniqueToken{1, 0}, UniqueToken{2, 0}, UniqueToken{3, 0}};
    BlockKey key(false, std::nullopt, uniqueTokens, {});

    testSerializeDeserialize(key);
}

TEST(SerializeUtilsTest, BlockKeyWithExtras)
{
    using namespace tensorrt_llm::batch_manager::kv_cache_manager;

    // Prepare multimodal extra keys
    std::array<uint8_t, 32> h1{};
    std::array<uint8_t, 32> h2{};
    for (size_t i = 0; i < h1.size(); ++i)
    {
        h1[i] = static_cast<uint8_t>(i);
        h2[i] = static_cast<uint8_t>(255 - i);
    }
    std::vector<MmKey> extraKeys{{h1, SizeType32{0}}, {h2, SizeType32{5}}};

    VecUniqueTokens uniqueTokens{UniqueToken{10, 100}, UniqueToken{20, 200}};
    std::optional<LoraTaskIdType> loraTaskId = LoraTaskIdType{42};

    // Note: cacheSaltID is intentionally not set since it is not serialized
    BlockKey key(true, loraTaskId, uniqueTokens, extraKeys);

    testSerializeDeserialize(key);
}

// Connection notification tests
namespace kv_cache = tensorrt_llm::executor::kv_cache;

template <typename T>
T serializeDeserializeNotification(T const& val)
{
    auto size = T::serializedSize(val);
    std::ostringstream oss;
    T::serialize(val, oss);
    EXPECT_EQ(oss.str().size(), size);

    std::istringstream iss(oss.str());
    return T::deserialize(iss);
}

TEST(SerializeUtilsTest, RequestAndBufferInfo)
{
    // Test with all fields populated
    {
        kv_cache::RequestAndBufferInfo original{"testAgent", "127.0.0.1:8080",
            tensorrt_llm::batch_manager::RequestInfo{}, kv_cache::MemoryDesc{nullptr, 1024, 0},
            std::make_optional<std::string>("metadata"), 1};

        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_EQ(original.mAgentName, deserialized.mAgentName);
        EXPECT_EQ(original.mAddress, deserialized.mAddress);
        EXPECT_EQ(original.mRequestInfo.getRequestId(), deserialized.mRequestInfo.getRequestId());
        EXPECT_EQ(original.mBufferDesc.getAddr(), deserialized.mBufferDesc.getAddr());
        EXPECT_EQ(original.mBufferDesc.getLen(), deserialized.mBufferDesc.getLen());
        EXPECT_EQ(original.mBufferDesc.getDeviceId(), deserialized.mBufferDesc.getDeviceId());
        EXPECT_EQ(original.mMetadata, deserialized.mMetadata);
        EXPECT_EQ(original.mValidConnectionIdx, deserialized.mValidConnectionIdx);
    }

    // Test with nullopt metadata
    {
        kv_cache::RequestAndBufferInfo original{"testAgent2", "192.168.1.1:9090",
            tensorrt_llm::batch_manager::RequestInfo{}, kv_cache::MemoryDesc{nullptr, 512, 0}, std::nullopt, 2};

        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_EQ(original.mAgentName, deserialized.mAgentName);
        EXPECT_EQ(original.mAddress, deserialized.mAddress);
        EXPECT_EQ(original.mRequestInfo.getRequestId(), deserialized.mRequestInfo.getRequestId());
        EXPECT_EQ(original.mBufferDesc.getAddr(), deserialized.mBufferDesc.getAddr());
        EXPECT_EQ(original.mBufferDesc.getLen(), deserialized.mBufferDesc.getLen());
        EXPECT_EQ(original.mBufferDesc.getDeviceId(), deserialized.mBufferDesc.getDeviceId());
        EXPECT_EQ(original.mMetadata, deserialized.mMetadata);
        EXPECT_EQ(original.mValidConnectionIdx, deserialized.mValidConnectionIdx);
    }
}

TEST(SerializeUtilsTest, ReadySignalInfo)
{
    // Test with isReady = true
    {
        kv_cache::ReadySignalInfo original{"agent1", kv_cache::DataContext{12345}, true};

        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_EQ(original.mAgentName, deserialized.mAgentName);
        EXPECT_EQ(original.mContext.getTag(), deserialized.mContext.getTag());
        EXPECT_EQ(original.mIsReady, deserialized.mIsReady);
    }

    // Test with isReady = false
    {
        kv_cache::ReadySignalInfo original{"agent2", kv_cache::DataContext{67890}, false};

        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_EQ(original.mAgentName, deserialized.mAgentName);
        EXPECT_EQ(original.mContext.getTag(), deserialized.mContext.getTag());
        EXPECT_EQ(original.mIsReady, deserialized.mIsReady);
    }

    // Test with different context tags
    {
        kv_cache::ReadySignalInfo original{"agent3", kv_cache::DataContext{0}, true};

        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_EQ(original.mAgentName, deserialized.mAgentName);
        EXPECT_EQ(original.mContext.getTag(), deserialized.mContext.getTag());
        EXPECT_EQ(original.mIsReady, deserialized.mIsReady);
    }
}

TEST(SerializeUtilsTest, NotificationSyncInfo)
{
    // Test basic functionality
    {
        kv_cache::NotificationSyncInfo original{"syncAgent", kv_cache::DataContext{54321}};

        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_EQ(original.mAgentName, deserialized.mAgentName);
        EXPECT_EQ(original.mContext.getTag(), deserialized.mContext.getTag());
    }

    // Test with different agent names and context tags
    {
        kv_cache::NotificationSyncInfo original{"anotherAgent", kv_cache::DataContext{98765}};

        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_EQ(original.mAgentName, deserialized.mAgentName);
        EXPECT_EQ(original.mContext.getTag(), deserialized.mContext.getTag());
    }

    // Test with empty agent name
    {
        kv_cache::NotificationSyncInfo original{"", kv_cache::DataContext{11111}};

        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_EQ(original.mAgentName, deserialized.mAgentName);
        EXPECT_EQ(original.mContext.getTag(), deserialized.mContext.getTag());
    }
}

TEST(SerializeUtilsTest, NotificationInfo)
{
    // Test with RequestAndBufferInfo variant
    {
        kv_cache::RequestAndBufferInfo requestInfo{"testAgent", "127.0.0.1:8080",
            tensorrt_llm::batch_manager::RequestInfo{}, kv_cache::MemoryDesc{nullptr, 1024, 0},
            std::make_optional<std::string>("test_metadata"), 1};

        kv_cache::NotificationInfo original{requestInfo};
        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_TRUE(std::holds_alternative<kv_cache::RequestAndBufferInfo>(deserialized.mInfo));
        auto deserializedRequestInfo = std::get<kv_cache::RequestAndBufferInfo>(deserialized.mInfo);
        EXPECT_EQ(requestInfo.mAgentName, deserializedRequestInfo.mAgentName);
        EXPECT_EQ(requestInfo.mAddress, deserializedRequestInfo.mAddress);
        EXPECT_EQ(requestInfo.mRequestInfo.getRequestId(), deserializedRequestInfo.mRequestInfo.getRequestId());
        EXPECT_EQ(requestInfo.mMetadata, deserializedRequestInfo.mMetadata);
        EXPECT_EQ(requestInfo.mValidConnectionIdx, deserializedRequestInfo.mValidConnectionIdx);
    }

    // Test with NotificationSyncInfo variant
    {
        kv_cache::NotificationSyncInfo syncInfo{"syncAgent", kv_cache::DataContext{12345}};

        kv_cache::NotificationInfo original{syncInfo};
        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_TRUE(std::holds_alternative<kv_cache::NotificationSyncInfo>(deserialized.mInfo));
        auto deserializedSyncInfo = std::get<kv_cache::NotificationSyncInfo>(deserialized.mInfo);
        EXPECT_EQ(syncInfo.mAgentName, deserializedSyncInfo.mAgentName);
        EXPECT_EQ(syncInfo.mContext.getTag(), deserializedSyncInfo.mContext.getTag());
    }

    // Test with ReadySignalInfo variant
    {
        kv_cache::ReadySignalInfo readyInfo{"readyAgent", kv_cache::DataContext{67890}, true};

        kv_cache::NotificationInfo original{readyInfo};
        auto deserialized = serializeDeserializeNotification(original);

        EXPECT_TRUE(std::holds_alternative<kv_cache::ReadySignalInfo>(deserialized.mInfo));
        auto deserializedReadyInfo = std::get<kv_cache::ReadySignalInfo>(deserialized.mInfo);
        EXPECT_EQ(readyInfo.mAgentName, deserializedReadyInfo.mAgentName);
        EXPECT_EQ(readyInfo.mContext.getTag(), deserializedReadyInfo.mContext.getTag());
        EXPECT_EQ(readyInfo.mIsReady, deserializedReadyInfo.mIsReady);
    }
}
