/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorrt_llm/common/profiler.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace tc = tensorrt_llm::common;

namespace
{

struct Observation
{
    std::string name;
    std::string metadata;
    int starts{0};
    int ends{0};
};

Observation& observation()
{
    static thread_local Observation value;
    return value;
}

std::unique_ptr<at::ObserverContext> observeStart(at::RecordFunction const& record)
{
    auto& value = observation();
    value.name = record.name();
    ++value.starts;
    if (!record.inputs().empty())
    {
        value.metadata = record.inputs().front().toStringRef();
    }
    return nullptr;
}

std::unique_ptr<at::ObserverContext> observeName(at::RecordFunction const& record)
{
    auto& value = observation();
    value.name = record.name();
    ++value.starts;
    return nullptr;
}

void observeEnd(at::RecordFunction const&, at::ObserverContext*)
{
    ++observation().ends;
}

class CallbackGuard
{
public:
    explicit CallbackGuard(bool inputs = true)
        : mHandle{at::addThreadLocalCallback(at::RecordFunctionCallback(inputs ? observeStart : observeName, observeEnd)
                                                 .needsInputs(inputs)
                                                 .scopes({at::RecordScope::USER_SCOPE}))}
    {
        observation() = {};
    }

    ~CallbackGuard()
    {
        at::removeCallback(mHandle);
    }

    CallbackGuard(CallbackGuard const&) = delete;
    CallbackGuard& operator=(CallbackGuard const&) = delete;

private:
    at::CallbackHandle mHandle;
};

#if !defined(_WIN32)
void checkSelection(char const* selection, std::vector<tc::ProfilerRegion> const& expected)
{
    if (selection == nullptr)
    {
        unsetenv("TLLM_PROFILE_CUSTOM_REGIONS");
    }
    else
    {
        setenv("TLLM_PROFILE_CUSTOM_REGIONS", selection, 1);
    }
    tc::initializeProfiler();
    for (auto region : {tc::ProfilerRegion::kWorkflow, tc::ProfilerRegion::kMpi, tc::ProfilerRegion::kNccl,
             tc::ProfilerRegion::kNanobind, tc::ProfilerRegion::kKv, tc::ProfilerRegion::kMoe})
    {
        bool selected = false;
        for (auto candidate : expected)
        {
            selected = selected || candidate == region;
        }
        if (tc::customRegionEnabled(region) != selected)
        {
            std::exit(1);
        }
    }
    setenv("TLLM_PROFILE_CUSTOM_REGIONS", "unknown", 1);
    tc::initializeProfiler();
    std::exit(0);
}
#endif

} // namespace

#if !defined(_WIN32)
TEST(ProfilerStartupDeathTest, DefaultsAndOptInAreCached)
{
    EXPECT_EXIT(checkSelection(nullptr,
                    {tc::ProfilerRegion::kWorkflow, tc::ProfilerRegion::kMpi, tc::ProfilerRegion::kNccl,
                        tc::ProfilerRegion::kNanobind}),
        ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(checkSelection("workflow,moe", {tc::ProfilerRegion::kWorkflow, tc::ProfilerRegion::kMoe}),
        ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(checkSelection(" workflow , kv ", {tc::ProfilerRegion::kWorkflow, tc::ProfilerRegion::kKv}),
        ::testing::ExitedWithCode(0), "");
    EXPECT_EXIT(checkSelection("none", {}), ::testing::ExitedWithCode(0), "");
}

TEST(ProfilerStartupDeathTest, InvalidSelectionFailsAtExplicitStartupAndIsCached)
{
    for (auto const* selection : {"", " ", "workflow,", ",workflow", "workflow,,mpi", "unknown", "mpi,mpi", "none,kv"})
    {
        EXPECT_EXIT(
            {
                setenv("TLLM_PROFILE_CUSTOM_REGIONS", selection, 1);
                bool failed = false;
                try
                {
                    tc::initializeProfiler();
                }
                catch (std::invalid_argument const& error)
                {
                    failed = std::string{error.what()}.find("TLLM_PROFILE_CUSTOM_REGIONS") != std::string::npos;
                }
                setenv("TLLM_PROFILE_CUSTOM_REGIONS", "workflow", 1);
                try
                {
                    tc::initializeProfiler();
                }
                catch (std::invalid_argument const&)
                {
                    std::exit(failed ? 0 : 1);
                }
                std::exit(1);
            },
            ::testing::ExitedWithCode(0), "");
    }
}

TEST(ProfilerStartupDeathTest, ProbesBeforeStartupNeverParseInvalidConfiguration)
{
    EXPECT_EXIT(
        {
            setenv("TLLM_PROFILE_CUSTOM_REGIONS", "unknown", 1);
            CallbackGuard callback;
            tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
            scope.startWithLazyName([]() -> std::string { throw std::logic_error("name evaluated"); },
                [](tc::ProfilerMetadata&) { throw std::logic_error("metadata evaluated"); });
            std::exit(observation().starts == 0 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
}

TEST(ProfilerStartupDeathTest, NoneSelectionNeverEvaluatesFactoriesOrEmitsScopes)
{
    EXPECT_EXIT(
        {
            setenv("TLLM_PROFILE_CUSTOM_REGIONS", "none", 1);
            tc::initializeProfiler();
            CallbackGuard callback;
            tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
            scope.startWithLazyName([]() -> std::string { throw std::logic_error("name evaluated"); },
                [](tc::ProfilerMetadata&) { throw std::logic_error("metadata evaluated"); });
            std::exit(observation().starts == 0 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
}
#endif

TEST(ProfilerMetadataTest, NativeGoldenIncludesUnicodeEscapingSignedIntegersAndEmptyLists)
{
    tc::ProfilerMetadata metadata{"workflow"};
    metadata.add("quoted'key", "slash\\line\n\t'雪");
    metadata.add("rank", std::int64_t{-7});
    metadata.add("ids", std::vector<std::int64_t>{});
    EXPECT_EQ(
        std::move(metadata).take(), "TRTLLM18:workflow{'quoted\\'key':'slash\\\\line\\n\\t\\'雪','rank':-7,'ids':[]}");
}

TEST(ProfilerMetadataTest, ByteLengthsAndControlCharactersAreLiteralCompatible)
{
    tc::ProfilerMetadata metadata{"区域"};
    metadata.add("control", std::string_view{"\0\x01\r", 3});
    metadata.add("ids", std::vector<std::int64_t>{-7, 0, 9223372036854775807LL});
    EXPECT_EQ(std::move(metadata).take(), "TRTLLM16:区域{'control':'\\x00\\x01\\r','ids':[-7,0,9223372036854775807]}");
}

#if TRTLLM_ENABLE_NCCL_PROFILING
TEST(ProfilerMetadataTest, NcclFieldsPreserveOrderedRanksAndLogicalPayload)
{
    tc::ProfilerMetadata metadata{"nccl"};
    tc::addNcclMetadata(metadata, "all_reduce", std::vector<std::int64_t>{3, 7}, 7, "tp", "collective", 4096);
    metadata.add("nccl.reduce_op", "sum");
    metadata.add("nccl.selected_strategy", "NCCL");
    EXPECT_EQ(std::move(metadata).take(),
        "TRTLLM14:nccl{'nccl.operation':'all_reduce','nccl.group':[3,7],'nccl.rank':7,"
        "'nccl.parallelism_dimension':'tp','nccl.kind':'collective','nccl.logical_bytes':4096,"
        "'nccl.reduce_op':'sum','nccl.selected_strategy':'NCCL'}");
}

TEST(ProfilerMetadataTest, NcclPointToPointPreservesEmptyPayloadAndUnknownDimension)
{
    tc::ProfilerMetadata metadata{"nccl"};
    tc::addNcclMetadata(metadata, "recv", std::vector<std::int64_t>{0, 1}, 1, "unknown", "p2p", 0);
    metadata.add("nccl.src_rank", std::int64_t{0});
    metadata.add("nccl.dst_rank", std::int64_t{1});
    EXPECT_EQ(std::move(metadata).take(),
        "TRTLLM14:nccl{'nccl.operation':'recv','nccl.group':[0,1],'nccl.rank':1,"
        "'nccl.parallelism_dimension':'unknown','nccl.kind':'p2p','nccl.logical_bytes':0,"
        "'nccl.src_rank':0,'nccl.dst_rank':1}");
}
#endif

class ProfilerScopeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        tc::initializeProfiler();
    }
};

TEST_F(ProfilerScopeTest, ActiveCallbackReceivesOwnedMetadataAndEndsOnce)
{
    CallbackGuard callback;
    int names = 0;
    int fields = 0;
    {
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
        scope.startWithLazyName(
            [&]() -> std::string
            {
                ++names;
                return "foundation";
            },
            [&](tc::ProfilerMetadata& metadata)
            {
                ++fields;
                metadata.add("rank", std::int64_t{-7});
            });
        scope.start("duplicate", [](tc::ProfilerMetadata&) { FAIL() << "scope started twice"; });
        EXPECT_EQ(observation().name, "foundation");
        EXPECT_EQ(observation().metadata, "TRTLLM18:workflow{'rank':-7}");
    }
    EXPECT_EQ(names, 1);
    EXPECT_EQ(fields, 1);
    EXPECT_EQ(observation().starts, 1);
    EXPECT_EQ(observation().ends, 1);
}

TEST_F(ProfilerScopeTest, CallbackWithoutInputsDoesNotInspectMetadata)
{
    CallbackGuard callback{false};
    {
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
        scope.start("name-only", [](tc::ProfilerMetadata&) { FAIL() << "unneeded metadata evaluated"; });
    }
    EXPECT_EQ(observation().name, "name-only");
    EXPECT_EQ(observation().starts, 1);
    EXPECT_EQ(observation().ends, 1);
    EXPECT_TRUE(observation().metadata.empty());
}

TEST_F(ProfilerScopeTest, NcclBuildGateAndInputDemandKeepExtractionLazy)
{
    int fields{0};
    {
        CallbackGuard callback{false};
#if TRTLLM_ENABLE_NCCL_PROFILING
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kNccl};
        scope.startWithLazyName(
            []() -> std::string { return "trtllm::nccl_metadata"; }, [&](tc::ProfilerMetadata&) { ++fields; });
        EXPECT_EQ(observation().starts, 1);
#else
        EXPECT_EQ(observation().starts, 0);
#endif
        EXPECT_TRUE(observation().metadata.empty());
    }
    EXPECT_EQ(fields, 0);
#if TRTLLM_ENABLE_NCCL_PROFILING
    tc::RecordFunctionWithMetadata inactive{tc::ProfilerRegion::kNccl};
    inactive.startWithLazyName(
        []() -> std::string { return "trtllm::nccl_metadata"; }, [&](tc::ProfilerMetadata&) { ++fields; });
    EXPECT_EQ(fields, 0);
    CallbackGuard callback;
    {
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kNccl};
        scope.startWithLazyName([]() -> std::string { return "trtllm::nccl_metadata"; },
            [&](tc::ProfilerMetadata& metadata)
            {
                ++fields;
                tc::addNcclMetadata(metadata, "all_gather", {0, 1}, 0, "unknown", "collective", 32);
            });
    }
    EXPECT_EQ(fields, 1);
    EXPECT_EQ(observation().starts, 1);
    EXPECT_EQ(observation().ends, 1);
    EXPECT_EQ(observation().name, "trtllm::nccl_metadata");
    EXPECT_EQ(observation().metadata,
        "TRTLLM14:nccl{'nccl.operation':'all_gather','nccl.group':[0,1],'nccl.rank':0,"
        "'nccl.parallelism_dimension':'unknown','nccl.kind':'collective','nccl.logical_bytes':32}");
#endif
}

#if TRTLLM_ENABLE_NCCL_PROFILING
TEST_F(ProfilerScopeTest, NcclInactiveScopeNeverConstructsItsOwningName)
{
    int names{0};
    int fields{0};
    tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kNccl};
    scope.startWithLazyName(
        [&]() -> std::string
        {
            ++names;
            throw std::bad_alloc{};
        },
        [&](tc::ProfilerMetadata&) { ++fields; });
    EXPECT_EQ(names, 0);
    EXPECT_EQ(fields, 0);
}

TEST_F(ProfilerScopeTest, NcclThrowingNameDoesNotPreventBodyOrMaskItsException)
{
    CallbackGuard callback;
    int names{0};
    int fields{0};
    bool bodyRan{false};
    try
    {
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kNccl};
        scope.startWithLazyName(
            [&]() -> std::string
            {
                ++names;
                // Deterministic factory failure, without modifying the process allocator.
                throw std::bad_alloc{};
            },
            [&](tc::ProfilerMetadata&) { ++fields; });
        bodyRan = true;
        throw std::logic_error("original operation failure");
    }
    catch (std::logic_error const& error)
    {
        EXPECT_STREQ(error.what(), "original operation failure");
    }
    EXPECT_TRUE(bodyRan);
    EXPECT_EQ(names, 1);
    EXPECT_EQ(fields, 0);
    EXPECT_EQ(observation().starts, 0);
    EXPECT_EQ(observation().ends, 0);
}
#endif

TEST_F(ProfilerScopeTest, InactiveAndUnselectedScopesDoNotEvaluateFactories)
{
    int nameCalls = 0;
    int metadataCalls = 0;
    auto forbiddenName = [&]() -> std::string
    {
        ++nameCalls;
        return "inactive-name";
    };
    auto forbiddenMetadata = [&](tc::ProfilerMetadata&) { ++metadataCalls; };
    tc::RecordFunctionWithMetadata inactive{tc::ProfilerRegion::kWorkflow};
    inactive.startWithLazyName(forbiddenName, forbiddenMetadata);
    CallbackGuard callback;
    tc::RecordFunctionWithMetadata disabled{tc::ProfilerRegion::kKv};
    disabled.startWithLazyName(forbiddenName, forbiddenMetadata);
    EXPECT_EQ(nameCalls, 0);
    EXPECT_EQ(metadataCalls, 0);
    EXPECT_EQ(observation().starts, 0);
}

TEST_F(ProfilerScopeTest, MetadataFailureDoesNotPreventBodyOrMaskItsException)
{
    CallbackGuard callback;
    bool bodyRan = false;
    try
    {
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
        scope.start("failed-metadata", [](tc::ProfilerMetadata&) { throw std::runtime_error("metadata failed"); });
        bodyRan = true;
        throw std::logic_error("original body failure");
    }
    catch (std::logic_error const& error)
    {
        EXPECT_EQ(std::string{error.what()}, "original body failure");
    }
    EXPECT_TRUE(bodyRan);
    EXPECT_EQ(observation().starts, 0);
}
