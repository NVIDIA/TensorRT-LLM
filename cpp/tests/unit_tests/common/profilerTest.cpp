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

#if defined(__linux__) && TRTLLM_ENABLE_NANOBIND_PROFILING
#include <nanobind/nanobind.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__) && TRTLLM_ENABLE_NANOBIND_PROFILING
NB_MODULE(profiler_test_bindings, module)
{
    module.def("answer", [] { return 42; });
}
#endif

#if defined(__linux__)
namespace
{
thread_local bool failNextProfilerAllocation{false};
}

//! Inject one allocation failure from shared LibTorch in this test executable only.
void* operator new(std::size_t size)
{
    if (failNextProfilerAllocation)
    {
        failNextProfilerAllocation = false;
        throw std::bad_alloc{};
    }
    size = std::max(size, std::size_t{1});
    while (true)
    {
        if (auto* memory = std::malloc(size))
        {
            return memory;
        }
        auto const handler = std::get_new_handler();
        if (!handler)
        {
            throw std::bad_alloc{};
        }
        handler();
    }
}

void operator delete(void* memory) noexcept
{
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept
{
    std::free(memory);
}
#endif

namespace tc = tensorrt_llm::common;

namespace
{

struct Observation
{
    std::string name;
    std::string metadata;
    bool inputsEmpty{true};
    int starts{0};
    int ends{0};
};

Observation& observation()
{
    static thread_local Observation value;
    return value;
}

//! Capture an entered scope and its first string input when callbacks request metadata.
std::unique_ptr<at::ObserverContext> observeStart(at::RecordFunction const& record)
{
    auto& value = observation();
    value.name = record.name();
    ++value.starts;
    value.inputsEmpty = record.inputs().empty();
    if (!value.inputsEmpty)
    {
        value.metadata = record.inputs().front().toStringRef();
    }
    return nullptr;
}

void observeEnd(at::RecordFunction const&, at::ObserverContext*)
{
    ++observation().ends;
}

class CallbackGuard
{
public:
    //! Install a thread-local user-scope callback and clear observations from earlier tests.
    explicit CallbackGuard(bool inputs = true)
        : mHandle{at::addThreadLocalCallback(at::RecordFunctionCallback(observeStart, observeEnd)
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
//! Check selection in a child process and verify that later environment changes are ignored.
void checkSelection(nlohmann::json const& selection)
{
    if (selection["value"].is_null())
    {
        unsetenv("TLLM_PROFILE_CUSTOM_REGIONS");
    }
    else
    {
        auto const value = selection["value"].get<std::string>();
        setenv("TLLM_PROFILE_CUSTOM_REGIONS", value.c_str(), 1);
    }
    auto const error = tc::initializeProfilerRegions();
    if (error.has_value() == selection["valid"].get<bool>()
        || (error && error->find("TLLM_PROFILE_CUSTOM_REGIONS") == std::string::npos))
    {
        std::exit(1);
    }
    std::pair<tc::ProfilerRegion, char const*> const regions[]{{tc::ProfilerRegion::kWorkflow, "workflow"},
        {tc::ProfilerRegion::kMpi, "mpi"}, {tc::ProfilerRegion::kNccl, "nccl"},
        {tc::ProfilerRegion::kNanobind, "nanobind"}, {tc::ProfilerRegion::kKv, "kv"},
        {tc::ProfilerRegion::kMoe, "moe"}};
    auto const expected = selection["expected"].get<std::vector<std::string>>();
    for (auto const& [region, name] : regions)
    {
        bool const selected = std::find(expected.begin(), expected.end(), name) != expected.end();
        if (tc::customRegionEnabled(region) != selected)
        {
            std::exit(1);
        }
    }
    setenv("TLLM_PROFILE_CUSTOM_REGIONS", "unknown", 1);
    std::exit(tc::initializeProfilerRegions() == error ? 0 : 1);
}
#endif

} // namespace

#if !defined(_WIN32)
//! Check both language implementations against the same startup selector cases, including cached diagnostics.
TEST(ProfilerStartupDeathTest, SharedSelectorCasesAreCached)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    std::ifstream fixture{TRTLLM_PROFILER_SELECTOR_CASES};
    ASSERT_TRUE(fixture.is_open());
    auto const cases = nlohmann::json::parse(fixture).at("cases");
    ASSERT_FALSE(cases.empty());
    for (auto const& selection : cases)
    {
        SCOPED_TRACE(selection.dump());
        EXPECT_EXIT(checkSelection(selection), ::testing::ExitedWithCode(0), "");
    }
}

//! Check that repeated metadata failures warn once while the original operation exception survives.
TEST(ProfilerStartupDeathTest, ObservationFailureWarnsOnceAndPreservesOperationException)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            unsetenv("TLLM_PROFILE_CUSTOM_REGIONS");
            if (tc::initializeProfilerRegions())
            {
                std::exit(1);
            }
            CallbackGuard callback;
            bool bodyRan{false};
            for (int index = 0; index < 2; ++index)
            {
                try
                {
                    tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
                    scope.startWithLazyName([]() -> std::string { return "failed"; },
                        [](tc::ProfilerMetadata&) { throw std::runtime_error("factory failed"); });
                    bodyRan = true;
                    throw std::logic_error("body failed");
                }
                catch (std::logic_error const& error)
                {
                    if (std::string{error.what()} != "body failed")
                    {
                        std::exit(1);
                    }
                }
            }
            std::exit(bodyRan && observation().starts == 0 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "^TensorRT-LLM profiler warning: observation omitted: factory failed\n$");
}

//! Verify uninitialized probes neither parse configuration nor evaluate their factories.
TEST(ProfilerStartupDeathTest, ProbesBeforeStartupNeverParseInvalidConfiguration)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            setenv("TLLM_PROFILE_CUSTOM_REGIONS", "unknown", 1);
            CallbackGuard callback;
            int nameCalls = 0;
            int metadataCalls = 0;
            tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
            scope.startWithLazyName(
                [&]() -> std::string
                {
                    ++nameCalls;
                    return "uninitialized";
                },
                [&](tc::ProfilerMetadata&) { ++metadataCalls; });
            std::exit(nameCalls == 0 && metadataCalls == 0 && observation().starts == 0 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
}

//! Verify an explicit none selection suppresses callbacks and both factories.
TEST(ProfilerStartupDeathTest, NoneSelectionNeverEvaluatesFactoriesOrEmitsScopes)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            setenv("TLLM_PROFILE_CUSTOM_REGIONS", "none", 1);
            ASSERT_FALSE(tc::initializeProfilerRegions().has_value());
            CallbackGuard callback;
            int nameCalls = 0;
            int metadataCalls = 0;
            tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
            scope.startWithLazyName(
                [&]() -> std::string
                {
                    ++nameCalls;
                    return "unselected";
                },
                [&](tc::ProfilerMetadata&) { ++metadataCalls; });
            std::exit(nameCalls == 0 && metadataCalls == 0 && observation().starts == 0 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
}

#if defined(__linux__)
//! A failed global-callback snapshot must omit observation without aborting the operation.
TEST(ProfilerStartupDeathTest, CallbackProbeAllocationFailurePreservesOperationException)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            unsetenv("TLLM_PROFILE_CUSTOM_REGIONS");
            if (tc::initializeProfilerRegions())
            {
                std::exit(1);
            }
            auto const handle = at::addGlobalCallback(at::RecordFunctionCallback(observeStart, observeEnd)
                                                          .needsInputs(false)
                                                          .scopes({at::RecordScope::USER_SCOPE}));
            std::logic_error const bodyError{"original body failure"};
            bool bodyRan{false};
            bool bodyExceptionPreserved{false};
            failNextProfilerAllocation = true;
            try
            {
                tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
                scope.startWithLazyName(
                    []() -> std::string { return "failed-probe"; }, [](tc::ProfilerMetadata&) { std::exit(1); });
                bodyRan = true;
                throw bodyError;
            }
            catch (std::logic_error const& error)
            {
                bodyExceptionPreserved = std::string{error.what()} == "original body failure";
            }
            if (failNextProfilerAllocation || !bodyRan || !bodyExceptionPreserved || observation().starts != 0
                || observation().ends != 0)
            {
                std::exit(1);
            }
            {
                tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
                scope.startWithLazyName([]() -> std::string { return "recovered"; }, [](tc::ProfilerMetadata&) {});
            }
            at::removeCallback(handle);
            std::exit(observation().starts == 1 && observation().ends == 1 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "^TensorRT-LLM profiler warning: observation omitted: .+\n$");
}

#if TRTLLM_ENABLE_NANOBIND_PROFILING
//! A callback snapshot failure must leave the nanobind vectorcall result intact.
TEST(ProfilerStartupDeathTest, CallbackProbeAllocationFailurePreservesBindingResult)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(
        {
            setenv("TLLM_PROFILE_CUSTOM_REGIONS", "nanobind", 1);
            ASSERT_FALSE(tc::initializeProfilerRegions().has_value());
            ASSERT_EQ(PyImport_AppendInittab("profiler_test_bindings", PyInit_profiler_test_bindings), 0);
            Py_Initialize();
            nanobind::object answer = nanobind::module_::import_("profiler_test_bindings").attr("answer");
            auto const handle = at::addGlobalCallback(at::RecordFunctionCallback(observeStart, observeEnd)
                                                          .needsInputs(false)
                                                          .scopes({at::RecordScope::USER_SCOPE}));

            failNextProfilerAllocation = true;
            auto result = nanobind::steal<nanobind::object>(PyObject_CallNoArgs(answer.ptr()));
            ASSERT_FALSE(failNextProfilerAllocation);
            ASSERT_TRUE(result.is_valid());
            ASSERT_EQ(PyLong_AsLong(result.ptr()), 42);
            ASSERT_EQ(observation().starts, 0);
            ASSERT_EQ(observation().ends, 0);

            result = nanobind::steal<nanobind::object>(PyObject_CallNoArgs(answer.ptr()));
            ASSERT_TRUE(result.is_valid());
            ASSERT_EQ(PyLong_AsLong(result.ptr()), 42);
            at::removeCallback(handle);
            std::exit(observation().starts == 1 && observation().ends == 1 ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "^TensorRT-LLM profiler warning: observation omitted: .+\n$");
}
#endif
#endif
#endif

//! Compare native encoding of Unicode, escaped strings, signed integers, and empty lists to a golden payload.
TEST(ProfilerMetadataTest, NativeGoldenIncludesUnicodeEscapingSignedIntegersAndEmptyLists)
{
    tc::ProfilerMetadata metadata{"workflow"};
    metadata.add("quoted'雪", "slash\\line\n\t'雪🚀");
    metadata.add("rank", std::int64_t{-7});
    metadata.add("ids", std::vector<std::int64_t>{});
    auto const payload = std::move(metadata).take();
    EXPECT_TRUE(std::all_of(payload.begin(), payload.end(), [](unsigned char value) { return value < 0x80; }));
    EXPECT_EQ(
        payload, R"(TRTLLM1:8:workflow{"quoted'\u96ea":"slash\\line\n\t'\u96ea\ud83d\ude80","rank":-7,"ids":[]})");
}

//! Check UTF-8 region byte lengths, control-byte escapes, and signed integer boundary literals.
TEST(ProfilerMetadataTest, ByteLengthsAndControlCharactersAreJsonCompatible)
{
    tc::ProfilerMetadata metadata{"区域"};
    metadata.add("control", std::string_view{"\0\x01\r", 3});
    metadata.add("ids", std::vector<std::int64_t>{-7, 0, 9223372036854775807LL});
    EXPECT_EQ(
        std::move(metadata).take(), R"(TRTLLM1:6:区域{"control":"\u0000\u0001\r","ids":[-7,0,9223372036854775807]})");
}

//! Reject malformed UTF-8 instead of emitting an invalid metadata payload.
TEST(ProfilerMetadataTest, InvalidUtf8ThrowsDuringEncoding)
{
    tc::ProfilerMetadata metadata{"workflow"};
    metadata.add("invalid", std::string_view{"\xC3\x28", 2});
    EXPECT_THROW(std::move(metadata).take(), nlohmann::json::type_error);
}

class ProfilerScopeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Startup death tests re-execute with fresh state instead of inheriting this cached selection.
#if defined(_WIN32)
        ASSERT_EQ(_putenv_s("TLLM_PROFILE_CUSTOM_REGIONS", ""), 0);
#else
        ASSERT_EQ(unsetenv("TLLM_PROFILE_CUSTOM_REGIONS"), 0);
#endif
        ASSERT_FALSE(tc::initializeProfilerRegions().has_value());
    }
};

//! Verify active scopes own their metadata, evaluate factories once, and end exactly once.
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
        scope.startWithLazyName([]() -> std::string { return "duplicate"; },
            [](tc::ProfilerMetadata&) { FAIL() << "scope started twice"; });
        EXPECT_EQ(observation().name, "foundation");
        EXPECT_EQ(observation().metadata, R"(TRTLLM1:8:workflow{"rank":-7})");
    }
    EXPECT_EQ(names, 1);
    EXPECT_EQ(fields, 1);
    EXPECT_EQ(observation().starts, 1);
    EXPECT_EQ(observation().ends, 1);
}

//! Verify a name-only callback records the scope without evaluating the metadata factory.
TEST_F(ProfilerScopeTest, CallbackWithoutInputsDoesNotInspectMetadata)
{
    CallbackGuard callback{false};
    {
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
        scope.startWithLazyName([]() -> std::string { return "name-only"; },
            [](tc::ProfilerMetadata&) { FAIL() << "unneeded metadata evaluated"; });
    }
    EXPECT_EQ(observation().name, "name-only");
    EXPECT_EQ(observation().starts, 1);
    EXPECT_EQ(observation().ends, 1);
    EXPECT_TRUE(observation().inputsEmpty);
}

#if TRTLLM_ENABLE_NCCL_PROFILING
//! Evaluate NCCL fields only for an active callback that requests inputs.
TEST_F(ProfilerScopeTest, NcclMetadataExtractionIsLazy)
{
    int fields{0};
    {
        CallbackGuard callback{false};
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kNccl};
        scope.startWithLazyName(
            []() -> std::string { return "trtllm::nccl_metadata"; }, [&](tc::ProfilerMetadata&) { ++fields; });
        EXPECT_EQ(observation().starts, 1);
        EXPECT_TRUE(observation().inputsEmpty);
    }
    EXPECT_EQ(fields, 0);
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
                metadata.add("nccl.operation", "all_gather");
            });
    }
    EXPECT_EQ(fields, 1);
    EXPECT_EQ(observation().starts, 1);
    EXPECT_EQ(observation().ends, 1);
    EXPECT_EQ(observation().name, "trtllm::nccl_metadata");
    EXPECT_EQ(observation().metadata, R"(TRTLLM1:4:nccl{"nccl.operation":"all_gather"})");
}

//! Verify inactive NCCL probes skip both name allocation and metadata extraction.
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

//! Check that a throwing name factory leaves the operation and its exception unchanged.
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

//! Verify missing callbacks and excluded regions both bypass name and metadata factories.
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

//! Check that metadata failure suppresses observation while preserving the body's exception.
TEST_F(ProfilerScopeTest, MetadataFailureDoesNotPreventBodyOrMaskItsException)
{
    CallbackGuard callback;
    bool bodyRan = false;
    try
    {
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
        scope.startWithLazyName([]() -> std::string { return "failed-metadata"; },
            [](tc::ProfilerMetadata&) { throw std::runtime_error("metadata failed"); });
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

//! Check that name failure bypasses metadata and callbacks while preserving the body's exception.
TEST_F(ProfilerScopeTest, NameFailureDoesNotPreventBodyOrMaskItsException)
{
    CallbackGuard callback;
    int nameCalls = 0;
    int metadataCalls = 0;
    bool bodyRan = false;
    try
    {
        tc::RecordFunctionWithMetadata scope{tc::ProfilerRegion::kWorkflow};
        scope.startWithLazyName(
            [&]() -> std::string
            {
                ++nameCalls;
                throw std::runtime_error("name failed");
            },
            [&](tc::ProfilerMetadata&) { ++metadataCalls; });
        bodyRan = true;
        throw std::logic_error("original body failure");
    }
    catch (std::logic_error const& error)
    {
        EXPECT_EQ(std::string{error.what()}, "original body failure");
    }
    EXPECT_TRUE(bodyRan);
    EXPECT_EQ(nameCalls, 1);
    EXPECT_EQ(metadataCalls, 0);
    EXPECT_EQ(observation().starts, 0);
    EXPECT_EQ(observation().ends, 0);
}
