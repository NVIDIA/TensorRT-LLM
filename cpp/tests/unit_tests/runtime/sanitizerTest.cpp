/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/bufferManager.h"

#include <gtest/gtest.h>
#include <limits>
#include <sanitizer/common_interface_defs.h>

extern "C" char const* __asan_default_options()
{
    return "detect_leaks=0:protect_shadow_gap=0:replace_intrin=0:detect_invalid_pointer_pairs=2";
}

extern "C" char const* __ubsan_default_options()
{
    return "halt_on_error=1";
}

#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
bool constexpr has_asan = true;
#else
bool constexpr has_asan = false;
#endif

#if __has_feature(undefined_behavior_sanitizer) || defined(SANITIZE_UNDEFINED)
bool constexpr has_ubsan = true;
#else
bool constexpr has_ubsan = false;
#endif

TEST(SanitizerTest, ASanWorking)
{
    if (!has_asan)
    {
        GTEST_SKIP() << "ASan not enabled";
    }

    auto test_body = []
    {
        auto memory = std::make_unique<uint8_t[]>(1);

        // Catch out-of-bound read
        auto volatile _ = memory[1];
    };

    ASSERT_DEATH(test_body(), "AddressSanitizer") << "ASan didn't detect out-of-bound access";
}

TEST(SanitizerTest, ASanPointerArithWorking)
{
    if (!has_asan)
    {
        GTEST_SKIP() << "ASan not enabled";
    }

    auto test_body = []
    {
        auto memory1 = std::make_unique<uint8_t[]>(1);
        auto memory2 = std::make_unique<uint8_t[]>(1);

        // Catch arithmetic on unrelated pointers
        auto volatile _ = memory2.get() - memory1.get();
    };

    ASSERT_DEATH(test_body(), "AddressSanitizer") << "ASan didn't detect bad pointer arithmetic";
}

TEST(SanitizerTest, ASanMemcpyAsyncNoSyncDie)
{
    if (!has_asan)
    {
        GTEST_SKIP() << "ASan not enabled";
    }

    auto test_body = []
    {
        using namespace tensorrt_llm;
        auto manager = runtime::BufferManager(std::make_shared<runtime::CudaStream>());

        auto constexpr sizeAhead = 1 * 1024 * 1024 * 1024; // 1GB
        auto constexpr sizeDetect = 1024;

        auto hostAhead = manager.allocate(runtime::MemoryType::kPINNED, sizeAhead);
        auto deviceAhead = manager.allocate(runtime::MemoryType::kGPU, sizeAhead);

        auto hostDetect = manager.allocate(runtime::MemoryType::kPINNED, sizeDetect);
        auto deviceDetect = manager.allocate(runtime::MemoryType::kGPU, sizeDetect);

        manager.getStream().synchronize();
        manager.copy(*deviceAhead, hostAhead->data());
        manager.copy(*deviceDetect, hostDetect->data());

        auto const* ptr = static_cast<uint8_t*>(hostDetect->data());
        // Catch racing read on memory with inflight D2H copy.
        auto volatile _ = ptr[1];
    };

    ASSERT_DEATH(test_body(), "AddressSanitizer") << "ASan didn't detect unsync memory access";
}

TEST(SanitizerTest, ASanMemcpyAsyncSyncOK)
{
    if (!has_asan)
    {
        GTEST_SKIP() << "ASan not enabled";
    }

    auto test_body = []
    {
        using namespace tensorrt_llm;
        auto manager = runtime::BufferManager(std::make_shared<runtime::CudaStream>());

        auto constexpr sizeAhead = 1024 * 1024 * 1024; // 1GB
        auto constexpr sizeDetect = 1024;

        auto hostAhead = manager.allocate(runtime::MemoryType::kPINNED, sizeAhead);
        auto deviceAhead = manager.allocate(runtime::MemoryType::kGPU, sizeAhead);

        auto hostDetect = manager.allocate(runtime::MemoryType::kPINNED, sizeDetect);
        auto deviceDetect = manager.allocate(runtime::MemoryType::kGPU, sizeDetect);

        manager.getStream().synchronize();
        manager.copy(*deviceAhead, hostAhead->data());
        manager.copy(*deviceDetect, hostDetect->data());

        auto const* ptr = static_cast<uint8_t*>(hostDetect->data());

        manager.getStream().synchronize();
        // After synchronize, the memory region should be unpoisoned and safe to read.
        auto volatile _ = ptr[1];
    };

    ASSERT_NO_FATAL_FAILURE(test_body()) << "Unexpected crash";
}

TEST(SanitizerTest, UBSanWorking)
{
    if (!has_ubsan)
    {
        GTEST_SKIP() << "UBSan not enabled";
    }

    auto test_body = []
    {
        auto volatile value = std::numeric_limits<int>::max();

        // Catch signed-integer-overflow
        value++;
    };

    ASSERT_DEATH(test_body(), "runtime error") << "UBSan didn't detect undefined behavior";
}
