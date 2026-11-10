/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/logger.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace tensorrt_llm::common;

TEST(LoggerModuleTest, FormatModuleNoTrailingSpaces)
{
    for (auto const* raw : {"batch_manager", "common", "cutlass_extensions", "deep_ep", "deep_gemm", "executor",
             "flash_mla", "kernels", "layers", "nanobind", "runtime", "testing", "thop"})
    {
        auto const fmt = formatModule(raw);
        EXPECT_FALSE(fmt.empty());
        EXPECT_NE(fmt.back(), ' ') << "trailing space for \"" << raw << "\": \"" << fmt << "\"";
    }
}

TEST(LoggerModuleTest, ExtractModuleMatchesEnvKey)
{
    EXPECT_EQ(Logger::extractModule("/code/tensorrt_llm/runtime/foo.cpp"), "runtime");
    EXPECT_EQ(Logger::extractModule("/code/tensorrt_llm/batch_manager/s.cpp"), "batchmgr");
    EXPECT_EQ(Logger::extractModule("/code/tensorrt_llm/common/logger.h"), "common");
    EXPECT_EQ(Logger::extractModule("/code/tensorrt_llm/thop/alloc.cpp"), "thop");
    EXPECT_EQ(Logger::extractModule("/random/path/foo.cpp"), "others");
}

TEST(LoggerModuleTest, PerModuleFilterGainControl)
{
    // Set per-module override: runtime gets DEBUG while global stays WARNING.
    // Use a new thread so its thread_local Logger picks up the env var.
    setenv("TLLM_LOG_LEVEL_BY_MODULE", "debug:runtime", 1);

    std::thread t(
        []
        {
            auto* logger = Logger::getLogger();
            logger->setLevel(Logger::WARNING);

            constexpr auto runtimeMod = Logger::extractModule("/code/tensorrt_llm/runtime/foo.cpp");
            constexpr auto commonMod = Logger::extractModule("/code/tensorrt_llm/common/bar.cpp");

            // runtime has per-module DEBUG override — verbose messages enabled
            EXPECT_TRUE(logger->isEnabled(Logger::DEBUG, runtimeMod));
            // common has no override — falls back to global WARNING
            EXPECT_FALSE(logger->isEnabled(Logger::DEBUG, commonMod));
            // WARNING enabled for both
            EXPECT_TRUE(logger->isEnabled(Logger::WARNING, runtimeMod));
            EXPECT_TRUE(logger->isEnabled(Logger::WARNING, commonMod));
        });
    t.join();

    unsetenv("TLLM_LOG_LEVEL_BY_MODULE");
}

namespace
{

size_t countOccurrences(std::string const& haystack, std::string const& needle)
{
    size_t count = 0;
    for (size_t pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + needle.size()))
    {
        ++count;
    }
    return count;
}

// The log-once registry is process-wide, so a fixed key would stay consumed when the test binary is
// run more than once in one process (--gtest_repeat, IDE re-runs). Tag each test invocation instead.
std::string uniqueKeyPrefix()
{
    static std::atomic<uint64_t> invocationCount{0};
    return "loggerTest/" + std::to_string(++invocationCount) + "/";
}

// Capture everything the Logger writes to std::cout and std::cerr while this object is alive.
// The stream is chosen from the logger's current level (level_ < WARNING => std::cout), not from
// the message level, so both streams have to be captured for filtered-level assertions to be real.
// getLogger() is thread-local and writes can come from several threads, so the captured buffer is
// mutex-protected.
class StreamCapture
{
public:
    StreamCapture()
        : mOldOut(std::cout.rdbuf(&mBuffer))
        , mOldErr(std::cerr.rdbuf(&mBuffer))
    {
    }

    ~StreamCapture()
    {
        std::cout.rdbuf(mOldOut);
        std::cerr.rdbuf(mOldErr);
    }

    std::string str() const
    {
        return mBuffer.str();
    }

private:
    class ThreadSafeBuffer : public std::streambuf
    {
    public:
        std::string str() const
        {
            std::lock_guard<std::mutex> lock(mMutex);
            return mData;
        }

    protected:
        std::streamsize xsputn(char const* data, std::streamsize size) override
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mData.append(data, static_cast<size_t>(size));
            return size;
        }

        int_type overflow(int_type ch) override
        {
            if (ch == traits_type::eof())
            {
                return traits_type::not_eof(ch);
            }
            std::lock_guard<std::mutex> lock(mMutex);
            mData.push_back(static_cast<char>(ch));
            return ch;
        }

    private:
        mutable std::mutex mMutex;
        std::string mData;
    };

    ThreadSafeBuffer mBuffer;
    std::streambuf* mOldOut;
    std::streambuf* mOldErr;
};

// Minimal C++17 barrier: release all threads at once so the concurrent log-once calls overlap.
class Barrier
{
public:
    explicit Barrier(int count)
        : mCount(count)
    {
    }

    void arriveAndWait()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        if (--mCount == 0)
        {
            mCv.notify_all();
        }
        else
        {
            mCv.wait(lock, [this] { return mCount == 0; });
        }
    }

private:
    std::mutex mMutex;
    std::condition_variable mCv;
    int mCount;
};

} // namespace

TEST(LoggerLogOnceTest, LogsEachKeyOnce)
{
    auto* logger = Logger::getLogger();
    // Keep the logger level below WARNING: Logger::log() selects std::cout vs std::cerr from the
    // logger's current level, so warnings reach the captured stream only while level_ < WARNING.
    logger->setLevel(Logger::INFO);
    std::string const prefix = uniqueKeyPrefix();

    std::string output;
    {
        StreamCapture capture;
        // Same key twice: the second message must be suppressed.
        logger->logOnce(Logger::WARNING, prefix + "keyA", "keyA first %d", 1);
        logger->logOnce(Logger::WARNING, prefix + "keyA", "keyA second %d", 2);
        // A different key is independent and logged once as well.
        logger->logOnce(Logger::WARNING, prefix + "keyB", "keyB first");
        logger->logOnce(Logger::WARNING, prefix + "keyB", "keyB second");
        output = capture.str();
    }

    EXPECT_EQ(countOccurrences(output, "keyA first"), 1u);
    EXPECT_EQ(countOccurrences(output, "keyA second"), 0u);
    EXPECT_EQ(countOccurrences(output, "keyB first"), 1u);
    EXPECT_EQ(countOccurrences(output, "keyB second"), 0u);
}

TEST(LoggerLogOnceTest, DisabledLevelDoesNotConsumeKey)
{
    auto* logger = Logger::getLogger();
    std::string const key = uniqueKeyPrefix() + "levelKey";

    // WARNING filtered out: the key must not be consumed, so enabling the level later still
    // logs the message once. Both output streams are captured, so this check is real.
    logger->setLevel(Logger::ERROR);
    std::string filteredOutput;
    {
        StreamCapture capture;
        logger->logOnce(Logger::WARNING, key, "gated warning");
        filteredOutput = capture.str();
    }
    EXPECT_EQ(countOccurrences(filteredOutput, "gated warning"), 0u);

    // Lowering the level (ERROR -> INFO) surfaces the still-unconsumed key exactly once.
    logger->setLevel(Logger::INFO);
    std::string enabledOutput;
    {
        StreamCapture capture;
        logger->logOnce(Logger::WARNING, key, "gated warning");
        logger->logOnce(Logger::WARNING, key, "gated warning");
        enabledOutput = capture.str();
    }
    EXPECT_EQ(countOccurrences(enabledOutput, "gated warning"), 1u);
}

TEST(LoggerLogOnceTest, WarningOnceMacroDedupes)
{
    auto* logger = Logger::getLogger();
    logger->setLevel(Logger::INFO);
    std::string const key = uniqueKeyPrefix() + "macroKey";

    std::string output;
    {
        StreamCapture capture;
        TLLM_LOG_WARNING_ONCE(key, "macro warning %d", 1);
        TLLM_LOG_WARNING_ONCE(key, "macro warning %d", 2);
        output = capture.str();
    }

    EXPECT_EQ(countOccurrences(output, "macro warning 1"), 1u);
    EXPECT_EQ(countOccurrences(output, "macro warning 2"), 0u);
}

TEST(LoggerLogOnceTest, KeysAreScopedPerModule)
{
    auto* logger = Logger::getLogger();
    logger->setLevel(Logger::INFO);
    std::string const key = uniqueKeyPrefix() + "scopedKey";

    std::string output;
    {
        StreamCapture capture;
        // Same key text through the module-less API and through the macro (module-tagged): the two
        // call sites must not suppress each other.
        logger->logOnce(Logger::WARNING, key, "module-less message");
        TLLM_LOG_WARNING_ONCE(key, "module-tagged message");
        output = capture.str();
    }

    EXPECT_EQ(countOccurrences(output, "module-less message"), 1u);
    EXPECT_EQ(countOccurrences(output, "module-tagged message"), 1u);
}

TEST(LoggerLogOnceTest, ThreadSafeSingleEmission)
{
    constexpr int kNumThreads = 8;
    constexpr int kNumKeys = 200;
    std::string const prefix = uniqueKeyPrefix();

    std::vector<std::string> keys;
    keys.reserve(kNumKeys);
    for (int i = 0; i < kNumKeys; ++i)
    {
        keys.push_back(prefix + "threadKey/" + std::to_string(i));
    }

    std::string output;
    {
        StreamCapture capture;
        Barrier barrier(kNumThreads);
        std::vector<std::thread> threads;
        threads.reserve(kNumThreads);
        for (int t = 0; t < kNumThreads; ++t)
        {
            threads.emplace_back(
                [&barrier, &keys]
                {
                    // getLogger() is thread-local, so set the level on this thread: without it the
                    // logger keeps the TLLM_LOG_LEVEL default (e.g. WARNING writes to std::cerr).
                    Logger::getLogger()->setLevel(Logger::INFO);
                    barrier.arriveAndWait();
                    // Every thread logs the same keys; each key must be emitted exactly once.
                    for (auto const& key : keys)
                    {
                        TLLM_LOG_WARNING_ONCE(key, "threaded warning [%s]", key.c_str());
                    }
                });
        }
        for (auto& thread : threads)
        {
            thread.join();
        }
        output = capture.str();
    }

    EXPECT_EQ(countOccurrences(output, "threaded warning "), static_cast<size_t>(kNumKeys));
    for (auto const& key : keys)
    {
        // Bracket the key so e.g. ".../threadKey/4" does not match ".../threadKey/40".
        EXPECT_EQ(countOccurrences(output, "[" + key + "]"), 1u);
    }
}
