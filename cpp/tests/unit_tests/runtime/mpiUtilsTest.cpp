/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <gtest/gtest.h>

#include <csignal>
#include <cstdlib>
#include <functional>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace
{

// Match installFaultToleranceSignalHandlers()'s exit code; if this drifts, the
// implementation comment about the "137 = 128+9 SIGKILL convention" needs updating too.
constexpr int kExpectedFtAbortExitCode = 137;

class FaultToleranceModeEnvGuard
{
public:
    explicit FaultToleranceModeEnvGuard(char const* value)
    {
        char const* prev = std::getenv("TLLM_FAULT_TOLERANCE_MODE");
        if (prev != nullptr)
        {
            mPrev = prev;
            mHadPrev = true;
        }
        if (value == nullptr)
        {
            ::unsetenv("TLLM_FAULT_TOLERANCE_MODE");
        }
        else
        {
            ::setenv("TLLM_FAULT_TOLERANCE_MODE", value, /*overwrite=*/1);
        }
    }

    ~FaultToleranceModeEnvGuard()
    {
        if (mHadPrev)
        {
            ::setenv("TLLM_FAULT_TOLERANCE_MODE", mPrev.c_str(), /*overwrite=*/1);
        }
        else
        {
            ::unsetenv("TLLM_FAULT_TOLERANCE_MODE");
        }
    }

    FaultToleranceModeEnvGuard(FaultToleranceModeEnvGuard const&) = delete;
    FaultToleranceModeEnvGuard& operator=(FaultToleranceModeEnvGuard const&) = delete;
    FaultToleranceModeEnvGuard(FaultToleranceModeEnvGuard&&) = delete;
    FaultToleranceModeEnvGuard& operator=(FaultToleranceModeEnvGuard&&) = delete;

private:
    std::string mPrev{};
    bool mHadPrev{false};
};

// Sentinel exit code used by runInChildAndWait() when the body returns normally
// instead of terminating the child as expected. Distinct from kExpectedFtAbortExitCode
// so the parent can tell "handler did not fire" apart from "handler fired".
constexpr int kChildBodyReturnedSentinel = 99;

} // namespace

// ---------------------------------------------------------------------------
// isFaultToleranceModeEnabled()
// ---------------------------------------------------------------------------

TEST(MpiUtilsFaultToleranceTest, EnvVarUnsetIsDisabled)
{
    FaultToleranceModeEnvGuard guard{nullptr};
    EXPECT_FALSE(tensorrt_llm::mpi::isFaultToleranceModeEnabled());
}

TEST(MpiUtilsFaultToleranceTest, EnvVarOneIsEnabled)
{
    FaultToleranceModeEnvGuard guard{"1"};
    EXPECT_TRUE(tensorrt_llm::mpi::isFaultToleranceModeEnabled());
}

TEST(MpiUtilsFaultToleranceTest, EnvVarZeroIsDisabled)
{
    FaultToleranceModeEnvGuard guard{"0"};
    EXPECT_FALSE(tensorrt_llm::mpi::isFaultToleranceModeEnabled());
}

TEST(MpiUtilsFaultToleranceTest, EnvVarEmptyStringIsDisabled)
{
    FaultToleranceModeEnvGuard guard{""};
    EXPECT_FALSE(tensorrt_llm::mpi::isFaultToleranceModeEnabled());
}

TEST(MpiUtilsFaultToleranceTest, EnvVarNonOneStringIsDisabled)
{
    // The contract is "exactly the string \"1\"" — we deliberately reject other
    // common truthy spellings to avoid future ambiguity if the knob ever grows
    // multi-valued (e.g. "1=signal-handler-only", "2=full"). Keep this strict
    // until the LLMArgs replacement (PR 1d.1) defines a richer schema.
    for (char const* val : {"true", "yes", "TRUE", "01", " 1", "1 ", "10"})
    {
        FaultToleranceModeEnvGuard guard{val};
        EXPECT_FALSE(tensorrt_llm::mpi::isFaultToleranceModeEnabled())
            << "expected disabled for TLLM_FAULT_TOLERANCE_MODE=\"" << val << "\"";
    }
}

// ---------------------------------------------------------------------------
// installFaultToleranceSignalHandlers()
// ---------------------------------------------------------------------------
//
// We run the handler in a forked child so the parent gtest process stays alive
// to inspect the child's exit code. The fork-based approach avoids any need
// for MPI_Init or a multi-rank launcher; this test exercises the handler in
// isolation (the survivor-survives-peer-death story is validated end-to-end
// in PR 1d.4's fault-injection harness and in
// docs/design/wide-ep-fault-tolerance/research-pass-prototypes/mpi_signal_handler.py).

namespace
{

// Run @c body in a forked child, wait for it to exit, and return the raw
// status (as filled by waitpid). Aborts the test with FAIL() if fork or
// waitpid fails. On success the returned value can be inspected with
// WIFEXITED / WEXITSTATUS / WIFSIGNALED / WTERMSIG.
int runInChildAndWait(std::function<void()> const& body)
{
    pid_t pid = ::fork();
    if (pid < 0)
    {
        ADD_FAILURE() << "fork failed: errno=" << errno;
        return -1;
    }
    if (pid == 0)
    {
        // Child: invoke body. body is expected to terminate the process; if
        // it returns, fall through to a sentinel exit code so the parent can
        // distinguish "handler did not fire" from the expected exit codes.
        body();
        ::_exit(kChildBodyReturnedSentinel);
    }
    int status = 0;
    pid_t waited = ::waitpid(pid, &status, 0);
    if (waited != pid)
    {
        ADD_FAILURE() << "waitpid returned " << waited << ", errno=" << errno;
        return -1;
    }
    return status;
}

} // namespace

TEST(MpiUtilsFaultToleranceTest, HandlerExitsCleanlyOnSIGABRT)
{
    int const status = runInChildAndWait(
        []
        {
            tensorrt_llm::mpi::installFaultToleranceSignalHandlers();
            // raise(SIGABRT) re-enters the handler synchronously and should not return;
            // discard the return value because the handler terminates the process before
            // any control flow could reach a check on it.
            (void) ::raise(SIGABRT);
        });

    ASSERT_TRUE(WIFEXITED(status)) << "child did not exit cleanly: WIFSIGNALED=" << WIFSIGNALED(status)
                                   << " WTERMSIG=" << (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    EXPECT_EQ(WEXITSTATUS(status), kExpectedFtAbortExitCode)
        << "expected _exit(" << kExpectedFtAbortExitCode << ") from FT handler; got " << WEXITSTATUS(status);
}

TEST(MpiUtilsFaultToleranceTest, HandlerExitsCleanlyOnSIGSEGV)
{
    int const status = runInChildAndWait(
        []
        {
            tensorrt_llm::mpi::installFaultToleranceSignalHandlers();
            (void) ::raise(SIGSEGV);
        });

    ASSERT_TRUE(WIFEXITED(status)) << "child did not exit cleanly on SIGSEGV: WIFSIGNALED=" << WIFSIGNALED(status)
                                   << " WTERMSIG=" << (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
    EXPECT_EQ(WEXITSTATUS(status), kExpectedFtAbortExitCode);
}

TEST(MpiUtilsFaultToleranceTest, HandlerIsIdempotent)
{
    int const status = runInChildAndWait(
        []
        {
            // Two installs in a row must not crash, and the resulting handler
            // must still exit with our chosen code on SIGABRT.
            tensorrt_llm::mpi::installFaultToleranceSignalHandlers();
            tensorrt_llm::mpi::installFaultToleranceSignalHandlers();
            (void) ::raise(SIGABRT);
        });

    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), kExpectedFtAbortExitCode);
}

TEST(MpiUtilsFaultToleranceTest, DefaultHandlerKillsViaSignal)
{
    // Sanity check: without our handler, SIGABRT terminates the child via the
    // signal (WIFSIGNALED == true), not via _exit. This locks in the contract
    // that the FT handler is the thing intercepting the signal — a regression
    // that silently no-ops the install would still pass HandlerExits* if the
    // default behavior happened to also produce a clean exit code (it doesn't).
    int const status = runInChildAndWait([] { (void) ::raise(SIGABRT); });

    EXPECT_TRUE(WIFSIGNALED(status)) << "default SIGABRT handler unexpectedly produced a clean exit; "
                                     << "WIFEXITED=" << WIFEXITED(status) << " WEXITSTATUS=" << WEXITSTATUS(status);
    if (WIFSIGNALED(status))
    {
        EXPECT_EQ(WTERMSIG(status), SIGABRT);
    }
}
