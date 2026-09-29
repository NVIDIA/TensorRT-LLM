/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/nanobind/visual_gen/coordinatorWatchdog.h"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

#if defined(__linux__)
#include <poll.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace tensorrt_llm::nanobind::visual_gen
{
namespace
{

#if defined(__linux__)
constexpr auto kParentPollInterval = std::chrono::seconds{1};

[[noreturn]] void terminateWorker(pid_t workerPid) noexcept
{
    ::kill(workerPid, SIGKILL);
    ::_exit(EXIT_FAILURE);
}

int openPidfd(pid_t pid)
{
#if defined(SYS_pidfd_open)
    return static_cast<int>(::syscall(SYS_pidfd_open, pid, 0U));
#else
    errno = ENOSYS;
    return -1;
#endif
}

void watchCoordinator(int coordinatorFd, pid_t workerPid) noexcept
{
    ::pthread_setname_np(::pthread_self(), "visualgen-watch");
    pollfd descriptor{coordinatorFd, POLLIN, 0};
    int result;
    do
    {
        result = ::poll(&descriptor, 1, -1);
    } while (result < 0 && errno == EINTR);

    ::close(coordinatorFd);
    terminateWorker(workerPid);
}

void watchCoordinatorByParentPid(pid_t expectedParentPid, pid_t workerPid) noexcept
{
    ::pthread_setname_np(::pthread_self(), "visualgen-watch");
    while (::getppid() == expectedParentPid)
    {
        std::this_thread::sleep_for(kParentPollInterval);
    }
    terminateWorker(workerPid);
}

using OpenPidfd = std::function<int(pid_t)>;

std::optional<std::string> startCoordinatorWatchdogImpl(std::int64_t coordinatorPid, OpenPidfd const& openPidfd)
{
    if (coordinatorPid <= 0 || coordinatorPid > std::numeric_limits<pid_t>::max())
    {
        throw std::invalid_argument("coordinator PID is outside the pid_t range");
    }

    auto const expectedParentPid = static_cast<pid_t>(coordinatorPid);
    if (::getppid() != expectedParentPid)
    {
        throw std::runtime_error("VisualGen coordinator exited before worker supervision started");
    }

    int const coordinatorFd = openPidfd(expectedParentPid);
    if (coordinatorFd < 0)
    {
        int const errorCode = errno;
        if (errorCode != ENOSYS && errorCode != EPERM)
        {
            throw std::system_error(errorCode, std::generic_category(), "pidfd_open for VisualGen coordinator failed");
        }

        // A seccomp profile may reject pidfd_open even on a recent kernel. A
        // native polling thread preserves supervision without depending on
        // the Python GIL. One getppid() call per worker per second is
        // negligible compared with model serving and bounds detection to one
        // second.
        if (::getppid() != expectedParentPid)
        {
            terminateWorker(::getpid());
        }
        std::thread(watchCoordinatorByParentPid, expectedParentPid, ::getpid()).detach();
        auto const* errorName = errorCode == ENOSYS ? "ENOSYS" : "EPERM";
        return "pidfd_open for VisualGen coordinator failed with " + std::string(errorName) + " ("
            + std::generic_category().message(errorCode) + "); using the native 1-second parent-PID polling fallback";
    }

    // pidfd_open() and the parent check must be paired. If the coordinator
    // exited and its PID was reused between them, getppid() still names the
    // init process or subreaper rather than the unrelated process we opened.
    if (::getppid() != expectedParentPid)
    {
        ::close(coordinatorFd);
        throw std::runtime_error("VisualGen coordinator exited while worker supervision was starting");
    }

    try
    {
        std::thread(watchCoordinator, coordinatorFd, ::getpid()).detach();
    }
    catch (...)
    {
        ::close(coordinatorFd);
        throw;
    }
    return std::nullopt;
}
#endif

} // namespace

std::optional<std::string> startCoordinatorWatchdog(std::int64_t coordinatorPid)
{
#if defined(__linux__)
    return startCoordinatorWatchdogImpl(coordinatorPid, openPidfd);
#else
    static_cast<void>(coordinatorPid);
    throw std::runtime_error("VisualGen coordinator supervision requires Linux");
#endif
}

namespace testing
{

std::optional<std::string> startCoordinatorWatchdogWithPidfdError(std::int64_t coordinatorPid, int pidfdErrorCode)
{
#if defined(__linux__)
    return startCoordinatorWatchdogImpl(coordinatorPid,
        [pidfdErrorCode](pid_t)
        {
            errno = pidfdErrorCode;
            return -1;
        });
#else
    static_cast<void>(coordinatorPid);
    static_cast<void>(pidfdErrorCode);
    throw std::runtime_error("VisualGen coordinator supervision requires Linux");
#endif
}

} // namespace testing

} // namespace tensorrt_llm::nanobind::visual_gen
