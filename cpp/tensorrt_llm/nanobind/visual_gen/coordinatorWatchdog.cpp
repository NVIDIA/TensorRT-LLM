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
#include <csignal>
#include <cstdlib>
#include <limits>
#include <stdexcept>
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
#endif

} // namespace

void startCoordinatorWatchdog(std::int64_t coordinatorPid)
{
#if defined(__linux__)
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
        throw std::system_error(errorCode, std::generic_category(), "pidfd_open for VisualGen coordinator failed");
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
#else
    static_cast<void>(coordinatorPid);
    throw std::runtime_error("VisualGen coordinator supervision requires Linux pidfds");
#endif
}

} // namespace tensorrt_llm::nanobind::visual_gen
