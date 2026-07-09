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

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static sig_atomic_t volatile sStopRequested = 0;

static void handleSignal(int signalNumber)
{
    (void) signalNumber;
    sStopRequested = 1;
}

static void readLinkOrUnknown(char const* path, char* output, size_t outputSize)
{
    ssize_t const length = readlink(path, output, outputSize - 1U);
    if (length < 0)
    {
        snprintf(output, outputSize, "?");
        return;
    }
    output[length] = '\0';
}

static bool isTorchShmEvent(struct inotify_event const* event)
{
    if ((event->mask & (IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT | IN_IGNORED | IN_Q_OVERFLOW)) != 0U)
    {
        return true;
    }
    return event->len > 0U && strncmp(event->name, "torch_", sizeof("torch_") - 1U) == 0;
}

static void traceWatchState(int traceFd, char const* state, char const* watchPath)
{
    struct timespec timestamp;
    clock_gettime(CLOCK_REALTIME, &timestamp);

    struct stat status;
    unsigned long long device = 0;
    unsigned long long inode = 0;
    if (stat(watchPath, &status) == 0)
    {
        device = (unsigned long long) status.st_dev;
        inode = (unsigned long long) status.st_ino;
    }

    char mountNamespace[128];
    char ipcNamespace[128];
    readLinkOrUnknown("/proc/self/ns/mnt", mountNamespace, sizeof(mountNamespace));
    readLinkOrUnknown("/proc/self/ns/ipc", ipcNamespace, sizeof(ipcNamespace));

    dprintf(traceFd, "time=%lld.%09ld event=watch_%s pid=%d path=%s device=%llu inode=%llu mntns=%s ipcns=%s\n",
        (long long) timestamp.tv_sec, timestamp.tv_nsec, state, getpid(), watchPath, device, inode, mountNamespace,
        ipcNamespace);
}

static void traceEvent(int traceFd, struct inotify_event const* event)
{
    if (!isTorchShmEvent(event))
    {
        return;
    }

    struct timespec timestamp;
    clock_gettime(CLOCK_REALTIME, &timestamp);
    char const* name = event->len > 0U ? event->name : "-";
    dprintf(traceFd,
        "time=%lld.%09ld event=shm_namespace mask=0x%x cookie=%u create=%d delete=%d moved_from=%d "
        "moved_to=%d delete_self=%d move_self=%d unmount=%d overflow=%d ignored=%d name=%s\n",
        (long long) timestamp.tv_sec, timestamp.tv_nsec, event->mask, event->cookie, (event->mask & IN_CREATE) != 0U,
        (event->mask & IN_DELETE) != 0U, (event->mask & IN_MOVED_FROM) != 0U, (event->mask & IN_MOVED_TO) != 0U,
        (event->mask & IN_DELETE_SELF) != 0U, (event->mask & IN_MOVE_SELF) != 0U, (event->mask & IN_UNMOUNT) != 0U,
        (event->mask & IN_Q_OVERFLOW) != 0U, (event->mask & IN_IGNORED) != 0U, name);
}

static int installSignalHandlers(void)
{
    struct sigaction action = {0};
    action.sa_handler = handleSignal;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) != 0 || sigaction(SIGTERM, &action, NULL) != 0)
    {
        return -1;
    }
    return 0;
}

static int drainEvents(int traceFd, int inotifyFd)
{
    char buffer[64U * 1024U] __attribute__((aligned(__alignof__(struct inotify_event))));
    while (true)
    {
        ssize_t const bytesRead = read(inotifyFd, buffer, sizeof(buffer));
        if (bytesRead < 0)
        {
            if (errno == EAGAIN)
            {
                return 0;
            }
            if (errno == EINTR)
            {
                continue;
            }
            return -1;
        }
        if (bytesRead == 0)
        {
            return 0;
        }

        char const* position = buffer;
        char const* const end = buffer + bytesRead;
        while (position < end)
        {
            struct inotify_event const* event = (struct inotify_event const*) position;
            traceEvent(traceFd, event);
            position += sizeof(*event) + event->len;
        }
    }
}

static int runTrace(int traceFd, char const* watchPath)
{
    int const inotifyFd = inotify_init1(IN_CLOEXEC | IN_NONBLOCK);
    if (inotifyFd < 0)
    {
        perror("inotify_init1");
        return EXIT_FAILURE;
    }

    uint32_t const eventMask
        = IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT | IN_IGNORED;
    int const watchDescriptor = inotify_add_watch(inotifyFd, watchPath, eventMask);
    if (watchDescriptor < 0)
    {
        perror("inotify_add_watch");
        close(inotifyFd);
        return EXIT_FAILURE;
    }

    traceWatchState(traceFd, "start", watchPath);
    int result = EXIT_SUCCESS;
    int const kPollTimeoutMilliseconds = 1000;
    while (!sStopRequested)
    {
        struct pollfd pollDescriptor = {
            .fd = inotifyFd,
            .events = POLLIN,
        };
        int const pollResult = poll(&pollDescriptor, 1U, kPollTimeoutMilliseconds);
        if (pollResult < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            perror("poll");
            result = EXIT_FAILURE;
            break;
        }
        if (pollResult == 0)
        {
            continue;
        }

        if (drainEvents(traceFd, inotifyFd) != 0)
        {
            perror("read");
            result = EXIT_FAILURE;
            break;
        }
    }

    if (drainEvents(traceFd, inotifyFd) != 0)
    {
        perror("read");
        result = EXIT_FAILURE;
    }

    traceWatchState(traceFd, "stop", watchPath);
    inotify_rm_watch(inotifyFd, watchDescriptor);
    close(inotifyFd);
    return result;
}

int main(int argc, char* argv[])
{
    int const kExpectedArgumentCount = 3;
    if (argc != kExpectedArgumentCount)
    {
        fprintf(stderr, "Usage: %s TRACE_FILE WATCH_PATH\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (installSignalHandlers() != 0)
    {
        perror("sigaction");
        return EXIT_FAILURE;
    }

    int const traceFd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (traceFd < 0)
    {
        perror("open trace file");
        return EXIT_FAILURE;
    }
    int const result = runTrace(traceFd, argv[2]);
    close(traceFd);
    return result;
}
