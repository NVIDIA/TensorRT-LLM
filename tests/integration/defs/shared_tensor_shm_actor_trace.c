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

/*
 * Node-level unlink-ACTOR tracer for NVBug 6336747.
 *
 * The inotify watcher (shared_tensor_shm_event_trace.c) proves that a
 * /dev/shm/torch_* file is removed, but not by whom; the LD_PRELOAD shim and
 * the ptrace/seccomp wrapper only see processes inside the pytest tree. This
 * tracer closes that gap using fanotify directory-entry events, which are
 * delivered with the pid of the process that caused them, for every actor that
 * can modify the marked /dev/shm directory inode -- including processes outside
 * the pytest process tree (a reparented torch_shm_manager, an MPI rank launched
 * as a sibling, or, if /dev/shm is the shared host tmpfs, a host-side process).
 *
 * For each removal (FAN_DELETE / FAN_MOVED_FROM) of a torch_* entry it records
 * the actor's pid, ppid, uid, comm, exe, and mount + ipc namespaces, so the
 * remover can be classified as in-container (mntns == the pytest container's)
 * versus external, and identified by executable and call context. An actor_pid
 * of 0 means the remover lives in a pid namespace not visible to this reader
 * (i.e. it is external to the container) -- itself a decisive signal.
 *
 * Requires CAP_SYS_ADMIN and a filesystem that supports file handles (tmpfs
 * does on modern kernels). When either is unavailable, fanotify_init /
 * fanotify_mark fail and the tracer emits a single event=watch_error line and
 * exits non-zero so the caller can degrade without failing the run.
 *
 * Usage: shared_tensor_shm_actor_trace TRACE_FILE WATCH_DIR
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
#include <sys/fanotify.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Fallbacks for headers that predate the FID/DFID_NAME reporting mode. */
#ifndef FAN_REPORT_DFID_NAME
#define FAN_REPORT_DFID_NAME 0x00000c00
#endif
#ifndef FAN_EVENT_INFO_TYPE_DFID_NAME
#define FAN_EVENT_INFO_TYPE_DFID_NAME 2
#endif

static sig_atomic_t volatile sStopRequested = 0;

static void handleSignal(int signalNumber)
{
    (void) signalNumber;
    sStopRequested = 1;
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

static void readProcComm(pid_t pid, char* output, size_t outputSize)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/comm", (int) pid);
    int const fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        snprintf(output, outputSize, "?");
        return;
    }
    ssize_t const bytesRead = read(fd, output, outputSize - 1U);
    close(fd);
    if (bytesRead <= 0)
    {
        snprintf(output, outputSize, "?");
        return;
    }
    size_t length = (size_t) bytesRead;
    while (length > 0U && (output[length - 1U] == '\n' || output[length - 1U] == '\r'))
    {
        length--;
    }
    output[length] = '\0';
}

static void readProcStatus(pid_t pid, long* parentPid, long* userId)
{
    *parentPid = -1;
    *userId = -1;
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/status", (int) pid);
    FILE* const file = fopen(path, "re");
    if (file == NULL)
    {
        return;
    }
    char line[256];
    while (fgets(line, sizeof(line), file) != NULL)
    {
        if (strncmp(line, "PPid:", sizeof("PPid:") - 1U) == 0)
        {
            *parentPid = strtol(line + (sizeof("PPid:") - 1U), NULL, 10);
        }
        else if (strncmp(line, "Uid:", sizeof("Uid:") - 1U) == 0)
        {
            *userId = strtol(line + (sizeof("Uid:") - 1U), NULL, 10);
        }
    }
    fclose(file);
}

static void readProcLink(pid_t pid, char const* entry, char* output, size_t outputSize)
{
    char path[80];
    snprintf(path, sizeof(path), "/proc/%d/%s", (int) pid, entry);
    readLinkOrUnknown(path, output, outputSize);
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

static void traceRemoval(int traceFd, char const* op, char const* name, pid_t actorPid)
{
    struct timespec timestamp;
    clock_gettime(CLOCK_REALTIME, &timestamp);

    /* pid == 0: the actor is in a pid namespace not visible to this reader,
     * i.e. it is outside this container (external / host / sibling). */
    if (actorPid <= 0)
    {
        dprintf(traceFd, "time=%lld.%09ld event=removal op=%s name=%s actor_pid=%d actor_state=nspid_hidden\n",
            (long long) timestamp.tv_sec, timestamp.tv_nsec, op, name, (int) actorPid);
        return;
    }

    char comm[64];
    char exe[256];
    char mountNamespace[128];
    char ipcNamespace[128];
    long parentPid = -1;
    long userId = -1;
    readProcComm(actorPid, comm, sizeof(comm));
    readProcStatus(actorPid, &parentPid, &userId);
    readProcLink(actorPid, "exe", exe, sizeof(exe));
    readProcLink(actorPid, "ns/mnt", mountNamespace, sizeof(mountNamespace));
    readProcLink(actorPid, "ns/ipc", ipcNamespace, sizeof(ipcNamespace));

    /* If /proc identity is fully gone the actor exited before we sampled it. */
    char const* state = (strcmp(comm, "?") == 0 && strcmp(exe, "?") == 0) ? "gone" : "live";

    dprintf(traceFd,
        "time=%lld.%09ld event=removal op=%s name=%s actor_pid=%d actor_ppid=%ld actor_uid=%ld actor_comm=%s "
        "actor_exe=%s actor_mntns=%s actor_ipcns=%s actor_state=%s\n",
        (long long) timestamp.tv_sec, timestamp.tv_nsec, op, name, (int) actorPid, parentPid, userId, comm, exe,
        mountNamespace, ipcNamespace, state);
}

static char const* extractEntryName(struct fanotify_event_metadata const* metadata)
{
    char const* position = (char const*) metadata + metadata->metadata_len;
    char const* const end = (char const*) metadata + metadata->event_len;
    while (position + sizeof(struct fanotify_event_info_header) <= end)
    {
        struct fanotify_event_info_header const* header = (struct fanotify_event_info_header const*) position;
        if (header->len == 0U || position + header->len > end)
        {
            break;
        }
        if (header->info_type == FAN_EVENT_INFO_TYPE_DFID_NAME)
        {
            struct fanotify_event_info_fid const* fid = (struct fanotify_event_info_fid const*) position;
            struct file_handle const* handle = (struct file_handle const*) fid->handle;
            char const* const name = (char const*) handle->f_handle + handle->handle_bytes;
            if (name >= end)
            {
                return NULL;
            }
            return name;
        }
        position += header->len;
    }
    return NULL;
}

static int drainEvents(int traceFd, int fanotifyFd)
{
    char buffer[64U * 1024U] __attribute__((aligned(__alignof__(struct fanotify_event_metadata))));
    while (true)
    {
        ssize_t bytesRead = read(fanotifyFd, buffer, sizeof(buffer));
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

        struct fanotify_event_metadata* metadata = (struct fanotify_event_metadata*) buffer;
        while (FAN_EVENT_OK(metadata, bytesRead))
        {
            if (metadata->vers != FANOTIFY_METADATA_VERSION)
            {
                return -1;
            }
            if ((metadata->mask & FAN_Q_OVERFLOW) != 0U)
            {
                struct timespec overflowTime;
                clock_gettime(CLOCK_REALTIME, &overflowTime);
                dprintf(traceFd, "time=%lld.%09ld event=queue_overflow\n", (long long) overflowTime.tv_sec,
                    overflowTime.tv_nsec);
            }
            else if ((metadata->mask & (FAN_DELETE | FAN_MOVED_FROM)) != 0U)
            {
                char const* const name = extractEntryName(metadata);
                if (name != NULL && strncmp(name, "torch_", sizeof("torch_") - 1U) == 0)
                {
                    char const* const op = (metadata->mask & FAN_DELETE) != 0U ? "delete" : "moved_from";
                    traceRemoval(traceFd, op, name, (pid_t) metadata->pid);
                }
            }
            /* FID mode reports FAN_NOFD, but close any fd defensively. */
            if (metadata->fd >= 0)
            {
                close(metadata->fd);
            }
            metadata = FAN_EVENT_NEXT(metadata, bytesRead);
        }
    }
}

static int openFanotify(int traceFd)
{
    unsigned int const baseFlags = FAN_CLOEXEC | FAN_CLASS_NOTIF | FAN_NONBLOCK | FAN_REPORT_DFID_NAME;

    /* FAN_UNLIMITED_QUEUE requires CAP_SYS_ADMIN. Besides preventing queue
     * overflow, requiring it rejects unprivileged fanotify groups, for which
     * Linux reports pid 0 for every event generated by another process. */
    int const fanotifyFd = fanotify_init(baseFlags | FAN_UNLIMITED_QUEUE, O_RDONLY | O_CLOEXEC);
    if (fanotifyFd >= 0)
    {
        return fanotifyFd;
    }

    dprintf(traceFd, "event=watch_error stage=fanotify_init errno=%d error=%s\n", errno, strerror(errno));
    return -1;
}

static int runTrace(int traceFd, char const* watchPath)
{
    int const fanotifyFd = openFanotify(traceFd);
    if (fanotifyFd < 0)
    {
        return EXIT_FAILURE;
    }

    if (fanotify_mark(fanotifyFd, FAN_MARK_ADD, FAN_DELETE | FAN_MOVED_FROM, AT_FDCWD, watchPath) != 0)
    {
        dprintf(traceFd, "event=watch_error stage=fanotify_mark errno=%d error=%s\n", errno, strerror(errno));
        close(fanotifyFd);
        return EXIT_FAILURE;
    }

    traceWatchState(traceFd, "start", watchPath);
    int result = EXIT_SUCCESS;
    int const kPollTimeoutMilliseconds = 1000;
    while (!sStopRequested)
    {
        struct pollfd pollDescriptor = {
            .fd = fanotifyFd,
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
        if (drainEvents(traceFd, fanotifyFd) != 0)
        {
            perror("read");
            result = EXIT_FAILURE;
            break;
        }
    }

    if (drainEvents(traceFd, fanotifyFd) != 0)
    {
        perror("read");
        result = EXIT_FAILURE;
    }

    traceWatchState(traceFd, "stop", watchPath);
    close(fanotifyFd);
    return result;
}

int main(int argc, char* argv[])
{
    int const kExpectedArgumentCount = 3;
    if (argc != kExpectedArgumentCount)
    {
        fprintf(stderr, "Usage: %s TRACE_FILE WATCH_DIR\n", argv[0]);
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
