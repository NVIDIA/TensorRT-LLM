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

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

typedef int (*ShmOpenFn)(char const* name, int flags, mode_t mode);
typedef int (*ShmUnlinkFn)(char const* name);
typedef int (*UnlinkFn)(char const* path);
typedef int (*UnlinkAtFn)(int directoryFd, char const* path, int flags);

static ShmOpenFn realShmOpen;
static ShmUnlinkFn realShmUnlink;
static UnlinkFn realUnlink;
static UnlinkAtFn realUnlinkAt;

__attribute__((constructor)) static void initializeShmTrace(void)
{
    realShmOpen = (ShmOpenFn) dlsym(RTLD_NEXT, "shm_open");
    realShmUnlink = (ShmUnlinkFn) dlsym(RTLD_NEXT, "shm_unlink");
    realUnlink = (UnlinkFn) dlsym(RTLD_NEXT, "unlink");
    realUnlinkAt = (UnlinkAtFn) dlsym(RTLD_NEXT, "unlinkat");
}

static bool isTorchShmName(char const* name)
{
    return name != NULL && strncmp(name, "/torch_", sizeof("/torch_") - 1U) == 0;
}

static bool isTorchShmPath(char const* path)
{
    if (path == NULL)
    {
        return false;
    }
    char const* basename = strrchr(path, '/');
    basename = basename == NULL ? path : basename + 1;
    return strncmp(basename, "torch_", sizeof("torch_") - 1U) == 0;
}

static bool readRefcount(char const* name, char const* resolvedPath, int32_t* refcount)
{
    char shmPath[4096];
    if (resolvedPath != NULL)
    {
        snprintf(shmPath, sizeof(shmPath), "%s", resolvedPath);
    }
    else if (name != NULL && name[0] == '/')
    {
        snprintf(shmPath, sizeof(shmPath), "/dev/shm%s", name);
    }
    else
    {
        return false;
    }

    int const fd = open(shmPath, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        return false;
    }
    ssize_t const bytesRead = pread(fd, refcount, sizeof(*refcount), 0);
    close(fd);
    return bytesRead == (ssize_t) sizeof(*refcount);
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

static void readComm(char* output, size_t outputSize)
{
    int const fd = open("/proc/self/comm", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        snprintf(output, outputSize, "?");
        return;
    }
    ssize_t const length = read(fd, output, outputSize - 1U);
    close(fd);
    if (length <= 0)
    {
        snprintf(output, outputSize, "?");
        return;
    }
    output[length] = '\0';
    if (output[length - 1] == '\n')
    {
        output[length - 1] = '\0';
    }
}

static void resolveUnlinkAtPath(int directoryFd, char const* path, char* output, size_t outputSize)
{
    if (path == NULL || path[0] == '\0' || path[0] == '/')
    {
        snprintf(output, outputSize, "%s", path == NULL ? "?" : path);
        return;
    }

    char directory[4096];
    if (directoryFd == AT_FDCWD)
    {
        if (getcwd(directory, sizeof(directory)) == NULL)
        {
            snprintf(output, outputSize, "%s", path);
            return;
        }
    }
    else
    {
        char fdPath[64];
        snprintf(fdPath, sizeof(fdPath), "/proc/self/fd/%d", directoryFd);
        readLinkOrUnknown(fdPath, directory, sizeof(directory));
        if (directory[0] == '?')
        {
            snprintf(output, outputSize, "%s", path);
            return;
        }
    }
    snprintf(output, outputSize, "%s/%s", directory, path);
}

static void traceEvent(char const* operation, char const* name, char const* resolvedPath, int flags, long result,
    int resultErrno, bool refcountValid, int32_t refcount, void* caller)
{
    char const* traceDirectory = getenv("TLLM_SHM_TRACE_DIR");
    if (traceDirectory == NULL)
    {
        return;
    }

    char tracePath[4096];
    int const pathLength = snprintf(tracePath, sizeof(tracePath), "%s/shm_trace.%d.log", traceDirectory, getpid());
    if (pathLength < 0 || (size_t) pathLength >= sizeof(tracePath))
    {
        return;
    }

    int const traceFd = open(tracePath, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (traceFd < 0)
    {
        return;
    }

    struct timespec timestamp;
    clock_gettime(CLOCK_REALTIME, &timestamp);

    Dl_info callerInfo = {0};
    dladdr(caller, &callerInfo);
    char const* callerObject = callerInfo.dli_fname != NULL ? callerInfo.dli_fname : "?";
    char const* callerSymbol = callerInfo.dli_sname != NULL ? callerInfo.dli_sname : "?";

    char executable[4096];
    char processName[256];
    char ipcNamespace[128];
    char mountNamespace[128];
    readLinkOrUnknown("/proc/self/exe", executable, sizeof(executable));
    readComm(processName, sizeof(processName));
    readLinkOrUnknown("/proc/self/ns/ipc", ipcNamespace, sizeof(ipcNamespace));
    readLinkOrUnknown("/proc/self/ns/mnt", mountNamespace, sizeof(mountNamespace));

    char refcountText[32];
    if (refcountValid)
    {
        snprintf(refcountText, sizeof(refcountText), "%d", refcount);
    }
    else
    {
        snprintf(refcountText, sizeof(refcountText), "?");
    }

    dprintf(traceFd,
        "time=%lld.%09ld op=%s pid=%d tid=%ld ppid=%d uid=%d euid=%d flags=0x%x result=%ld errno=%d "
        "refcount=%s name=%s resolved=%s comm=%s exe=%s ipcns=%s mntns=%s caller=%s object=%s\n",
        (long long) timestamp.tv_sec, timestamp.tv_nsec, operation, getpid(), syscall(SYS_gettid), getppid(), getuid(),
        geteuid(), flags, result, resultErrno, refcountText, name == NULL ? "?" : name,
        resolvedPath == NULL ? "-" : resolvedPath, processName, executable, ipcNamespace, mountNamespace, callerSymbol,
        callerObject);
    close(traceFd);
}

int shm_open(char const* name, int flags, mode_t mode)
{
    if (realShmOpen == NULL)
    {
        errno = ENOSYS;
        return -1;
    }

    int const result = realShmOpen(name, flags, mode);
    int const savedErrno = errno;
    if (isTorchShmName(name) && (((flags & O_CREAT) != 0) || result < 0))
    {
        traceEvent(
            "shm_open", name, NULL, flags, result, result < 0 ? savedErrno : 0, false, 0, __builtin_return_address(0));
    }
    errno = savedErrno;
    return result;
}

int shm_unlink(char const* name)
{
    if (realShmUnlink == NULL)
    {
        errno = ENOSYS;
        return -1;
    }

    int32_t refcount = 0;
    bool const refcountValid = isTorchShmName(name) && readRefcount(name, NULL, &refcount);
    int const result = realShmUnlink(name);
    int const savedErrno = errno;
    if (isTorchShmName(name))
    {
        traceEvent("shm_unlink", name, NULL, 0, result, result < 0 ? savedErrno : 0, refcountValid, refcount,
            __builtin_return_address(0));
    }
    errno = savedErrno;
    return result;
}

int unlink(char const* path)
{
    if (realUnlink == NULL)
    {
        errno = ENOSYS;
        return -1;
    }

    int32_t refcount = 0;
    bool const refcountValid = isTorchShmPath(path) && readRefcount(NULL, path, &refcount);
    int const result = realUnlink(path);
    int const savedErrno = errno;
    if (isTorchShmPath(path))
    {
        traceEvent("unlink", path, path, 0, result, result < 0 ? savedErrno : 0, refcountValid, refcount,
            __builtin_return_address(0));
    }
    errno = savedErrno;
    return result;
}

int unlinkat(int directoryFd, char const* path, int flags)
{
    if (realUnlinkAt == NULL)
    {
        errno = ENOSYS;
        return -1;
    }

    char resolvedPath[4096];
    resolveUnlinkAtPath(directoryFd, path, resolvedPath, sizeof(resolvedPath));
    int32_t refcount = 0;
    bool const refcountValid = isTorchShmPath(resolvedPath) && readRefcount(NULL, resolvedPath, &refcount);
    int const result = realUnlinkAt(directoryFd, path, flags);
    int const savedErrno = errno;
    if (isTorchShmPath(resolvedPath))
    {
        traceEvent("unlinkat", path, resolvedPath, flags, result, result < 0 ? savedErrno : 0, refcountValid, refcount,
            __builtin_return_address(0));
    }
    errno = savedErrno;
    return result;
}
