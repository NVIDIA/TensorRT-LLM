/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if !defined(__x86_64__)
#error "The NVBug 6336747 syscall tracer currently supports x86_64 only"
#endif

static int installUnlinkFilter(void)
{
    struct sock_filter const filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_unlink, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRACE),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_unlinkat, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRACE),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog const program = {
        .len = (unsigned short) (sizeof(filter) / sizeof(filter[0])),
        .filter = (struct sock_filter*) filter,
    };

    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0)
    {
        return -1;
    }
    return prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program);
}

static bool readTraceeString(pid_t const pid, unsigned long long const address, char* output, size_t const outputSize)
{
    size_t offset = 0;
    while (offset < outputSize)
    {
        errno = 0;
        long const word = ptrace(PTRACE_PEEKDATA, pid, (void*) (address + offset), NULL);
        if (word == -1 && errno != 0)
        {
            return false;
        }

        size_t const bytesToCopy = outputSize - offset < sizeof(word) ? outputSize - offset : sizeof(word);
        memcpy(output + offset, &word, bytesToCopy);
        if (memchr(&word, '\0', bytesToCopy) != NULL)
        {
            return true;
        }
        offset += bytesToCopy;
    }
    output[outputSize - 1] = '\0';
    return true;
}

static void readProcessName(pid_t const pid, char* output, size_t const outputSize)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/comm", pid);
    int const fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0)
    {
        snprintf(output, outputSize, "?");
        return;
    }

    ssize_t const length = read(fd, output, outputSize - 1);
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

static void readExecutable(pid_t const pid, char* output, size_t const outputSize)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/exe", pid);
    ssize_t const length = readlink(path, output, outputSize - 1);
    if (length < 0)
    {
        snprintf(output, outputSize, "?");
        return;
    }
    output[length] = '\0';
}

static void traceUnlink(int const traceFd, pid_t const pid)
{
    struct user_regs_struct registers;
    if (ptrace(PTRACE_GETREGS, pid, NULL, &registers) != 0)
    {
        return;
    }

    char const* operation;
    unsigned long long pathAddress;
    long long directoryFd = AT_FDCWD;
    unsigned long long flags = 0;
    if (registers.orig_rax == SYS_unlink)
    {
        operation = "unlink";
        pathAddress = registers.rdi;
    }
    else if (registers.orig_rax == SYS_unlinkat)
    {
        operation = "unlinkat";
        directoryFd = (int) registers.rdi;
        pathAddress = registers.rsi;
        flags = registers.rdx;
    }
    else
    {
        return;
    }

    char path[4096];
    if (!readTraceeString(pid, pathAddress, path, sizeof(path)))
    {
        snprintf(path, sizeof(path), "?");
    }
    char processName[256];
    char executable[4096];
    readProcessName(pid, processName, sizeof(processName));
    readExecutable(pid, executable, sizeof(executable));

    struct timespec timestamp;
    clock_gettime(CLOCK_REALTIME, &timestamp);
    long long const result = (long long) registers.rax;
    long long const resultErrno = result < 0 ? -result : 0;
    dprintf(traceFd,
        "time=%lld.%09ld pid=%d op=%s dirfd=%lld flags=0x%llx result=%lld errno=%lld path=%s comm=%s exe=%s\n",
        (long long) timestamp.tv_sec, timestamp.tv_nsec, pid, operation, directoryFd, flags, result, resultErrno, path,
        processName, executable);
}

static int setTraceOptions(pid_t const pid)
{
    long const options = PTRACE_O_TRACECLONE | PTRACE_O_TRACEEXEC | PTRACE_O_TRACEFORK | PTRACE_O_TRACESECCOMP
        | PTRACE_O_TRACEVFORK | PTRACE_O_TRACESYSGOOD;
    if (ptrace(PTRACE_SETOPTIONS, pid, NULL, options) != 0 && errno != ESRCH)
    {
        return -1;
    }
    return 0;
}

static int continueTracee(pid_t const pid, int const signal)
{
    if (setTraceOptions(pid) != 0)
    {
        return -1;
    }
    if (ptrace(PTRACE_CONT, pid, NULL, signal) != 0 && errno != ESRCH)
    {
        return -1;
    }
    return 0;
}

static int runTracer(int const traceFd, char* const command[])
{
    pid_t const child = fork();
    if (child < 0)
    {
        perror("fork");
        return EXIT_FAILURE;
    }
    if (child == 0)
    {
        close(traceFd);
        if (ptrace(PTRACE_TRACEME, 0, NULL, NULL) != 0)
        {
            perror("ptrace(PTRACE_TRACEME)");
            _exit(127);
        }
        raise(SIGSTOP);
        if (installUnlinkFilter() != 0)
        {
            perror("installUnlinkFilter");
            _exit(127);
        }
        execvp(command[0], command);
        perror("execvp");
        _exit(127);
    }

    int status;
    if (waitpid(child, &status, 0) != child || !WIFSTOPPED(status))
    {
        fprintf(stderr, "NVBug 6336747 syscall tracer failed to stop child %d\n", child);
        return EXIT_FAILURE;
    }
    if (continueTracee(child, 0) != 0)
    {
        perror("continueTracee");
        return EXIT_FAILURE;
    }

    while (true)
    {
        pid_t const pid = waitpid(-1, &status, __WALL);
        if (pid < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            if (errno == ECHILD)
            {
                return EXIT_FAILURE;
            }
            perror("waitpid");
            return EXIT_FAILURE;
        }

        if (WIFEXITED(status))
        {
            if (pid == child)
            {
                return WEXITSTATUS(status);
            }
            continue;
        }
        if (WIFSIGNALED(status))
        {
            if (pid == child)
            {
                return 128 + WTERMSIG(status);
            }
            continue;
        }
        if (!WIFSTOPPED(status))
        {
            continue;
        }

        int const stopSignal = WSTOPSIG(status);
        unsigned int const event = (unsigned int) status >> 16U;
        if (stopSignal == SIGTRAP && event == PTRACE_EVENT_SECCOMP)
        {
            if (setTraceOptions(pid) != 0 || ptrace(PTRACE_SYSCALL, pid, NULL, 0) != 0)
            {
                perror("continueToSyscallExit");
                return EXIT_FAILURE;
            }
            continue;
        }
        if (stopSignal == (SIGTRAP | 0x80))
        {
            traceUnlink(traceFd, pid);
        }
        int const deliverySignal
            = event != 0U || stopSignal == SIGSTOP || stopSignal == (SIGTRAP | 0x80) ? 0 : stopSignal;
        if (continueTracee(pid, deliverySignal) != 0)
        {
            perror("continueTracee");
            return EXIT_FAILURE;
        }
    }
}

int main(int const argc, char* const argv[])
{
    if (argc < 4 || strcmp(argv[2], "--") != 0)
    {
        fprintf(stderr, "Usage: %s TRACE_FILE -- COMMAND [ARG ...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int const traceFd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (traceFd < 0)
    {
        perror("open trace file");
        return EXIT_FAILURE;
    }
    int const result = runTracer(traceFd, &argv[3]);
    close(traceFd);
    return result;
}
