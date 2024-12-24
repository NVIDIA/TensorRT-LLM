/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <memory.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tensorrt_llm::runtime::ub
{

typedef enum
{
    ipcSocketSuccess = 0,
    ipcSocketUnhandledCudaError = 1,
    ipcSocketSystemError = 2,
    ipcSocketInternalError = 3,
    ipcSocketInvalidArgument = 4,
    ipcSocketInvalidUsage = 5,
    ipcSocketRemoteError = 6,
    ipcSocketInProgress = 7,
    ipcSocketNumResults = 8
} ipcSocketResult_t;

char const* ipcSocketGetErrorString(ipcSocketResult_t res);

#define IPC_SOCKNAME_LEN 64

struct IpcSocketHandle
{
    int fd;
    char socketName[IPC_SOCKNAME_LEN];
    uint32_t volatile* abortFlag;
};

ipcSocketResult_t ipcSocketInit(IpcSocketHandle* handle, int rank, uint64_t hash, uint32_t volatile* abortFlag);
ipcSocketResult_t ipcSocketClose(IpcSocketHandle* handle);
ipcSocketResult_t ipcSocketGetFd(IpcSocketHandle* handle, int* fd);

ipcSocketResult_t ipcSocketRecvFd(IpcSocketHandle* handle, int* fd);
ipcSocketResult_t ipcSocketSendFd(IpcSocketHandle* handle, int const fd, int rank, uint64_t hash);
} // namespace tensorrt_llm::runtime::ub
