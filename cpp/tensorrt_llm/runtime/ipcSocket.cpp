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
#if ENABLE_MULTI_DEVICE

#include "ipcSocket.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <memory.h>
#include <nccl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

// Check system calls
#define SYSCHECKSYNC(statement, name, retval)                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        retval = (statement);                                                                                          \
        if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN))                               \
        {                                                                                                              \
            TLLM_LOG_INFO("Call to " name " returned %s, retrying", strerror(errno));                                  \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            break;                                                                                                     \
        }                                                                                                              \
    } while (true)

#define SYSCHECK(statement, name)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        int retval;                                                                                                    \
        SYSCHECKSYNC((statement), name, retval);                                                                       \
        if (retval == -1)                                                                                              \
        {                                                                                                              \
            TLLM_LOG_WARNING("Call to " name " failed: %s", strerror(errno));                                          \
            TLLM_NCCL_CHECK(ncclSystemError);                                                                          \
        }                                                                                                              \
    } while (false)

// Enable Linux abstract socket naming
#define USE_ABSTRACT_SOCKET
#define NCCL_IPC_SOCKNAME_LEN 64
#define NCCL_IPC_SOCKNAME_STR "/tmp/ub-socket-%d-%lx"

namespace tensorrt_llm::runtime
{
struct NcclIpcSocket
{
    int fd;
    char socketName[NCCL_IPC_SOCKNAME_LEN];
    uint32_t volatile* abortFlag;
    uint64_t hash;
};

/*
 * Create a Unix Domain Socket
 */
std::shared_ptr<NcclIpcSocket> ncclIpcSocketInit(int rank, uint64_t hash, uint32_t volatile* abortFlag)
{
    int fd = -1;
    struct sockaddr_un cliaddr;
    char temp[NCCL_IPC_SOCKNAME_LEN] = "";
    auto handle = std::make_shared<NcclIpcSocket>();

    if (handle == NULL)
    {
        TLLM_NCCL_CHECK(ncclInternalError); // throws
    }

    handle->hash = hash;
    handle->fd = -1;
    handle->socketName[0] = '\0';
    if ((fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0)
    {
        TLLM_LOG_WARNING("UDS: Socket creation error : %s (%d)", strerror(errno), errno);
        TLLM_NCCL_CHECK(ncclSystemError); // throws
    }

    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;

    // Create unique name for the socket.
    int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
    if (len > int(sizeof(cliaddr.sun_path) - 1))
    {
        TLLM_LOG_WARNING("UDS: Cannot bind provided name to socket. Name too large");
        TLLM_NCCL_CHECK(ncclInternalError); // throws
    }
#ifndef USE_ABSTRACT_SOCKET
    unlink(temp);
#endif

    TLLM_LOG_TRACE("UDS: Creating socket %s", temp);

    strncpy(cliaddr.sun_path, temp, len);
#ifdef USE_ABSTRACT_SOCKET
    cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif
    if (bind(fd, (struct sockaddr*) &cliaddr, sizeof(cliaddr)) < 0)
    {
        TLLM_LOG_WARNING("UDS: Binding to socket %s failed : %s (%d)", temp, strerror(errno), errno);
        close(fd);
        TLLM_NCCL_CHECK(ncclSystemError); // throws
    }

    handle->fd = fd;
    strcpy(handle->socketName, temp);

    handle->abortFlag = abortFlag;
    // Mark socket as non-blocking
    if (handle->abortFlag)
    {
        int flags = fcntl(fd, F_GETFL);
        TLLM_CHECK_WITH_INFO(flags != -1, "fcntl failed with error: %s", strerror(errno));
        SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
    }

    return handle;
}

void ncclIpcSocketClose(std::shared_ptr<NcclIpcSocket> handle)
{
    TLLM_CHECK_WITH_INFO(handle != nullptr, "handle is null.");
    if (handle->fd <= 0)
    {
        return; // nothing to do
    }
#ifndef USE_ABSTRACT_SOCKET
    if (handle->socketName[0] != '\0')
    {
        unlink(handle->socketName);
    }
#endif
    close(handle->fd);
}

void ncclIpcSocketRecvMsg(std::shared_ptr<NcclIpcSocket> handle, void* hdr, int hdrLen, int* recvFd)
{
    struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
    struct iovec iov[1];

    // Union to guarantee alignment requirements for control array
    union
    {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr* cmptr;
    char dummy_buffer[1];
    int ret;

    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    if (hdr == NULL)
    {
        iov[0].iov_base = (void*) dummy_buffer;
        iov[0].iov_len = sizeof(dummy_buffer);
    }
    else
    {
        iov[0].iov_base = hdr;
        iov[0].iov_len = hdrLen;
    }

    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
        {
            TLLM_LOG_WARNING("UDS: Receiving data over socket failed : %d", errno);
            TLLM_NCCL_CHECK(ncclSystemError); // throws
        }
        if (handle->abortFlag && *handle->abortFlag)
        {
            TLLM_NCCL_CHECK(ncclInternalError); // throws
        }
    }

    if (recvFd != NULL)
    {
        if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int))))
        {
            if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS))
            {
                TLLM_LOG_WARNING("UDS: Receiving data over socket failed");
                TLLM_NCCL_CHECK(ncclSystemError); // throws
            }

            memmove(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
        }
        else
        {
            TLLM_LOG_WARNING("UDS: Receiving data over socket %s failed", handle->socketName);
            TLLM_NCCL_CHECK(ncclSystemError); // throws
        }
        TLLM_LOG_TRACE("UDS: Got recvFd %d from socket %s", *recvFd, handle->socketName);
    }
}

int ncclIpcSocketRecvFd(std::shared_ptr<NcclIpcSocket> handle)
{
    int recvFd = -1;
    ncclIpcSocketRecvMsg(handle, NULL, 0, &recvFd);
    return recvFd;
}

void ncclIpcSocketSendMsg(
    std::shared_ptr<NcclIpcSocket> handle, void* hdr, int hdrLen, int sendFd, int rank, uint64_t hash)
{
    struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
    struct iovec iov[1];
    char temp[NCCL_IPC_SOCKNAME_LEN];

    union
    {
        struct cmsghdr cm;
        char control[CMSG_SPACE(sizeof(int))];
    } control_un;

    struct cmsghdr* cmptr;
    char dummy_buffer[1];
    struct sockaddr_un cliaddr;

    // Construct client address to send this shareable handle to
    bzero(&cliaddr, sizeof(cliaddr));
    cliaddr.sun_family = AF_UNIX;

    int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
    if (len > int(sizeof(cliaddr.sun_path) - 1))
    {
        TLLM_LOG_WARNING("UDS: Cannot connect to provided name for socket. Name too large");
        TLLM_NCCL_CHECK(ncclInternalError); // throws
    }
    (void) strncpy(cliaddr.sun_path, temp, len);

#ifdef USE_ABSTRACT_SOCKET
    cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
#endif

    TLLM_LOG_TRACE("UDS: Sending hdr %p len %d to UDS socket %s", hdr, hdrLen, temp);

    if (sendFd != -1)
    {
        TLLM_LOG_TRACE("UDS: Sending fd %d to UDS socket %s", sendFd, temp);

        msg.msg_control = control_un.control;
        msg.msg_controllen = sizeof(control_un.control);

        cmptr = CMSG_FIRSTHDR(&msg);
        cmptr->cmsg_len = CMSG_LEN(sizeof(int));
        cmptr->cmsg_level = SOL_SOCKET;
        cmptr->cmsg_type = SCM_RIGHTS;
        memmove(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));
    }

    msg.msg_name = (void*) &cliaddr;
    msg.msg_namelen = sizeof(struct sockaddr_un);

    if (hdr == NULL)
    {
        iov[0].iov_base = (void*) dummy_buffer;
        iov[0].iov_len = sizeof(dummy_buffer);
    }
    else
    {
        iov[0].iov_base = hdr;
        iov[0].iov_len = hdrLen;
    }
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    msg.msg_flags = 0;

    ssize_t sendResult;
    while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
        {
            TLLM_LOG_WARNING("UDS: Sending data over socket %s failed : %s (%d)", temp, strerror(errno), errno);
            TLLM_NCCL_CHECK(ncclSystemError); // throws
        }
        if (handle->abortFlag && *handle->abortFlag)
        {
            TLLM_NCCL_CHECK(ncclInternalError); // throws
        }
    }
}

void ncclIpcSocketSendFd(std::shared_ptr<NcclIpcSocket> handle, int sendFd, int rank)
{
    ncclIpcSocketSendMsg(handle, NULL, 0, sendFd, rank, handle->hash);
}

} // namespace tensorrt_llm::runtime

#endif // ENABLE_MULTI_DEVICE
