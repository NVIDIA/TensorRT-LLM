/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ipUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

std::string getLocalIpByNic(std::string const& interface, int rank)
{
    struct ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == -1)
    {
        TLLM_LOG_ERROR(rank,
            "getLocalIpByNic: Can't get local ip from NIC Interface. Please check whether corresponding INTERFACE is "
            "set "
            "correctly.");
        return std::string{};
    }

    for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next)
    {
        if (ifa->ifa_addr == nullptr)
        {
            continue;
        }

        if (ifa->ifa_name == interface)
        {
            if (ifa->ifa_addr->sa_family == AF_INET)
            {
                char ip[INET_ADDRSTRLEN]{};
                void* addr = &((reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr))->sin_addr);
                if ((inet_ntop(AF_INET, addr, ip, sizeof(ip)) != nullptr) && std::strcmp(ip, "0.0.0.0") != 0)
                {
                    freeifaddrs(ifaddr);
                    return std::string(ip);
                }
            }
            else if (ifa->ifa_addr->sa_family == AF_INET6)
            {
                char ip[INET6_ADDRSTRLEN]{};
                void* addr = &((reinterpret_cast<struct sockaddr_in6*>(ifa->ifa_addr))->sin6_addr);
                if ((inet_ntop(AF_INET6, addr, ip, sizeof(ip)) != nullptr) && std::strncmp(ip, "fe80::", 6) != 0
                    && std::strcmp(ip, "::1") != 0)
                {
                    freeifaddrs(ifaddr);
                    return std::string(ip);
                }
            }
        }
    }

    freeifaddrs(ifaddr);
    TLLM_LOG_ERROR(
        rank, "Can't get local ip from NIC Interface. Please check whether corresponding INTERFACE is set correctly.");
    return std::string{};
}

std::string getLocalIpByHostname(int rank)
{
    char hostname[256]{};
    if (gethostname(hostname, sizeof(hostname)) == -1)
    {
        TLLM_LOG_ERROR(rank, "getLocalIpByHostname: Can't get hostname");
        return std::string{};
    }

    struct addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_CANONNAME;

    struct addrinfo* res = nullptr;
    if (getaddrinfo(hostname, nullptr, &hints, &res) != 0)
    {
        TLLM_LOG_WARNING(rank, "getLocalIpByHostname: Can't get address info for hostname");
        return std::string{};
    }

    for (struct addrinfo* p = res; p != nullptr; p = p->ai_next)
    {

        if (p->ai_family == AF_INET)
        { // IPv4
            char ip[INET_ADDRSTRLEN]{};
            struct sockaddr_in* ipv4 = reinterpret_cast<struct sockaddr_in*>(p->ai_addr);
            void* addr = &(ipv4->sin_addr);
            if ((inet_ntop(AF_INET, addr, ip, sizeof(ip)) != nullptr) && std::strcmp(ip, "127.0.0.1") != 0
                && std::strcmp(ip, "0.0.0.0") != 0)
            {
                freeaddrinfo(res);
                return std::string(ip);
            }
        }
        else if (p->ai_family == AF_INET6)
        { // IPv6
            char ip[INET6_ADDRSTRLEN]{};
            struct sockaddr_in6* ipv6 = reinterpret_cast<struct sockaddr_in6*>(p->ai_addr);
            void* addr = &(ipv6->sin6_addr);
            if ((inet_ntop(AF_INET6, addr, ip, sizeof(ip)) != nullptr) && std::strncmp(ip, "fe80::", 6) != 0
                && std::strcmp(ip, "::1") != 0)
            {
                freeaddrinfo(res);
                return std::string(ip);
            }
        }
    }

    freeaddrinfo(res);
    TLLM_LOG_WARNING(rank, "getLocalIpByHostname: Can't get local ip from hostname");
    return std::string{};
}

std::string getLocalIpByRemoteOrHostName(int rank)
{

    // Try IPv4
    struct sockaddr_in addr
    {
    };

    addr.sin_family = AF_INET;
    addr.sin_port = htons(80);
    // using google's public dns server to get the local ip which can be accessed from remote
    char const* dns_ip_v4 = "8.8.8.8";
    inet_pton(AF_INET, dns_ip_v4, &addr.sin_addr);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock != -1)
    {
        if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != -1)
        {
            socklen_t addr_len = sizeof(addr);
            if (getsockname(sock, reinterpret_cast<struct sockaddr*>(&addr), &addr_len) != -1)
            {
                char ip[INET_ADDRSTRLEN]{};
                inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
                close(sock);
                return std::string(ip);
            }
        }
        close(sock);
    }

    // Try IPv6
    struct sockaddr_in6 addr6
    {
    };

    addr6.sin6_family = AF_INET6;
    addr6.sin6_port = htons(80);
    // using google's public dns server
    char const* dns_ipv6 = "2001:4860:4860::8888";
    inet_pton(AF_INET6, dns_ipv6, &addr6.sin6_addr);

    sock = socket(AF_INET6, SOCK_DGRAM, 0);
    if (sock != -1)
    {
        if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr6), sizeof(addr6)) != -1)
        {
            socklen_t addr_len = sizeof(addr6);
            if (getsockname(sock, reinterpret_cast<struct sockaddr*>(&addr6), &addr_len) != -1)
            {
                char ip[INET6_ADDRSTRLEN]{};
                inet_ntop(AF_INET6, &addr6.sin6_addr, ip, sizeof(ip));
                close(sock);
                return std::string(ip);
            }
        }
        close(sock);
    }

    // Try hostname
    return getLocalIpByHostname(rank);
}

std::string getLocalIp(std::string interface, int rank)
{
    std::string localIP = {};
    if (!interface.empty())
    {
        localIP = getLocalIpByNic(interface, rank);
    }
    if (localIP.empty())
    {
        localIP = getLocalIpByRemoteOrHostName(rank);
    }
    // check whether the localIP is valid
    if (localIP.empty())
    {
        TLLM_THROW("getLocalIp: Can't get local ip");
    }
    return localIP;
}
} // namespace common

TRTLLM_NAMESPACE_END
