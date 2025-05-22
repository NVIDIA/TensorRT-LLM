/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/executor/serialization.h"
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

class CommState;

struct DataContext
{
public:
    explicit DataContext(int tag)
        : mTag{tag}
    {
    }

    [[nodiscard]] int getTag() const noexcept
    {
        return mTag;
    }

private:
    int const mTag;
};

class Connection
{
public:
    virtual ~Connection() = default;

    virtual void send(DataContext const& ctx, void const* data, size_t size) const = 0;

    virtual void recv(DataContext const& ctx, void* data, size_t size) const = 0;

    [[nodiscard]] virtual bool isThreadSafe() const noexcept
    {
        return false;
    }
};

class ConnectionManager
{
public:
    virtual ~ConnectionManager() = default;

    virtual Connection const* recvConnect(DataContext const& ctx, void* data, size_t size) = 0;

    [[nodiscard]] virtual std::vector<Connection const*> getConnections(CommState const& state) = 0;

    [[nodiscard]] virtual CommState const& getCommState() const = 0;
};

} // namespace tensorrt_llm::executor::kv_cache
