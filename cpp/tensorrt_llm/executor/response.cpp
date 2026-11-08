/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/responseImpl.h"
#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{

Response::Response(IdType requestId, std::string errorMsg, std::optional<IdType> clientId)
    : mImpl(std::make_unique<Impl>(requestId, std::move(errorMsg), clientId))
{
}

Response::Response(IdType requestId, Result Result, std::optional<IdType> clientId)
    : mImpl(std::make_unique<Impl>(requestId, std::move(Result), clientId))
{
}

Response::~Response() = default;

Response::Response(Response const& other)
    : mImpl(std::make_unique<Impl>(*other.mImpl))
{
}

Response::Response(Response&& other) noexcept = default;

Response& Response::operator=(Response const& other)
{
    if (this != &other)
    {
        mImpl = std::make_unique<Impl>(*other.mImpl);
    }
    return *this;
}

Response& Response::operator=(Response&& other) noexcept = default;

bool Response::hasError() const
{
    return mImpl->hasError();
}

std::string const& Response::getErrorMsg() const
{
    return mImpl->getErrorMsg();
}

IdType Response::getRequestId() const
{
    return mImpl->getRequestId();
}

std::optional<IdType> Response::getClientId() const
{
    return mImpl->getClientId();
}

Result const& Response::getResult() const
{
    return mImpl->getResult();
}

} // namespace tensorrt_llm::executor
