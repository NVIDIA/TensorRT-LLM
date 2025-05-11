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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/executor.h"
#include <variant>

namespace tensorrt_llm::executor
{

class Response::Impl
{
public:
    Impl(IdType requestId, std::string errorMsg, std::optional<IdType> clientId)
        : mRequestId(requestId)
        , mErrOrResult(std::move(errorMsg))
        , mClientId(clientId)
    {
        TLLM_CHECK_WITH_INFO(!std::get<std::string>(mErrOrResult).empty(), "Error message should not be empty");
    }

    Impl(IdType requestId, Result Result, std::optional<IdType> clientId)
        : mRequestId(requestId)
        , mErrOrResult(std::move(Result))
        , mClientId(clientId)
    {
    }

    ~Impl() = default;

    [[nodiscard]] bool hasError() const
    {
        return std::holds_alternative<std::string>(mErrOrResult);
    }

    [[nodiscard]] bool hasResult() const
    {
        return std::holds_alternative<Result>(mErrOrResult);
    }

    [[nodiscard]] IdType getRequestId() const
    {
        return mRequestId;
    }

    [[nodiscard]] std::optional<IdType> getClientId() const
    {
        return mClientId;
    }

    /// Could throw exception if no result is available
    [[nodiscard]] Result const& getResult() const
    {
        if (hasResult())
        {
            return std::get<Result>(mErrOrResult);
        }

        TLLM_THROW(
            "Cannot get the result for a response with an error: %s", std::get<std::string>(mErrOrResult).c_str());
    }

    [[nodiscard]] std::string const& getErrorMsg() const
    {
        if (hasError())
        {
            return std::get<std::string>(mErrOrResult);
        }

        TLLM_THROW("Cannot get the error message for a response without error");
    }

private:
    friend class Serialization;
    IdType mRequestId;
    std::variant<std::string, Result> mErrOrResult;
    std::optional<IdType> mClientId = std::nullopt;
};

} // namespace tensorrt_llm::executor
