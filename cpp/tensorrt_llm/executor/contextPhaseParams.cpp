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

#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/serialization.h"
#include "tensorrt_llm/executor/serializeUtils.h"

#include <optional>

namespace su = tensorrt_llm::executor::serialize_utils;

namespace tensorrt_llm::executor
{

ContextPhaseParams::ContextPhaseParams(VecTokens firstGenTokens, RequestIdType reqId, void* state)
    : mReqId{reqId}
    , mFirstGenTokens{std::move(firstGenTokens)}
    , mState{StatePtr{state, deleter}}
{
}

ContextPhaseParams::ContextPhaseParams(VecTokens firstGenTokens, RequestIdType reqId)
    : mReqId{reqId}
    , mFirstGenTokens{std::move(firstGenTokens)}
{
}

ContextPhaseParams::ContextPhaseParams(
    VecTokens firstGenTokens, RequestIdType reqId, std::vector<char> const& serializedState)
    : mReqId{reqId}
    , mFirstGenTokens{std::move(firstGenTokens)}
{

    su::VectorWrapBuf<char> strbuf(const_cast<std::vector<char>&>(serializedState));
    std::istream is(&strbuf);

    auto dataTransceiverState = Serialization::deserializeDataTransceiverState(is);
    auto dataTransceiverStatePtr = std::make_unique<executor::DataTransceiverState>(std::move(dataTransceiverState));
    mState = StatePtr{dataTransceiverStatePtr.release(), deleter};
}

ContextPhaseParams::ContextPhaseParams(ContextPhaseParams const& other)
{
    // Since the internal header files implement the destructor while using the declaration of this
    // type, a `unique_ptr` with a custom destructor member is used here.
    mReqId = other.mReqId;
    mFirstGenTokens = other.mFirstGenTokens;
    if (other.mState)
    {
        auto* otherState = static_cast<DataTransceiverState*>(other.mState.get());
        mState = StatePtr{std::make_unique<DataTransceiverState>(*otherState).release(), deleter};
    }
}

ContextPhaseParams::ContextPhaseParams(ContextPhaseParams&&) noexcept = default;

ContextPhaseParams& ContextPhaseParams::operator=(ContextPhaseParams const& other)
{
    *this = ContextPhaseParams{other};
    return *this;
}

ContextPhaseParams& ContextPhaseParams::operator=(ContextPhaseParams&&) noexcept = default;

ContextPhaseParams::~ContextPhaseParams() = default;

VecTokens const& ContextPhaseParams::getFirstGenTokens() const& noexcept
{
    return mFirstGenTokens;
}

VecTokens ContextPhaseParams::popFirstGenTokens() && noexcept
{
    return std::move(mFirstGenTokens);
}

ContextPhaseParams::RequestIdType ContextPhaseParams::getReqId() const noexcept
{
    return mReqId;
}

void const* ContextPhaseParams::getState() const noexcept
{
    return mState.get();
}

void* ContextPhaseParams::getState() noexcept
{
    return mState.get();
}

std::vector<char> ContextPhaseParams::getSerializedState() const noexcept
{
    return Serialization::serialize(*static_cast<DataTransceiverState const*>(mState.get()));
}

void* ContextPhaseParams::releaseState() noexcept
{
    return mState.release();
}

void ContextPhaseParams::deleter(void const* data)
{
    using StateT = DataTransceiverState const;
    std::default_delete<StateT>()(static_cast<StateT*>(data));
}

bool ContextPhaseParams::operator==(ContextPhaseParams const& other) const noexcept
{
    if (mFirstGenTokens != other.mFirstGenTokens || mReqId != other.mReqId
        || static_cast<bool>(mState) != static_cast<bool>(other.mState))
    {
        return false;
    }
    return !mState
        || *static_cast<DataTransceiverState const*>(mState.get())
        == *static_cast<DataTransceiverState const*>(other.mState.get());
}

} // namespace tensorrt_llm::executor
