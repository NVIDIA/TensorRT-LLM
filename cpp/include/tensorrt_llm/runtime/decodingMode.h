/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

namespace tensorrt_llm
{
namespace runtime
{

class DecodingMode
{
public:
    static auto constexpr None()
    {
        return DecodingMode{kNone};
    }

    static auto constexpr TopK()
    {
        return DecodingMode{kTopK};
    }

    static auto constexpr TopP()
    {
        return DecodingMode{kTopP};
    }

    static auto constexpr TopKTopP()
    {
        return DecodingMode{kTopKTopP};
    }

    static auto constexpr BeamSearch()
    {
        return DecodingMode{kBeamSearch};
    }

    bool constexpr isNone()
    {
        return mState == 0;
    }

    bool constexpr isTopK()
    {
        return anyBitSet(kTopK);
    }

    bool constexpr isTopP()
    {
        return anyBitSet(kTopP);
    }

    bool constexpr isTopKorTopP()
    {
        return anyBitSet(kTopKTopP);
    }

    bool constexpr isTopKandTopP()
    {
        return allBitSet(kTopKTopP);
    }

    bool constexpr isBeamSearch()
    {
        return anyBitSet(kBeamSearch);
    }

    using UnderlyingType = uint8_t;

private:
    constexpr DecodingMode(UnderlyingType state)
        : mState(state)
    {
    }

    // No mode specified. Config will be determined from the beam width of the first request at runtime
    // TopKTopP if beamWidth == 1, BeamSearch otherwise
    static UnderlyingType constexpr kNone{0};
    static UnderlyingType constexpr kTopK{1u << 0};
    static UnderlyingType constexpr kTopP{1u << 1};
    static UnderlyingType constexpr kBeamSearch{1u << 2};
    static UnderlyingType constexpr kTopKTopP{kTopK | kTopP};

    bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    bool constexpr allBitSet(UnderlyingType bits) const
    {
        return (mState & bits) == bits;
    }

    UnderlyingType mState{};
};

static_assert(DecodingMode::None().isNone());
static_assert(!DecodingMode::None().isTopK());
static_assert(!DecodingMode::None().isTopP());
static_assert(!DecodingMode::None().isBeamSearch());

static_assert(DecodingMode::TopK().isTopK());
static_assert(DecodingMode::TopK().isTopKorTopP());
static_assert(!DecodingMode::TopK().isTopKandTopP());
static_assert(!DecodingMode::TopK().isTopP());
static_assert(!DecodingMode::TopK().isBeamSearch());

static_assert(DecodingMode::TopP().isTopP());
static_assert(DecodingMode::TopP().isTopKorTopP());
static_assert(!DecodingMode::TopP().isTopKandTopP());
static_assert(!DecodingMode::TopP().isTopK());
static_assert(!DecodingMode::TopP().isBeamSearch());

static_assert(DecodingMode::TopKTopP().isTopK());
static_assert(DecodingMode::TopKTopP().isTopP());
static_assert(DecodingMode::TopKTopP().isTopKorTopP());
static_assert(DecodingMode::TopKTopP().isTopKandTopP());
static_assert(!DecodingMode::TopKTopP().isBeamSearch());

static_assert(DecodingMode::BeamSearch().isBeamSearch());
static_assert(!DecodingMode::BeamSearch().isTopKorTopP());

} // namespace runtime
} // namespace tensorrt_llm
