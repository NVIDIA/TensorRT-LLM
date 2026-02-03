/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * Adapted from Baseten's sa_spec library (Apache-2.0)
 * https://github.com/basetenlabs/sa_spec
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <unordered_map>

#include <cassert>

#include "saConfig.h"
#include "saCudaCallable.h"
#include "saFlatGraph.h"
#include "saNamedType.h"
#include "saOptional.h"
#include "saTypes.h"
#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

struct SuffixAutomaton
{

    struct NodeData
    {
        int len{0};
        SAOptional<NodeIndex> link;
        SAOptional<TextIndex> pos;
    };

    using Graph = SAFlatGraph<Token, NodeData, 2 * SAConfig::MAX_SEQUENCE_LENGTH>;
    using TokenVec = SADynamicBuffer<Token, SAConfig::MAX_SEQUENCE_LENGTH, TextIndex>;

    Graph mStates;
    TokenVec mTokens;
    NodeIndex mLast;

    template <typename Func>
    void visitChunks(Func&& func) const
    {
        mStates.visitChunks(func);
        mTokens.visitChunks(func);
        func(static_cast<void const*>(&mLast), sizeof(mLast));
    }

    SuffixAutomaton() = default;

    // prevent accidental move construction
    SuffixAutomaton(SuffixAutomaton&& other) = delete;

    void clear()
    {
        mStates.clear();
        mTokens.clear();
        mLast = NodeIndex(0);
    }

    SA_CUDA_CALLABLE void extend(Token token)
    {
        if (mStates.empty())
        {
            // add root state
            NodeIndex root = mStates.size();
            NodeData& rootData = mStates.pushBack();
            rootData = NodeData{};
            mLast = root;
        }

        mTokens.pushBack(token);

        NodeIndex cur = mStates.size();
        NodeData& curData = mStates.pushBack();
        curData.len = +mStates.at(mLast).len + 1;
        curData.link = SAOptional<NodeIndex>();
        curData.pos = TextIndex(+mTokens.size() - 1);

        auto pOpt = SAOptional<NodeIndex>(mLast);
        while (pOpt.hasValue() && mStates.at(*pOpt, token) == nullptr)
        {
            *mStates.at(*pOpt, token, true) = cur;
            pOpt = mStates.at(*pOpt).link;
        }

        if (!pOpt.hasValue())
        {
            curData.link = NodeIndex(0);
        }
        else
        {
            NodeIndex p = pOpt.value();

            NodeIndex* qPtr = mStates.at(p, token);
            assert(qPtr != nullptr);
            NodeIndex q = *qPtr;

            auto& pData = mStates.at(p);

            if (pData.len + 1 == mStates.at(q).len)
            {
                curData.link = q;
            }
            else
            {
                NodeIndex clone = mStates.size();
                auto& cloneData = mStates.pushBackClone(q);
                cloneData.len = pData.len + 1;

                auto pCurrentOpt = SAOptional<NodeIndex>(p);
                while (pCurrentOpt.hasValue())
                {
                    NodeIndex* edgePtr = mStates.at(*pCurrentOpt, token);
                    if (edgePtr == nullptr || *edgePtr != q)
                    {
                        break;
                    }
                    *edgePtr = clone;
                    pCurrentOpt = mStates.at(*pCurrentOpt).link;
                }

                mStates.at(q).link = clone;
                mStates.at(cur).link = clone;
            }
        }
        mLast = cur;
    }

    struct LookupResult
    {
        TextIndex pos{0};
        int len{0};
    };

    SA_CUDA_CALLABLE SAOptional<LookupResult> lookup() const
    {
        if (mStates.empty())
        {
            return SAOptional<LookupResult>();
        }

        NodeIndex state = mLast;
        NodeIndex bestState = NodeIndex(0);

        // Walk up the suffix links to find the longest proper suffix that appears earlier
        while (state != NodeIndex(0))
        {
            auto& nodeData = mStates.at(state);
            SAOptional<TextIndex> posOpt = nodeData.pos;
            assert(!posOpt.hasValue() || +*posOpt <= +mTokens.size());

            bool isLast = posOpt.hasValue() && +*posOpt + 1 >= +mTokens.size();

            if (!isLast)
            {
                bestState = state;
                break;
            }

            auto linkOpt = nodeData.link;
            if (!linkOpt.hasValue())
            {
                break;
            }
            state = *linkOpt;
        }

        if (bestState == NodeIndex(0))
        {
            return SAOptional<LookupResult>();
        }

        auto posOpt = mStates.at(bestState).pos;
        if (!posOpt.hasValue())
        {
            return SAOptional<LookupResult>();
        }

        assert(*posOpt < mTokens.size());

        // Return the position after the suffix match
        TextIndex matchEnd = TextIndex(+*posOpt + 1);

        LookupResult result;
        result.pos = matchEnd;
        result.len = mStates.at(bestState).len;
        return SAOptional<LookupResult>(result);
    }

    SA_CUDA_CALLABLE void getDraftTokens(Token::ValueType* buf, int bufLen, TextIndex startPos) const
    {
        int availableLen = +mTokens.size() - +startPos;
        bufLen = (bufLen < availableLen) ? bufLen : availableLen;
        for (int i = 0; i < bufLen; i++)
        {
            buf[i] = +mTokens.at(TextIndex(+startPos + i));
        }
    }
};

static_assert(std::is_trivially_copyable<SuffixAutomaton>::value, "SuffixAutomaton must be trivially copyable");

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
