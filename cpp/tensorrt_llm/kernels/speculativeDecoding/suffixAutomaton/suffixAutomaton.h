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

#include <cassert>

#include "saCudaCallable.h"
#include "saFlatGraph.h"
#include "saNamedType.h"
#include "saOptional.h"
#include "saTypes.h"
#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

/**
 * @brief Helper function to align a size up to a given alignment.
 */
inline constexpr size_t alignUp(size_t size, size_t alignment)
{
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Suffix Automaton with runtime-configurable maximum sequence length.
 *
 * This is a compact state machine that recognizes all suffixes of a string.
 * It's used for speculative decoding to find longest patterns in previously
 * generated tokens to predict future tokens.
 *
 * Memory Layout (for a single slot):
 * [SuffixAutomaton header][nodes data][tokens data][allocator data]
 *
 * The header contains pointers that point INTO the same memory block.
 * Total size is computed by getRequiredMemorySize(maxSeqLen).
 */
struct SuffixAutomaton
{

    struct NodeData
    {
        int len{0};
        SAOptional<NodeIndex> link;
        SAOptional<TextIndex> pos;
    };

    using Graph = SAFlatGraph<Token, NodeData>;
    using TokenVec = SADynamicBuffer<Token, TextIndex>;

    Graph mStates;
    TokenVec mTokens;
    NodeIndex mLast;
    size_t mMaxSeqLen;

    SuffixAutomaton() = default;

    // prevent accidental move construction
    SuffixAutomaton(SuffixAutomaton&& other) = delete;

    /**
     * @brief Calculate the required memory size for a given max sequence length.
     *
     * Memory layout: [header][nodes][tokens][allocator]
     * Each section is aligned to 64 bytes for GPU cache efficiency.
     *
     * @param maxSeqLen Maximum sequence length to support
     * @return Total size in bytes needed for one SuffixAutomaton slot
     */
    static size_t getRequiredMemorySize(size_t maxSeqLen)
    {
        size_t maxNodes = 2 * maxSeqLen;

        size_t headerSize = sizeof(SuffixAutomaton);
        size_t nodeSize = maxNodes * sizeof(Graph::Node);
        size_t tokenSize = maxSeqLen * sizeof(Token);
        // Allocator needs space for hash maps: ~10x nodes * (key + value size)
        size_t allocatorSize = 10 * maxNodes * (sizeof(Token) + sizeof(NodeIndex));

        return alignUp(headerSize, 64) + alignUp(nodeSize, 64) + alignUp(tokenSize, 64) + alignUp(allocatorSize, 64);
    }

    /**
     * @brief Initialize the suffix automaton with external memory.
     *
     * Must be called before any operations on the automaton.
     * The memory block must be at least getRequiredMemorySize(maxSeqLen) bytes.
     *
     * @param base Pointer to the start of the memory block (where this struct lives)
     * @param maxSeqLen Maximum sequence length to support
     */
    void init(void* base, size_t maxSeqLen)
    {
        mMaxSeqLen = maxSeqLen;
        size_t maxNodes = 2 * maxSeqLen;

        // Data starts after the header (aligned)
        uint8_t* ptr = static_cast<uint8_t*>(base) + alignUp(sizeof(SuffixAutomaton), 64);

        // Initialize nodes buffer
        Graph::Node* nodeData = reinterpret_cast<Graph::Node*>(ptr);
        ptr += alignUp(maxNodes * sizeof(Graph::Node), 64);

        // Initialize tokens buffer
        Token* tokenData = reinterpret_cast<Token*>(ptr);
        ptr += alignUp(maxSeqLen * sizeof(Token), 64);

        // Initialize allocator
        char* allocatorMemory = reinterpret_cast<char*>(ptr);
        size_t allocatorCapacity = 10 * maxNodes * (sizeof(Token) + sizeof(NodeIndex));

        // Set up the graph and tokens
        mStates.init(nodeData, maxNodes, allocatorMemory, allocatorCapacity);
        mTokens.init(tokenData, maxSeqLen);

        // Reset state
        mLast = NodeIndex(0);
    }

    /**
     * @brief Relocate all internal pointers by a given delta.
     *
     * Used when copying between host and GPU memory to adjust pointers.
     * Call this on the host-side copy BEFORE memcpy to GPU.
     *
     * @param oldBase The current base address (host address)
     * @param newBase The target base address (GPU address)
     */
    void relocate(void* oldBase, void* newBase)
    {
        ptrdiff_t delta = static_cast<uint8_t*>(newBase) - static_cast<uint8_t*>(oldBase);
        mStates.relocate(delta);
        mTokens.relocate(delta);
    }

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

    /**
     * @brief Find suffix of exactly targetLen that appears earlier in the text.
     *
     * This is used for fixed-size ngram matching where we want to find
     * a match of a specific length rather than the longest possible match.
     *
     * Uses O(L) complexity by walking suffix links. Each state in the SA covers
     * a range of lengths [link.len + 1, state.len]. If targetLen falls within
     * this range and the state has an earlier occurrence, we have a match.
     *
     * @param targetLen The exact length of suffix to match
     * @return Optional LookupResult with position after match and length,
     *         or empty if no match found
     */
    SA_CUDA_CALLABLE SAOptional<LookupResult> lookupFixed(int targetLen) const
    {
        if (mStates.empty() || targetLen <= 0)
        {
            return SAOptional<LookupResult>();
        }

        int textLen = +mTokens.size();

        // Need at least targetLen + 1 tokens to have a non-self match
        if (textLen < targetLen + 1)
        {
            return SAOptional<LookupResult>();
        }

        NodeIndex state = mLast;

        // Walk up suffix links to find a state that covers our target length
        while (state != NodeIndex(0))
        {
            auto& nodeData = mStates.at(state);

            // Determine the minimum length this state covers
            // Each state covers lengths [minLen, nodeData.len]
            int minLen = 1; // Default for states linked directly from root
            auto linkOpt = nodeData.link;
            if (linkOpt.hasValue() && *linkOpt != NodeIndex(0))
            {
                minLen = mStates.at(*linkOpt).len + 1;
            }

            // Check if this state covers our target length
            if (nodeData.len >= targetLen && minLen <= targetLen)
            {
                SAOptional<TextIndex> posOpt = nodeData.pos;
                if (posOpt.hasValue())
                {
                    TextIndex pos = *posOpt;
                    // Check if this is an earlier occurrence (not at current suffix end)
                    // pos is the END position of an occurrence
                    if (+pos < textLen - 1)
                    {
                        LookupResult result;
                        result.pos = TextIndex(+pos + 1); // Continuation starts after match
                        result.len = targetLen;
                        return SAOptional<LookupResult>(result);
                    }
                }
            }

            // If we've gone below target length entirely, stop searching
            if (nodeData.len < targetLen)
            {
                break;
            }

            if (!linkOpt.hasValue())
            {
                break;
            }
            state = *linkOpt;
        }

        return SAOptional<LookupResult>();
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
