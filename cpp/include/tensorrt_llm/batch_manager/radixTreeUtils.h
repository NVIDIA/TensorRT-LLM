/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <string>
#include <cstdio>
#include "tensorrt_llm/batch_manager/radixTree.h"

namespace tensorrt_llm::batch_manager::radix_tree
{

class RadixTreeStringSet : public RadixTree<char,std::hash<char>,int,std::hash<int>,int,false>
{
    public:
        RadixTreeStringSet() = default;

        void insert(std::string str)
        {
            std::vector<char> prefix(str.begin(), str.end());
            auto matches = insertNodes(prefix);
            printf("1. prefix.size() = %ld, str.length() = %ld, matches.exactMatches.size() = %ld\n", prefix.size(), str.length(), matches.exactMatches.size());
            auto last_match = matches.exactMatches.back();
            [[maybe_unused]] auto wasOverwritten = last_match.node->setValue(1, static_cast<int>(str.size()), /*overwrite*/true); // store value for last node so nodes don't get deleted.
        }

        void erase(std::string str)
        {
            std::vector<char> prefix(str.begin(), str.end());
            auto matches = lookupNodes(prefix, /*allowPartialMatch*/false);
            if (matches.exactMatches.size() == prefix.size())
            {
                auto last_match = matches.exactMatches.back();
                [[maybe_unused]] auto wasCleared = last_match.node->clearValue(1); // clearing value should delete all empty nodes.
            }
        }

        [[nodiscard]] bool contains(std::string str) const
        {
            std::vector<char> prefix(str.begin(), str.end());
            auto matches = lookupNodes(prefix, /*allowPartialMatch*/false);
            printf("2. prefix.size() = %ld, str.length() = %ld, matches.exactMatches.size() = %ld\n", prefix.size(), str.length(), matches.exactMatches.size());
            if (matches.exactMatches.size() == prefix.size())
            {
                auto last_match = matches.exactMatches.back();
                auto last_val = last_match.node->getValue(1);
                return last_val.has_value();
            }
            return false;
        }
};

}
