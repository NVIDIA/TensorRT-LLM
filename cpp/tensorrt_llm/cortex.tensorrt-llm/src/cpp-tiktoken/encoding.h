/*
 * Copyright (c) 2023 by Mark Tarrabain All rights reserved. Redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
 * disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided with the distribution.
 * Neither the name of the nor the names of its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#include "byte_pair_encoding.h"
#include "modelparams.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class IResourceReader;

class GptEncoding {
    int max_token_value_, n_words;
    std::unordered_map<std::string, int> special_token_mappings_;
    BytePairEncodingCore byte_pair_encoding_core_processor_;

public:
    GptEncoding(const std::string &pattern_string, const std::unordered_map<std::vector<uint8_t>, int, VectorHash> &byte_pair_ranks,
        const std::unordered_map<std::string, int> &special_token_mappings, int explicit_n_vocab);
    static std::shared_ptr<GptEncoding> get_encoding(LanguageModel model, IResourceReader* resource_reader = nullptr);
    static std::shared_ptr<GptEncoding> get_encoding_llama3(LanguageModel model, IResourceReader* resource_reader = nullptr);
    std::vector<int> encode(const std::string &line_to_encode, const std::unordered_set<std::string> &allowed_special = {},
        const std::unordered_set<std::string> &disallowed_special = { "all" });
    std::string decode(const std::vector<int> &input_tokens_to_decode);

    [[nodiscard]] const std::unordered_map<std::vector<uint8_t>, int, VectorHash>& get_byte_pair_token_map() const;
};