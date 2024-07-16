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
#include "encoding.h"
#include "modelparams.h"
#include "pcre2_regex.h"

#include <stdexcept>
#include <fmt/format.h>
#include <iostream>
#define PCRE2_CODE_UNIT_WIDTH 0
#include <pcre2.h>

GptEncoding::GptEncoding(const std::string &pattern_string, const std::unordered_map<std::vector<uint8_t>, int, VectorHash> &byte_pair_ranks,
    const std::unordered_map<std::string, int> &special_token_mappings, int explicit_n_vocab) :
    max_token_value_(0),
    n_words(explicit_n_vocab),
    special_token_mappings_(special_token_mappings),
    byte_pair_encoding_core_processor_(byte_pair_ranks, special_token_mappings,
        std::make_shared<PCRERegex>(pattern_string, PCRE2_CASELESS)) { }

std::shared_ptr<GptEncoding> GptEncoding::get_encoding(LanguageModel model, IResourceReader* resource_reader)
{
    ModelParams model_params = ModelParamsGenerator::get_model_params(model, resource_reader);
    // std::cout<<"Len mergeble rank" <<model_params.mergeable_ranks().size()<<"\n";
    return std::make_shared<GptEncoding>(model_params.pat_str(), model_params.mergeable_ranks(),
        model_params.special_tokens(), model_params.explicit_n_vocab());
}

// Implement with custom additional params for LLAMA 3

std::shared_ptr<GptEncoding> GptEncoding::get_encoding_llama3(LanguageModel model, IResourceReader* resource_reader)
{
    ModelParams model_params = ModelParamsGenerator::get_model_params(model, resource_reader);
    std::cout<<"Len mergeble rank" <<model_params.mergeable_ranks().size()<<"\n";
    int num_reserved_special_tokens = 256;
    int num_base_tokens = model_params.mergeable_ranks().size();
    std::unordered_map<std::string, int> special_llama3_token_mappings;
    std::cout<<"Model pat string"<< model_params.pat_str()<<"\n"; 
    std::vector<std::string> list_special_tokens = {"<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>"};
    for(int i=5;i<num_reserved_special_tokens-5;i++){
        list_special_tokens.push_back(fmt::format("<|reserved_special_token_{}|>", i));
    }
    for (int i = 0 ; i < list_special_tokens.size();i++){
        special_llama3_token_mappings.insert({list_special_tokens[i],num_base_tokens+i});
    }

    return std::make_shared<GptEncoding>(model_params.pat_str(), model_params.mergeable_ranks(),
        special_llama3_token_mappings, model_params.explicit_n_vocab());
}

std::vector<int> GptEncoding::encode(const std::string &line_to_encode, const std::unordered_set<std::string> &allowed_special,
    const std::unordered_set<std::string> &disallowed_special)
{
    // Check for disallowed special tokens
    if (disallowed_special.count("all") > 0) {
        for (const auto &special_token: special_token_mappings_) {
            if (line_to_encode.find(special_token.first) != std::string::npos) {
                throw std::invalid_argument("Disallowed special token found: " + special_token.first);
            }
        }
    }
    // Call the encode_native function from the BytePairEncodingCore class
    return byte_pair_encoding_core_processor_.encode_native(line_to_encode, allowed_special).first;
}

std::string GptEncoding::decode(const std::vector<int> &input_tokens_to_decode)
{
    // Call the decode_native function from the BytePairEncodingCore class
    return byte_pair_encoding_core_processor_.decode_native(input_tokens_to_decode);
}

const std::unordered_map<std::vector<uint8_t>, int, VectorHash>& GptEncoding::get_byte_pair_token_map() const {
    return byte_pair_encoding_core_processor_.getBytePairRanks();
}
