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
#include "pcre2_regex.h"

#include <stdexcept>

#define PCRE2_CODE_UNIT_WIDTH 0
#include <pcre2.h>

class PCRERegex::Impl {
public:
    Impl(const std::string &pattern, int flags);
    Impl(Impl&& other);
    ~Impl();

    std::vector<std::string> get_all_matches(const std::string &text) const;
    std::vector<std::pair<std::string::size_type, std::string::size_type>> all_matches(const std::string &text) const;
    bool contains(const std::string& text) const;
    void replace_all(std::string &text, const std::string &replacement) const;

private:
    pcre2_code_8 *regex_ = nullptr;
};

PCRERegex::Impl::Impl(const std::string &pattern, int flags) {
    int error = 0;
    PCRE2_SIZE error_offset = 0;
    flags |= PCRE2_UCP | PCRE2_UTF;
    regex_ = pcre2_compile_8(reinterpret_cast<PCRE2_SPTR8>(pattern.c_str()), pattern.size(),
        static_cast<uint32_t>(flags), &error, &error_offset, nullptr);
    if (!regex_) {
        char buffer[512];
        pcre2_get_error_message_8(error, reinterpret_cast<PCRE2_UCHAR8 *>(buffer), sizeof(buffer));
        throw std::runtime_error(buffer);
    }
}

PCRERegex::Impl::~Impl() {
    if (regex_) {
        pcre2_code_free_8(regex_);
    }
}

std::vector<std::string> PCRERegex::Impl::get_all_matches(const std::string &text) const 
{
    std::vector<std::string> matches;
    auto pairs = all_matches(text);
    for(auto &x:pairs) {
        matches.emplace_back(text.substr(x.first, x.second));
    }
    return matches;
}

std::vector<std::pair<std::string::size_type, std::string::size_type>>
PCRERegex::Impl::all_matches(const std::string &text) const 
{
    std::vector<std::pair<std::string::size_type, std::string::size_type>> result;
    auto text_ptr = reinterpret_cast<PCRE2_SPTR8>(text.c_str());
    PCRE2_SIZE text_length = text.size();
    pcre2_match_data_8 *match_data = pcre2_match_data_create_from_pattern_8(regex_, nullptr);
    int start_offset = 0;
    int match_length;
    int rc;
    do {
        rc = pcre2_match_8(regex_, text_ptr, text_length, start_offset, 0, match_data, nullptr);
        if (rc >= 0) {
            PCRE2_SIZE *o_vec = pcre2_get_ovector_pointer_8(match_data);
            match_length = static_cast<int>(o_vec[1] - o_vec[0]);
            result.emplace_back(o_vec[0], match_length);
            start_offset = static_cast<int>(o_vec[1]);
        }
    } while (rc >= 0 && start_offset < text_length && match_length > 0);
    pcre2_match_data_free_8(match_data);
    return result;
}

bool PCRERegex::Impl::contains(const std::string& text) const 
{
    return !all_matches(text).empty();
}

void PCRERegex::Impl::replace_all(std::string &text, const std::string &replacement) const 
{
    std::string result;
    std::string::size_type last = 0;
    auto pairs = all_matches(text);
    for(auto &x:pairs) {
        result.append(text.substr(last, x.first - last));
        last = x.first + x.second;
        result.append(replacement);
    }
    result.append(text.substr(last));
    text = result;
}

PCRERegex::PCRERegex(const std::string &pattern, int flags)
    : impl_(std::make_shared<Impl>(pattern, flags))
{
}

PCRERegex::PCRERegex(const std::string &pattern)
    : impl_(std::make_shared<Impl>(pattern, 0))
{
}

PCRERegex::PCRERegex(PCRERegex && other)
    : impl_(std::move(other.impl_))
{
}

PCRERegex::~PCRERegex()
{
}


std::vector<std::string> PCRERegex::get_all_matches(const std::string &text) const 
{
    return impl_->get_all_matches(text);
}

void PCRERegex::replace_all(std::string &text, const std::string &replacement) const
{
    impl_->replace_all(text, replacement);
}

bool PCRERegex::contains(const std::string& text) const {
    return impl_->contains(text);
}

std::vector<std::pair<std::string::size_type, std::string::size_type>> PCRERegex::all_matches(const std::string &text) const
{
    return impl_->all_matches(text);
}