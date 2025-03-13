/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#if ENABLE_BF16
#include <cuda_bf16.h>
#endif // ENABLE_BF16
#include <cuda_fp16.h>

#include <memory>  // std::make_unique
#include <sstream> // std::stringstream
#include <string>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::common
{
#if ENABLE_BF16
static inline std::basic_ostream<char>& operator<<(std::basic_ostream<char>& stream, __nv_bfloat16 const& val)
{
    stream << __bfloat162float(val);
    return stream;
}
#endif // ENABLE_BF16

static inline std::basic_ostream<char>& operator<<(std::basic_ostream<char>& stream, __half const& val)
{
    stream << __half2float(val);
    return stream;
}

// Add forward declaration before printElement functions
template <typename... Args>
std::ostream& operator<<(std::ostream& os, std::tuple<Args...> const& t);

namespace
{

// Print element - default case for non-tuple types
template <typename T>
void printElement(std::ostream& os, T const& t)
{
    os << t;
}

// Print tuple implementation
template <typename Tuple, std::size_t... Is>
void printTupleImpl(std::ostream& os, Tuple const& t, std::index_sequence<Is...>)
{
    os << "(";
    ((Is == 0 ? os : (os << ", "), printElement(os, std::get<Is>(t))), ...);
    os << ")";
}

// Print element - specialized for tuples
template <typename... Args>
void printElement(std::ostream& os, std::tuple<Args...> const& t)
{
    printTupleImpl(os, t, std::index_sequence_for<Args...>{});
}

} // namespace

// Override operator<< for any tuple
template <typename... Args>
std::ostream& operator<<(std::ostream& os, std::tuple<Args...> const& t)
{
    printElement(os, t);
    return os;
}

template <typename... Args>
std::string to_string(std::tuple<Args...> const& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

inline std::string fmtstr(std::string const& s)
{
    return s;
}

inline std::string fmtstr(std::string&& s)
{
    return s;
}

#if defined(_MSC_VER)
std::string fmtstr(char const* format, ...);
#else
std::string fmtstr(char const* format, ...) __attribute__((format(printf, 1, 2)));
#endif

// __PRETTY_FUNCTION__ is used for neat debugging printing but is not supported on Windows
// The alternative is __FUNCSIG__, which is similar but not identical
#if defined(_WIN32)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

auto constexpr kDefaultDelimiter = ", ";

template <typename U, typename TStream, typename T>
inline TStream& arr2outCasted(TStream& out, T* arr, size_t size, char const* delim = kDefaultDelimiter)
{
    out << "(";
    if (size > 0)
    {
        for (size_t i = 0; i < size - 1; ++i)
        {
            out << static_cast<U>(arr[i]) << delim;
        }
        out << static_cast<U>(arr[size - 1]);
    }
    out << ")";
    return out;
}

template <typename TStream, typename T>
inline TStream& arr2out(TStream& out, T* arr, size_t size, char const* delim = kDefaultDelimiter)
{
    return arr2outCasted<T>(out, arr, size, delim);
}

template <typename T>
inline std::string arr2str(T* arr, size_t size, char const* delim = kDefaultDelimiter)
{
    std::stringstream ss;
    return arr2out(ss, arr, size, delim).str();
}

template <typename T>
inline std::string vec2str(std::vector<T> const& vec, char const* delim = kDefaultDelimiter)
{
    return arr2str(vec.data(), vec.size(), delim);
}

inline bool strStartsWith(std::string const& str, std::string const& prefix)
{
    return str.rfind(prefix, 0) == 0;
}

/// @brief Split a string into a set of strings using a delimiter
std::unordered_set<std::string> str2set(std::string const& input, char delimiter);

} // namespace tensorrt_llm::common
