/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include "checkMacrosPlugin.h"
#include <cstdint>
#include <functional>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

PluginFieldParser::PluginFieldParser(int32_t nbFields, nvinfer1::PluginField const* fields)
    : mFields{fields}
{
    for (int32_t i = 0; i < nbFields; i++)
    {
        mMap.emplace(fields[i].name, PluginFieldParser::Record{i});
    }
}

PluginFieldParser::~PluginFieldParser()
{
    for (auto const& [name, record] : mMap)
    {
        if (!record.retrieved)
        {
            std::stringstream ss;
            ss << "unused plugin field with name: " << name;
            tensorrt_llm::plugins::logError(ss.str().c_str(), __FILE__, FN_NAME, __LINE__);
        }
    }
}

template <typename T>
nvinfer1::PluginFieldType toFieldType();
#define SPECIALIZE_TO_FIELD_TYPE(T, type)                                                                              \
    template <>                                                                                                        \
    nvinfer1::PluginFieldType toFieldType<T>()                                                                         \
    {                                                                                                                  \
        return nvinfer1::PluginFieldType::type;                                                                        \
    }
SPECIALIZE_TO_FIELD_TYPE(half, kFLOAT16)
SPECIALIZE_TO_FIELD_TYPE(float, kFLOAT32)
SPECIALIZE_TO_FIELD_TYPE(double, kFLOAT64)
SPECIALIZE_TO_FIELD_TYPE(int8_t, kINT8)
SPECIALIZE_TO_FIELD_TYPE(int16_t, kINT16)
SPECIALIZE_TO_FIELD_TYPE(int32_t, kINT32)
SPECIALIZE_TO_FIELD_TYPE(char, kCHAR)
SPECIALIZE_TO_FIELD_TYPE(nvinfer1::Dims, kDIMS)
SPECIALIZE_TO_FIELD_TYPE(void, kUNKNOWN)
#undef SPECIALIZE_TO_FIELD_TYPE

template <typename T>
std::optional<T> PluginFieldParser::getScalar(std::string_view const& name)
{
    auto const iter = mMap.find(name);
    if (iter == mMap.end())
    {
        return std::nullopt;
    }
    auto& record = mMap.at(name);
    auto const& f = mFields[record.index];
    TLLM_CHECK(toFieldType<T>() == f.type && f.length == 1);
    record.retrieved = true;
    return std::optional{*static_cast<T const*>(f.data)};
}

#define INSTANTIATE_PluginFieldParser_getScalar(T)                                                                     \
    template std::optional<T> PluginFieldParser::getScalar(std::string_view const&)
INSTANTIATE_PluginFieldParser_getScalar(half);
INSTANTIATE_PluginFieldParser_getScalar(float);
INSTANTIATE_PluginFieldParser_getScalar(double);
INSTANTIATE_PluginFieldParser_getScalar(int8_t);
INSTANTIATE_PluginFieldParser_getScalar(int16_t);
INSTANTIATE_PluginFieldParser_getScalar(int32_t);
INSTANTIATE_PluginFieldParser_getScalar(char);
INSTANTIATE_PluginFieldParser_getScalar(nvinfer1::Dims);
#undef INSTANTIATE_PluginFieldParser_getScalar

template <typename T>
std::optional<std::set<T>> PluginFieldParser::getSet(std::string_view const& name)
{
    auto const iter = mMap.find(name);
    if (iter == mMap.end())
    {
        return std::nullopt;
    }
    auto& record = mMap.at(name);
    auto const& f = mFields[record.index];
    TLLM_CHECK(toFieldType<T>() == f.type);
    std::set<T> group;
    auto const* r = static_cast<T const*>(f.data);
    for (int j = 0; j < f.length; ++j)
    {
        group.insert(*r);
        ++r;
    }

    record.retrieved = true;
    return std::optional{group};
}

#define INSTANTIATE_PluginFieldParser_getVector(T)                                                                     \
    template std::optional<std::set<T>> PluginFieldParser::getSet(std::string_view const&)
INSTANTIATE_PluginFieldParser_getVector(int32_t);
#undef INSTANTIATE_PluginFieldParser_getVector
