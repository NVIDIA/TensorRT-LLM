/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"

#include <NvInferRuntime.h>

#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

namespace tensorrt_llm::plugins
{

using namespace tensorrt_llm::common::op;

class BasePlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

class BasePluginV3 : public nvinfer1::IPluginV3,
                     public nvinfer1::IPluginV3OneCore,
                     public nvinfer1::IPluginV3OneBuild,
                     public nvinfer1::IPluginV3OneRuntime
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

class BaseCreator : public nvinfer1::IPluginCreator
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

class BaseCreatorV3 : public nvinfer1::IPluginCreatorV3One
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept
    {
        mNamespace = libNamespace;
    }

    [[nodiscard]] char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace{api::kDefaultNamespace};
};

} // namespace tensorrt_llm::plugins

// Init with O(n) and retrieve with O(1)
class PluginFieldParser
{
public:
    // field array must remain valid when calling getScalar() later.
    PluginFieldParser(int32_t nbFields, nvinfer1::PluginField const* fields);
    // delete to remind accidental mis-use (copy) which may result in false-alarm warnings about unused fields.
    PluginFieldParser(PluginFieldParser const&) = delete;
    PluginFieldParser& operator=(PluginFieldParser const&) = delete;
    // check if all fields are retrieved and emit warning if some of them are not.
    ~PluginFieldParser();
    template <typename T>
    std::optional<T> getScalar(std::string_view const& name);
    template <typename T>
    std::optional<std::set<T>> getSet(std::string_view const& name);

private:
    nvinfer1::PluginField const* mFields;

    struct Record
    {
        Record(int32_t idx)
            : index{idx}
        {
        }

        int32_t const index;
        bool retrieved{false};
    };

    std::unordered_map<std::string_view, Record> mMap;
};
