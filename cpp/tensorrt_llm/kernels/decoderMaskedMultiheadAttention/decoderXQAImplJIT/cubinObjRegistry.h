/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "cubinObj.h"

#include "compileEngine.h"
#include "serializationUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"
#include <functional>
#include <unordered_map>

namespace tensorrt_llm
{
namespace kernels
{
namespace jit
{

// A collection of CubinObjs, with caching functionality.
template <typename Key, class Hash = std::hash<Key>>
class CubinObjRegistryTemplate
{
public:
    CubinObjRegistryTemplate() = default;

    CubinObjRegistryTemplate(void const* buffer_, size_t buffer_size)
    {
        size_t remaining_buffer_size = buffer_size;
        uint8_t const* buffer = static_cast<uint8_t const*>(buffer_);
        // First 4 bytes: num of elements.
        uint32_t n = readFromBuffer<uint32_t>(buffer, remaining_buffer_size);

        for (uint32_t i = 0; i < n; ++i)
        {
            uint32_t key_size = readFromBuffer<uint32_t>(buffer, remaining_buffer_size);
            TLLM_CHECK(key_size <= remaining_buffer_size);
            Key key(buffer, key_size);
            buffer += key_size;
            remaining_buffer_size -= key_size;

            uint32_t obj_size = readFromBuffer<uint32_t>(buffer, remaining_buffer_size);
            TLLM_CHECK(obj_size <= remaining_buffer_size);
            CubinObj obj(buffer, obj_size);
            buffer += obj_size;
            remaining_buffer_size -= obj_size;

            mMap.insert({key, std::move(obj)});
        }
        TLLM_CHECK(remaining_buffer_size == 0);
    }

    std::unique_ptr<CubinObjRegistryTemplate<Key, Hash>> clone() const noexcept
    {
        auto result = std::make_unique<CubinObjRegistryTemplate<Key, Hash>>();
        for (auto const& p : mMap)
        {
            result->mMap.insert(p);
        }
        return result;
    }

    size_t getSerializationSize() const noexcept
    {
        size_t result = sizeof(uint32_t);
        for (auto&& p : mMap)
        {
            result += 2 * sizeof(uint32_t);
            result += p.first.getSerializationSize() + p.second.getSerializationSize();
        }
        return result;
    }

    void serialize(void* buffer_, size_t buffer_size) const noexcept
    {
        size_t remaining_buffer_size = buffer_size;
        uint8_t* buffer = static_cast<uint8_t*>(buffer_);
        uint32_t n = mMap.size();
        writeToBuffer<uint32_t>(n, buffer, remaining_buffer_size);
        for (auto&& p : mMap)
        {
            uint32_t key_size = p.first.getSerializationSize();
            TLLM_CHECK(key_size <= remaining_buffer_size);
            writeToBuffer<uint32_t>(key_size, buffer, remaining_buffer_size);
            p.first.serialize(buffer, key_size);
            buffer += key_size;
            remaining_buffer_size -= key_size;

            uint32_t obj_size = p.second.getSerializationSize();
            TLLM_CHECK(obj_size <= remaining_buffer_size);
            writeToBuffer<uint32_t>(obj_size, buffer, remaining_buffer_size);
            p.second.serialize(buffer, obj_size);
            buffer += obj_size;
            remaining_buffer_size -= obj_size;
        }
        TLLM_CHECK(remaining_buffer_size == 0);
    }

    // Returns directly if the Cubin already exists in the registry, otherwise call compileEngine to compile it.
    //
    // compileEngine may be nullptr.
    CubinObj* getCubin(Key const& key, CompileEngine* compileEngine)
    {
        auto iter = mMap.find(key);
        if (iter != mMap.end())
        {
            return &(iter->second);
        }

        TLLM_CHECK_WITH_INFO(compileEngine != nullptr, "Key not found; compileEngine shouldn't be nullptr.");

        CubinObj obj = compileEngine->compile();
        auto insertResultIter = mMap.insert({key, std::move(obj)}).first;
        return &(insertResultIter->second);
    }

    void clear()
    {
        mMap.clear();
    }

private:
    std::unordered_map<Key, CubinObj, Hash> mMap;
};

using CubinObjKey = XQAKernelFullHashKey;
using CubinObjHasher = XQAKernelFullHasher;
using CubinObjRegistry = CubinObjRegistryTemplate<CubinObjKey, CubinObjHasher>;

} // namespace jit
} // namespace kernels
} // namespace tensorrt_llm
