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

#include <cstdint>
#include <cstring>
#include <utility>

#include <cassert>

#include "saBuffer.h"
#include "saCudaCallable.h"
#include "saNamedType.h"

#include "tensorrt_llm/common/config.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding::suffix_automaton
{

template <typename Key, typename Value>
struct SAFlatHashMap
{

    struct Bucket
    {
        Key key;
        Value value;
        bool occupied = false;
    };

    size_t const mCapacity = 0;

    SA_CUDA_CALLABLE void clear()
    {
        std::memset(static_cast<void*>(data()), 0, sizeof(Bucket) * mCapacity);
    }

    SA_CUDA_CALLABLE explicit SAFlatHashMap(size_t capacity)
        : mCapacity(capacity)
    {
        clear();
    }

    SA_CUDA_CALLABLE SAFlatHashMap& operator=(SAFlatHashMap const& other)
    {
        assert(mCapacity == other.mCapacity);
        std::memcpy(static_cast<void*>(data()), static_cast<void const*>(other.data()), sizeof(Bucket) * mCapacity);
        return *this;
    }

    static constexpr size_t sizeBytes(size_t capacity)
    {
        return sizeof(size_t) + sizeof(Bucket) * capacity;
    }

    SA_CUDA_CALLABLE size_t sizeBytes() const
    {
        return sizeBytes(mCapacity);
    }

    SA_CUDA_CALLABLE size_t capacity() const
    {
        return mCapacity;
    }

    SA_CUDA_CALLABLE Bucket* data()
    {
        return reinterpret_cast<Bucket*>(reinterpret_cast<char*>(this) + sizeof(size_t));
    }

    SA_CUDA_CALLABLE Bucket const* data() const
    {
        return reinterpret_cast<Bucket const*>(reinterpret_cast<char const*>(this) + sizeof(size_t));
    }

    SA_CUDA_CALLABLE size_t hash(Key key) const
    {
        return static_cast<size_t>(+key) % mCapacity;
    }

    // returns nullptr if the key is not found or no space to insert
    SA_CUDA_CALLABLE Value* at(Key key, bool insertIfNotExists = false)
    {
        size_t index = hash(key);
        size_t originalIndex = index;

        // Linear probe to find the key or an empty slot
        do
        {
            Bucket& bucket = data()[index];

            if (!bucket.occupied)
            {
                // Empty slot found
                if (insertIfNotExists)
                {
                    // Insert new key-value pair
                    bucket.key = key;
                    bucket.value = Value{};
                    bucket.occupied = true;
                    return &bucket.value;
                }
                else
                {
                    // Key not found
                    return nullptr;
                }
            }
            else if (bucket.key == key)
            {
                // Key found
                return &bucket.value;
            }

            // Move to next slot (linear probe)
            index = (index + 1) % mCapacity;

        } while (index != originalIndex);

        return nullptr;
    }

    SA_CUDA_CALLABLE Value const* at(Key key) const
    {
        return const_cast<SAFlatHashMap*>(this)->at(key);
    }

    struct Iterator
    {
        SAFlatHashMap const* map;
        size_t index;

        SA_CUDA_CALLABLE Iterator(SAFlatHashMap const* map, size_t index)
            : map(map)
            , index(index)
        {
            // Skip to first occupied bucket
            while (this->index < map->mCapacity && !map->data()[this->index].occupied)
            {
                this->index++;
            }
        }

        SA_CUDA_CALLABLE std::pair<Key const&, Value&> operator*() const
        {
            Bucket const& bucket = map->data()[index];
            return {bucket.key, const_cast<Value&>(bucket.value)};
        }

        SA_CUDA_CALLABLE Iterator& operator++()
        {
            index++;
            // Skip to next occupied bucket
            while (index < map->mCapacity && !map->data()[index].occupied)
            {
                index++;
            }
            return *this;
        }

        SA_CUDA_CALLABLE bool operator==(Iterator const& other) const
        {
            return index == other.index;
        }

        SA_CUDA_CALLABLE bool operator!=(Iterator const& other) const
        {
            return !(*this == other);
        }
    };

    SA_CUDA_CALLABLE Iterator begin() const
    {
        return Iterator(this, 0);
    }

    SA_CUDA_CALLABLE Iterator end() const
    {
        return Iterator(this, mCapacity);
    }
};

template <typename Key, typename Value, size_t MaxSizeBytes>
struct SAHashMapAllocator
{

    using HashMap = SAFlatHashMap<Key, Value>;
    using Ptr = NamedType<size_t, struct HashMapPtrTag>;

    SA_CUDA_CALLABLE HashMap& at(Ptr ptr)
    {
        return reinterpret_cast<HashMap&>(mMem.at(ptr));
    }

    SA_CUDA_CALLABLE HashMap const& at(Ptr ptr) const
    {
        return reinterpret_cast<HashMap const&>(mMem.at(ptr));
    }

    SA_CUDA_CALLABLE Ptr alloc(size_t capacity)
    {
        // no available pointers, allocate a new one
        auto index = mMem.size();

        mMem.extend(HashMap::sizeBytes(capacity));

        new (&at(index)) HashMap(capacity);
        at(index).clear();

        return index;
    }

    SA_CUDA_CALLABLE void free(Ptr)
    {
        // todo
    }

    template <typename Func>
    void visitChunks(Func&& func) const
    {
        mMem.visitChunks(func);
    }

    SA_CUDA_CALLABLE void clear()
    {
        mMem.clear();
    }

    SADynamicBuffer<char, MaxSizeBytes, Ptr> mMem;
};

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
