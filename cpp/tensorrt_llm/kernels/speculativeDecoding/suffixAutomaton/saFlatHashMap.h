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

/**
 * @brief A memory allocator for hash maps with runtime-configurable capacity.
 *
 * Uses external memory (pointer-based) instead of embedded arrays.
 * The allocator manages a pool of memory for dynamically creating hash maps.
 *
 * @tparam Key Key type for the hash maps
 * @tparam Value Value type for the hash maps
 */
template <typename Key, typename Value>
struct SAHashMapAllocator
{

    using HashMap = SAFlatHashMap<Key, Value>;
    using Ptr = NamedType<size_t, struct HashMapPtrTag>;

    // Memory pool (pointer to external memory)
    char* mMemory{nullptr};
    size_t mCapacity{0};
    Ptr mUsed{0};

    SAHashMapAllocator() = default;

    SA_CUDA_CALLABLE void init(char* memory, size_t capacity)
    {
        mMemory = memory;
        mCapacity = capacity;
        mUsed = Ptr(0);
    }

    SA_CUDA_CALLABLE HashMap& at(Ptr ptr)
    {
        assert(mMemory != nullptr);
        return reinterpret_cast<HashMap&>(mMemory[+ptr]);
    }

    SA_CUDA_CALLABLE HashMap const& at(Ptr ptr) const
    {
        assert(mMemory != nullptr);
        return reinterpret_cast<HashMap const&>(mMemory[+ptr]);
    }

    SA_CUDA_CALLABLE Ptr alloc(size_t capacity)
    {
        assert(mMemory != nullptr);

        // Allocate from current position
        auto index = mUsed;
        size_t required = HashMap::sizeBytes(capacity);

        assert(+mUsed + required <= mCapacity);
        mUsed = Ptr(+mUsed + required);

        new (&at(index)) HashMap(capacity);
        at(index).clear();

        return index;
    }

    SA_CUDA_CALLABLE void free(Ptr)
    {
        // No-op - memory is managed externally
    }

    SA_CUDA_CALLABLE void clear()
    {
        mUsed = Ptr(0);
    }

    /**
     * @brief Relocate the memory pointer by a given delta.
     *
     * Used when copying between host and GPU memory to adjust pointers.
     */
    void relocate(ptrdiff_t delta)
    {
        if (mMemory)
        {
            mMemory = reinterpret_cast<char*>(reinterpret_cast<uint8_t*>(mMemory) + delta);
        }
    }
};

// Verify that SAHashMapAllocator is trivially copyable
static_assert(
    std::is_trivially_copyable<SAHashMapAllocator<int, int>>::value, "SAHashMapAllocator must be trivially copyable");

} // namespace kernels::speculative_decoding::suffix_automaton

TRTLLM_NAMESPACE_END
