/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kv_cache_manager_v2/common.h"

#include <cstddef>
#include <memory>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// StaticHostMem — host memory the GPU can reach, sized once.
//
// For an allocation that is made once and released once: no resize, no partial
// release, and no reader that has to wait for part of it. That is every staging
// buffer, which is why they do not go through HostMem and its fill worker,
// watermark and per-unit registration bookkeeping.
//
// The driver allocates and registers in one call, so the range is one
// registration: a copy may span all of it, unlike a range the host tier
// registers unit by unit.
// ---------------------------------------------------------------------------
class StaticHostMem
{
public:
    //! Holds nothing. Move-assign an allocated one over it to fill it.
    StaticHostMem() = default;

    //! A size of zero holds nothing, as the default constructor does.
    //!
    //! Throws CuError when the allocation fails, which is also how an oversized
    //! request is reported -- there is no internal chunking to fall back on.
    explicit StaticHostMem(size_t size);

    [[nodiscard]] bool empty() const noexcept
    {
        return mSize == 0;
    }

    [[nodiscard]] MemAddress address() const noexcept
    {
        return reinterpret_cast<MemAddress>(mPtr.get());
    }

    [[nodiscard]] size_t size() const noexcept
    {
        return mSize;
    }

private:
    struct Deleter
    {
        void operator()(void* ptr) const noexcept;
    };

    std::unique_ptr<void, Deleter> mPtr;
    size_t mSize = 0;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
