/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tensorrt_llm::kv_cache_compression
{

//! Algorithm-neutral descriptor ABI borrowed by one cold-page callback.
struct alignas(8) ColdPageIndexPair
{
    std::int32_t dst;
    std::int32_t src;
};

static_assert(sizeof(ColdPageIndexPair) == 8);
static_assert(alignof(ColdPageIndexPair) == 8);
static_assert(offsetof(ColdPageIndexPair, dst) == 0);
static_assert(offsetof(ColdPageIndexPair, src) == 4);
static_assert(std::is_trivially_copyable_v<ColdPageIndexPair>);

} // namespace tensorrt_llm::kv_cache_compression
