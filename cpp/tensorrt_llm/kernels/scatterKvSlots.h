/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Strides, in elements, of a paged K/V pool ``[pages, 2, heads, tokens_per_page, head_dim]``
//! (the head_dim stride is 1) and of a ``[tokens, heads, head_dim]`` K or V source.
struct ScatterKvSlotsParams
{
    void* pool;
    int64_t poolPageStride;
    int64_t poolKvStride;
    int64_t poolHeadStride;
    int64_t poolSlotStride;
    int64_t tokensPerPage;

    void const* k;
    void const* v;
    int64_t kTokenStride;
    int64_t kHeadStride;
    int64_t vTokenStride;
    int64_t vHeadStride;

    int64_t numHeads;
    int64_t headDim;

    //! Entry ``e`` copies source token ``src[e]`` (``e`` when ``src`` is null) to the pool slot
    //! ``dst[e]`` and, when ``dst2`` is not null, also to ``dst2[e]``. A slot id is
    //! ``page * tokensPerPage + slot``. Every head of K and V is written.
    int64_t const* dst;
    int64_t const* dst2;
    int64_t const* src;
    int64_t numEntries;
};

//! Scatter K/V rows into a paged pool. Each source row is read once whatever the number of
//! destinations. ``k`` and ``v`` must not overlap the pool. Two entries naming the same slot
//! race; the result is one of the two writes. ``elemSize`` is the element size in bytes; the copy is a byte copy, so
//! any dtype. Copies in the widest of 16, 8, 4, 2 or 1 bytes that divides the row size, every base pointer and every
//! stride.
void invokeScatterKvSlots(ScatterKvSlotsParams const& params, int elemSize, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
