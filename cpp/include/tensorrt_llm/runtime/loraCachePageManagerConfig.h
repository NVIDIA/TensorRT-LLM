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

#pragma once

#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <NvInferRuntime.h>

#include <ostream>
#include <sstream>
#include <type_traits>

namespace tensorrt_llm::runtime
{
/**
 * Configuration for LoraCachePageManager
 *
 * See LoraCache docs for description of pages, slots, and page blocks.
 */
class LoraCachePageManagerConfig
{
public:
    explicit constexpr LoraCachePageManagerConfig(runtime::MemoryType memType, nvinfer1::DataType dType,
        SizeType totalNumPages, SizeType maxPagesPerBlock, SizeType slotsPerPage, SizeType pageWidth,
        SizeType numCopyStreams)
        : mMemoryType(memType)
        , mDataType(dType)
        , mTotalNumPages(totalNumPages)
        , mMaxPagesPerBlock(maxPagesPerBlock)
        , mSlotsPerPage(slotsPerPage)
        , mPageWidth(pageWidth)
        , mInitToZero(false)
    {
    }

    [[nodiscard]] runtime::MemoryType constexpr getMemoryType() const noexcept
    {
        return mMemoryType;
    }

    void constexpr setMemoryType(runtime::MemoryType const& memoryType) noexcept
    {
        mMemoryType = memoryType;
    }

    [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept
    {
        return mDataType;
    }

    void constexpr setDataType(nvinfer1::DataType const& dtype) noexcept
    {
        mDataType = dtype;
    }

    [[nodiscard]] SizeType constexpr getTotalNumPages() const noexcept
    {
        return mTotalNumPages;
    }

    void constexpr setTotalNumPage(SizeType const& totalNumPages) noexcept
    {
        mTotalNumPages = totalNumPages;
    }

    [[nodiscard]] SizeType constexpr getMaxPagesPerBlock() const noexcept
    {
        return mMaxPagesPerBlock;
    }

    void constexpr setMaxPagesPerBlock(SizeType const& maxPagesPerBlock) noexcept
    {
        mMaxPagesPerBlock = maxPagesPerBlock;
    }

    [[nodiscard]] SizeType constexpr getSlotsPerPage() const noexcept
    {
        return mSlotsPerPage;
    }

    void constexpr setSlotsPerPage(SizeType const& slotsPerPage) noexcept
    {
        mSlotsPerPage = slotsPerPage;
    }

    [[nodiscard]] SizeType constexpr getPageWidth() const noexcept
    {
        return mPageWidth;
    }

    void constexpr setPageWidth(SizeType const& pageWidth) noexcept
    {
        mPageWidth = pageWidth;
    }

    [[nodiscard]] bool constexpr getInitToZero() const noexcept
    {
        return mInitToZero;
    }

    void constexpr setInitToZero(bool initToZero) noexcept
    {
        mInitToZero = initToZero;
    }

    [[nodiscard]] SizeType constexpr getNumCopyStreams() const noexcept
    {
        return mNumCopyStreams;
    }

    void constexpr setNumCopyStreams(SizeType numCopyStreams) noexcept
    {
        mNumCopyStreams = numCopyStreams;
    }

private:
    runtime::MemoryType mMemoryType;
    nvinfer1::DataType mDataType;

    /*
     * Number cache pages in the cache.
     * Generally corresponds to the number of opt sized LoRAs that can be stored in the cache
     */
    SizeType mTotalNumPages;
    // number of pages to allocate in one block
    SizeType mMaxPagesPerBlock;
    // number of slots per page, where a slot corresponds to a adapterSize=1, 1-layer, 1-module set or weights
    SizeType mSlotsPerPage;
    SizeType mPageWidth;

    // number of streams used to copy pages to device cache
    SizeType mNumCopyStreams = 1;

    bool mInitToZero; // for testing
};

inline std::ostream& operator<<(std::ostream& os, LoraCachePageManagerConfig const& c)
{
    os << "{"
       << "memoryType=" << static_cast<typename std::underlying_type<runtime::MemoryType>::type>(c.getMemoryType())
       << " dataType=" << static_cast<typename std::underlying_type<nvinfer1::DataType>::type>(c.getDataType())
       << " totalNumPages=" << c.getTotalNumPages() << " maxPagesPerBlock=" << c.getMaxPagesPerBlock()
       << " slotsPerPage=" << c.getSlotsPerPage() << " pageWidth=" << c.getPageWidth()
       << " initToZero=" << c.getInitToZero() << "}";
    return os;
}

inline std::string to_string(LoraCachePageManagerConfig const& c)
{
    std::stringstream sstream;
    sstream << c;
    return sstream.str();
}
} // namespace tensorrt_llm::runtime
