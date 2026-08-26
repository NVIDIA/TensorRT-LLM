/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

#include "tensorrt_llm/kernels/nvfp4ColdPageKernels.h"

#include <c10/util/intrusive_ptr.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <torch/extension.h>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

using kernels::Nvfp4ColdPageBufferPlan;
using kernels::ColdPageIndexPair;
using kernels::Nvfp4ColdPageKernelParams;
using kernels::Nvfp4ColdPagePreparedPlan;
using kernels::Nvfp4ColdPageRuntimeType;
using kernels::Nvfp4ColdPageTransform;

enum BufferIntegerField : std::size_t
{
    kRawBase,
    kRawSlotBytes,
    kRawBytes,
    kColdDataOffset,
    kColdScaleOffset,
    kColdPaddingOffset,
    kColdPaddingBytes,
    kTransform,
    kNumKvHeads,
    kTokensPerPage,
    kHeadDim,
    kNumBufferIntegerFields,
};

enum BufferScaleField : std::size_t
{
    kNvfp4ScaleOrigQuant,
    kNvfp4ScaleQuantOrig,
    kFp8ScaleOrigQuant,
    kFp8ScaleQuantOrig,
    kNumBufferScaleFields,
};

template <typename T>
T checkedNonNegativeCast(std::int64_t value, char const* field)
{
    TORCH_CHECK(value >= 0, field, " must be non-negative");
    TORCH_CHECK(static_cast<std::uint64_t>(value) <= static_cast<std::uint64_t>(std::numeric_limits<T>::max()), field,
        " exceeds its native range");
    return static_cast<T>(value);
}

Nvfp4ColdPageRuntimeType parseRuntimeType(std::int64_t value)
{
    switch (value)
    {
    case 0: return Nvfp4ColdPageRuntimeType::kFloat16;
    case 1: return Nvfp4ColdPageRuntimeType::kBfloat16;
    case 2: return Nvfp4ColdPageRuntimeType::kFp8E4m3;
    default: TORCH_CHECK(false, "Unsupported NVFP4 cold-page runtime type ", value);
    }
    return Nvfp4ColdPageRuntimeType::kFloat16;
}

Nvfp4ColdPageTransform parseTransform(std::int64_t value)
{
    switch (value)
    {
    case 0: return Nvfp4ColdPageTransform::kNvfp4;
    case 1: return Nvfp4ColdPageTransform::kLosslessCopy;
    default: TORCH_CHECK(false, "Unsupported NVFP4 cold-page transform ", value);
    }
    return Nvfp4ColdPageTransform::kNvfp4;
}

struct ColdPageInvocation
{
    std::uintptr_t coldBase;
    void const* pagePairs;
    std::size_t numPages;
    cudaStream_t stream;
};

ColdPageInvocation parseInvocation(
    std::int64_t coldBase, std::int64_t pagePairs, std::int64_t pageCount, std::int64_t stream)
{
    auto const coldAddress = checkedNonNegativeCast<std::uintptr_t>(coldBase, "cold_base");
    auto const pairsAddress = checkedNonNegativeCast<std::uintptr_t>(pagePairs, "page_pairs");
    auto const numPages = checkedNonNegativeCast<std::size_t>(pageCount, "page_count");
    auto const streamAddress = checkedNonNegativeCast<std::uintptr_t>(stream, "stream");
    if (numPages != 0)
    {
        TORCH_CHECK(coldAddress != 0, "cold_base must not be null");
        TORCH_CHECK(pairsAddress != 0, "page_pairs must not be null");
        TORCH_CHECK(pairsAddress % alignof(ColdPageIndexPair) == 0, "page_pairs is misaligned");
    }
    return {coldAddress, reinterpret_cast<void const*>(pairsAddress), numPages,
        reinterpret_cast<cudaStream_t>(streamAddress)};
}

} // namespace

//! Opaque, immutable configure-time plan consumed by the runtime custom ops.
class Nvfp4ColdPageProgram : public torch::CustomClassHolder
{
public:
    explicit Nvfp4ColdPageProgram(Nvfp4ColdPagePreparedPlan plan)
        : mPlan(std::move(plan))
    {
    }

    [[nodiscard]] Nvfp4ColdPagePreparedPlan const& getPlan() const noexcept
    {
        return mPlan;
    }

private:
    Nvfp4ColdPagePreparedPlan const mPlan;
};

c10::intrusive_ptr<Nvfp4ColdPageProgram> prepareNvfp4ColdPageProgram(
    c10::List<c10::List<std::int64_t>> const& bufferIntegers, c10::List<c10::List<double>> const& bufferScales,
    std::int64_t coldPageBytes, std::int64_t runtimeType)
{
    TORCH_CHECK(bufferIntegers.size() == bufferScales.size(), "Each cold-page buffer needs one scale row");

    std::vector<Nvfp4ColdPageBufferPlan> buffers;
    buffers.reserve(bufferIntegers.size());
    for (std::size_t index = 0; index < bufferIntegers.size(); ++index)
    {
        auto const integers = bufferIntegers.get(index);
        auto const scales = bufferScales.get(index);
        TORCH_CHECK(integers.size() == kNumBufferIntegerFields, "NVFP4 buffer ", index, " requires ",
            kNumBufferIntegerFields, " integer fields, got ", integers.size());
        TORCH_CHECK(scales.size() == kNumBufferScaleFields, "NVFP4 buffer ", index, " requires ", kNumBufferScaleFields,
            " scale fields, got ", scales.size());

        Nvfp4ColdPageBufferPlan buffer{};
        buffer.rawBase = checkedNonNegativeCast<std::uintptr_t>(integers.get(kRawBase), "raw_base");
        buffer.rawSlotBytes = checkedNonNegativeCast<std::size_t>(integers.get(kRawSlotBytes), "raw_slot_bytes");
        buffer.rawBytes = checkedNonNegativeCast<std::size_t>(integers.get(kRawBytes), "raw_bytes");
        buffer.coldDataOffset = checkedNonNegativeCast<std::size_t>(integers.get(kColdDataOffset), "cold_data_offset");
        buffer.coldScaleOffset
            = checkedNonNegativeCast<std::size_t>(integers.get(kColdScaleOffset), "cold_scale_offset");
        buffer.coldPaddingOffset
            = checkedNonNegativeCast<std::size_t>(integers.get(kColdPaddingOffset), "cold_padding_offset");
        buffer.coldPaddingBytes
            = checkedNonNegativeCast<std::uint32_t>(integers.get(kColdPaddingBytes), "cold_padding_bytes");
        buffer.transform = parseTransform(integers.get(kTransform));
        buffer.params = Nvfp4ColdPageKernelParams{
            checkedNonNegativeCast<std::int32_t>(integers.get(kNumKvHeads), "num_kv_heads"),
            checkedNonNegativeCast<std::int32_t>(integers.get(kTokensPerPage), "tokens_per_page"),
            checkedNonNegativeCast<std::int32_t>(integers.get(kHeadDim), "head_dim"),
            static_cast<float>(scales.get(kNvfp4ScaleOrigQuant)), static_cast<float>(scales.get(kNvfp4ScaleQuantOrig)),
            static_cast<float>(scales.get(kFp8ScaleOrigQuant)), static_cast<float>(scales.get(kFp8ScaleQuantOrig))};
        buffers.push_back(buffer);
    }

    auto plan = kernels::prepareNvfp4ColdPagePlan(
        buffers, checkedNonNegativeCast<std::size_t>(coldPageBytes, "cold_page_bytes"), parseRuntimeType(runtimeType));
    return c10::make_intrusive<Nvfp4ColdPageProgram>(std::move(plan));
}

void nvfp4ColdPageEncode(c10::intrusive_ptr<Nvfp4ColdPageProgram> const& program, std::int64_t coldBase,
    std::int64_t pagePairs, std::int64_t pageCount, std::int64_t stream)
{
    TORCH_CHECK(program, "NVFP4 cold-page program must not be null");
    auto const batch = parseInvocation(coldBase, pagePairs, pageCount, stream);
    kernels::invokeNvfp4ColdPageEncode(
        batch.pagePairs, batch.numPages, program->getPlan(), reinterpret_cast<void*>(batch.coldBase), batch.stream);
}

void nvfp4ColdPageDecode(c10::intrusive_ptr<Nvfp4ColdPageProgram> const& program, std::int64_t coldBase,
    std::int64_t pagePairs, std::int64_t pageCount, std::int64_t stream)
{
    TORCH_CHECK(program, "NVFP4 cold-page program must not be null");
    auto const batch = parseInvocation(coldBase, pagePairs, pageCount, stream);
    kernels::invokeNvfp4ColdPageDecode(batch.pagePairs, batch.numPages, program->getPlan(),
        reinterpret_cast<void const*>(batch.coldBase), batch.stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<tensorrt_llm::torch_ext::Nvfp4ColdPageProgram>("Nvfp4ColdPageProgram");
    m.def(
        "prepare_nvfp4_cold_page_program(int[][] buffer_ints, float[][] buffer_scales, int cold_page_bytes, "
        "int runtime_type) -> __torch__.torch.classes.trtllm.Nvfp4ColdPageProgram");
    m.def(
        "nvfp4_cold_page_encode(__torch__.torch.classes.trtllm.Nvfp4ColdPageProgram program, int cold_base, "
        "int page_pairs, int page_count, int stream) -> ()");
    m.def(
        "nvfp4_cold_page_decode(__torch__.torch.classes.trtllm.Nvfp4ColdPageProgram program, int cold_base, "
        "int page_pairs, int page_count, int stream) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CompositeExplicitAutograd, m)
{
    m.impl("prepare_nvfp4_cold_page_program", &tensorrt_llm::torch_ext::prepareNvfp4ColdPageProgram);
    m.impl("nvfp4_cold_page_encode", &tensorrt_llm::torch_ext::nvfp4ColdPageEncode);
    m.impl("nvfp4_cold_page_decode", &tensorrt_llm::torch_ext::nvfp4ColdPageDecode);
}
