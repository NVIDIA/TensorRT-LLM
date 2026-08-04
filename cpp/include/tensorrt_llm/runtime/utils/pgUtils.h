/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/torch.h>
#include <vector>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"

// Check async op.
inline c10::intrusive_ptr<c10d::Work> pgCheckHelper(
    c10::intrusive_ptr<c10d::Work> work, char const* const file, int const line, char const* info)
{
    if (work == nullptr)
    {
        auto const msg = std::string("[TensorRT-LLM][ERROR] empty work returned from: ") + info;
        tensorrt_llm::common::throwRuntimeError(file, line, msg);
    }

    try
    {
        work->wait();
    }
    catch (...)
    {
        auto msg = std::string("[TensorRT-LLM][ERROR] Torch distributed operation error: ") + info;
        std::throw_with_nested(tensorrt_llm::common::TllmException(file, line, msg.c_str()));
    }

    return work;
}

// Check sync op.
inline void pgCheckHelper(bool success, char const* const file, int const line, char const* info)
{
    if (!success)
    {
        throw std::runtime_error(std::string("[TensorRT-LLM][ERROR] Torch distributed operation error: ") + info);
    }
}

#define PGCHECK_THROW(op) pgCheckHelper(op, __FILE__, __LINE__, #op)
#define PGCHECK_THROW_WITH_INFO(op, info) pgCheckHelper(op, __FILE__, __LINE__, info)

inline bool useMPI()
{
    bool useMPI = true;
    char* val = std::getenv("TLLM_DISABLE_MPI");
    if (val != nullptr && std::string(val) == "1")
    {
        useMPI = false;
    }
    return useMPI;
}

namespace tensorrt_llm::pg_utils
{

// ProcessGroup management functions
c10::intrusive_ptr<c10d::ProcessGroup> get_world_pg();

c10::intrusive_ptr<c10d::ProcessGroup> get_local_pg();

void init_pg(c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_world,
    c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_local);

// Tensor wrapping utilities for ProcessGroup operations
inline torch::Tensor wrap_tensor(torch::Tensor data)
{
    return data;
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
torch::Tensor wrap_tensor(T* data, size_t size)
{
    if constexpr (std::is_same_v<std::decay_t<T>, char>)
    {
        // `char` does not have a guaranteed specialization in CppTypeToScalarType
        // across PyTorch builds. Treat `char` as kChar (int8) explicitly.
        return at::from_blob(data, {static_cast<int64_t>(size)}, c10::TensorOptions{}.dtype(torch::kChar));
    }
    else if constexpr (std::is_same_v<std::decay_t<T>, uint64_t>)
    {
        // `uint64_t` may not have a guaranteed specialization in CppTypeToScalarType
        // across PyTorch builds. Treat `uint64_t` as kLong (int64) explicitly.
        return at::from_blob(data, {static_cast<int64_t>(size)}, c10::TensorOptions{}.dtype(torch::kLong));
    }
    else
    {
        return at::from_blob(data, {static_cast<int64_t>(size)},
            c10::TensorOptions{}.dtype(torch::CppTypeToScalarType<std::decay_t<T>>::value));
    }
}

template <typename T, typename = std::enable_if_t<std::is_void_v<T>>, typename = void>
torch::Tensor wrap_tensor(T* data, size_t size)
{
    return at::from_blob(data, {static_cast<int64_t>(size)}, c10::TensorOptions{}.dtype(torch::kChar));
}

template <typename T>
torch::Tensor wrap_tensor(T const* data, size_t size)
{
    return wrap_tensor(const_cast<T*>(data), size);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
torch::Tensor wrap_tensor(T& data)
{
    return wrap_tensor(&data, 1);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
torch::Tensor wrap_tensor(std::reference_wrapper<T> data)
{
    return wrap_tensor(&data.get(), 1);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
torch::Tensor wrap_tensor(T* data)
{
    return wrap_tensor(data, 1);
}

template <typename T>
torch::Tensor wrap_tensor(std::vector<T>& data)
{
    return wrap_tensor(data.data(), data.size());
}

template <typename T>
torch::Tensor wrap_tensor(std::vector<T> const& data)
{
    return wrap_tensor(data.data(), data.size());
}

template <typename T>
torch::Tensor wrap_tensor(std::reference_wrapper<std::vector<T>> data)
{
    auto& ref = data.get();
    return wrap_tensor(ref.data(), ref.size());
}

template <typename T>
torch::Tensor wrap_tensor(std::reference_wrapper<std::vector<T> const> data)
{
    auto const& ref = data.get();
    return wrap_tensor(ref.data(), ref.size());
}

template <typename T>
torch::Tensor wrap_tensor(std::vector<T>* data)
{
    return wrap_tensor(data->data(), data->size());
}

// ProcessGroup Helper - convenient wrapper around ProcessGroup operations
struct PgHelper
{
    c10::intrusive_ptr<c10d::ProcessGroup> pg;

    PgHelper(c10::intrusive_ptr<c10d::ProcessGroup> pg)
        : pg(pg)
    {
    }

    template <typename Input, typename Output>
    c10::intrusive_ptr<c10d::Work> allgather(
        Input input, Output output, c10d::AllgatherOptions options = c10d::AllgatherOptions())
    {
        auto inputTensor = wrap_tensor(input);
        auto outputTensor = wrap_tensor(output);

        return pg->_allgather_base(outputTensor, inputTensor, options);
    }

    template <typename Input>
    c10::intrusive_ptr<c10d::Work> allreduce(Input input, c10d::AllreduceOptions options = c10d::AllreduceOptions())
    {
        std::vector inputs{wrap_tensor(input)};

        return pg->allreduce(inputs, options);
    }

    template <typename Input>
    c10::intrusive_ptr<c10d::Work> send(Input input, int dstRank, int tag)
    {
        std::vector inputs{wrap_tensor(input)};

        return pg->send(inputs, dstRank, tag);
    }

    template <typename Output>
    c10::intrusive_ptr<c10d::Work> recv(Output output, int srcRank, int tag)
    {
        std::vector outputs{wrap_tensor(output)};

        return pg->recv(outputs, srcRank, tag);
    }

    // Variable-size allgather helper implemented via padding + slicing on Tensors.
    template <typename Input, typename Output, typename SizeT = int64_t>
    bool allgatherv(Input input, Output output, std::vector<SizeT> const& sizes,
        c10d::AllgatherOptions options = c10d::AllgatherOptions())
    {
        auto const worldSize = pg->getSize();

        TLLM_CHECK_WITH_INFO(
            static_cast<int>(sizes.size()) == worldSize, "sizes.size() must equal worldSize in allgatherv");

        at::Tensor inputTensor = wrap_tensor(input);
        SizeT const localSize = static_cast<SizeT>(inputTensor.numel());
        TLLM_CHECK_WITH_INFO(
            sizes[pg->getRank()] == localSize, "sizes[rank] must equal local input size in allgatherv");

        SizeT const maxSize = *std::max_element(sizes.begin(), sizes.end());
        auto tensorOptions = inputTensor.options();

        at::Tensor paddedInput = at::zeros({static_cast<int64_t>(maxSize)}, tensorOptions);
        if (localSize > 0)
        {
            paddedInput.narrow(0, 0, static_cast<int64_t>(localSize)).copy_(inputTensor);
        }

        at::Tensor paddedOutput
            = at::empty({static_cast<int64_t>(maxSize) * static_cast<int64_t>(worldSize)}, tensorOptions);

        PGCHECK_THROW(pg->_allgather_base(paddedOutput, paddedInput, options)->wait());

        // Prepare compact output tensor backed by 'output'
        SizeT const totalSize = std::accumulate(sizes.begin(), sizes.end(), static_cast<SizeT>(0));
        at::Tensor outputTensor = wrap_tensor(output);
        TLLM_CHECK_WITH_INFO(outputTensor.numel() == static_cast<int64_t>(totalSize),
            "output tensor numel must equal total size in allgatherv");

        // Slice and compact
        size_t writeOffset = 0;
        for (int r = 0; r < worldSize; ++r)
        {
            int64_t const validCount = static_cast<int64_t>(sizes[static_cast<size_t>(r)]);
            int64_t const srcOffset = static_cast<int64_t>(r) * static_cast<int64_t>(maxSize);
            if (validCount > 0)
            {
                outputTensor.narrow(0, static_cast<int64_t>(writeOffset), validCount)
                    .copy_(paddedOutput.narrow(0, srcOffset, validCount));
                writeOffset += static_cast<size_t>(validCount);
            }
        }

        return true;
    }

    // Convenience overload to accept sizes passed via std::cref(...)
    template <typename Input, typename Output, typename SizeT = int64_t>
    bool allgatherv(Input input, Output output, std::reference_wrapper<std::vector<SizeT> const> sizes,
        c10d::AllgatherOptions options = c10d::AllgatherOptions())
    {
        return allgatherv<Input, Output, SizeT>(input, output, sizes.get(), options);
    }
};

} // namespace tensorrt_llm::pg_utils
