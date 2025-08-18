/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <functional>
#include <memory>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/torch.h>
#include <vector>

#include "tensorrt_llm/common/tllmException.h"

// Must in sync with tensorrt_llm/_torch/distributed/pg_broker.cpp
namespace tensorrt_llm::pg_broker
{
void init_pg(c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_world,
    c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_local);

void init_store(c10::intrusive_ptr<c10d::Store> const& default_store);

} // namespace tensorrt_llm::pg_broker

namespace tensorrt_llm::pg_utils
{

// ProcessGroup management functions
c10::intrusive_ptr<c10d::ProcessGroup> get_world_pg();

c10::intrusive_ptr<c10d::ProcessGroup> get_local_pg();

// Tensor wrapping utilities for ProcessGroup operations
inline torch::Tensor wrap_tensor(torch::Tensor data)
{
    return data;
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
torch::Tensor wrap_tensor(T* data, size_t size)
{
    return at::from_blob(data, {static_cast<int64_t>(size)},
        c10::TensorOptions{}.dtype(torch::CppTypeToScalarType<std::decay_t<T>>::value));
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
torch::Tensor wrap_tensor(std::reference_wrapper<std::vector<T>> data)
{
    auto& ref = data.get();
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
};

} // namespace tensorrt_llm::pg_utils

// Check async op.
inline c10::intrusive_ptr<c10d::Work> pgCheckHelper(
    c10::intrusive_ptr<c10d::Work> work, char const* const file, int const line, char const* info)
{
    work->wait();
    if (work->exception())
    {
        try
        {
            std::rethrow_exception(work->exception());
        }
        catch (...)
        {
            auto msg = std::string("[TensorRT-LLM][ERROR] Torch distributed operation error: ") + info;
            std::throw_with_nested(tensorrt_llm::common::TllmException(file, line, msg.c_str()));
        }
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
