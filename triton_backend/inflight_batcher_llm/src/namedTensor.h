// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "tensorrt_llm/runtime/iTensor.h"

#include <string>

namespace triton::backend::inflight_batcher_llm::utils
{
template <typename TTensor>
class GenericNamedTensor
{
public:
    using TensorPtr = TTensor;

    TensorPtr tensor;
    std::string name;

    GenericNamedTensor() = default;
    ~GenericNamedTensor() = default;

    GenericNamedTensor(TensorPtr _tensor, std::string _name)
        : tensor{std::move(_tensor)}
        , name{std::move(_name)}
    {
    }

    explicit GenericNamedTensor(std::string _name)
        : tensor{}
        , name{std::move(_name)}
    {
    }

    TensorPtr operator()()
    {
        return tensor;
    }

    TensorPtr const& operator()() const
    {
        return tensor;
    }
};

class NamedTensor : public GenericNamedTensor<tensorrt_llm::runtime::ITensor::SharedPtr>
{
public:
    using Base = GenericNamedTensor<tensorrt_llm::runtime::ITensor::SharedPtr>;
    using TensorPtr = Base::TensorPtr;

    NamedTensor(
        nvinfer1::DataType _type, std::vector<int64_t> const& _shape, std::string _name, void const* _data = nullptr);

    NamedTensor(TensorPtr _tensor, std::string _name)
        : Base(std::move(_tensor), std::move(_name)){};

    explicit NamedTensor(std::string _name)
        : Base(std::move(_name)){};

    [[nodiscard]] std::vector<int64_t> serialize() const;

    void serialize(int64_t* out, const size_t totalSize) const;

    [[nodiscard]] size_t serializedSize() const;

    static NamedTensor deserialize(int64_t const* packed);
};
} // namespace triton::backend::inflight_batcher_llm::utils
