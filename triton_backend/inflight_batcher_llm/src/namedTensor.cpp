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

#include "namedTensor.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include <cstring>
#include <vector>

namespace triton::backend::inflight_batcher_llm::utils
{

NamedTensor::NamedTensor(
    nvinfer1::DataType _type, std::vector<int64_t> const& _shape, std::string _name, void const* _data)
    : Base(std::move(_name))
{
    nvinfer1::Dims dims;
    dims.nbDims = _shape.size();
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        dims.d[i] = _shape[i];
    }
    tensor = tensorrt_llm::runtime::BufferManager::pinnedPool(dims, _type);
    if (_data)
    {
        std::memcpy(tensor->data(), _data, tensor->getSizeInBytes());
    }
}

size_t NamedTensor::serializedSize() const
{
    size_t totalSize = 1;

    int n = (name.size() + sizeof(int64_t)) / sizeof(int64_t);
    totalSize += n;

    // memType
    // dataType
    // nbDims
    totalSize += 3;
    totalSize += tensor->getShape().nbDims;

    int m = tensor->getSizeInBytes();
    int mm = (m + sizeof(int64_t) - 1) / sizeof(int64_t);
    totalSize += mm;
    return totalSize;
}

void NamedTensor::serialize(int64_t* vpacked, const size_t totalSize) const
{
    int n = (name.size() + sizeof(int64_t)) / sizeof(int64_t);

    int m = tensor->getSizeInBytes();

    vpacked[0] = name.size();
    std::memcpy(&(vpacked[1]), name.c_str(), name.size());

    int64_t* tensorPtr = &(vpacked[n + 1]);
    *tensorPtr++ = static_cast<int64_t>(tensor->getMemoryType());
    *tensorPtr++ = static_cast<int64_t>(tensor->getDataType());
    *tensorPtr++ = static_cast<int64_t>(tensor->getShape().nbDims);
    for (size_t i = 0; i < static_cast<size_t>(tensor->getShape().nbDims); ++i)
    {
        *tensorPtr++ = static_cast<int64_t>(tensor->getShape().d[i]);
    }
    std::memcpy(tensorPtr, tensor->data(), m);

    tensorPtr += (m + sizeof(int64_t) - 1) / sizeof(int64_t);

    TLLM_CHECK_WITH_INFO(tensorPtr - vpacked == (int64_t) totalSize, "serialize and serializedSize are out of sync");
}

std::vector<int64_t> NamedTensor::serialize() const
{
    size_t totalSize = serializedSize();

    std::vector<int64_t> vpacked(totalSize);
    serialize(vpacked.data(), totalSize);

    return vpacked;
}

NamedTensor NamedTensor::deserialize(int64_t const* packed)
{
    int n = *packed++;
    char const* cname = reinterpret_cast<char const*>(packed);
    int nn = (n + sizeof(int64_t)) / sizeof(int64_t);
    packed += nn;
    ++packed; // tensorrt_llm::runtime::MemoryType
    nvinfer1::DataType trtDType = static_cast<nvinfer1::DataType>(*packed++);
    int64_t nshape = *packed++;
    std::vector<int64_t> shape(nshape);
    memcpy(shape.data(), packed, nshape * sizeof(int64_t));
    packed += nshape;
    return NamedTensor{trtDType, shape, cname, packed};
}

} // namespace triton::backend::inflight_batcher_llm::utils
