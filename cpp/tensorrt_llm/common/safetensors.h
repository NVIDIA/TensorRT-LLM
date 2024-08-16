/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include <NvInferRuntime.h>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

namespace tensorrt_llm::common::safetensors
{
class INdArray
{
public:
    [[nodiscard]] virtual void const* data() const = 0;
    [[nodiscard]] virtual int ndim() const = 0;
    [[nodiscard]] virtual std::vector<int64_t> const& dims() const = 0;
    [[nodiscard]] virtual nvinfer1::DataType dtype() const = 0;

    [[nodiscard]] nvinfer1::Dims trtDims() const
    {
        nvinfer1::Dims dims;
        dims.nbDims = ndim();
        TLLM_CHECK(dims.nbDims <= nvinfer1::Dims::MAX_DIMS);
        memset(dims.d, 0, sizeof(dims.d));
        for (int i = 0; i < dims.nbDims; ++i)
        {
            dims.d[i] = this->dims()[i];
        }
        return dims;
    }

    virtual ~INdArray() = default;
};

class ISafeTensor
{
public:
    static std::shared_ptr<ISafeTensor> open(char const* filename);
    virtual std::shared_ptr<INdArray> getTensor(char const* name) = 0;
    virtual std::vector<std::string> keys() = 0;
    virtual ~ISafeTensor() = default;
};

} // namespace tensorrt_llm::common::safetensors
