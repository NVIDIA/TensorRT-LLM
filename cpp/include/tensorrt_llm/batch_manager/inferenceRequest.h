/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "tensorrt_llm/batch_manager/NamedTensor.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::batch_manager
{

class InferenceRequest
{
public:
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
    using TensorMap = tensorrt_llm::runtime::StringPtrMap<tensorrt_llm::runtime::ITensor>;

    InferenceRequest(uint64_t requestId)
        : mRequestId(requestId)
        , mIsStreaming(false)
    {
    }

    InferenceRequest(TensorMap const& inputTensors, uint64_t requestId)
        : mInputTensors(inputTensors)
        , mRequestId(requestId)
        , mIsStreaming(false)
    {
    }

    InferenceRequest(TensorMap&& inputTensors, uint64_t requestId)
        : mInputTensors(std::move(inputTensors))
        , mRequestId(requestId)
        , mIsStreaming(false)
    {
    }

    ~InferenceRequest() {}

    template <typename T>
    std::tuple<bool, T> getScalarValueFromTensor(
        const std::string& inputTensorName, const std::vector<int64_t>& expectedShape, const bool is_optional) const
    {
        T scalarValue;
        try
        {
            const auto& t = getInputTensor(inputTensorName);
            std::vector<int64_t> tensorShape(t->getShape().nbDims);
            for (int32_t i = 0; i < t->getShape().nbDims; ++i)
            {
                tensorShape[i] = t->getShape().d[i];
            }

            if (tensorShape != expectedShape)
            {
                std::string err = "Invalid shape for " + inputTensorName + ". Expected shape: [";
                for (auto shape : expectedShape)
                {
                    err += std::to_string(shape) + ",";
                }
                if (!expectedShape.empty())
                {
                    // Remove last comma
                    err.pop_back();
                }
                err += "]";

                throw std::runtime_error(err);
            }
            scalarValue = *static_cast<T*>(t->data());
        }
        catch (const std::exception& e)
        {
            // If parameter is optional, just ignore it
            if (is_optional)
            {
                return {false, scalarValue};
            }
            else
            {
                std::cerr << "Out of Range error for tensor: " << inputTensorName << ": " << e.what() << '\n';
                throw;
            }
        }
        return {true, scalarValue};
    }

    const TensorPtr& getInputTensor(std::string const& inputTensorName) const
    {
        return mInputTensors.at(inputTensorName);
    }

    void emplaceInputTensor(std::string const& inputTensorName, TensorPtr&& inputTensor)
    {
        mInputTensors.emplace(inputTensorName, std::move(inputTensor));
    }

    void setIsStreaming(bool isStreaming)
    {
        mIsStreaming = isStreaming;
    }

    bool isStreaming() const
    {
        return mIsStreaming;
    }

    uint64_t getRequestId() const
    {
        return mRequestId;
    }

    const std::vector<int64_t> serialize() const
    {
        std::list<int64_t> packed;
        // mInputTensors
        packed.push_back(static_cast<int64_t>(mInputTensors.size()));
        for (auto it = mInputTensors.begin(); it != mInputTensors.end(); ++it)
        {
            NamedTensor nt(it->second, it->first);
            auto packed_tensor = nt.serialize();
            packed.push_back(static_cast<int64_t>(packed_tensor.size()));
            packed.insert(packed.end(), packed_tensor.begin(), packed_tensor.end());
        }
        // mRequestId
        packed.push_back(static_cast<int64_t>(mRequestId));
        // mIsStreaming
        packed.push_back(mIsStreaming ? 1 : 0);
        // done
        std::vector<int64_t> vpacked{
            std::make_move_iterator(std::begin(packed)), std::make_move_iterator(std::end(packed))};
        return vpacked;
    }

    static std::shared_ptr<InferenceRequest> deserialize(const std::vector<int64_t>& packed)
    {
        return InferenceRequest::deserialize(packed.data());
    }

    static std::shared_ptr<InferenceRequest> deserialize(const int64_t* packed_ptr)
    {
        int64_t num_tensors = *packed_ptr++;
        TensorMap InputTensors;
        for (int64_t i = 0; i < num_tensors; ++i)
        {
            int64_t n = *packed_ptr++;
            auto inputTensor = NamedTensor::deserialize(packed_ptr);
            packed_ptr += n;
            auto inputTensorName = inputTensor.name;
            InputTensors.emplace(inputTensorName, std::move(inputTensor.tensor));
        }
        uint64_t RequestId = static_cast<uint64_t>(*packed_ptr++);
        bool IsStreaming = *packed_ptr++ != 0;
        std::shared_ptr<InferenceRequest> ir = std::make_shared<InferenceRequest>(InputTensors, RequestId);
        ir->setIsStreaming(IsStreaming);
        return ir;
    }

private:
    TensorMap mInputTensors;
    uint64_t mRequestId;
    bool mIsStreaming;
};

} // namespace tensorrt_llm::batch_manager
