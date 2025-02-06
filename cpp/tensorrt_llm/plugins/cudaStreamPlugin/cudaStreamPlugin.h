/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "NvInferPlugin.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/runtime/cudaMemPool.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace pluginInternal
{
class SideWorkspace
{
public:
    SideWorkspace(cudaStream_t stream)
        : mWorkspaceSize{0}
        , mWorkspacePtr{nullptr}
        , mStream{stream}
    {
    }

    ~SideWorkspace()
    {
        if (mWorkspacePtr)
        {
            TLLM_CUDA_CHECK(cudaFreeAsync(mWorkspacePtr, mStream));
        }
    }

    void* get(size_t workspaceSize)
    {
        if (mWorkspacePtr && mWorkspaceSize < workspaceSize)
        {
            TLLM_CUDA_CHECK(cudaFreeAsync(mWorkspacePtr, mStream));
            mWorkspacePtr = nullptr;
        }
        if (!mWorkspacePtr)
        {
            mWorkspaceSize = workspaceSize;
            auto pool_ptr
                = tensorrt_llm::runtime::CudaMemPool::getPrimaryPoolForDevice(tensorrt_llm::common::getDevice());
            TLLM_CUDA_CHECK(cudaMallocFromPoolAsync(&mWorkspacePtr, mWorkspaceSize, pool_ptr->getPool(), mStream));
        }
        return mWorkspacePtr;
    }

private:
    size_t mWorkspaceSize;
    void* mWorkspacePtr;
    cudaStream_t mStream;
};

class SideStream : public IPluginResource
{
public:
    SideStream(bool init = false)
        : mStream{}
        , mMainEvent{}
        , mSideEvent{}
        , mWorkspace{}
        , mInit{init}
    {
        // The object passed to acquirePluginResource should use the default value init=false
        if (init)
        {
            TLLM_CUDA_CHECK(cudaStreamCreate(&mStream));
            TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&mMainEvent, cudaEventDisableTiming));
            TLLM_CUDA_CHECK(cudaEventCreateWithFlags(&mSideEvent, cudaEventDisableTiming));
            mWorkspace = std::make_shared<SideWorkspace>(mStream);
        }
    }

    void free()
    {
        if (mInit)
        {
            mWorkspace = nullptr;
            TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
            TLLM_CUDA_CHECK(cudaStreamDestroy(mStream));
            TLLM_CUDA_CHECK(cudaEventDestroy(mMainEvent));
            TLLM_CUDA_CHECK(cudaEventDestroy(mSideEvent));
            mInit = false;
        }
    }

    int32_t release() noexcept override
    {
        try
        {
            free();
        }
        catch (std::exception const& e)
        {
            return -1;
        }
        return 0;
    }

    IPluginResource* clone() noexcept override
    {
        // An object is cloned only when calling acquirePluginResource for the first time for each key
        std::unique_ptr<SideStream> cloned{};
        try
        {
            if (!mInit)
            {
                cloned = std::make_unique<SideStream>(/* init */ true);
            }
            else
            {
                return nullptr;
            }
        }
        catch (std::exception const& e)
        {
            return nullptr;
        }
        return cloned.release();
    }

    ~SideStream() override
    {
        free();
    }

    void* getWorkspacePtr(size_t workspaceSize)
    {
        return mWorkspace->get(workspaceSize);
    }

    cudaStream_t getStream() const
    {
        return mStream;
    }

    void waitMainStreamOnSideStream(cudaStream_t const stream) const
    {
        TLLM_CUDA_CHECK(cudaEventRecord(mMainEvent, stream));
        TLLM_CUDA_CHECK(cudaStreamWaitEvent(mStream, mMainEvent));
    }

    void waitSideStreamOnMainStream(cudaStream_t const stream) const
    {
        TLLM_CUDA_CHECK(cudaEventRecord(mSideEvent, mStream));
        TLLM_CUDA_CHECK(cudaStreamWaitEvent(stream, mSideEvent));
    }

    void stallMainStream(char const* name, cudaStream_t const stream, std::optional<int> delay = std::nullopt) const
    {
        tensorrt_llm::runtime::utils::stallStream(name, stream, delay);
    }

    void stallSideStream(char const* name, std::optional<int> delay = std::nullopt) const
    {
        tensorrt_llm::runtime::utils::stallStream(name, mStream, delay);
    }

    static std::string getResourceKey(int const stream_id)
    {
        return "side_stream_" + std::to_string(stream_id);
    }

private:
    cudaStream_t mStream;
    cudaEvent_t mMainEvent;
    cudaEvent_t mSideEvent;
    std::shared_ptr<SideWorkspace> mWorkspace;
    bool mInit;
};

} // namespace pluginInternal
} // namespace nvinfer1

namespace tensorrt_llm::plugins
{

class CudaStreamPlugin : public BasePlugin
{
public:
    CudaStreamPlugin(int sideStreamId, int nbInputs, nvinfer1::DataType type);

    CudaStreamPlugin(void const* data, size_t length);

    CudaStreamPlugin(CudaStreamPlugin const&);

    void init();

    ~CudaStreamPlugin() override = default;

    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(
        int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 Methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;

private:
    const std::string mLayerName;
    int mSideStreamId;
    int mNbInputs;
    nvinfer1::DataType mType;
    nvinfer1::pluginInternal::SideStream* mSideStreamPtr;
};

class CudaStreamPluginCreator : public BaseCreator
{
public:
    CudaStreamPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
