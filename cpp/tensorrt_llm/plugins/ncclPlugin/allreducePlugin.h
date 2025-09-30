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

#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"

#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{
namespace tk = ::tensorrt_llm::kernels;

class AllreducePlugin : public BasePlugin
{
public:
    AllreducePlugin(std::set<int> group, nvinfer1::DataType type, tk::AllReduceStrategyType strategy,
        tk::AllReduceStrategyConfig config, tk::AllReduceFusionOp op, int32_t counter, float eps, int8_t affine,
        int8_t bias, int8_t scale);

    AllreducePlugin(void const* data, size_t length);

    ~AllreducePlugin() override = default;

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
    bool isCustomAllReduceSupported(int ranks_per_node) const noexcept;
    void initGroupTopology() noexcept;
    void setGroupTopology() noexcept;
    tk::AllReduceStrategyType selectImplementation(size_t messageSize, int worldSize, nvinfer1::DataType type) noexcept;
    void check() noexcept;

private:
    std::string const mLayerName;
    std::set<int> mGroup;
    bool mIsNVLINKSupported;
    bool mIsP2PSupported;
    nvinfer1::DataType mType;
    tk::AllReduceStrategyType mStrategy;
    tk::AllReduceStrategyConfig mConfig;
    tk::AllReduceFusionOp mOp;
    float mEps;
    std::shared_ptr<ncclComm_t> mNcclComm;
    int8_t mAffine;
    int8_t mBias;
    int8_t mScale;
};

class AllreducePluginCreator : public BaseCreator
{
public:
    AllreducePluginCreator();

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
