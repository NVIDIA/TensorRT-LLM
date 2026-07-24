/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

class CpSplitPlugin : public BasePluginV3
{
public:
    CpSplitPlugin();
    CpSplitPlugin(int cpSize, int cpRank);
    CpSplitPlugin(CpSplitPlugin const& p) = default;
    void initFieldsToSerialize();

    // IPluginV3 methods
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;

    // IPluginV3OneBuild methods
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override; // nochange
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;                                          // fixed
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;                                       // fixed
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;                                                         // fixed
    int32_t getNbOutputs() const noexcept override;                                                   // fixed
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override; // fixed
    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override;
    int32_t getNbTactics() noexcept override;
    char const* getTimingCacheID() noexcept override;
    int32_t getFormatCombinationLimit() noexcept override;
    char const* getMetadataString() noexcept override;

    // IPluginV3OneRuntime methods
    int32_t setTactic(int32_t tactic) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override; // fixed
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int mCpSize;
    int mCpRank;
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;

    enum IdxEntry
    {
        INPUT_IDS,
        REQUEST_TYPES,
        HOST_CONTEXT_LENGTH,
    };

    enum class RequestType : int32_t
    {
        kCONTEXT = 0,
        kGENERATION = 1
    };
};

class CpSplitPluginCreator : public BaseCreatorV3
{
public:
    CpSplitPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
