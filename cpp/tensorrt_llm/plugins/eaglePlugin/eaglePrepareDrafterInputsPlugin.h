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

class EaglePrepareDrafterInputsPlugin : public nvinfer1::IPluginV3,
                                        public nvinfer1::IPluginV3OneCore,
                                        public nvinfer1::IPluginV3OneBuild,
                                        public nvinfer1::IPluginV3OneRuntime
{
public:
    EaglePrepareDrafterInputsPlugin(EaglePrepareDrafterInputsPlugin const& p) = default;

    EaglePrepareDrafterInputsPlugin(int32_t layerIdx, int32_t numLayers, int32_t maxNonLeavesPerLayer);

    nvinfer1::IPluginV3* clone() noexcept override;

    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

    void initFieldsToSerialize();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;

    int32_t getNbOutputs() const noexcept override;

    bool supportsFormatCombination(
        int pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int nbOutputs) const noexcept override;
    int enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    enum class InputIdxEntry : int32_t
    {
        //! [batch_size]
        SEQUENCE_LENGTHS = 0,
        //! [batch_size]
        CONTEXT_LENGTHS,
        //! [num_tokens]
        INPUT_IDS,
        //! [batch_size]
        CHUNKED_CONTEXT_NEXT_TOKENS,
        //! [batch_size, max_path_len]
        ACCEPTED_TOKENS,
        //! [batch_size]
        ACCEPTED_LENS,
        //! [batch_size]
        ACCEPTED_PATHS,
        //! [batch_size, max_decoding_draft_tokens]
        NEXT_DRAFT_TOKENS,
        //! [batch_size]
        NEXT_DRAFT_LENS,
        //! [batch_size, max_decoding_tokens, max_path_len]
        NEXT_DRAFT_PATHS,
        //! [batch_size]
        PREV_DRAFT_LENS,
        //! [batch_size, max_decoding_tokens, max_path_len]
        PREV_DRAFT_PATHS,
        //! [(max_path_len - 1) * batch_size + 1]
        HIDDEN_SIZE_BATCH_LEVEL_STARTS,
        //! [num_gen_tokens]
        INPUT_GEN_TOKENS,
        //! [num_gen_requests]
        SPEC_DECODING_GENERATION_LENGTHS,
    };

    enum class OutputIdxEntry : int32_t
    {
        //! [batch_size]
        SEQUENCE_LENGTHS = 0,
        //! [batch_size]
        CONTEXT_LENGTHS,
        //! [batch_size]
        SPEC_DECODING_GENERATION_LENGTHS,
        //! [batch_size, max_decoding_tokens]
        SPEC_DECODING_POSITION_OFFSETS,
        //! [batchSize, maxDecodingTokens, ceil(maxDecodingTokens / 32)]
        SPEC_DECODING_PACKED_MASK,
        //! [batchSize * mMaxNonLeavesPerLayer * layerIdx] for layerIdx > 0
        //! [num_tokens - numGenTokens + numGenRequests * (mNumLayers + 1)] for layerIdx == 0
        OUTPUT_IDS,
        //! [batchSize] for layerIdx > 0
        //! [num_tokens - numGenTokens + numGenRequests * (mNumLayers + 1)] for layerIdx == 0
        POSITION_IDS,
        //! [batchSize * mMaxNonLeavesPerLayer * layerIdx] for layerIdx > 0
        //! [num_tokens - numGenTokens + numGenRequests * (mNumLayers + 1)] for layerIdx == 0
        HIDDEN_STATES_INDICES,
        //! [batchSize * mMaxNonLeavesPerLayer]
        LAST_TOKEN_INDICES,
        //! [1]
        NUM_LAST_TOKEN_INDICES,
        //! [(max_path_len - 1) * batch_size + 1]
        HIDDEN_SIZE_BATCH_LEVEL_STARTS,
    };

    int32_t getIdx(InputIdxEntry idx) const
    {
        return static_cast<int32_t>(idx);
    }

    int32_t getIdx(OutputIdxEntry idx) const
    {
        return static_cast<int32_t>(idx);
    }

private:
    void prepareCtxEagleNetData(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept;

    void prepareGenEagleNetData(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept;

private:
    int32_t mLayerIdx{0};
    int32_t mNumLayers{0};
    int32_t mMaxNonLeavesPerLayer{0};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
};

class EaglePrepareDrafterInputsPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    EaglePrepareDrafterInputsPluginCreator();

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
