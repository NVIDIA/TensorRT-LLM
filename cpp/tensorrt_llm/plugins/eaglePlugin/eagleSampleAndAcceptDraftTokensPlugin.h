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
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

class EagleSampleAndAcceptDraftTokensPlugin : public BasePlugin
{
public:
    EagleSampleAndAcceptDraftTokensPlugin(nvinfer1::DataType type);

    EagleSampleAndAcceptDraftTokensPlugin(void const* data, size_t length);

    ~EagleSampleAndAcceptDraftTokensPlugin() override = default;

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
    enum class InputIdxEntry : int32_t
    {
        //! [num_tokens, vocab_size_padded]
        LOGITS = 0,
        //! [batch_size, max_decoding_draft_tokens]
        DRAFT_TOKEN_IDS,
        //! [batch_size]
        DRAFT_LENS,
        //! [batch_size]
        TEMPERATURE,
        //! [batch_size, max_decoding_tokens]
        RAND_VALIDATION,
        //! [batch_size]
        POSTERIOR_ALPHA,
        //! [batch_size]
        POSTERIOR_THRESHOLD,
        //! [batch_size, max_decoding_tokens, max_path_len]
        PATHS,
        //! [1]
        GREEDY_SAMPLING,
        //! [1]
        USE_DYNAMIC_TREE
    };

    enum class OutputIdxEntry : int32_t
    {
        //! [batch_size, max_path_len]
        ACCEPTED_TOKENS = 0,
        //! [batch_size]
        ACCEPTED_LENS,
        //! [batch_size]
        BEST_ACCEPTED_PATHS,
        //! [batch_size, max_decoding_draft_tokens]
        NEXT_DRAFT_TOKEN_IDS,
        //! [batch_size]
        NEXT_DRAFT_LENS,
        //! [batch_size,  max_decoding_tokens, max_path_len]
        NEXT_DRAFT_PATHS,
        //! [max_draft_path_len * batch_size]
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
    template <typename T>
    size_t getWorkspaceSizeType(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept;

    template <typename T>
    void samplePrimeHeadTokens(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept;

    template <typename T>
    void doTypicalAcceptance(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    template <typename T>
    void acceptDraftTokens(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    template <typename T>
    void enqueueType(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

private:
    nvinfer1::DataType mDtype;
    int32_t mSmCnt{0};
};

class EagleSampleAndAcceptDraftTokensPluginCreator : public BaseCreator
{
public:
    EagleSampleAndAcceptDraftTokensPluginCreator();

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
