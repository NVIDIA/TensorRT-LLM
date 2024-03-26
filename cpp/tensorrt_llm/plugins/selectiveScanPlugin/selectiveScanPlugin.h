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

#ifndef TRT_SELECTIVE_SCAN_PLUGIN_H
#define TRT_SELECTIVE_SCAN_PLUGIN_H
#include "tensorrt_llm/kernels/selectiveScan.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>

namespace tensorrt_llm::plugins
{
// batch_size = num_ctx_requests or num_gen_requests
// num_ctx_requests = number of context requests (single sequence per request).
// num_gen_requests = number of generation requests (single sequences per request).
// can not support beam search

// inputs
//     0.  input_tensor [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     1.  state [batch_size, dstate, dim] or host [1] containing only pointer for paged_state
//     2.  delta [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     3.  delta_bias [dim]
//     4.  A [dstate, dim]
//     5.  BC [batch_size, seq_len, dstate * 2] or [num_tokens, dstate * 2] for remove_input_padding
//     6.  D [dim]
//     7.  z [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     8.  host_request_types [batch_size] int32. 0: context; 1: generation; 2: none.
//     9.  last_token_ids [batch_size] int32
//    10.  state_slot_mapping [batch_size] int32, optional for paged state
// outputs
//     0. output_tensor [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     1. state [batch_size, dstate, dim]

class SelectiveScanPlugin : public BasePlugin
{
public:
    SelectiveScanPlugin(int dim, int dstate, int dt_rank, bool isVariableB, bool isVariableC, bool deltaSoftplus,
        nvinfer1::DataType type, bool removePadding, bool pagedState);

    SelectiveScanPlugin(void const* data, size_t length);

    ~SelectiveScanPlugin() override = default;

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
    template <typename T>
    int enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

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

    enum class RequestType : int32_t
    {
        kCONTEXT = 0,
        kGENERATION = 1
    };

private:
    using IndexType = std::int32_t;

    IndexType getInputTensorIdx() const
    {
        return 0;
    };

    IndexType getStateIdx() const
    {
        return 1;
    };

    IndexType getDeltaIdx() const
    {
        return 2;
    };

    IndexType getDeltaBiasIdx() const
    {
        return 3;
    };

    IndexType getAIdx() const
    {
        return 4;
    };

    IndexType getBCIdx() const
    {
        return 5;
    };

    IndexType getDIdx() const
    {
        return 6;
    };

    IndexType getZIdx() const
    {
        return 7;
    };

    IndexType getHostRequestTypesIdx() const
    {
        return 8;
    };

    IndexType getLastTokenIdsIdx() const
    {
        return 9;
    };

    IndexType getSlotMappingIdx() const
    {
        return 10;
    };

    void setSSMParams(tensorrt_llm::kernels::SSMParamsBase& params,
        // sizes
        const size_t batch, const size_t dim, const size_t maxSeqLen, const size_t dstate, const size_t dtRank,
        bool const isVariableB, bool const isVariableC,
        // device pointers
        void* statePtr, void const* x, void const* delta, void const* deltaBias, void const* A, void const* BC,
        void const* D, void const* z, int const* lastTokenIds, int const* slotMapping, void* out, bool deltaSoftplus,
        bool removePadding);

private:
    int mDim;
    int mDState;
    int mDtRank;
    bool mIsVariableB;
    bool mIsVariableC;
    bool mDeltaSoftplus;
    nvinfer1::DataType mType;
    bool mRemovePadding = false;
    bool mPagedState = false;
};

class SelectiveScanPluginCreator : public BaseCreator
{
public:
    SelectiveScanPluginCreator();

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

#endif // TRT_SELECTIVE_SCAN_PLUGIN_H
