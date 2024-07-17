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

#ifndef TRT_LRU_PLUGIN_H
#define TRT_LRU_PLUGIN_H
#include "tensorrt_llm/kernels/lruKernel.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>

namespace tensorrt_llm::plugins
{
// batch_size = num_ctx_requests or num_gen_requests
// num_ctx_requests = number of context requests (single sequence per request).
// num_gen_requests = number of generation requests (single sequences per request).
// can not support beam search

// inputs
//     0.  x [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     1.  A [dim]
//     2.  state [batch_size, dim] or host [1] containing only pointer for paged_state
//     3.  host_request_types [batch_size] int32. 0: context; 1: generation; 2: none.
//     4.  last_token_ids [batch_size] int32
//     5.  state_slot_mapping [batch_size] int32, optional for paged state
//     6.  y [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     7.  y_bias [dim]
//     8.  gate [batch_size, seq_len, 2 * dim] or [num_tokens, 2 * dim] for remove_input_padding
//     9.  gate_bias [2 * dim]
//    10.  gate_x [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//    11.  gate_a [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//    12.  gate_x_bias [2 * dim]
//    13.  gate_a_bias [2 * dim]
// outputs
//     0. output_tensor [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     1. state [batch_size, dim]

class lruPlugin : public BasePlugin
{
public:
    lruPlugin(int dim, int block_size, nvinfer1::DataType type, bool removePadding, bool pagedState, bool yEnabled,
        bool yBiasEnabled, bool fuseGateEnabled, bool gateBiasEnabled);

    lruPlugin(void const* data, size_t length);

    ~lruPlugin() override = default;

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

    IndexType getXIdx() const
    {
        return 0;
    };

    IndexType getAIdx() const
    {
        return 1;
    };

    IndexType getStateIdx() const
    {
        return 2;
    };

    IndexType getHostRequestTypesIdx() const
    {
        return 3;
    };

    IndexType getLastTokenIdsIdx() const
    {
        return 4;
    };

    IndexType getSlotMappingIdx() const
    {
        if (mPagedState)
            return 5;
        else
            return 4;
    };

    IndexType getYIdx() const
    {
        if (mYEnabled)
            return getSlotMappingIdx() + 1;
        else
            return getSlotMappingIdx();
    };

    IndexType getYBiasIdx() const
    {
        if (mYBiasEnabled)
            return getYIdx() + 1;
        else
            return getYIdx();
    };

    IndexType getGateIdx() const
    {
        if (mFuseGateEnabled)
            return getYBiasIdx() + 1;
        else
            return getYBiasIdx();
    };

    IndexType getGateBiasIdx() const
    {
        if (mFuseGateEnabled && mGateBiasEnabled)
            return getGateIdx() + 1;
        else
            return getGateIdx();
    };

    IndexType getGateXIdx() const
    {
        if (mFuseGateEnabled)
            return getGateBiasIdx();
        else
            return getGateBiasIdx() + 1;
    };

    IndexType getGateAIdx() const
    {
        if (mFuseGateEnabled)
            return getGateXIdx();
        else
            return getGateXIdx() + 1;
    };

    IndexType getGateXBiasIdx() const
    {
        if (!mFuseGateEnabled && mGateBiasEnabled)
            return getGateAIdx() + 1;
        else
            return getGateAIdx();
    };

    IndexType getGateABiasIdx() const
    {
        if (!mFuseGateEnabled && mGateBiasEnabled)
            return getGateXBiasIdx() + 1;
        else
            return getGateXBiasIdx();
    };

    static void setLruParams(tensorrt_llm::kernels::lruParams& params,
        // sizes
        const size_t batch, const size_t dim, const size_t block_size, const size_t maxSeqLen,
        // device pointers
        void* statePtr, void const* x, void const* gate, void const* gate_bias, void const* gate_x,
        void const* gate_x_bias, void const* gate_a, void const* gate_a_bias, void const* y, void const* y_bias,
        void const* A, int const* lastTokenIds, int const* slotMapping, void* out, bool removePadding);

private:
    int mDim;
    int mBlockSize;
    nvinfer1::DataType mType;
    bool mRemovePadding = false;
    bool mPagedState = false;
    bool mYEnabled = false;
    bool mYBiasEnabled = false;
    bool mFuseGateEnabled = false;
    bool mGateBiasEnabled = false;
};

class lruPluginCreator : public BaseCreator
{
public:
    lruPluginCreator();

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

#endif // TRT_LRU_PLUGIN_H
