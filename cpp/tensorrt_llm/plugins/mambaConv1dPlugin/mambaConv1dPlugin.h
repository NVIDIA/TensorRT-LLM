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

#ifndef TRT_MAMBA_CONV1D_PLUGIN_H
#define TRT_MAMBA_CONV1D_PLUGIN_H
#include "tensorrt_llm/kernels/mambaConv1dKernels.h"
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
//     1.  conv_state [batch_size, dconv - 1, dim] or host [1] containing only pointer for paged_state
//     2.  weight [1, dconv, dim]
//     3.  bias [dim]
//     4.  host_request_types [batch_size] int32. 0: context; 1: generation; 2: none.
//     5.  last_token_ids [batch_size] int32
//     6.  host_context_lengths [batch_size] int32, optional for remove_input_padding
//     7.  state_slot_mapping [batch_size] int32, optional
// outputs
//     0. output_tensor [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     1. conv_state [batch_size, dconv - 1, dim]

class MambaConv1dPlugin : public BasePlugin
{
public:
    MambaConv1dPlugin(int dim, int dconv, int preStride, int postStride, nvinfer1::DataType type, bool removePadding,
        bool pagedState, bool applySilu);

    MambaConv1dPlugin(void const* data, size_t length);

    ~MambaConv1dPlugin() override = default;

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

    IndexType getConvStateIdx() const
    {
        return 1;
    };

    IndexType getWeightIdx() const
    {
        return 2;
    };

    IndexType getBiasIdx() const
    {
        return 3;
    };

    IndexType getHostRequestTypesIdx() const
    {
        return 4;
    };

    IndexType getLastTokenIdsIdx() const
    {
        return 5;
    };

    IndexType getHostContextLengthIdx() const
    {
        return 6;
    };

    IndexType getSlotMappingIdx() const
    {
        // if not remove input padding, host_context_length is not used, so the index is 6
        return mRemovePadding ? 7 : 6;
    };

    void setMambaConv1dParams(tensorrt_llm::kernels::MambaConv1dParamsBase& params,
        // sizes
        const size_t batch, const size_t dim, const size_t maxSeqLen, const size_t dconv, const size_t preStride,
        const size_t postStride,
        // device pointers
        void const* inPtr, void const* stateInPtr, void* stateOutPtr, void const* convWeight, void const* convBias,
        void* outPtr, int const* lastTokenIds, int const* stateSlotMapping, bool removePadding, bool applySilu);

private:
    int mDim;
    int mDConv;
    int mPreStride;
    int mPostStride;
    nvinfer1::DataType mType;
    bool mRemovePadding = false;
    bool mPagedState = false;
    bool mApplySilu = true;
};

class MambaConv1dPluginCreator : public BaseCreator
{
public:
    MambaConv1dPluginCreator();

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

#endif // TRT_MAMBA_CONV1D_PLUGIN_H
