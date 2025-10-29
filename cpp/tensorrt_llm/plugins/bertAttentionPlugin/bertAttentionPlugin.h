/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/fmhaDispatcher.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <cassert>
#include <cuda_runtime.h>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

class BertAttentionPlugin : public BasePlugin
{
public:
    BertAttentionPlugin() = delete;

    BertAttentionPlugin(int num_heads, int head_size, float q_scaling,
        tensorrt_llm::kernels::ContextFMHAType context_fmha_type, nvinfer1::DataType type,
        bool do_relative_attention = false, int max_distance = 0, bool remove_padding = false, bool sage_attn = false,
        int sage_attn_q_block_size = 0, int sage_attn_k_block_size = 0, int sage_attn_v_block_size = 0, int cp_size = 1,
        int cp_rank = 0, std::set<int> cp_group = {});

    BertAttentionPlugin(void const* data, size_t length);

    ~BertAttentionPlugin() override = default;

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

private:
    const std::string mLayerName;

    int mNumHeads;
    int mHeadSize;
    float mQScaling;
    nvinfer1::DataType mType;
    bool mRelativeAttention = false;
    int mMaxDistance = 0;
    bool mRemovePadding = false;

    // unfused mha
    bool mQKHalfAccum = false;

    // fmha runner (disable by default)
    bool mEnableContextFMHA = false;
    bool mFMHAForceFP32Acc = false;

    // sage attention
    bool mSageAttn = false;
    int mSageAttnQBlockSize = 0;
    int mSageAttnKBlockSize = 0;
    int mSageAttnVBlockSize = 0;
    std::set<std::vector<int>> mSageAttnSupportedBlockSizes{{64, 64, 256}, {64, 32, 32}};

    int mSM = tensorrt_llm::common::getSMVersion();

    // comm group for RingAttention
    int mCpSize = 1;
    int mCpRank = 0;
    std::set<int> mCpGroup = {};
#if ENABLE_MULTI_DEVICE
    std::shared_ptr<ncclComm_t> mNcclComm;
#endif // ENABLE_MULTI_DEVICE
    cudaStream_t mNcclStream;

    // The default copy constructor will leave them as nullptr. clone() shall initialize it.
    UniqPtrWNullCopy<tensorrt_llm::kernels::FmhaDispatcher> mFmhaDispatcher;
    UniqPtrWNullCopy<tensorrt_llm::common::CublasMMWrapper> mCublasWrapper;
};

class BertAttentionPluginCreator : public BaseCreator
{
public:
    BertAttentionPluginCreator();

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
