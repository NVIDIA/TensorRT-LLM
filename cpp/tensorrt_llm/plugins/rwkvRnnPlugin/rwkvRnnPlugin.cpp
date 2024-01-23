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

#include "rwkvRnnPlugin.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/rwkvRnnKernels.h"
#include <NvInferImpl.h>

#include <NvInferRuntimeBase.h>
#include <iostream>
#include <stdexcept>
// #include "tensorrt_llm/kernels/layernormKernels.h"

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

void printStackTrace()
{
    const int max_frames = 64;
    void* callstack[max_frames];
    int num_frames = backtrace(callstack, max_frames);
    char** symbols = backtrace_symbols(callstack, num_frames);

    if (symbols == nullptr)
    {
        fprintf(stderr, "Failed to retrieve backtrace symbols\n");
        return;
    }

    // 打印调用栈信息
    for (int i = 0; i < num_frames; i++)
    {
        printf("%s\n", symbols[i]);
    }

    free(symbols);
}

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::RwkvRnnPluginCreator;
using tensorrt_llm::plugins::RwkvRnnPlugin;

static const char* RWKV_RNN_PLUGIN_VERSION{"5.2"};
static const char* RWKV_RNN_PLUGIN_NAME{"RwkvRnn"};
PluginFieldCollection RwkvRnnPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> RwkvRnnPluginCreator::mPluginAttributes;

RwkvRnnPlugin::RwkvRnnPlugin(nvinfer1::DataType type, int C)
    :mType(type), C(C)
{
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

// Parameterized constructor
RwkvRnnPlugin::RwkvRnnPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mType);
    read(d, C);
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((getSMVersion() >= 80) || (mType != DataType::kBF16), "Unsupported data type");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* RwkvRnnPlugin::clone() const noexcept
{
    auto* plugin = new RwkvRnnPlugin(mType, C);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs RwkvRnnPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    auto B = inputs[1].d[0];
    auto T = inputs[1].d[2];
    auto H = inputs[4].d[0];
    DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = B;
    ret.d[1] = T;
    ret.d[2] = exprBuilder.constant(C);
    std::cout << ">>>>>>>>>>>>>>>>>>>>>>>> " << B->getConstantValue() << "    " <<
        T->getConstantValue() << "     " << H->getConstantValue() << std::endl;
    return ret;
}

bool RwkvRnnPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    TLLM_CHECK(0 <= pos && pos < 7);
    if (inOut[pos].format != TensorFormat::kLINEAR)
    {
        if (inOut[pos].format == TensorFormat::kCHW32 && mType == DataType::kFLOAT)
        {
        }
        else
        {
            // std::cout << "==================================" << std::endl;
            // std::cout << pos << "       " << int(inOut[pos].format) << std::endl;
            // std::cout << "==================================" << std::endl;
            // TLLM_CHECK(false);
            return false;
        }
    }
    if ((pos == 0 || pos == 4) && inOut[pos].type != DataType::kFLOAT)
    {
        // TLLM_CHECK(false);
        return false; // state and w should be always float32
    }
    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<< dtype = " << int(inOut[pos].type) << "   " << int(mType) << "  pos=" << pos
              << "  inputs=" << nbInputs << "  outputs=" << nbOutputs << std::endl;
    // TLLM_CHECK(inOut[pos].type == mType);
    return inOut[pos].type == mType;
}

void RwkvRnnPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t RwkvRnnPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int RwkvRnnPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int B = inputDesc[1].dims.d[0];
    int T = inputDesc[1].dims.d[1];
    int H = inputDesc[4].dims.d[0];

    if (mType == DataType::kHALF)
    {
        float* state = reinterpret_cast<float*>(const_cast<void*>(inputs[0]));
        const half* r = reinterpret_cast<const half*>(inputs[1]);
        const half* k = reinterpret_cast<const half*>(inputs[2]);
        const half* v = reinterpret_cast<const half*>(inputs[3]);
        const float* w = reinterpret_cast<const float*>(inputs[4]);
        const half* u = reinterpret_cast<const half*>(inputs[5]);
        half* output = reinterpret_cast<half*>(outputs[0]);
        invokeGeneralRwkvRnn(state, r, k, v, w, u, output, B, T, C, H, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        float* state = reinterpret_cast<float*>(const_cast<void*>(inputs[0]));
        const float* r = reinterpret_cast<const float*>(inputs[1]);
        const float* k = reinterpret_cast<const float*>(inputs[2]);
        const float* v = reinterpret_cast<const float*>(inputs[3]);
        const float* w = reinterpret_cast<const float*>(inputs[4]);
        const float* u = reinterpret_cast<const float*>(inputs[5]);
        float* output = reinterpret_cast<float*>(outputs[0]);
        invokeGeneralRwkvRnn(state, r, k, v, w, u, output, B, T, C, H, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        float* state = reinterpret_cast<float*>(const_cast<void*>(inputs[0]));
        const __nv_bfloat16* r = reinterpret_cast<const __nv_bfloat16*>(inputs[1]);
        const __nv_bfloat16* k = reinterpret_cast<const __nv_bfloat16*>(inputs[2]);
        const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(inputs[3]);
        const float* w = reinterpret_cast<const float*>(inputs[4]);
        const __nv_bfloat16* u = reinterpret_cast<const __nv_bfloat16*>(inputs[5]);
        __nv_bfloat16* output = reinterpret_cast<__nv_bfloat16*>(outputs[0]);
        invokeGeneralRwkvRnn(state, r, k, v, w, u, output, B, T, C, H, stream);
    }
#endif

    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType RwkvRnnPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return mType; // same with r
}

// IPluginV2 Methods

const char* RwkvRnnPlugin::getPluginType() const noexcept
{
    return RWKV_RNN_PLUGIN_NAME;
}

const char* RwkvRnnPlugin::getPluginVersion() const noexcept
{
    return RWKV_RNN_PLUGIN_VERSION;
}

int RwkvRnnPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int RwkvRnnPlugin::initialize() noexcept
{
    return 0;
}

void RwkvRnnPlugin::terminate() noexcept {}

size_t RwkvRnnPlugin::getSerializationSize() const noexcept
{
    return sizeof(mType) + sizeof(C);
}

void RwkvRnnPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, C);
    assert(d == a + getSerializationSize());
}

void RwkvRnnPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

RwkvRnnPluginCreator::RwkvRnnPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("C", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RwkvRnnPluginCreator::getPluginName() const noexcept
{
    return RWKV_RNN_PLUGIN_NAME;
}

const char* RwkvRnnPluginCreator::getPluginVersion() const noexcept
{
    return RWKV_RNN_PLUGIN_VERSION;
}

const PluginFieldCollection* RwkvRnnPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* RwkvRnnPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    nvinfer1::DataType type;
    int C;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "C"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            C = static_cast<int>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new RwkvRnnPlugin(type, C);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* RwkvRnnPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call RwkvRnnPlugin::destroy()
    try
    {
        auto* obj = new RwkvRnnPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}