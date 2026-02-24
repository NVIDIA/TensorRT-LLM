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
#include "layernormQuantizationPlugin.h"
#include "pluginUtils.h"
#include "tensorrt_llm/kernels/layernormKernels.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::LayernormQuantizationPluginCreator;
using tensorrt_llm::plugins::LayernormQuantizationPlugin;

static char const* LAYERNORM_QUANTIZATION_PLUGIN_VERSION{"1"};
static char const* LAYERNORM_QUANTIZATION_PLUGIN_NAME{"LayernormQuantization"};
PluginFieldCollection LayernormQuantizationPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> LayernormQuantizationPluginCreator::mPluginAttributes;

LayernormQuantizationPlugin::LayernormQuantizationPlugin(float eps, bool useDiffOfSquares,
    bool dynamicActivationScaling, bool sumPerToken, bool clampValEnabled, tensorrt_llm::common::QuantMode quantMode,
    nvinfer1::DataType type, nvinfer1::DataType outputType)
    : mEps(eps)
    , mUseDiffOfSquares(useDiffOfSquares)
    , mDynActScaling(dynamicActivationScaling)
    , mType(type)
    , mOutputType(outputType)
    , mClampValEnabled(clampValEnabled)
    , mQuantMode(quantMode)
    , mSumPerToken(sumPerToken)
{
    TLLM_CHECK_WITH_INFO(mOutputType == nvinfer1::DataType::kINT8 || mOutputType == nvinfer1::DataType::kFP8,
        "Only int8 or fp8 output type is allowed.");
    // Check if the quant mode is valid.
    TLLM_CHECK_WITH_INFO(mQuantMode.hasPerTokenScaling(), "The quant mode is not valid.");
}

// Parameterized constructor
LayernormQuantizationPlugin::LayernormQuantizationPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mEps);
    read(d, mUseDiffOfSquares);
    read(d, mDynActScaling);
    read(d, mSumPerToken);
    read(d, mClampValEnabled);
    read(d, mQuantMode);
    read(d, mType);
    read(d, mOutputType);
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* LayernormQuantizationPlugin::clone() const noexcept
{
    auto* plugin = new LayernormQuantizationPlugin(
        mEps, mUseDiffOfSquares, mDynActScaling, mSumPerToken, mClampValEnabled, mQuantMode, mType, mOutputType);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs LayernormQuantizationPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        // Quantized output
        return inputs[outputIndex];
    }

    // Dynamic scaling or per-token sum if enabled
    try
    {
        if (outputIndex == 1)
        {
            TLLM_CHECK(mDynActScaling);
        }
        else if (outputIndex == 2)
        {
            TLLM_CHECK(mSumPerToken);
        }
        else
        {
            TLLM_CHECK(false);
        }
        DimsExprs ret;
        ret.nbDims = inputs[0].nbDims;
        for (int di = 0; di < ret.nbDims - 1; ++di)
        {
            ret.d[di] = inputs[0].d[di];
        }
        ret.d[ret.nbDims - 1] = exprBuilder.constant(1);
        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool LayernormQuantizationPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    int const totalPoses
        = 6 + static_cast<int>(mClampValEnabled) + static_cast<int>(mDynActScaling) + static_cast<int>(mSumPerToken);
    TLLM_CHECK(0 <= pos && pos < totalPoses);
    TLLM_CHECK(nbInputs == 4 + static_cast<int>(mClampValEnabled));
    if (pos < nbInputs)
    {
        if (pos < 3)
        {
            // activatation, weight, bias
            return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
        else if (pos == 3)
        {
            // scale
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
        else if (pos == 4 && mClampValEnabled)
        {
            // clamp_max_v
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
    }
    else
    {
        auto const output_pos = pos - nbInputs;
        if (output_pos == 0)
        {
            // Quantized output
            return (inOut[pos].type == mOutputType) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
        else if (output_pos == 1 && mDynActScaling)
        {
            // Dynamic scaling if enabled
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
        else if (output_pos == 2 && static_cast<int>(mClampValEnabled))
        {
            // Clamp value
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
    }

    // We should never reach this point
    TLLM_CHECK_WITH_INFO(false, "The input/output is not supported.");
    return false;
}

void LayernormQuantizationPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t LayernormQuantizationPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return 0;
}

int LayernormQuantizationPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //     input [M(*), N]
    //     weight [N, ]
    //     bias [N, ]
    //     scale_to_int [1]
    //     clamp_max_v  [2], contains min val, and max val (optional)
    // outputs
    //     output           [M(*), N]   Normalized activations, potentially with quantization applied.
    //     dynamic_scaling  [M(*), 1]   (Optional) Per-token scales if quantization is enabled.
    //     token_sums       [M(*), 1]   (Optional) Per-token sums of all the channels (before quantization).

    int64_t m64 = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims - 1; ++i)
    {
        m64 *= inputDesc[0].dims.d[i];
    }
    int const m = TLLM_INT32_CAST(m64);
    int const n = TLLM_INT32_CAST(inputDesc[1].dims.d[0]);

    void const* input = inputs[0];
    void const* weight = inputs[1];
    void const* bias = inputs[2];
    void const* scale = inputs[3];
    void const* clampValPtr = mClampValEnabled ? inputs[4] : nullptr;
    void* output = outputs[0];
    void* dynamic_scale = mDynActScaling ? outputs[1] : nullptr;
    void* sum_per_token = mSumPerToken ? outputs[2] : nullptr;

    if (inputDesc[0].type == DataType::kFLOAT && mOutputType == DataType::kINT8)
    {
        dispatchDataType<float, int8_t>(nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares,
            clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#ifdef ENABLE_FP8
    else if (inputDesc[0].type == DataType::kFLOAT && mOutputType == DataType::kFP8)
    {
        dispatchDataType<float, __nv_fp8_e4m3>(nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares,
            clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#endif // ENABLE_FP8
    else if (inputDesc[0].type == DataType::kHALF && mOutputType == DataType::kINT8)
    {
        dispatchDataType<half, int8_t>(nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares, clampValPtr,
            scale, dynamic_scale, sum_per_token, output);
    }
#ifdef ENABLE_FP8
    else if (inputDesc[0].type == DataType::kHALF && mOutputType == DataType::kFP8)
    {
        dispatchDataType<half, __nv_fp8_e4m3>(nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares,
            clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#endif // ENABLE_FP8
#ifdef ENABLE_BF16
    else if (inputDesc[0].type == DataType::kBF16 && mOutputType == DataType::kINT8)
    {
        dispatchDataType<__nv_bfloat16, int8_t>(nullptr, input, weight, bias, mEps, m, n, stream, mUseDiffOfSquares,
            clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#ifdef ENABLE_FP8
    else if (inputDesc[0].type == DataType::kBF16 && mOutputType == DataType::kFP8)
    {
        dispatchDataType<__nv_bfloat16, __nv_fp8_e4m3>(nullptr, input, weight, bias, mEps, m, n, stream,
            mUseDiffOfSquares, clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#endif // ENABLE_FP8
#endif // ENABLE_BF16
    sync_check_cuda_error(stream);
    return 0;
}

template <typename T, typename QuantT>
void LayernormQuantizationPlugin::dispatchDataType(void* out, void const* input, void const* gamma, void const* beta,
    float const eps, int const tokens, int const hidden_dim, cudaStream_t stream, bool use_diff_of_squares,
    void const* clampValPtr, void const* scale, void* dynamic_scale, void* sum_per_token,
    void* normed_output_quant) noexcept
{
    // inputs
    //     activation     [dim0(*), dim1]
    //     clamp_value    [2], contains min val, and max val (optional)
    // outputs
    //     quant          [dim0(*), dim1]
    //     scale_tokens   [dim0(*), 1]

    invokeGeneralLayerNorm(reinterpret_cast<T*>(out), reinterpret_cast<T const*>(input),
        reinterpret_cast<T const*>(gamma), reinterpret_cast<T const*>(beta), eps, tokens, hidden_dim, mQuantMode,
        stream, use_diff_of_squares, reinterpret_cast<float const*>(clampValPtr), reinterpret_cast<float const*>(scale),
        reinterpret_cast<float*>(dynamic_scale), reinterpret_cast<float*>(sum_per_token),
        reinterpret_cast<QuantT*>(normed_output_quant));
}

// IPluginV2Ext Methods
nvinfer1::DataType LayernormQuantizationPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index <= 2);

    if (index == 0)
    {
        // Output 0 quantized output of layer norm
        return mOutputType;
    }
    else if (index == 1)
    {
        assert(mDynActScaling);
        // Output 1 dynamic act scaling
        return nvinfer1::DataType::kFLOAT;
    }
    else if (index == 2)
    {
        assert(mDynActScaling && mSumPerToken);
        // Output 2 per-token sums
        return nvinfer1::DataType::kFLOAT;
    }

    // We should never reach this point
    TLLM_CHECK_WITH_INFO(false, "The output index is not supported.");
    return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods

char const* LayernormQuantizationPlugin::getPluginType() const noexcept
{
    return LAYERNORM_QUANTIZATION_PLUGIN_NAME;
}

char const* LayernormQuantizationPlugin::getPluginVersion() const noexcept
{
    return LAYERNORM_QUANTIZATION_PLUGIN_VERSION;
}

int LayernormQuantizationPlugin::getNbOutputs() const noexcept
{
    return 1 + static_cast<int>(mDynActScaling) + static_cast<int>(mSumPerToken);
}

int LayernormQuantizationPlugin::initialize() noexcept
{
    return 0;
}

void LayernormQuantizationPlugin::terminate() noexcept {}

size_t LayernormQuantizationPlugin::getSerializationSize() const noexcept
{
    return sizeof(mEps) + sizeof(mUseDiffOfSquares) + sizeof(mDynActScaling) + sizeof(mSumPerToken)
        + sizeof(mClampValEnabled) + sizeof(mQuantMode) + sizeof(mType) + sizeof(mOutputType);
}

void LayernormQuantizationPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mEps);
    write(d, mUseDiffOfSquares);
    write(d, mDynActScaling);
    write(d, mSumPerToken);
    write(d, mClampValEnabled);
    write(d, mQuantMode);
    write(d, mType);
    write(d, mOutputType);
    TLLM_CHECK(d == a + getSerializationSize());
}

void LayernormQuantizationPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

LayernormQuantizationPluginCreator::LayernormQuantizationPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("eps", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("use_diff_of_squares", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("dyn_act_scaling", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("sum_per_token", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("clamp_val_enabled", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("quant_mode", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("out_type_id", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* LayernormQuantizationPluginCreator::getPluginName() const noexcept
{
    return LAYERNORM_QUANTIZATION_PLUGIN_NAME;
}

char const* LayernormQuantizationPluginCreator::getPluginVersion() const noexcept
{
    return LAYERNORM_QUANTIZATION_PLUGIN_VERSION;
}

PluginFieldCollection const* LayernormQuantizationPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LayernormQuantizationPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    tensorrt_llm::common::QuantMode quantMode{};
    float eps{};
    nvinfer1::DataType type{};
    nvinfer1::DataType outputType{};
    bool useDiffOfSquares{};
    bool dynamicActivationScaling{};
    bool sumPerToken{};
    bool clampValEnabled{};

    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "eps"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            eps = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "dyn_act_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dynamicActivationScaling = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "use_diff_of_squares"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            useDiffOfSquares = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "sum_per_token"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            sumPerToken = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "clamp_val_enabled"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            clampValEnabled = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "quant_mode"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            quantMode = QuantMode(*(static_cast<int32_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "out_type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            outputType = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new LayernormQuantizationPlugin(
            eps, useDiffOfSquares, dynamicActivationScaling, sumPerToken, clampValEnabled, quantMode, type, outputType);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* LayernormQuantizationPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call LayernormQuantizationPlugin::destroy()
    try
    {
        auto* obj = new LayernormQuantizationPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
