/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "weightOnlyGroupwiseQuantMatmulPlugin.h"

#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::WeightOnlyGroupwiseQuantMatmulPluginCreator;
using tensorrt_llm::plugins::WeightOnlyGroupwiseQuantMatmulPlugin;
using tensorrt_llm::plugins::WeightOnlyGroupwiseQuantGemmPluginProfiler;

// Flags for indicating whether the corresponding inputs are applied in mQuantAlgo
// mQuantAlgo = pre_quant_scale * PRE_QUANT_SCALE + zero * ZERO + bias * BIAS
// Here pre_quant_scale, zero and bias are boolean type
static constexpr int BIAS = int(1) << 0;
static constexpr int ZERO = int(1) << 1;
static constexpr int PRE_QUANT_SCALE = int(1) << 2;
static constexpr int FP8_ALPHA = int(1) << 3;
using tensorrt_llm::plugins::read;
using tensorrt_llm::plugins::write;

static char const* WOQ_GROUPWISE_MATMUL_PLUGIN_VERSION{"1"};
static char const* WOQ_GROUPWISE_MATMUL_PLUGIN_NAME{"WeightOnlyGroupwiseQuantMatmul"};
PluginFieldCollection WeightOnlyGroupwiseQuantMatmulPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> WeightOnlyGroupwiseQuantMatmulPluginCreator::mPluginAttributes;

void WeightOnlyGroupwiseQuantGemmPluginProfiler::runTactic(int m, int n, int k,
    WeightOnlyGroupwiseQuantGemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream)
{
    // Quantized weights are packed in FP16 format (INT4*4 -> FP16)
    int const originalN = n * FP16_INT4_RATIO;
    half* actPtr = reinterpret_cast<half*>(workspace);
    cutlass::uint4b_t* weightPtr = reinterpret_cast<cutlass::uint4b_t*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half)));
    half* inputScalesPtr
        = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), n * k * sizeof(float)));
    half* zerosPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(inputScalesPtr), k * originalN * sizeof(half) / mGroupSize));
    half* biasesPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(zerosPtr), k * originalN * sizeof(half) / mGroupSize));
    half* outputPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(biasesPtr), m * sizeof(half)));
    char* workspacePtr
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * originalN * sizeof(half)));

    if ((mQuantAlgo & ZERO) == 0)
    {
        zerosPtr = nullptr;
    }

    if ((mQuantAlgo & BIAS) == 0)
    {
        biasesPtr = nullptr;
    }

    int const wsSize = mRunner->getWorkspaceSize(m, originalN, k);

    mRunner->gemm(actPtr, weightPtr, inputScalesPtr, zerosPtr, biasesPtr, outputPtr, m, originalN, k, mGroupSize,
        tactic, workspacePtr, wsSize, stream);
}

void WeightOnlyGroupwiseQuantGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k)
{
    // Quantized weights are packed in FP16 format (INT4*4 -> FP16)
    int const originalN = n * FP16_INT4_RATIO;
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(half),                      // A
        k * n * sizeof(float),                        // B
        k * originalN * sizeof(half) / mGroupSize,    // scales
        k * originalN * sizeof(half) / mGroupSize,    // zeros
        maxM * sizeof(half),                          // biases
        maxM * originalN * sizeof(half),              // C
        mRunner->getWorkspaceSize(maxM, originalN, k) // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<WeightOnlyGroupwiseQuantGemmPluginProfiler::Config> WeightOnlyGroupwiseQuantGemmPluginProfiler::getTactics(
    int m, int n, int k) const
{
    return mRunner->getConfigs();
}

WeightOnlyGroupwiseQuantMatmulPlugin::WeightOnlyGroupwiseQuantMatmulPlugin(nvinfer1::DataType type, int quant_algo,
    int group_size, WeightOnlyGroupwiseQuantMatmulPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    init(type, quant_algo, group_size);
}

// Parameterized constructor
WeightOnlyGroupwiseQuantMatmulPlugin::WeightOnlyGroupwiseQuantMatmulPlugin(
    void const* data, size_t length, WeightOnlyGroupwiseQuantMatmulPlugin::PluginProfilerPtr const& pluginProfiler)
    : mPluginProfiler(pluginProfiler)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    nvinfer1::DataType type;
    int quant_algo = 0;
    int group_size = 0;
    read(d, type);
    read(d, quant_algo);
    read(d, group_size);
    read(d, mDims);

    init(type, quant_algo, group_size);

    mPluginProfiler->deserialize(d, mDims, mGemmId);

    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
}

void WeightOnlyGroupwiseQuantMatmulPlugin::init(nvinfer1::DataType type, int quant_algo, int group_size)
{
    mArch = tensorrt_llm::common::getSMVersion();
    mType = type;
    mQuantAlgo = quant_algo;
    mGroupSize = group_size;

    // quant_algo = fp8_alpha * 8 + pre_quant_scale * 4 + zero * 2 + bias
    mPreQuantScaleInputIdx = (quant_algo & PRE_QUANT_SCALE) ? 1 : 0;
    mWeightInputIdx = mPreQuantScaleInputIdx + 1;
    mScalesInputIdx = mWeightInputIdx + 1;
    mZerosInputIdx = (quant_algo & ZERO) ? mScalesInputIdx + 1 : mScalesInputIdx;
    mBiasesInputIdx = (quant_algo & BIAS) ? mZerosInputIdx + 1 : mZerosInputIdx;
    mAlphaInputIdx = (quant_algo & FP8_ALPHA) ? mBiasesInputIdx + 1 : mBiasesInputIdx;

    if (mType == nvinfer1::DataType::kHALF)
    {
        if (quant_algo & FP8_ALPHA)
        {
            // Ada & Hopper style kernels
            if (mArch < 89)
            {
                TLLM_THROW("W4A(fp)8 kernel is unsupported on pre-Ada (sm<89) architectures!");
            }
            if (quant_algo & ZERO)
            {
                // has zeros
                m_weightOnlyGroupwiseGemmRunner = std::make_shared<
                    tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_fp8_e4m3, cutlass::uint4b_t,
                        cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, half, half, half>>();
            }
            else
            {
                // no zeros
                m_weightOnlyGroupwiseGemmRunner
                    = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_fp8_e4m3,
                        cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, half, half, half>>();
            }
        }
        else
        {
            if (quant_algo & ZERO)
            {
                // has zeros
                m_weightOnlyGroupwiseGemmRunner
                    = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<half,
                        cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
            }
            else
            {
                // no zeros
                m_weightOnlyGroupwiseGemmRunner
                    = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<half,
                        cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
            }
        }
        mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
            mArch, tensorrt_llm::kernels::weight_only::KernelType::FP16Int4Groupwise);
        mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::FP16Int4Groupwise;
    }
#if defined(ENABLE_BF16)
    else if (mType == nvinfer1::DataType::kBF16)
    {
        if (quant_algo & FP8_ALPHA)
        {
            // FP8 requires at least sm89 devices
            if (mArch < 89)
            {
                TLLM_THROW("W4A(fp)8 kernel is unsupported on pre-Ada (sm<89) architectures!");
            }
            TLLM_THROW("FP8 is unsupported on with BF16 scales and zero-points!");
        }
        else
        {
            if (quant_algo & ZERO)
            {
                // has zeros
                m_weightOnlyGroupwiseGemmRunner
                    = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_bfloat16,
                        cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
            }
            else
            {
                // no zeros
                m_weightOnlyGroupwiseGemmRunner
                    = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_bfloat16,
                        cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
            }
        }
        mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
            mArch, tensorrt_llm::kernels::weight_only::KernelType::BF16Int4Groupwise);
        mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::BF16Int4Groupwise;
    }
#endif
    else
    {
        TLLM_THROW("Unsupported data type");
    }
    mPluginProfiler->setQuantAlgo(mQuantAlgo);
    mPluginProfiler->setGroupSize(mGroupSize);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* WeightOnlyGroupwiseQuantMatmulPlugin::clone() const noexcept
{
    auto* plugin = new WeightOnlyGroupwiseQuantMatmulPlugin(*this);
    return plugin;
}

void WeightOnlyGroupwiseQuantMatmulPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_weightOnlyGroupwiseGemmRunner, mType, mDims, mGemmId);
}

nvinfer1::DimsExprs WeightOnlyGroupwiseQuantMatmulPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{

    // inputs
    //   0 activations      [M, K]
    //   1 pre-quant scales [K] (optional)
    //   2 weights          [K, N/2]
    //   3 scales           [K // group_size, N]
    //   4 zeros            [K // group_size, N] (optional)
    //   5 biases           [M] (optional)
    //   6 alpha            [1] (optional)

    try
    {
        TLLM_CHECK(nbInputs == mAlphaInputIdx + 1);
        TLLM_CHECK(outputIndex == 0);
        int const nbDimsA = inputs[0].nbDims;
        int const nbDimsB = inputs[mWeightInputIdx].nbDims;
        TLLM_CHECK(nbDimsA >= 2);
        TLLM_CHECK(nbDimsB == 2);
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }

        // int4 weight only quant
        ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[mWeightInputIdx].d[1]->getConstantValue() * FP16_INT4_RATIO);

        return ret;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool WeightOnlyGroupwiseQuantMatmulPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos < mAlphaInputIdx + 2)
    {
        if (pos == mWeightInputIdx)
        {
            // weights
            return inOut[mWeightInputIdx].type == mType && inOut[mWeightInputIdx].format == TensorFormat::kLINEAR;
        }
        else if ((mQuantAlgo & FP8_ALPHA) && pos == mAlphaInputIdx)
        {
            return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
        }
        else
        {
            return inOut[pos].type == mType && inOut[pos].format == TensorFormat::kLINEAR;
        }
    }
    else
    {
        // Never should be here
        assert(false);
        return false;
    }
}

void WeightOnlyGroupwiseQuantMatmulPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1];

    // Quantized weights are packed in FP16 format (INT4*4 -> FP16)
    int const maxN = in[mWeightInputIdx].max.d[1] * FP16_INT4_RATIO;

    auto const K = maxK;
    auto const N = maxN / FP16_INT4_RATIO;

    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, N, K};
    }
    mGemmId = {N, K, mType};

    size_t smoothedActSize = static_cast<size_t>(maxM) * static_cast<size_t>(maxK)
        * (in[0].desc.type == nvinfer1::DataType::kFLOAT ? sizeof(float) : sizeof(half));
    m_workspaceMaxSize = smoothedActSize + m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t WeightOnlyGroupwiseQuantMatmulPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return m_workspaceMaxSize;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // inputs
    //   0 activations      [M, K]
    //   1 pre-quant scales [K]
    //   2 weights          [K, N/2]
    //   3 scales           [K // group_size, N]
    //   4 zeros            [K // group_size, N]
    //   5 biases           [M]
    //   6 alpha            [1]
    // outputs
    //   mat                [M, N]

    int64_t m64 = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        m64 *= inputDesc[0].dims.d[ii];
    }
    int const m = TLLM_INT32_CAST(m64);
    int const n = TLLM_INT32_CAST(inputDesc[mWeightInputIdx].dims.d[1]);
    int const k = TLLM_INT32_CAST(inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1]);

    bool use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabled;
    bool use_pre_quant_scale = mQuantAlgo & PRE_QUANT_SCALE;

    half const* zeros_ptr = (mQuantAlgo & ZERO) ? reinterpret_cast<half const*>(inputs[mZerosInputIdx]) : nullptr;
    half const* biases_ptr = (mQuantAlgo & BIAS) ? reinterpret_cast<half const*>(inputs[mBiasesInputIdx]) : nullptr;
    half const* act_ptr = reinterpret_cast<half const*>(inputs[0]);
    float alpha = 1.0;
    if (mQuantAlgo & FP8_ALPHA)
    {
        cudaMemcpy(&alpha, const_cast<void*>(inputs[mAlphaInputIdx]), sizeof(float), cudaMemcpyDeviceToHost);
    }

    if (use_pre_quant_scale && !use_cuda_kernel)
    {
        // Apply pre-quant per channel scale on activations
        act_ptr = reinterpret_cast<half const*>(workspace);
        if (mType == nvinfer1::DataType::kHALF)
        {
            if (mQuantAlgo & FP8_ALPHA)
            {
                tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<half, __nv_fp8_e4m3>(
                    reinterpret_cast<__nv_fp8_e4m3*>(workspace), reinterpret_cast<half const*>(inputs[0]),
                    reinterpret_cast<half const*>(inputs[mPreQuantScaleInputIdx]), m, k, stream);
            }
            else
            {
                tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<half, half>(
                    reinterpret_cast<half*>(workspace), reinterpret_cast<half const*>(inputs[0]),
                    reinterpret_cast<half const*>(inputs[mPreQuantScaleInputIdx]), m, k, stream);
            }
        }
#if defined(ENABLE_BF16)
        else if (mType == nvinfer1::DataType::kBF16)
        {
            if (mQuantAlgo & FP8_ALPHA)
            {
                tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<__nv_bfloat16, __nv_fp8_e4m3>(
                    reinterpret_cast<__nv_fp8_e4m3*>(workspace), reinterpret_cast<__nv_bfloat16 const*>(inputs[0]),
                    reinterpret_cast<__nv_bfloat16 const*>(inputs[mPreQuantScaleInputIdx]), m, k, stream);
            }
            else
            {
                tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<__nv_bfloat16, __nv_bfloat16>(
                    reinterpret_cast<__nv_bfloat16*>(workspace), reinterpret_cast<__nv_bfloat16 const*>(inputs[0]),
                    reinterpret_cast<__nv_bfloat16 const*>(inputs[mPreQuantScaleInputIdx]), m, k, stream);
            }
        }
#endif
    }

#if defined(ENABLE_BF16)
    TLLM_CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16,
        "No valid weightOnlyGropwiseQuantMatmul configuration");
#else
    TLLM_CHECK_WITH_INFO(mType == nvinfer1::DataType::kHALF, "No valid weightOnlyGropwiseQuantMatmul configuration");
#endif

    // Quantized weights are packed in FP16 format (INT4*4 -> FP16)
    int real_n = n * FP16_INT4_RATIO;
    if (use_cuda_kernel)
    {
        void const* pre_quant_scale_ptr = nullptr;
        if (use_pre_quant_scale)
            pre_quant_scale_ptr = inputs[mPreQuantScaleInputIdx];
        void const* cuda_kernel_act_ptr = inputs[0];
        void const* cuda_kernel_act_scale_ptr = pre_quant_scale_ptr;
        void const* cuda_kernel_weight_ptr = inputs[mWeightInputIdx];
        void const* cuda_kernel_scales_ptr = inputs[mScalesInputIdx];
        void const* cuda_kernel_zeros_ptr = zeros_ptr;
        void const* cuda_kernel_bias_ptr = biases_ptr;
        void* cuda_kernel_out_ptr = outputs[0];
        tensorrt_llm::kernels::weight_only::Params params{cuda_kernel_act_ptr, cuda_kernel_act_scale_ptr,
            cuda_kernel_weight_ptr, cuda_kernel_scales_ptr, cuda_kernel_zeros_ptr, cuda_kernel_bias_ptr,
            cuda_kernel_out_ptr, alpha, m, real_n, k, mGroupSize, mCudaKernelType,
            static_cast<bool>(mQuantAlgo & FP8_ALPHA)};
        tensorrt_llm::kernels::weight_only::kernel_launcher(mArch, params, stream);
    }
    else
    {
        int const ws_bytes = m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(m, real_n, k);

        int32_t* weight_ptr = const_cast<int32_t*>(reinterpret_cast<int32_t const*>(inputs[mWeightInputIdx]));

        auto const& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
        TLLM_CHECK_WITH_INFO(bestTactic,
            "No valid weight only groupwise GEMM tactic(It is usually caused by the failure to execute all "
            "candidate "
            "configurations of the CUTLASS kernel, please pay attention to the warning information when building "
            "the "
            "engine.)");
        m_weightOnlyGroupwiseGemmRunner->gemm(act_ptr, weight_ptr, inputs[mScalesInputIdx], zeros_ptr, biases_ptr,
            alpha, outputs[0], m, real_n, k, mGroupSize, *bestTactic,
            reinterpret_cast<char*>(workspace) + m * k * sizeof(half), ws_bytes, stream);
    }
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType WeightOnlyGroupwiseQuantMatmulPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return mType;
}

// IPluginV2 Methods

char const* WeightOnlyGroupwiseQuantMatmulPlugin::getPluginType() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_NAME;
}

char const* WeightOnlyGroupwiseQuantMatmulPlugin::getPluginVersion() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_VERSION;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void WeightOnlyGroupwiseQuantMatmulPlugin::terminate() noexcept {}

size_t WeightOnlyGroupwiseQuantMatmulPlugin::getSerializationSize() const noexcept
{
    return sizeof(nvinfer1::DataType) +                 // mType
        sizeof(int) +                                   // mQuantAlgo
        sizeof(int) +                                   // mGroupSize
        sizeof(mDims) +                                 // Dimensions
        mPluginProfiler->getSerializationSize(mGemmId); // selected tactics container size
}

void WeightOnlyGroupwiseQuantMatmulPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mType);
    write(d, mQuantAlgo);
    write(d, mGroupSize);
    write(d, mDims);

    mPluginProfiler->serialize(d, mGemmId);
    assert(d == a + getSerializationSize());
}

void WeightOnlyGroupwiseQuantMatmulPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

///////////////

WeightOnlyGroupwiseQuantMatmulPluginCreator::WeightOnlyGroupwiseQuantMatmulPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("quant_algo", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_size", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* WeightOnlyGroupwiseQuantMatmulPluginCreator::getPluginName() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_NAME;
}

char const* WeightOnlyGroupwiseQuantMatmulPluginCreator::getPluginVersion() const noexcept
{
    return WOQ_GROUPWISE_MATMUL_PLUGIN_VERSION;
}

PluginFieldCollection const* WeightOnlyGroupwiseQuantMatmulPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* WeightOnlyGroupwiseQuantMatmulPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    nvinfer1::DataType type;
    int QuantAlgo;
    int GroupSize;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "quant_algo"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            QuantAlgo = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "group_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            GroupSize = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
    }
    try
    {
        // WeightOnlyGroupwiseQuantMatmulPluginCreator is unique and shared for an engine generation
        // Create plugin profiler with shared tactics map
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ false);
        auto* obj = new WeightOnlyGroupwiseQuantMatmulPlugin(type, QuantAlgo, GroupSize, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* WeightOnlyGroupwiseQuantMatmulPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call weightOnlyGroupwiseQuantMatmulPlugin::destroy()
    try
    {
        // Create plugin profiler with private tactics map which is read from the serialized engine
        auto pluginProfiler = gemmPluginProfileManager.createGemmPluginProfiler(/* inference */ true);
        auto* obj = new WeightOnlyGroupwiseQuantMatmulPlugin(serialData, serialLength, pluginProfiler);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
