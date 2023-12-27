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
#include "bertAttentionPlugin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/iBuffer.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

using tensorrt_llm::plugins::BertAttentionPluginCreator;
using tensorrt_llm::plugins::BertAttentionPlugin;

static const char* BERT_ATTENTION_PLUGIN_VERSION{"1"};
static const char* BERT_ATTENTION_PLUGIN_NAME{"BertAttention"};
PluginFieldCollection BertAttentionPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> BertAttentionPluginCreator::mPluginAttributes;

BertAttentionPlugin::BertAttentionPlugin(int num_heads, int head_size, float q_scaling, bool qk_half_accum,
    ContextFMHAType context_fmha_type, nvinfer1::DataType type, bool do_relative_attention, int max_distance,
    bool remove_padding)
    : mNumHeads(num_heads)
    , mHeadSize(head_size)
    , mQScaling(q_scaling)
    , mQKHalfAccum(qk_half_accum)
    , mEnableContextFMHA(context_fmha_type != ContextFMHAType::DISABLED)
    , mFMHAForceFP32Acc(context_fmha_type == ContextFMHAType::ENABLED_WITH_FP32_ACC)
    , mType(type)
    , mRelativeAttention(do_relative_attention)
    , mMaxDistance(max_distance)
    , mRemovePadding(remove_padding)
{
    // pre-check whether FMHA is supported in order to save memory allocation
    mEnableContextFMHA = mEnableContextFMHA && (mType == DataType::kHALF) && MHARunner::fmha_supported(mHeadSize, mSM)
        && !mRelativeAttention;
}

// Parameterized constructor
BertAttentionPlugin::BertAttentionPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    read(d, mNumHeads);
    read(d, mHeadSize);
    read(d, mQScaling);
    read(d, mQKHalfAccum);
    read(d, mEnableContextFMHA);
    read(d, mFMHAForceFP32Acc);
    read(d, mType);
    read(d, mRelativeAttention);
    read(d, mMaxDistance);
    read(d, mRemovePadding);
    TLLM_CHECK(d == a + length);
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* BertAttentionPlugin::clone() const noexcept
{
    auto* plugin = new BertAttentionPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->initialize();
    return plugin;
}

nvinfer1::DimsExprs BertAttentionPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex == 0);
    auto ret = inputs[0];
    ret.d[mRemovePadding ? 1 : 2] = exprBuilder.constant(ret.d[mRemovePadding ? 1 : 2]->getConstantValue() / 3);
    return ret;
}

bool BertAttentionPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // inputs: [0] qkv, [1] input_lengths, [2] max_input_length (optional), [3] relative_attention_bias (optional)
    // outputs: [X] hidden_states
    if (nbInputs == 2)
    { // BERT
        if (pos == 1)
        {
            return inOut[pos].type == nvinfer1::DataType::kINT32;
        }
        else
        {
            return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
    }
    else if (nbInputs > 2)
    { // Encoder in encoder-decoder
        if (pos == 1 || pos == 2)
        {
            return inOut[pos].type == nvinfer1::DataType::kINT32;
        }
        else
        {
            return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
        }
    }
    else
    {
        return false;
    }
}

void BertAttentionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t BertAttentionPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    // if remove padding, inputs[0] "qkv_hidden_states" dim is [num_tokens, 3*hidden_dim] which doesn't have shape
    // info should get max_batch_size and max_input_length from inputs[1] "input_lengths" and input[2]
    // "max_input_length"
    const int batch_size = mRemovePadding ? inputs[1].dims.d[0] : inputs[0].dims.d[0];
    const int input_seq_len = mRemovePadding ? inputs[2].dims.d[0] : inputs[0].dims.d[1];
    const int local_hidden_units_ = inputs[0].dims.d[mRemovePadding ? 1 : 2] / 3;

    auto const size = tensorrt_llm::runtime::BufferDataType(inputs[0].type).getSize();

    const size_t attention_mask_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * input_seq_len;
    const size_t cu_seqlens_size = sizeof(int) * (batch_size + 1);
    const size_t q_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    const size_t k_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    const size_t v_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_size = mEnableContextFMHA ? 0 : size * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t qkv_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t padding_offset_size = sizeof(int) * batch_size * input_seq_len;

    const int NUM_BUFFERS = 10;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = CUBLAS_WORKSPACE_SIZE;
    workspaces[1] = attention_mask_size;
    workspaces[2] = cu_seqlens_size;
    workspaces[3] = q_buf_2_size;
    workspaces[4] = k_buf_2_size;
    workspaces[5] = v_buf_2_size;
    workspaces[6] = qk_buf_size;
    workspaces[7] = qkv_buf_2_size;
    workspaces[8] = qk_buf_float_size;
    workspaces[9] = padding_offset_size;

    return tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
}

template <typename T>
int BertAttentionPlugin::enqueueImpl(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{

    // inputs
    //     input_tensor [batch_size, seq_len, local_hidden_size*3] or [num_tokens, local_hidden_size*3]
    //     input_lengths [batch_size]
    //     max_input_length [max_input_length] -- use shape dim to represent max value. If remove padding, this records
    //     the max input length among sequences; otherwise same as input_tensor's padded dim[1] relative_attention_bias
    //     [num_heads, num_buckets] (optional)
    // outputs
    //     output_tensor [batch_size, seq_len, local_hidden_size] or [num_tokens, local_hidden_size]

    // if remove padding, inputs[0] dim is [num_tokens] which doesn't have workspace info
    // should get max_batch_size from inputs[1] and max_input_length from plugin attribute
    const int batch_size = mRemovePadding ? inputDesc[1].dims.d[0] : inputDesc[0].dims.d[0];
    const int input_seq_len = mRemovePadding ? inputDesc[2].dims.d[0] : inputDesc[0].dims.d[1];
    const int num_tokens = mRemovePadding ? inputDesc[0].dims.d[0] : batch_size * input_seq_len;
    const int request_batch_size = batch_size;
    const int request_seq_len = input_seq_len;
    const int local_hidden_units_ = inputDesc[0].dims.d[mRemovePadding ? 1 : 2] / 3;
    const float q_scaling = mQScaling;

    const T* attention_input = reinterpret_cast<const T*>(inputs[0]);
    const int* input_lengths = reinterpret_cast<const int*>(inputs[1]);
    const T* relative_attn_table = mRelativeAttention ? reinterpret_cast<const T*>(inputs[3]) : nullptr;
    T* context_buf_ = (T*) (outputs[0]);

    auto cublasHandle = mCublasWrapper->getCublasHandle();
    TLLM_CUDA_CHECK(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(workspace);
    if (inputDesc[0].type == DataType::kHALF)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if (inputDesc[0].type == DataType::kFLOAT)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif

    const size_t attention_mask_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * input_seq_len;
    const size_t cu_seqlens_size = sizeof(int) * (batch_size + 1);
    const size_t q_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t k_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t v_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_size
        = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t qkv_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    const size_t qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    const size_t padding_offset_size = sizeof(int) * batch_size * input_seq_len;

    // Workspace pointer shift
    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(workspace);
    size_t offset = CUBLAS_WORKSPACE_SIZE;

    T* attention_mask = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, attention_mask_size));
    int* cu_seqlens = reinterpret_cast<int*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    T* q_buf_2_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, q_buf_2_size));
    T* k_buf_2_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, k_buf_2_size));
    T* v_buf_2_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, v_buf_2_size));
    T* qk_buf_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_size));
    T* qkv_buf_2_ = reinterpret_cast<T*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, qkv_buf_2_size));
    float* qk_buf_float_
        = reinterpret_cast<float*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_float_size));
    int* padding_offset = reinterpret_cast<int*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, padding_offset_size));

    // build attention_mask, cu_seqlens, and padding_offset tensors
    BuildDecoderInfoParams<T> params;
    memset(&params, 0, sizeof(params));
    params.seqQOffsets = cu_seqlens;
    params.paddingOffsets = padding_offset;
    params.attentionMask = attention_mask;
    params.seqQLengths = input_lengths;
    params.batchSize = batch_size;
    params.maxSeqLength = input_seq_len;
    params.numTokens = num_tokens;
    params.attentionMaskType = AttentionMaskType::PADDING;
    invokeBuildDecoderInfo(params, stream);
    sync_check_cuda_error();

    const auto gemm_data_type = tc::CudaDataType<T>::value;
    const int attention_seq_len_1 = request_seq_len; // q length
    const int attention_seq_len_2 = request_seq_len; // kv length

    // If the model has relative attentiona bias, q scaling should be applied in QK gemm stage and use 1 in
    // softamax stage (because to get softmax[scale(Q*K) + rel pos bias] here, q_scaling can't be applied during
    // softmax phase by qk_scale); otherwise, use 1 in gemm stage and apply scaling in softmax stage
    const float qk_scale
        = 1.0f / (sqrtf(mHeadSize * 1.0f) * q_scaling); // q_scaling in denominator. by default q_scaling =1.0f
    const float qk_scale_gemm = mRelativeAttention ? qk_scale : 1.0f;
    const T qk_scale_softmax = static_cast<T>(mRelativeAttention ? 1.0f : qk_scale);

    T* linear_bias_slopes = nullptr;

    // FMHA doesn't apply to MHA with relative attention bias, i.e. softmax(QK + bias) * V
    // We update mEnableContextFMHA in constructor to check this condition
    if (mEnableContextFMHA)
    {
        // b, max_seqlen, actual_total_seqlen
        mFMHARunner->setup(request_batch_size, request_seq_len, request_seq_len, request_batch_size * request_seq_len);
        mFMHARunner->run(const_cast<T*>(attention_input), cu_seqlens, context_buf_, stream);
    }
    else
    {
        // only non-FMHA path needs to split Q,K,V from QKV
        invokeAddFusedQKVBiasTranspose(q_buf_2_, k_buf_2_, v_buf_2_, const_cast<T*>(attention_input), input_lengths,
            mRemovePadding ? padding_offset : nullptr, batch_size, input_seq_len, num_tokens, mNumHeads, mNumHeads,
            mHeadSize, 0, 0.0f, RotaryScalingType::kNONE, 0.0f, 0, PositionEmbeddingType::kLEARNED_ABSOLUTE,
            (float*) nullptr, 0, stream);

        if (!mQKHalfAccum && gemm_data_type != CUDA_R_32F)
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,             // n
                attention_seq_len_1,             // m
                mHeadSize,                       // k
                qk_scale_gemm, k_buf_2_, gemm_data_type,
                mHeadSize,                       // k
                attention_seq_len_2 * mHeadSize, // n * k
                q_buf_2_, gemm_data_type,
                mHeadSize,                       // k
                attention_seq_len_1 * mHeadSize, // m * k
                0.0f, qk_buf_float_, CUDA_R_32F,
                attention_seq_len_2,             // n
                attention_seq_len_2 * attention_seq_len_1,
                request_batch_size * mNumHeads,  // global batch size
                CUDA_R_32F);

            // add relative position bias
            if (mRelativeAttention)
            {
                // add rel pos bias
                // QK is (batch_size, local_head_num, q_length, k_length), rel pos bias is (1, local_head_num,
                // max_output_len + 1, max_output_len + 1). broadcast along 1st dim. max_seq_len is already
                // max_output_len + 1. In implicit mode, relative_attention_bias is rel attn table
                // [num_heads, num_buckets], with necessary params (max_distance, num_buckets) passed at the end
                invokeAddRelativeAttentionBiasUnaligned(qk_buf_float_, relative_attn_table, request_batch_size,
                    mNumHeads, attention_seq_len_1, attention_seq_len_2, stream, mMaxDistance > 0,
                    inputDesc[3].dims.d[1], mMaxDistance, true /* bidirectional */);
            }

            MaskedSoftmaxParam<T, float> param;
            param.attention_score = qk_buf_;       // (batch_size, head_num, q_length, k_length)
            param.qk = qk_buf_float_;              // (batch_size, head_num, q_length, k_length)
            param.attention_mask = attention_mask; // (batch_size, q_length, k_length)
            param.batch_size = request_batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale_softmax;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            invokeMaskedSoftmax(param, stream);
        }
        else
        {
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N, attention_seq_len_2, attention_seq_len_1,
                mHeadSize, k_buf_2_, mHeadSize, attention_seq_len_2 * mHeadSize, q_buf_2_, mHeadSize,
                attention_seq_len_1 * mHeadSize, qk_buf_, attention_seq_len_2,
                attention_seq_len_2 * attention_seq_len_1, request_batch_size * mNumHeads, qk_scale_gemm,
                0.0f); // alpha, beta

            // add relative position bias
            if (mRelativeAttention)
            {
                // add rel pos bias
                // QK is (batch_size, local_head_num, q_length, k_length), rel pos bias is (1, local_head_num,
                // max_output_len + 1, max_output_len + 1). broadcast along 1st dim. max_seq_len is already
                // max_output_len + 1. In implicit mode, relative_attention_bias is rel attn table
                // [num_heads, num_buckets], with necessary params (max_distance, num_buckets) passed at the end
                invokeAddRelativeAttentionBiasUnaligned(qk_buf_, relative_attn_table, request_batch_size, mNumHeads,
                    attention_seq_len_1, attention_seq_len_2, stream, mMaxDistance > 0, inputDesc[3].dims.d[1],
                    mMaxDistance, true /* bidirectional */);
            }

            MaskedSoftmaxParam<T, T> param;
            param.attention_score = qk_buf_;       // (batch_size, head_num, q_length, k_length)
            param.qk = qk_buf_;                    // (batch_size, head_num, q_length, k_length)
            param.attention_mask = attention_mask; // (batch_size, q_length, k_length)
            param.batch_size = request_batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale_softmax;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            invokeMaskedSoftmax(param, stream);
        }

        mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N, mHeadSize, attention_seq_len_1,
            attention_seq_len_2, v_buf_2_, mHeadSize, attention_seq_len_2 * mHeadSize, qk_buf_, attention_seq_len_2,
            attention_seq_len_1 * attention_seq_len_2, qkv_buf_2_, mHeadSize, attention_seq_len_1 * mHeadSize,
            request_batch_size * mNumHeads);

        if (!mRemovePadding)
        {
            invokeTransposeQKV(context_buf_, qkv_buf_2_, request_batch_size, attention_seq_len_1, mNumHeads, mHeadSize,
                (float*) nullptr, 0, stream);
        }
        else
        {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_2_, context_buf_, num_tokens, request_batch_size,
                request_seq_len, mNumHeads, mHeadSize, padding_offset, (float*) nullptr, 0, stream);
        }
    }
    return 0;
}

template int BertAttentionPlugin::enqueueImpl<half>(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

template int BertAttentionPlugin::enqueueImpl<float>(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

#ifdef ENABLE_BF16
template int BertAttentionPlugin::enqueueImpl<__nv_bfloat16>(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);
#endif

int BertAttentionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (mType == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        return enqueueImpl<__nv_bfloat16>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#endif
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType BertAttentionPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

const char* BertAttentionPlugin::getPluginType() const noexcept
{
    return BERT_ATTENTION_PLUGIN_NAME;
}

const char* BertAttentionPlugin::getPluginVersion() const noexcept
{
    return BERT_ATTENTION_PLUGIN_VERSION;
}

int BertAttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int BertAttentionPlugin::initialize() noexcept
{
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();
    mCublasWrapper.reset(new tc::CublasMMWrapper(cublasHandle, cublasLtHandle, nullptr, nullptr));
    if (mEnableContextFMHA)
    {
        mFMHARunner.reset(new FusedMHARunnerV2(DATA_TYPE_FP16, mNumHeads, mHeadSize, mQScaling));
        // set flags: force_fp32_acc, is_s_padded, causal_mask, num_kv_heads = num_heads
        mFMHARunner->setup_flags(mFMHAForceFP32Acc, true, false, mNumHeads);
    }

    return 0;
}

void BertAttentionPlugin::destroy() noexcept
{
    delete this;
}

size_t BertAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize) + sizeof(mQScaling) + sizeof(mQKHalfAccum) + sizeof(mEnableContextFMHA)
        + sizeof(mFMHAForceFP32Acc) + sizeof(mType) + sizeof(mRelativeAttention) + sizeof(mMaxDistance)
        + sizeof(mRemovePadding);
}

void BertAttentionPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mNumHeads);
    write(d, mHeadSize);
    write(d, mQScaling);
    write(d, mQKHalfAccum);
    write(d, mEnableContextFMHA);
    write(d, mFMHAForceFP32Acc);
    write(d, mType);
    write(d, mRelativeAttention);
    write(d, mMaxDistance);
    write(d, mRemovePadding);
    assert(d == a + getSerializationSize());
}

void BertAttentionPlugin::terminate() noexcept {}

///////////////

BertAttentionPluginCreator::BertAttentionPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("q_scaling", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mPluginAttributes.emplace_back(PluginField("enable_qk_half_accum", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("context_fmha_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("do_relative_attention", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("max_distance", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("remove_padding", nullptr, PluginFieldType::kINT8, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* BertAttentionPluginCreator::getPluginName() const noexcept
{
    return BERT_ATTENTION_PLUGIN_NAME;
}

const char* BertAttentionPluginCreator::getPluginVersion() const noexcept
{
    return BERT_ATTENTION_PLUGIN_VERSION;
}

const PluginFieldCollection* BertAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* BertAttentionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int num_heads, head_size;
    ContextFMHAType context_fmha_type;
    bool qk_half_accum;
    float q_scaling;
    nvinfer1::DataType type;
    bool do_relative_attention;
    int max_distance;
    bool remove_padding;
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "num_heads"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            num_heads = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "head_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            head_size = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "q_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            q_scaling = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "enable_qk_half_accum"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            qk_half_accum = static_cast<bool>(*(static_cast<const int8_t*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "context_fmha_type"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            context_fmha_type = static_cast<ContextFMHAType>(*(static_cast<const int8_t*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<const nvinfer1::DataType*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "do_relative_attention"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            do_relative_attention = static_cast<bool>(*(static_cast<const int8_t*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "max_distance"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            max_distance = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "remove_padding"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            remove_padding = static_cast<bool>(*(static_cast<const int8_t*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new BertAttentionPlugin(num_heads, head_size, q_scaling, qk_half_accum, context_fmha_type, type,
            do_relative_attention, max_distance, remove_padding);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* BertAttentionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call BertAttentionPlugin::destroy()
    try
    {
        auto* obj = new BertAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
