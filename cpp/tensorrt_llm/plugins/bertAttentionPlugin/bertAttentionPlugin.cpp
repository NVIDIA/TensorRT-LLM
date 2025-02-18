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
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/iBuffer.h"

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

using tensorrt_llm::plugins::BertAttentionPluginCreator;
using tensorrt_llm::plugins::BertAttentionPlugin;

static char const* BERT_ATTENTION_PLUGIN_VERSION{"1"};
static char const* BERT_ATTENTION_PLUGIN_NAME{"BertAttention"};
PluginFieldCollection BertAttentionPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> BertAttentionPluginCreator::mPluginAttributes;

BertAttentionPlugin::BertAttentionPlugin(int num_heads, int head_size, float q_scaling,
    ContextFMHAType context_fmha_type, nvinfer1::DataType type, bool do_relative_attention, int max_distance,
    bool remove_padding)
    : mNumHeads(num_heads)
    , mHeadSize(head_size)
    , mQScaling(q_scaling)
    , mType(type)
    , mRelativeAttention(do_relative_attention)
    , mMaxDistance(max_distance)
    , mRemovePadding(remove_padding)
    , mEnableContextFMHA(context_fmha_type != ContextFMHAType::DISABLED)
    , mFMHAForceFP32Acc(context_fmha_type == ContextFMHAType::ENABLED_WITH_FP32_ACC)
{
    // pre-check whether FMHA is supported in order to save memory allocation
    if (mEnableContextFMHA)
    {
        mEnableContextFMHA = false;
        if (!(mType == DataType::kHALF || mType == DataType::kBF16))
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of unsupported data type.");
        }
        else if (mRelativeAttention)
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of relative position embedding.");
        }
        else
        {
            mEnableContextFMHA = true;
        }
    }
}

// Parameterized constructor
BertAttentionPlugin::BertAttentionPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
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
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT-LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
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
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex == 0);
    auto ret = inputs[0];
    ret.d[mRemovePadding ? 1 : 2] = exprBuilder.constant(ret.d[mRemovePadding ? 1 : 2]->getConstantValue() / 3);
    return ret;
}

bool BertAttentionPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    // inputs: [0] qkv, [1] input_lengths, [2] max_input_length (optional), [3] relative_attention_bias (optional)
    // outputs: [X] hidden_states
    if (nbInputs == 2)
    { // BERT
        if (pos == 1)
        {
            return inOut[pos].type == nvinfer1::DataType::kINT32;
        }

        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    if (nbInputs > 2)
    { // Encoder in encoder-decoder
        if (pos == 1 || pos == 2)
        {
            return inOut[pos].type == nvinfer1::DataType::kINT32;
        }

        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }

    return false;
}

void BertAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t BertAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    // if remove padding, inputs[0] "qkv_hidden_states" dim is [num_tokens, 3*hidden_dim] which doesn't have shape
    // info should get max_batch_size and max_input_length from inputs[1] "input_lengths" and input[2]
    // "max_input_length"
    int const batch_size = mRemovePadding ? inputs[1].dims.d[0] : inputs[0].dims.d[0];
    int const input_seq_len = mRemovePadding ? inputs[2].dims.d[0] : inputs[0].dims.d[1];
    int const local_hidden_units_ = inputs[0].dims.d[mRemovePadding ? 1 : 2] / 3;

    auto const size = tensorrt_llm::runtime::BufferDataType(inputs[0].type).getSize();

    size_t const attention_mask_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * input_seq_len;
    size_t const cu_seqlens_size = sizeof(int) * (batch_size + 1);
    size_t const q_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    size_t const k_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    size_t const v_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    size_t const qk_buf_size = mEnableContextFMHA ? 0 : size * batch_size * mNumHeads * input_seq_len * input_seq_len;
    size_t const qkv_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_len * local_hidden_units_;
    size_t const qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    size_t const padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * batch_size * input_seq_len;
    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;

    int const NUM_BUFFERS = 11;
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
    workspaces[10] = fmha_scheduler_counter;

    return tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
}

template <typename T>
int BertAttentionPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
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
    int const batch_size = mRemovePadding ? inputDesc[1].dims.d[0] : inputDesc[0].dims.d[0];
    int const input_seq_len = mRemovePadding ? inputDesc[2].dims.d[0] : inputDesc[0].dims.d[1];
    int const num_tokens = mRemovePadding ? inputDesc[0].dims.d[0] : batch_size * input_seq_len;
    int const request_batch_size = batch_size;
    int const request_seq_len = input_seq_len;
    int const local_hidden_units_ = inputDesc[0].dims.d[mRemovePadding ? 1 : 2] / 3;
    float const q_scaling = mQScaling;

    T const* attention_input = reinterpret_cast<T const*>(inputs[0]);
    int const* input_lengths = reinterpret_cast<int const*>(inputs[1]);
    T const* relative_attn_table = mRelativeAttention ? reinterpret_cast<T const*>(inputs[3]) : nullptr;
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
    size_t const cu_seqlens_size = sizeof(int) * (batch_size + 1);
    size_t const q_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    size_t const k_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    size_t const v_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    size_t const qk_buf_size
        = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    size_t const qkv_buf_2_size = mEnableContextFMHA ? 0 : sizeof(T) * batch_size * input_seq_len * local_hidden_units_;
    size_t const qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_len * input_seq_len;
    size_t const padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * batch_size * input_seq_len;
    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;

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
    uint32_t* fmha_tile_counter_ptr
        = reinterpret_cast<uint32_t*>(tc::nextWorkspacePtr(workspace_byte_ptr, offset, fmha_scheduler_counter));

    // build attention_mask, cu_seqlens, and padding_offset tensors
    BuildDecoderInfoParams<T> params{};
    params.seqQOffsets = cu_seqlens;
    params.paddingOffsets = padding_offset;
    params.attentionMask = attention_mask;
    params.seqQLengths = input_lengths;
    params.batchSize = batch_size;
    params.maxQSeqLength = input_seq_len;
    params.numTokens = num_tokens;
    params.attentionMaskType = AttentionMaskType::PADDING;
    params.fmhaTileCounter = fmha_tile_counter_ptr;
    invokeBuildDecoderInfo(params, stream);
    sync_check_cuda_error();

    auto const gemm_data_type = tc::CudaDataType<T>::value;
    int const attention_seq_len_1 = request_seq_len; // q length
    int const attention_seq_len_2 = request_seq_len; // kv length

    // If the model has relative attentiona bias, q scaling should be applied in QK gemm stage and use 1 in
    // softamax stage (because to get softmax[scale(Q*K) + rel pos bias] here, q_scaling can't be applied during
    // softmax phase by qk_scale); otherwise, use 1 in gemm stage and apply scaling in softmax stage
    float const qk_scale
        = 1.0f / (sqrtf(mHeadSize * 1.0f) * q_scaling); // q_scaling in denominator. by default q_scaling =1.0f
    float const qk_scale_gemm = mRelativeAttention ? qk_scale : 1.0f;
    T const qk_scale_softmax = static_cast<T>(mRelativeAttention ? 1.0f : qk_scale);

    T* linear_bias_slopes = nullptr;

    // FMHA doesn't apply to MHA with relative attention bias, i.e. softmax(QK + bias) * V
    // We update mEnableContextFMHA in constructor to check this condition
    if (mEnableContextFMHA)
    {
        // Construct the fmha params for running kernels.
        MHARunnerParams fmhaParams{};
        fmhaParams.b = request_batch_size;
        fmhaParams.qSeqLen = request_seq_len;
        fmhaParams.kvSeqLen = request_seq_len;
        fmhaParams.totalQSeqLen = request_batch_size * request_seq_len;
        // Device buffer pointers.
        fmhaParams.qkvPtr = attention_input;
        fmhaParams.outputPtr = context_buf_;
        fmhaParams.cuQSeqLenPtr = cu_seqlens;
        fmhaParams.cuKvSeqLenPtr = cu_seqlens;
        fmhaParams.tileCounterPtr = fmha_tile_counter_ptr;
        fmhaParams.stream = stream;

        // Run the fmha kernel.
        mFMHARunner->run(fmhaParams);
    }
    else
    {
        // FIXME: a temporary solution to make sure the padding part of key/value buffer is 0
        // NOTE: pointer subtraction is used below since there could be some extra gap due to alignment.
        //  Otherwise, we could do cudaMemsetAsync(k_buf_2_, 0, k_buf_2_size + v_buf_2_size, stream);
        // cudaMemsetAsync(k_buf_2_, 0, reinterpret_cast<int8_t*>(qk_buf_) - reinterpret_cast<int8_t*>(k_buf_2_),
        // stream);
        // FIXME: the final solution is to change the add_fusedQKV_bias_transpose_kernel to map CTAs corresponding to
        // the output shape, and set the padding part to 0. Without zero-initialize guarantee, these workspace buffers
        // may contain random NaN values when IFB workload is high.
        cudaMemsetAsync(k_buf_2_, 0,
            reinterpret_cast<int8_t*>(v_buf_2_) - reinterpret_cast<int8_t*>(k_buf_2_) + v_buf_2_size, stream);

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
    sync_check_cuda_error();
    return 0;
}

template int BertAttentionPlugin::enqueueImpl<half>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

template int BertAttentionPlugin::enqueueImpl<float>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);

#ifdef ENABLE_BF16
template int BertAttentionPlugin::enqueueImpl<__nv_bfloat16>(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream);
#endif

int BertAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
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
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0);
    return inputTypes[0];
}

// IPluginV2 Methods

char const* BertAttentionPlugin::getPluginType() const noexcept
{
    return BERT_ATTENTION_PLUGIN_NAME;
}

char const* BertAttentionPlugin::getPluginVersion() const noexcept
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
        // Pre-checked during constructing.
        Data_type data_type;
        if (mType == DataType::kHALF)
        {
            data_type = DATA_TYPE_FP16;
        }
        else if (mType == DataType::kBF16)
        {
            data_type = DATA_TYPE_BF16;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "GPTAttentionPlugin received wrong data type.");
        }

        // Construct the fmha runner.
        MHARunnerFixedParams fmhaParams{};
        fmhaParams.dataType = data_type;
        fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;
        fmhaParams.attentionMaskType = ContextAttentionMaskType::PADDING;
        fmhaParams.isSPadded = !mRemovePadding;
        fmhaParams.numQHeads = mNumHeads;
        fmhaParams.numKvHeads = mNumHeads;
        fmhaParams.headSize = mHeadSize;
        fmhaParams.qScaling = mQScaling;

        // Load kernels from the pre-compiled cubins.
        mFMHARunner.reset(new FusedMHARunnerV2(fmhaParams));

        // Fall back to unfused MHA kernels if not supported.
        mEnableContextFMHA = mFMHARunner->isFmhaSupported();
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
    TLLM_CHECK(d == a + getSerializationSize());
}

void BertAttentionPlugin::terminate() noexcept {}

///////////////

BertAttentionPluginCreator::BertAttentionPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("q_scaling", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("context_fmha_type", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("do_relative_attention", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("max_distance", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("remove_padding", nullptr, PluginFieldType::kINT8));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* BertAttentionPluginCreator::getPluginName() const noexcept
{
    return BERT_ATTENTION_PLUGIN_NAME;
}

char const* BertAttentionPluginCreator::getPluginVersion() const noexcept
{
    return BERT_ATTENTION_PLUGIN_VERSION;
}

PluginFieldCollection const* BertAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* BertAttentionPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int num_heads{};
    int head_size{};
    ContextFMHAType context_fmha_type{};
    float q_scaling{};
    nvinfer1::DataType type{};
    bool do_relative_attention{};
    int max_distance{};
    bool remove_padding{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "num_heads"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            num_heads = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "head_size"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            head_size = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "q_scaling"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kFLOAT32);
            q_scaling = static_cast<float>(*(static_cast<float const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "context_fmha_type"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            context_fmha_type = static_cast<ContextFMHAType>(*(static_cast<int8_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "do_relative_attention"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            do_relative_attention = static_cast<bool>(*(static_cast<int8_t const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "max_distance"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            max_distance = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "remove_padding"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            remove_padding = static_cast<bool>(*(static_cast<int8_t const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new BertAttentionPlugin(num_heads, head_size, q_scaling, context_fmha_type, type,
            do_relative_attention, max_distance, remove_padding);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* BertAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call BertAttentionPlugin::destroy()
    try
    {
        auto* obj = new BertAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
