/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"

using namespace tensorrt_llm::common;
using tensorrt_llm::common::op::UniqPtrWNullCopy;

namespace tensorrt_llm::kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct XqaFixedParams
{
    // Whether the attention is MLA.
    bool isMLA;
    // The QKV input data type.
    kernels::Data_type inputDataType;
    // The XQA KV cache data type.
    kernels::Data_type kvDataType;
    // The XQA output data type.
    kernels::Data_type outputDataType;
    // The XQA BMM dtype.
    kernels::Data_type mathDataType;
    // The number of Q heads.
    int numQHeads;
    // The number of Kv Heads.
    int numKvHeads;
    // The number of tokens per kv cache block.
    int numTokensPerBlock;
    // The head size.
    int headSize;
    // The scaling applied to bmm1_scale.
    float qScaling;
    // Whether to enable multi block mode.
    bool multiBlockMode;
    // The KV cache layout.
    bool isPagedKv;
    // Is speculative decoding enabled.
    bool isSpecDecoding;
    // Do we apply alibi ?
    bool hasAlibi;
};

class XqaDispatcher
{
public:
    // Constructor.
    XqaDispatcher(XqaFixedParams fixedParams);

    // Deconstructor.
    ~XqaDispatcher() = default;

    // Prepare for DecoderXQARunner.
    void prepare(XQAParams const& params);

    // Check whether XQA is supported.
    bool isSupported();

    // Run the XQA kernel.
    void run(XQAParams const& params, KVLinearBuffer const& kv_cache_buffer,
        KVLinearBuffer const& kv_cache_block_scales_buffer);

    void run(
        XQAParams const& params, KVBlockArray const& kv_cache_buffer, KVBlockArray const& kv_cache_block_scales_buffer);

    int getWorkspaceAlignment();

    size_t getWorkspaceSize(int max_num_tokens);

    bool shouldUse(XQAParams const& params);

private:
    // The fixed XQA parameters.
    XqaFixedParams mFixedParams;
    // The data type of tensor Q, which determines the Q input data type of fmha kernels.
    Data_type mQDataType;
    // Whether to enable trtllm-gen kernels.
    bool mUseTllmGen;
    // The multi-processor count.
    int mMultiProcessorCount;
    // Runner for decoder XQA kernels (for SM <= 90)
    UniqPtrWNullCopy<DecoderXQARunner> mDecoderXqaRunner;
    // Runner for trtllm-gen XQA kernels (for SM == 100)
    UniqPtrWNullCopy<TllmGenFmhaRunner> mTllmGenFMHARunner;

protected:
    template <typename T, typename KVCacheBuffer>
    void runImpl(
        XQAParams params, KVCacheBuffer const& kv_cache_buffer, KVCacheBuffer const& kv_cache_block_scales_buffer);
};

constexpr uint32_t xqaMlaCgaXBufSize = 8704 * 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tensorrt_llm::kernels
