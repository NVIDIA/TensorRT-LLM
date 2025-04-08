/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
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

#include "fused_multihead_attention_common.h"

#include "cubin/fmha_cubin.h"
#include "cuda_runtime_api.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tmaDescriptor.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>

namespace tensorrt_llm::kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// Base Class

template <typename TKernelMeta, typename TKernelParam>
class TFusedMultiHeadAttentionXMMAKernel
{
public:
    using KernelMeta = TKernelMeta;
    using KernelParam = TKernelParam;

    inline uint64_t hashID(unsigned int s, unsigned int d) const;
    virtual uint64_t hashID(KernelMeta const& kernelMeta) const;

    TFusedMultiHeadAttentionXMMAKernel(TKernelMeta const* pMetaStart, unsigned int nMetaCount, Data_type inputType,
        Data_type outputType, unsigned int sm);

    void loadXMMAKernels();
    bool isValid(int s) const;
    virtual void run(TKernelParam& params, Launch_params& launch_params, cudaStream_t stream) const;
    virtual bool checkIfKernelExist(MHARunnerFixedParams params) const = 0;

    virtual void getStepSize(uint32_t& out_step_q, uint32_t& out_step_kv, TKernelParam const& params,
        Launch_params const& launch_params) const
        = 0;

    virtual ~TFusedMultiHeadAttentionXMMAKernel() = default;

protected:
    struct FusedMultiHeadAttentionKernelInfo
    {
        unsigned int mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };

    struct KernelExistPredicate
    {
        KernelExistPredicate(uint64_t id)
            : mId(id)
        {
        }

        bool operator()(std::pair<uint64_t, FusedMultiHeadAttentionKernelInfo> const& v) const
        {
            return (v.first & mId) == mId;
        }

    private:
        uint64_t mId;
    };

protected:
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;

    Data_type mInputDataType;
    Data_type mOutputDataType;
    TKernelMeta const* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    std::unordered_map<unsigned char const*, CUmodule> mModules;
    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
    std::set<int> mValidSequences;
};

template <typename TFusedMHAKernelList>
class TFusedMHAKernelFactory
{
public:
    TFusedMHAKernelList const* getXMMAKernels(const typename TFusedMHAKernelList::KernelMeta* pKernelList,
        unsigned int nbKernels, Data_type inputType, Data_type outputType, unsigned int sm);

    static TFusedMHAKernelFactory<TFusedMHAKernelList>& Get();

private:
    TFusedMHAKernelFactory() = default;

    inline uint64_t hashID(Data_type inputType, Data_type outputType, unsigned int sm) const;

    std::unordered_map<uint64_t, const std::unique_ptr<TFusedMHAKernelList>> mKernels;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class FusedMultiHeadAttentionXMMAKernelV2
    : public TFusedMultiHeadAttentionXMMAKernel<FusedMultiHeadAttentionKernelMetaInfoV2,
          Fused_multihead_attention_params_v2>
{
public:
    FusedMultiHeadAttentionXMMAKernelV2(FusedMultiHeadAttentionKernelMetaInfoV2 const* pMetaStart,
        unsigned int nMetaCount, Data_type inputType, Data_type outputType, unsigned int sm);

    uint64_t hashID(unsigned int s, unsigned int d, unsigned int dv, bool interleaved, bool unroll, bool force_fp32_acc,
        bool flash_attention, bool warp_specialization, bool is_alibi_supported, int attention_mask_type,
        int input_layout, bool tiled, bool enable_attn_logit_softcapping, unsigned int sage_block_size_q,
        unsigned int sage_block_size_k, unsigned int sage_block_size_v, bool return_softmax) const;

    uint64_t hashID(KernelMeta const& kernelMeta) const override;

    // FMHA runner.
    void run(
        Fused_multihead_attention_params_v2& params, Launch_params& launch_params, cudaStream_t stream) const override;

    // Check if any kernels support the attention types during building the engines.
    bool checkIfKernelExist(MHARunnerFixedParams params) const override;

    void getStepSize(uint32_t& out_step_q, uint32_t& out_step_kv, Fused_multihead_attention_params_v2 const& params,
        Launch_params const& launch_params) const override;

private:
    bool useForceUnroll(Fused_multihead_attention_params_v2 const& params, Launch_params const& launch_params) const;
    uint64_t hashFromParams(
        Fused_multihead_attention_params_v2 const& params, Launch_params const& launch_params) const;
};

using FusedMHAKernelFactoryV2 = TFusedMHAKernelFactory<FusedMultiHeadAttentionXMMAKernelV2>;

FusedMultiHeadAttentionXMMAKernelV2 const* getXMMAKernelsV2(Data_type inputType, Data_type outputType, unsigned int sm);

} // namespace tensorrt_llm::kernels
