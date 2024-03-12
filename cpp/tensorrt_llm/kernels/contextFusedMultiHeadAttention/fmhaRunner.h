/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "fused_multihead_attention_common.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tmaDescriptor.h"

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

class MHARunner
{
public:
    MHARunner(const Data_type dataType, int const numHeads, int const headSize, float const qScaling);

    MHARunner() = default;

    virtual ~MHARunner() = default;

    virtual void setup(int const b, int const s, int const sliding_window_size, int const total_seqlen,
        bool const has_alibi = false, bool const scale_alibi = false, int const tp_size = 1, int const tp_rank = 0)
        = 0;

    virtual void setup_paged_kv(int const b, int const s_q, int const s_kv, int const blocks_per_context_sequence,
        int const tokens_per_kv_block, int const sliding_window_size, int const total_seqlen,
        bool const has_alibi = false, bool const scale_alibi = false, int const tp_size = 1, int const tp_rank = 0)
        = 0;

    static bool fmha_supported(int const headSize, int const sm);

    virtual bool fmha_supported() = 0;

    virtual void setup_flags(bool const force_fp32_acc, bool const is_s_padded, bool const causal_mask,
        int const num_kv_heads /* MQA or GQA */)
        = 0;

    virtual void run(void const* input, void const* cu_seqlens, void* output, cudaStream_t stream) = 0;

    virtual void run_paged_kv(void const* q_input, void* paged_kv_tma_desc, void const* paged_kv_block_ptrs_on_host,
        const KVBlockArray paged_kv_cache, void const* cu_q_seqlens, void const* cu_kv_seqlens, void* output,
        cudaStream_t stream)
        = 0;

    virtual bool isValid(int s) const = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Workflow of fmha runner:
// 1. check if FMHA kernels are supported statically.
// 2. construct FMHA runner object.
// 3. setup_flags (used by all kernels).
// 4. setup runtime parameters (used by this specific case).
// 5. run the kernel (with all needed device pointers).

class FusedMHARunnerV2 : public MHARunner
{
public:
    FusedMHARunnerV2(const Data_type dataType, int const numHeads, int const headSize, float const qScaling);

    ~FusedMHARunnerV2(); // for pimpl

    void setup(int const b, int const s, int const sliding_window_size, int const total_seqlen,
        bool const has_alibi = false, bool const scale_alibi = false, int const tp_size = 1,
        int const tp_rank = 0) override;

    void setup_paged_kv(int const b, int const s_q, int const s_kv, int const blocks_per_context_sequence,
        int const tokens_per_kv_block, int const sliding_window_size, int const total_seqlen,
        bool const has_alibi = false, bool const scale_alibi = false, int const tp_size = 1,
        int const tp_rank = 0) override;

    bool fmha_supported() override;

    void run(void const* input, void const* cu_seqlens, void* output, cudaStream_t stream) override;
    void run_paged_kv(void const* q_input, void* paged_kv_tma_desc, void const* paged_kv_block_ptrs_on_host,
        const KVBlockArray paged_kv_cache, void const* cu_q_seqlens, void const* cu_kv_seqlens, void* output,
        cudaStream_t stream) override;

    void setup_flags(bool const force_fp32_acc, bool const is_s_padded, bool const causal_mask,
        int const num_kv_heads /* MQA or GQA */) override;

    bool isValid(int s) const override;

private:
    class mhaImpl;
    std::unique_ptr<mhaImpl> pimpl;
};

} // namespace kernels
} // namespace tensorrt_llm
