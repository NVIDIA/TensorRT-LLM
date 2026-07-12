/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_GATED_DELTA_NET_PLUGIN_H
#define TRT_GATED_DELTA_NET_PLUGIN_H
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <cstdint>

namespace tensorrt_llm::plugins
{
// Gated DeltaNet (GDN) linear-attention recurrence plugin.
//
// A naive, correctness-first port of the GDN recurrence into the TensorRT
// plugin framework. The recurrence is, per (batch b, v-head h), sequential over
// tokens, with the recurrent state S[D_k, D_v] kept in fp32:
//
//   qn = l2norm(q_t) ; kn = l2norm(k_t)          # over D_k, eps = 1e-6
//   qn *= 1 / sqrt(D_k)                           # SCALE on q only
//   S  *= exp(g_t)                                # elementwise decay
//   kvmem[j] = sum_i S[i,j] * kn[i]               # read with normalized key
//   delta[j] = (v_t[j] - kvmem[j]) * beta_t       # correction
//   S[i,j]  += kn[i] * delta[j]                   # rank-1 outer-product update
//   y_t[j]   = sum_i S[i,j] * qn[i]               # read with scaled query (updated S)
//
// batch_size = num_ctx_requests or num_gen_requests
//   num_ctx_requests = number of context requests (single sequence per request).
//   num_gen_requests = number of generation requests (single sequence per request).
// Cannot support beam search.
//
// inputs
//     0.  q     [B, T, H_v, D_k] or [num_tokens, H_v, D_k] for remove_input_padding
//     1.  k     same shape as q
//     2.  v     [B, T, H_v, D_v] or [num_tokens, H_v, D_v] for remove_input_padding
//     3.  g     [B, T, H_v]      or [num_tokens, H_v]      (log-decay, <= 0)
//     4.  beta  [B, T, H_v]      or [num_tokens, H_v]      (in (0,1))
//     5.  state_or_ptr [B, H_v, D_k, D_v] fp32, or host [1] int64 pointer for paged_state
//     6.  host_request_types [B] int32. 0: context; 1: generation.
//     7.  last_token_ids     [B] int32
//     8.  host_context_lengths [B] int32, optional iff remove_input_padding
//     9.  slot_mapping         [B] int32, optional iff paged_state
// outputs
//     0.  y             [B, T, H_v, D_v] or [num_tokens, H_v, D_v] for remove_input_padding
//     1.  present_state [B, H_v, D_k, D_v] fp32, omitted iff paged_state (updated in place via ptr)

// Parameters passed to the CUDA launcher. Kept POD so it can live in this header
// shared between the plugin .cpp and the kernel .cu.
struct GatedDeltaNetParams
{
    // sizes
    int batch;     // B
    int maxSeqLen; // T (max sequence length of the dense [B,T,...] tensors)
    int numVHeads; // H_v
    int headKDim;  // D_k
    int headVDim;  // D_v

    // device pointers (all fp32 unless noted)
    void const* q;           // [B,T,H_v,D_k]
    void const* k;           // [B,T,H_v,D_k]
    void const* v;           // [B,T,H_v,D_v]
    void const* g;           // [B,T,H_v]
    void const* beta;        // [B,T,H_v]
    void const* statePtrIn;  // [B,H_v,D_k,D_v] input recurrent state
    int const* seqLens;      // [B] int32, per-request valid token count (gen => 1)
    int const* hostReqTypes; // host [B] int32, 0: context, 1: generation

    // outputs
    void* y;           // [B,T,H_v,D_v]
    void* statePtrOut; // [B,H_v,D_k,D_v] output recurrent state

    bool useQkL2norm;  // whether to L2-normalize q,k inside the kernel
};

// Launcher implemented in gatedDeltaNetKernel.cu.
void invokeGatedDeltaNet(GatedDeltaNetParams const& params, cudaStream_t stream);

class GatedDeltaNetPlugin : public BasePlugin
{
public:
    GatedDeltaNetPlugin(int numVHeads, int headKDim, int headVDim, int chunkSize, bool useQkL2norm,
        nvinfer1::DataType type, bool removePadding, bool pagedState);

    GatedDeltaNetPlugin(void const* data, size_t length);

    ~GatedDeltaNetPlugin() override = default;

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

    IndexType getQIdx() const
    {
        return 0;
    };

    IndexType getKIdx() const
    {
        return 1;
    };

    IndexType getVIdx() const
    {
        return 2;
    };

    IndexType getGIdx() const
    {
        return 3;
    };

    IndexType getBetaIdx() const
    {
        return 4;
    };

    IndexType getStateIdx() const
    {
        return 5;
    };

    IndexType getHostRequestTypesIdx() const
    {
        return 6;
    };

    IndexType getLastTokenIdsIdx() const
    {
        return 7;
    };

    IndexType getHostContextLengthIdx() const
    {
        if (mRemovePadding)
            return 8;
        else
            return 7;
    };

    IndexType getSlotMappingIdx() const
    {
        if (mPagedState)
            return getHostContextLengthIdx() + 1;
        else
            return getHostContextLengthIdx();
    };

private:
    int mNumVHeads;
    int mHeadKDim;
    int mHeadVDim;
    int mChunkSize;
    bool mUseQkL2norm;
    nvinfer1::DataType mType;
    bool mRemovePadding = false;
    bool mPagedState = false;
};

class GatedDeltaNetPluginCreator : public BaseCreator
{
public:
    GatedDeltaNetPluginCreator();

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

#endif // TRT_GATED_DELTA_NET_PLUGIN_H
