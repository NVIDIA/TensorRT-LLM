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

// The attention op of the target's 16 attention layers. It takes the fused [Q | K | V] rows the target's
// qkv_gemma_rmsnorm_rope_gate wrote (RoPE and the q/k norms already applied) and, per step:
//  - context rows [0, numCtxTokens) of sequences [0, numContexts): sequence offsets and FMHA scales
//    (invokeBuildDecoderInfo), K/V appended to the FP8 paged cache and Q quantized to FP8 (invokeQKVPreprocessing),
//    then the target's SM120 context FMHA (csrc/fmha) over the cached prefix and the chunk;
//  - generation rows [numCtxTokens, tokens) of the remaining sequences: K/V of the new token appended and Q copied
//    (invokeQKVPreprocessing), then the target's XQA decode kernel (csrc/xqa) with the shared-prefix pass.
// It reproduces what TensorRT-LLM's attention op runs for this configuration (head size 256, 24 query / 4 KV heads,
// BF16, FP8 KV cache, paged context FMHA, beam width 1, no speculative decoding), with this target's kernels.

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/contextFmhaSm120Run.h"
#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/xqaDecodeSm120.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

using torch::Tensor;
namespace tk = tensorrt_llm::kernels;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

constexpr int kQHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadSize = 256;
constexpr int kQWidth = kQHeads * kHeadSize;                    // 6144
constexpr int kQkvWidth = (kQHeads + 2 * kKvHeads) * kHeadSize; // 8192
constexpr size_t kAlign = 256;

size_t alignUp(size_t a)
{
    return (a + kAlign - 1) / kAlign * kAlign;
}

// The op's workspace, carved in this order (each slice 256-byte aligned). Context and generation slices do not
// overlap, so a mixed step's generation part never waits on the context part's buffers.
struct WorkspaceLayout
{
    size_t cuQ, cuKv, tokensInfo, qFp8, bmm1, bmm2, tileCounter, split, genQ, xqaScratch, total;

    WorkspaceLayout(int64_t maxTokens, int64_t maxSeqs)
    {
        size_t o = 0;
        auto take = [&o](size_t bytes)
        {
            size_t const at = o;
            o += alignUp(bytes);
            return at;
        };
        cuQ = take(sizeof(int32_t) * (maxSeqs + 1));
        cuKv = take(sizeof(int32_t) * (maxSeqs + 1));
        tokensInfo = take(sizeof(int2) * maxTokens);
        qFp8 = take(size_t{kQWidth} * maxTokens);
        bmm1 = take(sizeof(float) * 2);
        bmm2 = take(sizeof(float));
        tileCounter = take(sizeof(uint32_t));
        split = take(kernels::contextFmhaSm120SplitWorkspaceBytes());
        genQ = take(sizeof(__nv_bfloat16) * kQWidth * maxSeqs);
        xqaScratch = take(kernels::xqaSm120ScratchBytes());
        total = o;
    }
};

int multiProcessorCount(int device)
{
    static int cached[64] = {};
    TORCH_CHECK(device >= 0 && device < 64, "device index out of range");
    if (cached[device] == 0)
    {
        TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&cached[device], cudaDevAttrMultiProcessorCount, device));
    }
    return cached[device];
}

int hostMax(Tensor const& t, int64_t lo, int64_t hi)
{
    auto const* p = t.data_ptr<int32_t>();
    return *std::max_element(p + lo, p + hi);
}

// The layer's KV cache view for sequences [seqOffset, seqOffset + batch) (as the stock op builds it).
tk::KVBlockArray kvCacheView(Tensor const& blockOffsets, Tensor const& hostPoolPointers, Tensor const& hostPoolMapping,
    int64_t layerIdx, int64_t seqOffset, int batch, int tokensPerBlock, int maxSeqLen)
{
    auto const* mapping = hostPoolMapping.data_ptr<int32_t>();
    int32_t const poolIdx = mapping[layerIdx * hostPoolMapping.size(1) + 0];
    int32_t const layerInPool = mapping[layerIdx * hostPoolMapping.size(1) + 1];
    int64_t const pointerStride = hostPoolPointers.numel() / hostPoolPointers.size(0);
    auto const* pointers = hostPoolPointers.data_ptr<int64_t>();
    int32_t const bytesPerToken = kKvHeads * kHeadSize; // FP8
    int64_t const bytesPerBlock = int64_t{tokensPerBlock} * bytesPerToken;
    int64_t const layerOffset = int64_t{layerInPool} * 2 * bytesPerBlock;
    auto* primary = reinterpret_cast<char*>(pointers[poolIdx * pointerStride + 0]) + layerOffset;
    void* secondary = nullptr;
    if (pointerStride > 1 && pointers[poolIdx * pointerStride + 1] != 0)
    {
        secondary = reinterpret_cast<char*>(pointers[poolIdx * pointerStride + 1]) + layerOffset;
    }
    int32_t const maxBlocksPerSeq = static_cast<int32_t>(blockOffsets.size(-1));
    auto* offsets = reinterpret_cast<tk::KVBlockArray::DataType*>(blockOffsets.index({poolIdx, seqOffset}).data_ptr());
    return tk::KVBlockArray(batch, maxBlocksPerSeq, tokensPerBlock, bytesPerToken, maxSeqLen, maxSeqLen,
        /*sinkTokenLen=*/0, /*canUseOneMoreBlock=*/false, primary, secondary, offsets);
}

template <typename P>
void setNoRope(P& p)
{
    // The target applies RoPE itself (qkv_gemma_rmsnorm_rope_gate); the op sees learned-absolute positions.
    p.rotary_embedding_dim = 0;
    p.rotary_embedding_base = 10000.f;
    p.rotary_scale_type = tk::RotaryScalingType::kNONE;
    p.rotary_embedding_scale = 1.f;
    p.rotary_embedding_max_positions = 1024;
    p.position_embedding_type = tk::PositionEmbeddingType::kLEARNED_ABSOLUTE;
    p.rotary_vision_start = -1;
    p.rotary_vision_length = -1;
}

} // namespace

int64_t attention_sm120_workspace_bytes(int64_t max_tokens, int64_t max_seqs)
{
    return static_cast<int64_t>(WorkspaceLayout(max_tokens, max_seqs).total);
}

int64_t attention_sm120_semaphore_words(int64_t max_seqs)
{
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    return static_cast<int64_t>(
        kernels::xqaSm120SemaphoreWords(static_cast<uint32_t>(max_seqs), multiProcessorCount(device)));
}

// qkv:                        [tokens, 8192] BF16, Q | K | V, RoPE applied. Not modified.
// output:                     [tokens, 6144] BF16
// sequence_length:            [seqs] int32 device: KV length including this step's tokens
// host_past_key_value_lengths [seqs] int32 host:   the same, on the host
// context_lengths:            [seqs] int32 device: tokens of this step's chunk (contexts)
// host_context_lengths:       [seqs] int32 host
// kv_cache_block_offsets:     [pools, max sequences, 2, max blocks per sequence] int32
// host_kv_cache_pool_pointers [pools, 2] (or [pools, 2, ...]) int64 host
// host_kv_cache_pool_mapping: [layers, 2] int32 host; layer_idx is the row of this layer
// kv_scale_orig_quant / kv_scale_quant_orig: [1] FP32 device
// workspace:                  attention_sm120_workspace_bytes(workspace_max_tokens, workspace_max_seqs) bytes
// semaphores:                 attention_sm120_semaphore_words(workspace_max_seqs) int32 words, zeroed once
void attention_sm120(Tensor const& qkv, Tensor& output, int64_t num_contexts, int64_t num_ctx_tokens,
    Tensor const& sequence_length, Tensor const& host_past_key_value_lengths, Tensor const& context_lengths,
    Tensor const& host_context_lengths, Tensor const& kv_cache_block_offsets, Tensor const& host_kv_cache_pool_pointers,
    Tensor const& host_kv_cache_pool_mapping, int64_t layer_idx, int64_t tokens_per_block, int64_t max_seq_len,
    Tensor const& kv_scale_orig_quant, Tensor const& kv_scale_quant_orig, Tensor& workspace,
    int64_t workspace_max_tokens, int64_t workspace_max_seqs, Tensor& semaphores, double q_scaling,
    bool skip_leading_q_tiles, int64_t shared_prefix_min_batch)
{
    CHECK_TH_CUDA(qkv);
    CHECK_TH_CUDA(output);
    TORCH_CHECK(qkv.dim() == 2 && qkv.size(1) == kQkvWidth && qkv.scalar_type() == at::ScalarType::BFloat16
            && qkv.is_contiguous(),
        "qkv must be a contiguous [tokens, 8192] BF16 tensor");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == qkv.size(0) && output.size(1) == kQWidth
            && output.scalar_type() == at::ScalarType::BFloat16 && output.is_contiguous(),
        "output must be a contiguous [tokens, 6144] BF16 tensor");
    TORCH_CHECK(tokens_per_block == kernels::kXqaSm120TokensPerPage, "the target's XQA kernel is built for ",
        kernels::kXqaSm120TokensPerPage, " tokens per KV page, got ", tokens_per_block);
    TORCH_CHECK(host_past_key_value_lengths.device().is_cpu() && host_context_lengths.device().is_cpu()
            && host_kv_cache_pool_pointers.device().is_cpu() && host_kv_cache_pool_mapping.device().is_cpu(),
        "host_* tensors must be on the host");
    int64_t const numTokens = qkv.size(0);
    int64_t const numSeqs = host_context_lengths.size(0);
    int64_t const numGenerations = numSeqs - num_contexts;
    int64_t const numGenTokens = numTokens - num_ctx_tokens;
    TORCH_CHECK(num_contexts >= 0 && numGenerations >= 0 && num_ctx_tokens >= 0 && numGenTokens == numGenerations,
        "attention_sm120: one token per generation request (tokens ", numTokens, ", context tokens ", num_ctx_tokens,
        ", sequences ", numSeqs, ", contexts ", num_contexts, ")");
    TORCH_CHECK(num_ctx_tokens <= workspace_max_tokens && numSeqs <= workspace_max_seqs,
        "attention_sm120: the step exceeds the workspace's sizing");
    WorkspaceLayout const ws(workspace_max_tokens, workspace_max_seqs);
    TORCH_CHECK(workspace.is_cuda() && static_cast<size_t>(workspace.numel() * workspace.element_size()) >= ws.total,
        "attention_sm120: workspace is smaller than attention_sm120_workspace_bytes");
    TORCH_CHECK(semaphores.is_cuda() && semaphores.scalar_type() == at::ScalarType::Int
            && semaphores.numel() >= attention_sm120_semaphore_words(workspace_max_seqs),
        "attention_sm120: semaphores is smaller than attention_sm120_semaphore_words");

    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(qkv));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    int const sms = multiProcessorCount(qkv.get_device());
    auto* const base = static_cast<char*>(workspace.data_ptr());
    auto* const qkvPtr = static_cast<__nv_bfloat16*>(const_cast<void*>(qkv.data_ptr()));
    auto* const outPtr = static_cast<__nv_bfloat16*>(output.data_ptr());
    auto const* seqLens = sequence_length.data_ptr<int32_t>();
    auto const* ctxLens = context_lengths.data_ptr<int32_t>();
    auto const* kvScaleOrigQuant = kv_scale_orig_quant.data_ptr<float>();
    auto const* kvScaleQuantOrig = kv_scale_quant_orig.data_ptr<float>();
    float const hostBmm1Scale = 1.f / (std::sqrt(static_cast<float>(kHeadSize)) * static_cast<float>(q_scaling));
    int const tpb = static_cast<int>(tokens_per_block);
    int const window = static_cast<int>(max_seq_len);

    if (num_contexts > 0)
    {
        int const batch = static_cast<int>(num_contexts);
        int const tokens = static_cast<int>(num_ctx_tokens);
        int const maxQ = hostMax(host_context_lengths, 0, num_contexts);
        int const maxKv = hostMax(host_past_key_value_lengths, 0, num_contexts);
        auto* const cuQ = reinterpret_cast<int*>(base + ws.cuQ);
        auto* const cuKv = reinterpret_cast<int*>(base + ws.cuKv);
        auto* const tokensInfo = reinterpret_cast<int2*>(base + ws.tokensInfo);
        auto* const bmm1 = reinterpret_cast<float*>(base + ws.bmm1);
        auto* const bmm2 = reinterpret_cast<float*>(base + ws.bmm2);
        auto* const tileCounter = reinterpret_cast<uint32_t*>(base + ws.tileCounter);
        tk::KVBlockArray const kv = kvCacheView(kv_cache_block_offsets, host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping, layer_idx, 0, batch, tpb, window);

        tk::BuildDecoderInfoParams<__nv_bfloat16> info{};
        info.seqQOffsets = cuQ;
        info.seqKVOffsets = cuKv;
        info.precomputedSeqQOffsets = nullptr;
        info.precomputedSeqKVOffsets = nullptr;
        info.seqCpPartialOffsets = nullptr;
        info.cpSize = 1;
        info.packedMaskRowOffsets = nullptr;
        info.paddingOffsets = nullptr;
        info.tokensInfo = tokensInfo;
        info.encoderPaddingOffsets = nullptr;
        info.attentionMask = nullptr;
        info.seqQLengths = ctxLens;
        info.seqKVLengths = seqLens;
        info.batchSize = batch;
        info.maxQSeqLength = maxQ;
        info.maxEncoderQSeqLength = 0;
        info.attentionWindowSize = window;
        info.numTokens = tokens;
        info.removePadding = true;
        info.attentionMaskType = tk::AttentionMaskType::CAUSAL;
        info.fmhaTileCounter = tileCounter;
        info.quantScaleO = nullptr;
        info.dequantScaleQkv = kvScaleQuantOrig;
        info.separateQkvScales = false;
        info.fmhaHostBmm1Scale = hostBmm1Scale;
        info.fmhaBmm1Scale = bmm1;
        info.fmhaBmm2Scale = bmm2;
        info.rotaryEmbeddingScale = 1.f;
        info.rotaryEmbeddingBase = 10000.f;
        info.rotaryEmbeddingDim = 0;
        info.rotaryScalingType = tk::RotaryScalingType::kNONE;
        info.rotaryEmbeddingInvFreq = nullptr;
        info.rotaryEmbeddingInvFreqCache = nullptr;
        info.rotaryEmbeddingCoeffCache = nullptr;
        info.rotaryEmbeddingMaxPositions = 1024;
        tk::invokeBuildDecoderInfo(info, stream);

        tk::QKVPreprocessingParams<__nv_bfloat16, tk::KVBlockArray> pre;
        pre.qkv_input = qkvPtr;
        pre.quantized_qkv_output = nullptr;
        // The kernel writes FP8 Q here when quantized_fp8_output is set; the pointer type is the input type.
        pre.q_output = reinterpret_cast<__nv_bfloat16*>(base + ws.qFp8);
        pre.kv_cache_buffer = kv;
        pre.tokens_info = tokensInfo;
        pre.seq_lens = ctxLens;
        pre.cache_seq_lens = seqLens;
        pre.cu_seq_lens = cuQ;
        pre.cu_kv_seq_lens = cuKv;
        pre.rotary_embedding_inv_freq = nullptr;
        pre.qkv_scale_orig_quant = kvScaleOrigQuant;
        pre.batch_size = batch;
        pre.max_input_seq_len = maxQ;
        pre.max_kv_seq_len = maxKv;
        pre.cyclic_kv_cache_len = window;
        pre.token_num = tokens;
        pre.remove_padding = true;
        pre.head_num = kQHeads;
        pre.kv_head_num = kKvHeads;
        pre.qheads_per_kv_head = kQHeads / kKvHeads;
        pre.size_per_head = kHeadSize;
        pre.cache_type = tk::KvCacheDataType::FP8;
        pre.separate_q_kv_output = true;
        pre.quantized_fp8_output = true;
        pre.generation_phase = false;
        pre.multi_processor_count = sms;
        pre.is_last_chunk = true;
        setNoRope(pre);
        tk::invokeQKVPreprocessing(pre, stream);

        kernels::ContextFmhaSm120Args fmha{};
        fmha.qFp8 = base + ws.qFp8;
        fmha.output = outPtr;
        fmha.cuQSeqlens = cuQ;
        fmha.cuKvSeqlens = cuKv;
        fmha.kvCache = kv;
        fmha.batch = batch;
        fmha.maxQSeqLen = maxQ;
        fmha.slidingWindowSize = window;
        fmha.hostBmm1Scale = hostBmm1Scale;
        fmha.bmm1ScaleDevice = bmm1;
        fmha.bmm2ScaleDevice = bmm2;
        fmha.tileCounter = tileCounter;
        fmha.splitWorkspace = base + ws.split;
        fmha.skipLeadingQTiles = skip_leading_q_tiles;
        fmha.multiProcessorCount = sms;
        kernels::runContextFmhaSm120(fmha, stream);
    }

    if (numGenerations > 0)
    {
        int const batch = static_cast<int>(numGenerations);
        int const maxKv = hostMax(host_past_key_value_lengths, num_contexts, numSeqs);
        tk::KVBlockArray const kv = kvCacheView(kv_cache_block_offsets, host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping, layer_idx, num_contexts, batch, tpb, window);
        auto* const genQ = reinterpret_cast<__nv_bfloat16*>(base + ws.genQ);
        int const* genSeqLens = seqLens + num_contexts;

        tk::QKVPreprocessingParams<__nv_bfloat16, tk::KVBlockArray> pre;
        pre.qkv_input = qkvPtr + num_ctx_tokens * kQkvWidth;
        pre.quantized_qkv_output = nullptr;
        pre.q_output = genQ;
        pre.kv_cache_buffer = kv;
        pre.tokens_info = reinterpret_cast<int2*>(base + ws.tokensInfo); // unused in the generation kernel
        pre.seq_lens = nullptr;
        pre.cache_seq_lens = genSeqLens;
        pre.cu_seq_lens = nullptr;
        pre.qkv_scale_orig_quant = kvScaleOrigQuant;
        pre.batch_size = batch;
        pre.max_input_seq_len = 1;
        pre.max_kv_seq_len = maxKv;
        pre.cyclic_kv_cache_len = window;
        pre.token_num = batch;
        pre.remove_padding = true;
        pre.head_num = kQHeads;
        pre.kv_head_num = kKvHeads;
        pre.qheads_per_kv_head = kQHeads / kKvHeads;
        pre.size_per_head = kHeadSize;
        pre.cache_type = tk::KvCacheDataType::FP8;
        pre.separate_q_kv_output = true;
        pre.quantized_fp8_output = false;
        pre.generation_phase = true;
        pre.multi_processor_count = sms;
        setNoRope(pre);
        tk::invokeQKVPreprocessing(pre, stream);

        kernels::XqaSm120DecodeArgs x{};
        x.output = outPtr + num_ctx_tokens * kQWidth;
        x.q = genQ;
        x.kvPool = kv.mPrimaryPoolPtr;
        x.kvPageList = reinterpret_cast<int32_t const*>(kv.data);
        x.seqLens = reinterpret_cast<uint32_t const*>(genSeqLens);
        x.maxNbPagesPerSeq = static_cast<uint32_t>(kv.mMaxBlocksPerSeq);
        x.batch = static_cast<uint32_t>(batch);
        x.slidingWindowSize = static_cast<uint32_t>(window);
        x.qScale = static_cast<float>(q_scaling);
        x.kvCacheScale = kvScaleQuantOrig;
        x.semaphores = reinterpret_cast<uint32_t*>(semaphores.data_ptr<int32_t>());
        x.scratch = base + ws.xqaScratch;
        x.multiBlock
            = kernels::xqaSm120MultiBlockCount(static_cast<uint32_t>(batch), static_cast<uint32_t>(maxKv), sms);
        x.sharedPrefixMinBatch = static_cast<uint32_t>(shared_prefix_min_batch);
        x.multiProcessorCount = sms;
        x.enablePdl = tensorrt_llm::common::getEnvEnablePDL();
        kernels::xqaSm120Decode(x, stream);
    }
}

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def("attention_sm120_workspace_bytes(int max_tokens, int max_seqs) -> int");
    m.def("attention_sm120_semaphore_words(int max_seqs) -> int");
    m.def(
        "attention_sm120(Tensor qkv, Tensor(a!) output, int num_contexts, int num_ctx_tokens, Tensor sequence_length,"
        " Tensor host_past_key_value_lengths, Tensor context_lengths, Tensor host_context_lengths,"
        " Tensor kv_cache_block_offsets, Tensor host_kv_cache_pool_pointers, Tensor host_kv_cache_pool_mapping,"
        " int layer_idx, int tokens_per_block, int max_seq_len, Tensor kv_scale_orig_quant,"
        " Tensor kv_scale_quant_orig, Tensor(b!) workspace, int workspace_max_tokens, int workspace_max_seqs,"
        " Tensor(c!) semaphores, float q_scaling, bool skip_leading_q_tiles, int shared_prefix_min_batch) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("attention_sm120_workspace_bytes", &TRTLLM_MV2_NS::attention_sm120_workspace_bytes);
    m.impl("attention_sm120_semaphore_words", &TRTLLM_MV2_NS::attention_sm120_semaphore_words);
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("attention_sm120", &TRTLLM_MV2_NS::attention_sm120);
}
