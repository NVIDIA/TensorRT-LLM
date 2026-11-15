/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "xqaDispatcher.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunnerUtils.h"
#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include "tensorrt_llm/kernels/swaDecodeAttention.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace
{

template <typename T>
constexpr inline T divUp(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
constexpr inline T roundUp(T a, T b)
{
    return divUp(a, b) * b;
}

} // namespace

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// TLLM_DECODER_INFO_ONCE_LAYERS=N (default unset = off). The trtllm-gen generation path rebuilds cu_seqlens,
// cu_kv_seqlens and the per-sequence rotary inv_freq (computeSeqAndPaddingOffsets) in every layer, from inputs
// that do not change between the layers of one forward step, into the same workspace bytes. With N > 0, local
// layer k in [1, N) reuses what layer k-1 left there when all of these hold:
//   - layer k-1 was the last attention enqueue on this thread (context, CP and MMHA enqueues invalidate);
//   - every input and output pointer, size and rotary field of the build is identical (DecoderInfoKey).
// N must be the number of target attention layers on this rank (36 for GPT-OSS-120B), so a draft layer
// (local index >= N), whose lengths change between iterations, always rebuilds. Layer 0 always rebuilds.
// Under CUDA graphs the decision is taken at capture: layer 0's build is captured and replayed every step.
struct DecoderInfoKey
{
    cudaStream_t stream;
    void const* outQ;
    void const* outKv;
    void const* outInvFreq;
    void const* inQLengths;
    void const* inKvLengths;
    void const* inInvFreqCache;
    int batchSize;
    int maxQSeqLength;
    int numTokens;
    int rotaryDim;
    int rotaryType;
    int rotaryMaxPositions;
    float rotaryScale;
    float rotaryBase;

    bool operator==(DecoderInfoKey const& o) const
    {
        return stream == o.stream && outQ == o.outQ && outKv == o.outKv && outInvFreq == o.outInvFreq
            && inQLengths == o.inQLengths && inKvLengths == o.inKvLengths && inInvFreqCache == o.inInvFreqCache
            && batchSize == o.batchSize && maxQSeqLength == o.maxQSeqLength && numTokens == o.numTokens
            && rotaryDim == o.rotaryDim && rotaryType == o.rotaryType && rotaryMaxPositions == o.rotaryMaxPositions
            && rotaryScale == o.rotaryScale && rotaryBase == o.rotaryBase;
    }
};

struct DecoderInfoReuseState
{
    bool valid = false;
    int layer = -1;
    DecoderInfoKey key{};
    // Highest layer_idx that called reuseDecoderInfo since the last layer 0 (logged when it changes, to check N).
    int stepMaxLayer = -1;
};

thread_local DecoderInfoReuseState gDecoderInfoReuse;

int decoderInfoOnceLayers()
{
    static int const layers = []
    {
        char const* v = std::getenv("TLLM_DECODER_INFO_ONCE_LAYERS");
        int n = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 0;
        n = n > 0 ? n : 0;
        if (n > 0)
        {
            std::fprintf(stderr,
                "[TRT-LLM] [INFO] TLLM_DECODER_INFO_ONCE_LAYERS=%d active: trtllm-gen generation layers 1..%d "
                "reuse the previous layer's decoder info\n",
                n, n - 1);
        }
        return n;
    }();
    return layers;
}

template <typename T>
DecoderInfoKey makeDecoderInfoKey(BuildDecoderInfoParams<T> const& p, cudaStream_t stream)
{
    DecoderInfoKey k{};
    k.stream = stream;
    k.outQ = p.seqQOffsets;
    k.outKv = p.seqKVOffsets;
    k.outInvFreq = p.rotaryEmbeddingInvFreq;
    k.inQLengths = p.seqQLengths;
    k.inKvLengths = p.seqKVLengths;
    k.inInvFreqCache = p.rotaryEmbeddingInvFreqCache;
    k.batchSize = p.batchSize;
    k.maxQSeqLength = p.maxQSeqLength;
    k.numTokens = p.numTokens;
    k.rotaryDim = p.rotaryEmbeddingDim;
    k.rotaryType = static_cast<int>(p.rotaryScalingType);
    k.rotaryMaxPositions = p.rotaryEmbeddingMaxPositions;
    k.rotaryScale = p.rotaryEmbeddingScale;
    k.rotaryBase = p.rotaryEmbeddingBase;
    return k;
}

// Returns true when this layer may skip invokeBuildDecoderInfo, and records this layer as the last builder or reuser.
template <typename T>
bool reuseDecoderInfo(BuildDecoderInfoParams<T> const& p, int layerIdx, cudaStream_t stream)
{
    int const layers = decoderInfoOnceLayers();
    if (layers <= 0)
    {
        return false;
    }
    DecoderInfoKey const key = makeDecoderInfoKey(p, stream);
    auto& s = gDecoderInfoReuse;
    if (layerIdx == 0)
    {
        // A new forward step. Log the previous step's highest layer_idx (among calls that build decoder info)
        // whenever it differs from the last value logged, so the log shows every distinct value: N with a
        // one-model drafter (its draft layer has index N), N-1 for a target-only forward. Above N is a misconfig.
        static int loggedMax = -1;
        if (s.stepMaxLayer > 0 && s.stepMaxLayer != loggedMax)
        {
            loggedMax = s.stepMaxLayer;
            std::fprintf(stderr,
                "[TRT-LLM] [INFO] TLLM_DECODER_INFO_ONCE_LAYERS=%d: step max layer_idx %d (%s)\n", layers,
                s.stepMaxLayer,
                s.stepMaxLayer == layers       ? "== N: draft layer seen, N matches the target layer count"
                    : s.stepMaxLayer == layers - 1 ? "== N-1: target-only forward"
                    : s.stepMaxLayer > layers      ? "> N: N is below the target layer count"
                                                   : "< N-1: N exceeds the layers seen; check N");
        }
        s.stepMaxLayer = 0;
    }
    else if (layerIdx > s.stepMaxLayer)
    {
        s.stepMaxLayer = layerIdx;
    }
    bool const reuse = s.valid && layerIdx > 0 && layerIdx < layers && s.layer == layerIdx - 1 && s.key == key;
    s.valid = true;
    s.layer = layerIdx;
    s.key = key;
    if (reuse)
    {
        static bool logged = false;
        if (!logged)
        {
            logged = true;
            std::fprintf(stderr,
                "[TRT-LLM] [INFO] TLLM_DECODER_INFO_ONCE_LAYERS: layer %d reuses decoder info "
                "(batch=%d maxQ=%d tokens=%d)\n",
                layerIdx, p.batchSize, p.maxQSeqLength, p.numTokens);
        }
    }
    return reuse;
}

template <typename T, typename KVCacheBuffer>
QKVPreprocessingParams<T, KVCacheBuffer> makeQKVPreprocessingParams(XQAParams const& params,
    XQALaunchParam<KVCacheBuffer> const& launchParams, void* xqa_q_input_ptr, Data_type QDataType,
    KvCacheDataType cache_type, int32_t batch_beam_size, KVCacheBuffer const& kv_cache_buffer,
    KVCacheBuffer const& kv_cache_block_scales_buffer, int32_t const* cu_seqlens, int32_t const* cu_kv_seqlens,
    float const* rotary_inv_freq_buf, int multiProcessorCount)
{
    QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParms;
    memset(&preprocessingParms, 0, sizeof(preprocessingParms));
    // Set parameters.
    preprocessingParms.qkv_input = static_cast<T*>(const_cast<void*>(params.qkv));
    preprocessingParms.q_output = static_cast<T*>(xqa_q_input_ptr);
    preprocessingParms.kv_cache_buffer = kv_cache_buffer;
    preprocessingParms.kv_cache_block_scales_buffer = kv_cache_block_scales_buffer;
    preprocessingParms.qkv_bias = static_cast<T const*>(params.qkv_bias);
    // Prepare values for fmha.
    preprocessingParms.fmha_bmm1_scale = launchParams.bmm1_scale_ptr;
    preprocessingParms.fmha_bmm2_scale = launchParams.bmm2_scale_ptr;
    bool const is_fp8_q_input = (QDataType == DATA_TYPE_E4M3);
    if (params.kv_cache_quant_mode.hasFp8KvCache() || params.kv_cache_quant_mode.hasFp4KvCache())
    {
        preprocessingParms.qkv_scale_quant_orig = params.kv_scale_quant_orig;
    }
    if (params.is_fp8_output)
    {
        preprocessingParms.o_scale_orig_quant = params.fp8_out_scale;
    }
    // Buffers.
    preprocessingParms.logn_scaling = params.logn_scaling_ptr;
    preprocessingParms.seq_lens = params.spec_decoding_generation_lengths;
    preprocessingParms.cache_seq_lens = params.sequence_lengths;
    preprocessingParms.cu_seq_lens = cu_seqlens;
    preprocessingParms.rotary_embedding_inv_freq = rotary_inv_freq_buf;
    preprocessingParms.rotary_coef_cache_buffer = params.rotary_cos_sin;
    preprocessingParms.qkv_scale_orig_quant = params.kv_scale_orig_quant;
    preprocessingParms.spec_decoding_position_offsets
        = params.cross_attention ? nullptr : params.spec_decoding_position_offsets;
    preprocessingParms.mrope_position_deltas = params.mrope_position_deltas;
    preprocessingParms.helix_position_offsets = params.helix_position_offsets;
    preprocessingParms.helix_is_inactive_rank = params.helix_is_inactive_rank;
    // Scalar parameters.
    preprocessingParms.batch_size = int(batch_beam_size);
    preprocessingParms.max_input_seq_len = params.generation_input_length;
    preprocessingParms.max_kv_seq_len = params.max_past_kv_length;
    preprocessingParms.cyclic_kv_cache_len
        = params.cross_attention ? params.max_past_kv_length : params.cyclic_attention_window_size;
    preprocessingParms.sink_token_len = params.cross_attention ? 0 : params.sink_token_length;
    preprocessingParms.token_num = params.total_num_input_tokens;
    preprocessingParms.remove_padding = true;
    preprocessingParms.cross_attention = params.cross_attention;
    preprocessingParms.head_num = params.num_q_heads;
    preprocessingParms.kv_head_num = params.num_kv_heads;
    preprocessingParms.qheads_per_kv_head = params.num_q_heads / params.num_kv_heads;
    preprocessingParms.size_per_head = params.head_size;
    preprocessingParms.fmha_host_bmm1_scale = 1.0f / (sqrtf(params.head_size * 1.0f) * params.q_scaling);
    preprocessingParms.rotary_embedding_dim = params.rotary_embedding_dim;
    preprocessingParms.rotary_embedding_base = params.rotary_embedding_base;
    preprocessingParms.rotary_scale_type = params.rotary_embedding_scale_type;
    preprocessingParms.rotary_embedding_scale = params.rotary_embedding_scale;
    preprocessingParms.rotary_embedding_max_positions = params.rotary_embedding_max_positions;
    preprocessingParms.position_embedding_type = params.position_embedding_type;
    preprocessingParms.position_shift_enabled = params.position_shift_enabled;
    preprocessingParms.cache_type = cache_type;
    preprocessingParms.separate_q_kv_output = true;
    preprocessingParms.quantized_fp8_output = is_fp8_q_input;
    preprocessingParms.generation_phase = true;
    preprocessingParms.multi_processor_count = multiProcessorCount;
    preprocessingParms.rotary_vision_start = params.rotary_vision_start;
    preprocessingParms.rotary_vision_length = params.rotary_vision_length;

    // Cross-attention only.

    preprocessingParms.encoder_seq_lens = params.encoder_input_lengths;

    return preprocessingParms;
}

// TLLM_FMHA_SWA_DECODE_KERNEL=1 (exactly "1"). Read once per process.
bool swaDecodeKernelFromEnv()
{
    static bool const value = []()
    {
        char const* env = std::getenv("TLLM_FMHA_SWA_DECODE_KERNEL");
        return env != nullptr && std::strcmp(env, "1") == 0;
    }();
    return value;
}

// Opt-in (TLLM_FMHA_SWA_DECODE_KERNEL=1): run a short sliding-window generation attention (window + query tokens - 1
// <= 160 keys, the GPT-OSS window-128 layers) on the dedicated kernel (swaDecodeAttention.h) instead of the
// trtllm-gen FMHA, after the same QKV preprocessing. Same inputs (E4M3 Q, the paged E4M3 cache, the device scales,
// the sinks) and the same mask; the kernel runs one 128-thread CTA per (request, KV head), so several short windows
// are in flight per SM. Returns false, leaving the stock path to run, unless every condition holds.
template <typename KVCacheBuffer>
bool runSwaDecodeIfRequested(XqaFixedParams const& fixed, Data_type qDataType, XQAParams const& params,
    KVCacheBuffer const& kvCacheBuffer, float const* bmm1Scale, float const* bmm2Scale, void const* q,
    int const* cuSeqLensQ)
{
    if (!swaDecodeKernelFromEnv())
    {
        return false;
    }
    int const window = params.cyclic_attention_window_size;
    int const headsPerKv = params.num_kv_heads > 0 ? params.num_q_heads / params.num_kv_heads : 0;
    // Only short sliding windows are candidates; every other layer keeps the stock path silently.
    if (window <= 0 || window + params.generation_input_length - 1 > 160)
    {
        return false;
    }
    char const* reject = nullptr;
    if constexpr (!std::is_same_v<KVCacheBuffer, KVBlockArray>)
    {
        reject = "contiguous KV cache";
    }
    else if (qDataType != DATA_TYPE_E4M3 || fixed.kvDataType != DATA_TYPE_E4M3)
    {
        reject = "Q or KV is not E4M3";
    }
    else if (fixed.outputDataType != DATA_TYPE_BF16 || params.is_fp8_output || params.output_sf != nullptr)
    {
        reject = "output is not BF16";
    }
    else if (fixed.isMLA || fixed.hasAlibi || fixed.useTllmGenSparseAttention || params.use_sparse_attention_gen_paged)
    {
        reject = "MLA, ALiBi or sparse attention";
    }
    else if (params.beam_width != 1 || params.cross_attention || (params.is_spec_dec_tree && params.multi_query_tokens))
    {
        reject = "beam search, cross attention or a spec-dec tree";
    }
    else if (params.softmax_stats != nullptr || params.skip_softmax_threshold_scale_factor != 0.f
        || params.helix_position_offsets != nullptr || params.helix_is_inactive_rank != nullptr)
    {
        reject = "softmax stats, skip-softmax or helix";
    }
    else if (params.sink_token_length != 0 || params.position_shift_enabled
        || (params.chunked_attention_size != INT_MAX && params.chunked_attention_size > 0))
    {
        reject = "StreamingLLM sink tokens or chunked attention";
    }
    else if (bmm1Scale == nullptr || bmm2Scale == nullptr)
    {
        reject = "no device BMM scales";
    }
    else if (params.num_kv_heads <= 0 || headsPerKv * params.num_kv_heads != params.num_q_heads
        || !swaDecodeAttentionSupported(params.head_size, headsPerKv, params.generation_input_length, window))
    {
        reject = "unsupported shape";
    }
    if (reject != nullptr)
    {
        static std::atomic<bool> loggedSkip{false};
        if (!loggedSkip.exchange(true))
        {
            std::fprintf(stderr,
                "[TRT-LLM] [INFO] TLLM_FMHA_SWA_DECODE_KERNEL=1 skipped (stock kept): %s (layer=%d window=%d "
                "maxSeqLenQ=%d headDim=%d headsPerKv=%d)\n",
                reject, params.layer_idx, window, params.generation_input_length, params.head_size, headsPerKv);
        }
        return false;
    }
    if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
    {
        SwaDecodeAttentionParams p{};
        p.q = q;
        p.out = params.output;
        p.kv = kvCacheBuffer;
        p.seqLensKv = params.sequence_lengths;
        p.cuSeqLensQ = cuSeqLensQ;
        p.scaleSoftmaxLog2 = bmm1Scale + kIdxScaleSoftmaxLog2Ptr;
        p.outputScale = bmm2Scale;
        p.sinks = params.attention_sinks;
        p.batch = static_cast<int>(params.batch_size);
        p.numHeadsQ = params.num_q_heads;
        p.numHeadsKv = params.num_kv_heads;
        p.maxSeqLenQ = params.generation_input_length;
        p.window = window;
        static std::atomic<bool> loggedActive{false};
        if (!loggedActive.exchange(true))
        {
            std::fprintf(stderr,
                "[TRT-LLM] [INFO] TLLM_FMHA_SWA_DECODE_KERNEL=1 active: sliding-window generation attention on "
                "swaDecodeFmhaKernel (layer=%d window=%d batch=%d maxSeqLenQ=%d headsPerKv=%d grid %dx%d, sinks=%d, "
                "cuSeqLensQ=%d)\n",
                params.layer_idx, window, p.batch, p.maxSeqLenQ, headsPerKv, p.numHeadsKv, p.batch,
                p.sinks != nullptr ? 1 : 0, cuSeqLensQ != nullptr ? 1 : 0);
        }
        launchSwaDecodeAttention(p, params.stream);
        return true;
    }
    return false;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

XqaDispatcher::XqaDispatcher(XqaFixedParams fixedParams)
    : mFixedParams(fixedParams)
    , mQDataType(mFixedParams.inputDataType)
    , mUseTllmGen(tensorrt_llm::common::isSM100Family())
    , mMultiProcessorCount(getMultiProcessorCount())
{
    if (mUseTllmGen)
    {
        // The preprocessing kernel will convert Q from inputDataType to fp8 if the kv cache dtype e4m3 or e2m1,
        // as both the NVFP4 KV kernels and FP8 KV kernels uses FP8 input for Q.
        mQDataType = (mFixedParams.kvDataType == DATA_TYPE_E4M3 || mFixedParams.kvDataType == DATA_TYPE_E2M1)
            ? DATA_TYPE_E4M3
            : mFixedParams.inputDataType;
        mTllmGenFMHARunner.reset(new TllmGenFmhaRunner(
            mQDataType, mFixedParams.kvDataType, mFixedParams.kvDataType, mFixedParams.outputDataType));
    }
    else
    {
        mDecoderXqaRunner.reset(new DecoderXQARunner(mFixedParams.inputDataType, mFixedParams.numQHeads,
            mFixedParams.numKvHeads, mFixedParams.headSize, mFixedParams.multiBlockMode));
    }
}

void XqaDispatcher::prepare(XQAParams const& params)
{
    if (!mUseTllmGen)
    {
        if (mDecoderXqaRunner->shouldUse(params, /*forConfigurePlugin=*/true))
        {
            mDecoderXqaRunner->prepare(params);
        }
    }
}

int XqaDispatcher::getWorkspaceAlignment()
{
    constexpr size_t kXQA_OUT_ELEM_SIZE = 2; // fp16 or bf16.
    constexpr int kMaxBeamWidth = 4;
    int group_size = mFixedParams.numQHeads / mFixedParams.numKvHeads;
    int32_t const multi_block_workspace_alignment
        = roundUp<int32_t>(kXQA_OUT_ELEM_SIZE * kMaxBeamWidth * group_size * mFixedParams.headSize, 128);
    return mFixedParams.multiBlockMode ? multi_block_workspace_alignment : 128;
}

size_t XqaDispatcher::getWorkspaceSize(int max_num_tokens)
{
    // buffer for RoPE / output quantization.
    constexpr size_t kXQA_OUT_ELEM_SIZE = 2; // fp16 or bf16.
    constexpr int kMaxBeamWidth = 4;
    size_t workspace_size = roundUp<size_t>(
        kXQA_OUT_ELEM_SIZE * mFixedParams.headSize * mFixedParams.numQHeads * max_num_tokens, 128); // rope
    // output conversion.
    workspace_size = roundUp<size_t>(
        workspace_size + kXQA_OUT_ELEM_SIZE * mFixedParams.headSize * mFixedParams.numQHeads * max_num_tokens, 128);
    workspace_size = roundUp<size_t>(workspace_size, 128) + xqaMlaCgaXBufSize * max_num_tokens;
    if (mFixedParams.multiBlockMode)
    {
        size_t workspaces[4];
        size_t const nbSubSeq = getXqaMaxNumSubSeq(mFixedParams.isMLA);
        size_t const nbSeq = nbSubSeq / 2;
        int group_size = mFixedParams.numQHeads / mFixedParams.numKvHeads;
        workspaces[0] = sizeof(uint32_t) * nbSeq;                           // semaphores
        workspaces[1] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq; // rowMax
        workspaces[2] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq; // rowSum
        size_t const multi_block_workspace_alignment
            = roundUp<size_t>(kXQA_OUT_ELEM_SIZE * kMaxBeamWidth * group_size * mFixedParams.headSize, 128);
        workspaces[3] = multi_block_workspace_alignment * nbSubSeq;
        workspace_size = roundUp<size_t>(workspace_size, multi_block_workspace_alignment)
            + roundUp(workspaces[0], multi_block_workspace_alignment)
            + roundUp(workspaces[1], multi_block_workspace_alignment)
            + roundUp(workspaces[2], multi_block_workspace_alignment)
            + roundUp(workspaces[3], multi_block_workspace_alignment)
            + multi_block_workspace_alignment; // extra space reserved for alignment
    }
    return workspace_size;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define SHOULD_NOT_USE(message)                                                                                        \
    TLLM_LOG_DEBUG(message);                                                                                           \
    return false;

bool XqaDispatcher::shouldUse(XQAParams const& params)
{
    if (mUseTllmGen)
    {
        // Fall-back to MMHA for some unsupported cases.
        if (params.beam_width > 1)
        {
            SHOULD_NOT_USE("Fallback to MMHA as beam searching is not supported by TRTLLM-GEN kernels.")
        }
        if (params.position_shift_enabled || params.sink_token_length > 0)
        {
            SHOULD_NOT_USE("Fallback to MMHA as streamingLLM is not supported by TRTLLM-GEN kernels.")
        }
        if (params.unidirectional != 1)
        {
            SHOULD_NOT_USE("Fallback to MMHA as unidirectional is not supported by TRTLLM-GEN kernels.");
        }
        if (params.cross_attention && !params.paged_kv_cache)
        {
            SHOULD_NOT_USE(
                "Fallback to MMHA as cross attention without paged KV Cache is not supported by TRTLLM-GEN kernels.");
        }
        if (params.paged_kv_cache && params.tokens_per_block < 8)
        {
            SHOULD_NOT_USE("Fallback to MMHA as tokens_per_block < 8 is not supported by TRTLLM-GEN kernels.");
        }
        if (params.cyclic_attention_window_size != params.max_attention_window_size)
        {
            SHOULD_NOT_USE(
                "Fallback to MMHA as variable attention_window_size is not supported by TRTLLM-GEN kernels.");
        }
        if ((float(params.num_q_heads) / float(params.num_kv_heads)) > 32)
        {
            SHOULD_NOT_USE("Fallback to MMHA as num_q_heads per kv_head > 32 is not supported by TRTLLM-GEN kernels.");
        }

        return true;
    }
    return mDecoderXqaRunner->shouldUse(params, /*forConfigurePlugin=*/false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool XqaDispatcher::isSupported()
{
    if (mUseTllmGen)
    {
        // TODO (perkzz): add the support of fp8-kv fp16/bf16-mma fmha.
        if (mQDataType != mFixedParams.mathDataType)
        {
            TLLM_LOG_WARNING("Unsupported data type combination.");
            return false;
        }
        if (mFixedParams.isSpecDecoding && mFixedParams.isMLA)
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support tree-based speculative decoding with MLA.");
            return false;
        }
        if (mFixedParams.hasAlibi)
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support ALiBi.");
            return false;
        }

        // Create TllmGenFmhaRunnerParams.
        TllmGenFmhaRunnerParams tllmRunnerParams;
        memset(&tllmRunnerParams, 0, sizeof(tllmRunnerParams));
        tllmRunnerParams.mQkvLayout = mFixedParams.isPagedKv ? QkvLayout::PagedKv : QkvLayout::ContiguousKv;
        tllmRunnerParams.mMaskType
            = mFixedParams.isSpecDecoding ? TrtllmGenAttentionMaskType::Custom : TrtllmGenAttentionMaskType::Causal;
        tllmRunnerParams.mIsSpecDecTree = mFixedParams.isSpecDecoding;
        tllmRunnerParams.mSpecDecodingTargetMaxGenLen = mFixedParams.specDecodingTargetMaxGenLen;
        tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
        tllmRunnerParams.mTileScheduler = TileScheduler::Static;
        tllmRunnerParams.mMultiCtasKvMode = true;
        // Assume same head size for Qk and V here.
        tllmRunnerParams.mHeadDimQk = mFixedParams.headSize;
        tllmRunnerParams.mHeadDimV = mFixedParams.headSize;
        tllmRunnerParams.mNumHeadsQ = mFixedParams.numQHeads;
        tllmRunnerParams.mNumHeadsKv = mFixedParams.numKvHeads;
        tllmRunnerParams.mNumHeadsQPerKv = mFixedParams.numQHeads / mFixedParams.numKvHeads;
        // Align with ExportCubin: only PagedKv kernels are built with numTokensPerPage > 0.
        // ContiguousKv must use 0 so hash matches registered cubins.
        tllmRunnerParams.mNumTokensPerPage = mFixedParams.isPagedKv ? mFixedParams.numTokensPerBlock : 0;
        // Set the chunked attention size and sliding window size to INT_MAX to disable them when checking if
        // the kernel is supported.
        tllmRunnerParams.mChunkedAttentionSize = INT_MAX;
        tllmRunnerParams.mAttentionWindowSize = INT_MAX;
        // Problem size fields used by FMHA kernel selection (computeNumCtas). Must be non-zero to avoid
        // integer divide-by-zero when mMultiCtasKvMode is true. Actual launch uses real sizes from runImpl.
        tllmRunnerParams.mBatchSize = 1;
        tllmRunnerParams.mMaxSeqLenQ = 1;
        tllmRunnerParams.mMaxSeqLenKv = 1;
        tllmRunnerParams.mMultiProcessorCount = mMultiProcessorCount;

        // Sparse MQA/GQA uses trtllm-gen sparse kernel.
        if (mFixedParams.useTllmGenSparseAttention)
        {
            tllmRunnerParams.mSparseAttention = SparseType::StaticTokenSparse;
            tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
        }

        // Check if it is supported or not.
        auto [isSupported, info] = mTllmGenFMHARunner->isSupportedWithInfo(tllmRunnerParams);

        if (!isSupported)
        {
            TLLM_LOG_WARNING("TRTLLLM-Gen kernels are not selected: " + info);
            return false;
        }
    }
    else
    {
        if (!(mFixedParams.inputDataType == DATA_TYPE_FP16 || mFixedParams.inputDataType == DATA_TYPE_BF16))
        {
            TLLM_LOG_WARNING("Unsupported datatype.");
            return false;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename KVCacheBuffer>
void XqaDispatcher::runImpl(
    XQAParams params, KVCacheBuffer const& kv_cache_buffer, KVCacheBuffer const& kv_cache_block_scales_buffer)
{
    if (mUseTllmGen)
    {
        TLLM_LOG_DEBUG("Running TRTLLM-GEN generation kernel.");
        TLLM_CHECK_WITH_INFO(mTllmGenFMHARunner.get(), "mTllmGenFMHARunner not initialized.");

        int num_q_heads = params.num_q_heads;
        int num_kv_heads = params.num_kv_heads;
        TLLM_CHECK_WITH_INFO(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
        unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
        unsigned int beam_width = params.beam_width;
        unsigned int batch_beam_size = params.batch_size * beam_width;

        KvCacheDataType cache_type = cacheTypeFromQuantMode(params.kv_cache_quant_mode);

        XQALaunchParam<KVCacheBuffer> launchParams;
        void* inputScratch = nullptr;
        buildXQALaunchParams(launchParams, inputScratch, /*needOutputCvt*/ false, params, kv_cache_buffer);

        // Build cu_seqlens, padding_offset, and rotary inv freq tensors
        BuildDecoderInfoParams<T> decoder_params{};
        decoder_params.seqQOffsets = launchParams.cu_seq_lens;
        decoder_params.seqKVOffsets = launchParams.cu_kv_seq_lens;
        decoder_params.seqQLengths = params.spec_decoding_generation_lengths;
        decoder_params.seqKVLengths = params.cross_attention ? params.encoder_input_lengths : params.sequence_lengths;
        decoder_params.batchSize = static_cast<int>(batch_beam_size);
        decoder_params.maxQSeqLength = params.generation_input_length;
        decoder_params.numTokens = params.total_num_input_tokens;
        decoder_params.removePadding = true;
        TLLM_CHECK_WITH_INFO(!params.multi_query_tokens || params.spec_decoding_generation_lengths != nullptr,
            "Spec_decoding_generation_lengths must be provided.");
        // Rotary embedding inv_freq buffer.
        decoder_params.rotaryEmbeddingScale = params.rotary_embedding_scale;
        decoder_params.rotaryEmbeddingBase = params.rotary_embedding_base;
        decoder_params.rotaryEmbeddingDim = params.rotary_embedding_dim;
        decoder_params.rotaryScalingType = params.rotary_embedding_scale_type;
        decoder_params.rotaryEmbeddingInvFreq = launchParams.rotary_inv_freq_buf;
        decoder_params.rotaryEmbeddingInvFreqCache = params.rotary_embedding_inv_freq_cache;
        decoder_params.rotaryEmbeddingMaxPositions = params.rotary_embedding_max_positions;

        // The rotary_embedding_inv_freq_cache for QKVPreprocessing.
        // Use the params.rotary_embedding_inv_freq_cache input when the buildDecoderInfoKernel is skipped.
        float const* rotary_inv_freq_buf = params.rotary_embedding_inv_freq_cache;
        // Use the nullptr for cu_seqlens when it is not computed.
        int const* cu_seqlens{nullptr};
        int const* cu_kv_seqlens{nullptr};
        if (decoder_params.isBuildDecoderInfoKernelNeeded())
        {
            rotary_inv_freq_buf = launchParams.rotary_inv_freq_buf;
            cu_seqlens = launchParams.cu_seq_lens;
            cu_kv_seqlens = launchParams.cu_kv_seq_lens;
            // TLLM_DECODER_INFO_ONCE_LAYERS: the previous layer built identical values into these buffers.
            if (!reuseDecoderInfo(decoder_params, params.layer_idx, params.stream))
            {
                invokeBuildDecoderInfo(decoder_params, params.stream);
                sync_check_cuda_error(params.stream);
            }
        }
        else
        {
            invalidateXqaDecoderInfoReuse();
        }

        // IDEA: Store rotary_processed Q buffer to output buffer.
        // NOTE: MHA kernels should read kv cache that has already been appended with new tokens' kv cache.
        void* xqa_q_input_ptr = inputScratch;
        // The preprocessing kernel that applies RoPE and updates kv cache.

        auto preprocessingParms = makeQKVPreprocessingParams<T, KVCacheBuffer>(params, launchParams, xqa_q_input_ptr,
            mQDataType, cache_type, batch_beam_size, kv_cache_buffer, kv_cache_block_scales_buffer, cu_seqlens,
            cu_kv_seqlens, rotary_inv_freq_buf, mMultiProcessorCount);

        invokeQKVPreprocessing<T, KVCacheBuffer>(preprocessingParms, params.stream);
        sync_check_cuda_error(params.stream);

        // TLLM_FMHA_SWA_DECODE_KERNEL=1: short sliding windows run on the dedicated kernel instead of the FMHA below.
        if (runSwaDecodeIfRequested(mFixedParams, mQDataType, params, kv_cache_buffer, launchParams.bmm1_scale_ptr,
                launchParams.bmm2_scale_ptr, xqa_q_input_ptr, cu_seqlens))
        {
            return;
        }

        // Build runner parameters.
        TllmGenFmhaRunnerParams tllmRunnerParams;
        memset(&tllmRunnerParams, 0, sizeof(tllmRunnerParams));

        // Parameters to select kernels.
        tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
        tllmRunnerParams.mMultiCtasKvMode = params.multi_block_mode;
        // Note that the tileScheduler and multiCtasKvMode will be automatically tuned when using multi_block mode.
        // Otherwise, always enable the persistent scheduler for better performance.
        tllmRunnerParams.mTileScheduler = params.multi_block_mode ? TileScheduler::Static : TileScheduler::Persistent;

        // The sequence lengths for K/V.
        tllmRunnerParams.seqLensKvPtr = params.cross_attention ? params.encoder_input_lengths : params.sequence_lengths;

        // Q buffer.
        tllmRunnerParams.qPtr = xqa_q_input_ptr;

        // Use block sparse attention.
        tllmRunnerParams.mUseBlockSparseAttention = false;

        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            // Paged KV
            tllmRunnerParams.mQkvLayout = QkvLayout::PagedKv;
            tllmRunnerParams.kvPtr = kv_cache_buffer.mPrimaryPoolPtr;
            tllmRunnerParams.kvSfPtr = kv_cache_block_scales_buffer.mPrimaryPoolPtr;
            tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(kv_cache_buffer.data);
            tllmRunnerParams.mMaxNumPagesPerSeqKv = kv_cache_buffer.mMaxBlocksPerSeq;
            tllmRunnerParams.mNumTokensPerPage = kv_cache_buffer.mTokensPerBlock;

            // Gather kv page offsets for block sparse attention (with offsets).
            if (params.use_sparse_attention_gen_paged)
            {
                invokeGatherKvPageOffsets(reinterpret_cast<int32_t*>(launchParams.sparse_kv_block_offsets),
                    launchParams.sparse_seq_lengths, reinterpret_cast<int32_t const*>(kv_cache_buffer.data),
                    params.sequence_lengths, params.sparse_params, batch_beam_size, num_kv_heads,
                    kv_cache_buffer.mTokensPerBlock, kv_cache_buffer.mMaxBlocksPerSeq, params.stream);
                sync_check_cuda_error(params.stream);
                tllmRunnerParams.seqLensKvPtr = launchParams.sparse_seq_lengths;
                tllmRunnerParams.kvPageIdxPtr
                    = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(launchParams.sparse_kv_block_offsets);
                tllmRunnerParams.mUseBlockSparseAttention = true;
            }
            // Sparse MQA/GQA attention: use trtllm-gen sparse kernel.
            else if (mFixedParams.useTllmGenSparseAttention)
            {
                tllmRunnerParams.mSparseAttention = SparseType::StaticTokenSparse;
                tllmRunnerParams.mSparseTopK = params.sparse_params.num_sparse_topk;
                tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<int const*>(params.sparse_params.sparse_attn_indices);
                tllmRunnerParams.kvPtr = params.sparse_params.sparse_kv_cache_pool;
            }
        }
        else
        {
            TLLM_CHECK_WITH_INFO(!(params.use_sparse_attention_gen_paged || mFixedParams.useTllmGenSparseAttention),
                "Sparse attention is not supported for KVLinearBuffer.");
            static_assert(std::is_same_v<KVCacheBuffer, KVLinearBuffer>);
            // Contiguous KV
            tllmRunnerParams.mQkvLayout = QkvLayout::ContiguousKv;
            tllmRunnerParams.kvPtr = kv_cache_buffer.data;
        }

        // The partial buffers' pointers when the multiCtasKv mode is enabled.
        tllmRunnerParams.multiCtasKvCounterPtr = launchParams.semaphores;
        tllmRunnerParams.multiCtasKvScratchPtr = launchParams.scratch;
        // launchParams.scratch is followed by at least the multi-block partial-output region that
        // getWorkspaceSize reserves (nbSubSeq rows of multi_block_workspace_alignment bytes; the rowMax, rowSum and
        // semaphore regions it also reserves are not counted). 0 keeps the FMHA runner's default layout.
        if (mFixedParams.multiBlockMode)
        {
            constexpr size_t kXQA_OUT_ELEM_SIZE = 2; // fp16 or bf16.
            constexpr size_t kMaxBeamWidth = 4;
            size_t const groupSize = static_cast<size_t>(mFixedParams.numQHeads / mFixedParams.numKvHeads);
            size_t const multiBlockWorkspaceAlignment
                = roundUp<size_t>(kXQA_OUT_ELEM_SIZE * kMaxBeamWidth * groupSize * mFixedParams.headSize, 128);
            tllmRunnerParams.mMultiCtasKvScratchBytes
                = multiBlockWorkspaceAlignment * static_cast<size_t>(getXqaMaxNumSubSeq(mFixedParams.isMLA));
        }

        tllmRunnerParams.attentionSinksPtr = params.attention_sinks;
        tllmRunnerParams.cumSeqLensQPtr = cu_seqlens;
        tllmRunnerParams.cumSeqLensKvPtr = reinterpret_cast<int const*>(launchParams.cu_kv_seq_lens);
        // Attention scales device pointers (only fp8 kernels need to load scales from the device memory).
        if (mQDataType == DATA_TYPE_E4M3)
        {
            tllmRunnerParams.outputScalePtr = reinterpret_cast<float const*>(launchParams.bmm2_scale_ptr);
            tllmRunnerParams.scaleSoftmaxLog2Ptr = launchParams.bmm1_scale_ptr
                ? reinterpret_cast<float const*>(launchParams.bmm1_scale_ptr + kIdxScaleSoftmaxLog2Ptr)
                : nullptr;
        }
        tllmRunnerParams.oSfScalePtr = params.fp4_out_sf_scale;

        tllmRunnerParams.oPtr = params.output;
        tllmRunnerParams.oSfPtr = params.output_sf;
        tllmRunnerParams.mHeadDimQk = params.head_size;
        tllmRunnerParams.mHeadDimV = params.head_size;
        tllmRunnerParams.mNumHeadsQ = params.num_q_heads;
        tllmRunnerParams.mNumHeadsKv = params.num_kv_heads;
        tllmRunnerParams.mNumHeadsQPerKv = num_q_heads_over_kv;
        tllmRunnerParams.mBatchSize = params.batch_size;
        // It is used to construct contiguous kv cache TMA descriptors.
        tllmRunnerParams.mMaxSeqLenCacheKv = params.max_attention_window_size;
        tllmRunnerParams.mMaxSeqLenQ = params.generation_input_length;
        bool const isSpecDecTree = params.is_spec_dec_tree && params.multi_query_tokens;
        if (isSpecDecTree)
        {
            TLLM_CHECK_WITH_INFO(params.spec_decoding_max_generation_length > 0,
                "spec_decoding_max_generation_length must be positive for spec-dec tree.");
            tllmRunnerParams.mMaxSeqLenQ
                = std::min(tllmRunnerParams.mMaxSeqLenQ, params.spec_decoding_max_generation_length);
            tllmRunnerParams.mMaxSeqLenKv = params.max_past_kv_length + tllmRunnerParams.mMaxSeqLenQ;
        }
        else
        {
            tllmRunnerParams.mMaxSeqLenKv = params.max_past_kv_length;
        }
        tllmRunnerParams.mJITWarmup = params.trtllm_gen_jit_warmup;
        tllmRunnerParams.mJITWarmupMaxNumRequests = params.trtllm_gen_jit_warmup_max_num_requests;
        tllmRunnerParams.mJITWarmupMaxSeqLenQ = params.trtllm_gen_jit_warmup_max_seq_len_q;
        tllmRunnerParams.mJITWarmupMaxSeqLenKv = params.trtllm_gen_jit_warmup_max_seq_len_kv;
        tllmRunnerParams.mSumOfSeqLensQ = int(params.batch_size * beam_width * tllmRunnerParams.mMaxSeqLenQ);
        // The sliding window attention size.
        tllmRunnerParams.mAttentionWindowSize = params.cyclic_attention_window_size;
        // The chunked attention size.
        // The generation-phase chunked attention is disabled for now.
        tllmRunnerParams.mChunkedAttentionSize = params.chunked_attention_size;
        // Not used in the generation kernels as contiguous_kv or paged_kv layouts are used.
        tllmRunnerParams.mSumOfSeqLensKv = int(params.batch_size * beam_width * tllmRunnerParams.mMaxSeqLenKv);
        tllmRunnerParams.mScaleQ = params.q_scaling;
        // Set it to INT_MAX as the kv cache pageOffsets will ensure that there is no out-of-bounds access.
        tllmRunnerParams.mNumPagesInMemPool = INT_MAX;
        tllmRunnerParams.mMultiProcessorCount = mMultiProcessorCount;
        tllmRunnerParams.stream = params.stream;
        tllmRunnerParams.mSfStartTokenIdx = params.start_token_idx_sf;
        tllmRunnerParams.mIsSpecDecTree = params.is_spec_dec_tree && params.multi_query_tokens;
        tllmRunnerParams.mMaskType
            = tllmRunnerParams.mIsSpecDecTree ? TrtllmGenAttentionMaskType::Custom : TrtllmGenAttentionMaskType::Causal;
        tllmRunnerParams.mLayerIdx = params.layer_idx;
        tllmRunnerParams.seqLensQPtr = params.spec_decoding_generation_lengths;
        tllmRunnerParams.generalPackedCustoMaskPtr = params.spec_decoding_packed_mask;
        tllmRunnerParams.mPackedMaskMaxSeqLenQ = params.spec_decoding_max_generation_length;
        tllmRunnerParams.mSpecDecodingTargetMaxGenLen = mFixedParams.specDecodingTargetMaxGenLen;
        tllmRunnerParams.mForcePrepareSpecDecTreeMask = params.force_prepare_spec_dec_tree_mask;
        tllmRunnerParams.customMaskPtr = params.spec_decoding_bl_tree_mask;
        tllmRunnerParams.customMaskOffsetsPtr = params.spec_decoding_bl_tree_mask_offset;
        tllmRunnerParams.firstSparseMaskOffsetsKvPtr = params.spec_bl_tree_first_sparse_mask_offset_kv;
        tllmRunnerParams.mSkipSoftmaxThresholdScaleFactor = params.skip_softmax_threshold_scale_factor;
        tllmRunnerParams.softmaxStatsPtr = params.softmax_stats;
        mTllmGenFMHARunner->run(tllmRunnerParams);
    }
    else
    {
        invalidateXqaDecoderInfoReuse();
        mDecoderXqaRunner->template dispatch<KVCacheBuffer>(params, kv_cache_buffer, params.stream);
    }
}

void invalidateXqaDecoderInfoReuse()
{
    gDecoderInfoReuse.valid = false;
}

void XqaDispatcher::run(
    XQAParams const& params, KVLinearBuffer const& kv_cache_buffer, KVLinearBuffer const& kv_cache_block_scales_buffer)
{
    TLLM_CHECK_WITH_INFO((mFixedParams.inputDataType == DATA_TYPE_FP16 || mFixedParams.inputDataType == DATA_TYPE_BF16),
        "The input Qkv tensor must be fp16/bf16.");
    if (mFixedParams.inputDataType == DATA_TYPE_FP16)
    {
        this->runImpl<__half, KVLinearBuffer>(params, kv_cache_buffer, kv_cache_block_scales_buffer);
    }
    else
    {
        this->runImpl<__nv_bfloat16, KVLinearBuffer>(params, kv_cache_buffer, kv_cache_block_scales_buffer);
    }
}

void XqaDispatcher::run(
    XQAParams const& params, KVBlockArray const& kv_cache_buffer, KVBlockArray const& kv_cache_block_scales_buffer)
{
    TLLM_CHECK_WITH_INFO((mFixedParams.inputDataType == DATA_TYPE_FP16 || mFixedParams.inputDataType == DATA_TYPE_BF16),
        "The input Qkv tensor must be fp16/bf16.");
    if (mFixedParams.inputDataType == DATA_TYPE_FP16)
    {
        this->runImpl<__half, KVBlockArray>(params, kv_cache_buffer, kv_cache_block_scales_buffer);
    }
    else
    {
        this->runImpl<__nv_bfloat16, KVBlockArray>(params, kv_cache_buffer, kv_cache_block_scales_buffer);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels

TRTLLM_NAMESPACE_END
