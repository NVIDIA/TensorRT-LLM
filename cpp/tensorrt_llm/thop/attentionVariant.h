/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/thop/attentionFeature.h"

#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

/// The shared algorithm of a TensorRT-LLM attention op. `Derived` supplies five hooks:
/// `initContext()` and `initGeneration()`, run once before that phase's first launch;
/// `prepareInputs(p, isGen)`, `getRequiredWorkspaceSize(p, isGen)`, `enqueueContext(p)`
/// and `enqueueGeneration(p)`, run every call. Features are private bases; the folds
/// below call their generic hooks, and `applyFeatures()` calls whichever `apply`
/// overload each feature has for a given kernel struct.
template <typename Derived, AttentionFeature... Features>
class TrtllmAttention : public AttentionBase, private Features...
{
public:
    /// Derives the device constants and the plain layer geometry. Does no CUDA work:
    /// runners and the cuBLAS handle are created on the first call of each phase, which
    /// keeps construction legal while a stream is capturing.
    explicit TrtllmAttention(AttentionBase const& key)
        : AttentionBase(key)
        , Features(key)...
    {
        mSM = tensorrt_llm::common::getSMVersion();
        mUseTllmGen = (mSM >= 100) && (mSM != 120) && (mSM != 121);
        mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
        mMaxSharedMemoryPerBlockOptin = tensorrt_llm::common::getMaxSharedMemoryPerBlockOptin();
        mNumKVHeads = static_cast<int>(num_kv_heads);
        mHeadSize = static_cast<int>(head_size);
        // One rank per attention op: neither tensor nor context parallelism applies here.
        mNumAttnHeads = static_cast<int>(num_heads);
        mNumAttnKVHeads = mNumKVHeads;
        mFMHAForceFP32Acc = type == tensorrt_llm::DataType::kBF16;
    }

    void runContext(FmhaParams& p) final
    {
        prepare(p, /*isGen=*/false);
        derived().enqueueContext(p);
    }

    void runGeneration(FmhaParams& p) final
    {
        prepare(p, /*isGen=*/true);
        derived().enqueueGeneration(p);
    }

protected:
    [[nodiscard]] Derived& derived()
    {
        return static_cast<Derived&>(*this);
    }

    [[nodiscard]] Derived const& derived() const
    {
        return static_cast<Derived const&>(*this);
    }

    template <AttentionFeature F>
    [[nodiscard]] F const& feature() const
    {
        static_assert((std::is_same_v<F, Features> || ...), "feature not in this op's feature list");
        return static_cast<F const&>(*this);
    }

    void applyFeatures(auto& params) const
    {
        (
            [&]<typename F>(F const& f)
            {
                if constexpr (requires { f.apply(params); })
                {
                    f.apply(params);
                }
            }(static_cast<Features const&>(*this)),
            ...);
    }

    /// Runner construction, scratch allocation and workspace growth are illegal while a
    /// stream is capturing; warmup must have exercised the phase first.
    static void checkNotCapturing(char const* what)
    {
        TLLM_CHECK_WITH_INFO(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None,
            "%s is not allowed during CUDA graph capture; run a warmup that covers this phase first.", what);
    }

    /// Every forward phase: once-only phase initialization, feature preparation, the
    /// derived op's input validation and derived fields, then workspace growth. A
    /// context call that also carries generation tokens launches both phases from one
    /// workspace, so it initializes both and reserves both phases' requirements before
    /// the first launch.
    void prepare(FmhaParams& p, bool isGen)
    {
        if (!isGen && !mContextInitialized)
        {
            checkNotCapturing("context initialization");
            derived().initContext();
            mContextInitialized = true;
        }
        if ((isGen || p.num_gen_tokens > 0) && !mGenerationInitialized)
        {
            checkNotCapturing("generation initialization");
            derived().initGeneration();
            mGenerationInitialized = true;
        }

        (Features::prepare(p, isGen), ...);
        derived().prepareInputs(p, isGen);

        bool const mixed = !isGen && p.num_gen_tokens > 0;
        std::size_t const featureBytes = (std::size_t{0} + ... + Features::getRequiredWorkspaceSize(isGen))
            + (mixed ? (std::size_t{0} + ... + Features::getRequiredWorkspaceSize(true)) : 0);
        std::size_t const bytes = derived().getRequiredWorkspaceSize(p, isGen) + featureBytes;
        if (p.workspace.numel() < static_cast<int64_t>(bytes))
        {
            checkNotCapturing("workspace growth");
            p.workspace.resize_({static_cast<int64_t>(bytes)});
        }
    }

    /// Multi-CTA KV counter semaphores on \p device, allocated on first use and grown when
    /// \p maxNumSequences needs more. Consumers dereference it without a null check -- MMHA
    /// at `params.block_counter[bhi]`, XQA at `semaphores[idxSeq]`, and the trtllm-gen
    /// kernels for their multi-CTA reduction. One counter per head per decoder sequence,
    /// floored at the SM count for the kernels that split one head across blocks, rounded
    /// up to the group size; the kernels reset it at the end of each launch.
    [[nodiscard]] std::int32_t* getMultiCtasKvCounter(std::int64_t maxNumSequences, c10::Device device) const
    {
        auto constexpr kCounterAlignment = std::int64_t{8};
        auto const numCounters = std::max<std::int64_t>(num_heads * maxNumSequences, mMultiProcessorCount);
        auto const alignedCounters = (numCounters + kCounterAlignment - 1) / kCounterAlignment * kCounterAlignment;
        auto const counterBytes = alignedCounters * static_cast<std::int64_t>(sizeof(std::int32_t));
        if (!mMultiCtasKvCounter.defined() || mMultiCtasKvCounter.device() != device
            || mMultiCtasKvCounter.numel() < counterBytes)
        {
            checkNotCapturing("multi-CTA KV counter allocation");
            mMultiCtasKvCounter = at::zeros({counterBytes}, at::TensorOptions().dtype(at::kByte).device(device));
        }
        return static_cast<std::int32_t*>(mMultiCtasKvCounter.data_ptr());
    }

    bool mContextInitialized = false;
    bool mGenerationInitialized = false;
    mutable at::Tensor mMultiCtasKvCounter;
};

/// Bits per KV-cache element for a quantization mode and activation width.
[[nodiscard]] inline int kvCacheElemSizeInBits(tensorrt_llm::common::QuantMode quantMode, size_t dTypeSize)
{
    if (quantMode.hasInt8KvCache() || quantMode.hasFp8KvCache())
    {
        return 8;
    }
    if (quantMode.hasFp4KvCache())
    {
        return 4;
    }
    return dTypeSize * 8;
}

/// Multi-head (and grouped-query) attention over the FMHA / XQA / MMHA runners.
class MHA : public TrtllmAttention<MHA, PagedKVCacheFeature, CustomMaskFeature, SageAttnFeature, OutputQuantFeature,
                RopeFeature, RelativePositionFeature, CrossAttentionFeature, SpecDecodingFeature, SparseAttnFeature>
{
public:
    explicit MHA(AttentionBase const& key);

    using RotaryScalingType = tensorrt_llm::kernels::RotaryScalingType;
    using PositionEmbeddingType = tensorrt_llm::kernels::PositionEmbeddingType;
    using AttentionMaskType = tensorrt_llm::kernels::AttentionMaskType;

    // ---- hooks called by TrtllmAttention ----
    /// Build the context FMHA runner and decide `mEnableContextFMHA`.
    void initContext();
    /// Build the XQA runner and decide `mEnableXQA`.
    void initGeneration();
    /// Validates the call and derives the per-call state into \p p.
    void prepareInputs(FmhaParams& p, bool isGen);
    /// Bytes this call needs. A context call with generation tokens covers both phases,
    /// sizing generation from the batch bound so the later generation call agrees.
    [[nodiscard]] std::size_t getRequiredWorkspaceSize(FmhaParams const& p, bool isGen) const;
    void enqueueContext(FmhaParams& p);
    void enqueueGeneration(FmhaParams& p);

    /// Whether the fused context and generation kernels for this configuration exist;
    /// builds the runners if they have not been built yet.
    [[nodiscard]] bool supportsNvFp4Output();

    [[nodiscard]] size_t getFmhaMultiCtasKvScratchSize(FmhaParams const& p) const noexcept;
    [[nodiscard]] int getMaxNumSeqLenTile(FmhaParams const& p, int batch_beam_size = 1) const;
    /// Context-phase bytes from the prepared params: num_seqs, input_seq_length, num_tokens,
    /// total_kv_len and the cross-KV extent.
    [[nodiscard]] size_t getWorkspaceSizeForContext(FmhaParams const& p) const;
    /// The context workspace layout for this call: the one place its slices are sized.
    [[nodiscard]] tensorrt_llm::common::op::AttentionContextWorkspaceLayout contextWorkspaceLayout(
        FmhaParams const& p) const;
    /// Generation-phase bytes from the prepared params: the batch bound (sum of beam widths),
    /// the attention window, num_gen_tokens and the block-table width.
    [[nodiscard]] size_t getWorkspaceSizeForGeneration(FmhaParams const& p) const noexcept;

    template <typename T, typename KVCacheBuffer>
    int enqueueContext(FmhaParams const& p, cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    int enqueueGeneration(FmhaParams const& p, cudaStream_t stream);

    template <typename T>
    [[nodiscard]] int getKvCacheElemSizeInBits(FmhaParams const& p) const
    {
        return kvCacheElemSizeInBits(quant_mode, sizeof(T));
    }

    /// Defaulted so the dtype dispatch can name this directly; paged KV is the only
    /// layout the generation path prepares for.
    template <typename T, typename KVCacheBuffer = kernels::KVBlockArray>
    void prepareEnqueueGeneration(FmhaParams const& p);

    template <typename T, typename KVCacheBuffer>
    bool convertMMHAParamsToXQAParams(
        tensorrt_llm::kernels::XQAParams& xqaParams, FmhaParams const& p, bool forConfigurePlugin);

    /// The op's own derived state, for debugging a dispatch decision.
    [[nodiscard]] std::string toString() const;

    // ---------------------------------------------------------------------------
    // Predicates over the runners chosen at initialization, and over the per-call
    // state, which is passed explicitly. Config-only predicates are on AttentionBase.
    // ---------------------------------------------------------------------------
    [[nodiscard]] bool isUnfusedCrossAttention(FmhaParams const& p) const
    {
        return !mEnableContextFMHA && p.is_cross;
    }

    [[nodiscard]] bool isCrossAttention(FmhaParams const& p) const
    {
        return p.is_cross;
    }

    [[nodiscard]] bool useKVCache(FmhaParams const& p) const
    {
        return p.hasKvCache();
    }

    [[nodiscard]] bool usePackedCustomMask(FmhaParams const& p) const
    {
        return useCustomMask() && mEnableContextFMHA;
    }

    [[nodiscard]] bool useSparseAttention() const
    {
        return use_sparse_attention && mPagedKVCache && mEnableXQA;
    }

    [[nodiscard]] bool useTllmGenSparseAttentionPaged(FmhaParams const& p) const
    {
        return p.hasPagedSparseAttnIndices() && useSparseAttention();
    }

    [[nodiscard]] bool useTllmGenSparseAttention() const
    {
        return use_sparse_attention && mUseTllmGen && use_tllm_gen_sparse_attention;
    }

    [[nodiscard]] bool useSageAttn() const
    {
        return feature<torch_ext::SageAttnFeature>().enabled();
    }

    [[nodiscard]] bool useHopperSageAttn() const
    {
        return feature<torch_ext::SageAttnFeature>().isHopperLayout();
    }

private:
    /// One per phase, instantiated per activation dtype.
    template <typename T>
    void runContextImpl(FmhaParams& p);
    template <typename T>
    void runGenerationImpl(FmhaParams& p);

    /// CUDA-side state both phases share (cuBLAS handle, env caches); runs once, from
    /// whichever init hook comes first.
    void initShared();

    bool mPagedKVCache = true;
    bool mFP8ContextFMHA = false;

    // fmha runner (enabled by default)
    // flag: disabled = 0, enabled = 1, enabled with fp32 accumulation = 2
    bool mEnableContextFMHA = true;
    bool mMultiBlockMode = true;
    bool mEnableXQA = true;

    using XqaPrepareProfile = std::tuple<bool, bool, bool, int64_t, bool>;
    std::map<XqaPrepareProfile, int64_t> mXqaPreparedBeamWidths;

#ifdef SKIP_SOFTMAX_STAT
    at::Tensor mDefaultSkipSoftmaxStat;
#endif

    static constexpr int kReservedMaxSeqLenTilePerSeq = 64;

    bool mForceMultiBlockWarned = false;
    std::shared_ptr<CUDADriverWrapper> mDriver;
    std::unique_ptr<tensorrt_llm::kernels::FmhaDispatcher> mFmhaDispatcher;
    std::unique_ptr<tensorrt_llm::kernels::XqaDispatcher> mXqaDispatcher;
    std::unique_ptr<tensorrt_llm::common::CublasMMWrapper> mCublasWrapper;
};

/// Multi-head latent attention (DeepSeek). The MLA RoPE and FP8 staging kernels feed the
/// separate-Q/K/V (or, for sparse MLA, paged-latent) context FMHA; generation runs the
/// trtllm-gen MLA kernel, FlashMLA on Hopper, XQA on SM120, or the decoder FMHA over the
/// paged latent cache.
class MLA : public TrtllmAttention<MLA, PagedKVCacheFeature, OutputQuantFeature, RopeFeature, SpecDecodingFeature,
                SparseMlaFeature>
{
public:
    explicit MLA(AttentionBase const& key);

    using AttentionMaskType = tensorrt_llm::kernels::AttentionMaskType;

    // ---- hooks called by TrtllmAttention ----
    /// Build the context FMHA dispatcher with the latent (sparse) or separate-Q/K/V layout.
    void initContext();
    /// Build the trtllm-gen MLA runner, the decoder FMHA runner or XQA, and decide `mEnableXQA`.
    void initGeneration();
    /// Validates the call and derives the per-call state into \p p.
    void prepareInputs(FmhaParams& p, bool isGen);
    /// Bytes this call needs: the MLA context staging buffers and the generation scratch.
    [[nodiscard]] std::size_t getRequiredWorkspaceSize(FmhaParams const& p, bool isGen) const;
    void enqueueContext(FmhaParams& p);
    void enqueueGeneration(FmhaParams& p);

    /// The op's own derived state, for debugging a dispatch decision.
    [[nodiscard]] std::string toString() const;

    [[nodiscard]] bool useSparseMLA() const
    {
        return feature<torch_ext::SparseMlaFeature>().enabled();
    }

    [[nodiscard]] int getFlashMlaNumSmParts(int s_q, int num_heads, int num_kv_heads, int head_size_v) const
    {
        static constexpr int block_size_m = 64;
        int num_heads_per_head_k = s_q * num_heads / num_kv_heads;
        int sm_cnt = mMultiProcessorCount;
        int num_sm_parts = sm_cnt / num_kv_heads / cutlass::ceil_div(num_heads_per_head_k, block_size_m);
        return num_sm_parts;
    }

    [[nodiscard]] static int getFlashMlaNumSmPartsStatic(int s_q, int num_heads, int num_kv_heads, int head_size_v)
    {
        static constexpr int block_size_m = 64;
        int num_heads_per_head_k = s_q * num_heads / num_kv_heads;
        int device;
        cudaGetDevice(&device);
        int sm_cnt;
        cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, device);
        int num_sm_parts = sm_cnt / num_kv_heads / cutlass::ceil_div(num_heads_per_head_k, block_size_m);
        return num_sm_parts;
    }

private:
    /// The context launch, per activation dtype: decoder info, MLA RoPE, FP8 K/V staging,
    /// then the FMHA. The MlaParams it fills lives on its stack for the launch.
    template <typename T>
    int enqueueContext(FmhaParams const& p, cudaStream_t stream);
    /// The generation launch, per activation dtype: trtllm-gen, FlashMLA, XQA or the
    /// decoder FMHA.
    template <typename T>
    int enqueueGeneration(FmhaParams const& p, cudaStream_t stream);

    template <typename T>
    void prepareEnqueueGeneration(FmhaParams const& p);
    /// XQA parameters for the SM120 generation path; false when T has no XQA kernels.
    template <typename T>
    bool fillXqaParams(tensorrt_llm::kernels::XQAParams& xqaParams, FmhaParams const& p, bool forConfigurePlugin) const;

    /// The context workspace layout for this call: the one place its slices are sized.
    [[nodiscard]] tensorrt_llm::common::op::AttentionContextWorkspaceLayout contextWorkspaceLayout(
        FmhaParams const& p) const;
    [[nodiscard]] size_t getWorkspaceSizeForContext(FmhaParams const& p) const;
    [[nodiscard]] size_t getWorkspaceSizeForGeneration(FmhaParams const& p) const noexcept;
    [[nodiscard]] size_t getFmhaMultiCtasKvScratchSize() const noexcept;

    tensorrt_llm::kernels::MlaMetaParams mMLAParams{};
    bool mIsGenerationMLA = false;
    bool mUseGenFlashMLA = false;
    // Static sparse MLA reads a separately dequantized FP8 scratch pool, so an NVFP4
    // paged cache still runs the FP8 kernels.
    bool mUseNvfp4MlaKvCache = false;
    // Skip correction when the row-max increase is within this base-2 threshold.
    float mSkipCorrectionThreshold = 0.0F;
    bool mFP8ContextMLA = false;
    bool mFP8GenerationMLA = false;
    bool mMultiBlockMode = true;
    bool mEnableXQA = false;

    using XqaPrepareProfile = std::tuple<bool, bool, bool, int64_t, bool>;
    std::map<XqaPrepareProfile, int64_t> mXqaPreparedBeamWidths;

#ifdef SKIP_SOFTMAX_STAT
    at::Tensor mDefaultSkipSoftmaxStat;
#endif

    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2> mDecoderFMHARunner;
    std::unique_ptr<tensorrt_llm::kernels::FmhaDispatcher> mFmhaDispatcher;
    std::unique_ptr<tensorrt_llm::kernels::XqaDispatcher> mXqaDispatcher;
    std::unique_ptr<tensorrt_llm::kernels::TllmGenFmhaRunner> mTllmGenFMHARunner;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END
