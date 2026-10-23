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
#include "tensorrt_llm/thop/phasedFmha.h"

#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

/// The shared algorithm of a TensorRT-LLM attention op. `Derived` supplies four hooks:
/// `initContext()` and `initGeneration()`, run once before that phase's first launch;
/// `prepareInputs(p, isGen)`, `enqueueContext(p)` and `enqueueGeneration(p)`, run every
/// call. Its constructor lists its FMHA libraries with `addFmha<T>()` in priority order;
/// each call runs the first library that supports it, as `selectFmha()` finds. Features
/// are private bases; the folds below call their generic hooks, and `applyFeatures()`
/// calls whichever `apply` overload each feature has for a given kernel struct.
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
        mUseTllmGen = tensorrt_llm::common::isSM100Family(mSM);
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

    /// Lists \p Fmha after the libraries already added, when its key check admits it.
    template <typename Fmha>
    void addFmha()
    {
        if (Fmha::isAvailable(*this))
        {
            mFmhas.push_back(std::make_unique<Fmha>(*this));
        }
    }

    /// The first listed library that supports this call's phase.
    [[nodiscard]] PhasedFmha<Derived>& selectFmha(FmhaParams const& p, bool isGen) const
    {
        for (auto const& fmha : mFmhas)
        {
            if (fmha->isSupported(p, isGen))
            {
                return *fmha;
            }
        }
        TLLM_THROW("No FMHA library supports this %s call for the attention key: %s", isGen ? "generation" : "context",
            toString().c_str());
    }

    /// The phase's once-only initialization: the derived op's hook, then every library's.
    void initPhase(bool isGen)
    {
        if (!isGen && !mContextInitialized)
        {
            checkNotCapturing("context initialization");
            derived().initContext();
            for (auto& fmha : mFmhas)
            {
                fmha->initContext(derived());
            }
            mContextInitialized = true;
        }
        if (isGen && !mGenerationInitialized)
        {
            checkNotCapturing("generation initialization");
            derived().initGeneration();
            for (auto& fmha : mFmhas)
            {
                fmha->initGeneration(derived());
            }
            mGenerationInitialized = true;
        }
    }

    /// Bytes the FMHA libraries need for this call: the selected context library's, and the
    /// largest generation requirement among the libraries, so a generation call can switch
    /// library without growing the workspace.
    [[nodiscard]] std::size_t getFmhaWorkspaceSize(FmhaParams const& p, bool isGen) const
    {
        std::size_t bytes = 0;
        if (!isGen && p.num_tokens > 0)
        {
            bytes = selectFmha(p, /*isGen=*/false).getWorkspaceSize(derived(), p, /*isGen=*/false);
        }
        for (auto const& fmha : mFmhas)
        {
            bytes = std::max(bytes, fmha->getWorkspaceSize(derived(), p, /*isGen=*/true));
        }
        return bytes;
    }

    /// Every forward phase: once-only phase initialization, feature preparation, the
    /// derived op's input validation and derived fields, then workspace growth. A
    /// context call that also carries generation tokens launches both phases from one
    /// workspace, so it initializes both and reserves both phases' requirements before
    /// the first launch.
    void prepare(FmhaParams& p, bool isGen)
    {
        initPhase(isGen);
        if (!isGen && p.num_gen_tokens > 0)
        {
            initPhase(/*isGen=*/true);
        }

        (Features::prepare(p, isGen), ...);
        derived().prepareInputs(p, isGen);

        bool const mixed = !isGen && p.num_gen_tokens > 0;
        std::size_t const featureBytes = (std::size_t{0} + ... + Features::getRequiredWorkspaceSize(isGen))
            + (mixed ? (std::size_t{0} + ... + Features::getRequiredWorkspaceSize(true)) : 0);
        std::size_t const bytes = getFmhaWorkspaceSize(p, isGen) + featureBytes;
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

    /// In priority order, filtered by each library's `isAvailable(key)`.
    std::vector<std::unique_ptr<PhasedFmha<Derived>>> mFmhas;
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

/// Multi-head (and grouped-query) attention. Context runs trtllm-gen, fmha_v2 or the unfused
/// cuBLAS path; generation runs trtllm-gen, XQA or MMHA.
class MHA : public TrtllmAttention<MHA, PagedKVCacheFeature, CustomMaskFeature, SageAttnFeature, OutputQuantFeature,
                RopeFeature, RelativePositionFeature, CrossAttentionFeature, SpecDecodingFeature, SparseAttnFeature>
{
public:
    explicit MHA(AttentionBase const& key);

    using RotaryScalingType = tensorrt_llm::kernels::RotaryScalingType;
    using PositionEmbeddingType = tensorrt_llm::kernels::PositionEmbeddingType;
    using AttentionMaskType = tensorrt_llm::kernels::AttentionMaskType;

    // ---- hooks called by TrtllmAttention ----
    void initContext();
    /// Resolves the multi-block mode from the environment.
    void initGeneration();
    /// Validates the call and derives the per-call state into \p p.
    void prepareInputs(FmhaParams& p, bool isGen);
    void enqueueContext(FmhaParams& p);
    void enqueueGeneration(FmhaParams& p);

    /// Whether the fused context and generation kernels for this configuration exist;
    /// builds the runners if they have not been built yet.
    [[nodiscard]] bool supportsNvFp4Output();

    /// The op's own derived state, for debugging a dispatch decision.
    [[nodiscard]] std::string toString() const;

    [[nodiscard]] bool isCrossAttention(FmhaParams const& p) const
    {
        return p.is_cross;
    }

    [[nodiscard]] bool useKVCache(FmhaParams const& p) const
    {
        return p.hasKvCache();
    }

    [[nodiscard]] bool useTllmGenSparseAttentionPaged(FmhaParams const& p) const
    {
        return p.hasPagedSparseAttnIndices() && use_sparse_attention && mPagedKVCache;
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
    template <typename>
    friend class TrtllmGenFmha;
    template <typename>
    friend class FmhaV2Fmha;
    template <typename>
    friend class XqaFmha;
    friend class MmhaFmha;
    friend class UnfusedFmha;

    /// The fused context runners' fixed parameters: layout, data types, heads and features.
    [[nodiscard]] kernels::MHARunnerFixedParams contextFixedParams() const;
    /// Whether the fused context kernels read Q apart from K/V.
    [[nodiscard]] bool isSeparateQAndKvInput() const;

    [[nodiscard]] size_t getFmhaMultiCtasKvScratchSize(FmhaParams const& p) const noexcept;
    /// The context workspace layout for this call: the one place its slices are sized.
    /// \p unfused selects the unfused path's staging buffers.
    [[nodiscard]] tensorrt_llm::common::op::AttentionContextWorkspaceLayout contextWorkspaceLayout(
        FmhaParams const& p, bool unfused) const;
    [[nodiscard]] size_t getWorkspaceSizeForContext(FmhaParams const& p, bool unfused) const;
    /// Generation bytes of the XQA kernels, sized from the batch bound (sum of beam widths),
    /// num_gen_tokens and the block-table width.
    [[nodiscard]] size_t getXqaWorkspaceSize(FmhaParams const& p) const;

    template <typename T>
    [[nodiscard]] int getKvCacheElemSizeInBits() const
    {
        return kvCacheElemSizeInBits(quant_mode, sizeof(T));
    }

    /// The paged KV cache views for this call's rows.
    template <typename T>
    [[nodiscard]] KvCacheBuffers<kernels::KVBlockArray> getKvCacheBuffers(FmhaParams const& p, bool isGen) const;
    /// cu_seqlens, tokens info, the FP8 scales, and for \p unfused the attention mask and
    /// padding offsets.
    template <typename T>
    void buildContextDecoderInfo(FmhaParams const& p,
        tensorrt_llm::common::op::AttentionContextWorkspaceViews<T> const& workspaceViews, bool unfused,
        cudaStream_t stream) const;
    /// The fused context path: decoder info, QKV preprocessing or SageQuant, \p launch with
    /// the filled runner parameters, then KV-cache postprocessing.
    template <typename T>
    void enqueueFusedContext(FmhaParams const& p, cudaStream_t stream,
        std::function<void(kernels::MHARunnerParams const&)> const& launch) const;
    template <typename T, typename KVCacheBuffer>
    bool convertMMHAParamsToXQAParams(
        tensorrt_llm::kernels::XQAParams& xqaParams, FmhaParams const& p, bool forConfigurePlugin) const;

    /// The FP8 output scale, which only FP8 context FMHA quantizes the output with.
    [[nodiscard]] float* getOutQuantScale(FmhaParams const& p) const
    {
        return mFP8ContextFMHA ? p.getOutQuantScale() : nullptr;
    }

    bool mPagedKVCache = true;
    bool mFP8ContextFMHA = false;
    bool mMultiBlockMode = true;

#ifdef SKIP_SOFTMAX_STAT
    at::Tensor mDefaultSkipSoftmaxStat;
#endif
};

/// Multi-head latent attention (DeepSeek). The MLA RoPE and FP8 staging kernels feed the
/// separate-Q/K/V (or, for sparse MLA, paged-latent) context FMHA; generation runs the
/// trtllm-gen MLA kernel, FlashMLA on Hopper, XQA on SM120, or the fmha_v2 kernels over the
/// paged latent cache.
class MLA : public TrtllmAttention<MLA, PagedKVCacheFeature, OutputQuantFeature, RopeFeature, SpecDecodingFeature,
                SparseMlaFeature>
{
public:
    explicit MLA(AttentionBase const& key);

    using AttentionMaskType = tensorrt_llm::kernels::AttentionMaskType;

    // ---- hooks called by TrtllmAttention ----
    void initContext();
    /// Resolves the multi-block mode from the environment.
    void initGeneration();
    /// Validates the call and derives the per-call state into \p p.
    void prepareInputs(FmhaParams& p, bool isGen);
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
    template <typename>
    friend class TrtllmGenFmha;
    template <typename>
    friend class FmhaV2Fmha;
    template <typename>
    friend class XqaFmha;
    friend class FlashMlaFmha;

    /// The context runners' fixed parameters: the latent (sparse) or separate-Q/K/V layout.
    [[nodiscard]] kernels::MHARunnerFixedParams contextFixedParams() const;

    /// The context launch, per activation dtype: decoder info, MLA RoPE, FP8 K/V staging,
    /// then \p launch with the filled runner parameters.
    template <typename T>
    void enqueueFusedContext(FmhaParams const& p, cudaStream_t stream,
        std::function<void(kernels::MHARunnerParams const&)> const& launch) const;
    /// The generation parameters every generation kernel reads.
    template <typename T>
    [[nodiscard]] kernels::MlaParams<T> getGenerationParams(FmhaParams const& p) const;
    /// The paged latent cache for this call's rows.
    template <typename T>
    [[nodiscard]] kernels::KVBlockArray getGenerationKvCacheBuffer(FmhaParams const& p) const;
    /// XQA parameters for the SM120 generation path; false when T has no XQA kernels.
    template <typename T>
    bool fillXqaParams(tensorrt_llm::kernels::XQAParams& xqaParams, FmhaParams const& p, bool forConfigurePlugin) const;

    /// The context workspace layout for this call: the one place its slices are sized.
    [[nodiscard]] tensorrt_llm::common::op::AttentionContextWorkspaceLayout contextWorkspaceLayout(
        FmhaParams const& p) const;
    [[nodiscard]] size_t getWorkspaceSizeForContext(FmhaParams const& p) const;
    /// Generation bytes of the trtllm-gen, fmha_v2 and (with \p flashMla) FlashMLA kernels.
    [[nodiscard]] size_t getGenerationWorkspaceSize(FmhaParams const& p, bool flashMla) const noexcept;
    /// Generation bytes of the XQA kernels.
    [[nodiscard]] size_t getXqaWorkspaceSize(FmhaParams const& p) const noexcept;
    [[nodiscard]] size_t getFmhaMultiCtasKvScratchSize() const noexcept;

    /// The FP8 output scale, which only FP8 context FMHA quantizes the output with.
    [[nodiscard]] float* getOutQuantScale(FmhaParams const& p) const
    {
        bool const fp8ContextFmha
            = is_fp8_out || (quant_mode.hasFp8KvCache() && paged_context_fmha) || sageAttnRequested();
        return fp8ContextFmha ? p.getOutQuantScale() : nullptr;
    }

    tensorrt_llm::kernels::MlaMetaParams mMLAParams{};
    bool mIsGenerationMLA = false;
    // Static sparse MLA reads a separately dequantized FP8 scratch pool, so an NVFP4
    // paged cache still runs the FP8 kernels.
    bool mUseNvfp4MlaKvCache = false;
    // Skip correction when the row-max increase is within this base-2 threshold.
    float mSkipCorrectionThreshold = 0.0F;
    bool mFP8ContextMLA = false;
    bool mFP8GenerationMLA = false;
    bool mMultiBlockMode = true;

#ifdef SKIP_SOFTMAX_STAT
    at::Tensor mDefaultSkipSoftmaxStat;
#endif
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END
