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

#include "tensorrt_llm/common/attentionWorkspace.h"
#include "tensorrt_llm/thop/attentionFeature.h"
#include "tensorrt_llm/thop/phasedFmha.h"

#include <c10/cuda/CUDAGraphsC10Utils.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

/// The shared algorithm of a TensorRT-LLM attention op. `Derived` supplies the static
/// `checkParams(p, isGen)`, checked before the op for a call's key is built, which ends with
/// `checkFeatureParams()`, and four hooks:
/// `initContext()` and `initGeneration()`, run once before that phase's first launch;
/// `prepareInputs(p, isGen)`, `enqueueContext(p, fmha)` and `enqueueGeneration(p, fmha)`,
/// run every call. It also holds one member per FMHA library it can use, built with
/// `makeFmha<T>()` when the key admits the library, and lists them with `fmhas()`. The
/// caller picks the library of each call with `isFmhaSupported()` and runs it with
/// `runContext()` or `runGeneration()`. Features
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

    /// Whether \p fmha serves this call's phase; false when it is null, a library the key
    /// does not admit. Prepares the call as a launch does, without growing the workspace.
    [[nodiscard]] bool isFmhaSupported(PhasedFmha<Derived> const* fmha, FmhaParams& p, bool isGen)
    {
        if (fmha == nullptr)
        {
            return false;
        }
        preparePhase(p, isGen);
        return fmha->isSupported(p, isGen);
    }

    /// Every phase call, launched or only queried: once-only phase initialization, feature
    /// preparation, and the derived op's input validation and derived fields.
    void preparePhase(FmhaParams& p, bool isGen)
    {
        initPhase(isGen);
        (Features::prepare(p, isGen), ...);
        derived().prepareInputs(p, isGen);
    }

    /// Runs this call's context phase on \p fmha. The caller chose it with `isFmhaSupported()`,
    /// possibly for an earlier call of the same shape.
    void runContext(PhasedFmha<Derived>* fmha, FmhaParams& p)
    {
        TLLM_CHECK_WITH_INFO(fmha != nullptr,
            "The attention op does not list the requested FMHA library for the key: %s", toString().c_str());
        prepare(p, /*isGen=*/false, *fmha);
        derived().enqueueContext(p, *fmha);
    }

    /// Runs this call's generation phase on \p fmha; see `runContext()`.
    void runGeneration(PhasedFmha<Derived>* fmha, FmhaParams& p)
    {
        TLLM_CHECK_WITH_INFO(fmha != nullptr,
            "The attention op does not list the requested FMHA library for the key: %s", toString().c_str());
        prepare(p, /*isGen=*/true, *fmha);
        derived().enqueueGeneration(p, *fmha);
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

    /// The checks every variant shares: the query dtype, then every feature's `checkParams()`.
    [[nodiscard]] static bool checkFeatureParams(FmhaParams const& p, bool isGen)
    {
        return p.getQueryInput().scalar_type() != torch::kFloat8_e4m3fn && (Features::checkParams(p, isGen) && ...);
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

    /// A \p Fmha for this op when its key check admits it, null otherwise.
    template <typename Fmha>
    [[nodiscard]] std::unique_ptr<Fmha> makeFmha() const
    {
        return Fmha::isAvailable(*this) ? std::make_unique<Fmha>(*this) : nullptr;
    }

    /// The phase's once-only initialization: the derived op's hook, then every library's.
    void initPhase(bool isGen)
    {
        if (!isGen && !mContextInitialized)
        {
            checkNotCapturing("context initialization");
            derived().initContext();
            for (auto* fmha : derived().fmhas())
            {
                if (fmha != nullptr)
                {
                    fmha->initContext(derived());
                }
            }
            mContextInitialized = true;
        }
        if (isGen && !mGenerationInitialized)
        {
            checkNotCapturing("generation initialization");
            derived().initGeneration();
            for (auto* fmha : derived().fmhas())
            {
                if (fmha != nullptr)
                {
                    fmha->initGeneration(derived());
                }
            }
            mGenerationInitialized = true;
        }
    }

    /// Bytes the FMHA libraries need for this call: the context library's, and the largest
    /// generation requirement among the libraries, so a generation call can switch library
    /// without growing the workspace.
    [[nodiscard]] std::size_t getFmhaWorkspaceSize(
        FmhaParams const& p, bool isGen, PhasedFmha<Derived> const& fmha) const
    {
        std::size_t bytes = 0;
        if (!isGen && p.num_tokens > 0)
        {
            bytes = fmha.getWorkspaceSize(derived(), p, /*isGen=*/false);
        }
        for (auto const* listed : derived().fmhas())
        {
            if (listed != nullptr)
            {
                bytes = std::max(bytes, listed->getWorkspaceSize(derived(), p, /*isGen=*/true));
            }
        }
        return bytes;
    }

    /// Every launched phase: `preparePhase()`, then workspace growth. A context call that
    /// also carries generation tokens launches both phases from one workspace, so it
    /// initializes both and reserves both phases' requirements before the first launch.
    void prepare(FmhaParams& p, bool isGen, PhasedFmha<Derived> const& fmha)
    {
        preparePhase(p, isGen);
        bool const mixed = !isGen && p.num_gen_tokens > 0;
        if (mixed)
        {
            initPhase(/*isGen=*/true);
        }

        std::size_t const featureBytes = (std::size_t{0} + ... + Features::getRequiredWorkspaceSize(isGen))
            + (mixed ? (std::size_t{0} + ... + Features::getRequiredWorkspaceSize(true)) : 0);
        std::size_t const bytes = getFmhaWorkspaceSize(p, isGen, fmha) + featureBytes;
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

    /// Whether this op serves the call's inputs. Static, so a call it declines builds no op.
    /// Separate Q/K/V input is read only by cross attention and SageAttention.
    [[nodiscard]] static bool checkParams(FmhaParams const& p, bool isGen)
    {
        return (p.fwd.is_fused_qkv || p.is_cross || torch_ext::SageAttnFeature::isRequested(p))
            && checkFeatureParams(p, isGen);
    }

    // ---- hooks called by TrtllmAttention ----
    void initContext();
    /// Resolves the multi-block mode from the environment.
    void initGeneration();
    /// Validates the call and derives the per-call state into \p p.
    void prepareInputs(FmhaParams& p, bool isGen);
    void enqueueContext(FmhaParams& p, PhasedFmha<MHA>& fmha);
    void enqueueGeneration(FmhaParams& p, PhasedFmha<MHA>& fmha);

    /// Whether the fused context and generation kernels for this configuration exist;
    /// builds the runners if they have not been built yet.
    [[nodiscard]] bool supportsNvFp4Output();
    /// Whether a variable-window context kernel exists for this configuration.
    [[nodiscard]] bool supportsVariableWindow();

    /// The op's own derived state, for debugging a dispatch decision.
    [[nodiscard]] std::string toString() const;

    // The FMHA libraries; null for a library the key does not admit.
    [[nodiscard]] TrtllmGenFmha<MHA>* trtllmGen() const
    {
        return mTrtllmGen.get();
    }

    [[nodiscard]] FmhaV2Fmha<MHA>* fmhaV2() const
    {
        return mFmhaV2.get();
    }

    [[nodiscard]] XqaFmha<MHA>* xqa() const
    {
        return mXqa.get();
    }

    [[nodiscard]] MmhaFmha* mmha() const
    {
        return mMmha.get();
    }

    [[nodiscard]] UnfusedFmha* unfused() const
    {
        return mUnfused.get();
    }

    /// Every FMHA library in priority order, null for those the key does not admit.
    [[nodiscard]] std::array<PhasedFmha<MHA>*, 5> fmhas() const
    {
        return {mTrtllmGen.get(), mFmhaV2.get(), mXqa.get(), mMmha.get(), mUnfused.get()};
    }

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

    /// The fused context workspace slices for this call: the one place they are sized.
    [[nodiscard]] tensorrt_llm::common::op::AttentionContextWorkspaceSizes contextWorkspaceSizes(
        FmhaParams const& p) const;
    /// The fused context workspace layout for this call.
    [[nodiscard]] tensorrt_llm::common::op::AttentionContextWorkspaceLayout contextWorkspaceLayout(
        FmhaParams const& p) const;
    [[nodiscard]] size_t getWorkspaceSizeForContext(FmhaParams const& p) const;

    template <typename T>
    [[nodiscard]] int getKvCacheElemSizeInBits() const
    {
        return kvCacheElemSizeInBits(quant_mode, sizeof(T));
    }

    /// The paged KV cache views for this call's rows.
    template <typename T>
    [[nodiscard]] KvCacheBuffers<kernels::KVBlockArray> getKvCacheBuffers(FmhaParams const& p, bool isGen) const;
    /// cu_seqlens, tokens info, the FP8 scales, and the attention mask and padding offsets
    /// where \p workspaceViews has slices for them.
    template <typename T>
    void buildContextDecoderInfo(FmhaParams const& p,
        tensorrt_llm::common::op::AttentionContextWorkspaceViews<T> const& workspaceViews, cudaStream_t stream) const;
    /// The fused context path: decoder info, QKV preprocessing or SageQuant, \p launch with
    /// the filled runner parameters, then KV-cache postprocessing.
    template <typename T>
    void enqueueFusedContext(FmhaParams const& p, cudaStream_t stream,
        std::function<void(kernels::MHARunnerParams const&)> const& launch) const;

    /// The FP8 output scale, which only FP8 context FMHA quantizes the output with.
    [[nodiscard]] float* getOutQuantScale(FmhaParams const& p) const
    {
        return mFP8ContextFMHA ? p.getOutQuantScale() : nullptr;
    }

    std::unique_ptr<TrtllmGenFmha<MHA>> mTrtllmGen;
    std::unique_ptr<FmhaV2Fmha<MHA>> mFmhaV2;
    std::unique_ptr<XqaFmha<MHA>> mXqa;
    std::unique_ptr<MmhaFmha> mMmha;
    std::unique_ptr<UnfusedFmha> mUnfused;
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

    /// Whether this op serves the call's inputs. Static, so a call it declines builds no op.
    [[nodiscard]] static bool checkParams(FmhaParams const& p, bool isGen)
    {
        return static_cast<AttentionMaskType>(p.fwd.mask_type) != AttentionMaskType::CUSTOM_MASK
            && checkFeatureParams(p, isGen);
    }

    // ---- hooks called by TrtllmAttention ----
    void initContext();
    /// Resolves the multi-block mode from the environment.
    void initGeneration();
    /// Validates the call and derives the per-call state into \p p.
    void prepareInputs(FmhaParams& p, bool isGen);
    void enqueueContext(FmhaParams& p, PhasedFmha<MLA>& fmha);
    void enqueueGeneration(FmhaParams& p, PhasedFmha<MLA>& fmha);

    /// The op's own derived state, for debugging a dispatch decision.
    [[nodiscard]] std::string toString() const;

    [[nodiscard]] bool useSparseMLA() const
    {
        return feature<torch_ext::SparseMlaFeature>().enabled();
    }

    // The FMHA libraries; null for a library the key does not admit.
    [[nodiscard]] TrtllmGenFmha<MLA>* trtllmGen() const
    {
        return mTrtllmGen.get();
    }

    [[nodiscard]] FlashMlaFmha* flashMla() const
    {
        return mFlashMla.get();
    }

    [[nodiscard]] XqaFmha<MLA>* xqa() const
    {
        return mXqa.get();
    }

    [[nodiscard]] FmhaV2Fmha<MLA>* fmhaV2() const
    {
        return mFmhaV2.get();
    }

    /// Every FMHA library in priority order, null for those the key does not admit.
    [[nodiscard]] std::array<PhasedFmha<MLA>*, 4> fmhas() const
    {
        return {mTrtllmGen.get(), mFlashMla.get(), mXqa.get(), mFmhaV2.get()};
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

    /// The context workspace layout for this call: the one place its slices are sized.
    [[nodiscard]] tensorrt_llm::common::op::AttentionContextWorkspaceLayout contextWorkspaceLayout(
        FmhaParams const& p) const;
    [[nodiscard]] size_t getWorkspaceSizeForContext(FmhaParams const& p) const;
    /// Generation bytes of the trtllm-gen, fmha_v2 and FlashMLA kernels: the buffers they
    /// share, then the \p kernelWorkspaceSize bytes of the calling library's own buffers.
    [[nodiscard]] size_t getGenerationWorkspaceSize(FmhaParams const& p, size_t kernelWorkspaceSize) const noexcept;

    /// The FP8 output scale, which only FP8 context FMHA quantizes the output with.
    [[nodiscard]] float* getOutQuantScale(FmhaParams const& p) const
    {
        bool const fp8ContextFmha
            = is_fp8_out || (quant_mode.hasFp8KvCache() && paged_context_fmha) || sageAttnRequested();
        return fp8ContextFmha ? p.getOutQuantScale() : nullptr;
    }

    std::unique_ptr<TrtllmGenFmha<MLA>> mTrtllmGen;
    std::unique_ptr<FlashMlaFmha> mFlashMla;
    std::unique_ptr<XqaFmha<MLA>> mXqa;
    std::unique_ptr<FmhaV2Fmha<MLA>> mFmhaV2;
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
