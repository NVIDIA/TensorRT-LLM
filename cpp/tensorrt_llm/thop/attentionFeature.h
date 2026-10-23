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

#include "tensorrt_llm/thop/attentionOp.h"

#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

#include <climits>
#include <concepts>
#include <cstddef>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

/// A feature: constructed from the key, prepared on every forward phase, and mapped onto
/// a backend's parameter structs through `apply(KernelStruct&) const` overloads it
/// declares for the structs it affects.
template <typename F>
concept AttentionFeature
    = std::constructible_from<F, AttentionBase const&> && requires(F f, F const cf, FmhaParams const& p)
{
    {
        f.prepare(p, bool{})
    } -> std::same_as<void>;
    {
        cf.getRequiredWorkspaceSize(bool{})
    } -> std::same_as<std::size_t>;
};

/// The MMHA launch parameters, defined with the op; features map onto it like onto the
/// runners' structs.
template <typename T, typename KVCacheBuffer>
struct FusedQKVMaskedAttentionDispatchParams;

/// trtllm-gen kernels serve SM100-family devices except the consumer Blackwell parts.
[[nodiscard]] inline bool isTllmGenDevice()
{
    int const sm = tensorrt_llm::common::getSMVersion();
    return sm >= 100 && sm != 120 && sm != 121;
}

// ---- Features. Each is constructed from the key, prepared on every forward phase, and maps
// what it captured onto a backend's parameter structs through apply() overloads. Features hold
// no tensor references between calls.

/// Paged KV cache: the only cache layout the phased path runs. First in every pack so features
/// that interpret block indices read what it captured.
class PagedKVCacheFeature
{
public:
    explicit PagedKVCacheFeature(AttentionBase const& c)
        : mEnabled(c.use_kv_cache)
        , mPagedContextFmha(c.paged_context_fmha)
        , mTokensPerBlock(static_cast<int32_t>(c.tokens_per_block))
        , mQuantMode(c.quant_mode)
        , mKvHeads(c.is_mla_enable ? 1 : static_cast<int32_t>(c.num_kv_heads))
        , mKvHeadSize(static_cast<int32_t>(c.is_mla_enable ? c.kv_lora_rank + c.qk_rope_head_dim : c.head_size))
        , mKvFactor(c.is_mla_enable ? 1 : 2)
        , mElemBits(kvElemBits(c))
        , mMla(c.is_mla_enable)
    {
    }

    void prepare(FmhaParams const& p, bool /*isGen*/)
    {
        mActive = mEnabled;
        if (mQuantMode.hasFp4KvCache())
        {
            validateFp4Scales(p);
        }
        mMaxBlocksPerSeq = static_cast<int32_t>(p.getMaxBlocksPerSequence());
        mPoolPointers = {};
        mBlockOffsets = nullptr;
        if (!mActive)
        {
            return;
        }
        TLLM_CHECK_WITH_INFO(p.hasKvCache(),
            "KV cache is enabled but kv_cache_block_offsets, host_kv_cache_pool_pointers or "
            "host_kv_cache_pool_mapping is missing.");
        int32_t const poolIndex = p.getKvCachePoolIndex(p.local_layer_idx);
        int32_t const layerIdxInCachePool = p.getLayerIdxInCachePool(p.local_layer_idx);
        auto const blockSize = static_cast<int64_t>(mTokensPerBlock) * mKvHeads * mKvHeadSize;
        auto const bytesPerBlock = blockSize * mElemBits / CHAR_BIT;
        mPoolPointers = buildKvCachePoolPointers(p.getHostKvCachePoolPointers(), poolIndex,
            layerIdxInCachePool * mKvFactor * bytesPerBlock, blockSize, layerIdxInCachePool, mKvFactor,
            mQuantMode.hasFp4KvCache());
        mBlockOffsets = p.getKvCacheBlockOffsets(poolIndex);
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    [[nodiscard]] KvCachePoolPointers const& poolPointers() const
    {
        return mPoolPointers;
    }

    [[nodiscard]] kernels::KVBlockArray::DataType* blockOffsets() const
    {
        return mBlockOffsets;
    }

    [[nodiscard]] int32_t maxBlocksPerSeq() const
    {
        return mMaxBlocksPerSeq;
    }

    void apply(kernels::MHARunnerFixedParams& fp) const
    {
        fp.numTokensPerBlock = mTokensPerBlock;
        if (mEnabled && mPagedContextFmha)
        {
            if (mQuantMode.hasFp8KvCache())
            {
                fp.dataTypeKv = kernels::DATA_TYPE_E4M3;
            }
            else if (mQuantMode.hasFp4KvCache())
            {
                fp.dataTypeKv = kernels::DATA_TYPE_E2M1;
            }
        }
    }

    void apply(kernels::XqaFixedParams& fp) const
    {
        fp.numTokensPerBlock = mTokensPerBlock;
        fp.isPagedKv = mEnabled;
        if (mQuantMode.hasInt8KvCache())
        {
            fp.kvDataType = kernels::DATA_TYPE_INT8;
            fp.mathDataType = fp.inputDataType;
        }
        else if (mQuantMode.hasFp8KvCache())
        {
            fp.kvDataType = kernels::DATA_TYPE_E4M3;
            fp.mathDataType = kernels::DATA_TYPE_E4M3;
        }
        else if (mQuantMode.hasFp4KvCache())
        {
            fp.kvDataType = kernels::DATA_TYPE_E2M1;
            fp.mathDataType = kernels::DATA_TYPE_E4M3;
        }
        else
        {
            fp.kvDataType = fp.inputDataType;
            fp.mathDataType = fp.inputDataType;
        }
    }

    void apply(kernels::XQAParams& xp) const
    {
        xp.paged_kv_cache = mEnabled;
        xp.tokens_per_block = mTokensPerBlock;
        xp.kv_cache_quant_mode = mQuantMode;
        xp.max_blocks_per_sequence = mMaxBlocksPerSeq;
        if (mQuantMode.hasInt8KvCache())
        {
            xp.kv_cache_data_type = kernels::DATA_TYPE_INT8;
        }
        else if (mQuantMode.hasFp8KvCache())
        {
            xp.kv_cache_data_type = kernels::DATA_TYPE_E4M3;
        }
        else if (mQuantMode.hasFp4KvCache())
        {
            xp.kv_cache_data_type = kernels::DATA_TYPE_E2M1;
        }
        else
        {
            xp.kv_cache_data_type = xp.data_type;
        }
    }

private:
    /// An FP4 cache reads the scales as raw float pointers, so the shape and layout are
    /// part of the ABI rather than something the kernels can check.
    void validateFp4Scales(FmhaParams const& p) const
    {
        auto const& origQuant = p.fwd.kv_scale_orig_quant;
        auto const& quantOrig = p.fwd.kv_scale_quant_orig;
        if (!mMla)
        {
            TORCH_CHECK(!origQuant.has_value() || origQuant->size(0) == 3,
                "FP4 KV cache expects kv_scale_orig_quant to have 3 elements.");
            TORCH_CHECK(!quantOrig.has_value() || quantOrig->size(0) == 3,
                "FP4 KV cache expects kv_scale_quant_orig to have 3 elements.");
            return;
        }
        if (!origQuant.has_value() || !quantOrig.has_value())
        {
            return;
        }
        TORCH_CHECK(origQuant->scalar_type() == torch::kFloat32,
            "kv_scale_orig_quant must have float32 dtype for MLA with FP4 KV cache");
        TORCH_CHECK(quantOrig->scalar_type() == torch::kFloat32,
            "kv_scale_quant_orig must have float32 dtype for MLA with FP4 KV cache");
        TORCH_CHECK(origQuant->is_contiguous(), "kv_scale_orig_quant must be contiguous for MLA with FP4 KV cache");
        TORCH_CHECK(quantOrig->is_contiguous(), "kv_scale_quant_orig must be contiguous for MLA with FP4 KV cache");
        TORCH_CHECK(origQuant->dim() == 1 && origQuant->size(0) == 1,
            "kv_scale_orig_quant must have shape [1] for MLA with FP4 KV cache");
        TORCH_CHECK(quantOrig->dim() == 1 && quantOrig->size(0) == 1,
            "kv_scale_quant_orig must have shape [1] for MLA with FP4 KV cache");
    }

    [[nodiscard]] static int kvElemBits(AttentionBase const& c)
    {
        if (c.quant_mode.hasInt8KvCache() || c.quant_mode.hasFp8KvCache())
        {
            return 8;
        }
        if (c.quant_mode.hasFp4KvCache())
        {
            return 4;
        }
        return c.type == tensorrt_llm::DataType::kFLOAT ? 32 : 16;
    }

    bool mEnabled;
    bool mPagedContextFmha;
    int32_t mTokensPerBlock;
    tensorrt_llm::common::QuantMode mQuantMode;
    int32_t mKvHeads;
    int32_t mKvHeadSize;
    int32_t mKvFactor;
    int mElemBits;
    bool mMla;
    // Per call, set by prepare().
    bool mActive{false};
    KvCachePoolPointers mPoolPointers{};
    kernels::KVBlockArray::DataType* mBlockOffsets{nullptr};
    int32_t mMaxBlocksPerSeq{0};
};

static_assert(AttentionFeature<PagedKVCacheFeature>);

/// Custom attention mask. The predefined masks are plain values of mask_type in the fixed
/// params; only CUSTOM_MASK is a feature. The phased path carries no packed mask tensor.
class CustomMaskFeature
{
public:
    explicit CustomMaskFeature(AttentionBase const& c)
        : mEnabled(c.useCustomMask())
    {
    }

    void prepare(FmhaParams const& /*p*/, bool isGen)
    {
        mActive = mEnabled && !isGen;
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    void apply(kernels::MHARunnerFixedParams& fp) const
    {
        if (mEnabled)
        {
            fp.attentionMaskType = kernels::ContextAttentionMaskType::CUSTOM_MASK;
        }
    }

    void apply(kernels::MHARunnerParams& rp) const
    {
        if (mActive)
        {
            rp.packedMaskPtr = nullptr;
        }
    }

private:
    bool mEnabled;
    bool mActive{false};
};

static_assert(AttentionFeature<CustomMaskFeature>);

/// SageAttention: per-block Q/K/V quantization in the context FMHA. Needs the FP8 context FMHA
/// without a KV cache; the scale buffers are op workspace views written next to the quant launches.
class SageAttnFeature
{
public:
    explicit SageAttnFeature(AttentionBase const& c)
        : mEnabled(c.sageAttnRequested() && !c.is_mla_enable && !c.use_kv_cache)
        , mCross(c.cross_attention)
        , mMaskType(c.mask_type)
        , mBlockQ(static_cast<int>(c.sage_attn_num_elts_per_blk_q))
        , mBlockK(static_cast<int>(c.sage_attn_num_elts_per_blk_k))
        , mBlockV(static_cast<int>(c.sage_attn_num_elts_per_blk_v))
        , mQkInt8(c.sage_attn_qk_int8)
        , mKvCacheQuant(c.quant_mode.hasKvCacheQuant())
        , mHopperLayout(mEnabled && tensorrt_llm::common::getSMVersion() == kernels::kSM_90 && mQkInt8 && mBlockQ == 2
              && mBlockK == 16 && mBlockV == 1)
    {
    }

    void prepare(FmhaParams const& p, bool isGen)
    {
        mActive = mEnabled && !isGen;
        if (!mActive)
        {
            return;
        }
        TLLM_CHECK_WITH_INFO(mMaskType == kernels::AttentionMaskType::PADDING,
            "SageAttention only supports dense (padding) mask, got mask type %d.", static_cast<int>(mMaskType));
        TLLM_CHECK_WITH_INFO(mBlockQ > 0 && mBlockK > 0 && mBlockV == 1,
            "SageQuant requires positive block sizes for Q and K while the block size for V must be 1.");
        // The op keeps the KV-scale pair only when the cache is quantized and both arrived.
        TLLM_CHECK_WITH_INFO(
            !(mKvCacheQuant && p.fwd.kv_scale_orig_quant.has_value() && p.fwd.kv_scale_quant_orig.has_value()),
            "SageAttention disregards the configured p.fwd.kv_scale_quant_orig, invalidating the result.");
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    [[nodiscard]] bool enabled() const
    {
        return mEnabled;
    }

    [[nodiscard]] bool isHopperLayout() const
    {
        return mHopperLayout;
    }

    void apply(kernels::MHARunnerFixedParams& fp) const
    {
        if (!mEnabled)
        {
            return;
        }
        fp.sageBlockSizeQ = mBlockQ;
        fp.sageBlockSizeK = mBlockK;
        fp.sageBlockSizeV = mBlockV;
        if (mQkInt8)
        {
            fp.dataType = kernels::DATA_TYPE_INT8;
            fp.dataTypeKv = kernels::DATA_TYPE_KV_INT8_E4M3;
        }
        if (!mCross)
        {
            fp.attentionInputLayout = kernels::AttentionInputLayout::SEPARATE_Q_K_V;
        }
    }

private:
    bool mEnabled;
    bool mCross;
    kernels::AttentionMaskType mMaskType;
    int mBlockQ;
    int mBlockK;
    int mBlockV;
    bool mQkInt8;
    bool mKvCacheQuant;
    bool mHopperLayout;
    bool mActive{false};
};

static_assert(AttentionFeature<SageAttnFeature>);

/// FP8 or NVFP4 attention output. Fixed side: the runners' output dtype. Per launch: the output
/// scale-factor tensor and the quantization scales for this phase's tokens.
class OutputQuantFeature
{
public:
    explicit OutputQuantFeature(AttentionBase const& c)
        : mFp8Out(c.is_fp8_out)
        , mFp4Out(c.is_fp4_out)
    {
    }

    void prepare(FmhaParams const& p, bool /*isGen*/)
    {
        TORCH_CHECK(!mFp4Out || p.fwd.output_sf.has_value(), "NVFP4 output requires output_sf.");
        mOutputSf = p.getOutputSf();
        mQuantScale = mFp8Out ? p.getOutQuantScale() : nullptr;
        mSfScale = p.getOutSfScale();
        mTokenOffset = static_cast<int32_t>(p.token_offset);
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    void apply(kernels::MHARunnerFixedParams& fp) const
    {
        if (mFp4Out)
        {
            fp.dataTypeOut = kernels::DATA_TYPE_E2M1;
        }
        else if (mFp8Out)
        {
            fp.dataTypeOut = kernels::DATA_TYPE_E4M3;
        }
    }

    void apply(kernels::XqaFixedParams& fp) const
    {
        if (mFp4Out)
        {
            fp.outputDataType = kernels::DATA_TYPE_E2M1;
        }
        else if (mFp8Out)
        {
            fp.outputDataType = kernels::DATA_TYPE_E4M3;
        }
    }

    void apply(kernels::MHARunnerParams& rp) const
    {
        rp.outputSfPtr = mOutputSf;
        rp.oSfScalePtr = mSfScale;
    }

    void apply(kernels::XQAParams& xp) const
    {
        xp.is_fp8_output = mFp8Out;
        xp.fp8_out_scale = mQuantScale;
        xp.output_sf = mOutputSf;
        xp.fp4_out_sf_scale = mSfScale;
        xp.start_token_idx_sf = mTokenOffset;
    }

private:
    bool mFp8Out;
    bool mFp4Out;
    // Per call, set by prepare().
    void* mOutputSf{nullptr};
    float const* mQuantScale{nullptr};
    float const* mSfScale{nullptr};
    int32_t mTokenOffset{0};
};

static_assert(AttentionFeature<OutputQuantFeature>);

/// Rotary position embedding, including the LongRoPE and mRoPE variants selected through the
/// layer's rope parameters. The inv-freq scratch is an op workspace view; the kernels that take
/// positional arguments read the layer's rope parameters through the op.
class RopeFeature
{
public:
    explicit RopeFeature(AttentionBase const& c)
        : mEnabled(c.isRoPE())
        , mDim(static_cast<int>(c.rotary_embedding_dim))
        , mType(c.position_embedding_type)
    {
    }

    void prepare(FmhaParams const& p, bool /*isGen*/)
    {
        // Spelled out per member: the generator emits an attn.rope_params member when C++ reads it.
        mTheta = p.attn.rope_params.theta;
        mScaleType = p.attn.rope_params.scale_type;
        mScale = p.attn.rope_params.scale;
        mMaxPositions = static_cast<int>(p.attn.rope_params.max_positions);
        mOriginalMaxPositions = static_cast<int>(p.attn.rope_params.original_max_positions);
        mShortMScale = p.attn.rope_params.short_m_scale;
        mLongMScale = p.attn.rope_params.long_m_scale;
        mCosSin = p.getRotaryCosSin();
        mInvFreqCache = p.getRotaryInvFreq();
        mMropeDeltas = p.getMropePositionDeltas();
        mMropeCosSin = p.getMropeRotaryCosSin();
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    [[nodiscard]] bool enabled() const
    {
        return mEnabled;
    }

    // The scalars are written for every layer, as before: a non-RoPE layer carries dim 0.
    void apply(kernels::XQAParams& xp) const
    {
        xp.rotary_embedding_dim = mDim;
        xp.rotary_embedding_base = mTheta;
        xp.rotary_embedding_scale_type = mScaleType;
        xp.rotary_embedding_scale = mScale;
        xp.rotary_embedding_max_positions = mMaxPositions;
        xp.rotary_vision_start = -1;
        xp.rotary_vision_length = -1;
        xp.rotary_cos_sin = mCosSin;
        xp.position_embedding_type = mType;
        xp.rotary_embedding_inv_freq_cache = mInvFreqCache;
        xp.mrope_position_deltas = mMropeDeltas;
    }

    template <typename T, typename KVCacheBuffer>
    void apply(kernels::QKVPreprocessingParams<T, KVCacheBuffer>& pp) const
    {
        pp.rotary_coef_cache_buffer = mCosSin;
        pp.mrope_rotary_cos_sin = mMropeCosSin;
        pp.rotary_embedding_dim = mDim;
        pp.rotary_embedding_base = mTheta;
        pp.rotary_scale_type = mScaleType;
        pp.rotary_embedding_scale = mScale;
        pp.rotary_embedding_max_positions = mMaxPositions;
        pp.position_embedding_type = mType;
        pp.rotary_vision_start = -1;
        pp.rotary_vision_length = -1;
    }

    template <typename T>
    void apply(kernels::BuildDecoderInfoParams<T>& dp) const
    {
        dp.rotaryEmbeddingScale = mScale;
        dp.rotaryEmbeddingBase = mTheta;
        dp.rotaryEmbeddingDim = mDim;
        dp.rotaryScalingType = mScaleType;
        dp.rotaryEmbeddingInvFreqCache = mInvFreqCache;
        dp.rotaryEmbeddingMaxPositions = mMaxPositions;
    }

    template <typename T, typename KVCacheBuffer>
    void apply(FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer>& d) const;

private:
    bool mEnabled;
    int mDim;
    kernels::PositionEmbeddingType mType;
    // Per call, set by prepare().
    float mTheta{0.F};
    kernels::RotaryScalingType mScaleType{};
    float mScale{0.F};
    int mMaxPositions{0};
    int mOriginalMaxPositions{0};
    float mShortMScale{0.F};
    float mLongMScale{0.F};
    float2* mCosSin{nullptr};
    float* mInvFreqCache{nullptr};
    std::int32_t* mMropeDeltas{nullptr};
    float2* mMropeCosSin{nullptr};
};

static_assert(AttentionFeature<RopeFeature>);

/// T5-style relative position bias and ALiBi. The bias only reaches the unfused kernels, which
/// read it through the accessors; the fixed params carry the ALiBi bits.
class RelativePositionFeature
{
public:
    explicit RelativePositionFeature(AttentionBase const& c)
        : mRelative(c.isRelativePosition())
        , mAlibi(c.isALiBi())
        , mAlibiScale(c.isAliBiWithScale())
    {
    }

    void prepare(FmhaParams const& p, bool /*isGen*/)
    {
        mBias = nullptr;
        mStride = 0;
        mMaxDistance = static_cast<int>(p.fwd.relative_attention_max_distance);
        if (!p.fwd.relative_attention_bias.has_value())
        {
            return;
        }
        auto const& bias = p.fwd.relative_attention_bias.value();
        TORCH_CHECK(bias.dim() == 2 || bias.dim() == 3,
            "relative_attention_bias must be [num_heads, num_buckets] for implicit mode or "
            "[num_heads, max_seq_len, max_seq_len] for explicit mode");
        TORCH_CHECK(bias.dim() != 2 || mMaxDistance > 0,
            "relative_attention_max_distance must be positive when relative_attention_bias is a bucket table");
        TORCH_CHECK(bias.dim() != 3 || mMaxDistance == 0,
            "relative_attention_max_distance must be 0 when relative_attention_bias is precomputed");
        TLLM_CHECK_WITH_INFO(mRelative, "relative_attention_bias requires position_embedding_type to be relative.");
        TORCH_CHECK(bias.is_contiguous(), "relative_attention_bias must be contiguous");
        TORCH_CHECK(bias.scalar_type() == p.getQueryInput().scalar_type(),
            "relative_attention_bias dtype must match attention input dtype");
        mBias = bias.data_ptr();
        mStride = static_cast<int>(bias.size(1));
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    template <typename T>
    [[nodiscard]] T const* relativeAttentionBias() const
    {
        return static_cast<T const*>(mBias);
    }

    [[nodiscard]] int biasStride() const
    {
        return mStride;
    }

    [[nodiscard]] int maxDistance() const
    {
        return mMaxDistance;
    }

    void apply(kernels::MHARunnerFixedParams& fp) const
    {
        fp.hasAlibi = mAlibi;
        fp.scaleAlibi = mAlibiScale;
    }

    void apply(kernels::XqaFixedParams& fp) const
    {
        fp.hasAlibi = mAlibi;
    }

    void apply(kernels::XQAParams& xp) const
    {
        xp.max_distance = mMaxDistance;
    }

    template <typename T, typename KVCacheBuffer>
    void apply(FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer>& d) const;

private:
    bool mRelative;
    bool mAlibi;
    bool mAlibiScale;
    // Per call, set by prepare().
    void const* mBias{nullptr};
    int mStride{0};
    int mMaxDistance{0};
};

static_assert(AttentionFeature<RelativePositionFeature>);

/// Cross attention: the decoder attends to encoder KV, so the KV extent, lengths and padding
/// come from the encoder side. The unfused mask fill and the KV-cache window read the accessors.
class CrossAttentionFeature
{
public:
    explicit CrossAttentionFeature(AttentionBase const& c)
        : mEnabled(c.cross_attention)
        , mKvCache(c.use_kv_cache)
    {
    }

    void prepare(FmhaParams const& p, bool /*isGen*/)
    {
        mActive = mEnabled;
        mCrossKvLength = 0;
        mNumEncoderTokens = 0;
        mEncoderLengths = nullptr;
        mContextLengths = nullptr;
        mEncoderKv = nullptr;
        if (!mActive)
        {
            return;
        }
        mCrossKvLength = p.fwd.cross_kv.has_value() && p.batch_size > 0
            ? p.getMaxHostPastKeyValueLength(p.seq_offset, p.batch_size)
            : 0;
        mNumEncoderTokens = p.batch_size > 0 ? p.getCrossKvNumTokens() : 0;
        mEncoderLengths = p.getSequenceLengths();
        mContextLengths = p.getContextLengths();
        mEncoderKv = p.fwd.cross_kv.has_value() ? p.fwd.cross_kv->data_ptr() : nullptr;
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    [[nodiscard]] bool active() const
    {
        return mActive;
    }

    /// Encoder sequence extent for this phase.
    [[nodiscard]] int32_t crossKvLength() const
    {
        return mCrossKvLength;
    }

    [[nodiscard]] int32_t const* encoderInputLengths() const
    {
        return mEncoderLengths;
    }

    [[nodiscard]] int64_t numEncoderTokens() const
    {
        return mNumEncoderTokens;
    }

    template <typename T>
    [[nodiscard]] T* encoderKv() const
    {
        return static_cast<T*>(mEncoderKv);
    }

    void apply(kernels::MHARunnerParams& rp) const
    {
        if (!mActive)
        {
            return;
        }
        rp.kvSeqLen = mCrossKvLength;
        rp.slidingWindowSize = mCrossKvLength;
        rp.totalKvSeqLen = static_cast<int>(mNumEncoderTokens);
        if (!mKvCache)
        {
            rp.kvPtr = mEncoderKv;
        }
    }

    template <typename T, typename KVCacheBuffer>
    void apply(kernels::QKVPreprocessingParams<T, KVCacheBuffer>& pp) const
    {
        if (!mActive)
        {
            return;
        }
        pp.cross_kv_input = static_cast<T*>(mEncoderKv);
        // Callers do not consistently use sequence_lengths as the decoder length; the decoder
        // context lengths open the encoder KV-cache write gate.
        pp.cache_seq_lens = mContextLengths;
        pp.encoder_seq_lens = mEncoderLengths;
        pp.max_kv_seq_len = mCrossKvLength;
        pp.cyclic_kv_cache_len = mCrossKvLength;
        pp.cross_attention = true;
        pp.separate_q_kv_output = true;
    }

    template <typename T>
    void apply(kernels::BuildDecoderInfoParams<T>& dp) const
    {
        if (!mActive)
        {
            return;
        }
        dp.attentionMask = nullptr;
        dp.seqKVLengths = mEncoderLengths;
        dp.maxEncoderQSeqLength = mCrossKvLength;
    }

    void apply(kernels::XQAParams& xp) const
    {
        xp.cross_attention = mActive;
        xp.encoder_input_lengths = mEncoderLengths;
    }

    template <typename T, typename KVCacheBuffer>
    void apply(FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer>& d) const;

private:
    bool mEnabled;
    bool mKvCache;
    // Per call, set by prepare().
    bool mActive{false};
    int32_t mCrossKvLength{0};
    int64_t mNumEncoderTokens{0};
    int32_t const* mEncoderLengths{nullptr};
    int32_t const* mContextLengths{nullptr};
    void* mEncoderKv{nullptr};
};

static_assert(AttentionFeature<CrossAttentionFeature>);

/// Speculative decoding in the generation phase. The configuration bound goes into the fixed
/// params; the launch extent is this call's mask stride, read from the tensors.
class SpecDecodingFeature
{
public:
    explicit SpecDecodingFeature(AttentionBase const& c)
        : mEnabled(c.is_spec_decoding_enabled)
        , mTargetMaxGenLen(static_cast<int>(c.spec_decoding_target_max_gen_len))
    {
    }

    void prepare(FmhaParams const& p, bool isGen)
    {
        mIsSpecDecTree = p.meta.is_spec_dec_tree;
        mForcePrepareTreeMask = p.meta.force_prepare_spec_dec_tree_mask;
        mActive = mEnabled && isGen && p.use_spec_decoding;
        mGenerationLengths = nullptr;
        mPositionOffsets = nullptr;
        mPackedMask = nullptr;
        mBlTreeMaskOffset = nullptr;
        mBlTreeMask = nullptr;
        mBlTreeMaskCapacity = 0;
        mFirstSparseMaskOffsetKv = nullptr;
        mMaxGenLen = 1;
        if (!mActive)
        {
            return;
        }
        TORCH_CHECK(p.spec_decoding_generation_lengths.has_value(),
            "Expecting spec_decoding_generation_lengths in spec-dec mode.");
        TORCH_CHECK(
            p.spec_decoding_position_offsets.has_value(), "Expecting spec_decoding_position_offsets in spec-dec mode.");
        TORCH_CHECK(p.spec_decoding_packed_mask.has_value(), "Expecting spec_decoding_packed_mask in spec-dec mode.");
        // [batch_size, max_draft_len + 1]
        TORCH_CHECK(p.spec_decoding_position_offsets->dim() == 2, "spec_decoding_position_offsets must be 2-D.");
        if (mSm100Family)
        {
            TORCH_CHECK(p.spec_decoding_bl_tree_mask_offset.has_value(),
                "Expecting spec_decoding_bl_tree_mask_offset in trtllm-gen spec-dec mode.");
            TORCH_CHECK(p.spec_decoding_bl_tree_mask.has_value(),
                "Expecting spec_decoding_bl_tree_mask in trtllm-gen spec-dec mode.");
            TORCH_CHECK(p.spec_decoding_bl_tree_mask->is_contiguous() && p.spec_decoding_bl_tree_mask->numel() > 0,
                "spec_decoding_bl_tree_mask must be a non-empty contiguous tensor.");
            TORCH_CHECK(p.spec_bl_tree_first_sparse_mask_offset_kv.has_value(),
                "Expecting spec_bl_tree_first_sparse_mask_offset_kv in trtllm-gen spec-dec mode.");
            TORCH_CHECK(p.spec_decoding_packed_mask->dim() == 3,
                "spec_decoding_packed_mask must be 3-D in trtllm-gen spec-dec mode.");
        }
        mGenerationLengths = p.getSpecDecodingGenerationLengths();
        mPositionOffsets = p.getSpecDecodingPositionOffsets();
        mPackedMask = p.getSpecDecodingPackedMask();
        mBlTreeMaskOffset = p.getSpecDecodingBlTreeMaskOffset();
        mBlTreeMask = p.getSpecDecodingBlTreeMask();
        mBlTreeMaskCapacity = p.spec_decoding_bl_tree_mask.has_value() ? p.spec_decoding_bl_tree_mask->numel() : 0;
        mFirstSparseMaskOffsetKv = p.getSpecBlTreeFirstSparseMaskOffsetKv();
        // Blackwell uses the padded packed-mask row dim as the mask stride.
        mMaxGenLen = static_cast<int>(
            mSm100Family ? p.spec_decoding_packed_mask->size(1) : p.spec_decoding_position_offsets->size(1));
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    /// Launch extent for this call; 1 when speculative generation is not running.
    [[nodiscard]] int maxGenerationLength() const
    {
        return mMaxGenLen;
    }

    /// Speculative generation is running this call.
    [[nodiscard]] bool active() const
    {
        return mActive;
    }

    void apply(kernels::XqaFixedParams& fp) const
    {
        fp.isSpecDecoding = mEnabled;
        fp.specDecodingTargetMaxGenLen = mTargetMaxGenLen;
        if (mEnabled && !mTllmGen)
        {
            fp.outputDataType = kernels::DATA_TYPE_E4M3;
            TLLM_CHECK_WITH_INFO(fp.numQHeads % fp.numKvHeads == 0, "num_heads should be multiples of mNumKVHeads.");
        }
    }

    void apply(kernels::XQAParams& xp) const
    {
        xp.multi_query_tokens = mActive;
        xp.is_spec_dec_tree = mIsSpecDecTree;
        xp.force_prepare_spec_dec_tree_mask = mForcePrepareTreeMask;
        xp.spec_decoding_packed_mask = mPackedMask;
        xp.spec_decoding_position_offsets = mPositionOffsets;
        xp.spec_decoding_generation_lengths = mGenerationLengths;
        xp.spec_decoding_is_generation_length_variable = mActive;
        xp.spec_decoding_max_generation_length = mMaxGenLen;
        xp.spec_decoding_bl_tree_mask_offset = mBlTreeMaskOffset;
        xp.spec_decoding_bl_tree_mask = mBlTreeMask;
        xp.spec_decoding_bl_tree_mask_capacity = mBlTreeMaskCapacity;
        xp.spec_bl_tree_first_sparse_mask_offset_kv = mFirstSparseMaskOffsetKv;
    }

private:
    bool mEnabled;
    int mTargetMaxGenLen;
    bool mTllmGen = isTllmGenDevice();
    bool mSm100Family = tensorrt_llm::common::isSM100Family();
    // Per call, set by prepare().
    bool mActive{false};
    bool mIsSpecDecTree{true};
    bool mForcePrepareTreeMask{false};
    int mMaxGenLen{1};
    int32_t const* mPackedMask{nullptr};
    int32_t const* mPositionOffsets{nullptr};
    int32_t const* mGenerationLengths{nullptr};
    int64_t* mBlTreeMaskOffset{nullptr};
    uint32_t* mBlTreeMask{nullptr};
    int64_t mBlTreeMaskCapacity{0};
    int32_t* mFirstSparseMaskOffsetKv{nullptr};
};

static_assert(AttentionFeature<SpecDecodingFeature>);

/// Dense-model sparse attention (top-k or paged KV indices) through XQA and the trtllm-gen sparse
/// context kernels.
class SparseAttnFeature
{
public:
    explicit SparseAttnFeature(AttentionBase const& c)
        : mRequested(c.use_sparse_attention)
        , mEnabled(c.use_sparse_attention)
        , mTllmGen(mEnabled && c.use_tllm_gen_sparse_attention && isTllmGenDevice())
        , mKvCache(c.use_kv_cache)
    {
    }

    void prepare(FmhaParams const& p, bool /*isGen*/)
    {
        mActive = mEnabled;
        mSparse = p.getSparseParams();
        mPagedIndices = p.hasPagedSparseAttnIndices();
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    void apply(kernels::MHARunnerFixedParams& fp) const
    {
        if (mTllmGen)
        {
            fp.useTllmGenSparseAttention = true;
        }
    }

    void apply(kernels::XqaFixedParams& fp) const
    {
        if (mTllmGen)
        {
            fp.useTllmGenSparseAttention = true;
        }
    }

    void apply(kernels::MHARunnerParams& rp) const
    {
        if (mActive && mTllmGen)
        {
            rp.sparse_params = mSparse;
        }
    }

    void apply(kernels::XQAParams& xp) const
    {
        xp.sparse_params = mSparse;
        xp.use_sparse_attention_gen_paged = mPagedIndices && mRequested && mKvCache;
    }

private:
    bool mRequested;
    bool mEnabled;
    bool mTllmGen;
    bool mKvCache;
    // Per call, set by prepare().
    bool mActive{false};
    bool mPagedIndices{false};
    kernels::SparseAttentionParams mSparse{};
};

static_assert(AttentionFeature<SparseAttnFeature>);

/// Sparse MLA (DeepSeek sparse attention) through the trtllm-gen MLA kernels: the context runs
/// in absorption mode over the paged latent cache and generation gathers the selected indices.
/// Needs a trtllm-gen device; elsewhere the sparse runtime params only reach XQA.
class SparseMlaFeature
{
public:
    explicit SparseMlaFeature(AttentionBase const& c)
        : mRequested(c.use_sparse_attention)
        , mEnabled(c.use_sparse_attention && isTllmGenDevice())
    {
    }

    void prepare(FmhaParams const& p, bool /*isGen*/)
    {
        mSparse = p.getSparseParams();
        mPagedIndices = p.hasPagedSparseAttnIndices();
    }

    [[nodiscard]] std::size_t getRequiredWorkspaceSize(bool /*isGen*/) const
    {
        return 0;
    }

    [[nodiscard]] bool enabled() const
    {
        return mEnabled;
    }

    [[nodiscard]] bool pagedIndices() const
    {
        return mPagedIndices;
    }

    [[nodiscard]] kernels::SparseAttentionParams const& params() const
    {
        return mSparse;
    }

    void apply(kernels::MHARunnerFixedParams& fp) const
    {
        if (mEnabled)
        {
            fp.useSparseMLA = true;
            fp.useTllmGenSparseAttention = true;
        }
    }

    void apply(kernels::XqaFixedParams& fp) const
    {
        if (mEnabled)
        {
            fp.useTllmGenSparseAttention = true;
        }
    }

    void apply(kernels::MHARunnerParams& rp) const
    {
        if (mEnabled)
        {
            rp.sparse_params = mSparse;
        }
    }

    void apply(kernels::XQAParams& xp) const
    {
        xp.sparse_params = mSparse;
        xp.use_sparse_attention_gen_paged = mPagedIndices && mRequested;
    }

private:
    bool mRequested;
    bool mEnabled;
    // Per call, set by prepare().
    bool mPagedIndices{false};
    kernels::SparseAttentionParams mSparse{};
};

static_assert(AttentionFeature<SparseMlaFeature>);

} // namespace torch_ext

TRTLLM_NAMESPACE_END
