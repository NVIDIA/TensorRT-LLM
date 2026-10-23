/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"
#include "tensorrt_llm/thop/attentionOp.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

class MHA;
class MLA;

/// C++ counterpart of the Python PhasedFmha: one kernel library serving one or both phases
/// of an attention variant (`MHA` or `MLA`).
///
/// Each library also provides `static bool isAvailable(AttentionBase const& key)`, which
/// reads only the key and the device and decides whether the variant lists the library at
/// all. The variant keeps its libraries in priority order; every call runs the first one
/// whose `isSupported` is true.
template <typename AttentionVariant>
class PhasedFmha
{
public:
    explicit PhasedFmha(AttentionBase const& attention)
        : attention(attention)
    {
    }

    virtual ~PhasedFmha() = default;

    PhasedFmha(PhasedFmha const&) = delete;
    PhasedFmha(PhasedFmha&&) = delete;
    PhasedFmha& operator=(PhasedFmha const&) = delete;
    PhasedFmha& operator=(PhasedFmha&&) = delete;

    /// Per call: whether this library serves this phase for these parameters. Reads `p`,
    /// `isGen`, the key members of `attention`, and whether the phase's runner accepted
    /// the configuration at init.
    [[nodiscard]] virtual bool isSupported(FmhaParams const& p, bool isGen) const = 0;

    /// Once per phase, before the first launch and outside CUDA graph capture. Builds the
    /// phase's runner and records whether it accepts the configuration. A library that
    /// does not serve a phase keeps the default.
    virtual void initContext(AttentionVariant const& op) {}

    virtual void initGeneration(AttentionVariant const& op) {}

    /// Workspace bytes this library needs for the phase; 0 for a phase it does not serve.
    [[nodiscard]] virtual std::size_t getWorkspaceSize(
        AttentionVariant const& op, FmhaParams const& p, bool isGen) const
        = 0;

    /// Called only after `isSupported(p, isGen)` returned true for the same phase.
    virtual void runContext(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream)
    {
        TLLM_THROW("This FMHA library does not serve the context phase.");
    }

    virtual void runGeneration(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream)
    {
        TLLM_THROW("This FMHA library does not serve the generation phase.");
    }

protected:
    /// The op that owns this library: its key members and op state.
    AttentionBase const& attention;
};

/// trtllm-gen kernels for both phases: the context FMHA, the GQA/MQA generation kernels and
/// the MLA generation kernels.
template <typename AttentionVariant>
class TrtllmGenFmha final : public PhasedFmha<AttentionVariant>
{
public:
    explicit TrtllmGenFmha(AttentionBase const& attention)
        : PhasedFmha<AttentionVariant>(attention)
    {
    }

    [[nodiscard]] static bool isAvailable(AttentionBase const& key);
    /// Whether a trtllm-gen context kernel exists for \p fixedParams.
    [[nodiscard]] static bool isContextSupported(kernels::MHARunnerFixedParams const& fixedParams);

    [[nodiscard]] bool isSupported(FmhaParams const& p, bool isGen) const override;
    void initContext(AttentionVariant const& op) override;
    void initGeneration(AttentionVariant const& op) override;
    [[nodiscard]] std::size_t getWorkspaceSize(
        AttentionVariant const& op, FmhaParams const& p, bool isGen) const override;
    void runContext(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream) override;
    void runGeneration(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream) override;

private:
    friend class MHA;

    template <typename T>
    void runGenerationImpl(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream);
    void launchContext(kernels::MHARunnerParams const& runnerParams);

    kernels::MHARunnerFixedParams mContextParams{};
    std::unique_ptr<kernels::TllmGenFmhaRunner> mContextRunner;
    /// Q is FP8 over FP8 and NVFP4 caches, the activation type otherwise.
    kernels::Data_type mGenerationQDataType{};
    std::unique_ptr<kernels::TllmGenFmhaRunner> mGenerationRunner;
};

/// fmha_v2 kernels: the context FMHA, and the MLA generation kernels over the paged latent cache.
template <typename AttentionVariant>
class FmhaV2Fmha final : public PhasedFmha<AttentionVariant>
{
public:
    explicit FmhaV2Fmha(AttentionBase const& attention)
        : PhasedFmha<AttentionVariant>(attention)
    {
    }

    [[nodiscard]] static bool isAvailable(AttentionBase const& key);
    /// Whether an fmha_v2 context kernel exists for \p fixedParams.
    [[nodiscard]] static bool isContextSupported(kernels::MHARunnerFixedParams fixedParams);

    [[nodiscard]] bool isSupported(FmhaParams const& p, bool isGen) const override;
    void initContext(AttentionVariant const& op) override;
    void initGeneration(AttentionVariant const& op) override;
    [[nodiscard]] std::size_t getWorkspaceSize(
        AttentionVariant const& op, FmhaParams const& p, bool isGen) const override;
    void runContext(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream) override;
    void runGeneration(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream) override;

private:
    friend class MHA;

    template <typename T>
    void runGenerationImpl(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream);

    std::unique_ptr<kernels::FusedMHARunnerV2> mContextRunner;
    std::unique_ptr<kernels::FusedMHARunnerV2> mGenerationRunner;
};

/// XQA generation kernels.
template <typename AttentionVariant>
class XqaFmha final : public PhasedFmha<AttentionVariant>
{
public:
    explicit XqaFmha(AttentionBase const& attention)
        : PhasedFmha<AttentionVariant>(attention)
    {
    }

    [[nodiscard]] static bool isAvailable(AttentionBase const& key);

    [[nodiscard]] bool isSupported(FmhaParams const& p, bool isGen) const override;
    void initGeneration(AttentionVariant const& op) override;
    [[nodiscard]] std::size_t getWorkspaceSize(
        AttentionVariant const& op, FmhaParams const& p, bool isGen) const override;
    void runGeneration(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream) override;

private:
    friend class MHA;

    /// The XQA parameters of this call; false when T has no XQA kernels.
    template <typename T>
    bool fillParams(
        AttentionVariant const& op, FmhaParams const& p, kernels::XQAParams& params, bool forConfigurePlugin) const;
    template <typename T>
    void runGenerationImpl(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream);
    /// Compiles the XQA kernels for a new execution mode or a wider beam than any prepared so far.
    template <typename T>
    void prepare(AttentionVariant const& op, FmhaParams const& p);

    std::unique_ptr<kernels::DecoderXQARunner> mRunner;
    using PrepareProfile = std::tuple<bool, bool, bool, std::int64_t, bool>;
    std::map<PrepareProfile, std::int64_t> mPreparedBeamWidths;
};

/// FlashMLA generation kernels (SM90, 64-token pages, 576-wide latent heads).
class FlashMlaFmha final : public PhasedFmha<MLA>
{
public:
    explicit FlashMlaFmha(AttentionBase const& attention)
        : PhasedFmha<MLA>(attention)
    {
    }

    [[nodiscard]] static bool isAvailable(AttentionBase const& key);

    [[nodiscard]] bool isSupported(FmhaParams const& p, bool isGen) const override;
    [[nodiscard]] std::size_t getWorkspaceSize(MLA const& op, FmhaParams const& p, bool isGen) const override;
    void runGeneration(MLA const& op, FmhaParams const& p, cudaStream_t stream) override;

private:
    template <typename T>
    void runGenerationImpl(MLA const& op, FmhaParams const& p, cudaStream_t stream);
};

/// Masked multi-head attention: the generation fallback for every configuration XQA and
/// trtllm-gen decline.
class MmhaFmha final : public PhasedFmha<MHA>
{
public:
    explicit MmhaFmha(AttentionBase const& attention)
        : PhasedFmha<MHA>(attention)
    {
    }

    [[nodiscard]] static bool isAvailable(AttentionBase const& key);

    [[nodiscard]] bool isSupported(FmhaParams const& p, bool isGen) const override;
    void initGeneration(MHA const& op) override;
    [[nodiscard]] std::size_t getWorkspaceSize(MHA const& op, FmhaParams const& p, bool isGen) const override;
    void runGeneration(MHA const& op, FmhaParams const& p, cudaStream_t stream) override;

private:
    template <typename T>
    void runGenerationImpl(MHA const& op, FmhaParams const& p, cudaStream_t stream);
    [[nodiscard]] int getMaxNumSeqLenTile(MHA const& op, int batchBeamSize) const;

    static constexpr int kReservedMaxSeqLenTilePerSeq = 64;
    bool mForceMultiBlockWarned = false;
};

/// Unfused context attention over cuBLAS GEMMs and a masked softmax: the context fallback.
class UnfusedFmha final : public PhasedFmha<MHA>
{
public:
    explicit UnfusedFmha(AttentionBase const& attention)
        : PhasedFmha<MHA>(attention)
    {
    }

    [[nodiscard]] static bool isAvailable(AttentionBase const& key);

    [[nodiscard]] bool isSupported(FmhaParams const& p, bool isGen) const override;
    void initContext(MHA const& op) override;
    [[nodiscard]] std::size_t getWorkspaceSize(MHA const& op, FmhaParams const& p, bool isGen) const override;
    void runContext(MHA const& op, FmhaParams const& p, cudaStream_t stream) override;

private:
    template <typename T>
    void runContextImpl(MHA const& op, FmhaParams const& p, cudaStream_t stream);

    std::unique_ptr<tensorrt_llm::common::CublasMMWrapper> mCublasWrapper;
};

/// Bytes the XQA kernels need for \p max_num_tokens query tokens.
[[nodiscard]] std::size_t xqaKernelWorkspaceSize(
    int headSize, int numQHeads, int numKvHeads, bool isMLA, bool multiBlockMode, int max_num_tokens);
/// The alignment of the XQA kernel workspace.
[[nodiscard]] int xqaKernelWorkspaceAlignment(int headSize, int numQHeads, int numKvHeads, bool multiBlockMode);

extern template class TrtllmGenFmha<MHA>;
extern template class TrtllmGenFmha<MLA>;
extern template class FmhaV2Fmha<MHA>;
extern template class FmhaV2Fmha<MLA>;
extern template class XqaFmha<MHA>;
extern template class XqaFmha<MLA>;

} // namespace torch_ext

TRTLLM_NAMESPACE_END
