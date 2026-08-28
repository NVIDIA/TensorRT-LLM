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

// Bit-exact equivalence between the two DeepSeek-FP8 MoE activation kernels.
//
// `moe::dev::activation::run` picks between
//   * `activationDeepSeekKernel`         - grids over the expanded (numTokens x
//     topK) index space and discovers work through expandedIdxToPermutedIdx,
//   * `activationDeepSeekPermutedKernel` - grids directly over the permuted row
//     space with one warp per (row, 128-element scale block),
// via `shouldUsePermutedActivation()`. Both must produce *identical bits* for
// every row that carries a real token. DevKernel.cu documents that the permuted
// kernel must not hoist a reciprocal out of `out / scaleOut`, because `x / s`
// and `x * (1/s)` round differently and one ulp was enough to flip a
// greedy-decoded token. An `isClose`-style comparison would not catch that
// regression, so everything below compares raw bit patterns.
//
// Note on coverage: fp8 e4m3 carries three mantissa bits, so most 1-ulp fp32
// differences vanish when the result is rounded back down to fp8 -- only values
// sitting on a rounding boundary survive. A single small shape can therefore
// miss the reciprocal regression by chance, which is why several shapes with
// different scale-block counts are instantiated below.
//
// The dispatch inputs `Data::numExperts` and `Data::tileTokensDim` exist *only*
// for that choice: `KernelParams::setKernelParams` does not forward them, so
// neither kernel can observe them. Overriding `tileTokensDim` therefore selects
// the kernel without perturbing a single input byte, which is what lets this
// test run the very same inputs through both paths.

#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace tensorrt_llm::tests::kernels::blockscalemoe
{

namespace tc = tensorrt_llm::common;
namespace tg = batchedGemm::trtllm::gen;

using tensorrt_llm::runtime::BufferManager;
using tensorrt_llm::runtime::CudaStream;
using tensorrt_llm::runtime::ITensor;
using tensorrt_llm::runtime::MemoryType;
using tensorrt_llm::runtime::bufferCast;

////////////////////////////////////////////////////////////////////////////////////////////////////

// `shouldUsePermutedActivation()` requires
//     realRowsPerExpert = numTokens * topK / numExperts >= tileTokensDim.
// A tile of 1 always satisfies it (given numTokens * topK >= numExperts); a
// tile larger than the whole expanded space never can.
constexpr int32_t kTileForcePermuted = 1;
constexpr int32_t kTileForceLegacy = 1 << 20;

// The activation scale factor is `aMax / 448.f` with `aMax >= 0`, so a negative
// value can never be produced by either kernel and makes an unambiguous
// "this entry was not written" marker.
constexpr float kScaleSentinel = -12345.0F;
constexpr int8_t kDataSentinel = static_cast<int8_t>(0x5A);

constexpr int32_t kEltsPerSf = 128;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ActivationEquivParam
{
    std::string name;
    int32_t numTokens;
    int32_t topK;
    int32_t numExperts;      // global expert count
    int32_t numLocalExperts; // this rank's share, i.e. numExperts / epSize
    int32_t intermediateSize;
    int32_t paddingTile;     // routing tile the permuted layout was built with
    bool hasSwigluLimit;
    float swigluLimit;
    uint32_t seed;
};

inline std::ostream& operator<<(std::ostream& os, ActivationEquivParam const& p)
{
    return os << p.name;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Host-side stand-in for what the routing kernel produces: a permuted row space
// grouped by local expert, with each expert's row count padded up to
// `paddingTile`.
struct PermutedLayout
{
    std::vector<int32_t> expandedIdxToPermutedIdx; // numTokens * topK, -1 for non-local
    std::vector<int32_t> realRows;                 // rows carrying a token
    std::vector<int32_t> paddingRows;              // rows that exist only because of tile padding
    int32_t totalNumPaddedTokens{0};
};

// One activation run, read back to the host as raw bytes / raw float bits.
struct ActivationResult
{
    std::vector<int8_t> bytes;
    std::vector<float> scales;
};

inline PermutedLayout buildPermutedLayout(ActivationEquivParam const& p)
{
    std::mt19937 rng(p.seed);

    // Pick topK distinct experts per token, then keep the ones this rank owns
    // (local expert offset 0, i.e. experts [0, numLocalExperts)).
    std::vector<std::vector<int32_t>> slotsPerLocalExpert(p.numLocalExperts);
    std::vector<int32_t> experts(p.numExperts);
    std::iota(experts.begin(), experts.end(), 0);
    for (int32_t token = 0; token < p.numTokens; ++token)
    {
        std::shuffle(experts.begin(), experts.end(), rng);
        for (int32_t k = 0; k < p.topK; ++k)
        {
            int32_t const expert = experts[k];
            if (expert < p.numLocalExperts)
            {
                slotsPerLocalExpert[expert].push_back(token * p.topK + k);
            }
        }
    }

    PermutedLayout layout;
    layout.expandedIdxToPermutedIdx.assign(p.numTokens * p.topK, -1);
    int32_t offset = 0;
    for (int32_t expert = 0; expert < p.numLocalExperts; ++expert)
    {
        auto const& slots = slotsPerLocalExpert[expert];
        auto const numSlots = static_cast<int32_t>(slots.size());
        for (int32_t i = 0; i < numSlots; ++i)
        {
            layout.expandedIdxToPermutedIdx[slots[i]] = offset + i;
            layout.realRows.push_back(offset + i);
        }
        int32_t const paddedCount = tc::ceilDiv(numSlots, p.paddingTile) * p.paddingTile;
        for (int32_t row = numSlots; row < paddedCount; ++row)
        {
            layout.paddingRows.push_back(offset + row);
        }
        offset += paddedCount;
    }
    layout.totalNumPaddedTokens = offset;
    return layout;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int8_t toFp8Byte(float value)
{
    cutlass::float_e4m3_t const converted(value);
    int8_t byte{};
    std::memcpy(&byte, &converted, sizeof(byte));
    return byte;
}

inline uint32_t floatBits(float value)
{
    uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

class BlockScaleMoeActivationEquivalenceTest : public ::testing::TestWithParam<ActivationEquivParam>
{
protected:
    void SetUp() override
    {
        if (tc::getSMVersion() < 90)
        {
            GTEST_SKIP() << "The trtllm-gen block-scale MoE activation kernels target SM90+.";
        }
        mStream = std::make_shared<CudaStream>();
        mBufferManager = std::make_shared<BufferManager>(mStream);
    }

    // Allocates device buffers, fills the inputs deterministically and uploads
    // them. `zeroedRowBlock`, when set, forces one (row, scale block) pair of
    // the input to all-zero so both kernels exercise the finite aMax floor on
    // exactly the same element.
    void setUp(ActivationEquivParam const& param, PermutedLayout const& layout,
        std::optional<std::pair<int32_t, int32_t>> zeroedRowBlock = std::nullopt)
    {
        mParam = param;
        mInnerDim = 2 * param.intermediateSize;
        mOutputDim = param.intermediateSize;
        mTotalRows = layout.totalNumPaddedTokens;
        mNumOutSfBlocks = mOutputDim / kEltsPerSf;

        auto const numInElts = static_cast<int64_t>(mTotalRows) * mInnerDim;
        auto const numInSfElts = static_cast<int64_t>(mInnerDim / kEltsPerSf) * mTotalRows;
        mNumOutElts = static_cast<int64_t>(mTotalRows) * mOutputDim;
        mNumOutSfElts = static_cast<int64_t>(mNumOutSfBlocks) * mTotalRows;

        std::mt19937 rng(param.seed + 977U);
        std::uniform_real_distribution<float> valueDist(-4.F, 4.F);
        std::uniform_real_distribution<float> scaleDist(0.05F, 2.F);

        std::vector<int8_t> hostIn(numInElts);
        for (auto& byte : hostIn)
        {
            byte = toFp8Byte(valueDist(rng));
        }
        std::vector<float> hostInSf(numInSfElts);
        for (auto& scale : hostInSf)
        {
            scale = scaleDist(rng);
        }

        if (zeroedRowBlock.has_value())
        {
            auto const [row, sfBlock] = *zeroedRowBlock;
            auto const base = static_cast<int64_t>(row) * mInnerDim + static_cast<int64_t>(sfBlock) * kEltsPerSf;
            // Both halves: `up` at [base, base+128) and `gate` at [base+outputDim, ...).
            std::fill_n(hostIn.begin() + base, kEltsPerSf, int8_t{0});
            std::fill_n(hostIn.begin() + base + mOutputDim, kEltsPerSf, int8_t{0});
        }

        auto upload = [this](auto const& host) {
            return mBufferManager->copyFrom(
                host, ITensor::makeShape({static_cast<int64_t>(host.size())}), MemoryType::kGPU);
        };

        mInDevice = upload(hostIn);
        mInSfDevice = upload(hostInSf);
        mExpandedMapDevice = upload(layout.expandedIdxToPermutedIdx);
        mTotalPaddedDevice = upload(std::vector<int32_t>{mTotalRows});

        mOutDevice = mBufferManager->gpu(ITensor::makeShape({mNumOutElts}), tensorrt_llm::DataType::kINT8);
        mOutSfDevice = mBufferManager->gpu(ITensor::makeShape({mNumOutSfElts}), tensorrt_llm::DataType::kFLOAT);

        mStream->synchronize();
    }

    // Resets the outputs to sentinels, runs the activation with the requested
    // dispatch override, and reads the results back.
    ActivationResult runOnce(int32_t tileTokensDimOverride)
    {
        ActivationResult result;
        std::vector<int8_t> const outSentinel(mNumOutElts, kDataSentinel);
        std::vector<float> const outSfSentinel(mNumOutSfElts, kScaleSentinel);
        mBufferManager->copy(outSentinel.data(), *mOutDevice);
        mBufferManager->copy(outSfSentinel.data(), *mOutSfDevice);

        moe::dev::activation::Data data;
        data.mDtypeElt = tg::Dtype::E4m3;
        data.mUsePdl = false;
        data.mUseDeepSeekFp8 = true;
        data.inPtr = bufferCast<int8_t>(*mInDevice);
        data.outPtr = bufferCast<int8_t>(*mOutDevice);
        data.inDqSfsPtr = bufferCast<float>(*mInSfDevice);
        data.outDqSfsPtr = bufferCast<float>(*mOutSfDevice);
        data.innerDim = mInnerDim;
        data.numTokens = mParam.numTokens;
        data.topK = mParam.topK;
        data.expandedIdxToPermutedIdx = bufferCast<int32_t>(*mExpandedMapDevice);
        data.numExperts = mParam.numExperts;
        data.tileTokensDim = tileTokensDimOverride;
        data.totalNumPaddedTokens = bufferCast<int32_t>(*mTotalPaddedDevice);
        data.swigluLimit = mParam.swigluLimit;
        data.hasSwigluLimit = mParam.hasSwigluLimit;

        moe::dev::activation::run(data, mStream->get());
        TLLM_CUDA_CHECK(cudaGetLastError());

        result.bytes.resize(mNumOutElts);
        result.scales.resize(mNumOutSfElts);
        mBufferManager->copy(*mOutDevice, result.bytes.data());
        mBufferManager->copy(*mOutSfDevice, result.scales.data());
        mStream->synchronize();
        return result;
    }

    std::shared_ptr<CudaStream> mStream;
    std::shared_ptr<BufferManager> mBufferManager;

    ITensor::SharedPtr mInDevice;
    ITensor::SharedPtr mInSfDevice;
    ITensor::SharedPtr mOutDevice;
    ITensor::SharedPtr mOutSfDevice;
    ITensor::SharedPtr mExpandedMapDevice;
    ITensor::SharedPtr mTotalPaddedDevice;

    ActivationEquivParam mParam{};
    int32_t mInnerDim{0};
    int32_t mOutputDim{0};
    int32_t mTotalRows{0};
    int32_t mNumOutSfBlocks{0};
    int64_t mNumOutElts{0};
    int64_t mNumOutSfElts{0};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_P(BlockScaleMoeActivationEquivalenceTest, BothKernelsAgreeBitForBit)
{
    auto const param = GetParam();
    auto const layout = buildPermutedLayout(param);

    ASSERT_FALSE(layout.realRows.empty()) << "the test config produced no local rows";
    // The dispatch assertions below rely on padding existing.
    ASSERT_FALSE(layout.paddingRows.empty()) << "the test config produced no tile padding";
    ASSERT_GE(static_cast<int64_t>(param.numTokens) * param.topK, param.numExperts)
        << "kTileForcePermuted only forces the permuted kernel when numTokens * topK >= numExperts";

    setUp(param, layout);

    auto const legacy = runOnce(kTileForceLegacy);
    auto const permuted = runOnce(kTileForcePermuted);

    // Guard the dispatch itself. Only the permuted kernel sweeps the per-expert
    // tile padding, so the padding rows tell the two kernels apart without
    // reaching into DevKernel.cu internals. If a future refactor made both runs
    // take the same branch, this fires instead of the comparison passing
    // vacuously.
    for (auto const row : layout.paddingRows)
    {
        for (int32_t sfBlock = 0; sfBlock < mNumOutSfBlocks; ++sfBlock)
        {
            auto const idx = static_cast<int64_t>(row) + static_cast<int64_t>(mTotalRows) * sfBlock;
            ASSERT_EQ(floatBits(legacy.scales[idx]), floatBits(kScaleSentinel))
                << "the expanded-space kernel must not touch padding row " << row;
            ASSERT_NE(floatBits(permuted.scales[idx]), floatBits(kScaleSentinel))
                << "the permuted-space kernel must sweep padding row " << row;
        }
    }

    // Every real row must have been written by both kernels, bit for bit.
    // ASSERT (not EXPECT) so a regression that hits every row reports the first
    // offending element instead of hundreds of thousands of them.
    for (auto const row : layout.realRows)
    {
        for (int32_t sfBlock = 0; sfBlock < mNumOutSfBlocks; ++sfBlock)
        {
            auto const idx = static_cast<int64_t>(row) + static_cast<int64_t>(mTotalRows) * sfBlock;
            ASSERT_NE(floatBits(legacy.scales[idx]), floatBits(kScaleSentinel))
                << "real row " << row << " scale block " << sfBlock << " was never written";
            ASSERT_EQ(floatBits(legacy.scales[idx]), floatBits(permuted.scales[idx]))
                << "scale mismatch at row " << row << " block " << sfBlock;
        }

        for (int32_t elt = 0; elt < mOutputDim; ++elt)
        {
            auto const idx = static_cast<int64_t>(row) * mOutputDim + elt;
            ASSERT_EQ(static_cast<uint8_t>(legacy.bytes[idx]), static_cast<uint8_t>(permuted.bytes[idx]))
                << "fp8 mismatch at row " << row << " element " << elt;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(BlockScaleMoeActivation, BlockScaleMoeActivationEquivalenceTest,
    ::testing::Values(
        // Small shape, heavy padding (paddingTile 8 against ~8 rows per expert).
        ActivationEquivParam{"small", /*numTokens=*/64, /*topK=*/4, /*numExperts=*/32, /*numLocalExperts=*/8,
            /*intermediateSize=*/128, /*paddingTile=*/8, /*hasSwigluLimit=*/false, /*swigluLimit=*/0.F, /*seed=*/13U},
        // Two scale blocks per row, so the warp-per-(row, block) mapping is
        // exercised with more than one block per row.
        ActivationEquivParam{"two_sf_blocks", 128, 4, 32, 8, 256, 8, false, 0.F, 17U},
        // Production-sized intermediate dims: 4 and 8 scale blocks per row.
        ActivationEquivParam{"four_sf_blocks", 96, 8, 64, 16, 512, 16, false, 0.F, 29U},
        ActivationEquivParam{"eight_sf_blocks", 64, 10, 40, 10, 1024, 8, false, 0.F, 31U},
        // The clamped SwiGLU branch (gemm1_clamp_limit) must match too.
        ActivationEquivParam{"swiglu_limit", 128, 4, 32, 8, 256, 8, true, 1.5F, 37U}),
    [](::testing::TestParamInfo<ActivationEquivParam> const& info) { return info.param.name; });

////////////////////////////////////////////////////////////////////////////////////////////////////

// An all-zero scale block must remain finite. A zero dequantization scale would
// make quantization evaluate 0 / 0, which is undefined and writes FP8 NaNs into
// that row. Both kernels floor aMax with the same epsilon and must emit
// identical zero bytes and a finite, positive scale.
TEST_F(BlockScaleMoeActivationEquivalenceTest, ZeroScaleBlockProducesFiniteZeros)
{
    ActivationEquivParam const param{"zero_block", /*numTokens=*/64, /*topK=*/4, /*numExperts=*/32,
        /*numLocalExperts=*/8, /*intermediateSize=*/256, /*paddingTile=*/8, /*hasSwigluLimit=*/false,
        /*swigluLimit=*/0.F, /*seed=*/41U};
    auto const layout = buildPermutedLayout(param);
    ASSERT_FALSE(layout.realRows.empty());

    int32_t const zeroedRow = layout.realRows.front();
    int32_t const zeroedBlock = 0;
    setUp(param, layout, std::make_pair(zeroedRow, zeroedBlock));

    auto const legacy = runOnce(kTileForceLegacy);
    auto const permuted = runOnce(kTileForcePermuted);

    auto const sfIdx = static_cast<int64_t>(zeroedRow) + static_cast<int64_t>(mTotalRows) * zeroedBlock;
    EXPECT_TRUE(std::isfinite(legacy.scales[sfIdx]));
    EXPECT_GT(legacy.scales[sfIdx], 0.F);
    EXPECT_EQ(floatBits(legacy.scales[sfIdx]), floatBits(permuted.scales[sfIdx]));

    for (int32_t elt = 0; elt < kEltsPerSf; ++elt)
    {
        auto const idx = static_cast<int64_t>(zeroedRow) * mOutputDim + zeroedBlock * kEltsPerSf + elt;
        EXPECT_EQ(legacy.bytes[idx], toFp8Byte(0.F)) << "zero block emitted non-zero FP8 at element " << elt;
        ASSERT_EQ(static_cast<uint8_t>(legacy.bytes[idx]), static_cast<uint8_t>(permuted.bytes[idx]))
            << "zero-block encoding differs at element " << elt;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TEST(BlockScaleMoeActivationBackingTest, PadsActivationUsingItsOwnRowWidth)
{
    // A single-token Qwen-style decode can have only 32 padded rows. FC1 writes
    // 2 * intermediateSize elements per row, while the gated activation read by
    // FC2 is half as wide, so reusing FC1's capacity delivers only half of
    // maybeGetMinTokenCount's 128 KiB floor. This is host-side arithmetic: it
    // pins the sizing invariant, it does not observe the actual allocation.
    constexpr int32_t maxNumPaddedTokens = 32;
    constexpr int32_t intermediateSize = 2304;
    constexpr int64_t minActivationBytes = 128 * 1024;
    // Not an independent requirement: the FP32 scales are indexed by the same
    // token capacity, so they scale with the activation and land just past 4 KiB.
    constexpr int64_t minScaleBytes = 4 * 1024;
    auto const fp8Bits = tg::dtypeGetNumBits(tg::Dtype::E4m3);

    auto const gemm1Capacity = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::maybeGetMinTokenCount(
        maxNumPaddedTokens, 2 * intermediateSize, fp8Bits);
    auto const activationCapacity = tensorrt_llm::kernels::trtllmGenFp8BlockScaleMoe::Routing::maybeGetMinTokenCount(
        maxNumPaddedTokens, intermediateSize, fp8Bits);

    auto const activationBytes = static_cast<int64_t>(activationCapacity) * intermediateSize * fp8Bits / 8;
    auto const activationBytesWithGemm1Capacity = static_cast<int64_t>(gemm1Capacity) * intermediateSize * fp8Bits / 8;
    auto const scaleBytes = static_cast<int64_t>(activationCapacity) * (intermediateSize / kEltsPerSf) * sizeof(float);
    auto const scaleBytesWithGemm1Capacity
        = static_cast<int64_t>(gemm1Capacity) * (intermediateSize / kEltsPerSf) * sizeof(float);

    EXPECT_GE(activationBytes, minActivationBytes);
    EXPECT_GE(scaleBytes, minScaleBytes);
    EXPECT_LT(activationBytesWithGemm1Capacity, minActivationBytes);
    EXPECT_LT(scaleBytesWithGemm1Capacity, minScaleBytes);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tensorrt_llm::tests::kernels::blockscalemoe
