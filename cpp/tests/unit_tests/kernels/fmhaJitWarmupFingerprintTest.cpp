/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Tests for TllmGenFmhaKernel::jitWarmupFingerprint(), the key that
// deduplicates warmup sweeps (in sync and async mode alike). Two sweeps that
// would compile different kernels must never share a fingerprint. No kernel is
// loaded or launched: the kernel objects here carry an empty kernel table.

#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h"

#include <gtest/gtest.h>

#include <memory>

namespace
{

using tensorrt_llm::kernels::Data_type;
using tensorrt_llm::kernels::FmhaKernelType;
using tensorrt_llm::kernels::kSM_100;
using tensorrt_llm::kernels::TllmGenFmhaKernel;

using RunnerParams = TllmGenFmhaKernel::RunnerParams;

// Kernel objects are not copyable; hand them out on the heap.
std::unique_ptr<TllmGenFmhaKernel> makeKernel(Data_type dtypeOut = Data_type::DATA_TYPE_BF16)
{
    return std::make_unique<TllmGenFmhaKernel>(/*pMetaStart=*/nullptr, /*nMetaCount=*/0, Data_type::DATA_TYPE_BF16,
        Data_type::DATA_TYPE_BF16, Data_type::DATA_TYPE_BF16, dtypeOut, kSM_100);
}

// A generation-phase sweep request as the warmup forward issues it.
RunnerParams makeParams()
{
    RunnerParams params{};
    params.mKernelType = FmhaKernelType::Generation;
    params.mHeadDimQk = 128;
    params.mHeadDimV = 128;
    params.mNumHeadsQ = 32;
    params.mNumHeadsKv = 8;
    params.mNumHeadsQPerKv = 4;
    params.mMaxSeqLenQ = 1;
    params.mNumTokensPerPage = 64;
    params.mMaxNumPagesPerSeqKv = 256;
    params.mJITWarmupMaxNumRequests = 64;
    params.mJITWarmupMaxSeqLenQ = 1;
    params.mJITWarmupMaxSeqLenKv = 16384;
    params.mMultiProcessorCount = 148;
    return params;
}

TEST(TllmGenFmhaJitWarmupFingerprint, IsDeterministicForIdenticalRequests)
{
    auto const kernel = makeKernel();
    auto const params = makeParams();
    EXPECT_EQ(kernel->jitWarmupFingerprint(params, 0), kernel->jitWarmupFingerprint(params, 0));
    // A second kernel object with the same configuration derives the same key.
    EXPECT_EQ(makeKernel()->jitWarmupFingerprint(params, 0), kernel->jitWarmupFingerprint(params, 0));
}

TEST(TllmGenFmhaJitWarmupFingerprint, IgnoresPerCallPointersAndStream)
{
    auto const kernel = makeKernel();
    auto const base = makeParams();
    auto withPointers = base;
    withPointers.stream = reinterpret_cast<cudaStream_t>(0x1);
    withPointers.qPtr = reinterpret_cast<void const*>(0x2);
    withPointers.kvPtr = reinterpret_cast<void const*>(0x3);
    EXPECT_EQ(kernel->jitWarmupFingerprint(withPointers, 0), kernel->jitWarmupFingerprint(base, 0));
}

TEST(TllmGenFmhaJitWarmupFingerprint, DistinguishesDevices)
{
    auto const kernel = makeKernel();
    auto const params = makeParams();
    EXPECT_NE(kernel->jitWarmupFingerprint(params, 0), kernel->jitWarmupFingerprint(params, 1));
}

TEST(TllmGenFmhaJitWarmupFingerprint, DistinguishesKernelObjectConfigurations)
{
    auto const params = makeParams();
    EXPECT_NE(makeKernel(Data_type::DATA_TYPE_BF16)->jitWarmupFingerprint(params, 0),
        makeKernel(Data_type::DATA_TYPE_E4M3)->jitWarmupFingerprint(params, 0));
}

TEST(TllmGenFmhaJitWarmupFingerprint, DistinguishesKernelSelectingRunnerParams)
{
    auto const kernel = makeKernel();
    auto const base = makeParams();
    auto const baseKey = kernel->jitWarmupFingerprint(base, 0);

    // Eagle3 one-model: target layers (Q > 1, causal spec-decoding kernel) and
    // draft layers (Q = 1) must get separate sweeps.
    auto specTarget = base;
    specTarget.mMaxSeqLenQ = 5;
    EXPECT_NE(kernel->jitWarmupFingerprint(specTarget, 0), baseKey);

    auto specTree = specTarget;
    specTree.mIsSpecDecTree = true;
    EXPECT_NE(kernel->jitWarmupFingerprint(specTree, 0), kernel->jitWarmupFingerprint(specTarget, 0));

    auto wideGrid = base;
    wideGrid.mJITWarmupMaxSeqLenKv = base.mJITWarmupMaxSeqLenKv * 2;
    EXPECT_NE(kernel->jitWarmupFingerprint(wideGrid, 0), baseKey);

    auto moreHeads = base;
    moreHeads.mNumHeadsQPerKv = 32;
    EXPECT_NE(kernel->jitWarmupFingerprint(moreHeads, 0), baseKey);

    auto sparse = base;
    sparse.mSparseTopK = 2048;
    EXPECT_NE(kernel->jitWarmupFingerprint(sparse, 0), baseKey);
}

TEST(TllmGenFmhaJitWarmupFingerprint, DistinguishesDsv4EpilogueFusion)
{
    // The fused epilogue selects a different kernel (and forces an E4M3
    // output), so a fused and a non-fused sweep with otherwise identical
    // params, on an E4M3-output kernel object, must not deduplicate to one.
    auto const kernel = makeKernel(Data_type::DATA_TYPE_E4M3);
    auto const unfused = makeParams();
    auto fused = unfused;
    fused.mDsv4EpilogueFusion.enabled = true;
    EXPECT_NE(kernel->jitWarmupFingerprint(fused, 0), kernel->jitWarmupFingerprint(unfused, 0));

    // The fusion's per-call buffers do not take part.
    auto fusedWithBuffers = fused;
    fusedWithBuffers.mDsv4EpilogueFusion.cosSinCache = reinterpret_cast<float const*>(0x4);
    EXPECT_EQ(kernel->jitWarmupFingerprint(fusedWithBuffers, 0), kernel->jitWarmupFingerprint(fused, 0));
}

} // namespace
