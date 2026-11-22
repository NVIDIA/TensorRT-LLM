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

// Python extension for gen_stage_kv_cubins.py: selects the trtllm-gen FMHA kernel the runner would JIT for an E4M3
// Q/KV, BF16-out paged causal context call with a caller-fixed head dim per K/V stage, and has the generator
// (exported by libtensorrt_llm.so) dump that kernel's source and cubin.

#include <torch/extension.h>

#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaKernels.h"

#include <climits>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace tk = tensorrt_llm::kernels;

namespace
{

tk::TllmGenFmhaRunnerParams makeParams(int64_t numSms, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    int64_t batch, int64_t maxQ, int64_t maxKv, int64_t tokensPerPage, int64_t headDimPerStageKv)
{
    tk::TllmGenFmhaRunnerParams p{};
    p.mQkvLayout = tk::QkvLayout::PagedKv;
    p.mMaskType = tk::TrtllmGenAttentionMaskType::Causal;
    p.mKernelType = tk::FmhaKernelType::Context;
    p.mTileScheduler = tk::TileScheduler::Persistent;
    p.mMultiCtasKvMode = false;
    p.mHeadDimQk = static_cast<int>(headDim);
    p.mHeadDimV = static_cast<int>(headDim);
    p.mNumHeadsQ = static_cast<int>(numHeadsQ);
    p.mNumHeadsKv = static_cast<int>(numHeadsKv);
    p.mNumHeadsQPerKv = static_cast<int>(numHeadsQ / numHeadsKv);
    p.mBatchSize = static_cast<int>(batch);
    p.mMaxSeqLenQ = static_cast<int>(maxQ);
    p.mMaxSeqLenKv = static_cast<int>(maxKv);
    p.mMaxSeqLenCacheKv = static_cast<int>(maxKv);
    p.mSumOfSeqLensQ = static_cast<int>(std::min<int64_t>(batch * maxQ, INT_MAX));
    p.mSumOfSeqLensKv = static_cast<int>(std::min<int64_t>(batch * maxKv, INT_MAX));
    p.mMaxNumPagesPerSeqKv = static_cast<int>((maxKv + tokensPerPage - 1) / tokensPerPage);
    p.mNumTokensPerPage = static_cast<int>(tokensPerPage);
    p.mNumPagesInMemPool = INT_MAX;
    p.mChunkedAttentionSize = INT_MAX;
    p.mLeftSlidingWindow = -1;
    p.mRightSlidingWindow = -1;
    p.mScaleQ = 1.0f;
    p.mMultiProcessorCount = static_cast<int>(numSms);
    p.mHeadDimPerStageKv = static_cast<int>(headDimPerStageKv);
    return p;
}

tk::TllmGenFmhaKernel& kernelFor(int64_t sm)
{
    // No cubin tables: only the selection and the NVRTC config are used.
    static std::map<int64_t, std::unique_ptr<tk::TllmGenFmhaKernel>> kernels;
    auto& k = kernels[sm];
    if (k == nullptr)
    {
        k = std::make_unique<tk::TllmGenFmhaKernel>(nullptr, 0, tk::DATA_TYPE_E4M3, tk::DATA_TYPE_E4M3,
            tk::DATA_TYPE_E4M3, tk::DATA_TYPE_BF16, static_cast<unsigned int>(sm));
    }
    return *k;
}

// The kernel-table fields of the selected config (see TllmGenFmhaKernelMetaInfo), plus its name and JSON config.
py::dict describe(fmha::FmhaConfig const& cfg, std::string const& name)
{
    auto const& o = cfg.mOptions;
    TORCH_CHECK(o.mDtypeQ == trtllm::gen::Dtype::E4m3 && o.mDtypeKv == trtllm::gen::Dtype::E4m3
            && o.mDtypeOut == trtllm::gen::Dtype::Bfloat16,
        "export_kernel: unexpected dtypes");
    py::dict d;
    d["name"] = name;
    d["json"] = cfg.mGenCfgJsonStr;
    d["ctaDim"] = cfg.mCtaDim;
    d["tileSizeQ"] = o.mTileSizeQ;
    d["tileSizeKv"] = o.mTileSizeKv;
    d["stepQ"] = o.mTileSizeQ * o.mNumInstsQ;
    d["stepKv"] = o.mTileSizeKv * o.mNumInstsKv;
    d["headDimPerCtaV"] = o.mHeadDimPerCtaV;
    d["headDimQk"] = o.mHeadDimQk;
    d["headDimV"] = o.mHeadDimV;
    d["headDimPerStageKv"] = o.mHeadDimPerStageKv;
    d["qkvLayout"] = static_cast<int>(o.mQkvLayout);
    d["numTokensPerPage"] = o.mNumTokensPerPage;
    d["maskType"] = static_cast<int>(o.mMaskType);
    d["kernelType"] = static_cast<int>(o.mFmhaKernelType);
    d["tileScheduler"] = static_cast<int>(o.mTileScheduler);
    d["multiCtasKvMode"] = static_cast<int>(o.mMultiCtasKvMode);
    d["groupsHeadsQ"] = o.mGroupsHeadsQ;
    d["groupsTokensHeadsQ"] = o.mGroupsTokensHeadsQ;
    d["reuseSmemKForV"] = o.mReuseSmemKForV;
    d["uses2CtaMma"] = o.mClusterDimX == 2;
    d["sparseAttn"] = static_cast<int>(o.mSparseType);
    d["skipsSoftmax"] = o.mSkipsSoftmaxWhenPossible;
    d["fp16Softmax"] = o.mEnablesFp16Softmax;
    d["usesSpcompress"] = o.mUsesSpcompress;
    d["enablesBf16QFp8KvKOnlyTransform"] = o.mEnablesBf16QFp8KvKOnlyTransform;
    d["separateTransformedKv"] = o.mSeparateTransformedKv;
    d["fusesDsv4InvRopeFp8Quant"] = o.mFusesDsv4InvRopeFp8Quant;
    d["usesDsv4Ue8m0ScaleO"] = o.mUsesDsv4Ue8m0ScaleO;
    d["clusterDimX"] = o.mClusterDimX;
    return d;
}

} // namespace

// The config the runner selects for these shapes, without generating anything.
py::dict selectKernel(int64_t sm, int64_t numSms, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batch,
    int64_t maxQ, int64_t maxKv, int64_t tokensPerPage, int64_t headDimPerStageKv)
{
    auto const params
        = makeParams(numSms, numHeadsQ, numHeadsKv, headDim, batch, maxQ, maxKv, tokensPerPage, headDimPerStageKv);
    auto& kernel = kernelFor(sm);
    fmha::FmhaConfig cfg = kernel.nvrtcConfig(params);
    fmha::FmhaInterface const fmhaInterface{false, 1};
    return describe(cfg, fmhaInterface.getKernelNameFromConfigs(cfg));
}

// Generates and compiles the selected kernel with source and cubin export on. execPath stands in for the runner's
// exec path (the fmha build dir with the kernel headers); the generator writes <name>.nvrtc.cu / .nvrtc.cubin there.
py::dict generateKernel(int64_t sm, int64_t numSms, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    int64_t batch, int64_t maxQ, int64_t maxKv, int64_t tokensPerPage, int64_t headDimPerStageKv,
    std::string const& execPath)
{
    auto const params
        = makeParams(numSms, numHeadsQ, numHeadsKv, headDim, batch, maxQ, maxKv, tokensPerPage, headDimPerStageKv);
    auto& kernel = kernelFor(sm);
    fmha::FmhaConfig cfg = kernel.nvrtcConfig(params);
    // Generator switches that only write the NVRTC source and cubin next to the compile; the kernel is unchanged.
    std::string const open = "{\n";
    TORCH_CHECK(cfg.mGenCfgJsonStr.rfind(open, 0) == 0, "export_kernel: unexpected generator config");
    cfg.mGenCfgJsonStr.insert(open.size(), "\"emitsNvRtcSource\": true,\n\"emitsCubin\": true,\n");
    // The generator reads the kernel headers from, and writes its dumps to, the exec path.
    cfg.mExecPath = execPath.c_str();
    fmha::FmhaInterface const fmhaInterface{/*exportsCubin=*/true, /*numRotations=*/1, /*verbose=*/true};
    fmhaInterface.generateAndCompileKernel(cfg);
    py::dict d = describe(cfg, cfg.mFunctionName);
    return d;
}

// The standard-header stand-ins the generator's NVRTC front end (jitify2) compiles with, read from its function-local
// map in libtensorrt_llm.so once a generation has initialized it.
py::dict jitifyHeaders(uintptr_t mapAddress)
{
    TORCH_CHECK(mapAddress != 0, "export_kernel: null jitify2 header map address");
    auto const& headers = *reinterpret_cast<std::unordered_map<std::string, std::string> const*>(mapAddress);
    TORCH_CHECK(!headers.empty(), "export_kernel: the jitify2 header map is empty");
    py::dict d;
    for (auto const& [name, source] : headers)
    {
        d[py::str(name)] = py::bytes(source);
    }
    return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("jitify_headers", &jitifyHeaders, "Read the generator's built-in NVRTC header map at the given address");
    m.def("select_kernel", &selectKernel, "Select the trtllm-gen FMHA context kernel config the runner would JIT");
    m.def("generate_kernel", &generateKernel, "Generate the selected kernel with source and cubin export");
}
