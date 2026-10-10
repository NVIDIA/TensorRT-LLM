// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Torch binding over TllmGenFmhaRunner for dense separate-QKV context attention.

#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaRunner.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <climits>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>

namespace tk = tensorrt_llm::kernels;

namespace
{

tk::Data_type toDataType(at::ScalarType t)
{
    switch (t)
    {
    case at::kBFloat16: return tk::DATA_TYPE_BF16;
    case at::kHalf: return tk::DATA_TYPE_FP16;
    case at::kFloat8_e4m3fn: return tk::DATA_TYPE_E4M3;
    default: TORCH_CHECK(false, "unsupported dtype ", t);
    }
}

tk::TllmGenFmhaRunner& runnerFor(tk::Data_type dtypeQkv, tk::Data_type dtypeOut)
{
    static std::map<std::pair<int, int>, std::unique_ptr<tk::TllmGenFmhaRunner>> runners;
    auto key = std::make_pair(static_cast<int>(dtypeQkv), static_cast<int>(dtypeOut));
    auto it = runners.find(key);
    if (it == runners.end())
    {
        it = runners.emplace(key, std::make_unique<tk::TllmGenFmhaRunner>(dtypeQkv, dtypeQkv, dtypeQkv, dtypeOut))
                 .first;
    }
    return *it->second;
}

tk::TllmGenFmhaRunnerParams makeParams(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v,
    torch::Tensor const& out, torch::Tensor const& cuSeqLens, torch::Tensor const& seqLens,
    torch::Tensor const& scaleBmm1, torch::Tensor const& scaleBmm2, int64_t batch, int64_t seqLen, bool fp16Softmax,
    bool persistent)
{
    tk::TllmGenFmhaRunnerParams p;
    std::memset(&p, 0, sizeof(p));
    p.mQkvLayout = tk::QkvLayout::SeparateQkv;
    p.mMaskType = tk::TrtllmGenAttentionMaskType::Dense;
    p.mKernelType = tk::FmhaKernelType::Context;
    p.mTileScheduler = persistent ? tk::TileScheduler::Persistent : tk::TileScheduler::Static;
    p.mMultiCtasKvMode = false;
    p.qPtr = q.data_ptr();
    p.kPtr = k.data_ptr();
    p.vPtr = v.data_ptr();
    p.oPtr = out.data_ptr();
    p.cumSeqLensQPtr = cuSeqLens.data_ptr<int>();
    p.cumSeqLensKvPtr = cuSeqLens.data_ptr<int>();
    p.seqLensKvPtr = seqLens.data_ptr<int>();
    // Kernels read both scales from device memory.
    p.scaleSoftmaxLog2Ptr = scaleBmm1.data_ptr<float>() + tk::kIdxScaleSoftmaxLog2Ptr;
    p.outputScalePtr = scaleBmm2.data_ptr<float>();
    int const heads = static_cast<int>(q.size(1));
    int const headDim = static_cast<int>(q.size(2));
    p.mHeadDimQk = headDim;
    p.mHeadDimV = headDim;
    p.mNumHeadsQ = heads;
    p.mNumHeadsKv = heads;
    p.mNumHeadsQPerKv = 1;
    p.mBatchSize = static_cast<int>(batch);
    p.mMaxSeqLenQ = static_cast<int>(seqLen);
    p.mMaxSeqLenKv = static_cast<int>(seqLen);
    p.mSumOfSeqLensQ = static_cast<int>(batch * seqLen);
    p.mSumOfSeqLensKv = static_cast<int>(batch * seqLen);
    p.mChunkedAttentionSize = INT_MAX;
    p.mAttentionWindowSize = INT_MAX;
    p.mScaleQ = 1.0f;
    p.mNumPagesInMemPool = INT_MAX;
    int smCount = 0;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, q.get_device());
    p.mMultiProcessorCount = smCount;
    p.mFp16Softmax = fp16Softmax;
    p.stream = at::cuda::getCurrentCUDAStream(q.get_device());
    return p;
}

} // namespace

// q, k, v [B*S, H, D]; scale_bmm1 {scale, scale * log2e}.
torch::Tensor run_fmha(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor cu_seqlens,
    torch::Tensor seqlens, torch::Tensor scale_bmm1, torch::Tensor scale_bmm2, int64_t batch, int64_t seq_len,
    bool fp16_softmax, bool persistent, std::optional<torch::Tensor> out_opt)
{
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "q/k/v must be contiguous");
    auto out = out_opt.has_value() ? *out_opt : torch::empty(q.sizes(), q.options().dtype(at::kBFloat16));
    auto& runner = runnerFor(toDataType(q.scalar_type()), toDataType(out.scalar_type()));
    auto p = makeParams(
        q, k, v, out, cu_seqlens, seqlens, scale_bmm1, scale_bmm2, batch, seq_len, fp16_softmax, persistent);
    auto [ok, info] = runner.isSupportedWithInfo(p);
    TORCH_CHECK(ok, "TRTLLM-gen FMHA: no kernel for this configuration: ", info);
    runner.run(p);
    return out;
}

bool is_supported(torch::Tensor q, int64_t batch, int64_t seq_len, bool fp16_softmax, bool persistent)
{
    auto out = torch::empty({1}, q.options().dtype(at::kBFloat16));
    auto& runner = runnerFor(toDataType(q.scalar_type()), tk::DATA_TYPE_BF16);
    auto dummyI = torch::zeros({batch + 1}, q.options().dtype(at::kInt));
    auto dummyF = torch::ones({2}, q.options().dtype(at::kFloat));
    auto p = makeParams(q, q, q, out, dummyI, dummyI, dummyF, dummyF, batch, seq_len, fp16_softmax, persistent);
    return runner.isSupported(p);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fmha", &run_fmha, "TRTLLM-gen dense separate-QKV context FMHA", py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("cu_seqlens"), py::arg("seqlens"), py::arg("scale_bmm1"), py::arg("scale_bmm2"), py::arg("batch"),
        py::arg("seq_len"), py::arg("fp16_softmax") = false, py::arg("persistent") = true, py::arg("out") = py::none());
    m.def("is_supported", &is_supported, py::arg("q"), py::arg("batch"), py::arg("seq_len"),
        py::arg("fp16_softmax") = false, py::arg("persistent") = true);
}
