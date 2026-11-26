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
// Torch ops of the target's CUDA kernels: tensors in, outputs written in place, launches on the current stream.
// The kernels are compiled for fixed shapes and read their operands with fixed strides (some through TMA bulk copies
// or 16-byte accesses), so every operand's device, dtype, sizes, contiguity and, where it matters, alignment is
// checked on the host before the launch.
#include "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/csrc/decodeMoe.h"
#include "glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/csrc/mhcArBoundary.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

//! The alignment TMA bulk copies and 16-byte vector accesses need.
constexpr uintptr_t kAccessAlignment = 16;

//! Whether an operand is read by TMA bulk copies or written in 16-byte stores, so must start 16-byte aligned.
enum class Alignment
{
    kAny,
    k16Bytes,
};

//! Checks that `tensor`, the argument `name` of op `op`, is a contiguous `dtype` tensor of sizes `sizes` on `device`,
//! starting on a 16-byte boundary when `alignment` asks for it.
void checkOperand(char const* op, char const* name, at::Tensor const& tensor, at::Device device, at::ScalarType dtype,
    at::IntArrayRef sizes, Alignment alignment)
{
    TORCH_CHECK(tensor.device() == device, op, ": ", name, " must be on ", device, ", got ", tensor.device());
    TORCH_CHECK(tensor.scalar_type() == dtype, op, ": ", name, " must be ", dtype, ", got ", tensor.scalar_type());
    TORCH_CHECK(tensor.sizes() == sizes, op, ": ", name, " must have sizes ", sizes, ", got ", tensor.sizes());
    TORCH_CHECK(tensor.is_contiguous(), op, ": ", name, " must be contiguous");
    TORCH_CHECK(alignment == Alignment::kAny || reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kAccessAlignment == 0,
        op, ": ", name, " must start on a 16-byte boundary");
}

void decodeMoe(at::Tensor const& x, at::Tensor const& logits, at::Tensor const& bias, double scaling, double limit,
    at::Tensor const& w31, at::Tensor const& s31, at::Tensor const& w2, at::Tensor const& s2, at::Tensor const& sw31,
    at::Tensor const& ss31, at::Tensor const& sw2, at::Tensor const& ss2, at::Tensor& out, bool pdl)
{
    char const* const op = "decode_moe";
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= kernels::kDecodeMoeMaxRows,
        "decode_moe: x must be a CUDA [rows, hidden] tensor of 1 to ", kernels::kDecodeMoeMaxRows, " rows");
    int64_t const rows = x.size(0);
    int64_t const hidden = kernels::kDecodeMoeHidden;
    int64_t const inter = kernels::kDecodeMoeInter;
    int64_t const experts = kernels::kDecodeMoeExperts;
    int64_t const block = kernels::kDecodeMoeScaleBlock;
    at::Device const device = x.device();
    auto const fp8 = at::kFloat8_e4m3fn;
    checkOperand(op, "x", x, device, at::kBFloat16, {rows, hidden}, Alignment::k16Bytes);
    checkOperand(op, "logits", logits, device, at::kFloat, {rows, experts}, Alignment::kAny);
    checkOperand(op, "bias", bias, device, at::kFloat, {experts}, Alignment::kAny);
    checkOperand(op, "w31", w31, device, fp8, {experts, 2 * inter, hidden}, Alignment::k16Bytes);
    checkOperand(op, "s31", s31, device, at::kFloat, {experts, 2 * inter / block, hidden / block}, Alignment::k16Bytes);
    checkOperand(op, "w2", w2, device, fp8, {experts, hidden, inter}, Alignment::k16Bytes);
    checkOperand(op, "s2", s2, device, at::kFloat, {experts, hidden / block, inter / block}, Alignment::k16Bytes);
    checkOperand(op, "sw31", sw31, device, fp8, {2 * inter, hidden}, Alignment::k16Bytes);
    checkOperand(op, "ss31", ss31, device, at::kFloat, {2 * inter / block, hidden / block}, Alignment::k16Bytes);
    checkOperand(op, "sw2", sw2, device, fp8, {hidden, inter}, Alignment::k16Bytes);
    checkOperand(op, "ss2", ss2, device, at::kFloat, {hidden / block, inter / block}, Alignment::k16Bytes);
    checkOperand(op, "out", out, device, at::kBFloat16, {rows, hidden}, Alignment::kAny);

    c10::cuda::CUDAGuard const deviceGuard{device};
    kernels::DecodeMoeArgs args{};
    args.x = x.data_ptr();
    args.logits = logits.data_ptr<float>();
    args.bias = bias.data_ptr<float>();
    args.w31 = static_cast<uint8_t const*>(w31.data_ptr());
    args.s31 = s31.data_ptr<float>();
    args.w2 = static_cast<uint8_t const*>(w2.data_ptr());
    args.s2 = s2.data_ptr<float>();
    args.sw31 = static_cast<uint8_t const*>(sw31.data_ptr());
    args.ss31 = ss31.data_ptr<float>();
    args.sw2 = static_cast<uint8_t const*>(sw2.data_ptr());
    args.ss2 = ss2.data_ptr<float>();
    args.out = out.data_ptr();
    args.scaling = static_cast<float>(scaling);
    args.limit = static_cast<float>(limit);
    args.rows = static_cast<int>(rows);
    args.pdl = pdl;
    cudaError_t const err = kernels::decodeMoe(args, at::cuda::getCurrentCUDAStream(device.index()));
    TORCH_CHECK(err != cudaErrorCooperativeLaunchTooLarge,
        "decode_moe: the kernel needs all of its CTAs resident at once (one per SM), and this device cannot hold its "
        "grid; it would hang rather than fail");
    TORCH_CHECK(err == cudaSuccess, "decode_moe launch failed: ", cudaGetErrorString(err));
}

void mhcArBoundary(at::Tensor const& partial, at::Tensor const& workspace, int64_t rank, at::Tensor const& residual,
    at::Tensor const& post, at::Tensor const& comb, at::Tensor const& fn, at::Tensor const& scale,
    at::Tensor const& base, at::Tensor const& normWeight, int64_t sinkhornIters, double normEps, double hcEps,
    double sinkhornEps, double postMult, double layerInputEps, at::Tensor& residualOut, at::Tensor& postOut,
    at::Tensor& combOut, at::Tensor& layerInput, bool pdl)
{
    char const* const op = "mhc_ar_boundary";
    TORCH_CHECK(partial.is_cuda() && partial.dim() == 2 && partial.size(0) >= 1
            && partial.size(0) <= kernels::kMhcArBoundaryMaxRows,
        "mhc_ar_boundary: partial must be a CUDA [rows, hidden] tensor of 1 to ", kernels::kMhcArBoundaryMaxRows,
        " rows");
    TORCH_CHECK(rank >= 0 && rank < kernels::kMhcArBoundaryRanks, "mhc_ar_boundary: rank must be in [0, ",
        kernels::kMhcArBoundaryRanks, "), got ", rank);
    TORCH_CHECK(sinkhornIters >= 1, "mhc_ar_boundary: sinkhorn_iters must be at least 1, got ", sinkhornIters);
    int64_t const rows = partial.size(0);
    int64_t const hidden = kernels::kMhcArBoundaryHidden;
    int64_t const mult = kernels::kMhcArBoundaryMult;
    int64_t const mix = kernels::kMhcArBoundaryMix;
    int64_t const numScales = 3; // the mixer's scales of pre, post and comb
    at::Device const device = partial.device();
    // The workspace is the pointer table of the group's all-reduce workspace, one int64 per pointer.
    checkOperand(
        op, "workspace", workspace, device, at::kLong, {kernels::kMhcArBoundaryWorkspacePointers}, Alignment::kAny);
    checkOperand(op, "partial", partial, device, at::kBFloat16, {rows, hidden}, Alignment::k16Bytes);
    checkOperand(op, "residual", residual, device, at::kBFloat16, {rows, mult, hidden}, Alignment::k16Bytes);
    checkOperand(op, "post", post, device, at::kFloat, {rows, mult}, Alignment::k16Bytes);
    checkOperand(op, "comb", comb, device, at::kFloat, {rows, mult, mult}, Alignment::k16Bytes);
    checkOperand(op, "fn", fn, device, at::kFloat, {mix, mult * hidden}, Alignment::k16Bytes);
    checkOperand(op, "scale", scale, device, at::kFloat, {numScales}, Alignment::kAny);
    checkOperand(op, "base", base, device, at::kFloat, {mix}, Alignment::kAny);
    checkOperand(op, "norm_weight", normWeight, device, at::kBFloat16, {hidden}, Alignment::kAny);
    checkOperand(op, "residual_out", residualOut, device, at::kBFloat16, {rows, mult, hidden}, Alignment::k16Bytes);
    checkOperand(op, "post_out", postOut, device, at::kFloat, {rows, mult}, Alignment::kAny);
    checkOperand(op, "comb_out", combOut, device, at::kFloat, {rows, mult, mult}, Alignment::kAny);
    checkOperand(op, "layer_input", layerInput, device, at::kBFloat16, {rows, hidden}, Alignment::k16Bytes);

    c10::cuda::CUDAGuard const deviceGuard{device};
    kernels::MhcArBoundaryArgs args{};
    args.workspace = static_cast<void**>(workspace.data_ptr());
    args.rank = static_cast<int>(rank);
    args.partial = partial.data_ptr();
    args.residual = residual.data_ptr();
    args.postPrev = post.data_ptr<float>();
    args.combPrev = comb.data_ptr<float>();
    args.fn = fn.data_ptr<float>();
    args.scale = scale.data_ptr<float>();
    args.base = base.data_ptr<float>();
    args.normWeight = normWeight.data_ptr();
    args.residualOut = residualOut.data_ptr();
    args.postOut = postOut.data_ptr<float>();
    args.combOut = combOut.data_ptr<float>();
    args.layerInput = layerInput.data_ptr();
    args.rows = static_cast<int>(rows);
    args.sinkhornIters = static_cast<int>(sinkhornIters);
    args.normEps = static_cast<float>(normEps);
    args.hcEps = static_cast<float>(hcEps);
    args.sinkhornEps = static_cast<float>(sinkhornEps);
    args.postMult = static_cast<float>(postMult);
    args.layerInputEps = static_cast<float>(layerInputEps);
    args.pdl = pdl;
    cudaError_t const err = kernels::mhcArBoundary(args, at::cuda::getCurrentCUDAStream(device.index()));
    TORCH_CHECK(err == cudaSuccess, "mhc_ar_boundary launch failed: ", cudaGetErrorString(err));
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "decode_moe(Tensor x, Tensor logits, Tensor bias, float scaling, float limit, Tensor w31, Tensor s31, "
        "Tensor w2, Tensor s2, Tensor sw31, Tensor ss31, Tensor sw2, Tensor ss2, Tensor(a!) out, bool pdl) -> ()");
    m.def(
        "mhc_ar_boundary(Tensor partial, Tensor workspace, int rank, Tensor residual, Tensor post, Tensor comb, "
        "Tensor fn, Tensor scale, Tensor base, Tensor norm_weight, int sinkhorn_iters, float norm_eps, "
        "float hc_eps, float sinkhorn_eps, float post_mult, float layer_input_eps, Tensor(a!) residual_out, "
        "Tensor(b!) post_out, Tensor(c!) comb_out, Tensor(d!) layer_input, bool pdl) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("decode_moe", &TRTLLM_MV2_NS::decodeMoe);
    m.impl("mhc_ar_boundary", &TRTLLM_MV2_NS::mhcArBoundary);
}
