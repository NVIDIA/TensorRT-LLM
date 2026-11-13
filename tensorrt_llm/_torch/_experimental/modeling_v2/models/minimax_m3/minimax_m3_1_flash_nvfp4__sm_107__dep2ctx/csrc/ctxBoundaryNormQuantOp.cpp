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

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep2ctx/csrc/ctxBoundaryNormQuantSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace tk = TRTLLM_MV2_NS::kernels::minimaxM31;

bool aligned16(torch::Tensor const& t)
{
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

int64_t sfBytes(int64_t rows, int64_t hidden)
{
    return (rows + 127) / 128 * 128 * ((hidden / 32 + 3) / 4 * 4);
}

//! inter / residual / weight checks shared by both ops; returns (rows, hidden).
std::pair<int64_t, int64_t> checkInputs(char const* op, torch::Tensor const& inter,
    std::optional<torch::Tensor> const& residual, torch::Tensor const& weight)
{
    TORCH_CHECK(inter.is_cuda() && inter.is_contiguous() && inter.scalar_type() == at::kBFloat16 && inter.dim() >= 2
            && aligned16(inter),
        op, ": inter must be a contiguous 16-byte-aligned bf16 CUDA tensor of rank >= 2");
    if (residual.has_value())
    {
        TORCH_CHECK(residual->is_contiguous() && residual->scalar_type() == at::kBFloat16
                && residual->sizes() == inter.sizes() && residual->device() == inter.device() && aligned16(*residual),
            op, ": residual must be a contiguous 16-byte-aligned bf16 tensor shaped like inter");
    }
    int64_t const hidden = inter.size(-1);
    TORCH_CHECK(tk::ctxBoundaryNormQuantSupported(static_cast<int>(hidden)), op,
        ": hidden must be a multiple of 256 in [256, 6144], got ", hidden);
    int64_t const rows = inter.numel() / hidden;
    TORCH_CHECK(weight.is_contiguous() && weight.scalar_type() == at::kBFloat16 && weight.numel() == hidden
            && weight.device() == inter.device() && aligned16(weight),
        op, ": weight must be a contiguous bf16 [hidden] on inter's device");
    TORCH_CHECK(rows <= INT32_MAX / 2, op, ": too many rows");
    return {rows, hidden};
}

void checkOut(char const* op, torch::Tensor const& t, int64_t bytes, char const* what, at::Device device)
{
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device() == device, op, ": ", what,
        " must be a contiguous CUDA tensor on inter's device");
    TORCH_CHECK(t.numel() * t.element_size() == bytes, op, ": ", what, " holds ", t.numel() * t.element_size(),
        " bytes, expected ", bytes);
    TORCH_CHECK(aligned16(t), op, ": ", what, " must be 16-byte aligned");
}

//! Row stride in bytes of an NVFP4 data / scale destination of `rowBytes` bytes per row: a 2-D [>= rows, rowBytes]
//! byte tensor with unit column stride (e.g. a MegaMoE stage's rows) or a 1-D contiguous one of >= rows * rowBytes.
int64_t rowStride(
    char const* op, torch::Tensor const& t, int64_t rows, int64_t rowBytes, char const* what, at::Device device)
{
    TORCH_CHECK(t.is_cuda() && t.device() == device && t.element_size() == 1, op, ": ", what,
        " must be a 1-byte CUDA tensor on inter's device");
    if (t.dim() == 1)
    {
        TORCH_CHECK(t.is_contiguous() && t.numel() >= rows * rowBytes, op, ": ", what, " holds too few bytes");
        return rowBytes;
    }
    TORCH_CHECK(t.dim() == 2 && t.size(1) == rowBytes && t.stride(1) == 1 && t.size(0) >= rows
            && (rows <= 1 || t.stride(0) >= rowBytes),
        op, ": ", what, " must be [>= rows, ", rowBytes, "] with unit column stride and non-overlapping rows");
    return t.stride(0);
}

void launch(char const* op, tk::CtxBoundaryNormQuantParams const& p, bool enablePdl, torch::Tensor const& inter)
{
    cudaError_t const e
        = tk::invokeCtxBoundaryNormQuant(p, enablePdl, at::cuda::getCurrentCUDAStream(inter.get_device()));
    TORCH_CHECK(e != cudaErrorNotSupported, op, ": needs an sm_10x device (and hidden in [256, 6144])");
    TORCH_CHECK(e == cudaSuccess, op, " launch: ", cudaGetErrorString(e));
}

} // namespace

//! DEP2CTX_NORMQ_V2: minimax_m31_ctx_residual_rmsnorm_mxfp8 / its _f32 fold on the ring kernel.
//! Returns (norm_out, mxfp8, sf, fp4, fp4_sf, f32), each bit for bit the served op's field (norm_out empty unless
//! write_norm, f32 empty unless write_f32, fp4 / fp4_sf empty unless fp4_global_scale). Unlike the served _f32 op,
//! write_f32 does not need write_norm. inter becomes inter + residual in place.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
minimaxM31CtxBoundaryNormQ(torch::Tensor& inter, torch::Tensor const& residual, torch::Tensor const& weight, double eps,
    bool writeNorm, bool writeF32, std::optional<torch::Tensor> const& fp4GlobalScale, bool enablePdl)
{
    char const* op = "minimax_m31_ctx_boundary_normq";
    auto const [rows, hidden] = checkInputs(op, inter, residual, weight);
    at::cuda::CUDAGuard const guard(inter.device());
    auto const opts = inter.options();
    torch::Tensor const empty = torch::empty({0}, opts);
    torch::Tensor normOut = writeNorm ? torch::empty_like(inter) : empty;
    torch::Tensor qOut = torch::empty(inter.sizes(), opts.dtype(at::ScalarType::Float8_e4m3fn));
    torch::Tensor sfOut = torch::empty({sfBytes(rows, hidden)}, opts.dtype(at::ScalarType::Byte));
    bool const fp4 = fp4GlobalScale.has_value();
    if (fp4)
    {
        TORCH_CHECK(fp4GlobalScale->is_cuda() && fp4GlobalScale->scalar_type() == at::kFloat
                && fp4GlobalScale->numel() >= 1 && fp4GlobalScale->device() == inter.device(),
            op, ": fp4_global_scale must be fp32 with at least one element on inter's device");
    }
    std::vector<int64_t> fp4Shape(inter.sizes().begin(), inter.sizes().end());
    fp4Shape.back() = hidden / 2;
    torch::Tensor fp4Out = fp4 ? torch::empty(fp4Shape, opts.dtype(at::ScalarType::Byte)) : empty;
    torch::Tensor fp4SfOut = fp4 ? torch::empty({rows * (hidden / 16)}, opts.dtype(at::ScalarType::Byte)) : empty;
    torch::Tensor f32Out = writeF32 ? torch::empty(inter.sizes(), opts.dtype(at::ScalarType::Float)) : empty;
    tk::CtxBoundaryNormQuantParams p;
    p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
    p.residual = reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr());
    p.weight = reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
    p.normOut = writeNorm ? reinterpret_cast<__nv_bfloat16*>(normOut.data_ptr()) : nullptr;
    p.qOut = reinterpret_cast<uint64_t*>(qOut.data_ptr());
    p.sfOut = sfOut.data_ptr<uint8_t>();
    p.f32Out = writeF32 ? f32Out.data_ptr<float>() : nullptr;
    if (fp4)
    {
        p.fp4Out = reinterpret_cast<uint32_t*>(fp4Out.data_ptr());
        p.fp4SfOut = fp4SfOut.data_ptr<uint8_t>();
        p.fp4GlobalScale = fp4GlobalScale->data_ptr<float>();
        p.fp4StrideWords = hidden / 8;
        p.fp4SfStride = hidden / 16;
    }
    p.rows = static_cast<int>(rows);
    p.hidden = static_cast<int>(hidden);
    p.eps = static_cast<float>(eps);
    launch(op, p, enablePdl, inter);
    return {normOut, qOut, sfOut, fp4Out, fp4SfOut, f32Out};
}

//! DEP2CTX_NORMQ_V2 / DEP2CTX_NORMQ_STAGE: the boundary writing into caller buffers.
//! - residual set: inter += residual in place; q_out (rows * hidden bytes) and sf_out (the swizzled scale block of
//!   ceil(rows / 128) * 128 rows) are required; norm_out / f32_out optional (f32 without norm is allowed);
//!   fp4_out / fp4_sf_out with fp4_global_scale write fp4_quantize(norm, scale, 16, False, False) of the rows into
//!   1-D buffers or 2-D row-strided ones (a MegaMoE input stage: [>= rows, hidden / 2] and [>= rows, hidden / 16],
//!   unit column stride, 16-byte-aligned scale rows); stage_ids / stage_weights with stage_topk > 0 also write the
//!   stage's padded tail (rows [rows, size(0)), first stage_topk columns: -1 / 0), as
//!   noaux_tc_fp4_quantize_stage_out does.
//! - residual None: inter already holds inter + residual (a boundary that ran with norm_out None); only norm_out is
//!   written, the BF16 norm that boundary would have written, bit for bit.
void minimaxM31CtxBoundaryNormQOut(torch::Tensor& inter, std::optional<torch::Tensor> const& residual,
    torch::Tensor const& weight, double eps, std::optional<torch::Tensor> normOut, std::optional<torch::Tensor> qOut,
    std::optional<torch::Tensor> sfOut, std::optional<torch::Tensor> f32Out,
    std::optional<torch::Tensor> const& fp4GlobalScale, std::optional<torch::Tensor> fp4Out,
    std::optional<torch::Tensor> fp4SfOut, std::optional<torch::Tensor> stageIds,
    std::optional<torch::Tensor> stageWeights, int64_t stageTopk, bool enablePdl)
{
    char const* op = "minimax_m31_ctx_boundary_normq_out";
    auto const [rows, hidden] = checkInputs(op, inter, residual, weight);
    at::Device const device = inter.device();
    tk::CtxBoundaryNormQuantParams p;
    p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
    p.weight = reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
    p.rows = static_cast<int>(rows);
    p.hidden = static_cast<int>(hidden);
    p.eps = static_cast<float>(eps);
    if (normOut.has_value())
    {
        TORCH_CHECK(normOut->scalar_type() == at::kBFloat16, op, ": norm_out must be bfloat16");
        checkOut(op, *normOut, rows * hidden * 2, "norm_out", device);
        p.normOut = reinterpret_cast<__nv_bfloat16*>(normOut->data_ptr());
    }
    if (!residual.has_value())
    {
        TORCH_CHECK(normOut.has_value() && !qOut.has_value() && !sfOut.has_value() && !f32Out.has_value()
                && !fp4Out.has_value() && !fp4SfOut.has_value() && !stageIds.has_value() && !stageWeights.has_value(),
            op, ": without residual only norm_out is written (the rows already hold the sum)");
        at::cuda::CUDAGuard const guard(device);
        launch(op, p, enablePdl, inter);
        return;
    }
    p.residual = reinterpret_cast<__nv_bfloat16 const*>(residual->data_ptr());
    TORCH_CHECK(qOut.has_value() && sfOut.has_value(), op, ": q_out and sf_out are required with residual");
    checkOut(op, *qOut, rows * hidden, "q_out", device);
    checkOut(op, *sfOut, sfBytes(rows, hidden), "sf_out", device);
    p.qOut = reinterpret_cast<uint64_t*>(qOut->data_ptr());
    p.sfOut = reinterpret_cast<uint8_t*>(sfOut->data_ptr());
    if (f32Out.has_value())
    {
        TORCH_CHECK(f32Out->scalar_type() == at::kFloat, op, ": f32_out must be float32");
        checkOut(op, *f32Out, rows * hidden * 4, "f32_out", device);
        p.f32Out = f32Out->data_ptr<float>();
    }
    bool const fp4 = fp4GlobalScale.has_value();
    TORCH_CHECK(fp4 == fp4Out.has_value() && fp4 == fp4SfOut.has_value(), op,
        ": fp4_out / fp4_sf_out are given iff fp4_global_scale is");
    if (fp4)
    {
        TORCH_CHECK(fp4GlobalScale->is_cuda() && fp4GlobalScale->scalar_type() == at::kFloat
                && fp4GlobalScale->numel() >= 1 && fp4GlobalScale->device() == device,
            op, ": fp4_global_scale must be fp32 with at least one element on inter's device");
        int64_t const fp4Stride = rowStride(op, *fp4Out, rows, hidden / 2, "fp4_out", device);
        TORCH_CHECK(fp4Stride % 4 == 0 && reinterpret_cast<uintptr_t>(fp4Out->data_ptr()) % 4 == 0, op,
            ": fp4_out rows must be 4-byte aligned");
        int64_t const sfStride = rowStride(op, *fp4SfOut, rows, hidden / 16, "fp4_sf_out", device);
        TORCH_CHECK(sfStride % 16 == 0 && aligned16(*fp4SfOut), op, ": fp4_sf_out rows must be 16-byte aligned");
        p.fp4Out = reinterpret_cast<uint32_t*>(fp4Out->data_ptr());
        p.fp4SfOut = reinterpret_cast<uint8_t*>(fp4SfOut->data_ptr());
        p.fp4GlobalScale = fp4GlobalScale->data_ptr<float>();
        p.fp4StrideWords = fp4Stride / 4;
        p.fp4SfStride = sfStride;
    }
    if (stageTopk > 0)
    {
        TORCH_CHECK(
            stageIds.has_value() && stageWeights.has_value(), op, ": stage_topk > 0 needs stage_ids and stage_weights");
        for (torch::Tensor const* t : {&*stageIds, &*stageWeights})
        {
            TORCH_CHECK(t->is_cuda() && t->device() == device && t->dim() == 2 && t->stride(1) == 1
                    && t->size(0) >= rows && t->size(1) >= stageTopk && t->size(0) <= INT32_MAX
                    && (t->size(0) <= 1 || t->stride(0) >= stageTopk),
                op,
                ": stage tensors must be 2-D, unit column stride, >= rows rows and >= stage_topk columns, rows not "
                "overlapping");
        }
        TORCH_CHECK(stageIds->scalar_type() == at::kInt && stageWeights->scalar_type() == at::kFloat, op,
            ": int32 stage_ids, fp32 stage_weights");
        p.tailIds = stageIds->data_ptr<int32_t>();
        p.tailWeights = stageWeights->data_ptr<float>();
        p.tailIdsStride = stageIds->stride(0);
        p.tailWeightsStride = stageWeights->stride(0);
        p.tailIdsRows = static_cast<int>(stageIds->size(0));
        p.tailWeightsRows = static_cast<int>(stageWeights->size(0));
        p.tailTopk = static_cast<int>(stageTopk);
    }
    at::cuda::CUDAGuard const guard(device);
    launch(op, p, enablePdl, inter);
}

//! DEP2CTX_TOPK_FOLD (K5 U2): the next-layer boundary of a MoE layer whose MegaMoE launch deferred its top-k reduce.
//! Equals MegaMoEDeferredTopkReduce(plane, scores).add_into(shared) followed by
//! minimax_m31_ctx_boundary_normq(shared, residual, weight, eps, write_norm) bit for bit: shared becomes
//! moe + residual in place (moe = bf16(shared + bf16(sum_k plane[:, k] * scores[:, k]))) and the returned
//! (norm_out, mxfp8, sf) are that op's first three outputs (norm_out empty unless write_norm). plane is the
//! [rows, 4, hidden] BF16 plane, scores the [rows, 4] FP32 weights; every tensor contiguous and 16-byte aligned.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> minimaxM31CtxBoundaryNormQTopk(torch::Tensor& shared,
    torch::Tensor const& plane, torch::Tensor const& scores, torch::Tensor const& residual, torch::Tensor const& weight,
    double eps, bool writeNorm, bool enablePdl)
{
    char const* op = "minimax_m31_ctx_boundary_normq_topk";
    auto const [rows, hidden] = checkInputs(op, shared, residual, weight);
    TORCH_CHECK(shared.dim() == 2, op, ": shared must be [rows, hidden]");
    TORCH_CHECK(plane.is_cuda() && plane.device() == shared.device() && plane.is_contiguous()
            && plane.scalar_type() == at::kBFloat16 && plane.dim() == 3 && plane.size(0) == rows
            && plane.size(2) == hidden && aligned16(plane),
        op, ": plane must be a contiguous 16-byte-aligned bf16 [rows, K, hidden] tensor on shared's device");
    int64_t const topk = plane.size(1);
    TORCH_CHECK(topk == 4, op, ": the fold reduces top-4 planes, got K = ", topk);
    TORCH_CHECK(scores.is_cuda() && scores.device() == shared.device() && scores.is_contiguous()
            && scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.size(0) == rows
            && scores.size(1) == topk && aligned16(scores),
        op, ": scores must be a contiguous 16-byte-aligned fp32 [rows, K] tensor on shared's device");
    at::cuda::CUDAGuard const guard(shared.device());
    auto const opts = shared.options();
    torch::Tensor normOut = writeNorm ? torch::empty_like(shared) : torch::empty({0}, opts);
    torch::Tensor qOut = torch::empty(shared.sizes(), opts.dtype(at::ScalarType::Float8_e4m3fn));
    torch::Tensor sfOut = torch::empty({sfBytes(rows, hidden)}, opts.dtype(at::ScalarType::Byte));
    tk::CtxBoundaryTopkNormQuantParams p;
    p.shared = reinterpret_cast<__nv_bfloat16*>(shared.data_ptr());
    p.plane = reinterpret_cast<__nv_bfloat16 const*>(plane.data_ptr());
    p.scores = scores.data_ptr<float>();
    p.topk = static_cast<int>(topk);
    p.residual = reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr());
    p.weight = reinterpret_cast<__nv_bfloat16 const*>(weight.data_ptr());
    p.normOut = writeNorm ? reinterpret_cast<__nv_bfloat16*>(normOut.data_ptr()) : nullptr;
    p.qOut = reinterpret_cast<uint64_t*>(qOut.data_ptr());
    p.sfOut = sfOut.data_ptr<uint8_t>();
    p.rows = static_cast<int>(rows);
    p.hidden = static_cast<int>(hidden);
    p.eps = static_cast<float>(eps);
    cudaError_t const e
        = tk::invokeCtxBoundaryTopkNormQuant(p, enablePdl, at::cuda::getCurrentCUDAStream(shared.get_device()));
    TORCH_CHECK(e != cudaErrorNotSupported, op,
        ": needs an sm_10x device whose per-block shared-memory opt-in holds the three-stage ring (and hidden in "
        "[256, 6144])");
    TORCH_CHECK(e == cudaSuccess, op, " launch: ", cudaGetErrorString(e));
    return {normOut, qOut, sfOut};
}

namespace
{

//! Launch parameters of a flavor on rows shaped like inter (rows x hidden), with placeholder pointers: for the shape
//! queries only (nothing is launched or read).
tk::CtxBoundaryNormQuantParams shapeParams(
    char const* op, torch::Tensor const& inter, bool add, bool writeNorm, bool writeF32, bool fp4)
{
    TORCH_CHECK(inter.is_cuda() && inter.dim() >= 2, op, ": a CUDA [rows, hidden]");
    int64_t const hidden = inter.size(-1);
    alignas(16) static uint64_t dummy[2];
    tk::CtxBoundaryNormQuantParams p;
    p.rows = static_cast<int>(inter.numel() / std::max<int64_t>(hidden, 1));
    p.hidden = static_cast<int>(hidden);
    auto* any = reinterpret_cast<__nv_bfloat16*>(dummy);
    p.inter = any;
    p.weight = any;
    p.residual = add ? any : nullptr;
    p.normOut = writeNorm ? any : nullptr;
    p.qOut = add ? dummy : nullptr;
    p.sfOut = add ? reinterpret_cast<uint8_t*>(dummy) : nullptr;
    p.f32Out = writeF32 ? reinterpret_cast<float*>(dummy) : nullptr;
    if (fp4)
    {
        p.fp4Out = reinterpret_cast<uint32_t*>(dummy);
        p.fp4SfOut = reinterpret_cast<uint8_t*>(dummy);
        p.fp4GlobalScale = reinterpret_cast<float const*>(dummy);
        p.fp4SfStride = hidden / 16;
    }
    return p;
}

} // namespace

//! (CTAs per SM, grid, dynamic shared memory bytes) the boundary launch would use on rows shaped like inter (rows x
//! hidden, on its device) with these flags; nothing is launched or read. The UT checks it against the pure-Python
//! plan (ctx_boundary_plan.py).
std::tuple<int64_t, int64_t, int64_t> minimaxM31CtxBoundaryNormQShape(
    torch::Tensor const& inter, bool add, bool writeNorm, bool writeF32, bool fp4)
{
    char const* op = "minimax_m31_ctx_boundary_normq_shape";
    tk::CtxBoundaryNormQuantParams const p = shapeParams(op, inter, add, writeNorm, writeF32, fp4);
    int ctas = 0;
    int grid = 0;
    int smem = 0;
    at::cuda::CUDAGuard const guard(inter.device());
    cudaError_t const e = tk::ctxBoundaryNormQuantLaunchShape(p, &ctas, &grid, &smem);
    TORCH_CHECK(e == cudaSuccess, op, ": ", cudaGetErrorString(e));
    return {ctas, grid, smem};
}

//! (chunks per thread, CTAs per SM, grid, dynamic shared memory bytes, ring depth) of that launch: chunks 2 is the
//! CBNQ_P2 kernel (hidden / 16 threads), 1 the hidden / 8-thread kernel. For the UT and the MB receipts.
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> minimaxM31CtxBoundaryNormQForm(
    torch::Tensor const& inter, bool add, bool writeNorm, bool writeF32, bool fp4)
{
    char const* op = "minimax_m31_ctx_boundary_normq_form";
    tk::CtxBoundaryNormQuantParams const p = shapeParams(op, inter, add, writeNorm, writeF32, fp4);
    int chunks = 0;
    int ctas = 0;
    int grid = 0;
    int smem = 0;
    int stages = 0;
    at::cuda::CUDAGuard const guard(inter.device());
    cudaError_t const e = tk::ctxBoundaryNormQuantLaunchForm(p, &chunks, &ctas, &grid, &smem, &stages);
    TORCH_CHECK(e == cudaSuccess, op, ": ", cudaGetErrorString(e));
    return {chunks, ctas, grid, smem, stages};
}

//! (dual, CTAs per SM, grid, dynamic shared memory bytes) of the top-k fold's launch on rows shaped like shared (rows
//! x hidden): dual 1 is the CBNQ_TOPK_DUAL kernel (two row groups per CTA), 0 the one-group fold. Nothing is launched
//! or read. For the UT and the MB receipts.
std::tuple<int64_t, int64_t, int64_t, int64_t> minimaxM31CtxBoundaryNormQTopkForm(
    torch::Tensor const& shared, bool writeNorm)
{
    char const* op = "minimax_m31_ctx_boundary_normq_topk_form";
    TORCH_CHECK(shared.is_cuda() && shared.dim() == 2, op, ": a CUDA [rows, hidden]");
    alignas(16) static uint64_t dummy[2];
    auto* any = reinterpret_cast<__nv_bfloat16*>(dummy);
    tk::CtxBoundaryTopkNormQuantParams p;
    p.shared = any;
    p.plane = any;
    p.scores = reinterpret_cast<float const*>(dummy);
    p.topk = 4;
    p.residual = any;
    p.weight = any;
    p.normOut = writeNorm ? any : nullptr;
    p.qOut = dummy;
    p.sfOut = reinterpret_cast<uint8_t*>(dummy);
    p.rows = static_cast<int>(shared.size(0));
    p.hidden = static_cast<int>(shared.size(1));
    int dual = 0;
    int ctas = 0;
    int grid = 0;
    int smem = 0;
    at::cuda::CUDAGuard const guard(shared.device());
    cudaError_t const e = tk::ctxBoundaryTopkNormQuantLaunchForm(p, &dual, &ctas, &grid, &smem);
    TORCH_CHECK(e == cudaSuccess, op, ": ", cudaGetErrorString(e));
    return {dual, ctas, grid, smem};
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_ctx_boundary_normq(Tensor(a!) inter, Tensor residual, Tensor weight, float eps, bool write_norm, "
        "bool write_f32=False, Tensor? fp4_global_scale=None, bool enable_pdl=False) -> (Tensor, Tensor, Tensor, "
        "Tensor, Tensor, Tensor)");
    m.def(
        "minimax_m31_ctx_boundary_normq_out(Tensor(a!) inter, Tensor? residual, Tensor weight, float eps, "
        "Tensor(b!)? norm_out, Tensor(c!)? q_out, Tensor(d!)? sf_out, Tensor(e!)? f32_out=None, "
        "Tensor? fp4_global_scale=None, Tensor(f!)? fp4_out=None, Tensor(g!)? fp4_sf_out=None, "
        "Tensor(h!)? stage_ids=None, Tensor(i!)? stage_weights=None, int stage_topk=0, bool enable_pdl=False) -> ()");
    m.def(
        "minimax_m31_ctx_boundary_normq_shape(Tensor inter, bool add, bool write_norm, bool write_f32, bool fp4) -> "
        "(int, int, int)");
    m.def(
        "minimax_m31_ctx_boundary_normq_topk(Tensor(a!) shared, Tensor plane, Tensor scores, Tensor residual, "
        "Tensor weight, float eps, bool write_norm, bool enable_pdl=False) -> (Tensor, Tensor, Tensor)");
    m.def(
        "minimax_m31_ctx_boundary_normq_form(Tensor inter, bool add, bool write_norm, bool write_f32, bool fp4) -> "
        "(int, int, int, int, int)");
    m.def("minimax_m31_ctx_boundary_normq_topk_form(Tensor shared, bool write_norm) -> (int, int, int, int)");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_ctx_boundary_normq", &TRTLLM_MV2_NS::torch_ext::minimaxM31CtxBoundaryNormQ);
    m.impl("minimax_m31_ctx_boundary_normq_out", &TRTLLM_MV2_NS::torch_ext::minimaxM31CtxBoundaryNormQOut);
    m.impl("minimax_m31_ctx_boundary_normq_shape", &TRTLLM_MV2_NS::torch_ext::minimaxM31CtxBoundaryNormQShape);
    m.impl("minimax_m31_ctx_boundary_normq_topk", &TRTLLM_MV2_NS::torch_ext::minimaxM31CtxBoundaryNormQTopk);
    m.impl("minimax_m31_ctx_boundary_normq_form", &TRTLLM_MV2_NS::torch_ext::minimaxM31CtxBoundaryNormQForm);
    m.impl("minimax_m31_ctx_boundary_normq_topk_form", &TRTLLM_MV2_NS::torch_ext::minimaxM31CtxBoundaryNormQTopkForm);
}
