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

#include "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified/csrc/fp8BlockMoeTpb.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <initializer_list>
#include <optional>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

namespace kd = TRTLLM_MV2_NS::kernels::fp8_block_moe_tpb;

void checkAligned(at::Tensor const& t, char const* name)
{
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

// Returns this rank's [T, 2560] bf16 partial of the routed experts for T <= 8 rows. `xsf` holds the transposed 1x128
// scales of `xq`; `w1`..`s2` are this `rank`'s 640 blocks (layouts in fp8BlockMoeTpb.h). `workspace` is a zeroed
// uint8 buffer of at least `fp8_block_moe_tpb_workspace_bytes()`, one per device. The optional `sh_*` operands
// (3..8 rows) fold in this rank's TP slice of the bf16 shared expert (see TpbParams).
at::Tensor fp8BlockMoeTpb(at::Tensor const& logits, at::Tensor const& xq, at::Tensor const& xsf, at::Tensor const& w1,
    at::Tensor const& s1, at::Tensor const& w2, at::Tensor const& s2, int64_t rank, at::Tensor& workspace,
    bool enablePdl, std::optional<at::Tensor> const& shX, std::optional<at::Tensor> const& shGu,
    std::optional<at::Tensor> const& shGl, std::optional<at::Tensor> const& shDn)
{
    int64_t const rows = logits.size(0);
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kBFloat16 && logits.dim() == 2
            && logits.size(1) == kd::kExperts && logits.is_contiguous(),
        "logits must be contiguous bf16 [T, 512]");
    TORCH_CHECK(rows >= 1 && rows <= kd::kMaxRows, "1..8 rows");
    TORCH_CHECK(
        xq.scalar_type() == at::kFloat8_e4m3fn && xq.is_contiguous() && xq.size(0) == rows && xq.size(1) == kd::kHidden,
        "xq must be contiguous e4m3 [T, 2560]");
    TORCH_CHECK(xsf.scalar_type() == at::kFloat && xsf.dim() == 2 && xsf.size(0) == kd::kHidden / 128
            && xsf.size(1) >= rows && xsf.stride(1) == 1,
        "xsf must be fp32 [20, >= T] with unit column stride");
    int64_t const blocks = w1.size(0);
    TORCH_CHECK(w1.scalar_type() == at::kFloat8_e4m3fn && w1.is_contiguous() && w1.dim() == 3 && w1.size(1) == 256
            && w1.size(2) == kd::kHidden,
        "w1 must be contiguous e4m3 [blocks, 256, 2560]");
    TORCH_CHECK(
        s1.scalar_type() == at::kFloat && s1.is_contiguous() && s1.size(0) == blocks && s1.numel() == blocks * 40,
        "s1 must be contiguous fp32 [blocks, 2, 20]");
    TORCH_CHECK(w2.scalar_type() == at::kFloat8_e4m3fn && w2.is_contiguous() && w2.size(0) == blocks
            && w2.size(1) == kd::kHidden && w2.size(2) == 128,
        "w2 must be contiguous e4m3 [blocks, 2560, 128]");
    TORCH_CHECK(
        s2.scalar_type() == at::kFloat && s2.is_contiguous() && s2.size(0) == blocks && s2.numel() == blocks * 20,
        "s2 must be contiguous fp32 [blocks, 20]");
    TORCH_CHECK(blocks == kd::kTpbBlocks, "a rank holds 640 blocks");
    TORCH_CHECK(rank >= 0 && rank < kd::kTpbRanks, "rank must be in [0, 4)");
    TORCH_CHECK(workspace.is_cuda()
            && workspace.numel() * workspace.element_size() >= static_cast<int64_t>(kd::tpbWorkspaceBytes()),
        "workspace too small");
    for (at::Tensor const* t : std::initializer_list<at::Tensor const*>{&logits, &xq, &w1, &s1, &w2, &s2, &workspace})
    {
        checkAligned(*t, "operand");
    }
    for (at::Tensor const* t : std::initializer_list<at::Tensor const*>{&xq, &xsf, &w1, &s1, &w2, &s2, &workspace})
    {
        TORCH_CHECK(t->device() == logits.device(), "every operand must be on the logits' CUDA device");
    }
    c10::cuda::CUDAGuard const guard(logits.device());
    int const device = logits.get_device();
    auto out = at::empty({rows, kd::kHidden}, logits.options());
    kd::TpbParams p{};
    p.logits = logits.data_ptr();
    p.xq = static_cast<uint8_t const*>(xq.data_ptr());
    p.xsf = xsf.data_ptr<float>();
    p.xsfStride = static_cast<int>(xsf.stride(0));
    p.w1 = static_cast<uint8_t const*>(w1.data_ptr());
    p.s1 = s1.data_ptr<float>();
    p.w2 = static_cast<uint8_t const*>(w2.data_ptr());
    p.s2 = s2.data_ptr<float>();
    p.rank = static_cast<int>(rank);
    p.numRows = static_cast<int>(rows);
    if (shGu.has_value())
    {
        TORCH_CHECK(
            shX.has_value() && shGl.has_value() && shDn.has_value(), "the shared fold takes sh_x, sh_gu, sh_gl, sh_dn");
        TORCH_CHECK(shX->scalar_type() == at::kBFloat16 && shX->is_contiguous() && shX->size(0) == rows
                && shX->size(1) == kd::kHidden,
            "sh_x must be contiguous bf16 [T, 2560]");
        TORCH_CHECK(shGu->scalar_type() == at::kBFloat16 && shGu->is_contiguous() && shGu->dim() == 3
                && shGu->size(0) == kd::kShItems && shGu->size(1) == 64 && shGu->size(2) == kd::kHidden,
            "sh_gu must be contiguous bf16 [5, 64, 2560]");
        TORCH_CHECK(shGl->scalar_type() == at::kBFloat16 && shGl->is_contiguous() && shGl->numel() == kd::kHidden,
            "sh_gl must be contiguous bf16 [2560]");
        TORCH_CHECK(shDn->scalar_type() == at::kBFloat16 && shDn->is_contiguous() && shDn->dim() == 3
                && shDn->size(0) == kd::kShItems && shDn->size(1) == kd::kHidden && shDn->size(2) == kd::kShItemCols,
            "sh_dn must be contiguous bf16 [5, 2560, 32]");
        for (at::Tensor const* t : std::initializer_list<at::Tensor const*>{&*shX, &*shGu, &*shGl, &*shDn})
        {
            checkAligned(*t, "shared operand");
            TORCH_CHECK(t->device() == logits.device(), "every shared operand must be on the logits' CUDA device");
        }
        p.shX = shX->data_ptr();
        p.shGu = shGu->data_ptr();
        p.shGl = shGl->data_ptr();
        p.shDn = shDn->data_ptr();
        p.shBlocks = kd::kShItems;
    }
    p.out = out.data_ptr();
    p.workspace = workspace.data_ptr();
    kd::launchTpb(p, enablePdl, c10::cuda::getCurrentCUDAStream(device).stream());
    return out;
}

int64_t fp8BlockMoeTpbWorkspaceBytes()
{
    return static_cast<int64_t>(kd::tpbWorkspaceBytes());
}

} // namespace

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "fp8_block_moe_tpb(Tensor logits, Tensor xq, Tensor xsf, Tensor w1, Tensor s1, Tensor w2, Tensor s2, int rank, "
        "Tensor(a!) workspace, bool enable_pdl=False, Tensor? sh_x=None, Tensor? sh_gu=None, Tensor? sh_gl=None, "
        "Tensor? sh_dn=None) -> Tensor");
    m.def("fp8_block_moe_tpb_workspace_bytes() -> int");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("fp8_block_moe_tpb", &TRTLLM_MV2_NS::torch_ext::fp8BlockMoeTpb);
}

TRTLLM_MV2_LIBRARY_IMPL(CompositeExplicitAutograd, m)
{
    m.impl("fp8_block_moe_tpb_workspace_bytes", &TRTLLM_MV2_NS::torch_ext::fp8BlockMoeTpbWorkspaceBytes);
}
