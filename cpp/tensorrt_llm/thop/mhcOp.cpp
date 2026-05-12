/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/mhcKernels/mhcKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels::mhc;

namespace
{

void mhcBigFuseOp(torch::Tensor y_acc, torch::Tensor r_acc, torch::Tensor residual, torch::Tensor hc_scale,
    torch::Tensor hc_base, torch::Tensor post_mix, torch::Tensor comb_mix, torch::Tensor layer_input, int64_t M,
    int64_t K, int64_t hidden_size, double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, int64_t num_splits, int64_t block_size)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    tk::mhcBigFuseLaunch(y_acc.data_ptr<float>(), r_acc.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr<at::BFloat16>()), hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(), post_mix.data_ptr<float>(), comb_mix.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr<at::BFloat16>()), static_cast<int>(M),
        static_cast<int>(K), static_cast<int>(hidden_size), static_cast<float>(rms_eps), static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps), static_cast<float>(hc_post_mult_value), static_cast<int>(sinkhorn_repeat),
        static_cast<int>(num_splits), static_cast<int>(block_size), /*norm_weight=*/nullptr, /*norm_eps=*/0.f, stream);
}

void mhcGemmSqrsumFmaOp(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor r, int64_t M, int64_t N,
    int64_t K, int64_t tile_n, int64_t tile_m)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    tk::mhcGemmSqrsumFmaLaunch(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>()), w.data_ptr<float>(),
        y.data_ptr<float>(), r.data_ptr<float>(), static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(tile_n), static_cast<int>(tile_m), stream);
}

void mhcHcHeadApplyOp(torch::Tensor mixes, torch::Tensor sqrsum, torch::Tensor x, torch::Tensor out,
    torch::Tensor scale, torch::Tensor base_t, int64_t M, int64_t mult, int64_t hidden_size, int64_t K, double norm_eps,
    double eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    tk::mhcHcHeadApplyLaunch(mixes.data_ptr<float>(), sqrsum.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), scale.data_ptr<float>(),
        base_t.data_ptr<float>(), static_cast<int>(M), static_cast<int>(mult), static_cast<int>(hidden_size),
        static_cast<int>(K), static_cast<float>(norm_eps), static_cast<float>(eps), stream);
}

void mhcPostMappingOp(torch::Tensor residual, torch::Tensor x, torch::Tensor post_mix, torch::Tensor comb_mix,
    torch::Tensor out, int64_t B, int64_t hidden_size)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    tk::mhcPostMappingLaunch(reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr<at::BFloat16>()), post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), static_cast<int>(B),
        static_cast<int>(hidden_size), stream);
}

void mhcFusedHcOp(torch::Tensor x_prev, torch::Tensor residual_prev, torch::Tensor post_mix_prev,
    torch::Tensor comb_mix_prev, torch::Tensor w, torch::Tensor hc_scale, torch::Tensor hc_base,
    torch::Tensor residual_cur, torch::Tensor post_mix_cur, torch::Tensor comb_mix_cur, torch::Tensor layer_input_cur,
    torch::Tensor y_acc_workspace, torch::Tensor r_acc_workspace, torch::Tensor done_counter_workspace, int64_t M,
    int64_t hidden_size, int64_t hc_mult, double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, int64_t backend, int64_t tile_n, int64_t num_k_splits,
    int64_t bigfuse_block_size, int64_t tile_m, c10::optional<torch::Tensor> norm_weight, double norm_eps)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    // Fused next-layer RMSNorm on layer_input_cur. All four backends (Path B/D
    // for MMA and Path E/F for FMA) support it now: Path B/E fuse the norm into
    // the bigfuse kernel's Phase 2 layer_input write; Path D/F fuse it into the
    // single-kernel Phase 4 epilogue. When norm_weight is null, layer_input is
    // left un-normalized (caller must run RMSNorm separately).
    __nv_bfloat16 const* norm_weight_ptr = nullptr;
    if (norm_weight.has_value() && norm_weight->defined())
    {
        TORCH_CHECK(norm_weight->dtype() == torch::kBFloat16, "mhc_fused_hc: norm_weight must be bfloat16");
        TORCH_CHECK(norm_weight->is_contiguous(), "mhc_fused_hc: norm_weight must be contiguous");
        TORCH_CHECK(norm_weight->numel() == hidden_size,
            "mhc_fused_hc: norm_weight numel=%ld must equal hidden_size=%ld", norm_weight->numel(), hidden_size);
        norm_weight_ptr = reinterpret_cast<__nv_bfloat16 const*>(norm_weight->data_ptr<at::BFloat16>());
    }

    // backend codes:
    //   0 = fused_half_mma (2-kernel: fused_tf32_pmap_gemm_rout + mhcBigFuseKernel)
    //   1 = fused_half_fma (2-kernel: fused_pmap_gemm_fma_ksplit + mhcBigFuseKernel)
    //   2 = fused_all_mma  (1-kernel: fused_allinone_tf32_pmap_gemm_atomic_impl, Path D)
    //   3 = fused_all_fma  (1-kernel: fused_pmap_gemm_fma_allinone,            Path F)
    if (backend == 3)
    {
        tk::mhcFusedHcFmaAllInOneLaunch(reinterpret_cast<__nv_bfloat16 const*>(x_prev.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16 const*>(residual_prev.data_ptr<at::BFloat16>()),
            post_mix_prev.data_ptr<float>(), comb_mix_prev.data_ptr<float>(), w.data_ptr<float>(),
            hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(residual_cur.data_ptr<at::BFloat16>()), post_mix_cur.data_ptr<float>(),
            comb_mix_cur.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(layer_input_cur.data_ptr<at::BFloat16>()),
            y_acc_workspace.data_ptr<float>(), r_acc_workspace.data_ptr<float>(),
            done_counter_workspace.data_ptr<int>(), static_cast<int>(M), static_cast<int>(hidden_size),
            static_cast<int>(hc_mult), static_cast<int>(tile_n), static_cast<int>(num_k_splits),
            static_cast<int>(tile_m), static_cast<float>(rms_eps), static_cast<float>(hc_pre_eps),
            static_cast<float>(hc_sinkhorn_eps), static_cast<float>(hc_post_mult_value),
            static_cast<int>(sinkhorn_repeat), norm_weight_ptr, static_cast<float>(norm_eps), stream);
        return;
    }
    if (backend == 2)
    {
        tk::mhcFusedHcAllInOneLaunch(reinterpret_cast<__nv_bfloat16 const*>(x_prev.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16 const*>(residual_prev.data_ptr<at::BFloat16>()),
            post_mix_prev.data_ptr<float>(), comb_mix_prev.data_ptr<float>(), w.data_ptr<float>(),
            hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(residual_cur.data_ptr<at::BFloat16>()), post_mix_cur.data_ptr<float>(),
            comb_mix_cur.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(layer_input_cur.data_ptr<at::BFloat16>()),
            y_acc_workspace.data_ptr<float>(), r_acc_workspace.data_ptr<float>(),
            done_counter_workspace.data_ptr<int>(), static_cast<int>(M), static_cast<int>(hidden_size),
            static_cast<int>(hc_mult), static_cast<int>(num_k_splits), static_cast<float>(rms_eps),
            static_cast<float>(hc_pre_eps), static_cast<float>(hc_sinkhorn_eps), static_cast<float>(hc_post_mult_value),
            static_cast<int>(sinkhorn_repeat), norm_weight_ptr, static_cast<float>(norm_eps), stream);
        return;
    }
    if (backend == 1)
    {
        tk::mhcFusedHcFmaLaunch(reinterpret_cast<__nv_bfloat16 const*>(x_prev.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16 const*>(residual_prev.data_ptr<at::BFloat16>()),
            post_mix_prev.data_ptr<float>(), comb_mix_prev.data_ptr<float>(), w.data_ptr<float>(),
            hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(residual_cur.data_ptr<at::BFloat16>()), post_mix_cur.data_ptr<float>(),
            comb_mix_cur.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(layer_input_cur.data_ptr<at::BFloat16>()),
            y_acc_workspace.data_ptr<float>(), r_acc_workspace.data_ptr<float>(), static_cast<int>(M),
            static_cast<int>(hidden_size), static_cast<int>(hc_mult), static_cast<int>(tile_n),
            static_cast<int>(num_k_splits), static_cast<int>(bigfuse_block_size), static_cast<float>(rms_eps),
            static_cast<float>(hc_pre_eps), static_cast<float>(hc_sinkhorn_eps), static_cast<float>(hc_post_mult_value),
            static_cast<int>(sinkhorn_repeat), norm_weight_ptr, static_cast<float>(norm_eps), stream);
        return;
    }

    tk::mhcFusedHcLaunch(reinterpret_cast<__nv_bfloat16 const*>(x_prev.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 const*>(residual_prev.data_ptr<at::BFloat16>()), post_mix_prev.data_ptr<float>(),
        comb_mix_prev.data_ptr<float>(), w.data_ptr<float>(), hc_scale.data_ptr<float>(), hc_base.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(residual_cur.data_ptr<at::BFloat16>()), post_mix_cur.data_ptr<float>(),
        comb_mix_cur.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(layer_input_cur.data_ptr<at::BFloat16>()),
        y_acc_workspace.data_ptr<float>(), r_acc_workspace.data_ptr<float>(), static_cast<int>(M),
        static_cast<int>(hidden_size), static_cast<int>(hc_mult), static_cast<int>(num_k_splits),
        static_cast<int>(bigfuse_block_size), static_cast<float>(rms_eps), static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps), static_cast<float>(hc_post_mult_value), static_cast<int>(sinkhorn_repeat),
        norm_weight_ptr, static_cast<float>(norm_eps), stream);
}

} // anonymous namespace

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mhc_big_fuse("
        "Tensor y_acc, Tensor r_acc, Tensor residual, "
        "Tensor hc_scale, Tensor hc_base, "
        "Tensor(a!) post_mix, Tensor(b!) comb_mix, Tensor(c!) layer_input, "
        "int M, int K, int hidden_size, "
        "float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, "
        "float hc_post_mult_value, int sinkhorn_repeat, int num_splits, "
        "int block_size=0) -> ()");

    m.def(
        "mhc_gemm_sqrsum_fma("
        "Tensor x, Tensor w, Tensor(a!) y, Tensor(b!) r, "
        "int M, int N, int K, "
        "int tile_n=0, int tile_m=0) -> ()");

    m.def(
        "mhc_hc_head_apply("
        "Tensor mixes, Tensor sqrsum, Tensor x, Tensor(a!) out, "
        "Tensor scale, Tensor base_t, "
        "int M, int mult, int hidden_size, int K, "
        "float norm_eps, float eps) -> ()");

    m.def(
        "mhc_post_mapping("
        "Tensor residual, Tensor x, "
        "Tensor post_mix, Tensor comb_mix, Tensor(a!) out, "
        "int B, int hidden_size) -> ()");

    m.def(
        "mhc_fused_hc("
        "Tensor x_prev, Tensor residual_prev, "
        "Tensor post_mix_prev, Tensor comb_mix_prev, "
        "Tensor w, Tensor hc_scale, Tensor hc_base, "
        "Tensor(a!) residual_cur, Tensor(b!) post_mix_cur, "
        "Tensor(c!) comb_mix_cur, Tensor(d!) layer_input_cur, "
        "Tensor(e!) y_acc_workspace, Tensor(f!) r_acc_workspace, "
        "Tensor(g!) done_counter_workspace, "
        "int M, int hidden_size, int hc_mult, "
        "float rms_eps, float hc_pre_eps, float hc_sinkhorn_eps, "
        "float hc_post_mult_value, int sinkhorn_repeat, "
        "int backend=0, int tile_n=0, int num_k_splits=0, int bigfuse_block_size=0, "
        "int tile_m=1, Tensor? norm_weight=None, float norm_eps=0.0) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mhc_big_fuse", &mhcBigFuseOp);
    m.impl("mhc_gemm_sqrsum_fma", &mhcGemmSqrsumFmaOp);
    m.impl("mhc_hc_head_apply", &mhcHcHeadApplyOp);
    m.impl("mhc_post_mapping", &mhcPostMappingOp);
    m.impl("mhc_fused_hc", &mhcFusedHcOp);
}
