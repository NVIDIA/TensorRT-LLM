/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cuda_profiler_api.h>
#include <fused_multihead_attention_utils.h>

#include "hopper/fmha_dgrad.h"
#include "hopper/fmha_fprop.h"

#include "test.h"

using std::cout;
using std::endl;

// using fmha::MAX_E4M3;
// using fmha::MAX_E5M2;

using namespace fmha::hopper;
using namespace torch::indexing;

std::array<at::Tensor, 3> fprop_seq(at::Tensor const& Q8, at::Tensor const& K8, at::Tensor const& V8t,
    at::Tensor const& amask, float const scale_q_k, float const d_scale_qkv, at::Tensor const& amax_exp_p,
    MetaMap const& recipe)
{
    float d_scale_q_k = d_scale_qkv * d_scale_qkv;
    auto sizes = Q8.sizes();
    int b = sizes[0];
    int h = sizes[1];
    int s = sizes[2];
    int d = sizes[3];

    auto K8step = K8.view({b, h, -1, 64, d});
    auto V8tstep = V8t.view({b, h, d, -1, 64});
    auto amaskstep = amask.view({b, 1, s, -1, 64});

    int steps = K8step.sizes()[2];
    TORCH_CHECK(steps * 64 == s);

    auto K8c = K8step.index({Slice(), Slice(), 0, Slice(), Slice()}).contiguous();
    auto V8c = V8tstep.index({Slice(), Slice(), Slice(), 0, Slice()}).contiguous();
    auto amaskc = amaskstep.index({Slice(), Slice(), Slice(), 0, Slice()}).contiguous();

    auto Pc = matmul_nt(Q8, K8c, recipe.at("QKV").dtype, recipe.at("QKV").dtype, d_scale_q_k * scale_q_k);
    auto M = std::get<0>(Pc.max(-1, true)); // [b,h,s,1]
    Pc = Pc.where(amaskc == 1.f, torch::ones_like(Pc) * -std::numeric_limits<float>::infinity());
    auto Tmp0 = Pc - M;
    auto ExpP = torch::exp(Tmp0);
    auto Z = ExpP.sum(-1, true);

    float q_scale_exp_p = recipe.at("expP").q_scale(
        amax_exp_p.item<float>()); // pow(2, (int) log2f(fmha::MAX_E4M3 / amax_exp_p.item<float>()));

    auto ExpP8 = convert_fp32_to_fp8(ExpP, q_scale_exp_p, recipe.at("expP").dtype);

    // BMM2.
    float d_scale_exp_p_v = d_scale_qkv / q_scale_exp_p;
    auto O = matmul_nt(ExpP8, V8c, recipe.at("expP").dtype, recipe.at("QKV").dtype, d_scale_exp_p_v);

    for (int step = 1; step < steps; step++)
    {
        auto K8c = K8step.index({Slice(), Slice(), step, Slice(), Slice()}).contiguous();
        auto V8c = V8tstep.index({Slice(), Slice(), Slice(), step, Slice()}).contiguous();

        auto Pc = matmul_nt(Q8, K8c, recipe.at("QKV").dtype, recipe.at("QKV").dtype, d_scale_q_k * scale_q_k);
        auto Mc = torch::maximum(M, std::get<0>(Pc.max(-1, true))); // [b,h,s,1]
        Pc = Pc.where(amaskc == 1.f, torch::ones_like(Pc) * -std::numeric_limits<float>::infinity());
        auto Tmp0 = Pc - Mc;
        auto ExpP = torch::exp(Tmp0);
        auto Zc = ExpP.sum(-1, true);

        auto correction = torch::exp(M - Mc);
        M = Mc;
        Z = correction * Z + Zc;
        auto ExpP8 = convert_fp32_to_fp8(ExpP, q_scale_exp_p, recipe.at("expP").dtype);

        // BMM2.
        auto Oc = matmul_nt(ExpP8, V8c, recipe.at("expP").dtype, recipe.at("QKV").dtype, d_scale_exp_p_v);
        O = O * correction + Oc;
    }

    auto Zinv = 1.f / Z; // [b,h,s,1]
    // Set Zinv to 0 where Z was 0 to remove inf.
    Zinv = Zinv.where(Z != 0.f, torch::zeros_like(Z));
    return {O * Zinv, Zinv, M};
}

std::array<at::Tensor, 18> reference(at::Tensor QKV8, at::Tensor dO8_, at::Tensor& amask, float const scale_q_k,
    float const d_scale_qkv, float const d_scale_do, MetaMap const& recipe)
{

    // [b,s,h,d] => [b,h,s,d]
    auto Q8 = QKV8.index({Slice(), Slice(), 0, Slice(), Slice()}).permute({0, 2, 1, 3}).contiguous();
    auto Q8t = QKV8.index({Slice(), Slice(), 0, Slice(), Slice()}).permute({0, 2, 3, 1}).contiguous();
    auto K8 = QKV8.index({Slice(), Slice(), 1, Slice(), Slice()}).permute({0, 2, 1, 3}).contiguous();
    auto K8t = QKV8.index({Slice(), Slice(), 1, Slice(), Slice()}).permute({0, 2, 3, 1}).contiguous();
    auto V8 = QKV8.index({Slice(), Slice(), 2, Slice(), Slice()}).permute({0, 2, 1, 3}).contiguous();
    auto V8t = QKV8.index({Slice(), Slice(), 2, Slice(), Slice()}).permute({0, 2, 3, 1}).contiguous();

    auto dO8 = dO8_.permute({0, 2, 1, 3}).contiguous();
    auto dO8t = dO8_.permute({0, 2, 3, 1}).contiguous();

    // BMM1 and scale.
    float d_scale_q_k = d_scale_qkv * d_scale_qkv;
    auto P = matmul_nt(Q8, K8, recipe.at("QKV").dtype, recipe.at("QKV").dtype, d_scale_q_k * scale_q_k);

    // Masked Softmax.
    auto M = std::get<0>(P.max(-1, true)); // [b,h,s,1]
    P = P.where(amask == 1.f, torch::ones_like(P) * -std::numeric_limits<float>::infinity());

    auto Tmp0 = P - M;
    auto ExpP = torch::exp(Tmp0);
    auto Z = ExpP.sum(-1, true);
    auto Zinv = 1.f / Z; // [b,h,s,1]
    // Set Zinv to 0 where Z was 0 to remove inf.
    Zinv = Zinv.where(Z != 0.f, torch::zeros_like(Z));
    auto S = ExpP * Zinv;

    // Convert ExpP and S.
    auto amax_exp_p = ExpP.max();
    TORCH_CHECK((amax_exp_p.item<float>() == 1.f));
    auto amax_s = S.max();
    float q_scale_exp_p = recipe.at("expP").q_scale(amax_exp_p.item<float>()); // MAX_E4M3 / amax_exp_p.item<float>();
    float q_scale_s = recipe.at("S").q_scale(amax_s.item<float>());            // MAX_E4M3 / amax_s.item<float>();
    auto ExpP8 = convert_fp32_to_fp8(ExpP, q_scale_exp_p, recipe.at("expP").dtype);
    auto S8 = convert_fp32_to_fp8(S, q_scale_s, recipe.at("S").dtype);

    // BMM2.
    float d_scale_exp_p_v = d_scale_qkv / q_scale_exp_p;
    auto O = matmul_nt(ExpP8, V8t, recipe.at("expP").dtype, recipe.at("QKV").dtype, d_scale_exp_p_v);
    O *= Zinv;

    auto [Oseq, Zinvseq, Mseq] = fprop_seq(Q8, K8, V8t, amask, scale_q_k, d_scale_qkv, amax_exp_p, recipe);
    cout << "O rel: " << ((O - Oseq).abs().sum() / O.abs().sum()).item<float>() << endl;
    cout << "Z rel: " << ((Zinv - Zinvseq).abs().sum() / Zinv.abs().sum()).item<float>() << endl;
    cout << "M rel: " << ((M - Mseq).abs().sum() / M.abs().sum()).item<float>() << endl;

    O = Oseq;
    Zinv = Zinvseq;
    // Convert O.
    //
    auto amax_o = O.abs().max();
    float q_scale_o = recipe.at("O").q_scale(amax_o.item<float>()); // MAX_E4M3 / amax_o.item<float>();
    auto O8 = convert_fp32_to_fp8(O, q_scale_o, recipe.at("O").dtype);

    float d_scale_do_o = d_scale_do / q_scale_o;
    auto dOO = matmul_nt(dO8, O8, recipe.at("dO").dtype, recipe.at("O").dtype, d_scale_do_o);

    auto tmp = torch::diagonal(dOO, 0, 2, 3).unsqueeze(-1);

    float d_scale_do_v = d_scale_do * d_scale_qkv;
    auto dS = matmul_nt(dO8, V8, recipe.at("dO").dtype, recipe.at("QKV").dtype, d_scale_do_v) * amask;

    auto dP = (dS - tmp) * S * scale_q_k;

    auto amax_dp = dP.abs().max();
    float q_scale_dp = recipe.at("dP").q_scale(amax_dp.item<float>()); // MAX_E5M2 / amax_dp.item<float>();

    auto dP8 = convert_fp32_to_fp8(dP, q_scale_dp, recipe.at("dP").dtype);

    float d_scale_dp_qkv = d_scale_qkv / q_scale_dp;
    float d_scale_s_do = d_scale_do / q_scale_s;
    // dQ = dP x K
    auto dQ = matmul_nt(dP8, K8t, recipe.at("dP").dtype, recipe.at("QKV").dtype, d_scale_dp_qkv);
    // dK = dP' x Q
    auto dK = matmul_nt(
        dP8.permute({0, 1, 3, 2}).contiguous(), Q8t, recipe.at("dP").dtype, recipe.at("QKV").dtype, d_scale_dp_qkv);
    // dV = S' x dO
    auto dV = matmul_nt(
        S8.permute({0, 1, 3, 2}).contiguous(), dO8t, recipe.at("S").dtype, recipe.at("dO").dtype, d_scale_s_do);

    auto amax_dqkv = torch::concat({dQ.abs().max().view(1), dK.abs().max().view(1), dV.abs().max().view(1)}).max();
    float q_scale_dqkv = recipe.at("dQKV").q_scale(amax_dqkv.item<float>()); // MAX_E4M3 / amax_dqkv.item<float>();

    auto dQ8 = convert_fp32_to_fp8(dQ, q_scale_dqkv, recipe.at("dQKV").dtype);
    auto dK8 = convert_fp32_to_fp8(dK, q_scale_dqkv, recipe.at("dQKV").dtype);
    auto dV8 = convert_fp32_to_fp8(dV, q_scale_dqkv, recipe.at("dQKV").dtype);
    return {P, S, M, Zinv, O8, dS, dP, dQ, dK, dV, dQ8, dK8, dV8, amax_s, amax_exp_p, amax_o, amax_dp, amax_dqkv};
}

// TODO Features
// [done] Mixed precision for dQ, dK.
// Optimized traversal.

// TODO Testing
// - [done] In test mode we should dump and check intermediate tensors: S, dS, dP.
// - [done] Test all outputs: dQKV, amaxs
// - [done] Test on one tile with different masks.
// - [done] Test on two tile with different masks.
// - [done] Test on full size with different masks.
// - [done] Test for all scale factors.
// - [done] Test dropout mask.

// TODO Benchmarking.

C10_DEFINE_int32(b, 32, "b");
C10_DEFINE_int32(s, 512, "s");
C10_DEFINE_int32(h, 16, "h");
C10_DEFINE_int32(seed, 1234, "seed");

C10_DEFINE_int32(maskgen, 0, "maskgen");
C10_DEFINE_int32(verbose, 1, "verbosity");
C10_DEFINE_int32(datagen, 0, "datagen");

C10_DEFINE_int32(show_batch, 0, "which output batch to print");
C10_DEFINE_int32(num_iters, 1, "which output batch to print");

C10_DEFINE_double(p_dropout, 0.0, "Dropout probability");

C10_DEFINE_double(scale_q_k, 0.125, "Attention scale (usually 1/sqrt(d))");
C10_DEFINE_double(epsilon, 0.075, "Error tol.");

C10_DEFINE_int32(all_e5m2, 0, "Whether all activation dgrads are in E5M2.");

int main(int argc, char** argv)
{

    auto launch = &dgrad::run_fmha_dgrad_fp8_512_64_sm90;

    TORCH_CHECK(c10::ParseCommandLineFlags(&argc, &argv));
    int const verbose = FLAGS_verbose;

    bool const all_e5m2 = FLAGS_all_e5m2 == 1;

    torch::manual_seed(FLAGS_seed);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    c10::optional<at::Generator> gen_;
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, at::cuda::detail::getDefaultCUDAGenerator());

    int const b = FLAGS_b;
    int const s = FLAGS_s;
    int const h = FLAGS_h;
    int const d = 64;

    if (verbose > 0)
    {

        int sm = props->major * 10 + props->minor;
        // Device info.
        printf("Device........: %s\n", props->name);
        printf("Arch.(sm).....: %d\n", sm);
        printf("#.of.SMs......: %d\n", props->multiProcessorCount);
        printf("SMEM/SM.......: %lu\n", props->sharedMemPerMultiprocessor);
        // Problem info.
        printf("Batch ........: %d\n", b);
        printf("Heads ........: %d\n", h);
        printf("Dimension ....: %d\n", d);
        printf("Seq length ...: %d\n", s);
    }
    int const show_batch = std::min(FLAGS_show_batch, b - 1);

    TORCH_CHECK(FLAGS_maskgen >= 0 && FLAGS_maskgen < 3);
    Maskgen maskgen = static_cast<Maskgen>(FLAGS_maskgen);

    TORCH_CHECK(FLAGS_datagen >= 0 && FLAGS_datagen < 3);
    Datagen datagen = static_cast<Datagen>(FLAGS_datagen);

    auto recipe = get_recipe(FLAGS_all_e5m2);

    auto QKV = draw_tensor({b, s, 3, h, d}, options, datagen);
    auto dO = draw_tensor({b, s, h, d}, options, datagen);

    float amax_qkv = QKV.abs().max().item<float>();
    float amax_do = dO.abs().max().item<float>();

    float q_scale_qkv = recipe.at("QKV").q_scale(amax_qkv);
    float q_scale_do = recipe.at("dO").q_scale(amax_do);
    float d_scale_qkv = 1.f / q_scale_qkv;
    float d_scale_do = 1.f / q_scale_do;

    // [b,s,h,3,d] => [b,s,h,d]
    auto Q = QKV.index({Slice(), Slice(), 0, Slice(), Slice()});
    auto K = QKV.index({Slice(), Slice(), 1, Slice(), Slice()});
    auto V = QKV.index({Slice(), Slice(), 2, Slice(), Slice()});

    // Convert inputs.
    auto QKV8 = convert_fp32_to_fp8(QKV, q_scale_qkv, recipe.at("QKV").dtype);
    auto dO8 = convert_fp32_to_fp8(dO, q_scale_do, recipe.at("dO").dtype);

    auto [cu_seqlens, amask] = make_mask(b, s, options, maskgen);

    if (verbose > 1)
    {
        cout << "cu_seqlens\n" << cu_seqlens << endl;
    }

    // Run the reference implementation.
    auto [Pref, Sref, M, Zinv, O8ref, dSref, dPref, dQref, dKref, dVref, dQ8ref, dK8ref, dV8ref, amax_s_ref,
        amax_exp_p_ref, amax_o_ref, amax_dp_ref, amax_dqkv_ref]
        = reference(QKV8, dO8, amask, FLAGS_scale_q_k, d_scale_qkv, d_scale_do, recipe);
    FMHA_CHECK_CUDA(cudaDeviceSynchronize());
    float q_scale_s = recipe.at("S").q_scale(amax_s_ref.item<float>());    // MAX_E4M3 / amax_s_ref.item<float>();
    float q_scale_dp = recipe.at("dP").q_scale(amax_dp_ref.item<float>()); // MAX_E5M2 / amax_dp_ref.item<float>();
    float q_scale_o = recipe.at("O").q_scale(amax_o_ref.item<float>());    // MAX_E4M3 / amax_o_ref.item<float>();
    float q_scale_dqkv
        = recipe.at("dQKV").q_scale(amax_dqkv_ref.item<float>());          // MAX_E4M3 / amax_dqkv_ref.item<float>();
    float q_scale_exp_p
        = recipe.at("expP").q_scale(amax_exp_p_ref.item<float>());         // MAX_E4M3 / amax_exp_p_ref.item<float>();

    float d_scale_s_ref = 1.f / q_scale_s;
    float d_scale_o_ref = 1.f / q_scale_o;
    float d_scale_exp_p = 1.f / q_scale_exp_p;

    float d_scale_dp_ref = 1.f / q_scale_dp;
    float d_scale_dqkv_ref = 1.f / q_scale_dqkv;

    float d_scale_s_do = d_scale_s_ref * d_scale_do;
    float d_scale_do_o = d_scale_do * d_scale_o_ref;

    float d_scale_q_k = d_scale_qkv * d_scale_qkv;
    float d_scale_do_v = d_scale_do * d_scale_qkv;
    float d_scale_dp_qkv = d_scale_qkv / q_scale_dp;
    float d_scale_exp_p_v = d_scale_qkv / q_scale_exp_p;

    // Allocate debug tensors.
    auto S = torch::zeros({s, b, h, s}, options);
    auto dS = torch::zeros({s, b, h, s}, options);
    auto dP = torch::zeros({s, b, h, s}, options);
    auto print_buf = torch::zeros({32 * 1024}, options);

    // Unpad the inputs.
    auto QKV8unpad = unpad(QKV8, cu_seqlens);
    auto dO8unpad = unpad(dO8, cu_seqlens);
    O8ref = O8ref.permute({0, 2, 1, 3}).contiguous();
    cout << O8ref.sizes() << endl; // b,s,h,d
    auto O8unpad_ref = unpad(O8ref, cu_seqlens);
    // Allocate outputs.
    auto dQKV8unpad = torch::empty_like(QKV8unpad);
    // Allocate workspace for dQ.
    auto dQtmp = torch::zeros({b, h, s, d}, options);
    // Allocate space for output amax.
    auto amax_dp = torch::zeros({1}, options);
    auto amax_dqkv = torch::zeros({1}, options);

    auto d_scale_dp = torch::zeros({1}, options);
    auto d_scale_dqkv = torch::zeros({1}, options);

    auto philox_unpacked = torch::tensor({FLAGS_seed, 0}, options.dtype(torch::kInt64));

    FP8DgradMeta dgrad_meta(d_scale_qkv, d_scale_s_ref, d_scale_o_ref, d_scale_do, d_scale_dp_ref, q_scale_s,
        q_scale_dp, q_scale_dqkv, options);

    typename dgrad::Launch_params::Params params_(dQKV8unpad.data_ptr(), QKV8unpad.data_ptr(), dO8unpad.data_ptr(),
        O8unpad_ref.data_ptr(), M.data_ptr<float>(), Zinv.data_ptr<float>(), cu_seqlens.data_ptr<int>(),
        amax_dp.data_ptr<float>(), amax_dqkv.data_ptr<float>(), dQtmp.data_ptr(), b, s, h, d, FLAGS_p_dropout,
        FLAGS_scale_q_k, dgrad_meta.d_scale_qkv.data_ptr<float>(), dgrad_meta.d_scale_s.data_ptr<float>(),
        dgrad_meta.d_scale_o.data_ptr<float>(), dgrad_meta.d_scale_do.data_ptr<float>(), d_scale_dp.data_ptr<float>(),
        d_scale_dqkv.data_ptr<float>(), dgrad_meta.q_scale_s.data_ptr<float>(), dgrad_meta.q_scale_dp.data_ptr<float>(),
        dgrad_meta.q_scale_dqkv.data_ptr<float>(), reinterpret_cast<uint64_t*>(philox_unpacked.data_ptr()));

    dgrad::Launch_params launch_params(props, stream, params_, all_e5m2);

    launch(launch_params, /*configure=*/true);

    auto& params = launch_params.params;

    TORCH_CHECK(params.qkv_stride_in_bytes == h * 3 * d);
    TORCH_CHECK(params.o_stride_in_bytes == h * d);
    TORCH_CHECK(params.ds_stride_in_bytes == b * h * s * 4);

    params.print_buf = print_buf.data_ptr();

    params.s_ptr = S.data_ptr();
    params.ds_ptr = dS.data_ptr();
    params.dp_ptr = dP.data_ptr();

    // cudaProfilerStart();
    for (int it = 0; it < FLAGS_num_iters; it++)
        launch(launch_params, /*configure=*/false);

    // cudaProfilerStop();

    FMHA_CHECK_CUDA(cudaDeviceSynchronize());
    FMHA_CHECK_CUDA(cudaPeekAtLastError());

    auto dQKV8 = pad(dQKV8unpad, cu_seqlens, s);

    // [s,b,h,s] => [b,h,s,s]
    dS = dS.permute({1, 2, 0, 3});
    dP = dP.permute({1, 2, 0, 3});
    S = S.permute({1, 2, 0, 3});

    auto O32ref = convert_fp8_to_fp32(O8ref, d_scale_o_ref, recipe.at("O").dtype);
    auto O32unpad_ref = convert_fp8_to_fp32(O8unpad_ref, d_scale_o_ref, recipe.at("O").dtype);

    auto dQ32ref = convert_fp8_to_fp32(dQ8ref.contiguous(), d_scale_dqkv_ref, recipe.at("dQKV").dtype);
    auto dK32ref = convert_fp8_to_fp32(dK8ref.contiguous(), d_scale_dqkv_ref, recipe.at("dQKV").dtype);
    auto dV32ref = convert_fp8_to_fp32(dV8ref.contiguous(), d_scale_dqkv_ref, recipe.at("dQKV").dtype);

    // [b,s,h,d] => [b,h,s,d]
    auto dQ8 = dQKV8.index({Slice(), Slice(), 0, Slice(), Slice()}).permute({0, 2, 1, 3}).contiguous();
    auto dK8 = dQKV8.index({Slice(), Slice(), 1, Slice(), Slice()}).permute({0, 2, 1, 3}).contiguous();
    auto dV8 = dQKV8.index({Slice(), Slice(), 2, Slice(), Slice()}).permute({0, 2, 1, 3}).contiguous();
    auto dQKV32 = convert_fp8_to_fp32(dQKV8, d_scale_dqkv_ref, recipe.at("dQKV").dtype);
    auto dQ32 = convert_fp8_to_fp32(dQ8, d_scale_dqkv_ref, recipe.at("dQKV").dtype);
    auto dK32 = convert_fp8_to_fp32(dK8, d_scale_dqkv_ref, recipe.at("dQKV").dtype);
    auto dV32 = convert_fp8_to_fp32(dV8, d_scale_dqkv_ref, recipe.at("dQKV").dtype);

    if (verbose > 0)
    {

        cout << "Amask\n" << amask.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << "S\n" << S.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << "Sref\n" << Sref.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << "dS\n" << dS.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << "dSref\n" << dSref.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << "dP\n" << dP.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << "dPref\n" << dPref.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << "M \n" << M.index({show_batch, 0, Slice(None, 4), 0}) << endl;
        cout << "Zinv \n" << Zinv.index({show_batch, 0, Slice(None, 4), 0}) << endl;

        cout << dQ32.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << dQ32ref.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;

        cout << dK32.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << dK32ref.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;

        cout << dV32.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << dV32ref.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;

        cout << std::boolalpha << torch::allclose(S, Sref) << endl;
        cout << std::boolalpha << torch::allclose(dS, dSref) << endl;
        cout << std::boolalpha << torch::allclose(dP, dPref) << endl;

        cout << " S abs: " << (S - Sref).abs().max().item() << endl;
        cout << "dS abs: " << (dS - dSref).abs().max().item() << endl;
        cout << "dP abs: " << (dP - dPref).abs().max().item() << endl;

        cout << " S rel: " << ((S - Sref).abs().sum() / Sref.abs().sum()).item() << endl;
        cout << "dS rel: " << ((dS - dSref).abs().sum() / dSref.abs().sum()).item() << endl;
        cout << "dP rel: " << ((dP - dPref).abs().sum() / dPref.abs().sum()).item() << endl;
    }

    cout << "dQ rel: " << ((dQ32 - dQ32ref).abs().sum() / (dQ32ref).abs().sum()).item() << endl;
    cout << "dK rel: " << ((dK32 - dK32ref).abs().sum() / (dK32ref).abs().sum()).item() << endl;
    cout << "dV rel: " << ((dV32 - dV32ref).abs().sum() / (dV32ref).abs().sum()).item() << endl;

    auto tmp = torch::zeros({4}, options); // convert needs multiple of 4 for vectorized loads/stores.

    printf("[amax_dp  ] CMP: %f REF(self): %f REF(ref): %f\n", amax_dp.item<float>(), dP.abs().max().item<float>(),
        dPref.abs().max().item<float>());

    tmp[0] = amax_dqkv[0];
    printf("[amax_dqkv] CMP: %f CMP(FP8): %f REF(FP8): %f\n", amax_dqkv.item<float>(),
        convert_fp8_to_fp32(convert_fp32_to_fp8(tmp, q_scale_dqkv, recipe.at("dQKV").dtype), d_scale_dqkv_ref,
            recipe.at("dQKV").dtype)[0]
            .item<float>(),
        dQKV32.abs().max().item<float>());

    auto amax_s = torch::zeros({1}, options);
    auto amax_o = torch::zeros({1}, options);
    // FP8FpropMeta fprop_meta(d_scale_qkv, d_scale_exp_p, q_scale_exp_p, q_scale_o, options);
    FP8FpropMeta fprop_meta(d_scale_qkv, d_scale_s_ref, q_scale_s, q_scale_o, options);
    {
        auto Mfprop = torch::empty({b, h, s, 1}, options);
        auto Zfprop = torch::empty({b, h, s, 1}, options);
        auto O8unpad = torch::empty_like(O8unpad_ref);

        auto d_scale_s = torch::empty({1}, options);
        auto d_scale_o = torch::empty({1}, options);

        // This will contain two uint64_t as returned by at::cuda::philox::unpack.
        // Using signed + casting, since there is no unsigned type in torch.
        auto philox_unpacked = torch::empty({2}, options.dtype(torch::kInt64));

        typename fprop::Launch_params::Params params_(QKV8unpad.data_ptr(), O8unpad.data_ptr(),
            Mfprop.data_ptr<float>(), Zfprop.data_ptr<float>(), cu_seqlens.data_ptr<int>(), amax_s.data_ptr<float>(),
            amax_o.data_ptr<float>(), b, s, h, d, FLAGS_p_dropout, FLAGS_scale_q_k,
            fprop_meta.d_scale_qkv.data_ptr<float>(), d_scale_s.data_ptr<float>(), d_scale_o.data_ptr<float>(),
            fprop_meta.q_scale_s.data_ptr<float>(), fprop_meta.q_scale_o.data_ptr<float>(),
            reinterpret_cast<uint64_t*>(philox_unpacked.data_ptr()));

        auto launch = &fprop::run_fmha_fprop_fp8_512_64_sm90;
        fprop::Launch_params launch_params(props, stream, params_, true);

        torch::manual_seed(FLAGS_seed);
        launch(launch_params, /*configure=*/true);

        launch_params.init_philox_state(gen);
        auto& params = launch_params.params;

        TORCH_CHECK(params.qkv_stride_in_bytes == h * 3 * d);
        TORCH_CHECK(params.o_stride_in_bytes == h * d);
        TORCH_CHECK(params.s_stride_in_bytes == b * h * s * 4);

        params.print_buf = print_buf.data_ptr();
        auto P = torch::zeros({s, b, h, s}, options);
        auto Dfprop = torch::zeros({s, b, h, s}, options);

        params.p_ptr = P.data_ptr();
        params.s_ptr = Dfprop.data_ptr();

        for (int it = 0; it < FLAGS_num_iters; it++)
            launch(launch_params, /*configure=*/false);

        FMHA_CHECK_CUDA(cudaDeviceSynchronize());

        auto O32unpad = convert_fp8_to_fp32(O8unpad, d_scale_o_ref, recipe.at("O").dtype);
        auto O32 = pad(O32unpad, cu_seqlens, s);

        P = P.permute({1, 2, 0, 3});
        Dfprop = Dfprop.permute({1, 2, 0, 3});
        auto Sfprop = Zfprop * (P - Mfprop).exp();
        cout << P.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << Pref.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;

        cout << Sfprop.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << Sref.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;

        cout << O32.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;
        cout << O32ref.index({show_batch, 0, Slice(None, 4), Slice(None, 10)}) << endl;

        cout << "M rel: " << ((Mfprop - M).abs().sum() / M.abs().sum()).item<float>() << endl;
        cout << "Z rel: " << ((Zfprop - Zinv).abs().sum() / Zinv.abs().sum()).item<float>() << endl;
        cout << "S rel: " << ((Sfprop - Sref).abs().sum() / Zinv.abs().sum()).item<float>() << endl;
        cout << "O rel: " << ((O32 - O32ref).abs().sum() / O32ref.abs().sum()).item<float>() << endl;

        auto dmask_d = (S > 0).to(torch::kFloat32);
        auto dmask_f = (Dfprop > 0).to(torch::kFloat32);

        cout << (dmask_d - dmask_f).abs().sum().item<float>() << endl;
        if (maskgen == Maskgen::FULL)
        {

            printf("Frac Zeros in S: %f\n", (S == 0.f).sum().item<float>() / float(S.numel()));
            cout << dmask_d.abs().sum().item<float>() / dmask_d.numel() << " p_keep: " << 1.f - FLAGS_p_dropout << endl;
            cout << dmask_f.abs().sum().item<float>() / dmask_f.numel() << " p_keep: " << 1.f - FLAGS_p_dropout << endl;
        }

        printf("[amax_s] CMP: %f REF(self): %f REF(ref): %f\n", amax_s.item<float>(), Sfprop.abs().max().item<float>(),
            Sref.abs().max().item<float>());

        tmp[0] = amax_o[0];
        printf("[amax_o] CMP: %f CMP(FP8): %f REF(FP8): %f\n", amax_o.item<float>(),
            convert_fp8_to_fp32(
                convert_fp32_to_fp8(tmp, q_scale_o, recipe.at("O").dtype), d_scale_o_ref, recipe.at("O").dtype)[0]
                .item<float>(),
            O32ref.abs().max().item<float>());

        printf("[d_scale_s   ] CMP: %f REF: %f\n", d_scale_s.item<float>(), d_scale_s_ref);
        printf("[d_scale_o   ] CMP: %f REF: %f\n", d_scale_o.item<float>(), d_scale_o_ref);
        printf("[d_scale_dp  ] CMP: %f REF: %f\n", d_scale_dp.item<float>(), d_scale_dp_ref);
        printf("[d_scale_dqkv] CMP: %f REF: %f\n", d_scale_dqkv.item<float>(), d_scale_dqkv_ref);
    }

    if (verbose > 1)
    {
        auto print = torch::from_blob(print_buf.data_ptr(), {8}, {1}, torch::Deleter(), options.dtype(torch::kFloat32));
        cout << print << endl;
    }
    return 0;
}
