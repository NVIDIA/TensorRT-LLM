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
// Launchers of this target's elementwise, norm and mHC kernels (elementwise.cu). All take an explicit stream and only
// device pointers for per-step data, so they are CUDA-graph capturable.
#pragma once

#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/kernels/allreduce.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include "glm5_next/glm_5_3_flash_nvfp4__sm_10x__tp4/csrc/common.h"

TRTLLM_MV2_NAMESPACE_BEGIN

namespace glm
{
namespace k
{

// ---------------------------------------------------------------- elementwise / norms
// streams[t, i, :] = embed[ids[t], :] for i < hc
void embed_streams(bf16* streams, bf16 const* embed, int const* ids, int T, int hidden, int hc, cudaStream_t st);
// h = rmsnorm(mean_i streams[t, i, :], w)
void hc_head_norm(bf16* out, bf16 const* streams, bf16 const* w, int T, int hidden, int hc, float eps, cudaStream_t st);
// MLA: rmsnorm of q (columns [0, Dq) of x) into q_out with its FP8 1x128 quantization (qq / qs,
// optional) and of the latent (columns [Dq, Dq + Dkv)) into kv_out, in one launch (a row's sum of
// squares: per thread over its columns tid + 256 k in increasing k, then block_sum over the 256
// threads; 512 <= Dq, Dkv < 2048)
// kv_cache (decode, with cache_meta): the latent rows written to the tokens' rows of that paged
// latent cache instead of kv_out
struct MlaTokMeta;
void mla_qkv_norm(bf16* q_out, int ldq, fp8* qq, float* qs, bf16* kv_out, int ldkv, bf16 const* x, int ldx,
    bf16 const* wq, bf16 const* wkv, int T, int Dq, int Dkv, float eps, cudaStream_t st, bf16* kv_cache = nullptr,
    MlaTokMeta const* cache_meta = nullptr);
// out = bf16(clamp-SwiGLU(gu[:, :I], gu[:, I:])). m_dev (optional): device-side row count <= M;
// rows past it are skipped. gu64: gate / up columns interleaved by 64 (kGu64: [gate 0..63 |
// up 0..63 | gate 64..127 | ...], the fused MoE kernel's weight layout)
void swiglu_bf16(bf16* out, bf16 const* gu, int M, int I, float limit, cudaStream_t st, int const* m_dev = nullptr,
    bool gu64 = false);

// dst[i, :] = src[idx[i], :] for rows of `row_elems` bf16 values (row_elems % 8 == 0)
void gather_rows(bf16* dst, bf16 const* src, int const* idx, int n, int row_elems, cudaStream_t st);
// bulk L2 prefetch of [p, p + bytes) in chunks of `chunk` bytes (16 B .. 512 KB) from up to ctas
// CTAs of 32 threads (plain launch on st)
void l2_prefetch(void const* p, size_t bytes, int ctas, size_t chunk, cudaStream_t st);

// bulk L2 prefetch of a MoE layer's expert weights recorded at the previous step: hist = the fused
// kernel's routing history (moe_fp4.h rt_hist: count, local experts, token counts); the experts by
// descending token count, each expert's 4 segments seg[s] + g * stride[s] (w13, sf13, w2, sf2);
// two_phase: every expert's gate / up segments (0, 1) first, then the down ones (2, 3); at most
// cap bytes, whole segments of whole experts (plain launch on st, ctas CTAs of 128 threads)
struct MoeL2pf
{
    int const* hist = nullptr;
    int hist_max = 0;
    int n_groups = 0; // (recorded experts outside [0, n_groups) are skipped)
    uint8_t const* seg[4] = {};
    size_t stride[4] = {};
    size_t cap = 0;
    bool two_phase = true;
    bool by_count = true; // (false: the recorded order, i.e. the fused kernel's work item order)
};

void moe_l2_prefetch(MoeL2pf const& a, int ctas, size_t chunk, cudaStream_t st);
// reads [p, p + bytes) to evict the L2 (timing); sink: one float of device memory
void l2_flush(void const* p, size_t bytes, float* sink, cudaStream_t st);

// ---------------------------------------------------------------- MTP layer
// out[t] = [rmsnorm(e_t) * enorm | rmsnorm(hidden[t]) * hnorm] ([T, 2 * hidden]) where e_t is
// the embedding of id_t = src[t] >= 0 ? sampled[src[t]] : ids[t] (src may be null), zeroed at
// position 0.
void mtp_eh_norm(bf16* out, bf16 const* embed, int const* ids, int const* src, int const* sampled, int const* positions,
    bf16 const* hidden, bf16 const* enorm, bf16 const* hnorm, int T, int hidden_size, float eps, cudaStream_t st);
// For n rows (row r uses residual row rows ? rows[r] : r): v = resid (+ delta, bf16-rounded,
// written back when write_resid); out[r] = bf16(rmsnorm(v) * w); xq / xs (optional): FP8 1x128
// quantization of out (token-major scales).
void add_rmsnorm(bf16* resid, bf16 const* delta, int const* rows, bf16 const* w, bf16* out, fp8* xq, float* xs, int n,
    int hidden, float eps, bool write_resid, cudaStream_t st);

// ---------------------------------------------------------------- mHC
// Given mixes = streams_flat @ fn^T (fp32 [T, 24]), compute post [T, hc], comb [T, hc, hc] and
// x = rmsnorm(sum_i pre_i * streams[t, i, :], norm_w).
// xq / xs (optional, may be null): FP8 1x128 quantization of x (token-major scales).
// rs_scratch (optional, [T] f32): one read of the streams, Sinkhorn in a second small kernel
// (bitwise the same outputs; hidden 4096).
void mhc_pre_apply(float* post, float* comb, bf16* x, fp8* xq, float* xs, bf16 const* streams, float const* mixes,
    float const* base, float const* scale, bf16 const* norm_w, int T, int hidden, int hc, float rms_eps, float hc_eps,
    int iters, float post_mult, float norm_eps, cudaStream_t st, float* rs_scratch = nullptr);
// Fused sublayer boundary: optional post-mapping of the previous sublayer output x_prev (in
// place on streams, using post/comb), then the pre-mapping of the next sublayer (post/comb
// overwritten, x_out = rmsnorm(sum_i pre_i * streams_i)). xq / xs (optional): FP8 1x128
// quantization of x_out. Few tokens: mhc_fused_groups(T) CTAs per token with a last-CTA
// finish (partial: [T * groups * 32] f32 scratch, counter: [T] ints, zero-initialized, reset by
// the kernel); many tokens: one 8-CTA cluster per token (hidden 4096).
int mhc_fused_groups(int T);
// ar (optional, cluster path only: mhc_fused_cluster(T, hidden)): x_prev is this rank's partial
// sum and the boundary all-reduces it itself (ArFuse), instead of a separate all-reduce kernel.
// x4 / x4s / q4_gs (optional, cluster path only): also x_out's NVFP4 quantization with global scale
// q4_gs (nvfp4_quant swz layout, bitwise its output); returns whether it was written.
// moe_tm (with ar's MoE combine, where mhc_moe_tm_ok): its rows are token-major (pair s of token t at ar->cm.y row
// t K1 + s, moe_fp4's y_tm), read by the kTm instances without the pair positions
// rf_done (with ar_rf_x(*ar), an attention-side boundary): set to true when the boundary wrote the tensor-core router's
// RF copies there, i.e. the RF router (moe_router_mma_ks_rf) may read them; untouched otherwise
bool mhc_fused(bf16* streams, bf16 const* x_prev, float* post, float* comb, bf16* x_out, fp8* xq, float* xs,
    bf16 const* fn, float const* base, float const* scale, bf16 const* norm_w, float* partial, int* counter, int T,
    int hidden, int hc, float rms_eps, float hc_eps, int iters, float post_mult, float norm_eps, cudaStream_t st,
    ArFuse const* ar = nullptr, uint8_t* x4 = nullptr, uint8_t* x4s = nullptr, float q4_gs = 0.f, bool moe_tm = false,
    bool* rf_done = nullptr);
bool mhc_fused_cluster(int T, int hidden);
// the next call's MoE-side boundary at <= 4 tokens (the kMp form) launches its dependents at its start (1), right
// after its PDL wait (2) or after its all-reduce push (3); 0: after its post-poll barrier
void mhc_set_et(int mode);
// the kTrig mode the last mhc_fused call launched (0: none)
int mhc_et_used();
// whether the cluster boundary of T tokens has the deferred Sinkhorn (ArFuse.sk)
bool mhc_sk_ok(int T, int hidden);
// whether the cluster boundary of T tokens after a MoE (tp ranks, K1 pairs per token) reads the MoE's outputs
// token-major (mhc_fused moe_tm: its kTm instances), given the deferred Sinkhorn's input (ArFuse.sk & 2)
bool mhc_moe_tm_ok(int T, int hidden, int tp, int K1);
// streams[t, j, :] = sum_i comb[t, i, j] * streams[t, i, :] + post[t, j] * x[t, :]  (in place)
void mhc_post(
    bf16* streams, bf16 const* x, float const* post, float const* comb, int T, int hidden, int hc, cudaStream_t st);

} // namespace k
} // namespace glm

TRTLLM_MV2_NAMESPACE_END
