// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Wan 2.2 FP8 block fusions: GEMM epilogues and a residual/norm/quant row kernel.

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <map>
#include <mutex>
#include <tuple>

namespace
{

#define CUBLAS_CHECK(x)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t st_ = (x);                                                                                      \
        TORCH_CHECK(st_ == CUBLAS_STATUS_SUCCESS, "cuBLASLt error ", static_cast<int>(st_), " at ", #x);               \
    } while (0)

cublasLtHandle_t ltHandle()
{
    static cublasLtHandle_t h = []
    {
        cublasLtHandle_t t;
        cublasLtCreate(&t);
        return t;
    }();
    return h;
}

struct AlgoKey
{
    int64_t m, n, k;
    int epi;
    int outFp8;

    bool operator<(AlgoKey const& o) const
    {
        return std::tie(m, n, k, epi, outFp8) < std::tie(o.m, o.n, o.k, o.epi, o.outFp8);
    }
};

// epilogue: 0 none, 1 bias, 2 bias + tanh GELU.
void gemm_fp8(torch::Tensor a, torch::Tensor w, torch::Tensor scale_a, torch::Tensor scale_w,
    std::optional<torch::Tensor> bias, int64_t epilogue, std::optional<torch::Tensor> d_scale, torch::Tensor out)
{
    TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn && w.scalar_type() == at::kFloat8_e4m3fn);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && out.is_contiguous());
    int64_t const m = a.size(0), k = a.size(1), n = w.size(0);
    TORCH_CHECK(w.size(1) == k && out.size(0) == m && out.size(1) == n);
    bool const outFp8 = out.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK(outFp8 || out.scalar_type() == at::kBFloat16);
    TORCH_CHECK(!outFp8 || d_scale.has_value(), "FP8 output needs d_scale");
    TORCH_CHECK(epilogue == 0 || bias.has_value(), "bias epilogues need bias");

    cublasLtMatmulDesc_t op;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
    // Column-major: D^T = W (A, transposed) x a^T (B).
    void const* sA = scale_w.data_ptr();
    void const* sB = scale_a.data_ptr();
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sA, sizeof(sA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sB, sizeof(sB)));
    int8_t fastAcc = 1;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAcc, sizeof(fastAcc)));
    if (outFp8)
    {
        void const* sD = d_scale->data_ptr();
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &sD, sizeof(sD)));
    }
    cublasLtEpilogue_t epi = epilogue == 2 ? CUBLASLT_EPILOGUE_GELU_BIAS
        : epilogue == 1                    ? CUBLASLT_EPILOGUE_BIAS
                                           : CUBLASLT_EPILOGUE_DEFAULT;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    if (epilogue != 0)
    {
        TORCH_CHECK(bias->scalar_type() == at::kBFloat16 && bias->numel() == n);
        void const* bp = bias->data_ptr();
        cudaDataType_t bt = CUDA_R_16BF;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
    }
    cudaDataType_t const dType = outFp8 ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
    cublasLtMatrixLayout_t lA, lB, lC, lD;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lA, CUDA_R_8F_E4M3, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lB, CUDA_R_8F_E4M3, k, m, k));
    // FP8 output still needs a BF16 C descriptor.
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lC, CUDA_R_16BF, n, m, n));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lD, dType, n, m, n));

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    static std::map<AlgoKey, cublasLtMatmulAlgo_t> algos;
    static std::mutex mu;
    size_t const wsBytes = 64ull << 20;
    static torch::Tensor ws;
    cublasLtMatmulAlgo_t algo;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!ws.defined())
            ws = torch::empty({static_cast<int64_t>(wsBytes)}, a.options().dtype(at::kByte));
        AlgoKey key{m, n, k, static_cast<int>(epilogue), outFp8 ? 1 : 0};
        auto it = algos.find(key);
        if (it == algos.end())
        {
            cublasLtMatmulPreference_t pref;
            CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
            CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes)));
            cublasLtMatmulHeuristicResult_t res[1];
            int got = 0;
            CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle(), op, lA, lB, lC, lD, pref, 1, res, &got));
            cublasLtMatmulPreferenceDestroy(pref);
            TORCH_CHECK(got > 0, "no cuBLASLt algo for FP8 GEMM m=", m, " n=", n, " k=", k, " epi=", epilogue);
            it = algos.emplace(key, res[0].algo).first;
        }
        algo = it->second;
    }
    float alpha = 1.f, beta = 0.f;
    // C may alias D only when dtypes match.
    void* cPtr = outFp8 ? nullptr : out.data_ptr();
    CUBLAS_CHECK(cublasLtMatmul(ltHandle(), op, &alpha, w.data_ptr(), lA, a.data_ptr(), lB, &beta, cPtr, lC,
        out.data_ptr(), lD, &algo, ws.data_ptr(), wsBytes, stream));
    cublasLtMatrixLayoutDestroy(lA);
    cublasLtMatrixLayoutDestroy(lB);
    cublasLtMatrixLayoutDestroy(lC);
    cublasLtMatrixLayoutDestroy(lD);
    cublasLtMatmulDescDestroy(op);
}

// Row kernel, D = 5120; thread mapping matches fusedAdaptiveLayerNormKernel.
// mode: 0 residual only, 1 LayerNorm affine, 2 AdaLN.
constexpr int kD = 5120;
constexpr int kBlock = 128;
constexpr int kElts = kD / kBlock; // 40
constexpr int kChunks = kElts / 8; // 5

__device__ __forceinline__ void load8(__nv_bfloat16 const* p, float* v)
{
    uint4 const u = *reinterpret_cast<uint4 const*>(p);
    __nv_bfloat162 const* h = reinterpret_cast<__nv_bfloat162 const*>(&u);
#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        float2 f = __bfloat1622float2(h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

template <int MODE, bool HAS_Y, bool HAS_GATE, bool HAS_YBIAS, bool HAS_Q>
__global__ void __launch_bounds__(kBlock, 4) residLnQuantKernel(__nv_bfloat16 const* x, __nv_bfloat16 const* y,
    __nv_bfloat16 const* ybias, float const* gate, float const* w, float const* b, int seqPerBatch, float eps,
    float const* invScale, __nv_bfloat16* xOut, __nv_fp8_e4m3* qOut)
{
    int const tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int const row = blockIdx.x;
    int64_t const base = static_cast<int64_t>(row) * kD;
    int const batch = row / seqPerBatch;
    int64_t const modBase = MODE == 2 ? static_cast<int64_t>(batch) * kD : 0;
    // Issue all 128-bit loads up front.
    uint4 xr[kChunks], yr[kChunks], br[kChunks];
    float4 gr[2 * kChunks], wr[2 * kChunks], bbr[2 * kChunks];
#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const off = (c * kBlock + tid) * 8;
        xr[c] = *reinterpret_cast<uint4 const*>(x + base + off);
        if constexpr (HAS_Y)
            yr[c] = *reinterpret_cast<uint4 const*>(y + base + off);
        if constexpr (HAS_YBIAS)
            br[c] = *reinterpret_cast<uint4 const*>(ybias + off);
        if constexpr (HAS_GATE)
        {
            gr[2 * c] = *reinterpret_cast<float4 const*>(gate + static_cast<int64_t>(batch) * kD + off);
            gr[2 * c + 1] = *reinterpret_cast<float4 const*>(gate + static_cast<int64_t>(batch) * kD + off + 4);
        }
        if constexpr (MODE != 0)
        {
            wr[2 * c] = *reinterpret_cast<float4 const*>(w + modBase + off);
            wr[2 * c + 1] = *reinterpret_cast<float4 const*>(w + modBase + off + 4);
            bbr[2 * c] = *reinterpret_cast<float4 const*>(b + modBase + off);
            bbr[2 * c + 1] = *reinterpret_cast<float4 const*>(b + modBase + off + 4);
        }
    }
    float xv[kElts];
#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const off = (c * kBlock + tid) * 8;
        __nv_bfloat162 const* xh = reinterpret_cast<__nv_bfloat162 const*>(&xr[c]);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            float2 f = __bfloat1622float2(xh[i]);
            xv[c * 8 + 2 * i] = f.x;
            xv[c * 8 + 2 * i + 1] = f.y;
        }
        if constexpr (HAS_Y)
        {
            __nv_bfloat162 const* yh = reinterpret_cast<__nv_bfloat162 const*>(&yr[c]);
            __nv_bfloat162 const* bh = reinterpret_cast<__nv_bfloat162 const*>(&br[c]);
            float const* gv = reinterpret_cast<float const*>(&gr[2 * c]);
            __nv_bfloat162 o[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 yy = __bfloat1622float2(yh[i]);
                if constexpr (HAS_YBIAS)
                {
                    // y + bias in bf16, as torch does.
                    float2 bb = __bfloat1622float2(bh[i]);
                    yy = __bfloat1622float2(__floats2bfloat162_rn(yy.x + bb.x, yy.y + bb.y));
                }
                float2 r;
                // Round the product before the add; no FMA.
                r.x = HAS_GATE ? __fadd_rn(xv[c * 8 + 2 * i], __fmul_rn(yy.x, gv[2 * i]))
                               : __fadd_rn(xv[c * 8 + 2 * i], yy.x);
                r.y = HAS_GATE ? __fadd_rn(xv[c * 8 + 2 * i + 1], __fmul_rn(yy.y, gv[2 * i + 1]))
                               : __fadd_rn(xv[c * 8 + 2 * i + 1], yy.y);
                o[i] = __floats2bfloat162_rn(r.x, r.y);
                float2 rf = __bfloat1622float2(o[i]);
                xv[c * 8 + 2 * i] = rf.x;
                xv[c * 8 + 2 * i + 1] = rf.y;
            }
            *reinterpret_cast<uint4*>(xOut + base + off) = *reinterpret_cast<uint4 const*>(o);
        }
    }
    if constexpr (MODE == 0)
        return;
    __shared__ float wsum[kBlock / 32], wsq[kBlock / 32], stats[2];
    float s = 0.f, s2 = 0.f;
#pragma unroll
    for (int i = 0; i < kElts; i++)
    {
        s += xv[i];
        s2 += xv[i] * xv[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o /= 2)
    {
        s += __shfl_xor_sync(0xffffffffu, s, o);
        s2 += __shfl_xor_sync(0xffffffffu, s2, o);
    }
    if (lane == 0)
    {
        wsum[warp] = s;
        wsq[warp] = s2;
    }
    __syncthreads();
    if (warp == 0)
    {
        float a = lane < kBlock / 32 ? wsum[lane] : 0.f;
        float a2 = lane < kBlock / 32 ? wsq[lane] : 0.f;
#pragma unroll
        for (int o = 16; o > 0; o /= 2)
        {
            a += __shfl_xor_sync(0xffffffffu, a, o);
            a2 += __shfl_xor_sync(0xffffffffu, a2, o);
        }
        if (lane == 0)
        {
            float const invD = 1.0f / static_cast<float>(kD);
            float const mean = a * invD;
            float const var = a2 * invD - mean * mean;
            stats[0] = mean;
            stats[1] = rsqrtf(var + eps);
        }
    }
    __syncthreads();
    float const mean = stats[0], rstd = stats[1];
    float const qmul = HAS_Q ? invScale[0] : 1.f;
#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const off = (c * kBlock + tid) * 8;
        float const* wv = reinterpret_cast<float const*>(&wr[2 * c]);
        float const* bv = reinterpret_cast<float const*>(&bbr[2 * c]);
        __nv_fp8x2_storage_t packed[4];
#pragma unroll
        for (int i = 0; i < 8; i += 2)
        {
            float yv[2];
#pragma unroll
            for (int j = 0; j < 2; j++)
            {
                float const xn = (xv[c * 8 + i + j] - mean) * rstd;
                float const ww = MODE == 2 ? 1.0f + wv[i + j] : wv[i + j];
                // Round to bf16, then saturating static FP8 quant.
                yv[j] = __bfloat162float(__float2bfloat16_rn(xn * ww + bv[i + j])) * qmul;
            }
            packed[i / 2] = __nv_cvt_float2_to_fp8x2(make_float2(yv[0], yv[1]), __NV_SATFINITE, __NV_E4M3);
        }
        *reinterpret_cast<uint2*>(qOut + base + off) = *reinterpret_cast<uint2 const*>(packed);
    }
}

// x [T, D] bf16; outputs x_out (if y given) and q_out (if mode != 0).
void resid_ln_quant(torch::Tensor x, std::optional<torch::Tensor> y, std::optional<torch::Tensor> ybias,
    std::optional<torch::Tensor> gate, int64_t mode, std::optional<torch::Tensor> w, std::optional<torch::Tensor> b,
    int64_t seq_per_batch, double eps, std::optional<torch::Tensor> inv_scale, std::optional<torch::Tensor> x_out,
    std::optional<torch::Tensor> q_out)
{
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.size(-1) == kD);
    int64_t const rows = x.numel() / kD;
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto ptr = [](std::optional<torch::Tensor> const& t) -> void* { return t.has_value() ? t->data_ptr() : nullptr; };
    TORCH_CHECK(!y.has_value() || (y->is_contiguous() && x_out.has_value()));
    TORCH_CHECK(mode == 0 || (q_out.has_value() && inv_scale.has_value() && w.has_value() && b.has_value()));
    auto const* xp = reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr());
    auto const* yp = reinterpret_cast<__nv_bfloat16 const*>(ptr(y));
    auto const* ybp = reinterpret_cast<__nv_bfloat16 const*>(ptr(ybias));
    auto const* gp = reinterpret_cast<float const*>(ptr(gate));
    auto const* wp = reinterpret_cast<float const*>(ptr(w));
    auto const* bp = reinterpret_cast<float const*>(ptr(b));
    auto const* sp = reinterpret_cast<float const*>(ptr(inv_scale));
    auto* xo = reinterpret_cast<__nv_bfloat16*>(ptr(x_out));
    auto* qo = reinterpret_cast<__nv_fp8_e4m3*>(ptr(q_out));
    int const spb = static_cast<int>(seq_per_batch);
    float const e = static_cast<float>(eps);
    bool const hy = y.has_value(), hg = gate.has_value(), hb = ybias.has_value();
#define RLQ(MODE, HY, HG, HB, HQ)                                                                                      \
    residLnQuantKernel<MODE, HY, HG, HB, HQ><<<rows, kBlock, 0, stream>>>(xp, yp, ybp, gp, wp, bp, spb, e, sp, xo, qo)
    if (mode == 0)
    {
        TORCH_CHECK(hy, "mode 0 needs y");
        if (hg && hb)
            RLQ(0, true, true, true, false);
        else if (hg)
            RLQ(0, true, true, false, false);
        else if (hb)
            RLQ(0, true, false, true, false);
        else
            RLQ(0, true, false, false, false);
    }
    else if (mode == 1)
    {
        if (!hy)
            RLQ(1, false, false, false, true);
        else if (hg && hb)
            RLQ(1, true, true, true, true);
        else if (hg)
            RLQ(1, true, true, false, true);
        else if (hb)
            RLQ(1, true, false, true, true);
        else
            RLQ(1, true, false, false, true);
    }
    else
    {
        if (!hy)
            RLQ(2, false, false, false, true);
        else if (hg && hb)
            RLQ(2, true, true, true, true);
        else if (hg)
            RLQ(2, true, true, false, true);
        else if (hb)
            RLQ(2, true, false, true, true);
        else
            RLQ(2, true, false, false, true);
    }
#undef RLQ
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_fp8", &gemm_fp8);
    m.def("resid_ln_quant", &resid_ln_quant);
}
