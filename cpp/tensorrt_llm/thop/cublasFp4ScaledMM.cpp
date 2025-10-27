/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "userbuffersTensor.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <unordered_map>

using torch::Tensor;

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::check;
using tensorrt_llm::common::CublasMMWrapper;

// Helper function: Get or create a workspace tensor for the given device
// Workspace is reused across multiple GEMM calls to avoid repeated allocation
inline at::Tensor const& getWorkspaceTensor(c10::Device device)
{
    thread_local std::unordered_map<int, at::Tensor> workspace_tensors;
    int device_id = device.index();

    if (workspace_tensors.find(device_id) == workspace_tensors.end())
    {
        workspace_tensors[device_id]
            = torch::empty(CUBLAS_WORKSPACE_SIZE, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    }

    return workspace_tensors[device_id];
}

// Helper function: Convert PyTorch ScalarType to CUDA datatype for FP4 GEMM output
inline cudaDataType_t getCudaDataType(at::ScalarType dtype)
{
    if (dtype == at::ScalarType::Half)
    {
        return CUDA_R_16F;
    }
    else if (dtype == at::ScalarType::BFloat16)
    {
        return CUDA_R_16BF;
    }
    else if (dtype == at::ScalarType::Float)
    {
        return CUDA_R_32F;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            false, "Unsupported output dtype for FP4 GEMM. Supported types: Float16, BFloat16, Float32");
        return CUDA_R_16BF; // Unreachable, but satisfy compiler
    }
}

void cublas_fp4_gemm_caller(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b, torch::Tensor const& alpha)
{
    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[0];
    int32_t k_compressed = a.sizes()[1];
    int32_t k = k_compressed * 2;

    // Use device-aware thread-local CublasMMWrapper for FP4 GEMM
    at::cuda::CUDAGuard deviceGuard(a.device());

    thread_local std::unordered_map<int, std::shared_ptr<CublasMMWrapper>> cublasWrappers;
    auto& cublasWrapper = cublasWrappers[a.get_device()];
    if (!cublasWrapper)
    {
        auto cublasHandle = getCublasHandle();
        auto cublasLtHandle = getCublasLtHandle();
        cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
    }

    // Set FP4 configuration based on output tensor dtype
    cudaDataType_t outType = getCudaDataType(out.scalar_type());
    cublasWrapper->setFP4GemmConfig(outType);

    // Get workspace (reuse cached workspace for this device)
    auto const& workspace = getWorkspaceTensor(a.device());

    // Get stream
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    // Get data pointers
    auto* a_ptr = static_cast<void*>(a.data_ptr());
    auto* b_ptr = static_cast<void*>(b.data_ptr());
    auto* out_ptr = static_cast<void*>(out.data_ptr());
    auto* ws_ptr = static_cast<void*>(workspace.data_ptr());

    // Convert scaling factors to __nv_fp8_e4m3 format for cuBLASLt
    void const* a_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_a.data_ptr());
    void const* b_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_b.data_ptr());

    // Validate pointers
    TLLM_CHECK_WITH_INFO(a_sf_ptr != nullptr, "a_sf_ptr is null");
    TLLM_CHECK_WITH_INFO(b_sf_ptr != nullptr, "b_sf_ptr is null");

    // Validate alpha tensor before accessing data
    TLLM_CHECK_WITH_INFO(alpha.numel() > 0, "Alpha tensor is empty");
    TLLM_CHECK_WITH_INFO(alpha.dtype() == torch::kFloat32, "Alpha tensor must be float32");

    auto* alpha_ptr = alpha.data_ptr<float>();

    TLLM_CHECK_WITH_INFO(alpha_ptr != nullptr, "alpha_ptr is null");

    // Set workspace and stream
    cublasWrapper->setStream(stream);
    cublasWrapper->setWorkspace(ws_ptr);

    // Perform FP4 GEMM using CublasMMWrapper
    // Matrix layout conversion for cuBLASLt:
    //   PyTorch uses row-major layout: A[m, k] x B[n, k]^T = C[m, n]
    //   cuBLASLt expects column-major layout: B^T[k, n] x A^T[k, m] = C[m, n]
    // We achieve this conversion by:
    //   1. Swapping A and B matrices (b_ptr comes before a_ptr)
    //   2. Using CUBLAS_OP_T for first matrix, CUBLAS_OP_N for second
    //   3. Passing dimensions as (n, m, k) instead of (m, n, k)
    //   4. Swapping scaling factors to match (b_sf_ptr, a_sf_ptr)
    // Note: beta is always 0 and is managed internally by BlockScaleGemm
    cublasWrapper->BlockScaleGemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, b_ptr, k, // B matrix (swapped to first position)
        a_ptr, k,                                                              // A matrix (swapped to second position)
        out_ptr, n,                                                            // Output: C[m, n] in row-major
        b_sf_ptr, a_sf_ptr,                                                    // Scaling factors (also swapped)
        alpha_ptr);                                                            // Uses default algorithm (nullptr)
}

} // namespace

// CublasLt FP4 GEMM Runner with auto-tuning support
class CublasLtFP4GemmRunner : public torch::CustomClassHolder
{
public:
    explicit CublasLtFP4GemmRunner(at::ScalarType outputDtype)
        : mOutputDtype(outputDtype)
    {
    }

    // Get number of heuristic algorithms for a given matrix shape
    int64_t getNumHeuristicAlgos(
        at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1_scale, at::Tensor const& mat2_scale)
    {
        int m = mat1.size(0);
        int k_compressed = mat1.size(1);
        int k = k_compressed * 2; // FP4 is 2 elements per byte
        int n = mat2.size(0);

        auto& cache = getOrCreateAlgoCache(m, k, n, mat1.device(), mat1_scale, mat2_scale);
        size_t num_algos = cache.heuristics.size();
        return static_cast<int64_t>(num_algos);
    }

    // Run GEMM with specified tactic (-1 for default/best)
    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1_scale,
        at::Tensor const& mat2_scale, at::Tensor const& alpha, bool to_userbuffers, int64_t tactic) const
    {
        int m = mat1.size(0);
        int k_compressed = mat1.size(1);
        int k = k_compressed * 2;
        int n = mat2.size(0);

        // Prepare output tensor
        at::Tensor out;
        std::vector<int64_t> output_size = {m, n};

        if (to_userbuffers)
        {
            out = torch_ext::create_userbuffers_tensor(output_size, mOutputDtype).first;
        }
        else
        {
            out = at::empty(output_size, mat1.options().dtype(mOutputDtype));
        }

        // Get algorithm cache
        auto& cache = getOrCreateAlgoCache(m, k, n, mat1.device(), mat1_scale, mat2_scale);

        // Select algorithm
        bool has_algo = false;
        cublasLtMatmulAlgo_t const* algo_ptr = nullptr;

        if (tactic >= 0 && tactic < static_cast<int64_t>(cache.heuristics.size()))
        {
            // Use specified tactic
            algo_ptr = &cache.heuristics[tactic].algo;
            has_algo = true;
            TLLM_LOG_DEBUG(
                "CublasLtFP4GemmRunner: Using specified tactic %ld (out of %zu) for shape (m=%d, n=%d, k=%d)", tactic,
                cache.heuristics.size(), m, n, k);
        }
        else if (tactic == -1 && !cache.heuristics.empty())
        {
            // Use best tactic (default is first one)
            int64_t best_idx
                = cache.best_tactic < static_cast<int64_t>(cache.heuristics.size()) ? cache.best_tactic : 0;
            algo_ptr = &cache.heuristics[best_idx].algo;
            has_algo = true;
            TLLM_LOG_DEBUG("CublasLtFP4GemmRunner: Using best tactic %ld (out of %zu) for shape (m=%d, n=%d, k=%d)",
                best_idx, cache.heuristics.size(), m, n, k);
        }

        // Execute GEMM (beta is always 0 and is managed internally)
        if (has_algo)
        {
            cublas_fp4_gemm_caller_with_algo(out, mat1, mat2, mat1_scale, mat2_scale, alpha, *algo_ptr, mOutputDtype);
        }
        else
        {
            // Fall back to default (no algorithm specified)
            TLLM_LOG_WARNING(
                "CublasLtFP4GemmRunner: No valid algorithm found (tactic=%ld, available=%zu), falling back to default "
                "for shape (m=%d, n=%d, k=%d)",
                tactic, cache.heuristics.size(), m, n, k);
            cublas_fp4_gemm_caller(out, mat1, mat2, mat1_scale, mat2_scale, alpha);
        }

        return out;
    }

private:
    struct AlgoCache
    {
        std::vector<cublasLtMatmulHeuristicResult_t> heuristics;
        int64_t best_tactic = 0; // Index of the best algorithm
    };

    // Cache key: (m, k, n, device_id, output_dtype) for algorithm list storage
    // Different output dtypes may have different optimal algorithms
    using ShapeKey = std::tuple<int, int, int, int, int>;

    struct ShapeKeyHash
    {
        size_t operator()(ShapeKey const& k) const
        {
            // Use boost-style hash_combine for better distribution
            size_t seed = 0;
            hash_combine(seed, std::get<0>(k));
            hash_combine(seed, std::get<1>(k));
            hash_combine(seed, std::get<2>(k));
            hash_combine(seed, std::get<3>(k));
            hash_combine(seed, std::get<4>(k));
            return seed;
        }

    private:
        // Standard hash combination algorithm (Boost-style)
        template <typename T>
        static void hash_combine(size_t& seed, T const& v)
        {
            std::hash<T> hasher;
            seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    };

    mutable std::unordered_map<ShapeKey, AlgoCache, ShapeKeyHash> mAlgoCache;
    at::ScalarType mOutputDtype;

    AlgoCache& getOrCreateAlgoCache(
        int m, int k, int n, c10::Device device, at::Tensor const& mat1_scale, at::Tensor const& mat2_scale) const
    {
        ShapeKey key = std::make_tuple(m, k, n, device.index(), static_cast<int>(mOutputDtype));

        if (mAlgoCache.find(key) == mAlgoCache.end())
        {
            TLLM_LOG_DEBUG(
                "CublasLtFP4GemmRunner: Cache miss for shape (m=%d, k=%d, n=%d, device=%d, dtype=%d), creating new "
                "cache entry",
                m, k, n, device.index(), static_cast<int>(mOutputDtype));

            AlgoCache cache;

            // Create cublas wrapper
            at::cuda::CUDAGuard deviceGuard(device);
            auto cublasHandle = getCublasHandle();
            auto cublasLtHandle = getCublasLtHandle();
            auto cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);

            // Set FP4 configuration
            cudaDataType_t outType = mOutputDtype == at::ScalarType::Half
                ? CUDA_R_16F
                : (mOutputDtype == at::ScalarType::BFloat16 ? CUDA_R_16BF : CUDA_R_32F);

            cublasWrapper->setFP4GemmConfig(outType);

            // Create descriptors
            cublasWrapper->createDescriptors(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, k, k, n, 0);

            // Use provided scale tensors for descriptor setup
            // FP4 GEMM always requires scale tensors
            void* a_sf_ptr = const_cast<void*>(reinterpret_cast<void const*>(mat1_scale.data_ptr()));
            void* b_sf_ptr = const_cast<void*>(reinterpret_cast<void const*>(mat2_scale.data_ptr()));

            // Set scale descriptors (required for FP4 GEMM heuristics)
            cublasWrapper->setScaleDescriptors(a_sf_ptr, b_sf_ptr);

            // Get heuristic algorithms
            auto heuristics = cublasWrapper->getTactics(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, k, k, n);

            // Filter valid algorithms
            for (auto const& h : heuristics)
            {
                if (h.state == CUBLAS_STATUS_SUCCESS && h.workspaceSize <= CUBLAS_WORKSPACE_SIZE)
                {
                    cache.heuristics.push_back(h);
                }
            }

            TLLM_LOG_DEBUG(
                "CublasLtFP4GemmRunner: Found %zu valid algorithms for shape (m=%d, k=%d, n=%d) on device %d",
                cache.heuristics.size(), m, k, n, device.index());

            if (cache.heuristics.empty())
            {
                TLLM_LOG_WARNING(
                    "CublasLtFP4GemmRunner: No valid cuBLASLt algorithms found for shape (m=%d, k=%d, n=%d), will fall "
                    "back to default",
                    m, k, n);
            }

            cublasWrapper->destroyDescriptors();

            mAlgoCache[key] = std::move(cache);
        }
        else
        {
            TLLM_LOG_DEBUG(
                "CublasLtFP4GemmRunner: Cache hit for shape (m=%d, k=%d, n=%d, device=%d, dtype=%d), %zu algorithms "
                "available",
                m, k, n, device.index(), static_cast<int>(mOutputDtype), mAlgoCache[key].heuristics.size());
        }

        return mAlgoCache[key];
    }

    // Helper function to run GEMM with a specific algorithm
    static void cublas_fp4_gemm_caller_with_algo(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
        torch::Tensor const& scale_a, torch::Tensor const& scale_b, torch::Tensor const& alpha,
        cublasLtMatmulAlgo_t const& algo, at::ScalarType output_dtype)
    {
        int32_t m = a.sizes()[0];
        int32_t n = b.sizes()[0];
        int32_t k_compressed = a.sizes()[1];
        int32_t k = k_compressed * 2;

        at::cuda::CUDAGuard deviceGuard(a.device());

        thread_local std::unordered_map<int, std::shared_ptr<CublasMMWrapper>> cublasWrappers;
        auto& cublasWrapper = cublasWrappers[a.get_device()];
        if (!cublasWrapper)
        {
            auto cublasHandle = getCublasHandle();
            auto cublasLtHandle = getCublasLtHandle();
            cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
        }

        // Set FP4 configuration with correct output type
        cudaDataType_t outType = getCudaDataType(output_dtype);
        cublasWrapper->setFP4GemmConfig(outType);

        // Get workspace (reuse cached workspace for this device)
        auto const& workspace = getWorkspaceTensor(a.device());

        auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

        auto* a_ptr = static_cast<void*>(a.data_ptr());
        auto* b_ptr = static_cast<void*>(b.data_ptr());
        auto* out_ptr = static_cast<void*>(out.data_ptr());
        auto* ws_ptr = static_cast<void*>(workspace.data_ptr());

        void const* a_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_a.data_ptr());
        void const* b_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_b.data_ptr());

        // Validate alpha tensor before accessing data
        TLLM_CHECK_WITH_INFO(alpha.numel() > 0, "Alpha tensor is empty");
        TLLM_CHECK_WITH_INFO(alpha.dtype() == torch::kFloat32, "Alpha tensor must be float32");

        auto* alpha_ptr = alpha.data_ptr<float>();

        TLLM_CHECK_WITH_INFO(alpha_ptr != nullptr, "alpha_ptr is null");

        cublasWrapper->setStream(stream);
        cublasWrapper->setWorkspace(ws_ptr);

        // Matrix layout conversion for cuBLASLt (same as in cublas_fp4_gemm_caller):
        //   PyTorch uses row-major layout: A[m, k] x B[n, k]^T = C[m, n]
        //   cuBLASLt expects column-major layout: B^T[k, n] x A^T[k, m] = C[m, n]
        // Conversion is achieved by:
        //   1. Swapping A and B matrices (b_ptr comes before a_ptr)
        //   2. Using CUBLAS_OP_T for first matrix, CUBLAS_OP_N for second
        //   3. Passing dimensions as (n, m, k) instead of (m, n, k)
        //   4. Swapping scaling factors to match matrices (b_sf_ptr, a_sf_ptr)

        // Use BlockScaleGemm with specified algorithm for autotuning
        // Note: beta is always 0 and is managed internally by BlockScaleGemm
        cublasWrapper->BlockScaleGemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, b_ptr,
            k,                  // B matrix (swapped to first position)
            a_ptr, k,           // A matrix (swapped to second position)
            out_ptr, n,         // Output: C[m, n] in row-major
            b_sf_ptr, a_sf_ptr, // Scaling factors (also swapped)
            alpha_ptr,          // Alpha
            &algo);             // Use specified algorithm
    }
};

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::CublasLtFP4GemmRunner>("CublasLtFP4GemmRunner")
        .def(torch::init<at::ScalarType>())
        .def("run_gemm", &torch_ext::CublasLtFP4GemmRunner::runGemm)
        .def("get_num_heuristic_algos", &torch_ext::CublasLtFP4GemmRunner::getNumHeuristicAlgos);
}
