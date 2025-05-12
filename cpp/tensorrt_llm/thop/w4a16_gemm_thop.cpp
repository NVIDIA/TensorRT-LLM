#include "w4a16_gemm_thop.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/cutlass_extensions/include/cutlass_extensions/weight_only_quant_op.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace
{

using ActivationT = half;
using WeightPackedT = cutlass::uint4b_t;
using ScaleT = half;
using BiasT = half;
using OutputT = half;

const cutlass::WeightOnlyQuantOp QuantMode = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

using W4A16Runner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<ActivationT, WeightPackedT,
    QuantMode, ScaleT, BiasT, OutputT>;

W4A16Runner& get_w4a16_runner_instance()
{
    static W4A16Runner runner_instance;
    return runner_instance;
}

std::map<std::tuple<int, int, int, int>, tensorrt_llm::cutlass_extensions::CutlassGemmConfig> g_w4a16_tactics_map;
std::mutex g_w4a16_tactics_mutex;

} // namespace

torch::Tensor w4a16_gemm_op(torch::Tensor A_tensor, torch::Tensor B_packed_tensor, torch::Tensor scales_tensor,
    torch::Tensor bias_tensor, int64_t group_size_long)
{
    // --- Input Checks (Essential!) ---
    TORCH_CHECK(A_tensor.is_cuda(), "Activation tensor A must be on CUDA");
    TORCH_CHECK(B_packed_tensor.is_cuda(), "Packed Weight tensor B must be on CUDA");
    TORCH_CHECK(scales_tensor.is_cuda(), "Scales tensor must be on CUDA");

    TORCH_CHECK(A_tensor.scalar_type() == torch::kFloat16, "Activation A must be FP16");
    TORCH_CHECK(B_packed_tensor.scalar_type() == torch::kQUInt4x2 || B_packed_tensor.scalar_type() == torch::kInt8,
        "Packed Weights B must be quint4x2 or Int8 (representing Int4s)");
    TORCH_CHECK(scales_tensor.scalar_type() == torch::kFloat16, "Scales must be FP16");

    TORCH_CHECK(A_tensor.is_contiguous(), "Activation tensor A must be contiguous");
    TORCH_CHECK(B_packed_tensor.is_contiguous(), "Packed weight tensor B must be contiguous");
    TORCH_CHECK(scales_tensor.is_contiguous(), "Scales tensor must be contiguous");

    // --- Get Dimensions ---
    TORCH_CHECK(A_tensor.dim() >= 2, "Activation A must be at least 2D");
    // Assuming B_packed_tensor is [K, N_packed] where N_packed = N_orig // 2
    TORCH_CHECK(B_packed_tensor.dim() == 2, "Packed Weights B must be 2D");

    int M = 0;
    int K_act = 0;
    if (A_tensor.dim() == 2)
    {
        M = A_tensor.size(0);
        K_act = A_tensor.size(1);
    }
    else
    {                              // Handle batched input (e.g. >2D tensor)
        M = A_tensor.size(0);      // First dim is batch
        for (int i = 1; i < A_tensor.dim() - 1; ++i)
            M *= A_tensor.size(i); // Flatten batch dimensions
        K_act = A_tensor.size(A_tensor.dim() - 1);
    }

    int K_weights = B_packed_tensor.size(0);
    int N_packed_int4 = B_packed_tensor.size(1);
    int N_orig = N_packed_int4 * 2;

    TORCH_CHECK(
        K_act == K_weights, "K dimension of A (", K_act, ") must match K dimension of B_packed (", K_weights, ")");
    int K = K_act;
    int group_size = static_cast<int>(group_size_long);

    // --- Prepare Output Tensor ---
    std::vector<int64_t> output_shape;
    if (A_tensor.dim() == 2)
    {
        output_shape.clear();
        output_shape.push_back(static_cast<int64_t>(M));
        output_shape.push_back(static_cast<int64_t>(N_orig));
    }
    else
    {
        output_shape.clear();
        for (int i = 0; i < A_tensor.dim() - 1; ++i)
            output_shape.push_back(A_tensor.size(i));
        output_shape.push_back(N_orig);
    }
    torch::Tensor C_tensor = torch::empty(output_shape, A_tensor.options().dtype(torch::kFloat16));

    void* A_ptr = A_tensor.data_ptr();

    torch::Tensor B_internal = B_packed_tensor;

    // If B is quint4x2, we need to ensure it's correctly handled for CUTLASS
    if (B_packed_tensor.scalar_type() == torch::kQUInt4x2)
    {
        B_internal = B_packed_tensor.view(torch::kUInt8);
    }
    else if (B_packed_tensor.scalar_type() == torch::kInt8)
    {
        // If it's int8, we reinterpret as uint8 for consistency with CUTLASS
        B_internal = B_packed_tensor.view(torch::kUInt8);
    }

    void* B_packed_ptr = B_internal.data_ptr();

    void* scales_ptr = scales_tensor.data_ptr();

    void* bias_ptr = nullptr;
    if (bias_tensor.defined() && bias_tensor.numel() > 0)
    {
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias tensor must be on CUDA if provided and not empty");
        TORCH_CHECK(bias_tensor.scalar_type() == torch::kFloat16, "Bias tensor must be FP16 if provided and not empty");
        TORCH_CHECK(bias_tensor.is_contiguous(), "Bias tensor must be contiguous");
        bias_ptr = bias_tensor.data_ptr();
    }
    void* C_ptr = C_tensor.data_ptr();

    W4A16Runner& runner = get_w4a16_runner_instance();
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig gemm_config_to_run;
    std::tuple<int, int, int, int> tactic_key = std::make_tuple(M, N_orig, K, group_size);
    {
        std::lock_guard<std::mutex> lock(g_w4a16_tactics_mutex);
        auto it = g_w4a16_tactics_map.find(tactic_key);
        if (it != g_w4a16_tactics_map.end())
        {
            gemm_config_to_run = it->second;
        }
        else
        {
            int sm_version = tensorrt_llm::common::getSMVersion();
            std::cout << "[WARNING] w4a16_gemm_thop: Using first *valid* available CUTLASS config. PERFORMANCE MAY BE "
                         "SUBOPTIMAL."
                      << std::endl;
            auto all_configs = runner.getConfigs();
            TORCH_CHECK(!all_configs.empty(), "No CUTLASS GEMM configs available for W4A16 runner.");
            std::cout << "[DEBUG] Available configs from runner.getConfigs():" << std::endl;
            for (auto const& config : all_configs)
            {
                std::cout << "  - TileConfSM80: " << static_cast<int>(config.tile_config_sm80)
                          << ", Stages: " << config.stages
                          << ", SplitKStyle: " << static_cast<int>(config.split_k_style)
                          << ", SplitKFactor: " << config.split_k_factor
                          << ", CudaKernel: " << (config.enableCudaKernel ? "yes" : "no") << std::endl;
            }
            gemm_config_to_run = all_configs[0];
            if (sm_version >= 89 && gemm_config_to_run.stages == 2)
            {
                std::cout << "[WARNING] First config had stages=2 on SM89+, attempting to find another." << std::endl;
                bool found_valid_config = false;
                for (auto const& config : all_configs)
                {
                    bool config_ok = true;
                    if (sm_version >= 89 && config.stages == 2)
                    {
                        config_ok = false;
                    }
                    if (config_ok)
                    {
                        gemm_config_to_run = config;
                        found_valid_config = true;
                        std::cout << "Selected valid fallback config: Stages=" << config.stages << std::endl;
                        break;
                    }
                }
                TORCH_CHECK(found_valid_config, "Could not find a valid CUTLASS config (stages!=2) for SM 89+.");
            }
            else
            {
                std::cout << "Selected fallback config: Stages=" << gemm_config_to_run.stages << std::endl;
            }
            g_w4a16_tactics_map[tactic_key] = gemm_config_to_run;
        }
    }

    size_t workspace_bytes = runner.getWorkspaceSize(M, N_orig, K);
    torch::Tensor workspace_tensor = torch::empty(
        {static_cast<int64_t>(workspace_bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(A_tensor.device()));
    char* workspace_ptr = workspace_bytes > 0 ? reinterpret_cast<char*>(workspace_tensor.data_ptr()) : nullptr;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A_tensor.device().index());

    runner.gemm(A_ptr, B_packed_ptr, scales_ptr,
        nullptr, // NOTE: zeros is not used for this QuantMode - FINEGRAINED_SCALE_ONLY
        bias_ptr,
        1.0f,    // Alpha
        C_ptr, M, N_orig, K, group_size, gemm_config_to_run, workspace_ptr, workspace_bytes, stream);

    return C_tensor;
}

TORCH_LIBRARY(tensorrt_llm, m)
{
    m.def(
        "w4a16_gemm(Tensor A, Tensor B_packed, Tensor scales, Tensor bias, int group_size) -> Tensor", &w4a16_gemm_op);
}

TORCH_LIBRARY_IMPL(tensorrt_llm, CUDA, m)
{
    m.impl("w4a16_gemm", &w4a16_gemm_op);
}
