/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <string>

#include "fused_gated_gemm_util.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fused_gated_gemm/fused_gated_gemm.h"

#include "cutlass/arch/mma.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

using namespace tensorrt_llm::kernels::cutlass_kernels;

Options g_options;

template <typename ElementT, typename LayoutB, typename ElementD2x = ElementT>
struct Buffers
{
    cutlass::HostTensor<ElementT, cutlass::layout::RowMajor> tensor_a;
    cutlass::HostTensor<ElementT, LayoutB> tensor_b;
    cutlass::HostTensor<ElementT, cutlass::layout::RowMajor> tensor_d;
    cutlass::HostTensor<ElementD2x, cutlass::layout::RowMajor> tensor_ref_d_2x;
    cutlass::HostTensor<ElementT, cutlass::layout::RowMajor> tensor_ref_d;
    cutlass::HostTensor<ElementT, cutlass::layout::RowMajor> tensor_c_bias;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Execute a given example GEMM computation
template <typename ElementT, typename LayoutB, typename ElementD2x>
Result run(std::string description, Options& options, Buffers<ElementT, LayoutB, ElementD2x> buffers)
{

    // Display test description
    std::cout << std::endl << description << std::endl;

    // Initialize
    Result result;

    // Zero-initialize test output matrix D
    cutlass::reference::host::TensorFill(buffers.tensor_d.host_view());
    buffers.tensor_d.sync_device();

    // Instantiate CUTLASS kernel depending on templates
    std::shared_ptr<CutlassFusedGatedGemmRunnerInterface> runner
        = std::make_shared<CutlassFusedGatedGemmRunner<typename CutlassToTllmTypeAdapter<ElementT>::type>>();

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size
        = runner->getWorkspaceSize(options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

    // Allocate workspace memory
    cutlass::device_memory::allocation<char> workspace(workspace_size);

    std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> configs = runner->getConfigs();

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float bestTime = std::numeric_limits<float>::max();
    tensorrt_llm::cutlass_extensions::CutlassGemmConfig bestConfig;
    for (auto const& config : configs)
    {
        std::cout << config << std::endl;
        try
        {
            // Correctness / Warmup iteration
            runner->gemm(buffers.tensor_d.device_data(), buffers.tensor_a.device_data(), buffers.tensor_b.device_data(),
                buffers.tensor_c_bias.device_data(), tk::QuantMode{}, options.problem_size.m(),
                options.problem_size.n(), options.problem_size.k(), options.scale_d0, options.scale_d1,
                options.scale_output, config, workspace.get(), workspace_size, 0);
        }
        catch (std::runtime_error& e)
        {
            // We can ignore these error because most are related to SMEM oversubscription
            std::cout << e.what() << std::endl;
            continue;
        }
        // Copy output data from CUTLASS and reference kernel to host for comparison
        buffers.tensor_d.sync_host();

        // Check if output from CUTLASS kernel and reference kernel are equal or not
        if (!options.no_check)
        {
            result.passed = cutlass::reference::host::TensorRelativelyEquals(
                buffers.tensor_d.host_view(), buffers.tensor_ref_d.host_view(), ElementT{1e-3}, ElementT{1e-3});

            EXPECT_TRUE(result.passed);

            double err = cutlass::reference::host::TensorRelativeErrorMetric(
                buffers.tensor_d.host_view(), buffers.tensor_ref_d.host_view());

            std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << " \t Relative error: " << err
                      << std::endl;

            if (!result.passed || options.debug)
            {
                std::cout << "ref_output=\n"
                          << buffers.tensor_ref_d.host_view() << "\noutput=\n"
                          << buffers.tensor_d.host_view() << std::endl;
            }
        }

        // Run profiling loop
        if (options.iterations > 0)
        {
            cudaDeviceSynchronize();
            cudaEventRecord(start, 0);
            for (int iter = 0; iter < options.iterations; ++iter)
            {
                runner->gemm(buffers.tensor_d.device_data(), buffers.tensor_a.device_data(),
                    buffers.tensor_b.device_data(), buffers.tensor_c_bias.device_data(), tk::QuantMode{},
                    options.problem_size.m(), options.problem_size.n(), options.problem_size.k(), options.scale_d0,
                    options.scale_d1, options.scale_output, config, workspace.get(), workspace_size, 0);
            }
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);

            result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
            result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

            std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
            std::cout << "  GFLOPs: " << result.gflops << std::endl;

            if (result.avg_runtime_ms < bestTime)
            {
                bestTime = result.avg_runtime_ms;
                bestConfig = config;
            }
        }
    }

    std::cout << "Best runtime: " << bestTime << " ms" << std::endl;
    std::cout << "Best config: " << bestConfig << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

template <typename ElementT_>
using Activation = cutlass::epilogue::thread::SiLu<ElementT_>;

TEST(GemmSwigluRunner, Sm90FP8)
{
    using ElementT = cutlass::float_e4m3_t;
    using ElementAccumulatorT = float;
    using ElementComputeT = float;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementD2x = float;

    Buffers<ElementT, LayoutB, ElementD2x> buffers;

    // Parse commandline options
    Options options(g_options);

    std::cout << options.iterations << " timing iterations of " << options.problem_size.m() << " x "
              << options.problem_size.n() << " x " << options.problem_size.k() << " matrix-matrix multiply"
              << std::endl;

    if (!options.valid())
    {
        std::cerr << "Invalid problem." << std::endl;
        FAIL();
    }

    if (options.debug)
    {
        std::cout << "scale_d0: " << options.scale_d0 << ", scale_d1: " << options.scale_d1
                  << ", scale_output: " << options.scale_output << std::endl;
    }

    //
    // Initialize GEMM datasets
    //

    // Initialize tensors using CUTLASS helper functions
    buffers.tensor_a.resize(options.problem_size.mk());          // <- Create matrix A with dimensions M x K
    buffers.tensor_b.resize(options.problem_size.kn());          // <- Create matrix B with dimensions K x N
    buffers.tensor_c_bias.resize({1, options.problem_size.n()}); // <- Create broadcast vector with dimensions 1 x N
    buffers.tensor_d.resize(
        options.problem_size_out
            .mn()); // <- Create matrix D with dimensions M x N/2 used to store output from CUTLASS kernel
    buffers.tensor_ref_d_2x.resize(
        options.problem_size
            .mn()); // <- Create temp matrix D with dimensions M x N used to store output from reference kernel
    buffers.tensor_ref_d.resize(
        options.problem_size_out
            .mn()); // <- Create matrix D with dimensions M x N/2 used to store output from reference kernel

    int _init_bits = options.real ? -1 : 0;

    // Fill matrix A on host with uniform-random data [-2, 2]
    if (options.debug)
    {
        cutlass::Array<ElementT, 2> range;
        range[0] = ElementT(256);
        range[1] = ElementT(1);
        cutlass::reference::host::TensorFillLinear(buffers.tensor_a.host_view(), range);
    }
    else
    {
        cutlass::reference::host::TensorFillRandomUniform(
            buffers.tensor_a.host_view(), 1, ElementT(2), ElementT(-2), _init_bits);
    }

    // Fill matrix B on host with uniform-random data [-2, 2]
    if (options.debug)
    {
        cutlass::reference::host::TensorFillIdentity(buffers.tensor_b.host_view());
    }
    else
    {
        cutlass::reference::host::TensorFillRandomUniform(
            buffers.tensor_b.host_view(), 1, ElementT(2), ElementT(-2), _init_bits);
    }

    if (options.debug || !options.has_bias)
    {
        cutlass::reference::host::TensorFill(buffers.tensor_c_bias.host_view());
    }
    else
    {
        cutlass::reference::host::TensorFillRandomUniform(
            buffers.tensor_c_bias.host_view(), 1, ElementT(2), ElementT(-2), _init_bits);
    }

    if (options.debug)
    {
        std::cout << "A=" << std::endl << buffers.tensor_a.host_view() << std::endl;
        std::cout << "B=" << std::endl << buffers.tensor_b.host_view() << std::endl;
        std::cout << "C=" << std::endl << buffers.tensor_c_bias.host_view() << std::endl;
    }

    //
    // Compute reference output
    //

    // Copy data from host to GPU
    buffers.tensor_a.sync_device();
    buffers.tensor_b.sync_device();
    buffers.tensor_c_bias.sync_device();

    // Zero-initialize reference output matrix D
    cutlass::reference::host::TensorFill(buffers.tensor_ref_d_2x.host_view());
    buffers.tensor_ref_d_2x.sync_device();

    // Create instantiation for device reference gemm kernel
    // Reference device GEMM implementation type
    using DeviceGemmReference = cutlass::reference::device::Gemm<ElementT, cutlass::layout::RowMajor, ElementT, LayoutB,
        ElementD2x, cutlass::layout::RowMajor, ElementAccumulatorT, ElementAccumulatorT>;
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel
    gemm_reference(options.problem_size, ElementAccumulatorT(options.alpha), buffers.tensor_a.device_ref(),
        buffers.tensor_b.device_ref(), ElementAccumulatorT(options.beta), buffers.tensor_ref_d_2x.device_ref(),
        buffers.tensor_ref_d_2x.device_ref());

    // Wait for kernels to finish
    tk::check_cuda_error(cudaDeviceSynchronize());

    // Copy output data from reference kernel to host for comparison
    buffers.tensor_ref_d_2x.sync_host();

    // Add broadcast vector (without multiplier)
    // Vector broadcast on host
    // for (int i = 0; i < options.problem_size.m(); ++i)
    // {
    //     for (int j = 0; j < options.problem_size.n(); ++j)
    //     {
    //         buffers.tensor_ref_d_2x.host_view().ref().at({i, j}) += buffers.tensor_c_bias.host_view().ref().at({0,
    //         j});
    //     }
    // }
    cutlass::NumericConverter<ElementT, ElementComputeT, cutlass::FloatRoundStyle::round_to_nearest> converter;
    int half_n = options.problem_size.n() / 2;
    for (int i = 0; i < options.problem_size.m(); i++)
    {
        for (int j = 0; j < half_n; j++)
        {
            auto s = options.scale_output
                * ElementComputeT(options.scale_d0 * buffers.tensor_ref_d_2x.host_view().ref().at({i, j}))
                * Activation<ElementComputeT>{}(options.scale_d1 * buffers.tensor_ref_d_2x.at({i, j + half_n}));
            auto t = converter(s);
            buffers.tensor_ref_d.host_view().ref().at({i, j}) = t;
        }
    }

    tk::check_cuda_error(cudaDeviceSynchronize());

    if (options.debug)
    {
        std::cout << "tensor_ref_d_2x=" << buffers.tensor_ref_d_2x.host_view() << std::endl;
    }

    //
    // Evaluate CUTLASS kernels
    //

#ifdef COMPILE_HOPPER_TMA_GEMMS
    Result hopperFp8 = run("SM90 FP8 WS GEMM", options, buffers);
    EXPECT_TRUE(hopperFp8.passed);
#else  // COMPILE_HOPPER_TMA_GEMMS
    std::cout << "[TensorRT LLM Error][GemmSwigluRunnerTest] Please recompile with support for hopper by passing "
                 "90-real as an arch to build_wheel.py."
              << std::endl;
#endif // COMPILE_HOPPER_TMA_GEMMS
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    g_options.parse(argc, const_cast<char const**>(argv));

    return RUN_ALL_TESTS();
}
