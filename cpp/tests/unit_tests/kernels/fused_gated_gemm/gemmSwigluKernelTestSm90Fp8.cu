/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/***************************************************************************************************
 This test code is adapted from CUTLASS
 https://github.com/NVIDIA/cutlass/tree/main/examples/54_hopper_fp8_warp_specialized_gemm

 Requires NVIDIA Hopper or newer device (SM00+).
 **************************************************************************************************/

#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "fused_gated_gemm_util.h"

#include "tensorrt_llm/kernels/cutlass_kernels/fused_gated_gemm/fused_gated_gemm_kernel_template_sm90.h"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/tensor_ref.h"

#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/error_metrics.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr bool SwapAB = true;

// A matrix configuration
using ElementA = cutlass::float_e4m3_t;    // Element type for A matrix operand
using LayoutA = cutlass::layout::RowMajor; // Layout type for A matrix operand

// B matrix configuration
using ElementB = cutlass::float_e4m3_t;       // Element type for B matrix operand
using LayoutB = cutlass::layout::ColumnMajor; // Layout type for B matrix operand

// C matrix configuration
using ElementC = cutlass::float_e4m3_t;    // Element type for C and D matrix operands
using LayoutC = cutlass::layout::RowMajor; // Layout type for C and D matrix operands

// D matrix configuration
using ElementD = cutlass::float_e4m3_t;
using LayoutD = cutlass::layout::RowMajor;

// Core kernel configurations
using ElementAccumulator = float;    // Element type for internal accumulation
using ElementCompute = float;        // Element type for epilogue computation
using ArchTag = cutlass::arch::Sm90; // Tag indicating the minimum SM that supports the intended feature
using OperatorClass = cutlass::arch::OpClassTensorOp; // Operator class tag
using TileShape = Shape<_64, _16, _128>;              // Threadblock-level tile size
using ClusterShape = Shape<_8, _1, _1>;               // Shape of the threadblocks in a cluster
// using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
// using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecializedCooperative;
using MainloopScheduleType = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;

// Reference device GEMM implementation type
// always use float for ElementC here because we need float output
using DeviceGemmReference = cutlass::reference::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, float, LayoutC,
    ElementAccumulator, ElementAccumulator>;

// NOTE: debug purpose
template <typename T>
struct Passthrough
{

    CUTLASS_HOST_DEVICE
    T operator()(T const& value) const
    {
        return 1;
    }
};

struct Buffers
{
    cutlass::HostTensor<ElementA, LayoutA> tensor_a;
    cutlass::HostTensor<ElementB, LayoutB> tensor_b;
    cutlass::HostTensor<ElementC, LayoutC> tensor_c_bias;
    cutlass::HostTensor<ElementD, LayoutD> tensor_d;
    cutlass::HostTensor<ElementD, LayoutD> tensor_ref_d;
    // we need float dtype for reference GEMM output
    cutlass::HostTensor<float, LayoutD> tensor_ref_d_2x;
};

// Activation
template <typename T>
using Activation = cutlass::epilogue::thread::SiLu<T>;
// using Activation = Passthrough<T>;

// using TileSchedulerType = cutlass::gemm::StreamKScheduler;
using TileSchedulerType = void;

using Gemm = typename tensorrt_llm::kernels::cutlass_kernels::DeviceGemmGatedSm90<ElementA, ElementAccumulator,
    TileShape, ClusterShape, MainloopScheduleType, EpilogueScheduleType, TileSchedulerType, Activation, SwapAB>::Gemm;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

typename Gemm::Arguments args_from_options(Gemm const& gemm, Options const& options,
    cutlass::HostTensor<ElementA, LayoutA>& tensor_a, cutlass::HostTensor<ElementB, LayoutB>& tensor_b,
    cutlass::HostTensor<ElementD, LayoutD>& tensor_d, cutlass::HostTensor<ElementC, LayoutC>& tensor_c_bias)
{
    using ElementT = typename Gemm::ElementA;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
    if constexpr (cute::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>)
    {
        scheduler_args = {2};
    }
    if constexpr (SwapAB)
    {
        int m = options.problem_size.n() / 2;
        int n = options.problem_size.m();
        int k = options.problem_size.k();
        std::cout << "m: " << m << ", n: " << n << ", k: " << k << std::endl;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
        StrideC stride_C;
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
        printf("stride_A: ");
        cute::print(stride_A);
        printf("\nstride_B: ");
        cute::print(stride_B);
        printf("\nstride_D: ");
        cute::print(stride_D);
        printf("\n");
        typename Gemm::Arguments args = {cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
            {tensor_b.device_data(), stride_A, tensor_a.device_data(), stride_B, options.scale_d0, options.scale_d1},
            {{}, tensor_c_bias.device_data(), stride_C, tensor_d.device_data(), stride_D}};
        args.epilogue.thread.alpha = options.scale_output;
        args.scheduler = scheduler_args;
        return args;
    }
    else
    {
        int m = options.problem_size.m();
        int n = options.problem_size.n() / 2;
        int k = options.problem_size.k();
        std::cout << "m: " << m << ", n: " << n << ", k: " << k << std::endl;
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
        StrideC stride_C;
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));
        printf("stride_A: ");
        cute::print(stride_A);
        printf("\nstride_B: ");
        cute::print(stride_B);
        printf("\nstride_D: ");
        cute::print(stride_D);
        printf("\n");
        typename Gemm::Arguments args = {cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
            {tensor_a.device_data(), stride_A, tensor_b.device_data(), stride_B, options.scale_d0, options.scale_d1},
            {{}, tensor_c_bias.device_data(), stride_C, tensor_d.device_data(), stride_D}};
        args.epilogue.thread.alpha = options.scale_output;
        args.scheduler = scheduler_args;
        return args;
    }
}

/// Execute a given example GEMM computation
template <typename DeviceGemmT>
Result run(std::string description, Options& options, Buffers& buffers)
{
    // Display test description
    std::cout << std::endl << description << std::endl;

    // Zero-initialize test output matrix D
    cutlass::reference::host::TensorFill(buffers.tensor_d.host_view());
    buffers.tensor_d.sync_device();

    // Instantiate CUTLASS kernel depending on templates
    DeviceGemmT device_gemm;

    // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmT
    auto arguments = args_from_options(device_gemm, options, buffers.tensor_a, buffers.tensor_b, buffers.tensor_d,
        buffers.tensor_c_bias /*, buffers.tensor_Tensor*/);

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = DeviceGemmT::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check the problem size is supported or not
    // device_gemm.can_implement(arguments);
    auto can_implement = device_gemm.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess)
    {
        throw std::runtime_error("[TensorRT LLM Error][fusedGatedGemm Runner]");
    }

    // Initialize CUTLASS kernel with arguments and workspace pointer
    device_gemm.initialize(arguments, workspace.get());

    // Correctness / Warmup iteration
    device_gemm();

    // Copy output data from CUTLASS and reference kernel to host for comparison
    buffers.tensor_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    Result result;

    if (!options.no_check)
    {
        result.passed = cutlass::reference::host::TensorRelativelyEquals(
            buffers.tensor_d.host_view(), buffers.tensor_ref_d.host_view(), ElementD{1e-2}, ElementD{1e-2});
        result.passed
            = cutlass::reference::host::TensorEquals(buffers.tensor_d.host_view(), buffers.tensor_ref_d.host_view());
        EXPECT_TRUE(result.passed);

        double err = cutlass::reference::host::TensorRelativeErrorMetric(
            buffers.tensor_d.host_view(), buffers.tensor_ref_d.host_view());

        std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << " \t Relative error: " << err
                  << std::endl;

        if (!result.passed && options.debug)
        {
            std::cout << "ref_output=\n"
                      << buffers.tensor_ref_d.host_view() << "\noutput=\n"
                      << buffers.tensor_d.host_view() << std::endl;
        }
    }

    // Run profiling loop
    if (options.iterations > 0)
    {
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);
        for (int iter = 0; iter < options.iterations; ++iter)
        {
            device_gemm();
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
        result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

        std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
        std::cout << "  GFLOPs: " << result.gflops << std::endl;
    }

    return result;
}

/// Program entrypoint
int main(int argc, char const** argv)
{

    // Current device must must have compute capability at least 80
    cudaDeviceProp props;
    int current_device_id;
    cudaGetDevice(&current_device_id);
    cudaGetDeviceProperties(&props, current_device_id);
    if (!((props.major * 10 + props.minor) >= 90))
    {
        std::cerr << "Hopper Tensor Core operations must be run on a machine with compute capability at least 90."
                  << std::endl;

        // Returning zero so this test passes on older Toolkits. Its actions are no-op.
        exit(0);
    }

    Buffers buffers;
    // Parse commandline options
    Options options("hopper_fp8_gemm_swiglu");
    options.parse(argc, argv);

    if (options.help)
    {
        options.print_usage(std::cout) << std::endl;
        exit(0);
    }

    std::cout << options.iterations << " timing iterations of " << options.problem_size.m() << " x "
              << options.problem_size.n() << " x " << options.problem_size.k() << " matrix-matrix multiply"
              << std::endl;

    if (!options.valid())
    {
        std::cerr << "Invalid problem." << std::endl;
        EXPECT_TRUE(false);
        exit(-1);
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
        cutlass::Array<ElementA, 2> range;
        range[0] = ElementA(256);
        range[1] = ElementA(1);
        cutlass::reference::host::TensorFillLinear(buffers.tensor_a.host_view(), range);
    }
    else
    {
        cutlass::reference::host::TensorFillRandomUniform(
            buffers.tensor_a.host_view(), 1, ElementA(2), ElementA(-2), _init_bits);
    }

    // Fill matrix B on host with uniform-random data [-2, 2]
    if (options.debug)
    {
        cutlass::reference::host::TensorFillIdentity(buffers.tensor_b.host_view());
    }
    else
    {
        cutlass::reference::host::TensorFillRandomUniform(
            buffers.tensor_b.host_view(), 1, ElementB(2), ElementB(-2), _init_bits);
    }

    if (options.debug || !options.has_bias)
    {
        cutlass::reference::host::TensorFill(buffers.tensor_c_bias.host_view());
    }
    else
    {
        cutlass::reference::host::TensorFillRandomUniform(
            buffers.tensor_c_bias.host_view(), 1, ElementC(2), ElementC(-2), _init_bits);
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
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel
    gemm_reference(options.problem_size, ElementAccumulator(options.alpha), buffers.tensor_a.device_ref(),
        buffers.tensor_b.device_ref(), ElementAccumulator(options.beta), buffers.tensor_ref_d_2x.device_ref(),
        buffers.tensor_ref_d_2x.device_ref());

    // Wait for kernels to finish
    cudaDeviceSynchronize();

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
    cutlass::NumericConverter<ElementD, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest> converter;
    int half_n = options.problem_size.n() / 2;
    for (int i = 0; i < options.problem_size.m(); i++)
    {
        for (int j = 0; j < half_n; j++)
        {
            auto s = options.scale_output
                * ElementCompute(options.scale_d0 * buffers.tensor_ref_d_2x.host_view().ref().at({i, j}))
                * Activation<ElementCompute>{}(options.scale_d1 * buffers.tensor_ref_d_2x.at({i, j + half_n}));
            auto t = converter(s);
            buffers.tensor_ref_d.host_view().ref().at({i, j}) = t;
        }
    }

    cudaDeviceSynchronize();

    if (options.debug)
    {
        std::cout << "tensor_ref_d_2x=" << buffers.tensor_ref_d_2x.host_view() << std::endl;
    }

    //
    // Evaluate CUTLASS kernels
    //
#ifdef COMPILE_HOPPER_TMA_GEMMS
    Result hopperFp8 = run<Gemm>(std::string("Hopper fp8 swiglu"), options, buffers);
#else  // COMPILE_HOPPER_TMA_GEMMS
    std::cout << "[TensorRT LLM Error][GemmSwigluKernelTestSm90Fp8] Please recompile with support for hopper by "
                 "passing 90-real as an arch to build_wheel.py."
              << std::endl;
#endif // COMPILE_HOPPER_TMA_GEMMS
    // for (int i = 0; i < options.problem_size_out.m(); i++)
    // {
    //     for (int j = 0; j < options.problem_size_out.n(); j++)
    //     {
    //         std::cout << "i: " << i << ", j: " << j;
    //         std::cout << ", ref val: " << buffers.tensor_ref_d.host_view().ref().at({i, j});
    //         std::cout << ", val: " << buffers.tensor_d.host_view().ref().at({i, j}) << std::endl;
    //     }
    // }

    return 0;
}
