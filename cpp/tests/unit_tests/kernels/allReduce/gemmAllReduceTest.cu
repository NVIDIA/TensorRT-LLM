/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <nccl.h>
#include <vector>

#include "common.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/allreduce_gemm/allreduce_gemm_runner.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include <NvInferRuntime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_fill.h"

using namespace cutlass;
using namespace nvinfer1;
using namespace tensorrt_llm::mpi;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::kernels::cutlass_kernels;

///////////////////////////
// CLI Args
///////////////////////////
struct Options
{
    bool help = false;
    int seed = 0;

    // problem shape
    int M = 13;
    int N = 4096;
    int K = 8192;
    int K_tp; // K per GPU
    int rank = 0;
    int tp = 1;
    std::set<int> tp_group;
    float alpha = 1.0f;
    float beta = 0.0f;

    int iterations = 100;

    bool valid() const
    {
        return M > 0 && N >= 16 && K_tp >= 16 && tp > 0 && iterations > 0;
    }

    // Parses the command line
    void parse(int argc, char** args)
    {
        cutlass::CommandLine cmd(argc, const_cast<char const**>(args));

        if (cmd.check_cmd_line_flag("help"))
        {
            help = true;
        }

        cmd.get_cmd_line_argument("m", M);
        cmd.get_cmd_line_argument("n", N);
        cmd.get_cmd_line_argument("k", K);

        cmd.get_cmd_line_argument("alpha", alpha);
        cmd.get_cmd_line_argument("beta", beta);
        cmd.get_cmd_line_argument("iterations", iterations);

        rank = COMM_SESSION.getRank();
        tp = COMM_SESSION.getSize();
        for (int rank = 0; rank < tp; ++rank)
        {
            tp_group.insert(rank);
        }
        assert(K % tp == 0);
        K_tp = K / tp;

        // Should be same across all ranks
        srand(time(NULL));
        seed = static_cast<int>(rand());
        COMM_SESSION.bcastValue(seed, 0);

#if 1
        printf("rank: %d, m: %d, n: %d, k: %d, tp: %d, seed: %d\n", rank, M, N, K, tp, seed);
#endif
    }

    /// Prints the usage statement.
    std::ostream& print_usage(std::ostream& out) const
    {
        out << "\n"
               "Options:\n"
               "  --help                      If specified, displays this usage statement.\n"
               "  --m=<int>                   GEMM M dimension (LLM batch size)\n"
               "  --n=<int>                   GEMM N dimension (needs to be >= 16)\n"
               "  --k=<int>                   GEMM K dimension (needs to be >= 16 * nranks)\n"
               "  --alpha=<float>             GEMM alpha parameter\n"
               "  --beta=<float>              GEMM beta parameter\n"
               "  --iterations=<int>          Number of profiling iterations to perform.\n"
               "\n"
               "Examples:\n"
               "\n"
               "$ mpirun -np 8 ./test/gemmAllReduceTest --m=8192 --n=8192 --k=8192 --iterations=1000\n";

        return out;
    }

    /// Compute performance in GFLOP/s
    double gflops(double runtime_s) const
    {
        // Two flops per multiply-add
        uint64_t flop = uint64_t(2) * M * N * K_tp;
        double gflop = double(flop) / double(1.0e9);
        return gflop / runtime_s;
    }

    double effective_bandwidth(double runtime_s, size_t bytes_a, size_t bytes_b, size_t bytes_c, size_t bytes_d) const
    {
        static double const kBytesPerGiB = double(1ull << 30);

        double bytes_in = (double) (M) * (double) (K_tp) * (double) (bytes_a) +     // A
            (double) (N) * (double) (K_tp) * (double) (bytes_b) +                   // B
            (beta != 0.f ? (double) (M) * (double) (N) * (double) (bytes_c) : 0.f); // C
        double bytes_out = (double) (M) * (double) (N) * (double) (bytes_d);        // D

        double gb_total = (bytes_in + bytes_out) / kBytesPerGiB;
        return gb_total / runtime_s;
    }

    double effective_allreduce_bandwidth(double runtime_s, size_t bytes_d)
    {
        static double const kBytesPerGiB = double(1ull << 30);
        double bytes = (double) (M) * (double) (N) * (double) (bytes_d);
        double gb_total = bytes / kBytesPerGiB;
        return gb_total / runtime_s;
    }
};

struct Result
{
    double avg_runtime_us;
    double gflops;
    double eff_bw;
    double eff_AR_bw;
    bool passed;

    Result(double avg_runtime_us = 0, double gflops = 0, double eff_bw = 0, double eff_AR_bw = 0)
        : avg_runtime_us(avg_runtime_us)
        , gflops(gflops)
        , eff_bw(eff_bw)
        , eff_AR_bw(eff_AR_bw)
        , passed(false)
    {
    }
};

///////////////////////////
// NCCL types
///////////////////////////
template <typename CutlassType>
struct ToType
{
};

template <>
struct ToType<cutlass::bfloat16_t>
{
    ncclDataType_t nccl_value = ncclBfloat16;
    char const* str_value = "bf16";
};

template <>
struct ToType<cutlass::half_t>
{
    ncclDataType_t nccl_value = ncclFloat16;
    char const* str_value = "fp16";
};

template <>
struct ToType<cutlass::float_e4m3_t>
{
    char const* str_value = "fp8_e4m3";
};

class NcclCommunicator
{
public:
    static NcclCommunicator& instance()
    {
        static NcclCommunicator communicator;
        return communicator;
    }

    ncclComm_t comm;

private:
    NcclCommunicator()
    {
        int rank = COMM_SESSION.getRank();
        int world_size = COMM_SESSION.getSize();

        ncclUniqueId id;
        if (rank == 0)
            ncclGetUniqueId(&id);
        COMM_SESSION.bcastValue(id, 0);

        TLLM_NCCL_CHECK(ncclCommInitRank(&comm, world_size, id, rank));
    }
};

/////////////////////////////////////
// Gemm+AR Functional test fixture
/////////////////////////////////////
static Options options;

template <typename _ElementA, typename _ElementB, typename _ElementC, typename _ElementD>
struct TestConfig
{
    using ElementA = _ElementA;
    using ElementB = _ElementB;
    using ElementC = _ElementC;
    using ElementD = _ElementD;
};

template <typename T>
class GemmAllReduceFixture : public ::testing::Test
{
protected:
    using ElementA = typename T::ElementA;
    using ElementB = typename T::ElementB;
    using ElementC = typename T::ElementC;
    using ElementD = typename T::ElementD;

    static bool isMultiGpu()
    {
        return COMM_SESSION.getSize() > 1;
    }

    void SetUp() override
    {
        using GemmTraits = GemmTypes<ElementA, ElementB, ElementC, ElementD, cutlass::layout::RowMajor,
            cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>;

        _gemm = std::make_shared<GemmAllReduceImplRunner<GemmTraits>>();

        auto const M = options.M;
        auto const N = options.N;
        auto const K_tp = options.K_tp;

        _A.reset(M * K_tp);
        _B.reset(N * K_tp);
        _C.reset(M * N);
        if (isMultiGpu())
        {
            _D_nvls.reset(M * N, options.tp_group);
            // Create workspace for max problem size
            GemmAllReduceImplInterface::LaunchConfig launch_config = {GemmAllReduceImpl::NVLS_2SHOT,
                MainloopScheduleType::PINGPONG, TileShape::TileShape_128x16x128, ClusterShape::ClusterShape_1x1x1};
            GemmAllReduceImplInterface::ProblemArgs max_problem;
            max_problem.argProblemShape(M, N, K_tp, 1)
                .argRanks(options.rank, options.tp_group)
                .argLaunchConfig(launch_config);
            _workspace = _gemm->getPersistentWorkspace(max_problem);
            _workspace->allocate();
        }
        else
        {
            _D.reset(M * N);
        }
        _D_ref.reset(M * N);

        initialize_block(_A, options.seed + options.rank + 2024);
        initialize_block(_B, options.seed + options.rank);
        initialize_block(_C, options.seed);
    }

    void TearDown() override
    {
        if (isMultiGpu())
        {
            _workspace->free();
            _D_nvls.free();
        }
    }

    void run(cudaStream_t stream = NULL)
    {
        // Test
        GemmAllReduceImplInterface::ProblemArgs args;
        args.argProblemShape(options.M, options.N, options.K_tp, 1)
            .argA(_A.get())
            .argB(_B.get())
            .argC(_C.get())
            .argAlpha(options.alpha)
            .argBeta(options.beta)
            .argRanks(options.rank, options.tp_group);

        if (isMultiGpu())
        {
            args.argD(_D_nvls.getUnicastPointer(), _D_nvls.getMulticastPointer()).argWorkspace(_workspace.get());
        }
        else
        {
            args.argD(_D.get());
        }

        Result result;
        result.passed = true;

        // Ensure all configs pass
        auto launch_configs = _gemm->getSupportedLaunchConfigs();
        for (auto launch_config : launch_configs)
        {
            args.argLaunchConfig(launch_config);

            _gemm->run(args, stream);
            TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

            bool passed = verify(stream);
            if (!passed)
            {
                std::cout << "config failed: " << launch_config.str() << std::endl;
            }
            result.passed &= passed;
        }

        // Benchmark
        tensorrt_llm::testing::GpuTimer timer;
        int const warmup = 20;

        float best_elapsed_us = std::numeric_limits<float>::max();
        GemmAllReduceImplInterface::LaunchConfig best_launch_config;

        for (auto launch_config : launch_configs)
        {
            args.argLaunchConfig(launch_config);

            for (int i = 0; i < options.iterations + warmup; ++i)
            {
                if (i == warmup)
                {
                    // Synchronize ranks
                    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
                    COMM_SESSION.barrier();
                    timer.start(stream);
                }
                _gemm->run(args, stream);
            }
            timer.stop();
            float elapsed_us = timer.elapsed_millis() * 1000.f;
            if (options.rank == 0)
            {
                double avg_runtime_us = double(elapsed_us) / double(options.iterations);
                std::cout << launch_config.str() << std::endl;
                std::cout << "  Avg runtime: " << avg_runtime_us << " us" << std::endl;
            }
            if (elapsed_us < best_elapsed_us)
            {
                best_elapsed_us = elapsed_us;
                best_launch_config = launch_config;
            }
        }

        result.avg_runtime_us = double(best_elapsed_us) / double(options.iterations);
        double avg_runtime_s = (double) (result.avg_runtime_us / 1000000.0);
        result.gflops = options.gflops(avg_runtime_s);
        result.eff_bw = options.effective_bandwidth(
            avg_runtime_s, sizeof(ElementA), sizeof(ElementB), sizeof(ElementC), sizeof(ElementD));
        result.eff_AR_bw = options.effective_allreduce_bandwidth(avg_runtime_s, sizeof(ElementD));

        if (options.rank == 0)
        {
            std::cout << std::endl;
            std::cout << "  Precision: " << ToType<ElementA>{}.str_value << "x" << ToType<ElementB>{}.str_value << "="
                      << ToType<ElementD>{}.str_value << std::endl;
            std::cout << "  Problem Size: " << options.M << 'x' << options.N << 'x' << options.K << std::endl;
            std::cout << "  Local Problem Size: " << options.M << 'x' << options.N << 'x' << options.K_tp << std::endl;
            std::cout << "  " << best_launch_config.str() << std::endl;
            std::cout << "  Verify: " << (result.passed ? "Pass" : "Fail") << std::endl;
            std::cout << "  Avg runtime: " << result.avg_runtime_us << " us" << std::endl;
            std::cout << "  GFLOPS: " << result.gflops << std::endl;
            std::cout << "  Effective GEMM bandwidth: " << result.eff_bw << " GB/s" << std::endl;
            std::cout << "  Effective AR bandwidth: " << result.eff_AR_bw << " GB/s" << std::endl;
        }
        ASSERT_TRUE(result.passed);
    }

private:
    template <typename ElementT>
    void print_tensor(std::string name, ElementT* data, int const H, int const W)
    {
        std::vector<ElementT> host(H * W);
        cutlass::device_memory::copy_to_host(host.data(), data, H * W);
        auto host_tensor = cute::make_tensor(host.data(), cute::make_shape(H, W), cute::make_stride(W, _1{}));
        cute::print_tensor(host_tensor);
    }

    template <typename ElementT>
    auto find_relative_differences(ElementT const* d_ptr_A, ElementT const* d_ptr_B, size_t capacity, ElementT epsilon,
        ElementT nonzero_floor, size_t max_count = 5)
    {
        std::vector<ElementT> h_ptr_A(capacity);
        std::vector<ElementT> h_ptr_B(capacity);
        cutlass::device_memory::copy_to_host(h_ptr_A.data(), d_ptr_A, capacity);
        cutlass::device_memory::copy_to_host(h_ptr_B.data(), d_ptr_B, capacity);
        std::vector<std::tuple<ElementT, ElementT, size_t>> differences;
        for (size_t i = 0; i < capacity; ++i)
        {
            auto a = h_ptr_A[i];
            auto b = h_ptr_B[i];
            if (!cutlass::relatively_equal(a, b, epsilon, nonzero_floor))
            {
                differences.push_back(std::make_tuple(a, b, i));
                if (differences.size() >= max_count)
                {
                    break;
                }
            }
        }
        return differences;
    }

    bool verify(cudaStream_t stream)
    {
        using LayoutA = cutlass::layout::RowMajor;
        using LayoutB = cutlass::layout::ColumnMajor;
        using LayoutC = cutlass::layout::RowMajor;
        using LayoutD = cutlass::layout::RowMajor;
        using ElementScalar = float;
        using ElementAccumulator = float;

        auto const M = options.M;
        auto const N = options.N;
        auto const K = options.K_tp;

        cutlass::TensorRef ref_A(_A.get(), LayoutA::packed({M, K}));
        cutlass::TensorRef ref_B(_B.get(), LayoutB::packed({K, N}));
        cutlass::TensorRef ref_C(_C.get(), LayoutC::packed({M, N}));
        cutlass::TensorRef ref_D(_D_ref.get(), LayoutD::packed({M, N}));

        // Reference device GEMM implementation type
        using DeviceGemmReference = cutlass::reference::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC,
            LayoutC, ElementAccumulator, ElementAccumulator>;

        // Create instantiation for device reference gemm kernel
        DeviceGemmReference gemm_reference;

        // Launch device reference gemm kernel
        gemm_reference(
            {M, N, K}, ElementAccumulator(options.alpha), ref_A, ref_B, ElementAccumulator(options.beta), ref_C, ref_D);

        TLLM_CUDA_CHECK(cudaDeviceSynchronize());

        // AllReduce across ranks
        ncclComm_t comm = NcclCommunicator::instance().comm;
        auto dtype = ToType<ElementD>{}.nccl_value;
        TLLM_NCCL_CHECK(ncclAllReduce(_D_ref.get(), _D_ref.get(), _D_ref.size(), dtype, ncclSum, comm, stream));
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Compare results
        const ElementC epsilon(1e-2f);
        const ElementC nonzero_floor(1e-4f);

        auto D_ptr = isMultiGpu() ? _D_nvls.getUnicastPointer() : _D.get();

        int local_failed = 0;
        if (!cutlass::reference::device::BlockCompareRelativelyEqual(
                _D_ref.get(), D_ptr, _D_ref.size(), epsilon, nonzero_floor))
        {
            if (options.rank == 0)
            {
#if 1
                auto differences
                    = find_relative_differences(_D_ref.get(), D_ptr, _D_ref.size(), epsilon, nonzero_floor);

                std::cerr << "Differences:" << std::endl;
                for (auto [exp, act, pos] : differences)
                {
                    std::cerr << "expected: " << std::setprecision(3) << std::setw(5) << exp
                              << ", actual: " << std::setprecision(3) << std::setw(5) << act << ", at pos: " << pos
                              << std::endl;
                }
#endif
#if 0
            print_tensor("Actual", D_ptr, M, N);
            print_tensor("Ref   ", _D_ref.get(), M, N);
#endif
            }
            local_failed = 1;
        }

        // Aggregate results - if 1 rank fails, then all ranks fail.
        int global_failed;
        COMM_SESSION.allreduce(&local_failed, &global_failed, 1, MpiType::kINT32, MpiOp::SUM);

        return global_failed == 0;
    }

    template <class Element>
    static void initialize_block(cutlass::DeviceAllocation<Element>& block, int seed)
    {
        double scope_max, scope_min;
        int bits_input = cutlass::sizeof_bits<Element>::value;
        int bits_output = cutlass::sizeof_bits<Element>::value;

        if (bits_input == 1)
        {
            scope_max = 2;
            scope_min = 0;
        }
        else if (bits_input <= 8)
        {
            scope_max = 2;
            scope_min = -2;
        }
        else if (bits_output == 16)
        {
            scope_max = 5;
            scope_min = -5;
        }
        else
        {
            scope_max = 8;
            scope_min = -8;
        }

        using Real = typename cutlass::RealType<Element>::Type;
        cutlass::reference::device::BlockFillRandomUniform(
            block.get(), block.size(), seed, static_cast<Real>(scope_max), static_cast<Real>(scope_min), 0);
    }

    cutlass::DeviceAllocation<ElementA> _A;
    cutlass::DeviceAllocation<ElementB> _B;
    cutlass::DeviceAllocation<ElementC> _C;
    cutlass::DeviceAllocation<ElementD> _D;
    cutlass::DeviceAllocation<ElementD> _D_ref;
    DeviceAllocationNvls<ElementD> _D_nvls;
    std::shared_ptr<PersistentWorkspaceInterface> _workspace;
    std::shared_ptr<GemmAllReduceImplInterface> _gemm;
};

using MyTypes = testing::Types<TestConfig<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t>,
    TestConfig<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t, cutlass::half_t>>;

TYPED_TEST_SUITE(GemmAllReduceFixture, MyTypes);

/////////////////////////////////////////////////////////////////////
// ATTENTION: run test with mpi `mpi -np <NP> ./gemmAllReduceTest'
/////////////////////////////////////////////////////////////////////
TYPED_TEST(GemmAllReduceFixture, RunnerTest)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    this->run(stream);
    cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    bool notSupported = false;
    // CUDA 12 minimum required
    if (__CUDACC_VER_MAJOR__ < 12)
    {
        std::cerr << "This example requires CUDA Toolkit version 12 or later.\n";
        notSupported = true;
    }

    TLLM_CUDA_CHECK(cudaSetDevice(COMM_SESSION.getRank()));

    cudaDeviceProp props;
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&props, COMM_SESSION.getRank()));

    if (props.major < 9)
    {
        std::cerr << "This example requires a device with compute capability 90 or higher.\n";
        notSupported = true;
    }

    if (!ipcNvlsSupported())
    {
        std::cerr << "NVLS not supported on this system.\n";
        notSupported = true;
    }

    if (notSupported)
    {
        return EXIT_SUCCESS; // Do not fail CI checks on unsupported systems
    }

    options.parse(argc, argv);

    if (options.help)
    {
        options.print_usage(std::cout) << "\n";
        return EXIT_SUCCESS;
    }

    if (!options.valid())
    {
        std::cerr << "Invalid arguments."
                  << "\n";
        return EXIT_FAILURE;
    }

    return RUN_ALL_TESTS();
}
