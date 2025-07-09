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

#if defined(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
#include "tensorrt_llm/kernels/cutlass_kernels/include/allreduce_gemm_runner.h"
#else
#include "allreduce_gemm_runner.h"
#endif
#include "common.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/userbuffers/ub_interface.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
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
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"

using namespace cutlass;
using namespace nvinfer1;
using namespace tensorrt_llm::mpi;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime::ub;
using namespace tensorrt_llm::kernels::ub;
#if defined(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
namespace cutlass_kernels = ::tensorrt_llm::kernels::opened_cutlass_kernels;
#else
namespace cutlass_kernels = ::tensorrt_llm::kernels::cutlass_kernels;
#endif
///////////////////////////
// CLI Args
///////////////////////////
struct Options
{
    bool help = false;
    bool verify = true;
    bool use_UB = false;
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

        if (cmd.check_cmd_line_flag("skip_check"))
        {
            verify = false;
        }

        if (cmd.check_cmd_line_flag("userbuffers"))
        {
            use_UB = true;
        }

        cmd.get_cmd_line_argument("m", M);
        cmd.get_cmd_line_argument("n", N);
        cmd.get_cmd_line_argument("k", K);

        cmd.get_cmd_line_argument("alpha", alpha);
        cmd.get_cmd_line_argument("beta", beta);
        cmd.get_cmd_line_argument("iterations", iterations);

        rank = COMM_SESSION.getRank();
        tp = COMM_SESSION.getSize();
        for (int i = 0; i < tp; ++i)
        {
            tp_group.insert(i);
        }
        assert(K % tp == 0);
        K_tp = K / tp;

        // Should be same across all ranks
        srand(time(NULL));
        seed = static_cast<int>(rand());
        COMM_SESSION.bcastValue(seed, 0);

#if 1
        printf("rank: %d, m: %d, n: %d, k: %d, tp: %d, seed: %d\n", this->rank, M, N, K, tp, seed);
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
               "  --skip_check                Skips verification (verification is slow for large shapes)\n"
               "  --userbuffers               Uses UserBuffers for AR reference benchmarking.\n"
               "\n"
               "Examples:\n"
               "\n"
               "$ mpirun -np 8 ./test/gemmAllReduceTest --m=8192 --n=8192 --k=8192 --iterations=1000\n";

        return out;
    }

    /// Compute performance in GFLOP/s
    double tflops(double runtime_s) const
    {
        // Two flops per multiply-add
        uint64_t flop = uint64_t(2) * M * N * K_tp;
        double gflop = double(flop) / double(1.0e9);
        double tflops = gflop / 1e3;
        return tflops / runtime_s;
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

    double overlap_efficiency(double gemm_runtime, double AR_runtime, double gemm_AR_runtime)
    {
        double effective_gemm_time_fused = gemm_AR_runtime - AR_runtime;
        double effective_comm_time_fused = gemm_AR_runtime - gemm_runtime;

        double overlap_gemm_efficiency = 1 - effective_gemm_time_fused / gemm_runtime;
        double overlap_comm_efficiency = 1 - effective_comm_time_fused / AR_runtime;

        return max(overlap_gemm_efficiency, overlap_comm_efficiency);
    }
};

struct Result
{
    double avg_runtime_us;
    double avg_runtime_AR_us;
    double tflops;
    double eff_bw;
    double eff_AR_bw;
    bool passed;
    cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig best_config;

    Result(double avg_runtime_us = 0, double avg_runtime_AR_us = 0, double tflops = 0, double eff_bw = 0,
        double eff_AR_bw = 0)
        : avg_runtime_us(avg_runtime_us)
        , avg_runtime_AR_us(avg_runtime_AR_us)
        , tflops(tflops)
        , eff_bw(eff_bw)
        , eff_AR_bw(eff_AR_bw)
        , passed(false)
    {
    }
};

///////////////////////////
// CUTLASS type converter
///////////////////////////
template <typename CutlassType>
struct ToType
{
};

template <>
struct ToType<cutlass::bfloat16_t>
{
    nvinfer1::DataType trt_value = nvinfer1::DataType::kBF16;
    ncclDataType_t nccl_value = ncclBfloat16;
    char const* str_value = "bf16";
};

template <>
struct ToType<cutlass::half_t>
{
    nvinfer1::DataType trt_value = nvinfer1::DataType::kHALF;
    ncclDataType_t nccl_value = ncclFloat16;
    char const* str_value = "fp16";
};

template <>
struct ToType<cutlass::float_e4m3_t>
{
    nvinfer1::DataType trt_value = nvinfer1::DataType::kFP8;
    ncclDataType_t nccl_value = ncclFloat8e4m3;
    char const* str_value = "fp8_e4m3";
};

template <>
struct ToType<cutlass::float_e2m1_t>
{
    char const* str_value = "fp4_e2m1";
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

// Required for subbyte reference GEMM (i.e FP4)
template <typename T>
auto make_iterator(T* ptr)
{
    using namespace cute;
    if constexpr (cute::is_subbyte_v<T>)
    {
        return subbyte_iterator<T>(ptr);
    }
    else
    {
        return ptr;
    }
}

/////////////////////////////////////
// Gemm+AR Functional test fixture
/////////////////////////////////////
static Options options;

template <typename _ElementA, typename _ElementB, typename _ElementC, typename _ElementD, typename _ElementSFA = void,
    typename _ElementSFB = void>
struct TestConfig
{
    using ElementA = _ElementA;
    using ElementB = _ElementB;
    using ElementC = _ElementC;
    using ElementD = _ElementD;
    using ElementSFB = _ElementSFA;
    using ElementSFA = _ElementSFB;
};

template <typename T>
class GemmAllReduceFixture : public ::testing::Test
{
protected:
    using ElementA = typename T::ElementA;
    using ElementB = typename T::ElementB;
    using ElementC = typename T::ElementC;
    using ElementD = typename T::ElementD;
    using ElementSFA = typename T::ElementSFA;
    using ElementSFB = typename T::ElementSFB;
    static_assert(std::is_same_v<ElementA, ElementB> && "A & B types must be same");

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
    using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
    using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
    using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

    // Only currently supported for FP4 GEMM
    static constexpr bool IsInputScalingNeeded = std::is_same_v<ElementSFA, cutlass::float_ue4m3_t>;
    static constexpr bool IsFP4 = std::is_same_v<ElementA, cutlass::float_e2m1_t>;

    static bool isMultiGpu()
    {
        return COMM_SESSION.getSize() > 1;
    }

    static void SetUpTestSuite()
    {
        // Hopper skip FP4 GEMMs
        if (getSMVersion() < 100 && IsFP4)
        {
            GTEST_SKIP() << "Skipping FP4 GEMM";
        }
        // Allocate UB
        ub_initialize(COMM_SESSION.getSize());
        if (!ub_is_initialized())
        {
            options.use_UB = false;
        }
        if (options.use_UB)
        {
            void* p0 = ub_allocate(options.M * options.N * sizeof(ElementD)).addr;
            ASSERT_NE(p0, nullptr);
        }
    }

    void SetUp() override
    {
        using GemmTraits = cutlass_kernels::GemmTypes<ElementA, ElementB, ElementC, ElementD, ElementSFA, ElementSFB,
            LayoutA, LayoutB, LayoutC, LayoutD>;

        _gemm = std::make_shared<cutlass_kernels::GemmAllReduceImplRunner<GemmTraits>>();

        auto const M = options.M;
        auto const N = options.N;
        auto const K_tp = options.K_tp;

        _A.reset(cutlass::make_Coord(M * K_tp));
        _B.reset(cutlass::make_Coord(N * K_tp));
        _C.reset(cutlass::make_Coord(M * N));
        if (isMultiGpu())
        {
            _D_nvls.reset(M * N, options.tp_group);
            // Create workspace for max problem size
            cutlass_kernels::GemmAllReduceImplInterface::ProblemArgs max_problem;
            cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig launch_config
                = _gemm->getSupportedLaunchConfigs()[0];
            max_problem.argProblemShape(M, N, K_tp, 1)
                .argRanks(options.rank, options.tp_group)
                .argLaunchConfig(launch_config);
            // max_problem.argProblemShape(M, N, K_tp, 1).argRanks(options.rank, options.tp_group);
            _workspace = _gemm->getPersistentWorkspace(max_problem);
            _workspace->allocate();
        }
        else
        {
            _D.reset(cutlass::make_Coord(M * N));
        }
        _D_ref.reset(cutlass::make_Coord(M * N));
        _alpha_vec.resize(cutlass::make_Coord(1));

        if constexpr (IsInputScalingNeeded)
        {
            auto [layout_SFA, layout_SFB] = getLayoutSF_AB(M, N, K_tp);
            auto size_SFA = size(filter_zeros(layout_SFA));
            auto size_SFB = size(filter_zeros(layout_SFB));
            _SFA.reset(cutlass::make_Coord(size_SFA));
            _SFB.reset(cutlass::make_Coord(size_SFB));
        }

        initializeTensor(_A.host_view(), options.seed + options.rank + 2024);
        initializeTensor(_B.host_view(), options.seed + options.rank);
        initializeTensor(_C.host_view(), options.seed);
        _A.sync_device();
        _B.sync_device();
        _C.sync_device();

        if constexpr (IsInputScalingNeeded)
        {
            initializeTensor(_SFA.host_view(), options.seed + options.rank + 2023);
            initializeTensor(_SFB.host_view(), options.seed + options.rank + 2022);
            _SFA.sync_device();
            _SFB.sync_device();
        }

        _alpha_vec.host_data()[0] = options.alpha;
        _alpha_vec.sync_device();
    }

    void TearDown() override
    {
        if (isMultiGpu())
        {
            _workspace->free();
            _D_nvls.free();
        }
    }

    /*
     * Benchmarks each config.
     * Benchmarks no fusion for comparison.
     */
    void bench(cudaStream_t stream)
    {
        cutlass_kernels::GemmAllReduceImplInterface::ProblemArgs args = get_arguments();
        int const warmup = 20;

        auto sweep_configs = [&]()
        {
            Result result;
            tensorrt_llm::testing::GpuTimer timer;
            float best_elapsed_us = std::numeric_limits<float>::max();
            cutlass_kernels::GemmAllReduceImplInterface::LaunchConfig best_launch_config;

            auto launch_configs = _gemm->getSupportedLaunchConfigs();
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

            result.best_config = best_launch_config;
            result.avg_runtime_us = double(best_elapsed_us) / double(options.iterations);
            double avg_runtime_s = (double) (result.avg_runtime_us / 1000000.0);
            result.tflops = options.tflops(avg_runtime_s);
            result.eff_bw = options.effective_bandwidth(
                avg_runtime_s, sizeof(ElementA), sizeof(ElementB), sizeof(ElementC), sizeof(ElementD));
            result.eff_AR_bw = options.effective_allreduce_bandwidth(avg_runtime_s, sizeof(ElementD));

            return result;
        };

        // Benchmark each config.
        auto result = sweep_configs();

        // Let clocks spin up again for fair benchmark.
        sleep(3);

        // set to single device
        args.argRanks(0, {0});
        // Benchmark GEMM with no fusion.
        auto result_no_fusion = sweep_configs();
        result_no_fusion.eff_AR_bw = 0;
        result_no_fusion.avg_runtime_AR_us = 0;

        // Benchmark AR with no fusion
        if (isMultiGpu())
        {
            tensorrt_llm::testing::GpuTimer timer;
            for (int i = 0; i < options.iterations + warmup; ++i)
            {
                if (i == warmup)
                {
                    // Synchronize ranks
                    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
                    COMM_SESSION.barrier();
                    timer.start(stream);
                }

                if (options.use_UB)
                {
                    auto comm = ub_comm();
                    auto dtype = ToType<ElementD>{}.trt_value;
                    auto ub_buf = ub_get(0);
                    EXPECT_TRUE(not ub_buf.invalid());
                    allreduce2_userbuff_inplace_launcher(ub_buf.handle, 0, _D_ref.size(), dtype, comm, stream);
                }
                else
                {
                    ncclComm_t comm = NcclCommunicator::instance().comm;
                    auto dtype = ToType<ElementD>{}.nccl_value;
                    TLLM_NCCL_CHECK(ncclAllReduce(
                        _D_ref.device_data(), _D_ref.device_data(), _D_ref.size(), dtype, ncclSum, comm, stream));
                }
            }
            timer.stop();
            float elapsed_us = timer.elapsed_millis() * 1000.f;
            result_no_fusion.avg_runtime_AR_us = double(elapsed_us) / double(options.iterations);
            double avg_runtime_AR_s = (double) (result_no_fusion.avg_runtime_AR_us / 1000000.0);
            result_no_fusion.eff_AR_bw = options.effective_allreduce_bandwidth(avg_runtime_AR_s, sizeof(ElementD));
        }

        if (options.rank == 0)
        {
            std::cout << std::endl;
            std::cout << "  Precision: " << ToType<ElementA>{}.str_value << "x" << ToType<ElementB>{}.str_value << "="
                      << ToType<ElementD>{}.str_value << std::endl;
            std::cout << "  Problem Size: " << options.M << 'x' << options.N << 'x' << options.K << std::endl;
            std::cout << "  Local Problem Size: " << options.M << 'x' << options.N << 'x' << options.K_tp << std::endl;
            std::cout << "\n  GEMM->AR\n" << std::endl;
            std::cout << "  " << result.best_config.str() << std::endl;
            std::cout << "  GEMM runtime: " << result_no_fusion.avg_runtime_us << " us" << std::endl;
            std::cout << "  GEMM TFLOPS: " << result_no_fusion.tflops << std::endl;
            std::cout << "  GEMM effective bandwidth: " << result_no_fusion.eff_bw << " GB/s" << std::endl;
            std::cout << "  AR runtime: " << result_no_fusion.avg_runtime_AR_us << " us" << std::endl;
            std::cout << "  AR algo bandwidth: " << result_no_fusion.eff_AR_bw << " GB/s" << std::endl;
            std::cout << "\n  GEMM+AR fusion\n" << std::endl;
            std::cout << "  " << result.best_config.str() << std::endl;
            std::cout << "  GEMM runtime: " << result.avg_runtime_us << " us" << std::endl;
            std::cout << "  GEMM TFLOPS: " << result.tflops << std::endl;
            std::cout << "  GEMM effective bandwidth: " << result.eff_bw << " GB/s" << std::endl;
            std::cout << "  AR algo bandwidth: " << result.eff_AR_bw << " GB/s" << std::endl;
            float speedup
                = (result_no_fusion.avg_runtime_us + result_no_fusion.avg_runtime_AR_us) / result.avg_runtime_us;
            std::cout << "\n  Speedup: " << speedup << std::endl;
            double overlap_efficiency = options.overlap_efficiency(
                result_no_fusion.avg_runtime_us, result_no_fusion.avg_runtime_AR_us, result.avg_runtime_us);
            std::cout << "  Overlap efficiency: " << overlap_efficiency << std::endl;
            std::cout << std::endl;
        }
    }

    /**
     * Run each config to ensure each one passes numerical check.
     */
    void run(cudaStream_t stream = NULL)
    {
        cutlass_kernels::GemmAllReduceImplInterface::ProblemArgs args = get_arguments();

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
            std::cout << launch_config.str() << std::endl;
            std::cout << "  Verify: " << (passed ? "Pass" : "Fail") << std::endl;
            result.passed &= passed;
        }

        ASSERT_TRUE(result.passed);
    }

private:
    cutlass_kernels::GemmAllReduceImplInterface::ProblemArgs get_arguments()
    {
        cutlass_kernels::GemmAllReduceImplInterface::ProblemArgs args;
        args.argProblemShape(options.M, options.N, options.K_tp, 1)
            .argA(_A.device_data())
            .argB(_B.device_data())
            .argC(_C.device_data())
            .argAlphaPtr(_alpha_vec.device_data())
            .argBeta(options.beta)
            .argRanks(options.rank, options.tp_group)
            .argWorkspace(_workspace.get());

        if constexpr (IsInputScalingNeeded)
        {
            args.argAScale(_SFA.device_data());
            args.argBScale(_SFB.device_data());
        }

        if (isMultiGpu())
        {
            args.argD(
                _D_nvls.getUnicastPointer(), _D_nvls.getMulticastPointer(), (void**) _D_nvls.getIpcUnicastPointers());
        }
        else
        {
            args.argD(_D.device_data());
        }

        return args;
    }

    template <typename ElementT>
    void print_tensor(std::string name, ElementT* data, int const H, int const W)
    {
        std::vector<ElementT> host(H * W);
        cutlass::device_memory::copy_to_host(host.data(), data, H * W);
        auto host_tensor = cute::make_tensor(host.data(), cute::make_shape(H, W), cute::make_stride(W, _1{}));
        cute::print_tensor(host_tensor);
    }

    template <typename ElementT>
    auto findRelativeDifferences(ElementT const* d_ptr_A, ElementT const* d_ptr_B, size_t capacity, ElementT epsilon,
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

    template <typename ElementT>
    bool compareRelativelyEqual(ElementT* expected, ElementT* actual, size_t size, float epsilon, float nonzero_floor)
    {
        ElementT eps = static_cast<ElementT>(epsilon);
        ElementT floor = static_cast<ElementT>(nonzero_floor);
        if (!cutlass::reference::device::BlockCompareRelativelyEqual(expected, actual, size, eps, floor))
        {
            if (options.rank == 0)
            {
#if 1
                auto differences = findRelativeDifferences(expected, actual, size, eps, floor);

                std::cerr << "Differences:" << std::endl;
                for (auto [exp, act, pos] : differences)
                {
                    std::cerr << "expected: " << std::setprecision(3) << std::setw(5) << float(exp)
                              << ", actual: " << std::setprecision(3) << std::setw(5) << float(act)
                              << ", at pos: " << pos << std::endl;
                }
#endif
#if 0
            // print_tensor("Actual", D_actual, M, N);
            // print_tensor("Ref   ", D_expect, M, N);
#endif
            }
            return false;
        }
        return true;
    }

    bool verify(cudaStream_t stream)
    {
        auto const M = options.M;
        auto const N = options.N;
        auto const K = options.K_tp;

        // Prepare arguments for reference GEMM
        auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
        auto layout_A = cute::make_layout(cute::make_shape(M, K, 1), stride_A);
        auto tensor_A = cute::make_tensor(make_iterator(_A.host_data()), layout_A);

        auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
        auto layout_B = cute::make_layout(cute::make_shape(N, K, 1), stride_B);
        auto tensor_B = cute::make_tensor(make_iterator(_B.host_data()), layout_B);

        auto get_mainloop_params = [&]()
        {
            if constexpr (IsInputScalingNeeded)
            {
                auto [layout_SFA, layout_SFB] = getLayoutSF_AB(M, N, K);
                auto tensor_SFA = cute::make_tensor(_SFA.host_data(), layout_SFA);
                auto tensor_SFB = cute::make_tensor(_SFB.host_data(), layout_SFB);

                return cutlass::reference::host::GettMainloopParams<float, decltype(tensor_A), decltype(tensor_B),
                    decltype(tensor_SFA), decltype(tensor_SFB)>{tensor_A, tensor_SFA, tensor_B, tensor_SFB};
            }
            else
            {
                return cutlass::reference::host::GettMainloopParams<float, decltype(tensor_A), decltype(tensor_B)>{
                    tensor_A, tensor_B};
            }
        };

        auto mainloop_params = get_mainloop_params();

        auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
        auto layout_C = make_layout(make_shape(M, N, 1), stride_C);
        auto tensor_C = cute::make_tensor(make_iterator(_C.host_data()), layout_C);

        auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));
        auto layout_D = make_layout(make_shape(M, N, 1), stride_D);
        auto tensor_D = cute::make_tensor(make_iterator(_D_ref.host_data()), layout_D);

        cutlass::reference::host::GettEpilogueParams<float, float, float, float, decltype(tensor_C), decltype(tensor_D)>
            epilogue_params{};

        epilogue_params.C = tensor_C;
        epilogue_params.D = tensor_D;
        epilogue_params.alpha = options.alpha;
        epilogue_params.beta = options.beta;

        // Run reference gemm
        cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
        // Reference is run on host, so copy results to device
        _D_ref.sync_device();

        // run reference allreduce
        ncclComm_t comm = NcclCommunicator::instance().comm;
        auto dtype = ToType<ElementD>{}.nccl_value;
        TLLM_NCCL_CHECK(
            ncclAllReduce(_D_ref.device_data(), _D_ref.device_data(), _D_ref.size(), dtype, ncclSum, comm, stream));
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Compare results
        float const epsilon(0.1f);
        float const nonzero_floor(std::numeric_limits<float>::min());
        int local_passed = 1;

        ElementD* ptr = isMultiGpu() ? _D_nvls.getUnicastPointer() : _D.device_data();
        ElementD* ptr_ref = _D_ref.device_data();
        // Compare D output
        local_passed &= compareRelativelyEqual(ptr_ref, ptr, M * N, epsilon, nonzero_floor);

        // Aggregate results - if 1 rank fails, then all ranks fail.
        int ranks_passed = 0;
        COMM_SESSION.allreduce(&local_passed, &ranks_passed, 1, MpiType::kINT32, MpiOp::SUM);
        return ranks_passed == options.tp;
    }

    template <typename Element, typename Layout>
    bool initializeTensor(cutlass::TensorView<Element, Layout> view, uint64_t seed)
    {
        double scope_max, scope_min;
        constexpr int bits_input = cutlass::sizeof_bits<Element>::value;

        if constexpr (bits_input == 1)
        {
            scope_max = 2;
            scope_min = 0;
        }
        else if constexpr (bits_input <= 6)
        {
            scope_max = 2;
            scope_min = -2;
        }
        else if constexpr (bits_input <= 8)
        {
            if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>)
            {
                scope_max = 4;
                scope_min = 1;
            }
            else
            {
                scope_max = 1;
                scope_min = -1;
            }
        }
        else
        {
            scope_max = 4;
            scope_min = -4;
        }
        cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);

        return true;
    }

    // Return scale-factor A & B tensor layouts
    auto getLayoutSF_AB(int M, int N, int K)
    {
        switch (getSMVersion())
        {
        case 100: // blackwell
        {
            // Unfortunately have to construct mainloop in order to extract SFA/SFB layouts
            using MainloopElementA = cute::tuple<ElementA, ElementSFA>;
            using MainloopElementB = cute::tuple<ElementB, ElementSFB>;
            constexpr static int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
            constexpr static int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
            using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
                cutlass::arch::OpClassBlockScaledTensorOp, MainloopElementA, LayoutA, AlignmentA, MainloopElementB,
                LayoutB, AlignmentB, float, Shape<_128, _128, _128>, Shape<_1, _1, _1>,
                cutlass::gemm::collective::StageCount<1>,
                cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100>::CollectiveOp;
            using Sm1xxBlkScaledConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

            auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
            auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));
            return std::make_tuple(layout_SFA, layout_SFB);
        }
        case 90: // hopper
            TLLM_THROW("A/B tensor scaling not supported on Sm90 yet");
        default: TLLM_THROW("SM version not supported");
        }
    }

    cutlass::HostTensor<ElementA, cutlass::layout::PackedVectorLayout> _A;
    cutlass::HostTensor<ElementB, cutlass::layout::PackedVectorLayout> _B;
    cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> _C;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> _D;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> _D_ref;
    cutlass::HostTensor<float, cutlass::layout::PackedVectorLayout> _alpha_vec;
    // Requires conditional because cannot have HostTensor with type void
    // which is the case when we have no scale-factors.
    typename std::conditional<IsInputScalingNeeded,
        cutlass::HostTensor<ElementSFA, cutlass::layout::PackedVectorLayout>, void*>::type _SFA;
    typename std::conditional<IsInputScalingNeeded,
        cutlass::HostTensor<ElementSFB, cutlass::layout::PackedVectorLayout>, void*>::type _SFB;
    DeviceAllocationNvls<ElementD> _D_nvls;
    std::shared_ptr<cutlass_kernels::PersistentWorkspaceInterface> _workspace;
    std::shared_ptr<cutlass_kernels::GemmAllReduceImplInterface> _gemm;
};

using MyTypes = testing::Types<
    // fp4xfp4=fp16
    TestConfig<cutlass::float_e2m1_t, cutlass::float_e2m1_t, cutlass::half_t, cutlass::half_t, cutlass::float_ue4m3_t,
        cutlass::float_ue4m3_t>,
    // fp8xfp8=fp16
    TestConfig<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t, cutlass::half_t>,
    // fp16xfp16=fp16
    TestConfig<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t>>;

TYPED_TEST_SUITE(GemmAllReduceFixture, MyTypes);

/////////////////////////////////////////////////////////////////////
// ATTENTION: run test with mpi `mpi -np <NP> ./gemmAllReduceTest'
/////////////////////////////////////////////////////////////////////
TYPED_TEST(GemmAllReduceFixture, RunnerTest)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    if (!options.verify)
    {
        TLLM_LOG_WARNING("Skipping verify - return success");
    }
    else
    {
        this->run(stream);
    }
    this->bench(stream);
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

    int device_count;
    TLLM_CUDA_CHECK(cudaGetDeviceCount(&device_count));

    int device_id = COMM_SESSION.getRank() % device_count;
    TLLM_CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp props;
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));

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

    // Ensure only 1 rank prints
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (COMM_SESSION.getRank() != 0)
    {
        delete listeners.Release(listeners.default_result_printer());
    }

    return RUN_ALL_TESTS();
}
