/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "thUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include <pybind11/stl.h> // For std::vector conversion

#if defined(USING_OSS_CUTLASS_ALLREDUCE_GEMM)
#include "tensorrt_llm/kernels/cutlass_kernels/include/allreduce_gemm_runner.h"
using namespace tensorrt_llm::kernels::opened_cutlass_kernels;
#else
#include "allreduce_gemm_runner.h"
using namespace tensorrt_llm::kernels::cutlass_kernels;
#endif

namespace tr = tensorrt_llm::runtime;

namespace torch_ext
{
// TODO: remove isFP4 when supported natively by torch.
class GemmAllReduceRunnerFactory
{
public:
    GemmAllReduceRunnerFactory()
    {
        // FP16
        registerRunner<at::ScalarType::Half, at::ScalarType::Half, at::ScalarType::Half>();
        // BF16
        registerRunner<at::ScalarType::BFloat16, at::ScalarType::BFloat16, at::ScalarType::BFloat16>();
        // FP8
        registerRunner<at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e4m3fn, at::ScalarType::Half>();
        registerRunner<at::ScalarType::Float8_e4m3fn, at::ScalarType::Float8_e4m3fn, at::ScalarType::BFloat16>();
        // FP4
        registerRunner<torch_ext::FLOAT4_E2M1X2, torch_ext::FLOAT4_E2M1X2, at::ScalarType::Half, true>();
        registerRunner<torch_ext::FLOAT4_E2M1X2, torch_ext::FLOAT4_E2M1X2, at::ScalarType::BFloat16, true>();
    }

    ~GemmAllReduceRunnerFactory() = default;

    std::shared_ptr<GemmAllReduceImplInterface> create(at::ScalarType A, at::ScalarType B, at::ScalarType D, bool isFP4)
    {
        auto key = std::make_tuple(A, B, D, isFP4);
        TLLM_CHECK_WITH_INFO(mFunctionMap.count(key) > 0, "No cutlass gemm for impl found in factory.");
        return std::shared_ptr<GemmAllReduceImplInterface>(mFunctionMap[key]());
    }

private:
    template <at::ScalarType A, at::ScalarType B, at::ScalarType D, bool isFP4 = false, bool sfUseUE8M0 = false>
    void registerRunner()
    {  
        using TypeA = decltype(tr::TorchUtils::cutlassType<A>());
        using TypeB = decltype(tr::TorchUtils::cutlassType<B>());
        using TypeD = decltype(tr::TorchUtils::cutlassType<D>());
        using TypeSF = std::conditional_t<isFP4,
                                          std::conditional_t<sfUseUE8M0, cutlass::float_ue8m0_t, cutlass::float_ue4m3_t>,
                                          void>;

        using RM = cutlass::layout::RowMajor;
        using CM = cutlass::layout::ColumnMajor;

        using GemmTraits = GemmTypes<
            TypeA, TypeB, TypeD, TypeD, TypeSF, TypeSF,
            RM, CM, RM, RM>;

        auto key = std::make_tuple(A, B, D, isFP4);

        mFunctionMap[key] = [=]() -> GemmAllReduceImplInterface* {
            return new GemmAllReduceImplRunner<GemmTraits>();
        };
    }

    using Key = std::tuple<at::ScalarType, at::ScalarType, at::ScalarType, bool>;
    using Value = std::function<GemmAllReduceImplInterface*()>;

    std::map<Key, Value> mFunctionMap;
};

class GemmAllReduceRunner : public torch::CustomClassHolder
{
public:
    // TODO: remove inputIsFP4 & sfUseUE8M0 when supported natively by torch.
    explicit GemmAllReduceRunner(
        at::IntArrayRef max_problem_shape,
        at::ScalarType A_dtype,
        at::ScalarType B_dtype,
        at::ScalarType outputDtype,
        bool inputIsFP4,
        bool sfUseUE8M0,
        int64_t rank,
        at::IntArrayRef tp_group)
        : mRank(static_cast<int>(rank)), mOutputDtype(outputDtype)
    {
        TORCH_CHECK(!sfUseUE8M0, "use UE8M0 for FP4 Block Scale Factors is not supported yet");

        // Convert at::IntArrayRef to std::set<int>
        printf("rank: %d\n", mRank);
        printf("tp_group: ");
        for (size_t i = 0; i < tp_group.size(); ++i) {
            mTPGroup.insert(tp_group[i]);
            printf("%d ", int(tp_group[i]));
        }
        printf("\n");

        // allocate gemm
        GemmAllReduceRunnerFactory factory;
        mGemm = factory.create(A_dtype, B_dtype, outputDtype, inputIsFP4);

        printf("allocated mGEMM\n");

        // allocate workspace
        auto m = static_cast<int>(max_problem_shape[0]);
        auto n = static_cast<int>(max_problem_shape[1]);
        auto k = static_cast<int>(max_problem_shape[2]);

        printf("GemmAllReduceRunner constructor: max_m: %d, n: %d, k: %d\n", m, n, k);

        GemmAllReduceImplInterface::ProblemArgs args;
        args.argProblemShape(m, n, k, 1)
            .argRanks(mRank, mTPGroup);

        mWorkspace = mGemm->getPersistentWorkspace(args);
        mWorkspace->allocate();

        printf("allocated mWORKSPACE\n");

        // allocate output
        const size_t output_bytes = m * n * torch::elementSize(outputDtype);
        mOutput.reset(output_bytes, mTPGroup);

        printf("allocated mOUTPUT\n");

        // int device_id;
        // cudaGetDevice(&device_id);

        // mReusableTensor = at::for_blob(
        //     mOutput.getUnicastPointer(),
        //     {m, n},
        //     {n, 1}, // row-major strides
        //     [](void* ptr) { /* no op */ },
        //     torch::dtype(mOutputDtype).device(torch::kCUDA, device_id));

        cudaMalloc(&mTmp, output_bytes);
    }

    ~GemmAllReduceRunner()
    {
        printf("torch::GemmAllReduceRunner::~GemmAllReduceRunner\n");
        mOutput.free();
        mWorkspace->free();
    }

    at::Tensor runGemm(
        at::Tensor const& mat1,
        at::Tensor const& mat2,
        at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale,
        at::Tensor const& globalScale)
    {
        printf("[DEBUG] runGemm: Starting function\n");
        
        TORCH_CHECK(mat1.dim() == 2, "mat1 must be of size [M, K]");
        TORCH_CHECK(mat2.dim() == 2, "mat2 must be of size [N, K]");
        TORCH_CHECK(mat1.get_device() == mat2.get_device(), "mat1 and mat2 must be on the same device");

        printf("torch::GemmAllReduceRunner::runGemm\n");
        // cudaStream_t stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
        int m = static_cast<int>(mat1.sizes()[0]);
        int n = static_cast<int>(mat2.sizes()[0]);
        int k = static_cast<int>(mat1.sizes()[1]);

        printf("[DEBUG] runGemm: Dimensions - m=%d, n=%d, k=%d\n", m, n, k);

        printf("[DEBUG] runGemm: Creating PyTorch tensor with custom allocator\n");
        
        // Option 1: Use torch::from_blob with shared_ptr for proper memory management
        int device_id;
        cudaGetDevice(&device_id);

        // Use the newer API that takes shared_ptr directly
        torch::Tensor output = torch::from_blob(
            mTmp,
            {m, n}, // shape
            {n, 1}, // row-major strides
            [](void* ptr) {
                // This will be called when the tensor is destroyed
                // Since we're using pre-allocated memory, we don't free it here
                printf("[DEBUG] Custom deleter called - not freeing memory\n");
            }, // torch::Deleter
            torch::dtype(mOutputDtype).device(torch::kCUDA, mat1.get_device()));

        // GemmAllReduceImplInterface::ProblemArgs args;
        // args.argProblemShape(m, n, k, 1)
        //     .argA(mat1.data_ptr())
        //     .argB(mat2.data_ptr())
        //     .argD(output.data_ptr(), mOutput.getMulticastPointer(), (void**)mOutput.getIpcUnicastPointers())
        //     .argAScale(mat1Scale.data_ptr())
        //     .argBScale(mat2Scale.data_ptr())
        //     .argBeta(0.f) // no bias
        //     .argAlphaPtr((float const*)(globalScale.data_ptr()))
        //     .argRanks(mRank, mTPGroup)
        //     .argWorkspace(mWorkspace.get());

        // mGemm->run(args, stream);

        // torch::Tensor output = mReusableTensor.view({m, n});
        // torch::Tensor output = mReusableTensor.narrow(0, 0, m).narrow(1, 0, n);
        // auto output = torch::empty({m, n}, torch::dtype(mOutputDtype).device(torch::kCUDA, mat1.get_device()));

        return output;
    }

    // Overloaded operator() to match Python call pattern
    at::Tensor operator()(
        at::Tensor const& mat1,
        at::Tensor const& mat2,
        at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale,
        at::Tensor const& globalScale)
    {
        return runGemm(mat1, mat2, mat1Scale, mat2Scale, globalScale);
    }

private:
    int mRank;
    at::ScalarType mOutputDtype;
    std::set<int> mTPGroup;
    std::shared_ptr<GemmAllReduceImplInterface> mGemm;
    std::shared_ptr<PersistentWorkspaceInterface> mWorkspace;
    tr::DeviceAllocationNvls<char> mOutput;
    void* mTmp;
    torch::Tensor mReusableTensor;
};

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::GemmAllReduceRunner>("GemmAllReduceRunner")
        .def(torch::init<at::IntArrayRef, at::ScalarType, at::ScalarType, at::ScalarType, bool, bool, int64_t, at::IntArrayRef>())
        .def("runGemm", &torch_ext::GemmAllReduceRunner::runGemm);
}