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
    explicit GemmAllReduceRunner(at::IntArrayRef max_problem_shape, at::ScalarType A_dtype, at::ScalarType B_dtype, at::ScalarType outputDtype, bool inputIsFP4, bool sfUseUE8M0, int64_t rank, at::IntArrayRef tp_group)
        : mRank(static_cast<int>(rank)), mOutputDtype(outputDtype)
    {
        TORCH_CHECK(!sfUseUE8M0, "use UE8M0 for FP4 Block Scale Factors is not supported yet");

        // Convert at::IntArrayRef to std::set<int>
        for (int i = 0; i < tp_group.size(); ++i) {
            mTPGroup.insert(tp_group[i]);
        }

        // allocate gemm
        GemmAllReduceRunnerFactory factory;
        mGemm = factory.create(A_dtype, B_dtype, outputDtype, inputIsFP4);

        printf("allocated mGEMM\n");

        // allocate workspace
        auto m = static_cast<int>(max_problem_shape[0]);
        auto n = static_cast<int>(max_problem_shape[1]);
        auto k = static_cast<int>(max_problem_shape[2]);

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
    }

    ~GemmAllReduceRunner()
    {
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
        TORCH_CHECK(mat1.dim() == 2, "mat1 must be of size [M, K]");
        TORCH_CHECK(mat2.dim() == 2, "mat2 must be of size [N, K]");

        printf("torch::GemmAllReduceRunner::runGemm\n");
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
        int32_t m = mat1.sizes()[0];
        int32_t n = mat2.sizes()[0];
        int32_t k = mat1.sizes()[1];

        printf("gemm_allreduce, m: %d, n: %d, k: %d\n", m, n, k);

        at::IntArrayRef output_shape = {m, n};

        // TODO (xsimmons): use torch_ext::create_multicast_tensor instead of DeviceAllocationNvls for output and workspace.

        torch::Tensor output = torch::from_blob(
            mOutput.getUnicastPointer(),
            output_shape,
            [](void* ptr) { /* no op */ },
            mOutputDtype);

        GemmAllReduceImplInterface::ProblemArgs args;
        args.argA(mat1.data_ptr())
            .argB(mat2.data_ptr())
            .argD(output.data_ptr(), mOutput.getMulticastPointer(), (void**)mOutput.getIpcUnicastPointers())
            .argAScale(mat1Scale.data_ptr())
            .argBScale(mat2Scale.data_ptr())
            .argBeta(0.f) // no bias
            .argAlphaPtr((float const*)(globalScale.data_ptr()))
            .argRanks(mRank, mTPGroup)
            .argWorkspace(mWorkspace.get());

        mGemm->run(args, stream);

        return output;
    }

    // Overloaded operator() to match Python call pattern
    at::Tensor operator()(std::vector<at::Tensor> inputs)
    {
        TORCH_CHECK(inputs.size() >= 5, "Expected at least 5 input tensors");
        return runGemm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
    }

private:
    at::ScalarType mOutputDtype;
    int mRank;
    std::set<int> mTPGroup;
    std::shared_ptr<GemmAllReduceImplInterface> mGemm;
    std::shared_ptr<PersistentWorkspaceInterface> mWorkspace;
    tr::DeviceAllocationNvls<char> mOutput;
};

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::GemmAllReduceRunner>("GemmAllReduceRunner")
        .def(torch::init<at::IntArrayRef, at::ScalarType, at::ScalarType, at::ScalarType, bool, bool, int64_t, at::IntArrayRef>())
        .def("__call__", &torch_ext::GemmAllReduceRunner::operator());

}