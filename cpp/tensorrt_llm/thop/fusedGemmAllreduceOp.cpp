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

#include "allreduce_gemm_runner.h"
#include "cutlass_extensions/gemm_configs.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/allreduce_gemm_runner.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>

#include <cstddef>
#include <cuda_fp16.h>

#include <cstdint>
#include <functional>
#include <type_traits>
#include <vector>

using tensorrt_llm::kernels::opened_cutlass_kernels::GemmAllReduceImplRunner;
using tensorrt_llm::kernels::opened_cutlass_kernels::GemmAllReduceImplInterface;
using tensorrt_llm::kernels::opened_cutlass_kernels::GemmTypes;
using tensorrt_llm::kernels::opened_cutlass_kernels::PersistentWorkspaceInterface;

namespace torch_ext
{
PersistentWorkspaceInterface* getWorkspace(
    GemmAllReduceImplInterface* runner, GemmAllReduceImplInterface::ProblemArgs const& problem)
{
    thread_local std::shared_ptr<PersistentWorkspaceInterface> curWorkspace;
    thread_local size_t curWorkspaceSize = 0;
    auto newWorkspace = runner->getPersistentWorkspace(problem);
    if (newWorkspace->size() > curWorkspaceSize)
    {
        newWorkspace->allocate();
        curWorkspaceSize = newWorkspace->size();
        curWorkspace = newWorkspace;
    }
    return curWorkspace.get();
}

class Fp4GemmAllreduceRunner : public torch::CustomClassHolder
{
public:
    explicit Fp4GemmAllreduceRunner(at::ScalarType outputDtype, int64_t rank, torch::List<int64_t> group)
        : mOutputDtype(outputDtype)
        , mRank(rank)
    {
        for (int64_t rank : group)
        {
            mGroup.insert(static_cast<int>(rank));
        }

        if (outputDtype == at::ScalarType::Half)
        {
            using Traits = GemmTypes<cutlass::float_e2m1_t, cutlass::float_e2m1_t, cutlass::half_t, cutlass::half_t,
                cutlass::float_ue4m3_t, cutlass::float_ue4m3_t, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
                cutlass::layout::RowMajor, cutlass::layout::RowMajor>;
            mRunner = std::make_shared<GemmAllReduceImplRunner<Traits>>();
        }
        else if (outputDtype == at::ScalarType::BFloat16)
        {
            using Traits = GemmTypes<cutlass::float_e2m1_t, cutlass::float_e2m1_t, cutlass::bfloat16_t,
                cutlass::bfloat16_t, cutlass::float_ue4m3_t, cutlass::float_ue4m3_t, cutlass::layout::RowMajor,
                cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>;
            mRunner = std::make_shared<GemmAllReduceImplRunner<Traits>>();
        }
        else
        {
            C10_THROW_ERROR(NotImplementedError, "Unsupported input or output dtype");
        }

        mConfigs = mRunner->getSupportedLaunchConfigs();
    }

    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale, at::Tensor const& alpha, int64_t configIdx) const
    {
        if (configIdx < 0)
            configIdx = 0;

        assert(configIdx < int64_t(mConfigs.size()));
        const int64_t M = mat1.size(0);
        const int64_t N = mat2.size(0);
        const int64_t K = mat1.size(1) * 2;

        GemmAllReduceImplInterface::ProblemArgs problemArgs;
        problemArgs.argProblemShape(M, N, K, 1);
        problemArgs.argA(mat1.data_ptr());
        problemArgs.argB(mat2.data_ptr());
        problemArgs.argAScale(mat1Scale.data_ptr());
        problemArgs.argBScale(mat2Scale.data_ptr());
        problemArgs.argC(nullptr);
        problemArgs.argAlphaPtr(alpha.const_data_ptr<float>());
        problemArgs.argBeta(0.f);
        problemArgs.argRanks(mRank, mGroup);
        problemArgs.argLaunchConfig(mConfigs[configIdx]);

        size_t dSize = M * N * c10::elementSize(mOutputDtype);
        auto handle = tensorrt_llm::runtime::ipcNvlsAllocate(dSize, mGroup);
        problemArgs.argD((void*) handle->uc_ptr, (void*) handle->mc_ptr, (void**) handle->ipc_uc_ptrs.data());

        auto workspace = getWorkspace(mRunner.get(), problemArgs);
        problemArgs.argWorkspace(workspace);

        auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
        mRunner->run(problemArgs, stream);

        auto options = mat1.options().dtype(mOutputDtype);
        auto deleter = [=](void* unused) { ipcNvlsFree(handle); };
        auto D = at::from_blob((void*) handle->uc_ptr, {M, N}, {N, 1}, deleter, options);
        return D;
    }

    int64_t getNumConfigs() const
    {
        return static_cast<int64_t>(mConfigs.size());
    }

private:
    at::ScalarType mOutputDtype;
    int mRank;
    std::set<int> mGroup;
    std::shared_ptr<GemmAllReduceImplInterface> mRunner{nullptr};
    std::vector<GemmAllReduceImplInterface::LaunchConfig> mConfigs;
};

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::Fp4GemmAllreduceRunner>("Fp4GemmAllreduceRunner")
        .def(torch::init<at::ScalarType, int64_t, torch::List<int64_t>>())
        .def("run_gemm", &torch_ext::Fp4GemmAllreduceRunner::runGemm)
        .def("get_num_configs", &torch_ext::Fp4GemmAllreduceRunner::getNumConfigs);
}
