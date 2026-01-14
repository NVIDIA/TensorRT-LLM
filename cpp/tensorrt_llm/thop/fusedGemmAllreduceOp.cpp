/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass_extensions/gemm_configs.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/allreduce_gemm_runner.h"
#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/EmptyTensor.h>

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

namespace
{
struct AllocationKey
{
    int64_t device_index;
    std::set<int> group;

    bool operator==(AllocationKey const& other) const
    {
        return device_index == other.device_index && group == other.group;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "AllocationKey(device: " << device_index << ", group: [";
        for (int rank : group)
        {
            ss << rank << ", ";
        }
        ss << "])";
        return ss.str();
    }
};

struct AllocationKeyHash
{
    size_t operator()(AllocationKey const& key) const
    {
        size_t seed = 0;

        // Hash the device index
        hash_combine(seed, key.device_index);

        // Hash the set elements
        for (auto const& elem : key.group)
        {
            hash_combine(seed, elem);
        }

        return seed;
    }

private:
    template <typename T>
    static void hash_combine(size_t& seed, T const& val)
    {
        seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
};

class IpcNvlsHandleWrapper
{
public:
    IpcNvlsHandleWrapper(size_t size, std::set<int> groups)
        : mSize(size)
    {
        mHandle = tensorrt_llm::runtime::ipcNvlsAllocate(size, groups);
    }

    tensorrt_llm::runtime::IpcNvlsHandle* getHandle() const
    {
        return mHandle;
    }

    size_t getSize() const
    {
        return mSize;
    }

    ~IpcNvlsHandleWrapper()
    {
        tensorrt_llm::runtime::ipcNvlsFree(mHandle);
    }

private:
    size_t mSize;
    tensorrt_llm::runtime::IpcNvlsHandle* mHandle;
};

std::once_flag init_flag;

size_t getPreferredWorkspaceSize()
{
    // 128MB
    static size_t preferredWorkspaceSize = 134217728;
    std::call_once(init_flag,
        [&]()
        {
            char const* envWorkspaceSize = std::getenv("TRTLLM_GEMM_ALLREDUCE_WORKSPACE_SIZE");
            size_t workspaceSize = 0;
            if (envWorkspaceSize != nullptr)
            {
                workspaceSize = std::atoi(envWorkspaceSize);
            }
            preferredWorkspaceSize = std::max(preferredWorkspaceSize, workspaceSize);
        });
    return preferredWorkspaceSize;
}

class GemmAllreduceNvlsMemoryManager
{
public:
    GemmAllreduceNvlsMemoryManager()
    {
        TLLM_LOG_DEBUG("GemmAllreduceNvlsMemoryManager constructor");
    }

    ~GemmAllreduceNvlsMemoryManager()
    {
        TLLM_LOG_DEBUG("GemmAllreduceNvlsMemoryManager destructor");
    }

    std::pair<PersistentWorkspaceInterface*, tensorrt_llm::runtime::IpcNvlsHandle*> getWorkspace(
        GemmAllReduceImplInterface* runner, GemmAllReduceImplInterface::ProblemArgs const& problem,
        AllocationKey const& key)
    {
        int M = std::get<0>(problem.problem_size);
        int N = std::get<1>(problem.problem_size);
        size_t requiredSize = M * N * 2;
        size_t preferredWorkspaceSize = getPreferredWorkspaceSize();
        if (requiredSize > preferredWorkspaceSize)
        {
            std::stringstream ss;
            ss << "Please set TRTLLM_GEMM_ALLREDUCE_WORKSPACE_SIZE to at least " << requiredSize << " bytes";
            TLLM_THROW("%s", ss.str().c_str());
        }

        auto handle = mHandles[key];
        if (handle == nullptr)
        {
            TLLM_LOG_DEBUG("Creating allreduce workspace for %s", key.toString().c_str());
            handle = std::make_shared<IpcNvlsHandleWrapper>(preferredWorkspaceSize, key.group);
            GemmAllReduceImplInterface::ProblemArgs tmpArgs;
            int maxN = 16384;
            int maxM = preferredWorkspaceSize / (maxN * 2);
            tmpArgs.argProblemShape(maxM, maxN, 512, 1)
                .argRanks(problem.rank, problem.ranks)
                .argLaunchConfig(runner->getSupportedLaunchConfigs()[0]);
            auto workspace = runner->getPersistentWorkspace(tmpArgs);
            workspace->allocate();
            mWorkspaces[key] = workspace;
            mHandles[key] = handle;
        }
        return std::make_pair(mWorkspaces[key].get(), mHandles[key]->getHandle());
    }

private:
    std::unordered_map<AllocationKey, std::shared_ptr<PersistentWorkspaceInterface>, AllocationKeyHash> mWorkspaces;
    std::unordered_map<AllocationKey, std::shared_ptr<IpcNvlsHandleWrapper>, AllocationKeyHash> mHandles;
};

GemmAllreduceNvlsMemoryManager* getGemmAllreduceNvlsMemoryManager()
{
    static GemmAllreduceNvlsMemoryManager gNvlsMemoryManager;
    return &gNvlsMemoryManager;
}

at::Tensor runGemmImpl(GemmAllReduceImplInterface* runner, GemmAllReduceImplInterface::ProblemArgs& problem,
    at::ScalarType outputDtype, c10::cuda::CUDAStream stream)
{
    AllocationKey key{stream.device_index(), problem.ranks};
    auto [workspace, handle] = getGemmAllreduceNvlsMemoryManager()->getWorkspace(runner, problem, key);
    problem.argD((void*) handle->uc_ptr, (void*) handle->mc_ptr, (void**) handle->ipc_uc_ptrs.data());
    problem.argWorkspace(workspace);
    runner->run(problem, stream);
    size_t dSize
        = std::get<0>(problem.problem_size) * std::get<1>(problem.problem_size) * c10::elementSize(outputDtype);
    auto D = at::detail::empty_cuda({std::get<0>(problem.problem_size), std::get<1>(problem.problem_size)}, outputDtype,
        stream.device(), std::nullopt);
    TLLM_CUDA_CHECK(cudaMemcpyAsync(
        D.data_ptr(), reinterpret_cast<void const*>(handle->uc_ptr), dSize, cudaMemcpyDeviceToDevice, stream));
    return D;
}
} // namespace

namespace torch_ext
{

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
            TLLM_THROW("Unsupported output dtype: %s", torch::toString(outputDtype));
        }

        mConfigs = mRunner->getSupportedLaunchConfigs();
    }

    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale, at::Tensor const& alpha, int64_t configIdx) const
    {
        if (configIdx < 0)
            configIdx = 0;

        TORCH_CHECK(configIdx < int64_t(mConfigs.size()), "configIdx out of bounds");
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
        problemArgs.argAlphaPtr(reinterpret_cast<float const*>(alpha.const_data_ptr()));
        problemArgs.argBeta(0.f);
        problemArgs.argRanks(mRank, mGroup);
        problemArgs.argLaunchConfig(mConfigs[configIdx]);

        auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
        return runGemmImpl(mRunner.get(), problemArgs, mOutputDtype, stream);
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
