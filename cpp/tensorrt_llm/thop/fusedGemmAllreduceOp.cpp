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
#include "tensorrt_llm/runtime/ipcNvlsMemoryTorch.h"
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
enum class GemmAllreduceRunnerKind : int64_t
{
    kDefault,
    kFp8BlockScale,
};

struct AllocationKey
{
    int64_t device_index;
    std::set<int> group;
    uintptr_t rendezvous_identity{0};
    at::ScalarType input_dtype;
    at::ScalarType output_dtype;
    GemmAllreduceRunnerKind runner_kind{GemmAllreduceRunnerKind::kDefault};

    bool operator==(AllocationKey const& other) const
    {
        return device_index == other.device_index && group == other.group
            && rendezvous_identity == other.rendezvous_identity && input_dtype == other.input_dtype
            && output_dtype == other.output_dtype && runner_kind == other.runner_kind;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "AllocationKey(device: " << device_index << ", group: [";
        for (int rank : group)
        {
            ss << rank << ", ";
        }
        ss << "], rendezvous: " << rendezvous_identity << ", input dtype: " << static_cast<int>(input_dtype)
           << ", output dtype: " << static_cast<int>(output_dtype)
           << ", runner kind: " << static_cast<int64_t>(runner_kind) << ")";
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
        hash_combine(seed, key.rendezvous_identity);
        hash_combine(seed, static_cast<int>(key.input_dtype));
        hash_combine(seed, static_cast<int>(key.output_dtype));
        hash_combine(seed, static_cast<int64_t>(key.runner_kind));

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
    IpcNvlsHandleWrapper(size_t size, tensorrt_llm::runtime::IpcNvlsRendezvousPtr rendezvous)
        : mSize(size)
        , mRendezvous(std::move(rendezvous))
    {
        TLLM_CHECK_WITH_INFO(mRendezvous != nullptr, "NVLS rendezvous must not be null");
        mHandle = mRendezvous->allocate(size);
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
    tensorrt_llm::runtime::IpcNvlsRendezvousPtr mRendezvous;
    tensorrt_llm::runtime::IpcNvlsHandle* mHandle{nullptr};
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
        AllocationKey const& key, at::ScalarType outputDtype,
        tensorrt_llm::runtime::IpcNvlsRendezvousPtr const& rendezvous)
    {
        int M = std::get<0>(problem.problem_size);
        int N = std::get<1>(problem.problem_size);
        size_t const elementSize = c10::elementSize(outputDtype);
        size_t requiredSize = static_cast<size_t>(M) * static_cast<size_t>(N) * elementSize;
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
            handle = std::make_shared<IpcNvlsHandleWrapper>(preferredWorkspaceSize, rendezvous);
            GemmAllReduceImplInterface::ProblemArgs tmpArgs;
            int maxN = 16384;
            int maxM = preferredWorkspaceSize / (maxN * elementSize);
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
    at::ScalarType inputDtype, at::ScalarType outputDtype, c10::cuda::CUDAStream stream,
    tensorrt_llm::runtime::IpcNvlsRendezvousPtr rendezvous = nullptr,
    GemmAllreduceRunnerKind runnerKind = GemmAllreduceRunnerKind::kDefault)
{
    if (!rendezvous)
    {
        rendezvous = tensorrt_llm::runtime::makeMpiIpcNvlsRendezvous(problem.ranks);
    }
    AllocationKey key{
        stream.device_index(), problem.ranks, rendezvous->identity(), inputDtype, outputDtype, runnerKind};
    auto [workspace, handle]
        = getGemmAllreduceNvlsMemoryManager()->getWorkspace(runner, problem, key, outputDtype, rendezvous);
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
        for (int64_t groupRank : group)
        {
            mGroup.insert(static_cast<int>(groupRank));
        }
        initializeRunner();
    }

#ifdef USING_OSS_CUTLASS_ALLREDUCE_GEMM
    void setProcessGroup(c10::intrusive_ptr<c10d::ProcessGroup> processGroup)
    {
        mRendezvous = tensorrt_llm::runtime::makeTorchDistIpcNvlsRendezvous(std::move(processGroup));
        mRank = mRendezvous->rank();
        mGroup.clear();
        for (int rank = 0; rank < mRendezvous->size(); ++rank)
        {
            mGroup.insert(rank);
        }
        initializeRunner();
    }
#endif

    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale, at::Tensor const& alpha, int64_t configIdx) const
    {
        if (configIdx < 0)
        {
            configIdx = 0;
        }

        TORCH_CHECK(configIdx < int64_t(mConfigs.size()), "configIdx out of bounds");
        const int64_t M = mat1.size(0);
        const int64_t N = mat2.size(0);
        const int64_t K = mat1.size(1) * 2;

        GemmAllReduceImplInterface::ProblemArgs problemArgs;
        problemArgs.argProblemShape(M, N, K, 1)
            .argA(mat1.data_ptr())
            .argB(mat2.data_ptr())
            .argAScale(mat1Scale.data_ptr())
            .argBScale(mat2Scale.data_ptr())
            .argC(nullptr)
            .argAlphaPtr(reinterpret_cast<float const*>(alpha.const_data_ptr()))
            .argBeta(0.f)
            .argRanks(mRank, mGroup)
            .argLaunchConfig(mConfigs[configIdx]);

        auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
        return runGemmImpl(mRunner.get(), problemArgs, mat1.scalar_type(), mOutputDtype, stream, mRendezvous);
    }

    int64_t getNumConfigs() const
    {
        return static_cast<int64_t>(mConfigs.size());
    }

private:
    template <typename Traits>
    void createRunner()
    {
#ifdef USING_OSS_CUTLASS_ALLREDUCE_GEMM
        if (mRendezvous)
        {
            mRunner = std::make_shared<GemmAllReduceImplRunner<Traits>>(mRendezvous);
            return;
        }
#endif
        mRunner = std::make_shared<GemmAllReduceImplRunner<Traits>>();
    }

    void initializeRunner()
    {
        if (mOutputDtype == at::ScalarType::Half)
        {
            using Traits = GemmTypes<cutlass::float_e2m1_t, cutlass::float_e2m1_t, cutlass::half_t, cutlass::half_t,
                cutlass::float_ue4m3_t, cutlass::float_ue4m3_t, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
                cutlass::layout::RowMajor, cutlass::layout::RowMajor>;
            createRunner<Traits>();
        }
        else if (mOutputDtype == at::ScalarType::BFloat16)
        {
            using Traits = GemmTypes<cutlass::float_e2m1_t, cutlass::float_e2m1_t, cutlass::bfloat16_t,
                cutlass::bfloat16_t, cutlass::float_ue4m3_t, cutlass::float_ue4m3_t, cutlass::layout::RowMajor,
                cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>;
            createRunner<Traits>();
        }
        else
        {
            TLLM_THROW("Unsupported output dtype: %s", torch::toString(mOutputDtype));
        }
        mConfigs = mRunner->getSupportedLaunchConfigs();
    }

    at::ScalarType mOutputDtype;
    tensorrt_llm::runtime::IpcNvlsRendezvousPtr mRendezvous;
    int mRank;
    std::set<int> mGroup;
    std::shared_ptr<GemmAllReduceImplInterface> mRunner{nullptr};
    std::vector<GemmAllReduceImplInterface::LaunchConfig> mConfigs;
};

#ifdef USING_OSS_CUTLASS_ALLREDUCE_GEMM
class Fp8BlockScaleGemmAllreduceRunner : public torch::CustomClassHolder
{
public:
    Fp8BlockScaleGemmAllreduceRunner(at::ScalarType outputDtype, c10::intrusive_ptr<c10d::ProcessGroup> processGroup)
        : mOutputDtype(outputDtype)
        , mRendezvous(tensorrt_llm::runtime::makeTorchDistIpcNvlsRendezvous(std::move(processGroup)))
        , mRank(mRendezvous->rank())
    {
        for (int rank = 0; rank < mRendezvous->size(); ++rank)
        {
            mGroup.insert(rank);
        }

        auto const smVersion = tensorrt_llm::common::getSMVersion();
        if (smVersion == 90 && outputDtype == at::ScalarType::BFloat16)
        {
            using Traits = GemmTypes<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
                cutlass::bfloat16_t, float, float, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
                cutlass::layout::RowMajor, cutlass::layout::RowMajor>;
            mRunner = std::make_shared<GemmAllReduceImplRunner<Traits>>(mRendezvous);
        }
        else
        {
            TLLM_THROW("FP8 block-scale GEMM+allreduce supports only BF16 output on SM90; got output dtype %s on SM%d",
                torch::toString(outputDtype), smVersion);
        }
        mConfigs = mRunner->getSupportedLaunchConfigs();
    }

    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1Scale,
        at::Tensor const& mat2Scale, int64_t configIdx) const
    {
        TORCH_CHECK(
            mat1.scalar_type() == at::ScalarType::Float8_e4m3fn && mat2.scalar_type() == at::ScalarType::Float8_e4m3fn,
            "MXFP8 GEMM+allreduce operands must be float8_e4m3fn");
        TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "MXFP8 GEMM+allreduce inputs must be matrices");
        TORCH_CHECK(mat1.size(1) == mat2.size(1), "MXFP8 GEMM+allreduce K dimensions must match");
        TORCH_CHECK(mat1.is_contiguous() && mat2.is_contiguous(), "MXFP8 GEMM+allreduce inputs must be contiguous");
        TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda(), "FP8 block-scale GEMM+allreduce inputs must be CUDA tensors");
        TORCH_CHECK(mat1.get_device() == mat2.get_device(),
            "FP8 block-scale GEMM+allreduce operands must be on the same device");
        TORCH_CHECK(mat1Scale.is_cuda() && mat2Scale.is_cuda(), "FP8 block scale factors must be CUDA tensors");
        TORCH_CHECK(mat1Scale.get_device() == mat1.get_device() && mat2Scale.get_device() == mat1.get_device(),
            "FP8 block scale factors and operands must be on the same device");
        TORCH_CHECK(
            mat1Scale.is_contiguous() && mat2Scale.is_contiguous(), "FP8 block scale factors must be contiguous");
        TORCH_CHECK(
            mat1Scale.scalar_type() == at::ScalarType::Float && mat2Scale.scalar_type() == at::ScalarType::Float,
            "SM90 FP8 block-scale GEMM+allreduce requires FP32 scales");
        auto const m = mat1.size(0);
        auto const n = mat2.size(0);
        auto const kBlocks = (mat1.size(1) + 127) / 128;
        auto const paddedM = (m + 3) / 4 * 4;
        auto const nBlocks = (n + 127) / 128;
        TORCH_CHECK(
            mat1Scale.numel() >= paddedM * kBlocks, "SM90 FP8 activation scale tensor is too small for the GEMM shape");
        TORCH_CHECK(
            mat2Scale.numel() >= nBlocks * kBlocks, "SM90 FP8 weight scale tensor is too small for the GEMM shape");
        if (configIdx < 0)
        {
            configIdx = 0;
        }
        TORCH_CHECK(configIdx < static_cast<int64_t>(mConfigs.size()), "configIdx out of bounds");

        GemmAllReduceImplInterface::ProblemArgs problemArgs;
        problemArgs.argProblemShape(mat1.size(0), mat2.size(0), mat1.size(1), 1)
            .argA(mat1.data_ptr())
            .argB(mat2.data_ptr())
            .argAScale(mat1Scale.data_ptr())
            .argBScale(mat2Scale.data_ptr())
            .argC(nullptr)
            .argAlpha(1.0F)
            .argBeta(0.0F)
            .argRanks(mRank, mGroup)
            .argLaunchConfig(mConfigs[configIdx]);

        auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
        return runGemmImpl(mRunner.get(), problemArgs, at::ScalarType::Float8_e4m3fn, mOutputDtype, stream, mRendezvous,
            GemmAllreduceRunnerKind::kFp8BlockScale);
    }

    int64_t getNumConfigs() const
    {
        return static_cast<int64_t>(mConfigs.size());
    }

private:
    at::ScalarType mOutputDtype;
    tensorrt_llm::runtime::IpcNvlsRendezvousPtr mRendezvous;
    int mRank;
    std::set<int> mGroup;
    std::shared_ptr<GemmAllReduceImplInterface> mRunner;
    std::vector<GemmAllReduceImplInterface::LaunchConfig> mConfigs;
};

class GemmAllreduceRunner : public torch::CustomClassHolder
{
public:
    GemmAllreduceRunner(
        at::ScalarType inputDtype, at::ScalarType outputDtype, c10::intrusive_ptr<c10d::ProcessGroup> processGroup)
        : mInputDtype(inputDtype)
        , mOutputDtype(outputDtype)
        , mRendezvous(tensorrt_llm::runtime::makeTorchDistIpcNvlsRendezvous(std::move(processGroup)))
        , mRank(mRendezvous->rank())
    {
        for (int rank = 0; rank < mRendezvous->size(); ++rank)
        {
            mGroup.insert(rank);
        }

        if (inputDtype == at::ScalarType::Half && outputDtype == at::ScalarType::Half)
        {
            using Traits = GemmTypes<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, void, void,
                cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                cutlass::layout::RowMajor>;
            mRunner = std::make_shared<GemmAllReduceImplRunner<Traits>>(mRendezvous);
        }
        else if (inputDtype == at::ScalarType::BFloat16 && outputDtype == at::ScalarType::BFloat16)
        {
            using Traits = GemmTypes<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t,
                void, void, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                cutlass::layout::RowMajor>;
            mRunner = std::make_shared<GemmAllReduceImplRunner<Traits>>(mRendezvous);
        }
        else if (inputDtype == at::ScalarType::Float8_e4m3fn && outputDtype == at::ScalarType::Half)
        {
            using Traits = GemmTypes<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::half_t, cutlass::half_t,
                void, void, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor,
                cutlass::layout::RowMajor>;
            mRunner = std::make_shared<GemmAllReduceImplRunner<Traits>>(mRendezvous);
        }
        else if (inputDtype == at::ScalarType::Float8_e4m3fn && outputDtype == at::ScalarType::BFloat16)
        {
            using Traits = GemmTypes<cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t,
                cutlass::bfloat16_t, void, void, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor,
                cutlass::layout::RowMajor, cutlass::layout::RowMajor>;
            mRunner = std::make_shared<GemmAllReduceImplRunner<Traits>>(mRendezvous);
        }
        else
        {
            TLLM_THROW("Unsupported GEMM+allreduce dtype combination: input=%s, output=%s", torch::toString(inputDtype),
                torch::toString(outputDtype));
        }
        mConfigs = mRunner->getSupportedLaunchConfigs();
    }

    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, int64_t configIdx) const
    {
        TORCH_CHECK(mat1.is_cuda() && mat2.is_cuda(), "GEMM+allreduce inputs must be CUDA tensors");
        TORCH_CHECK(mat1.dim() == 2 && mat2.dim() == 2, "GEMM+allreduce inputs must be matrices");
        TORCH_CHECK(mat1.scalar_type() == mInputDtype && mat2.scalar_type() == mInputDtype,
            "GEMM+allreduce operand dtypes must match the runner input dtype");
        TORCH_CHECK(mat1.is_contiguous() && mat2.is_contiguous(), "GEMM+allreduce inputs must be contiguous");
        TORCH_CHECK(mat1.size(1) == mat2.size(1), "GEMM+allreduce K dimensions must match");
        if (configIdx < 0)
        {
            configIdx = 0;
        }
        TORCH_CHECK(configIdx < static_cast<int64_t>(mConfigs.size()), "configIdx out of bounds");

        int64_t const M = mat1.size(0);
        int64_t const N = mat2.size(0);
        int64_t const K = mat1.size(1);

        GemmAllReduceImplInterface::ProblemArgs problemArgs;
        problemArgs.argProblemShape(M, N, K, 1)
            .argA(mat1.data_ptr())
            .argB(mat2.data_ptr())
            .argC(nullptr)
            .argAlpha(1.0F)
            .argBeta(0.0F)
            .argRanks(mRank, mGroup)
            .argLaunchConfig(mConfigs[configIdx]);

        auto stream = at::cuda::getCurrentCUDAStream(mat1.get_device());
        return runGemmImpl(mRunner.get(), problemArgs, mInputDtype, mOutputDtype, stream, mRendezvous);
    }

    int64_t getNumConfigs() const
    {
        return static_cast<int64_t>(mConfigs.size());
    }

private:
    at::ScalarType mInputDtype;
    at::ScalarType mOutputDtype;
    tensorrt_llm::runtime::IpcNvlsRendezvousPtr mRendezvous;
    int mRank;
    std::set<int> mGroup;
    std::shared_ptr<GemmAllReduceImplInterface> mRunner;
    std::vector<GemmAllReduceImplInterface::LaunchConfig> mConfigs;
};
#endif // USING_OSS_CUTLASS_ALLREDUCE_GEMM

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    auto fp4Runner = m.class_<torch_ext::Fp4GemmAllreduceRunner>("Fp4GemmAllreduceRunner");
    fp4Runner.def(torch::init<at::ScalarType, int64_t, torch::List<int64_t>>());
#ifdef USING_OSS_CUTLASS_ALLREDUCE_GEMM
    fp4Runner.def("set_process_group", &torch_ext::Fp4GemmAllreduceRunner::setProcessGroup);
#endif
    fp4Runner.def("run_gemm", &torch_ext::Fp4GemmAllreduceRunner::runGemm)
        .def("get_num_configs", &torch_ext::Fp4GemmAllreduceRunner::getNumConfigs);
#ifdef USING_OSS_CUTLASS_ALLREDUCE_GEMM
    m.class_<torch_ext::Fp8BlockScaleGemmAllreduceRunner>("Fp8BlockScaleGemmAllreduceRunner")
        .def(torch::init<at::ScalarType, c10::intrusive_ptr<c10d::ProcessGroup>>())
        .def("run_gemm", &torch_ext::Fp8BlockScaleGemmAllreduceRunner::runGemm)
        .def("get_num_configs", &torch_ext::Fp8BlockScaleGemmAllreduceRunner::getNumConfigs);

    m.class_<torch_ext::GemmAllreduceRunner>("GemmAllreduceRunner")
        .def(torch::init<at::ScalarType, at::ScalarType, c10::intrusive_ptr<c10d::ProcessGroup>>())
        .def("run_gemm", &torch_ext::GemmAllreduceRunner::runGemm)
        .def("get_num_configs", &torch_ext::GemmAllreduceRunner::getNumConfigs);
#endif
}
