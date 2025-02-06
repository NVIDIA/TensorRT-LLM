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
#pragma once

#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <set>

#include "cutlass/layout/layout.h"
#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

using namespace cute;
using namespace tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm::kernels::cutlass_kernels
{
enum GemmAllReduceImpl
{
    NVLS_1SHOT,
    NVLS_2SHOT
};

// Decouples IPluginResource from the GemmAllReduce runner interface.
class PersistentWorkspaceInterface
{
public:
    virtual ~PersistentWorkspaceInterface() = default;
    virtual void allocate() = 0;
    virtual int free() = 0;
};

class GemmAllReduceImplInterface
{
public:
    ////////////////////////////////////////
    // For selecting optimal GEMM config
    ////////////////////////////////////////
    struct LaunchConfig
    {
        GemmAllReduceImpl impl;
        MainloopScheduleType schedule;
        TileShape tile_shape;
        ClusterShape cluster_shape;

        std::string str() const
        {
            std::stringstream ss;
            ss << "LaunchConfig(";
            ss << (impl == GemmAllReduceImpl::NVLS_1SHOT ? "1shot" : "2shot");
            ss << ", Schedule_" << get_mainloop_schedule_name(schedule);
            ss << ", TileShape_" << get_tile_shape_name(tile_shape);
            ss << ", ClusterShape_" << get_cluster_shape_name(cluster_shape);
            ss << ")";
            return ss.str();
        }
    };

    ////////////////////////////////////////
    // Builder to prevent param explosion
    ////////////////////////////////////////
    struct ProblemArgs
    {
        // current iteration
        std::tuple<int, int, int, int> problem_size;
        void* A = nullptr;
        void* B = nullptr;
        void* C = nullptr;
        void* D = nullptr;
        void* D_mc = nullptr; // NVLS
        float alpha = 1.f;
        float beta = 0.f;
        PersistentWorkspaceInterface* workspace = nullptr;
        int rank;
        std::set<int> ranks;
        LaunchConfig launch_config;
        // next iteration
        void* D_next = nullptr;

        ProblemArgs()
            : launch_config({GemmAllReduceImpl::NVLS_2SHOT, MainloopScheduleType::PINGPONG,
                TileShape::TileShape_128x128x128, ClusterShape::ClusterShape_1x1x1})
        {
        }

        ProblemArgs& argA(void const* A)
        {
            this->A = const_cast<void*>(A);
            return *this;
        }

        ProblemArgs& argB(void const* B)
        {
            this->B = const_cast<void*>(B);
            return *this;
        }

        ProblemArgs& argC(void const* C)
        {
            this->C = const_cast<void*>(C);
            return *this;
        }

        ProblemArgs& argD(void* D, void* D_mc = nullptr, void* D_next = nullptr)
        {
            this->D = D;
            this->D_mc = D_mc;
            this->D_next = D_next;
            return *this;
        }

        ProblemArgs& argAlpha(float alpha)
        {
            this->alpha = alpha;
            return *this;
        }

        ProblemArgs& argBeta(float beta)
        {
            this->beta = beta;
            return *this;
        }

        ProblemArgs& argWorkspace(PersistentWorkspaceInterface* workspace)
        {
            this->workspace = workspace;
            return *this;
        }

        ProblemArgs& argProblemShape(int M, int N, int K, int L)
        {
            this->problem_size = std::make_tuple(M, N, K, L);
            return *this;
        }

        ProblemArgs& argLaunchConfig(LaunchConfig config)
        {
            this->launch_config = config;
            return *this;
        }

        ProblemArgs& argRanks(int rank, std::set<int> const& ranks)
        {
            this->rank = rank;
            this->ranks = ranks;
            return *this;
        }
    };

    virtual ~GemmAllReduceImplInterface() = default;

    virtual std::shared_ptr<PersistentWorkspaceInterface> getPersistentWorkspace(ProblemArgs const& problem) = 0;

    virtual int run(ProblemArgs const& problem, cudaStream_t stream) = 0;

    virtual std::vector<LaunchConfig> getSupportedLaunchConfigs() const
    {
        return {};
    };
};

template <typename ElementA_, typename ElementB_, typename ElementC_, typename ElementD_, typename LayoutA_,
    typename LayoutB_, typename LayoutC_, typename LayoutD_>
struct GemmTypes
{
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using ElementD = ElementD_;
    using LayoutA = LayoutA_;
    using LayoutB = LayoutB_;
    using LayoutC = LayoutC_;
    using LayoutD = LayoutD_;
};

///////////////////////////////////////////////////////////////////////////////////
// Dispatches to SM implementation with best launch configurations
///////////////////////////////////////////////////////////////////////////////////
template <typename GemmTraits>
class GemmAllReduceImplRunner : public GemmAllReduceImplInterface
{
public:
    GemmAllReduceImplRunner();

    ~GemmAllReduceImplRunner() override = default;

    std::shared_ptr<PersistentWorkspaceInterface> getPersistentWorkspace(ProblemArgs const& max_problem) override;

    int run(ProblemArgs const& problem, cudaStream_t stream) override;

    std::vector<LaunchConfig> getSupportedLaunchConfigs() const override;

private:
    ProblemArgs swapAB(ProblemArgs const& problem) const;

    using KeyType = std::tuple<GemmAllReduceImpl, MainloopScheduleType, TileShape, ClusterShape>;
    using ValueType = std::shared_ptr<GemmAllReduceImplInterface>;

    std::map<KeyType, ValueType> mGemmRegistry;
};

} // namespace tensorrt_llm::kernels::cutlass_kernels
