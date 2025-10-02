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

#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__

#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/assert.h"

#ifdef __GNUC__ // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif          // __GNUC

#include <cuda_runtime_api.h>
#include <set>
#include <vector>

using namespace tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

struct TileShape
{
    int m;
    int n;
};

TileShape get_cta_shape_for_config(CutlassTileConfig tile_config)
{
    switch (tile_config)
    {
    case CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64: return TileShape{16, 128};
    case CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64: return TileShape{16, 256};
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64: return TileShape{32, 128};
    case CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64: return TileShape{64, 64};
    case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64: return TileShape{64, 128};
    case CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64: return TileShape{128, 64};
    case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64: return TileShape{128, 128};
    case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64: return TileShape{128, 256};
    case CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64: return TileShape{256, 128};
    case CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128: return TileShape{16, 256};
    default: TLLM_THROW("[get_grid_shape_for_config] Invalid config");
    }
}

bool is_valid_split_k_factor(int64_t const m, int64_t const n, int64_t const k, TileShape const tile_shape,
    int const split_k_factor, size_t const workspace_bytes, bool const is_weight_only)
{

    // All tile sizes have a k_tile of 64.
    static constexpr int k_tile = 64;

    // For weight-only quant, we need k and k_elements_per_split to be a multiple of cta_k
    if (is_weight_only)
    {
        if ((k % k_tile) != 0)
        {
            return false;
        }

        if ((k % split_k_factor) != 0)
        {
            return false;
        }

        int const k_elements_per_split = k / split_k_factor;
        if ((k_elements_per_split % k_tile) != 0)
        {
            return false;
        }
    }

    // Check that the workspace has sufficient space for this split-k factor
    int const ctas_in_m_dim = (m + tile_shape.m - 1) / tile_shape.m;
    int const ctas_in_n_dim = (n + tile_shape.n - 1) / tile_shape.n;
    int const required_ws_bytes = split_k_factor == 1 ? 0 : sizeof(int) * ctas_in_m_dim * ctas_in_n_dim;

    if (required_ws_bytes > workspace_bytes)
    {
        return false;
    }

    return true;
}

std::vector<CutlassTileConfig> get_candidate_tiles(
    int const sm, CutlassGemmConfig::CandidateConfigTypeParam const config_type_param)
{
    enum class CutlassGemmType : char
    {
        Default,
        WeightOnly,
        Simt,
        Int8,
        Fp8
    };

    CutlassGemmType gemm_type = CutlassGemmType::Default;
    if (config_type_param & CutlassGemmConfig::SIMT_ONLY)
    {
        gemm_type = CutlassGemmType::Simt;
    }
    else if (config_type_param & CutlassGemmConfig::WEIGHT_ONLY)
    {
        gemm_type = CutlassGemmType::WeightOnly;
    }
    else if (config_type_param & CutlassGemmConfig::INT8_ONLY)
    {
        gemm_type = CutlassGemmType::Int8;
    }
    else if (config_type_param & CutlassGemmConfig::FP8_ONLY)
    {
        gemm_type = CutlassGemmType::Fp8;
    }

    std::vector<CutlassTileConfig> base_configs{
        CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64, CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64};
    if (sm >= 75)
    {
        base_configs.push_back(CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64);
    }

    switch (gemm_type)
    {
    case CutlassGemmType::Simt: return {CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8};
    case CutlassGemmType::WeightOnly:
        if (sm >= 75)
        {
            return {CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64,
                CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64,
                CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64};
        }
        else
        {
            return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64};
        }
    case CutlassGemmType::Int8:
        return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
            CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
            CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
            CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64,
            CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64,
            CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64};
    case CutlassGemmType::Fp8:
        if (config_type_param & CutlassGemmConfig::GROUPED_GEMM)
        {
            if (sm == 89 || sm >= 120)
            {
                return {CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128,
                    CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                    CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                    CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64,
                    CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
                    CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64,
                    CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64};
            }
            else
            {
                // no valid ampere style fp8 configs for sm90
                return {};
            }
        }
        else
        {
            if (sm == 89 || sm >= 120)
            {
                return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                    CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64,
                    CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64,
                    CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                    CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
                    CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64,
                    CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64,
                    CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64,
                    CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64,
                    CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64,
                    CutlassTileConfig::CtaShape128x64x128_WarpShape64x32x128,
                    CutlassTileConfig::CtaShape16x256x128_WarpShape16x64x128};
            }
            else
            {
                return {};
            }
        }
    default: return base_configs;
    }
}

std::vector<CutlassTileConfigSM90> get_candidate_tiles_sm90(CutlassGemmConfig::CandidateConfigTypeParam const config)
{
#ifdef FAST_BUILD
    // Fast build disables all configs except this one for SM90
    return {CutlassTileConfigSM90::CtaShape128x128x128B};
#else
    if (config & CutlassGemmConfig::GROUPED_GEMM)
    {
        if (config & CutlassGemmConfig::WEIGHT_ONLY)
        {
            return {CutlassTileConfigSM90::CtaShape64x16x128B, CutlassTileConfigSM90::CtaShape64x32x128B,
                CutlassTileConfigSM90::CtaShape64x64x128B, CutlassTileConfigSM90::CtaShape64x128x128B,
                CutlassTileConfigSM90::CtaShape128x16x128B, CutlassTileConfigSM90::CtaShape128x32x128B,
                CutlassTileConfigSM90::CtaShape128x64x128B, CutlassTileConfigSM90::CtaShape128x128x128B};
        }
        else
        {
            return {CutlassTileConfigSM90::CtaShape128x16x128B, CutlassTileConfigSM90::CtaShape128x32x128B,
                CutlassTileConfigSM90::CtaShape128x64x128B, CutlassTileConfigSM90::CtaShape128x128x128B,
                CutlassTileConfigSM90::CtaShape128x256x128B, CutlassTileConfigSM90::CtaShape256x128x128B};
        }
    }
    else
    {
        return {CutlassTileConfigSM90::CtaShape64x16x128B, CutlassTileConfigSM90::CtaShape64x32x128B,
            CutlassTileConfigSM90::CtaShape64x64x128B, CutlassTileConfigSM90::CtaShape64x128x128B,
            CutlassTileConfigSM90::CtaShape64x256x128B, CutlassTileConfigSM90::CtaShape128x16x128B,
            CutlassTileConfigSM90::CtaShape128x32x128B, CutlassTileConfigSM90::CtaShape128x64x128B,
            CutlassTileConfigSM90::CtaShape128x128x128B, CutlassTileConfigSM90::CtaShape128x256x128B};
    }
#endif
}

bool sm90_supports_coop(CutlassTileConfigSM90 const tile)
{
#ifdef FAST_BUILD
    return false;
#else
    std::set<CutlassTileConfigSM90> valid_tiles{CutlassTileConfigSM90::CtaShape128x16x128B,
        CutlassTileConfigSM90::CtaShape128x32x128B, CutlassTileConfigSM90::CtaShape128x64x128B,
        CutlassTileConfigSM90::CtaShape128x128x128B, CutlassTileConfigSM90::CtaShape128x256x128B,
        CutlassTileConfigSM90::CtaShape256x128x128B, CutlassTileConfigSM90::CtaShape256x256x128B};
    return valid_tiles.count(tile) == 1;
#endif
}

// We only compile CUTLASS kernels with multi-cast along M if the M tile is >= 128. This is purely to improve
// compilation speed.
bool sm90_supports_mcast_along_m(CutlassTileConfigSM90 const tile)
{
#ifdef FAST_BUILD
    return false;
#else
    std::set<CutlassTileConfigSM90> valid_tiles{CutlassTileConfigSM90::CtaShape128x16x128B,
        CutlassTileConfigSM90::CtaShape128x32x128B, CutlassTileConfigSM90::CtaShape128x64x128B,
        CutlassTileConfigSM90::CtaShape128x128x128B, CutlassTileConfigSM90::CtaShape128x256x128B,
        CutlassTileConfigSM90::CtaShape256x128x128B};
    return valid_tiles.count(tile) == 1;
#endif
}

// We only compile CUTLASS kernels with multi-cast along N if the N tile is >= 128. This is purely to improve
// compilation speed.
bool sm90_supports_mcast_along_n(CutlassTileConfigSM90 const tile)
{
#ifdef FAST_BUILD
    return false;
#else
    std::set<CutlassTileConfigSM90> valid_tiles{CutlassTileConfigSM90::CtaShape64x128x128B,
        CutlassTileConfigSM90::CtaShape64x256x128B, CutlassTileConfigSM90::CtaShape128x128x128B,
        CutlassTileConfigSM90::CtaShape128x256x128B, CutlassTileConfigSM90::CtaShape256x128x128B};
    return valid_tiles.count(tile) == 1;
#endif
}

std::vector<CutlassGemmConfig> get_candidate_configs_sm90(CutlassGemmConfig::CandidateConfigTypeParam const config)
{
    auto tiles = get_candidate_tiles_sm90(config);
    std::vector<CutlassGemmConfig> candidate_configs;
    for (auto const& tile_config : tiles)
    {
        bool const has_m_mcast = sm90_supports_mcast_along_m(tile_config);
        bool const has_n_mcast = sm90_supports_mcast_along_n(tile_config);
        bool const has_w4afp8 = (config & CutlassGemmConfig::WEIGHT_ONLY) && (config & CutlassGemmConfig::GROUPED_GEMM);
        if (has_w4afp8)
        {
            bool const has_coop_supported = sm90_supports_coop(tile_config);
            std::set<MainloopScheduleType> mainloop_schedules{MainloopScheduleType::PINGPONG};
            if (has_coop_supported)
            {
                mainloop_schedules.insert(MainloopScheduleType::COOPERATIVE);
            }
            auto const epilogue_schedule = EpilogueScheduleType::AUTO;
            for (auto const& mainloop_schedule : mainloop_schedules)
            {
                CutlassGemmConfig candidate(
                    tile_config, mainloop_schedule, epilogue_schedule, ClusterShape::ClusterShape_1x1x1);
                candidate_configs.push_back(candidate);
                candidate = CutlassGemmConfig(
                    tile_config, mainloop_schedule, epilogue_schedule, ClusterShape::ClusterShape_2x1x1);
                candidate_configs.push_back(candidate);
                candidate = CutlassGemmConfig(
                    tile_config, mainloop_schedule, epilogue_schedule, ClusterShape::ClusterShape_1x2x1);
                candidate_configs.push_back(candidate);
                candidate = CutlassGemmConfig(
                    tile_config, mainloop_schedule, epilogue_schedule, ClusterShape::ClusterShape_2x2x1);
                candidate_configs.push_back(candidate);
            }
        }
        else
        {
            CutlassGemmConfig candidate(
                tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1);
            candidate_configs.push_back(candidate);
            if (has_m_mcast)
            {
                CutlassGemmConfig candidate(tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                    ClusterShape::ClusterShape_2x1x1);
                candidate_configs.push_back(candidate);
            }

            if (has_n_mcast)
            {
                CutlassGemmConfig candidate(tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                    ClusterShape::ClusterShape_1x2x1);
                candidate_configs.push_back(candidate);
            }

            if (has_m_mcast && has_n_mcast)
            {
                CutlassGemmConfig candidate(tile_config, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
                    ClusterShape::ClusterShape_2x2x1);
                candidate_configs.push_back(candidate);
            }
        }
    }
    // add cuda kernel profiler to tactics for weight-only plugins
    if (config & CutlassGemmConfig::WEIGHT_ONLY)
    {
        if (tiles.size() > 0 && !(config & CutlassGemmConfig::GROUPED_GEMM))
        {
            CutlassGemmConfig CudaKernelConfig(
                tiles[0], MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1);
            CudaKernelConfig.enableCudaKernel = true;
            candidate_configs.push_back(CudaKernelConfig);
        }
    }
    return candidate_configs;
}

std::vector<CutlassGemmConfig> get_candidate_configs_sm100_dynamic_cluster_shape(int sm,
    CutlassGemmConfig::CandidateConfigTypeParam const config, EpilogueScheduleType schedule,
    ClusterShape const dynamic_cluster_shape, ClusterShape const fallback_cluster_shape)
{
    auto cluster1sm = ClusterShape::ClusterShape_1x1x1;
    auto cluster2sm = ClusterShape::ClusterShape_2x1x1;
    bool supports_2sm = dynamic_cluster_shape == ClusterShape::Undefined
        || std::get<0>(enum_to_shape_tuple(dynamic_cluster_shape)) % 2 == 0;

    std::vector<CutlassGemmConfig> candidate_configs;
    if ((config & CutlassGemmConfig::FP4_ONLY) != 0)
    {
        if (sm == 100)
        {
            if (schedule != EpilogueScheduleType::TMA)
                return {};
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM100::CtaShape128x64x128B,
                MainloopScheduleType::AUTO, schedule, cluster1sm, dynamic_cluster_shape, fallback_cluster_shape, sm});
            if (supports_2sm)
            {
                candidate_configs.push_back(
                    CutlassGemmConfig{CutlassTileConfigSM100::CtaShape128x64x128B, MainloopScheduleType::AUTO, schedule,
                        cluster2sm, dynamic_cluster_shape, fallback_cluster_shape, sm});
            }
        }

        candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM100::CtaShape128x128x128B,
            MainloopScheduleType::AUTO, schedule, cluster1sm, dynamic_cluster_shape, fallback_cluster_shape, sm});
        candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM100::CtaShape128x256x128B,
            MainloopScheduleType::AUTO, schedule, cluster1sm, dynamic_cluster_shape, fallback_cluster_shape, sm});
        if (supports_2sm)
        {
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM100::CtaShape128x128x128B,
                MainloopScheduleType::AUTO, schedule, cluster2sm, dynamic_cluster_shape, fallback_cluster_shape, sm});
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM100::CtaShape128x256x128B,
                MainloopScheduleType::AUTO, schedule, cluster2sm, dynamic_cluster_shape, fallback_cluster_shape, sm});
        }
        return candidate_configs;
    }

    std::vector<std::pair<CutlassTileConfigSM100, ClusterShape>> tile_configs{
        {CutlassTileConfigSM100::CtaShape128x128x128B, cluster1sm},
        {CutlassTileConfigSM100::CtaShape128x256x128B, cluster1sm},
        {CutlassTileConfigSM100::CtaShape128x32x128B, cluster1sm},
        {CutlassTileConfigSM100::CtaShape64x64x128B, cluster1sm},
        {CutlassTileConfigSM100::CtaShape64x32x128B, cluster1sm},
        {CutlassTileConfigSM100::CtaShape64x128x128B, cluster1sm},
        {CutlassTileConfigSM100::CtaShape64x256x128B, cluster1sm},
        {CutlassTileConfigSM100::CtaShape128x64x128B, cluster1sm},
    };

    if (supports_2sm)
    {
        tile_configs.push_back({CutlassTileConfigSM100::CtaShape64x128x128B, cluster2sm});
        tile_configs.push_back({CutlassTileConfigSM100::CtaShape64x256x128B, cluster2sm});
        tile_configs.push_back({CutlassTileConfigSM100::CtaShape64x64x128B, cluster2sm});
        tile_configs.push_back({CutlassTileConfigSM100::CtaShape128x64x128B, cluster2sm});
        tile_configs.push_back({CutlassTileConfigSM100::CtaShape128x128x128B, cluster2sm});
        tile_configs.push_back({CutlassTileConfigSM100::CtaShape128x256x128B, cluster2sm});
    }

    if (config & CutlassGemmConfig::FP8_ONLY)
    {
        tile_configs.push_back({CutlassTileConfigSM100::CtaShape128x16x128B, cluster1sm});
        // TODO: re-enable when handled by the MoE GEMM dispatch
        // tile_configs.push_back({ CutlassTileConfigSM100::CtaShape128x8x256B, ClusterShape::ClusterShape_1x1x1 });
    }

    for (auto [tile, cluster] : tile_configs)
    {
        CutlassGemmConfig config{
            tile, MainloopScheduleType::AUTO, schedule, cluster, dynamic_cluster_shape, fallback_cluster_shape, sm};
        candidate_configs.push_back(config);
    }
    return candidate_configs;
}

std::vector<CutlassGemmConfig> get_candidate_configs_sm100(
    CutlassGemmConfig::CandidateConfigTypeParam const config, int sm)
{
#ifdef FAST_BUILD
    // Fast build disables all configs except this one for SM100
    return {CutlassGemmConfig{CutlassTileConfigSM100::CtaShape128x128x128B, MainloopScheduleType::AUTO,
        EpilogueScheduleType::TMA, ClusterShape::ClusterShape_1x1x1, ClusterShape::Undefined, ClusterShape::Undefined,
        sm}};
#else
    if (config & CutlassGemmConfig::GROUPED_GEMM)
    {
        std::vector<CutlassGemmConfig> candidate_configs;
        for (auto schedule : {EpilogueScheduleType::TMA, EpilogueScheduleType::NO_SMEM})
        {
            // TODO The tactic profiling is a bit long with all of these shapes enabled
            //   Shape 4x4x1 shapes do not seem to give better performance in the cases I tested so we disable it here
            auto cluster_shapes = {ClusterShape::ClusterShape_1x1x1, ClusterShape::ClusterShape_4x1x1,
                ClusterShape::ClusterShape_4x2x1 /*, ClusterShape::ClusterShape_4x4x1*/};
            for (auto cluster_shape : cluster_shapes)
            {
                auto fallback_cluster_shape = cluster_shape == ClusterShape::ClusterShape_1x1x1
                    ? ClusterShape::ClusterShape_1x1x1
                    : ClusterShape::ClusterShape_2x1x1;
                auto configs = get_candidate_configs_sm100_dynamic_cluster_shape(
                    sm, config, schedule, cluster_shape, fallback_cluster_shape);
                candidate_configs.insert(candidate_configs.end(), configs.begin(), configs.end());
            }

            auto configs = get_candidate_configs_sm100_dynamic_cluster_shape(
                sm, config, schedule, ClusterShape::Undefined, ClusterShape::Undefined);
            candidate_configs.insert(candidate_configs.end(), configs.begin(), configs.end());
        }
        return candidate_configs;
    }
    else
    {
        TLLM_THROW("Not Implemented: SM100 GEMM candidates have not been defined.");
    }
#endif
}

std::vector<CutlassGemmConfig> get_candidate_configs_sm120(CutlassGemmConfig::CandidateConfigTypeParam const config)
{
#ifdef FAST_BUILD
    // Fast build disables all configs except this
    if (config & CutlassGemmConfig::GROUPED_GEMM)
    {
        return {CutlassGemmConfig{CutlassTileConfigSM120::CtaShape128x128x128B, MainloopScheduleType::AUTO,
            EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1}};
    }
    else
    {
        return {CutlassGemmConfig{CutlassTileConfigSM120::CtaShape128x128x256B, MainloopScheduleType::AUTO,
            EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1}};
    }
#else
    if (config & CutlassGemmConfig::GROUPED_GEMM)
    {
        std::vector<CutlassGemmConfig> candidate_configs;
        if (config & CutlassGemmConfig::FP8FP4_MIXED)
        {
            // Mixed FP8 x FP4: restrict to 128x128x128B only
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM120::CtaShape128x128x128B,
                MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1});
            return candidate_configs;
        }
        else if (config & CutlassGemmConfig::FP4_ONLY)
        {
            // FP4 x FP4: allow all four tiles
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM120::CtaShape128x128x128B,
                MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1});
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM120::CtaShape128x128x64B,
                MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1});
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM120::CtaShape128x256x64B,
                MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1});
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM120::CtaShape256x128x64B,
                MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1});
            return candidate_configs;
        }
        TLLM_THROW("Not Implemented: SM120 group GEMM only supports mxfp8-mxfp4 mixed or nvfp4.");
    }
    else
    {
        std::vector<CutlassGemmConfig> candidate_configs;
        if ((config & CutlassGemmConfig::FP4_ONLY) != 0)
        {
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM120::CtaShape128x128x256B,
                MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1});
            candidate_configs.push_back(CutlassGemmConfig{CutlassTileConfigSM120::CtaShape256x128x128B,
                MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO, ClusterShape::ClusterShape_1x1x1});
            return candidate_configs;
        }
        else
        {
            TLLM_THROW("Not Implemented: SM120 GEMM only supports nvfp4.");
        }
    }
#endif

} // namespace kernels

std::vector<CutlassGemmConfig> get_candidate_configs(
    int sm, int const max_split_k, CutlassGemmConfig::CandidateConfigTypeParam const config_type_param)
{
    if ((config_type_param & CutlassGemmConfig::FP4_ONLY) && !(config_type_param & CutlassGemmConfig::BLACKWELL))
    {
        // FP4 is only supported on blackwell
        return {};
    }

    if (sm == 90 && (config_type_param & CutlassGemmConfig::HOPPER))
    {
        return get_candidate_configs_sm90(config_type_param);
    }
    if (sm >= 100 && sm < 120 && (config_type_param & CutlassGemmConfig::BLACKWELL))
    {
        return get_candidate_configs_sm100(config_type_param, sm);
    }
    if (sm >= 120 && (config_type_param & CutlassGemmConfig::BLACKWELL))
    {
        return get_candidate_configs_sm120(config_type_param);
    }

    std::vector<CutlassTileConfig> tiles = get_candidate_tiles(sm, config_type_param);

    std::vector<CutlassGemmConfig> candidate_configs;

    bool const int8_configs_only = config_type_param & CutlassGemmConfig::INT8_ONLY;
    int const min_stages = (sm == 89) ? 3 : int8_configs_only ? 3 : 2;
    int const max_stages = int8_configs_only ? 6 : (sm >= 80 ? 4 : 2);
    for (auto const& tile_config : tiles)
    {
        for (int stages = min_stages; stages <= max_stages; ++stages)
        {
            CutlassGemmConfig config(tile_config, SplitKStyle::NO_SPLIT_K, 1, stages);
            candidate_configs.push_back(config);
            if (sm >= 75)
            {
                for (int split_k_factor = 2; split_k_factor <= max_split_k; ++split_k_factor)
                {
                    auto config = CutlassGemmConfig{tile_config, SplitKStyle::SPLIT_K_SERIAL, split_k_factor, stages};
                    candidate_configs.push_back(config);
                }
            }
        }
    }
    // add cuda kernel profiler to tactics for weight-only plugins
    if (config_type_param & CutlassGemmConfig::WEIGHT_ONLY)
    {
        if (tiles.size() > 0)
        {
            CutlassGemmConfig CudaKernelConfig(tiles[0], SplitKStyle::NO_SPLIT_K, 1, min_stages);
            CudaKernelConfig.enableCudaKernel = true;
            candidate_configs.push_back(CudaKernelConfig);
        }
    }
    return candidate_configs;
}

CutlassGemmConfig estimate_best_config_from_occupancies(std::vector<CutlassGemmConfig> const& candidate_configs,
    std::vector<int> const& occupancies, int64_t const m, int64_t const n, int64_t const k, int64_t const num_experts,
    int const split_k_limit, size_t const workspace_bytes, int const multi_processor_count, int const is_weight_only)
{

    if (occupancies.size() != candidate_configs.size())
    {
        TLLM_THROW(
            "[estimate_best_config_from_occupancies] occpancies and "
            "candidate configs vectors must have equal length.");
    }

    CutlassGemmConfig best_config;
    // Score will be [0, 1]. The objective is to minimize this score.
    // It represents the fraction of SM resources unused in the last wave.
    float config_score = 1.0f;
    int config_waves = INT_MAX;
    int current_m_tile = 0;

    int const max_split_k = n >= multi_processor_count * 256 ? 1 : split_k_limit;
    for (int ii = 0; ii < candidate_configs.size(); ++ii)
    {
        CutlassGemmConfig candidate_config = candidate_configs[ii];
        TileShape tile_shape = get_cta_shape_for_config(candidate_config.tile_config_sm80);
        int occupancy = occupancies[ii];

        if (occupancy == 0)
        {
            continue;
        }

        // Keep small tile sizes when possible.
        if (best_config.tile_config_sm80 != CutlassTileConfig::ChooseWithHeuristic && m < current_m_tile
            && current_m_tile < tile_shape.m)
        {
            continue;
        }

        int const ctas_in_m_dim = (m + tile_shape.m - 1) / tile_shape.m;
        int const ctas_in_n_dim = (n + tile_shape.n - 1) / tile_shape.n;

        for (int split_k_factor = 1; split_k_factor <= max_split_k; ++split_k_factor)
        {
            if (is_valid_split_k_factor(m, n, k, tile_shape, split_k_factor, workspace_bytes, is_weight_only))
            {
                int const ctas_per_wave = occupancy * multi_processor_count;
                int const ctas_for_problem = ctas_in_m_dim * ctas_in_n_dim * split_k_factor;

                int const num_waves_total = (ctas_for_problem + ctas_per_wave - 1) / ctas_per_wave;
                float const num_waves_fractional = ctas_for_problem / float(ctas_per_wave);
                float const current_score = float(num_waves_total) - num_waves_fractional;

                float const score_slack = 0.1f;
                if (current_score < config_score
                    || ((config_waves > num_waves_total) && (current_score < config_score + score_slack)))
                {
                    config_score = current_score;
                    config_waves = num_waves_total;
                    SplitKStyle split_style
                        = split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K;
                    best_config = CutlassGemmConfig(
                        candidate_config.tile_config_sm80, split_style, split_k_factor, candidate_config.stages);
                    current_m_tile = tile_shape.m;
                }
                else if (current_score == config_score
                    && (best_config.stages < candidate_config.stages || split_k_factor < best_config.split_k_factor
                        || current_m_tile < tile_shape.m))
                {
                    // Prefer deeper pipeline or smaller split-k
                    SplitKStyle split_style
                        = split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K;
                    best_config = CutlassGemmConfig(
                        candidate_config.tile_config_sm80, split_style, split_k_factor, candidate_config.stages);
                    current_m_tile = tile_shape.m;
                    config_waves = num_waves_total;
                }
            }
        }
    }

    if (best_config.tile_config_sm80 == CutlassTileConfig::ChooseWithHeuristic)
    {
        TLLM_THROW("Heuristic failed to find a valid config.");
    }

    return best_config;
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
