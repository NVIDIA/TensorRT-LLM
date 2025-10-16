/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

#include "cute/tensor.hpp"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/tllmException.h"

namespace tensorrt_llm
{
namespace cutlass_extensions
{

// Note: The shapes are in the format MxNxK. The K shape of the runtime config MUST match the K shape
//       in the kernel layout details when doing weight only quantization.
enum class CutlassTileConfig
{
    // Signals that we should run heuristics do choose a config
    Undefined = 0,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic = 1,

    // SiMT config
    CtaShape128x128x8_WarpShape64x64x8,

    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=16
    CtaShape16x128x64_WarpShape16x32x64,
    // Warp configs for M=32
    CtaShape32x128x64_WarpShape32x32x64,

    // Warp configs for M=64
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x64x128_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,

    // Warp configs for M=128
    CtaShape128x64x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x64x64,
    CtaShape128x128x64_WarpShape128x32x64,
    CtaShape128x256x64_WarpShape64x64x64,

    // Warp configs for M=256
    CtaShape256x128x64_WarpShape64x64x64,

    // TensorCore config CTA_N = 64, CTA_K = 128
    CtaShape128x64x128_WarpShape64x32x128,

    // TensorCore config CTA_N = 256, CTA_K = 64
    CtaShape16x256x64_WarpShape16x64x64,

    // TensorCore config CTA_N = 256, CTA_K = 128
    CtaShape16x256x128_WarpShape16x64x128

};

enum class SplitKStyle
{
    NO_SPLIT_K,
    SPLIT_K_SERIAL,
    STREAM_K, // Sm80+
    // SPLIT_K_PARALLEL // Not supported yet
};

constexpr static int shape_tuple_to_enum(int m, int n, int k)
{
    assert(m >= 0 && n >= 0 && k >= 0);
    assert(m < 1000 && n < 1000 && k < 1000);
    return m * 1000000 + n * 1000 + k;
}

template <typename TEnum>
constexpr static std::tuple<int, int, int> enum_to_shape_tuple(TEnum shape_id_enum)
{
    static_assert(std::is_enum_v<TEnum> && std::is_same_v<std::underlying_type_t<TEnum>, int>,
        "TEnum must be an enum with underlying type int");
    auto shape_id = static_cast<int>(shape_id_enum);
    assert(shape_id >= 0);
    assert(shape_id < (int) 1e9);
    return std::make_tuple(shape_id / 1000000, (shape_id % 1000000) / 1000, shape_id % 1000);
}

enum class CutlassTileConfigSM90 : int
{
    // Signals that we should run heuristics do choose a config
    Undefined = 0,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic = 1,

    // CTA configs for M=64
    CtaShape64x16x128B = shape_tuple_to_enum(64, 16, 128),
    CtaShape64x32x128B = shape_tuple_to_enum(64, 32, 128),
    CtaShape64x64x128B = shape_tuple_to_enum(64, 64, 128),
    CtaShape64x128x128B = shape_tuple_to_enum(64, 128, 128),
    CtaShape64x256x128B = shape_tuple_to_enum(64, 256, 128),

    // CTA configs for M=128
    CtaShape128x16x128B = shape_tuple_to_enum(128, 16, 128),
    CtaShape128x32x128B = shape_tuple_to_enum(128, 32, 128),
    CtaShape128x64x128B = shape_tuple_to_enum(128, 64, 128),
    CtaShape128x128x128B = shape_tuple_to_enum(128, 128, 128),
    CtaShape128x256x128B = shape_tuple_to_enum(128, 256, 128),

    // CTA configs for M=256
    CtaShape256x128x128B = shape_tuple_to_enum(256, 128, 128),
    CtaShape256x256x128B = shape_tuple_to_enum(256, 256, 128),
};

enum class CutlassTileConfigSM100 : int
{
    // Signals that we should run heuristics do choose a config
    Undefined = 0,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic = 1,

    /*
     * Grouped GEMM
     */
    // M=64
    CtaShape64x32x128B = shape_tuple_to_enum(64, 32, 128),
    CtaShape64x64x128B = shape_tuple_to_enum(64, 64, 128),
    CtaShape64x128x128B = shape_tuple_to_enum(64, 128, 128),
    CtaShape64x256x128B = shape_tuple_to_enum(64, 256, 128),

    // M=128
    CtaShape128x8x256B = shape_tuple_to_enum(128, 8, 256),
    CtaShape128x16x128B = shape_tuple_to_enum(128, 16, 128),
    CtaShape128x32x128B = shape_tuple_to_enum(128, 32, 128),
    CtaShape128x64x128B = shape_tuple_to_enum(128, 64, 128),
    CtaShape128x128x128B = shape_tuple_to_enum(128, 128, 128),
    CtaShape128x256x128B = shape_tuple_to_enum(128, 256, 128),
    CtaShape128x128x256B = shape_tuple_to_enum(128, 128, 256),
    CtaShape128x256x256B = shape_tuple_to_enum(128, 256, 256),
};

// An alias to make the SHAPE_CASE macro work
using CutlassTileConfigSM103 = CutlassTileConfigSM100;

enum class CutlassTileConfigSM120 : int
{
    // Signals that we should run heuristics do choose a config
    Undefined = 0,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic = 1,

    CtaShape128x128x128B = shape_tuple_to_enum(128, 128, 128),
    CtaShape128x128x64B = shape_tuple_to_enum(128, 128, 64),
    CtaShape256x128x64B = shape_tuple_to_enum(256, 128, 64),
    CtaShape128x256x64B = shape_tuple_to_enum(128, 256, 64),
    CtaShape128x128x256B = shape_tuple_to_enum(128, 128, 256),
    CtaShape256x128x128B = shape_tuple_to_enum(256, 128, 128),
};

enum class MainloopScheduleType
{
    AUTO, // Automatically selects between pingpong and cooperative schedules on Hopper. On older architectures, this
          // defaults to the "legacy" main loop schedule.
    PINGPONG,
    COOPERATIVE,
    WARPSPECIALIZED
};

static auto get_mainloop_schedule_name(MainloopScheduleType schedule)
{
    if (schedule == MainloopScheduleType::AUTO)
    {
        return "auto";
    }
    else if (schedule == MainloopScheduleType::PINGPONG)
    {
        return "pingpong";
    }
    else if (schedule == MainloopScheduleType::COOPERATIVE)
    {
        return "cooperative";
    }
    else if (schedule == MainloopScheduleType::WARPSPECIALIZED)
    {
        return "warpspecialized";
    }
    return "unknown schedule";
}

enum class EpilogueScheduleType
{
    AUTO, // Automatically chooses an epilogue schedule compatible with the selected main loop schedule for Hopper. For
          // architectures older than hopper, the epilogue is always performed by the same thread block as the main
          // loop.
    NO_SMEM,
    TMA
};

enum class TileShape : int
{
    Undefined = 0,
    TileShape_64x16x128 = shape_tuple_to_enum(64, 16, 128),
    TileShape_64x32x128 = shape_tuple_to_enum(64, 32, 128),
    TileShape_64x64x128 = shape_tuple_to_enum(64, 64, 128),
    TileShape_64x128x128 = shape_tuple_to_enum(64, 128, 128),
    TileShape_64x256x128 = shape_tuple_to_enum(64, 256, 128),
    TileShape_64x512x128 = shape_tuple_to_enum(64, 512, 128),
    TileShape_128x16x128 = shape_tuple_to_enum(128, 16, 128),
    TileShape_128x32x128 = shape_tuple_to_enum(128, 32, 128),
    TileShape_128x64x128 = shape_tuple_to_enum(128, 64, 128),
    TileShape_128x128x128 = shape_tuple_to_enum(128, 128, 128),
    TileShape_128x256x128 = shape_tuple_to_enum(128, 256, 128),
    TileShape_256x128x128 = shape_tuple_to_enum(256, 128, 128),
    TileShape_256x256x128 = shape_tuple_to_enum(256, 256, 128)
};

template <TileShape Shape_MNK>
constexpr auto get_tile_shape()
{
    using namespace cute;
    static_assert(Shape_MNK != TileShape::Undefined, "TileShape is undefined");

    constexpr auto shape_tuple = enum_to_shape_tuple(Shape_MNK);
    return cute::Shape<cute::Int<std::get<0>(shape_tuple)>, cute::Int<std::get<1>(shape_tuple)>,
        cute::Int<std::get<2>(shape_tuple)>>{};
}

template <class TEnum>
static std::string get_tile_shape_name(TEnum Shape_MNK)
{
    static_assert(std::is_enum_v<TEnum> && std::is_same_v<std::underlying_type_t<TEnum>, int>,
        "TEnum must be an enum with underlying type int");
    if ((int) Shape_MNK == 0)
    {
        return "undefined";
    }
    else if ((int) Shape_MNK == 1)
    {
        return "heuristic";
    }
    else
    {
        auto [m, n, k] = enum_to_shape_tuple(Shape_MNK);
        return std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    }
}

enum class ClusterShape : int
{
    Undefined = 0,
    ClusterShape_1x1x1 = shape_tuple_to_enum(1, 1, 1),
    ClusterShape_2x1x1 = shape_tuple_to_enum(2, 1, 1),
    ClusterShape_1x2x1 = shape_tuple_to_enum(1, 2, 1),
    ClusterShape_2x2x1 = shape_tuple_to_enum(2, 2, 1),
    ClusterShape_1x4x1 = shape_tuple_to_enum(1, 4, 1),
    ClusterShape_4x1x1 = shape_tuple_to_enum(4, 1, 1),
    ClusterShape_4x2x1 = shape_tuple_to_enum(4, 2, 1),
    ClusterShape_2x4x1 = shape_tuple_to_enum(2, 4, 1),
    ClusterShape_4x4x1 = shape_tuple_to_enum(4, 4, 1),
    ClusterShape_1x8x1 = shape_tuple_to_enum(1, 8, 1),
    ClusterShape_8x1x1 = shape_tuple_to_enum(8, 1, 1)
};

static std::string get_cluster_shape_name(ClusterShape Shape_MNK)
{
    if (Shape_MNK == ClusterShape::Undefined)
    {
        return "undefined";
    }
    else
    {
        auto [m, n, k] = enum_to_shape_tuple(Shape_MNK);
        return std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    }
}

template <ClusterShape Shape_MNK>
constexpr auto get_cluster_shape()
{
    using namespace cute;
    if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x1x1)
    {
        return cute::Shape<_1, _1, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_2x1x1)
    {
        return cute::Shape<_2, _1, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x2x1)
    {
        return cute::Shape<_1, _2, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_2x2x1)
    {
        return cute::Shape<_2, _2, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_4x1x1)
    {
        return cute::Shape<_4, _1, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x8x1)
    {
        return cute::Shape<_1, _8, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_8x1x1)
    {
        return cute::Shape<_8, _1, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_1x4x1)
    {
        return cute::Shape<_1, _4, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_4x2x1)
    {
        return cute::Shape<_4, _2, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_2x4x1)
    {
        return cute::Shape<_2, _4, _1>{};
    }
    else if constexpr (Shape_MNK == ClusterShape::ClusterShape_4x4x1)
    {
        return cute::Shape<_4, _4, _1>{};
    }
    else
    {
        return cute::Shape<_0, _0, _0>{};
    }
}

struct CutlassGemmConfig
{
    enum CandidateConfigTypeParam : int
    {
        NONE = 0,
        WEIGHT_ONLY = 1u << 0,
        SIMT_ONLY = 1u << 1,
        INT8_ONLY = 1u << 2,
        HOPPER = 1u << 3,
        BLACKWELL = 1u << 4,
        GROUPED_GEMM = 1u << 5,
        FP8_ONLY = 1u << 6,
        FP4_ONLY = 1u << 7,
        FP8FP4_MIXED = 1u << 8
    };

    CutlassTileConfig tile_config_sm80 = CutlassTileConfig::ChooseWithHeuristic;
    SplitKStyle split_k_style = SplitKStyle::NO_SPLIT_K;
    int split_k_factor = -1;
    int stages = -1;

    // config options for sm90
    CutlassTileConfigSM90 tile_config_sm90 = CutlassTileConfigSM90::ChooseWithHeuristic;
    CutlassTileConfigSM100 tile_config_sm100 = CutlassTileConfigSM100::ChooseWithHeuristic;
    CutlassTileConfigSM120 tile_config_sm120 = CutlassTileConfigSM120::ChooseWithHeuristic;
    MainloopScheduleType mainloop_schedule = MainloopScheduleType::AUTO;
    EpilogueScheduleType epilogue_schedule = EpilogueScheduleType::AUTO;
    ClusterShape cluster_shape = ClusterShape::ClusterShape_1x1x1;
    ClusterShape dynamic_cluster_shape = ClusterShape::Undefined;
    ClusterShape fallback_cluster_shape = ClusterShape::Undefined;
    bool enableCudaKernel = false;
    int sm_version = 80; // Use 80 as a catch all for <90
    bool is_tma_warp_specialized = false;

    enum class EpilogueFusionType : int
    {
        NONE,
        FINALIZE
    };

    EpilogueFusionType epilogue_fusion_type = EpilogueFusionType::NONE;
    bool swap_ab = false;

    CutlassGemmConfig() = default;

    CutlassGemmConfig(CutlassTileConfig tile_config, SplitKStyle split_k_style, int split_k_factor, int stages)
        : tile_config_sm80(tile_config)
        , split_k_style(split_k_style)
        , split_k_factor(split_k_factor)
        , stages(stages)
        , sm_version(80)
    {
    }

    CutlassGemmConfig(CutlassTileConfigSM90 tile_config_sm90, MainloopScheduleType mainloop_schedule,
        EpilogueScheduleType epilogue_schedule, ClusterShape cluster_shape)
        : tile_config_sm90(tile_config_sm90)
        , mainloop_schedule(mainloop_schedule)
        , epilogue_schedule(epilogue_schedule)
        , cluster_shape(cluster_shape)
        , sm_version(90)
        , is_tma_warp_specialized(true)
    {
    }

    // If dynamic_cluster_shape is provided, dynamic CGA will be enabled and cluster_shape will be interpreted as
    // whether to use 1 or 2 SM mode, otherwise static cluster shape is used.
    CutlassGemmConfig(CutlassTileConfigSM100 tile_config_sm100, MainloopScheduleType mainloop_schedule,
        EpilogueScheduleType epilogue_schedule, ClusterShape cluster_shape,
        ClusterShape dynamic_cluster_shape = ClusterShape::Undefined,
        ClusterShape fallback_cluster_shape = ClusterShape::Undefined, int sm_version = 100)
        : tile_config_sm100(tile_config_sm100)
        , mainloop_schedule(mainloop_schedule)
        , epilogue_schedule(epilogue_schedule)
        , cluster_shape(cluster_shape)
        , dynamic_cluster_shape(dynamic_cluster_shape)
        , fallback_cluster_shape(fallback_cluster_shape)
        , sm_version(sm_version)
        , is_tma_warp_specialized(true)
    {
        TLLM_CHECK_WITH_INFO(sm_version >= 100 && sm_version < 120, "Expected SM 10x version");
    }

    CutlassGemmConfig(CutlassTileConfigSM120 tile_config_sm120, MainloopScheduleType mainloop_schedule,
        EpilogueScheduleType epilogue_schedule, ClusterShape cluster_shape)
        : tile_config_sm120(tile_config_sm120)
        , mainloop_schedule(mainloop_schedule)
        , epilogue_schedule(epilogue_schedule)
        , cluster_shape(cluster_shape)
        , sm_version(120)
        , is_tma_warp_specialized(true)
    {
    }

    int getTileConfigAsInt() const
    {
        if (sm_version == 120 || sm_version == 121)
            return (int) tile_config_sm120;
        if (sm_version >= 100 && sm_version < 120)
            return (int) tile_config_sm100;
        if (sm_version == 90)
            return (int) tile_config_sm90;
        if (sm_version < 90)
            return (int) tile_config_sm80;
        assert(false && "Invalid SM version");
        return -1;
    }

    std::string getTileConfigAsName() const
    {
        if (sm_version == 120 || sm_version == 121)
            return get_tile_shape_name(tile_config_sm120);
        if (sm_version >= 100 && sm_version < 120)
            return get_tile_shape_name(tile_config_sm100);
        if (sm_version == 90)
            return get_tile_shape_name(tile_config_sm90);
        if (sm_version < 90)
            return std::to_string((int) tile_config_sm80);
        assert(false && "Invalid SM version");
        return "invalid";
    }

    std::string toString() const
    {
        std::stringstream tactic;
        tactic << "Cutlass GEMM Tactic";
        if (is_tma_warp_specialized)
        {
            assert(sm_version >= 90 && "Invalid cutlass GEMM config");
            tactic << "\n\tstyle=TMA Warp Specialized"
                   << "\n\tsm: " << sm_version << "\n\ttile shape ID: " << getTileConfigAsName()
                   << "\n\tcluster shape ID: " << get_cluster_shape_name(cluster_shape)
                   << "\n\tdynamic cluster shape ID: " << get_cluster_shape_name(dynamic_cluster_shape)
                   << "\n\tfallback cluster shape ID: " << get_cluster_shape_name(fallback_cluster_shape)
                   << "\n\tmainloop sched: " << (int) mainloop_schedule << "\n\tepi sched: " << (int) epilogue_schedule
                   << "\n\tenable cuda kernel: " << (enableCudaKernel ? "true" : "false")
                   << "\n\tepilogue fusion type: " << (int) epilogue_fusion_type
                   << "\n\tswap_ab: " << (swap_ab ? "true" : "false");
        }
        else if (tile_config_sm80 != tensorrt_llm::cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic)
        {
            assert(sm_version < 90 && "Invalid cutlass GEMM config");
            tactic << "\n\tstyle=compatible"
                   << "\n\ttile shape ID: " << (int) tile_config_sm80 << "\n\tstages: " << (int) stages
                   << "\n\tsplit k: " << (int) split_k_factor
                   << "\n\tenable cuda kernel: " << (enableCudaKernel ? "true" : "false");
        }
        else if (enableCudaKernel)
        {
            tactic << "\n\tenable cuda kernel: " << (enableCudaKernel ? "true" : "false");
        }
        else
        {
            tactic << "\n\tundefined";
        }
        tactic << "\n";
        return tactic.str();
    }
};

inline std::ostream& operator<<(std::ostream& out, CutlassGemmConfig const& config)
{
    // clang-format off
    if (config.is_tma_warp_specialized)
    {
        out << "tile_config_sm90_enum: " << config.getTileConfigAsInt()
            << ", mainloop_schedule_enum: " << int(config.mainloop_schedule)
            << ", epilogue_schedule_enum: " << int(config.epilogue_schedule)
            << ", cluster_shape_enum: " << int(config.cluster_shape)
            << ", dynamic_cluster_shape_enum: " << int(config.dynamic_cluster_shape)
            << ", fallback_cluster_shape_enum: " << int(config.fallback_cluster_shape)
            << ", enable_cuda_kernel: " << (config.enableCudaKernel ? "true" : "false")
            << ", epilogue_fusion_type: " << int(config.epilogue_fusion_type)
            << ", swap_ab: " << (config.swap_ab ? "true" : "false");
    }
    else
    {
        out << "tile_config_enum: " << config.getTileConfigAsInt()
            << ", split_k_style_enum: " << int(config.split_k_style)
            << ", split_k_factor: " << config.split_k_factor
            << ", stages: " << config.stages
            << ", enable_cuda_kernel: " << (config.enableCudaKernel ? "true" : "false");
    }
    // clang-format on
    return out;
}

} // namespace cutlass_extensions
} // namespace tensorrt_llm
