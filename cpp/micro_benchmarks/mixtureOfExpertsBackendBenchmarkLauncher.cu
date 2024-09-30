/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Include the fixture with the actual benchmark code
#include "mixtureOfExpertsBackendBenchmarkFixture.h"

/*
 * Below is all the setup for parameterising the benchmarks
 */

template <class DataType_, class WeightType_ = DataType_, class OutputType_ = DataType_>
struct WeightParams
{
    using DataType = DataType_;
    using WeightType = WeightType_;
    using OutputType = OutputType_;
};

#define BENCHMARK_BASIC(atype, wtype, otype)                                                                           \
    BENCHMARK_TEMPLATE_DEFINE_F(MixtureOfExpertsBenchmark, Basic_##atype##_##wtype, WeightParams<atype, wtype, otype>) \
    (benchmark::State & state)                                                                                         \
    {                                                                                                                  \
        runBenchmark(state);                                                                                           \
    }

#define BENCHMARK_BASIC_DO_REGISTER(atype, wtype, otype)                                                               \
    BENCHMARK_REGISTER_F(MixtureOfExpertsBenchmark, Basic_##atype##_##wtype)                                           \
        ->Apply(argGen<MixtureOfExpertsBenchmark<WeightParams<atype, wtype, otype>>>)

template <class BenchClass>
auto listAllTactics()
{
    int const sm = getSMVersion();
    using RunnerType = decltype(BenchClass::mMoERunner);
    return RunnerType::getTactics(sm);
}

template <class BenchClass>
int parseTacticToId(nlohmann::json tactic_config)
{
    bool is_sm90 = tactic_config.at("is_sm90").get<bool>();
    int tile_shape_id = -1;
    std::array<int, 3> tile_shape;
    if (tactic_config.at("tile_shape").is_array())
        tactic_config.at("tile_shape").get_to(tile_shape);
    else
        tile_shape_id = tactic_config.at("tile_shape").get<int>();

    std::vector<tensorrt_llm::cutlass_extensions::CutlassGemmConfig> confs = listAllTactics<BenchClass>();

    try
    {
        for (int i = 0; i < confs.size(); i++)
        {
            auto const& c = confs[i];
            if (c.is_sm90 != is_sm90)
                continue;

            if (!is_sm90)
            {
                int stages = tactic_config.at("stages").get<int>();
                if (c.stages != stages)
                    continue;
            }

            if (tile_shape_id != -1)
            {
                int comp = is_sm90 ? (int) c.tile_config_sm90 : (int) c.tile_config;
                if (tile_shape_id != comp)
                    continue;
                if (is_sm90 && (int) c.cluster_shape != tactic_config.at("cluster_shape").get<int>())
                    continue;

                // Found matching config
                return i;
            }

            // Handle if the user provided a shape instead of the enum value
            if (is_sm90)
            {
                using Kv = uint64_t;
                constexpr static auto K = [](int m, int n) { return (uint64_t(m) << 32) | uint64_t(n); };
                static std::unordered_map<Kv, CutlassTileConfigSM90> const tile_map{
                    {K(64, 16), CutlassTileConfigSM90::CtaShape64x16x128B},
                    {K(64, 32), CutlassTileConfigSM90::CtaShape64x32x128B},
                    {K(64, 64), CutlassTileConfigSM90::CtaShape64x64x128B},
                    {K(64, 128), CutlassTileConfigSM90::CtaShape64x128x128B},
                    {K(64, 256), CutlassTileConfigSM90::CtaShape64x256x128B},

                    {K(128, 16), CutlassTileConfigSM90::CtaShape128x16x128B},
                    {K(128, 32), CutlassTileConfigSM90::CtaShape128x32x128B},
                    {K(128, 64), CutlassTileConfigSM90::CtaShape128x64x128B},
                    {K(128, 128), CutlassTileConfigSM90::CtaShape128x128x128B},
                    {K(128, 256), CutlassTileConfigSM90::CtaShape128x256x128B},
                    {K(256, 128), CutlassTileConfigSM90::CtaShape256x128x128B},
                };

                if (c.tile_config_sm90 != tile_map.at(K(tile_shape[0], tile_shape[1])))
                    continue;

                static std::unordered_map<Kv, ClusterShape> const cluster_map{
                    // CTA configs for M=64
                    {K(1, 1), ClusterShape::ClusterShape_1x1x1},
                    {K(2, 1), ClusterShape::ClusterShape_2x1x1},
                    {K(1, 2), ClusterShape::ClusterShape_1x2x1},
                    {K(2, 2), ClusterShape::ClusterShape_2x2x1},
                };

                std::array<int, 3> cluster_shape;
                tactic_config.at("cluster_shape").get_to(cluster_shape);

                if (c.cluster_shape != cluster_map.at(K(cluster_shape[0], cluster_shape[1])))
                    continue;

                // Found matching config
                return i;
            }
            else
            {
                std::array<int, 3> warp_shape;
                tactic_config.at("warp_shape").get_to(warp_shape);

                using Kv = uint64_t;
                constexpr static auto K = [](std::array<int, 3> a, std::array<int, 3> b)
                {
                    uint64_t sum = 0;
                    for (auto v : a)
                        sum = sum * 512 + v;
                    for (auto v : b)
                        sum = sum * 256 + v;
                    return sum;
                };
                static std::unordered_map<Kv, CutlassTileConfig> tile_map{
                    {K({128, 128, 8}, {64, 64, 8}), CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8},

                    {K({16, 128, 64}, {16, 32, 64}), CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64},
                    {K({32, 128, 64}, {32, 32, 64}), CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64},

                    {K({64, 128, 64}, {32, 64, 64}), CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64},
                    {K({64, 64, 128}, {32, 64, 64}), CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64},
                    {K({64, 128, 64}, {64, 32, 64}), CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64},

                    {K({128, 64, 64}, {64, 32, 64}), CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64},
                    {K({128, 128, 64}, {64, 32, 64}), CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64},
                    {K({128, 128, 64}, {64, 64, 64}), CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64},
                    {K({128, 128, 64}, {64, 32, 64}), CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64},
                    {K({128, 256, 64}, {64, 64, 64}), CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64},

                    {K({256, 128, 64}, {64, 64, 64}), CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64},

                    {K({16, 256, 64}, {16, 64, 64}), CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64}

                };
                if (c.tile_config != tile_map.at(K(tile_shape, warp_shape)))
                    continue;

                // Found matching config
                return i;
            }
        }
    }
    catch (std::out_of_range const& e)
    {
        std::cerr << "Warning: error parsing tactic " << tactic_config.dump(2) << std::endl;
    }

    return -1;
}

template <class BenchClass>
void parseTacticToVectorID(nlohmann::json& tactic, std::vector<int>& tactic_ids)
{
    if (tactic.is_number_integer())
    {
        tactic_ids.push_back(tactic.get<int>());
    }
    else if (tactic.is_array())
    {
        for (auto c : tactic)
        {
            parseTacticToVectorID<BenchClass>(c, tactic_ids);
        }
    }
    else if (tactic.is_object())
    {
        tactic_ids.push_back(parseTacticToId<BenchClass>(tactic));
    }
    else if (tactic.is_string())
    {
        assert(tactic.is_string());
        auto tactic_name = tactic.get<std::string>();
        if (tactic_name == "all")
        {
            auto all_tactics = listAllTactics<BenchClass>();
            tactic_ids.resize(all_tactics.size());
            std::iota(tactic_ids.begin(), tactic_ids.end(), 0);
        }
        else
        {
            assert(tactic.get<std::string>() == "auto");
            tactic_ids.push_back(-1);
        }
    }
    else
    {
        throw std::invalid_argument("Invalid tactic format");
    }
}

// This interdependence of globals could be better, but it works ok for this limited case.
std::unordered_map<std::string, std::pair<int, int>> name_info_map{
    {routingConfigCache[LOAD_BALANCED_ROUTING_CONFIG]->getName(), {-1, LOAD_BALANCED_ROUTING_CONFIG}},
    {routingConfigCache[UNIFORM_ROUTING_CONFIG]->getName(), {-1, UNIFORM_ROUTING_CONFIG}},
};

int getNameCacheIdx(std::string const& name)
{
    if (name_info_map.find(name) == name_info_map.end())
    {
        return -1;
    }
    return name_info_map.at(name).second;
}

void setNameCacheIdx(std::string const& name, int id)
{
    name_info_map.at(name).second = id;
}

template <class ConfigType>
std::optional<int> loadRoutingValues(nlohmann::json entry, int64_t num_experts, std::string config_name)
{
    std::optional<int> routing_config;
    if (entry.is_string())
    {
        routing_config = getNameCacheIdx(entry.get<std::string>());
        if (routing_config < 0)
        {
            throw std::invalid_argument("Invalid routing value, could not find valid config");
        }
    }
    else
    {
        if (config_name.empty())
        {
            throw std::invalid_argument("Explicit routing configurations must specify a name");
        }
        std::vector<float> routing_values;
        entry.get_to(routing_values);

        int64_t shape = routing_values.size() / num_experts;
        routingConfigCache.push_back(std::make_shared<ConfigType>(
            std::move(routing_values), std::pair<int64_t, int64_t>{shape, num_experts}, config_name));
        routing_config = routingConfigCache.size() - 1;
    }

    auto conf = routingConfigCache[*routing_config];

    bool const is_supported = conf->supportsConfig(num_experts, {}, {});
    auto conf_derived = std::dynamic_pointer_cast<ConfigType>(conf);
    auto conf_default = std::dynamic_pointer_cast<std::conditional_t<std::is_same_v<ConfigType, VectoredRoutingConfig>,
        LoadBalancedRoutingConfig, UniformRoutingConfig>>(conf);
    bool const is_valid_type = conf_derived || conf_default;

    if (!is_supported || !is_valid_type)
    {
        throw std::invalid_argument("Incompatible config selected. "
            + ((conf_derived) ? "Expected " + std::to_string(num_experts)
                        + " experts in routing configuration. Found: " + std::to_string(conf_derived->shape.second)
                              : "Found incompatible routing config type"));
    }

    return routing_config;
}

// This is suboptimal for large benchmark files as we reread it for every data type
template <class BenchClass>
void argGenLoadFile(benchmark::internal::Benchmark* benchmark)
{
    /*
     * See help text for schema description
     */

    std::ifstream file{workloadFile};
    std::stringstream buffer;
    buffer << file.rdbuf();
    auto file_contents = buffer.str();
    if (LOG_LEVEL >= INFO)
        std::cout << "Loaded benchmark file: " << file_contents << std::endl;
    auto source_data = nlohmann::json::parse(file_contents);

    int i = 0;
    for (auto run_config : source_data)
    {
        if (LOG_LEVEL >= VERBOSE)
            std::cout << "Parsing run config: " << run_config.dump(2) << std::endl;
        std::string config_name = "config_" + std::to_string(i);

        // WARNING: Process the routing configuration immediately, so we can guarantee all configs get processed for all
        // data types. We should not skip any test cases as a later test config may depend on this config
        if (run_config.contains("routing_values_name"))
        {
            run_config["routing_values_name"].get_to(config_name);
            if (!run_config.contains("routing_values") && !run_config.contains("routing_distribution"))
            {
                throw std::invalid_argument("Setting routing value configuration name but missing routing values");
            }
        }

        std::optional<int> routing_config;
        auto res = name_info_map.emplace(config_name, std::pair{i, -1});
        // We must check i is not equal since this function gets called for each data type
        if (!res.second && res.first->second.first != i)
        {
            throw std::invalid_argument("Redefinition of routing_values_name " + config_name + " at config "
                + std::to_string(i) + ". First declared at " + std::to_string(res.first->second.first));
        }
        else if (!res.second)
        {
            // Reuse the existing config from a previous parse
            routing_config = getNameCacheIdx(config_name);
        }
        i++;

        int num_experts = run_config.at("num_experts").get<int>();

        if (!routing_config)
        {
            if (run_config.contains("routing_values"))
            {
                routing_config
                    = loadRoutingValues<VectoredRoutingConfig>(run_config["routing_values"], num_experts, config_name);
            }
            else if (run_config.contains("routing_distribution"))
            {
                routing_config = loadRoutingValues<RandomDistributionRoutingConfig>(
                    run_config["routing_distribution"], num_experts, config_name);
            }
        }
        // Use the selected config or fall back to balanced
        routing_config = routing_config.value_or(LOAD_BALANCED_ROUTING_CONFIG);
        setNameCacheIdx(config_name, *routing_config);

        // Filter out the types we don't care about testing
        if (run_config.contains("dtypes"))
        {
            std::vector<std::string> dtypes;
            run_config["dtypes"].get_to(dtypes);

            auto hasDtype = [&](char const* d)
            { return std::any_of(dtypes.begin(), dtypes.end(), [&](auto const& n) { return n == d; }); };

            if (BenchClass::FP8 && !hasDtype("fp8"))
            {
                continue;
            }
            else if (BenchClass::INT4 && !hasDtype("int4"))
            {
                continue;
            }
            else if (!BenchClass::INT4 && BenchClass::INT_QUANT && !hasDtype("int8"))
            {
                continue;
            }
            else if (std::is_same_v<typename BenchClass::WeightType, float> && !hasDtype("float")
                && !hasDtype("float32"))
            {
                continue;
            }
            else if (std::is_same_v<typename BenchClass::WeightType, half> && !hasDtype("float16") && !hasDtype("half"))
            {
                continue;
            }
            else if (std::is_same_v<typename BenchClass::WeightType, __nv_bfloat16> && !hasDtype("bfloat16")
                && !hasDtype("bf16"))
            {
                continue;
            }
        }

        // Do this after filtering datatypes as tactics only make sense if we know the data type
        bool has_tactic_ids2 = false;
        std::vector<int> tactic_ids1{};
        std::vector<int> tactic_ids2{};
        if (run_config.contains("tactic_id1") || run_config.contains("tactic_id2"))
        {
            if (run_config.contains("tactic_id"))
            {
                throw std::invalid_argument("Cannot use tactic_id and tactic_idX");
            }
            has_tactic_ids2 = true;
            parseTacticToVectorID<BenchClass>(run_config["tactic_id1"], tactic_ids1);
            parseTacticToVectorID<BenchClass>(run_config["tactic_id2"], tactic_ids2);
        }
        else
        {
            parseTacticToVectorID<BenchClass>(run_config["tactic_id"], tactic_ids1);
            has_tactic_ids2 = false;
            tactic_ids2.resize(1); // Dummy value so we loop exactly once below
        }
        if (tactic_ids1.empty() || tactic_ids2.empty())
        {
            std::cerr << "Warning: Skipping benchmark, no valid tactic found" << std::endl;
            static bool printed = false;
            if (!printed)
            {
                printed = true;
                std::cerr << __PRETTY_FUNCTION__ << ": Valid Tactics are:\n";
                auto confs = listAllTactics<BenchClass>();
                for (auto c : confs)
                    std::cerr << c.toString();
            }

            continue;
        }

        auto get_or = [&](auto name, auto def)
        { return run_config.contains(name) ? run_config[name].template get<decltype(def)>() : def; };
        int tp_size = get_or("tp_size", 1);
        int ep_size = get_or("ep_size", 1);
        int world_rank = get_or("world_rank", 0);
        int bias = get_or("bias", 0);
        TLLM_CHECK_WITH_INFO(world_rank < tp_size * ep_size, "Rank is out of bounds of tp*ep");

        auto get_range = [&](std::string name, int min = 1, int max = INT32_MAX)
        {
            auto val = run_config.at(name).get<int>();
            if (val < min || val > max)
            {
                throw std::invalid_argument(name + " must be a positive integer");
            }
            return val;
        };

        for (auto t1 : tactic_ids1)
        {
            // tactic_ids2 will have one dummy value if has_tactic_ids2 = false
            for (auto t2 : tactic_ids2)
            {
                if (!has_tactic_ids2)
                    t2 = t1;

                benchmark->Args({num_experts,                                                      //
                    get_range("k"),                                                                //
                    get_range("hidden_size"),                                                      //
                    get_range("inter_size"),                                                       //
                    tp_size, ep_size, world_rank,                                                  //
                    get_range("num_tokens"),                                                       //
                    bias,                                                                          //
                    get_range("act_fn", 0, (int) tensorrt_llm::ActivationType::Identity),          //
                    get_range("norm_mode", 0, (int) MOEExpertScaleNormalizationMode::RENORMALIZE), //
                    t1,                                                                            //
                    t2,                                                                            //
                    *routing_config});
            }
        }
    }
}

template <class BenchClass>
void argGenHardcoded(benchmark::internal::Benchmark* benchmark)
{
    auto num_experts = {1, 8, 9, 64, 65, 257}; // {1, 8, 64, 65, 1024};
    auto top_k = {1, 2, 3, 16};                // {1, 2, 3, 42};
    auto hidden_size = {4096};
    auto inter_size_mul = {4.f};               // {7.f/2.f, 4.f};
    auto num_tokens = {2048};                  // {1, 20, 200, 2048};
    auto use_bias = {0};                       // {0, 1};
    auto activation_type = {tensorrt_llm::ActivationType::Gelu};
    // {tensorrt_llm::ActivationType::Relu, tensorrt_llm::ActivationType::Gelu,
    // tensorrt_llm::ActivationType::Silu, tensorrt_llm::ActivationType::Geglu,
    // tensorrt_llm::ActivationType::Swiglu};
    auto norm_mode = {MOEExpertScaleNormalizationMode::NONE};
    auto cutlass_tactic = {-1};                           // {0,..., listAllTactics<BenchClass>().size()};
    auto routing_config = {LOAD_BALANCED_ROUTING_CONFIG}; // {0, 1, 2};

    for (auto num_expert : num_experts)
        for (auto k : top_k)
            if (k <= num_expert)
                for (auto size : hidden_size)
                    for (auto inter_mul : inter_size_mul)
                    {
                        auto inter_size = static_cast<int>(size * inter_mul);
                        for (auto tokens : num_tokens)
                            for (auto bias : use_bias)
                                for (auto act : activation_type)
                                    for (auto norm : norm_mode)
                                        for (auto tactic1 : cutlass_tactic)
                                            for (auto tactic2 : cutlass_tactic)
                                                for (auto routing : routing_config)
                                                    benchmark->Args({num_expert, k, size, inter_size, 1, 1, 0, tokens,
                                                        bias, (int) act, (int) norm, tactic1, tactic2, routing});
                    }
}

template <class BenchClass>
void argGen(benchmark::internal::Benchmark* benchmark)
{
    if (LOG_LEVEL >= VERBOSE)
    {
        std::cout << "List of all tactics for dtype " << (int) BenchClass::toDTypeID() << ":\n";
        int i = 0;
        for (auto& t : listAllTactics<BenchClass>())
        {
            std::cout << "Tactic " << i << ":\n";
            std::cout << t.toString() << std::endl;

            i++;
        }
    }

    // Generic setup
    benchmark->UseManualTime();
    benchmark->ArgNames({"Num Experts", "K", "Hidden Size", "Inter Size", "TP Size", "EP Size", "World Rank",
        "Num Tokens", "Use Bias", "Activation Function", "Norm Mode", "Tactic ID 1", "Tactic ID 2", "Routing ID"});

    if (workloadFile)
        argGenLoadFile<BenchClass>(benchmark);
    else
        argGenHardcoded<BenchClass>(benchmark);
}

BENCHMARK_BASIC(float, float, float)
BENCHMARK_BASIC(half, half, half)
using uint8 = uint8_t;
BENCHMARK_BASIC(half, uint8, half)
using cutlass::uint4b_t;
BENCHMARK_BASIC(half, uint4b_t, half)
#ifdef ENABLE_BF16
BENCHMARK_BASIC(nv_bfloat16, nv_bfloat16, nv_bfloat16)
#endif
#ifdef ENABLE_FP8
BENCHMARK_BASIC(SafeFP8, SafeFP8, half)
#endif

void delayedRegisterBenchmark()
{
    BENCHMARK_BASIC_DO_REGISTER(half, half, half);
#ifdef ENABLE_FP8
    BENCHMARK_BASIC_DO_REGISTER(SafeFP8, SafeFP8, half);
#endif
    if (workloadFile)
    {
        // Extra ones we don't want for hardcoded runs
        BENCHMARK_BASIC_DO_REGISTER(float, float, float);
        BENCHMARK_BASIC_DO_REGISTER(half, uint8, half);
        BENCHMARK_BASIC_DO_REGISTER(half, uint4b_t, half);
#ifdef ENABLE_BF16
        BENCHMARK_BASIC_DO_REGISTER(nv_bfloat16, nv_bfloat16, nv_bfloat16);
#endif
    }
}

void doCleanup()
{
    bufferManager.reset();
    streamPtr.reset();
}

void help()
{
    std::cout << "Usage: mixtureOfExpertsBackendBenchmark [--input_file <file>] [benchmark options]\n";
    std::cout
        << "--input_file\t\tA JSON file describing the benchmark configurations\n\n"
        << "File schema\n"
           "[\n"
           "  {\n"
           "    \"num_experts\": int,\n"
           "    \"k\": int,\n"
           "    \"hidden_size\": int,\n"
           "    \"inter_size\": int,\n"
           "    \"tp_size\": int, (optional)\n"
           "    \"ep_size\": int, (optional)\n"
           "    \"world_rank\": int, (optional)\n"
           "    \"num_tokens\": int,\n"
           "    \"bias\": int,\n"
           "    \"act_fn\": int,\n"
           "    \"norm_mode\": int,\n"
           "    \"tactic_id\": tactic, (see below)\n"
           "    \"tactic_id1\": tactic, (see below)\n"
           "    \"tactic_id2\": tactic, (see below)\n"
           "    \"dtypes\": [string, ...], (optional)\n"
           "    \"routing_values_name\": string, (optional)\n"
           "    \"routing_values\": [float, ...], or string, (optional, length is a multiple of num_experts)\n"
           "    \"routing_distribution\": [float, ...], or string, (optional, length is num_experts)\n"
           "  },\n"
           "  ...\n"
           "]\n"
           "Explanation:\n"
           "- \"num_experts\" - The number of experts\n"
           "- \"k\" - The top k\n"
           "- \"hidden_size\" - The hidden size\n"
           "- \"inter_size\" - The inter size\n"
           "- \"tp_size\" - The TP size to use\n"
           "- \"ep_size\" - The EP size to use\n"
           "- \"world_rank\" - The world rank = tp_rank * ep_size + ep_rank\n"
           "- \"num_tokens\" - The total number of tokens to benchmark\n"
           "- \"bias\" - If bias should be used, 0 = no bias, 1 = bias\n"
           "- \"act_fn\" - The enum value of the activation function. See\n"
           "\"cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h\"\n"
           "- \"norm_mode\" - The normalization mode. 0 = NONE, 1 = RENORM, 2 = SPARSE_MIXER. See\n"
           "\"cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h\"\n"
           "- \"tactic_id, tactic_id1, tactic_id2\"\n"
           "The config for the CUTLASS GEMM. tactic_id sets the same tactic for both to the same tactic (except in "
           "auto mode)\n"
           "Use tactic_idX to set the tactic for the corresponding GEMM"
           "Valid tactics are:\n"
           " - An object:\n"
           "   {\n"
           "      \"is_sm90\": bool,\n"
           "      \"tile_shape\": [int, int, int] or int,\n"
           "      \"cluster_shape\": [int, int, int] or int, (required for sm90, type must be an int if tile_shape "
           "is "
           "an int)\n"
           "      \"warp_shape\": [int, int, int], (required for non-sm90 if tile_shape is an array)\n"
           "      \"stages\": int, (required for non-sm90)\n"
           "    },\n"
           " - An integer: corresponds to an index in the tactics array. WARNING this is not stable between test "
           "configurations\n"
           " - An array: of integers or objects, forms a list of tactics to sweep\n"
           " - The string \"all\": This will sweep through all possible tactics\n"
           " - The string \"auto\": This runs a short benchmark to pick the fastest tactic before each benchmark "
           "case. "
           "Useful for quick perf tests, prefer a full sweep and manually setting the tactic for more accurate "
           "results"
           "- dtypes - A list of dtypes to run this config through.\n"
           "Allowed values are: fp8, int4, int8, float, half, bfloat16\n"
           "If this argument is omitted all dtypes will be run. Note, not all tactics are supported for all "
           "dtypes,\n"
           "unsupported tactics will be skipped with a warning.\n"
           "- \"routing_values_name\" - a name to help identify the routing pattern. This can be used by later "
           "benchmarks to reuse the config\n"
           "- \"routing_values\" - a flat array of routing values to define a new config, or a string referencing "
           "the name of a\n"
           "previous config. Defaults to pre-defined config \"balanced\", which is short-hand for a perfectly balanced "
           "expert distribution\n"
           "These define the routing values used as input to the moe backend, and is intended to allow comparing "
           "different routing behaviours.\n"
           "When defining an array, it must have `T*num_experts` floating point values. Each set of\n"
           "`num_experts` values defines the input for a single token. If `num_tokens` is greater than `T` it will "
           "repeat from the beginning\n"
           "- \"routing_distribution\" - instead of explicitly setting routing_values, define a random distribution "
           "that experts will be randomly sampled from."
           "There is also pre-defined config \"uniform\", which is short-hand for a random uniform distribution\n"
           "\n";

    std::cout << "benchmark options:\n";
    benchmark::PrintDefaultHelp();
}

void gbenchCustomHelp()
{
    help();
    // google-benchmark calls exit() so we need to cleanup manually
    doCleanup();
}

int parseArgsAndRunBench(int argc, char** argv)
{
    try
    {
        int shift = 0;
        for (int i = 1; i < argc; i++)
        {
            argv[i - shift] = argv[i];
            if (strcmp("--input_file", argv[i]) == 0)
            {
                i += 1;
                if (i == argc)
                {
                    std::cerr << "Missing file name for input_file\n";
                    return -1;
                }
                workloadFile = argv[i];
                if (workloadFile[0] == '-')
                {
                    std::cerr << "Workload file " << workloadFile << " not a valid file name\n";
                    return -2;
                }
                shift += 2;
            }
            else if (strcmp("--help", argv[i]) == 0 || strcmp("-h", argv[i]) == 0)
            {
                help();
                return 0;
            }
        }
        argc -= shift;

        // Delay after we know if the user passed a config file
        delayedRegisterBenchmark();

        benchmark::Initialize(&argc, argv, &gbenchCustomHelp);

        if (argc > 1)
        {
            help();
            std::cout << std::flush; // Force flush
            // Print the error second, so it's easy to see
            std::cerr << "\nUnrecognised argument: " << argv[1] << std::endl;
            return -4;
        }

        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();

        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Exiting benchmarks with exception: " << e.what() << std::endl;
        return -3;
    }
}

int main(int argc, char** argv)
{
    deviceCount = getDeviceCount();
    if (deviceCount < 0)
        return 0;
    streamPtr = std::make_shared<CudaStream>();
    bufferManager = std::make_unique<BufferManager>(streamPtr);

    int res = -1;
    try
    {
        res = parseArgsAndRunBench(argc, argv);
    }
    catch (std::exception const& e)
    {
        std::cout << "Benchmark exited with unhandled exception: " << e.what() << std::endl;
    }

    doCleanup();
    return res;
}
