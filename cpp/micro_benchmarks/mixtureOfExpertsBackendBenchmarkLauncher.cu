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
auto listAllTactics(MoeGemmId gemm_id)
{
    int const sm = getSMVersion();
    using RunnerType = decltype(BenchClass::mMoERunner);
    return RunnerType::getTactics(sm, gemm_id);
}

template <class BenchClass>
void parseTacticToVectorID(nlohmann::json& tactic, std::vector<int>& tactic_ids, MoeGemmId gemm_id)
{
    if (tactic.is_number_integer())
    {
        tactic_ids.push_back(tactic.get<int>());
    }
    else if (tactic.is_array())
    {
        for (auto c : tactic)
        {
            parseTacticToVectorID<BenchClass>(c, tactic_ids, gemm_id);
        }
    }
    else if (tactic.is_string())
    {
        assert(tactic.is_string());
        auto tactic_name = tactic.get<std::string>();
        if (tactic_name == "all")
        {
            auto all_tactics = listAllTactics<BenchClass>(gemm_id);
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
std::optional<int> loadRoutingValues(nlohmann::json entry, int64_t num_experts, int64_t k, std::string config_name)
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

        std::vector<typename ConfigType::ElementType> values;
        entry.get_to(values);

        routingConfigCache.push_back(std::make_shared<ConfigType>(std::move(values), num_experts, k, config_name));
        routing_config = routingConfigCache.size() - 1;
    }

    auto conf = routingConfigCache[*routing_config];

    bool const is_supported = conf->supportsConfig(num_experts, k, {});
    auto conf_derived = std::dynamic_pointer_cast<ConfigType>(conf);
    auto conf_default = std::dynamic_pointer_cast<std::conditional_t<std::is_same_v<ConfigType, VectoredRoutingConfig>,
        LoadBalancedRoutingConfig, UniformRoutingConfig>>(conf);
    bool const is_valid_type = conf_derived || conf_default;

    if (!is_supported || !is_valid_type)
    {
        throw std::invalid_argument("Incompatible config selected. "
            + ((conf_derived) ? "Expected experts: " + std::to_string(num_experts) + " and k: " + std::to_string(k)
                        + " in routing configuration. Found: " + std::to_string(conf_derived->num_experts) + " and "
                        + std::to_string(conf_derived->k)
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
    if (!file.is_open())
    {
        throw std::invalid_argument("Failed to open benchmark file: " + std::string(workloadFile));
    }
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
        if (run_config.contains("routing_name"))
        {
            run_config["routing_name"].get_to(config_name);
            if (!run_config.contains("selected_experts") && !run_config.contains("expert_distribution"))
            {
                throw std::invalid_argument("Setting routing value configuration name but missing routing values");
            }
        }

        if (run_config.contains("routing_values") || run_config.contains("routing_distribution"))
        {
            throw std::invalid_argument(
                "Using deprecated routing_values or routing_distribution. Please use selected_experts or "
                "expert_distribution instead.");
        }

        std::optional<int> routing_config;
        auto res = name_info_map.emplace(config_name, std::pair{i, -1});
        // We must check i is not equal since this function gets called for each data type
        if (!res.second && res.first->second.first != i)
        {
            throw std::invalid_argument("Redefinition of routing_name " + config_name + " at config "
                + std::to_string(i) + ". First declared at " + std::to_string(res.first->second.first));
        }
        else if (!res.second)
        {
            // Reuse the existing config from a previous parse
            routing_config = getNameCacheIdx(config_name);
        }
        i++;

        int num_experts = run_config.at("num_experts").get<int>();
        int k = run_config.at("k").get<int>();

        if (!routing_config)
        {
            if (run_config.contains("selected_experts"))
            {
                routing_config = loadRoutingValues<VectoredRoutingConfig>(
                    run_config["selected_experts"], num_experts, k, config_name);
            }
            else if (run_config.contains("expert_distribution"))
            {
                routing_config = loadRoutingValues<RandomDistributionRoutingConfig>(
                    run_config["expert_distribution"], num_experts, k, config_name);
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
            else if (BenchClass::NVFP4 && !hasDtype("fp4"))
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
            // else if (std::is_same_v<typename BenchClass::WeightType, float> && !hasDtype("float")
            //     && !hasDtype("float32"))
            // {
            //     continue;
            // }
            else if (std::is_same_v<typename BenchClass::WeightType, half> && !hasDtype("float16") && !hasDtype("half"))
            {
                continue;
            }
            else if (std::is_same_v<typename BenchClass::WeightType, __nv_bfloat16> && !hasDtype("bfloat16")
                && !hasDtype("bf16"))
            {
                continue;
            }
            else if (BenchClass::WFP4AFP8 && !hasDtype("wfp4afp8"))
            {
                continue;
            }
        }

        // Do this after filtering datatypes as tactics only make sense if we know the data type
        std::vector<int> tactic_ids1{};
        std::vector<int> tactic_ids2{};
        if (run_config.contains("tactic_id1"))
        {
            parseTacticToVectorID<BenchClass>(run_config["tactic_id1"], tactic_ids1, MoeGemmId::GEMM_1);
        }
        if (run_config.contains("tactic_id2"))
        {
            parseTacticToVectorID<BenchClass>(run_config["tactic_id2"], tactic_ids2, MoeGemmId::GEMM_2);
        }

        auto get_or = [&](auto name, auto def)
        { return run_config.contains(name) ? run_config[name].template get<decltype(def)>() : def; };
        int tp_size = get_or("tp_size", 1);
        int ep_size = get_or("ep_size", 1);
        int world_rank = get_or("world_rank", 0);
        int bias = get_or("bias", 0);
        int do_final_scale = get_or("do_final_scale", 1); // Default to scales on
        int gemm_to_profile = get_or("gemm_to_profile", (int) GemmToProfile::LAYER);
        TLLM_CHECK_WITH_INFO(world_rank < tp_size * ep_size, "Rank is out of bounds of tp*ep");

        if (gemm_to_profile != (int) GemmToProfile::LAYER)
        {
            static bool info_printed = false;
            if (!info_printed && LOG_LEVEL >= INFO)
            {
                std::cerr << "Warning: GEMM profiling is experimental, results may be inaccurate" << std::endl;
                info_printed = true;
            }

            static bool printed = false;
            if (routing_config != UNIFORM_ROUTING_CONFIG && LOG_LEVEL >= ERROR && !printed)
            {
                std::cerr << "Warning: Profiling a specific GEMM will always use uniform random token distribution"
                          << std::endl;
                printed = true;
            }
            routing_config = UNIFORM_ROUTING_CONFIG;

            if (gemm_to_profile == (int) GemmToProfile::GEMM_1)
            {
                tactic_ids2 = {-1};
            }
            else if (gemm_to_profile == (int) GemmToProfile::GEMM_2)
            {
                tactic_ids1 = {-1};
            }
        }

        auto get_range = [&](std::string name, int min = 1, int max = INT32_MAX)
        {
            auto val = run_config.at(name).get<int>();
            if (val < min || val > max)
            {
                throw std::invalid_argument(name + " must be a positive integer");
            }
            return val;
        };

        if (tactic_ids1.empty() || tactic_ids2.empty())
        {
            std::cerr << "Warning: Skipping benchmark, no valid tactic found" << std::endl;
            static bool printed = false;
            if (!printed)
            {
                printed = true;
                std::cerr << __PRETTY_FUNCTION__ << ": Valid Tactics are:\n";
                for (auto gemm_id : {MoeGemmId::GEMM_1, MoeGemmId::GEMM_2})
                {
                    std::cerr << "GEMM " << (int) gemm_id << ":\n";
                    auto confs = listAllTactics<BenchClass>(gemm_id);
                    for (auto c : confs)
                        std::cerr << c.toString();
                    std::cerr << std::endl;
                }
            }

            continue;
        }

        for (auto t1 : tactic_ids1)
        {
            for (auto t2 : tactic_ids2)
            {
                benchmark->Args({num_experts,                               //
                    get_range("k"),                                         //
                    get_range("hidden_size"),                               //
                    get_range("inter_size"),                                //
                    tp_size, ep_size, world_rank,                           //
                    get_range("num_tokens"),                                //
                    bias, do_final_scale,                                   //
                    get_range("act_fn", 0, (int) ActivationType::Identity), //
                    t1,                                                     //
                    t2,                                                     //
                    *routing_config, gemm_to_profile});
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
    auto activation_type = {ActivationType::Gelu};
    // {ActivationType::Relu, ActivationType::Gelu,
    // ActivationType::Silu, ActivationType::Geglu,
    // ActivationType::Swiglu};
    auto cutlass_tactic = {-1};                           // {0,..., listAllTactics<BenchClass>(MoeGemmId).size()};
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
                                    for (auto tactic1 : cutlass_tactic)
                                        for (auto tactic2 : cutlass_tactic)
                                            for (auto routing : routing_config)
                                                benchmark->Args({num_expert, k, size, inter_size, 1, 1, 0, tokens, bias,
                                                    1, (int) act, tactic1, tactic2, routing,
                                                    (int) GemmToProfile::LAYER});
                    }
}

template <class BenchClass>
void argGen(benchmark::internal::Benchmark* benchmark)
{
    if (LOG_LEVEL >= VERBOSE)
    {
        std::cout << "== List of all tactics for dtype " << (int) BenchClass::toDTypeID() << " ==\n";
        for (auto gemm_id : {MoeGemmId::GEMM_1, MoeGemmId::GEMM_2})
        {
            int i = 0;
            std::cout << "=== GEMM " << (int) gemm_id << " ===\n";
            for (auto& t : listAllTactics<BenchClass>(gemm_id))
            {
                std::cout << "==== Tactic " << i << " ====\n";
                std::cout << t.toString() << std::endl;

                i++;
            }
        }
    }

    // Generic setup
    benchmark->UseManualTime();
    benchmark->ArgNames(
        {"Num Experts", "K", "Hidden Size", "Inter Size", "TP Size", "EP Size", "World Rank", "Num Tokens", "Use Bias",
            "Use Final Scale", "Activation Function", "Tactic ID 1", "Tactic ID 2", "Routing ID", "Gemm To Profile"});

    if (workloadFile)
        argGenLoadFile<BenchClass>(benchmark);
    else
        argGenHardcoded<BenchClass>(benchmark);
}

// No one cares about float32
// BENCHMARK_BASIC(float, float, float)
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
#ifdef ENABLE_FP4
BENCHMARK_BASIC(SafeFP4, SafeFP4, half)
BENCHMARK_BASIC(SafeFP8, SafeFP4, half)
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
        // BENCHMARK_BASIC_DO_REGISTER(float, float, float);
        BENCHMARK_BASIC_DO_REGISTER(half, uint8, half);
        BENCHMARK_BASIC_DO_REGISTER(half, uint4b_t, half);
#ifdef ENABLE_BF16
        BENCHMARK_BASIC_DO_REGISTER(nv_bfloat16, nv_bfloat16, nv_bfloat16);
#endif
#ifdef ENABLE_FP4
        BENCHMARK_BASIC_DO_REGISTER(SafeFP4, SafeFP4, half);
        BENCHMARK_BASIC_DO_REGISTER(SafeFP8, SafeFP4, half);
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
    std::cout << "**Disclaimer: This benchmark is intended for developers to help evaluating the impact of new "
                 "optimisations. This benchmark does not meet the same quality standards as other parts of TRT-LLM. "
                 "Please use with caution**\n\n";
    std::cout << "Usage: mixtureOfExpertsBackendBenchmark [--disable_cuda_graphs] [--input_file <file>] [benchmark "
                 "options]\n";
    std::cout
        << "--disable_cuda_graphs\t\tPrevent the benchmark from using cuda graphs. Useful for getting the performance "
           "of specific kernels\n"
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
           "    \"bias\": int, (optional)\n"
           "    \"do_final_scale\": int, (optional)\n"
           "    \"act_fn\": int,\n"
           "    \"tactic_id1\": tactic, (see below)\n"
           "    \"tactic_id2\": tactic, (see below)\n"
           "    \"dtypes\": [string, ...], (optional)\n"
           "    \"routing_name\": string, (optional)\n"
           "    \"selected_experts\": [int, ...], or string, (optional, length is a multiple of k)\n"
           "    \"expert_distribution\": [float, ...], or string, (optional, length is num_experts)\n"
           "    \"gemm_to_profile\": int, (experimental, optional, 1 = gemm1, 2 = gemm2, 3 = layer)\n"
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
           "- \"do_final_scale\" - If final scales should be applied, 0 = no scale, 1 = scale\n"
           "- \"act_fn\" - The activation function to use, 0 = identity, 1 = relu, 2 = gelu, 3 = silu, 4 = geglu, 5 = "
           "swiglu\n"
           "- \"tactic_id1, tactic_id2\"\n"
           "The config for the CUTLASS GEMM. tactic_idX sets the tactic for the corresponding GEMM"
           "Valid tactics are:\n"
           " - An integer: corresponds to an index in the tactics array. WARNING this is not stable between data types "
           "or GPU architectures\n"
           " - An array: of integers, forms a list of tactics to sweep\n"
           " - The string \"all\": This will sweep through all possible tactics\n"
           " - The string \"auto\": This runs a short benchmark to pick the fastest tactic before each benchmark case. "
           "Useful for quick perf tests, prefer a full sweep and manually setting the tactic for more accurate "
           "results"
           "- dtypes - A list of dtypes to run this config through.\n"
           "Allowed values are: fp8, fp4, wfp4afp8, int4, int8, half, bfloat16\n"
           "If this argument is omitted all dtypes will be run. Note, not all tactics are supported for all "
           "dtypes,\n"
           "unsupported tactics will be skipped with a warning.\n"
           "- \"routing_name\" - a name to help identify the routing pattern. This can be used by later "
           "benchmarks to reuse the config\n"
           "- \"selected_experts\" - a flat array of selected experts to define a new config,\n or a string "
           "referencing the name of a previous config. Defaults to pre-defined config \"balanced\",\n"
           "which is short-hand for a perfectly balanced expert distribution\n"
           "These define the routing values used as input to the moe backend, and is intended to allow comparing "
           "different routing behaviours.\n"
           "When defining an array, it must have `T*k` floating point values. Each set of\n"
           "`k` values defines the input for a single token. If `num_tokens` is greater than `T` it will "
           "repeat from the beginning\n"
           "- \"expert_distribution\" - instead of explicitly setting selected_experts, define a random distribution "
           "that experts will be randomly sampled from."
           "There is also pre-defined config \"uniform\", which is short-hand for a random uniform distribution\n"
           "- \"gemm_to_profile\" - the gemm to profile, 1 = gemm1, 2 = gemm2, 3 = full layer. (default layer). If a "
           "specific GEMM is profiled, it will always use uniform random token distribution\n"
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
            else if (strcmp("--disable_cuda_graphs", argv[i]) == 0)
            {
                useCudaGraph = false;
                shift++;
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
