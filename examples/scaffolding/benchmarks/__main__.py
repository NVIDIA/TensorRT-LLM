"""Main entry point for scaffolding benchmarks.

This module provides the command-line interface for running various benchmarks:
- Normal agent benchmark
- Burst agent benchmark
- Chat benchmark
- Multiround chat benchmark

Usage:
    python -m examples.scaffolding.benchmarks --model_dir /path/to/model ...

Or from the benchmarks directory:
    python -m . --model_dir /path/to/model ...
"""

import argparse
import sys

from tensorrt_llm.scaffolding import TaskMetricsCollector

from .agent_benchmark import async_agent_benchmark, async_burst_agent_benchmark
from .benchmark_utils import run_benchmark_in_thread
from .chat_benchmark import async_chat_benchmark
from .multiround_chat_benchmark import async_multiround_chat_benchmark


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Scaffolding benchmarks for agentic and chat workloads"
    )
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="gpt-oss-20b")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the directory containing the generation model",
    )

    # Benchmark enable flags
    parser.add_argument(
        "--enable_normal_agent",
        action="store_true",
        help="Enable normal agent benchmark (uses OpenAI API with MCP tools)",
    )
    parser.add_argument(
        "--enable_chat",
        action="store_true",
        help="Enable chat benchmark (simple single-turn generation)",
    )
    parser.add_argument(
        "--enable_multiround_chat",
        action="store_true",
        help="Enable multiround chat benchmark (requires --multiround_synthetic or --multiround_sharegpt_file)",
    )

    # Benchmark parameters
    parser.add_argument(
        "--agent_prompt_num",
        type=int,
        default=100,
        help="Number of prompts to send for agent benchmark (default: 100)",
    )
    parser.add_argument(
        "--chat_prompt_num",
        type=int,
        default=100,
        help="[Chat only] Number of prompts for chat benchmark (default: 100).",
    )
    parser.add_argument(
        "--multiround_num_conversations",
        type=int,
        default=100,
        help="[Multiround] Max conversations for multiround benchmark (default: 100). "
        "Applies to both ShareGPT and synthetic data sources.",
    )
    parser.add_argument(
        "--multiround_max_rounds",
        type=int,
        default=20,
        help="[Multiround] Maximum rounds per conversation (default: 20). "
        "This caps the number of turns executed, even if generated conversations have more.",
    )
    parser.add_argument(
        "--normal_agent_concurrency",
        type=int,
        default=32,
        help="Concurrency for normal agent benchmark (default: 32)",
    )
    parser.add_argument(
        "--chat_concurrency",
        type=int,
        default=32,
        help="Concurrency for chat benchmark (default: 32)",
    )
    parser.add_argument(
        "--multiround_concurrency",
        type=int,
        default=32,
        help="Concurrency for multiround chat benchmark (default: 32)",
    )
    parser.add_argument(
        "--max_tokens_agent",
        type=int,
        default=65536,
        help="[Agent only] Maximum number of tokens to generate (default: 65536)",
    )
    parser.add_argument(
        "--max_tokens_chat",
        type=int,
        default=8192,
        help="[Chat/Multiround only] Maximum number of tokens to generate per turn (default: 8192)",
    )
    parser.add_argument(
        "--max_parallel_requests",
        type=int,
        default=1024,
        help="[All benchmarks] Maximum number of parallel requests (default: 1024)",
    )
    parser.add_argument(
        "--enable_statistics",
        action="store_true",
        help="[All benchmarks] Enable task metrics statistics",
    )
    parser.add_argument(
        "--enable_query_collector",
        action="store_true",
        help="[Agent only] Enable query collector for debugging",
    )

    # Burst agent parameters
    parser.add_argument(
        "--enable_burst_agent",
        action="store_true",
        help="Enable burst agent benchmark (simulates sudden traffic spike)",
    )
    parser.add_argument(
        "--burst_delay",
        type=float,
        default=240,
        help="Delay in seconds before burst traffic starts (default: 240)",
    )
    parser.add_argument(
        "--burst_prompt_num",
        type=int,
        default=32,
        help="Number of prompts for burst traffic (default: 32)",
    )
    parser.add_argument(
        "--burst_agent_concurrency",
        type=int,
        default=32,
        help="Concurrency for burst agent benchmark (default: 32)",
    )

    # =========================================================================
    # Multiround Chat Configuration
    # =========================================================================

    # Data source configuration
    parser.add_argument(
        "--multiround_sharegpt_file",
        type=str,
        default=None,
        help="[Multiround] Path to ShareGPT JSON file for conversation data. "
        "Number of conversations controlled by --multiround_num_conversations.",
    )
    parser.add_argument(
        "--multiround_synthetic",
        action="store_true",
        help="[Multiround] Enable synthetic data generation (alternative to --multiround_sharegpt_file).",
    )

    # -------------------------------------------------------------------------
    # Synthetic data generation parameters (used when --multiround_synthetic is set)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--multiround_text_files",
        type=str,
        nargs="+",
        default=["examples/scaffolding/benchmarks/pg1184.txt"],
        help="[Multiround/Synthetic] Text files to use for content generation (default: pg1184.txt)",
    )
    parser.add_argument(
        "--multiround_print_stats",
        action="store_true",
        help="[Multiround/Synthetic] Print conversation statistics after generation",
    )
    parser.add_argument(
        "--multiround_seed",
        type=int,
        default=None,
        help="[Multiround/Synthetic] Random seed for reproducible synthetic data generation. "
        "If not set, results will vary between runs.",
    )
    # num_turns distribution
    parser.add_argument(
        "--multiround_num_turns_distribution",
        type=str,
        default="uniform",
        choices=["constant", "uniform", "zipf", "poisson"],
        help="[Multiround/Synthetic] Distribution for number of turns (default: uniform)",
    )
    parser.add_argument(
        "--multiround_num_turns_min",
        type=int,
        default=12,
        help="[Multiround/Synthetic] Min turns for uniform distribution (default: 12)",
    )
    parser.add_argument(
        "--multiround_num_turns_max",
        type=int,
        default=18,
        help="[Multiround/Synthetic] Max turns for uniform distribution (default: 18)",
    )
    parser.add_argument(
        "--multiround_num_turns_value",
        type=int,
        default=10,
        help="[Multiround/Synthetic] Constant value for num_turns (default: 10)",
    )
    # prefix_num_tokens distribution
    parser.add_argument(
        "--multiround_prefix_tokens_distribution",
        type=str,
        default="lognormal",
        choices=["constant", "uniform", "lognormal"],
        help="[Multiround/Synthetic] Distribution for prefix tokens (default: lognormal)",
    )
    parser.add_argument(
        "--multiround_prefix_tokens_average",
        type=int,
        default=1000,
        help="[Multiround/Synthetic] Average prefix tokens for lognormal (default: 1000)",
    )
    parser.add_argument(
        "--multiround_prefix_tokens_max",
        type=int,
        default=5000,
        help="[Multiround/Synthetic] Max prefix tokens (default: 5000)",
    )
    parser.add_argument(
        "--multiround_prefix_tokens_min",
        type=int,
        default=500,
        help="[Multiround/Synthetic] Min prefix tokens for uniform (default: 500)",
    )
    parser.add_argument(
        "--multiround_prefix_tokens_value",
        type=int,
        default=1000,
        help="[Multiround/Synthetic] Constant value for prefix tokens (default: 1000)",
    )
    # input_num_tokens distribution
    parser.add_argument(
        "--multiround_input_tokens_distribution",
        type=str,
        default="uniform",
        choices=["constant", "uniform", "lognormal"],
        help="[Multiround/Synthetic] Distribution for input tokens per turn (default: uniform)",
    )
    parser.add_argument(
        "--multiround_input_tokens_min",
        type=int,
        default=200,
        help="[Multiround/Synthetic] Min input tokens for uniform (default: 200)",
    )
    parser.add_argument(
        "--multiround_input_tokens_max",
        type=int,
        default=400,
        help="[Multiround/Synthetic] Max input tokens (default: 400)",
    )
    parser.add_argument(
        "--multiround_input_tokens_average",
        type=int,
        default=300,
        help="[Multiround/Synthetic] Average input tokens for lognormal (default: 300)",
    )
    parser.add_argument(
        "--multiround_input_tokens_value",
        type=int,
        default=300,
        help="[Multiround/Synthetic] Constant value for input tokens (default: 300)",
    )
    # output_num_tokens distribution
    parser.add_argument(
        "--multiround_output_tokens_distribution",
        type=str,
        default="uniform",
        choices=["constant", "uniform", "lognormal"],
        help="[Multiround/Synthetic] Distribution for output tokens per turn (default: uniform)",
    )
    parser.add_argument(
        "--multiround_output_tokens_min",
        type=int,
        default=200,
        help="[Multiround/Synthetic] Min output tokens for uniform (default: 200)",
    )
    parser.add_argument(
        "--multiround_output_tokens_max",
        type=int,
        default=400,
        help="[Multiround/Synthetic] Max output tokens (default: 400)",
    )
    parser.add_argument(
        "--multiround_output_tokens_average",
        type=int,
        default=300,
        help="[Multiround/Synthetic] Average output tokens for lognormal (default: 300)",
    )
    parser.add_argument(
        "--multiround_output_tokens_value",
        type=int,
        default=300,
        help="[Multiround/Synthetic] Constant value for output tokens (default: 300)",
    )
    # -------------------------------------------------------------------------
    # User delay distribution (simulates user thinking/typing time)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--multiround_user_delay_disabled",
        action="store_true",
        help="[Multiround] Disable user response delays (enabled by default)",
    )
    parser.add_argument(
        "--multiround_user_delay_distribution",
        type=str,
        default="exponential",
        choices=["exponential", "poisson", "constant", "uniform"],
        help="[Multiround] Distribution for user delay (default: exponential)",
    )
    parser.add_argument(
        "--multiround_user_delay_lambda",
        type=float,
        default=1.0,
        help="[Multiround] Lambda/mean for exponential/poisson delay in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--multiround_user_delay_constant",
        type=float,
        default=1.0,
        help="[Multiround] Constant delay value in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--multiround_user_delay_min",
        type=float,
        default=0.5,
        help="[Multiround] Min delay for uniform distribution in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--multiround_user_delay_max",
        type=float,
        default=2.0,
        help="[Multiround] Max delay for uniform distribution in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--multiround_user_delay_cap",
        type=float,
        default=10.0,
        help="[Multiround] Maximum cap for any delay in seconds (default: 10.0)",
    )
    # KV cache hint settings (per-benchmark)
    parser.add_argument(
        "--kv_cache_hint_enabled",
        action="store_true",
        help="[All benchmarks] Enable KV cache hint for all benchmarks (overrides individual settings)",
    )
    parser.add_argument(
        "--kv_cache_hint_agent",
        action="store_true",
        help="[Agent only] Enable KV cache hint for agent benchmarks (normal and burst)",
    )
    parser.add_argument(
        "--kv_cache_hint_chat",
        action="store_true",
        help="[Chat only] Enable KV cache hint for chat benchmark",
    )
    parser.add_argument(
        "--kv_cache_hint_multiround",
        action="store_true",
        help="[Multiround only] Enable KV cache hint for multiround chat benchmark",
    )
    parser.add_argument(
        "--export_task_metrics_path",
        type=str,
        default=None,
        help="[All benchmarks] Export task metrics to this JSON file",
    )

    return parser.parse_args()


# Benchmark registry: (async_func, display_name, flag_name)
BENCHMARK_REGISTRY = [
    (async_agent_benchmark, "Agent-Benchmark", "enable_normal_agent"),
    (async_burst_agent_benchmark, "Burst-Agent-Benchmark", "enable_burst_agent"),
    (async_chat_benchmark, "Chat-Benchmark", "enable_chat"),
    (async_multiround_chat_benchmark, "Multiround-Chat-Benchmark", "enable_multiround_chat"),
]


def main():
    args = parse_arguments()

    # Handle user delay enabled/disabled logic (enabled by default)
    args.multiround_user_delay_enabled = not args.multiround_user_delay_disabled

    # Resolve KV cache hint settings: --kv_cache_hint_enabled overrides all individual settings
    if args.kv_cache_hint_enabled:
        args.kv_cache_hint_agent = True
        args.kv_cache_hint_chat = True
        args.kv_cache_hint_multiround = True

    # Validate multiround chat arguments
    if args.enable_multiround_chat:
        if not args.multiround_sharegpt_file and not args.multiround_synthetic:
            print(
                "Error: --enable_multiround_chat requires either "
                "--multiround_synthetic or --multiround_sharegpt_file"
            )
            sys.exit(1)
        if args.multiround_sharegpt_file and args.multiround_synthetic:
            print(
                "Warning: Both --multiround_sharegpt_file and --multiround_synthetic provided. "
                "Using --multiround_synthetic (takes precedence)."
            )

    # Select enabled benchmarks from registry
    enabled_benchmarks = [
        (async_func, name, flag)
        for async_func, name, flag in BENCHMARK_REGISTRY
        if getattr(args, flag, False)
    ]

    if not enabled_benchmarks:
        print(
            "No benchmark enabled. Use --enable_normal_agent, --enable_burst_agent, "
            "--enable_chat, or --enable_multiround_chat"
        )
        sys.exit(1)

    # Create and start all benchmark threads
    enabled_names = [name for _, name, _ in enabled_benchmarks]
    print(f"Starting benchmarks: {', '.join(enabled_names)}")
    print("=" * 60)

    threads = []
    for async_func, name, _ in enabled_benchmarks:
        thread = run_benchmark_in_thread(async_func, args, name)
        threads.append(thread)
        print(f"- {name} thread")
    print()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("\n" + "=" * 60)
    print("All benchmarks completed!")
    print("=" * 60)

    # Print and export task metrics after all benchmarks complete
    if args.enable_statistics:
        TaskMetricsCollector.print_summary()

    if args.export_task_metrics_path:
        TaskMetricsCollector.export_to_json(args.export_task_metrics_path)
        print(f"Task metrics exported to: {args.export_task_metrics_path}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
