"""Main entry point for scaffolding benchmarks.

This module provides the command-line interface for running various benchmarks:
- Normal agent benchmark
- Burst agent benchmark
- Chat benchmark
- Multiround chat benchmark

Usage:
    python -m examples.scaffolding.benchmarks.benchmark_agent_chat --model_dir /path/to/model ...

Or from the benchmarks directory:
    python benchmark_agent_chat.py --model_dir /path/to/model ...
"""

import argparse
import sys

from .agent_benchmark import async_agent_benchmark, async_burst_agent_benchmark
from .benchmark_utils import run_benchmark_in_thread
from .chat_benchmark import async_chat_benchmark
from .multiround_chat_benchmark import async_multiround_chat_benchmark


def parse_arguments():
    parser = argparse.ArgumentParser()
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
        help="Enable normal agent benchmark",
    )
    parser.add_argument(
        "--enable_chatbot",
        action="store_true",
        help="Enable chatbot benchmark",
    )
    parser.add_argument(
        "--enable_multiround_chatbot",
        action="store_true",
        help="Enable multiround chatbot benchmark",
    )

    # Benchmark parameters
    parser.add_argument(
        "--agent_prompt_num",
        type=int,
        default=100,
        help="Number of prompts to send for agent benchmark (default: 10)",
    )
    parser.add_argument(
        "--chat_prompt_num",
        type=int,
        default=20,
        help="Number of prompts to send for chat benchmark (default: 20)",
    )
    parser.add_argument(
        "--chat_multiround_rounds",
        type=int,
        default=3,
        help="Number of rounds for multiround chat benchmark (default: 3)",
    )
    parser.add_argument(
        "--times", type=int, default=1, help="Number of times to run the benchmark (default: 1)"
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
        help="Concurrency for chatbot benchmark (default: 32)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8 * 1024,
        help="Maximum number of tokens to generate (default: 16384)",
    )
    parser.add_argument(
        "--max_tokens_chat",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate for chat (default: 1024)",
    )
    parser.add_argument(
        "--max_parallel_requests",
        type=int,
        default=1024,
        help="Maximum number of parallel requests (default: 1024)",
    )
    parser.add_argument("--enable_statistics", action="store_true")

    parser.add_argument("--enable_query_collector", action="store_true")

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
        "--multiround_data_source",
        type=str,
        default="synthetic",
        choices=["sharegpt", "synthetic", "json_config"],
        help="Data source: 'sharegpt', 'synthetic', or 'json_config' (default: synthetic)",
    )
    parser.add_argument(
        "--multiround_sharegpt_file",
        type=str,
        default=None,
        help="Path to ShareGPT JSON file (required if data_source=sharegpt)",
    )
    parser.add_argument(
        "--multiround_json_config_file",
        type=str,
        default=None,
        help="Path to JSON config file like generate_multi_turn.json (for data_source=json_config)",
    )
    parser.add_argument(
        "--multiround_text_files",
        type=str,
        nargs="+",
        default=[],
        help="Text files for synthetic content generation (space-separated paths)",
    )
    parser.add_argument(
        "--multiround_print_stats",
        action="store_true",
        help="Print detailed statistics about generated conversations",
    )

    # -------------------------------------------------------------------------
    # Turn count distribution (supports: uniform, constant, zipf, poisson, lognormal)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--multiround_turns_distribution",
        type=str,
        default="uniform",
        choices=["uniform", "constant", "zipf", "poisson", "lognormal"],
        help="Distribution for number of turns per conversation (default: uniform)",
    )
    parser.add_argument(
        "--multiround_min_turns",
        type=int,
        default=4,
        help="Min turns for uniform distribution (default: 4)",
    )
    parser.add_argument(
        "--multiround_max_turns",
        type=int,
        default=12,
        help="Max turns for uniform/zipf/poisson/lognormal (default: 12)",
    )
    parser.add_argument(
        "--multiround_turns_alpha",
        type=float,
        default=2.0,
        help="Alpha for zipf/poisson turn distribution (default: 2.0)",
    )
    parser.add_argument(
        "--multiround_turns_average",
        type=int,
        default=8,
        help="Average for lognormal turn distribution (default: 8)",
    )

    # -------------------------------------------------------------------------
    # Input token distribution
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--multiround_input_tokens_distribution",
        type=str,
        default="uniform",
        choices=["uniform", "constant", "zipf", "poisson", "lognormal"],
        help="Distribution for input tokens per turn (default: uniform)",
    )
    parser.add_argument(
        "--multiround_min_input_tokens",
        type=int,
        default=50,
        help="Min input tokens for uniform distribution (default: 50)",
    )
    parser.add_argument(
        "--multiround_max_input_tokens",
        type=int,
        default=200,
        help="Max input tokens (default: 200)",
    )
    parser.add_argument(
        "--multiround_input_tokens_alpha",
        type=float,
        default=2.0,
        help="Alpha for zipf/poisson input token distribution (default: 2.0)",
    )
    parser.add_argument(
        "--multiround_input_tokens_average",
        type=int,
        default=100,
        help="Average for lognormal input token distribution (default: 100)",
    )

    # -------------------------------------------------------------------------
    # Output token distribution
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--multiround_output_tokens_distribution",
        type=str,
        default="uniform",
        choices=["uniform", "constant", "zipf", "poisson", "lognormal"],
        help="Distribution for output tokens per turn (default: uniform)",
    )
    parser.add_argument(
        "--multiround_min_output_tokens",
        type=int,
        default=50,
        help="Min output tokens for uniform distribution (default: 50)",
    )
    parser.add_argument(
        "--multiround_max_output_tokens",
        type=int,
        default=200,
        help="Max output tokens (default: 200)",
    )
    parser.add_argument(
        "--multiround_output_tokens_alpha",
        type=float,
        default=2.0,
        help="Alpha for zipf/poisson output token distribution (default: 2.0)",
    )
    parser.add_argument(
        "--multiround_output_tokens_average",
        type=int,
        default=100,
        help="Average for lognormal output token distribution (default: 100)",
    )

    # -------------------------------------------------------------------------
    # Prefix token distribution (context per conversation)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--multiround_prefix_distribution",
        type=str,
        default="lognormal",
        choices=["uniform", "constant", "zipf", "poisson", "lognormal"],
        help="Distribution for prefix tokens (default: lognormal)",
    )
    parser.add_argument(
        "--multiround_prefix_min",
        type=int,
        default=100,
        help="Min prefix tokens for uniform distribution (default: 100)",
    )
    parser.add_argument(
        "--multiround_prefix_max",
        type=int,
        default=5000,
        help="Max prefix tokens (default: 5000)",
    )
    parser.add_argument(
        "--multiround_prefix_alpha",
        type=float,
        default=2.0,
        help="Alpha for zipf/poisson prefix distribution (default: 2.0)",
    )
    parser.add_argument(
        "--multiround_prefix_average",
        type=int,
        default=1000,
        help="Average for lognormal prefix distribution (default: 1000)",
    )

    # -------------------------------------------------------------------------
    # User delay distribution (simulates user thinking/typing time)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--multiround_user_delay_enabled",
        action="store_true",
        default=True,
        help="Enable user response delays (default: True)",
    )
    parser.add_argument(
        "--multiround_user_delay_disabled",
        action="store_true",
        help="Disable user response delays",
    )
    parser.add_argument(
        "--multiround_user_delay_distribution",
        type=str,
        default="exponential",
        choices=["exponential", "poisson", "constant", "uniform"],
        help="Distribution for user delay (default: exponential)",
    )
    parser.add_argument(
        "--multiround_user_delay_lambda",
        type=float,
        default=1.0,
        help="Lambda/mean for exponential/poisson delay in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--multiround_user_delay_constant",
        type=float,
        default=1.0,
        help="Constant delay value in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--multiround_user_delay_min",
        type=float,
        default=0.5,
        help="Min delay for uniform distribution in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--multiround_user_delay_max",
        type=float,
        default=2.0,
        help="Max delay for uniform distribution in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--multiround_user_delay_cap",
        type=float,
        default=10.0,
        help="Maximum cap for any delay in seconds (default: 10.0)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Handle user delay enabled/disabled logic
    if hasattr(args, "multiround_user_delay_disabled") and args.multiround_user_delay_disabled:
        args.multiround_user_delay_enabled = False

    # Select benchmarks based on enable flags
    benchmarks = []
    if args.enable_normal_agent:
        benchmarks.append((async_agent_benchmark, "Agent-Benchmark"))
    if args.enable_burst_agent:
        benchmarks.append((async_burst_agent_benchmark, "Burst-Agent-Benchmark"))
    if args.enable_chatbot:
        benchmarks.append((async_chat_benchmark, "Chat-Benchmark"))
    if args.enable_multiround_chatbot:
        benchmarks.append((async_multiround_chat_benchmark, "Multiround-Chat-Benchmark"))

    if not benchmarks:
        print(
            "No benchmark enabled. Use --enable_normal_agent, --enable_burst_agent, "
            "--enable_chatbot, or --enable_multiround_chatbot"
        )
        sys.exit(1)

    # Create and start all benchmark threads
    threads = []
    enabled_flags = []
    if args.enable_normal_agent:
        enabled_flags.append("normal_agent")
    if args.enable_burst_agent:
        enabled_flags.append("burst_agent")
    if args.enable_chatbot:
        enabled_flags.append("chatbot")
    if args.enable_multiround_chatbot:
        enabled_flags.append("multiround_chatbot")
    print(f"Starting benchmarks: {', '.join(enabled_flags)}")
    print("=" * 60)
    for async_func, name in benchmarks:
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
    sys.stdout.flush()


if __name__ == "__main__":
    main()
