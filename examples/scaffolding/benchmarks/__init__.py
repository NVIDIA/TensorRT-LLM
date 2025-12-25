"""Scaffolding benchmarks package.

Usage:
    python -m examples.scaffolding.benchmarks --model_dir /path/to/model ...
"""

from .agent_benchmark import async_agent_benchmark, async_burst_agent_benchmark
from .benchmark_utils import (
    load_prompts_from_json,
    print_benchmark_results,
    print_lock,
    run_async_in_thread,
    run_benchmark_in_thread,
    shutdown_llm,
)
from .chat_benchmark import async_chat_benchmark
from .multiround_chat_benchmark import async_multiround_chat_benchmark

__all__ = [
    # Agent benchmarks
    "async_agent_benchmark",
    "async_burst_agent_benchmark",
    # Chat benchmarks
    "async_chat_benchmark",
    # Multiround chat benchmarks
    "async_multiround_chat_benchmark",
    # Utilities
    "load_prompts_from_json",
    "print_benchmark_results",
    "print_lock",
    "run_async_in_thread",
    "run_benchmark_in_thread",
    "shutdown_llm",
]
