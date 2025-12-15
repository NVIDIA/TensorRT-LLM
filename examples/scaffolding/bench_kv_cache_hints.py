"""Benchmark TensorRT-LLM with KV cache hints for multi-turn conversations.

This script benchmarks multi-turn conversation performance, with optional
KV cache management (truncate hints at conversation end).

Enhanced with infrastructure from benchmark_serving.py for comprehensive
metrics, configurable traffic patterns, and detailed result analysis.

Usage:
    # Baseline mode (no KV cache management)
    python bench_kv_cache_hints.py --model <model_name> --num-conversations 100

    # With KV cache drop enabled
    python bench_kv_cache_hints.py --model <model_name> --use-kv-cache-drop

    # With custom traffic pattern
    python bench_kv_cache_hints.py --model <model_name> --request-rate 10 --burstiness 0.5
"""

import argparse
import asyncio
import gc
import json
import os
import random
import sys
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

import aiohttp
import numpy as np
import requests
from datasets import load_dataset
from tqdm.asyncio import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

MILLISECONDS_TO_SECONDS_CONVERSION = 1000
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RequestFuncOutput:
    """Output from a single chat completion request."""

    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # Inter-token latencies
    tpot: float = 0.0  # Average time per output token
    prompt_len: int = 0
    error: str = ""
    exception_type: Optional[str] = None


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics matching benchmark_serving.py."""

    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]
    tput_user: float
    # Multi-turn specific metrics
    total_conversations: int = 0
    completed_conversations: int = 0
    mean_turns_per_conversation: float = 0.0


@dataclass
class ConversationState:
    """Tracks the state of a multi-turn conversation."""

    conversation_id: int
    messages: list[dict] = field(default_factory=list)
    completed_turns: int = 0
    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@dataclass
class TurnRequest:
    """Represents a single turn request to be dispatched."""

    conversation_id: int
    turn_idx: int
    user_message: dict  # The user message for this turn
    dispatch_time: float  # When this request should be dispatched (relative to start)


@dataclass
class TurnResult:
    """Result of a single turn in a conversation."""

    conversation_id: int
    turn_idx: int
    success: bool
    latency: float  # seconds
    ttft: float  # time to first token (seconds)
    itl: list[float] = field(default_factory=list)  # inter-token latencies
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    exception_type: Optional[str] = None


# =============================================================================
# Tokenizer Utilities
# =============================================================================


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Get tokenizer for accurate token counting."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def count_tokens(
    tokenizer: Optional[PreTrainedTokenizerBase],
    text: str,
) -> int:
    """Count tokens in text using tokenizer, or estimate by word count."""
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    # Fallback to rough word-based estimate
    return len(text.split())


def count_message_tokens(
    tokenizer: Optional[PreTrainedTokenizerBase],
    messages: list[dict],
) -> int:
    """Count total tokens in a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(tokenizer, content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    total += count_tokens(tokenizer, item["text"])
    # Add overhead for message structure (role tokens, separators)
    # This is an approximation; actual overhead depends on chat template
    total += len(messages) * 4
    return total


# =============================================================================
# KV Cache Hint Functions
# =============================================================================


def send_kv_cache_hint(
    base_url: str,
    messages: list[dict],
    messages_to_retain: list[dict],
    model: str,
    api_key: Optional[str] = None,
    **extra_params,
) -> requests.Response:
    """Send a KV cache hint to the TensorRT-LLM endpoint (synchronous).

    Args:
        base_url: Base URL of the TensorRT-LLM server.
        messages: List of message dicts representing the full conversation.
        messages_to_retain: List of message dicts to retain in the KV cache.
        model: Model name.
        api_key: Optional API key for authentication.
        **extra_params: Additional parameters to include in the request.

    Returns:
        Response from the server.
    """
    if not base_url.endswith("/"):
        base_url += "/"
    url = base_url + "v1/kv_cache_hints"

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "action": "truncate",
        "messages": messages,
        "messages_to_retain": messages_to_retain,
    }
    payload.update(extra_params)

    return requests.post(url, json=payload, headers=headers)


async def send_kv_cache_hint_async(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict],
    messages_to_retain: list[dict],
    model: str,
    api_key: Optional[str] = None,
) -> bool:
    """Async: Send a KV cache hint (truncate request) to the TensorRT-LLM endpoint."""
    if not base_url.endswith("/"):
        base_url += "/"
    url = base_url + "v1/kv_cache_hints"

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "action": "truncate",
        "messages": messages,
        "messages_to_retain": messages_to_retain,
    }

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            return response.status == 200
    except Exception as e:
        print(f"KV cache hint failed: {e}")
        return False


# =============================================================================
# Chat Completion Function
# =============================================================================


async def send_chat_completion_async(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    extra_body: Optional[dict] = None,
    api_key: Optional[str] = None,
    streaming: bool = True,
    ignore_eos: bool = False,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    """Send an async chat completion request with ITL tracking.

    Returns:
        RequestFuncOutput with detailed timing information.
    """
    if not base_url.endswith("/"):
        base_url += "/"
    url = base_url + "v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": streaming,
    }
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    if ignore_eos:
        payload["ignore_eos"] = ignore_eos
    if extra_body:
        payload.update(extra_body)

    output = RequestFuncOutput()
    generated_text = ""
    ttft = 0.0
    st = time.perf_counter()
    most_recent_timestamp = st

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                if streaming:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk == "[DONE]":
                            break

                        try:
                            data = json.loads(chunk)
                            if choices := data.get("choices"):
                                content = choices[0].get("delta", {}).get("content")
                                timestamp = time.perf_counter()

                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft
                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""
                                most_recent_timestamp = timestamp

                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens", 0)
                                output.prompt_len = usage.get("prompt_tokens", 0)
                        except json.JSONDecodeError:
                            pass

                    output.success = ttft > 0.0
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st

                    if not output.success:
                        output.error = "Never received valid chunk to calculate TTFT"
                else:
                    content = await response.content.read()
                    data = json.loads(content.decode())
                    output.success = True
                    output.generated_text = data["choices"][0]["message"]["content"]
                    output.output_tokens = data["usage"]["completion_tokens"]
                    output.prompt_len = data["usage"]["prompt_tokens"]
                    output.latency = time.perf_counter() - st
                    output.ttft = output.latency  # Non-streaming: TTFT = latency
                    output.itl = []
            else:
                output.error = f"HTTP {response.status}: {response.reason or ''}"
                output.success = False

    except asyncio.TimeoutError:
        output.success = False
        output.error = "Request timeout"
        output.exception_type = "TimeoutError"
    except Exception as e:
        output.success = False
        exc_info = sys.exc_info()
        output.error = "".join(traceback.format_exception(*exc_info))
        output.exception_type = e.__class__.__name__

    if pbar:
        pbar.update(1)

    return output


# =============================================================================
# Dataset Building
# =============================================================================


def build_dataset(
    num_conversations: int,
    zipfian_alpha: float,
    max_turns: int,
    min_turns: int = 1,
) -> list[list[dict]]:
    """Build a dataset of multi-turn conversations from GSM8K questions.

    Args:
        num_conversations: Number of multi-turn conversations to generate.
        zipfian_alpha: Alpha parameter for Zipfian distribution (controls turn count).
        max_turns: Maximum number of turns allowed per conversation.
        min_turns: Minimum number of turns per conversation.

    Returns:
        List of conversations, where each conversation is a list of user messages.
    """
    dataset = load_dataset("gsm8k", "main", split="train")
    questions = [example["question"] for example in dataset]

    # Sample turn counts from Zipfian distribution
    turns = []
    while len(turns) < num_conversations:
        samples = np.random.zipf(zipfian_alpha, num_conversations * 2)
        valid_samples = samples[(samples >= min_turns) & (samples <= max_turns)]
        turns.extend(valid_samples.tolist())
    turns = turns[:num_conversations]

    # Synthesize multi-turn conversations by sampling questions from GSM8K
    conversations = []
    question_idx = 0

    for num_turns in turns:
        conversation = []
        for _ in range(int(num_turns)):
            question = questions[question_idx % len(questions)]
            conversation.append({"role": "user", "content": question})
            question_idx += 1
        conversations.append(conversation)

    return conversations


# =============================================================================
# Request Scheduling
# =============================================================================


def create_turn_requests(
    conversations: list[list[dict]],
    request_rate: float,
    burstiness: float = 1.0,
) -> list[TurnRequest]:
    """Create turn requests with gamma-distributed inter-arrival times.

    Args:
        conversations: List of conversations, each containing user messages.
        request_rate: Average requests per second (lambda for Poisson).
        burstiness: Burstiness factor (1.0 = Poisson, <1 = more bursty, >1 = more uniform).

    Returns:
        List of TurnRequest objects sorted by dispatch time.
    """
    requests_list = []

    # Flatten all turns from all conversations into individual requests
    for conv_id, conversation in enumerate(conversations):
        for turn_idx, user_message in enumerate(conversation):
            requests_list.append(
                TurnRequest(
                    conversation_id=conv_id,
                    turn_idx=turn_idx,
                    user_message=user_message,
                    dispatch_time=0.0,
                )
            )

    # Shuffle to ensure random interleaving across conversations
    random.shuffle(requests_list)

    # Assign dispatch times using gamma distribution
    if request_rate == float("inf"):
        for req in requests_list:
            req.dispatch_time = 0.0
    else:
        assert burstiness > 0, f"Burstiness must be positive, got {burstiness}"
        theta = 1.0 / (request_rate * burstiness)
        current_time = 0.0
        for req in requests_list:
            req.dispatch_time = current_time
            interval = np.random.gamma(shape=burstiness, scale=theta)
            current_time += interval

    # Sort by dispatch time
    requests_list.sort(key=lambda r: r.dispatch_time)

    return requests_list


# =============================================================================
# Metrics Calculation
# =============================================================================


def calculate_metrics(
    results: list[TurnResult],
    duration: float,
    conversations: list[list[dict]],
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> BenchmarkMetrics:
    """Calculate comprehensive benchmark metrics."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Track errors by type
    error_counts: dict[str, int] = {}
    for r in failed:
        exc_type = r.exception_type or "Unknown"
        error_counts[exc_type] = error_counts.get(exc_type, 0) + 1

    if error_counts:
        print("\nError breakdown:")
        for exc_type, count in error_counts.items():
            print(f"  {exc_type}: {count} requests")
        print(f"Total failed requests: {len(failed)}")

    total_input = sum(r.input_tokens for r in successful)
    total_output = sum(r.output_tokens for r in successful)

    # Collect timing metrics
    ttfts = [r.ttft for r in successful if r.ttft > 0]
    e2els = [r.latency for r in successful]
    itls = []
    tpots = []
    all_tpots = []
    tput_user = []

    for r in successful:
        itls.extend(r.itl)
        tpot = 0.0
        if r.output_tokens > 1:
            latency_minus_ttft = r.latency - r.ttft
            tpot = latency_minus_ttft / (r.output_tokens - 1)
            tpots.append(tpot)
        all_tpots.append(tpot)
        if r.latency > 0:
            tput_user.append(r.output_tokens / r.latency)

    # Calculate goodput
    good_completed = 0
    if goodput_config_dict and successful:
        for i, r in enumerate(successful):
            is_good = True
            if "ttft" in goodput_config_dict:
                if r.ttft > goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION:
                    is_good = False
            if "tpot" in goodput_config_dict and i < len(all_tpots):
                if all_tpots[i] > goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION:
                    is_good = False
            if "e2el" in goodput_config_dict:
                if r.latency > goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION:
                    is_good = False
            if is_good:
                good_completed += 1

    # Count completed conversations
    conv_completed = defaultdict(int)
    for r in successful:
        conv_completed[r.conversation_id] += 1
    completed_conversations = sum(
        1 for conv_id, count in conv_completed.items() if count == len(conversations[conv_id])
    )

    if not successful:
        warnings.warn(
            "All requests failed. Check benchmark arguments and server status.",
            stacklevel=2,
        )
        # Return zero metrics
        return BenchmarkMetrics(
            completed=0,
            total_input=0,
            total_output=0,
            request_throughput=0,
            request_goodput=0,
            output_throughput=0,
            total_token_throughput=0,
            mean_ttft_ms=0,
            median_ttft_ms=0,
            std_ttft_ms=0,
            percentiles_ttft_ms=[(p, 0) for p in selected_percentiles],
            mean_tpot_ms=0,
            median_tpot_ms=0,
            std_tpot_ms=0,
            percentiles_tpot_ms=[(p, 0) for p in selected_percentiles],
            mean_itl_ms=0,
            median_itl_ms=0,
            std_itl_ms=0,
            percentiles_itl_ms=[(p, 0) for p in selected_percentiles],
            mean_e2el_ms=0,
            median_e2el_ms=0,
            std_e2el_ms=0,
            percentiles_e2el_ms=[(p, 0) for p in selected_percentiles],
            tput_user=0,
            total_conversations=len(conversations),
            completed_conversations=0,
            mean_turns_per_conversation=0,
        )

    return BenchmarkMetrics(
        completed=len(successful),
        total_input=total_input,
        total_output=total_output,
        request_throughput=len(successful) / duration,
        request_goodput=good_completed / duration,
        output_throughput=total_output / duration,
        total_token_throughput=(total_input + total_output) / duration,
        mean_ttft_ms=float(np.mean(ttfts or [0])) * 1000,
        median_ttft_ms=float(np.median(ttfts or [0])) * 1000,
        std_ttft_ms=float(np.std(ttfts or [0])) * 1000,
        percentiles_ttft_ms=[
            (p, float(np.percentile(ttfts or [0], p)) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=float(np.mean(tpots or [0])) * 1000,
        median_tpot_ms=float(np.median(tpots or [0])) * 1000,
        std_tpot_ms=float(np.std(tpots or [0])) * 1000,
        percentiles_tpot_ms=[
            (p, float(np.percentile(tpots or [0], p)) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=float(np.mean(itls or [0])) * 1000,
        median_itl_ms=float(np.median(itls or [0])) * 1000,
        std_itl_ms=float(np.std(itls or [0])) * 1000,
        percentiles_itl_ms=[
            (p, float(np.percentile(itls or [0], p)) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=float(np.mean(e2els or [0])) * 1000,
        median_e2el_ms=float(np.median(e2els or [0])) * 1000,
        std_e2el_ms=float(np.std(e2els or [0])) * 1000,
        percentiles_e2el_ms=[
            (p, float(np.percentile(e2els or [0], p)) * 1000) for p in selected_percentiles
        ],
        tput_user=float(np.mean(tput_user or [0])),
        total_conversations=len(conversations),
        completed_conversations=completed_conversations,
        mean_turns_per_conversation=len(results) / len(conversations) if conversations else 0,
    )


def print_metrics(metrics: BenchmarkMetrics, selected_percentile_metrics: list[str]):
    """Print comprehensive benchmark metrics."""
    print("{s:{c}^{n}}".format(s=" Multi-Turn Benchmark Result ", n=60, c="="))
    print(
        "{:<45} {:<10}".format(
            "Total requests:", metrics.completed + (metrics.total_conversations * 0)
        )
    )  # placeholder
    print("{:<45} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<45} {:<10}".format("Total conversations:", metrics.total_conversations))
    print("{:<45} {:<10}".format("Completed conversations:", metrics.completed_conversations))
    print(
        "{:<45} {:<10.2f}".format("Mean turns/conversation:", metrics.mean_turns_per_conversation)
    )
    print("{:<45} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<45} {:<10}".format("Total output tokens:", metrics.total_output))
    print("{:<45} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    if metrics.request_goodput > 0:
        print("{:<45} {:<10.2f}".format("Request goodput (req/s):", metrics.request_goodput))
    print("{:<45} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print(
        "{:<45} {:<10.2f}".format("Total token throughput (tok/s):", metrics.total_token_throughput)
    )
    print("{:<45} {:<10.2f}".format("User throughput (tok/s):", metrics.tput_user))

    def print_metric_section(attr_name: str, display_name: str, header: str):
        if attr_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=header, n=60, c="-"))
        print(
            "{:<45} {:<10.2f}".format(
                f"Mean {display_name} (ms):",
                getattr(metrics, f"mean_{attr_name}_ms"),
            )
        )
        print(
            "{:<45} {:<10.2f}".format(
                f"Median {display_name} (ms):",
                getattr(metrics, f"median_{attr_name}_ms"),
            )
        )
        print(
            "{:<45} {:<10.2f}".format(
                f"Std {display_name} (ms):",
                getattr(metrics, f"std_{attr_name}_ms"),
            )
        )
        for p, value in getattr(metrics, f"percentiles_{attr_name}_ms"):
            p_str = str(int(p)) if int(p) == p else str(p)
            print("{:<45} {:<10.2f}".format(f"P{p_str} {display_name} (ms):", value))

    print_metric_section("ttft", "TTFT", "Time to First Token")
    print_metric_section("tpot", "TPOT", "Time per Output Token (excl. 1st)")
    print_metric_section("itl", "ITL", "Inter-token Latency")
    print_metric_section("e2el", "E2EL", "End-to-end Latency")

    print("=" * 60)


# =============================================================================
# Performance Metrics Fetching
# =============================================================================


async def fetch_perf_metrics(base_url: str) -> list:
    """Fetch performance metrics from the /perf_metrics endpoint."""
    if not base_url.endswith("/"):
        base_url += "/"
    perf_url = f"{base_url}perf_metrics"

    async with aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.get(perf_url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"Failed to fetch perf metrics. Status: {response.status}")
                    return []
        except Exception as e:
            print(f"Error fetching perf metrics: {e}")
            return []


def calculate_kv_cache_hit_ratio(perf_metrics: list) -> dict:
    """Calculate KV cache hit ratio from performance metrics.

    Args:
        perf_metrics: List of performance metric entries from /perf_metrics endpoint.

    Returns:
        Dictionary with hit ratio statistics.
    """
    hit_ratios = []
    for entry in perf_metrics:
        if "perf_metrics" not in entry:
            continue
        kv_metrics = entry.get("perf_metrics", {}).get("kv_cache_metrics", {})
        num_reused = kv_metrics.get("num_reused_blocks", 0)
        num_missed = kv_metrics.get("num_missed_blocks", 0)

        total = num_reused + num_missed
        if total > 0:
            hit_ratio = num_reused / total
        else:
            hit_ratio = 0.0

        hit_ratios.append(hit_ratio)

    if not hit_ratios:
        return {
            "total_entries": 0,
            "mean_hit_ratio": 0.0,
            "min_hit_ratio": 0.0,
            "max_hit_ratio": 0.0,
        }

    return {
        "total_entries": len(hit_ratios),
        "mean_hit_ratio": sum(hit_ratios) / len(hit_ratios),
        "min_hit_ratio": min(hit_ratios),
        "max_hit_ratio": max(hit_ratios),
    }


def print_kv_cache_hit_ratio(perf_metrics: list):
    """Print KV cache hit ratio statistics."""
    stats = calculate_kv_cache_hit_ratio(perf_metrics)

    print("\n" + "=" * 60)
    print("  KV Cache Hit Ratio Statistics")
    print("=" * 60)
    print(f"Total entries: {stats['total_entries']}")
    print(f"Mean hit ratio: {stats['mean_hit_ratio']:.4f}")
    print(f"Min hit ratio: {stats['min_hit_ratio']:.4f}")
    print(f"Max hit ratio: {stats['max_hit_ratio']:.4f}")
    print("=" * 60)


# =============================================================================
# Main Benchmark Logic
# =============================================================================


async def run_benchmark(
    base_url: str,
    model: str,
    conversations: list[list[dict]],
    request_rate: float,
    burstiness: float,
    max_tokens: int,
    extra_body: dict,
    use_kv_cache_drop: bool,
    api_key: Optional[str] = None,
    timeout: float = 600.0,
    max_concurrency: Optional[int] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    streaming: bool = True,
    ignore_eos: bool = False,
    disable_tqdm: bool = False,
) -> tuple[list[TurnResult], float]:
    """Run the multi-turn benchmark.

    Args:
        base_url: Base URL of the TensorRT-LLM server.
        model: Model name.
        conversations: List of multi-turn conversations.
        request_rate: Request rate (requests per second).
        burstiness: Burstiness factor for request distribution.
        max_tokens: Maximum tokens per response.
        extra_body: Additional parameters for chat completion.
        use_kv_cache_drop: Whether to send KV cache drop requests.
        api_key: Optional API key.
        timeout: Request timeout in seconds.
        max_concurrency: Maximum concurrent requests (None = unlimited).
        tokenizer: Tokenizer for accurate token counting.
        streaming: Whether to use streaming responses.
        ignore_eos: Whether to ignore EOS token.
        disable_tqdm: Whether to disable progress bar.

    Returns:
        Tuple of (list of TurnResult, total benchmark duration).
    """
    # Create turn requests with traffic pattern
    turn_requests = create_turn_requests(conversations, request_rate, burstiness)
    total_turns = len(turn_requests)

    # Print traffic info
    if burstiness == 1.0:
        distribution = "Poisson process"
    else:
        distribution = "Gamma distribution"
    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency or 'unlimited'}")

    # Track conversation states
    conv_states: dict[int, ConversationState] = {}
    system_message = {"role": "system", "content": "You are a helpful assistant."}
    for conv_id, conversation in enumerate(conversations):
        conv_states[conv_id] = ConversationState(
            conversation_id=conv_id,
            messages=[system_message.copy()],
            total_turns=len(conversation),
        )

    # Track pending requests per conversation (to ensure turn order)
    pending_turns: dict[int, asyncio.Queue] = defaultdict(asyncio.Queue)
    results: list[TurnResult] = []
    results_lock = asyncio.Lock()

    # Setup semaphore for concurrency limiting
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    # Setup aiohttp session
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, force_close=True)
    timeout_config = aiohttp.ClientTimeout(total=timeout)

    # Progress bar
    pbar = None if disable_tqdm else tqdm(total=total_turns, desc="Benchmarking")

    async def process_turn_with_semaphore(
        conv_id: int,
        turn_req: TurnRequest,
        state: ConversationState,
        session: aiohttp.ClientSession,
    ) -> TurnResult:
        """Process a single turn, optionally with semaphore."""
        if semaphore:
            async with semaphore:
                return await process_turn(conv_id, turn_req, state, session)
        return await process_turn(conv_id, turn_req, state, session)

    async def process_turn(
        conv_id: int,
        turn_req: TurnRequest,
        state: ConversationState,
        session: aiohttp.ClientSession,
    ) -> TurnResult:
        """Process a single turn in the conversation."""
        # Build full message history for this turn
        messages = state.messages.copy()
        messages.append(turn_req.user_message)

        # Count input tokens
        input_tokens = count_message_tokens(tokenizer, messages)

        # Send chat completion
        output = await send_chat_completion_async(
            session=session,
            base_url=base_url,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
            api_key=api_key,
            streaming=streaming,
            ignore_eos=ignore_eos,
            pbar=pbar,
        )

        # Use server-reported token counts if available
        if output.prompt_len > 0:
            input_tokens = output.prompt_len

        result = TurnResult(
            conversation_id=conv_id,
            turn_idx=turn_req.turn_idx,
            success=output.success,
            latency=output.latency,
            ttft=output.ttft,
            itl=output.itl,
            input_tokens=input_tokens,
            output_tokens=output.output_tokens or count_tokens(tokenizer, output.generated_text),
            error=output.error,
            exception_type=output.exception_type,
        )

        return result, output.generated_text, output.success

    async def process_conversation(conv_id: int, session: aiohttp.ClientSession):
        """Process all turns for a single conversation sequentially."""
        state = conv_states[conv_id]

        while state.completed_turns < state.total_turns:
            # Wait for next turn request
            turn_req = await pending_turns[conv_id].get()

            result, response_text, success = await process_turn_with_semaphore(
                conv_id, turn_req, state, session
            )

            async with results_lock:
                results.append(result)

            if success:
                # Update conversation state with user message and assistant response
                state.messages.append(turn_req.user_message)
                state.messages.append({"role": "assistant", "content": response_text})
                state.total_input_tokens += result.input_tokens
                state.total_output_tokens += result.output_tokens

            state.completed_turns += 1

            # If last turn and KV cache drop enabled, send drop request
            if use_kv_cache_drop and state.completed_turns == state.total_turns:
                await send_kv_cache_hint_async(
                    session=session,
                    base_url=base_url,
                    messages=state.messages,
                    messages_to_retain=[system_message],
                    model=model,
                    api_key=api_key,
                )

    async def dispatch_requests(session: aiohttp.ClientSession):
        """Dispatch requests according to configured traffic pattern."""
        start_time = time.perf_counter()

        for turn_req in turn_requests:
            # Wait until dispatch time
            elapsed = time.perf_counter() - start_time
            wait_time = turn_req.dispatch_time - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Queue the turn for its conversation
            await pending_turns[turn_req.conversation_id].put(turn_req)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout_config,
        trust_env=True,
    ) as session:
        benchmark_start = time.perf_counter()

        # Start conversation processors
        conv_tasks = [
            asyncio.create_task(process_conversation(conv_id, session))
            for conv_id in conv_states.keys()
        ]

        # Start dispatcher
        dispatch_task = asyncio.create_task(dispatch_requests(session))

        # Wait for dispatch to complete
        await dispatch_task

        # Wait for all conversations to complete
        await asyncio.gather(*conv_tasks)

        benchmark_duration = time.perf_counter() - benchmark_start

    if pbar is not None:
        pbar.close()

    return results, benchmark_duration


async def run_initial_test(
    base_url: str,
    model: str,
    max_tokens: int,
    extra_body: dict,
    api_key: Optional[str],
    streaming: bool,
) -> bool:
    """Run initial test to validate configuration."""
    print("Starting initial single prompt test run...")

    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout_config = aiohttp.ClientTimeout(total=120.0)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout_config,
        trust_env=True,
    ) as session:
        output = await send_chat_completion_async(
            session=session,
            base_url=base_url,
            model=model,
            messages=test_messages,
            max_tokens=max_tokens,
            extra_body=extra_body,
            api_key=api_key,
            streaming=streaming,
        )

        if not output.success:
            raise ValueError(
                f"Initial test run failed - Please check benchmark arguments. Error: {output.error}"
            )

        print("Initial test run completed. Starting main benchmark run...")
        return True


# =============================================================================
# Goodput Configuration
# =============================================================================


def parse_goodput(slo_pairs: list[str]) -> dict[str, float]:
    """Parse goodput SLO configuration."""
    goodput_config = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format for SLO. Use 'KEY:VALUE' pairs (e.g., 'ttft:100')."
        ) from err
    return goodput_config


def check_goodput_args(args) -> dict[str, float]:
    """Check and parse goodput arguments."""
    goodput_config = {}
    valid_names = ["ttft", "tpot", "e2el"]

    if args.goodput:
        goodput_config = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config.items():
            if slo_name not in valid_names:
                raise ValueError(f"Invalid SLO metric '{slo_name}'. Must be one of {valid_names}.")
            if slo_val < 0:
                raise ValueError(f"SLO value must be non-negative: {slo_name}={slo_val}")

    return goodput_config


# =============================================================================
# Result Saving
# =============================================================================


def save_results(
    args: argparse.Namespace,
    metrics: BenchmarkMetrics,
    duration: float,
    results: list[TurnResult],
) -> Optional[str]:
    """Save benchmark results to JSON file."""
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")

    result_json: dict[str, Any] = {
        "date": current_dt,
        "model": args.model,
        "num_conversations": args.num_conversations,
        "max_turns": args.max_turns,
        "zipfian_alpha": args.zipfian_alpha,
        "use_kv_cache_drop": args.use_kv_cache_drop,
        "request_rate": args.request_rate if args.request_rate < float("inf") else "inf",
        "burstiness": args.burstiness,
        "max_concurrency": args.max_concurrency,
        "seed": args.seed,
        "duration": duration,
        "completed": metrics.completed,
        "total_conversations": metrics.total_conversations,
        "completed_conversations": metrics.completed_conversations,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "user_throughput": metrics.tput_user,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "std_ttft_ms": metrics.std_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "std_tpot_ms": metrics.std_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "std_itl_ms": metrics.std_itl_ms,
        "mean_e2el_ms": metrics.mean_e2el_ms,
        "median_e2el_ms": metrics.median_e2el_ms,
        "std_e2el_ms": metrics.std_e2el_ms,
    }

    # Add percentiles
    for p, value in metrics.percentiles_ttft_ms:
        p_str = str(int(p)) if int(p) == p else str(p)
        result_json[f"p{p_str}_ttft_ms"] = value
    for p, value in metrics.percentiles_tpot_ms:
        p_str = str(int(p)) if int(p) == p else str(p)
        result_json[f"p{p_str}_tpot_ms"] = value
    for p, value in metrics.percentiles_itl_ms:
        p_str = str(int(p)) if int(p) == p else str(p)
        result_json[f"p{p_str}_itl_ms"] = value
    for p, value in metrics.percentiles_e2el_ms:
        p_str = str(int(p)) if int(p) == p else str(p)
        result_json[f"p{p_str}_e2el_ms"] = value

    # Add metadata if provided
    if args.metadata:
        for item in args.metadata:
            if "=" in item:
                key, value = item.split("=", 1)
                result_json[key.strip()] = value.strip()

    # Add detailed per-request data if requested
    if args.save_detailed:
        result_json["turn_results"] = [
            {
                "conversation_id": r.conversation_id,
                "turn_idx": r.turn_idx,
                "success": r.success,
                "latency": r.latency,
                "ttft": r.ttft,
                "itl": r.itl,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "error": r.error,
            }
            for r in results
        ]

    # Determine filename
    base_model_id = args.model.split("/")[-1]
    mode_str = "kv_drop" if args.use_kv_cache_drop else "baseline"
    concurrency_str = f"-concurrency{args.max_concurrency}" if args.max_concurrency else ""

    if args.result_filename:
        file_name = args.result_filename
    else:
        file_name = f"multiturn-{mode_str}-{args.request_rate}qps{concurrency_str}-{base_model_id}-{current_dt}.json"

    if args.result_dir:
        os.makedirs(args.result_dir, exist_ok=True)
        file_name = os.path.join(args.result_dir, file_name)

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    print(f"\nResults saved to: {file_name}")
    return file_name


# =============================================================================
# Main Entry Point
# =============================================================================


async def async_main(args: argparse.Namespace):
    """Async main function."""
    mode_label = "KV CACHE DROP" if args.use_kv_cache_drop else "BASELINE"

    # Initialize tokenizer if specified
    tokenizer = None
    if args.tokenizer:
        print(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = get_tokenizer(
            args.tokenizer,
            tokenizer_mode=args.tokenizer_mode,
            trust_remote_code=args.trust_remote_code,
        )

    # Build sampling parameters
    extra_body = {}
    if args.temperature is not None:
        extra_body["temperature"] = args.temperature
    if args.top_p is not None:
        extra_body["top_p"] = args.top_p
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if "temperature" not in extra_body:
        extra_body["temperature"] = 0.0  # Default to greedy

    # Parse goodput config
    goodput_config = check_goodput_args(args)

    # Run initial test
    if not args.no_test_input:
        await run_initial_test(
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
            api_key=args.api_key,
            streaming=not args.non_streaming,
        )
    else:
        print("Skipping initial test run.")

    # Build dataset
    print("\n" + "=" * 60)
    print("  Building Dataset")
    print("=" * 60)

    conversations = build_dataset(
        num_conversations=args.num_conversations,
        zipfian_alpha=args.zipfian_alpha,
        max_turns=args.max_turns,
        min_turns=args.min_turns,
    )

    total_turns = sum(len(c) for c in conversations)
    print(f"  Conversations: {len(conversations)}")
    print(f"  Total turns:   {total_turns}")
    print(f"  Mode:          {mode_label}")

    # Run benchmark
    print("\n" + "=" * 60)
    print(f"  Running {mode_label} Benchmark")
    print("=" * 60)

    # Disable GC for more consistent timing
    gc.disable()

    results, duration = await run_benchmark(
        base_url=args.base_url,
        model=args.model,
        conversations=conversations,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        max_tokens=args.max_tokens,
        extra_body=extra_body,
        use_kv_cache_drop=args.use_kv_cache_drop,
        api_key=args.api_key,
        timeout=args.timeout,
        max_concurrency=args.max_concurrency,
        tokenizer=tokenizer,
        streaming=not args.non_streaming,
        ignore_eos=args.ignore_eos,
        disable_tqdm=args.disable_tqdm,
    )

    # Re-enable GC
    gc.enable()

    # Calculate and print metrics
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]
    selected_percentile_metrics = args.percentile_metrics.split(",")

    metrics = calculate_metrics(
        results=results,
        duration=duration,
        conversations=conversations,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config,
    )

    print_metrics(metrics, selected_percentile_metrics)
    print(f"Benchmark duration: {duration:.2f}s")

    # Save results
    if args.save_result:
        save_results(args, metrics, duration, results)

    # Fetch server perf metrics and report KV cache hit ratio
    print("\nFetching server performance metrics...")
    perf_metrics = await fetch_perf_metrics(args.base_url)
    if perf_metrics:
        # Report KV cache hit ratio
        print_kv_cache_hit_ratio(perf_metrics)

        # Save perf metrics to file if requested
        if args.save_request_time_breakdown:
            current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            base_model_id = args.model.split("/")[-1]
            perf_filename = (
                f"multiturn-{args.request_rate}qps-{base_model_id}-{current_dt}-perf_metrics.json"
            )
            if args.result_dir:
                perf_filename = os.path.join(args.result_dir, perf_filename)
            with open(perf_filename, "w", encoding="utf-8") as f:
                json.dump(perf_metrics, f, indent=2)
            print(f"Server perf metrics saved to: {perf_filename}")
    else:
        print("Failed to fetch server performance metrics.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT-LLM with KV cache hints for multi-turn conversations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Server configuration
    server_group = parser.add_argument_group("server options")
    server_group.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the TensorRT-LLM server",
    )
    server_group.add_argument(
        "--model",
        type=str,
        default="gpt-oss-20b",
        help="Model name to use for inference",
    )
    server_group.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication",
    )
    server_group.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Request timeout in seconds",
    )

    # KV cache options
    kv_group = parser.add_argument_group("kv cache options")
    kv_group.add_argument(
        "--use-kv-cache-drop",
        action="store_true",
        help="Enable explicit KV cache drop at end of each conversation",
    )

    # Dataset options
    dataset_group = parser.add_argument_group("dataset options")
    dataset_group.add_argument(
        "--num-conversations",
        type=int,
        default=100,
        help="Number of multi-turn conversations to generate",
    )
    dataset_group.add_argument(
        "--zipfian-alpha",
        type=float,
        default=1.5,
        help="Alpha parameter for Zipfian distribution (turn count)",
    )
    dataset_group.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of turns per conversation",
    )
    dataset_group.add_argument(
        "--min-turns",
        type=int,
        default=1,
        help="Minimum number of turns per conversation",
    )

    # Traffic options
    traffic_group = parser.add_argument_group("traffic options")
    traffic_group.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (requests/second). inf = all at once",
    )
    traffic_group.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor. 1.0 = Poisson, <1 = bursty, >1 = uniform",
    )
    traffic_group.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests",
    )

    # Generation options
    gen_group = parser.add_argument_group("generation options")
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per response",
    )
    gen_group.add_argument(
        "--non-streaming",
        action="store_true",
        help="Disable streaming mode",
    )
    gen_group.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS token during generation",
    )

    # Sampling options
    sampling_group = parser.add_argument_group("sampling options")
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling parameter",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter",
    )

    # Tokenizer options
    tokenizer_group = parser.add_argument_group("tokenizer options")
    tokenizer_group.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path for accurate token counting",
    )
    tokenizer_group.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow"],
        help="Tokenizer mode",
    )
    tokenizer_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )

    # Metrics options
    metrics_group = parser.add_argument_group("metrics options")
    metrics_group.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-separated list of metrics for percentile reporting",
    )
    metrics_group.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,90,99",
        help="Comma-separated list of percentiles to report",
    )
    metrics_group.add_argument(
        "--goodput",
        nargs="+",
        default=None,
        help="SLO config for goodput as KEY:VALUE pairs (e.g., ttft:100 e2el:5000)",
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--save-result",
        action="store_true",
        help="Save benchmark results to JSON file",
    )
    output_group.add_argument(
        "--save-detailed",
        action="store_true",
        help="Include per-request details in saved results",
    )
    output_group.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory to save result files",
    )
    output_group.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Custom filename for results",
    )
    output_group.add_argument(
        "--metadata",
        nargs="*",
        metavar="KEY=VALUE",
        help="Additional metadata to include in results",
    )
    output_group.add_argument(
        "--save-request-time-breakdown",
        action="store_true",
        help="Fetch and save server-side performance metrics",
    )

    # Misc options
    misc_group = parser.add_argument_group("misc options")
    misc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    misc_group.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bar",
    )
    misc_group.add_argument(
        "--no-test-input",
        action="store_true",
        help="Skip initial test run",
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run benchmark
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
