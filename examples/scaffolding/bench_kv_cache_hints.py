import argparse
import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import numpy as np
import requests
from datasets import load_dataset
from tqdm.asyncio import tqdm


@dataclass
class RequestResult:
    """Result of a single LLM inference request."""

    conversation_id: int
    turn_idx: int
    success: bool
    latency: float  # seconds
    ttft: float  # time to first token (seconds)
    output_tokens: int
    error: Optional[str] = None


@dataclass
class ConversationState:
    """Tracks the state of a multi-turn conversation."""

    conversation_id: int
    messages: list[dict] = field(default_factory=list)
    completed_turns: int = 0
    total_turns: int = 0


@dataclass
class TurnRequest:
    """Represents a single turn request to be dispatched."""

    conversation_id: int
    turn_idx: int
    user_message: dict  # The user message for this turn
    dispatch_time: float  # When this request should be dispatched (relative to start)


def send_kv_cache_hint(
    base_url: str,
    messages: list[dict],
    messages_to_retain: list[dict],
    api_key: Optional[str] = None,
    **extra_params,
) -> requests.Response:
    """Send a KV cache hint to the TensorRT-LLM endpoint.

    Args:
        base_url: Base URL of the TensorRT-LLM server.
        messages: List of message dicts representing the full conversation.
        messages_to_retain: List of message dicts to retain in the KV cache.
        api_key: Optional API key for authentication.
        **extra_params: Additional parameters to include in the request.

    Returns:
        Response from the server.
    """
    if not base_url.endswith("/"):
        base_url += "/"
    url = base_url + "v1/kv_cache_hints"

    headers = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    kv_cache_hint_params = {
        "action": "truncate",
        "messages": messages,
        "messages_to_retain": messages_to_retain,
    }
    kv_cache_hint_params.update(extra_params)

    return requests.post(
        url,
        json=kv_cache_hint_params,
        headers=headers,
    )


async def send_kv_cache_hint_async(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict],
    messages_to_retain: list[dict],
    model: str,
    api_key: Optional[str] = None,
) -> bool:
    """Async version: Send a KV cache hint (drop request) to the TensorRT-LLM endpoint."""
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
        "messages_to_retain": messages_to_retain,  # Empty list to drop all
    }

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            return response.status == 200
    except Exception as e:
        print(f"KV cache hint failed: {e}")
        return False


async def send_chat_completion_async(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    streaming: bool = False,
) -> tuple[bool, float, float, int, str, Optional[str]]:
    """Send an async chat completion request.

    Returns:
        Tuple of (success, latency, ttft, output_tokens, response_text, error_message)
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
        "temperature": temperature,
        "stream": streaming,
    }

    start_time = time.perf_counter()
    ttft = 0.0
    output_tokens = 0
    first_token_received = False
    response_text = ""

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            # await fetch_metrics(session=session, base_url=base_url)
            # print(f"Metrics: {metrics}")

            if response.status != 200:
                error_text = await response.text()
                return False, 0, 0, 0, "", f"HTTP {response.status}: {error_text}"

            if streaming:
                async for line in response.content:
                    if line:
                        line_str = line.decode("utf-8").strip()
                        if line_str == "data: [DONE]":
                            break
                        if line_str.startswith("data: "):
                            if not first_token_received:
                                ttft = time.perf_counter() - start_time
                                first_token_received = True
                            try:
                                data = json.loads(line_str[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        response_text += delta["content"]
                                        output_tokens += 1
                            except json.JSONDecodeError:
                                pass
            else:
                response_json = await response.json()
                ttft = time.perf_counter() - start_time
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    response_text = (
                        response_json["choices"][0].get("message", {}).get("content", "")
                    )
                    output_tokens = len(response_text.split())  # Rough estimate

        latency = time.perf_counter() - start_time
        return True, latency, ttft, output_tokens, response_text, None

    except asyncio.TimeoutError:
        return False, 0, 0, 0, "", "Timeout"
    except Exception as e:
        return False, 0, 0, 0, "", str(e)


def build_dataset(num_conversations: int, zipfian_alpha: float, max_turns: int):
    """Build a dataset of multi-turn conversations from GSM8K questions.

    Args:
        num_conversations: Number of multi-turn conversations to generate.
        zipfian_alpha: Alpha parameter for Zipfian distribution (controls turn count distribution).
        max_turns: Maximum number of turns allowed per conversation.

    Returns:
        List of conversations, where each conversation is a list of user messages (questions).
    """
    dataset = load_dataset("gsm8k", "main", split="train")
    questions = [example["question"] for example in dataset]

    # The number of turns for each conversation follows a Zipfian distribution
    # Keep sampling until we have enough valid turn counts
    turns = []
    while len(turns) < num_conversations:
        samples = np.random.zipf(zipfian_alpha, num_conversations * 2)
        valid_samples = samples[(samples >= 1) & (samples <= max_turns)]
        turns.extend(valid_samples.tolist())
    turns = turns[:num_conversations]

    # Synthesize multi-turn conversations by sampling questions from GSM8K
    conversations = []
    question_idx = 0

    for num_turns in turns:
        conversation = []
        for _ in range(num_turns):
            # Wrap around if we run out of questions
            question = questions[question_idx % len(questions)]
            conversation.append({"role": "user", "content": question})
            question_idx += 1
        conversations.append(conversation)

    return conversations


def create_turn_requests(
    conversations: list[list[dict]],
    request_rate: float,
) -> list[TurnRequest]:
    """Create a list of turn requests with Poisson-distributed dispatch times.

    Each turn from each conversation is treated as a distinct request.
    The dispatch times follow a Poisson process (exponential inter-arrival times).

    Args:
        conversations: List of conversations, each containing user messages.
        request_rate: Average number of requests per second (lambda for Poisson).

    Returns:
        List of TurnRequest objects sorted by dispatch time.
    """
    requests = []

    # Flatten all turns from all conversations into individual requests
    for conv_id, conversation in enumerate(conversations):
        for turn_idx, user_message in enumerate(conversation):
            requests.append(
                TurnRequest(
                    conversation_id=conv_id,
                    turn_idx=turn_idx,
                    user_message=user_message,
                    dispatch_time=0.0,  # Will be assigned later
                )
            )

    # Shuffle to ensure random interleaving across conversations
    np.random.shuffle(requests)

    # Assign Poisson-distributed dispatch times
    # Inter-arrival times follow exponential distribution with mean 1/lambda
    if request_rate == float("inf"):
        # All requests dispatched immediately
        for req in requests:
            req.dispatch_time = 0.0
    else:
        current_time = 0.0
        for req in requests:
            req.dispatch_time = current_time
            # Exponential inter-arrival time (Poisson process)
            interval = np.random.exponential(1.0 / request_rate)
            current_time += interval

    # Sort by dispatch time
    requests.sort(key=lambda r: r.dispatch_time)

    return requests


async def run_benchmark(
    base_url: str,
    model: str,
    conversations: list[list[dict]],
    request_rate: float,
    max_tokens: int,
    temperature: float,
    use_kv_cache_drop: bool,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> tuple[list[RequestResult], float]:
    """Run the benchmark with the specified configuration.

    Args:
        base_url: Base URL of the TensorRT-LLM server.
        model: Model name.
        conversations: List of multi-turn conversations.
        request_rate: Request rate (requests per second).
        max_tokens: Maximum tokens per response.
        temperature: Sampling temperature.
        use_kv_cache_drop: Whether to send KV cache drop requests.
        api_key: Optional API key.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (list of RequestResult, total benchmark duration).
    """
    # Create turn requests with Poisson timing
    turn_requests = create_turn_requests(conversations, request_rate)

    # Track conversation states
    conv_states: dict[int, ConversationState] = {}
    for conv_id, conversation in enumerate(conversations):
        conv_states[conv_id] = ConversationState(
            conversation_id=conv_id,
            messages=[{"role": "system", "content": "You are a helpful assistant."}],
            total_turns=len(conversation),
        )

    # Track pending requests per conversation (to ensure turn order)
    pending_turns: dict[int, asyncio.Queue] = defaultdict(asyncio.Queue)
    results: list[RequestResult] = []
    results_lock = asyncio.Lock()

    # Create aiohttp session with timeout
    connector = aiohttp.TCPConnector(limit=100)
    timeout_config = aiohttp.ClientTimeout(total=timeout)

    async def process_conversation(conv_id: int, session: aiohttp.ClientSession):
        """Process all turns for a single conversation sequentially."""
        state = conv_states[conv_id]

        while state.completed_turns < state.total_turns:
            # Wait for next turn request
            turn_req = await pending_turns[conv_id].get()

            # Build full message history for this turn
            messages = state.messages.copy()
            messages.append(turn_req.user_message)

            # Send chat completion
            (
                success,
                latency,
                ttft,
                output_tokens,
                response_text,
                error,
            ) = await send_chat_completion_async(
                session=session,
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
            )

            result = RequestResult(
                conversation_id=conv_id,
                turn_idx=turn_req.turn_idx,
                success=success,
                latency=latency,
                ttft=ttft,
                output_tokens=output_tokens,
                error=error,
            )

            async with results_lock:
                results.append(result)

            if success:
                # Update conversation state with user message and assistant response
                state.messages.append(turn_req.user_message)
                state.messages.append({"role": "assistant", "content": response_text})

            state.completed_turns += 1

            # If this was the last turn and KV cache drop is enabled, send drop request
            if use_kv_cache_drop and state.completed_turns == state.total_turns:
                await send_kv_cache_hint_async(
                    session=session,
                    base_url=base_url,
                    messages=state.messages,
                    messages_to_retain=[],  # Drop all
                    model=model,
                    api_key=api_key,
                )

    async def dispatch_requests(session: aiohttp.ClientSession):
        """Dispatch requests according to Poisson timing."""
        start_time = time.perf_counter()

        for turn_req in tqdm(turn_requests, desc="Dispatching requests"):
            # Wait until dispatch time
            elapsed = time.perf_counter() - start_time
            wait_time = turn_req.dispatch_time - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Queue the turn for its conversation
            await pending_turns[turn_req.conversation_id].put(turn_req)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_config) as session:
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

    return results, benchmark_duration


def compute_metrics(results: list[RequestResult], duration: float) -> dict:
    """Compute benchmark metrics from results."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        return {
            "total_requests": len(results),
            "successful_requests": 0,
            "failed_requests": len(failed),
            "throughput_rps": 0,
            "mean_latency_ms": 0,
            "p50_latency_ms": 0,
            "p90_latency_ms": 0,
            "p99_latency_ms": 0,
            "mean_ttft_ms": 0,
            "p50_ttft_ms": 0,
            "p90_ttft_ms": 0,
            "p99_ttft_ms": 0,
            "total_output_tokens": 0,
            "tokens_per_second": 0,
        }

    latencies = [r.latency * 1000 for r in successful]  # Convert to ms
    ttfts = [r.ttft * 1000 for r in successful]
    total_tokens = sum(r.output_tokens for r in successful)

    return {
        "total_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "throughput_rps": len(successful) / duration,
        "mean_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p90_latency_ms": np.percentile(latencies, 90),
        "p99_latency_ms": np.percentile(latencies, 99),
        "mean_ttft_ms": np.mean(ttfts),
        "p50_ttft_ms": np.percentile(ttfts, 50),
        "p90_ttft_ms": np.percentile(ttfts, 90),
        "p99_ttft_ms": np.percentile(ttfts, 99),
        "total_output_tokens": total_tokens,
        "tokens_per_second": total_tokens / duration,
    }


def print_metrics(metrics: dict, label: str):
    """Pretty print benchmark metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Total Requests:      {metrics['total_requests']}")
    print(f"  Successful:          {metrics['successful_requests']}")
    print(f"  Failed:              {metrics['failed_requests']}")
    print(f"  Throughput:          {metrics['throughput_rps']:.2f} req/s")
    print(f"  Tokens/sec:          {metrics['tokens_per_second']:.2f}")
    print("  ─────────────────────────────────────")
    print("  Latency (ms):")
    print(f"    Mean:              {metrics['mean_latency_ms']:.2f}")
    print(f"    P50:               {metrics['p50_latency_ms']:.2f}")
    print(f"    P90:               {metrics['p90_latency_ms']:.2f}")
    print(f"    P99:               {metrics['p99_latency_ms']:.2f}")
    print("  ─────────────────────────────────────")
    print("  TTFT (ms):")
    print(f"    Mean:              {metrics['mean_ttft_ms']:.2f}")
    print(f"    P50:               {metrics['p50_ttft_ms']:.2f}")
    print(f"    P90:               {metrics['p90_ttft_ms']:.2f}")
    print(f"    P99:               {metrics['p99_ttft_ms']:.2f}")
    print(f"{'=' * 60}\n")


async def run_single_benchmark(args):
    """Run a single benchmark (either baseline or with KV cache drop)."""
    mode_label = "KV CACHE DROP" if args.use_kv_cache_drop else "BASELINE"

    print("\n" + "=" * 60)
    print("  Building Dataset")
    print("=" * 60)

    conversations = build_dataset(
        num_conversations=args.num_conversations,
        zipfian_alpha=args.zipfian_alpha,
        max_turns=args.max_turns,
    )

    total_turns = sum(len(c) for c in conversations)
    print(f"  Conversations: {len(conversations)}")
    print(f"  Total turns:   {total_turns}")
    print(f"  Request rate:  {args.request_rate} req/s")
    print(f"  Mode:          {mode_label}")

    # Run benchmark
    print("\n" + "=" * 60)
    print(f"  Running {mode_label} Benchmark")
    print("=" * 60)

    results, duration = await run_benchmark(
        base_url=args.base_url,
        model=args.model,
        conversations=conversations,
        request_rate=args.request_rate,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_kv_cache_drop=args.use_kv_cache_drop,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    metrics = compute_metrics(results, duration)
    print_metrics(metrics, f"{mode_label} Results")

    # Save results to JSON
    if args.output:
        results_data = {
            "config": {
                "num_conversations": args.num_conversations,
                "zipfian_alpha": args.zipfian_alpha,
                "max_turns": args.max_turns,
                "request_rate": args.request_rate,
                "max_tokens": args.max_tokens,
                "model": args.model,
                "use_kv_cache_drop": args.use_kv_cache_drop,
                "seed": args.seed,
            },
            "metrics": metrics,
            "duration_seconds": duration,
        }
        with open(args.output, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\n  Results saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT-LLM with KV cache hints. "
        "Run separately for baseline and KV cache drop modes to ensure fair comparison."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/",
        help="Base URL of the TensorRT-LLM server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-20b",
        help="Model name to use for inference",
    )
    parser.add_argument(
        "--use-kv-cache-drop",
        action="store_true",
        help="Enable explicit KV cache drop at end of each conversation. "
        "Without this flag, runs baseline mode (no KV cache management).",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=50,
        help="Number of multi-turn conversations to generate",
    )
    parser.add_argument(
        "--zipfian-alpha",
        type=float,
        default=1.5,
        help="Alpha parameter for Zipfian distribution (turn count)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum number of turns per conversation",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=5.0,
        help="Request rate (requests per second) for Poisson distribution",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Run the benchmark
    asyncio.run(run_single_benchmark(args))


if __name__ == "__main__":
    main()
