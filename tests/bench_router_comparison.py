"""Micro-benchmark comparing KvCacheAwareRouter vs ConversationRouter.

Both routers receive the same **text** input so the comparison includes
KvCacheAwareRouter's tokenization cost (its main extra overhead vs
ConversationRouter which hashes raw unicode code-points).

Measures routing latency (get_next_server) across:
  1. Input length sweep  — 1k to 128k characters
  2. Multi-turn scenario — successive turns with growing prefix
  3. Session-count sweep — how ConversationRouter scales with active sessions
  4. Server-count sweep  — how both routers scale with backend count

Usage:
    python tests/bench_router_comparison.py
    python tests/bench_router_comparison.py --num-servers 8 --iters 100
    python tests/bench_router_comparison.py --sweep sessions
    python tests/bench_router_comparison.py --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import copy
import random
import statistics
import time
from collections import Counter

from transformers import AutoTokenizer

from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams
from tensorrt_llm.serve.router import ConversationRouter, KvCacheAwareRouter, block_key_hasher

# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def generate_text(length: int, seed: int = 42) -> str:
    """Generate deterministic pseudo-random text of *length* characters."""
    rng = random.Random(seed)
    # Mix of ASCII that tokenizers handle normally
    chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?\n"
    return "".join(rng.choice(chars) for _ in range(length))


def compute_block_hashes(token_ids: list[int], tokens_per_block: int) -> list[int]:
    hash_list = []
    for t in range(0, len(token_ids) - 1, tokens_per_block):
        t_end = min(t + tokens_per_block, len(token_ids) - 1)
        parent = None if t == 0 else hash_list[-1]
        hash_list.append(block_key_hasher(token_ids[t:t_end], parent))
    return hash_list


def make_text_request(text: str, model: str = "mock", conversation_id=None) -> CompletionRequest:
    """Create a text CompletionRequest used by both routers."""
    params = DisaggregatedParams(
        request_type="context_only",
        conversation_id=conversation_id,
    )
    return CompletionRequest(model=model, prompt=text, disaggregated_params=params)


def percentile(data: list[float], p: float) -> float:
    idx = min(int(len(data) * p), len(data) - 1)
    return sorted(data)[idx]


def print_header(title: str, **params):
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    if params:
        print(f"  {', '.join(f'{k}={v}' for k, v in params.items())}")
    print(f"{'=' * 90}")


def print_table_header(columns: list[tuple[str, int]]):
    header = "  ".join(f"{name:>{width}s}" for name, width in columns)
    sep = "  ".join("-" * width for _, width in columns)
    print(f"\n{header}")
    print(f"{sep}")


def print_row(values: list[tuple], columns: list[tuple[str, int]]):
    parts = []
    for (val, fmt), (_, width) in zip(values, columns):
        parts.append(f"{val:>{width}{fmt}}")
    print("  ".join(parts))


# ────────────────────────────────────────────────────────────────────
# Router factories
# ────────────────────────────────────────────────────────────────────


def make_kv_router(
    servers: list[str], tokenizer_name: str, tokens_per_block: int = 32
) -> KvCacheAwareRouter:
    """Create a KvCacheAwareRouter with a pre-loaded tokenizer.

    The tokenizer is injected into the router's cache so that the first
    call doesn't include download/load overhead, but every subsequent
    ``get_next_server`` call still goes through the real tokenization
    path.
    """
    router = KvCacheAwareRouter(
        servers=servers,
        tokens_per_block=tokens_per_block,
        max_batch_size=64,
    )
    # Pre-load tokenizer into the router's cache — avoids measuring
    # one-time AutoTokenizer.from_pretrained() cost.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    router._tokenizers[tokenizer_name] = tokenizer
    return router


def make_conv_router(servers: list[str], tokens_per_block: int = 32) -> ConversationRouter:
    return ConversationRouter(
        server_role=None,
        servers=servers,
        tokens_per_block=tokens_per_block,
    )


# ────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ────────────────────────────────────────────────────────────────────


async def time_routing(router, request, warmup: int, iters: int) -> list[float]:
    """Run warmup + timed iterations and return latencies in ms."""
    for _ in range(warmup):
        req = copy.copy(request)
        await router.get_next_server(req)
        await router.finish_request(req)

    lats = []
    srvs = []
    for _ in range(iters):
        req = copy.copy(request)
        t0 = time.perf_counter()
        srv, _ = await router.get_next_server(req)
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000)
        srvs.append(srv)
        await router.finish_request(req)
    return lats, srvs


# ────────────────────────────────────────────────────────────────────
# Benchmark 1: Input-length sweep
# ────────────────────────────────────────────────────────────────────


async def bench_length_sweep(
    num_servers: int,
    tokenizer_name: str,
    tokens_per_block: int,
    hit_ratio: float,
    warmup: int,
    iters: int,
):
    """Compare routing latency as input length grows.

    Both routers receive the same text. KvCacheAwareRouter tokenizes
    it; ConversationRouter converts to unicode code-points.
    """
    servers = [f"http://server-{i}:8000" for i in range(num_servers)]
    lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    # Pre-load tokenizer for cache pre-population
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    cols = [
        ("chars", 8),
        ("kv_mean_ms", 10),
        ("kv_p50", 7),
        ("kv_p99", 7),
        ("conv_mean_ms", 11),
        ("conv_p50", 8),
        ("conv_p99", 8),
        ("ratio", 7),
    ]

    print_header(
        "Input-length sweep (text → both routers)",
        servers=num_servers,
        tokenizer=tokenizer_name,
        tokens_per_block=tokens_per_block,
        hit_ratio=f"{hit_ratio:.0%}",
        warmup=warmup,
        iters=iters,
    )
    print_table_header(cols)

    for length in lengths:
        text = generate_text(length, seed=length)

        # ── KvCacheAwareRouter ──
        kv_router = make_kv_router(servers, tokenizer_name, tokens_per_block)
        # Pre-populate caches: tokenize the text to get block hashes
        token_ids = tokenizer.encode(text)
        block_hashes = compute_block_hashes(token_ids, tokens_per_block)
        for srv in servers:
            n_hits = int(len(block_hashes) * hit_ratio)
            kv_router._server_state[srv].add_blocks(block_hashes[:n_hits])

        kv_req = make_text_request(text, model=tokenizer_name)
        kv_lats, _ = await time_routing(kv_router, kv_req, warmup, iters)

        # ── ConversationRouter ──
        conv_router = make_conv_router(servers, tokens_per_block)
        # Pre-populate sessions so prefix matching has work to do
        for i in range(num_servers):
            sess_text = generate_text(length, seed=length + i + 1000)
            req = make_text_request(sess_text, model=tokenizer_name, conversation_id=f"sess-{i}")
            await conv_router.get_next_server(req)
            await conv_router.finish_request(req)

        conv_req = make_text_request(text, model=tokenizer_name)
        conv_lats, _ = await time_routing(conv_router, conv_req, warmup, iters)

        kv_mean = statistics.mean(kv_lats)
        conv_mean = statistics.mean(conv_lats)
        ratio = kv_mean / conv_mean if conv_mean > 0 else float("inf")

        print_row(
            [
                (length, "d"),
                (kv_mean, ".3f"),
                (percentile(kv_lats, 0.50), ".3f"),
                (percentile(kv_lats, 0.99), ".3f"),
                (conv_mean, ".3f"),
                (percentile(conv_lats, 0.50), ".3f"),
                (percentile(conv_lats, 0.99), ".3f"),
                (ratio, ".2f"),
            ],
            cols,
        )


# ────────────────────────────────────────────────────────────────────
# Benchmark 2: Multi-turn scenario
# ────────────────────────────────────────────────────────────────────


async def bench_multi_turn(
    num_servers: int, tokenizer_name: str, tokens_per_block: int, warmup: int, iters: int
):
    """Simulate a multi-turn conversation and measure routing per turn.

    Both routers receive text. KvCacheAwareRouter tokenizes each turn.
    """
    servers = [f"http://server-{i}:8000" for i in range(num_servers)]
    base_length = 4096
    turn_extension = 512
    num_turns = 8

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    cols = [
        ("turn", 5),
        ("chars", 8),
        ("kv_mean_ms", 10),
        ("kv_p50", 7),
        ("conv_mean_ms", 11),
        ("conv_p50", 8),
        ("kv_server", 10),
        ("conv_server", 11),
    ]

    print_header(
        "Multi-turn conversation (text → both routers)",
        servers=num_servers,
        tokenizer=tokenizer_name,
        base_length=base_length,
        turn_extension=turn_extension,
        turns=num_turns,
        warmup=warmup,
        iters=iters,
    )
    print_table_header(cols)

    base_text = generate_text(base_length, seed=99)

    for turn in range(num_turns):
        total_len = base_length + turn * turn_extension
        turn_text = base_text + generate_text(turn * turn_extension, seed=200 + turn)

        # ── KvCacheAwareRouter (fresh per turn for fair comparison) ──
        kv_router = make_kv_router(servers, tokenizer_name, tokens_per_block)
        if turn > 0:
            prev_len = base_length + (turn - 1) * turn_extension
            prev_text = turn_text[:prev_len]
            prev_tokens = tokenizer.encode(prev_text)
            prev_hashes = compute_block_hashes(prev_tokens, tokens_per_block)
            kv_router._server_state[servers[0]].add_blocks(prev_hashes)

        kv_req = make_text_request(turn_text, model=tokenizer_name)
        kv_lats, kv_srvs = await time_routing(kv_router, kv_req, warmup, iters)

        # ── ConversationRouter (persists across turns) ──
        if turn == 0:
            conv_router = make_conv_router(servers, tokens_per_block)

        conv_req = make_text_request(turn_text, model=tokenizer_name)
        conv_lats, conv_srvs = await time_routing(conv_router, conv_req, warmup, iters)

        kv_top = Counter(kv_srvs).most_common(1)[0][0].split("-")[-1]
        conv_top = Counter(conv_srvs).most_common(1)[0][0].split("-")[-1]

        print_row(
            [
                (turn, "d"),
                (total_len, "d"),
                (statistics.mean(kv_lats), ".3f"),
                (percentile(kv_lats, 0.50), ".3f"),
                (statistics.mean(conv_lats), ".3f"),
                (percentile(conv_lats, 0.50), ".3f"),
                (f"srv-{kv_top}", "s"),
                (f"srv-{conv_top}", "s"),
            ],
            cols,
        )


# ────────────────────────────────────────────────────────────────────
# Benchmark 3: Session-count sweep (ConversationRouter specific)
# ────────────────────────────────────────────────────────────────────


async def bench_session_sweep(num_servers: int, tokens_per_block: int, warmup: int, iters: int):
    """Measure ConversationRouter latency as the number of tracked sessions grows."""
    servers = [f"http://server-{i}:8000" for i in range(num_servers)]
    text_length = 4096
    session_counts = [0, 10, 50, 100, 500, 1000, 5000]

    cols = [
        ("sessions", 8),
        ("mean_ms", 8),
        ("p50_ms", 7),
        ("p90_ms", 7),
        ("p99_ms", 7),
    ]

    print_header(
        "ConversationRouter session-count sweep",
        servers=num_servers,
        text_length=text_length,
        warmup=warmup,
        iters=iters,
    )
    print_table_header(cols)

    for n_sessions in session_counts:
        conv_router = make_conv_router(servers, tokens_per_block)

        # Pre-populate sessions with distinct texts
        for i in range(n_sessions):
            sess_text = generate_text(text_length, seed=i + 7000)
            req = make_text_request(sess_text, conversation_id=f"s-{i}")
            await conv_router.get_next_server(req)
            await conv_router.finish_request(req)

        # Benchmark: route a new request (no session match expected)
        new_text = generate_text(text_length, seed=999999)
        bench_req = make_text_request(new_text)

        lats, _ = await time_routing(conv_router, bench_req, warmup, iters)

        print_row(
            [
                (n_sessions, "d"),
                (statistics.mean(lats), ".3f"),
                (percentile(lats, 0.50), ".3f"),
                (percentile(lats, 0.90), ".3f"),
                (percentile(lats, 0.99), ".3f"),
            ],
            cols,
        )


# ────────────────────────────────────────────────────────────────────
# Benchmark 4: Server-count sweep
# ────────────────────────────────────────────────────────────────────


async def bench_server_sweep(
    tokenizer_name: str, tokens_per_block: int, hit_ratio: float, warmup: int, iters: int
):
    """Measure how both routers scale with the number of servers."""
    text_length = 16384
    server_counts = [1, 2, 4, 8, 16, 32]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    cols = [
        ("servers", 7),
        ("kv_mean_ms", 10),
        ("kv_p50", 7),
        ("conv_mean_ms", 11),
        ("conv_p50", 8),
    ]

    print_header(
        "Server-count sweep (text → both routers)",
        text_length=text_length,
        tokenizer=tokenizer_name,
        hit_ratio=f"{hit_ratio:.0%}",
        warmup=warmup,
        iters=iters,
    )
    print_table_header(cols)

    text = generate_text(text_length, seed=text_length)
    token_ids = tokenizer.encode(text)
    block_hashes = compute_block_hashes(token_ids, tokens_per_block)

    for num_servers in server_counts:
        servers = [f"http://server-{i}:8000" for i in range(num_servers)]

        # ── KvCacheAwareRouter ──
        kv_router = make_kv_router(servers, tokenizer_name, tokens_per_block)
        for srv in servers:
            n_hits = int(len(block_hashes) * hit_ratio)
            kv_router._server_state[srv].add_blocks(block_hashes[:n_hits])

        kv_req = make_text_request(text, model=tokenizer_name)
        kv_lats, _ = await time_routing(kv_router, kv_req, warmup, iters)

        # ── ConversationRouter ──
        conv_router = make_conv_router(servers, tokens_per_block)
        for i in range(num_servers):
            sess_text = generate_text(text_length, seed=text_length + i + 1000)
            req = make_text_request(sess_text, conversation_id=f"sess-{i}")
            await conv_router.get_next_server(req)
            await conv_router.finish_request(req)

        conv_req = make_text_request(text, model=tokenizer_name)
        conv_lats, _ = await time_routing(conv_router, conv_req, warmup, iters)

        print_row(
            [
                (num_servers, "d"),
                (statistics.mean(kv_lats), ".3f"),
                (percentile(kv_lats, 0.50), ".3f"),
                (statistics.mean(conv_lats), ".3f"),
                (percentile(conv_lats, 0.50), ".3f"),
            ],
            cols,
        )


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KvCacheAwareRouter vs ConversationRouter"
    )
    parser.add_argument(
        "--num-servers", type=int, default=4, help="Number of backend servers (default: 4)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="HF tokenizer for KvCacheAwareRouter (default: gpt2)",
    )
    parser.add_argument(
        "--tokens-per-block", type=int, default=32, help="Tokens/chars per block (default: 32)"
    )
    parser.add_argument(
        "--hit-ratio", type=float, default=0.5, help="Fraction of blocks pre-cached (default: 0.5)"
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument(
        "--iters", type=int, default=50, help="Timed iterations per config (default: 50)"
    )
    parser.add_argument(
        "--sweep",
        choices=["all", "length", "multi_turn", "sessions", "servers"],
        default="all",
        help="Which benchmark(s) to run (default: all)",
    )
    args = parser.parse_args()

    # Validate tokenizer loads
    print(f"Loading tokenizer: {args.tokenizer} ...")
    AutoTokenizer.from_pretrained(args.tokenizer)
    print("OK\n")

    sweeps = {
        "length": lambda: bench_length_sweep(
            args.num_servers,
            args.tokenizer,
            args.tokens_per_block,
            args.hit_ratio,
            args.warmup,
            args.iters,
        ),
        "multi_turn": lambda: bench_multi_turn(
            args.num_servers, args.tokenizer, args.tokens_per_block, args.warmup, args.iters
        ),
        "sessions": lambda: bench_session_sweep(
            args.num_servers, args.tokens_per_block, args.warmup, args.iters
        ),
        "servers": lambda: bench_server_sweep(
            args.tokenizer, args.tokens_per_block, args.hit_ratio, args.warmup, args.iters
        ),
    }

    if args.sweep == "all":
        for fn in sweeps.values():
            await fn()
    else:
        await sweeps[args.sweep]()

    print()


if __name__ == "__main__":
    asyncio.run(main())
