# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict comparisons shared by the sparse offload model tests and their CPU guards."""

import json
import time
from pathlib import Path

# One complete short-split document per LongBench v2 domain. Selection uses prompt
# length, not model answers. Hash the benchmark's formatted 0-shot text before the
# model-specific chat template so a changed dataset cannot silently change CI.
LONGBENCH_CASES = {
    "66f37eb9821e116aacb2d295": (
        "624f1fc15b907b72538d17a19dfb6d41b39900e788005a18393db15ad36db6c1",
        "B",
    ),
    "66f9625fbb02136c067c5456": (
        "2a6b255a62f364b928631ba69a385377e89260078a57b5d1280cc7d1342baa39",
        "B",
    ),
    "66f91f2cbb02136c067c4b1e": (
        "e0876657d6b27c6106f9976eda7ef061754e8c4e645782642574dd4fbb4d45e5",
        "B",
    ),
    "66ed2c87821e116aacb1f149": (
        "3c6356927dbfc6e0502befa1bc09234346ec442916b560bfd284fdcc10e5b04f",
        "A",
    ),
    "6713066fbb02136c067d3214": (
        "341f1742e596e4d10a6e0ce709fb38c6c2abc2f5ff9ac3ba2d1474baa64e3136",
        "D",
    ),
    "6708a096bb02136c067d1789": (
        "63ce30ecd635782a918b314a0bb57519128b69117811822f5d1ea9991740fe18",
        "D",
    ),
}


def assert_token_parity(reference: dict, candidate: dict, *, label: str) -> None:
    """Compare paired completions independently of benchmark answer scoring."""
    assert reference["id"] == candidate["id"], f"{label}: sample IDs differ"
    assert reference["prompt_sha256"] == candidate["prompt_sha256"], (
        f"{label}: tokenized prompts differ"
    )
    left, right = reference["token_ids"], candidate["token_ids"]
    assert left and right, f"{label}: empty completion"
    mismatch = next(
        (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)),
    )
    assert left == right, (
        f"{label}: first different token at {mismatch}; "
        f"resident={left[mismatch : mismatch + 8]}, candidate={right[mismatch : mismatch + 8]}, "
        f"lengths={len(left)}/{len(right)}"
    )
    assert reference["finish_reason"] == candidate["finish_reason"], (
        f"{label}: finish reasons differ"
    )


def make_boundary_prompts(
    prompts: list[list[int]], *, cache_tokens: int, tokens_per_block: int
) -> list[tuple[str, list[int]]]:
    """Build token-prefix continuations around cache-block boundaries.

    These continuations test KV layout and decoding, without assigning benchmark
    answers to incomplete documents. Every history exceeds the GPU cache capacity.
    """
    boundaries = (cache_tokens + 2 * tokens_per_block, 8192, 16384)
    lengths = [boundary + offset for boundary in boundaries for offset in (-1, 0, 1)]
    assert all(boundary % tokens_per_block == 0 for boundary in boundaries)
    assert min(lengths) // tokens_per_block > cache_tokens // tokens_per_block
    assert prompts and max(map(len, prompts)) >= max(lengths), (
        f"Boundary coverage requires a source prompt of at least {max(lengths)} tokens"
    )
    cases = []
    for index, length in enumerate(lengths):
        sources = [prompt for prompt in prompts if len(prompt) >= length]
        cases.append((f"boundary-{length}", sources[index % len(sources)][:length]))
    return cases


class DecodeBatchGate:
    """Start a test wave with all histories ready, then permit normal retirement."""

    def __init__(self) -> None:
        self._wave: int | None = None
        self._opened = False
        self._waiting_since: float | None = None

    def select(
        self,
        *,
        wave: int,
        expected_lengths: list[int],
        request_lengths: list[int],
        decoding_iters: list[int],
    ) -> list[int]:
        """Return ready request positions in the wave's declared order."""
        if wave != self._wave:
            self._wave = wave
            self._opened = False
            self._waiting_since = None
        assert len(request_lengths) == len(decoding_iters), (
            f"{len(request_lengths)} request lengths for {len(decoding_iters)} decode counters"
        )
        if len(expected_lengths) < 2 or not request_lengths:
            return list(range(len(request_lengths)))
        assert len(request_lengths) <= len(expected_lengths), (
            f"Unexpected requests in parity wave {wave}: {request_lengths}; expected {expected_lengths}"
        )
        assert all(length in expected_lengths for length in request_lengths), (
            f"Unexpected prompt lengths: {request_lengths}; expected {expected_lengths}"
        )
        if not self._opened:
            assert all(iteration <= 1 for iteration in decoding_iters), (
                "Parity batch started decoding before all histories were ready: "
                f"decode iterations {decoding_iters} for lengths {request_lengths}"
            )
            if len(request_lengths) < len(expected_lengths):
                now = time.monotonic()
                if self._waiting_since is None:
                    self._waiting_since = now
                assert now - self._waiting_since < 30, (
                    f"Timed out waiting for the parity batch of wave {wave}: "
                    f"{request_lengths} of {expected_lengths}"
                )
                return []
            assert sorted(request_lengths) == sorted(expected_lengths), (
                f"Parity wave {wave} lengths {request_lengths}; expected {expected_lengths}"
            )
            self._opened = True
        return sorted(
            range(len(request_lengths)), key=lambda i: expected_lengths.index(request_lengths[i])
        )


def assert_concurrent_decode(
    stats: list[dict],
    *,
    min_batched_steps: int,
    require_retirement: bool = False,
    expected_initial_batch_size: int | None = None,
) -> list[int]:
    """Require sustained overlap and, optionally, decode after a request retires."""
    active_counts = [
        stat.get("inflightBatchingStats", {}).get("numGenRequests", 0) for stat in stats
    ]
    batched = [i for i, count in enumerate(active_counts) if count > 1]
    assert len(batched) >= min_batched_steps, f"Only {len(batched)} batched decode iterations"
    if require_retirement:
        assert 1 in active_counts[batched[-1] + 1 :], (
            "Mixed-length wave did not decode after one request retired: "
            f"active requests per step {active_counts}"
        )
    trace = [count for count in active_counts if count > 0]
    if expected_initial_batch_size is not None:
        assert trace[0] == expected_initial_batch_size, (
            f"Parity wave started with {trace[0]} of {expected_initial_batch_size} requests"
        )
    print(
        f"Concurrent decode evidence: {len(batched)} batched steps, retirement={require_retirement}"
    )
    return trace


def assert_host_fetches(
    events: list[dict],
    *,
    prompt_lengths: list[int],
    world_size: int,
    index_topk: int,
    unit_tokens: int,
) -> None:
    """Require fetch participation and positive native misses on every rank.

    Counters are per native Batch. Serial requests get individual evidence;
    concurrent requests have overlapping counter intervals, which must not be
    summed or interpreted as individual request miss counts.
    """
    assert prompt_lengths and all(
        length // unit_tokens > index_topk for length in prompt_lengths
    ), "Prompts do not activate offload"
    assert {event["rank"] for event in events} == set(range(world_size)), (
        f"Expected ranks 0..{world_size - 1}; observed {[event['rank'] for event in events]}"
    )
    for rank in range(world_size):
        rows = [event for event in events if event["rank"] == rank]
        assert len(rows) == len(prompt_lengths), f"rank {rank}: missing or duplicate fetch evidence"
        # Concurrent requests may complete in either order. Match initial host
        # history lengths, allowing the first decode token to have been appended.
        rows.sort(key=lambda event: event["history_tokens"])
        for row, length in zip(rows, sorted(prompt_lengths), strict=True):
            assert abs(row["history_tokens"] - length) <= 1, (
                f"rank {rank}: host history {row['history_tokens']} does not match prompt {length}"
            )
            assert row["index_topk"] == index_topk and row["unit_tokens"] == unit_tokens
            assert any(
                abs(units - length // unit_tokens) <= 1 for units in row["selectable_units"]
            ), (
                f"rank {rank}: selectable history {row['selectable_units']} does not match prompt {length}"
            )
            assert row["misses"] > 0, f"rank {rank}: no native host KV fetches"


def read_fetch_events(directory: Path) -> list[dict]:
    """Read complete observer records; an in-progress final write is retried."""
    events = []
    for path in sorted(directory.glob("rank-*.jsonl")):
        lines = path.read_text().splitlines(keepends=True)
        for line in lines:
            if line.endswith("\n"):
                events.append(json.loads(line))
    return events


def wait_for_fetch_events(directory: Path, *, previous: int, expected: int) -> list[dict]:
    """Wait for request cleanup, which can follow delivery of the final token."""
    deadline = time.monotonic() + 30
    while True:
        events = read_fetch_events(directory)
        if len(events) >= previous + expected:
            return events
        assert time.monotonic() < deadline, (
            f"Only {len(events) - previous}/{expected} native offload records arrived"
        )
        time.sleep(0.1)
