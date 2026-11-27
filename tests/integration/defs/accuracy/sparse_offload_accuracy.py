# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict comparisons shared by the sparse offload model tests and their CPU guards."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

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
    """Reject empty, truncated, reordered, or unequal paired completions."""
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
    assert reference["answer"] == candidate["answer"], f"{label}: extracted answers differ"
    assert reference["correct"] == candidate["correct"], f"{label}: answer correctness differs"


def assert_forced_token_parity(
    reference: dict,
    candidate: dict,
    *,
    ordinary_reference: dict,
    ordinary_candidate: dict,
    label: str,
) -> int | None:
    """Require ordinary-prefix and metadata equality; return the first suffix difference."""
    assert_token_parity(ordinary_reference, ordinary_candidate, label=f"{label} ordinary")
    left, right = reference["token_ids"], candidate["token_ids"]
    assert len(left) == len(right), f"{label}: forced completion lengths differ"
    prefix = ordinary_reference["token_ids"]
    for forced in (reference, candidate):
        assert forced["id"] == ordinary_reference["id"], f"{label}: ordinary sample ID differs"
        assert forced["prompt_sha256"] == ordinary_reference["prompt_sha256"], (
            f"{label}: ordinary tokenized prompt differs"
        )
        assert forced["token_ids"][: len(prefix)] == prefix, f"{label}: valid prefix differs"
    if ordinary_reference["finish_reason"] != "stop":
        assert_token_parity(reference, candidate, label=label)
        return None
    first_diff = next(
        (
            offset
            for offset, (before, after) in enumerate(zip(left, right, strict=True))
            if before != after
        ),
        None,
    )
    difference = (
        None
        if first_diff is None
        else {
            "offset": first_diff,
            "resident_token": left[first_diff],
            "offload_token": right[first_diff],
        }
    )
    print(f"First post-stop token diff {reference['id']}: {difference}")
    assert_token_parity(
        {**reference, "token_ids": left[: len(prefix)]},
        {**candidate, "token_ids": right[: len(prefix)]},
        label=label,
    )
    return first_diff


class SparseHistoryVerifier:
    """Compare selected host-history bytes and track observed arena-slot turnover."""

    def __init__(self) -> None:
        self.checked_rows = 0
        self.mismatched_rows = 0
        self.evictions = 0
        self.refetches = 0
        self._slots: dict[int, dict[int, int]] = {}
        self._evicted: dict[int, set[int]] = {}
        self._checked_positions: dict[str, list[int]] = {}

    def check_tensors(
        self,
        layer: int,
        units: torch.Tensor,
        slots: torch.Tensor,
        host_view: torch.Tensor,
        host_pages: list[int],
        arena: torch.Tensor,
        *,
        token_position: int,
    ) -> None:
        """Gather uint8 rows from host [page, unit, byte] and attention [slot, byte] buffers."""
        import torch

        units_per_page = host_view.shape[1]
        assert bool((slots < arena.shape[0]).all())
        logical_pages = units // units_per_page
        assert int(logical_pages.max()) < len(host_pages)
        physical_pages = torch.tensor(host_pages, dtype=torch.int64)[logical_pages]
        assert bool((physical_pages >= 0).all())
        expected = (
            host_view[physical_pages.to(arena.device), (units % units_per_page).to(arena.device)]
            .cpu()
            .numpy()
            .tobytes()
        )
        actual = arena[slots.to(arena.device)].cpu().numpy().tobytes()
        self.check(
            layer,
            units.tolist(),
            slots.tolist(),
            expected,
            actual,
            row_bytes=arena.shape[1],
            token_position=token_position,
        )

    def check(
        self,
        layer: int,
        units: list[int],
        slots: list[int],
        expected: bytes,
        actual: bytes,
        *,
        row_bytes: int,
        token_position: int,
    ) -> None:
        """Check packed rows in selection order, with one physical slot per logical unit."""
        assert units and len(units) == len(slots), "Missing selected history rows"
        assert len(expected) == len(actual) == len(units) * row_bytes
        assert len(set(zip(slots, units, strict=True))) == len(set(slots)), (
            "Different history units share an attention slot"
        )
        self.checked_rows += len(units)
        if expected != actual:
            different = [
                row
                for row in range(len(units))
                if expected[row * row_bytes : (row + 1) * row_bytes]
                != actual[row * row_bytes : (row + 1) * row_bytes]
            ]
            self.mismatched_rows += len(different)
            first = different[0]
            raise AssertionError(
                f"Offload history row mismatch: layer={layer}, "
                f"unit={units[first]}, slot={slots[first]}"
            )
        self._checked_positions.setdefault(str(layer), []).append(token_position)
        previous = self._slots.setdefault(layer, {})
        evicted = self._evicted.setdefault(layer, set())
        selected = set(units)
        returned = selected & evicted
        self.refetches += len(returned)
        evicted.difference_update(returned)
        for slot, unit in zip(slots, units, strict=True):
            old_unit = previous.get(slot)
            if old_unit is not None and old_unit != unit and old_unit not in selected:
                self.evictions += 1
                evicted.add(old_unit)
            previous[slot] = unit

    def metrics(self) -> dict:
        """Return request-local layer/position coverage and observed turnover counts."""
        return {
            "checked_rows": self.checked_rows,
            "mismatched_rows": self.mismatched_rows,
            "evictions": self.evictions,
            "refetches": self.refetches,
            "checked_positions": self._checked_positions,
        }


def assert_history_integrity(
    events: list[dict],
    *,
    prompt_lengths: list[int],
    completion_lengths: list[int],
    world_size: int,
    require_eviction: bool,
) -> None:
    """Reconcile every configured offload layer and decode position per request/rank.

    Disaggregated context samples the first output token. Generation attention
    consumes the remaining output tokens' predecessors, starting at prompt length.
    """
    expected = sorted(zip(prompt_lengths, completion_lengths, strict=True))
    assert expected and events, "No offload integrity evidence"
    assert {event["rank"] for event in events} == set(range(world_size)), (
        "Missing rank integrity evidence"
    )
    for rank in range(world_size):
        rows = sorted(
            (event for event in events if event["rank"] == rank),
            key=lambda event: event["history_tokens"],
        )
        assert len(rows) == len(expected), f"rank {rank}: missing or duplicate integrity evidence"
        for event, (prompt_length, completion_length) in zip(rows, expected, strict=True):
            label = f"rank {rank}, request {event['request_id']}"
            assert abs(event["history_tokens"] - prompt_length) <= 1, (
                f"{label}: integrity history does not match prompt"
            )
            metrics = event.get("integrity")
            assert metrics is not None and metrics["checked_rows"] > 0, (
                f"{label}: native attention history integrity check did not run"
            )
            assert metrics["mismatched_rows"] == 0, "Offload history rows differ"
            layers = event["offload_layers"]
            coverage = metrics["checked_positions"]
            assert layers and set(coverage) == {str(layer) for layer in layers}, (
                f"{label}: incomplete offload layers"
            )
            positions = list(range(prompt_length, prompt_length + completion_length - 1))
            for layer, checked in coverage.items():
                assert checked == positions, f"{label}, layer {layer}: incomplete decode steps"
            if require_eviction:
                assert metrics["evictions"] > 0, "No observed sparse-cache eviction"
                assert metrics["refetches"] > 0, "No observed sparse-history re-fetch"


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
