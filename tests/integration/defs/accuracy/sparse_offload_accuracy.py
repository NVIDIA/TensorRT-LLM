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
    """Require ordinary-prefix parity and report the first post-stop difference.

    Args:
        reference: Resident forced completion, including its full token list.
        candidate: Offloaded forced completion with the same request and prompt.
        ordinary_reference: Resident ordinary completion defining the valid prefix.
        ordinary_candidate: Offloaded ordinary completion; all its fields stay exact.
        label: Diagnostic prefix identifying the comparison.

    Returns:
        The first differing post-stop token offset, or None when none differs.

    Forced answer/correct fields describe the entire stress continuation, not the
    ordinary answer. Exact prefix tokens protect that answer; suffix-derived
    metadata cannot be used to reject an otherwise permitted suffix difference.
    Identical forced tokens still require identical metadata. Without an ordinary
    stop, full token and metadata equality remains required.
    """
    assert_token_parity(ordinary_reference, ordinary_candidate, label=f"{label} ordinary")
    left, right = reference["token_ids"], candidate["token_ids"]
    assert len(left) == len(right), f"{label}: forced completion lengths differ"
    prefix = ordinary_reference["token_ids"]
    assert 0 < len(prefix) <= len(left), (
        f"{label}: forced completions are empty or shorter than the ordinary completion"
    )
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
    assert reference["finish_reason"] == candidate["finish_reason"], (
        f"{label}: finish reasons differ"
    )
    if first_diff is None:
        assert_token_parity(reference, candidate, label=label)
    return first_diff


class SparseHistoryVerifier:
    """Compare selected host-history bytes and track observed arena-slot turnover."""

    def __init__(self) -> None:
        self.checked_rows = 0
        self.mismatched_rows = 0
        self.aliased_rows = 0
        self.pool_mismatches = 0
        self.evictions = 0
        self.refetches = 0
        self._slots: dict[int, dict[int, int]] = {}
        self._evicted: dict[int, set[int]] = {}
        self._checked_positions: dict[str, list[int]] = {}
        self._first_mismatch: dict[str, int] | None = None
        self._first_alias: dict[str, int] | None = None
        self._first_pool_mismatch: dict | None = None

    def check_tensors(
        self,
        layer: int,
        units: torch.Tensor,
        slots: torch.Tensor,
        host_view: torch.Tensor,
        host_pages: list[int],
        arena: torch.Tensor,
        *,
        consumer_pool: torch.Tensor,
        token_position: int,
    ) -> None:
        """Compare actual host and attention rows selected by the supplied indices.

        Args:
            layer: Configured offload layer index.
            units: CPU int64 tensor [selected_rows] of logical history units.
            slots: CPU int64 tensor [selected_rows] of final attention slots.
            host_view: uint8 tensor [physical_page, unit, byte] viewing host pages.
            host_pages: Logical-to-physical page mapping for this request.
            arena: uint8 tensor [physical_slot, byte] viewing the attention pool.
            consumer_pool: Cached tensor [physical_slot, 1, byte] returned to the
                PrimTS consumer. Layout violations fail in the worker; a different
                backing pointer with matching layout is recorded for the parent.
            token_position: Absolute decode position that selected these rows.
        """
        import torch

        units_per_page = host_view.shape[1]
        assert bool(((slots >= 0) & (slots < arena.shape[0])).all()), (
            f"layer {layer}, position {token_position}: slots outside arena of {arena.shape[0]} rows"
        )
        expected_shape = (arena.shape[0], 1, arena.shape[1])
        expected_stride = (arena.shape[1], arena.shape[1], 1)
        assert (
            consumer_pool.device == arena.device
            and consumer_pool.element_size() == 1
            and tuple(consumer_pool.shape) == expected_shape
            and consumer_pool.stride() == expected_stride
        ), (
            f"layer {layer}, position {token_position}: Attention consumer pool layout differs; "
            f"shape {tuple(consumer_pool.shape)}, strides {consumer_pool.stride()}, "
            f"element bytes {consumer_pool.element_size()}, device {consumer_pool.device}; "
            f"expected shape {expected_shape}, strides {expected_stride}, "
            f"one-byte elements on {arena.device}"
        )
        if consumer_pool.data_ptr() != arena.data_ptr():
            self.pool_mismatches += 1
            if self._first_pool_mismatch is None:
                self._first_pool_mismatch = {
                    "layer": layer,
                    "position": token_position,
                    "expected_pointer": arena.data_ptr(),
                    "consumer_pointer": consumer_pool.data_ptr(),
                    "expected_shape": [arena.shape[0], 1, arena.shape[1]],
                    "consumer_shape": list(consumer_pool.shape),
                    "consumer_stride": list(consumer_pool.stride()),
                    "consumer_element_size": consumer_pool.element_size(),
                    "expected_device": str(arena.device),
                    "consumer_device": str(consumer_pool.device),
                }
        logical_pages = units // units_per_page
        assert bool(((logical_pages >= 0) & (logical_pages < len(host_pages))).all()), (
            f"layer {layer}, position {token_position}: units outside {len(host_pages)} host pages"
        )
        physical_pages = torch.tensor(host_pages, dtype=torch.int64)[logical_pages]
        assert bool((physical_pages >= 0).all()), (
            f"layer {layer}, position {token_position}: negative physical host page"
        )
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
        """Record row comparisons and slot turnover without interrupting TP decode.

        Args:
            layer: Configured offload layer index.
            units: Logical history unit indices in selection order.
            slots: Final attention slots corresponding to units.
            expected: Packed host-reference bytes in selection order.
            actual: Packed bytes read from the corresponding attention slots.
            row_bytes: Number of bytes in one selected row.
            token_position: Absolute decode position of this comparison.

        Byte mismatches and in-bounds slot aliases are retained in metrics and
        rejected by the parent after completion, so one rank does not strand peers.
        """
        assert units and len(units) == len(slots), "Missing selected history rows"
        assert len(expected) == len(actual) == len(units) * row_bytes, (
            f"layer {layer}, position {token_position}: expected/actual byte counts "
            f"{len(expected)}/{len(actual)} do not match {len(units)} rows of {row_bytes} bytes"
        )
        selected_slots = {}
        for slot, unit in zip(slots, units, strict=True):
            previous_unit = selected_slots.get(slot)
            if previous_unit is not None and previous_unit != unit:
                self.aliased_rows += 1
                if self._first_alias is None:
                    self._first_alias = {
                        "layer": layer,
                        "unit": unit,
                        "previous_unit": previous_unit,
                        "slot": slot,
                        "position": token_position,
                    }
            selected_slots[slot] = unit
        self.checked_rows += len(units)
        if expected != actual:
            different = [
                row
                for row in range(len(units))
                if expected[row * row_bytes : (row + 1) * row_bytes]
                != actual[row * row_bytes : (row + 1) * row_bytes]
            ]
            self.mismatched_rows += len(different)
            if self._first_mismatch is None:
                first = different[0]
                self._first_mismatch = {
                    "layer": layer,
                    "unit": units[first],
                    "slot": slots[first],
                    "position": token_position,
                }
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
            "aliased_rows": self.aliased_rows,
            "pool_mismatches": self.pool_mismatches,
            "evictions": self.evictions,
            "refetches": self.refetches,
            "checked_positions": self._checked_positions,
            "first_mismatch": self._first_mismatch,
            "first_alias": self._first_alias,
            "first_pool_mismatch": self._first_pool_mismatch,
        }


def assert_history_integrity(
    events: list[dict],
    *,
    prompt_lengths: list[int],
    completion_lengths: list[int],
    world_size: int,
    require_eviction: bool,
) -> None:
    """Require distinct requests and complete FP8 offload history on each rank.

    Disaggregated context samples the first output token. Generation attention
    consumes the remaining output tokens' predecessors, starting at prompt length.

    Args:
        events: Released observer records, including opt-in offload_layers and
            integrity metrics for every request/rank.
        prompt_lengths: Input token counts for the requests in this wave.
        completion_lengths: API completion lengths paired with prompt_lengths.
        world_size: Number of generation tensor-parallel ranks.
        require_eviction: Require observed eviction and re-fetch for stress waves.
    """
    expected = sorted(zip(prompt_lengths, completion_lengths, strict=True))
    for rank, rows in _match_rank_events(
        events, prompt_lengths=prompt_lengths, world_size=world_size, kind="integrity"
    ):
        request_ids = [row["request_id"] for row in rows]
        assert len(set(request_ids)) == len(request_ids), (
            f"rank {rank}: duplicate or split request integrity evidence"
        )
        for event, (prompt_length, completion_length) in zip(rows, expected, strict=True):
            label = f"rank {rank}, request {event['request_id']}"
            metrics = event.get("integrity")
            assert metrics is not None and metrics["checked_rows"] > 0, (
                f"{label}: native attention history integrity check did not run"
            )
            assert metrics["pool_mismatches"] == 0, (
                f"{label}: Attention consumer pool differs: "
                f"first={metrics.get('first_pool_mismatch')}, "
                f"pool_mismatches={metrics['pool_mismatches']}"
            )
            assert metrics["aliased_rows"] == 0, (
                f"{label}: Different history units share an attention slot: "
                f"first={metrics.get('first_alias')}, aliased_rows={metrics['aliased_rows']}"
            )
            assert metrics["mismatched_rows"] == 0, (
                f"{label}: Offload history row mismatch: "
                f"first={metrics.get('first_mismatch')}, mismatched_rows={metrics['mismatched_rows']}"
            )
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


def _match_rank_events(
    events: list[dict], *, prompt_lengths: list[int], world_size: int, kind: str
) -> list[tuple[int, list[dict]]]:
    assert prompt_lengths and events, f"No offload {kind} evidence"
    assert {event["rank"] for event in events} == set(range(world_size)), (
        f"Missing rank {kind} evidence"
    )
    matched = []
    for rank in range(world_size):
        rows = sorted(
            (event for event in events if event["rank"] == rank),
            key=lambda event: event["history_tokens"],
        )
        assert len(rows) == len(prompt_lengths), (
            f"rank {rank}: missing or duplicate {kind} evidence"
        )
        for row, length in zip(rows, sorted(prompt_lengths), strict=True):
            assert abs(row["history_tokens"] - length) <= 1, (
                f"rank {rank}: host history {row['history_tokens']} does not match prompt {length}; "
                "one newly appended decode token is allowed"
            )
        matched.append((rank, rows))
    return matched


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
    for rank, rows in _match_rank_events(
        events, prompt_lengths=prompt_lengths, world_size=world_size, kind="fetch"
    ):
        for row, length in zip(rows, sorted(prompt_lengths), strict=True):
            assert row["index_topk"] == index_topk, (
                f"rank {rank}: index_topk {row['index_topk']} differs from {index_topk}"
            )
            assert row["unit_tokens"] == unit_tokens, (
                f"rank {rank}: unit_tokens {row['unit_tokens']} differs from {unit_tokens}"
            )
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
