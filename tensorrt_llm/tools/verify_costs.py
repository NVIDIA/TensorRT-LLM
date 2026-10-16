# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workload-independent offline GPU measurements and strict raw-cost export.

This module needs no CUDA or native TRT-LLM import to reduce existing measurements.
Its exports are not serving cost tables, accuracy certificates or admission receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Protocol

__all__ = [
    "EventLeasePool",
    "canonical_sha256",
    "parse_json",
    "sha256_file",
    "reduce_measurements",
    "write_new_json",
    "isolated_control_communicator",
    "create_private_output_directory",
]


class TimingEvent(Protocol):
    def record(self, stream=None) -> None: ...
    def query(self) -> bool: ...
    def elapsed_time(self, other: TimingEvent) -> float: ...


class ControlCommunicator(Protocol):
    def Dup(self) -> ControlCommunicator: ...
    def Free(self) -> None: ...


@contextmanager
def isolated_control_communicator(runtime: ControlCommunicator) -> Iterator[ControlCommunicator]:
    """Duplicate collectively before workers; never modify the runtime communicator.

    The caller must finish worker shutdown and evidence closure inside this
    context. Its dedicated communicator remains owned on every error path.
    """
    control = runtime.Dup()
    try:
        yield control
    finally:
        control.Free()


class EventLeasePool:
    """Keep bounded event pairs alive until completed-row readback releases them.

    The caller materializes CUDA events before capture and reads them only after
    its normal completion boundary. No per-step synchronization is introduced.
    """

    def __init__(self, event_factory: Callable[[], TimingEvent], size: int = 32) -> None:
        if type(size) is not int or not 2 <= size <= 256:
            raise ValueError("Event pool size must be in 2..256")
        self._events = [(event_factory(), event_factory()) for _ in range(size)]
        self._owners: dict[int, int] = {}
        self._ended: set[int] = set()
        self._next_id = 0
        self._initialized = False

    def initialize(self, stream=None) -> None:
        """Materialize lazy handles outside capture; caller establishes completion."""
        if self._initialized:
            raise RuntimeError("Event pool already initialized")
        for start, end in self._events:
            start.record(stream)
            end.record(stream)
        self._initialized = True

    def begin(self) -> int:
        if not self._initialized:
            raise RuntimeError("Initialize timing events before capture")
        occupied = set(self._owners.values())
        slot = next((i for i in range(len(self._events)) if i not in occupied), None)
        if slot is None:
            raise RuntimeError("No completed event lease available")
        lease = self._next_id
        self._next_id += 1
        self._owners[lease] = slot
        self._events[slot][0].record()
        return lease

    def end(self, lease: int) -> None:
        if type(lease) is not int or lease not in self._owners:
            raise RuntimeError("Unknown timing lease")
        if lease in self._ended:
            raise RuntimeError("Timing lease ended twice")
        self._events[self._owners[lease]][1].record()
        self._ended.add(lease)

    def complete(self, lease: int) -> float:
        if type(lease) is not int or lease not in self._owners:
            raise RuntimeError("Unknown timing lease")
        if lease not in self._ended:
            raise RuntimeError("Timing lease has no end boundary")
        start, end = self._events[self._owners[lease]]
        if not end.query():
            raise RuntimeError("Normal completion did not finish the timing events")
        elapsed = float(start.elapsed_time(end))
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Measured duration must be positive and finite")
        del self._owners[lease]
        self._ended.remove(lease)
        return elapsed

    @property
    def pending(self) -> int:
        return len(self._owners)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def parse_json(data: bytes | str) -> object:
    """Reject ambiguous duplicate keys and nonstandard nonfinite JSON values."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def sha256_file(path: Path) -> str:
    """Hash actual artifact bytes without loading a whole native library into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: object) -> None:
    """Create a new owner-only POSIX file; never follow/overwrite an existing target.

    No existing ACL, umask or host setting is changed. POSIX permission semantics
    need Linux qualification; Windows ACL behavior is not attested by this helper.
    """
    payload = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as stream:
        stream.write(payload)


def create_private_output_directory(path: Path) -> None:
    """Create only a new diagnostic directory, with owner-only POSIX permissions."""
    path.mkdir(mode=0o700, parents=False, exist_ok=False)


def reduce_measurements(
    documents: list[dict],
    *,
    excluded_generations: int = 1,
    warmup_samples: int = 2,
    minimum_samples: int = 24,
    minimum_samples_per_generation: int = 12,
) -> dict:
    """Join exact rank/sample identities before reducing rank maxima.

    Warm-up exclusions remain in the output. Missing ranks, unequal ordinal sets,
    mixed identities or scopes, and incomplete lifecycle closure are errors.
    Samples are not summed across ranks or across overlapping timing scopes.
    """
    for value in (excluded_generations, warmup_samples):
        if type(value) is not int or value < 0:
            raise ValueError("Warm-up exclusions must be nonnegative integers")
    if (
        type(minimum_samples) is not int
        or minimum_samples < 1
        or not documents
        or type(minimum_samples_per_generation) is not int
        or minimum_samples_per_generation < 1
    ):
        raise ValueError("A positive sample floor and raw documents are required")
    if any(not isinstance(row, dict) for row in documents):
        raise ValueError("Raw measurements must be JSON objects")
    first = documents[0]
    world = first["world_size"]
    if type(world) is not int or world < 1:
        raise ValueError("Actual world size required")
    identity = first["identity"]
    scope = first["timing_scope"]
    if not isinstance(scope, str) or not scope or not isinstance(identity, dict) or not identity:
        raise ValueError("Observed identity and explicit timing scope required")
    declared = identity.get("declared_shape_ids", [])
    if (
        not isinstance(declared, list)
        or not declared
        or any(
            not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key) for key in declared
        )
        or len(set(declared)) != len(declared)
    ):
        raise ValueError("Adapter-declared finite shape identities required")
    expected_generations = identity.get("generations_per_shape")
    if type(expected_generations) is not int or expected_generations < 1:
        raise ValueError("Declared generations per shape required")
    seen_shapes = set()
    declared_to_observed = {}
    cells: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    shapes = {}
    raw_hashes = []
    for row in documents:
        if (
            row.get("schema") != "verify-cost-raw-v1"
            or type(row["world_size"]) is not int
            or row["world_size"] != world
            or row["identity"] != identity
            or row["timing_scope"] != scope
            or row.get("generation_closed") is not True
            or row.get("run_complete") is not True
            or row.get("model_shutdown_completed") is not True
        ):
            raise ValueError("Raw source, timing scope, generation or shutdown closure differs")
        rank, generation = row["rank"], row["generation"]
        if (
            type(rank) is not int
            or not 0 <= rank < world
            or type(generation) is not int
            or not 0 <= generation < expected_generations
        ):
            raise ValueError("Invalid rank/generation identity")
        if not isinstance(row["execution_shape"], dict) or not row["execution_shape"]:
            raise ValueError("Explicit execution shape/layout required")
        key = canonical_sha256(row["execution_shape"])
        if row.get("declared_shape_id") not in declared:
            raise ValueError("Measurement is not part of the declared workload grid")
        seen_shapes.add(row["declared_shape_id"])
        observed = declared_to_observed.setdefault(row["declared_shape_id"], key)
        if observed != key:
            raise ValueError("A declared shape changed its observed execution layout")
        shapes[key] = row["execution_shape"]
        if rank in cells[key][generation]:
            raise ValueError("Duplicate cell/generation/rank")
        cells[key][generation][rank] = row
        raw_hashes.append(canonical_sha256(row))
    if seen_shapes != set(declared):
        raise ValueError("A whole declared workload shape is missing")
    results = []
    rank_ordinals = [dict() for _ in range(world)]
    for key, generations in sorted(cells.items()):
        if sorted(generations) != list(range(expected_generations)):
            raise ValueError("Declared per-cell generation coverage is incomplete")
        retained, excluded, joined_rows = [], [], []
        previous_maximum = -1
        for generation, ranks in sorted(generations.items()):
            if set(ranks) != set(range(world)):
                raise ValueError("Missing rank in a generation")
            samples = []
            for rank in range(world):
                rank_samples = {}
                if not isinstance(ranks[rank]["samples"], list) or not ranks[rank]["samples"]:
                    raise ValueError("A generation has no raw samples")
                for sample in ranks[rank]["samples"]:
                    ordinal, iteration = sample["sample_index"], sample["executor_iteration"]
                    duration = sample["elapsed_ms"]
                    if (
                        type(ordinal) is not int
                        or ordinal < 0
                        or ordinal in rank_samples
                        or type(iteration) is not int
                        or iteration < 0
                        or type(duration) not in (int, float)
                        or not math.isfinite(duration)
                        or duration <= 0
                        or type(sample["eligible"]) is not bool
                    ):
                        raise ValueError("Invalid or duplicated raw sample")
                    rank_samples[ordinal] = sample
                    if ordinal in rank_ordinals[rank]:
                        raise ValueError("A replay ordinal was reused in the run")
                    rank_ordinals[rank][ordinal] = iteration
                samples.append(rank_samples)
            ordinals = set(samples[0])
            if any(set(rank_samples) != ordinals for rank_samples in samples):
                raise ValueError("Ranks have different replay ordinal sets")
            if min(ordinals) <= previous_maximum:
                raise ValueError("Replay ordinals reset between generations")
            previous_maximum = max(ordinals)
            eligible_index = 0
            generation_retained = 0
            for ordinal in sorted(ordinals):
                joined = [samples[rank][ordinal] for rank in range(world)]
                if len({sample["executor_iteration"] for sample in joined}) != 1:
                    raise ValueError("Replay ordinal joins different executor iterations")
                duration = max(sample["elapsed_ms"] for sample in joined)
                reason = None
                if not all(sample["eligible"] for sample in joined):
                    reason = "non_plateau_or_incomplete_update"
                elif generation < excluded_generations:
                    reason = "warmup_generation"
                else:
                    if eligible_index < warmup_samples:
                        reason = "initial_plateau_sample"
                    eligible_index += 1
                detail = {
                    "generation": generation,
                    "sample_index": ordinal,
                    "executor_iteration": joined[0]["executor_iteration"],
                    "rank_elapsed_ms": [sample["elapsed_ms"] for sample in joined],
                    "rank_max_elapsed_ms": duration,
                    "excluded_reason": reason,
                }
                joined_rows.append(detail)
                (excluded if reason else retained).append(duration)
                if reason is None:
                    generation_retained += 1
            if (
                generation >= excluded_generations
                and generation_retained < minimum_samples_per_generation
            ):
                raise ValueError("Insufficient retained exact samples in a generation")
        if len(retained) < minimum_samples:
            raise ValueError("Insufficient exact retained samples for a cell")
        results.append(
            {
                "execution_shape": shapes[key],
                "retained_count": len(retained),
                "rank_max_median_ms": statistics.median(retained),
                "retained_rank_max_samples_ms": retained,
                "joins": joined_rows,
                "excluded_count": len(excluded),
            }
        )
    for ordinals in rank_ordinals:
        ordered = sorted(ordinals)
        if ordered != list(range(len(ordered))):
            raise ValueError("Whole-run replay ordinal coverage is incomplete")
        iterations = [ordinals[ordinal] for ordinal in ordered]
        if any(right <= left for left, right in zip(iterations, iterations[1:])):
            raise ValueError("Executor iterations do not advance with replay ordinals")
    return {
        "schema": "verify-cost-measured-summary-v1",
        "identity": identity,
        "timing_scope": scope,
        "cells": results,
        "raw_document_sha256": raw_hashes,
        "exclusions": {"generations": excluded_generations, "initial_samples": warmup_samples},
        "sample_floors": {
            "per_generation": minimum_samples_per_generation,
            "pooled_per_shape": minimum_samples,
        },
        "production_sps_table_eligible": False,
        "production_admission_qualified": False,
        "policy_overhead_measured": False,
        "task_accuracy_qualified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("profile", help="Collect authentic DSpark graph replays")
    collect.add_argument("--model", type=Path, required=True)
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--grid", type=Path, required=True)
    collect.add_argument("--physical-k", type=int, required=True)
    collect.add_argument(
        "--cost-scope", choices=("graph-replay", "whole-step"), default="graph-replay"
    )
    collect.add_argument(
        "--source-root",
        type=Path,
        help="Original public checkout used for recorded whole-step metadata.",
    )
    collect.add_argument("--inputs", type=Path, required=True, help="JSONL prompt_token_ids")
    collect.add_argument(
        "--recorded-source-origin",
        type=Path,
        help="Optional recorded HEAD/full-diff JSON for this immutable checkout.",
    )
    collect.add_argument(
        "--recorded-source-origin-sha256",
        help="SHA256 of the exact optional recorded source-origin bytes.",
    )
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--generations", type=int, default=3)
    collect.add_argument("--output-tokens", type=int, default=768)
    collect.add_argument("--timeout", type=float, default=12180.0)
    export = commands.add_parser("export", help="Reduce raw observations; never grant admission")
    export.add_argument("--raw", type=Path, nargs="+", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--exclude-generations", type=int, default=1)
    export.add_argument("--warmup-samples", type=int, default=2)
    export.add_argument("--minimum-samples", type=int, default=24)
    export.add_argument("--minimum-generation-samples", type=int, default=12)
    args = parser.parse_args(argv)
    if (
        args.command == "profile"
        and args.cost_scope != "whole-step"
        and (
            args.recorded_source_origin is not None
            or args.recorded_source_origin_sha256 is not None
        )
    ):
        parser.error("Recorded source-origin input applies only to --cost-scope whole-step")
    if args.command == "profile":
        from tensorrt_llm._torch.speculative.dspark_cost_profiler import run_profile

        run_profile(args)
    else:
        raw_sources = [(path, path.read_bytes()) for path in args.raw]
        documents = [parse_json(raw) for _, raw in raw_sources]
        summary = reduce_measurements(
            documents,
            excluded_generations=args.exclude_generations,
            warmup_samples=args.warmup_samples,
            minimum_samples=args.minimum_samples,
            minimum_samples_per_generation=args.minimum_generation_samples,
        )
        summary["raw_file_sha256"] = {
            str(path): hashlib.sha256(raw).hexdigest() for path, raw in raw_sources
        }
        write_new_json(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
