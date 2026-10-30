#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boot the engine a run's runtime block builds, and print the serving geometry it runs.

Usage (rank 0 under ``trtllm-llmapi-launch`` when the world size is above one)::

    <tree's python> geometry_probe.py --spec spec.json [--quickstart <path>] [-- <quickstart argv>]

The engine is built exactly as :mod:`.engine_driver` builds it -- the tree's
own ``quickstart_advanced.py`` over the gates' argv, the target selection
from ``TRTLLM_MODELING_V2`` and the handed-over KV-cache keys -- with one
addition: iteration statistics on (``enable_iter_perf_stats``), which change
no pool, limit or capture list. What it prints, each one JSON line behind a
marker (see :mod:`.workload_io`):

* ``MV2FLOW_ENGINE`` once the engine is up: the configuration it runs, its
  pools and the device memory left free (:mod:`.engine_facts`), the record
  every gate attests an engine by;
* ``MV2FLOW_GEOMETRY`` once: the KV page view (tokens per block, blocks,
  pages a sequence can hold), the limits, the snapshot policy, and for each
  probe length of the spec (``chunk_probes``) the context chunks the
  scheduler ran one prompt of that length in, read from the iteration
  statistics. A prompt is random token ids generating two tokens, sent alone,
  so each iteration's context tokens are that prompt's chunk.

A value that cannot be read is left out and its reason is kept under
``errors``; reading the engine is an observation and never stops the probe.
Exits non-zero with a traceback only when the engine cannot be built.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_driver  # noqa: E402 -- imported by path, see the package docstring
import engine_facts  # noqa: E402
import workload_io as wio  # noqa: E402

#: The probe's geometry record, once per engine.
GEOMETRY_MARKER = "MV2FLOW_GEOMETRY"
#: The LLM-API argument that turns iteration statistics on.
ITERATION_STATS_KWARG = "enable_iter_perf_stats"
#: Tokens each probe prompt generates: one context step and one generation step.
PROBE_MAX_TOKENS = 2
#: How long ``get_stats`` waits for an iteration's record, in seconds.
STATS_TIMEOUT_S = 2.0


@contextlib.contextmanager
def iteration_stats(quickstart: Any) -> Iterator[None]:
    """While open, the quickstart's ``LLM`` is built with iteration statistics on.

    The quickstart resolves ``LLM`` from its module when ``setup_llm`` runs,
    and :func:`engine_driver.build_llm` wraps whatever it finds there, so
    this wrapper is the innermost one either way.
    """
    original = getattr(quickstart, "LLM", None)
    if original is None:
        yield
        return

    def build(*positional: Any, **kwargs: Any) -> Any:
        kwargs.setdefault(ITERATION_STATS_KWARG, True)
        return original(*positional, **kwargs)

    quickstart.LLM = build
    try:
        yield
    finally:
        quickstart.LLM = original


def _stat(entry: Any) -> Mapping[str, Any]:
    """One iteration's statistics as a mapping; a JSON string is parsed."""
    if isinstance(entry, str):
        try:
            entry = json.loads(entry)
        except ValueError:
            return {}
    return entry if isinstance(entry, Mapping) else {}


def context_tokens(entry: Any) -> int | None:
    """The context tokens one iteration ran, from its in-flight batching statistics."""
    stat = _stat(entry)
    batching = stat.get("inflightBatchingStats", stat.get("inflight_batching_stats"))
    if not isinstance(batching, Mapping):
        return None
    value = batching.get("numCtxTokens", batching.get("num_ctx_tokens"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def chunks_of(stats: Sequence[Any]) -> list[int]:
    """The context chunks a lone prompt ran in: each iteration's context tokens, in order."""
    rows = [_stat(entry) for entry in stats]
    if all(isinstance(row.get("iter"), int) for row in rows):
        rows = sorted(rows, key=lambda row: row["iter"])
    chunks = [context_tokens(row) for row in rows]
    return [c for c in chunks if c]


def page_view(llm: Any, errors: dict[str, str]) -> dict[str, Any]:
    """The KV page view and the limits the engine serves under."""
    view: dict[str, Any] = {}
    capacity = engine_facts.kv_capacity(llm)
    for name, key in (
        ("tokens_per_block", "tokensPerBlock"),
        ("blocks", "maxNumBlocks"),
        ("max_tokens", "maxNumTokens"),
    ):
        value = capacity.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            view[name] = value
    if not capacity:
        errors["page_view"] = "the engine reported no KV-cache capacity"
    args = getattr(llm, "args", None)
    if args is None:
        errors["limits"] = f"{type(llm).__name__} has no `args`"
        return view
    for name in ("max_seq_len", "max_num_tokens", "max_batch_size"):
        engine_facts._read(
            view, errors, name, lambda name=name: engine_facts._plain(getattr(args, name))
        )
    engine_facts._read(view, errors, "chunked_prefill", lambda: bool(args.enable_chunked_prefill))
    engine_facts._read(
        view,
        errors,
        "snapshot_policy",
        lambda: engine_facts.snapshot_policy(args.kv_cache_config.mamba_state_config),
    )
    per_block, longest = view.get("tokens_per_block"), view.get("max_seq_len")
    if isinstance(per_block, int) and per_block > 0 and isinstance(longest, int):
        view["max_blocks_per_seq"] = -(-longest // per_block)
    return view


def probe_lengths(lengths: Sequence[Any], max_seq_len: Any) -> list[int]:
    """The probe lengths a prompt can have here: positive, and leaving room to generate."""
    room = int(max_seq_len) - PROBE_MAX_TOKENS if isinstance(max_seq_len, int) else None
    kept = sorted({int(n) for n in lengths if isinstance(n, int) and n > 0})
    return [n for n in kept if room is None or n <= room]


def probe_chunks(
    llm: Any, base: Any, lengths: Sequence[int], *, seed: int = 0
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """For each length, the context chunks one prompt of it ran in; and per length what failed."""
    chunks: dict[str, list[int]] = {}
    errors: dict[str, str] = {}
    get_stats = getattr(llm, "get_stats", None)
    if not callable(get_stats):
        return chunks, {"stats": f"{type(llm).__name__} has no get_stats"}
    rng = random.Random(seed)
    vocab = engine_driver._vocab_size(llm)
    low, high = 100, max(101, vocab - 100)
    params = engine_driver.sampling_for(base, PROBE_MAX_TOKENS, True)
    for length in lengths:
        # Whatever is queued belongs to an earlier request.
        try:
            get_stats(timeout=0.1)
            prompt = {"prompt_token_ids": [rng.randrange(low, high) for _ in range(length)]}
            llm.generate([prompt], params)
            found = chunks_of(get_stats(timeout=STATS_TIMEOUT_S) or ())
        # The probe drives the engine of the tree under test; what it raises is
        # that engine's to report, in the record, not this harness's to die of.
        except Exception as exc:  # noqa: BLE001
            errors[str(length)] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        if found:
            chunks[str(length)] = found
        else:
            errors[str(length)] = "the iteration statistics carried no context tokens"
    return chunks, errors


def geometry(llm: Any, base: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    """The ``MV2FLOW_GEOMETRY`` record of an engine that is up; see the module docstring."""
    errors: dict[str, str] = {}
    record: dict[str, Any] = {"page": page_view(llm, errors)}
    lengths = probe_lengths(spec.get("chunk_probes") or (), record["page"].get("max_seq_len"))
    if lengths:
        chunks, failed = probe_chunks(llm, base, lengths, seed=int(spec.get("seed", 0)))
        record["chunks"] = chunks
        errors.update({f"chunks[{k}]": v for k, v in failed.items()})
    record["errors"] = errors
    return record


def main(argv: list[str] | None = None) -> int:
    """Read the spec, build the engine, print its engine and geometry records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, help="the probe spec, a JSON file")
    parser.add_argument(
        "--quickstart", help="the tree's quickstart_advanced.py (default: the spec's)"
    )
    parser.add_argument(
        "quickstart_argv",
        nargs=argparse.REMAINDER,
        help="after `--`: the quickstart's argv (default: the spec's)",
    )
    options = parser.parse_args(argv)
    spec = json.loads(Path(options.spec).read_text(encoding="utf-8"))
    if options.quickstart:
        spec["quickstart"] = options.quickstart
    words = [w for w in options.quickstart_argv if w != "--"] if options.quickstart_argv else []
    if words:
        spec["quickstart_argv"] = words

    quickstart = engine_driver.load_quickstart(spec["quickstart"])
    observer = engine_facts.WorkerObserver()
    observer.install()
    argv = list(spec.get("quickstart_argv") or ())
    with iteration_stats(quickstart):
        llm, base = engine_driver.build_llm(quickstart, argv)
    try:
        engine_facts.emit(llm, observer, argv=argv)
        try:
            record = geometry(llm, base, spec)
        except Exception as exc:  # noqa: BLE001 -- an observation; see the module docstring
            record = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        wio.emit(GEOMETRY_MARKER, record)
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
