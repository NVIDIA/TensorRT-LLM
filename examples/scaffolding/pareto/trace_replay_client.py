# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Single-ladder-step trace replay against an external ``trtllm-serve``.

Pure OpenAI client. Expects ``trtllm-serve`` (or any OpenAI-compatible
completions endpoint that honors token-id prompts and ``ignore_eos``) to be
already running at ``--base_url`` with the desired ``max_batch_size``, TP,
EP, etc. Submits ``--total_sessions`` :class:`ReplayEngine` tasks with at most
``--concurrency`` of them in flight at any time (the remaining sessions
wait on an :class:`asyncio.Semaphore`), measures per-session and aggregate
throughput, and writes a single **step JSON** summarizing this ladder point.

The three load knobs are fully decoupled, matching the pattern used by
``examples/scaffolding/benchmarks/`` (see ``__main__.py`` /
``chat_benchmark.py``):

* ``--max_batch_size``: the server's ``trtllm-serve --max_batch_size`` for
  this step (metadata only, the actual cap is enforced server-side).
* ``--total_sessions``: total number of trace replays to execute (total work).
* ``--concurrency``: maximum number of replays in flight at any moment
  (load shaping, equivalent to ``chat_concurrency`` in the benchmarks).

Server lifecycle (start trtllm-serve with ``--max_batch_size``, poll
``/health``, kill after the client exits) lives in the Slurm driver script,
not here.

Example::

    python examples/scaffolding/pareto/trace_replay_client.py \
        --base_url http://127.0.0.1:8000/v1 \
        --model /path/to/Qwen3-235B-A22B \
        --trace_dir .../traces/swebench/django__django-14787 \
        --total_sessions 32 --concurrency 16 --max_batch_size 16 \
        --ladder_index 1 --ladder_step 16 \
        --tensor_parallel_size 4 --moe_expert_parallel_size 4 \
        --output_json .../step16.json

Add ``--enable_attention_dp`` for the attention-DP configuration (e.g.
``--tp_size 4 --ep_size 4 --enable_attention_dp`` i.e. "DP4 + EP4"); the
flag is server-side state (whether trtllm-serve was launched with
``--enable_attention_dp``) and is recorded here as metadata so the
aggregator can stamp the right artifact-naming suffix (``_adp``).

Add ``--enable_chunked_prefill`` when ``trtllm-serve`` was launched with
the same flag. Unlike ``--enable_attention_dp``, chunked prefill is
orthogonal to the parallelism topology, so it does *not* participate in
the ``_adp``-style filename suffix — it is recorded purely as server-side
provenance in ``llm_effective_config`` (and, from the aggregator, in
``llm_fixed_config``) so with-vs-without-chunked-prefill sweeps can be
told apart after the fact.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import random
import shlex
import socket
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from _common import (
    TeeTextIO,
    args_to_dict,
    atomic_write_json,
    collect_host_info,
    collect_trace_file_stats,
    count_assistant_completion_tokens,
    count_parallel_regions,
    find_compact_trace_file,
    is_oom_exception,
    pareto_config_filename_suffix,
    percentile,
    summarize_trace_events,
)
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import TRTOpenaiWorker
from tensorrt_llm.scaffolding.execution_trace import ExecutionTrace, TraceEvent
from tensorrt_llm.scaffolding.replay import (
    DropPathStats,
    ReplayEngine,
    ReplayGenerationStats,
    RetentionProbeStats,
    RngFactory,
    make_seeded_rng_factory,
)
from tensorrt_llm.scaffolding.task import GenerationTask, TaskStatus

# Schema version for the single-step JSON written by this client. The
# aggregator expects this identifier and concatenates ``runs[0]`` from N
# step JSONs into the combined ``trace_replay_pareto_frontier.v4`` record.
STEP_SCHEMA = "trace_replay_pareto_frontier.step.v4"


def _collect_tpot_seconds(
    stats_list: List[ReplayGenerationStats],
) -> List[float]:
    """Flatten per-LLM-call entries into the TPOT list SemiAnalysis uses.

    Per request: ``tpot = (latency - ttft) / (output_len - 1)``. Entries
    with ``output_len <= 1`` or missing timing are skipped — matching the
    ``output_len > 1`` guard in ``benchmark_serving.py``.
    """
    out: List[float] = []
    for s in stats_list:
        for entry in s.entries:
            lat = entry.get("latency_s")
            ttft = entry.get("ttft_s")
            out_len = entry.get("usage_completion_tokens")
            if out_len is None:
                out_len = entry.get("replay_output_token_len")
            if lat is None or ttft is None or out_len is None:
                continue
            if out_len <= 1:
                continue
            tpot = (float(lat) - float(ttft)) / (int(out_len) - 1)
            if tpot > 0:
                out.append(tpot)
    return out


# ---------------------------------------------------------------------------
# Server-side perf-metrics drain (per-request KV cache hit accounting)
# ---------------------------------------------------------------------------
#
# trtllm-serve's /perf_metrics endpoint atomically swaps its accumulated
# request-record deque for a fresh one and returns the contents.  We drain
# it ONCE at the end of the run and join the records back into
# ``replay_assistant_generations_detail`` by ``request_id``, so every
# per-LLM-call row carries:
#
#   * num_reused_blocks / num_missed_blocks / kv_cache_hit_rate
#   * free_num_blocks / used_num_blocks / utilization snapshots
#
# The endpoint lives at the server's host ROOT (not under /v1), so we
# strip the "/v1" suffix from the OpenAI client base_url before dialing.
# When the server is launched without ``return_perf_metrics`` the drain
# returns ``[]`` and augmentation is a no-op (the rows stay as-is).


def _control_plane_base_url(base_url: str) -> str:
    """Return the server's root URL given an OpenAI-compatible base URL.

    OpenAI clients are typically configured with
    ``http://host:port/v1`` (or ``/v1/``); the perf and control-plane
    endpoints live at ``http://host:port/perf_metrics`` and
    ``http://host:port/_control/...``.
    """
    out = base_url.rstrip("/")
    for suffix in ("/v1", "/v1/"):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
            break
    return out.rstrip("/")


async def _drain_perf_metrics(
    base_url: str,
    api_key: Optional[str],
    timeout_s: float = 120.0,
) -> List[Dict[str, Any]]:
    """GET /perf_metrics; return the list of per-request records (or [])."""
    url = _control_plane_base_url(base_url) + "/perf_metrics"
    headers = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"GET /perf_metrics failed: HTTP {resp.status_code} {resp.text!r}")
    body = resp.json()
    if not isinstance(body, list):
        raise RuntimeError(f"/perf_metrics returned unexpected body type: {type(body).__name__}")
    return body


async def _drain_kv_cache_events_once(
    base_url: str,
    api_key: Optional[str],
    timeout_s: float = 30.0,
) -> List[Dict[str, Any]]:
    """POST /kv_cache_events; return the list of events emitted since last drain."""
    url = _control_plane_base_url(base_url) + "/kv_cache_events"
    headers = {}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"POST /kv_cache_events failed: HTTP {resp.status_code} {resp.text!r}")
    body = resp.json()
    if not isinstance(body, list):
        raise RuntimeError(f"/kv_cache_events returned unexpected body type: {type(body).__name__}")
    return body


async def _kv_cache_event_drain_loop(
    base_url: str,
    api_key: Optional[str],
    output_path: Path,
    interval_s: float,
    stop_event,
):
    """Background task: drain the server's KV-cache event ring on a fixed cadence.

    Every ``interval_s`` seconds, drain the server's KV-cache event ring
    and append every event as one JSONL line.
    Loop exits when ``stop_event`` is set; one final drain happens after
    the stop so any events emitted in the last interval are captured.

    File I/O hazard: this task is one coroutine on the same event loop
    as the trace replay sessions and the retention probe phase. If the
    file write ever blocks (the cluster's Lustre OST flapping is the
    observed cause; see feedback_no_direct_lustre_writes.md), the
    entire asyncio loop stops — every session, every probe — until the
    fs unstuck. To avoid that:
      1. Caller must point ``output_path`` at container-local
         ``/tmp/`` (NVMe). Lustre is forbidden as a target.
      2. Even on /tmp, all writes go through ``asyncio.to_thread`` so
         a transient stall on the underlying device cannot block the
         loop.
    The bash driver's safety_net step exfiltrates ``output_path`` from
    /tmp to the run output dir on Lustre after python exits, with
    ``timeout 120 cp`` so a still-stalled fs cannot stall bash either.
    """
    import json as _json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(output_path, "w", encoding="utf-8")

    def _write_events(events):
        if not events:
            return
        for ev in events:
            fp.write(_json.dumps(ev) + "\n")
        fp.flush()

    total = 0
    try:
        while not stop_event.is_set():
            try:
                events = await _drain_kv_cache_events_once(base_url, api_key)
                if events:
                    await asyncio.to_thread(_write_events, events)
                    total += len(events)
            except Exception as exc:
                print(
                    f"[kv_cache_events] drain error (continuing): {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                pass
        # Final drain after stop_event fires.
        try:
            events = await _drain_kv_cache_events_once(base_url, api_key)
            if events:
                await asyncio.to_thread(_write_events, events)
                total += len(events)
        except Exception as exc:
            print(f"[kv_cache_events] final drain error: {exc!r}", file=sys.stderr, flush=True)
    finally:
        await asyncio.to_thread(fp.close)
    print(f"[kv_cache_events] drain task exiting after {total} events to {output_path}", flush=True)


def _normalize_request_id(rid: Any) -> Optional[str]:
    """Map both sides of the request-id join to the same string form.

    The OpenAI streaming Completions client returns ``chunk.id`` as
    ``"cmpl-<executor_request_id>"`` (the prefix matches the
    ``CompletionStreamResponse.id`` schema default; trtllm-serve sets the
    suffix to ``str(rsp.id)``, which is the executor's int request id).
    The /perf_metrics record's ``request_id`` is the bare executor id.
    Strip the ``cmpl-`` prefix from chunk ids and stringify the perf-side
    int so both sides land on the same key.
    """
    if rid is None:
        return None
    s = str(rid)
    for prefix in ("cmpl-", "chatcmpl-"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    return s


def _augment_entries_with_perf_metrics(
    entries: List[Dict[str, Any]],
    perf_records: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Merge ``/perf_metrics`` records into the per-LLM-call ``entries``.

    Joins on the executor's request id with both sides normalized via
    :func:`_normalize_request_id`.  Each matched entry is augmented with
    two groups of server-side fields:

    * KV cache accounting (``num_reused_blocks``, ``num_missed_blocks``,
      ``num_total_allocated_blocks``, ``num_new_allocated_blocks``,
      ``kv_cache_hit_rate`` and the pool-state snapshot fields suffixed
      with ``_after``).
    * Per-request scheduling timestamps (``server_arrival_time``,
      ``arrival_time``, ``first_scheduled_time``, ``first_token_time``,
      ``server_first_token_time``, ``last_token_time``) plus
      ``first_iter`` / ``last_iter`` iteration counters.  All timing
      fields are monotonic-clock seconds on the rank-0 server; downstream
      code can derive queue wait, prefill duration and decode span
      without re-implementing the clock convention.

    Returns counters for sanity-checking by the caller.
    """
    by_rid: Dict[str, Dict[str, Any]] = {}
    for rec in perf_records:
        rid = _normalize_request_id(rec.get("request_id"))
        if rid is None:
            continue
        by_rid[rid] = rec

    counters = {
        "perf_records": len(perf_records),
        "entries_total": len(entries),
        "entries_matched": 0,
        "entries_no_request_id": 0,
        "entries_no_perf_record": 0,
    }

    for entry in entries:
        # Compute agent_role from branch_path regardless of whether perf
        # metrics matched — this label is used by every downstream
        # aggregator and shouldn't depend on the server's perf knob.
        bp = entry.get("branch_path") or []
        entry["agent_role"] = "Supervisor" if not bp else "Researcher"

        rid = _normalize_request_id(entry.get("request_id"))
        if rid is None:
            counters["entries_no_request_id"] += 1
            continue
        rec = by_rid.get(rid)
        if rec is None:
            counters["entries_no_perf_record"] += 1
            continue

        perf = rec.get("perf_metrics") or {}
        kv = perf.get("kv_cache_metrics") or {}
        reused = kv.get("num_reused_blocks")
        missed = kv.get("num_missed_blocks")
        if reused is not None and missed is not None and (reused + missed) > 0:
            entry["kv_cache_hit_rate"] = reused / (reused + missed)
        else:
            entry["kv_cache_hit_rate"] = None
        entry["num_reused_blocks"] = reused
        entry["num_missed_blocks"] = missed
        entry["num_total_allocated_blocks"] = kv.get("num_total_allocated_blocks")
        entry["num_new_allocated_blocks"] = kv.get("num_new_allocated_blocks")
        entry["free_num_blocks_after"] = kv.get("free_num_blocks")
        entry["used_num_blocks_after"] = kv.get("used_num_blocks")
        entry["max_num_blocks"] = kv.get("max_num_blocks")
        entry["kv_utilization_after"] = kv.get("utilization")

        timing = perf.get("timing_metrics") or {}
        entry["first_iter"] = perf.get("first_iter")
        entry["last_iter"] = perf.get("last_iter")
        entry["server_arrival_time"] = timing.get("server_arrival_time")
        entry["arrival_time"] = timing.get("arrival_time")
        entry["first_scheduled_time"] = timing.get("first_scheduled_time")
        entry["first_token_time"] = timing.get("first_token_time")
        entry["server_first_token_time"] = timing.get("server_first_token_time")
        entry["last_token_time"] = timing.get("last_token_time")
        counters["entries_matched"] += 1

    return counters


def _percentile_sorted(xs_sorted: List[float], q: float) -> Optional[float]:
    if not xs_sorted:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile q must be in [0, 1] (got {q!r})")
    # Nearest-rank percentile: same convention as SemiAnalysis's
    # utils/process_result.py (``statistics.quantiles`` with ``n=100`` and
    # then indexing by nearest rank).
    k = max(0, min(len(xs_sorted) - 1, int(round(q * (len(xs_sorted) - 1)))))
    return xs_sorted[k]


# ---------------------------------------------------------------------------
# Warmup: pin a short system-prompt prefill to every (DP rank, trace) pair
# ---------------------------------------------------------------------------
#
# ``attention_dp`` splits the KV cache radix tree per DP rank: each rank only
# sees the blocks that trtllm-serve's ADP router routes to it. The router
# uses load balance (optionally cache-affinity) to pick a rank for each
# request, which means an arrival-order-based warmup cannot guarantee that
# every rank was populated with every trace's system prefix. To close this
# gap, we use the ``attention_dp_rank`` scheduling parameter (exposed on
# ``CompletionRequest`` in ``openai_protocol.py`` and threaded into
# ``SchedulingParams`` by ``openai_server.py``) to explicitly pin each
# warmup request to the target rank. ``attention_dp_relax=False`` makes the
# router wait for the pinned rank to free a slot instead of falling back to
# load-balanced placement, so the pin is a strict guarantee rather than a
# best-effort hint.
#
# One warmup request per (DP rank, trace, system event) triple, with
# ``max_tokens=1`` and the prompt set to exactly the trace's system-role
# token ids. The server prefills the 2269-ish token system prefix onto that
# rank's radix tree, generates a single placeholder token, and returns. The
# same synthetic token ids are also stored into the per-trace
# ``system_token_cache`` dict so the subsequent measurement burst's
# :class:`ReplayEngine` resolves the trace's system event to exactly those
# ids and hits the server-side blocks that the warmup wrote.


def _build_seeded_rng(seed_parts: Tuple[Any, ...]) -> random.Random:
    """Return a :class:`random.Random` seeded with SHA-256(seed_parts).

    Used to derive the per-(namespace, trace, conv_id) RNG for the
    client's pre-generated system-prefix tokens.  SHA-256 is used so the
    seed is stable across Python versions and ``PYTHONHASHSEED`` settings
    — important because pass-1 and pass-2 of the same experiment usually
    run in different processes.
    """
    key = "\x00".join(repr(p) for p in seed_parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "big")
    return random.Random(seed_int)


def _random_token_ids(length: int, rng: Optional[random.Random] = None) -> List[int]:
    """Synthetic token id stream for a single message segment.

    Matches the generator used by :mod:`tensorrt_llm.scaffolding.replay`
    (``_generate_random_token_ids`` there) so that, when the client passes
    the same ids to ``ReplayEngine`` via the shared ``system_token_cache``,
    the radix-tree block hashes computed by the server on the warmup
    prefill are identical to the hashes on the measurement burst's
    subsequent prompts.

    *rng* is the per-(namespace, trace) :class:`random.Random` used to
    pre-generate every system prefix's tokens deterministically when
    ``--pass_seed_namespace`` is set.  When *rng* is ``None`` the function
    falls back to the module-level random (non-deterministic across
    processes), preserving the historical behaviour for callers that have
    not opted in.
    """
    if length <= 0:
        return []
    if rng is not None:
        return [rng.randint(100, 30000) for _ in range(length)]
    return [random.randint(100, 30000) for _ in range(length)]


def _extract_trace_system_segments(
    trace: ExecutionTrace,
) -> List[Tuple[int, int]]:
    """Return ``(conv_id, token_count)`` for every system-role message.

    SWE-bench traces have a single ``conv_id == 0`` system event, but we
    iterate so any future multi-conversation trace is handled without
    special-casing. Events without a usable token count (pre-tokenization
    or non-``message`` types) are skipped.
    """
    out: List[Tuple[int, int]] = []
    ev: TraceEvent
    for ev in trace.events:
        if ev.event_type != "message":
            continue
        if ev.role != "system":
            continue
        tok = ev.tokens
        if tok is None or tok <= 0:
            continue
        out.append((int(ev.conversation_id or 0), int(tok)))
    return out


async def _warmup_pinned_prefill(
    client: AsyncOpenAI,
    model: str,
    *,
    prompt_token_ids: List[int],
    attention_dp_rank: int,
    trace_index: int,
    conv_id: int,
    semaphore: asyncio.Semaphore,
    request_timeout_s: float,
) -> float:
    """Issue one rank-pinned, single-token completion to warm a rank's KV cache.

    The request pins itself to ``attention_dp_rank`` with
    ``attention_dp_relax=False`` so trtllm-serve's ADP router strictly
    places it on the target rank. ``max_tokens=1`` + ``ignore_eos=True``
    keeps the response trivial — the point is the prefill, which writes
    the system prompt's prefix blocks into this rank's radix tree.
    """
    async with semaphore:
        t0 = time.perf_counter()
        label = (
            f"[warmup] trace={trace_index} conv={conv_id} "
            f"dp_rank={attention_dp_rank} prefix_tokens={len(prompt_token_ids)}"
        )
        print(f"{label}: start", flush=True)
        try:
            stream = await client.completions.create(
                model=model,
                prompt=prompt_token_ids,
                max_tokens=1,
                stream=True,
                stream_options={"include_usage": True},
                timeout=request_timeout_s,
                extra_body={
                    "ignore_eos": True,
                    "attention_dp_rank": attention_dp_rank,
                    "attention_dp_relax": False,
                },
            )
            async for _ in stream:
                pass
        except Exception:
            print(
                f"{label}: FAILED (see traceback below)",
                file=sys.stderr,
                flush=True,
            )
            raise
        elapsed = time.perf_counter() - t0
        print(f"{label}: done in {elapsed:.3f}s", flush=True)
        return elapsed


# ---------------------------------------------------------------------------
# One concurrent session (one full trace replay)
# ---------------------------------------------------------------------------


async def _one_session(
    worker: TRTOpenaiWorker,
    trace: ExecutionTrace,
    *,
    semaphore: asyncio.Semaphore,
    session_index: int,
    trace_index: int,
    total_sessions: int,
    concurrency: int,
    max_batch_size: int,
    ladder_step: int,
    system_token_cache: Optional[Dict[int, List[int]]] = None,
    rng_factory: Optional[RngFactory] = None,
    drop_path_stats: Optional[DropPathStats] = None,
    retention_probe_stats: Optional[RetentionProbeStats] = None,
    arrival_delay_s: float = 0.0,
    label_prefix: str = "",
) -> Tuple[float, ReplayGenerationStats]:
    label = (
        f"{label_prefix}[step={ladder_step} N={total_sessions} C={concurrency} B={max_batch_size}] "
        f"session {session_index + 1}/{total_sessions}"
    )
    # Arrival-time jitter: sleep a per-session offset BEFORE acquiring the
    # concurrency semaphore. Purpose: when multiple sessions replay the same
    # trace under round-robin assignment (traces[i % K] with K=4), they all
    # start at t=0 and execute an identical event script in lockstep, so the
    # server sees wave-like pulses (32+ copies simultaneously prefilling, then
    # 32+ simultaneously in tool-sleep, etc.). Uniform[0, arrival_jitter_s)
    # offsets break the lockstep: by the time session i+K has issued its
    # first LLM call, session i is already deep into its own event stream at
    # a different offset, smoothing the server-side batch occupancy and
    # raising time-averaged in-flight batch from ~C * LLM-fraction to the
    # full decode_fraction * C ceiling.
    #
    # The sleep happens before `async with semaphore` so waiting sessions
    # don't hold semaphore slots while idling. For C == total_sessions
    # (the saturation-sweep use case), the semaphore admits all sessions
    # immediately and this sleep IS the arrival process.
    if arrival_delay_s > 0.0:
        print(f"{label}: arrival_delay={arrival_delay_s:.3f}s", flush=True)
        await asyncio.sleep(arrival_delay_s)
    async with semaphore:
        print(f"{label}: replay start", flush=True)
        stats = ReplayGenerationStats(
            session_index=session_index,
            trace_index=trace_index,
        )
        t0 = time.perf_counter()
        await ReplayEngine(
            worker,
            generation_stats=stats,
            system_token_cache=system_token_cache,
            rng_factory=rng_factory,
            drop_path_stats=drop_path_stats,
            retention_probe_stats=retention_probe_stats,
        ).launch_trace(trace)
        elapsed = time.perf_counter() - t0
        print(f"{label}: replay done in {elapsed:.3f}s", flush=True)
        return elapsed, stats


# ---------------------------------------------------------------------------
# Throughput / latency accounting (backend-agnostic)
# ---------------------------------------------------------------------------


def compute_run_row(
    *,
    traces: List[ExecutionTrace],
    results: List[Tuple[float, ReplayGenerationStats]],
    wall_s: float,
    total_sessions: int,
    concurrency: int,
    max_batch_size: int,
    ladder_index: int,
    ladder_step: int,
    tensor_parallel_size: int,
) -> Dict[str, Any]:
    """Build one ``runs[]`` row from successful per-session results.

    The three load dimensions (``total_sessions``, ``concurrency``,
    ``max_batch_size``) are recorded independently so plot helpers can
    distinguish a step that stacked more total work at the same in-flight
    concurrency from one that raised the concurrency itself. Field names
    and formulas otherwise match ``trace_replay_pareto_frontier.v3``
    (produced by the older in-process script), so existing plot helpers
    keep working unchanged.

    With multiple traces, sessions are assigned round-robin
    (``traces[i % len(traces)]``); the trace-metadata-derived totals sum
    over the actual round-robin assignment so per-trace work is correctly
    weighted regardless of how N divides.
    """
    durations = [r[0] for r in results]
    stats_list = [r[1] for r in results]

    tokens_per_trace_list = [count_assistant_completion_tokens(t.events) for t in traces]
    total_out_tokens_trace_metadata = float(
        sum(tokens_per_trace_list[i % len(traces)] for i in range(total_sessions))
    )
    # For backward compat: when only one trace is in play, expose the same
    # scalar field name; when mixed, expose the round-robin mean.
    tokens_per_trace_trace_metadata = (
        tokens_per_trace_list[0]
        if len(traces) == 1
        else (total_out_tokens_trace_metadata / total_sessions if total_sessions else 0)
    )

    per_session_replay_output = [s.sum_replay_output_tokens() for s in stats_list]
    per_session_trace_completion = [s.sum_trace_completion_tokens() for s in stats_list]
    total_out_tokens_replay_actual = float(sum(per_session_replay_output))

    # All per-LLM-call entries (one per assistant turn per session). Each
    # entry already carries ``session_index`` / ``trace_index`` so downstream
    # consumers can break the N×R scatter into per-trace clusters or compute
    # alternative aggregates without rerunning the sweep. For a typical
    # SWE-bench sweep (N=32, R≈21) this is ~670 entries / ~30 KB JSON.
    detail_all_entries: List[Dict[str, Any]] = []
    for s in stats_list:
        detail_all_entries.extend(s.entries)

    tp_sizes = [
        per_session_replay_output[i] / durations[i]
        for i in range(len(durations))
        if durations[i] > 0
    ]

    # Per-request TPOT → intvty, matching SemiAnalysis's InferenceMAX
    # ``benchmark_serving.py`` (skip-on-``output_len<=1`` included). The
    # per-request sample count is N×R when every call succeeded with
    # ``output_len > 1``, which gives much tighter percentile bands than
    # the N per-session numbers below.
    tpot_seconds = _collect_tpot_seconds(stats_list)
    tpot_ms_sorted = sorted(t * 1000.0 for t in tpot_seconds)

    row: Dict[str, Any] = {
        "ladder_index": ladder_index,
        "ladder_step": ladder_step,
        "total_sessions": total_sessions,
        "concurrency": concurrency,
        "max_batch_size": max_batch_size,
        "status": "success",
        "error": None,
        "error_traceback": None,
        "wall_clock_s": wall_s,
        # From trace file ``completion_tokens`` (original recording).
        "assistant_output_tokens_per_trace": tokens_per_trace_trace_metadata,
        "total_output_tokens_trace_metadata": total_out_tokens_trace_metadata,
        # Backward-compatible alias.
        "total_output_tokens_estimated": total_out_tokens_trace_metadata,
        # Measured during this replay (decoder output token ids per session).
        "per_session_replay_output_token_sum": per_session_replay_output,
        "per_session_total_output_tokens": list(per_session_replay_output),
        "per_session_trace_completion_token_sum": per_session_trace_completion,
        "assistant_output_tokens_per_trace_replay_actual_mean": (
            statistics.mean(per_session_replay_output) if per_session_replay_output else None
        ),
        "total_output_tokens_replay_actual": total_out_tokens_replay_actual,
        "replay_assistant_generations_detail": detail_all_entries,
        "session_duration_s": durations,
        "session_duration_min_s": min(durations) if durations else None,
        "session_duration_max_s": max(durations) if durations else None,
        "session_duration_sum_s": sum(durations) if durations else None,
        "session_duration_stdev_s": (statistics.stdev(durations) if len(durations) > 1 else 0.0),
        "session_duration_p50_s": statistics.median(durations) if durations else None,
        "session_duration_p90_s": percentile(durations, 0.9) if durations else None,
        "session_duration_p99_s": percentile(durations, 0.99) if durations else None,
        "session_duration_mean_s": statistics.mean(durations) if durations else None,
        "session_duration_cv": (
            (statistics.stdev(durations) / statistics.mean(durations))
            if len(durations) > 1 and statistics.mean(durations) > 0
            else None
        ),
        "aggregate_latency_person_s": sum(durations) if durations else None,
        "median_tps_per_user": statistics.median(tp_sizes) if tp_sizes else None,
        "mean_tps_per_user": statistics.mean(tp_sizes) if tp_sizes else None,
        "min_tps_per_user": min(tp_sizes) if tp_sizes else None,
        "max_tps_per_user": max(tp_sizes) if tp_sizes else None,
        "output_tps_aggregate": (total_out_tokens_replay_actual / wall_s) if wall_s > 0 else None,
        "output_tokens_per_wall_s_per_session_mean": (
            (total_out_tokens_replay_actual / wall_s / total_sessions)
            if wall_s > 0 and total_sessions
            else None
        ),
        "mean_tps_per_user_session_time": (
            (total_out_tokens_replay_actual / sum(durations))
            if durations and sum(durations) > 0
            else None
        ),
    }

    tp = int(tensor_parallel_size or 0)
    agg = row["output_tps_aggregate"]
    row["output_tps_per_gpu"] = (agg / tp) if (agg is not None and tp > 0) else None

    # InferenceMAX-aligned per-request summary. Field names mirror
    # SemiAnalysis's ``utils/process_result.py`` (``median_tpot_ms``,
    # ``median_intvty``, etc.); note the ``intvty`` inversion:
    # ``p99_intvty = 1000 / p99_tpot_ms`` is the *slow tail*, so it is the
    # smallest value in the intvty set.
    def _intvty_from_tpot_ms(v: Optional[float]) -> Optional[float]:
        if v is None or v <= 0:
            return None
        return 1000.0 / v

    tpot_count = len(tpot_ms_sorted)
    median_tpot_ms = statistics.median(tpot_ms_sorted) if tpot_ms_sorted else None
    mean_tpot_ms = statistics.mean(tpot_ms_sorted) if tpot_ms_sorted else None
    p90_tpot_ms = _percentile_sorted(tpot_ms_sorted, 0.90)
    p99_tpot_ms = _percentile_sorted(tpot_ms_sorted, 0.99)
    p99_9_tpot_ms = _percentile_sorted(tpot_ms_sorted, 0.999)
    p1_tpot_ms = _percentile_sorted(tpot_ms_sorted, 0.01)
    min_tpot_ms = tpot_ms_sorted[0] if tpot_ms_sorted else None
    max_tpot_ms = tpot_ms_sorted[-1] if tpot_ms_sorted else None
    row.update(
        {
            "tpot_ms_count": tpot_count,
            "median_tpot_ms": median_tpot_ms,
            "mean_tpot_ms": mean_tpot_ms,
            "p90_tpot_ms": p90_tpot_ms,
            "p99_tpot_ms": p99_tpot_ms,
            "p99_9_tpot_ms": p99_9_tpot_ms,
            "p1_tpot_ms": p1_tpot_ms,
            "min_tpot_ms": min_tpot_ms,
            "max_tpot_ms": max_tpot_ms,
            "median_intvty": _intvty_from_tpot_ms(median_tpot_ms),
            "mean_intvty": _intvty_from_tpot_ms(mean_tpot_ms),
            "p90_intvty": _intvty_from_tpot_ms(p90_tpot_ms),
            "p99_intvty": _intvty_from_tpot_ms(p99_tpot_ms),
            "p99_9_intvty": _intvty_from_tpot_ms(p99_9_tpot_ms),
            # Fast / slow extremes. ``fastest_intvty`` = 1/min_tpot (largest),
            # ``slowest_intvty`` = 1/max_tpot (smallest). Used by the
            # cross-config scatter's ``min..max`` per-request band.
            "fastest_intvty": _intvty_from_tpot_ms(min_tpot_ms),
            "slowest_intvty": _intvty_from_tpot_ms(max_tpot_ms),
            "p1_intvty_from_tpot": _intvty_from_tpot_ms(p1_tpot_ms),
        }
    )

    # InferenceMAX headline x = median_intvty; legacy (session-level) x is
    # preserved so historical plot helpers keep working.
    row["pareto_x_median_intvty"] = row.get("median_intvty")
    row["pareto_x_median_tps_per_user"] = row.get("median_tps_per_user")
    row["pareto_y_output_tps_per_gpu"] = row.get("output_tps_per_gpu")
    row["output_tps_per_aggregate_1gpu_equiv"] = agg

    try:
        import torch

        if torch.cuda.is_available():
            ng = torch.cuda.device_count()
            row["output_tps_per_cuda_device_count"] = (
                (agg / ng) if (agg is not None and ng > 0) else None
            )
    except Exception:
        pass

    return row


def _failed_run_row(
    *,
    exc: BaseException,
    total_sessions: int,
    concurrency: int,
    max_batch_size: int,
    ladder_index: int,
    ladder_step: int,
    wall_s: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ladder_index": ladder_index,
        "ladder_step": ladder_step,
        "total_sessions": total_sessions,
        "concurrency": concurrency,
        "max_batch_size": max_batch_size,
        "status": "failed",
        "error": repr(exc),
        "error_traceback": traceback.format_exception_only(type(exc), exc),
        "wall_clock_s": wall_s,
    }
    if is_oom_exception(exc):
        row["error_kind"] = "out_of_memory"
    return row


# ---------------------------------------------------------------------------
# Trace metadata (single or multi-trace round-robin mix)
# ---------------------------------------------------------------------------


# Numeric trace_meta fields for which the round-robin per-session mean is the
# right aggregate when sessions span multiple traces. Plot helpers read the
# *_sum fields directly and multiply by N, so storing the per-session mean
# keeps "N * value" equal to the realised total work across all sessions.
_TRACE_META_NUMERIC = (
    "num_events",
    "assistant_output_tokens_sum",
    "assistant_turns",
    "prompt_tokens_assistant_sum",
    "completion_tokens_sum",
    "reasoning_tokens_sum",
    "non_assistant_message_tokens_sum",
    "tool_call_count",
    "tool_call_duration_ms_sum",
    "tool_call_duration_ms_max",
    "replay_tool_sleep_wall_s_estimated",
    "drop_kv_cache_events",
    "trace_file_size_bytes",
)


def _per_trace_meta_one(trace_dir: Path, trace_path: Path, trace: ExecutionTrace) -> Dict[str, Any]:
    return {
        "trace_dir": str(trace_dir),
        "trace_file": str(trace_path),
        "trace_id": trace.trace_id,
        "num_events": len(trace.events),
        "parallel_region_counts": count_parallel_regions(trace.events),
        "assistant_output_tokens_sum": count_assistant_completion_tokens(trace.events),
        **summarize_trace_events(trace.events),
        **collect_trace_file_stats(trace_path),
    }


def _build_trace_meta(
    trace_dirs: List[Path],
    trace_paths: List[Path],
    traces: List[ExecutionTrace],
    total_sessions: int,
) -> Dict[str, Any]:
    per_trace = [_per_trace_meta_one(d, p, t) for d, p, t in zip(trace_dirs, trace_paths, traces)]
    if len(per_trace) == 1:
        return per_trace[0]

    # Mix mode: round-robin per-session means for the numeric fields, joined
    # trace_id, plus a `traces` array carrying the per-trace details.
    K = len(per_trace)
    aggregate: Dict[str, Any] = {}
    for k in _TRACE_META_NUMERIC:
        vals = [m.get(k) for m in per_trace]
        if any(v is None for v in vals):
            continue
        aggregate[k] = sum(vals[i % K] for i in range(total_sessions)) / max(total_sessions, 1)
    aggregate["trace_id"] = "+".join(m["trace_id"] for m in per_trace)
    aggregate["mix_strategy"] = "round_robin_1to1"
    aggregate["mix_num_traces"] = K
    aggregate["traces"] = per_trace
    return aggregate


# ---------------------------------------------------------------------------
# Banner / startup diagnostics
# ---------------------------------------------------------------------------


def _print_banner(
    *,
    args: argparse.Namespace,
    trace_dirs: List[Path],
    trace_paths: List[Path],
    output_json: Path,
    output_log: Path,
) -> None:
    width = 78
    bar = "#" * width
    rule = "=" * width
    cli = args_to_dict(args)
    print("", flush=True)
    print(bar, flush=True)
    print(
        "#" + " TRACE REPLAY CLIENT (single ladder step) ".center(width - 2) + "#",
        flush=True,
    )
    print(bar, flush=True)
    print(f"  hostname          : {socket.gethostname()}", flush=True)
    print(f"  cwd               : {os.getcwd()}", flush=True)
    print(f"  command           : {shlex.join(sys.argv)}", flush=True)
    print(rule, flush=True)
    print(f"  base_url          : {args.base_url}", flush=True)
    print(f"  model             : {args.model}", flush=True)
    if len(trace_dirs) == 1:
        print(f"  trace_dir         : {trace_dirs[0]}", flush=True)
        print(f"  trace_file        : {trace_paths[0]}", flush=True)
    else:
        print(f"  trace_dirs ({len(trace_dirs)}, round-robin 1:1):", flush=True)
        for d, p in zip(trace_dirs, trace_paths):
            print(f"    - {d} -> {p.name}", flush=True)
    print(f"  output_json       : {output_json}", flush=True)
    print(f"  output_log        : {output_log}", flush=True)
    print(f"  ladder_index      : {args.ladder_index}", flush=True)
    print(f"  ladder_step       : {args.ladder_step}", flush=True)
    print(f"  total_sessions (N): {args.total_sessions}   # total trace replays", flush=True)
    print(f"  concurrency    (C): {args.concurrency}   # max in-flight replays", flush=True)
    print(f"  max_batch_size (B): {args.max_batch_size}   # server scheduler cap", flush=True)
    print(f"  tensor_parallel   : {args.tensor_parallel_size}", flush=True)
    print(f"  moe_expert_parallel: {args.moe_expert_parallel_size}", flush=True)
    print(f"  enable_attention_dp: {bool(args.enable_attention_dp)}", flush=True)
    print(f"  enable_chunked_prefill: {bool(args.enable_chunked_prefill)}", flush=True)
    print(f"  warmup            : {'off' if args.no_warmup else 'on'}", flush=True)
    _jitter_s = float(getattr(args, "arrival_jitter_s", 0.0))
    if _jitter_s > 0.0:
        print(
            f"  arrival_jitter    : U[0, {_jitter_s:.3f}s), "
            f"seed={int(getattr(args, 'arrival_jitter_seed', 0))}",
            flush=True,
        )
    else:
        print("  arrival_jitter    : off (all sessions start at t=0)", flush=True)
    print(rule, flush=True)
    for key in sorted(cli.keys()):
        print(f"  {key:28s} = {cli[key]!r}", flush=True)
    print(bar + "\n", flush=True)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


async def run_client(args: argparse.Namespace) -> int:
    trace_dirs = [Path(d).expanduser().resolve() for d in args.trace_dir]
    trace_paths = [find_compact_trace_file(d) for d in trace_dirs]
    traces = [ExecutionTrace.load(str(p)) for p in trace_paths]

    client = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url=args.base_url,
        timeout=args.request_timeout_s,
    )
    worker = TRTOpenaiWorker(client, args.model, kv_cache_hint_enabled=False)

    total_sessions = int(args.total_sessions)
    concurrency = int(args.concurrency)
    max_batch_size = int(args.max_batch_size)
    ladder_index = int(args.ladder_index)
    ladder_step = int(args.ladder_step)
    if total_sessions <= 0:
        raise ValueError(f"--total_sessions must be > 0 (got {total_sessions})")
    if concurrency <= 0:
        raise ValueError(f"--concurrency must be > 0 (got {concurrency})")
    if max_batch_size <= 0:
        raise ValueError(f"--max_batch_size must be > 0 (got {max_batch_size})")

    # Per-trace shared system-prompt token-id caches. Keying at the trace
    # level (rather than a single global dict) is required because the four
    # default SWE-bench traces all use ``conversation_id == 0``; a single
    # global ``Dict[int, List[int]]`` would let one trace overwrite another
    # trace's entry for ``conv_id == 0``. One dict per trace mirrors the
    # per-trace structure of the server's prefix block tree.
    trace_system_token_caches: List[Dict[int, List[int]]] = [{} for _ in traces]

    # When ``--pass_seed_namespace`` is non-empty, every per-session and
    # per-system-prefix RNG is seeded by SHA-256 of (namespace, ...keys).
    # Two consequences fall out of this:
    #
    #   * Pass-2 of the same session, run as a separate process with the
    #     same namespace, generates a token stream identical to pass-1 — so
    #     the radix-tree lookups for pass-2 prompts walk the chain pass-1
    #     committed instead of branching off at the first random redraw.
    #   * Within one process, two independent runs with the same namespace
    #     are bit-identical, which is what makes the with-drop /
    #     without-drop comparison clean of RNG noise.
    #
    # Empty namespace preserves the historical behaviour (module-level
    # ``random``, non-deterministic across processes).
    pass_seed_namespace = getattr(args, "pass_seed_namespace", "") or ""
    deterministic_replay = bool(pass_seed_namespace)
    if deterministic_replay:
        print(
            f"[determinism] seeded RNG enabled (--pass_seed_namespace={pass_seed_namespace!r})",
            flush=True,
        )
    else:
        print(
            "[determinism] seeded RNG disabled "
            "(--pass_seed_namespace not set); RNG draws are not "
            "reproducible across processes",
            flush=True,
        )

    # Pre-generate the synthetic token ids for every trace's system event(s)
    # up front, regardless of whether warmup is enabled. Populating the
    # per-trace caches here means every measurement-burst session of trace
    # ``k`` — whether warmup ran or not — resolves ``role == "system"`` to
    # the same deterministic id stream, so client-side token generation
    # cost does not appear inside the burst's wall clock, and ``--no_warmup``
    # reproduces the historical cold-cache behaviour only on the server
    # side (client-side cross-session sharing is always on).
    for ti, trace in enumerate(traces):
        for conv_id, tok_count in _extract_trace_system_segments(trace):
            if conv_id not in trace_system_token_caches[ti]:
                if deterministic_replay:
                    sys_rng = _build_seeded_rng(("system", pass_seed_namespace, ti, conv_id))
                    trace_system_token_caches[ti][conv_id] = _random_token_ids(tok_count, sys_rng)
                else:
                    trace_system_token_caches[ti][conv_id] = _random_token_ids(tok_count)

    # Number of attention DP ranks. With ``--enable_attention_dp`` the
    # server splits the KV cache per rank (the ``tp_size`` attention layers
    # run data-parallel, each with its own radix tree), so warmup must
    # populate every rank explicitly via the ``attention_dp_rank`` pin.
    # Without attention DP there is a single shared cache tree and pinning
    # to rank 0 for each trace is equivalent to the old behaviour.
    dp_size = int(args.tensor_parallel_size) if args.enable_attention_dp else 1

    # Warmup: one rank-pinned, single-token completion per (DP rank, trace,
    # system event) triple. ``attention_dp_relax=False`` makes the pin
    # strict, so trtllm-serve's ADP router holds the request for the
    # target rank instead of load-balancing it elsewhere — which is what
    # makes the "every rank gets every trace's system prefix" guarantee
    # actually hold end-to-end. Warmup cost is recorded in
    # ``run_row.warmup_wall_s``; it does not participate in any throughput
    # or session-duration metric.
    warmup_wall_s: Optional[float] = None
    if not args.no_warmup:
        warmup_tasks: List[asyncio.Future] = []
        # All pinned warmup requests can run in parallel; the per-rank
        # ``max_num_active_requests`` limit is enforced server-side.
        warmup_sem = asyncio.Semaphore(dp_size * len(traces))
        for dp_rank in range(dp_size):
            for ti, trace in enumerate(traces):
                for conv_id, _ in _extract_trace_system_segments(trace):
                    prompt_tokens = trace_system_token_caches[ti][conv_id]
                    if not prompt_tokens:
                        continue
                    warmup_tasks.append(
                        _warmup_pinned_prefill(
                            client,
                            args.model,
                            prompt_token_ids=prompt_tokens,
                            attention_dp_rank=dp_rank,
                            trace_index=ti,
                            conv_id=conv_id,
                            semaphore=warmup_sem,
                            request_timeout_s=args.request_timeout_s,
                        )
                    )
        warmup_t0 = time.perf_counter()
        print(
            f"[warmup] dispatching {len(warmup_tasks)} pinned prefills "
            f"(dp_size={dp_size} x traces={len(traces)})...",
            flush=True,
        )
        try:
            await asyncio.gather(*warmup_tasks)
        except Exception as warmup_exc:
            # Failing warmup should not sink the ladder step: log and
            # continue into the measurement phase. The measurement burst
            # will surface any fundamental server-side failure on its own.
            print(
                f"[warmup] failed: {warmup_exc!r}; continuing into measurement burst anyway",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
        warmup_wall_s = time.perf_counter() - warmup_t0
        print(
            f"[warmup] done in {warmup_wall_s:.3f}s; "
            f"system_token_cache sizes = "
            f"{[len(c) for c in trace_system_token_caches]}",
            flush=True,
        )

    # Pre-compute per-session arrival-time offsets. Drawn from
    # Uniform[0, args.arrival_jitter_s) with a seed separate from any
    # other RNG so jitter is reproducible across reruns independently of
    # e.g. the synthetic-token RNG used by _random_token_ids. When
    # arrival_jitter_s == 0.0 every delay is 0.0 and _one_session's
    # `if arrival_delay_s > 0.0` guard short-circuits; preserves the
    # historical lockstep behaviour bit-for-bit.
    arrival_jitter_s = float(getattr(args, "arrival_jitter_s", 0.0))
    arrival_jitter_seed = int(getattr(args, "arrival_jitter_seed", 0))
    if arrival_jitter_s > 0.0:
        jitter_rng = random.Random(arrival_jitter_seed)
        arrival_delays = [jitter_rng.uniform(0.0, arrival_jitter_s) for _ in range(total_sessions)]
        print(
            f"[arrival_jitter] U[0, {arrival_jitter_s:.3f}s), seed={arrival_jitter_seed}, "
            f"N={total_sessions} -> max_offset={max(arrival_delays):.3f}s, "
            f"mean={sum(arrival_delays) / len(arrival_delays):.3f}s",
            flush=True,
        )
    else:
        arrival_delays = [0.0] * total_sessions

    # Per-session :data:`RngFactory` builders.  When deterministic_replay
    # is on, the factory's seed key is
    # ``("session", namespace, trace_index, session_index)`` plus the
    # branch_path that the QueueManager passes in for each child executor;
    # when off, we pass ``rng_factory=None`` to ReplayEngine so it uses its
    # own non-deterministic default (back-compat).
    def _session_rng_factory(trace_index: int, session_index: int) -> Optional[RngFactory]:
        if not deterministic_replay:
            return None
        return make_seeded_rng_factory("session", pass_seed_namespace, trace_index, session_index)

    # Drop-path verification harness (P0.6) — opt in via
    # ``--verify_drop_path``.  When on, each session is given its own
    # :class:`DropPathStats` ledger; the ReplayEngine's drop_kv_cache
    # handler emits a sentinel ``max_tokens=1`` probe before and after
    # every truncate, the probe's request_id is recorded, and downstream
    # the perf-metrics drain joins ``num_reused_blocks`` /
    # ``free_num_blocks`` onto each probe so an offline analyzer can
    # mechanically prove the truncate freed the blocks it asked to free.
    # Off by default — issuing 2*N probes per drop is cheap (1 token
    # each) but doubles the request count, so headline N=512 runs leave
    # it off and the dedicated single-session verification driver turns
    # it on.
    verify_drop_path = bool(getattr(args, "verify_drop_path", False))
    if verify_drop_path:
        print(
            "[verify_drop_path] sentinel probes enabled "
            "(2 probes per dropped conv id per drop event)",
            flush=True,
        )
        per_session_drop_stats: List[Optional[DropPathStats]] = [
            DropPathStats() for _ in range(total_sessions)
        ]
    else:
        per_session_drop_stats = [None] * total_sessions

    # Post-run long-term retention probes — opt in via
    # ``--retention_probe``.  During the run, each QueueExecutor saves
    # one probe definition (branch_path, conv_id, prefix) per conv id at
    # sentinel time into its RetentionProbeStats.  After ALL sessions
    # complete, the client fires every saved probe sequentially through
    # the worker so every probe observes the same post-workload radix
    # tree, measuring long-term retention rather than the transient
    # per-session-end snapshot.
    retention_probe = bool(getattr(args, "retention_probe", False))
    if retention_probe:
        print(
            "[retention_probe] post-run long-term probes enabled "
            "(prefixes saved during run, fired after all sessions complete)",
            flush=True,
        )
        per_session_retention_stats: List[Optional[RetentionProbeStats]] = [
            RetentionProbeStats() for _ in range(total_sessions)
        ]
    else:
        per_session_retention_stats = [None] * total_sessions

    # Optional KV-cache event ring drain.  When --kv_cache_events_jsonl is
    # set, a background asyncio task polls POST /kv_cache_events every
    # ``--kv_cache_events_drain_interval_s`` seconds (default 2.0) and
    # appends every event as one JSONL line.  Server-side ring is sized via
    # the launch_server.sh ``--kv_cache_event_buffer_max_size`` flag; the
    # client just drains and persists.
    kv_events_path_arg = getattr(args, "kv_cache_events_jsonl", None)
    kv_events_path = Path(kv_events_path_arg) if kv_events_path_arg else None
    kv_events_interval = float(getattr(args, "kv_cache_events_drain_interval_s", 2.0))
    kv_events_stop = asyncio.Event() if kv_events_path is not None else None
    kv_events_task = None
    if kv_events_path is not None:
        kv_events_task = asyncio.create_task(
            _kv_cache_event_drain_loop(
                base_url=args.base_url,
                api_key=args.openai_api_key,
                output_path=kv_events_path,
                interval_s=kv_events_interval,
                stop_event=kv_events_stop,
            )
        )
        print(
            f"[kv_cache_events] drain task started; output={kv_events_path} "
            f"interval={kv_events_interval}s",
            flush=True,
        )

    semaphore = asyncio.Semaphore(concurrency)
    wall_t0 = time.perf_counter()
    try:
        results = await asyncio.gather(
            *[
                _one_session(
                    worker,
                    traces[i % len(traces)],
                    semaphore=semaphore,
                    session_index=i,
                    trace_index=i % len(traces),
                    total_sessions=total_sessions,
                    concurrency=concurrency,
                    max_batch_size=max_batch_size,
                    ladder_step=ladder_step,
                    system_token_cache=trace_system_token_caches[i % len(traces)],
                    rng_factory=_session_rng_factory(i % len(traces), i),
                    drop_path_stats=per_session_drop_stats[i],
                    retention_probe_stats=per_session_retention_stats[i],
                    arrival_delay_s=arrival_delays[i],
                )
                for i in range(total_sessions)
            ]
        )
        wall_s = time.perf_counter() - wall_t0
        run_row = compute_run_row(
            traces=traces,
            results=results,
            wall_s=wall_s,
            total_sessions=total_sessions,
            concurrency=concurrency,
            max_batch_size=max_batch_size,
            ladder_index=ladder_index,
            ladder_step=ladder_step,
            tensor_parallel_size=int(args.tensor_parallel_size),
        )
        exit_code = 0
    except Exception as exc:
        wall_s = time.perf_counter() - wall_t0
        print(f"Replay failed: {exc!r}", file=sys.stderr)
        traceback.print_exc()
        run_row = _failed_run_row(
            exc=exc,
            total_sessions=total_sessions,
            concurrency=concurrency,
            max_batch_size=max_batch_size,
            ladder_index=ladder_index,
            ladder_step=ladder_step,
            wall_s=wall_s,
        )
        exit_code = 1

    # Exposed regardless of success/failure so downstream analysis can
    # always tell whether a warmup ran for this step, and if so, how long
    # it took. ``None`` means warmup was disabled via ``--no_warmup``.
    run_row["warmup_wall_s"] = warmup_wall_s

    # ------------------------------------------------------------------
    # Post-run retention probe phase: fire all saved probes NOW, after
    # every session has completed.  Every probe observes the same
    # post-workload radix tree, so the retention rate answers "after the
    # full N-session burst, how much of session i's Supervisor chain
    # survived?"  This is the long-term retention measurement.
    #
    # Probes fire in REVERSE admission order (session N-1 first, session 0
    # last). Rationale: each probe is a real ``max_tokens=1, ignore_eos=True``
    # /v1/completions request — the engine commits the probe's prefix to the
    # radix tree on completion, evicting whatever was there. Firing the
    # latest session first lets it observe its own post-burst chain BEFORE
    # any other probe writes interfere; earlier sessions then probe a
    # progressively-polluted cache, but their burst-end content was already
    # LRU-evicted long before the probe phase started, so the polluted
    # measurement is no worse than the truth for them.
    # The server walks the radix tree for the prefix and reports
    # ``num_reused_blocks`` in its perf record, which downstream becomes
    # the retention rate numerator.
    # ------------------------------------------------------------------
    if retention_probe:
        total_pending = sum(
            len(s.pending_probes) for s in per_session_retention_stats if s is not None
        )
        print(
            f"[retention_probe] all {total_sessions} sessions done; "
            f"firing {total_pending} post-run probes (reverse session order) ...",
            flush=True,
        )
        probe_t0 = time.perf_counter()
        probes_fired = 0
        probes_failed = 0
        for sess_idx in range(len(per_session_retention_stats) - 1, -1, -1):
            stats = per_session_retention_stats[sess_idx]
            if stats is None:
                continue
            for pending in stats.pending_probes:
                probe = GenerationTask(
                    input_tokens=pending["prefix"],
                    max_tokens=1,
                    ignore_eos=True,
                )
                status = await worker.run_task(probe)
                if status != TaskStatus.SUCCESS:
                    probes_failed += 1
                    print(
                        f"[retention_probe] WARNING: probe failed for "
                        f"session={sess_idx} branch={pending['branch_path']} "
                        f"conv={pending['conv_id']}: {status}",
                        flush=True,
                    )
                    continue
                stats.record(
                    branch_path=tuple(pending["branch_path"]),
                    conv_id=pending["conv_id"],
                    request_id=probe.request_id,
                    prefix_len=len(pending["prefix"]),
                )
                probes_fired += 1
        probe_wall = time.perf_counter() - probe_t0
        print(
            f"[retention_probe] {probes_fired} probes fired in "
            f"{probe_wall:.1f}s" + (f" ({probes_failed} failed)" if probes_failed else ""),
            flush=True,
        )

    # Stop the KV-cache event drain task BEFORE the /perf_metrics drain so
    # the JSONL is closed cleanly while requests are still settled.  The
    # final drain inside the loop captures any events from the retention
    # probe phase that hadn't been flushed yet.
    if kv_events_task is not None and kv_events_stop is not None:
        kv_events_stop.set()
        try:
            await asyncio.wait_for(kv_events_task, timeout=30.0)
        except asyncio.TimeoutError:
            print("[kv_cache_events] drain task did not exit in 30s; cancelling", flush=True)
            kv_events_task.cancel()

    # Drain trtllm-serve's /perf_metrics deque ONCE after the
    # measurement burst AND the post-run retention probes have fired.
    # The drain captures perf records for both the real generation
    # requests and the retention probes, so downstream joins by
    # ``request_id`` pick up ``num_reused_blocks`` for every probe.
    perf_metrics_drain_counters: Dict[str, int] = {}
    detail_entries = run_row.get("replay_assistant_generations_detail")
    drop_probe_entries: List[Dict[str, Any]] = []
    if verify_drop_path:
        for s in per_session_drop_stats:
            if s is None:
                continue
            drop_probe_entries.extend(s.records)

    # Collect retention probe records (populated by the post-run firing
    # phase above) and stamp each with its owning session_index.
    retention_probe_entries: List[Dict[str, Any]] = []
    if retention_probe:
        for sess_idx, s in enumerate(per_session_retention_stats):
            if s is None:
                continue
            for rec in s.records:
                rec_with_session = dict(rec)
                rec_with_session["session_index"] = sess_idx
                rec_with_session["trace_index"] = sess_idx % max(len(traces), 1)
                retention_probe_entries.append(rec_with_session)

    if isinstance(detail_entries, list):
        try:
            perf_records = await _drain_perf_metrics(args.base_url, args.openai_api_key)
        except Exception as drain_exc:
            print(
                f"[perf_metrics] drain failed: {drain_exc!r}; "
                "per-call rows will not carry kv_cache_hit_rate",
                file=sys.stderr,
                flush=True,
            )
            perf_records = []
            perf_metrics_drain_counters = {"drain_error": 1}
        else:
            perf_metrics_drain_counters = _augment_entries_with_perf_metrics(
                detail_entries, perf_records
            )
            # Drop-path probes share the same /perf_metrics drain (every
            # request — generation or probe — appends to the deque server-side
            # in arrival order), so we can join them with the same lookup.
            if drop_probe_entries:
                probe_counters = _augment_entries_with_perf_metrics(
                    drop_probe_entries, perf_records
                )
                run_row["drop_path_probe_drain_counters"] = probe_counters
            # Retention probes piggyback on the same drain.
            if retention_probe_entries:
                retention_counters = _augment_entries_with_perf_metrics(
                    retention_probe_entries, perf_records
                )
                run_row["retention_probe_drain_counters"] = retention_counters
            print(
                f"[perf_metrics] drained "
                f"{perf_metrics_drain_counters['perf_records']} records, "
                f"matched {perf_metrics_drain_counters['entries_matched']} of "
                f"{perf_metrics_drain_counters['entries_total']} per-call rows"
                + (
                    f"; {len(drop_probe_entries)} drop-path probes recorded"
                    if drop_probe_entries
                    else ""
                )
                + (
                    f"; {len(retention_probe_entries)} retention probes recorded"
                    if retention_probe_entries
                    else ""
                ),
                flush=True,
            )
    run_row["perf_metrics_drain_counters"] = perf_metrics_drain_counters
    run_row["drop_path_probes"] = drop_probe_entries
    run_row["verify_drop_path_enabled"] = verify_drop_path
    run_row["retention_probes"] = retention_probe_entries
    run_row["retention_probe_enabled"] = retention_probe

    run_row["llm_effective_config"] = {
        "backend": "trtllm-serve",
        "base_url": args.base_url,
        "model": args.model,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "moe_expert_parallel_size": int(args.moe_expert_parallel_size),
        "enable_attention_dp": bool(args.enable_attention_dp),
        "enable_chunked_prefill": bool(args.enable_chunked_prefill),
        "max_batch_size": max_batch_size,
        "total_sessions": total_sessions,
        "concurrency": concurrency,
        "ladder_index": ladder_index,
        "ladder_step": ladder_step,
        "warmup_enabled": not bool(args.no_warmup),
        "arrival_jitter_s": float(getattr(args, "arrival_jitter_s", 0.0)),
        "arrival_jitter_seed": int(getattr(args, "arrival_jitter_seed", 0)),
        "pass_seed_namespace": pass_seed_namespace,
        "deterministic_replay": deterministic_replay,
    }

    # Self-describing step JSON: embeds enough metadata that the aggregator
    # can reconstruct a combined v4 report without re-reading the trace.
    step_record: Dict[str, Any] = {
        "schema": STEP_SCHEMA,
        "run_started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cli_argv": sys.argv,
        "cli_args": args_to_dict(args),
        "base_url": args.base_url,
        "model": args.model,
        "artifact_naming": {
            "model_name": Path(os.path.expanduser(args.model or "")).name or "model",
            "tensor_parallel_size": int(args.tensor_parallel_size),
            "moe_expert_parallel_size": int(args.moe_expert_parallel_size),
            "enable_attention_dp": bool(args.enable_attention_dp),
            "filename_suffix": pareto_config_filename_suffix(
                args.model,
                int(args.tensor_parallel_size),
                int(args.moe_expert_parallel_size),
                enable_attention_dp=bool(args.enable_attention_dp),
            ),
        },
        "trace_dir": (str(trace_dirs[0]) if len(trace_dirs) == 1 else [str(d) for d in trace_dirs]),
        "trace_file": (
            str(trace_paths[0]) if len(trace_paths) == 1 else [str(p) for p in trace_paths]
        ),
        "host": collect_host_info(),
        "trace_meta": _build_trace_meta(trace_dirs, trace_paths, traces, total_sessions),
        "run_row": run_row,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    output_json = args.output_json.expanduser().resolve()

    # Safety net: dump step_record to node-local /tmp BEFORE attempting the
    # Lustre write. Lustre OST flapping causes write()/close() on dirty
    # pages to hang indefinitely; both pickle and JSON go to NVMe first so
    # the driver's exfil step always has a recoverable copy.
    import pickle as _pkl

    _pid = os.getpid()
    _safe_pkl = Path("/tmp") / f"step_record_{_pid}.pkl"
    _safe_json = Path("/tmp") / f"step_record_{_pid}.json"
    try:
        with open(_safe_pkl, "wb") as _fp:
            _pkl.dump(step_record, _fp)
        print(f"[safety_net] wrote {_safe_pkl}", flush=True)
    except Exception as _e:
        print(f"[safety_net] pickle FAILED: {_e}", flush=True)
    try:
        import json as _json

        with open(_safe_json, "w", encoding="utf-8") as _fp:
            _json.dump(step_record, _fp, indent=2, ensure_ascii=False, default=str)
        print(f"[safety_net] wrote {_safe_json}", flush=True)
    except Exception as _e:
        print(f"[safety_net] json FAILED: {_e}", flush=True)

    # Best-effort Lustre write in a daemon thread so a hung OST cannot
    # prevent process exit. The bash exfil step copies _safe_json to the
    # Lustre output dir as the authoritative source of truth.
    import threading

    def _try_lustre_write():
        try:
            atomic_write_json(output_json, step_record)
            print(f"Wrote {output_json}", flush=True)
        except Exception as _e:
            print(f"[lustre_write] failed: {_e}", flush=True)

    _t = threading.Thread(target=_try_lustre_write, daemon=True)
    _t.start()
    _t.join(timeout=60)
    if _t.is_alive():
        print(
            f"[lustre_write] timeout after 60s; rely on safety_net /tmp/{_safe_json.name}",
            flush=True,
        )
    return exit_code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Pareto trace replay against an external trtllm-serve (single ladder step).")
    )
    p.add_argument(
        "--base_url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL exposed by trtllm-serve.",
    )
    p.add_argument(
        "--openai_api_key",
        type=str,
        default="tensorrt_llm",
        help="Placeholder API key (trtllm-serve does not authenticate).",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model identifier as exposed by trtllm-serve (usually the "
            "--model argument that was passed to trtllm-serve, i.e. a "
            "checkpoint directory path)."
        ),
    )
    p.add_argument(
        "--trace_dir",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more directories each containing a *.trace.json. With more "
            "than one, sessions are assigned round-robin (session i uses "
            "trace[i %% K]) for a 1:1:...:1 mix; per-session trace metadata in "
            "the step JSON is the round-robin per-session mean."
        ),
    )
    p.add_argument(
        "--total_sessions",
        type=int,
        required=True,
        help=(
            "Total number of trace replays to execute at this ladder step "
            "(each session replays the full compact trace exactly once)."
        ),
    )
    p.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help=(
            "Maximum number of trace replays in flight at any moment "
            "(an asyncio.Semaphore gates admission). When concurrency == "
            "total_sessions every session starts at t=0; when concurrency < "
            "total_sessions the remaining sessions wait their turn."
        ),
    )
    p.add_argument(
        "--max_batch_size",
        type=int,
        required=True,
        help=(
            "trtllm-serve --max_batch_size this step was started with; "
            "recorded in the step JSON as metadata. The actual capacity is "
            "enforced server-side and is independent of --concurrency and "
            "--total_sessions."
        ),
    )
    p.add_argument(
        "--ladder_index",
        type=int,
        default=0,
        help="Zero-based index of this ladder step within the full ladder.",
    )
    p.add_argument(
        "--ladder_step",
        type=int,
        required=True,
        help=(
            "Ladder identifier for this step (recorded verbatim and used for "
            "sorting/annotation by the aggregator and plot helpers). Not "
            "tied to any of --total_sessions / --concurrency / --max_batch_size."
        ),
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="TP size the server is running with (metadata, used for per-GPU normalization).",
    )
    p.add_argument(
        "--moe_expert_parallel_size",
        type=int,
        default=4,
        help="MoE expert-parallel size the server is running with (metadata).",
    )
    p.add_argument(
        "--enable_attention_dp",
        action="store_true",
        default=False,
        help=(
            "Pass this when trtllm-serve was launched with "
            "--enable_attention_dp (attention-DP + MoE-EP configuration, "
            'e.g. "DP4 + EP4" on a 4-GPU node). Recorded in '
            "artifact_naming so the Pareto aggregator emits the "
            "matching '_adp' filename suffix."
        ),
    )
    p.add_argument(
        "--enable_chunked_prefill",
        action="store_true",
        default=False,
        help=(
            "Pass this when trtllm-serve was launched with "
            "--enable_chunked_prefill. Recorded in llm_effective_config "
            "(and lifted into llm_fixed_config by the aggregator) so "
            "with-vs-without-chunked-prefill sweeps remain distinguishable "
            "in the archived JSON. Orthogonal to the parallelism topology, "
            "so it does NOT participate in the artifact-naming filename "
            "suffix."
        ),
    )
    p.add_argument(
        "--no_warmup",
        action="store_true",
        default=False,
        help=(
            "Disable the warmup phase. By default, the client runs one "
            "concurrent replay per unique trace before the measurement "
            "burst, sharing the per-trace system_token_cache so the "
            "server's prefix block tree is primed with each trace's "
            "system-prompt token ids (and, transitively, the full "
            "single-session prefix tree of every turn). Warmup results are "
            "not counted into run_row's throughput/session figures; only "
            "the wall-clock cost is recorded under run_row.warmup_wall_s. "
            "Pass --no_warmup to reproduce the historical "
            "cold-start-on-every-step behaviour."
        ),
    )
    p.add_argument(
        "--arrival_jitter_s",
        type=float,
        default=0.0,
        help=(
            "Per-session arrival-time jitter (seconds). Before each "
            "measurement-burst session acquires the concurrency "
            "semaphore, sleep U[0, arrival_jitter_s) (seeded by "
            "--arrival_jitter_seed) so the K round-robin clones of each "
            "trace are desynchronised in phase and the server does not "
            "see lock-step waves of prefill/decode. Must be >= 0; the "
            "default of 0 reproduces the historical simultaneous-start "
            "behaviour bit-for-bit. Reasonable values are on the order "
            "of one average per-turn wall clock (~60 s for the "
            "SWE-bench mix) so by the time the last session issues its "
            "first LLM call, the first session is already past its "
            "first call into subsequent turns, smoothing the server-side "
            "batch occupancy to its decode_fraction * C ceiling. "
            "Warmup is not jittered (it deliberately fires ranks in "
            "parallel to prime the radix tree). Only the measurement "
            "burst sessions see this delay."
        ),
    )
    p.add_argument(
        "--arrival_jitter_seed",
        type=int,
        default=0,
        help=(
            "RNG seed for the per-session --arrival_jitter_s draws. "
            "Kept separate from other seeds so arrival timing is "
            "reproducible across reruns independently of the synthetic "
            "token RNG."
        ),
    )
    p.add_argument(
        "--verify_drop_path",
        action="store_true",
        default=False,
        help=(
            "Enable the drop-path verification harness (P0.6). For every "
            "drop_kv_cache trace event, the ReplayEngine emits a "
            "max_tokens=1 sentinel probe with the about-to-be-dropped "
            "prefix BEFORE the truncate, then a second probe AFTER. The "
            "client drains /perf_metrics at end-of-run and joins every "
            "probe's request_id back to its num_reused_blocks / "
            "free_num_blocks server-side accounting. An offline analyzer "
            "(tools/verify_drop_path.py) then asserts that "
            "after.num_reused_blocks <= ceil(num_tokens_to_keep / "
            "tokens_per_block) per drop, mechanically proving the truncate "
            "actually freed the blocks the engine asked it to. Off by "
            "default because the probes double the per-drop request count; "
            "use the dedicated single-session verification driver."
        ),
    )
    p.add_argument(
        "--retention_probe",
        action="store_true",
        default=False,
        help=(
            "Enable post-run long-term retention probes (the headline "
            "measurement for the proactive KV-cache drop experiment). "
            "During the run, each QueueExecutor saves the probe prefix "
            "(conv id's last assistant-call input) at sentinel time. "
            "After ALL sessions complete, the client fires every saved "
            "probe sequentially so every probe observes the same "
            "post-workload radix tree, measuring long-term retention. "
            "num_reused_blocks (drained from /perf_metrics) directly "
            "answers 'after the full burst, how much of session i's "
            "chain survived in cache?'. Records land in "
            "run_row.retention_probes with branch_path / conv_id / "
            "session_index / prefix_len + the perf-metrics fields. "
            "Cheap: ~5–10 probes per session vs the burst's ~30 real "
            "generation calls. Off by default — turn on for the "
            "headline cross-condition sweep."
        ),
    )
    p.add_argument(
        "--kv_cache_events_jsonl",
        type=str,
        default=None,
        help=(
            "Path to write KV-cache event ring drain output. When set, a "
            "background asyncio task polls POST /kv_cache_events every "
            "--kv_cache_events_drain_interval_s seconds and appends each "
            "event as one JSONL line. Server-side ring must be enabled "
            "by passing --kv_cache_event_buffer_max_size N>0 to "
            "launch_server.sh (the run_proactive_drop_single_pass.sh "
            "driver wires this through). Off by default."
        ),
    )
    p.add_argument(
        "--kv_cache_events_drain_interval_s",
        type=float,
        default=2.0,
        help=(
            "Polling interval for the KV-cache event drain background "
            "task (default 2.0s). Only used when --kv_cache_events_jsonl "
            "is set."
        ),
    )
    p.add_argument(
        "--pass_seed_namespace",
        type=str,
        default="",
        help=(
            "Opt in to deterministic per-session RNG seeding. When set to "
            "any non-empty string, every per-session and per-system-prefix "
            "RNG draw is keyed off SHA-256(namespace, ...keys), so two "
            "client invocations that pass the same namespace generate "
            "byte-identical token streams for every (trace_index, "
            "session_index) pair. This is the prerequisite for the "
            "two-pass proactive KV-cache drop measurement: pass-2 of "
            "session i can only hit pass-1's radix-tree blocks if the "
            "prompt tokens it sends are the same ones pass-1 committed. "
            "Empty string (the default) reproduces the historical "
            "non-deterministic behaviour bit-for-bit."
        ),
    )
    p.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Path for the single-step JSON output.",
    )
    p.add_argument(
        "--request_timeout_s",
        type=float,
        default=3600.0,
        help="Per-HTTP-request timeout for the OpenAI client.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json.expanduser().resolve()
    output_log = output_json.with_suffix(".log")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(output_log, "w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeTextIO(original_stdout, log_fp)
    sys.stderr = TeeTextIO(original_stderr, log_fp)
    try:
        trace_dirs = [Path(d).expanduser().resolve() for d in args.trace_dir]
        trace_paths = [find_compact_trace_file(d) for d in trace_dirs]
        _print_banner(
            args=args,
            trace_dirs=trace_dirs,
            trace_paths=trace_paths,
            output_json=output_json,
            output_log=output_log,
        )
        exit_code = asyncio.run(run_client(args))
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fp.close()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
