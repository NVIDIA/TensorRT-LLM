"""Per-session real KV-cache hit-rate report from a Pareto run's step JSON.

Companion to :mod:`cache_hit` (offline upper bound). Where ``cache_hit``
simulates an infinite cache against the trace JSON, this module reads the
**real, engine-measured** per-request hit/miss counts that the
trace_replay_client drained from /perf_metrics and merged into
``step<STEP>.json::run_row.replay_assistant_generations_detail``.

Each entry in ``replay_assistant_generations_detail`` is one LLM call
issued by one session as it walked the trace. We:

  1. Group entries by ``(trace_index, session_index)`` and sort each
     group by ``first_iter`` ascending. That order matches the trace's
     assistant-event order (sessions replay events sequentially).
  2. Zip each group to the trace's assistant events in index order.
  3. Per session: sum ``num_reused_blocks`` / ``num_missed_blocks`` over
     all its LLM calls and compute an aggregate block-hit rate.
  4. Per trace: sum across all sessions.
  5. Deep-copy the trace events and attach an ``observed_replays``
     array to each scored assistant event so the schema mirrors
     ``*.trace.cachehit.json`` (with N entries per event for N sessions).

Engine source of truth: ``KvCacheMeasure`` per request emits
``num_reused_blocks`` / ``num_missed_blocks`` from the C++ KV manager
(see ``tensorrt_llm/_torch/pyexecutor/py_executor.py:1154`` setting
``req_stat.kv_cache_hit_rate_per_request``). ``num_reused_blocks`` is
exactly the count of prefix blocks the request matched to existing
entries in the radix tree; ``num_missed_blocks`` is the count newly
allocated for that request's prompt. Hit rate at the block level is
``num_reused_blocks / (num_reused_blocks + num_missed_blocks)``.

Token-level fields (``cache_hit_tokens``, ``cache_miss_tokens``) are
approximated as ``blocks * tokens_per_block``. The engine does not
report a token-granular split per request, so the partial trailing
block beyond ``floor(prompt_tokens / tokens_per_block)`` is excluded
(matches the cache_hit upper-bound module's
``exclude_last_token_from_blocks=True`` default).

Branched traces: the per-session order-based zip assumes a single
linear walk of assistant events per session. If any detail entry has a
non-empty ``branch_path`` the result is best-effort and a warning is
attached.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "trace_replay_real_cache_hit.v1"
REAL_ANNOTATED_TRACE_SUFFIX = ".trace.realcachehit.json"

# Fields copied verbatim from each detail entry into the per-session
# per-event view (and into observed_replays on the annotated event).
DETAIL_FIELDS = (
    "request_id",
    "session_index",
    "trace_index",
    "branch_path",
    "agent_role",
    "kv_cache_hit_rate",
    "num_reused_blocks",
    "num_missed_blocks",
    "num_total_allocated_blocks",
    "num_new_allocated_blocks",
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "first_iter",
    "last_iter",
    "ttft_s",
    "latency_s",
    "server_arrival_time",
    "arrival_time",
    "first_scheduled_time",
    "first_token_time",
    "server_first_token_time",
    "last_token_time",
)


def _assistant_event_indices(events: List[Dict[str, Any]]) -> List[int]:
    return [
        i for i, e in enumerate(events)
        if isinstance(e, dict) and e.get("role") == "assistant"
    ]


def _project_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: detail[k] for k in DETAIL_FIELDS if k in detail}
    return out


def _group_details_by_session(
    details: List[Dict[str, Any]],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """Group by (trace_index, session_index); within group sort by first_iter."""
    by_session: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for d in details:
        ti = int(d.get("trace_index", 0) or 0)
        si = int(d.get("session_index", 0) or 0)
        by_session[(ti, si)].append(d)
    for k in by_session:
        by_session[k].sort(
            key=lambda d: (d.get("first_iter") if d.get("first_iter") is not None else
                           d.get("arrival_time") or 0.0))
    return by_session


def compute_real_cache_hit(
    step_json: Dict[str, Any],
    trace_data: Dict[str, Any],
    tokens_per_block: int = 32,
) -> Dict[str, Any]:
    """Build the real-cache-hit record from a run's step JSON + trace JSON.

    Args:
        step_json: Parsed ``step<STEP>.json`` (must contain
            ``run_row.replay_assistant_generations_detail``).
        trace_data: Parsed ``*.trace.json``.
        tokens_per_block: KV block size for token-level approximations
            (engine reports only block counts per request).

    Returns:
        A JSON-serializable dict with ``schema``, ``source``,
        ``algorithm``, ``trace_id``, ``events`` (annotated deep copy),
        ``sessions`` (per-session per-event rollup), and ``summary``
        (trace-level aggregate).
    """
    run_row = step_json.get("run_row") or {}
    details = run_row.get("replay_assistant_generations_detail") or []
    if not isinstance(details, list) or not details:
        raise ValueError(
            "step JSON has no run_row.replay_assistant_generations_detail")

    events = trace_data.get("events")
    if not isinstance(events, list):
        raise ValueError("trace JSON has no 'events' array")
    assistant_idx = _assistant_event_indices(events)
    n_assistant = len(assistant_idx)

    by_session = _group_details_by_session(details)

    warnings: List[str] = []
    if any(d.get("branch_path") for d in details):
        warnings.append(
            "branched_trace: at least one detail entry has non-empty branch_path; "
            "the per-event mapping assumes a single linear walk per session and "
            "may misalign for branched traces.")

    sessions_out: List[Dict[str, Any]] = []
    annotated_events = copy.deepcopy(events)

    for (trace_index, session_index), group in sorted(by_session.items()):
        if len(group) != n_assistant:
            warnings.append(
                f"session (trace_index={trace_index}, session_index={session_index}): "
                f"{len(group)} LLM calls but trace has {n_assistant} assistant events; "
                "per-event mapping will skip the unmatched tail.")

        per_event: List[Dict[str, Any]] = []
        sum_reused = sum_missed = 0
        for i, d in enumerate(group):
            ev_idx: Optional[int] = assistant_idx[i] if i < n_assistant else None
            row = _project_detail(d)
            row["assistant_event_index"] = ev_idx
            if ev_idx is not None and isinstance(events[ev_idx], dict):
                row["message_index"] = events[ev_idx].get("message_index")
            reused = int(d.get("num_reused_blocks") or 0)
            missed = int(d.get("num_missed_blocks") or 0)
            sum_reused += reused
            sum_missed += missed
            row["cache_hit_tokens"] = reused * tokens_per_block
            row["cache_miss_tokens"] = missed * tokens_per_block
            lookup_i = reused + missed
            row["cache_block_hit_rate"] = (reused /
                                           lookup_i) if lookup_i else 0.0
            per_event.append(row)

            if ev_idx is not None and isinstance(annotated_events[ev_idx],
                                                 dict):
                annotated_events[ev_idx].setdefault("observed_replays",
                                                    []).append(row)

        lookup = sum_reused + sum_missed
        sessions_out.append({
            "session_index": session_index,
            "trace_index": trace_index,
            "n_assistant_events": len(group),
            "total_cache_hit_blocks": sum_reused,
            "total_cache_miss_blocks": sum_missed,
            "total_lookup_blocks": lookup,
            "overall_cache_block_hit_rate":
            (sum_reused / lookup) if lookup else 0.0,
            "total_cache_hit_tokens": sum_reused * tokens_per_block,
            "total_cache_miss_tokens": sum_missed * tokens_per_block,
            "per_event": per_event,
        })

    # Trace-level aggregate over all sessions.
    total_reused = sum(s["total_cache_hit_blocks"] for s in sessions_out)
    total_missed = sum(s["total_cache_miss_blocks"] for s in sessions_out)
    total_lookup = total_reused + total_missed
    total_calls = sum(s["n_assistant_events"] for s in sessions_out)

    record = {
        "schema":
        SCHEMA,
        "created_at_utc":
        datetime.now(timezone.utc).isoformat(),
        "source": {
            "step_json_run_started_at_utc":
            step_json.get("run_started_at_utc"),
            "step_json_artifact_naming":
            step_json.get("artifact_naming"),
            "trace_file":
            step_json.get("trace_file"),
            "trace_dir":
            step_json.get("trace_dir"),
            "model":
            step_json.get("model"),
            "base_url":
            step_json.get("base_url"),
            "host":
            step_json.get("host"),
        },
        "algorithm": {
            "source":
            "engine_per_request_kv_cache_metrics",
            "tokens_per_block":
            tokens_per_block,
            "request_definition":
            "one entry per assistant LLM call recorded in "
            "step_json.run_row.replay_assistant_generations_detail",
            "session_to_event_mapping":
            "group details by (trace_index, session_index), sort by first_iter "
            "ascending, zip to assistant events in trace order",
            "hit_definition":
            "engine-reported num_reused_blocks / (num_reused_blocks + "
            "num_missed_blocks) from KvCacheMeasure (block-granular)",
            "token_estimate":
            "cache_{hit,miss}_tokens = num_{reused,missed}_blocks * "
            "tokens_per_block (excludes the partial trailing block, matches "
            "cache_hit upper-bound's exclude_last_token_from_blocks=True)",
        },
        "trace_id":
        trace_data.get("trace_id"),
        "n_sessions":
        len(sessions_out),
        "n_assistant_events":
        n_assistant,
        "events":
        annotated_events,
        "sessions":
        sessions_out,
        "summary": {
            "n_sessions":
            len(sessions_out),
            "n_assistant_events_per_session":
            n_assistant,
            "total_llm_calls":
            total_calls,
            "total_cache_hit_blocks":
            total_reused,
            "total_cache_miss_blocks":
            total_missed,
            "total_lookup_blocks":
            total_lookup,
            "overall_cache_block_hit_rate":
            (total_reused / total_lookup) if total_lookup else 0.0,
            "total_cache_hit_tokens":
            total_reused * tokens_per_block,
            "total_cache_miss_tokens":
            total_missed * tokens_per_block,
        },
        "warnings":
        warnings,
    }
    return record
