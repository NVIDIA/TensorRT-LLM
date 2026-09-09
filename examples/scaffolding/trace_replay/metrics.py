from __future__ import annotations

import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tensorrt_llm.scaffolding.trace_replay.execution_trace import ExecutionTrace, TraceEvent


def count_assistant_completion_tokens(events: List[TraceEvent]) -> int:
    total = 0
    for ev in events:
        if ev.event_type == "message" and ev.role == "assistant":
            total += ev.completion_tokens or 0
    return total


def count_parallel_regions(events: List[TraceEvent]) -> Dict[str, int]:
    starts = sum(1 for e in events if e.event_type == "parallel_start")
    ends = sum(1 for e in events if e.event_type == "parallel_end")
    return {"parallel_start": starts, "parallel_end": ends}


def summarize_trace_events(events: List[TraceEvent]) -> Dict[str, Any]:
    """Aggregate trace structure for JSON (token budgets, tools, roles)."""
    event_type_counts: Dict[str, int] = {}
    role_counts: Dict[str, int] = {}
    assistant_turns = 0
    prompt_tokens_assistant_sum = 0
    completion_tokens_sum = 0
    reasoning_tokens_sum = 0
    tool_calls = 0
    tool_duration_ms_sum = 0.0
    tool_duration_ms_max = 0.0
    message_tokens_sum = 0
    drop_kv = 0

    for ev in events:
        et = ev.event_type or ""
        event_type_counts[et] = event_type_counts.get(et, 0) + 1
        if et == "message" and ev.role:
            role_counts[ev.role] = role_counts.get(ev.role, 0) + 1
            if ev.tokens:
                message_tokens_sum += ev.tokens
        if et == "message" and ev.role == "assistant":
            assistant_turns += 1
            prompt_tokens_assistant_sum += ev.prompt_tokens or 0
            completion_tokens_sum += ev.completion_tokens or 0
            reasoning_tokens_sum += ev.reasoning_tokens or 0
        if et == "tool_call":
            tool_calls += 1
            duration_ms = ev.duration_ms or 0.0
            tool_duration_ms_sum += duration_ms
            tool_duration_ms_max = max(tool_duration_ms_max, duration_ms)
        if et == "drop_kv_cache":
            drop_kv += 1

    return {
        "event_type_counts": event_type_counts,
        "message_role_counts": role_counts,
        "assistant_turns": assistant_turns,
        "prompt_tokens_assistant_sum": prompt_tokens_assistant_sum,
        "completion_tokens_sum": completion_tokens_sum,
        "reasoning_tokens_sum": reasoning_tokens_sum,
        "non_assistant_message_tokens_sum": message_tokens_sum,
        "tool_call_count": tool_calls,
        "tool_call_duration_ms_sum": tool_duration_ms_sum,
        "tool_call_duration_ms_mean": (tool_duration_ms_sum / tool_calls) if tool_calls else None,
        "tool_call_duration_ms_max": tool_duration_ms_max,
        "replay_tool_sleep_wall_s_estimated": tool_duration_ms_sum / 1000.0,
        "drop_kv_cache_events": drop_kv,
    }


def collect_trace_file_stats(trace_path: Path) -> Dict[str, Any]:
    st = trace_path.stat()
    return {
        "trace_file_name": trace_path.name,
        "trace_file_size_bytes": st.st_size,
        "trace_file_mtime_iso": datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
    }


def percentile(data: List[float], q: float) -> float:
    if not data:
        raise ValueError("empty data")
    sorted_values = sorted(data)
    idx = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[idx]


def compute_replay_run_metrics(
    *,
    trace: ExecutionTrace,
    n_sessions: int,
    wall_clock_s: float,
    session_duration_s: List[float],
    replay_output_token_sum_by_session: List[int],
    trace_completion_token_sum_by_session: List[int],
    replay_detail_session0: List[Dict[str, Any]],
    tensor_parallel_size: int,
    cuda_device_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute all replay metrics for one ladder step (or single replay run)."""
    durations = session_duration_s
    per_session_replay_output = replay_output_token_sum_by_session
    per_session_trace_completion = trace_completion_token_sum_by_session
    total_out_tokens_replay_actual = float(sum(per_session_replay_output))
    tokens_per_trace_trace_metadata = count_assistant_completion_tokens(trace.events)
    total_out_tokens_trace_metadata = float(n_sessions * tokens_per_trace_trace_metadata)

    per_session_tp = [
        per_session_replay_output[i] / durations[i]
        for i in range(len(durations))
        if durations[i] > 0
    ]
    result: Dict[str, Any] = {
        "wall_clock_s": wall_clock_s,
        "assistant_output_tokens_per_trace": tokens_per_trace_trace_metadata,
        "total_output_tokens_trace_metadata": total_out_tokens_trace_metadata,
        "total_output_tokens_estimated": total_out_tokens_trace_metadata,
        "per_session_replay_output_token_sum": per_session_replay_output,
        "per_session_total_output_tokens": list(per_session_replay_output),
        "per_session_trace_completion_token_sum": per_session_trace_completion,
        "assistant_output_tokens_per_trace_replay_actual_mean": (
            statistics.mean(per_session_replay_output) if per_session_replay_output else None
        ),
        "total_output_tokens_replay_actual": total_out_tokens_replay_actual,
        "replay_assistant_generations_detail_session0": replay_detail_session0,
        "session_duration_s": durations,
        "session_duration_min_s": min(durations) if durations else None,
        "session_duration_max_s": max(durations) if durations else None,
        "session_duration_sum_s": sum(durations) if durations else None,
        "session_duration_stdev_s": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "session_duration_p50_s": statistics.median(durations) if durations else None,
        "session_duration_p90_s": percentile(durations, 0.9) if durations else None,
        "session_duration_p99_s": percentile(durations, 0.99) if durations else None,
        "session_duration_mean_s": statistics.mean(durations) if durations else None,
        "session_duration_cv": (statistics.stdev(durations) / statistics.mean(durations))
        if len(durations) > 1 and statistics.mean(durations) > 0
        else None,
        "aggregate_latency_person_s": sum(durations) if durations else None,
        "median_tps_per_user": statistics.median(per_session_tp) if per_session_tp else None,
        "mean_tps_per_user": statistics.mean(per_session_tp) if per_session_tp else None,
        "min_tps_per_user": min(per_session_tp) if per_session_tp else None,
        "max_tps_per_user": max(per_session_tp) if per_session_tp else None,
        "output_tps_aggregate": total_out_tokens_replay_actual / wall_clock_s
        if wall_clock_s > 0
        else None,
        "output_tokens_per_wall_s_per_session_mean": (
            total_out_tokens_replay_actual / wall_clock_s / n_sessions
        )
        if wall_clock_s > 0 and n_sessions > 0
        else None,
        "mean_tps_per_user_session_time": (total_out_tokens_replay_actual / sum(durations))
        if durations and sum(durations) > 0
        else None,
    }
    result["output_tps_per_gpu"] = (
        result["output_tps_aggregate"] / tensor_parallel_size
        if result["output_tps_aggregate"] is not None and tensor_parallel_size > 0
        else None
    )
    result["pareto_x_median_tps_per_user"] = result.get("median_tps_per_user")
    result["pareto_y_output_tps_per_gpu"] = result.get("output_tps_per_gpu")
    result["output_tps_per_aggregate_1gpu_equiv"] = result.get("output_tps_aggregate")
    if cuda_device_count is not None and cuda_device_count > 0:
        result["output_tps_per_cuda_device_count"] = (
            result["output_tps_aggregate"] / cuda_device_count
            if result["output_tps_aggregate"] is not None
            else None
        )
    return result


# ---------------------------------------------------------------------------
# Steady-state window and job-level Pareto coordinates
# ---------------------------------------------------------------------------
#
# A replayed agent session lasts minutes, so a concurrent run has a fill-up
# ramp, a saturated middle, and a drain. Both job- and token-level metrics are
# measured only over the saturated window, defined in admission order as::
#
#     window start = start time of the ``excl``-th admitted session
#                    (the admission that fills the last slot)
#     window end   = end time of the ``(N - excl + 1)``-th admitted session
#                    (the first drain session's completion)
#
# with ``excl = max(1, min(concurrency, max_batch_size * dp_size))``. Job
# membership is completion-based: a session counts if it *completes* inside the
# window, whenever it was admitted. Requiring a session to be entirely inside
# drops the in-flight population at both edges while keeping the full window in
# the denominator, which undercounts jobs/h/GPU roughly twofold. LLM calls last
# seconds rather than minutes, so the token-level metrics can use the stricter
# fully-inside rule without measurable loss.


def steady_state_excl_count(concurrency: int, max_batch_size: int, dp_size: int = 1) -> int:
    """Sessions excluded from each end of the window (the saturation depth)."""
    return max(1, min(int(concurrency), int(max_batch_size) * max(1, int(dp_size))))


def compute_steady_state_window(
    *,
    session_start_offset_s: List[float],
    session_end_offset_s: List[float],
    excl_count: int,
) -> Dict[str, Any]:
    """Return the saturation window over which the Pareto metrics are measured."""
    n_sessions = len(session_start_offset_s)
    if n_sessions <= 2 * excl_count:
        return {
            "valid": False,
            "reason": f"total_sessions ({n_sessions}) <= 2 * excl_count ({excl_count})",
            "excl_count": excl_count,
        }
    admission_order = sorted(range(n_sessions), key=lambda i: (session_start_offset_s[i], i))
    start_s = session_start_offset_s[admission_order[excl_count - 1]]
    end_s = session_end_offset_s[admission_order[n_sessions - excl_count]]
    if end_s <= start_s:
        return {"valid": False, "reason": "empty window", "excl_count": excl_count}
    return {
        "valid": True,
        "start_offset_s": start_s,
        "end_offset_s": end_s,
        "duration_s": end_s - start_s,
        "excl_count": excl_count,
    }


def _tpot_ms(entry: Dict[str, Any]) -> Optional[float]:
    """Per-output-token latency of one LLM call, excluding time to first token."""
    output_tokens = entry.get("replay_output_token_len") or 0
    latency_s = entry.get("latency_s")
    ttft_s = entry.get("ttft_s")
    if output_tokens > 1 and latency_s is not None and ttft_s is not None:
        return (float(latency_s) - float(ttft_s)) * 1000.0 / (output_tokens - 1)
    return None


def compute_steady_state_pareto(
    *,
    window: Dict[str, Any],
    session_end_offset_s: List[float],
    session_duration_mean_s: Optional[float],
    replay_detail: List[Dict[str, Any]],
    tensor_parallel_size: int,
) -> Dict[str, Any]:
    """Return the job- and token-level Pareto coordinates for one run.

    ``job_y`` and ``token_y`` are per-GPU rates, normalized by
    ``tensor_parallel_size``, which is the GPU count for the single-node
    tensor/expert-parallel configurations this driver targets. ``token_y``
    counts prefill input plus decode output, since a multi-turn agent workload
    is prefill-dominated and an output-only axis understates it several-fold;
    ``token_x`` stays decode-only because TPOT is a per-output-token latency.
    """
    if not window.get("valid"):
        return {"valid": False, "reason": window.get("reason")}
    num_gpus = int(tensor_parallel_size)
    if num_gpus <= 0:
        return {"valid": False, "reason": "tensor_parallel_size <= 0"}

    start_s = window["start_offset_s"]
    end_s = window["end_offset_s"]
    duration_s = window["duration_s"]

    jobs_completed = sum(1 for e in session_end_offset_s if start_s <= e <= end_s)
    in_window = [
        e
        for e in replay_detail
        if e.get("client_request_start_offset_s") is not None
        and e.get("client_request_end_offset_s") is not None
        and e["client_request_start_offset_s"] >= start_s
        and e["client_request_end_offset_s"] <= end_s
    ]
    output_tokens = sum(int(e.get("usage_completion_tokens") or 0) for e in in_window)
    input_tokens = sum(int(e.get("usage_prompt_tokens") or 0) for e in in_window)
    tpots_ms = [t for t in (_tpot_ms(e) for e in in_window) if t is not None]

    return {
        "valid": True,
        "job_x_jobs_per_h_per_user": (3600.0 / session_duration_mean_s)
        if session_duration_mean_s
        else None,
        "job_y_jobs_per_h_per_gpu": 3600.0 * jobs_completed / (duration_s * num_gpus),
        "token_x_tps_per_user": (1000.0 / statistics.median(tpots_ms)) if tpots_ms else None,
        "token_y_tps_per_gpu": (input_tokens + output_tokens) / duration_s / num_gpus,
        "token_y_tps_per_gpu_output_only": output_tokens / duration_s / num_gpus,
        "jobs_completed_in_window": jobs_completed,
        "llm_calls_in_window": len(in_window),
        "window_duration_s": duration_s,
        "num_gpus": num_gpus,
    }
