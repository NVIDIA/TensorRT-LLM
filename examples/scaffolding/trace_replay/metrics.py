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

from __future__ import annotations

import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tensorrt_llm.scaffolding.execution_trace import ExecutionTrace, TraceEvent

Point3 = Tuple[float, float, Union[int, str]]


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


def _step_from_run(r: Dict[str, Any]) -> Union[int, str]:
    step = r.get("ladder_step")
    if step is None:
        step = r.get("max_batch_size")
    return step if step is not None else "?"


def collect_token_pareto_points(runs: List[Dict[str, Any]]) -> List[Point3]:
    """Return ``(median_tps_per_user, output_tps_per_gpu, ladder_step)``."""
    pts: List[Point3] = []
    for r in runs:
        if r.get("status") != "success":
            continue
        x = r.get("median_tps_per_user")
        if x is None:
            x = r.get("pareto_x_median_tps_per_user")
        y = r.get("output_tps_per_gpu")
        if y is None:
            y = r.get("pareto_y_output_tps_per_gpu")
        if x is None or y is None:
            continue
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            continue
        pts.append((xf, yf, _step_from_run(r)))
    pts.sort(key=lambda t: t[0])
    return pts


def _gpu_count_for_run(r: Dict[str, Any], data: Dict[str, Any]) -> Optional[int]:
    """GPU count aligned with ``output_tps_per_gpu`` (aggregate / TP size)."""
    cfg = r.get("llm_effective_config") or {}
    tp = cfg.get("tensor_parallel_size")
    if tp is not None:
        try:
            n = int(tp)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    cli = data.get("cli_args") or {}
    tp = cli.get("tensor_parallel_size")
    if tp is not None:
        try:
            n = int(tp)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    host = data.get("host") or {}
    n = host.get("cuda_device_count")
    if n is not None:
        try:
            c = int(n)
            if c > 0:
                return c
        except (TypeError, ValueError):
            pass
    return None


def _mean_trace_duration_s(r: Dict[str, Any]) -> Optional[float]:
    """Mean wall time per trace (seconds), from ``session_duration_mean_s`` or sum/n."""
    mean_s = r.get("session_duration_mean_s")
    if mean_s is not None:
        try:
            mean_value = float(mean_s)
            if mean_value >= 0:
                return mean_value
        except (TypeError, ValueError):
            pass
    total_s = r.get("session_duration_sum_s")
    if total_s is None:
        total_s = r.get("aggregate_latency_person_s")
    n = r.get("n_sessions")
    if total_s is None or n is None:
        return None
    try:
        total_value = float(total_s)
        sessions = float(n)
    except (TypeError, ValueError):
        return None
    if sessions <= 0:
        return None
    return total_value / sessions


def collect_agent_pareto_points(
    runs: List[Dict[str, Any]],
    data: Dict[str, Any],
) -> List[Point3]:
    """Return ``(task/user/h, task/gpu/h, ladder_step)``."""
    pts: List[Point3] = []
    for r in runs:
        if r.get("status") != "success":
            continue
        mean_s = _mean_trace_duration_s(r)
        if mean_s is None or mean_s <= 0:
            continue
        x = 3600.0 / mean_s
        n_raw = r.get("n_sessions")
        wall = r.get("wall_clock_s")
        g = _gpu_count_for_run(r, data)
        if n_raw is None or wall is None or g is None or g <= 0:
            continue
        try:
            n_sessions = float(n_raw)
            wall_s = float(wall)
        except (TypeError, ValueError):
            continue
        if wall_s <= 0:
            continue
        y = n_sessions * 3600.0 / (float(g) * wall_s)
        pts.append((x, y, _step_from_run(r)))
    pts.sort(key=lambda t: t[0])
    return pts
