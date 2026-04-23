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

"""Shared helpers for the Pareto trace-replay client and aggregator.

Kept backend-agnostic: no TensorRT-LLM runtime imports here, so both the
client (which talks to an external ``trtllm-serve`` via HTTP) and the
offline aggregator can depend on this module without pulling in CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tensorrt_llm.scaffolding.execution_trace import TraceEvent


# ---------------------------------------------------------------------------
# Trace discovery and summarization
# ---------------------------------------------------------------------------


def find_compact_trace_file(trace_dir: Path) -> Path:
    """Return the compact ``*.trace.json`` under ``trace_dir``.

    Prefers ``*.trace.json`` over ``*.full.trace.json`` when both exist.
    """
    all_traces = sorted(trace_dir.glob("*.trace.json"))
    if not all_traces:
        raise FileNotFoundError(f"No *.trace.json under {trace_dir}")
    for p in all_traces:
        if ".full.trace" not in p.name:
            return p
    return all_traces[0]


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
            d = ev.duration_ms or 0.0
            tool_duration_ms_sum += d
            tool_duration_ms_max = max(tool_duration_ms_max, d)
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


# ---------------------------------------------------------------------------
# Host info / CLI snapshots
# ---------------------------------------------------------------------------


def collect_host_info() -> Dict[str, Any]:
    """Runtime, CUDA (if available) snapshot for reproducibility."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }
    env_keys = (
        "CUDA_VISIBLE_DEVICES",
        "LLM_MODELS_ROOT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "OMP_NUM_THREADS",
        "SCAFFOLDING_DETERMINISTIC",
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
    )
    info["env_subset"] = {k: os.environ.get(k) for k in env_keys if k in os.environ}

    # torch is optional on pure-client hosts but usually present; probe defensively.
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        info["torch_error"] = repr(exc)

    return info


def args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """CLI snapshot (Path -> str) suitable for JSON serialization."""
    d = vars(args).copy()
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def percentile(data: List[float], q: float) -> float:
    if not data:
        raise ValueError("empty data")
    s = sorted(data)
    idx = int(round(q * (len(s) - 1)))
    return s[idx]


# ---------------------------------------------------------------------------
# OOM classification (shared between client and aggregator)
# ---------------------------------------------------------------------------


def is_oom_exception(exc: BaseException) -> bool:
    """Heuristic: CUDA OOM, host MemoryError, or common runtime OOM strings."""
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch

        oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_cls is not None and isinstance(exc, oom_cls):
            return True
        oom_cls2 = getattr(torch, "OutOfMemoryError", None)
        if oom_cls2 is not None and isinstance(exc, oom_cls2):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    if "out of memory" in msg or "cuda out of memory" in msg:
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in msg:
        return True
    return False


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically (temp file + replace) and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Artifact naming (model_basename + parallelism suffix)
# ---------------------------------------------------------------------------


def pareto_config_filename_suffix(
    model_name_or_dir: Optional[str],
    tensor_parallel_size: int,
    moe_expert_parallel_size: int,
    enable_attention_dp: bool = False,
) -> str:
    """Return ``<model_basename>_tp<TP>_ep<EP>[_adp]`` for artifact names."""
    raw = (model_name_or_dir or "").strip()
    if raw:
        name = Path(os.path.expanduser(raw)).resolve().name
    else:
        name = "model"
    if not name or name == ".":
        name = "model"
    safe = "".join((ch if ch.isalnum() or ch in "._-" else "_") for ch in name)
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_") or "model"
    tp = int(tensor_parallel_size or 1)
    ep = int(moe_expert_parallel_size or 0)
    adp = "_adp" if enable_attention_dp else ""
    return f"{safe}_tp{tp}_ep{ep}{adp}"


# ---------------------------------------------------------------------------
# stdout/stderr tee
# ---------------------------------------------------------------------------


class TeeTextIO:
    """Duplicate writes to a primary stream (console) and a log file."""

    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        n = self._primary.write(s)
        self._secondary.write(s)
        return n

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)
