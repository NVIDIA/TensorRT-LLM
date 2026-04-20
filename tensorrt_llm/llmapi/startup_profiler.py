#
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

"""Startup profiling utilities for TRT-LLM server bring-up."""

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from tensorrt_llm.logger import logger

ENABLE_ENV_VAR = "TRTLLM_PROFILE_STARTUP"
OUTPUT_PATH_ENV_VAR = "TRTLLM_STARTUP_PROFILE_OUTPUT"
SCHEMA_VERSION = 1


@dataclass
class _StartupRecord:
    name: str
    start_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list["_StartupRecord"] = field(default_factory=list)
    end_time_s: Optional[float] = None

    def finish(self) -> None:
        self.end_time_s = time.perf_counter()

    def to_dict(self, root_start_time_s: float) -> dict[str, Any]:
        end_time_s = self.end_time_s if self.end_time_s is not None else time.perf_counter()
        return {
            "name": self.name,
            "start_offset_s": self.start_time_s - root_start_time_s,
            "duration_s": end_time_s - self.start_time_s,
            "metadata": _make_json_safe(self.metadata),
            "children": [child.to_dict(root_start_time_s) for child in self.children],
        }


class StartupProfiler:
    """Hierarchical startup profiler controlled by environment variables."""

    def __init__(self) -> None:
        self.enabled = os.environ.get(ENABLE_ENV_VAR, "0") == "1"
        self._start_time_s = time.perf_counter()
        self._completed_at_s: Optional[float] = None
        self._completion_metadata: dict[str, Any] = {}
        self._root_records: list[_StartupRecord] = []
        self._attached_profiles: dict[str, Any] = {}
        self._thread_state = threading.local()
        self._write_lock = threading.Lock()
        self._wrote_output = False

    def _get_stack(self) -> list[_StartupRecord]:
        stack = getattr(self._thread_state, "stack", None)
        if stack is None:
            stack = []
            self._thread_state.stack = stack
        return stack

    @contextmanager
    def timer(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        stack = self._get_stack()
        record = _StartupRecord(name=name, start_time_s=time.perf_counter(), metadata=metadata)
        if stack:
            stack[-1].children.append(record)
        else:
            self._root_records.append(record)

        stack.append(record)
        try:
            yield
        finally:
            record.finish()
            stack.pop()

    def attach_profile(self, name: str, profile: Any) -> None:
        if not self.enabled or profile is None:
            return
        self._attached_profiles[name] = profile

    def complete(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        self._completed_at_s = time.perf_counter()
        self._completion_metadata.update(metadata)

    def total_duration_s(self) -> float:
        end_time_s = (
            self._completed_at_s if self._completed_at_s is not None else time.perf_counter()
        )
        return end_time_s - self._start_time_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "enabled": self.enabled,
            "completed": self._completed_at_s is not None,
            "total_duration_s": self.total_duration_s(),
            "records": [record.to_dict(self._start_time_s) for record in self._root_records],
            "metadata": _make_json_safe(self._completion_metadata),
            "attached_profiles": _make_json_safe(self._attached_profiles),
            "pid": os.getpid(),
        }

    def summary(self) -> str:
        if not self.enabled:
            return "Startup profiling is disabled."

        lines = [
            "TRT-LLM STARTUP TIMING BREAKDOWN",
            f"Total startup time: {self.total_duration_s():.3f}s",
        ]

        def add_record(record: _StartupRecord, depth: int) -> None:
            indent = "  " * depth
            duration_s = (
                record.end_time_s if record.end_time_s is not None else time.perf_counter()
            ) - record.start_time_s
            lines.append(f"{indent}{record.name}: {duration_s:.3f}s")
            for child in record.children:
                add_record(child, depth + 1)

        for record in self._root_records:
            add_record(record, 0)

        if self._attached_profiles:
            lines.append("Attached profiles:")
            for name in sorted(self._attached_profiles):
                lines.append(f"  {name}")

        return "\n".join(lines)

    def write_if_requested(self, profile: Optional[dict[str, Any]] = None) -> Optional[Path]:
        output_path = os.environ.get(OUTPUT_PATH_ENV_VAR)
        if not self.enabled or not output_path:
            return None

        with self._write_lock:
            if self._wrote_output:
                return Path(output_path)

            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as f:
                json.dump(
                    self.to_dict() if profile is None else profile, f, indent=2, sort_keys=True
                )
            self._wrote_output = True

        logger.info("Startup profile written to %s", target)
        return target


_startup_profiler: Optional[StartupProfiler] = None
_startup_profiler_lock = threading.Lock()


def get_startup_profiler() -> StartupProfiler:
    global _startup_profiler

    if _startup_profiler is None:
        with _startup_profiler_lock:
            if _startup_profiler is None:
                _startup_profiler = StartupProfiler()

    return _startup_profiler


@contextmanager
def startup_timer(name: str, **metadata: Any) -> Iterator[None]:
    with get_startup_profiler().timer(name, **metadata):
        yield


def reset_startup_profiler_for_test() -> None:
    global _startup_profiler
    with _startup_profiler_lock:
        _startup_profiler = None


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
