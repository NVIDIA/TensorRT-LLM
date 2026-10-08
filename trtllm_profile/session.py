# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sherlock-style operation timings, per-source-file host timings, and telemetry."""

from __future__ import annotations

import atexit
import cProfile
import io
import json
import os
import pstats
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Iterator

from .resources import ResourceSampler

_COMPONENT: ContextVar[str | None] = ContextVar("trtllm_profile_component", default=None)
_TRACE_DEPTH: ContextVar[tuple[int, int, int]] = ContextVar(
    "trtllm_profile_depth", default=(0, 0, 0)
)
_active: Profiler | None = None


@dataclass
class Operation:
    component: str
    source: str
    line: int
    start_ns: int
    host_us: float
    thread_id: int
    span: object | None = None
    failed: bool = False


def active_session() -> Profiler | None:
    """Return the active profiler in this process, never a profiler inherited by fork."""
    return _active if _active is not None and _active.pid == os.getpid() else None


@contextmanager
def component(name: str) -> Iterator[None]:
    """Attribute operations to a component without adding a second layer of timing."""
    if active_session() is None:
        yield
        return
    token = _COMPONENT.set(name)
    try:
        yield
    finally:
        _COMPONENT.reset(token)


@contextmanager
def trace_active() -> Iterator[None]:
    """Enroll a worker thread in the active session; no-op when profiling is disabled."""
    profiler = active_session()
    if profiler is None:
        yield
    else:
        with profiler.trace():
            yield


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        temporary = Path(file.name)
        try:
            file.write(content)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Profiler:
    """An opt-in, process-local profiler for HIP, CUDA, or CPU programs.

    Device times are current-stream event spans around dispatched operations,
    not individual ISA instruction or kernel execution counters. Event gaps are
    reported as idle time per stream. No synchronization is done per operation.
    Files contain exclusive and inclusive host wall times from cProfile; inclusive
    times overlap. Each thread explicitly enrolled with ``trace()`` is included.
    """

    def __init__(
        self,
        output: str | Path | None = None,
        interval: float = 0.1,
        max_ops: int = 100000,
        tensors: bool = True,
        calibration_ops: int = 256,
    ) -> None:
        if max_ops < 1 or calibration_ops < 1:
            raise ValueError("max_ops and calibration_ops must be positive")
        self.pid = os.getpid()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        name = Path(sys.argv[0] or "python").stem
        base = Path(output) if output else Path("profiles") / f"{name}-{self.pid}-{stamp}"
        if base.suffix in (".json", ".txt", ".prof"):
            base = base.with_suffix("")
        self.output = base
        self.max_ops = max_ops
        self.tensors = tensors
        self.calibration_ops = calibration_ops
        self.status = "running"
        self.warnings: list[str] = []
        self.report_data: dict | None = None
        self._sampler = ResourceSampler(interval=interval)
        self._stack = ExitStack()
        self._profiles: dict[int, cProfile.Profile] = {}
        self._records: list[Operation] = []
        self._lock = threading.Lock()
        self._reserved = 0
        self._dropped = 0
        self._recorder = None
        self._trace_module = None
        self._calibration: list[dict] = []
        self._started = False
        self._finished = False
        self._origin_ns = 0
        self._duration_s = 0.0
        self._command = list(sys.argv)

    def reserve(self) -> bool:
        with self._lock:
            if self._finished:
                return False
            if self._reserved >= self.max_ops:
                self._dropped += 1
                return False
            self._reserved += 1
            return True

    def begin(self, device: int | None = None) -> tuple[int, object | None]:
        span = self._recorder.begin(device) if self._recorder is not None else None
        return time.perf_counter_ns(), span

    def end(
        self,
        started: tuple[int, object | None],
        name: str,
        source: tuple[str, int],
        failed: bool = False,
        store: bool = True,
    ) -> Operation:
        ended_ns = time.perf_counter_ns()
        start_ns, span = started
        device_error = None
        if span is not None:
            try:
                self._recorder.end(span)
            except RuntimeError as error:
                self.warnings.append(f"Device event recording failed: {error}")
                device_error = error
                span = None
        record = Operation(
            name,
            source[0],
            source[1],
            start_ns,
            (ended_ns - start_ns) / 1000,
            threading.get_ident(),
            span,
            failed or device_error is not None,
        )
        if store:
            with self._lock:
                self._records.append(record)
        if device_error is not None and not failed:
            raise device_error
        return record

    @contextmanager
    def operation(self, name: str, device: int | None = None) -> Iterator[None]:
        """Time an external operation not otherwise visible to PyTorch dispatch.

        Do not wrap already-dispatched tensor operations with this method, since
        that would double count. Use ``component()`` to label those instead.
        """
        if not self._started or self._finished or not self.reserve():
            yield
            return
        frame = sys._getframe(2)
        started = self.begin(device)
        success = False
        try:
            yield
            success = True
        finally:
            self.end(started, name, (frame.f_code.co_filename, frame.f_lineno), failed=not success)

    def start(self) -> Profiler:
        global _active
        if self._started:
            raise RuntimeError("Profiler sessions cannot be restarted")
        if active_session() is not None:
            raise RuntimeError("A profiler is already active")
        self._started = True
        if self.tensors:
            try:
                from . import torch_trace
            except ImportError as error:
                self.warnings.append(f"Tensor tracing unavailable: {error}")
            else:
                self._trace_module = torch_trace
                self._recorder = torch_trace.EventRecorder()
        devices = self._recorder.devices if self._recorder is not None else []
        # Calibrate CPU and each visible GPU before starting workload sampling.
        for device in [None, *devices]:
            records = []
            for _ in range(self.calibration_ops):
                started = self.begin(device)
                records.append(self.end(started, "floor", ("<calibration>", 0), store=False))
            if self._recorder is not None:
                self._recorder.synchronize()
            spans = [record.span.resolve() for record in records if record.span is not None]
            self._calibration.append(
                {
                    "device": device,
                    "ops": self.calibration_ops,
                    "host_us_per_op": sum(record.host_us for record in records) / len(records),
                    "dev_us_per_op": sum(span[0] for span in spans) / len(spans) if spans else None,
                }
            )
        if self._recorder is not None:
            self._recorder.clear()
        self._origin_ns = time.perf_counter_ns()
        self._sampler.start()
        _active = self
        self._stack.enter_context(self.trace())
        return self

    @contextmanager
    def trace(self) -> Iterator[None]:
        """Include the current thread's Python files and tensor operations."""
        owner, owner_thread, depth = _TRACE_DEPTH.get()
        thread_id = threading.get_ident()
        if (owner == self.pid and owner_thread == thread_id and depth) or self._finished:
            yield
            return
        token = _TRACE_DEPTH.set((self.pid, thread_id, 1))
        with self._lock:
            host_profile = self._profiles.setdefault(thread_id, cProfile.Profile())
        with ExitStack() as stack:
            if self._trace_module is not None:
                stack.enter_context(self._trace_module.TensorTrace(self))
            host_profile.enable()
            try:
                yield
            finally:
                host_profile.disable()
                _TRACE_DEPTH.reset(token)

    def _file_stats(self) -> tuple[dict[str, dict], pstats.Stats | None]:
        files: dict[str, dict] = {}
        stats = None
        for host_profile in self._profiles.values():
            if not host_profile.getstats():
                continue
            if stats is None:
                stats = pstats.Stats(host_profile, stream=io.StringIO())
            else:
                stats.add(host_profile)
        if stats is not None:
            for (filename, line, function), (
                _,
                calls,
                exclusive,
                inclusive,
                _,
            ) in stats.stats.items():
                if str(Path(__file__).parent) in filename:
                    continue
                file = files.setdefault(
                    filename,
                    {
                        "source": filename,
                        "calls": 0,
                        "host_self_us": 0.0,
                        "host_inclusive_us": 0.0,
                        "functions": [],
                        "ops": 0,
                        "op_host_us": 0.0,
                        "dev_us": None,
                        "idle_us": None,
                    },
                )
                file["calls"] += calls
                file["host_self_us"] += exclusive * 1e6
                file["host_inclusive_us"] += inclusive * 1e6
                file["functions"].append(
                    {
                        "name": function,
                        "line": line,
                        "calls": calls,
                        "host_self_us": exclusive * 1e6,
                        "host_inclusive_us": inclusive * 1e6,
                    }
                )
        return files, stats

    def _build_report(self) -> tuple[dict, pstats.Stats | None]:
        files, stats = self._file_stats()
        components: dict[str, dict] = defaultdict(
            lambda: {
                "ops": 0,
                "device_ops": 0,
                "dev_us": None,
                "idle_us": None,
                "host_us": 0.0,
            }
        )
        trace = []
        unresolved_spans = 0
        for record in self._records:
            row = components[record.component]
            row["ops"] += 1
            row["host_us"] += record.host_us
            file = files.setdefault(
                record.source,
                {
                    "source": record.source,
                    "calls": 0,
                    "host_self_us": 0.0,
                    "host_inclusive_us": 0.0,
                    "functions": [],
                    "ops": 0,
                    "op_host_us": 0.0,
                    "dev_us": None,
                    "idle_us": None,
                },
            )
            file["ops"] += 1
            file["op_host_us"] += record.host_us
            host_start_us = (record.start_ns - self._origin_ns) / 1000
            trace.append(
                {
                    "name": record.component,
                    "cat": "host submission",
                    "ph": "X",
                    "ts": host_start_us,
                    "dur": record.host_us,
                    "pid": self.pid,
                    "tid": record.thread_id,
                    "args": {"source": record.source, "line": record.line, "failed": record.failed},
                }
            )
            if record.span is not None:
                try:
                    device_us, idle_us, timestamp_us, device, stream = record.span.resolve()
                except RuntimeError:
                    unresolved_spans += 1
                    continue
                row["device_ops"] += 1
                for target in (row, file):
                    target["dev_us"] = (target["dev_us"] or 0.0) + device_us
                    target["idle_us"] = (target["idle_us"] or 0.0) + idle_us
                trace.append(
                    {
                        "name": record.component,
                        "cat": "device event span",
                        "ph": "X",
                        "ts": timestamp_us - self._origin_ns / 1000,
                        "dur": device_us,
                        "pid": self.pid + 1000000 + device,
                        "tid": stream,
                        "args": {"source": record.source, "line": record.line},
                    }
                )
        if unresolved_spans:
            self.warnings.append(f"Device event timings unavailable for {unresolved_spans} spans")
        total_dev_us = sum(row["dev_us"] or 0 for row in components.values())
        rows = [
            {
                "component": name,
                **row,
                "percent_dev": 100 * row["dev_us"] / total_dev_us
                if row["dev_us"] is not None and total_dev_us > 0
                else None,
            }
            for name, row in components.items()
        ]
        rows.sort(key=lambda row: row["dev_us"] or row["host_us"], reverse=True)
        return {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pid": self.pid,
            "command": self._command,
            "status": self.status,
            "wall_s": self._duration_s,
            "devices": self._recorder.metadata() if self._recorder is not None else [],
            "components": rows,
            "files": sorted(files.values(), key=lambda file: file["host_self_us"], reverse=True),
            "calibration": self._calibration,
            "resources": self._sampler.report(),
            "recorded_ops": len(self._records),
            "dropped_ops": self._dropped,
            "warnings": self.warnings,
            "timing_semantics": {
                "ops": "PyTorch dispatch operations, not kernel counts",
                "dev_us": "Uncorrected current-stream event span; includes instrumentation floor",
                "idle_us": "Event gap from previous end to next begin on the same stream",
                "host_us": "Host call/submission wall time, not device completion time",
                "trace_clock": "Device clocks aligned approximately to each stream's first host marker",
                "file_host_self_us": "Exclusive cProfile host wall time; includes blocking calls",
                "file_host_inclusive_us": "Inclusive host wall time; overlapping, do not sum",
                "telemetry": "Sampled system-wide counters; process RSS and CPU are separately labeled",
                "coverage": "Current thread and explicitly enrolled workers; current streams only",
            },
            "traceEvents": trace,
        }, stats

    @staticmethod
    def format_report(report: dict) -> str:
        def number(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.2f}"

        lines = [
            "\nSherlock performance report (event spans, not kernel counts)",
            f"wall: {report['wall_s']:.6f} s; recorded ops: {report['recorded_ops']}; "
            f"dropped ops: {report['dropped_ops']}",
            f"{'component':<22} {'ops':>8} {'%dev':>7} {'dev us':>13} {'idle us':>13} {'host us':>13}",
        ]
        for row in report["components"]:
            percent = "N/A" if row["percent_dev"] is None else f"{row['percent_dev']:.1f}%"
            lines.append(
                f"{row['component']:<22} {row['ops']:>8} {percent:>7} "
                f"{number(row['dev_us']):>13} {number(row['idle_us']):>13} {row['host_us']:>13.2f}"
            )
        if not report["components"]:
            lines.append(
                "No tensor/external operations captured; per-file host timings are still recorded."
            )
        for floor in report["calibration"]:
            device = "CPU" if floor["device"] is None else f"device {floor['device']}"
            lines.append(
                f"instrumentation floor ({device}), {floor['ops']} empty ops timed through the same "
                f"begin/end path: {floor['host_us_per_op']:.2f} us host, "
                f"{number(floor['dev_us_per_op'])} us device each."
            )
        lines.append("\nResource utilization (sampled mean / peak; N/A means unavailable)")
        summary = report["resources"]["summary"]
        for key, label in (
            ("cpu_system_percent", "CPU system %"),
            ("cpu_process_percent", "CPU process % (100 = one core)"),
            ("ram_percent", "RAM system %"),
            ("ram_used_bytes", "RAM system GiB"),
            ("process_rss_bytes", "RAM process RSS GiB"),
            ("process_ram_percent", "RAM process % of system total"),
        ):
            metric = summary.get(key)
            divisor = 2**30 if key.endswith("bytes") else 1
            lines.append(
                f"{label:<36} "
                + (
                    f"{metric['mean'] / divisor:.2f} / {metric['peak'] / divisor:.2f}"
                    if metric
                    else "N/A"
                )
            )
        gpu_keys = [
            key
            for key in summary
            if "/" in key
            and key.rsplit("/", 1)[-1]
            in (
                "gpu_percent",
                "vram_percent",
                "vram_used_bytes",
                "vram_total_bytes",
            )
        ]
        for key in gpu_keys:
            metric = summary[key]
            divisor = 2**30 if key.endswith("bytes") else 1
            lines.append(
                f"{key} ({'GiB' if key.endswith('bytes') else '%'}, mean / peak): "
                f"{metric['mean'] / divisor:.2f} / {metric['peak'] / divisor:.2f}"
            )
        if not gpu_keys:
            lines.append(
                "GPU utilization % / VRAM used, total, %: N/A (no readable GPU telemetry counters)"
            )
        physical_devices = {}
        for sample in report["resources"]["samples"]:
            for gpu in sample["gpus"]:
                physical_devices[gpu["id"]] = gpu
        for identity in physical_devices:
            for counter in ("gpu_percent", "vram_used_bytes", "vram_total_bytes", "vram_percent"):
                if f"{identity}/{counter}" not in summary:
                    lines.append(f"{identity}/{counter}: N/A (counter unavailable)")
        for device in report["devices"]:
            peak = device.get("process_lifetime_peak_allocated_bytes")
            total = device.get("total_bytes")
            allocated = device.get("process_allocated_bytes")
            lines.append(
                f"{device['backend']} logical device {device['logical_index']} "
                f"process allocated GiB at report / lifetime peak: "
                f"{number(allocated / 2**30 if allocated is not None else None)} / "
                f"{number(peak / 2**30 if peak is not None else None)}; "
                f"total VRAM GiB: {number(total / 2**30 if total is not None else None)}"
            )
        lines.append("\nPer-file host timings (top 20; all files/functions are in JSON and .prof)")
        for file in report["files"][:20]:
            lines.append(
                f"{file['host_self_us']:>13.2f} us self  {file['calls']:>8} calls  {file['source']}"
            )
        lines.extend(f"WARNING: {warning}" for warning in report["warnings"])
        lines.extend(f"WARNING: {warning}" for warning in report["resources"]["warnings"])
        if report["dropped_ops"]:
            lines.append(
                "WARNING: Operation trace is truncated; increase --profile-max-ops for complete timing."
            )
        lines.append("Device spans are not floor-subtracted; file inclusive times overlap.\n")
        return "\n".join(lines)

    def finish(self, print_report: bool = True) -> dict | None:
        """Write JSON, text, Chrome trace, and cProfile reports exactly once."""
        global _active
        if self.pid != os.getpid() or not self._started:
            return None
        if self._finished:
            return self.report_data
        self._duration_s = (time.perf_counter_ns() - self._origin_ns) / 1e9
        self._stack.close()
        self._finished = True
        if _active is self:
            _active = None
        self._sampler.stop()
        device_error = None
        if self._recorder is not None:
            try:
                self._recorder.synchronize()
            except RuntimeError as error:
                device_error = error
                if not self.status.startswith("failed"):
                    self.status = "failed: device synchronization"
                self.warnings.append(f"Device timings unavailable after a device error: {error}")
                for record in self._records:
                    record.span = None
        if self.status == "running":
            self.status = "completed"
        report, stats = self._build_report()
        trace = report.pop("traceEvents")
        self.report_data = report
        text = self.format_report(report)
        _atomic_write(
            Path(f"{self.output}.json"), json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        _atomic_write(Path(f"{self.output}.txt"), text)
        _atomic_write(
            Path(f"{self.output}.trace.json"),
            json.dumps({"traceEvents": trace, "displayTimeUnit": "us"}),
        )
        if stats is not None:
            stats.dump_stats(f"{self.output}.prof")
        if print_report:
            print(text)
            print(f"Profile files: {self.output}.{{json,txt,trace.json,prof}}", flush=True)
        if device_error is not None and sys.exc_info()[0] is None:
            # Retain reports, but never turn a deferred GPU execution failure into
            # application success or mask an exception already being unwound.
            raise device_error
        return report

    def __enter__(self) -> Profiler:
        return self.start()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception_type is not None:
            self.status = f"failed: {exception_type.__name__}"
        self.finish()


def enable_from_argv(argv: list[str] | None = None) -> Profiler | None:
    """Consume only profiler options, before application CLI parsing.

    A literal ``--`` protects subsequent arguments. Importing this module does
    nothing; flags are consumed only when this function is explicitly called.
    """
    arguments = sys.argv if argv is None else argv
    limit = arguments.index("--") if "--" in arguments else len(arguments)
    if "--profile" not in arguments[1:limit]:
        return active_session()
    options: dict[str, str] = {}
    remaining = [arguments[0]]
    tensors = True
    index = 1
    keys = {"--profile-output", "--profile-interval", "--profile-max-ops"}
    while index < limit:
        argument = arguments[index]
        key, equals, value = argument.partition("=")
        if argument == "--profile":
            index += 1
            continue
        if argument == "--profile-no-tensors":
            tensors = False
            index += 1
            continue
        if key in keys:
            if not equals:
                index += 1
                if index >= limit:
                    raise SystemExit(f"{key} needs a value")
                value = arguments[index]
            options[key] = value
        else:
            remaining.append(argument)
        index += 1
    remaining.extend(arguments[limit:])
    arguments[:] = remaining
    if active_session() is not None:
        return active_session()
    try:
        profiler = Profiler(
            output=options.get("--profile-output"),
            interval=float(options.get("--profile-interval", "0.1")),
            max_ops=int(options.get("--profile-max-ops", "100000")),
            tensors=tensors,
        )
    except ValueError as error:
        raise SystemExit(f"Invalid profiler option: {error}") from error
    profiler.start()
    # The bootstrap cannot know the application's exit code; launchers can.
    profiler.status = "process-exit (exit status unknown)"
    atexit.register(profiler.finish)
    return profiler
