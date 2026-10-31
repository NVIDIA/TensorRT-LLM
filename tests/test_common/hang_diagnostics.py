# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect hang reports outside the lifetime and capture buffers of inner pytest."""

import os
import shutil
import sys
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

import pytest

# A report holds every thread's stack for one process, normally tens of KiB.
# Several ranks land in one retained stdout.log, so cap a pathological report
# rather than letting it crowd out the rest of the log.
_MAX_REPORT_BYTES = 256 * 1024


def hang_diagnostics_root(output_dir: str) -> str:
    """Reserve a unique invocation path without creating a directory."""
    return os.path.join(os.path.abspath(output_dir), f"hang-diagnostics-{uuid.uuid4().hex}")


def prepare_hang_diagnostics(directory: str) -> str | None:
    """Create a report directory only when a test opts in."""
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        return os.path.abspath(directory)
    except OSError as error:
        print(f"Could not prepare hang diagnostics: {error}", flush=True)
        return None


def collect_hang_diagnostics(directory: str | None) -> None:
    """Copy already-written reports into the outer test's retained stdout log."""
    if directory is None:
        return
    root = Path(directory)
    if not root.exists():
        return
    has_reports = False
    for path in sorted(root.rglob("*.txt")):
        try:
            if path.stat().st_size == 0:
                continue
            has_reports = True
            print(f"\n===== collected hang diagnostics: {path} =====", flush=True)
            # Binary mode: seeking a text stream needs an opaque tell() cookie,
            # and a split multi-byte character is handled by errors="replace".
            with path.open("rb") as report:
                size = path.stat().st_size
                if size <= _MAX_REPORT_BYTES:
                    sys.stdout.write(report.read().decode("utf-8", "replace"))
                else:
                    half = _MAX_REPORT_BYTES // 2
                    head = report.read(half)
                    report.seek(-half, os.SEEK_END)
                    tail = report.read()
                    sys.stdout.write(head.decode("utf-8", "replace"))
                    sys.stdout.write(
                        f"\n===== elided {size - len(head) - len(tail)} bytes "
                        f"of {path.name} =====\n"
                    )
                    sys.stdout.write(tail.decode("utf-8", "replace"))
            print(f"\n===== end hang diagnostics: {path.name} =====", flush=True)
        except OSError as error:
            has_reports = True
            print(f"Could not collect hang diagnostics from {path}: {error}", flush=True)
    if not has_reports:
        try:
            shutil.rmtree(root)
        except OSError as error:
            print(f"Could not remove empty hang diagnostics from {directory}: {error}", flush=True)


def record_hang_diagnostic_test(directory: str | None, nodeid: str) -> None:
    """Refresh test identity for cached workers without depending on worker RPC."""
    if directory is None:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        temporary = Path(directory) / f".current-test-{os.getpid()}"
        temporary.write_text(nodeid, encoding="utf-8")
        temporary.replace(Path(directory) / "current-test")
    except OSError as error:
        print(f"Could not record hang diagnostic test: {error}", flush=True)


@contextmanager
def hang_diagnostics_for_test(item: pytest.Item) -> Iterator[str | None]:
    """Enable failure-only reports for a marked test, including setup and teardown."""
    if item.get_closest_marker("hang_diagnostics") is None:
        yield None
        return
    with ExitStack() as stack:
        patch = stack.enter_context(pytest.MonkeyPatch.context())
        patch.setenv("TLLM_HANG_DIAGNOSTICS", "1")
        root = item.config.getoption("hang_diagnostics_root", default=None)
        directory = os.environ.get("TLLM_HANG_DIAGNOSTICS_DIR")
        if root is not None:
            worker = os.environ.get("PYTEST_XDIST_WORKER")
            directory = prepare_hang_diagnostics(os.path.join(root, worker) if worker else root)
        if directory is not None:
            patch.setenv("TLLM_HANG_DIAGNOSTICS_DIR", directory)
            record_hang_diagnostic_test(directory, item.nodeid)
        else:
            # An inherited value would point workers at a directory this runner
            # never collects, so reports would be written and silently dropped.
            patch.delenv("TLLM_HANG_DIAGNOSTICS_DIR", raising=False)
        reporter = item.config.pluginmanager.get_plugin("periodic_junit")
        traceback_dir = directory if directory is not None else root
        if reporter is not None and traceback_dir is not None:
            stack.enter_context(reporter.hang_traceback_directory(traceback_dir))
        yield directory
