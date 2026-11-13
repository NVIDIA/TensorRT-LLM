# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Run pytest collection with a deadline and retry only timed-out attempts."""

import argparse
import faulthandler
import math
import os
import runpy
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import psutil


def _pytest_child(args: list[str]) -> None:
    # Keep the diagnostic FD outside pytest's stdout/stderr capture.
    with os.fdopen(os.dup(sys.stderr.fileno()), "w") as trace:
        faulthandler.register(signal.SIGUSR1, file=trace, all_threads=True)
        sys.argv = ["pytest", *args]
        try:
            runpy.run_module("pytest", run_name="__main__")
        finally:
            faulthandler.unregister(signal.SIGUSR1)


def _cleanup(process: subprocess.Popen, children: list[psutil.Process]) -> None:
    """Reap the attempt and kill its children before another attempt starts."""
    try:
        children.extend(psutil.Process(process.pid).children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    # A subprocess can start its own session, so also kill the saved descendants.
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            if child.status() != psutil.STATUS_ZOMBIE:
                raise RuntimeError("Collection children survived cleanup; refusing to retry")
        except psutil.NoSuchProcess:
            pass


def _show_tail(path: Path) -> None:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        print(stream.read().decode(errors="replace"), file=sys.stderr, flush=True)


def collect(args: list[str], log_dir: Path, timeout: float, retries: int) -> int:
    """Save each attempt's output; publish output.txt only after success."""
    log_dir.mkdir(parents=True, exist_ok=True)
    output = log_dir / "output.txt"
    output.unlink(missing_ok=True)
    for attempt in range(1, retries + 2):
        stdout_path = log_dir / f"attempt-{attempt}.stdout.log"
        stderr_path = log_dir / f"attempt-{attempt}.stderr.log"
        print(
            f"Pytest collection attempt {attempt}/{retries + 1}: timeout={timeout}s; "
            f"logs={log_dir}",
            flush=True,
        )
        timed_out = False
        children: list[psutil.Process] = []
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--child", *args],
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # The leader can exit while dumping stacks; retain detached children now.
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
                print(
                    f"Collection attempt {attempt} timed out; requesting Python stacks", flush=True
                )
                try:
                    process.send_signal(signal.SIGUSR1)
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            finally:
                _cleanup(process, children)
        if timed_out:
            _show_tail(stdout_path)
            _show_tail(stderr_path)
            continue
        if code == 0:
            shutil.copyfile(stdout_path, output)
            return 0
        _show_tail(stdout_path)
        _show_tail(stderr_path)
        return code if code > 0 else 128 - code
    return 124


def _interrupt(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main() -> int:
    """Parse runner options separately from the unchanged pytest arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=30 * 60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if not math.isfinite(options.timeout_seconds) or options.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive and finite")
    if options.retries < 0:
        parser.error("--retries must be nonnegative")
    args = options.pytest_args
    if args[:1] == ["--"]:
        args = args[1:]
    if "--collect-only" not in args:
        parser.error("pytest arguments must include --collect-only")
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGINT, _interrupt)
    return collect(args, options.log_dir, options.timeout_seconds, options.retries)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        _pytest_child(sys.argv[2:])
    else:
        sys.exit(main())
