# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Per-process function-level coverage bootstrap for CBTS Layer C, active only when CBTS_COVERAGE_CONFIG is set."""

import os
import sys


def _parent_is_pytest():
    """Return True if our parent process is also running pytest."""
    try:
        with open(f"/proc/{os.getppid()}/cmdline", "rb") as f:
            parent_cmdline = f.read().split(b"\x00")
    except OSError:
        return False
    return any(b"pytest" in part for part in parent_cmdline)


def _is_dependency_build_process():
    """Return True for pip / setuptools / native build-tool processes, which opt the subtree out."""
    argv = getattr(sys, "orig_argv", sys.argv) or [""]
    # Scan each token's basename: the tool may be in argv[0] (bare) or argv[1] (shebang / setup.py).
    tools = {"pip", "pip3", "cmake", "ninja", "ninja-build", "meson"}
    for a in argv:
        base = os.path.basename(a or "").lower()
        if base in tools or base == "setup.py" or (a or "").endswith("setup.py"):
            return True
    joined = " ".join(argv)
    return any(n in joined for n in ("-m pip", "-m build", "_in_process", "pyproject_hooks"))


# Drop the gate var so build tooling and everything it spawns opts out of instrumentation.
if os.getenv("CBTS_COVERAGE_CONFIG") and _is_dependency_build_process():
    os.environ.pop("CBTS_COVERAGE_CONFIG", None)


if os.getenv("CBTS_COVERAGE_CONFIG"):
    import atexit
    import configparser
    import threading

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cbts_pystart import PyStartTracker

    _CONFIG = os.getenv("CBTS_COVERAGE_CONFIG")

    # Read [run] source (product roots) and data_file (dir + stage name) from the rendered rcfile.
    def _read_config(path):
        cp = configparser.ConfigParser()
        try:
            cp.read(path)
        except configparser.Error:
            return [], ".", "stage"
        src = [ln.strip() for ln in cp.get("run", "source", fallback="").splitlines() if ln.strip()]
        data_file = cp.get("run", "data_file", fallback="")
        data_dir = os.path.dirname(data_file) or "."
        base = os.path.basename(data_file)
        stage = base.split(".coverage.", 1)[1] if ".coverage." in base else "stage"
        return src, data_dir, stage

    _src, _data_dir, _stage = _read_config(_CONFIG)

    _PERIODIC_SAVE_SECONDS = 5
    _stop_event = threading.Event()

    _tracker = PyStartTracker(_src, _data_dir, _stage)
    _tracker.start()

    def switch_test_context(nodeid):
        """Switch the active test context; each test's entered functions are recorded separately."""
        if _stop_event.is_set():
            return
        _tracker.switch_test_context(nodeid or "")

    def _save_active():
        try:
            _tracker.save()
        except Exception as e:
            print(f"[cbts] periodic save failed in pid {os.getpid()}: {e!r}", file=sys.stderr)

    def _final_save():
        _stop_event.set()
        try:
            _tracker.save()
            _tracker.stop()
        except Exception as e:
            print(f"[cbts] final save failed in pid {os.getpid()}: {e!r}", file=sys.stderr)

    atexit.register(_final_save)

    # sys.orig_argv preserves the launching cmdline; sys.argv has not yet gained "pytest" when sitecustomize runs.
    _orig_argv = getattr(sys, "orig_argv", sys.argv)
    _is_pytest_main = any("pytest" in a for a in _orig_argv[:4])
    _is_nested_pytest = _parent_is_pytest() and _is_pytest_main
    # mpi4py.futures pool workers serve one test via the inherited CBTS_TEST_ID and the atexit save; no daemons.
    _is_mpi_pool_worker = any("mpi4py.futures" in a for a in _orig_argv)
    _skip_daemons = _is_pytest_main or _is_mpi_pool_worker

    # Subprocesses inherit the current nodeid via CBTS_TEST_ID; the outer pytest re-switches per test via the plugin.
    _initial_nodeid = os.environ.get("CBTS_TEST_ID", "").strip()
    if _initial_nodeid:
        switch_test_context(_initial_nodeid)

    if _is_nested_pytest:
        # Inner pytest: apply the mpi_session env-whitelist patch synchronously instead of via the watcher thread.
        try:
            from cbts_plugin import install_mpi_pool_patch

            install_mpi_pool_patch(raise_on_refactor=False)
        except Exception as _exc:
            print(
                f"[cbts] nested-pytest mpi patch skipped in pid {os.getpid()}: {_exc!r}",
                file=sys.stderr,
            )

    if not _skip_daemons:

        def _watch_mpi_session():
            # Wait until the host has imported the target modules so this daemon triggers no racing import.
            while not _stop_event.is_set():
                mod = sys.modules.get("tensorrt_llm.llmapi.mpi_session")
                if (
                    mod is not None
                    and hasattr(mod, "MpiPoolSession")
                    and "mpi4py.futures" in sys.modules
                ):
                    try:
                        from cbts_plugin import install_mpi_pool_patch

                        install_mpi_pool_patch(raise_on_refactor=False)
                    except Exception as exc:
                        print(
                            f"[cbts] mpi_session patch in pid {os.getpid()} failed: {exc!r}",
                            file=sys.stderr,
                        )
                    return
                _stop_event.wait(0.1)

        threading.Thread(
            target=_watch_mpi_session,
            daemon=True,
            name="cbts-mpi-patcher",
        ).start()

        def _periodic_save():
            while not _stop_event.wait(_PERIODIC_SAVE_SECONDS):
                _save_active()

        threading.Thread(
            target=_periodic_save,
            daemon=True,
            name="cbts-periodic-save",
        ).start()

        _MARKER_FILE = os.environ.get("CBTS_MARKER_FILE", "/tmp/cbts/current_test.txt")

        def _poll_marker():
            last_seen = _initial_nodeid
            while not _stop_event.is_set():
                try:
                    with open(_MARKER_FILE) as f:
                        nodeid = f.read().strip()
                    if nodeid and nodeid != last_seen:
                        # Long-lived non-pytest processes (e.g. trtllm-serve) switch context on marker change.
                        switch_test_context(nodeid)
                        last_seen = nodeid
                except (FileNotFoundError, OSError):
                    pass
                _stop_event.wait(0.1)

        threading.Thread(
            target=_poll_marker,
            daemon=True,
            name="cbts-context-poller",
        ).start()
