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
"""Per-process coverage bootstrap for CBTS Layer C baseline build, active only when CBTS_COVERAGE_CONFIG is set."""

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


if os.getenv("CBTS_COVERAGE_CONFIG"):
    import atexit

    # Explicit suffix string avoids the data_suffix=True rename that drops near-identical filenames.
    import secrets as _secrets
    import socket as _socket
    import threading

    import coverage

    _suffix = f"{_socket.gethostname()}.pid{os.getpid()}.X{_secrets.token_urlsafe(6)}"
    cov = coverage.Coverage(
        config_file=os.getenv("CBTS_COVERAGE_CONFIG"),
        auto_data=True,
        data_suffix=_suffix,
    )
    cov.start()

    # sys.orig_argv preserves the launching cmdline; sys.argv has not yet gained "pytest" when sitecustomize runs.
    _orig_argv = getattr(sys, "orig_argv", sys.argv)
    _is_pytest_main = any("pytest" in a for a in _orig_argv[:4])
    _is_nested_pytest = _parent_is_pytest() and _is_pytest_main
    _skip_daemons = _is_pytest_main

    _PERIODIC_SAVE_SECONDS = 5
    _stop_event = threading.Event()
    # Serialize switch_context/save/stop across the daemon threads and atexit.
    _cov_lock = threading.Lock()

    def _periodic_save():
        while not _stop_event.wait(_PERIODIC_SAVE_SECONDS):
            try:
                with _cov_lock:
                    cov.save()
            except Exception as e:
                print(
                    f"[cbts] periodic save failed in pid {os.getpid()}: "
                    f"{e!r}; periodic save thread exiting, atexit will "
                    f"still attempt one final save",
                    file=sys.stderr,
                )
                return

    def _final_save():
        # Quiesce the daemon threads, then take the lock for the final stop+save.
        _stop_event.set()
        try:
            with _cov_lock:
                cov.stop()
                cov.save()
        except Exception as e:
            print(
                f"[cbts] final save failed in pid {os.getpid()}: {e!r}",
                file=sys.stderr,
            )

    atexit.register(_final_save)

    if not _skip_daemons:
        threading.Thread(
            target=_periodic_save,
            daemon=True,
            name="cbts-periodic-save",
        ).start()

    # In worker processes, attribute coverage to the current test via the CBTS_TEST_ID env var and a polled marker file.
    _initial_nodeid = os.environ.get("CBTS_TEST_ID", "").strip()
    if _initial_nodeid and not _is_pytest_main:
        with _cov_lock:
            cov.switch_context(_initial_nodeid)

    if _is_nested_pytest:
        # Inner pytest: apply the mpi_session env-whitelist patch synchronously instead of via the watcher thread.
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from cbts_plugin import install_mpi_pool_patch

            install_mpi_pool_patch(raise_on_refactor=False)
        except Exception as _exc:
            print(
                f"[cbts] nested-pytest mpi patch skipped in pid {os.getpid()}: {_exc!r}",
                file=sys.stderr,
            )

    if not _skip_daemons:
        import time as _time

        def _watch_mpi_session():
            try:
                from cbts_plugin import install_mpi_pool_patch
            except ImportError:
                return
            while True:
                mod = sys.modules.get("tensorrt_llm.llmapi.mpi_session")
                # Gate on MpiPoolSession existing to avoid patching a half-initialized module.
                if mod is not None and hasattr(mod, "MpiPoolSession"):
                    try:
                        install_mpi_pool_patch(raise_on_refactor=False)
                    except Exception as exc:
                        print(
                            f"[cbts] mpi_session patch in pid {os.getpid()} failed: {exc!r}",
                            file=sys.stderr,
                        )
                    return
                _time.sleep(0.1)

        threading.Thread(
            target=_watch_mpi_session,
            daemon=True,
            name="cbts-mpi-patcher",
        ).start()

        _MARKER_FILE = os.environ.get("CBTS_MARKER_FILE", "/tmp/cbts/current_test.txt")

        def _poll_marker():
            last_seen = _initial_nodeid
            while not _stop_event.is_set():
                try:
                    with open(_MARKER_FILE) as f:
                        nodeid = f.read().strip()
                    if nodeid and nodeid != last_seen:
                        with _cov_lock:
                            if _stop_event.is_set():
                                break
                            cov.switch_context(nodeid)
                            try:
                                # Save now so short-lived workers persist context before the periodic save.
                                cov.save()
                            except Exception as e:
                                print(
                                    f"[cbts] immediate save failed in pid {os.getpid()}: {e!r}",
                                    file=sys.stderr,
                                )
                        last_seen = nodeid
                except (FileNotFoundError, OSError):
                    pass
                _stop_event.wait(0.1)

        threading.Thread(
            target=_poll_marker,
            daemon=True,
            name="cbts-context-poller",
        ).start()
